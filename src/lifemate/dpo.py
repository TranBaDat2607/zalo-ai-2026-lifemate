"""Step-preference training at the first wrong step: DPO with an SFT term on the chosen step, on top of a soup.

    python -m lifemate.dpo --out {out}/dpo --adapter A,B --pairs auto:pairs.jsonl --transcripts auto:avg5_train.json

The adapters' mean delta (or weighted sum, --adapter-weights) is merged into the base weights, exactly as inference
loads them: that merged model is the reference. A fresh LoRA on the same modules as train.py is the policy, so the
result is one more adapter that inference stacks on top of the soup with a weight w:
``infer --adapter A,B,{out}/dpo/adapter --adapter-weights 0.5,0.5,w`` (w = 1 is the trained model, w = 0.5 the 50/50
average of before and after).

Each pair (scripts/build_pairs.py) is rebuilt as the agent loop would have fed it: the system prompt with every
installed app (apps unrelated to the gold calls leave only while the sequence is longer than --max-seq-len, as in
training), the history, the request as heard (--transcripts, the run's own), the run's restated request in the first
think block, then gold calls 0..t-1 with their gold results. Chosen and rejected are the next assistant turn (a call,
or prompts.DONE for "DONE"). Log-probs count only the tokens after the part the two sequences share, so the restated
request is never a target.

    loss = -log sigmoid(beta * ((lp_c - ref_c) - (lp_r - ref_r))) + sft_weight * NLL(chosen tokens) / n_chosen_tokens

Reference log-probs come from the same model with the LoRA switched off, computed once before training.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import data, models, prompts, schedule

DONE = "DONE"
# The default kernel on a T4 holds a full n x n attention matrix per head; the memory-efficient one does not.
KERNELS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def continuation_text(step_value) -> str:
    """The assistant turn of a pair side: a call, or the closing reply."""
    if step_value == DONE:
        return prompts.DONE + "<|im_end|>"
    return prompts.call_text(step_value["name"], step_value.get("arguments") or {}) + "<|im_end|>"


def encode_pair(item: dict, texts: dict, catalog: dict, tok, restated: str, step: int, chosen, rejected,
                dropped: frozenset = frozenset()) -> tuple:
    """(chosen ids, rejected ids, start): token ids of both sequences and the first position where they differ."""
    parts = [prompts.turn("system", prompts.system_text(item, catalog, dropped))]
    parts += [text for text, _ in prompts.history_segments(item, texts)]
    parts.append(prompts.turn("user", texts[item["audio"]]) + prompts.THINK_OPEN)
    # build_pairs.py stores the restated request with the newline that THINK_CLOSE adds back; without this strip the
    # think block read 'text\n\n</think>', one newline more than the model wrote (found 10 Nov, probe tests).
    lead = prompts.think_lead(restated[:-1] if restated.endswith("\n") else restated)
    calls, results = item["label"]["tool_calls"], item["label"]["tool_results"]
    for call, result in zip(calls[:step], results[:step]):
        parts.append(lead + prompts.call_text(call["name"], call.get("arguments") or {}) + "<|im_end|>")
        lead = ""
        parts.append("\n" + prompts.turn("user", prompts.result_text(result)) + prompts.GEN_PREFIX)
    prefix = [t for p in parts for t in tok(p)]
    ids_c = prefix + tok(lead + continuation_text(chosen))
    ids_r = prefix + tok(lead + continuation_text(rejected))
    start = 0
    while start < min(len(ids_c), len(ids_r)) and ids_c[start] == ids_r[start]:
        start += 1
    if start >= min(len(ids_c), len(ids_r)):
        raise ValueError(f"{item['id']}: chosen and rejected do not differ")
    return ids_c, ids_r, start


def encode_within(item: dict, texts: dict, catalog: dict, tok, pair: dict, max_len: int):
    """``encode_pair`` with unrelated apps dropped one at a time while too long; None if it still does not fit."""
    removable = prompts.apps_to_drop(item) if item["label"]["tool_calls"] else []
    dropped = frozenset()
    while True:
        enc = encode_pair(item, texts, catalog, tok, pair["restated"], pair["step"], pair["chosen"], pair["rejected"],
                          dropped)
        if not max_len or max(len(enc[0]), len(enc[1])) <= max_len:
            return enc
        if not removable:
            return None
        dropped = dropped | {removable.pop(0)}


def seq_logprob(model, ids: torch.Tensor, start: int) -> tuple:
    """Sum of log-probs of ids[start:] and their count. The head is frozen and the vocabulary has 248k entries, so the
    logits are made for the scored positions only (a pair side scores tens of tokens, not the whole prompt)."""
    base = model.get_base_model()
    with sdpa_kernel(KERNELS):
        hidden = base.model(input_ids=ids, use_cache=False).last_hidden_state[0, start - 1:-1]
    logits = base.lm_head(hidden).float()
    targets = ids[0, start:]
    return torch.log_softmax(logits, dim=-1).gather(1, targets[:, None]).sum(), len(targets)


def pair_loss(lp_c, lp_r, ref_c: float, ref_r: float, n_c: int, beta: float, sft_weight: float) -> tuple:
    """(loss, dpo part, sft part, margin) for one pair."""
    margin = beta * ((lp_c - ref_c) - (lp_r - ref_r))
    dpo = -torch.nn.functional.logsigmoid(margin)
    sft = -lp_c / n_c
    return dpo + sft_weight * sft, dpo, sft, margin


def pair_backward(model, ids_c, ids_r, start: int, ref_c: float, ref_r: float, beta: float, sft_weight: float,
                  scale=lambda x: x) -> tuple:
    """Backpropagates ``pair_loss`` of one pair; returns (loss, dpo, sft, margin, lp_c, lp_r) as floats.

    With gradient checkpointing the layers run again during backward, so backward must run under the same attention
    kernels as the forward, as ``train.backward_loss`` does: outside them the recompute takes the default kernel, saves
    other tensors, and torch.utils.checkpoint raises CheckpointError (found by tests/test_dpo.py on the tiny model).
    """
    with sdpa_kernel(KERNELS):
        lp_c, n_c = seq_logprob(model, ids_c, start)
        lp_r, _ = seq_logprob(model, ids_r, start)
        loss, dpo, sft, margin = pair_loss(lp_c, lp_r, ref_c, ref_r, n_c, beta, sft_weight)
        scale(loss).backward()
    return loss.item(), dpo.item(), sft.item(), margin.item(), lp_c.item(), lp_r.item()


def load_pairs(spec: str) -> list:
    if spec.startswith("auto:"):
        name = spec[5:]
        found = [folder / name for folder in data.walk_input(name)]  # walk_input yields the folders holding the file
        if not found:
            raise SystemExit(f"no {name} under /kaggle/input")
        print("pairs found:", [str(p) for p in found], flush=True)
        spec = str(found[0])
    return [json.loads(line) for line in Path(spec).read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm", default=models.DEFAULT_LLM)
    ap.add_argument("--adapter", required=True, help="soup adapters, comma-separated (auto:NAME allowed), merged first")
    ap.add_argument("--adapter-weights", default=None, help="weighted sum instead of the mean, as in infer")
    ap.add_argument("--pairs", required=True, help="pairs.jsonl from scripts/build_pairs.py, or auto:FILENAME")
    ap.add_argument("--transcripts", required=True, help="the transcripts the runs heard, e.g. auto:avg5_train.json")
    ap.add_argument("--exclude", default=None, help="id spec (data.id_set) whose pairs are skipped, e.g. splits")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sft-weight", type=float, default=1.0)
    # 58 steps at peak 2e-5 sum to ~6e-4 of learning rate: the SFT runs changed nothing in their first 6e-4 (log.md).
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=3700)
    ap.add_argument("--max-hours", type=float, default=0.0, help="stop and save after this long (0 = no limit)")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)

    from .infer import load_adapted, load_transcripts, resolve_adapter
    from .train import _TARGETS
    models.seed_everything(args.seed)
    started = time.monotonic()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = data.find_root("train")
    items = {i["id"]: i for i in data.load_split("train", root)}
    catalog = data.load_catalog("train", root)
    texts = load_transcripts(args.transcripts)
    pairs = load_pairs(args.pairs)
    exclude = data.id_set(args.exclude) if args.exclude else set()
    n_excluded = sum(p["id"] in exclude for p in pairs)
    pairs = [p for p in pairs if p["id"] not in exclude]
    paths = ",".join(resolve_adapter(a) for a in args.adapter.split(",") if a)
    tokenizer, model = load_adapted(args.llm, paths, args.adapter_weights)
    tok = lambda t: tokenizer(t, add_special_tokens=False)["input_ids"]  # noqa: E731

    encoded, skipped = [], 0
    for pair in pairs:
        enc = encode_within(items[pair["id"]], texts, catalog, tok, pair, args.max_seq_len)
        if enc is None:
            skipped += 1
            continue
        ids_c, ids_r, start = enc
        encoded.append((pair, torch.tensor([ids_c]), torch.tensor([ids_r]), start))
    print(f"pairs: {len(encoded)} encoded, {skipped} longer than {args.max_seq_len} tokens, "
          f"{n_excluded} excluded by --exclude", flush=True)
    if not encoded:
        raise SystemExit("no pairs to train on")

    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0.0,
                                             target_modules=_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    device, dtype = models.device_and_dtype()

    # Reference log-probs: the merged soup alone. The LoRA's B starts at zero, so this equals the policy at step 0.
    refs = []
    with model.disable_adapter(), torch.no_grad():
        for _, ids_c, ids_r, start in encoded:
            ref_c, _ = seq_logprob(model, ids_c.to(device), start)
            ref_r, _ = seq_logprob(model, ids_r.to(device), start)
            refs.append((ref_c.item(), ref_r.item()))
    print(f"reference log-probs done in {(time.monotonic() - started) / 60:.1f} min; mean chosen "
          f"{sum(r[0] for r in refs) / len(refs):.2f}, rejected {sum(r[1] for r in refs) / len(refs):.2f}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    # A first run (10 Nov) lost its 5 warm-up steps to overflow skips while the scale fell from 2^16; it settled by 2^11.
    scaler = torch.amp.GradScaler(device, init_scale=2.0 ** 11, enabled=dtype == torch.float16)
    total_steps = max(1, math.ceil(len(encoded) * args.epochs / args.grad_accum))
    warmup = max(1, total_steps // 10)
    rng = random.Random(args.seed)
    order, position, step, seen, skipped = list(range(len(encoded))), len(encoded), 0, 0, 0
    window = {"loss": 0.0, "dpo": 0.0, "sft": 0.0, "margin": 0.0, "acc": 0.0, "dc": 0.0, "dr": 0.0, "n": 0}
    history = []
    model.train()
    # The time-based decay counts training time only: the reference pass (~20 min on a T4) is not part of the schedule.
    train_started = time.monotonic()
    budget = args.max_hours * 3600 - (train_started - started) if args.max_hours else 0.0
    while step < total_steps:
        if args.max_hours and time.monotonic() - started > args.max_hours * 3600:
            print("time limit reached", flush=True)
            break
        # As in train.py: if the time cap comes before the last step, the learning rate still reaches zero at the cap.
        time_frac = (time.monotonic() - train_started) / budget if budget > 0 else 0.0
        for group in optim.param_groups:
            group["lr"] = schedule.cosine(step, warmup, total_steps, args.lr, time_frac)
        for _ in range(args.grad_accum):
            if position >= len(order):
                rng.shuffle(order)
                position = 0
            k = order[position]
            position += 1
            pair, ids_c, ids_r, start = encoded[k]
            ref_c, ref_r = refs[k]
            loss, dpo, sft, margin, lp_c, lp_r = pair_backward(
                model, ids_c.to(device), ids_r.to(device), start, ref_c, ref_r, args.beta, args.sft_weight,
                lambda x: scaler.scale(x / args.grad_accum))
            seen += 1
            for key, value in (("loss", loss), ("dpo", dpo), ("sft", sft), ("margin", margin)):
                window[key] += value
            window["acc"] += float(margin > 0)
            window["dc"] += lp_c - ref_c
            window["dr"] += lp_r - ref_r
            window["n"] += 1
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scale_before = scaler.get_scale() if scaler.is_enabled() else 1.0
        scaler.step(optim)
        scaler.update()
        # fp16: the scaler skips a step whose gradients overflowed and halves its scale (the first few steps from 2^16)
        skipped += int(scaler.is_enabled() and scaler.get_scale() < scale_before)
        optim.zero_grad(set_to_none=True)
        step += 1
        if step % 5 == 0 or step == total_steps:
            n = window["n"]
            row = {"step": step, "pairs_seen": seen, "lr": optim.param_groups[0]["lr"], "skipped_steps": skipped,
                   "scale": scaler.get_scale() if scaler.is_enabled() else 1.0,
                   **{key: round(window[key] / n, 4) for key in ("loss", "dpo", "sft", "margin", "acc", "dc", "dr")},
                   "minutes": round((time.monotonic() - started) / 60, 1)}
            history.append(row)
            print(json.dumps(row), flush=True)  # dc / dr: chosen / rejected log-prob change since the start
            window = {key: 0.0 for key in window} | {"n": 0}

    model.save_pretrained(out / "adapter")
    metrics = {"pairs": len(encoded), "skipped_long": skipped, "steps": step, "total_steps": total_steps,
               "pairs_seen": seen, "skipped_steps": skipped, "beta": args.beta, "sft_weight": args.sft_weight, "lr": args.lr,
               "lora_r": args.lora_r, "adapter": paths, "adapter_weights": args.adapter_weights,
               "minutes": round((time.monotonic() - started) / 60, 1), "history": history[-20:]}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(f"saved {out / 'adapter'}", flush=True)
    return metrics


if __name__ == "__main__":
    main()
