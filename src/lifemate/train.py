"""Fine-tune the LLM with LoRA on gold tool-call conversations.

    python -m lifemate.train --out outputs/train --holdout-file ids.json --epochs 2

Each sample is rendered with noisy ASR transcripts (when given) and with gold transcripts, so the model learns the
request behind the errors. Loss only falls on the assistant turns of the latest request.
"""
import argparse
import ctypes
import gc
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import contrast, data, models, prompts, schedule, stream, view
from .noise import ConfusionNoise
from .prompts import encode_conversation

_LIBC = ctypes.CDLL("libc.so.6")
_BUNDLED_SYNTH = {"package": "synth_samples.jsonl", "held": "synth_held.jsonl"}  # in this package: public-only apps, held-out apps (E2)


def rss_gb() -> float:
    """Current resident memory of this process."""
    with open("/proc/self/statm", encoding="ascii") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30


# Qwen3.5 mixes full attention (q/k/v/o_proj, every 4th layer) and linear attention (in_proj_*, out_proj); adapt both and the MLP.
_TARGETS = r"^(?!.*(visual|vision)).*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)$"


FLA_VERSION = "0.5.2"  # flash-linear-attention and fla-core, July 2026


def enable_fla(folder: str) -> None:
    """Make flash-linear-attention importable for this process only, before transformers builds Qwen3.5.

    transformers picks FLA's chunked gated-delta-rule kernel at import time when the ``fla`` package is importable, and
    its reference PyTorch loop otherwise. The package goes into a private folder on sys.path (not site-packages), so the
    validation and test processes keep the reference path that the Docker image runs. A failed install only costs speed.
    """
    import importlib
    import subprocess
    target = Path(folder)
    if not (target / "fla").is_dir():
        res = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(target),
                              f"fla-core=={FLA_VERSION}", f"flash-linear-attention=={FLA_VERSION}"])
        if res.returncode:
            print(f"flash-linear-attention install failed (exit {res.returncode}); training on the reference path", flush=True)
            return
    sys.path.insert(0, str(target))
    try:
        fla = importlib.import_module("fla")
        importlib.import_module("fla.ops.gated_delta_rule")
    except Exception as e:  # noqa: BLE001
        sys.path.remove(str(target))
        print(f"flash-linear-attention import failed ({type(e).__name__}: {e}); training on the reference path", flush=True)
        return
    print(f"flash-linear-attention {getattr(fla, '__version__', '?')} from {fla.__file__}", flush=True)


def linear_attention_backend() -> str:
    """Which chunked gated-delta-rule function transformers bound for Qwen3.5 (it decides once, at import)."""
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as mq
    except ImportError:
        return "unknown (no qwen3_5 module)"
    for cell in getattr(mq.torch_chunk_gated_delta_rule, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if callable(value) and str(getattr(value, "__module__", "")).startswith("fla"):
            return f"flash-linear-attention ({value.__module__})"
    return "reference PyTorch"


def backward_loss(model, ids, labels, scaler, scale: float, chunk: int = 96, contrasts: list = None,
                  pair_weight: float = 0.0, stats: dict = None) -> float:
    """Backpropagates the assistant-turn loss and returns its value.

    The head is frozen and the vocabulary has 248k entries, so logits of a long turn alone would take gigabytes. The loss
    is therefore computed chunk by chunk on detached hidden states, and the summed gradient then goes through the network.

    ``contrasts`` ([(position, gold token, wrong token, weight)], contrast.locate) add ``pair_weight`` times the weighted
    mean of -log sigmoid(z_gold - z_wrong) at those positions, from the same logits; ``stats`` collects their margins.
    """
    base = model.get_base_model()
    targets = labels[0, 1:]
    positions = (targets != -100).nonzero()[:, 0]
    rows = contrast.rows_for(positions.tolist(), contrasts) if contrasts and pair_weight else []
    weight_sum = sum(w for _, _, _, w in rows) or 1.0
    # The default kernel on a T4 holds a full n x n attention matrix per head; the memory-efficient one does not.
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        hidden = base.model(input_ids=ids, use_cache=False).last_hidden_state[0, positions]
        leaf = hidden.detach().requires_grad_(True)
        total = 0.0
        for start in range(0, len(positions), chunk):
            logits = base.lm_head(leaf[start:start + chunk])
            loss = torch.nn.functional.cross_entropy(logits.float(), targets[positions[start:start + chunk]], reduction="sum") / len(positions)
            here = [(k - start, c, r, w) for k, c, r, w in rows if start <= k < start + chunk]
            if here:
                z = logits.float()[[k for k, _, _, _ in here]]
                at = torch.arange(len(here), device=z.device)
                margin = z[at, [c for _, c, _, _ in here]] - z[at, [r for _, _, r, _ in here]]
                w = torch.tensor([w for _, _, _, w in here], device=z.device)
                loss = loss + pair_weight * (w * -torch.nn.functional.logsigmoid(margin)).sum() / weight_sum
                if stats is not None:
                    stats["n"] = stats.get("n", 0) + len(here)
                    stats["margin"] = stats.get("margin", 0.0) + float(margin.detach().sum())
                    stats["right"] = stats.get("right", 0) + int((margin > 0).sum())
            scaler.scale(loss * scale).backward()
            total += loss.item()
        hidden.backward(gradient=leaf.grad)
    return total


NOTES_PACKAGES = {"package": "notes.jsonl", "package-v2": "notes_v2.jsonl"}


def notes_path(arg: str) -> Path:
    """`--notes package` is the bundled v1 notes, `package-v2` the v2 notes (rule-style, branch-blind); else a file path."""
    return Path(__file__).parent / NOTES_PACKAGES[arg] if arg in NOTES_PACKAGES else Path(arg)


def load_notes(arg: str, holdout: set, drop_held: bool = False) -> dict:
    """{sample id: note} of a notes file (see ``notes_path``); refuses a file with notes for held-out samples, unless
    ``drop_held``: fold models outside the validation set hold out samples that have notes, and must not see them."""
    path = notes_path(arg)
    notes = {r["id"]: r["note"] for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r.get("note")}
    leaked = sorted(set(notes) & holdout)
    if leaked and drop_held:
        for sid in leaked:
            del notes[sid]
        print(f"dropped the notes of {len(leaked)} held-out samples", flush=True)
    elif leaked:
        raise SystemExit(f"notes exist for {len(leaked)} held-out samples, e.g. {leaked[:3]}")
    print(f"worked notes: {len(notes)} from {path}", flush=True)
    return notes


def load_init_adapter(model, spec: str, rank_r: int) -> None:
    """Loads a saved LoRA adapter into the freshly wrapped model (plan 3: continue a trained adapter)."""
    from .infer import resolve_adapter
    folder = Path(resolve_adapter(spec))
    config = json.loads((folder / "adapter_config.json").read_text())
    if config.get("r") != rank_r:
        raise SystemExit(f"--init-adapter has rank {config.get('r')}, the run uses --lora-r {rank_r}")
    weights = load_file(str(folder / "adapter_model.safetensors"))
    result = set_peft_model_state_dict(model, weights)
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    missing = [k for k in (getattr(result, "missing_keys", []) or []) if "lora_" in k]
    if unexpected or missing:
        raise SystemExit(f"--init-adapter does not fit: {len(unexpected)} unexpected keys (e.g. {unexpected[:2]}), "
                         f"{len(missing)} LoRA weights missing (e.g. {missing[:2]})")
    print(f"init adapter: {len(weights)} tensors from {folder}", flush=True)


def resolve_file(spec: str) -> Path:
    if not spec.startswith("auto:"):
        return Path(spec)
    name = spec[len("auto:"):]
    found = [folder / name for folder in data.walk_input(name)]
    if not found:
        raise SystemExit(f"no {name} under /kaggle/input")
    return found[0]


def load_extras(spec: str, holdout: set) -> tuple:
    """(rows, file paths) of the new-data files. Rows derived from a held-out sample (field "source") are refused."""
    paths = [resolve_file(s) for s in spec.split(",") if s.strip()]
    rows = [json.loads(line) for p in paths for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    kinds = {}
    for r in rows:
        kinds[r.get("kind")] = kinds.get(r.get("kind"), 0) + 1
    bad = set(kinds) - {"conv", "drill", "tools"}
    if bad:
        raise SystemExit(f"unknown extra kinds {sorted(bad)}")
    leaked = sorted({r["source"] for r in rows if r.get("source") in holdout})
    if leaked:
        raise SystemExit(f"{len(leaked)} held-out samples are sources of extra rows, e.g. {leaked[:3]}")
    print(f"extra rows: {len(rows)} {kinds} from {[str(p) for p in paths]}", flush=True)
    return rows, paths


def train_stream(args, model, params, optim, scaler, encoded, n_main, catalog, tok, history_ok, holdout, out, device,
                 dtype, rank, world, n_samples, bank=None, noise=None) -> dict:
    """Plan 3: one pass over --extra with replay, on ``world`` processes, resumable (lifemate.stream)."""
    extras, paths = load_extras(args.extra, holdout)
    # New conversations may use apps train lacks: their tools come from the public catalog, train's specs win otherwise.
    extra_catalog = {**data.load_catalog("public_test", data.find_root("public_test")), **catalog}
    n_side = len(encoded) - n_main
    order = stream.build_order(len(extras), n_main, n_side, args.replay_share, args.replay_side, args.seed)
    fingerprint = stream.order_hash(order, stream.files_hash(paths))
    G = args.grad_accum
    if G % world:
        raise SystemExit(f"--grad-accum {G} must be a multiple of the {world} processes")
    total_steps = math.ceil(len(order) / G)
    warmup = max(10, round(total_steps * args.warmup_frac))
    counts = {k: sum(1 for kind, _ in order if kind == k) for k in "xms"}
    print(json.dumps({"stream": {"items": len(order), "extra": counts["x"], "replay_main": counts["m"],
                                 "replay_side": counts["s"], "steps": total_steps, "warmup": warmup,
                                 "processes": world, "items_per_step": G, "fingerprint": fingerprint}}), flush=True)

    def encode_extra(k: int):
        row, r = extras[k], random.Random(f"{args.seed}:x:{k}")  # its own stream: the same on any resume
        if row["kind"] == "conv":
            # 14 Nov review: extra rows get what the replayed synthetic rows get (they had neither): made-up names at
            # --mask-names, and the run's hearing simulation (the readings view in a view run, else the confusion noise)
            # instead of the file's own texts (about a third of real hearing errors).
            item, texts, rename = row["item"], row["texts"], None
            if args.mask_names and r.random() < args.mask_names:
                rename = prompts.renamer(prompts.masked_names(item, extra_catalog, r))
            if bank is not None or noise is not None:
                gold = prompts.gold_transcripts(item)
                if bank is not None:
                    texts = {a: bank.view(t, r) if a == item["audio"] else bank.readings(t, r)[0] for a, t in gold.items()}
                else:
                    texts = {a: noise.apply(t, r) for a, t in gold.items()}
            ids, labels = encode_conversation(item, texts, extra_catalog, tok, args.denoise, args.max_seq_len,
                                              args.tool_drop, r, history_ok, rename, None)
        elif row["kind"] == "tools":
            ids, labels = prompts.encode_tool_pack(row["pack"], extra_catalog, tok)
        else:
            ids, labels = prompts.encode_drill_pack(row["pack"], tok)
        if len(ids) > args.max_seq_len:
            return None
        return torch.tensor([ids]), torch.tensor([labels])

    state_file = out / "state.pt"
    source = out if state_file.exists() else stream.find_state(args.resume) if args.resume else None
    step, position, tokens, skipped, restarts, sessions = 0, 0, 0, 0, 0, []
    if source is not None:
        state = torch.load(source / "state.pt", map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint or state["total_steps"] != total_steps:
            raise SystemExit(f"{source}/state.pt belongs to another pass (data, order or settings differ)")
        set_peft_model_state_dict(model, state["lora"])
        optim.load_state_dict(state["optim"])
        scaler.load_state_dict(state["scaler"])
        step, position, tokens, skipped = state["step"], state["position"], state["tokens"], state["skipped"]
        restarts, sessions = state["restarts"], state["sessions"]
        print(f"resumed from {source}: step {step} of {total_steps}, item {position} of {len(order)}", flush=True)
        del state
    else:
        stream.broadcast_params(params, world)
    if source is not out:
        sessions = sessions + [{"started_step": step}]

    lead = rank == 0
    started = time.monotonic()
    deadline = started + args.max_hours * 3600
    stop = {"term": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("term", True))  # jobs.py's deadline: save, then exit

    saved = {"step": -1}

    def save(final: bool = False) -> None:
        if saved["step"] == step and not final:
            return
        saved["step"] = step
        if lead:
            sessions[-1].update({"ended_step": step, "minutes": round((time.monotonic() - started) / 60, 1)})
            stream.save_state(state_file, {"lora": {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()},
                                           "optim": optim.state_dict(), "scaler": scaler.state_dict(), "step": step,
                                           "position": position, "tokens": tokens, "skipped": skipped,
                                           "restarts": restarts, "sessions": sessions, "fingerprint": fingerprint,
                                           "total_steps": total_steps, "finished": final})
            model.save_pretrained(out / "adapter")
        stream.barrier(world)

    model.train()
    running, running_n = 0.0, 0
    while position < len(order):
        size = min(G, len(order) - position)
        for p in stream.my_positions(position, size, rank, world):
            kind, i = order[p]
            enc = encode_extra(i) if kind == "x" else encoded[i] if kind == "m" else encoded[n_main + i]
            if enc is None:
                skipped += 1
                continue
            ids, labels = enc
            try:
                with torch.autocast(device, dtype=dtype, enabled=dtype == torch.float16):
                    running += backward_loss(model, ids.to(device), labels.to(device), scaler, 1 / size)
                running_n += 1
                tokens += ids.shape[1]
            except torch.cuda.OutOfMemoryError:  # this process loses its part of the step; the others keep theirs
                print(json.dumps({"skipped_tokens": ids.shape[1], "step": step}), flush=True)
                for q in params:
                    q.grad = None
                torch.cuda.empty_cache()
                skipped += 1
        position += size
        stream.allreduce_grads(params, world)
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        lr = schedule.cosine(step, warmup, total_steps, args.lr)  # by steps only: every process and session agree
        for group in optim.param_groups:
            group["lr"] = lr
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        step += 1
        want_stop = time.monotonic() > deadline or stop["term"] or bool(args.stop_step and step >= args.stop_step)
        want_restart = rss_gb() > args.max_rss_gb and restarts < args.max_restarts
        loss_sum, loss_n, n_stop, n_restart = stream.all_sum([running, running_n, want_stop, want_restart], world)
        if step % 20 == 0 or step == 1:
            print(json.dumps({"step": step, "of": total_steps, "item": position, "loss": round(loss_sum / max(loss_n, 1), 4),
                              "lr": round(lr, 7), "elapsed_min": round((time.monotonic() - started) / 60, 1),
                              "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
                              "rss_gb": round(rss_gb(), 1), "skipped": skipped}), flush=True)
        if step % 20 == 0 or step == 1:
            running, running_n = 0.0, 0
        if step % 20 == 0:
            gc.collect()
            _LIBC.malloc_trim(0)
        if position >= len(order):
            break
        if n_stop or n_restart or step % args.save_every == 0:
            if n_restart and not n_stop:
                restarts += 1
            save()
            if n_stop:
                print(f"stopping at step {step} of {total_steps} (deadline or SIGTERM); state saved", flush=True)
                break
            if n_restart:
                print(f"restarting at step {step}: RSS {rss_gb():.1f} GB", flush=True)
                sys.stdout.flush()
                if world > 1:
                    sys.exit(stream.RESTART_EXIT)
                os.execv(sys.executable, [sys.executable, "-m", "lifemate.train", *sys.argv[1:]])
    finished = position >= len(order)
    if finished or saved["step"] != step:
        saved["step"] = -1
        save(final=finished)
    metrics = {"finished": finished, "steps": step, "total_steps": total_steps, "items": position, "of_items": len(order),
               "tokens": tokens, "skipped": skipped, "processes": world, "samples": n_samples, "conversations": n_main,
               "side_examples": n_side, "extra": len(extras), "sessions": sessions,
               "minutes": (time.monotonic() - started) / 60,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    if rank == 0:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        print(json.dumps(metrics), flush=True)
    return metrics


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/train")
    ap.add_argument("--llm", default=models.DEFAULT_LLM)
    ap.add_argument("--holdout-file", default=None, help="JSON list of sample ids kept out of training")
    ap.add_argument("--drop-held-notes", action="store_true", help="drop the notes of held-out samples instead of "
                    "refusing the notes file (fold models outside the validation set, e.g. --holdout-file fold:1)")
    ap.add_argument("--transcripts", default=None, help="JSON file or folder {audio: text} with out-of-fold ASR text; "
                    "view:MAIN,OTHER,... for the readings view (lifemate.view)")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-seq-len", type=int, default=3700)
    ap.add_argument("--longest-first", action="store_true")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--denoise", action="store_true", help="the first turn restates the request before calling tools")
    ap.add_argument("--tool-drop", type=float, default=0.0, help="chance that an app the gold calls do not use leaves the tool list")
    ap.add_argument("--noisy-only", action="store_true", help="with --transcripts: skip the variant with gold transcripts")
    ap.add_argument("--label-history", action="store_true", help="also learn from history tool calls that match final-turn habits")
    ap.add_argument("--synth", default=None, help="synthetic conversations, comma-separated: JSONL files, 'package' (public-only "
                                                  "apps) or 'held' (the six held-out apps, experiment E2), both bundled")
    ap.add_argument("--mask-names", type=float, default=0.0, help="E1: share of conversations whose app and tool names become made-up words")
    ap.add_argument("--notes", default=None, help="worked notes after the restated request (needs --denoise): a JSONL file of "
                                                  "{id, note}, 'package' for the bundled notes.jsonl, or 'package-v2' for notes_v2.jsonl")
    ap.add_argument("--notes-aux", default=None, help="distilling step-by-step (needs --denoise): each noted sample adds one "
                                                      "plan-only side example (prompts.PLAN_TAG; loss on the note only), "
                                                      "used once per run; the main conversations stay as without it and "
                                                      "test prompts never carry the tag. Same values as --notes")
    ap.add_argument("--drills", default=None, help="time/number drills (improve_plan.md step 4): a JSONL of packs from "
                                                   "scripts/drills/make_drills.py, a path or a file name in the package; "
                                                   "each pack is one side example used once per run (prompts.encode_drill_pack)")
    ap.add_argument("--pairs", default=None, help="preference pairs (probe_pairs.py; a path or auto:NAME): each adds a "
                                                  "contrast at the first token where the gold step and the wrong step part, "
                                                  "inside its own training conversation (contrast.py)")
    ap.add_argument("--pair-weight", type=float, default=0.5, help="weight of the pair contrast against the SFT loss")
    ap.add_argument("--save-every", type=int, default=100, help="optimizer steps between adapter saves")
    ap.add_argument("--snapshot-epochs", default="", help="e.g. 1.0,1.5: also keep the adapter as adapter_e<mark> once past each mark")
    ap.add_argument("--no-time-decay", action="store_true", help="decay by steps only, even when --max-hours ends the run first")
    ap.add_argument("--max-rss-gb", type=float, default=9.0, help="replace the process with a fresh one above this much RAM")
    ap.add_argument("--max-restarts", type=int, default=8)
    ap.add_argument("--init-adapter", default="", help="plan 3: start from this LoRA adapter (a folder or auto:NAME) "
                    "instead of a fresh one; same rank and target modules")
    ap.add_argument("--extra", default="", help="plan 3 stream mode: new-data JSONL files, comma-separated (paths or "
                    "auto:FILENAME). Rows {kind: conv, item, texts}, {kind: drill, pack} or {kind: tools, pack}, each trained once, "
                    "shuffled with replayed training conversations (lifemate.stream); --epochs is then ignored")
    ap.add_argument("--replay-share", type=float, default=0.2, help="stream mode: share of the items that replay training data")
    ap.add_argument("--replay-side", type=float, default=0.15, help="stream mode: share of the replay taken from side examples "
                    "(--notes-aux plans, --drills)")
    ap.add_argument("--warmup-frac", type=float, default=0.03, help="stream mode: warmup as a share of all steps")
    ap.add_argument("--resume", default="", help="stream mode: continue the pass saved in this folder's state.pt (a path, or "
                    "auto:NAME for an attached kernel's output); out/state.pt, when present, wins")
    ap.add_argument("--stop-step", type=int, default=0, help="stream mode: end the session after this optimizer step, "
                    "state saved (checks: same step count on 1 and 2 GPUs, stop and resume)")
    ap.add_argument("--fla", default="", help="speed only: install flash-linear-attention into this folder and train with its "
                                              "linear-attention kernels (same maths; inference processes never load it)")
    args = ap.parse_args(argv)

    rank, world = stream.init_distributed()
    if world > 1 and not args.extra:
        raise SystemExit("several processes only train together in stream mode (--extra)")
    if rank:
        sys.stdout = open(os.devnull, "w")  # process 0 logs and saves for all
    if args.fla:
        enable_fla(args.fla)
    models.seed_everything(args.seed)
    root = data.find_root("train")
    items = data.load_split("train", root)
    catalog = data.load_catalog("train", root)
    holdout = data.id_set(args.holdout_file) if args.holdout_file else set()
    items = [i for i in items if i["id"] not in holdout]
    noisy = readings = None
    if args.transcripts:
        from .infer import load_readings, load_transcripts
        noisy = load_transcripts(args.transcripts, {i["audio"] for i in items})  # a view: on the latest requests only
        if args.transcripts.startswith("view:"):  # the readings view: synthetic texts need the plain readings too
            readings = load_readings(args.transcripts)
            braced = sum("{" in t for t in noisy.values())
            print(f"readings view from {len(readings)} readings: braces in {braced} of {len(noisy)} clips", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer, model = models.load_llm(args.llm)
    print("parameters", models.count_parameters(model), flush=True)
    if args.fla:
        print("linear attention kernels:", linear_attention_backend(), flush=True)
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=args.lora_dropout,
                                             target_modules=_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    if args.init_adapter:
        load_init_adapter(model, args.init_adapter, args.lora_r)
    torch.manual_seed(args.seed + 1009 * rank)  # each process its own dropout
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False

    device, dtype = models.device_and_dtype()
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler(device, enabled=dtype == torch.float16)

    encoded = []
    drop_rng = random.Random(args.seed)
    history_ok = prompts.final_call_policy(items) if args.label_history else None
    tok = lambda t: tokenizer(t, add_special_tokens=False)["input_ids"]  # noqa: E731

    mask_rng, masked = random.Random(args.seed + 7), []  # its own stream: without --mask-names nothing else changes
    notes, noted = {}, []
    if args.notes and args.notes_aux:
        raise SystemExit("--notes (note before the calls) and --notes-aux (note as a side task) exclude each other")
    if args.notes:
        if not args.denoise:
            raise SystemExit("--notes needs --denoise: the note follows the restated request")
        notes = load_notes(args.notes, holdout, args.drop_held_notes)
    aux_notes = {}
    if args.notes_aux:
        if not args.denoise:
            raise SystemExit("--notes-aux needs --denoise: the note follows the restated request")
        aux_notes = load_notes(args.notes_aux, holdout, args.drop_held_notes)

    pairs_of = contrast.load_pairs(args.pairs, holdout) if args.pairs else {}
    contrasts_of, pair_report = {}, {"attached": 0, "skipped": 0, "conversations": 0}

    def add(item: dict, texts: dict) -> None:
        rename = None
        if args.mask_names and mask_rng.random() < args.mask_names:
            rename = prompts.renamer(prompts.masked_names(item, catalog, mask_rng))
        note = notes.get(item.get("id"))
        ids, labels = encode_conversation(item, texts, catalog, tok, args.denoise, args.max_seq_len, args.tool_drop, drop_rng, history_ok,
                                          rename, note)
        if len(ids) <= args.max_seq_len:
            if item.get("id") in pairs_of:  # after encoding: pairs draw no random numbers, so the data stays as without them
                found, skipped = contrast.locate(ids, labels, item, pairs_of[item["id"]],
                                                 tok if rename is None else (lambda t: tok(rename(t))), args.denoise, note)
                pair_report["attached"] += len(found)
                pair_report["skipped"] += skipped
                if found:
                    contrasts_of[len(encoded)] = found
                    pair_report["conversations"] += 1
            encoded.append((torch.tensor([ids]), torch.tensor([labels])))
            masked.append(rename is not None)
            noted.append(note is not None)

    for item in items:
        if noisy is None or not args.noisy_only:
            add(item, prompts.gold_transcripts(item))
        if noisy is not None:
            add(item, noisy)
    real = len(encoded)
    if args.synth:
        # Synthetic conversations cover apps train does not have, so the prompts need their tools too.
        catalog.update(data.load_catalog("public_test", data.find_root("public_test")))
        synth = []
        for spec in args.synth.split(","):
            path = Path(__file__).parent / _BUNDLED_SYNTH[spec] if spec in _BUNDLED_SYNTH else Path(spec)
            synth += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"synthetic conversations: {len(synth)} from {args.synth}", flush=True)
        noise = bank = None
        if readings is not None:
            # Views for texts without audio replay how the real readings heard each word (independent confusion noise
            # offers the right word in braces far more often than real readings do: view_check.py).
            bank = view.ReadingsBank.from_rows([(ref, [r.get(audio, "") for r in readings]) for i in items
                                                for audio, ref in prompts.gold_transcripts(i).items() if audio in readings[0]])
            view_rng = random.Random(args.seed + 17)
            print(f"readings bank: {len(bank.seen)} said words, {len(bank.bank)} with disagreements", flush=True)
        elif noisy is not None:
            pairs = [(ref, noisy[audio]) for i in items for audio, ref in prompts.gold_transcripts(i).items() if audio in noisy]
            noise = ConfusionNoise.from_pairs(pairs)
            print(f"noise model: WER {noise.wer:.3f}, {len(noise.table)} confused words", flush=True)
        for item in synth:
            texts = prompts.gold_transcripts(item)
            if bank is not None:  # a simulated view on the latest request, the simulated main reading elsewhere
                add(item, {a: bank.view(t, view_rng) if a == item["audio"] else bank.readings(t, view_rng)[0]
                           for a, t in texts.items()})
            else:
                add(item, {a: noise.apply(t, drop_rng) for a, t in texts.items()} if noise else texts)
    sizes = sorted(e[0].shape[1] for e in encoded)
    print(f"encoded {len(encoded)} conversations ({real} real) from {len(items)} samples, mean {sum(sizes) / len(sizes):.0f} tokens, longest {sizes[-1]}", flush=True)
    if args.mask_names:
        print(f"made-up tool names in {sum(masked)} of {len(masked)} conversations", flush=True)
    if args.notes:
        print(f"worked notes in {sum(noted)} of {len(noted)} conversations", flush=True)
    if args.pairs:
        print(f"pair contrasts: {json.dumps(pair_report)}, weight {args.pair_weight}", flush=True)
        contrast.write_report(out / "pairs_attached.json", pair_report)

    n_main = len(encoded)
    if aux_notes:
        # After every main conversation, with their own random stream: the main conversations are encoded as without it.
        side_rng = random.Random(args.seed + 11)
        for item in items:
            note = aux_notes.get(item["id"])
            if note is None:
                continue
            texts = noisy if noisy is not None else prompts.gold_transcripts(item)
            ids, labels = prompts.encode_plan_example(item, texts, catalog, tok, note, args.max_seq_len, args.tool_drop, side_rng)
            if len(ids) <= args.max_seq_len:
                encoded.append((torch.tensor([ids]), torch.tensor([labels])))
        side_sizes = [e[0].shape[1] for e in encoded[n_main:]]
        print(f"plan side examples: {len(side_sizes)} of {len(aux_notes)} notes, mean {sum(side_sizes) / max(1, len(side_sizes)):.0f} "
              f"tokens, {sum(side_sizes) / max(1, sum(sizes)) * 100:.1f} % of the main tokens per epoch, each used once", flush=True)
    if args.drills:
        path = Path(args.drills) if Path(args.drills).exists() else Path(__file__).parent / args.drills
        packs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        before = len(encoded)
        for pack in packs:
            ids, labels = prompts.encode_drill_pack(pack, tok)
            if len(ids) <= args.max_seq_len:
                encoded.append((torch.tensor([ids]), torch.tensor([labels])))
        drill_sizes = [e[0].shape[1] for e in encoded[before:]]
        print(f"drill side examples: {len(drill_sizes)} packs of {len(packs)} from {path} "
              f"({sum(len(p['drills']) for p in packs)} drills), mean {sum(drill_sizes) / max(1, len(drill_sizes)):.0f} tokens, "
              f"{sum(drill_sizes) / max(1, sum(sizes)) * 100:.1f} % of the main tokens per epoch, each used once", flush=True)
    if args.extra:
        return train_stream(args, model, params, optim, scaler, encoded, n_main, catalog, tok, history_ok, holdout, out,
                            device, dtype, rank, world, len(items), bank=bank if args.synth else None,
                            noise=noise if args.synth else None)
    side = schedule.side_by_epoch(n_main, len(encoded) - n_main, args.epochs, args.seed + 13)
    per_epoch = n_main + (len(encoded) - n_main) / max(args.epochs, 1e-9)  # what one epoch of progress means, side examples spread

    total_steps = args.max_steps or schedule.planned_steps(n_main, len(encoded) - n_main, args.epochs, args.grad_accum)
    warmup = max(10, total_steps // 20)
    snapshots = sorted(float(x) for x in args.snapshot_epochs.split(",") if x.strip())

    rng = random.Random(args.seed)
    step, seen, running, epoch, restarts, spent = 0, 0, 0.0, 0, 0, 0.0
    order, position = None, 0
    resume_file = out / "resume.pt"
    if resume_file.exists():  # this process replaced one that had grown too large
        state = torch.load(resume_file, map_location=device, weights_only=False)
        set_peft_model_state_dict(model, load_file(str(out / "adapter" / "adapter_model.safetensors")))
        optim.load_state_dict(state["optim"])
        scaler.load_state_dict(state["scaler"])
        rng.setstate(state["rng"])
        step, seen, epoch, restarts, spent = state["step"], state["seen"], state["epoch"], state["restarts"], state["spent"]
        order, position = state["order"], state["position"]
        print(f"resumed at step {step}, restart {restarts}", flush=True)
    started = time.monotonic() - spent
    deadline = started + args.max_hours * 3600

    def lr_at(step):
        """Cosine to zero at total_steps, or at the deadline if that comes first."""
        used = 0.0 if args.no_time_decay else (time.monotonic() - started) / max(1.0, deadline - started)
        return schedule.cosine(step, warmup, total_steps, args.lr, used)

    model.train()
    done = False
    pair_stats = {}
    while not done:
        if order is None:
            order, position = schedule.epoch_order(n_main, side, epoch), 0
            rng.shuffle(order)
            if args.longest_first:  # smoke runs: reach the memory peak in the first steps
                order.sort(key=lambda i: -encoded[i][0].shape[1])
        while position < len(order):
            ids, labels = encoded[order[position]]
            position += 1
            try:
                with torch.autocast(device, dtype=dtype, enabled=dtype == torch.float16):
                    running += backward_loss(model, ids.to(device), labels.to(device), scaler, 1 / args.grad_accum,
                                             contrasts=contrasts_of.get(order[position - 1]),
                                             pair_weight=args.pair_weight, stats=pair_stats)
            except torch.cuda.OutOfMemoryError:  # one long sample must not cost hours of training
                print(json.dumps({"skipped_tokens": ids.shape[1], "step": step}), flush=True)
                optim.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                seen -= seen % args.grad_accum  # the partial accumulation is gone
                continue
            seen += 1
            if seen % args.grad_accum:
                continue
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            for group in optim.param_groups:
                group["lr"] = lr_at(step)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0 or step == 1:
                print(json.dumps({"epoch": round(seen / per_epoch, 3), "step": step, "of": total_steps,
                                  "loss": round(running / (args.grad_accum * (20 if step % 20 == 0 else 1)), 4),
                                  "lr": round(lr_at(step), 7), "elapsed_min": round((time.monotonic() - started) / 60, 1),
                                  "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
                                  "rss_gb": round(rss_gb(), 1),
                                  **({"pair_margin": round(pair_stats["margin"] / pair_stats["n"], 3),
                                      "pair_right": round(pair_stats["right"] / pair_stats["n"], 3),
                                      "pairs_seen": pair_stats["n"]} if pair_stats.get("n") else {})}), flush=True)
                running = 0.0
                pair_stats.clear()
            if step % args.save_every == 0:  # a killed process must not lose hours of training
                model.save_pretrained(out / "adapter")
            for mark in snapshots:  # the folder is the record, so a restarted process does not save one twice
                folder = out / f"adapter_e{mark:g}"
                if seen >= mark * per_epoch and not folder.exists():
                    model.save_pretrained(folder)
                    print(json.dumps({"snapshot": folder.name, "step": step, "epoch": round(seen / per_epoch, 3)}), flush=True)
            if step % 20 == 0:  # RAM creeps up over a run and two runs share one host, so hand freed pages back
                gc.collect()
                _LIBC.malloc_trim(0)
            if step >= total_steps or time.monotonic() > deadline:
                done = True
                break
            if rss_gb() > args.max_rss_gb and restarts < args.max_restarts:
                # RAM grows slowly over hours, and near the host's limit training first crawls and is then killed. A fresh process
                # costs a few minutes and works whatever the cause.
                model.save_pretrained(out / "adapter")
                torch.save({"optim": optim.state_dict(), "scaler": scaler.state_dict(), "rng": rng.getstate(), "step": step, "seen": seen,
                            "epoch": epoch, "restarts": restarts + 1, "spent": time.monotonic() - started, "order": order, "position": position}, resume_file)
                print(f"restarting at step {step}: RSS {rss_gb():.1f} GB", flush=True)
                os.execv(sys.executable, [sys.executable, "-m", "lifemate.train", *sys.argv[1:]])
        if not done:
            order = None
            epoch += 1

    resume_file.unlink(missing_ok=True)
    model.save_pretrained(out / "adapter")
    metrics = {"steps": step, "seen": seen, "epochs": seen / per_epoch, "samples": len(items), "conversations": n_main,
               "side_examples": len(encoded) - n_main,
               "minutes": (time.monotonic() - started) / 60,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics), flush=True)
    return metrics


if __name__ == "__main__":
    main()
