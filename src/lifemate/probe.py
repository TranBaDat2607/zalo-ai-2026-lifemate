"""Step probing: the model's next step at every gold step, for preference pairs (data v2, 10 Nov 2026).

    python -m lifemate.probe --out {out}/probe_f1a --adapter {path} --ids-file fold:1 --split train \
        --transcripts auto:avg5_train.json --samples 4 --temperature 0.8 --shard 0 --shards 1

The first-divergence runs (build_pairs.py) see one mistake per sample: the agent's own first wrong step. Here the
history is always gold, as in training, so every gold step t is a decision point: the prompt (the request as heard),
the model's own restated request in the first think block, then gold calls 0..t-1 with their gold results. At each
point the model answers greedily and ``--samples`` more times with temperature sampling; the gold continuation's
log-probability is measured on the way, since the prefix must be fed anyway to reach step t + 1. After the last gold
call the gold answer is the closing reply (prompts.DONE), so stopping too early and calling once too often are probed as
well; an out-of-scope sample (no gold call) has one point, step 0, whose gold answer is the closing reply.

The prompt is fed once per sample. Before the answers at a step the cache is copied (``infer.snapshot_cache``) and put
back after each answer (``infer.restore_cache``, the rollback that prompt lookup uses), so the answers at one step all
see the same prefix and cost only their own tokens.

One JSON row per (sample, step) in ``probe.{shard}.jsonl``: gold (a call or "DONE"), the answers (text, parsed call or
"DONE" or null, log-prob of the call part under the model at temperature 1), lp_gold, and the restated request.
Training samples only (gold is needed); never part of inference.
"""
import argparse
import json
import math
import time
from pathlib import Path

from . import data, prompts
from .infer import (MAX_CONTEXT, load_adapted, load_transcripts, parse_call, resolve_adapter, restore_cache,
                    snapshot_cache, tidy)

DONE = "DONE"


def restated_of(text: str) -> str:
    """The restated request of a first turn, without the newline THINK_CLOSE adds back (think_lead(restated) must give
    back exactly what the model wrote: 'text\\n</think>\\n\\n')."""
    head = text.split("</think>")[0] if "</think>" in text else ""
    return head[:-1] if head.endswith("\n") else head


def answer_of(text: str, catalog: dict):
    """What a generated turn does: {"name", "arguments"} for a call, "DONE" for a closing reply, None if malformed."""
    tail = text.split("</think>", 1)[1] if "</think>" in text else text
    parsed = parse_call(tail)
    if parsed is not None:
        name, arguments = parsed
        return {"name": name, "arguments": tidy(name, arguments, catalog)}
    if "<tool_call>" in tail:
        return None
    return DONE


def call_start(tokenizer, tokens: list, first: bool) -> int:
    """Index of the first token after the think block of a first turn (0 for later turns, whose think block is in the
    prefix), so log-probs compare the call parts only."""
    if not first:
        return 0
    for k in range(1, len(tokens) + 1):
        if tokenizer.decode(tokens[:k], skip_special_tokens=False).endswith("</think>\n\n"):
            return k
    return len(tokens)


class Prober:
    """Holds the model, its cache and the last logits of the prefix fed so far."""

    def __init__(self, tokenizer, model, max_new_tokens: int, temperature: float, top_p: float, seed: int):
        import torch
        self.torch, self.tokenizer, self.model = torch, tokenizer, model
        self.max_new_tokens, self.temperature, self.top_p, self.seed = max_new_tokens, temperature, top_p, seed
        self.stop_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.reset()

    def reset(self) -> None:
        # fed: every token of the prefix, in order (cheap, and tests compare it with dpo.encode_pair)
        self.cache, self.last, self.length, self.fed = None, None, 0, []

    def encode(self, text: str) -> list:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def feed(self, tokens: list, keep_all: bool = False):
        """Feeds ``tokens`` after the prefix; returns the float logits of every position when ``keep_all``."""
        torch = self.torch
        with torch.inference_mode():
            out = self.model(input_ids=torch.tensor([tokens], device=self.model.device), past_key_values=self.cache,
                             use_cache=True, logits_to_keep=0 if keep_all else 1)
        self.cache, self.length = out.past_key_values, self.length + len(tokens)
        self.fed.extend(tokens)
        logits = out.logits[0].float()
        self.last = logits[-1]
        return logits if keep_all else None

    def score(self, tokens: list, start: int) -> float:
        """Feeds ``tokens`` (they stay in the prefix) and returns the summed log-prob of tokens[start:]."""
        before = self.last
        logits = self.feed(tokens, keep_all=True)
        dists = self.torch.cat([before[None], logits[:-1]])
        lps = self.torch.log_softmax(dists, dim=-1).gather(1, self.torch.tensor(tokens, device=dists.device)[:, None])
        return float(lps[start:].sum())

    def pick(self, logits, rng):
        torch = self.torch
        if rng is None:
            return int(logits.argmax())
        probs = torch.softmax(logits / self.temperature, dim=-1)
        if self.top_p < 1.0:
            sorted_p, order = probs.sort(descending=True)
            cut = int((sorted_p.cumsum(0) < self.top_p).sum()) + 1
            probs = torch.zeros_like(probs).scatter(0, order[:cut], sorted_p[:cut])
        return int(torch.multinomial(probs, 1, generator=rng))

    def answer(self, rng=None) -> tuple:
        """One answer from the current prefix (greedy when ``rng`` is None); the prefix is left as it was.
        Returns (tokens, per-token log-probs at temperature 1)."""
        torch = self.torch
        last, length = self.last, self.length
        tokens, lps, logits = [], [], self.last
        with torch.inference_mode():  # the cache tensors are inference tensors: copy and put back inside the mode
            saved = snapshot_cache(self.cache)
            for _ in range(self.max_new_tokens):
                token = self.pick(logits, rng)
                tokens.append(token)
                lps.append(float(torch.log_softmax(logits, dim=-1)[token]))
                if token == self.stop_id:
                    break
                out = self.model(input_ids=torch.tensor([[token]], device=self.model.device),
                                 past_key_values=self.cache, use_cache=True, logits_to_keep=1)
                self.cache, logits = out.past_key_values, out.logits[0, -1].float()
            restore_cache(self.cache, saved)
        self.last, self.length = last, length
        return tokens, lps


def probe_sample(prober: Prober, item: dict, texts: dict, catalog: dict, samples: int, sample_seed: int) -> list:
    """The rows of one sample: one per gold step plus the closing step."""
    torch = prober.torch
    calls, results = item["label"]["tool_calls"], item["label"]["tool_results"]
    prober.reset()
    prober.feed(prober.encode(prompts.prompt_text(item, texts, catalog) + prompts.THINK_OPEN))
    rows, restated = [], ""
    for t in range(len(calls) + 1):
        if prober.length + prober.max_new_tokens > MAX_CONTEXT:
            break
        answers = []
        for k in range(samples + 1):
            rng = None
            if k:
                rng = torch.Generator(device=prober.model.device)
                rng.manual_seed(sample_seed * 1000 + t * 50 + k)
            tokens, lps = prober.answer(rng)
            text = prober.tokenizer.decode(tokens, skip_special_tokens=False).replace("<|im_end|>", "")
            start = call_start(prober.tokenizer, tokens, first=t == 0)
            answers.append({"text": text, "answer": answer_of(text, catalog), "lp": round(sum(lps[start:]), 4),
                            "ended": bool(tokens) and tokens[-1] == prober.stop_id,
                            **({"restated": restated_of(text)} if t == 0 else {})})
        if t == 0:
            restated = restated_of(answers[0]["text"])  # the greedy restated request leads every later prefix
        lead = prompts.think_lead(restated) if t == 0 else ""
        gold = calls[t] if t < len(calls) else DONE
        gold_text = (prompts.call_text(gold["name"], gold.get("arguments") or {}) if gold != DONE else prompts.DONE)
        tokens = prober.encode(lead + gold_text + "<|im_end|>")
        lead_tokens = prober.encode(lead) if lead else []
        start = len(lead_tokens) if tokens[:len(lead_tokens)] == lead_tokens else 0
        lp_gold = prober.score(tokens, start)
        rows.append({"id": item["id"], "step": t, "steps": len(calls), "gold": gold, "lp_gold": round(lp_gold, 4),
                     "restated": restated, "answers": answers})
        if t < len(calls):
            prober.feed(prober.encode("\n" + prompts.turn("user", prompts.result_text(results[t])) + prompts.GEN_PREFIX))
    return rows


def main(argv=None) -> None:
    from . import models
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm", default=models.DEFAULT_LLM)
    ap.add_argument("--adapter", default=None, help="adapter folder(s), comma-separated for a soup (auto:NAME allowed)")
    ap.add_argument("--adapter-weights", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--transcripts", default="gold", help="'gold' or the transcripts the agent hears")
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--samples", type=int, default=4, help="sampled answers per step besides the greedy one")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-hours", type=float, default=0.0, help="stop starting samples after this long (0 = no limit)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args(argv)
    started = time.monotonic()
    models.seed_everything(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = data.find_root(args.split)
    items = data.load_split(args.split, root)
    catalog = data.load_catalog(args.split, root)
    if args.ids_file:
        keep = data.id_set(args.ids_file)
        items = [i for i in items if i["id"] in keep]
    if args.limit:
        items = items[:args.limit]
    items = items[args.shard::args.shards]
    adapter = ",".join(resolve_adapter(a) for a in args.adapter.split(",")) if args.adapter else None
    tokenizer, model = load_adapted(args.llm, adapter, args.adapter_weights)
    model.eval()
    prober = Prober(tokenizer, model, args.max_new_tokens, args.temperature, args.top_p, args.seed)
    transcripts = None if args.transcripts == "gold" else load_transcripts(args.transcripts)
    path = out / f"probe.{args.shard}.jsonl"
    done = set()
    if path.exists():  # a restarted run continues where the last one stopped
        done = {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    stats = {"samples": 0, "rows": 0, "skipped_done": len(done), "stopped_by_time": False}
    with open(path, "a", encoding="utf-8") as fout:
        for n, item in enumerate(items):
            if item["id"] in done:
                continue
            if args.max_hours and time.monotonic() - started > args.max_hours * 3600:
                stats["stopped_by_time"] = True
                print("time limit reached", flush=True)
                break
            tic = time.monotonic()
            texts = prompts.gold_transcripts(item) if transcripts is None else transcripts
            rows = probe_sample(prober, item, texts, catalog, args.samples, args.seed + n)
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            stats["samples"] += 1
            stats["rows"] += len(rows)
            wrong = sum(1 for r in rows for a in r["answers"] if a["answer"] != r["gold"])
            print(f"{n + 1}/{len(items)} {item['id']} steps={len(rows)} differing answers={wrong}/"
                  f"{sum(len(r['answers']) for r in rows)} {time.monotonic() - tic:.1f}s", flush=True)
    stats.update({"adapter": adapter, "samples_per_step": args.samples, "temperature": args.temperature,
                  "top_p": args.top_p, "minutes": round((time.monotonic() - started) / 60, 1)})
    (out / f"probe_metrics.{args.shard}.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
