"""Preference pairs inside SFT: a contrast at the first token where the gold step and a wrong step part (10 Nov 2026).

The stand-alone step-preference stage (dpo.py) dropped 12 points on validation: 229 pairs that were all exceptions,
trained alone, pushed the wrong steps down by 14-25 nats and moved the freed probability to asking (log.md). Here a
pair is attached to its own training conversation instead. At step t of a gold conversation the prefix is gold, so the
gold step's tokens are already targets of the SFT loss; the wrong step shares them up to the first token where the two
part. The logits at that position are computed anyway, so the contrast costs nothing:

    loss = SFT + pair_weight * sum_pairs w * -log sigmoid(z[gold token] - z[wrong token]) / sum_pairs w

It only acts where the wrong token competes with the gold one (the sigmoid saturates past a few logits), every other
token of every conversation keeps its SFT target as the anchor, and there is no reference model to drift from.
"""
import json
from pathlib import Path

from . import prompts

DONE = "DONE"


def side_text(side) -> str:
    """The assistant turn of a pair side without its think block: a call, or the closing reply."""
    if side == DONE:
        return prompts.DONE
    return prompts.call_text(side["name"], side.get("arguments") or {})


def load_pairs(spec: str, holdout: set) -> dict:
    """{sample id: [pairs]} from a pairs file (probe_pairs.py / build_pairs.py, or auto:NAME under /kaggle/input).
    Pairs of held-out samples are refused: they would carry the held-out answers into training."""
    from .dpo import load_pairs as read
    pairs = read(spec)
    leaked = sorted({p["id"] for p in pairs} & holdout)
    if leaked:
        raise SystemExit(f"pairs exist for {len(leaked)} held-out samples, e.g. {leaked[:3]}")
    out = {}
    for p in pairs:
        out.setdefault(p["id"], []).append(p)
    print(f"preference pairs: {len(pairs)} for {len(out)} samples from {spec}", flush=True)
    return out


def tail_segments(item: dict, tok, denoise: bool, note: str = None) -> list:
    """[(step, lead, tokens)] of the target turns of the latest request plus [(None, '', tokens)] for the result
    turns between them, in order, exactly as prompts._encode_once encodes the end of a conversation."""
    lead = prompts.think_lead(item["audio_text"], note) if denoise else ""
    calls, results = item["label"]["tool_calls"], item["label"]["tool_results"]
    segs = []
    for t, (call, result) in enumerate(zip(calls, results)):
        segs.append((t, lead, tok(lead + prompts.call_text(call["name"], call["arguments"]) + "<|im_end|>")))
        lead = ""
        segs.append((None, "", tok("\n" + prompts.turn("user", prompts.result_text(result)) + prompts.GEN_PREFIX)))
    segs.append((len(calls), lead, tok(lead + prompts.DONE + "<|im_end|>")))
    return segs


def locate(ids: list, labels: list, item: dict, pairs: list, tok, denoise: bool, note: str = None) -> tuple:
    """([(position, gold token, wrong token, weight)], skipped): for each pair, the position in ``ids`` of the first gold
    token that differs from the wrong step. ``tok`` must be the tokenizer the conversation was encoded with (with its
    renaming, if any). A pair whose chosen side is not this conversation's gold step is skipped."""
    segs = tail_segments(item, tok, denoise, note)
    end, starts = len(ids), {}
    for step, lead, tokens in reversed(segs):
        end -= len(tokens)
        if step is not None:
            starts[step] = (end, lead, tokens)
    calls = item["label"]["tool_calls"]
    found, skipped = [], 0
    for p in pairs:
        t = p["step"]
        gold = calls[t] if t < len(calls) else DONE
        if t not in starts or p["chosen"] != gold:
            skipped += 1
            continue
        start, lead, chosen = starts[t]
        if ids[start:start + len(chosen)] != chosen:
            skipped += 1  # the conversation was cut or encoded differently
            continue
        rejected = tok(lead + side_text(p["rejected"]) + "<|im_end|>")
        d = 0
        while d < min(len(chosen), len(rejected)) and chosen[d] == rejected[d]:
            d += 1
        if d >= min(len(chosen), len(rejected)) or labels[start + d] == -100:
            skipped += 1
            continue
        found.append((start + d, chosen[d], rejected[d], float(p.get("weight", 1.0))))
    return found, skipped


def rows_for(positions: list, contrasts: list) -> list:
    """[(row in the loss positions, gold token, wrong token, weight)]: the loss row j predicts ids[j + 1]."""
    index = {j: k for k, j in enumerate(positions)}
    return [(index[p - 1], c, r, w) for p, c, r, w in contrasts if p - 1 in index]


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
