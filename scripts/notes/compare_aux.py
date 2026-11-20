"""The notes side task (--notes-aux) against the baseline on the 386 validation samples, with the rule fixed in log.md
(8 Nov 2026, "How to train on the v2 notes") before any result:

    into the final only if the validation mean is at least 67.71 (+1.0 over 0008's 66.71), neither half (random,
    held-out apps) is lower by more than 1.0, and no note is written in normal mode.

    python scripts/notes/compare_aux.py outputs/<run>/aux_eval outputs/harm-check5/avg5_eval [--out decision.json]

"Note written in normal mode": the first turn's think block holds more than the restated request (a second non-empty
line, or an app.tool name). The baseline is measured the same way, so a habit both share is not counted against the
side task.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_notes import BASELINE_MEAN, SPLITS, batch, load, paired, seconds  # noqa: E402

TOOL_NAME = re.compile(r"(?<![\w@.])[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*(?![\w.@])")
REPO = Path(__file__).resolve().parents[2]
BAR, HALF_FLOOR = BASELINE_MEAN + 1.0, -1.0


def think_block(row: dict) -> str:
    turns = row.get("turns") or [""]
    first = turns[0]
    return first.split("</think>")[0] if "</think>" in first else first


def tool_names() -> set:
    sys.path.insert(0, str(REPO / "src"))
    from lifemate import data
    names = set()
    for split in ("train", "public_test"):
        names |= set(data.load_catalog(split, data.find_root(split)))
    return names


def noted(row: dict, tools: set) -> bool:
    think = think_block(row)
    lines = [line for line in think.strip().split("\n") if line.strip()]
    return len(lines) > 1 or any(m in tools for m in TOOL_NAME.findall(think))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("aux")
    ap.add_argument("baseline")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rounds", type=int, default=20000)
    args = ap.parse_args()
    a, b = load(args.aux), load(args.baseline)
    ids = sorted(set(a) & set(b))
    assert len(ids) == 386 == len(a) == len(b), (len(ids), len(a), len(b))
    rng = random.Random(2026)
    res = {"overall": paired(ids, a, b, args.rounds, rng)}
    for k in "ABC":
        res[f"batch {k}"] = paired([i for i in ids if batch(i) == k], a, b, args.rounds, rng)
    res["random half"] = paired([i for i in ids if i in SPLITS["random"]], a, b, args.rounds, rng)
    res["held-out-app half"] = paired([i for i in ids if i not in SPLITS["random"]], a, b, args.rounds, rng)
    res["seconds aux"], res["seconds baseline"] = seconds(a, ids), seconds(b, ids)
    tools = tool_names()
    leaks = [i for i in ids if noted(a[i], tools)]
    base_leaks = [i for i in ids if noted(b[i], tools)]
    res["notes in normal mode"] = {"aux": len(leaks), "baseline": len(base_leaks), "aux ids": leaks[:20],
                                   "aux examples": [think_block(a[i])[:300] for i in leaks[:5]]}
    rule = {
        f"validation mean at least {BAR:.2f}": res["overall"]["notes"] >= BAR,
        "random half not lower by more than 1.0": res["random half"]["diff"] >= HALF_FLOOR,
        "held-out-app half not lower by more than 1.0": res["held-out-app half"]["diff"] >= HALF_FLOOR,
        "no note written in normal mode": len(leaks) <= len(base_leaks),
    }
    res["rule"] = rule
    res["decision"] = "KEEP" if all(rule.values()) else "DROP"
    text = json.dumps(res, indent=1, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
