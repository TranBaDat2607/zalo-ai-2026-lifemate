#!/usr/bin/env python3
"""Compare two validation runs sample by sample, on the samples both have.

    python scripts/compare_runs.py outputs/exp9/fold_dn_eval outputs/exp10/fold_eval

Prints a minus b with a paired bootstrap 95 % interval, overall and by group.
"""
import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPLITS = json.loads((REPO / "src" / "lifemate" / "splits.json").read_text(encoding="utf-8"))


def load(run: str) -> dict:
    rows = {}
    for f in sorted(Path(run).glob("detail.*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def bucket(sample_id: str) -> str:
    n = int(sample_id.split("_")[1])
    return "A" if n < 400 else ("B" if n < 900 else "C")


def interval(diffs: list, rounds: int, rng: random.Random) -> tuple:
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(rounds))
    return means[int(rounds * 0.025)], means[int(rounds * 0.975) - 1]


def report(name: str, ids: list, a: dict, b: dict, rounds: int, rng: random.Random) -> None:
    diffs = [a[i]["score"] - b[i]["score"] for i in ids]
    low, high = interval(diffs, rounds, rng)
    mean = lambda rows: 100 * sum(r[i]["score"] for r, i in ((rows, i) for i in ids)) / len(ids)  # noqa: E731
    changed = sum(abs(d) > 1e-9 for d in diffs)
    rms = (sum(d * d for d in diffs if abs(d) > 1e-9) / max(changed, 1)) ** 0.5
    print(f"{name:14s} n={len(ids):3d}  a={mean(a):6.2f}  b={mean(b):6.2f}  a-b={100 * sum(diffs) / len(ids):+6.2f}  "
          f"95% [{100 * low:+6.2f}, {100 * high:+6.2f}]  changed={100 * changed / len(ids):5.1f}%  rms={rms:.2f}  "
          f"a_wins={sum(d > 1e-9 for d in diffs)}  b_wins={sum(d < -1e-9 for d in diffs)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rounds", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    ids = sorted(set(a) & set(b))
    if not ids:
        sys.exit("no samples in common")
    rng = random.Random(args.seed)
    print(f"a = {args.a} ({len(a)} samples), b = {args.b} ({len(b)} samples), scores in points")
    report("all", ids, a, b, args.rounds, rng)
    for key in "ABC":
        part = [i for i in ids if bucket(i) == key]
        if part:
            report(f"bucket {key}", part, a, b, args.rounds, rng)
    report("random", [i for i in ids if i in SPLITS["random"]], a, b, args.rounds, rng)
    report("held-out apps", [i for i in ids if i not in SPLITS["random"]], a, b, args.rounds, rng)
    for name, rows in (("a", a), ("b", b)):
        secs = sorted(rows[i]["seconds"] for i in ids)
        print(f"seconds {name}: mean {sum(secs) / len(secs):.2f}  p95 {secs[int(len(secs) * .95)]}  max {secs[-1]}  "
              f"over 30 s: {sum(s > 30 for s in secs)}")


if __name__ == "__main__":
    main()
