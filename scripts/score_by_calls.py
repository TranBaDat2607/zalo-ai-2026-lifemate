"""Validation score by the number of gold calls, from the detail.*.jsonl files of one or more runs.

    python scripts/score_by_calls.py twin=outputs/harm-check5/avg5_eval aux=outputs/notes-aux/aux_eval

Per bucket: samples, mean score, perfect (score 1), zero (wrong state hash, or right hash but no matched call),
partial, and whether the run made fewer / as many / more calls than gold. "lost" is the bucket's share of the
points lost on the whole set (100 - overall score = the sum of the column). The first run also gets a split by
batch (A/B/C) and by request type (act / ask = gold ends with sys.ask_missing_info).
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "notes"))
from compare_notes import batch, load  # noqa: E402

BUCKETS = ["0", "1", "2", "3", "4", "5+"]


def bucket(n: int) -> str:
    return "5+" if n >= 5 else str(n)


def kind(row: dict) -> str:
    gold = row["gold"]
    return "none" if not gold else ("ask" if gold[-1]["name"] == "sys.ask_missing_info" else "act")


def table(name: str, rows: dict, ids: list) -> None:
    n_all = len(ids)
    print(f"\n== {name}: overall {100 * sum(rows[i]['score'] for i in ids) / n_all:.2f} on {n_all} samples")
    print(f"{'calls':>5} {'n':>4} {'share':>6} {'score':>6} {'perfect':>8} {'zero':>5} {'partial':>8} "
          f"{'fewer':>6} {'same':>5} {'more':>5} {'lost':>5}")
    groups = collections.defaultdict(list)
    for i in ids:
        groups[bucket(len(rows[i]["gold"]))].append(rows[i])
    for b in BUCKETS:
        rs = groups.get(b, [])
        if not rs:
            continue
        n = len(rs)

        def pct(test) -> float:
            return 100 * sum(1 for r in rs if test(r)) / n

        print(f"{b:>5} {n:>4} {100 * n / n_all:>5.0f}% {100 * sum(r['score'] for r in rs) / n:>6.1f} "
              f"{pct(lambda r: r['score'] > 0.999):>7.0f}% {pct(lambda r: r['score'] == 0):>4.0f}% "
              f"{pct(lambda r: 0 < r['score'] < 0.999):>7.0f}% {pct(lambda r: len(r['pred']) < len(r['gold'])):>5.0f}% "
              f"{pct(lambda r: len(r['pred']) == len(r['gold'])):>4.0f}% {pct(lambda r: len(r['pred']) > len(r['gold'])):>4.0f}% "
              f"{100 * sum(1 - r['score'] for r in rs) / n_all:>5.1f}")
    zero_hash = sum(1 for i in ids if not rows[i]["hash_ok"])
    zero_match = sum(1 for i in ids if rows[i]["hash_ok"] and rows[i]["score"] == 0)
    print(f"zeros: wrong state {zero_hash}, right state but no matched call {zero_match}")


def split(rows: dict, ids: list, label: str, key) -> None:
    cells = collections.defaultdict(list)
    for i in ids:
        cells[(bucket(len(rows[i]["gold"])), key(i, rows[i]))].append(rows[i]["score"])
    print(f"\nby calls x {label} (n, score)")
    for b in BUCKETS:
        parts = [f"{k}: n={len(s)} {100 * sum(s) / len(s):.1f}" for k in sorted({k for bb, k in cells if bb == b})
                 for s in [cells[(b, k)]]]
        if parts:
            print(f"  {b:>2} calls | " + " | ".join(parts))


def main() -> None:
    runs = {name: load(path) for name, path in (a.split("=", 1) for a in sys.argv[1:])}
    ids = sorted(set.intersection(*(set(r) for r in runs.values())))
    for name, rows in runs.items():
        table(name, rows, ids)
    first = next(iter(runs.values()))
    split(first, ids, "batch", lambda i, r: batch(i))
    split(first, ids, "type", lambda i, r: kind(r))


if __name__ == "__main__":
    main()
