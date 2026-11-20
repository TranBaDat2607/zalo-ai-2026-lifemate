#!/usr/bin/env python3
"""Compare two recipes over the four cross-validation folds (src/lifemate/folds_cv.json), sample by sample.

    python scripts/compare_cv.py "outputs/view-cv0/view_eval,outputs/cv1-view/view_eval,..." \
                                 "outputs/notes-aux/aux_eval,outputs/cv1-aux/aux_eval,..." [--rule]

Each side is a comma-separated list of eval folders (one per fold, any order); rows are joined on sample id, and only
ids scored on both sides and listed in their fold's score set count. Prints a minus b with a paired bootstrap 95 %
interval (resampling within each fold, so every fold keeps its size), overall, for the unseen-apps part ("held") and
the random part, per fold and per batch. --rule applies improve_plan.md's step-0 rule: KEEP if the gain is at least
+1.0, the interval's lower end is above -0.5, and the unseen-apps part is not down by more than 1.0.
"""
import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from compare_runs import bucket, load  # noqa: E402

FOLDS = json.loads((REPO / "src" / "lifemate" / "folds_cv.json").read_text(encoding="utf-8"))
FOLD_OF = {s: k for k in FOLDS["held"] for part in ("held", "random") for s in FOLDS[part][k]}
PART_OF = {s: part for part in ("held", "random") for k in FOLDS[part] for s in FOLDS[part][k]}


def load_side(spec: str) -> dict:
    rows = {}
    for folder in filter(None, spec.split(",")):
        found = load(folder)
        if not found:
            sys.exit(f"{folder}: no detail.*.jsonl")
        rows.update(found)
    return rows


def interval(diffs_by_fold: dict, rounds: int, rng: random.Random) -> tuple:
    n = sum(len(d) for d in diffs_by_fold.values())
    means = sorted(sum(sum(rng.choices(d, k=len(d))) for d in diffs_by_fold.values() if d) / n for _ in range(rounds))
    return means[int(rounds * 0.025)], means[int(rounds * 0.975) - 1]


def report(name: str, ids: list, a: dict, b: dict, rounds: int, rng: random.Random) -> dict:
    by_fold = {}
    for i in ids:
        by_fold.setdefault(FOLD_OF[i], []).append(a[i]["score"] - b[i]["score"])
    diffs = [d for v in by_fold.values() for d in v]
    low, high = interval(by_fold, rounds, rng)
    mean_a = 100 * sum(a[i]["score"] for i in ids) / len(ids)
    mean_b = 100 * sum(b[i]["score"] for i in ids) / len(ids)
    gain = 100 * sum(diffs) / len(diffs)
    print(f"{name:12s} n={len(ids):4d}  a={mean_a:6.2f}  b={mean_b:6.2f}  a-b={gain:+6.2f}  "
          f"95% [{100 * low:+6.2f}, {100 * high:+6.2f}]  a_wins={sum(d > 1e-9 for d in diffs)}  "
          f"b_wins={sum(d < -1e-9 for d in diffs)}")
    return {"n": len(ids), "a": mean_a, "b": mean_b, "gain": gain, "low": 100 * low, "high": 100 * high}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rounds", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--rule", action="store_true", help="apply the step-0 KEEP rule")
    ap.add_argument("--out", default=None, help="write the numbers as JSON")
    args = ap.parse_args()
    a, b = load_side(args.a), load_side(args.b)
    ids = sorted(i for i in set(a) & set(b) if i in FOLD_OF)
    if not ids:
        sys.exit("no samples in common")
    folds = sorted({FOLD_OF[i] for i in ids})
    missing = {k: len(FOLDS["held"][k]) + len(FOLDS["random"][k]) - sum(FOLD_OF[i] == k for i in ids) for k in folds}
    print(f"folds {','.join(folds)}; {len(ids)} samples; missing per fold {missing}; scores in points")
    rng = random.Random(args.seed)
    result = {"folds": folds, "missing": missing, "all": report("all", ids, a, b, args.rounds, rng)}
    for part in ("held", "random"):
        result[part] = report(part, [i for i in ids if PART_OF[i] == part], a, b, args.rounds, rng)
    for k in folds:
        result[f"fold {k}"] = report(f"fold {k}", [i for i in ids if FOLD_OF[i] == k], a, b, args.rounds, rng)
    for key in "ABC":
        part = [i for i in ids if bucket(i) == key]
        if part:
            result[f"batch {key}"] = report(f"batch {key}", part, a, b, args.rounds, rng)
    if args.rule:
        r = result
        keep = r["all"]["gain"] >= 1.0 and r["all"]["low"] > -0.5 and r["held"]["gain"] >= -1.0
        result["decision"] = "KEEP" if keep else "DROP"
        print(f"step-0 rule: gain {r['all']['gain']:+.2f} (>= +1.0), lower end {r['all']['low']:+.2f} (> -0.5), "
              f"unseen apps {r['held']['gain']:+.2f} (>= -1.0) -> {result['decision']}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
