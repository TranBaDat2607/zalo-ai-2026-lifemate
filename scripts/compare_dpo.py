"""Step-preference arms against the side-task pair (0012, m1+m2, 71.57) on the 386 validation samples, with the rule
fixed in log.md (10 Nov 2026, "Step-preference validation run") before any result:

    the arm with the higher validation mean is judged; it needs a mean >= 73.07 (71.57 + 1.5, the extra 0.5 pays for
    picking the better of two arms), neither half (random, held-out apps) lower than 0012's by more than 1.0, and no
    more invalid calls or broken asks than 0012.

"Invalid calls" = the agent loop's `repairs` (a generated call the environment's validator rejected); "broken asks" =
samples that ended for any reason other than "done" or "asked" (time budget, context, step limit). Both counted over
all 386 samples and compared with 0012's counts (7 and 0).

    python scripts/compare_dpo.py outputs/dpo-val2/dpoA_eval outputs/dpo-val2/dpoB_eval \
        --baseline outputs/notes-aux/aux_eval --out outputs/dpo-val2/decision.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "notes"))
from compare_notes import SPLITS, batch, load, paired, seconds  # noqa: E402

BASELINE_MEAN, BAR, HALF_FLOOR = 71.57, 73.07, -1.0
NORMAL_STOPS = {"done", "asked"}


def summary(rows: dict, ids: list) -> dict:
    return {"mean": round(100 * sum(rows[i]["score"] for i in ids) / len(ids), 2),
            "repairs": sum(rows[i].get("repairs", 0) for i in ids),
            "abnormal_stops": sum(rows[i].get("stop") not in NORMAL_STOPS for i in ids),
            "asked": sum(rows[i].get("stop") == "asked" for i in ids),
            "perfect": sum(rows[i]["score"] > 0.999 for i in ids), "zero": sum(rows[i]["score"] < 1e-9 for i in ids)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="eval folders with detail.*.jsonl (one per arm)")
    ap.add_argument("--baseline", default="outputs/notes-aux/aux_eval")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rounds", type=int, default=20000)
    ap.add_argument("--bar", type=float, default=BAR, help="mean the judged arm needs (73.07 for two arms; one arm: 72.57)")
    args = ap.parse_args()
    base = load(args.baseline)
    ids = sorted(base)
    assert len(ids) == 386, len(ids)
    b = summary(base, ids)
    assert abs(b["mean"] - BASELINE_MEAN) < 0.01, b  # the reference run is the one the rule names
    results = {}
    for run in args.arms:
        rows = load(run)
        missing = [i for i in ids if i not in rows]
        if missing:
            raise SystemExit(f"{run}: {len(missing)} validation samples missing")
        rng = random.Random(2026)
        res = {"summary": summary(rows, ids), "all": paired(ids, rows, base, args.rounds, rng),
               "random half": paired([i for i in ids if i in SPLITS["random"]], rows, base, args.rounds, rng),
               "held-out-app half": paired([i for i in ids if i not in SPLITS["random"]], rows, base, args.rounds, rng),
               "by batch": {k: paired([i for i in ids if batch(i) == k], rows, base, args.rounds, rng) for k in "ABC"},
               "seconds": seconds(rows, ids)}
        s = res["summary"]
        res["checks"] = {f"mean >= {args.bar}": s["mean"] >= args.bar,
                         "random half not lower by more than 1.0": res["random half"]["diff"] >= HALF_FLOOR,
                         "held-out-app half not lower by more than 1.0": res["held-out-app half"]["diff"] >= HALF_FLOOR,
                         f"invalid calls (repairs) <= {b['repairs']}": s["repairs"] <= b["repairs"],
                         f"broken asks (abnormal stops) <= {b['abnormal_stops']}": s["abnormal_stops"] <= b["abnormal_stops"]}
        results[run] = res
    judged = max(results, key=lambda r: results[r]["summary"]["mean"])
    decision = "KEEP" if all(results[judged]["checks"].values()) else "DROP"
    out = {"baseline": {"run": args.baseline, **b}, "arms": results, "judged": judged, "decision": decision}
    text = json.dumps(out, indent=1, ensure_ascii=False)
    print(text)
    for run, res in results.items():
        a = res["all"]
        print(f"{run}: mean {res['summary']['mean']} ({a['diff']:+.2f} vs 0012, 95% {a['low']:+.2f} .. {a['high']:+.2f}; "
              f"wins {a['wins']} / losses {a['losses']}); random {res['random half']['diff']:+.2f}, held-out apps "
              f"{res['held-out-app half']['diff']:+.2f}; repairs {res['summary']['repairs']}, abnormal stops "
              f"{res['summary']['abnormal_stops']}, asked {res['summary']['asked']} (0012: {b['asked']})")
    print(f"judged: {judged} -> {decision}")
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
