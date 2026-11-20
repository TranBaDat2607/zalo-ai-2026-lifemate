"""Step 3 check (improve_plan.md; rule in log.md, fixed before the run): voting between 0012's two fold copies and
their soup against the soup alone (0012, outputs/notes-aux/aux_eval, 71.57), same 386 samples, same settings.

    python3 scripts/review/vote_check.py outputs/vote-check/vote_eval [outputs/notes-aux/aux_eval]

Pass: mean gain >= +1.0 and neither half (random / held-out apps) more than 1.0 below. Code check (must hold for the
result to count): every sample where the three members agreed at every turn has the soup's exact calls.
"""
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from compare_runs import SPLITS, interval, load  # noqa: E402

GAIN_MIN, HALF_FLOOR = 1.0, -1.0


def gain(ids, a, b, rng):
    diffs = [a[i]["score"] - b[i]["score"] for i in ids]
    low, high = interval(diffs, 20000, rng)
    return {"n": len(ids), "vote": round(100 * sum(a[i]["score"] for i in ids) / len(ids), 2),
            "soup": round(100 * sum(b[i]["score"] for i in ids) / len(ids), 2),
            "gain": round(100 * sum(diffs) / len(ids), 2), "ci95": [round(100 * low, 2), round(100 * high, 2)],
            "vote_wins": sum(d > 1e-9 for d in diffs), "soup_wins": sum(d < -1e-9 for d in diffs)}


def main() -> None:
    run = Path(sys.argv[1])
    base = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "outputs/notes-aux/aux_eval"
    a, b = load(str(run)), load(str(base))
    ids = sorted(set(a) & set(b))
    print(f"vote {len(a)} samples, soup {len(b)} samples, {len(ids)} in common")
    rng = random.Random(2026)
    report = {"all": gain(ids, a, b, rng),
              "random": gain([i for i in ids if i in SPLITS["random"]], a, b, rng),
              "held-out apps": gain([i for i in ids if i not in SPLITS["random"]], a, b, rng)}
    turns = [t for i in ids for t in a[i].get("vote_trace", [])]
    agreed = [i for i in ids if all(len({json.dumps(k) for k in t["keys"]}) == 1 for t in a[i].get("vote_trace", []))]
    drift = [i for i in agreed if a[i]["pred"] != b[i]["pred"]]
    outvoted = [i for i in ids if any(t["keys"][t["chosen"]] != t["keys"][0] for t in a[i].get("vote_trace", []))]
    report["votes"] = {"turns": len(turns),
                       "all agree": sum(len({json.dumps(k) for k in t["keys"]}) == 1 for t in turns),
                       "soup outvoted": sum(t["keys"][t["chosen"]] != t["keys"][0] for t in turns),
                       "no majority": sum(max(sum(json.dumps(x) == json.dumps(k) for x in t["keys"]) for k in t["keys"]) == 1
                                          for t in turns),
                       "samples with an outvote": len(outvoted),
                       "points on those samples": round(100 * sum(a[i]["score"] - b[i]["score"] for i in outvoted) / len(ids), 2)}
    report["code check"] = {"samples agreeing at every turn": len(agreed), "of them with calls unlike the soup": len(drift),
                            "ids": drift[:10]}
    asks = lambda rows: sum(r["stop"] == "asked" for i, r in rows.items() if i in ids)  # noqa: E731
    report["asks"] = {"vote": asks(a), "soup": asks(b)}
    for name, rows in (("vote", a), ("soup", b)):
        secs = sorted(rows[i]["seconds"] for i in ids)
        report[f"seconds {name}"] = {"mean": round(sum(secs) / len(secs), 2), "p95": secs[int(len(secs) * .95)],
                                     "max": secs[-1], "over 25 s": sum(s > 25 for s in secs), "over 30 s": sum(s > 30 for s in secs)}
    checks = {f"mean gain >= +{GAIN_MIN}": report["all"]["gain"] >= GAIN_MIN,
              f"random half >= {HALF_FLOOR}": report["random"]["gain"] >= HALF_FLOOR,
              f"held-out apps half >= {HALF_FLOOR}": report["held-out apps"]["gain"] >= HALF_FLOOR}
    report["checks"] = checks
    report["code ok"] = not drift
    report["decision"] = ("PASS" if all(checks.values()) else "FAIL") if not drift else "CODE CHECK FAILED: result does not count"
    print(json.dumps(report, indent=1, ensure_ascii=False))
    (run / "vote_check.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
