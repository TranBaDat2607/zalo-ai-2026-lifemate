"""Step 4 screen (improve_plan.md; rule in log.md, fixed before the run): 0012's recipe plus the time/number drills
(train --drills drills_fold0.jsonl) against 0012 itself (outputs/notes-aux/aux_eval, 71.57), same 386 samples, same
answer-time settings (no lookup, budget 60 s, --denoise).

    LIFEMATE_DATA=$PWD/train+public_test/train+public_test python scripts/review/drills_check.py outputs/drills-fold0/drills_eval

KEEP when all hold:
- the target moves: first errors on a number or date/time argument (scripts/first_error.py classes) fall to <= 10,
  or gold calls with a `minutes` argument done right (same tool, same minutes) reach >= 22;
- mean >= 0012 - 0.5; neither half (random / held-out apps) more than 1.0 below 0012;
- repairs rise by <= 2 and asks by <= 10.
Otherwise DROP, no second try. A KEEP goes on to the four-fold test (folds 1-3 with per-fold drills).
"""
import collections
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from compare_runs import SPLITS, interval, load  # noqa: E402
from first_error import classify  # noqa: E402
from lifemate import data, scorer  # noqa: E402

TARGET_MAX, MINUTES_MIN, MEAN_FLOOR, HALF_FLOOR, REPAIRS_MAX, ASKS_MAX = 10, 22, -0.5, -1.0, 2, 10


def numeric_first_errors(rows: dict, catalog: dict) -> list:
    out = []
    for sid, row in sorted(rows.items()):
        if row["score"] > 0.999:
            continue
        label, _ = classify(row, catalog)
        if label.startswith("wrong arguments") and ("number" in label or "date/time" in label):
            out.append(sid)
    return out


def minutes_right(rows: dict, catalog: dict) -> tuple:
    right = total = 0
    for row in rows.values():
        for g in row["gold"]:
            if "minutes" not in (g.get("arguments") or {}):
                continue
            total += 1
            right += any(p["name"] == g["name"] and (p.get("arguments") or {}).get("minutes") == g["arguments"]["minutes"]
                         for p in row["pred"])
    return right, total


def repairs(rows: dict) -> int:
    return sum(len(r.get("repairs") or []) if isinstance(r.get("repairs"), list) else int(r.get("repairs") or 0)
               for r in rows.values())


def gain(ids, a, b, rng):
    if not ids:  # e.g. a cv fold has none of fold 0's random half
        return {"n": 0, "drills": None, "0012": None, "gain": 0.0, "ci95": [0.0, 0.0]}
    diffs = [a[i]["score"] - b[i]["score"] for i in ids]
    low, high = interval(diffs, 20000, rng)
    return {"n": len(ids), "drills": round(100 * sum(a[i]["score"] for i in ids) / len(ids), 2),
            "0012": round(100 * sum(b[i]["score"] for i in ids) / len(ids), 2),
            "gain": round(100 * sum(diffs) / len(ids), 2), "ci95": [round(100 * low, 2), round(100 * high, 2)]}


def main() -> None:
    run = Path(sys.argv[1])
    base = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "outputs/notes-aux/aux_eval"
    catalog = data.load_catalog("train", data.find_root("train"))
    a, b = load(str(run)), load(str(base))
    ids = sorted(set(a) & set(b))
    a, b = {i: a[i] for i in ids}, {i: b[i] for i in ids}
    rng = random.Random(2026)
    report = {"all": gain(ids, a, b, rng),
              "random": gain([i for i in ids if i in SPLITS["random"]], a, b, rng),
              "held-out apps": gain([i for i in ids if i not in SPLITS["random"]], a, b, rng)}
    na, nb = numeric_first_errors(a, catalog), numeric_first_errors(b, catalog)
    ma, mb = minutes_right(a, catalog), minutes_right(b, catalog)
    report["number/date first errors"] = {"drills": len(na), "0012": len(nb), "fixed": sorted(set(nb) - set(na)),
                                          "new": sorted(set(na) - set(nb))}
    report["minutes right"] = {"drills": f"{ma[0]}/{ma[1]}", "0012": f"{mb[0]}/{mb[1]}"}
    asks = lambda rows: sum(r["stop"] == "asked" for r in rows.values())  # noqa: E731
    report["asks"] = {"drills": asks(a), "0012": asks(b)}
    report["repairs"] = {"drills": repairs(a), "0012": repairs(b)}
    first = collections.Counter(classify(r, catalog)[0].split(":")[0] for r in a.values() if r["score"] <= 0.999)
    report["first errors (drills)"] = dict(first.most_common(12))
    checks = {f"target moved (number/date first errors <= {TARGET_MAX} or minutes right >= {MINUTES_MIN})":
              len(na) <= TARGET_MAX or ma[0] >= MINUTES_MIN,
              f"mean >= 0012 {MEAN_FLOOR}": report["all"]["gain"] >= MEAN_FLOOR,
              f"random half >= {HALF_FLOOR}": report["random"]["gain"] >= HALF_FLOOR,
              f"held-out apps half >= {HALF_FLOOR}": report["held-out apps"]["gain"] >= HALF_FLOOR,
              f"repairs + <= {REPAIRS_MAX}": report["repairs"]["drills"] - report["repairs"]["0012"] <= REPAIRS_MAX,
              f"asks + <= {ASKS_MAX}": report["asks"]["drills"] - report["asks"]["0012"] <= ASKS_MAX}
    report["checks"] = checks
    report["decision"] = "KEEP" if all(checks.values()) else "DROP"
    print(json.dumps(report, indent=1, ensure_ascii=False))
    (run / "drills_check.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
