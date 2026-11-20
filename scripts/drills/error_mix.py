"""Error-driven drill mix (log.md 12 Nov, the user's rule): each value category gets drills in proportion to the
values the model got wrong, measured out of fold (never on the fold the drills are for), with a 5 % floor.

    python scripts/drills/error_mix.py --exclude cv:1 --out mix_fold1.json outputs/notes-aux/aux_eval outputs/cv2-aux/aux_eval ...

A value counts when the prediction has a call of the gold call's tool; wrong = no such call carries the gold value.
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from compare_runs import load  # noqa: E402
from first_error import arg_kind  # noqa: E402
from lifemate import data  # noqa: E402

FLOOR = 0.05


def category(name, key, value, catalog):
    kind = arg_kind(name, key, value, catalog)
    if kind == "id":
        return "id"
    if kind == "date/time":
        v = str(value)
        return "hh:mm" if re.fullmatch(r"\d\d:\d\d", v) else "date" if len(v) == 10 else "datetime"
    if kind == "number":
        return "money" if key == "amount" else "minutes" if key in ("minutes", "delay_minutes", "hours") else "number"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default=None, help="holdout spec (e.g. cv:1) whose samples are left out: the fold the "
                                                     "drills are for, where an earlier run's holdout overlaps it")
    args = ap.parse_args()
    skip = data.id_set(args.exclude) if args.exclude else set()
    catalog = data.load_catalog("train", data.find_root("train"))
    seen, wrong = collections.Counter(), collections.Counter()
    for run in args.runs:
        for sid, r in load(run).items():
            if sid in skip:
                continue
            for g in r["gold"]:
                same = [p for p in r["pred"] if p["name"] == g["name"]]
                if not same:
                    continue
                for k, v in (g.get("arguments") or {}).items():
                    cat = category(g["name"], k, v, catalog)
                    if cat:
                        seen[cat] += 1
                        wrong[cat] += not any((p.get("arguments") or {}).get(k) == v for p in same)
    cats = ["datetime", "date", "hh:mm", "minutes", "money", "number", "id"]
    raw = {c: wrong[c] / max(1, sum(wrong.values())) for c in cats}
    w = {c: max(FLOOR, raw[c]) for c in cats}
    total = sum(w.values())
    w = {c: round(v / total, 4) for c, v in w.items()}
    out = {"runs": args.runs, "excluded": args.exclude, "seen": dict(seen), "wrong": dict(wrong), "weights": w}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
