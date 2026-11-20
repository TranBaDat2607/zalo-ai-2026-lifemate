"""Split the training samples outside the fixed validation set into 3 folds for the step-preference plan.

    LIFEMATE_DATA=$PWD/train+public_test/train+public_test python scripts/make_dpo_folds.py

Fold k's model trains on everything except fold k (the validation set included) and then runs on fold k, so its
mistakes there are genuine. Stratified by batch (A/B/C) and by the number of gold calls, seed 2026. Writes
src/lifemate/folds_dpo.json ({"1": [...], "2": [...], "3": [...]}), read by data.id_set("fold:K").
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from lifemate import data  # noqa: E402


def batch(sid: str) -> str:
    n = int(sid.split("_")[1])
    return "A" if n < 425 else ("B" if n < 919 else "C")


def main() -> None:
    root = data.find_root("train")
    holdout = data.id_set("splits")
    items = [i for i in data.load_split("train", root) if i["id"] not in holdout]
    strata = defaultdict(list)
    for item in items:
        strata[(batch(item["id"]), min(len(item["label"]["tool_calls"]), 4))].append(item["id"])
    rng = random.Random(2026)
    folds = {"1": [], "2": [], "3": []}
    offset = 0
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        for n, sid in enumerate(ids):
            folds[str((n + offset) % 3 + 1)].append(sid)
        offset += len(ids)  # carry the round-robin position so small strata do not all start at fold 1
    for k in folds:
        folds[k].sort()
    out = REPO / "src" / "lifemate" / "folds_dpo.json"
    out.write_text(json.dumps(folds, indent=0) + "\n", encoding="utf-8")
    print(out, {k: len(v) for k, v in folds.items()}, "of", len(items))


if __name__ == "__main__":
    main()
