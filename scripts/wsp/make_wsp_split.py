"""Fixed discovery / confirmation halves of the 386 fold-0 samples for the weight-space edit search (11 Nov 2026).

Written before any edit was scored. "fold0" = the 386 (stratified by part and batch); "fold1" = folds_dpo fold 1 (387
random samples outside the 386, stratified by batch), the discovery data once the search model became the dpo-fold1
pair (trained without fold 1). Alternating after a seeded shuffle.

    python scripts/wsp/make_wsp_split.py   # -> scripts/wsp/wsp_split.json
"""
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lifemate import data  # noqa: E402


def batch(sid: str) -> str:
    n = int(sid.split("_")[1])
    return "A" if n < 425 else ("B" if n < 919 else "C")


def main() -> None:
    out = {}
    for name, parts in (("fold0", ("cvheld:0", "cvrandom:0")), ("fold1", ("fold:1",))):
        rng = random.Random(20261111)
        d1, d2 = [], []
        for part in parts:
            ids = data.id_set(part)
            for b in "ABC":
                group = sorted(i for i in ids if batch(i) == b)
                rng.shuffle(group)
                for k, sid in enumerate(group):
                    (d1 if k % 2 == 0 else d2).append(sid)
        assert not set(d1) & set(d2)
        out[name] = {"discover": sorted(d1), "confirm": sorted(d2)}
        print(name, len(d1), len(d2))
    path = REPO / "scripts" / "wsp" / "wsp_split.json"
    old = json.loads(path.read_text()) if path.exists() else None
    if old is not None and old != out:
        raise SystemExit("the saved split differs from the one this script makes")
    path.write_text(json.dumps(out, indent=0), encoding="utf-8")


if __name__ == "__main__":
    main()
