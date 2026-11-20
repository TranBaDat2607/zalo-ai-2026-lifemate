"""Inputs for the Gemini note writer: all 1,159 non-validation train samples, nothing carried over from the
Claude-written v1 notes. Same records, same shuffle (seed 2026), same 50-per-batch order and 10-per-part packing as
make_inputs_v2.py, written to notes_v2/gemini/inputs/batch_NNN_pK.txt (branch-blind for kinds K1-K3).

    LIFEMATE_DATA=$PWD/train+public_test/train+public_test python scripts/notes/gemini_inputs.py
"""
import json
import random

from make_inputs_v2 import BATCH, PART, SEED, as_text, record
from notes_common import REPO, load_train

OUT = REPO / "notes_v2" / "gemini" / "inputs"


def main() -> None:
    items, catalog, validation = load_train()
    pool = [i for i in items if i["id"] not in validation]
    assert len(validation) == 386 and len(pool) == 1159, (len(validation), len(pool))
    random.Random(SEED).shuffle(pool)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("batch_*"):
        old.unlink()
    order = []
    for b in range(0, len(pool), BATCH):
        part = pool[b:b + BATCH]
        recs = [record(item, catalog)[0] for item in part]
        for p in range(0, len(recs), PART):
            (OUT / f"batch_{b // BATCH:03d}_p{p // PART}.txt").write_text(
                "\n".join(as_text(r) for r in recs[p:p + PART]), encoding="utf-8")
        order.append([i["id"] for i in part])
    assert order == json.loads((REPO / "notes_v2" / "inputs" / "order.json").read_text()), "order differs from v2"
    (OUT / "order.json").write_text(json.dumps(order, indent=0), encoding="utf-8")
    leaks = [(p.name, v) for p in OUT.glob("*") for v in validation if v in p.read_text(encoding="utf-8")]
    assert not leaks, leaks[:5]
    parts = sorted(OUT.glob("batch_*_p*.txt"))
    print(f"{sum(map(len, order))} samples, {len(order)} batches, {len(parts)} parts; no validation id in any file")


if __name__ == "__main__":
    main()
