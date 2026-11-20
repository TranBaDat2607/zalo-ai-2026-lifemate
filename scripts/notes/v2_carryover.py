"""Step 5: how many v1 notes survive under v2. A v1 note is kept unchanged only if its sample is K0 and it passes v2.

    python scripts/notes/v2_carryover.py            # prints counts; writes notes_v2/carryover.json (ids kept)
"""
import collections
import json

import check_notes as cn
from check_notes_v2 import NOTES2, _analysis, check_note_v2

def main():
    info = _analysis()
    v1 = {}
    for p in sorted(cn.NOTES.glob("batch_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line); v1[r["id"]] = r["note"]
    order = json.loads((NOTES2 / "inputs" / "order.json").read_text())
    batch = {i: b for b, ids in enumerate(order) for i in ids}
    kept, why = [], collections.Counter()
    fails = collections.Counter()
    for sid, note in v1.items():
        k = info[sid]["primary"]
        probs = check_note_v2(sid, note, batch[sid])
        if k != "K0":
            why["not K0 -> rewrite"] += 1
        elif probs:
            why["K0 but fails v2 -> rewrite"] += 1
            for p in probs: fails[p.split(" ")[0] + " " + p.split(" ")[1]] += 1
        else:
            kept.append(sid); why["K0 and passes -> kept"] += 1
    byk = collections.Counter(info[s]["primary"] for s in v1)
    print(dict(why)); print("v1 notes by kind", dict(byk)); print("top v2 failures on K0", fails.most_common(12))
    (NOTES2 / "carryover.json").write_text(json.dumps(kept))
    print("kept", len(kept), "of", len(v1), "; to write", len(info) - len(kept))
main()
