"""Assemble v2 batch files: notes_v2/batch_NNN.jsonl = carried-over v1 K0 notes + the writers' notes, in order.

    python scripts/notes/assemble_v2.py            # all batches whose parts exist: report; writes failures list
    python scripts/notes/assemble_v2.py --final    # also write batch files; notes that still fail are dropped (and listed)

A note comes from (in priority) notes_v2/parts/batch_NNN_fixK.jsonl (highest K wins), batch_NNN_pK.jsonl, or the v1 carry-over.
Failures go to notes_v2/checks/failures.json with their problems, which is what a fix writer is given.
"""
import argparse
import json
import re

import check_notes as cn
from check_notes_v2 import NOTES2, check_note_v2

PARTS = NOTES2 / "parts"


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    order = json.loads((NOTES2 / "inputs" / "order.json").read_text())
    carried = set(json.loads((NOTES2 / "carryover.json").read_text()))
    v1 = {}
    for p in sorted(cn.NOTES.glob("batch_*.jsonl")):
        for r in read_jsonl(p):
            v1[r["id"]] = r["note"]
    validation = cn.world()[2]
    stats = {"first_pass": 0, "first_pass_ok": 0, "final_ok": 0, "missing": [], "dropped": []}
    failures, final_notes = {}, {}
    for b, ids in enumerate(order):
        first, latest = {}, {}
        for p in sorted(PARTS.glob(f"batch_{b:03d}_p*.jsonl")):
            for r in read_jsonl(p):
                first.setdefault(r["id"], r["note"]); latest[r["id"]] = r["note"]
        fixes = sorted(PARTS.glob(f"batch_{b:03d}_fix*.jsonl"), key=lambda p: int(re.search(r"_fix(\d+)", p.name).group(1)))
        for p in fixes:
            for r in read_jsonl(p):
                latest[r["id"]] = r["note"]
        bad = [i for i in latest if i in validation or i not in ids]
        assert not bad, bad
        keep = []
        for sid in ids:
            if sid in carried:
                note = v1[sid]
            elif sid in latest:
                note = latest[sid]
                probs0 = check_note_v2(sid, first[sid], b) if sid in first else ["no first"]
                stats["first_pass"] += 1; stats["first_pass_ok"] += not probs0
            else:
                stats["missing"].append(sid); continue
            probs = check_note_v2(sid, note, b)
            if probs:
                failures[sid] = {"note": note, "problems": probs}
            else:
                keep.append(sid); final_notes[sid] = note
        stats["final_ok"] += len(keep)
        if args.final and not [s for s in ids if s not in latest and s not in carried]:
            out = NOTES2 / f"batch_{b:03d}.jsonl"
            out.write_text("".join(json.dumps({"id": s, "note": final_notes[s]}, ensure_ascii=False) + "\n" for s in keep), encoding="utf-8")
            stats["dropped"] += [s for s in ids if s not in keep]
    (NOTES2 / "checks").mkdir(exist_ok=True)
    (NOTES2 / "checks" / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({**stats, "missing": len(stats["missing"]), "failing_now": len(failures), "dropped": len(stats["dropped"])}))


if __name__ == "__main__":
    main()
