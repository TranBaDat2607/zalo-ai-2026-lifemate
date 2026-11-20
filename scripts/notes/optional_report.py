"""List notes that state a value which may depend on a read of the final turn (to review by hand, or send back).

    python scripts/notes/optional_report.py 0 1
"""
import json
import sys

from check_notes import NOTES, world, _DM, _HM, _ISO_DATE, note_numbers, numbers_in, times_in
from notes_common import classify, said_times, spoken_numbers


def stated_optional(sid: str, note: str) -> list:
    """Optional values the note states that the request and history do not show: these need a look by hand.
    A value that is also shown (said, or in an earlier turn) is fine to state and is not listed."""
    by_id, catalog = world()[0], world()[1]
    item = by_id[sid]
    facts = classify(item, catalog)
    vis = facts["visible"]
    req = {json.dumps(c, sort_keys=True) if not isinstance(c, str) else c for _, _, c in facts["required"]}
    req_times = {c for kind, _, c in facts["required"] if kind == "time"}
    shown_nums = numbers_in(vis) | spoken_numbers(vis)
    shown_times = times_in(vis) | said_times(item)
    nums = numbers_in(note) | note_numbers(note)
    times = {(int(h), int(m)) for h, m in _HM.findall(note)}
    dates = {(int(mo), int(d)) for d, mo, _ in _DM.findall(note)} | {(int(mo), int(d)) for _, mo, d in _ISO_DATE.findall(note)}
    out = []
    for kind, display, c in facts["optional"]:
        if kind == "number" and c in nums and c not in req and c not in shown_nums:
            out.append(display)
        elif kind == "datetime" and c[2:] in times and c[2:] not in req_times and c[2:] not in shown_times:
            out.append(display)
        elif kind == "time" and c in times and c not in shown_times:
            out.append(display)
        elif kind == "date" and c in dates and json.dumps(c) not in req:
            out.append(display)
    return out


if __name__ == "__main__":
    for b in map(int, sys.argv[1:]):
        for line in ((NOTES / f"batch_{b:03d}.jsonl").read_text(encoding="utf-8").splitlines() if (NOTES / f"batch_{b:03d}.jsonl").exists() else []):
            rec = json.loads(line)
            found = stated_optional(rec["id"], rec["note"])
            if found:
                print(f"{rec['id']}: {found} | {rec['note']}")
