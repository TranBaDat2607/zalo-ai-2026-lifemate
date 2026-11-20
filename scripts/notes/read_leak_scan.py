"""Second leak check: numbers and times a note states that a tool result of the final turn holds, but that neither the
request nor the history shows. The checker's arithmetic allowance (a × 1000, a + b, ...) can explain such a value by
chance, so these are listed for a look by hand. Small numbers (12 and below) are skipped.

    python scripts/notes/read_leak_scan.py            # every note as it stands (parts, then fix files)
"""
import json

from check_notes import NOTES, _DM, _HH, _HM, _ID, _ISO_DATE, _TOOL, note_numbers, numbers_in, times_in, world
from notes_common import classify, results_datetimes, results_numbers, said_times, spoken_numbers


def scan(sid: str, note: str) -> list:
    by_id, catalog = world()[0], world()[1]
    item = by_id[sid]
    vis = classify(item, catalog)["visible"]
    calls = item["label"]["tool_calls"]
    # results of read calls only: a write's result echoes the arguments the model chose, which are not leaks
    results = [r for c, r in zip(calls, item["label"]["tool_results"]) if catalog.get(c["name"], {}).get("kind") == "read"]
    shown_nums = numbers_in(vis) | spoken_numbers(vis)
    shown_times = times_in(vis) | said_times(item)
    read_nums = results_numbers(results)
    read_times = {(d.hour, d.minute) for d in results_datetimes(results)}
    rest = _TOOL.sub(" ", note)
    rest = _ISO_DATE.sub(" ", rest)
    rest = _DM.sub(" ", rest)
    times = {(int(h), int(m)) for h, m in _HM.findall(rest)}
    rest = _HH.sub(" ", _HM.sub(" ", rest))
    rest = _ID.sub(" ", rest)
    out = []
    for n in sorted(note_numbers(rest) | numbers_in(rest)):
        try:
            v = float(n)
        except ValueError:
            continue
        if v > 12 and n in read_nums and n not in shown_nums:
            out.append(n)
    out += [f"{h}:{m:02d}" for h, m in sorted(times) if (h, m) in read_times and (h, m) not in shown_times]
    return out


def latest_notes() -> dict:
    """Every note as it stands: the writer's parts, then its fix files in order (a later fix wins)."""
    notes = {}
    parts = NOTES / "parts"
    for path in sorted(parts.glob("batch_*_p*.jsonl")) + sorted(parts.glob("batch_*_fix*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                notes[rec["id"]] = rec["note"]
    return notes


def main() -> None:
    hits = 0
    for sid, note in sorted(latest_notes().items()):
        found = scan(sid, note)
        if found:
            hits += 1
            print(f"{sid}: {found} | {note}")
    print(f"{hits} notes to look at")


if __name__ == "__main__":
    main()
