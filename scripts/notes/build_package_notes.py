"""Join the checked batch files into src/lifemate/notes.jsonl, the file `train --notes package` reads.

    python scripts/notes/build_package_notes.py

Every note is checked again here. The script fails if a note does not pass, if any validation id appears (as a
record id or anywhere in the text of any note file), or if a batch file is missing.
"""
import json

from check_notes import NOTES, check_note, tokenizer, world

OUT = NOTES.parents[0] / "src" / "lifemate" / "notes.jsonl"


def main() -> None:
    validation = world()[2]
    order = json.loads((NOTES / "inputs" / "order.json").read_text(encoding="utf-8"))
    notes, words, tokens = {}, [], []
    for b in range(len(order)):
        path = NOTES / f"batch_{b:03d}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing {path.name}")
        raw = path.read_text(encoding="utf-8")
        leaked = [v for v in validation if v in raw]
        if leaked:
            raise SystemExit(f"validation ids in {path.name}: {leaked[:5]}")
        for line in raw.splitlines():
            rec = json.loads(line)
            if rec["id"] in validation or rec["id"] in notes or rec["id"] not in order[b]:
                raise SystemExit(f"bad id {rec['id']} in {path.name}")
            problems = check_note(rec["id"], rec["note"], b)
            if problems:
                raise SystemExit(f"{rec['id']} fails: {problems}")
            notes[rec["id"]] = rec["note"]
            words.append(len(rec["note"].split()))
            tokens.append(len(tokenizer()(rec["note"], add_special_tokens=False)["input_ids"]))
    OUT.write_text("".join(json.dumps({"id": i, "note": notes[i]}, ensure_ascii=False) + "\n" for i in sorted(notes)),
                   encoding="utf-8")
    total = sum(len(ids) for ids in order)
    print(json.dumps({"notes": len(notes), "train samples without validation": total, "dropped": total - len(notes),
                      "mean words": round(sum(words) / len(words), 1), "max words": max(words),
                      "mean tokens": round(sum(tokens) / len(tokens), 1), "max tokens": max(tokens),
                      "validation ids found": 0, "file": str(OUT)}))


if __name__ == "__main__":
    main()
