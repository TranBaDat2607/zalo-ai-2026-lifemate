"""Inputs for the second Gemini pass over branch-blind notes that miss a sub-task (independent review, 8 Nov 2026).

    python scripts/notes/redo_v2.py IDS.json      # writes notes_v2/gemini/inputs_r2/batch_NNN_r2.txt

The ids are the notes that fail check_notes_v2 after the 'every key tool named' and 'both branches thôi' rules. Each block
is the v2 input of make_inputs_v2.record, with one change: every sample lists all installed tools (with descriptions), so
the writer can name the tools of every sub-task. The key's later calls are still not shown: the full tool list would tell
the branch (answers that ask stop early), the installed list is the same either way. No validation id is written.
"""
import json
import sys

from make_inputs_v2 import as_text, record, short_tool
from notes_common import REPO, load_train

OUT = REPO / "notes_v2" / "gemini" / "inputs_r2"


def main() -> None:
    ids = set(json.loads(open(sys.argv[1]).read()))
    items, catalog, validation = load_train()
    by_id = {i["id"]: i for i in items}
    order = json.loads((REPO / "notes_v2" / "gemini" / "inputs" / "order.json").read_text())
    assert not ids & set(validation)
    OUT.mkdir(parents=True, exist_ok=True)
    for b, batch in enumerate(order):
        todo = [s for s in batch if s in ids]
        if not todo:
            continue
        blocks = []
        for s in todo:
            rec, info = record(by_id[s], catalog)
            item = by_id[s]
            names = [t for app in item["context"]["installed_apps"].values() for t in app["tools"]]
            extra = [short_tool(catalog[n]) for n in names if n in catalog and n != "sys.ask_missing_info"]
            have = {t.split("(")[0] for t in rec["tools"]}
            rec["tools"] = rec["tools"] + [t for t in extra if t.split("(")[0] not in have]
            rec["installed_tools"] = True
            blocks.append(as_text(rec))
        text = "\n".join(blocks)
        assert not [v for v in validation if v in text]
        (OUT / f"batch_{b:03d}_r2.txt").write_text(text, encoding="utf-8")
        print(b, len(todo))


if __name__ == "__main__":
    main()
