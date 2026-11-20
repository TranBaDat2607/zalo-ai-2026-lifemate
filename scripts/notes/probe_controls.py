"""Leak probe controls for the Gemini v2 notes (post hoc, after the probe gave +0.030 / +0.031 against a 0.02 bar).

    LIFEMATE_DATA=... python scripts/notes/probe_controls.py

Same tasks, folds and model as leak_probe.py (K1 asks-vs-acts, K2 writes-vs-read-only; request+history baseline; 5x5
repeated stratified CV). The "note" is swapped for:
- gemini: the Gemini notes (src/lifemate/notes_v2.jsonl), the reference;
- C1 writer input: the branch-blind writer input minus the request and history (already in the baseline): time, profile,
  apps, the calls up to the deciding read, the tool list, must_include. Outcome-free by construction;
- C2 shown calls: only the calls up to the deciding read and must_include (for K1 this names the decision tool and parameter);
- M masked: the Gemini notes with tool names and snake_case parameter names replaced by placeholders.
If C1/C2 add as much as the Gemini notes, the notes carry no more than what the writer was shown without the outcome.
"""
import json
import re
from pathlib import Path

from gemini_notes import blocks
from leak_probe import analyze_all, history_lines, run_task, tasks

REPO = Path(__file__).resolve().parents[2]


def strip_request_history(block: str, compact: bool) -> str:
    keep, section = [], None
    for line in block.splitlines()[1:]:
        if not line.startswith("  "):
            section = line.split(":")[0]
        if section in ("history", "request"):
            continue
        if compact and not (section.startswith("gold_calls") or section == "must_include"):
            continue
        keep.append(line.strip())
    return " ".join(keep)


def main() -> None:
    pool, catalog, infos = analyze_all()
    base_text = {it["id"]: it["audio_text"] + " || " + " ".join(history_lines(it)) for it in pool}
    gem = {json.loads(x)["id"]: json.loads(x)["note"] for x in (REPO / "src/lifemate/notes_v2.jsonl").read_text().splitlines()}
    inputs = {}
    for p in sorted((REPO / "notes_v2/gemini/inputs").glob("batch_*_p*.txt")):
        inputs.update(blocks(p))
    mask = lambda n: re.sub(r"\b[a-z]+(?:_[a-z]+)+\b", "param", re.sub(r"\b[a-z_]+\.[a-z_]+\b", "tool", n))
    sets = {"gemini": gem,
            "C1 writer input": {i: strip_request_history(b, False) for i, b in inputs.items()},
            "C2 shown calls": {i: strip_request_history(b, True) for i, b in inputs.items()},
            "M masked gemini": {i: mask(n) for i, n in gem.items()}}
    out = {}
    for tname, (ids, y) in tasks(pool, infos).items():
        keep = [(i, l) for i, l in zip(ids, y) if i in gem]  # the same samples for every set
        ids2, y2 = [i for i, _ in keep], [l for _, l in keep]
        for name, notes in sets.items():
            r = run_task(ids2, y2, base_text, notes)
            out[f"{name}|{tname}"] = r
            print(f"{tname:24s} {name:16s} n={r['n']} base {r['auc_request_history']:.3f} +note {r['auc_with_note']:.3f} "
                  f"gain {r['gain']:+.3f} (se {r['gain_se_repeats']:.3f})  note alone {r['auc_note_only']:.3f}", flush=True)
    (REPO / "notes_v2/gemini/probe_controls.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
