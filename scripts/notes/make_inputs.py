"""Write the note writer's inputs: notes/inputs/batch_NNN.jsonl, 50 non-validation train samples each.

    python scripts/notes/make_inputs.py

The 386 validation samples (src/lifemate/splits.json) are removed before anything is written, and the script checks
the files afterwards. Samples are shuffled with a fixed seed so every batch (and the 20-sample pilot, the first 20 of
batch 000) mixes the three generator batches. Values that only a tool result of the final turn reveals are masked.
"""
import json
import random
from datetime import datetime

from notes_common import REPO, WEEKDAYS, classify, load_train, must_strings, prompts

OUT = REPO / "notes" / "inputs"
BATCH = 50
PART = 10
SEED = 2026


def compact(value, limit: int = 1800) -> str:
    text = prompts.dumps(value)
    return text if len(text) <= limit else text[:limit] + "…"


def history_lines(item: dict) -> list:
    out = []
    for t in item["history"]:
        if t["role"] == "user":
            out.append(f"Người dùng: {t['audio_text']}")
            continue
        calls, results = t.get("tool_calls") or [], t.get("tool_results") or []
        for i, call in enumerate(calls):
            res = compact(results[i]) if i < len(results) else "?"
            out.append(f"Trợ lý gọi {call['name']} {prompts.dumps(call.get('arguments') or {})} → {res}")
        out.append(f"Trợ lý: {t.get('text') or prompts.DONE}")
    return out


def record(item: dict, catalog: dict) -> dict:
    facts = classify(item, catalog)
    now = datetime.fromisoformat(item["context"]["current_time"])
    calls = item["label"]["tool_calls"]
    tools = []
    for call in calls:
        for name in (call["name"], (call.get("arguments") or {}).get("tool_name") if call["name"] == "sys.ask_missing_info" else None):
            if name and name in catalog and name != "sys.ask_missing_info":
                line = prompts.tool_line(catalog[name])
                if line not in tools:
                    tools.append(line)
    if not calls:
        kind = "nothing"
    elif calls[-1]["name"] == "sys.ask_missing_info":
        kind = "ask"
    elif all(catalog.get(c["name"], {}).get("kind") == "read" for c in calls):
        kind = "read_only"
    else:
        kind = "act"
    profile = {k: v for k, v in (item["context"].get("user_profile") or {}).items()}
    return {
        "id": item["id"],
        "now": f"{now:%Y-%m-%d %H:%M}, {WEEKDAYS[now.weekday()]}",
        "days": prompts.calendar_hint(item["context"]["current_time"]),
        "profile": profile,
        "apps": sorted(item["context"]["installed_apps"]),
        "history": history_lines(item),
        "request": item["audio_text"],
        "answer_type": kind,
        "gold_calls": [f"{c['name']} {prompts.dumps(c['arguments'])}" for c in facts["shown_calls"]],
        "tools": tools,
        "must_include": must_strings(facts["required"]),
        "may_depend_on_reads": [d for _, d, _ in facts["optional"]],
    }


def as_text(rec: dict) -> str:
    lines = [f"=== {rec['id']} ===", f"now: {rec['now']}", f"days: {rec['days']}",
             f"profile: {prompts.dumps(rec['profile'])}", f"apps: {', '.join(rec['apps'])}", "history:"]
    lines += [f"  {h}" for h in rec["history"]] or ["  (none)"]
    lines += [f"request: {rec['request']}", f"answer_type: {rec['answer_type']}", "gold_calls:"]
    lines += [f"  {c}" for c in rec["gold_calls"]] or ["  (none: gold calls no tool)"]
    lines += ["tools:"] + [f"  {t}" for t in rec["tools"]]
    lines += [f"must_include: {json.dumps(rec['must_include'], ensure_ascii=False)}",
              f"may_depend_on_reads: {json.dumps(rec['may_depend_on_reads'], ensure_ascii=False)}", ""]
    return "\n".join(lines)


def main() -> None:
    items, catalog, validation = load_train()
    pool = [i for i in items if i["id"] not in validation]
    assert len(validation) == 386 and len(pool) == len(items) - 386, (len(validation), len(pool))
    random.Random(SEED).shuffle(pool)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("batch_*.jsonl"):
        old.unlink()
    for old in OUT.glob("batch_*.txt"):
        old.unlink()
    order = []
    for b in range(0, len(pool), BATCH):
        part = pool[b:b + BATCH]
        path = OUT / f"batch_{b // BATCH:03d}.jsonl"
        recs = [record(item, catalog) for item in part]
        with open(path, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for p in range(0, len(recs), PART):  # the writer reads these: short lines, 10 samples per file
            (OUT / f"batch_{b // BATCH:03d}_p{p // PART}.txt").write_text(
                "\n".join(as_text(r) for r in recs[p:p + PART]), encoding="utf-8")
        order.append([i["id"] for i in part])
    (OUT / "order.json").write_text(json.dumps(order, indent=0), encoding="utf-8")
    # independent check: no validation id anywhere in the inputs, not even inside a text
    leaks = []
    for path in sorted(OUT.glob("*.json*")):
        text = path.read_text(encoding="utf-8")
        leaks += [(path.name, v) for v in validation if v in text]
    assert not leaks, leaks[:5]
    sizes = [len(line) for p in OUT.glob("batch_*.jsonl") for line in p.read_text(encoding="utf-8").splitlines()]
    print(f"{len(pool)} samples in {len(order)} batches; no validation id in any input file; "
          f"record chars mean {sum(sizes) / len(sizes):.0f}, max {max(sizes)}")


if __name__ == "__main__":
    main()
