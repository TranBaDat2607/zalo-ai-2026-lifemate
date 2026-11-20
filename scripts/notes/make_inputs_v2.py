"""Write the v2 note writer's inputs: notes_v2/inputs/batch_NNN_pK.txt (10 samples each, 50 per batch).

    python scripts/notes/make_inputs_v2.py

Same 1,159 non-validation train samples and the same shuffle (seed 2026) as v1, so batch NNN holds the same ids as in v1.
Samples whose v1 note is kept unchanged (notes_v2/carryover.json, from v2_carryover.py) get no input: only the samples to
be written are in the parts (10 per part, re-packed per batch).
Kinds K0 and K4 are written as in v1 (full gold calls, with values that only a read reveals masked). Kinds K1, K2 and K3 are
BRANCH-BLIND: the writer sees the gold calls only up to and including the deciding read(s), and one neutral line in place of
everything after, identical whether the key goes on to act, ask or stop. `must_include` for those comes from the request,
the history and the read part only. A K1-only sample also names the write tool and the parameter the read must supply.
A K2/K3 sample lists the installed tools instead (the writer has to name the branch tools itself).

The metadata (kind, branch, cut) goes to notes_v2/meta.json, which the writers never read. No validation id may appear in
any input file; the script asserts it.
"""
import json
import random
from datetime import datetime

from make_inputs import compact, history_lines
from notes_common import REPO, WEEKDAYS, classify, load_train, must_strings, prompts
from notes_v2_common import analyze, blind_kinds

OUT = REPO / "notes_v2" / "inputs"
META = REPO / "notes_v2" / "meta.json"
CARRY = REPO / "notes_v2" / "carryover.json"
BATCH, PART, SEED = 50, 10, 2026
KIND_TEXT = {
    "K0": "K0 explicit: no read of this turn decides anything; write the plan as in the guide",
    "K1": "K1 pick one item: a read must identify one item; rule with both branches (one match / none or several)",
    "K2": "K2 condition on a read: the request has a condition a read decides; rule with both branches",
    "K3": "K3 select by a rule: a read lists items and the action applies to those matching a rule; no counts, no IDs",
    "K4": "K4 value lookup: a read only supplies a value for an action that happens anyway; read, then act with that value or formula",
}
NEUTRAL = "(the rest of the answer depends on what the read returns; you are not shown it)"


def truncated(item: dict, cut: int) -> dict:
    t = json.loads(json.dumps(item))
    t["label"]["tool_calls"] = item["label"]["tool_calls"][:cut]
    t["label"]["tool_results"] = item["label"]["tool_results"][:cut]
    return t


def short_tool(tool: dict) -> str:
    params = tool.get("parameters") or {}
    req = set(params.get("required") or [])
    args = ", ".join(n + ("*" if n in req else "") for n in (params.get("properties") or {}))
    return f"{tool['name']}({args}) - {tool.get('description', '')[:90]}"


def record(item: dict, catalog: dict) -> tuple:
    info = analyze(item, catalog)
    blind = blind_kinds(info)
    calls = item["label"]["tool_calls"]
    now = datetime.fromisoformat(item["context"]["current_time"])
    src = truncated(item, info["cut"]) if blind else item
    facts = classify(src, catalog)
    tools, lines = [], []
    if blind:
        # the installed tools (names and descriptions) are all the writer has to name the branch tools
        for call in src["label"]["tool_calls"]:
            if call["name"] in catalog:
                lines.append(prompts.tool_line(catalog[call["name"]]))
        names = [t for app in item["context"]["installed_apps"].values() for t in app["tools"]]
        known = {c["name"] for c in src["label"]["tool_calls"]}
        if "K2" in info["kinds"] or "K3" in info["kinds"] or info["decision"] is None:
            tools = [short_tool(catalog[n]) for n in names if n in catalog and n not in known and n != "sys.ask_missing_info"]
        else:
            dec_tool = info["decision"][0]
            tools = [prompts.tool_line(catalog[dec_tool])] if dec_tool in catalog else []
        tools = lines + [t for t in tools if t not in lines]
        gold = [f"{c['name']} {prompts.dumps(c['arguments'])}" for c in facts["shown_calls"]]
        gold.append(NEUTRAL)
        if "K2" not in info["kinds"] and "K3" not in info["kinds"] and info["decision"]:
            gold.append(f"decision tool: {info['decision'][0]}; parameter the read must supply: {info['decision'][1]}")
        must = must_strings(facts["required"])
        if info["decision"] and "K2" not in info["kinds"] and "K3" not in info["kinds"]:
            must += [x for x in (info["decision"][0], info["decision"][1], "sys.ask_missing_info") if x and x not in must]
        answer_type = None
    else:
        for call in calls:
            for name in (call["name"], (call.get("arguments") or {}).get("tool_name") if call["name"] == "sys.ask_missing_info" else None):
                if name and name in catalog and name != "sys.ask_missing_info":
                    line = prompts.tool_line(catalog[name])
                    if line not in tools:
                        tools.append(line)
        gold = [f"{c['name']} {prompts.dumps(c['arguments'])}" for c in facts["shown_calls"]]
        must = must_strings(facts["required"])
        if not calls:
            answer_type = "nothing"
        elif calls[-1]["name"] == "sys.ask_missing_info":
            answer_type = "ask"
        elif all(catalog.get(c["name"], {}).get("kind") == "read" for c in calls):
            answer_type = "read_only"
        else:
            answer_type = "act"
    kind = info["primary"]
    rec = {
        "id": item["id"], "kind": KIND_TEXT[kind], "now": f"{now:%Y-%m-%d %H:%M}, {WEEKDAYS[now.weekday()]}",
        "days": prompts.calendar_hint(item["context"]["current_time"]),
        "profile": item["context"].get("user_profile") or {}, "apps": sorted(item["context"]["installed_apps"]),
        "history": history_lines(item), "request": item["audio_text"], "answer_type": answer_type,
        "gold_calls": gold, "tools": tools, "must_include": must,
        "may_depend_on_reads": [d for _, d, _ in facts["optional"]],
        "installed_tools": bool(blind and ("K2" in info["kinds"] or "K3" in info["kinds"] or info["decision"] is None)),
    }
    return rec, info


def as_text(rec: dict) -> str:
    lines = [f"=== {rec['id']} ===", f"kind: {rec['kind']}", f"now: {rec['now']}", f"days: {rec['days']}",
             f"profile: {prompts.dumps(rec['profile'])}", f"apps: {', '.join(rec['apps'])}", "history:"]
    lines += [f"  {h}" for h in rec["history"]] or ["  (none)"]
    lines += [f"request: {rec['request']}"]
    if rec["answer_type"]:
        lines.append(f"answer_type: {rec['answer_type']}")
    lines.append("gold_calls (up to the read that decides):" if rec["installed_tools"] or "depends on what the read" in " ".join(rec["gold_calls"]) else "gold_calls:")
    lines += [f"  {c}" for c in rec["gold_calls"]] or ["  (none: gold calls no tool)"]
    lines.append("installed tools you may name after the read:" if rec["installed_tools"] else "tools:")
    lines += [f"  {t}" for t in rec["tools"]]
    lines += [f"must_include: {json.dumps(rec['must_include'], ensure_ascii=False)}",
              f"may_depend_on_reads: {json.dumps(rec['may_depend_on_reads'], ensure_ascii=False)}", ""]
    return "\n".join(lines)


def main() -> None:
    items, catalog, validation = load_train()
    pool = [i for i in items if i["id"] not in validation]
    assert len(validation) == 386 and len(pool) == len(items) - 386, (len(validation), len(pool))
    random.Random(SEED).shuffle(pool)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in list(OUT.glob("batch_*")) + list(OUT.glob("order.json")):
        old.unlink()
    order, meta = [], {}
    carried = set(json.loads(CARRY.read_text())) if CARRY.exists() else set()  # v1 notes kept as they are (K0, pass v2)
    for b in range(0, len(pool), BATCH):
        part = pool[b:b + BATCH]
        recs = []
        for item in part:
            rec, info = record(item, catalog)
            meta[item["id"]] = info
            if item["id"] not in carried:
                recs.append(rec)
        for p in range(0, len(recs), PART):
            (OUT / f"batch_{b // BATCH:03d}_p{p // PART}.txt").write_text(
                "\n".join(as_text(r) for r in recs[p:p + PART]), encoding="utf-8")
        (OUT / f"batch_{b // BATCH:03d}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
        order.append([i["id"] for i in part])
    (OUT / "order.json").write_text(json.dumps(order, indent=0), encoding="utf-8")
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=0), encoding="utf-8")
    leaks = []
    for path in sorted(OUT.glob("*")) + [META]:
        text = path.read_text(encoding="utf-8")
        leaks += [(path.name, v) for v in validation if v in text]
    assert not leaks, leaks[:5]
    sizes = [len(p.read_text(encoding="utf-8")) for p in OUT.glob("batch_*_p*.txt")]
    print(f"{len(pool)} samples in {len(order)} batches; no validation id in any input file; "
          f"part chars mean {sum(sizes) / len(sizes):.0f}, max {max(sizes)}")


if __name__ == "__main__":
    main()
