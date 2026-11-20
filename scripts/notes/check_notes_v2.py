"""Check v2 worked notes: the v1 checks with v2 `must_include`, plus the R1 shape and the R2 no-invented-details rules.

    python scripts/notes/check_notes_v2.py notes_v2/batch_000.jsonl [more]   # report, exit 1 if any note fails
    python scripts/notes/check_notes_v2.py --all
    python scripts/notes/check_notes_v2.py --v1 notes/batch_000.jsonl        # how the v1 notes fare under v2 (no files written)

What is checked on top of v1 (`check_notes.check_note`, run on the full gold):
- must_include of a branch-blind sample (kinds K1, K2, K3) comes only from what the writer was shown (the calls up to and
  including the first read). v1 'missing ...' complaints about values after that point are dropped.
- R1 shape. K1: the decision tool, its parameter and sys.ask_missing_info are named, with both branches. K2: both branches,
  and the first write tool of the key is named. K3: the first write tool and a selection word, no count.
  Branch-revealing phrases (BRANCH_PHRASES) are rejected in every note of a branch-blind sample, "chỉ đọc" everywhere.
- R2. The current time or date is not mentioned unless something else feeds it. A branch-blind note has no number that the
  request does not say (counts, IDs). A parameter name is allowed only when the key sets it for that tool (K0, K4), or, in
  a branch-blind note, when it is in a shown call or is the decision parameter. An option value (enum) of a named tool
  must be the key's value (K0, K4) or be absent (K1-K3), unless the request says it.
"""
import argparse
import json
import re
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import check_notes as cn
from notes_common import REPO, must_strings, said_times, spoken_numbers
from notes_v2_common import analyze, blind_kinds

NOTES2 = REPO / "notes_v2"

# phrases that tell the branch. Banned in every note of a branch-blind sample (kinds K1-K3)
BRANCH_PHRASES = [
    r"không xác định được", r"chỉ đọc", r"chỉ cần đọc", r"chỉ xem", r"không có nên", r"nên thôi", r"vẫn còn", r"đã có\b",
    r"đã đủ", r"đủ rồi", r"chưa đủ nên", r"chắc là", r"có lẽ", r"trùng tên", r"không tìm thấy", r"tìm thấy nên",
    r"có nhiều (?:gói|người|kết quả|mục|bài|liên hệ|chuyến|đơn|việc|cái)", r"có (?:hai|ba|2|3) ", r"chỉ có một",
    r"điều kiện (?:đúng|sai|không|chưa|đã)", r"đã hết hạn", r"hết hạn rồi", r"không mưa nên", r"có mưa nên", r"nên (?:hỏi|chỉ|không|huỷ|bỏ)",
    r"kết quả (?:cho thấy|là|sẽ)", r"sẽ (?:có|là|không)",
]
BANNED_EVERYWHERE = [r"chỉ đọc", r"không xác định được thì"]
ELSE = r"(?:không|còn|ngược lại|ngoài ra|thôi|khác|sai|chưa|nhiều|hoặc)"
BOTH_BRANCHES = re.compile(r"(?:thì|→).*?" + ELSE)
SELECT_WORDS = re.compile(r"(?<!\w)(từng|mỗi|mọi|tất cả|nào|những|các|hết)(?!\w)", re.I)
_WORD = re.compile(r"(?<![\w.@/-])([a-z][a-z_]{2,})(?![\w@])")


@lru_cache(maxsize=1)
def _analysis():
    by_id, catalog, validation, batch_of, params, apps = cn.world()
    return {i: analyze(item, catalog) for i, item in by_id.items() if i not in validation}


def duration_words(text: str) -> set:
    """Numbers hidden in spoken durations: 'nửa tiếng' = 30, 'tiếng rưỡi' = 90, 'một tiếng' = 60, 'hai tiếng' = 120."""
    low = text.lower()
    out = set()
    if re.search(r"nửa (?:tiếng|giờ)|30 phút|ba mươi phút", low):
        out.add("30")
    if re.search(r"(?:một|1) (?:tiếng|giờ)(?! rưỡi)", low):
        out.add("60")
    if re.search(r"tiếng rưỡi|giờ rưỡi", low):
        out |= {"90", "30"}
    if re.search(r"(?:hai|2) (?:tiếng|giờ)", low):
        out.add("120")
    if re.search(r"(?:một|1) phần tư|mười lăm phút", low):
        out.add("15")
    return out


def build_ctx(sample_id: str) -> dict:
    """Everything the extra checks need about one sample, from the data (the tests build the same dict by hand)."""
    by_id, catalog, validation, batch_of, params, apps = cn.world()
    item = by_id[sample_id]
    info = _analysis()[sample_id]
    calls = item["label"]["tool_calls"]
    blind = blind_kinds(info)
    shown = calls[:info["cut"]] if blind else calls
    allowed, gold_values, shown_values = {}, {}, set()
    for c in shown:
        if c["name"] != "sys.ask_missing_info":
            shown_values |= {str(v) for v in (c.get("arguments") or {}).values()}
        a = c.get("arguments") or {}
        if c["name"] == "sys.ask_missing_info":
            allowed.setdefault(a.get("tool_name"), set()).update(a.get("params") or [])
            continue
        allowed.setdefault(c["name"], set()).update(a)
    for c in calls:  # values the key sets (K0/K4 notes may name them; used for enum checks)
        for k, v in (c.get("arguments") or {}).items():
            if c["name"] != "sys.ask_missing_info":
                gold_values.setdefault(c["name"], {}).setdefault(k, set()).add(str(v))
    if blind and info["decision"]:
        allowed.setdefault(info["decision"][0], set()).add(info["decision"][1])
    first_write = next((c["name"] for c in calls if (catalog.get(c["name"]) or {}).get("kind") == "write"), None)
    params_of, enums_of = {}, {}
    for name, spec in catalog.items():
        props = (spec.get("parameters") or {}).get("properties") or {}
        params_of[name] = set(props)
        enums_of[name] = {p: {str(e) for e in (s.get("enum") or [])} for p, s in props.items() if s.get("enum")}
    texts = [item["audio_text"]] + [t["audio_text"] for t in item["history"] if t["role"] == "user"]
    now = datetime.fromisoformat(item["context"]["current_time"])
    facts = cn.classify(item, catalog)
    gold_t = {c[2:] for k, _, c in facts["required"] + facts["optional"] if k == "datetime"}
    gold_t |= {c for k, _, c in facts["required"] + facts["optional"] if k == "time"}
    gold_d = {c[:2] for k, _, c in facts["required"] + facts["optional"] if k == "datetime"}
    gold_d |= {c for k, _, c in facts["required"] + facts["optional"] if k == "date"}
    key_tools = list(dict.fromkeys(((c.get("arguments") or {}).get("tool_name") or c["name"]) if c["name"] == "sys.ask_missing_info"
                                   else c["name"] for c in calls))
    return {"primary": info["primary"], "kinds": info["kinds"], "blind": blind, "decision": info["decision"],
            "key_tools": key_tools,
            "first_write": first_write, "request": item["audio_text"], "user_text": " ".join(texts),
            "allowed_params": allowed, "gold_values": gold_values, "shown_values": shown_values, "params_of": params_of, "enums_of": enums_of,
            "now_t": (now.hour, now.minute), "now_d": (now.month, now.day), "gold_times": gold_t, "gold_dates": gold_d,
            "said": said_times(item)}


def extra_checks(note: str, ctx: dict) -> list:
    problems = []
    low = note.lower()
    for pat in BANNED_EVERYWHERE:
        if re.search(pat, low):
            problems.append(f"banned phrase /{pat}/ (it tells the outcome of a read)")
    named = set(cn._TOOL.findall(note))
    plain = cn._TOOL.sub(" ", note)
    if ctx["blind"]:
        for pat in BRANCH_PHRASES:
            if re.search(pat, low):
                problems.append(f"branch-revealing phrase /{pat}/")
        primary = ctx["primary"]
        if ctx["primary"] in ("K1", "K2") and "K3" not in ctx["kinds"] and not BOTH_BRANCHES.search(low):
            problems.append("states one branch only: write 'điều kiện/quy tắc thì ..., không thì ...' with both branches")
        if primary == "K1" and ctx["decision"]:
            tool, param = ctx["decision"]
            if tool not in named:
                problems.append(f"K1: does not name the decision tool {tool}")
            if param and not re.search(r"(?<![\w])" + re.escape(param) + r"(?![\w])", note):
                problems.append(f"K1: does not name the parameter {param}")
            if "sys.ask_missing_info" not in named:
                problems.append("K1: the other branch must be sys.ask_missing_info")
        if primary in ("K2", "K3") and ctx["first_write"] and ctx["first_write"] not in named:
            problems.append(f"{primary}: does not name the write tool {ctx['first_write']} of the branch")
        # every sub-task: each tool the key calls (an ask counts as the tool it asks for) is named (review of 8 Nov)
        for t in ctx.get("key_tools", []):
            if t and t not in named:
                problems.append(f"does not name {t}: name the tools of every sub-task of the request, in order, "
                                f"including the lookups that supply IDs")
        if re.search(r"thì thôi[^.;]{0,60}không[^.;]{0,40}thì thôi", low):
            problems.append("both branches are 'thôi': if the request has no real condition, just plan the calls")
        if "K3" in ctx["kinds"] and not SELECT_WORDS.search(low):
            problems.append("K3: say how items are selected (từng / nào / mọi ...)")
        # no number the request does not say (counts, IDs, quantities): times and dates are checked by v1
        rest = cn._DM.sub(" ", cn._ISO_DATE.sub(" ", cn._HM.sub(" ", cn._HH.sub(" ", cn._ID.sub(" ", plain)))))
        said = spoken_numbers(ctx["user_text"]) | cn.numbers_in(ctx["user_text"]) | duration_words(ctx["user_text"])
        for m in re.finditer(r"(?<![\w.,/:])(\d+)(?![\w.,/:])", rest):
            n = m.group(1)
            if int(n) <= 100 and n not in said:
                problems.append(f"number {n} in a branch-blind note is not said in the request (a count the read would show?)")
    # current time or date that nothing feeds
    n_times = {(int(h), int(m)) for h, m in cn._HM.findall(plain)}
    n_dates = {(int(mo), int(d)) for d, mo, _ in cn._DM.findall(plain)}
    if ctx["now_t"] in n_times and len(n_times) == 1 and ctx["now_t"] not in ctx["gold_times"] and ctx["now_t"] not in ctx["said"]:
        problems.append(f"mentions the current time {ctx['now_t'][0]}:{ctx['now_t'][1]:02d}, which nothing in the answer uses")
    if ctx["now_d"] in n_dates and len(n_dates) == 1 and ctx["now_d"] not in ctx["gold_dates"]:
        problems.append(f"mentions today's date {ctx['now_d'][1]}/{ctx['now_d'][0]}, which nothing in the answer uses")
    # parameter names and option values of the tools the note names. A word the key itself sets elsewhere (a parameter of
    # a shown call, the decision parameter, an asked parameter) or a value of a shown call is not an invented detail, even
    # when it is also a parameter or option value of another named tool (phone, contact_id, program, volume, ron95).
    keyed = set().union(*ctx["allowed_params"].values()) | ctx.get("shown_values", set())
    said_compact = re.sub(r"\s+", "", (ctx["request"] + " " + ctx["user_text"]).lower())
    for tool in sorted(named):
        if tool not in ctx["params_of"] or tool == "sys.ask_missing_info":
            continue
        allowed = ctx["allowed_params"].get(tool, set())
        for p in sorted(ctx["params_of"][tool]):
            if len(p) < 3 or p in allowed or p in keyed:
                continue
            if re.search(r"(?<![\w.])" + re.escape(p) + r"(?![\w])", plain):
                problems.append(f"names parameter {p} of {tool}, which the key does not set there")
        gold = ctx["gold_values"].get(tool, {})
        for p, enum in ctx["enums_of"].get(tool, {}).items():
            for val in sorted(enum):
                if len(val) < 4 or val in ctx["request"].lower() or val in ctx["user_text"].lower() or val in said_compact:
                    continue
                if val in keyed:
                    continue
                if re.search(r"(?<![\w])" + re.escape(val) + r"(?![\w])", plain):
                    if ctx["blind"]:
                        problems.append(f"names option value {val} of {tool}, which the request does not say")
                    elif val not in gold.get(p, set()):
                        problems.append(f"names option value {val} for {p} of {tool}; the key uses {sorted(gold.get(p, set()))}")
    return problems


def v2_must(sample_id: str) -> set:
    """The strings a note must contain: the v1 required values of what the writer was shown."""
    by_id, catalog = cn.world()[:2]
    item = by_id[sample_id]
    info = _analysis()[sample_id]
    if not blind_kinds(info):
        return None
    cut = info["cut"]
    t = json.loads(json.dumps(item))
    t["label"]["tool_calls"] = item["label"]["tool_calls"][:cut]
    t["label"]["tool_results"] = item["label"]["tool_results"][:cut]
    facts = cn.classify(t, catalog)
    return set(must_strings(facts["required"]))


_MISSING = re.compile(r"^missing (?:date|time|number|ID|ask tool|ask parameter) (.+)$")


def check_note_v2(sample_id: str, note, batch: int = None) -> list:
    problems = cn.check_note(sample_id, note, batch)
    if not isinstance(note, str) or sample_id in cn.world()[2] or sample_id not in cn.world()[0]:
        return problems
    must = v2_must(sample_id)
    if must is not None:  # branch-blind: only what the writer was shown can be required
        kept = []
        for p in problems:
            m = _MISSING.match(p)
            if m and not (set(m.group(1).split(" ")) & must or m.group(1) in must):
                continue
            kept.append(p)
        problems = kept
    return problems + extra_checks(note, build_ctx(sample_id))


def check_file(path: Path, check=check_note_v2, write=True) -> dict:
    by_id, catalog, validation, batch_of, params, apps = cn.world()
    m = re.search(r"batch_(\d{3})", path.name)
    batch = int(m.group(1)) if m else None
    raw = path.read_text(encoding="utf-8")
    leaked = sorted(v for v in validation if v in raw)
    results, seen, file_problems = {}, set(), []
    for n, line in enumerate(raw.splitlines()):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            file_problems.append(f"line {n + 1}: not JSON ({e})")
            continue
        if set(rec) != {"id", "note"}:
            file_problems.append(f"line {n + 1}: keys {sorted(rec)}, expected id and note")
        sid = rec.get("id")
        if sid in seen:
            file_problems.append(f"duplicate id {sid}")
        seen.add(sid)
        problems = check(sid, rec.get("note"), batch)
        note = rec.get("note") or ""
        results[sid] = {"ok": not problems, "problems": problems, "words": len(note.split()),
                        "tokens": len(cn.tokenizer()(note, add_special_tokens=False)["input_ids"]) if note else 0}
    if leaked:
        file_problems.append(f"VALIDATION IDS IN FILE: {leaked}")
    ok = [r for r in results.values() if r["ok"]]
    summary = {"file": path.name, "notes": len(results), "passed": len(ok), "file_problems": file_problems,
               "mean_tokens": round(sum(r["tokens"] for r in results.values()) / max(len(results), 1), 1),
               "mean_words": round(sum(r["words"] for r in results.values()) / max(len(results), 1), 1),
               "validation_ids_found": leaked}
    if write:
        out = NOTES2 / "checks"
        out.mkdir(parents=True, exist_ok=True)
        (out / (path.stem + ".json")).write_text(json.dumps({"summary": summary, "notes": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"summary": summary, "notes": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--v1", action="store_true", help="check v1 note files under the v2 rules (nothing is written)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    files = sorted(NOTES2.glob("batch_*.jsonl")) if args.all else [Path(f) for f in args.files]
    bad = False
    for f in files:
        res = check_file(f, write=not args.v1)
        s = res["summary"]
        print(json.dumps(s, ensure_ascii=False))
        bad |= bool(s["file_problems"]) or s["passed"] < s["notes"]
        if not args.quiet:
            for sid, r in res["notes"].items():
                if not r["ok"]:
                    print(f"  FAIL {sid}: {'; '.join(r['problems'])}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
