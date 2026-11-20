"""Check worked notes against gold, one batch file at a time.

    python scripts/notes/check_notes.py notes/batch_000.jsonl [more files]   # report, exit 1 if any note fails
    python scripts/notes/check_notes.py --all                                # every notes/batch_*.jsonl, summary

A note passes when:
- it is one line, at most 60 words and 100 Qwen tokens, without markup that could break the prompt (< > { });
- its sample is a non-validation train sample of the right batch, listed once;
- it names the first gold tool, and every tool it names is installed (or sys.ask_missing_info);
- it states every required gold value: dates as D/M (or YYYY-MM-DD), times as H:MM, numbers as digits, IDs verbatim,
  and for an ask the tool and each missing parameter; a "do nothing" note says "không gọi tool";
- it contradicts nothing: each date, time, number and ID it states is a gold value, visible before the first call
  (time, profile, history, request), or simple arithmetic on those; it never states an ID that only a tool result of
  the final turn reveals.
Results go to notes/checks/<file>.json.
"""
import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

from notes_common import REPO, classify, contains_token, load_train, norm_number, param_names, said_times, spoken_numbers

NOTES = REPO / "notes"
MAX_WORDS, MAX_TOKENS = 60, 100
NOTHING_PHRASE = "không gọi tool"
_TOOL = re.compile(r"(?<![\w.@])([a-z][a-z0-9_]*\.[a-z][a-z0-9_]*)(?![\w@])")
_ISO_DATE = re.compile(r"(?<![\d\-])(\d{4})-(\d{2})-(\d{2})(?![\d])")
_DM = re.compile(r"(?<![\d/:.])(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?(?![\d/])")
_HM = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?::\d{2})?(?![\d])")
_HH = re.compile(r"(?<![\w:])(\d{1,2})h(\d{2})?(?![\w])")
_ID = re.compile(r"(?<![\w])([A-Za-z][A-Za-z0-9]*(?:[_\-][A-Za-z0-9]+)+|[A-Za-z]+\d[A-Za-z0-9]*)(?![\w])")
_NUM = re.compile(r"(?<![\w.,])(\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)(?![\w])")
_WEEKDAY_TOKEN = re.compile(r"^T[2-8]$")


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")


@lru_cache(maxsize=1)
def world():
    items, catalog, validation = load_train()
    by_id = {i["id"]: i for i in items}
    order = json.loads((NOTES / "inputs" / "order.json").read_text(encoding="utf-8"))
    batch_of = {sid: b for b, ids in enumerate(order) for sid in ids}
    params = param_names(catalog)
    apps = {name.split(".")[0] for name in catalog}
    return by_id, catalog, validation, batch_of, params, apps


def canon_number(text: str) -> str:
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", text):
        return text.replace(".", "").replace(",", "")
    return text.replace(",", ".")


def numbers_in(text: str) -> set:
    out = set()
    for m in _NUM.finditer(text):
        c = canon_number(m.group(1))
        out.add(c)
        try:
            out.add(norm_number(float(c)) if "." in c else str(int(c)))
        except ValueError:
            pass
    return out


_SCALE = re.compile(r"\s*(triệu|tr|nghìn|ngàn|ngìn|k|tỷ|tỉ)(?![\w])", re.I)
_SCALES = {"triệu": 10 ** 6, "tr": 10 ** 6, "nghìn": 1000, "ngàn": 1000, "ngìn": 1000, "k": 1000, "tỷ": 10 ** 9, "tỉ": 10 ** 9}
_URL = re.compile(r"https?://[^\s,;()]+[^\s,;().]")
_EMAIL = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")


def note_numbers(text: str) -> set:
    """Numbers a note states, canonical; '35,2 triệu' is 35200000 and '15 nghìn' is 15000."""
    out = set()
    for m in _NUM.finditer(text):
        c = canon_number(m.group(1))
        scale = _SCALE.match(text, m.end())
        if scale:
            try:
                out.add(norm_number(float(c) * _SCALES[scale.group(1).lower()]))
                continue
            except ValueError:
                pass
        out.add(c)
    return out


def times_in(text: str) -> set:
    out = {(int(h), int(m)) for h, m in _HM.findall(text)}
    out |= {(int(h), int(m)) for h, m in re.findall(r"T(\d{2}):(\d{2})", text)}
    return out


def dates_in_iso(text: str) -> set:
    return {(int(mo), int(d)) for _, mo, d in _ISO_DATE.findall(text)}


def days_list(now_iso: str) -> set:
    """The dates of the prompt's 'Các ngày tới' line (today and the next 14 days): visible to the model."""
    now = datetime.fromisoformat(now_iso)
    return {((now + timedelta(days=k)).month, (now + timedelta(days=k)).day) for k in range(-7, 15)}  # and last week


def weekday_errors(note: str, now_iso: str) -> list:
    """'T6 13/11' or 'CN 8/11' must name the right weekday for that date (year taken around the current date)."""
    now = datetime.fromisoformat(now_iso)
    out = []
    for wd, d, mo in re.findall(r"(?<![\w])(T[2-7]|CN)\s*(?:\(|,)?\s*(\d{1,2})/(\d{1,2})(?![\d/])", note):
        year = now.year + (1 if int(mo) < now.month - 6 else 0)
        try:
            real = datetime(year, int(mo), int(d)).weekday()
        except ValueError:
            out.append(f"{d}/{mo} is not a date")
            continue
        want = {"T2": 0, "T3": 1, "T4": 2, "T5": 3, "T6": 4, "T7": 5, "CN": 6}[wd]
        if real != want:
            out.append(f"{wd} {d}/{mo} names the wrong weekday")
    return out


def allowed_sets(item: dict, facts: dict) -> tuple:
    vis = facts["visible"]
    now = item["context"]["current_time"]
    gold = facts["required"] + facts["optional"]
    g_times, g_dates, g_nums = set(), set(), set()
    for kind, _, c in gold:
        if kind == "datetime":
            g_dates.add(c[:2]); g_times.add(c[2:])
        elif kind == "date":
            g_dates.add(c)
        elif kind == "time":
            g_times.add(c)
        elif kind == "number":
            g_nums.add(c)
    # numbers that sit inside gold arguments as plain values (quantities, IDs of digits)
    vis_nums = numbers_in(vis)
    now_t = (int(now[11:13]), int(now[14:16]))
    now_d = (int(now[5:7]), int(now[8:10]))
    base_times = g_times | {now_t} | times_in(vis)
    spoken = spoken_numbers(vis)
    hours = {int(float(x)) for x in spoken if re.fullmatch(r"\d{1,2}(\.5)?", x) and float(x) <= 24}
    hours |= {int(h) for h in re.findall(r"(?<!\d)(\d{1,2})\s*(?:giờ|h)(?![\w])", vis)}
    for n in hours:  # an hour said in words or digits: any 5-minute time in it, morning or afternoon, or "kém"
        for hh in {n % 24, (n + 12) % 24, (n - 1) % 24, (n + 11) % 24}:
            base_times |= {(hh, mm) for mm in range(0, 60, 5)}
    tight = g_times | {now_t} | times_in(vis) | said_times(item)  # exact times, not the 5-minute grid of an hour
    small = set()
    for c in vis_nums | spoken:  # a duration shown or said earlier ("đi xe máy mất chừng 42 phút")
        try:
            v = float(c)
        except ValueError:
            continue
        if v.is_integer() and 1 <= v <= 240:
            small.add(int(v))
    deltas = {60}
    for c in g_nums:
        try:
            v = float(c)
        except ValueError:
            continue
        if v.is_integer() and 0 < v <= 1440:
            deltas.add(int(v))
    times = set(base_times)
    for h, m in base_times:
        for d in deltas:
            for sign in (1, -1):
                t = (h * 60 + m + sign * d) % 1440
                times.add((t // 60, t % 60))
    for h, m in tight:
        for d in small:
            for sign in (1, -1):
                t = (h * 60 + m + sign * d) % 1440
                times.add((t // 60, t % 60))
    dates = g_dates | {now_d} | dates_in_iso(vis) | days_list(now) | {(int(mo), int(d)) for d, mo, _ in _DM.findall(vis)}
    g_digits = {c for kind, _, c in gold if kind == "id" and isinstance(c, str) and c.isdigit()}
    nums = set(str(i) for i in range(13)) | {str(i) for i in range(0, 61, 5)} | vis_nums | g_nums | g_digits | spoken | {"2026"}
    for h, m in times:
        nums |= {str(h), str(m), f"{m:02d}"}
    for mo, d in dates:
        nums |= {str(mo), str(d)}
    mins = sorted({h * 60 + m for h, m in base_times})
    for a in mins:
        for b in mins:
            if a < b:
                nums.add(str(b - a))
                if (b - a) % 60 == 0:
                    nums.add(str((b - a) // 60))
    operands = []
    for c in vis_nums | g_nums | spoken:
        try:
            v = float(c)
        except ValueError:
            continue
        if v >= 2 and len(operands) < 80:
            operands.append(v)
    if "rưỡi" in vis.lower():  # "tiếng rưỡi" = 1,5; "triệu rưỡi" = 1500000; "giá × 1,5"
        halves = [k + 0.5 for k in range(13)]
        for x in halves:
            nums |= {norm_number(x), norm_number(x * 1000), norm_number(x * 1000000)}
            nums |= {norm_number(round(a * x, 2)) for a in operands}
    for a in operands:
        nums |= {norm_number(a * 1000), norm_number(a * 1000000)}
        for b in operands:
            for v in (a + b, a - b, a * b):
                if 0 <= v < 1e11:
                    nums.add(norm_number(round(v, 2)))
    return times, dates, nums


def check_note(sample_id: str, note, batch: int = None) -> list:
    by_id, catalog, validation, batch_of, params, apps = world()
    problems = []
    if sample_id in validation:
        return ["VALIDATION SAMPLE"]
    item = by_id.get(sample_id)
    if item is None:
        return ["unknown id"]
    if batch is not None and batch_of.get(sample_id) != batch:
        problems.append(f"id belongs to batch {batch_of.get(sample_id)}, not {batch}")
    if not isinstance(note, str) or not note.strip():
        return problems + ["empty note"]
    if "\n" in note or re.search(r"[<>{}]", note) or "think" in note.lower():
        problems.append("newline or markup (< > { } think) in the note")
    words = len(note.split())
    tokens = len(tokenizer()(note, add_special_tokens=False)["input_ids"])
    if words > MAX_WORDS:
        problems.append(f"{words} words (max {MAX_WORDS})")
    if tokens > MAX_TOKENS:
        problems.append(f"{tokens} tokens (max {MAX_TOKENS})")
    facts = classify(item, catalog)
    calls = item["label"]["tool_calls"]
    installed = {t for app in item["context"]["installed_apps"].values() for t in app["tools"]} | {"sys.ask_missing_info"}
    gold_text = json.dumps(item["label"]["tool_calls"], ensure_ascii=False)
    facts_visible = None
    plain = note
    for url in set(_URL.findall(note)):  # a link the history shows or gold sends is a value, not a tool name or an ID
        if facts_visible is None:
            facts_visible = classify(item, catalog)["visible"]
        if url in gold_text or url in facts_visible:
            plain = plain.replace(url, " ")
    named = set(_TOOL.findall(plain))
    for t in sorted(named - installed):
        problems.append(f"names a tool that is not installed: {t}")
    if calls and calls[0]["name"] not in named:
        problems.append(f"does not name the first tool {calls[0]['name']}")
    if not calls and NOTHING_PHRASE not in note.lower():
        problems.append(f"gold does nothing but the note does not say '{NOTHING_PHRASE}'")
    rest = _TOOL.sub(" ", plain)
    n_dates = {(int(mo), int(d)) for _, mo, d in _ISO_DATE.findall(rest)}
    rest = _ISO_DATE.sub(" ", rest)
    for d, mo, _ in _DM.findall(rest):
        n_dates.add((int(mo), int(d)))
    rest = _DM.sub(" ", rest)
    n_times = {(int(h), int(m)) for h, m in _HM.findall(rest)}
    rest = _HM.sub(" ", rest)
    n_times |= {(int(h), int(m or 0)) for h, m in _HH.findall(rest) if int(h) < 24}  # "Góc Phố 24h" is a name
    rest = _HH.sub(" ", rest)
    for email in set(_EMAIL.findall(rest)):  # an e-mail gold sends or the history shows is a value, not an ID
        if email in gold_text or email in facts["visible"]:
            rest = rest.replace(email, " ")
    n_ids = [t for t in _ID.findall(rest) if not _WEEKDAY_TOKEN.match(t) and t not in params and t not in apps]
    rest_no_ids = _ID.sub(" ", rest)
    n_nums = numbers_in(rest_no_ids) | note_numbers(rest_no_ids)
    raw_nums = note_numbers(rest_no_ids)
    # required values
    for kind, display, c in facts["required"]:
        if kind == "datetime":
            if c[:2] not in n_dates:
                problems.append(f"missing date {display.split(' ')[0]}")
            if c[2:] not in n_times:
                problems.append(f"missing time {display.split(' ')[1]}")
        elif kind == "date" and c not in n_dates:
            problems.append(f"missing date {display}")
        elif kind == "time" and c not in n_times:
            problems.append(f"missing time {display}")
        elif kind == "number" and c not in n_nums:
            problems.append(f"missing number {display}")
        elif kind == "id":
            if not (contains_token(note, c) or (c.isdigit() and c in raw_nums)):
                problems.append(f"missing ID {c}")
        elif kind == "ask":
            if c["tool"] not in named:
                problems.append(f"missing ask tool {c['tool']}")
            for p in c["params"]:
                if not contains_token(note, p):
                    problems.append(f"missing ask parameter {p}")
    # contradictions
    problems += weekday_errors(note, item["context"]["current_time"])
    times, dates, nums = allowed_sets(item, facts)
    for d in sorted(n_dates - dates):
        problems.append(f"date {d[1]}/{d[0]} is neither gold nor visible")
    for t in sorted(n_times - times):
        problems.append(f"time {t[0]}:{t[1]:02d} is neither gold, visible nor simple arithmetic")
    gold_times = {c[2:] for kind, _, c in facts["required"] + facts["optional"] if kind == "datetime"}
    gold_times |= {c for kind, _, c in facts["required"] + facts["optional"] if kind == "time"}
    for t in sorted(n_times - gold_times):  # 4:00 where gold has 16:00: morning and afternoon mixed up
        twin = ((t[0] + 12) % 24, t[1])
        if twin in gold_times and t not in times_in(facts["visible"]):
            problems.append(f"time {t[0]}:{t[1]:02d} looks like gold {twin[0]}:{twin[1]:02d} with morning and afternoon mixed up")
    gold_ids = {c for kind, _, c in facts["required"] if kind == "id"}
    for t in installed:  # fixed choices of an installed tool (vehicle car4, cycle quick) are part of its schema
        for spec in ((catalog.get(t, {}).get("parameters") or {}).get("properties") or {}).values():
            gold_ids |= {str(e) for e in spec.get("enum") or []}
    for t in n_ids:
        if t in facts["hidden_ids"] and t not in {part for x in installed for part in x.split(".")}:
            problems.append(f"states {t}, which only a tool result of this turn reveals")
        elif t not in gold_ids and not contains_token(facts["visible"], t):
            problems.append(f"ID-like token {t} is neither gold nor visible")
    for n in sorted(raw_nums):
        if n not in nums and canon_number(n) not in nums:
            try:
                alt = norm_number(float(n))
            except ValueError:
                alt = n
            if alt not in nums:
                problems.append(f"number {n} is neither gold, visible nor simple arithmetic")
    tool_parts = {part for t in installed for part in t.split(".")}
    for h in facts["hidden_ids"]:
        if h in tool_parts or h in params or h in apps:
            continue
        if h not in n_ids and contains_token(_TOOL.sub(" ", note), h):
            problems.append(f"states {h}, which only a tool result of this turn reveals")
    return problems


def check_file(path: Path) -> dict:
    by_id, catalog, validation, batch_of, params, apps = world()
    m = re.search(r"batch_(\d{3})", path.name)
    batch = int(m.group(1)) if m else None
    raw = path.read_text(encoding="utf-8")
    leaked = sorted(v for v in validation if v in raw)  # independent of the parsing below
    rows, results, seen = [], {}, set()
    file_problems = []
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
        problems = check_note(sid, rec.get("note"), batch)
        note = rec.get("note") or ""
        results[sid] = {"ok": not problems, "problems": problems, "words": len(note.split()),
                        "tokens": len(tokenizer()(note, add_special_tokens=False)["input_ids"]) if note else 0}
        rows.append(rec)
    if leaked:
        file_problems.append(f"VALIDATION IDS IN FILE: {leaked}")
    ok = [r for r in results.values() if r["ok"]]
    summary = {"file": path.name, "notes": len(results), "passed": len(ok), "file_problems": file_problems,
               "mean_tokens": round(sum(r["tokens"] for r in results.values()) / max(len(results), 1), 1),
               "mean_words": round(sum(r["words"] for r in results.values()) / max(len(results), 1), 1),
               "validation_ids_found": leaked}
    out = NOTES / "checks"
    out.mkdir(parents=True, exist_ok=True)
    (out / (path.stem + ".json")).write_text(json.dumps({"summary": summary, "notes": results}, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    return {"summary": summary, "notes": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="summary lines only")
    args = ap.parse_args()
    files = sorted(NOTES.glob("batch_*.jsonl")) if args.all else [Path(f) for f in args.files]
    bad = False
    for f in files:
        res = check_file(f)
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
