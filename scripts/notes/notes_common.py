"""Shared logic for the worked-notes experiment: what a note may and must say about one train sample.

A note is written for the first step of the final turn, before any tool runs in that turn. It may use only what the
model can see then: the current time, the profile, the earlier turns with their tool calls and results, and the
latest request. Gold values are sorted into three kinds for each sample:

- required: knowable at step 1 (spoken, or visible in the history), and a number, time, date or ID. The note must
  state each one, in the note format (dates D/M, times H:MM, IDs verbatim, numbers as digits).
- optional: a value that may depend on a read made in this turn (a datetime on the same day as one a read returned,
  or a number a read returned). The note may state it but need not.
- hidden: an ID (or other value) that only a tool result of this turn reveals. The writer never sees it, and the
  checker rejects a note that contains it.
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from lifemate import data, prompts  # noqa: E402

_ISO_DT = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?$")
_ISO_D = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_HHMM = re.compile(r"^(\d{1,2}):(\d{2})$")
_DIGIT = re.compile(r"\d")
WEEKDAYS = prompts._WEEKDAYS


_DIG = {"không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "lăm": 5, "nhăm": 5, "sáu": 6,
        "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9}
_MULT = {"nghìn": 1000, "ngàn": 1000, "k": 1000, "triệu": 10 ** 6, "tỷ": 10 ** 9, "tỉ": 10 ** 9}
_NUMWORDS = set(_DIG) | set(_MULT) | {"mười", "mươi", "chục", "trăm", "rưỡi", "linh", "lẻ"}


def _spoken_value(run: list):
    """Value of a run of Vietnamese number words and digits, standard and colloquial forms ("hai lăm ngàn rưỡi" = 25500,
    "hai triệu tám" = 2800000, "ba trăm hai" = 320). None if the run makes no number."""
    total, section, cur, prev, abbrev = 0, 0, None, None, 0
    seen = False
    for t in run:
        d = _DIG.get(t, int(t) if t.isdigit() else None)
        if d is not None:
            seen = True
            if prev == "mươi" or prev == "mười":
                section += d
                cur = None
            elif prev in ("linh", "lẻ"):
                section += d
                cur = None
            elif prev == "trăm" and d < 10:
                cur, abbrev = d, 10
            elif prev in _MULT and d < 10:
                cur, abbrev = d, _MULT[prev] // 10
            elif cur is not None and cur < 10 and d < 10 and prev is not None and (prev in _DIG or prev.isdigit()):
                section += cur * 10 + d
                cur = None
            else:
                cur, abbrev = d, 0
        elif t == "mười":
            seen = True
            section += 10
            cur = None
        elif t in ("mươi", "chục"):  # "chín chục" = 90, as "chín mươi"
            section += (cur if cur is not None else 1) * 10
            cur = None
        elif t == "trăm":
            section += (cur if cur is not None else 1) * 100
            cur, abbrev = None, 0
        elif t in _MULT:
            section += cur or 0
            total += (section or 1) * _MULT[t]
            section, cur, abbrev = 0, None, 0
        elif t == "rưỡi":
            if prev in _MULT:
                total += _MULT[prev] // 2
            elif prev == "trăm":
                section += 50
            else:
                section += 0.5
        prev = t
    if not seen:
        return None
    if cur is not None:
        section += cur * abbrev if abbrev else cur
    value = total + section
    return int(value) if float(value).is_integer() else value


def spoken_numbers(text: str) -> set:
    """Every number a run of number words (or digits) in ``text`` can mean, sub-runs included, as strings."""
    words = []
    for m in re.finditer(r"[0-9]+|[^\W\d_]+", text):
        w = m.group(0)
        before = text[:m.start()].rstrip()
        # "Cô Ba", "Chị Hai", "Bác Sáu": a capitalised number word inside a sentence is a name, not a number
        if w[0].isupper() and w.lower() in _NUMWORDS and before and (before[-1].isalnum() or before[-1] == ","):
            words.append("·")
        else:
            words.append(w.lower())
    out = set()
    i = 0
    while i < len(words):
        if words[i] in _NUMWORDS or words[i].isdigit():
            j = i
            while j < len(words) and (words[j] in _NUMWORDS or words[j].isdigit()):
                j += 1
            run = words[i:j]
            digits = [str(_DIG[w]) if w in _DIG else w for w in run]
            for a in range(len(run)):  # digit by digit, as phone numbers and codes are read ("đuôi tám sáu tám")
                for b in range(a + 2, len(run) + 1):
                    if all(len(x) == 1 and x.isdigit() for x in digits[a:b]):
                        out.add("".join(digits[a:b]))
            for a in range(len(run)):
                for b in range(a + 1, min(len(run), a + 8) + 1):
                    v = _spoken_value(run[a:b])
                    if v is not None:
                        out.add(norm_number(v))
            i = j
        else:
            i += 1
    return out


def said_times(item: dict) -> set:
    """(hour, minute) pairs the user says outright: 'N giờ', 'N giờ rưỡi', 'N giờ M', 'N giờ kém M', or H:MM in digits,
    morning or afternoon reading. A gold time in this set is knowable at step 1 whatever reads come before it."""
    texts = [item["audio_text"]] + [t["audio_text"] for t in item["history"] if t["role"] == "user"]
    out = set()
    for text in texts:
        low = text.lower()
        for h, m in re.findall(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", low):
            out.add((int(h), int(m)))
        words = re.findall(r"[0-9]+|[^\W\d_]+", low)
        for i, w in enumerate(words):
            if w not in ("giờ", "h"):
                continue
            hour = None
            for a in range(max(0, i - 3), i):
                v = _spoken_value(words[a:i])
                if v is not None and float(v).is_integer() and 0 <= v <= 24:
                    hour = int(v)
                    break
            if hour is None:
                continue
            tail = words[i + 1:i + 5]
            minutes = {0}
            if tail[:1] == ["rưỡi"]:
                minutes = {30}
            elif tail[:1] == ["kém"]:
                for b in range(2, 5):
                    v = _spoken_value(tail[1:b])
                    if v is not None and float(v).is_integer() and 0 < v < 60:
                        minutes = {60 - int(v)}
                        hour -= 1
                        break
            else:
                for b in range(1, 5):
                    v = _spoken_value(tail[:b])
                    if v is not None and float(v).is_integer() and 0 < v < 60:
                        minutes = {0, int(v)}
                        break
            for m in minutes:
                out |= {(hour % 24, m), ((hour + 12) % 24, m)}
    return out


def has_duration(results: list) -> bool:
    """A result that carries a duration or a time (route minutes, ETA, remaining time)."""
    text = json.dumps(results, ensure_ascii=False)
    return re.search(r'"[a-z_]*(min|duration|eta|time|hour|minute)[a-z_]*":\s*\d', text) is not None


def durations(results: list) -> set:
    """Minute-like numbers of a result (route time, ETA, remaining minutes)."""
    out = set()
    for m in re.finditer(r'"[a-z_]*(?:min|duration|eta|time|hour|minute)[a-z_]*":\s*(\d+)', json.dumps(results, ensure_ascii=False)):
        v = int(m.group(1))
        if 0 < v <= 1440:
            out.add(v)
    return out


def has_list(results: list) -> bool:
    return any(isinstance(v, list) and v for r in results for v in (r.values() if isinstance(r, dict) else []))


def load_train():
    root = data.find_root("train")
    items = data.load_split("train", root)
    catalog = data.load_catalog("train", root)
    validation = data.id_set("splits")
    return items, catalog, validation


def param_names(catalog: dict) -> set:
    out = set()
    for spec in catalog.values():
        out |= set(((spec.get("parameters") or {}).get("properties") or {}))
        for p in ((spec.get("parameters") or {}).get("properties") or {}).values():
            items = p.get("items")
            if isinstance(items, dict):
                out |= set(items.get("properties") or {})
    return out


def visible_text(item: dict) -> str:
    """What the model sees at step 1 apart from the tool list (whose examples would pollute value matching)."""
    ctx = item["context"]
    now = ctx["current_time"]
    head = [f"Bây giờ: {now}", f"Các ngày tới: {prompts.calendar_hint(now)}",
            f"Người dùng: {json.dumps(ctx.get('user_profile', {}), ensure_ascii=False)}"]
    texts = prompts.gold_transcripts(item)
    return "\n".join(head) + "\n" + prompts.history_text(item, texts) + "\n" + item["audio_text"]


def is_id_like(value, spec: dict) -> bool:
    if not isinstance(value, str) or not value or " " in value.strip():
        return False
    if spec.get("x-ref"):
        return True
    if _ISO_DT.match(value) or _ISO_D.match(value) or _HHMM.match(value):
        return False
    return bool(_DIGIT.search(value)) and bool(re.fullmatch(r"[\w\-.+]+", value))


def contains_token(text: str, value: str) -> bool:
    return re.search(r"(?<![\w])" + re.escape(value) + r"(?![\w])", text, re.I) is not None


def results_datetimes(results: list) -> list:
    out = []
    for r in results:
        for m in re.finditer(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", json.dumps(r, ensure_ascii=False)):
            out.append(datetime.fromisoformat(m.group(0)))
    return out


def results_numbers(results: list) -> set:
    out = set()

    def walk(v):
        if isinstance(v, bool):
            return
        if isinstance(v, str):  # amounts inside texts too ("chuyển 2.000 USD")
            for m in re.findall(r"\d{1,3}(?:[.,]\d{3})+|\d+", v):
                out.add(str(int(re.sub(r"[.,]", "", m))))
            return
        if isinstance(v, (int, float)):
            out.add(norm_number(v))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    for r in results:
        walk(r)
    return out


def norm_number(v) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v)


def fmt_date(y, mo, d) -> str:
    return f"{int(d)}/{int(mo)}"


def fmt_time(h, mi) -> str:
    return f"{int(h)}:{int(mi):02d}"


def leaf_values(arguments, spec_props: dict, path=""):
    """(param name, value, its schema) for every scalar of a call's arguments, lists and objects included."""
    for key, value in arguments.items():
        spec = spec_props.get(key, {})
        if isinstance(value, list):
            items = spec.get("items") if isinstance(spec.get("items"), dict) else {}
            for v in value:
                if isinstance(v, dict):
                    yield from leaf_values(v, items.get("properties") or {}, key + "[]")
                else:
                    yield key, v, items
        elif isinstance(value, dict):
            yield from leaf_values(value, spec.get("properties") or {}, key)
        else:
            yield key, value, spec


def read_arithmetic(value, before_nums: set, vis: str) -> bool:
    """value = a + b, a - b or a x b with a from this turn's results (b from them, the visible text, or small counts)."""
    reads = []
    for c in before_nums:
        try:
            reads.append(float(c))
        except ValueError:
            pass
    if not reads:
        return False
    others = set(reads) | {float(x) for x in spoken_numbers(vis) if _is_float(x)} | {float(i) for i in range(1, 21)}
    ratios = {k / 10 for k in range(1, 101)} | {1 / k for k in range(2, 21)}
    for a in reads[:200]:
        for b in others:
            if value in (a + b, a - b, b - a, a * b):
                return True
        if a and any(abs(value - a * r) < 0.5 for r in ratios):  # 2 chỉ = 0.2 lượng of a price per lượng, half, a third
            return True
    return False


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def classify(item: dict, catalog: dict) -> dict:
    """Required, optional and hidden values of one sample, plus the facts the checker needs."""
    vis = visible_text(item)
    calls, results = item["label"]["tool_calls"], item["label"]["tool_results"]
    said = said_times(item)
    said_now = said_times({"audio_text": item["audio_text"], "history": []})  # the latest request only
    said_nums = spoken_numbers(item["audio_text"])
    now_dt = datetime.fromisoformat(item["context"]["current_time"])
    vis_dts = [datetime.fromisoformat(m) for m in re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", vis)]
    required, optional, hidden = [], [], []   # entries: (kind, display, canonical)
    shown_calls = []
    any_read_before = False
    for k, call in enumerate(calls):
        name, args = call["name"], call.get("arguments") or {}
        before = results[:k]
        before_text = json.dumps(before, ensure_ascii=False)
        before_dts = results_datetimes(before)
        before_nums = results_numbers(before)
        props = (catalog.get(name, {}).get("parameters") or {}).get("properties") or {}
        shown = {}
        read_ref = False
        # IDs first: a call that points at an item found in this turn depends on that read.
        for key, value, spec in leaf_values(args, props):
            if is_id_like(value, spec) and not contains_token(vis, value) and value in before_text:
                read_ref = True
        if name == "sys.ask_missing_info":
            entry = ("ask", f"{args.get('tool_name')} [{', '.join(args.get('params') or [])}]",
                     {"tool": args.get("tool_name"), "params": list(args.get("params") or [])})
            (optional if any_read_before else required).append(entry)
            shown_calls.append({"name": name, "arguments": args})
            continue
        for key, value, spec in leaf_values(args, props):
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (int, float)):
                canon = norm_number(value)
                entry = ("number", canon, canon)
                in_history = contains_token(vis, canon) and (not isinstance(value, (int, float)) or value > 20)
                if canon in said_nums or (in_history and not before):
                    required.append(entry)  # said in digits or words, or (if not a small count) shown in the history
                elif before:
                    optional.append(entry)  # after a read of this turn, anything not said may come from that read
                else:
                    required.append(entry)
                continue
            if not isinstance(value, str):
                continue
            m = _ISO_DT.match(value)
            if m:
                y, mo, d, h, mi = m.groups()
                dt = datetime(int(y), int(mo), int(d), int(h), int(mi))
                entry = ("datetime", f"{fmt_date(y, mo, d)} {fmt_time(h, mi)}", (int(mo), int(d), int(h), int(mi)))
                near_read = any(abs((x.replace(tzinfo=None) - dt).total_seconds()) <= 12 * 3600 for x in before_dts)
                tight = said | {(now_dt.hour, now_dt.minute)} | {(x.hour, x.minute) for x in vis_dts}
                mins = int(h) * 60 + int(mi)
                offsets = {0} | set(range(5, 125, 5))  # a spoken margin: "tới sớm hai mươi phút"
                by_duration = any((b[0] * 60 + b[1] + s1 * d + s2 * o) % 1440 == mins for b in tight
                                  for d in durations(before) for o in offsets for s1 in (1, -1) for s2 in (1, -1))
                hhmm = f"{int(h):02d}:{int(mi):02d}"
                if value[:16] in before_text and value[:16] not in vis:
                    optional.append(entry)
                elif f'"{hhmm}"' in before_text and hhmm not in vis and (int(h), int(mi)) not in said:
                    optional.append(entry)  # a schedule a read of this turn returns ("times": ["07:30"])
                elif before:
                    # a call of this turn came first: its result may set the date or the time (a voucher's expiry, an
                    # event's start), and a match with an older turn can be chance. Only a time said in the latest
                    # request stays required; the date and everything else may come from the read.
                    if (int(h), int(mi)) in said_now:
                        required.append(("time", fmt_time(h, mi), (int(h), int(mi))))
                    optional.append(entry)
                elif (int(h), int(mi)) in said or value[:16] in vis:
                    required.append(entry)  # said outright or shown: knowable whatever reads come first
                elif near_read or by_duration:
                    optional.append(entry)
                else:
                    required.append(entry)
                continue
            m = _ISO_D.match(value)
            if m:
                y, mo, d = m.groups()
                entry = ("date", fmt_date(y, mo, d), (int(mo), int(d)))
                read_dates = re.search(r"\d{4}-\d{2}-\d{2}", before_text) is not None
                (optional if (value in before_text or read_dates) and value not in vis else required).append(entry)
                continue
            m = _HHMM.match(value)
            if m:
                h, mi = m.groups()
                entry = ("time", fmt_time(h, mi), (int(h), int(mi)))
                read_given = value in before_text and value not in vis and (int(h), int(mi)) not in said
                (optional if read_given else required).append(entry)
                continue
            if is_id_like(value, spec):
                if contains_token(vis, value):
                    required.append(("id", value, value))
                elif value in before_text:
                    hidden.append(("id", value, value))
                    shown[key] = "⟨từ kết quả gọi trước trong lượt này⟩"
                else:
                    required.append(("id", value, value))  # spoken (phone number, channel) or a fixed choice
        shown_args = {}
        for key, value in args.items():
            shown_args[key] = shown.get(key, value) if not isinstance(value, (list, dict)) else _mask(value, hidden)
        shown_calls.append({"name": name, "arguments": shown_args})
        if catalog.get(name, {}).get("kind") == "read":
            any_read_before = True
    # every value a result of this turn reveals, with nothing visible to explain it, is hidden too
    future = set()
    for v in string_values(results):
        if re.fullmatch(r"[A-Za-z][\w\-]*\d[\w\-]*|[a-z]+_[a-z_]+\d*", v) and not contains_token(vis, v):
            future.add(v)
    req_ids = {c for kind, _, c in required if kind == "id"}
    enums = {str(e) for tool in catalog.values() for spec in ((tool.get("parameters") or {}).get("properties") or {}).values()
             for e in (spec.get("enum") or (spec.get("items") or {}).get("enum") or [])}  # fixed choices of a tool's schema
    hidden_ids = sorted((future | {c for kind, _, c in hidden if kind == "id"}) - req_ids - enums)
    return {"required": _dedupe(required), "optional": _dedupe(optional), "hidden_ids": hidden_ids,
            "shown_calls": shown_calls, "visible": vis}


def string_values(value) -> list:
    """Every string value (not key) inside a JSON value."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in string_values(v)]
    if isinstance(value, list):
        return [s for v in value for s in string_values(v)]
    return []


def _mask(value, hidden):
    hid = {c for kind, _, c in hidden if kind == "id"}
    if isinstance(value, list):
        return [_mask(v, hidden) for v in value]
    if isinstance(value, dict):
        return {k: _mask(v, hidden) for k, v in value.items()}
    return "⟨từ kết quả gọi trước trong lượt này⟩" if isinstance(value, str) and value in hid else value


def _dedupe(entries):
    seen, out = set(), []
    for e in entries:
        key = (e[0], json.dumps(e[2], ensure_ascii=False, sort_keys=True) if not isinstance(e[2], str) else e[2])
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def must_strings(required) -> list:
    """The required values as the writer should write them."""
    out = []
    for kind, display, canon in required:
        if kind == "ask":
            out.append(canon["tool"])
            out += canon["params"]
        elif kind == "datetime":
            out += display.split(" ")
        else:
            out.append(display)
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq
