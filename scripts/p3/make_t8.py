"""T8 of improvment_plan3.md: new answers from real train samples, by changing one said value (a contact or a number).

    python scripts/p3/make_t8.py prepare            # swaps + new labels (replayed) + agy batches in outputs/p3/t8/
    python scripts/p3/make_t8.py run                # Gemini (agy) rewords each changed request (resumable)
    python scripts/p3/make_t8.py collect            # checks every Gemini variant (same rules as T7)
    python scripts/p3/make_t8.py build              # outputs/p3/t8_full.jsonl, t8_fold0.jsonl

Why: T2a/T2b/T7 repeat the gold calls of the 1,545 train samples (each sample ~7 times), so the new data mostly teaches
answers the model has already learned twice. T8 keeps a real sample's situation, structure and tool chain but changes
one value the user says, so the answer is new and has to be read from the request:
- contact: a contacts.search query said word for word is replaced by another contact of the same sample's phone book
  (not named anywhere in the conversation); the search is replayed, and the old contact's id, phone and name are
  replaced by the new contact's everywhere later in the calls.
- number: an integer argument said in the request (money, minutes, steps, degrees, people...) gets another value of the
  same roundness and size (0.4x-2.5x); every argument with that value changes with it, as does the said phrase.
Every changed call chain is replayed on the sample's own train state (zaic_tools.ToolEnv): each call must run, every
write must succeed as the original did, and no later argument may still point at the old contact. The new results
are the replayed ones. Conditional requests ("nếu") keep their numbers (the condition could flip).
The request text is first changed by plain substitution (always valid); Gemini then rewords it in a regional style
keeping every value word for word (T7's checks). A swap whose Gemini variants all fail keeps the substituted text.
Claude writes no request text: substitution only swaps the value phrase, all rewording comes from Gemini.
"""
import argparse
import copy
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import common
import make_t7
import review_rules
from make_t6 import SPAN as NUM_SPAN, asr_numbers  # noqa: F401
from lifemate.drills import DIGIT_SHARE
from lifemate.drills import parse_number, say_number

sys.path.insert(0, str(common.REPO / "scripts" / "t7"))
import agy_paraphrase as agy  # noqa: E402

OUT = common.OUT / "t8"
BATCHES, ANSWERS = OUT / "batches", OUT / "answers"
SWAPS = OUT / "swaps.jsonl"
NUMW = ("không|một|mốt|hai|ba|bốn|tư|năm|lăm|nhăm|sáu|bảy|tám|chín|mười|mươi|trăm|linh|lẻ|chục|nghìn|ngàn|triệu|"
        "rưỡi")
SPAN = re.compile(rf"(?<![\wÀ-ỹ.,])((?:\d+(?:[.,]\d+)?k?|{NUMW})(?:\s+(?:\d+|{NUMW}))*)(?![\wÀ-ỹ])", re.I)
UNIT_AFTER = re.compile(r"\s*(nghìn|ngàn|triệu|đồng|k|phút|giờ|tiếng|độ|bước|người|chỗ|vé|ngày|lần|phần|cái|ly|"
                        r"suất|km|kg|ký|lít|%|phần trăm)\b", re.I)
COND = re.compile(r"(?<![\wÀ-ỹ])(nếu|dưới|trên|quá|vượt|đủ|thiếu|hơn|ít nhất|tối đa|tối thiểu|nhỏ hơn|lớn hơn|"
                  r"không quá|trở lên|trở xuống|còn lại|chia|mỗi|tổng|cộng)(?![\wÀ-ỹ])")
# sensible ranges per argument (the catalog rarely has bounds): degrees, percentages, minutes
RANGES = {"temperature": (16, 30, 1), "brightness": (10, 100, 10), "volume": (10, 100, 10), "percent": (5, 100, 5),
          "humidity": (30, 80, 5), "speed": (1, 5, 1), "level": (1, 5, 1), "fan_speed": (1, 5, 1)}
STATEFUL = {"state", "status", "enabled", "power", "locked", "balance", "done", "completed", "stock", "quantity",
            "assignee", "paid", "due_date"}
SKIP_KEYS = {"limit", "page", "year", "month", "day", "hour", "minute"}
WRITE = re.compile(r"\.(send|create|update|transfer|book|order|set|add|delete|cancel|pay|topup|call|complete|play|"
                   r"open|turn|light|mark|log|reply|remove|schedule|start|stop|join)", re.I)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def env_of():
    root = common.data.find_root("train")
    sys.path.insert(0, str(root / "train"))
    from zaic_tools import ToolEnv
    return lambda sid: ToolEnv("train", sid, root=str(root))


def replay(make_env, sid: str, calls: list) -> list:
    env = make_env(sid)
    return [env.call(c["name"], copy.deepcopy(c.get("arguments") or {})) for c in calls]


def state_of(sid: str) -> dict:
    p = common.data.find_root("train") / "train" / "train_environment" / sid / "state.json"
    return json.loads(p.read_text(encoding="utf-8"))


def history_text(row: dict) -> str:
    return " ".join(norm(t) for a, t in common.prompts.gold_transcripts(row).items() if a != row["audio"])


def same_outcome(old: list, new: list) -> bool:
    """Every call runs and every result has the original's success flags ("ok", "done")."""
    for a, b in zip(old, new):
        if not isinstance(b, dict) or b.get("ok") is False:
            return False
        for k in ("ok", "done", "success"):
            if k in a and a.get(k) != b.get(k):
                return False
        for k, v in a.items():  # a list that was empty (nothing found) or not must stay so
            if isinstance(v, list) and bool(v) != bool(b.get(k)):
                return False
    return True


def stale_copies(old_calls: list, old_res: list, new_calls: list, new_res: list) -> bool:
    """True when a later argument copied a value from an earlier result (a converted amount, a bill total) and that
    result changed with the swap while the argument did not: the new answer would carry the old value."""
    for i, (a, b) in enumerate(zip(old_calls, new_calls)):
        before_old = {json.dumps(x, ensure_ascii=False) for x in scalars(old_res[:i])}
        before_new = {json.dumps(x, ensure_ascii=False) for x in scalars(new_res[:i])}
        for key, x in (a.get("arguments") or {}).items():
            if isinstance(x, bool) or not isinstance(x, (int, float, str)):
                continue
            j = json.dumps(x, ensure_ascii=False)
            if (b.get("arguments") or {}).get(key) == x and j in before_old and j not in before_new:
                return True
    return False


# ---------------------------------------------------------------------------------------------- entity swaps
def records_with(obj, values: set) -> list:
    """Dict records anywhere in ``obj`` that hold one of ``values``."""
    out = []
    if isinstance(obj, dict):
        if any(isinstance(v, str) and v in values for v in obj.values()):
            out.append(obj)
        for v in obj.values():
            out += records_with(v, values)
    elif isinstance(obj, list):
        for v in obj:
            out += records_with(v, values)
    return out


def scalars(obj) -> list:
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in scalars(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in scalars(v)]
    return [obj]


def id_lists(state: dict) -> list:
    """Every list of records with an "id" in a train state (contacts, devices, events, restaurants...)."""
    out = []

    def walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list) and v and all(isinstance(x, dict) and "id" in x for x in v):
                    out.append(v)
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
    walk({k: v for k, v in state.items() if k != "_meta"})
    return out


ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T")


def name_of(rec: dict):
    for k in ("name", "title", "label"):
        if isinstance(rec.get(k), str) and len(rec[k]) >= 2:
            return k, rec[k]
    return None, None


def entity_swaps(row: dict, make_env, rng: random.Random, per: int) -> list:
    """Another record of the same list (same type) for a record whose name the request says word for word."""
    calls, results = row["label"]["tool_calls"], row["label"]["tool_results"]
    text, hist = row["audio_text"], history_text(row)
    lists = id_lists(state_of(row["id"]))
    by_id = {x["id"]: (x, lst) for lst in lists for x in lst}
    out, tried = [], set()
    for c in calls:
        for v in (c.get("arguments") or {}).values():
            if not isinstance(v, str) or v not in by_id or v in tried:
                continue
            tried.add(v)
            old, lst = by_id[v]
            nkey, oname = name_of(old)
            if not oname or norm(oname) not in norm(text) or norm(oname) in hist:
                continue
            if any(isinstance(x, str) and ISO.match(x) for x in old.values()) and \
                    any(isinstance(x, str) and ISO.match(x) for c2 in calls for x in (c2.get("arguments") or {}).values()):
                continue  # times computed from the old record ("muộn 1 tiếng") would not follow the swap
            if COND.search(text.lower()) and any(k in old for k in STATEFUL):
                continue  # "tắt đèn nếu đang bật": the other device's state could change the answer
            linked = set()
            for res in results:
                for rec in records_with(res, {old["id"], oname}):
                    if rec is not old and rec.get("id") != old["id"]:
                        linked |= {str(x) for x in scalars(rec) if isinstance(x, str)}
            hist_blob = json.dumps(row["history"], ensure_ascii=False).lower()
            sibs = [x for x in lst if x["id"] != old["id"] and x.get("type") == old.get("type")
                    and x["id"].lower() not in hist_blob and name_of(x)[1].lower() not in hist_blob
                    and name_of(x)[1] and norm(name_of(x)[1]) not in norm(text) and norm(name_of(x)[1]) not in hist
                    and not (norm(name_of(x)[1]) in norm(oname) or norm(oname) in norm(name_of(x)[1]))]
            rng.shuffle(sibs)
            made = 0
            for new in sibs:
                if made >= per:
                    break
                nname = name_of(new)[1]
                said_name = nname  # in the request only: "việc dọn nhà vệ sinh", not "việc Dọn nhà vệ sinh"
                if oname[:1].islower() and nname[:1].isupper() and nkey != "name":
                    said_name = nname[:1].lower() + nname[1:]
                mapping = {str(old[k]): str(new[k]) for k in old if k in new and isinstance(old[k], str)
                           and isinstance(new[k], str) and len(old[k]) >= 2 and old[k] != new[k] and not ISO.match(old[k])}
                new_calls, bad = copy.deepcopy(calls), False
                for c2 in new_calls:
                    for key, x in (c2.get("arguments") or {}).items():
                        if isinstance(x, str):
                            if x in mapping:
                                c2["arguments"][key] = mapping[x]
                            elif oname in x:
                                c2["arguments"][key] = x.replace(oname, nname)
                            elif x in linked and x not in mapping:
                                bad = True
                        elif isinstance(x, list):
                            if any(isinstance(y, str) and y in linked and y not in mapping for y in x):
                                bad = True
                            c2["arguments"][key] = [mapping.get(y, y) if isinstance(y, str) else y for y in x]
                        elif isinstance(x, (int, float)) and not isinstance(x, bool):
                            # a number copied from the old record (a bill amount, a price) follows the new record
                            for k in old:
                                if k in new and old[k] == x and isinstance(new[k], (int, float)) and k not in ("id",):
                                    c2["arguments"][key] = new[k]
                if bad:
                    break
                new_results = replay(make_env, row["id"], new_calls)
                if not same_outcome(results, new_results) or stale_copies(calls, results, new_calls, new_results):
                    continue
                blob = json.dumps(new_calls, ensure_ascii=False)
                if old["id"] in blob or oname in blob:
                    continue
                new_text = re.sub(re.escape(oname), said_name, text, flags=re.I)
                if norm(new_text) == norm(text):
                    continue
                out.append({"kind": "entity", "list": nkey, "old": oname, "new": nname, "text": new_text,
                            "calls": new_calls, "results": new_results})
                made += 1
    return out


# ----------------------------------------------------------------------------------------------- number swaps
def spans(text: str) -> list:
    """[(start, end, value, scale)] of the numbers said in ``text`` (digits or words, with nghìn/triệu scales)."""
    out = []
    for m in SPAN.finditer(text):
        raw = m.group(1)
        words = raw.lower().split()
        if not any(w[0].isdigit() or w in NUMW.split("|")[:20] for w in words):
            continue
        if words[-1] in ("triệu", "nghìn", "ngàn") or words[-1].endswith("k"):
            pass
        if len(words) == 1 and not words[0][0].isdigit() and not UNIT_AFTER.match(text[m.end():]):
            continue  # a lone "ba"/"năm"/"không" is usually no number
        s = raw.lower().replace("ngàn", "nghìn")
        if re.fullmatch(r"\d+k", s):
            s = s[:-1] + " nghìn"
        try:
            v = parse_number(s)
        except Exception:  # noqa: BLE001
            continue
        if v is None:
            continue
        unit = UNIT_AFTER.match(text[m.end():])
        scale = 1
        if unit and unit.group(1).lower() in ("nghìn", "ngàn", "k"):
            scale = 1000
        elif unit and unit.group(1).lower() == "triệu":
            scale = 10**6
        out.append((m.start(1), m.end(1), v, scale, raw))
    return out


def roundness(v: int) -> int:
    r = 1
    while v % (r * 10) == 0 and r < 10**8:
        r *= 10
    return r


def new_value(v: int, rng: random.Random, low: int = 1) -> int:
    r = roundness(v)
    if v // r in (1, 2) and r >= 10:
        r //= 10  # 100,000 -> steps of 10,000, not only 100,000 / 200,000
    lo, hi = max(low, int(v * 0.4)), max(int(v * 2.5), v + 2 * r)
    picks = [x for x in range((lo // r + 1) * r, hi + 1, r) if x != v]
    return rng.choice(picks) if picks else v


def say_like(raw: str, v_new: int, scale: int, rng: random.Random) -> str:
    """The new value said the way ``raw`` was said (digits stay digits, words stay words)."""
    if re.fullmatch(r"\d+k", raw.lower()):
        return f"{v_new // 1000}k"
    m = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s+(nghìn|ngàn|triệu)", raw, flags=re.I)
    if m:  # "100 nghìn" stays digits + the same scale word
        unit = 10**6 if m.group(2).lower() == "triệu" else 1000
        return f"{v_new // unit} {m.group(2)}" if v_new % unit == 0 else say_number(v_new, rng)
    if raw.isdigit():
        return str(v_new // scale if scale > 1 else v_new)
    if raw.replace(",", "").replace(".", "").isdigit():
        return str(v_new // scale)
    return say_number(v_new // scale if scale > 1 else v_new, rng)


def number_swaps(row: dict, make_env, rng: random.Random, per: int, catalog: dict) -> list:
    calls, results, text = row["label"]["tool_calls"], row["label"]["tool_results"], row["audio_text"]
    if COND.search(text.lower()):
        return []  # a threshold or condition ("dưới 11 triệu thì", "đủ thì") could flip with the number
    said = spans(text)
    out, done = [], set()
    for c in calls:
        props = (catalog.get(c["name"]) or {}).get("parameters", {}).get("properties", {})
        for key, v in (c.get("arguments") or {}).items():
            if not isinstance(v, int) or isinstance(v, bool) or v < 1 or key in SKIP_KEYS or v in done:
                continue
            hits = [s for s in said if s[2] * s[3] == v or s[2] == v]
            if len(hits) != 1:
                continue
            s0, e0, val, scale, raw = hits[0]
            scale = scale if val * scale == v else 1
            done.add(v)
            lo = props.get(key, {}).get("minimum", 1)
            hi = props.get(key, {}).get("maximum")
            for _ in range(per):
                if key in RANGES:
                    a, b, step = RANGES[key]
                    picks = [x for x in range(a, b + 1, step) if x != v and abs(x - v) <= max(3 * step, v // 2)]
                    v2 = rng.choice(picks) if picks else v
                else:
                    v2 = new_value(v, rng, int(lo or 1))
                if hi is not None and v2 > hi:
                    continue
                if scale > 1 and v2 % scale:
                    v2 = (v2 // scale) * scale or scale
                if v2 == v:
                    continue
                new_calls = copy.deepcopy(calls)
                for c2 in new_calls:
                    for k2, x in (c2.get("arguments") or {}).items():
                        if isinstance(x, int) and not isinstance(x, bool) and x == v:
                            c2["arguments"][k2] = v2
                        elif isinstance(x, str) and re.search(rf"(?<!\d){v}(?!\d)", x):
                            c2["arguments"][k2] = re.sub(rf"(?<!\d){v}(?!\d)", str(v2), x)
                new_results = replay(make_env, row["id"], new_calls)
                if not same_outcome(results, new_results) or stale_copies(calls, results, new_calls, new_results):
                    continue
                phrase = say_like(raw, v2, scale, rng)
                if raw[:1].isupper():
                    phrase = phrase[:1].upper() + phrase[1:]
                new_text = text[:s0] + phrase + text[e0:]
                out.append({"kind": "number", "old": raw, "new": phrase, "value": [v, v2], "text": new_text,
                            "calls": new_calls, "results": new_results})
    return out


# ------------------------------------------------------------------------------------------------ time swaps
HOUR_WORDS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười", "mười một", "mười hai"]
HW = "|".join(sorted(HOUR_WORDS[1:], key=len, reverse=True))
CLOCK = re.compile(rf"(?<![\wÀ-ỹ:.,])(\d{{1,2}}|{HW})(\s*(?:giờ|h|g)(?![\wÀ-ỹ])|:(?=\d\d)|(?=\s+rưỡi(?![\wÀ-ỹ])))", re.I)
PERIODS = {"sáng": (5, 11), "trưa": (11, 13), "chiều": (13, 18), "tối": (18, 23), "đêm": (21, 24)}
SWAPS_TIME = OUT / "swaps_time.jsonl"


def clock_spans(text: str) -> list:
    """[(start, end, hour, raw)] of the clock times said in ``text`` ("7 giờ", "7h30", "bảy giờ", "19:00")."""
    out = []
    for m in CLOCK.finditer(text):
        raw = m.group(1)
        h = int(raw) if raw.isdigit() else HOUR_WORDS.index(raw.lower()) if raw.lower() in HOUR_WORDS else None
        if h is None or h > 23:
            continue
        out.append((m.start(1), m.end(1), h, raw))
    return out


def period_of(text: str, end: int):
    """The part of day said within a few words after (or right before) the clock time, if any."""
    after, before = text[end:end + 25].lower(), text[max(0, end - 30):end].lower()
    for w in PERIODS:
        if re.search(rf"(?<![\wÀ-ỹ]){w}(?![\wÀ-ỹ])", after) or re.search(rf"(?<![\wÀ-ỹ]){w}(?![\wÀ-ỹ])", before):
            return w
    return None


def time_swaps(row: dict, make_env, rng: random.Random, per: int) -> list:
    """A said clock hour moves to another hour of the same part of day; the ISO time arguments that equal the said
    time (and the other times of the same call, e.g. an event's end) move by the same number of hours."""
    from datetime import datetime, timedelta
    calls, results, text = row["label"]["tool_calls"], row["label"]["tool_results"], row["audio_text"]
    low = text.lower()
    if re.search(r"(?<![\wÀ-ỹ])(nếu|kém)(?![\wÀ-ỹ])", low):
        return []
    said = clock_spans(text)
    isos = [(i, k, v) for i, c in enumerate(calls) for k, v in (c.get("arguments") or {}).items()
            if isinstance(v, str) and ISO.match(v)]
    # one said clock whose hour (mod 12) no other said clock shares, matching exactly one ISO value
    unique = [x for x in said if sum(1 for y in said if y[2] % 12 == x[2] % 12) == 1]
    picks = []
    for x in unique:
        hits = [(i, k, v) for i, k, v in isos if datetime.fromisoformat(v).hour % 12 == x[2] % 12]
        if hits and len({v for _, _, v in hits}) == 1:
            picks.append((x, hits))
    if not picks or (len(said) > 1 and len(picks) != 1):
        return []
    (s0, e0, h, raw), hits = picks[0]
    if norm(text[s0:e0 + 6]) in history_text(row):
        return []
    period = period_of(text, e0)
    t_old = datetime.fromisoformat(hits[0][2])
    offset = t_old.hour - h  # 0, or 12 when "3 giờ chiều" is 15:00
    if offset not in (0, 12):
        return []
    now = datetime.fromisoformat(row["context"]["current_time"])
    lo, hi = PERIODS.get(period, (6, 22))
    choices = [x for x in range(lo, hi) if x != t_old.hour and abs(x - t_old.hour) <= 4 and (x - offset) in range(0, 24)
               and (period or (x - offset) <= 12 or raw.isdigit())]
    rng.shuffle(choices)
    out = []
    calls_with_hit = {i for i, _, _ in hits}
    for x in choices[:per * 2]:
        if len(out) >= per:
            break
        delta = timedelta(hours=x - t_old.hour)
        if (t_old > now) and (t_old + delta <= now):
            continue
        new_calls = copy.deepcopy(calls)
        for i, k, v in isos:
            if i in calls_with_hit:
                new_calls[i]["arguments"][k] = (datetime.fromisoformat(v) + delta).isoformat()
        new_results = replay(make_env, row["id"], new_calls)
        if not same_outcome(results, new_results) or stale_copies(calls, results, new_calls, new_results):
            continue
        h2 = x - offset
        phrase = str(h2) if raw.isdigit() else HOUR_WORDS[h2] if h2 < len(HOUR_WORDS) else str(h2)
        if raw[:1].isupper():
            phrase = phrase[:1].upper() + phrase[1:]
        out.append({"kind": "time", "old": raw, "new": phrase, "value": [t_old.hour, x],
                    "text": text[:s0] + phrase + text[e0:], "calls": new_calls, "results": new_results})
    return out


def prepare_time(args) -> None:
    """Time swaps for the samples that gave no entity/number swap; written next to swaps.jsonl, ids <sid>_t8_h<k>."""
    d = common.load()
    make_env = env_of()
    rng = random.Random(args.seed)
    used = {json.loads(x)["source"] for x in SWAPS.read_text(encoding="utf-8").splitlines()}
    swaps, stats = [], Counter()
    for r in d["rows"]:
        if not r["label"]["tool_calls"] or r["id"] in used:
            continue
        got = time_swaps(r, make_env, rng, args.per_time)
        for k, s in enumerate(got):
            s["id"], s["source"] = f"{r['id']}_t8_h{k}", r["id"]
            swaps.append(s)
        stats["samples"] += bool(got)
    with open(SWAPS_TIME, "w", encoding="utf-8") as f:
        for s in swaps:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(json.dumps({**stats, "swaps": len(swaps)}, ensure_ascii=False))


def source_shares(reading: str) -> dict:
    """Per unit, how the speech model wrote this sample's numbers in its averaged reading: 1.0 digits, 0.0 words."""
    out = {}
    for m in NUM_SPAN.finditer(reading):
        out.setdefault(m.group(2).lower(), []).append(m.group(1).isdigit())
    return {u: sum(v) / len(v) for u, v in out.items()}


def asr_numbers_like(text: str, rng: random.Random, shares: dict) -> str:
    """make_t6.asr_numbers with this sample's own digit habit per unit (T8 keeps the source's numbers and units)."""
    def one(m):
        num, unit = m.group(1), m.group(2)
        try:
            n = parse_number(num.lower())
        except Exception:  # noqa: BLE001
            return m.group(0)
        if n is None or n >= 1000 or (num.isdigit() and str(n) != num):
            return m.group(0)
        share = shares.get(unit.lower(), DIGIT_SHARE.get(unit.lower(), 0.0))
        said = str(n) if rng.random() < share else (num if not num.isdigit() else say_number(n, rng, unit_follows=True))
        return f"{said} {unit}"
    return NUM_SPAN.sub(one, text)


# ------------------------------------------------------------------------------------------------- commands
def prepare(args) -> None:
    d = common.load()
    make_env = env_of()
    rng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    BATCHES.mkdir(parents=True, exist_ok=True)
    swaps, stats = [], Counter()
    for r in d["rows"]:
        if not r["label"]["tool_calls"]:
            continue
        got = entity_swaps(r, make_env, rng, args.per_entity) + number_swaps(r, make_env, rng, args.per_number,
                                                                               d["catalog"])
        for k, s in enumerate(got):
            s["id"], s["source"] = f"{r['id']}_t8_{k}", r["id"]
            stats[s["kind"]] += 1
            swaps.append(s)
        stats["samples"] += bool(got)
    with open(SWAPS, "w", encoding="utf-8") as f:
        for s in swaps:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    by_id = {r["id"]: r for r in d["rows"]}
    items = []
    for s in swaps:
        r = copy.deepcopy(by_id[s["source"]])
        r["audio_text"], r["label"] = s["text"], {"tool_calls": s["calls"], "tool_results": s["results"]}
        keep = agy.anchors(r)
        if s["new"] not in keep:
            keep.append(s["new"])
        items.append({"id": s["id"], "text": s["text"], "styles": rng.sample(sorted(agy.STYLES), 2), "keep": keep})
    for f in BATCHES.glob("b*.json"):
        f.unlink()
    for b in range(0, len(items), agy.BATCH_SIZE):
        (BATCHES / f"b{b // agy.BATCH_SIZE:04d}.json").write_text(
            json.dumps(items[b:b + agy.BATCH_SIZE], ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({**stats, "swaps": len(swaps), "batches": (len(items) + 9) // 10}, ensure_ascii=False))


BATCHES2, ANSWERS2 = OUT / "batches25", OUT / "answers25"  # items left after the first account's quota, 25 per call


def run(args) -> None:
    """Items whose batch has no answer yet are regrouped 25 per call (agy adds ~9k tokens to every call)."""
    import make_t9
    BATCHES2.mkdir(parents=True, exist_ok=True)
    if not any(BATCHES2.glob("b*.json")):
        left = [it for b in sorted(BATCHES.glob("b*.json")) if not (ANSWERS / b.name).exists()
                for it in json.loads(b.read_text(encoding="utf-8"))]
        for k in range(0, len(left), 25):
            (BATCHES2 / f"b{k // 25:04d}.json").write_text(json.dumps(left[k:k + 25], ensure_ascii=False, indent=1),
                                                           encoding="utf-8")
    todo = [b for b in sorted(BATCHES2.glob("b*.json")) if not (ANSWERS2 / b.name).exists()]
    if args.limit:
        todo = todo[:args.limit]
    make_t9.run_batches(todo, ANSWERS2, lambda batch: agy.prompt_of(batch), agy.SCHEMA, args, "t8")


def collect(args) -> None:
    rows = []
    for k, (b, a) in enumerate([(BATCHES, ANSWERS), (BATCHES2, ANSWERS2)]):
        if not a.exists():
            continue
        part = OUT / f"set{k}"
        part.mkdir(exist_ok=True)
        agy.BATCHES, agy.ANSWERS, agy.OUT = b, a, part
        agy.collect(args)
        rows += (part / "paraphrases.jsonl").read_text(encoding="utf-8").splitlines()
    (OUT / "paraphrases.jsonl").write_text("".join(x + "\n" for x in rows), encoding="utf-8")
    print(f"T8 variants: {len(rows)}")


def build(args) -> None:
    d = common.load()
    review_rules.set_marker_rates(d["rows"])
    swaps = [json.loads(x) for x in SWAPS.read_text(encoding="utf-8").splitlines()]
    if SWAPS_TIME.exists():
        swaps += [json.loads(x) for x in SWAPS_TIME.read_text(encoding="utf-8").splitlines()]
    import agy_review
    para = {}
    if True:  # agy_review.passed(): lexically checked, not judged different  # only rewordings the review judged to mean the substituted request
        for (rid, style), text in sorted(agy_review.passed().items()):
            if "_t8_" in rid and rid not in para:
                para[rid] = (style, text)
    else:
        p = OUT / "paraphrases.jsonl"
        for v in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if p.exists() else []:
            if v.get("ok") and v["id"] not in para:
                para[v["id"]] = (v["style"], v["text"])
    by_id = {r["id"]: r for r in d["rows"]}
    for name, rows in common.pools(d):
        ids = {r["id"] for r in rows}
        noise = common.real_noise_from(rows, d["avg5"])  # simulated readings, as T2b/T7
        forms = common.spelling_forms(rows, d["avg5"])
        out, kinds, dropped, per_kind = [], Counter(), Counter(), Counter()
        for s in sorted(swaps, key=lambda s: random.Random(f"order:{s['id']}").random()):  # the cap keeps a random 2
            if s["source"] not in ids:
                continue
            r = by_id[s["source"]]
            style, text = para.get(s["id"], ("subst", s["text"]))
            if s["kind"] == "number":  # the new value phrase (code-made) in the original's digit form, as said
                fixed = say_like(s["old"], s["value"][1], 1, random.Random(0))
                if fixed != s["new"] and re.match(r"\d", fixed):
                    text = re.sub(re.escape(s["new"]), fixed, text, flags=re.I)
            text = common.match_case(text, [r["audio_text"]], s["calls"])  # label capitals back to the request's casing
            text = common.match_spelling(text, [r["audio_text"]], s["calls"], forms)  # hòa -> hoà
            item = copy.deepcopy(r)
            item["audio_text"] = text
            item["label"] = {"tool_calls": s["calls"], "tool_results": s["results"]}
            rng = random.Random(f"{args.seed}:{s['id']}")
            texts = {a: d["avg5"][a] for a in common.prompts.gold_transcripts(r) if a != item["audio"]}
            shares = source_shares(d["avg5"][r["audio"]])
            # a swap that leaves a number or id with no source (a stale derived value such as "237 cm (1,58 m)", a new
            # contact's id no call returned) or a Gemini rewording with overdone regional style is left out
            if common.ungrounded(item, {**texts, item["audio"]: text}) and not common.ungrounded(r, {r["audio"]: r["audio_text"]}):
                dropped["stale or unsourced value"] += 1
                continue
            why = review_rules.swap_problem(r, item, s["kind"])  # 13 Nov Opus review: label-breaking swaps
            if why:
                dropped[why] += 1
                continue
            if style != "subst":  # a Gemini rewording with a style habit falls back to the plain substitution
                text = review_rules.fix_format(text)
                hist = [d["avg5"].get(h["audio"], h.get("audio_text", "")) for h in r["history"] if h["role"] == "user"]
                habit = "overdone style" if make_t7.overdone(s["text"], text) else \
                    review_rules.style_problem(s["text"], text, item, hist)
                if habit:
                    dropped[f"rewording -> substitution ({habit})"] += 1
                    style, text = "subst", common.match_case(s["text"], [r["audio_text"]], s["calls"])
                item["audio_text"] = text
                why = review_rules.swap_problem(r, item, s["kind"])
                if why:
                    dropped[why] += 1
                    continue
            if per_kind[(s["source"], s["kind"])] >= 2:  # one source's swaps of one kind differ by one word: 2 at most
                dropped["more than 2 per source and kind"] += 1
                continue
            per_kind[(s["source"], s["kind"])] += 1
            text = text[:1].upper() + text[1:]  # third review: substitutions kept a lower-case start
            item["audio_text"] = text
            texts[item["audio"]] = noise.apply(asr_numbers_like(text, rng, shares), rng, common.protected(item))
            out.append(common.conv("T8", r, texts, item=item, swap=s["kind"], style=style))
            kinds[f"{s['kind']}/{'gemini' if style != 'subst' else 'subst'}"] += 1
        common.write(f"t8_{name}", out)
        print(name, dict(kinds), "left out:", dict(dropped))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--seed", type=int, default=81)
    p.add_argument("--per-entity", type=int, default=4)
    p.add_argument("--per-number", type=int, default=4)
    pt = sub.add_parser("prepare-time")
    pt.add_argument("--seed", type=int, default=83)
    pt.add_argument("--per-time", type=int, default=4)
    r = sub.add_parser("run")
    r.add_argument("--agy", default=str(Path.home() / ".local" / "bin" / "agy"))
    r.add_argument("--model", default=None)
    r.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--workers", type=int, default=2)
    sub.add_parser("collect")
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=82)
    args = ap.parse_args()
    {"prepare": prepare, "prepare-time": prepare_time, "run": run, "collect": collect, "build": build}[args.cmd](args)


if __name__ == "__main__":
    main()
