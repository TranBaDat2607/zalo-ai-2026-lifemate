"""Data gate of the time/number drills (improve_plan.md step 4; rules in log.md, fixed before the data was built).

    python scripts/drills/drill_gate.py DRILLS.jsonl [--holdout splits] [--readings outputs/readings/avg5_train.json]

The pool is the training rows outside the holdout (fold 0: the 386 rows of splits.json). Everything is measured on the
pool only; the holdout's texts are used for one thing, the leakage check D7. Every gate must pass; on a failure the
generator is improved and the data rebuilt (the step is not dropped on a data failure).

D1 correctness  every recorded phrase reads back to its value (drills.verify), and the readers agree with train gold
                on real pool phrases: clock phrases >= 95 % on gold text, >= 90 % on readings; money the same
D2 coverage     >= 95 % of the in-scope gold argument signatures seen >= 2 times in the pool have >= 3 drills
D3 forms        every clock / day / money form seen >= 2 times in pool readings has a drill share >= 0.5 x its pool
                share within its family
D4 words        >= 97 % of drill words are pool words (gold text + readings); no app id or tool name in a drill
D5 style        politeness markers: |drill share - pool share| <= 5 points per marker; digits before each unit:
                |drill share - pool share| <= 10 points (units with >= 20 pool cases)
D6 context      total variation distance of the context hours (pack times vs pool times) <= 0.1
D7 leakage      no 8-word sequence of a drill appears in a holdout text (gold or reading)
D8 conventions  each convention the drills use agrees with >= 90 % of the pool's gold cases (with fewer than 4 cases:
                no case against it)
"""
import argparse
import ast
import collections
import json
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lifemate import drills  # noqa: E402

WORD = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[^\W\d_]+", re.UNICODE)
NUMWORDS = set(drills.UNITS) | {"mười", "mươi", "mốt", "tư", "lăm", "nhăm", "linh", "lẻ", "trăm", "rưỡi", "chục"}
PARTS = ("sáng", "trưa", "chiều", "tối", "đêm")
MARKERS = ["nhé", "nha", "nghe", "nghen", "hen", "hỉ", "ạ", "giùm", "dùm", "hộ", "giúp", "với"]
UNITS = ["giờ", "phút", "tiếng", "nghìn", "ngàn", "triệu", "độ", "phần trăm", "chỗ"]
DATETIME_PARAMS = {"time", "start", "end", "schedule_time", "due", "valid_until", "since", "delivery_time", "from_time"}
DATE_PARAMS = {"date", "date_from", "date_to"}
NUMBER_PARAMS = {"amount", "temperature", "minutes", "delay_minutes", "hours", "passengers", "seats", "threshold_kwh",
                 "term_months", "brightness", "quantity", "price", "max_price"}
# a stated value is not the gold value when the request computes one from it ("dời muộn một tiếng", "trước hai tiếng",
# "cộng lại", "đổi ra"): such rows are left out of the golden tests
CUES = re.compile(r"\b(?:trước|sau|sớm|muộn|trễ|lùi|dời|cộng|trừ|nữa|chia|mỗi|đổi|quy|tính|hơn|dưới|trên|bằng|ấy|đó)\b")
MONEY_CUES = re.compile(r"\b(?:cộng|trừ|chia|mỗi|đổi|quy|đô|yên|tỷ giá)\b")
PAY = re.compile(r"\b(?:chuyển|nạp|gửi|trả|đóng|góp|thanh toán|hoàn)\b")
THRESH = re.compile(r"\b(?:trên|dưới|hơn|quá|tới|đủ|còn)\b")
WEEKDAYS = {w: i for i, w in enumerate(drills.WEEKDAY_WORDS)} | {f"thứ {i + 2}": i for i in range(6)} | {"cn": 6}


def words(text: str) -> list:
    return [w.lower() for w in WORD.findall(text)]


def is_num(w: str) -> bool:
    return w in NUMWORDS or w.replace(",", "").replace(".", "").isdigit()


# ------------------------------------------------------------------------------------------------------- data

def load_pool(holdout: str, readings_path: str):
    root = Path(os.environ.get("LIFEMATE_DATA", REPO / "train+public_test/train+public_test")) / "train"
    rows = [json.loads(line) for line in open(root / "train.jsonl", encoding="utf-8")]
    from lifemate import data as ldata
    held = ldata.id_set(holdout)  # "splits" (fold 0) or "cv:K", the ids train.py holds out
    readings = json.loads(Path(readings_path).read_text(encoding="utf-8"))
    for r in rows:
        r["reading"] = readings.get(r["audio"], "")
        label = r["label"]
        r["calls"] = (ast.literal_eval(label) if isinstance(label, str) else label)["tool_calls"]
        r["now"] = datetime.fromisoformat(r["context"]["current_time"])
    pool = [r for r in rows if r["id"] not in held]
    hold = [r for r in rows if r["id"] in held]
    catalog = json.loads((root / "train_tools_catalog.json").read_text(encoding="utf-8"))
    return pool, hold, catalog


def gold_args(r):
    for call in r["calls"]:
        for k, v in (call.get("arguments") or {}).items():
            yield call["name"], k, v


# ------------------------------------------------------------------------------------------------- signatures

def offset_bucket(d: int) -> str:
    return {0: "0", 1: "+1", 2: "+2", -1: "-1"}.get(d, "+3..6" if 3 <= d <= 6 else "+7..13" if 7 <= d <= 13 else "other")


def signature(param: str, value, now: datetime):
    """In-scope argument -> its signature, or None."""
    if param in DATETIME_PARAMS and isinstance(value, str):
        if re.fullmatch(r"\d\d:\d\d", value):
            return ("hh:mm", drills.part_of_day(int(value[:2])))
        try:
            t = datetime.fromisoformat(value)
        except ValueError:
            return None
        return ("datetime", offset_bucket((t.date() - now.date()).days), drills.part_of_day(t.hour))
    if param in DATE_PARAMS and isinstance(value, str):
        try:
            d = date.fromisoformat(value[:10])
        except ValueError:
            return None
        return ("date", offset_bucket((d - now.date()).days))
    if param in NUMBER_PARAMS and isinstance(value, (int, float)) and not isinstance(value, bool):
        mag = next(b for b in (10, 100, 1000, 10**4, 10**5, 10**6, 10**7, 10**12) if abs(value) < b)
        return ("number", param, mag)
    return None


# ------------------------------------------------------------------------------------------------------- forms

def clock_forms(text: str) -> list:
    """Form of each clock phrase: a number (word or digits) followed by "giờ", or "<num> rưỡi" after a part of day."""
    w, out = words(text), []
    for i, x in enumerate(w):
        if x != "giờ" or i == 0 or not is_num(w[i - 1]):
            continue
        if i >= 2 and w[i - 2] in ("mấy", "bao"):
            continue
        nxt = w[i + 1] if i + 1 < len(w) else ""
        digits = w[i - 1].isdigit()
        if nxt == "rưỡi":
            out.append("rưỡi")
        elif nxt == "kém":
            out.append("kém")
        elif digits:
            out.append("digits")
        elif is_num(nxt):
            out.append("giờ+phút")
        elif nxt in ("tiếng", "đồng"):
            continue
        else:
            out.append("giờ")
    for i, x in enumerate(w[:-1]):  # "bảy rưỡi tối"
        if x == "rưỡi" and is_num(w[i - 1]) and i >= 1 and (i < 2 or w[i - 2] != "giờ") and \
                ((i + 1 < len(w) and w[i + 1] in PARTS) or (i >= 2 and w[i - 2] in PARTS)):
            out.append("rưỡi")
    return out


def day_forms(text: str) -> list:
    t = " ".join(words(text))
    out = []
    for pat, form in ((r"\b(?:ngày |sáng |trưa |chiều |tối )?mai\b", "tomorrow"), (r"\bmốt\b|\bngày kia\b", "day after"),
                      (r"\bhôm nay\b|\bbữa nay\b|\bhôm ni\b|\b(?:sáng|trưa|chiều|tối) (?:nay|ni)\b", "today"),
                      (r"\bhôm qua\b|\bbữa qua\b|\b(?:sáng|trưa|chiều|tối) qua\b", "yesterday"),
                      (r"\b(?:thứ (?:hai|ba|tư|năm|sáu|bảy|[2-7])|chủ nhật) (?:tuần )?(?:sau|tới)\b", "next week"),
                      (r"\b(?:thứ (?:hai|ba|tư|năm|sáu|bảy|[2-7])|chủ nhật)\b(?! (?:tuần )?(?:sau|tới))", "weekday"),
                      (r"\b(?:ngày|hôm|bữa) (?:\d+|mười|hai mươi|ba mươi)(?: \w+)?(?: tháng)", "day+month"),
                      (r"\b(?:ngày|hôm|bữa) (?:\d+|mười|hai mươi|ba mươi)\b(?! tháng)(?! \w+ tháng)", "day of month")):
        out += [form] * len(re.findall(pat, t))
    return out


def money_forms(text: str) -> list:
    w, out = words(text), []
    for i, x in enumerate(w):
        if x in ("nghìn", "ngàn", "triệu", "k", "tỷ") and i and is_num(w[i - 1]):
            if x == "triệu" and i + 1 < len(w) and w[i + 1] == "rưỡi":
                out.append("triệu rưỡi")
            elif w[i - 1].replace(",", "").isdigit():
                out.append("digits")
            elif x == "k":
                out.append("k")
            elif x == "tỷ":
                out.append("tỷ")
            elif x == "triệu":
                out.append("triệu")
            elif i >= 2 and w[i - 1] in ("chục", "mươi"):
                out.append("chục nghìn")
            else:
                out.append("nghìn")
    return out


def family_shares(texts, fn):
    c = collections.Counter(f for t in texts for f in fn(t))
    n = sum(c.values()) or 1
    return c, {k: v / n for k, v in c.items()}


# ---------------------------------------------------------------------------------------------- golden tests

def phrase_runs(text: str):
    """Maximal runs of number words/digits around a unit word: (run words, unit, index)."""
    w = words(text)
    i = 0
    while i < len(w):
        if is_num(w[i]):
            j = i
            while j < len(w) and (is_num(w[j]) or w[j] in ("nghìn", "ngàn", "triệu") or
                                  (w[j] == "giờ" and j + 1 < len(w) and (is_num(w[j + 1]) or w[j + 1] == "kém")) or
                                  w[j] == "kém"):
                j += 1
            yield w[i:j], w[max(0, i - 6):i], w[j - 1] if j > i else ""
            i = j
        i += 1


def golden_clock(rows, field):
    """Pool rows whose text has exactly one clock phrase and whose gold datetimes share one hour:minute."""
    agree = total = 0
    misses = []
    for r in rows:
        text = r[field]
        w = words(text)
        hm = {(datetime.fromisoformat(v).hour, datetime.fromisoformat(v).minute) if "T" in v else (int(v[:2]), int(v[3:5]))
              for _, k, v in gold_args(r) if k in DATETIME_PARAMS and isinstance(v, str) and
              (re.fullmatch(r"\d\d:\d\d", v) or re.match(r"\d{4}-\d\d-\d\dT", v))}
        phrases = []
        for i, x in enumerate(w):
            if x == "giờ" and i and is_num(w[i - 1]) and (i < 2 or w[i - 2] not in ("mấy", "bao")):
                a = i - 1
                while a > 0 and is_num(w[a - 1]) and len(w[a - 1:i]) <= 2:
                    a -= 1
                b = i + 1
                if b < len(w) and w[b] in ("tiếng", "đồng"):
                    continue
                if b < len(w) and w[b] == "rưỡi":
                    b += 1
                else:
                    if b < len(w) and w[b] == "kém":
                        b += 1
                    while b < len(w) and is_num(w[b]) and w[b] != "rưỡi" and b - i <= 3:
                        b += 1
                    if b < len(w) and w[b] == "phút":
                        b += 1
                phrases.append((" ".join(w[a:b]), a, b))
        if len(phrases) != 1 or len(hm) != 1 or CUES.search(text.lower()):
            continue
        p, a, b = phrases[0]
        near = [x for x in w[max(0, a - 4):a] + w[b:b + 4] if x in PARTS + ("khuya",)]
        part = near[0] if near else None
        want = next(iter(hm))
        try:
            got = drills.parse_clock(p, part) if part else None
            if part is None:
                h, m = drills.parse_clock(p, "sáng")
                got = (h, m) if (h, m) == want or ((h + 12) % 24, m) != want else ((h + 12) % 24, m)
        except Exception:  # noqa: BLE001
            got = None
        total += 1
        agree += got == want
        if got != want:
            misses.append({"id": r["id"], "phrase": p, "part": part, "gold": want, "read": got})
    return agree, total, misses


def golden_money(rows, field):
    """Money phrases said right after a paying verb ("chuyển hai triệu", "nạp năm mươi nghìn") in rows without a
    computing cue: each must read as one of the row's gold amounts."""
    agree = total = 0
    misses = []
    for r in rows:
        amounts = sorted(v for _, k, v in gold_args(r) if k == "amount" and isinstance(v, (int, float)))
        if not amounts or MONEY_CUES.search(r[field].lower()):
            continue
        runs = [run for run, before, last in phrase_runs(r[field]) if any(u in run for u in ("nghìn", "ngàn", "triệu"))
                and PAY.search(" ".join(before)) and not THRESH.search(" ".join(before[-3:]))]
        for run in runs:
            try:
                got = drills.parse_number(" ".join(run))
            except Exception:  # noqa: BLE001
                got = None
            total += 1
            agree += got in amounts
            if got not in amounts:
                misses.append({"id": r["id"], "phrase": " ".join(run), "gold": amounts, "read": got})
    return agree, total, misses


# ------------------------------------------------------------------------------------------------- conventions

CURRENCIES = ("đô", "usd", "ơ", "euro", "yên", "won", "bảng", "tệ", "bạt")


def conventions(pool):
    """Gold agreement of the conventions the drills use, measured on gold text."""
    out = {}

    def dates_of(r):
        ds = set()
        for _, k, v in gold_args(r):
            if (k in DATETIME_PARAMS or k in DATE_PARAMS) and isinstance(v, str) and re.match(r"\d{4}-\d\d-\d\d", v):
                ds.add(date.fromisoformat(v[:10]))
        return ds

    def check(name, cases):
        out[name] = {"agree": sum(cases), "cases": len(cases)}

    weekday, next_week, mai, mot, bare = [], [], [], [], []
    for r in pool:
        t = " ".join(words(r["audio_text"]))
        ds, today = dates_of(r), r["now"].date()
        found = re.findall(r"\b(thứ (?:hai|ba|tư|năm|sáu|bảy|[2-7])|chủ nhật)( (?:tuần )?(?:sau|tới))?", t)
        other_day = re.search(r"\bmai\b|\bmốt\b|ngày kia|\bnay\b|\bqua\b|ngày \w+ tháng|\bhôm \w+\b", t)
        if len(found) == 1 and ds and not other_day:
            wd = WEEKDAYS[found[0][0]]
            nxt = today + timedelta(days=(wd - today.weekday()) % 7 or 7)
            if not found[0][1] and wd != today.weekday() and wd != 6:  # the drills never use bare "chủ nhật"
                weekday.append(nxt in ds)
            elif found[0][1] and wd < today.weekday():
                monday = today + timedelta(days=7 - today.weekday())
                next_week.append(monday + timedelta(days=wd) in ds)
        if re.search(r"\bmai\b", t) and not re.search(r"\bmốt\b|ngày kia|thứ|chủ nhật|tuần|ngày \d|hôm qua", t) and ds:
            mai.append(today + timedelta(days=1) in ds)
        tm = re.sub(r"\b(?:hai|ba) mốt\b", "", t)
        if re.search(r"\bmốt\b|\bngày kia\b", tm) and not re.search(r"thứ|chủ nhật|tuần|ngày \d|\bmai\b", tm) and ds:
            mot.append(today + timedelta(days=2) in ds)
        amounts = [v for _, k, v in gold_args(r) if k == "amount" and isinstance(v, (int, float))]
        w = words(r["audio_text"])
        for i, x in enumerate(w):
            if x == "trăm" and amounts and not any(u in w[i + 1:i + 4] for u in ("nghìn", "ngàn", "triệu", "k", "đồng", "tỷ")
                                                   ) and not any(u in w[i + 1:i + 7] for u in CURRENCIES) \
                    and not any(u in w[max(0, i - 4):i] for u in ("triệu", "nghìn", "ngàn")):
                a = i - 1
                while a > 0 and is_num(w[a - 1]):
                    a -= 1
                run = w[a:i + 3]
                run = run[:next((j for j, y in enumerate(run) if j > i - a and not is_num(y)), len(run))]
                try:
                    k = drills.parse_number(" ".join(run))
                except Exception:  # noqa: BLE001
                    continue
                if k < 1000:
                    bare.append(k * 1000 in amounts)
    check("thứ X = next such day", weekday)
    check("thứ X tuần sau (before today's weekday) = next Monday-start week", next_week)
    check("mai = +1", mai)
    check("mốt / ngày kia = +2", mot)
    check("bare hundreds = thousands", bare)
    return out


# ------------------------------------------------------------------------------------------------------- main

def run_gate(packs, pool, hold, catalog):
    ds = [d for p in packs for d in p["drills"]]
    texts = [d["text"] for d in ds]
    readings = [r["reading"] for r in pool if r["reading"]]
    golds = [r["audio_text"] for r in pool]
    report, ok = {"drills": len(ds), "packs": len(packs), "pool": len(pool)}, {}

    # D1
    bad = [(d["text"], drills.verify(d)) for d in ds if drills.verify(d)]
    gc_g, gc_r = golden_clock(pool, "audio_text"), golden_clock(pool, "reading")
    gm_g, gm_r = golden_money(pool, "audio_text"), golden_money(pool, "reading")
    share = lambda a: a[0] / a[1] if a[1] else 0  # noqa: E731
    report["D1"] = {"verify failures": len(bad), "examples": bad[:5],
                    "clock gold": gc_g[:2], "clock readings": gc_r[:2], "money gold": gm_g[:2], "money readings": gm_r[:2],
                    "clock gold misses": gc_g[2][:8], "money gold misses": gm_g[2][:8],
                    "clock reading misses": gc_r[2][:5], "money reading misses": gm_r[2][:5]}
    ok["D1"] = (not bad and share(gc_g) >= .95 and share(gm_g) >= .95 and share(gc_r) >= .90 and share(gm_r) >= .90
                and min(gc_g[1], gm_g[1]) >= 20)

    # D2
    pool_sig = collections.Counter(s for r in pool for _, k, v in gold_args(r) if (s := signature(k, v, r["now"])))
    drill_sig = collections.Counter(s for d in ds for k, v in d["target"].items()
                                    if (s := signature(k, v, datetime.fromisoformat(d["now"]))))
    wanted = {s: n for s, n in pool_sig.items() if n >= 2}
    missing = {str(s): n for s, n in wanted.items() if drill_sig[s] < 3}
    covered = 1 - sum(1 for _ in missing) / max(1, len(wanted))
    report["D2"] = {"pool signatures (>= 2)": len(wanted), "covered": round(covered, 3),
                    "missing (pool count)": dict(sorted(missing.items(), key=lambda kv: -kv[1])[:25])}
    ok["D2"] = covered >= .95

    # D3
    d3 = {}
    good = True
    for name, fn, fam in (("clock", clock_forms, None), ("day", day_forms, None), ("money", money_forms, None)):
        pc, ps = family_shares(readings, fn)
        dc, dsh = family_shares(texts, fn)
        rows = {}
        for form, n in pc.items():
            if n < 2:
                continue
            fine = dsh.get(form, 0) >= .5 * ps[form]
            good &= fine
            rows[form] = {"pool": n, "pool share": round(ps[form], 3), "drill share": round(dsh.get(form, 0), 3),
                          "ok": fine}
        d3[name] = rows
    report["D3"] = d3
    ok["D3"] = good

    # D4
    vocab = {w for t in readings + golds for w in words(t)}
    for r in pool:
        vocab |= {w for h in (json.loads(r["history"]) if isinstance(r["history"], str) and r["history"].startswith("[")
                              else []) if isinstance(h, dict) for w in words(json.dumps(h, ensure_ascii=False))}
    dw = [w for t in texts for w in words(t)]
    oov = collections.Counter(w for w in dw if w not in vocab and not w.isdigit())
    names = {a["id"] for a in catalog["apps"]} | {t["name"] for a in catalog["apps"] for t in a["tools"]}
    names |= {n.split(".")[-1] for n in list(names) if "." in n and "_" in n.split(".")[-1]}
    leaked_names = sorted({n for n in names for t in texts if n in t})
    data_names = sorted({n for n in names if "." in n for d in ds if n in json.dumps(d.get("data", {}))})
    in_vocab = 1 - sum(oov.values()) / max(1, len(dw))
    report["D4"] = {"in pool vocabulary": round(in_vocab, 4), "out of vocabulary": oov.most_common(25),
                    "app/tool names in text": leaked_names[:10], "tool names in data": data_names[:10]}
    ok["D4"] = in_vocab >= .97 and not leaked_names

    # D5
    def marker_share(ts):
        return {m: sum(bool(re.search(rf"(?<!\w){m}(?!\w)", t.lower())) for t in ts) / max(1, len(ts)) for m in MARKERS}

    pm, dm = marker_share(readings), marker_share(texts)
    markers = {m: {"pool": round(pm[m], 3), "drills": round(dm[m], 3), "ok": abs(pm[m] - dm[m]) <= .05} for m in MARKERS}

    def digit_share(ts):
        out = {}
        for unit in UNITS:
            uw = unit.split()
            tot = dig = 0
            for t in ts:
                w = words(t)
                for i in range(1, len(w) - len(uw) + 1):
                    if w[i:i + len(uw)] == uw and is_num(w[i - 1]):
                        if unit == "giờ" and i + 1 < len(w) and w[i + 1] in ("tiếng", "đồng"):
                            continue
                        tot += 1
                        dig += w[i - 1].replace(",", "").isdigit()
            out[unit] = (dig, tot)
        return out

    pdg, ddg = digit_share(readings), digit_share(texts)
    digits = {}
    for unit in UNITS:
        (pd_, pt), (dd, dt) = pdg[unit], ddg[unit]
        if pt < 20:
            continue
        a, b = pd_ / pt, dd / dt if dt else None
        digits[unit] = {"pool": f"{pd_}/{pt} = {a:.2f}", "drills": f"{dd}/{dt} = {b:.2f}" if dt else "none",
                        "ok": b is not None and abs(a - b) <= .10}
    report["D5"] = {"markers": markers, "digits": digits}
    ok["D5"] = all(v["ok"] for v in markers.values()) and all(v["ok"] for v in digits.values())

    # D6
    ph = collections.Counter(r["now"].hour for r in pool)
    dh = collections.Counter(datetime.fromisoformat(p["now"]).hour for p in packs for _ in p["drills"])
    tv = .5 * sum(abs(ph[h] / len(pool) - dh[h] / max(1, len(ds))) for h in range(24))
    report["D6"] = {"total variation": round(tv, 3)}
    ok["D6"] = tv <= .1

    # D7
    grams = set()
    for r in hold:
        for t in (r["audio_text"], r["reading"]):
            w = words(t)
            grams |= {tuple(w[i:i + 8]) for i in range(len(w) - 7)}
    hits = [t for t in texts if any(tuple(words(t)[i:i + 8]) in grams for i in range(len(words(t)) - 7))]
    report["D7"] = {"drills with a holdout 8-gram": len(hits), "examples": hits[:5]}
    ok["D7"] = not hits

    # D8
    conv = conventions(pool)
    report["D8"] = conv
    ok["D8"] = all(v["agree"] / v["cases"] >= .9 if v["cases"] >= 4 else v["agree"] == v["cases"] for v in conv.values())

    report["gates"] = ok
    report["decision"] = "PASS" if all(ok.values()) else "FAIL: improve the data"
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("drills_file")
    ap.add_argument("--holdout", default="splits")
    ap.add_argument("--readings", default=str(REPO / "outputs/readings/avg5_train.json"))
    ap.add_argument("--out")
    args = ap.parse_args()
    pool, hold, catalog = load_pool(args.holdout, args.readings)
    packs = [json.loads(line) for line in open(args.drills_file, encoding="utf-8")]
    report = run_gate(packs, pool, hold, catalog)
    text = json.dumps(report, indent=1, ensure_ascii=False, default=str)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
