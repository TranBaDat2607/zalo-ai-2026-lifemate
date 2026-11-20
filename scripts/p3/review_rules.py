"""Row checks from the 13 Nov Opus review of T7/T8 (findings in log.md). Each returns a reason string or "".

Label-breaking (T8 swaps): a past action moved to the future, the old value still written in a text argument in
another surface form, an id kept from the old record's read result, an attribute of the old record still said, an
honorific clash around the new name, a number swap on a request with a direction word (giảm/tăng...: the new value can
reverse it against the current state), a mobile top-up amount no carrier sells.
Style (Gemini rewordings of T7/T8): "ạ" added for an adult speaker (real adults 0.5 %, Gemini's Bắc 41 %), stacked
particles, "xem" inserted before an action verb, a self-pronoun the speaker never uses, a dialect other than the
history's. Formatting (lowercase start, no final punctuation) is repaired, not dropped.
"""
import json
import re
from datetime import datetime

_W = re.compile(r"[^\W_]+")
HON = r"(?:anh|chị|em|cô|chú|bác|ông|bà|sếp|thầy|con|bé|thằng|cháu|dì|cậu|mợ)"
FEMALE = {"lan", "hương", "mai", "hằng", "trang", "yến", "hoa", "linh", "thảo", "ngọc", "nga", "hạnh", "my", "trâm",
          "thu", "hà", "vy", "nhung", "phương", "loan", "huyền", "hòa", "hoà"}
MALE = {"tuấn", "minh", "dũng", "hùng", "nam", "khoa", "toàn", "sơn", "hải", "phúc", "long", "quân", "việt", "thành",
        "hưng", "hoàng", "bình"}
PAST_TOOL = re.compile(r"\.(log_\w+|mark_\w+|complete)$")
DIRECTION = re.compile(r"(?<![\wÀ-ỹ])(giảm|hạ|xuống|tăng|lên|bớt|nâng)(?![\wÀ-ỹ])", re.I)
TOPUP_OK = {10000, 20000, 30000, 50000, 100000, 200000, 300000, 500000}
TRUNG = {"nớ", "mô", "rứa", "tê", "răng", "hỉ", "mần", "chừ", "ri", "nì", "hè", "nỏ", "ni", "mạ", "o"}
NAM = {"nghen", "hổng", "hông", "nhen", "bển", "dìa", "hen", "nè", "ổng", "bả", "dzậy", "zậy", "nha", "ghê", "lẹ"}
BAC = {"nhé", "đấy", "thế", "vâng", "cơ", "nhỉ", "ạ"}
KIN = {"anh", "chị", "em", "con", "cô", "chú", "bác", "cháu", "ông", "bà"}
JUNIOR = ("em", "bé", "cháu", "con")
ATTR = ("artist", "author", "relation", "provider", "merchant", "bank", "bank_name", "company", "brand", "singer", "host", "owner")


def words(t: str) -> list:
    return _W.findall(t.lower())


def _iso(v):
    return isinstance(v, str) and re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d", v)


def _number_forms(v: int) -> set:
    out = {f"{v:,}".replace(",", "."), f"{v:,}"}
    if v >= 1000 and v % 1000 == 0:
        out |= {f"{v // 1000}k", f"{v // 1000} nghìn", f"{v // 1000} ngàn"}
    if v >= 10 ** 6 and v % 10 ** 5 == 0:
        m = f"{v / 10 ** 6:g}"
        out |= {f"{m} triệu", f"{m.replace('.', ',')} triệu"}
    return {o for o in out if len(o) >= 3}


def _clock_forms(t: datetime) -> set:
    h, m = t.hour, t.minute
    out = {f"{h}:{m:02d}", f"{h:02d}:{m:02d}", f"{h}h{m:02d}" if m else f"{h}h"}
    if not m:
        out |= {f"{h} giờ", f"{h % 12 or 12} giờ"}
    return out


def swap_problem(src: dict, item: dict, kind: str) -> str:
    """Label-breaking problems of a T8 row (item) made from train row src by a swap of ``kind``."""
    now = datetime.fromisoformat(item["context"]["current_time"])
    oc, orr = src["label"]["tool_calls"], src["label"]["tool_results"]
    nc, nr = item["label"]["tool_calls"], item["label"]["tool_results"]
    text, stext = item["audio_text"], src["audio_text"]
    sargs = [v for c in nc for v in (c.get("arguments") or {}).values() if isinstance(v, str) and not _iso(v)]
    for c_old, c_new in zip(oc, nc):
        for k, v in (c_new.get("arguments") or {}).items():
            if not _iso(v):
                continue
            t = datetime.fromisoformat(v)
            ov = (c_old.get("arguments") or {}).get(k)
            if PAST_TOOL.search(c_new["name"]) and t > now:
                return "past action logged in the future"
            if _iso(ov) and ov != v:
                to = datetime.fromisoformat(ov)
                if to <= now < t:
                    return "past time moved to the future"
                if any(re.search(rf"(?<![\d:]){re.escape(f)}(?![\d])", s) for f in _clock_forms(to) for s in sargs):
                    return "old clock time left in a text argument"
    if re.search(r"(?<![\wÀ-ỹ])không giờ", text.lower()) and not re.search(r"(?<![\wÀ-ỹ])không giờ", stext.lower()):
        return "midnight said"  # third review: "chiều hôm đó ... lúc không giờ" with the label at 12:00
    if kind == "number":
        olds = {v for c_old, c_new in zip(oc, nc) for k, v in (c_old.get("arguments") or {}).items()
                if isinstance(v, int) and not isinstance(v, bool) and (c_new.get("arguments") or {}).get(k) != v}
        for v in olds:
            if any(re.search(rf"(?<![\d.,]){re.escape(f)}(?![\d])", s) for f in _number_forms(v) for s in sargs):
                return "old number left in a text argument"
        if DIRECTION.search(stext):
            return "number swap on a request with a direction word"
        for c in nc:
            if c["name"].endswith("topup_mobile") and (c.get("arguments") or {}).get("amount") not in TOPUP_OK:
                return "top-up amount no carrier sells"
    if kind == "entity":
        changed = [i for i, (a, b) in enumerate(zip(oc, nc)) if a.get("arguments") != b.get("arguments")]
        names = {}
        for res in nr + orr:
            for rec in re.findall(r"\{[^{}]*\}", json.dumps(res, ensure_ascii=False)):
                try:
                    dct = json.loads(rec)
                except Exception:  # noqa: BLE001
                    continue
                if "id" in dct:
                    for nk in ("title", "name", "label"):
                        if isinstance(dct.get(nk), str) and len(dct[nk]) >= 3:
                            names.setdefault(dct["id"], set()).add(dct[nk])
        for i in changed:
            if i >= len(orr) or i >= len(nr):
                continue
            ob, nb = json.dumps(orr[i], ensure_ascii=False), json.dumps(nr[i], ensure_ascii=False)
            for j in range(i + 1, len(nc)):
                for k, v in (nc[j].get("arguments") or {}).items():
                    if isinstance(v, str) and j < len(oc) and v == (oc[j].get("arguments") or {}).get(k) and \
                            any(nm in ob and nm not in nb for nm in names.get(v, ())):
                        return "id kept from the old record's read result"
        # an attribute of the replaced record (artist, author, relation, provider...) still said: the record whose id the
        # old calls use and the new calls do not, against the record that took its place
        old_ids = {v for c in oc for v in (c.get("arguments") or {}).values() if isinstance(v, str)}
        new_ids = {v for c in nc for v in (c.get("arguments") or {}).values() if isinstance(v, str)}
        recs = {}
        for res in orr + nr:
            for rec in re.findall(r"\{[^{}]*\}", json.dumps(res, ensure_ascii=False)):
                try:
                    dct = json.loads(rec)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(dct.get("id"), str):
                    recs[dct["id"]] = dct
        gone = [recs[i] for i in old_ids - new_ids if i in recs]
        came = [recs[i] for i in new_ids - old_ids if i in recs]
        for o in gone:
            for k in ATTR:
                ov = o.get(k)
                if isinstance(ov, str) and len(ov) >= 3 and ov.lower() in text.lower() and \
                        any(isinstance(n.get(k), str) and n.get(k) != ov for n in came):
                    return "attribute of the old record still said"
        new = [v for c in nc for v in (c.get("arguments") or {}).values()
               if isinstance(v, str) and len(v) > 1 and v in text and v not in stext]
        for nm in new:
            m = re.search(rf"(?<![\wÀ-ỹ]){HON}\s+{re.escape(nm)}", text, re.I)
            if not m:
                continue
            if re.match(rf"{HON}(?![\wÀ-ỹ])", nm, re.I):
                return "double honorific"
            h, first = m.group(0).split()[0].lower(), nm.split()[-1].lower()
            if (h in ("anh", "chú", "ông", "bác", "thằng") and first in FEMALE) or \
                    (h in ("chị", "cô", "bà", "dì") and first in MALE):
                return "honorific of the wrong gender"
    return ""


def dialect(text: str) -> str:
    """T / N / B when one region's markers dominate, else ''."""
    w = words(text)
    sc = {"T": sum(x in TRUNG for x in w), "N": sum(x in NAM for x in w), "B": sum(x in BAC for x in w)}
    best = max(sc, key=sc.get)
    return best if sc[best] and sorted(sc.values())[-2] < sc[best] else ""


def style_problem(source: str, variant: str, item: dict, history_texts: list) -> str:
    """Gemini style habits in a rewording (T7, T8 Gemini rows)."""
    v, s = words(variant), words(source)
    title = (item["context"].get("user_profile") or {}).get("name", "").split()[:1]
    title = title[0].lower() if title else ""
    if "ạ" in v and "ạ" not in s and title not in JUNIOR:
        return "ạ for an adult speaker"
    if re.search(r"(?<![\wÀ-ỹ])(nhé|nha|nghe|nghen|hen|nhen)\s+(ạ|nghe|nha|nhé|nghen)(?![\wÀ-ỹ])", variant.lower()):
        return "stacked particles"
    if re.search(r"(?<![\wÀ-ỹ])xem\s+(mở|ghi|bật|đặt|tắt|giúp|gọi|nhắn|chuyển|hủy|huỷ|lưu|thêm|tạo)", variant.lower()) \
            and "xem" not in s:
        return "xem inserted before an action"
    selves = set(re.findall(r"(?<![\wÀ-ỹ])(?:giúp|giùm|hộ|dùm)\s+([^\W\d_]+)", variant.lower())) & KIN
    allowed = set(re.findall(r"(?<![\wÀ-ỹ])(?:giúp|giùm|hộ|dùm|cho)\s+([^\W\d_]+)", source.lower()))
    allowed |= {x for t in history_texts for x in words(t)} & KIN | {title}
    if selves - allowed:
        return "self-pronoun the speaker never uses"
    hd = dialect(" ".join(history_texts))
    vd = dialect(variant)
    if hd and vd and hd != vd:
        return "dialect other than the history's"
    # second Opus review: Gemini changed the speaker's dialect in 27 % of T7 rows. Real speakers keep theirs, and it
    # follows the profile city (train: Hà Nội Trung 3/791, Cần Thơ Nam only, Huế Trung or none)
    sd = dialect(source)
    if sd and vd and sd != vd:
        return "dialect other than the source's"
    city = (item["context"].get("user_profile") or {}).get("city", "")
    if vd and vd not in CITY_DIALECTS.get(city, "BTN"):
        return "dialect the city's speakers do not use"
    if final_particles(variant) >= 2 and final_particles(variant) > final_particles(source):
        return "two closing particles"
    sv, ss = set(v) & SELF_PLAIN, set(s) & SELF_PLAIN
    if ss and sv - ss:
        return "self-pronoun replaced"
    rates = MARKER_RATES.get(city)
    if rates is not None and any(m in v and m not in s and rates.get(m, 0.0) < 0.05 for m in REGIONAL):
        return "regional word the city rarely uses"
    return ""


# third Opus review: T7 added "nghe" in Hà Nội 32 % (train 3 %), "coi" 13.5 % (0.6 %), "vô" 7 % (0.3 %)...
REGIONAL = ("nghe", "coi", "vô", "nghen", "hổng", "hông", "hỉ", "chi", "tề", "mần", "tui", "nhe", "mô", "rứa", "nớ",
            "dìa", "nhen", "hen", "trễ", "miết", "trỏng")
MARKER_RATES: dict = {}


def set_marker_rates(rows: list) -> None:
    """Share of each city's train requests that say each regional word (REGIONAL)."""
    by = {}
    for r in rows:
        by.setdefault((r["context"].get("user_profile") or {}).get("city", ""), []).append(set(words(r["audio_text"])))
    MARKER_RATES.clear()
    MARKER_RATES.update({c: {m: sum(m in w for w in ws) / len(ws) for m in REGIONAL} for c, ws in by.items() if len(ws) >= 30})


CLOSING = {"nhé", "nha", "nghe", "nghen", "hen", "nhen", "nhá", "ạ", "với", "nè", "nhe"}


def keep_ending(source: str, variant: str) -> str:
    """Third review: of the T7 sources that end with no particle, the rewording added one in 72 %. When the source's
    last word is not a closing particle, a closing particle the rewording ends with is removed."""
    if (words(source) or [""])[-1] in CLOSING:
        return variant
    m = re.search(r"[,\s]+(nhé|nha|nghe|nghen|hen|nhen|nhá|ạ|với|nè|nhe)(\s*[.?!…]?)$", variant, re.I)
    return variant[:m.start()] + (m.group(2).strip() or ".") if m else variant


CITY_DIALECTS = {"Hà Nội": "BN", "TP.HCM": "BN", "Đà Nẵng": "BTN", "Cần Thơ": "N", "Huế": "T"}
SELF_PLAIN = {"tôi", "tui", "mình", "tớ", "tao"}
_FINAL = re.compile(r"(?<![\wÀ-ỹ])(nhé|nha|nhá|nghe|nghen|hen|nhen|ạ|nhỉ|nè|với|luôn|hông|hả|ha)(?=\s*[,.!?…]|\s*$)")


def final_particles(text: str) -> int:
    """Particles closing a clause (train: two or more in 2.1 % of requests, T7 before this rule 22 %)."""
    return len(_FINAL.findall(text.lower()))


def fix_format(text: str) -> str:
    """Capital first letter and final punctuation, as every real request has."""
    text = text.strip()
    if text[:1].islower():
        text = text[0].upper() + text[1:]
    if not re.search(r"[.?!…]$", text):
        text += "."
    return text


HELPERS = ("giúp", "giùm", "dùm", "hộ")
HELPER_OBJ = "tôi|tui|mình|tớ|em|anh|chị|con|cô|chú|bác|cháu|bà|ông|ba|mẹ|bố|má"


CITY_HELPERS = {"TP.HCM": {"giúp": 40, "dùm": 34, "giùm": 68, "hộ": 4}, "Đà Nẵng": {"giúp": 43, "dùm": 5, "giùm": 51, "hộ": 8},
                "Hà Nội": {"giúp": 249, "dùm": 5, "giùm": 26, "hộ": 37}, "Cần Thơ": {"dùm": 7, "giùm": 28},
                "Huế": {"giúp": 3, "giùm": 32}}


def keep_helper(source: str, variant: str, city: str = "", rng=None) -> str:
    """The speaker's own "help" word (second Opus review: Gemini turns giúp into giùm, T7 39 % vs train 13 %). When the
    source uses exactly one of giúp / giùm / dùm / hộ, the rewording's other ones become it (same meaning)."""
    have = {h for h in HELPERS if h in words(source)}
    if len(have) > 1 or not any(h in words(variant) for h in HELPERS):
        return variant
    if have:
        own = have.pop()
    else:
        # the source has none: Gemini added "giúp tôi" (T7 82 % of rows with a helper word, train 41 %); drop the
        # helper with its pronoun; a bare one left ("với", no pronoun) takes the city's own mix (train counts)
        variant = re.sub(rf"(?<![\wÀ-ỹ])(?:giúp|giùm|dùm|hộ)\s+(?:{HELPER_OBJ})(?![\wÀ-ỹ])\s*", "", variant, flags=re.I)
        variant = re.sub(r"\s+([,.!?…])", r"\1", variant).strip()
        variant = variant[:1].upper() + variant[1:]
        if not any(h in words(variant) for h in HELPERS):
            return variant
        mix = CITY_HELPERS.get(city, CITY_HELPERS["Hà Nội"])
        own = rng.choices(list(mix), list(mix.values()))[0] if rng else max(mix, key=mix.get)
    def one(m):
        w = m.group(0)
        return (own.capitalize() if w[0].isupper() else own) if w.lower() != own else w
    return re.sub(r"(?<![\wÀ-ỹ])(?:giúp|giùm|dùm|hộ)(?![\wÀ-ỹ])", one, variant, flags=re.I)
