"""Shared helpers of the plan 3 generators (scripts/p3/make_*.py): data, protected words, hearing noise, output files.

Every generator builds two files: <name>_full.jsonl from all of train, and <name>_fold0.jsonl from train without the 386
validation samples, with its hearing-noise table also learned without them. Rows are train --extra rows
({kind: conv, type, source, item, texts}); outputs/ is gitignored because they are derived from competition data.
"""
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lifemate import data, prompts  # noqa: E402
from lifemate.noise import ConfusionNoise  # noqa: E402

OUT = REPO / "outputs" / "p3"
READINGS = REPO / "outputs" / "readings"
NUM_WORDS = {"không", "một", "mốt", "hai", "ba", "bốn", "tư", "năm", "lăm", "nhăm", "sáu", "bảy", "tám", "chín", "mười",
             "mươi", "trăm", "nghìn", "ngàn", "triệu", "tỷ", "tỉ", "rưỡi", "nửa", "linh", "lẻ", "chục"}
_WORD = re.compile(r"[^\W_]+", re.U)


def load() -> dict:
    root = data.find_root("train")
    return {"rows": data.load_split("train", root), "catalog": data.load_catalog("train", root),
            "hold": data.id_set("splits"),
            "avg5": json.loads((READINGS / "avg5_train.json").read_text(encoding="utf-8"))}


def pools(d: dict) -> list:
    """[("full", all rows), ("fold0", rows outside the validation set)]."""
    return [("full", d["rows"]), ("fold0", [r for r in d["rows"] if r["id"] not in d["hold"]])]


def bare(word: str) -> str:
    """The word without tone marks and with đ as d."""
    s = "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def noise_from(rows: list, avg5: dict) -> ConfusionNoise:
    """Word confusions of the averaged reading against the gold text, on these rows' clips only. Only one-for-one
    substitutions (a replaced span as long as the span heard) and single dropped words count: longer replaced spans
    are aligned arbitrarily and would pair unrelated words ("giảm" -> "nhắn"); a pair also has to sound alike (half the
    letters shared once tone marks are gone: "muộn" -> "mượn", "vàng" -> "bằng")."""
    import difflib
    table, seen, errors, total = {}, {}, 0, 0
    for r in rows:
        for a, ref in prompts.gold_transcripts(r).items():
            if a not in avg5:
                continue
            x, y = [w.lower() for w in _WORD.findall(unicodedata.normalize("NFC", ref))], \
                   [w.lower() for w in _WORD.findall(unicodedata.normalize("NFC", avg5[a]))]
            total += len(x)
            for w in x:
                seen[w] = seen.get(w, 0) + 1
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=x, b=y, autojunk=False).get_opcodes():
                if op == "replace" and i2 - i1 == j2 - j1:
                    for u, v in zip(x[i1:i2], y[j1:j2]):
                        if difflib.SequenceMatcher(a=bare(u), b=bare(v)).ratio() >= 0.5:  # sounds alike
                            table.setdefault(u, []).append(v)
                            errors += 1
                elif op == "delete" and i2 - i1 == 1:
                    table.setdefault(x[i1], []).append("")
                    errors += 1
    return ConfusionNoise(table, seen, errors / max(total, 1))


def protected(row: dict) -> set:
    """Words that carry exact argument values: words of string arguments said as such, digits, number words."""
    low = row["audio_text"].lower()
    keep = set(NUM_WORDS)
    for call in row["label"]["tool_calls"]:
        for v in (call.get("arguments") or {}).values():
            if isinstance(v, str) and v.lower() in low:
                keep |= {w.lower() for w in _WORD.findall(v)}
            if isinstance(v, str) and re.match(r"\d{4}-\d\d-\d\dT", v):
                keep |= TIME_WORDS | OFFSET_WORDS  # 13 Nov review: "tối mai" -> "tối mười", "hai tiếng" -> "hai tiền"
            if isinstance(v, int) and not isinstance(v, bool):
                keep |= OFFSET_WORDS  # second review: "lên thêm hai nghìn" -> "lên hẹn": the value is 8500 + 2000
    return keep


def value_lost(item: dict, texts: dict, catalog: dict) -> str:
    """An exactly scored value the request says that the shown texts, context, history and earlier results no longer
    carry: an integer argument said in the request, or the clock hour of a time argument. '' when none."""
    calls, res = item["label"]["tool_calls"], item["label"]["tool_results"]
    said = _said(item["audio_text"])
    # not the context: its timestamps put every small number in the pool
    base = _said(" ".join(texts.values())) | _nums(json.dumps([h.get("tool_calls") for h in item["history"]],
                                                              ensure_ascii=False))
    for i, c in enumerate(calls):
        pool = base | _nums(json.dumps(res[:i], ensure_ascii=False))
        for k, v in (c.get("arguments") or {}).items():
            if isinstance(v, int) and not isinstance(v, bool) and str(v) in said and str(v) not in pool:
                return f"{c['name']}.{k}={v}"
            if isinstance(v, str) and re.match(r"\d{4}-\d\d-\d\dT(\d\d)", v):
                h = int(v[11:13])
                hs = {str(h), str(h % 12 or 12)}
                if hs & said and not hs & pool:
                    return f"{c['name']}.{k} hour {h}"
    return ""


QUESTION = re.compile(r"(bao nhiêu|mấy giờ|có\b.*\bkhông\W*$|chưa\W*$|ở đâu|thế nào|ra sao|nào không)", re.I)
# avg5 ends a question-like history turn with "?" 94 % (703 train turns). For the latest request a gold "." becomes "?"
# only 4 % (3 of 72, final check 13 Nov); the 21 % overall comes from golds that already end in "?". Templates and
# Gemini texts (T6, T9) never write "?", so they take the overall 18 %.
QUESTION_MARK = 0.04
QUESTION_MARK_SYNTH = 0.18
QUESTION_MARK_HISTORY = 0.94


def asr_question(text: str, rng: random.Random, rate: float = QUESTION_MARK) -> str:
    """Third Opus review: the speech model writes "?" after a question; simulated readings kept the gold "."."""
    if QUESTION.search(text) and re.search(r"[.…]$", text) and rng.random() < rate:
        return re.sub(r"[.…]$", "?", text)
    return text


def tidy_punct(text: str) -> str:
    """After a dropped word: no " ,", no ", ,", no text opening with punctuation; capital first letter."""
    out = " ".join(text.split())
    out = re.sub(r"\s+([,.!?…])", r"\1", out)
    out = re.sub(r"([,.!?…])(?:\s*[,])+", r"\1", out)
    out = re.sub(r"^[\s,.;:!?…]+", "", out)
    return out[:1].upper() + out[1:]


def noisy(text: str, noise: ConfusionNoise, rng: random.Random, keep: set) -> str:
    """``text`` as the ASR might hear it, leaving protected words and digits alone."""
    text = unicodedata.normalize("NFC", text)

    def one(m):
        w = m.group()
        return w if w.lower() in keep or any(c.isdigit() for c in w) else noise.word(w, rng)
    return asr_question(tidy_punct(_WORD.sub(one, text)), rng)


def conv(kind: str, row: dict, texts: dict, item: dict = None, **extra) -> dict:
    return {"kind": "conv", "type": kind, "source": row["id"], **extra, "item": item or row, "texts": texts}


def write(name: str, rows: list) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} rows", flush=True)


# ------------------------------------------------------------------------- realistic hearing noise (13 Nov review)
# The first noise table's commonest confusions changed nothing a model needs to undo: particles swapped for each other
# (nhé/nha, nghe/nha), the same syllable with its tone mark placed elsewhere (huỷ/hủy, khoá/khóa), tôi/tui, dùm/giùm,
# dropped fillers; and it drew errors word by word, so 4+ errors per request (17.5 % of real latest readings) were rare
# (1.5 %). RealNoise keeps the confusions that change a word and draws the number of errors per request from the real
# readings' distribution of such errors.
NUMBER_WORDS = {"không", "một", "mốt", "hai", "ba", "bốn", "tư", "năm", "lăm", "sáu", "bảy", "tám", "chín", "mười",
                "mươi", "trăm", "nghìn", "ngàn", "triệu", "linh", "lẻ", "rưỡi"}
OFFSET_WORDS = {"tiếng", "trước", "sau", "sớm", "muộn", "trễ", "thêm", "bớt", "kém", "lùi", "hơn", "nữa", "lên", "xuống",
                "giảm", "tăng"}
TIME_WORDS = {"sáng", "trưa", "chiều", "tối", "đêm", "khuya", "mai", "nay", "mốt", "kia", "qua", "giờ", "phút", "thứ",
              "tuần", "tháng", "ngày", "hôm"}
PARTICLES = {"nhé", "nha", "nhá", "nghe", "nghen", "nhen", "hen", "ạ", "à", "á", "ha", "hả", "hở", "nhỉ", "hỉ", "nè",
             "đi", "đó", "đấy", "ấy", "vậy", "thế", "nhứt", "hông", "không", "chứ", "luôn", "với"}
SAME_WORD = [{"tôi", "tui"}, {"giúp", "giùm", "dùm", "hộ"}, {"không", "hông", "hổng", "hong"}, {"vậy", "dậy", "zậy"},
             {"gì", "chi"}, {"này", "nầy"}, {"nghìn", "ngàn"}, {"ơi", "ới"}, {"mẹ", "mạ"}, {"nớ", "đó", "đấy"},
             {"ni", "này", "nay"}]
DROP_OK = PARTICLES | {"cho", "em", "anh", "chị", "mình", "ơi", "cái", "thì", "là", "mà", "rồi", "nữa", "này"}
_TONES = {"̀", "́", "̉", "̃", "̣"}


def tones(word: str) -> tuple:
    return tuple(c for c in unicodedata.normalize("NFD", word) if c in _TONES)


def trivial_pair(u: str, v: str) -> bool:
    """A confusion that changes nothing the model has to undo."""
    if not v:
        return u in DROP_OK
    if bare(u) == bare(v) and tones(u) == tones(v):
        return True  # huỷ / hủy: one syllable, tone mark placed elsewhere
    if u in PARTICLES and v in PARTICLES:
        return True
    return any(u in g and v in g for g in SAME_WORD)


def _confusions(ref: str, hyp: str):
    """(one-for-one sound-alike substitutions, single drops) of hyp against ref, as (ref word, hyp word or '')."""
    import difflib
    x = [w.lower() for w in _WORD.findall(unicodedata.normalize("NFC", ref))]
    y = [w.lower() for w in _WORD.findall(unicodedata.normalize("NFC", hyp))]
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=x, b=y, autojunk=False).get_opcodes():
        if op == "replace" and i2 - i1 == j2 - j1:
            out += [(u, v) for u, v in zip(x[i1:i2], y[j1:j2])
                    if difflib.SequenceMatcher(a=bare(u), b=bare(v)).ratio() >= 0.5]
        elif op == "delete" and i2 - i1 == 1:
            out.append((x[i1], ""))
    return x, out


class RealNoise:
    """Simulated readings: each word occurrence is misheard with its own measured rate (errors / times seen in the gold
    text of train clips, +1 smoothing) times a per-request difficulty drawn from Gamma(1.2) (some clips are hard, most
    are clean), as one of the confusions the averaged reading made of it. Fitted on the latest train requests, the
    number of meaningful errors per request comes out 60.5/23.5/9.3/4.4/1.4/1.0 % for 0/1/2/3/4/5+ (real
    60.0/23.6/10.4/3.7/1.5/0.8). Errors land on the words and requests that real readings get wrong; the first version
    (13 Nov) drew the count first and forced it onto any request, so easy requests had their few common words misheard
    (gọi, nhắc, tối, ví 2-8x real)."""
    SHAPE, SCALE, PRIOR = 1.2, 1.1, 1.0

    def __init__(self, table: dict, errs: dict, seen: dict, k_dist: list):
        from collections import Counter
        clean = {}
        for u, vs in table.items():
            pairs = Counter(vs)
            # 13 Nov Opus review: no number word made from a non-number unless the pair was seen twice (mưa → mười,
            # chị → chín), no particle turned into a content word (alignment slips such as ạ → hoà)
            keep = [v for v in vs if not (v in NUMBER_WORDS and u not in NUMBER_WORDS and pairs[v] < 2)
                    and not (u in PARTICLES and v and v not in PARTICLES)]
            if keep:
                clean[u] = keep
        self.table, self.k_dist = clean, k_dist
        self.rate = {u: errs[u] * len(clean[u]) / len(table[u]) / (seen.get(u, 0) + self.PRIOR) for u in clean}

    def apply(self, text: str, rng: random.Random, keep: set, qmark: float = None) -> str:
        """One simulated reading of ``text``; protected words and digits are never changed (often unchanged: the
        caller decides whether it needs an error)."""
        text = unicodedata.normalize("NFC", text)
        hard = rng.gammavariate(self.SHAPE, 1 / self.SHAPE) * self.SCALE
        out, last = [], 0
        for m in _WORD.finditer(text):
            w = m.group()
            lw = w.lower()
            if lw not in self.table or lw in keep or any(c.isdigit() for c in w):
                continue
            if rng.random() >= min(1.0, hard * self.rate[lw]):
                continue
            v = rng.choice(self.table[lw])
            v = v.capitalize() if v and w[0].isupper() else v
            out.append(text[last:m.start()] + v)
            last = m.end()
        out.append(text[last:])
        return asr_question(tidy_punct("".join(out)), rng, QUESTION_MARK if qmark is None else qmark)


def real_noise_from(rows: list, avg5: dict) -> RealNoise:
    """RealNoise learned on these rows: meaningful confusions of the averaged reading (all clips) and, for the report,
    the real number of such errors per latest request."""
    table, errs, seen, ks = {}, {}, {}, []
    for r in rows:
        for a, ref in prompts.gold_transcripts(r).items():
            if a not in avg5:
                continue
            x, conf = _confusions(ref, avg5[a])
            for w in x:
                seen[w] = seen.get(w, 0) + 1
            good = [(u, v) for u, v in conf if not trivial_pair(u, v)]
            for u, v in good:
                table.setdefault(u, []).append(v)
                errs[u] = errs.get(u, 0) + 1
            if a == r["audio"]:
                ks.append(len(good))
    hist = [0] * (min(max(ks), 8) + 1)
    for k in ks:
        hist[min(k, 8)] += 1
    return RealNoise(table, errs, seen, [h / len(ks) for h in hist])


def match_case(variant: str, sources: list, calls: list) -> str:
    """A string argument value copied into ``variant`` with the label's capital first letter ("nhắc anh Đem máy sấy")
    goes back to how the request said it ("đem máy sấy"): the capital is no ASR output and would point at the value."""
    for c in calls:
        for v in (c.get("arguments") or {}).values():
            if isinstance(v, str) and len(v) >= 2 and v[0].isupper() and v in variant:
                low = v[0].lower() + v[1:]
                if any(low in s and v not in s for s in sources):
                    variant = re.sub(rf"(?<=[^\s.!?…])(\s+){re.escape(v)}", lambda m: m.group(1) + low, variant)
    return variant


def _spelling_key(w: str) -> tuple:
    n = unicodedata.normalize("NFD", w.lower())
    return "".join(c for c in n if c not in _TONES), tuple(c for c in n if c in _TONES)


def spelling_forms(rows: list, avg5: dict) -> dict:
    """{word: the data's spelling} for words Gemini writes with the tone mark on another vowel (hòa / hoà): where the
    gold requests and the averaged readings use one placement >= 90 % of the time and the other <= 10 %."""
    from collections import Counter, defaultdict
    f = defaultdict(Counter)
    for t in [r["audio_text"] for r in rows] + list(avg5.values()):
        for w in _WORD.findall(unicodedata.normalize("NFC", t).lower()):
            f[_spelling_key(w)][w] += 1
    out = {}
    for c in f.values():
        if len(c) < 2:
            continue
        dom, n = c.most_common(1)[0]
        tot = sum(c.values())
        if n >= 0.9 * tot:
            out |= {w: dom for w in c if w != dom}
    return out


def match_spelling(variant: str, sources: list, calls: list, forms: dict) -> str:
    """Gemini's tone-mark placement (điều hòa, hóa đơn, xóa) back to the data's (hoà, hoá, xoá), the form the speech
    model writes; a spelling the request itself or a call argument uses is kept."""
    keep = " ".join(sources + [json.dumps(calls, ensure_ascii=False)]).lower()

    def fix(m):
        w = m.group()
        dom = forms.get(w.lower())
        if not dom or re.search(rf"(?<![\wÀ-ỹ]){re.escape(w.lower())}(?![\wÀ-ỹ])", keep):
            return w
        return dom.capitalize() if w[0].isupper() else dom
    return _WORD.sub(fix, unicodedata.normalize("NFC", variant))


# ------------------------------------------------------------------------- grounding checks (13 Nov T8/T9 review)
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_NUMW = r"(?:không|một|mốt|hai|ba|bốn|tư|năm|lăm|nhăm|sáu|bảy|tám|chín|mười|mươi|trăm|linh|lẻ|nghìn|ngàn|triệu|rưỡi)"
TIME_KEYS = {"time", "start", "end", "date", "date_from", "date_to", "schedule_time", "delivery_time", "since"}
PLACE_KEYS = {"destination", "origin", "pickup", "dropoff", "address", "location"}


def _nums(s: str) -> set:
    out = set()
    for m in _NUM.findall(s):
        out.add(m)
        if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", m):
            out.add(re.sub(r"[.,]", "", m))
        elif re.fullmatch(r"\d+[.,]\d+", m):
            out |= {m.replace(",", "."), m.replace(".", ",")}
    return out


def _said(text: str) -> set:
    from lifemate.drills import parse_number
    out = _nums(text)
    for m in re.finditer(rf"{_NUMW}(?:\s+{_NUMW})*", text.lower()):
        try:
            n = parse_number(m.group())
        except Exception:  # noqa: BLE001
            n = None
        if n is not None:
            out.add(str(n))
    return out


def ungrounded(item: dict, texts: dict) -> list:
    """Numbers (incl. the digits of ids such as c_8081) in string arguments that appear nowhere before the call: not
    in the shown texts, the request, context, history or the results of earlier calls (a call's own result echoes its
    arguments, so it does not count). [(tool, key, number)]"""
    calls, res = item["label"]["tool_calls"], item["label"]["tool_results"]
    base = _said(" ".join(texts.values()) + " " + item["audio_text"] + " "
                 + json.dumps(item["context"], ensure_ascii=False) + " " + json.dumps(item["history"], ensure_ascii=False))
    out = []
    for i, c in enumerate(calls):
        pool = base | _nums(json.dumps(res[:i], ensure_ascii=False))
        for k, v in (c.get("arguments") or {}).items():
            if k in TIME_KEYS or not isinstance(v, str) or re.fullmatch(r"[\d\-T:+.]+", v):
                continue
            out += [(c["name"], k, n) for n in _nums(v)
                    if n not in pool and len(re.sub(r"\D", "", n)) >= 2 and not re.fullmatch(r"20\d\d", n)]
    return out


def unsourced_places(item: dict, texts: dict) -> list:
    """Place arguments (destination, pickup...) whose text appears nowhere before the call."""
    src = (" ".join(texts.values()) + " " + item["audio_text"] + json.dumps(item["context"], ensure_ascii=False)
           + json.dumps(item["history"], ensure_ascii=False)).lower()
    out = []
    for i, c in enumerate(item["label"]["tool_calls"]):
        pool = src + json.dumps(item["label"]["tool_results"][:i], ensure_ascii=False).lower()
        out += [(c["name"], k, v) for k, v in (c.get("arguments") or {}).items()
                if k in PLACE_KEYS and isinstance(v, str) and v.lower() not in pool]
    return out
