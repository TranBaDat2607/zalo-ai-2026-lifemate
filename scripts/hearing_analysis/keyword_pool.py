"""Where hearing costs points on 0012 (outputs/notes-aux/aux_eval, 71.57), latest request only.

For each validation sample: gold text G, the text the agent heard H (avg5), the agent's restated request R.
Word spans are aligned (difflib on normalised tokens, fillers dropped). A wrong span is a KEY error when its gold words
include a word of a gold argument value, a number word or a time word. Each key error is typed (number / time / name or
foreign / other argument word) and marked common (the gold word occurs >= 10 times in train gold texts outside
validation) or rare (< 3). We also check whether the right word is visible to the agent (history texts, earlier tool
results, profile) and whether any of the 5 out-of-fold readings got it right.
"""
import collections as C
import difflib
import json
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "scripts" / "notes"))
from build_pairs import FILLERS  # noqa: E402
from compare_notes import SPLITS, load  # noqa: E402
from first_error import restated  # noqa: E402
from lifemate import data  # noqa: E402
from lifemate.asr_eval import normal_tokens  # noqa: E402

TIME = set("sáng trưa chiều tối đêm khuya mai nay mốt qua hôm giờ phút rưỡi kém thứ tuần tháng năm ngày sau trước "
           "tới hai ba tư bốn năm sáu bảy tám chín mười mươi chủ nhật".split())
NUMW = set("không một mốt hai ba bốn tư năm lăm sáu bảy bẩy tám chín mười mươi trăm nghìn ngàn triệu tỷ linh lẻ rưỡi".split())


def toks(text):
    return [w for w in normal_tokens(text) if w not in FILLERS]


def value_tokens(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return set(normal_tokens(str(int(v)) if float(v).is_integer() else str(v)))
    if isinstance(v, str):
        return set(normal_tokens(v))
    if isinstance(v, list):
        return set().union(*[value_tokens(x) for x in v]) if v else set()
    if isinstance(v, dict):
        return set().union(*[value_tokens(x) for x in v.values()]) if v else set()
    return set()


def spans(gold, heard):
    sm = difflib.SequenceMatcher(a=gold, b=heard, autojunk=False)
    return [(gold[i1:i2], heard[j1:j2]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]


def is_foreign(w):
    base = unicodedata.normalize("NFD", w)
    plain = "".join(c for c in base if unicodedata.category(c) != "Mn")
    return plain == w and any(c in w for c in "fjwz") or w.endswith(("ing", "er", "ck", "ll", "st"))


root = data.find_root("train")
items = {s["id"]: s for s in data.load_split("train", root)}
val = set(SPLITS["random"]) | set(SPLITS["apps"])
run = load(str(REPO / "outputs/notes-aux/aux_eval"))
N = len(run)
readings = {k: json.load(open(REPO / p)) for k, p in {
    "exp7": "outputs/exp7/asr_medium/medium_train.json", "m10": "outputs/w1-asr/asr_m10/medium10_train.json",
    "B": "outputs/b-synth/b/b_train.json", "s2027": "outputs/asr-s2027/s2027_train.json",
    "s2028": "outputs/asr-s2028/s2028_train.json", "avg5": "outputs/asr-avg5/avg5_train.json"}.items()}

# word frequencies in train gold texts outside validation (every turn)
freq = C.Counter()
for sid, it in items.items():
    if sid in val:
        continue
    freq.update(toks(it["audio_text"]))
    for h in it.get("history") or []:
        if h.get("role") == "user" and h.get("audio_text"):
            freq.update(toks(h["audio_text"]))


def visible_words(it):
    """Words the agent can read without hearing: earlier turns (gold texts are what history shows? use assistant and
    tool content), profile, earlier tool results."""
    ctx = it.get("context") or {}
    blob = json.dumps(ctx.get("user_profile") or {}, ensure_ascii=False)
    for h in it.get("history") or []:
        if h.get("role") != "user":
            blob += " " + json.dumps(h, ensure_ascii=False)
    return set(normal_tokens(blob))


rows = []
for sid, r in run.items():
    it = items[sid]
    G, H, R = toks(it["audio_text"]), toks(r["request"]), toks(restated(r))
    argw = set()
    for c in r["gold"]:
        argw |= value_tokens(c.get("arguments") or {})
    vis = visible_words(it)
    others = {k: toks(v[it["audio"]]) for k, v in readings.items() if it["audio"] in v}
    errs = []
    for g_words, h_words in spans(G, R):  # what the agent works from after its own restatement
        key = [w for w in g_words if w in argw or w in NUMW or w in TIME]
        if not key:
            continue
        kinds = set()
        for w in key:
            if w in NUMW:
                kinds.add("number")
            elif w in TIME:
                kinds.add("time")
            elif w in argw and (is_foreign(w) or any(x[:1].isupper() for x in it["audio_text"].split()
                                                     if normal_tokens(x)[:1] == [w])):
                kinds.add("name/foreign")
            else:
                kinds.add("argument word")
        kind = sorted(kinds, key=["name/foreign", "number", "time", "argument word"].index)[0]
        common = all(freq[w] >= 10 for w in key)
        rare = any(freq[w] < 3 for w in key)
        fixed_by = [k for k, o in others.items() if all(w in o for w in key)]
        errs.append(dict(gold=" ".join(g_words), said=" ".join(h_words), key=key, kind=kind,
                         common=common, rare=rare, visible=all(w in vis for w in key), in_any_reading=bool(fixed_by),
                         in_heard=all(w in H for w in key)))
    other_errs = [s for s in spans(G, R) if not any(w in argw or w in NUMW or w in TIME for w in s[0])]
    rows.append(dict(id=sid, score=r["score"], lost=(1 - r["score"]) * 100 / N, half="random" if sid in SPLITS["random"] else "apps",
                     key_errs=errs, n_other=len(other_errs), heard_exact=not spans(G, H), restated_exact=not spans(G, R),
                     H_wrong=len(spans(G, H)), R_wrong=len(spans(G, R))))

json.dump(rows, open(Path(__file__).with_name("keyword_rows.json"), "w"), ensure_ascii=False, indent=0)


def group(rs, name):
    lost = sum(x["lost"] for x in rs)
    mean = 100 * sum(x["score"] for x in rs) / max(1, len(rs))
    print(f"  {name:<44} n={len(rs):>3}  mean {mean:5.1f}  points lost {lost:5.2f}")


print(f"0012 mean {100 * sum(x['score'] for x in rows) / N:.2f}, lost {sum(x['lost'] for x in rows):.2f}")
print("by what the agent works from (restated request vs gold):")
group([x for x in rows if x["restated_exact"]], "restated exact")
group([x for x in rows if not x["restated_exact"] and not x["key_errs"]], "only non-key words wrong")
group([x for x in rows if x["key_errs"]], "a key word wrong")
print("LLM correction: heard exact %d -> restated exact %d; spans wrong heard %d -> restated %d" % (
    sum(x["heard_exact"] for x in rows), sum(x["restated_exact"] for x in rows),
    sum(x["H_wrong"] for x in rows), sum(x["R_wrong"] for x in rows)))
ke = [e | {"lost": x["lost"], "score": x["score"], "id": x["id"]} for x in rows for e in x["key_errs"]]
print(f"\nkey errors after restating: {len(ke)} in {sum(bool(x['key_errs']) for x in rows)} samples")
for f in ["kind", "common", "rare", "visible", "in_any_reading", "in_heard"]:
    c = C.Counter(e[f] for e in ke)
    print(f"  {f}: {dict(c)}")
print("\nsamples with a key error, points lost, by the first error's type (failed samples only):")
fail = [x for x in rows if x["key_errs"] and x["score"] < 0.999]
for k in ["name/foreign", "number", "time", "argument word"]:
    s = [x for x in fail if x["key_errs"][0]["kind"] == k]
    print(f"  {k:<14} n={len(s):>3} lost {sum(x['lost'] for x in s):5.2f}  common-word {sum(x['key_errs'][0]['common'] for x in s)}"
          f"  rare {sum(x['key_errs'][0]['rare'] for x in s)}  right in another reading {sum(x['key_errs'][0]['in_any_reading'] for x in s)}"
          f"  visible in context {sum(x['key_errs'][0]['visible'] for x in s)}")
print("\nexamples (gold -> restated), failed samples:")
for x in fail[:60]:
    e = x["key_errs"][0]
    print(f"  {x['id']} {x['score']:.2f} [{e['kind']}{', common' if e['common'] else ''}{', rare' if e['rare'] else ''}"
          f"{', other reading ok' if e['in_any_reading'] else ''}{', visible' if e['visible'] else ''}] '{e['gold']}' -> '{e['said']}'")
