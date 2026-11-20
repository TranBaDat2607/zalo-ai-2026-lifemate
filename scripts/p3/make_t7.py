"""T7 of improvment_plan3.md: each train request reworded by Gemini (Bắc/Trung/Nam) becomes its own conversation.

    python scripts/p3/make_t7.py            # outputs/p3/t7_full.jsonl, t7_fold0.jsonl (needs t7/paraphrases.jsonl)

scripts/t7/agy_paraphrase.py has Gemini rewrite each request in up to three regional styles and keeps only the
variants that pass its checks (anchors kept, no new or lost numbers, no app/tool names). Here each kept variant is one
training conversation of the same sample: the latest request is the variant, heard through T2b's hearing noise (its
numbers first written as the speech model writes them, make_t6.asr_numbers), and restated back to the clean variant in
the think block (train.py sets item["audio_text"] as the restatement target). The gold calls, results and the history
(its real averaged readings) are unchanged. So the model meets the same intent said many ways, which the real train
set never shows. Variants whose exact string arguments the noise would lose are dropped by gate G2.
The fold-0 file leaves out the 386 validation samples and its noise table is learned without them.
"""
import copy
import difflib
import re
import json
import random
from collections import defaultdict

import common
import review_rules
from collections import Counter
from make_t6 import asr_numbers

PARA = common.OUT / "t7" / "paraphrases.jsonl"


def paraphrases() -> dict:
    """{id: [(style, text)]}: after the review loop (agy_review.py), the variants Gemini judged to mean the same as
    the request (a flagged one replaced by its reviewed rewrite); before it, every variant that passed collect."""
    import agy_review
    out = defaultdict(list)
    if True:  # agy_review.passed(): lexically checked, not judged different
        for (rid, style), text in sorted(agy_review.passed().items()):
            if not rid.startswith("train_") or "_t8_" in rid:
                continue
            out[rid].append((style, text))
        return out
    for r in map(json.loads, PARA.read_text(encoding="utf-8").splitlines()):
        if r.get("ok"):
            out[r["id"]].append((r["style"], r["text"]))
    return out


# Particles, fillers and forms of address: a variant that changes only these words teaches nothing a T2b hearing does
# not, and repeats the sample's context and answer once more (13 Nov review: 400 of 2,948 kept rows, dropped).
FILLERS = {"ơi", "à", "ạ", "nhé", "nha", "nhá", "nghe", "hen", "hỉ", "giúp", "giùm", "dùm", "với", "đi", "luôn", "cho",
           "em", "anh", "chị", "mình", "tôi", "tui", "tao", "con", "cô", "chú", "bác", "bạn", "ấy", "đấy", "nhỉ", "vậy",
           "thế", "đó", "này", "coi", "xem", "hộ", "nào", "rồi", "thì", "mà", "cái", "á", "ha", "hở", "hông", "không",
           "nhen"}
_W = re.compile(r"[\wÀ-ỹ]+")


def fillers_only(source: str, variant: str) -> bool:
    """True when every word the variant adds, drops or replaces is a particle, filler or form of address."""
    a, b = _W.findall(source.lower()), _W.findall(variant.lower())
    ops = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    return all(w in FILLERS for op, i1, i2, j1, j2 in ops if op != "equal" for w in a[i1:i2] + b[j1:j2])


# Regional markers: real train requests carry 0/1/2/3/4 of them in 72/17/8.5/2.1/0.4 % of cases; Gemini's Trung and
# Nam rewordings piled them up (4+ in 6.9 % of variants). A variant with 4+ markers, or 3+ more than its source, says a
# request in a way no speaker in the data does (13 Nov review, item 3) and is dropped; the text itself is never edited.
MARKERS = {"nớ", "mô", "rứa", "tê", "răng", "hỉ", "mần", "bả", "lận", "ni", "nghen", "chừ", "ri", "nì", "hè", "nhứt",
           "hổng", "hông", "dzậy", "zậy", "nhen", "bển", "trển", "dìa", "dô", "hen", "tui", "nè", "ổng", "chớ", "hử",
           "đặng", "giùm", "dùm", "coi", "nỏ"}


def markers(text: str) -> int:
    return sum(w in MARKERS for w in _W.findall(text.lower()))


def overdone(source: str, variant: str) -> bool:
    return markers(variant) >= 4 or markers(variant) - markers(source) >= 3


# second Opus review (13 Nov): rewordings that lost the action the label needs and slipped past the verb judge
LOST_ACTION = {"train_1679": "ghi chú"}  # "ghi vô ghi chú workshop" -> "nhớ chuyển ghi chú ... vô trong nhà"


def lost_action(r: dict, text: str) -> bool:
    w = LOST_ACTION.get(r["id"])
    return bool(w) and not re.search(rf"(?<![\wÀ-ỹ])ghi\s+(?:vô|vào)\s+{w}", text.lower())


def usable(r: dict, para: dict, drop: set) -> list:
    """The kept variants of row r: not filler-only, not overdone, not judged a different action (verb_review)."""
    import agy_review
    return [(st, t) for st, t in para.get(r["id"]) or [] if not fillers_only(r["audio_text"], t)
            and not overdone(r["audio_text"], t) and agy_review.key(r["id"], st, t) not in drop
            and not lost_action(r, t)]


def build(rows: list, noise, avg5: dict, para: dict, seed: int, drop: set = frozenset()) -> list:
    out, forms = [], common.spelling_forms(rows, avg5)
    dropped = Counter()
    for r in rows:
        variants = usable(r, para, drop)
        if not variants:
            continue
        keep = common.protected(r)
        for style, text in variants:
            item = copy.deepcopy(r)
            # a label value copied with its capital ("nhắc Đem máy sấy") goes back to the request's own casing
            text = common.match_case(text, [r["audio_text"]], r["label"]["tool_calls"])
            text = common.match_spelling(text, [r["audio_text"]], r["label"]["tool_calls"], forms)  # hòa -> hoà
            text = review_rules.fix_format(review_rules.keep_ending(r["audio_text"], review_rules.keep_helper(
                r["audio_text"], text, (r["context"].get("user_profile") or {}).get("city", ""),
                random.Random(f"helper:{r['id']}:{text}"))))
            hist = [avg5.get(h["audio"], h.get("audio_text", "")) for h in r["history"] if h["role"] == "user"]
            why = review_rules.style_problem(r["audio_text"], text, r, hist)  # 13 Nov Opus review
            if why:
                dropped[why] += 1
                continue
            item["audio_text"] = text  # the restatement target
            rng = random.Random(f"{seed}:{r['id']}:{style}")
            texts = {a: avg5[a] for a in common.prompts.gold_transcripts(r) if a != item["audio"]}
            texts[item["audio"]] = noise.apply(asr_numbers(text, rng), rng, keep)
            out.append(common.conv("T7", r, texts, item=item, style=style))
    print(json.dumps({"t7_style_dropped": dropped}, ensure_ascii=False))
    return out


def main() -> None:
    d = common.load()
    para = paraphrases()
    print(json.dumps({"samples_with_paraphrases": len(para), "variants": sum(len(v) for v in para.values())}))
    import verb_review
    drop = verb_review.different()
    print(json.dumps({"judged_different_action": len(drop)}))
    review_rules.set_marker_rates(d["rows"])
    for name, rows in common.pools(d):
        noise = common.real_noise_from(rows, d["avg5"])  # simulated readings, unchanged ones included
        out = build(rows, noise, d["avg5"], para, 71, drop)
        common.write(f"t7_{name}", out)


if __name__ == "__main__":
    main()
