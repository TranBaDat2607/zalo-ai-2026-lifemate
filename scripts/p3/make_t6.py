"""T6 of improvment_plan3.md: more synthetic conversations for the public-only apps, from scripts/make_synth.py.

    python scripts/p3/make_t6.py --per-scenario 45 --seed 7     # outputs/p3/t6_full.jsonl, t6_fold0.jsonl

make_synth.py writes requests and app data from the public tool catalog only (no public sample, no *_environment/)
and runs the calls through the organisers' simulator, so every result is real (gate G1). A new seed gives new entities,
numbers, days and phrasings of the same 47 scenarios. Conversations whose request text is already in the bundled
synth_samples.jsonl (trained by the recipe) or earlier in this file are dropped. Every clip is heard through RealNoise
(simulated readings, as T2b), after its numbers are written as the speech model writes them (asr_numbers: digits before
"triệu", "phút", "độ" about as often as avg5 writes them, counts such as "hai vé" in words). None of these apps is held out, so the fold-0 file differs only in its noise table (learned without
the validation clips).
"""
import argparse
import collections
import json
import random
import re
import subprocess
import sys

import common
from lifemate.drills import DIGIT_SHARE, parse_number, say_number

NUMW = r"(?:không|một|mốt|hai|ba|bốn|tư|năm|lăm|nhăm|sáu|bảy|tám|chín|mười|mươi|trăm|linh|lẻ|chục)"
UNIT = (r"(?:nghìn|ngàn|triệu|giờ|phút|độ|phần trăm|tiếng|chỗ|vé|cái|chiếc|gói|hộp|khay|ký|kg|chai|lít|lon|lốc|bó|"
        r"túi|vỉ|quả|trái|hũ|bịch|thùng|cổ phiếu|cuốn|người|đêm|ngày|lần|phần|suất|bộ|đôi|con|xuất)")
SPAN = re.compile(rf"(?<![\wÀ-ỹ.,])(\d+|{NUMW}(?:\s+{NUMW})*)\s+({UNIT})(?![\wÀ-ỹ])", re.I)

RAW = common.OUT / "t6_raw.jsonl"


def asr_numbers(text: str, rng: random.Random) -> str:
    """Numbers before a unit as the averaged speech model writes them: digits with DIGIT_SHARE[unit] (measured on
    avg5_train), else words; units outside the table are counts ("hai vé", "ba khay"), which it writes in words."""
    def one(m):
        num, unit = m.group(1), m.group(2)
        try:
            n = parse_number(num.lower())
        except Exception:  # noqa: BLE001
            return m.group(0)
        if n is None or n >= 1000 or (num.isdigit() and str(n) != num):
            return m.group(0)
        share = DIGIT_SHARE.get(unit.lower(), 0.0)
        said = str(n) if rng.random() < share else (num if not num.isdigit() else say_number(n, rng, unit_follows=True))
        return f"{said} {unit}"
    return SPAN.sub(one, text)


KIN = r"(?:em|anh|chị|cô|chú|bác|con|cháu)"


def address_clash(text: str) -> bool:
    """The template's speaker calls the assistant by the word it uses for itself ("tìm hộ em ... nhé em", "Em ơi, tìm
    hộ em ..."): no real request does that (13 Nov T6 review: ~30 rows)."""
    t = text.lower()
    selves = set(re.findall(rf"(?<![\wÀ-ỹ])(?:giùm|giúp|hộ|dùm)\s+({KIN})(?![\wÀ-ỹ])", t))
    called = set(re.findall(rf"^(?:\W*(?:ờ|à|này|ơi)\W+)?({KIN})\s+ơi", t))
    called |= set(re.findall(rf"(?:nhé|nha|hen|nghe|nhen|đi|với)\s+({KIN})\W*$", t))
    return bool(selves & called)


OPENER = re.compile(r"^(?:(?:em|bạn|trợ lý)\s+ơi|ờ\s+à|này|ơi|à|ờ)\s*,\s*", re.I)
STACK = re.compile(r"(?<![\wÀ-ỹ])(nhé|ơi)(?:\s*,)?\s+(nhé|nha|nghe|hen|nhen|với|ừ)(?![\wÀ-ỹ])", re.I)
TAIL = re.compile(r"( nhé em| nha em| nhé| nha| nghe| với| đi| hen)?([.?!]?)$")
TRAIN_TAILS = [("", 0.26), (" nhé", 0.245), (" nha", 0.172), (" nghe", 0.065), (" đi", 0.026), (" với", 0.013),
               (" hen", 0.007), (" nhé em", 0.001), (" nha em", 0.001)]  # train closings; "" lowered from 0.412: templates lack the
# mid-sentence particles real speech has, so the overall particle share (gate G4) needs more closing ones
NO_RAIL = {"Cần Thơ", "Đà Lạt"}
OPENER_SHARE = 0.15  # real train requests open with a filler/call 14.4 % of the time, make_synth's 67 %
TEMPLATE_CAP = 25    # rows per call sequence (13 sequences sat at 45, the recipe's synth already holds that many)


def tidy(r: dict, rng: random.Random) -> dict:
    """13 Nov Opus review of T6: strip most filler openers (keep share ~ train's), unstack particles ("nhé nha",
    "ơi, ừ"), write prices in assistant replies as real replies do (200.000 đồng, not 200000 đồng). Text edits only;
    calls and results stay as the simulator made them."""
    def fix(t):
        # train speakers say "Sài Gòn" (33 readings, never "TP.HCM"); the argument stays "TP.HCM", as in train's
        # lottery.get_results(province="TP.HCM") for "đài Sài Gòn"
        t = re.sub(r"TP\.\s?HCM|TP HCM", "Sài Gòn", t)
        # make_synth draws the closing particle uniformly from 9 (hen 11 %, train 0.7 %; "nha em" 11 %, train 0.1 %):
        # redraw it from train's own closing distribution (second Opus review)
        m = TAIL.search(t)
        if m and not re.search(r"[?]$", t):
            t = t[:m.start()] + rng.choices([x for x, _ in TRAIN_TAILS], [w for _, w in TRAIN_TAILS])[0] + m.group(2)
        if OPENER.match(t) and rng.random() > OPENER_SHARE / 0.67:
            t = OPENER.sub("", t, count=1)
            t = t[:1].upper() + t[1:]
        return STACK.sub(lambda m: m.group(1), t)
    r["audio_text"] = fix(r["audio_text"])
    for h in r["history"]:
        if h["role"] == "user":
            h["audio_text"] = fix(h["audio_text"])
        elif isinstance(h.get("text"), str):
            h["text"] = re.sub(r"(?<![\d.])(\d{4,})(?= đồng)", lambda m: f"{int(m.group(1)):,}".replace(",", "."), h["text"])
    return r


def label_problem(r: dict) -> str:
    """Second Opus review: "rẻ nhất" with a tied lowest price (the booked id is a guess); a train to or from a city
    with no railway."""
    calls, res = r["label"]["tool_calls"], r["label"]["tool_results"]
    if "rẻ nhất" in r["audio_text"].lower():
        for x in res:
            prices = [t.get("price") for k in ("trips", "products", "items", "results", "events", "flights") for t in
                      (x.get(k) or []) if isinstance(x, dict) and isinstance(t, dict) and isinstance(t.get("price"), (int, float))]
            if len(prices) >= 2 and sorted(prices)[0] == sorted(prices)[1]:
                return "cheapest tied"
    for c in calls:
        a = c.get("arguments") or {}
        if c["name"].startswith("train_bus_tickets.") and a.get("type") == "train" and {a.get("origin"), a.get("destination")} & NO_RAIL:
            return "train to a city with no railway"
    return ""


def ask_problem(r: dict) -> str:
    """Asks train never makes: two or more parameters at once, or a quantity (train defaults it to 1). The bus-ticket
    ask names the origin, which the user's city already gives: it asks the date only (an ask's result does not depend
    on its params, so the simulator's answer stays valid)."""
    for c in r["label"]["tool_calls"]:
        if c["name"] != "sys.ask_missing_info":
            continue
        ps = c["arguments"].get("params") or []
        if c["arguments"].get("tool_name") == "train_bus_tickets.search" and ps == ["date", "origin"]:
            c["arguments"]["params"] = ["date"]
            ps = ["date"]
        if "quantity" in ps:
            return "quantity ask"
        if len(ps) > 1:
            return "multi-param ask"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scenario", type=int, default=45)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    common.OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(common.REPO / "scripts" / "make_synth.py"), "--per-scenario", str(args.per_scenario),
                    "--seed", str(args.seed), "--out", str(RAW)], check=True)
    bundled = {json.loads(line)["audio_text"] for line in (common.REPO / "src" / "lifemate" / "synth_samples.jsonl")
               .read_text(encoding="utf-8").splitlines() if line.strip()}
    raw = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen, kept = set(bundled), []
    for k, r in enumerate(raw):
        if r["audio_text"] in seen:
            continue
        seen.add(r["audio_text"])
        sid = f"t6_{k:05d}"
        for t in r["history"]:
            if t["role"] == "user":
                t["audio"] = t["audio"].replace(r["id"], sid)
        r["audio"], r["id"] = r["audio"].replace(r["id"], sid), sid
        kept.append(r)
    print(json.dumps({"made": len(raw), "kept": len(kept), "dropped_repeats": len(raw) - len(kept)}))
    d = common.load()
    clash = [r for r in kept if any(address_clash(t) for t in common.prompts.gold_transcripts(r).values())]
    kept = [r for r in kept if r not in clash]
    print(json.dumps({"address_clash_dropped": len(clash), "kept": len(kept)}))
    asks = collections.Counter(ask_problem(r) for r in kept)
    kept = [r for r in kept if not ask_problem(r)]
    labels = collections.Counter(label_problem(r) for r in kept)
    kept = [r for r in kept if not label_problem(r)]
    kept = [tidy(r, random.Random(f"tidy:{args.seed}:{r['id']}")) for r in kept]
    # after tidying: no repeated (text, calls); a no-call sentence ("app not installed") at most twice
    seen_tc, nocall, dedup = set(), collections.Counter(), []
    for r in kept:
        key = (r["audio_text"], json.dumps(r["label"]["tool_calls"], ensure_ascii=False))
        if key in seen_tc or (not r["label"]["tool_calls"] and nocall[r["audio_text"]] >= 2):
            continue
        seen_tc.add(key)
        nocall[r["audio_text"]] += not r["label"]["tool_calls"]
        dedup.append(r)
    print(json.dumps({"label_drops": {k: v for k, v in labels.items() if k}, "repeat_drops": len(kept) - len(dedup)}))
    kept = dedup
    by_seq = collections.defaultdict(list)
    for r in kept:
        by_seq[tuple(c["name"] for c in r["label"]["tool_calls"])].append(r)
    capped = []
    for seq, rs in by_seq.items():
        random.Random(f"cap:{args.seed}:{seq}").shuffle(rs)
        capped += rs[:TEMPLATE_CAP]
    kept = sorted(capped, key=lambda r: r["id"])
    opener = sum(bool(OPENER.match(r["audio_text"])) for r in kept) / max(len(kept), 1)
    print(json.dumps({"ask_drops": {k: v for k, v in asks.items() if k}, "after_cap": len(kept),
                      "opener_share": round(opener, 3)}))
    for name, rows in common.pools(d):
        noise = common.real_noise_from(rows, d["avg5"])  # simulated readings, as T2b/T7/T9
        forms = common.spelling_forms(rows, d["avg5"])
        out = []
        for r in kept:
            keep = common.protected(r)
            rng = random.Random(f"{args.seed}:{r['id']}")
            out.append(common.conv("T6", r, {a: noise.apply(asr_numbers(common.match_spelling(t, [], r["label"]["tool_calls"], forms), rng), rng, keep,
                                                            common.QUESTION_MARK_SYNTH if a == r["audio"] else common.QUESTION_MARK_HISTORY)
                                             for a, t in common.prompts.gold_transcripts(r).items()}))
        common.write(f"t6_{name}", out)


if __name__ == "__main__":
    main()
