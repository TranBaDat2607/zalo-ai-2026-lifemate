"""Review loop of plan 3: Gemini checks that every reworded request (T7, T8) means exactly what its source says, and
rewrites the ones it flags; the loop repeats until no rewrite is left to check (or --rounds).

    python scripts/p3/agy_review.py loop [--rounds 3] [--workers 3]
    python scripts/p3/agy_review.py report

One round: (1) every variant not reviewed yet goes to Gemini in batches of 15 pairs (A = source request, B = variant):
"same request?" with a short reason when not; (2) every flagged variant gets one new rewording from Gemini, told what
the flagged one changed; the new text must pass T7's checks (agy_paraphrase.check_variant: anchors, numbers, no
app/tool names) and is reviewed in the next round. Verdicts are kept in outputs/p3/review/verdicts.jsonl (key = id,
style, text), so a rerun only reviews what is new. make_t7.py / make_t8.py drop every variant whose latest verdict is
"different" and use a passed rewrite instead.
Claude writes no request text: rewrites come from Gemini, the code only routes and checks them.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import common
import make_t9

sys.path.insert(0, str(common.REPO / "scripts" / "t7"))
import agy_paraphrase as agy  # noqa: E402

OUT = common.OUT / "review"
VERDICTS = OUT / "verdicts.jsonl"
PAIRS = 40  # pairs per call: agy adds ~9k tokens to every call, so fewer, larger calls stretch an account's quota
REWRITES = OUT / "rewrites.jsonl"
SOURCES = {"t7": common.OUT / "t7" / "paraphrases.jsonl", "t8": common.OUT / "t8" / "paraphrases.jsonl"}

CHECK = """Bạn kiểm tra dữ liệu cho trợ lý ảo trên điện thoại. Mỗi mục có câu A (yêu cầu gốc của người dùng) và câu B (cách
nói khác). B ĐÚNG khi người nghe B sẽ làm đúng y những việc như A: cùng các việc, cùng thứ tự, cùng người/vật/địa điểm,
cùng số tiền/số lượng/giờ/ngày, cùng điều kiện "nếu... thì...", cùng cách làm (ví hay ngân hàng, SMS hay Zalo, gọi hay
nhắn...), không thêm, không bớt. Khác giọng địa phương, từ đệm, trật tự câu là bình thường, vẫn ĐÚNG.

Trả về JSON: {{"items": [{{"id": ..., "same": true/false, "why": "nếu sai: B đổi gì, ngắn gọn"}}, ...]}} đúng thứ tự.

{pairs}"""

REWRITE = """Bạn viết lại câu ra lệnh bằng giọng nói gửi trợ lý ảo trên điện thoại (tiếng Việt). Với mỗi mục: viết đúng 1
cách nói khác của câu gốc theo kiểu nói được ghi, GIỮ NGUYÊN ý định và mọi chi tiết (ai, việc gì, bao nhiêu, khi nào,
điều kiện, thứ tự, cách làm). Lần trước đã viết sai: xem "lỗi lần trước" và tránh lỗi đó. Mỗi cụm trong "giữ nguyên"
phải xuất hiện y hệt. Không thêm số, tên, việc mới; không nhắc tên ứng dụng. Một câu, không giải thích, không ngoặc kép.

Trả về JSON: {{"items": [{{"id": ..., "variants": [câu]}}, ...]}} đúng thứ tự.

{items}"""

SCHEMA_CHECK = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "required": ["id", "same"],
    "properties": {"id": {"type": "string"}, "same": {"type": "boolean"}, "why": {"type": "string"}}}}}}
SCHEMA_REWRITE = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "required": ["id", "variants"],
    "properties": {"id": {"type": "string"}, "variants": {"type": "array", "items": {"type": "string"},
                                                          "minItems": 1, "maxItems": 1}}}}}}


def key(rid: str, style: str, text: str) -> str:
    return hashlib.sha1(f"{rid}|{style}|{text}".encode()).hexdigest()[:16]


def load_jsonl(p: Path) -> list:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def sources() -> dict:
    """Source text of every variant id: the train request (T7) or the substituted request (T8)."""
    d = common.load()
    src = {r["id"]: r["audio_text"] for r in d["rows"]}
    for s in load_jsonl(common.OUT / "t8" / "swaps.jsonl"):
        src[s["id"]] = s["text"]
    return src


def variants() -> list:
    """Every variant that passed the lexical checks: originals and passed rewrites, [(kind, id, style, text)]."""
    out = []
    for kind, p in SOURCES.items():
        out += [(kind, v["id"], v["style"], v["text"]) for v in load_jsonl(p) if v.get("ok")]
    out += [(w["kind"], w["id"], w["style"], w["text"]) for w in load_jsonl(REWRITES) if w.get("ok")]
    return out


def latest_verdicts() -> dict:
    out = {}
    for v in load_jsonl(VERDICTS):
        out[v["key"]] = v
    return out


def review_round(args, src: dict, rnd: int) -> list:
    """Reviews every unreviewed variant; returns the flagged ones."""
    done = latest_verdicts()
    todo = [(k, i, s, t) for k, i, s, t in variants() if key(i, s, t) not in done]
    print(f"round {rnd}: {len(todo)} variants to review", flush=True)
    bdir, adir = OUT / f"check40_b{rnd}", OUT / f"check40_a{rnd}"
    bdir.mkdir(parents=True, exist_ok=True)
    pending = {x["key"] for f in bdir.glob("c*.json") for x in json.loads(f.read_text(encoding="utf-8"))}
    todo = [] if pending else todo  # batches keep their make-up across a quota stop (answers are matched by file)
    for b in range(0, len(todo), PAIRS):
        part = [{"key": key(i, s, t), "kind": k, "id": i, "style": s, "a": src[i], "b": t} for k, i, s, t in
                todo[b:b + PAIRS]]
        (bdir / f"c{b // PAIRS:04d}.json").write_text(json.dumps(part, ensure_ascii=False), encoding="utf-8")

    def prompt(batch):
        return CHECK.format(pairs="\n".join(f"- id: {x['key']}\n  A: {x['a']}\n  B: {x['b']}" for x in batch))
    files = [f for f in sorted(bdir.glob("c*.json")) if not (adir / f.name).exists()]
    stopped = None
    try:
        make_t9.run_batches(files, adir, prompt, SCHEMA_CHECK, args, "check")
    except SystemExit as e:  # quota: keep what was answered, then stop
        stopped = e
    flagged, n = [], Counter()
    done = latest_verdicts()
    with open(VERDICTS, "a", encoding="utf-8") as fout:
        for f in sorted(bdir.glob("c*.json")):
            a = adir / f.name
            if not a.exists():
                continue
            ans = {str(x.get("id")): x for x in json.loads(a.read_text(encoding="utf-8"))["reply"] if isinstance(x, dict)}
            for x in json.loads(f.read_text(encoding="utf-8")):
                v = ans.get(x["key"])
                if v is None or x["key"] in done:
                    continue
                same = bool(v.get("same"))
                n["same" if same else "different"] += 1
                rec = {**x, "same": same, "why": v.get("why", ""), "round": rnd}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if not same:
                    flagged.append(rec)
    print(f"round {rnd}: {dict(n)}", flush=True)
    if stopped:
        raise stopped
    if all((adir / f.name).exists() for f in bdir.glob("c*.json")):  # round complete: next round starts fresh
        for f in list(bdir.glob("c*.json")) + list(adir.glob("c*.json")):
            f.rename(f.with_name(f"done_r{rnd}_{f.name}"))
    return flagged


def rewrite_round(args, flagged: list, rnd: int) -> int:
    if not flagged:
        return 0
    rows = {r["id"]: r for r in common.load()["rows"]}
    swaps = {s["id"]: s for s in load_jsonl(common.OUT / "t8" / "swaps.jsonl")}
    bdir, adir = OUT / f"rew20_b{rnd}", OUT / f"rew20_a{rnd}"
    bdir.mkdir(parents=True, exist_ok=True)
    for f in bdir.glob("*.json"):
        f.unlink()
    items = []
    for x in flagged:
        base = rows.get(x["id"]) or rows[swaps[x["id"]]["source"]]
        if x["id"] in swaps:
            base = dict(base, audio_text=swaps[x["id"]]["text"],
                        label={"tool_calls": swaps[x["id"]]["calls"], "tool_results": swaps[x["id"]]["results"]})
        keep = agy.anchors(base)
        items.append({"rid": f"r{len(items)}", "kind": x["kind"], "id": x["id"], "style": x["style"], "text": x["a"],
                      "keep": keep, "why": x["why"], "bad": x["b"]})
    for b in range(0, len(items), 20):
        (bdir / f"r{b // 20:04d}.json").write_text(json.dumps(items[b:b + 20], ensure_ascii=False), encoding="utf-8")

    def prompt(batch):
        return REWRITE.format(items="\n".join(
            f"- id: {x['rid']}\n  câu gốc: {x['text']}\n  kiểu nói: {agy.STYLES.get(x['style'], x['style'])}\n"
            f"  giữ nguyên: {json.dumps(x['keep'], ensure_ascii=False)}\n  lỗi lần trước: {x['why']}" for x in batch))
    files = [f for f in sorted(bdir.glob("r*.json")) if not (adir / f.name).exists()]
    stopped = None
    try:
        make_t9.run_batches(files, adir, prompt, SCHEMA_REWRITE, args, "rewrite")
    except SystemExit as e:
        stopped = e
    tool_words = {"app", "ứng", "dụng", "calendar", "sms", "ewallet", "api", "tool"}
    for name in common.load()["catalog"]:
        tool_words |= set(re.split(r"[._]", name.lower()))
    tool_words -= {"get", "set", "list", "search", "book", "add", "create", "send", "cancel", "status", "price"}
    made = 0
    with open(REWRITES, "a", encoding="utf-8") as fout:
        for f in sorted(bdir.glob("r*.json")):
            a = adir / f.name
            if not a.exists():
                continue
            ans = {str(x.get("id")): x for x in json.loads(a.read_text(encoding="utf-8"))["reply"] if isinstance(x, dict)}
            for x in json.loads(f.read_text(encoding="utf-8")):
                v = (ans.get(x["rid"]) or {}).get("variants") or []
                if not v:
                    continue
                t = re.sub(r"\s+", " ", str(v[0])).strip()
                bad = agy.check_variant(t, {"text": x["text"], "keep": x["keep"]}, tool_words)
                fout.write(json.dumps({"kind": x["kind"], "id": x["id"], "style": x["style"], "text": t, "ok": not bad,
                                       "reasons": bad, "round": rnd, "replaces": x["bad"]}, ensure_ascii=False) + "\n")
                made += not bad
    print(f"round {rnd}: {made} rewrites passed the lexical checks", flush=True)
    if stopped:
        raise stopped
    return made


def loop(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = sources()
    for rnd in range(1, args.rounds + 1):
        review_round(args, src, rnd)
        # every variant flagged in this round, also those judged before a quota stop
        flagged = [v for v in latest_verdicts().values() if not v["same"] and v.get("round") == rnd]
        if not flagged or rnd == args.rounds:
            break
        if not rewrite_round(args, flagged, rnd):
            break
    report(args)


def passed() -> dict:
    """{(id, style): text} of the variants to use: every variant that passed the lexical checks and was not judged
    "different" (not yet reviewed counts as kept); a flagged one is replaced by its rewrite when that one passed."""
    verdict = latest_verdicts()
    out = {}
    for kind, i, s, t in variants():  # originals first, rewrites after: a kept rewrite replaces a flagged original
        v = verdict.get(key(i, s, t))
        if v is None or v["same"]:
            if (i, s) not in out or (v is not None and verdict.get(key(i, s, out[(i, s)]), {}).get("same") is False):
                out[(i, s)] = t
        elif (i, s) in out and out[(i, s)] == t:
            del out[(i, s)]
    for kind, i, s, t in variants():  # drop originals judged different when no rewrite took their place
        v = verdict.get(key(i, s, t))
        if v is not None and not v["same"] and out.get((i, s)) == t:
            del out[(i, s)]
    return out


def report(args) -> None:
    verdict = latest_verdicts()
    alls = variants()
    rep = {"variants": len(alls), "reviewed": sum(key(i, s, t) in verdict for _, i, s, t in alls),
           "same": sum(1 for _, i, s, t in alls if verdict.get(key(i, s, t), {}).get("same")),
           "different": sum(1 for _, i, s, t in alls if key(i, s, t) in verdict and not verdict[key(i, s, t)]["same"]),
           "usable (id, style)": len(passed())}
    whys = Counter(re.sub(r"[^\wÀ-ỹ ]", "", (v.get("why") or "")).split(" ")[0].lower()
                   for v in verdict.values() if not v["same"])
    rep["why (first word)"] = dict(whys.most_common(10))
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("loop")
    lp.add_argument("--rounds", type=int, default=3)
    lp.add_argument("--agy", default=str(Path.home() / ".local" / "bin" / "agy"))
    lp.add_argument("--model", default=None)
    lp.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    lp.add_argument("--workers", type=int, default=3)
    lp.add_argument("--limit", type=int, default=0)
    sub.add_parser("report")
    args = ap.parse_args()
    {"loop": loop, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
