"""Second, targeted Gemini check of T7 rewordings whose action verb changed (13 Nov review, item 2).

    python scripts/p3/verb_review.py flag            # count the flagged variants
    python scripts/p3/verb_review.py run [--workers 3]   # Gemini (agy) verdicts -> outputs/p3/review/verb_verdicts.jsonl

A variant is flagged when its action verbs point to a different kind of tool than its source's (play / add / off /
delete / move / send / call / create; "check" verbs ignored): "mở bài X" -> "thêm bài X" may be a different tool. Gemini
is asked one narrow question per pair: would a phone assistant need a different action (a different tool, or the same
tool with a different operation) to carry out B than A? make_t7.py drops every variant judged "different".
Claude writes no request text here: Gemini only judges, the code routes.
"""
import argparse
import json
import re

import common
import make_t7
import make_t9

OUT = common.OUT / "review"
VERDICTS = OUT / "verb_verdicts.jsonl"
PAIRS = 40
G = {"play": ["mở", "phát", "bật", "chơi"],
     "add": ["thêm", "lấy", "cho vào", "bỏ vào", "đưa vào", "vô danh sách", "hàng chờ", "lưu"],
     "off": ["tắt", "dừng", "ngừng"], "del": ["xóa", "xoá", "hủy", "huỷ", "bỏ"],
     "move": ["dời", "lùi", "đổi", "chuyển sang", "hoãn"], "send": ["nhắn", "gửi"], "call": ["gọi"],
     "check": ["kiểm tra", "xem", "coi", "tra", "dò"], "create": ["tạo", "đặt", "ghi", "lên lịch", "nhắc"]}

PROMPT = """Bạn kiểm tra dữ liệu cho trợ lý ảo trên điện thoại. Mỗi mục có câu A (yêu cầu gốc) và câu B (cách nói khác
của cùng người). Câu hỏi duy nhất: để làm theo B, trợ lý có phải dùng một THAO TÁC KHÁC với A không? Thao tác khác là:
một chức năng khác (ví dụ "mở/phát bài hát" khác "thêm bài vào danh sách", "tắt" khác "xoá", "nhắn tin" khác "gọi",
"tạo nhắc nhở" khác "đặt lịch hẹn", "dời" khác "huỷ"), hoặc cùng chức năng nhưng làm việc khác đi (thêm/bớt việc, đổi đối
tượng). Chỉ khác cách diễn đạt, giọng địa phương, từ đồng nghĩa cho CÙNG thao tác thì là "same".

Trả về JSON: {{"items": [{{"id": ..., "same": true/false, "why": "nếu khác: thao tác của A và của B, ngắn gọn"}}, ...]}} đúng thứ tự.

{pairs}"""
SCHEMA = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "required": ["id", "same"],
    "properties": {"id": {"type": "string"}, "same": {"type": "boolean"}, "why": {"type": "string"}}}}}}


def groups(t: str) -> set:
    t = t.lower()
    return {g for g, ws in G.items() for w in ws if re.search(rf"(?<![\wÀ-ỹ]){w}(?![\wÀ-ỹ])", t)}


def key(rid: str, style: str, text: str) -> str:
    import agy_review
    return agy_review.key(rid, style, text)


def flagged() -> list:
    """[(key, id, style, source, variant)] of kept T7 variants whose verb groups differ from the source's."""
    rows = {r["id"]: r for r in common.load()["rows"]}
    out = []
    for rid, vs in sorted(make_t7.paraphrases().items()):
        a = rows[rid]["audio_text"]
        for style, t in vs:
            if make_t7.fillers_only(a, t):
                continue
            ga, gb = groups(a) - {"check"}, groups(t) - {"check"}
            if ga != gb:
                out.append((key(rid, style, t), rid, style, a, t))
    return out


def verdicts() -> dict:
    p = VERDICTS
    return {v["key"]: v for v in map(json.loads, p.read_text(encoding="utf-8").splitlines())} if p.exists() else {}


def run(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    done = verdicts()
    todo = [x for x in flagged() if x[0] not in done]
    bdir, adir = OUT / "verb_b", OUT / "verb_a"
    bdir.mkdir(parents=True, exist_ok=True)
    if not list(bdir.glob("v*.json")):
        for b in range(0, len(todo), PAIRS):
            part = [{"key": k, "id": i, "style": s, "a": a, "b": t} for k, i, s, a, t in todo[b:b + PAIRS]]
            (bdir / f"v{b // PAIRS:04d}.json").write_text(json.dumps(part, ensure_ascii=False), encoding="utf-8")

    def prompt(batch):
        return PROMPT.format(pairs="\n".join(f"- id: {x['key']}\n  A: {x['a']}\n  B: {x['b']}" for x in batch))
    files = [f for f in sorted(bdir.glob("v*.json")) if not (adir / f.name).exists()]
    try:
        make_t9.run_batches(files, adir, prompt, SCHEMA, args, "verb")
    finally:
        n = {"same": 0, "different": 0, "missing": 0}
        with open(VERDICTS, "a", encoding="utf-8") as fout:
            for f in sorted(bdir.glob("v*.json")):
                a = adir / f.name
                ans = {}
                if a.exists():
                    ans = {str(x.get("id")): x for x in json.loads(a.read_text(encoding="utf-8"))["reply"]
                           if isinstance(x, dict)}
                for x in json.loads(f.read_text(encoding="utf-8")):
                    v = ans.get(x["key"])
                    if x["key"] in done:
                        continue
                    if v is None:
                        n["missing"] += 1
                        continue
                    same = bool(v.get("same"))
                    n["same" if same else "different"] += 1
                    fout.write(json.dumps({**x, "same": same, "why": v.get("why", "")}, ensure_ascii=False) + "\n")
                    done[x["key"]] = 1
        print(json.dumps(n), flush=True)


def different() -> set:
    """Keys of variants Gemini judged to need a different action."""
    return {k for k, v in verdicts().items() if not v["same"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("flag")
    rp = sub.add_parser("run")
    rp.add_argument("--agy", default=str(common.Path.home() / ".local" / "bin" / "agy") if hasattr(common, "Path")
                    else str(__import__("pathlib").Path.home() / ".local" / "bin" / "agy"))
    rp.add_argument("--model", default=None)
    rp.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    rp.add_argument("--workers", type=int, default=3)
    rp.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "flag":
        f = flagged()
        print(json.dumps({"flagged": len(f), "already_judged": sum(x[0] in verdicts() for x in f)}))
    else:
        run(args)


if __name__ == "__main__":
    main()
