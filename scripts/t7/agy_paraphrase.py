"""T7 of improvment_plan3.md: Gemini rewrites each train request in two spoken styles, through the Antigravity CLI.

    python scripts/t7/agy_paraphrase.py prepare            # Claude: batches of train requests + their anchors
    python scripts/t7/agy_paraphrase.py run [--model SLUG] # YOU, in your own terminal (not inside Claude Code)
    python scripts/t7/agy_paraphrase.py collect            # Claude: check every variant, write paraphrases.jsonl

Antigravity's terms allow your own scripts to call `agy -p` but not a third-party agent driving it, so `run` is meant
to be started by you; it refuses to start when CLAUDECODE is set. Claude writes no request text: the variants come only
from Gemini, and `collect` only keeps or drops them by rule (anchors kept, no new numbers, no app/tool names).

Run with LIFEMATE_DATA set. Everything written goes to outputs/p3/t7/ (gitignored: derived from competition data).
`run` is resumable: a batch with an answer file is skipped, so after a quota stop (429) run it again later.
Each agy call runs in an empty folder, so the agent has no files to read.
"""
import argparse
import json
import os
import random
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lifemate import data  # noqa: E402

OUT = REPO / "outputs" / "p3" / "t7"
BATCHES, ANSWERS = OUT / "batches", OUT / "answers"
BATCH_SIZE = 10
STYLES = {"Bắc": "giọng miền Bắc (nhé, giúp, xem, ạ)", "Trung": "giọng miền Trung (nghe, nha, coi, hỉ vừa phải)",
          "Nam": "giọng miền Nam (nha, giùm, coi, hen)"}
NUM_WORDS = {"không", "một", "mốt", "hai", "ba", "bốn", "tư", "năm", "lăm", "nhăm", "sáu", "bảy", "tám", "chín", "mười",
             "mươi", "trăm", "nghìn", "ngàn", "triệu", "tỷ", "tỉ", "rưỡi", "nửa", "linh", "lẻ", "chục"}
TIME_WORDS = ["hôm nay", "mai", "ngày kia", "tuần sau", "tuần này", "tháng sau", "tháng này", "cuối tuần", "thứ hai",
              "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật", "sáng", "trưa", "chiều", "tối", "đêm",
              "bây giờ", "hôm qua", "kém", "rưỡi"]
# Words that pick the tool or the channel: a variant that swaps "chuyển ví" for "gửi tiền" changes the right answer.
TOOL_WORDS = ["ví", "ngân hàng", "tài khoản", "tiết kiệm", "thẻ", "sms", "zalo", "mail", "email", "tin nhắn", "nhắn", "gọi",
              "báo thức", "nhắc", "lịch", "ghi chú", "hóa đơn", "nạp", "chuyển", "hủy", "xe", "kênh", "đèn", "quạt",
              "điều hòa", "máy lạnh", "máy sưởi", "cửa", "đơn", "vé", "podcast", "nhạc", "tivi", "công việc", "danh sách"]
UNIT_WORDS = {"nghìn", "ngàn", "triệu", "tỷ", "tỉ", "giờ", "phút", "tiếng", "độ", "phần", "trăm", "người", "chỗ", "ngày",
              "tuần", "tháng", "năm", "lần", "cái", "chiếc", "kg", "km", "bước", "đồng", "k", "rưỡi", "kém"}
TONE_PLACE = {"oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa", "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe",
              "oẹ": "ọe", "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy"}
FILLERS = {"ơi", "à", "ờ", "này", "nếu", "trợ", "em", "anh", "chị", "bạn", "nhờ", "cho", "giúp", "mình", "tôi", "hãy",
           "thôi", "ừ", "ừm", "vâng", "dạ", "khoan", "mà", "còn", "rồi", "với", "và", "lúc", "nhắc", "đặt", "gọi"}
INSTRUCTIONS = """Bạn viết lại các câu ra lệnh bằng giọng nói gửi trợ lý ảo trên điện thoại (tiếng Việt).
Với MỖI câu trong danh sách, viết đúng 2 cách nói khác, mỗi cách theo kiểu nói được ghi kèm.

Bắt buộc:
- Giữ nguyên ý định và mọi chi tiết: ai, việc gì, bao nhiêu, khi nào, ở đâu, điều kiện "nếu... thì...", thứ tự các việc.
- Mỗi cụm trong "giữ nguyên" phải xuất hiện y hệt (cùng chữ, cùng cách viết số) trong câu mới. Trong đó có các từ
  chỉ cách làm (ví, ngân hàng, SMS, Zalo, gọi, nhắn, báo thức, nhắc...): không thay bằng từ chung chung ("gửi tiền",
  "báo cho") hay từ khác nghĩa.
- Không thêm số, tên, giờ, ngày hay việc mới. Không bỏ việc nào.
- Không nhắc tên ứng dụng hay tên chức năng (không "app", "ví ABC", "calendar"...).
- Nói tự nhiên như người thật nói với điện thoại: có thể đổi trật tự, đổi từ đệm, thêm/bớt "ơi", "nhé", "giùm"... vừa phải.
- Một câu, không giải thích, không ngoặc kép.

Trả về JSON: {"items": [{"id": ..., "variants": [câu 1, câu 2]}, ...]} theo đúng thứ tự đầu vào.

Danh sách:
"""
SCHEMA = {"type": "object", "required": ["items"], "properties": {"items": {
    "type": "array", "items": {"type": "object", "required": ["id", "variants"],
                               "properties": {"id": {"type": "string"},
                                              "variants": {"type": "array", "items": {"type": "string"},
                                                           "minItems": 2, "maxItems": 2}}}}}}


def norm(s: str) -> str:
    s = s.lower().replace("ngàn", "nghìn")
    for old, new in TONE_PLACE.items():  # "huỷ" and "hủy" are one word
        s = s.replace(old, new)
    return re.sub(r"\s+", " ", s).strip()


def words(s: str) -> list:
    return re.findall(r"[\wÀ-ỹ]+", s, flags=re.UNICODE)


def number_tokens(s: str) -> list:
    """Digits, and number words inside a number: a run of two or more, or one before a unit ("hai tiếng"). A lone
    "không" (not) or "ba" (dad) is no number."""
    ws = [x.lower().replace("ngàn", "nghìn") for x in words(s)]
    out = [w for w in ws if w.isdigit()]
    i = 0
    while i < len(ws):
        if ws[i] in NUM_WORDS and not ws[i].isdigit():
            j = i
            while j < len(ws) and ws[j] in NUM_WORDS:
                j += 1
            if j - i >= 2 or (j < len(ws) and ws[j] in UNIT_WORDS):
                out += ws[i:j]
            i = j
        else:
            i += 1
    return sorted(out)


def anchors(row: dict) -> list:
    """Phrases a variant must keep: string arguments said word for word, numbers, time words, names."""
    text, low = row["audio_text"], norm(row["audio_text"])
    keep = []
    for call in row["label"]["tool_calls"]:
        for v in call["arguments"].values():
            if isinstance(v, str) and len(v) >= 2 and norm(v) in low:
                keep.append(v)
    for m in re.finditer(r"\d+(?:[.,:h]\d+)*(?:\s*(?:giờ|phút|nghìn|ngàn|triệu|k|%|tiếng|ngày|người|chỗ))?", text):
        keep.append(m.group(0))
    for t in TIME_WORDS + TOOL_WORDS:
        if re.search(rf"(?<![\wÀ-ỹ]){t}(?![\wÀ-ỹ])", low):
            keep.append(t)
    for sent in re.split(r"[.!?;,]\s*", text):
        for w in words(sent)[1:]:
            if w[0].isupper() and w.lower() not in FILLERS:
                keep.append(w)
    out = []
    for k in keep:  # drop phrases contained in a longer one
        if not any(k != o and norm(k) in norm(o) for o in keep) and k not in out:
            out.append(k)
    return out


def prepare(args) -> None:
    rows = data.load_split("train", data.find_root("train"))
    rng = random.Random(args.seed)
    BATCHES.mkdir(parents=True, exist_ok=True)
    items = []
    for r in rows:
        styles = rng.sample(sorted(STYLES), 2)
        items.append({"id": r["id"], "text": r["audio_text"], "styles": styles, "keep": anchors(r)})
    for b in range(0, len(items), BATCH_SIZE):
        (BATCHES / f"b{b // BATCH_SIZE:04d}.json").write_text(
            json.dumps(items[b:b + BATCH_SIZE], ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(items)} requests in {(len(items) + BATCH_SIZE - 1) // BATCH_SIZE} batches → {BATCHES}")


def prompt_of(batch: list) -> str:
    lines = []
    for it in batch:
        lines.append(f'- id: {it["id"]}\n  câu: {it["text"]}\n  kiểu nói: {STYLES[it["styles"][0]]}; '
                     f'{STYLES[it["styles"][1]]}\n  giữ nguyên: {json.dumps(it["keep"], ensure_ascii=False)}')
    return INSTRUCTIONS + "\n".join(lines)


def parse_reply(stdout: str):
    """The answer list from agy's JSON envelope (structured_output, else the first JSON list in the response)."""
    stdout = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", stdout).replace("\r", "")
    start = stdout.find("{")
    env = json.loads(stdout[start:stdout.rfind("}") + 1]) if start >= 0 else {}
    if env.get("status") not in (None, "SUCCESS"):
        raise RuntimeError(f"agy status {env.get('status')}: {env.get('error')}")
    out = env.get("structured_output")
    if isinstance(out, dict) and isinstance(out.get("items"), list):
        return out["items"], env.get("usage", {})
    if isinstance(out, list):
        return out, env.get("usage", {})
    resp = env.get("response", "")
    m = re.search(r"\{.*\}", resp, flags=re.S)
    if m:
        try:
            got = json.loads(m.group(0))
            if isinstance(got.get("items"), list):
                return got["items"], env.get("usage", {})
        except json.JSONDecodeError:
            pass
    m = re.search(r"\[.*\]", resp, flags=re.S)
    if not m:
        raise RuntimeError(f"no JSON answer in the reply: {resp[:200]!r}")
    return json.loads(m.group(0)), env.get("usage", {})


def call_agy(prompt: str, args, workdir: str) -> str:
    cmd = [args.agy, "-p", prompt, "--output-format", "json", "--json-schema", json.dumps(SCHEMA),
           "--print-timeout", "10m", "--effort", args.effort]
    if args.model:
        cmd += ["--model", args.model]
    p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        raise RuntimeError(f"agy exit {p.returncode}: {p.stderr.strip()[-500:]}")
    if p.stdout.strip():
        return p.stdout
    # agy issue #76: some builds print nothing when stdout is not a terminal; run it once more under a pseudo-terminal
    p = subprocess.run(["script", "-qefc", shlex.join(cmd), "/dev/null"], cwd=workdir, capture_output=True, text=True,
                       timeout=900)
    if p.returncode != 0 or not p.stdout.strip():
        raise RuntimeError(f"agy gave no output (exit {p.returncode}): {p.stderr.strip()[-500:]}")
    return p.stdout


def run(args) -> None:
    if os.environ.get("CLAUDECODE"):
        raise SystemExit("run this yourself in a normal terminal, not from Claude Code (Antigravity terms)")
    ANSWERS.mkdir(parents=True, exist_ok=True)
    todo = [b for b in sorted(BATCHES.glob("b*.json")) if not (ANSWERS / b.name).exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} batches to do ({len(list(ANSWERS.glob('b*.json')))} done)")
    with tempfile.TemporaryDirectory() as workdir:
        for i, b in enumerate(todo, 1):
            batch = json.loads(b.read_text(encoding="utf-8"))
            for attempt in range(3):
                try:
                    t = time.time()
                    reply, usage = parse_reply(call_agy(prompt_of(batch), args, workdir))
                    break
                except Exception as e:  # noqa: BLE001
                    msg = str(e)
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                        raise SystemExit(f"quota reached after {i - 1} batches; run again later (resumes). {msg[:300]}")
                    print(f"  {b.name} attempt {attempt + 1} failed: {msg[:300]}")
                    time.sleep(10 * (attempt + 1))
            else:
                print(f"  {b.name} skipped after 3 failures")
                continue
            (ANSWERS / b.name).write_text(json.dumps({"reply": reply, "usage": usage, "seconds": round(time.time() - t, 1)},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[{i}/{len(todo)}] {b.name} ok in {time.time() - t:.0f}s, tokens {usage.get('total_tokens')}")
            time.sleep(args.pause)


def check_variant(v: str, item: dict, tool_words: set) -> list:
    bad = []
    nv, src = norm(v), norm(item["text"])
    if nv == src:
        bad.append("same as source")
    for k in item["keep"]:
        if norm(k) not in nv:
            bad.append(f"lost {k!r}")
    got, want = set(number_tokens(v)), set(number_tokens(item["text"]))
    if got - want:
        bad.append(f"new numbers {sorted(got - want)}")
    if want - got:
        bad.append(f"lost numbers {sorted(want - got)}")
    hits = sorted(w for w in set(x.lower() for x in words(v)) if w in tool_words and w not in src)
    if hits:
        bad.append(f"app/tool words {hits}")
    if not 0.5 <= len(v) / max(len(item["text"]), 1) <= 1.8:
        bad.append("length")
    if "\n" in v or '"' in v:
        bad.append("format")
    return bad


def collect(args) -> None:
    root = data.find_root("train")
    tool_words = {"app", "ứng", "dụng", "calendar", "sms", "ewallet", "api", "tool"}
    for name in data.load_catalog("train", root):
        tool_words |= set(re.split(r"[._]", name.lower()))
    tool_words -= {"get", "set", "list", "search", "book", "add", "create", "send", "cancel", "status", "price"}
    rows, report = [], {"batches": 0, "variants": 0, "kept": 0, "reasons": {}}
    for a in sorted(ANSWERS.glob("b*.json")):
        batch = {it["id"]: it for it in json.loads((BATCHES / a.name).read_text(encoding="utf-8"))}
        report["batches"] += 1
        for ans in json.loads(a.read_text(encoding="utf-8"))["reply"]:
            item = batch.get(ans.get("id"))
            if not item:
                continue
            for style, v in zip(item["styles"], ans.get("variants", [])[:2]):
                v = re.sub(r"\s+", " ", str(v)).strip()
                bad = check_variant(v, item, tool_words)
                report["variants"] += 1
                report["kept"] += not bad
                for r in bad:
                    key = r.split(" ")[0] + (" " + r.split(" ")[1] if r.startswith(("new", "app", "lost numbers")) else "")
                    report["reasons"][key] = report["reasons"].get(key, 0) + 1
                rows.append({"id": item["id"], "style": style, "text": v, "ok": not bad, "reasons": bad})
    with open(OUT / "paraphrases.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "collect_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--seed", type=int, default=7)
    r = sub.add_parser("run")
    r.add_argument("--agy", default=str(Path.home() / ".local" / "bin" / "agy"))
    r.add_argument("--model", default=None, help="slug from `agy models`; default: agy's default model")
    r.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    r.add_argument("--limit", type=int, default=0, help="only this many batches (0 = all)")
    r.add_argument("--pause", type=float, default=3.0, help="seconds between calls")
    sub.add_parser("collect")
    args = ap.parse_args()
    {"prepare": prepare, "run": run, "collect": collect}[args.cmd](args)


if __name__ == "__main__":
    main()
