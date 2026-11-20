"""T9 of improvment_plan3.md: new requests written by Gemini inside real train situations, kept only when a second,
blind Gemini pass gives the same calls and the simulator accepts them.

    python scripts/p3/make_t9.py prepare [--per 4]     # one batch per train sample: its apps, state, train examples
    python scripts/p3/make_t9.py run --stage gen       # Gemini writes requests + calls        (parallel, resumable)
    python scripts/p3/make_t9.py run --stage judge     # Gemini answers the same requests blind (parallel, resumable)
    python scripts/p3/make_t9.py collect               # agreement + replay + convention checks -> t9/kept.jsonl
    python scripts/p3/make_t9.py build                 # outputs/p3/t9_full.jsonl, t9_fold0.jsonl

Why: every other type either repeats train answers (T2a/T2b/T7) or covers few situations (T4/T5/T8). T9 brings new
answers on the train apps, in the 1,545 real phone states (contacts, devices, events, bills...), with the train
conventions shown by real examples of the same apps.

A batch shows Gemini one sample's time, profile, a few installed apps (tool specs from the train catalog), that app
data from the sample's train state, and up to three real train requests with their gold calls on those apps. It
writes new requests a user could say there, with the calls. The judge sees the same page without the calls and answers
each request. A request is kept when:
- judge and writer give the same tool sequence and the same arguments (free-text content excepted, only present);
- replaying the calls on the sample's train state runs every call (zaic_tools.ToolEnv), writes succeed, and every id
  argument (x-ref in the catalog) appeared in an earlier call's result (train's convention: read before write);
- it names no app or tool, has no question to the user and no out-of-scope request (T4/T5 cover those).
The results in the conversation are the replayed ones. Requests are heard through T2b's noise like T7/T8.
Claude writes no request text and no calls: both come from Gemini; Claude's code only selects and checks.
Fold 0: batches from the 386 validation samples are left out, and their requests are never shown as examples.
"""
import argparse
import copy
import json
import random
import re
import shlex
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import common
import review_rules
from make_t6 import asr_numbers

sys.path.insert(0, str(common.REPO / "scripts" / "t7"))
import agy_paraphrase as agy  # noqa: E402

OUT = common.OUT / "t9"
BATCHES = OUT / "batches"
ANS = {"gen": OUT / "gen", "judge": OUT / "judge"}
KEPT = OUT / "kept.jsonl"
ASK = "sys.ask_missing_info"

GEN = """Bạn tạo dữ liệu huấn luyện cho một trợ lý ảo trên điện thoại người Việt: người dùng NÓI một yêu cầu, trợ lý
gọi các công cụ (tool) theo đúng thứ tự.

Dưới đây là: thời điểm hiện tại và hồ sơ người dùng, các công cụ của vài ứng dụng đã cài (tên, mô tả, tham số), dữ liệu
hiện có trong máy của các ứng dụng đó, và vài VÍ DỤ THẬT (câu nói + các lời gọi đúng).

Hãy viết {n} yêu cầu MỚI, khác các ví dụ, mỗi yêu cầu kèm danh sách lời gọi đúng. Yêu cầu:
- Câu nói tự nhiên như người Việt nói với điện thoại (có thể có "ơi", "nhé", "nha", "giùm", "với", nói tắt, kiểu Bắc/
  Trung/Nam), một hoặc hai câu, dùng 1 đến 3 việc. Không nói tên ứng dụng hay tên công cụ.
- Chỉ nhắc tới người, thiết bị, lịch, món... CÓ TRONG dữ liệu, gọi đúng tên như trong dữ liệu (có thể kèm danh xưng
  như "anh", "chị", "cô" nếu dữ liệu có). Số tiền, giờ, số lượng nói như người thật (bằng chữ hoặc số).
- Yêu cầu phải đủ thông tin để làm ngay (không phải hỏi lại), và làm được bằng các công cụ đã cho.
- Lời gọi theo đúng quy ước của các ví dụ: mọi ID (contact_id, device_id, event_id, chat_id...) phải lấy từ kết quả
  của một lời gọi ĐỌC trước đó trong cùng danh sách (vd contacts.search trước sms.send, smart_home.get_status trước
  khi bật đèn), dù bạn đã thấy ID trong dữ liệu. Thời gian dạng ISO có múi giờ +07:00, tính từ thời điểm hiện tại.
  Tham số đúng kiểu và đúng tên như mô tả, không thêm tham số không cần.
- Nội dung tin nhắn (content...) ngắn, tự nhiên, đúng ý người nói.

Trả về JSON: {{"items": [{{"request": "...", "calls": [{{"name": "...", "arguments": {{...}}}}, ...]}}, ...]}}

{page}"""

JUDGE = """Bạn là trợ lý ảo trên điện thoại người Việt. Với MỖI yêu cầu bên dưới, hãy cho danh sách lời gọi công cụ đúng
mà bạn sẽ thực hiện, theo đúng quy ước của các ví dụ thật: mọi ID phải lấy từ kết quả của một lời gọi ĐỌC trước đó trong
cùng danh sách (vd contacts.search trước sms.send, smart_home.get_status trước khi bật đèn); thời gian dạng ISO có múi giờ
+07:00 tính từ thời điểm hiện tại; tham số đúng tên và kiểu, không thêm tham số không cần. Nếu yêu cầu thiếu thông tin
bắt buộc, trả về lời gọi sys.ask_missing_info; nếu không làm được bằng các công cụ đã cho, trả về danh sách rỗng.

Trả về JSON: {{"items": [{{"id": ..., "calls": [{{"name": "...", "arguments": {{...}}}}, ...]}}, ...]}} theo đúng thứ tự.

{page}

Các yêu cầu:
{requests}"""

CALL = {"type": "object", "required": ["name", "arguments"],
        "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}}}
SCHEMA = {"gen": {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
              "type": "object", "required": ["request", "calls"],
              "properties": {"request": {"type": "string"}, "calls": {"type": "array", "items": CALL}}}}}},
          "judge": {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
              "type": "object", "required": ["id", "calls"],
              "properties": {"id": {"type": "string"}, "calls": {"type": "array", "items": CALL}}}}}}}


def apps_of(calls: list) -> set:
    return {c["name"].split(".")[0] for c in calls} - {"sys"}


def trim(obj, depth: int = 0):
    """App data small enough for a prompt: lists cut to 10 records, long strings cut."""
    if isinstance(obj, list):
        return [trim(x, depth + 1) for x in obj[:10]]
    if isinstance(obj, dict):
        return {k: trim(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, str) and len(obj) > 120:
        return obj[:120] + "…"
    return obj


def spec_line(spec: dict) -> str:
    props = spec.get("parameters", {}).get("properties", {})
    req = spec.get("parameters", {}).get("required", [])
    ps = []
    for k, v in props.items():
        t = v.get("type", "")
        extra = f" {v['enum']}" if v.get("enum") else ""
        ps.append(f"{k}{'*' if k in req else ''}: {t}{extra}" + (f" — {v['description']}" if v.get("description") else ""))
    one = spec.get("parameters", {}).get("x-one-of")
    return (f"- {spec['name']} ({spec.get('kind', '')}): {spec.get('description', '')}\n    "
            + ("; ".join(ps) or "(không tham số)") + (f"\n    bắt buộc một trong: {one}" if one else ""))


def page_of(row: dict, apps: list, catalog: dict, examples: list) -> str:
    ctx = row["context"]
    prof = ctx.get("user_profile", {})
    state = json.loads((common.data.find_root("train") / "train" / "train_environment" / row["id"] / "state.json")
                       .read_text(encoding="utf-8"))
    tools = [t for a in apps for t in ctx["installed_apps"][a].get("tools", [])]
    lines = [f"Thời điểm hiện tại: {ctx['current_time']} (Asia/Ho_Chi_Minh). Người dùng: {prof.get('name', '')}, "
             f"{prof.get('city', '')}; nhà: {prof.get('home_address', '')}; nơi làm: {prof.get('work_address', '')}.",
             "", "Công cụ (* = bắt buộc):"]
    lines += [spec_line(catalog[t]) for t in tools if t in catalog]
    lines += ["", "Dữ liệu trong máy:"]
    for a in apps:
        if a in state:
            lines.append(f"{a}: {json.dumps(trim(state[a]), ensure_ascii=False)}")
    if examples:
        lines += ["", "Ví dụ thật (ở các tình huống khác, ID có thể khác):"]
        for e in examples:
            calls = [{"name": c["name"], "arguments": c.get("arguments") or {}} for c in e["label"]["tool_calls"]]
            lines.append(f"- Câu nói: {e['audio_text']}\n  Lời gọi: {json.dumps(calls, ensure_ascii=False)}")
    return "\n".join(lines)


def prepare(args) -> None:
    d = common.load()
    rng = random.Random(args.seed)
    rows = d["rows"]
    pool = [r for r in rows if r["id"] not in d["hold"] and not r["history"] and r["label"]["tool_calls"]
            and r["label"]["tool_calls"][-1]["name"] != ASK]
    by_app = defaultdict(list)
    for r in pool:
        for a in apps_of(r["label"]["tool_calls"]):
            by_app[a].append(r)
    BATCHES.mkdir(parents=True, exist_ok=True)
    for f in BATCHES.glob("b*.json"):
        f.unlink()
    n = 0
    order = rows[:]
    rng.shuffle(order)
    for r in order[:args.samples]:
        installed = [a for a in r["context"]["installed_apps"] if a != "contacts" and by_app.get(a)]
        if not installed:
            continue
        k = min(len(installed), rng.choice([2, 3, 3, 4]))
        # favour apps the model meets least in train, so the new answers widen what it knows
        weight = {a: 1 / (len(by_app[a]) ** 0.5) for a in installed}
        apps = []
        for _ in range(k):
            left = [a for a in installed if a not in apps]
            apps.append(rng.choices(left, [weight[a] for a in left])[0])
        if "contacts" in r["context"]["installed_apps"]:
            apps.append("contacts")
        exs = [e for a in apps if a != "contacts" for e in by_app[a] if e["id"] != r["id"]
               and apps_of(e["label"]["tool_calls"]) <= set(r["context"]["installed_apps"])]
        rng.shuffle(exs)
        exs.sort(key=lambda e: not apps_of(e["label"]["tool_calls"]) <= set(apps))  # examples on the shown apps first
        seen, picked = set(), []
        for e in exs:
            if e["id"] not in seen and len(picked) < 3:
                seen.add(e["id"])
                picked.append(e)
        page = page_of(r, apps, d["catalog"], picked)
        (BATCHES / f"b{n:04d}.json").write_text(json.dumps(
            {"id": f"b{n:04d}", "source": r["id"], "apps": apps, "examples": [e["id"] for e in picked],
             "page": page, "per": args.per}, ensure_ascii=False, indent=1), encoding="utf-8")
        n += 1
    print(json.dumps({"batches": n, "requests": n * args.per}))


def call_agy(prompt: str, schema: dict, args, workdir: str) -> str:
    cmd = [args.agy, "-p", prompt, "--output-format", "json", "--json-schema", json.dumps(schema),
           "--print-timeout", "10m", "--effort", args.effort]
    if args.model:
        cmd += ["--model", args.model]
    p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        raise RuntimeError(f"agy exit {p.returncode}: {p.stderr.strip()[-500:]}")
    if p.stdout.strip():
        return p.stdout
    p = subprocess.run(["script", "-qefc", shlex.join(cmd), "/dev/null"], cwd=workdir, capture_output=True, text=True,
                       timeout=900)
    if p.returncode != 0 or not p.stdout.strip():
        raise RuntimeError(f"agy gave no output (exit {p.returncode}): {p.stderr.strip()[-500:]}")
    return p.stdout


def prompt_for(stage: str, batch: dict) -> str:
    if stage == "gen":
        return GEN.format(n=batch["per"], page=batch["page"])
    gen = json.loads((ANS["gen"] / f"{batch['id']}.json").read_text(encoding="utf-8"))["reply"]
    reqs = "\n".join(f"- id: {batch['id']}_{k}\n  câu: {it.get('request', '')}" for k, it in enumerate(gen))
    return JUDGE.format(page=batch["page"], requests=reqs)


def reset_seconds(msg: str):
    """Seconds until agy's capacity resets, from "Resets in 1h2m3s" / "Resets in 40s"; None when not given."""
    m = re.search(r"Resets in ((?:\d+h)?(?:\d+m)?(?:\d+s)?)", msg)
    if not m or not m.group(1):
        return None
    t = m.group(1)
    return sum(int(n) * {"h": 3600, "m": 60, "s": 1}[u] for n, u in re.findall(r"(\d+)([hms])", t))


def run(args) -> None:
    stage = args.stage
    todo = [b for b in sorted(BATCHES.glob("b*.json")) if not (ANS[stage] / b.name).exists()
            and (stage == "gen" or (ANS["gen"] / b.name).exists())]
    run_batches(todo, ANS[stage], lambda batch: prompt_for(stage, batch), SCHEMA[stage], args, stage)


def run_batches(todo: list, out: Path, prompt_fn, schema: dict, args, stage: str = "") -> None:
    """agy over batch files in parallel; a short capacity limit waits and retries, a long one stops the run."""
    import os
    if os.environ.get("CLAUDECODE"):
        raise SystemExit("start with CLAUDECODE unset (see scripts/t7/agy_paraphrase.py)")
    out.mkdir(parents=True, exist_ok=True)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{stage}: {len(todo)} batches to do with {args.workers} workers", flush=True)
    stop = {"quota": None}

    def one(path: Path):
        if stop["quota"]:
            return path.name, "skipped"
        batch = json.loads(path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as wd:
            for attempt in range(6):
                try:
                    t = time.time()
                    reply, usage = agy.parse_reply(call_agy(prompt_fn(batch), schema, args, wd))
                    (out / path.name).write_text(json.dumps({"reply": reply, "usage": usage,
                                                             "seconds": round(time.time() - t, 1)},
                                                            ensure_ascii=False, indent=1), encoding="utf-8")
                    return path.name, f"ok {time.time() - t:.0f}s"
                except Exception as e:  # noqa: BLE001
                    msg = str(e)
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                        wait = reset_seconds(msg)
                        if wait is not None and wait <= 600:  # a per-minute capacity limit: wait and retry
                            time.sleep(wait + 20 + 10 * attempt)
                            continue
                        stop["quota"] = msg[:300]
                        return path.name, "quota"
                    time.sleep(5 * (attempt + 1))
            return path.name, f"failed: {msg[:200]}"

    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(one, p) for p in todo]):
            name, status = f.result()
            done += 1
            print(f"[{done}/{len(todo)}] {name} {status}", flush=True)
    if stop["quota"]:
        raise SystemExit(f"QUOTA: {stop['quota']}")


# ------------------------------------------------------------------------------------------------- checking
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def same_calls(a: list, b: list, catalog: dict) -> bool:
    if [c.get("name") for c in a] != [c.get("name") for c in b]:
        return False
    for x, y in zip(a, b):
        props = (catalog.get(x["name"]) or {}).get("parameters", {}).get("properties", {})
        ax, bx = x.get("arguments") or {}, y.get("arguments") or {}
        if set(ax) != set(bx):
            return False
        for k in ax:
            if props.get(k, {}).get("x-match") == "fuzzy":
                continue  # free text: both have it, wording may differ
            if norm(json.dumps(ax[k], ensure_ascii=False, sort_keys=True)) != norm(json.dumps(bx[k], ensure_ascii=False,
                                                                                               sort_keys=True)):
                return False
    return True


def tool_words(catalog: dict) -> set:
    out = {"app", "ứng dụng", "calendar", "ewallet", "api", "tool"}
    for name in catalog:
        out |= {w for w in re.split(r"[._]", name.lower()) if len(w) >= 4}
    return out - {"list", "search", "send", "create", "cancel", "status", "price", "update", "book", "order", "play",
                  "call", "chat", "news", "home", "note", "notes", "tasks", "music", "alarm", "phone", "weather"}


def scalars(obj) -> list:
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in scalars(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in scalars(v)]
    return [obj]


DAYS = [(re.compile(r"(?<![\wÀ-ỹ])(hôm nay|bữa nay|sáng nay|trưa nay|chiều nay|tối nay|đêm nay|nay)(?![\wÀ-ỹ])"), 0),
        (re.compile(r"(?<![\wÀ-ỹ])(ngày mai|sáng mai|trưa mai|chiều mai|tối mai|mai)(?![\wÀ-ỹ])"), 1),
        (re.compile(r"(?<![\wÀ-ỹ])(ngày mốt|ngày kia|mốt)(?![\wÀ-ỹ])"), 2)]
OTHER_DAY = re.compile(r"(?<![\wÀ-ỹ])(thứ|chủ nhật|tuần|tháng|cuối tuần|hàng ngày|mỗi ngày|ngày \d|hôm qua|tới|sau)(?![\wÀ-ỹ])")
CLOCK = re.compile(r"(?<![\wÀ-ỹ\d])(\d{1,2}|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mười một|mười hai)\s*(?:giờ|h)"
                   r"(?:\s*(rưỡi|\d{1,2}|kém))?(?:\s+\w+)?\s*(sáng|trưa|chiều|tối|đêm)?")
HOURW = {"một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10,
         "mười một": 11, "mười hai": 12}


def times_ok(req: str, calls: list, now: str) -> str:
    """Said day and clock against the ISO times of the calls: "mai" is tomorrow, "8 giờ tối" is 20:00."""
    import datetime as dt
    isos = [v for c in calls for v in (c.get("arguments") or {}).values()
            if isinstance(v, str) and re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", v)]
    dates = [v for c in calls for v in (c.get("arguments") or {}).values()
             if isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v)]
    if not isos and not dates:
        return ""
    low = req.lower()
    today = dt.date.fromisoformat(now[:10])
    said = {off for rx, off in DAYS if rx.search(low)}
    if len(said) == 1 and not OTHER_DAY.search(low):
        want = today + dt.timedelta(days=said.pop())
        got = {dt.date.fromisoformat(v[:10]) for v in isos + dates}
        if want not in got:
            return f"day {sorted(map(str, got))} != {want}"
    for m in CLOCK.finditer(low):
        h = int(m.group(1)) if m.group(1).isdigit() else HOURW.get(m.group(1))
        if h is None or h > 24:
            continue
        part = m.group(3)
        hours = {h % 24}
        if part in ("chiều", "tối") and h < 12:
            hours = {h + 12}
        elif part == "đêm" and h < 12:
            hours = {h + 12, h}
        elif part == "trưa":
            hours = {h, h + 12 if h < 12 else h}
        elif part is None and h < 12:
            hours = {h, h + 12}
        if m.group(2) == "kém":
            hours |= {(x - 1) % 24 for x in hours}
        if isos and not any(int(v[11:13]) in hours for v in isos):
            return f"clock {m.group(0)!r} vs {[v[11:16] for v in isos]}"
    return ""


def check(item: dict, judge: list, row: dict, catalog: dict, make_env, twords: set) -> str:
    req, calls = str(item.get("request", "")).strip(), item.get("calls") or []
    if not req or not calls or len(calls) > 8:
        return "empty or too long"
    if any(c.get("name") == ASK for c in calls) or any(c.get("name") == ASK for c in judge):
        return "ask"
    if not same_calls(calls, judge, catalog):
        return "judge differs"
    low = req.lower()
    if any(re.search(rf"(?<![\wÀ-ỹ]){re.escape(w)}(?![\wÀ-ỹ])", low) for w in twords):
        return "names an app/tool"
    if "?" in req and not re.search(r"(không|chưa|nào|bao nhiêu|gì|mấy|sao|đâu)\s*[?]", low):
        return "question mark"
    bad_time = times_ok(req, calls, row["context"]["current_time"])
    if bad_time:
        return "time " + bad_time
    allowed = {t for a in row["context"]["installed_apps"].values() for t in a.get("tools", [])}
    from zaic_tools.validate import schema_error
    for c in calls:
        if c["name"] not in allowed:
            return f"tool not installed {c['name']}"
        err = schema_error(c["name"], c.get("arguments") or {}, catalog, allowed)
        if err:
            return f"schema {c['name']}"
    env = make_env(row["id"])
    results, seen = [], set()
    for c in calls:
        props = (catalog.get(c["name"]) or {}).get("parameters", {}).get("properties", {})
        for k, v in (c.get("arguments") or {}).items():
            if props.get(k, {}).get("x-ref") and isinstance(v, str) and v not in seen:
                return f"id {k} not read before"
        res = env.call(c["name"], copy.deepcopy(c.get("arguments") or {}))
        if not isinstance(res, dict) or res.get("ok") is False:
            return f"call fails {c['name']}"
        if catalog.get(c["name"], {}).get("kind") == "write" and res.get("done") is False:
            return f"write not done {c['name']}"
        if catalog.get(c["name"], {}).get("kind") == "read" and not any(
                isinstance(v, list) and v for v in res.values()) and len(res) <= 1:
            return f"read found nothing {c['name']}"
        seen |= {str(x) for x in scalars(res) if isinstance(x, (str, int))}
        results.append(res)
    item["results"] = results
    return ""


def collect(args) -> None:
    d = common.load()
    by_id = {r["id"]: r for r in d["rows"]}
    root = common.data.find_root("train")
    sys.path.insert(0, str(root / "train"))
    from zaic_tools import ToolEnv
    make_env = lambda sid: ToolEnv("train", sid, root=str(root))  # noqa: E731
    twords = tool_words(d["catalog"])
    train_texts = {norm(r["audio_text"]) for r in d["rows"]}
    reasons, kept, seen = Counter(), [], set()
    for g in sorted(ANS["gen"].glob("b*.json")):
        j = ANS["judge"] / g.name
        if not j.exists():
            continue
        batch = json.loads((BATCHES / g.name).read_text(encoding="utf-8"))
        gen = json.loads(g.read_text(encoding="utf-8"))["reply"]
        judge = {str(it.get("id")): it.get("calls") or [] for it in json.loads(j.read_text(encoding="utf-8"))["reply"]}
        row = by_id[batch["source"]]
        for k, item in enumerate(gen):
            if not isinstance(item, dict):
                reasons["bad format"] += 1
                continue
            why = check(item, judge.get(f"{batch['id']}_{k}", []), row, d["catalog"], make_env, twords)
            key = norm(item.get("request", ""))
            if not why and (key in train_texts or key in seen):
                why = "duplicate"
            reasons[why.split(" ")[0] + (" " + why.split(" ")[1] if why.startswith(("judge", "call", "write", "id", "time"))
                                         else "") if why else "kept"] += 1
            if why:
                continue
            seen.add(key)
            kept.append({"id": f"{batch['id']}_{k}", "source": batch["source"], "apps": batch["apps"],
                         "request": re.sub(r"\s+", " ", item["request"]).strip(),
                         "calls": [{"name": c["name"], "arguments": c.get("arguments") or {}} for c in item["calls"]],
                         "results": item["results"]})
    with open(KEPT, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tools = Counter(c["name"] for r in kept for c in r["calls"])
    report = {"kept": len(kept), "reasons": dict(reasons.most_common()), "tools": len(tools),
              "top_tools": tools.most_common(8)}
    (OUT / "collect_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


# Train conventions (13 Nov T9 review). The scorer needs the exact key set of every call, and Gemini (writer and judge
# alike) fills optional filters train's gold never uses: tasks.list status 98 % (train 0 of 14), list_bills type 89 %
# (0 of 112), calendar.list_events dates 100 % (17 % of 105), banking.transfer note 97 % (1 %), alarm label/repeat...
# A row with a call key that train uses in under OFF_SHARE of that tool's calls (tools with >= 5 train calls) is left
# out: it would teach calls that score 0. Gemini also reuses made-up phone numbers (0988776655 in 30 rows) and piles up
# some tools (contacts.update 147 rows, 16x its train share): each phone-like value is kept in at most 3 rows and no tool
# appears in more than max(3 %, 3x its train share) of T9 rows.
OFF_SHARE = 0.20


def conventions(rows: list) -> tuple:
    n, keys = Counter(), defaultdict(Counter)
    for r in rows:
        for c in r["label"]["tool_calls"]:
            n[c["name"]] += 1
            keys[c["name"]].update((c.get("arguments") or {}).keys())
    return n, keys


def off_convention(calls: list, n: Counter, keys: dict) -> list:
    return [(c["name"], k) for c in calls if n[c["name"]] >= 5
            for k in (c.get("arguments") or {}) if keys[c["name"]][k] / n[c["name"]] < OFF_SHARE]


def key_sets(rows: list) -> set:
    """(tool, key set) pairs that train's gold calls or history calls use."""
    out = set()
    for r in rows:
        # gold only (third review: get_quotes with "vehicle" is a history shape; train's gold is pickup + dropoff, 2/2)
        out |= {(c["name"], frozenset(c.get("arguments") or {})) for c in r["label"]["tool_calls"]}
    gold = {c["name"] for r in rows for c in r["label"]["tool_calls"]}
    return {x for x in out if x[0] in gold}  # contacts.get: history only (4), never a gold answer


def shape_problem(calls: list, seen: set) -> str:
    """Second Opus review (13 Nov): a (tool, key set) train never shows (contacts.get, list_transactions with limit),
    a route to bare "nhà"/"công ty" (train gives the address from the profile), note content opening with punctuation."""
    for c in calls:
        a = c.get("arguments") or {}
        if (c["name"], frozenset(a)) not in seen:
            return "key set train never uses"
        if c["name"] == "maps.get_route" and any(str(v).strip().lower() in ("nhà", "công ty", "nhà riêng") for v in a.values()):
            return "bare home/work place"
        if c["name"].startswith("notes.") and isinstance(a.get("content"), str) and re.match(r"^\W", a["content"]):
            return "note content opens with punctuation"
    return ""


def arg_orders(rows: list) -> dict:
    """(tool, key set) -> train's most common key order (second Opus review: 35 % of T9 calls had keys in alphabetical
    order, which train never uses and which marks a row as synthetic)."""
    c = defaultdict(Counter)
    for r in rows:
        for x in list(r["label"]["tool_calls"]) + [y for h in r["history"] for y in h.get("tool_calls") or []]:
            a = x.get("arguments") or {}
            c[(x["name"], frozenset(a))][tuple(a)] += 1
    return {k: v.most_common(1)[0][0] for k, v in c.items()}


def enum_values(rows: list) -> dict:
    """(tool, key) -> the values train's gold uses, for keys with a small closed set (>= 5 uses, <= 8 values)."""
    c = defaultdict(Counter)
    for r in rows:
        for x in r["label"]["tool_calls"]:
            for k, v in (x.get("arguments") or {}).items():
                if isinstance(v, str):
                    c[(x["name"], k)][v] += 1
    # only code-like values ("daily", "buy", "bus"); ids, phones, names and places are open
    return {k: set(v) for k, v in c.items() if sum(v.values()) >= 5 and len(v) <= 8
            and all(re.fullmatch(r"[a-z_]+", x) for x in v)}


def value_problem(k: dict, enums: dict, queries: set) -> str:
    """Second Opus review: contacts.search by a bare relation word ("vợ": train searches the stored name), a closed-set
    value train never uses (medication days "weekdays"), a 10-digit phone (train's are 11), a digit run of four or more
    in the request (the speech model writes lottery and phone numbers in words)."""
    for c in k["calls"]:
        for key, v in (c.get("arguments") or {}).items():
            if c["name"] == "contacts.search" and key == "query" and str(v).lower() in RELATION_WORDS and v not in queries:
                return "contacts lookup by relation word"
            if (c["name"], key) in enums and isinstance(v, str) and re.fullmatch(r"[a-z_]+", v) \
                    and v not in enums[(c["name"], key)]:
                return "value train never uses"
            if "phone" in key and len(re.sub(r"\D", "", str(v))) == 10:
                return "10-digit phone"
    if re.search(r"\d{4,}", re.sub(r"\d{1,3}(?:[.,]\d{3})+", "", k["request"])):
        return "digit run in speech"
    return ""


RELATION_WORDS = {"vợ", "chồng", "con", "con gái", "con trai", "anh trai", "em gái", "mẹ", "bố", "ba", "má", "sếp"}


def select(kept: list, rows: list) -> tuple:
    """The kept requests that follow train's conventions, with repeated made-up values and over-used tools capped."""
    n, keys = conventions(rows)
    seen = key_sets(rows)
    row_share = Counter(t for r in rows for t in {c["name"] for c in r["label"]["tool_calls"]})
    why, out, phones, tools = Counter(), [], Counter(), Counter()
    ok = [k for k in kept if not off_convention(k["calls"], n, keys) or why.update(["convention"])]
    ok = [k for k in ok if not shape_problem(k["calls"], seen) or why.update([shape_problem(k["calls"], seen)])]
    enums = enum_values(rows)
    queries = {str(c["arguments"].get("query")) for r in rows for c in r["label"]["tool_calls"] if c["name"] == "contacts.search"}
    ok = [k for k in ok if not value_problem(k, enums, queries) or why.update([value_problem(k, enums, queries)])]
    rng = random.Random(9)
    rng.shuffle(ok)  # caps keep a random subset, not the first batches
    for k in ok:
        ph = {str(v) for c in k["calls"] for v in (c.get("arguments") or {}).values()
              if isinstance(v, (str, int)) and not isinstance(v, bool) and len(re.sub(r"\D", "", str(v))) >= 7
              and not re.search(r"[-:T]", str(v))}
        if any(phones[p] >= 3 for p in ph):
            why["phone value reused"] += 1
            continue
        names = {c["name"] for c in k["calls"]}
        cap = {t: max(0.03, 3 * row_share[t] / len(rows)) * len(ok) for t in names}
        if any(tools[t] + 1 > cap[t] for t in names):
            why["tool cap"] += 1
            continue
        phones.update(ph)
        tools.update(names)
        out.append(k)
    out.sort(key=lambda k: k["id"])
    return out, why


TAIL = re.compile(r"( nhé| nha| nghe| nghen| với| đi| hen| nhen)?([.?!]?)$")
TRAIN_TAILS = [("", 0.412), (" nhé", 0.245), (" nha", 0.172), (" nghe", 0.065), (" đi", 0.026), (" nghen", 0.021),
               (" với", 0.013), (" hen", 0.007)]


def speech_form(text: str, rng: random.Random) -> str:
    """Second Opus review: "TP.HCM" said as train speakers say it ("Sài Gòn"); the closing particle redrawn from train's
    closings (T9 ended in "với" 17 %, train 1.3 %). Questions keep theirs."""
    text = re.sub(r"TP\.\s?HCM|TP HCM", "Sài Gòn", text)
    text = re.sub(r"(?<![\wÀ-ỹ])nhe(?![\wÀ-ỹ])", "nha", text)  # Gemini's "nhe": in no real reading
    m = TAIL.search(text)
    if m and m.group(1) and not text.endswith("?"):
        text = text[:m.start()] + rng.choices([x for x, _ in TRAIN_TAILS], [w for _, w in TRAIN_TAILS])[0] + m.group(2)
    return text


def build(args) -> None:
    d = common.load()
    review_rules.set_marker_rates(d["rows"])
    by_id = {r["id"]: r for r in d["rows"]}
    kept = [json.loads(x) for x in KEPT.read_text(encoding="utf-8").splitlines()]
    for name, rows in common.pools(d):
        ids = {r["id"] for r in rows}
        noise = common.real_noise_from(rows, d["avg5"])  # simulated readings, as T2b/T7
        forms = common.spelling_forms(rows, d["avg5"])
        chosen, why = select([k for k in kept if k["source"] in ids], rows)
        orders = arg_orders(rows)
        out = []
        for k in chosen:
            k = copy.deepcopy(k)
            for c in k["calls"]:
                a = c.get("arguments") or {}
                order = orders.get((c["name"], frozenset(a)))
                if order:
                    c["arguments"] = {key: a[key] for key in order}
            k["request"] = speech_form(k["request"], random.Random(f"form:{k['id']}"))
            r = by_id[k["source"]]
            item = copy.deepcopy(r)
            item["id"] = k["id"]
            item["history"] = []
            item["audio_text"] = common.match_spelling(k["request"], [], k["calls"], forms)  # hòa -> hoà
            item["label"] = {"tool_calls": k["calls"], "tool_results": k["results"]}
            rng = random.Random(f"{args.seed}:{k['id']}")
            text = noise.apply(asr_numbers(item["audio_text"], rng), rng, common.protected(item), common.QUESTION_MARK_SYNTH)
            city = (item["context"].get("user_profile") or {}).get("city", "")
            rates = review_rules.MARKER_RATES.get(city, {})
            if any(m in review_rules.words(item["audio_text"]) and rates.get(m, 0.0) < 0.05 for m in review_rules.REGIONAL):
                why["regional word the city rarely uses"] += 1  # final check: T9 12.8 % vs train 7.1 %
                continue
            if common.unsourced_places(item, {item["audio"]: item["audio_text"]}):  # an address Gemini knew, no data gave
                why["unsourced place"] += 1
                continue
            out.append(common.conv("T9", r, {item["audio"]: text}, item=item, apps=k["apps"]))
        common.write(f"t9_{name}", out)
        print(json.dumps({"pool": name, "kept_by_gemini": sum(k["source"] in ids for k in kept), "left_out": why}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--seed", type=int, default=91)
    p.add_argument("--per", type=int, default=4)
    p.add_argument("--samples", type=int, default=1545)
    r = sub.add_parser("run")
    r.add_argument("--stage", choices=["gen", "judge"], required=True)
    r.add_argument("--agy", default=str(Path.home() / ".local" / "bin" / "agy"))
    r.add_argument("--model", default=None)
    r.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--limit", type=int, default=0)
    sub.add_parser("collect")
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=92)
    args = ap.parse_args()
    {"prepare": prepare, "run": run, "collect": collect, "build": build}[args.cmd](args)


if __name__ == "__main__":
    main()
