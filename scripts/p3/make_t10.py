"""T10: new requests written by Gemini for the 11 apps that only the public catalog has (no train sample uses them),
checked like T9. Research behind it (14 Nov): diversity of new situations, not copies, carries generalisation to unseen
tools (ToolACE, DIVE, arXiv 2601.17829); the fold-0 pilot lost points on apps the model had not seen.

    python scripts/p3/make_t10.py prepare --batches 40 [--per 4]   # phone states from make_synth's code, specs, examples
    env -u CLAUDECODE python scripts/p3/make_t10.py run --stage gen
    env -u CLAUDECODE python scripts/p3/make_t10.py run --stage judge
    python scripts/p3/make_t10.py collect                          # agreement + simulator replay + checks -> kept.jsonl
    python scripts/p3/make_t10.py build                            # outputs/p3/t10_full.jsonl (no fold-0 copy needed:
                                                                   # these apps are in no validation sample)

Sources (improvment_plan3.md rules): tool specs from the public catalog; app data built by scripts/make_synth.py's
state code (no public sample or *_environment/ is read); examples are real non-validation train requests that show the
read-before-write habit. Request text and calls come from Gemini only; this code selects and checks.
A batch: one public-only app (+ sometimes one common app: reminders, calendar, notes, or contacts + sms), its data, a
profile and time, three train examples. Kept when the blind judge gives the same calls, every call runs on the
organisers' simulator (zaic_tools Simulator), ids were read first, no app/tool named, times consistent with "now".
"""
import argparse
import copy
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import common
import make_t9 as t9
import review_rules
from make_t6 import SPAN, asr_numbers  # noqa: F401
from lifemate.drills import DIGIT_SHARE, parse_number, say_number

sys.path.insert(0, str(common.REPO / "scripts"))
import make_synth as ms  # noqa: E402
from zaic_tools.executor import Simulator  # noqa: E402  (make_synth put the public zaic_tools on the path)
from zaic_tools.validate import catalog_tools  # noqa: E402

OUT = common.OUT / "t10"
BATCHES = OUT / "batches"
ANS = {"gen": OUT / "gen", "judge": OUT / "judge"}
KEPT = OUT / "kept.jsonl"
STATES = {"grocery": ms.groc_state, "ecommerce": ms.ecom_state, "recipes": ms.rec_state, "pantry": ms.pan_state,
          "habit_tracker": ms.hab_state, "parking": ms.park_state, "public_transit": ms.tr_state,
          "train_bus_tickets": ms.trip_state, "event_tickets": ms.ev_state, "stocks": ms.st_state,
          "translator": lambda c: {}}
COMPANIONS = ["reminders", "calendar", "notes", "sms", None, None]
# Full-run review (14 Nov): train's 217 asks never ask a count; a user booking for themselves without a number gets one
# seat/ticket; "đã tập yoga rồi" means today. These are not asked (rows asking them are dropped in collect too).
NO_ASK = {("train_bus_tickets.book", "seats"), ("event_tickets.book", "quantity"), ("ecommerce.add_to_cart", "quantity"),
          ("habit_tracker.log", "date"), ("grocery.add_to_cart", "quantity"),
          ("stocks.set_price_alert", "direction")}  # comprehensive review: the quote just read makes the direction plain
ASK_TOOL_CAP = {"ecommerce.request_return": 12}  # comprehensive review: 92 parking asks vs 45 bookings, 25 return asks
ASK_TOOL_CAP_DEFAULT = 25                         # vs 0 executed returns  # final review: grocery count asks 47 / 130
# Final review (14 Nov): returns only for delivered orders, cancels only for undelivered ones (catalog descriptions).
ORDER_STATUS = {"ecommerce.request_return": lambda s: s == "delivered", "ecommerce.cancel_order": lambda s: s != "delivered"}
NON_LATIN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")  # kana, CJK, Hangul: no Vietnamese ASR writes them
SMS_TIME = ("hôm nọ", "hôm qua", "hôm kia", "hôm trước", "bữa trước", "bữa nọ", "tuần trước", "tháng trước", "ngày mai",
            "sáng nay", "chiều nay", "tối nay", "tối qua", "sáng mai", "chiều mai", "tối mai")
TEMPLATE = ("dạo này", "dạo ni", "duy trì")  # final review: 26 / 48 habit-list rows say "dạo này", 23 "duy trì"
TEMPLATE_CAP = 0.25
KIN = {"vợ", "chồng", "mẹ", "bố", "ba", "má", "anh", "chị", "em", "cô", "chú", "bác", "dì", "cậu", "ông", "bà", "con", "sếp"}
ASK_SHARE = 0.14  # train: 14.0 % of requests end in sys.ask_missing_info (217 / 1,545); per request, so x per batch
NONE_SHARE = 0.07  # train: 7.2 % have no call; made in build by removing the app the request needs (as T5)


APP_TEXT = {a["id"]: f"{a.get('name_vi', '')}. {a.get('description', '')}" for a in json.loads(
    (ms.PUBLIC / "public_test_tools_catalog.json").read_text(encoding="utf-8"))["apps"]}
CONTENT_KEYS = {"content", "body", "message", "text"}


def catalogs() -> dict:
    pub = catalog_tools(json.loads((ms.PUBLIC / "public_test_tools_catalog.json").read_text(encoding="utf-8")))
    return {**pub, **common.load()["catalog"]}


def make_state(app: str, c) -> dict:
    out = STATES[app](c)
    return out[-1] if isinstance(out, tuple) else out


def companion_state(app: str, c) -> dict:
    if app == "sms":
        return {"sms": {"messages": []}, "contacts": {"contacts": ms.contact_rows(c)}}
    return {app: {"reminders": {"reminders": []}, "calendar": {"events": []}, "notes": {"notes": []}}[app]}


def examples(rows: list, hold: set, catalog: dict, rng: random.Random) -> list:
    """Real train requests (no history, no ask, outside validation) with a read call before a write call."""
    pool = []
    for r in rows:
        calls = r["label"]["tool_calls"]
        kinds = [catalog.get(c["name"], {}).get("kind") for c in calls]
        if r["id"] in hold or r["history"] or not calls or calls[-1]["name"] == t9.ASK:
            continue
        if "read" in kinds and "write" in kinds[kinds.index("read"):]:
            pool.append(r)
    return pool


# Fix 4 (trial review, 14 Nov): people say "cốc giữ nhiệt Elmich", not "... 500ml" or "Sony WH-CH520". Gemini sees a
# spoken name (trailing pack size / unit / model code cut); the simulator keeps the full name and its search finds it.
UNIT = re.compile(r"(?:\s+(?:túi|chai|hộp|khay|bó|gói|thùng|lốc|ký|lon|vỉ))?\s+[\d.,]+\s*(?:kg|g|ml|l|lít|quả|gói|hộp|mah|"
                  r"inch)\b.*$", re.I)
MODEL = re.compile(r"\s+(?:[A-Za-z]*-[A-Za-z0-9]+|[A-Za-z]*\d{3,}[A-Za-z]*)$")


def spoken_name(name: str) -> str:
    out = MODEL.sub("", UNIT.sub("", name)).strip()
    return out if len(out) >= 3 else name


def shown_state(app: str, state: dict) -> dict:
    if app not in ("grocery", "ecommerce"):
        return state
    s = copy.deepcopy(state)
    for p in s.get("products", []):
        if isinstance(p, dict) and isinstance(p.get("name"), str):
            p["name"] = spoken_name(p["name"])
    return s


# Fix 1, 3, 5, 6 told to the writer (the checks in collect enforce them; these lines only cut the waste).
RULES = """Yêu cầu riêng cho lượt này:
{targets}
- Mỗi yêu cầu một kiểu nói và một việc khác nhau; không lặp cùng một khuôn câu.
- Không chép lại chữ trong mô tả công cụ hay tên ứng dụng (người thật không nói "ghi nhận đã dùng", "phiên đỗ xe").
- Gọi món, sản phẩm, sự kiện bằng tên ngắn như người ta nói.
- Người dùng không tự nói tên mình hay địa chỉ của mình: nói "nhà", "chỗ làm", "công ty"; trong lời gọi thì điền
  địa chỉ đầy đủ trong hồ sơ (như ví dụ thật), không điền chữ "nhà".
- Số tiền, ngày tháng nói bằng chữ như người thật nói (không viết "120000", "29/3").
- Không thêm tham số tuỳ chọn người dùng không nói (vd không thêm repeat khi tạo nhắc nhở).
- Đặt/mua cho chính mình mà không nói số lượng thì số lượng là 1 (như dữ liệu thật), không hỏi lại.
- Nội dung tin nhắn/ghi chú đúng ý người nói, không thêm ý mới, không bỏ ý của họ.
- Địa điểm trong lời gọi là đúng chữ người dùng nói (tên bãi, tên trạm), không thay bằng địa chỉ trong dữ liệu.
- Chỉ một từ cuối câu (nhé/nha/nghe...), không lặp "giùm chị" hai lần.
- Mã cổ phiếu: chỉ dùng mã khi người dùng nói mã đó (vd "mã MLG"), hoặc lấy mã từ kết quả của stocks.get_portfolio
  trước đó; không đoán mã từ tên công ty."""


def page(ctx: dict, states: dict, catalog: dict, exs: list, targets: list, usual: list = (), ask_example=None,
         habits: tuple = ((), ())) -> str:
    prof = ctx["user_profile"]
    lines = [f"Thời điểm hiện tại: {ctx['current_time']} (Asia/Ho_Chi_Minh). Người dùng: {prof['name']}, {prof['city']}; "
             f"nhà: {prof['home_address']}; nơi làm: {prof['work_address']}."]
    avoid, prefer = habits
    if avoid:  # trial 4: "kiểm tra", "xem", "tìm" said by Sài Gòn / Huế / Cần Thơ speakers, who in train say them < 1 %
        lines.append(f"Người {prof['city']} nói theo giọng địa phương: không dùng " + ", ".join(f'"{w}"' for w in avoid)
                     + (("; hay dùng " + ", ".join(f'"{w}"' for w in prefer)) if prefer else "") + ".")
    if usual:  # trial 2: Bác Tư said "bác" (train 0.5 %); real speakers keep their own forms
        lines.append("Người này thường xưng hô bằng: " + ", ".join(f'"{w}"' for w in usual)
                     + " (dùng các từ này, không dùng cách xưng khác).")
    lines += ["", "Công cụ (* = bắt buộc):"]
    for a in states:
        lines += [t9.spec_line(catalog[t]) for t in sorted(catalog) if t.startswith(a + ".")]
    lines += ["", "Dữ liệu trong máy:"]
    lines += [f"{a}: {json.dumps(t9.trim(shown_state(a, s)), ensure_ascii=False)}" for a, s in states.items() if s]
    lines += ["", "Ví dụ thật (ở các ứng dụng khác, chỉ để thấy quy ước gọi):"]
    for e in exs:
        calls = [{"name": c["name"], "arguments": c.get("arguments") or {}} for c in e["label"]["tool_calls"]]
        lines.append(f"- Câu nói: {e['audio_text']}\n  Lời gọi: {json.dumps(calls, ensure_ascii=False)}")
    def line(k, t):
        if t.startswith("ask:"):
            _, tool, param = t.split(":")
            return (f"- Yêu cầu {k + 1}: người dùng muốn làm việc của {tool} nhưng KHÔNG nói {param} (thông tin bắt buộc). "
                    f"Lời gọi: các lời gọi đọc cần trước (nếu có), rồi kết thúc bằng sys.ask_missing_info với "
                    f'{{"tool_name": "{tool}", "params": ["{param}"]}}. Câu nói vẫn tự nhiên, không hỏi lại trong câu.')
        tool, _, other = t.partition("|")
        if other:
            return (f"- Yêu cầu {k + 1} BẮT BUỘC gồm HAI việc trong một câu nói: một việc dùng {tool} và một việc dùng ứng "
                    f"dụng {other} (vd nhắc giờ, ghi chú, lịch hoặc nhắn tin liên quan)")
        return (f"- Yêu cầu {k + 1} phải dùng {tool} (thêm các lời gọi đọc cần trước nó, hoặc một việc ở ứng dụng khác nếu "
                f"tự nhiên)")
    if any(t.startswith("ask:") for t in targets) and ask_example:
        e = ask_example
        calls = [{"name": c["name"], "arguments": c.get("arguments") or {}} for c in e["label"]["tool_calls"]]
        lines.append(f"- Câu nói (thiếu thông tin): {e['audio_text']}\n  Lời gọi: {json.dumps(calls, ensure_ascii=False)}")
    tl = "\n".join(line(k, t) for k, t in enumerate(targets))
    lines += ["", RULES.format(targets=tl)]
    return "\n".join(lines)


def prepare(args) -> None:
    d = common.load()
    catalog = catalogs()
    rng = random.Random(args.seed)
    pool = examples(d["rows"], d["hold"], catalog, rng)
    rates, _ = profile_tables(d["rows"])
    north = phrase_rates(d["rows"], NORTHERN + LOCAL)
    habits = {c: ([w for w in NORTHERN if north[c].get(w, 0) < 0.01 and north["Hà Nội"].get(w, 0) >= 0.02],
                  [w for w in LOCAL if north[c].get(w, 0) >= 0.05]) if c != "Hà Nội" else ((), ()) for c in north}
    asks = [r for r in d["rows"] if r["id"] not in d["hold"] and not r["history"] and r["label"]["tool_calls"]
            and r["label"]["tool_calls"][-1]["name"] == t9.ASK]
    BATCHES.mkdir(parents=True, exist_ok=True)
    if not args.offset:
        for f in BATCHES.glob("b*.json"):
            f.unlink()
    apps = sorted(args.apps.split(",")) if args.apps else sorted(STATES)
    turn = Counter()  # fix 6: each app's tools in rotation, so every tool gets requests
    for n in range(args.offset, args.offset + args.batches):
        c = ms.Ctx(random.Random(f"{args.seed}:{n}"))
        app = apps[n % len(apps)]
        states = {app: make_state(app, c)}
        small = sum(t.startswith(app + ".") for t in catalog) <= 2
        comp = c.rng.choice([x for x in COMPANIONS if x] if small else COMPANIONS)  # trial 2: 1-2 tool apps repeat
        if comp:
            states.update(companion_state(comp, c))
        extra = [a for a in ms.DISTRACTORS if a not in states and any(t.startswith(a + ".") for t in catalog)]
        installed = list(states) + c.rng.sample(extra, c.rng.randint(2, 6))
        ctx = {"current_time": c.iso(c.now), "timezone": "Asia/Ho_Chi_Minh", "location": {"lat": c.lat, "lng": c.lng},
               "user_profile": {"name": c.name, "home_address": c.home, "work_address": c.work, "city": c.city},
               "installed_apps": {a: {"tools": [t for t in catalog if t.startswith(a + ".")], "state": states.get(a, {})}
                                  for a in installed}}
        exs = rng.sample(pool, 3)
        own = sorted(t for t in catalog if t.startswith(app + "."))
        askable = [(t, k) for t in own if catalog[t].get("kind") == "write"
                   for k in catalog[t].get("parameters", {}).get("required", [])
                   if not catalog[t]["parameters"]["properties"].get(k, {}).get("x-ref") and (t, k) not in NO_ASK]
        targets = []
        for k in range(args.per):
            tgt = own[turn[app] % len(own)]
            two = comp and ((small and k % 2 == 1) or k == args.per - 1)  # trial 3: 40 / 113 one-call requests
            targets.append(f"{tgt}|{comp}" if two else tgt)
            turn[app] += 1
        if askable and args.ask_per:  # extra ask batches: the first ask_per requests are asks
            for j in range(min(args.ask_per, args.per)):
                t, k = c.rng.choice(askable)
                targets[j] = f"ask:{t}:{k}"
        elif askable and c.rng.random() < ASK_SHARE * args.per:  # When2Call: requests that lack one required value
            t, k = c.rng.choice(askable)
            targets[0] = f"ask:{t}:{k}"
        if app == "ecommerce" and any("ecommerce.request_return" in t for t in targets):
            orders = states["ecommerce"]["orders"]  # final review: returns asked on shipping/placed orders
            for o in (orders if any(t.startswith("ask:ecommerce.request_return") for t in targets) else orders[:1]):
                o["status"] = "delivered"
        (BATCHES / f"b{n:04d}.json").write_text(json.dumps(
            {"id": f"t10b{n:04d}", "file": f"b{n:04d}.json", "app": app, "apps": list(states), "context": ctx, "examples": [e["id"] for e in exs],
             "targets": targets,
             "page": page(ctx, states, catalog, exs, targets,
                                                    [w for w, v in sorted(rates.get(c.name, {}).items(), key=lambda x: -x[1])
                                                     if v >= 0.10], rng.choice(asks),
                                                    habits.get(c.city, ((), ()))), "per": args.per}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"batches": args.batches, "requests": args.batches * args.per, "apps": apps}))


def prompt_for(stage: str, batch: dict) -> str:
    if stage == "gen":
        return t9.GEN.format(n=batch["per"], page=batch["page"])
    gen = json.loads((ANS["gen"] / batch["file"]).read_text(encoding="utf-8"))["reply"]
    reqs = "\n".join(f"- id: {batch['id']}_{k}\n  câu: {it.get('request', '')}" for k, it in enumerate(gen))
    return t9.JUDGE.format(page=batch["page"], requests=reqs)


def run(args) -> None:
    stage = args.stage
    todo = [b for b in sorted(BATCHES.glob("b*.json")) if not (ANS[stage] / b.name).exists()
            and (stage == "gen" or (ANS["gen"] / b.name).exists())]
    t9.run_batches(todo, ANS[stage], lambda batch: prompt_for(stage, batch), t9.SCHEMA[stage], args, stage)


def grams3(text: str) -> set:
    w = review_rules.words(text)
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def train_phrases(rows: list) -> set:
    """3-word runs said in >= 2 real train requests ("vào ghi chú", "đánh dấu đã"): a description's wording that real
    users also say is not a tell (train: 5.4 % of requests share a 3-word run with their apps' descriptions)."""
    n = Counter(g for r in rows for g in grams3(r["audio_text"]))
    return {g for g, k in n.items() if k >= 2}


HOME_WORDS = re.compile(r"^(?:về |tới |đến )?(nhà|nhà tôi|nhà mình|nhà em|nhà anh|nhà chị|chỗ làm|công ty|cơ quan|văn phòng|"
                        r"chỗ tôi làm|nơi làm)$", re.I)
PLACE_KEYS = set(common.PLACE_KEYS) | {"near", "from", "to"}


def map_home(calls: list, prof: dict) -> list:
    """Trial 2 review: Gemini passed "nhà" / "chỗ làm" literally; train always passes the profile's full address."""
    out = copy.deepcopy(calls or [])
    for c in out:
        a = c.get("arguments") or {}
        for k, v in list(a.items()):
            m = HOME_WORDS.match(v.strip()) if k in PLACE_KEYS and isinstance(v, str) else None
            if m:
                a[k] = prof["home_address"] if m.group(1).lower().startswith("nhà") else prof["work_address"]
    return out


SELF_PLACE = re.compile(r"(?<![\wÀ-ỹ])(nhà|chỗ làm|công ty|cơ quan|văn phòng|chỗ tôi làm|nơi làm)(?![\wÀ-ỹ])", re.I)


PRONOUNS = ("tôi", "tui", "mình", "tớ", "em", "anh", "chị", "bác", "cô", "chú", "con", "cháu")
# Trial 2 review: description verbs real users never say (each checked to have 0 train requests at run time)
BOOKKEEPING = ("ghi nhận", "kết thúc", "điểm danh", "tra cứu", "tạo yêu cầu", "trạng thái", "phiên đỗ", "chuỗi ngày",
               "đang nắm", "tích thói quen", "báo giá", "cập nhật", "danh mục", "theo dõi thói quen", "tích",
               "tiện thể", "vậy kìa", "danh sách đã lưu", "mục đã lưu",  # trial 4
               "trừ bớt")  # full run: 43 of 72 pantry consume rows, echoing "(trừ vào số còn)"
BOOKKEEPING_MAX = 2  # trial 3: "cập nhật" 2, "danh mục" 1 train requests: banned when said in <= 2 of 1,545


SELF_SLOT = {"giúp", "giùm", "dùm", "hộ", "cho", "của"}


def pronoun_words(text: str) -> set:
    """Pronouns the speaker uses for themselves: right after giúp/giùm/dùm/hộ/cho/của, or opening the request, and not a
    title before a name ("cho anh Minh"). Trial 3 review: "em" addressing the assistant ("em ơi") had been counted."""
    toks = re.findall(r"[^\W_]+", text)
    out = set()
    for i, t in enumerate(toks):
        w = t.lower()
        if w not in PRONOUNS or (i + 1 < len(toks) and toks[i + 1][:1].isupper()):
            continue
        if (i > 0 and toks[i - 1].lower() in SELF_SLOT) or (i == 0 and not (len(toks) > 1 and toks[1].lower() == "ơi")):
            out.add(w)
    return out


def profile_tables(rows: list) -> tuple:
    by, n = {}, Counter()
    for r in rows:
        name = (r["context"].get("user_profile") or {}).get("name", "")
        n[name] += 1
        by.setdefault(name, Counter()).update(pronoun_words(r["audio_text"]))
    rates = {p: {w: c[w] / n[p] for w in PRONOUNS} for p, c in by.items() if n[p] >= 30}
    said = Counter()
    for r in rows:
        low = r["audio_text"].lower()
        said.update(b for b in BOOKKEEPING if re.search(rf"(?<![\wÀ-ỹ]){b}(?![\wÀ-ỹ])", low))
    return rates, [b for b in BOOKKEEPING if said[b] <= BOOKKEEPING_MAX]


def extra_problem(item: dict, batch: dict, catalog: dict, phrases: set, tables: tuple) -> str:
    """Trial review fixes (14 Nov): 1, 3, 5 of trial 1; pronouns and bookkeeping words of trial 2."""
    req, calls, results = item["request"], item["calls"], item.get("results") or []
    low = req.lower()
    rates, banned, conv = tables[:3]
    if NON_LATIN.search(req):
        return "non-latin script"
    state = batch["context"]["installed_apps"]
    for c in calls:
        ok = ORDER_STATUS.get(c["name"])
        oid = (c.get("arguments") or {}).get("order_id")
        if ok and oid:
            st = [o.get("status") for o in (state.get("ecommerce", {}).get("state") or {}).get("orders", []) if o.get("id") == oid]
            if st and not ok(st[0]):
                return "order status does not allow it"
    if calls and calls[-1]["name"] == t9.ASK and (calls[-1].get("arguments") or {}).get("tool_name") == "ecommerce.request_return":
        if any(o.get("status") != "delivered" for o in (state.get("ecommerce", {}).get("state") or {}).get("orders", [])):
            return "order status does not allow it"
    if re.search(r"(cách (nấu|làm)|nấu (sao|răng|thế nào|ra sao)|mấy bước|các bước)", low) and not any(
            c["name"] in ("recipes.get", "recipes.save") for c in calls) and any(c["name"] == "recipes.search" for c in calls):
        return "recipe method without recipes.get"  # full-run review: 27 of 54 method requests stopped at search
    rel = tables[3] if len(tables) > 3 else {}
    pname = batch["context"]["user_profile"]["name"]
    if any(re.search(rf"(?<![\wÀ-ỹ]){w}(?![\wÀ-ỹ])", low) and rel.get(pname, {}).get(w, 0) == 0 for w in ("vợ", "chồng")):
        return "relation the profile never says"  # Em Linh "vợ Hằng": 0 of 219 train rows
    n, keys = conv
    off = [(t, k) for t, k in t9.off_convention(calls, n, keys)
           if not (t == "calendar.list_events" and k in ("date_from", "date_to"))]  # 17 % in train, right on a named day
    if off:
        return f"optional key {off[0][0]}.{off[0][1]}"
    prof0 = batch["context"]["user_profile"]
    for j, c in enumerate(calls):
        before = json.dumps(results[:j], ensure_ascii=False).lower()
        for k, v in (c.get("arguments") or {}).items():
            if not isinstance(v, str) or not v.strip():
                continue
            vw = review_rules.words(v)
            if k in PLACE_KEYS and v not in (prof0["home_address"], prof0["work_address"]) and vw:
                said = sum(w in review_rules.words(req) for w in vw) / len(vw)
                if said < 0.6 and v.lower() not in before:
                    return "place not said"  # trial 3: a lot's street from the app data, the user said its name
            if k in CONTENT_KEYS and len(vw) >= 4 and v.lower() not in before:
                new = [w for w in vw if w not in review_rules.words(req) and w not in before]
                if len(new) / len(vw) > 0.3:
                    return "content not said"  # trial 3: SMS that invents and drops parts of the request
            if k in CONTENT_KEYS:
                vl = v.lower()
                if any(t in vl and t not in low and t not in before for t in SMS_TIME):
                    return "content time not said"  # final review: "hôm nọ" added to an SMS
                if re.search(r"(?<![\wÀ-ỹ])đang(?![\wÀ-ỹ]).*(?<![\wÀ-ỹ])rồi(?![\wÀ-ỹ])", vl) and not re.search(
                        r"(?<![\wÀ-ỹ])đang(?![\wÀ-ỹ]).*(?<![\wÀ-ỹ])rồi(?![\wÀ-ỹ])", low):
                    return "content time not said"
    for c, r in zip(calls, results):
        if catalog.get(c["name"], {}).get("kind") == "read" and isinstance(r, dict):
            lists = [v for v in r.values() if isinstance(v, list)]
            if lists and not any(lists):
                return "read finds nothing"  # trial 4: "vé tôi đã đặt" with no booking
    tl = batch.get("targets") or []
    target = tl[int(item.get("_k", 0))] if int(item.get("_k", 0)) < len(tl) else ""
    other = target.partition("|")[2]
    if other and not any(c["name"].startswith(other + ".") for c in calls):
        return "two-app target not met"
    name = batch["context"]["user_profile"]["name"]
    if name in rates and any(rates[name].get(w, 0.0) < 0.03 for w in pronoun_words(req)):
        return "pronoun the profile rarely says"
    if any(re.search(rf"(?<![\wÀ-ỹ]){b}(?![\wÀ-ỹ])", low) for b in banned):
        return "bookkeeping word"
    # 1: stocks.symbol has no x-ref, so T9's read-first check skips it; the symbol must be said or read before
    for k, c in enumerate(calls):
        sym = (c.get("arguments") or {}).get("symbol")
        if isinstance(sym, str) and sym.lower() not in low and sym not in json.dumps(results[:k], ensure_ascii=False):
            return "symbol not said or read"
    # 3: the request echoes a tool description or app name of its apps
    desc = set()
    for a in batch["apps"]:
        desc |= grams3(APP_TEXT.get(a, ""))  # trial 3: app descriptions too ("theo dõi thói quen")
        for t in catalog:
            if t.startswith(a + "."):
                desc |= grams3(catalog[t].get("description", ""))
    if (grams3(req) & desc) - phrases:
        return "echoes a description"
    # 5: the user says their own name or address, or a place argument is their address without "nhà"/"chỗ làm"
    prof = batch["context"]["user_profile"]
    own = [prof["name"]] + [prof[k].split(",")[0] for k in ("home_address", "work_address")]
    if any(o and o.lower() in low for o in own):
        return "says own profile"
    for c in calls:
        for k, v in (c.get("arguments") or {}).items():
            if k in PLACE_KEYS and v in (prof["home_address"], prof["work_address"]) and not SELF_PLACE.search(req):
                return "profile place not said"
    return ""


class Env:
    """zaic_tools Simulator on the batch's state, with the ToolEnv-like .call make_t9.check expects."""

    def __init__(self, ctx: dict, catalog: dict):
        self.sim = Simulator({"id": "t10", "context": copy.deepcopy(ctx)}, {"tools": list(catalog.values())})

    def call(self, name: str, args: dict):
        return self.sim.call(name, args)


def check_any(item: dict, judge: list, row: dict, catalog: dict, make_env, twords: set, target: str) -> str:
    """make_t9.check, plus ask targets: writer and judge both end in the asked tool and parameter; the calls before are
    checked and replayed as usual, the ask gets train's result."""
    calls = item.get("calls") or []
    if not target.startswith("ask:"):
        return t9.check(item, judge, row, catalog, make_env, twords)
    _, tool, param = target.split(":")
    if (tool, param) in NO_ASK:
        return "count/date ask train never makes"
    if param == "quantity" and re.search(r"(bán|mua) (hết|sạch)|toàn bộ|tất cả", str(item.get("request", "")).lower()):
        return "count ask with a count said"  # comprehensive review: "bán hết" after a portfolio read
    if param == "quantity" and re.search(r"(?<![\wÀ-ỹ])(cái|chiếc|một)(?![\wÀ-ỹ])", str(item.get("request", "")).lower()):
        return "count ask with a count said"
    want = {"tool_name": tool, "params": [param]}
    for cs in (calls, judge):
        if not cs or cs[-1].get("name") != t9.ASK or (cs[-1].get("arguments") or {}) != want:
            return "ask not as targeted"
    if any(c.get("name") == t9.ASK for c in calls[:-1] + judge[:-1]):
        return "ask before the last call"
    head = {**item, "calls": calls[:-1]}
    if head["calls"]:
        why = t9.check(head, judge[:-1], row, catalog, make_env, twords)
        if why:
            return why
        results = head["results"]
    else:
        req = str(item.get("request", "")).lower()
        if any(re.search(rf"(?<![\wÀ-ỹ]){re.escape(w)}(?![\wÀ-ỹ])", req) for w in twords):
            return "names an app/tool"
        if "?" in req:
            return "question mark"
        results = []
    item["results"] = results + [dict(ASK_RESULT)]
    return ""


ASK_RESULT = {"ok": True, "waiting_for_user": True}


def lower_case_table(rows: list) -> set:
    """Words real requests write in lower case mid-sentence and never capitalised there (trial 4: "Rau muống", "Gạo",
    "Vợ Hằng" copied from the app data with their capitals)."""
    low, cap = Counter(), Counter()
    for r in rows:
        toks = re.findall(r"[^\W\d_]+", r["audio_text"])
        for t in toks[1:]:
            (cap if t[:1].isupper() else low)[t.lower()] += 1
    desc = Counter(w for t in catalogs().values() for w in re.findall(r"[^\W\d_]+", t.get("description", "")) if w.islower())
    return {w for w, n in low.items() if n >= 2 and cap[w] == 0} | {w for w in desc if cap[w] == 0}


def strings_of(x) -> list:
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for v in x.values() for s in strings_of(v)]
    if isinstance(x, list):
        return [s for v in x for s in strings_of(v)]
    return []


def cap(w: str) -> bool:
    """A capitalised word; codes ("ST25", "VNM") are not."""
    return w[:1].isupper() and not any(ch.isdigit() for ch in w) and not (len(w) > 1 and w.isupper())


def name_tables(ctx: dict) -> tuple:
    """Final review (14 Nov): the trial-4 word-list pass broke names ("cầu Giấy", "Kim mã") and missed item capitals
    ("Lẩu", "Vở"). Protected: runs of 2+ capitalised words in the phone's data, profile and city ("Cầu Giấy", "Phú Quý",
    "Nam Ngư"), a leading kinship word left out ("Vợ Hằng" -> "Hằng"). Items: the first word of a data string that is
    capitalised only as the string's start ("Lẩu thái", "Gạo"), lowered mid-sentence unless inside a protected run."""
    prof = ctx.get("user_profile") or {}
    texts = strings_of(ctx.get("installed_apps")) + [str(prof.get(k, "")) for k in ("city", "home_address", "work_address")]
    runs, firsts = set(), set()
    for t in texts:
        for part in re.split(r"[,.;:()\-–/]", t):
            ws = re.findall(r"[^\W_]+", part)
            run = []
            for w in ws + [""]:
                if cap(w):
                    run.append(w)
                    continue
                while run and run[0].lower() in KIN:
                    run = run[1:]
                if len(run) >= 2:
                    runs.add(" ".join(run))
                run = []
            if ws and cap(ws[0]) and (len(ws) == 1 or not cap(ws[1])):
                firsts.add(ws[0])
    for t in texts:
        ws = re.findall(r"[^\W_]+", t)
        if ws and ws[0].lower() in KIN and len(ws) > 1 and ws[1][:1].isupper():
            firsts.add(ws[0])
    return runs, firsts


def lower_names(text: str, ok: set, ctx: dict = None) -> str:
    runs, firsts = name_tables(ctx or {})
    for r in sorted(runs, key=len, reverse=True):  # restore a protected name said with other capitals
        text = re.sub(rf"(?<![\wÀ-ỹ]){re.escape(r)}(?![\wÀ-ỹ])", lambda m: r if m.group(0).lower() == r.lower() else m.group(0),
                      text, flags=re.IGNORECASE)
    guard = [(m.start(), m.end()) for r in runs for m in re.finditer(rf"(?<![\wÀ-ỹ]){re.escape(r)}(?![\wÀ-ỹ])", text)]
    toks = list(re.finditer(r"[^\W\d_]+", text))
    out, last = [], 0
    for i, m in enumerate(toks):
        w = m.group(0)
        start_of_sentence = i == 0 or re.search(r"[.!?]\s*$", text[:m.start()])
        inside = any(a <= m.start() < b for a, b in guard)
        near = [toks[j].group(0) for j in (i - 1, i + 1) if 0 <= j < len(toks) and (j > i or not start_of_sentence)]
        if w.lower() not in KIN and any(cap(x) for x in near if x != toks[0].group(0) or i - 1 != 0):
            inside = True  # part of a capitalised run in the request itself ("Cầu Giấy", "Kim Mã")
        if not start_of_sentence and not inside and w[:1].isupper() and not w.isupper() and (w in firsts or w.lower() in ok):
            out.append(text[last:m.start()] + w.lower())
            last = m.end()
    return "".join(out) + text[last:]


def bare_route(calls: list) -> list:
    """Comprehensive review (14 Nov): next_departures route "Tuyến 8" in 44 of 48 calls; the catalog example is "32" and
    the replayed synth rows and T6 pass the bare number (the simulator does not filter on it)."""
    out = copy.deepcopy(calls or [])
    for c in out:
        a = c.get("arguments") or {}
        if isinstance(a.get("route"), str):
            a["route"] = re.sub(r"^(?:tuyến|Tuyến)\s*(?:số\s*)?", "", a["route"]).strip() or a["route"]
    return out


def said_case(calls: list, request: str) -> list:
    """Trial 4: a search query takes the stored item's capitals ("Rau muống") where the user said "rau muống"; the
    scorer compares query exactly. Non-contact queries take the request's own span (the simulator search ignores
    case, so results do not change); contacts keep the stored name, as train does."""
    out = copy.deepcopy(calls or [])
    low = request.lower()
    for c in out:
        a = c.get("arguments") or {}
        q = a.get("query")
        if (str(c.get("name", "")).endswith(".search") and not str(c.get("name", "")).startswith("contacts.")
                and isinstance(q, str) and q and q not in request and q.lower() in low):
            i = low.index(q.lower())
            a["query"] = request[i:i + len(q)]
    return out


def collect(args) -> None:
    d = common.load()
    catalog = catalogs()
    twords = t9.tool_words(catalog)
    phrases = train_phrases(d["rows"])
    rel = {}
    for r in d["rows"]:
        nm = (r["context"].get("user_profile") or {}).get("name", "")
        low_r = r["audio_text"].lower()
        rel.setdefault(nm, Counter()).update(w for w in ("vợ", "chồng") if re.search(rf"(?<![\wÀ-ỹ]){w}(?![\wÀ-ỹ])", low_r))
    tables = (*profile_tables(d["rows"]), t9.conventions(d["rows"]), rel)
    lower_ok = lower_case_table(d["rows"])
    train_texts = {t9.norm(r["audio_text"]) for r in d["rows"]}
    reasons, kept, seen, shapes = Counter(), [], set(), Counter()
    for g in sorted(ANS["gen"].glob("b*.json")):
        j = ANS["judge"] / g.name
        if not j.exists():
            continue
        batch = json.loads((BATCHES / g.name).read_text(encoding="utf-8"))
        gen = json.loads(g.read_text(encoding="utf-8"))["reply"]
        judge = {str(it.get("id")): it.get("calls") or [] for it in json.loads(j.read_text(encoding="utf-8"))["reply"]}
        row = {"id": batch["id"], "context": batch["context"]}
        for k, item in enumerate(gen):
            if not isinstance(item, dict):
                reasons["bad format"] += 1
                continue
            if k >= len(batch.get("targets") or [None] * batch["per"]):
                reasons["more items than asked"] += 1
                continue
            prof = batch["context"]["user_profile"]
            item["request"] = lower_names(str(item.get("request", "")), lower_ok, batch["context"])
            item["calls"] = bare_route(said_case(map_home(item.get("calls"), prof), item["request"]))
            jcalls = bare_route(said_case(map_home(judge.get(f"{batch['id']}_{k}", []), prof), item["request"]))
            target = batch["targets"][k] if k < len(batch.get("targets", [])) else ""
            why = check_any(item, jcalls, row, catalog, lambda _sid: Env(batch["context"], catalog), twords, target)
            if not why and not any(c["name"].startswith(batch["app"] + ".") for c in item["calls"]):
                why = "no public-only app call"
            if not why:
                item["_k"] = k
                why = extra_problem(item, batch, catalog, phrases, tables)
            shape = (g.name, tuple(c["name"] for c in item.get("calls") or []))
            if not why and shapes[shape] >= 2:  # fix 6: at most 2 rows per batch with one tool sequence
                why = "same shape in batch"
            key = t9.norm(item.get("request", ""))
            if not why and (key in train_texts or key in seen):
                why = "duplicate"
            reasons[" ".join(why.split(" ")[:2]) if why else "kept"] += 1
            if why:
                continue
            seen.add(key)
            shapes[shape] += 1
            kept.append({"id": f"{batch['id']}_{k}", "batch": g.name, "app": batch["app"], "apps": batch["apps"],
                         "request": re.sub(r"\s+", " ", item["request"]).strip(),
                         "calls": [{"name": c["name"], "arguments": c.get("arguments") or {}} for c in item["calls"]],
                         "results": item["results"]})
    with open(KEPT, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"kept": len(kept), "reasons": dict(reasons.most_common()),
              "per_app": dict(Counter(r["app"] for r in kept).most_common()),
              "tools": dict(Counter(c["name"] for r in kept for c in r["calls"]).most_common())}
    (OUT / "collect_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


NORTHERN = ("kiểm tra", "xem", "xem lại", "vào", "tìm", "tra", "tiện thể")
LOCAL = ("coi", "vô", "kiếm", "giùm", "nghe", "nha", "hông")


def phrase_rates(rows: list, words: tuple) -> dict:
    by, size = {}, Counter()
    for r in rows:
        c = (r["context"].get("user_profile") or {}).get("city", "")
        size[c] += 1
        low = r["audio_text"].lower()
        by.setdefault(c, Counter()).update(w for w in words if re.search(rf"(?<![\wÀ-ỹ]){w}(?![\wÀ-ỹ])", low))
    return {c: {w: by[c][w] / size[c] for w in words} for c in size}


EXTRA_REGIONAL = ("liền", "lẹ", "tớ", "nhé", "nha", "ký")  # trial 3: "tớ" Đà Nẵng 0 %, "nhé" Huế 0 %, "ba ký" Hà Nội  # trial 2: "liền" in Hà Nội (0 % of its train requests); not in review_rules.REGIONAL


def city_tables(rows: list) -> tuple:
    """Per city from real train requests: closing particle shares and the share of requests with a helper word."""
    tails, helped, n = {}, Counter(), Counter()
    for r in rows:
        city = (r["context"].get("user_profile") or {}).get("city", "")
        m = t9.TAIL.search(r["audio_text"])
        tails.setdefault(city, Counter())[m.group(1) or ""] += 1
        n[city] += 1
        helped[city] += any(h in review_rules.words(r["audio_text"]) for h in review_rules.HELPERS)
    return tails, {c: helped[c] / n[c] for c in n}


def profile_helpers(rows: list) -> dict:
    """Trial 4: Em Linh says "giúp" ~90 % in train, but the city mix gave her "giùm"; each (profile, city) keeps its
    own helper mix when it has >= 10 helper uses in train, else the city's."""
    mix = {}
    for r in rows:
        p = r["context"].get("user_profile") or {}
        for h in review_rules.HELPERS:
            n = len(re.findall(rf"(?<![\wÀ-ỹ]){h}(?![\wÀ-ỹ])", r["audio_text"].lower()))
            if n:
                mix.setdefault((p.get("name", ""), p.get("city", "")), Counter())[h] += n
    out = {k: dict(v) for k, v in mix.items() if sum(v.values()) >= 10}
    by_name = {}
    for (name, _city), v in mix.items():
        by_name.setdefault(name, Counter()).update(v)
    out.update({(name, "*"): dict(v) for name, v in by_name.items() if sum(v.values()) >= 10})
    return out


def city_speech(text: str, city: str, tails: dict, help_rate: float, have_rate: float, rng: random.Random,
                mix: dict = None) -> str:
    """Fix 2 (trial review): Gemini's helper and closing words follow its own habit (giùm in 53 % of Hà Nội rows, train
    3 %; "nha" in Huế, train 1 %). Same-meaning substitutions only: a helper with its pronoun is dropped so the share of
    rows with one matches the city's, the rest take the city's helper mix; the closing particle is redrawn from the
    city's real closings."""
    text = re.sub(r"TP\.\s?HCM|TP HCM", "Sài Gòn", text)
    text = re.sub(r"(?<![\wÀ-ỹ])nhe(?![\wÀ-ỹ])", "nha", text)
    tail_words = review_rules.words(text)[-3:]
    if (any(h in review_rules.words(text) for h in review_rules.HELPERS) and not any(h in tail_words for h in review_rules.HELPERS)
            and rng.random() < max(0.0, 1 - help_rate / have_rate)):
        text = review_rules.keep_helper("", text, city, rng)  # source without one: drops "giúp tôi", maps a bare one
    mix = mix or review_rules.CITY_HELPERS.get(city, review_rules.CITY_HELPERS["Hà Nội"])
    pick = rng.choices(list(mix), list(mix.values()))[0]
    text = re.sub(r"(?<![\wÀ-ỹ])(giúp|giùm|dùm|hộ)(?![\wÀ-ỹ])",
                  lambda m: pick.capitalize() if m.group(0)[0].isupper() else pick, text, flags=re.I)
    m = t9.TAIL.search(text)
    if m and m.group(1) and m.group(1) != " với" and not text.endswith("?"):
        t = tails.get(city) or tails["Hà Nội"]
        text = text[:m.start()] + rng.choices(list(t), list(t.values()))[0] + m.group(2)
    return review_rules.fix_format(text)


def measured_shares(d: dict) -> dict:
    """Digit share before each unit in train's averaged readings, as the gate measures it (G4). The full run failed G4
    by 0.003 with drills.DIGIT_SHARE (ngàn 0.39 there, 0 of 11 in the readings; phút 0.58, readings 0.79)."""
    import data_gate
    ref = data_gate.unit_counts([d["avg5"][r["audio"]] for r in d["rows"]])
    out = dict(DIGIT_SHARE)
    out.update({u: dg / (dg + wd) for u, (dg, wd) in ref.items() if dg + wd >= 10})
    return out


def heard_numbers(text: str, rng: random.Random, shares: dict) -> str:
    """make_t6.asr_numbers with the measured shares."""
    def one(m):
        num, unit = m.group(1), m.group(2)
        try:
            n = parse_number(num.lower())
        except Exception:  # noqa: BLE001
            return m.group(0)
        if n is None or n >= 1000 or (num.isdigit() and str(n) != num):
            return m.group(0)
        said = str(n) if rng.random() < shares.get(unit.lower(), 0.0) else (
            num if not num.isdigit() else say_number(n, rng, unit_follows=True))
        return f"{said} {unit}"
    return SPAN.sub(one, text)


def build(args) -> None:
    d = common.load()
    review_rules.set_marker_rates(d["rows"])
    noise = common.real_noise_from(d["rows"], d["avg5"])
    forms = common.spelling_forms(d["rows"], d["avg5"])
    kept = [json.loads(x) for x in KEPT.read_text(encoding="utf-8").splitlines() if x.strip()]
    tails, help_rates = city_tables(d["rows"])
    shares = measured_shares(d)
    north = phrase_rates(d["rows"], NORTHERN)
    helper_mix = profile_helpers(d["rows"])
    extra_rates = {}
    for r in d["rows"]:
        c = (r["context"].get("user_profile") or {}).get("city", "")
        extra_rates.setdefault(c, Counter()).update(set(review_rules.words(r["audio_text"])) & set(EXTRA_REGIONAL))
    size = Counter((r["context"].get("user_profile") or {}).get("city", "") for r in d["rows"])
    extra_rates = {c: {w: k[w] / size[c] for w in EXTRA_REGIONAL} for c, k in extra_rates.items()}
    have = sum(any(h in review_rules.words(k["request"]) for h in review_rules.HELPERS) for k in kept) / max(len(kept), 1)
    out, why = [], Counter()
    for k in kept:
        batch = json.loads((BATCHES / k["batch"]).read_text(encoding="utf-8"))
        ctx = copy.deepcopy(batch["context"])
        ctx["installed_apps"] = {a: {"tools": v["tools"]} for a, v in ctx["installed_apps"].items()}
        city = ctx["user_profile"]["city"]
        req = city_speech(k["request"], city, tails, help_rates.get(city, 0.4), max(have, 1e-9),
                          random.Random(f"form:{k['id']}"), helper_mix.get((ctx["user_profile"]["name"], city))
                          or helper_mix.get((ctx["user_profile"]["name"], "*")))
        item = {"id": k["id"], "audio": f"synth/{k['id']}.wav", "audio_text": common.match_spelling(req, [], k["calls"], forms),
                "context": ctx, "history": [], "label": {"tool_calls": k["calls"], "tool_results": k["results"]}}
        if re.search(r"\d+/\d+", item["audio_text"]) or [x for x in re.findall(r"\d{4,}", item["audio_text"])
                                                             if not x.startswith("0")]:
            why["number written as digits"] += 1  # trial 2: "120000", "29/3"; train writes 4+ digits only for phones
            continue
        if review_rules.final_particles(item["audio_text"]) > 1 or re.search(
                rf"((?:giúp|giùm|dùm|hộ) (?:{review_rules.HELPER_OBJ}))\b.*\b\1\b", item["audio_text"].lower()):
            why["two closings or a helper twice"] += 1  # trial 3; train: 1.7 % and 1 of 1,545
            continue
        lowt = item["audio_text"].lower()
        if re.search(r"(?<![\wÀ-ỹ])không đi\W*$", lowt):
            why["ends in 'không đi'"] += 1  # full-run review
            continue
        if re.search(r"(?<![\wÀ-ỹ])với\s+(nhé|nha|nghe|nghen|hen|nhen)\W*$", lowt) or len(re.findall(
                r"(?<![\wÀ-ỹ])với(?![\wÀ-ỹ])", lowt)) > 1:
            why["với with a closing or twice"] += 1  # trial 4
            continue
        if city != "Hà Nội" and any(re.search(rf"(?<![\wÀ-ỹ]){w}(?![\wÀ-ỹ])", lowt)
                                    and north[city].get(w, 0.0) < 0.01 and north["Hà Nội"].get(w, 0.0) >= 0.02
                                    for w in NORTHERN):
            why["northern word in another city"] += 1  # trial 4: "kiểm tra" 0 of 754 non-Hà Nội train requests
            continue
        low_words = set(review_rules.words(item["audio_text"]))
        if any(w in low_words and extra_rates.get(city, {}).get(w, 0.0) < 0.05 for w in EXTRA_REGIONAL):
            why["regional word the city rarely uses"] += 1
            continue
        rates = review_rules.MARKER_RATES.get(ctx["user_profile"]["city"], {})
        if any(m in review_rules.words(item["audio_text"]) and rates.get(m, 0.0) < 0.05 for m in review_rules.REGIONAL):
            why["regional word the city rarely uses"] += 1
            continue
        if common.unsourced_places(item, {item["audio"]: item["audio_text"]}):
            why["unsourced place"] += 1
            continue
        rng = random.Random(f"{args.seed}:{k['id']}")
        text = noise.apply(heard_numbers(item["audio_text"], rng, shares), rng, common.protected(item),
                           common.QUESTION_MARK_SYNTH)
        out.append(common.conv("T10", {"id": k["id"]}, {item["audio"]: text}, item=item, apps=k["apps"]))
    out = drop_near_duplicates(out, why)
    out = balance(out, why, args.seed)
    nones = no_call_rows(out, args.seed)
    common.write("t10_full", out + nones)
    common.write("t10_fold0", out + nones)  # the 11 apps are in no validation sample, fold 0 = full
    asks = sum(r["item"]["label"]["tool_calls"][-1]["name"] == t9.ASK for r in out)
    print(json.dumps({"kept_by_gemini": len(kept), "rows": len(out) + len(nones), "asks": asks, "no_call": len(nones),
                      "left_out": dict(why)}, ensure_ascii=False))


TRANSLATOR_CAP = 50  # full run: 91 translator rows (one tool, not in train), 78 opening "Dịch"


def balance(rows: list, why: Counter, seed: int) -> list:
    """Full-run review: translator over-represented by the per-app rotation; asks capped at train's 14 %."""
    rng = random.Random(f"balance:{seed}")
    rng.shuffle(rows)
    out, tr, dich, asks = [], 0, 0, []
    seq = lambda r: tuple(c["name"] for c in r["item"]["label"]["tool_calls"])  # noqa: E731
    size, used, per_tool = Counter(seq(r) for r in rows), Counter(), Counter()
    for r in rows:
        calls = r["item"]["label"]["tool_calls"]
        low = r["item"]["audio_text"].lower()
        hit = [t for t in TEMPLATE if t in low]
        if hit and size[seq(r)] >= 8:  # final review: one wording per tool sequence at most a quarter of its rows
            if any(used[(seq(r), t)] >= TEMPLATE_CAP * size[seq(r)] for t in hit):
                why["template cap"] += 1
                continue
            used.update((seq(r), t) for t in hit)
        if any(c["name"].startswith("translator.") for c in calls):
            opens = r["item"]["audio_text"].lower().startswith("dịch")
            if tr >= TRANSLATOR_CAP or (opens and dich >= TRANSLATOR_CAP // 2):
                why["translator cap"] += 1
                continue
            tr += 1
            dich += opens
        if calls and calls[-1]["name"] == t9.ASK:
            tool = (calls[-1].get("arguments") or {}).get("tool_name", "")
            if per_tool[tool] >= ASK_TOOL_CAP.get(tool, ASK_TOOL_CAP_DEFAULT):
                why["ask tool cap"] += 1
                continue
            per_tool[tool] += 1
            asks.append(r)
            continue
        out.append(r)
    keep = round(ASK_SHARE / (1 - ASK_SHARE - NONE_SHARE) * len(out))  # asks / all = 14 % once no-call rows are added
    why["ask cap"] += max(0, len(asks) - keep)
    out += asks[:keep]
    out.sort(key=lambda r: r["item"]["id"])
    return out


def drop_near_duplicates(rows: list, why: Counter) -> list:
    """Full run: "nạp một trăm ngàn vô thẻ xe buýt" five times. A request whose word set overlaps an earlier kept
    one by Jaccard >= 0.8 is dropped (the acceptance rule's near-duplicate measure)."""
    kept, sets = [], []
    for r in rows:
        w = set(re.findall(r"[^\W_]+", r["item"]["audio_text"].lower()))
        if any(w and len(w & x) / len(w | x) >= 0.8 for x in sets):
            why["near duplicate"] += 1
            continue
        kept.append(r)
        sets.append(w)
    return kept


GROUP = {a["id"]: a.get("group", "") for a in json.loads(
    (ms.PUBLIC / "public_test_tools_catalog.json").read_text(encoding="utf-8"))["apps"]}


def no_call_rows(rows: list, seed: int) -> list:
    """When2Call's "cannot answer" case, as T5 does it: a request whose calls all use one public-only app, with that
    app removed from the phone, and no other installed app of its group left -> the right answer is no call. About
    NONE_SHARE of the file; each source request used once."""
    rng = random.Random(f"none:{seed}")
    pool = []
    for r in rows:
        calls = r["item"]["label"]["tool_calls"]
        apps = {c["name"].split(".")[0] for c in calls}
        if len(apps) != 1 or calls[-1]["name"] == t9.ASK:
            continue
        app = apps.pop()
        if app not in STATES:
            continue
        left = [a for a in r["item"]["context"]["installed_apps"] if a != app]
        if any(GROUP.get(a) == GROUP.get(app) for a in left):
            continue
        pool.append((r, app))
    rng.shuffle(pool)
    want = round(NONE_SHARE / (1 - NONE_SHARE) * len(rows))
    out = []
    for r, app in pool[:want]:
        item = copy.deepcopy(r["item"])
        item["id"] += "_none"
        item["audio"] = f"synth/{item['id']}.wav"
        item["context"]["installed_apps"] = {a: v for a, v in item["context"]["installed_apps"].items() if a != app}
        item["label"] = {"tool_calls": [], "tool_results": []}
        out.append(common.conv("T10", {"id": item["id"]}, {item["audio"]: list(r["texts"].values())[0]}, item=item,
                               apps=[], removed=app))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--per", type=int, default=4)
    p.add_argument("--batches", type=int, default=40)
    p.add_argument("--offset", type=int, default=0, help="first batch number; > 0 keeps the batches already there")
    p.add_argument("--ask-per", type=int, default=0, help="ask targets per batch (extra ask batches)")
    p.add_argument("--apps", default="", help="comma list: only these apps (extra ask batches on askable apps)")
    r = sub.add_parser("run")
    r.add_argument("--stage", choices=["gen", "judge"], required=True)
    r.add_argument("--agy", default=str(Path.home() / ".local" / "bin" / "agy"))
    r.add_argument("--model", default=None)
    r.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--limit", type=int, default=0)
    sub.add_parser("collect")
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=102)
    args = ap.parse_args()
    {"prepare": prepare, "run": run, "collect": collect, "build": build}[args.cmd](args)


if __name__ == "__main__":
    main()
