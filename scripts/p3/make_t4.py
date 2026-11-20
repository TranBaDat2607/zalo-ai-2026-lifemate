"""T4 of improvment_plan3.md: ask for what is missing, made from real samples by taking the needed value out of the request.

    python scripts/p3/make_t4.py --per-source 3     # outputs/p3/t4_full.jsonl, t4_fold0.jsonl

Only patterns that train itself has as real asks, with train's habits (the ask is the last call; the calls before it,
reads such as contacts.search included, are made as usual):
  amount  the last call is ewallet.transfer and the request says the amount once ("50 nghìn") -> the amount becomes
          "chút tiền" / "ít tiền" -> calls before the transfer, then sys.ask_missing_info(ewallet.transfer, [amount])
          (train: 19 real asks of this kind, e.g. "chuyển ví cho bố em chút tiền nhé")
  time    the last call is reminders.create and every time expression can be taken out of the request -> calls before
          it, then sys.ask_missing_info(reminders.create, [time])   (train: 7, e.g. "nhắc tôi gọi điện cho mẹ với nha")
  contact contacts.search found one contact and the next, last call (sms.send, ewallet.transfer, phone.call) uses it ->
          the search result gets a second contact of the same name with a relation the request does not say, and
          the call becomes sys.ask_missing_info(<tool>, [contact_id])   (train: 66 real asks after a two-contact search,
          e.g. "gọi cho bạn Minh" with Minh đồng nghiệp and Minh khách hàng). Only bare names (no "anh", "sếp"...)
          and requests that do not name the found contact's relation. The request itself is unchanged, so it keeps its
          real averaged reading.
The amount must be the transfer's amount (the phrase is read back as a number); after the time is taken out, no
relative time ("trước ... hai tiếng", "tối ni") may be left.
Requests with "nếu" are skipped (a condition may hang on the removed value). The kept calls keep their real results
(they ran before the removed value mattered); the ask gets train's result {ok: true, waiting_for_user: true}.
The edited request is heard through the hearing noise of T2b; the history keeps its real averaged readings.
"""
import argparse
import copy
import json
import random
import re

import common
from lifemate.drills import parse_number

NUMW = r"(?:không|một|mốt|hai|ba|bốn|tư|năm|lăm|nhăm|sáu|bảy|tám|chín|mười|mươi|trăm|linh|lẻ|chục|rưỡi|nửa)"
NUM = rf"(?:\d[\d.,]*|{NUMW}(?:\s+{NUMW})*)"
MONEY = re.compile(rf"(?<![\wÀ-ỹ]){NUM}\s*(?:nghìn|ngàn|triệu|tỷ|tỉ|k|đồng|đ)(?:\s+{NUM}(?:\s*(?:nghìn|ngàn|đồng))?)?"
                   rf"(?:\s+(?:đồng|rưỡi))?(?![\wÀ-ỹ])", re.I)
ASK_RESULT = {"ok": True, "waiting_for_user": True}

DAYPART = r"(?:sáng|trưa|chiều|tối|đêm|khuya)"
DAY = (r"(?:hôm nay|ngày mai|ngày mốt|ngày kia|mai|mốt|nay|hôm sau|tuần sau|tuần tới|tuần này|cuối tuần|tháng sau|"
       r"(?:thứ (?:hai|ba|tư|năm|sáu|bảy)|chủ nhật)(?: tuần (?:này|sau|tới))?|ngày \d{1,2}(?:[/-]\d{1,2}| tháng \d{1,2})?)")
CLOCK = rf"(?:\d{{1,2}}(?:[:h]\d{{2}})?|{NUMW}(?:\s+{NUMW})?)\s*(?:giờ|g|h)(?:\s*(?:\d{{1,2}}|{NUMW}(?:\s+{NUMW})?)(?:\s*phút)?|\s*rưỡi|\s*kém\s+(?:\d{{1,2}}|{NUMW}(?:\s+{NUMW})?)(?:\s*phút)?)?"
LEAD = r"(?:(?:vào |đúng |khoảng |tầm |lúc |vào lúc )+)?"
TIME = re.compile(rf"(?<![\wÀ-ỹ]){LEAD}(?:{CLOCK}(?:\s+{DAYPART})?(?:\s+{DAY})?|{DAYPART}(?:\s+{DAY})?|{DAY}(?:\s+{DAYPART})?)"
                  rf"(?![\wÀ-ỹ])", re.I)
LEFT = re.compile(rf"\d|(?<![\wÀ-ỹ])(?:giờ|phút|tiếng|sáng|trưa|chiều|tối|đêm|khuya|hôm|tuần|tháng|thứ|chủ nhật|ngày|"
                  rf"mai|mốt|lát|nữa|trước|sau|khi|lúc|sớm|muộn|trễ|ni|nớ|nay|này|tới|xong|hết|đến|rưỡi|kém|{NUMW})(?![\wÀ-ỹ])")
TITLES = {"anh", "chị", "cô", "chú", "bác", "sếp", "mẹ", "bố", "ba", "má", "ông", "bà", "em", "bạn", "dì", "cậu", "thầy",
          "mợ", "thím", "con", "cháu", "vợ", "chồng"}
CONTACT_TOOLS = {"sms.send", "ewallet.transfer", "phone.call"}
AMOUNT_WORDS = ["chút tiền", "ít tiền", "một ít tiền"]
# 13 Nov Opus review: the twin's relation comes from the found contact's own class (a colleague for a colleague, a
# customer for a customer; a family member gets a friend), never an impossible one ("thằng Tí" = vợ); requests that
# put a title or role word before the name are skipped (train acts directly when a qualifier settles who is meant)
REL_CLASS = [("friend", r"^bạn|hàng xóm"), ("school", r"chủ nhiệm|cô giáo|giảng viên|giáo sư|huấn luyện"),
             ("service", r"bác sĩ|thợ|phòng khám|tài xế"),
             ("business", r"^khách|đối tác|nhà cung cấp|^mối|thương lái|thu mua|chủ nhà|chủ xe"),
             ("work", r"đồng nghiệp|sếp|trưởng nhóm|kế toán|hành chính|kinh doanh|pháp chế|phòng|nhóm|bộ phận|quản lý|"
                      r"mentor|điều phối|phụ trách")]
ORG = re.compile(r"(?i)\b(foods?|hotel|interiors?|residence|company|co|ltd|shop|store|group|corp|studio|spa|clinic|"
                 r"mart|logistics|farm|trading|service[s]?)\b")
TITLED = r"(?:thầy|ông|bà|cô|chú|bác|anh|chị|thằng|con|bé|út|sếp|em|dì|cậu|mợ|o|mệ|mạ|bố|mẹ|ba|má)"
PER_TOOL = {"phone.call": 3, "sms.send": 1, "ewallet.transfer": 1}  # train's asks: phone 43 / sms 27 / ewallet 2


def rel_class(rel: str) -> str:
    return next((name for name, pat in REL_CLASS if re.search(pat, rel.lower())), "family")


def tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;])", r"\1", text)
    text = re.sub(r"(?<![\wÀ-ỹ])(?:lúc|vào lúc|vào|đúng|khoảng)\s*([,.!?;]|$)", r"\1", text)
    text = re.sub(r",\s*([,.!?;])", r"\1", text)
    return re.sub(r"^[,\s]+", "", text).strip()


def ask_item(row: dict, k: int, text: str, tool: str, param: str) -> dict:
    item = copy.deepcopy(row)
    item["audio_text"] = text
    calls, results = row["label"]["tool_calls"][:k], row["label"]["tool_results"][:k]
    item["label"] = {"tool_calls": calls + [{"name": "sys.ask_missing_info", "arguments": {"tool_name": tool, "params": [param]}}],
                     "tool_results": results + [ASK_RESULT], "final_state": {}}
    return item


def amount_variants(row: dict, rng: random.Random) -> list:
    calls = row["label"]["tool_calls"]
    if not calls or calls[-1]["name"] != "ewallet.transfer" or "amount" not in calls[-1]["arguments"]:
        return []
    last = next((t for t in reversed(row["history"]) if t["role"] == "assistant"), None)
    if last and ("bao nhiêu" in (last.get("text") or "").lower() or any(
            c["name"] == "sys.ask_missing_info" for c in last.get("tool_calls") or [])):
        return []  # the user just answered "bao nhiêu?": removing the answer makes a re-ask loop train never has
    found = list(MONEY.finditer(row["audio_text"]))
    if len(found) != 1:
        return []
    m = found[0]
    phrase = re.sub(r"(\d)\s*k\b", r"\1 nghìn", m.group(0).lower()).replace("ngàn", "nghìn")
    phrase = re.sub(r"\s*(?:đồng|đ)$", "", phrase).strip()
    try:
        value = parse_number(phrase)
    except Exception:  # noqa: BLE001
        return []
    if value != calls[-1]["arguments"]["amount"]:
        return []  # the phrase is another amount (e.g. a top-up), not the transfer's
    out = []
    for words in rng.sample(AMOUNT_WORDS, len(AMOUNT_WORDS)):
        after = row["audio_text"][m.end():].lstrip()
        say = words.replace(" tiền", "") if after.lower().startswith("tiền") else words
        text = tidy(row["audio_text"][:m.start()] + say + row["audio_text"][m.end():])
        out.append(ask_item(row, len(calls) - 1, text, "ewallet.transfer", "amount"))
    return out


def time_variants(row: dict, rng: random.Random) -> list:
    calls = row["label"]["tool_calls"]
    if not calls or calls[-1]["name"] != "reminders.create" or "time" not in calls[-1]["arguments"]:
        return []
    text = row["audio_text"]
    spans = list(TIME.finditer(text))
    if len(spans) != 1:
        return []  # several time phrases: cutting all may remove another call's time (13 Nov review, train_1703)
    if any(isinstance(v, str) and re.match(r"\d{4}-\d\d-\d\dT", v) for c in calls[:-1] for v in (c.get("arguments") or {}).values()):
        return []
    hist = " ".join(t.get("audio_text") or t.get("text") or "" for t in row["history"]).lower()
    if spans[0].group(0).strip().lower() in hist:
        return []  # the history still says when
    cut = tidy(TIME.sub("", text))
    if LEFT.search(cut.lower()) or len(cut.split()) < 3:
        return []  # a time word is left (or nothing is): the request may still say when
    return [ask_item(row, len(calls) - 1, cut, "reminders.create", "time")]


def contact_variants(row: dict, rng: random.Random, relations: list, per: int) -> list:
    calls, results = row["label"]["tool_calls"], row["label"]["tool_results"]
    if len(calls) < 2 or calls[-1]["name"] not in CONTACT_TOOLS or calls[-2]["name"] != "contacts.search":
        return []
    found = results[-2].get("contacts") or []
    if len(found) != 1 or calls[-1]["arguments"].get("contact_id") != found[0]["id"]:
        return []
    person = found[0]
    if person["name"].split()[0].lower() in TITLES:
        return []
    low = row["audio_text"].lower()
    if person.get("relation", "").lower() in low:
        return []
    first = re.escape(person["name"].split()[0].lower())
    if ORG.search(person["name"]) or re.match(rf"{TITLED}(?![\wÀ-ỹ])", person["name"].lower()) or \
            re.match(rf"{TITLED}(?![\wÀ-ỹ])", str(calls[-2]["arguments"].get("query", "")).lower()) or re.search(rf"(?<![\wÀ-ỹ]){TITLED}\s+{first}", low):
        return []
    cls = rel_class(person.get("relation", ""))
    pool = [r for r in relations if r != person.get("relation") and r.lower() not in low
            and rel_class(r) == ("friend" if cls == "family" else cls)]
    per = min(per, PER_TOOL.get(calls[-1]["name"], 1), len(pool))
    used = {c.get("id") for r in results for c in (r.get("contacts") or [])}
    out = []
    for rel in rng.sample(pool, per):
        width = len(person["id"].split("_")[-1])  # second review: twins were always c_NN, real ids c_NNN(N)
        lo = 10 ** (width - 1)
        twin = {"id": next(f"c_{n}" for n in iter(lambda: rng.randrange(lo, 10 * lo), None) if f"c_{n}" not in used),
                "name": person["name"], "relation": rel, "phone": "09" + "".join(rng.choice("0123456789") for _ in range(9))}
        both = [person, twin] if rng.random() < 0.5 else [twin, person]
        item = ask_item(row, len(calls) - 1, row["audio_text"], calls[-1]["name"], "contact_id")
        item["label"]["tool_results"][-2] = {**results[-2], "contacts": both}
        out.append(item)
    return out


def build(rows: list, noise, avg5: dict, per: int, seed: int) -> tuple:
    out, sources, seen = [], {"amount": 0, "time": 0, "contact": 0}, set()
    relations = sorted({c["relation"] for r in rows for res in r["label"]["tool_results"] for c in (res.get("contacts") or [])
                        if isinstance(res, dict) and c.get("relation")})
    for r in rows:
        if re.search(r"(?<![\wÀ-ỹ])nếu(?![\wÀ-ỹ])", r["audio_text"].lower()):
            continue
        rng = random.Random(f"{seed}:{r['id']}")
        for kind, made in (("amount", amount_variants(r, rng)), ("time", time_variants(r, rng)),
                           ("contact", contact_variants(r, rng, relations, per))):
            if not made:
                continue
            sources[kind] += 1
            keep = common.protected(r)
            for v in range(min(per, len(made)) if kind != "amount" else per):
                item = made[v % len(made)]
                texts = {a: avg5[a] for a in common.prompts.gold_transcripts(item) if a != item["audio"]}
                if kind == "contact" and v == 0:
                    texts[item["audio"]] = avg5[item["audio"]]  # the request is as it was said: its real reading
                else:
                    texts[item["audio"]] = common.noisy(item["audio_text"], noise, random.Random(f"{seed}:{r['id']}:{v}"), keep)
                key = (texts[item["audio"]], json.dumps(item["label"]["tool_calls"], ensure_ascii=False),
                       json.dumps(item["label"]["tool_results"], ensure_ascii=False))
                if key in seen:
                    continue
                seen.add(key)
                out.append(common.conv("T4", r, texts, item=item, pattern=kind))
    return out, sources


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=3)
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args()
    d = common.load()
    for name, rows in common.pools(d):
        noise = common.noise_from(rows, d["avg5"])
        out, sources = build(rows, noise, d["avg5"], args.per_source, args.seed)
        common.write(f"t4_{name}", out)
        print(json.dumps({"pool": name, "sources": sources}))


if __name__ == "__main__":
    main()
