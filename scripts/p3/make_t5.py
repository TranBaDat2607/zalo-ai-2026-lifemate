"""T5 of improvment_plan3.md: out of scope, made from real samples by uninstalling the app the request needs.

    python scripts/p3/make_t5.py     # outputs/p3/t5_full.jsonl, t5_fold0.jsonl

Source: a sample whose calls all use one app (contacts aside) and whose history does not use it. That app and the
apps that could stand in for it (GROUPS: sms / chat_ott / email, banking / ewallet / bill_payment, ...) leave the
installed apps, and the answer becomes no call at all, as train does for requests no installed app can serve (111 real
empty answers, e.g. "đặt dùm em chiếc xe ra sân bay" with no ride app). Contacts stays installed: looking a person up
is no use when nothing can call or message them. Requests with "nếu" are skipped. The request keeps its real
averaged reading.
"""
import copy
import json
import re
from collections import Counter

import common

GROUPS = [{"sms", "chat_ott", "email"}, {"banking", "ewallet", "bill_payment"}, {"ride_hailing", "bike_rental"},
          {"reminders", "tasks", "alarm"}, {"smart_home", "home_automation", "appliances"}, {"music", "podcast_radio"},
          {"phone", "video_meeting"}, {"gold_price", "exchange_rate"}, {"news"}, {"food_delivery", "shopping_list"},
          {"notes", "tasks"}]


APP_CAP = 15


def group_of(app: str) -> set:
    return next((g for g in GROUPS if app in g), {app})


def history_apps(row: dict) -> set:
    """Apps the history's calls use, the app an ask in the history was for included (13 Nov review)."""
    out = set()
    for t in row["history"]:
        for c in (t.get("tool_calls") or []) if t["role"] == "assistant" else []:
            out.add(c["name"].split(".")[0])
            if c["name"] == "sys.ask_missing_info":
                out.add(str((c.get("arguments") or {}).get("tool_name", "")).split(".")[0])
    return out


def offered(row: dict) -> bool:
    """The last assistant turn asks the user something without a call: it was offering the action."""
    last = next((t for t in reversed(row["history"]) if t["role"] == "assistant"), None)
    return bool(last) and not last.get("tool_calls") and "?" in (last.get("text") or "")


def build(rows: list, avg5: dict) -> tuple:
    out, apps = [], Counter()
    for r in rows:
        calls = r["label"]["tool_calls"]
        used = {c["name"].split(".")[0] for c in calls} - {"contacts"}
        if len(used) != 1 or "sys" in used:
            continue
        if any(c["name"].startswith("contacts.") and c["name"] != "contacts.search" for c in calls):
            continue  # a contacts write is still doable (13 Nov review: train_0474, 0541, 0699, 0746)
        if offered(r):
            continue
        if re.search(r"(?<![\wÀ-ỹ])nếu(?![\wÀ-ỹ])", r["audio_text"].lower()):
            continue  # a condition another installed app might still act on
        gone = group_of(next(iter(used)))
        if gone & history_apps(r):
            continue
        if apps[next(iter(used))] >= APP_CAP:
            continue  # second review: smart_home was 16 % of T5 (one light/air-con template)
        item = copy.deepcopy(r)
        installed = item["context"]["installed_apps"]
        for app in gone:
            installed.pop(app, None)
        if not installed:
            continue
        item["label"] = {"tool_calls": [], "tool_results": [], "final_state": {}}
        texts = {a: avg5[a] for a in common.prompts.gold_transcripts(r)}
        out.append(common.conv("T5", r, texts, item=item, removed=sorted(gone & set(r["context"]["installed_apps"]))))
        apps[next(iter(used))] += 1
    return out, apps


def main() -> None:
    d = common.load()
    for name, rows in common.pools(d):
        out, apps = build(rows, d["avg5"])
        common.write(f"t5_{name}", out)
        print(json.dumps({"pool": name, "top_apps": apps.most_common(8)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
