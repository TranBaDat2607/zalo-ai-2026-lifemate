"""T3 of improvment_plan3.md: tool-choice drills (prompts.encode_tool_pack), first tool of a request in a real context.

    python scripts/p3/make_t3.py --packs 1200 --per-pack 5     # outputs/p3/t3_full.jsonl, t3_fold0.jsonl

A pack takes one sample's context (time, profile, installed apps: the tool list as a real prompt shows it) and asks
for the first tool of several requests. Requests come from train samples without history (their first call needs no
earlier turn, in their real readings) and from the bundled synthetic conversations of the public-only apps (synth_samples.jsonl, written by
scripts/make_synth.py from the public catalog only). In scope: every app the request's calls use is installed; the
answer is the gold first call. Out of scope (~12 %): a one-app request (T5's rule: no "nếu") whose app and stand-ins
are all missing; the answer is "none". Requests whose first call is an ask are left out.

Each request is shown in one of its readings (train: avg5, medium, medium10, b, s2027, s2028; synthetic: T2b's
hearing noise), and every (reading, answer) pair is used once, so no drill repeats. Tools are drawn so frequent tools
do not crowd out the rest (gate G7); a tool with fewer than six readings gets more hearings of its gold text through
the same noise, and while a tool has fewer than MIN_DRILLS drills, half the packs take a real context in which its app
is installed. Synthetic requests get the speech model's number writing (make_t6.asr_numbers) before the noise. The fold-0 file leaves out the validation samples and removes the six held-out apps from every context.
"""
import argparse
import copy
import json
import random
import re
from collections import Counter, defaultdict

import common
from make_t5 import group_of
from make_t6 import asr_numbers

MIN_DRILLS = 5
NONE_CAP = 12  # "none" drills per app (second review: smart_home was 22 % of them)
READS = ["avg5", "medium", "medium10", "b", "s2027", "s2028"]
SYNTH = common.REPO / "src" / "lifemate" / "synth_samples.jsonl"


LOOKUP = [{"reminders.list", "tasks.list", "calendar.list_events", "video_meeting.list_meetings", "alarm.list"},
          {"sms.read_recent", "chat_ott.list_chats", "email.list"}]
CONTACT_SAID = re.compile(r"danh bạ|liên lạc|liên hệ|số điện thoại|số của")


def apps_of(row: dict) -> set:
    """Apps the calls use, the app an ask is for included (13 Nov review: a phone.call ask needs a phone app)."""
    out = set()
    for c in row["label"]["tool_calls"]:
        name = c["name"] if c["name"] != "sys.ask_missing_info" else str((c.get("arguments") or {}).get("tool_name", ""))
        out.add(name.split(".")[0])
    return out - {"sys", ""}


def none_ok(row: dict) -> bool:
    """A request that can be a "none" drill: one app besides contacts, no condition, and no contacts part that
    stays doable (a contacts write, or the request asks for the contact itself)."""
    apps = apps_of(row)
    if len(apps - {"contacts"}) != 1 or re.search(r"(?<![\wÀ-ỹ])nếu(?![\wÀ-ỹ])", row["audio_text"].lower()):
        return False
    if any(c["name"].startswith("contacts.") and c["name"] != "contacts.search" for c in row["label"]["tool_calls"]):
        return False
    return not CONTACT_SAID.search(row["audio_text"].lower())


def sibling_added(tool: str, src_apps: set, installed: set) -> bool:
    """A lookup drill whose context installs a sibling source the request's own context lacked: where to look
    becomes a guess (13 Nov review, 26 drills)."""
    for g in LOOKUP:
        if tool in g:
            sib = {t.split(".")[0] for t in g} - {tool.split(".")[0]}
            return bool((sib & installed) - src_apps)
    return False


def requests(rows: list, synth: list, readings: dict, noise, seed: int) -> list:
    """[(text, tool, apps, one_app_ok)] of every usable request reading (and T7 variants, heard through the noise)."""
    out = []
    for r in rows + synth:
        calls = r["label"]["tool_calls"]
        if r["history"] or not calls or calls[0]["name"].startswith("sys."):
            continue
        apps = apps_of(r)
        single = none_ok(r)
        src = set(r["context"]["installed_apps"])
        if r["id"].startswith("synth"):
            keep = common.protected(r)
            texts = {common.noisy(asr_numbers(r["audio_text"], random.Random(f"{seed}:{r['id']}:n{k}")), noise,
                                  random.Random(f"{seed}:{r['id']}:{k}"), keep) for k in range(3)}
        else:
            texts = {readings[n][r["audio"]] for n in READS}  # the real readings only: paraphrases carry T7's
            # politeness drift (gate G4), and tool choice needs the real hearings, not more wordings, so T7 stays out
        out += [(t, calls[0]["name"], apps, single, src) for t in sorted(texts)]
    # a tool with very few requests (alarm.list, video_meeting.create: one train sample each) gets more hearings of
    # its gold text, so it still reaches MIN_DRILLS (gate G7); the answer and the protected words stay the same
    count = Counter(q[1] for q in out)
    for r in rows:
        calls = r["label"]["tool_calls"]
        if r["history"] or not calls or calls[0]["name"].startswith("sys.") or r["id"].startswith("synth"):
            continue
        tool, keep, have = calls[0]["name"], common.protected(r), {q[0] for q in out if q[1] == calls[0]["name"]}
        apps = apps_of(r)
        single, src = none_ok(r), set(r["context"]["installed_apps"])
        for k in range(40):
            if count[tool] >= MIN_DRILLS + 1:
                break
            t = common.noisy(r["audio_text"], noise, random.Random(f"{seed}:{r['id']}:g{k}"), keep)
            if t not in have:
                have.add(t)
                out.append((t, tool, apps, single, src))
                count[tool] += 1
    return out


def build(rows: list, synth: list, contexts: list, readings: dict, noise, args, held: set) -> tuple:
    rng = random.Random(args.seed)
    pool = requests(rows, synth, readings, noise, args.seed)
    rng.shuffle(pool)
    by_tool = defaultdict(list)
    for q in pool:
        by_tool[q[1]].append(q)
    used_tool = Counter()
    packs, nones, none_app = [], 0, Counter()
    by_apps = defaultdict(list)  # app: contexts that have it installed
    for c in contexts:
        for app in c["installed_apps"]:
            by_apps[app].append(c)
    for _ in range(args.packs):
        short = sorted((t for t, qs in by_tool.items() if qs and used_tool[t] < MIN_DRILLS), key=lambda t: used_tool[t])
        fit = [c for t in short[:1] for q in by_tool[t] for c in by_apps[min(q[2] or {"sys"}, key=lambda a: len(by_apps[a]))]
               if q[2] <= set(c["installed_apps"]) and not q[2] & held] if short and rng.random() < 0.5 else []
        ctx = copy.deepcopy(rng.choice(fit or contexts))  # half the time, a context where a short tool can be drilled
        for app in held:
            ctx["installed_apps"].pop(app, None)
        installed = set(ctx["installed_apps"])
        drills = []
        if rng.random() < 0.28:  # one out-of-scope request in about a quarter of the packs: ~7 % "none", as in train
            for tool in sorted(by_tool, key=lambda t: -len(by_tool[t])):  # from the tools with most requests left
                q = next((q for q in by_tool[tool] if q[3] and none_app[next(iter(q[2] - {"contacts"}))] < NONE_CAP
                          and not (group_of(next(iter(q[2] - {"contacts"}))) & installed)), None)
                if q:
                    none_app[next(iter(q[2] - {"contacts"}))] += 1
                    by_tool[tool].remove(q)
                    drills.append({"text": q[0], "tool": "none"})
                    break
        for tool in sorted(by_tool, key=lambda t: (used_tool[t], rng.random())):
            if len(drills) >= args.per_pack:
                break
            q = next((q for q in by_tool[tool] if q[2] <= installed and not sibling_added(q[1], q[4], installed)), None)
            if q:
                by_tool[tool].remove(q)
                drills.append({"text": q[0], "tool": q[1]})
                used_tool[tool] += 1
        if len(drills) < 3:
            continue
        rng.shuffle(drills)
        nones += sum(x["tool"] == "none" for x in drills)
        packs.append({"kind": "tools", "type": "T3", "pack": {"context": ctx, "drills": drills}})
    total = sum(used_tool.values())
    return packs, {"packs": len(packs), "drills": total + nones, "none": nones, "tools": len(used_tool),
                   "under_5": sorted(t for t, n in used_tool.items() if n < 5),
                   "top_share": round(max(used_tool.values()) / max(total, 1), 3) if used_tool else 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", type=int, default=1200)
    ap.add_argument("--per-pack", type=int, default=5)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()
    d = common.load()
    readings = {n: json.loads((common.READINGS / f"{n}_train.json").read_text(encoding="utf-8")) for n in READS}
    synth = [json.loads(line) for line in SYNTH.read_text(encoding="utf-8").splitlines() if line.strip()]
    held_apps = set(json.loads((common.REPO / "src" / "lifemate" / "splits.json").read_text())["held_apps"])
    for name, rows in common.pools(d):
        held = held_apps if name == "fold0" else set()
        noise = common.noise_from(rows, d["avg5"])
        contexts = [r["context"] for r in rows + synth]
        rows_ok = [r for r in rows if not apps_of(r) & held]
        packs, report = build(rows_ok, synth, contexts, readings, noise, args, held)
        common.write(f"t3_{name}", packs)
        print(json.dumps({"pool": name, **report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
