"""Per-batch profile of train, validation and public (batches as in common.gen_by_id / public_batch).

Usage: python scripts/eda/batch_table.py outputs/harm-check5/avg5_eval outputs/asr-avg5/avg5_train.json \
           outputs/final1/avg5_asr/public_test.json
"""
import collections
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from asr_slices import align, normal_tokens, region  # noqa: E402  (normal_tokens comes from lifemate.asr_eval)
from common import REPO, gen_by_id, load, num, public_batch  # noqa: E402
from errors import load_run  # noqa: E402
from hash_diff import DATA  # noqa: E402  (puts the train tool library on the path)
from zaic_tools.apps import TOOLS  # noqa: E402


def apps(s):
    return [a if isinstance(a, str) else a.get("name", a.get("app")) for a in s["context"]["installed_apps"]]


def outcome(calls):
    names = [c["name"] for c in calls]
    if not names:
        return "empty"
    if names[-1] == "sys.ask_missing_info":
        return "ask"
    return "act" if any(TOOLS.get(n, "write") == "write" for n in names) else "read"


def main(run, oof, pub_asr):
    tr = load("train")
    pub = load("public_test")
    hyp = json.load(open(oof, encoding="utf-8"))
    pasr = json.load(open(pub_asr, encoding="utf-8"))
    splits = json.loads((REPO / "src" / "lifemate" / "splits.json").read_text(encoding="utf-8"))
    rand = set(splits["random"])
    rows = load_run(run)
    trapps = {a for s in tr for a in apps(s)}
    by = collections.defaultdict(list)
    for s in tr:
        by[gen_by_id(s["id"])].append(s)

    print("TRAIN")
    for b in "ABC":
        S = by[b]
        ids = [num(s["id"]) for s in S]
        oc = collections.Counter(outcome(s["label"]["tool_calls"]) for s in S)
        print(f" {b}: n {len(S)} ids {min(ids)}-{max(ids)} apps {st.mean(len(apps(s)) for s in S):.1f} "
              f"hist {st.mean(sum(h['role'] == 'user' for h in s['history']) for s in S):.2f} "
              f"calls {st.mean(len(s['label']['tool_calls']) for s in S):.2f} words {st.mean(len(s['audio_text'].split()) for s in S):.1f} "
              f"cities {len({s['context']['user_profile'].get('city') for s in S})} "
              f"personas {len({s['context']['user_profile'].get('name') for s in S})} "
              + " ".join(f"{k} {oc[k] / len(S):.2f}" for k in ("act", "ask", "read", "empty")))

    print("SPEECH (out of fold)")
    wer = collections.defaultdict(lambda: [0, 0, 0, 0])  # err, n, exact, turns
    final_err = {}
    reg = collections.Counter()
    for s in tr:
        b = gen_by_id(s["id"])
        turns = [(s["audio"], s["audio_text"], True)] + [(t["audio"], t["audio_text"], False) for t in s["history"] if t["role"] == "user"]
        for a, ref, final in turns:
            if a not in hyp:
                continue
            r, h = normal_tokens(ref), normal_tokens(hyp[a])
            ops, ins = align(r, h)
            e = sum(o != "ok" for o in ops) + ins
            for k in (b, b + ("-final" if final else "-hist")):
                w = wer[k]
                w[0] += e; w[1] += len(r); w[2] += e == 0; w[3] += 1
            if final:
                final_err[s["id"]] = e
                reg[(b, region(ref))] += 1
    for k in sorted(wer):
        e, n, x, t = wer[k]
        print(f" {k:8s} turns {t:5d} WER {100 * e / n:5.2f}% exact {100 * x / t:4.1f}%")
    for b in "ABC":
        tot = sum(v for (g, _), v in reg.items() if g == b)
        print(f" dialect {b} (final turns): " + " ".join(f"{r} {reg[(b, r)] / tot:.2f}" for r in ("North", "South", "Central", "unmarked")))

    print("VALIDATION", run)
    vb = collections.defaultdict(list)
    for sid, r in rows.items():
        vb[gen_by_id(sid)].append(r)
    for b in "ABC":
        R = vb[b]
        ra = [r["score"] for r in R if r["id"] in rand]
        he = [r["score"] for r in R if r["id"] not in rand]
        g = collections.defaultdict(list)
        for r in R:
            e = final_err.get(r["id"])
            g["exact" if e == 0 else ("1-2" if e is not None and e <= 2 else "3+")].append(r["score"])
        print(f" {b}: n {len(R)} score {100 * st.mean(r['score'] for r in R):.1f} random {100 * st.mean(ra):.1f} (n {len(ra)})"
              + (f" held {100 * st.mean(he):.1f} (n {len(he)})" if he else "")
              + " | final-turn words wrong: " + " ".join(f"{k} {100 * st.mean(g[k]):.1f} (n {len(g[k])})" for k in ("exact", "1-2", "3+") if g[k]))
    vshare = {b: len(vb[b]) / len(rows) for b in "ABC"}

    print("PUBLIC")
    pby = collections.defaultdict(list)
    for s in pub:
        pby[public_batch(s["id"])].append(s)
    pshare = {}
    for b in "ABC":
        S = pby[b]
        pshare[b] = len(S) / len(pub)
        inst = [a for s in S for a in apps(s)]
        rg = collections.Counter(region(pasr.get(s["audio"], "")) for s in S)
        print(f" {b}: n {len(S)} share {pshare[b]:.3f} months {dict(collections.Counter(s['context']['current_time'][5:7] for s in S))} "
              f"apps {st.mean(len(apps(s)) for s in S):.1f} hist {st.mean(sum(h['role'] == 'user' for h in s['history']) for s in S):.2f} "
              f"personas {len({s['context']['user_profile'].get('name') for s in S})} "
              f"new-app samples {st.mean(bool(set(apps(s)) - trapps) for s in S):.2f} new among installs {sum(a not in trapps for a in inst) / len(inst):.3f} "
              + "dialect " + " ".join(f"{r} {rg[r] / len(S):.2f}" for r in ("North", "South", "Central", "unmarked")))
    tshare = {b: len(by[b]) / len(tr) for b in "ABC"}
    vs = {b: st.mean(r["score"] for r in vb[b]) * 100 for b in "ABC"}
    print("shares train", {b: round(v, 3) for b, v in tshare.items()}, "validation", {b: round(v, 3) for b, v in vshare.items()},
          "public", {b: round(v, 3) for b, v in pshare.items()})
    print(f"validation reweighted to the public mix: {sum(pshare[b] * vs[b] for b in 'ABC'):.2f} (plain {100 * st.mean(r['score'] for r in rows.values()):.2f})")

    print("HABITS (gold, train)")
    hab = collections.Counter()
    for s in tr:
        b = gen_by_id(s["id"])
        calls = s["label"]["tool_calls"]
        for c in calls:
            a = c.get("arguments") or {}
            n = c["name"]
            if n == "calendar.create_event":
                hab[(n, "end", b, "end" in a)] += 1
            elif n == "home_automation.create_rule":
                hab[(n, "title", b, "title" in a)] += 1
            elif n == "phone.call":
                hab[(n, "contact_id", b, "contact_id" in a)] += 1
            elif n == "calendar.list_events":
                hab[(n, "dates", b, bool(a))] += 1
            elif n == "notes.create":
                hab[(n, "title", b, "title" in a)] += 1
            elif n == "maps.search_place":
                hab[(n, "near", b, "near" in a)] += 1
            elif n == "appliances.start_cycle":
                hab[(n, "delay_minutes", b, "delay_minutes" in a)] += 1
            elif n == "contacts.search":
                hab[(n, "query", b, bool(a.get("query")))] += 1
    keys = sorted({k[:2] for k in hab})
    for t, f in keys:
        print(f" {t:30s} {f:14s} " + "  ".join(f"{b}: {hab[(t, f, b, True)]}/{hab[(t, f, b, True)] + hab[(t, f, b, False)]}" for b in "ABC"))
    last_end = max((num(s["id"]) for s in tr for c in s["label"]["tool_calls"]
                    if c["name"] == "calendar.create_event" and "end" in (c.get("arguments") or {})), default=None)
    print(" last train id with an explicit calendar end:", last_end)


if __name__ == "__main__":
    main(*sys.argv[1:4])
