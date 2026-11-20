"""Post-processing rules that cannot change the saved state, measured on a validation run.

1. Gold habit: when a read call with a `query` finds an item that a later call uses by id, is the query that item's
   exact stored name?
2. Contact-name rewrite: replay the predicted calls; where a `contacts.search` found exactly one contact that a later
   call uses by id, set its query to that contact's stored name. Reads do not change the state, so the hash cannot
   move (checked by replay); the score is recomputed with the local scorer.
3. Batch conventions (calendar end, create_rule title) applied with the batch read from visible inputs only.
4. Accuracy of the input-only batch detector, and the shape of predicted asks.

Usage: python scripts/eda/post_rules.py outputs/harm-check5/avg5_eval
"""
import collections
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, gen_by_id, gen_by_input, load, public_batch  # noqa: E402
from errors import load_run, scorer, T  # noqa: E402

sys.path.insert(0, str(DATA / "train"))
from zaic_tools import ToolEnv  # noqa: E402


def run(sid, calls):
    """Replay calls; return (results, state hash)."""
    env = ToolEnv("train", sid, root=str(DATA))
    res = [env.call(c["name"], c.get("arguments") or {}) for c in calls]
    return res, env.get_state_hash()


def items(x):
    """Every dict with an id found in a tool result."""
    out = []
    if isinstance(x, dict):
        if "id" in x:
            out.append(x)
        for v in x.values():
            out += items(v)
    elif isinstance(x, list):
        for v in x:
            out += items(v)
    return out


def values(x):
    if isinstance(x, dict):
        return [w for v in x.values() for w in values(v)]
    if isinstance(x, list):
        return [w for v in x for w in values(v)]
    return [x]


def used_later(calls, i, ids):
    later = {v for c in calls[i + 1:] for v in values(c.get("arguments") or {}) if isinstance(v, str)}
    return [k for k in ids if k in later]


def rewrite_contacts(calls, results):
    out, n = [], 0
    for i, (c, r) in enumerate(zip(calls, results)):
        a = dict(c.get("arguments") or {})
        if c["name"] == "contacts.search" and a.get("query"):
            found = {it["id"]: it for it in items(r) if it.get("name")}
            hit = used_later(calls, i, list(found))
            if len(hit) == 1 and a["query"] != found[hit[0]]["name"]:
                a["query"] = found[hit[0]]["name"]
                n += 1
        out.append({"name": c["name"], "arguments": a})
    return out, n


def conventions(calls, batch):
    out, n = [], 0
    for c in calls:
        a = dict(c.get("arguments") or {})
        if c["name"] == "calendar.create_event" and a.get("start"):
            one_h = (dt.datetime.fromisoformat(a["start"]) + dt.timedelta(hours=1)).isoformat()
            if batch == "B" and "end" not in a:
                a["end"], n = one_h, n + 1
            elif batch == "C" and a.get("end") == one_h:
                del a["end"]
                n += 1
        if c["name"] == "home_automation.create_rule" and batch == "C" and "title" in a:
            del a["title"]
            n += 1
        out.append({"name": c["name"], "arguments": a})
    return out, n


def main(path):
    train = {s["id"]: s for s in load("train")}

    # 1. gold habit
    habit = collections.Counter()
    for s in train.values():
        calls = s["label"]["tool_calls"]
        if not any((c.get("arguments") or {}).get("query") for c in calls):
            continue
        res, _ = run(s["id"], calls)
        for i, (c, r) in enumerate(zip(calls, res)):
            q = (c.get("arguments") or {}).get("query")
            if not isinstance(q, str) or not q:
                continue
            found = {it["id"]: it for it in items(r)}
            hit = used_later(calls, i, list(found))
            if len(hit) == 1:
                it = found[hit[0]]
                name = it.get("name") or it.get("title")
                habit[(c["name"], "exact" if q == name else "other")] += 1
    print("1. gold query = stored name of the item used later")
    for tool in sorted({t for t, _ in habit}):
        print(f"   {tool:32s} exact {habit[(tool, 'exact')]:4d}  other {habit[(tool, 'other')]:4d}")

    # 2-3. rewrites on the validation run
    rows = load_run(path)
    tot = collections.Counter()
    changed = collections.defaultdict(list)
    for sid, r in rows.items():
        s = train[sid]
        gold = s["label"]["tool_calls"]
        pred = r["pred"]
        res, h0 = run(sid, pred)
        _, gh = run(sid, gold)
        base = scorer.score_sample(pred, h0, gold, gh, T)
        tot["base"] += base
        variants = {
            "contacts": rewrite_contacts(pred, res),
            "conv_input": conventions(pred, gen_by_input(s)),
            "conv_id": conventions(pred, gen_by_id(sid)),
        }
        both, _ = conventions(variants["contacts"][0], gen_by_input(s))
        variants["contacts+conv_input"] = (both, 0)
        for k, (calls, n) in variants.items():
            _, h = run(sid, calls)
            assert h == h0, (sid, k, "state changed")
            sc = scorer.score_sample(calls, h, gold, gh, T)
            tot[k] += sc
            if abs(sc - base) > 1e-9:
                changed[k].append((sid, round(base, 3), round(sc, 3)))
    n = len(rows)
    print(f"\n2-3. validation {path} (n={n}); base {100 * tot['base'] / n:.2f} (run file {100 * sum(r['score'] for r in rows.values()) / n:.2f})")
    for k in ("contacts", "conv_input", "conv_id", "contacts+conv_input"):
        ups = sum(b < a for _, b, a in changed[k])
        print(f"   {k:20s} {100 * tot[k] / n:.2f} ({100 * (tot[k] - tot['base']) / n:+.2f}); changed {len(changed[k])}: {ups} up, {len(changed[k]) - ups} down")
        for x in changed[k][:12]:
            print("      ", x)

    # 4. batch detector and asks
    conf = collections.Counter((gen_by_id(s["id"]), gen_by_input(s)) for s in train.values())
    acc = sum(v for (a, b), v in conf.items() if a == b) / len(train)
    print(f"\n4. input-only batch rule, train accuracy {acc:.3f}; confusion (true, rule): {dict(sorted(conf.items()))}")
    pub = load("public_test")
    pconf = collections.Counter((public_batch(s["id"]), gen_by_input(s)) for s in pub)
    pacc = sum(v for (a, b), v in pconf.items() if a == b) / len(pub)
    print(f"   public agreement with the input-switch batch {pacc:.3f}; {dict(sorted(pconf.items()))}")
    asks = [(sid, i, c) for sid, r in rows.items() for i, c in enumerate(r["pred"]) if c["name"] == "sys.ask_missing_info"]
    multi = sum(len((c.get("arguments") or {}).get("params") or []) != 1 for _, _, c in asks)
    notlast = sum(i != len(rows[sid]["pred"]) - 1 for sid, i, _ in asks)
    print(f"   predicted asks {len(asks)}: params != 1 in {multi}, not the last call in {notlast}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/harm-check5/avg5_eval")
