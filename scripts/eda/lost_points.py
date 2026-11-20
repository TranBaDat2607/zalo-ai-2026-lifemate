"""Where the validation points go: hash failures vs partial F1, and counterfactual gains per error type."""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import catalog, load, num  # noqa: E402
from errors import load_run, T, scorer  # noqa: E402

FAM = [(1, 438, "F1 core multi-intent"), (438, 1095, "F2 broad apps"), (1095, 1533, "F3 follow-up"),
       (1533, 1643, "F4 mid history"), (1643, 9999, "F5 long chit-chat")]


def fam(sid):
    n = num(sid)
    return next(name for lo, hi, name in FAM if lo <= n < hi)


def f1(pred, gold, match):
    if not gold:
        return float(not pred)
    left = list(gold); m = 0
    for p in pred:
        for i, g in enumerate(left):
            if match(p, g):
                del left[i]; m += 1; break
    if m == 0:
        return 0.0
    P, R = m / len(pred), m / len(gold)
    return 2 * P * R / (P + R)


def loose(p, g, case=False, fuzzy=False, lists=False, reads=False):
    if p["name"] != g["name"]:
        return False
    a, b = p.get("arguments") or {}, g.get("arguments") or {}
    if a.keys() != b.keys():
        return False
    fz = scorer._fuzzy_params(g["name"], T)
    for k in b:
        x, y = a[k], b[k]
        if k in fz:
            if fuzzy:
                continue
            if scorer.token_f1(x, y) <= 0.6:
                return False
        elif x != y:
            if case and isinstance(x, str) and isinstance(y, str) and x.lower() == y.lower():
                continue
            if lists and isinstance(x, list) and isinstance(y, list) and sorted(map(str, x)) == sorted(map(str, y)):
                continue
            return False
    return True


if __name__ == "__main__":
    run = load_run(sys.argv[1])
    n = len(run)
    base = sum(r["score"] for r in run.values()) / n * 100
    hashfail = [r for r in run.values() if not r["hash_ok"] and r["gold"]]
    emptygold = [r for r in run.values() if not r["gold"]]
    print(f"samples {n} score {base:.2f}")
    print(f"gold empty: {len(emptygold)} (scored {sum(r['score'] for r in emptygold):.0f})")
    print(f"hash fail (gold non-empty): {len(hashfail)} -> {len(hashfail)/n*100:.2f} points")
    partial = [r for r in run.values() if r["hash_ok"] and r["gold"] and r["score"] < 1]
    print(f"hash ok but F1<1: {len(partial)} -> {sum(1-r['score'] for r in partial)/n*100:.2f} points")
    hm = [r for r in run.values() if r["hash_ok"] and r["gold"] and r["score"] == 0]
    print(f"hash ok but no call matched: {len(hm)}")
    # counterfactuals on hash-ok samples
    for label, kw in [("fuzzy always ok", dict(fuzzy=True)), ("case-insensitive", dict(case=True)), ("list order free", dict(lists=True))]:
        s = 0
        for r in run.values():
            if r["gold"] and r["hash_ok"]:
                s += f1(r["pred"], r["gold"], lambda p, g: loose(p, g, **kw)) - r["score"]
        print(f"  counterfactual {label}: +{s/n*100:.2f}")
    # reads perfect: drop reads from both sides
    s = 0
    for r in run.values():
        if r["gold"] and r["hash_ok"]:
            gold = r["gold"]; pred = r["pred"]
            kinds = lambda c: T.get(c["name"], {}).get("kind", "meta")
            # perfect reads = pred reads replaced by gold reads
            newpred = [c for c in pred if kinds(c) != "read"] + [c for c in gold if kinds(c) == "read"]
            s += f1(newpred, gold, lambda p, g: loose(p, g)) - r["score"]
    print(f"  counterfactual perfect read calls (hash-ok samples only): +{s/n*100:.2f}")
    # by family
    print()
    byf = collections.defaultdict(list)
    for r in run.values():
        byf[fam(r["id"])].append(r)
    for k in sorted(byf):
        rs = byf[k]
        hf = sum(1 for r in rs if r["gold"] and not r["hash_ok"])
        print(f"{k:22s} n={len(rs):3d} score={sum(r['score'] for r in rs)/len(rs)*100:5.1f} hashfail={hf/len(rs)*100:4.1f}% perfect={sum(r['score']==1 for r in rs)/len(rs)*100:4.1f}%")
