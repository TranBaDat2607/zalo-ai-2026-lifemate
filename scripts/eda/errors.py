"""Call-level error taxonomy of a validation run (detail.*.jsonl with gold, pred, hash_ok, score)."""
import collections
import glob
import json
import sys
import unicodedata

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import catalog, load, num  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from lifemate import scorer  # noqa: E402

_, T = catalog("train")


def fold(s):
    t = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


def load_run(path):
    rows = {}
    for f in sorted(glob.glob(f"{path}/detail.*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            rows[r["id"]] = r
    return rows


def why(gold, preds):
    """Best explanation for an unmatched gold call given the unmatched predicted calls."""
    same = [p for p in preds if p["name"] == gold["name"]]
    if not same:
        return "missing_tool", None
    fz = scorer._fuzzy_params(gold["name"], T)
    best = None
    for p in same:
        a, b = p.get("arguments") or {}, gold.get("arguments") or {}
        reasons = []
        for k in set(a) | set(b):
            if k not in a:
                reasons.append(("gold_only_arg", k))
            elif k not in b:
                reasons.append(("pred_only_arg", k))
            elif k in fz:
                if scorer.token_f1(a[k], b[k]) <= 0.6:
                    reasons.append(("fuzzy_low", k))
            elif a[k] != b[k]:
                if isinstance(a[k], str) and isinstance(b[k], str) and a[k].lower() == b[k].lower():
                    reasons.append(("case_only", k))
                elif isinstance(a[k], str) and isinstance(b[k], str) and fold(a[k]) == fold(b[k]):
                    reasons.append(("diacritics_only", k))
                elif isinstance(a[k], list) and isinstance(b[k], list) and sorted(map(str, a[k])) == sorted(map(str, b[k])):
                    reasons.append(("list_order", k))
                else:
                    reasons.append(("value", k))
        if best is None or len(reasons) < len(best):
            best = reasons
    return "arg", best


if __name__ == "__main__":
    run = load_run(sys.argv[1])
    tr = {s["id"]: s for s in load("train")}
    agg = collections.Counter()
    argc = collections.Counter()
    ex = collections.defaultdict(list)
    for sid, r in run.items():
        gold, pred = r["gold"], r["pred"]
        left = list(pred)
        unmatched_gold = []
        for g in gold:
            for i, p in enumerate(left):
                if scorer.calls_match(p, g, T):
                    del left[i]
                    break
            else:
                unmatched_gold.append(g)
        for g in unmatched_gold:
            kind, det = why(g, left)
            gk = T.get(g["name"], {}).get("kind", "meta")
            if kind == "missing_tool":
                agg[(gk, "missing_tool")] += 1
            else:
                for rk, k in det:
                    agg[(gk, rk)] += 1
                    argc[(g["name"], k, rk)] += 1
                    if len(ex[rk]) < 12:
                        p = next(p for p in left if p["name"] == g["name"])
                        ex[rk].append((sid, g["name"], k, (g["arguments"].get(k)), (p.get("arguments") or {}).get(k)))
    for k, v in agg.most_common():
        print(k, v)
    print()
    for k, v in argc.most_common(40):
        print(k, v)
    for rk in ex:
        print("==", rk)
        for e in ex[rk]:
            print("   ", e)
