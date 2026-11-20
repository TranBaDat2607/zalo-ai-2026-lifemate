"""Why the state hash failed: replay predicted and gold calls on train samples and diff the canonical states."""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, catalog, load, num  # noqa: E402
from errors import load_run, T  # noqa: E402

sys.path.insert(0, str(DATA / "train"))
from zaic_tools import ToolEnv  # noqa: E402
from zaic_tools.utils import canon  # noqa: E402


def replay(sid, calls):
    env = ToolEnv("train", sid, root=str(DATA))
    for c in calls:
        env.call(c["name"], c.get("arguments") or {})
    return canon(env.get_state())


def diff(a, b, path=""):
    """List of (path, a_value, b_value) leaves that differ."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append((f"{path}.{k}", "<absent>", b[k]))
            elif k not in b:
                out.append((f"{path}.{k}", a[k], "<absent>"))
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((f"{path}[len]", len(a), len(b)))
            ids_a = [x.get("id") for x in a if isinstance(x, dict)]
            ids_b = [x.get("id") for x in b if isinstance(x, dict)]
            if ids_a or ids_b:
                for x in a:
                    if isinstance(x, dict) and x.get("id") not in ids_b:
                        out.append((f"{path}[extra]", x, None))
                for x in b:
                    if isinstance(x, dict) and x.get("id") not in ids_a:
                        out.append((f"{path}[missing]", None, x))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append((path, a, b))
    return out


def classify(d):
    kinds = set()
    for p, a, b in d:
        if p.endswith("[len]"):
            kinds.add("item_added_or_removed")
        elif "[extra]" in p or "[missing]" in p:
            continue
        else:
            leaf = p.split(".")[-1]
            kinds.add(f"field:{leaf}")
    return kinds


if __name__ == "__main__":
    run = load_run(sys.argv[1])
    tr = {s["id"]: s for s in load("train")}
    reasons = collections.Counter()
    by_app = collections.Counter()
    examples = collections.defaultdict(list)
    n = 0
    for sid, r in run.items():
        if r["hash_ok"] or not r["gold"]:
            continue
        n += 1
        a = replay(sid, r["pred"]); b = replay(sid, r["gold"])
        d = diff(a, b)
        ks = classify(d) or {"no_diff?"}
        for k in ks:
            reasons[k] += 1
        for p, _, _ in d:
            by_app[p.split(".")[1] if p.count(".") else p] += 0
        apps = sorted({p.split(".")[1] for p, _, _ in d if p.count(".")})
        for ap in apps:
            by_app[ap] += 1
        for k in ks:
            if len(examples[k]) < 6:
                examples[k].append((sid, r["request"][:80], [(p, str(x)[:60], str(y)[:60]) for p, x, y in d][:3]))
    print("hash failures:", n)
    for k, v in reasons.most_common(40):
        print(f"  {k:40s} {v}")
    print("apps in diffs:", by_app.most_common(25))
    for k in list(examples)[:25]:
        print("==", k)
        for e in examples[k]:
            print("   ", e)
