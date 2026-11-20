"""Counterfactual: normalise trailing punctuation of fuzzy arguments to each field's gold habit (learned on train folds
outside the validation ids), then rescore validation. Fuzzy fields are not hashed, so the state hash cannot change."""
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, catalog, load  # noqa: E402
from errors import load_run, scorer  # noqa: E402

_, T = catalog("train")
val = set(json.load(open(REPO / "src/lifemate/splits.json"))["random"]) | set(json.load(open(REPO / "src/lifemate/splits.json"))["apps"])
tr = [s for s in load("train") if s["id"] not in val]
habit = collections.defaultdict(lambda: [0, 0])
for s in tr:
    for c in s["label"]["tool_calls"]:
        for k in scorer._fuzzy_params(c["name"], T) & set(c["arguments"]):
            v = str(c["arguments"][k]).rstrip()
            habit[(c["name"], k)][0] += 1
            habit[(c["name"], k)][1] += v.endswith((".", "!", "?"))


def norm(name, args):
    out = dict(args)
    for k in scorer._fuzzy_params(name, T) & set(args):
        n, p = habit.get((name, k), [0, 0])
        v = str(args[k]).rstrip()
        if n >= 5 and p / n < 0.2:
            v = re.sub(r"[.!?…]+$", "", v).rstrip()
        elif n >= 5 and p / n > 0.8 and not v.endswith((".", "!", "?")):
            v = v + "."
        out[k] = v if isinstance(args[k], str) else args[k]
    return out


run = load_run(sys.argv[1])
base = new = 0.0
changed = 0
for r in run.values():
    base += r["score"]
    if not r["gold"] or not r["hash_ok"]:
        new += r["score"]
        continue
    pred = [{"name": c["name"], "arguments": norm(c["name"], c.get("arguments") or {})} for c in r["pred"]]
    s = scorer.score_sample(pred, "h", r["gold"], "h", T)
    changed += s != r["score"]
    new += s
n = len(run)
print(f"validation {base/n*100:.2f} -> {new/n*100:.2f} (+{(new-base)/n*100:.2f}), samples changed {changed}")
print("habits:", {f"{a}.{b}": f"{p}/{n}" for (a, b), (n, p) in sorted(habit.items(), key=lambda x: -x[1][0])[:14]})
