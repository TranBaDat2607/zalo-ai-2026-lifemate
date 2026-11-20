"""Four-fold cross-validation for improve_plan.md step 0.

    LIFEMATE_DATA=$PWD/train+public_test/train+public_test python scripts/make_cv_folds.py

Fold 0 is the fixed validation split (src/lifemate/splits.json: 202 random + 184 whose gold calls use one of six
held-out apps), unchanged, so earlier runs stay comparable. Folds 1-3 split the other 1,159 samples the same way:
each holds out a group of long-tail apps (every sample whose gold calls use one of them), plus a random share
stratified by batch and number of gold calls. "Held out" means what it means for fold 0: no gold call on those apps
is trained on; the apps may still be installed or appear in history elsewhere.

A fold-k model must not see a gold call on a group-k app anywhere, so its training holdout is fold k plus every other
sample whose gold calls use a group-k app (a few fold-0 random samples, or samples that also use another group's app).
Those extra samples are skipped in training and not scored in fold k.

Batch A uses core apps only, so apps with gold calls in batch A are not candidates (a single batch-A sample of a
candidate app becomes a held sample). Apps the template conversations (synth_samples.jsonl) call are not candidates
either: every fold trains on those. Groups are chosen by a seeded search that balances the held counts and keeps
samples using two groups' apps rare.

Writes src/lifemate/folds_cv.json, read by data.id_set("cv:K") (training holdout) and data.id_set("cvscore:K")
(samples scored in fold K; "cvheld:K" / "cvrandom:K" are its two parts).
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from lifemate import data  # noqa: E402

SEED = 2026
MAX_SAMPLES = 45      # an app with more non-validation gold samples than this counts as core
MAX_BATCH_A = 1       # an app with more batch-A gold samples than this counts as core
TRIES = 50000


def batch(sid: str) -> str:
    n = int(sid.split("_")[1])
    return "A" if n < 425 else ("B" if n < 919 else "C")


def gold_apps(item: dict) -> set:
    return {c["name"].split(".")[0] for c in item["label"]["tool_calls"]}


def build(items: list, splits: dict) -> dict:
    val_random, val_apps = set(splits["random"]), set(splits["apps"])
    val = val_random | val_apps
    pool = [i for i in items if i["id"] not in val]
    apps_of = {i["id"]: gold_apps(i) for i in items}

    count, in_a = Counter(), Counter()
    for item in pool:
        for app in apps_of[item["id"]]:
            count[app] += 1
            in_a[app] += batch(item["id"]) == "A"
    # Every fold trains on the template conversations (train --synth package), so an app they call cannot be held out.
    synth = [json.loads(line) for line in (REPO / "src" / "lifemate" / "synth_samples.jsonl").read_text(encoding="utf-8")
             .splitlines() if line.strip()]
    in_synth = set().union(*(gold_apps(s) for s in synth))
    candidates = sorted(a for a, n in count.items()
                        if n <= MAX_SAMPLES and in_a[a] <= MAX_BATCH_A and a not in splits["held_apps"]
                        and a not in in_synth)

    users = {a: [i["id"] for i in pool if a in apps_of[i["id"]]] for a in candidates}
    rng = random.Random(SEED)
    best = None
    for _ in range(TRIES):
        order = candidates[:]
        rng.shuffle(order)
        group = {a: n % 3 + 1 for n, a in enumerate(order)}
        held = defaultdict(set)
        for a in candidates:
            held[group[a]].update(users[a])
        cross = sum(len(held[x] & held[y]) for x, y in ((1, 2), (1, 3), (2, 3)))
        sizes = [len(held[k]) for k in (1, 2, 3)]
        cost = 10 * cross + (max(sizes) - min(sizes))
        if best is None or cost < best[0]:
            best = (cost, dict(group))
    group = best[1]
    groups = {str(k): sorted(a for a in candidates if group[a] == k) for k in (1, 2, 3)}

    # A sample using apps of two groups is scored in the fold of its lowest group and skipped by the other.
    score_held = {k: set() for k in "123"}
    for item in pool:
        ks = sorted(str(group[a]) for a in apps_of[item["id"]] if a in group)
        if ks:
            score_held[ks[0]].add(item["id"])
    taken = set().union(*score_held.values())

    # The random share fills each fold to a third of the pool, stratified like make_dpo_folds.py.
    rest = [i for i in pool if i["id"] not in taken]
    strata = defaultdict(list)
    for item in rest:
        strata[(batch(item["id"]), min(len(item["label"]["tool_calls"]), 4))].append(item["id"])
    target = {k: len(pool) / 3 - len(score_held[k]) for k in "123"}
    score_random = {k: set() for k in "123"}
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        for sid in ids:  # each sample goes to the fold furthest below its target share
            k = max("123", key=lambda f: (target[f] - len(score_random[f])) / max(target[f], 1))
            score_random[k].add(sid)

    holdout = {}
    for k in "123":
        uses_group = {i["id"] for i in items if apps_of[i["id"]] & set(groups[k])}
        holdout[k] = sorted(score_held[k] | score_random[k] | uses_group)
    return {
        "seed": SEED,
        "groups": {"0": sorted(splits["held_apps"]), **groups},
        "held": {"0": sorted(val_apps), **{k: sorted(v) for k, v in score_held.items()}},
        "random": {"0": sorted(val_random), **{k: sorted(v) for k, v in score_random.items()}},
        "holdout": {"0": sorted(val), **holdout},
    }


def report(folds: dict, items: list) -> None:
    calls = {i["id"]: len(i["label"]["tool_calls"]) for i in items}
    for k in "0123":
        score = folds["held"][k] + folds["random"][k]
        mix = Counter(batch(s) for s in score)
        print(f"fold {k}: score {len(score)} (held {len(folds['held'][k])}, random {len(folds['random'][k])}), "
              f"training holdout {len(folds['holdout'][k])} -> trains on {len(items) - len(folds['holdout'][k])}, "
              f"batches A/B/C {mix['A']}/{mix['B']}/{mix['C']}, mean gold calls "
              f"{sum(calls[s] for s in score) / len(score):.2f}")
        print(f"        apps: {', '.join(folds['groups'][k])}")


def main() -> None:
    root = data.find_root("train")
    items = data.load_split("train", root)
    splits = json.loads((REPO / "src" / "lifemate" / "splits.json").read_text(encoding="utf-8"))
    folds = build(items, splits)
    scored = [s for k in "0123" for s in folds["held"][k] + folds["random"][k]]
    assert len(scored) == len(set(scored)) == len(items), "every sample is scored in exactly one fold"
    report(folds, items)
    out = REPO / "src" / "lifemate" / "folds_cv.json"
    out.write_text(json.dumps(folds, indent=0) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
