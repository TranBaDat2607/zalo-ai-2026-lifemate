"""Preference pairs v2 from step-probing rows (lifemate.probe), with mirror pairs and gold-noise filters.

    LIFEMATE_DATA=$PWD/train+public_test/train+public_test python scripts/probe_pairs.py \
        --out outputs/pairs/pairs_v2.jsonl outputs/probe-fold1/probe_f1a outputs/probe-fold1/probe_f1b ...

Why (log.md, "Step-preference: potential and data needs"): the 231 first-divergence pairs covered 121 (tool, error)
patterns, 87 of them once; the only pattern with mass (chosen = ask, 26 against 6 rejected) became a global habit.

Genuine pairs. At every probed step the history is gold, so an answer that differs from the gold step is a candidate
rejected side, kept when it is a real mistake: a call that matches no gold call of the sample (a later gold call is
another valid order, skipped; an earlier one is a repeat, kept), the closing reply while gold calls remain, or a call
after the last gold call. Filters, as in build_pairs.py: the restated request heard ok (<= 2 word errors, fillers
ignored), not free-text only. New filters:
- gold noise: chosen and rejected differ only in whether an optional argument is present, and gold itself uses that
  argument in 20-80 % of the tool's calls (train outside validation): no rule to learn;
- values: they differ only in number or date/time arguments, which need arithmetic the model does not write out
  (``--keep-values`` keeps them).
Each pair records how many of the step's answers made that mistake (``n_wrong`` of ``n_answers``): a mistake the greedy
answer makes, or most samples make, is one the model really holds.

Mirror pairs fill each move's shortfall only (``--mirror-ratio`` of chosen minus rejected, counted on the genuine
pairs), on the least sure heard-ok gold steps first (lowest lp_gold; a pair the model already gets right adds almost no
gradient):
- ask chosen more often than rejected: at gold write calls with an id argument, rejected = asking for that id instead
  (same tool when possible);
- DONE chosen more often than rejected (extra calls outnumber early stops): at a later gold call, rejected = stopping;
- an argument present on the rejected side more often (e.g. contacts.search {} chosen against {query}): at gold calls
  of that tool that have it, rejected = the same call without it.
A move still one-sided by more than ``--max-ratio`` after mirroring (an argument present on the chosen side more often
cannot be mirrored: the rejected side would need a value) is capped by weight: the dominant side's pairs get weight
max_ratio * minor / major, 0 if the move never appears on the other side (those pairs are left out). Pair weights are
read by lifemate.contrast. The summary prints the genuine balance and, per move, counts and weighted counts after
mirroring.
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from build_pairs import first_pair, free_text_only, heard_errors, load_rows as load_eval  # noqa: E402
from first_error import arg_kind, restated as eval_restated, wrong_args  # noqa: E402
from lifemate import data, scorer  # noqa: E402

DONE = "DONE"
ASK = "sys.ask_missing_info"


def load_rows(folder: str) -> list:
    rows = []
    for f in sorted(Path(folder).glob("probe.*.jsonl")):
        rows += [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def is_call(x) -> bool:
    return isinstance(x, dict) and "name" in x


def mistake_kind(answer, gold, step: int, calls: list, catalog: dict):
    """The kind of a wrong answer at ``step``, or None when it is not a mistake to push down."""
    if answer is None:
        return None  # malformed: a format problem, not a preference
    if not is_call(gold):
        return "extra call after finishing" if is_call(answer) else None
    if answer == DONE:
        return "stopped early"
    if scorer.calls_match(answer, gold, catalog):
        return None
    if any(scorer.calls_match(answer, g, catalog) for g in calls[step + 1:]):
        return None  # a later gold call: another valid order
    if any(scorer.calls_match(answer, g, catalog) for g in calls[:step]):
        return "repeated call"
    if answer["name"] == ASK:
        return "asked instead of acting"
    if gold["name"] == ASK:
        return "acted instead of asking"
    return "wrong arguments" if answer["name"] == gold["name"] else "wrong tool"


def presence_rates(items: list) -> dict:
    """{(tool, argument): share of the tool's gold calls that have the argument}."""
    calls = collections.defaultdict(int)
    has = collections.defaultdict(int)
    for item in items:
        for c in item["label"]["tool_calls"]:
            calls[c["name"]] += 1
            for key in c.get("arguments") or {}:
                has[(c["name"], key)] += 1
    return {k: v / calls[k[0]] for k, v in has.items()}, dict(calls)


def differing(chosen: dict, rejected: dict) -> tuple:
    a, b = chosen.get("arguments") or {}, rejected.get("arguments") or {}
    presence = {k for k in set(a) ^ set(b)}
    values = {k for k in set(a) & set(b) if a[k] != b[k]}
    return presence, values


def noise_or_value(chosen, rejected, catalog: dict, rates: dict, low: float, high: float) -> str:
    """'gold noise', 'values' or '' for a same-tool pair."""
    if not (is_call(chosen) and is_call(rejected)) or chosen["name"] != rejected["name"]:
        return ""
    presence, values = differing(chosen, rejected)
    name = chosen["name"]
    if presence and not values and all(low <= rates.get((name, k), 0.0) <= high for k in presence):
        return "gold noise"
    if values and not presence:
        merged = {**(rejected.get("arguments") or {}), **(chosen.get("arguments") or {})}
        if all(arg_kind(name, k, merged[k], catalog) in ("number", "date/time") for k in values):
            return "values"
    return ""


def move_of(side) -> list:
    """The moves a side makes, for the balance table: 'ask', 'DONE', 'omit <tool>.<arg>' / 'fill <tool>.<arg>' are
    filled in by the caller from the pair; here only the coarse ones."""
    if side == DONE:
        return ["DONE"]
    if is_call(side) and side["name"] == ASK:
        return ["ask"]
    return []


def moves_with_side(p: dict) -> list:
    """[(move, True if on the chosen side)] of a pair: 'ask', 'DONE', and '<tool>.<arg> present' when chosen and
    rejected are the same tool and differ in whether that argument is present."""
    out = [(m, True) for m in move_of(p["chosen"])] + [(m, False) for m in move_of(p["rejected"])]
    if is_call(p["chosen"]) and is_call(p["rejected"]) and p["chosen"]["name"] == p["rejected"]["name"]:
        presence, _ = differing(p["chosen"], p["rejected"])
        for k in presence:
            out.append((f"{p['chosen']['name']}.{k} present", k in (p["chosen"].get("arguments") or {})))
    return out


def balance_of(pairs: list, weighted: bool = False) -> dict:
    """{move: [chosen, rejected]} counts (or summed weights)."""
    balance = collections.defaultdict(lambda: [0, 0])
    for p in pairs:
        for move, chosen_side in moves_with_side(p):
            balance[move][0 if chosen_side else 1] += p.get("weight", 1.0) if weighted else 1
    return dict(balance)


def id_params(name: str, catalog: dict) -> list:
    props = (catalog.get(name, {}).get("parameters") or {}).get("properties") or {}
    return sorted(k for k, v in props.items() if v.get("x-ref"))


def pattern(chosen, rejected, kind: str, catalog: dict) -> tuple:
    """(gold tool, error) with the argument kind for wrong arguments, as in the 10 Nov coverage analysis."""
    tool = chosen["name"] if is_call(chosen) else (rejected["name"] if is_call(rejected) else DONE)
    if kind == "wrong arguments":
        kind = "args: " + wrong_args(rejected, chosen, catalog)
    return tool, kind


def coverage(run: str, pairs: list, items: dict, catalog: dict, max_word_errors: int) -> dict:
    """Points of a validation run's heard-ok errors whose (tool, error) pattern has >= 5 / 1-4 / 0 genuine pairs."""
    counts = collections.Counter(pattern(p["chosen"], p["rejected"], p["kind"], catalog) for p in pairs)
    tools = collections.Counter(t for t, _ in counts.elements())
    rows = load_eval(run)
    out = collections.defaultdict(lambda: [0, 0.0])
    for r in rows:
        if r["score"] > 0.999 or heard_errors(eval_restated(r), items[r["id"]]["audio_text"]) > max_word_errors:
            continue
        found = first_pair(r["pred"], r["gold"], catalog)
        lost = 100 * (1 - r["score"]) / len(rows)
        if found is None:
            key = "no clean first mistake"
        else:
            n = counts[pattern(found[1], found[2], found[3], catalog)]
            tool = found[1]["name"] if is_call(found[1]) else DONE
            key = (">= 5 pairs" if n >= 5 else "1-4 pairs" if n else
                   "tool in pairs, other error" if tools[tool] else "tool in no pair")
        out[key][0] += 1
        out[key][1] = round(out[key][1] + lost, 2)
    return {k: {"samples": v[0], "points": v[1]} for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="probe output folders (probe.*.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-word-errors", type=int, default=2)
    ap.add_argument("--keep-values", action="store_true")
    ap.add_argument("--keep-free-text", action="store_true")
    ap.add_argument("--noise-low", type=float, default=0.2)
    ap.add_argument("--noise-high", type=float, default=0.8)
    ap.add_argument("--mirror-ratio", type=float, default=1.0, help="share of each move's shortfall filled by mirrors")
    ap.add_argument("--max-ratio", type=float, default=2.0, help="cap, by weight, on chosen:rejected of any move")
    ap.add_argument("--exclude", default="splits", help="ids that give no pairs (default: the validation set)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--coverage", default="outputs/notes-aux/aux_eval",
                    help="validation run whose heard-ok errors the pairs should cover ('' = skip)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    root = data.find_root("train")
    items = {s["id"]: s for s in data.load_split("train", root)}
    catalog = data.load_catalog("train", root)
    exclude = data.id_set(args.exclude) if args.exclude else set()
    rates, _ = presence_rates([i for k, i in items.items() if k not in exclude])

    stats = collections.Counter()
    genuine, seen, steps = {}, set(), []
    for run in args.runs:
        for row in load_rows(run):
            if row["id"] in exclude:
                stats["excluded rows"] += 1
                continue
            steps.append((run, row))
            item = items[row["id"]]
            calls, gold = item["label"]["tool_calls"], row["gold"]
            counts = collections.Counter(json.dumps(a["answer"], sort_keys=True, ensure_ascii=False)
                                         for a in row["answers"])
            for k, answer in enumerate(row["answers"]):
                stats["answers"] += 1
                kind = mistake_kind(answer["answer"], gold, row["step"], calls, catalog)
                if kind is None:
                    continue
                stats["mistakes"] += 1
                restated = answer.get("restated", row["restated"]) if row["step"] == 0 else row["restated"]
                if heard_errors(restated, item["audio_text"]) > args.max_word_errors:
                    stats["dropped: misheard"] += 1
                    continue
                rejected = answer["answer"]
                if not args.keep_free_text and free_text_only(gold, rejected, catalog):
                    stats["dropped: free text only"] += 1
                    continue
                why = noise_or_value(gold, rejected, catalog, rates, args.noise_low, args.noise_high)
                if why == "gold noise" or (why == "values" and not args.keep_values):
                    stats[f"dropped: {why}"] += 1
                    continue
                key = (row["id"], row["step"], json.dumps(rejected, sort_keys=True, ensure_ascii=False))
                if key in seen:
                    genuine[key]["runs"] = sorted(set(genuine[key]["runs"]) | {run})
                    stats["duplicates merged"] += 1
                    continue
                seen.add(key)
                genuine[key] = {"id": row["id"], "step": row["step"], "kind": kind, "chosen": gold, "rejected": rejected,
                                "restated": restated, "source": "probe", "greedy": k == 0,
                                "n_wrong": counts[key[2]], "n_answers": len(row["answers"]),
                                "lp_gold": row["lp_gold"], "lp_rejected": answer["lp"], "runs": [run]}
    pairs = list(genuine.values())

    # Mirrors fill only the shortfall of each move (10 Nov: mirroring every ask/omit pair one for one overshot where
    # genuine pairs already pushed the other way, e.g. contacts.search query 21:14 became 35:14). Candidate steps:
    # probed gold steps, one per (id, step), heard ok; the least sure gold steps (lowest lp_gold) first, since a pair
    # on a step the model already gets right adds almost no gradient (the contrast saturates).
    by_step = {}
    for run, row in steps:
        by_step.setdefault((row["id"], row["step"]), row)
    heard = [r for r in by_step.values() if heard_errors(r["restated"], items[r["id"]]["audio_text"]) <= args.max_word_errors]
    heard.sort(key=lambda r: (r["lp_gold"], rng.random()))
    before = balance_of(pairs)
    mirrors = []

    def short(move):
        c, r = before.get(move, (0, 0))
        return round(max(0, c - r) * args.mirror_ratio)

    # ask chosen more often than rejected: act with the id where gold writes, rejected = asking for it
    asked_tools = collections.Counter((p["chosen"].get("arguments") or {}).get("tool_name") for p in pairs
                                      if is_call(p["chosen"]) and p["chosen"]["name"] == ASK)
    write_steps = [r for r in heard if is_call(r["gold"]) and r["gold"]["name"] != ASK
                   and catalog.get(r["gold"]["name"], {}).get("kind") == "write" and id_params(r["gold"]["name"], catalog)]
    write_steps.sort(key=lambda r: -asked_tools.get(r["gold"]["name"], 0))  # same tools first (stable: then least sure)
    for r in write_steps[:short("ask")]:
        name = r["gold"]["name"]
        mirrors.append({"id": r["id"], "step": r["step"], "kind": "mirror: act, not ask", "chosen": r["gold"],
                        "rejected": {"name": ASK, "arguments": {"tool_name": name, "params": id_params(name, catalog)}},
                        "restated": r["restated"], "source": "mirror"})
    # DONE chosen more often than rejected (extra calls outnumber early stops): keep going where gold calls again
    go_steps = [r for r in heard if is_call(r["gold"]) and r["step"] > 0]
    for r in go_steps[:short("DONE")]:
        mirrors.append({"id": r["id"], "step": r["step"], "kind": "mirror: keep going", "chosen": r["gold"],
                        "rejected": DONE, "restated": r["restated"], "source": "mirror"})
    # an argument present on the rejected side more often: gold calls that have it, rejected = without it
    for tag, (c, r_) in before.items():
        if not tag.endswith(" present") or r_ <= c:
            continue
        name, key = tag[:-len(" present")].rsplit(".", 1)
        cands = [r for r in heard if is_call(r["gold"]) and r["gold"]["name"] == name
                 and key in (r["gold"].get("arguments") or {})]
        for r in cands[:round((r_ - c) * args.mirror_ratio)]:
            args_wo = {a: v for a, v in r["gold"]["arguments"].items() if a != key}
            mirrors.append({"id": r["id"], "step": r["step"], "kind": f"mirror: keep {key}", "chosen": r["gold"],
                            "rejected": {"name": name, "arguments": args_wo}, "restated": r["restated"],
                            "source": "mirror"})
    stats["genuine pairs"], stats["mirror pairs"] = len(pairs), len(mirrors)

    # A move still one-sided by more than --max-ratio (no mirror possible, e.g. an argument gold rarely has, so no
    # value to add) is capped by weight: its dominant side's pairs get weight max_ratio * minor / major (0 when the
    # move never appears on the other side).
    counted = balance_of(pairs + mirrors)
    for p in pairs + mirrors:
        p["weight"] = 1.0
    for _ in range(20):  # repeated: a pair that swaps two arguments carries two moves, so capping one moves the other
        weighted = balance_of(pairs + mirrors, weighted=True)
        factor = {}
        for move, (c, r) in weighted.items():
            major, minor = max(c, r), min(c, r)
            if major > 1e-9 and major > args.max_ratio * minor + 1e-9:
                factor[(move, c >= r)] = args.max_ratio * minor / major
        if not factor:
            break
        for p in pairs + mirrors:
            for move, chosen_side in moves_with_side(p):
                p["weight"] *= factor.get((move, chosen_side), 1.0)
    for p in pairs + mirrors:
        p["weight"] = round(p["weight"], 4)
    stats["pairs down-weighted"] = sum(p["weight"] < 1 for p in pairs + mirrors)
    stats["pairs at weight 0"] = sum(p["weight"] == 0 for p in pairs + mirrors)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs + mirrors if p["weight"] > 0),
                   encoding="utf-8")
    weighted = balance_of(pairs + mirrors, weighted=True)
    balance = {k: [counted[k][0], counted[k][1], round(weighted[k][0], 2), round(weighted[k][1], 2)] for k in counted}
    patterns = collections.Counter(pattern(p["chosen"], p["rejected"], p["kind"], catalog) for p in pairs)
    summary = {"stats": dict(stats), "samples": len({p["id"] for p in pairs}),
               "kinds": dict(collections.Counter(p["kind"] for p in pairs + mirrors)),
               "greedy mistakes": sum(p["greedy"] for p in pairs),
               "patterns": len(patterns), "patterns with >= 5": sum(v >= 5 for v in patterns.values()),
               "singleton patterns": sum(v == 1 for v in patterns.values()),
               "coverage of validation errors": (coverage(args.coverage, [p for p in pairs if p["weight"] > 0], items,
                                                          catalog, args.max_word_errors)
                                                 if args.coverage else None),
               "genuine balance (chosen, rejected)": {k: v for k, v in sorted(before.items(), key=lambda kv: -sum(kv[1]))},
               "balance (chosen, rejected, weighted chosen, weighted rejected)":
                   {k: v for k, v in sorted(balance.items(), key=lambda kv: -sum(kv[1][:2]))},
               "worst weighted ratio": max((max(a, b) / min(a, b) if min(a, b) else (float("inf") if max(a, b) else 1.0))
                                           for _, _, a, b in balance.values()) if balance else 1.0}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    (out.with_suffix(".summary.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
