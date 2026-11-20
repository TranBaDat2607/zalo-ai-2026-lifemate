"""Preference pairs at the first wrong step, from runs of models that never trained on those samples.

    LIFEMATE_DATA=$PWD/train+public_test/train+public_test python scripts/build_pairs.py \
        --out outputs/pairs/pairs.jsonl outputs/dpo-fold1/f1a_eval outputs/dpo-fold1/f1soup_eval ...

For every imperfect sample, walk the predicted and gold calls in order (official call matching). At the first step t
where they part, the history up to t is gold (the run matched it), so the pair is: prefix = gold calls 0..t-1 with
their gold results, chosen = gold call t (or "DONE" when gold has no more calls), rejected = the run's own step t
(its call, or "DONE" when it stopped early).

Kept only when the rejected step is a real mistake, not another valid order: the rejected call matches no gold call
at all (or it is DONE while gold calls remain). Kept only when the run's restated request (first think block) has at
most --max-word-errors word errors against the gold transcript, fillers ignored (FILLERS): a pair built on a misheard request would teach the
model to produce words it never heard. The restated request is stored and reused as the pair's think block, so the
prefix is the state the model was really in. Duplicate (id, step, rejected) pairs from several runs are merged.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from first_error import restated, word_errors, words  # noqa: E402
from lifemate import data, scorer  # noqa: E402

DONE = "DONE"
# Fillers and spelling variants that change nothing a call depends on; without this, 68 % of perfect samples count as
# misheard (only 78 of 242 perfect validation samples have 0 raw word errors).
FILLERS = {"nha", "nhé", "nhá", "nhe", "nghen", "nhen", "hen", "giùm", "dùm", "giúp", "hộ", "ạ", "à", "á", "ơi", "ừ",
           "ờ", "nhỉ", "đi", "luôn", "cái", "thì", "là", "mà", "với", "nhá", "nè", "đó", "đấy", "ấy", "này", "vậy"}


def heard_errors(said: str, gold: str) -> int:
    keep = lambda text: [w for w in words(text) if w not in FILLERS]  # noqa: E731
    return word_errors(keep(said), keep(gold))


def free_text_only(chosen, rejected, catalog: dict) -> bool:
    """Same tool, and every differing argument is a fuzzy (free-text) one: costs one call's match, never the state."""
    if not isinstance(chosen, dict) or not isinstance(rejected, dict) or chosen["name"] != rejected["name"]:
        return False
    fuzzy = scorer._fuzzy_params(chosen["name"], catalog)
    a, b = chosen.get("arguments") or {}, rejected.get("arguments") or {}
    differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    return bool(differing) and differing <= fuzzy


def load_rows(run: str) -> list:
    rows = []
    for f in sorted(Path(run).glob("detail.*.jsonl")) or sorted(Path(run).glob("detail.jsonl")):
        rows += [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def first_pair(pred: list, gold: list, catalog: dict):
    """(step, chosen, rejected, kind) at the first divergence, or None when there is no clean mistake to learn from."""
    if not gold:
        return None  # out of scope: there is no next gold call to prefer
    i = 0
    while i < len(pred) and i < len(gold) and scorer.calls_match(pred[i], gold[i], catalog):
        i += 1
    if i == len(pred) and i == len(gold):
        return None
    if i == len(pred):
        return i, gold[i], DONE, "stopped early"
    wrong = not any(scorer.calls_match(pred[i], g, catalog) for g in gold)
    if not wrong:
        return None  # a gold call made in another order: not a mistake to push down
    if i == len(gold):
        return i, DONE, pred[i], "extra call after finishing"
    kind = "wrong arguments" if pred[i]["name"] == gold[i]["name"] else "wrong tool"
    if pred[i]["name"] == "sys.ask_missing_info":
        kind = "asked instead of acting"
    elif gold[i]["name"] == "sys.ask_missing_info":
        kind = "acted instead of asking"
    return i, gold[i], pred[i], kind


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="folders with detail.*.jsonl (infer --split train outputs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-word-errors", type=int, default=2)
    ap.add_argument("--keep-free-text", action="store_true", help="also keep pairs that differ only in fuzzy arguments")
    ap.add_argument("--exclude", default=None, help="id spec (data.id_set) whose samples give no pairs, e.g. splits")
    args = ap.parse_args()
    root = data.find_root("train")
    items = {s["id"]: s for s in data.load_split("train", root)}
    catalog = data.load_catalog("train", root)
    exclude = data.id_set(args.exclude) if args.exclude else set()
    pairs, seen = [], set()
    stats = {"rows": 0, "imperfect": 0, "no clean mistake": 0, "free text only": 0, "misheard": 0, "excluded": 0, "duplicate": 0}
    for run in args.runs:
        for row in load_rows(run):
            stats["rows"] += 1
            if row["score"] > 0.999:
                continue
            stats["imperfect"] += 1
            if row["id"] in exclude:
                stats["excluded"] += 1
                continue
            found = first_pair(row["pred"], row["gold"], catalog)
            if found is None:
                stats["no clean mistake"] += 1
                continue
            step, chosen, rejected, kind = found
            if not args.keep_free_text and free_text_only(chosen, rejected, catalog):
                stats["free text only"] += 1
                continue
            said = restated(row)
            errors = heard_errors(said, items[row["id"]]["audio_text"])
            if errors > args.max_word_errors:
                stats["misheard"] += 1
                continue
            key = (row["id"], step, json.dumps(rejected, sort_keys=True, ensure_ascii=False))
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)
            pairs.append({"id": row["id"], "step": step, "kind": kind, "chosen": chosen, "rejected": rejected,
                          "restated": said, "word_errors": errors, "score": row["score"], "run": run})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
    kinds = {}
    for p in pairs:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    print(json.dumps({**stats, "pairs": len(pairs), "samples": len({p['id'] for p in pairs}), "kinds": kinds},
                     ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
