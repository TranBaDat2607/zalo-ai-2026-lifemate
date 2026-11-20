"""Paired comparison of two transcript sets of the same train clips, with asr_eval's normalised metrics.

    python -m lifemate.asr_compare outputs/exp7/asr_medium/medium_train.json outputs/a0:ctx

A spec ``folder:pass`` stands for every ``fold*_<pass>.json`` in the folder; a pass missing for some clips (``reads``
exists for latest requests only) falls back to ``folder:<fallback>`` when given as ``folder:reads+ctx``. Only clips both
sets have are compared. The 95 % intervals come from a bootstrap that resamples whole samples, so the turns of one
conversation stay together. Only train.jsonl is read.
"""
import argparse
import json
import random
from pathlib import Path

from . import data
from .asr_eval import score_turn, turns_of

KEYS = ("norm_errors", "norm_words", "exact", "name_found", "name_total", "foreign_found", "foreign_total",
        "number_found", "number_total")


def load_set(spec: str) -> dict:
    if ":" not in spec or Path(spec).is_file():
        return json.loads(Path(spec).read_text(encoding="utf-8"))
    folder, passes = spec.rsplit(":", 1)
    out = {}
    for name in reversed(passes.split("+")):  # the first pass named wins where it exists
        for f in sorted(Path(folder).glob(f"fold*_{name}.json")):
            out.update(json.loads(f.read_text(encoding="utf-8")))
    return out


def sums(rows: list) -> dict:
    out = {k: sum(r[k] for r in rows) for k in KEYS}
    out["turns"] = len(rows)
    critical = [r for r in rows if r["has_critical"]]
    out["critical_ok"], out["critical_turns"] = sum(r["critical_ok"] for r in critical), len(critical)
    return out


def pct(a, b):
    return 100 * a / b if b else float("nan")


def interval(by_sample: dict, rounds: int, rng: random.Random) -> tuple:
    """95 % bootstrap interval of the WER difference (b - a), resampling samples."""
    ids = list(by_sample)
    diffs = []
    for _ in range(rounds):
        ea = wa = eb = wb = 0
        for sid in (rng.choice(ids) for _ in ids):
            a, b = by_sample[sid]
            ea, wa, eb, wb = ea + a[0], wa + a[1], eb + b[0], wb + b[1]
        diffs.append(pct(eb, wb) - pct(ea, wa))
    diffs.sort()
    return diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds) - 1]


def report(name: str, turns: list, a: dict, b: dict, rounds: int, rng: random.Random) -> dict:
    ra = [score_turn(t["ref"], a[t["audio"]]) for t in turns]
    rb = [score_turn(t["ref"], b[t["audio"]]) for t in turns]
    sa, sb = sums(ra), sums(rb)
    by_sample = {}
    for t, x, y in zip(turns, ra, rb):
        cur = by_sample.setdefault(t["id"], ([0, 0], [0, 0]))
        cur[0][0] += x["norm_errors"]; cur[0][1] += x["norm_words"]  # noqa: E702
        cur[1][0] += y["norm_errors"]; cur[1][1] += y["norm_words"]  # noqa: E702
    lo, hi = interval(by_sample, rounds, rng)
    broke = sum(x["exact"] and not y["exact"] for x, y in zip(ra, rb))
    fixed = sum(y["exact"] and not x["exact"] for x, y in zip(ra, rb))
    row = {"slice": name, "turns": sa["turns"], "wer_a": round(pct(sa["norm_errors"], sa["norm_words"]), 2),
           "wer_b": round(pct(sb["norm_errors"], sb["norm_words"]), 2)}
    row["change"] = round(row["wer_b"] - row["wer_a"], 2)
    row["ci95"] = [round(lo, 2), round(hi, 2)]
    for kind in ("name", "foreign", "number"):
        row[f"{kind}_a"] = round(pct(sa[f"{kind}_found"], sa[f"{kind}_total"]), 1)
        row[f"{kind}_b"] = round(pct(sb[f"{kind}_found"], sb[f"{kind}_total"]), 1)
        row[f"{kind}_terms"] = sa[f"{kind}_total"]
    row["critical_ok_a"] = round(pct(sa["critical_ok"], sa["critical_turns"]), 1)
    row["critical_ok_b"] = round(pct(sb["critical_ok"], sb["critical_turns"]), 1)
    row["exact_a"] = round(pct(sa["exact"], sa["turns"]), 1)
    row["exact_b"] = round(pct(sb["exact"], sb["turns"]), 1)
    row["exact_fixed"], row["exact_broken"] = fixed, broke
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def main(argv=None) -> list:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="baseline transcripts: a JSON file or folder:pass")
    ap.add_argument("b", help="new transcripts: a JSON file or folder:pass[+fallback]")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", help="also write the rows to this JSON file")
    args = ap.parse_args(argv)
    a, b = load_set(args.a), load_set(args.b)
    items = data.load_split("train", data.find_root("train"))
    turns = [t for t in turns_of(items) if t["audio"] in a and t["audio"] in b]
    splits = json.loads((Path(__file__).parent / "splits.json").read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    rows = []
    for name, keep in (("all turns", lambda t: True), ("latest requests", lambda t: t["kind"] == "latest"),
                       ("earlier turns", lambda t: t["kind"] == "history"),
                       ("latest, validation 386", lambda t: t["kind"] == "latest" and t["id"] in set(splits["random"]) | set(splits["apps"])),
                       ("latest, random half", lambda t: t["kind"] == "latest" and t["id"] in set(splits["random"])),
                       ("latest, held-out apps", lambda t: t["kind"] == "latest" and t["id"] in set(splits["apps"]))):
        part = [t for t in turns if keep(t)]
        if part:
            rows.append(report(name, part, a, b, args.rounds, rng))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


if __name__ == "__main__":
    main()
