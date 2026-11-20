"""T2b of improvment_plan3.md: train samples heard through the measured word confusions (synthetic hearing noise).

    python scripts/p3/make_t2b.py --per-sample 1     # outputs/p3/t2b_full.jsonl, t2b_fold0.jsonl

Each clip's gold text gets the substitutions and drops the averaged reading made on train audio (common.RealNoise
learned from gold vs avg5_train.json, the fold-0 table without the validation clips): only confusions that change a
word (no particle swaps, tone-mark placement, tôi/tui...), each word misheard at its measured rate times a per-request
difficulty; up to 6 simulations per sample, the first that changes the latest request is kept, so requests real readings
rarely get wrong are often left out. The history shows the real averaged readings. Words of exact argument values,
digits and number words are never changed (gate G2). A variant must change the latest request and differ from its
real readings, so it adds something T2a does not have. The gold calls stay as they are.
"""
import argparse
import json
import random
import re

import common

SINGLES = ["avg5", "medium", "medium10", "b", "s2027", "s2028"]


def build(rows: list, noise, readings: dict, avg5: dict, per: int, seed: int, catalog: dict = None) -> tuple:
    out, failed = [], 0
    for r in rows:
        clips = common.prompts.gold_transcripts(r)
        keep = common.protected(r)
        bare = lambda t: re.sub(r"[.?!…\s]+$", "", t or "")  # noqa: E731  a "?" alone is no new hearing
        known = {bare(readings[n].get(r["audio"])) for n in SINGLES} | {bare(r["audio_text"])}
        got = 0
        # 13 Nov Opus review: one simulation per variant, no retries (retrying forced errors onto the verbs of short,
        # easy requests); when the averaged reading has no meaningful error, the noise goes onto that reading, so the
        # text keeps the speech model's own punctuation, digits and spacing instead of the gold text's
        ref = r["audio_text"]
        clean = not [p for p in common._confusions(ref, avg5[r["audio"]])[1] if not common.trivial_pair(*p)]
        base = avg5[r["audio"]] if clean else clips[r["audio"]]
        for attempt in range(per):
            rng = random.Random(f"{seed}:{r['id']}:{attempt}")
            # the history as the speech model heard it (avg5, what the test shows); the latest request simulated
            texts = {a: avg5[a] for a in clips if a != r["audio"]}
            texts[r["audio"]] = noise.apply(base, rng, keep)
            if bare(texts[r["audio"]]) in known:  # unchanged or a real reading: no new hearing
                continue
            known.add(bare(texts[r["audio"]]))
            if common.value_lost(r, texts, catalog):
                continue
            out.append(common.conv("T2b", r, texts))
            got += 1
        failed += got < per
    return out, failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sample", type=int, default=1)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()
    d = common.load()
    readings = {n: json.loads((common.READINGS / f"{n}_train.json").read_text(encoding="utf-8")) for n in SINGLES}
    for name, rows in common.pools(d):
        noise = common.real_noise_from(rows, d["avg5"])
        out, failed = build(rows, noise, readings, d["avg5"], args.per_sample, args.seed, d["catalog"])
        common.write(f"t2b_{name}", out)
        print(json.dumps({"pool": name, "k_dist": [round(p, 3) for p in noise.k_dist],
                          "samples_short_of_variants": failed}))


if __name__ == "__main__":
    main()
