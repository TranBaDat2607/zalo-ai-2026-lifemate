#!/usr/bin/env python3
"""Join the shard outputs of an end-to-end run (solve.py --shard k --shards n) into one submission in split order.

    python scripts/merge_e2e.py outputs/w3-e2e-dry/e2e_0 outputs/w3-e2e-dry/e2e_1 --out outputs/w3-e2e-dry/public

Writes <out>/submission.jsonl and <out>/timing.jsonl, and fails if a sample is missing or appears twice, so a
submission is never made from a run that did not cover the whole split.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lifemate import data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="public_test")
    args = ap.parse_args()
    lines, timing = {}, {}
    for shard in map(Path, args.shards):
        for name, store in (("submission.jsonl", lines), ("timing.jsonl", timing)):
            for line in (shard / name).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    sample = json.loads(line)["id"]
                    if sample in store:
                        raise SystemExit(f"{sample} appears twice in {name}")
                    store[sample] = line
    order = [i["id"] for i in data.load_split(args.split, data.find_root(args.split))]
    missing = [i for i in order if i not in lines]
    extra = sorted(set(lines) - set(order))
    if missing or extra:
        raise SystemExit(f"missing {len(missing)} (first {missing[:3]}), not in the split {len(extra)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "submission.jsonl").write_text("\n".join(lines[i] for i in order) + "\n", encoding="utf-8")
    (out / "timing.jsonl").write_text("\n".join(timing[i] for i in order if i in timing) + "\n", encoding="utf-8")
    print(f"{len(order)} samples -> {out / 'submission.jsonl'}")


if __name__ == "__main__":
    main()
