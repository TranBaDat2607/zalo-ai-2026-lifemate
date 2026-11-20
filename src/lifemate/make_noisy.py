"""Write transcripts of the train split with simulated ASR errors, for LLM work before real fine-tuned transcripts exist.

    python -m lifemate.make_noisy --out outputs/noised --scale 0.5
"""
import argparse
import json
import random
from pathlib import Path

from . import data
from .noise import ConfusionNoise
from .transcribe import wer_against_gold


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.5, help="share of the benchmark's untuned error rate to apply")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)
    root = data.find_root("train")
    items = data.load_split("train", root)
    noise = ConfusionNoise.packaged(args.scale)
    rng = random.Random(args.seed)
    texts = {}
    for item in items:
        for t in item["history"]:
            if t["role"] == "user" and t.get("audio"):
                texts[t["audio"]] = noise.apply(t["audio_text"], rng)
        texts[item["audio"]] = noise.apply(item["audio_text"], rng)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.json").write_text(json.dumps(texts, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(texts)} clips, simulated WER {wer_against_gold(items, texts):.2f}", flush=True)


if __name__ == "__main__":
    main()
