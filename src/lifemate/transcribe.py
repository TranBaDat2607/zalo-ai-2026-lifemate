"""Transcribe every user turn of the given splits with one ASR model, one process per GPU.

    python -m lifemate.transcribe --out outputs/asr --model vinai/PhoWhisper-large --splits train public_test

Writes <out>/<split>.json = {audio path: text}, the input of infer.py and train.py. Audio only; state.json is never read.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import data


def paths_of(items: list) -> list:
    out = []
    for item in items:
        out += [t["audio"] for t in item["history"] if t["role"] == "user" and t.get("audio")]
        out.append(item["audio"])
    return sorted(set(out))


def wer_against_gold(items: list, texts: dict) -> float:
    from .asr_eval import edit_distance, normal_tokens
    errors, words = 0, 0
    for item in items:
        pairs = [(t["audio"], t["audio_text"]) for t in item["history"] if t["role"] == "user" and t.get("audio")]
        for audio, ref in pairs + [(item["audio"], item["audio_text"])]:
            if audio in texts:
                r, h = normal_tokens(ref), normal_tokens(texts[audio])
                errors += edit_distance(r, h)
                words += len(r)
    return 100 * errors / max(words, 1)


def worker(args) -> None:
    from . import asr
    from .infer import resolve_adapter

    # "auto:a+auto:b": each adapter is looked up under /kaggle/input, then their deltas are averaged (asr.Whisper)
    adapter = "+".join(resolve_adapter(a) for a in args.adapter.split("+")) if args.adapter else None
    backend = asr.load_with_adapter(args.model, adapter)
    for split in args.splits:
        root = data.find_root(split)
        items = data.load_split(split, root)
        audio = paths_of(items)[args.shard::args.shards]
        start = time.monotonic()
        texts = backend.transcribe([str(Path(root) / split / p) for p in audio], args.batch_size)
        name = f"{split}.json" if args.shards == 1 else f"{split}.part{args.shard}.json"
        (Path(args.out) / name).write_text(json.dumps(dict(zip(audio, texts)), ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"{split} shard {args.shard}: {len(audio)} clips in {time.monotonic() - start:.0f} s", flush=True)
        if split == "train":
            print(f"train WER {wer_against_gold(items, dict(zip(audio, texts))):.2f}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/asr")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "public_test"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args(argv)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.shard >= 0:
        worker(args)
        return
    procs = []
    for gpu in range(args.gpus):
        cmd = [sys.executable, "-m", "lifemate.transcribe", "--out", args.out, "--model", args.model, "--splits", *args.splits,
               "--batch-size", str(args.batch_size), "--shard", str(gpu), "--shards", str(args.gpus)]
        if args.adapter:
            cmd += ["--adapter", args.adapter]
        procs.append(subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}))
    codes = [p.wait() for p in procs]
    print("worker exit codes", codes, flush=True)
    if any(codes):
        raise SystemExit(1)
    for split in args.splits:
        merged = {}
        for part in sorted(Path(args.out).glob(f"{split}.part*.json")):
            merged.update(json.loads(part.read_text(encoding="utf-8")))
            part.unlink()
        (Path(args.out) / f"{split}.json").write_text(json.dumps(merged, ensure_ascii=False, indent=0), encoding="utf-8")
        print(split, len(merged), "clips", flush=True)


if __name__ == "__main__":
    main()
