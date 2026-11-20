"""Compare ASR checkpoints on the same random train turns, one process per GPU.

    python -m lifemate.asr_bench --out outputs/asr_bench --clips 500 --models a b c d

Only train.jsonl and the audio are read.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


def pick_turns(items: list, clips: int, seed: int) -> list:
    from .asr_eval import turns_of
    turns = turns_of(items)
    random.Random(seed).shuffle(turns)
    return turns[:clips]


def run_one(name: str, turns: list, audio_root: Path, out: Path, batch_size: int) -> dict:
    import torch
    from . import asr
    from .asr_eval import score_turn, summarise

    start = time.monotonic()
    backend = asr.load(name)
    load_s = time.monotonic() - start
    model = getattr(backend, "model", None)
    params = sum(p.numel() for p in model.parameters()) if model is not None else None
    start = time.monotonic()
    hyps = backend.transcribe([str(audio_root / t["audio"]) for t in turns], batch_size)
    run_s = time.monotonic() - start
    rows = [{**t, "hyp": h, **score_turn(t["ref"], h)} for t, h in zip(turns, hyps)]
    summary = summarise(rows)
    result = {"model": name, "parameters": params, "load_s": round(load_s, 1), "run_s": round(run_s, 1),
              "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), **summary}
    stem = name.replace("/", "__")
    (out / f"{stem}.json").write_text(json.dumps({**result, "rows": rows}, ensure_ascii=False, indent=0), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return result


def worker(args) -> None:
    from . import data
    root = data.find_root("train")
    turns = pick_turns(data.load_split("train", root), args.clips, args.seed)
    out = Path(args.out)
    for name in args.models:
        try:
            run_one(name, turns, Path(root) / "train", out, args.batch_size)
        except Exception as e:  # one broken checkpoint must not lose the others
            print(json.dumps({"model": name, "error": f"{type(e).__name__}: {e}"}), flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/asr_bench")
    ap.add_argument("--clips", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args(argv)

    if args.worker:
        worker(args)
        return
    Path(args.out).mkdir(parents=True, exist_ok=True)
    procs = []
    for gpu in range(args.gpus):
        models = args.models[gpu::args.gpus]
        if not models:
            continue
        cmd = [sys.executable, "-m", "lifemate.asr_bench", "--worker", "--out", args.out, "--clips", str(args.clips),
               "--seed", str(args.seed), "--batch-size", str(args.batch_size), "--models", *models]
        procs.append(subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}))
    codes = [p.wait() for p in procs]
    print("worker exit codes", codes, flush=True)
    rows = []
    for f in sorted(Path(args.out).glob("*.json")):
        row = json.loads(f.read_text(encoding="utf-8"))
        row.pop("rows", None)
        rows.append(row)
    print(json.dumps(sorted(rows, key=lambda r: r.get("wer_normalised_pct", 1e9)), indent=1), flush=True)


if __name__ == "__main__":
    main()
