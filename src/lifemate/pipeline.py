"""One kernel, two T4s: fine-tune and validate on GPU 0, fine-tune on everything and run public test on GPU 1.

    python -m lifemate.pipeline --out /kaggle/working --asr-dir PATH --epochs 2

Both runs write under --out, so a single kernel gives the validation score and the submission file together.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import data


def asr_file(asr_dir: str, split: str) -> str:
    if asr_dir == "gold":
        return "gold"
    if asr_dir in (None, "auto"):
        for d in data.walk_input(f"{split}.json"):
            return str(d / f"{split}.json")
        raise SystemExit(f"no {split}.json under /kaggle/input")
    return str(Path(asr_dir) / f"{split}.json")


def run(cmds: list, gpu: int, log: Path) -> subprocess.Popen:
    """Commands run one after the other on one GPU; the first failure stops that GPU's chain."""
    script = " && ".join(" ".join(f"'{c}'" for c in cmd) for cmd in cmds)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    return subprocess.Popen(["bash", "-c", f"({script}) 2>&1 | tee -a '{log}'; exit ${{PIPESTATUS[0]}}"], env=env)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/pipeline")
    ap.add_argument("--asr-dir", default="auto")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--denoise", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke runs: samples per inference")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke runs: optimizer steps")
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--skip-fold", action="store_true")
    ap.add_argument("--baseline", action="store_true", help="run the untuned model on the holdout on GPU 1 instead of the full model")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = json.loads((Path(__file__).parent / "splits.json").read_text(encoding="utf-8"))
    holdout = out / "holdout_ids.json"
    holdout.write_text(json.dumps(splits["random"] + splits["apps"]))
    train_asr = asr_file(args.asr_dir, "train")
    public_asr = None if args.asr_dir == "gold" else asr_file(args.asr_dir, "public_test")
    py = sys.executable
    common = ["--epochs", str(args.epochs), "--lr", str(args.lr), "--lora-r", str(args.lora_r)]
    if train_asr != "gold":
        common += ["--transcripts", train_asr]
    if args.max_steps:
        common += ["--max-steps", str(args.max_steps)]
    limit = ["--limit", str(args.limit)] if args.limit else []
    if args.denoise:
        common.append("--denoise")
        limit.append("--denoise")

    procs = []
    if not args.skip_fold:
        procs.append(run([
            [py, "-m", "lifemate.train", "--out", str(out / "fold"), "--holdout-file", str(holdout), *common],
            [py, "-m", "lifemate.infer", "--split", "train", "--out", str(out / "fold_eval"), "--adapter", str(out / "fold" / "adapter"),
             "--transcripts", train_asr, "--ids-file", str(holdout), "--shard", "0", "--shards", "1", *limit],
        ], 0, out / "fold.log"))
    if args.baseline:
        procs.append(run([
            [py, "-m", "lifemate.infer", "--split", "train", "--out", str(out / "base_eval"), "--transcripts", train_asr,
             "--ids-file", str(holdout), "--shard", "0", "--shards", "1", *limit],
        ], 1, out / "base.log"))
    elif not args.skip_full:
        procs.append(run([
            [py, "-m", "lifemate.train", "--out", str(out / "full"), *common],
            [py, "-m", "lifemate.infer", "--split", "public_test", "--out", str(out / "public"), "--adapter", str(out / "full" / "adapter"),
             "--transcripts", public_asr, "--shard", "0", "--shards", "1", *limit],
        ], 1, out / "full.log"))
    codes = [p.wait() for p in procs]
    print("exit codes", codes, flush=True)
    if any(codes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
