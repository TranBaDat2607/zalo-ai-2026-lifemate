"""Block a jobs.py chain until another GPU's chain has written a file.

    python -m lifemate.waitfor --out IGNORED --path {out}/asr/train.json --minutes 120
"""
import argparse
import time
from pathlib import Path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--minutes", type=float, default=240)
    args = ap.parse_args(argv)
    deadline = time.monotonic() + args.minutes * 60
    while time.monotonic() < deadline:
        if Path(args.path).exists():
            time.sleep(5)  # let the writer finish
            print("found", args.path, flush=True)
            return
        time.sleep(15)
    raise SystemExit(f"{args.path} did not appear within {args.minutes} min")


if __name__ == "__main__":
    main()
