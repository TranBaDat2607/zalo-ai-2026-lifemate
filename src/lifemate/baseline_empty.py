"""A submission that calls no tool at all: checks the upload path and shows the share of out-of-scope samples.

    python -m lifemate.baseline_empty --out outputs/empty
"""
import argparse
import json
from pathlib import Path

from . import data


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/empty")
    ap.add_argument("--split", default="public_test")
    args = ap.parse_args(argv)
    root = data.find_root(args.split)
    tool_env = data.import_tool_env(args.split, root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "submission.jsonl", "w", encoding="utf-8") as fout:
        for item in data.load_split(args.split, root):
            env = tool_env(args.split, item["id"], root=root)  # exactly one per sample
            fout.write(json.dumps({"id": item["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}, ensure_ascii=False) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
