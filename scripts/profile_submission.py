#!/usr/bin/env python3
"""Check a submission's shape before spending a daily submission on it. Uses no labels.

    python scripts/profile_submission.py outputs/NAME/public/submission.jsonl

Compares with train gold: the share of empty answers (public test is about 6 % from the empty baseline), the share that ask for
missing information, calls per sample, tools outside the installed apps, and whether the local replay reproduces the state hash.
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("LIFEMATE_DATA", REPO / "train+public_test"))  # the folder that holds <split>/


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--split", default="public_test")
    args = ap.parse_args()
    rows = [json.loads(x) for x in Path(args.submission).read_text(encoding="utf-8").splitlines() if x.strip()]
    items = {i["id"]: i for i in map(json.loads, (DATA / args.split / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines())}
    train = [json.loads(x) for x in (DATA / "train" / "train.jsonl").read_text(encoding="utf-8").splitlines()]

    print(f"{len(rows)} lines, {len(items)} samples, missing {len(set(items) - {r['id'] for r in rows})}, duplicates {len(rows) - len({r['id'] for r in rows})}")
    n = len(rows)
    calls = [r["tool_calls"] for r in rows]
    gold = [i["label"]["tool_calls"] for i in train]

    def shape(lists):
        return {"empty %": round(100 * sum(not c for c in lists) / len(lists), 1),
                "asks %": round(100 * sum(any(x["name"] == "sys.ask_missing_info" for x in c) for c in lists) / len(lists), 1),
                "calls per sample": round(sum(len(c) for c in lists) / len(lists), 2),
                "max calls": max(len(c) for c in lists)}

    print("submission", shape(calls))
    print("train gold", shape(gold))
    outside = 0
    for r in rows:
        allowed = {t for a in items[r["id"]]["context"]["installed_apps"].values() for t in a["tools"]} | {"sys.ask_missing_info"}
        outside += sum(c["name"] not in allowed for c in r["tool_calls"])
    print("calls to tools not installed:", outside)
    print("top tools:", collections.Counter(c["name"] for cs in calls for c in cs).most_common(30))
    apps = collections.Counter(c["name"].split(".")[0] for cs in calls for c in cs)
    train_apps = {c["name"].split(".")[0] for cs in gold for c in cs}
    print("apps train never saw:", {a: v for a, v in apps.most_common() if a not in train_apps})

    sys.path.insert(0, str(DATA / args.split))
    from zaic_tools import ToolEnv
    bad = 0
    for r in rows:
        env = ToolEnv(args.split, r["id"], root=str(DATA))
        for c in r["tool_calls"]:
            env.call(c["name"], c["arguments"])
        bad += env.get_state_hash() != r["state_hash"]
    print(f"state hash differs on replay: {bad} of {n}")


if __name__ == "__main__":
    main()
