#!/usr/bin/env python3
"""Break a validation run down by sample family and error type.

    python scripts/analyze.py outputs/NAME/fold_eval [--show N] [--bucket C] [--only-failed]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from lifemate import scorer  # noqa: E402

SPLITS = json.loads((REPO / "src" / "lifemate" / "splits.json").read_text(encoding="utf-8"))


def bucket(sample_id: str) -> str:
    n = int(sample_id.split("_")[1])
    return "A" if n < 400 else ("B" if n < 900 else "C")


def load_catalog() -> dict:
    cat = json.loads((REPO / "train+public_test" / "train" / "train_tools_catalog.json").read_text(encoding="utf-8"))
    return {t["name"]: t for a in cat["apps"] for t in a["tools"]}


def diff_calls(pred: list, gold: list, catalog: dict) -> tuple:
    left = list(gold)
    extra = []
    for call in pred:
        for i, g in enumerate(left):
            if scorer.calls_match(call, g, catalog):
                del left[i]
                break
        else:
            extra.append(call)
    return extra, left


def fmt(call: dict) -> str:
    return f"{call['name']}{json.dumps(call.get('arguments', {}), ensure_ascii=False)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--bucket")
    ap.add_argument("--group", choices=["random", "apps"])
    ap.add_argument("--max-score", type=float, default=0.999)
    args = ap.parse_args()

    catalog = load_catalog()
    rows = []
    for f in sorted(Path(args.run).glob("detail.*.jsonl")):
        rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    print(f"{len(rows)} samples, mean score {100 * sum(r['score'] for r in rows) / max(len(rows), 1):.2f}")
    groups = collections.defaultdict(list)
    for r in rows:
        groups["bucket " + bucket(r["id"])].append(r)
        groups["group random" if r["id"] in SPLITS["random"] else "group held-out apps"].append(r)
        if not r["gold"]:
            groups["gold empty"].append(r)
        elif any(c["name"] == "sys.ask_missing_info" for c in r["gold"]):
            groups["gold asks"].append(r)
        else:
            groups["gold acts"].append(r)
    for name, part in sorted(groups.items()):
        ok = sum(r["hash_ok"] for r in part)
        print(f"  {name:22s} n={len(part):4d} score={100 * sum(r['score'] for r in part) / len(part):6.2f} "
              f"hash_ok={100 * ok / len(part):6.2f} perfect={100 * sum(r['score'] > 0.999 for r in part) / len(part):6.2f}")
    extra_tools, missing_tools = collections.Counter(), collections.Counter()
    for r in rows:
        extra, missing = diff_calls(r["pred"], r["gold"], catalog)
        extra_tools.update(c["name"] for c in extra)
        missing_tools.update(c["name"] for c in missing)
    print("extra calls  :", extra_tools.most_common(25))
    print("missing calls:", missing_tools.most_common(25))
    seconds = sorted(r["seconds"] for r in rows)
    print(f"seconds mean {sum(seconds) / len(seconds):.1f} p50 {seconds[len(seconds) // 2]} p95 {seconds[int(len(seconds) * .95)]} max {seconds[-1]}")
    print("stops:", collections.Counter(r["stop"] for r in rows).most_common())

    shown = 0
    for r in sorted(rows, key=lambda x: x["id"]):
        if shown >= args.show:
            break
        if r["score"] > args.max_score:
            continue
        if args.bucket and bucket(r["id"]) != args.bucket:
            continue
        if args.group and (r["id"] in SPLITS["random"]) != (args.group == "random"):
            continue
        shown += 1
        extra, missing = diff_calls(r["pred"], r["gold"], catalog)
        print(f"--- {r['id']} score={r['score']:.2f} hash_ok={r['hash_ok']} stop={r['stop']}\n  REQ: {r['request']}")
        for c in missing:
            print("  MISS:", fmt(c))
        for c in extra:
            print("  XTRA:", fmt(c))


if __name__ == "__main__":
    main()
