"""Plain greedy decoding against --lookup on the same validation samples, and the rule fixed in log.md (8 Nov 2026):

    --lookup is used later only if the calls are identical (up to 3 differing samples whose summed score is not
    lower) and the mean seconds per sample fall by at least 15 %.

    python scripts/lookup_compare.py outputs/lookup-check [--reference outputs/harm-check5/avg5_eval]

--reference also checks that the plain run reproduces an earlier run of the same model on the same samples (drift).
"""
import argparse
import json
from pathlib import Path


def load(folder: Path) -> dict:
    rows = {}
    for f in sorted(folder.glob("detail.*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--reference", default=None)
    args = ap.parse_args()
    plain, fast = load(Path(args.run) / "plain"), load(Path(args.run) / "lookup")
    ids = sorted(set(plain) & set(fast))
    differ = [i for i in ids if plain[i]["pred"] != fast[i]["pred"]]
    turns_differ = [i for i in ids if plain[i]["turns"] != fast[i]["turns"]]
    score_change = sum(fast[i]["score"] - plain[i]["score"] for i in differ)
    mean = lambda rows: sum(rows[i]["seconds"] for i in ids) / len(ids)  # noqa: E731
    gen = lambda rows: sum(rows[i].get("gen_tokens", 0) for i in ids)  # noqa: E731
    last = max(ids, key=lambda i: fast[i].get("lookup", {}).get("rounds", 0))
    stats = fast[last].get("lookup", {})
    res = {
        "samples": len(ids), "plain rows": len(plain), "lookup rows": len(fast),
        "differing calls": len(differ), "differing ids": differ[:20], "differing generated text": len(turns_differ),
        "summed score change on differing samples": round(score_change, 4),
        "score plain": round(100 * sum(plain[i]["score"] for i in ids) / len(ids), 2),
        "score lookup": round(100 * sum(fast[i]["score"] for i in ids) / len(ids), 2),
        "mean seconds plain": round(mean(plain), 2), "mean seconds lookup": round(mean(fast), 2),
        "time saved": round(1 - mean(fast) / mean(plain), 3),
        "max seconds plain": max(plain[i]["seconds"] for i in ids), "max seconds lookup": max(fast[i]["seconds"] for i in ids),
        "generated tokens plain": gen(plain), "generated tokens lookup": gen(fast),
        "lookup counts": stats,
        "tokens per pass": round(gen(fast) / max(1, stats.get("rounds", 0)), 2) if stats else None,
    }
    if args.reference:
        ref = load(Path(args.reference))
        common = [i for i in ids if i in ref]
        res["drift: plain = reference calls"] = f"{sum(plain[i]['pred'] == ref[i]['pred'] for i in common)}/{len(common)}"
    ok_calls = len(differ) == 0 or (len(differ) <= 3 and score_change >= 0)
    res["rule"] = {"calls identical (<= 3 diffs, summed score not lower)": ok_calls,
                   "mean seconds down at least 15 %": res["time saved"] >= 0.15}
    res["decision"] = "USE" if all(res["rule"].values()) else "DO NOT USE"
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
