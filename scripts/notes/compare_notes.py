"""Notes against the baseline on the 386 validation samples, and the keep-or-drop rule fixed in research_report.md.

    python scripts/notes/compare_notes.py outputs/<run>/notes_eval outputs/harm-check5/avg5_eval \
        [--control outputs/<run>/control_eval] [--out outputs/<run>/decision.json]

Both runs use the same five-copy transcripts (avg5_train.json). Scores are paired on the same samples: the difference
and its 95 % interval come from a bootstrap over samples (20,000 rounds). Batches are the generator batches by id
(A 1-424, B 425-918, C 919+), as in eda.md; "reweighted" weights them 24/32/44 like public test.

Time: seconds per sample as logged by infer.py on Kaggle's T4. The rule asks for under 30 s at A30 speed (T4 time / 2,
the conservative end of the 2-2.7x seen between the two) and a mean rise of at most 50 %. --control is the baseline LLM
run on 100 validation samples in the notes session; it measures how fast that session's machine was, since one T4
session can run 10 % slower than another.
"""
import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPLITS = json.loads((REPO / "src" / "lifemate" / "splits.json").read_text(encoding="utf-8"))
MIX = {"A": 0.24, "B": 0.32, "C": 0.44}
BASELINE_MEAN = 66.71


def load(run: str) -> dict:
    rows = {}
    for f in sorted(Path(run).glob("detail.*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def batch(sid: str) -> str:
    n = int(sid.split("_")[1])
    return "A" if n < 425 else ("B" if n < 919 else "C")


def paired(ids, a, b, rounds, rng) -> dict:
    diffs = [a[i]["score"] - b[i]["score"] for i in ids]
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(rounds))
    return {"n": n, "notes": round(100 * sum(a[i]["score"] for i in ids) / n, 2),
            "baseline": round(100 * sum(b[i]["score"] for i in ids) / n, 2),
            "diff": round(100 * sum(diffs) / n, 2), "low": round(100 * means[int(rounds * 0.025)], 2),
            "high": round(100 * means[int(rounds * 0.975) - 1], 2),
            "wins": sum(d > 1e-9 for d in diffs), "losses": sum(d < -1e-9 for d in diffs)}


def seconds(rows: dict, ids) -> dict:
    s = sorted(rows[i]["seconds"] for i in ids)
    n = len(s)
    return {"mean": round(sum(s) / n, 2), "median": s[n // 2], "p95": s[int(n * 0.95)], "max": s[-1],
            "over_30s_t4": sum(x > 30 for x in s), "a30_max_estimate": round(s[-1] / 2, 1),
            "gen_tokens_mean": round(sum(rows[i].get("gen_tokens", 0) for i in ids) / n, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("notes")
    ap.add_argument("baseline")
    ap.add_argument("--control", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rounds", type=int, default=20000)
    args = ap.parse_args()
    a, b = load(args.notes), load(args.baseline)
    ids = sorted(set(a) & set(b))
    assert len(ids) == 386 == len(a) == len(b), (len(ids), len(a), len(b))
    rng = random.Random(2026)
    res = {"overall": paired(ids, a, b, args.rounds, rng)}
    for k in "ABC":
        res[f"batch {k}"] = paired([i for i in ids if batch(i) == k], a, b, args.rounds, rng)
    res["random half"] = paired([i for i in ids if i in SPLITS["random"]], a, b, args.rounds, rng)
    res["held-out-app half"] = paired([i for i in ids if i not in SPLITS["random"]], a, b, args.rounds, rng)
    res["reweighted 24/32/44"] = {"notes": round(sum(MIX[k] * res[f"batch {k}"]["notes"] for k in "ABC"), 2),
                                  "baseline": round(sum(MIX[k] * res[f"batch {k}"]["baseline"] for k in "ABC"), 2)}
    res["seconds notes"], res["seconds baseline"] = seconds(a, ids), seconds(b, ids)
    ratio = res["seconds notes"]["mean"] / res["seconds baseline"]["mean"]
    res["time ratio (raw)"] = round(ratio, 3)
    if args.control:
        c = load(args.control)
        common = sorted(set(c) & set(b))
        speed = sum(c[i]["seconds"] for i in common) / sum(b[i]["seconds"] for i in common)
        same = sum(c[i]["pred"] == b[i]["pred"] for i in common)
        res["control"] = {"n": len(common), "session speed (control / baseline seconds)": round(speed, 3),
                          "identical calls": same, "time ratio (machine-adjusted)": round(ratio / speed, 3)}
    rule = {
        "validation mean above 66.71": res["overall"]["notes"] > BASELINE_MEAN,
        "random half not worse": res["random half"]["diff"] >= 0,
        "held-out-app half not worse": res["held-out-app half"]["diff"] >= 0,
        "batch C up": res["batch C"]["diff"] > 0,
        "under 30 s at A30 speed": res["seconds notes"]["a30_max_estimate"] < 30,
        "mean time rise at most 50 %": ratio <= 1.5,  # raw means, as earlier runs were compared; adjusted one is shown
    }
    res["rule"] = rule
    res["decision"] = "KEEP" if all(rule.values()) else "DROP"
    text = json.dumps(res, indent=1, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
