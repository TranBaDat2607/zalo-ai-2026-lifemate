"""Stage 4 bench report (lifemate.view_bench output): the speech step per request today (the averaged model on every
clip) and with the readings view (plus the five single readings of the latest request), and whether the answer-time
readings equal the saved ones the agent trains and validates on.

    python3 scripts/hearing_analysis/bench_report.py outputs/view-bench4/view_bench [outputs/view-bench3-r2/view_bench]

With a second folder (an earlier bench of the same requests), its single readings are compared with these: plain
batched decoding (view-bench3) against decoding with the main reading as draft (lookup_decode).
"""
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lifemate import view  # noqa: E402

SAVED = ["avg5_public_test.json", "medium_public_test.json", "medium10_public_test.json", "b_public_test.json",
         "s2027_public_test.json", "s2028_public_test.json"]
NAMES = ["exp7", "m10", "B", "s2027", "s2028"]
A30 = 2.5  # the A30's speed over a T4 assumed in ASR_IMPROVEMENT_PLAN.md (decoding small batches may gain less)


def merged(folder: Path, stem: str) -> dict:
    out = {}
    for f in sorted(folder.glob(f"{stem}.*.json")):
        out.update(json.loads(f.read_text(encoding="utf-8")))
    return out


def spread(values: list) -> str:
    s = sorted(values)
    return f"mean {statistics.mean(s):.2f}  p95 {s[int(.95 * (len(s) - 1))]:.2f}  max {s[-1]:.2f}"


folder = Path(sys.argv[1])
rows = [json.loads(line) for f in sorted(folder.glob("timing.*.jsonl")) for line in f.read_text().splitlines() if line]
for f in sorted(folder.glob("info.*.json")):
    print(f.name, f.read_text())
print(f"requests timed: {len(rows)} (T4 seconds)")
print(f"  today, averaged model on every clip:  {spread([r['main_s'] for r in rows])}")
print(f"  added, five single readings (latest): {spread([r['singles_s'] for r in rows])}")
print(f"  with the view, in all:                {spread([r['main_s'] + r['singles_s'] for r in rows])}")
total = sorted(r["main_s"] + r["singles_s"] for r in rows)
print(f"  on an A30 at {A30}x: mean {statistics.mean(total) / A30:.2f} s, max {total[-1] / A30:.2f} s per request")
main, singles = merged(folder, "main"), merged(folder, "singles")
saved = [json.load(open(REPO / "outputs/readings" / s)) for s in SAVED]
print(f"main reading = saved avg5 reading: {sum(main[a] == saved[0][a] for a in main)}/{len(main)} clips")
for k, name in enumerate(NAMES):
    print(f"  {name:<6} alone = saved {name} reading: {sum(singles[a][k] == saved[k + 1][a] for a in singles)}/{len(singles)} latest clips")
now = {a: view.build(main[a], singles[a]) for a in singles}
then = {a: view.build(saved[0][a], [saved[k][a] for k in range(1, 6)]) for a in singles}
print(f"view of the latest request = view from saved readings: {sum(now[a] == then[a] for a in singles)}/{len(singles)}")
print(f"braces in the latest request: answer time {sum('{' in v for v in now.values())}, saved {sum('{' in v for v in then.values())}")
if any(r.get("passes") for r in rows):
    passes = [r["passes"] / 5 for r in rows if r.get("passes")]
    tokens = [r["tokens"] for r in rows if r.get("passes")]
    print(f"decoder passes per single reading: mean {statistics.mean(passes):.1f} for {statistics.mean(tokens):.1f} tokens "
          f"(plain decoding: one pass per token)")
if len(sys.argv) > 2:
    plain = merged(Path(sys.argv[2]), "singles")
    both = [a for a in singles if a in plain]
    same = sum(singles[a][k] == plain[a][k] for a in both for k in range(5))
    print(f"single readings, this bench = earlier bench: {same}/{5 * len(both)} "
          f"({sum(singles[a] == plain[a] for a in both)}/{len(both)} latest clips with all five equal)")
    for a in [a for a in both if singles[a] != plain[a]][:3]:
        k = next(k for k in range(5) if singles[a][k] != plain[a][k])
        print(f"  {a} {NAMES[k]}\n    this    {singles[a][k]}\n    earlier {plain[a][k]}")
diff = [a for a in singles if now[a] != then[a]][:5]
for a in diff:
    print(f"  {a}\n    now   {now[a]}\n    saved {then[a]}")
