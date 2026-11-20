"""Speech weight-space edit search: each arm's out-of-fold transcripts against the unedited arm (log.md, 11 Nov).

    python scripts/wsp/analyze_asr.py outputs/wsp-asr1 --base e_base [--avg5 outputs/asr-avg5/avg5_train.json]

Each subfolder of RUN with fold*_none.json is an arm (all 5,036 train clips, decoded by the fold's five averaged
adapters with the arm's edit). Halves from scripts/wsp/asr_split.json: D1 discover, D2 confirm (samples outside the
386), test = the 386. Bootstrap resamples whole samples (asr_compare). Rules, fixed before any result:
  A candidate: latest-request WER change on D1 <= -0.10 with the 95 % upper end < 0.
  B confirmed: candidate, D2 latest change <= 0, D1+D2 latest upper end < 0, D1+D2 all-turn change <= 0,
    D1+D2 latest critical_ok not lower, and the latest change negative in at least 2 of the 3 folds (D1+D2).
  D test (only for the one arm chosen afterwards): latest change on the 386 <= 0 and critical_ok there not lower.
"""
import argparse
import io
import json
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lifemate import data  # noqa: E402
from lifemate.asr_compare import load_set, report  # noqa: E402
from lifemate.asr_eval import turns_of  # noqa: E402
from lifemate.asr_train import sample_folds  # noqa: E402

SPLIT = json.loads((REPO / "scripts" / "wsp" / "asr_split.json").read_text())


def row(name, turns, a, b, seed=2026):
    with redirect_stdout(io.StringIO()):
        return report(name, turns, a, b, 2000, random.Random(seed))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--base", default="e_base")
    ap.add_argument("--avg5", default=None, help="today's saved five-copy transcripts: drift check of the base arm")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    items = data.load_split("train", data.find_root("train"))
    folds = sample_folds(items, 3, 2026)
    run = Path(args.run)
    arms = {d.name: load_set(f"{d}:none") for d in sorted(run.iterdir()) if d.is_dir() and list(d.glob("fold*_none.json"))}
    base = arms[args.base]
    turns = [t for t in turns_of(items) if t["audio"] in base]
    D1, D2, TEST = set(SPLIT["discover"]), set(SPLIT["confirm"]), set(SPLIT["test"])
    report_all = {}
    if args.avg5:
        saved = json.loads(Path(args.avg5).read_text(encoding="utf-8"))
        same = sum(base[a] == saved.get(a) for a in base)
        report_all["_drift"] = {"clips": len(base), "identical_to_avg5": same}
        print(f"drift check: base arm identical to the saved five-copy text on {same}/{len(base)} clips")
    print(f"{'arm':14} {'D1 lat':>7} {'[95%]':>15} {'D2 lat':>7} {'D12 lat':>8} {'[95%]':>15} {'D12 all':>8} "
          f"{'crit':>9} {'folds-':>6} {'test lat':>9} {'crit':>9}")
    for name, arm in arms.items():
        if name == args.base:
            continue
        T = [t for t in turns if t["audio"] in arm]  # a deadline may cut an arm's last fold: compare its clips only
        latest = lambda ids: [t for t in T if t["kind"] == "latest" and t["id"] in ids]  # noqa: E731
        r = {"D1": row("D1 latest", latest(D1), base, arm), "D2": row("D2 latest", latest(D2), base, arm),
             "D12": row("D1+D2 latest", latest(D1 | D2), base, arm),
             "D12_all": row("D1+D2 all turns", [t for t in T if t["id"] in D1 | D2], base, arm),
             "test": row("386 latest", latest(TEST), base, arm)}
        r["clips"] = len(T)
        r["folds"] = [row(f"fold {k}", part, base, arm)["change"] if part else None
                      for k in range(3) for part in [[t for t in latest(D1 | D2) if folds[t["id"]] == k]]]
        cand = r["D1"]["change"] <= -0.10 and r["D1"]["ci95"][1] < 0
        conf = bool(cand and r["D2"]["change"] <= 0 and r["D12"]["ci95"][1] < 0 and r["D12_all"]["change"] <= 0
                    and r["D12"]["critical_ok_b"] >= r["D12"]["critical_ok_a"] and sum(c is not None and c < 0 for c in r["folds"]) >= 2)
        r["candidate (A)"], r["confirmed (B)"] = cand, conf
        r["test (D)"] = r["test"]["change"] <= 0 and r["test"]["critical_ok_b"] >= r["test"]["critical_ok_a"]
        report_all[name] = r
        ci = lambda x: f"[{x['ci95'][0]:+.2f},{x['ci95'][1]:+.2f}]"  # noqa: E731
        print(f"{name:14} {r['D1']['change']:+7.2f} {ci(r['D1']):>15} {r['D2']['change']:+7.2f} {r['D12']['change']:+8.2f} "
              f"{ci(r['D12']):>15} {r['D12_all']['change']:+8.2f} {r['D12']['critical_ok_a']:4.1f}>{r['D12']['critical_ok_b']:4.1f} "
              f"{sum(c is not None and c < 0 for c in r['folds']):>6} {r['test']['change']:+9.2f} {r['test']['critical_ok_a']:4.1f}>{r['test']['critical_ok_b']:4.1f}"
              + (f"  ({len(T)}/{len(turns)} clips)" if len(T) < len(turns) else "")
              + ("  CONFIRMED" if conf else ("  candidate" if cand else "")))
    latest = lambda ids: [t for t in turns if t["kind"] == "latest" and t["id"] in ids]  # noqa: E731
    print(f"\nbase latest WER on D1+D2: {row('b', latest(D1 | D2), base, base)['wer_a']}  "
          f"(test 386: {row('t', latest(TEST), base, base)['wer_a']})")
    if args.out:
        Path(args.out).write_text(json.dumps(report_all, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
