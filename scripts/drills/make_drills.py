"""Builds the time/number drills of one fold and runs the data gate on them (improve_plan.md step 4).

    python scripts/drills/make_drills.py --holdout splits --out src/lifemate/drills_fold0.jsonl

Context times are drawn from the pool (the training rows outside the holdout). The holdout's texts (gold and readings)
are used only to reject a drill that shares an 8-word sequence with one of them. The file is written only when every
gate passes (scripts/drills/drill_gate.py); the gate report goes next to it.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import drill_gate  # noqa: E402
from lifemate import drills  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="splits")
    ap.add_argument("--readings", default=str(REPO / "outputs/readings/avg5_train.json"))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mix", default=None, help="error-category weights (scripts/drills/error_mix.py); default: FAMILIES")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mix:
        drills.set_mix(json.loads(Path(args.mix).read_text())["weights"])
    pool, hold, catalog = drill_gate.load_pool(args.holdout, args.readings)
    avoid = set()
    for r in hold:
        for t in (r["audio_text"], r["reading"]):
            avoid |= drills.ngrams(t)
    packs = drills.generate([r["context"]["current_time"] for r in pool], args.n, args.seed, avoid=avoid)
    report = drill_gate.run_gate(packs, pool, hold, catalog)
    report["made by"] = {"holdout": args.holdout, "n": args.n, "seed": args.seed, "mix": args.mix,
                         "families": {f.__name__: round(w, 3) for f, w in drills.FAMILIES}}
    out = Path(args.out)
    gate_file = out.with_name(out.stem + "_gate.json")
    gate_file.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"gates": report["gates"], "decision": report["decision"], "packs": len(packs)}, ensure_ascii=False))
    if report["decision"] != "PASS":
        raise SystemExit(f"gate failed, nothing written; report in {gate_file}")
    with open(out, "w", encoding="utf-8") as f:
        for p in packs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(packs)} packs to {out}")


if __name__ == "__main__":
    main()
