"""Step 1c check (improve_plan.md; rule in log.md, fixed before the run): do the averaged speech model's runner-up
hypotheses offer the right key word where today's braces do not?

    LIFEMATE_DATA=$PWD/train+public_test/train+public_test python3 scripts/hearing_analysis/nbest_check.py outputs/nbest-check/nbest

Reads ``fold{0,1,2}_nbest.json`` / ``fold{k}_timing.json`` (lifemate.asr_nbest) and today's readings in
outputs/readings. Views: main = avg5 (unchanged), others = the five single readings (today) or the five single readings
plus the beam hypotheses (new). Measured on the 1,545 latest requests (out-of-fold everywhere):
1. brace share: of the key words the main reading gets wrong, the share whose said word is offered in a brace
   (stage3_report.brace_share; today 214 of 629 = 34.0 %);
2. brace text: mean words added to the latest request by the braces (today 2.9);
3. time: mean seconds of the beam decode per latest request on a T4.
Pass: (1) >= 50 %, (2) <= 2 x today's, (3) <= 1.5 s.
"""
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "hearing_analysis"))
from lifemate import data, view  # noqa: E402
from stage3_report import brace_share  # noqa: E402

READ = REPO / "outputs/readings"
SINGLES = ["medium_train.json", "medium10_train.json", "b_train.json", "s2027_train.json", "s2028_train.json"]
BRACE_SHARE_MIN, WORDS_FACTOR_MAX, SECONDS_MAX = 50.0, 2.0, 1.5


def extra_words(main: dict, others: list, items: list) -> float:
    return statistics.mean(len(view.build(main[it["audio"]], [o.get(it["audio"], "") for o in others]).split())
                           - len(main[it["audio"]].split()) for it in items)


def main() -> None:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "outputs/nbest-check/nbest"
    nbest, seconds = {}, {}
    for k in range(3):
        nbest.update(json.loads((run / f"fold{k}_nbest.json").read_text(encoding="utf-8")))
        seconds.update(json.loads((run / f"fold{k}_timing.json").read_text(encoding="utf-8")))
    items = data.load_split("train", data.find_root("train"))
    missing = [it["audio"] for it in items if it["audio"] not in nbest]
    print(f"hypotheses for {len(nbest)} latest requests; missing {len(missing)}")
    avg5 = json.loads((READ / "avg5_train.json").read_text(encoding="utf-8"))
    singles = [json.loads((READ / f).read_text(encoding="utf-8")) for f in SINGLES]
    width = max(len(h) for h in nbest.values())
    beams = [{a: (h[i] if i < len(h) else "") for a, h in nbest.items()} for i in range(width)]
    splits = json.loads((REPO / "src/lifemate/splits.json").read_text())
    val = set(splits["random"]) | set(splits["apps"])
    report = {}
    for name, subset in (("all latest (1,545)", items), ("validation latest (386)", [it for it in items if it["id"] in val])):
        row = {}
        for arm, others in (("today", singles), ("with beams", singles + beams), ("beams only", beams)):
            c, t = brace_share(avg5, others, subset)
            row[arm] = {"offered": c, "misheard": t, "share %": round(100 * c / max(t, 1), 1),
                        "extra words mean": round(extra_words(avg5, others, subset), 2)}
        report[name] = row
    top_same = sum(nbest[a][0] == avg5[a] for a in nbest if nbest[a]) / max(1, len(nbest))
    hyps = statistics.mean(len(h) for h in nbest.values())
    secs = sorted(seconds.values())
    report["time"] = {"mean s": round(statistics.mean(secs), 3), "p95 s": secs[int(0.95 * len(secs))], "max s": secs[-1]}
    report["beams"] = {"distinct hypotheses per request": round(hyps, 2), "best beam equals the greedy main reading": round(top_same, 3)}
    full = report["all latest (1,545)"]
    checks = {
        f"1. brace share >= {BRACE_SHARE_MIN} %": full["with beams"]["share %"] >= BRACE_SHARE_MIN,
        f"2. brace text <= {WORDS_FACTOR_MAX} x today": full["with beams"]["extra words mean"]
        <= WORDS_FACTOR_MAX * full["today"]["extra words mean"],
        f"3. beam decode <= {SECONDS_MAX} s per request": report["time"]["mean s"] <= SECONDS_MAX,
    }
    report["checks"] = checks
    report["decision"] = "PASS" if all(checks.values()) else "FAIL"
    print(json.dumps(report, indent=1, ensure_ascii=False))
    (run / "nbest_check.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    rng_items = [it for it in items if it["id"] in val and it["audio"] in nbest][:6]
    print("\nexamples (validation latest requests):")
    for it in rng_items:
        a = it["audio"]
        print(f"  said : {it['audio_text']}\n  today: {view.build(avg5[a], [o.get(a, '') for o in singles])}\n"
              f"  new  : {view.build(avg5[a], [o.get(a, '') for o in singles + beams])}")


if __name__ == "__main__":
    main()
