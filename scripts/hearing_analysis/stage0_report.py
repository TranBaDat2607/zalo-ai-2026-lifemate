"""Stage 0 of ASR_IMPROVEMENT_PLAN.md: 0012 (m1+m2 soup, 71.57) on the 386 validation samples with the said text in
place of the heard text. Rule fixed before the run (log.md, 10 Nov 2026): go on to Stage 2 only if the every-turn
oracle (a) scores >= 76.57 (0012 + 5.0).

    python scripts/hearing_analysis/stage0_report.py outputs/stage0-oracle/oracle_all \
        [outputs/stage0-latest/oracle_latest]

For each oracle run against 0012 (`outputs/notes-aux/aux_eval`), paired over the same samples:
- mean, paired difference with a bootstrap 95 % interval, wins/losses, on all 386, both halves and batches A/B/C;
- the gain split by what 0012 heard in the latest request (exact / only non-key words wrong / a key word wrong) and
  by whether an earlier user turn was misheard;
- the restated request with clean input: how often it still differs from what was said, and key words it breaks;
- the largest flips both ways.
Key word = a token of a gold argument value, a number word or a time word (as keyword_pool.py).
"""
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "scripts" / "notes"))
from compare_notes import SPLITS, batch, load, paired  # noqa: E402
from first_error import restated  # noqa: E402
from lifemate import data  # noqa: E402
from keyword_pool_rules import NUMW, TIME, spans, toks, value_tokens  # noqa: E402

BASELINE, BAR = "outputs/notes-aux/aux_eval", 76.57


def key_words(row: dict) -> set:
    out = set()
    for c in row["gold"]:
        out |= value_tokens(c.get("arguments") or {})
    return out


def key_breaks(gold: list, text: list, argw: set) -> int:
    return sum(1 for g, _ in spans(gold, text) for w in g if w in argw or w in NUMW or w in TIME)


def main() -> None:
    runs = sys.argv[1:]
    items = {s["id"]: s for s in data.load_split("train", data.find_root("train"))}
    avg5 = json.load(open(REPO / "outputs/asr-avg5/avg5_train.json"))
    base = load(str(REPO / BASELINE))
    ids = sorted(base)
    assert len(ids) == 386, len(ids)
    rng = random.Random(2026)

    heard, history = {}, {}
    for sid in ids:
        it, r = items[sid], base[sid]
        G, H = toks(it["audio_text"]), toks(r["request"])
        argw = key_words(r)
        if not spans(G, H):
            heard[sid] = "latest heard exactly"
        elif key_breaks(G, H, argw):
            heard[sid] = "latest: a key word misheard"
        else:
            heard[sid] = "latest: only other words misheard"
        turns = [h for h in it.get("history") or [] if h.get("role") == "user" and h.get("audio")]
        if not turns:
            history[sid] = "no earlier turns"
        elif any(spans(toks(h["audio_text"]), toks(avg5.get(h["audio"], ""))) for h in turns):
            history[sid] = "an earlier turn misheard"
        else:
            history[sid] = "earlier turns heard exactly"

    report = {}
    for run in runs:
        rows = load(str(REPO / run) if not Path(run).is_absolute() else run)
        missing = [i for i in ids if i not in rows]
        assert not missing, f"{run}: {len(missing)} samples missing, e.g. {missing[:3]}"
        name = Path(run).name
        print(f"\n=== {name} vs 0012 ({BASELINE}) ===")
        out = {"all": paired(ids, rows, base, 20000, rng)}
        for half in ("random", "apps"):
            out[half] = paired([i for i in ids if i in set(SPLITS[half])], rows, base, 20000, rng)
        for b in "ABC":
            out[f"batch {b}"] = paired([i for i in ids if batch(i) == b], rows, base, 20000, rng)
        for k, v in out.items():
            print(f"  {k:<10} n={v['n']:>3}  oracle {v['notes']:6.2f}  0012 {v['baseline']:6.2f}  diff {v['diff']:+6.2f} "
                  f"[{v['low']:+.2f}, {v['high']:+.2f}]  wins {v['wins']:>3} losses {v['losses']:>3}")
        groups = {}
        for label, by in (("by the latest request", heard), ("by earlier turns", history)):
            print(f"  {label}:")
            for g in sorted(set(by.values())):
                gi = [i for i in ids if by[i] == g]
                v = paired(gi, rows, base, 5000, rng)
                pts = round(100 * sum(rows[i]["score"] - base[i]["score"] for i in gi) / len(ids), 2)
                groups[g] = {**v, "points_of_386": pts}
                print(f"    {g:<36} n={v['n']:>3}  0012 {v['baseline']:6.2f} -> oracle {v['notes']:6.2f}  "
                      f"= {pts:+.2f} points of the 386 mean  (wins {v['wins']}, losses {v['losses']})")
        # the restated request with clean input
        exact = broken = broken_samples = 0
        for sid in ids:
            G, R = toks(items[sid]["audio_text"]), toks(restated(rows[sid]))
            exact += not spans(G, R)
            k = key_breaks(G, R, key_words(rows[sid]))
            broken += k
            broken_samples += k > 0
        print(f"  restated request with the said text as input: exact {exact}/386, key words changed {broken} "
              f"in {broken_samples} samples")
        diffs = sorted(ids, key=lambda i: rows[i]["score"] - base[i]["score"])
        flips = {"up": [(i, round(base[i]["score"], 2), round(rows[i]["score"], 2)) for i in diffs[::-1][:8]],
                 "down": [(i, round(base[i]["score"], 2), round(rows[i]["score"], 2)) for i in diffs[:8]]}
        print(f"  largest gains {flips['up']}\n  largest losses {flips['down']}")
        report[name] = {"paired": out, "groups": groups, "restated_exact": exact, "restated_key_words_changed": broken,
                        "restated_key_break_samples": broken_samples, "flips": flips,
                        "repairs": sum(rows[i].get("repairs", 0) for i in ids),
                        "abnormal_stops": sum(rows[i].get("stop") not in ("done", "asked") for i in ids)}
    if runs:
        a = report[Path(runs[0]).name]["paired"]["all"]["notes"]
        verdict = "CONTINUE to Stage 2" if a >= BAR else "STOP: hearing is not the bottleneck"
        print(f"\nRule: every-turn oracle {a:.2f} vs bar {BAR} -> {verdict}")
        report["rule"] = {"oracle_all": a, "bar": BAR, "verdict": verdict}
        out_path = REPO / Path(runs[0]).parent / "stage0_report.json"
        out_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"written {out_path}")


if __name__ == "__main__":
    main()
