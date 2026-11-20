"""Stage 3 screen report (log.md, rule fixed before the run): exp7's new out-of-fold readings, from the fold adapters
continued with key-word weights and repeats, against exp7's old ones.

    python3 scripts/hearing_analysis/stage3_report.py outputs/stage3-screen/s3

Merges oof_{0,1,2}.json into s3_medium_train.json (same 5,036 clips), runs asr_compare (the rule) and keyword_gain,
and measures the brace share: of the key words the main reading (avg5) gets wrong in latest requests, the share whose
said word is offered in a brace of the view, with exp7's old and with its new reading among the alternatives.
"""
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lifemate import asr_compare, data, keyword_gain, view  # noqa: E402
from lifemate.asr_train import clips_of, key_words  # noqa: E402

OLD = REPO / "outputs/exp7/asr_medium/medium_train.json"
READ = REPO / "outputs/readings"
OTHERS = ["medium10_train.json", "b_train.json", "s2027_train.json", "s2028_train.json"]


def brace_share(main: dict, others: list, items: list) -> tuple:
    """(covered, misheard): key words of latest requests wrong in the main reading, and how many of them are offered."""
    covered = total = 0
    for it in items:
        a = it["audio"]
        keys = key_words(it)
        ref, hyp = keyword_gain.toks(it["audio_text"]), keyword_gain.toks(main[a])
        wrong = [w for op, i1, i2, _, _ in SequenceMatcher(a=ref, b=hyp, autojunk=False).get_opcodes()
                 if op in ("replace", "delete") for w in ref[i1:i2] if w in keys or w.isdigit()]
        if not wrong:
            continue
        shown = view.build(main[a], [o.get(a, "") for o in others])
        offered = set()
        for part in shown.split("{")[1:]:
            offered |= set(keyword_gain.toks(part.split("}")[0].replace("|", " ")))
        total += len(wrong)
        covered += sum(w in offered for w in wrong)
    return covered, total


def main() -> None:
    run = Path(sys.argv[1])
    new = {}
    for f in sorted(run.glob("oof_*.json")):
        new.update(json.loads(f.read_text(encoding="utf-8")))
    old = json.loads(OLD.read_text(encoding="utf-8"))
    missing = set(old) - set(new)
    print(f"new readings: {len(new)} clips from {len(list(run.glob('oof_*.json')))} folds; missing {len(missing)}")
    merged = run / "s3_medium_train.json"
    merged.write_text(json.dumps(new, ensure_ascii=False, indent=0), encoding="utf-8")
    rows = {r["slice"]: r for r in asr_compare.main([str(OLD), str(merged), "--rounds", "2000",
                                                    "--out", str(run / "compare_vs_exp7.json")])}
    latest, earlier = rows["latest requests"], rows["earlier turns"]
    checks = {
        "1. latest key terms right +2.0 or more": latest["critical_ok_b"] - latest["critical_ok_a"] >= 2.0,
        "2. latest WER interval not above 0": latest["ci95"][0] <= 0,
        "3. earlier turns WER +0.3 at most": earlier["change"] <= 0.3,
    }
    print(f"\nlatest key terms right {latest['critical_ok_a']} -> {latest['critical_ok_b']}; WER {latest['wer_a']} -> "
          f"{latest['wer_b']} ({latest['change']:+.2f}, 95 % {latest['ci95']}); earlier turns WER {earlier['change']:+.2f}")
    for name, ok in checks.items():
        print(f"  {name}: {'yes' if ok else 'NO'}")
    print("screen:", "PASS" if all(checks.values()) else "FAIL")
    print("\nkeyword_gain on the 386 validation requests:")
    keyword_gain.main([str(OLD), str(merged)])
    items = data.load_split("train", data.find_root("train"))
    avg5 = json.load(open(READ / "avg5_train.json"))
    rest = [json.load(open(READ / f)) for f in OTHERS]
    for name, exp7 in (("old exp7", old), ("new exp7", new)):
        c, t = brace_share(avg5, [exp7] + rest, items)
        print(f"brace share with {name}: {c}/{t} = {100 * c / max(t, 1):.1f} % of the main reading's misheard key words offered")


if __name__ == "__main__":
    main()
