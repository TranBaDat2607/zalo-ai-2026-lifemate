"""Stage 1 checks of the readings view (src/lifemate/view.py, ASR_IMPROVEMENT_PLAN.md) and the calibration of the
simulated views that synthetic conversations get.

    LIFEMATE_DATA=$PWD/train+public_test/train+public_test python3 scripts/hearing_analysis/view_check.py \
        [--examples 8]

- Real train views: main = the avg5 out-of-fold reading, others = the five single out-of-fold readings (exp7, m10, B,
  s2027, s2028), scored against the said text: how many clips show braces, how long the text gets, how often the main
  option / another option / no option is right inside a brace, and how often words outside braces are right.
- Public views: the all-data models' readings of the public clips (no said text there: shape only). The full models
  agree more than fold models, so fewer braces are expected.
- Simulated views (what synthetic conversations get): view.ReadingsBank built from clips outside validation, replayed
  on validation's said texts and compared with validation's real views; independent confusion noise for contrast.
"""
import argparse
import difflib
import json
import random
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lifemate import data, view  # noqa: E402
from lifemate.asr_eval import normal_tokens  # noqa: E402
from lifemate.noise import ConfusionNoise  # noqa: E402

TRAIN = {"avg5": "outputs/asr-avg5/avg5_train.json", "exp7": "outputs/exp7/asr_medium/medium_train.json",
         "m10": "outputs/w1-asr/asr_m10/medium10_train.json", "B": "outputs/b-synth/b/b_train.json",
         "s2027": "outputs/asr-s2027/s2027_train.json", "s2028": "outputs/asr-s2028/s2028_train.json"}
PUBLIC = {"avg5": "outputs/final1/avg5_asr/avg5_public_test.json", "exp7": "outputs/exp7/asr_medium/medium_public_test.json",
          "m10": "outputs/w1-asr/asr_m10/medium10_public_test.json", "B": "outputs/b-synth-s2/b/public_test.json",
          "s2027": "outputs/asr-s2027/s2027/public_test.json", "s2028": "outputs/asr-s2028/s2028/public_test.json"}
FIVE = ["exp7", "m10", "B", "s2027", "s2028"]


def key(text: str) -> str:
    return " ".join(t for w in text.split() for t in normal_tokens(w) if t not in view.IGNORED)


def anchored(match: dict, a: int, b: int, n: int, n_gold: int) -> tuple:
    """Gold token range for main tokens [a, b): from the nearest matched token before a to the nearest after b - 1."""
    p = a - 1
    while p >= 0 and p not in match:
        p -= 1
    q = b
    while q < n and q not in match:
        q += 1
    return (match[p] + 1 if p >= 0 else 0), (match[q] if q < n else n_gold)


def measure(rows: list) -> dict:
    """rows: (gold or None, main, others, latest). Counts per kind of clip (latest / earlier)."""
    st = {}
    for gold, main, others, latest in rows:
        s = st.setdefault("latest" if latest else "earlier", {
            "clips": 0, "with braces": 0, "spans": 0, "extra words": [], "spans with alternatives before caps": 0,
            "brace: main right": 0, "brace: another option right": 0, "brace: none right": 0,
            "tokens in braces": 0, "right in braces": 0, "tokens outside": 0, "right outside": 0})
        words, kept = view.choose(main, others)
        _, raw = view.spans_of(main, others)
        shown = view.build(main, others)
        s["clips"] += 1
        s["with braces"] += bool(kept)
        s["spans"] += len(kept)
        s["spans with alternatives before caps"] += sum(len(o) > 1 for _, _, o, _ in raw)
        s["extra words"].append(len(shown.split()) - len(main.split()))
        if gold is None:
            continue
        _, units = view._units(main)
        seq = [t for t, _ in units]
        gseq = key(gold).split()
        match = {}
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=seq, b=gseq, autojunk=False).get_opcodes():
            if op == "equal":
                match.update({i1 + d: j1 + d for d in range(i2 - i1)})
        inside = set()
        for first, last, options in kept:
            toks = [i for i, (_, k) in enumerate(units) if first <= k <= last]
            if not toks:
                continue
            inside.update(toks)
            g0, g1 = anchored(match, toks[0], toks[-1] + 1, len(seq), len(gseq))
            gk = " ".join(gseq[g0:g1])
            keys = [key(o) for o in options]
            if keys[0] == gk:
                s["brace: main right"] += 1
            elif gk in keys[1:]:
                s["brace: another option right"] += 1
            else:
                s["brace: none right"] += 1
        for i in range(len(seq)):
            if i in inside:
                s["tokens in braces"] += 1
                s["right in braces"] += i in match
            else:
                s["tokens outside"] += 1
                s["right outside"] += i in match
    out = {}
    for kind, s in st.items():
        e = sorted(s.pop("extra words"))
        n_tok = s["tokens in braces"] + s["tokens outside"]
        braces = s["brace: main right"] + s["brace: another option right"] + s["brace: none right"]
        out[kind] = {**s, "share with braces": round(s["with braces"] / s["clips"], 3),
                     "spans per clip": round(s["spans"] / s["clips"], 2),
                     "extra words mean": round(statistics.mean(e), 2), "extra words p95": e[int(0.95 * len(e))],
                     "share of words in braces": round(s["tokens in braces"] / max(1, n_tok), 3),
                     "right in braces %": round(100 * s["right in braces"] / max(1, s["tokens in braces"]), 1),
                     "right outside %": round(100 * s["right outside"] / max(1, s["tokens outside"]), 1),
                     "brace main right %": round(100 * s["brace: main right"] / max(1, braces), 1),
                     "brace other right %": round(100 * s["brace: another option right"] / max(1, braces), 1)}
    return out


def show(title: str, result: dict) -> None:
    print(f"\n== {title}")
    cols = ["clips", "share with braces", "spans per clip", "extra words mean", "extra words p95",
            "share of words in braces", "right in braces %", "right outside %", "brace main right %", "brace other right %"]
    for kind, r in result.items():
        print(f"  {kind:<8} " + "  ".join(f"{c.split(' %')[0]}={r[c]}" for c in cols if c in r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()
    items = data.load_split("train", data.find_root("train"))
    train = {k: json.load(open(REPO / v)) for k, v in TRAIN.items()}
    clips, seen = [], set()
    for it in items:
        turns = [(h["audio"], h["audio_text"], False) for h in it.get("history") or []
                 if h.get("role") == "user" and h.get("audio") and h.get("audio_text")]
        for audio, gold, latest in turns + [(it["audio"], it["audio_text"], True)]:
            if audio in seen or audio not in train["avg5"]:
                continue
            seen.add(audio)
            clips.append((audio, gold, latest, it["id"]))
    rows = [(g, train["avg5"][a], [train[k].get(a, "") for k in FIVE], latest) for a, g, latest, _ in clips]
    report = {"real train": measure(rows)}
    show("real train views (out-of-fold readings)", report["real train"])

    public = {k: json.load(open(REPO / v)) for k, v in PUBLIC.items()}
    prow = [(None, public["avg5"][a], [public[k].get(a, "") for k in FIVE], True) for a in public["avg5"]]
    report["public"] = measure(prow)
    show("public views (all-data readings; every clip counted as 'latest')", report["public"])

    # simulated views: the bank is built from clips outside validation and replays them on validation's said texts,
    # which it has not seen as whole texts (as for synthetic conversations); compared with validation's real views
    val = set(json.load(open(REPO / "src/lifemate/splits.json"))["random"]) | set(json.load(open(REPO / "src/lifemate/splits.json"))["apps"])
    bank = view.ReadingsBank.from_rows([(g, [train[k].get(a, "") for k in ["avg5"] + FIVE])
                                        for a, g, _, sid in clips if sid not in val])
    print(f"\nreadings bank: {len(bank.seen)} said words, {len(bank.bank)} with disagreements, {len(bank.rare)} rare patterns")
    vrows = [(g, train["avg5"][a], [train[k].get(a, "") for k in FIVE], latest) for a, g, latest, sid in clips if sid in val]
    report["real validation"] = measure(vrows)
    show("real validation views", report["real validation"])
    rng = random.Random(2026)
    srows = []
    for a, g, latest, sid in clips:
        if sid in val:
            m, o = bank.readings(g, rng)
            srows.append((g, m, o, latest))
    report["bank-simulated validation"] = measure(srows)
    show("bank-simulated validation views", report["bank-simulated validation"])
    noise = ConfusionNoise.from_pairs([(g, train["avg5"][a]) for a, g, _, sid in clips if sid not in val])
    rng = random.Random(2026)
    irows = [(g, noise.apply(g, rng), [noise.apply(g, rng) for _ in FIVE], latest) for a, g, latest, sid in clips if sid in val]
    report["independent-noise validation"] = measure(irows)
    show("independent confusion noise (for contrast)", report["independent-noise validation"])

    rng = random.Random(7)
    latest_val = [c for c in clips if c[2] and c[3] in val]
    print("\n== examples (validation latest requests)")
    for a, g, _, sid in rng.sample(latest_val, args.examples):
        print(f"  {sid} said : {g}\n  {' ' * len(sid)} view : {view.build(train['avg5'][a], [train[k].get(a, '') for k in FIVE])}")
    print("\n== bank-simulated examples")
    srng = random.Random(3)
    for a, g, _, sid in rng.sample(latest_val, 4):
        print(f"  {sid} said : {g}\n  {' ' * len(sid)} sim  : {bank.view(g, srng)}")
    out = REPO / "scripts/hearing_analysis/view_check.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
