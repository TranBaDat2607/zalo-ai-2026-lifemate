"""Why the worked notes helped batch A and hurt batch C (analysis of the notes-folds run, 8 Nov 2026).

    LIFEMATE_DATA=$PWD/train+public_test/train+public_test \
        python scripts/notes/why_notes.py outputs/notes-folds/notes_eval outputs/harm-check5/avg5_eval

Prints, for the paired runs (notes soup against the accepted baseline, same five-copy transcripts):
1. the difference by batch and validation half;
2. the difference by what the answer key does: act with no read, act after reads, ask after reads, ask directly;
3. ask/act agreement with the answer key for both runs;
4. the training notes whose wording gives away what a read will show (the hedge phrase and "chỉ đọc");
5. simple routing rules replayed on validation. These are chosen after seeing the results: an upper estimate.
"""
import argparse
import collections
import itertools
import json
import re
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lifemate import data  # noqa: E402

SPLITS = json.loads((REPO / "src" / "lifemate" / "splits.json").read_text(encoding="utf-8"))
RANDOM = set(SPLITS["random"])
MIX = {"A": 0.24, "B": 0.32, "C": 0.44}
HEDGE = "không xác định được"


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


def is_ask(calls) -> bool:
    return bool(calls) and calls[-1]["name"] == "sys.ask_missing_info"


def line(name, diffs) -> str:
    n = len(diffs)
    return (f"{name:<26} n={n:>3}  diff={100 * sum(diffs) / n:+7.2f}  up={sum(d > 1e-9 for d in diffs):>2}"
            f"  down={sum(d < -1e-9 for d in diffs):>2}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("notes")
    ap.add_argument("baseline")
    args = ap.parse_args()
    a, b = load(args.notes), load(args.baseline)
    ids = sorted(set(a) & set(b))
    root = data.find_root("train")
    kind = {name: spec["kind"] for name, spec in data.load_catalog("train", root).items()}
    diff = {i: a[i]["score"] - b[i]["score"] for i in ids}

    print("1. By batch and half")
    for k in "ABC":
        for half, sel in (("random", True), ("held-out apps", False)):
            cell = [diff[i] for i in ids if batch(i) == k and (i in RANDOM) == sel]
            if cell:
                print("  " + line(f"{k} {half}", cell))
    a_cell = [d for i, d in diff.items() if batch(i) == "A" and abs(d) > 1e-9]
    up = sum(d > 0 for d in a_cell)
    p = 2 * sum(comb(len(a_cell), j) for j in range(max(up, len(a_cell) - up), len(a_cell) + 1)) / 2 ** len(a_cell)
    flips = [sum(s * abs(d) for s, d in zip(signs, a_cell)) for signs in itertools.product((1, -1), repeat=len(a_cell))]
    exact = sum(abs(f) >= abs(sum(a_cell)) - 1e-12 for f in flips) / len(flips)
    print(f"  batch A: {len(a_cell)} samples changed, {up} up; two-sided sign test p = {min(p, 1):.3f};"
          f" exact sign-flip test on the sizes p = {exact:.3f}")

    print("2. By what the answer key does")

    def gold_type(calls) -> str:
        if not calls:
            return "empty"
        read = any(kind.get(c["name"]) == "read" for c in calls)
        if is_ask(calls):
            return "ask after read" if read else "ask directly"
        return "act after read" if read else "act, no read"

    groups = collections.defaultdict(list)
    for i in ids:
        t = gold_type(a[i]["gold"])
        groups[(t, "all")].append(diff[i])
        groups[(t, batch(i))].append(diff[i])
    for t in ("act, no read", "act after read", "ask after read", "ask directly", "empty"):
        for k in ("all", "A", "B", "C"):
            if groups[(t, k)]:
                print("  " + line(f"{t} ({k})", groups[(t, k)]))

    print("3. Ask/act against the answer key")
    for name, run in (("baseline", b), ("notes", a)):
        conf = collections.Counter()
        for i in ids:
            g, p_ = run[i]["gold"], run[i]["pred"]
            conf[("ask" if is_ask(g) else ("empty" if not g else "act"),
                  "ask" if is_ask(p_) else ("empty" if not p_ else "act"))] += 1
        print(f"  {name:<9}" + "  ".join(f"key {g} -> {p_}: {conf[(g, p_)]}" for g in ("act", "ask")
                                         for p_ in ("act", "ask", "empty")))

    print("4. Training notes that give away what a read will show")
    train = {r["id"]: r for r in data.load_split("train", root)}
    notes = {n["id"]: n["note"] for n in map(json.loads, (REPO / "src/lifemate/notes.jsonl").open(encoding="utf-8"))}
    hedge = collections.Counter("ask" if is_ask(train[i]["label"]["tool_calls"]) else "not ask"
                                for i, t in notes.items() if HEDGE in t)
    print(f"  '{HEDGE} thì sys.ask_missing_info': {sum(hedge.values())} notes; answer key asks in {hedge['ask']}")
    ro = collections.Counter()
    for i, t in notes.items():
        if "chỉ đọc" in t:
            calls = train[i]["label"]["tool_calls"]
            writes = any(kind.get(c["name"]) == "write" for c in calls)
            text = train[i].get("audio_text")
            text = (text[-1] if isinstance(text, list) else str(text or "")).lower()
            cond = bool(re.search(r"\b(nếu|thì|hông)\b", text))
            ro[("conditional" if cond else "plain", "key writes" if writes else "key only reads")] += 1
    print("  'chỉ đọc' notes: " + ", ".join(f"{k[0]} request, {k[1]}: {v}" for k, v in sorted(ro.items())))

    print("5. Routing rules replayed on validation (chosen after seeing the results)")

    def has_read(calls) -> bool:
        return any(kind.get(c["name"]) == "read" for c in calls)

    rules = {
        "baseline everywhere": lambda i: b,
        "notes everywhere": lambda i: a,
        "notes for batch A only": lambda i: a if batch(i) == "A" else b,
        "notes unless baseline reads": lambda i: b if has_read(b[i]["pred"]) else a,
        "best of the two (ceiling)": lambda i: a if a[i]["score"] > b[i]["score"] else b,
    }
    for name, pick in rules.items():
        by = {k: 100 * sum(pick(i)[i]["score"] for i in ids if batch(i) == k) / sum(batch(i) == k for i in ids)
              for k in "ABC"}
        total = 100 * sum(pick(i)[i]["score"] for i in ids) / len(ids)
        print(f"  {name:<30} {total:6.2f}   A {by['A']:5.1f}  B {by['B']:5.1f}  C {by['C']:5.1f}"
              f"   reweighted {sum(MIX[k] * by[k] for k in 'ABC'):6.2f}")


if __name__ == "__main__":
    main()
