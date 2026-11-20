"""Audit behind METHODOLOGY_REVIEW.md (11 Nov 2026). CPU only, saved runs only, no public labels.

    LIFEMATE_DATA=$PWD/train+public_test/train+public_test python scripts/review/audit_11nov.py

Sections:
1. validation (386): hearing loss (perfect latest text minus heard text) and agent loss (100 minus perfect text),
   by half, batch, answer type, number of gold calls;
2. agent losses with perfect text by first divergence (scripts/first_error.py) and half; wrong state vs call shape;
3. flip ceiling: best score per sample over every saved validation run;
4. asks and empty answers against gold, heard vs perfect text;
5. time spans and `minutes` arguments;
6. public (no labels): answer mix against train gold rates per batch; how often the speech models disagree on the
   latest request, public (all-data models) against train (out-of-fold models).
"""
import collections
import glob
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in ("src", "scripts", "scripts/notes"):
    sys.path.insert(0, str(REPO / p))
import first_error as fe  # noqa: E402
from compare_notes import SPLITS, batch, load  # noqa: E402
from lifemate import data, view  # noqa: E402

HEARD, PERFECT = "outputs/notes-aux/aux_eval", "outputs/stage0-latest/oracle_latest"
PUBLIC_FILE = "outputs/public-view2/submission_public_view2.jsonl"
READINGS = ["medium", "medium10", "b", "s2027", "s2028"]


def public_batch(sid: str) -> str:
    n = int(sid.split("_")[1])
    return "A" if n < 174 else ("B" if n < 393 else "C")


def answer_type(calls: list, catalog: dict) -> str:
    if not calls:
        return "nothing"
    if calls[-1]["name"] == "sys.ask_missing_info":
        return "ask"
    return "act" if any(catalog.get(c["name"], {}).get("kind") == "write" for c in calls) else "read-only"


def main() -> None:
    root = data.find_root("train")
    train = {s["id"]: s for s in data.load_split("train", root)}
    catalog = data.load_catalog("train", root)
    H, O = load(HEARD), load(PERFECT)
    ids, n_all, rnd = sorted(H), len(H), set(SPLITS["random"])
    half = lambda i: "random" if i in rnd else "unseen apps"

    print("1. hearing vs agent (points of the slice; last two columns = points of the 386 mean)")

    def row(name: str, sel: list) -> None:
        if not sel:
            return
        h = 100 * sum(H[i]["score"] for i in sel) / len(sel)
        o = 100 * sum(O[i]["score"] for i in sel) / len(sel)
        print(f"  {name:22s} n={len(sel):3d} heard {h:5.1f} perfect {o:5.1f} hearing {o - h:5.1f} agent {100 - o:5.1f}"
              f" | {(o - h) * len(sel) / n_all:5.2f} {(100 - o) * len(sel) / n_all:5.2f}")

    row("all", ids)
    for k in ("random", "unseen apps"):
        row(k, [i for i in ids if half(i) == k])
    for b in "ABC":
        row("batch " + b, [i for i in ids if batch(i) == b])
    for k in ("random", "unseen apps"):
        for b in "ABC":
            row(f"{k} {b}", [i for i in ids if half(i) == k and batch(i) == b])
    for t in ("act", "ask", "read-only", "nothing"):
        row("gold " + t, [i for i in ids if answer_type(H[i]["gold"], catalog) == t])
    for k in range(6):
        row(f"gold calls {k}{'+' if k == 5 else ''}",
            [i for i in ids if (len(H[i]["gold"]) == k if k < 5 else len(H[i]["gold"]) >= 5)])

    print("\n2. first divergence, points of the 386 mean: heard | perfect text (random / unseen apps)")
    table = collections.defaultdict(lambda: collections.defaultdict(float))
    for name, run in (("heard", H), ("perfect", O)):
        for i in ids:
            r = run[i]
            if r["score"] > 0.999:
                continue
            label = fe.classify(r, catalog)[0]
            top = label.split(" (")[0].split(":")[0]
            if top.startswith("wrong arguments"):
                kinds = label.split(": ")[1]
                top += " / " + ("free text" if kinds == "free text" else "missing or extra argument"
                                if "missing/extra" in kinds else "number or date"
                                if ("number" in kinds or "date" in kinds) else "id, name or choice")
            table[top][name] += 100 * (1 - r["score"]) / n_all
            if name == "perfect":
                table[top][half(i)] += 100 * (1 - r["score"]) / n_all
    for top, v in sorted(table.items(), key=lambda kv: -kv[1]["perfect"]):
        print(f"  {top:52s} {v['heard']:5.2f} | {v['perfect']:5.2f} ({v['random']:.2f} / {v['unseen apps']:.2f})")
    for name, run in (("heard", H), ("perfect", O)):
        wrong = [i for i in ids if run[i]["gold"] and not run[i]["hash_ok"]]
        shape = [i for i in ids if run[i]["gold"] and run[i]["hash_ok"] and run[i]["score"] < 0.999]
        lost = lambda sel: 100 * sum(1 - run[i]["score"] for i in sel) / n_all
        print(f"  {name}: wrong state {len(wrong)} samples {lost(wrong):.1f} pts; right state, calls differ "
              f"{len(shape)} samples {lost(shape):.1f} pts")

    print("\n3. flip ceiling")
    runs = {}
    for d in sorted(glob.glob(str(REPO / "outputs/*/*/detail.0.jsonl"))):
        r = load(str(Path(d).parent))
        if set(r) == set(ids):
            runs[str(Path(d).parent.relative_to(REPO))] = r
    heard_runs = {p: r for p, r in runs.items() if "oracle" not in p and "dpo-val" not in p}
    best = {i: max(r[i]["score"] for r in heard_runs.values()) for i in ids}
    never = [i for i in ids if max(best[i], O[i]["score"]) < 0.999]
    print(f"  best per sample over {len(heard_runs)} heard-text runs: {100 * sum(best.values()) / n_all:.2f}")
    print(f"  never perfect in any run, perfect-text run included: {len(never)} samples, "
          f"{100 * sum(1 - O[i]['score'] for i in never) / n_all:.2f} pts of the perfect-text run's loss")

    print("\n4. asks and empty answers")
    for name, run in (("heard", H), ("perfect", O)):
        for b in "ABC":
            sel = [i for i in ids if batch(i) == b]
            ask = lambda calls: bool(calls) and calls[-1]["name"] == "sys.ask_missing_info"
            false_ask = [i for i in sel if ask(run[i]["pred"]) and not ask(run[i]["gold"])]
            false_empty = [i for i in sel if not run[i]["pred"] and run[i]["gold"]]
            print(f"  {name:7s} {b}: gold asks {sum(ask(run[i]['gold']) for i in sel):2d} ours "
                  f"{sum(ask(run[i]['pred']) for i in sel):2d}; needless asks cost "
                  f"{100 * sum(1 - run[i]['score'] for i in false_ask) / n_all:.2f} pts; needless empty answers "
                  f"{len(false_empty)} ({100 * len(false_empty) / n_all:.2f} pts)")

    print("\n5. time spans and minutes")
    span = re.compile(r"\btừ\b.{0,40}\b(đến|tới)\b.{0,25}(giờ|rưỡi|kém|phút)|\brưỡi\b.{0,30}\b(đến|tới)\b", re.I)
    sel = [i for i in ids if span.search(train[i]["audio_text"])]
    print(f"  requests with a time span: validation {len(sel)}, heard "
          f"{100 * sum(H[i]['score'] for i in sel) / len(sel):.1f}, perfect text "
          f"{100 * sum(O[i]['score'] for i in sel) / len(sel):.1f}; training samples with one: "
          f"{sum(1 for i, s in train.items() if i not in H and span.search(s['audio_text']))}")
    has_min = lambda calls: any("minutes" in (c.get("arguments") or {}) for c in calls)
    sel = [i for i in ids if has_min(H[i]["gold"])]
    right = lambda run, i: all(any(p["name"] == c["name"] and (p.get("arguments") or {}).get("minutes") ==
                                   c["arguments"]["minutes"] for p in run[i]["pred"])
                               for c in run[i]["gold"] if "minutes" in (c.get("arguments") or {}))
    print(f"  gold `minutes` argument: validation {len(sel)} samples, right heard {sum(right(H, i) for i in sel)}, "
          f"perfect text {sum(right(O, i) for i in sel)}; training samples outside validation: "
          f"{sum(1 for i, s in train.items() if i not in H and has_min(s['label']['tool_calls']))}")

    print("\n6. public, no labels")
    proot = data.find_root("public_test")
    public = data.load_split("public_test", proot)
    pcat = data.load_catalog("public_test", proot)
    new_apps = {n.split(".")[0] for n in pcat} - {n.split(".")[0] for n in catalog}
    rows = {json.loads(line)["id"]: json.loads(line) for line in (REPO / PUBLIC_FILE).read_text().splitlines()}
    gold_mix, gold_n = collections.defaultdict(collections.Counter), collections.Counter()
    for s in train.values():
        gold_mix[batch(s["id"])][answer_type(s["label"]["tool_calls"], catalog)] += 1
        gold_n[batch(s["id"])] += 1
    ours = collections.defaultdict(collections.Counter)
    for sid, r in rows.items():
        ours[public_batch(sid)][answer_type(r["tool_calls"], pcat)] += 1
    for b in "ABC":
        n = sum(ours[b].values())
        print(f"  batch {b} (n={n}): ours / train gold, %: " + ", ".join(
            f"{t} {100 * ours[b][t] / n:.1f} / {100 * gold_mix[b][t] / gold_n[b]:.1f}"
            for t in ("act", "ask", "read-only", "nothing")))
    used_new = sum(any(c["name"].split(".")[0] in new_apps for c in r["tool_calls"]) for r in rows.values())
    print(f"  public answers that call one of the {len(new_apps)} public-only apps: {used_new} of {len(rows)}")

    def disagreement(split: str, items: list, group) -> None:
        main_r = json.loads((REPO / f"outputs/readings/avg5_{split}.json").read_text())
        others = [json.loads((REPO / f"outputs/readings/{n}_{split}.json").read_text()) for n in READINGS]
        acc = collections.defaultdict(lambda: [0, 0, 0, 0])
        for s in items:
            k = s["audio"]
            if k not in main_r:
                continue
            braces = len(re.findall(r"\{[^}]*\}", view.build(main_r[k], [o.get(k, "") for o in others])))
            a = acc[group(s)]
            a[0] += 1
            a[1] += braces > 0
            a[2] += braces
            a[3] += len(main_r[k].split())
        for key, a in sorted(acc.items()):
            print(f"  {key:10s} n={a[0]:4d} requests with braces {100 * a[1] / a[0]:5.1f} %, braces per 100 words "
                  f"{100 * a[2] / a[3]:5.2f}")

    disagreement("train", list(train.values()), lambda s: "train " + batch(s["id"]))
    disagreement("public_test", public, lambda s: "public " + public_batch(s["id"]))


if __name__ == "__main__":
    main()
