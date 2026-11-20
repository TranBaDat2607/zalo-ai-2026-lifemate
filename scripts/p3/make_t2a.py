"""T2a of improvment_plan3.md: every train sample again with each single out-of-fold reading (real hearing errors).

    python scripts/p3/make_t2a.py            # writes outputs/p3/t2a_full.jsonl and outputs/p3/t2a_fold0.jsonl

The recipe (0012, aux-full4) trains on the medium reading (--transcripts medium_train.json --noisy-only), which the
replay already holds. The other out-of-fold readings (avg5 = the averaged reading the package hears at test time,
medium10, b, s2027, s2028) each heard the clips a little differently. A variant is kept only when its texts differ
from the medium reading and from the variants already kept for that sample. The gold calls stay
as they are. The fold-0 file leaves out the 386 validation samples (train --extra refuses them anyway).
Rows: {kind: conv, type: T2a, source, reading, item, texts} for train --extra. Outputs are gitignored (competition data).
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lifemate import data, prompts  # noqa: E402
import common  # noqa: E402

READINGS = REPO / "outputs" / "readings"
TRAINED = "medium"
SINGLES = ["avg5", "medium10", "b", "s2027", "s2028"]
OUT = REPO / "outputs" / "p3"


def main() -> None:
    rows = data.load_split("train", data.find_root("train"))
    trained = json.loads((READINGS / f"{TRAINED}_train.json").read_text(encoding="utf-8"))
    singles = {n: json.loads((READINGS / f"{n}_train.json").read_text(encoding="utf-8")) for n in SINGLES}
    hold = data.id_set("splits")
    out, same, drops = [], 0, {"same words as gold": 0, "same words as medium": 0, "exact value lost": 0}
    catalog = {}
    for r in rows:
        gold = prompts.gold_transcripts(r)
        clips = list(gold)
        # 13 Nov Opus review: a variant counts only when its words differ (punctuation and case alone teach nothing),
        # and not when it is word for word the gold text (the replay's gold-text rows already hold it)
        wkey = lambda texts: tuple(tuple(common._WORD.findall(t.lower())) for t in texts)  # noqa: E731
        seen = {wkey(trained[a] for a in clips)}
        gold_key = wkey(gold[a] for a in clips)
        for n in SINGLES:
            texts = {a: singles[n][a] for a in clips}
            key = wkey(texts.values())
            if key in seen:
                same += 1
                drops["same words as medium"] += key == wkey(trained[a] for a in clips)
                continue
            seen.add(key)
            if key == gold_key or key[-1] == gold_key[-1]:  # the latest request word for word the gold text
                drops["same words as gold"] += 1
                continue
            if common.value_lost(r, texts, catalog):
                drops["exact value lost"] += 1
                continue
            out.append({"kind": "conv", "type": "T2a", "source": r["id"], "reading": n, "item": r, "texts": texts})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, keep in (("full", out), ("fold0", [x for x in out if x["source"] not in hold])):
        with open(OUT / f"t2a_{name}.jsonl", "w", encoding="utf-8") as f:
            for x in keep:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"t2a_{name}: {len(keep)} rows")
    print(json.dumps({"samples": len(rows), "variants": len(out), "dropped_same_text": same, "drops": drops,
                      "per_reading": {n: sum(x["reading"] == n for x in out) for n in SINGLES}}))


if __name__ == "__main__":
    main()
