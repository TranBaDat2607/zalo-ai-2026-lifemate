"""Batch-A routing file for public: the no-notes file with the rows the input classifier puts in batch A replaced by the
notes model's rows. Uploaded next to the no-notes file, the score difference measures the notes on batch A only.

    python scripts/notes/route_batch_a.py \
        outputs/notes-pub/public_base.jsonl outputs/notes-pub/public_notes.jsonl outputs/notes-pub/public_routed.jsonl

The classifier is the one from scripts/eda/batch_detect.py (metadata + request words, logistic regression, C=3),
fitted on train with the gold transcripts and applied to public with the five-copy speech transcripts. A row is swapped
when P(batch A) >= --threshold (0.9, fixed before any public score). Needs scikit-learn (kaggle-grandmaster env).
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eda"))
from common import gen_by_id, load, public_batch  # noqa: E402


def apps(s):
    return [a if isinstance(a, str) else a.get("name", a.get("app")) for a in s["context"]["installed_apps"]]


def batch_a_probability(asr_path: str) -> dict:
    tr, pub = load("train"), load("public_test")
    asr = json.loads(Path(asr_path).read_text(encoding="utf-8"))
    core = {a for s in tr if gen_by_id(s["id"]) == "A" for a in apps(s)}
    cities = sorted({s["context"]["user_profile"].get("city") or "" for s in tr + pub})

    def meta(s):
        m = s["context"]["current_time"][5:7]
        city = s["context"]["user_profile"].get("city") or ""
        ap = apps(s)
        hu = sum(h.get("role") == "user" for h in s["history"])
        ha = sum(h.get("role") == "assistant" for h in s["history"])
        return [m == "09", m == "10", m == "11", m == "12"] + [city == c for c in cities] + [
            len(ap) / 10, set(ap) <= core, len(set(ap) - core) / 5, hu / 5, ha / 5, hu == 0, hu >= 8]

    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    xtr = hstack([csr_matrix(np.array([meta(s) for s in tr], float)) * 3,
                  v.fit_transform([s["audio_text"] for s in tr])]).tocsr()
    xpub = hstack([csr_matrix(np.array([meta(s) for s in pub], float)) * 3,
                   v.transform([asr.get(s["audio"], "") for s in pub])]).tocsr()
    clf = LogisticRegression(C=3, max_iter=3000).fit(xtr, np.array([gen_by_id(s["id"]) for s in tr]))
    col = list(clf.classes_).index("A")
    return {s["id"]: float(p) for s, p in zip(pub, clf.predict_proba(xpub)[:, col])}


def read(path: str) -> list:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("notes")
    ap.add_argument("out")
    ap.add_argument("--asr", default=str(REPO / "outputs/final1/avg5_asr/public_test.json"))
    ap.add_argument("--threshold", type=float, default=0.9)
    args = ap.parse_args()
    base, notes = read(args.base), read(args.notes)
    assert [r["id"] for r in base] == [r["id"] for r in notes] and len(base) == 634, "files differ in ids or order"
    prob = batch_a_probability(args.asr)
    by_id = {r["id"]: r for r in notes}
    swap = [r["id"] for r in base if prob[r["id"]] >= args.threshold]
    out = [by_id[r["id"]] if prob[r["id"]] >= args.threshold else r for r in base]
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    changed = sum(by_id[i]["tool_calls"] != r["tool_calls"] for i, r in zip([r["id"] for r in base], base) if i in swap)
    print(f"rows swapped to the notes model: {len(swap)} of {len(base)} (P(A) >= {args.threshold}); "
          f"of these, {changed} have different calls")
    print("swapped rows by the date/city batch:", dict(collections.Counter(public_batch(i) for i in swap)))
    print("written:", args.out)


if __name__ == "__main__":
    main()
