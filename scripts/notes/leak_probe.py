"""Leak probe: does a note tell the answer-key's branch beyond what the request and the history already say?

    python scripts/notes/leak_probe.py            # v1 and v2
    ... leak_probe.py --notes notes/batch_*.jsonl --name v1                                             # one set

K1 samples: predict 'the key asks' (ends with sys.ask_missing_info) against 'the key acts'.
K2 samples: predict 'the key writes' against 'the key only reads'.
Model: logistic regression on word 1-2-gram TF-IDF, 5x5 repeated stratified CV (the same folds for every variant).
(a) request + history text alone; (b) the same plus the note, as separate feature blocks, so the note's own n-grams
can be listed. Result per task: AUC(a), AUC(b) and AUC(b) - AUC(a) (mean over the 25 folds, with the standard deviation
across folds and the standard error of the paired difference over the 5 repeats).
The v2 acceptance test: AUC(b) - AUC(a) <= 0.02 for both tasks. The v1 notes are the positive control.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from make_inputs import history_lines
from notes_common import REPO
from notes_v2_common import analyze_all


def load_notes(patterns):
    out = {}
    for pat in patterns:
        for p in sorted(Path().glob(pat)) if not Path(pat).is_absolute() else sorted(Path(pat).parent.glob(Path(pat).name)):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    out[r["id"]] = r["note"]
    return out


def tasks(pool, infos):
    """name -> (ids, labels). K1: asks=1. K2: writes=1."""
    k1 = [i for i in pool if infos[i["id"]]["primary"] == "K1"]
    k2 = [i for i in pool if infos[i["id"]]["primary"] == "K2"]
    return {"K1 ask-vs-act": ([i["id"] for i in k1], [int(infos[i["id"]]["ends_ask"]) for i in k1]),
            "K2 write-vs-read-only": ([i["id"] for i in k2], [int(infos[i["id"]]["any_write"]) for i in k2])}


def vec(**kw):
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True, **kw)


def run_task(ids, y, base_text, notes, seed=0, C=1.0, repeats=5):
    y = np.array(y)
    ids = list(ids)
    txt_a = [base_text[i] for i in ids]
    txt_n = [notes[i] for i in ids]
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    aucs_a, aucs_b, aucs_n = [], [], []
    for tr, te in rskf.split(ids, y):
        va, vn = vec(), vec()
        Xa_tr = va.fit_transform([txt_a[i] for i in tr]); Xa_te = va.transform([txt_a[i] for i in te])
        Xn_tr = vn.fit_transform([txt_n[i] for i in tr]); Xn_te = vn.transform([txt_n[i] for i in te])
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(Xa_tr, y[tr]); aucs_a.append(roc_auc_score(y[te], clf.decision_function(Xa_te)))
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(hstack([Xa_tr, Xn_tr]).tocsr(), y[tr]); aucs_b.append(roc_auc_score(y[te], clf.decision_function(hstack([Xa_te, Xn_te]).tocsr())))
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(Xn_tr, y[tr]); aucs_n.append(roc_auc_score(y[te], clf.decision_function(Xn_te)))
    a, b, n = np.array(aucs_a), np.array(aucs_b), np.array(aucs_n)
    d = b - a
    per_rep = d.reshape(repeats, 5).mean(axis=1)
    return {"n": len(ids), "pos": int(y.sum()), "auc_request_history": a.mean(), "sd_a": a.std(), "auc_with_note": b.mean(), "sd_b": b.std(),
            "gain": d.mean(), "gain_sd_folds": d.std(), "gain_se_repeats": per_rep.std(ddof=1) / np.sqrt(repeats), "auc_note_only": n.mean()}


def top_note_ngrams(ids, y, notes, k=12, C=1.0):
    v = vec()
    X = v.fit_transform([notes[i] for i in ids])
    clf = LogisticRegression(C=C, max_iter=2000).fit(X, np.array(y))
    names = np.array(v.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    return [(names[j], round(float(clf.coef_[0][j]), 2)) for j in order[::-1][:k]], [(names[j], round(float(clf.coef_[0][j]), 2)) for j in order[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", nargs="+", action="append", default=None, help="note files (glob), repeat with --name")
    ap.add_argument("--name", action="append", default=None)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    sets = list(zip(args.name or ["v1", "v2"], args.notes or [[str(REPO / "notes" / "batch_*.jsonl")], [str(REPO / "notes_v2" / "batch_*.jsonl")]]))
    pool, catalog, infos = analyze_all()
    base_text = {}
    for it in pool:
        base_text[it["id"]] = it["audio_text"] + " || " + " ".join(history_lines(it))
    out = {}
    for name, pats in sets:
        notes = load_notes(pats)
        print(f"\n=== {name}: {len(notes)} notes")
        for tname, (ids, y) in tasks(pool, infos).items():
            keep = [(i, l) for i, l in zip(ids, y) if i in notes]
            if len(keep) < 40 or len({l for _, l in keep}) < 2:
                print(tname, "too few notes", len(keep)); continue
            ids2, y2 = [i for i, _ in keep], [l for _, l in keep]
            r = run_task(ids2, y2, base_text, notes, repeats=args.repeats, C=args.C)
            out[f"{name}|{tname}"] = r
            print(f"{tname}: n={r['n']} positives={r['pos']}  AUC request+history {r['auc_request_history']:.3f}±{r['sd_a']:.3f}  "
                  f"+note {r['auc_with_note']:.3f}±{r['sd_b']:.3f}  gain {r['gain']:+.3f} (sd over folds {r['gain_sd_folds']:.3f}, "
                  f"se over repeats {r['gain_se_repeats']:.3f})  note alone {r['auc_note_only']:.3f}")
            pos, neg = top_note_ngrams(ids2, y2, notes)
            print("   note n-grams pointing to the positive class:", pos)
            print("   note n-grams pointing to the negative class:", neg)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
