"""Can the batch be read from visible inputs? Logistic regression on metadata (date, city, app counts, history
length), optionally plus the words of the final request; 5-fold accuracy on train, agreement on public (batches from
common.gen_by_id / public_batch). Persona names are left out because public brings new ones.

Needs scikit-learn (kaggle-grandmaster environment).
Usage: python scripts/eda/batch_detect.py outputs/final1/avg5_asr/public_test.json
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).parent))
from common import gen_by_id, load, public_batch  # noqa: E402


def apps(s):
    return [a if isinstance(a, str) else a.get("name", a.get("app")) for a in s["context"]["installed_apps"]]


def main(pub_asr):
    tr, pub = load("train"), load("public_test")
    asr = json.load(open(pub_asr, encoding="utf-8"))
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

    ytr = np.array([gen_by_id(s["id"]) for s in tr])
    ypub = np.array([public_batch(s["id"]) for s in pub])
    mtr = csr_matrix(np.array([meta(s) for s in tr], float))
    mpub = csr_matrix(np.array([meta(s) for s in pub], float))
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for name, text in (("metadata only", False), ("metadata + request words", True)):
        if text:
            v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            xtr = hstack([mtr * 3, v.fit_transform([s["audio_text"] for s in tr])]).tocsr()
            xpub = hstack([mpub * 3, v.transform([asr.get(s["audio"], "") for s in pub])]).tocsr()
        else:
            xtr, xpub = mtr, mpub
        clf = LogisticRegression(C=3, max_iter=3000)
        oof = cross_val_predict(clf, xtr, ytr, cv=skf, method="predict_proba")
        cls = np.array(sorted(set(ytr)))
        clf.fit(xtr, ytr)
        pp = clf.predict_proba(xpub)
        pred = clf.classes_[pp.argmax(1)]
        sure = pp.max(1) >= 0.9
        print(f"{name}: train 5-fold accuracy {(cls[oof.argmax(1)] == ytr).mean():.3f}; public agreement {(pred == ypub).mean():.3f}; "
              f"p>=0.9 on {sure.mean():.2f} of public, agreement there {(pred[sure] == ypub[sure]).mean():.3f}")
        print("   public (batch, predicted):", {f"{a}->{b}": n for (a, b), n in sorted(collections.Counter(zip(ypub, pred)).items())})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/final1/avg5_asr/public_test.json")
