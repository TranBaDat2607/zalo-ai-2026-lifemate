"""The four cross-validation folds (scripts/make_cv_folds.py): fold 0 is the fixed validation split, every sample is
scored exactly once, and no fold-K model sees a gold call on one of fold K's held-out apps."""
import json
import os
from pathlib import Path

import pytest

from lifemate import data

PKG = Path(data.__file__).parent
REPO = PKG.parents[1]
FOLDS = json.loads((PKG / "folds_cv.json").read_text(encoding="utf-8"))


def test_fold_zero_is_the_fixed_validation_split():
    splits = json.loads((PKG / "splits.json").read_text(encoding="utf-8"))
    assert data.id_set("cv:0") == data.id_set("cvscore:0") == data.id_set("splits")
    assert data.id_set("cvheld:0") == set(splits["apps"])
    assert data.id_set("cvrandom:0") == set(splits["random"])
    assert FOLDS["groups"]["0"] == sorted(splits["held_apps"])


def test_every_sample_is_scored_once_and_training_never_sees_its_own_fold():
    scored = [s for k in "0123" for s in FOLDS["held"][k] + FOLDS["random"][k]]
    assert len(scored) == len(set(scored)) == 1545
    for k in "0123":
        assert data.id_set(f"cvscore:{k}") <= data.id_set(f"cv:{k}")
        assert not (data.id_set(f"cvheld:{k}") & data.id_set(f"cvrandom:{k}"))
        assert 380 <= len(data.id_set(f"cvscore:{k}")) <= 392
    groups = [set(FOLDS["groups"][k]) for k in "0123"]
    assert all(not (groups[a] & groups[b]) for a in range(4) for b in range(a + 1, 4))


def test_template_conversations_never_call_a_held_app():
    """Every fold trains with --synth package, so those conversations must not carry a held app's gold calls."""
    held = set().union(*(FOLDS["groups"][k] for k in "0123"))
    lines = (PKG / "synth_samples.jsonl").read_text(encoding="utf-8").splitlines()
    called = {c["name"].split(".")[0] for line in lines if line.strip() for c in json.loads(line)["label"]["tool_calls"]}
    assert not (called & held)


def test_no_gold_call_on_a_held_app_in_fold_training():
    root = Path(os.environ.get("LIFEMATE_DATA", REPO / "train+public_test" / "train+public_test"))
    if not (root / "train" / "train.jsonl").is_file():
        pytest.skip("competition data not available")
    items = data.load_split("train", root)
    for k in "0123":
        held_apps, skipped = set(FOLDS["groups"][k]), data.id_set(f"cv:{k}")
        trained = [i for i in items if i["id"] not in skipped]
        assert not any(c["name"].split(".")[0] in held_apps for i in trained for c in i["label"]["tool_calls"])
        # the held part is exactly the scored samples whose gold uses a held app
        uses = {i["id"] for i in items if any(c["name"].split(".")[0] in held_apps for c in i["label"]["tool_calls"])}
        assert data.id_set(f"cvheld:{k}") == uses & data.id_set(f"cvscore:{k}")
