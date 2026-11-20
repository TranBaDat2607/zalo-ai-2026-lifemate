"""The step-preference folds: disjoint, outside the validation set, balanced, and reachable through data.id_set."""
import json
from pathlib import Path

from lifemate import data

PKG = Path(data.__file__).parent


def test_folds_disjoint_and_outside_validation():
    folds = json.loads((PKG / "folds_dpo.json").read_text(encoding="utf-8"))
    assert sorted(folds) == ["1", "2", "3"]
    sets = [set(folds[k]) for k in "123"]
    assert all(len(s) == len(folds[k]) for s, k in zip(sets, "123"))  # no duplicates inside a fold
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    holdout = data.id_set("splits")
    assert not (set().union(*sets) & holdout)
    sizes = [len(s) for s in sets]
    assert max(sizes) - min(sizes) <= 2
    assert sum(sizes) == 1545 - len(holdout)


def test_id_set_reads_folds():
    folds = json.loads((PKG / "folds_dpo.json").read_text(encoding="utf-8"))
    for k in "123":
        assert data.id_set(f"fold:{k}") == set(folds[k])


def test_load_notes_refuses_held_notes_unless_dropped(tmp_path):
    import pytest
    from lifemate import train
    notes = tmp_path / "notes.jsonl"
    notes.write_text("\n".join(json.dumps({"id": i, "note": f"n {i}"}) for i in ("train_0001", "train_0002", "train_0003")),
                     encoding="utf-8")
    with pytest.raises(SystemExit):
        train.load_notes(str(notes), {"train_0002"})
    kept = train.load_notes(str(notes), {"train_0002"}, drop_held=True)
    assert sorted(kept) == ["train_0001", "train_0003"]
    assert train.load_notes(str(notes), set()) == {i: f"n {i}" for i in ("train_0001", "train_0002", "train_0003")}
