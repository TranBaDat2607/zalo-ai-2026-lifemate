"""Extra speech copies to average (log.md, 6 Nov 2026): --fold-seed keeps exp7's split while --seed changes the rest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate import asr_train  # noqa: E402

ITEMS = [{"id": f"s{i:03d}", "history": [{"role": "user", "audio": f"h{i}.wav", "audio_text": "x"}],
          "audio": f"a{i}.wav", "audio_text": "y"} for i in range(30)]


def _run(monkeypatch, tmp_path, *extra) -> dict:
    seen = {}
    monkeypatch.setattr(asr_train.data, "find_root", lambda split: str(tmp_path))
    monkeypatch.setattr(asr_train.data, "load_split", lambda split, root: ITEMS)

    def fake_train(args, train_clips, eval_clips, root, tag):
        seen.update(train=sorted(a for a, *_ in train_clips), eval=sorted(a for a, *_ in eval_clips), seed=args.seed)
        return tmp_path / tag / "best"

    monkeypatch.setattr(asr_train, "train_one", fake_train)
    monkeypatch.setattr(asr_train, "transcribe_with", lambda best, args, audio, root: {a: "" for a in audio})
    asr_train.main(["--out", str(tmp_path / "out"), "--folds", "3", "--plan", "0", *extra])
    return seen


def test_fold_seed_keeps_the_split_and_seed_still_reaches_training(monkeypatch, tmp_path):
    base = _run(monkeypatch, tmp_path)
    kept = _run(monkeypatch, tmp_path, "--seed", "2027", "--fold-seed", "2026")
    moved = _run(monkeypatch, tmp_path, "--seed", "2027")
    folds = asr_train.sample_folds(ITEMS, 3, 2026)
    held = sorted(a for i in ITEMS if folds[i["id"]] == 0 for a, _ in asr_train.clips_of(i))
    assert base["eval"] == kept["eval"] == held
    assert base["train"] == kept["train"]
    assert not set(kept["train"]) & set(kept["eval"])
    assert (base["seed"], kept["seed"]) == (2026, 2027)
    assert moved["eval"] != held  # without --fold-seed a new seed moves the folds, as before
