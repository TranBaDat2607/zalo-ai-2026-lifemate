"""Distilling step-by-step: the worked note as a training-only side task (--notes-aux).

The side example is the sample's own prompt with prompts.PLAN_TAG after the latest request; the loss falls on the note
only. Main conversations, inference prompts and the step count of a run without side examples must not change.
"""
import json
import math
import os
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
REPO = Path(__file__).resolve().parents[1]


def _item(gold=True):
    calls = [{"name": "smart_home.get_status", "arguments": {}}] if gold else []
    return {"id": "train_9999", "audio": "a.wav", "audio_text": "Bật đèn phòng khách",
            "history": [{"role": "user", "audio": "h.wav", "audio_text": "Chào"}, {"role": "assistant", "text": "Chào bạn"}],
            "context": {"current_time": "2026-11-07T09:00:00+07:00", "user_profile": {}, "installed_apps": {
                "smart_home": {"tools": ["smart_home.get_status"]},
                "weather": {"tools": ["weather.get_forecast"]},
                "music": {"tools": ["music.play"]}}},
            "label": {"tool_calls": calls, "tool_results": [{"ok": True, "devices": []}] if gold else []}}


def _catalog():
    def spec(name, kind="read"):
        return {"name": name, "description": name, "kind": kind, "parameters": {"properties": {}}}
    return {n: spec(n) for n in ("smart_home.get_status", "weather.get_forecast", "music.play")} | {
        "sys.ask_missing_info": spec("sys.ask_missing_info", "meta")}


TOK = lambda t: [ord(c) for c in t]  # noqa: E731
TEXTS = {"a.wav": "bật đèn phòng khác", "h.wav": "chào"}
NOTE = "smart_home.get_status tìm đèn phòng khách; khớp đúng một thì bật."


def _decode(ids):
    return "".join(chr(i) for i in ids)


def test_plan_example_trains_only_the_note_after_the_tagged_request():
    from lifemate import prompts
    ids, labels = prompts.encode_plan_example(_item(), TEXTS, _catalog(), TOK, NOTE)
    text = _decode(ids)
    assert text.endswith("<|im_start|>user\nbật đèn phòng khác" + prompts.PLAN_TAG + "<|im_end|>\n"
                         "<|im_start|>assistant\n<think>\nBật đèn phòng khách\n" + NOTE)
    assert _decode([i for i, l in zip(ids, labels) if l != -100]) == NOTE  # the loss is on the note, all of it
    assert labels[-len(NOTE):] == ids[-len(NOTE):] and all(l == -100 for l in labels[:-len(NOTE)])
    last = text.split(prompts.PLAN_TAG)[1]
    assert "</think>" not in last and "<tool_call>" not in last  # no call, no end of turn: nothing teaches stopping there


def test_plan_prompt_is_the_main_prompt_plus_the_tag():
    from lifemate import prompts
    item, catalog = _item(), _catalog()
    main, _ = prompts.encode_conversation(item, TEXTS, catalog, TOK, denoise=True)
    plan, _ = prompts.encode_plan_example(item, TEXTS, catalog, TOK, NOTE)
    request = "<|im_start|>user\nbật đèn phòng khác"
    head = _decode(main).split(request)[0]
    assert _decode(plan).startswith(head + request + prompts.PLAN_TAG + "<|im_end|>")


def test_test_time_prompts_never_carry_the_tag():
    from lifemate import prompts
    item, catalog = _item(), _catalog()
    assert prompts.PLAN_TAG not in prompts.prompt_text(item, TEXTS, catalog)
    for note in (None, NOTE):
        ids, _ = prompts.encode_conversation(item, TEXTS, catalog, TOK, denoise=True, note=note)
        assert prompts.PLAN_TAG not in _decode(ids)


def test_plan_example_keeps_every_app_the_note_names():
    from lifemate import prompts
    item, catalog = _item(), _catalog()
    note = "weather.get_forecast xem mai; mưa thì smart_home.get_status, không thì thôi."
    assert prompts.note_apps(note, catalog) == {"weather", "smart_home"}
    ids, _ = prompts.encode_plan_example(item, TEXTS, catalog, TOK, note, drop_prob=1.0, rng=random.Random(0))
    system = _decode(ids).split("<|im_end|>")[0]
    assert "weather.get_forecast" in system and "smart_home.get_status" in system
    assert "music.play" not in system  # unnamed and unused: dropped as in the main conversations
    # an out-of-scope sample keeps its whole list, as encode_conversation does
    ids, _ = prompts.encode_plan_example(_item(gold=False), TEXTS, catalog, TOK, "Không có tool phù hợp.", drop_prob=1.0,
                                         rng=random.Random(0))
    assert "music.play" in _decode(ids).split("<|im_end|>")[0]


def test_plan_example_drops_unnamed_apps_to_fit_the_length_limit():
    from lifemate import prompts
    item, catalog = _item(), _catalog()
    full, _ = prompts.encode_plan_example(item, TEXTS, catalog, TOK, NOTE)
    short, labels = prompts.encode_plan_example(item, TEXTS, catalog, TOK, NOTE, max_len=len(full) - 1)
    assert len(short) < len(full) and _decode([i for i, l in zip(short, labels) if l != -100]) == NOTE
    assert "smart_home.get_status" in _decode(short).split("<|im_end|>")[0]


def test_side_examples_each_land_in_one_epoch():
    from lifemate import schedule
    side = schedule.side_by_epoch(10, 7, 2.0, seed=1)
    assert sorted(side) == [0, 1] and sorted(side[0] + side[1]) == list(range(10, 17))
    assert abs(len(side[0]) - len(side[1])) <= 1
    assert side == schedule.side_by_epoch(10, 7, 2.0, seed=1)  # a restarted process rebuilds the same split
    assert schedule.side_by_epoch(10, 7, 1.0, seed=1)[0] == sorted(side[0] + side[1], key=(side[0] + side[1]).index) or \
        sorted(schedule.side_by_epoch(10, 7, 1.0, seed=1)[0]) == list(range(10, 17))
    assert schedule.epoch_order(10, side, 0) == list(range(10)) + side[0]
    assert schedule.epoch_order(10, side, 2) == list(range(10))  # a partial third epoch: main conversations only


def test_without_side_examples_the_run_is_unchanged():
    from lifemate import schedule
    for n, epochs, accum in [(1716, 2.0, 4), (2058, 2.0, 4), (1715, 1.5, 4), (3, 2.0, 4)]:
        side = schedule.side_by_epoch(n, 0, epochs, seed=2026)
        for epoch in range(3):
            assert schedule.epoch_order(n, side, epoch) == list(range(n))
        assert schedule.planned_steps(n, 0, epochs, accum) == math.ceil(n * epochs / accum)
    assert schedule.planned_steps(1716, 0, 2.0, 4) == 858  # the fold copies of the baseline
    assert schedule.planned_steps(1716, 1055, 2.0, 4) == math.ceil((3432 + 1055) / 4)


def test_train_refuses_bad_combinations(tmp_path):
    import importlib
    train = importlib.import_module("lifemate.train")
    notes = tmp_path / "n.jsonl"
    notes.write_text(json.dumps({"id": "train_0003", "note": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="held-out"):
        train.load_notes(str(notes), {"train_0003"})
    assert train.load_notes(str(notes), set()) == {"train_0003": "x"}


# ---------------------------------------------------------------- the real notes with the real tokenizer
def _real():
    root = Path(os.environ.get("LIFEMATE_DATA", REPO / "train+public_test" / "train+public_test"))
    if not (root / "train" / "train.jsonl").exists():
        pytest.skip("data package not available")
    os.environ["LIFEMATE_DATA"] = str(root)
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3.5-4B tokenizer not cached")
    from lifemate import data
    return data, tokenizer


def test_real_notes_hold_no_validation_sample_and_train_exactly_the_note():
    data, tokenizer = _real()
    from lifemate import prompts, train
    holdout = data.id_set("splits")
    notes = train.load_notes("package-v2", holdout)  # raises on any validation id
    assert len(notes) == 1055 and not set(notes) & holdout
    root = data.find_root("train")
    items = {i["id"]: i for i in data.load_split("train", root)}
    catalog = data.load_catalog("train", root)
    tok = lambda t: tokenizer(t, add_special_tokens=False)["input_ids"]  # noqa: E731
    rng = random.Random(0)
    for sample_id in sorted(notes)[::20]:  # 53 samples across the whole file
        item, note = items[sample_id], notes[sample_id]
        texts = prompts.gold_transcripts(item)
        ids, labels = prompts.encode_plan_example(item, texts, catalog, tok, note, 3700, 0.5, rng)
        trained = tokenizer.decode([i for i, l in zip(ids, labels) if l != -100])
        assert trained.strip() == note.strip(), sample_id
        text = tokenizer.decode(ids)
        assert text.count(prompts.PLAN_TAG) == 1 and text.endswith(note)
        system = text.split("<|im_end|>")[0]
        for app in prompts.note_apps(note, catalog) - {"sys"}:
            assert f"{app}." in system, (sample_id, app)
