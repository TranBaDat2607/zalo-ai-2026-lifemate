"""E1 (made-up tool names in training prompts) and the E2 data file."""
import json
import random
from pathlib import Path

from test_parse import _load

PKG = Path(__file__).resolve().parents[1] / "src" / "lifemate"


def _fixture():
    catalog = {
        "a.read": {"name": "a.read", "description": "Đọc", "parameters": {"properties": {}}},
        "a.write": {"name": "a.write", "description": "Ghi", "parameters": {"properties": {
            "item_id": {"type": "string", "description": "ID mục, lấy từ a.read"}, "n": {"type": "integer"}}, "required": ["item_id", "n"]}},
        "b.x": {"name": "b.x", "description": "Khác", "parameters": {"properties": {}}},
        "c.y": {"name": "c.y", "description": "Không cài", "parameters": {"properties": {}}},
        "sys.ask_missing_info": {"name": "sys.ask_missing_info", "description": "Hỏi", "parameters": {"properties": {}}},
    }
    item = {"id": "t", "audio": "x.wav", "audio_text": "gold", "history": [
        {"role": "user", "audio": "h.wav", "audio_text": "trước"},
        {"role": "assistant", "text": "Dạ.", "tool_calls": [{"name": "a.read", "arguments": {}}], "tool_results": [{"ok": True, "items": [{"id": "i1"}]}]}],
        "context": {"current_time": "2026-11-16T17:00:00+07:00", "user_profile": {},
                    "installed_apps": {"a": {"tools": ["a.read", "a.write"]}, "b": {"tools": ["b.x"]}}},
        "label": {"tool_calls": [{"name": "a.write", "arguments": {"item_id": "i1"}},
                                 {"name": "sys.ask_missing_info", "arguments": {"tool_name": "a.write", "params": ["n"]}}],
                  "tool_results": [{"ok": True}, {"ok": True}]}}
    return catalog, item


def test_made_up_names_replace_every_mention_consistently():
    prompts = _load("prompts")
    catalog, item = _fixture()
    mapping = prompts.masked_names(item, catalog, random.Random(0))
    assert set(mapping) == {"a.read", "a.write", "b.x"}  # installed apps only, never sys
    assert len({v.split(".")[0] for v in mapping.values()}) == 2 and len(set(mapping.values())) == 3
    assert mapping["a.read"].split(".")[0] == mapping["a.write"].split(".")[0]  # one new word per app
    tok = lambda text: list(text)  # noqa: E731
    texts = {"x.wav": "noisy", "h.wav": "trước"}
    ids, labels = prompts.encode_conversation(item, texts, catalog, tok, rename=prompts.renamer(mapping))
    plain, plain_labels = prompts.encode_conversation(item, texts, catalog, tok)
    assert len(ids) == len(labels)
    text = "".join(ids)
    for real in ("a.read", "a.write", "b.x"):
        assert real not in text
        assert mapping[real] in text
    assert "(từ " + mapping["a.read"] + ")" in text  # a description that points to another tool follows the new name
    assert '{"name":"' + mapping["a.read"] + '","arguments":{}}' in text  # history call
    target = "".join(i for i, lab in zip(ids, labels) if lab != -100)
    assert '"tool_name":"' + mapping["a.write"] + '"' in target and '{"name":"sys.ask_missing_info"' in target
    plain_target = "".join(i for i, lab in zip(plain, plain_labels) if lab != -100)
    assert target == prompts.renamer(mapping)(plain_target)
    assert "sys.ask_missing_info" in text and "c.y" not in text


def test_renamer_keeps_name_boundaries():
    prompts = _load("prompts")
    rename = prompts.renamer({"alarm.list": "ko.ba", "alarm.list_all": "ko.mi"})
    assert rename("alarm.list(x) alarm.list_all() (từ alarm.list). xalarm.list") == "ko.ba(x) ko.mi() (từ ko.ba). xalarm.list"
    assert prompts.renamer({}) is None


def test_held_app_synth_file():
    """The bundled E2 file: valid rows that only add held-out apps' conversations and never reuse a real sample id."""
    path = PKG / "synth_held.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    held = set(json.loads((PKG / "splits.json").read_text(encoding="utf-8"))["held_apps"])
    assert len(rows) > 200 and len({r["id"] for r in rows}) == len(rows)
    assert all(r["id"].startswith("synth_held_") for r in rows)
    for r in rows:
        calls = r["label"]["tool_calls"]
        assert len(calls) == len(r["label"]["tool_results"])
        apps = {c["name"].split(".")[0] for c in calls} | {c["arguments"]["tool_name"].split(".")[0] for c in calls if c["name"] == "sys.ask_missing_info"}
        assert not calls or apps & held
        for c in calls:
            if c["name"] == "sys.ask_missing_info":
                assert c["arguments"]["params"] == sorted(c["arguments"]["params"])
