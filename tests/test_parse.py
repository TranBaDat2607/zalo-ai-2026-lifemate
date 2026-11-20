import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "lifemate"


def _load(name):
    """The prompt and parsing code needs no torch, so load it directly and keep these tests fast on any machine."""
    spec = importlib.util.spec_from_file_location(f"lifemate.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _infer():
    import sys
    import types
    pkg = sys.modules.setdefault("lifemate", types.ModuleType("lifemate"))
    pkg.__path__ = [str(ROOT)]
    return importlib.import_module("lifemate.infer")


import importlib  # noqa: E402


def test_parse_call_with_and_without_closing_tag():
    infer = _infer()
    assert infer.parse_call('<tool_call>\n{"name":"a.b","arguments":{"n":1}}\n</tool_call><|im_end|>') == ("a.b", {"n": 1})
    assert infer.parse_call('<tool_call>\n{"name":"a.b","arguments":{}}') == ("a.b", {})


def test_parse_call_rejects_text_and_malformed():
    infer = _infer()
    assert infer.parse_call("Xong.<|im_end|>") is None
    assert infer.parse_call("<tool_call>{oops</tool_call>") is None


def test_merge_readings_drops_duplicates_and_keeps_order():
    infer = _infer()
    assert infer.merge_readings(["Bật đèn nhé.", "bật đèn nhé", "bật điện nhé"]) == "Bật đèn nhé. | bật điện nhé"
    assert infer.merge_readings(["", "chào"]) == "chào"


def test_tidy_sorts_missing_info_params():
    infer = _infer()
    assert infer.tidy("sys.ask_missing_info", {"tool_name": "a.b", "params": ["z", "a"]})["params"] == ["a", "z"]
    assert infer.tidy("a.b", {"params": ["z", "a"]})["params"] == ["z", "a"]


def test_tidy_puts_arguments_in_gold_form():
    infer = _infer()
    catalog = {"a.b": {"parameters": {"properties": {
        "when": {"type": "string", "description": "ISO-8601 đầy đủ giây"}, "at": {"type": "string", "format": "date-time"},
        "n": {"type": "integer"}, "code": {"type": "string"}, "note": {"type": "string"}}}}}
    out = infer.tidy("a.b", {"when": "2026-11-11T05:30", "at": "2026-11-11T05:30:00+07:00", "n": "5", "code": 12, "note": ""}, catalog)
    assert out == {"when": "2026-11-11T05:30:00+07:00", "at": "2026-11-11T05:30:00+07:00", "n": 5, "code": "12"}


class _ScriptedGen:
    """Stands in for the model: each generate() returns the next scripted turn, one token per character."""
    max_new_tokens = 256
    stop_id = -1

    def __init__(self, turns):
        self.turns = list(turns)
        self.tokenizer = self

    def encode(self, text):
        return [0] * len(text)

    def decode(self, ids, skip_special_tokens=False):
        return self._last

    def generate(self, ids, cache, fed):
        self._last = self.turns.pop(0)
        return [1, self.stop_id], cache, fed


class _RecordingEnv:
    def __init__(self):
        self.calls = []

    def validate(self, name, arguments):
        return {"ok": True, "error": None}

    def call(self, name, arguments):
        self.calls.append(name)
        return {"ok": True, "items": [{"id": "x_1"}]}


def _agent(turns):
    import time
    infer = _infer()
    catalog = {"a.write": {"parameters": {"properties": {"id": {"type": "string", "x-ref": "a.items"}}}}}
    env = _RecordingEnv()
    stats = infer.run_agent(_ScriptedGen(turns), env, "prompt", time.monotonic() + 60, 8, catalog=catalog)
    return env, stats


def _call(name, **arguments):
    import json
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


def test_agent_blocks_an_id_nothing_has_shown():
    env, stats = _agent([_call("a.write", id="x_9"), _call("a.read"), _call("a.write", id="x_1"), "Xong."])
    assert env.calls == ["a.read", "a.write"]
    assert stats["repairs"] == 1 and stats["stop"] == "done"


def test_agent_stops_after_asking_for_missing_info():
    env, stats = _agent([_call("sys.ask_missing_info", tool_name="a.write", params=["id"]), _call("a.read")])
    assert env.calls == ["sys.ask_missing_info"]
    assert stats["stop"] == "asked"


def test_agent_stops_on_a_repeated_call():
    env, stats = _agent([_call("a.read"), _call("a.read"), "Xong."])
    assert env.calls == ["a.read"]
    assert stats["stop"] == "repeat"


def test_tool_line_is_compact_and_keeps_requirements():
    prompts = _load("prompts")
    tool = {"name": "sms.send", "description": "Gửi SMS", "parameters": {
        "properties": {"contact_id": {"type": "string", "description": "ID liên hệ"}, "n": {"type": "integer"},
                       "mode": {"type": "string", "enum": ["a", "b"]},
                       "when": {"type": "string", "description": "ISO-8601 đầy đủ giây và offset múi giờ, ví dụ x"}},
        "required": ["n"], "x-one-of": [["contact_id"], ["when"]]}}
    assert prompts.tool_line(tool) == ("sms.send(contact_id (ID liên hệ), n*:int, mode:a|b, when:datetime) - Gửi SMS"
                                       " Bắt buộc một trong: contact_id / when.")


def test_step_text_is_a_prefix_of_the_next_prompt():
    prompts = _load("prompts")
    step = prompts.step_text("a.b", {"x": "é"}, {"ok": True})
    assert step.startswith(prompts.GEN_PREFIX + '<tool_call>\n{"name":"a.b","arguments":{"x":"é"}}\n</tool_call><|im_end|>\n')
    assert step.endswith('</tool_response><|im_end|>\n')


def test_encode_conversation_labels_only_assistant_turns():
    prompts = _load("prompts")
    catalog = {"a.b": {"name": "a.b", "description": "x", "parameters": {"properties": {}}}, "sys.ask_missing_info": {"name": "sys.ask_missing_info", "description": "y", "parameters": {"properties": {}}}}
    item = {"id": "t", "audio": "x.wav", "audio_text": "gold words", "history": [],
            "context": {"current_time": "2026-11-16T17:00:00+07:00", "user_profile": {}, "installed_apps": {"a": {"tools": ["a.b"]}}},
            "label": {"tool_calls": [{"name": "a.b", "arguments": {}}], "tool_results": [{"ok": True}]}}
    tok = lambda text: list(text)  # one token per character is enough to check the alignment
    ids, labels = prompts.encode_conversation(item, {"x.wav": "noisy words"}, catalog, tok, denoise=True)
    assert len(ids) == len(labels)
    target = "".join(i for i, l in zip(ids, labels) if l != -100)
    assert target == 'gold words\n</think>\n\n<tool_call>\n{"name":"a.b","arguments":{}}\n</tool_call><|im_end|>Xong.<|im_end|>'
    prompt = "".join(ids)
    assert prompt.endswith(prompts.GEN_PREFIX + "Xong.<|im_end|>")
    assert "<|im_start|>user\nnoisy words<|im_end|>\n<|im_start|>assistant\n<think>\ngold words" in prompt
