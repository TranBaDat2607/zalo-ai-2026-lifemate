"""Agent rules (wider read after an empty search, ask bias), the job deadline and the time-aware schedules."""
import time
import types
from pathlib import Path

import pytest

from test_parse import _call, _infer, _load, _ScriptedGen

SRC = Path(__file__).resolve().parents[1] / "src"


_CATALOG = {
    "contacts.search": {"kind": "read", "parameters": {"properties": {"query": {"type": "string"}}}},
    "news.search": {"kind": "read", "parameters": {"properties": {"query": {"type": "string"}}, "required": ["query"]}},
    "cal.list": {"kind": "read", "parameters": {"properties": {
        "date": {"type": "string", "format": "date"}, "title": {"type": "string"}, "status": {"type": "string", "enum": ["a", "b"]},
        "when": {"type": "string", "description": "ISO-8601 đầy đủ giây"}, "n": {"type": "integer"},
        "event_id": {"type": "string", "x-ref": "cal.list"}}}},
    "notes.add": {"kind": "write", "parameters": {"properties": {"title": {"type": "string"}}}},
    "ewallet.transfer": {"kind": "write", "parameters": {"properties": {
        "contact_id": {"type": "string", "x-ref": "contacts.search"}, "amount": {"type": "integer"}}, "required": ["contact_id", "amount"]}},
    "lib.search": {"kind": "read", "parameters": {"properties": {
        "query": {"type": "string"}, "branch": {"type": "string", "enum": ["A", "B", "C"]}}}},
}


def test_name_params_are_free_text_strings_only():
    infer = _infer()
    assert infer.name_params(_CATALOG["cal.list"]) == {"title"}
    assert infer.name_params(_CATALOG["contacts.search"]) == {"query"}


def test_broader_call_only_drops_optional_name_filters():
    infer = _infer()
    empty = {"ok": True, "contacts": []}
    assert infer.broader_call("contacts.search", {"query": "Cô Mai"}, empty, _CATALOG) == {}
    assert infer.broader_call("contacts.search", {"query": "Cô Mai"}, {"ok": True, "contacts": [{"id": "c_1"}]}, _CATALOG) is None
    assert infer.broader_call("contacts.search", {"query": "Cô Mai"}, {"ok": False, "error": "x"}, _CATALOG) is None
    assert infer.broader_call("contacts.search", {}, empty, _CATALOG) is None
    assert infer.broader_call("news.search", {"query": "giá vàng"}, {"ok": True, "articles": []}, _CATALOG) is None
    events = {"ok": True, "events": []}
    assert infer.broader_call("cal.list", {"date": "2026-11-04", "title": "họp", "status": "a"}, events, _CATALOG) == {"date": "2026-11-04", "status": "a"}
    assert infer.broader_call("cal.list", {"date": "2026-11-04"}, events, _CATALOG) is None  # "nothing tomorrow" is an answer
    assert infer.broader_call("notes.add", {"title": "x"}, {"ok": True, "items": []}, _CATALOG) is None


class _SearchEnv:
    """A search finds nothing while it has a query, and lists everything without one, as substring search does."""

    def __init__(self):
        self.calls = []

    def validate(self, name, arguments):
        return {"ok": True, "error": None}

    def call(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if name in ("contacts.search", "lib.search"):
            return {"ok": True, "items": [] if arguments.get("query") else [{"contact_id": "c_06", "name": "Cô giáo Mai"}]}
        return {"ok": True}


class _TextGen(_ScriptedGen):
    """Also keeps every text the loop feeds the model."""

    def __init__(self, turns):
        super().__init__(turns)
        self.fed = []

    def encode(self, text):
        self.fed.append(text)
        return [0] * len(text)


def _search_agent(turns, broaden):
    infer = _infer()
    env, gen = _SearchEnv(), _TextGen(turns)
    stats = infer.run_agent(gen, env, "prompt", time.monotonic() + 60, 8, catalog=_CATALOG, broaden=broaden)
    return env, gen, stats


def test_agent_reads_wider_after_an_empty_name_search():
    prompts = _load("prompts")
    turns = [_call("contacts.search", query="Cô Mai"), _call("ewallet.transfer", contact_id="c_06", amount=5000), "Xong."]
    env, gen, stats = _search_agent(turns, broaden=True)
    assert env.calls == [("contacts.search", {"query": "Cô Mai"}), ("contacts.search", {}),
                         ("ewallet.transfer", {"contact_id": "c_06", "amount": 5000})]
    assert stats["broadened"] == 1 and stats["steps"] == 3 and stats["stop"] == "done"
    assert stats["repairs"] == 0  # the ID the wider read showed counts as seen
    inserted = gen.fed[1]
    assert prompts.step_text("contacts.search", {}, {"ok": True, "items": [{"contact_id": "c_06", "name": "Cô giáo Mai"}]}) in inserted
    assert inserted.endswith(prompts.GEN_PREFIX)


def test_agent_without_broadening_keeps_the_empty_result():
    turns = [_call("contacts.search", query="Cô Mai"), _call("ewallet.transfer", contact_id="c_06", amount=5000), "Xong."]
    env, _, stats = _search_agent(turns, broaden=False)
    assert env.calls == [("contacts.search", {"query": "Cô Mai"})]
    assert stats["broadened"] == 0 and stats["repairs"] == 1


def test_agent_broadens_at_most_twice_and_never_repeats_a_wider_read():
    turns = [_call("lib.search", query="x", branch="A"), _call("lib.search", query="y", branch="A"),
             _call("lib.search", query="y", branch="B"), _call("lib.search", query="z", branch="C"), "Xong."]
    env, _, stats = _search_agent(turns, broaden=True)
    wider = [args for name, args in env.calls if "query" not in args]
    assert wider == [{"branch": "A"}, {"branch": "B"}]
    assert stats["broadened"] == 2 and len(env.calls) == 6


class _Vocab:
    """A tokenizer that splits a call the way Qwen3.5 does: '{"', 'name', '":"', then the first piece of the name."""
    pieces = ["<|im_end|>", '{"', "name", '":"', "sys", ".ask", "_missing", "_info", '"', "ew", "allet", "x"]

    def __call__(self, text, add_special_tokens=False):
        ids, at = [], 0
        while at < len(text):
            n = max((i for i, p in enumerate(self.pieces) if text.startswith(p, at)), key=lambda i: len(self.pieces[i]))
            ids.append(n)
            at += len(self.pieces[n])
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces[i] for i in ids)

    def convert_tokens_to_ids(self, token):
        return self.pieces.index(token)


class _Model:
    """Each forward pass returns the next scripted row of logits."""
    device = "cpu"

    def __init__(self, rows):
        self.rows = list(rows)

    def __call__(self, input_ids, past_key_values, use_cache, logits_to_keep):
        import torch
        return types.SimpleNamespace(logits=torch.tensor([[self.rows.pop(0)]]), past_key_values=None)


def _row(best, second=None, margin=1.0):
    row = [0.0] * len(_Vocab.pieces)
    row[_Vocab.pieces.index(best)] = 10.0
    if second is not None:
        row[_Vocab.pieces.index(second)] = 10.0 - margin
    return row


def _decode_with_bias(rows, bias):
    pytest.importorskip("torch")
    gen = _infer().Generator(_Vocab(), _Model(rows), max_new_tokens=len(rows), ask_bias=bias)
    new, _, _ = gen.generate([11], None, 0)
    return gen.tokenizer.decode(new), gen.flips


def test_ask_bias_flips_a_close_ask_into_the_runner_up_tool():
    rows = [_row('{"'), _row("name"), _row('":"'), _row("sys", "ew", margin=1.0), _row("allet"), _row("<|im_end|>")]
    assert _decode_with_bias(rows, 2.0) == ('{"name":"ewallet<|im_end|>', 1)
    assert _decode_with_bias(rows, 0.0)[0].startswith('{"name":"sys')


def test_stop_at_cuts_a_running_step_after_one_token_and_none_leaves_it_alone():
    pytest.importorskip("torch")
    import time
    rows = [_row("name")] * 5
    gen = _infer().Generator(_Vocab(), _Model(rows), max_new_tokens=5)
    assert len(gen.generate([11], None, 0)[0]) == 5 and gen.cut == 0  # solve.py only: validation never sets stop_at
    gen = _infer().Generator(_Vocab(), _Model(rows), max_new_tokens=5)
    gen.stop_at = time.monotonic() - 1.0
    assert len(gen.generate([11], None, 0)[0]) == 1 and gen.cut == 1


def test_ask_bias_leaves_a_clear_ask_and_other_positions_alone():
    clear = [_row('{"'), _row("name"), _row('":"'), _row("sys", "ew", margin=3.0), _row("<|im_end|>")]
    assert _decode_with_bias(clear, 2.0) == ('{"name":"sys<|im_end|>', 0)
    elsewhere = [_row("sys", "ew", margin=0.5), _row("<|im_end|>")]  # not where a tool name starts
    assert _decode_with_bias(elsewhere, 2.0) == ("sys<|im_end|>", 0)


def test_negative_ask_bias_turns_a_close_act_into_an_ask_and_nothing_else():
    """11 Nov: a boost (negative bias) moves a close choice where a tool name starts to sys.ask_missing_info."""
    close = [_row('{"'), _row("name"), _row('":"'), _row("ew", "sys", margin=1.0), _row(".ask"), _row("<|im_end|>")]
    assert _decode_with_bias(close, -2.0) == ('{"name":"sys.ask<|im_end|>', 1)
    assert _decode_with_bias(close, 0.0) == ('{"name":"ew.ask<|im_end|>', 0)
    clear = [_row('{"'), _row("name"), _row('":"'), _row("ew", "sys", margin=3.0), _row("<|im_end|>")]
    assert _decode_with_bias(clear, -2.0) == ('{"name":"ew<|im_end|>', 0)
    elsewhere = [_row("ew", "sys", margin=0.5), _row("<|im_end|>")]  # not where a tool name starts
    assert _decode_with_bias(elsewhere, -2.0) == ("ew<|im_end|>", 0)


def test_schedules_finish_their_decay_at_the_deadline():
    schedule = _load("schedule")
    assert schedule.cosine(0, 10, 100, 1.0) == pytest.approx(0.1)
    assert schedule.cosine(55, 10, 100, 1.0) == pytest.approx(0.5)
    assert schedule.cosine(80, 10, 100, 1.0) > 0.1  # where exp9 stopped, by steps alone
    assert schedule.cosine(80, 10, 100, 1.0, time_frac=1.0) == pytest.approx(0.0, abs=1e-12)
    assert schedule.cosine(80, 10, 100, 1.0, time_frac=0.1) == schedule.cosine(80, 10, 100, 1.0)  # steps lead
    assert schedule.linear(0, 20, 1000, 1.0) == pytest.approx(0.05)
    assert schedule.linear(500, 20, 1000, 1.0) == pytest.approx(0.5)
    assert schedule.linear(500, 20, 1000, 1.0, time_frac=0.99) == pytest.approx(0.05)  # the floor


def test_jobs_deadline_stops_a_running_step_and_skips_the_rest(tmp_path, monkeypatch):
    jobs = _load("jobs")
    monkeypatch.setenv("PYTHONPATH", str(SRC))
    monkeypatch.setattr(jobs, "START_MARGIN_S", 0)
    never = str(tmp_path / "never")
    chain = [["waitfor", "--path", never, "--minutes", "5"], ["?waitfor", "--path", never, "--minutes", "5"]]
    results, start = {}, time.monotonic()
    jobs.run_chain("0", chain, str(tmp_path), results, deadline=start + 3)
    assert time.monotonic() - start < 30
    first, second = results["0"]
    assert first["deadline"] and first["exit"] not in (0, None)
    assert second == {"module": "waitfor", "exit": None, "minutes": 0.0, "deadline": True}
