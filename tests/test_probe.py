"""Step probing (lifemate.probe): answers leave the prefix as it was, greedy answers and gold log-probs match a plain
uncached forward pass, and the prefix at every step is the one dpo.encode_pair builds."""
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate import dpo, probe, prompts  # noqa: E402

STOP = 3


class Tok:
    """Characters as tokens (the tiny model has 128); '<|im_end|>' is token STOP."""

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            if text.startswith("<|im_end|>", i):
                ids.append(STOP)
                i += len("<|im_end|>")
            else:
                ids.append(ord(text[i]) % 128 if ord(text[i]) % 128 != STOP else 4)
                i += 1
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        return "".join("<|im_end|>" if i == STOP else chr(i) for i in ids)

    def convert_tokens_to_ids(self, token):
        return STOP


TOK = lambda t: Tok()(t)["input_ids"]  # noqa: E731
SEARCH = {"name": "contacts.search", "arguments": {"query": "Ha"}}
CALL = {"name": "phone.call", "arguments": {"contact_id": "c_01"}}


def _item():
    return {"id": "train_9999", "audio": "a.wav", "audio_text": "Goi chi Ha", "history": [],
            "context": {"current_time": "2026-11-07T09:00:00+07:00", "user_profile": {}, "installed_apps": {
                "contacts": {"tools": ["contacts.search"]}, "phone": {"tools": ["phone.call"]}}},
            "label": {"tool_calls": [SEARCH, CALL],
                      "tool_results": [{"ok": True, "contacts": [{"id": "c_01", "name": "Ha"}]}, {"ok": True}]}}


def _catalog():
    def spec(name, kind):
        return {"name": name, "description": name, "kind": kind, "parameters": {"properties": {}}}
    return {"contacts.search": spec("contacts.search", "read"), "phone.call": spec("phone.call", "write"),
            "sys.ask_missing_info": spec("sys.ask_missing_info", "meta")}


TEXTS = {"a.wav": "goi chi ha"}


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    torch.manual_seed(0)
    config = Qwen3_5Config(
        text_config=dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
                         linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
                         layer_types=["linear_attention"] * 3 + ["full_attention"], full_attention_interval=4,
                         mtp_num_hidden_layers=0, tie_word_embeddings=False),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           deepstack_visual_indexes=[]))
    return Qwen3_5ForConditionalGeneration(config).eval()


def _prober(model, max_new=6):
    return probe.Prober(Tok(), model, max_new, temperature=0.8, top_p=0.95, seed=1)


def _full_logits(model, ids):
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids])).logits[0].float()


def test_restated_and_answer_parsing():
    assert probe.restated_of("Goi chi Ha.\n</think>\n\n<tool_call>x") == "Goi chi Ha."
    assert prompts.think_lead(probe.restated_of("Goi chi Ha.\n</think>\n\n")) == "Goi chi Ha.\n</think>\n\n"
    cat = _catalog()
    text = "Goi.\n</think>\n\n" + prompts.call_text("phone.call", {"contact_id": "c_01"})
    assert probe.answer_of(text, cat) == CALL
    assert probe.answer_of(prompts.DONE, cat) == probe.DONE
    assert probe.answer_of('<tool_call>\n{"name": broken', cat) is None


def test_answer_leaves_prefix_and_greedy_matches_uncached(tiny):
    p = _prober(tiny)
    prefix = TOK("hello there, " * 3)
    p.feed(prefix)
    last_before, length_before = p.last.clone(), p.length
    rng = torch.Generator().manual_seed(5)
    p.answer(rng)                       # a sampled answer, then
    tokens, lps = p.answer()            # the greedy one: both from the same prefix
    assert p.length == length_before and torch.equal(p.last, last_before)
    ids = list(prefix)
    for token in tokens:                # greedy, recomputed without any cache
        assert token == int(_full_logits(tiny, ids)[-1].argmax())
        ids.append(token)
    # after the answers, the cache still continues the prefix exactly
    nxt = TOK("abc")
    p.feed(nxt)
    assert torch.allclose(p.last, _full_logits(tiny, prefix + nxt)[-1], atol=1e-4)


def test_score_matches_uncached_logprob(tiny):
    p = _prober(tiny)
    prefix, cont = TOK("prefix text "), TOK("call<|im_end|>")
    p.feed(prefix)
    got = p.score(cont, start=1)
    full = torch.log_softmax(_full_logits(tiny, prefix + cont), -1)
    want = sum(float(full[len(prefix) + i - 1, cont[i]]) for i in range(1, len(cont)))
    assert math.isclose(got, want, abs_tol=1e-3)


def test_probe_rows_and_prefix_match_dpo_pairs(tiny):
    p = _prober(tiny)
    item, cat = _item(), _catalog()
    rows = probe.probe_sample(p, item, TEXTS, cat, samples=2, sample_seed=7)
    assert [r["step"] for r in rows] == [0, 1, 2]
    assert rows[0]["gold"] == SEARCH and rows[2]["gold"] == probe.DONE
    assert all(len(r["answers"]) == 3 and math.isfinite(r["lp_gold"]) for r in rows)
    assert "restated" in rows[0]["answers"][0] and "restated" not in rows[1]["answers"][0]
    restated = rows[0]["restated"]
    # The prefix fed up to the last step is the chosen side of the pair dpo.encode_pair builds for that step, so pairs
    # from the probe rows train on exactly the state that was probed.
    ids_c, _, _ = dpo.encode_pair(item, TEXTS, cat, TOK, restated, 2, probe.DONE, CALL)
    assert p.fed == ids_c
    # The same holds for the first step (minus the restated request, which the model generates there).
    ids_c0, _, start0 = dpo.encode_pair(item, TEXTS, cat, TOK, restated, 0, SEARCH, CALL)
    first = TOK(prompts.prompt_text(item, TEXTS, cat) + prompts.THINK_OPEN)
    assert ids_c0[:len(first)] == first


def test_sampling_is_seeded(tiny):
    p = _prober(tiny)
    p.feed(TOK("same prefix "))
    a = p.answer(torch.Generator().manual_seed(11))[0]
    b = p.answer(torch.Generator().manual_seed(11))[0]
    assert a == b


def test_dpo_strips_the_restated_newline():
    item, cat = _item(), _catalog()
    a = dpo.encode_pair(item, TEXTS, cat, TOK, "Goi chi Ha.\n", 0, SEARCH, CALL)
    b = dpo.encode_pair(item, TEXTS, cat, TOK, "Goi chi Ha.", 0, SEARCH, CALL)
    assert a == b


def _pairs_module():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import probe_pairs
    return probe_pairs


def test_mistake_kinds():
    pp, cat = _pairs_module(), _catalog()
    calls = [SEARCH, CALL]
    assert pp.mistake_kind(SEARCH, SEARCH, 0, calls, cat) is None
    assert pp.mistake_kind(CALL, SEARCH, 0, calls, cat) is None              # a later gold call: another order
    assert pp.mistake_kind(SEARCH, CALL, 1, calls, cat) == "repeated call"   # an earlier one: a repeat
    assert pp.mistake_kind(probe.DONE, CALL, 1, calls, cat) == "stopped early"
    assert pp.mistake_kind(CALL, probe.DONE, 2, calls, cat) == "extra call after finishing"
    assert pp.mistake_kind(probe.DONE, probe.DONE, 2, calls, cat) is None
    assert pp.mistake_kind(None, CALL, 1, calls, cat) is None
    ask = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "phone.call", "params": ["contact_id"]}}
    assert pp.mistake_kind(ask, CALL, 1, calls, cat) == "asked instead of acting"
    other = {"name": "contacts.search", "arguments": {"query": "Minh"}}
    assert pp.mistake_kind(other, SEARCH, 0, calls, cat) == "wrong arguments"


def test_noise_and_value_filters():
    pp, cat = _pairs_module(), _catalog()
    cat["calendar.create_event"] = {"name": "calendar.create_event", "kind": "write", "parameters": {"properties": {
        "title": {"type": "string"}, "start": {"type": "string"}, "end": {"type": "string"}}}}
    rates = {("calendar.create_event", "end"): 0.6, ("contacts.search", "query"): 0.88}
    with_end = {"name": "calendar.create_event", "arguments": {"title": "x", "start": "2026-10-05T19:00:00+07:00",
                                                               "end": "2026-10-05T20:00:00+07:00"}}
    no_end = {"name": "calendar.create_event", "arguments": {"title": "x", "start": "2026-10-05T19:00:00+07:00"}}
    assert pp.noise_or_value(with_end, no_end, cat, rates, 0.2, 0.8) == "gold noise"       # gold uses end 60 %
    empty = {"name": "contacts.search", "arguments": {}}
    assert pp.noise_or_value(empty, SEARCH, cat, rates, 0.2, 0.8) == ""                    # query 88 %: a rule
    shifted = {**no_end, "arguments": {**no_end["arguments"], "start": "2026-10-05T07:00:00+07:00"}}
    assert pp.noise_or_value(no_end, shifted, cat, rates, 0.2, 0.8) == "values"
