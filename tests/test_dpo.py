"""Step-preference training (lifemate.dpo): the pair layout, the reference = the policy at step 0, and that a few
steps on a tiny random Qwen3.5 move the preference the right way."""
import copy
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate import dpo, prompts  # noqa: E402

TOK = lambda t: [ord(c) % 128 for c in t]  # noqa: E731  (the tiny model has 128 tokens)
FOLD = lambda t: "".join(chr(i) for i in TOK(t))  # noqa: E731  (text as TOK sees it: non-ASCII folded)
SEARCH = {"name": "contacts.search", "arguments": {}}
SEARCH_NAMED = {"name": "contacts.search", "arguments": {"query": "Minh"}}
CALL = {"name": "phone.call", "arguments": {"contact_id": "c_01"}}


def _item():
    return {"id": "train_9999", "audio": "a.wav", "audio_text": "Gọi chị Hà",
            "history": [],
            "context": {"current_time": "2026-11-07T09:00:00+07:00", "user_profile": {}, "installed_apps": {
                "contacts": {"tools": ["contacts.search"]}, "phone": {"tools": ["phone.call"]}}},
            "label": {"tool_calls": [SEARCH, CALL],
                      "tool_results": [{"ok": True, "contacts": [{"id": "c_01", "name": "Hà"}]}, {"ok": True}]}}


def _catalog():
    def spec(name, kind):
        return {"name": name, "description": name, "kind": kind, "parameters": {"properties": {}}}
    return {"contacts.search": spec("contacts.search", "read"), "phone.call": spec("phone.call", "write"),
            "sys.ask_missing_info": spec("sys.ask_missing_info", "meta")}


TEXTS = {"a.wav": "gọi chị hà"}


def test_pair_layout_first_step():
    ids_c, ids_r, start = dpo.encode_pair(_item(), TEXTS, _catalog(), TOK, "Gọi chị Hà.", 0, SEARCH, SEARCH_NAMED)
    shared = "".join(chr(i) for i in ids_c[:start])
    # the restated request sits in the shared part: never a target
    assert FOLD(prompts.think_lead("Gọi chị Hà.")) in shared
    assert shared.endswith('<tool_call>\n{"name":"contacts.search","arguments":{')
    assert ids_c[:start] == ids_r[:start] and ids_c[start] != ids_r[start]
    assert ids_c[-len(TOK("<|im_end|>")):] == TOK("<|im_end|>")


def test_pair_layout_later_step_has_gold_history():
    ids_c, ids_r, start = dpo.encode_pair(_item(), TEXTS, _catalog(), TOK, "Gọi chị Hà.", 1, CALL, dpo.DONE)
    text = "".join(chr(i) for i in ids_c[:start])
    assert FOLD(prompts.call_text("contacts.search", {})) in text            # gold call 0 is in the prefix
    assert FOLD(prompts.result_text(_item()["label"]["tool_results"][0])) in text  # with its gold result
    assert text.endswith(prompts.GEN_PREFIX)                           # and the next turn starts after it
    rejected_tail = "".join(chr(i) for i in ids_r[start:])
    assert rejected_tail == prompts.DONE + "<|im_end|>"


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


def _policy(base):
    from peft import LoraConfig, get_peft_model
    from lifemate.train import _TARGETS
    return get_peft_model(copy.deepcopy(base), LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, target_modules=_TARGETS,
                                                          task_type="CAUSAL_LM"))


def test_reference_equals_policy_at_start(tiny):
    model = _policy(tiny)
    ids_c, ids_r, start = dpo.encode_pair(_item(), TEXTS, _catalog(), TOK, "Gọi chị Hà.", 1, CALL, dpo.DONE)
    ids = torch.tensor([ids_c])
    with torch.no_grad():
        lp, n = dpo.seq_logprob(model, ids, start)
        with model.disable_adapter():
            ref, _ = dpo.seq_logprob(model, ids, start)
    assert n == len(ids_c) - start
    assert torch.allclose(lp, ref)
    loss, d, s, margin = dpo.pair_loss(lp, lp, ref.item(), ref.item(), n, beta=0.1, sft_weight=1.0)
    assert abs(d.item() - math.log(2)) < 1e-6 and abs(margin.item()) < 1e-6


def test_steps_move_preference_toward_chosen(tiny):
    model = _policy(tiny)
    model.train()
    ids_c, ids_r, start = dpo.encode_pair(_item(), TEXTS, _catalog(), TOK, "Gọi chị Hà.", 1, CALL, dpo.DONE)
    c, r = torch.tensor([ids_c]), torch.tensor([ids_r])
    with torch.no_grad(), model.disable_adapter():
        ref_c, _ = dpo.seq_logprob(model, c, start)
        ref_r, _ = dpo.seq_logprob(model, r, start)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=1e-2)
    for _ in range(15):
        lp_c, n_c = dpo.seq_logprob(model, c, start)
        lp_r, _ = dpo.seq_logprob(model, r, start)
        loss, _, _, margin = dpo.pair_loss(lp_c, lp_r, ref_c.item(), ref_r.item(), n_c, beta=0.1, sft_weight=1.0)
        optim.zero_grad()
        loss.backward()
        optim.step()
    with torch.no_grad():
        lp_c, _ = dpo.seq_logprob(model, c, start)
        lp_r, _ = dpo.seq_logprob(model, r, start)
    assert (lp_c - ref_c).item() > 0          # the chosen step became more likely (the SFT term keeps it up)
    assert (lp_c - ref_c - (lp_r - ref_r)).item() > 0  # and more likely relative to the rejected one


def test_main_end_to_end_on_tiny_model(tiny, tmp_path, monkeypatch):
    """The whole trainer (pairs file -> encode -> reference -> steps -> saved adapter -> merge on top of the soup), with
    the real LLM swapped for the tiny one; catches wiring slips before a GPU session is spent on them."""
    import json
    from lifemate import data, infer
    from lifemate.export_models import merge_mean

    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": TOK(text)}

    pairs = [{"id": "train_9999", "step": 1, "chosen": CALL, "rejected": dpo.DONE, "restated": "Gọi chị Hà."},
             {"id": "train_9999", "step": 0, "chosen": SEARCH, "rejected": SEARCH_NAMED, "restated": "Gọi chị Hà."},
             {"id": "train_0001", "step": 0, "chosen": SEARCH, "rejected": CALL, "restated": "x"}]  # excluded below
    pairs_file = tmp_path / "pairs.jsonl"
    pairs_file.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
    monkeypatch.setattr(infer, "load_adapted", lambda llm, paths, weights: (Tok(), copy.deepcopy(tiny)))
    monkeypatch.setattr(infer, "resolve_adapter", lambda a: a)
    monkeypatch.setattr(infer, "load_transcripts", lambda spec: TEXTS)
    monkeypatch.setattr(data, "find_root", lambda split: tmp_path)
    monkeypatch.setattr(data, "load_split", lambda split, root: [_item()])
    monkeypatch.setattr(data, "load_catalog", lambda split, root: _catalog())
    monkeypatch.setattr(data, "id_set", lambda spec: {"train_0001"})
    out = tmp_path / "out"
    metrics = dpo.main(["--out", str(out), "--adapter", "a,b", "--pairs", str(pairs_file), "--transcripts", "x",
                        "--exclude", "splits", "--epochs", "4", "--grad-accum", "2", "--lr", "1e-2", "--lora-r", "4"])
    assert metrics["pairs"] == 2 and metrics["steps"] == metrics["total_steps"] == 4
    assert json.loads((out / "metrics.json").read_text())["pairs_seen"] == 8
    last = metrics["history"][-1]
    assert last["step"] == 4 and last["margin"] > 0 and last["dc"] > 0  # chosen up, preference toward chosen
    merged = merge_mean(copy.deepcopy(tiny), [str(out / "adapter")], [1.0])  # loads like any adapter
    assert any(not torch.equal(a, b) for a, b in zip(merged.state_dict().values(), tiny.state_dict().values()))


def test_pair_backward_matches_autograd(tiny):
    """pair_backward (one side at a time, as checkpointing needs) gives the gradient of pair_loss exactly."""
    ids_c, ids_r, start = dpo.encode_pair(_item(), TEXTS, _catalog(), TOK, "Gọi chị Hà.", 0, SEARCH, SEARCH_NAMED)
    c, r = torch.tensor([ids_c]), torch.tensor([ids_r])
    model = _policy(tiny)
    torch.manual_seed(3)
    for name, p in model.named_parameters():
        if "lora_B" in name:
            p.data.normal_(0, 0.05)  # away from the start, so the DPO and SFT parts both have a gradient
    ref_c, ref_r, beta, w = -250.0, -120.0, 0.1, 1.0
    params = [p for p in model.parameters() if p.requires_grad]

    lp_c, n_c = dpo.seq_logprob(model, c, start)  # plain autograd, both graphs kept (no checkpointing here)
    lp_r, _ = dpo.seq_logprob(model, r, start)
    dpo.pair_loss(lp_c, lp_r, ref_c, ref_r, n_c, beta, w)[0].backward()
    want = [p.grad.clone() for p in params]
    model.zero_grad()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    loss, _, _, margin, got_c, _ = dpo.pair_backward(model, c, r, start, ref_c, ref_r, beta, w)
    got = [p.grad for p in params]
    assert abs(got_c - lp_c.item()) < 1e-3 and abs(margin - beta * ((lp_c.item() - ref_c) - (lp_r.item() - ref_r))) < 1e-4
    for a, b in zip(want, got):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-7)


def test_load_pairs_auto_finds_the_file(tmp_path, monkeypatch):
    """auto:NAME reads the file in the folder data.walk_input yields (it yields folders, not files)."""
    from lifemate import data
    (tmp_path / "pairs_folds.jsonl").write_text('{"id": "train_0001"}\n\n{"id": "train_0002"}\n', encoding="utf-8")
    monkeypatch.setattr(data, "walk_input", lambda name: iter([tmp_path]))
    assert [p["id"] for p in dpo.load_pairs("auto:pairs_folds.jsonl")] == ["train_0001", "train_0002"]
