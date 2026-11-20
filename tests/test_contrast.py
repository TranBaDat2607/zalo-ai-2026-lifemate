"""Pairs inside SFT (lifemate.contrast + train.backward_loss): the contrast sits at the first token where the gold step
and the wrong step part, inside the conversation the SFT loss already scores, and a few steps widen its margin."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate import contrast, prompts  # noqa: E402

TOK = lambda t: [ord(c) % 128 for c in t]  # noqa: E731
EMPTY = {"name": "contacts.search", "arguments": {}}
NAMED = {"name": "contacts.search", "arguments": {"query": "Ha"}}
CALL = {"name": "phone.call", "arguments": {"contact_id": "c_01"}}
ASK = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "phone.call", "params": ["contact_id"]}}


def _item():
    return {"id": "train_9999", "audio": "a.wav", "audio_text": "Goi chi Ha", "history": [],
            "context": {"current_time": "2026-11-07T09:00:00+07:00", "user_profile": {}, "installed_apps": {
                "contacts": {"tools": ["contacts.search"]}, "phone": {"tools": ["phone.call"]}}},
            "label": {"tool_calls": [NAMED, CALL],
                      "tool_results": [{"ok": True, "contacts": [{"id": "c_01", "name": "Ha"}]}, {"ok": True}]}}


def _catalog():
    def spec(name, kind):
        return {"name": name, "description": name, "kind": kind, "parameters": {"properties": {}}}
    return {"contacts.search": spec("contacts.search", "read"), "phone.call": spec("phone.call", "write"),
            "sys.ask_missing_info": spec("sys.ask_missing_info", "meta")}


TEXTS = {"a.wav": "goi chi ha"}


def _encoded(rename=None):
    return prompts.encode_conversation(_item(), TEXTS, _catalog(), TOK, denoise=True, rename=rename)


def test_positions_are_the_first_differing_gold_token():
    ids, labels = _encoded()
    pairs = [{"id": "train_9999", "step": 0, "chosen": NAMED, "rejected": EMPTY},
             {"id": "train_9999", "step": 1, "chosen": CALL, "rejected": ASK},
             {"id": "train_9999", "step": 2, "chosen": contrast.DONE, "rejected": CALL},
             {"id": "train_9999", "step": 1, "chosen": NAMED, "rejected": ASK}]  # not this step's gold: skipped
    found, skipped = contrast.locate(ids, labels, _item(), pairs, TOK, denoise=True)
    assert skipped == 1 and len(found) == 3
    text = "".join(chr(i) for i in ids)
    for (p, c, r, _), (shared, gold_char, wrong_char) in zip(found, [
            ('<tool_call>\n{"name":"contacts.search","arguments":{', '"', "}"),
            ('<tool_call>\n{"name":"', "p", "s"),
            ("", prompts.DONE[0], "<")]):
        assert ids[p] == c and labels[p] == c         # a gold token the SFT loss already scores
        assert chr(c) == gold_char and chr(r) == wrong_char
        assert text[:p].endswith(shared)
    # the first pair sits inside the first turn, after the restated request of the think block
    assert prompts.think_lead("Goi chi Ha") in text[:found[0][0]]


def test_renamed_conversation_uses_the_renaming_tokenizer():
    rename = prompts.renamer({"contacts.search": "zuxo.kepa", "contacts": "zuxo"})
    ids, labels = _encoded(rename)
    pairs = [{"id": "train_9999", "step": 0, "chosen": NAMED, "rejected": EMPTY}]
    found, skipped = contrast.locate(ids, labels, _item(), pairs, lambda t: TOK(rename(t)), denoise=True)
    assert skipped == 0 and len(found) == 1
    found_plain, skipped_plain = contrast.locate(ids, labels, _item(), pairs, TOK, denoise=True)
    assert skipped_plain == 1  # with the wrong tokenizer the gold turn is not found: skipped, never misplaced


@pytest.fixture(scope="module")
def policy():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from lifemate.train import _TARGETS
    torch.manual_seed(0)
    config = Qwen3_5Config(
        text_config=dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
                         linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
                         layer_types=["linear_attention"] * 3 + ["full_attention"], full_attention_interval=4,
                         mtp_num_hidden_layers=0, tie_word_embeddings=False),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           deepstack_visual_indexes=[]))
    model = Qwen3_5ForConditionalGeneration(config)
    return get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=_TARGETS,
                                            task_type="CAUSAL_LM"))


def test_zero_weight_is_plain_sft_and_contrast_widens_margin(policy):
    from lifemate.train import backward_loss
    ids, labels = _encoded()
    found, _ = contrast.locate(ids, labels, _item(), [{"id": "train_9999", "step": 0, "chosen": NAMED,
                                                       "rejected": EMPTY}], TOK, denoise=True)
    ids_t, labels_t = torch.tensor([ids]), torch.tensor([labels])
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    plain = backward_loss(policy, ids_t, labels_t, scaler, 1.0)
    policy.zero_grad(set_to_none=True)
    off = backward_loss(policy, ids_t, labels_t, scaler, 1.0, contrasts=found, pair_weight=0.0)
    policy.zero_grad(set_to_none=True)
    assert abs(plain - off) < 1e-6
    params = [p for p in policy.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=5e-3)
    margins = []
    for _ in range(12):
        stats = {}
        backward_loss(policy, ids_t, labels_t, scaler, 1.0, contrasts=found, pair_weight=1.0, stats=stats)
        optim.step()
        optim.zero_grad(set_to_none=True)
        margins.append(stats["margin"] / stats["n"])
    assert margins[-1] > margins[0] + 0.5
