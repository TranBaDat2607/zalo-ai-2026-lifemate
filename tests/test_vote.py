"""Voting at answer time (lifemate.vote, improve_plan.md step 3) on a tiny random Qwen3.5.

- the hooks turn the soup into each adapter alone (same logits as merging that adapter), and sign 0 is the soup;
- the majority rule;
- the members' caches are put back after every turn, so a voting run continues exactly as a fresh run of the chosen
  turns would.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_lookup import VOCAB, _Tok  # noqa: E402

TARGETS = r"^(?!.*(visual|vision)).*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)$"


class _DecTok(_Tok):
    def decode(self, ids, skip_special_tokens=False):
        return ""  # no tool call in random tokens: every member's key is None, and the soup's turn goes on


def _config():
    from transformers import Qwen3_5Config
    return Qwen3_5Config(
        text_config=dict(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
                         linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
                         layer_types=["linear_attention"] * 3 + ["full_attention"], full_attention_interval=4,
                         mtp_num_hidden_layers=0, tie_word_embeddings=True),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           deepstack_visual_indexes=[]))


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForConditionalGeneration
    root = tmp_path_factory.mktemp("vote")
    torch.manual_seed(0)
    base = Qwen3_5ForConditionalGeneration(_config()).eval()
    with torch.no_grad():
        base.get_input_embeddings().weight.mul_(8.0)
    base.save_pretrained(root / "base")
    for k in range(2):
        torch.manual_seed(10 + k)
        m = Qwen3_5ForConditionalGeneration.from_pretrained(root / "base")
        cfg = LoraConfig(r=4, lora_alpha=8, target_modules=TARGETS, init_lora_weights=False)
        peft = get_peft_model(m, cfg)
        with torch.no_grad():
            for n, p in peft.named_parameters():
                if "lora_B" in n:
                    p.mul_(0.3)  # a real but moderate change
        peft.save_pretrained(root / f"a{k}")
    return root


def _base(root):
    from transformers import Qwen3_5ForConditionalGeneration
    return Qwen3_5ForConditionalGeneration.from_pretrained(root / "base").eval()


def _soup(root, paths):
    from lifemate.export_models import merge_mean
    return merge_mean(_base(root), [str(p) for p in paths]).eval()


def _single(root, path):
    from peft import PeftModel
    return PeftModel.from_pretrained(_base(root), str(path)).merge_and_unload().eval()


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids])).logits[0]


def test_hooks_turn_the_soup_into_each_adapter(saved):
    from lifemate.vote import DeltaSwitch
    paths = [saved / "a0", saved / "a1"]
    soup = _soup(saved, paths)
    ids = list(range(5, 25))
    plain_soup = _logits(soup, ids)
    switch = DeltaSwitch(soup, [str(p) for p in paths])
    assert switch.parameters > 0
    switch.set(0)
    assert torch.equal(_logits(soup, ids), plain_soup)  # hooks off: the merged soup, untouched
    for sign, path in ((1, paths[0]), (-1, paths[1])):
        switch.set(sign)
        assert torch.allclose(_logits(soup, ids), _logits(_single(saved, path), ids), atol=1e-4)
    assert not torch.allclose(_logits(_single(saved, paths[0]), ids), _logits(_single(saved, paths[1]), ids), atol=1e-3)


def test_majority_rule():
    from lifemate.vote import pick
    a, b, c = ("x", "1"), ("y", "2"), ("z", "3")
    assert pick([a, a, a]) == 0
    assert pick([a, b, b]) == 1
    assert pick([b, a, b]) == 0
    assert pick([a, b, c]) == 0  # no majority: the soup
    assert pick([None, b, b]) == 1
    assert pick([a, None, None]) == 1  # two members end the turn without a call: that wins


def _turns(gen, prompt, tails):
    ids, cache, fed, out, proposals = list(prompt), None, 0, [], []
    for tail in tails:
        new, cache, fed = gen.generate(ids, cache, fed)
        out.append(new)
        proposals.append(list(getattr(gen, "proposals", [])))
        ids = ids + new + list(tail)
    return out, proposals


@pytest.mark.parametrize("lookup", [0, 4])
def test_voting_run_continues_like_a_fresh_run_of_the_chosen_turns(saved, lookup):
    """Members differ, the soup's turn goes on (no call in random tokens); every later turn must equal the soup's greedy
    turn computed from scratch on the whole conversation, which fails if any member's cache was not put back."""
    from lifemate.infer import Generator
    from lifemate.vote import DeltaSwitch, VotingGenerator
    paths = [str(saved / "a0"), str(saved / "a1")]
    soup = _soup(saved, paths)
    gen = Generator(_DecTok(VOCAB - 1), soup, max_new_tokens=12, lookup=lookup)
    voter = VotingGenerator(gen, DeltaSwitch(soup, paths), catalog={})
    prompt, tails = list(range(3, 40)), [[7, 8, 9], [11, 12], [13]]
    voted, proposals = _turns(voter, prompt, tails)
    assert voter.votes["turns"] == 3
    ref_model = _soup(saved, paths)
    ref_switch = DeltaSwitch(ref_model, paths)
    ref = Generator(_DecTok(VOCAB - 1), ref_model, max_new_tokens=12, lookup=0)
    ids = list(prompt)
    differ = 0
    for new, props, tail in zip(voted, proposals, tails):
        for sign, proposal in zip((0, 1, -1), props):  # every member, not only the chosen one
            ref_switch.set(sign)
            fresh, _, _ = ref.generate(ids, None, 0)
            assert proposal == fresh
        ref_switch.set(0)
        assert new == props[0]
        differ += len({tuple(p) for p in props}) > 1
        ids = ids + new + list(tail)
    assert differ  # the members really propose different turns
