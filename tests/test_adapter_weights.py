"""Weighted adapter merges (--adapter-weights): the merged model holds exactly the weighted sum of the LoRA deltas,
negative weights included, on a tiny random Qwen3.5 with the training run's LoRA targets.

Task arithmetic (log.md, 9 Nov 2026): soup + (side-task pair - baseline pair) is a merge with weights
1/3, 1/3, 1/3, 1/2, 1/2, -1/2, -1/2, so a sign or scale slip here would silently give another model.
"""
import copy
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="module")
def base():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    torch.manual_seed(0)
    config = Qwen3_5Config(
        text_config=dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
                         linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
                         layer_types=["linear_attention"] * 3 + ["full_attention"], full_attention_interval=4,
                         mtp_num_hidden_layers=0, tie_word_embeddings=True),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           deepstack_visual_indexes=[]))
    return Qwen3_5ForConditionalGeneration(config).eval()


@pytest.fixture(scope="module")
def adapters(base, tmp_path_factory):
    """Four adapters with different A inits (seeds) and random nonzero B, like independently trained copies."""
    from peft import LoraConfig, get_peft_model
    from lifemate.train import _TARGETS
    folder = tmp_path_factory.mktemp("adapters")
    paths = []
    for k in range(4):
        torch.manual_seed(100 + k)
        m = get_peft_model(copy.deepcopy(base), LoraConfig(r=8, lora_alpha=8, target_modules=_TARGETS, task_type="CAUSAL_LM"))
        for name, p in m.named_parameters():
            if "lora_B" in name:
                p.data.normal_(0, 0.02)
        path = folder / f"a{k}"
        m.save_pretrained(path)
        paths.append(str(path))
    return paths


def _deltas(base, paths):
    from peft import PeftModel
    ref = base.state_dict()
    out = []
    for p in paths:
        merged = PeftModel.from_pretrained(copy.deepcopy(base), p).merge_and_unload().state_dict()
        out.append({n: merged[n] - ref[n] for n in ref})
    return out


def _assert_weighted(base, merged, deltas, weights):
    ref, got = base.state_dict(), merged.state_dict()
    changed = 0
    for n in ref:
        want = sum(w * d[n] for w, d in zip(weights, deltas))
        if torch.count_nonzero(want) == 0:
            assert torch.equal(got[n], ref[n]), n  # untouched weights stay exactly as they were
            continue
        changed += 1
        err = (got[n] - ref[n] - want).abs().max().item() / want.abs().max().item()
        assert err < 1e-4, (n, err)
    assert changed > 10  # the LoRA targets of every layer type were changed


def test_parse_weights():
    from lifemate.export_models import parse_weights
    w = parse_weights("1/3, 1/3,1/3,1/2,1/2,-1/2,-0.5", 7)
    assert w[0] == pytest.approx(1 / 3) and w[-1] == -0.5 and sum(w) == pytest.approx(1.0)
    with pytest.raises(SystemExit):
        parse_weights("1/2,1/2", 3)


def test_negative_weights_give_the_exact_weighted_sum(base, adapters):
    from lifemate.export_models import merge_mean
    weights = [1 / 3, 1 / 2, -1 / 2, 2 / 3]
    merged = merge_mean(copy.deepcopy(base), adapters, weights)
    _assert_weighted(base, merged, _deltas(base, adapters), weights)


def test_default_is_the_plain_mean(base, adapters):
    from lifemate.export_models import merge_mean
    merged = merge_mean(copy.deepcopy(base), adapters[:3])
    _assert_weighted(base, merged, _deltas(base, adapters[:3]), [1 / 3] * 3)


def test_single_adapter_scaled(base, adapters):
    from lifemate.export_models import merge_mean
    merged = merge_mean(copy.deepcopy(base), adapters[:1], [0.5])
    _assert_weighted(base, merged, _deltas(base, adapters[:1]), [0.5])


def test_mixed_ranks_soup_plus_step_adapter(base, adapters, tmp_path):
    """The step-preference stage (lifemate.dpo) adds an r=16 adapter to r=32 soup copies, and inference stacks them with
    --adapter-weights 0.5,0.5,1.0: PEFT's "cat" merge must keep each adapter's own scaling when the ranks differ."""
    from peft import LoraConfig, get_peft_model
    from lifemate.export_models import merge_mean
    from lifemate.train import _TARGETS
    torch.manual_seed(7)
    m = get_peft_model(copy.deepcopy(base), LoraConfig(r=4, lora_alpha=4, target_modules=_TARGETS, task_type="CAUSAL_LM"))
    for name, p in m.named_parameters():
        if "lora_B" in name:
            p.data.normal_(0, 0.02)
    step = str(tmp_path / "step")
    m.save_pretrained(step)
    paths, weights = adapters[:2] + [step], [0.5, 0.5, 1.0]
    merged = merge_mean(copy.deepcopy(base), paths, weights)
    _assert_weighted(base, merged, _deltas(base, paths), weights)
