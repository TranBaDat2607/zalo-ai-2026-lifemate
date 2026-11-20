"""Weight-space edit programs (lifemate.wsp): an empty program is today's soup exactly, a program scales the averaged
LoRA change of exactly the selected modules, and the teacher-forced probe reads one conversation."""
import copy
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_adapter_weights import _deltas, adapters, base  # noqa: E402,F401  (fixtures)


def _merged(base, paths, program=None, weights=None):
    from lifemate.wsp import load_edited
    return load_edited(copy.deepcopy(base), paths, weights, program)


def _group(name):
    from lifemate.wsp import module_group
    return module_group(name.removesuffix(".weight"), 4)


def test_selectors():
    from lifemate.wsp import factor_for, module_group, parse_program
    assert parse_program("none") == {} and parse_program("") == {}
    with pytest.raises(ValueError):
        parse_program('{"lm_head": 2}')
    p = parse_program('{"all": 1.2, "mlp:late": 1.5, "L0": 0.5}')
    assert factor_for(p, 31, "mlp", "late") == pytest.approx(1.8)
    assert factor_for(p, 31, "attn", "late") == pytest.approx(1.2)
    assert factor_for(p, 0, "attn", "early") == pytest.approx(0.6)
    assert module_group("model.language_model.layers.21.mlp.down_proj", 32) == (21, "mlp", "mid")
    assert module_group("model.language_model.layers.22.linear_attn.out_proj", 32) == (22, "attn", "late")
    assert module_group("model.language_model.layers.10.self_attn.q_proj", 32) == (10, "attn", "early")
    assert module_group("lm_head", 32) is None


def test_empty_program_is_the_soup(base, adapters):
    from lifemate.export_models import merge_mean
    want = merge_mean(copy.deepcopy(base), adapters[:2]).state_dict()
    got = _merged(base, adapters[:2], "none").state_dict()
    for n in want:
        assert torch.allclose(got[n], want[n], atol=1e-6), n


def test_scaled_groups(base, adapters):
    deltas = _deltas(base, adapters[:2])
    ref = base.state_dict()
    got = _merged(base, adapters[:2], {"all": 1.5, "mlp:late": 2.0}).state_dict()
    checked = {"mlp-late": 0, "other": 0}
    for n in ref:
        mean = 0.5 * (deltas[0][n] + deltas[1][n])
        if torch.count_nonzero(mean) == 0:
            assert torch.equal(got[n], ref[n]), n
            continue
        g = _group(n)
        f = 1.5 * (2.0 if g and g[1] == "mlp" and g[2] == "late" else 1.0)
        checked["mlp-late" if f == 3.0 else "other"] += 1
        err = (got[n] - ref[n] - f * mean).abs().max().item() / mean.abs().max().item()
        assert err < 1e-4, (n, f, err)
    assert checked["mlp-late"] >= 3 and checked["other"] > 10


def test_probe_scores(base, adapters):
    from lifemate.wsp import apply_program, gold_scores, weighted_peft
    model = weighted_peft(copy.deepcopy(base), adapters[:2]).eval()
    ids = torch.randint(0, 128, (1, 40))
    labels = ids.clone()
    labels[0, :25] = -100
    a = gold_scores(model.get_base_model(), ids, labels)
    apply_program(model, {"all": 3.0})
    b = gold_scores(model.get_base_model(), ids, labels)
    apply_program(model, {})
    c = gold_scores(model.get_base_model(), ids, labels)
    assert a["n"] == 15 and 0 <= a["hit"] <= 15 and a["nll"] > 0
    assert a["nll"] != b["nll"] and a["nll"] == pytest.approx(c["nll"], rel=1e-5)  # scaling is undone exactly


def test_programs_inline_or_file(tmp_path):
    """A long inline JSON list must not be taken for a path (Errno 36 on Kaggle, 11 Nov)."""
    import json
    from lifemate.wsp import load_programs
    progs = [{"all": 1.0 + k / 100} for k in range(40)]
    assert load_programs(json.dumps(progs)) == progs
    f = tmp_path / "p.json"
    f.write_text(json.dumps(progs[:3]))
    assert load_programs(str(f)) == progs[:3]


def _whisper(layers=3):
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    torch.manual_seed(0)
    cfg = WhisperConfig(vocab_size=64, num_mel_bins=8, d_model=16, encoder_layers=layers, decoder_layers=layers,
                        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
                        max_source_positions=8, max_target_positions=16, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                        decoder_start_token_id=1)
    return WhisperForConditionalGeneration(cfg)


def _whisper_adapter(tmp, seed):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    # alpha / r = 2 and the target modules of asr_train, as in the real speech adapters
    model = get_peft_model(_whisper(), LoraConfig(r=4, lora_alpha=8,
                                                  target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.5)
    model.save_pretrained(tmp / f"w{seed}")
    return str(tmp / f"w{seed}")


def test_whisper_parts_and_kinds():
    from lifemate.wsp import module_group, module_part, parse_program
    assert module_part("model.encoder.layers.3.self_attn.q_proj") == "enc"
    assert module_part("model.decoder.layers.3.encoder_attn.q_proj") == "dec"  # cross-attention sits in the decoder
    assert module_part("model.language_model.layers.3.mlp.up_proj") == ""
    assert module_group("model.decoder.layers.23.encoder_attn.k_proj", 24) == (23, "xattn", "late")
    assert module_group("model.encoder.layers.0.fc2", 24) == (0, "mlp", "early")
    assert module_group("model.decoder.layers.10.self_attn.out_proj", 24) == (10, "attn", "mid")
    assert parse_program('{"enc": 0.8, "dec:xattn": 1.2, "enc:late": 1.1, "xattn": 1.0}')
    with pytest.raises(ValueError):
        parse_program('{"enc:dec": 2}')


def test_whisper_edit_scales_exactly_the_selected_part(tmp_path):
    """The empty program is today's averaged speech model; {"enc": 0.5, "dec:xattn": 2} scales the mean delta of
    encoder modules by 0.5, of decoder cross-attention by 2, and leaves decoder self-attention and fc layers alone."""
    from lifemate.asr import merge_adapters
    from lifemate.wsp import load_edited, module_group, module_part
    paths = [_whisper_adapter(tmp_path, s) for s in (1, 2)]
    ref = _whisper().state_dict()
    avg = merge_adapters(_whisper(), paths).state_dict()
    empty = load_edited(_whisper(), paths, None, {}).state_dict()
    for n in avg:
        assert torch.allclose(empty[n], avg[n], atol=1e-6), n
    got = load_edited(_whisper(), paths, None, {"enc": 0.5, "dec:xattn": 2.0}).state_dict()
    seen = {0.5: 0, 2.0: 0, 1.0: 0}
    for n in ref:
        delta = avg[n] - ref[n]
        if torch.count_nonzero(delta) == 0:
            assert torch.equal(got[n], ref[n]), n
            continue
        name = n.removesuffix(".weight")
        g = module_group(name, 3)
        f = 0.5 if module_part(name) == "enc" else (2.0 if g[1] == "xattn" else 1.0)
        seen[f] += 1
        err = (got[n] - ref[n] - f * delta).abs().max().item() / delta.abs().max().item()
        assert err < 1e-4, (n, f, err)
    # encoder: 3 layers x (q, k, v, out, fc1, fc2); decoder: 3 x (q, k, v, out) cross; 3 x (q, k, v, out, fc1, fc2) other
    assert seen == {0.5: 18, 2.0: 12, 1.0: 18}
