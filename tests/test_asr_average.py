"""The averaged speech model (log.md, 5 Nov 2026): adapters joined by "+" must give the exact mean of their deltas."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _tiny_whisper():
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    torch.manual_seed(0)
    cfg = WhisperConfig(vocab_size=64, num_mel_bins=8, d_model=16, encoder_layers=1, decoder_layers=1,
                        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
                        max_source_positions=8, max_target_positions=16, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                        decoder_start_token_id=1)
    return WhisperForConditionalGeneration(cfg)


def _adapter(tmp: Path, seed: int) -> str:
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    # alpha / r = 2, as in the real adapters (r 32, alpha 64): a wrong scaling in the average would show
    model = get_peft_model(_tiny_whisper(), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "fc1"]))
    for name, p in model.named_parameters():
        if "lora_B" in name:  # B starts at zero; a trained adapter has some delta
            torch.nn.init.normal_(p, std=0.5)
    model.save_pretrained(tmp / f"w{seed}")
    return str(tmp / f"w{seed}")


def _single(path: str) -> dict:
    from peft import PeftModel
    return PeftModel.from_pretrained(_tiny_whisper(), path).merge_and_unload().state_dict()


def test_average_of_three_whisper_adapters_is_the_mean_of_the_merged_models(tmp_path):
    from lifemate.asr import merge_adapters
    paths = [_adapter(tmp_path, s) for s in (1, 2, 3)]
    singles = [_single(p) for p in paths]
    base = _tiny_whisper().state_dict()
    avg = merge_adapters(_tiny_whisper(), paths).state_dict()
    assert set(avg) == set(base)
    changed = 0
    for key, value in avg.items():
        assert torch.allclose(value, sum(s[key] for s in singles) / 3, atol=1e-5), key
        changed += not torch.equal(value, base[key])
    # q_proj: encoder self, decoder self, decoder cross attention; v_proj the same; fc1 in both layers
    assert changed == 8


def test_one_adapter_and_the_same_adapter_twice(tmp_path):
    from lifemate.asr import merge_adapters
    path = _adapter(tmp_path, 1)
    single = _single(path)
    one = merge_adapters(_tiny_whisper(), [path]).state_dict()
    twice = merge_adapters(_tiny_whisper(), [path, path]).state_dict()
    for key in single:
        assert torch.equal(one[key], single[key]), key
        assert torch.allclose(twice[key], single[key], atol=1e-6), key
    base = _tiny_whisper().state_dict()
    plain = merge_adapters(_tiny_whisper(), []).state_dict()
    assert all(torch.equal(plain[k], base[k]) for k in base)
