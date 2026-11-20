"""asr.AdapterSwap: after use(k) the weights equal PEFT's merge of adapter k alone (asr.merge_adapters), bit for bit,
also after switching back and forth (no drift), in float32 and with float16 base weights (the GPU case)."""
import pytest
import torch

from lifemate import asr

TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


def tiny_whisper(seed: int, dtype):
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    torch.manual_seed(seed)
    config = WhisperConfig(vocab_size=64, num_mel_bins=8, d_model=16, encoder_layers=1, decoder_layers=1,
                           encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
                           max_source_positions=16, max_target_positions=16, pad_token_id=0, bos_token_id=1,
                           eos_token_id=2, decoder_start_token_id=1)
    return WhisperForConditionalGeneration(config).to(dtype).eval()


def save_adapter(base_dir, out_dir, seed: int):
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration
    torch.manual_seed(seed)
    model = WhisperForConditionalGeneration.from_pretrained(base_dir)
    config = LoraConfig(r=2, lora_alpha=4, target_modules=TARGETS, init_lora_weights=False)  # random B: a real delta
    get_peft_model(model, config).save_pretrained(out_dir)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_swap_equals_single_merge(tmp_path, dtype):
    from transformers import WhisperForConditionalGeneration
    tiny_whisper(0, torch.float32).save_pretrained(tmp_path / "base")
    paths = []
    for k in range(3):
        save_adapter(tmp_path / "base", tmp_path / f"a{k}", seed=10 + k)
        paths.append(str(tmp_path / f"a{k}"))

    def fresh():
        return WhisperForConditionalGeneration.from_pretrained(tmp_path / "base", dtype=dtype).eval()

    swap = asr.AdapterSwap(fresh(), paths)
    assert not any("lora" in n for n, _ in swap.model.named_parameters())
    assert swap.extra_parameters == sum(p.numel() for k in range(3) for p in _adapter_tensors(paths[k]))
    for k in (2, 0, 1, 2, 0):
        swap.use(k)
        ref = asr.merge_adapters(fresh(), [paths[k]])
        got = dict(swap.model.named_parameters())
        for name, p in ref.named_parameters():
            assert torch.equal(got[name], p), (k, name)
    base = dict(fresh().named_parameters())
    changed = [n for n, p in swap.model.named_parameters() if not torch.equal(p, base[n])]
    assert changed and all(n.rsplit(".", 2)[-2] in TARGETS for n in changed)


def _adapter_tensors(path):
    from safetensors.torch import load_file
    return list(load_file(f"{path}/adapter_model.safetensors").values())
