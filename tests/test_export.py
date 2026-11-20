"""The seed soup of export_models: the merged model must carry the exact mean of the LoRA deltas."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=16)
    return Qwen3ForCausalLM(cfg)


def _adapter(tmp: Path, seed: int) -> str:
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    model = get_peft_model(_tiny(), LoraConfig(r=4, lora_alpha=4, target_modules=["q_proj", "down_proj"]))
    for name, p in model.named_parameters():
        if "lora_B" in name:  # B starts at zero; a trained adapter has some delta
            torch.nn.init.normal_(p, std=0.5)
    model.save_pretrained(tmp / f"a{seed}")
    return str(tmp / f"a{seed}")


def test_merge_mean_is_the_mean_of_the_merged_models(tmp_path):
    from peft import PeftModel
    from lifemate.export_models import merge_mean
    paths = [_adapter(tmp_path, 1), _adapter(tmp_path, 2)]
    singles = [PeftModel.from_pretrained(_tiny(), p).merge_and_unload().state_dict() for p in paths]
    soup = merge_mean(_tiny(), paths).state_dict()
    base = _tiny().state_dict()
    assert set(soup) == set(base)
    changed = 0
    for key, value in soup.items():
        expected = (singles[0][key] + singles[1][key]) / 2
        assert torch.allclose(value, expected, atol=1e-5), key
        changed += not torch.equal(value, base[key])
    assert changed == 4  # q_proj and down_proj of both layers


def test_infer_loads_one_adapter_or_the_soup_of_several(tmp_path, monkeypatch):
    from lifemate import infer, models
    from lifemate.export_models import merge_mean
    base = tmp_path / "base"
    _tiny().save_pretrained(base)
    paths = [_adapter(tmp_path, 1), _adapter(tmp_path, 2)]
    monkeypatch.setattr(models, "AutoTokenizer", type("T", (), {"from_pretrained": staticmethod(lambda name: None)}))
    _, one = infer.load_adapted(str(base), paths[0])
    _, expected_one = models.load_llm(str(base), paths[0])
    _, soup = infer.load_adapted(str(base), ",".join(paths))
    expected_soup = merge_mean(_tiny(), paths)
    for got, want in ((one, expected_one), (soup, expected_soup)):
        for (k, v), (k2, w) in zip(got.state_dict().items(), want.state_dict().items()):
            assert k == k2 and torch.allclose(v, w, atol=1e-6), k
    assert infer.resolve_adapter(paths[0]) == paths[0]  # a path stays as it is
