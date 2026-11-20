"""Stage 3 hard-example speech training (asr_train): key words, their token weights, the misheard-key check and the
weighted loss."""
from pathlib import Path

import pytest
import torch

from lifemate import asr_train

PROCESSOR = Path(__file__).resolve().parents[1] / "outputs/asr-avg5-full-stage/asr_medium/full/best"
ITEM = {"history": [{"role": "user", "audio": "u1.wav", "audio_text": "Mai tôi có báo thức nào?"},
                    {"role": "assistant", "text": "", "tool_calls": [{"name": "alarm.delete", "arguments": {"alarm_id": "alm_01"}}]}],
        "audio": "u2.wav", "audio_text": "Gọi cho Quyên lúc 6 giờ",
        "label": {"tool_calls": [{"name": "phone.call", "arguments": {"contact": "chị Quyên"}}]}}


def test_key_words_and_spans():
    keys = asr_train.key_words(ITEM)
    assert {"quyên", "chị", "giờ", "mai"} <= keys and "alm_01" not in keys and "gọi" not in keys
    text = ITEM["audio_text"]
    assert [text[s:e] for s, e in asr_train.key_spans(text, keys)] == ["Quyên", "6", "giờ"]


def test_misheard_key():
    keys, text = asr_train.key_words(ITEM), ITEM["audio_text"]
    assert asr_train.misheard_key(text, "Gọi cho Quyền lúc 6 giờ", keys)  # a name replaced
    assert asr_train.misheard_key(text, "Gọi cho Quyên lúc giờ", keys)  # a digit dropped
    assert not asr_train.misheard_key(text, "gọi cho quyên lúc 6 giờ nhé", keys)  # case and an added word only
    assert not asr_train.misheard_key(text, "Gọi chơ Quyên lúc 6 giờ", keys)  # only a non-key word wrong


def test_weighted_loss_equals_model_loss_with_unit_weights():
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    torch.manual_seed(0)
    config = WhisperConfig(vocab_size=64, num_mel_bins=8, d_model=16, encoder_layers=1, decoder_layers=1,
                           encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
                           max_source_positions=16, max_target_positions=16, pad_token_id=0, bos_token_id=1,
                           eos_token_id=2, decoder_start_token_id=1)
    model = WhisperForConditionalGeneration(config).eval()
    feats = torch.randn(2, 8, 32)
    labels = torch.tensor([[5, 6, 7, 2], [8, 9, 2, -100]])
    with torch.no_grad():
        out = model(input_features=feats, labels=labels)
        plain = asr_train.weighted_loss(out.logits, labels, torch.ones(labels.shape))
        assert torch.allclose(plain, out.loss, atol=1e-6)
        heavy = torch.tensor([[1.0, 3.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
        ce = torch.nn.functional.cross_entropy(out.logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
        assert torch.allclose(asr_train.weighted_loss(out.logits, labels, heavy), (ce * heavy).sum() / 9.0, atol=1e-6)


@pytest.mark.skipif(not PROCESSOR.exists(), reason="needs a PhoWhisper processor folder")
def test_collate_weights_key_tokens(tmp_path):
    import numpy as np
    import soundfile as sf
    from transformers import AutoProcessor
    sf.write(str(tmp_path / "u2.wav"), np.zeros(16000, dtype=np.float32), 16000)
    keys = asr_train.key_words(ITEM)
    spans = {"u2.wav": asr_train.key_spans(ITEM["audio_text"], keys)}
    collate = asr_train.Collate(AutoProcessor.from_pretrained(PROCESSOR), tmp_path, spans, 3.0)
    feats, dec, labels, weights = collate([("u2.wav", ITEM["audio_text"])])
    assert dec is None and weights.shape == labels.shape
    tok = collate.processor.tokenizer
    heavy = "".join(tok.convert_tokens_to_string([tok.convert_ids_to_tokens(i)]) for i, w in
                    zip(labels[0].tolist(), weights[0].tolist()) if w == 3.0)
    assert heavy.split() == ["Quyên", "6", "giờ"]
    assert set(weights[0].tolist()) == {1.0, 3.0}
    plain = asr_train.Collate(AutoProcessor.from_pretrained(PROCESSOR), tmp_path)([("u2.wav", ITEM["audio_text"])])
    assert plain[3] is None and torch.equal(plain[2], labels)
