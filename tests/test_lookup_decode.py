"""asr.lookup_decode gives greedy_decode's tokens whatever the draft, in fewer decoder passes when the draft is right."""
import random

import torch

from lifemate import asr


def tiny_whisper():
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    torch.manual_seed(0)
    config = WhisperConfig(vocab_size=64, num_mel_bins=8, d_model=32, encoder_layers=2, decoder_layers=2,
                           encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64,
                           max_source_positions=16, max_target_positions=40, pad_token_id=0, bos_token_id=1,
                           eos_token_id=2, decoder_start_token_id=1)
    return WhisperForConditionalGeneration(config).eval()


def test_lookup_equals_greedy():
    model = tiny_whisper()
    prompt, eos = [1, 5, 6], 2
    rng = random.Random(0)
    for seed in range(6):
        torch.manual_seed(100 + seed)
        feats = torch.randn(1, 8, 32)
        limit = 20
        ref = asr.greedy_decode(model, feats, prompt, eos, [limit])[0]
        ref = ref[:ref.index(eos) + 1] if eos in ref else ref
        wrong = list(ref)
        wrong[len(wrong) // 2] = (wrong[len(wrong) // 2] + 7) % 64
        for name, draft in (("same", ref), ("one wrong", wrong), ("empty", []),
                            ("shifted", ref[1:]), ("random", [rng.randrange(64) for _ in range(20)])):
            got, passes = asr.lookup_decode(model, feats, prompt, eos, limit, draft)
            assert got == ref, (seed, name)
            if name == "same":
                assert passes <= 2, passes
            if name == "empty":
                assert passes == len(ref)


def test_aligner_follows_the_draft():
    a = asr.DraftAligner([10, 11, 12, 13, 11, 12, 14, 2])
    assert a.propose([], 3) == [10, 11, 12]
    a.accept(3)
    assert a.propose([10, 11, 12], 4) == [13, 11, 12, 14]
    a.accept(1)
    assert a.propose([10, 11, 12, 13, 11, 12], 5) == [14, 2]  # the repeated "11 12" nearest the current position
    assert a.propose([99], 3) == []
