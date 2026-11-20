"""Beam search of the speech model (improve_plan.md step 1c): one beam is the greedy decode, more beams give distinct
hypotheses ranked by length-normalised log-probability, and the best of a wide beam scores at least as well as greedy."""
import pytest
import torch

from lifemate import asr

from test_adapter_swap import tiny_whisper


def features(seed: int, frames: int = 32):
    torch.manual_seed(seed)
    return torch.randn(1, 8, frames)


def seq_logp(model, feats, prompt, tokens, eos):
    """Summed log-probability of ``tokens`` + end token after ``prompt`` (teacher forcing)."""
    ids = torch.tensor([list(prompt) + list(tokens)])
    with torch.no_grad():
        out = model(input_features=feats, decoder_input_ids=ids)
    logp = torch.log_softmax(out.logits[0].float(), -1)
    targets = list(tokens) + [eos]
    return sum(logp[len(prompt) - 1 + i, t].item() for i, t in enumerate(targets))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_one_beam_is_greedy(seed):
    model = tiny_whisper(seed, torch.float32)
    feats, prompt, eos = features(seed), [1, 5], 2
    greedy = asr.greedy_decode(model, feats, prompt, eos, [9])[0]
    greedy = greedy[:greedy.index(eos)] if eos in greedy else greedy
    (tokens, _), = asr.beam_decode(model, feats, prompt, eos, 9, beams=1, keep=1)
    assert tokens == greedy


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_beams_are_distinct_ranked_and_scored_right(seed):
    model = tiny_whisper(seed, torch.float32)
    feats, prompt, eos = features(seed), [1, 5], 2
    hyps = asr.beam_decode(model, feats, prompt, eos, 9, beams=5, keep=5)
    assert 1 <= len(hyps) <= 5
    assert len({tuple(t) for t, _ in hyps}) == len(hyps)
    assert [s for _, s in hyps] == sorted((s for _, s in hyps), reverse=True)
    for tokens, score in hyps:
        assert eos not in tokens and len(tokens) <= 9
        if len(tokens) < 9:  # finished with an end token: the score is the teacher-forced one, normalised
            assert score == pytest.approx(seq_logp(model, feats, prompt, tokens, eos) / (len(tokens) + 1), abs=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_wide_beam_is_not_worse_than_greedy(seed):
    model = tiny_whisper(seed, torch.float32)
    feats, prompt, eos = features(seed), [1, 5], 2
    (g, gs), = asr.beam_decode(model, feats, prompt, eos, 9, beams=1, keep=1)
    best = asr.beam_decode(model, feats, prompt, eos, 9, beams=8, keep=1)[0]
    assert best[1] >= gs - 1e-6 or len(g) == 9
