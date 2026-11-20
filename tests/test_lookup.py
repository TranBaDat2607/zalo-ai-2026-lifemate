"""Prompt-lookup decoding (--lookup): the same greedy tokens as the one-token loop, on a tiny random Qwen3.5.

The draft only decides how many tokens one forward pass checks; it never decides a token. So any draft, good or bad,
must give exactly the plain loop's tokens, and the cache must hold exactly `fed` tokens after every turn.
"""
import random
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

VOCAB = 512


class _Tok:
    def __init__(self, stop):
        self.stop = stop

    def convert_tokens_to_ids(self, token):
        return self.stop


@pytest.fixture(scope="module")
def model():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    torch.manual_seed(0)
    config = Qwen3_5Config(
        text_config=dict(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
                         linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
                         layer_types=["linear_attention"] * 3 + ["full_attention"], full_attention_interval=4,
                         mtp_num_hidden_layers=0, tie_word_embeddings=True),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           deepstack_visual_indexes=[]))
    m = Qwen3_5ForConditionalGeneration(config).eval()
    # A random model's greedy output soon loops; sharper logits (the head is tied to the embeddings) make it wander, so
    # drafts are both right and wrong: 576 tokens took 343 passes, 233 guessed tokens kept, 38 rollbacks (8 Nov).
    with torch.no_grad():
        m.get_input_embeddings().weight.mul_(8.0)
    return m


def _gen(model, stop=VOCAB - 1, lookup=0):
    from lifemate.infer import Generator
    return Generator(_Tok(stop), model, max_new_tokens=48, lookup=lookup)


def _turns(gen, prompt, tails, limit=0):
    """Several turns on one cache, as run_agent does: each turn's tokens plus a new tail go into the next prompt."""
    ids, cache, fed, out = list(prompt), None, 0, []
    for tail in tails:
        new, cache, fed = gen.generate(ids, cache, fed, limit)
        out.append(new)
        ids = ids + new + list(tail)
        assert cache.get_seq_length() == fed  # the attention caches hold exactly the tokens fed so far
    return out


def _prompts(seed=1):
    rng = random.Random(seed)
    out = []
    for _ in range(4):
        base = [rng.randrange(VOCAB - 1) for _ in range(rng.randrange(20, 60))]
        out.append(base + base[5:25] + [rng.randrange(VOCAB - 1) for _ in range(7)])  # a repeated span to look up
    return out


TAILS = [[3, 4, 5], [9, 9, 10, 11], [2]]


def test_lookup_gives_the_plain_greedy_tokens(model):
    for prompt in _prompts():
        plain = _turns(_gen(model), prompt, TAILS)
        fast_gen = _gen(model, lookup=6)
        assert _turns(fast_gen, prompt, TAILS) == plain
    assert fast_gen.lookup_stats["rounds"] > 0


def test_any_draft_gives_the_plain_tokens(model, monkeypatch):
    """Drafts from the plain answer, corrupted at random places: every pass checks, rolls back and goes on."""
    from lifemate import infer
    for prompt in _prompts(2):
        plain = _turns(_gen(model), prompt, TAILS)
        answer = [t for turn in plain for t in turn]
        rng = random.Random(3)

        def draft(self, k):
            if k <= 0 or rng.random() < 0.2:
                return []
            # where the plain answer continues from here (the draft index knows the tokens so far)
            done = len(self.tokens) - len(prompt)
            guess = list(answer[max(0, done):max(0, done) + k]) or [rng.randrange(VOCAB)]
            if rng.random() < 0.6:
                guess[rng.randrange(len(guess))] = rng.randrange(VOCAB)
            return guess

        monkeypatch.setattr(infer.NgramDraft, "draft", draft)
        fast_gen = _gen(model, lookup=5)
        # the tails make later turns' positions differ from `answer`; wrong guesses there are fine, they are checked
        assert _turns(fast_gen, prompt, TAILS) == plain
        assert fast_gen.lookup_stats["rollbacks"] > 0 and fast_gen.lookup_stats["accepted"] > 0
        monkeypatch.undo()


def test_stop_token_and_token_limit(model):
    prompt = _prompts(4)[0]
    plain = _turns(_gen(model), prompt, [[1]])[0]
    stop = plain[10]
    plain_stop = _turns(_gen(model, stop=stop), prompt, TAILS)
    assert plain_stop[0][-1] == stop and len(plain_stop[0]) <= 11
    assert _turns(_gen(model, stop=stop, lookup=8), prompt, TAILS) == plain_stop
    # a turn's token limit is kept exactly
    for limit in (1, 2, 7):
        assert _turns(_gen(model, lookup=8), prompt, TAILS, limit) == _turns(_gen(model), prompt, TAILS, limit)


def test_ngram_draft_index():
    from lifemate.infer import NgramDraft
    d = NgramDraft([1, 2, 3, 4, 1, 2])
    assert d.draft(3) == [3, 4, 1]          # after the latest earlier "1 2"
    d.extend([9])
    assert d.draft(2) == []                 # "9" never came before
    d.extend([1])
    assert d.draft(2) == [2, 9]             # the latest "1" with a next token is the one at position 4
    assert NgramDraft([5, 6, 7]).draft(4) == [] and NgramDraft([5, 5]).draft(0) == []
