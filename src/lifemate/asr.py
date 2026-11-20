"""ASR backends behind one interface: ``Backend(name).transcribe(paths, batch_size) -> list[str]``.

Decoding is always greedy so that re-runs give the same text.
"""
import contextlib

import numpy as np
import soundfile as sf
import torch

TAIL_FRAMES = 200  # silence after the longest clip of a batch: with less, some clips never decode an end token
TOKENS_PER_SECOND = 14  # generous for Vietnamese speech; anything longer is the decoder repeating itself
PROMPT_TOKENS = 160  # most context-prompt tokens; Whisper's decoder has 448 positions for prompt and text together


def read_audio(path: str) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != 16000:  # every train and public clip is 16 kHz; a different test clip is resampled rather than lost
        n = int(round(len(audio) * 16000 / rate))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    return audio


def _batches(items: list, size: int):
    """Longest first within the list so padding stays small; callers restore the order."""
    order = sorted(range(len(items)), key=lambda i: -len(items[i]))
    for start in range(0, len(order), size):
        yield order[start:start + size]


def enable_short_context(model) -> None:
    """Let the Whisper encoder take mel features shorter than 30 s.

    Clips here average 5 s, and encoding 30 s of padding is most of the cost of a step. The encoder only insists on the
    full length through a config check and the size of its position table, so both follow the batch length.
    """
    encoder = model.get_encoder()
    if getattr(encoder, "_full_positions", None) is not None:
        return
    encoder._full_positions = encoder.embed_positions
    encoder._full_length = encoder.config.max_source_positions

    def pre_hook(module, args, kwargs):
        features = args[0] if args else kwargs["input_features"]
        n = features.shape[-1] // 2
        if module.embed_positions.num_embeddings != n:
            full = module._full_positions.weight
            module.embed_positions = torch.nn.Embedding(n, full.shape[1], _weight=full[:n].detach(), _freeze=True)
        module.config.max_source_positions = n

    encoder.register_forward_pre_hook(pre_hook, with_kwargs=True)


def restore_full_context(model) -> None:
    """Put the original position table back, so that a saved model is a normal Whisper checkpoint."""
    encoder = model.get_encoder()
    encoder.embed_positions = encoder._full_positions
    encoder.config.max_source_positions = encoder._full_length


def log_mel(feature_extractor, audio: list) -> torch.Tensor:
    """Batch of log-mel features padded to the longest clip plus a short tail, with an even frame count."""
    tail = np.zeros(TAIL_FRAMES * 160, dtype=np.float32)
    feats = feature_extractor([np.concatenate([a, tail]) for a in audio], sampling_rate=16000, padding="longest",
                              truncation=False, return_tensors="pt").input_features
    frames = min(feats.shape[-1], 3000)
    frames -= frames % 2
    return feats[..., :frames]


def token_limits(audio: list) -> list:
    """Most tokens a clip of this length can honestly need, which stops a decoder stuck repeating itself."""
    return [int(len(a) / 16000 * TOKENS_PER_SECOND) + 10 for a in audio]


@torch.inference_mode()
def greedy_decode(model, features: torch.Tensor, prompt: list, eos: int, limits: list) -> list:
    """Plain greedy loop on the decoder. Whisper's own generate() pads everything back to 30 s.

    ``prompt`` is one token list for the whole batch, or one list per clip (a context prompt). Lists of different
    lengths are padded on the left and masked, with positions counted from each clip's first real token, so a clip
    decodes the same in a batch as alone.
    """
    device = features.device
    encoded = model.get_encoder()(features).last_hidden_state
    batch = encoded.shape[0]
    rows = [list(p) for p in prompt] if prompt and isinstance(prompt[0], (list, tuple)) else [list(prompt)] * batch
    width = max(len(r) for r in rows)
    mask = positions = None
    if any(len(r) != width for r in rows):
        mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows], device=device)
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        rows = [[eos] * (width - len(r)) + r for r in rows]
    room = getattr(model.config, "max_target_positions", 448) - width  # the decoder has 448 positions in all
    limits = [max(1, min(n, room)) for n in limits]
    tokens = torch.tensor(rows, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    caps = torch.tensor(limits, device=device)
    out, cache, step = [], None, tokens
    decoder, head = model.get_decoder(), model.proj_out
    for count in range(max(limits)):
        extra = {} if mask is None else {"attention_mask": mask, "position_ids": positions}
        result = decoder(input_ids=step, encoder_hidden_states=encoded, past_key_values=cache, use_cache=True, **extra)
        cache = result.past_key_values
        next_id = head(result.last_hidden_state[:, -1]).argmax(-1)
        next_id = next_id.masked_fill(finished, eos)
        out.append(next_id)
        finished |= (next_id == eos) | (count + 1 >= caps)
        if finished.all():
            break
        step = next_id[:, None]
        if mask is not None:
            mask = torch.cat([mask, mask.new_ones(batch, 1)], dim=1)
            positions = positions[:, -1:] + 1
    return torch.stack(out, dim=1).tolist()


class DraftAligner:
    """Guesses for ``lookup_decode`` from a draft of the same clip: the draft tokens that follow the decoded tail's
    latest n-gram (n from 3 down to 1), taking the occurrence nearest to where the draft was last followed."""

    def __init__(self, draft: list):
        self.draft, self.pos = list(draft), 0

    def propose(self, decoded: list, k: int) -> list:
        if k <= 0:
            return []
        if not decoded:
            return self.draft[:k]
        for n in range(min(3, len(decoded)), 0, -1):
            tail = decoded[-n:]
            ends = [j for j in range(n, len(self.draft)) if self.draft[j - n:j] == tail]
            if ends:
                self.pos = min(ends, key=lambda j: abs(j - self.pos))
                return self.draft[self.pos:self.pos + k]
        return []

    def accept(self, kept: int) -> None:
        self.pos += kept


@torch.inference_mode()
def lookup_decode(model, features: torch.Tensor, prompt: list, eos: int, limit: int, draft: list, window: int = 24) -> tuple:
    """Greedy decoding of one clip that checks guesses taken from ``draft`` (another reading's tokens of the clip) in one
    decoder pass each: the greedy token at every position is computed as usual, the guess is kept up to the first
    position where it differs, and that position's greedy token is added. The same tokens as ``greedy_decode`` up to
    float16 near-ties (a pass over several tokens sums in another order), in a few passes where the readings agree.
    Self-attention only, so rejected guesses are just cut from the cache. Returns (tokens, decoder passes)."""
    encoded = model.get_encoder()(features).last_hidden_state
    decoder, head = model.get_decoder(), model.proj_out
    limit = max(1, min(limit, getattr(model.config, "max_target_positions", 448) - len(prompt)))
    aligner = DraftAligner(draft)
    out, cache, step, passes = [], None, list(prompt), 0
    while len(out) < limit:
        guess = aligner.propose(out, min(window, limit - len(out) - 1))
        result = decoder(input_ids=torch.tensor([step + guess], device=features.device), encoder_hidden_states=encoded,
                         past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        firsts = head(result.last_hidden_state[0, len(step) - 1:]).argmax(-1).tolist()
        passes += 1
        new = []
        for j, token in enumerate(firsts):
            new.append(token)
            if token == eos or j == len(guess) or token != guess[j]:
                break
        kept = len(new) - 1
        out += new
        aligner.accept(kept)
        if len(guess) > kept:  # guessed tokens after the first wrong one are in the cache: cut them
            cache.crop(-(len(guess) - kept))
        step = [new[-1]]
        if new[-1] == eos:
            break
    return out[:limit], passes


@torch.inference_mode()
def beam_decode(model, features: torch.Tensor, prompt: list, eos: int, limit: int, beams: int = 5, keep: int = 5) -> list:
    """Beam search over one clip (``features`` of batch 1): the ``keep`` best finished hypotheses as (tokens, score),
    best first, tokens without the end token, score = summed log-probability / length (the length penalty of 1.0 that
    transformers' generate uses). Stops when ``beams`` hypotheses have finished or at ``limit`` tokens, where the
    unfinished beams count as finished. With ``beams=1`` it is the greedy decode of ``greedy_decode``
    (tests/test_beam_decode.py). Runner-up hypotheses as extra options: improve_plan.md step 1c (HyPoradise)."""
    encoded = model.get_encoder()(features).last_hidden_state
    decoder, head = model.get_decoder(), model.proj_out
    limit = max(1, min(limit, getattr(model.config, "max_target_positions", 448) - len(prompt)))
    device = features.device
    result = decoder(input_ids=torch.tensor([list(prompt)], device=device), encoder_hidden_states=encoded, use_cache=True)
    cache = result.past_key_values
    logp = torch.log_softmax(head(result.last_hidden_state[:, -1]).float(), -1)[0]
    alive, finished = [([], 0.0)], []  # (tokens so far, summed log-probability)
    while True:
        # logp[b, v]: summed log-probability of beam b followed by token v; the best 2 x beams candidates in order
        vocab = logp.shape[-1]
        top = torch.topk(logp.reshape(-1), min(2 * beams, logp.numel()))
        nxt, rows = [], []
        for total, flat in zip(top.values.tolist(), top.indices.tolist()):
            b, token = divmod(flat, vocab)
            tokens = alive[b][0]
            if token == eos:
                if len(finished) < beams:
                    finished.append((tokens, total / (len(tokens) + 1)))
                continue
            nxt.append((tokens + [token], total))
            rows.append(b)
            if len(nxt) == beams:
                break
        if len(finished) >= beams or not nxt or len(nxt[0][0]) >= limit:
            if len(finished) < beams:  # the token limit: unfinished beams count as finished
                finished += [(t, s / len(t)) for t, s in nxt]
            break
        alive = nxt
        index = torch.tensor(rows, device=device)
        cache.reorder_cache(index)
        result = decoder(input_ids=torch.tensor([[t[-1]] for t, _ in alive], device=device),
                         encoder_hidden_states=encoded.expand(len(alive), -1, -1), past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        logp = torch.log_softmax(head(result.last_hidden_state[:, -1]).float(), -1) + torch.tensor(
            [s for _, s in alive], device=device)[:, None]
    finished.sort(key=lambda h: -h[1])
    return finished[:keep]


def merge_adapters(model, adapters: list):
    """``model`` with one LoRA adapter merged in, or with the mean of several adapters' deltas (an average of
    independently trained readers, log.md 5 Nov 2026), exactly as ``export_models.merge_mean`` makes the LLM soup."""
    if not adapters:
        return model
    if len(adapters) == 1:
        from peft import PeftModel
        return PeftModel.from_pretrained(model, adapters[0]).merge_and_unload().eval()
    from .export_models import merge_mean
    return merge_mean(model, adapters).eval()


class Whisper:
    """Any Whisper checkpoint, optionally with a fine-tuned LoRA adapter merged into it. ``adapter`` may join several
    adapters of the same base with "+": their deltas are averaged into one model of the same size."""

    def __init__(self, name: str, adapter: str = None, edit=None, device: str = None):
        """``edit``: a weight-space edit program (lifemate.wsp) applied to the averaged change before it is merged; None
        merges as before. ``device``: e.g. "cuda:1" to keep speech on a second GPU (a Kaggle T4 check only); default
        the LLM's device."""
        from transformers import AutoProcessor, WhisperForConditionalGeneration
        # GPU: float16 as everywhere else. CPU (only a smoke test of the offline package): the LLM's CPU dtype, float32 unless
        # LIFEMATE_CPU_DTYPE says otherwise (bfloat16 lets both models fit in 16 GB of RAM).
        from .models import device_and_dtype
        self.device, self.dtype = device_and_dtype()
        self.device = device or self.device
        adapters = [a for a in (adapter or "").split("+") if a]
        self.processor = AutoProcessor.from_pretrained(adapters[0] if adapters else name)
        self.model = WhisperForConditionalGeneration.from_pretrained(name, dtype=self.dtype).to(self.device).eval()
        if edit is not None and adapters:
            from .wsp import load_edited
            self.model = load_edited(self.model, adapters, None, edit).eval()
        else:
            self.model = merge_adapters(self.model, adapters)
        enable_short_context(self.model)
        tok = self.processor.tokenizer
        tok.set_prefix_tokens(language="vi", task="transcribe")
        self.prompt = tok.prefix_tokens
        self.eos = tok.eos_token_id

    def prompt_tokens(self, text: str = "") -> list:
        """Decoder start for one clip: Whisper's previous-text prompt when there is a context text, then the usual prefix."""
        if not text:
            return list(self.prompt)
        tok = self.processor.tokenizer
        ids = tok.encode(" " + text.strip(), add_special_tokens=False)[:PROMPT_TOKENS]
        return [tok.convert_tokens_to_ids("<|startofprev|>")] + ids + list(self.prompt)

    def transcribe(self, paths: list, batch_size: int = 16, prompts: list = None) -> list:
        """``prompts``: an optional context text per clip ("" for none)."""
        return self.transcribe_ids(paths, batch_size, prompts)[0]

    def transcribe_ids(self, paths: list, batch_size: int = 16, prompts: list = None) -> tuple:
        """(texts, tokens): each clip's decoded tokens up to and including the end token, as well as its text."""
        audio = [read_audio(p) for p in paths]
        starts = [self.prompt_tokens(t) for t in prompts] if prompts else None
        out, tokens = [""] * len(paths), [[] for _ in paths]
        for index in _batches(audio, batch_size):
            part = [audio[i] for i in index]
            feats = log_mel(self.processor.feature_extractor, part).to(self.device, dtype=self.dtype)
            start = [starts[i] for i in index] if starts else self.prompt
            with torch.autocast("cuda", dtype=torch.float16) if str(self.device).startswith("cuda") else contextlib.nullcontext():
                ids = greedy_decode(self.model, feats, start, self.eos, token_limits(part))
            for i, row, text in zip(index, ids, self.processor.batch_decode(ids, skip_special_tokens=True)):
                out[i] = text.strip()
                tokens[i] = row[:row.index(self.eos) + 1] if self.eos in row else row
        return out, tokens


class QwenAsr:
    """Qwen3-ASR native transformers checkpoints (0.6B-hf, 1.7B-hf)."""

    def __init__(self, name: str):
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForMultimodalLM.from_pretrained(name, dtype=torch.float16, device_map={"": "cuda"}).eval()

    @torch.inference_mode()
    def transcribe(self, paths: list, batch_size: int = 16) -> list:
        audio = [read_audio(p) for p in paths]
        out = [""] * len(paths)
        for index in _batches(audio, batch_size):
            inputs = self.processor.apply_transcription_request(
                audio=[audio[i] for i in index], language="vi").to(self.model.device, self.model.dtype)
            ids = self.model.generate(**inputs, max_new_tokens=220, do_sample=False)
            new = ids[:, inputs["input_ids"].shape[1]:]
            for i, text in zip(index, self.processor.decode(new, return_format="transcription_only")):
                out[i] = text.strip()
        return out


class Ctc:
    """Transformers CTC checkpoints, e.g. a BF16 conversion of Parakeet."""

    def __init__(self, name: str):
        from transformers import AutoModelForCTC, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForCTC.from_pretrained(name, dtype=torch.float32).to("cuda").eval()

    @torch.inference_mode()
    def transcribe(self, paths: list, batch_size: int = 16) -> list:
        audio = [read_audio(p) for p in paths]
        out = [""] * len(paths)
        for index in _batches(audio, batch_size):
            inputs = self.processor([audio[i] for i in index], sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = inputs.to("cuda", dtype=torch.float32)
            ids = self.model.generate(**inputs)
            for i, text in zip(index, self.processor.batch_decode(ids, skip_special_tokens=True)):
                out[i] = text.strip()
        return out


class AdapterSwap:
    """A plain model (its LoRA layers removed again) whose LoRA target weights can be set to the base plus any one of
    several adapters: W = W0 + (B @ A) * scaling, the same operation PEFT's merge does, so after ``use(k)`` the weights
    equal ``merge_adapters(base, [adapter k])`` exactly (tests/test_view.py). Unmerged LoRA layers cost several extra
    small kernels per layer and decoding step: with five adapters active a request took 19x the merged model's time
    (log.md, 10 Nov, view-bench2). A switch costs one copy and one small matmul per target layer instead."""

    def __init__(self, model, paths: list):
        from peft import PeftModel
        from peft.tuners.lora import LoraLayer
        names = [f"r{k}" for k in range(len(paths))]
        peft = PeftModel.from_pretrained(model, paths[0], adapter_name=names[0])
        for name, path in zip(names[1:], paths[1:]):
            peft.load_adapter(path, adapter_name=name)
        self.count, self.targets, self.current = len(paths), [], None
        for m in peft.modules():
            if isinstance(m, LoraLayer):
                w = m.get_base_layer().weight
                factors = [(m.lora_A[n].weight.detach(), m.lora_B[n].weight.detach(), m.scaling[n]) if n in m.lora_A else None
                           for n in names]
                self.targets.append((w, w.detach().clone(), factors, bool(getattr(m, "fan_in_fan_out", False))))
        self.extra_parameters = sum(a.numel() + b.numel() for *_, fs, _ in self.targets for f in fs if f for a, b in [f[:2]])
        self.model = peft.unload().eval()  # the base layers back in place, their weights untouched

    @torch.no_grad()
    def use(self, k: int) -> None:
        if k == self.current:
            return
        for w, w0, factors, fan_in_fan_out in self.targets:
            w.data.copy_(w0)
            if factors[k] is not None:
                a, b, scaling = factors[k]
                delta = (b @ a) * scaling
                w.data += delta.T if fan_in_fan_out else delta
        self.current = k


class WhisperReadings:
    """Several LoRA adapters of one Whisper base for the readings view (lifemate.view, ASR_IMPROVEMENT_PLAN.md Stage 4).
    The main reading comes from today's averaged model (``Whisper`` with "a+b+...": the mean of the deltas merged in),
    the single readings from one more copy of the base whose weights are switched to each adapter (``AdapterSwap``), so
    every pass runs at a merged model's speed. Distinct parameters: the base and the adapters (PhoWhisper-medium and
    five r32 adapters: 0.94B; the averaged model is computed from them at load time)."""

    def __init__(self, name: str, adapters: str, edit=None, device: str = None):
        """``edit``: weight-space edit of the main (averaged) reading only; the single readings stay as trained.
        ``device``: as for ``Whisper``."""
        from transformers import WhisperForConditionalGeneration
        paths = [a for a in adapters.split("+") if a]
        self.main = Whisper(name, "+".join(paths), edit, device)
        self.processor, self.prompt, self.eos = self.main.processor, self.main.prompt, self.main.eos
        self.device, self.dtype = self.main.device, self.main.dtype
        base = WhisperForConditionalGeneration.from_pretrained(name, dtype=self.dtype).to(self.device).eval()
        self.swap = AdapterSwap(base, paths)
        self.single = self.swap.model
        enable_short_context(self.single)
        self.model = self.main.model
        self.distinct_parameters = sum(p.numel() for p in self.single.parameters()) + self.swap.extra_parameters

    def singles(self, paths: list, batch_size: int = 16, drafts: list = None) -> list:
        """[[adapter 1 alone, ..., adapter k alone] for each clip]; log-mel once per batch for all k passes. With
        ``drafts`` (each clip's main-reading tokens) every clip is decoded alone by ``lookup_decode``, which checks the
        draft in a few passes: the readings mostly agree, and plain decoding of a long latest request took about 1.5 s
        per adapter on a T4 (log.md, 10 Nov, view-bench3)."""
        audio = [read_audio(p) for p in paths]
        if drafts is not None:
            return self._singles_lookup(audio, drafts)
        batches = []
        for index in _batches(audio, batch_size):
            part = [audio[i] for i in index]
            batches.append((index, log_mel(self.processor.feature_extractor, part).to(self.device, dtype=self.dtype),
                            token_limits(part)))
        out = [[""] * self.swap.count for _ in paths]
        for k in range(self.swap.count):
            self.swap.use(k)
            for index, feats, limits in batches:
                with torch.autocast("cuda", dtype=torch.float16) if str(self.device).startswith("cuda") else contextlib.nullcontext():
                    ids = greedy_decode(self.single, feats, self.prompt, self.eos, limits)
                for i, text in zip(index, self.processor.batch_decode(ids, skip_special_tokens=True)):
                    out[i][k] = text.strip()
        return out

    def _singles_lookup(self, audio: list, drafts: list) -> list:
        feats = [log_mel(self.processor.feature_extractor, [a]).to(self.device, dtype=self.dtype) for a in audio]
        limits = token_limits(audio)
        out = [[""] * self.swap.count for _ in audio]
        self.passes = 0
        for k in range(self.swap.count):
            self.swap.use(k)
            for i, (f, limit, draft) in enumerate(zip(feats, limits, drafts)):
                with torch.autocast("cuda", dtype=torch.float16) if str(self.device).startswith("cuda") else contextlib.nullcontext():
                    ids, passes = lookup_decode(self.single, f, self.prompt, self.eos, limit, draft)
                self.passes += passes
                out[i][k] = self.processor.decode(ids, skip_special_tokens=True).strip()
        return out

    def transcribe(self, paths: list, batch_size: int = 16, alternatives: list = None) -> list:
        """[[main reading, adapter 1 alone, ..., adapter k alone] for each clip in ``alternatives`` (indexes; default:
        every clip), [main reading] for the others. The single readings are decoded with the main reading as draft."""
        main, tokens = self.main.transcribe_ids(paths, batch_size)
        index = list(range(len(paths))) if alternatives is None else list(alternatives)
        alt = dict(zip(index, self.singles([paths[i] for i in index], batch_size, [tokens[i] for i in index]))) if index else {}
        return [[m] + alt.get(i, []) for i, m in enumerate(main)]


def load(name: str):
    return load_with_adapter(name, None)


def load_with_adapter(name: str, adapter: str = None):
    low = name.lower()
    if "qwen3-asr" in low or "qwen3_asr" in low:
        return QwenAsr(name)
    if "parakeet" in low:
        return Ctc(name)
    return Whisper(name, adapter)
