"""Fine-tune a Whisper checkpoint on the competition audio, with folds for honest out-of-fold transcripts.

    python -m lifemate.asr_train --out outputs/asr --model vinai/PhoWhisper-large --folds 4 --plan 0 --lora-r 32

A run with --plan k trains on every fold except k and transcribes fold k, so the LLM can train on realistic ASR errors.
--plan full trains on all samples and transcribes public test. Only train.jsonl and the audio are read.

--context trains with Whisper context prompts (method A, log.md 5 Nov 2026): each clip gets the phrases that
``asr_context`` picks from what the agent can see before it, matched against a first-pass transcript (--first-pass,
the exp7 out-of-fold texts), sometimes none (--prompt-drop). The held-out fold is then decoded by ``asr_ctx_eval``.

Extra copies to average (log.md 6 Nov 2026) keep exp7's folds with --fold-seed 2026 and change the batch order with --seed.

Hard-example training (ASR_IMPROVEMENT_PLAN.md Stage 3): --init-adapter continues an existing adapter ('{fold}' is
replaced by the held-out fold), --key-weight makes the tokens of key words (keyword_gain's: argument words of the
conversation's calls, number and time words, digits) count that many times in the loss, and --key-repeat N with
--key-oof puts the training clips whose out-of-fold reading got a key word wrong N times into each epoch.
"""
import argparse
import json
import random
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

from . import data, schedule


def sample_folds(items: list, folds: int, seed: int) -> dict:
    """Whole samples stay together, so turns of one conversation never sit on both sides."""
    ids = sorted(i["id"] for i in items)
    random.Random(seed).shuffle(ids)
    return {sid: n % folds for n, sid in enumerate(ids)}


def clips_of(item: dict) -> list:
    out = [(t["audio"], t["audio_text"]) for t in item["history"] if t["role"] == "user" and t.get("audio")]
    return out + [(item["audio"], item["audio_text"])]


def key_words(item: dict) -> set:
    """keyword_gain's key words for every clip of a sample: words of the arguments of its calls (the earlier turns' and
    the label's; ids such as lock_992 are skipped), and number and time words. Digits are added per text."""
    from .keyword_gain import NUMTIME, arg_words
    calls = [c for t in item["history"] for c in t.get("tool_calls") or []] + list(item["label"]["tool_calls"])
    return arg_words(calls) | NUMTIME


def key_spans(text: str, keys: set) -> list:
    """Character spans of the key words in ``text``, for the token weights."""
    from .keyword_gain import norm
    return [(m.start(), m.end()) for m in re.finditer(r"\w+", text) if m.group().isdigit() or norm(m.group()) in keys]


def misheard_key(text: str, heard: str, keys: set) -> bool:
    """A key word of ``text`` is replaced or dropped in ``heard`` (keyword_gain.group's alignment)."""
    from .keyword_gain import toks
    ref, hyp = toks(text), toks(heard)
    keys = keys | {w for w in ref if w.isdigit()}
    return any(op in ("replace", "delete") and any(w in keys for w in ref[i1:i2])
               for op, i1, i2, _, _ in SequenceMatcher(a=ref, b=hyp, autojunk=False).get_opcodes())


def weighted_loss(logits, labels, weights):
    """Mean token loss with each token counted ``weights`` times; with every weight 1 it equals the model's own loss
    (tests/test_asr_keys.py)."""
    import torch
    ce = torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
    w = weights.to(ce.device) * (labels != -100)
    return (ce * w).sum() / w.sum()


class Collate:
    """Runs in DataLoader workers: audio files and log-mel features are the CPU part of a step. Gives (features,
    decoder inputs or None, labels, token weights or None); weights only with ``spans`` ({audio: key-word spans})."""

    def __init__(self, processor, root: Path, spans: dict = None, key_weight: float = 1.0):
        self.processor, self.root, self.spans, self.key_weight = processor, root, spans, key_weight
        self.processor.tokenizer.set_prefix_tokens(language="vi", task="transcribe")

    def __call__(self, batch: list):
        import torch
        from .asr import log_mel, read_audio
        feats = log_mel(self.processor.feature_extractor, [read_audio(str(self.root / c[0])) for c in batch])
        if len(batch[0]) == 3:  # (audio, text, prompt): context training
            return (feats, *self.with_prompts(batch), None)
        enc = self.processor.tokenizer([t for _, t in batch], return_tensors="pt", padding=True, truncation=True, max_length=220,
                                       return_offsets_mapping=self.spans is not None)
        labels = enc.input_ids.masked_fill(enc.attention_mask == 0, -100)
        weights = None
        if self.spans is not None:  # a token weighs key_weight when it overlaps a key word (special tokens: offset 0-0)
            weights = torch.ones(labels.shape, dtype=torch.float32)
            starts, ends = enc.offset_mapping[..., 0], enc.offset_mapping[..., 1]
            for n, (audio, _) in enumerate(batch):
                for s, e in self.spans.get(audio, ()):
                    weights[n][(starts[n] < e) & (ends[n] > s)] = self.key_weight
        return feats, None, labels, weights

    def with_prompts(self, batch: list):
        """Decoder inputs and labels: <|startofprev|> prompt <|startoftranscript|> <|vi|> ... text <|endoftext|>.

        The loss skips the prompt, exactly as Whisper was pre-trained; without a prompt this equals the plain path.
        """
        import torch
        from .asr import PROMPT_TOKENS
        tok = self.processor.tokenizer
        prev = tok.convert_tokens_to_ids("<|startofprev|>")
        rows, skips = [], []
        for _, text, prompt in batch:
            ids = tok(text, truncation=True, max_length=220).input_ids
            head = [prev] + tok.encode(" " + prompt, add_special_tokens=False)[:PROMPT_TOKENS] if prompt else []
            rows.append(head + ids)
            skips.append(len(head))
        width = max(len(r) for r in rows) - 1
        dec = torch.full((len(rows), width), tok.eos_token_id, dtype=torch.long)
        lab = torch.full((len(rows), width), -100, dtype=torch.long)
        for n, (row, skip) in enumerate(zip(rows, skips)):
            dec[n, :len(row) - 1] = torch.tensor(row[:-1])
            lab[n, skip:len(row) - 1] = torch.tensor(row[skip + 1:])
        return dec, lab


def pick_prompts(clips: list, rng: random.Random, drop: float, reads_share: float) -> list:
    """(audio, text, prompt) for one epoch: no prompt with probability ``drop``; for a latest request, the prompt with
    the gold path's read results with probability ``reads_share``, else the one from earlier turns only."""
    out = []
    for audio, text, prompts in clips:
        prompt = ""
        if rng.random() >= drop:
            prompt = prompts["reads"] if prompts.get("reads") is not None and rng.random() < reads_share else prompts["ctx"]
        out.append((audio, text, prompt))
    return out


def context_prompts(items: list, first_pass: dict, catalog: dict, tokenizer) -> dict:
    """{audio: {"ctx": prompt, "reads": prompt or None}} chosen against the first-pass transcripts, as at inference."""
    from . import asr_context as C
    from .asr_ctx_eval import BUDGET, LIMIT, SINGLE_THRESHOLD, THRESHOLD

    def prompt(pool, text):
        return C.prompt_text(C.fit_tokens(C.select(pool, text, THRESHOLD, SINGLE_THRESHOLD, LIMIT), tokenizer, BUDGET))

    out = {}
    for item in items:
        clips = clips_of(item)
        reads = C.read_results(item["label"]["tool_calls"], item["label"].get("tool_results") or [], catalog)
        for n, ((audio, _), pool) in enumerate(zip(clips, C.turn_pools(item))):
            first = first_pass[audio]
            out[audio] = {"ctx": prompt(pool, first), "reads": prompt(C.merge(pool, reads), first) if n == len(clips) - 1 else None}
    return out


def synth_prompts(synth: list, items: list, catalog: dict, tokenizer, first_pass: dict, seed: int) -> dict:
    """Context prompts for synthetic clips (combination C), built like a real clip's: the pool of the clip it was made
    from, plus the values inserted into it with the chance a real misheard name has of being visible (0.3 from earlier
    turns, 0.6 once read results are in), matched against the filter's base-model transcript."""
    from . import asr_context as C
    from .asr_ctx_eval import BUDGET, LIMIT, SINGLE_THRESHOLD, THRESHOLD

    def prompt(pool, text):
        return C.prompt_text(C.fit_tokens(C.select(pool, text, THRESHOLD, SINGLE_THRESHOLD, LIMIT), tokenizer, BUDGET))

    sources = {}
    for item in items:
        clips = clips_of(item)
        reads = C.read_results(item["label"]["tool_calls"], item["label"].get("tool_results") or [], catalog)
        for n, ((audio, _), pool) in enumerate(zip(clips, C.turn_pools(item))):
            sources[audio] = (pool, C.merge(pool, reads) if n == len(clips) - 1 else None)
    rng = random.Random(seed)
    out = {}
    for c in synth:
        pool, with_reads = sources.get(c["source"], ({}, None))
        new = {}
        for value in c.get("new", []):
            C._add(new, value, "read")
        first = first_pass.get(c["id"]) or c["text"]
        ctx_pool = C.merge(pool, new) if rng.random() < 0.3 else pool
        reads_pool = None if with_reads is None else (C.merge(with_reads, new) if rng.random() < 0.6 else with_reads)
        out[c["id"]] = {"ctx": prompt(ctx_pool, first), "reads": None if reads_pool is None else prompt(reads_pool, first)}
    return out


def make_batches(clips: list, batch_size: int, rng: random.Random) -> list:
    """Pools of similar length keep padding, and so time and memory, small."""
    order = list(clips)
    rng.shuffle(order)
    size = batch_size * 40
    pools = [sorted(order[i:i + size], key=lambda c: len(c[1])) for i in range(0, len(order), size)]
    batches = [p[i:i + batch_size] for p in pools for i in range(0, len(p), batch_size)]
    rng.shuffle(batches)
    return batches


def train_one(args, train_clips: list, eval_clips: list, root: Path, tag: str):
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    from .asr import PROMPT_TOKENS, enable_short_context, greedy_decode, log_mel, read_audio, restore_full_context, token_limits
    from .asr_eval import edit_distance, normal_tokens

    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model)
    dtype = torch.float16 if args.lora_r else torch.float32  # only the adapter trains, so the rest can stay in half precision
    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=dtype)
    model.config.apply_spec_augment = args.spec_augment > 0
    model.config.mask_time_prob = args.spec_augment
    model.config.mask_feature_prob = args.spec_augment
    model.config.use_cache = False
    enable_short_context(model)
    base = model
    if args.init_path:  # continue an existing adapter (its own rank and targets)
        from peft import PeftModel
        print(tag, "continues", args.init_path, flush=True)
        model = PeftModel.from_pretrained(model, args.init_path, is_trainable=True)
        model.print_trainable_parameters()
    elif args.lora_r:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                                                 target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
        model.print_trainable_parameters()
    elif args.freeze_encoder:
        for p in base.get_encoder().parameters():
            p.requires_grad_(False)
    if not args.no_checkpointing:  # without it, a batch of long clips takes more than a T4 has
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if args.lora_r:
            model.enable_input_require_grads()
    model.to("cuda")
    params = [p for p in model.parameters() if p.requires_grad]
    print(tag, "trainable", sum(p.numel() for p in params), flush=True)
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, foreach=False)
    scaler = torch.amp.GradScaler("cuda")
    collate = Collate(processor, root, args.spans, args.key_weight)
    tokenizer = processor.tokenizer
    prompt, eos, start = tokenizer.prefix_tokens, tokenizer.eos_token_id, base.config.decoder_start_token_id

    prev = tokenizer.convert_tokens_to_ids("<|startofprev|>")

    def start_of(clip):
        """Context runs are scored with the prompt the clip gets at inference: reads for a latest request, else earlier turns."""
        if len(clip) < 3:
            return list(prompt)
        text = clip[2]["reads"] if clip[2].get("reads") is not None else clip[2]["ctx"]
        return [prev] + tokenizer.encode(" " + text, add_special_tokens=False)[:PROMPT_TOKENS] + list(prompt) if text else list(prompt)

    def evaluate(clips, batch_size=16):
        model.eval()
        base.config.use_cache = True
        errors, words = 0, 0
        order = sorted(range(len(clips)), key=lambda i: len(clips[i][1]))
        for begin in range(0, len(order), batch_size):
            part = [clips[i] for i in order[begin:begin + batch_size]]
            audio = [read_audio(str(root / c[0])) for c in part]
            feats = log_mel(processor.feature_extractor, audio).to("cuda", dtype=dtype)
            with torch.autocast("cuda", dtype=torch.float16):
                ids = greedy_decode(base, feats, [start_of(c) for c in part], eos, token_limits(audio))
            for (_, ref, *_), text in zip(part, processor.batch_decode(ids, skip_special_tokens=True)):
                r, h = normal_tokens(ref), normal_tokens(text.strip())
                errors += edit_distance(r, h)
                words += len(r)
        model.train()
        base.config.use_cache = False
        return 100 * errors / max(words, 1)

    def backward(feats, ids, dec=None, weights=None) -> float:
        """A function, so that the graph of a step that runs out of memory dies with its frame."""
        extra = {} if dec is None else {"decoder_input_ids": dec.to("cuda")}
        labels = ids.to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(input_features=feats.to("cuda", dtype=dtype), labels=labels, **extra)
        loss = out.loss if weights is None else weighted_loss(out.logits, labels, weights)
        scaler.scale(loss).backward()
        return loss.item()

    def save(path):
        restore_full_context(base)  # the forward pre-hook shortens the table again on the next batch
        model.save_pretrained(path)
        processor.save_pretrained(path)

    steps_per_epoch = len(train_clips) // args.batch_size
    total = int(steps_per_epoch * args.epochs)
    warmup = max(20, total // 20)
    rng = random.Random(args.seed)
    started, best, step, epoch = time.monotonic(), 1e9, 0, 0
    history = []

    def report_wer():
        history.append({"epoch": epoch, "wer": round(evaluate(eval_clips[:args.eval_clips]), 2)})
        print(json.dumps({"tag": tag, **history[-1], "min": round((time.monotonic() - started) / 60, 1)}), flush=True)
        return history[-1]["wer"]

    if eval_clips:
        report_wer()
    trained_from = time.monotonic()
    deadline = trained_from + args.max_hours * 3600
    model.train()
    while step < total and time.monotonic() < deadline:
        epoch += 1
        running, count = 0.0, 0
        epoch_clips = pick_prompts(train_clips, rng, args.prompt_drop, args.reads_share) if args.context else train_clips
        loader = torch.utils.data.DataLoader(make_batches(epoch_clips, args.batch_size, rng), batch_size=None, num_workers=2,
                                             collate_fn=collate, prefetch_factor=4)
        for feats, dec, ids, weights in loader:  # dec: context training, decoder inputs and labels come ready
            if dec is None and (ids[:, 0] == start).all():  # the model puts the start token back when it shifts the labels
                ids = ids[:, 1:]
                weights = None if weights is None else weights[:, 1:]
            used = 0.0 if args.no_time_decay else (time.monotonic() - trained_from) / max(1.0, deadline - trained_from)
            lr = schedule.linear(step, warmup, total, args.lr, used)  # the decay ends at the deadline if that comes first
            for g in optim.param_groups:
                g["lr"] = lr
            try:
                value = backward(feats, ids, dec, weights)
            except torch.cuda.OutOfMemoryError:  # one very long batch must not cost the whole run
                value = None
            if value is None:  # outside the except block: the failed step's graph is gone by now
                print(json.dumps({"tag": tag, "skipped_batch_frames": feats.shape[-1]}), flush=True)
                optim.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                step += 1
                continue
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            running += value
            count += 1
            step += 1
            if step % 100 == 0:
                print(json.dumps({"tag": tag, "step": step, "of": total, "loss": round(running / count, 4), "lr": round(lr, 7),
                                  "min": round((time.monotonic() - started) / 60, 1),
                                  "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
                running, count = 0.0, 0
            if step >= total or time.monotonic() > deadline:
                break
        if eval_clips:
            wer = report_wer()
            if wer < best:
                best = wer
                save(out / "best")
    if not eval_clips:
        save(out / "best")
    (out / "history.json").write_text(json.dumps(history), encoding="utf-8")
    del model, optim
    torch.cuda.empty_cache()
    return out / "best"


def transcribe_with(best: Path, args, audio: list, root: Path) -> dict:
    from . import asr
    backend = asr.Whisper(args.model, adapter=str(best)) if args.lora_r else asr.Whisper(str(best))
    return dict(zip(audio, backend.transcribe([str(root / a) for a in audio], args.batch_size_eval)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/asr")
    ap.add_argument("--model", default="vinai/PhoWhisper-large")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--plan", default="0", help="fold to hold out, or 'full' for the all-data model")
    ap.add_argument("--epochs", type=float, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--batch-size-eval", type=int, default=16)
    ap.add_argument("--eval-clips", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--freeze-encoder", action="store_true", help="without LoRA: train the decoder only, which fits a T4")
    ap.add_argument("--no-checkpointing", action="store_true")
    ap.add_argument("--spec-augment", type=float, default=0.05)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--no-time-decay", action="store_true", help="decay by steps only, even when --max-hours ends the run first")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--fold-seed", type=int, default=None,
                    help="seed of the fold split (default --seed); 2026 keeps exp7's folds while --seed changes the batch order")
    ap.add_argument("--context", action="store_true", help="train with context prompts (method A)")
    ap.add_argument("--first-pass", default="auto:medium_train.json", help="out-of-fold transcripts the prompts are matched against")
    ap.add_argument("--prompt-drop", type=float, default=0.2, help="share of clips trained without a prompt")
    ap.add_argument("--reads-share", type=float, default=0.5, help="latest requests: share given the prompt with read results")
    ap.add_argument("--synth", action="store_true", help="add synthetic clips (method B) from synth_manifest_*.json under /kaggle/input")
    ap.add_argument("--synth-keep", default="", help="synth_keep.json of synth_filter (a path, or auto:<name> under /kaggle/input): only these synthetic clips")
    ap.add_argument("--init-adapter", default="", help="continue this adapter (auto:<name> under /kaggle/input; '{fold}' = the held-out fold)")
    ap.add_argument("--key-weight", type=float, default=1.0, help="loss weight of key-word tokens (1 = plain loss)")
    ap.add_argument("--key-repeat", type=int, default=1, help="copies per epoch of training clips whose --key-oof reading got a key word wrong")
    ap.add_argument("--key-oof", default="", help="out-of-fold transcripts that pick the clips --key-repeat repeats")
    args = ap.parse_args(argv)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    fold = -1 if args.plan == "full" else int(args.plan)
    root_dir = data.find_root("train")
    root = Path(root_dir) / "train"
    items = data.load_split("train", root_dir)
    folds = sample_folds(items, args.folds, args.seed if args.fold_seed is None else args.fold_seed)
    clip_list = clips_of
    if args.context:
        from transformers import AutoTokenizer

        from .infer import load_transcripts
        prompts = context_prompts(items, load_transcripts(args.first_pass), data.load_catalog("train", root_dir),
                                  AutoTokenizer.from_pretrained(args.model))
        clip_list = lambda item: [(a, t, prompts[a]) for a, t in clips_of(item)]  # noqa: E731
        shown = sum(bool(p["ctx"]) for p in prompts.values())
        print(f"context prompts: {len(prompts)} clips, {shown} with a non-empty earlier-turns prompt", flush=True)
    if fold >= 0:
        train_clips = [c for i in items if folds[i["id"]] != fold for c in clip_list(i)]
        eval_clips = [c for i in items if folds[i["id"]] == fold for c in clip_list(i)]
    else:
        train_clips, eval_clips = [c for i in items for c in clip_list(i)], []
    if args.synth:  # method B: synthetic clips of the other folds only; they are never evaluated
        from .synth_filter import load_manifests
        synth = load_manifests()
        if args.synth_keep.startswith("auto:"):  # the filter's output from an attached dataset or kernel
            name = args.synth_keep[len("auto:"):]
            args.synth_keep = str(next(data.walk_input(name)) / name)
            print("synth keep list:", args.synth_keep, flush=True)
        if args.synth_keep:
            kept = set(json.loads(Path(args.synth_keep).read_text(encoding="utf-8"))["keep"])
            synth = [c for c in synth if c["id"] in kept]
        synth = [c for c in synth if c["fold"] != fold]
        if args.context:  # combination C: synthetic clips get prompts built like real ones
            from transformers import AutoTokenizer
            hyp = json.loads(Path(args.synth_keep).read_text(encoding="utf-8")).get("hyp", {}) if args.synth_keep else {}
            sp = synth_prompts(synth, items, data.load_catalog("train", root_dir), AutoTokenizer.from_pretrained(args.model),
                               hyp, args.seed)
            train_clips += [(c["path"], c["text"], sp[c["id"]]) for c in synth]
        else:
            train_clips += [(c["path"], c["text"]) for c in synth]
        print(f"synthetic clips added: {len(synth)} (folds other than {fold})", flush=True)
    args.init_path, args.spans = "", None
    if args.init_adapter:
        from .infer import resolve_adapter
        args.init_path = resolve_adapter(args.init_adapter.replace("{fold}", str(fold)))
    if args.key_weight != 1.0 or args.key_repeat > 1:
        if args.context:
            raise SystemExit("--key-weight and --key-repeat are for plain training, not --context")
        keys = {a: key_words(i) for i in items for a, _ in clips_of(i)}
        if args.key_weight != 1.0:
            args.spans = {a: key_spans(t, keys[a]) for i in items for a, t in clips_of(i)}
            marked = [c for c in train_clips if args.spans.get(c[0])]
            print(f"key-word weight {args.key_weight}: {len(marked)} of {len(train_clips)} training clips have key words, "
                  f"{sum(len(args.spans[c[0]]) for c in marked)} words", flush=True)
        if args.key_repeat > 1:
            from .infer import load_transcripts
            heard = load_transcripts(args.key_oof)
            hard = {a for a, t, *_ in train_clips if a in heard and a in keys and misheard_key(t, heard[a], keys[a])}
            train_clips += [c for c in train_clips if c[0] in hard] * (args.key_repeat - 1)
            print(f"key-word repeats x{args.key_repeat}: {len(hard)} training clips misheard a key word in {args.key_oof}; "
                  f"{len(train_clips)} clips per epoch", flush=True)
    random.Random(args.seed).shuffle(eval_clips)
    tag = f"fold{fold}" if fold >= 0 else "full"
    print(tag, "train clips", len(train_clips), "eval clips", len(eval_clips), flush=True)
    best = train_one(args, train_clips, eval_clips, root, tag)
    if eval_clips and args.context:  # the three passes of method A on the held-out fold, with this fold's model
        from . import asr
        from .asr_ctx_eval import fold_clips, run_fold
        backend = asr.Whisper(args.model, adapter=str(best))
        run_fold(backend, fold_clips(items, folds, fold, data.load_catalog("train", root_dir)), root, Path(args.out),
                 tag, args.batch_size_eval)
    elif eval_clips:
        texts = transcribe_with(best, args, [a for a, _ in eval_clips], root)
        (Path(args.out) / f"oof_{fold}.json").write_text(json.dumps(texts, ensure_ascii=False, indent=0), encoding="utf-8")
    else:
        from .transcribe import paths_of
        pub_dir = data.find_root("public_test")
        audio = paths_of(data.load_split("public_test", pub_dir))
        texts = transcribe_with(best, args, audio, Path(pub_dir) / "public_test")
        (Path(args.out) / "public_test.json").write_text(json.dumps(texts, ensure_ascii=False, indent=0), encoding="utf-8")


if __name__ == "__main__":
    main()
