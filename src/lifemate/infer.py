"""Run the agent on a split and write submission.jsonl plus metrics.json, one process per GPU.

    python -m lifemate.infer --split public_test --out outputs/run --transcripts-dir path
    python -m lifemate.infer --split train --transcripts gold --ids-file holdout.json --out outputs/run

Each sample gets exactly one ToolEnv and only zaic_tools touches its data; <split>_environment/ is never read here.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import data, prompts

_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_DATETIME = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})(:\d{2})?(Z|[+-]\d{2}:?\d{2})?$")
MAX_CONTEXT = 16384  # competition limit on the LLM context length


def parse_call(text: str):
    """(name, arguments) of the first tool call in a generated turn, None if the turn holds no well-formed call."""
    match = _CALL.search(text)
    raw = match.group(1) if match else None
    if raw is None:
        open_at = text.find("<tool_call>")  # the turn may have hit the token limit before the closing tag
        if open_at < 0:
            return None
        raw = text[open_at + len("<tool_call>"):].strip()
    try:
        call = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(call, dict) and isinstance(call.get("name"), str) and isinstance(call.get("arguments", {}), dict):
        return call["name"], call.get("arguments") or {}
    return None


def tidy(name: str, arguments: dict, catalog: dict = None) -> dict:
    """Put arguments in the exact form gold always uses, since the scorer and the state hash compare them as they are.

    ask_missing_info params are sorted, date-times carry seconds and offset, numbers have their schema type, and an
    empty optional value is left out.
    """
    if name == "sys.ask_missing_info" and isinstance(arguments.get("params"), list):
        arguments = {**arguments, "params": sorted(str(p) for p in arguments["params"])}
    props = ((catalog or {}).get(name, {}).get("parameters") or {}).get("properties") or {}
    out = {}
    for key, value in arguments.items():
        if value in (None, "", [], {}):
            continue
        spec = props.get(key, {})
        if spec.get("type") == "integer" and isinstance(value, str) and re.fullmatch(r"-?\d+", value):
            value = int(value)
        elif spec.get("type") == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
        if isinstance(value, str) and (spec.get("format") == "date-time" or "ISO-8601" in (spec.get("description") or "")):
            match = _DATETIME.match(value)
            if match:
                value = f"{match.group(1)}{match.group(2) or ':00'}{match.group(3) or '+07:00'}"
        out[key] = value
    return out


def unseen_reference(name: str, arguments: dict, catalog: dict, context: str):
    """An ID argument nothing has shown so far (gold: 1 case in 1,278). Such a call would do nothing but cost precision."""
    props = ((catalog or {}).get(name, {}).get("parameters") or {}).get("properties") or {}
    for key, value in arguments.items():
        if props.get(key, {}).get("x-ref") and isinstance(value, str) and value not in context:
            return key, value
    return None


def name_params(spec: dict) -> set:
    """Parameters that name something in words (a contact, a room, a title): strings that are not IDs, enums, dates or times."""
    out = set()
    for key, p in ((spec.get("parameters") or {}).get("properties") or {}).items():
        desc = p.get("description") or ""
        if p.get("type") != "string" or p.get("enum") or p.get("x-ref") or p.get("format") in ("date", "date-time", "time"):
            continue
        if "ISO-8601" in desc or desc.startswith("YYYY-MM-DD") or desc.startswith("HH:MM"):
            continue
        out.add(key)
    return out


def found_nothing(result: dict) -> bool:
    """A read that ran and returned only empty lists."""
    lists = [v for v in result.values() if isinstance(v, list)]
    return bool(result.get("ok")) and bool(lists) and all(not v for v in lists)


def broader_call(name: str, arguments: dict, result: dict, catalog: dict):
    """Arguments of a wider read after an optional name filter found nothing, or None.

    Search is a plain substring match, so a name said differently ("Cô Mai" for "Cô giáo Mai") or misheard finds
    nothing. Without the filter the read lists everything (3 to 10 items on validation), and the model picks from the
    list. One more read costs a little precision; a missing or wrong write costs the whole sample. Required filters,
    dates and enums are left alone: an empty answer to "anything tomorrow?" is a real answer.
    """
    spec = (catalog or {}).get(name) or {}
    if spec.get("kind") != "read" or not found_nothing(result):
        return None
    required = set((spec.get("parameters") or {}).get("required") or [])
    drop = {k for k in name_params(spec) - required if isinstance(arguments.get(k), str)}
    return {k: v for k, v in arguments.items() if k not in drop} if drop else None


class Generator:
    """Greedy decoding that keeps the KV cache between steps: each step only feeds what is new.

    ``ask_bias`` lowers the logit of the first token of ``sys.ask_missing_info`` where a tool name starts. The model asks
    when gold acts about three times as often as the reverse (24 against 7 of 386 validation samples), and the change is
    deterministic, so greedy decoding stays reproducible. A negative ``ask_bias`` raises that logit instead (11 Nov: the
    view-recipe models act where gold asks about 2.5 times as often as the reverse, 15 against 6 of 387 samples); it is
    checked at every position where a tool name starts, since the greedy token there is then not ``sys``.
    """

    def __init__(self, tokenizer, model, max_new_tokens: int, ask_bias: float = 0.0, lookup: int = 0):
        import torch
        self.torch, self.tokenizer, self.model = torch, tokenizer, model
        self.max_new_tokens = max_new_tokens
        # Prompt lookup (speed only, 8 Nov 2026): up to `lookup` drafted tokens checked in one forward pass. 0 = the
        # one-token loop below, unchanged.
        self.lookup = lookup
        self.lookup_stats = {"rounds": 0, "drafted": 0, "accepted": 0, "rollbacks": 0}
        self.stop_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.ask_bias, self.flips = ask_bias, 0
        # solve.py sets stop_at per sample: a step already running stops there, so no sample runs past the time limit.
        # Validation leaves it at None and decodes exactly as before.
        self.stop_at, self.cut = None, 0
        self.name_prefix, self.sys_id = self.ask_tokens() if ask_bias else (None, None)
        if ask_bias and self.sys_id is None:
            print("ask bias disabled: no token boundary after '{\"name\":\"'", flush=True)
            self.ask_bias = 0.0

    def encode(self, text: str) -> list:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def ask_tokens(self) -> tuple:
        """(tokens of '{"name":"', first token of 'sys'), as the training text tokenizes a call.

        Qwen3.5 gives '{"', 'name', '":"', 'sys', the same alone and inside a turn; 'system' is a different token.
        """
        ids = self.encode('{"name":"sys.ask_missing_info"')
        for cut in range(1, len(ids)):
            if self.tokenizer.decode(ids[:cut]) == '{"name":"':
                return ids[:cut], ids[cut]
        return None, None

    def generate(self, ids: list, cache, fed: int, limit: int = 0):
        """Returns (generated ids, cache, tokens the cache has seen). ``ids`` is the whole conversation so far.
        ``limit`` overrides max_new_tokens for this turn (the first turn, which may carry a worked note)."""
        torch = self.torch
        device = self.model.device
        if self.lookup:
            return self.generate_lookup(ids, cache, fed, limit)
        generated, step = [], ids[fed:]
        with torch.inference_mode():
            for _ in range(limit or self.max_new_tokens):
                out = self.model(input_ids=torch.tensor([step], device=device), past_key_values=cache,
                                 use_cache=True, logits_to_keep=1)
                cache, fed = out.past_key_values, fed + len(step)
                logits = out.logits[0, -1]
                token = int(logits.argmax())
                if self.watch(token):
                    token = self.pick(logits, generated)
                generated.append(token)
                if token == self.stop_id:
                    break
                if self.stop_at is not None and time.monotonic() > self.stop_at:  # after one token: callers read new[-1]
                    self.cut += 1
                    break
                step = [token]
        return generated, cache, fed

    def pick(self, logits, before: list) -> int:
        """The greedy token for one position, with the ask bias of ``generate``; ``before`` = this turn's tokens so far."""
        token = int(logits.argmax())
        if self.ask_bias and (token == self.sys_id or self.ask_bias < 0) and before[-len(self.name_prefix):] == self.name_prefix:
            biased = logits.float().clone()
            biased[self.sys_id] -= self.ask_bias
            new = int(biased.argmax())
            self.flips += new != token
            token = new
        return token

    def watch(self, token: int) -> bool:
        """Whether ``pick`` must look at this position: a penalty only matters where greedy picks ``sys``, a boost
        anywhere (``pick`` itself checks that a tool name starts here)."""
        return bool(self.ask_bias) and (token == self.sys_id or self.ask_bias < 0)

    def generate_lookup(self, ids: list, cache, fed: int, limit: int = 0):
        """``generate`` with prompt-lookup drafts: same greedy tokens, fewer forward passes.

        Tool calls mostly copy text already in the conversation (tool names, ids, the heard request), so the tokens that
        followed the latest earlier occurrence of the last few tokens are a good guess. One forward pass checks the
        guess: the greedy token at each position is computed as usual, the guess is kept up to the first position where
        it differs, and that position's greedy token is added as well. The linear-attention layers keep only a running
        state, so when part of a guess is rejected their states are put back from a copy taken before the pass, the
        attention caches are cut back, and the kept tokens are fed again at the start of the next pass.
        """
        torch = self.torch
        device = self.model.device
        budget = limit or self.max_new_tokens
        drafts = NgramDraft(ids)
        generated, step = [], ids[fed:]
        with torch.inference_mode():
            while len(generated) < budget:
                room = budget - len(generated) - 1  # the pass always adds one token of its own
                # No guess in a turn's first pass (it feeds the new tail, which a rollback would feed twice) or before
                # there is a cache to put back.
                draft = drafts.draft(min(self.lookup, room)) if generated and cache is not None and room > 0 else []
                saved = snapshot_cache(cache) if draft else None
                feed = step + draft
                out = self.model(input_ids=torch.tensor([feed], device=device), past_key_values=cache, use_cache=True,
                                 logits_to_keep=len(draft) + 1)
                cache = out.past_key_values
                logits = out.logits[0]
                firsts = logits.argmax(-1).tolist()  # one transfer for the whole pass
                new = []
                for j in range(len(draft) + 1):
                    token = firsts[j]
                    if self.watch(token):
                        token = self.pick(logits[j], generated + new)
                    new.append(token)
                    if token == self.stop_id or j == len(draft) or token != draft[j]:
                        break
                kept = len(new) - 1  # guessed tokens that were right
                self.lookup_stats["rounds"] += 1
                self.lookup_stats["drafted"] += len(draft)
                self.lookup_stats["accepted"] += kept
                generated += new
                drafts.extend(new)
                if kept < len(draft):  # the cache took tokens that are not part of the answer: back to before the pass
                    restore_cache(cache, saved)
                    self.lookup_stats["rollbacks"] += 1
                    step = step + new  # fed again next pass, or by the next turn (``fed`` stays where it was)
                else:
                    fed += len(feed)
                    step = [new[-1]]
                if new[-1] == self.stop_id:
                    break
                if self.stop_at is not None and time.monotonic() > self.stop_at:
                    self.cut += 1
                    break
        return generated, cache, fed


class NgramDraft:
    """Prompt lookup: the tokens that followed the latest earlier occurrence of the sequence's last n tokens (n from
    ``max_n`` down to 1), from an index kept up to date as tokens are added."""

    def __init__(self, tokens: list, max_n: int = 3):
        self.tokens, self.max_n, self.index = [], max_n, {}
        self.extend(tokens)

    def extend(self, new: list) -> None:
        for token in new:
            end = len(self.tokens)  # the n-grams that end here now have a next token: `token`, at position `end`
            for n in range(1, min(self.max_n, end) + 1):
                self.index[tuple(self.tokens[end - n:end])] = end
            self.tokens.append(token)

    def draft(self, k: int) -> list:
        if k <= 0:
            return []
        for n in range(min(self.max_n, len(self.tokens)), 0, -1):
            start = self.index.get(tuple(self.tokens[-n:]))
            if start is not None:
                return self.tokens[start:start + k]
        return []


def snapshot_cache(cache) -> list:
    """What a forward pass changes in a hybrid cache: copies of the fixed-size linear-attention states (updated in
    place) and the length of each attention cache (which only grows)."""
    saved = []
    for layer in cache.layers:
        entry = {}
        if hasattr(layer, "recurrent_states"):
            entry["conv"] = {i: s.clone() for i, s in layer.conv_states.items() if s is not None}
            entry["recurrent"] = {i: s.clone() for i, s in layer.recurrent_states.items() if s is not None}
        if hasattr(layer, "keys"):
            entry["length"] = layer.get_seq_length()
        saved.append(entry)
    return saved


def restore_cache(cache, saved: list) -> None:
    """Puts a cache back to ``snapshot_cache``'s state."""
    for layer, entry in zip(cache.layers, saved):
        for i, s in entry.get("conv", {}).items():
            layer.conv_states[i].copy_(s)
        for i, s in entry.get("recurrent", {}).items():
            layer.recurrent_states[i].copy_(s)
        if "length" in entry:
            extra = layer.get_seq_length() - entry["length"]
            if extra > 0:
                layer.crop(-extra)


def run_agent(gen: Generator, env, prompt: str, deadline: float, max_steps: int, repairs: int = 2, denoise: bool = False,
              catalog: dict = None, broaden: bool = False, max_broaden: int = 2, relisten=None, first_tokens: int = 0) -> dict:
    """Calls ``env`` until the model stops. env.validate is free, so a rejected call never reaches the call log.

    ``first_tokens`` (when set) is the token limit of the first generated turn instead of gen.max_new_tokens: with
    worked notes that turn holds the restated request, the note and the first call.

    With ``broaden``, a read whose name filter found nothing is followed by a wider read (``broader_call``), written into
    the conversation as if the model had made it, at most ``max_broaden`` times per sample.

    ``relisten(reads)`` (method A, second pass) is called after each read while nothing has been written, with every
    (name, arguments, result) read so far. When it returns a new prompt (the request decoded again with names from those
    results), the agent starts again from that prompt, once. A call it repeats gets its earlier result back without a
    second call to ``env``, so the call log holds each call once.
    """
    stats = {"steps": 0, "repairs": 0, "max_prompt_tokens": 0, "stop": "done", "gen_tokens": 0, "turns": [], "broadened": 0,
             "relistened": 0}
    flips_before = getattr(gen, "flips", 0)
    ids = gen.encode(prompt + (prompts.THINK_OPEN if denoise else prompts.GEN_PREFIX))
    cache, fed = None, 0
    seen_calls = set()
    context = prompt
    reads, wrote, memo = [], False, {}
    while stats["steps"] < max_steps:
        if time.monotonic() > deadline:
            stats["stop"] = "budget"
            break
        limit = first_tokens if first_tokens and not stats["turns"] else gen.max_new_tokens
        if len(ids) + limit > MAX_CONTEXT:
            stats["stop"] = "context"
            break
        stats["max_prompt_tokens"] = max(stats["max_prompt_tokens"], len(ids))
        if limit != gen.max_new_tokens:
            new, cache, fed = gen.generate(ids, cache, fed, limit)
        else:
            new, cache, fed = gen.generate(ids, cache, fed)
        stats["gen_tokens"] += len(new)
        text = gen.tokenizer.decode(new, skip_special_tokens=False)
        stats["turns"].append(text.replace("<|im_end|>", ""))
        ids = ids + new
        if new[-1] != gen.stop_id:  # cut off by the token limit: close the turn before feeding anything else
            ids += gen.encode("<|im_end|>")
        parsed = parse_call(text)
        if parsed is None:
            break
        name, arguments = parsed
        arguments = tidy(name, arguments, catalog)
        check = env.validate(name, arguments)
        unseen = unseen_reference(name, arguments, catalog, context) if check["ok"] else None
        wider = None
        if not check["ok"] or unseen:
            stats["repairs"] += 1
            if stats["repairs"] > repairs:
                stats["stop"] = "invalid"
                break
            error = check["error"] if not check["ok"] else f"{unseen[0]}={unseen[1]} chưa có trong kết quả nào, hãy gọi tool đọc trước"
            result = {"ok": False, "error": error}
        else:
            key = _call_key(name, arguments)
            if key in seen_calls:  # the same call twice is a loop, and it would only cost precision
                stats["stop"] = "repeat"
                break
            seen_calls.add(key)
            if key in memo:  # made before the request was heard again: the same result, and no second entry in the log
                result = memo[key]
            else:
                result = env.call(name, arguments)
                memo[key] = result
                stats["steps"] += 1
            if name == "sys.ask_missing_info":  # always the last call in gold: the agent waits for the user
                stats["stop"] = "asked"
                break
            if (catalog or {}).get(name, {}).get("kind") != "read":
                wrote = True
            elif relisten is not None and not wrote and not stats["relistened"] and time.monotonic() < deadline:
                reads.append((name, arguments, result))
                again = relisten(reads)
                if again is not None:
                    prompt, stats["relisten_text"] = again
                    stats["relistened"] += 1
                    stats["turns"].append("[relisten]")
                    ids = gen.encode(prompt + (prompts.THINK_OPEN if denoise else prompts.GEN_PREFIX))
                    cache, fed, context, seen_calls = None, 0, prompt, set()
                    continue
            if broaden and stats["broadened"] < max_broaden and stats["steps"] < max_steps:
                wider = broader_call(name, arguments, result, catalog)
        text = prompts.result_text(result)
        context += text
        tail = "\n" + prompts.turn("user", text)
        if wider is not None and _call_key(name, wider) not in seen_calls and env.validate(name, wider)["ok"]:
            seen_calls.add(_call_key(name, wider))
            wider_result = env.call(name, wider)
            stats["steps"] += 1
            stats["broadened"] += 1
            context += prompts.result_text(wider_result)
            tail += prompts.step_text(name, wider, wider_result)  # as if the model had made the call itself
        ids += gen.encode(tail + prompts.GEN_PREFIX)
    else:
        stats["stop"] = "max_steps"
    stats["ask_flips"] = getattr(gen, "flips", 0) - flips_before
    return stats


def _call_key(name: str, arguments: dict) -> tuple:
    return name, json.dumps(arguments, sort_keys=True, ensure_ascii=False)


def make_relisten(backend, item: dict, texts: dict, catalog: dict, audio_path: str):
    """The second pass of method A for one sample: decode the latest request again with Whisper's prompt holding the
    phrases of the agent's own read results (and of the profile and earlier turns) that sound like the current text."""
    from . import asr_context as C
    from .asr_ctx_eval import BUDGET, LIMIT, SINGLE_THRESHOLD, THRESHOLD
    pool = C.turn_pools(item)[-1]
    current = texts[item["audio"]]
    words = lambda t: re.findall(r"[^\W_]+", t)  # noqa: E731  (case kept: names must match the address book)

    def relisten(reads: list):
        found = C.read_results([{"name": n, "arguments": a} for n, a, _ in reads], [r for _, _, r in reads], catalog)
        if not found:
            return None
        phrases = C.fit_tokens(C.select(C.merge(pool, found), current, THRESHOLD, SINGLE_THRESHOLD, LIMIT),
                               backend.processor.tokenizer, BUDGET)
        if not phrases:
            return None
        new = backend.transcribe([audio_path], 1, [C.prompt_text(phrases)])[0]
        if not new or words(new) == words(current):
            return None
        return prompts.prompt_text(item, {**texts, item["audio"]: new}, catalog), new

    return relisten


def load_transcripts(path: str, latest: set = None) -> dict:
    """{audio path: text} from a JSON file or from every *.json under a folder.

    A comma-separated list merges several ASR systems: the model sees each differing reading, split by " | ".
    'view:MAIN,OTHER,...' gives the readings view (lifemate.view): the main reading, with the others in braces where
    they disagree with it; with ``latest`` (the latest requests' clips) only those get braces and every other clip
    keeps the main reading, since a perfect hearing of earlier turns adds nothing (log.md, Stage 0 (b): 79.78 with only
    the latest request said right, 79.46 with every turn).
    """
    if path.startswith("view:"):
        from . import view
        main, *others = load_readings(path)
        return {audio: view.build(text, [o.get(audio, "") for o in others]) if latest is None or audio in latest else text
                for audio, text in main.items()}
    if "," in path:
        parts = [load_transcripts(p) for p in path.split(",")]
        return {audio: merge_readings([part.get(audio, "") for part in parts]) for audio in parts[0]}
    if path.startswith("auto:"):  # a file from an attached kernel, wherever Kaggle mounted it
        name = path[len("auto:"):]
        for d, _, files in os.walk("/kaggle/input"):
            if name in files:
                return load_transcripts(os.path.join(d, name))
        raise FileNotFoundError(f"{name} not found under /kaggle/input")
    p = Path(path)
    files = sorted(p.rglob("*.json")) if p.is_dir() else [p]
    out = {}
    for f in files:
        out.update(json.loads(f.read_text(encoding="utf-8")))
    return out


def load_readings(spec: str) -> list:
    """[{audio: text}] of each reading in a 'view:MAIN,OTHER,...' spec, the main reading first."""
    return [load_transcripts(p) for p in spec[len("view:"):].split(",")]


def merge_readings(texts: list) -> str:
    seen, out = set(), []
    for text in texts:
        key = " ".join(re.findall(r"[^\W_]+", text.lower()))
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return " | ".join(out)


def resolve_adapter(spec: str) -> str:
    """'auto:fold_dn' picks the adapter under /kaggle/input whose path contains that run name; a path stays as it is."""
    if not spec.startswith("auto"):
        return spec
    wanted = spec[5:]
    found = [str(d) for d in data.walk_input("adapter_config.json") if not wanted or f"/{wanted}/" in str(d) + "/"]
    print("adapters found:", found, flush=True)
    if not found:
        raise SystemExit(f"no adapter matches {spec}")
    return found[0]


def load_adapted(llm: str, adapter: str = None, weights: str = None, edit: str = None):
    """The LLM with one adapter merged in, or with the mean of several (comma-separated), exactly as export_models makes it.
    ``weights`` (comma-separated, one per adapter, negative allowed) merges their weighted sum instead of the mean.
    ``edit`` (a lifemate.wsp program, JSON) rescales that change per module group before it is merged."""
    from . import models
    paths = [a for a in (adapter or "").split(",") if a]
    if edit and edit.strip().lower() != "none":
        from .export_models import parse_weights
        from .wsp import load_edited
        tokenizer, model = models.load_llm(llm)
        return tokenizer, load_edited(model, paths, parse_weights(weights, len(paths)) if weights else None, edit)
    if len(paths) <= 1 and not weights:
        return models.load_llm(llm, paths[0] if paths else None)
    from .export_models import merge_mean, parse_weights
    tokenizer, model = models.load_llm(llm)
    return tokenizer, merge_mean(model, paths, parse_weights(weights, len(paths)) if weights else None)


def worker(args) -> None:
    from . import models, scorer
    import torch

    models.seed_everything(args.seed)
    root = data.find_root(args.split)
    items = data.load_split(args.split, root)
    catalog = data.load_catalog(args.split, root)
    tool_env = data.import_tool_env(args.split, root)
    if args.ids_file:
        keep = data.id_set(args.ids_file)
        items = [i for i in items if i["id"] in keep]
    if args.limit:
        items = items[: args.limit]
    items = items[args.shard::args.shards]

    tokenizer, model = load_adapted(args.llm, args.adapter, args.adapter_weights, args.edit)
    model.eval()
    print("llm parameters", models.count_parameters(model), flush=True)
    gen = Generator(tokenizer, model, args.max_new_tokens, ask_bias=args.ask_bias, lookup=args.lookup)
    if args.vote:  # improve_plan.md step 3: the soup and each of its two adapters vote on every call
        from .vote import DeltaSwitch, VotingGenerator
        switch = DeltaSwitch(model, args.adapter.split(","))
        print("vote: soup + 2 adapters, hook parameters", switch.parameters, flush=True)
        gen = VotingGenerator(gen, switch, catalog)
    transcripts = None if args.transcripts == "gold" else load_transcripts(args.transcripts, {i["audio"] for i in items})
    fold_of, backend, loaded = {}, None, None
    if args.relisten_asr:  # method A's second pass; with '{fold}', each sample uses the fold model that never saw it
        from . import asr
        if "{fold}" in args.relisten_asr:
            from .asr_train import sample_folds
            fold_of = sample_folds(data.load_split("train", data.find_root("train")), args.relisten_folds, args.relisten_seed)
            items = sorted(items, key=lambda i: fold_of[i["id"]])
    out = Path(args.out)
    with open(out / f"submission.{args.shard}.jsonl", "w", encoding="utf-8") as fout, \
            open(out / f"detail.{args.shard}.jsonl", "w", encoding="utf-8") as fdet:
        for n, item in enumerate(items):
            start = time.monotonic()
            env = tool_env(args.split, item["id"], root=root)  # exactly one per sample
            texts = prompts.gold_transcripts(item) if transcripts is None else transcripts
            relisten = None
            if args.relisten_asr:
                want = args.relisten_asr.replace("{fold}", str(fold_of.get(item["id"], "")))
                if want != loaded:
                    backend = None
                    torch.cuda.empty_cache()
                    backend, loaded = asr.Whisper(args.asr_model, adapter=resolve_adapter(want)), want
                relisten = make_relisten(backend, item, texts, catalog, str(Path(root) / args.split / item["audio"]))
            stats = run_agent(gen, env, prompts.prompt_text(item, texts, catalog), start + args.budget_s, args.max_steps,
                              denoise=args.denoise, catalog=catalog, broaden=args.broaden_search, relisten=relisten,
                              first_tokens=args.first_tokens)
            calls, state_hash = env.get_tool_calls(), env.get_state_hash()
            fout.write(json.dumps({"id": item["id"], "tool_calls": calls, "state_hash": state_hash}, ensure_ascii=False) + "\n")
            fout.flush()
            row = {"id": item["id"], "seconds": round(time.monotonic() - start, 2), **stats, "request": texts[item["audio"]]}
            if gen.lookup:  # cumulative counts; a sample's own are the difference from the row before
                row["lookup"] = dict(gen.lookup_stats)
            if args.vote:
                row["vote_trace"], gen.trace = gen.trace, []
            if "label" in item:
                row["gold"] = item["label"]["tool_calls"]
                row["pred"] = calls
                gold_hash = scorer.gold_hash(tool_env, args.split, item, root)
                row["hash_ok"] = gold_hash == state_hash
                row["score"] = scorer.score_sample(calls, state_hash, item["label"]["tool_calls"], gold_hash, catalog)
            fdet.write(json.dumps(row, ensure_ascii=False) + "\n")
            fdet.flush()
            print(f"{n + 1}/{len(items)} {item['id']} {row['seconds']}s steps={stats['steps']} {stats['stop']} "
                  f"score={row.get('score')}", flush=True)
    del torch


def merge(args) -> dict:
    out = Path(args.out)
    rows, lines = [], {}
    for f in sorted(out.glob("detail.*.jsonl")):
        rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    for f in sorted(out.glob("submission.*.jsonl")):
        for x in f.read_text(encoding="utf-8").splitlines():
            if x.strip():
                lines[json.loads(x)["id"]] = x
    root = data.find_root(args.split)
    order = [i["id"] for i in data.load_split(args.split, root)]
    kept = [lines[i] for i in order if i in lines]
    (out / "submission.jsonl").write_text("\n".join(kept) + "\n", encoding="utf-8")
    seconds = sorted(r["seconds"] for r in rows)
    metrics = {"split": args.split, "samples": len(rows), "llm": args.llm, "adapter": args.adapter,
               "transcripts": args.transcripts, "edit": getattr(args, "edit", None), "seconds_mean": sum(seconds) / max(len(seconds), 1),
               "seconds_max": seconds[-1] if seconds else None, "over_30s": sum(s > 30 for s in seconds),
               "stops": {s: sum(r["stop"] == s for r in rows) for s in {r["stop"] for r in rows}},
               "broaden_search": getattr(args, "broaden_search", False), "ask_bias": getattr(args, "ask_bias", 0.0),
               "broadened": sum(r.get("broadened", 0) for r in rows), "ask_flips": sum(r.get("ask_flips", 0) for r in rows)}
    traces = [r["vote_trace"] for r in rows if "vote_trace" in r]
    if traces:
        turns = [t for trace in traces for t in trace]
        metrics["votes"] = {"turns": len(turns), "all_agree": sum(len({json.dumps(k) for k in t["keys"]}) == 1 for t in turns),
                            "soup_outvoted": sum(t["keys"][t["chosen"]] != t["keys"][0] for t in turns),
                            "samples_outvoted": sum(any(t["keys"][t["chosen"]] != t["keys"][0] for t in trace) for trace in traces)}
    if rows and "score" in rows[0]:
        metrics["score"] = 100 * sum(r["score"] for r in rows) / len(rows)
        metrics["hash_ok_pct"] = 100 * sum(r["hash_ok"] for r in rows) / len(rows)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics


def main(argv=None) -> None:
    from .models import DEFAULT_LLM
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--out", default="outputs/infer")
    ap.add_argument("--llm", default=DEFAULT_LLM)
    ap.add_argument("--adapter", default=None, help="adapter folder, or 'auto' to find it under /kaggle/input")
    ap.add_argument("--adapter-weights", default=None,
                    help="one weight per --adapter, comma-separated (fractions allowed, negative too): merge the weighted "
                         "sum of the deltas instead of their mean")
    ap.add_argument("--edit", default=None, help="weight-space edit of the adapter change (lifemate.wsp program, JSON)")
    ap.add_argument("--transcripts", default="gold", help="'gold' (train only), a JSON file {audio: text} or a folder of them")
    ap.add_argument("--ids-file", default=None, help="JSON list of sample ids to keep")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--first-tokens", type=int, default=0, help="token limit of the first turn (worked notes); 0 = --max-new-tokens")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--budget-s", type=float, default=25.0, help="no new step after this long; the limit is 30 s")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--denoise", action="store_true")
    ap.add_argument("--broaden-search", action="store_true", help="after an empty read with an optional name filter, read without it")
    ap.add_argument("--ask-bias", type=float, default=0.0, help="logit penalty on starting a sys.ask_missing_info call")
    ap.add_argument("--lookup", type=int, default=0, help="speed only: prompt-lookup drafts of up to N tokens, checked in one pass (0 = off)")
    ap.add_argument("--vote", action="store_true", help="step 3: --adapter a,b; the soup and a and b alone vote on every call")
    ap.add_argument("--relisten-asr", default="", help="method A pass 2: Whisper adapter ('{fold}' = the sample's fold model)")
    ap.add_argument("--asr-model", default="vinai/PhoWhisper-medium")
    ap.add_argument("--relisten-folds", type=int, default=3)
    ap.add_argument("--relisten-seed", type=int, default=2026)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--shards", type=int, default=1)
    args, _ = ap.parse_known_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.adapter:  # comma-separated: a soup, the mean of their deltas
        args.adapter = ",".join(resolve_adapter(a) for a in args.adapter.split(","))
    if args.shard >= 0:
        worker(args)
        if args.shards == 1:
            merge(args)
        return
    argv_rest = sys.argv[1:] if argv is None else argv
    argv_rest = [a for a in argv_rest]
    if args.adapter:
        argv_rest = _replace_arg(argv_rest, "--adapter", args.adapter)
    procs = []
    for gpu in range(args.gpus):
        cmd = [sys.executable, "-m", "lifemate.infer", *argv_rest, "--shard", str(gpu), "--shards", str(args.gpus)]
        procs.append(subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}))
    codes = [p.wait() for p in procs]
    print("worker exit codes", codes, flush=True)
    merge(args)
    if any(codes):
        raise SystemExit(1)


def _replace_arg(argv: list, flag: str, value: str) -> list:
    out = list(argv)
    if flag in out:
        out[out.index(flag) + 1] = value
    else:
        out += [flag, value]
    return out


if __name__ == "__main__":
    main()
