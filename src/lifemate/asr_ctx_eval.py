"""Out-of-fold transcripts with context prompts: context-aware decoding against plain decoding on the same clips.

    python -m lifemate.asr_ctx_eval --out outputs/a0 --adapter auto:asr_medium/fold{fold}/best --fold 0 --fold 1

For each held-out fold, three passes over its clips with the fold's adapter (trained without that fold):
- ``none``: no prompt, what the system does today;
- ``ctx``: phrases from the user profile and earlier turns, picked by their likeness to the ``none`` text;
- ``reads`` (latest requests only): the same plus the results of the gold path's read calls, picked by their likeness to
  the ``ctx`` text. The gold reads stand in for the agent's own reads, which the end-to-end run uses instead.
Writes ``{out}/fold{k}_{pass}.json`` ({audio: text}) and ``{out}/fold{k}_prompts.json``. Only train.jsonl, the tool
catalog and the audio are read, never ``train_environment/``.
"""
import argparse
import json
import time
from pathlib import Path

from . import asr_context as C
from . import data
from .asr_train import clips_of, sample_folds

# Retrieval settings, fixed before any run (log.md, "Context-aware decoding and synthetic voices", 5 Nov 2026)
THRESHOLD, SINGLE_THRESHOLD, LIMIT, BUDGET = 0.5, 0.6, 20, 150


def fold_clips(items: list, folds: dict, fold: int, catalog: dict) -> list:
    """(audio, context pool, context + reads pool or None) for every user clip of the fold's samples."""
    out = []
    for item in items:
        if folds[item["id"]] != fold:
            continue
        clips = clips_of(item)
        reads = C.read_results(item["label"]["tool_calls"], item["label"].get("tool_results") or [], catalog)
        for n, ((audio, _), pool) in enumerate(zip(clips, C.turn_pools(item))):
            out.append((audio, pool, C.merge(pool, reads) if n == len(clips) - 1 else None))
    return out


def prompts_for(pools: list, texts: list, tokenizer) -> list:
    return [C.prompt_text(C.fit_tokens(C.select(pool, text, THRESHOLD, SINGLE_THRESHOLD, LIMIT), tokenizer, BUDGET))
            for pool, text in zip(pools, texts)]


def run_fold(backend, clips: list, root: Path, out: Path, tag: str, batch_size: int, plain_only: bool = False) -> None:
    paths = [str(root / audio) for audio, _, _ in clips]
    tok = backend.processor.tokenizer
    started = time.monotonic()
    none = backend.transcribe(paths, batch_size)
    print(json.dumps({"tag": tag, "pass": "none", "clips": len(paths), "min": round((time.monotonic() - started) / 60, 1)}), flush=True)
    if plain_only:
        (out / f"{tag}_none.json").write_text(json.dumps(dict(zip([a for a, _, _ in clips], none)), ensure_ascii=False, indent=0),
                                              encoding="utf-8")
        return
    ctx_prompts = prompts_for([pool for _, pool, _ in clips], none, tok)
    ctx = backend.transcribe(paths, batch_size, ctx_prompts)
    print(json.dumps({"tag": tag, "pass": "ctx", "min": round((time.monotonic() - started) / 60, 1)}), flush=True)
    finals = [i for i, c in enumerate(clips) if c[2] is not None]
    reads_prompts = prompts_for([clips[i][2] for i in finals], [ctx[i] for i in finals], tok)
    reads = backend.transcribe([paths[i] for i in finals], batch_size, reads_prompts)
    print(json.dumps({"tag": tag, "pass": "reads", "clips": len(finals), "min": round((time.monotonic() - started) / 60, 1)}), flush=True)
    audio = [a for a, _, _ in clips]
    write = lambda name, obj: (out / f"{tag}_{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=0), encoding="utf-8")  # noqa: E731
    write("none", dict(zip(audio, none)))
    write("ctx", dict(zip(audio, ctx)))
    write("reads", {audio[i]: t for i, t in zip(finals, reads)})
    write("prompts", {"ctx": dict(zip(audio, ctx_prompts)), "reads": {audio[i]: p for i, p in zip(finals, reads_prompts)}})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="vinai/PhoWhisper-medium")
    ap.add_argument("--adapter", required=True, help="'{fold}' is replaced by the fold number; 'auto:' searches /kaggle/input; "
                                                     "several joined by '+' are averaged into one model")
    ap.add_argument("--fold", type=int, action="append", required=True)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026, help="the fold assignment of asr_train (exp7: 3 folds, seed 2026)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--edit", default=None, help="weight-space edit program (lifemate.wsp, JSON) of the averaged change")
    ap.add_argument("--plain-only", action="store_true", help="only the plain pass (no context prompts): {out}/fold{k}_none.json")
    args = ap.parse_args(argv)
    from . import asr
    from .infer import resolve_adapter
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root_dir = data.find_root("train")
    items = data.load_split("train", root_dir)
    catalog = data.load_catalog("train", root_dir)
    folds = sample_folds(items, args.folds, args.seed)
    for fold in args.fold:
        adapter = "+".join(resolve_adapter(s) for s in args.adapter.replace("{fold}", str(fold)).split("+"))
        print("fold", fold, "adapter", adapter, flush=True)
        backend = asr.Whisper(args.model, adapter=adapter, edit=args.edit)
        run_fold(backend, fold_clips(items, folds, fold, catalog), Path(root_dir) / "train", out, f"fold{fold}", args.batch_size,
                 args.plain_only)
        del backend


if __name__ == "__main__":
    main()
