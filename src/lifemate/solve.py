"""The final offline solution: audio in, submission.jsonl out, ASR and LLM in one process on one GPU.

    python -m lifemate.solve --data /data --out result --llm models/llm --asr models/asr_medium

--data is the folder of a split (same layout as public_test) or a folder holding the split folder. Each sample gets exactly one
ToolEnv, and only zaic_tools touches its data. Models load from local paths, so nothing here needs a network.
"""
import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BUDGET_S = 27.0  # the limit is 30 s per sample; no new LLM step starts after this long
HARD_S = 28.5  # and a step still decoding stops here, so a slow last step cannot run past the limit


def locate_split(data_dir: Path) -> tuple:
    """(split name, root) such that root/split/split.jsonl exists, as ToolEnv expects."""
    for f in data_dir.glob("*.jsonl"):
        if f.stem != "sample_submission" and (data_dir / f"{f.stem}_environment").is_dir():
            # The split folder itself was mounted: give ToolEnv a parent folder that holds it.
            root = Path(tempfile.mkdtemp(prefix="lifemate_root_"))
            (root / f.stem).symlink_to(data_dir.resolve(), target_is_directory=True)
            return f.stem, root
    for d in data_dir.iterdir():
        if d.is_dir() and (d / f"{d.name}.jsonl").is_file():
            return d.name, data_dir
    raise SystemExit(f"no split found under {data_dir}")


def seed_all(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--out", default="result")
    ap.add_argument("--llm", required=True, help="local folder with the merged, text-only LLM")
    ap.add_argument("--asr", required=True, help="comma-separated local folders of ASR checkpoints")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--first-tokens", type=int, default=0, help="token limit of the first turn (worked notes); 0 = --max-new-tokens")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--denoise", action="store_true")
    ap.add_argument("--view-adapters", default=None,
                    help="the readings view (lifemate.view, ASR_IMPROVEMENT_PLAN.md): --asr is then the base Whisper folder, "
                         "and these adapters, joined by '+' in the training order, give the main reading (the mean of "
                         "their deltas, merged at load time) and, for the latest request, each adapter's own reading, "
                         "in braces where they disagree with the main one (asr.WhisperReadings)")
    ap.add_argument("--broaden-search", action="store_true", help="after an empty read with an optional name filter, read without it")
    ap.add_argument("--ask-bias", type=float, default=0.0, help="logit penalty on starting a sys.ask_missing_info call")
    ap.add_argument("--lookup", type=int, default=0, help="speed only: prompt-lookup drafts of up to N tokens (0 = off; see infer.Generator)")
    ap.add_argument("--stack-model", default=None,
                    help="lifemate.stack_view chooser (JSON): picks the main reading of the latest request among the readings")
    ap.add_argument("--asr-device", default=None,
                    help="Kaggle T4 check only: put the speech models on another GPU (e.g. cuda:1), since a 15 GB T4 cannot "
                         "hold both models; the package runs both on its one A30 (24 GB)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0, help="with --shards: run every n-th sample from this one (one process per GPU)")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--budget-s", type=float, default=BUDGET_S, help="only for a CPU smoke test of the package")
    ap.add_argument("--hard-s", type=float, default=HARD_S, help="only for a CPU smoke test of the package")
    args = ap.parse_args(argv)

    import torch

    from . import asr, data, models, prompts, view
    from .infer import Generator, merge_readings, run_agent

    seed_all(args.seed)
    from importlib.metadata import version
    print("versions:", {"torch": torch.__version__, "cuda": torch.version.cuda,
                        **{p: version(p) for p in ("transformers", "peft", "accelerate", "numpy", "soundfile")},
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, flush=True)
    if args.data.startswith("auto:"):  # a Kaggle check: the split wherever the data package is mounted
        split = args.data[len("auto:"):]
        root = data.find_root(split)
    else:
        split, root = locate_split(Path(args.data))
    items = data.load_split(split, root)
    catalog = data.load_catalog(split, root)
    tool_env = data.import_tool_env(split, root)
    if args.limit:
        items = items[: args.limit]
    items = items[args.shard::args.shards]

    if args.view_adapters:
        readers = [asr.WhisperReadings(args.asr, args.view_adapters, device=args.asr_device)]
    else:  # without the view, differing readings of several --asr models are joined by ' | '
        readers = [asr.load(p) for p in args.asr.split(",")]
    tokenizer, model = models.load_llm(args.llm)
    model.eval()
    params = models.count_parameters(model)["total"] + sum(getattr(r, "distinct_parameters", None) or
                                                           sum(p.numel() for p in r.model.parameters()) for r in readers)
    print(f"{split}: {len(items)} samples, {params / 1e9:.3f}B parameters in total", flush=True)
    if params >= 6e9:
        raise SystemExit("over the 6B parameter limit")
    gen = Generator(tokenizer, model, args.max_new_tokens, ask_bias=args.ask_bias, lookup=args.lookup)
    stack = None
    if args.stack_model:
        from . import stack_view
        stack = stack_view.load_model(args.stack_model)
        print("stack model", args.stack_model, flush=True)
    warm_up(readers, gen)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total_s, slowest = 0.0, 0.0
    with open(out / "submission.jsonl", "w", encoding="utf-8") as fout, open(out / "timing.jsonl", "w", encoding="utf-8") as flog:
        for n, item in enumerate(items):
            start = time.monotonic()
            env, stats, error = None, {}, None
            gen.stop_at, cut_before = start + args.hard_s, gen.cut
            if torch.cuda.is_available():
                for d in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(d)
            try:  # one failing sample must not cost the lines of the others
                env = tool_env(split, item["id"], root=root)  # exactly one per sample
                audio = [t["audio"] for t in item["history"] if t["role"] == "user" and t.get("audio")] + [item["audio"]]
                paths = [str(Path(root) / split / a) for a in audio]
                if args.view_adapters:  # braces on the latest request only, as in training (audio[-1] is the latest)
                    got = readers[0].transcribe(paths, 8, alternatives=[len(paths) - 1])
                    texts = {a: view.build(stack_view.restack(r[0], r[1:], stack) if stack else r[0], r[1:])
                             if len(r) > 1 else r[0] for a, r in zip(audio, got)}
                else:
                    readings = [r.transcribe(paths, 8) for r in readers]
                    texts = {a: merge_readings([reading[i] for reading in readings]) for i, a in enumerate(audio)}
                stats = run_agent(gen, env, prompts.prompt_text(item, texts, catalog), start + args.budget_s, args.max_steps,
                                  denoise=args.denoise, catalog=catalog, broaden=args.broaden_search,
                                  first_tokens=args.first_tokens)
            except Exception as e:  # noqa: BLE001  (logged, and the calls made so far still go in the file)
                error = f"{type(e).__name__}: {e}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            line = {"id": item["id"], "tool_calls": env.get_tool_calls() if env else [], "state_hash": env.get_state_hash() if env else ""}
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")
            fout.flush()
            seconds = time.monotonic() - start
            total_s, slowest = total_s + seconds, max(slowest, seconds)
            flog.write(json.dumps({"id": item["id"], "seconds": round(seconds, 2), "steps": stats.get("steps"), "stop": stats.get("stop"),
                                   "broadened": stats.get("broadened"), "ask_flips": stats.get("ask_flips"), "hard_stop": gen.cut - cut_before, "error": error,
                                   "gpu_peak_gb": round(sum(torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())) / 2**30, 2)
                                   if torch.cuda.is_available() else None,
                                   "gpu_reserved_gb": round(sum(torch.cuda.max_memory_reserved(d) for d in range(torch.cuda.device_count())) / 2**30, 2)
                                   if torch.cuda.is_available() else None}, ensure_ascii=False) + "\n")
            flog.flush()
            print(f"{n + 1}/{len(items)} {item['id']} {seconds:.1f}s steps={stats.get('steps')} {stats.get('stop')}{' ' + error if error else ''}", flush=True)
    print(f"mean {total_s / max(len(items), 1):.1f} s, slowest {slowest:.1f} s", flush=True)
    del torch


def warm_up(readers: list, gen) -> None:
    """The first calls on a GPU compile kernels and fill allocator caches. Do that here, not in the first timed sample."""
    import numpy as np
    import soundfile as sf

    from . import prompts
    path = Path(tempfile.mkdtemp(prefix="lifemate_warm_")) / "silence.wav"
    sf.write(str(path), np.zeros(16000, dtype=np.float32), 16000)
    for r in readers:
        r.transcribe([str(path)], 1)
    gen.generate(gen.encode(prompts.turn("user", "Xin chào") + prompts.GEN_PREFIX), None, 0)
    gen.flips = 0


if __name__ == "__main__":
    main()
