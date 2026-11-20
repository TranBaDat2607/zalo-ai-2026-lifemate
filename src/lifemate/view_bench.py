"""Stage 4 check of the readings view (ASR_IMPROVEMENT_PLAN.md): the package's speech step with the five all-data
adapters (asr.WhisperReadings), timed per request the way solve.py runs it. Today's step is the averaged model on every
user turn of the sample (``main_s``); the view adds the five single readings of the latest request only
(``singles_s``, five weight switches included; decoded with the main reading as draft, asr.lookup_decode, unless
``--plain``). The readings are saved so they can be compared with the saved ones the
agent is trained and validated on (avg5_public_test.json and the five single readings). Results are rewritten every
``--save-every`` samples, so a run stopped at its deadline keeps what it measured.

    python -m lifemate.view_bench --out {out}/view_bench --model vinai/PhoWhisper-medium --shard 0 --shards 2 \
        --adapter auto:asr_medium/full/best+auto:asr_m10/full/best+auto:b/full/best+auto:s2027/full/best+auto:s2028/full/best
"""
import argparse
import json
import time
from pathlib import Path

from . import data


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="vinai/PhoWhisper-medium")
    ap.add_argument("--adapter", required=True, help="the adapters joined by '+', the order of the saved readings")
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8, help="solve.py's batch")
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--plain", action="store_true", help="single readings by plain batched decoding (view-bench3)")
    args = ap.parse_args(argv)

    import torch

    from . import asr, view
    from .infer import resolve_adapter

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = data.find_root(args.split)
    items = data.load_split(args.split, root)
    items = (items[:args.limit] if args.limit else items)[args.shard::args.shards]
    adapters = "+".join(resolve_adapter(a) for a in args.adapter.split("+"))
    start = time.monotonic()
    reader = asr.WhisperReadings(args.model, adapters)
    cuda = torch.cuda.is_available()
    info = {"load_s": round(time.monotonic() - start, 1), "distinct parameters": reader.distinct_parameters,
            "averaged model parameters": sum(p.numel() for p in reader.main.model.parameters()),
            "gpu_gb_after_load": round(torch.cuda.memory_allocated() / 2**30, 2) if cuda else None}
    print("loaded", info, flush=True)

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def clips(item):
        audio = [t["audio"] for t in item["history"] if t["role"] == "user" and t.get("audio")] + [item["audio"]]
        return audio, [str(Path(root) / args.split / a) for a in audio]

    _, warm = clips(items[0])  # first calls compile kernels and fill caches: not in the first timed sample
    reader.transcribe(warm[:1], 1)
    sync()
    switch = time.monotonic()
    for k in range(reader.swap.count):
        reader.swap.use(k)
    sync()
    info["switch_s"] = round((time.monotonic() - switch) / reader.swap.count, 4)
    main_reading, singles, rows = {}, {}, []

    def save():
        (out / f"main.{args.shard}.json").write_text(json.dumps(main_reading, ensure_ascii=False, indent=0), encoding="utf-8")
        (out / f"singles.{args.shard}.json").write_text(json.dumps(singles, ensure_ascii=False, indent=0), encoding="utf-8")
        (out / f"timing.{args.shard}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        info["gpu_gb_peak"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None
        (out / f"info.{args.shard}.json").write_text(json.dumps(info), encoding="utf-8")

    for n, item in enumerate(items):
        audio, paths = clips(item)
        sync()
        t0 = time.monotonic()
        main, tokens = reader.main.transcribe_ids(paths, args.batch_size)
        sync()
        t1 = time.monotonic()
        alt = reader.singles(paths[-1:], args.batch_size, None if args.plain else tokens[-1:])[0]
        shown = view.build(main[-1], alt)
        sync()
        t2 = time.monotonic()
        main_reading.update(zip(audio, main))
        singles[audio[-1]] = alt
        rows.append({"id": item["id"], "clips": len(paths), "main_s": round(t1 - t0, 3), "singles_s": round(t2 - t1, 3),
                     "braces": shown.count("{"), "tokens": len(tokens[-1]),
                     "passes": None if args.plain else getattr(reader, "passes", None)})
        if n % args.save_every == 0 or n == len(items) - 1:
            save()
            print(f"{n + 1}/{len(items)} {item['id']} clips {len(paths)} main {t1 - t0:.2f}s singles {t2 - t1:.2f}s", flush=True)
    save()
    print("done", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
