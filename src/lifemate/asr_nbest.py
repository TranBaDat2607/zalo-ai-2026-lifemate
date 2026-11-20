"""Out-of-fold runner-up hypotheses of the averaged speech model on latest requests (improve_plan.md step 1c, check).

    python -m lifemate.asr_nbest --out {out}/nbest --adapter auto:asr_medium/fold{fold}/best+... --fold 0 --fold 2

For each held-out fold, the fold's averaged model (the five adapters trained without that fold, as for
avg5_train.json) decodes every latest request of the fold's samples alone with beam search (``asr.beam_decode``,
5 beams, 5 kept). Writes ``{out}/fold{k}_nbest.json`` ({audio: [hypothesis texts, best first, distinct]}) and
``{out}/fold{k}_timing.json`` ({audio: seconds of the beam decode, encoder included}), timed one clip at a time on the
GPU as at answer time. Only train.jsonl and the audio are read.
"""
import argparse
import contextlib
import json
import time
from pathlib import Path

from . import data
from .asr_train import sample_folds


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="vinai/PhoWhisper-medium")
    ap.add_argument("--adapter", required=True, help="'{fold}' is replaced by the fold number; 'auto:' searches /kaggle/input; "
                                                     "several joined by '+' are averaged into one model")
    ap.add_argument("--fold", type=int, action="append", required=True)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026, help="the fold assignment of asr_train (exp7: 3 folds, seed 2026)")
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--keep", type=int, default=5)
    ap.add_argument("--limit-clips", type=int, default=0, help="first N clips of each fold only (a smoke test)")
    args = ap.parse_args(argv)
    import torch
    from . import asr
    from .infer import resolve_adapter
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root_dir = data.find_root("train")
    items = data.load_split("train", root_dir)
    folds = sample_folds(items, args.folds, args.seed)
    for fold in args.fold:
        adapter = "+".join(resolve_adapter(s) for s in args.adapter.replace("{fold}", str(fold)).split("+"))
        print("fold", fold, "adapter", adapter, flush=True)
        backend = asr.Whisper(args.model, adapter=adapter)
        clips = [it["audio"] for it in items if folds[it["id"]] == fold]
        if args.limit_clips:
            clips = clips[:args.limit_clips]
        tok = backend.processor.tokenizer
        texts, seconds = {}, {}
        started = time.monotonic()
        for n, audio in enumerate(clips):
            wave = asr.read_audio(str(Path(root_dir) / "train" / audio))
            t0 = time.monotonic()
            feats = asr.log_mel(backend.processor.feature_extractor, [wave]).to(backend.device, dtype=backend.dtype)
            with torch.autocast("cuda", dtype=torch.float16) if backend.device == "cuda" else contextlib.nullcontext():
                hyps = asr.beam_decode(backend.model, feats, backend.prompt, backend.eos, asr.token_limits([wave])[0],
                                       beams=args.beams, keep=args.keep)
            if backend.device == "cuda":
                torch.cuda.synchronize()
            seconds[audio] = round(time.monotonic() - t0, 4)
            distinct = []
            for tokens, _ in hyps:
                text = tok.decode(tokens, skip_special_tokens=True).strip()
                if text not in distinct:
                    distinct.append(text)
            texts[audio] = distinct
            if (n + 1) % 100 == 0:
                print(json.dumps({"fold": fold, "done": n + 1, "of": len(clips),
                                  "min": round((time.monotonic() - started) / 60, 1)}), flush=True)
        (out / f"fold{fold}_nbest.json").write_text(json.dumps(texts, ensure_ascii=False, indent=0), encoding="utf-8")
        (out / f"fold{fold}_timing.json").write_text(json.dumps(seconds, indent=0), encoding="utf-8")
        mean = sum(seconds.values()) / max(1, len(seconds))
        print(json.dumps({"fold": fold, "clips": len(clips), "mean_s": round(mean, 3),
                          "min": round((time.monotonic() - started) / 60, 1)}), flush=True)
        del backend
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
