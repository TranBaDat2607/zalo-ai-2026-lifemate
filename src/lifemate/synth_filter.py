"""Drop synthetic clips that a speech model cannot follow, i.e. where the TTS skipped or garbled words.

    python -m lifemate.synth_filter --out /kaggle/working --adapter auto:asr_medium/fold{fold}/best --fold 0 --fold 2 --part a
    python -m lifemate.synth_filter --out /kaggle/working --merge

Reads every synth_manifest_*.json under /kaggle/input (outputs of ``synth_tts``) and decodes the clips. With
``--adapter`` containing ``{fold}``, the clips made from fold k's texts are decoded by fold k's out-of-fold model, which
never trained on those texts (the base model, used when ``--adapter`` is empty, loops and mangles English words on
this domain: log.md, 5 Nov 2026). Writes ``{out}/synth_keep.json`` (or ``synth_keep_<part>.json`` for a share of the
folds, joined by ``--merge``): the ids whose normalised WER against their text is at most ``--max-wer``, the WER
distribution, and each clip's transcript (``hyp``), which combination C uses as the clip's first-pass text.
"""
import argparse
import json
import os
from pathlib import Path

from .asr_eval import edit_distance, normal_tokens


def load_manifests(base: str = "/kaggle/input") -> list:
    """Synthetic clips with absolute audio paths, from every synth_manifest_*.json found under ``base``."""
    out = []
    for d, _, files in os.walk(base):
        for name in sorted(files):
            if name.startswith("synth_manifest_") and name.endswith(".json"):
                for entry in json.loads((Path(d) / name).read_text(encoding="utf-8")):
                    out.append({**entry, "path": str(Path(d) / entry["audio"])})
    return out


def summary(wers: list, kept: int, max_wer: float) -> dict:
    wers = sorted(wers)
    q = lambda p: round(wers[min(len(wers) - 1, int(p * len(wers)))], 3) if wers else None  # noqa: E731
    return {"clips": len(wers), "kept": kept, "max_wer": max_wer, "wer_p10": q(0.1), "wer_p50": q(0.5),
            "wer_p90": q(0.9), "wer_mean": round(sum(wers) / max(1, len(wers)), 3)}


def merge(out: Path, max_wer: float) -> None:
    keep, hyp, wers = [], {}, {}
    for f in sorted(out.glob("synth_keep_*.json")):
        part = json.loads(f.read_text(encoding="utf-8"))
        keep += part["keep"]
        hyp.update(part["hyp"])
        wers.update(part["wer"])
    stats = summary(list(wers.values()), len(keep), max_wer)
    print(json.dumps(stats), flush=True)
    (out / "synth_keep.json").write_text(json.dumps({"keep": sorted(keep), "stats": stats, "hyp": hyp, "wer": wers},
                                                    ensure_ascii=False), encoding="utf-8")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="vinai/PhoWhisper-medium")
    ap.add_argument("--adapter", default="", help="'{fold}' = the fold of the clip's source text; 'auto:' searches /kaggle/input")
    ap.add_argument("--fold", type=int, action="append", help="only clips of these folds (default: all)")
    ap.add_argument("--part", default="", help="write synth_keep_<part>.json, for --merge")
    ap.add_argument("--merge", action="store_true", help="join every synth_keep_*.json in --out into synth_keep.json")
    ap.add_argument("--max-wer", type=float, default=0.4)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(out, args.max_wer)
        return
    from . import asr
    from .infer import resolve_adapter
    clips = load_manifests()
    folds = sorted({c["fold"] for c in clips}) if args.fold is None else args.fold
    print(f"{len(clips)} synthetic clips; folds {folds}", flush=True)
    keep, hyp, wers = [], {}, {}
    for fold in folds:
        part = [c for c in clips if c["fold"] == fold]
        adapter = resolve_adapter(args.adapter.replace("{fold}", str(fold))) if args.adapter else None
        print(f"fold {fold}: {len(part)} clips, adapter {adapter}", flush=True)
        backend = asr.Whisper(args.model, adapter=adapter) if adapter else asr.Whisper(args.model)
        texts = backend.transcribe([c["path"] for c in part], args.batch_size)
        del backend
        for c, text in zip(part, texts):
            ref = normal_tokens(c["text"])
            wer = edit_distance(ref, normal_tokens(text)) / max(1, len(ref))
            hyp[c["id"]], wers[c["id"]] = text, round(wer, 4)
            if wer <= args.max_wer:
                keep.append(c["id"])
        print(json.dumps({"fold": fold, **summary([wers[c["id"]] for c in part], sum(wers[c["id"]] <= args.max_wer for c in part), args.max_wer)}), flush=True)
    stats = summary(list(wers.values()), len(keep), args.max_wer)
    print(json.dumps(stats), flush=True)
    name = f"synth_keep_{args.part}.json" if args.part else "synth_keep.json"
    (out / name).write_text(json.dumps({"keep": keep, "stats": stats, "hyp": hyp, "wer": wers}, ensure_ascii=False),
                            encoding="utf-8")


if __name__ == "__main__":
    main()
