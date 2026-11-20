"""External Vietnamese speech for ASR training (14 Nov 2026, the final round): FPT Open Speech Dataset (FOSD, CC BY 4.0,
released 2018, Hugging Face mirror doof-ferb/fpt_fosd) as extra clips for asr_train --synth.

    python -m lifemate.ext_speech --out {out}/ext --clips 6000

Writes 16 kHz mono WAV files and synth_manifest_ext.json ({id, audio, text, fold: 99, source, seconds}); fold 99 (no real fold) means
the clip is never held out, so every fold model and the all-data model train on it (it is never evaluated). FOSD's
transcripts are lower case without punctuation; the first letter is capitalised and a full stop added, the rest is
left as written (numbers stay in words, as many train requests say them). Needs internet (a Kaggle kernel).
"""
import argparse
import io
import json
import random
from pathlib import Path


def normalise(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".?!" else text + "."


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="doof-ferb/fpt_fosd")
    ap.add_argument("--clips", type=int, default=6000)
    ap.add_argument("--min-s", type=float, default=1.0)
    ap.add_argument("--max-s", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)

    import librosa
    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    out = Path(args.out)
    (out / "ext").mkdir(parents=True, exist_ok=True)
    ds = load_dataset(args.repo, split="train").cast_column("audio", Audio(decode=False))
    print("rows", len(ds), "columns", ds.column_names, flush=True)
    order = list(range(len(ds)))
    random.Random(args.seed).shuffle(order)
    manifest, tried, failed = [], 0, 0
    for i in order:
        if len(manifest) >= args.clips:
            break
        row = ds[i]
        text = normalise(row.get("transcription") or "")
        if len(text.split()) < 2:
            continue
        tried += 1
        try:
            audio, _ = librosa.load(io.BytesIO(row["audio"]["bytes"]), sr=16000, mono=True)
        except Exception as e:  # noqa: BLE001  (a broken clip is skipped)
            failed += 1
            if failed <= 5:
                print("decode failed", i, type(e).__name__, e, flush=True)
            continue
        seconds = len(audio) / 16000
        if not args.min_s <= seconds <= args.max_s or not np.isfinite(audio).all():
            continue
        cid = f"ext_{i:06d}"
        sf.write(str(out / "ext" / f"{cid}.wav"), audio, 16000, subtype="PCM_16")
        manifest.append({"id": cid, "audio": f"ext/{cid}.wav", "text": text, "fold": 99, "source": f"{args.repo}:{i}",
                         "seconds": round(seconds, 2)})
        if len(manifest) % 500 == 0:
            print(len(manifest), "clips", flush=True)
    (out / "synth_manifest_ext.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    hours = sum(m["seconds"] for m in manifest) / 3600
    print(json.dumps({"clips": len(manifest), "hours": round(hours, 2), "tried": tried, "decode_failed": failed,
                      "examples": [m["text"] for m in manifest[:5]]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
