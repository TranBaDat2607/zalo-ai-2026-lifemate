"""Synthetic voices for method B (log.md, 5 Nov 2026): read the synthetic request texts aloud with VieNeu-TTS v3 Turbo
(Apache-2.0, 0.1B), each group of texts in a voice cloned from a train clip or in a preset voice.

    python -m lifemate.synth_tts --out /kaggle/working --texts auto:texts.json --shard 0 --shards 2

A development tool only: the competition allows synthetic data from other models during development, and nothing of
this module or of the TTS goes into the offline package. Needs ``vieneu[cuda]`` and a GPU. Writes
``{out}/synth/<id>.wav`` (16 kHz mono 16-bit) and ``{out}/synth_manifest_<shard>.json``, rewritten after every voice so that a
stopped run keeps what it made. Reference clips come from train audio only.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import data

RATE_IN, RATE_OUT = 48_000, 16_000


def find_input(spec: str) -> Path:
    if not spec.startswith("auto:"):
        return Path(spec)
    name = spec[len("auto:"):]
    for d, _, files in os.walk("/kaggle/input"):
        if name in files:
            return Path(d) / name
    raise FileNotFoundError(f"{name} not found under /kaggle/input")


def augment(y: np.ndarray, rng: random.Random) -> np.ndarray:
    """Studio-clean TTS against phone recordings: a random level, and light white noise on half the clips."""
    peak = float(np.max(np.abs(y))) or 1.0
    y = y / peak * rng.uniform(0.3, 0.95)
    if rng.random() < 0.5:
        snr_db = rng.uniform(20.0, 40.0)
        power = float(np.mean(y ** 2)) or 1e-8
        noise = np.random.default_rng(rng.randrange(2**31)).standard_normal(len(y)).astype(np.float32)
        y = y + noise * np.sqrt(power / 10 ** (snr_db / 10))
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--texts", default="auto:texts.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="voice groups per shard, for a smoke test")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--dtype", default="auto", help="auto | float16 | bfloat16 | float32 (T4 has no native bfloat16)")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)

    texts = json.loads(find_input(args.texts).read_text(encoding="utf-8"))
    groups = {}
    for t in texts:
        groups.setdefault(json.dumps(t["voice"], sort_keys=True, ensure_ascii=False), []).append(t)
    keys = [k for n, k in enumerate(sorted(groups)) if n % args.shards == args.shard]
    if args.limit:  # a smoke test: preset voices sort first and cloned voices last, so take some of each
        keys = keys[:args.limit // 2] + keys[len(keys) - (args.limit - args.limit // 2):]
    todo = sum(len(groups[k]) for k in keys)
    print(f"shard {args.shard}/{args.shards}: {len(keys)} voices, {todo} texts", flush=True)

    import soxr
    from vieneu import Vieneu
    tts = Vieneu(dtype=args.dtype)
    print(json.dumps({"backend": getattr(tts, "backend", "?"), "dtype": str(getattr(getattr(tts, "engine", None), "dtype", "?"))}),
          flush=True)
    root = Path(data.find_root("train")) / "train"
    out_dir = Path(args.out) / "synth"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.out) / f"synth_manifest_{args.shard}.json"
    rng = random.Random(args.seed * 100 + args.shard)
    manifest, failed, audio_s = [], 0, 0.0
    started = time.monotonic()
    for gi, key in enumerate(keys):
        voice, items = json.loads(key), groups[key]
        name = None
        try:
            if "ref" in voice:
                name = tts.add_voice(f"ref{args.shard}_{gi}", str(root / voice["ref"]), denoise=True)
            wavs = tts.infer_batch([t["text"] for t in items], voice=name or voice["preset"],
                                   temperature=args.temperature, apply_watermark=False)
        except Exception as e:  # one bad reference clip must not stop the run
            failed += len(items)
            print(json.dumps({"group": gi, "voice": voice, "error": repr(e)[:300]}, ensure_ascii=False), flush=True)
            continue
        finally:
            if name:
                try:
                    tts.remove_voice(name)
                except Exception:
                    pass
        for t, wav in zip(items, wavs):
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            seconds = len(wav) / RATE_IN
            if seconds < 0.5 or seconds > 30.0:
                failed += 1
                continue
            y = augment(soxr.resample(wav, RATE_IN, RATE_OUT).astype(np.float32), rng)
            sf.write(out_dir / f"{t['id']}.wav", y, RATE_OUT, subtype="PCM_16")
            audio_s += seconds
            manifest.append({"id": t["id"], "audio": f"synth/{t['id']}.wav", "text": t["text"], "fold": t["fold"],
                             "source": t["source"], "new": t.get("new", []), "voice": voice, "seconds": round(seconds, 2)})
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=0), encoding="utf-8")  # a killed run keeps it
        if args.limit or gi % 10 == 9 or gi == len(keys) - 1:
            spent = time.monotonic() - started
            done = sum(len(groups[k]) for k in keys[:gi + 1])
            print(json.dumps({"shard": args.shard, "voices": gi + 1, "of": len(keys), "clips": len(manifest),
                              "failed": failed, "audio_min": round(audio_s / 60, 1), "min": round(spent / 60, 1),
                              "x_realtime": round(audio_s / max(spent, 1e-6), 1),
                              "eta_min": round(spent / max(done, 1) * (todo - done) / 60, 1)}), flush=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"done: {len(manifest)} clips, {failed} failed, {audio_s / 3600:.2f} h of audio", flush=True)


if __name__ == "__main__":
    main()
