"""Cheap per-clip audio features for every train and public clip (final turn and history turns)."""
import csv
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA  # noqa: E402


def feats(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    info = sf.info(path)
    if x.ndim > 1:
        x = x.mean(1)
    n = len(x)
    dur = n / sr
    peak = float(np.max(np.abs(x))) if n else 0.0
    clip = float(np.mean(np.abs(x) > 0.99)) if n else 0.0
    # 20 ms frames
    fl = int(0.02 * sr)
    nf = max(1, n // fl)
    fr = x[: nf * fl].reshape(nf, fl) if n >= fl else x.reshape(1, -1)
    e = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-10)
    thr = max(e.max() - 35, -60)
    voiced = e > thr
    idx = np.where(voiced)[0]
    lead = idx[0] * 0.02 if len(idx) else dur
    trail = (nf - 1 - idx[-1]) * 0.02 if len(idx) else dur
    speech = voiced.mean() * dur
    noise = float(np.percentile(e, 10)); loud = float(np.percentile(e, 95))
    # bandwidth: frequency below which 99% of spectral energy lies (average spectrum)
    spec = np.abs(np.fft.rfft(fr * np.hanning(fr.shape[1]), axis=1)) ** 2
    ps = spec.mean(0); cum = np.cumsum(ps) / (ps.sum() + 1e-12)
    freqs = np.fft.rfftfreq(fr.shape[1], 1 / sr)
    bw99 = float(freqs[min(len(freqs) - 1, np.searchsorted(cum, 0.99))])
    # energy above 4 kHz share (phone-band detector)
    hi = float(ps[freqs > 4000].sum() / (ps.sum() + 1e-12))
    dc = float(np.mean(x)) if n else 0.0
    # 32 bands of 250 Hz: average log power, and the highest band within 45 dB of the loudest band (recording cut-off)
    bands = np.array([ps[(freqs >= b * 250) & (freqs < (b + 1) * 250)].sum() for b in range(32)]) + 1e-12
    lb = 10 * np.log10(bands)
    above = np.where(lb > lb.max() - 45)[0]
    cutoff = int((above.max() + 1) * 250) if len(above) else 0
    extra = {f"b{b:02d}": round(float(lb[b] - lb.max()), 1) for b in range(32)}
    return dict(cutoff=cutoff, **extra, path=os.path.relpath(path, DATA), sr=sr, subtype=info.subtype, channels=info.channels, dur=round(dur, 3),
                peak=round(peak, 4), clip=round(clip, 5), lead=round(lead, 3), trail=round(trail, 3), speech=round(speech, 3),
                noise_db=round(noise, 1), loud_db=round(loud, 1), snr_db=round(loud - noise, 1), bw99=round(bw99), hi4k=round(hi, 4),
                dc=round(dc, 5))


if __name__ == "__main__":
    files = sorted(glob.glob(str(DATA / "train" / "audio" / "*" / "*.wav")) + glob.glob(str(DATA / "public_test" / "audio" / "*" / "*.wav")))
    out = Path(sys.argv[1])
    with ProcessPoolExecutor(10) as ex:
        rows = list(ex.map(feats, files, chunksize=32))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(len(rows), "clips ->", out)
