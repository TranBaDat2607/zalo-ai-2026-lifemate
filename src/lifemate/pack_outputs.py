"""Pack the outputs of an attached kernel into one zip, so that thousands of small files download as one.

    python -m lifemate.pack_outputs --out /kaggle/working --marker synth_manifest_0.json --zip synth_audio.zip

Finds the folder under /kaggle/input that holds ``--marker`` and stores everything in it (WAV barely compresses, so the
files are stored, not deflated). A CPU kernel is enough; it reads only the attached outputs.
"""
import argparse
import json
import time
import zipfile
from pathlib import Path

from . import data


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--marker", required=True, help="a file name that marks the folder to pack")
    ap.add_argument("--zip", default="outputs.zip")
    args = ap.parse_args(argv)
    folder = next(data.walk_input(args.marker))
    started = time.monotonic()
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix in (".wav", ".json"))
    target = Path(args.out) / args.zip
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as z:
        for p in files:
            z.write(p, p.relative_to(folder))
    print(json.dumps({"folder": str(folder), "files": len(files), "zip": str(target),
                      "mb": round(target.stat().st_size / 1e6, 1), "min": round((time.monotonic() - started) / 60, 1)}),
          flush=True)


if __name__ == "__main__":
    main()
