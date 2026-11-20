"""Join the out-of-fold transcripts of all folds into one file and report their WER.

    python -m lifemate.merge_oof --dir outputs/asr --name medium

Writes <dir>/<name>_train.json. A public test transcript from the all-data model, if present, is copied to
<dir>/<name>_public_test.json, so that later kernels can find both by name.
"""
import argparse
import json
from pathlib import Path

from . import data
from .transcribe import wer_against_gold


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args(argv)
    folder = Path(args.dir)
    merged = {}
    for f in sorted(folder.glob("oof_*.json")):
        part = json.loads(f.read_text(encoding="utf-8"))
        print(f.name, len(part), flush=True)
        merged.update(part)
    root = data.find_root("train")
    items = data.load_split("train", root)
    wanted = {a for i in items for a in [i["audio"]] + [t["audio"] for t in i["history"] if t["role"] == "user" and t.get("audio")]}
    missing = wanted - set(merged)
    print(f"{len(merged)} clips, {len(missing)} missing, WER {wer_against_gold(items, merged):.2f}", flush=True)
    (folder / f"{args.name}_train.json").write_text(json.dumps(merged, ensure_ascii=False, indent=0), encoding="utf-8")
    public = folder / "public_test.json"
    if public.exists():
        (folder / f"{args.name}_public_test.json").write_text(public.read_text(encoding="utf-8"), encoding="utf-8")
    if missing:
        raise SystemExit(f"{len(missing)} clips have no transcript, first: {sorted(missing)[0]}")


if __name__ == "__main__":
    main()
