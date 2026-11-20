"""Plan 3 smoke test: how far apart the adapters of runs that should agree are, measured against how far they moved.

    python -m lifemate.p3_check --init auto:m1 --pair {out}/one,{out}/two --pair {out}/two,{out}/resumed --out {out}/check.json

For each pair (a, b) of run folders: rel = ||W_a - W_b|| / ||W_b - W_init|| over all LoRA tensors (W = the saved adapter),
plus the metrics.json of both runs (steps, minutes, tokens). 0 means identical; fp16 kernels on a GPU are not bitwise
deterministic, so runs that agree land well below 1 (pass rule in log.md), runs that drifted apart land near or above it.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .infer import resolve_adapter


def lora(folder: str) -> dict:
    path = Path(folder)
    path = path / "adapter" if (path / "adapter").is_dir() else path
    return {k: v.float() for k, v in load_file(str(path / "adapter_model.safetensors")).items()}


def norm(tensors) -> float:
    return float(torch.sqrt(sum((t.double() ** 2).sum() for t in tensors)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="the starting adapter (folder or auto:NAME)")
    ap.add_argument("--pair", action="append", required=True, help="two run folders, comma-separated")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    init = lora(resolve_adapter(a.init))
    report = []
    for pair in a.pair:
        x, y = pair.split(",")
        wa, wb = lora(x), lora(y)
        if wa.keys() != wb.keys() or wa.keys() != init.keys():
            raise SystemExit(f"{pair}: adapters have different tensors")
        moved = norm(wb[k] - init[k] for k in wb)
        apart = norm(wa[k] - wb[k] for k in wa)
        metrics = {r: json.loads((Path(r) / "metrics.json").read_text()) for r in (x, y)}
        row = {"pair": pair, "moved": moved, "apart": apart, "rel": apart / max(moved, 1e-12),
               "runs": {r: {k: m.get(k) for k in ("steps", "minutes", "tokens", "processes", "skipped", "sessions",
                                                   "peak_gpu_gb")} for r, m in metrics.items()}}
        print(json.dumps(row), flush=True)
        report.append(row)
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
