"""Merge the fine-tuned adapters into plain checkpoints for the offline solution.

    python -m lifemate.export_models --out models --llm-adapter auto:full --asr-model vinai/PhoWhisper-medium --asr-adapter auto:full

The LLM is saved as the text model only, in half precision, so the vision tower of the checkpoint neither ships nor counts.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

from . import models
from .infer import resolve_adapter as find_adapter




def parse_weights(text: str, count: int) -> list:
    """'1/3,1/3,1/3,1/2,1/2,-1/2,-1/2' -> floats, one per adapter (fractions or decimals, negative allowed)."""
    from fractions import Fraction
    weights = [float(Fraction(w.strip())) for w in text.split(",") if w.strip()]
    if len(weights) != count:
        raise SystemExit(f"{len(weights)} adapter weights for {count} adapters")
    return weights


def merge_mean(model, paths: list, weights: list = None):
    """``model`` with the mean of several LoRA deltas merged in (a soup of seeds), or with their weighted sum.

    Averaging the A and B factors separately would not average the deltas B·A. PEFT's "cat" combination stacks the
    factors with the weights folded into B, which gives the weighted sum of the deltas exactly, also for negative weights
    (task arithmetic, e.g. soup + (side-task pair - baseline pair); tests/test_adapter_weights.py).
    """
    from peft import PeftModel
    weights = [1 / len(paths)] * len(paths) if weights is None else list(weights)
    assert len(weights) == len(paths), (len(weights), len(paths))
    print("merging adapter deltas:", [f"{w:+.6f} {p}" for w, p in zip(weights, paths)], "sum", round(sum(weights), 6),
          flush=True)
    names = [f"seed{i}" for i in range(len(paths))]
    # loading and merging weights only: no forward pass, so no autograd graph and no need for inference_mode here
    model = PeftModel.from_pretrained(model, paths[0], adapter_name=names[0])
    for name, path in zip(names[1:], paths[1:]):
        model.load_adapter(path, adapter_name=name)
    model.add_weighted_adapter(names, weights, "mean", combination_type="cat")
    model.set_adapter("mean")
    return model.merge_and_unload(adapter_names=["mean"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models")
    ap.add_argument("--llm", default=models.DEFAULT_LLM)
    ap.add_argument("--llm-adapter", required=True, help="an adapter, or several comma-separated whose deltas are averaged")
    ap.add_argument("--llm-weights", default=None,
                    help="one weight per --llm-adapter, comma-separated (fractions allowed, negative too): the weighted sum "
                         "of the deltas instead of their mean")
    ap.add_argument("--llm-edit", default=None, help="weight-space edit program of the LLM change (lifemate.wsp, JSON)")
    ap.add_argument("--asr-model", default="vinai/PhoWhisper-medium")
    ap.add_argument("--asr-adapter", default=None,
                    help="an adapter, or two comma-separated for two readings (saved as asr and asr2, in this order); "
                         "adapters joined by '+' are averaged into one reading")
    ap.add_argument("--asr-view-adapters", default=None,
                    help="instead of --asr-adapter, the readings view (ASR_IMPROVEMENT_PLAN.md Stage 4): adapters joined by "
                         "'+' in the training order; the base Whisper is saved as asr_base and adapter k as asr_view/k, "
                         "and solve.py --view-adapters merges their mean at load time")
    args = ap.parse_args(argv)
    if bool(args.asr_adapter) == bool(args.asr_view_adapters):
        ap.error("give exactly one of --asr-adapter and --asr-view-adapters")
    out = Path(args.out)

    adapters = [find_adapter(s) for s in args.llm_adapter.split(",")]
    weights = parse_weights(args.llm_weights, len(adapters)) if args.llm_weights else None
    if args.llm_edit and args.llm_edit.strip().lower() != "none":
        from .wsp import load_edited
        tokenizer, llm = models.load_llm(args.llm)
        llm = load_edited(llm, adapters, weights, args.llm_edit)
    elif len(adapters) == 1 and weights is None:
        tokenizer, llm = models.load_llm(args.llm, adapters[0])
    else:
        tokenizer, llm = models.load_llm(args.llm)
        llm = merge_mean(llm, adapters, weights)
    llm_params = models.count_parameters(llm)
    llm.half().save_pretrained(out / "llm", safe_serialization=True)
    tokenizer.save_pretrained(out / "llm")
    del llm
    torch.cuda.empty_cache()

    from transformers import AutoProcessor, WhisperForConditionalGeneration
    from .asr import merge_adapters
    asr_params = 0
    if args.asr_view_adapters:  # unmerged: distinct parameters = the base and the adapters
        from safetensors.torch import load_file
        paths = [find_adapter(s) for s in args.asr_view_adapters.split("+")]
        whisper = WhisperForConditionalGeneration.from_pretrained(args.asr_model, dtype=torch.float16)
        asr_params += sum(p.numel() for p in whisper.parameters())
        whisper.save_pretrained(out / "asr_base", safe_serialization=True)
        AutoProcessor.from_pretrained(paths[0]).save_pretrained(out / "asr_base")
        del whisper
        for k, path in enumerate(paths):
            shutil.copytree(path, out / "asr_view" / str(k))
            asr_params += sum(t.numel() for t in load_file(str(out / "asr_view" / str(k) / "adapter_model.safetensors")).values())
    for i, spec in enumerate(args.asr_adapter.split(",") if args.asr_adapter else []):  # asr2: the second reading
        adapters = [find_adapter(s) for s in spec.split("+")]  # "a+b+c": one reading, the mean of the three deltas
        whisper = WhisperForConditionalGeneration.from_pretrained(args.asr_model, dtype=torch.float16)
        whisper = merge_adapters(whisper, adapters)
        asr_params += sum(p.numel() for p in whisper.parameters())
        folder = out / ("asr" if i == 0 else f"asr{i + 1}")
        whisper.save_pretrained(folder, safe_serialization=True)
        AutoProcessor.from_pretrained(adapters[0]).save_pretrained(folder)
        del whisper

    # Checksums, so that a downloaded copy (the one baked into the Docker image) can be shown to be the one checked here.
    sums = {str(f.relative_to(out)): sha256(f) for f in sorted(out.rglob("*")) if f.is_file()}
    (out / "sha256.json").write_text(json.dumps(sums, indent=1), encoding="utf-8")
    summary = {"llm": llm_params, "asr": asr_params, "total": llm_params["total"] + asr_params}
    (out / "parameters.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")  # written last: the done marker
    print(json.dumps(summary), flush=True)
    if summary["total"] >= 6e9 or llm_params["total"] >= 5e9:
        raise SystemExit("over the parameter limit")


if __name__ == "__main__":
    main()
