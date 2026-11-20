"""Weight-space edits of the fine-tuned model, after Hu et al., "Exploring weight-space programming" (arXiv 2611.09382).

The paper changes a trained model by direct, structured edits of its weights (rescalings, sparse or rank-one changes),
with no gradient step. An edit is found on development data, confirmed on fresh development data and kept only if it
holds there; otherwise the original weights stay. Here the target is the averaged LoRA change of our agent, and an edit
program rescales that change per group of modules:

    {"all": 1.2}                      every LoRA module at 1.2 times its averaged change
    {"mlp:late": 1.3, "attn": 0.9}    factors multiply where selectors overlap

Selectors: "all"; a kind, "attn" (q/k/v/o, linear attention in_proj_qkv/in_proj_z/out_proj), "mlp" (gate/up/down,
Whisper fc1/fc2) or "xattn" (Whisper's decoder cross-attention, encoder_attn); a depth band, "early" / "mid" / "late"
(thirds of the layers); "kind:band"; "L<i>" or "L<i>:kind" for one layer. Encoder-decoder models (the speech model)
also have a part, "enc" or "dec", alone or as "part:kind" / "part:band"; bands count each part's own layers.

Two uses:
- ``program`` in ``load_edited``: infer/export merge the edited change into the base weights, so answering costs nothing.
- ``python -m lifemate.wsp tf ...``: the paper's teacher-forced request. The unmerged model reads gold conversations
  once per program; per sample it records the summed negative log-likelihood of the gold assistant tokens, the share of
  tokens the model would pick itself, and whether it would pick all of them (then greedy decoding repeats gold).
"""
import argparse
import json
import re
import time
from pathlib import Path

_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj")
_MLP = ("gate_proj", "up_proj", "down_proj", "fc1", "fc2")
_KINDS = ("attn", "mlp", "xattn")
_BANDS = ("early", "mid", "late")


def parse_program(spec) -> dict:
    """A program from a dict, a JSON string, or ''/'none' (no edit)."""
    if isinstance(spec, dict):
        prog = dict(spec)
    elif not spec or str(spec).strip().lower() == "none":
        prog = {}
    else:
        prog = json.loads(spec)
    for key, value in prog.items():
        _check_selector(key)
        float(value)
    return {k: float(v) for k, v in prog.items()}


def load_programs(arg: str) -> list:
    """Programs from an inline JSON list or from a JSON file (a long inline list is never tried as a path)."""
    text = arg if arg.lstrip().startswith("[") else Path(arg).read_text()
    return [parse_program(p) for p in json.loads(text)]


def _check_selector(key: str) -> None:
    parts = key.split(":")
    ok = (key == "all" or key in _KINDS + _BANDS + ("enc", "dec")
          or (len(parts) == 2 and parts[0] in _KINDS and parts[1] in _BANDS)
          or (len(parts) == 2 and parts[0] in ("enc", "dec") and parts[1] in _KINDS + _BANDS)
          or re.fullmatch(r"L\d+(:(attn|mlp|xattn))?", key) is not None)
    if not ok:
        raise ValueError(f"unknown selector {key!r}")


def module_group(name: str, n_layers: int):
    """(layer, kind, band) of a LoRA module name, or None if it is no layer module."""
    m = re.search(r"layers\.(\d+)\.", name)
    leaf = name.rsplit(".", 1)[-1]
    if not m or leaf not in _ATTN + _MLP:
        return None
    layer = int(m.group(1))
    kind = "mlp" if leaf in _MLP else ("xattn" if ".encoder_attn." in name else "attn")
    band = "early" if layer < n_layers / 3 else ("late" if layer >= 2 * n_layers / 3 else "mid")
    return layer, kind, band


def module_part(name: str) -> str:
    """'enc' / 'dec' for a layer of an encoder-decoder model (model.encoder.layers.N / model.decoder.layers.N), else ''."""
    m = re.search(r"(encoder|decoder)\.layers\.\d+\.", name)
    return {"encoder": "enc", "decoder": "dec"}[m.group(1)] if m else ""


def factor_for(program: dict, layer: int, kind: str, band: str, part: str = "") -> float:
    f = 1.0
    names = {"all", kind, band, f"{kind}:{band}", f"L{layer}", f"L{layer}:{kind}"}
    if part:
        names |= {part, f"{part}:{kind}", f"{part}:{band}"}
    for key, value in program.items():
        if key in names:
            f *= value
    return f


def lora_modules(model) -> list:
    """[(name, module)] of the LoRA layers of a PEFT model."""
    from peft.tuners.lora import LoraLayer
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoraLayer)]


def apply_program(model, program: dict, adapter: str = "mean") -> dict:
    """Sets each LoRA module's scaling to its original value times the program's factor; returns factor counts."""
    mods = lora_modules(model)
    n_layers = {}  # per part: the encoder and decoder count their own layers
    for name, _ in mods:
        g = module_group(name, 0)
        if g:
            n_layers[module_part(name)] = max(n_layers.get(module_part(name), 0), g[0] + 1)
    seen = {}
    for name, module in mods:
        if not hasattr(module, "_wsp_base"):
            module._wsp_base = dict(module.scaling)
        part = module_part(name)
        g = module_group(name, n_layers.get(part, 0))
        f = factor_for(program, *g, part) if g else 1.0
        module.scaling[adapter] = module._wsp_base[adapter] * f
        seen[round(f, 6)] = seen.get(round(f, 6), 0) + 1
    return seen


def weighted_peft(model, paths: list, weights: list = None):
    """The base model with the (weighted) mean of the adapters' deltas as one unmerged adapter named 'mean'."""
    from peft import PeftModel
    weights = [1 / len(paths)] * len(paths) if weights is None else list(weights)
    names = [f"seed{i}" for i in range(len(paths))]
    model = PeftModel.from_pretrained(model, paths[0], adapter_name=names[0])
    for name, path in zip(names[1:], paths[1:]):
        model.load_adapter(path, adapter_name=name)
    model.add_weighted_adapter(names, weights, "mean", combination_type="cat")
    model.set_adapter("mean")
    for name in names:
        model.delete_adapter(name)
    return model


def load_edited(model, paths: list, weights: list = None, program=None):
    """Merged model with the edited mean change: base + program(sum_k w_k delta_k)."""
    program = parse_program(program)
    peft_model = weighted_peft(model, paths, weights)
    counts = apply_program(peft_model, program)
    print("weight-space edit", json.dumps(program), "module factors", counts, flush=True)
    return peft_model.merge_and_unload()


# --------------------------------------------------------------------------------------------- teacher-forced probe

def gold_scores(base, ids, labels, chunk: int = 256) -> dict:
    """Summed gold NLL, token count, greedy-match count of one encoded conversation (labels -100 outside targets)."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    targets = labels[0, 1:]
    positions = (targets != -100).nonzero()[:, 0]
    with torch.inference_mode(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        hidden = base.model(input_ids=ids, use_cache=False).last_hidden_state[0, positions]
        nll, hit = 0.0, 0
        for start in range(0, len(positions), chunk):
            logits = base.lm_head(hidden[start:start + chunk]).float()
            gold = targets[positions[start:start + chunk]]
            nll += float(torch.nn.functional.cross_entropy(logits, gold, reduction="sum"))
            hit += int((logits.argmax(-1) == gold).sum())
    n = len(positions)
    return {"nll": round(nll, 4), "n": n, "hit": hit, "exact": int(hit == n)}


def tf_main(args) -> None:
    import torch
    from . import data, models, prompts
    from .infer import load_transcripts, resolve_adapter
    from .export_models import parse_weights

    models.seed_everything(args.seed)
    root = data.find_root("train")
    items = [i for i in data.load_split("train", root) if i["id"] in data.id_set(args.ids_file)]
    items = items[args.shard::args.shards] if args.shards > 1 else items
    if args.limit:
        items = items[: args.limit]
    catalog = data.load_catalog("train", root)
    latest = {i["audio"] for i in items}
    texts = None if args.transcripts == "gold" else load_transcripts(args.transcripts, latest)
    paths = [resolve_adapter(a) for a in args.adapter.split(",")]
    weights = parse_weights(args.adapter_weights, len(paths)) if args.adapter_weights else None
    programs = load_programs(args.programs)
    tokenizer, model = models.load_llm(args.llm)
    model = weighted_peft(model, paths, weights)
    model.eval()
    base = model.get_base_model()
    tok = lambda t: tokenizer(t, add_special_tokens=False)["input_ids"]  # noqa: E731
    device = next(base.parameters()).device
    encoded = []
    for item in items:
        t = prompts.gold_transcripts(item) if texts is None else texts
        ids, labels = prompts.encode_conversation(item, t, catalog, tok, args.denoise)
        encoded.append((item["id"], torch.tensor([ids]), torch.tensor([labels])))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"tf.{max(args.shard, 0)}.jsonl"
    done = set()
    if path.exists():  # resume: a deadline may have cut an earlier pass
        done = {(r["program"], r["id"]) for r in map(json.loads, path.read_text().splitlines()) if r}
    start = time.monotonic()
    with open(path, "a", encoding="utf-8") as f:
        for k, prog in enumerate(programs):
            key = json.dumps(prog, sort_keys=True)
            apply_program(model, prog)
            for sid, ids, labels in encoded:
                if (key, sid) in done:
                    continue
                row = gold_scores(base, ids.to(device), labels.to(device))
                f.write(json.dumps({"program": key, "id": sid, **row}) + "\n")
            f.flush()
            print(f"program {k + 1}/{len(programs)} {key} done, {time.monotonic() - start:.0f}s", flush=True)


def main(argv=None) -> None:
    from .models import DEFAULT_LLM
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tf", help="teacher-forced gold scores of several edit programs")
    p.add_argument("--out", required=True)
    p.add_argument("--llm", default=DEFAULT_LLM)
    p.add_argument("--adapter", required=True)
    p.add_argument("--adapter-weights", default=None)
    p.add_argument("--transcripts", default="gold")
    p.add_argument("--ids-file", required=True)
    p.add_argument("--programs", required=True, help="JSON list of programs, inline or a file")
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--shard", type=int, default=-1)
    p.add_argument("--shards", type=int, default=1)
    args, _ = ap.parse_known_args(argv)
    if args.cmd == "tf":
        tf_main(args)


if __name__ == "__main__":
    main()
