"""Model Stock (Jang et al., ECCV 2024) ratio for a pair of LoRA copies, from their deltas' geometry, no data needed.

cos θ between the two copies' deltas (per module and over all modules); Model Stock's merged weight is
w0 + t·mean(Δ) with t = N cos / (1 + (N-1) cos). Frobenius products of B·A are computed in rank space:
<B1A1, B2A2> = tr((B1ᵀB2)(A2A1ᵀ)).

    python scripts/wsp/model_stock.py outputs/aux-adapters-dl/m1/adapter outputs/aux-adapters-dl/m2/adapter
"""
import json
import re
import sys
from collections import defaultdict

import torch
from safetensors.torch import load_file


def factors(path):
    sd = load_file(f"{path}/adapter_model.safetensors")
    cfg = json.load(open(f"{path}/adapter_config.json"))
    s = cfg["lora_alpha"] / cfg["r"]
    out = {}
    for k, v in sd.items():
        if ".lora_A." in k:
            mod = k.split(".lora_A.")[0]
            out[mod] = (v.float(), sd[k.replace("lora_A", "lora_B")].float() * s)
    return out


def dot(a, b):
    (A1, B1), (A2, B2) = a, b
    return float(torch.trace((B1.T @ B2) @ (A2 @ A1.T)))


def main():
    paths = sys.argv[1:]
    fs = [factors(p) for p in paths]
    n = len(fs)
    groups = defaultdict(lambda: [0.0, 0.0, 0.0])  # cross, norm1, norm2 (pair 0,1)
    per = {}
    for mod in fs[0]:
        c, n1, n2 = dot(fs[0][mod], fs[1][mod]), dot(fs[0][mod], fs[0][mod]), dot(fs[1][mod], fs[1][mod])
        layer = int(re.search(r"layers\.(\d+)\.", mod).group(1))
        leaf = mod.rsplit(".", 1)[-1]
        kind = "mlp" if leaf in ("gate_proj", "up_proj", "down_proj") else "attn"
        band = "early" if layer < 32 / 3 else ("late" if layer >= 64 / 3 else "mid")
        for key in ("all", kind, band, f"{kind}:{band}", f"L{layer}:{kind}"):
            g = groups[key]
            g[0] += c; g[1] += n1; g[2] += n2
        per[mod] = c / (n1 * n2) ** 0.5
    def ratio(cos, k=2):
        return k * cos / (1 + (k - 1) * cos)
    rows = {}
    for key, (c, n1, n2) in groups.items():
        cos = c / (n1 * n2) ** 0.5
        rows[key] = {"cos": round(cos, 4), "t": round(ratio(cos), 4), "norm_ratio": round((n2 / n1) ** 0.5, 3)}
    for key in ["all", "attn", "mlp", "early", "mid", "late"] + [f"{k}:{b}" for k in ("attn", "mlp") for b in ("early", "mid", "late")]:
        print(f"{key:12} cos {rows[key]['cos']:.4f}  Model Stock t {rows[key]['t']:.4f}  |Δ2|/|Δ1| {rows[key]['norm_ratio']}")
    vals = sorted(per.values())
    print("per-module cos: min %.3f median %.3f max %.3f" % (vals[0], vals[len(vals) // 2], vals[-1]))
    json.dump(rows, open(sys.argv[0].replace(".py", "_out.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
