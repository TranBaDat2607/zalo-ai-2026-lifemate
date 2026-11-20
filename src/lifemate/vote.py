"""Voting between models at answer time (improve_plan.md step 3, check).

Three members share one base model on one GPU: the soup of two adapters (their mean delta merged into the weights,
exactly the model ``infer --adapter a,b`` runs) and each adapter alone. Adapter k alone = soup + (+1 or -1) x D with
D = (delta_a - delta_b) / 2, applied by forward hooks on the LoRA target layers, so the soup member runs the merged
weights untouched (hooks off) and switching member costs nothing.

At every agent step each member generates its turn from the same conversation; the call two of three propose (after
``infer.tidy``) is the turn that goes on, and with no majority the soup's turn. Only the chosen call reaches the
environment, so one ToolEnv per sample still holds. The members' caches are put back to the state before the turn
(``infer.snapshot_cache``) and the chosen turn is fed to all of them with the next tail.
"""
import json
from pathlib import Path

import torch

from . import infer

MEMBERS = (0, 1, -1)  # soup, adapter a, adapter b: the soup proposes first and wins without a majority


def _lora_factors(path: str) -> tuple:
    """({module name: (A, B)}, scaling) of a saved PEFT LoRA adapter."""
    from safetensors.torch import load_file
    config = json.loads((Path(path) / "adapter_config.json").read_text())
    scaling = config["lora_alpha"] / config["r"]
    if config.get("use_rslora"):
        scaling = config["lora_alpha"] / config["r"] ** 0.5
    weights = load_file(str(Path(path) / "adapter_model.safetensors"))
    out = {}
    for key, value in weights.items():
        if ".lora_A." not in key:
            continue
        name = key.split(".lora_A.")[0]
        name = name[len("base_model.model."):] if name.startswith("base_model.model.") else name
        out[name] = (value, weights[key.replace(".lora_A.", ".lora_B.")])
    return out, scaling


class DeltaSwitch:
    """Forward hooks that add ``sign`` x (delta_a - delta_b) / 2 to the LoRA target layers of the soup model; sign 0
    leaves every output untouched. ``model`` must already hold the soup (``infer.load_adapted`` with both adapters)."""

    def __init__(self, model, paths: list):
        assert len(paths) == 2, "voting needs exactly two adapters (their soup is the third member)"
        (fa, sa), (fb, sb) = (_lora_factors(p) for p in paths)
        assert fa.keys() == fb.keys(), "the two adapters must target the same layers"
        modules = dict(model.named_modules())
        self.sign, self.hooks, self.parameters = 0, [], 0
        for name in fa:
            module = modules.get(name)
            if module is None:  # a wrapper prefix the saved names do not carry
                hits = [m for n, m in modules.items() if n.endswith("." + name)]
                assert len(hits) == 1, f"no single module for {name}"
                module = hits[0]
            w = module.weight
            (a1, b1), (a2, b2) = fa[name], fb[name]
            a = torch.cat([a1, a2], 0).to(w.device, w.dtype)
            b = (torch.cat([b1 * sa, -b2 * sb], 1) / 2).to(w.device, w.dtype)
            self.parameters += a.numel() + b.numel()
            self.hooks.append(module.register_forward_hook(self._hook(a, b)))

    def _hook(self, a, b):
        def hook(module, inputs, output):
            if not self.sign:
                return None
            x = inputs[0]
            return output + self.sign * torch.nn.functional.linear(torch.nn.functional.linear(x.to(a.dtype), a), b).to(output.dtype)
        return hook

    def set(self, sign: int) -> None:
        self.sign = sign

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()


def call_key(text: str, catalog: dict):
    """What a turn does, as run_agent would execute it: the tidied call, or None for a turn without a call."""
    parsed = infer.parse_call(text)
    if parsed is None:
        return None
    name, arguments = parsed
    return infer._call_key(name, infer.tidy(name, arguments, catalog))


def pick(keys: list) -> int:
    """Index of the chosen member: the first one whose call has a majority, else 0 (the soup)."""
    for k, key in enumerate(keys):
        if sum(x == key for x in keys) * 2 > len(keys):
            return k
    return 0


class VotingGenerator:
    """``infer.Generator``'s interface (generate, encode, tokenizer, stop_id, max_new_tokens) over several members.
    The cache run_agent passes around is the list of the members' (cache, tokens fed)."""

    def __init__(self, gen, switch: DeltaSwitch, catalog: dict, members=MEMBERS):
        self.gen, self.switch, self.catalog, self.members = gen, switch, catalog, tuple(members)
        self.tokenizer, self.stop_id, self.max_new_tokens = gen.tokenizer, gen.stop_id, gen.max_new_tokens
        self.lookup, self.lookup_stats = gen.lookup, gen.lookup_stats
        self.votes = {"turns": 0, "all_agree": 0, "soup_outvoted": 0, "no_majority": 0}
        self.trace = []  # per turn: (keys of every member, chosen index); the caller clears it per sample

    @property
    def flips(self):
        return self.gen.flips

    def encode(self, text: str) -> list:
        return self.gen.encode(text)

    def _prefill(self, cache, fed: int, ids: list):
        """Feeds ids[fed:-1], so the turn's own first pass feeds only the last prompt token."""
        torch_ = self.gen.torch
        if fed < len(ids) - 1:
            with torch_.inference_mode():
                out = self.gen.model(input_ids=torch_.tensor([ids[fed:len(ids) - 1]], device=self.gen.model.device),
                                     past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
        return cache, len(ids) - 1

    def generate(self, ids: list, cache, fed: int, limit: int = 0):
        states = list(cache) if cache is not None else [(None, 0)] * len(self.members)
        news, keys, after = [], [], []
        for k, sign in enumerate(self.members):
            self.switch.set(sign)
            c, f = self._prefill(*states[k], ids)
            with torch.inference_mode():  # the caches hold inference tensors: copies into them need this mode
                saved = infer.snapshot_cache(c)
            new, c2, _ = self.gen.generate(ids, c, f, limit)
            assert c2 is c, "the cache must be updated in place to be put back"
            with torch.inference_mode():
                infer.restore_cache(c, saved)
            after.append((c, f))
            news.append(new)
            keys.append(call_key(self.tokenizer.decode(new, skip_special_tokens=False), self.catalog))
        self.switch.set(0)
        self.proposals = news  # every member's turn, for tests and traces
        chosen = pick(keys)
        self.votes["turns"] += 1
        self.votes["all_agree"] += len(set(keys)) == 1
        self.votes["soup_outvoted"] += keys[chosen] != keys[0]
        self.votes["no_majority"] += all(sum(x == key for x in keys) * 2 <= len(keys) for key in keys)
        self.trace.append({"keys": [None if key is None else list(key) for key in keys], "chosen": chosen})
        return news[chosen], after, len(ids)
