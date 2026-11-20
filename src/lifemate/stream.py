"""One pass over new data with replay, on one or more processes, resumable across Kaggle sessions (improvment_plan3.md).

The run is one fixed order of items, built from the seed: every new ("extra") item exactly once, shuffled together with
replayed training conversations ("m", the main conversations) and side examples ("s", e.g. the plan notes), which make
``replay_share`` of the items. Each optimizer step takes the next ``grad_accum`` items of that order; with W processes,
process r trains the items at offsets r, r + W, ... of the step, the gradients are summed over the processes, and every
process then makes the same update. The order, the step and the optimizer state go into one file, so a new session (or
a fresh process) continues exactly where the last one stopped.

Processes talk over gloo on the CPU: one all-reduce of the LoRA gradients (~61 M floats) every ~30 s step costs well
under a second, and avoids NCCL setup on Kaggle's T4s. ``lifemate.ddp`` starts the processes.
"""
import hashlib
import json
import os
import random
from pathlib import Path

import torch

RESTART_EXIT = 75  # a process that saved its state and wants to be started again (lifemate.ddp relaunches all ranks)


def build_order(n_extra: int, n_main: int, n_side: int, replay_share: float, side_share: float, seed: int) -> list:
    """[(kind, index)]: ("x", i) for every extra item once, plus replayed ("m", i) main conversations and ("s", i) side
    examples making ``replay_share`` of all items (``side_share`` of the replay from side examples), shuffled.
    Replay goes through shuffled copies of each pool in turn, so every conversation is replayed about equally often."""
    if not 0 <= replay_share < 1 or not 0 <= side_share <= 1:
        raise ValueError("replay_share must be in [0, 1) and side_share in [0, 1]")
    replay = round(n_extra * replay_share / (1 - replay_share))
    n_s = round(replay * side_share) if n_side else 0
    n_m = replay - n_s if n_main else 0

    def cycle(n: int, k: int, rng: random.Random) -> list:
        out = []
        while len(out) < k:
            perm = list(range(n))
            rng.shuffle(perm)
            out += perm
        return out[:k]

    order = ([("x", i) for i in range(n_extra)] + [("m", i) for i in cycle(n_main, n_m, random.Random(seed + 1))]
             + [("s", i) for i in cycle(n_side, n_s, random.Random(seed + 2))])
    random.Random(seed).shuffle(order)
    return order


def order_hash(order: list, data_hash: str = "") -> str:
    """Fingerprint of the order and the data it indexes: a resumed run must continue the same pass."""
    return hashlib.md5((json.dumps(order) + data_hash).encode()).hexdigest()


def files_hash(paths: list) -> str:
    h = hashlib.md5()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def my_positions(position: int, size: int, rank: int, world: int) -> list:
    """Positions of the order this process trains in the step that starts at ``position`` and has ``size`` items."""
    return [position + j for j in range(size) if j % world == rank]


def init_distributed() -> tuple:
    """(rank, world) from the environment lifemate.ddp sets; (0, 1) for a plain single process."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("gloo", rank=rank, world_size=world)
    return rank, world


def allreduce_grads(params: list, world: int) -> None:
    """Sums the gradients of ``params`` over all processes (missing gradients count as zeros). With the loss of each item
    scaled by 1 / (items in the step), the sum is the mean over the whole step, as on one process."""
    if world == 1:
        return
    import torch.distributed as dist
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch.cat([g.detach().reshape(-1).float().cpu() for g in grads])
    dist.all_reduce(flat)
    offset = 0
    for p, g in zip(params, grads):
        n = g.numel()
        p.grad = flat[offset:offset + n].view_as(p).to(device=p.device, dtype=p.dtype)
        offset += n


def broadcast_params(params: list, world: int) -> None:
    """Every process starts from process 0's weights."""
    if world == 1:
        return
    import torch.distributed as dist
    flat = torch.cat([p.detach().reshape(-1).float().cpu() for p in params])
    dist.broadcast(flat, 0)
    offset = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(flat[offset:offset + n].view_as(p))
            offset += n


def all_sum(values: list, world: int) -> list:
    """Element-wise sum of a few numbers over the processes (loss logging, stop and restart flags)."""
    if world == 1:
        return list(values)
    import torch.distributed as dist
    t = torch.tensor([float(v) for v in values], dtype=torch.float64)
    dist.all_reduce(t)
    return t.tolist()


def barrier(world: int) -> None:
    if world > 1:
        import torch.distributed as dist
        dist.barrier()


def save_state(path: Path, payload: dict) -> None:
    """Writes the state file in one piece: a kill while saving leaves the previous one intact."""
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def find_state(spec: str) -> Path:
    """The folder holding state.pt: a path, or auto:NAME for the attached kernel output whose path contains NAME."""
    if not spec.startswith("auto:"):
        return Path(spec)
    from . import data
    wanted = spec[len("auto:"):]
    found = [d for d in data.walk_input("state.pt") if not wanted or f"/{wanted}/" in str(d) + "/"]
    print("states found:", [str(d) for d in found], flush=True)
    if not found:
        raise SystemExit(f"no state.pt matches {spec}")
    return found[0]
