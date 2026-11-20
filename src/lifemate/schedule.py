"""Learning-rate schedules that also finish their decay at the deadline when --max-hours ends a run before its last step.

``time_frac`` is the share of the time limit already used. exp9 hit its limit at 81% of the planned steps, with the rate
still at a tenth of its peak, so the last update was never annealed.
"""
import math


def progress(step: int, total: int, time_frac: float = 0.0) -> float:
    """Share of the decay done: by steps, or by time if that is further along."""
    return min(1.0, max(step / max(1, total), time_frac))


def cosine(step: int, warmup: int, total: int, peak: float, time_frac: float = 0.0) -> float:
    """Linear warmup, then cosine down to zero (the LLM)."""
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * progress(step - warmup, total - warmup, time_frac)))


def linear(step: int, warmup: int, total: int, peak: float, time_frac: float = 0.0, floor: float = 0.05) -> float:
    """Linear warmup, then linear decay to ``floor`` times the peak (Whisper)."""
    return peak * min(1.0, (step + 1) / warmup) * max(floor, 1 - progress(step, total, time_frac))


def side_by_epoch(first: int, count: int, epochs: float, seed: int) -> dict:
    """{epoch: indices} that puts each of ``count`` side examples (indices ``first`` .. ``first + count - 1``) into exactly
    one epoch of a run of ``epochs`` epochs, spread evenly in a shuffled order. It has its own random stream, so the
    main conversations' order and encoding draw the same numbers as in a run without side examples."""
    import random
    idx = list(range(first, first + count))
    random.Random(seed).shuffle(idx)
    chunks = max(1, math.ceil(epochs))
    return {e: idx[e::chunks] for e in range(chunks)}


def epoch_order(n_main: int, side: dict, epoch: int) -> list:
    """The indices of one epoch before shuffling: every main conversation, then that epoch's side examples. With no side
    examples this is ``list(range(n_main))``, the order a run without them shuffles."""
    return list(range(n_main)) + list(side.get(epoch, []))


def planned_steps(n_main: int, n_side: int, epochs: float, grad_accum: int) -> int:
    """Optimizer steps of a run: every main conversation ``epochs`` times, every side example once. With no side
    examples this is the old count, ``ceil(n_main * epochs / grad_accum)``."""
    return math.ceil((n_main * epochs + n_side) / grad_accum)
