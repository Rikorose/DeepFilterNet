import math

import numpy as np


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0,
    warmup_steps: int = -1,
    initial_ep_per_cycle: float = -1,
    cycle_decay: float = 1,
    cycle_mul: float = 1,
):
    """Adopted from official ConvNeXt repo."""
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters_after_warmup = epochs * niter_per_ep - warmup_iters
    if initial_ep_per_cycle == -1:
        initial_ep_per_cycle = iters_after_warmup
        num_cycles = 1
        cycle_lengths = [iters_after_warmup]
    else:
        initial_cycle_iter = int(round(initial_ep_per_cycle * niter_per_ep))
        if cycle_mul == 1:
            num_cycles = int(math.ceil(iters_after_warmup / (initial_ep_per_cycle * niter_per_ep)))
            cycle_lengths = [initial_cycle_iter] * num_cycles
        else:
            num_cycles = 0
            cycle_lengths = []
            i = 0
            while sum(cycle_lengths) < iters_after_warmup:
                num_cycles += 1
                cycle_lengths.append(initial_cycle_iter * cycle_mul**i)
                i += 1
    schedule_cycles = []
    for i in range(num_cycles):
        cycle_base_value = base_value * cycle_decay**i
        iters = np.arange(cycle_lengths[i])
        schedule = final_value + 0.5 * (cycle_base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )
        schedule_cycles.append(schedule)

    schedule = np.concatenate((warmup_schedule, *schedule_cycles))
    schedule = schedule[: epochs * niter_per_ep]

    assert len(schedule) == epochs * niter_per_ep
    return schedule
