import math


def cosine_lr_scheduler(
    lr: float,
    n_warmup: int = 0,
    warmup_init_lr: float = -1,
    min_lr: float = 0.0,
    t_mult: float = 1.0,
    lr_period_updates: float = -1,
    lr_shrink: float = 0.1,
    max_update: int = -1,
):
    """Cosine annealing learning rate scheduler with warmup and step decay for LambdaLR.

    Based on fairseq.optim.lr_scheduler.cosine_lr_scheduler.

    Args:
        lr (float): (Maximum) learning rate.
        n_warmup (int): Number of warmup steps with a linear lr increase. Default is 0.
        warmup_init_lr (float): Initial learning rate during warmup phase. Default is `lr`.
        min_lr (float): Minimum learning rate during cosine annealing. Default is 0.
        t_mult (float): Factor to grow the length of each period. Default is 1.
        lr_period_updates (float): Initial number of updates per period.
        lr_shrink (float): Shrink factor for each period. Default 0.1.
        max_update (int): Number of maximum updates (epochs). If specified, will result in 1 period
            over all updates.
    """
    max_lr_base = lr
    min_lr_base = min_lr
    warmup_end_lr = max_lr_base
    warmup_init_lr = min_lr if warmup_init_lr < 0 else warmup_init_lr
    period = lr_period_updates
    if period <= 0:
        assert max_update > 0, "Either lr_period_updates or max_update must be set."
        period = max_update - n_warmup
    if n_warmup > 0:
        step_lr = (warmup_end_lr - warmup_init_lr) / n_warmup
    else:
        step_lr = 1
    lr_shrink_base = lr_shrink

    def step(epoch: int) -> float:
        if epoch < n_warmup:
            return (warmup_init_lr + epoch * step_lr) / max_lr_base
        cur_updates = epoch - n_warmup

        if t_mult != 1:
            i = math.floor(math.log(1 - cur_updates / period * (1 - t_mult), t_mult))
            t_i = t_mult**i * period
            t_cur = cur_updates - (1 - t_mult**i) / (1 - t_mult) * period
        else:
            i = math.floor(cur_updates / period)
            t_i = period
            t_cur = cur_updates - (period * i)

        lr_shrink = lr_shrink_base**i
        min_lr = min_lr_base * lr_shrink
        max_lr = max_lr_base * lr_shrink

        return (
            min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t_cur / t_i))
        ) / max_lr_base

    return step
