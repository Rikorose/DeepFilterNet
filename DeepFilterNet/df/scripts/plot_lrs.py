import matplotlib.pyplot as plt
import numpy as np
from icecream import install  # noqa

from df.lr import cosine_scheduler

install()

LR = 1e-2
WARMUP = 3
LR_WARMUP_INIT = 1e-5
LR_MIN = 0
CYCLE_DECAY = 1
CYCLE_LIM = 1
CYCLE_MUL = 1

NUM_EPOCHS = 50
STEPS_PER_EPOCH = 20


def get_lr_updates():
    lrs = cosine_scheduler(
        LR,
        LR_MIN,
        NUM_EPOCHS,
        STEPS_PER_EPOCH,
        WARMUP,
        LR_WARMUP_INIT,
        initial_ep_per_cycle=5,
        cycle_decay=0.9,
        cycle_mul=1.5,
    )
    return lrs


if __name__ == "__main__":
    lr_data_per_step = get_lr_updates()
    t = np.linspace(0, NUM_EPOCHS, NUM_EPOCHS * STEPS_PER_EPOCH)
    plt.plot(t, lr_data_per_step, label="Update per step")
    plt.grid()
    plt.xticks(range(NUM_EPOCHS))
    plt.legend()
    plt.show()
