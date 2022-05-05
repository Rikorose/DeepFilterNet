import matplotlib.pyplot as plt
import numpy as np
from icecream import install  # noqa

from df.lr import cosine_scheduler

install()

LR = 1e-3
WARMUP = 3
LR_WARMUP_INIT = 1e-4
LR_MIN = 1e-6
CYCLE_DECAY = 1
CYCLE_LIM = -1
CYCLE_MUL = 1

NUM_EPOCHS = 100
STEPS_PER_EPOCH = 20


if __name__ == "__main__":
    lrs = cosine_scheduler(
        LR,
        LR_MIN,
        NUM_EPOCHS,
        STEPS_PER_EPOCH,
        WARMUP,
        LR_WARMUP_INIT,
        initial_ep_per_cycle=CYCLE_LIM,
        cycle_decay=CYCLE_DECAY,
        cycle_mul=CYCLE_MUL,
    )
    wds = cosine_scheduler(1e-12, 0.05, niter_per_ep=STEPS_PER_EPOCH, epochs=NUM_EPOCHS)
    t = np.linspace(0, NUM_EPOCHS, NUM_EPOCHS * STEPS_PER_EPOCH)
    bs = [8] + [16] + [24] * 3 + [32] * 5 + [64] * 10
    bs += [96] * (NUM_EPOCHS - len(bs))
    bs = np.repeat(bs, STEPS_PER_EPOCH)

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.subplots_adjust(right=0.75)
    twin1 = ax.twinx()
    twin2 = ax.twinx()

    # Offset the right spine of twin2.  The ticks and label have already been
    # placed on the right by twinx above.
    twin2.spines.right.set_position(("axes", 1.12))

    (p1,) = ax.plot(t, lrs, "b-", label="Learning Rate")
    (p2,) = twin1.plot(t, wds, "r-", label="Weight Decay")
    (p3,) = twin2.plot(t, bs, "g-", label="Batch Size")

    ax.set_xlim(0, NUM_EPOCHS)
    ax.set_ylim(0, 0.0011)
    twin1.set_ylim(0, 0.055)
    twin2.set_ylim(0, 128)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Learning Rate")
    twin1.set_ylabel("Weight Decay")
    twin2.set_ylabel("Batch Size")

    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin2.yaxis.label.set_color(p3.get_color())

    ax.grid()
    ax.set_xticks(range(0, NUM_EPOCHS, 5))
    twin2.set_yticks([0, 8, 16, 24, 32, 64, 96, 128])

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis="y", colors=p1.get_color(), **tkw)
    twin1.tick_params(axis="y", colors=p2.get_color(), **tkw)
    twin2.tick_params(axis="y", colors=p3.get_color(), **tkw)
    ax.tick_params(axis="x", **tkw)

    ax.legend(handles=[p1, p2, p3], loc="lower right")

    fig.set_tight_layout(True)

    plt.savefig("out/lrs.pdf")
    # plt.show()
