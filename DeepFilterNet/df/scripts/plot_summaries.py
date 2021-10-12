import argparse
import os
import signal
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torchaudio

from df.visualization import spec_figure

should_stop = False


def main():
    """Poor man's tensorboard summaries"""
    global should_stop
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_dir")
    parser.add_argument("--snr", default=0, type=int)
    parser.add_argument("--watch", "-w", action="store_true")
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--interval", "-i", default=1, type=int)
    parser.add_argument("--max-freq", "-f", default=-1, type=int)
    args = parser.parse_args()
    f, ax = plt.subplots(5, 1, figsize=(15, 9), sharex=True)
    f.set_tight_layout(True)
    if args.watch:
        plt.ion()
    model = os.path.basename(os.path.abspath(os.path.join(args.summary_dir, os.pardir)))
    h = plot(f, ax, args.summary_dir, args.snr)
    ax[4].set_xlabel("Time [s]")
    ax[4].set_ylabel("LSNR [dB]")
    if args.max_freq > 0:
        if args.max_freq > 50:
            args.max_freq /= 1000  # to kHz
        for i in range(4):
            ax[i].set_ylim(0, args.max_freq)
    if args.watch:
        fn = os.path.join(args.summary_dir, f"clean_snr{args.snr}.wav")
        last_mtime = os.stat(fn).st_mtime
        dt = datetime.fromtimestamp(last_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"Found new summary from {dt}")
        ax[0].set_title(f"{model} SNR {args.snr}: {dt}")
        signal.signal(signal.SIGINT, get_sigint_handler())
        while not should_stop and plt.get_fignums():
            if (cur_mtime := os.stat(fn).st_mtime) != last_mtime:
                last_mtime = cur_mtime
                dt = datetime.fromtimestamp(last_mtime).strftime("%Y-%m-%d %H:%M")
                print(f"Found new summary from {dt}")
                plot(f, ax, args.summary_dir, args.snr, update_handle=h)
                ax[0].set_title(f"{model} SNR {args.snr}: {dt}")
            plt.pause(args.interval)
        plt.close(f)
    elif args.save:
        plt.savefig("out/summaries.pdf")
    else:
        plt.show()


def plot(f, ax, summary_dir, snr, update_handle=None) -> int:
    clean, sr = torchaudio.load(os.path.join(summary_dir, f"clean_snr{snr}.wav"))
    spec_figure(clean, sr, from_audio=True, ax=ax[0], figure=f, ylabel="Frequency [kHz]", kHz=True)
    noisy, sr = torchaudio.load(os.path.join(summary_dir, f"noisy_snr{snr}.wav"))
    spec_figure(noisy, sr, from_audio=True, ax=ax[1], figure=f, ylabel="Frequency [kHz]", kHz=True)
    ideal, sr = torchaudio.load(os.path.join(summary_dir, f"idealmask_snr{snr}.wav"))
    spec_figure(ideal, sr, from_audio=True, ax=ax[2], figure=f, ylabel="Frequency [kHz]", kHz=True)
    enh, sr = torchaudio.load(os.path.join(summary_dir, f"enh_snr{snr}.wav"))
    spec_figure(enh, sr, from_audio=True, ax=ax[3], figure=f, ylabel="Frequency [kHz]", kHz=True)
    lsnr = np.loadtxt(os.path.join(summary_dir, f"lsnr_snr{snr}.txt"))
    T = enh.shape[-1] / sr
    t = np.linspace(0, T, lsnr.shape[-1])
    if update_handle is not None:
        h = update_handle
        h[0].set_ydata(lsnr)
    else:
        ax[0].set_xlim(0, T)
        h = []
        h.extend(ax[4].plot(t, lsnr, "b", label="lsnr"))
        ax[4].set_ylim(-15, 30)
    try:
        df_alpha = np.loadtxt(os.path.join(summary_dir, f"df_alpha_snr{snr}.txt"))
        ax_a = ax[4].twinx()
        if update_handle is not None:
            h[1].set_ydata(df_alpha)
        else:
            h.extend(ax_a.plot(t, df_alpha, "r", label="df_alpha"))
            ax_a.set_ylabel("DF alpha")
            ax_a.set_ylim(0, 1)
            lines, labels = ax[4].get_legend_handles_labels()
            lines2, labels2 = ax_a.get_legend_handles_labels()
            ax_a.set_ylabel("DF alpha")
            ax[4].legend(lines + lines2, labels + labels2, loc=0)
    except OSError:
        pass  # file not found
    return h


def get_sigint_handler():
    def h(*__args):  # type: ignore
        global should_stop
        if should_stop:
            raise KeyboardInterrupt
        should_stop = True

    return h


if __name__ == "__main__":
    main()
