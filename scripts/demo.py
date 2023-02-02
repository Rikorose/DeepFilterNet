import struct
import subprocess
import tkinter as tk
from math import ceil
from multiprocessing import Process, Queue
from tkinter import ttk

import numpy as np
import pyaudio as pa
from icecream import ic
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from libdf import DF

ic.includeContext = True

CHUNK = 2048
FORMAT = pa.paInt16
N_SECS = 10
CHANNELS = 1
SR = 48000

N_FFT = CHUNK * 2
HOP = CHUNK
MAX_FREQ = 8000
MAX_BIN = int(ceil(MAX_FREQ / (SR / N_FFT)))

N_FRAMES = 4

plt.rcParams["figure.figsize"] = [7.00, 7.50]
plt.rcParams["figure.autolayout"] = True


root = tk.Tk()
root.wm_title("DeepFilterNet Demo")

ttk.Style().configure("TButton", padding=6, relief="flat", background="#ccc")
ttk.Style().configure("TScale", padding=6, relief="flat", background="#ccc")
ttk.Style().configure("TOptionMenu", padding=6, relief="flat", background="#ccc")

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)
opt_frame = ttk.Frame()
fig_frame = ttk.Frame()
opt_frame.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
fig_frame.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)

opt_frame.columnconfigure(0, weight=1)
opt_frame.columnconfigure(1, weight=2)

### Setup pyaudio ###
p = pa.PyAudio()

stream_non_df = None
stream_df = None
device_id_non_df = 0
device_id_df = 0

samples_rx_non_df = np.zeros(CHUNK * CHANNELS)
samples_q_non_df = Queue()
spec_q_non_df = Queue()
samples_rx_df = np.zeros(CHUNK * CHANNELS)
samples_q_df = Queue()
spec_q_df = Queue()


def spec_worker():
    state_df = DF(SR, N_FFT, HOP, 32, 2)
    state_non_df = DF(SR, N_FFT, HOP, 32, 2)

    while True:
        x = np.asarray(samples_q_non_df.get())
        x = x.astype(np.float32).reshape(CHANNELS, -1).mean(0, keepdims=True) / 32767.0
        X = state_non_df.analysis(x, reset=False).squeeze().T[:MAX_BIN]
        X = np.reshape(np.log10(np.abs(X) + 1e-12) * 20, (-1, 1))
        spec_q_non_df.put(X)

        x = np.asarray(samples_q_df.get())
        x = x.astype(np.float32).reshape(CHANNELS, -1).mean(0, keepdims=True) / 32767.0
        X = state_df.analysis(x, reset=False).squeeze().T[:MAX_BIN]
        spec_q_df.put(np.reshape(np.log10(np.abs(X) + 1e-12) * 20, (-1, 1)))


spec_worker_process = Process(target=spec_worker, daemon=True)
spec_worker_process.start()


def callback_df(in_data, frame_count, time_info, status):
    dataInt = struct.unpack(str(CHUNK * CHANNELS) + "h", in_data)

    # To keep this function fast, just copy out to samples_rx
    samples_q_df.put(dataInt)

    return in_data, pa.paContinue


def callback_non_df(in_data, frame_count, time_info, status):
    dataInt = struct.unpack(str(CHUNK * CHANNELS) + "h", in_data)

    # To keep this function fast, just copy out to samples_rx
    samples_q_non_df.put(dataInt)

    return in_data, pa.paContinue


def init_non_df_pa_stream(input=None):
    ic(input)
    global stream_non_df, device_id_non_df
    if stream_non_df is not None:
        stream_non_df.stop_stream()
        stream_non_df.close()
    device_id_non_df = int((input or tk_non_df.get()).split(":")[0])
    ic(device_id_non_df)
    info = p.get_device_info_by_index(device_id_non_df)
    s = p.get_host_api_info_by_index(info["hostApi"])["name"]
    ic(s)
    stream_non_df = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SR,
        input_device_index=device_id_non_df,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=callback_non_df,
    )
    ic()


def init_df_pa_stream(input=None):
    ic(input)
    global stream_df, device_id_df
    if stream_df is not None:
        stream_df.stop_stream()
        stream_df.close()
    device_id_df = int((input or tk_df.get()).split(":")[0])
    ic(device_id_df)
    stream_df = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SR,
        input_device_index=device_id_df,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=callback_df,
    )
    ic()


# Select Device
devices = {}
for i in range(0, p.get_device_count()):
    info = p.get_device_info_by_index(i)
    s = p.get_host_api_info_by_index(info["hostApi"])["name"]
    devices[info["index"]] = f"{info['index']}: {info['name']} ({s})"

### Configuration GUI ###
button = ttk.Button(master=opt_frame, text="Quit", command=root.quit)
button.grid(column=1, row=0, sticky=tk.E, padx=5, pady=5)

df_i_label = ttk.Label(opt_frame, text="DeepFilter input:")
df_o_label = ttk.Label(opt_frame, text="DeepFilter ouptut:")
df_i_label.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
df_o_label.grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
choices_non_df = [v for v in devices.values() if "deepfilter" not in v.lower()]
choices_df = [v for v in devices.values() if "deepfilter" in v.lower()]
if len(choices_df) == 0:
    raise ValueError("No deep filter sink/source detected.")
tk_non_df = tk.StringVar(opt_frame)
tk_non_df_default = [i for (i, v) in enumerate(choices_non_df) if "built-in" in v.lower()]
tk_non_df.set(choices_non_df[tk_non_df_default[0] if len(tk_non_df_default) > 0 else 0])
tk_df = tk.StringVar(opt_frame)
tk_df.set(choices_df[0])
drop_non_df = ttk.OptionMenu(opt_frame, tk_non_df, *choices_non_df, command=init_non_df_pa_stream)
drop_non_df.grid(column=1, row=1, sticky=tk.W, padx=5, pady=5)
drop_df = tk.OptionMenu(opt_frame, tk_df, *choices_df, command=init_df_pa_stream)
drop_df.grid(column=1, row=2, sticky=tk.W, padx=5, pady=5)

df_attenlim_value = tk.DoubleVar()
df_attenlim_value.set(100)
df_attenlim_label = ttk.Label(opt_frame, text="DeepFilter attenuation limit:")
df_attenlim_label.grid(column=0, row=3, sticky=tk.W, padx=5, pady=5)


def attenlim_callback(input=None):
    lim = get_attenlim(input)
    df_cur_attenlim_label.configure(text=lim)
    lim = lim.rstrip(" [dB]")
    args = [
        "busctl",
        "--user",
        "call",
        "org.deepfilter.DeepFilterLadspa",
        "/org/deepfilter/DeepFilterLadspa",
        "org.deepfilter.DeepFilterLadspa",
        "AttenLim",
        "u",
        lim,
    ]
    try:
        subprocess.run(args, timeout=0.01)
    except subprocess.TimeoutExpired:
        print("DBUS timeout")


def get_attenlim(input=None):
    return str(int(float(input or df_attenlim_value.get()))) + " [dB]"


ic()


df_cur_attenlim_label = ttk.Label(opt_frame, text=get_attenlim())
df_cur_attenlim_label.grid(column=1, row=3, sticky=tk.E, padx=5, pady=5)
df_attenlim_slider = ttk.Scale(
    opt_frame,
    from_=0,
    to=100,
    length=200,
    orient="horizontal",
    variable=df_attenlim_value,
    command=attenlim_callback,
)
df_attenlim_slider.grid(column=1, row=3, sticky=tk.W, padx=5, pady=5)

### Animation GUI ###
fig, (ax_non_df, ax_df) = plt.subplots(2)
n_steps = int(N_SECS / (HOP / SR))
f = (np.arange(0, N_FFT // 2 + 1) * SR // 2 / (N_FFT // 2))[:MAX_BIN]
t = np.arange(0, n_steps) * HOP / SR
spec_non_df = np.full((MAX_BIN, n_steps), -100)
spec_df = np.full((MAX_BIN, n_steps), -100)
im_non_df = ax_non_df.pcolormesh(
    t, f, spec_non_df, rasterized=True, shading="auto", cmap="inferno", vmin=-100, vmax=0
)
im_df = ax_df.pcolormesh(
    t, f, spec_df, rasterized=True, shading="auto", cmap="inferno", vmin=-100, vmax=0
)
ax_non_df.set_title("Input")
ax_df.set_title("DeepFilterNet Output")
ax_non_df.set_ylim(0, MAX_FREQ)
ax_df.set_ylim(0, MAX_FREQ)
ax_non_df.set_xlim = (0, N_SECS)
ax_df.set_xlim = (0, N_SECS)
ax_non_df.set_ylabel("Frequency [Hz]")
ax_df.set_ylabel("Frequency [Hz]")
ax_df.set_xlabel("Time [s]")
device_id_non_df = None
device_id_df = None

canvas = FigureCanvasTkAgg(fig, master=fig_frame)
canvas.draw()

canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


def init():
    global device_id_non_df, device_id_df
    init_non_df_pa_stream()
    init_df_pa_stream()
    return (im_non_df, im_df)


def animate(i):
    global spec_df, spec_non_df, im_df, im_non_df

    spec_non_df = np.roll(spec_non_df, -N_FRAMES, 1)
    X = np.concatenate([np.asarray(spec_q_non_df.get()) for _ in range(N_FRAMES)], 1)
    spec_non_df[:, -N_FRAMES:] = X
    im_non_df.set_array(spec_non_df.ravel())

    spec_df = np.roll(spec_df, -N_FRAMES, 1)
    X = np.concatenate([np.asarray(spec_q_df.get()) for _ in range(N_FRAMES)], 1)
    spec_df[:, -N_FRAMES:] = X
    im_df.set_array(spec_df.ravel())
    return (im_non_df, im_df)


anim = animation.FuncAnimation(
    fig, animate, init_func=init, interval=ic(HOP / SR * 1000), blit=True
)

root.mainloop()
