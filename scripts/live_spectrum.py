import struct
from queue import Queue

import inquirer
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pyaudio as pa
from icecream import ic

from libdf import DF

CHUNK = 2048
FORMAT = pa.paInt16
N_SECS = 10

p = pa.PyAudio()

# Select Device
devices = {}
for i in range(0, p.get_device_count()):
    info = p.get_device_info_by_index(i)
    s = p.get_host_api_info_by_index(info["hostApi"])["name"]
    devices[info["index"]] = f"{info['index']}: {info['name']} ({s})"

q = [inquirer.List("id", "Select device", choices=devices.values())]
answers = inquirer.prompt(q)

device_id = int(answers["id"].split(":")[0])
device_info = p.get_device_info_by_index(device_id)
sr = int(device_info["defaultSampleRate"])
channels = (
    device_info["maxInputChannels"]
    if (device_info["maxOutputChannels"] < device_info["maxInputChannels"])
    else device_info["maxOutputChannels"]
)
channels = 1

n_fft = CHUNK * 2
hop = CHUNK
df = DF(sr, n_fft, hop, 32, 2)

samples_rx = np.zeros(CHUNK * channels)
samples_q = Queue()

fig, ax = plt.subplots()
n_steps = int(N_SECS / (hop / sr))
f = np.arange(0, n_fft // 2 + 1) * sr // 2 / (n_fft // 2)
t = np.arange(0, n_steps) * hop / sr
spec = np.random.random((n_fft // 2 + 1, n_steps))
im = ax.pcolormesh(t, f, spec, rasterized=True, shading="auto", cmap="inferno", vmin=-100, vmax=0)
ax.set_ylim(0, 8000)
ax.set_xlim = (0, N_SECS)
ax.set_title(devices[device_id])
fig.show()
fig.canvas.draw()


def callback(in_data, frame_count, time_info, status):
    dataInt = struct.unpack(str(CHUNK * channels) + "h", in_data)

    # To keep this function fast, just copy out to samples_rx
    samples_q.put(dataInt)

    return in_data, pa.paContinue


stream = p.open(
    format=FORMAT,
    channels=channels,
    rate=sr,
    input_device_index=device_id,
    input=True,
    frames_per_buffer=CHUNK,
    stream_callback=callback,
)
ic("Opened stream")

stream.start_stream()


def animate(i):
    global im
    global spec
    N = 4

    spec = np.roll(spec, -N, 1)
    x = np.stack([np.asarray(samples_q.get()) for _ in range(N)], 0)
    x = x.astype(np.float32).reshape(N, channels, -1).mean(1) / 32767.0
    X = df.analysis(x, reset=False).squeeze().T
    spec[:, -N:] = np.reshape(np.log10(np.abs(X) + 1e-12) * 20, (-1, N))
    im.set_array(spec.ravel())


ani = animation.FuncAnimation(fig, animate, interval=1)
plt.show()

stream.stop_stream()
stream.close()

p.terminate()
