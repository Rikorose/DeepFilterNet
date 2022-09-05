import argparse
import os
from typing import List, Tuple

import numpy as np
from torch import Tensor

from df.io import load_audio
from df.logger import init_logger, log_metrics
from df.scripts.dnsmos import SR, get_ort_session, isclose
from df.utils import download_file, get_cache_dir

URL_ONNX = "https://github.com/microsoft/DNS-Challenge/raw/82f1b17e7776a43eee395d0f45bae8abb700ad00/DNSMOS/DNSMOS/"
# Coefficients for polynomial fitting
P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
NAMES = ("SIG", "BAK", "OVL")
INPUT_LENGTH = 9.01


def main(args):
    file: str = args.file
    verbose = args.debug
    target_mos: List[float] = args.target_mos
    audio = load_audio(file, sr=SR, verbose=verbose)[0].squeeze(0)
    sig_bak_ovr = download_onnx_model()
    dnsmos = dnsmos_local(audio, sig_bak_ovr)
    log_metrics("Predicted", {n: v for (n, v) in zip(NAMES, dnsmos)})
    if target_mos is not None:
        if len(target_mos) > 0:
            assert len(target_mos) == len(dnsmos)
        log_metrics("Target   ", {n: v for (n, v) in zip(NAMES, target_mos)})
        for d, t in zip(dnsmos, target_mos):
            if not isclose(d, t):
                diff = (np.asarray(target_mos) - np.asarray(dnsmos)).tolist()
                log_metrics("Diff     ", {n: v for (n, v) in zip(NAMES, diff)}, level="ERROR")
                exit(2)
    exit(0)


def download_onnx_model():
    cache_dir = get_cache_dir()
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    name = "sig_bak_ovr.onnx"
    onnx = os.path.join(cache_dir, name)
    if not os.path.exists(onnx):
        onnx = download_file(URL_ONNX + name, download_dir=cache_dir)
    return onnx


def dnsmos_local(audio: Tensor, onnx: str) -> Tuple[float, float, float]:
    assert len(audio) >= SR, f"Audio to short: {audio.shape}"

    session = get_ort_session(onnx)

    if len(audio) < INPUT_LENGTH * SR:
        audio = np.pad(audio, (0, int(INPUT_LENGTH * SR - len(audio))), mode="wrap")
    num_hops = int(np.floor(len(audio) / SR) - INPUT_LENGTH) + 1
    hop_len_samples = SR
    predicted_mos_sig_seg = []
    predicted_mos_bak_seg = []
    predicted_mos_ovr_seg = []
    assert num_hops > 0

    for idx in range(num_hops):
        audio_seg = audio[int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)]
        if len(audio_seg) < INPUT_LENGTH * SR:
            continue
        input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
        oi = {"input_1": input_features}

        mos_sig_raw, mos_bak_raw, mos_ovr_raw = session.run(None, oi)[0][0]

        mos_sig = P_SIG(mos_sig_raw)
        mos_bak = P_BAK(mos_bak_raw)
        mos_ovr = P_BAK(mos_ovr_raw)

        predicted_mos_sig_seg.append(mos_sig)
        predicted_mos_bak_seg.append(mos_bak)
        predicted_mos_ovr_seg.append(mos_ovr)

    mod_sig = np.mean(predicted_mos_sig_seg)
    mod_bak = np.mean(predicted_mos_bak_seg)
    mod_ovr = np.mean(predicted_mos_ovr_seg)
    return mod_sig, mod_bak, mod_ovr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-mos", "-t", type=float, nargs="*")
    parser.add_argument("--debug", "-d", "-v", action="store_true")
    parser.add_argument("file", type=str, help="Path to audio file for DNSMOS evaluation.")
    args = parser.parse_args()
    init_logger(level="DEBUG" if args.debug else "INFO")
    main(args)
