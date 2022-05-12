import argparse
import os
import shutil
from typing import List, Tuple

import numpy as np
import numpy.polynomial.polynomial as poly
import requests
import torch
from torch import Tensor

from df.enhance import load_audio
from df.evaluation_utils import as_numpy, dnsmos_api_req

URL_P808 = "https://dnsmos.azurewebsites.net/score"
URL_P835 = "https://dnsmos.azurewebsites.net/v1/dnsmosp835/score"
URL_ONNX = "https://github.com/microsoft/DNS-Challenge/raw/6017eee40aaa39373c15fc897a600a3cfffc7133/DNSMOS/"
# Coefficients for polynomial fitting
COEFS_SIG = np.array([9.651228012789436761e-01, 6.592637550310214145e-01, 7.572372955623894730e-02])
COEFS_BAK = np.array([-3.733460011101781717e00, 2.700114234092929166e00, -1.721332907340922813e-01])
COEFS_OVR = np.array([8.924546794696789354e-01, 6.609981731940616223e-01, 7.600269530243179694e-02])
SR = 16000

ort_providers = [
    (
        "CUDAExecutionProvider",
        {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        },
    ),
    "CPUExecutionProvider",
]

__a_tol = 1e-4
__r_tol = 1e-4


def main(args):
    file: str = args.file
    target_value: List[float] = args.target_value
    audio = load_audio(file, sr=SR, verbose=False)[0].squeeze(0)
    if args.local:
        assert args.method == "p835"
        sig, bak_ovr = download_onnx_models()
        dnsmos = dnsmos_local(audio, sig, bak_ovr)
    else:
        key = os.environ.get("DNS_AUTH_KEY", args.api_key)
        if key is None:
            raise ValueError(
                "No DNSMOS api key found. "
                "Either specify via `--api-key` parameter "
                "or via `DNS_AUTH_KEY` os environmental vairable."
            )
        if args.method == "p808":
            dnsmos = [dnsmos_api_req(URL_P808, key, audio)["mos"]]
        else:
            dnsmos = [
                float(dnsmos_api_req(URL_P835, key, audio)[c])
                for c in ("mos_sig", "mos_bak", "mos_ovr")
            ]
    for d in dnsmos:
        print(d, end=" ")
    print()
    if target_value is not None:
        if len(target_value) > 0:
            assert len(target_value) == len(dnsmos)
        for d, t in zip(dnsmos, target_value):
            if not isclose(d, t):
                print(f"Is not close to target: {target_value}")
                exit(2)
    exit(0)


def audio_logpowspec(audio, nfft=320, hop_length=160) -> np.ndarray:
    try:
        import librosa

        audio = as_numpy(audio)
        powspec = (np.abs(librosa.core.stft(audio, n_fft=nfft, hop_length=hop_length))) ** 2
    except ImportError:
        audio = torch.as_tensor(audio)
        powspec = (
            torch.stft(
                audio,
                n_fft=nfft,
                hop_length=hop_length,
                window=torch.hann_window(nfft),
                return_complex=True,
            )
            .abs()
            .square()
            .numpy()
        )
    logpowspec = np.log10(np.maximum(powspec, 10 ** (-12)))
    return logpowspec.T


def isclose(a, b) -> bool:
    return abs(a - b) <= (__a_tol + __r_tol * abs(b))


def download_onnx_models():
    from appdirs import user_cache_dir

    cache_dir = user_cache_dir("DeepFilterNet")
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    bak_ovr = os.path.join(cache_dir, "bak_ovr.onnx")
    if not os.path.exists(bak_ovr):
        bak_ovr = download_file(URL_ONNX + "bak_ovr.onnx", download_dir=cache_dir)
    sig = os.path.join(cache_dir, "sig.onnx")
    if not os.path.exists(sig):
        sig = download_file(URL_ONNX + "sig.onnx", download_dir=cache_dir)
    return sig, bak_ovr


def dnsmos_local(audio: Tensor, sig: str, bak_ovr: str) -> Tuple[float, float, float]:
    import onnxruntime as ort

    session_sig = ort.InferenceSession(sig, providers=ort_providers)
    session_bak_ovr = ort.InferenceSession(bak_ovr, providers=ort_providers)
    input_length = 9

    num_hops = int(np.floor(len(audio) / SR) - input_length) + 1
    hop_len_samples = SR
    predicted_mos_sig_seg = []
    predicted_mos_bak_seg = []
    predicted_mos_ovr_seg = []

    for idx in range(num_hops):
        audio_seg = audio[int(idx * hop_len_samples) : int((idx + input_length) * hop_len_samples)]
        input_features = np.array(audio_logpowspec(audio=audio_seg)).astype("float32")[
            np.newaxis, :, :
        ]

        onnx_inputs_sig = {inp.name: input_features for inp in session_sig.get_inputs()}
        mos_sig = poly.polyval(session_sig.run(None, onnx_inputs_sig), COEFS_SIG)

        onnx_inputs_bak_ovr = {inp.name: input_features for inp in session_bak_ovr.get_inputs()}
        mos_bak_ovr = session_bak_ovr.run(None, onnx_inputs_bak_ovr)

        mos_bak = poly.polyval(mos_bak_ovr[0][0][1], COEFS_BAK)
        mos_ovr = poly.polyval(mos_bak_ovr[0][0][2], COEFS_OVR)

        predicted_mos_sig_seg.append(mos_sig)
        predicted_mos_bak_seg.append(mos_bak)
        predicted_mos_ovr_seg.append(mos_ovr)

    sig = np.mean(predicted_mos_sig_seg)
    bak = np.mean(predicted_mos_bak_seg)
    ovr = np.mean(predicted_mos_ovr_seg)
    return sig, bak, ovr


def download_file(url, download_dir: str):
    local_filename = url.split("/")[-1]
    print(local_filename)
    local_filename = os.path.join(download_dir, local_filename)
    with requests.get(url, stream=True) as r:
        with open(local_filename, "wb") as f:
            shutil.copyfileobj(r.raw, f)

    return local_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", "-m", default="p808", choices=["p808", "p835"])
    parser.add_argument("--local", "-l", action="store_true")
    parser.add_argument("--api-key", "-k", type=str, default=None)
    parser.add_argument("--target-value", "-t", type=float, nargs="*")
    parser.add_argument("file", type=str, help="Path to audio file for DNSMOS evaluation.")
    args = parser.parse_args()
    main(args)
