import argparse
import json
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import numpy.polynomial.polynomial as poly
import torch
from torch import Tensor

from df.io import load_audio
from df.logger import init_logger, log_metrics
from df.utils import download_file, get_cache_dir

try:
    import requests
except ImportError:
    requests = None

URL_P808 = "https://dnsmos.azurewebsites.net/score"
URL_P835 = "https://dnsmos.azurewebsites.net/v1/dnsmosp835/score"
URL_ONNX = "https://github.com/microsoft/DNS-Challenge/raw/6017eee40aaa39373c15fc897a600a3cfffc7133/DNSMOS/"
# Coefficients for polynomial fitting
COEFS_SIG = np.array([9.651228012789436761e-01, 6.592637550310214145e-01, 7.572372955623894730e-02])
COEFS_BAK = np.array([-3.733460011101781717e00, 2.700114234092929166e00, -1.721332907340922813e-01])
COEFS_OVR = np.array([8.924546794696789354e-01, 6.609981731940616223e-01, 7.600269530243179694e-02])
NAMES = ("SIG", "BAK", "OVL")
SR = 16000
INPUT_LENGTH = 9

ORT_PROVIDERS_ALL = [
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
ORT_PROVIDERS_CPU = [
    "CPUExecutionProvider",
]
ORT_SESS = {}

__a_tol = 1e-4
__r_tol = 1e-4


def get_ort_session(onnx: str, providers="gpu"):
    global ORT_SESS

    import onnxruntime as ort

    providers = ORT_PROVIDERS_ALL if providers == "gpu" else ORT_PROVIDERS_CPU
    if onnx not in ORT_SESS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                sess = ort.InferenceSession(onnx, providers=ORT_PROVIDERS_ALL)
        except (ValueError, RuntimeError):
            sess = ort.InferenceSession(onnx, providers=ORT_PROVIDERS_CPU)
        ORT_SESS[onnx] = sess
    return ORT_SESS[onnx]


def main(args):
    file: str = args.file
    target_mos: List[float] = args.target_mos
    verbose = args.debug
    audio = load_audio(file, sr=SR, verbose=verbose)[0].squeeze(0)
    if not args.api:
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
            dnsmos = [dnsmos_api_req(URL_P808, key, audio, verbose=verbose)["mos"]]
        else:
            dnsmos = [
                float(dnsmos_api_req(URL_P835, key, audio, verbose=verbose)[c])
                for c in ("mos_sig", "mos_bak", "mos_ovr")
            ]
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
    cache_dir = get_cache_dir()
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    bak_ovr = os.path.join(cache_dir, "bak_ovr.onnx")
    if not os.path.exists(bak_ovr):
        bak_ovr = download_file(URL_ONNX + "bak_ovr.onnx", download_dir=cache_dir)
    sig = os.path.join(cache_dir, "sig.onnx")
    if not os.path.exists(sig):
        sig = download_file(URL_ONNX + "sig.onnx", download_dir=cache_dir)
    return sig, bak_ovr


def dnsmos_local(
    audio: Tensor, sig: str, bak_ovr: str
) -> Tuple[List[float], List[float], List[float]]:
    session_sig = get_ort_session(sig)
    session_bak_ovr = get_ort_session(bak_ovr)

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

    mod_sig = np.mean(predicted_mos_sig_seg)
    mod_bak = np.mean(predicted_mos_bak_seg)
    mod_ovr = np.mean(predicted_mos_ovr_seg)
    return mod_sig, mod_bak, mod_ovr


def dnsmos_api_req(url: str, key: str, audio: Tensor, verbose=False) -> Dict[str, float]:
    assert requests is not None
    # Set the content type
    headers = {"Content-Type": "application/json"}
    # If authentication is enabled, set the authorization header
    headers["Authorization"] = f"Basic {key}"

    data = {"data": audio.tolist(), "filename": "audio.wav"}
    input_data = json.dumps(data)

    tries = 0
    timeout = 50
    while True:
        try:
            resp = requests.post(url, data=input_data, headers=headers, timeout=timeout)
            score_dict = resp.json()
            if verbose:
                log_metrics("DNSMOS", score_dict, level="DEBUG")
            return score_dict
        except Exception as e:
            if verbose:
                print(e)
            tries += 1
            timeout *= 2
            if tries < 20:
                continue
            raise ValueError(f"Error gettimg mos {e}")


def as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", "-m", default="p835", choices=["p808", "p835"])
    parser.add_argument("--no-local", action="store_true", dest="api")
    parser.add_argument("--api", action="store_true", dest="api")
    parser.add_argument("--api-key", "-k", type=str, default=None)
    parser.add_argument("--debug", "-d", "-v", action="store_true")
    parser.add_argument("--target-mos", "-t", type=float, nargs="*")
    parser.add_argument("file", type=str, help="Path to audio file for DNSMOS evaluation.")
    args = parser.parse_args()
    init_logger(level="DEBUG" if args.debug else "INFO")
    main(args)
