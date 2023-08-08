# Usage:
# python dnsmos_local.py -t c:\temp\DNSChallenge4_Blindset -o DNSCh4_Blind.csv -p
#
import argparse
import concurrent.futures
import glob
import os
from time import sleep
from typing import List, Optional, Union

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from loguru import logger
from tqdm import tqdm

from df.io import load_audio, resample, save_audio
from df.logger import init_logger, log_metrics
from df.scripts.dnsmos import get_ort_session
from df.utils import download_file, get_cache_dir

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01
URL_ONNX = "https://github.com/microsoft/DNS-Challenge/raw/e14b010/DNSMOS/DNSMOS"
NAMES = ["SIG", "BAK", "OVRL", "P808_MOS"]
SLEEP_MS = int(os.environ.get("SLEEP", "0"))
if torch.cuda.is_available():
    TORCH_DEVICE = "cuda"
else:
    TORCH_DEVICE = "cpu"


class ComputeScore:
    def __init__(self, primary_model_path, p808_model_path, cpu: bool = False) -> None:
        self.onnx_sess = get_ort_session(primary_model_path, providers="cpu" if cpu else "gpu")
        self.p808_onnx_sess = get_ort_session(p808_model_path, providers="cpu" if cpu else "gpu")
        n_fft = 320
        self.w = torch.hann_window(n_fft + 1).to(TORCH_DEVICE)
        self.mel = torch.as_tensor(librosa.filters.mel(n_mels=120, sr=16000, n_fft=n_fft)).to(
            TORCH_DEVICE
        )

        # melspec: np.ndarray = np.einsum("...ft,mf->...mt", S, mel_basis, optimize=True)

    def audio_melspec_torch(
        self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True
    ):
        audio = torch.as_tensor(audio).to(TORCH_DEVICE).to(torch.float32)
        powspec = (
            torch.stft(
                audio,
                n_fft=frame_size + 1,
                hop_length=hop_length,
                window=self.w,
                return_complex=True,
            )
            .abs()
            .square()
        )
        mel_spec = torch.einsum("...ft,mf->mt", powspec, self.mel).t().cpu().numpy()
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec

    def audio_melspec(
        self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True
    ):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(
        self,
        fpath_or_tensor: Union[str, torch.Tensor],
        sampling_rate: int,
        is_personalized_MOS: bool = False,
        fname: Optional[str] = None,
    ):
        fs = sampling_rate
        if isinstance(fpath_or_tensor, str):
            fpath = fpath_or_tensor
            logger.debug(f"Processing file: {fpath}")
            aud, input_fs = sf.read(fpath)
            if input_fs != fs:
                audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs)
            else:
                audio = aud
            feat_impl = self.audio_melspec
        else:
            audio = fpath_or_tensor
            feat_impl = self.audio_melspec_torch
            fpath = fname or "unknown"
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)
            ]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
            p808_input_features = np.array(feat_impl(audio=audio_seg[:-160])).astype("float32")[
                np.newaxis, :, :
            ]
            oi = {"input_1": input_features}
            p808_oi = {"input_1": p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS
            )
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        clip_dict = {"filename": fname or fpath, "len_in_sec": actual_audio_len / fs, "sr": fs}
        clip_dict["num_hops"] = num_hops
        clip_dict["OVRL_raw"] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict["SIG_raw"] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict["BAK_raw"] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict["OVRL"] = np.mean(predicted_mos_ovr_seg)
        clip_dict["SIG"] = np.mean(predicted_mos_sig_seg)
        clip_dict["BAK"] = np.mean(predicted_mos_bak_seg)
        clip_dict["P808_MOS"] = np.mean(predicted_p808_mos)
        return clip_dict


def download_onnx_models():
    cache_dir = os.path.join(get_cache_dir(), "DNS5")
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    sig_bak_ovr = os.path.join(cache_dir, "sig_bak_ovr.onnx")
    if not os.path.exists(sig_bak_ovr):
        sig_bak_ovr = download_file(URL_ONNX + "/sig_bak_ovr.onnx", download_dir=cache_dir)
    p808 = os.path.join(cache_dir, "model_v8.onnx")
    if not os.path.exists(p808):
        p808 = download_file(URL_ONNX + "/model_v8.onnx", download_dir=cache_dir)
    return sig_bak_ovr, p808


def eval_sample_dnsmos(
    file: str,
    target_mos: Optional[List[float]] = None,
    log: bool = True,
    use_torch: bool = False,
    compute_score: Optional[ComputeScore] = None,
):
    primary_model_path, p808_model_path = download_onnx_models()
    compute_score = ComputeScore(primary_model_path, p808_model_path)
    if compute_score is None:
        primary_model_path, p808_model_path = download_onnx_models()
        compute_score = ComputeScore(primary_model_path, p808_model_path)
    desired_fs = SAMPLING_RATE
    if use_torch:
        audio = load_audio(file, desired_fs)[0].squeeze(0).numpy()
        scores = compute_score(audio, desired_fs, False, fname=file)
    scores = compute_score(file, desired_fs, False)
    scores = {n: scores[n] for n in NAMES}
    if log:
        logger.info(f"Processing file: {file}")
    if target_mos is not None:
        assert len(target_mos) == 4
        for n, t in zip(NAMES, target_mos):
            if not isclose(scores[n], t):
                diff = (np.asarray(target_mos) - np.fromiter(scores.values(), dtype=float)).tolist()
                log_metrics("Target   ", {n: v for (n, v) in zip(NAMES, target_mos)}, level="ERROR")
                log_metrics("Predicted", {n: v for (n, v) in scores.items()}, level="ERROR")
                log_metrics("Diff     ", {n: v for (n, v) in zip(NAMES, diff)}, level="ERROR")
                print(scores.values())
                exit(2)

    if log:
        log_metrics("Predicted", {n: v for (n, v) in scores.items()})
    return scores


def eval_dir_dnsmos(args):
    primary_model_path, p808_model_path = download_onnx_models()

    compute_score = ComputeScore(primary_model_path, p808_model_path)

    rows = []
    clips = []
    clips = glob.glob(os.path.join(args.testset_dir, "*.wav"))
    is_personalized_eval = False
    desired_fs = SAMPLING_RATE

    if len(clips) == 0:
        print(f"No samples found in dir {args.testset_dir}")
        exit(1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_url = {
            executor.submit(compute_score, clip, desired_fs, is_personalized_eval): clip
            for clip in clips
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_url)):
            clip = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print("%r generated an exception: %s" % (clip, exc))
                raise exc
            else:
                rows.append(data)

    df = pd.DataFrame(rows)
    if args.csv_file:
        csv_file = args.csv_file
        df.to_csv(csv_file)
    print_csv(df)


def load_encoded(buffer: np.ndarray, codec: str):
    import io

    import torchaudio as ta

    # In some rare cases, torch audio failes to fully decode vorbis resulting in a way shorter signal
    wav, _ = ta.load(io.BytesIO(buffer[...].tobytes()), format=codec.lower())
    return wav


def eval_ds(args):
    from tempfile import NamedTemporaryFile

    import h5py
    import torch

    primary_model_path, p808_model_path = download_onnx_models()
    compute_score = ComputeScore(primary_model_path, p808_model_path)

    rows = []
    is_personalized_eval = False
    desired_fs = SAMPLING_RATE

    def compute_score_audio(audio: torch.Tensor, sr: int, fname: str):
        if args.use_torch:
            audio = audio.squeeze(0)
            if audio.dtype == torch.int16:
                audio = audio.to(torch.float32) / 32767.0
            if sr != desired_fs:
                audio = resample(audio, sr, desired_fs)
            return compute_score(audio, desired_fs, is_personalized_eval, fname=fname)
        with NamedTemporaryFile(suffix=".wav") as nf:
            save_audio(nf.name, audio, sr, dtype=torch.float32)
            return compute_score(nf.name, desired_fs, is_personalized_eval, fname=fname)

    for path in args.ds:
        assert os.path.isfile(path)
        group = "speech"
        with h5py.File(path, "r", libver="latest") as f:
            print(f"Evaluating ds {path}")
            assert group in f
            sr = int(f.attrs["sr"])
            codec = f.attrs.get("codec", "pcm")
            for n, sample in f[group].items():  # type: ignore
                print(n)
                if codec == "pcm":
                    audio = torch.from_numpy(sample[...])
                    if audio.dim() == 1:
                        audio.unsqueeze_(0)
                else:
                    audio = load_encoded(sample, codec)
                if SLEEP_MS > 0:
                    sleep(SLEEP_MS / 1000)
                rows.append(compute_score_audio(audio, sr, fname=n))

    df = pd.DataFrame(rows)
    if args.csv_file:
        csv_file = args.csv_file
        df.to_csv(csv_file)
    print_csv(df)


def print_csv(df: Union[pd.DataFrame, List[str]]):
    if isinstance(df, list):
        df = [pd.read_csv(f) for f in df]
        df = pd.concat(df)
    print(df.describe())
    print(
        np.mean(df["SIG"]),
        np.mean(df["BAK"]),
        np.mean(df["OVRL"]),
        np.mean(df["P808_MOS"]),
    )


def isclose(a, b) -> bool:
    __a_tol = 1e-4
    __r_tol = 1e-4
    return abs(a - b) <= (__a_tol + __r_tol * abs(b))


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser_name")
    mn_parser = subparsers.add_parser("mean", aliases=["m"])
    mn_parser.add_argument("csv_file", type=str, nargs="+")
    eval_sample_parser = subparsers.add_parser("eval-sample")
    eval_sample_parser.add_argument("file", type=str)
    eval_sample_parser.add_argument("--target_mos", "-t", type=float, nargs="*")
    eval_dir_parser = subparsers.add_parser("eval-dir", aliases=["e"])
    eval_dir_parser.add_argument(
        "testset_dir", help="Path to the dir containing audio clips in .wav to be evaluated"
    )
    eval_dir_parser.add_argument(
        "-o", "--csv-file", help="If you want the scores in a CSV file provide the full path"
    )
    eval_dir_parser.add_argument("--cpu", help="Only run on CPU", action="store_true")
    eval_dir_parser.add_argument("--num-workers", type=int, default=1)
    eval_ds_parser = subparsers.add_parser("eval-ds")
    eval_ds_parser.add_argument("ds", help="Path to the hdf5 dataset file", nargs="+")
    eval_ds_parser.add_argument("--use-torch", action="store_true")
    eval_ds_parser.add_argument(
        "-o", "--csv-file", help="If you want the scores in a CSV file provide the full path"
    )

    args = parser.parse_args()
    if args.subparser_name is None:
        parser.print_help()
        exit(1)
    if args.subparser_name in ("m", "mean"):
        print_csv(args.csv_file)
    elif args.subparser_name == "eval-sample":
        eval_sample_dnsmos(args.file, args.target_mos)
    elif args.subparser_name == "eval-dir":
        eval_dir_dnsmos(args)
    elif args.subparser_name == "eval-ds":
        eval_ds(args)
    else:
        raise NotImplementedError()
