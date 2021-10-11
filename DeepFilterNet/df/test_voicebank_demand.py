import argparse
import glob
import os

import numpy as np
import pesq as pypesq
import pystoi
import torch
import torchaudio
from df import DF, config
from df.enhance import df_features, save_audio
from df.logger import init_logger
from df.model import ModelParams
from df.modules import get_device
from df.train import load_model
from df.utils import as_complex
from icecream import ic
from loguru import logger
from pystoi.utils import resample_oct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_base_dir", type=str, help="Directory e.g. for checkpoint loading.")
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Voicebank Demand Test set directory. Must contain 'noisy_testset_wav' and 'clean_testset_wav'",
    )
    parser.add_argument("--disable-df", action="store_true")
    parser.add_argument("--pf", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", "-o", type=str, default=None)
    args = parser.parse_args()
    if not os.path.isdir(args.model_base_dir):
        NotADirectoryError("Base directory not found at {}".format(args.model_base_dir))
    init_logger(file=os.path.join(args.model_base_dir, "test_voicebank_demand.log"))
    config.load(os.path.join(args.model_base_dir, "config.ini"), doraise=True)
    if args.pf:
        config.set(ModelParams().section, "mask_pf", True, bool)
    p = ModelParams()
    df_state = DF(
        sr=p.sr,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_bands=p.nb_erb,
        min_nb_erb_freqs=p.min_nb_freqs,
    )
    checkpoint_dir = os.path.join(args.model_base_dir, "checkpoints")
    model_n = os.path.basename(os.path.abspath(args.model_base_dir))
    model, _ = load_model(checkpoint_dir, df_state, mask_only=args.disable_df)
    logger.info("Model loaded")
    assert os.path.isdir(args.dataset_dir)
    noisy_dir = os.path.join(args.dataset_dir, "noisy_testset_wav")
    clean_dir = os.path.join(args.dataset_dir, "clean_testset_wav")
    enh_dir = os.path.join(args.dataset_dir, "enhanced")
    assert os.path.isdir(noisy_dir) and os.path.isdir(clean_dir)
    os.makedirs(enh_dir, exist_ok=True)
    enh_stoi = []
    noisy_stoi = []
    enh_pesq = []
    noisy_pesq = []
    enh_sdr = []
    noisy_sdr = []
    for noisyfn, cleanfn in zip(glob.iglob(noisy_dir + "/*wav"), glob.iglob(clean_dir + "/*wav")):
        noisy, sr = torchaudio.load(noisyfn)
        clean, sr = torchaudio.load(cleanfn)
        if sr != p.sr:
            noisy = torchaudio.functional.resample(noisy, sr, p.sr)
            clean = torchaudio.functional.resample(clean, sr, p.sr)
        enh = enhance(model, df_state, noisy)[0]
        clean = df_state.synthesis(df_state.analysis(clean.numpy()))[0]
        noisy = df_state.synthesis(df_state.analysis(noisy.numpy()))[0]
        enh_stoi.append(stoi(clean, enh, sr))
        noisy_stoi.append(stoi(clean, noisy, sr))
        enh_pesq.append(pesq(clean, enh, sr))
        noisy_pesq.append(pesq(clean, noisy, sr))
        enh_sdr.append(si_sdr_speechmetrics(clean, enh))
        noisy_sdr.append(si_sdr_speechmetrics(clean, noisy))
        if args.verbose:
            print(cleanfn, enh_stoi[-1], enh_pesq[-1], enh_sdr[-1])
        enh = torch.as_tensor(enh).to(torch.float32).view(1, -1)
        save_audio(
            os.path.basename(cleanfn),
            enh,
            p.sr,
            output_dir=enh_dir,
            suffix=f"{model_n}_{enh_pesq[-1]:.3f}",
            log=args.verbose,
        )
    logger.info(f"noisy stoi: {np.mean(noisy_stoi)}")
    logger.info(f"enhanced stoi: {np.mean(enh_stoi)}")
    logger.info(f"noisy pesq: {np.mean(noisy_pesq)}")
    logger.info(f"enhanced pesq: {np.mean(enh_pesq)}")
    logger.info(f"noisy sdr: {np.mean(noisy_sdr)}")
    logger.info(f"enhanced sdr: {np.mean(enh_sdr)}")


def stoi(clean, degraded, sr, extended=False):
    if sr != 10000:
        clean = resample_oct(clean, 10000, sr)
        degraded = resample_oct(degraded, 10000, sr)
        sr = 10000
    return pystoi.stoi(clean, degraded, sr, extended=extended)


def pesq(clean, degraded, sr, mode="wb"):
    if sr != 16000:
        clean = resample_oct(clean, 16000, sr)
        degraded = resample_oct(degraded, 16000, sr)
        sr = 16000
    return pypesq.pesq(sr, clean, degraded, mode)


@torch.no_grad()
def enhance(model, df_state, audio):
    model.eval()
    if hasattr(model, "reset_h0"):
        model.reset_h0(batch_size=1, device=get_device())
    spec, erb_feat, spec_feat = df_features(audio, df_state, get_device())
    spec = model(spec, erb_feat, spec_feat)[0]
    return df_state.synthesis(as_complex(spec.squeeze(0)).cpu().numpy())


def si_sdr_speechmetrics(reference: np.ndarray, estimate: np.ndarray):
    """This implementation is adopted from https://github.com/aliutkus/speechmetrics/blob/dde303e/speechmetrics/relative/sisdr.py"""
    # as provided by @Jonathan-LeRoux and slightly adapted for the case of just one reference
    # and one estimate.
    # see original code here: https://github.com/sigsep/bsseval/issues/3#issuecomment-494995846
    reference = reference.reshape(-1, 1)
    estimate = estimate.reshape(-1, 1)
    eps = np.finfo(reference.dtype).eps
    Rss = np.dot(reference.T, reference)

    # get the scaling factor for clean sources
    a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)

    e_true = a * reference
    e_res = estimate - e_true

    Sss = (e_true ** 2).sum()
    Snn = (e_res ** 2).sum()

    sisdr = 10 * np.log10((eps + Sss) / (eps + Snn))
    return sisdr


if __name__ == "__main__":
    main()
