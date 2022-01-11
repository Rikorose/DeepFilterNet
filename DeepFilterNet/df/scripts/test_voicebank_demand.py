import glob
import os
import sys
import tempfile

import numpy as np
import pystoi
import torch
from loguru import logger

from df.enhance import df_features, init_df, load_audio, save_audio, setup_df_argument_parser
from df.model import ModelParams
from df.modules import get_device
from df.utils import as_complex, resample

HAS_OCTAVE = True
try:
    import semetrics
except OSError:
    HAS_OCTAVE = False

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, desc="Progress", total=None, fallback_estimate=1000):
        # tqdm not available using fallback
        e = ""
        try:
            L = iterable.__len__()
        except AttributeError:
            L = total or None

        print("{}:  {: >2d}".format(desc, 0), end="")
        for k, i in enumerate(iterable):
            yield i
            if L is not None:
                p = (k + 1) / L
                e = "" if k < (L - 1) else "\n"
            else:
                # Use an exponentially decaying function
                p = 1 - np.exp(-k / fallback_estimate)
            print("\b\b\b\b {: >2d}%".format(int(100 * p)), end=e)
            sys.stdout.flush()
        if L is None:
            print()


__resample_method = "sinc_fast"


def main(args):
    model, df_state, suffix = init_df(
        args.model_base_dir, post_filter=args.pf, log_level=args.log_level, config_allow_defaults=True
    )
    assert os.path.isdir(args.dataset_dir)
    sr = ModelParams().sr
    noisy_dir = os.path.join(args.dataset_dir, "noisy_testset_wav")
    clean_dir = os.path.join(args.dataset_dir, "clean_testset_wav")
    enh_dir = None
    assert os.path.isdir(noisy_dir) and os.path.isdir(clean_dir)
    enh_stoi = []
    noisy_stoi = []
    enh_sisdr = []
    noisy_sisdr = []
    enh_comp = []
    noisy_comp = []
    noisy_files = glob.glob(noisy_dir + "/*wav")
    clean_files = glob.glob(clean_dir + "/*wav")
    for noisyfn, cleanfn in tqdm(zip(noisy_files, clean_files), total=len(noisy_files)):
        noisy, _ = load_audio(noisyfn, sr)
        clean, _ = load_audio(cleanfn, sr)
        enh = enhance(model, df_state, noisy)[0]
        clean = df_state.synthesis(df_state.analysis(clean.numpy()))[0]
        noisy = df_state.synthesis(df_state.analysis(noisy.numpy()))[0]
        enh_stoi.append(stoi(clean, enh, sr))
        noisy_stoi.append(stoi(clean, noisy, sr))
        enh_sisdr.append(si_sdr_speechmetrics(clean, enh))
        noisy_sisdr.append(si_sdr_speechmetrics(clean, noisy))
        noisy_comp.append(composite(clean, noisy, sr))
        enh_comp.append(composite(clean, enh, sr))
        if args.log_level.upper() == "DEBUG":
            print(cleanfn, enh_stoi[-1], enh_comp[-1], enh_sisdr[-1])
        enh = torch.as_tensor(enh).to(torch.float32).view(1, -1)
        if enh_dir is not None:
            save_audio(
                os.path.basename(cleanfn),
                enh,
                sr,
                output_dir=enh_dir,
                suffix=f"{suffix}_{enh_comp[-1][0]:.3f}",
            )
    logger.info(f"noisy stoi: {np.mean(noisy_stoi)}")
    logger.info(f"enhanced stoi: {np.mean(enh_stoi)}")
    noisy_comp = np.stack(noisy_comp)
    enh_comp = np.stack(enh_comp)
    noisy_pesq = np.mean(noisy_comp[:, 0])
    noisy_csig = np.mean(noisy_comp[:, 1])
    noisy_cbak = np.mean(noisy_comp[:, 2])
    noisy_covl = np.mean(noisy_comp[:, 3])
    noisy_ssnr = np.mean(noisy_comp[:, 4])
    enh_pesq = np.mean(enh_comp[:, 0])
    enh_csig = np.mean(enh_comp[:, 1])
    enh_cbak = np.mean(enh_comp[:, 2])
    enh_covl = np.mean(enh_comp[:, 3])
    enh_ssnr = np.mean(enh_comp[:, 4])
    logger.info(f"noisy pesq: {np.mean(noisy_pesq)}")
    logger.info(f"enhanced pesq: {np.mean(enh_pesq)}")
    logger.info(f"noisy csig: {np.mean(noisy_csig)}")
    logger.info(f"enhanced csig: {np.mean(enh_csig)}")
    logger.info(f"noisy cbak: {np.mean(noisy_cbak)}")
    logger.info(f"enhanced cbak: {np.mean(enh_cbak)}")
    logger.info(f"noisy covl: {np.mean(noisy_covl)}")
    logger.info(f"enhanced covl: {np.mean(enh_covl)}")
    logger.info(f"noisy ssnr: {np.mean(noisy_ssnr)}")
    logger.info(f"enhanced ssnr: {np.mean(enh_ssnr)}")
    logger.info(f"noisy sisdr: {np.mean(noisy_sisdr)}")
    logger.info(f"enhanced sisdr: {np.mean(enh_sisdr)}")


def stoi(clean, degraded, sr, extended=False):
    assert len(clean.shape) == 1
    if sr != 10000:
        clean = resample(torch.as_tensor(clean), sr, 10000, method=__resample_method).numpy()
        degraded = resample(torch.as_tensor(degraded), sr, 10000, method=__resample_method).numpy()
        sr = 10000
    return pystoi.stoi(clean, degraded, sr, extended=extended)


def composite(clean: np.ndarray, degraded: np.ndarray, sr: int) -> np.ndarray:
    """Compute pesq, csig, cbak, covl, ssnr"""
    assert HAS_OCTAVE
    assert len(clean.shape) == 1
    if sr != 16000:
        clean = resample(torch.as_tensor(clean), sr, 16000, method=__resample_method).numpy()
        degraded = resample(torch.as_tensor(degraded), sr, 16000, method=__resample_method).numpy()
        sr = 16000
    cf = tempfile.NamedTemporaryFile(suffix=".wav")
    save_audio(cf.name, clean, sr)
    nf = tempfile.NamedTemporaryFile(suffix=".wav")
    save_audio(nf.name, degraded, sr)
    c = semetrics.composite(cf.name, nf.name)
    cf.close()
    nf.close()
    return np.asarray(c)


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
    parser = setup_df_argument_parser()
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Voicebank Demand Test set directory. Must contain 'noisy_testset_wav' and 'clean_testset_wav'",
    )
    args = parser.parse_args()
    main(args)
