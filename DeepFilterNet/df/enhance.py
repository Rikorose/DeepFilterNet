import argparse
import os
import time
import warnings
from typing import Optional, Tuple

import torch
import torchaudio
from loguru import logger
from torch import Tensor, nn

from df import DF, config, erb, erb_norm, unit_norm
from df.logger import init_logger
from df.model import ModelParams
from df.modules import get_device
from df.train import get_norm_alpha, load_model
from df.utils import as_complex, as_real


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_base_dir", type=str, help="Model directory containing checkpoints and config.")
    parser.add_argument(
        "noisy_audio_files",
        type=str,
        nargs="+",
        help="List of noise files to mix with the clean speech file.",
    )
    parser.add_argument("--pf", action="store_true")
    parser.add_argument("--output-dir", "-o", type=str, default=None)
    args = parser.parse_args()
    if not os.path.isdir(args.model_base_dir):
        NotADirectoryError("Base directory not found at {}".format(args.model_base_dir))
    init_logger(file=os.path.join(args.model_base_dir, "enhance.log"))
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
    model, _ = load_model(checkpoint_dir, df_state)
    model = model.to(get_device())
    model_n = os.path.basename(os.path.abspath(args.model_base_dir))
    logger.info("Model loaded")
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    for file in args.noisy_audio_files:
        audio = enhance(model, df_state, file, log=True)
        save_audio(file, audio, p.sr, args.output_dir, model_n, log=True)


def df_features(audio: Tensor, df: DF, device=None) -> Tuple[Tensor, Tensor, Tensor]:
    p = ModelParams()
    spec = df.analysis(audio.numpy())  # [C, Tf] -> [C, Tf, F]
    a = get_norm_alpha(False)
    erb_feat = torch.as_tensor(erb_norm(erb(spec, p.nb_erb), a)).unsqueeze(0)
    spec_feat = as_real(torch.as_tensor(unit_norm(spec[..., : p.nb_df], a)).unsqueeze(0))
    spec = as_real(torch.as_tensor(spec).unsqueeze(0))
    if device is not None:
        spec = spec.to(device)
        erb_feat = erb_feat.to(device)
        spec_feat = spec_feat.to(device)
    return spec, erb_feat, spec_feat


def save_audio(
    file: str,
    audio: Tensor,
    sr: int,
    output_dir: Optional[str] = None,
    suffix: str = "enhanced",
    log: bool = False,
):
    file, ext = os.path.splitext(file)
    outpath = file + f"_{suffix}" + ext
    if output_dir is not None:
        outpath = os.path.join(output_dir, os.path.basename(outpath))
    if log:
        logger.info(f"Saving audio file '{outpath}'")

    torchaudio.save(outpath, torch.as_tensor(audio), sr)


@torch.no_grad()
def enhance(model: nn.Module, df_state: DF, file: str, log: bool = False):
    p = ModelParams()
    model.eval()
    if hasattr(model, "reset_h0"):
        model.reset_h0(batch_size=1, device=get_device())
    audio, sr = torchaudio.load(file)
    t_audio = audio.shape[-1] / sr
    if sr != p.sr:
        warnings.warn(
            f"Audio sampling rate does not match model sampling rate ({sr}, {p.sr}). Resampling..."
        )
        audio = torchaudio.functional.resample(audio, sr, p.sr)
    t0 = time.time()
    spec, erb_feat, spec_feat = df_features(audio, df_state, device=get_device())
    spec = model(spec, erb_feat, spec_feat)[0].cpu()
    t1 = time.time()
    audio = df_state.synthesis(as_complex(spec.squeeze(0)).numpy())
    t = t1 - t0
    rtf = t_audio / t
    if log:
        logger.info(
            "Enhanced noisy audio file '{}' in {:.1f}s (RT factor: {})".format(file, t, rtf)
        )
    return audio


if __name__ == "__main__":
    main()
