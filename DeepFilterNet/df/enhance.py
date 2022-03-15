import argparse
import os
import time
import warnings
from typing import Optional, Tuple, Union

import torch
import torchaudio as ta
from loguru import logger
from numpy import ndarray
from torch import Tensor, nn
from torch.nn import functional as F
from torchaudio.backend.common import AudioMetaData

import df
from df import config
from df.checkpoint import load_model as load_model_cp
from df.logger import init_logger, warn_once
from df.model import ModelParams
from df.modules import get_device
from df.utils import as_complex, as_real, get_norm_alpha, resample
from libdf import DF, erb, erb_norm, unit_norm


def main(args):
    model, df_state, suffix = init_df(
        args.model_base_dir, post_filter=args.pf, log_level=args.log_level
    )
    if args.output_dir is None:
        args.output_dir = "."
    elif not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    df_sr = ModelParams().sr
    for file in args.noisy_audio_files:
        audio, meta = load_audio(file, df_sr)
        t0 = time.time()
        audio = enhance(model, df_state, audio, pad=args.compensate_delay)
        t1 = time.time()
        t_audio = audio.shape[-1] / df_sr
        t = t1 - t0
        rtf = t_audio / t
        logger.info(f"Enhanced noisy audio file '{file}' in {t:.1f}s (RT factor: {rtf:.1f})")
        audio = resample(audio, df_sr, meta.sample_rate)
        save_audio(
            file, audio, sr=meta.sample_rate, output_dir=args.output_dir, suffix=suffix, log=True
        )


def init_df(
    model_base_dir: Optional[str] = None,
    post_filter: bool = False,
    log_level: str = "INFO",
    config_allow_defaults: bool = False,
) -> Tuple[nn.Module, DF, str]:
    """Initializes and loads config, model and deep filtering state.

    Args:
        model_base_dir (str): Path to the model directory containing checkpoint and config. If None,
            load the default pretrained model.
        post_filter (bool): Enable post filter for some minor, extra noise reduction.
        log (bool): Enable logging. This initializes the logger globaly if not already initilzed.

    Returns:
        model (nn.Modules): Intialized model, moved to GPU if available.
        df_state (DF): Deep filtering state for stft/istft/erb
        suffix (str): Suffix based on the model name. This can be used for saving the enhanced
            audio.
    """
    use_default_model = False
    if model_base_dir is None:
        use_default_model = True
        model_base_dir = os.path.join(
            os.path.dirname(df.__file__), os.pardir, "pretrained_models", "DeepFilterNet"
        )
    if not os.path.isdir(model_base_dir):
        raise NotADirectoryError("Base directory not found at {}".format(model_base_dir))
    init_logger(file=os.path.join(model_base_dir, "enhance.log"), level=log_level)
    if use_default_model:
        logger.info(f"Using default model at {model_base_dir}")
    config.load(
        os.path.join(model_base_dir, "config.ini"),
        config_must_exist=True,
        allow_defaults=config_allow_defaults,
    )
    if post_filter:
        config.set("mask_pf", True, bool, ModelParams().section)
    p = ModelParams()
    df_state = DF(
        sr=p.sr,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_bands=p.nb_erb,
        min_nb_erb_freqs=p.min_nb_freqs,
    )
    checkpoint_dir = os.path.join(model_base_dir, "checkpoints")
    model, epoch = load_model_cp(checkpoint_dir, df_state, best=True)
    if epoch is None:
        logger.error("Could not find a checkpoint")
        exit(1)
    logger.debug(f"Loaded checkpoint from epoch {epoch}")
    model = model.to(get_device())
    # Set suffix to model name
    suffix = os.path.basename(os.path.abspath(model_base_dir))
    if post_filter:
        suffix += "_pf"
    logger.info("Model loaded")
    return model, df_state, suffix


def df_features(audio: Tensor, df: DF, device=None) -> Tuple[Tensor, Tensor, Tensor]:
    p = ModelParams()
    spec = df.analysis(audio.numpy())  # [C, Tf] -> [C, Tf, F]
    a = get_norm_alpha(False)
    erb_fb = df.erb_widths()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        erb_feat = torch.as_tensor(erb_norm(erb(spec, erb_fb), a)).unsqueeze(1)
    spec_feat = as_real(torch.as_tensor(unit_norm(spec[..., : p.nb_df], a)).unsqueeze(1))
    spec = as_real(torch.as_tensor(spec).unsqueeze(1))
    if device is not None:
        spec = spec.to(device)
        erb_feat = erb_feat.to(device)
        spec_feat = spec_feat.to(device)
    return spec, erb_feat, spec_feat


def load_audio(file: str, sr: int, **kwargs) -> Tuple[Tensor, AudioMetaData]:
    """Loads an audio file using torchaudio.

    Args:
        file (str): Path to an audio file.
        sr (int): Target sampling rate. May resample the audio.
        **kwargs: Passed to torchaudio.load(). Depend on the backend.

    Returns:
        audio (Tensor): Audio tensor of shape [C, T], if channels_first=True (default).
        info (AudioMetaData): Meta data or the original audio file. Contains the original sr.
    """
    ikwargs = {}
    if "format" in kwargs:
        ikwargs["format"] = kwargs["format"]
    info = ta.info(file, **ikwargs)
    audio, orig_sr = ta.load(file, **kwargs)
    if orig_sr != sr:
        warn_once(
            f"Audio sampling rate does not match model sampling rate ({orig_sr}, {sr}). "
            "Resampling..."
        )
        audio = resample(audio, orig_sr, sr)
    return audio, info


def save_audio(
    file: str,
    audio: Union[Tensor, ndarray],
    sr: int,
    output_dir: Optional[str] = None,
    suffix: str = None,
    log: bool = False,
):
    outpath = file
    if suffix is not None:
        file, ext = os.path.splitext(file)
        outpath = file + f"_{suffix}" + ext
    if output_dir is not None:
        outpath = os.path.join(output_dir, os.path.basename(outpath))
    if log:
        logger.info(f"Saving audio file '{outpath}'")
    audio = torch.as_tensor(audio)
    if audio.ndim == 1:
        audio.unsqueeze_(0)
    if audio.dtype != torch.int16:
        audio = (audio * (1 << 15)).to(torch.int16)
    ta.save(outpath, audio, sr)


@torch.no_grad()
def enhance(model: nn.Module, df_state: DF, audio: Tensor, pad=False):
    p = ModelParams()
    model.eval()
    bs = audio.shape[0]
    if hasattr(model, "reset_h0"):
        model.reset_h0(batch_size=bs, device=get_device())
    orig_len = audio.shape[-1]
    if pad:
        # Pad audio to compensate for the delay due to the real-time STFT implementation
        audio = F.pad(audio, (0, p.fft_size))
    spec, erb_feat, spec_feat = df_features(audio, df_state, device=get_device())
    spec = model(spec, erb_feat, spec_feat)[0].cpu()
    audio = torch.as_tensor(df_state.synthesis(as_complex(spec.squeeze(1)).numpy()))
    if pad:
        # The frame size is equal to p.hop_size. Given a new frame, the STFT loop requires e.g.
        # ceil((p.fft_size-p.hop_size)/p.hop_size). I.e. for 50% overlap, then p.hop_size=p.fft_size//2
        # requires 1 additional frame lookahead; 75% requires 3 additional frames lookahead.
        # Thus, the STFT/ISTFT loop introduces an algorithmic delay of p.fft_size - p.hop_size.
        assert p.fft_size % p.hop_size == 0  # This is only tested for 50% and 75% overlap
        d = p.fft_size - p.hop_size
        audio = audio[:, d : orig_len + d]
    return audio


def setup_df_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-base-dir",
        "-m",
        type=str,
        default=None,
        help="Model directory containing checkpoints and config. By default, the pretrained model is loaded.",
    )
    parser.add_argument(
        "--pf",
        help="Post-filter that slightly over-attenuates very noisy sections.",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory in which the enhanced audio files will be stored.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Logger verbosity. Can be one of (debug, info, error, none)",
    )
    return parser


if __name__ == "__main__":
    parser = setup_df_argument_parser()
    parser.add_argument(
        "--compensate-delay",
        "-d",
        action="store_true",
        help="Add some paddig to compensate the delay introduced by the real-time STFT/ISTFT implementation.",
    )
    parser.add_argument(
        "noisy_audio_files",
        type=str,
        nargs="+",
        help="List of noise files to mix with the clean speech file.",
    )
    main(parser.parse_args())
