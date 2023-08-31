import argparse
import glob
import os
import time
import warnings
from typing import List, Optional, Tuple, Union

import torch
from loguru import logger
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from df.checkpoint import load_model as load_model_cp
from df.config import config
from df.io import load_audio, resample, save_audio
from df.logger import init_logger
from df.model import ModelParams
from df.modules import get_device
from df.utils import as_complex, as_real, download_file, get_cache_dir, get_norm_alpha
from df.version import version
from libdf import DF, erb, erb_norm, unit_norm

PRETRAINED_MODELS = ("DeepFilterNet", "DeepFilterNet2", "DeepFilterNet3")
DEFAULT_MODEL = "DeepFilterNet3"


class AudioDataset(Dataset):
    def __init__(self, files: List[str], sr: int) -> None:
        super().__init__()
        self.files = []
        for file in files:
            if not os.path.isfile(file):
                logger.warning(f"File not found: {file}. Skipping...")
            self.files.append(file)
        self.sr = sr

    def __getitem__(self, index) -> Tuple[str, Tensor, int]:
        fn = self.files[index]
        audio, meta = load_audio(fn, self.sr, "cpu")
        return fn, audio, meta.sample_rate

    def __len__(self):
        return len(self.files)


def main(args):
    model, df_state, suffix = init_df(
        args.model_base_dir,
        post_filter=args.pf,
        log_level=args.log_level,
        config_allow_defaults=True,
        epoch=args.epoch,
        mask_only=args.no_df_stage,
    )
    suffix = suffix if args.suffix else None
    if args.output_dir is None:
        args.output_dir = "."
    elif not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    df_sr = ModelParams().sr
    if args.noisy_dir is not None:
        if len(args.noisy_audio_files) > 0:
            logger.error("Only one of `noisy_audio_files` or `noisy_dir` arguments are supported.")
            exit(1)
        input_files = glob.glob(args.noisy_dir + "/*")
    else:
        assert len(args.noisy_audio_files) > 0, "No audio files provided"
        input_files = args.noisy_audio_files
    ds = AudioDataset(input_files, df_sr)
    loader = DataLoader(ds, num_workers=2, pin_memory=True)
    n_samples = len(ds)
    for i, (file, audio, audio_sr) in enumerate(loader):
        file = file[0]
        audio = audio.squeeze(0)
        progress = (i + 1) / n_samples * 100
        t0 = time.time()
        audio = enhance(
            model, df_state, audio, pad=args.compensate_delay, atten_lim_db=args.atten_lim
        )
        t1 = time.time()
        t_audio = audio.shape[-1] / df_sr
        t = t1 - t0
        rtf = t / t_audio
        fn = os.path.basename(file)
        p_str = f"{progress:2.0f}% | " if n_samples > 1 else ""
        logger.info(f"{p_str}Enhanced noisy audio file '{fn}' in {t:.2f}s (RT factor: {rtf:.3f})")
        audio = resample(audio.to("cpu"), df_sr, audio_sr)
        save_audio(file, audio, sr=audio_sr, output_dir=args.output_dir, suffix=suffix, log=False)


def get_model_basedir(m: Optional[str]) -> str:
    if m is None:
        m = DEFAULT_MODEL
    is_default_model = m in PRETRAINED_MODELS
    if is_default_model:
        return maybe_download_model(m)
    return m


def init_df(
    model_base_dir: Optional[str] = None,
    post_filter: bool = False,
    log_level: str = "INFO",
    log_file: Optional[str] = "enhance.log",
    config_allow_defaults: bool = True,
    epoch: Union[str, int, None] = "best",
    default_model: str = DEFAULT_MODEL,
    mask_only: bool = False,
) -> Tuple[nn.Module, DF, str]:
    """Initializes and loads config, model and deep filtering state.

    Args:
        model_base_dir (str): Path to the model directory containing checkpoint and config. If None,
            load the pretrained DeepFilterNet2 model.
        post_filter (bool): Enable post filter for some minor, extra noise reduction.
        log_level (str): Control amount of logging. Defaults to `INFO`.
        log_file (str): Optional log file name. None disables it. Defaults to `enhance.log`.
        config_allow_defaults (bool): Whether to allow initializing new config values with defaults.
        epoch (str): Checkpoint epoch to load. Options are `best`, `latest`, `<int>`, and `none`.
            `none` disables checkpoint loading. Defaults to `best`.

    Returns:
        model (nn.Modules): Intialized model, moved to GPU if available.
        df_state (DF): Deep filtering state for stft/istft/erb
        suffix (str): Suffix based on the model name. This can be used for saving the enhanced
            audio.
    """
    try:
        from icecream import ic, install

        ic.configureOutput(includeContext=True)
        install()
    except ImportError:
        pass
    use_default_model = model_base_dir is None or model_base_dir in PRETRAINED_MODELS
    model_base_dir = get_model_basedir(model_base_dir or default_model)

    if not os.path.isdir(model_base_dir):
        raise NotADirectoryError("Base directory not found at {}".format(model_base_dir))
    log_file = os.path.join(model_base_dir, log_file) if log_file is not None else None
    init_logger(file=log_file, level=log_level, model=model_base_dir)
    if use_default_model:
        logger.info(f"Using {default_model} model at {model_base_dir}")
    config.load(
        os.path.join(model_base_dir, "config.ini"),
        config_must_exist=True,
        allow_defaults=config_allow_defaults,
        allow_reload=True,
    )
    if post_filter:
        config.set("mask_pf", True, bool, ModelParams().section)
        try:
            beta = config.get("pf_beta", float, ModelParams().section)
            beta = f"(beta: {beta})"
        except KeyError:
            beta = ""
        logger.info(f"Running with post-filter {beta}")
    p = ModelParams()
    df_state = DF(
        sr=p.sr,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_bands=p.nb_erb,
        min_nb_erb_freqs=p.min_nb_freqs,
    )
    checkpoint_dir = os.path.join(model_base_dir, "checkpoints")
    load_cp = epoch is not None and not (isinstance(epoch, str) and epoch.lower() == "none")
    if not load_cp:
        checkpoint_dir = None
    mask_only = mask_only or config(
        "mask_only", cast=bool, section="train", default=False, save=False
    )
    model, epoch = load_model_cp(checkpoint_dir, df_state, epoch=epoch, mask_only=mask_only)
    if (epoch is None or epoch == 0) and load_cp:
        logger.error("Could not find a checkpoint")
        exit(1)
    logger.debug(f"Loaded checkpoint from epoch {epoch}")
    model = model.to(get_device())
    # Set suffix to model name
    suffix = os.path.basename(os.path.abspath(model_base_dir))
    if post_filter:
        suffix += "_pf"
    logger.info("Running on device {}".format(get_device()))
    logger.info("Model loaded")
    return model, df_state, suffix


def df_features(audio: Tensor, df: DF, nb_df: int, device=None) -> Tuple[Tensor, Tensor, Tensor]:
    spec = df.analysis(audio.numpy())  # [C, Tf] -> [C, Tf, F]
    a = get_norm_alpha(False)
    erb_fb = df.erb_widths()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        erb_feat = torch.as_tensor(erb_norm(erb(spec, erb_fb), a)).unsqueeze(1)
    spec_feat = as_real(torch.as_tensor(unit_norm(spec[..., :nb_df], a)).unsqueeze(1))
    spec = as_real(torch.as_tensor(spec).unsqueeze(1))
    if device is not None:
        spec = spec.to(device)
        erb_feat = erb_feat.to(device)
        spec_feat = spec_feat.to(device)
    return spec, erb_feat, spec_feat


@torch.no_grad()
def enhance(
    model: nn.Module, df_state: DF, audio: Tensor, pad=True, atten_lim_db: Optional[float] = None
):
    """Enhance a single audio given a preloaded model and DF state.

    Args:
        model (nn.Module): A DeepFilterNet model.
        df_state (DF): DF state for STFT/ISTFT and feature calculation.
        audio (Tensor): Time domain audio of shape [C, T]. Sampling rate needs to match to `model` and `df_state`.
        pad (bool): Pad the audio to compensate for delay due to STFT/ISTFT.
        atten_lim_db (float): An optional noise attenuation limit in dB. E.g. an attenuation limit of
            12 dB only suppresses 12 dB and keeps the remaining noise in the resulting audio.

    Returns:
        enhanced audio (Tensor): If `pad` was `False` of shape [C, T'] where T'<T slightly delayed due to STFT.
            If `pad` was `True` it has the same shape as the input.
    """
    model.eval()
    bs = audio.shape[0]
    if hasattr(model, "reset_h0"):
        model.reset_h0(batch_size=bs, device=get_device())
    orig_len = audio.shape[-1]
    n_fft, hop = 0, 0
    if pad:
        n_fft, hop = df_state.fft_size(), df_state.hop_size()
        # Pad audio to compensate for the delay due to the real-time STFT implementation
        audio = F.pad(audio, (0, n_fft))
    nb_df = getattr(model, "nb_df", getattr(model, "df_bins", ModelParams().nb_df))
    spec, erb_feat, spec_feat = df_features(audio, df_state, nb_df, device=get_device())
    enhanced = model(spec.clone(), erb_feat, spec_feat)[0].cpu()
    enhanced = as_complex(enhanced.squeeze(1))
    if atten_lim_db is not None and abs(atten_lim_db) > 0:
        lim = 10 ** (-abs(atten_lim_db) / 20)
        enhanced = as_complex(spec.squeeze(1).cpu()) * lim + enhanced * (1 - lim)
    audio = torch.as_tensor(df_state.synthesis(enhanced.numpy()))
    if pad:
        # The frame size is equal to p.hop_size. Given a new frame, the STFT loop requires e.g.
        # ceil((n_fft-hop)/hop). I.e. for 50% overlap, then hop=n_fft//2
        # requires 1 additional frame lookahead; 75% requires 3 additional frames lookahead.
        # Thus, the STFT/ISTFT loop introduces an algorithmic delay of n_fft - hop.
        assert n_fft % hop == 0  # This is only tested for 50% and 75% overlap
        d = n_fft - hop
        audio = audio[:, d : orig_len + d]
    return audio


def maybe_download_model(name: str = DEFAULT_MODEL) -> str:
    """Download a DeepFilterNet model.

    Args:
        - name (str): Model name. Currently needs to one of `[DeepFilterNet, DeepFilterNet2]`.

    Returns:
        - base_dir: Return the model base directory as string.
    """
    cache_dir = get_cache_dir()
    if name.endswith(".zip"):
        name = name.removesuffix(".zip")
    model_dir = os.path.join(cache_dir, name)
    if os.path.isfile(os.path.join(model_dir, "config.ini")) or os.path.isdir(
        os.path.join(model_dir, "checkpoints")
    ):
        return model_dir
    os.makedirs(cache_dir, exist_ok=True)
    url = f"https://github.com/Rikorose/DeepFilterNet/raw/main/models/{name}"
    download_file(url + ".zip", cache_dir, extract=True)
    return model_dir


def parse_epoch_type(value: str) -> Union[int, str]:
    try:
        return int(value)
    except ValueError:
        assert value in ("best", "latest")
        return value


class PrintVersion(argparse.Action):
    def __init__(self, option_strings, dest):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            required=False,
            help="Print DeepFilterNet version information",
        )

    def __call__(self, *args):
        print("DeepFilterNet", version)
        exit(0)


def setup_df_argument_parser(
    default_log_level: str = "INFO", parser=None
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-base-dir",
        "-m",
        type=str,
        default=None,
        help="Model directory containing checkpoints and config. "
        "To load a pretrained model, you may just provide the model name, e.g. `DeepFilterNet`. "
        "By default, the pretrained DeepFilterNet2 model is loaded.",
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
        default=default_log_level,
        help="Logger verbosity. Can be one of (debug, info, error, none)",
    )
    parser.add_argument("--debug", "-d", action="store_const", const="DEBUG", dest="log_level")
    parser.add_argument(
        "--epoch",
        "-e",
        default="best",
        type=parse_epoch_type,
        help="Epoch for checkpoint loading. Can be one of ['best', 'latest', <int>].",
    )
    parser.add_argument("-v", "--version", action=PrintVersion)
    return parser


def run():
    parser = setup_df_argument_parser()
    parser.add_argument(
        "--no-delay-compensation",
        dest="compensate_delay",
        action="store_false",
        help="Dont't add some paddig to compensate the delay introduced by the real-time STFT/ISTFT implementation.",
    )
    parser.add_argument(
        "--atten-lim",
        "-a",
        type=int,
        default=None,
        help="Attenuation limit in dB by mixing the enhanced signal with the noisy signal.",
    )
    parser.add_argument(
        "noisy_audio_files",
        type=str,
        nargs="*",
        help="List of noise files to mix with the clean speech file.",
    )
    parser.add_argument(
        "--noisy-dir",
        "-i",
        type=str,
        default=None,
        help="Input directory containing noisy audio files. Use instead of `noisy_audio_files`.",
    )
    parser.add_argument(
        "--no-suffix",
        action="store_false",
        dest="suffix",
        help="Don't add the model suffix to the enhanced audio files",
    )
    parser.add_argument("--no-df-stage", action="store_true")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    run()
