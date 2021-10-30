import argparse
import os
import signal
from typing import Dict, Optional

import numpy as np
import torch
import torchaudio
from loguru import logger
from torch import Tensor, nn
from torch.optim import Adam, AdamW, Optimizer, RMSprop
from torch.types import Number

from df.checkpoint import read_cp, write_cp
from df.config import config
from df.logger import init_logger, log_metrics, log_model_summary
from df.loss import Istft, Loss, MaskLoss
from df.model import ModelParams, load_model
from df.modules import get_device
from df.utils import (
    as_complex,
    as_real,
    check_finite_module,
    check_manual_seed,
    clip_grad_norm_,
    detach_hidden,
    get_norm_alpha,
    make_np,
)
from libdf import DF
from libdfdata import PytorchDataLoader as DataLoader

should_stop = False
debug = False
state: Optional[DF] = None
istft: Optional[nn.Module]


@logger.catch
def main():
    global should_stop, debug, state

    parser = argparse.ArgumentParser()
    parser.add_argument("data_config_file", type=str, help="Path to a dataset config file.")
    parser.add_argument(
        "data_dir", type=str, help="Path to the dataset directory containing .hdf5 files."
    )
    parser.add_argument(
        "base_dir", type=str, help="Directory to store logs, summaries, checkpoints, etc."
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if not os.path.isfile(args.data_config_file):
        raise FileNotFoundError("Dataset config not found at {}".format(args.data_config_file))
    if not os.path.isdir(args.data_dir):
        NotADirectoryError("Data directory not found at {}".format(args.data_dir))
    os.makedirs(args.base_dir, exist_ok=True)
    summary_dir = os.path.join(args.base_dir, "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    debug = args.debug
    log_level = "DEBUG" if debug else "INFO"
    init_logger(file=os.path.join(args.base_dir, "train.log"), level=log_level)
    config.load(os.path.join(args.base_dir, "config.ini"))
    seed = config("SEED", 42, int, section="train")
    check_manual_seed(seed)
    logger.info("Running on device {}".format(get_device()))

    signal.signal(signal.SIGUSR1, get_sigusr1_handler(args.base_dir))

    p = ModelParams()
    state = DF(
        sr=p.sr,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_bands=p.nb_erb,
        min_nb_erb_freqs=p.min_nb_freqs,
    )
    checkpoint_dir = os.path.join(args.base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    mask_only: bool = config("MASK_ONLY", False, bool, section="train")
    train_df_only: bool = config("DF_ONLY", False, bool, section="train")
    model, epoch = load_model(
        checkpoint_dir,
        state,
        jit=config("JIT", False, cast=bool, section="train"),
        mask_only=mask_only,
        train_df_only=train_df_only,
    )
    opt = load_opt(checkpoint_dir, model, mask_only, train_df_only)
    lrs = torch.optim.lr_scheduler.StepLR(opt, 3, 0.9, last_epoch=epoch - 1)
    try:
        log_model_summary(model, verbose=args.debug)
    except Exception as e:
        logger.warning(f"Failed to print model summary: {e}")

    bs: int = config("BATCH_SIZE", 1, int, section="train")
    bs_eval: int = config("BATCH_SIZE_EVAL", 0, int, section="train")
    bs_eval = bs_eval if bs_eval > 0 else bs
    dataloader = DataLoader(
        ds_dir=args.data_dir,
        ds_config=args.data_config_file,
        sr=p.sr,
        batch_size=bs,
        batch_size_eval=bs_eval,
        num_workers=config("NUM_WORKERS", 4, int, section="train"),
        max_len_s=config("MAX_SAMPLE_LEN_S", 5.0, float, section="train"),
        fft_dataloader=True,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_erb=p.nb_erb,
        nb_spec=p.nb_df,
        norm_alpha=get_norm_alpha(),
        p_atten_lim=config("p_atten_lim", 0.2, float, section="train"),
        p_reverb=config("p_reverb", 0.2, float, section="train"),
        prefetch=10,
        overfit=config("OVERFIT", False, bool, section="train"),
        seed=seed,
        min_nb_erb_freqs=p.min_nb_freqs,
    )

    losses = setup_losses()

    if config("START_EVAL", False, cast=bool, section="train"):
        val_loss = run_epoch(
            model=model,
            epoch=epoch - 1,
            loader=dataloader,
            split="valid",
            opt=opt,
            losses=losses,
            summary_dir=summary_dir,
        )
        metrics = {"loss": val_loss}
        metrics.update(
            {n: torch.mean(torch.stack(vals)).item() for n, vals in losses.get_summaries()}
        )
        log_metrics(f"[{epoch - 1}] [valid]", metrics)
    losses.reset_summaries()
    max_epochs = config("MAX_EPOCHS", 10, int, section="train")
    # Save default values to disk
    config.save(os.path.join(args.base_dir, "config.ini"))
    for epoch in range(epoch, max_epochs):
        train_loss = run_epoch(
            model=model,
            epoch=epoch,
            loader=dataloader,
            split="train",
            opt=opt,
            losses=losses,
            summary_dir=summary_dir,
        )
        metrics = {"loss": train_loss, "lr": lrs.get_last_lr()[0]}
        if debug:
            metrics.update(
                {n: torch.mean(torch.stack(vals)).item() for n, vals in losses.get_summaries()}
            )
        log_metrics(f"[{epoch}] [train]", metrics)
        write_cp(model, "model", checkpoint_dir, epoch + 1)
        write_cp(opt, "opt", checkpoint_dir, epoch + 1)
        losses.reset_summaries()
        val_loss = run_epoch(
            model=model,
            epoch=epoch,
            loader=dataloader,
            split="valid",
            opt=opt,
            losses=losses,
            summary_dir=summary_dir,
        )
        metrics = {"loss": val_loss}
        metrics.update(
            {n: torch.mean(torch.stack(vals)).item() for n, vals in losses.get_summaries()}
        )
        log_metrics(f"[{epoch}] [valid]", metrics)
        losses.reset_summaries()
        if should_stop:
            logger.info("Stopping training")
            exit(0)
        lrs.step()
    test_loss = run_epoch(
        model=model,
        epoch=epoch,
        loader=dataloader,
        split="test",
        opt=opt,
        losses=losses,
        summary_dir=summary_dir,
    )
    metrics: Dict[str, Number] = {"loss": test_loss}
    metrics.update({n: torch.mean(torch.stack(vals)).item() for n, vals in losses.get_summaries()})
    log_metrics(f"[{epoch}] [test]", metrics)
    logger.info("Finished training")


def run_epoch(
    model: nn.Module,
    epoch: int,
    loader: DataLoader,
    split: str,
    opt: Optimizer,
    losses: Loss,
    summary_dir: str,
) -> float:
    global debug

    logger.info("Start {} epoch {}".format(split, epoch))
    log_freq = config("LOG_FREQ", cast=int, default=100, section="train")
    if split != "train" and loader.batch_size_eval is not None:
        bs = loader.batch_size_eval
    else:
        bs = loader.batch_size

    detect_anomaly: bool = config("DETECT_ANOMALY", False, bool, section="train")
    if detect_anomaly:
        logger.info("Running with autograd profiling")
    dev = get_device()
    l_mem = []
    is_train = split == "train"
    summary_fn = summary_write  # or summary_noop
    model.train(mode=is_train)
    losses.store_losses = debug or not is_train
    max_steps = loader.len(split)
    seed = epoch if is_train else 42
    n_nans = 0
    logger.info("Dataloader len: {}".format(loader.len(split)))

    for i, batch in enumerate(loader.iter_epoch(split, seed)):
        opt.zero_grad()
        assert batch.feat_spec is not None
        assert batch.feat_erb is not None
        feat_erb = batch.feat_erb.to(dev, non_blocking=True)
        feat_spec = as_real(batch.feat_spec.to(dev, non_blocking=True))
        noisy = batch.noisy.to(dev, non_blocking=True)
        clean = batch.speech.to(dev, non_blocking=True)
        atten = batch.atten.to(dev, non_blocking=True)
        snrs = batch.snr.to(dev, non_blocking=True)
        with torch.autograd.set_detect_anomaly(detect_anomaly):
            enh, m, lsnr, df_alpha = model.forward(
                spec=as_real(noisy),
                feat_erb=feat_erb,
                feat_spec=feat_spec,
                atten_lim=atten,
            )
            try:
                err = losses.forward(
                    clean,
                    noisy,
                    enh,
                    m,
                    lsnr,
                    df_alpha=df_alpha,
                    max_freq=batch.max_freq,
                    snrs=snrs,
                )
            except Exception as e:
                if "nan" in str(e).lower() or "finite" in str(e).lower():
                    logger.warning("NaN in loss computation: {}. Skipping backward.".format(str(e)))
                    check_finite_module(model)
                    n_nans += 1
                    if n_nans > 10:
                        raise e
                    continue
                raise e
            if is_train:
                try:
                    err.backward()
                    clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                except RuntimeError as e:
                    if "nan" in str(e).lower():
                        cleanup(err, noisy, clean, enh, m, feat_erb, feat_spec, batch)
                        logger.error(str(e))
                        continue
                    else:
                        raise e
                opt.step()
            detach_hidden(model)
        l_mem.append(err.detach())
        if i % log_freq == 0:
            l_mean = torch.stack(l_mem[-100:]).mean().cpu()
            if torch.isnan(l_mean):
                check_finite_module(model)
            l_dict = {"loss": l_mean.item()}
            if debug:
                l_dict.update(
                    {
                        n: torch.mean(torch.stack(vals[-bs:])).item()
                        for n, vals in losses.get_summaries()
                    }
                )
            log_metrics(f"[{epoch}] [{i}/{max_steps}]", l_dict)
            summary_fn(
                clean,
                noisy,
                enh,
                batch.snr,
                lsnr,
                df_alpha,
                summary_dir,
                mask_loss=losses.ml,
            )
    cleanup(err, noisy, clean, enh, m, feat_erb, feat_spec, batch)
    return torch.stack(l_mem).mean().cpu().item()


def setup_losses() -> Loss:
    global state, istft
    assert state is not None

    p = ModelParams()

    istft = Istft(p.sr, p.fft_size, p.hop_size, torch.as_tensor(state.fft_window().copy())).to(
        get_device()
    )
    loss = Loss(state, istft).to(get_device())
    # loss = torch.jit.script(loss)
    return loss


def load_opt(
    cp_dir: str, model: nn.Module, mask_only: bool = False, df_only: bool = False
) -> torch.optim.Optimizer:
    lr = config("LR", 1e-4, float, section="train")
    decay = config("WEIGHT_DECAY", 1e-3, float, section="train")
    optimizer = config("OPTIMIZER", "adamw", str, section="train").lower()
    if mask_only:
        params = []
        for n, p in model.named_parameters():
            if not ("dfrnn" in n or "df_dec" in n):
                params.append(p)
    elif df_only:
        params = (p for n, p in model.named_parameters() if "df" in n.lower())
    else:
        params = model.parameters()
    if optimizer == "adamw":
        opt = AdamW(params, lr=lr, weight_decay=decay)
    elif optimizer == "adam":
        opt = Adam(params, lr=lr, weight_decay=decay)
    elif optimizer == "rmsprop":
        opt = RMSprop(params, lr=lr, weight_decay=decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    try:
        read_cp(opt, "opt", cp_dir)
    except ValueError as e:
        logger.error(f"Could not load optimizer state: {e}")
    for group in opt.param_groups:
        group.setdefault("initial_lr", lr)
    return opt


@torch.no_grad()
def summary_write(
    clean: Tensor,
    noisy: Tensor,
    enh: Tensor,
    snrs: Tensor,
    lsnr: Tensor,
    df_alpha: Tensor,
    summary_dir: str,
    mask_loss: Optional[MaskLoss] = None,
):
    global state
    assert state is not None

    p = ModelParams()
    snr = snrs[0].detach().cpu().item()

    def synthesis(x: Tensor) -> Tensor:
        return torch.as_tensor(state.synthesis(make_np(as_complex(x.detach()))))

    if mask_loss is not None:
        ideal = mask_loss.erb_mask_compr(clean[0], noisy[0], compressed=False)
        ideal = noisy[0] * mask_loss.erb_inv(ideal)
        torchaudio.save(
            os.path.join(summary_dir, f"idealmask_snr{snr}.wav"), synthesis(ideal), p.sr
        )
    torchaudio.save(os.path.join(summary_dir, f"clean_snr{snr}.wav"), synthesis(clean[0]), p.sr)
    torchaudio.save(os.path.join(summary_dir, f"noisy_snr{snr}.wav"), synthesis(noisy[0]), p.sr)
    torchaudio.save(os.path.join(summary_dir, f"enh_snr{snr}.wav"), synthesis(enh[0]), p.sr)
    np.savetxt(os.path.join(summary_dir, f"lsnr_snr{snr}.txt"), lsnr[0].detach().cpu().numpy())
    np.savetxt(
        os.path.join(summary_dir, f"df_alpha_snr{snr}.txt"), df_alpha[0].detach().cpu().numpy()
    )


def summary_noop(*__args, **__kwargs):  # type: ignore
    pass


def get_sigusr1_handler(base_dir):
    def h(*__args):  # type: ignore
        global should_stop
        logger.warning("Received timeout signal. Stopping after current epoch")
        should_stop = True
        continue_file = os.path.join(base_dir, "continue")
        logger.warning(f"Writing {continue_file}")
        open(continue_file, "w").close()

    return h


def cleanup(*args):
    import gc

    for arg in args:
        del arg
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
