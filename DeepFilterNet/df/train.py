import argparse
import os
import signal
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio
from loguru import logger
from torch import Tensor, nn, optim
from torch.autograd.anomaly_mode import set_detect_anomaly
from torch.autograd.grad_mode import set_grad_enabled
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.types import Number

from df.checkpoint import check_patience, load_model, read_cp, write_cp
from df.config import Csv, config
from df.logger import init_logger, log_metrics, log_model_summary
from df.loss import Istft, Loss, MaskLoss
from df.lr import cosine_scheduler
from df.model import ModelParams
from df.modules import get_device
from df.utils import (
    as_complex,
    as_real,
    check_finite_module,
    check_manual_seed,
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
    parser.add_argument("--no-resume", action="store_false", dest="resume")
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
    jit = config("JIT", False, cast=bool, section="train")
    model, epoch = load_model(
        checkpoint_dir if args.resume else None,
        state,
        jit=False,
        mask_only=mask_only,
        train_df_only=train_df_only,
    )
    opt = load_opt(
        checkpoint_dir if args.resume else None,
        model,
        mask_only,
        train_df_only,
    )
    if not args.resume and os.path.isfile(os.path.join(checkpoint_dir, ".patience")):
        os.remove(os.path.join(checkpoint_dir, ".patience"))
    try:
        log_model_summary(model, verbose=args.debug)
    except Exception as e:
        logger.warning(f"Failed to print model summary: {e}")
    if jit:
        # Load as jit after log_model_summary
        model = torch.jit.script(model)

    bs: int = config("BATCH_SIZE", 1, int, section="train")
    bs_eval: int = config("BATCH_SIZE_EVAL", 0, int, section="train")
    bs_eval = bs_eval if bs_eval > 0 else bs
    overfit = config("OVERFIT", False, bool, section="train")
    dataloader = DataLoader(
        ds_dir=args.data_dir,
        ds_config=args.data_config_file,
        sr=p.sr,
        batch_size=bs,
        batch_size_eval=bs_eval,
        num_workers=config("NUM_WORKERS", 4, int, section="train"),
        max_len_s=config("MAX_SAMPLE_LEN_S", 5.0, float, section="train"),
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        nb_erb=p.nb_erb,
        nb_spec=p.nb_df,
        norm_alpha=get_norm_alpha(),
        p_atten_lim=config("p_atten_lim", 0.2, float, section="train"),
        p_reverb=config("p_reverb", 0.2, float, section="train"),
        prefetch=10,
        overfit=overfit,
        seed=seed,
        min_nb_erb_freqs=p.min_nb_freqs,
        log_timings=True,
    )

    max_epochs = config("MAX_EPOCHS", 10, int, section="train")
    assert epoch >= 0
    lrs = load_lrs(len(dataloader))

    # Validation optimization target. Used for early stopping and selecting best checkpoint
    val_criteria = []
    val_criteria_type = config("VALIDATION_CRITERIA", "loss", section="train")  # must be in metrics
    val_criteria_rule = config("VALIDATION_CRITERIA_RULE", "min", section="train")
    val_criteria_rule = val_criteria_rule.replace("less", "min").replace("more", "max")
    patience = config("EARLY_STOPPING_PATIENCE", 5, int, section="train")

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
            lr_scheduler_values=lrs,
        )
        metrics = {"loss": train_loss}
        try:
            metrics["lr"] = opt.param_groups[0]["lr"]
        except AttributeError:
            pass
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
        val_criteria = metrics[val_criteria_type]
        write_cp(
            model, "model", checkpoint_dir, epoch + 1, metric=val_criteria, cmp=val_criteria_rule
        )
        log_metrics(f"[{epoch}] [valid]", metrics)
        check_patience(
            checkpoint_dir, max_patience=patience, new_metric=val_criteria, cmp=val_criteria_rule
        )
        if should_stop:
            logger.info("Stopping training")
            exit(0)
        losses.reset_summaries()
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
    opt: optim.Optimizer,
    losses: Loss,
    summary_dir: str,
    lr_scheduler_values: Optional[np.ndarray] = None,
    wd_scheduler_values: Optional[np.ndarray] = None,
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
    model.train(mode=is_train)
    losses.store_losses = debug or not is_train
    max_steps = loader.len(split) - 1
    seed = epoch if is_train else 42
    n_nans = 0
    logger.info("Dataloader len: {}".format(loader.len(split)))
    start_steps = epoch * loader.len(split)

    for i, batch in enumerate(loader.iter_epoch(split, seed)):
        opt.zero_grad()
        it = start_steps + i  # global training iteration
        if lr_scheduler_values is not None or wd_scheduler_values is not None:
            for param_group in opt.param_groups:
                if lr_scheduler_values is not None:
                    param_group["lr"] = lr_scheduler_values[it] * param_group.get("lr_scale", 1)
                if wd_scheduler_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_scheduler_values[it]
        assert batch.feat_spec is not None
        assert batch.feat_erb is not None
        feat_erb = batch.feat_erb.to(dev, non_blocking=True)
        feat_spec = as_real(batch.feat_spec.to(dev, non_blocking=True))
        noisy = batch.noisy.to(dev, non_blocking=True)
        clean = batch.speech.to(dev, non_blocking=True)
        atten = batch.atten.to(dev, non_blocking=True)
        snrs = batch.snr.to(dev, non_blocking=True)
        with set_detect_anomaly(detect_anomaly and is_train), set_grad_enabled(is_train):
            enh, m, lsnr, other = model.forward(
                spec=as_real(noisy),
                feat_erb=feat_erb,
                feat_spec=feat_spec,
                atten_lim=atten,
            )
            df_alpha, multi_stage_specs = None, []
            if isinstance(other, Tensor):
                df_alpha = other
            elif isinstance(other, (list, tuple)):
                multi_stage_specs = other
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
                    multi_stage_specs=multi_stage_specs,
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
                    e_str = str(e)
                    if "nan" in e_str.lower() or "non-finite" in e_str:
                        check_finite_module(model)
                        cleanup(err, noisy, clean, enh, m, feat_erb, feat_spec, batch)
                        logger.error(e_str)
                        n_nans += 1
                        if n_nans > 10:
                            raise e
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
            if lr_scheduler_values is not None:
                l_dict["lr"] = opt.param_groups[0]["lr"]
            l_dict["t_sample"] = batch.timings[:-1].mean()
            l_dict["t_batch"] = batch.timings[-1].mean()  # last if for whole batch
            if debug:
                l_dict.update(
                    {
                        n: torch.mean(torch.stack(vals[-bs:])).item()
                        for n, vals in losses.get_summaries()
                    }
                )
            log_metrics(f"[{epoch}] [{i}/{max_steps}]", l_dict)
            summary_write(
                clean,
                noisy,
                enh,
                batch.snr,
                lsnr,
                df_alpha,
                summary_dir,
                mask_loss=losses.ml,
                split=split,
            )
    try:
        cleanup(err, noisy, clean, enh, m, feat_erb, feat_spec, batch)
    except UnboundLocalError as err:
        logger.error(str(err))
    return torch.stack(l_mem).mean().cpu().item()


def setup_losses() -> Loss:
    global state, istft
    assert state is not None

    p = ModelParams()

    istft = Istft(p.fft_size, p.hop_size, torch.as_tensor(state.fft_window().copy())).to(
        get_device()
    )
    loss = Loss(state, istft).to(get_device())
    # loss = torch.jit.script(loss)
    return loss


def load_opt(
    cp_dir: Optional[str], model: nn.Module, mask_only: bool = False, df_only: bool = False
) -> optim.Optimizer:
    lr = config("LR", 5e-4, float, section="optim")
    momentum = config("momentum", 0, float, section="optim")  # For sgd, rmsprop
    decay = config("weight_decay", 0.05, float, section="optim")
    optimizer = config("optimizer", "adamw", str, section="optim").lower()
    betas: Tuple[int, int] = config(
        "opt_betas", [0.9, 0.98], Csv(float), section="optim", save=False  # type: ignore
    )
    if mask_only:
        params = []
        for n, p in model.named_parameters():
            if not ("dfrnn" in n or "df_dec" in n):
                params.append(p)
    elif df_only:
        params = (p for n, p in model.named_parameters() if "df" in n.lower())
    else:
        params = model.parameters()
    supported = {
        "adam": lambda p: optim.Adam(p, lr=lr, weight_decay=decay, betas=betas),
        "adamw": lambda p: optim.AdamW(p, lr=lr, weight_decay=decay, betas=betas),
        "sgd": lambda p: optim.SGD(p, lr=lr, momentum=momentum, nesterov=True, weight_decay=decay),
        "rmsprop": lambda p: optim.RMSprop(p, lr=lr, momentum=momentum, weight_decay=decay),
    }
    if optimizer not in supported:
        raise ValueError(
            f"Unsupported optimizer: {optimizer}. Must be one of {list(supported.keys())}"
        )
    opt = supported[optimizer](params)
    logger.debug(f"Training with optimizer {opt}")
    if cp_dir is not None:
        try:
            read_cp(opt, "opt", cp_dir)
        except ValueError as e:
            logger.error(f"Could not load optimizer state: {e}")
    for group in opt.param_groups:
        group.setdefault("initial_lr", lr)
    return opt


def load_lrs(steps_per_epoch: int) -> np.ndarray:
    lr = config.get("lr", float, "optim")
    num_epochs = config.get("max_epochs", int, "train")
    lr_min = config("lr_min", 1e-6, float, section="optim")
    lr_warmup = config("lr_warmup", 1e-4, float, section="optim")
    assert lr_warmup < lr
    warmup_epochs = config("warmup_epochs", 3, int, section="optim")
    lr_cycle_mul = config("lr_cycle_mul", 1.0, float, section="optim")
    lr_cycle_decay = config("lr_cycle_decay", 0.5, float, section="optim")
    lr_cycle_epochs = config("lr_cycle_epochs", -1, int, section="optim")
    lr_values = cosine_scheduler(
        lr,
        lr_min,
        epochs=num_epochs,
        niter_per_ep=steps_per_epoch,
        warmup_epochs=warmup_epochs,
        start_warmup_value=lr_warmup,
        initial_ep_per_cycle=lr_cycle_epochs,
        cycle_decay=lr_cycle_decay,
        cycle_mul=lr_cycle_mul,
    )
    return lr_values


@torch.no_grad()
def summary_write(
    clean: Tensor,
    noisy: Tensor,
    enh: Tensor,
    snrs: Tensor,
    lsnr: Tensor,
    df_alpha: Optional[Tensor],
    summary_dir: str,
    mask_loss: Optional[MaskLoss] = None,
    split="train",
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
            os.path.join(summary_dir, f"{split}_idealmask_snr{snr}.wav"), synthesis(ideal), p.sr
        )
    torchaudio.save(
        os.path.join(summary_dir, f"{split}_clean_snr{snr}.wav"), synthesis(clean[0]), p.sr
    )
    torchaudio.save(
        os.path.join(summary_dir, f"{split}_noisy_snr{snr}.wav"), synthesis(noisy[0]), p.sr
    )
    torchaudio.save(os.path.join(summary_dir, f"{split}_enh_snr{snr}.wav"), synthesis(enh[0]), p.sr)
    np.savetxt(
        os.path.join(summary_dir, f"{split}_lsnr_snr{snr}.txt"),
        lsnr[0].detach().cpu().numpy(),
        fmt="%.3f",
    )
    if df_alpha is not None:
        np.savetxt(
            os.path.join(summary_dir, f"{split}_df_alpha_snr{snr}.txt"),
            df_alpha[0].detach().cpu().numpy(),
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
    from icecream import ic, install

    ic.includeContext = True
    install()
    main()
