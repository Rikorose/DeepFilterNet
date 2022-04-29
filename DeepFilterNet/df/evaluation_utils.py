import json
import os
from abc import ABC, abstractmethod
from collections import deque
from functools import partial
from multiprocessing.dummy import Pool as DummyPool
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pystoi
import requests
import torch
import torch.multiprocessing as mp
from loguru import logger
from pesq import pesq
from torch import Tensor
from torch.multiprocessing.pool import Pool
from torchaudio.functional import highpass_biquad
from torchaudio.transforms import Resample

from df.enhance import df_features, load_audio, save_audio
from df.sepm import composite as composite_py
from df.utils import as_complex, get_device, get_resample_params, resample
from libdf import DF

RESAMPLE_METHOD = "sinc_fast"


def log_progress(iterable, total: Optional[int] = None, log_freq_percent=25, desc="Progress"):
    disable_logging = log_freq_percent < 0 or log_freq_percent >= 100
    logged = set()
    try:
        L = iterable.__len__()
    except AttributeError:
        assert total is not None
        L = total

    for k, i in enumerate(iterable):
        yield i
        if disable_logging:
            continue
        p = (k + 1) / L
        progress = int(100 * p)
        if progress % log_freq_percent == 0 and progress > 0:
            if progress not in logged:
                logger.info("{}: {: >2d}%".format(desc, progress))
                logged.add(progress)


@torch.no_grad()
def enhance(model, df_state, audio, f_hp_cutoff: Optional[int] = None):
    model.eval()
    if hasattr(model, "reset_h0"):
        model.reset_h0(batch_size=1, device=get_device())
    spec, erb_feat, spec_feat = df_features(audio, df_state, get_device())
    spec = model(spec, erb_feat, spec_feat)[0].squeeze(0)  # [C, T, F, 2]
    audio = df_state.synthesis(as_complex(spec).cpu().numpy())
    if f_hp_cutoff is not None:
        audio = highpass_biquad(
            torch.from_numpy(audio), df_state.sr(), cutoff_freq=f_hp_cutoff
        ).numpy()
    return audio


def evaluation_loop(
    df_state: DF,
    model,
    clean_files: List[str],
    noisy_files: List[str],
    metrics: List[str] = ["stoi", "composite", "sisdr"],  # type: ignore
    save_audio_callback: Optional[Callable[[str, Tensor], None]] = None,
    n_workers: int = 4,
    log_percent: int = 25,
) -> Dict[str, float]:
    sr = df_state.sr()
    metrics_dict = {
        "stoi": partial(StoiMetric, sr=sr),
        "sisdr": SiSDRMetric,
        "composite": partial(CompositeMetric, sr=sr),
        "composite-octave": partial(CompositeMetric, sr=sr, use_octave=True),
        "pesq": partial(PesqMetric, sr=sr),
    }
    if n_workers >= 1:
        pool_fn = mp.Pool
    else:
        pool_fn = DummyPool
    pesqs = []
    with pool_fn(processes=max(1, n_workers)) as pool:
        metrics: List[Metric] = [metrics_dict[m.lower()](pool=pool) for m in metrics]
        for noisyfn, cleanfn in log_progress(
            zip(noisy_files, clean_files), len(noisy_files), log_percent
        ):
            noisy, _ = load_audio(noisyfn, 16000)
            clean, _ = load_audio(cleanfn, 16000)
            pesqs.append(pesq_(clean, noisy, 16000))
            noisy, _ = load_audio(noisyfn, sr)
            clean, _ = load_audio(cleanfn, sr)
            logger.debug(f"Processing {os.path.basename(noisyfn)}, {os.path.basename(cleanfn)}")
            enh = enhance(model, df_state, noisy)[0]
            clean = df_state.synthesis(df_state.analysis(clean.numpy()))[0]
            noisy = df_state.synthesis(df_state.analysis(noisy.numpy()))[0]
            for m in metrics:
                m.add(clean=clean, enhanced=enh, noisy=noisy)
            if save_audio_callback is not None:
                enh = torch.as_tensor(enh).to(torch.float32).view(1, -1)
                save_audio_callback(cleanfn, enh)
        logger.info("Waiting for metrics computation completion. This could take a few minutes.")
        out_dict = {}
        for m in metrics:
            for k, v in m.mean().items():
                out_dict[k] = v
        print(f"pesq mean: {np.mean(pesqs)}")
        return out_dict


def evaluation_loop_dns(
    df_state: DF,
    model,
    noisy_files: List[str],
    metrics: List[str] = ["p808"],  # type: ignore
    save_audio_callback: Optional[Callable[[str, Tensor], None]] = None,
    n_workers: int = 4,
    log_percent: int = 25,
) -> Dict[str, float]:
    sr = df_state.sr()
    metrics_dict = {
        "p808": partial(DnsMosP808ApiMetric, sr=sr),
        "p835": partial(DnsMosP835ApiMetric, sr=sr),
    }
    pesqs = []
    metrics: List[NoisyMetric] = [metrics_dict[m.lower()]() for m in metrics]
    for noisyfn in log_progress(noisy_files, len(noisy_files), log_percent):
        noisy, _ = load_audio(noisyfn, sr)
        logger.debug(f"Processing {os.path.basename(noisyfn)}")
        enh = enhance(model, df_state, noisy)[0]
        noisy = df_state.synthesis(df_state.analysis(noisy.numpy()))[0]
        for m in metrics:
            m.add(enhanced=enh, noisy=noisy)
        if save_audio_callback is not None:
            enh = torch.as_tensor(enh).to(torch.float32).view(1, -1)
            save_audio_callback(noisyfn, enh)
    logger.info("Waiting for metrics computation completion. This could take a few minutes.")
    out_dict = {}
    for m in metrics:
        for k, v in m.mean().items():
            out_dict[k] = v
    print(f"pesq mean: {np.mean(pesqs)}")
    return out_dict


def stoi(clean, degraded, sr, extended=False):
    assert len(clean.shape) == 1
    if sr != 10000:
        clean = resample(torch.as_tensor(clean), sr, 10000, method=RESAMPLE_METHOD).numpy()
        degraded = resample(torch.as_tensor(degraded), sr, 10000, method=RESAMPLE_METHOD).numpy()
        sr = 10000
    stoi = pystoi.stoi(x=clean, y=degraded, fs_sig=sr, extended=extended)
    return stoi


def pesq_(
    clean: Union[np.ndarray, Tensor], degraded: Union[np.ndarray, Tensor], sr: int
) -> np.ndarray:
    if sr != 16000:
        clean = resample(torch.as_tensor(clean), sr, 16000, method=RESAMPLE_METHOD).numpy()
        degraded = resample(torch.as_tensor(degraded), sr, 16000, method=RESAMPLE_METHOD).numpy()
        sr = 16000
    return pesq(sr, as_numpy(clean).squeeze(), as_numpy(degraded).squeeze(), "wb")


def composite(
    clean: Union[np.ndarray, Tensor], degraded: Union[np.ndarray, Tensor], sr: int, use_octave=False
) -> np.ndarray:
    """Compute pesq, csig, cbak, covl, ssnr"""
    assert len(clean.shape) == 1, f"Input must be 1D array, but got input shape {clean.shape}"
    if sr != 16000:
        clean = resample(torch.as_tensor(clean), sr, 16000, method=RESAMPLE_METHOD).numpy()
        degraded = resample(torch.as_tensor(degraded), sr, 16000, method=RESAMPLE_METHOD).numpy()
        sr = 16000
    if use_octave:
        from tempfile import NamedTemporaryFile

        import semetrics

        with NamedTemporaryFile(suffix=".wav") as cf, NamedTemporaryFile(suffix=".wav") as nf:
            # Note: Quantizing to int16 results in slightly modified metrics. Thus, keep f32 prec.
            save_audio(cf.name, clean, sr, dtype=torch.float32)
            save_audio(nf.name, degraded, sr, dtype=torch.float32)
            c = semetrics.composite(cf.name, nf.name)
    else:
        c = composite_py(as_numpy(clean), as_numpy(degraded), sr)
    return np.asarray(c)


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

    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()

    sisdr = 10 * np.log10((eps + Sss) / (eps + Snn))
    return sisdr


class Metric(ABC):
    def __init__(
        self,
        name: Union[str, List[str]],
        source_sr: Optional[int] = None,
        target_sr: Optional[int] = None,
        device="cpu",
    ):
        self.name = name
        self.sr = target_sr
        self.resampler = None
        if source_sr is not None and target_sr is not None and source_sr != target_sr:
            params = get_resample_params(RESAMPLE_METHOD)
            self.resampler = Resample(source_sr, target_sr, **params).to(device)
        self.enh_values: Dict[str, List[float]] = (
            {name: []} if isinstance(name, str) else {n: [] for n in name}
        )
        self.noisy_values: Dict[str, List[float]] = (
            {name: []} if isinstance(name, str) else {n: [] for n in name}
        )

    @abstractmethod
    def compute_metric(self, clean, degraded) -> Union[float, np.ndarray]:
        pass

    def _add_values_enh(self, values_enh: Union[float, np.ndarray]):
        if isinstance(values_enh, float):
            values_enh = np.asarray([values_enh])
        for k, v in zip(self.enh_values.keys(), values_enh):
            self.enh_values[k].append(v)

    def _add_values_noisy(self, values_noisy: Union[float, np.ndarray]):
        if isinstance(values_noisy, float):
            values_noisy = np.asarray([values_noisy])
        for k, v in zip(self.noisy_values.keys(), values_noisy):
            self.noisy_values[k].append(v)

    def maybe_resample(self, x) -> Tensor:
        if self.resampler is not None:
            x = self.resampler.forward(torch.as_tensor(x))
        return x

    def add(self, clean, enhanced, noisy):
        clean = self.maybe_resample(clean).squeeze(0)
        enhanced = self.maybe_resample(enhanced).squeeze(0)
        values_enh = self.compute_metric(clean=clean, degraded=enhanced)
        self._add_values_enh(values_enh)
        if noisy is not None:
            noisy = self.maybe_resample(noisy).squeeze(0)
            values_noisy = self.compute_metric(clean=clean, degraded=enhanced)
            self._add_values_noisy(values_noisy)

    def mean(self) -> Dict[str, float]:
        out = {}
        for k in self.enh_values.keys():
            if k in self.noisy_values and len(self.noisy_values[k]) > 0:
                out[f"Noisy    {k}"] = np.mean(self.noisy_values[k])
            out[f"Enhanced {k}"] = np.mean(self.enh_values[k])
        return out


# Multiprocessing Metric
class MPMetric(Metric):
    def __init__(
        self,
        name,
        pool: Pool,
        source_sr: Optional[int] = None,
        target_sr: Optional[int] = None,
    ):
        super().__init__(name, source_sr=source_sr, target_sr=target_sr)
        self.pool = pool
        self.worker_results = deque()

    def add(self, clean, enhanced, noisy):
        clean = self.maybe_resample(torch.as_tensor(clean)).squeeze(0)
        enhanced = self.maybe_resample(torch.as_tensor(enhanced)).squeeze(0)
        h = self.pool.apply_async(
            self.compute_metric,
            (clean, enhanced),
            callback=self._add_values_enh,
            error_callback=logger.error,
        )
        h.get()
        if noisy is not None:
            noisy = self.maybe_resample(torch.as_tensor(noisy)).squeeze(0)
            h = self.pool.apply_async(
                self.compute_metric,
                (clean, noisy),
                callback=self._add_values_noisy,
                error_callback=logger.error,
            )
            self.worker_results.append(h)

    def mean(self) -> Dict[str, float]:
        while len(self.worker_results) > 0:
            h = self.worker_results.popleft()
            h.get()
        self.pool.close()
        self.pool.join()
        return super().mean()

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict["pool"]
        del self_dict["worker_results"]
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


class SiSDRMetric(MPMetric):
    def __init__(self, pool: Pool):
        super().__init__(name="SISDR", pool=pool)

    def compute_metric(self, clean, degraded) -> float:
        return si_sdr_speechmetrics(reference=as_numpy(clean), estimate=as_numpy(degraded))


class StoiMetric(MPMetric):
    def __init__(self, sr: int, pool: Pool):
        super().__init__(name="STOI", pool=pool, source_sr=sr, target_sr=10000)

    def compute_metric(self, clean, degraded) -> float:
        assert self.sr is not None
        return stoi(clean=as_numpy(clean), degraded=as_numpy(degraded), sr=self.sr)


class PesqMetric(MPMetric):
    def __init__(self, sr: int, pool: Pool):
        super().__init__(name="PESQ", pool=pool, source_sr=sr, target_sr=16000)

    def compute_metric(self, clean, degraded) -> Union[float, np.ndarray]:
        assert self.sr is not None
        return pesq(self.sr, as_numpy(clean), as_numpy(degraded), "wb")


class CompositeMetric(MPMetric):
    def __init__(self, sr: int, pool: Pool, use_octave: bool = False):
        names = ["PESQ", "CSIG", "CBAK", "COVL", "SSNR"]
        super().__init__(names, pool=pool, source_sr=sr, target_sr=16000)
        self.use_octave = use_octave

    def compute_metric(self, clean, degraded) -> Union[float, np.ndarray]:
        assert self.sr is not None
        c = composite(
            clean=clean.squeeze(0),
            degraded=degraded.squeeze(0),
            sr=self.sr,
            use_octave=self.use_octave,
        )
        return c


class NoisyMetric(Metric):
    def add(self, enhanced, noisy):
        enhanced = self.maybe_resample(enhanced).squeeze(0)
        values_enh = self.compute_metric(degraded=enhanced)
        self._add_values_enh(values_enh)
        if noisy is not None:
            noisy = self.maybe_resample(noisy).squeeze(0)
            values_noisy = self.compute_metric(degraded=enhanced)
            self._add_values_noisy(values_noisy)

    @abstractmethod
    def compute_metric(self, degraded) -> Union[float, np.ndarray]:
        pass


class DnsMosP808ApiMetric(NoisyMetric):
    def __init__(self, sr: int):
        super().__init__(name="MOS", source_sr=sr, target_sr=16000)
        self.url = "https://dnsmos.azurewebsites.net/score"
        self.key = os.environ["DNS_AUTH_KEY"]

    def mos(self, audio) -> Union[float, np.ndarray]:
        # Set the content type
        headers = {"Content-Type": "application/json"}
        # If authentication is enabled, set the authorization header
        headers["Authorization"] = f"Basic {self.key}"

        data = {"data": audio.tolist(), "filename": "audio.wav"}
        input_data = json.dumps(data)

        try_flag = 1
        e = ""
        while try_flag:
            try:
                resp = requests.post(self.url, data=input_data, headers=headers, timeout=50)
                try_flag = 0
                score_dict = resp.json()
            except Exception as e:
                print(e)
                try_flag = 1
                print("retry_1")
                continue
            try:
                print(score_dict)
                return float(score_dict["mos"])
            except Exception as e:
                print(e)
                try_flag = 1
                print("retry_2")
                continue
        raise ValueError("Error gettimg mos:", e)

    def compute_metric(self, degraded) -> Union[float, np.ndarray]:
        assert self.sr is not None
        return self.mos(degraded)


class DnsMosP835ApiMetric(NoisyMetric):
    def __init__(self, sr: int):
        super().__init__(name=["SIGMOS", "BAKMOS", "OVLMOS"], source_sr=sr, target_sr=16000)
        self.url = "https://dnsmos.azurewebsites.net/v1/dnsmosp835/score"
        self.key = os.environ["DNS_AUTH_KEY"]

    def mos(self, audio) -> Union[float, np.ndarray]:
        # Set the content type
        headers = {"Content-Type": "application/json"}
        # If authentication is enabled, set the authorization header
        headers["Authorization"] = f"Basic {self.key}"

        data = {"data": audio.tolist(), "filename": "audio.wav"}
        input_data = json.dumps(data)

        try_flag = 1
        e = ""
        while try_flag:
            try:
                resp = requests.post(self.url, data=input_data, headers=headers, timeout=50)
                try_flag = 0
                score_dict = resp.json()
            except Exception as e:
                print(e)
                try_flag = 1
                print("retry_1")
                continue
            try:
                print(score_dict)
                c = np.asarray([float(score_dict[c]) for c in ("mos_sig", "mos_bak", "mos_ovr")])
                return c
            except Exception as e:
                print(e)
                try_flag = 1
                print("retry_2")
                continue
        raise ValueError("Error gettimg mos:", e)

    def compute_metric(self, degraded) -> Union[float, np.ndarray]:
        assert self.sr is not None
        return self.mos(degraded)


def as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    return x
