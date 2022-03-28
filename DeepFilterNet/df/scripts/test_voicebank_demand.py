import glob
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pystoi
import torch
import torch.multiprocessing as mp
from loguru import logger
from pesq import pesq
from torch.multiprocessing.pool import Pool
from torchaudio.transforms import Resample

from df.enhance import df_features, init_df, load_audio, save_audio, setup_df_argument_parser
from df.model import ModelParams
from df.modules import get_device
from df.utils import as_complex, get_resample_params, resample

RESAMPLE_METHOD = "sinc_fast"

HAS_OCTAVE = True
try:
    import semetrics
except OSError or ImportError:
    HAS_OCTAVE = False

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, total: Optional[int] = None, log_freq_percent=25, desc="Progress"):
        assert 0 < log_freq_percent < 100
        # tqdm not available using fallback
        logged = set()
        try:
            L = iterable.__len__()
        except AttributeError:
            assert total is not None
            L = total

        for k, i in enumerate(iterable):
            yield i
            p = (k + 1) / L
            progress = int(100 * p)
            if progress % log_freq_percent == 0 and progress > 0:
                if progress not in logged:
                    logger.info("{}: {: >2d}%".format(desc, progress))
                    logged.add(progress)


def main(args):
    model, df_state, suffix = init_df(
        args.model_base_dir,
        post_filter=args.pf,
        log_level=args.log_level,
        config_allow_defaults=True,
    )
    assert os.path.isdir(args.dataset_dir)
    if not HAS_OCTAVE:
        logger.warning("Running without octave. Skipping composite metrics")
    sr = ModelParams().sr
    noisy_dir = os.path.join(args.dataset_dir, "noisy_testset_wav")
    clean_dir = os.path.join(args.dataset_dir, "clean_testset_wav")
    assert os.path.isdir(noisy_dir) and os.path.isdir(clean_dir)
    with mp.Pool(processes=args.metric_workers) as pool:
        metrics: List[Metric] = [StoiMetric(sr, pool), SiSDRMetric(pool), CompositeMetric(sr, pool)]
        noisy_files = glob.glob(noisy_dir + "/*wav")
        clean_files = glob.glob(clean_dir + "/*wav")
        for noisyfn, cleanfn in tqdm(zip(noisy_files, clean_files), total=len(noisy_files)):
            noisy, _ = load_audio(noisyfn, sr)
            clean, _ = load_audio(cleanfn, sr)
            enh = enhance(model, df_state, noisy)[0]
            clean = df_state.synthesis(df_state.analysis(clean.numpy()))[0]
            noisy = df_state.synthesis(df_state.analysis(noisy.numpy()))[0]
            for m in metrics:
                m.add(clean=clean, enhanced=enh, noisy=noisy)
            enh = torch.as_tensor(enh).to(torch.float32).view(1, -1)
            if args.output_dir is not None:
                save_audio(
                    os.path.basename(cleanfn),
                    enh,
                    sr,
                    output_dir=args.output_dir,
                    suffix=f"{suffix}",
                )
        for m in metrics:
            for k, v in m.mean().items():
                logger.info(f"{k}: {v}")


def stoi(clean, degraded, sr, extended=False):
    assert len(clean.shape) == 1
    if sr != 10000:
        clean = resample(torch.as_tensor(clean), sr, 10000, method=RESAMPLE_METHOD).numpy()
        degraded = resample(torch.as_tensor(degraded), sr, 10000, method=RESAMPLE_METHOD).numpy()
        sr = 10000
    return pystoi.stoi(x=clean, y=degraded, fs_sig=sr, extended=extended)


def composite(clean: np.ndarray, degraded: np.ndarray, sr: int) -> np.ndarray:
    """Compute pesq, csig, cbak, covl, ssnr"""
    assert len(clean.shape) == 1
    if sr != 16000:
        clean = resample(torch.as_tensor(clean), sr, 16000, method=RESAMPLE_METHOD).numpy()
        degraded = resample(torch.as_tensor(degraded), sr, 16000, method=RESAMPLE_METHOD).numpy()
        sr = 16000
    clean = as_numpy(clean)
    degraded = as_numpy(degraded)
    if HAS_OCTAVE:
        c = semetrics.composite(clean, degraded, sr=sr, mp=True)
    else:
        c = [pesq(sr, clean, degraded, "wb"), 0, 0, 0, 0]
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

    def resample_and_compute(
        self, clean, enhanced, noisy
    ) -> Tuple[Union[float, np.ndarray, None], ...]:
        clean = self.maybe_resample(clean)
        enhanced = self.maybe_resample(enhanced)
        m_enh = self.compute_metric(clean=clean, degraded=enhanced)
        m_noisy = None
        if noisy is not None:
            noisy = self.maybe_resample(noisy)
            m_noisy = self.compute_metric(clean=clean, degraded=noisy)
        return m_enh, m_noisy

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

    def maybe_resample(self, x):
        if self.resampler is not None:
            x = self.resampler.forward(torch.as_tensor(x))
        return x

    def add(self, clean, enhanced, noisy):
        clean = self.maybe_resample(clean)
        enhanced = self.maybe_resample(enhanced)
        values_enh = self.compute_metric(clean=clean, degraded=enhanced)
        self._add_values_enh(values_enh)
        if noisy is not None:
            noisy = self.maybe_resample(noisy)
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

    def add(self, clean, enhanced, noisy):
        clean = self.maybe_resample(torch.as_tensor(clean))
        enhanced = self.maybe_resample(torch.as_tensor(enhanced))
        self.pool.apply_async(self.compute_metric, (clean, enhanced), callback=self._add_values_enh)
        if noisy is not None:
            noisy = self.maybe_resample(torch.as_tensor(noisy))
            self.pool.apply_async(
                self.compute_metric, (clean, noisy), callback=self._add_values_noisy
            )

    def mean(self) -> Dict[str, float]:
        self.pool.close()
        self.pool.join()
        return super().mean()

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict["pool"]
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


class CompositeMetric(MPMetric):
    def __init__(self, sr: int, pool: Pool):
        names = ["PESQ", "CSIG", "CBAK", "COVL", "SSNR"] if HAS_OCTAVE else "PESQ"
        super().__init__(names, pool=pool, source_sr=sr, target_sr=16000)

    def compute_metric(self, clean, degraded) -> Union[float, np.ndarray]:
        assert self.sr is not None
        if HAS_OCTAVE:
            return composite(clean=clean, degraded=degraded, sr=self.sr)
        else:
            return composite(clean=clean, degraded=degraded, sr=self.sr)[0]


def as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    return x


if __name__ == "__main__":
    parser = setup_df_argument_parser()
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Voicebank Demand Test set directory. Must contain 'noisy_testset_wav' and 'clean_testset_wav'",
    )
    parser.add_argument("--metric-workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
