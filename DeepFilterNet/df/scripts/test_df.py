#!/usr/bin/env python

import os
import unittest
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from loguru import logger
from torch import Tensor

import df
from df.enhance import DF, enhance, init_df
from df.evaluation_utils import HAS_OCTAVE, composite, si_sdr_speechmetrics, stoi
from df.io import load_audio, save_audio

__a_tol = 1e-4
__r_tol = 1e-4


def eval_metric(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    clean: Tensor,
    enhanced: Tensor,
    m_target,
    prefix: str = "",
    metric_name: str = "",
) -> bool:
    logger.info(prefix + f"Computing {metric_name} metrics")
    m_t = torch.as_tensor(m_target)  # target metric
    m_e = torch.as_tensor(
        f(clean.squeeze(0).numpy(), enhanced.squeeze(0).numpy())
    )  # enhanced metric
    m_e = m_e.to(torch.float32)
    logger.info(prefix + f"Expected {m_t}")
    logger.info(prefix + f"Got      {m_e}")
    is_close = torch.isclose(m_e, m_t, atol=__a_tol, rtol=__r_tol).all()
    if not is_close:
        logger.error(prefix + f"Diff     {m_t-m_e}")
    return is_close


TARGET_METRICS = {
    "DeepFilterNet": {
        "composite": [
            2.30728650093078,
            3.83064246177673,
            2.36408281326293,
            3.05453467369079,
            -2.7631254196166,
        ],
        "stoi": 0.9689496585281197,
        "sdr": 18.88543128967285,
    },
    "DeepFilterNet2": {
        "composite": [
            2.87284946441650,
            4.17169523239136,
            2.75626921653748,
            3.51172018051147,
            -0.91267710924149,
        ],
        "stoi": 0.9725977621169399,
        "sdr": 19.41733717918396,
    },
    "DeepFilterNet3": {
        "composite": [
            3.04712939262390,
            4.23114347457886,
            2.77058529853821,
            3.61812996864319,
            -1.51538455486298,
        ],
        "stoi": 0.9742409586906433,
        "sdr": 20.014915466308594,
    },
}


def _get_metric(name: str, sr: int):
    METRICS = {
        "composite": [partial(composite, sr=sr), partial(composite, sr=sr, use_octave=True)],
        "stoi": [partial(stoi, sr=sr)],
        "sdr": [si_sdr_speechmetrics],
    }
    return METRICS[name]


def _load_model(model_n: str, **kwargs):
    kwargs.setdefault("config_allow_defaults", True)
    model, df_state, _, epoch = init_df(model_n, **kwargs)
    logger.info(f"Loaded model {model_n} with epoch {epoch}")
    return model, df_state


class TestDfModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_printoptions(precision=14, linewidth=120)
        os.makedirs("out", exist_ok=True)
        cls.df_dir = os.path.abspath(os.path.join(os.path.dirname(df.__file__), os.pardir))
        cls.models = {m: _load_model(m) for m in TARGET_METRICS.keys()}
        return cls

    def _test_model(
        self,
        model: torch.nn.Module,
        df_state: DF,
        target_metrics: Dict[str, Union[float, List[float]]],
        prefix: Optional[str] = None,
    ):
        out_n = f"out/enhanced_{prefix}.wav"
        prefix = prefix + " | " if prefix is not None else ""
        sr = df_state.sr()
        logger.info(prefix + "Loading audios")
        noisy, _ = load_audio(
            os.path.join(self.df_dir, os.path.pardir, "assets", "noisy_snr0.wav"), sr
        )
        clean, _ = load_audio(
            os.path.join(self.df_dir, os.path.pardir, "assets", "clean_freesound_33711.wav"), sr
        )
        enhanced = enhance(model, df_state, noisy, pad=True)
        save_audio(out_n, enhanced, sr)
        is_close = True
        for n in target_metrics.keys():
            for m in _get_metric(n, sr=sr):
                if not isinstance(m, partial) or "use_octave" not in m.keywords or HAS_OCTAVE:
                    cur_is_close = eval_metric(
                        m, clean, enhanced, m_target=target_metrics[n], prefix=prefix, metric_name=n
                    )
                    is_close = cur_is_close and is_close
        assert is_close

    def test_deepfilternet(self):
        model = "DeepFilterNet"
        self._test_model(*self.models[model], target_metrics=TARGET_METRICS[model], prefix=model)

    def test_deepfilternet2(self):
        model = "DeepFilterNet2"
        self._test_model(*self.models[model], target_metrics=TARGET_METRICS[model], prefix=model)

    def test_deepfilternet3(self):
        model = "DeepFilterNet3"
        self._test_model(*self.models[model], target_metrics=TARGET_METRICS[model], prefix=model)


if __name__ == "__main__":
    unittest.main()
