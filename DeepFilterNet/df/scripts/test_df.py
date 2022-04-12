#!/usr/bin/env python

import os
import unittest
from typing import Dict, List, Union

import numpy as np
import torch
from loguru import logger

import df
from df.enhance import DF, enhance, init_df, load_audio
from df.evaluation_utils import composite, si_sdr_speechmetrics, stoi

__a_tol = 1e-4


def eval_composite(clean, enhanced, sr, m_target: List[float]):  # type: ignore
    logger.info("Computing composite metrics")
    try:
        m_enh_octave = torch.as_tensor(
            composite(clean.squeeze(0).numpy(), enhanced.squeeze(0).numpy(), sr, use_octave=True)
        ).to(torch.float32)
    except (OSError, ImportError, ModuleNotFoundError):
        m_enh_octave = None
        logger.warning("No octave available")
    m_enh = torch.as_tensor(
        composite(clean.squeeze(0).numpy(), enhanced.squeeze(0).numpy(), sr)
    ).to(torch.float32)
    logger.info(f"Got {m_enh}")
    m_target: torch.Tensor = torch.as_tensor(m_target)
    assert torch.isclose(
        m_enh, m_target, atol=__a_tol
    ).all(), f"Metric output not close. Expected {m_target}, got {m_enh}, diff: {m_target-m_enh}"
    if m_enh_octave is not None:
        assert torch.isclose(
            m_enh_octave, m_target, atol=__a_tol
        ).all(), (
            f"Metric output not close. Expected {m_target}, got {m_enh}, diff: {m_target-m_enh}"
        )


def eval_pystoi(clean, enhanced, sr, m_target: float):
    logger.info("Computing STOI")
    m_enh = stoi(clean.squeeze(0), enhanced.squeeze(0), sr)
    logger.info(f"Got {m_enh}")
    assert np.isclose(
        [m_enh], [m_target], atol=__a_tol
    ), f"Metric output not close. Expected {m_target}, got {m_enh}, diff: {m_target-m_enh}"


def eval_sdr(clean, enhanced, m_target: float):
    logger.info("Computing SI-SDR")
    m_enh = si_sdr_speechmetrics(clean.numpy(), enhanced.numpy())
    logger.info(f"Got {m_enh}")
    assert np.isclose(
        [m_enh], [m_target]
    ), f"Metric output not close. Expected {m_target}, got {m_enh}, diff: {m_target-m_enh}"


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
            2.86751246452332,
            4.03339815139771,
            2.56429362297058,
            3.41470885276794,
            -2.79574084281921,
        ],
        "stoi": 0.9707452525900906,
        "sdr": 13.40160727500915,
    },
}


def _load_model(df_dir: str, model_n: str):
    model_base_dir = os.path.join(df_dir, "pretrained_models", model_n)
    model, df_state, _ = init_df(model_base_dir, config_allow_defaults=True)
    logger.info(f"Loaded model {model_n}")
    return model, df_state


class TestDfModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_printoptions(precision=14, linewidth=120)
        cls.df_dir = os.path.abspath(os.path.join(os.path.dirname(df.__file__), os.pardir))
        cls.models = {m: _load_model(cls.df_dir, m) for m in TARGET_METRICS.keys()}
        return cls

    def _test_model(
        self,
        model: torch.nn.Module,
        df_state: DF,
        target_metrics: Dict[str, Union[float, List[float]]],
    ):
        sr = df_state.sr()
        logger.info("Loading audios")
        noisy, _ = load_audio(
            os.path.join(self.df_dir, os.path.pardir, "assets", "noisy_snr0.wav"), sr
        )
        clean, _ = load_audio(
            os.path.join(self.df_dir, os.path.pardir, "assets", "clean_freesound_33711.wav"), sr
        )
        enhanced = enhance(model, df_state, noisy, pad=True)
        eval_composite(clean, enhanced, sr, target_metrics["composite"])  # type: ignore
        eval_pystoi(clean, enhanced, sr, m_target=target_metrics["stoi"])  # type: ignore
        eval_sdr(clean, enhanced, m_target=target_metrics["sdr"])  # type: ignore

    def test_deepfilternet(self):
        model = "DeepFilterNet"
        self._test_model(*self.models[model], target_metrics=TARGET_METRICS[model])

    def test_deepfilternet2(self):
        model = "DeepFilterNet2"
        self._test_model(*self.models[model], target_metrics=TARGET_METRICS[model])


if __name__ == "__main__":
    unittest.main()
