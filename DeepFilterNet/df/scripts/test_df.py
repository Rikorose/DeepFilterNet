#!/usr/bin/env python

import os
import unittest
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from loguru import logger

import df
from df.enhance import DF, enhance, init_df, load_audio
from df.evaluation_utils import composite, si_sdr_speechmetrics, stoi

__a_tol = 1e-4


def eval_composite(clean, enhanced, sr, m_target: List[float], prefix: str = ""):  # type: ignore
    logger.info(prefix + "Computing composite metrics")
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
    m_target: torch.Tensor = torch.as_tensor(m_target)
    logger.info(prefix + f"Expected {m_target}")
    logger.info(prefix + f"Got      {m_enh}")
    assert torch.isclose(m_enh, m_target, atol=__a_tol).all(), (
        prefix
        + f"Metric output not close. Expected {m_target}, got {m_enh}, diff: {m_target-m_enh}"
    )
    if m_enh_octave is not None:
        assert torch.isclose(m_enh_octave, m_target, atol=__a_tol).all(), (
            prefix
            + f"Metric output not close. Expected {m_target}, got {m_enh}, diff: {m_target-m_enh}"
        )


def eval_pystoi(clean, enhanced, sr, m_target: float, prefix: str = ""):
    logger.info(prefix + "Computing STOI")
    m_enh = stoi(clean.squeeze(0), enhanced.squeeze(0), sr)
    logger.info(prefix + f"Expected {m_target}")
    logger.info(prefix + f"Got      {m_enh}")
    assert np.isclose([m_enh], [m_target], atol=__a_tol), (
        prefix
        + f"Metric output not close. Expected {m_target}, got {m_enh}, diff: {m_target-m_enh}"
    )


def eval_sdr(clean, enhanced, m_target: float, prefix: str = ""):
    logger.info(prefix + "Computing SI-SDR")
    m_enh = si_sdr_speechmetrics(clean.numpy(), enhanced.numpy())
    logger.info(prefix + f"Expected {m_target}")
    logger.info(prefix + f"Got      {m_enh}")
    assert np.isclose([m_enh], [m_target]), (
        prefix
        + f"Metric output not close. Expected {m_target}, got {m_enh}, diff: {m_target-m_enh}"
    )


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
        prefix: Optional[str] = None,
    ):
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
        success = True
        try:
            eval_composite(clean, enhanced, sr, target_metrics["composite"], prefix=prefix)  # type: ignore
        except AssertionError as e:
            print(e)
            success = False
        try:
            eval_pystoi(clean, enhanced, sr, m_target=target_metrics["stoi"], prefix=prefix)  # type: ignore
        except AssertionError as e:
            print(e)
            success = False
        try:
            eval_sdr(clean, enhanced, m_target=target_metrics["sdr"], prefix=prefix)  # type: ignore
        except AssertionError as e:
            print(e)
            success = False
        assert success

    def test_deepfilternet(self):
        model = "DeepFilterNet"
        self._test_model(*self.models[model], target_metrics=TARGET_METRICS[model], prefix=model)

    def test_deepfilternet2(self):
        model = "DeepFilterNet2"
        self._test_model(*self.models[model], target_metrics=TARGET_METRICS[model], prefix=model)


if __name__ == "__main__":
    unittest.main()
