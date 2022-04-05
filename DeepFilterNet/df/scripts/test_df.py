#!/usr/bin/env python

import os

import numpy as np
import torch
from loguru import logger

import df
from df.deepfilternet import ModelParams
from df.enhance import enhance, init_df, load_audio
from df.evaluation_utils import composite, si_sdr_speechmetrics, stoi

__a_tol = 1e-4


def try_eval_composite(clean, enhanced, sr):
    logger.info("Computing composite metrics")
    m_enh = torch.as_tensor(
        # `use_octave=True` should not make a difference
        composite(clean.squeeze(0).numpy(), enhanced.squeeze(0).numpy(), sr)
    ).to(torch.float32)
    logger.info(f"Got {m_enh}")
    m_target = torch.as_tensor(
        [2.30728650093078, 3.830642461776733, 2.364082813262939, 3.05453467369079, -2.7631254196166]
    )
    assert torch.isclose(
        m_enh, m_target, atol=__a_tol
    ).all(), f"Metric output not close. Expected {m_target}, got {m_enh}, diff: {m_target-m_enh}"


def eval_pystoi(clean, enhanced, sr):
    logger.info("Computing STOI")
    m_enh = stoi(clean.squeeze(0), enhanced.squeeze(0), sr)
    m_target = 0.9689496585281197
    logger.info(f"Got {m_enh:.4f}")
    assert np.isclose(
        [m_enh], [m_target], atol=__a_tol
    ), f"Metric output not close. Expected {m_target}, got {m_enh}, diff: {m_target-m_enh}"


def eval_sdr(clean, enhanced):
    logger.info("Computing SI-SDR")
    m_enh = si_sdr_speechmetrics(clean.numpy(), enhanced.numpy())
    m_target = 18.88543128967285
    logger.info(f"Got {m_enh:.4f}")
    assert np.isclose(
        [m_enh], [m_target]
    ), f"Metric output not close. Expected {m_target}, got {m_enh}, diff: {m_target-m_enh}"


if __name__ == "__main__":
    df_dir = os.path.abspath(os.path.join(os.path.dirname(df.__file__), os.pardir))
    model_base_dir = os.path.join(df_dir, "pretrained_models", "DeepFilterNet")
    model, df_state, _ = init_df(model_base_dir)
    sr = ModelParams().sr
    logger.info("Loading audios")
    noisy, _ = load_audio(os.path.join(df_dir, os.path.pardir, "assets", "noisy_snr0.wav"), sr)
    clean, _ = load_audio(
        os.path.join(df_dir, os.path.pardir, "assets", "clean_freesound_33711.wav"), sr
    )
    logger.info("Running model")
    enhanced = enhance(model, df_state, noisy, pad=True)
    try_eval_composite(clean, enhanced, sr)
    eval_pystoi(clean, enhanced, sr)
    eval_sdr(clean, enhanced)
