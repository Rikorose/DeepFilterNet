#!/usr/bin/env python

import os

import numpy as np
import torch
from loguru import logger

import df
from df.deepfilternet import ModelParams
from df.enhance import enhance, init_df, load_audio
from df.scripts.test_voicebank_demand import HAS_OCTAVE, composite, si_sdr_speechmetrics, stoi

__a_tol = 1e-4


def try_eval_composite(clean, enhanced, sr):
    if not HAS_OCTAVE:
        logger.warning("Octave not found. Skipping.")
        return
    logger.info("Computing composite metrics")
    m_enh = torch.as_tensor(
        composite(clean.squeeze(0).numpy(), enhanced.squeeze(0).numpy(), sr)
    ).to(torch.float32)
    logger.info(f"Got {m_enh}")
    m_target = torch.as_tensor(
        [2.3057448863983, 3.832368850708, 2.3624868392944, 3.054983377456, -2.792978048324]
    )
    assert torch.isclose(
        m_enh, m_target, atol=__a_tol
    ).all(), f"Metric output not close. Expected {m_target}, diff: {m_target-m_enh}"


def eval_pystoi(clean, enhanced, sr):
    logger.info("Computing STOI")
    m_enh = stoi(clean.squeeze(0), enhanced.squeeze(0), sr)
    m_target = 0.9689496585281197
    logger.info(f"Got {m_enh:.4f}")
    assert np.isclose(
        [m_enh], [m_target], atol=__a_tol
    ), f"Metric output not close. Expected {m_target}, diff: {m_target-m_enh}"


def eval_sdr(clean, enhanced):
    logger.info("Computing SI-SDR")
    m_enh = si_sdr_speechmetrics(clean.numpy(), enhanced.numpy())
    m_target = 18.878527879714966
    logger.info(f"Got {m_enh:.4f}")
    assert np.isclose(
        [m_enh], [m_target]
    ), f"Metric output not close. Expected {m_target}, diff: {m_target-m_enh}"


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
