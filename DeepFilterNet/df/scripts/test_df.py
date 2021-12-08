#!/usr/bin/env python

import os

import numpy as np
from loguru import logger

import df
from df.deepfilternet import ModelParams
from df.enhance import enhance, init_df, load_audio
from df.scripts.test_voicebank_demand import composite

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
    logger.info("Computing composite metrics")
    c_enh = composite(clean.numpy(), enhanced.numpy(), sr)
    assert np.isclose(
        c_enh, [2.63813972, 3.85677449, 2.51349003, 3.22993828, -2.69618571]
    ).all(), f"Metric output not close: {c_enh}"
