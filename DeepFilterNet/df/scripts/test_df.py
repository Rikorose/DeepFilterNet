import numpy as np

from df.deepfilternet import ModelParams
from df.enhance import enhance, init_df, load_audio
from df.scripts.test_voicebank_demand import composite

if __name__ == "__main__":
    model, df_state, _ = init_df("DeepFilterNet/pretrained_models/DeepFilterNet/")
    noisy, orig_sr = load_audio("assets/noisy_snr0.wav")
    clean, orig_sr = load_audio("assets/clean_freesound_33711.wav")
    enhanced = enhance(model, df_state, noisy, pad=True)
    c_enh = composite(clean.numpy(), enhanced.numpy(), ModelParams().sr)
    assert np.isclose(c_enh, [2.63813972, 3.85677449, 2.51349003, 3.22993828, -2.69618571]).all()
