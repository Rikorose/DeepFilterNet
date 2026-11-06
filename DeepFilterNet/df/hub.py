"""Load DeepFilterNet models from the Hugging Face Hub or a local directory."""

import os
from typing import Optional, Union

from .enhance import enhance, init_df
from .io import resample


class DeepFilterNet:
    """DeepFilterNet inference model and its audio processing state."""

    def __init__(self, model, df_state):
        self.model = model
        self.df_state = df_state
        self.sample_rate = df_state.sr()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ):
        """Load a local model directory or an unpacked Hugging Face model repository.

        The directory or repository must contain config.ini and checkpoints/.
        Hub options are passed to huggingface_hub.snapshot_download.
        """
        source = os.fspath(pretrained_model_name_or_path)
        if os.path.isdir(source):
            model_dir = source
        else:
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(
                repo_id=source,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                allow_patterns=["config.ini", "checkpoints/*"],
            )
        # Model snapshots are shared caches and should not contain per-run logs.
        model, state, _, _ = init_df(model_base_dir=model_dir, log_file=None)
        return cls(model, state)

    def enhance(self, audio, sample_rate: int):
        """Enhance a tensor [channels, samples], returning audio at the input rate."""
        if audio.ndim != 2:
            raise ValueError("audio must have shape [channels, samples]")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        audio = audio.cpu()
        if sample_rate != self.sample_rate:
            audio = resample(audio, sample_rate, self.sample_rate)
        audio = enhance(self.model, self.df_state, audio)
        if sample_rate != self.sample_rate:
            audio = resample(audio, self.sample_rate, sample_rate)
        return audio
