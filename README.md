# DeepFilterNet
A Low Complexity Speech Enhancement Framework for Full-Band Audio (48kHz) based on Deep Filtering.
Audio samples from the voice bank/DEMAND test set can found at https://rikorose.github.io/DeepFilterNet-Samples/

* `libDF` contains Rust code used for data loading and augmentation.
* `DeepFilterNet` contains Python code including a libDF wrapper for data loading, DeepFilterNet training, testing and visualization.
* `models` contains DeepFilterNet model weights and config.

## Usage
This framework is currently only tested under Linux.

### PyPI

Install the DeepFilterNet python package via pip:
```bash
pip install deepfilternet
```

To enhance noisy audio files using DeepFilterNet run
```bash
# Specify an output directory with --output-dir [OUTPUT_DIR]
deepFilter path/to/noisy_audio.wav
```

### Manual Installation

Install cargo via [rustup](https://rustup.rs/). Usage of a `conda` or `virtualenv` recommended.

Installation of python dependencies and libDF:
```bash
cd path/to/DeepFilterNet/  # cd into repository
# Recommended: Install or activate a python env
pip install maturin poetry  # Used to compile libdf and DeepFilterNet python wheels
# Build and install libdf python package required for enhance.py
maturin develop --release -m pyDF/Cargo.toml
# Optional: Install libdfdata python package with dataset and dataloading functionality for training
maturin develop --release -m pyDF-data/Cargo.toml
# Mandatory: Install cpu/cuda pytorch dependency from pytorch.org, e.g.:
pip install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
# Install remaining DeepFilterNet python dependencies
cd DeepFilterNet
poetry install
```

To enhance noisy audio files using DeepFilterNet run
```bash
# usage: enhance.py [-h] [--output-dir OUTPUT_DIR] [--model_base_dir MODEL_BASE_DIR] noisy_audio_files [noisy_audio_files ...]
python DeepFilterNet/df/enhance.py DeepFilterNet/pretrained_models/DeepFilterNet/ path/to/noisy_audio.wav
```

## Citation

This code accompanies the paper 'DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering'.

```bibtex
@misc{schröter2021deepfilternet,
      title={DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering}, 
      author={Hendrik Schröter and Alberto N. Escalante-B. and Tobias Rosenkranz and Andreas Maier},
      year={2021},
      eprint={2110.05588},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

## License

DeepFilterNet is free and open source! All code in this repository is dual-licensed under either:

* MIT License ([LICENSE-MIT](docs/LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))
* Apache License, Version 2.0 ([LICENSE-APACHE](docs/LICENSE-APACHE) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))

at your option. This means you can select the license you prefer!

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
