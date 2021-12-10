# DeepFilterNet
A Low Complexity Speech Enhancement Framework for Full-Band Audio (48kHz) based on Deep Filtering.
Audio samples from the voice bank/DEMAND test set can be found at https://rikorose.github.io/DeepFilterNet-Samples/

* `libDF` contains Rust code used for data loading and augmentation.
* `DeepFilterNet` contains DeepFilterNet code training, evaluation and visualization as well as pretrained model weights.
* `pyDF` contains a Python wrapper of libDF STFT/ISTFT processing loop.
* `pyDF-data` contains a Python wrapper of libDF dataset functionality and provides a pytorch data loader.

## Usage
This framework supports Linux, MacOS and Windows. Training is only tested under Linux.

### PyPI

Install the DeepFilterNet python package via pip:
```bash
# Install cpu/cuda pytorch (>=1.8) dependency from pytorch.org, e.g.:
pip install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
# Install DeepFilterNet
pip install deepfilternet
# Or install DeepFilterNet including data loading functionality for training (Linux only)
pip install deepfilternet[train]
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
# Mandatory: Install cpu/cuda pytorch (>=1.8) dependency from pytorch.org, e.g.:
pip install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
# Install build dependencies used to compile libdf and DeepFilterNet python wheels
pip install maturin poetry
# Build and install libdf python package required for enhance.py
maturin develop --release -m pyDF/Cargo.toml
# Optional: Install libdfdata python package with dataset and dataloading functionality for training
maturin develop --release -m pyDF-data/Cargo.toml
# Install remaining DeepFilterNet python dependencies
cd DeepFilterNet
poetry install
```

To enhance noisy audio files using DeepFilterNet run
```bash
# usage: enhance.py [-h] [--output-dir OUTPUT_DIR] [--model_base_dir MODEL_BASE_DIR] noisy_audio_files [noisy_audio_files ...]
python DeepFilterNet/df/enhance.py DeepFilterNet/pretrained_models/DeepFilterNet/ path/to/noisy_audio.wav
```

### Training

The entry point is `DeepFilterNet/df/train.py`. It expects a data directory containing HDF5 dataset
as well as a dataset configuration json file.

So, you first need to create your datasets in HDF5 format. Each dataset typically only
holds training, validation, or test set of noise, speech or RIRs.
```py
# Install additional dependencies for dataset creation
pip install h5py librosa soundfile
# Go to DeepFilterNet python package
cd path/to/DeepFilterNet/DeepFilterNet
# Prepare text file (e.g. called training_set.txt) containing paths to .wav files
#
# usage: prepare_data.py [-h] [--num_workers NUM_WORKERS] [--max_freq MAX_FREQ] [--sr SR] [--dtype DTYPE]
#                        [--codec CODEC] [--mono] [--compression COMPRESSION]
#                        type audio_files hdf5_db
#
# where:
#   type: One of `speech`, `noise`, `rir`
#   audio_files: Text file containing paths to audio files to include in the dataset
#   hdf5_db: Output HDF5 dataset.
python df/prepare_data.py --sr 48000 speech training_set.txt TRAIN_SET_SPEECH.hdf5
```
All datasets should be made available in one dataset folder for the train script.

The dataset configuration file should contain 3 entries: "train", "valid", "test". Each of those
contains a list of datasets (e.g. a speech, noise and a RIR dataset). Optionally a sampling factor
may be specified that can be used to over/under-sample the dataset. Say, you have a specific dataset
with transient noises and want to increase the amount of non-stationary noises by oversampling.
In most cases you want to set this factor to 1.

<details>
  <summary>
`dataset.cfg` example:
  </summary>
```json
{
  "train": [
    [
      "TRAIN_SET_SPEECH.hdf5",
      1.0
    ],
    [
      "TRAIN_SET_NOISE.hdf5",
      1.0
    ],
    [
      "TRAIN_SET_RIR.hdf5",
      1.0
    ]
  ],
  "valid": [
    [
      "VALID_SET_SPEECH.hdf5",
      1.0
    ],
    [
      "VALID_SET_NOISE.hdf5",
      1.0
    ],
    [
      "VALID_SET_RIR.hdf5",
      1.0
    ]
  ],
  "test": [
    [
      "TEST_SET_SPEECH.hdf5",
      1.0
    ],
    [
      "TEST_SET_NOISE.hdf5",
      1.0
    ],
    [
      "TEST_SET_RIR.hdf5",
      1.0
    ]
  ]
}
```
</details>

Finally, start the training script. The training script may create a model `base_dir` if not
existing used for logging, some audio samples, model checkpoints, and config. If no config file is
found, it will create a default config. See
[DeepFilterNet/pretrained_models/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet/blob/main/DeepFilterNet/pretrained_models/DeepFilterNet/config.ini)
for a config file.
```py
# usage: train.py [-h] [--debug] data_config_file data_dir base_dir
python df/train.py path/to/dataset.cfg path/to/data_dir/ path/to/base_dir/
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
