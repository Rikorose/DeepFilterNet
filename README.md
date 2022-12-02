# DeepFilterNet
A Low Complexity Speech Enhancement Framework for Full-Band Audio (48kHz) using on Deep Filtering.

### News

- New real-time version and a LADSPA plugin
  - [Pre-compiled binary](#deep-filter), no python dependencies. Usage: `deep-filter audio-file.wav`
  - [LADSPA plugin](ladspa/) with pipewire filter-chain integration for real-time noise reduction on your mic.

- New DeepFilterNet2 Paper: *DeepFilterNet2: Towards Real-Time Speech Enhancement on Embedded Devices for Full-Band Audio*
  - Paper: https://arxiv.org/abs/2205.05474
  - Samples: https://rikorose.github.io/DeepFilterNet2-Samples/
  - Demo: https://huggingface.co/spaces/hshr/DeepFilterNet2

- Original DeepFilterNet Paper: *DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering*
  - Paper: https://arxiv.org/abs/2110.05588
  - Samples: https://rikorose.github.io/DeepFilterNet-Samples/
  - Demo: https://huggingface.co/spaces/hshr/DeepFilterNet
  - Video Lecture: https://youtu.be/it90gBqkY6k

## Usage

### deep-filter

Download a pre-compiled deep-filter binary from the [release page](https://github.com/Rikorose/DeepFilterNet/releases/).
You can use `deep-filter` to suppress noise in noisy .wav audio files. Currently, only wav files with a sampling rate of 48kHz are supported.

```bash
USAGE:
    deep-filter [OPTIONS] [FILES]...

ARGS:
    <FILES>...

OPTIONS:
    -D, --compensate-delay
            Compensate delay of STFT and model lookahead
    -h, --help
            Print help information
    -m, --model <MODEL>
            Path to model tar.gz. Defaults to DeepFilterNet2.
    -o, --out-dir <OUT_DIR>
            [default: out]
    --pf
            Enable postfilter
    -v, --verbose
            Logging verbosity
    -V, --version
            Print version information
```

If you want to use the pytorch backend e.g. for GPU processing, see further below for the Python usage.

### DeepFilterNet Framework

This framework supports Linux, MacOS and Windows. Training is only tested under Linux. The framework is structured as follows:

* `libDF` contains Rust code used for data loading and augmentation.
* `DeepFilterNet` contains DeepFilterNet code training, evaluation and visualization as well as pretrained model weights.
* `pyDF` contains a Python wrapper of libDF STFT/ISTFT processing loop.
* `pyDF-data` contains a Python wrapper of libDF dataset functionality and provides a pytorch data loader.
* `ladspa` contains a LADSPA plugin for real-time noise suppression.
* `models` contains pretrained for usage in DeepFilterNet (Python) or libDF/deep-filter (Rust)

### DeepFilterNet Python: PyPI

Install the DeepFilterNet Python wheel via pip:
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
# Required build dependency: HDF5 headers (e.g. ubuntu: libhdf5-dev)
maturin develop --release -m pyDF-data/Cargo.toml
# If you have troubles with hdf5 you may try to build and link hdf5 statically:
maturin develop --release --features hdf5-static -m pyDF-data/Cargo.toml
# Install remaining DeepFilterNet python dependencies
cd DeepFilterNet
poetry install -E train -E eval # Note: This globally installs DeepFilterNet in your environment
# Alternatively for developement: Install only dependencies and work with the repository version
poetry install -E train -E eval --no-root
# You may need to set the python path
export PYTHONPATH=$PWD
```

To enhance noisy audio files using DeepFilterNet run
```bash
$ python DeepFilterNet/df/enhance.py --help
usage: enhance.py [-h] [--model-base-dir MODEL_BASE_DIR] [--pf] [--output-dir OUTPUT_DIR] [--log-level LOG_LEVEL] [--compensate-delay]
                  noisy_audio_files [noisy_audio_files ...]

positional arguments:
  noisy_audio_files     List of noise files to mix with the clean speech file.

optional arguments:
  -h, --help            show this help message and exit
  --model-base-dir MODEL_BASE_DIR, -m MODEL_BASE_DIR
                        Model directory containing checkpoints and config.
                        To load a pretrained model, you may just provide the model name, e.g. `DeepFilterNet`.
                        By default, the pretrained DeepFilterNet2 model is loaded.
  --pf                  Post-filter that slightly over-attenuates very noisy sections.
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Directory in which the enhanced audio files will be stored.
  --log-level LOG_LEVEL
                        Logger verbosity. Can be one of (debug, info, error, none)
  --compensate-delay, -D
                        Add some paddig to compensate the delay introduced by the real-time STFT/ISTFT implementation.

# Enhance audio with original DeepFilterNet
python DeepFilterNet/df/enhance.py -m DeepFilterNet path/to/noisy_audio.wav

# Enhance audio with DeepFilterNet2
python DeepFilterNet/df/enhance.py -m DeepFilterNet2 path/to/noisy_audio.wav
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
python df/scripts/prepare_data.py --sr 48000 speech training_set.txt TRAIN_SET_SPEECH.hdf5
```
All datasets should be made available in one dataset folder for the train script.

The dataset configuration file should contain 3 entries: "train", "valid", "test". Each of those
contains a list of datasets (e.g. a speech, noise and a RIR dataset). You can use multiple speech
or noise dataset. Optionally, a sampling factor may be specified that can be used to over/under-sample
the dataset. Say, you have a specific dataset with transient noises and want to increase the amount
of non-stationary noises by oversampling. In most cases you want to set this factor to 1.

<details>
  <summary>Dataset config example:</summary>
<p>
  
`dataset.cfg`

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

</p>
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

## Citation Guide

If you use this framework, please cite: *DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering*

```bibtex
@inproceedings{schroeter2022deepfilternet,
  title={{DeepFilterNet}: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering}, 
  author = {Schröter, Hendrik and Escalante-B., Alberto N. and Rosenkranz, Tobias and Maier, Andreas},
  booktitle={ICASSP 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2022},
  organization={IEEE}
}
```

If you use the DeepFilterNet2 model, please cite: *DeepFilterNet2: Towards Real-Time Speech Enhancement on Embedded Devices for Full-Band Audio*

```bibtex
@inproceedings{schroeter2022deepfilternet2,
  title = {{DeepFilterNet2}: Towards Real-Time Speech Enhancement on Embedded Devices for Full-Band Audio},
  author = {Schröter, Hendrik and Escalante-B., Alberto N. and Rosenkranz, Tobias and Maier, Andreas},
  booktitle={17th International Workshop on Acoustic Signal Enhancement (IWAENC 2022)},
  year = {2022},
}

```

## License

DeepFilterNet is free and open source! All code in this repository is dual-licensed under either:

* MIT License ([LICENSE-MIT](LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))
* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))

at your option. This means you can select the license you prefer!

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
