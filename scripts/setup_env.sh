#!/bin/sh

pytorch_v="1.10"
cuda_version=11.1
PYTHON_V=3.9

PYTORCH_NIGHTLY=${PYTORCH_NIGHTLY:-0}
if [ "$PYTORCH_NIGHTLY" -eq 1 ]; then
  nightly="-nightly"
  pytorch_v_arg="pytorch"
  echo Running on pytorch nightly
else
  pytorch_v_arg="pytorch=$pytorch_v"
fi
INSTALL_LIBDF=${INSTALL_LIBDF:-1}
INSTALL_PYDEPS=${INSTALL_PYDEPS:-0}
SUFFIX=""

setup_env() {
  CLUSTER=$1
  MINICONDA_DIR="$CLUSTER/miniconda"
  PROJECT_HOME=$2
  if [ ! -z "$3" ]; then
    SUFFIX="-$3"
  fi

  env="df-$pytorch_v$nightly-cuda$cuda_version$SUFFIX"
  env_path="$MINICONDA_DIR/envs/$env"
  if ! [ -f "$env_path/bin/python$PYTHON_V" ]; then
    # Check if miniconda is already installed
    echo "Virtualenv $env not found. Installing..."
    if ! [ -f "$MINICONDA_DIR"/bin/python$PYTHON_V ]; then
      if [ -d "$MINICONDA_DIR" ]; then
        echo "Miniconda directory already exists, but python not found."
        echo "Please cleanup miniconda dir first for an automatic installation."
        exit 1
      fi
      echo "Miniconda not found. Installing..."
      wget -nv https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O "$CLUSTER"/miniconda.sh
      bash "$CLUSTER"/miniconda.sh -b -p "$MINICONDA_DIR"
      rm "$CLUSTER"/miniconda.sh
    fi
    export PATH="$MINICONDA_DIR/bin:$PATH"

    echo "Running on CUDA: $cuda_version"

    echo "Installing conda env to $env_path"
    conda create -y -q -p "$env_path" \
      python=$PYTHON_V \
      "$pytorch_v_arg" \
      torchaudio \
      cudatoolkit=$cuda_version -c "pytorch$nightly" -c conda-forge
    source activate "$env"

    INSTALL_LIBDF=1
    INSTALL_PYDEPS=1
  else
    # Only need to activate the existing env
    if ! [ -x "$(command -v conda)" ]; then
      export PATH="$MINICONDA_DIR/bin:$PATH"
    fi
    source activate "$env"

    echo "Running on env: $CONDA_DEFAULT_ENV"
  fi
  if [ $INSTALL_LIBDF -eq 1 ]; then
    echo "Installing DeepFilterLib"
    cd "$PROJECT_HOME"/ || exit 10
    rustup default stable
    rustup update stable
    cargo build --release
    pip install -U maturin
    maturin develop --release -m "$PROJECT_HOME"/pyDF/Cargo.toml
    maturin develop --release -m "$PROJECT_HOME"/pyDF-data/Cargo.toml
  fi
  if [ $INSTALL_PYDEPS -eq 1 ]; then
    echo "Installing requirements"
    echo pip install -r "$PROJECT_HOME"/DeepFilterNet/requirements.txt
    pip install -r "$PROJECT_HOME"/DeepFilterNet/requirements.txt
  fi
}

export setup_env
