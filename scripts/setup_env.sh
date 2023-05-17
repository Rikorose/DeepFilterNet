#!/bin/sh

pytorch_v="1.12"
cuda_version=11.3
PYTHON_V=3.9
MINICONDA_DIR=${MINICONDA_DIR:-"$HOME/miniconda"}

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
INSTALL_PYTESTDEPS=${INSTALL_PYTESTDEPS:-0}
SUFFIX=""

check_install_conda() {
  # Check if miniconda is already installed
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
}

setup_conda() {
  # Setup miniconda if found; this isn't done lazily, so expect some bash starup delay
  __conda_setup=$("$MINICONDA_DIR/bin/conda" 'shell.bash' 'hook')
  if [ $? -eq 0 ]; then
    eval "$__conda_setup"
  else
    if [ -f "$MINICONDA_DIR/etc/profile.d/conda.sh" ]; then
      . "$MINICONDA_DIR/etc/profile.d/conda.sh"
    else
      export PATH="$MINICONDA_DIR/bin:$PATH"
    fi
  fi
  unset __conda_setup
}

setup_env() {
  CLUSTER=$1
  PROJECT_HOME=$2
  if [ -n "$3" ]; then
    SUFFIX="-$3"
  fi
  echo "Running on CUDA: $cuda_version"

  env="df-$pytorch_v$nightly-cuda$cuda_version$SUFFIX"
  env_path="$MINICONDA_DIR/envs/$env"
  if ! [ -f "$env_path/bin/python$PYTHON_V" ]; then
    echo "Virtualenv $env not found. Installing..."
    check_install_conda
    setup_conda

    echo "Installing conda env to $env_path"
    conda create -y -q -p "$env_path" \
      python=$PYTHON_V \
      patchelf \
      "$pytorch_v_arg" \
      torchaudio \
      cudatoolkit=$cuda_version -c "pytorch$nightly" -c conda-forge
    conda activate "$env"

    INSTALL_LIBDF=1
    INSTALL_PYDEPS=1
  else
    # Only need to activate the existing env
    setup_conda
    conda activate "$env"
    conda install patchelf

    echo "Running on env: $CONDA_DEFAULT_ENV"
  fi
  if [ "$INSTALL_PYDEPS" -eq 1 ]; then
    echo "Installing requirements"
    echo pip install -r "$PROJECT_HOME"/DeepFilterNet/requirements.txt
    pip install -r "$PROJECT_HOME"/DeepFilterNet/requirements.txt
    pip install h5py
  fi
  if [ "$INSTALL_PYTESTDEPS" -eq 1 ]; then
    echo "Installing test requirements"
    echo pip install -r "$PROJECT_HOME"/DeepFilterNet/requirements_eval.txt
    pip install -r "$PROJECT_HOME"/DeepFilterNet/requirements_eval.txt
  fi
  if [ "$INSTALL_LIBDF" -eq 1 ]; then
    echo "Installing DeepFilterLib"
    cd "$PROJECT_HOME"/ || exit 10
    rustup default nightly
    # rustup update stable
    pip install -U maturin
    maturin develop --profile=release-lto -m "$PROJECT_HOME"/pyDF/Cargo.toml
    maturin develop --profile=release-lto -m "$PROJECT_HOME"/pyDF-data/Cargo.toml
  fi
}

export setup_env
