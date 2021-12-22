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
      maturin \
      poetry \
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
    maturin build --release -i python$PYTHON_V -m "$PROJECT_HOME"/pyDF/Cargo.toml
    maturin build --release -i python$PYTHON_V -m "$PROJECT_HOME"/pyDF-data/Cargo.toml
    # Python version without dot
    PV=$(echo $PYTHON_V | tr -d ".")
    # DF crate version
    DFV=$(sed -nr "/^\[package\]/ { :l /^version[ ]*=/ { s/.*=[ ]*//; p; q;}; n; b l;}" pyDF/Cargo.toml | tr -d "\"")
    DFV=$(echo "$DFV" | sed -r "s/-pre/_pre/g")
    echo "Found df version: $DFV"
    pip install --force-reinstall --no-deps "$PROJECT_HOME"/target/wheels/DeepFilterLib-"$DFV"-cp"$PV"-cp"$PV"-*linux*_x86_64.whl
    pip install --force-reinstall --no-deps "$PROJECT_HOME"/target/wheels/DeepFilterDataLoader-"$DFV"-cp"$PV"-cp"$PV"-*linux*_x86_64.whl
  fi
  if [ $INSTALL_PYDEPS -eq 1 ]; then
    echo "Installing requirements"
    (cd "$PROJECT_HOME"/DeepFilterNet && poetry install -E "train")
  fi
}

export setup_env
