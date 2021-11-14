#!/bin/sh

pytorch_v="1.8"
cuda_version=11.1
PYTHON_V=3.9

PYTORCH_NIGHTLY=${PYTORCH_NIGHTLY:-0}
if [[ "$PYTORCH_NIGHTLY" -eq 1 ]]; then
  nightly="-nightly"
  pytorch_v_arg="pytorch"
  echo Running on pytorch nightly
else
  pytorch_v_arg="pytorch=$pytorch_v"
fi
INSTALL_LIBDF=${INSTALL_LIBDF:-1}
SUFFIX=""

setup_env() {
  CLUSTER=$1
  MINICONDA_DIR="$CLUSTER/miniconda"
  PROJECT_HOME=$2
  if [ ! -z "$3" ]; then
    SUFFIX="-$3"
  fi
  env="df-$pytorch_v$nightly-cuda$cuda_version$SUFFIX"
  if ! [ -f $MINICONDA_DIR/envs/$env/bin/python$PYTHON_V ]; then
    # Check if miniconda is already installed
    echo "Virtualenv $env not found. Installing..."
    if ! [ -f $MINICONDA_DIR/bin/python$PYTHON_V ]; then
      if [ -d $MINICONDA_DIR ]; then
        echo "Miniconda directory already exists, but python not found."
        echo "Please cleanup miniconda dir first for an automatic installation."
        exit 1
      fi
      echo "Miniconda not found. Installing..."
      wget -nv https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O $CLUSTER/miniconda.sh
      bash $CLUSTER/miniconda.sh -b -p $MINICONDA_DIR
      rm $CLUSTER/miniconda.sh
    fi
    export PATH="$MINICONDA_DIR/bin:$PATH"

    echo "Running on CUDA: $cuda_version"

    conda create -y -q -n $env python=$PYTHON_V \
      "$pytorch_v_arg" \
      torchaudio \
      maturin \
      cudatoolkit=$cuda_version -c "pytorch$nightly" -c conda-forge
    source activate $env

    echo "Installing requirements.txt"
    pip install -q -r $PROJECT_HOME/requirements.txt
    INSTALL_LIBDF=1
  else
    # Only need to activate the existing env
    if ! [ -x "$(command -v conda)" ]; then
      export PATH="$MINICONDA_DIR/bin:$PATH"
    fi
    source activate $env

    echo "Running on env: $CONDA_DEFAULT_ENV"
  fi
  if [ $INSTALL_LIBDF -eq 1 ]; then
    cd $PROJECT_HOME/; rustup default nightly
    maturin build --release -i python$PYTHON_V -m $PROJECT_HOME/DeepFilterNet/Cargo.toml
    # Python version without dot
    PV=$(echo $PYTHON_V | tr -d ".")
    # DF crate version
    DFV=$(sed -nr "/^\[package\]/ { :l /^version[ ]*=/ { s/.*=[ ]*//; p; q;}; n; b l;}" DeepFilterNet/Cargo.toml | tr -d "\"")
    pip install --force-reinstall $PROJECT_HOME/target/wheels/DeepFilterNet-DFV-cpDF-cpDF-linux_x86_64.whl
  fi
}

export setup_env
