#!/bin/bash

PYTHON_V=$1
MANIFEST=$2
PYTHON_TAG=$(echo $PYTHON_V | tr -d ".")
PYBIN=$(ls /opt/python/cp$PYTHON_TAG*/python/bin)

$PYBIN/pip install maturin
$PYBIN/maturin build --release -m $MANIFEST
