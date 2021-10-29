#!/bin/bash

PYTHON_V=$1
MANIFEST=$2
PYTHON_TAG=$(echo $PYTHON_V | tr -d ".")
PYBIN=$(ls /opt/python/cp$PYTHON_TAG*/bin)

$PYBIN/pip install maturin
$PYBIN/maturin build --release -m $MANIFEST

for wheel in target/wheels/*.whl; do
    auditwheel repair "${wheel}"
done
