#!/bin/bash

PYTHON_V=$1
MANIFEST=$2
PYTHON_TAG=$(echo $PYTHON_V | tr -d ".")
PYBIN="/opt/python/cp$PYTHON_TAG*/bin"

echo $PYBIN
echo ${PYBIN}
ls ${PYBIN}

${PYBIN}/pip install maturin
${PYBIN}/maturin build --release -m $MANIFEST -i ${PYBIN}/python
