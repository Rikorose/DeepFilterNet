#!/bin/bash

set -e

USAGE="$0 <export-dir>"

if [ "$#" -ne 1 ]; then
    echo "$USAGE"
    exit 1
fi

if ! [ -d "$1" ]; then
    echo "$USAGE"
    exit 2
fi

cd "$1"

tract -v -O --partial \
  --onnx-ignore-output-shapes \
  enc.onnx run \
  --input-from-bundle enc_input.npz \
  --assert-output-bundle enc_output.npz
