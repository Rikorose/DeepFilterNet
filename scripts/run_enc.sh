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

tract -v -O \
  --onnx-ignore-output-shapes \
  "$1"/enc.onnx run \
  --io-long --steps \
  --input-from-bundle "$1"/enc_input.npz \
  --assert-output-bundle "$1"/enc_output.npz \
  --save-outputs "$1"/enc_output_tract.npz

./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz lsnr
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz e0
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz e1
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz e2
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz e3
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz emb
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz c0

# Filter encoder output for erb decoder usage (i.e. remove c0)
./scripts/split_npz.py "$1"/enc_output_tract.npz "$1"/erb_dec_input_tract.npz emb e0 e1 e2 e3

# Why does this not work?
tract -v -O \
  --onnx-ignore-output-shapes \
  "$1"/erb_dec.onnx run \
  --allow-random-input \
  --input-from-bundle "$1"/erb_dec_input_tract.npz \
  --assert-output-bundle "$1"/erb_dec_output.npz \
  --save-outputs "$1"/erb_dec_output_tract.npz

./scripts/assert_close_npz.py "$1"/erb_dec_output_tract.npz "$1"/erb_dec_output.npz m
