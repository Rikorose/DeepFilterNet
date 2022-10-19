#!/bin/bash

set -e

USAGE="$0 <erb_dec.onnx>"

if [ "$#" -ne 1 ]; then
    echo "$USAGE"
    exit 1
fi

if ! [ -f "$1" ]; then
    echo "$USAGE"
    exit 2
fi

tract -v -O --partial --pulse 1 \
  -i 1,S,256,f32 \
  -i 1,64,S,8,f32 \
  -i 1,64,S,8,f32 \
  -i 1,64,S,16,f32 \
  -i 1,64,S,32,f32 \
  --onnx-ignore-output-shapes \
  "$1" dump --allow-random-input --profile --cost
