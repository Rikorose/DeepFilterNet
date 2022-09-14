#!/bin/bash

set -e

USAGE="$0 <enc.onnx>"

if [ "$#" -ne 1 ]; then
    echo "$USAGE"
    exit 1
fi

if ! [ -f "$1" ]; then
    echo "$USAGE"
    exit 2
fi

tract --allow-random-input -v -O --partial --pulse 1 -i 1,1,S,32,f32 -i 1,2,S,96,f32 "$1" dump --profile --cost
