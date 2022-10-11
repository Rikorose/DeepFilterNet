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

./scripts/assert_close_npz.py "$1"/enc_output.npz "$1"/enc_output_tract.npz lsnr
./scripts/assert_close_npz.py "$1"/enc_output.npz "$1"/enc_output_tract.npz e0
./scripts/assert_close_npz.py "$1"/enc_output.npz "$1"/enc_output_tract.npz e1
./scripts/assert_close_npz.py "$1"/enc_output.npz "$1"/enc_output_tract.npz e2
./scripts/assert_close_npz.py "$1"/enc_output.npz "$1"/enc_output_tract.npz e3
./scripts/assert_close_npz.py "$1"/enc_output.npz "$1"/enc_output_tract.npz emb
./scripts/assert_close_npz.py "$1"/enc_output.npz "$1"/enc_output_tract.npz c0

# Filter encoder output for erb decoder usage (i.e. remove c0)
./scripts/split_npz.py "$1"/enc_output_tract.npz "$1"/erb_dec_input_tract.npz emb e0 e1 e2 e3

# This works
tract -v -O \
  --input-facts-from-bundle "$1"/erb_dec_input_tract.npz \
  --onnx-ignore-output-shapes \
  "$1"/erb_dec.onnx dump --allow-random-input

# Why does this not work?
tract -v -O \
  --onnx-ignore-output-shapes \
  "$1"/erb_dec.onnx run \
  --allow-random-input \
  --input-from-bundle "$1"/erb_dec_input_tract.npz \
  --assert-output-bundle "$1"/erb_dec_output.npz \
  --save-outputs "$1"/erb_dec_output_tract.npz

# Fails with:
#
# [2022-10-11T13:57:29.170512788Z INFO  tract::tensor] Using fixed input for input called emb (1 turn(s))
# [2022-10-11T13:57:29.170541763Z INFO  tract::tensor] Using fixed input for input called e3 (1 turn(s))
# [2022-10-11T13:57:29.170562695Z INFO  tract::tensor] Using fixed input for input called e2 (1 turn(s))
# [2022-10-11T13:57:29.170587231Z INFO  tract::tensor] Using fixed input for input called e1 (1 turn(s))
# [2022-10-11T13:57:29.170656776Z INFO  tract::tensor] Using fixed input for input called e0 (1 turn(s))
# thread 'main' panicked at 'called `Result::unwrap()` on an `Err` value: Undetermined symbol in expression: t', /home/hendrik/.cargo/registry/src/github.com-1ecc6299db9ec823/tract-core-0.18.1/src/ops/cnn/deconv/deconv_sum.rs:48:58
# stack backtrace:
#    0: rust_begin_unwind
#              at /rustc/4b91a6ea7258a947e59c6522cd5898e7c0a6a88f/library/std/src/panicking.rs:584:5
#    1: core::panicking::panic_fmt
#              at /rustc/4b91a6ea7258a947e59c6522cd5898e7c0a6a88f/library/core/src/panicking.rs:142:14
#    2: core::result::unwrap_failed
#              at /rustc/4b91a6ea7258a947e59c6522cd5898e7c0a6a88f/library/core/src/result.rs:1805:5
#    3: <smallvec::SmallVec<A> as core::iter::traits::collect::Extend<<A as smallvec::Array>::Item>>::extend
#    4: <tract_core::ops::cnn::deconv::deconv_sum::DeconvSum as tract_core::ops::EvalOp>::eval
#    5: tract_core::plan::SimpleState<F,O,M,P>::run_plan_with_eval
#    6: tract::run::run_regular
#    7: tract::run::handle
#    8: tract::main
# note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.
