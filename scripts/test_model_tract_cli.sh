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

CONV_CH=$(./scripts/read_toml.py "$1/config.ini" deepfilternet conv_ch)
EMB_HIDDEN_DIM=$(./scripts/read_toml.py "$1/config.ini" deepfilternet emb_hidden_dim)
EMB_HIDDEN_DIM="$(($EMB_HIDDEN_DIM * 2))"
NB_ERB=$(./scripts/read_toml.py "$1/config.ini" df NB_ERB)
NB_DF=$(./scripts/read_toml.py "$1/config.ini" df NB_DF)

echo
echo "*** Now enc w/o pulse ***"
echo
# ENC no-pulse
tract -v -O \
  --onnx-ignore-output-shapes \
  "$1"/enc.onnx run \
  --io-long --steps \
  --input-from-bundle "$1"/enc_input.npz \
  --assert-output-bundle "$1"/enc_output.npz \
  --save-outputs "$1"/enc_output_tract.npz
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz lsnr || echo "LSNR NOT CLOSE"
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz e0 || echo "e0 NOT CLOSE"
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz e1 || echo "e1 NOT CLOSE"
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz e2 || echo "e2 NOT CLOSE"
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz e3 || echo "e3 NOT CLOSE"
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz emb || echo "emb NOT CLOSE"
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz c0 || echo "c0 NOT CLOSE"
echo "*** OK ***"

# ERB decoder no-pulse
# Filter encoder output for erb decoder usage (i.e. remove c0)
echo
echo "*** Now ERB dec w/o pulse ***"
echo
./scripts/split_npz.py "$1"/enc_output_tract.npz "$1"/erb_dec_input_tract.npz emb e0 e1 e2 e3
tract -v -O \
  --onnx-ignore-output-shapes \
  "$1"/erb_dec.onnx run \
  --allow-random-input \
  --input-from-bundle "$1"/erb_dec_input_tract.npz \
  --assert-output-bundle "$1"/erb_dec_output.npz \
  --save-outputs "$1"/erb_dec_output_tract.npz
./scripts/assert_close_npz.py "$1"/erb_dec_output_tract.npz "$1"/erb_dec_output.npz m || echo "m NOT CLOSE"
echo "*** OK ***"

# DF decoder no-pulse
echo
echo "*** Now DF dec w/o pulse ***"
echo
./scripts/split_npz.py "$1"/enc_output_tract.npz "$1"/df_dec_input_tract.npz emb c0
tract -v -O \
  --onnx-ignore-output-shapes \
  "$1"/df_dec.onnx run \
  --allow-random-input \
  --input-from-bundle "$1"/df_dec_input_tract.npz \
  --save-outputs "$1"/df_dec_output_tract.npz
./scripts/assert_close_npz.py "$1"/df_dec_output_tract.npz "$1"/df_dec_output.npz coefs || echo "coefs NOT CLOSE"
echo "*** OK ***"

echo
echo "*** Now enc with pulse 1 ***"
echo
# Encoder pulse 1
tract -v \
  "$1"/enc.onnx \
  --onnx-ignore-output-shapes \
  -i 1,1,S,"$NB_ERB",f32 -i 1,2,S,"$NB_DF",f32 \
  --pulse 1 run \
  --input-from-bundle "$1"/enc_input.npz \
  --assert-output-bundle "$1"/enc_output.npz \
  --save-outputs "$1"/enc_output_tract.npz
echo "*** Testing output ***"
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz lsnr 1 || echo "LSNR NOT CLOSE"
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz e0 2 || echo "e0 NOT CLOSE"
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz e1 2 || echo "e1 NOT CLOSE"
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz e2 2 || echo "e2 NOT CLOSE"
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz e3 2 || echo "e3 NOT CLOSE"
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz emb 1 || echo "emb NOT CLOSE"
./scripts/assert_close_npz.py "$1"/enc_output_tract.npz "$1"/enc_output.npz c0 2 || echo "c0 NOT CLOSE"
echo "*** OK ***"

echo
echo "*** Now ERB dec with pulse ***"
echo "*** Hidden dim $EMB_HIDDEN_DIM ***"
echo
tract -v \
  "$1"/erb_dec.onnx \
  --onnx-ignore-output-shapes \
  -i 1,S,"$EMB_HIDDEN_DIM",f32 \
  -i 1,"$CONV_CH",S,$(("$NB_ERB" / 4)),f32 \
  -i 1,"$CONV_CH",S,$(("$NB_ERB" / 4)),f32 \
  -i 1,"$CONV_CH",S,$(("$NB_ERB" / 2)),f32 \
  -i 1,"$CONV_CH",S,$(("$NB_ERB")),f32 \
  --pulse 1 run \
  --input-from-bundle "$1"/erb_dec_input.npz \
  --assert-output-bundle "$1"/erb_dec_output.npz \
  --save-outputs "$1"/erb_dec_output_tract.npz
echo "*** Testing output ***"
./scripts/assert_close_npz.py "$1"/erb_dec_output_tract.npz "$1"/erb_dec_output.npz m 2 || echo "m NOT CLOSE"
echo "*** OK ***"

echo
echo "*** Now DF dec with pulse ***"
echo
tract -v \
  "$1"/df_dec.onnx \
  --onnx-ignore-output-shapes \
  -i 1,S,"$EMB_HIDDEN_DIM",f32 \
  -i 1,"$CONV_CH",S,$(("$NB_DF")),f32 \
  --pulse 1 run \
  --input-from-bundle "$1"/df_dec_input_tract.npz \
  --save-outputs "$1"/df_dec_output_tract.npz
echo "*** Testing output ***"
./scripts/assert_close_npz.py "$1"/df_dec_output_tract.npz "$1"/df_dec_output.npz coefs 1 || echo "coefs NOT CLOSE"
echo "*** OK ***"
