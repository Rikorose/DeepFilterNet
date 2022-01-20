#!/bin/bash

set -e

WORKING_DIR=${WORKING_DIR:-out/}
OUT_DIR=${OUT_DIR:-out/}

mkdir -p "$WORKING_DIR"
mkdir -p "$OUT_DIR"

echo "Downloading DNS4 script"
wget -q \
  https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/download-dns-challenge-4.sh \
  -O "$WORKING_DIR"/dns4_blob.sh

# Delete all stuff starting from WORKING_DIR="./datasets_fullband"
sed -i -e '/^OUTPUT_PATH*/,$d' "$WORKING_DIR"/dns4_blob.sh

. "$WORKING_DIR"/dns4_blob.sh --source-only

for BLOB in "${BLOB_NAMES[@]}"; do
    URL="$AZURE_URL/$BLOB"
    echo "Download: $BLOB"
    curl -C - "$URL" | tar -C "$WORKING_DIR" -f - -x -j
    echo "Download finished"

    if [[ $BLOB == *"noise"* ]]; then
      CODEC="vorbis"
      TYPE="noise"
    elif [[ $BLOB == *"impulse_responses"* ]]; then
      CODEC="flac"
      TYPE="rir"
    else
      CODEC="flac"
      TYPE="speech"
    fi
    if [[ $BLOB == *"dev_testset"* ]]; then
      SPLIT="VALID"
    else
      SPLIT="TRAIN"
    fi
    fd -I . -e wav "$WORKING_DIR"/datasets_fullband/ > "$WORKING_DIR"/files.txt
    HDF5_NAME="$(basename "$(dirname "$(head -n 1 "$WORKING_DIR"/files.txt)")")_$SPLIT.hdf5"
    echo "Processing hdf5 dataset $OUT_DIR/$HDF5_NAME"
    python DeepFilterNet/df/scripts/prepare_data.py "$TYPE" "$WORKING_DIR"/files.txt "$OUT_DIR"/"$HDF5_NAME" --codec "$CODEC"
    rm -rf "$WORKING_DIR"/datasets_fullband/*
    rm "$WORKING_DIR"/files.txt
done
