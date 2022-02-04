#!/bin/bash

set -e

WORKING_DIR=${WORKING_DIR:-out/}
OUT_DIR=${OUT_DIR:-out/}

mkdir -p "$WORKING_DIR"
mkdir -p "$OUT_DIR"

if ! [[ -f "$WORKING_DIR"/dns4_blob.sh ]]; then
  echo "Downloading DNS4 script"
  wget -q \
    https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/download-dns-challenge-4.sh \
    -O "$WORKING_DIR"/dns4_blob.sh
  # Delete all stuff starting from WORKING_DIR="./datasets_fullband"
  sed -i -e '/^OUTPUT_PATH*/,$d' "$WORKING_DIR"/dns4_blob.sh
fi

. "$WORKING_DIR"/dns4_blob.sh --source-only

for BLOB in "${BLOB_NAMES[@]}"; do
    URL="$AZURE_URL/$BLOB"
    echo "Download: $BLOB"
    mkdir -p $(basename "$WORKING_DIR/$BLOB")
    if command -v aria2c &> /dev/null; then
      aria2c --dir "$WORKING_DIR" --log-level=warn "$URL"
      BLOB=$(basename $BLOB) # aria2c removes any folders
    else
      wget --continue "$URL" -O "$WORKING_DIR/$BLOB"
    fi
    echo "Download finished. Extracting..."
    tar -C "$WORKING_DIR" -f "$WORKING_DIR/$BLOB" -x -j
    echo "extraction finished"

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
    if [ -d "$WORKING_DIR"/datasets_fullband/noise_fullband/ ]; then
      CURDIRS=$(fd -t d -d 1 . "$WORKING_DIR"/datasets_fullband/)
    else
      CURDIRS=$(fd -t d -d 1 . "$WORKING_DIR"/datasets_fullband/*)
    fi
    echo "Found dirs: $CURDIRS"
    for d in "$CURDIRS"; do
      echo "dir: $d"
      fd -I . -e wav "$d" > "$WORKING_DIR"/files.txt
      HDF5_NAME="$(basename $d)_$SPLIT.hdf5"
      echo "Processing hdf5 dataset $OUT_DIR/$HDF5_NAME"
      python DeepFilterNet/df/scripts/prepare_data.py "$TYPE" "$WORKING_DIR"/files.txt "$OUT_DIR"/"$HDF5_NAME" --codec "$CODEC"
      rm "$WORKING_DIR"/files.txt
      # Delete dataset name from blob list
      BLOB_ID=${BLOB%.tar.bz2}  # Remove extension
      BLOB_ID=${BLOB_ID##*.}  # Select part after last dot
      # Sed: Start from begining, find first matching id, delete it, stop
      echo sed -i -e '0,/*'"$BLOB_ID"'*/{/'"$BLOB_ID"'/d;}' "$WORKING_DIR"/dns4_blob.sh
      sed -i '0,/*'"$BLOB_ID"'*/{/'"$BLOB_ID"'/d;}' "$WORKING_DIR"/dns4_blob.sh
    done
    # Cleanup
    echo "Deleting $WORKING_DIR"/datasets_fullband/*
    rm -rf "$WORKING_DIR"/datasets_fullband/*
    echo "Deleting $WORKING_DIR/$BLOB"
    rm "$WORKING_DIR/$BLOB"
done
