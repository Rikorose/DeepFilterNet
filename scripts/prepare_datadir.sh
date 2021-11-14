#!/bin/bash

DATADIR="$1"
DATACFG="$2"
TARGETDIR="$3"
MAX_GB=${MAX_GB:-500}  # rest will be linked

USAGE="usage: prepare_datadir.sh <DATADIR> <DATACFG> <TARGETDIR>"

if ! [[ -d "$DATADIR" ]]; then
  echo "$USAGE"
  exit 1
fi

if ! [[ -f "$DATACFG" ]]; then
  echo "$USAGE"
  exit 2
fi

if [[ -d "$TARGETDIR" ]]; then
  if [ $(ls "$TARGETDIR" | wc -l) -gt 0 ]; then
    echo "TARGETDIR must be empty"
    echo "$USAGE"
    exit 3
  fi
else
  mkdir "$TARGETDIR"
fi

# Copy or link hdf5 file
gb_copied=0
function copy_or_link() {
  HDF5="$1"
  s=$(ls -lH --block-size=G "$HDF5" | cut -f5 -d ' ' | tr -d G)
  gb_copied=$(($gb_copied+$s))
  if [ "$gb_copied" -gt "$MAX_GB" ]; then
    echo "$HDF5"
    ln -Ls "$HDF5" "$TARGETDIR"
  else
    if ! rsync -aPL "$HDF5" "$TARGETDIR"; then
      # Most likely file system is full
      HDF5_N=$(basename "$HDF5")
      rm "$TARGETDIR/$HDF5_N"
      echo "$HDF5"
      ln -Ls "$HDF5" "$TARGETDIR"
    fi
  fi
}

# First try to copy train datasets sorted by the sampling factor
hdf5s=()
while read -r row; do
  file=$(echo "$row" | jq '.[0]' | tr -d '\"')
  factor=$(echo "$row" | jq '.[1]')
  row="$file $factor"
  hdf5s+=("$row")
done <<< "$(jq -c '.train[]' "$DATACFG")"
sorted=($(printf '%s\n' "${hdf5s[@]}" | sort -k 2 -r))

for HDF5 in "${hdf5s[@]}"; do
  # HDF5 contains the sampling factor
  HDF5=$(echo "$HDF5" | cut -f1 -d " ")
  copy_or_link "$DATADIR/$HDF5"
  break
done

# Copy or link the rest
for HDF5 in "$DATADIR"/*hdf5; do
  if ! [[ -e "$TARGETDIR"/$(basename "$HDF5") ]]; then
    copy_or_link "$HDF5"
  fi
done
