#!/bin/bash

DATADIR="$1"
DATACFG="$2"
TARGETDIR="$3"
MAX_GB=${MAX_GB:-100} # rest will be linked

USAGE="usage: prepare_datadir.sh <DATADIR> <DATACFG> <TARGETDIR>"

if ! [[ -d "$DATADIR" ]]; then
  echo "$USAGE"
  exit 1
fi

if ! [[ -f "$DATACFG" ]]; then
  echo "$USAGE"
  exit 2
fi

if ! [[ -d "$TARGETDIR" ]]; then
  mkdir "$TARGETDIR"
fi

# Copy or link hdf5 file
gb_copied=0
function copy_or_link() {
  HDF5="$1"
  if ! [[ -e "$HDF5" ]]; then
    echo "Dataset $HDF5 not found"
    return
  fi
  s=$(ls -lH --block-size=G "$HDF5" | cut -f5 -d ' ' | tr -d G)
  new_gb=$(($gb_copied + $s))
  if [ "$new_gb" -gt "$MAX_GB" ]; then
    echo "linking: $HDF5"
    ln -Ls "$HDF5" "$TARGETDIR"
  else
    if ! rsync -aL --info=name,progress2 "$HDF5" "$TARGETDIR" | stdbuf -oL tr '\r' '\n'; then
      # Most likely file system is full
      echo
      echo "copy failed, linking: $HDF5"
      ln -Ls "$HDF5" "$TARGETDIR"
    else
      gb_copied=$(($gb_copied + $s))
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
done <<<"$(jq -c '.train[]' "$DATACFG")"
sorted=($(printf '%s\n' "${hdf5s[@]}" | sort -n -k 2 -r))

for HDF5 in "${sorted[@]}"; do
  # Every second element is the sampling factor.
  # Thus, check if $HDF5 is a file
  if [[ -e "$DATADIR/$HDF5" ]]; then
    if ! [[ -e "$TARGETDIR"/$(basename "$HDF5") ]]; then
      copy_or_link "$DATADIR/$HDF5"
    fi
  fi
done

# Copy or link the rest
while read -r row; do
  HDF5=$(echo "$row" | jq '.[0]' | tr -d '\"')
  if ! [[ -e "$TARGETDIR"/$(basename "$HDF5") ]]; then
    copy_or_link "$DATADIR/$HDF5"
  fi
done <<<"$(jq -c '.valid, .test | .[]' "$DATACFG")"
