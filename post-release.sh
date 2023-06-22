#!/bin/bash

set -e

MODIFIED=$(fd "(pyproject)|(Cargo)" -t f -e toml -X git diff --name-only {})
[[ -n "$MODIFIED" ]] && { echo "Project files are modified:" && echo "$MODIFIED" && exit 1; }

CUR_VERSION=$(sed -nr "/^\[package\]/ { :l /^version[ ]*=/ { s/.*=[ ]*//; p; q;}; n; b l;}" libDF/Cargo.toml | tr -d "\"")
# Increment last part
VERSION=$(echo "$CUR_VERSION" | awk -F. -v OFS=. 'NF==1{print ++$NF}; NF>1{$NF=sprintf("%0*d", length($NF), ($NF+1)); print}')-pre

set_version() {
  FILE=$1
  VERSION=$2
  sed -i "0,/^version/s/^version *= *\".*\"/version = \"$VERSION\"/" "$FILE"
}
export -f set_version

fd "(pyproject)|(Cargo)" -t f -e toml -x bash -c "set_version {} $VERSION"

(
  cd DeepFilterNet/
  # Workaround for 'poetry add ../pyDF/' which gives some obscure error message
  sed -i "s/^deepfilterlib.*/deepfilterlib = { path = \"..\/pyDF\/\" }/" pyproject.toml
  sed -i "s/^deepfilterdataloader.*/deepfilterdataloader = { path = \"..\/pyDF-data\/\", optional = true }/" pyproject.toml

)
echo cargo add --manifest-path ./pyDF/Cargo.toml --features transforms --path ./libDF
# cargo add --manifest-path ./pyDF/Cargo.toml --features transforms --path ./libDF deep_filter
# cargo add --manifest-path ./pyDF-data/Cargo.toml --features dataset --path ./libDF deep_filter

cargo update
(
  cd DeepFilterNet/
  echo "Running poetry update"
  poetry update
  echo "done"
  git add poetry.lock
)
fd "(pyproject)|(Cargo)" -I -t f -e toml -e lock -X git add {}

git commit -m "post-release v$VERSION"
