#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
  echo "Usage: release.sh <new-version>"
  exit 1
fi

VERSION=$1
CUR_VERSION=$(sed -nr "/^\[package\]/ { :l /^version[ ]*=/ { s/.*=[ ]*//; p; q;}; n; b l;}" libDF/Cargo.toml | tr -d "\"")

verle() {
  [  "$1" = $(echo -e "$1\n$2" | sort -V | head -n1) ]
}

verlt() {
    [ "$1" = "$2" ] && return 1 || verle $1 $2
}

if verle $VERSION $CUR_VERSION; then
  echo "New version ($VERSION) needs to be greater then current version ($CUR_VERSION)"
  exit 2
fi

echo "Setting new version $VERSION"

set_version() {
  FILE=$1
  VERSION=$2
  sed -i "0,/^version/s/^version *= *\".*\"/version = \"$VERSION\"/" $FILE
}
export -f set_version

fd "(pyproject)|(Cargo)" -t f -e toml -x bash -c "set_version {} $VERSION"

fd "(pyproject)|(Cargo)" -t f -e toml -X git add {}

git commit -m "v$VERSION"
git push
git tag -f "v$VERSION"
git push -f --tags

cargo update
