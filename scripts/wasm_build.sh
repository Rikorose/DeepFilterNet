#!/bin/sh

set -ex

cd ../libDF/
wasm-pack build --target no-modules --features wasm