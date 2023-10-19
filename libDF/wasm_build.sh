#!/bin/sh

set -ex

wasm-pack build --target no-modules --features wasm
# zip -r df3_wasm.zip pkg