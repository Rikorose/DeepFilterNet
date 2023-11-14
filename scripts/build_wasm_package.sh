#!/bin/sh -ex

# look at DeepFilterNet/.github/workflows/build_wasm.yml for enviroment setup
cd ./libDF/
wasm-pack build --target no-modules --features wasm