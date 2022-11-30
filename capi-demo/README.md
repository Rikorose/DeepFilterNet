# Compile and install libdf

```bash
cargo install cbindgen cargo-c
cargo cinstall --destdir staging --prefix=usr --libdir=/usr/lib64 -p deep_filter
cbindgen --config cbindgen.toml --crate deep_filter --output staging/usr/include/df/deep_filter.h
sudo cp -a staging/* /
```

# Compile demo

```bash
cd capi-demo
meson build
meson compile -C build
```

# Run demo

```bash
sox ../assets/noisy_snr0.wav -e signed-integer -b 16 noisy_snr0.raw
./build/demo noisy_snr0.raw enh.raw
sox -r 48k -b 16 -e signed-integer enh.raw enh.wav
```
