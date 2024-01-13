```bash
# If you are on ubuntu, install the requirements:
sudo apt -y install build-essential cmake libfontconfig1-dev libasound2-dev
# Setup rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Run demo
cargo +nightly run -p df-demo --features ui --bin df-demo --release
```
