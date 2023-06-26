# DeepFilterNet LADSPA Plugin

The LADSPA plugin uses a new model without any lookahead. The minimum latency is 20 ms (STFT processing) and additional latency depending on your LADSPA host such as Pipewire.

## Installation

You can download a release build (look for `libdeep_filter_ladspa`) from [here](https://github.com/Rikorose/DeepFilterNet/releases).

Or you can manually build the plugin via:

```bash
cargo build --release -p deep-filter-ladspa
ls target/release/libdeep_filter_ladspa* # here should be the compiled plugin
```

## Pipewire filter-chain source

You can use Pipewire to create a virtual microphone that suppresses noise. You may use it for any
type of application that accesses your microphone, such as VoIP applications like Zoom or Discord.

Follow the instructions in the [configuration file](filter-chain-configs/deepfilter-mono-source.conf).

To run the filter-chain module use:

```bash
pipewire -c filter-chain.conf # <- After setting up your filter chain config!
```
To debug you may increase the log level via:
```bash
RUST_LOG=DEBUG pipewire -c filter-chain.conf
```

## Pipewire filter-chain sink

You can use Pipewire to create a virtual output that suppresses noise comming from an application.
You may use it for any type of application that produces audio such as a browser, video player, VoIP
application. The output will only contain speech and suppress most of the noise.

Follow the instructions in the [configuration file](filter-chain-configs/deepfilter-stereo-sink.conf).

More information about Pipewire filter chain can be found in the [Pipewire wiki](https://gitlab.freedesktop.org/pipewire/pipewire/-/wikis/Filter-Chain).


## D-Bus
