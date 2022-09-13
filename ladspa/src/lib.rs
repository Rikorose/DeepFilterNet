use std::collections::VecDeque;
use std::sync::Once;

use df::tract::*;
use ladspa::{Plugin, PluginDescriptor, Port, PortConnection, PortDescriptor};
use ndarray::prelude::*;

static INIT_LOGGER: Once = Once::new();

struct DfMono {
    df: DfTract,
    inbuf: Vec<f32>,
    outframe: Vec<f32>,
    outbuf: VecDeque<f32>,
}
struct DfStereo {
    df: DfTract,
    inbuf: Vec<Vec<f32>>,
}

const ID_MONO: u64 = 7843795;
const ID_STEREO: u64 = 7843796;

pub fn new_df_mono(_: &PluginDescriptor, sample_rate: u64) -> Box<dyn Plugin + Send> {
    INIT_LOGGER.call_once(|| {
        let _ = env_logger::builder().filter_level(log::LevelFilter::Info).init();
    });

    let models = DfModelParams::from_bytes(
        include_bytes!("../../models/DeepFilterNet2_onnx/enc.onnx"),
        include_bytes!("../../models/DeepFilterNet2_onnx/erb_dec.onnx"),
        include_bytes!("../../models/DeepFilterNet2_onnx/df_dec.onnx"),
    );
    let params = DfParams::with_bytes_config(
        include_bytes!("../../models/DeepFilterNet2_onnx/config.ini"),
        1,
        false,
        -15.,
        30.,
        30.,
    );
    let m = DfTract::new(&models, &params).expect("Could not initialize DeepFilter runtime.");
    assert_eq!(m.sr as u64, sample_rate, "Unsupported sample rate");
    let inbuf = Vec::with_capacity(m.hop_size);
    let outbuf = VecDeque::with_capacity(m.hop_size);
    let outframe = vec![0f32; m.hop_size];
    let plugin = DfMono {
        df: m,
        inbuf,
        outbuf,
        outframe,
    };
    log::info!("Initilized DeepFilter_Mono plugin.");
    Box::new(plugin)
}

pub fn new_df_stereo(_: &PluginDescriptor, sample_rate: u64) -> Box<dyn Plugin + Send> {
    let models = DfModelParams::from_bytes(
        include_bytes!("../../models/DeepFilterNet2_onnx/enc.onnx"),
        include_bytes!("../../models/DeepFilterNet2_onnx/erb_dec.onnx"),
        include_bytes!("../../models/DeepFilterNet2_onnx/df_dec.onnx"),
    );
    let params = DfParams::with_bytes_config(
        include_bytes!("../../models/DeepFilterNet2_onnx/config.ini"),
        2,
        false,
        -15.,
        30.,
        30.,
    );
    let m = DfTract::new(&models, &params).expect("Could not initialize DeepFilter runtime.");
    assert_eq!(m.sr as u64, sample_rate, "Unsupported sample rate");
    let inbuf = vec![Vec::new(); 2];
    let plugin = DfStereo { df: m, inbuf };
    Box::new(plugin)
}

impl Plugin for DfMono {
    fn activate(&mut self) {
        log::info!("DfMono::activate()");
    }
    fn deactivate(&mut self) {
        log::info!("DfMono::deactivate()");
    }
    fn run<'a>(&mut self, _sample_count: usize, ports: &[&'a PortConnection<'a>]) {
        let n = self.df.hop_size;
        let mut i_idx = 0;
        let mut o_idx = 0;

        let input = ports[0].unwrap_audio();
        let mut output = ports[1].unwrap_audio_mut();
        log::info!(
            "DfMono::run() with input {} and output {}",
            input.len(),
            output.len()
        );

        // 1. Pass alrady processed samples to the output buffer
        if output.len() <= self.outbuf.len() {
            o_idx = output.len().max(self.outbuf.len());
            for o in output.iter_mut().take(o_idx) {
                *o = self.outbuf.pop_front().unwrap();
            }
        }

        // 2. Check self.inbuf has some samples from previous run calls and fill up
        let missing = n.saturating_sub(self.inbuf.len());
        if !self.inbuf.is_empty() && missing > 0 {
            self.inbuf.extend_from_slice(&input[..missing]);
            debug_assert_eq!(self.inbuf.len(), n);
            i_idx = missing;
        }

        // Check if self.inbuf has enough samples and process
        debug_assert!(
            self.inbuf.len() <= n,
            "inbuf len should not exceed frame size."
        );
        if self.inbuf.len() == n {
            let mut used_outframe = false;
            let i_f = slice_as_arrayview(self.inbuf.as_slice(), &[1, n])
                .into_dimensionality::<Ix2>()
                .unwrap();
            let o_f = mut_slice_as_arrayviewmut(
                if output.len().saturating_sub(o_idx) >= n {
                    o_idx += n;
                    &mut output[o_idx - n..o_idx] // Directly write into output buffer
                } else {
                    used_outframe = true;
                    self.outframe.as_mut_slice()
                },
                &[1, n],
            )
            .into_dimensionality::<Ix2>()
            .unwrap();
            self.df.process(i_f, o_f).unwrap();
            if used_outframe {
                // Copy samples from self.outframe to output
                let n_copy = (output.len().saturating_sub(o_idx)).min(n);
                output[o_idx..o_idx + n_copy].copy_from_slice(&self.outframe[..n_copy]);
                if n_copy < n {
                    // Store remaining that did not fit into output in self.outbuf
                    for &x in self.outframe.iter().skip(n_copy) {
                        self.outbuf.push_back(x)
                    }
                }
                o_idx += n_copy;
            }
            self.inbuf.clear();
        }

        // Check if new input has enough samples and run
        if input.len() - i_idx >= n {
            let n_frames = (input.len() - i_idx) / (n);
            for i in 0..n_frames {
                let mut used_outframe = false;

                let i_f = slice_as_arrayview(&input[i_idx..i_idx + n], &[1, n])
                    .into_dimensionality::<Ix2>()
                    .unwrap();
                i_idx += n;
                let o_f = mut_slice_as_arrayviewmut(
                    if output.len().saturating_sub(o_idx) >= n {
                        o_idx += n;
                        &mut output[o_idx - n..o_idx] // Directly write into output buffer
                    } else {
                        used_outframe = true;
                        self.outframe.as_mut_slice()
                    },
                    &[1, n],
                )
                .into_dimensionality::<Ix2>()
                .unwrap();
                let i_rms = df::rms(i_f.iter());
                self.df.process(i_f, o_f).unwrap();
                let o_rms = df::rms(output[o_idx-n..o_idx].iter());
                if used_outframe {
                    // Copy samples from self.outframe to output
                    let n_copy = (output.len().saturating_sub(o_idx)).min(n);
                    output[o_idx..o_idx + n_copy].copy_from_slice(&self.outframe[..n_copy]);
                    if n_copy < n {
                        // Store remaining that did not fit into output in self.outbuf
                        for &x in self.outframe.iter().skip(n_copy) {
                            self.outbuf.push_back(x)
                        }
                    }
                    o_idx += n_copy;
                }
                log::info!(
                    "Processed frame {}, input rms: {}, output rms: {}",
                    i,
                    i_rms,
                    o_rms
                );
            }
        }

        // Check if input has remaining samples that have not been processed and save them for later
        if i_idx < input.len() {
            self.inbuf.extend_from_slice(&input[i_idx..]);
        }
    }
}
impl Plugin for DfStereo {
    fn activate(&mut self) {}
    fn run<'a>(&mut self, _sample_count: usize, ports: &[&'a PortConnection<'a>]) {
        let input_l = ports[0].unwrap_audio();
        let input_r = ports[0].unwrap_audio();
        let mut output = ports[1].unwrap_audio_mut();
        // let input = slice_as_arrayview(input, &[1, input.len()])
        //     .into_dimensionality::<Ix2>()
        //     .unwrap();
        // let mut output = mut_slice_as_arrayviewmut(&mut output, &[1, input.len()])
        //     .into_dimensionality()
        //     .unwrap();
        // for (i_f, o_f) in input
        //     .axis_chunks_iter(Axis(1), self.df.hop_size)
        //     .zip(output.axis_chunks_iter_mut(Axis(1), self.df.hop_size))
        // {
        //     self.df.process(i_f, o_f).unwrap();
        // }
    }
}

#[no_mangle]
pub fn get_ladspa_descriptor(index: u64) -> Option<PluginDescriptor> {
    match index {
        0 => Some(PluginDescriptor {
            unique_id: ID_MONO,
            label: "deep_filter_mono",
            properties: ladspa::PROP_NONE,
            name: "DeepFilter_Mono",
            maker: "Hendrik Schröter",
            copyright: "MIT/Apache",
            ports: vec![
                Port {
                    name: "Audio In",
                    desc: PortDescriptor::AudioInput,
                    ..Default::default()
                },
                Port {
                    name: "Audio Out",
                    desc: PortDescriptor::AudioOutput,
                    ..Default::default()
                },
            ],
            new: new_df_mono,
        }),
        1 => Some(PluginDescriptor {
            unique_id: ID_STEREO,
            label: "deep_filter_stereo",
            properties: ladspa::PROP_NONE,
            name: "DeepFilter_Stereo",
            maker: "Hendrik Schröter",
            copyright: "MIT/Apache",
            ports: vec![
                Port {
                    name: "Audio In L",
                    desc: PortDescriptor::AudioInput,
                    ..Default::default()
                },
                Port {
                    name: "Audio In R",
                    desc: PortDescriptor::AudioInput,
                    ..Default::default()
                },
                Port {
                    name: "Audio Out L",
                    desc: PortDescriptor::AudioOutput,
                    ..Default::default()
                },
                Port {
                    name: "Audio Out R",
                    desc: PortDescriptor::AudioOutput,
                    ..Default::default()
                },
            ],
            new: new_df_stereo,
        }),
        _ => None,
    }
}
