use std::collections::VecDeque;
use std::sync::Once;

use df::tract::*;
use ladspa::{Plugin, PluginDescriptor, Port, PortConnection, PortDescriptor};
use ndarray::prelude::*;

static INIT_LOGGER: Once = Once::new();

struct DfMono {
    df: DfTract,
    inframe: Vec<f32>,
    outframe: Vec<f32>,
    outbuf: VecDeque<f32>,
}
struct DfStereo {
    df: DfTract,
    inframe: StereoBuffer,
    outframe: StereoBuffer,
    outbuf: [VecDeque<f32>; 2],
}

const ID_MONO: u64 = 7843795;
const ID_STEREO: u64 = 7843796;

pub fn new_df_mono(_: &PluginDescriptor, sample_rate: u64) -> Box<dyn Plugin + Send> {
    INIT_LOGGER.call_once(|| {
        env_logger::builder().filter_level(log::LevelFilter::Info).init();
    });

    let df_params = DfParams::from_bytes(include_bytes!("../../models/DeepFilterNet2_onnx.tar.gz"))
        .expect("Could not load model tar.");
    let r_params = RuntimeParams::new(1, false, -15., 30., 30., ReduceMask::MEAN);
    let m = DfTract::new(df_params, &r_params).expect("Could not initialize DeepFilter runtime.");
    assert_eq!(m.sr as u64, sample_rate, "Unsupported sample rate");
    let inframe = Vec::with_capacity(m.hop_size);
    let outbuf = VecDeque::with_capacity(m.hop_size);
    let outframe = vec![0f32; m.hop_size];
    let plugin = DfMono {
        df: m,
        inframe,
        outframe,
        outbuf,
    };
    log::info!("Initialized DeepFilter Mono plugin");
    Box::new(plugin)
}

pub fn new_df_stereo(_: &PluginDescriptor, sample_rate: u64) -> Box<dyn Plugin + Send> {
    INIT_LOGGER.call_once(|| {
        env_logger::builder().filter_level(log::LevelFilter::Info).init();
    });

    let df_params = DfParams::from_bytes(include_bytes!("../../models/DeepFilterNet2_onnx.tar.gz"))
        .expect("Could not load model tar.");
    let r_params = RuntimeParams::new(2, false, -15., 30., 30., ReduceMask::MEAN);
    let m = DfTract::new(df_params, &r_params).expect("Could not initialize DeepFilter runtime.");
    assert_eq!(m.sr as u64, sample_rate, "Unsupported sample rate");
    let inframe = StereoBuffer::with_frame_size(m.hop_size);
    let mut outframe = StereoBuffer::with_frame_size(m.hop_size);
    outframe.as_uninit();
    let outbuf = [
        VecDeque::with_capacity(m.hop_size),
        VecDeque::with_capacity(m.hop_size),
    ];
    let plugin = DfStereo {
        df: m,
        inframe,
        outframe,
        outbuf,
    };
    log::info!("Initialized DeepFilter Stereo plugin");
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

        // Pass alrady processed samples to the output buffer
        if output.len() <= self.outbuf.len() {
            o_idx = output.len().max(self.outbuf.len());
            for o in output.iter_mut().take(o_idx) {
                *o = self.outbuf.pop_front().unwrap();
            }
        }

        // Check self.inbuf has some samples from previous run calls and fill up
        let missing = n.saturating_sub(self.inframe.len());
        if !self.inframe.is_empty() && missing > 0 {
            self.inframe.extend_from_slice(&input[..missing]);
            debug_assert_eq!(self.inframe.len(), n);
            i_idx = missing;
        }

        // Check if self.inbuf has enough samples and process
        debug_assert!(
            self.inframe.len() <= n,
            "inbuf len should not exceed frame size."
        );
        if self.inframe.len() == n {
            let mut used_outframe = false;
            let i_f = slice_as_arrayview(self.inframe.as_slice(), &[1, n])
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
                if n_copy > 0 {
                    output[o_idx..o_idx + n_copy].copy_from_slice(&self.outframe[..n_copy]);
                }
                if n_copy < n {
                    // Store remaining that did not fit into output in self.outbuf
                    for &x in self.outframe.iter().skip(n_copy) {
                        self.outbuf.push_back(x)
                    }
                }
                o_idx += n_copy;
            }
            self.inframe.clear();
        }

        // Check if new input has enough samples and run
        if input.len() - i_idx >= n {
            let n_frames = (input.len() - i_idx) / (n);
            for _ in 0..n_frames {
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
                self.df.process(i_f, o_f).unwrap();
                if used_outframe {
                    // Copy samples from self.outframe to output
                    let n_copy = (output.len().saturating_sub(o_idx)).min(n);
                    if n_copy > 0 {
                        output[o_idx..o_idx + n_copy].copy_from_slice(&self.outframe[..n_copy]);
                    }
                    if n_copy < n {
                        // Store remaining that did not fit into output in self.outbuf
                        for &x in self.outframe.iter().skip(n_copy) {
                            self.outbuf.push_back(x)
                        }
                    }
                    o_idx += n_copy;
                }
            }
        }

        // Check if input has remaining samples that have not been processed and save them for later
        if i_idx < input.len() {
            self.inframe.extend_from_slice(&input[i_idx..]);
        }
    }
}
impl Plugin for DfStereo {
    fn activate(&mut self) {
        log::info!("DfStereo::activate()");
    }
    fn deactivate(&mut self) {
        log::info!("DfStereo::deactivate()");
    }
    fn run<'a>(&mut self, _sample_count: usize, ports: &[&'a PortConnection<'a>]) {
        let n = self.df.hop_size;
        let mut i_idx = 0;
        let mut o_idx = 0;

        let input_l = ports[0].unwrap_audio();
        let input_r = ports[1].unwrap_audio();
        let mut output_l = ports[2].unwrap_audio_mut();
        let mut output_r = ports[3].unwrap_audio_mut();
        log::info!(
            "DfStereo::run() with input {} and output {}",
            input_l.len(),
            output_l.len()
        );

        // Pass alrady processed samples to the output buffer
        for (o_ch, b_ch) in [&mut output_l, &mut output_r].iter_mut().zip(self.outbuf.iter_mut()) {
            if o_ch.len() <= b_ch.len() {
                o_idx = o_ch.len().max(b_ch.len());
                for o in o_ch.iter_mut().take(o_idx) {
                    *o = b_ch.pop_front().unwrap();
                }
            }
        }

        // Check self.inbuf has some samples from previous run calls and fill up
        loop {
            let missing = n.saturating_sub(self.inframe.len()).min(input_l.len() - i_idx);
            if missing > 0 {
                self.inframe.extend(
                    &input_l[i_idx..i_idx + missing],
                    &input_r[i_idx..i_idx + missing],
                );
                i_idx += missing;
            }

            // Check if self.inbuf has enough samples and process
            debug_assert!(
                self.inframe.len() <= n,
                "inbuf len should not exceed frame size."
            );
            if self.inframe.len() < n {
                break;
            }

            let i_f = self.inframe.as_arrayview();
            let o_f = self.outframe.as_arrayviewmut();
            self.df.process(i_f, o_f).unwrap();
            // Copy samples from self.outframe to output
            let n_copy = (output_l.len().saturating_sub(o_idx)).min(n);
            for (o_ch, of_ch, ob_ch) in df::zip3(
                [&mut output_l, &mut output_r].iter_mut(),
                self.outframe.channels().into_iter(),
                self.outbuf.iter_mut(),
            ) {
                if n_copy > 0 {
                    o_ch[o_idx..o_idx + n_copy].copy_from_slice(&of_ch[..n_copy]);
                }
                if n_copy < n {
                    // Store remaining that did not fit into output in self.outbuf
                    for &x in of_ch.iter().skip(n_copy) {
                        ob_ch.push_back(x)
                    }
                }
            }
            o_idx += n_copy;
            self.inframe.clear();
        }

        // Check if input has remaining samples that have not been processed and save them for later
        if i_idx < input_l.len() {
            self.inframe.extend(&input_l[i_idx..], &input_r[i_idx..]);
        }
    }
}

struct StereoBuffer {
    b: Vec<f32>, // buffer
    idx: usize,  // current idx
    c: usize,    // capacity
}
impl StereoBuffer {
    pub fn with_frame_size(n: usize) -> Self {
        Self {
            b: vec![0.; n * 2],
            idx: 0,
            c: n,
        }
    }
    #[inline]
    pub fn capacity(&self) -> usize {
        self.c
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.idx
    }
    // #[inline]
    // pub fn is_empty(&self) -> bool {
    //     self.idx == 0
    // }
    #[inline]
    pub fn clear(&mut self) {
        self.idx = 0;
    }
    pub fn extend(&mut self, left: &[f32], right: &[f32]) {
        debug_assert_eq!(
            left.len(),
            right.len(),
            "Left and right channels have different amount of samples."
        );
        debug_assert!(
            left.len() + self.len() <= self.capacity(),
            "New samples exceed capacity."
        );
        let n = left.len();
        self.b[self.idx..self.idx + n].clone_from_slice(left);
        self.b[self.c + self.idx..self.c + self.idx + n].clone_from_slice(right);
        self.idx += n;
    }
    // #[inline]
    // pub fn frames_left(&self) -> &[f32] {
    //     &self.b[..self.idx]
    // }
    // #[inline]
    // pub fn frames_right(&self) -> &[f32] {
    //     &self.b[self.c..self.c + self.idx]
    // }
    pub fn channels(&self) -> [&[f32]; 2] {
        let (left, right) = self.b.split_at(self.c);
        [&left[..self.idx], &right[..self.idx]]
    }
    #[inline]
    pub fn as_arrayview(&self) -> ArrayView2<f32> {
        debug_assert!(self.idx > 0);
        slice_as_arrayview(&self.b, &[2, self.idx]).into_dimensionality().unwrap()
    }
    #[inline]
    pub fn as_arrayviewmut(&mut self) -> ArrayViewMut2<f32> {
        debug_assert!(self.idx > 0);
        mut_slice_as_arrayviewmut(&mut self.b, &[2, self.idx])
            .into_dimensionality()
            .unwrap()
    }
    /// Sets the len of buffer to its capacity and provides and provides access to uninitialized
    /// data.
    pub fn as_uninit(&mut self) {
        self.idx = self.c
    }
}

#[no_mangle]
pub fn get_ladspa_descriptor(index: u64) -> Option<PluginDescriptor> {
    match index {
        0 => Some(PluginDescriptor {
            unique_id: ID_MONO,
            label: "deep_filter_mono",
            properties: ladspa::PROP_NONE,
            name: "DeepFilter Mono",
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
            name: "DeepFilter Stereo",
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
