use std::env;
use std::io::{self, stdout, Write};
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Once,
};
use std::thread::{self, sleep, JoinHandle};
use std::time::Duration;

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Data, Device, Stream, StreamConfig};
use crossbeam_channel::{unbounded, Receiver, Sender};
use df::{tract::*, Complex32};
use ndarray::prelude::*;
use ringbuf::{Consumer, HeapRb, Producer, SharedRb};

pub type RbProd = Producer<f32, Arc<SharedRb<f32, Vec<MaybeUninit<f32>>>>>;
pub type RbCons = Consumer<f32, Arc<SharedRb<f32, Vec<MaybeUninit<f32>>>>>;
pub type SendLsnr = Sender<f32>;
pub type RecvLsnr = Receiver<f32>;
pub type SendSpec = Sender<Box<[f32]>>;
pub type RecvSpec = Receiver<Box<[f32]>>;
pub type SendControl = Sender<(DfControl, f32)>;
pub type RecvControl = Receiver<(DfControl, f32)>;

static mut MODEL: Option<DfTract> = None;
pub(crate) static INIT_LOGGER: Once = Once::new();

const SAMPLE_FORMAT: cpal::SampleFormat = cpal::SampleFormat::F32;

pub struct AudioSink {
    stream: Option<Stream>,
    config: StreamConfig,
    device: Device,
}
pub struct AudioSource {
    stream: Option<Stream>,
    config: StreamConfig,
    device: Device,
}

fn get_cpal_stream_cfg(channels: u16, sample_rate: u32) -> cpal::StreamConfig {
    let n = unsafe { get_frame_size() } as u32 * 2;
    let buffer_size = cpal::BufferSize::Fixed(n);
    cpal::StreamConfig {
        channels,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size,
    }
}

#[derive(PartialEq)]
pub enum DfControl {
    AttenLim,
    MinThreshDb,
    MaxErbThreshDb,
    MaxDfThreshDb,
}

/// Initialize DF model and returns sample rate, frame size, and number of frequency bins
fn init_df(model_path: Option<PathBuf>, channels: usize) -> (usize, usize, usize) {
    unsafe {
        if let Some(m) = MODEL.as_ref() {
            if m.ch == channels {
                return (m.sr, m.hop_size, m.n_freqs);
            }
        }
    }
    // let df_params = DfParams::default();
    let df_params = if let Some(path) = model_path {
        DfParams::new(path).expect("Failed to read DF model")
    } else {
        DfParams::default()
    };
    let r_params = RuntimeParams::new(channels, false, 100., -15., 35., 35., ReduceMask::MEAN);
    let df = DfTract::new(df_params, &r_params).expect("Could not initialize DeepFilter runtime");
    let (sr, frame_size, freq_size) = (df.sr, df.hop_size, df.n_freqs);
    unsafe { MODEL = Some(df) };
    (sr, frame_size, freq_size)
}

unsafe fn get_frame_size() -> usize {
    let df = MODEL.clone().unwrap();
    df.hop_size
}

impl AudioSink {
    fn new(channels: u16, sample_rate: u32, device_str: Option<String>) -> Result<Self> {
        let host = cpal::default_host();
        let config = get_cpal_stream_cfg(channels, sample_rate);
        let mut device = host.default_output_device().expect("no output device available");
        if let Some(device_str) = device_str {
            for avail_dev in host.output_devices()? {
                if avail_dev.name()?.to_lowercase().contains(&device_str.to_lowercase()) {
                    device = avail_dev
                }
            }
        }

        Ok(Self {
            stream: None,
            config,
            device,
        })
    }
    fn start(&mut self, mut rb: RbCons) -> Result<()> {
        let stream = self.device.build_output_stream_raw(
            &self.config,
            SAMPLE_FORMAT,
            move |data: &mut Data, _: &cpal::OutputCallbackInfo| {
                let len = data.len();
                let data = data.as_slice_mut().expect("Failed to caputre audio");
                let mut n = 0;
                while n < len {
                    n += rb.pop_slice(&mut data[n..]);
                }
                debug_assert_eq!(n, data.len());
            },
            move |err| log::error!("Error during audio output {:?}", err),
            None, // None=blocking, Some(Duration)=timeout
        )?;
        stream.play()?;
        log::info!("Starting playback stream on device {}", self.device.name()?);
        self.stream = Some(stream);
        Ok(())
    }
    fn pause(&mut self) -> Result<()> {
        if let Some(s) = self.stream.as_mut() {
            s.pause()?;
        }
        Ok(())
    }
}

impl AudioSource {
    fn new(channels: u16, sample_rate: u32, device_str: Option<String>) -> Result<Self> {
        let host = cpal::default_host();
        let config = get_cpal_stream_cfg(channels, sample_rate);
        let mut device = host.default_input_device().expect("no output device available");
        if let Some(device_str) = device_str {
            for avail_dev in host.input_devices()? {
                if avail_dev.name()?.to_lowercase().contains(&device_str.to_lowercase()) {
                    device = avail_dev
                }
            }
        }

        Ok(Self {
            stream: None,
            config,
            device,
        })
    }
    fn start(&mut self, mut rb: RbProd) -> Result<()> {
        let stream = self.device.build_input_stream_raw(
            &self.config,
            SAMPLE_FORMAT,
            move |data: &Data, _: &cpal::InputCallbackInfo| {
                let len = data.len();
                let data = data.as_slice().expect("Failed to caputre audio");
                let mut n = 0;
                while n < len {
                    n += rb.push_slice(&data[n..]);
                }
                debug_assert_eq!(n, data.len());
            },
            move |err| log::error!("Error during audio output {:?}", err),
            None, // None=blocking, Some(Duration)=timeout
        )?;
        log::info!("Starting caputre stream on device {}", self.device.name()?);
        stream.play()?;
        self.stream = Some(stream);
        Ok(())
    }
    fn pause(&mut self) -> Result<()> {
        if let Some(s) = self.stream.as_mut() {
            s.pause()?;
        }
        Ok(())
    }
}

fn get_worker_fn(
    mut rb_in: RbCons,
    mut rb_out: RbProd,
    has_init: Arc<AtomicBool>,
    should_stop: Arc<AtomicBool>,
    mut s_lsnr: Option<SendLsnr>,
    mut s_spec: Option<(SendSpec, SendSpec)>,
    mut r_opt: Option<RecvControl>,
) -> impl FnMut() {
    move || {
        let mut df = unsafe { MODEL.clone().unwrap() };
        let mut inframe = Array2::zeros((df.ch, df.hop_size));
        let mut outframe = inframe.clone();
        df.process(inframe.view(), outframe.view_mut())
            .expect("Failed to run DeepFilterNet");
        has_init.store(true, Ordering::Relaxed);
        log::info!("Wokrer init");
        while !should_stop.load(Ordering::Relaxed) {
            if rb_in.len() < df.hop_size {
                // Sleep for half a hop size
                sleep(Duration::from_secs_f32(
                    df.hop_size as f32 / df.sr as f32 / 2.,
                ));
                continue;
            }
            let n = rb_in.pop_slice(inframe.as_slice_mut().unwrap());
            debug_assert_eq!(n, df.hop_size * df.ch);
            let lsnr = df
                .process(inframe.view(), outframe.view_mut())
                .expect("Failed to run DeepFilterNet");
            rb_out.push_slice(outframe.as_slice().unwrap());
            if let Some(ref mut s_lsnr) = s_lsnr.as_mut() {
                s_lsnr.send(lsnr).expect("Failed to send to LSNR rb");
            }
            if let Some((ref mut s_noisy, ref mut s_enh)) = s_spec.as_mut() {
                push_spec(df.get_spec_noisy(), s_noisy);
                push_spec(df.get_spec_enh(), s_enh);
            }
            if let Some(ref mut r_opt) = r_opt.as_mut() {
                while let Ok((c, v)) = r_opt.try_recv() {
                    match c {
                        DfControl::AttenLim => {
                            df.set_atten_lim(v).expect("Failed to set attenuation limit.")
                        }
                        DfControl::MinThreshDb => df.min_db_thresh = v,
                        DfControl::MaxErbThreshDb => df.max_db_erb_thresh = v,
                        DfControl::MaxDfThreshDb => df.max_db_df_thresh = v,
                    }
                }
            }
        }
    }
}

fn push_spec(spec: ArrayView2<Complex32>, sender: &mut SendSpec) {
    debug_assert_eq!(spec.len_of(Axis(0)), 1); // only single channel for now
    let out = spec.iter().map(|x| x.norm_sqr().max(1e-10).log10() * 10.).collect::<Vec<f32>>();
    sender.send(out.into_boxed_slice()).expect("Failed to send spectrogram")
}

pub fn log_format(buf: &mut env_logger::fmt::Formatter, record: &log::Record) -> io::Result<()> {
    let ts = buf.timestamp_millis();
    let module = record.module_path().unwrap_or("").to_string();
    let level_style = buf.default_level_style(log::Level::Info);

    writeln!(
        buf,
        "{} | {} | {} {}",
        ts,
        level_style.value(record.level()),
        module,
        record.args()
    )
}

pub struct DeepFilterCapture {
    pub sr: usize,
    pub frame_size: usize,
    pub freq_size: usize,
    should_stop: Arc<AtomicBool>,
    worker_handle: Option<JoinHandle<()>>,
    source: AudioSource,
    sink: AudioSink,
}

impl Default for DeepFilterCapture {
    fn default() -> Self {
        DeepFilterCapture::new(None, None, None, None, None)
            .expect("Error during DeepFilterCapture initialization")
    }
}
impl DeepFilterCapture {
    pub fn new(
        model_path: Option<PathBuf>,
        s_lsnr: Option<SendLsnr>,
        s_noisy: Option<SendSpec>,
        s_enh: Option<SendSpec>,
        r_contr: Option<RecvControl>,
    ) -> Result<Self> {
        let ch = 1;
        let (sr, frame_size, freq_size) = init_df(model_path, ch);
        let in_rb = HeapRb::<f32>::new(frame_size * 100);
        let out_rb = HeapRb::<f32>::new(frame_size * 100);
        let (in_prod, in_cons) = in_rb.split();
        let (out_prod, out_cons) = out_rb.split();

        let mut source = AudioSource::new(ch as u16, sr as u32, None)?;
        let mut sink = AudioSink::new(ch as u16, sr as u32, None)?;
        let should_stop = Arc::new(AtomicBool::new(false));
        let has_init = Arc::new(AtomicBool::new(false));
        let rb_spec = match (s_noisy, s_enh) {
            (Some(n), Some(e)) => Some((n, e)),
            _ => None,
        };
        let worker_handle = Some(thread::spawn(get_worker_fn(
            in_cons,
            out_prod,
            has_init.clone(),
            should_stop.clone(),
            s_lsnr,
            rb_spec,
            r_contr,
        )));
        while !has_init.load(Ordering::Relaxed) {
            sleep(Duration::from_secs_f32(0.01));
        }
        log::info!("DeepFilter Capture init");
        source.start(in_prod)?;
        sink.start(out_cons)?;

        Ok(Self {
            sr,
            frame_size,
            freq_size,
            should_stop,
            worker_handle,
            source,
            sink,
        })
    }

    pub fn should_stop(&mut self) -> Result<()> {
        self.sink.pause()?;
        self.source.pause()?;
        if let Some(h) = self.worker_handle.take() {
            log::info!("Joining DF Worker");
            self.should_stop.swap(true, Ordering::Relaxed);
            h.join().expect("Error during DF worker join");
        }
        Ok(())
    }
}

#[allow(unused)]
pub fn main() -> Result<()> {
    INIT_LOGGER.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
            .filter_module("tract_onnx", log::LevelFilter::Error)
            .filter_module("tract_core", log::LevelFilter::Error)
            .filter_module("tract_hir", log::LevelFilter::Error)
            .filter_module("tract_linalg", log::LevelFilter::Error)
            .format(log_format)
            .init();
    });

    let (lsnr_prod, mut lsnr_cons) = unbounded();
    let model_path = env::var("DF_MODEL").ok().map(PathBuf::from);
    let _c = DeepFilterCapture::new(model_path, Some(lsnr_prod), None, None, None);

    loop {
        sleep(Duration::from_millis(200));
        while let Ok(lsnr) = lsnr_cons.try_recv() {
            print!("\rCurrent SNR: {:>5.1} dB{esc}[1;", lsnr, esc = 27 as char);
        }
        stdout().flush().unwrap();
    }
}
