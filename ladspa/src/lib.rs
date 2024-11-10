use std::collections::VecDeque;
use std::fmt;
use std::io::{self, Write};
use std::sync::Weak;
use std::sync::{
    mpsc::{sync_channel, Receiver, SyncSender},
    Arc, Mutex, Once,
};
use std::thread::{self, sleep, JoinHandle};
use std::time::{Duration, Instant};

use df::tract::*;
use ladspa::{DefaultValue, Plugin, PluginDescriptor, Port, PortConnection, PortDescriptor};
use ndarray::{prelude::*, OwnedRepr};
use uuid::Uuid;

static INIT_LOGGER: Once = Once::new();

type AudioFrame = ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>;
type ControlProd = SyncSender<(DfControl, f32)>;
type ControlRecv = Receiver<(DfControl, f32)>;
#[cfg(feature = "dbus")]
use ::{
    event_listener::Event,
    zbus::{blocking::ConnectionBuilder, dbus_interface},
};
#[cfg(feature = "dbus")]
const DBUS_NAME: &str = "org.deepfilter.DeepFilterLadspa";
#[cfg(feature = "dbus")]
const DBUS_PATH: &str = "/org/deepfilter/DeepFilterLadspa";

const ATTEN_LIM_DEF: DefaultValue = DefaultValue::Maximum;
const ATTEN_LIM_MIN: f32 = 0.;
const ATTEN_LIM_MAX: f32 = 100.;
const PF_BETA_DEF: DefaultValue = DefaultValue::Minimum;
const PF_BETA_MIN: f32 = 0.;
const PF_BETA_MAX: f32 = 0.05;
const MIN_PROC_THRESH_DEF: DefaultValue = DefaultValue::Minimum;
const MIN_PROC_THRESH_MIN: f32 = -15.;
const MIN_PROC_THRESH_MAX: f32 = 35.;
const MAX_ERB_BUF_DEF: DefaultValue = DefaultValue::Maximum;
const MAX_ERB_BUF_MIN: f32 = -15.;
const MAX_ERB_BUF_MAX: f32 = 35.;
const MAX_DF_BUF_DEF: DefaultValue = DefaultValue::Maximum;
const MAX_DF_BUF_MIN: f32 = -15.;
const MAX_DF_BUF_MAX: f32 = 35.;
const MIN_PROC_BUF_DEF: DefaultValue = DefaultValue::Minimum;
const MIN_PROC_BUF_MIN: f32 = 0.;
const MIN_PROC_BUF_MAX: f32 = 10.;

struct DfPlugin {
    raw_audio_queue: Vec<VecDeque<f32>>,
    raw_audio_sender: SyncSender<AudioFrame>,
    cleaned_audio_queue: Vec<VecDeque<f32>>,
    cleaned_audio_receiver: Receiver<AudioFrame>,
    control_tx: ControlProd,
    id: String,
    channel_count: usize,
    hop_size: usize,
    control_hist: DfControlHistory,
    #[cfg(feature = "dbus")]
    _dbus: Option<(JoinHandle<()>, Arc<Event>)>, // dbus thread handle
}

const ID_MONO: u64 = 7843795;
const ID_STEREO: u64 = 7843796;

fn log_format(buf: &mut env_logger::fmt::Formatter, record: &log::Record) -> io::Result<()> {
    let ts = buf.timestamp_millis();
    let module = if let Some(m) = record.module_path() {
        format!(" {} |", m.replace("::reexport_dataset_modules:", ""))
    } else {
        "".to_string()
    };
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

fn syslog_format(buf: &mut env_logger::fmt::Formatter, record: &log::Record) -> io::Result<()> {
    writeln!(
        buf,
        "<{}>{}: {}",
        match record.level() {
            log::Level::Error => 3,
            log::Level::Warn => 4,
            log::Level::Info => 6,
            log::Level::Debug => 7,
            log::Level::Trace => 7,
        },
        record.target(),
        record.args()
    )
}

fn new_df_frame(df: &DfTract) -> AudioFrame {
    Array2::zeros((df.ch, df.hop_size))
}

fn get_worker_fn(
    raw_receiver: Receiver<AudioFrame>,
    clean_sender: SyncSender<AudioFrame>,
    channels: usize,
    controls: ControlRecv,
    id: String,
) -> impl FnMut() + Send {
    move || {
        let (mut df, _, _) = init_df(channels);

        let mut outframe = new_df_frame(&df);

        let t_audio_ms = df.hop_size as f32 / df.sr as f32 * 1000.;

        if let Ok((c, v)) = controls.try_recv() {
            log::info!("DF {} | Setting '{}' to {:.1}", id, c, v);
            match c {
                DfControl::AttenLim => df.set_atten_lim(v),
                DfControl::PfBeta => df.set_pf_beta(v),
                DfControl::MinThreshDb => df.min_db_thresh = v,
                DfControl::MaxErbThreshDb => df.max_db_erb_thresh = v,
                DfControl::MaxDfThreshDb => df.max_db_df_thresh = v,
                _ => (),
            }
        }

        while let Ok(inframe) = raw_receiver.recv() {
            let t0 = Instant::now();

            let lsnr = df
                .process(inframe.view(), outframe.view_mut())
                .expect("Error during df::process");

            let td_ms = t0.elapsed().as_secs_f32() * 1000.;
            log::debug!(
                "DF {} | Enhanced {:.1}ms frame. SNR: {:>5.1}, Processing time: {:>4.1}ms, RTF: {:.2}",
                id,
                t_audio_ms,
                lsnr,
                td_ms,
                td_ms / t_audio_ms
            );

            if clean_sender.send(outframe).is_err() {
                break;
            }

            outframe = inframe;
        }

        println!("thread exiting")
    }
}

/// Initialize DF model and returns sample rate and frame size
fn init_df(channels: usize) -> (DfTract, usize, usize) {
    let df_params = DfParams::default();
    let r_params = RuntimeParams::default_with_ch(channels);
    let df = DfTract::new(df_params, &r_params).expect("Could not initialize DeepFilter runtime");
    let (sr, frame_size) = (df.sr, df.hop_size);
    (df, sr, frame_size)
}

fn get_new_df(channels: usize) -> impl Fn(&PluginDescriptor, u64) -> DfPlugin {
    move |_: &PluginDescriptor, sample_rate: u64| {
        let t0 = Instant::now();
        let f = match std::env::var("RUST_LOG_STYLE") {
            Ok(s) if s == "SYSTEMD" => syslog_format,
            _ => log_format,
        };
        INIT_LOGGER.call_once(|| {
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
                .filter_module("polling", log::LevelFilter::Error)
                .filter_module("async_io", log::LevelFilter::Error)
                .filter_module("tract_onnx", log::LevelFilter::Error)
                .filter_module("tract_core", log::LevelFilter::Error)
                .filter_module("tract_hir", log::LevelFilter::Error)
                .filter_module("tract_linalg", log::LevelFilter::Error)
                .format(f)
                .init();
        });

        let (_, m_sr, hop_size) = init_df(channels);
        assert_eq!(m_sr as u64, sample_rate, "Unsupported sample rate");

        let id = Uuid::new_v4().as_urn().to_string().split_at(33).1.to_string();

        let (control_tx, control_rx) = sync_channel(32);

        let (raw_tx, raw_rx) = sync_channel(128);
        let (clean_tx, clean_rx) = sync_channel(128);

        thread::spawn(get_worker_fn(
            raw_rx,
            clean_tx,
            channels,
            control_rx,
            id.clone(),
        ));
        let hist = DfControlHistory::default();
        log::info!(
            "DF {} | Initialized plugin in {:.1}ms",
            &id,
            t0.elapsed().as_secs_f32() * 1000.
        );
        dbg!(&channels);
        DfPlugin {
            raw_audio_queue: vec![VecDeque::new(); channels],
            cleaned_audio_queue: vec![VecDeque::from(vec![0.0; hop_size * 2]); channels],
            raw_audio_sender: raw_tx,
            cleaned_audio_receiver: clean_rx,
            control_tx,
            channel_count: channels,
            hop_size,
            id,
            control_hist: hist,
            #[cfg(feature = "dbus")]
            _dbus: None,
        }
    }
}

#[derive(PartialEq)]
enum DfControl {
    AttenLim,
    PfBeta,
    MinThreshDb,
    MaxErbThreshDb,
    MaxDfThreshDb,
    MinBufferFrames,
}
impl DfControl {
    fn from_port_name(name: &str) -> Self {
        match name {
            "Attenuation Limit (dB)" => Self::AttenLim,
            "Post Filter Beta" => Self::PfBeta,
            "Min processing threshold (dB)" => Self::MinThreshDb,
            "Max ERB processing threshold (dB)" => Self::MaxErbThreshDb,
            "Max DF processing threshold (dB)" => Self::MaxDfThreshDb,
            "Min Processing Buffer (frames)" => Self::MinBufferFrames,
            _ => panic!("name not found"),
        }
    }
}
impl fmt::Display for DfControl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DfControl::AttenLim => write!(f, "Attenuation Limit (dB)"),
            DfControl::PfBeta => write!(f, "Post Filter Beta"),
            DfControl::MinThreshDb => write!(f, "Min processing threshold (dB)"),
            DfControl::MaxErbThreshDb => write!(f, "Max ERB processing threshold (dB)"),
            DfControl::MaxDfThreshDb => write!(f, "Max DF processing threshold (dB)"),
            DfControl::MinBufferFrames => write!(f, "Min Processing Buffer (frames)"),
        }
    }
}

struct DfControlHistory {
    atten_lim: f32,
    pf_beta: f32,
    min_thresh_db: f32,
    max_erb_thresh_db: f32,
    max_df_thresh_db: f32,
    min_buffer_frames: f32,
}
impl Default for DfControlHistory {
    fn default() -> Self {
        Self {
            atten_lim: 100.,
            pf_beta: 0.0,
            min_thresh_db: -10.,
            max_erb_thresh_db: 30.,
            max_df_thresh_db: 20.,
            min_buffer_frames: 0.,
        }
    }
}
impl DfControlHistory {
    fn get(&self, c: &DfControl) -> f32 {
        match c {
            DfControl::AttenLim => self.atten_lim,
            DfControl::PfBeta => self.pf_beta,
            DfControl::MinThreshDb => self.min_thresh_db,
            DfControl::MaxErbThreshDb => self.max_erb_thresh_db,
            DfControl::MaxDfThreshDb => self.max_df_thresh_db,
            DfControl::MinBufferFrames => self.min_buffer_frames,
        }
    }
    fn set(&mut self, c: &DfControl, v: f32) {
        match c {
            DfControl::AttenLim => self.atten_lim = v,
            DfControl::PfBeta => self.pf_beta = v,
            DfControl::MinThreshDb => self.min_thresh_db = v,
            DfControl::MaxErbThreshDb => self.max_erb_thresh_db = v,
            DfControl::MaxDfThreshDb => self.max_df_thresh_db = v,
            DfControl::MinBufferFrames => self.min_buffer_frames = v,
        }
    }
}

impl Plugin for DfPlugin {
    fn activate(&mut self) {
        log::info!("DF {} | activate", self.id);
        #[cfg(feature = "dbus")]
        {
            if !test_dbus_name_avail() {
                return;
            }
            let init = Arc::new(Event::new());
            let done = Arc::new(Event::new());
            let init_listen = init.listen();
            self._dbus = Some((
                thread::spawn(get_dbus_worker(
                    self.control_tx.clone(),
                    init,
                    done.clone(),
                    self.id.clone(),
                )),
                done,
            ));
            init_listen.wait(); // Wait for dbus server init
            log::debug!("dbus thread spawned")
        }
    }

    fn deactivate(&mut self) {
        log::info!("DF {} | deactivate", self.id);
        #[cfg(feature = "dbus")]
        {
            if let Some((handle, done)) = self._dbus.take() {
                done.notify(1);
                for _ in 0..20 {
                    sleep(Duration::from_millis(5));
                    if handle.is_finished() {
                        match handle.join() {
                            Ok(_) => log::debug!("{} | dbus thread joined", self.id),
                            Err(e) => log::error!("{} | dbus thread error: {:?}", self.id, e),
                        }
                        break;
                    }
                    log::error!("{} | Joining dbus thread timed out.", self.id);
                }
            }
        }
    }

    fn run<'a>(&mut self, sample_count: usize, ports: &[&'a PortConnection<'a>]) {
        let mut i = 0;
        let mut input_ports = Vec::with_capacity(self.channel_count);
        let mut output_ports = Vec::with_capacity(self.channel_count);
        for _ in 0..self.channel_count {
            input_ports.push(ports[i].unwrap_audio());
            i += 1;
        }
        for _ in 0..self.channel_count {
            output_ports.push(ports[i].unwrap_audio_mut());
            i += 1;
        }
        for p in ports[i..].iter() {
            let &v = p.unwrap_control();
            let c = DfControl::from_port_name(p.port.name);
            if c == DfControl::AttenLim && v >= 100. {
                for (i_ch, o_ch) in input_ports.iter().zip(output_ports.iter_mut()) {
                    for (&i, o) in i_ch.iter().zip(o_ch.iter_mut()) {
                        *o = i
                    }
                }
            }
            if v != self.control_hist.get(&c) {
                self.control_hist.set(&c, v);
                self.control_tx.send((c, v)).expect("Failed to send control parameter");
            }
        }

        for (port_buffer, queue_buffer) in input_ports.iter().zip(self.raw_audio_queue.iter_mut()) {
            assert_eq!(port_buffer.len(), sample_count);
            for &i in port_buffer.iter() {
                queue_buffer.push_back(i);
            }
        }

        while self.raw_audio_queue[0].len() >= self.hop_size {
            let mut frame = Array2::zeros((self.channel_count, self.hop_size));
            for (mut frame_channel, queue_channel) in
                frame.outer_iter_mut().zip(self.raw_audio_queue.iter_mut())
            {
                for i in frame_channel.iter_mut() {
                    *i = queue_channel.pop_front().unwrap();
                }
            }
            if let Err(e) = self.raw_audio_sender.try_send(frame) {
                log::warn!(
                    "DF {} | Processing thread is overloaded! Dropping frame",
                    self.id,
                );
                dbg!(&e);
                dbg!(self.hop_size);
            }
        }

        while let Ok(frame) = self.cleaned_audio_receiver.try_recv() {
            for (frame_channel, queue_channel) in
                frame.outer_iter().zip(self.cleaned_audio_queue.iter_mut())
            {
                for &o in frame_channel.iter() {
                    queue_channel.push_back(o);
                }
            }
        }

        for (queue_channel, port_channel) in
            self.cleaned_audio_queue.iter_mut().zip(output_ports.iter_mut())
        {
            assert_eq!(port_channel.len(), sample_count);
            for o in port_channel.iter_mut() {
                *o = queue_channel.pop_front().unwrap_or_default();
            }
        }

        while self.cleaned_audio_queue[0].len() > sample_count.max(self.hop_size) {
            for channel in self.cleaned_audio_queue.iter_mut() {
                channel.pop_front().unwrap();
            }
        }
    }
}

#[cfg(feature = "dbus")]
fn build_dbus_session<I>(control: I) -> Result<zbus::blocking::Connection, zbus::Error>
where
    I: zbus::Interface,
{
    ConnectionBuilder::session()?
        .name(DBUS_NAME)?
        .serve_at(DBUS_PATH, control)?
        .build()
}
#[cfg(feature = "dbus")]
fn test_dbus_name_avail() -> bool {
    let control = DfDbusControlDummy {};
    match build_dbus_session(control) {
        Ok(con) => {
            con.release_name(DBUS_NAME).expect("Failed to release dbus name");
            true
        }
        Err(e) => {
            log::error!("Failed to init dbus session {}", e);
            false
        }
    }
}

#[cfg(feature = "dbus")]
fn get_dbus_worker(
    tx: ControlProd,
    init: Arc<Event>,
    done: Arc<Event>,
    id: String,
) -> impl FnMut() {
    move || {
        log::debug!("{id} | Initializing dbus server");
        let done_listener = done.clone().listen();
        let control = DfDbusControl { tx: tx.clone() };
        let con = build_dbus_session(control).expect("Failed to init dbus session");
        init.notify(1); // Notify caller that dbus server has been initialized.
        done_listener.wait();
        con.release_name(DBUS_NAME).expect("Failed to release dbus name");
        log::debug!("{id} | Got done notification. Releasing dbus name");
    }
}

#[cfg(feature = "dbus")]
struct DfDbusControlDummy {}
#[cfg(feature = "dbus")]
#[dbus_interface(name = "org.deepfilter.DeepFilterLadspa")]
impl DfDbusControlDummy {}

#[cfg(feature = "dbus")]
struct DfDbusControl {
    tx: ControlProd,
}

#[cfg(feature = "dbus")]
#[dbus_interface(name = "org.deepfilter.DeepFilterLadspa")]
impl DfDbusControl {
    fn atten_lim(&self, lim: u32) {
        self.tx
            .send((DfControl::AttenLim, lim as f32))
            .expect("Failed to send DfControl");
    }
    fn pf_beta(&self, beta: f32) {
        self.tx.send((DfControl::PfBeta, beta)).expect("Failed to send DfControl");
    }
    fn min_processing_thresh(&self, thresh: i32) {
        self.tx
            .send((DfControl::MinThreshDb, thresh as f32))
            .expect("Failed to send DfControl")
    }
    fn max_erb_thresh(&self, thresh: i32) {
        self.tx
            .send((DfControl::MaxErbThreshDb, thresh as f32))
            .expect("Failed to send DfControl")
    }
    fn max_df_thresh(&self, thresh: i32) {
        self.tx
            .send((DfControl::MaxDfThreshDb, thresh as f32))
            .expect("Failed to send DfControl")
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
                Port {
                    name: "Attenuation Limit (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(ATTEN_LIM_DEF),
                    lower_bound: Some(ATTEN_LIM_MIN),
                    upper_bound: Some(ATTEN_LIM_MAX),
                },
                Port {
                    name: "Min processing threshold (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(MIN_PROC_THRESH_DEF),
                    lower_bound: Some(MIN_PROC_THRESH_MIN),
                    upper_bound: Some(MIN_PROC_THRESH_MAX),
                },
                Port {
                    name: "Max ERB processing threshold (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(MAX_ERB_BUF_DEF),
                    lower_bound: Some(MAX_ERB_BUF_MIN),
                    upper_bound: Some(MAX_ERB_BUF_MAX),
                },
                Port {
                    name: "Max DF processing threshold (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(MAX_DF_BUF_DEF),
                    lower_bound: Some(MAX_DF_BUF_MIN),
                    upper_bound: Some(MAX_DF_BUF_MAX),
                },
                Port {
                    name: "Min Processing Buffer (frames)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(MIN_PROC_BUF_DEF),
                    lower_bound: Some(MIN_PROC_BUF_MIN),
                    upper_bound: Some(MIN_PROC_BUF_MAX),
                },
                Port {
                    name: "Post Filter Beta",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(PF_BETA_DEF),
                    lower_bound: Some(PF_BETA_MIN),
                    upper_bound: Some(PF_BETA_MAX),
                },
            ],
            new: |d, sr| Box::new(get_new_df(1)(d, sr)),
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
                Port {
                    name: "Attenuation Limit (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(ATTEN_LIM_DEF),
                    lower_bound: Some(ATTEN_LIM_MIN),
                    upper_bound: Some(ATTEN_LIM_MAX),
                },
                Port {
                    name: "Min processing threshold (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(MIN_PROC_THRESH_DEF),
                    lower_bound: Some(MIN_PROC_THRESH_MIN),
                    upper_bound: Some(MIN_PROC_THRESH_MAX),
                },
                Port {
                    name: "Max ERB processing threshold (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(MAX_ERB_BUF_DEF),
                    lower_bound: Some(MAX_ERB_BUF_MIN),
                    upper_bound: Some(MAX_ERB_BUF_MAX),
                },
                Port {
                    name: "Max DF processing threshold (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(MAX_DF_BUF_DEF),
                    lower_bound: Some(MAX_DF_BUF_MIN),
                    upper_bound: Some(MAX_DF_BUF_MAX),
                },
                Port {
                    name: "Min Processing Buffer (frames)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(MIN_PROC_BUF_DEF),
                    lower_bound: Some(MIN_PROC_BUF_MIN),
                    upper_bound: Some(MIN_PROC_BUF_MAX),
                },
                Port {
                    name: "Post Filter Beta",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(PF_BETA_DEF),
                    lower_bound: Some(PF_BETA_MIN),
                    upper_bound: Some(PF_BETA_MAX),
                },
            ],
            new: |d, sr| Box::new(get_new_df(2)(d, sr)),
        }),
        _ => None,
    }
}
