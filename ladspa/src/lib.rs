use std::collections::VecDeque;
use std::fmt;
use std::io::{self, Write};
use std::sync::{
    mpsc::{sync_channel, Receiver, SyncSender},
    Arc, Mutex, Once,
};
use std::thread::{self, sleep, JoinHandle};
use std::time::{Duration, Instant};

use df::tract::*;
use ladspa::{DefaultValue, Plugin, PluginDescriptor, Port, PortConnection, PortDescriptor};
use ndarray::prelude::*;
use uuid::Uuid;

static INIT_LOGGER: Once = Once::new();

type SampleQueue = Arc<Mutex<Vec<VecDeque<f32>>>>;
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
    i_tx: SampleQueue,
    o_rx: SampleQueue,
    control_tx: ControlProd,
    id: String,
    ch: usize,
    sr: usize,
    frame_size: usize,
    proc_delay: usize,
    t_proc_change: usize,
    sleep_duration: Duration,
    control_hist: DfControlHistory,
    _h: JoinHandle<()>, // Worker thread handle
    #[cfg(feature = "dbus")]
    _dbus: Option<(JoinHandle<()>, Arc<Event>)>, // dbus thread handle
}

const ID_MONO: u64 = 7843795;
const ID_STEREO: u64 = 7843796;
static mut MODEL: Option<DfTract> = None;

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

fn get_worker_fn(
    inqueue: SampleQueue,
    outqueue: SampleQueue,
    controls: ControlRecv,
    sleep_duration: Duration,
    id: String,
) -> impl FnMut() {
    move || {
        let mut df = unsafe { MODEL.clone().unwrap() };
        let mut inframe = Array2::zeros((df.ch, df.hop_size));
        let mut outframe = Array2::zeros((df.ch, df.hop_size));
        let t_audio_ms = df.hop_size as f32 / df.sr as f32 * 1000.;
        loop {
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
            let got_samples = {
                let mut q = inqueue.lock().unwrap();
                if q[0].len() >= df.hop_size {
                    for (i_q_ch, mut i_ch) in q.iter_mut().zip(inframe.outer_iter_mut()) {
                        for i in i_ch.iter_mut() {
                            *i = i_q_ch.pop_front().unwrap();
                        }
                    }
                    true
                } else {
                    false
                }
            };
            if !got_samples {
                sleep(sleep_duration);
                continue;
            }
            let t0 = Instant::now();
            let lsnr = df
                .process(inframe.view(), outframe.view_mut())
                .expect("Error during df::process");
            {
                let mut o_q = outqueue.lock().unwrap();
                for (o_ch, o_q_ch) in outframe.outer_iter().zip(o_q.iter_mut()) {
                    for &o in o_ch.iter() {
                        o_q_ch.push_back(o)
                    }
                }
            }
            let td_ms = t0.elapsed().as_secs_f32() * 1000.;
            log::debug!(
                "DF {} | Enhanced {:.1}ms frame. SNR: {:>5.1}, Processing time: {:>4.1}ms, RTF: {:.2}",
                id,
                t_audio_ms,
                lsnr,
                td_ms,
                td_ms / t_audio_ms
            );
        }
    }
}

/// Initialize DF model and returns sample rate and frame size
fn init_df(channels: usize) -> (usize, usize) {
    unsafe {
        if let Some(m) = MODEL.as_ref() {
            if m.ch == channels {
                return (m.sr, m.hop_size);
            }
        }
    }

    let df_params = DfParams::default();
    let r_params = RuntimeParams::default_with_ch(channels);
    let df = DfTract::new(df_params, &r_params).expect("Could not initialize DeepFilter runtime");
    let (sr, frame_size) = (df.sr, df.hop_size);
    unsafe { MODEL = Some(df) };
    (sr, frame_size)
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

        let (m_sr, hop) = init_df(channels);
        assert_eq!(m_sr as u64, sample_rate, "Unsupported sample rate");
        let i_tx = Arc::new(Mutex::new(vec![VecDeque::with_capacity(hop * 4); channels]));
        let o_rx = Arc::new(Mutex::new(vec![VecDeque::with_capacity(hop * 4); channels]));
        let frame_size = hop;
        let proc_delay = hop;
        // Add a buffer of 1 frame to compensate processing delays causing underruns
        for o_ch in o_rx.lock().unwrap().iter_mut() {
            for _ in 0..proc_delay {
                o_ch.push_back(0f32)
            }
        }
        let sleep_duration = Duration::from_secs_f32(hop as f32 / m_sr as f32 / 5.);
        let id = Uuid::new_v4().as_urn().to_string().split_at(33).1.to_string();

        let (control_tx, control_rx) = sync_channel(32);

        let worker_handle = thread::spawn(get_worker_fn(
            Arc::clone(&i_tx),
            Arc::clone(&o_rx),
            control_rx,
            sleep_duration,
            id.clone(),
        ));
        let hist = DfControlHistory::default();
        log::info!(
            "DF {} | Initialized plugin in {:.1}ms",
            &id,
            t0.elapsed().as_secs_f32() * 1000.
        );
        DfPlugin {
            i_tx,
            o_rx,
            control_tx,
            ch: channels,
            sr: m_sr,
            id,
            frame_size,
            proc_delay,
            t_proc_change: 0,
            sleep_duration,
            control_hist: hist,
            _h: worker_handle,
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
        let t0 = Instant::now();

        let mut i = 0;
        let mut inputs = Vec::with_capacity(self.ch);
        let mut outputs = Vec::with_capacity(self.ch);
        for _ in 0..self.ch {
            inputs.push(ports[i].unwrap_audio());
            i += 1;
        }
        for _ in 0..self.ch {
            outputs.push(ports[i].unwrap_audio_mut());
            i += 1;
        }
        for p in ports[i..].iter() {
            let &v = p.unwrap_control();
            let c = DfControl::from_port_name(p.port.name);
            if c == DfControl::AttenLim && v >= 100. {
                for (i_ch, o_ch) in inputs.iter().zip(outputs.iter_mut()) {
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

        {
            let i_q = &mut self.i_tx.lock().unwrap();
            for (i_ch, i_q_ch) in inputs.iter().zip(i_q.iter_mut()) {
                for &i in i_ch.iter() {
                    i_q_ch.push_back(i)
                }
            }
        }

        'outer: loop {
            {
                let o_q = &mut self.o_rx.lock().unwrap();
                if o_q[0].len() >= sample_count {
                    for (o_q_ch, o_ch) in o_q.iter_mut().zip(outputs.iter_mut()) {
                        for o in o_ch.iter_mut() {
                            *o = o_q_ch.pop_front().unwrap();
                        }
                    }
                    break 'outer;
                }
            }
            sleep(self.sleep_duration);
        }

        let td = t0.elapsed();
        let t_audio = sample_count as f32 / self.sr as f32;
        let rtf = td.as_secs_f32() / t_audio;
        if rtf >= 1. {
            log::warn!(
                "DF {} | Underrun detected (RTF: {:.2}). Processing too slow!",
                self.id,
                rtf
            );
            if self.proc_delay >= self.sr {
                panic!(
                    "DF {} | Processing too slow! Please upgrade your CPU. Try to decrease 'Max DF processing threshold (dB)'.",
                    self.id,
                );
            }
            self.proc_delay += self.frame_size;
            self.t_proc_change = 0;
            log::info!(
                "DF {} | Increasing processing latency to {:.1}ms",
                self.id,
                self.proc_delay as f32 * 1000. / self.sr as f32
            );
            for o_ch in self.o_rx.lock().unwrap().iter_mut() {
                for _ in 0..self.frame_size {
                    o_ch.push_back(0f32)
                }
            }
        } else if self.t_proc_change > 10 * self.sr / self.frame_size
            && rtf < 0.5
            && self.proc_delay
                >= self.frame_size * (1 + self.control_hist.min_buffer_frames as usize)
        {
            // Reduce delay again
            let dropped_samples = {
                let o_q = &mut self.o_rx.lock().unwrap();
                if o_q[0].len() < self.frame_size {
                    false
                } else {
                    for o_q_ch in o_q.iter_mut().take(self.frame_size) {
                        o_q_ch.pop_front().unwrap();
                    }
                    true
                }
            };
            if dropped_samples {
                self.proc_delay -= self.frame_size;
                self.t_proc_change = 0;
                log::info!(
                    "DF {} | Decreasing processing latency to {:.1}ms",
                    self.id,
                    self.proc_delay as f32 * 1000. / self.sr as f32
                );
            }
        }
        self.t_proc_change += 1;
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
