use std::collections::VecDeque;
use std::io::{self, Write};
use std::sync::mpsc::Receiver;
use std::sync::{
    mpsc::{channel, Sender},
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
type ControlSender = Sender<(DfControl, f32)>;
type ControlReceiver = Receiver<(DfControl, f32)>;

struct DfPlugin {
    inqueue: SampleQueue,
    outqueue: SampleQueue,
    controlqueue: ControlSender,
    id: String,
    ch: usize,
    sr: usize,
    frame_size: usize,
    proc_delay: usize,
    t_proc_change: usize,
    sleep_duration: Duration,
    control_hist: DfControlHistory,
    _h: JoinHandle<()>, // Worker handle
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

fn get_worker_fn(
    mut df: DfTract,
    inqueue: SampleQueue,
    outqueue: SampleQueue,
    controls: ControlReceiver,
    sleep_duration: Duration,
    id: String,
) -> impl FnMut() {
    move || {
        let mut inframe = Array2::zeros((df.ch, df.hop_size));
        let mut outframe = Array2::zeros((df.ch, df.hop_size));
        let t_audio_ms = df.hop_size as f32 / df.sr as f32 * 1000.;
        loop {
            if let Ok((c, v)) = controls.try_recv() {
                match c {
                    DfControl::AttenLim => {
                        df.set_atten_lim(v).expect("Failed to set attenuation limit.")
                    }
                    DfControl::MinThreshDb => df.min_db_thresh = v,
                    DfControl::MaxErbThreshDb => df.max_db_erb_thresh = v,
                    DfControl::MaxDfThreshDb => df.max_db_df_thresh = v,
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

fn init_df(channels: usize) {
    unsafe {
        if let Some(m) = MODEL.as_ref() {
            if m.ch == channels {
                return;
            }
        }
    }
    let df_params = DfParams::default();
    let r_params = RuntimeParams::new(channels, false, 100., -10., 30., 20., ReduceMask::MEAN);
    let df = DfTract::new(df_params, &r_params).expect("Could not initialize DeepFilter runtime.");
    unsafe { MODEL = Some(df) }
}

fn get_new_df(channels: usize) -> impl Fn(&PluginDescriptor, u64) -> DfPlugin {
    move |_: &PluginDescriptor, sample_rate: u64| {
        let t0 = Instant::now();
        INIT_LOGGER.call_once(|| {
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
                .format(log_format)
                .init();
        });

        init_df(channels);
        let m = unsafe { MODEL.clone().unwrap() };
        assert_eq!(m.sr as u64, sample_rate, "Unsupported sample rate");
        let inqueue = Arc::new(Mutex::new(vec![
            VecDeque::with_capacity(m.hop_size * 4);
            channels
        ]));
        let outqueue = Arc::new(Mutex::new(vec![
            VecDeque::with_capacity(m.hop_size * 4);
            channels
        ]));
        let sr = m.sr;
        let frame_size = m.hop_size;
        let proc_delay = m.hop_size;
        // Add a buffer of 1 frame to compensate processing delays causing underruns
        for o_ch in outqueue.lock().unwrap().iter_mut() {
            for _ in 0..proc_delay {
                o_ch.push_back(0f32)
            }
        }
        let sleep_duration = Duration::from_secs_f32(m.hop_size as f32 / sr as f32 / 5.);
        let id = Uuid::new_v4().as_urn().to_string().split_at(33).1.to_string();

        let (controlsender, recv) = channel();

        let worker_handle = thread::spawn(get_worker_fn(
            m,
            Arc::clone(&inqueue),
            Arc::clone(&outqueue),
            recv,
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
            inqueue,
            outqueue,
            controlqueue: controlsender,
            ch: channels,
            sr,
            id,
            frame_size,
            proc_delay,
            t_proc_change: 0,
            sleep_duration,
            control_hist: hist,
            _h: worker_handle,
        }
    }
}

#[derive(PartialEq)]
enum DfControl {
    AttenLim,
    MinThreshDb,
    MaxErbThreshDb,
    MaxDfThreshDb,
}
impl DfControl {
    fn from_port_name(name: &str) -> Self {
        match name {
            "Attenuation Limit (dB)" => Self::AttenLim,
            "Min processing threshold (dB)" => Self::MinThreshDb,
            "Max ERB processing threshold (dB)" => Self::MaxErbThreshDb,
            "Max DF processing threshold (dB)" => Self::MaxDfThreshDb,
            _ => panic!("name not found"),
        }
    }
}
struct DfControlHistory {
    atten_lim: f32,
    min_thresh_db: f32,
    max_erb_thresh_db: f32,
    max_df_thresh_db: f32,
}
impl Default for DfControlHistory {
    fn default() -> Self {
        Self {
            atten_lim: 100.,
            min_thresh_db: -10.,
            max_erb_thresh_db: 30.,
            max_df_thresh_db: 20.,
        }
    }
}
impl DfControlHistory {
    fn get(&self, c: &DfControl) -> f32 {
        match c {
            DfControl::AttenLim => self.atten_lim,
            DfControl::MinThreshDb => self.min_thresh_db,
            DfControl::MaxErbThreshDb => self.max_erb_thresh_db,
            DfControl::MaxDfThreshDb => self.max_df_thresh_db,
        }
    }
    fn set(&mut self, c: &DfControl, v: f32) {
        match c {
            DfControl::AttenLim => self.atten_lim = v,
            DfControl::MinThreshDb => self.min_thresh_db = v,
            DfControl::MaxErbThreshDb => self.max_erb_thresh_db = v,
            DfControl::MaxDfThreshDb => self.max_df_thresh_db = v,
        }
    }
}

impl Plugin for DfPlugin {
    fn activate(&mut self) {
        log::info!("DF {} | activate", self.id);
    }
    fn deactivate(&mut self) {
        log::info!("DF {} | deactivate", self.id);
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
                log::info!("DF {} | Setting '{}' to {:.1}", self.id, p.port.name, v);
                self.control_hist.set(&c, v);
                self.controlqueue.send((c, v)).expect("Failed to send control parameter");
            }
        }

        {
            let i_q = &mut self.inqueue.lock().unwrap();
            for (i_ch, i_q_ch) in inputs.iter().zip(i_q.iter_mut()) {
                for &i in i_ch.iter() {
                    i_q_ch.push_back(i)
                }
            }
        }

        'outer: loop {
            {
                let o_q = &mut self.outqueue.lock().unwrap();
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
            for o_ch in self.outqueue.lock().unwrap().iter_mut() {
                for _ in 0..self.frame_size {
                    o_ch.push_back(0f32)
                }
            }
        } else if self.t_proc_change > 10 * self.sr / self.frame_size
            && rtf < 0.5
            && self.proc_delay >= self.frame_size
        {
            // Reduce delay again
            let dropped_samples = {
                let o_q = &mut self.outqueue.lock().unwrap();
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
                    default: Some(DefaultValue::Maximum),
                    lower_bound: Some(0.),
                    upper_bound: Some(100.),
                },
                Port {
                    name: "Min processing threshold (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(DefaultValue::Minimum),
                    lower_bound: Some(-15.),
                    upper_bound: Some(35.),
                },
                Port {
                    name: "Max ERB processing threshold (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(DefaultValue::Maximum),
                    lower_bound: Some(-15.),
                    upper_bound: Some(35.),
                },
                Port {
                    name: "Max DF processing threshold (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(DefaultValue::High),
                    lower_bound: Some(-15.),
                    upper_bound: Some(35.),
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
                    default: Some(DefaultValue::Maximum),
                    lower_bound: Some(0.),
                    upper_bound: Some(100.),
                },
                Port {
                    name: "Min processing threshold (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(DefaultValue::Minimum),
                    lower_bound: Some(-15.),
                    upper_bound: Some(35.),
                },
                Port {
                    name: "Max ERB processing threshold (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(DefaultValue::Maximum),
                    lower_bound: Some(-15.),
                    upper_bound: Some(35.),
                },
                Port {
                    name: "Max DF processing threshold (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(DefaultValue::High),
                    lower_bound: Some(-15.),
                    upper_bound: Some(35.),
                },
            ],
            new: |d, sr| Box::new(get_new_df(2)(d, sr)),
        }),
        _ => None,
    }
}
