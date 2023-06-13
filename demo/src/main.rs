#![feature(slice_flatten, array_chunks, get_many_mut)]

use std::env;
use std::future::Future;
use std::path::PathBuf;
use std::process::exit;
use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::Duration;

use crossbeam_channel::unbounded;
use iced::widget::{self, column, container, image, row, slider, text, Container, Image};
use iced::{
    alignment, executor, Alignment, Application, Command, ContentFit, Element, Length, Settings,
    Subscription, Theme,
};
use image_rs::{imageops, Rgba, RgbaImage};

mod capture;
mod cmap;
use capture::*;

pub fn main() -> iced::Result {
    capture::INIT_LOGGER.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
            .filter_module("tract_onnx", log::LevelFilter::Error)
            .filter_module("tract_core", log::LevelFilter::Error)
            .filter_module("tract_hir", log::LevelFilter::Error)
            .filter_module("tract_linalg", log::LevelFilter::Error)
            .filter_module("iced_winit", log::LevelFilter::Error)
            .filter_module("iced_wgpu", log::LevelFilter::Error)
            .filter_module("wgpu_core", log::LevelFilter::Error)
            .filter_module("wgpu_hal", log::LevelFilter::Error)
            .filter_module("naga", log::LevelFilter::Error)
            .filter_module("crossfont", log::LevelFilter::Error)
            .format(capture::log_format)
            .init();
    });

    SpecView::run(Settings::default())
}

static mut SPEC_NOISY: Option<Arc<Mutex<SpecImage>>> = None;
static mut SPEC_ENH: Option<Arc<Mutex<SpecImage>>> = None;

struct SpecView {
    df_worker: DeepFilterCapture,
    lsnr: f32,
    atten_lim: f32,
    min_threshdb: f32,
    max_erbthreshdb: f32,
    max_dfthreshdb: f32,
    noisy_img: image::Handle,
    enh_img: image::Handle,
    r_lsnr: RecvLsnr,
    r_noisy: RecvSpec,
    r_enh: RecvSpec,
    s_controls: SendControl,
}

#[derive(Debug, Clone, Copy)]
pub enum Message {
    None,
    Tick,
    LsnrChanged(f32),
    NoisyChanged,
    EnhChanged,
    AttenLimChanged(f32),
    MinThreshDbChanged(f32),
    MaxErbThreshDbChanged(f32),
    MaxDfThreshDbChanged(f32),
    Exit,
}

struct SpecImage {
    im: RgbaImage,
    n_frames: u32,
    n_freqs: u32,
    vmin: f32,
    vmax: f32,
}

impl SpecImage {
    fn new(n_frames: u32, n_freqs: u32, vmin: f32, vmax: f32) -> Self {
        Self {
            // Store image transposed so we can iterate over rows quickly
            im: RgbaImage::new(n_freqs, n_frames),
            n_frames,
            n_freqs,
            vmin,
            vmax,
        }
    }
    fn w(&self) -> usize {
        self.n_frames as usize
    }
    fn h(&self) -> usize {
        self.n_freqs as usize
    }
    fn update<I>(&mut self, specs: I, mut n_specs: usize)
    where
        I: Iterator<Item = Box<[f32]>>,
    {
        if n_specs == 0 {
            return;
        }
        if n_specs >= self.n_frames as usize {
            // Just drop a few
            n_specs = self.n_frames as usize - 1;
        }
        for (spec, im_row) in specs.take(n_specs).zip(self.im.rows_mut()) {
            for (s, x) in spec.iter().zip(im_row) {
                // clamp and normalize
                let v = (s.min(self.vmax).max(self.vmin) - self.vmin) / (self.vmax - self.vmin);
                *x = Rgba(cmap::CMAP_INFERNO[(v * 255.) as usize]);
            }
        }
        let (w, h) = (self.w(), self.h());
        self.im.rotate_left((w - n_specs) * 4 * h);
    }
    fn image_handle(&self) -> image::Handle {
        let imt_buf = imageops::rotate270(&self.im).as_raw().to_vec();
        image::Handle::from_pixels(self.n_frames, self.n_freqs, imt_buf)
    }
}

impl Application for SpecView {
    type Executor = executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = ();

    fn new(_flags: ()) -> (Self, Command<Message>) {
        let (s_lsnr, r_lsnr) = unbounded();
        let (s_noisy, r_noisy) = unbounded();
        let (s_enh, r_enh) = unbounded();
        let (s_controls, r_controls) = unbounded();

        let model_path = env::var("DF_MODEL").ok().map(PathBuf::from);
        let df_worker = DeepFilterCapture::new(
            model_path,
            Some(s_lsnr),
            Some(s_noisy),
            Some(s_enh),
            Some(r_controls),
        )
        .expect("Failed to initialize DeepFilterNet audio capturing");

        let w = (df_worker.sr / df_worker.frame_size * 10) as u32;
        let freq_res = df_worker.sr / 2 / (df_worker.freq_size - 1);
        let h = (8000 / freq_res) as u32;
        let (noisy_img, enh_img) = unsafe {
            SPEC_NOISY = Some(Arc::new(Mutex::new(SpecImage::new(w, h, -100., -10.))));
            SPEC_ENH = Some(Arc::new(Mutex::new(SpecImage::new(w, h, -100., -10.))));
            (
                SPEC_NOISY.as_ref().unwrap().lock().unwrap().image_handle(),
                SPEC_ENH.as_ref().unwrap().lock().unwrap().image_handle(),
            )
        };
        (
            Self {
                df_worker,
                lsnr: 0.,
                atten_lim: 100.,
                min_threshdb: -15.,
                max_erbthreshdb: 35.,
                max_dfthreshdb: 35.,
                r_lsnr,
                r_noisy,
                r_enh,
                s_controls,
                noisy_img,
                enh_img,
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        "DeepFilterNet Demo".to_string()
    }

    // fn theme(&self) -> Self::Theme {
    //     Theme::Dark
    // }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::None => (),
            Message::Exit => {
                self.df_worker.should_stop().expect("Failed to stop DF worker");
                exit(0);
            }
            Message::Tick => {
                let mut commands = Vec::new();
                if let Some(task) = self.update_lsnr() {
                    commands.push(Command::perform(task, move |message| message))
                }
                if let Some(task) = self.update_noisy() {
                    commands.push(Command::perform(task, move |message| message))
                }
                if let Some(task) = self.update_enh() {
                    commands.push(Command::perform(task, move |message| message))
                }
                return Command::batch(commands);
            }
            Message::LsnrChanged(lsnr) => self.lsnr = lsnr,
            Message::NoisyChanged => {
                self.noisy_img = unsafe {
                    SPEC_NOISY
                        .as_ref()
                        .unwrap()
                        .lock()
                        .expect("Failed to lock SPEC_NOISY")
                        .image_handle()
                };
            }
            Message::EnhChanged => {
                self.enh_img = unsafe {
                    SPEC_ENH
                        .as_ref()
                        .unwrap()
                        .lock()
                        .expect("Failed to lock SPEC_ENH")
                        .image_handle()
                };
            }
            Message::AttenLimChanged(v) => {
                self.atten_lim = v;
                self.s_controls
                    .send((DfControl::AttenLim, v))
                    .expect("Failed to send DfControl")
            }
            Message::MinThreshDbChanged(v) => {
                self.min_threshdb = v;
                self.s_controls
                    .send((DfControl::MinThreshDb, v))
                    .expect("Failed to send DfControl")
            }
            Message::MaxErbThreshDbChanged(v) => {
                self.max_erbthreshdb = v;
                self.s_controls
                    .send((DfControl::MaxErbThreshDb, v))
                    .expect("Failed to send DfControl")
            }
            Message::MaxDfThreshDbChanged(v) => {
                self.max_dfthreshdb = v;
                self.s_controls
                    .send((DfControl::MaxDfThreshDb, v))
                    .expect("Failed to send DfControl")
            }
        }
        Command::none()
    }

    fn view(&self) -> Element<Message> {
        let content = column![row![
            text("DeepFilterNet Demo").size(40).width(Length::Fill),
            button("exit").on_press(Message::Exit)
        ]
        .width(1000),];
        #[cfg(feature = "thresholds")]
        let content = {
            content
                .push(slider_view(
                    "Threshold Min [dB]",
                    self.min_threshdb,
                    -15.,
                    35.,
                    Message::MinThreshDbChanged,
                    1000,
                ))
                .push(slider_view(
                    "Threshold ERB Max [dB]",
                    self.max_erbthreshdb,
                    -15.,
                    35.,
                    Message::MaxErbThreshDbChanged,
                    1000,
                ))
                .push(slider_view(
                    "Threshold DF  Max [dB]",
                    self.max_dfthreshdb,
                    -15.,
                    35.,
                    Message::MaxDfThreshDbChanged,
                    1000,
                ))
        };
        let content = content
            .push(slider_view(
                "Noise Attenuation [dB]",
                self.atten_lim,
                0.,
                100.,
                Message::AttenLimChanged,
                1000,
            ))
            .push(self.specs())
            .push(
                row![
                    text("Current SNR:").size(18),
                    text(format!("{:>5.1} dB", self.lsnr))
                        .size(18)
                        .width(80)
                        .horizontal_alignment(alignment::Horizontal::Right)
                ]
                .spacing(20)
                .align_items(Alignment::End),
            );

        container(content)
            .padding(50)
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x()
            .center_y()
            .into()
    }

    fn subscription(&self) -> Subscription<Message> {
        iced::time::every(std::time::Duration::from_millis(20)).map(|_| Message::Tick)
    }
}

impl SpecView {
    fn update_lsnr(&mut self) -> Option<impl Future<Output = Message>> {
        if self.r_lsnr.is_empty() {
            return None;
        }
        let recv = self.r_lsnr.clone();
        Some(async move {
            sleep(Duration::from_millis(100));
            let mut lsnr = 0.;
            let mut n = 0;
            while let Ok(v) = recv.try_recv() {
                lsnr += v;
                n += 1;
            }
            if n > 0 {
                lsnr /= n as f32;
                Message::LsnrChanged(lsnr)
            } else {
                Message::None
            }
        })
    }

    fn update_noisy(&mut self) -> Option<impl Future<Output = Message>> {
        if self.r_noisy.is_empty() {
            return None;
        }
        let recv = self.r_noisy.clone();
        Some(async move {
            let n = recv.len();
            unsafe {
                let mut spec =
                    SPEC_NOISY.as_mut().unwrap().lock().expect("Failed to lock SPEC_NOISY");
                spec.update(recv.iter().take(n), n);
            }
            Message::NoisyChanged
        })
    }

    fn update_enh(&mut self) -> Option<impl Future<Output = Message>> {
        if self.r_enh.is_empty() {
            return None;
        }
        let recv = self.r_enh.clone();
        Some(async move {
            let n = recv.len();
            unsafe {
                let mut spec = SPEC_ENH.as_mut().unwrap().lock().expect("Failed to lock SPEC_ENH");
                spec.update(recv.iter().take(n), n);
            }
            Message::EnhChanged
        })
    }
    fn specs(&self) -> Container<Message> {
        container(column![
            spec_view("Noisy", self.noisy_img.clone(), 1000, 350),
            spec_view("DeepFilterNet Enhanced", self.enh_img.clone(), 1000, 350),
        ])
    }
}

fn spec_view(title: &str, im: image::Handle, width: u16, height: u16) -> Element<Message> {
    column![
        text(title).size(24).width(Length::Fill),
        spec_raw(im, width, height)
    ]
    .max_width(width)
    .width(Length::Fill)
    .into()
}
fn spec_raw<'a>(im: image::Handle, width: u16, height: u16) -> Container<'a, Message> {
    container(Image::new(im).width(width).height(height).content_fit(ContentFit::Fill))
        .max_width(width)
        .max_height(height)
        .width(Length::Fill)
        .center_x()
        .center_y()
}

fn slider_view<'a>(
    title: &str,
    value: f32,
    min: f32,
    max: f32,
    message: impl Fn(f32) -> Message + 'a,
    width: u16,
) -> Element<'a, Message> {
    column![
        text(title).size(18).width(Length::Fill),
        row![
            container(slider(min..=max, value, message)).width(Length::Fill),
            text(format!("{:.0}", value))
                .size(18)
                .width(100)
                .horizontal_alignment(alignment::Horizontal::Right)
                .vertical_alignment(alignment::Vertical::Top),
        ]
    ]
    .max_width(width)
    .width(Length::Fill)
    .into()
}

fn button(text: &str) -> widget::Button<'_, Message> {
    widget::button(text).padding(10)
}
