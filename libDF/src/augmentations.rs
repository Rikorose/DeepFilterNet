use std::collections::BTreeMap;
use std::ops::Range;
#[cfg(feature = "timings")]
use std::time::Instant;

use ndarray::{concatenate, prelude::*, Slice};
use ndarray_rand::rand::{prelude::IteratorRandom, seq::SliceRandom, Rng};
use ndarray_rand::{rand_distr::Normal, rand_distr::Uniform, RandomExt};
use thiserror::Error;

use self::BiquadFilter::*;
use crate::transforms::*;
pub use crate::util::seed_from_u64;
use crate::util::*;
use crate::*;

type Result<T> = std::result::Result<T, AugmentationError>;

#[derive(Error, Debug)]
pub enum AugmentationError {
    #[error("DF UtilsError")]
    UtilsError(#[from] UtilsError),
    #[error("DF Transforms Error")]
    TransformError(#[from] crate::transforms::TransformError),
    #[error("Wrong input")]
    WrongInput,
    #[error("Transform {transform} not initalized: {msg}")]
    NotInitialized { transform: String, msg: String },
    #[error("DF error: {0}")]
    DfError(String),
    #[error("Ndarray Shape Error")]
    NdarrayShapeError(#[from] ndarray::ShapeError),
    #[error("Wav Reader Error")]
    WavReadError(#[from] crate::wav_utils::WavUtilsError),
}

pub enum TransformInput<'a> {
    Audio(&'a mut Array2<f32>),
    Spectrum(&'a mut Array3<Complex32>),
}
impl<'a> From<&'a mut Array2<f32>> for TransformInput<'a> {
    fn from(audio: &'a mut Array2<f32>) -> Self {
        TransformInput::Audio(audio)
    }
}
impl<'a> From<&'a mut Array3<Complex32>> for TransformInput<'a> {
    fn from(spec: &'a mut Array3<Complex32>) -> Self {
        TransformInput::Spectrum(spec)
    }
}

pub trait Transform {
    fn transform(&self, x: &mut TransformInput) -> Result<()>;
    fn default_with_prob(p: f32) -> Self
    where
        Self: Sized;
    fn box_clone(&self) -> Box<dyn Transform + Send>;
    fn name(&self) -> &str;
}

impl Clone for Box<dyn Transform> {
    fn clone(&self) -> Box<dyn Transform> {
        self.box_clone()
    }
}

pub struct Compose {
    pub transforms: Vec<Box<dyn Transform + Send>>,
    log_timings: bool,
}
unsafe impl Send for Compose {}
unsafe impl Sync for Compose {}

impl Compose {
    pub fn new(transforms: Vec<Box<dyn Transform + Send>>) -> Self {
        Compose {
            transforms,
            log_timings: false,
        }
    }

    pub fn log_timings(&mut self) {
        self.log_timings = true
    }

    pub fn push(&mut self, t: Box<dyn Transform + Send>) {
        self.transforms.push(t);
    }

    pub fn transform(&self, x: &mut TransformInput) -> Result<()> {
        #[cfg(feature = "timings")]
        let mut t0 = Instant::now();
        #[cfg(feature = "timings")]
        let mut timings = Vec::new();
        for t in self.transforms.iter() {
            match t.transform(x) {
                Ok(()) => (),
                Err(e) => log::error!("{:?}", e),
            };
            #[cfg(feature = "timings")]
            {
                let t1 = Instant::now();
                let d = (t1 - t0).as_micros();
                if d > 10 {
                    timings.push(format!("{}: {} ms", t.name(), d / 1000));
                }
                t0 = t1;
            }
        }
        #[cfg(feature = "timings")]
        if log::log_enabled!(log::Level::Trace) && !timings.is_empty() {
            log::trace!(
                "Calculated augmentation transforms in {:?}",
                timings.join(", ")
            );
        }
        Ok(())
    }
    pub fn len(&self) -> usize {
        self.transforms.len()
    }
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}
impl Clone for Compose {
    fn clone(&self) -> Self {
        Compose {
            transforms: self.transforms.iter().map(|t| t.box_clone()).collect(),
            log_timings: self.log_timings,
        }
    }
}

// Adopted from RNNoise/PercepNet
#[derive(Clone)]
pub struct RandLFilt {
    prob: f32,
    uniform: Uniform<f32>,
}
impl RandLFilt {
    pub fn new(p: f32, a: f32, b: f32) -> Self {
        let uniform = Uniform::new_inclusive(a, b);
        RandLFilt { prob: p, uniform }
    }
    fn sample_ab(&self) -> Result<[f32; 2]> {
        let mut rng = thread_rng()?;
        Ok([rng.sample(self.uniform), rng.sample(self.uniform)])
    }
}
impl Transform for RandLFilt {
    fn transform(&self, inp: &mut TransformInput) -> Result<()> {
        if self.prob == 0. || (self.prob < 1. && thread_rng()?.uniform(0f32, 1f32) > self.prob) {
            return Ok(());
        }
        let a: [f32; 2] = self.sample_ab()?;
        let b: [f32; 2] = self.sample_ab()?;
        let mut mem = [0f32; 2];
        let x = match inp {
            TransformInput::Spectrum(_) => return Err(AugmentationError::WrongInput),
            TransformInput::Audio(a) => a,
        };
        for x_ch in x.axis_iter_mut(Axis(0)) {
            biquad_norm_inplace(x_ch, &mut mem, &b, &a);
        }
        Ok(())
    }
    fn default_with_prob(p: f32) -> Self {
        Self::new(p, -3. / 8., 3. / 8.)
    }
    fn box_clone(&self) -> Box<dyn Transform + Send> {
        Box::new((*self).clone())
    }
    fn name(&self) -> &str {
        "RandLFilt"
    }
}

fn high_shelf(center_freq: f32, gain_db: f32, q_factor: f32, sr: usize) -> ([f32; 3], [f32; 3]) {
    let w0 = 2. * std::f32::consts::PI * center_freq / sr as f32;
    let amp = 10f32.powf(gain_db / 40.);
    let alpha = w0.sin() / 2. / q_factor;

    let b0 = amp * ((amp + 1.) + (amp - 1.) * w0.cos() + 2. * amp.sqrt() * alpha);
    let b1 = -2. * amp * ((amp - 1.) + (amp + 1.) * w0.cos());
    let b2 = amp * ((amp + 1.) + (amp - 1.) * w0.cos() - 2. * amp.sqrt() * alpha);
    let a0 = (amp + 1.) - (amp - 1.) * w0.cos() + 2. * amp.sqrt() * alpha;
    let a1 = 2. * ((amp - 1.) - (amp + 1.) * w0.cos());
    let a2 = (amp + 1.) - (amp - 1.) * w0.cos() - 2. * amp.sqrt() * alpha;
    ([b0, b1, b2], [a0, a1, a2])
}
fn high_pass(center_freq: f32, q_factor: f32, sr: usize) -> ([f32; 3], [f32; 3]) {
    let w0 = 2. * std::f32::consts::PI * center_freq / sr as f32;
    let alpha = w0.sin() / 2. / q_factor;

    let b0 = (1. + w0.cos()) / 2.;
    let b1 = -(1. + w0.cos());
    let b2 = (1. + w0.cos()) / 2.;
    let a0 = 1. + alpha;
    let a1 = -2. * w0.cos();
    let a2 = 1. - alpha;
    ([b0, b1, b2], [a0, a1, a2])
}
fn low_shelf(center_freq: f32, gain_db: f32, q_factor: f32, sr: usize) -> ([f32; 3], [f32; 3]) {
    let w0 = 2. * std::f32::consts::PI * center_freq / sr as f32;
    let amp = 10f32.powf(gain_db / 40.);
    let alpha = w0.sin() / 2. / q_factor;

    let b0 = amp * ((amp + 1.) - (amp - 1.) * w0.cos() + 2. * amp.sqrt() * alpha);
    let b1 = 2. * amp * ((amp - 1.) - (amp + 1.) * w0.cos());
    let b2 = amp * ((amp + 1.) - (amp - 1.) * w0.cos() - 2. * amp.sqrt() * alpha);
    let a0 = (amp + 1.) + (amp - 1.) * w0.cos() + 2. * amp.sqrt() * alpha;
    let a1 = -2. * ((amp - 1.) + (amp + 1.) * w0.cos());
    let a2 = (amp + 1.) + (amp - 1.) * w0.cos() - 2. * amp.sqrt() * alpha;
    ([b0, b1, b2], [a0, a1, a2])
}
pub fn low_pass(center_freq: f32, q_factor: f32, sr: usize) -> ([f32; 3], [f32; 3]) {
    let w0 = 2. * std::f32::consts::PI * center_freq / sr as f32;
    let alpha = w0.sin() / 2. / q_factor;

    let b0 = (1. - w0.cos()) / 2.;
    let b1 = 1. - w0.cos();
    let b2 = b0;
    let a0 = 1. + alpha;
    let a1 = -2. * w0.cos();
    let a2 = 1. - alpha;
    ([b0, b1, b2], [a0, a1, a2])
}
fn peaking_eq(center_freq: f32, gain_db: f32, q_factor: f32, sr: usize) -> ([f32; 3], [f32; 3]) {
    let w0 = 2. * std::f32::consts::PI * center_freq / sr as f32;
    let amp = 10f32.powf(gain_db / 40.);
    let alpha = w0.sin() / 2. / q_factor;

    let b0 = 1. + alpha * amp;
    let b1 = -2. * w0.cos();
    let b2 = 1. - alpha * amp;
    let a0 = 1. + alpha / amp;
    let a1 = -2. * w0.cos();
    let a2 = 1. - alpha / amp;
    ([b0, b1, b2], [a0, a1, a2])
}
fn notch(center_freq: f32, q_factor: f32, sr: usize) -> ([f32; 3], [f32; 3]) {
    let w0 = 2. * std::f32::consts::PI * center_freq / sr as f32;
    let alpha = w0.sin() / 2. / q_factor;

    let b0 = 1.;
    let b1 = -2. * w0.cos();
    let b2 = 1.;
    let a0 = 1. + alpha;
    let a1 = -2. * w0.cos();
    let a2 = 1. - alpha;
    ([b0, b1, b2], [a0, a1, a2])
}
fn biquad_filter(x: &mut Array2<f32>, b: &[f32; 3], a: &[f32; 3]) {
    for x_ch in x.axis_iter_mut(Axis(0)) {
        let mut mem = [0.; 2];
        biquad_inplace(x_ch, &mut mem, b, a);
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum BiquadFilter {
    HighShelf,
    LowShelf,
    HighPass,
    LowPass,
    PeakingEQ,
    Notch,
}
impl BiquadFilter {
    pub fn iterator() -> impl Iterator<Item = BiquadFilter> {
        [HighShelf, LowShelf, HighPass, LowPass, PeakingEQ, Notch].iter().copied()
    }
}

/// Apply random biquad filters based on https://www.w3.org/TR/audio-eq-cookbook/
///
/// # Available filters:
///  * LowPass
///  * LowShelf
///  * HighPass
///  * HighShelf
///  * PeakingEQ
///  * Notch
#[derive(Clone)]
pub struct RandBiquadFilter {
    prob: f32,
    sr: Option<usize>,
    n_freqs: usize,
    gain_db_low: f32,
    gain_db_high: f32,
    q_low: f32,
    q_high: f32,
    filters: Vec<BiquadFilter>,
    equalize_rms: bool,
}
impl RandBiquadFilter {
    pub fn with_sr(mut self, sr: usize) -> Self {
        self.sr = Some(sr);
        self
    }
    pub(crate) fn apply(
        &self,
        x: &mut Array2<f32>,
        filter: BiquadFilter,
        freq: f32,
        q: f32,
        gain_db: Option<f32>,
    ) {
        let sr = self.sr.unwrap();
        if log::log_enabled!(log::Level::Trace) {
            log::trace!(
                "Augmentation RandBiquadFilter (filter: {:?}, freq: {}, q: {}, db: {})",
                filter,
                freq,
                q,
                gain_db.unwrap_or_default()
            );
        }
        let (b, a) = match filter {
            HighShelf => high_shelf(freq, gain_db.unwrap(), q, sr),
            LowShelf => low_shelf(freq, gain_db.unwrap(), q, sr),
            HighPass => high_pass(freq, q, sr),
            LowPass => low_pass(freq, q, sr),
            PeakingEQ => peaking_eq(freq, gain_db.unwrap(), q, sr),
            Notch => notch(freq, q, sr),
        };
        biquad_filter(x, &b, &a);
    }
}
impl Transform for RandBiquadFilter {
    fn transform(&self, x: &mut TransformInput) -> Result<()> {
        let x = match x {
            TransformInput::Spectrum(_) => return Err(AugmentationError::WrongInput),
            TransformInput::Audio(a) => a,
        };
        if self.sr.is_none() {
            return Err(AugmentationError::NotInitialized {
                transform: "RandBiquadFilter".into(),
                msg: "No sampling rate provided.".into(),
            });
        }
        let mut rng = thread_rng()?;
        if self.prob == 0. || (self.prob < 1. && rng.uniform(0f32, 1f32) > self.prob) {
            return Ok(());
        }
        let rms = x.map(|&x| x.powi(2)).mean().unwrap().sqrt();
        for _ in 0..rng.uniform_inclusive(1, self.n_freqs) {
            let filter = self.filters.choose(&mut rng).unwrap();
            let (f_low, f_high) = match filter {
                LowPass => (4000, 8000),
                HighShelf => (1000, 8000),
                HighPass => (40, 400),
                LowShelf => (40, 1000),
                _ => (40, 4000),
            };
            let freq = rng.log_uniform(f_low as f32, f_high as f32);
            let gain_db = rng.uniform_inclusive(self.gain_db_low, self.gain_db_high);
            let q = rng.uniform_inclusive(self.q_low, self.q_high);
            self.apply(x, *filter, freq, q, Some(gain_db));
        }
        if self.equalize_rms {
            let rms_new = x.map(|&x| x.powi(2)).mean().unwrap().sqrt();
            x.mapv_inplace(|s| s * rms / rms_new);
        }
        // Guard against clipping
        let max = find_max_abs(x.iter()).unwrap();
        if (max - 1.) > 1e-10 {
            let f = 1. / (max + 1e-10);
            log::debug!(
                "RandBiquadFilter: Clipping detected. Reducing gain by: {}",
                max
            );
            x.mapv_inplace(|s| s * f);
        }
        Ok(())
    }
    fn default_with_prob(p: f32) -> Self {
        RandBiquadFilter {
            prob: p,
            sr: None,
            n_freqs: 3,
            gain_db_high: 15.,
            gain_db_low: -15.,
            q_low: 0.5,
            q_high: 1.5,
            filters: BiquadFilter::iterator().collect(),
            equalize_rms: true,
        }
    }
    fn box_clone(&self) -> Box<dyn Transform + Send> {
        Box::new((*self).clone())
    }
    fn name(&self) -> &str {
        "RandBiquadFilter"
    }
}

#[derive(Clone)]
pub struct RandResample {
    prob: f32,
    sr: Option<usize>,
    r_low: f32,
    r_high: f32,
    chunk_size: usize,
}
impl RandResample {
    pub fn new(p: f32, sr: usize, r_low: f32, r_high: f32, chunk_size: usize) -> Self {
        RandResample {
            prob: p,
            sr: Some(sr),
            r_low,
            r_high,
            chunk_size,
        }
    }
    pub fn with_sr(mut self, sr: usize) -> Self {
        self.sr = Some(sr);
        self
    }
}
impl Transform for RandResample {
    fn transform(&self, x: &mut TransformInput) -> Result<()> {
        let x = match x {
            TransformInput::Spectrum(_) => return Err(AugmentationError::WrongInput),
            TransformInput::Audio(a) => a,
        };
        if self.sr.is_none() {
            return Err(AugmentationError::NotInitialized {
                transform: "RandEQ".into(),
                msg: "No sampling rate provided.".into(),
            });
        }
        let mut rng = thread_rng()?;
        let sr = self.sr.unwrap();
        if self.prob == 0. || (self.prob < 1. && rng.uniform(0f32, 1f32) > self.prob) {
            return Ok(());
        }
        let ch = x.len_of(Axis(0));
        let len = x.len_of(Axis(1));
        let new_sr = rng.uniform_inclusive(self.r_low, self.r_high) * sr as f32;
        // round so we get a better gcd
        let new_sr = ((new_sr / 500.).round() * 500.) as usize;
        if new_sr == sr {
            return Ok(());
        }
        let out = resample(x.view(), sr, new_sr, Some(self.chunk_size))?;
        let new_len = out.len_of(Axis(1));
        if new_len > len {
            x.append(Axis(1), Array2::zeros((ch, new_len - len)).view())?;
        } else {
            x.slice_axis_inplace(Axis(1), Slice::from(0..new_len));
        }
        x.clone_from(&out);
        // out.move_into(x);
        Ok(())
    }
    fn default_with_prob(p: f32) -> Self {
        RandResample {
            prob: p,
            sr: None,
            r_low: 0.9,
            r_high: 1.1,
            chunk_size: 1024,
        }
    }
    fn box_clone(&self) -> Box<dyn Transform + Send> {
        Box::new((*self).clone())
    }
    fn name(&self) -> &str {
        "RandResample"
    }
}

#[derive(Clone)]
pub struct RandClipping {
    prob: f32,
    db_range: Option<Range<f32>>,
    c_range: Option<Range<f32>>,
    eps: f32,
    eps_c: f32,
}
impl RandClipping {
    pub fn new(p: f32, eps: f32, convergence_eps: f32) -> Self {
        RandClipping {
            prob: p,
            db_range: None,
            c_range: None,
            eps,
            eps_c: convergence_eps,
        }
    }
    pub fn with_snr(mut self, db_range: Range<f32>) -> Self {
        self.db_range = Some(db_range);
        self.c_range = None;
        self
    }
    pub fn with_c(mut self, c_range: Range<f32>) -> Self {
        self.db_range = None;
        self.c_range = Some(c_range);
        self
    }
    fn clip_inplace(&self, x: &mut Array2<f32>, c: f32) {
        x.mapv_inplace(|x| x.clamp(-c, c))
    }
    fn clip(&self, x: ArrayView2<f32>, c: f32) -> Array2<f32> {
        x.map(|x| x.clamp(-c, c))
    }
    pub fn sdr(&self, orig: ArrayView2<f32>, processed: ArrayView2<f32>) -> f32 {
        debug_assert_eq!(orig.shape(), processed.shape());
        let numel = orig.len();
        debug_assert!(numel > 0);
        let noise = orig.to_owned() - processed;
        let a = orig.fold(0., |acc, x| acc + x.powi(2)) / numel as f32;
        let b = noise.fold(0., |acc, x| acc + x.powi(2)) / numel as f32;
        (a / (b + self.eps)).log10() * 20.
    }
    fn find_root(&self, x: ArrayView2<f32>, target_snr: f32, max: Option<f32>) -> Option<f32> {
        let max = max.unwrap_or(1.0);
        let f = |c| self.sdr(x.view(), self.clip(x.view(), c).view()) - target_snr;
        let (a, b) = (0.01 * max, 0.99 * max);
        match roots::find_root_brent(a, b, f, &mut self.eps_c.clone()) {
            Ok(c) => Some(c),
            Err(e) => {
                log::warn!("RandClipping: Failed to find root: {:?}", e);
                dbg!(max, f(0.01 * max), f(0.99 * max));
                None
            }
        }
    }
}
impl Transform for RandClipping {
    fn transform(&self, x: &mut TransformInput) -> Result<()> {
        let x = match x {
            TransformInput::Spectrum(_) => return Err(AugmentationError::WrongInput),
            TransformInput::Audio(a) => a,
        };
        let mut rng = thread_rng()?;
        if self.prob == 0. || (self.prob < 1. && rng.uniform(0f32, 1f32) > self.prob) {
            return Ok(());
        }
        let max = x.fold(0.0, |acc, x| x.abs().max(acc));
        let c = if let Some(db_range) = self.db_range.as_ref() {
            let target_snr = rng.uniform(db_range.start, db_range.end);
            if let Some(c) = self.find_root(x.view(), target_snr, Some(max)) {
                c
            } else {
                return Ok(());
            }
        } else {
            let c_range = self.c_range.as_ref().unwrap();
            rng.uniform_inclusive(c_range.start * max, c_range.end * max)
        };
        if log::log_enabled!(log::Level::Trace) {
            log::trace!("Augmentation RandClipping (c: {})", c);
        }
        self.clip_inplace(x, c);
        Ok(())
    }
    fn default_with_prob(p: f32) -> Self {
        RandClipping {
            prob: p,
            db_range: None,
            c_range: Some(0.01..0.25),
            eps: 1e-10,
            eps_c: 0.001,
        }
    }
    fn box_clone(&self) -> Box<dyn Transform + Send> {
        Box::new((*self).clone())
    }
    fn name(&self) -> &str {
        "RandClipping"
    }
}
#[derive(Clone)]
pub struct RandZeroingTD {
    prob: f32,
    max_percent: f32,
    min_sequential_samples: usize,
    max_sequential_samples: usize,
}
impl RandZeroingTD {
    pub fn with_n_samples(mut self, min: usize, max: usize) -> Self {
        self.min_sequential_samples = min;
        self.max_sequential_samples = max;
        self
    }
    pub fn with_max_percent(mut self, p: f32) -> Self {
        assert!(p < 100.);
        assert!(p >= 1.);
        self.max_percent = p;
        self
    }
}
impl Transform for RandZeroingTD {
    fn transform(&self, x: &mut TransformInput) -> Result<()> {
        let x = match x {
            TransformInput::Spectrum(_) => return Err(AugmentationError::WrongInput),
            TransformInput::Audio(a) => a,
        };
        let mut rng = thread_rng()?;
        if self.prob == 0. || (self.prob < 1. && rng.uniform(0f32, 1f32) > self.prob) {
            return Ok(());
        }
        // Loop as long as we dropped up to `perc` samples
        let a_len = x.len_of(Axis(1));
        let p = rng.uniform(0.01f32, self.max_percent / 100.);
        let mut cur = 0.;
        let min = self.min_sequential_samples;
        let max = self.max_sequential_samples;
        while cur < p {
            let pos = rng.uniform(0, a_len - max);
            let z_len = rng.uniform(min, max);
            x.slice_mut(s![.., pos..pos + z_len]).map_inplace(|s| *s = 0.);
            cur += z_len as f32 / a_len as f32;
        }
        Ok(())
    }
    fn default_with_prob(p: f32) -> Self {
        Self {
            prob: p,
            max_percent: 10.,
            min_sequential_samples: 120,
            max_sequential_samples: 1800,
        }
    }
    fn box_clone(&self) -> Box<dyn Transform + Send> {
        Box::new((*self).clone())
    }
    fn name(&self) -> &str {
        "RandZeroing"
    }
}

#[derive(Clone)]
pub struct RandRemoveDc {
    prob: f32,
}
impl Transform for RandRemoveDc {
    fn transform(&self, x: &mut TransformInput) -> Result<()> {
        let x = match x {
            TransformInput::Spectrum(_) => return Err(AugmentationError::WrongInput),
            TransformInput::Audio(a) => a,
        };
        if self.prob == 0. || (self.prob < 1. && thread_rng()?.uniform(0f32, 1f32) > self.prob) {
            return Ok(());
        }
        let mean = x.sum() / x.len() as f32;
        for x_s in x.iter_mut() {
            *x_s -= mean;
        }
        Ok(())
    }
    fn default_with_prob(p: f32) -> Self {
        RandRemoveDc { prob: p }
    }
    fn box_clone(&self) -> Box<dyn Transform + Send> {
        Box::new((*self).clone())
    }
    fn name(&self) -> &str {
        "RandRemoveDc"
    }
}

pub(crate) fn gen_noise(
    f_decay: f32,
    num_channels: u16,
    num_samples: usize,
    sr: u32,
) -> Result<Array2<f32>> {
    let mut fft = RealFftPlanner::new();
    let fft_forward = fft.plan_fft_forward(sr as usize);
    let fft_inverse = fft.plan_fft_inverse(sr as usize);
    let mut scratch_forward = fft_forward.make_scratch_vec();
    let mut scratch_inverse = fft_inverse.make_scratch_vec();
    gen_noise_with_scratch(
        f_decay,
        num_channels,
        num_samples,
        sr,
        fft_forward.as_ref(),
        fft_inverse.as_ref(),
        &mut scratch_forward,
        &mut scratch_inverse,
    )
}
#[allow(clippy::too_many_arguments)]
fn gen_noise_with_scratch(
    f_decay: f32,
    num_channels: u16,
    num_samples: usize,
    sr: u32,
    fft_forward: &dyn RealToComplex<f32>,
    fft_inverse: &dyn ComplexToReal<f32>,
    scratch_forward: &mut [Complex32],
    scratch_inverse: &mut [Complex32],
) -> Result<Array2<f32>> {
    // Adopted from torch_audiomentations
    let sr = sr as usize;
    let ch = num_channels as usize;
    let mut noise = if f_decay != 0. {
        let mut noise = Array::random((ch, sr), Normal::new(0., 1.).unwrap());
        let spec = Array2::uninit([ch, sr / 2 + 1]);
        // Safety: Will be fully overwritten by fft transform.
        let mut spec = unsafe { spec.assume_init() };
        fft_with_output(&mut noise, fft_forward, scratch_forward, &mut spec)?;
        let mut mask = Array::linspace(1., ((sr / 2 + 1) as f32).sqrt(), spec.len_of(Axis(1)));
        mask.mapv_inplace(|x| x.powf(f_decay));
        spec = spec / mask.to_shape([1, sr / 2 + 1]).unwrap();
        ifft_with_output(&mut spec, fft_inverse, scratch_inverse, &mut noise)?;
        if log::log_enabled!(log::Level::Trace) {
            log::trace!("Generated random noise (f_decay : {})", f_decay);
        }
        noise
    } else {
        // Fast path for white noise
        Array::random((ch, sr), Normal::new(0., 1.).unwrap())
    };
    let f = thread_rng()?.uniform(0.01, 0.95) / find_max_abs(&noise).unwrap().max(1.);
    noise *= f;
    let mut noises = concatenate(
        Axis(1),
        &vec![noise.view(); (num_samples as f32 / sr as f32).ceil() as usize],
    )?;
    noises.slice_axis_inplace(Axis(1), Slice::from(..num_samples));
    Ok(noises)
}

pub(crate) struct NoiseGenerator {
    p: f32,
    sr: u32,
    fft_forward: Arc<dyn RealToComplex<f32>>,
    fft_inverse: Arc<dyn ComplexToReal<f32>>,
}

impl NoiseGenerator {
    pub fn new(sr: usize, p: f32) -> Self {
        let mut fft = RealFftPlanner::new();
        let fft_forward = fft.plan_fft_forward(sr);
        let fft_inverse = fft.plan_fft_inverse(sr);
        NoiseGenerator {
            p,
            sr: sr as u32,
            fft_forward,
            fft_inverse,
        }
    }
    /// Generate a random noise signal.
    ///
    /// # Arguments
    ///
    /// * `f_decay`: Decay variable. Typical values for common noises are:
    ///     - white: `0.0`
    ///     - pink: `1.0`
    ///     - brown: `2.0`
    ///     - blue: `-1.0`
    ///     - purple: `-2.0`
    /// * `num_channels`: Number of output channels.
    /// * `num_samples`: Number of output samples.
    ///
    /// # Returns
    ///
    /// * `noise`: 2D array of shape `(num_channels, num_samples)`.
    pub fn generate(
        &self,
        f_decay: f32,
        num_channels: u16,
        num_samples: usize,
    ) -> Result<Array2<f32>> {
        gen_noise_with_scratch(
            f_decay,
            num_channels,
            num_samples,
            self.sr,
            self.fft_forward.as_ref(),
            self.fft_inverse.as_ref(),
            &mut self.fft_forward.make_scratch_vec(),
            &mut self.fft_inverse.make_scratch_vec(),
        )
    }
    pub fn generate_random_noise(
        &self,
        f_decay_min: f32,
        f_decay_max: f32,
        num_channels: u16,
        num_samples: usize,
    ) -> Result<Option<Array2<f32>>> {
        debug_assert!(f_decay_min < f_decay_max);
        let mut rng = thread_rng()?;
        let f_decay = rng.uniform(f_decay_min, f_decay_max);
        Ok(Some(self.generate(f_decay, num_channels, num_samples)?))
    }
    pub fn maybe_generate_random_noise(
        &self,
        f_decay_min: f32,
        f_decay_max: f32,
        num_channels: u16,
        num_samples: usize,
    ) -> Result<Option<Array2<f32>>> {
        debug_assert!(f_decay_min < f_decay_max);
        let mut rng = thread_rng()?;
        if self.p == 0. || self.p < rng.uniform(0., 1.) {
            return Ok(None);
        }
        self.generate_random_noise(f_decay_min, f_decay_max, num_channels, num_samples)
    }
}

pub(crate) struct RandReverbSim {
    prob_speech: f32,
    prob_noise: f32,
    prob_resample: f32,
    prob_decay: f32,
    sr: usize,
    rt60: f32,
    offset_late: usize,
    drr_f: Option<f32>, // Direct-to-Reverberant-Ratio
}
impl RandReverbSim {
    fn supress_late(
        &self,
        mut rir: Array2<f32>,
        sr: usize,
        offset: usize,
        rt60: f32,
    ) -> Result<Array2<f32>> {
        let len = rir.len_of(Axis(1));
        let mut decay: Array2<f32> = Array2::ones((1, len));
        let dt = 1. / sr as f32;
        let rt60_level = 10f32.powi(-60 / 20);
        let tau = -rt60 / rt60_level.log10();
        if offset >= len {
            return Ok(rir);
        }
        decay.slice_mut(s![0, offset..]).assign(&Array1::from_iter(
            (0..(len - offset)).map(|v| 10f32.powf(-(v as f32) * dt / tau)),
        ));
        rir = rir * decay;
        Ok(rir)
    }
    /// Trim the RIR based on the maximum reference level -80dB.
    ///
    /// Returns the trimed RIR and the index of the absolute maximum used as reference level.
    fn trim(&self, mut rir: Array2<f32>, ref_idx: usize) -> Result<Array2<f32>> {
        let min_db = -80.;
        let len = rir.len_of(Axis(1));
        let rir_mono = rir.mean_axis(Axis(0)).unwrap();
        let ref_level: f32 = rir_mono[ref_idx];
        let min_level = 10f32.powf((min_db + ref_level.log10() * 20.) / 20.);
        let mut idx = len;
        for (i, v) in rir_mono.iter().rev().enumerate() {
            if v.abs() < min_level {
                idx = len - i;
            } else {
                break;
            }
        }
        rir.slice_collapse(s![.., ..idx]);
        Ok(rir)
    }
    fn good_fft_size(&self, len: usize) -> usize {
        // Zero pad RIR for better FFT efficiency by finding prime factors up to a limit of 11.
        let mut missing = len;
        let primes = [2, 3, 5, 7, 11];
        let mut factors = [0u32; 5];
        for (p, f) in primes.iter().zip(factors.iter_mut()) {
            while missing % p == 0 {
                missing /= p;
                *f += 1;
            }
        }
        if missing > 1 {
            factors[0] += (missing as f32).log2().ceil() as u32;
        }
        let fft_size = primes.iter().zip(factors).fold(1, |acc, (p, f)| acc * p.pow(f));
        debug_assert!(fft_size >= len);
        fft_size
    }
    fn pad(&self, x: &mut Array2<f32>, pad_front: usize, pad_back: usize) -> Result<()> {
        if pad_front == 0 && pad_back == 0 {
            return Ok(());
        }
        let ch = x.len_of(Axis(0));
        x.append(Axis(1), Array2::zeros((ch, pad_front + pad_back)).view())?;
        if pad_front > 0 {
            for mut x_ch in x.outer_iter_mut() {
                x_ch.as_slice_memory_order_mut().unwrap().rotate_right(pad_front);
            }
        }
        Ok(())
    }
    pub fn transform_single(&self, sample: &mut Array2<f32>, mut rir: Array2<f32>) -> Result<()> {
        let mut fft_t = FftTransform::new();
        let rir_mono = rir.mean_axis(Axis(0)).unwrap();
        let max_idx = argmax_abs(rir_mono.iter()).unwrap();
        // Normalize and flip RIR for convolution
        rir = self.trim(rir, max_idx)?;
        let rir_e = rir.map(|v| v * v).sum().sqrt();
        let rir = rir / rir_e;
        self.convolve(sample, rir, &mut fft_t, None)?;
        Ok(())
    }
    /// Applies random reverberation to either noise or speech or both.
    ///
    /// We have 3 scenarious:
    ///
    /// 1. Only noise will get some reverberation. No `speech_rev` will be returned.
    /// 2. Only speech will get some reverberation. The return value`speech_rev` will contain the
    ///    reverberant speech to be used for generating a noisy mixture and `speech` will be
    ///    modified inplace to be a less reverberant version of `speech_rev` to be used as training
    ///    target.
    /// 3. Speech and noise will get reverberation.
    ///
    /// # Arguments
    ///
    /// * `speech` - A speech signal of shape `[C, N]`. Will be modified in place.
    /// * `noise` - A noise signal of shape `[C, N]`. Will be modified in place.
    /// * `rir_callback` - A callback which will generate a room impulse response.
    ///
    /// # Returns
    ///
    /// * `speech_rev` - An optional reverberant speech sample for mixing. This will contain a
    ///                  more reverberation then the in place modified `speech` signal.
    pub fn transform<F>(
        &self,
        speech: &mut Array2<f32>,
        noise: &mut Array2<f32>,
        rir_callback: F,
    ) -> Result<Option<Array2<f32>>>
    where
        F: FnOnce() -> std::result::Result<Array2<f32>, Box<dyn std::error::Error>>,
    {
        if self.prob_noise == 0. && self.prob_speech == 0. {
            return Ok(None);
        }
        let mut rng = thread_rng()?;
        let apply_speech = self.prob_speech > rng.uniform(0f32, 1f32);
        let apply_noise = self.prob_noise > rng.uniform(0f32, 1f32);
        if !(apply_speech || apply_noise) {
            return Ok(None);
        }
        #[cfg(feature = "timings")]
        let t0 = Instant::now();
        let mut fft_t = FftTransform::new();
        // Get room impulse response
        let mut rir = match rir_callback() {
            Ok(r) => r,
            Err(e) => {
                return Err(AugmentationError::DfError(format!(
                    "Error getting RIR in RandReverbSim::transform() {e:?}"
                )));
            }
        };
        let orig_len = speech.len_of(Axis(1));
        // Maybe resample RIR as augmentation
        if self.prob_resample > rng.uniform(0f32, 1f32) {
            let new_sr: f32 = rng.uniform(0.8, 1.2) * self.sr as f32;
            let new_sr = ((new_sr / 500.).round() * 500.) as usize;
            rir = resample(rir.view(), self.sr, new_sr, Some(512))?;
        }
        let rir_mono = rir.mean_axis(Axis(0)).unwrap();
        let max_idx = argmax_abs(rir_mono.iter()).unwrap();
        if self.prob_decay > rng.uniform(0f32, 1f32) {
            let rt60 = rng.uniform(0.2, 1.);
            rir = self.supress_late(rir, self.sr, max_idx, rt60)?;
        }
        rir = self.trim(rir, max_idx)?;
        // Normalize and flip RIR for convolution
        let rir_e = rir.map(|v| v * v).sum().sqrt();
        let rir_noise = rir / rir_e;

        // speech_rev contains reverberant speech for mixing with noise
        let speech_rev = if apply_speech {
            let speech_rms = rms(speech.iter());
            // self.pad(speech, pad_front, pad_back)?; // Pad since STFT will truncate at the end
            let mut speech_rev = speech.clone();
            self.convolve(
                &mut speech_rev,
                rir_noise.clone(),
                &mut fft_t,
                Some(orig_len),
            )?;
            // Speech should be a slightly dereverberant signal as target
            // TODO: Make dereverberation parameters configurable.
            //
            // Add extra offset since these are releveant for speech intelligibility
            let offset = max_idx + self.offset_late * self.sr / 1000;
            let mut rir_speech =
                self.supress_late(rir_noise.clone(), self.sr, offset, self.rt60)?;
            let rir_e = rir_speech.map(|v| v * v).sum().sqrt();
            rir_speech *= 1. / rir_e;
            // Generate target speech signal containing less reverberation
            let mut speech_little_rev = speech.clone();
            self.convolve(
                &mut speech_little_rev,
                rir_speech,
                &mut fft_t,
                Some(orig_len),
            )?;
            // Maybe mix in some original clean speech as target
            if let Some(f) = self.drr_f {
                // speech.slice_axis_inplace(Axis(1), Slice::from(pad_front..pad_front + orig_len));
                *speech *= f;
                speech.scaled_add(1. - f, &speech_little_rev);
            } else {
                *speech = speech_little_rev;
            }
            let speech_rms_after = rms(speech.iter());
            *speech *= speech_rms / (speech_rms_after + 1e-10);
            debug_assert_eq!(speech.shape(), speech_rev.shape());
            debug_assert_eq!(speech.len_of(Axis(1)), noise.len_of(Axis(1)));
            Some(speech_rev)
        } else {
            None
        };
        if apply_noise {
            // Noisy contains reverberant noise
            self.convolve(noise, rir_noise, &mut fft_t, Some(orig_len))?;
            debug_assert_eq!(speech.len_of(Axis(1)), noise.len_of(Axis(1)));
        }
        #[cfg(feature = "timings")]
        if log::log_enabled!(log::Level::Trace) {
            log::trace!("Calculated RandReverbSim in {:?}", Instant::now() - t0);
        }
        Ok(speech_rev)
    }
    fn convolve(
        &self,
        x: &mut Array2<f32>,
        mut rir: Array2<f32>,
        fft_transform: &mut FftTransform,
        truncate: Option<usize>,
    ) -> Result<()> {
        let x_len = x.len_of(Axis(1));
        let rir_len = rir.len_of(Axis(1));
        let fft_size = self.good_fft_size(rir.len_of(Axis(1)) + x.len_of(Axis(1)) - 1);
        let forward = fft_transform.planer.plan_fft_forward(fft_size);
        let inverse = fft_transform.planer.plan_fft_inverse(fft_size);
        self.pad(x, 0, fft_size - x_len)?;
        self.pad(&mut rir, 0, fft_size - rir_len)?;
        let mut x_fd = fft(x, forward.as_ref(), &mut fft_transform.scratch)?;
        let rir_fd = fft(&mut rir, forward.as_ref(), &mut fft_transform.scratch)?;
        x_fd = x_fd * rir_fd / fft_size as f32;
        ifft_with_output(&mut x_fd, inverse.as_ref(), &mut fft_transform.scratch, x)?;
        let max_len = truncate.unwrap_or(x_len);
        debug_assert!(max_len <= x_len);
        x.slice_collapse(s![.., ..max_len]);
        Ok(())
    }
    pub fn new(p: f32, sr: usize) -> Self
    where
        Self: Sized,
    {
        RandReverbSim {
            prob_speech: p,
            prob_noise: p,
            prob_resample: p,
            prob_decay: p.max(0.5),
            sr,
            rt60: 0.5,
            offset_late: 20,
            drr_f: None,
        }
    }
    // Include the original signal within the target by specifying the Direct-to-Reverberant ratio
    // in [dB].
    pub fn with_drr(mut self, f: f32) -> Self {
        assert!((0.0..=1.0).contains(&f));
        self.drr_f = Some(f);
        self
    }
    pub fn with_rt60(mut self, rt60: f32) -> Self {
        assert!(rt60 > 0.);
        self.rt60 = rt60;
        self
    }
    pub fn with_offset_late_reflections(mut self, offset: usize) -> Self {
        self.offset_late = offset;
        self
    }
    pub fn with_prob_resample(mut self, p: f32) -> Self {
        self.prob_resample = p;
        self
    }
    pub fn with_prob_decay(mut self, p: f32) -> Self {
        self.prob_decay = p;
        self
    }
}

#[derive(Clone)]
pub(crate) struct BandwidthLimiterAugmentation {
    prob: f32,
    sr: usize,
    cut_off_freqs: Vec<usize>,
}

impl BandwidthLimiterAugmentation {
    pub fn name(&self) -> &str {
        "BandwidthLimiter"
    }
    pub fn new(p: f32, sr: usize) -> Self {
        BandwidthLimiterAugmentation {
            prob: p,
            sr,
            cut_off_freqs: vec![4000, 6000, 8000, 10000, 12000, 16000, 20000, 22050],
        }
    }
    pub fn transform(&self, audio: &mut Array2<f32>, max_freq: usize) -> Result<usize> {
        #[cfg(feature = "timings")]
        let t0 = Instant::now();
        let mut rng = thread_rng()?;
        let &f = self.cut_off_freqs.iter().filter(|&f| *f < max_freq).choose(&mut rng).unwrap();
        let d = low_pass_resample(audio.view(), f, self.sr).unwrap();
        audio.clone_from(&d);
        #[cfg(feature = "timings")]
        if log::log_enabled!(log::Level::Trace) {
            log::trace!(
                "Calculated BandwidthLimiterAugmentation in {:?}",
                Instant::now() - t0
            );
        }
        Ok(f)
    }
}

/// Implement a low pass filterbank in frequency domain. Adopted from audiomentations.
/// Absorption coefs based on pyroomacoustics.
///
/// Absorption is given by:
///
///    `att = exp(- distance * absorption_coefficient)`
#[derive(Clone)]
pub(crate) struct AirAbsorptionAugmentation {
    prob: f32,
    sr: Option<usize>,
    pub air_absorption: BTreeMap<String, [f32; 9]>,
    center_freqs: [usize; 9],
    distance_low: f32,
    distance_high: f32,
}
/// Concat 3 slices into a Vec.
pub fn concat3<T: Clone>(a: &[T], b: &[T], c: &[T]) -> Vec<T> {
    [a, b, c].concat()
}
/// Linear interpolation between two points at x=0 and x=1
fn interp_lin(x: f32, yvals: &[f32; 2]) -> f32 {
    (1. - x) * yvals[0] + x * yvals[1]
}
impl AirAbsorptionAugmentation {
    fn insert_coefs(map: &mut BTreeMap<String, [f32; 9]>, key: &str, scaled_coefs: [f32; 9]) {
        map.insert(key.to_string(), scaled_coefs.map(|x| x * 1e-3));
    }
    pub fn new(sr: usize, prob: f32) -> Self {
        let center_freqs = [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 24000];
        let mut air_absorption = BTreeMap::new();
        Self::insert_coefs(
            &mut air_absorption,
            "10C_30-50%",
            [0.1, 0.2, 0.5, 1.1, 2.7, 9.4, 29.0, 91.5, 289.0],
        );
        Self::insert_coefs(
            &mut air_absorption,
            "10C_50-70%",
            [0.1, 0.2, 0.5, 0.8, 1.8, 5.9, 21.1, 76.6, 280.2],
        );
        Self::insert_coefs(
            &mut air_absorption,
            "10C_70-90%",
            [0.1, 0.2, 0.5, 0.7, 1.4, 4.4, 15.8, 58.0, 214.9],
        );
        Self::insert_coefs(
            &mut air_absorption,
            "20C_30-50",
            [0.1, 0.3, 0.6, 1.0, 1.9, 5.8, 20.3, 72.3, 259.9],
        );
        Self::insert_coefs(
            &mut air_absorption,
            "20C_50-70%",
            [0.1, 0.3, 0.6, 1.0, 1.7, 4.1, 13.5, 44.4, 148.7],
        );
        Self::insert_coefs(
            &mut air_absorption,
            "20C_70-90%",
            [0.1, 0.3, 0.6, 1.1, 1.7, 3.5, 10.6, 31.2, 93.8],
        );
        // The following coefficients are artificial to produce strong absorption
        Self::insert_coefs(
            &mut air_absorption,
            "Strong-High-1",
            [0.1, 0.2, 0.7, 1.5, 3.9, 8.1, 21.6, 80.2, 213.1],
        );
        Self::insert_coefs(
            &mut air_absorption,
            "Strong-High-2",
            [0.1, 0.3, 0.9, 3.8, 8.9, 21.1, 44.6, 80.2, 153.1],
        );
        Self {
            center_freqs,
            air_absorption,
            distance_low: 1.0,
            distance_high: 20.0,
            sr: Some(sr),
            prob,
        }
    }
    /// Interpolate frequency attenuation from center bands stft frequency bins.
    ///
    /// Args:
    ///   - `atten_vals`: Attenuation values of shape [8]
    ///   - `n_freqs`: Number of stft frequency bins.
    fn interp_atten(&self, atten_vals: &[f32], n_freqs: usize) -> Array1<f32> {
        let atten_vals = concat3(&[atten_vals[0]], atten_vals, &[atten_vals[8]]);
        let sr = self.sr.unwrap();
        let freqs = Array1::linspace(0., (sr / 2) as f32, n_freqs);
        let mut atten_vals_interp = Array1::zeros(n_freqs);
        let center_freqs = concat3(&[0], &self.center_freqs, &[sr / 2]);
        let mut i = 0;
        for (c, a) in center_freqs.windows(2).zip(atten_vals.windows(2)) {
            let (c0, c1) = (c[0] as f32, c[1] as f32);
            let (a0, a1) = (a[0], a[1]);
            while i < n_freqs && freqs[i] <= c1 {
                let x = (freqs[i] - c1) / (c0 - c1);
                atten_vals_interp[i] = a0 * x + a1 * (1. - x);
                i += 1;
            }
        }
        atten_vals_interp
    }
    /// Get available absorption coefficient keys
    pub fn keys(&self) -> Vec<String> {
        self.air_absorption.keys().map(|k| k.to_owned()).collect()
    }
    /// Get absorption coefficients for a predifined key
    pub fn get_coefs(&self, key: &str) -> Option<&[f32; 9]> {
        self.air_absorption.get(key)
    }
    pub fn apply(&self, spec: &mut Array3<Complex32>, coefs: &[f32; 9], d: f32) {
        #[cfg(feature = "timings")]
        let t0 = Instant::now();
        let atten_vals = coefs.map(|c| (-d * c).exp());
        let n_freqs = spec.len_of(Axis(2));
        let atten_vals = self.interp_atten(&atten_vals, n_freqs);
        for (mut f, a) in spec.axis_iter_mut(Axis(2)).zip(atten_vals) {
            f.mapv_inplace(|x| x.scale(a));
        }
        #[cfg(feature = "timings")]
        if log::log_enabled!(log::Level::Trace) {
            log::trace!(
                "Calculated AirAbsorptionAugmentation in {:?}",
                Instant::now() - t0
            );
        }
    }
}

impl Transform for AirAbsorptionAugmentation {
    fn default_with_prob(p: f32) -> Self {
        Self::new(0, p)
    }
    fn box_clone(&self) -> Box<dyn Transform + Send> {
        Box::new((*self).clone())
    }
    fn name(&self) -> &str {
        "RandLFilt"
    }
    fn transform(&self, inp: &mut TransformInput) -> Result<()> {
        let spec = match inp {
            TransformInput::Spectrum(s) => s,
            TransformInput::Audio(_) => return Err(AugmentationError::WrongInput),
        };
        // Spec shape: [C, T, F]
        let mut rng = thread_rng()?;
        if self.prob == 0. || (self.prob < 1. && rng.uniform(0f32, 1f32) > self.prob) {
            return Ok(());
        }
        let d = rng.uniform_inclusive(self.distance_low, self.distance_high);
        let coefs = self.air_absorption.iter().choose(&mut rng).unwrap();
        self.apply(spec, coefs.1, d);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Once;

    use super::*;
    use crate::wav_utils::*;

    static INIT: Once = Once::new();

    /// Setup function that is only run once, even if called multiple times.
    fn setup() -> (Array2<f32>, usize) {
        seed_from_u64(42);
        create_out_dir().expect("Could not create output directory");

        INIT.call_once(|| {
            let _ = env_logger::builder()
                // Include all events in tests
                .filter_module("df", log::LevelFilter::max())
                // Ensure events are captured by `cargo test`
                .is_test(true)
                // Ignore errors initializing the logger if tests race to configure it
                .try_init();
        });
        let reader = ReadWav::new("../assets/clean_freesound_33711.wav").unwrap();
        let sr = reader.sr;
        let test_sample = reader.samples_arr2().unwrap();
        (test_sample, sr)
    }

    fn create_out_dir() -> std::io::Result<()> {
        match std::fs::create_dir("../out") {
            Err(ref e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
            r => r,
        }
    }

    #[test]
    pub fn test_compose() -> Result<()> {
        let (test_sample, sr) = setup();
        let ch = test_sample.len_of(Axis(0)) as u16;
        let transforms = Compose::new(vec![
            Box::new(RandRemoveDc::default_with_prob(1.)),
            Box::new(RandLFilt::default_with_prob(1.)),
            Box::new(RandBiquadFilter::default_with_prob(1.).with_sr(sr)),
            Box::new(RandResample::default_with_prob(1.).with_sr(sr)),
        ]);
        let mut out = test_sample.clone();
        let mut out_manual = test_sample.clone();
        write_wav_iter("../out/original.wav", test_sample.iter(), sr as u32, ch)?;
        seed_from_u64(42);
        transforms.transform(&mut (&mut out).into())?;
        write_wav_iter("../out/compose_all.wav", out.iter(), sr as u32, ch)?;
        seed_from_u64(42);
        for (i, t) in transforms.transforms.iter().enumerate() {
            t.transform(&mut (&mut out_manual).into())?;
            write_wav_iter(
                format!("../out/compose_{i}.wav").as_str(),
                out_manual.iter(),
                sr as u32,
                ch,
            )?;
        }
        assert_eq!(out.sum(), out_manual.sum());
        assert_eq!(out.var(1.), out_manual.var(1.));

        Ok(())
    }

    #[test]
    pub fn test_rand_resample() -> Result<()> {
        let (mut test_sample, sr) = setup();
        let ch = test_sample.len_of(Axis(0)) as u16;
        seed_from_u64(42);
        let rand_resample = RandResample::new(1., sr, 0.8, 1.2, 1024);
        rand_resample.transform(&mut (&mut test_sample).into()).unwrap();
        write_wav_iter("../out/resampled.wav", test_sample.iter(), sr as u32, ch)?;
        Ok(())
    }

    #[test]
    pub fn test_low_pass() -> Result<()> {
        let (sample, sr) = setup();
        let sr = sr as u32;
        let mut lowpass_res = sample.clone();
        let mut lowpass_biquad = sample.clone();
        let ch = sample.len_of(Axis(0)) as u16;
        let f = 8000.;
        let mut mem = [0.; 2];
        let (b, a) = low_pass(f, 0.707, sr as usize);
        biquad_inplace(&mut lowpass_biquad, &mut mem, &b, &a);
        write_wav_iter("../out/lowpass_biquad.wav", lowpass_biquad.iter(), sr, ch).unwrap();
        let xx: f64 = sample.iter().map(|&n| n as f64 * n as f64).sum();
        let yy: f64 = lowpass_biquad.iter().map(|&n| n as f64 * n as f64).sum();
        let xy: f64 = sample.iter().zip(lowpass_biquad).map(|(&n, m)| n as f64 * m as f64).sum();
        let corr = xy / (xx.sqrt() * yy.sqrt());
        dbg!(corr);

        lowpass_res = low_pass_resample(lowpass_res.view(), f as usize, sr as usize).unwrap();
        write_wav_iter("../out/lowpass_resample.wav", lowpass_res.iter(), sr, ch).unwrap();

        let xx: f64 = sample.iter().map(|&n| n as f64 * n as f64).sum();
        let yy: f64 = lowpass_res.iter().map(|&n| n as f64 * n as f64).sum();
        let xy: f64 = sample.iter().zip(lowpass_res).map(|(&n, m)| n as f64 * m as f64).sum();
        let corr = xy / (xx.sqrt() * yy.sqrt());
        dbg!(corr);
        Ok(())
    }

    #[test]
    pub fn test_reverb() -> Result<()> {
        let (mut speech, sr) = setup();
        let len = 4 * sr;
        speech.slice_collapse(s![0..1, 0..len]);
        let mut noise = ReadWav::new("../assets/noise_freesound_573577.wav")?.samples_arr2()?;
        noise.slice_axis_inplace(Axis(1), Slice::from(0..len));
        let rir = ReadWav::new("../assets/rir_sim_1001_w11.7_l2.6_h2.5_rt60_0.7919.wav")?
            .samples_arr2()?;
        let reverb = RandReverbSim::new(1., sr)
            .with_drr(0.2)
            .with_rt60(0.1)
            .with_offset_late_reflections(20);
        write_wav_arr2("../out/speech_noreverb.wav", speech.view(), sr as u32)?;
        write_wav_arr2("../out/noise_noreverb.wav", noise.view(), sr as u32)?;
        let speech_rev = reverb.transform(&mut speech, &mut noise, move || Ok(rir))?.unwrap();
        write_wav_arr2("../out/speech_target.wav", speech.view(), sr as u32)?;
        write_wav_arr2("../out/speech_reverb.wav", speech_rev.view(), sr as u32)?;
        write_wav_arr2("../out/noise_reverb.wav", noise.view(), sr as u32)?;
        Ok(())
    }

    #[test]
    pub fn test_clipping() -> Result<()> {
        let (test_sample, sr) = setup();
        let sr = sr as u32;
        let mut test_sample_c = test_sample.clone();
        let ch = test_sample.len_of(Axis(0)) as u16;
        let tsnr = 3.; // Test with 3dB
        let transform = RandClipping::new(1.0, 1e-10, 0.001).with_snr(tsnr..tsnr);
        transform.transform(&mut (&mut test_sample_c).into())?;
        let resulting_snr = transform.sdr(test_sample.view(), test_sample_c.view());
        write_wav_iter("../out/original.wav", test_sample.iter(), sr, ch)?;
        write_wav_iter("../out/clipped_snr.wav", test_sample_c.iter(), sr, ch)?;
        log::info!("Expecting target SNR {}, got SNR {}", tsnr, resulting_snr);
        // Test relative difference
        assert!(((resulting_snr - tsnr) / tsnr).abs() < 0.05);

        let mut test_sample_c = test_sample.clone();
        let c = 0.05;
        let transform = RandClipping::new(1.0, 1e-10, 0.001).with_c(c..c);
        transform.transform(&mut (&mut test_sample_c).into())?;
        let resulting_snr = transform.sdr(test_sample.view(), test_sample_c.view());
        write_wav_iter("../out/clipped.wav", test_sample_c.iter(), sr, ch)?;
        dbg!(c, resulting_snr);
        Ok(())
    }

    #[test]
    pub fn test_zeroing() -> Result<()> {
        let (test_sample, sr) = setup();
        let sr = sr as u32;
        let mut test_sample_c = test_sample.clone();
        let ch = test_sample.len_of(Axis(0)) as u16;
        let transform = RandZeroingTD::default_with_prob(1.0).with_n_samples(420, 1800);
        transform.transform(&mut (&mut test_sample_c).into())?;
        write_wav_iter("../out/original.wav", test_sample.iter(), sr, ch)?;
        write_wav_iter("../out/zeroed.wav", test_sample_c.iter(), sr, ch)?;
        Ok(())
    }

    #[test]
    pub fn test_gen_noise() -> Result<()> {
        setup();

        let sr = 48000;
        let ch = 2;
        let n = sr as usize * 3; // 3 seconds
        let white_noise = gen_noise(0., ch, n, sr)?;
        let pink_noise = gen_noise(1., ch, n, sr)?;
        let brown_noise = gen_noise(2., ch, n, sr)?;
        let blue_noise = gen_noise(-1., ch, n, sr)?;
        let violet_noise = gen_noise(-1., ch, n, sr)?;
        write_wav_iter("../out/white_noise.wav", white_noise.iter(), sr, ch)?;
        write_wav_iter("../out/pink_noise.wav", pink_noise.iter(), sr, ch)?;
        write_wav_iter("../out/brown_noise.wav", brown_noise.iter(), sr, ch)?;
        write_wav_iter("../out/blue_noise.wav", blue_noise.iter(), sr, ch)?;
        write_wav_iter("../out/violet_noise.wav", violet_noise.iter(), sr, ch)?;
        Ok(())
    }

    #[test]
    pub fn test_filters() -> Result<()> {
        let (s_orig, sr) = setup();
        let ch = s_orig.len_of(Axis(0)) as u16;
        let f = 1000.;
        let gain = -18.;
        let q = 0.5;
        // Low pass
        let (b, a) = low_pass(f, q, sr);
        let mut s_filt = s_orig.clone();
        biquad_filter(&mut s_filt, &b, &a);
        write_wav_iter("../out/filt_lowpass.wav", s_filt.iter(), sr as u32, ch)?;
        // High pass
        let (b, a) = high_pass(f, q, sr);
        let mut s_filt = s_orig.clone();
        biquad_filter(&mut s_filt, &b, &a);
        write_wav_iter("../out/filt_higpass.wav", s_filt.iter(), sr as u32, ch)?;
        // Low shelf
        let (b, a) = low_shelf(f, gain, q, sr);
        let mut s_filt = s_orig.clone();
        biquad_filter(&mut s_filt, &b, &a);
        write_wav_iter("../out/filt_lowshelf.wav", s_filt.iter(), sr as u32, ch)?;
        // High shelf
        let (b, a) = high_shelf(f, gain, q, sr);
        let mut s_filt = s_orig.clone();
        biquad_filter(&mut s_filt, &b, &a);
        write_wav_iter("../out/filt_highshelf.wav", s_filt.iter(), sr as u32, ch)?;
        // Peaking eq
        let (b, a) = peaking_eq(f, gain, q, sr);
        let mut s_filt = s_orig.clone();
        biquad_filter(&mut s_filt, &b, &a);
        write_wav_iter("../out/filt_peaking_eq.wav", s_filt.iter(), sr as u32, ch)?;
        // Notch
        let (b, a) = notch(f, q, sr);
        let mut s_filt = s_orig.clone();
        biquad_filter(&mut s_filt, &b, &a);
        write_wav_iter("../out/filt_notch.wav", s_filt.iter(), sr as u32, ch)?;
        // Test augmentation transform
        seed_from_u64(43);
        let aug = RandBiquadFilter::default_with_prob(1.0).with_sr(sr);
        let mut s_aug = s_orig;
        aug.transform(&mut (&mut s_aug).into())?;
        write_wav_iter("../out/filt_aug.wav", s_aug.iter(), sr as u32, ch)?;
        Ok(())
    }

    #[test]
    pub fn test_air_absorption() -> Result<()> {
        let (sample, sr) = setup();
        let fft_size = sr / 50;
        let hop_size = fft_size / 2;
        let mut state = DFState::new(sr, fft_size, hop_size, 1, 1);
        write_wav_arr2("../out/original.wav", sample.view(), sr as u32).unwrap();
        let x = stft(sample.view(), &mut state, false);
        let airabs = AirAbsorptionAugmentation::new(sr, 1.0);
        for (n, c) in airabs.air_absorption.iter() {
            let mut x_aug = x.clone();
            airabs.apply(&mut x_aug, c, 20.);
            let sample = istft(x_aug.view_mut(), &mut state, true);
            write_wav_arr2(&format!("../out/air_abs_{n}.wav"), sample.view(), sr as u32).unwrap();
        }
        Ok(())
    }
}
