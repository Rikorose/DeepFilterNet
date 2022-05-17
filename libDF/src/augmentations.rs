use std::mem::MaybeUninit;
use std::ops::Range;
#[cfg(feature = "dataset_timings")]
use std::time::Instant;

use ndarray::{concatenate, prelude::*, Slice};
use ndarray_rand::rand::{seq::SliceRandom, Rng};
use ndarray_rand::{rand_distr::Normal, rand_distr::Uniform, RandomExt};
use rubato::{FftFixedInOut, Resampler};
use thiserror::Error;

use self::BiquadFilters::*;
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
    #[error("Transform {transform} not initalized: {msg}")]
    NotInitialized { transform: String, msg: String },
    #[error("DF error: {0}")]
    DfError(String),
    #[error("Resample Error")]
    ResampleError(#[from] rubato::ResampleError),
    #[error("Ndarray Shape Error")]
    NdarrayShapeError(#[from] ndarray::ShapeError),
    #[error("Wav Reader Error")]
    WavReadError(#[from] crate::wav_utils::WavUtilsError),
}

pub trait Transform {
    fn transform(&self, x: &mut Array2<f32>) -> Result<()>;
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
    transforms: Vec<Box<dyn Transform + Send>>,
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

    pub fn transform(&self, x: &mut Array2<f32>) -> Result<()> {
        #[cfg(feature = "dataset_timings")]
        let mut t0 = Instant::now();
        #[cfg(feature = "dataset_timings")]
        let mut timings = Vec::new();
        for t in self.transforms.iter() {
            match t.transform(x) {
                Ok(()) => (),
                Err(e) => log::error!("{:?}", e),
            };
            #[cfg(feature = "dataset_timings")]
            {
                let t1 = Instant::now();
                let d = (t1 - t0).as_micros();
                if d > 100 {
                    timings.push(format!("{}: {} ms", t.name(), d / 1000));
                }
                t0 = t1;
            }
        }
        #[cfg(feature = "dataset_timings")]
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
    fn transform(&self, x: &mut Array2<f32>) -> Result<()> {
        if self.prob == 0. || (self.prob < 1. && thread_rng()?.uniform(0f32, 1f32) > self.prob) {
            return Ok(());
        }
        let a: [f32; 2] = self.sample_ab()?;
        let b: [f32; 2] = self.sample_ab()?;
        let mut mem = [0f32; 2];
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
    let b2 = (1. - w0.cos()) / 2.;
    let a0 = 1. + alpha;
    let a1 = -2. * w0.cos();
    let a2 = 1. - alpha;
    ([b0, b1, b2], [a0, a1, a2])
}
fn low_shelf(center_freq: f32, gain_db: f32, q_factor: f32, sr: usize) -> ([f32; 3], [f32; 3]) {
    let w0 = 2. * std::f32::consts::PI * center_freq / sr as f32;
    let amp = 10f32.powf(gain_db / 40.);
    let alpha = w0.sin() / 2. / q_factor;
    dbg!(amp, alpha);

    let b0 = amp * ((amp + 1.) - (amp - 1.) * w0.cos() + 2. * amp.sqrt() * alpha);
    let b1 = 2. * amp * ((amp - 1.) - (amp + 1.) * w0.cos());
    let b2 = amp * ((amp + 1.) - (amp - 1.) * w0.cos() - 2. * amp.sqrt() * alpha);
    let a0 = (amp + 1.) + (amp - 1.) * w0.cos() + 2. * amp.sqrt() * alpha;
    let a1 = -2. * ((amp - 1.) + (amp + 1.) * w0.cos());
    let a2 = (amp + 1.) + (amp - 1.) * w0.cos() - 2. * amp.sqrt() * alpha;
    dbg!([b0, b1, b2], [a0, a1, a2])
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
    let amp = 10f32.powf(gain_db as f32 / 40.);
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

#[derive(Clone, Copy)]
enum BiquadFilters {
    HighShelf,
    LowShelf,
    HighPass,
    LowPass,
    PeakingEQ,
    Notch,
}
impl BiquadFilters {
    pub fn iterator() -> impl Iterator<Item = BiquadFilters> {
        [HighShelf, LowShelf, HighPass, LowPass, PeakingEQ, Notch].iter().copied()
    }
}

#[derive(Clone)]
pub struct RandBiquadFilter {
    prob: f32,
    sr: Option<usize>,
    n_freqs: usize,
    f_low: usize,
    f_high: usize,
    gain_db_low: f32,
    gain_db_high: f32,
    q_low: f32,
    q_high: f32,
    filters: Vec<BiquadFilters>,
}
impl RandBiquadFilter {
    pub fn with_sr(mut self, sr: usize) -> Self {
        self.sr = Some(sr);
        self
    }
}
impl Transform for RandBiquadFilter {
    fn transform(&self, x: &mut Array2<f32>) -> Result<()> {
        if self.sr.is_none() {
            return Err(AugmentationError::NotInitialized {
                transform: "RandEQ".into(),
                msg: "No sampling rate provided.".into(),
            });
        }
        let sr = self.sr.unwrap();
        let mut rng = thread_rng()?;
        if self.prob == 0. || (self.prob < 1. && rng.uniform(0f32, 1f32) > self.prob) {
            return Ok(());
        }
        for _ in 0..self.n_freqs {
            let freq = rng.log_uniform(self.f_low as f32, self.f_high as f32);
            let gain_db = rng.uniform_inclusive(self.gain_db_low, self.gain_db_high);
            let q = rng.uniform_inclusive(self.q_low, self.q_high);
            let (b, a) = match self.filters.choose(&mut rng) {
                None => continue,
                Some(HighShelf) => high_shelf(freq, gain_db, q, sr),
                Some(LowShelf) => low_shelf(freq, gain_db, q, sr),
                Some(HighPass) => high_pass(freq, q, sr),
                Some(LowPass) => low_pass(freq, q, sr),
                Some(PeakingEQ) => peaking_eq(freq, gain_db, q, sr),
                Some(Notch) => notch(freq, q, sr),
            };
            biquad_filter(x, &b, &a);
        }
        Ok(())
    }
    fn default_with_prob(p: f32) -> Self {
        RandBiquadFilter {
            prob: p,
            sr: None,
            n_freqs: 3,
            f_low: 40,
            f_high: 8000,
            gain_db_high: 15.,
            gain_db_low: -15.,
            q_low: 0.5,
            q_high: 1.5,
            filters: BiquadFilters::iterator().collect(),
        }
    }
    fn box_clone(&self) -> Box<dyn Transform + Send> {
        Box::new((*self).clone())
    }
    fn name(&self) -> &str {
        "RandBiquadFilter"
    }
}

/// Low pass by resampling the data to `f_cut_off*2`.
pub(crate) fn low_pass_resample(
    x: ArrayView2<f32>,
    f_cut_off: usize,
    sr: usize,
) -> Result<Array2<f32>> {
    let x = resample(x, sr, f_cut_off * 2, None)?;
    let x = resample(x.view(), f_cut_off * 2, sr, None)?;
    Ok(x)
}

pub(crate) fn resample(
    x: ArrayView2<f32>,
    sr: usize,
    new_sr: usize,
    chunk_size: Option<usize>,
) -> Result<Array2<f32>> {
    let channels = x.len_of(Axis(0));
    let len = x.len_of(Axis(1));
    let out_len = (len as f32 * new_sr as f32 / sr as f32).ceil() as usize;
    let chunk_size = chunk_size.unwrap_or(2048);
    let mut resampler = FftFixedInOut::<f32>::new(sr, new_sr, chunk_size, channels)
        .expect("Could not initialize resampler");
    let chunk_size = resampler.input_frames_max();
    // One extra to get the remaining resampler state buffer
    let num_chunks = (len as f32 / chunk_size as f32).ceil() as usize + 1;
    let chunk_size_out = resampler.output_frames_max();
    let mut out = Array2::uninit((channels, chunk_size_out * num_chunks));
    let mut inbuf = vec![vec![0f32; chunk_size]; channels];
    let mut outbuf = resampler.output_buffer_allocate();
    let mut out_chunk_iter = out.axis_chunks_iter_mut(Axis(1), chunk_size_out);
    for chunk in x.axis_chunks_iter(Axis(1), chunk_size) {
        for (chunk_ch, buf_ch) in chunk.axis_iter(Axis(0)).zip(inbuf.iter_mut()) {
            if chunk_ch.len() == chunk_size {
                chunk_ch.assign_to(buf_ch);
            } else {
                chunk_ch.assign_to(&mut buf_ch[..chunk_ch.len()]);
                for b in buf_ch[chunk_ch.len()..].iter_mut() {
                    *b = 0. // Zero pad
                }
            }
        }
        resampler.process_into_buffer(&inbuf, &mut outbuf, None)?;
        for (res_ch, mut out_ch) in
            outbuf.iter().zip(out_chunk_iter.next().unwrap().axis_iter_mut(Axis(0)))
        {
            debug_assert_eq!(res_ch.len(), out_ch.len());
            for (&x, y) in res_ch.iter().zip(out_ch.iter_mut()) {
                *y = MaybeUninit::new(x);
            }
        }
    }
    // Another round with zeros to get remaining state buffer
    for in_ch in inbuf.iter_mut() {
        in_ch.fill(0.)
    }
    resampler.process_into_buffer(&inbuf, &mut outbuf, None)?;
    for (res_ch, mut out_ch) in
        outbuf.iter().zip(out_chunk_iter.next().unwrap().axis_iter_mut(Axis(0)))
    {
        debug_assert_eq!(res_ch.len(), out_ch.len());
        for (&x, y) in res_ch.iter().zip(out_ch.iter_mut()) {
            *y = MaybeUninit::new(x);
        }
    }
    let mut out = unsafe { out.assume_init() };
    out.slice_axis_inplace(
        Axis(1),
        Slice::from(chunk_size_out / 2..chunk_size_out / 2 + out_len),
    );
    Ok(out)
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
    fn transform(&self, x: &mut Array2<f32>) -> Result<()> {
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
        out.move_into(x);
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
        x.mapv_inplace(|x| x.max(-c).min(c))
    }
    fn clip(&self, x: ArrayView2<f32>, c: f32) -> Array2<f32> {
        x.map(|x| x.max(-c).min(c))
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
        match roots::find_root_brent(a, b, &f, &mut self.eps_c.clone()) {
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
    fn transform(&self, x: &mut Array2<f32>) -> Result<()> {
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
pub struct RandRemoveDc {
    prob: f32,
}
impl Transform for RandRemoveDc {
    fn transform(&self, x: &mut Array2<f32>) -> Result<()> {
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
        noise
    } else {
        // Fast path for white noise
        Array::random((ch, sr), Normal::new(-1., 1.).unwrap())
    };
    let max = find_max(&noise)? * 1.1;
    noise /= max;
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
        self.generate_with_scratch(
            f_decay,
            num_channels,
            num_samples,
            &mut self.fft_forward.make_scratch_vec(),
            &mut self.fft_inverse.make_scratch_vec(),
        )
    }
    pub fn generate_with_scratch(
        &self,
        f_decay: f32,
        num_channels: u16,
        num_samples: usize,
        scratch_forward: &mut [Complex32],
        scratch_inverse: &mut [Complex32],
    ) -> Result<Array2<f32>> {
        gen_noise_with_scratch(
            f_decay,
            num_channels,
            num_samples,
            self.sr,
            self.fft_forward.as_ref(),
            self.fft_inverse.as_ref(),
            scratch_forward,
            scratch_inverse,
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
        if self.p == 0. || self.p < rng.uniform(0., 1.) {
            return Ok(None);
        }
        let f_decay = rng.uniform(f_decay_min, f_decay_max);
        Ok(Some(self.generate(f_decay, num_channels, num_samples)?))
    }
}

pub(crate) struct RandReverbSim {
    prob_speech: f32,
    prob_noise: f32,
    prob_resample: f32,
    prob_decay: f32,
    sr: usize,
}
impl RandReverbSim {
    fn supress_late(
        &self,
        mut rir: Array2<f32>,
        sr: usize,
        offset_ms: f32,
        rt60: f32,
        trim: bool,
    ) -> Result<Array2<f32>> {
        // find first tap
        let mut start = argmax_abs(rir.iter()).unwrap();
        if offset_ms > 0. {
            start += (offset_ms * sr as f32 / 1000.) as usize;
        }
        let len = rir.len_of(Axis(1));
        let mut decay: Array2<f32> = Array2::ones((1, len));
        let dt = 1. / sr as f32;
        let rt60_level = 10f32.powi(-60 / 20);
        let tau = -rt60 / rt60_level.log10();
        if start >= len {
            return Ok(rir);
        }
        decay.slice_mut(s![0, start..]).assign(&Array1::from_iter(
            (0..(len - start)).map(|v| (-(v as f32) * dt / tau).exp()),
        ));
        rir = rir * decay;
        if trim {
            return self.trim(rir);
        }
        Ok(rir)
    }
    fn trim(&self, mut rir: Array2<f32>) -> Result<Array2<f32>> {
        let min_db = -80.;
        let len = rir.len_of(Axis(1));
        let rir_mono = rir.mean_axis(Axis(0)).unwrap();
        let argmax = argmax_abs(rir_mono.iter()).unwrap();
        let max_ref: f32 = rir_mono[argmax];
        let min_level = 10f32.powf((min_db + max_ref.log10() * 20.) / 20.);
        let mut idx = len;
        for (i, v) in rir_mono.slice(s![argmax..]).indexed_iter() {
            if v < &min_level {
                idx = i;
            } else {
                idx = len; // reset
            }
        }
        rir.slice_collapse(s![.., ..idx]);
        Ok(rir)
    }
    fn good_fft_size(&self, rir: &Array2<f32>) -> usize {
        // Zero pad RIR for better FFT efficiency by finding prime factors up to a limit of 11.
        let len = rir.len_of(Axis(1));
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
    fn pad(&self, x: &mut Array2<f32>, npad: usize) -> Result<()> {
        if npad == 0 {
            return Ok(());
        }
        x.append(Axis(1), Array2::zeros((x.len_of(Axis(0)), npad)).view())?;
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
        let mut rir = match rir_callback() {
            Ok(r) => r,
            Err(e) => {
                return Err(AugmentationError::DfError(format!(
                    "Error getting RIR in RandReverbSim::transform() {:?}",
                    e
                )));
            }
        };
        let mut rng = thread_rng()?;
        let apply_speech = self.prob_speech > rng.uniform(0f32, 1f32);
        let apply_noise = self.prob_noise > rng.uniform(0f32, 1f32);
        if !(apply_speech || apply_noise) {
            return Ok(None);
        }
        let orig_len = speech.len_of(Axis(1));
        // Maybe resample RIR as augmentation
        if self.prob_resample > rng.uniform(0f32, 1f32) {
            let new_sr: f32 = rng.uniform(0.8, 1.2) * self.sr as f32;
            let new_sr = ((new_sr / 500.).round() * 500.) as usize;
            rir = resample(rir.view(), self.sr, new_sr, Some(512))?;
        }
        if self.prob_decay > rng.uniform(0f32, 1f32) {
            let rt60 = rng.uniform(0.2, 1.);
            rir = self.supress_late(rir, self.sr, 0., rt60, false)?;
        }
        rir = self.trim(rir)?;
        // Normalize and flip RIR for convolution
        let rir_e = rir.map(|v| v * v).sum().sqrt();
        let fft_size = self.good_fft_size(&rir);
        let cur_len = rir.len_of(Axis(1));
        let mut rir_noise = rir / rir_e;
        self.pad(&mut rir_noise, fft_size - cur_len)?;
        // DF state for convolve FFT
        let hop_size = fft_size / 4;
        let mut state = DFState::new(self.sr, fft_size, hop_size, 1, 1);

        // speech_rev contains reverberant speech for mixing with noise
        let mut speech_rev = None;
        if apply_speech {
            self.pad(speech, fft_size)?; // Pad since STFT will truncate at the end
            speech_rev =
                Some(self.convolve(speech, rir_noise.clone(), &mut state, Some(orig_len))?);
            // Speech should be a slightly dereverberant signal as target
            // TODO: Make dereverberation parameters configurable.
            let rir_speech = self.supress_late(rir_noise.clone(), self.sr, 5., 0.2, false)?;
            *speech = self.convolve(speech, rir_speech, &mut state, Some(orig_len))?;
            debug_assert_eq!(speech.shape(), speech_rev.as_ref().unwrap().shape());
            debug_assert_eq!(speech.len_of(Axis(1)), noise.len_of(Axis(1)));
        }
        if apply_noise {
            // Noisy contains reverberant noise
            self.pad(noise, fft_size)?;
            *noise = self.convolve(noise, rir_noise, &mut state, Some(orig_len))?;
            debug_assert_eq!(speech.len_of(Axis(1)), noise.len_of(Axis(1)));
        }
        Ok(speech_rev)
    }
    fn convolve(
        &self,
        x: &Array2<f32>,
        mut rir: Array2<f32>,
        state: &mut DFState,
        truncate: Option<usize>,
    ) -> Result<Array2<f32>> {
        let mut x_ = stft(x.as_standard_layout().view(), state, true);
        let len = state.fft_forward.get_scratch_len();
        if len < state.analysis_scratch.len() {
            state.analysis_scratch.resize(len, Complex32::default())
        }
        let rir = fft(
            &mut rir,
            state.fft_forward.as_ref(),
            &mut state.analysis_scratch,
        )?;
        let rir: ArrayView3<Complex32> = match rir.broadcast(x_.shape()) {
            Some(r) => r.into_dimensionality()?,
            None => {
                return Err(AugmentationError::DfError(format!(
                    "Shape missmatch: {:?} {:?}",
                    x_.shape(),
                    rir.shape()
                )));
            }
        };
        x_ = x_ * rir;
        let mut out = istft(x_.view_mut(), state, true);
        out.slice_collapse(s![.., (state.window_size - state.frame_size)..]);
        if let Some(max_len) = truncate {
            out.slice_collapse(s![.., ..max_len]);
        }
        Ok(out)
    }
    pub fn new(p: f32, sr: usize) -> Self
    where
        Self: Sized,
    {
        RandReverbSim {
            prob_speech: p,
            prob_noise: p,
            prob_resample: p,
            prob_decay: p,
            sr,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Once;

    use super::*;
    use crate::wav_utils::*;

    static INIT: Once = Once::new();

    /// Setup function that is only run once, even if called multiple times.
    fn setup() {
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
    }

    fn create_out_dir() -> std::io::Result<()> {
        match std::fs::create_dir("../out") {
            Err(ref e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
            r => r,
        }
    }

    #[test]
    pub fn test_rand_resample() -> Result<()> {
        setup();
        let reader = ReadWav::new("../assets/clean_freesound_33711.wav")?;
        let sr = reader.sr as u32;
        let mut test_sample = reader.samples_arr2()?;
        let ch = test_sample.len_of(Axis(0)) as u16;
        seed_from_u64(42);
        let rand_resample = RandResample::new(1., sr as usize, 0.8, 1.2, 1024);
        rand_resample.transform(&mut test_sample).unwrap();
        write_wav_iter("../out/resampled.wav", test_sample.iter(), sr, ch)?;
        Ok(())
    }

    #[test]
    pub fn test_low_pass() -> Result<()> {
        setup();
        let reader = ReadWav::new("../assets/clean_freesound_33711.wav")?;
        let sr = reader.sr as u32;
        let mut test_sample = reader.samples_arr2()?;
        let mut test_sample2 = test_sample.clone();
        let ch = test_sample.len_of(Axis(0)) as u16;
        let f = 8000.;
        let mut mem = [0.; 2];
        let (b, a) = low_pass(f, 0.707, sr as usize);
        biquad_inplace(&mut test_sample, &mut mem, &b, &a);
        write_wav_iter("../out/lowpass.wav", test_sample.iter(), sr, ch)?;
        test_sample2 = low_pass_resample(test_sample2.view(), f as usize, sr as usize)?;
        write_wav_iter("../out/lowpass_resample.wav", test_sample2.iter(), sr, ch)?;
        Ok(())
    }

    #[test]
    pub fn test_reverb() -> Result<()> {
        setup();
        let reader = ReadWav::new("../assets/clean_freesound_33711.wav")?;
        let sr = reader.sr;
        let len = 4 * sr;
        let mut speech = reader.samples_arr2()?;
        speech.slice_collapse(s![0..1, 0..len]);
        let mut noise = ReadWav::new("../assets/noise_freesound_573577.wav")?.samples_arr2()?;
        noise.slice_axis_inplace(Axis(1), Slice::from(0..len));
        let rir = ReadWav::new("../assets/rir_sim_1001_w11.7_l2.6_h2.5_rt60_0.7919.wav")?
            .samples_arr2()?;
        let reverb = RandReverbSim::new(1., sr);
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
        setup();
        let reader = ReadWav::new("../assets/clean_freesound_33711.wav")?;
        let sr = reader.sr as u32;
        let test_sample = reader.samples_arr2()?;
        let mut test_sample_c = test_sample.clone();
        let ch = test_sample.len_of(Axis(0)) as u16;
        let tsnr = 3.; // Test with 3dB
        let transform = RandClipping::new(1.0, 1e-10, 0.001).with_snr(tsnr..tsnr);
        transform.transform(&mut test_sample_c)?;
        let resulting_snr = transform.sdr(test_sample.view(), test_sample_c.view());
        write_wav_iter("../out/clipped_snr.wav", test_sample_c.iter(), sr, ch)?;
        log::info!("Expecting target SNR {}, got SNR {}", tsnr, resulting_snr);
        // Test relative difference
        assert!(((resulting_snr - tsnr) / tsnr).abs() < 0.05);

        let mut test_sample_c = test_sample.clone();
        let c = 0.05;
        let transform = RandClipping::new(1.0, 1e-10, 0.001).with_c(c..c);
        transform.transform(&mut test_sample_c)?;
        let resulting_snr = transform.sdr(test_sample.view(), test_sample_c.view());
        write_wav_iter("../out/clipped.wav", test_sample_c.iter(), sr, ch)?;
        dbg!(c, resulting_snr);
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
        setup();
        let reader = ReadWav::new("../assets/clean_freesound_33711.wav")?;
        let sr = reader.sr;
        let s_orig = reader.samples_arr2()?;
        let ch = s_orig.len_of(Axis(0)) as u16;
        let f = 8000.;
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
        let mut s_filt = s_orig;
        biquad_filter(&mut s_filt, &b, &a);
        write_wav_iter("../out/filt_notch.wav", s_filt.iter(), sr as u32, ch)?;
        Ok(())
    }
}
