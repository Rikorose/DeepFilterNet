use std::mem::MaybeUninit;
use std::ops::Range;
#[cfg(feature = "dataset_timings")]
use std::time::Instant;

use ndarray::{prelude::*, Slice};
use ndarray_rand::rand::{distributions::uniform::Uniform, Rng};
use ndarray_rand::{rand_distr::Normal as NdNormal, RandomExt};
use rubato::{FftFixedInOut, Resampler};
use thiserror::Error;

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

#[derive(Clone)]
pub struct RandEQ {
    prob: f32,
    sr: Option<usize>,
    n_freqs: usize,
    f_low: usize,
    f_high: usize,
    gain_db: i32,
    q_low: f32,
    q_high: f32,
}
impl RandEQ {
    pub fn with_sr(mut self, sr: usize) -> Self {
        self.sr = Some(sr);
        self
    }
}
impl Transform for RandEQ {
    fn transform(&self, x: &mut Array2<f32>) -> Result<()> {
        if self.sr.is_none() {
            return Err(AugmentationError::NotInitialized {
                transform: "RandEQ".into(),
                msg: "No sampling rate provided.".into(),
            });
        }
        let mut rng = thread_rng()?;
        if self.prob == 0. || (self.prob < 1. && rng.uniform(0f32, 1f32) > self.prob) {
            return Ok(());
        }
        for _ in 0..self.n_freqs {
            let freq = rng.log_uniform(self.f_low as f32, self.f_high as f32);
            let gain = rng.uniform_inclusive(-self.gain_db, self.gain_db) as f32;
            let q = rng.uniform(self.q_low, self.q_high);
            let w0 = 2. * std::f32::consts::PI * freq / self.sr.unwrap() as f32;
            let amp = (gain / 40. * std::f32::consts::LN_10).exp();
            let alpha = w0.sin() / 2. / q;
            let w0_cos = w0.cos();

            let b0 = 1. + alpha * amp;
            let b1 = -2. * w0_cos;
            let b2 = 1. - alpha * amp;
            let a0 = 1. + alpha / amp;
            let a1 = -2. * w0_cos;
            let a2 = 1. - alpha / amp;

            let mut mem = [0.; 2];
            for x_ch in x.axis_iter_mut(Axis(0)) {
                biquad_inplace(x_ch, &mut mem, &[b0, b1, b2], &[a0, a1, a2]);
            }
        }
        Ok(())
    }
    fn default_with_prob(p: f32) -> Self {
        RandEQ {
            prob: p,
            sr: None,
            n_freqs: 3,
            f_low: 40,
            f_high: 8000,
            gain_db: 15,
            q_low: 0.5,
            q_high: 1.5,
        }
    }
    fn box_clone(&self) -> Box<dyn Transform + Send> {
        Box::new((*self).clone())
    }
    fn name(&self) -> &str {
        "RandEQ"
    }
}

pub(crate) fn low_pass_resample(x: &Array2<f32>, cut_off: usize, sr: usize) -> Result<Array2<f32>> {
    let x = resample(x, sr, cut_off * 2, None)?;
    let x = resample(&x, cut_off * 2, sr, None)?;
    Ok(x)
}

pub(crate) fn resample(
    x: &Array2<f32>,
    sr: usize,
    new_sr: usize,
    chunk_size: Option<usize>,
) -> Result<Array2<f32>> {
    let channels = x.len_of(Axis(0));
    let len = x.len_of(Axis(1));
    let mut resampler = FftFixedInOut::<f32>::new(sr, new_sr, chunk_size.unwrap_or(1024), channels)
        .expect("Could not initialize resampler");
    let chunk_size = resampler.input_frames_max();
    let num_chunks = (len as f32 / chunk_size as f32).ceil() as usize;
    let chunk_size_out = (chunk_size as u64 * new_sr as u64 / sr as u64) as usize;
    let mut out = Array2::uninit((channels, chunk_size_out * num_chunks));
    let mut inbuf = vec![vec![0f32; chunk_size]; channels];
    let mut outbuf = resampler.output_buffer_allocate();
    let mut i = 0;
    for (chunk, mut out_chunk) in x
        .axis_chunks_iter(Axis(1), chunk_size)
        .zip(out.axis_chunks_iter_mut(Axis(1), chunk_size_out))
    {
        for (chunk_ch, buf_ch) in chunk.axis_iter(Axis(0)).zip(inbuf.iter_mut()) {
            if chunk_ch.len() < chunk_size {
                chunk_ch.assign_to(&mut buf_ch[..chunk_ch.len()]);
                for b in buf_ch[chunk_ch.len()..].iter_mut() {
                    *b = 0. // Zero pad
                }
            } else {
                chunk_ch.assign_to(buf_ch);
            }
        }
        resampler.process_into_buffer(&inbuf, &mut outbuf, None)?;
        for (res_ch, mut out_ch) in outbuf.iter().zip(out_chunk.axis_iter_mut(Axis(0))) {
            debug_assert_eq!(res_ch.len(), out_ch.len());
            for (&x, y) in res_ch.iter().zip(out_ch.iter_mut()) {
                *y = MaybeUninit::new(x);
            }
        }
        i += 1;
    }
    assert_eq!(i, num_chunks);
    Ok(unsafe { out.assume_init() })
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
        let out = resample(x, sr, new_sr, Some(self.chunk_size))?;
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
                dbg!(c)
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

pub fn low_pass(x: &mut Array2<f32>, freq: f32, sr: usize, q: Option<f32>) -> Result<()> {
    let q = q.unwrap_or(0.707);
    let w0 = 2. * std::f32::consts::PI * freq / sr as f32;
    let alpha = w0.sin() / 2. / q;
    let w0_cos = w0.cos();

    let b0 = (1. - w0_cos) / 2.;
    let b1 = 1. - w0_cos;
    let b2 = b0;
    let a0 = 1. + alpha;
    let a1 = -2. * w0_cos;
    let a2 = 1. - alpha;

    for x_ch in x.axis_iter_mut(Axis(0)) {
        let mut mem = [0.; 2];
        biquad_inplace(x_ch, &mut mem, &[b0, b1, b2], &[a0, a1, a2]);
    }
    Ok(())
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
            rir = resample(&rir, self.sr, new_sr, Some(512))?;
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
                Some(self.convolve(speech, rir_noise.view(), &mut state, Some(orig_len))?);
            // Speech should be a slightly dereverberant signal as target
            // TODO: Make dereverberation parameters configurable.
            let rir_speech = self.supress_late(rir_noise.clone(), self.sr, 5., 0.2, false)?;
            *speech = self.convolve(speech, rir_speech.view(), &mut state, Some(orig_len))?;
            debug_assert_eq!(speech.shape(), speech_rev.as_ref().unwrap().shape());
            debug_assert_eq!(speech.len_of(Axis(1)), noise.len_of(Axis(1)));
        }
        if apply_noise {
            // Noisy contains reverberant noise
            self.pad(noise, fft_size)?;
            *noise = self.convolve(noise, rir_noise.view(), &mut state, Some(orig_len))?;
            debug_assert_eq!(speech.len_of(Axis(1)), noise.len_of(Axis(1)));
        }
        Ok(speech_rev)
    }
    fn convolve(
        &self,
        x: &Array2<f32>,
        rir: ArrayView2<f32>,
        state: &mut DFState,
        truncate: Option<usize>,
    ) -> Result<Array2<f32>> {
        let mut x_ = stft(x.as_standard_layout().view(), state, true);
        let rir = fft(rir.view(), state)?;
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
        create_out_dir().expect("Could not create output directory");
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
        create_out_dir().expect("Could not create output directory");
        let reader = ReadWav::new("../assets/clean_freesound_33711.wav")?;
        let sr = reader.sr as u32;
        let mut test_sample = reader.samples_arr2()?;
        let mut test_sample2 = test_sample.clone();
        let ch = test_sample.len_of(Axis(0)) as u16;
        let f = 8000.;
        low_pass(&mut test_sample, f, sr as usize, None)?;
        write_wav_iter("../out/lowpass.wav", test_sample.iter(), sr, ch)?;
        test_sample2 = low_pass_resample(&test_sample2, f as usize, sr as usize)?;
        write_wav_iter("../out/lowpass_resample.wav", test_sample2.iter(), sr, ch)?;
        Ok(())
    }

    #[test]
    pub fn test_reverb() -> Result<()> {
        setup();
        create_out_dir().expect("Could not create output directory");
        seed_from_u64(42);
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
        create_out_dir().expect("Could not create output directory");
        seed_from_u64(42);
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
}
