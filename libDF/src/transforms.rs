use std::mem::MaybeUninit;

use ndarray::{prelude::*, Slice};
use rand::{distributions::uniform::Uniform, Rng};
use rubato::{FftFixedInOut, Resampler};
use thiserror::Error;

pub use crate::util::seed_from_u64;
use crate::util::*;
use crate::*;

type Result<T> = std::result::Result<T, TransformError>;

#[derive(Error, Debug)]
pub enum TransformError {
    #[error("DF UtilsError")]
    UtilsError(#[from] UtilsError),
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

fn biquad_norm_inplace<'a, I>(xs: I, mem: &mut [f32; 2], b: &[f32; 2], a: &[f32; 2])
where
    I: IntoIterator<Item = &'a mut f32>,
{
    // a0 and b0 are assumed to be 1
    let a1 = a[0] as f64;
    let a2 = a[1] as f64;
    let b1 = b[0] as f64;
    let b2 = b[1] as f64;
    for x in xs.into_iter() {
        let x64 = *x as f64;
        let y64 = x64 + mem[0] as f64;
        mem[0] = (mem[1] as f64 + (b1 * x64 - a1 * y64)) as f32;
        mem[1] = (b2 * x64 - a2 * y64) as f32;
        *x = y64 as f32;
    }
}

fn biquad_inplace<'a, I>(xs: I, mem: &mut [f32; 2], b: &[f32; 3], a: &[f32; 3])
where
    I: IntoIterator<Item = &'a mut f32>,
{
    let a0 = a[0] as f64;
    let a1 = a[1] as f64 / a0;
    let a2 = a[2] as f64 / a0;
    let b0 = b[0] as f64 / a0;
    let b1 = b[1] as f64 / a0;
    let b2 = b[2] as f64 / a0;
    for x in xs.into_iter() {
        let x64 = *x as f64;
        let y64 = b0 * x64 + mem[0] as f64;
        mem[0] = (mem[1] as f64 + (b1 * x64 - a1 * y64)) as f32;
        mem[1] = (b2 * x64 - a2 * y64) as f32;
        *x = y64 as f32;
    }
}

pub trait Transform {
    fn transform(&self, x: &mut Array2<f32>) -> Result<()>;
    fn default_with_prob(p: f32) -> Self
    where
        Self: Sized;
    fn box_clone(&self) -> Box<dyn Transform>;
}

impl Clone for Box<dyn Transform> {
    fn clone(&self) -> Box<dyn Transform> {
        self.box_clone()
    }
}

pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}
unsafe impl Send for Compose {}
unsafe impl Sync for Compose {}

impl Compose {
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Compose { transforms }
    }

    pub fn transform(&self, x: &mut Array2<f32>) -> Result<()> {
        for t in self.transforms.iter() {
            t.transform(x)?;
        }
        Ok(())
    }
}
impl Clone for Compose {
    fn clone(&self) -> Self {
        Compose {
            transforms: self.transforms.clone(),
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
        if self.prob == 0. || (self.prob < 1. && thread_rng()?.gen_range(0f32..1f32) > self.prob) {
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
    fn box_clone(&self) -> Box<dyn Transform> {
        Box::new((*self).clone())
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
            return Err(TransformError::NotInitialized {
                transform: "RandEQ".into(),
                msg: "No sampling rate provided.".into(),
            });
        }
        let mut rng = thread_rng()?;
        if self.prob == 0. || (self.prob < 1. && rng.gen_range(0f32..1f32) > self.prob) {
            return Ok(());
        }
        for _ in 0..self.n_freqs {
            let freq = rng.log_uniform(self.f_low as f32, self.f_high as f32);
            let gain = rng.gen_range(-self.gain_db..=self.gain_db) as f32;
            let q = rng.gen_range(self.q_low..self.q_high);
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
        let sr = if is_init() { Some(common().sr) } else { None };
        RandEQ {
            prob: p,
            sr,
            n_freqs: 3,
            f_low: 40,
            f_high: 8000,
            gain_db: 15,
            q_low: 0.5,
            q_high: 1.5,
        }
    }
    fn box_clone(&self) -> Box<dyn Transform> {
        Box::new((*self).clone())
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
    let mut resampler = FftFixedInOut::<f32>::new(sr, new_sr, chunk_size.unwrap_or(1024), channels);
    let chunk_size = resampler.nbr_frames_needed();
    let num_chunks = (len as f32 / chunk_size as f32).ceil() as usize;
    let chunk_size_out = (chunk_size as u64 * new_sr as u64 / sr as u64) as usize;
    let mut out = Array2::uninit((channels, chunk_size_out * num_chunks));
    let mut inbuf = vec![vec![0f32; chunk_size]; channels];
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
        let resampled = resampler.process(&inbuf)?;
        for (res_ch, mut out_ch) in resampled.iter().zip(out_chunk.axis_iter_mut(Axis(0))) {
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
            return Err(TransformError::NotInitialized {
                transform: "RandEQ".into(),
                msg: "No sampling rate provided.".into(),
            });
        }
        let mut rng = thread_rng()?;
        let sr = self.sr.unwrap();
        if self.prob == 0. || (self.prob < 1. && rng.gen_range(0f32..1f32) > self.prob) {
            return Ok(());
        }
        let ch = x.len_of(Axis(0));
        let len = x.len_of(Axis(1));
        let new_sr = rng.gen_range(self.r_low..=self.r_high) * sr as f32;
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
        let sr = if is_init() { Some(common().sr) } else { None };
        RandResample {
            prob: p,
            sr,
            r_low: 0.9,
            r_high: 1.1,
            chunk_size: 1024,
        }
    }
    fn box_clone(&self) -> Box<dyn Transform> {
        Box::new((*self).clone())
    }
}

#[derive(Clone)]
pub struct RandRemoveDc {
    prob: f32,
}
impl Transform for RandRemoveDc {
    fn transform(&self, x: &mut Array2<f32>) -> Result<()> {
        if self.prob == 0. || (self.prob < 1. && thread_rng()?.gen_range(0f32..1f32) > self.prob) {
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
    fn box_clone(&self) -> Box<dyn Transform> {
        Box::new((*self).clone())
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

    let mut mem = [0.; 2];
    for x_ch in x.axis_iter_mut(Axis(0)) {
        biquad_inplace(x_ch, &mut mem, &[b0, b1, b2], &[a0, a1, a2]);
    }
    Ok(())
}

pub(crate) fn mix_f(clean: ArrayView2<f32>, noise: ArrayView2<f32>, snr_db: f32) -> f32 {
    let e_clean = clean.iter().fold(0f32, |acc, x| acc + x.powi(2)) + 1e-10;
    let e_noise = noise.iter().fold(0f32, |acc, x| acc + x.powi(2)) + 1e-10;
    let snr = 10f32.powf(snr_db / 10.);
    (1f64 / (((e_noise / e_clean) * snr + 1e-10) as f64).sqrt()) as f32
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
        // Zero pad RIR for better FFT efficiency: len = 2**a + 3**b + 5**c + 7**d
        let len = rir.len_of(Axis(1));
        let mut missing = len as i32;
        let d = (missing as f32).log(7.).floor() as u32;
        missing -= 7i32.pow(d);
        let c = (missing as f32).log(5.).floor() as u32;
        missing -= 5i32.pow(c);
        let b = (missing as f32).log(3.).floor() as u32;
        missing -= 3i32.pow(b);
        let mut a = (missing as f32).log2().floor() as u32;
        missing -= 2i32.pow(a);
        if missing > 0 {
            a += 1
        };
        let fft_size = (2u32.pow(a) + 3u32.pow(b) + 5u32.pow(c) + 7u32.pow(d)) as usize + 1;
        assert!(fft_size >= len);
        fft_size
    }
    fn pad(&self, x: &mut Array2<f32>, npad: usize) -> Result<()> {
        if npad == 0 {
            return Ok(());
        }
        x.append(Axis(1), Array2::zeros((x.len_of(Axis(0)), npad)).view())?;
        Ok(())
    }
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
                return Err(TransformError::DfError(format!(
                    "Error getting RIR in RandReverbSim::transform() {:?}",
                    e
                )));
            }
        };
        let mut rng = thread_rng()?;
        let apply_speech = self.prob_speech > rng.gen_range(0f32..1f32);
        let apply_noise = self.prob_noise > rng.gen_range(0f32..1f32);
        if !(apply_speech || apply_noise) {
            return Ok(None);
        }
        let orig_len = speech.len_of(Axis(1));
        // Maybe resample RIR as augmentation
        if self.prob_resample > rng.gen_range(0f32..1f32) {
            let new_sr: f32 = rng.gen_range(0.8..1.2) * self.sr as f32;
            let new_sr = ((new_sr / 500.).round() * 500.) as usize;
            rir = resample(&rir, self.sr, new_sr, Some(512))?;
        }
        if self.prob_decay > rng.gen_range(0f32..1f32) {
            let rt60 = rng.gen_range(0.2..1.);
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
        let c = init(self.sr, fft_size, hop_size, 1, 1);
        let mut state = DenoiseState::new(fft_size, hop_size);

        let apply_speech = true;
        let apply_noise = false;

        // speech_rev contains reverberant speech for mixing with noise
        let mut speech_rev = None;
        if apply_speech {
            // Pad since STFT will truncate at the end
            self.pad(speech, fft_size)?;
            if !apply_noise {
                speech_rev = match self.convolve(speech, rir_noise.view(), &mut state, &c) {
                    Ok(mut s) => {
                        s.slice_collapse(s![.., ..orig_len]);
                        if s.len_of(Axis(1)) != noise.len_of(Axis(1)) {
                            panic!(
                                "Len of speech {:?} and noise {:?} does not match.",
                                s.shape(),
                                noise.shape()
                            );
                        }
                        Some(s)
                    }
                    Err(e) => {
                        eprintln!("speech_rev {}", e);
                        return Ok(None);
                    }
                };
            }
            // Speech should be a slightly dereverberant signal as target
            let rir_speech = self.supress_late(rir_noise.clone(), self.sr, 5., 0.2, false)?;
            *speech = match self.convolve(speech, rir_speech.view(), &mut state, &c) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("speech {}", e);
                    return Ok(None);
                }
            };
            speech.slice_collapse(s![.., ..orig_len]);
        }
        if apply_noise {
            // Noisy contains reverberant noise
            self.pad(noise, fft_size)?;
            *noise = match self.convolve(noise, rir_noise.view(), &mut state, &c) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("noise {}", e);
                    return Ok(None);
                }
            };
        }
        // Trim zeros at the beginning
        noise.slice_collapse(s![.., ..orig_len]);
        Ok(speech_rev)
    }
    fn convolve(
        &self,
        x: &Array2<f32>,
        rir: ArrayView2<f32>,
        state: &mut DenoiseState,
        c: &CommonState,
    ) -> Result<Array2<f32>> {
        let mut x_ = stft(x.as_standard_layout().view(), state, true, Some(c));
        let rir = fft(rir.view(), state)?;
        let rir: ArrayView3<Complex32> = match rir.broadcast(x_.shape()) {
            Some(r) => r.into_dimensionality()?,
            None => {
                return Err(TransformError::DfError(format!(
                    "Shape missmatch: {:?} {:?}",
                    x_.shape(),
                    rir.shape()
                )));
            }
        };
        x_ = x_ * rir;
        let mut out = istft(x_.view_mut(), state, true, Some(c));
        out.slice_collapse(s![.., (c.fft_size - c.frame_size)..]);
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

pub fn fft(input: ArrayView2<f32>, state: &mut DenoiseState) -> Result<Array2<Complex32>> {
    let mut output = Array2::zeros((input.len_of(Axis(0)), state.freq_size));
    for (input_ch, mut output_ch) in input.outer_iter().zip(output.outer_iter_mut()) {
        state
            .fft_forward
            .process(
                input_ch.into_owned().as_slice_mut().unwrap(),
                //&mut apply_window(input_ch.as_slice().unwrap(), &c.window),
                &mut output_ch.as_slice_mut().unwrap(),
            )
            .map_err(|e| TransformError::DfError(format!("Error in fft(): {:?}", e)))?;
    }
    Ok(output)
}

pub fn stft(
    input: ArrayView2<f32>,
    state: &mut DenoiseState,
    reset: bool,
    c: Option<&CommonState>,
) -> Array3<Complex32> {
    if reset {
        state.reset();
    }
    let c = c.unwrap_or_else(|| common());
    let ch = input.len_of(Axis(0));
    let ttd = input.len_of(Axis(1));
    let tfd = ttd / state.frame_size;
    let mut output: Array3<Complex32> = Array3::zeros((ch, tfd, state.freq_size));
    for (input_ch, mut output_ch) in input.outer_iter().zip(output.outer_iter_mut()) {
        for (ichunk, mut ochunk) in input_ch
            .axis_chunks_iter(Axis(0), state.frame_size)
            .zip(output_ch.outer_iter_mut())
        {
            frame_analysis(
                ichunk.as_slice().unwrap(),
                ochunk.as_slice_mut().unwrap(),
                state,
                c,
            )
        }
    }
    output
}

pub fn istft(
    mut input: ArrayViewMut3<Complex32>,
    state: &mut DenoiseState,
    reset: bool,
    c: Option<&CommonState>,
) -> Array2<f32> {
    if reset {
        state.reset();
    }
    let c = c.unwrap_or_else(|| common());
    let ch = input.len_of(Axis(0));
    let tfd = input.len_of(Axis(1));
    let ttd = tfd * state.frame_size;
    let mut output: Array2<f32> = Array2::zeros((ch, ttd));
    for (mut input_ch, mut output_ch) in input.outer_iter_mut().zip(output.outer_iter_mut()) {
        for (mut ichunk, mut ochunk) in
            input_ch.outer_iter_mut().zip(output_ch.exact_chunks_mut(state.frame_size))
        {
            frame_synthesis(
                &mut ichunk.as_slice_mut().unwrap(),
                &mut ochunk.as_slice_mut().unwrap(),
                state,
                c,
            )
        }
    }
    output
}

pub fn erb_compr_with_output(
    input: &ArrayView3<f32>,
    output: &mut ArrayViewMut3<f32>,
    c: Option<&CommonState>,
) -> Result<()> {
    let c = c.unwrap_or_else(|| common());
    for (in_ch, mut out_ch) in input.outer_iter().zip(output.outer_iter_mut()) {
        for (in_t, mut out_t) in in_ch.outer_iter().zip(out_ch.outer_iter_mut()) {
            let ichunk = in_t.as_slice().unwrap();
            let ochunk = out_t.as_slice_mut().unwrap();
            band_compr(ochunk, ichunk, Some(c));
        }
    }
    Ok(())
}

pub fn erb_with_output(
    input: &ArrayView3<Complex32>,
    db: bool,
    output: &mut ArrayViewMut3<f32>,
    c: Option<&CommonState>,
) -> Result<()> {
    let c = c.unwrap_or_else(|| common());
    for (in_ch, mut out_ch) in input.outer_iter().zip(output.outer_iter_mut()) {
        for (in_t, mut out_t) in in_ch.outer_iter().zip(out_ch.outer_iter_mut()) {
            let ichunk = in_t.as_slice().unwrap();
            let ochunk = out_t.as_slice_mut().unwrap();
            compute_band_corr(ochunk, ichunk, ichunk, Some(c));
        }
    }
    if db {
        output.mapv_inplace(|v| (v + 1e-10).log10() * 10.);
    }
    Ok(())
}

pub fn erb(
    input: &ArrayView3<Complex32>,
    db: bool,
    c: Option<&CommonState>,
) -> Result<Array3<f32>> {
    // input shape: [C, T, F]
    let ch = input.len_of(Axis(0));
    let t = input.len_of(Axis(1));
    if !is_init() {
        return Err(TransformError::NotInitialized {
            transform: "erb".into(),
            msg: "DF common state was not initialized. Please call init() first.".into(),
        });
    }
    let mut output = Array3::<f32>::zeros((ch, t, common().nb_bands));

    erb_with_output(input, db, &mut output.view_mut(), c)?;
    Ok(output)
}

pub fn apply_erb_gains(
    gains: &ArrayView3<f32>,
    input: &mut ArrayViewMut3<Complex32>,
    c: Option<&CommonState>,
) -> Result<()> {
    // gains shape: [C, T, E]
    // input shape: [C, T, F]
    let c = c.unwrap_or_else(|| common());
    for (g_ch, mut in_ch) in gains.outer_iter().zip(input.outer_iter_mut()) {
        for (g_t, mut in_t) in g_ch.outer_iter().zip(in_ch.outer_iter_mut()) {
            apply_interp_band_gain(
                in_t.as_slice_mut().unwrap(),
                g_t.as_slice().unwrap(),
                Some(c),
            );
        }
    }
    Ok(())
}

pub fn erb_inv_with_output(
    gains: &ArrayView3<f32>,
    output: &mut ArrayViewMut3<f32>,
    c: Option<&CommonState>,
) -> Result<()> {
    // gains shape: [C, T, E]
    // output shape: [C, T, F]
    let c = c.unwrap_or_else(|| common());
    for (g_ch, mut o_ch) in gains.outer_iter().zip(output.outer_iter_mut()) {
        for (g_t, mut o_t) in g_ch.outer_iter().zip(o_ch.outer_iter_mut()) {
            interp_band_gain(
                o_t.as_slice_mut().unwrap(),
                g_t.as_slice().unwrap(),
                Some(c),
            );
        }
    }
    Ok(())
}

pub fn erb_norm(
    input: &mut ArrayViewMut3<f32>,
    state: Option<Array2<f32>>,
    alpha: f32,
) -> Result<Array2<f32>> {
    // input shape: [C, T, F]
    // state shape: [C, F]
    let mut state = state.unwrap_or_else(|| {
        let b = input.len_of(Axis(2));
        Array1::<f32>::linspace(MEAN_NORM_INIT[0], MEAN_NORM_INIT[1], b)
            .into_shape([1, b])
            .unwrap()
    });
    for (mut in_ch, mut s_ch) in input.outer_iter_mut().zip(state.outer_iter_mut()) {
        for mut in_step in in_ch.outer_iter_mut() {
            band_mean_norm_erb(
                in_step.as_slice_mut().unwrap(),
                s_ch.as_slice_mut().unwrap(),
                alpha,
            )
        }
    }
    Ok(state)
}

pub fn unit_norm(
    input: &mut ArrayViewMut3<Complex32>,
    state: Option<Array2<f32>>,
    alpha: f32,
) -> Result<Array2<f32>> {
    // input shape: [C, T, F]
    // state shape: [C, F]
    let mut state = state.unwrap_or_else(|| {
        let f = input.len_of(Axis(2));
        let state_ch0 = Array1::<f32>::linspace(UNIT_NORM_INIT[0], UNIT_NORM_INIT[1], f)
            .into_shape([1, f])
            .unwrap();
        let mut state = state_ch0.clone();
        for _ in 1..input.len_of(Axis(0)) {
            state.append(Axis(0), state_ch0.view()).unwrap()
        }
        state
    });
    debug_assert_eq!(state.len_of(Axis(0)), input.len_of(Axis(0)));
    for (mut in_ch, mut s_ch) in input.outer_iter_mut().zip(state.outer_iter_mut()) {
        for mut in_step in in_ch.outer_iter_mut() {
            band_unit_norm(
                in_step.as_slice_mut().unwrap(),
                s_ch.as_slice_mut().unwrap(),
                alpha,
            )
        }
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav_utils::*;

    #[test]
    pub fn test_rand_resample() -> Result<()> {
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
        seed_from_u64(42);
        let reader = ReadWav::new("../assets/clean_freesound_33711.wav")?;
        let sr = reader.sr;
        let len = 10 * sr;
        let mut speech = reader.samples_arr2()?;
        speech.slice_collapse(s![0..1, 0..len]);
        let mut noise = ReadWav::new("../assets/noise_freesound_2530.wav")?.samples_arr2()?;
        noise.slice_axis_inplace(Axis(1), Slice::from(0..len));
        //let rir = ReadWav::new("../assets/rir.wav")?.samples_arr2()?;
        //let rir = ReadWav::new("../assets/aachen_rir_air_booth_0_0_1.wav")?.samples_arr2()?;
        //let rir = ReadWav::new("../assets/rir_sim_0_w11.6_l7.4_h3.2_rt60_0.2164.wav")?.samples_arr2()?;
        let rir =
            ReadWav::new("../assets/rir_sim_0_w4.8_l10.2_h3.6_rt60_0.9466.wav")?.samples_arr2()?;
        let reverb = RandReverbSim::new(1., sr);
        let noisy = reverb.transform(&mut speech, &mut noise, move || Ok(rir))?.unwrap();
        write_wav_arr2("../out/speech_reverb.wav", speech.view(), sr as u32)?;
        write_wav_arr2("../out/noise_reverb.wav", noise.view(), sr as u32)?;
        write_wav_arr2("../out/noisy_reverb.wav", noisy.view(), sr as u32)?;
        Ok(())
    }
}
