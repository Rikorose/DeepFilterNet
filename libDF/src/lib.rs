#![allow(dead_code)]

use std::ops::MulAssign;
use std::sync::Arc;
use std::vec::Vec;

use itertools::izip;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

pub type Complex32 = num_complex::Complex32;

pub const MEAN_NORM_INIT: [f32; 2] = [-60., -90.];
pub const UNIT_NORM_INIT: [f32; 2] = [0.001, 0.0001];

#[cfg(any(feature = "transforms", feature = "dataset"))]
pub mod transforms;
#[cfg(feature = "dataset")]
#[path = ""]
mod reexport_dataset_modules {
    pub mod augmentations;
    pub mod dataloader;
    pub mod dataset;
    pub mod hdf5_key_cache;
    pub mod util;
    pub mod wav_utils;
}
#[cfg(feature = "dataset")]
pub use reexport_dataset_modules::*;
#[cfg(feature = "capi")]
mod capi;
#[cfg(feature = "logging")]
pub mod logging;
#[cfg(feature = "tract")]
pub mod tract;

#[cfg(feature = "wasm")]
mod wasm;

#[cfg(all(feature = "wav-utils", not(feature = "dataset")))]
pub mod wav_utils;

pub(crate) fn freq2erb(freq_hz: f32) -> f32 {
    9.265 * (freq_hz / (24.7 * 9.265)).ln_1p()
}
pub(crate) fn erb2freq(n_erb: f32) -> f32 {
    24.7 * 9.265 * ((n_erb / 9.265).exp() - 1.)
}

#[derive(Clone)]
pub struct DFState {
    pub sr: usize,
    pub frame_size: usize,  // hop_size
    pub window_size: usize, // Same as fft_size
    pub freq_size: usize,   // fft_size / 2 + 1
    pub fft_forward: Arc<dyn RealToComplex<f32>>,
    pub fft_inverse: Arc<dyn ComplexToReal<f32>>,
    pub window: Vec<f32>,
    pub wnorm: f32,
    pub erb: Vec<usize>, // frequencies bandwidth (in bands) per ERB band
    analysis_mem: Vec<f32>,
    analysis_scratch: Vec<Complex32>,
    synthesis_mem: Vec<f32>,
    synthesis_scratch: Vec<Complex32>,
    mean_norm_state: Vec<f32>,
    unit_norm_state: Vec<f32>,
}

pub fn erb_fb(sr: usize, fft_size: usize, nb_bands: usize, min_nb_freqs: usize) -> Vec<usize> {
    // Init ERB filter bank
    let nyq_freq = sr / 2;
    let freq_width = sr as f32 / fft_size as f32;
    let erb_low: f32 = freq2erb(0.);
    let erb_high: f32 = freq2erb(nyq_freq as f32);
    let mut erb = vec![0; nb_bands];
    let step = (erb_high - erb_low) / nb_bands as f32;
    let min_nb_freqs = min_nb_freqs as i32; // Minimum number of frequency bands per erb band
    let mut prev_freq = 0; // Last frequency band of the previous erb band
    let mut freq_over = 0; // Number of frequency bands that are already stored in previous erb bands
    for i in 1..nb_bands + 1 {
        let f = erb2freq(erb_low + i as f32 * step);
        let fb = (f / freq_width).round() as usize;
        let mut nb_freqs = fb as i32 - prev_freq as i32 - freq_over;
        if nb_freqs < min_nb_freqs {
            // Not enough freq bins in current bark bin
            freq_over = min_nb_freqs - nb_freqs; // keep track of number of enforced bins
            nb_freqs = min_nb_freqs; // enforce min_nb_freqs
        } else {
            freq_over = 0
        }
        erb[i - 1] = nb_freqs as usize;
        prev_freq = fb;
    }
    erb[nb_bands - 1] += 1; // since we have WINDOW_SIZE/2+1 frequency bins
    let too_large = erb.iter().sum::<usize>() - (fft_size / 2 + 1);
    if too_large > 0 {
        erb[nb_bands - 1] -= too_large;
    }
    debug_assert!(erb.iter().sum::<usize>() == fft_size / 2 + 1);
    erb
}

// TODO Check delay for diferent hop sizes
impl DFState {
    pub fn new(
        sr: usize,
        fft_size: usize,
        hop_size: usize,
        nb_bands: usize,
        min_nb_freqs: usize,
    ) -> Self {
        assert!(hop_size * 2 <= fft_size);
        let mut fft = RealFftPlanner::<f32>::new();
        let frame_size = hop_size;
        let window_size = fft_size;
        let window_size_h = fft_size / 2;
        let freq_size = fft_size / 2 + 1;
        let forward = fft.plan_fft_forward(fft_size);
        let backward = fft.plan_fft_inverse(fft_size);
        let analysis_mem = vec![0.; fft_size - frame_size];
        let synthesis_mem = vec![0.; fft_size - frame_size];
        let analysis_scratch = forward.make_scratch_vec();
        let synthesis_scratch = backward.make_scratch_vec();

        let erb = erb_fb(sr, fft_size, nb_bands, min_nb_freqs);

        let pi = std::f64::consts::PI;
        // Initialize the vorbis window: sin(pi/2*sin^2(pi*n/N))
        let mut window = vec![0.0; fft_size];
        for (i, w) in window.iter_mut().enumerate() {
            let sin = (0.5 * pi * (i as f64 + 0.5) / window_size_h as f64).sin();
            *w = (0.5 * pi * sin * sin).sin() as f32;
        }
        let wnorm = 1. / (window_size.pow(2) as f32 / (2 * frame_size) as f32);
        let mean_norm_state = Vec::new();
        let unit_norm_state = Vec::new();

        DFState {
            sr,
            frame_size,
            window_size,
            freq_size,
            fft_forward: forward,
            fft_inverse: backward,
            erb,
            analysis_mem,
            analysis_scratch,
            synthesis_mem,
            synthesis_scratch,
            window,
            wnorm,
            mean_norm_state,
            unit_norm_state,
        }
    }

    pub fn reset(&mut self) {
        self.analysis_mem.fill(0.);
        self.synthesis_mem.fill(0.);
    }

    pub fn process_frame(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.frame_size);
        debug_assert_eq!(output.len(), self.frame_size);
        process_frame(input, output, self);
    }

    pub fn analysis(&mut self, input: &[f32], output: &mut [Complex32]) {
        debug_assert_eq!(input.len(), self.frame_size);
        debug_assert_eq!(output.len(), self.freq_size);
        frame_analysis(input, output, self)
    }

    pub fn synthesis(&mut self, input: &mut [Complex32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.freq_size);
        debug_assert_eq!(output.len(), self.frame_size);
        frame_synthesis(input, output, self)
    }

    pub fn init_norm_states(&mut self, nb_df_freqs: usize) {
        self.init_mean_norm_state();
        self.init_unit_norm_state(nb_df_freqs);
    }

    pub fn init_mean_norm_state(&mut self) {
        let min = MEAN_NORM_INIT[0];
        let max = MEAN_NORM_INIT[1];
        let nb_erb = self.erb.len();
        let step = (max - min) / (nb_erb - 1) as f32;
        let mut state = Vec::with_capacity(nb_erb);
        for i in 0..nb_erb {
            state.push(min + i as f32 * step);
        }
        self.mean_norm_state = state;
    }
    pub fn init_unit_norm_state(&mut self, nb_freqs: usize) {
        let min = UNIT_NORM_INIT[0];
        let max = UNIT_NORM_INIT[1];
        let step = (max - min) / (nb_freqs - 1) as f32;
        let mut state = Vec::with_capacity(nb_freqs);
        for i in 0..nb_freqs {
            state.push(min + i as f32 * step);
        }
        self.unit_norm_state = state;
    }

    pub fn feat_erb(&mut self, input: &[Complex32], alpha: f32, output: &mut [f32]) {
        compute_band_corr(output, input, input, &self.erb); // ERB FB
        for o in output.iter_mut() {
            *o = (*o + 1e-10).log10() * 10.;
        }
        band_mean_norm_erb(output, &mut self.mean_norm_state, alpha); // Exponential mean norm
    }

    pub fn feat_cplx(&mut self, input: &[Complex32], alpha: f32, output: &mut [Complex32]) {
        output.clone_from_slice(input);
        band_unit_norm(output, &mut self.unit_norm_state, alpha)
    }

    pub fn feat_cplx_t(&mut self, input: &[Complex32], alpha: f32, output: &mut [f32]) {
        band_unit_norm_t(input, &mut self.unit_norm_state, alpha, output)
    }

    pub fn apply_mask(&self, output: &mut [Complex32], gains: &[f32]) {
        apply_interp_band_gain(output, gains, &self.erb)
    }
}

impl Default for DFState {
    fn default() -> Self {
        Self::new(48000, 960, 480, 32, 2)
    }
}

pub fn band_mean_norm_freq(xs: &[Complex32], xout: &mut [f32], state: &mut [f32], alpha: f32) {
    debug_assert_eq!(xs.len(), state.len());
    debug_assert_eq!(xout.len(), state.len());
    for (x, s, xo) in izip!(xs.iter(), state.iter_mut(), xout.iter_mut()) {
        let xabs = x.norm();
        *s = xabs * (1. - alpha) + *s * alpha;
        *xo = xabs - *s;
    }
}

pub fn band_mean_norm_erb(xs: &mut [f32], state: &mut [f32], alpha: f32) {
    debug_assert_eq!(xs.len(), state.len());
    for (x, s) in xs.iter_mut().zip(state.iter_mut()) {
        *s = *x * (1. - alpha) + *s * alpha;
        *x -= *s;
        *x /= 40.;
    }
}

pub fn band_unit_norm(xs: &mut [Complex32], state: &mut [f32], alpha: f32) {
    debug_assert_eq!(xs.len(), state.len());
    for (x, s) in xs.iter_mut().zip(state.iter_mut()) {
        *s = x.norm() * (1. - alpha) + *s * alpha;
        *x /= s.sqrt();
    }
}

/// Band unit norm, but with transposed output type. I.e. out contains first all real elements,
/// followed by all imaginary elements. This memory layout is different from Complex32 slice which
/// contains real and imaginary part as interleaved values.
pub fn band_unit_norm_t(xs: &[Complex32], state: &mut [f32], alpha: f32, out: &mut [f32]) {
    debug_assert_eq!(xs.len(), state.len());
    debug_assert_eq!(xs.len(), out.len() / 2);
    let (o_re, o_im) = out.split_at_mut(xs.len());
    for (x, s, o_re, o_im) in izip!(
        xs.iter(),
        state.iter_mut(),
        o_re.iter_mut(),
        o_im.iter_mut(),
    ) {
        *s = x.norm() * (1. - alpha) + *s * alpha;
        *o_re /= s.sqrt();
        *o_im /= s.sqrt();
    }
}

pub fn compute_band_corr(out: &mut [f32], x: &[Complex32], p: &[Complex32], erb_fb: &[usize]) {
    for y in out.iter_mut() {
        *y = 0.0;
    }
    debug_assert_eq!(erb_fb.len(), out.len());

    let mut bcsum = 0;
    for (&band_size, out_b) in erb_fb.iter().zip(out.iter_mut()) {
        let k = 1. / band_size as f32;
        for j in 0..band_size {
            let idx = bcsum + j;
            *out_b += (x[idx].re * p[idx].re + x[idx].im * p[idx].im) * k;
        }
        bcsum += band_size;
    }
}

pub fn band_compr(out: &mut [f32], x: &[f32], erb_fb: &[usize]) {
    for y in out.iter_mut() {
        *y = 0.0;
    }
    debug_assert_eq!(erb_fb.len(), out.len());

    let mut bcsum = 0;
    for (&band_size, out_b) in erb_fb.iter().zip(out.iter_mut()) {
        let k = 1. / band_size as f32;
        for j in 0..band_size {
            let idx = bcsum + j;
            *out_b += x[idx] * k;
        }
        bcsum += band_size;
    }
}

pub fn apply_interp_band_gain<T>(out: &mut [T], band_e: &[f32], erb_fb: &[usize])
where
    T: MulAssign<f32>,
{
    let mut bcsum = 0;
    for (&band_size, &b) in erb_fb.iter().zip(band_e.iter()) {
        for j in 0..band_size {
            let idx = bcsum + j;
            out[idx] *= b;
        }
        bcsum += band_size;
    }
}

fn interp_band_gain(out: &mut [f32], band_e: &[f32], erb_fb: &[usize]) {
    let mut bcsum = 0;
    for (&band_size, &b) in erb_fb.iter().zip(band_e.iter()) {
        for j in 0..band_size {
            let idx = bcsum + j;
            out[idx] = b;
        }
        bcsum += band_size;
    }
}

fn apply_band_gain(out: &mut [Complex32], band_e: &[f32], erb_fb: &[usize]) {
    let mut bcsum = 0;
    for (&band_size, b) in erb_fb.iter().zip(band_e.iter()) {
        for j in 0..band_size {
            let idx = bcsum + j;
            out[idx] *= *b;
        }
        bcsum += band_size;
    }
}

fn process_frame(input: &[f32], output: &mut [f32], state: &mut DFState) {
    let mut freq_mem = vec![Complex32::default(); state.freq_size];
    frame_analysis(input, &mut freq_mem, state);
    frame_synthesis(&mut freq_mem, output, state);
}

fn frame_analysis(input: &[f32], output: &mut [Complex32], state: &mut DFState) {
    debug_assert_eq!(input.len(), state.frame_size);
    debug_assert_eq!(output.len(), state.freq_size);

    let mut buf = state.fft_forward.make_input_vec();
    // First part of the window on the previous frame
    let (buf_first, buf_second) = buf.split_at_mut(state.window_size - state.frame_size);
    let (window_first, window_second) = state.window.split_at(state.window_size - state.frame_size);
    let analysis_split = state.analysis_mem.len() - state.frame_size;
    for (&y, &w, x) in izip!(
        state.analysis_mem.iter(),
        window_first.iter(),
        buf_first.iter_mut(),
    ) {
        *x = y * w;
    }
    // Second part of the window on the new input frame
    for ((&y, &w), x) in input.iter().zip(window_second.iter()).zip(buf_second.iter_mut()) {
        *x = y * w;
    }
    // Shift analysis_mem
    if analysis_split > 0 {
        // hop_size is < window_size / 2
        state.analysis_mem.rotate_left(state.frame_size);
    }
    // Copy input to analysis_mem for next iteration
    for (x, &y) in state.analysis_mem[analysis_split..].iter_mut().zip(input) {
        *x = y
    }
    state
        .fft_forward
        .process_with_scratch(&mut buf, output, &mut state.analysis_scratch)
        .expect("FFT forward failed");
    // Apply normalization in analysis only
    let norm = state.wnorm;
    for x in output.iter_mut() {
        *x *= norm;
    }
}

fn frame_synthesis(input: &mut [Complex32], output: &mut [f32], state: &mut DFState) {
    let mut x = state.fft_inverse.make_output_vec();
    match state
        .fft_inverse
        .process_with_scratch(input, &mut x, &mut state.synthesis_scratch)
    {
        Err(realfft::FftError::InputValues(_, _)) => (),
        Err(e) => panic!("Error during fft_inverse: {:?}", e),
        Ok(_) => (),
    }
    apply_window_in_place(&mut x, &state.window);
    let (x_first, x_second) = x.split_at(state.frame_size);
    for ((&xi, &mem), out) in x_first.iter().zip(state.synthesis_mem.iter()).zip(output.iter_mut())
    {
        *out = xi + mem;
    }

    let split = state.synthesis_mem.len() - state.frame_size;
    if split > 0 {
        state.synthesis_mem.rotate_left(state.frame_size);
    }
    let (s_first, s_second) = state.synthesis_mem.split_at_mut(split);
    let (xs_first, xs_second) = x_second.split_at(split);
    for (&xi, mem) in xs_first.iter().zip(s_first.iter_mut()) {
        // Overlap add for next frame
        *mem += xi;
    }
    for (&xi, mem) in xs_second.iter().zip(s_second.iter_mut()) {
        // Override left shifted buffer
        *mem = xi;
    }
}

fn apply_window(xs: &[f32], window: &[f32]) -> Vec<f32> {
    let mut out = vec![0.; window.len()];
    for (&x, &w, o) in izip!(xs.iter(), window.iter(), out.iter_mut()) {
        *o = x * w;
    }
    out
}

fn apply_window_in_place<'a, I>(xs: &mut [f32], window: I)
where
    I: IntoIterator<Item = &'a f32>,
{
    for (x, &w) in xs.iter_mut().zip(window) {
        *x *= w;
    }
}

pub fn post_filter(noisy: &[Complex32], enh: &mut [Complex32], beta: f32) {
    let beta_p1 = beta + 1.;
    let eps = 1e-12;
    let pi = std::f32::consts::PI;
    let mut g = [0.0; 4];
    let mut g_sin = [0.0; 4];
    let mut pf = [0.0; 4];
    for (n, e) in noisy.chunks_exact(4).zip(enh.chunks_exact_mut(4)) {
        g[0] = (e[0].norm() / (n[0].norm() + eps)).min(1.).max(eps);
        g[1] = (e[1].norm() / (n[1].norm() + eps)).min(1.).max(eps);
        g[2] = (e[2].norm() / (n[2].norm() + eps)).min(1.).max(eps);
        g[3] = (e[3].norm() / (n[3].norm() + eps)).min(1.).max(eps);
        g_sin[0] = g[0] * (g[0] * pi / 2.0).sin();
        g_sin[1] = g[1] * (g[1] * pi / 2.0).sin();
        g_sin[2] = g[2] * (g[2] * pi / 2.0).sin();
        g_sin[3] = g[3] * (g[3] * pi / 2.0).sin();
        pf[0] = (beta_p1 * g[0] / (1. + beta * (g[0] / g_sin[0]).powi(2))) / g[0];
        pf[1] = (beta_p1 * g[1] / (1. + beta * (g[1] / g_sin[1]).powi(2))) / g[1];
        pf[2] = (beta_p1 * g[2] / (1. + beta * (g[2] / g_sin[2]).powi(2))) / g[2];
        pf[3] = (beta_p1 * g[3] / (1. + beta * (g[3] / g_sin[3]).powi(2))) / g[3];
        e[0] *= pf[0];
        e[1] *= pf[1];
        e[2] *= pf[2];
        e[3] *= pf[3];
    }
}

pub(crate) struct NonNan(f32);

impl NonNan {
    fn new(val: f32) -> Option<NonNan> {
        if val.is_nan() {
            None
        } else {
            Some(NonNan(val))
        }
    }
    fn get(&self) -> f32 {
        self.0
    }
}

pub fn find_max<'a, I>(vals: I) -> Option<f32>
where
    I: IntoIterator<Item = &'a f32>,
{
    vals.into_iter().try_fold(f32::MIN, |acc, v| {
        let nonnan: NonNan = match NonNan::new(*v) {
            None => return None,
            Some(x) => x,
        };
        Some(nonnan.get().max(acc))
    })
}

pub fn find_max_abs<'a, I>(vals: I) -> Option<f32>
where
    I: IntoIterator<Item = &'a f32>,
{
    vals.into_iter().try_fold(0., |acc, v| {
        let nonnan: NonNan = match NonNan::new(v.abs()) {
            None => return None,
            Some(x) => x,
        };
        Some(nonnan.get().max(acc))
    })
}

pub fn find_min<'a, I>(vals: I) -> Option<f32>
where
    I: IntoIterator<Item = &'a f32>,
{
    vals.into_iter().try_fold(f32::MAX, |acc, v| {
        let nonnan: NonNan = match NonNan::new(*v) {
            None => return None,
            Some(x) => x,
        };
        Some(nonnan.get().min(acc))
    })
}

pub fn find_min_abs<'a, I>(vals: I) -> Option<f32>
where
    I: IntoIterator<Item = &'a f32>,
{
    vals.into_iter().try_fold(0., |acc, v| {
        let nonnan: NonNan = match NonNan::new(v.abs()) {
            None => return None,
            Some(x) => x,
        };
        Some(nonnan.get().min(acc))
    })
}

pub fn argmax<'a, I>(vals: I) -> Option<usize>
where
    I: IntoIterator<Item = &'a f32>,
{
    let mut index = 0;
    let mut high = f32::MIN;
    vals.into_iter().enumerate().for_each(|(i, v)| {
        if v > &high {
            high = *v;
            index = i;
        }
    });
    Some(index)
}

pub fn argmax_abs<'a, I>(vals: I) -> Option<usize>
where
    I: IntoIterator<Item = &'a f32>,
{
    let mut index = 0;
    let mut high = f32::MIN;
    vals.into_iter().enumerate().for_each(|(i, v)| {
        if v > &high {
            high = v.abs();
            index = i;
        }
    });
    Some(index)
}

pub fn rms<'a, I>(vals: I) -> f32
where
    I: IntoIterator<Item = &'a f32>,
{
    let mut n = 0;
    let pow_sum = vals.into_iter().fold(0., |acc, v| {
        n += 1;
        acc + v.powi(2)
    });
    (pow_sum / n as f32).sqrt()
}
pub fn rms_v<I>(vals: I) -> f32
where
    I: IntoIterator<Item = f32>,
{
    let mut n = 0;
    let pow_sum = vals.into_iter().fold(0., |acc, v| {
        n += 1;
        acc + v.powi(2)
    });
    (pow_sum / n as f32).sqrt()
}

pub fn mean<'a, I>(vals: I) -> f32
where
    I: IntoIterator<Item = &'a f32>,
{
    let mut n = 0;
    let sum = vals.into_iter().fold(0., |acc, v| {
        n += 1;
        acc + v
    });
    sum / n as f32
}

pub fn median<T>(x: &mut [T]) -> T
where
    T: PartialOrd<T> + Copy,
{
    if x.len() == 1 {
        return x[0];
    }
    if x.is_empty() {
        panic!("Empty input slice");
    }
    x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = x.len() / 2;
    x[mid]
}

#[cfg(test)]
mod tests {
    use rand::distributions::{Distribution, Uniform};

    use super::*;

    #[test]
    fn test_erb_inout() {
        let sr = 24000;
        let n_fft = 192;
        let n_freqs = n_fft / 2 + 1;
        let hop = n_fft / 2;
        let nb_bands = 24;
        let state = DFState::new(sr, n_fft, hop, nb_bands, 1);
        let d = Uniform::new(-1., 1.);
        let mut input = Vec::with_capacity(n_freqs);
        let mut rng = rand::thread_rng();
        for _ in 0..(n_freqs) {
            input.push(Complex32::new(d.sample(&mut rng), d.sample(&mut rng)))
        }
        let mut mask = vec![1.; nb_bands];
        mask[3] = 0.3;
        mask[nb_bands - 1] = 0.5;
        let mut output = input.clone();
        apply_band_gain(&mut output, mask.as_slice(), &state.erb);
        let mut cumsum = 0;
        for (erb_idx, erb_w) in state.erb.iter().enumerate() {
            for i in cumsum..cumsum + erb_w {
                assert_eq!(input[i] * mask[erb_idx], output[i])
            }
            cumsum += erb_w;
        }
    }
}
