#![allow(dead_code)]

use std::ops::MulAssign;
use std::sync::Arc;
use std::vec::Vec;

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
    pub mod util;
    pub mod wav_utils;
}
#[cfg(feature = "dataset")]
pub use reexport_dataset_modules::*;
#[cfg(feature = "cache")]
mod cache;

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
        let mut window = vec![0.0; fft_size];
        for (i, w) in window.iter_mut().enumerate() {
            let sin = (0.5 * pi * (i as f64 + 0.5) / window_size_h as f64).sin();
            *w = (0.5 * pi * sin * sin).sin() as f32;
        }
        let wnorm =
            1_f32 / window.iter().map(|x| x * x).sum::<f32>() * frame_size as f32 / fft_size as f32;

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
        frame_analysis(input, output, self)
    }

    pub fn synthesis(&mut self, input: &mut [Complex32], output: &mut [f32]) {
        debug_assert_eq!(output.len(), self.frame_size);
        frame_synthesis(input, output, self)
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
    for ((x, s), xo) in xs.iter().zip(state.iter_mut()).zip(xout.iter_mut()) {
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

fn apply_interp_band_gain<T>(out: &mut [T], band_e: &[f32], erb_fb: &[usize])
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
    for ((&y, &w), x) in
        state.analysis_mem.iter().zip(window_first.iter()).zip(buf_first.iter_mut())
    {
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
        .process_with_scratch(input, &mut x[..], &mut state.synthesis_scratch)
    {
        Err(realfft::FftError::InputValues(_, _)) => (),
        Err(e) => Err(e).unwrap(),
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
    for ((&x, &w), o) in xs.iter().zip(window.iter()).zip(out.iter_mut()) {
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
