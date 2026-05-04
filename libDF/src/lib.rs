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
        // apply_band_gain is the Complex32 specialisation of apply_interp_band_gain
        // and carries a SIMD-vectorised inner loop on wasm32.
        apply_band_gain(output, gains, &self.erb)
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
    band_mean_norm_erb_inner(xs, state, alpha);
}

pub fn band_unit_norm(xs: &mut [Complex32], state: &mut [f32], alpha: f32) {
    debug_assert_eq!(xs.len(), state.len());
    band_unit_norm_inner(xs, state, alpha);
}

/// Band unit norm, but with transposed output type. I.e. out contains first all real elements,
/// followed by all imaginary elements. This memory layout is different from Complex32 slice which
/// contains real and imaginary part as interleaved values.
pub fn band_unit_norm_t(xs: &[Complex32], state: &mut [f32], alpha: f32, out: &mut [f32]) {
    debug_assert_eq!(xs.len(), state.len());
    debug_assert_eq!(xs.len(), out.len() / 2);
    let (o_re, o_im) = out.split_at_mut(xs.len());
    band_unit_norm_t_inner(xs, state, alpha, o_re, o_im);
}

pub fn compute_band_corr(out: &mut [f32], x: &[Complex32], p: &[Complex32], erb_fb: &[usize]) {
    for y in out.iter_mut() {
        *y = 0.0;
    }
    debug_assert_eq!(erb_fb.len(), out.len());
    debug_assert_eq!(x.len(), p.len());

    // Each Complex32 occupies 2 contiguous f32 (re, im). Reinterpret the slices
    // as flat &[f32] of length 2*N so we can vectorize with f32x4 loads.
    // SAFETY: Complex32 is #[repr(C)] { re: f32, im: f32 } -> 8 bytes, alignment 4,
    // identical to two contiguous f32. Length is exactly 2 * x.len().
    let xf: &[f32] =
        unsafe { core::slice::from_raw_parts(x.as_ptr() as *const f32, x.len() * 2) };
    let pf: &[f32] =
        unsafe { core::slice::from_raw_parts(p.as_ptr() as *const f32, p.len() * 2) };

    let mut bcsum = 0usize;
    for (&band_size, out_b) in erb_fb.iter().zip(out.iter_mut()) {
        let k = 1.0f32 / band_size as f32;
        let f_start = bcsum * 2;
        let f_len = band_size * 2;
        let xb = &xf[f_start..f_start + f_len];
        let pb = &pf[f_start..f_start + f_len];
        // sum := sum over band of x[i].re*p[i].re + x[i].im*p[i].im
        // == sum over flattened pairs of xb[2j]*pb[2j] + xb[2j+1]*pb[2j+1]
        // == sum_lanes( sum over 4-wide chunks of xb[..]*pb[..] )
        let sum: f32 = compute_band_corr_inner(xb, pb);
        *out_b = sum * k;
        bcsum += band_size;
    }
}

#[cfg(target_arch = "wasm32")]
#[inline]
fn compute_band_corr_inner(xb: &[f32], pb: &[f32]) -> f32 {
    use core::arch::wasm32::*;
    debug_assert_eq!(xb.len(), pb.len());
    let n = xb.len();
    let n4 = n & !3; // round down to multiple of 4
    let mut acc = f32x4_splat(0.0);
    let xp = xb.as_ptr();
    let pp = pb.as_ptr();
    let mut i = 0usize;
    while i < n4 {
        // SAFETY: xp/pp are aligned to f32 (4 bytes); v128_load uses unaligned semantics.
        // We bounds-check via i < n4 <= n == xb.len() == pb.len().
        unsafe {
            let xv = v128_load(xp.add(i) as *const v128);
            let pv = v128_load(pp.add(i) as *const v128);
            let prod = f32x4_mul(xv, pv);
            acc = f32x4_add(acc, prod);
        }
        i += 4;
    }
    // Horizontal reduce the 4 lanes.
    let mut sum = f32x4_extract_lane::<0>(acc)
        + f32x4_extract_lane::<1>(acc)
        + f32x4_extract_lane::<2>(acc)
        + f32x4_extract_lane::<3>(acc);
    // Tail: 0..3 leftover f32 (i.e. 0 or 1 trailing complex pair if band_size is odd).
    while i < n {
        sum += unsafe { *xp.add(i) * *pp.add(i) };
        i += 1;
    }
    sum
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn compute_band_corr_inner(xb: &[f32], pb: &[f32]) -> f32 {
    debug_assert_eq!(xb.len(), pb.len());
    let mut sum = 0.0f32;
    for (a, b) in xb.iter().zip(pb.iter()) {
        sum += a * b;
    }
    sum
}

// Element-wise IIR mean-norm: state[i] = x[i]*(1-α) + state[i]*α; x[i] = (x[i] - state[i])/40.
// Per-bin independent (no recurrence between bins) — straightforward SIMD.
#[cfg(target_arch = "wasm32")]
#[inline]
fn band_mean_norm_erb_inner(xs: &mut [f32], state: &mut [f32], alpha: f32) {
    use core::arch::wasm32::*;
    debug_assert_eq!(xs.len(), state.len());
    let n = xs.len();
    let n4 = n & !3;
    let one_minus_a = f32x4_splat(1.0 - alpha);
    let alpha_v = f32x4_splat(alpha);
    let inv40 = f32x4_splat(1.0 / 40.0);
    let xp = xs.as_mut_ptr();
    let sp = state.as_mut_ptr();
    let mut i = 0usize;
    while i < n4 {
        // SAFETY: i < n4 <= n == xs.len() == state.len(). v128_load takes 16 bytes
        // (4 f32). xp/sp are aligned to f32 (4 bytes); v128_load uses unaligned semantics.
        unsafe {
            let xv = v128_load(xp.add(i) as *const v128);
            let sv = v128_load(sp.add(i) as *const v128);
            let new_s = f32x4_add(f32x4_mul(xv, one_minus_a), f32x4_mul(sv, alpha_v));
            v128_store(sp.add(i) as *mut v128, new_s);
            let x_norm = f32x4_mul(f32x4_sub(xv, new_s), inv40);
            v128_store(xp.add(i) as *mut v128, x_norm);
        }
        i += 4;
    }
    while i < n {
        unsafe {
            let new_s = *xp.add(i) * (1.0 - alpha) + *sp.add(i) * alpha;
            *sp.add(i) = new_s;
            *xp.add(i) = (*xp.add(i) - new_s) / 40.0;
        }
        i += 1;
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn band_mean_norm_erb_inner(xs: &mut [f32], state: &mut [f32], alpha: f32) {
    debug_assert_eq!(xs.len(), state.len());
    for (x, s) in xs.iter_mut().zip(state.iter_mut()) {
        *s = *x * (1. - alpha) + *s * alpha;
        *x -= *s;
        *x /= 40.;
    }
}

// Multiply every f32 lane in `xs` by scalar `k`, in place.
#[cfg(target_arch = "wasm32")]
#[inline]
fn f32_scale_inplace(xs: &mut [f32], k: f32) {
    use core::arch::wasm32::*;
    let n = xs.len();
    let n4 = n & !3;
    let kv = f32x4_splat(k);
    let xp = xs.as_mut_ptr();
    let mut i = 0usize;
    while i < n4 {
        unsafe {
            let xv = v128_load(xp.add(i) as *const v128);
            v128_store(xp.add(i) as *mut v128, f32x4_mul(xv, kv));
        }
        i += 4;
    }
    while i < n {
        unsafe {
            *xp.add(i) *= k;
        }
        i += 1;
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn f32_scale_inplace(xs: &mut [f32], k: f32) {
    for x in xs.iter_mut() {
        *x *= k;
    }
}

// Element-wise multiply: xs[i] *= ws[i] for the whole slice, in place.
#[cfg(target_arch = "wasm32")]
#[inline]
fn f32_mul_inplace(xs: &mut [f32], ws: &[f32]) {
    use core::arch::wasm32::*;
    debug_assert_eq!(xs.len(), ws.len());
    let n = xs.len();
    let n4 = n & !3;
    let xp = xs.as_mut_ptr();
    let wp = ws.as_ptr();
    let mut i = 0usize;
    while i < n4 {
        unsafe {
            let xv = v128_load(xp.add(i) as *const v128);
            let wv = v128_load(wp.add(i) as *const v128);
            v128_store(xp.add(i) as *mut v128, f32x4_mul(xv, wv));
        }
        i += 4;
    }
    while i < n {
        unsafe {
            *xp.add(i) *= *wp.add(i);
        }
        i += 1;
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn f32_mul_inplace(xs: &mut [f32], ws: &[f32]) {
    debug_assert_eq!(xs.len(), ws.len());
    for (x, &w) in xs.iter_mut().zip(ws.iter()) {
        *x *= w;
    }
}

// Three-slice element-wise add: out[i] = a[i] + b[i].
#[cfg(target_arch = "wasm32")]
#[inline]
fn f32_add_to(a: &[f32], b: &[f32], out: &mut [f32]) {
    use core::arch::wasm32::*;
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let n4 = n & !3;
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;
    while i < n4 {
        unsafe {
            let av = v128_load(ap.add(i) as *const v128);
            let bv = v128_load(bp.add(i) as *const v128);
            v128_store(op.add(i) as *mut v128, f32x4_add(av, bv));
        }
        i += 4;
    }
    while i < n {
        unsafe {
            *op.add(i) = *ap.add(i) + *bp.add(i);
        }
        i += 1;
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn f32_add_to(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for ((&x, &y), o) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
        *o = x + y;
    }
}

// In-place element-wise add: xs[i] += ys[i].
#[cfg(target_arch = "wasm32")]
#[inline]
fn f32_add_inplace(xs: &mut [f32], ys: &[f32]) {
    use core::arch::wasm32::*;
    debug_assert_eq!(xs.len(), ys.len());
    let n = xs.len();
    let n4 = n & !3;
    let xp = xs.as_mut_ptr();
    let yp = ys.as_ptr();
    let mut i = 0usize;
    while i < n4 {
        unsafe {
            let xv = v128_load(xp.add(i) as *const v128);
            let yv = v128_load(yp.add(i) as *const v128);
            v128_store(xp.add(i) as *mut v128, f32x4_add(xv, yv));
        }
        i += 4;
    }
    while i < n {
        unsafe {
            *xp.add(i) += *yp.add(i);
        }
        i += 1;
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn f32_add_inplace(xs: &mut [f32], ys: &[f32]) {
    debug_assert_eq!(xs.len(), ys.len());
    for (x, &y) in xs.iter_mut().zip(ys.iter()) {
        *x += y;
    }
}

// IIR per-bin unit-norm on interleaved Complex32:
//   state[i] = sqrt(re[i]^2 + im[i]^2) * (1 - α) + state[i] * α;
//   xs[i] /= sqrt(state[i])      (Complex32 / f32 = each component / f32)
//
// SIMD path processes 4 Complex32 per iteration. The interleaved layout
// [re0,im0,re1,im1,re2,im2,re3,im3] is loaded as two v128s, de-interleaved
// via i32x4_shuffle into pure-real and pure-imag vectors so the norm can be
// computed lane-wise. The normalisation step then divides each Complex32
// component by sqrt(state[i]) by re-interleaving the divisor.
#[cfg(target_arch = "wasm32")]
#[inline]
fn band_unit_norm_inner(xs: &mut [Complex32], state: &mut [f32], alpha: f32) {
    use core::arch::wasm32::*;
    debug_assert_eq!(xs.len(), state.len());
    let n = xs.len();
    let n4 = n & !3;
    let one_minus_a = f32x4_splat(1.0 - alpha);
    let alpha_v = f32x4_splat(alpha);
    let xf = xs.as_mut_ptr() as *mut f32;
    let sp = state.as_mut_ptr();
    let mut i = 0usize;
    while i < n4 {
        // SAFETY: i < n4 <= n, and Complex32 is #[repr(C)] {re: f32, im: f32},
        // so xs as &mut [f32] of length 2N is valid. v128_load is unaligned.
        unsafe {
            let lo = v128_load(xf.add(i * 2) as *const v128);
            let hi = v128_load(xf.add(i * 2 + 4) as *const v128);
            // De-interleave: re_v = [re0, re1, re2, re3], im_v = [im0, im1, im2, im3]
            let re_v = i32x4_shuffle::<0, 2, 4, 6>(lo, hi);
            let im_v = i32x4_shuffle::<1, 3, 5, 7>(lo, hi);
            // norm = sqrt(re² + im²) (note: this is (re²+im²).sqrt(), not libm hypot)
            let norm_sq = f32x4_add(f32x4_mul(re_v, re_v), f32x4_mul(im_v, im_v));
            let norm_v = f32x4_sqrt(norm_sq);
            // state update
            let sv = v128_load(sp.add(i) as *const v128);
            let new_s = f32x4_add(f32x4_mul(norm_v, one_minus_a), f32x4_mul(sv, alpha_v));
            v128_store(sp.add(i) as *mut v128, new_s);
            // xs /= sqrt(state): build duplicated divisor per Complex32
            //   for lo: [sqrt_s0, sqrt_s0, sqrt_s1, sqrt_s1]
            //   for hi: [sqrt_s2, sqrt_s2, sqrt_s3, sqrt_s3]
            let sqrt_s = f32x4_sqrt(new_s);
            let div_lo = i32x4_shuffle::<0, 0, 1, 1>(sqrt_s, sqrt_s);
            let div_hi = i32x4_shuffle::<2, 2, 3, 3>(sqrt_s, sqrt_s);
            v128_store(xf.add(i * 2) as *mut v128, f32x4_div(lo, div_lo));
            v128_store(xf.add(i * 2 + 4) as *mut v128, f32x4_div(hi, div_hi));
        }
        i += 4;
    }
    // Tail: 0..3 trailing Complex32. Use the SAME (re²+im²).sqrt() as the SIMD
    // path (NOT Complex32::norm() which is libm hypot) so vectorised + tail
    // produce identical results across the full length.
    while i < n {
        unsafe {
            let xi_re = *xf.add(i * 2);
            let xi_im = *xf.add(i * 2 + 1);
            let norm = (xi_re * xi_re + xi_im * xi_im).sqrt();
            let new_s = norm * (1.0 - alpha) + *sp.add(i) * alpha;
            *sp.add(i) = new_s;
            let sqrt_s = new_s.sqrt();
            *xf.add(i * 2) = xi_re / sqrt_s;
            *xf.add(i * 2 + 1) = xi_im / sqrt_s;
        }
        i += 1;
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn band_unit_norm_inner(xs: &mut [Complex32], state: &mut [f32], alpha: f32) {
    for (x, s) in xs.iter_mut().zip(state.iter_mut()) {
        *s = x.norm() * (1. - alpha) + *s * alpha;
        *x /= s.sqrt();
    }
}

// Same IIR norm as band_unit_norm but writes to o_re / o_im split halves of
// the output (xs read-only). The output halves are CONTIGUOUS so no
// re-interleave step is needed for the divide — simpler than band_unit_norm.
#[cfg(target_arch = "wasm32")]
#[inline]
fn band_unit_norm_t_inner(
    xs: &[Complex32],
    state: &mut [f32],
    alpha: f32,
    o_re: &mut [f32],
    o_im: &mut [f32],
) {
    use core::arch::wasm32::*;
    debug_assert_eq!(xs.len(), state.len());
    debug_assert_eq!(xs.len(), o_re.len());
    debug_assert_eq!(xs.len(), o_im.len());
    let n = xs.len();
    let n4 = n & !3;
    let one_minus_a = f32x4_splat(1.0 - alpha);
    let alpha_v = f32x4_splat(alpha);
    let xf = xs.as_ptr() as *const f32;
    let sp = state.as_mut_ptr();
    let rp = o_re.as_mut_ptr();
    let ip = o_im.as_mut_ptr();
    let mut i = 0usize;
    while i < n4 {
        unsafe {
            let lo = v128_load(xf.add(i * 2) as *const v128);
            let hi = v128_load(xf.add(i * 2 + 4) as *const v128);
            let re_v = i32x4_shuffle::<0, 2, 4, 6>(lo, hi);
            let im_v = i32x4_shuffle::<1, 3, 5, 7>(lo, hi);
            let norm_sq = f32x4_add(f32x4_mul(re_v, re_v), f32x4_mul(im_v, im_v));
            let norm_v = f32x4_sqrt(norm_sq);
            let sv = v128_load(sp.add(i) as *const v128);
            let new_s = f32x4_add(f32x4_mul(norm_v, one_minus_a), f32x4_mul(sv, alpha_v));
            v128_store(sp.add(i) as *mut v128, new_s);
            let sqrt_s = f32x4_sqrt(new_s);
            // o_re / o_im are stored contiguously, divide directly
            let or_v = v128_load(rp.add(i) as *const v128);
            let oi_v = v128_load(ip.add(i) as *const v128);
            v128_store(rp.add(i) as *mut v128, f32x4_div(or_v, sqrt_s));
            v128_store(ip.add(i) as *mut v128, f32x4_div(oi_v, sqrt_s));
        }
        i += 4;
    }
    while i < n {
        unsafe {
            let xi_re = *xf.add(i * 2);
            let xi_im = *xf.add(i * 2 + 1);
            let norm = (xi_re * xi_re + xi_im * xi_im).sqrt();
            let new_s = norm * (1.0 - alpha) + *sp.add(i) * alpha;
            *sp.add(i) = new_s;
            let sqrt_s = new_s.sqrt();
            *rp.add(i) /= sqrt_s;
            *ip.add(i) /= sqrt_s;
        }
        i += 1;
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn band_unit_norm_t_inner(
    xs: &[Complex32],
    state: &mut [f32],
    alpha: f32,
    o_re: &mut [f32],
    o_im: &mut [f32],
) {
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
    // Reinterpret &mut [Complex32] as &mut [f32] of length 2*N. Complex32 is
    // #[repr(C)] { re: f32, im: f32 }: 8 bytes, alignment 4 — identical layout
    // to two contiguous f32. Multiplying each Complex32 by a real f32 scalar `b`
    // is equivalent to multiplying every f32 lane by `b`.
    let n = out.len();
    let outf: &mut [f32] =
        unsafe { core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f32, n * 2) };
    let mut bcsum = 0usize;
    for (&band_size, &b) in erb_fb.iter().zip(band_e.iter()) {
        let f_start = bcsum * 2;
        let f_len = band_size * 2;
        f32_scale_inplace(&mut outf[f_start..f_start + f_len], b);
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
    // out[i] = x_first[i] + synthesis_mem[i] (zip-3 stops at shortest;
    // x_first.len() == output.len() == frame_size; synthesis_mem may be longer).
    let n_out = output.len();
    debug_assert_eq!(x_first.len(), n_out);
    debug_assert!(state.synthesis_mem.len() >= n_out);
    f32_add_to(x_first, &state.synthesis_mem[..n_out], output);

    let split = state.synthesis_mem.len() - state.frame_size;
    if split > 0 {
        state.synthesis_mem.rotate_left(state.frame_size);
    }
    let (s_first, s_second) = state.synthesis_mem.split_at_mut(split);
    let (xs_first, xs_second) = x_second.split_at(split);
    // Overlap-add for next frame: s_first[i] += xs_first[i].
    let n_first = xs_first.len().min(s_first.len());
    f32_add_inplace(&mut s_first[..n_first], &xs_first[..n_first]);
    // Override left-shifted buffer: s_second[i] = xs_second[i] (memcpy-shaped).
    let n_second = xs_second.len().min(s_second.len());
    s_second[..n_second].copy_from_slice(&xs_second[..n_second]);
}

fn apply_window(xs: &[f32], window: &[f32]) -> Vec<f32> {
    let mut out = vec![0.; window.len()];
    for (&x, &w, o) in izip!(xs.iter(), window.iter(), out.iter_mut()) {
        *o = x * w;
    }
    out
}

fn apply_window_in_place(xs: &mut [f32], window: &[f32]) {
    debug_assert_eq!(xs.len(), window.len());
    f32_mul_inplace(xs, window);
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
