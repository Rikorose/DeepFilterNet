use std::mem::MaybeUninit;

use ndarray::{prelude::*, Slice};
use rubato::{FftFixedInOut, Resampler};
use thiserror::Error;

use crate::*;

type Result<T> = std::result::Result<T, TransformError>;

#[derive(Error, Debug)]
pub enum TransformError {
    #[error("DF error: {0}")]
    DfError(String),
    #[error("NaN detected")]
    NaN,
    #[error("Ndarray Shape Error")]
    NdarrayShapeError(#[from] ndarray::ShapeError),
    #[error("Resample Error")]
    ResampleError(#[from] rubato::ResampleError),
}

pub(crate) fn biquad_norm_inplace<'a, I>(xs: I, mem: &mut [f32; 2], b: &[f32; 2], a: &[f32; 2])
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

pub(crate) fn biquad_inplace<'a, I>(xs: I, mem: &mut [f32; 2], b: &[f32; 3], a: &[f32; 3])
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

pub(crate) fn mix_f(clean: ArrayView2<f32>, noise: ArrayView2<f32>, snr_db: f32) -> f32 {
    let e_clean = clean.iter().fold(0f32, |acc, x| acc + x.powi(2)) + 1e-10;
    let e_noise = noise.iter().fold(0f32, |acc, x| acc + x.powi(2)) + 1e-10;
    let snr = 10f32.powf(snr_db / 10.);
    (1f64 / (((e_noise / e_clean) * snr + 1e-10) as f64).sqrt()) as f32
}

#[inline]
pub(crate) fn rms_normalize(x: Array2<f32>) -> Array2<f32> {
    let rms = x.map(|x| x.powi(2)).mean_axis(Axis(1)).unwrap().map(|x| x.sqrt() + 1e-8);
    let ch = x.len_of(Axis(0));
    x / rms.to_shape([ch, 1]).unwrap()
}

pub fn fft(
    input: &mut Array2<f32>,
    fft_transform: &dyn RealToComplex<f32>,
    scratch: &mut [Complex32],
) -> Result<Array2<Complex32>> {
    let mut output = Array2::zeros((input.len_of(Axis(0)), fft_transform.len() / 2 + 1));
    fft_with_output(input, fft_transform, scratch, &mut output)?;
    Ok(output)
}

pub fn fft_with_output(
    input: &mut Array2<f32>,
    fft_transform: &dyn RealToComplex<f32>,
    scratch: &mut [Complex32],
    output: &mut Array2<Complex32>,
) -> Result<()> {
    debug_assert_eq!(fft_transform.len(), input.len_of(Axis(1)));
    for (mut input_ch, mut output_ch) in input.outer_iter_mut().zip(output.outer_iter_mut()) {
        let i = input_ch.as_slice_mut().unwrap();
        let o = output_ch.as_slice_mut().unwrap();
        fft_transform
            .process_with_scratch(i, o, scratch)
            .map_err(|e| TransformError::DfError(format!("Error in fft(): {:?}", e)))?;
    }
    Ok(())
}

pub fn ifft(
    input: &mut Array2<Complex32>,
    fft_transform: &dyn ComplexToReal<f32>,
    scratch: &mut [Complex32],
) -> Result<Array2<f32>> {
    let mut output = Array2::zeros((input.len_of(Axis(0)), (fft_transform.len() - 1) * 2));
    ifft_with_output(input, fft_transform, scratch, &mut output)?;
    Ok(output)
}

pub fn ifft_with_output(
    input: &mut Array2<Complex32>,
    fft_transform: &dyn ComplexToReal<f32>,
    scratch: &mut [Complex32],
    output: &mut Array2<f32>,
) -> Result<()> {
    for (mut input_ch, mut output_ch) in input.outer_iter_mut().zip(output.outer_iter_mut()) {
        let i = input_ch.as_slice_mut().unwrap();
        let o = output_ch.as_slice_mut().unwrap();
        fft_transform
            .process_with_scratch(i, o, scratch)
            .map_err(|e| TransformError::DfError(format!("Error in ifft(): {:?}", e)))?;
    }
    Ok(())
}

/// Short time Fourier transform.
///
/// Args:
///   - `input`: array of shape (C, T)
///   - `state`: DFState
///   - `reset`: Whether to reset STFT buffers
///
/// Returns:
///   - `spectrum`: complex array of shape (C, T', F)
pub fn stft(input: ArrayView2<f32>, state: &mut DFState, reset: bool) -> Array3<Complex32> {
    if reset {
        state.reset();
    }
    let ch = input.len_of(Axis(0));
    let ttd = input.len_of(Axis(1));
    let n_pad = state.window_size / state.frame_size - 1;
    let tfd = (ttd as f32 / state.frame_size as f32).ceil() as usize + n_pad;
    let mut output: Array3<Complex32> = Array3::zeros((ch, tfd, state.freq_size));
    for (input_ch, mut output_ch) in input.outer_iter().zip(output.outer_iter_mut()) {
        for (ichunk, mut ochunk) in input_ch
            .axis_chunks_iter(Axis(0), state.frame_size)
            .zip(output_ch.outer_iter_mut())
        {
            let ichunk = ichunk.as_slice().expect("stft ichunk has wrong shape");
            if ichunk.len() == state.frame_size {
                frame_analysis(
                    ichunk,
                    ochunk.as_slice_mut().expect("stft ochunk has wrong shape"),
                    state,
                )
            } else {
                let pad = vec![0.; state.frame_size - ichunk.len()];
                frame_analysis(
                    &[ichunk, pad.as_slice()].concat(),
                    ochunk.as_slice_mut().expect("stft ochunk has wrong shape"),
                    state,
                )
            };
        }
    }
    output.slice_axis_inplace(Axis(1), Slice::from(n_pad..));
    output
}

/// .Inverse short time Fourier transform.
///
/// # Args:
///   - `input`: Complex array of shape (C, T, F)
///   - `state`: DFState
///   - `reset`: Whether to reset ISTFT buffers before transfrorm.
///
/// # Returns
pub fn istft(mut input: ArrayViewMut3<Complex32>, state: &mut DFState, reset: bool) -> Array2<f32> {
    if reset {
        state.reset();
    }
    let ch = input.len_of(Axis(0));
    let tfd = input.len_of(Axis(1));
    let ttd = tfd * state.frame_size;
    let mut output: Array2<f32> = Array2::zeros((ch, ttd));
    for (mut input_ch, mut output_ch) in input.outer_iter_mut().zip(output.outer_iter_mut()) {
        for (mut ichunk, mut ochunk) in
            input_ch.outer_iter_mut().zip(output_ch.exact_chunks_mut(state.frame_size))
        {
            frame_synthesis(
                ichunk.as_slice_mut().unwrap(),
                ochunk.as_slice_mut().unwrap(),
                state,
            )
        }
    }
    output
}

pub fn erb_compr_with_output(
    input: &ArrayView3<f32>,
    output: &mut ArrayViewMut3<f32>,
    erb_fb: &[usize],
) -> Result<()> {
    for (in_ch, mut out_ch) in input.outer_iter().zip(output.outer_iter_mut()) {
        for (in_t, mut out_t) in in_ch.outer_iter().zip(out_ch.outer_iter_mut()) {
            let ichunk = in_t.as_slice().unwrap();
            let ochunk = out_t.as_slice_mut().unwrap();
            band_compr(ochunk, ichunk, erb_fb);
        }
    }
    Ok(())
}

pub fn erb_with_output(
    input: &ArrayView3<Complex32>,
    db: bool,
    output: &mut ArrayViewMut3<f32>,
    erb_fb: &[usize],
) -> Result<()> {
    for (in_ch, mut out_ch) in input.outer_iter().zip(output.outer_iter_mut()) {
        for (in_t, mut out_t) in in_ch.outer_iter().zip(out_ch.outer_iter_mut()) {
            let ichunk = in_t.as_slice().unwrap();
            let ochunk = out_t.as_slice_mut().unwrap();
            compute_band_corr(ochunk, ichunk, ichunk, erb_fb);
        }
    }
    if db {
        output.mapv_inplace(|v| (v + 1e-10).log10() * 10.);
    }
    Ok(())
}

pub fn erb(input: &ArrayView3<Complex32>, db: bool, erb_fb: &[usize]) -> Result<Array3<f32>> {
    // input shape: [C, T, F]
    let ch = input.len_of(Axis(0));
    let t = input.len_of(Axis(1));
    let mut output = Array3::<f32>::zeros((ch, t, erb_fb.len()));

    erb_with_output(input, db, &mut output.view_mut(), erb_fb)?;
    Ok(output)
}

pub fn apply_erb_gains(
    gains: &ArrayView3<f32>,
    input: &mut ArrayViewMut3<Complex32>,
    erb_fb: &[usize],
) -> Result<()> {
    // gains shape: [C, T, E]
    // input shape: [C, T, F]
    // erb_fb shape: [N_erb]
    for (g_ch, mut in_ch) in gains.outer_iter().zip(input.outer_iter_mut()) {
        for (g_t, mut in_t) in g_ch.outer_iter().zip(in_ch.outer_iter_mut()) {
            apply_interp_band_gain(
                in_t.as_slice_mut().unwrap(),
                g_t.as_slice().unwrap(),
                erb_fb,
            );
        }
    }
    Ok(())
}

pub fn erb_inv_with_output(
    gains: &ArrayView3<f32>,
    output: &mut ArrayViewMut3<f32>,
    erb_fb: &[usize],
) -> Result<()> {
    // gains shape: [C, T, E]
    // output shape: [C, T, F]
    // erb_fb shape: [N_erb]
    for (g_ch, mut o_ch) in gains.outer_iter().zip(output.outer_iter_mut()) {
        for (g_t, mut o_t) in g_ch.outer_iter().zip(o_ch.outer_iter_mut()) {
            interp_band_gain(o_t.as_slice_mut().unwrap(), g_t.as_slice().unwrap(), erb_fb);
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
        let state_ch0 = Array1::<f32>::linspace(MEAN_NORM_INIT[0], MEAN_NORM_INIT[1], b)
            .into_shape([1, b])
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

/// Low pass by resampling the data to `f_cut_off*2`.
pub(crate) fn low_pass_resample(
    x: ArrayView2<f32>,
    f_cut_off: usize,
    sr: usize,
) -> Result<Array2<f32>> {
    let orig_len = x.len_of(Axis(1));
    let x = resample(x, sr, f_cut_off * 2, None)?;
    let mut x = resample(x.view(), f_cut_off * 2, sr, None)?;
    x.slice_axis_inplace(Axis(1), Slice::from(0..orig_len));
    Ok(x)
}
/// Resample using a synchronous resample from rubato
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

/// Bandwidth extension via spectral translation.
/// That is, copy spectrum from lower frequencies into higher frequencies.
///
/// Args:
///   - `x`: Spectrogram of shape (C, T, F)
///   - `freq`: Cut of frequency. Frequencies above will get extended based on lower Frequencies.
///   - `sr`: Original time-domain sampling rate.
///   - `n_bins_overlap`: Instead of starting at bin correspinging to `freq` start `n_bins_overlap` lower.
pub(crate) fn ext_bandwidth_spectral(
    x: &mut Array3<Complex32>,
    freq: usize,
    sr: usize,
    n_bins_overlap: Option<usize>,
) {
    let full_n_bins = x.len_of(Axis(2)); // Number of bins of non-downampled signal
    let n_ov = n_bins_overlap.unwrap_or(0); // Overlap at the edge of the downsampled spectrum
    let down_n_bins = full_n_bins * freq * 2 / sr - n_ov; // Number of bins containing energy in downsampled signal
    let min_bin = full_n_bins * 4000 * 2 / sr + 1; // Only start from 4kHz
    let max_copy_bins = down_n_bins - min_bin; // Number of bins to copy per iteration
    let missing_bins = full_n_bins - down_n_bins;
    let n_copies = (missing_bins as f32 / max_copy_bins as f32).ceil() as usize;
    let mut start_bin_tgt = down_n_bins;
    let start_bin_src = min_bin.max(down_n_bins.saturating_sub(missing_bins));
    debug_assert!(start_bin_tgt > start_bin_src);
    for _ in 0..n_copies {
        let cur_n_copy = max_copy_bins.min(full_n_bins - start_bin_tgt);
        let (src, target) = x.multi_slice_mut((
            s![.., .., start_bin_src..(start_bin_src + cur_n_copy)],
            s![.., .., start_bin_tgt..(start_bin_tgt + cur_n_copy)],
        ));
        src.assign_to(target);
        start_bin_tgt += cur_n_copy;
    }
}

/// Estimate bandwidth by finding the highest frequency bin containing a sufficient amount of
/// energy.
pub(crate) fn estimate_bandwidth(
    input: ArrayView3<Complex32>,
    db_cut_off: f32,
    n_avg: usize,
) -> usize {
    // input shape [C, T, F]
    let f_db = input
        .mean_axis(Axis(1))
        .unwrap()
        .mean_axis(Axis(0))
        .unwrap()
        .map(|x| (x.norm() + 1e-16).log10() * 10.);
    let n_freqs = f_db.len();
    let f_db_diff =
        (f_db.slice(s![..n_freqs - n_avg]).to_owned() - f_db.slice(s![n_avg..])) / n_avg as f32;
    let i = argmax(&f_db_diff).unwrap_or(n_freqs);
    if f_db_diff[i] < db_cut_off || i < n_freqs / 4 {
        return n_freqs;
    }
    // Some corrections if we are at a boundary of n_freqs/2, n_freqs/3, etc.
    if i == n_freqs - 1 {
        return n_freqs;
    } else if i == n_freqs / 2 || i == n_freqs / 2 + 2 {
        return n_freqs / 2 + 1;
    } else if i == n_freqs / 3 || i == n_freqs / 3 + 2 {
        return n_freqs / 3 + 1;
    } else if i == n_freqs / 4 || i == n_freqs / 4 + 2 {
        return n_freqs / 4 + 1;
    }
    i
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

pub(crate) fn find_max<'a, I>(vals: I) -> Result<f32>
where
    I: IntoIterator<Item = &'a f32>,
{
    vals.into_iter().try_fold(f32::MIN, |acc, v| {
        let nonnan: NonNan = match NonNan::new(*v) {
            None => return Err(TransformError::NaN),
            Some(x) => x,
        };
        Ok(nonnan.get().max(acc))
    })
}

pub(crate) fn find_max_abs<'a, I>(vals: I) -> Result<f32>
where
    I: IntoIterator<Item = &'a f32>,
{
    vals.into_iter().try_fold(0., |acc, v| {
        let nonnan: NonNan = match NonNan::new(v.abs()) {
            None => return Err(TransformError::NaN),
            Some(x) => x,
        };
        Ok(nonnan.get().max(acc))
    })
}

pub(crate) fn find_min<'a, I>(vals: I) -> Result<f32>
where
    I: IntoIterator<Item = &'a f32>,
{
    vals.into_iter().try_fold(f32::MAX, |acc, v| {
        let nonnan: NonNan = match NonNan::new(*v) {
            None => return Err(TransformError::NaN),
            Some(x) => x,
        };
        Ok(nonnan.get().min(acc))
    })
}

pub(crate) fn find_min_abs<'a, I>(vals: I) -> Result<f32>
where
    I: IntoIterator<Item = &'a f32>,
{
    vals.into_iter().try_fold(0., |acc, v| {
        let nonnan: NonNan = match NonNan::new(v.abs()) {
            None => return Err(TransformError::NaN),
            Some(x) => x,
        };
        Ok(nonnan.get().min(acc))
    })
}

pub(crate) fn median(x: &mut [f32]) -> f32 {
    x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = x.len() / 2;
    x[mid]
}

pub(crate) fn argmax<'a, I>(vals: I) -> Result<usize>
where
    I: IntoIterator<Item = &'a f32>,
{
    let mut index = 0;
    let mut high = std::f32::MIN;
    vals.into_iter().enumerate().for_each(|(i, v)| {
        if v > &high {
            high = *v;
            index = i;
        }
    });
    Ok(index)
}

pub(crate) fn argmax_abs<'a, I>(vals: I) -> Result<usize>
where
    I: IntoIterator<Item = &'a f32>,
{
    let mut index = 0;
    let mut high = std::f32::MIN;
    vals.into_iter().enumerate().for_each(|(i, v)| {
        if v > &high {
            high = v.abs();
            index = i;
        }
    });
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{augmentations::low_pass, wav_utils::*};

    /// Setup function that is only run once, even if called multiple times.
    fn setup() -> (Array2<f32>, usize) {
        create_out_dir().expect("Could not create output directory");
        let reader = ReadWav::new("../assets/clean_freesound_33711.wav").unwrap();
        let sr = reader.sr;
        let sample = reader.samples_arr2().unwrap();
        (sample, sr)
    }

    fn create_out_dir() -> std::io::Result<()> {
        match std::fs::create_dir("../out") {
            Err(ref e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
            r => r,
        }
    }

    #[test]
    pub fn test_stft_istft_delay() -> Result<()> {
        let (sample, sr) = setup();
        let ch = sample.len_of(Axis(0)) as u16;
        let fft_size = sr / 50;
        let hop_size = fft_size / 2;
        let mut state = DFState::new(sr, fft_size, hop_size, 1, 1);
        let mut x = stft(sample.view(), &mut state, true);
        let out = istft(x.view_mut(), &mut state, true);
        for (ich, och) in sample.outer_iter().zip(out.outer_iter()) {
            let xx: f32 = ich.iter().map(|&s| s * s).sum();
            let yy: f32 = ich.iter().map(|&s| s * s).sum();
            let xy: f32 = ich.iter().zip(och).map(|(&a, &b)| a * b).sum();
            let corr = xy / (xx.sqrt() * yy.sqrt());
            dbg!(corr);
            assert!((corr - 1.).abs() < 1e-6)
        }
        write_wav_iter("../out/original.wav", sample.iter(), sr as u32, ch).unwrap();
        write_wav_iter("../out/stft_istft.wav", out.iter(), sr as u32, ch).unwrap();
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
    fn test_estimate_bandwidth() -> Result<()> {
        let (sample, mut sr) = setup();
        let sample = resample(sample.view(), sr, sr / 2, None).unwrap();
        sr /= 2;
        write_wav_arr2("../out/original.wav", sample.view(), sr as u32).unwrap();

        let fft_size = sr / 50;
        let hop_size = fft_size / 2;
        let mut state = DFState::new(sr, fft_size, hop_size, 1, 1);
        let x = stft(sample.view(), &mut state, true);
        assert_eq!(estimate_bandwidth(x.view(), 8., 2), fft_size / 2 + 1);

        let sample_f2 = low_pass_resample(sample.view(), sr / 4, sr).unwrap();
        write_wav_arr2("../out/resampled_f2.wav", sample_f2.view(), sr as u32).unwrap();
        let x_f2 = stft(sample_f2.view(), &mut state, true);
        assert_eq!(estimate_bandwidth(x_f2.view(), 8., 2), fft_size / 4 + 1);

        let sample_f3 = low_pass_resample(sample.view(), sr / 6, sr).unwrap();
        write_wav_arr2("../out/resampled_f3.wav", sample_f3.view(), sr as u32).unwrap();
        let x_f3 = stft(sample_f3.view(), &mut state, true);
        assert_eq!(estimate_bandwidth(x_f3.view(), 8., 2), fft_size / 6 + 1);

        let sample_f4 = low_pass_resample(sample.view(), sr / 8, sr).unwrap();
        write_wav_arr2("../out/resampled_f4.wav", sample_f4.view(), sr as u32).unwrap();
        let x_f4 = stft(sample_f4.view(), &mut state, true);
        assert_eq!(estimate_bandwidth(x_f4.view(), 8., 2), fft_size / 8 + 1);
        Ok(())
    }

    #[test]
    fn test_ext_bandwidth_spectral() {
        let (sample, sr) = setup();
        let fft_size = sr / 50;
        let hop_size = fft_size / 2;
        let mut state = DFState::new(sr, fft_size, hop_size, 1, 1);
        let mut x = stft(sample.view(), &mut state, true);

        let f_cut_off = 20000;
        let sample_f2 = low_pass_resample(sample.view(), f_cut_off, sr).unwrap();
        write_wav_arr2("../out/sample_f2.wav", sample_f2.view(), sr as u32).unwrap();
        let mut x2 = stft(sample_f2.view(), &mut state, true);
        ext_bandwidth_spectral(&mut x2, f_cut_off, sr, Some(4));

        let sample = istft(x.view_mut(), &mut state, true);
        write_wav_arr2("../out/original.wav", sample.view(), sr as u32).unwrap();
        let sample_f2 = istft(x2.view_mut(), &mut state, true);
        write_wav_arr2("../out/sample_f2_ext.wav", sample_f2.view(), sr as u32).unwrap();
    }

    #[test]
    fn test_find_max_abs() -> Result<()> {
        let mut x = vec![vec![0f32; 10]; 1];
        x[0][2] = 3f32;
        x[0][5] = -10f32;
        let max = find_max_abs(x.iter().flatten())?;
        assert_eq!(max, 10.);
        Ok(())
    }
}
