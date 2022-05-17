use ndarray::prelude::*;
use thiserror::Error;

use crate::*;

type Result<T> = std::result::Result<T, TransformError>;

#[derive(Error, Debug)]
pub enum TransformError {
    #[error("DF error: {0}")]
    DfError(String),
    #[error("Ndarray Shape Error")]
    NdarrayShapeError(#[from] ndarray::ShapeError),
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
            let ichunk = ichunk.as_slice().unwrap();
            if ichunk.len() == state.frame_size {
                frame_analysis(ichunk, ochunk.as_slice_mut().unwrap(), state)
            } else {
                let pad = vec![0.; state.frame_size - ichunk.len()];
                frame_analysis(
                    &[ichunk, pad.as_slice()].concat(),
                    ochunk.as_slice_mut().unwrap(),
                    state,
                )
            };
        }
    }
    output.slice_axis_inplace(Axis(1), ndarray::Slice::from(n_pad..));
    output
}

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

/// Estimate bandwidth by finding the highest frequency bin containing a sufficient amount of
/// energy.
pub(crate) fn estimate_bandwidth(input: ArrayView3<Complex32>) -> usize {
    // input shape [C, T, F]
    let f_db = input
        .mean_axis(Axis(1))
        .unwrap()
        .mean_axis(Axis(0))
        .unwrap()
        .map(|x| (x.norm() + 1e-16).log10() * 10.);
    let n_freqs = f_db.len();
    let f_db_slc = f_db.slice(s![..n_freqs / 2]); // Compute median over first half of freqs
    let median = crate::util::median(f_db_slc.to_owned().as_slice_mut().unwrap());
    for (i, &f) in f_db.iter().enumerate().skip(1) {
        // assume 18 dB drop
        if f < median - 18. {
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
            return i;
        }
    }
    n_freqs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::augmentations::*;
    use crate::wav_utils::*;

    /// Setup function that is only run once, even if called multiple times.
    fn setup() {
        create_out_dir().expect("Could not create output directory");
    }

    fn create_out_dir() -> std::io::Result<()> {
        match std::fs::create_dir("../out") {
            Err(ref e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
            r => r,
        }
    }

    #[test]
    fn test_estimate_bandwidth() -> Result<()> {
        setup();
        let reader = ReadWav::new("../assets/clean_freesound_33711.wav").unwrap();
        let mut sr = reader.sr;
        let sample = reader.samples_arr2().unwrap();
        let sample = resample(sample.view(), sr, sr / 2, None).unwrap();
        sr /= 2;
        write_wav_arr2("../out/original.wav", sample.view(), sr as u32).unwrap();

        let fft_size = sr / 50;
        let hop_size = fft_size / 2;
        let mut state = DFState::new(sr, fft_size, hop_size, 1, 1);
        let x = stft(sample.view(), &mut state, true);
        assert_eq!(estimate_bandwidth(x.view()), fft_size / 2 + 1);

        let sample_f2 = low_pass_resample(sample.view(), sr / 4, sr).unwrap();
        write_wav_arr2("../out/resampled_f2.wav", sample_f2.view(), sr as u32).unwrap();
        let x_f2 = stft(sample_f2.view(), &mut state, true);
        assert_eq!(estimate_bandwidth(x_f2.view()), fft_size / 4 + 1);

        let sample_f3 = low_pass_resample(sample.view(), sr / 6, sr).unwrap();
        write_wav_arr2("../out/resampled_f3.wav", sample_f3.view(), sr as u32).unwrap();
        let x_f3 = stft(sample_f3.view(), &mut state, true);
        assert_eq!(estimate_bandwidth(x_f3.view()), fft_size / 6 + 1);

        let sample_f4 = low_pass_resample(sample.view(), sr / 8, sr).unwrap();
        write_wav_arr2("../out/resampled_f4.wav", sample_f4.view(), sr as u32).unwrap();
        let x_f4 = stft(sample_f4.view(), &mut state, true);
        assert_eq!(estimate_bandwidth(x_f4.view()), fft_size / 8 + 1);
        Ok(())
    }
}
