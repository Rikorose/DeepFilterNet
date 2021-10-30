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

pub fn fft(input: ArrayView2<f32>, state: &mut DFState) -> Result<Array2<Complex32>> {
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

pub fn stft(input: ArrayView2<f32>, state: &mut DFState, reset: bool) -> Array3<Complex32> {
    if reset {
        state.reset();
    }
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
            )
        }
    }
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
                &mut ichunk.as_slice_mut().unwrap(),
                &mut ochunk.as_slice_mut().unwrap(),
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
