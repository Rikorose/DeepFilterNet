use std::boxed::Box;
use std::ffi::{c_char, c_float, c_uint, CStr};
use std::path::PathBuf;

use ndarray::prelude::*;
use wasm_bindgen::prelude::*;

use crate::tract::*;

pub struct DFState(crate::tract::DfTract);

impl DFState {
    fn new(model_path: &str, channels: usize, atten_lim: f32) -> Self {
        let r_params = RuntimeParams::default_with_ch(channels).with_atten_lim(atten_lim);
        let df_params =
            DfParams::new(PathBuf::from(model_path)).expect("Could not load model from path");
        let m =
            DfTract::new(df_params, &r_params).expect("Could not initialize DeepFilter runtime.");
        DFState(m)
    }
    fn boxed(self) -> Box<DFState> {
        Box::new(self)
    }
}

/// Create a DeepFilterNet Model
///
/// Args:
///     - path: File path to a DeepFilterNet tar.gz onnx model
///     - atten_lim: Attenuation limit in dB.
///
/// Returns:
///     - DF state doing the full processing: stft, DNN noise reduction, istft.
#[wasm_bindgen]
#[no_mangle]
pub unsafe extern "C" fn df_create(
    path: *const c_char,
    // channels: usize,
    atten_lim: f32,
) -> *mut DFState {
    let c_str = CStr::from_ptr(path);
    let path = c_str.to_str().unwrap();
    let df = DFState::new(path, 1, atten_lim);
    Box::into_raw(df.boxed())
}

/// Get DeepFilterNet frame size in samples.
#[wasm_bindgen]
#[no_mangle]
pub unsafe extern "C" fn df_get_frame_length(st: *mut DFState) -> usize {
    let state = st.as_mut().expect("Invalid pointer");
    state.0.hop_size
}

/// Set DeepFilterNet attenuation limit.
///
/// Args:
///     - lim_db: New attenuation limit in dB.
#[wasm_bindgen]
#[no_mangle]
pub unsafe extern "C" fn df_set_atten_lim(st: *mut DFState, lim_db: f32) {
    let state = st.as_mut().expect("Invalid pointer");
    state.0.set_atten_lim(lim_db)
}

/// Set DeepFilterNet post filter beta. A beta of 0 disables the post filter.
///
/// Args:
///     - beta: Post filter attenuation. Suitable range between 0.05 and 0;
#[wasm_bindgen]
#[no_mangle]
pub unsafe extern "C" fn df_set_post_filter_beta(st: *mut DFState, beta: f32) {
    let state = st.as_mut().expect("Invalid pointer");
    state.0.set_pf_beta(beta)
}

/// Processes a chunk of samples.
///
/// Args:
///     - df_state: Created via df_create()
///     - input: Input buffer of length df_get_frame_length()
///     - output: Output buffer of length df_get_frame_length()
///
/// Returns:
///     - Local SNR of the current frame.
#[wasm_bindgen]
#[no_mangle]
pub unsafe extern "C" fn df_process_frame(
    st: *mut DFState,
    input: *mut c_float,
    output: *mut c_float,
) -> c_float {
    let state = st.as_mut().expect("Invalid pointer");
    let input = ArrayView2::from_shape_ptr((1, state.0.hop_size), input);
    let output = ArrayViewMut2::from_shape_ptr((1, state.0.hop_size), output);

    state.0.process(input, output).expect("Failed to process DF frame")
}