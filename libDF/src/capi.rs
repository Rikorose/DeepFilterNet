use std::boxed::Box;
use std::ffi::CStr;
use std::os::raw::{c_char, c_float};
use std::path::PathBuf;

use ndarray::prelude::*;

use crate::tract::*;

pub struct DFState(crate::tract::DfTract);

impl DFState {
    fn new(model_path: &str, channels: usize, atten_lim: f32) -> Self {
        let r_params =
            RuntimeParams::new(channels, false, atten_lim, -10., 30., 20., ReduceMask::MEAN);
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
#[no_mangle]
pub unsafe extern "C" fn df_get_frame_length(st: *mut DFState) -> usize {
    let state = st.as_mut().expect("Invalid pointer");
    state.0.hop_size
}

/// Get DeepFilterNet frame size in samples.
///
/// Args:
///     - lim_db: New attenuation limit in dB.
#[no_mangle]
pub unsafe extern "C" fn df_set_atten_lim(st: *mut DFState, lim_db: f32) {
    let state = st.as_mut().expect("Invalid pointer");
    state.0.set_atten_lim(lim_db).expect("Failed to set attenuation limit.")
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

/// Free a DeepFilterNet Model
#[no_mangle]
pub unsafe extern "C" fn df_free(model: *mut DFState) {
    let _ = Box::from_raw(model);
}
