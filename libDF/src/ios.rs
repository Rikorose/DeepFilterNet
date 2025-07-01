use std::boxed::Box;
use std::ffi::{c_char, c_float, c_uint, CStr, CString, c_short};
use std::path::PathBuf;
use std::str::FromStr;
use std::slice;

use crossbeam_channel::TryRecvError;
use ndarray::prelude::*;

use crate::tract::*;

#[repr(C)]
pub struct DFState(crate::tract::DfTract);

impl DFState {

    fn new(model_bytes: &[u8], channels: usize, atten_lim: f32) -> Self {
        let r_params = RuntimeParams::default_with_ch(channels).with_atten_lim(atten_lim);
        let df_params = DfParams::from_bytes(model_bytes).expect("Could not load model from path");
        let m = DfTract::new(df_params, &r_params).expect("Could not initialize DeepFilter runtime.");
        DFState(m)
    }

    fn boxed(self) -> Box<DFState> {
        Box::new(self)
    }
}

// C-compatible creation function
#[no_mangle]
pub extern "C" fn df_create(
    model_bytes: *const u8,
    model_size: usize,
    channels: usize,
    atten_lim: f32
) -> *mut DFState {
    // Safety: Convert raw pointer to Rust slice
    let bytes = unsafe {
        if model_bytes.is_null() || model_size == 0 {
            return std::ptr::null_mut();
        }
        slice::from_raw_parts(model_bytes, model_size)
    };

    let df = DFState::new(bytes, 1, atten_lim);
    Box::into_raw(df.boxed())
}


/// Get DeepFilterNet frame size in samples.
#[no_mangle]
pub unsafe extern "C" fn df_get_frame_length(st: *mut DFState) -> usize {
    let state = st.as_mut().expect("Invalid pointer");
    state.0.hop_size
}


/// Set DeepFilterNet attenuation limit.
///
/// Args:
///     - lim_db: New attenuation limit in dB.
#[no_mangle]
pub unsafe extern "C" fn df_set_atten_lim(st: *mut DFState, lim_db: f32) {
    let state = st.as_mut().expect("Invalid pointer");
    state.0.set_atten_lim(lim_db)
}

/// Set DeepFilterNet post filter beta. A beta of 0 disables the post filter.
///
/// Args:
///     - beta: Post filter attenuation. Suitable range between 0.05 and 0;
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
///
/// Returns:
///     - Local SNR of the current frame.
#[no_mangle]
pub unsafe extern "C" fn df_process_frame(
    st: *mut DFState,
    input: *mut i16,
    frame_size: usize,
) -> c_float {
    let state = st.as_mut().expect("Invalid pointer");
    let hop_size = state.0.hop_size; // 512

    if frame_size != hop_size {
        return -1.0;
    }

    let mut result: c_float = 0.0;

    // Convert first half of input from i16 to f32
    let mut input_float = Vec::with_capacity(hop_size);

    // Convert int16 values to float (scaling to the -1.0 to 1.0 range)
    for i in 0..hop_size {
        let int_value = *input.add(i);
         // Scale int16 (-32768 to 32767) to float (-1.0 to 1.0)
         let float_val = int_value as f32 / 32768.0;
         input_float.push(float_val);
    }
    // Create input view
   let input_view = match ArrayView2::from_shape((1, hop_size), input_float.as_slice()) {
            Ok(input) => input,
            Err(error) => {
                return -4.0; // Error code for shape mismatch
            }
        };
    // Create output buffer for first half
    let mut output_float = Array2::zeros((1, hop_size));
    let output_view = output_float.view_mut();
    // Process first half
    result = state.0.process(input_view, output_view).expect("Failed to process first half of DF frame");
    let output_slice = output_float.as_slice().unwrap();
    // Write processed data back to first half of input buffer
    for i in 0..hop_size {
        let float_val = output_slice[i];
        let clamped = float_val.max(-1.0).min(1.0);
        let short_val = (clamped * 32767.0) as i16;
        *input.add(i) = short_val;
    }
    result
}

/// Free a DeepFilterNet Model
#[no_mangle]
pub unsafe extern "C" fn df_free(model: *mut DFState) {
    let _ = Box::from_raw(model);
}
