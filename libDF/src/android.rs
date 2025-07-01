use std::boxed::Box;
use ndarray::Array2;
use crate::tract::*;
use jni::{
    sys::{jlong, jint, jfloat, jclass, jboolean, JNI_TRUE, JNI_FALSE},
    JNIEnv,
    objects::{JByteArray, JClass, JObject, JByteBuffer},
};

#[repr(C)]
pub struct NativeDeepFilterNet(crate::tract::DfTract);

impl NativeDeepFilterNet {
    /// Creates a new NativeDeepFilterNet instance from model bytes
    ///
    /// # Arguments
    /// * `model_bytes` - Bytes containing the model data
    /// * `channels` - Number of audio channels
    /// * `atten_lim` - Attenuation limit in dB
    ///
    /// # Returns
    /// A new NativeDeepFilterNet instance or panics if initialization fails
    #[no_mangle]
    pub fn new(model_bytes: &[u8], channels: usize, atten_lim: f32) -> Result<Self, String> {
        let r_params = RuntimeParams::default_with_ch(channels).with_atten_lim(atten_lim);

        let df_params = match DfParams::from_bytes(model_bytes) {
            Ok(params) => params,
            Err(e) => return Err(format!("Could not load model: {}", e)),
        };

        match DfTract::new(df_params, &r_params) {
            Ok(m) => Ok(NativeDeepFilterNet(m)),
            Err(e) => Err(format!("Could not initialize DeepFilter runtime: {}", e)),
        }
    }

    /// Returns the hop size of the current model
    pub fn hop_size(&self) -> usize {
        // Hop size: 480 samples for 10ms at 48kHz.
        self.0.hop_size
    }

    /// Sets the attenuation limit in dB
    pub fn set_atten_lim(&mut self, lim_db: f32) {
        self.0.set_atten_lim(lim_db);
    }

    /// Sets the post-filter beta value
    pub fn set_pf_beta(&mut self, beta: f32) {
        self.0.set_pf_beta(beta);
    }

    /// Returns the hop size of the current model
    pub fn frame_bytes(&self) -> usize {
        // For 16-bit PCM, this equates to 960 bytes (480 samples * 2 bytes/sample).
        self.0.hop_size * 2
    }
}

/// Helper module for JNI operations
mod jni_helpers {
    use super::*;
    use std::ptr::NonNull;

    /// Safely get a mutable reference from a JNI pointer
    pub fn get_native_ptr<T>(ptr: jlong) -> Result<&'static mut T, String> {
        if ptr == 0 {
            return Err("Null pointer provided".to_string());
        }

        // Convert the jlong to a pointer and then to a reference
        unsafe {
            NonNull::new(ptr as *mut T)
                .map(|mut p| p.as_mut())
                .ok_or_else(|| "Invalid pointer".to_string())
        }
    }

    pub fn log_error(message: &str) {
        eprintln!("[NativeDeepFilterNet Error] {}", message);
    }
}


#[no_mangle]
pub extern "C" fn Java_com_rikorose_deepfilternet_NativeDeepFilterNet_newNative(
    env: JNIEnv,
    _: JClass,
    model_bytes: JByteArray,
    atten_lim: jfloat,
) -> jlong {
    use jni_helpers::*;

    // Convert Java byte array to Rust vector
    let model_bytes: Vec<u8> = match env.convert_byte_array(&model_bytes) {
        Ok(bytes) => bytes,
        Err(e) => {
            log_error(&format!("Error converting jbyteArray: {:?}", e));
            return 0;
        }
    };

    // Create the NativeDeepFilterNet instance
    let df = match NativeDeepFilterNet::new(&model_bytes, 1, atten_lim) {
        Ok(df) => df,
        Err(e) => {
            log_error(&format!("Failed to create NativeDeepFilterNet: {}", e));
            return 0;
        }
    };

    // Box and convert to raw pointer
    Box::into_raw(Box::new(df)) as jlong
}

#[no_mangle]
pub extern "C" fn Java_com_rikorose_deepfilternet_NativeDeepFilterNet_freeNative(
    _env: JNIEnv,
    _: jclass,
    ptr: jlong,
) {
    if ptr == 0 {
        return;
    }

    unsafe {
        // Convert back to Box and drop to properly free the memory
        let _ = Box::from_raw(ptr as *mut NativeDeepFilterNet);
    }
}

#[no_mangle]
pub extern "C" fn Java_com_rikorose_deepfilternet_NativeDeepFilterNet_getFrameLengthNative(
    _env: JNIEnv,
    _: jclass,
    ptr: jlong,
) -> jlong {
    use jni_helpers::*;

    if ptr == 0 {
        log_error("Null NativeDeepFilterNet pointer");
        return 0;
    }

    match get_native_ptr::<NativeDeepFilterNet>(ptr) {
        Ok(state) => state.frame_bytes() as jlong,
        Err(err) => {
            log_error(&err);
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn Java_com_rikorose_deepfilternet_NativeDeepFilterNet_setAttenLimNative(
    _env: JNIEnv,
    _: jclass,
    ptr: jlong,
    lim_db: jfloat,
) -> jboolean {
    use jni_helpers::*;

    // Get NativeDeepFilterNet instance
    let state = match get_native_ptr::<NativeDeepFilterNet>(ptr) {
        Ok(state) => state,
        Err(err) => {
            log_error(&err);
            return JNI_FALSE;
        }
    };

    // Set attenuation limit
    state.set_atten_lim(lim_db);

    JNI_TRUE
}

#[no_mangle]
pub extern "C" fn Java_com_rikorose_deepfilternet_NativeDeepFilterNet_setPostFilterBetaNative(
    _env: JNIEnv,
    _: jclass,
    ptr: jlong,
    beta: jfloat,
) -> jboolean {
    use jni_helpers::*;

    // Get NativeDeepFilterNet instance
    let state = match get_native_ptr::<NativeDeepFilterNet>(ptr) {
        Ok(state) => state,
        Err(err) => {
            log_error(&err);
            return JNI_FALSE;
        }
    };

    // Set post-filter beta
    state.set_pf_beta(beta);

    JNI_TRUE
}

#[no_mangle]
pub extern "C" fn Java_com_rikorose_deepfilternet_NativeDeepFilterNet_processFrameNative<'a>(
    mut env: JNIEnv<'a>,
    _: jclass,
    ptr: jlong,
    byte_buffer: JObject<'a>
) -> jfloat {
    use jni_helpers::*;

    // Get the NativeDeepFilterNet instance
    let state = match get_native_ptr::<NativeDeepFilterNet>(ptr) {
        Ok(state) => state,
        Err(err) => {
            log_error(&err);
            return -10 as jfloat;
        }
    };

    // Verify buffer is direct
    let is_direct = env.call_method(&byte_buffer, "isDirect", "()Z", &[])
        .expect("Failed to call isDirect")
        .z()
        .expect("Failed to convert to boolean");

    if !is_direct {
        log_error("ByteBuffer must be direct");
        return -20 as jfloat; // Error code for non-direct buffer
    }

    let byte_buffer = unsafe { JByteBuffer::from_raw(byte_buffer.into_raw()) };

    let buffer_capacity = match env.get_direct_buffer_capacity(&byte_buffer) {
        Ok(capacity) => capacity,
        Err(e) => {
            log_error(&format!("Failed to get direct buffer capacity: {:?}", e));
            return -30 as jfloat;
        }
    };

    if buffer_capacity != state.frame_bytes() {
        log_error(&format!(
            "Invalid size for the ByteBuffer. Expected: {}, Got: {}",
            state.frame_bytes(),
            buffer_capacity
        ));
        return -40 as jfloat;
    }

    // Get buffer pointer and capacity
    let buffer_ptr = match env.get_direct_buffer_address(&byte_buffer) {
        Ok(ptr) => ptr as *mut i16,
        Err(e) => {
            log_error(&format!("Failed to get direct buffer address: {:?}", e));
            return -50 as jfloat;
        }
    };

    // Calculate hop size
    let hop_size = std::cmp::min(buffer_capacity, state.hop_size());

    // Create a slice view of the input buffer
    let input_slice = unsafe { std::slice::from_raw_parts(buffer_ptr, hop_size) };

    // Pre-allocate output buffer with zeros
    let mut output_float = Array2::zeros((1, hop_size));

    // Process in one step - convert i16 to float during array creation
    let array = Array2::from_shape_fn((1, hop_size), |(_, i)| {
        input_slice[i] as f32 / 32768.0
    });
    let input_float = array.view();

    // Process the data
    let lsnr = match state.0.process(input_float, output_float.view_mut()) {
        Ok(lsnr) => lsnr,
        Err(e) => {
            log_error(&format!("Failed to process audio frame: {:?}", e));
            return -60 as jfloat;
        }
    };

    // Convert processed floats back to shorts in-place
    let output_float_slice = match output_float.as_slice() {
        Some(slice) => slice,
        None => {
            log_error("Failed to get output as slice");
            return -70 as jfloat;
        }
    };

    for (i, &float_val) in output_float_slice.iter().enumerate() {
        // Clamp and convert to i16 in one step
        let short_val = (float_val.max(-1.0).min(1.0) * 32767.0) as i16;
        unsafe {
            *buffer_ptr.add(i) = short_val;
        }
    }

    // Return the LSNR (Level of Suppressed Noise Ratio)
    lsnr as jfloat
}

