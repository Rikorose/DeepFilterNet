use std::{cell::UnsafeCell, path::Path, rc::Rc};

use bincode::error::{DecodeError, EncodeError};
use thiserror::Error;

use crate::{dataset::Sample, util::thread_rng, Complex32};

#[allow(clippy::enum_variant_names)]
#[derive(Error, Debug)]
pub enum DfCacheError {
    #[error("Sled Error")]
    SledError(#[from] sled::Error),
    #[error("Bincode Encoder Error")]
    BincodeEncoderError(#[from] EncodeError),
    #[error("Bincode Decoder Error")]
    BincodeDecoderError(#[from] DecodeError),
    #[error("DF Utils Error")]
    UtilsError(#[from] crate::util::UtilsError),
}
type Result<T> = std::result::Result<T, DfCacheError>;

/// Cache for validation dataset
pub struct ValidCache {
    db: sled::Db,
    config: bincode::config::Configuration,
    hash: u64,
    n_samples: usize,
    max_size: f32, // in bytes
}

const BUFFER_SIZE: usize = 64 * 1024usize.pow(2); // 64 MB
thread_local! {
    static SAMPLE_BUFFER:  Rc<UnsafeCell<Vec<u8>>> = Rc::new(UnsafeCell::new(vec![0u8; BUFFER_SIZE]));
}

fn get_buffer() -> &'static mut [u8] {
    let ptr = SAMPLE_BUFFER.with(|b| b.clone()).get();
    unsafe { ptr.as_mut::<'static>().unwrap() }
}

#[inline]
fn u64_to_ivec(number: u64) -> sled::IVec {
    sled::IVec::from(number.to_be_bytes().to_vec())
}
#[inline]
fn bytes_to_u64(bytes: &[u8]) -> u64 {
    let array: [u8; 8] = bytes.try_into().unwrap();
    u64::from_be_bytes(array)
}
fn increment(old: Option<&[u8]>) -> Option<Vec<u8>> {
    let number = match old {
        Some(bytes) => {
            let number = bytes_to_u64(bytes);
            number + 1
        }
        None => 0,
    };

    Some(number.to_be_bytes().to_vec())
}

impl ValidCache {
    pub fn new(
        path: &impl AsRef<Path>,
        hash: u64,
        n_total_samples: usize,
        max_gb: Option<f32>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let db = sled::Config::new()
            .path(path)
            .use_compression(true)
            .compression_factor(10)
            .flush_every_ms(Some(10000))
            .open()?;
        let config = bincode::config::standard();
        let max_gb = std::env::var("DF_CACHE_MAX_GB")
            .map(|s| s.parse::<f32>().expect("Failed to parse `DF_CACHE_MAX_GB` env variable"))
            .unwrap_or_else(|_| max_gb.unwrap_or_default());
        log::info!(
            "Using validation dataset cache at {:?} (max size: {:?} GB)",
            &path,
            max_gb
        );
        let max_size = max_gb * 1000f32.powi(3);
        db.compare_and_swap(b"len", None as Option<&[u8]>, Some(&u64_to_ivec(0)))?
            .unwrap_or_default();
        Ok(ValidCache {
            db,
            config,
            hash,
            n_samples: n_total_samples,
            max_size,
        })
    }
    pub fn flush(&self) -> Result<usize> {
        Ok(self.db.flush()?)
    }
    #[inline]
    pub fn is_full(&self) -> Result<bool> {
        if self.max_size == 0. {
            return Ok(false);
        }
        Ok(self.db.size_on_disk()? as f32 > self.max_size)
    }
    fn should_cache(&self) -> Result<bool> {
        if self.max_size == 0. {
            return Ok(true);
        }
        if self.is_full()? {
            return Ok(false);
        }
        let avg_size = match self.len()? {
            0 => return Ok(true),
            n => self.db.size_on_disk()? as f32 / n as f32,
        };
        let total_estimate = avg_size * self.n_samples as f32;
        if total_estimate > self.max_size {
            let p = self.max_size / total_estimate;
            return Ok(p >= thread_rng()?.uniform(0f32, 1f32));
        }
        // If had an io error; return false
        Ok(!self.had_io_error()?)
    }
    pub fn had_io_error(&self) -> Result<bool> {
        Ok(self.db.contains_key(b"io_error")?)
    }
    pub fn log_io_error(&self) -> Result<()> {
        self.db.insert(b"io_error", &[1])?;
        Ok(())
    }
    pub fn increment_len(&self) -> Result<()> {
        self.db.update_and_fetch(b"len", increment)?;
        Ok(())
    }
    pub fn len(&self) -> Result<u64> {
        let bytes = self.db.get(b"len")?.unwrap();
        Ok(bytes_to_u64(&bytes))
    }
    pub fn cache_sample(&self, key: u64, sample: &Sample<Complex32>) -> Result<()> {
        #[cfg(feature = "dataset_timings")]
        let t0 = std::time::Instant::now();
        if !self.should_cache()? {
            return Ok(());
        }
        let data = get_buffer();
        let n_enc = bincode::serde::encode_into_slice(sample, data, self.config)?;
        let key = u64_to_ivec(key);
        match self.db.insert(&key, &data[..n_enc]) {
            Ok(_) => (),
            Err(sled::Error::Io(e)) => {
                eprintln!("io::Error during sled insert: {:?}", e);
                self.log_io_error()?;
            }
            Err(e) => return Err(e.into()),
        };
        self.increment_len()?;
        #[cfg(feature = "dataset_timings")]
        log::trace!(
            "Cached sample in {:?} us",
            (t0 - std::time::Instant::now()).as_micros()
        );
        Ok(())
    }
    pub fn load_sample(&self, key: u64) -> Result<Option<Sample<Complex32>>> {
        #[cfg(feature = "dataset_timings")]
        let t0 = std::time::Instant::now();
        let key = u64_to_ivec(key);
        let slice = match self.db.get(key)? {
            Some(s) => s,
            None => return Ok(None),
        };
        let (sample, _) = bincode::serde::decode_from_slice(slice.as_ref(), self.config)?;
        #[cfg(feature = "dataset_timings")]
        log::trace!(
            "Loaded cached sample in {:?} us",
            (t0 - std::time::Instant::now()).as_micros()
        );
        Ok(Some(sample))
    }
}
