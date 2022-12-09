use std::str::FromStr;
use std::thread;
use std::time::Instant;

use crossbeam_channel::{Receiver, TryRecvError};
use df::augmentations::seed_from_u64;
use df::dataloader::{DataLoader, DfDataloaderError};
use df::dataset::*;
use df::hdf5_key_cache::*;
use df::util::{init_logger, DfLogger, LogMessage};
use df::Complex32;
use ndarray::{ArrayD, ShapeError};
use numpy::{IntoPyArray, PyArray1, PyArray4};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;

#[pymodule]
fn libdfdata(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<_FdDataLoader>()?;
    Ok(())
}

#[pyclass]
struct _FdDataLoader {
    loader: DataLoader,
    finished: bool,
    logger: Receiver<LogMessage>,
}

// TODO: Does not work due to pyo3 restrictions; instead return tuples
// #[pyclass]
// pub struct _PyBatch {
//     pub speech: PyArray3<f32>,
//     pub noise: PyArray3<f32>,
//     pub noisy: PyArray3<f32>,
//     pub lengths: PyArray1<usize>,
//     pub snr: PyArray1<i8>,
//     pub gain: PyArray1<i8>,
// }
// unsafe impl Send for _PyBatch {}
//
// impl _PyBatch {
//     fn from_batch<'py>(py: Python<'py>, batch: DsBatch) -> Py<Self> {
//         Py::new(
//             py,
//             _PyBatch {
//                 speech: PyArray3::from_owned_array(py, batch.speech),
//                 noise: PyArray3::from_owned_array(py, batch.noise),
//                 noisy: PyArray3::from_owned_array(py, batch.noisy),
//                 lengths: PyArray1::from_owned_array(py, batch.lengths),
//                 snr: PyArray1::from_vec(py, batch.snr),
//                 gain: PyArray1::from_vec(py, batch.gain),
//             },
//         )
//         .unwrap()
//     }
// }

type FdBatch<'py> = (
    &'py PyArray4<Complex32>, // speech
    &'py PyArray4<Complex32>, // noisy
    &'py PyArray4<f32>,       // feat_erb
    &'py PyArray4<Complex32>, // feat_spec
    &'py PyArray1<usize>,     // lengths
    &'py PyArray1<usize>,     // max_freq
    &'py PyArray1<i8>,        // snr
    &'py PyArray1<i8>,        // gain
    &'py PyArray1<f32>,       // Timings until each sample and the overall batch was ready
    &'py PyArray1<usize>,     // clean ids
);

#[pymethods]
impl _FdDataLoader {
    #[allow(clippy::too_many_arguments)]
    #[new]
    fn new(
        py: Python,
        ds_dir: &str,
        config_path: &str,
        sr: usize,
        batch_size: usize,
        fft_size: usize,
        batch_size_eval: Option<usize>,
        max_len_s: Option<f32>,
        hop_size: Option<usize>,
        nb_erb: Option<usize>,
        nb_spec: Option<usize>,
        norm_alpha: Option<f32>,
        num_threads: Option<usize>,
        prefetch: Option<usize>,
        p_reverb: Option<f32>,
        p_bw_ext: Option<f32>,
        p_clipping: Option<f32>,
        p_zeroing: Option<f32>,
        p_air_absorption: Option<f32>,
        drop_last: Option<bool>,
        overfit: Option<bool>,
        seed: Option<u64>,
        min_nb_erb_freqs: Option<usize>,
        global_sampling_factor: Option<f32>,
        snrs: Option<Vec<i8>>,
        gains: Option<Vec<i8>>,
        log_level: Option<&str>,
    ) -> PyResult<Self> {
        let (logger, log_receiver) = DfLogger::build(
            log::Level::from_str(log_level.unwrap_or("info")).expect("Could not parse log level"),
        );
        init_logger(logger);
        seed_from_u64(42);
        let mut cfg = match DatasetConfigJson::open(config_path) {
            Err(e) => {
                return Err(PyRuntimeError::new_err(format!(
                    "DF dataset config not found at '{config_path}' ({e:?})"
                )))
            }
            Ok(cfg) => cfg,
        };
        load_hdf5_key_cache(config_path, &mut cfg);
        let mut ds_builder = DatasetBuilder::new(ds_dir, sr)
            .df_params(fft_size, hop_size, nb_erb, nb_spec, norm_alpha);
        py.check_signals()?;
        if let Some(max_len_s) = max_len_s {
            ds_builder = ds_builder.max_len(max_len_s)
        }
        if let Some(seed) = seed {
            ds_builder = ds_builder.seed(seed)
        }
        if let Some(p_reverb) = p_reverb {
            ds_builder = ds_builder.prob_reverberation(p_reverb)
        }
        if let Some(p) = p_bw_ext {
            ds_builder = ds_builder.bandwidth_extension(p)
        }
        if let Some(p) = p_clipping {
            ds_builder = ds_builder.clipping_distortion(p)
        }
        if let Some(p) = p_zeroing {
            ds_builder = ds_builder.zeroing_distortion(p)
        }
        if let Some(p) = p_air_absorption {
            ds_builder = ds_builder.air_absorption_distortion(p)
        }
        if let Some(nb_freqs) = min_nb_erb_freqs {
            ds_builder = ds_builder.min_nb_erb_freqs(nb_freqs)
        }
        if let Some(f) = global_sampling_factor {
            ds_builder = ds_builder.global_sample_factor(f)
        }
        if let Some(snrs) = snrs {
            ds_builder = ds_builder.snrs(snrs);
        }
        if let Some(gains) = gains {
            ds_builder = ds_builder.gains(gains);
        }
        if let Some(num_threads) = num_threads {
            ds_builder = ds_builder.num_threads(num_threads);
        }
        let valid_handle = {
            let valid_cfg = cfg.split_config(Split::Valid);
            let valid_ds_builder = ds_builder.clone();
            thread::spawn(|| valid_ds_builder.dataset(valid_cfg).build_fft_dataset())
        };
        let test_handle = {
            let test_cfg = cfg.split_config(Split::Test);
            let test_ds_builder = ds_builder.clone();
            thread::spawn(|| test_ds_builder.dataset(test_cfg).build_fft_dataset())
        };
        ds_builder = ds_builder.p_sample_full_speech(1.0);
        let train_handle = {
            let train_cfg = cfg.split_config(Split::Train);
            thread::spawn(|| ds_builder.dataset(train_cfg).build_fft_dataset())
        };
        let msg = "Unable to join dataset builder thread";
        let valid_ds = valid_handle.join().expect(msg).to_py_err()?;
        fetch_hdf5_keys_from_ds(ds_dir, &mut cfg.valid, &valid_ds);
        py.check_signals()?;
        let test_ds = test_handle.join().expect(msg).to_py_err()?;
        fetch_hdf5_keys_from_ds(ds_dir, &mut cfg.test, &test_ds);
        py.check_signals()?;
        let train_ds = train_handle.join().expect(msg).to_py_err()?;
        fetch_hdf5_keys_from_ds(ds_dir, &mut cfg.train, &train_ds);
        write_hdf5_key_cache(config_path, &cfg);
        py.check_signals()?;
        let ds = Datasets {
            train: train_ds,
            valid: valid_ds,
            test: test_ds,
        };
        let mut dl_builder = DataLoader::builder(ds).batch_size(batch_size);
        if let Some(num_threads) = num_threads {
            dl_builder = dl_builder.num_threads(num_threads);
        }
        if let Some(prefetch) = prefetch {
            dl_builder = dl_builder.prefetch(prefetch);
        }
        if let Some(bs_eval) = batch_size_eval {
            dl_builder = dl_builder.batch_size_eval(bs_eval);
        }
        if drop_last.unwrap_or(false) {
            dl_builder = dl_builder.drop_last();
        }
        if overfit.unwrap_or(false) {
            dl_builder = dl_builder.overfit();
        }
        py.check_signals()?;
        let loader = dl_builder.build().to_py_err()?;
        Ok(_FdDataLoader {
            loader,
            finished: false,
            logger: log_receiver,
        })
    }

    fn start_epoch(&mut self, split: &str, seed: usize) -> PyResult<()> {
        self.finished = false;
        match self.loader.start_epoch(split, seed) {
            Err(e) => Err(PyValueError::new_err(e.to_string())),
            Ok(()) => Ok(()),
        }
    }

    fn get_batch<'py>(&'py mut self, py: Python<'py>) -> PyResult<FdBatch<'py>> {
        let t0 = Instant::now();
        if self.finished {
            return Err(PyStopIteration::new_err("Epoch finished"));
        }
        match self.loader.get_batch::<Complex32>().to_py_err()? {
            Some(batch) => {
                let erb = batch.feat_erb.unwrap_or_else(|| ArrayD::zeros(vec![1, 1, 1, 1]));
                let spec = batch.feat_spec.unwrap_or_else(|| ArrayD::zeros(vec![1, 1, 1, 1]));
                Ok((
                    batch.speech.into_dimensionality().to_py_err()?.into_pyarray(py),
                    batch.noisy.into_dimensionality().to_py_err()?.into_pyarray(py),
                    erb.into_dimensionality().to_py_err()?.into_pyarray(py),
                    spec.into_dimensionality().to_py_err()?.into_pyarray(py),
                    batch.lengths.into_pyarray(py),
                    batch.max_freq.into_pyarray(py),
                    batch.snr.into_pyarray(py),
                    batch.gain.into_pyarray(py),
                    push_ret(batch.timings, (Instant::now() - t0).as_secs_f32()).into_pyarray(py),
                    batch.ids.into_pyarray(py),
                ))
            }
            None => {
                self.finished = true;
                Err(PyStopIteration::new_err("Epoch finished"))
            }
        }
    }

    fn cleanup(&mut self) -> PyResult<()> {
        self.loader.join_fill_thread().to_py_err()?;
        Ok(())
    }

    fn dataloader_len(&self, split: &str) -> usize {
        self.loader.dataloader_len(split)
    }

    fn dataset_len(&self, split: &str) -> usize {
        self.loader.dataset_len(split)
    }

    fn set_batch_size(&mut self, batch_size: usize, split: &str) {
        self.loader.set_batch_size(batch_size, split)
    }

    fn batch_size(&self, split: &str) -> usize {
        self.loader.batch_size(split)
    }

    fn get_log_messages(&mut self) -> Vec<(String, String, Option<String>, Option<u32>)> {
        let mut messages = Vec::new();
        loop {
            match self.logger.try_recv() {
                Ok(m) => messages.push((
                    m.0.as_str().to_owned().replace("WARN", "WARNING"),
                    m.1,
                    m.2,
                    m.3,
                )),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    eprintln!("Dataloader logger disconnected unexpectetly!");
                    break;
                }
            }
        }
        messages
    }
}

fn push_ret<T>(mut a: Vec<T>, b: T) -> Vec<T> {
    a.push(b);
    a
}

trait ResultExt<T> {
    fn to_py_err(self) -> PyResult<T>;
}

impl<T> ResultExt<T> for std::result::Result<T, ShapeError> {
    fn to_py_err(self) -> PyResult<T> {
        match self {
            Ok(x) => Ok(x),
            Err(e) => Err(PyRuntimeError::new_err(format!("DF shape error: {e:?}"))),
        }
    }
}

impl<T> ResultExt<T> for std::result::Result<T, DfDatasetError> {
    fn to_py_err(self) -> PyResult<T> {
        match self {
            Ok(x) => Ok(x),
            Err(e) => Err(PyRuntimeError::new_err(format!("DF dataset error: {e:?}"))),
        }
    }
}

impl<T> ResultExt<T> for std::result::Result<T, DfDataloaderError> {
    fn to_py_err(self) -> PyResult<T> {
        match self {
            Ok(x) => Ok(x),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "DF dataloader error: {e:?}"
            ))),
        }
    }
}
