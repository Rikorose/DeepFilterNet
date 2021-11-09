use std::sync::Arc;

use df::augmentations::seed_from_u64;
use df::dataset::{DataLoader, DatasetBuilder, DatasetConfig, Datasets, DfDatasetError};
use df::Complex32;
use ndarray::{ArrayD, ShapeError};
use numpy::{IntoPyArray, PyArray1, PyArray3, PyArray4};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;

#[pymodule]
fn libdfdata(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<_TdDataLoader>()?;
    m.add_class::<_FdDataLoader>()?;
    Ok(())
}

#[pyclass]
struct _TdDataLoader {
    loader: DataLoader<f32>,
}

#[pyclass]
struct _FdDataLoader {
    loader: DataLoader<Complex32>,
    finished: bool,
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

type TdBatch<'py> = (
    &'py PyArray3<f32>,   // speech
    &'py PyArray3<f32>,   // noise
    &'py PyArray3<f32>,   // noisy
    &'py PyArray1<usize>, // lengths
    &'py PyArray1<usize>, // max_freq
    &'py PyArray1<i8>,    // snr
    &'py PyArray1<i8>,    // gain
    &'py PyArray1<u8>,    // attenuation limit
);

#[pymethods]
impl _TdDataLoader {
    #[allow(clippy::too_many_arguments)]
    #[new]
    fn new(
        ds_dir: &str,
        config_path: &str,
        sr: usize,
        batch_size: usize,
        batch_size_eval: Option<usize>,
        max_len_s: Option<f32>,
        num_threads: Option<usize>,
        prefetch: Option<usize>,
        p_atten_lim: Option<f32>,
        p_reverb: Option<f32>,
        overfit: Option<bool>,
        seed: Option<u64>,
        min_nb_erb_freqs: Option<usize>,
    ) -> PyResult<Self> {
        seed_from_u64(42);
        let cfg = DatasetConfig::open(config_path).to_py_err()?;

        let mut ds_builder = DatasetBuilder::new(ds_dir, sr);
        if let Some(max_len_s) = max_len_s {
            ds_builder = ds_builder.max_len(max_len_s)
        }
        if let Some(p_atten_lim) = p_atten_lim {
            ds_builder = ds_builder.prob_atten_lim(p_atten_lim)
        }
        if let Some(seed) = seed {
            ds_builder = ds_builder.seed(seed)
        }
        if let Some(p_reverb) = p_reverb {
            ds_builder = ds_builder.prob_reverberation(p_reverb)
        }
        if let Some(nb_freqs) = min_nb_erb_freqs {
            ds_builder = ds_builder.min_nb_erb_freqs(nb_freqs)
        }
        let valid_ds = ds_builder.clone().dataset(cfg.valid).build_td_dataset().to_py_err()?;
        let test_ds = ds_builder.clone().dataset(cfg.test).build_td_dataset().to_py_err()?;
        ds_builder = ds_builder.p_sample_full_speech(1.0);
        let train_ds = ds_builder.dataset(cfg.train).build_td_dataset().to_py_err()?;
        let ds = Datasets::new(Arc::new(train_ds), Arc::new(valid_ds), Arc::new(test_ds));
        let mut builder = DataLoader::builder(ds).batch_size(batch_size);
        if let Some(num_threads) = num_threads {
            builder = builder.num_threads(num_threads);
        }
        if let Some(prefetch) = prefetch {
            builder = builder.prefetch(prefetch);
        }
        if let Some(bs_eval) = batch_size_eval {
            builder = builder.batch_size_eval(bs_eval);
        }
        if let Some(overfit) = overfit {
            builder = builder.overfit(overfit);
        }
        let loader = builder.build().to_py_err()?;
        Ok(_TdDataLoader { loader })
    }

    fn start_epoch(&mut self, split: &str, seed: usize) -> PyResult<()> {
        match self.loader.start_epoch(split, seed) {
            Err(e) => Err(PyValueError::new_err(e.to_string())),
            Ok(()) => Ok(()),
        }
    }

    fn get_batch<'py>(&'py mut self, py: Python<'py>) -> PyResult<TdBatch<'py>> {
        match self.loader.get_batch::<f32>().to_py_err()? {
            Some(batch) => Ok((
                batch.speech.into_dimensionality().to_py_err()?.into_pyarray(py),
                batch.noise.into_dimensionality().to_py_err()?.into_pyarray(py),
                batch.noisy.into_dimensionality().to_py_err()?.into_pyarray(py),
                batch.lengths.into_pyarray(py),
                batch.max_freq.into_pyarray(py),
                batch.snr.into_pyarray(py),
                batch.gain.into_pyarray(py),
                batch.atten.into_pyarray(py),
            )),
            None => Err(PyStopIteration::new_err("Epoch finished")),
        }
    }

    fn cleanup(&mut self) -> PyResult<()> {
        self.loader.join_fill_thread().to_py_err()?;
        Ok(())
    }

    fn len_of(&self, split: &str) -> usize {
        self.loader.len_of(split)
    }

    fn dataset_len(&self, split: &str) -> usize {
        self.loader.dataset_len(split)
    }
}

type FdBatch<'py> = (
    &'py PyArray4<Complex32>, // speech
    &'py PyArray4<Complex32>, // noise
    &'py PyArray4<Complex32>, // noisy
    &'py PyArray4<f32>,       // feat_erb
    &'py PyArray4<Complex32>, // feat_spec
    &'py PyArray1<usize>,     // lengths
    &'py PyArray1<usize>,     // max_freq
    &'py PyArray1<i8>,        // snr
    &'py PyArray1<i8>,        // gain
    &'py PyArray1<u8>,        // attenuation limit
);

#[pymethods]
impl _FdDataLoader {
    #[allow(clippy::too_many_arguments)]
    #[new]
    fn new(
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
        p_atten_lim: Option<f32>,
        p_reverb: Option<f32>,
        overfit: Option<bool>,
        seed: Option<u64>,
        min_nb_erb_freqs: Option<usize>,
    ) -> PyResult<Self> {
        seed_from_u64(42);
        let cfg = match DatasetConfig::open(config_path) {
            Err(e) => {
                return Err(PyRuntimeError::new_err(format!(
                    "DF dataset config not found at '{}' ({:?})",
                    config_path, e
                )))
            }
            Ok(cfg) => cfg,
        };
        let mut ds_builder = DatasetBuilder::new(ds_dir, sr)
            .df_params(fft_size, hop_size, nb_erb, nb_spec, norm_alpha);
        if let Some(max_len_s) = max_len_s {
            ds_builder = ds_builder.max_len(max_len_s)
        }
        if let Some(p_atten_lim) = p_atten_lim {
            ds_builder = ds_builder.prob_atten_lim(p_atten_lim)
        }
        if let Some(seed) = seed {
            ds_builder = ds_builder.seed(seed)
        }
        if let Some(p_reverb) = p_reverb {
            ds_builder = ds_builder.prob_reverberation(p_reverb)
        }
        if let Some(nb_freqs) = min_nb_erb_freqs {
            ds_builder = ds_builder.min_nb_erb_freqs(nb_freqs)
        }
        let valid_ds = ds_builder.clone().dataset(cfg.valid).build_fft_dataset().to_py_err()?;
        let test_ds = ds_builder.clone().dataset(cfg.test).build_fft_dataset().to_py_err()?;
        ds_builder = ds_builder.p_sample_full_speech(1.0);
        let train_ds = ds_builder.clone().dataset(cfg.train).build_fft_dataset().to_py_err()?;
        let ds = Datasets::new(Arc::new(train_ds), Arc::new(valid_ds), Arc::new(test_ds));
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
        if let Some(overfit) = overfit {
            dl_builder = dl_builder.overfit(overfit);
        }
        let loader = dl_builder.build().to_py_err()?;
        Ok(_FdDataLoader {
            loader,
            finished: false,
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
        if self.finished {
            return Err(PyStopIteration::new_err("Epoch finished"));
        }
        match self.loader.get_batch::<Complex32>().to_py_err()? {
            Some(batch) => {
                let erb = batch.feat_erb.unwrap_or_else(|| ArrayD::zeros(vec![1, 1, 1, 1]));
                let spec = batch.feat_spec.unwrap_or_else(|| ArrayD::zeros(vec![1, 1, 1, 1]));
                Ok((
                    batch.speech.into_dimensionality().to_py_err()?.into_pyarray(py),
                    batch.noise.into_dimensionality().to_py_err()?.into_pyarray(py),
                    batch.noisy.into_dimensionality().to_py_err()?.into_pyarray(py),
                    erb.into_dimensionality().to_py_err()?.into_pyarray(py),
                    spec.into_dimensionality().to_py_err()?.into_pyarray(py),
                    batch.lengths.into_pyarray(py),
                    batch.max_freq.into_pyarray(py),
                    batch.snr.into_pyarray(py),
                    batch.gain.into_pyarray(py),
                    batch.atten.into_pyarray(py),
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

    fn len_of(&self, split: &str) -> usize {
        self.loader.len_of(split)
    }

    fn dataset_len(&self, split: &str) -> usize {
        self.loader.dataset_len(split)
    }
}

trait ResultExt<T> {
    fn to_py_err(self) -> PyResult<T>;
}

impl<T> ResultExt<T> for std::result::Result<T, ShapeError> {
    fn to_py_err(self) -> PyResult<T> {
        match self {
            Ok(x) => Ok(x),
            Err(e) => Err(PyRuntimeError::new_err(format!("DF shape error: {:?}", e))),
        }
    }
}

impl<T> ResultExt<T> for std::result::Result<T, DfDatasetError> {
    fn to_py_err(self) -> PyResult<T> {
        match self {
            Ok(x) => Ok(x),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "DF dataset error: {:?}",
                e
            ))),
        }
    }
}
