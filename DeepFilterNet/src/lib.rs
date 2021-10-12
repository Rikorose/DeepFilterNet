#![feature(backtrace)]

use std::error::Error;
use std::sync::Arc;

use df::dataset::{DataLoader, DatasetBuilder, DatasetConfig, Datasets, DfDatasetError};
use df::transforms::{
    self, erb_inv_with_output as erb_inv_transform, erb_with_output as erb_transform,
    seed_from_u64, TransformError,
};
use df::{Complex32, DFState, UNIT_NORM_INIT};
use ndarray::{Array1, Array2, Array3, Array4, ArrayD, ArrayView4, Axis, ShapeError};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayDyn, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArrayDyn,
};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;

#[pyclass]
struct DF {
    state: DFState,
}

#[pymethods]
#[allow(clippy::upper_case_acronyms)]
impl DF {
    #[new]
    fn new(
        sr: usize,
        fft_size: usize,
        hop_size: usize,
        nb_bands: Option<usize>,
        min_nb_erb_freqs: Option<usize>,
    ) -> Self {
        DF {
            state: DFState::new(
                sr,
                fft_size,
                hop_size,
                nb_bands.unwrap_or(32),
                min_nb_erb_freqs.unwrap_or(1),
            ),
        }
    }

    fn analysis<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<&'py PyArray3<Complex32>> {
        self.state.reset();
        let frame_size = self.state.frame_size;
        let freq_size = self.state.freq_size;
        let channels = input.shape()[0];
        let freq_steps = input.shape()[1].div_euclid(frame_size);
        let mut output = Array3::<Complex32>::zeros((channels, freq_steps, freq_size));

        for (in_ch, mut out_ch) in
            input.as_array().axis_iter(Axis(0)).zip(output.axis_iter_mut(Axis(0)))
        {
            let in_slice = in_ch.as_slice().ok_or_else(|| {
                PyErr::new::<PyRuntimeError, _>("[df] Input array empty or not contiguous.")
            })?;
            let out_slice = out_ch.as_slice_mut().ok_or_else(|| {
                PyErr::new::<PyRuntimeError, _>("[df] Output array empty or not contiguous.")
            })?;
            let in_chunks = in_slice.chunks_exact(frame_size);
            let out_chunks = out_slice.chunks_exact_mut(freq_size);
            for (ichunk, ochunk) in in_chunks.into_iter().zip(out_chunks.into_iter()) {
                self.state.analysis(ichunk, ochunk)
            }
        }
        Ok(output.into_pyarray(py))
    }

    fn synthesis<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray3<Complex32>,
    ) -> PyResult<&'py PyArray2<f32>> {
        self.state.reset();
        let frame_size = self.state.frame_size;
        let freq_size = self.state.freq_size;
        let channels = input.shape()[0];
        let freq_steps = input.shape()[1];
        let out_steps = freq_steps * frame_size;
        let mut output = Array2::<f32>::zeros((channels, out_steps));

        let mut input = unsafe { input.as_array_mut() };
        for (mut in_ch, mut out_ch) in
            input.axis_iter_mut(Axis(0)).zip(output.axis_iter_mut(Axis(0)))
        {
            let in_slice = in_ch.as_slice_mut().ok_or_else(|| {
                PyErr::new::<PyRuntimeError, _>("[df] Input array empty or not contiguous.")
            })?;
            let out_slice = out_ch.as_slice_mut().ok_or_else(|| {
                PyErr::new::<PyRuntimeError, _>("[df] Output array empty or not contiguous.")
            })?;
            let in_chunks = in_slice.chunks_exact_mut(freq_size);
            let out_chunks = out_slice.chunks_exact_mut(frame_size);
            for (ichunk, ochunk) in in_chunks.into_iter().zip(out_chunks.into_iter()) {
                self.state.synthesis(ichunk, ochunk);
            }
        }
        Ok(output.into_pyarray(py))
    }

    fn erb_widths<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<usize>> {
        Ok(self.state.erb.clone().into_pyarray(py))
    }

    fn fft_window<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<f32>> {
        Ok(self.state.window.clone().into_pyarray(py))
    }

    fn reset(&mut self) {
        self.state.reset();
    }
}

#[pymodule]
fn pydf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DF>()?;
    m.add_class::<_TdDataLoader>()?;
    m.add_class::<_FdDataLoader>()?;

    #[pyfn(m)]
    #[pyo3(name = "erb")]
    fn erb<'py>(
        py: Python<'py>,
        input: PyReadonlyArrayDyn<Complex32>,
        erb_fb: Vec<usize>,
        db: Option<bool>,
    ) -> PyResult<&'py PyArrayDyn<f32>> {
        // Input shape [B, C, T, F]
        let indim = input.ndim();
        let input = input.as_array();
        let &f = input.shape().last().unwrap();
        let (bs, ch, t) = match indim {
            2 => (1, 1, input.len_of(Axis(0))),
            3 => (1, input.len_of(Axis(0)), input.len_of(Axis(1))),
            4 => (
                input.len_of(Axis(0)),
                input.len_of(Axis(1)),
                input.len_of(Axis(2)),
            ),
            n => {
                return Err(PyValueError::new_err(format!(
                    "Dimension not supported for erb: {}",
                    n,
                )))
            }
        };
        let input: ArrayView4<Complex32> = input
            .into_shape((bs, ch, t, f))
            .to_py_err()?
            .into_dimensionality()
            .to_py_err()?;
        let mut output = Array4::zeros((bs, ch, t, erb_fb.len()));

        for (in_b, mut out_b) in input.outer_iter().zip(output.outer_iter_mut()) {
            erb_transform(&in_b, db.unwrap_or(true), &mut out_b, &erb_fb).to_py_err()?;
        }
        let output: ArrayD<f32> = match indim {
            2 => output
                .into_shape((t, erb_fb.len()))
                .to_py_err()?
                .into_dimensionality()
                .to_py_err()?,
            3 => output
                .into_shape((ch, t, erb_fb.len()))
                .to_py_err()?
                .into_dimensionality()
                .to_py_err()?,
            _ => output.into_dimensionality().to_py_err()?,
        };
        Ok(output.into_pyarray(py))
    }

    #[pyfn(m)]
    #[pyo3(name = "erb_inv")]
    fn erb_inv<'py>(
        py: Python<'py>,
        input: PyReadonlyArrayDyn<f32>,
        erb_fb: Vec<usize>,
    ) -> PyResult<&'py PyArrayDyn<f32>> {
        // Input shape [B, C, T, E]
        let indim = input.ndim();
        let input = input.as_array();
        let &e = input.shape().last().unwrap();
        if e != erb_fb.len() {
            return Err(PyValueError::new_err(format!(
                "Number of erb bands do not match with input: {}, {}",
                e,
                erb_fb.len()
            )));
        }
        let (bs, ch, t) = match indim {
            2 => (1, 1, input.len_of(Axis(0))),
            3 => (1, input.len_of(Axis(0)), input.len_of(Axis(1))),
            4 => (
                input.len_of(Axis(0)),
                input.len_of(Axis(1)),
                input.len_of(Axis(2)),
            ),
            n => {
                return Err(PyValueError::new_err(format!(
                    "Dimension not supported for erb: {}",
                    n,
                )))
            }
        };
        let input: ArrayView4<f32> = input
            .into_shape((bs, ch, t, e))
            .to_py_err()?
            .into_dimensionality()
            .to_py_err()?;
        let freq_size = erb_fb.iter().sum();
        let mut output = Array4::zeros((bs, ch, t, freq_size));
        for (in_b, mut out_b) in input.outer_iter().zip(output.outer_iter_mut()) {
            erb_inv_transform(&in_b, &mut out_b, &erb_fb).to_py_err()?;
        }
        let output: ArrayD<f32> = match indim {
            2 => output
                .into_shape((t, freq_size))
                .to_py_err()?
                .into_dimensionality()
                .to_py_err()?,
            3 => output
                .into_shape((ch, t, freq_size))
                .to_py_err()?
                .into_dimensionality()
                .to_py_err()?,
            _ => output.into_dimensionality().to_py_err()?,
        };
        Ok(output.into_pyarray(py))
    }

    #[pyfn(m)]
    #[pyo3(name = "erb_norm")]
    fn erb_norm<'py>(
        py: Python<'py>,
        erb: PyReadonlyArray3<f32>,
        alpha: f32,
        state: Option<PyReadonlyArray2<f32>>,
    ) -> PyResult<&'py PyArray3<f32>> {
        // Input shape [C, T, F]
        // State shape [C, F]
        let mut erb = unsafe { erb.as_array_mut() };
        if let Some(state) = state {
            transforms::erb_norm(
                &mut erb.view_mut(),
                Some(unsafe { state.as_array_mut() }.to_owned()),
                alpha,
            )
            .to_py_err()?;
        } else {
            transforms::erb_norm(&mut erb.view_mut(), None, alpha).to_py_err()?;
        };
        Ok(erb.into_owned().into_pyarray(py))
    }

    #[pyfn(m)]
    #[pyo3(name = "unit_norm")]
    fn unit_norm<'py>(
        py: Python<'py>,
        spec: PyReadonlyArray3<Complex32>,
        alpha: f32,
        state: Option<PyReadonlyArray2<f32>>,
    ) -> PyResult<&'py PyArray3<Complex32>> {
        // Input shape [C, T, F]
        // State shape [C, F]
        let mut spec = spec.as_array().to_owned();
        if let Some(state) = state {
            transforms::unit_norm(
                &mut spec.view_mut(),
                Some(unsafe { state.as_array_mut() }.to_owned()),
                alpha,
            )
            .to_py_err()?;
        } else {
            transforms::unit_norm(&mut spec.view_mut(), None, alpha).to_py_err()?;
        };
        Ok(spec.into_pyarray(py))
    }

    #[pyfn(m)]
    #[pyo3(name = "unit_norm_init")]
    fn unit_norm_init<'py>(py: Python<'py>, num_freq_bins: usize) -> PyResult<&'py PyArray2<f32>> {
        let arr = Array1::<f32>::linspace(UNIT_NORM_INIT[0], UNIT_NORM_INIT[1], num_freq_bins)
            .into_shape([1, num_freq_bins])
            .to_py_err()?;
        Ok(arr.into_pyarray(py))
    }

    Ok(())
}

#[pyclass]
struct _TdDataLoader {
    loader: DataLoader<f32>,
}

#[pyclass]
struct _FdDataLoader {
    loader: DataLoader<Complex32>,
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
        let _e = df::dataset::hdf5_silence_errors();
        Ok(_FdDataLoader { loader })
    }

    fn start_epoch(&mut self, split: &str, seed: usize) -> PyResult<()> {
        match self.loader.start_epoch(split, seed) {
            Err(e) => Err(PyValueError::new_err(e.to_string())),
            Ok(()) => Ok(()),
        }
    }

    fn get_batch<'py>(&'py mut self, py: Python<'py>) -> PyResult<FdBatch<'py>> {
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
                println!("Epoch Finished");
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
            Err(e) => {
                if let Some(b) = e.backtrace() {
                    eprintln!("{}", b);
                }
                Err(PyRuntimeError::new_err(format!(
                    "DF dataset error: {:?}",
                    e
                )))
            }
        }
    }
}

impl<T> ResultExt<T> for std::result::Result<T, TransformError> {
    fn to_py_err(self) -> PyResult<T> {
        match self {
            Ok(x) => Ok(x),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "DF transform error: {:?}",
                e
            ))),
        }
    }
}
