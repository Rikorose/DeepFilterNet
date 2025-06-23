use df::transforms::{
    self, erb_inv_with_output as erb_inv_transform, erb_with_output as erb_transform,
    TransformError,
};
use df::{Complex32, DFState, UNIT_NORM_INIT};
use ndarray::{Array1, Array2, Array3, Array4, ArrayD, ArrayView4, Axis, ShapeError};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArrayDyn, PyUntypedArrayMethods, PyArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[pyclass]
struct DF {
    state: DFState,
}

#[pymethods]
#[allow(clippy::upper_case_acronyms)]
impl DF {
    #[new]
    #[pyo3(signature = (sr, fft_size, hop_size, nb_bands=None, min_nb_erb_freqs=None))]
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

    #[pyo3(signature = (input, reset=None))]
    fn analysis<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray2<'py, f32>,
        reset: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray3<Complex32>>> {
        let frame_size = self.state.frame_size;
        let freq_size = self.state.freq_size;
        let channels = input.shape()[0];
        let freq_steps = input.shape()[1].div_euclid(frame_size);
        let mut output = Array3::<Complex32>::zeros((channels, freq_steps, freq_size));

        for (in_ch, mut out_ch) in
            input.as_array().axis_iter(Axis(0)).zip(output.axis_iter_mut(Axis(0)))
        {
            if reset.unwrap_or(true) {
                self.reset();
            }
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

    #[pyo3(signature = (input, reset=None))]
    fn synthesis<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray3<Complex32>,
        reset: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
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
            if reset.unwrap_or(true) {
                self.reset();
            }
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

    fn erb_widths<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<usize>>> {
        Ok(self.state.erb.clone().into_pyarray(py))
    }

    fn fft_window<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(self.state.window.clone().into_pyarray(py))
    }

    fn sr(&self) -> usize {
        self.state.sr
    }

    fn fft_size(&self) -> usize {
        self.state.window_size
    }

    fn hop_size(&self) -> usize {
        self.state.frame_size
    }

    fn nb_erb(&self) -> usize {
        self.state.erb.len()
    }

    fn reset(&mut self) {
        self.state.reset();
    }
}

#[pymodule]
fn libdf(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DF>()?;

    #[pyfn(m)]
    #[pyo3(name = "erb")]
    #[pyo3(signature = (input, erb_fb, db=None))]
    fn erb<'py>(
        py: Python<'py>,
        input: PyReadonlyArrayDyn<Complex32>,
        erb_fb: PyReadonlyArray1<usize>,
        db: Option<bool>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
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
                    "Dimension not supported for erb: {n}",
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
            erb_transform(&in_b, db.unwrap_or(true), &mut out_b, erb_fb.as_slice()?).to_py_err()?;
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
        erb_fb: PyReadonlyArray1<usize>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
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
                    "Dimension not supported for erb: {n}",
                )))
            }
        };
        let input: ArrayView4<f32> = input
            .into_shape((bs, ch, t, e))
            .to_py_err()?
            .into_dimensionality()
            .to_py_err()?;
        let freq_size = erb_fb.as_array().sum();
        let mut output = Array4::zeros((bs, ch, t, freq_size));
        for (in_b, mut out_b) in input.outer_iter().zip(output.outer_iter_mut()) {
            erb_inv_transform(&in_b, &mut out_b, erb_fb.as_slice()?).to_py_err()?;
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
    #[pyo3(signature = (erb, alpha, state=None))]
    fn erb_norm<'py>(
        py: Python<'py>,
        erb: PyReadonlyArray3<f32>,
        alpha: f32,
        state: Option<PyReadonlyArray2<f32>>,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        // Input shape [C, T, F]
        // State shape [C, F]
        let mut erb = unsafe { erb.as_array_mut() };
        if let Some(state) = state {
            transforms::erb_norm(
                &mut erb.view_mut(),
                Some(unsafe { state.as_array() }.to_owned()),
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
    #[pyo3(signature = (spec, alpha, state=None))]
    fn unit_norm<'py>(
        py: Python<'py>,
        spec: PyReadonlyArray3<Complex32>,
        alpha: f32,
        state: Option<PyReadonlyArray2<f32>>,
    ) -> PyResult<Bound<'py, PyArray3<Complex32>>> {
        // Input shape [C, T, F]
        // State shape [C, F]
        let mut spec = spec.as_array().to_owned();
        if let Some(state) = state {
            transforms::unit_norm(
                &mut spec.view_mut(),
                Some(unsafe { state.as_array() }.to_owned()),
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
    fn unit_norm_init(py: Python, num_freq_bins: usize) -> PyResult<Bound<PyArray2<f32>>> {
        let arr = Array1::<f32>::linspace(UNIT_NORM_INIT[0], UNIT_NORM_INIT[1], num_freq_bins)
            .into_shape([1, num_freq_bins])
            .to_py_err()?;
        Ok(arr.into_pyarray(py))
    }

    Ok(())
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

impl<T> ResultExt<T> for std::result::Result<T, TransformError> {
    fn to_py_err(self) -> PyResult<T> {
        match self {
            Ok(x) => Ok(x),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "DF transform error: {e:?}"
            ))),
        }
    }
}
