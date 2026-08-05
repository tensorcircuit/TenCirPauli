use numpy::{Complex64 as NumpyComplex128, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tencir_pauli_core::{MvpPlan, MvpStrategy};

use crate::convert::{
    build_canonical_operator, map_error, numpy_complex_array, split_complex, BackendPlanOutput,
    CooOutput, CsrOutput, DenseOutput, NumpySparseOutput,
};
use crate::operator::NativePauliOperatorHandle;

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeMvpPlan {
    plan: MvpPlan,
}

#[pyfunction]
pub(crate) fn pauli_dense_handle(
    py: Python<'_>,
    handle: &NativePauliOperatorHandle,
    max_bytes: usize,
) -> PyResult<DenseOutput> {
    let (dimension, values) = py
        .allow_threads(|| handle.core().dense_matrix(max_bytes as u128))
        .map_err(map_error)?;
    let (real, imaginary) = split_complex(&values);
    Ok((dimension, real, imaginary))
}

#[pyfunction]
pub(crate) fn pauli_coo_handle(
    py: Python<'_>,
    handle: &NativePauliOperatorHandle,
    max_bytes: usize,
) -> PyResult<CooOutput> {
    let matrix = py
        .allow_threads(|| handle.core().coo_matrix(max_bytes as u128))
        .map_err(map_error)?;
    let (real, imaginary) = split_complex(&matrix.values);
    Ok((
        matrix.dimension,
        matrix.rows,
        matrix.columns,
        real,
        imaginary,
    ))
}

#[pyfunction]
pub(crate) fn pauli_csr_handle(
    py: Python<'_>,
    handle: &NativePauliOperatorHandle,
    max_bytes: usize,
) -> PyResult<CsrOutput> {
    let matrix = py
        .allow_threads(|| handle.core().csr_matrix(max_bytes as u128))
        .map_err(map_error)?;
    let (real, imaginary) = split_complex(&matrix.values);
    Ok((
        matrix.dimension,
        matrix.indptr,
        matrix.columns,
        real,
        imaginary,
    ))
}

#[pyfunction]
pub(crate) fn pauli_mvp_handle<'py>(
    py: Python<'py>,
    handle: &NativePauliOperatorHandle,
    state: PyReadonlyArray1<'py, NumpyComplex128>,
    max_bytes: usize,
) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
    let state_slice = state
        .as_slice()
        .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
    let values = py
        .allow_threads(|| handle.core().mvp(state_slice, max_bytes as u128))
        .map_err(map_error)?;
    Ok(PyArray1::from_vec(py, values))
}

#[pyfunction]
#[pyo3(signature = (handle, max_bytes, storage="lazy"))]
pub(crate) fn pauli_mvp_plan_handle(
    py: Python<'_>,
    handle: &NativePauliOperatorHandle,
    max_bytes: usize,
    storage: &str,
) -> PyResult<NativeMvpPlan> {
    let plan = py
        .allow_threads(|| match storage {
            "lazy" => tencir_pauli_core::MvpPlan::from_operator(handle.core()),
            "eager" => handle.core().mvp_plan(max_bytes as u128),
            _ => Err(tencir_pauli_core::PauliError::InvalidSector {
                context: "storage must be either 'eager' or 'lazy'",
            }),
        })
        .map_err(map_error)?;
    Ok(NativeMvpPlan { plan })
}

#[pyfunction]
pub(crate) fn pauli_backend_plan_handle(
    py: Python<'_>,
    handle: &NativePauliOperatorHandle,
    max_bytes: usize,
) -> PyResult<BackendPlanOutput> {
    let plan = py
        .allow_threads(|| handle.core().backend_mvp_plan(max_bytes as u128))
        .map_err(map_error)?;
    let (real, imaginary) = split_complex(&plan.coefficients);
    Ok((
        1,
        plan.nqubits,
        plan.word_count,
        plan.x_words,
        plan.z_words,
        real,
        imaginary,
    ))
}

#[pymethods]
impl NativeMvpPlan {
    #[getter]
    fn nqubits(&self) -> usize {
        self.plan.nqubits()
    }

    #[getter]
    fn term_count(&self) -> usize {
        self.plan.term_count()
    }

    #[getter]
    fn strategy(&self) -> &'static str {
        match self.plan.strategy() {
            MvpStrategy::XMaskDiagonal => "x_mask_diagonal",
            MvpStrategy::TermDirect => "term_direct",
        }
    }

    fn apply<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        max_bytes: usize,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let state_slice = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
        let values = py
            .allow_threads(|| self.plan.apply(state_slice, max_bytes as u128))
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn apply_into<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        mut output: PyReadwriteArray1<'py, NumpyComplex128>,
        max_bytes: usize,
    ) -> PyResult<()> {
        let state_slice = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
        let output_slice = output
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output must be C-contiguous"))?;
        py.allow_threads(|| {
            self.plan
                .apply_into(state_slice, output_slice, max_bytes as u128)
        })
        .map_err(map_error)
    }
}

#[pyfunction]
pub(crate) fn pauli_dense(
    py: Python<'_>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: usize,
) -> PyResult<DenseOutput> {
    let (dimension, values) = py.allow_threads(|| {
        let operator =
            build_canonical_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
        operator.dense_matrix(max_bytes as u128).map_err(map_error)
    })?;
    let (real, imaginary) = split_complex(&values);
    Ok((dimension, real, imaginary))
}

#[pyfunction]
pub(crate) fn pauli_dense_array<'py>(
    py: Python<'py>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: usize,
) -> PyResult<(usize, Bound<'py, PyArray1<NumpyComplex128>>)> {
    let (dimension, values) = py.allow_threads(|| {
        let operator =
            build_canonical_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
        operator.dense_matrix(max_bytes as u128).map_err(map_error)
    })?;
    Ok((dimension, numpy_complex_array(py, values)))
}

#[pyfunction]
pub(crate) fn pauli_coo(
    py: Python<'_>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: usize,
) -> PyResult<CooOutput> {
    let matrix = py.allow_threads(|| {
        let operator =
            build_canonical_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
        operator.coo_matrix(max_bytes as u128).map_err(map_error)
    })?;
    let (real, imaginary) = split_complex(&matrix.values);
    Ok((
        matrix.dimension,
        matrix.rows,
        matrix.columns,
        real,
        imaginary,
    ))
}

#[pyfunction]
pub(crate) fn pauli_coo_array<'py>(
    py: Python<'py>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: usize,
) -> PyResult<NumpySparseOutput<'py>> {
    let matrix = py.allow_threads(|| {
        let operator =
            build_canonical_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
        operator.coo_matrix(max_bytes as u128).map_err(map_error)
    })?;
    Ok((
        matrix.dimension,
        PyArray1::from_vec(py, matrix.rows),
        PyArray1::from_vec(py, matrix.columns),
        numpy_complex_array(py, matrix.values),
    ))
}

#[pyfunction]
pub(crate) fn pauli_csr(
    py: Python<'_>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: usize,
) -> PyResult<CsrOutput> {
    let matrix = py.allow_threads(|| {
        let operator =
            build_canonical_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
        operator.csr_matrix(max_bytes as u128).map_err(map_error)
    })?;
    let (real, imaginary) = split_complex(&matrix.values);
    Ok((
        matrix.dimension,
        matrix.indptr,
        matrix.columns,
        real,
        imaginary,
    ))
}

#[pyfunction]
pub(crate) fn pauli_csr_array<'py>(
    py: Python<'py>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: usize,
) -> PyResult<NumpySparseOutput<'py>> {
    let matrix = py.allow_threads(|| {
        let operator =
            build_canonical_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
        operator.csr_matrix(max_bytes as u128).map_err(map_error)
    })?;
    Ok((
        matrix.dimension,
        PyArray1::from_vec(py, matrix.indptr),
        PyArray1::from_vec(py, matrix.columns),
        numpy_complex_array(py, matrix.values),
    ))
}

#[pyfunction]
pub(crate) fn pauli_mvp_array<'py>(
    py: Python<'py>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    state: PyReadonlyArray1<'py, NumpyComplex128>,
    max_bytes: usize,
) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
    let operator = py.allow_threads(|| {
        build_canonical_operator(nqubits, &structures, &coefficients_re, &coefficients_im)
    })?;
    let state_slice = state
        .as_slice()
        .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
    let values = py
        .allow_threads(|| operator.mvp(state_slice, max_bytes as u128))
        .map_err(map_error)?;
    Ok(PyArray1::from_vec(py, values))
}

#[pyfunction]
#[pyo3(signature = (nqubits, structures, coefficients_re, coefficients_im, max_bytes, storage="lazy"))]
pub(crate) fn pauli_mvp_plan(
    py: Python<'_>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: usize,
    storage: &str,
) -> PyResult<NativeMvpPlan> {
    let plan = py.allow_threads(|| {
        let operator =
            build_canonical_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
        match storage {
            "lazy" => tencir_pauli_core::MvpPlan::from_operator(&operator),
            "eager" => operator.mvp_plan(max_bytes as u128),
            _ => Err(tencir_pauli_core::PauliError::InvalidSector {
                context: "storage must be either 'eager' or 'lazy'",
            }),
        }
        .map_err(map_error)
    })?;
    Ok(NativeMvpPlan { plan })
}

#[pyfunction]
pub(crate) fn pauli_backend_plan(
    py: Python<'_>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: usize,
) -> PyResult<BackendPlanOutput> {
    let plan = py.allow_threads(|| {
        let operator =
            build_canonical_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
        operator
            .backend_mvp_plan(max_bytes as u128)
            .map_err(map_error)
    })?;
    let (real, imaginary) = split_complex(&plan.coefficients);
    Ok((
        1,
        plan.nqubits,
        plan.word_count,
        plan.x_words,
        plan.z_words,
        real,
        imaginary,
    ))
}
