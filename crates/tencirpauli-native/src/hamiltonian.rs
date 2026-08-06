use numpy::{Complex64 as NumpyComplex128, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tencir_pauli_core::{MvpPlan, MvpStrategy};

use crate::convert::{map_error, numpy_complex_array};
use crate::operator::NativePauliOperatorHandle;

type NumpyDenseOutput<'py> = (usize, Bound<'py, PyArray1<NumpyComplex128>>);
type NumpyCooOutput<'py> = (
    usize,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);
type NumpyCsrOutput<'py> = (
    usize,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);
type BackendPlanOutput<'py> = (
    u8,
    usize,
    usize,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeMvpPlan {
    plan: MvpPlan,
}

#[pyfunction]
pub(crate) fn pauli_dense_handle<'py>(
    py: Python<'py>,
    handle: &NativePauliOperatorHandle,
    max_bytes: usize,
) -> PyResult<NumpyDenseOutput<'py>> {
    let (dimension, values) = py
        .allow_threads(|| handle.core().dense_matrix(max_bytes as u128))
        .map_err(map_error)?;
    Ok((dimension, numpy_complex_array(py, values)))
}

#[pyfunction]
pub(crate) fn pauli_coo_handle<'py>(
    py: Python<'py>,
    handle: &NativePauliOperatorHandle,
    max_bytes: usize,
) -> PyResult<NumpyCooOutput<'py>> {
    let matrix = py
        .allow_threads(|| handle.core().coo_matrix(max_bytes as u128))
        .map_err(map_error)?;
    Ok((
        matrix.dimension,
        PyArray1::from_vec(py, matrix.rows),
        PyArray1::from_vec(py, matrix.columns),
        numpy_complex_array(py, matrix.values),
    ))
}

#[pyfunction]
pub(crate) fn pauli_csr_handle<'py>(
    py: Python<'py>,
    handle: &NativePauliOperatorHandle,
    max_bytes: usize,
) -> PyResult<NumpyCsrOutput<'py>> {
    let matrix = py
        .allow_threads(|| handle.core().csr_matrix(max_bytes as u128))
        .map_err(map_error)?;
    Ok((
        matrix.dimension,
        PyArray1::from_vec(py, matrix.indptr),
        PyArray1::from_vec(py, matrix.columns),
        numpy_complex_array(py, matrix.values),
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
pub(crate) fn pauli_backend_plan_handle<'py>(
    py: Python<'py>,
    handle: &NativePauliOperatorHandle,
    max_bytes: usize,
) -> PyResult<BackendPlanOutput<'py>> {
    let plan = py
        .allow_threads(|| handle.core().backend_mvp_plan(max_bytes as u128))
        .map_err(map_error)?;
    Ok((
        1,
        plan.nqubits,
        plan.word_count,
        PyArray1::from_vec(py, plan.x_words),
        PyArray1::from_vec(py, plan.z_words),
        numpy_complex_array(py, plan.coefficients),
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
