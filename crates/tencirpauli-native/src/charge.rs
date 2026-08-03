use numpy::{Complex64 as NumpyComplex128, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use tencir_pauli_core::Complex64;

/// Apply a deterministic restricted transition list in one coarse native call.
///
/// Transition construction remains a Python-facing structured operation in the
/// first slice, but the repeated state-vector kernel releases the GIL and uses
/// contiguous borrowed arrays without per-transition Python objects.
#[pyfunction]
pub(crate) fn charge_mvp_apply<'py>(
    py: Python<'py>,
    dimension: usize,
    rows: PyReadonlyArray1<'py, u64>,
    columns: PyReadonlyArray1<'py, u64>,
    coefficients: PyReadonlyArray1<'py, NumpyComplex128>,
    state: PyReadonlyArray1<'py, NumpyComplex128>,
    max_bytes: usize,
) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
    let row_values = rows
        .as_slice()
        .map_err(|_| PyValueError::new_err("charge rows must be C-contiguous"))?;
    let column_values = columns
        .as_slice()
        .map_err(|_| PyValueError::new_err("charge columns must be C-contiguous"))?;
    let coefficient_values = coefficients
        .as_slice()
        .map_err(|_| PyValueError::new_err("charge coefficients must be C-contiguous"))?;
    let state_values = state
        .as_slice()
        .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
    if state_values.len() != dimension {
        return Err(PyValueError::new_err(format!(
            "state must have shape ({dimension},), got ({},)",
            state_values.len()
        )));
    }
    if row_values.len() != column_values.len() || row_values.len() != coefficient_values.len() {
        return Err(PyValueError::new_err(
            "restricted transition arrays must have equal lengths",
        ));
    }
    let output_bytes = dimension
        .checked_mul(std::mem::size_of::<NumpyComplex128>())
        .ok_or_else(|| PyMemoryError::new_err("charge MVP output size overflow"))?;
    if output_bytes > max_bytes {
        return Err(PyMemoryError::new_err(format!(
            "charge MVP output requires approximately {output_bytes} bytes, exceeding max_bytes={max_bytes}"
        )));
    }
    let values = py.allow_threads(|| {
        let mut output = vec![Complex64::new(0.0, 0.0); dimension];
        for ((&row, &column), &coefficient) in
            row_values.iter().zip(column_values).zip(coefficient_values)
        {
            let row = usize::try_from(row)
                .map_err(|_| PyValueError::new_err("restricted row index overflow"))?;
            let column = usize::try_from(column)
                .map_err(|_| PyValueError::new_err("restricted column index overflow"))?;
            if row >= dimension || column >= dimension {
                return Err(PyValueError::new_err(
                    "restricted transition index is outside the sector dimension",
                ));
            }
            output[row] += coefficient * state_values[column];
        }
        Ok::<Vec<Complex64>, PyErr>(output)
    })?;
    Ok(PyArray1::from_vec(py, values))
}
