use numpy::{Complex64 as NumpyComplex128, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tencir_pauli_core::{structured_dense_matrix, Complex64, StructuredOperation};

use crate::convert::map_error;

#[pyfunction]
pub(crate) fn structured_dense(
    py: Python<'_>,
    local_dimensions: Vec<usize>,
    operations: Vec<Vec<(usize, u8, u32, u32)>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: u128,
) -> PyResult<(usize, Bound<'_, PyArray1<NumpyComplex128>>)> {
    if coefficients_re.len() != coefficients_im.len() {
        return Err(PyValueError::new_err(
            "real and imaginary coefficient lengths differ",
        ));
    }
    if operations.len() != coefficients_re.len() {
        return Err(PyValueError::new_err(
            "operation and coefficient lengths differ",
        ));
    }
    let coefficients = coefficients_re
        .into_iter()
        .zip(coefficients_im)
        .map(|(real, imaginary)| Complex64::new(real, imaginary))
        .collect::<Vec<_>>();
    let operations = operations
        .into_iter()
        .map(|term| {
            term.into_iter()
                .map(|(axis, kind, p, q)| StructuredOperation { axis, kind, p, q })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (dimension, values) = py
        .allow_threads(|| {
            structured_dense_matrix(&local_dimensions, &operations, &coefficients, max_bytes)
        })
        .map_err(map_error)?;
    Ok((dimension, PyArray1::from_vec(py, values)))
}
