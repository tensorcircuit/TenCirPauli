use pyo3::prelude::*;
use tencir_pauli_core::{canonicalize_majorana_terms, multiply_majorana_terms, MajoranaBatch};

use crate::convert::{complex_coefficients, map_error, split_complex};

type MajoranaOutput = (Vec<Vec<u64>>, Vec<f64>, Vec<f64>);

#[pyfunction]
pub(crate) fn majorana_canonicalize(
    py: Python<'_>,
    n_modes: usize,
    indices: Vec<Vec<u64>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: u128,
) -> PyResult<MajoranaOutput> {
    let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
    let result = py
        .allow_threads(|| canonicalize_majorana_terms(n_modes, &indices, &coefficients, max_bytes))
        .map_err(map_error)?;
    let (real, imaginary) = split_complex(&result.1);
    Ok((result.0, real, imaginary))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn majorana_multiply(
    py: Python<'_>,
    n_modes: usize,
    left_indices: Vec<Vec<u64>>,
    left_coefficients_re: Vec<f64>,
    left_coefficients_im: Vec<f64>,
    right_indices: Vec<Vec<u64>>,
    right_coefficients_re: Vec<f64>,
    right_coefficients_im: Vec<f64>,
    max_bytes: u128,
) -> PyResult<MajoranaOutput> {
    let left_coefficients = complex_coefficients(left_coefficients_re, left_coefficients_im)?;
    let right_coefficients = complex_coefficients(right_coefficients_re, right_coefficients_im)?;
    let result = py
        .allow_threads(|| {
            multiply_majorana_terms(
                n_modes,
                MajoranaBatch {
                    indices: &left_indices,
                    coefficients: &left_coefficients,
                },
                MajoranaBatch {
                    indices: &right_indices,
                    coefficients: &right_coefficients,
                },
                max_bytes,
            )
        })
        .map_err(map_error)?;
    let (real, imaginary) = split_complex(&result.1);
    Ok((result.0, real, imaginary))
}
