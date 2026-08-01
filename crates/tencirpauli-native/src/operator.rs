use numpy::{
    Complex64 as NumpyComplex128, PyArray1, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tencir_pauli_core::{Complex64, PauliOperator};

use crate::convert::{
    build_canonical_operator, build_operator, code_rows, complex_coefficients, map_error,
    operator_output, phase_code, CanonicalizeBatchOutput, CanonicalizeInput, CanonicalizeOutput,
    NumpyCanonicalizeBatchOutput,
};

#[pyfunction]
pub(crate) fn pauli_canonicalize(
    py: Python<'_>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
) -> PyResult<CanonicalizeOutput> {
    let operator = py.allow_threads(|| {
        build_operator(nqubits, &structures, &coefficients_re, &coefficients_im)
    })?;
    Ok(operator_output(&operator))
}

#[pyfunction]
pub(crate) fn pauli_canonicalize_batch(
    py: Python<'_>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
) -> PyResult<CanonicalizeBatchOutput> {
    let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
    let result = py
        .allow_threads(|| PauliOperator::canonicalize(nqubits, &structures, &coefficients))
        .map_err(map_error)?;
    let mut result_structures = Vec::with_capacity(result.terms.len());
    let mut result_re = Vec::with_capacity(result.terms.len());
    let mut result_im = Vec::with_capacity(result.terms.len());
    for term in result.terms {
        result_structures.push(term.word.codes());
        result_re.push(term.coefficient.re);
        result_im.push(term.coefficient.im);
    }
    Ok((
        result_structures,
        result_re,
        result_im,
        result.input_to_canonical,
        result
            .phase_multipliers
            .into_iter()
            .map(phase_code)
            .collect(),
    ))
}

#[pyfunction]
pub(crate) fn pauli_canonicalize_array(
    py: Python<'_>,
    nqubits: usize,
    structures: PyReadonlyArray2<'_, u8>,
    coefficients: PyReadonlyArray1<'_, NumpyComplex128>,
) -> PyResult<CanonicalizeOutput> {
    let shape = structures.shape();
    if shape[1] != nqubits {
        return Err(PyValueError::new_err(format!(
            "expected structure width {nqubits}, got {}",
            shape[1]
        )));
    }
    if shape[0] != coefficients.len() {
        return Err(PyValueError::new_err(format!(
            "expected {} coefficients, got {}",
            shape[0],
            coefficients.len()
        )));
    }
    let code_slice = structures
        .as_slice()
        .map_err(|_| PyValueError::new_err("structures must be C-contiguous"))?;
    let coefficient_slice = coefficients
        .as_slice()
        .map_err(|_| PyValueError::new_err("coefficients must be C-contiguous"))?;
    let operator = py.allow_threads(|| {
        let rows = code_rows(code_slice, shape[0], nqubits);
        PauliOperator::from_terms(nqubits, &rows, coefficient_slice).map_err(map_error)
    })?;
    Ok(operator_output(&operator))
}

#[pyfunction]
pub(crate) fn pauli_canonicalize_batch_array(
    py: Python<'_>,
    nqubits: usize,
    structures: PyReadonlyArray2<'_, u8>,
    coefficients: PyReadonlyArray1<'_, NumpyComplex128>,
) -> PyResult<CanonicalizeBatchOutput> {
    let shape = structures.shape();
    if shape[1] != nqubits {
        return Err(PyValueError::new_err(format!(
            "expected structure width {nqubits}, got {}",
            shape[1]
        )));
    }
    if shape[0] != coefficients.len() {
        return Err(PyValueError::new_err(format!(
            "expected {} coefficients, got {}",
            shape[0],
            coefficients.len()
        )));
    }
    let code_slice = structures
        .as_slice()
        .map_err(|_| PyValueError::new_err("structures must be C-contiguous"))?;
    let coefficient_slice = coefficients
        .as_slice()
        .map_err(|_| PyValueError::new_err("coefficients must be C-contiguous"))?;
    let result = py
        .allow_threads(|| {
            let rows = code_rows(code_slice, shape[0], nqubits);
            PauliOperator::canonicalize(nqubits, &rows, coefficient_slice)
        })
        .map_err(map_error)?;
    let mut result_structures = Vec::with_capacity(result.terms.len());
    let mut result_re = Vec::with_capacity(result.terms.len());
    let mut result_im = Vec::with_capacity(result.terms.len());
    for term in result.terms {
        result_structures.push(term.word.codes());
        result_re.push(term.coefficient.re);
        result_im.push(term.coefficient.im);
    }
    Ok((
        result_structures,
        result_re,
        result_im,
        result.input_to_canonical,
        result
            .phase_multipliers
            .into_iter()
            .map(phase_code)
            .collect(),
    ))
}

#[pyfunction]
pub(crate) fn pauli_canonicalize_batch_numpy<'py>(
    py: Python<'py>,
    nqubits: usize,
    structures: PyReadonlyArray2<'py, u8>,
    coefficients: PyReadonlyArray1<'py, NumpyComplex128>,
) -> PyResult<NumpyCanonicalizeBatchOutput<'py>> {
    let shape = structures.shape();
    if shape[1] != nqubits {
        return Err(PyValueError::new_err(format!(
            "expected structure width {nqubits}, got {}",
            shape[1]
        )));
    }
    if shape[0] != coefficients.len() {
        return Err(PyValueError::new_err(format!(
            "expected {} coefficients, got {}",
            shape[0],
            coefficients.len()
        )));
    }
    let code_slice = structures
        .as_slice()
        .map_err(|_| PyValueError::new_err("structures must be C-contiguous"))?;
    let coefficient_slice = coefficients
        .as_slice()
        .map_err(|_| PyValueError::new_err("coefficients must be C-contiguous"))?;
    let result = py
        .allow_threads(|| {
            let rows = code_rows(code_slice, shape[0], nqubits);
            PauliOperator::canonicalize(nqubits, &rows, coefficient_slice)
        })
        .map_err(map_error)?;
    let canonical_count = result.terms.len();
    let mut canonical_codes = Vec::with_capacity(canonical_count.saturating_mul(nqubits));
    let mut canonical_coefficients = Vec::with_capacity(canonical_count);
    for term in result.terms {
        canonical_codes.extend(term.word.codes());
        canonical_coefficients.push(term.coefficient);
    }
    Ok((
        canonical_count,
        PyArray1::from_vec(py, canonical_codes),
        PyArray1::from_vec(py, canonical_coefficients),
        PyArray1::from_vec(py, result.input_to_canonical),
        PyArray1::from_vec(
            py,
            result
                .phase_multipliers
                .into_iter()
                .map(phase_code)
                .collect(),
        ),
    ))
}

#[pyfunction]
pub(crate) fn pauli_operator_binary(
    py: Python<'_>,
    nqubits: usize,
    left: CanonicalizeInput,
    right: CanonicalizeInput,
    operation: u8,
) -> PyResult<CanonicalizeOutput> {
    if operation > 3 {
        return Err(PyValueError::new_err("unknown Pauli operator operation"));
    }
    let result = py.allow_threads(|| {
        let left_operator = build_canonical_operator(nqubits, &left.0, &left.1, &left.2)?;
        let right_operator = build_canonical_operator(nqubits, &right.0, &right.1, &right.2)?;
        match operation {
            0 => left_operator.add(&right_operator),
            1 => left_operator.multiply(&right_operator),
            2 => left_operator.commutator(&right_operator),
            3 => left_operator.anticommutator(&right_operator),
            _ => unreachable!("operation was validated before releasing the GIL"),
        }
        .map_err(map_error)
    })?;
    Ok(operator_output(&result))
}

#[pyfunction]
pub(crate) fn pauli_operator_scale(
    py: Python<'_>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    scalar_re: f64,
    scalar_im: f64,
) -> PyResult<CanonicalizeOutput> {
    let result = py.allow_threads(|| {
        let operator =
            build_canonical_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
        operator
            .scale(Complex64::new(scalar_re, scalar_im))
            .map_err(map_error)
    })?;
    Ok(operator_output(&result))
}

#[pyfunction]
pub(crate) fn pauli_operator_adjoint(
    py: Python<'_>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
) -> PyResult<CanonicalizeOutput> {
    let operator = py.allow_threads(|| {
        build_canonical_operator(nqubits, &structures, &coefficients_re, &coefficients_im)
    })?;
    Ok(operator_output(&operator.adjoint()))
}

#[pyfunction]
pub(crate) fn pauli_operator_is_hermitian(
    py: Python<'_>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    tolerance: f64,
) -> PyResult<bool> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(PyValueError::new_err(
            "Hermiticity tolerance must be finite and non-negative",
        ));
    }
    py.allow_threads(|| {
        let operator =
            build_canonical_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
        Ok(operator.is_hermitian(tolerance))
    })
}
