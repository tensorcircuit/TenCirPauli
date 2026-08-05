use numpy::{Complex64 as NumpyComplex128, PyArray1};
use pyo3::exceptions::{PyMemoryError, PyOverflowError, PyValueError};
use pyo3::prelude::*;
use tencir_pauli_core::{Complex64, PauliError, PauliOperator};

pub(crate) type CanonicalizeOutput = (Vec<Vec<u8>>, Vec<f64>, Vec<f64>);
pub(crate) type CanonicalizeBatchOutput = (Vec<Vec<u8>>, Vec<f64>, Vec<f64>, Vec<usize>, Vec<u8>);
pub(crate) type CanonicalizeInput = (Vec<Vec<u8>>, Vec<f64>, Vec<f64>);
pub(crate) type DenseOutput = (usize, Vec<f64>, Vec<f64>);
pub(crate) type CooOutput = (usize, Vec<u64>, Vec<u64>, Vec<f64>, Vec<f64>);
pub(crate) type CsrOutput = (usize, Vec<u64>, Vec<u64>, Vec<f64>, Vec<f64>);
pub(crate) type BackendPlanOutput = (u8, usize, usize, Vec<u64>, Vec<u64>, Vec<f64>, Vec<f64>);
pub(crate) type NumpySparseOutput<'py> = (
    usize,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);
pub(crate) type NumpyCanonicalizeBatchOutput<'py> = (
    usize,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<u8>>,
);
pub(crate) type NumpyPauliPackedOutput<'py> = (
    usize,
    usize,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);
pub(crate) type NumpyPauliCodesOutput<'py> = (
    usize,
    usize,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);
pub(crate) type NumpyPauliStringsOutput<'py> = (Vec<String>, Bound<'py, PyArray1<NumpyComplex128>>);

pub(crate) fn numpy_complex_array<'py>(
    py: Python<'py>,
    values: Vec<Complex64>,
) -> Bound<'py, PyArray1<NumpyComplex128>> {
    PyArray1::from_vec(py, values)
}

pub(crate) fn map_error(error: PauliError) -> PyErr {
    let message = error.to_string();
    match error {
        PauliError::MemoryLimit { .. } => PyMemoryError::new_err(message),
        PauliError::Overflow { .. } => PyOverflowError::new_err(message),
        _ => PyValueError::new_err(message),
    }
}

pub(crate) fn complex_coefficients(re: Vec<f64>, im: Vec<f64>) -> PyResult<Vec<Complex64>> {
    if re.len() != im.len() {
        return Err(PyValueError::new_err(format!(
            "real and imaginary coefficient lengths differ: {} and {}",
            re.len(),
            im.len()
        )));
    }
    Ok(re
        .into_iter()
        .zip(im)
        .map(|(real, imaginary)| Complex64::new(real, imaginary))
        .collect())
}

pub(crate) fn operator_output(operator: &PauliOperator) -> CanonicalizeOutput {
    let mut result_structures = Vec::with_capacity(operator.terms().len());
    let mut result_re = Vec::with_capacity(operator.terms().len());
    let mut result_im = Vec::with_capacity(operator.terms().len());
    for term in operator.terms() {
        result_structures.push(term.word.codes());
        result_re.push(term.coefficient.re);
        result_im.push(term.coefficient.im);
    }
    (result_structures, result_re, result_im)
}

pub(crate) fn operator_packed_flat_output(
    operator: &PauliOperator,
) -> (usize, usize, Vec<u64>, Vec<u64>, Vec<Complex64>) {
    let word_count = operator.nqubits().div_ceil(64);
    let mut x_words = Vec::with_capacity(operator.terms().len() * word_count);
    let mut z_words = Vec::with_capacity(operator.terms().len() * word_count);
    let mut coefficients = Vec::with_capacity(operator.terms().len());
    for term in operator.terms() {
        x_words.extend_from_slice(term.word.x_words());
        z_words.extend_from_slice(term.word.z_words());
        coefficients.push(term.coefficient);
    }
    (
        operator.terms().len(),
        word_count,
        x_words,
        z_words,
        coefficients,
    )
}

pub(crate) fn operator_codes_flat_output(
    operator: &PauliOperator,
) -> (usize, usize, Vec<u8>, Vec<Complex64>) {
    let mut codes = Vec::with_capacity(operator.terms().len() * operator.nqubits());
    let mut coefficients = Vec::with_capacity(operator.terms().len());
    for term in operator.terms() {
        codes.extend(term.word.codes());
        coefficients.push(term.coefficient);
    }
    (
        operator.terms().len(),
        operator.nqubits(),
        codes,
        coefficients,
    )
}

pub(crate) fn operator_strings_flat_output(
    operator: &PauliOperator,
) -> (Vec<String>, Vec<Complex64>) {
    let mut strings = Vec::with_capacity(operator.terms().len());
    let mut coefficients = Vec::with_capacity(operator.terms().len());
    for term in operator.terms() {
        strings.push(
            term.word
                .codes()
                .into_iter()
                .map(|code| match code {
                    0 => 'I',
                    1 => 'X',
                    2 => 'Y',
                    3 => 'Z',
                    _ => unreachable!("canonical Pauli code must be in IXYZ"),
                })
                .collect(),
        );
        coefficients.push(term.coefficient);
    }
    (strings, coefficients)
}

pub(crate) fn phase_code(phase: tencir_pauli_core::PauliPhase) -> u8 {
    match phase {
        tencir_pauli_core::PauliPhase::PlusOne => 0,
        tencir_pauli_core::PauliPhase::PlusI => 1,
        tencir_pauli_core::PauliPhase::MinusOne => 2,
        tencir_pauli_core::PauliPhase::MinusI => 3,
    }
}

pub(crate) fn split_complex(values: &[Complex64]) -> (Vec<f64>, Vec<f64>) {
    values.iter().map(|value| (value.re, value.im)).unzip()
}

pub(crate) fn build_operator(
    nqubits: usize,
    structures: &[Vec<u8>],
    coefficients_re: &[f64],
    coefficients_im: &[f64],
) -> PyResult<PauliOperator> {
    let coefficients = complex_coefficients(coefficients_re.to_vec(), coefficients_im.to_vec())?;
    PauliOperator::from_terms(nqubits, structures, &coefficients).map_err(map_error)
}

pub(crate) fn build_canonical_operator(
    nqubits: usize,
    structures: &[Vec<u8>],
    coefficients_re: &[f64],
    coefficients_im: &[f64],
) -> PyResult<PauliOperator> {
    let coefficients = complex_coefficients(coefficients_re.to_vec(), coefficients_im.to_vec())?;
    PauliOperator::from_canonical_terms(nqubits, structures, &coefficients).map_err(map_error)
}

pub(crate) fn code_rows(codes: &[u8], row_count: usize, nqubits: usize) -> Vec<Vec<u8>> {
    if nqubits == 0 {
        return vec![Vec::new(); row_count];
    }
    codes.chunks_exact(nqubits).map(<[u8]>::to_vec).collect()
}
