//! Private PyO3 extension for the public `tencirpauli` Python package.

use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;
use tencir_pauli_core::{packed_word_count, Complex64, PauliError, PauliOperator, PauliWord};

type CanonicalizeOutput = (Vec<Vec<u8>>, Vec<f64>, Vec<f64>);

fn map_error(error: PauliError) -> PyErr {
    let message = error.to_string();
    match error {
        PauliError::Overflow { .. } => PyOverflowError::new_err(message),
        _ => PyValueError::new_err(message),
    }
}

fn build_word(nqubits: usize, x_words: Vec<u64>, z_words: Vec<u64>) -> PyResult<PauliWord> {
    PauliWord::from_words(nqubits, x_words, z_words).map_err(map_error)
}

fn complex_coefficients(re: Vec<f64>, im: Vec<f64>) -> PyResult<Vec<Complex64>> {
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

#[pyfunction]
fn pauli_weight(nqubits: usize, x_words: Vec<u64>, z_words: Vec<u64>) -> PyResult<u32> {
    Ok(build_word(nqubits, x_words, z_words)?.weight())
}

#[pyfunction]
fn pauli_support(nqubits: usize, x_words: Vec<u64>, z_words: Vec<u64>) -> PyResult<Vec<usize>> {
    Ok(build_word(nqubits, x_words, z_words)?.support())
}

#[pyfunction]
fn pauli_codes(nqubits: usize, x_words: Vec<u64>, z_words: Vec<u64>) -> PyResult<Vec<u8>> {
    Ok(build_word(nqubits, x_words, z_words)?.codes())
}

#[pyfunction]
fn pauli_from_codes(nqubits: usize, codes: Vec<u8>) -> PyResult<(Vec<u64>, Vec<u64>)> {
    let word = PauliWord::from_codes(nqubits, &codes).map_err(map_error)?;
    Ok((word.x_words().to_vec(), word.z_words().to_vec()))
}

#[pyfunction]
fn pauli_batch_from_codes(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
) -> PyResult<(usize, Vec<u64>, Vec<u64>)> {
    let word_count = packed_word_count(nqubits);
    let mut x_words = Vec::with_capacity(structures.len() * word_count);
    let mut z_words = Vec::with_capacity(structures.len() * word_count);
    for structure in structures {
        let word = PauliWord::from_codes(nqubits, &structure).map_err(map_error)?;
        x_words.extend_from_slice(word.x_words());
        z_words.extend_from_slice(word.z_words());
    }
    Ok((word_count, x_words, z_words))
}

#[pyfunction]
fn pauli_multiply(
    nqubits: usize,
    left_codes: Vec<u8>,
    right_codes: Vec<u8>,
) -> PyResult<(Vec<u8>, u8)> {
    let left = PauliWord::from_codes(nqubits, &left_codes).map_err(map_error)?;
    let right = PauliWord::from_codes(nqubits, &right_codes).map_err(map_error)?;
    let (result, phase) = left.multiply(&right).map_err(map_error)?;
    Ok((
        result.codes(),
        match phase {
            tencir_pauli_core::PauliPhase::PlusOne => 0,
            tencir_pauli_core::PauliPhase::PlusI => 1,
            tencir_pauli_core::PauliPhase::MinusOne => 2,
            tencir_pauli_core::PauliPhase::MinusI => 3,
        },
    ))
}

#[pyfunction]
fn pauli_symplectic_inner_product(
    nqubits: usize,
    x_words_left: Vec<u64>,
    z_words_left: Vec<u64>,
    x_words_right: Vec<u64>,
    z_words_right: Vec<u64>,
) -> PyResult<u8> {
    let left = build_word(nqubits, x_words_left, z_words_left)?;
    let right = build_word(nqubits, x_words_right, z_words_right)?;
    left.symplectic_inner_product(&right).map_err(map_error)
}

#[pyfunction]
fn pauli_commutes(
    nqubits: usize,
    x_words_left: Vec<u64>,
    z_words_left: Vec<u64>,
    x_words_right: Vec<u64>,
    z_words_right: Vec<u64>,
) -> PyResult<bool> {
    let left = build_word(nqubits, x_words_left, z_words_left)?;
    let right = build_word(nqubits, x_words_right, z_words_right)?;
    left.commutes_with(&right).map_err(map_error)
}

#[pyfunction]
fn pauli_canonicalize(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
) -> PyResult<CanonicalizeOutput> {
    let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
    let operator =
        PauliOperator::from_terms(nqubits, &structures, &coefficients).map_err(map_error)?;
    let mut result_structures = Vec::with_capacity(operator.terms().len());
    let mut result_re = Vec::with_capacity(operator.terms().len());
    let mut result_im = Vec::with_capacity(operator.terms().len());
    for term in operator.terms() {
        result_structures.push(term.word.codes());
        result_re.push(term.coefficient.re);
        result_im.push(term.coefficient.im);
    }
    Ok((result_structures, result_re, result_im))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(pauli_weight, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_support, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_codes, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_from_codes, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_batch_from_codes, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_multiply, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_symplectic_inner_product, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_commutes, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_canonicalize, module)?)?;
    Ok(())
}
