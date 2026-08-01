//! Private PyO3 extension for the public `tencirpauli` Python package.

use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;
use tencir_pauli_core::{packed_word_count, Complex64, PauliError, PauliOperator, PauliWord};

type CanonicalizeOutput = (Vec<Vec<u8>>, Vec<f64>, Vec<f64>);
type CanonicalizeInput = (Vec<Vec<u8>>, Vec<f64>, Vec<f64>);

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

fn operator_output(operator: &PauliOperator) -> CanonicalizeOutput {
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

fn build_operator(
    nqubits: usize,
    structures: &[Vec<u8>],
    coefficients_re: &[f64],
    coefficients_im: &[f64],
) -> PyResult<PauliOperator> {
    let coefficients = complex_coefficients(coefficients_re.to_vec(), coefficients_im.to_vec())?;
    PauliOperator::from_terms(nqubits, structures, &coefficients).map_err(map_error)
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
    let operator = build_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
    Ok(operator_output(&operator))
}

#[pyfunction]
fn pauli_operator_binary(
    nqubits: usize,
    left: CanonicalizeInput,
    right: CanonicalizeInput,
    operation: u8,
) -> PyResult<CanonicalizeOutput> {
    let left_operator = build_operator(nqubits, &left.0, &left.1, &left.2)?;
    let right_operator = build_operator(nqubits, &right.0, &right.1, &right.2)?;
    let result = match operation {
        0 => left_operator.add(&right_operator),
        1 => left_operator.multiply(&right_operator),
        2 => left_operator.commutator(&right_operator),
        3 => left_operator.anticommutator(&right_operator),
        _ => return Err(PyValueError::new_err("unknown Pauli operator operation")),
    }
    .map_err(map_error)?;
    Ok(operator_output(&result))
}

#[pyfunction]
fn pauli_operator_scale(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    scalar_re: f64,
    scalar_im: f64,
) -> PyResult<CanonicalizeOutput> {
    let operator = build_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
    let result = operator
        .scale(Complex64::new(scalar_re, scalar_im))
        .map_err(map_error)?;
    Ok(operator_output(&result))
}

#[pyfunction]
fn pauli_operator_adjoint(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
) -> PyResult<CanonicalizeOutput> {
    let operator = build_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
    Ok(operator_output(&operator.adjoint()))
}

#[pyfunction]
fn pauli_operator_is_hermitian(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    tolerance: f64,
) -> PyResult<bool> {
    let operator = build_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(PyValueError::new_err(
            "Hermiticity tolerance must be finite and non-negative",
        ));
    }
    Ok(operator.is_hermitian(tolerance))
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
    module.add_function(wrap_pyfunction!(pauli_operator_binary, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_operator_scale, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_operator_adjoint, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_operator_is_hermitian, module)?)?;
    Ok(())
}
