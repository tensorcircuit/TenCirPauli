//! Private PyO3 extension for the public `tencirpauli` Python package.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tencir_pauli_core::PauliWord;

fn build_word(nqubits: usize, x_words: Vec<u64>, z_words: Vec<u64>) -> PyResult<PauliWord> {
    PauliWord::from_words(nqubits, x_words, z_words)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn pauli_weight(nqubits: usize, x_words: Vec<u64>, z_words: Vec<u64>) -> PyResult<u32> {
    Ok(build_word(nqubits, x_words, z_words)?.weight())
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
    left.commutes_with(&right)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(pauli_weight, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_commutes, module)?)?;
    Ok(())
}
