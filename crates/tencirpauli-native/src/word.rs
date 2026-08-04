use pyo3::prelude::*;
use tencir_pauli_core::{packed_word_count, PauliWord};

use crate::convert::{map_error, phase_code};

fn build_word(nqubits: usize, x_words: Vec<u64>, z_words: Vec<u64>) -> PyResult<PauliWord> {
    PauliWord::from_words(nqubits, x_words, z_words).map_err(map_error)
}

#[pyfunction]
pub(crate) fn pauli_weight(nqubits: usize, x_words: Vec<u64>, z_words: Vec<u64>) -> PyResult<u32> {
    Ok(build_word(nqubits, x_words, z_words)?.weight())
}

#[pyfunction]
pub(crate) fn pauli_support(
    nqubits: usize,
    x_words: Vec<u64>,
    z_words: Vec<u64>,
) -> PyResult<Vec<usize>> {
    Ok(build_word(nqubits, x_words, z_words)?.support())
}

#[pyfunction]
pub(crate) fn pauli_codes(
    nqubits: usize,
    x_words: Vec<u64>,
    z_words: Vec<u64>,
) -> PyResult<Vec<u8>> {
    Ok(build_word(nqubits, x_words, z_words)?.codes())
}

#[pyfunction]
pub(crate) fn pauli_from_codes(nqubits: usize, codes: Vec<u8>) -> PyResult<(Vec<u64>, Vec<u64>)> {
    let word = PauliWord::from_codes(nqubits, &codes).map_err(map_error)?;
    Ok((word.x_words().to_vec(), word.z_words().to_vec()))
}

#[pyfunction]
pub(crate) fn pauli_batch_from_codes(
    py: Python<'_>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
) -> PyResult<(usize, Vec<u64>, Vec<u64>)> {
    py.allow_threads(|| {
        let word_count = packed_word_count(nqubits);
        let mut x_words = Vec::with_capacity(structures.len() * word_count);
        let mut z_words = Vec::with_capacity(structures.len() * word_count);
        for structure in structures {
            let word = PauliWord::from_codes(nqubits, &structure).map_err(map_error)?;
            x_words.extend_from_slice(word.x_words());
            z_words.extend_from_slice(word.z_words());
        }
        Ok((word_count, x_words, z_words))
    })
}

#[pyfunction]
pub(crate) fn pauli_multiply(
    py: Python<'_>,
    nqubits: usize,
    x_words_left: Vec<u64>,
    z_words_left: Vec<u64>,
    x_words_right: Vec<u64>,
    z_words_right: Vec<u64>,
) -> PyResult<(Vec<u64>, Vec<u64>, u8)> {
    py.allow_threads(|| {
        let left = PauliWord::from_words(nqubits, x_words_left, z_words_left).map_err(map_error)?;
        let right =
            PauliWord::from_words(nqubits, x_words_right, z_words_right).map_err(map_error)?;
        let (result, phase) = left.multiply(&right).map_err(map_error)?;
        Ok((
            result.x_words().to_vec(),
            result.z_words().to_vec(),
            phase_code(phase),
        ))
    })
}

#[pyfunction]
pub(crate) fn pauli_symplectic_inner_product(
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
pub(crate) fn pauli_commutes(
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
