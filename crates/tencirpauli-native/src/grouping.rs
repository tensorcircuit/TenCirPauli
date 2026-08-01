use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tencir_pauli_core::{
    compatibility_matrix, group_words_bounded, incompatibility_edges, GroupingAlgorithm,
    GroupingMode, PauliWord,
};

use crate::convert::map_error;

#[pyfunction]
pub(crate) fn pauli_group(
    py: Python<'_>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    mode: u8,
    algorithm: u8,
    max_entries: usize,
) -> PyResult<Vec<Vec<usize>>> {
    let grouping_mode = match mode {
        0 => GroupingMode::QubitWise,
        1 => GroupingMode::General,
        _ => return Err(PyValueError::new_err("grouping mode must be 0 or 1")),
    };
    let grouping_algorithm = match algorithm {
        0 => GroupingAlgorithm::LargestFirst,
        1 => GroupingAlgorithm::Dsatur,
        _ => return Err(PyValueError::new_err("grouping algorithm must be 0 or 1")),
    };
    py.allow_threads(|| {
        let words = structures
            .iter()
            .map(|structure| PauliWord::from_codes(nqubits, structure))
            .collect::<Result<Vec<_>, _>>()?;
        group_words_bounded(&words, grouping_mode, grouping_algorithm, max_entries)
    })
    .map_err(map_error)
}

fn parse_grouping_mode(mode: u8) -> PyResult<GroupingMode> {
    match mode {
        0 => Ok(GroupingMode::QubitWise),
        1 => Ok(GroupingMode::General),
        _ => Err(PyValueError::new_err("grouping mode must be 0 or 1")),
    }
}

#[pyfunction]
pub(crate) fn pauli_compatibility_matrix(
    py: Python<'_>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    mode: u8,
    max_entries: usize,
) -> PyResult<Vec<bool>> {
    let grouping_mode = parse_grouping_mode(mode)?;
    py.allow_threads(|| {
        let words = structures
            .iter()
            .map(|structure| PauliWord::from_codes(nqubits, structure))
            .collect::<Result<Vec<_>, _>>()?;
        compatibility_matrix(&words, grouping_mode, max_entries)
    })
    .map_err(map_error)
}

#[pyfunction]
pub(crate) fn pauli_incompatibility_edges(
    py: Python<'_>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    mode: u8,
    max_edges: usize,
) -> PyResult<Vec<(usize, usize)>> {
    let grouping_mode = parse_grouping_mode(mode)?;
    py.allow_threads(|| {
        let words = structures
            .iter()
            .map(|structure| PauliWord::from_codes(nqubits, structure))
            .collect::<Result<Vec<_>, _>>()?;
        incompatibility_edges(&words, grouping_mode, max_edges)
    })
    .map_err(map_error)
}
