use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tencir_pauli_core::{
    compatibility_matrix, group_words_bounded, incompatibility_edges, GroupingAlgorithm,
    GroupingMode, PauliWord,
};

use crate::convert::map_error;
use crate::operator::NativePauliOperatorHandle;

type GroupingOutput = (Vec<Vec<usize>>, Vec<Vec<u8>>, Vec<Vec<Vec<usize>>>);

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

#[pyfunction]
/// Group an already canonical native operator and return only public grouping
/// metadata. The operator words never cross into Python for reconstruction.
pub(crate) fn pauli_group_handle(
    py: Python<'_>,
    operator: &NativePauliOperatorHandle,
    mode: u8,
    algorithm: u8,
    max_entries: usize,
) -> PyResult<GroupingOutput> {
    let grouping_mode = parse_grouping_mode(mode)?;
    let grouping_algorithm = match algorithm {
        0 => GroupingAlgorithm::LargestFirst,
        1 => GroupingAlgorithm::Dsatur,
        _ => return Err(PyValueError::new_err("grouping algorithm must be 0 or 1")),
    };
    py.allow_threads(|| {
        let words = operator
            .core()
            .terms()
            .iter()
            .map(|term| term.word.clone())
            .collect::<Vec<_>>();
        let groups = group_words_bounded(&words, grouping_mode, grouping_algorithm, max_entries)?;
        if grouping_mode == GroupingMode::General {
            return Ok((groups, Vec::new(), Vec::new()));
        }
        let nqubits = operator.core().nqubits();
        let mut bases = Vec::with_capacity(groups.len());
        let mut supports = Vec::with_capacity(groups.len());
        for group in &groups {
            let mut basis = vec![0_u8; nqubits];
            let mut group_supports = Vec::with_capacity(group.len());
            for &term_index in group {
                let codes = words[term_index].codes();
                let mut support = Vec::new();
                for (qubit, code) in codes.into_iter().enumerate() {
                    if code != 0 {
                        support.push(qubit);
                        if basis[qubit] == 0 {
                            basis[qubit] = code;
                        }
                    }
                }
                group_supports.push(support);
            }
            bases.push(basis);
            supports.push(group_supports);
        }
        Ok((groups, bases, supports))
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
pub(crate) fn pauli_compatibility_matrix_handle(
    py: Python<'_>,
    operator: &NativePauliOperatorHandle,
    mode: u8,
    max_entries: usize,
) -> PyResult<Vec<bool>> {
    let grouping_mode = parse_grouping_mode(mode)?;
    py.allow_threads(|| {
        let words = operator
            .core()
            .terms()
            .iter()
            .map(|term| term.word.clone())
            .collect::<Vec<_>>();
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

#[pyfunction]
pub(crate) fn pauli_incompatibility_edges_handle(
    py: Python<'_>,
    operator: &NativePauliOperatorHandle,
    mode: u8,
    max_edges: usize,
) -> PyResult<Vec<(usize, usize)>> {
    let grouping_mode = parse_grouping_mode(mode)?;
    py.allow_threads(|| {
        let words = operator
            .core()
            .terms()
            .iter()
            .map(|term| term.word.clone())
            .collect::<Vec<_>>();
        incompatibility_edges(&words, grouping_mode, max_edges)
    })
    .map_err(map_error)
}
