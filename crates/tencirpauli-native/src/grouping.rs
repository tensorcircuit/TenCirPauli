use numpy::{PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tencir_pauli_core::{
    compatibility_matrix, group_words_bounded, incompatibility_edges, GroupingAlgorithm,
    GroupingMode,
};

use crate::convert::map_error;
use crate::operator::NativePauliOperatorHandle;

type GroupingOutput = (Vec<Vec<usize>>, Vec<Vec<u8>>, Vec<Vec<Vec<usize>>>);
type QwcGroupingOutput = (Vec<Vec<usize>>, Vec<Vec<u8>>, NativeQwcGroupingHandle);

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeQwcGroupingHandle {
    nqubits: usize,
    masks: Vec<Vec<Vec<u64>>>,
}

fn reconstruct_values(
    nqubits: usize,
    masks: &[Vec<Vec<u64>>],
    group_index: usize,
    values: &[i8],
    shots: usize,
) -> Result<Vec<i8>, &'static str> {
    let group_masks = masks
        .get(group_index)
        .ok_or("group index is out of range")?;
    for &value in values {
        if value != 0 && value != 1 {
            return Err("bitstrings must contain only 0 and 1");
        }
    }
    let group_size = group_masks.len();
    let output_len = shots
        .checked_mul(group_size)
        .ok_or("reconstruction output size overflow")?;
    let mut output = vec![0_i8; output_len];
    for shot in 0..shots {
        for (column, support_words) in group_masks.iter().enumerate() {
            let mut parity = 0_u32;
            for (word_index, &word) in support_words.iter().enumerate() {
                let mut remaining = word;
                while remaining != 0 {
                    let bit = remaining.trailing_zeros() as usize;
                    let qubit = word_index * 64 + bit;
                    parity ^= (values[shot * nqubits + qubit] != 0) as u32;
                    remaining &= remaining - 1;
                }
            }
            output[shot * group_size + column] = if parity == 0 { 1 } else { -1 };
        }
    }
    Ok(output)
}

#[pymethods]
impl NativeQwcGroupingHandle {
    fn reconstruct<'py>(
        &self,
        py: Python<'py>,
        group_index: usize,
        bitstrings: PyReadonlyArray2<'py, i8>,
    ) -> PyResult<(usize, usize, Bound<'py, PyArray1<i8>>)> {
        if group_index >= self.masks.len() {
            return Err(PyValueError::new_err("group index is out of range"));
        }
        let shape = bitstrings.shape();
        if shape.len() != 2 || shape[1] != self.nqubits {
            return Err(PyValueError::new_err(format!(
                "bitstrings must have shape (shots, {}), got {:?}",
                self.nqubits, shape
            )));
        }
        let values = bitstrings
            .as_slice()
            .map_err(|_| PyValueError::new_err("bitstrings must be C-contiguous"))?;
        let shots = shape[0];
        let group_size = self.masks[group_index].len();
        let output = py
            .allow_threads(|| {
                reconstruct_values(self.nqubits, &self.masks, group_index, values, shots)
            })
            .map_err(PyValueError::new_err)?;
        Ok((shots, group_size, PyArray1::from_vec(py, output)))
    }
}

#[pyfunction]
pub(crate) fn pauli_qwc_group_handle(
    py: Python<'_>,
    operator: &NativePauliOperatorHandle,
    algorithm: u8,
    max_entries: usize,
) -> PyResult<QwcGroupingOutput> {
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
        let groups = group_words_bounded(
            &words,
            GroupingMode::QubitWise,
            grouping_algorithm,
            max_entries,
        )?;
        let nqubits = operator.core().nqubits();
        let word_count = nqubits.div_ceil(64);
        let mut bases = Vec::with_capacity(groups.len());
        let mut masks = Vec::with_capacity(groups.len());
        for group in &groups {
            let mut basis = vec![0_u8; nqubits];
            let mut group_masks = Vec::with_capacity(group.len());
            for &term_index in group {
                let codes = words[term_index].codes();
                let mut support_words = vec![0_u64; word_count];
                for (qubit, code) in codes.into_iter().enumerate() {
                    if code != 0 {
                        support_words[qubit / 64] |= 1_u64 << (qubit % 64);
                        if basis[qubit] == 0 {
                            basis[qubit] = code;
                        }
                    }
                }
                group_masks.push(support_words);
            }
            bases.push(basis);
            masks.push(group_masks);
        }
        let handle = NativeQwcGroupingHandle { nqubits, masks };
        Ok((groups, bases, handle))
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
