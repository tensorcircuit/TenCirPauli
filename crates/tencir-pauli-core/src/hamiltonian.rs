use std::collections::{BTreeMap, BTreeSet};
use std::mem::size_of;

use rayon::prelude::*;

use crate::error::PauliError;
use crate::operator::{PauliOperator, PauliTerm};
use crate::scalar::{is_exact_zero, Complex64};
use crate::word::packed_word_count;

#[derive(Clone, Copy, Debug)]
struct MatrixTerm {
    x_mask: usize,
    z_mask: usize,
    weighted_phase: Complex64,
}

#[derive(Clone, Copy, Debug, Default)]
struct SparseEntry {
    column: u64,
    value: Complex64,
}

struct SparseEntries {
    entries: Vec<SparseEntry>,
    row_counts: Vec<usize>,
}

#[derive(Debug)]
struct MatrixGroup {
    x_mask: usize,
    terms: Vec<MatrixTerm>,
}

#[derive(Clone, Debug)]
struct MvpGroup {
    x_mask: usize,
    diagonal: Vec<Complex64>,
}

/// Strategy selected for a reusable matrix-free MVP plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MvpStrategy {
    /// Store one precomputed diagonal for every distinct X permutation mask.
    XMaskDiagonal,
    /// Evaluate every canonical Pauli term directly during application.
    TermDirect,
}

/// A reusable, phase-precomputed CPU matrix-free Pauli application plan.
#[derive(Clone, Debug)]
pub struct MvpPlan {
    nqubits: usize,
    term_count: usize,
    terms: Option<Vec<MatrixTerm>>,
    diagonal_groups: Option<Vec<MvpGroup>>,
    strategy: MvpStrategy,
}

impl MvpPlan {
    /// Compile matrix masks and fixed Y phases from a canonical operator.
    pub fn from_operator(operator: &PauliOperator) -> Result<Self, PauliError> {
        let terms = operator
            .terms
            .iter()
            .map(matrix_term)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            nqubits: operator.nqubits,
            term_count: terms.len(),
            terms: Some(terms),
            diagonal_groups: None,
            strategy: MvpStrategy::TermDirect,
        })
    }

    /// Compile a reusable plan with one precomputed diagonal per X mask.
    pub fn from_operator_reusable(
        operator: &PauliOperator,
        max_bytes: u128,
    ) -> Result<Self, PauliError> {
        let mut direct = Self::from_operator(operator)?;
        let dimension = matrix_dimension(operator.nqubits)?;
        let direct_terms = direct.terms.as_ref().expect("direct terms");
        let group_count = direct_terms
            .iter()
            .map(|term| term.x_mask)
            .collect::<BTreeSet<_>>()
            .len();
        let diagonal_bytes = group_count
            .checked_mul(dimension)
            .and_then(|count| count.checked_mul(size_of::<Complex64>()))
            .ok_or(PauliError::Overflow {
                context: "estimating reusable MVP plan memory",
            })?;
        let term_bytes = direct_terms
            .len()
            .checked_mul(size_of::<MatrixTerm>())
            .ok_or(PauliError::Overflow {
                context: "estimating reusable MVP term memory",
            })?;
        check_allocation(term_bytes as u128, max_bytes)?;
        let construction_bytes = (diagonal_bytes as u128)
            .checked_add((term_bytes as u128).saturating_mul(2))
            .and_then(|bytes| {
                bytes.checked_add(
                    (group_count as u128).saturating_mul(size_of::<MatrixGroup>() as u128),
                )
            })
            .ok_or(PauliError::Overflow {
                context: "estimating reusable MVP construction memory",
            })?;
        if construction_bytes > max_bytes {
            return Err(PauliError::MemoryLimit {
                requested: construction_bytes,
                limit: max_bytes,
            });
        }
        let groups = group_matrix_terms(direct.terms.take().expect("direct terms"));
        let diagonal_groups = groups
            .into_par_iter()
            .map(|group| {
                let mut diagonal = vec![Complex64::default(); dimension];
                for (column, value) in diagonal.iter_mut().enumerate() {
                    for term in &group.terms {
                        let mut contribution = term.weighted_phase;
                        if (term.z_mask & column).count_ones() & 1 != 0 {
                            contribution = -contribution;
                        }
                        *value += contribution;
                    }
                }
                MvpGroup {
                    x_mask: group.x_mask,
                    diagonal,
                }
            })
            .collect();
        Ok(Self {
            nqubits: direct.nqubits,
            term_count: direct.term_count,
            terms: None,
            diagonal_groups: Some(diagonal_groups),
            strategy: MvpStrategy::XMaskDiagonal,
        })
    }

    /// Return the number of qubits in this plan.
    pub fn nqubits(&self) -> usize {
        self.nqubits
    }

    /// Return the number of canonical matrix terms in this plan.
    pub fn term_count(&self) -> usize {
        self.term_count
    }

    /// Return the selected application strategy as a stable label.
    pub fn strategy(&self) -> MvpStrategy {
        self.strategy
    }

    /// Apply the plan without materializing a matrix.
    pub fn apply(
        &self,
        state: &[Complex64],
        max_bytes: u128,
    ) -> Result<Vec<Complex64>, PauliError> {
        let dimension = matrix_dimension(self.nqubits)?;
        if state.len() != dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: dimension,
                actual: state.len(),
            });
        }
        check_allocation(
            dimension as u128 * size_of::<Complex64>() as u128,
            max_bytes,
        )?;
        let mut result = vec![Complex64::default(); dimension];
        self.apply_into(state, &mut result, max_bytes)?;
        Ok(result)
    }

    /// Apply the plan into caller-owned storage without allocating a result.
    pub fn apply_into(
        &self,
        state: &[Complex64],
        result: &mut [Complex64],
        max_bytes: u128,
    ) -> Result<(), PauliError> {
        let dimension = matrix_dimension(self.nqubits)?;
        if state.len() != dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: dimension,
                actual: state.len(),
            });
        }
        if result.len() != dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: dimension,
                actual: result.len(),
            });
        }
        let _ = max_bytes;
        let work = self
            .diagonal_groups
            .as_ref()
            .map_or(self.term_count, Vec::len)
            .saturating_mul(dimension);
        if work >= 1 << 16 {
            result.par_iter_mut().enumerate().for_each(|(row, output)| {
                let mut value = Complex64::default();
                if let Some(groups) = &self.diagonal_groups {
                    for group in groups {
                        value += group.diagonal[row ^ group.x_mask] * state[row ^ group.x_mask];
                    }
                } else {
                    for term in self.terms.as_ref().expect("direct terms") {
                        let column = row ^ term.x_mask;
                        let contribution = term.weighted_phase * state[column];
                        if (term.z_mask & column).count_ones() & 1 == 0 {
                            value += contribution;
                        } else {
                            value -= contribution;
                        }
                    }
                }
                *output = value;
            });
        } else {
            for (row, output) in result.iter_mut().enumerate() {
                let mut value = Complex64::default();
                if let Some(groups) = &self.diagonal_groups {
                    for group in groups {
                        let column = row ^ group.x_mask;
                        value += group.diagonal[column] * state[column];
                    }
                } else {
                    for term in self.terms.as_ref().expect("direct terms") {
                        let column = row ^ term.x_mask;
                        let contribution = term.weighted_phase * state[column];
                        if (term.z_mask & column).count_ones() & 1 == 0 {
                            value += contribution;
                        } else {
                            value -= contribution;
                        }
                    }
                }
                *output = value;
            }
        }
        Ok(())
    }
}

/// Deterministic sparse COO output from the Rust core.
pub struct CooMatrix {
    /// Matrix dimension.
    pub dimension: usize,
    /// Row indices in row-major order.
    pub rows: Vec<u64>,
    /// Column indices in row-major order.
    pub columns: Vec<u64>,
    /// Complex128-compatible values.
    pub values: Vec<Complex64>,
}

/// Deterministic sparse CSR output from the Rust core.
pub struct CsrMatrix {
    /// Matrix dimension.
    pub dimension: usize,
    /// Row pointer array.
    pub indptr: Vec<u64>,
    /// Column indices.
    pub columns: Vec<u64>,
    /// Complex128-compatible values.
    pub values: Vec<Complex64>,
}

/// Pure-array backend MVP plan data.
pub struct BackendMvpPlan {
    /// Qubit count.
    pub nqubits: usize,
    /// Packed words per term.
    pub word_count: usize,
    /// Flat X masks.
    pub x_words: Vec<u64>,
    /// Flat Z masks.
    pub z_words: Vec<u64>,
    /// Coefficients in canonical term order.
    pub coefficients: Vec<Complex64>,
}

impl PauliOperator {
    /// Compile a dense row-major matrix using qubit zero as the MSB.
    pub fn dense_matrix(&self, max_bytes: u128) -> Result<(usize, Vec<Complex64>), PauliError> {
        let dimension = matrix_dimension(self.nqubits)?;
        let entries = dimension
            .checked_mul(dimension)
            .ok_or(PauliError::Overflow {
                context: "estimating dense matrix entries",
            })?;
        check_allocation(entries as u128 * 16, max_bytes)?;
        let terms = self
            .terms
            .iter()
            .map(matrix_term)
            .collect::<Result<Vec<_>, _>>()?;
        let mut matrix = vec![Complex64::default(); entries];
        let work = terms.len().saturating_mul(dimension);
        if work >= 1 << 16 {
            matrix
                .par_chunks_mut(dimension)
                .enumerate()
                .for_each(|(row, output)| {
                    for term in &terms {
                        let column = row ^ term.x_mask;
                        let contribution = term.weighted_phase;
                        if (term.z_mask & column).count_ones() & 1 == 0 {
                            output[column] += contribution;
                        } else {
                            output[column] -= contribution;
                        }
                    }
                });
        } else {
            for (row, output) in matrix.chunks_exact_mut(dimension).enumerate() {
                for term in &terms {
                    let column = row ^ term.x_mask;
                    let contribution = term.weighted_phase;
                    if (term.z_mask & column).count_ones() & 1 == 0 {
                        output[column] += contribution;
                    } else {
                        output[column] -= contribution;
                    }
                }
            }
        }
        Ok((dimension, matrix))
    }

    /// Compile deterministic, duplicate-aggregated COO entries.
    pub fn coo_matrix(&self, max_bytes: u128) -> Result<CooMatrix, PauliError> {
        let (dimension, groups, upper_bound) = self.matrix_groups()?;
        let output_bytes = (2 * size_of::<u64>() + size_of::<Complex64>()) as u128;
        if groups.iter().all(|group| group.terms.len() == 1) {
            check_allocation(
                (upper_bound as u128)
                    .checked_mul(output_bytes)
                    .ok_or(PauliError::Overflow {
                        context: "estimating direct COO output memory",
                    })?,
                max_bytes,
            )?;
            let (rows, columns, values) = direct_coo_arrays(&groups, upper_bound);
            return Ok(CooMatrix {
                dimension,
                rows,
                columns,
                values,
            });
        }
        let candidate_bytes = size_of::<SparseEntry>() as u128;
        let row_count_bytes = (dimension as u128)
            .checked_mul(size_of::<usize>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating COO row-count memory",
            })?;
        check_allocation(
            (upper_bound as u128)
                .checked_mul(candidate_bytes + output_bytes)
                .and_then(|bytes| bytes.checked_add(row_count_bytes))
                .ok_or(PauliError::Overflow {
                    context: "estimating COO working memory",
                })?,
            max_bytes,
        )?;
        let sparse = sparse_entries(dimension, &groups, upper_bound);
        let entry_count = sparse.entries.len();
        let mut rows = Vec::with_capacity(entry_count);
        let mut columns = Vec::with_capacity(entry_count);
        let mut values = Vec::with_capacity(entry_count);
        let mut offset = 0;
        for (row, &count) in sparse.row_counts.iter().enumerate() {
            for _ in 0..count {
                rows.push(row as u64);
            }
            for entry in &sparse.entries[offset..offset + count] {
                columns.push(entry.column);
                values.push(entry.value);
            }
            offset += count;
        }
        Ok(CooMatrix {
            dimension,
            rows,
            columns,
            values,
        })
    }

    /// Compile deterministic CSR from the canonical COO stream.
    pub fn csr_matrix(&self, max_bytes: u128) -> Result<CsrMatrix, PauliError> {
        let (dimension, groups, upper_bound) = self.matrix_groups()?;
        let output_bytes = (size_of::<u64>() as u128)
            .checked_mul((dimension + 1) as u128)
            .and_then(|indptr_bytes| {
                (size_of::<u64>() as u128 + size_of::<Complex64>() as u128)
                    .checked_mul(upper_bound as u128)
                    .and_then(|entries_bytes| indptr_bytes.checked_add(entries_bytes))
            })
            .ok_or(PauliError::Overflow {
                context: "estimating CSR working memory",
            })?;
        if groups.iter().all(|group| group.terms.len() == 1) {
            check_allocation(output_bytes, max_bytes)?;
            let (indptr, columns, values) = direct_csr_arrays(dimension, &groups, upper_bound);
            return Ok(CsrMatrix {
                dimension,
                indptr,
                columns,
                values,
            });
        }
        let candidate_bytes = size_of::<SparseEntry>() as u128;
        let row_count_bytes = (dimension as u128)
            .checked_mul(size_of::<usize>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating CSR row-count memory",
            })?;
        check_allocation(
            (upper_bound as u128)
                .checked_mul(candidate_bytes)
                .and_then(|candidate| candidate.checked_add(output_bytes))
                .and_then(|bytes| bytes.checked_add(row_count_bytes))
                .ok_or(PauliError::Overflow {
                    context: "estimating CSR working memory",
                })?,
            max_bytes,
        )?;
        let sparse = sparse_entries(dimension, &groups, upper_bound);
        let mut indptr = vec![0_u64; dimension + 1];
        for (row, &count) in sparse.row_counts.iter().enumerate() {
            indptr[row + 1] = count as u64;
        }
        for row in 1..indptr.len() {
            indptr[row] += indptr[row - 1];
        }
        let mut columns = Vec::with_capacity(sparse.entries.len());
        let mut values = Vec::with_capacity(sparse.entries.len());
        for entry in sparse.entries {
            columns.push(entry.column);
            values.push(entry.value);
        }
        Ok(CsrMatrix {
            dimension,
            indptr,
            columns,
            values,
        })
    }

    /// Apply the operator without materializing a matrix.
    pub fn mvp(&self, state: &[Complex64], max_bytes: u128) -> Result<Vec<Complex64>, PauliError> {
        MvpPlan::from_operator(self)?.apply(state, max_bytes)
    }

    /// Apply a one-shot matrix-free plan into caller-owned storage.
    pub fn mvp_into(
        &self,
        state: &[Complex64],
        result: &mut [Complex64],
        max_bytes: u128,
    ) -> Result<(), PauliError> {
        MvpPlan::from_operator(self)?.apply_into(state, result, max_bytes)
    }

    /// Compile a reusable CPU matrix-free application plan.
    pub fn mvp_plan(&self, max_bytes: u128) -> Result<MvpPlan, PauliError> {
        MvpPlan::from_operator_reusable(self, max_bytes)
    }

    /// Return a versioned pure-array backend MVP plan.
    pub fn backend_mvp_plan(&self, max_bytes: u128) -> Result<BackendMvpPlan, PauliError> {
        let word_count = packed_word_count(self.nqubits);
        let bytes = (self.terms.len() as u128)
            .checked_mul((word_count as u128).saturating_mul(16).saturating_add(16))
            .ok_or(PauliError::Overflow {
                context: "estimating backend plan bytes",
            })?;
        check_allocation(bytes, max_bytes)?;
        let mut x_words = Vec::with_capacity(self.terms.len() * word_count);
        let mut z_words = Vec::with_capacity(self.terms.len() * word_count);
        let mut coefficients = Vec::with_capacity(self.terms.len());
        for term in &self.terms {
            x_words.extend_from_slice(term.word.x_words());
            z_words.extend_from_slice(term.word.z_words());
            coefficients.push(term.coefficient);
        }
        Ok(BackendMvpPlan {
            nqubits: self.nqubits,
            word_count,
            x_words,
            z_words,
            coefficients,
        })
    }

    fn matrix_groups(&self) -> Result<(usize, Vec<MatrixGroup>, usize), PauliError> {
        let dimension = matrix_dimension(self.nqubits)?;
        let terms = self
            .terms
            .iter()
            .map(matrix_term)
            .collect::<Result<Vec<_>, _>>()?;
        let groups = group_matrix_terms(terms);
        let upper_bound = groups
            .len()
            .checked_mul(dimension)
            .ok_or(PauliError::Overflow {
                context: "estimating sparse matrix entries",
            })?;
        Ok((dimension, groups, upper_bound))
    }
}

fn matrix_dimension(nqubits: usize) -> Result<usize, PauliError> {
    if nqubits >= usize::BITS as usize {
        return Err(PauliError::Overflow {
            context: "computing matrix dimension",
        });
    }
    1_usize
        .checked_shl(nqubits as u32)
        .ok_or(PauliError::Overflow {
            context: "computing matrix dimension",
        })
}

fn check_allocation(requested: u128, limit: u128) -> Result<(), PauliError> {
    if requested > limit {
        return Err(PauliError::MemoryLimit { requested, limit });
    }
    Ok(())
}

fn matrix_term(term: &PauliTerm) -> Result<MatrixTerm, PauliError> {
    let word = &term.word;
    if word.nqubits >= usize::BITS as usize {
        return Err(PauliError::Overflow {
            context: "converting matrix X mask",
        });
    }
    let mut x_mask = 0_usize;
    let mut z_mask = 0_usize;
    let mut y_count = 0_u32;
    for qubit in 0..word.nqubits {
        let packed_mask = 1_u64 << (qubit % 64);
        let x = word.x_words[qubit / 64] & packed_mask != 0;
        let z = word.z_words[qubit / 64] & packed_mask != 0;
        let matrix_mask = 1_usize << (word.nqubits - 1 - qubit);
        if x {
            x_mask |= matrix_mask;
        }
        if z {
            z_mask |= matrix_mask;
        }
        if x && z {
            y_count += 1;
        }
    }
    let y_phase = match y_count % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    };
    Ok(MatrixTerm {
        x_mask,
        z_mask,
        weighted_phase: term.coefficient * y_phase,
    })
}

fn group_matrix_terms(terms: Vec<MatrixTerm>) -> Vec<MatrixGroup> {
    let mut grouped = BTreeMap::<usize, Vec<MatrixTerm>>::new();
    for term in terms {
        grouped.entry(term.x_mask).or_default().push(term);
    }
    grouped
        .into_iter()
        .map(|(x_mask, terms)| MatrixGroup { x_mask, terms })
        .collect()
}

fn sparse_entries(dimension: usize, groups: &[MatrixGroup], upper_bound: usize) -> SparseEntries {
    let width = groups.len();
    let mut entries = vec![SparseEntry::default(); upper_bound];
    let mut row_counts = vec![0_usize; dimension];
    if width == 0 {
        return SparseEntries {
            entries,
            row_counts,
        };
    }

    if upper_bound >= 1 << 18 {
        entries
            .par_chunks_mut(width)
            .zip(row_counts.par_iter_mut())
            .enumerate()
            .for_each(|(row, (output, count))| {
                *count = fill_sparse_row(row, groups, output);
            });
    } else {
        for (row, (output, count)) in entries
            .chunks_exact_mut(width)
            .zip(row_counts.iter_mut())
            .enumerate()
        {
            *count = fill_sparse_row(row, groups, output);
        }
    }

    let mut write = 0;
    for (row, &count) in row_counts.iter().enumerate() {
        let start = row * width;
        if count != 0 {
            entries.copy_within(start..start + count, write);
            write += count;
        }
    }
    entries.truncate(write);
    SparseEntries {
        entries,
        row_counts,
    }
}

fn direct_coo_arrays(
    groups: &[MatrixGroup],
    upper_bound: usize,
) -> (Vec<u64>, Vec<u64>, Vec<Complex64>) {
    let width = groups.len();
    if width == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let mut rows = vec![0_u64; upper_bound];
    let mut columns = vec![0_u64; upper_bound];
    let mut values = vec![Complex64::default(); upper_bound];
    if upper_bound >= 1 << 18 {
        rows.par_chunks_mut(width)
            .zip(columns.par_chunks_mut(width))
            .zip(values.par_chunks_mut(width))
            .enumerate()
            .for_each_init(
                || Vec::with_capacity(width),
                |scratch, (row, ((row_output, column_output), value_output))| {
                    scratch.clear();
                    scratch.resize(width, SparseEntry::default());
                    fill_sparse_row(row, groups, scratch);
                    row_output.fill(row as u64);
                    for (index, entry) in scratch.iter().enumerate() {
                        column_output[index] = entry.column;
                        value_output[index] = entry.value;
                    }
                },
            );
    } else {
        let mut scratch = vec![SparseEntry::default(); width];
        for (row, ((row_output, column_output), value_output)) in rows
            .chunks_exact_mut(width)
            .zip(columns.chunks_exact_mut(width))
            .zip(values.chunks_exact_mut(width))
            .enumerate()
        {
            fill_sparse_row(row, groups, &mut scratch);
            row_output.fill(row as u64);
            for (index, entry) in scratch.iter().enumerate() {
                column_output[index] = entry.column;
                value_output[index] = entry.value;
            }
        }
    }
    (rows, columns, values)
}

fn direct_csr_arrays(
    dimension: usize,
    groups: &[MatrixGroup],
    upper_bound: usize,
) -> (Vec<u64>, Vec<u64>, Vec<Complex64>) {
    let width = groups.len();
    let mut indptr = vec![0_u64; dimension + 1];
    if width == 0 {
        return (indptr, Vec::new(), Vec::new());
    }
    for (row, pointer) in indptr.iter_mut().enumerate() {
        *pointer = (row * width) as u64;
    }
    let mut columns = vec![0_u64; upper_bound];
    let mut values = vec![Complex64::default(); upper_bound];
    if upper_bound >= 1 << 18 {
        columns
            .par_chunks_mut(width)
            .zip(values.par_chunks_mut(width))
            .enumerate()
            .for_each_init(
                || Vec::with_capacity(width),
                |scratch, (row, (column_output, value_output))| {
                    scratch.clear();
                    scratch.resize(width, SparseEntry::default());
                    fill_sparse_row(row, groups, scratch);
                    for (index, entry) in scratch.iter().enumerate() {
                        column_output[index] = entry.column;
                        value_output[index] = entry.value;
                    }
                },
            );
    } else {
        let mut scratch = vec![SparseEntry::default(); width];
        for (row, (column_output, value_output)) in columns
            .chunks_exact_mut(width)
            .zip(values.chunks_exact_mut(width))
            .enumerate()
        {
            fill_sparse_row(row, groups, &mut scratch);
            for (index, entry) in scratch.iter().enumerate() {
                column_output[index] = entry.column;
                value_output[index] = entry.value;
            }
        }
    }
    (indptr, columns, values)
}

fn fill_sparse_row(row: usize, groups: &[MatrixGroup], output: &mut [SparseEntry]) -> usize {
    let mut count = 0;
    for group in groups {
        let column = row ^ group.x_mask;
        let mut value = Complex64::default();
        for term in &group.terms {
            let mut contribution = term.weighted_phase;
            if (term.z_mask & column).count_ones() & 1 != 0 {
                contribution = -contribution;
            }
            value += contribution;
        }
        if !is_exact_zero(value) {
            output[count] = SparseEntry {
                column: column as u64,
                value,
            };
            count += 1;
        }
    }
    sort_sparse_row(&mut output[..count]);
    count
}

fn sort_sparse_row(entries: &mut [SparseEntry]) {
    match entries.len() {
        0 | 1 => {}
        2 => {
            if entries[0].column > entries[1].column {
                entries.swap(0, 1);
            }
        }
        3 => {
            if entries[0].column > entries[1].column {
                entries.swap(0, 1);
            }
            if entries[1].column > entries[2].column {
                entries.swap(1, 2);
            }
            if entries[0].column > entries[1].column {
                entries.swap(0, 1);
            }
        }
        _ => entries.sort_unstable_by_key(|entry| entry.column),
    }
}
