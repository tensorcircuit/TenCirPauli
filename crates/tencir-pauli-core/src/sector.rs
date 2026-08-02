//! Fixed-Hamming-weight U(1) sectors and restricted Pauli operators.

use std::{mem::size_of, sync::Arc};

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::error::PauliError;
use crate::operator::{PauliOperator, PauliTerm};
use crate::scalar::{is_exact_zero, Complex64};
use crate::word::packed_word_count;

const U1_PARALLEL_TRANSITION_THRESHOLD: usize = 1 << 14;

/// A fixed-particle-number computational-basis sector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct U1Sector {
    nqubits: usize,
    particle_number: usize,
    dimension: u64,
    active_number: usize,
    complement: bool,
    /// Row-major `C(row, active_number)` values for `0 <= row <= nqubits`.
    choose: Arc<[u64]>,
}

/// A packed, row-major materialization of a U(1) basis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedU1Basis {
    pub dimension: u64,
    pub word_count: usize,
    pub words: Vec<u64>,
}

/// A matrix-free plan over a fixed-Hamming-weight basis.
#[derive(Clone, Debug, PartialEq)]
pub struct U1MvpPlan {
    sector: U1Sector,
    indptr: Arc<[usize]>,
    columns: Arc<[usize]>,
    values: Arc<[Complex64]>,
}

/// A restricted operator whose setup has already validated sector preservation.
#[derive(Clone, Debug, PartialEq)]
pub struct U1RestrictedOperator {
    plan: U1MvpPlan,
}

/// CSR arrays for a restricted operator.
#[derive(Clone, Debug, PartialEq)]
pub struct U1CsrMatrix {
    pub dimension: usize,
    pub indptr: Vec<u64>,
    pub columns: Vec<u64>,
    pub values: Vec<Complex64>,
}

/// COO arrays for a restricted operator.
#[derive(Clone, Debug, PartialEq)]
pub struct U1CooMatrix {
    pub dimension: usize,
    pub rows: Vec<u64>,
    pub columns: Vec<u64>,
    pub values: Vec<Complex64>,
}

impl U1Sector {
    pub fn new(nqubits: usize, particle_number: usize) -> Result<Self, PauliError> {
        if particle_number > nqubits {
            return Err(PauliError::InvalidSector {
                context: "particle_number must be between 0 and nqubits",
            });
        }
        let dimension = choose_u64(nqubits, particle_number)?;
        usize::try_from(dimension).map_err(|_| PauliError::Overflow {
            context: "converting U1 sector dimension to a native index",
        })?;
        let active_number = particle_number.min(nqubits - particle_number);
        let choose = build_choose_table(nqubits, active_number)?;
        Ok(Self {
            nqubits,
            particle_number,
            dimension,
            active_number,
            complement: particle_number > nqubits - particle_number,
            choose,
        })
    }

    pub fn nqubits(&self) -> usize {
        self.nqubits
    }

    pub fn particle_number(&self) -> usize {
        self.particle_number
    }

    /// Return the restricted dimension as a checked native index.
    pub fn dimension(&self) -> Result<usize, PauliError> {
        usize::try_from(self.dimension).map_err(|_| PauliError::Overflow {
            context: "converting U1 sector dimension to a native index",
        })
    }

    /// Return the restricted dimension before converting it to `usize`.
    pub fn dimension_u64(&self) -> u64 {
        self.dimension
    }

    /// Return the number of packed little-endian-by-qubit limbs.
    pub fn word_count(&self) -> usize {
        packed_word_count(self.nqubits)
    }

    /// Rank a TensorCircuit-order computational basis integer.
    pub fn rank(&self, bitstring: usize) -> Result<usize, PauliError> {
        ensure_integer_width(self.nqubits)?;
        let mut words = vec![0_u64; self.word_count()];
        for position in 0..self.nqubits {
            if (bitstring >> (self.nqubits - position - 1)) & 1 != 0 {
                words[position / 64] |= 1_u64 << (position % 64);
            }
        }
        usize::try_from(self.rank_words(&words)?).map_err(|_| PauliError::Overflow {
            context: "converting U1 basis rank to a native index",
        })
    }

    /// Rank a packed occupation word using qubit `q` at limb `q / 64`, bit `q % 64`.
    pub fn rank_words(&self, words: &[u64]) -> Result<u64, PauliError> {
        self.validate_words(words)?;
        let weight = words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum::<usize>();
        if weight != self.particle_number {
            return Err(PauliError::InvalidIndex {
                context: "packed basis word has the wrong Hamming weight",
            });
        }
        let active_rank = self.rank_active_words(words)?;
        if self.complement {
            self.dimension
                .checked_sub(1)
                .and_then(|last| last.checked_sub(active_rank))
                .ok_or(PauliError::Overflow {
                    context: "ranking complemented U1 basis state",
                })
        } else {
            Ok(active_rank)
        }
    }

    /// Unrank in ascending TensorCircuit computational-basis integer order.
    pub fn unrank(&self, index: usize) -> Result<usize, PauliError> {
        ensure_integer_width(self.nqubits)?;
        let mut words = vec![0_u64; self.word_count()];
        self.unrank_into(index as u64, &mut words)?;
        let mut value = 0_usize;
        for position in 0..self.nqubits {
            if words[position / 64] & (1_u64 << (position % 64)) != 0 {
                value |= 1_usize << (self.nqubits - position - 1);
            }
        }
        Ok(value)
    }

    /// Unrank into caller-owned packed little-endian-by-qubit storage.
    pub fn unrank_into(&self, index: u64, output: &mut [u64]) -> Result<(), PauliError> {
        let dimension = self.dimension;
        if index >= dimension {
            return Err(PauliError::InvalidIndex {
                context: "restricted basis index is out of range",
            });
        }
        if output.len() != self.word_count() {
            return Err(PauliError::InvalidWordLength {
                expected: self.word_count(),
                actual: output.len(),
            });
        }
        output.fill(0);
        let mut active_index = if self.complement {
            dimension
                .checked_sub(1)
                .and_then(|last| last.checked_sub(index))
                .ok_or(PauliError::Overflow {
                    context: "unranking complemented U1 basis state",
                })?
        } else {
            index
        };
        let mut remaining = self.active_number;
        for position in 0..self.nqubits {
            let remaining_sites = self.nqubits - position - 1;
            let zero_count = self.choose_value(remaining_sites, remaining)?;
            if active_index >= zero_count {
                active_index -= zero_count;
                if remaining == 0 {
                    return Err(PauliError::InvalidIndex {
                        context: "invalid combinatorial sector index",
                    });
                }
                output[position / 64] |= 1_u64 << (position % 64);
                remaining -= 1;
            }
        }
        if remaining != 0 || active_index != 0 {
            return Err(PauliError::InvalidIndex {
                context: "invalid combinatorial sector index",
            });
        }
        if self.complement {
            let output_len = output.len();
            for (index, word) in output.iter_mut().enumerate() {
                *word = !*word;
                if index + 1 == output_len {
                    *word &= tail_mask(self.nqubits);
                }
            }
        }
        Ok(())
    }

    /// Materialize the packed basis without constructing a full `2**n` basis.
    pub fn basis_words_packed(&self, max_bytes: u128) -> Result<PackedU1Basis, PauliError> {
        let dimension = self.dimension()?;
        let word_count = self.word_count();
        let count = dimension
            .checked_mul(word_count)
            .ok_or(PauliError::Overflow {
                context: "sizing packed U1 basis output",
            })?;
        let bytes = (count as u128)
            .checked_mul(size_of::<u64>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating packed U1 basis memory",
            })?;
        check_allocation(bytes, max_bytes)?;
        let mut words = vec![0_u64; count];
        for index in 0..dimension {
            self.unrank_into(index as u64, &mut words[index * word_count..][..word_count])?;
        }
        Ok(PackedU1Basis {
            dimension: self.dimension,
            word_count,
            words,
        })
    }

    /// Preserve the original narrow convenience API.
    pub fn basis_words(&self, max_bytes: u128) -> Result<Vec<usize>, PauliError> {
        ensure_integer_width(self.nqubits)?;
        let dimension = self.dimension()?;
        let bytes = (dimension as u128)
            .checked_mul(size_of::<usize>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating U1 basis memory",
            })?;
        check_allocation(bytes, max_bytes)?;
        (0..dimension).map(|index| self.unrank(index)).collect()
    }

    fn validate_words(&self, words: &[u64]) -> Result<(), PauliError> {
        if words.len() != self.word_count() {
            return Err(PauliError::InvalidWordLength {
                expected: self.word_count(),
                actual: words.len(),
            });
        }
        if let Some(last) = words.last() {
            let mask = tail_mask(self.nqubits);
            if last & !mask != 0 {
                return Err(PauliError::InvalidIndex {
                    context: "packed basis word has nonzero padding bits",
                });
            }
        }
        Ok(())
    }

    fn choose_value(&self, n: usize, k: usize) -> Result<u64, PauliError> {
        if k > n {
            return Ok(0);
        }
        let normalized = k.min(n - k);
        if normalized > self.active_number || n > self.nqubits {
            return Err(PauliError::Overflow {
                context: "looking up a U1 combinatorial coefficient",
            });
        }
        Ok(self.choose[n * (self.active_number + 1) + normalized])
    }

    fn rank_active_words(&self, words: &[u64]) -> Result<u64, PauliError> {
        let select_zero = self.complement;
        let mut rank = 0_u64;
        let mut selected = 0_usize;
        for (word_index, &word) in words.iter().enumerate() {
            let tail = if word_index + 1 == words.len() {
                tail_mask(self.nqubits)
            } else {
                u64::MAX
            };
            let mut active_bits = if select_zero { !word } else { word } & tail;
            while active_bits != 0 {
                let bit = active_bits.trailing_zeros() as usize;
                let position = word_index * 64 + bit;
                let remaining = self.nqubits - position - 1;
                let needed = self.active_number - selected;
                rank = rank
                    .checked_add(self.choose_value(remaining, needed)?)
                    .ok_or(PauliError::Overflow {
                        context: "ranking U1 basis state",
                    })?;
                selected += 1;
                active_bits &= active_bits - 1;
            }
        }
        if selected != self.active_number {
            return Err(PauliError::InvalidIndex {
                context: "packed basis word has the wrong active weight",
            });
        }
        Ok(rank)
    }
}

impl U1RestrictedOperator {
    pub fn new(
        operator: &PauliOperator,
        sector: U1Sector,
        max_bytes: u128,
    ) -> Result<Self, PauliError> {
        if operator.nqubits() != sector.nqubits {
            return Err(PauliError::IncompatibleQubitCounts {
                left: operator.nqubits(),
                right: sector.nqubits,
            });
        }
        let dimension = sector.dimension()?;
        let terms = compile_terms(operator)?;
        let upper_bound = transition_upper_bound(&sector, &terms)?;
        let estimated = estimate_plan_bytes(&sector, &terms, dimension, upper_bound)?;
        check_allocation(estimated, max_bytes)?;

        let mut row_counts = vec![0_usize; dimension];
        let mut source = vec![0_u64; sector.word_count()];
        let mut destination = vec![0_u64; sector.word_count()];
        let mut aggregate = Vec::with_capacity(terms.groups.len());
        for source_index in 0..dimension {
            sector.unrank_into(source_index as u64, &mut source)?;
            aggregate_source(
                source_index as u64,
                &source,
                &mut destination,
                &terms,
                &sector,
                &mut aggregate,
            )?;
            for &(destination, _) in &aggregate {
                row_counts[destination] =
                    row_counts[destination]
                        .checked_add(1)
                        .ok_or(PauliError::Overflow {
                            context: "counting U1 restricted transitions",
                        })?;
            }
        }

        let mut indptr = vec![0_usize; dimension + 1];
        for row in 0..dimension {
            indptr[row + 1] =
                indptr[row]
                    .checked_add(row_counts[row])
                    .ok_or(PauliError::Overflow {
                        context: "sizing U1 restricted transitions",
                    })?;
        }
        let entry_count = indptr[dimension];
        let mut columns = vec![0_usize; entry_count];
        let mut values = vec![Complex64::default(); entry_count];
        let mut next = indptr[..dimension].to_vec();
        for source_index in 0..dimension {
            sector.unrank_into(source_index as u64, &mut source)?;
            aggregate_source(
                source_index as u64,
                &source,
                &mut destination,
                &terms,
                &sector,
                &mut aggregate,
            )?;
            for &(destination, value) in &aggregate {
                let position = next[destination];
                columns[position] = source_index;
                values[position] = value;
                next[destination] += 1;
            }
        }

        Ok(Self {
            plan: U1MvpPlan {
                sector,
                indptr: Arc::from(indptr.into_boxed_slice()),
                columns: Arc::from(columns.into_boxed_slice()),
                values: Arc::from(values.into_boxed_slice()),
            },
        })
    }

    pub fn sector(&self) -> U1Sector {
        self.plan.sector.clone()
    }

    pub fn dimension(&self) -> usize {
        self.plan.dimension()
    }

    pub fn apply(
        &self,
        state: &[Complex64],
        max_bytes: u128,
    ) -> Result<Vec<Complex64>, PauliError> {
        self.plan.apply(state, max_bytes)
    }

    pub fn mvp_plan(&self, max_bytes: u128) -> Result<U1MvpPlan, PauliError> {
        self.plan.clone_with_budget(max_bytes)
    }

    pub fn dense(&self, max_bytes: u128) -> Result<(usize, Vec<Complex64>), PauliError> {
        let dimension = self.dimension();
        let entries = dimension
            .checked_mul(dimension)
            .ok_or(PauliError::Overflow {
                context: "estimating U1 dense matrix entries",
            })?;
        let bytes = (entries as u128)
            .checked_mul(size_of::<Complex64>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating U1 dense matrix memory",
            })?;
        check_allocation(bytes, max_bytes)?;
        let mut values = vec![Complex64::default(); entries];
        for destination in 0..dimension {
            let start = self.plan.indptr[destination];
            let stop = self.plan.indptr[destination + 1];
            for index in start..stop {
                values[destination * dimension + self.plan.columns[index]] +=
                    self.plan.values[index];
            }
        }
        Ok((dimension, values))
    }

    pub fn coo(&self, max_bytes: u128) -> Result<U1CooMatrix, PauliError> {
        let dimension = self.dimension();
        let entry_count = self.plan.values.len();
        let output_bytes = (entry_count as u128)
            .checked_mul((2 * size_of::<u64>() + size_of::<Complex64>()) as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating U1 COO output memory",
            })?;
        check_allocation(output_bytes, max_bytes)?;
        let mut rows = Vec::with_capacity(entry_count);
        let mut columns = Vec::with_capacity(entry_count);
        let mut values = Vec::with_capacity(entry_count);
        for destination in 0..dimension {
            let start = self.plan.indptr[destination];
            let stop = self.plan.indptr[destination + 1];
            for index in start..stop {
                rows.push(
                    u64::try_from(destination).map_err(|_| PauliError::Overflow {
                        context: "converting U1 COO row index",
                    })?,
                );
                columns.push(u64::try_from(self.plan.columns[index]).map_err(|_| {
                    PauliError::Overflow {
                        context: "converting U1 COO column index",
                    }
                })?);
                values.push(self.plan.values[index]);
            }
        }
        Ok(U1CooMatrix {
            dimension,
            rows,
            columns,
            values,
        })
    }

    pub fn csr(&self, max_bytes: u128) -> Result<U1CsrMatrix, PauliError> {
        self.plan.csr(max_bytes)
    }
}

impl U1MvpPlan {
    pub fn sector(&self) -> U1Sector {
        self.sector.clone()
    }

    pub fn dimension(&self) -> usize {
        self.indptr.len() - 1
    }

    pub fn apply(
        &self,
        state: &[Complex64],
        max_bytes: u128,
    ) -> Result<Vec<Complex64>, PauliError> {
        let dimension = self.dimension();
        if state.len() != dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: dimension,
                actual: state.len(),
            });
        }
        let bytes = (dimension as u128)
            .checked_mul(size_of::<Complex64>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating U1 MVP output memory",
            })?;
        check_allocation(bytes, max_bytes)?;
        let mut output = vec![Complex64::default(); dimension];
        self.apply_into(state, &mut output)?;
        Ok(output)
    }

    pub fn apply_into(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), PauliError> {
        if state.len() != self.dimension() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension(),
                actual: state.len(),
            });
        }
        if output.len() != self.dimension() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension(),
                actual: output.len(),
            });
        }
        output.fill(Complex64::default());
        if self.values.len() < U1_PARALLEL_TRANSITION_THRESHOLD {
            for (destination, result) in output.iter_mut().enumerate() {
                let mut value = Complex64::default();
                for index in self.indptr[destination]..self.indptr[destination + 1] {
                    value += self.values[index] * state[self.columns[index]];
                }
                *result = value;
            }
        } else {
            self.indptr
                .par_windows(2)
                .zip(output.par_iter_mut())
                .for_each(|(window, result)| {
                    let mut value = Complex64::default();
                    for index in window[0]..window[1] {
                        value += self.values[index] * state[self.columns[index]];
                    }
                    *result = value;
                });
        }
        Ok(())
    }

    fn clone_with_budget(&self, max_bytes: u128) -> Result<Self, PauliError> {
        let bytes = (self.indptr.len() as u128)
            .checked_mul(size_of::<usize>() as u128)
            .and_then(|pointers| {
                (self.columns.len() as u128)
                    .checked_mul(size_of::<usize>() as u128)
                    .and_then(|columns| pointers.checked_add(columns))
            })
            .and_then(|bytes| {
                (self.values.len() as u128)
                    .checked_mul(size_of::<Complex64>() as u128)
                    .and_then(|values| bytes.checked_add(values))
            })
            .ok_or(PauliError::Overflow {
                context: "estimating U1 MVP plan memory",
            })?;
        check_allocation(bytes, max_bytes)?;
        Ok(self.clone())
    }

    fn csr(&self, max_bytes: u128) -> Result<U1CsrMatrix, PauliError> {
        let dimension = self.dimension();
        let entries = self.values.len();
        let bytes = (dimension as u128 + 1)
            .checked_mul(size_of::<u64>() as u128)
            .and_then(|pointers| {
                (entries as u128)
                    .checked_mul((size_of::<u64>() + size_of::<Complex64>()) as u128)
                    .and_then(|values| pointers.checked_add(values))
            })
            .ok_or(PauliError::Overflow {
                context: "estimating U1 CSR memory",
            })?;
        check_allocation(bytes, max_bytes)?;
        let indptr = self
            .indptr
            .iter()
            .map(|value| {
                u64::try_from(*value).map_err(|_| PauliError::Overflow {
                    context: "converting U1 CSR offset",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let columns = self
            .columns
            .iter()
            .map(|value| {
                u64::try_from(*value).map_err(|_| PauliError::Overflow {
                    context: "converting U1 CSR column index",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(U1CsrMatrix {
            dimension,
            indptr,
            columns,
            values: self.values.to_vec(),
        })
    }
}

#[derive(Clone, Debug)]
struct U1XGroup {
    x_offset: usize,
    term_start: usize,
    term_end: usize,
}

#[derive(Clone, Debug)]
struct CompiledU1Terms {
    word_count: usize,
    x_words: Vec<u64>,
    z_words: Vec<u64>,
    weighted_coefficients: Vec<Complex64>,
    groups: Vec<U1XGroup>,
}

#[derive(Clone)]
struct PrecompiledU1Term {
    x_words: Vec<u64>,
    z_words: Vec<u64>,
    coefficient: Complex64,
}

fn compile_terms(operator: &PauliOperator) -> Result<CompiledU1Terms, PauliError> {
    let word_count = packed_word_count(operator.nqubits());
    let mut terms = Vec::with_capacity(operator.terms().len());
    for (index, term) in operator.terms().iter().enumerate() {
        let coefficient = weighted_coefficient(term, index)?;
        terms.push(PrecompiledU1Term {
            x_words: term.word.x_words().to_vec(),
            z_words: term.word.z_words().to_vec(),
            coefficient,
        });
    }

    // The hash map is used only for lookup. Group order and term order are
    // established by the already-canonical input stream, so hashing cannot
    // affect floating-point addition order or serialized outputs.
    let mut group_lookup =
        FxHashMap::<Vec<u64>, usize>::with_capacity_and_hasher(terms.len(), Default::default());
    let mut builders = Vec::<(Vec<u64>, Vec<usize>)>::new();
    for (term_index, term) in terms.iter().enumerate() {
        let group_index = if let Some(&group_index) = group_lookup.get(&term.x_words) {
            group_index
        } else {
            let group_index = builders.len();
            group_lookup.insert(term.x_words.clone(), group_index);
            builders.push((term.x_words.clone(), Vec::new()));
            group_index
        };
        builders[group_index].1.push(term_index);
    }

    let mut x_words = Vec::with_capacity(builders.len() * word_count);
    let mut z_words = Vec::with_capacity(terms.len() * word_count);
    let mut weighted_coefficients = Vec::with_capacity(terms.len());
    let mut groups = Vec::with_capacity(builders.len());
    for (x_mask, term_indices) in builders {
        let x_offset = x_words.len();
        x_words.extend_from_slice(&x_mask);
        let term_start = weighted_coefficients.len();
        for term_index in term_indices {
            z_words.extend_from_slice(&terms[term_index].z_words);
            weighted_coefficients.push(terms[term_index].coefficient);
        }
        groups.push(U1XGroup {
            x_offset,
            term_start,
            term_end: weighted_coefficients.len(),
        });
    }
    Ok(CompiledU1Terms {
        word_count,
        x_words,
        z_words,
        weighted_coefficients,
        groups,
    })
}

fn weighted_coefficient(term: &PauliTerm, index: usize) -> Result<Complex64, PauliError> {
    let y_count = term
        .word
        .x_words()
        .iter()
        .zip(term.word.z_words())
        .map(|(x, z)| (x & z).count_ones() as usize)
        .sum::<usize>();
    let y_phase = match y_count % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    };
    let coefficient = term.coefficient * y_phase;
    if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
        return Err(PauliError::NonFiniteCoefficient { index });
    }
    Ok(coefficient)
}

fn aggregate_source(
    source_index: u64,
    source: &[u64],
    destination: &mut [u64],
    terms: &CompiledU1Terms,
    sector: &U1Sector,
    aggregate: &mut Vec<(usize, Complex64)>,
) -> Result<(), PauliError> {
    aggregate.clear();
    for group in &terms.groups {
        let x_mask = &terms.x_words[group.x_offset..group.x_offset + terms.word_count];
        for ((destination_word, source_word), x_word) in
            destination.iter_mut().zip(source).zip(x_mask)
        {
            *destination_word = source_word ^ x_word;
        }
        let mut value = Complex64::default();
        for term_index in group.term_start..group.term_end {
            let z_start = term_index * terms.word_count;
            let parity = source
                .iter()
                .zip(&terms.z_words[z_start..z_start + terms.word_count])
                .fold(0_u32, |parity, (source_word, z_word)| {
                    parity ^ ((source_word & z_word).count_ones() & 1)
                });
            let contribution = terms.weighted_coefficients[term_index];
            if parity & 1 == 0 {
                value += contribution;
            } else {
                value -= contribution;
            }
        }
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(PauliError::NonFiniteCoefficient {
                index: group.term_start,
            });
        }
        if is_exact_zero(value) {
            continue;
        }
        let weight = destination
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum::<usize>();
        if weight != sector.particle_number {
            return Err(PauliError::SectorLeakage {
                source_index,
                expected: sector.particle_number,
                actual: weight,
            });
        }
        let restricted_destination = sector.rank_words(destination)?;
        let restricted_destination =
            usize::try_from(restricted_destination).map_err(|_| PauliError::Overflow {
                context: "converting U1 destination rank to a native index",
            })?;
        aggregate.push((restricted_destination, value));
    }
    Ok(())
}

fn transition_upper_bound(sector: &U1Sector, terms: &CompiledU1Terms) -> Result<usize, PauliError> {
    let mut total = 0_u128;
    for group in &terms.groups {
        let x_mask = &terms.x_words[group.x_offset..group.x_offset + terms.word_count];
        let x_weight = x_mask
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum::<usize>();
        if x_weight % 2 != 0 {
            continue;
        }
        let half = x_weight / 2;
        if half > sector.particle_number || half > sector.nqubits - sector.particle_number {
            continue;
        }
        let left = sector.combination(x_weight, half)?;
        let remaining_particles = sector.particle_number - half;
        let right = sector.combination(sector.nqubits - x_weight, remaining_particles)?;
        let candidate = (left as u128)
            .checked_mul(right as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating U1 transition candidates",
            })?;
        total = total.checked_add(candidate).ok_or(PauliError::Overflow {
            context: "summing U1 transition candidates",
        })?;
    }
    usize::try_from(total).map_err(|_| PauliError::Overflow {
        context: "converting U1 transition candidate count",
    })
}

fn estimate_plan_bytes(
    sector: &U1Sector,
    terms: &CompiledU1Terms,
    dimension: usize,
    upper_bound: usize,
) -> Result<u128, PauliError> {
    let compiled = (terms.x_words.len() as u128)
        .checked_mul(size_of::<u64>() as u128)
        .and_then(|bytes| {
            (terms.z_words.len() as u128)
                .checked_mul(size_of::<u64>() as u128)
                .and_then(|z| bytes.checked_add(z))
        })
        .and_then(|bytes| {
            (terms.weighted_coefficients.len() as u128)
                .checked_mul(size_of::<Complex64>() as u128)
                .and_then(|coefficients| bytes.checked_add(coefficients))
        })
        .and_then(|bytes| {
            (terms.groups.len() as u128)
                .checked_mul(size_of::<U1XGroup>() as u128)
                .and_then(|groups| bytes.checked_add(groups))
        })
        .ok_or(PauliError::Overflow {
            context: "estimating compiled U1 term memory",
        })?;
    let choose_bytes = (sector.choose.len() as u128)
        .checked_mul(size_of::<u64>() as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating U1 combinatorial table memory",
        })?;
    let row_bytes = (dimension as u128)
        .checked_mul(size_of::<usize>() as u128)
        .and_then(|bytes| {
            (dimension as u128 + 1)
                .checked_mul(size_of::<usize>() as u128)
                .and_then(|indptr| bytes.checked_add(indptr))
        })
        .and_then(|bytes| {
            (dimension as u128)
                .checked_mul(size_of::<usize>() as u128)
                .and_then(|next| bytes.checked_add(next))
        })
        .ok_or(PauliError::Overflow {
            context: "estimating U1 row workspace memory",
        })?;
    let transition_bytes = (upper_bound as u128)
        .checked_mul(size_of::<usize>() as u128 + size_of::<Complex64>() as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating U1 transition memory",
        })?;
    compiled
        .checked_add(choose_bytes)
        .and_then(|bytes| bytes.checked_add(row_bytes))
        .and_then(|bytes| bytes.checked_add(transition_bytes))
        .ok_or(PauliError::Overflow {
            context: "estimating U1 restricted plan memory",
        })
}

impl U1Sector {
    fn combination(&self, n: usize, k: usize) -> Result<u64, PauliError> {
        if k > n {
            return Ok(0);
        }
        let normalized = k.min(n - k);
        if normalized <= self.active_number && n <= self.nqubits {
            return Ok(self.choose[n * (self.active_number + 1) + normalized]);
        }
        choose_u64(n, normalized)
    }
}

fn build_choose_table(nqubits: usize, active_number: usize) -> Result<Arc<[u64]>, PauliError> {
    let width = active_number.checked_add(1).ok_or(PauliError::Overflow {
        context: "sizing U1 combinatorial table",
    })?;
    let entries = nqubits
        .checked_add(1)
        .and_then(|rows| rows.checked_mul(width))
        .ok_or(PauliError::Overflow {
            context: "sizing U1 combinatorial table",
        })?;
    let mut table = Vec::new();
    table
        .try_reserve_exact(entries)
        .map_err(|_| PauliError::Overflow {
            context: "allocating U1 combinatorial table",
        })?;
    table.resize(entries, 0_u64);
    for row in 0..=nqubits {
        table[row * width] = 1;
        let limit = active_number.min(row);
        for column in 1..=limit {
            table[row * width + column] = if column == row {
                1
            } else {
                table[(row - 1) * width + column - 1]
                    .checked_add(table[(row - 1) * width + column])
                    .ok_or(PauliError::Overflow {
                        context: "computing U1 combinatorial table",
                    })?
            };
        }
    }
    Ok(Arc::from(table.into_boxed_slice()))
}

fn choose_u64(n: usize, k: usize) -> Result<u64, PauliError> {
    if k > n {
        return Ok(0);
    }
    let k = k.min(n - k);
    let mut value = 1_u128;
    for index in 1..=k {
        value = value
            .checked_mul((n - k + index) as u128)
            .and_then(|product| product.checked_div(index as u128))
            .ok_or(PauliError::Overflow {
                context: "computing U1 sector dimension",
            })?;
        if value > u64::MAX as u128 {
            return Err(PauliError::Overflow {
                context: "representing U1 sector dimension as u64",
            });
        }
    }
    Ok(value as u64)
}

fn ensure_integer_width(nqubits: usize) -> Result<(), PauliError> {
    if nqubits > usize::BITS as usize {
        return Err(PauliError::Overflow {
            context: "representing a computational basis integer",
        });
    }
    Ok(())
}

fn tail_mask(nqubits: usize) -> u64 {
    match nqubits % 64 {
        0 if nqubits == 0 => 0,
        0 => u64::MAX,
        remainder => (1_u64 << remainder) - 1,
    }
}

fn check_allocation(requested: u128, limit: u128) -> Result<(), PauliError> {
    if requested > limit {
        return Err(PauliError::MemoryLimit { requested, limit });
    }
    Ok(())
}
