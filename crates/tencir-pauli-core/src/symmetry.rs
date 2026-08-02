//! Exact Pauli Z2 symmetry analysis and reusable Clifford tapering plans.

use std::mem::size_of;

use crate::error::PauliError;
use crate::operator::{PauliOperator, PauliTerm};
use crate::word::{PauliPhase, PauliWord};

/// A compact Clifford operation used by a [`Z2TaperingPlan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliffordOperation {
    /// Hadamard on one qubit.
    H { qubit: usize },
    /// Phase gate on one qubit.
    S { qubit: usize },
    /// Inverse phase gate on one qubit.
    Sdg { qubit: usize },
    /// Controlled-NOT with the given control and target.
    Cnot { control: usize, target: usize },
}

/// Deterministic analysis of Pauli-type Z2 symmetries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Z2SymmetryAnalysis {
    pub nqubits: usize,
    pub generators: Vec<PauliWord>,
    pub constraint_rank: usize,
}

impl Z2SymmetryAnalysis {
    /// Number of independent generators returned by the analysis.
    pub fn rank(&self) -> usize {
        self.generators.len()
    }
}

/// A reusable Clifford transform and selected Z2 sector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Z2TaperingPlan {
    nqubits_before: usize,
    generators: Vec<PauliWord>,
    sector: Vec<i8>,
    removed_qubits: Vec<usize>,
    kept_qubits: Vec<usize>,
    operations: Vec<CliffordOperation>,
    /// Each transformed row is a signed product of the original generators.
    row_combinations: Vec<Vec<u64>>,
    /// Sign in `U (product of original generators) U† = sign * Z_q`.
    row_signs: Vec<i8>,
}

/// Find all term-wise commuting Pauli candidates and choose a deterministic
/// independent pairwise-commuting generator basis.
pub fn find_z2_symmetries(
    operator: &PauliOperator,
    max_bytes: u128,
) -> Result<Z2SymmetryAnalysis, PauliError> {
    let nqubits = operator.nqubits();
    let variable_count = nqubits.checked_mul(2).ok_or(PauliError::Overflow {
        context: "estimating Z2 variables",
    })?;
    let packed_count = packed_bits(variable_count);
    let constraint_bytes = operator
        .terms()
        .len()
        .checked_mul(packed_count)
        .and_then(|value| value.checked_mul(size_of::<u64>()))
        .ok_or(PauliError::Overflow {
            context: "estimating Z2 constraint memory",
        })?;
    let nullspace_bytes = variable_count
        .checked_mul(packed_count)
        .and_then(|value| value.checked_mul(size_of::<u64>()))
        .ok_or(PauliError::Overflow {
            context: "estimating Z2 null-space memory",
        })?;
    check_allocation(
        (constraint_bytes as u128)
            .checked_add(nullspace_bytes as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating Z2 analysis memory",
            })?,
        max_bytes,
    )?;

    let mut constraints = Vec::with_capacity(operator.terms().len());
    for term in operator.terms() {
        constraints.push(constraint_for_term(term, nqubits, packed_count));
    }
    let (null_basis, constraint_rank) = nullspace(&mut constraints, variable_count);
    let mut candidates = null_basis
        .into_iter()
        .map(|bits| bits_to_word(&bits, nqubits))
        .collect::<Result<Vec<_>, _>>()?;
    candidates.sort_unstable();

    let candidate_bits = candidates
        .iter()
        .map(|word| word_bits(word, nqubits))
        .collect::<Vec<_>>();
    let selected_bits = select_isotropic_basis(candidate_bits, variable_count);
    let mut generators = selected_bits
        .into_iter()
        .map(|bits| bits_to_word(&bits, nqubits))
        .collect::<Result<Vec<_>, _>>()?;
    generators.sort_unstable();

    // The null-space construction is exact, but retain a final semantic gate
    // before exposing anything to callers.
    for generator in &generators {
        if generator.weight() == 0
            || operator
                .terms()
                .iter()
                .any(|term| !generator.commutes_with(&term.word).unwrap_or(false))
        {
            return Err(PauliError::IncompatibleSymmetry);
        }
    }
    for left in 0..generators.len() {
        for right in 0..left {
            if !generators[left].commutes_with(&generators[right])? {
                return Err(PauliError::IncompatibleSymmetry);
            }
        }
    }

    Ok(Z2SymmetryAnalysis {
        nqubits,
        generators,
        constraint_rank,
    })
}

impl Z2TaperingPlan {
    /// Construct a deterministic Clifford plan for independent commuting generators.
    pub fn new(
        nqubits: usize,
        generators: &[PauliWord],
        sector: &[i8],
    ) -> Result<Self, PauliError> {
        if generators.len() != sector.len() {
            return Err(PauliError::InvalidSector {
                context: "sector length must equal generator count",
            });
        }
        if sector.iter().any(|&value| value != 1 && value != -1) {
            return Err(PauliError::InvalidSector {
                context: "sector values must be +1 or -1",
            });
        }
        if generators.iter().any(|word| word.nqubits() != nqubits) {
            return Err(PauliError::IncompatibleQubitCounts {
                left: nqubits,
                right: generators
                    .iter()
                    .find(|word| word.nqubits() != nqubits)
                    .map_or(nqubits, PauliWord::nqubits),
            });
        }
        for (index, generator) in generators.iter().enumerate() {
            if generator.weight() == 0 {
                return Err(PauliError::InvalidSector {
                    context: "identity is not a Z2 generator",
                });
            }
            for previous in &generators[..index] {
                if !generator.commutes_with(previous)? {
                    return Err(PauliError::IncompatibleSymmetry);
                }
            }
        }
        if !is_independent(
            &generators
                .iter()
                .map(|word| word_bits(word, nqubits))
                .collect::<Vec<_>>(),
            nqubits * 2,
        ) {
            return Err(PauliError::InvalidSector {
                context: "generators must be linearly independent",
            });
        }

        let generator_count = generators.len();
        let combination_words = packed_bits(generator_count);
        let mut rows = generators.to_vec();
        let mut combinations = (0..generator_count)
            .map(|index| {
                let mut bits = vec![0_u64; combination_words];
                set_bit(&mut bits, index, true);
                bits
            })
            .collect::<Vec<_>>();
        let mut row_signs = vec![1_i8; generator_count];
        let mut operations = Vec::new();
        let mut removed_qubits = Vec::with_capacity(generator_count);
        let mut used = vec![false; nqubits];

        for row_index in 0..generator_count {
            // A row may contain Z on an already removed qubit. Multiply by
            // the corresponding mapped row to clear it before using gates on
            // the remaining qubits.
            for previous in 0..row_index {
                let pivot = removed_qubits[previous];
                if code_at(&rows[row_index], pivot) == 3 {
                    let (word, phase) = rows[row_index].multiply(&rows[previous])?;
                    rows[row_index] = word;
                    row_signs[row_index] *= row_signs[previous];
                    row_signs[row_index] *= phase_sign(phase)?;
                    let previous_combination = combinations[previous].clone();
                    xor_assign(&mut combinations[row_index], &previous_combination);
                }
            }

            let pivot = (0..nqubits)
                .find(|&qubit| !used[qubit] && code_at(&rows[row_index], qubit) != 0)
                .ok_or(PauliError::InvalidSector {
                    context: "independent generators have no distinct Clifford pivots",
                })?;
            used[pivot] = true;
            removed_qubits.push(pivot);

            match code_at(&rows[row_index], pivot) {
                1 => {}
                2 => apply_gate(
                    &mut rows,
                    &mut row_signs,
                    CliffordOperation::Sdg { qubit: pivot },
                    &mut operations,
                )?,
                3 => apply_gate(
                    &mut rows,
                    &mut row_signs,
                    CliffordOperation::H { qubit: pivot },
                    &mut operations,
                )?,
                _ => unreachable!(),
            }

            for (qubit, &is_used) in used.iter().enumerate() {
                if qubit != pivot && !is_used && matches!(code_at(&rows[row_index], qubit), 1 | 2) {
                    apply_gate(
                        &mut rows,
                        &mut row_signs,
                        CliffordOperation::Cnot {
                            control: pivot,
                            target: qubit,
                        },
                        &mut operations,
                    )?;
                }
            }

            let needs_z_cleanup = (0..nqubits).any(|qubit| {
                qubit != pivot && !used[qubit] && code_at(&rows[row_index], qubit) == 3
            });
            if needs_z_cleanup && code_at(&rows[row_index], pivot) == 1 {
                apply_gate(
                    &mut rows,
                    &mut row_signs,
                    CliffordOperation::S { qubit: pivot },
                    &mut operations,
                )?;
            }
            for (qubit, &is_used) in used.iter().enumerate() {
                if qubit != pivot && !is_used && code_at(&rows[row_index], qubit) == 3 {
                    apply_gate(
                        &mut rows,
                        &mut row_signs,
                        CliffordOperation::Cnot {
                            control: qubit,
                            target: pivot,
                        },
                        &mut operations,
                    )?;
                }
            }
            if needs_z_cleanup {
                apply_gate(
                    &mut rows,
                    &mut row_signs,
                    CliffordOperation::Sdg { qubit: pivot },
                    &mut operations,
                )?;
            }
            apply_gate(
                &mut rows,
                &mut row_signs,
                CliffordOperation::H { qubit: pivot },
                &mut operations,
            )?;

            if rows[row_index].weight() != 1 || code_at(&rows[row_index], pivot) != 3 {
                return Err(PauliError::InvalidClifford {
                    context: "failed to map a generator to a single-qubit Z",
                });
            }
        }

        let kept_qubits = (0..nqubits)
            .filter(|qubit| !used[*qubit])
            .collect::<Vec<_>>();
        Ok(Self {
            nqubits_before: nqubits,
            generators: generators.to_vec(),
            sector: sector.to_vec(),
            removed_qubits,
            kept_qubits,
            operations,
            row_combinations: combinations,
            row_signs,
        })
    }

    pub fn nqubits_before(&self) -> usize {
        self.nqubits_before
    }

    pub fn nqubits_after(&self) -> usize {
        self.nqubits_before - self.removed_qubits.len()
    }

    pub fn generators(&self) -> &[PauliWord] {
        &self.generators
    }

    pub fn sector(&self) -> &[i8] {
        &self.sector
    }

    pub fn removed_qubits(&self) -> &[usize] {
        &self.removed_qubits
    }

    pub fn operations(&self) -> &[CliffordOperation] {
        &self.operations
    }

    /// Apply the transform and substitute the selected symmetry eigenvalues.
    pub fn transform_operator(
        &self,
        operator: &PauliOperator,
    ) -> Result<PauliOperator, PauliError> {
        if operator.nqubits() != self.nqubits_before {
            return Err(PauliError::IncompatibleQubitCounts {
                left: self.nqubits_before,
                right: operator.nqubits(),
            });
        }
        for term in operator.terms() {
            if self
                .generators
                .iter()
                .any(|generator| !generator.commutes_with(&term.word).unwrap_or(false))
            {
                return Err(PauliError::IncompatibleSymmetry);
            }
        }

        let mut structures = Vec::with_capacity(operator.terms().len());
        let mut coefficients = Vec::with_capacity(operator.terms().len());
        for term in operator.terms() {
            let (word, mut sign) = transform_word(&term.word, &self.operations)?;
            for (index, &pivot) in self.removed_qubits.iter().enumerate() {
                match code_at(&word, pivot) {
                    0 => {}
                    3 => {
                        sign *= self.row_signs[index];
                        let combination = &self.row_combinations[index];
                        for generator_index in 0..self.generators.len() {
                            if get_bit(combination, generator_index) {
                                sign *= self.sector[generator_index];
                            }
                        }
                    }
                    _ => return Err(PauliError::IncompatibleSymmetry),
                }
            }
            let codes = word.codes();
            let structures_for_term = self
                .kept_qubits
                .iter()
                .map(|&qubit| codes[qubit])
                .collect::<Vec<_>>();
            structures.push(structures_for_term);
            coefficients.push(term.coefficient * f64::from(sign));
        }
        PauliOperator::from_terms(self.nqubits_after(), &structures, &coefficients)
    }
}

fn apply_gate(
    rows: &mut [PauliWord],
    row_signs: &mut [i8],
    operation: CliffordOperation,
    operations: &mut Vec<CliffordOperation>,
) -> Result<(), PauliError> {
    for (index, word) in rows.iter_mut().enumerate() {
        let sign = apply_operation_in_place(word, operation)?;
        row_signs[index] *= sign;
    }
    operations.push(operation);
    Ok(())
}

fn transform_word(
    word: &PauliWord,
    operations: &[CliffordOperation],
) -> Result<(PauliWord, i8), PauliError> {
    let mut result = word.clone();
    let mut sign = 1_i8;
    for &operation in operations {
        let operation_sign = apply_operation_in_place(&mut result, operation)?;
        sign *= operation_sign;
    }
    Ok((result, sign))
}

fn apply_operation_in_place(
    word: &mut PauliWord,
    operation: CliffordOperation,
) -> Result<i8, PauliError> {
    let nqubits = word.nqubits();
    match operation {
        CliffordOperation::H { qubit } => {
            ensure_qubit(qubit, nqubits)?;
            let (x, z) = packed_bits_at(word, qubit);
            let (new_x, new_z, sign) = match code_from_bits(x, z) {
                0 => (false, false, 1),
                1 => (false, true, 1),
                2 => (true, true, -1),
                3 => (true, false, 1),
                _ => unreachable!(),
            };
            set_packed_bit(&mut word.x_words, qubit, new_x);
            set_packed_bit(&mut word.z_words, qubit, new_z);
            Ok(sign)
        }
        CliffordOperation::S { qubit } => {
            ensure_qubit(qubit, nqubits)?;
            let (x, z) = packed_bits_at(word, qubit);
            let (new_x, new_z, sign) = match code_from_bits(x, z) {
                0 => (false, false, 1),
                1 => (true, true, 1),
                2 => (true, false, -1),
                3 => (false, true, 1),
                _ => unreachable!(),
            };
            set_packed_bit(&mut word.x_words, qubit, new_x);
            set_packed_bit(&mut word.z_words, qubit, new_z);
            Ok(sign)
        }
        CliffordOperation::Sdg { qubit } => {
            ensure_qubit(qubit, nqubits)?;
            let (x, z) = packed_bits_at(word, qubit);
            let (new_x, new_z, sign) = match code_from_bits(x, z) {
                0 => (false, false, 1),
                1 => (true, true, -1),
                2 => (true, false, 1),
                3 => (false, true, 1),
                _ => unreachable!(),
            };
            set_packed_bit(&mut word.x_words, qubit, new_x);
            set_packed_bit(&mut word.z_words, qubit, new_z);
            Ok(sign)
        }
        CliffordOperation::Cnot { control, target } => {
            ensure_qubit(control, nqubits)?;
            ensure_qubit(target, nqubits)?;
            if control == target {
                return Err(PauliError::InvalidClifford {
                    context: "CNOT control and target must differ",
                });
            }
            let control_mask = 1_u64 << (control % 64);
            let target_mask = 1_u64 << (target % 64);
            let control_word = control / 64;
            let target_word = target / 64;
            let control_x = word.x_words[control_word] & control_mask != 0;
            let target_x = word.x_words[target_word] & target_mask != 0;
            let control_z = word.z_words[control_word] & control_mask != 0;
            let target_z = word.z_words[target_word] & target_mask != 0;
            if control_x {
                word.x_words[target_word] ^= target_mask;
            }
            if target_z {
                word.z_words[control_word] ^= control_mask;
            }

            let control_code = code_from_bits(control_x, control_z);
            let target_code = code_from_bits(target_x, target_z);
            let control_target_code = u8::from(matches!(control_code, 1 | 2));
            let target_control_code = u8::from(matches!(target_code, 2 | 3)) * 3;
            let mut phase = PauliPhase::PlusOne;
            if control < target {
                phase = phase_multiply(phase, local_phase(control_code, target_control_code));
                phase = phase_multiply(phase, local_phase(control_target_code, target_code));
            } else {
                phase = phase_multiply(phase, local_phase(target_control_code, control_code));
                phase = phase_multiply(phase, local_phase(target_code, control_target_code));
            }
            Ok(phase_sign(phase)?)
        }
    }
}

fn ensure_qubit(qubit: usize, nqubits: usize) -> Result<(), PauliError> {
    if qubit >= nqubits {
        return Err(PauliError::InvalidClifford {
            context: "qubit index is outside the word",
        });
    }
    Ok(())
}

fn phase_sign(phase: PauliPhase) -> Result<i8, PauliError> {
    match phase {
        PauliPhase::PlusOne => Ok(1),
        PauliPhase::MinusOne => Ok(-1),
        PauliPhase::PlusI | PauliPhase::MinusI => Err(PauliError::InvalidClifford {
            context: "Clifford transform produced a non-Hermitian Pauli phase",
        }),
    }
}

fn packed_bits_at(word: &PauliWord, qubit: usize) -> (bool, bool) {
    let mask = 1_u64 << (qubit % 64);
    (
        word.x_words[qubit / 64] & mask != 0,
        word.z_words[qubit / 64] & mask != 0,
    )
}

fn set_packed_bit(words: &mut [u64], qubit: usize, value: bool) {
    let mask = 1_u64 << (qubit % 64);
    if value {
        words[qubit / 64] |= mask;
    } else {
        words[qubit / 64] &= !mask;
    }
}

fn code_at(word: &PauliWord, qubit: usize) -> u8 {
    let mask = 1_u64 << (qubit % 64);
    let x = word.x_words()[qubit / 64] & mask != 0;
    let z = word.z_words()[qubit / 64] & mask != 0;
    code_from_bits(x, z)
}

fn code_from_bits(x: bool, z: bool) -> u8 {
    match (x, z) {
        (false, false) => 0,
        (true, false) => 1,
        (true, true) => 2,
        (false, true) => 3,
    }
}

fn local_phase(left: u8, right: u8) -> PauliPhase {
    match (left, right) {
        (1, 2) | (2, 3) | (3, 1) => PauliPhase::PlusI,
        (2, 1) | (1, 3) | (3, 2) => PauliPhase::MinusI,
        _ => PauliPhase::PlusOne,
    }
}

fn phase_multiply(left: PauliPhase, right: PauliPhase) -> PauliPhase {
    let left = match left {
        PauliPhase::PlusOne => 0,
        PauliPhase::PlusI => 1,
        PauliPhase::MinusOne => 2,
        PauliPhase::MinusI => 3,
    };
    let right = match right {
        PauliPhase::PlusOne => 0,
        PauliPhase::PlusI => 1,
        PauliPhase::MinusOne => 2,
        PauliPhase::MinusI => 3,
    };
    match (left + right) % 4 {
        0 => PauliPhase::PlusOne,
        1 => PauliPhase::PlusI,
        2 => PauliPhase::MinusOne,
        _ => PauliPhase::MinusI,
    }
}

fn packed_bits(bits: usize) -> usize {
    bits / 64 + usize::from(bits % 64 != 0)
}

fn check_allocation(requested: u128, limit: u128) -> Result<(), PauliError> {
    if requested > limit {
        return Err(PauliError::MemoryLimit { requested, limit });
    }
    Ok(())
}

fn get_bit(bits: &[u64], index: usize) -> bool {
    bits[index / 64] & (1_u64 << (index % 64)) != 0
}

fn set_bit(bits: &mut [u64], index: usize, value: bool) {
    let mask = 1_u64 << (index % 64);
    if value {
        bits[index / 64] |= mask;
    } else {
        bits[index / 64] &= !mask;
    }
}

fn xor_assign(left: &mut [u64], right: &[u64]) {
    for (lhs, &rhs) in left.iter_mut().zip(right) {
        *lhs ^= rhs;
    }
}

fn parity_and(left: &[u64], right: &[u64]) -> bool {
    left.iter().zip(right).fold(false, |parity, (&lhs, &rhs)| {
        parity ^ ((lhs & rhs).count_ones() & 1 != 0)
    })
}

fn word_bits(word: &PauliWord, nqubits: usize) -> Vec<u64> {
    let variables = nqubits * 2;
    let mut result = vec![0_u64; packed_bits(variables)];
    for qubit in 0..nqubits {
        let code = code_at(word, qubit);
        if matches!(code, 1 | 2) {
            set_bit(&mut result, qubit, true);
        }
        if matches!(code, 2 | 3) {
            set_bit(&mut result, nqubits + qubit, true);
        }
    }
    result
}

fn bits_to_word(bits: &[u64], nqubits: usize) -> Result<PauliWord, PauliError> {
    let mut codes = vec![0_u8; nqubits];
    for (qubit, code) in codes.iter_mut().enumerate() {
        let x = get_bit(bits, qubit);
        let z = get_bit(bits, nqubits + qubit);
        *code = match (x, z) {
            (false, false) => 0,
            (true, false) => 1,
            (true, true) => 2,
            (false, true) => 3,
        };
    }
    PauliWord::from_codes(nqubits, &codes)
}

fn constraint_for_term(term: &PauliTerm, nqubits: usize, packed_count: usize) -> Vec<u64> {
    let mut row = vec![0_u64; packed_count];
    for qubit in 0..nqubits {
        let code = code_at(&term.word, qubit);
        if matches!(code, 2 | 3) {
            set_bit(&mut row, qubit, true);
        }
        if matches!(code, 1 | 2) {
            set_bit(&mut row, nqubits + qubit, true);
        }
    }
    row
}

fn nullspace(rows: &mut [Vec<u64>], variable_count: usize) -> (Vec<Vec<u64>>, usize) {
    let mut pivot_columns = Vec::new();
    let mut rank = 0;
    for column in 0..variable_count {
        let Some(pivot) = (rank..rows.len()).find(|&row| get_bit(&rows[row], column)) else {
            continue;
        };
        rows.swap(rank, pivot);
        let (prefix, suffix) = rows.split_at_mut(rank + 1);
        let pivot_row = &prefix[rank];
        for row in suffix {
            if get_bit(row, column) {
                xor_assign(row, pivot_row);
            }
        }
        pivot_columns.push(column);
        rank += 1;
    }
    let pivot_set = pivot_columns
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let mut result = Vec::new();
    for free in 0..variable_count {
        if pivot_set.contains(&free) {
            continue;
        }
        let mut vector = vec![0_u64; packed_bits(variable_count)];
        set_bit(&mut vector, free, true);
        for (row, &pivot) in pivot_columns.iter().enumerate().rev() {
            let value = parity_and(&rows[row], &vector);
            set_bit(&mut vector, pivot, value);
        }
        result.push(vector);
    }
    (result, rank)
}

fn symplectic(left: &[u64], right: &[u64], nqubits: usize) -> bool {
    let mut value = false;
    for qubit in 0..nqubits {
        value ^= get_bit(left, qubit) && get_bit(right, nqubits + qubit);
        value ^= get_bit(left, nqubits + qubit) && get_bit(right, qubit);
    }
    value
}

fn is_independent(vectors: &[Vec<u64>], variable_count: usize) -> bool {
    let mut rows = vectors.to_vec();
    let mut rank = 0;
    for column in 0..variable_count {
        let Some(pivot) = (rank..rows.len()).find(|&row| get_bit(&rows[row], column)) else {
            continue;
        };
        rows.swap(rank, pivot);
        let (prefix, suffix) = rows.split_at_mut(rank + 1);
        let pivot_row = &prefix[rank];
        for row in suffix {
            if get_bit(row, column) {
                xor_assign(row, pivot_row);
            }
        }
        rank += 1;
    }
    rank == vectors.len()
}

fn select_isotropic_basis(vectors: Vec<Vec<u64>>, variable_count: usize) -> Vec<Vec<u64>> {
    fn recurse(
        vectors: Vec<Vec<u64>>,
        nqubits: usize,
        variable_count: usize,
        output: &mut Vec<Vec<u64>>,
    ) {
        let Some(first) = vectors
            .iter()
            .find(|vector| vector.iter().any(|&word| word != 0))
        else {
            return;
        };
        let first = first.clone();
        output.push(first.clone());
        let partner = vectors
            .iter()
            .find(|vector| symplectic(&first, vector, nqubits))
            .cloned();
        let mut projected = Vec::new();
        for vector in vectors {
            if vector == first {
                continue;
            }
            let mut value = vector;
            if let Some(partner) = &partner {
                if symplectic(&first, &value, nqubits) {
                    xor_assign(&mut value, partner);
                }
            }
            if value.iter().any(|&word| word != 0) {
                let mut trial = output.clone();
                trial.extend(projected.iter().cloned());
                trial.push(value.clone());
                if is_independent(&trial, variable_count) {
                    projected.push(value);
                }
            }
        }
        recurse(projected, nqubits, variable_count, output);
    }

    if variable_count == 0 {
        return Vec::new();
    }
    let nqubits = variable_count / 2;
    let mut output = Vec::new();
    recurse(vectors, nqubits, variable_count, &mut output);
    output
}
