//! Rust-native classical-shadow snapshots and stabilizer kernels.
//!
//! The module deliberately keeps the scalable setting, frame, outcome, and
//! estimator data in Rust.  Python receives only immutable metadata and flat
//! owned arrays at explicit read-back boundaries.

use crate::error::PauliError;
use crate::gate::GateKind;
use crate::operator::PauliOperator;
use crate::pauli_action::StatevectorPauliAction;
use crate::propagation::{
    apply_clifford1_in_place, apply_clifford2_in_place, PackedKey, PropagationTape,
};
use crate::random::RandomStream;
use crate::scalar::Complex64;
use crate::word::{packed_word_count, PauliPhase, PauliWord};
use rayon::prelude::*;

const PAULI_FRAME_DOMAIN: u64 = 0x7061_756c_695f_6672;
const PAULI_OUTCOME_DOMAIN: u64 = 0x7061_756c_695f_6f75;
const CLIFFORD_FRAME_DOMAIN: u64 = 0x636c_6966_666f_726d;
const CLIFFORD_SIGN_DOMAIN: u64 = 0x636c_6966_665f_7367;
const CLIFFORD_OUTCOME_DOMAIN: u64 = 0x636c_6966_665f_6f75;
const MIN_CLIFFORD_STATEVECTOR_PARALLEL_WORK: usize = 32_768;
const MIN_CLIFFORD_STABILIZER_PARALLEL_WORK: usize = 2_048;
const MIN_DENSE_FIDELITY_PARALLEL_WORK: usize = 8_192;

fn checked_product(values: &[usize], context: &'static str) -> Result<usize, PauliError> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(PauliError::Overflow { context })
    })
}

fn checked_bytes(
    count: usize,
    element_size: usize,
    context: &'static str,
) -> Result<u128, PauliError> {
    (count as u128)
        .checked_mul(element_size as u128)
        .ok_or(PauliError::Overflow { context })
}

fn check_limit(requested: u128, limit: u128) -> Result<(), PauliError> {
    if requested > limit {
        Err(PauliError::MemoryLimit { requested, limit })
    } else {
        Ok(())
    }
}

fn validate_count(value: usize, context: &'static str) -> Result<(), PauliError> {
    if value == 0 {
        Err(PauliError::InvalidIndex { context })
    } else {
        Ok(())
    }
}

fn validate_sign(sign: i8) -> Result<(), PauliError> {
    if sign == 1 || sign == -1 {
        Ok(())
    } else {
        Err(PauliError::InvalidClifford {
            context: "stabilizer signs must be +1 or -1",
        })
    }
}

fn final_word_mask(nqubits: usize) -> u64 {
    match nqubits % 64 {
        0 => u64::MAX,
        bits => (1u64 << bits) - 1,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotProtocol {
    Pauli,
    Clifford,
}

impl SnapshotProtocol {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pauli => "pauli",
            Self::Clifford => "clifford",
        }
    }
}

#[derive(Clone, Debug)]
struct SignedPauli {
    x_words: Vec<u64>,
    z_words: Vec<u64>,
    sign: i8,
}

impl SignedPauli {
    fn new(nqubits: usize, x_words: &[u64], z_words: &[u64], sign: i8) -> Result<Self, PauliError> {
        let word_count = packed_word_count(nqubits);
        if x_words.len() != word_count || z_words.len() != word_count {
            return Err(PauliError::InvalidWordLength {
                expected: word_count,
                actual: x_words.len().min(z_words.len()),
            });
        }
        validate_sign(sign)?;
        let mut x = x_words.to_vec();
        let mut z = z_words.to_vec();
        if word_count != 0 {
            let mask = final_word_mask(nqubits);
            x[word_count - 1] &= mask;
            z[word_count - 1] &= mask;
        }
        Ok(Self {
            x_words: x,
            z_words: z,
            sign,
        })
    }

    fn from_word(word: &PauliWord, sign: i8) -> Result<Self, PauliError> {
        Self::new(word.nqubits(), word.x_words(), word.z_words(), sign)
    }

    fn symplectic_inner(&self, other: &Self) -> u8 {
        self.x_words
            .iter()
            .zip(&self.z_words)
            .zip(&other.x_words)
            .zip(&other.z_words)
            .fold(0u32, |parity, (((x, z), ox), oz)| {
                parity ^ ((x & oz).count_ones() & 1) ^ ((z & ox).count_ones() & 1)
            }) as u8
            & 1
    }

    fn word(&self, nqubits: usize) -> Result<PauliWord, PauliError> {
        PauliWord::from_words(nqubits, self.x_words.clone(), self.z_words.clone())
    }

    fn multiply(&self, other: &Self, nqubits: usize) -> Result<Self, PauliError> {
        let left = self.word(nqubits)?;
        let right = other.word(nqubits)?;
        let (word, phase) = left.multiply(&right)?;
        let phase_sign = match phase {
            PauliPhase::PlusOne => 1,
            PauliPhase::MinusOne => -1,
            PauliPhase::PlusI | PauliPhase::MinusI => {
                return Err(PauliError::InvalidClifford {
                    context: "multiplying stabilizer generators produced a non-Hermitian phase",
                })
            }
        };
        Self::from_word(&word, self.sign * other.sign * phase_sign)
    }

    fn multiply_in_place(&mut self, other: &Self) -> Result<(), PauliError> {
        let left_y = self
            .x_words
            .iter()
            .zip(&self.z_words)
            .map(|(x, z)| (x & z).count_ones())
            .sum::<u32>();
        let right_y = other
            .x_words
            .iter()
            .zip(&other.z_words)
            .map(|(x, z)| (x & z).count_ones())
            .sum::<u32>();
        let cross = self
            .z_words
            .iter()
            .zip(&other.x_words)
            .map(|(z, x)| (z & x).count_ones())
            .sum::<u32>();
        for (left, right) in self.x_words.iter_mut().zip(&other.x_words) {
            *left ^= *right;
        }
        for (left, right) in self.z_words.iter_mut().zip(&other.z_words) {
            *left ^= *right;
        }
        let result_y = self
            .x_words
            .iter()
            .zip(&self.z_words)
            .map(|(x, z)| (x & z).count_ones())
            .sum::<u32>();
        let exponent = (i64::from(left_y) + i64::from(right_y) - i64::from(result_y)
            + 2 * i64::from(cross))
        .rem_euclid(4) as u8;
        let phase_sign = match PauliPhase::from_exponent(exponent) {
            PauliPhase::PlusOne => 1,
            PauliPhase::MinusOne => -1,
            PauliPhase::PlusI | PauliPhase::MinusI => {
                return Err(PauliError::InvalidClifford {
                    context: "multiplying stabilizer generators produced a non-Hermitian phase",
                })
            }
        };
        self.sign *= other.sign * phase_sign;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct StabilizerState {
    nqubits: usize,
    generators: Vec<SignedPauli>,
}

impl StabilizerState {
    pub fn zero_state(nqubits: usize) -> Result<Self, PauliError> {
        let word_count = packed_word_count(nqubits);
        let mut generators = Vec::with_capacity(nqubits);
        for qubit in 0..nqubits {
            let mut z = vec![0u64; word_count];
            z[qubit / 64] |= 1u64 << (qubit % 64);
            generators.push(SignedPauli::new(nqubits, &vec![0; word_count], &z, 1)?);
        }
        Ok(Self {
            nqubits,
            generators,
        })
    }

    pub fn from_tableau(
        nqubits: usize,
        x_words: &[u64],
        z_words: &[u64],
        signs: &[i8],
    ) -> Result<Self, PauliError> {
        Self::from_flat_generators(nqubits, x_words, z_words, signs)
    }

    pub fn from_flat_generators(
        nqubits: usize,
        x_words: &[u64],
        z_words: &[u64],
        signs: &[i8],
    ) -> Result<Self, PauliError> {
        if signs.len() != nqubits {
            return Err(PauliError::InvalidStructureLength {
                expected: nqubits,
                actual: signs.len(),
            });
        }
        let word_count = packed_word_count(nqubits);
        let expected = nqubits
            .checked_mul(word_count)
            .ok_or(PauliError::Overflow {
                context: "estimating stabilizer tableau words",
            })?;
        if x_words.len() != expected {
            return Err(PauliError::InvalidStructureLength {
                expected,
                actual: x_words.len(),
            });
        }
        if z_words.len() != expected {
            return Err(PauliError::InvalidStructureLength {
                expected,
                actual: z_words.len(),
            });
        }
        let mut generators = Vec::with_capacity(nqubits);
        for (index, sign) in signs.iter().copied().enumerate() {
            let start = index * word_count;
            generators.push(SignedPauli::new(
                nqubits,
                &x_words[start..start + word_count],
                &z_words[start..start + word_count],
                sign,
            )?);
        }
        validate_generator_set(nqubits, &generators)?;
        Ok(Self {
            nqubits,
            generators,
        })
    }

    pub fn nqubits(&self) -> usize {
        self.nqubits
    }

    pub fn to_tableau(&self) -> (Vec<u64>, Vec<u64>, Vec<i8>) {
        let word_count = packed_word_count(self.nqubits);
        let mut x = Vec::with_capacity(self.nqubits * word_count);
        let mut z = Vec::with_capacity(self.nqubits * word_count);
        let mut signs = Vec::with_capacity(self.nqubits);
        for generator in &self.generators {
            x.extend_from_slice(&generator.x_words);
            z.extend_from_slice(&generator.z_words);
            signs.push(generator.sign);
        }
        (x, z, signs)
    }

    pub fn from_tape(tape: &PropagationTape) -> Result<Self, PauliError> {
        if tape.nparameters() != 0 {
            return Err(PauliError::UnsupportedCircuitGate {
                context: "StabilizerState requires a parameter-free Clifford tape",
            });
        }
        let mut state = Self::zero_state(tape.nqubits())?;
        for operation in tape.operations() {
            match &operation.kind {
                GateKind::Clifford1 { gate, wire } => {
                    for generator in &mut state.generators {
                        let mut key = PackedKey::from_word(&generator.word(state.nqubits)?);
                        let multiplier = apply_clifford1_in_place(&mut key, *gate, *wire);
                        generator.sign *= if multiplier < 0.0 { -1 } else { 1 };
                        *generator = signed_from_key(&key, state.nqubits, generator.sign)?;
                    }
                }
                GateKind::Clifford2 { gate, wire0, wire1 } => {
                    for generator in &mut state.generators {
                        let mut key = PackedKey::from_word(&generator.word(state.nqubits)?);
                        let multiplier = apply_clifford2_in_place(&mut key, *gate, *wire0, *wire1);
                        generator.sign *= if multiplier < 0.0 { -1 } else { 1 };
                        *generator = signed_from_key(&key, state.nqubits, generator.sign)?;
                    }
                }
                _ => {
                    return Err(PauliError::UnsupportedCircuitGate {
                        context: "StabilizerState supports only X/Y/Z/H/S/Sdg/CNOT/CZ/SWAP",
                    })
                }
            }
        }
        Ok(state)
    }

    fn expectation_signed(&self, measured: &SignedPauli) -> Result<f64, PauliError> {
        let word = measured.word(self.nqubits)?;
        let expectation = self.expectation_word(&word)?;
        Ok(expectation * f64::from(measured.sign))
    }

    fn measure_signed(
        &mut self,
        measured: &SignedPauli,
        random: &mut RandomStream,
    ) -> Result<u8, PauliError> {
        let pivot = self
            .generators
            .iter()
            .position(|generator| generator.symplectic_inner(measured) != 0);
        if let Some(pivot) = pivot {
            let old = self.generators[pivot].clone();
            for index in 0..self.generators.len() {
                if index != pivot && self.generators[index].symplectic_inner(measured) != 0 {
                    self.generators[index].multiply_in_place(&old)?;
                }
            }
            let bit = (random.next_u64() & 1) as u8;
            let eigenvalue = if bit == 0 { 1 } else { -1 };
            let mut replacement = measured.clone();
            replacement.sign *= eigenvalue;
            self.generators[pivot] = replacement;
            Ok(bit)
        } else {
            let expectation = self.expectation_signed(measured)?;
            if expectation > 0.0 {
                Ok(0)
            } else if expectation < 0.0 {
                Ok(1)
            } else {
                Err(PauliError::InvalidClifford {
                    context: "a stabilizer measurement had a non-deterministic commuting result",
                })
            }
        }
    }

    fn expectation_word(&self, word: &PauliWord) -> Result<f64, PauliError> {
        if word.nqubits() != self.nqubits {
            return Err(PauliError::IncompatibleQubitCounts {
                left: word.nqubits(),
                right: self.nqubits,
            });
        }
        let span = StabilizerSpan::new(&self.generators, self.nqubits);
        self.expectation_word_with_span(word, &span)
    }

    fn expectation_word_with_span(
        &self,
        word: &PauliWord,
        span: &StabilizerSpan,
    ) -> Result<f64, PauliError> {
        if word.x_words().iter().all(|value| *value == 0)
            && word.z_words().iter().all(|value| *value == 0)
        {
            return Ok(1.0);
        }
        let coefficients = span.solve(word.x_words(), word.z_words(), self.nqubits);
        let Some(coefficients) = coefficients else {
            return Ok(0.0);
        };
        let mut product = signed_identity(self.nqubits);
        for (index, bit) in coefficients.iter().enumerate() {
            if *bit {
                product = product.multiply(&self.generators[index], self.nqubits)?;
            }
        }
        let product_word = product.word(self.nqubits)?;
        if product_word != *word {
            return Err(PauliError::InvalidClifford {
                context: "stabilizer span reduction returned an inconsistent Pauli word",
            });
        }
        Ok(f64::from(product.sign))
    }
}

struct StabilizerSpan {
    basis: Vec<SpanRow>,
    coefficient_count: usize,
}

impl StabilizerSpan {
    fn new(generators: &[SignedPauli], nqubits: usize) -> Self {
        let mut basis = Vec::<SpanRow>::new();
        for (index, generator) in generators.iter().enumerate() {
            let mut row = SpanRow::new(
                generator.x_words.clone(),
                generator.z_words.clone(),
                nqubits,
                generators.len(),
            );
            row.coefficients[index] = true;
            reduce_span_row(&mut row, &basis, nqubits);
            if let Some(pivot) = first_set_bit(&row.x_words, &row.z_words) {
                let position = basis
                    .binary_search_by_key(&pivot, |value| value.pivot)
                    .unwrap_or_else(|position| position);
                basis.insert(position, SpanRow { pivot, ..row });
            }
        }
        Self {
            basis,
            coefficient_count: generators.len(),
        }
    }

    fn solve(&self, x: &[u64], z: &[u64], nqubits: usize) -> Option<Vec<bool>> {
        let mut target = SpanRow::new(x.to_vec(), z.to_vec(), nqubits, self.coefficient_count);
        reduce_span_row(&mut target, &self.basis, nqubits);
        if first_set_bit(&target.x_words, &target.z_words).is_some() {
            return None;
        }
        Some(target.coefficients)
    }
}

fn signed_identity(nqubits: usize) -> SignedPauli {
    let words = vec![0; packed_word_count(nqubits)];
    SignedPauli::new(nqubits, &words, &words, 1).expect("zero-width identity is valid")
}

fn signed_from_key(key: &PackedKey, nqubits: usize, sign: i8) -> Result<SignedPauli, PauliError> {
    let codes = (0..nqubits)
        .map(|qubit| key.code_at(qubit))
        .collect::<Vec<_>>();
    let word = PauliWord::from_codes(nqubits, &codes)?;
    SignedPauli::from_word(&word, sign)
}

fn validate_generator_set(nqubits: usize, generators: &[SignedPauli]) -> Result<(), PauliError> {
    if generators.len() != nqubits {
        return Err(PauliError::InvalidStructureLength {
            expected: nqubits,
            actual: generators.len(),
        });
    }
    for (left, generator) in generators.iter().enumerate() {
        for other in generators.iter().skip(left + 1) {
            if generator.symplectic_inner(other) != 0 {
                return Err(PauliError::InvalidClifford {
                    context: "stabilizer generators must commute",
                });
            }
        }
    }
    let mut basis = Vec::<SpanRow>::new();
    for (index, generator) in generators.iter().enumerate() {
        let mut row = SpanRow::new(
            generator.x_words.clone(),
            generator.z_words.clone(),
            nqubits,
            generators.len(),
        );
        row.coefficients[index] = true;
        reduce_span_row(&mut row, &basis, nqubits);
        if let Some(pivot) = first_set_bit(&row.x_words, &row.z_words) {
            let position = basis
                .binary_search_by_key(&pivot, |value| value.pivot)
                .unwrap_or_else(|position| position);
            basis.insert(position, SpanRow { pivot, ..row });
        } else {
            return Err(PauliError::InvalidClifford {
                context: "stabilizer generators must be independent",
            });
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct SpanRow {
    x_words: Vec<u64>,
    z_words: Vec<u64>,
    coefficients: Vec<bool>,
    pivot: usize,
}

impl SpanRow {
    fn new(
        x_words: Vec<u64>,
        z_words: Vec<u64>,
        _nqubits: usize,
        coefficient_count: usize,
    ) -> Self {
        Self {
            x_words,
            z_words,
            coefficients: vec![false; coefficient_count],
            pivot: 0,
        }
    }
}

fn first_set_bit(x_words: &[u64], z_words: &[u64]) -> Option<usize> {
    x_words
        .iter()
        .enumerate()
        .find_map(|(index, value)| {
            (*value != 0).then_some(index * 64 + value.trailing_zeros() as usize)
        })
        .or_else(|| {
            z_words.iter().enumerate().find_map(|(index, value)| {
                (*value != 0)
                    .then_some(x_words.len() * 64 + index * 64 + value.trailing_zeros() as usize)
            })
        })
}

fn xor_row(left: &mut SpanRow, right: &SpanRow) {
    for (target, value) in left.x_words.iter_mut().zip(&right.x_words) {
        *target ^= *value;
    }
    for (target, value) in left.z_words.iter_mut().zip(&right.z_words) {
        *target ^= *value;
    }
    for (target, value) in left.coefficients.iter_mut().zip(&right.coefficients) {
        *target ^= *value;
    }
}

fn reduce_span_row(row: &mut SpanRow, basis: &[SpanRow], _nqubits: usize) {
    while let Some(pivot) = first_set_bit(&row.x_words, &row.z_words) {
        let Some(position) = basis.binary_search_by_key(&pivot, |value| value.pivot).ok() else {
            break;
        };
        xor_row(row, &basis[position]);
    }
}

#[derive(Clone, Debug)]
struct BinaryVector {
    x_words: Vec<u64>,
    z_words: Vec<u64>,
}

impl BinaryVector {
    fn zero(word_count: usize) -> Self {
        Self {
            x_words: vec![0; word_count],
            z_words: vec![0; word_count],
        }
    }

    fn symplectic_inner(&self, other: &Self) -> u8 {
        self.x_words
            .iter()
            .zip(&self.z_words)
            .zip(&other.x_words)
            .zip(&other.z_words)
            .fold(0u32, |parity, (((x, z), ox), oz)| {
                parity ^ ((x & oz).count_ones() & 1) ^ ((z & ox).count_ones() & 1)
            }) as u8
            & 1
    }
}

fn xor_binary(left: &mut BinaryVector, right: &BinaryVector) {
    for (target, value) in left.x_words.iter_mut().zip(&right.x_words) {
        *target ^= *value;
    }
    for (target, value) in left.z_words.iter_mut().zip(&right.z_words) {
        *target ^= *value;
    }
}

fn combine_binary(basis: &[BinaryVector], coefficients: &[bool]) -> BinaryVector {
    let word_count = basis.first().map_or(0, |value| value.x_words.len());
    let mut result = BinaryVector::zero(word_count);
    for (vector, selected) in basis.iter().zip(coefficients) {
        if *selected {
            xor_binary(&mut result, vector);
        }
    }
    result
}

fn nullspace_basis(rows: &[Vec<u8>], dimension: usize) -> Vec<Vec<bool>> {
    let mut matrix = rows
        .iter()
        .map(|row| row.iter().map(|value| value & 1).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let mut pivots = Vec::new();
    let mut rank = 0usize;
    for column in 0..dimension {
        let Some(source) = (rank..matrix.len()).find(|row| matrix[*row][column] != 0) else {
            continue;
        };
        matrix.swap(rank, source);
        let pivot_values = matrix[rank][column..dimension].to_vec();
        for (row, current_row) in matrix.iter_mut().enumerate() {
            if row != rank && current_row[column] != 0 {
                for (value, pivot) in current_row[column..dimension].iter_mut().zip(&pivot_values) {
                    *value ^= *pivot;
                }
            }
        }
        pivots.push(column);
        rank += 1;
        if rank == matrix.len() {
            break;
        }
    }
    let mut is_pivot = vec![false; dimension];
    for pivot in &pivots {
        is_pivot[*pivot] = true;
    }
    let mut result = Vec::new();
    for (free, pivot) in is_pivot.iter().copied().enumerate() {
        if pivot {
            continue;
        }
        let mut coefficients = vec![false; dimension];
        coefficients[free] = true;
        for (row, pivot) in pivots.iter().enumerate() {
            coefficients[*pivot] = matrix[row][free] != 0;
        }
        result.push(coefficients);
    }
    result
}

fn sample_nonzero_subspace(
    basis: &[BinaryVector],
    random: &mut RandomStream,
) -> Result<BinaryVector, PauliError> {
    if basis.is_empty() {
        return Err(PauliError::InvalidClifford {
            context: "cannot sample a nonzero vector from a zero-dimensional subspace",
        });
    }
    loop {
        let coefficients = (0..basis.len())
            .map(|_| (random.next_u64() & 1) != 0)
            .collect::<Vec<_>>();
        if coefficients.iter().any(|value| *value) {
            return Ok(combine_binary(basis, &coefficients));
        }
    }
}

fn sample_anticommuting_vector(
    basis: &[BinaryVector],
    z: &BinaryVector,
    random: &mut RandomStream,
) -> Result<BinaryVector, PauliError> {
    let dual = basis
        .iter()
        .map(|vector| vector.symplectic_inner(z))
        .collect::<Vec<_>>();
    let Some(pivot) = dual.iter().position(|value| *value != 0) else {
        return Err(PauliError::InvalidClifford {
            context: "symplectic frame sampler could not find an anticommuting vector",
        });
    };
    let mut coefficients = vec![false; basis.len()];
    let mut parity = 0u8;
    for index in 0..basis.len() {
        if index == pivot {
            continue;
        }
        coefficients[index] = (random.next_u64() & 1) != 0;
        if coefficients[index] {
            parity ^= dual[index];
        }
    }
    coefficients[pivot] = parity == 0;
    Ok(combine_binary(basis, &coefficients))
}

#[derive(Clone, Debug)]
struct PackedBits {
    words: Vec<u64>,
}

impl PackedBits {
    fn zero(bit_count: usize) -> Self {
        Self {
            words: vec![0; bit_count.div_ceil(64)],
        }
    }

    fn get(&self, bit: usize) -> bool {
        self.words[bit / 64] & (1u64 << (bit % 64)) != 0
    }

    fn set(&mut self, bit: usize) {
        self.words[bit / 64] |= 1u64 << (bit % 64);
    }

    fn xor_assign(&mut self, other: &Self) {
        for (left, right) in self.words.iter_mut().zip(&other.words) {
            *left ^= *right;
        }
    }
}

fn nullspace_basis_packed(mut rows: Vec<PackedBits>, dimension: usize) -> Vec<PackedBits> {
    let mut pivots = Vec::new();
    let mut rank = 0usize;
    for column in 0..dimension {
        let Some(source) = (rank..rows.len()).find(|row| rows[*row].get(column)) else {
            continue;
        };
        rows.swap(rank, source);
        let pivot = rows[rank].clone();
        for (row, current) in rows.iter_mut().enumerate() {
            if row != rank && current.get(column) {
                current.xor_assign(&pivot);
            }
        }
        pivots.push(column);
        rank += 1;
        if rank == rows.len() {
            break;
        }
    }
    let mut is_pivot = vec![false; dimension];
    for pivot in &pivots {
        is_pivot[*pivot] = true;
    }
    let mut result = Vec::new();
    for (free, pivot) in is_pivot.iter().copied().enumerate() {
        if pivot {
            continue;
        }
        let mut coefficients = PackedBits::zero(dimension);
        coefficients.set(free);
        for (row, pivot) in pivots.iter().enumerate() {
            if rows[row].get(free) {
                coefficients.set(*pivot);
            }
        }
        result.push(coefficients);
    }
    result
}

fn combine_binary_packed(basis: &[BinaryVector], coefficients: &PackedBits) -> BinaryVector {
    let word_count = basis.first().map_or(0, |value| value.x_words.len());
    let mut result = BinaryVector::zero(word_count);
    for (coefficient_word, selected) in coefficients.words.iter().copied().enumerate() {
        let mut remaining = selected;
        while remaining != 0 {
            let offset = remaining.trailing_zeros() as usize;
            xor_binary(&mut result, &basis[coefficient_word * 64 + offset]);
            remaining &= remaining - 1;
        }
    }
    result
}

fn symplectic_complement_packed(
    basis: &[BinaryVector],
    x: &BinaryVector,
    z: &BinaryVector,
) -> Vec<BinaryVector> {
    let mut x_constraints = PackedBits::zero(basis.len());
    let mut z_constraints = PackedBits::zero(basis.len());
    for (index, vector) in basis.iter().enumerate() {
        if vector.symplectic_inner(x) != 0 {
            x_constraints.set(index);
        }
        if vector.symplectic_inner(z) != 0 {
            z_constraints.set(index);
        }
    }
    nullspace_basis_packed(vec![x_constraints, z_constraints], basis.len())
        .iter()
        .map(|coefficients| combine_binary_packed(basis, coefficients))
        .collect()
}

fn sample_symplectic_frame(
    nqubits: usize,
    random: &mut RandomStream,
) -> Result<Vec<BinaryVector>, PauliError> {
    let word_count = packed_word_count(nqubits);
    let mut basis = Vec::with_capacity(2 * nqubits);
    for qubit in 0..nqubits {
        let mut x = BinaryVector::zero(word_count);
        x.x_words[qubit / 64] |= 1u64 << (qubit % 64);
        basis.push(x);
        let mut z = BinaryVector::zero(word_count);
        z.z_words[qubit / 64] |= 1u64 << (qubit % 64);
        basis.push(z);
    }
    let mut retained_z = Vec::with_capacity(nqubits);
    for _ in 0..nqubits {
        let z = sample_nonzero_subspace(&basis, random)?;
        let x = sample_anticommuting_vector(&basis, &z, random)?;
        retained_z.push(z.clone());
        basis = symplectic_complement_packed(&basis, &x, &z);
    }
    Ok(retained_z)
}

#[derive(Clone, Debug)]
pub enum SnapshotData {
    Pauli {
        nqubits: usize,
        nsettings: usize,
        repeats: usize,
        bases: Vec<u8>,
        outcomes: Vec<u8>,
    },
    Clifford {
        nqubits: usize,
        nsettings: usize,
        repeats: usize,
        frame_x_words: Vec<u64>,
        frame_z_words: Vec<u64>,
        frame_signs: Vec<i8>,
        outcomes: Vec<u8>,
    },
}

#[derive(Clone, Debug)]
pub struct Snapshots {
    data: SnapshotData,
}

impl Snapshots {
    pub fn from_pauli_measurements(
        nsettings: usize,
        nqubits: usize,
        repeats: usize,
        bases: &[u8],
        outcomes: &[u8],
        max_bytes: u128,
    ) -> Result<Self, PauliError> {
        validate_count(nsettings, "nsettings must be positive")?;
        validate_count(repeats, "repeats must be positive")?;
        let basis_count = checked_product(&[nsettings, nqubits], "estimating Pauli basis storage")?;
        let outcome_count = checked_product(
            &[nsettings, repeats, nqubits],
            "estimating Pauli outcome storage",
        )?;
        if bases.len() != basis_count {
            return Err(PauliError::InvalidStructureLength {
                expected: basis_count,
                actual: bases.len(),
            });
        }
        if outcomes.len() != outcome_count {
            return Err(PauliError::InvalidStructureLength {
                expected: outcome_count,
                actual: outcomes.len(),
            });
        }
        if bases.iter().any(|code| !(1..=3).contains(code)) {
            return Err(PauliError::InvalidCode { code: 0, index: 0 });
        }
        if let Some(index) = outcomes.iter().position(|value| *value > 1) {
            return Err(PauliError::InvalidCode {
                code: outcomes[index],
                index,
            });
        }
        let requested = checked_bytes(
            basis_count + outcome_count,
            1,
            "estimating Pauli snapshot storage",
        )?;
        check_limit(requested, max_bytes)?;
        Ok(Self {
            data: SnapshotData::Pauli {
                nqubits,
                nsettings,
                repeats,
                bases: bases.to_vec(),
                outcomes: outcomes.to_vec(),
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_clifford_measurements(
        nsettings: usize,
        nqubits: usize,
        repeats: usize,
        frame_x_words: &[u64],
        frame_z_words: &[u64],
        frame_signs: &[i8],
        outcomes: &[u8],
        max_bytes: u128,
    ) -> Result<Self, PauliError> {
        validate_count(nsettings, "nsettings must be positive")?;
        validate_count(repeats, "repeats must be positive")?;
        let word_count = packed_word_count(nqubits);
        let frame_rows = checked_product(&[nsettings, nqubits], "estimating Clifford frame rows")?;
        let frame_words = checked_product(
            &[frame_rows, word_count],
            "estimating Clifford frame storage",
        )?;
        let outcome_count = checked_product(
            &[nsettings, repeats, nqubits],
            "estimating Clifford outcome storage",
        )?;
        if frame_x_words.len() != frame_words || frame_z_words.len() != frame_words {
            return Err(PauliError::InvalidStructureLength {
                expected: frame_words,
                actual: frame_x_words.len().min(frame_z_words.len()),
            });
        }
        if frame_signs.len() != frame_rows {
            return Err(PauliError::InvalidStructureLength {
                expected: frame_rows,
                actual: frame_signs.len(),
            });
        }
        if outcomes.len() != outcome_count {
            return Err(PauliError::InvalidStructureLength {
                expected: outcome_count,
                actual: outcomes.len(),
            });
        }
        if word_count != 0 {
            let mask = final_word_mask(nqubits);
            for row in 0..frame_rows {
                if frame_x_words[row * word_count + word_count - 1] & !mask != 0
                    || frame_z_words[row * word_count + word_count - 1] & !mask != 0
                {
                    return Err(PauliError::InvalidClifford {
                        context: "Clifford frame padding bits must be zero",
                    });
                }
            }
        }
        for sign in frame_signs.iter().copied() {
            validate_sign(sign).map_err(|_| PauliError::InvalidClifford {
                context: "Clifford frame signs must be +1 or -1",
            })?;
        }
        if let Some(index) = outcomes.iter().position(|value| *value > 1) {
            return Err(PauliError::InvalidCode {
                code: outcomes[index],
                index,
            });
        }
        let mut rows = Vec::with_capacity(frame_rows);
        for (row, sign) in frame_signs.iter().copied().enumerate() {
            let start = row * word_count;
            rows.push(SignedPauli::new(
                nqubits,
                &frame_x_words[start..start + word_count],
                &frame_z_words[start..start + word_count],
                sign,
            )?);
        }
        for setting in 0..nsettings {
            let setting_rows = &rows[setting * nqubits..(setting + 1) * nqubits];
            validate_generator_set(nqubits, setting_rows)?;
        }
        let requested = checked_bytes(
            frame_x_words.len() + frame_z_words.len(),
            8,
            "estimating Clifford frame storage",
        )?
        .checked_add(checked_bytes(
            frame_signs.len(),
            1,
            "estimating Clifford sign storage",
        )?)
        .and_then(|value| {
            value.checked_add(
                checked_bytes(outcome_count, 1, "estimating Clifford outcome storage").ok()?,
            )
        })
        .ok_or(PauliError::Overflow {
            context: "estimating Clifford snapshot storage",
        })?;
        check_limit(requested, max_bytes)?;
        Ok(Self {
            data: SnapshotData::Clifford {
                nqubits,
                nsettings,
                repeats,
                frame_x_words: frame_x_words.to_vec(),
                frame_z_words: frame_z_words.to_vec(),
                frame_signs: frame_signs.to_vec(),
                outcomes: outcomes.to_vec(),
            },
        })
    }

    pub fn sample_statevector(
        state: &[Complex64],
        nsettings: usize,
        protocol: SnapshotProtocol,
        repeats: usize,
        seed: u64,
        max_bytes: u128,
    ) -> Result<Self, PauliError> {
        validate_count(nsettings, "nsettings must be positive")?;
        validate_count(repeats, "repeats must be positive")?;
        let dimension = state.len();
        if dimension == 0 || !dimension.is_power_of_two() {
            return Err(PauliError::InvalidStructureLength {
                expected: dimension.next_power_of_two(),
                actual: dimension,
            });
        }
        let nqubits = dimension.trailing_zeros() as usize;
        if state
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(PauliError::InvalidClifford {
                context: "statevector entries must be finite",
            });
        }
        let norm = state.iter().map(|value| value.norm_sqr()).sum::<f64>();
        if !norm.is_finite() || (norm - 1.0).abs() > 1e-10 {
            return Err(PauliError::InvalidClifford {
                context: "statevector norm must be within 1e-10 of one",
            });
        }
        let scale = norm.sqrt().recip();
        let normalized = state.iter().map(|value| *value * scale).collect::<Vec<_>>();
        match protocol {
            SnapshotProtocol::Pauli => {
                sample_pauli_statevector(&normalized, nqubits, nsettings, repeats, seed, max_bytes)
            }
            SnapshotProtocol::Clifford => sample_clifford_statevector(
                &normalized,
                nqubits,
                nsettings,
                repeats,
                seed,
                max_bytes,
            ),
        }
    }

    pub fn sample_stabilizer(
        state: &StabilizerState,
        nsettings: usize,
        protocol: SnapshotProtocol,
        repeats: usize,
        seed: u64,
        max_bytes: u128,
    ) -> Result<Self, PauliError> {
        validate_count(nsettings, "nsettings must be positive")?;
        validate_count(repeats, "repeats must be positive")?;
        match protocol {
            SnapshotProtocol::Pauli => {
                sample_pauli_stabilizer(state, nsettings, repeats, seed, max_bytes)
            }
            SnapshotProtocol::Clifford => {
                sample_clifford_stabilizer(state, nsettings, repeats, seed, max_bytes)
            }
        }
    }

    pub fn protocol(&self) -> SnapshotProtocol {
        match &self.data {
            SnapshotData::Pauli { .. } => SnapshotProtocol::Pauli,
            SnapshotData::Clifford { .. } => SnapshotProtocol::Clifford,
        }
    }

    pub fn nqubits(&self) -> usize {
        match &self.data {
            SnapshotData::Pauli { nqubits, .. } | SnapshotData::Clifford { nqubits, .. } => {
                *nqubits
            }
        }
    }

    pub fn nsettings(&self) -> usize {
        match &self.data {
            SnapshotData::Pauli { nsettings, .. } | SnapshotData::Clifford { nsettings, .. } => {
                *nsettings
            }
        }
    }

    pub fn repeats(&self) -> usize {
        match &self.data {
            SnapshotData::Pauli { repeats, .. } | SnapshotData::Clifford { repeats, .. } => {
                *repeats
            }
        }
    }

    pub fn shots(&self) -> Result<usize, PauliError> {
        self.nsettings()
            .checked_mul(self.repeats())
            .ok_or(PauliError::Overflow {
                context: "computing snapshot shot count",
            })
    }

    pub fn pauli_arrays(&self) -> Option<(&[u8], &[u8])> {
        match &self.data {
            SnapshotData::Pauli {
                bases, outcomes, ..
            } => Some((bases, outcomes)),
            SnapshotData::Clifford { .. } => None,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn clifford_arrays(&self) -> Option<(&[u64], &[u64], &[i8], &[u8])> {
        match &self.data {
            SnapshotData::Clifford {
                frame_x_words,
                frame_z_words,
                frame_signs,
                outcomes,
                ..
            } => Some((frame_x_words, frame_z_words, frame_signs, outcomes)),
            SnapshotData::Pauli { .. } => None,
        }
    }

    fn setting_values<'a, I>(&self, operators: I) -> Result<Vec<f64>, PauliError>
    where
        I: IntoIterator<Item = &'a PauliOperator> + Clone,
        I::IntoIter: ExactSizeIterator + Clone,
    {
        let operators = operators.into_iter();
        let operator_count = operators.len();
        if operator_count == 0 {
            return Err(PauliError::InvalidStructureLength {
                expected: 1,
                actual: 0,
            });
        }
        for operator in operators.clone() {
            if operator.nqubits() != self.nqubits() {
                return Err(PauliError::IncompatibleQubitCounts {
                    left: operator.nqubits(),
                    right: self.nqubits(),
                });
            }
            if !operator.is_hermitian(0.0) {
                return Err(PauliError::NonHermitianExpectation);
            }
        }
        match &self.data {
            SnapshotData::Pauli {
                nqubits,
                nsettings,
                repeats,
                bases,
                outcomes,
            } => {
                let mut values = vec![0.0; *nsettings * operator_count];
                for setting in 0..*nsettings {
                    for repeat in 0..*repeats {
                        let outcome_start = (setting * *repeats + repeat) * *nqubits;
                        for (operator_index, operator) in operators.clone().enumerate() {
                            let mut value = 0.0;
                            for term in operator.terms() {
                                let mut matches = true;
                                let mut parity = 0u32;
                                for qubit in 0..*nqubits {
                                    let code = term.word.code_at(qubit);
                                    if code != 0 {
                                        if bases[setting * *nqubits + qubit] != code {
                                            matches = false;
                                            break;
                                        }
                                        parity ^= u32::from(outcomes[outcome_start + qubit]);
                                    }
                                }
                                if matches {
                                    let eigenvalue = if parity & 1 == 0 { 1.0 } else { -1.0 };
                                    value += term.coefficient.re
                                        * (3.0f64).powi(term.word.weight() as i32)
                                        * eigenvalue;
                                }
                            }
                            values[setting * operator_count + operator_index] +=
                                value / *repeats as f64;
                        }
                    }
                }
                Ok(values)
            }
            SnapshotData::Clifford {
                nqubits,
                nsettings,
                repeats,
                frame_x_words,
                frame_z_words,
                frame_signs,
                outcomes,
            } => {
                let word_count = packed_word_count(*nqubits);
                let mut values = vec![0.0; *nsettings * operator_count];
                for setting in 0..*nsettings {
                    for repeat in 0..*repeats {
                        let outcome_start = (setting * *repeats + repeat) * *nqubits;
                        let mut generators = Vec::with_capacity(*nqubits);
                        for generator in 0..*nqubits {
                            let row = setting * *nqubits + generator;
                            let start = row * word_count;
                            let sign = frame_signs[row]
                                * if outcomes[outcome_start + generator] == 0 {
                                    1
                                } else {
                                    -1
                                };
                            generators.push(SignedPauli::new(
                                *nqubits,
                                &frame_x_words[start..start + word_count],
                                &frame_z_words[start..start + word_count],
                                sign,
                            )?);
                        }
                        let snapshot = StabilizerState {
                            nqubits: *nqubits,
                            generators,
                        };
                        let span = StabilizerSpan::new(&snapshot.generators, snapshot.nqubits);
                        for (operator_index, operator) in operators.clone().enumerate() {
                            let value = operator
                                .terms()
                                .iter()
                                .map(|term| {
                                    Ok(term.coefficient.re
                                        * if term.word.weight() == 0 {
                                            1.0
                                        } else {
                                            (2.0f64.powi(*nqubits as i32) + 1.0)
                                                * snapshot
                                                    .expectation_word_with_span(&term.word, &span)?
                                        })
                                })
                                .collect::<Result<Vec<_>, PauliError>>()?
                                .into_iter()
                                .sum::<f64>();
                            values[setting * operator_count + operator_index] +=
                                value / *repeats as f64;
                        }
                    }
                }
                Ok(values)
            }
        }
    }
}

fn sample_pauli_statevector(
    state: &[Complex64],
    nqubits: usize,
    nsettings: usize,
    repeats: usize,
    seed: u64,
    max_bytes: u128,
) -> Result<Snapshots, PauliError> {
    let dimension = state.len();
    let basis_count = checked_product(&[nsettings, nqubits], "estimating Pauli basis storage")?;
    let outcome_count = checked_product(
        &[nsettings, repeats, nqubits],
        "estimating Pauli outcome storage",
    )?;
    let retained_bytes = checked_bytes(
        basis_count
            .checked_add(outcome_count)
            .ok_or(PauliError::Overflow {
                context: "estimating Pauli snapshot storage",
            })?,
        1,
        "estimating Pauli snapshot storage",
    )?;
    let worker_bytes = checked_bytes(dimension, 16, "estimating Pauli state scratch")?
        .checked_add(checked_bytes(
            dimension,
            8,
            "estimating Pauli probability scratch",
        )?)
        .ok_or(PauliError::Overflow {
            context: "estimating Pauli sampler memory",
        })?;
    let worker_count =
        bounded_shadow_worker_count(nsettings, retained_bytes, worker_bytes, max_bytes);
    let requested = retained_bytes
        .checked_add(worker_bytes.checked_mul(worker_count as u128).ok_or(
            PauliError::Overflow {
                context: "estimating parallel Pauli sampler memory",
            },
        )?)
        .ok_or(PauliError::Overflow {
            context: "estimating parallel Pauli sampler memory",
        })?;
    check_limit(requested, max_bytes)?;
    let mut bases = vec![0u8; basis_count];
    let mut outcomes = vec![0u8; outcome_count];
    if nqubits == 0 {
        return Snapshots::from_pauli_measurements(
            nsettings, nqubits, repeats, &bases, &outcomes, max_bytes,
        );
    }
    let outcome_stride = checked_product(&[repeats, nqubits], "estimating Pauli outcome stride")?;
    let settings_per_chunk = nsettings.div_ceil(worker_count);
    let basis_chunk_len = checked_product(
        &[settings_per_chunk, nqubits],
        "estimating Pauli basis chunk",
    )?;
    let outcome_chunk_len = checked_product(
        &[settings_per_chunk, outcome_stride],
        "estimating Pauli outcome chunk",
    )?;
    let mut sample_chunks = || {
        bases
            .par_chunks_mut(basis_chunk_len)
            .zip(outcomes.par_chunks_mut(outcome_chunk_len))
            .enumerate()
            .map(|(chunk, (basis_chunk, outcome_chunk))| {
                let chunk_settings = basis_chunk.len() / nqubits;
                sample_pauli_statevector_chunk(
                    state,
                    nqubits,
                    dimension,
                    repeats,
                    seed,
                    PauliStatevectorChunk {
                        setting_start: chunk * settings_per_chunk,
                        chunk_settings,
                        bases: basis_chunk,
                        outcomes: outcome_chunk,
                    },
                )
            })
            .collect::<Result<Vec<_>, PauliError>>()
    };
    if worker_count > 1 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build()
            .expect("shadow worker count is positive");
        pool.install(sample_chunks)?;
    } else {
        sample_chunks()?;
    }

    Snapshots::from_pauli_measurements(nsettings, nqubits, repeats, &bases, &outcomes, max_bytes)
}

struct PauliStatevectorChunk<'a> {
    setting_start: usize,
    chunk_settings: usize,
    bases: &'a mut [u8],
    outcomes: &'a mut [u8],
}

fn sample_pauli_statevector_chunk(
    state: &[Complex64],
    nqubits: usize,
    dimension: usize,
    repeats: usize,
    seed: u64,
    chunk: PauliStatevectorChunk<'_>,
) -> Result<(), PauliError> {
    let mut transformed = state.to_vec();
    let mut cumulative = Vec::with_capacity(dimension);
    for local_setting in 0..chunk.chunk_settings {
        let setting = chunk.setting_start + local_setting;
        let mut basis_random = RandomStream::new(seed, PAULI_FRAME_DOMAIN, setting, 0);
        for qubit in 0..nqubits {
            chunk.bases[local_setting * nqubits + qubit] = (basis_random.bounded(3)? + 1) as u8;
        }
        transformed.copy_from_slice(state);
        for qubit in 0..nqubits {
            match chunk.bases[local_setting * nqubits + qubit] {
                1 => apply_h_in_place(&mut transformed, qubit, nqubits),
                2 => apply_y_measurement_rotation(&mut transformed, qubit, nqubits),
                3 => {}
                _ => unreachable!(),
            }
        }
        cumulative.clear();
        let mut total = 0.0;
        for value in &transformed {
            total += value.norm_sqr();
        }
        if !total.is_finite() || total <= 0.0 {
            return Err(PauliError::InvalidClifford {
                context: "Pauli measurement probabilities are not finite and positive",
            });
        }
        let mut running = 0.0;
        for value in &transformed {
            running += value.norm_sqr() / total;
            cumulative.push(running);
        }
        if let Some(last) = cumulative.last_mut() {
            *last = 1.0;
        }
        let mut outcome_random = RandomStream::new(seed, PAULI_OUTCOME_DOMAIN, setting, 0);
        for repeat in 0..repeats {
            let draw = outcome_random.next_f64();
            let mut low = 0usize;
            let mut high = cumulative.len();
            while low < high {
                let middle = low + (high - low) / 2;
                if cumulative[middle] > draw {
                    high = middle;
                } else {
                    low = middle + 1;
                }
            }
            let index = low.min(dimension - 1);
            let start = (local_setting * repeats + repeat) * nqubits;
            for qubit in 0..nqubits {
                chunk.outcomes[start + qubit] = ((index >> (nqubits - 1 - qubit)) & 1) as u8;
            }
        }
    }
    Ok(())
}

impl Snapshots {
    pub fn expectation(
        &self,
        operator: &PauliOperator,
        blocks: usize,
        return_blocks: bool,
    ) -> Result<Vec<f64>, PauliError> {
        self.estimate_many(std::slice::from_ref(operator), blocks, return_blocks)
    }

    pub fn estimate_many(
        &self,
        operators: &[PauliOperator],
        blocks: usize,
        return_blocks: bool,
    ) -> Result<Vec<f64>, PauliError> {
        validate_linear_blocks(self.nsettings(), blocks)?;
        let setting_values = self.setting_values(operators.iter())?;
        self.reduce_estimates(setting_values, operators.len(), blocks, return_blocks)
    }

    pub fn estimate_many_refs(
        &self,
        operators: &[&PauliOperator],
        blocks: usize,
        return_blocks: bool,
    ) -> Result<Vec<f64>, PauliError> {
        validate_linear_blocks(self.nsettings(), blocks)?;
        let setting_values = self.setting_values(operators.iter().copied())?;
        self.reduce_estimates(setting_values, operators.len(), blocks, return_blocks)
    }

    fn reduce_estimates(
        &self,
        setting_values: Vec<f64>,
        operator_count: usize,
        blocks: usize,
        return_blocks: bool,
    ) -> Result<Vec<f64>, PauliError> {
        let block_values = block_means(&setting_values, self.nsettings(), operator_count, blocks)?;
        if return_blocks {
            Ok(block_values)
        } else {
            Ok(median_columns(&block_values, blocks, operator_count))
        }
    }

    pub fn energy(
        &self,
        operator: &PauliOperator,
        blocks: usize,
        return_blocks: bool,
    ) -> Result<Vec<f64>, PauliError> {
        self.expectation(operator, blocks, return_blocks)
    }

    pub fn rdm(
        &self,
        keep: &[usize],
        max_bytes: u128,
    ) -> Result<(usize, Vec<Complex64>), PauliError> {
        validate_subsystem(self.nqubits(), keep)?;
        let dimension =
            checked_dimension(keep.len(), "computing reduced density matrix dimension")?;
        let output_count = dimension
            .checked_mul(dimension)
            .ok_or(PauliError::Overflow {
                context: "estimating reduced density matrix output",
            })?;
        let requested =
            checked_bytes(output_count, 16, "estimating reduced density matrix output")?;
        check_limit(requested, max_bytes)?;
        let mut result = vec![Complex64::default(); output_count];
        let shot_count = self.shots()? as f64;
        for setting in 0..self.nsettings() {
            for repeat in 0..self.repeats() {
                let matrix = self.snapshot_rdm(setting, repeat, keep)?;
                for (target, value) in result.iter_mut().zip(matrix) {
                    *target += value / shot_count;
                }
            }
        }
        Ok((keep.len(), result))
    }

    pub fn renyi_entropy(
        &self,
        keep: &[usize],
        blocks: usize,
        method: u8,
        max_bytes: u128,
    ) -> Result<f64, PauliError> {
        validate_subsystem(self.nqubits(), keep)?;
        match method {
            0 => {
                if self.nsettings() < blocks.saturating_mul(2) || blocks == 0 {
                    return Err(PauliError::InvalidIndex {
                        context: "shadow Renyi-2 requires at least two settings per block",
                    });
                }
                let dimension = checked_dimension(keep.len(), "computing Renyi-2 dimension")?;
                let matrix_size = dimension
                    .checked_mul(dimension)
                    .ok_or(PauliError::Overflow {
                        context: "estimating Renyi-2 matrix storage",
                    })?;
                let matrix_bytes =
                    checked_bytes(matrix_size, 16, "estimating Renyi-2 matrix workspace")?;
                let block_bytes = checked_bytes(blocks, 8, "estimating Renyi-2 block workspace")?;
                let workspace = matrix_bytes
                    .checked_mul(3)
                    .and_then(|value| value.checked_add(block_bytes))
                    .ok_or(PauliError::Overflow {
                        context: "estimating Renyi-2 workspace",
                    })?;
                check_limit(workspace, max_bytes)?;
                let mut block_purities = Vec::with_capacity(blocks);
                let mut start = 0usize;
                for block in 0..blocks {
                    let size = balanced_block_size(self.nsettings(), blocks, block);
                    let mut sum = vec![Complex64::default(); matrix_size];
                    let mut diagonal = 0.0;
                    for setting in start..start + size {
                        let mut matrix = vec![Complex64::default(); matrix_size];
                        for repeat in 0..self.repeats() {
                            for (target, value) in matrix
                                .iter_mut()
                                .zip(self.snapshot_rdm(setting, repeat, keep)?)
                            {
                                *target += value / self.repeats() as f64;
                            }
                        }
                        for (target, entry) in sum.iter_mut().zip(&matrix) {
                            *target += *entry;
                        }
                        diagonal += trace_product(&matrix, &matrix);
                    }
                    block_purities.push(
                        (trace_product(&sum, &sum) - diagonal)
                            / (size as f64 * (size as f64 - 1.0)),
                    );
                    start += size;
                }
                let purity = median_scalar(&block_purities);
                if purity < 0.0 {
                    Ok(f64::NAN)
                } else if purity == 0.0 {
                    Ok(f64::INFINITY)
                } else {
                    Ok(-purity.ln())
                }
            }
            1 => {
                if self.protocol() != SnapshotProtocol::Pauli || self.repeats() < 2 {
                    return Err(PauliError::InvalidClifford {
                        context: "randomized-measurement Renyi-2 requires local Pauli repeats >= 2",
                    });
                }
                validate_linear_blocks(self.nsettings(), blocks)?;
                let workspace = checked_bytes(
                    self.nsettings(),
                    8,
                    "estimating randomized-measurement Renyi-2 workspace",
                )?
                .checked_add(checked_bytes(
                    blocks,
                    8,
                    "estimating randomized-measurement Renyi-2 block workspace",
                )?)
                .ok_or(PauliError::Overflow {
                    context: "estimating randomized-measurement Renyi-2 workspace",
                })?;
                check_limit(workspace, max_bytes)?;
                let k = keep.len();
                let subsystem_dimension =
                    checked_dimension(k, "computing randomized-measurement Renyi-2 dimension")?;
                let mut setting_purities = vec![0.0; self.nsettings()];
                let SnapshotData::Pauli {
                    nqubits,
                    repeats,
                    outcomes,
                    ..
                } = &self.data
                else {
                    unreachable!();
                };
                for (setting, purity) in setting_purities.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for left in 0..*repeats {
                        for right in 0..*repeats {
                            if left == right {
                                continue;
                            }
                            let left_start = (setting * *repeats + left) * *nqubits;
                            let right_start = (setting * *repeats + right) * *nqubits;
                            let distance = keep
                                .iter()
                                .filter(|qubit| {
                                    outcomes[left_start + **qubit]
                                        != outcomes[right_start + **qubit]
                                })
                                .count();
                            sum += (-2.0f64).powi(-(distance as i32));
                        }
                    }
                    *purity = subsystem_dimension as f64 * sum / (*repeats * (*repeats - 1)) as f64;
                }
                let block_values = block_means(&setting_purities, self.nsettings(), 1, blocks)?;
                let purity = median_scalar(&block_values);
                if purity < 0.0 {
                    Ok(f64::NAN)
                } else if purity == 0.0 {
                    Ok(f64::INFINITY)
                } else {
                    Ok(-purity.ln())
                }
            }
            _ => Err(PauliError::InvalidIndex {
                context: "Renyi-2 method must be shadow or randomized_measurement",
            }),
        }
    }

    pub fn energy_variance(
        &self,
        hamiltonian: &PauliOperator,
        blocks: usize,
        max_bytes: u128,
    ) -> Result<f64, PauliError> {
        if self.nsettings() < blocks.saturating_mul(2) || blocks == 0 {
            return Err(PauliError::InvalidIndex {
                context: "energy_variance requires at least two settings per block",
            });
        }
        if !hamiltonian.is_hermitian(0.0) {
            return Err(PauliError::NonHermitianExpectation);
        }
        let mut square = hamiltonian.multiply_with_limit(hamiltonian, max_bytes)?;
        // H is exactly Hermitian, so H^2 has real Pauli coefficients. Generic
        // multiplication can leave imaginary roundoff from anticommuting pairs.
        for term in &mut square.terms {
            term.coefficient.im = 0.0;
        }
        square.terms.retain(|term| term.coefficient.re != 0.0);
        let first = self.setting_values(std::slice::from_ref(hamiltonian))?;
        let second = self.setting_values(std::slice::from_ref(&square))?;
        let second_blocks = block_means(&second, self.nsettings(), 1, blocks)?;
        let mut values = Vec::with_capacity(blocks);
        let mut start = 0usize;
        for (block, second_value) in second_blocks.iter().copied().enumerate() {
            let size = balanced_block_size(self.nsettings(), blocks, block);
            let end = start + size;
            let sum = first[start..end].iter().sum::<f64>();
            let squares = first[start..end]
                .iter()
                .map(|value| value * value)
                .sum::<f64>();
            let cross_square = (sum * sum - squares) / (size as f64 * (size as f64 - 1.0));
            values.push(second_value - cross_square);
            start = end;
        }
        Ok(median_scalar(&values))
    }

    pub fn fidelity(
        &self,
        target: &StabilizerState,
        blocks: usize,
        return_blocks: bool,
    ) -> Result<Vec<f64>, PauliError> {
        if self.protocol() != SnapshotProtocol::Clifford {
            return Err(PauliError::InvalidClifford {
                context: "fidelity is supported only for global-Clifford snapshots",
            });
        }
        if target.nqubits != self.nqubits() {
            return Err(PauliError::IncompatibleQubitCounts {
                left: target.nqubits,
                right: self.nqubits(),
            });
        }
        validate_linear_blocks(self.nsettings(), blocks)?;
        let mut values = vec![0.0; self.nsettings()];
        for (setting, value) in values.iter_mut().enumerate() {
            for repeat in 0..self.repeats() {
                let snapshot = self.snapshot_stabilizer(setting, repeat)?;
                let overlap = stabilizer_overlap_squared(target, &snapshot)?;
                *value += ((2.0f64).powi(self.nqubits() as i32) + 1.0) * overlap - 1.0;
            }
            *value /= self.repeats() as f64;
        }
        let block_values = block_means(&values, self.nsettings(), 1, blocks)?;
        if return_blocks {
            Ok(block_values)
        } else {
            Ok(vec![median_scalar(&block_values)])
        }
    }

    pub fn fidelity_statevector(
        &self,
        target: &[Complex64],
        blocks: usize,
        return_blocks: bool,
    ) -> Result<Vec<f64>, PauliError> {
        if self.protocol() != SnapshotProtocol::Clifford {
            return Err(PauliError::InvalidClifford {
                context: "fidelity is supported only for global-Clifford snapshots",
            });
        }
        let dimension = checked_dimension(
            self.nqubits(),
            "computing dense global-Clifford fidelity dimension",
        )?;
        if target.len() != dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: dimension,
                actual: target.len(),
            });
        }
        if target
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(PauliError::InvalidClifford {
                context: "dense fidelity target entries must be finite",
            });
        }
        let norm = target.iter().map(|value| value.norm_sqr()).sum::<f64>();
        if !norm.is_finite() || (norm - 1.0).abs() > 1e-10 {
            return Err(PauliError::InvalidClifford {
                context: "dense fidelity target norm must be within 1e-10 of one",
            });
        }
        validate_linear_blocks(self.nsettings(), blocks)?;
        let scale = dimension as f64 + 1.0;
        let mut values = vec![0.0; self.nsettings()];
        let setting_value = |setting| {
            let mut value = 0.0;
            for repeat in 0..self.repeats() {
                let snapshot = self.snapshot_stabilizer(setting, repeat)?;
                let overlap = dense_stabilizer_overlap_squared(target, &snapshot)?;
                value += scale * overlap - 1.0;
            }
            Ok::<f64, PauliError>(value / self.repeats() as f64)
        };
        let parallel_work = self
            .nsettings()
            .saturating_mul(self.repeats())
            .saturating_mul(dimension);
        if parallel_work >= MIN_DENSE_FIDELITY_PARALLEL_WORK {
            values
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(setting, value)| {
                    *value = setting_value(setting)?;
                    Ok::<(), PauliError>(())
                })?;
        } else {
            for (setting, value) in values.iter_mut().enumerate() {
                *value = setting_value(setting)?;
            }
        }
        let block_values = block_means(&values, self.nsettings(), 1, blocks)?;
        if return_blocks {
            Ok(block_values)
        } else {
            Ok(vec![median_scalar(&block_values)])
        }
    }

    fn snapshot_stabilizer(
        &self,
        setting: usize,
        repeat: usize,
    ) -> Result<StabilizerState, PauliError> {
        let SnapshotData::Clifford {
            nqubits,
            repeats,
            frame_x_words,
            frame_z_words,
            frame_signs,
            outcomes,
            ..
        } = &self.data
        else {
            return Err(PauliError::InvalidClifford {
                context: "a stabilizer snapshot is available only for Clifford data",
            });
        };
        if setting >= self.nsettings() || repeat >= *repeats {
            return Err(PauliError::InvalidIndex {
                context: "snapshot setting or repeat is out of range",
            });
        }
        let word_count = packed_word_count(*nqubits);
        let outcome_start = (setting * *repeats + repeat) * *nqubits;
        let mut generators = Vec::with_capacity(*nqubits);
        for generator in 0..*nqubits {
            let row = setting * *nqubits + generator;
            let start = row * word_count;
            let sign = frame_signs[row]
                * if outcomes[outcome_start + generator] == 0 {
                    1
                } else {
                    -1
                };
            generators.push(SignedPauli::new(
                *nqubits,
                &frame_x_words[start..start + word_count],
                &frame_z_words[start..start + word_count],
                sign,
            )?);
        }
        Ok(StabilizerState {
            nqubits: *nqubits,
            generators,
        })
    }

    fn snapshot_rdm(
        &self,
        setting: usize,
        repeat: usize,
        keep: &[usize],
    ) -> Result<Vec<Complex64>, PauliError> {
        match &self.data {
            SnapshotData::Pauli {
                nqubits,
                repeats,
                bases,
                outcomes,
                ..
            } => {
                let start = (setting * *repeats + repeat) * *nqubits;
                local_pauli_rdm(
                    *nqubits,
                    keep,
                    &bases[setting * *nqubits..(setting + 1) * *nqubits],
                    &outcomes[start..start + *nqubits],
                )
            }
            SnapshotData::Clifford { .. } => {
                let snapshot = self.snapshot_stabilizer(setting, repeat)?;
                stabilizer_rdm(&snapshot, keep)
            }
        }
    }
}

fn sample_pauli_stabilizer(
    state: &StabilizerState,
    nsettings: usize,
    repeats: usize,
    seed: u64,
    max_bytes: u128,
) -> Result<Snapshots, PauliError> {
    let nqubits = state.nqubits;
    let basis_count = checked_product(&[nsettings, nqubits], "estimating Pauli basis storage")?;
    let outcome_count = checked_product(
        &[nsettings, repeats, nqubits],
        "estimating Pauli outcome storage",
    )?;
    let retained_count = basis_count
        .checked_add(outcome_count)
        .ok_or(PauliError::Overflow {
            context: "estimating Pauli snapshot storage",
        })?;
    let retained_bytes = checked_bytes(retained_count, 1, "estimating Pauli snapshot storage")?;
    check_limit(retained_bytes, max_bytes)?;
    let mut bases = vec![0u8; basis_count];
    let mut outcomes = vec![0u8; outcome_count];
    let word_count = packed_word_count(nqubits);
    let outcome_stride = repeats.checked_mul(nqubits).ok_or(PauliError::Overflow {
        context: "estimating Pauli outcome stride",
    })?;
    let worker_bytes = checked_bytes(
        nqubits
            .checked_mul(word_count)
            .ok_or(PauliError::Overflow {
                context: "estimating Pauli stabilizer worker workspace",
            })?,
        16,
        "estimating Pauli stabilizer worker workspace",
    )?;
    let worker_count = if nqubits >= 64 && nsettings >= 2 {
        bounded_shadow_worker_count(nsettings, retained_bytes, worker_bytes, max_bytes)
    } else {
        1
    };
    if worker_count > 1 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build()
            .expect("shadow worker count is positive");
        pool.install(|| {
            bases
                .par_chunks_mut(nqubits)
                .zip(outcomes.par_chunks_mut(outcome_stride))
                .enumerate()
                .map(|(setting, (setting_bases, setting_outcomes))| {
                    sample_pauli_stabilizer_setting(
                        state,
                        nqubits,
                        word_count,
                        repeats,
                        seed,
                        setting,
                        PauliSettingBuffers {
                            bases: setting_bases,
                            outcomes: setting_outcomes,
                        },
                    )
                })
                .collect::<Result<Vec<_>, PauliError>>()
        })?;
    } else {
        for setting in 0..nsettings {
            let basis_start = setting * nqubits;
            let outcome_start = setting * outcome_stride;
            sample_pauli_stabilizer_setting(
                state,
                nqubits,
                word_count,
                repeats,
                seed,
                setting,
                PauliSettingBuffers {
                    bases: &mut bases[basis_start..basis_start + nqubits],
                    outcomes: &mut outcomes[outcome_start..outcome_start + outcome_stride],
                },
            )?;
        }
    }
    Snapshots::from_pauli_measurements(nsettings, nqubits, repeats, &bases, &outcomes, max_bytes)
}

struct PauliSettingBuffers<'a> {
    bases: &'a mut [u8],
    outcomes: &'a mut [u8],
}

fn sample_pauli_stabilizer_setting(
    state: &StabilizerState,
    nqubits: usize,
    word_count: usize,
    repeats: usize,
    seed: u64,
    setting: usize,
    buffers: PauliSettingBuffers<'_>,
) -> Result<(), PauliError> {
    let mut basis_random = RandomStream::new(seed, PAULI_FRAME_DOMAIN, setting, 0);
    for basis in buffers.bases.iter_mut() {
        *basis = (basis_random.bounded(3)? + 1) as u8;
    }
    let mut x = vec![0u64; word_count];
    let mut z = vec![0u64; word_count];
    for repeat in 0..repeats {
        let mut current = state.clone();
        let mut random = RandomStream::new(seed, PAULI_OUTCOME_DOMAIN, setting, repeat);
        for qubit in 0..nqubits {
            x.fill(0);
            z.fill(0);
            match buffers.bases[qubit] {
                1 => x[qubit / 64] |= 1u64 << (qubit % 64),
                2 => {
                    x[qubit / 64] |= 1u64 << (qubit % 64);
                    z[qubit / 64] |= 1u64 << (qubit % 64);
                }
                3 => z[qubit / 64] |= 1u64 << (qubit % 64),
                _ => unreachable!(),
            }
            let measured = SignedPauli::new(nqubits, &x, &z, 1)?;
            let bit = current.measure_signed(&measured, &mut random)?;
            buffers.outcomes[repeat * nqubits + qubit] = bit;
        }
    }
    Ok(())
}

fn bounded_shadow_worker_count(
    nsettings: usize,
    retained_bytes: u128,
    worker_bytes: u128,
    max_bytes: u128,
) -> usize {
    let thread_count = rayon::current_num_threads().max(1);
    let memory_workers = max_bytes
        .saturating_sub(retained_bytes)
        .checked_div(worker_bytes.max(1))
        .unwrap_or(0)
        .max(1);
    thread_count
        .min(nsettings.max(1))
        .min(memory_workers.min(usize::MAX as u128) as usize)
}

struct CliffordSettingBuffers<'a> {
    frame_x_words: &'a mut [u64],
    frame_z_words: &'a mut [u64],
    frame_signs: &'a mut [i8],
    outcomes: &'a mut [u8],
}

fn sample_clifford_stabilizer_setting(
    state: &StabilizerState,
    nqubits: usize,
    word_count: usize,
    repeats: usize,
    seed: u64,
    setting: usize,
    buffers: CliffordSettingBuffers<'_>,
) -> Result<(), PauliError> {
    let mut frame_random = RandomStream::new(seed, CLIFFORD_FRAME_DOMAIN, setting, 0);
    let frame = sample_symplectic_frame(nqubits, &mut frame_random)?;
    let mut sign_random = RandomStream::new(seed, CLIFFORD_SIGN_DOMAIN, setting, 0);
    for (generator, vector) in frame.iter().enumerate() {
        let start = generator * word_count;
        buffers.frame_x_words[start..start + word_count].copy_from_slice(&vector.x_words);
        buffers.frame_z_words[start..start + word_count].copy_from_slice(&vector.z_words);
        buffers.frame_signs[generator] = if sign_random.next_u64() & 1 == 0 {
            1
        } else {
            -1
        };
    }
    for repeat in 0..repeats {
        let mut current = state.clone();
        let mut random = RandomStream::new(seed, CLIFFORD_OUTCOME_DOMAIN, setting, repeat);
        for generator in 0..nqubits {
            let start = generator * word_count;
            let measured = SignedPauli::new(
                nqubits,
                &buffers.frame_x_words[start..start + word_count],
                &buffers.frame_z_words[start..start + word_count],
                buffers.frame_signs[generator],
            )?;
            let bit = current.measure_signed(&measured, &mut random)?;
            buffers.outcomes[repeat * nqubits + generator] = bit;
        }
    }
    Ok(())
}

fn sample_clifford_statevector(
    state: &[Complex64],
    nqubits: usize,
    nsettings: usize,
    repeats: usize,
    seed: u64,
    max_bytes: u128,
) -> Result<Snapshots, PauliError> {
    let dimension = state.len();
    let word_count = packed_word_count(nqubits);
    let frame_rows = checked_product(&[nsettings, nqubits], "estimating Clifford frame rows")?;
    let frame_words = checked_product(
        &[frame_rows, word_count],
        "estimating Clifford frame storage",
    )?;
    let outcome_count = checked_product(
        &[nsettings, repeats, nqubits],
        "estimating Clifford outcome storage",
    )?;
    let retained_bytes = checked_bytes(frame_words * 2, 8, "estimating Clifford frame storage")?
        .checked_add(checked_bytes(
            frame_rows,
            1,
            "estimating Clifford sign storage",
        )?)
        .and_then(|value| {
            value.checked_add(
                checked_bytes(outcome_count, 1, "estimating Clifford outcome storage").ok()?,
            )
        })
        .ok_or(PauliError::Overflow {
            context: "estimating Clifford snapshot storage",
        })?;
    let worker_bytes = checked_bytes(dimension, 32, "estimating Clifford state scratch")?;
    let parallel_work = nsettings
        .saturating_mul(repeats)
        .saturating_mul(nqubits)
        .saturating_mul(dimension);
    let worker_count = if nsettings >= 2 && parallel_work >= MIN_CLIFFORD_STATEVECTOR_PARALLEL_WORK
    {
        bounded_shadow_worker_count(nsettings, retained_bytes, worker_bytes, max_bytes)
    } else {
        1
    };
    let requested = retained_bytes
        .checked_add(worker_bytes.checked_mul(worker_count as u128).ok_or(
            PauliError::Overflow {
                context: "estimating parallel Clifford sampler memory",
            },
        )?)
        .ok_or(PauliError::Overflow {
            context: "estimating parallel Clifford sampler memory",
        })?;
    check_limit(requested, max_bytes)?;
    let mut frame_x_words = vec![0u64; frame_words];
    let mut frame_z_words = vec![0u64; frame_words];
    let mut frame_signs = vec![1i8; frame_rows];
    let mut outcomes = vec![0u8; outcome_count];
    if nqubits == 0 {
        return Snapshots::from_clifford_measurements(
            nsettings,
            nqubits,
            repeats,
            &frame_x_words,
            &frame_z_words,
            &frame_signs,
            &outcomes,
            max_bytes,
        );
    }
    let frame_words_per_setting = checked_product(
        &[nqubits, word_count],
        "estimating Clifford frame setting storage",
    )?;
    let outcome_stride =
        checked_product(&[repeats, nqubits], "estimating Clifford outcome stride")?;
    let settings_per_chunk = nsettings.div_ceil(worker_count);
    let frame_chunk_words = checked_product(
        &[settings_per_chunk, frame_words_per_setting],
        "estimating Clifford frame chunk",
    )?;
    let sign_chunk_len = checked_product(
        &[settings_per_chunk, nqubits],
        "estimating Clifford sign chunk",
    )?;
    let outcome_chunk_len = checked_product(
        &[settings_per_chunk, outcome_stride],
        "estimating Clifford outcome chunk",
    )?;
    let mut sample_chunks = || {
        frame_x_words
            .par_chunks_mut(frame_chunk_words)
            .zip(frame_z_words.par_chunks_mut(frame_chunk_words))
            .zip(frame_signs.par_chunks_mut(sign_chunk_len))
            .zip(outcomes.par_chunks_mut(outcome_chunk_len))
            .enumerate()
            .map(|(chunk, (((frame_x, frame_z), signs), chunk_outcomes))| {
                let chunk_settings = signs.len() / nqubits;
                sample_clifford_statevector_chunk(
                    state,
                    nqubits,
                    dimension,
                    word_count,
                    repeats,
                    seed,
                    CliffordStatevectorChunk {
                        setting_start: chunk * settings_per_chunk,
                        chunk_settings,
                        frame_x_words: frame_x,
                        frame_z_words: frame_z,
                        frame_signs: signs,
                        outcomes: chunk_outcomes,
                    },
                )
            })
            .collect::<Result<Vec<_>, PauliError>>()
    };
    if worker_count > 1 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build()
            .expect("shadow worker count is positive");
        pool.install(sample_chunks)?;
    } else {
        sample_chunks()?;
    }

    Snapshots::from_clifford_measurements(
        nsettings,
        nqubits,
        repeats,
        &frame_x_words,
        &frame_z_words,
        &frame_signs,
        &outcomes,
        max_bytes,
    )
}

struct CliffordStatevectorChunk<'a> {
    setting_start: usize,
    chunk_settings: usize,
    frame_x_words: &'a mut [u64],
    frame_z_words: &'a mut [u64],
    frame_signs: &'a mut [i8],
    outcomes: &'a mut [u8],
}

fn sample_clifford_statevector_chunk(
    state: &[Complex64],
    nqubits: usize,
    dimension: usize,
    word_count: usize,
    repeats: usize,
    seed: u64,
    chunk: CliffordStatevectorChunk<'_>,
) -> Result<(), PauliError> {
    let mut current = vec![Complex64::default(); dimension];
    let mut scratch = vec![Complex64::default(); dimension];
    for local_setting in 0..chunk.chunk_settings {
        let setting = chunk.setting_start + local_setting;
        let mut frame_random = RandomStream::new(seed, CLIFFORD_FRAME_DOMAIN, setting, 0);
        let frame = sample_symplectic_frame(nqubits, &mut frame_random)?;
        let mut sign_random = RandomStream::new(seed, CLIFFORD_SIGN_DOMAIN, setting, 0);
        for (generator, vector) in frame.iter().enumerate() {
            let row = local_setting * nqubits + generator;
            let start = row * word_count;
            chunk.frame_x_words[start..start + word_count].copy_from_slice(&vector.x_words);
            chunk.frame_z_words[start..start + word_count].copy_from_slice(&vector.z_words);
            chunk.frame_signs[row] = if sign_random.next_u64() & 1 == 0 {
                1
            } else {
                -1
            };
        }
        for repeat in 0..repeats {
            current.copy_from_slice(state);
            let mut outcome_random =
                RandomStream::new(seed, CLIFFORD_OUTCOME_DOMAIN, setting, repeat);
            for generator in 0..nqubits {
                let row = local_setting * nqubits + generator;
                let start = row * word_count;
                let action = StatevectorPauliAction::from_packed(
                    nqubits,
                    &chunk.frame_x_words[start..start + word_count],
                    &chunk.frame_z_words[start..start + word_count],
                    f64::from(chunk.frame_signs[row]),
                )?;
                let expectation = action.apply_into_and_expectation(&current, &mut scratch)?;
                let probability_zero = ((1.0 + expectation) * 0.5).clamp(0.0, 1.0);
                let bit = if outcome_random.next_f64() < probability_zero {
                    0
                } else {
                    1
                };
                let eigenvalue = if bit == 0 { 1.0 } else { -1.0 };
                action.project_from_action_in_place(
                    &mut current,
                    &scratch,
                    eigenvalue,
                    expectation,
                )?;
                chunk.outcomes[(local_setting * repeats + repeat) * nqubits + generator] = bit;
            }
        }
    }
    Ok(())
}

fn sample_clifford_stabilizer(
    state: &StabilizerState,
    nsettings: usize,
    repeats: usize,
    seed: u64,
    max_bytes: u128,
) -> Result<Snapshots, PauliError> {
    let nqubits = state.nqubits;
    let word_count = packed_word_count(nqubits);
    let frame_rows = checked_product(&[nsettings, nqubits], "estimating Clifford frame rows")?;
    let frame_words = checked_product(
        &[frame_rows, word_count],
        "estimating Clifford frame storage",
    )?;
    let outcome_count = checked_product(
        &[nsettings, repeats, nqubits],
        "estimating Clifford outcome storage",
    )?;
    let requested = checked_bytes(frame_words * 2, 8, "estimating Clifford frame storage")?
        .checked_add(checked_bytes(
            frame_rows,
            1,
            "estimating Clifford sign storage",
        )?)
        .and_then(|value| {
            value.checked_add(
                checked_bytes(outcome_count, 1, "estimating Clifford outcome storage").ok()?,
            )
        })
        .ok_or(PauliError::Overflow {
            context: "estimating Clifford sampler memory",
        })?;
    check_limit(requested, max_bytes)?;
    let mut frame_x_words = vec![0u64; frame_words];
    let mut frame_z_words = vec![0u64; frame_words];
    let mut frame_signs = vec![1i8; frame_rows];
    let mut outcomes = vec![0u8; outcome_count];
    let rows_words = nqubits
        .checked_mul(word_count)
        .ok_or(PauliError::Overflow {
            context: "estimating Clifford frame row workspace",
        })?;
    let outcome_stride = repeats.checked_mul(nqubits).ok_or(PauliError::Overflow {
        context: "estimating Clifford outcome stride",
    })?;
    let worker_bytes = checked_bytes(
        nqubits
            .checked_mul(word_count)
            .ok_or(PauliError::Overflow {
                context: "estimating Clifford stabilizer worker workspace",
            })?,
        16,
        "estimating Clifford stabilizer worker workspace",
    )?;
    let parallel_work = nsettings
        .saturating_mul(repeats)
        .saturating_mul(nqubits)
        .saturating_mul(nqubits);
    let worker_count = if nsettings >= 2 && parallel_work >= MIN_CLIFFORD_STABILIZER_PARALLEL_WORK {
        bounded_shadow_worker_count(nsettings, requested, worker_bytes, max_bytes)
    } else {
        1
    };
    if worker_count > 1 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build()
            .expect("shadow worker count is positive");
        pool.install(|| {
            frame_x_words
                .par_chunks_mut(rows_words)
                .zip(frame_z_words.par_chunks_mut(rows_words))
                .zip(frame_signs.par_chunks_mut(nqubits))
                .zip(outcomes.par_chunks_mut(outcome_stride))
                .enumerate()
                .map(
                    |(setting, (((frame_x, frame_z), signs), setting_outcomes))| {
                        sample_clifford_stabilizer_setting(
                            state,
                            nqubits,
                            word_count,
                            repeats,
                            seed,
                            setting,
                            CliffordSettingBuffers {
                                frame_x_words: frame_x,
                                frame_z_words: frame_z,
                                frame_signs: signs,
                                outcomes: setting_outcomes,
                            },
                        )
                    },
                )
                .collect::<Result<Vec<_>, PauliError>>()
        })?;
    } else {
        for setting in 0..nsettings {
            let frame_start = setting * rows_words;
            let outcome_start = setting * outcome_stride;
            sample_clifford_stabilizer_setting(
                state,
                nqubits,
                word_count,
                repeats,
                seed,
                setting,
                CliffordSettingBuffers {
                    frame_x_words: &mut frame_x_words[frame_start..frame_start + rows_words],
                    frame_z_words: &mut frame_z_words[frame_start..frame_start + rows_words],
                    frame_signs: &mut frame_signs[setting * nqubits..(setting + 1) * nqubits],
                    outcomes: &mut outcomes[outcome_start..outcome_start + outcome_stride],
                },
            )?;
        }
    }
    Snapshots::from_clifford_measurements(
        nsettings,
        nqubits,
        repeats,
        &frame_x_words,
        &frame_z_words,
        &frame_signs,
        &outcomes,
        max_bytes,
    )
}

fn apply_h_in_place(state: &mut [Complex64], qubit: usize, nqubits: usize) {
    let mask = 1usize << (nqubits - 1 - qubit);
    let scale = 2.0f64.sqrt().recip();
    for index in 0..state.len() {
        if index & mask == 0 {
            let other = index | mask;
            let left = state[index];
            let right = state[other];
            state[index] = (left + right) * scale;
            state[other] = (left - right) * scale;
        }
    }
}

fn apply_y_measurement_rotation(state: &mut [Complex64], qubit: usize, nqubits: usize) {
    let mask = 1usize << (nqubits - 1 - qubit);
    let scale = 2.0f64.sqrt().recip();
    let minus_i = Complex64::new(0.0, -1.0);
    for index in 0..state.len() {
        if index & mask == 0 {
            let other = index | mask;
            let left = state[index];
            let right = state[other] * minus_i;
            state[index] = (left + right) * scale;
            state[other] = (left - right) * scale;
        }
    }
}

fn checked_dimension(nqubits: usize, context: &'static str) -> Result<usize, PauliError> {
    if nqubits >= usize::BITS as usize {
        return Err(PauliError::Overflow { context });
    }
    1usize
        .checked_shl(nqubits as u32)
        .ok_or(PauliError::Overflow { context })
}

fn validate_subsystem(nqubits: usize, keep: &[usize]) -> Result<(), PauliError> {
    let mut seen = vec![false; nqubits];
    for qubit in keep {
        if *qubit >= nqubits {
            return Err(PauliError::InvalidIndex {
                context: "subsystem qubit is outside the snapshot width",
            });
        }
        if seen[*qubit] {
            return Err(PauliError::InvalidIndex {
                context: "subsystem qubits must be unique",
            });
        }
        seen[*qubit] = true;
    }
    Ok(())
}

fn validate_linear_blocks(nsettings: usize, blocks: usize) -> Result<(), PauliError> {
    validate_count(blocks, "blocks must be positive")?;
    if blocks > nsettings {
        return Err(PauliError::InvalidIndex {
            context: "blocks cannot exceed nsettings",
        });
    }
    Ok(())
}

fn balanced_block_size(nsettings: usize, blocks: usize, block: usize) -> usize {
    let base = nsettings / blocks;
    base + usize::from(block < nsettings % blocks)
}

fn block_means(
    values: &[f64],
    nsettings: usize,
    nobservables: usize,
    blocks: usize,
) -> Result<Vec<f64>, PauliError> {
    if values.len() != nsettings * nobservables {
        return Err(PauliError::InvalidStructureLength {
            expected: nsettings * nobservables,
            actual: values.len(),
        });
    }
    validate_linear_blocks(nsettings, blocks)?;
    let mut result = Vec::with_capacity(blocks * nobservables);
    let mut start = 0usize;
    for block in 0..blocks {
        let size = balanced_block_size(nsettings, blocks, block);
        let end = start + size;
        for observable in 0..nobservables {
            let mut sum = 0.0;
            for setting in start..end {
                sum += values[setting * nobservables + observable];
            }
            result.push(sum / size as f64);
        }
        start = end;
    }
    Ok(result)
}

fn median_columns(values: &[f64], rows: usize, columns: usize) -> Vec<f64> {
    (0..columns)
        .map(|column| {
            let mut column_values = (0..rows)
                .map(|row| values[row * columns + column])
                .collect::<Vec<_>>();
            column_values.sort_by(f64::total_cmp);
            if rows % 2 == 1 {
                column_values[rows / 2]
            } else {
                (column_values[rows / 2 - 1] + column_values[rows / 2]) * 0.5
            }
        })
        .collect()
}

fn median_scalar(values: &[f64]) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    if ordered.len() % 2 == 1 {
        ordered[ordered.len() / 2]
    } else {
        (ordered[ordered.len() / 2 - 1] + ordered[ordered.len() / 2]) * 0.5
    }
}

fn trace_product(left: &[Complex64], right: &[Complex64]) -> f64 {
    let dimension = (left.len() as f64).sqrt() as usize;
    let mut result = 0.0;
    for row in 0..dimension {
        for column in 0..dimension {
            result += (left[row * dimension + column] * right[column * dimension + row]).re;
        }
    }
    result
}

fn local_pauli_rdm(
    nqubits: usize,
    keep: &[usize],
    bases: &[u8],
    outcomes: &[u8],
) -> Result<Vec<Complex64>, PauliError> {
    let mut matrix = vec![Complex64::new(1.0, 0.0)];
    let mut current_dimension = 1usize;
    for qubit in keep {
        let sign = if outcomes[*qubit] == 0 { 1.0 } else { -1.0 };
        let local = match bases[*qubit] {
            1 => [
                Complex64::new(0.5, 0.0),
                Complex64::new(1.5 * sign, 0.0),
                Complex64::new(1.5 * sign, 0.0),
                Complex64::new(0.5, 0.0),
            ],
            2 => [
                Complex64::new(0.5, 0.0),
                Complex64::new(0.0, -1.5 * sign),
                Complex64::new(0.0, 1.5 * sign),
                Complex64::new(0.5, 0.0),
            ],
            3 => [
                Complex64::new(0.5 + 1.5 * sign, 0.0),
                Complex64::default(),
                Complex64::default(),
                Complex64::new(0.5 - 1.5 * sign, 0.0),
            ],
            _ => {
                return Err(PauliError::InvalidCode {
                    code: bases[*qubit],
                    index: *qubit,
                })
            }
        };
        let next_dimension = current_dimension
            .checked_mul(2)
            .ok_or(PauliError::Overflow {
                context: "computing local Pauli RDM dimension",
            })?;
        let mut next = vec![Complex64::default(); next_dimension * next_dimension];
        for row in 0..current_dimension {
            for column in 0..current_dimension {
                let value = matrix[row * current_dimension + column];
                for local_row in 0..2 {
                    for local_column in 0..2 {
                        next[(row * 2 + local_row) * next_dimension + column * 2 + local_column] +=
                            value * local[local_row * 2 + local_column];
                    }
                }
            }
        }
        matrix = next;
        current_dimension = next_dimension;
    }
    let _ = nqubits;
    Ok(matrix)
}

fn add_pauli_matrix(
    matrix: &mut [Complex64],
    nqubits: usize,
    word: &PauliWord,
    sign: i8,
    scale: f64,
) -> Result<(), PauliError> {
    let dimension = checked_dimension(nqubits, "computing Pauli matrix dimension")?;
    if matrix.len() != dimension * dimension {
        return Err(PauliError::InvalidStructureLength {
            expected: dimension * dimension,
            actual: matrix.len(),
        });
    }
    let action = StatevectorPauliAction::from_packed(
        nqubits,
        word.x_words(),
        word.z_words(),
        f64::from(sign),
    )?;
    for column in 0..dimension {
        let row = column ^ action.x_mask();
        let mut value = action.phase();
        if (action.z_mask() & column).count_ones() & 1 != 0 {
            value = -value;
        }
        matrix[row * dimension + column] += value * scale;
    }
    Ok(())
}

fn stabilizer_rdm(state: &StabilizerState, keep: &[usize]) -> Result<Vec<Complex64>, PauliError> {
    validate_subsystem(state.nqubits, keep)?;
    let dimension = checked_dimension(keep.len(), "computing stabilizer RDM dimension")?;
    let mut kept = vec![false; state.nqubits];
    for qubit in keep {
        kept[*qubit] = true;
    }
    let mut constraints = Vec::new();
    for (qubit, is_kept) in kept.iter().copied().enumerate() {
        if is_kept {
            continue;
        }
        constraints.push(
            state
                .generators
                .iter()
                .map(|generator| {
                    let x = (generator.x_words[qubit / 64] >> (qubit % 64)) & 1;
                    x as u8
                })
                .collect::<Vec<_>>(),
        );
        constraints.push(
            state
                .generators
                .iter()
                .map(|generator| ((generator.z_words[qubit / 64] >> (qubit % 64)) & 1) as u8)
                .collect::<Vec<_>>(),
        );
    }
    let nullspace = nullspace_basis(&constraints, state.generators.len());
    if nullspace.len() >= usize::BITS as usize {
        return Err(PauliError::Overflow {
            context: "enumerating the restricted stabilizer group",
        });
    }
    let group_count = 1usize << nullspace.len();
    let mut rho = vec![Complex64::default(); dimension * dimension];
    for mask in 0..group_count {
        let mut product = signed_identity(state.nqubits);
        for (basis_index, coefficients) in nullspace.iter().enumerate() {
            if mask & (1usize << basis_index) == 0 {
                continue;
            }
            for (generator_index, selected) in coefficients.iter().enumerate() {
                if *selected {
                    product =
                        product.multiply(&state.generators[generator_index], state.nqubits)?;
                }
            }
        }
        let word = product.word(state.nqubits)?;
        let codes = keep
            .iter()
            .map(|qubit| word.code_at(*qubit))
            .collect::<Vec<_>>();
        let restricted = PauliWord::from_codes(keep.len(), &codes)?;
        add_pauli_matrix(
            &mut rho,
            keep.len(),
            &restricted,
            product.sign,
            1.0 / dimension as f64,
        )?;
    }
    let dimension_full = checked_dimension(state.nqubits, "computing Clifford shadow dimension")?;
    let traced_dimension = dimension_full / dimension;
    let scale = dimension_full as f64 + 1.0;
    for (index, value) in rho.iter_mut().enumerate() {
        *value *= scale;
        if index / dimension == index % dimension {
            *value -= traced_dimension as f64;
        }
    }
    Ok(rho)
}

fn dense_stabilizer_overlap_squared(
    target: &[Complex64],
    state: &StabilizerState,
) -> Result<f64, PauliError> {
    let dimension = checked_dimension(
        state.nqubits,
        "computing dense stabilizer-overlap dimension",
    )?;
    if target.len() != dimension {
        return Err(PauliError::InvalidStructureLength {
            expected: dimension,
            actual: target.len(),
        });
    }
    let x_basis = stabilizer_x_basis(state)?;
    let x_rank = x_basis.len();
    let support_size = checked_dimension(x_rank, "enumerating dense stabilizer support")?;
    let z_generators = stabilizer_z_generators(state)?;
    let equations = z_generators
        .iter()
        .map(|generator| {
            (0..state.nqubits)
                .map(|qubit| ((generator.z_words[qubit / 64] >> (qubit % 64)) & 1) as u8)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let right_hand_side = z_generators
        .iter()
        .map(|generator| u8::from(generator.sign < 0))
        .collect::<Vec<_>>();
    let base_solution = solve_binary_affine(&equations, &right_hand_side, state.nqubits)?;
    let mut base_packed = 0u64;
    for (qubit, bit) in base_solution.iter().copied().enumerate() {
        if bit != 0 {
            base_packed |= 1u64 << qubit;
        }
    }
    let base_index = packed_to_dense_index(&[base_packed], state.nqubits);
    let normalization = (support_size as f64).sqrt().recip();
    let mut overlap = Complex64::default();
    let mut product = signed_identity(state.nqubits);
    let mut previous_gray = 0usize;
    for mask in 0..support_size {
        let gray = mask ^ (mask >> 1);
        if mask != 0 {
            let changed = (gray ^ previous_gray).trailing_zeros() as usize;
            product.multiply_in_place(&x_basis[changed])?;
        }
        previous_gray = gray;
        let dense_index = base_index ^ packed_to_dense_index(&product.x_words, state.nqubits);
        let amplitude = signed_pauli_amplitude(&product, base_packed, normalization);
        overlap += target[dense_index].conj() * amplitude;
    }
    let result = overlap.norm_sqr();
    if !result.is_finite() {
        return Err(PauliError::InvalidClifford {
            context: "dense stabilizer overlap became non-finite",
        });
    }
    Ok(result.max(0.0))
}

fn stabilizer_x_basis(state: &StabilizerState) -> Result<Vec<SignedPauli>, PauliError> {
    let mut basis = (0..state.nqubits)
        .map(|_| None::<SignedPauli>)
        .collect::<Vec<_>>();
    for generator in &state.generators {
        let mut row = generator.clone();
        while let Some(pivot) = first_x_bit(&row.x_words) {
            if let Some(existing) = basis[pivot].as_ref() {
                row.multiply_in_place(existing)?;
            } else {
                basis[pivot] = Some(row);
                break;
            }
        }
    }
    Ok(basis.into_iter().flatten().collect())
}

fn stabilizer_z_generators(state: &StabilizerState) -> Result<Vec<SignedPauli>, PauliError> {
    let equations = (0..state.nqubits)
        .map(|qubit| {
            state
                .generators
                .iter()
                .map(|generator| ((generator.x_words[qubit / 64] >> (qubit % 64)) & 1) as u8)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    nullspace_basis(&equations, state.generators.len())
        .into_iter()
        .map(|coefficients| {
            let mut product = signed_identity(state.nqubits);
            for (generator, selected) in state.generators.iter().zip(coefficients) {
                if selected {
                    product.multiply_in_place(generator)?;
                }
            }
            if product.x_words.iter().any(|value| *value != 0) {
                return Err(PauliError::InvalidClifford {
                    context: "stabilizer X-nullspace reduction returned an X component",
                });
            }
            Ok(product)
        })
        .collect()
}

fn first_x_bit(x_words: &[u64]) -> Option<usize> {
    x_words.iter().enumerate().find_map(|(index, value)| {
        (*value != 0).then_some(index * 64 + value.trailing_zeros() as usize)
    })
}

fn solve_binary_affine(
    equations: &[Vec<u8>],
    right_hand_side: &[u8],
    dimension: usize,
) -> Result<Vec<u8>, PauliError> {
    if equations.len() != right_hand_side.len()
        || equations.iter().any(|row| row.len() != dimension)
        || right_hand_side.iter().any(|value| *value > 1)
    {
        return Err(PauliError::InvalidClifford {
            context: "stabilizer affine-support equations have an invalid shape",
        });
    }
    let mut matrix = equations
        .iter()
        .zip(right_hand_side)
        .map(|(row, value)| {
            let mut augmented = row.clone();
            augmented.push(*value);
            augmented
        })
        .collect::<Vec<_>>();
    let mut pivots = Vec::new();
    let mut rank = 0usize;
    for column in 0..dimension {
        let Some(source) = (rank..matrix.len()).find(|row| matrix[*row][column] != 0) else {
            continue;
        };
        matrix.swap(rank, source);
        let pivot_values = matrix[rank][column..=dimension].to_vec();
        for (row, current) in matrix.iter_mut().enumerate() {
            if row != rank && current[column] != 0 {
                for (value, pivot) in current[column..=dimension].iter_mut().zip(&pivot_values) {
                    *value ^= *pivot;
                }
            }
        }
        pivots.push(column);
        rank += 1;
        if rank == matrix.len() {
            break;
        }
    }
    if matrix[rank..]
        .iter()
        .any(|row| row[..dimension].iter().all(|value| *value == 0) && row[dimension] != 0)
    {
        return Err(PauliError::InvalidClifford {
            context: "stabilizer affine-support equations are inconsistent",
        });
    }
    let mut solution = vec![0u8; dimension];
    for (row, pivot) in pivots.into_iter().enumerate() {
        solution[pivot] = matrix[row][dimension];
    }
    Ok(solution)
}

fn packed_to_dense_index(x_words: &[u64], nqubits: usize) -> usize {
    if nqubits == 0 {
        return 0;
    }
    (x_words[0].reverse_bits() >> (64 - nqubits)) as usize
}

fn signed_pauli_amplitude(pauli: &SignedPauli, base_packed: u64, normalization: f64) -> Complex64 {
    let y_count = pauli
        .x_words
        .iter()
        .zip(&pauli.z_words)
        .map(|(x, z)| (x & z).count_ones())
        .sum::<u32>();
    let mut amplitude = match y_count % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    } * f64::from(pauli.sign)
        * normalization;
    if (pauli.z_words.first().copied().unwrap_or(0) & base_packed).count_ones() & 1 != 0 {
        amplitude = -amplitude;
    }
    amplitude
}

fn stabilizer_overlap_squared(
    left: &StabilizerState,
    right: &StabilizerState,
) -> Result<f64, PauliError> {
    if left.nqubits != right.nqubits {
        return Err(PauliError::IncompatibleQubitCounts {
            left: left.nqubits,
            right: right.nqubits,
        });
    }
    if left.nqubits == 0 {
        return Ok(1.0);
    }
    let dimension = 2 * left.nqubits;
    let mut constraints =
        vec![vec![0u8; left.generators.len() + right.generators.len()]; dimension];
    for (coordinate, row) in constraints.iter_mut().enumerate() {
        for (index, generator) in left.generators.iter().enumerate() {
            let words = if coordinate < left.nqubits {
                &generator.x_words
            } else {
                &generator.z_words
            };
            let bit = if coordinate < left.nqubits {
                coordinate
            } else {
                coordinate - left.nqubits
            };
            row[index] = ((words[bit / 64] >> (bit % 64)) & 1) as u8;
        }
        for (index, generator) in right.generators.iter().enumerate() {
            let offset = left.generators.len();
            let words = if coordinate < right.nqubits {
                &generator.x_words
            } else {
                &generator.z_words
            };
            let bit = if coordinate < right.nqubits {
                coordinate
            } else {
                coordinate - right.nqubits
            };
            row[offset + index] = ((words[bit / 64] >> (bit % 64)) & 1) as u8;
        }
    }
    let relations = nullspace_basis(&constraints, left.generators.len() + right.generators.len());
    for relation in &relations {
        let mut left_product = signed_identity(left.nqubits);
        let mut right_product = signed_identity(left.nqubits);
        for (index, selected) in relation.iter().enumerate() {
            if !*selected {
                continue;
            }
            if index < left.generators.len() {
                left_product = left_product.multiply(&left.generators[index], left.nqubits)?;
            } else {
                right_product = right_product.multiply(
                    &right.generators[index - left.generators.len()],
                    left.nqubits,
                )?;
            }
        }
        if left_product.word(left.nqubits)? != right_product.word(left.nqubits)?
            || left_product.sign != right_product.sign
        {
            return Ok(0.0);
        }
    }
    let rank = relations.len();
    let exponent = rank as i32 - left.nqubits as i32;
    Ok(2.0f64.powi(exponent))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::ThreadPoolBuilder;

    #[test]
    fn energy_variance_preserves_hermitian_square() {
        let snapshots =
            Snapshots::from_pauli_measurements(2, 2, 1, &[3, 3, 3, 3], &[0, 0, 0, 0], u128::MAX)
                .expect("Z measurements");
        for scale in [1.0, 1.0e-7] {
            let hamiltonian = PauliOperator::from_terms(
                2,
                &[vec![3, 1], vec![0, 1], vec![3, 2], vec![0, 2]],
                &[0.4, 0.3, 0.4, 0.5].map(|value| Complex64::new(value * scale, 0.0)),
            )
            .expect("Hermitian Hamiltonian");
            let variance = snapshots
                .energy_variance(&hamiltonian, 1, u128::MAX)
                .expect("variance of a Hermitian Hamiltonian");
            let expected = 2.58 * scale * scale;
            assert!((variance - expected).abs() < expected * 1.0e-12);
        }
        let nonhermitian =
            PauliOperator::from_terms(2, &[vec![3, 0]], &[Complex64::new(1.0, 1.0e-18)])
                .expect("complex Hamiltonian");
        assert!(matches!(
            snapshots.energy_variance(&nonhermitian, 1, u128::MAX),
            Err(PauliError::NonHermitianExpectation)
        ));
    }

    #[test]
    fn packed_nullspace_matches_byte_reference() {
        let rows = vec![
            vec![1, 0, 1, 1, 0, 0, 1, 0, 1],
            vec![0, 1, 1, 0, 1, 0, 1, 1, 0],
        ];
        let packed_rows = rows
            .iter()
            .map(|row| {
                let mut packed = PackedBits::zero(row.len());
                for (index, value) in row.iter().copied().enumerate() {
                    if value != 0 {
                        packed.set(index);
                    }
                }
                packed
            })
            .collect::<Vec<_>>();
        let expected = nullspace_basis(&rows, rows[0].len());
        let actual = nullspace_basis_packed(packed_rows, rows[0].len());
        assert_eq!(actual.len(), expected.len());
        for (packed, boolean) in actual.iter().zip(expected) {
            assert_eq!(
                (0..rows[0].len())
                    .map(|index| packed.get(index))
                    .collect::<Vec<_>>(),
                boolean
            );
        }
    }

    #[test]
    fn in_place_stabilizer_product_matches_allocating_product() {
        for left_codes in [[1, 2, 3], [3, 1, 0], [2, 3, 1]] {
            for right_codes in [[3, 2, 1], [1, 0, 3], [2, 1, 2]] {
                let left_word = PauliWord::from_codes(3, &left_codes).expect("left word");
                let right_word = PauliWord::from_codes(3, &right_codes).expect("right word");
                let left = SignedPauli::from_word(&left_word, -1).expect("left pauli");
                let right = SignedPauli::from_word(&right_word, 1).expect("right pauli");
                let expected = left.multiply(&right, 3);
                let mut actual = left;
                let actual_result = actual.multiply_in_place(&right);
                match (expected, actual_result) {
                    (Ok(expected), Ok(())) => {
                        assert_eq!(actual.x_words, expected.x_words);
                        assert_eq!(actual.z_words, expected.z_words);
                        assert_eq!(actual.sign, expected.sign);
                    }
                    (Err(_), Err(_)) => {}
                    _ => panic!("allocating and in-place products disagree"),
                }
            }
        }
    }

    #[test]
    fn stabilizer_sampling_replays_across_bounded_worker_counts() {
        let state = StabilizerState::zero_state(16).expect("zero state");
        let serial = ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("serial pool")
            .install(|| {
                Snapshots::sample_stabilizer(
                    &state,
                    16,
                    SnapshotProtocol::Clifford,
                    2,
                    901,
                    u128::MAX,
                )
            })
            .expect("serial sampling");
        let parallel = ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("parallel pool")
            .install(|| {
                Snapshots::sample_stabilizer(
                    &state,
                    16,
                    SnapshotProtocol::Clifford,
                    2,
                    901,
                    u128::MAX,
                )
            })
            .expect("parallel sampling");
        let serial_arrays = serial.clifford_arrays().expect("serial Clifford arrays");
        let parallel_arrays = parallel
            .clifford_arrays()
            .expect("parallel Clifford arrays");
        assert_eq!(serial_arrays.0, parallel_arrays.0);
        assert_eq!(serial_arrays.1, parallel_arrays.1);
        assert_eq!(serial_arrays.2, parallel_arrays.2);
        assert_eq!(serial_arrays.3, parallel_arrays.3);
    }

    #[test]
    fn pauli_statevector_sampling_replays_across_bounded_worker_counts() {
        let mut state = vec![Complex64::default(); 1usize << 8];
        state[0] = Complex64::new(1.0, 0.0);
        let serial = ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("serial pool")
            .install(|| {
                Snapshots::sample_statevector(
                    &state,
                    17,
                    SnapshotProtocol::Pauli,
                    2,
                    901,
                    u128::MAX,
                )
            })
            .expect("serial sampling");
        let parallel = ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("parallel pool")
            .install(|| {
                Snapshots::sample_statevector(
                    &state,
                    17,
                    SnapshotProtocol::Pauli,
                    2,
                    901,
                    u128::MAX,
                )
            })
            .expect("parallel sampling");
        let serial_arrays = serial.pauli_arrays().expect("serial Pauli arrays");
        let parallel_arrays = parallel.pauli_arrays().expect("parallel Pauli arrays");
        assert_eq!(serial_arrays.0, parallel_arrays.0);
        assert_eq!(serial_arrays.1, parallel_arrays.1);
    }

    #[test]
    fn clifford_statevector_sampling_replays_across_bounded_worker_counts() {
        let mut state = vec![Complex64::default(); 1usize << 8];
        state[0] = Complex64::new(3.0f64.sqrt() * 0.5, 0.0);
        state[128] = Complex64::new(0.0, 0.5);
        let serial = ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("serial pool")
            .install(|| {
                Snapshots::sample_statevector(
                    &state,
                    17,
                    SnapshotProtocol::Clifford,
                    2,
                    902,
                    u128::MAX,
                )
            })
            .expect("serial sampling");
        let parallel = ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("parallel pool")
            .install(|| {
                Snapshots::sample_statevector(
                    &state,
                    17,
                    SnapshotProtocol::Clifford,
                    2,
                    902,
                    u128::MAX,
                )
            })
            .expect("parallel sampling");
        let serial_arrays = serial.clifford_arrays().expect("serial Clifford arrays");
        let parallel_arrays = parallel
            .clifford_arrays()
            .expect("parallel Clifford arrays");
        assert_eq!(serial_arrays.0, parallel_arrays.0);
        assert_eq!(serial_arrays.1, parallel_arrays.1);
        assert_eq!(serial_arrays.2, parallel_arrays.2);
        assert_eq!(serial_arrays.3, parallel_arrays.3);
    }

    #[test]
    fn dense_fidelity_replays_across_worker_counts() {
        let state = StabilizerState::zero_state(10).expect("zero state");
        let snapshots =
            Snapshots::sample_stabilizer(&state, 16, SnapshotProtocol::Clifford, 1, 903, u128::MAX)
                .expect("Clifford snapshots");
        let mut target = (0..1usize << 10)
            .map(|index| {
                let value = index as f64 + 1.0;
                Complex64::new((value * 0.17).sin(), (value * 0.31).cos())
            })
            .collect::<Vec<_>>();
        let norm = target
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt();
        for value in &mut target {
            *value /= norm;
        }
        let serial = ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("serial pool")
            .install(|| snapshots.fidelity_statevector(&target, 4, true))
            .expect("serial fidelity");
        let parallel = ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("parallel pool")
            .install(|| snapshots.fidelity_statevector(&target, 4, true))
            .expect("parallel fidelity");
        assert_eq!(serial, parallel);
    }
}
