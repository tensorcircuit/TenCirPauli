//! Rust-native Heisenberg Pauli propagation.
//!
//! The engine deliberately keeps its dynamic state separate from the public
//! `PauliOperator`.  Small and medium systems use an inline two-word key; only
//! wider systems fall back to heap-backed masks.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::mem::size_of;

use rustc_hash::FxHashMap;

use crate::error::PauliError;
use crate::gate::{Clifford1, Clifford2, GateKind, GateOperation, ParameterRef, RotationAxis};
use crate::operator::{PauliOperator, PauliTerm};
use crate::scalar::{is_exact_zero, Complex64};
use crate::word::{packed_word_count, PauliPhase, PauliWord};

/// A product initial state for expectation evaluation.
#[derive(Clone, Debug)]
pub enum ProductState {
    /// `|0...0>`.
    Zero,
    /// Computational-basis bits in public qubit order, with bit 0 for `|0>`.
    ComputationalBasis(Vec<u8>),
    /// One Bloch vector `(x, y, z)` per qubit.
    Bloch(Vec<[f64; 3]>),
}

/// Lightweight counters returned by an explicit profile call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PropagationStats {
    pub gate_count: usize,
    pub initial_terms: usize,
    pub final_terms: usize,
    pub peak_terms: usize,
    pub estimated_peak_bytes: usize,
    pub final_weight_counts: Vec<usize>,
}

/// Result of one propagation, kept private to the Python boundary except for
/// its canonical terms.
#[derive(Clone, Debug)]
pub struct PropagationResult {
    pub terms: Vec<PauliTerm>,
    pub stats: PropagationStats,
}

/// An immutable compiled propagation engine.
#[derive(Clone, Debug)]
pub struct PropagationEngine {
    nqubits: usize,
    operations: Vec<GateOperation>,
    observable: PauliOperator,
    initial_state: ProductState,
    max_weight: Option<usize>,
    max_bytes: Option<u128>,
    nparameters: usize,
    hermitian: bool,
}

impl PropagationEngine {
    /// Compile a tape, observable and product-state descriptor.
    pub fn new(
        nqubits: usize,
        operations: Vec<GateOperation>,
        observable: PauliOperator,
        initial_state: ProductState,
        max_weight: Option<usize>,
        max_bytes: Option<u128>,
    ) -> Result<Self, PauliError> {
        if observable.nqubits() != nqubits {
            return Err(PauliError::IncompatibleQubitCounts {
                left: observable.nqubits(),
                right: nqubits,
            });
        }
        validate_state(nqubits, &initial_state)?;
        let mut slots = HashSet::new();
        for operation in &operations {
            if let Some(slot) = operation.parameter_slot() {
                slots.insert(slot);
            }
        }
        let nparameters = slots.iter().copied().max().map_or(0, |slot| slot + 1);
        if slots.len() != nparameters || (0..nparameters).any(|slot| !slots.contains(&slot)) {
            return Err(PauliError::InvalidClifford {
                context: "parameter slots must cover 0..nparameters-1 without holes",
            });
        }
        let observable_bytes = observable
            .terms()
            .len()
            .checked_mul(size_of::<PackedKey>() + size_of::<Complex64>())
            .ok_or(PauliError::Overflow {
                context: "estimating propagation observable storage",
            })?;
        check_budget(
            observable_bytes,
            max_bytes,
            "propagation observable storage",
        )?;
        let transition_bytes =
            operations
                .iter()
                .map(operation_storage_bytes)
                .try_fold(0usize, |sum, value| {
                    sum.checked_add(value).ok_or(PauliError::Overflow {
                        context: "estimating propagation transition storage",
                    })
                })?;
        check_budget(
            observable_bytes
                .checked_add(transition_bytes)
                .ok_or(PauliError::Overflow {
                    context: "estimating propagation engine storage",
                })?,
            max_bytes,
            "propagation engine storage",
        )?;
        let hermitian = observable.is_hermitian(0.0);
        Ok(Self {
            nqubits,
            operations,
            observable,
            initial_state,
            max_weight,
            max_bytes,
            nparameters,
            hermitian,
        })
    }

    pub fn nqubits(&self) -> usize {
        self.nqubits
    }

    pub fn nparameters(&self) -> usize {
        self.nparameters
    }

    pub fn gate_count(&self) -> usize {
        self.operations.len()
    }

    pub fn max_weight(&self) -> Option<usize> {
        self.max_weight
    }

    pub fn is_exact(&self) -> bool {
        self.max_weight.is_none_or(|cutoff| cutoff >= self.nqubits)
    }

    pub fn is_hermitian_observable(&self) -> bool {
        self.hermitian
    }

    /// Propagate and return a scalar product-state expectation.
    pub fn expectation(&self, parameters: &[f64]) -> Result<f64, PauliError> {
        if !self.hermitian {
            return Err(PauliError::NonHermitianExpectation);
        }
        let result = self.propagate(parameters)?;
        Ok(expectation_from_terms(
            &result.terms,
            &self.initial_state,
            self.nqubits,
        ))
    }

    /// Evaluate already propagated terms against the compiled product state.
    pub fn expectation_of_terms(&self, terms: &[PauliTerm]) -> f64 {
        expectation_from_terms(terms, &self.initial_state, self.nqubits)
    }

    /// Propagate and return canonical public terms plus structural counters.
    pub fn propagate(&self, parameters: &[f64]) -> Result<PropagationResult, PauliError> {
        validate_parameters(parameters, self.nparameters)?;
        let exact = self.is_exact();
        let cutoff = self.max_weight;
        let mut terms =
            self.observable
                .terms()
                .iter()
                .filter_map(|term| {
                    let key = PackedKey::from_word(&term.word);
                    (exact || cutoff.is_none_or(|limit| key.weight(self.nqubits) <= limit))
                        .then_some(DynamicTerm {
                            key,
                            coefficient: term.coefficient,
                        })
                })
                .collect::<Vec<_>>();
        let initial_terms = terms.len();
        let mut peak_terms = initial_terms;
        let mut estimated_peak_bytes =
            initial_terms
                .checked_mul(size_of::<DynamicTerm>())
                .ok_or(PauliError::Overflow {
                    context: "estimating initial propagation storage",
                })?;
        check_budget(
            estimated_peak_bytes,
            self.max_bytes,
            "initial propagation storage",
        )?;

        for operation in self.operations.iter().rev() {
            let branch_factor = operation_branch_factor(operation);
            let candidate_count =
                terms
                    .len()
                    .checked_mul(branch_factor)
                    .ok_or(PauliError::Overflow {
                        context: "estimating propagation contribution count",
                    })?;
            let candidate_bytes = candidate_count
                .checked_mul(size_of::<DynamicTerm>())
                .ok_or(PauliError::Overflow {
                    context: "estimating propagation contribution storage",
                })?;
            estimated_peak_bytes = estimated_peak_bytes.max(candidate_bytes);
            check_budget(
                candidate_bytes,
                self.max_bytes,
                "propagation contribution storage",
            )?;
            terms = apply_operation(self.nqubits, operation, terms, parameters, cutoff)?;
            peak_terms = peak_terms.max(terms.len());
            let actual_bytes =
                terms
                    .len()
                    .checked_mul(size_of::<DynamicTerm>())
                    .ok_or(PauliError::Overflow {
                        context: "estimating propagated term storage",
                    })?;
            estimated_peak_bytes = estimated_peak_bytes.max(actual_bytes);
        }

        let mut public_terms = Vec::with_capacity(terms.len());
        let mut final_weight_counts = vec![0usize; self.nqubits.saturating_add(1)];
        for term in terms {
            let word = term.key.to_word(self.nqubits)?;
            let weight = word.weight() as usize;
            if let Some(count) = final_weight_counts.get_mut(weight) {
                *count += 1;
            }
            public_terms.push(PauliTerm {
                word,
                coefficient: term.coefficient,
            });
        }
        public_terms.sort_unstable_by(|left, right| left.word.cmp(&right.word));
        Ok(PropagationResult {
            stats: PropagationStats {
                gate_count: self.operations.len(),
                initial_terms,
                final_terms: public_terms.len(),
                peak_terms,
                estimated_peak_bytes,
                final_weight_counts,
            },
            terms: public_terms,
        })
    }
}

#[derive(Clone, Debug)]
struct DynamicTerm {
    key: PackedKey,
    coefficient: Complex64,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum PackedKey {
    Inline {
        nqubits: usize,
        x: [u64; 2],
        z: [u64; 2],
    },
    Wide {
        nqubits: usize,
        x: Vec<u64>,
        z: Vec<u64>,
    },
}

impl PackedKey {
    fn from_word(word: &PauliWord) -> Self {
        let nqubits = word.nqubits();
        if nqubits <= 128 {
            let mut x = [0_u64; 2];
            let mut z = [0_u64; 2];
            for (index, value) in word.x_words().iter().copied().enumerate() {
                x[index] = value;
            }
            for (index, value) in word.z_words().iter().copied().enumerate() {
                z[index] = value;
            }
            Self::Inline { nqubits, x, z }
        } else {
            Self::Wide {
                nqubits,
                x: word.x_words().to_vec(),
                z: word.z_words().to_vec(),
            }
        }
    }

    fn identity(nqubits: usize) -> Self {
        if nqubits <= 128 {
            Self::Inline {
                nqubits,
                x: [0; 2],
                z: [0; 2],
            }
        } else {
            let words = packed_word_count(nqubits);
            Self::Wide {
                nqubits,
                x: vec![0; words],
                z: vec![0; words],
            }
        }
    }

    fn code_at(&self, qubit: usize) -> u8 {
        let index = qubit / 64;
        let shift = qubit % 64;
        let (x, z) = match self {
            Self::Inline { x, z, .. } => ((x[index] >> shift) & 1, (z[index] >> shift) & 1),
            Self::Wide { x, z, .. } => ((x[index] >> shift) & 1, (z[index] >> shift) & 1),
        };
        match (x, z) {
            (0, 0) => 0,
            (1, 0) => 1,
            (1, 1) => 2,
            (0, 1) => 3,
            _ => unreachable!("packed symplectic bits are binary"),
        }
    }

    fn set_code(&mut self, qubit: usize, code: u8) {
        let index = qubit / 64;
        let mask = 1_u64 << (qubit % 64);
        let (x_bit, z_bit) = match code {
            0 => (false, false),
            1 => (true, false),
            2 => (true, true),
            3 => (false, true),
            _ => unreachable!("local Pauli code must be in 0..4"),
        };
        match self {
            Self::Inline { x, z, .. } => {
                x[index] = (x[index] & !mask) | if x_bit { mask } else { 0 };
                z[index] = (z[index] & !mask) | if z_bit { mask } else { 0 };
            }
            Self::Wide { x, z, .. } => {
                x[index] = (x[index] & !mask) | if x_bit { mask } else { 0 };
                z[index] = (z[index] & !mask) | if z_bit { mask } else { 0 };
            }
        }
    }

    fn weight(&self, nqubits: usize) -> usize {
        match self {
            Self::Inline { x, z, .. } => (0..packed_word_count(nqubits))
                .map(|index| (x[index] | z[index]).count_ones() as usize)
                .sum(),
            Self::Wide { x, z, .. } => x
                .iter()
                .zip(z)
                .map(|(x, z)| (x | z).count_ones() as usize)
                .sum(),
        }
    }

    fn multiply(&self, other: &Self, nqubits: usize) -> (Self, PauliPhase) {
        let result = match (self, other) {
            (
                Self::Inline { nqubits, x, z },
                Self::Inline {
                    x: other_x,
                    z: other_z,
                    ..
                },
            ) => Self::Inline {
                nqubits: *nqubits,
                x: [x[0] ^ other_x[0], x[1] ^ other_x[1]],
                z: [z[0] ^ other_z[0], z[1] ^ other_z[1]],
            },
            _ => {
                let x_values = masks_x(self, nqubits)
                    .into_iter()
                    .zip(masks_x(other, nqubits))
                    .map(|(left, right)| left ^ right)
                    .collect::<Vec<_>>();
                let z_values = masks_z(self, nqubits)
                    .into_iter()
                    .zip(masks_z(other, nqubits))
                    .map(|(left, right)| left ^ right)
                    .collect::<Vec<_>>();
                Self::Wide {
                    nqubits,
                    x: x_values,
                    z: z_values,
                }
            }
        };
        let mut phase = PauliPhase::PlusOne;
        for qubit in 0..nqubits {
            phase = phase.multiply(local_phase(self.code_at(qubit), other.code_at(qubit)));
        }
        (result, phase)
    }

    fn to_word(&self, nqubits: usize) -> Result<PauliWord, PauliError> {
        let x = masks_x(self, nqubits);
        let z = masks_z(self, nqubits);
        PauliWord::from_words(nqubits, x, z)
    }
}

impl Ord for PackedKey {
    fn cmp(&self, other: &Self) -> Ordering {
        let left_n = key_nqubits(self);
        let right_n = key_nqubits(other);
        match left_n.cmp(&right_n) {
            Ordering::Equal => (0..left_n)
                .map(|qubit| self.code_at(qubit).cmp(&other.code_at(qubit)))
                .find(|order| *order != Ordering::Equal)
                .unwrap_or(Ordering::Equal),
            order => order,
        }
    }
}

impl PartialOrd for PackedKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn key_nqubits(key: &PackedKey) -> usize {
    match key {
        PackedKey::Inline { nqubits, .. } | PackedKey::Wide { nqubits, .. } => *nqubits,
    }
}

fn masks_x(key: &PackedKey, nqubits: usize) -> Vec<u64> {
    match key {
        PackedKey::Inline { x, .. } => x[..packed_word_count(nqubits)].to_vec(),
        PackedKey::Wide { x, .. } => x.clone(),
    }
}

fn masks_z(key: &PackedKey, nqubits: usize) -> Vec<u64> {
    match key {
        PackedKey::Inline { z, .. } => z[..packed_word_count(nqubits)].to_vec(),
        PackedKey::Wide { z, .. } => z.clone(),
    }
}

fn local_phase(left: u8, right: u8) -> PauliPhase {
    match (left, right) {
        (1, 2) | (2, 3) | (3, 1) => PauliPhase::PlusI,
        (2, 1) | (3, 2) | (1, 3) => PauliPhase::MinusI,
        _ => PauliPhase::PlusOne,
    }
}

fn phase_sign(phase: PauliPhase) -> f64 {
    match phase {
        PauliPhase::PlusOne => 1.0,
        PauliPhase::MinusOne => -1.0,
        PauliPhase::PlusI | PauliPhase::MinusI => {
            debug_assert!(
                false,
                "anticommuting rotation products must have an imaginary phase"
            );
            0.0
        }
    }
}

fn validate_state(nqubits: usize, state: &ProductState) -> Result<(), PauliError> {
    match state {
        ProductState::Zero => Ok(()),
        ProductState::ComputationalBasis(bits) => {
            if bits.len() != nqubits || bits.iter().any(|bit| *bit > 1) {
                return Err(PauliError::InvalidStructureLength {
                    expected: nqubits,
                    actual: bits.len(),
                });
            }
            Ok(())
        }
        ProductState::Bloch(vectors) => {
            if vectors.len() != nqubits {
                return Err(PauliError::InvalidStructureLength {
                    expected: nqubits,
                    actual: vectors.len(),
                });
            }
            for vector in vectors {
                if vector.iter().any(|value| !value.is_finite()) {
                    return Err(PauliError::NonFiniteParameter { index: 0 });
                }
                let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
                if norm > 1.0 + 1e-12 {
                    return Err(PauliError::InvalidClifford {
                        context: "Bloch vector norm exceeds 1 + 1e-12",
                    });
                }
            }
            Ok(())
        }
    }
}

fn validate_parameters(parameters: &[f64], expected: usize) -> Result<(), PauliError> {
    if parameters.len() != expected {
        return Err(PauliError::InvalidParameterLength {
            expected,
            actual: parameters.len(),
        });
    }
    if let Some(index) = parameters.iter().position(|value| !value.is_finite()) {
        return Err(PauliError::NonFiniteParameter { index });
    }
    Ok(())
}

fn operation_storage_bytes(operation: &GateOperation) -> usize {
    match &operation.kind {
        GateKind::CustomPtm { transitions, .. } => transitions
            .iter()
            .map(|row| row.len().saturating_mul(size_of::<(u8, f64)>()))
            .sum(),
        _ => size_of::<GateOperation>(),
    }
}

fn operation_branch_factor(operation: &GateOperation) -> usize {
    match &operation.kind {
        GateKind::Clifford1 { .. } | GateKind::Clifford2 { .. } => 1,
        GateKind::Rotation { .. } => 2,
        GateKind::CustomPtm { transitions, .. } => {
            transitions.iter().map(Vec::len).max().unwrap_or(0)
        }
    }
}

fn apply_operation(
    nqubits: usize,
    operation: &GateOperation,
    terms: Vec<DynamicTerm>,
    parameters: &[f64],
    cutoff: Option<usize>,
) -> Result<Vec<DynamicTerm>, PauliError> {
    match &operation.kind {
        GateKind::Clifford1 { gate, wire } => {
            let mut result = Vec::with_capacity(terms.len());
            for mut term in terms {
                let (key, multiplier) = map_clifford1(&term.key, *gate, *wire);
                term.key = key;
                term.coefficient *= multiplier;
                if !is_exact_zero(term.coefficient)
                    && cutoff.is_none_or(|limit| term.key.weight(nqubits) <= limit)
                {
                    result.push(term);
                }
            }
            Ok(result)
        }
        GateKind::Clifford2 { gate, wire0, wire1 } => {
            let mut result = Vec::with_capacity(terms.len());
            for mut term in terms {
                let (key, multiplier) = map_clifford2(nqubits, &term.key, *gate, *wire0, *wire1);
                term.key = key;
                term.coefficient *= multiplier;
                if !is_exact_zero(term.coefficient)
                    && cutoff.is_none_or(|limit| term.key.weight(nqubits) <= limit)
                {
                    result.push(term);
                }
            }
            Ok(result)
        }
        GateKind::Rotation {
            axis,
            wire0,
            wire1,
            parameter,
        } => {
            let (cosine, sine) = resolve_parameter(*parameter, parameters)?;
            let generator = generator_key(nqubits, *axis, *wire0, *wire1);
            let mut contributions = Vec::with_capacity(terms.len().saturating_mul(2));
            for term in terms {
                let (product, phase) = generator.multiply(&term.key, nqubits);
                match phase {
                    PauliPhase::PlusI | PauliPhase::MinusI => {
                        contributions.push((term.key.clone(), term.coefficient * cosine));
                        if sine != 0.0 {
                            contributions
                                .push((product, term.coefficient * (sine * phase_sign_i(phase))));
                        }
                    }
                    PauliPhase::PlusOne | PauliPhase::MinusOne => {
                        contributions.push((term.key, term.coefficient));
                    }
                }
            }
            aggregate(contributions, nqubits, cutoff)
        }
        GateKind::CustomPtm {
            wire0,
            wire1,
            transitions,
        } => {
            let mut contributions = Vec::new();
            for term in terms {
                let input = local_index(&term.key, *wire0, *wire1);
                for &(output, coefficient) in &transitions[input] {
                    let mut key = term.key.clone();
                    let (first, second) = local_codes(output, wire1.is_some());
                    key.set_code(*wire0, first);
                    if let Some(second_wire) = wire1 {
                        key.set_code(*second_wire, second);
                    }
                    contributions.push((key, term.coefficient * coefficient));
                }
            }
            aggregate(contributions, nqubits, cutoff)
        }
    }
}

fn aggregate(
    contributions: Vec<(PackedKey, Complex64)>,
    nqubits: usize,
    cutoff: Option<usize>,
) -> Result<Vec<DynamicTerm>, PauliError> {
    let mut values = FxHashMap::<PackedKey, Complex64>::with_capacity_and_hasher(
        contributions.len(),
        Default::default(),
    );
    for (key, coefficient) in contributions {
        if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
            return Err(PauliError::NonFiniteCoefficient {
                index: values.len(),
            });
        }
        values
            .entry(key)
            .and_modify(|current| *current += coefficient)
            .or_insert(coefficient);
    }
    let mut ordered = values.into_iter().collect::<Vec<_>>();
    ordered.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    Ok(ordered
        .into_iter()
        .filter_map(|(key, coefficient)| {
            (coefficient.re.is_finite()
                && coefficient.im.is_finite()
                && !is_exact_zero(coefficient)
                && cutoff.is_none_or(|limit| key.weight(nqubits) <= limit))
            .then_some(DynamicTerm { key, coefficient })
        })
        .collect())
}

fn map_clifford1(key: &PackedKey, gate: Clifford1, wire: usize) -> (PackedKey, f64) {
    let code = key.code_at(wire);
    let (mapped_code, sign) = match gate {
        Clifford1::X => match code {
            0 => (0, 1.0),
            1 => (1, 1.0),
            2 => (2, -1.0),
            3 => (3, -1.0),
            _ => unreachable!(),
        },
        Clifford1::Y => match code {
            0 => (0, 1.0),
            1 => (1, -1.0),
            2 => (2, 1.0),
            3 => (3, -1.0),
            _ => unreachable!(),
        },
        Clifford1::Z => match code {
            0 => (0, 1.0),
            1 => (1, -1.0),
            2 => (2, -1.0),
            3 => (3, 1.0),
            _ => unreachable!(),
        },
        Clifford1::H => match code {
            0 => (0, 1.0),
            1 => (3, 1.0),
            2 => (2, -1.0),
            3 => (1, 1.0),
            _ => unreachable!(),
        },
        Clifford1::S => match code {
            0 => (0, 1.0),
            1 => (2, -1.0),
            2 => (1, 1.0),
            3 => (3, 1.0),
            _ => unreachable!(),
        },
        Clifford1::Sdg => match code {
            0 => (0, 1.0),
            1 => (2, 1.0),
            2 => (1, -1.0),
            3 => (3, 1.0),
            _ => unreachable!(),
        },
    };
    let mut result = key.clone();
    result.set_code(wire, mapped_code);
    (result, sign)
}

fn map_clifford2(
    nqubits: usize,
    key: &PackedKey,
    gate: Clifford2,
    wire0: usize,
    wire1: usize,
) -> (PackedKey, f64) {
    let first = key.code_at(wire0);
    let second = key.code_at(wire1);
    let (mapped_first, mapped_second) = match gate {
        Clifford2::Swap => (first, second),
        Clifford2::Cnot => (map_cnot_control(first), map_cnot_target(second)),
        Clifford2::Cz => (map_cz_first(first), map_cz_second(second)),
    };
    let mut result = key.clone();
    result.set_code(wire0, 0);
    result.set_code(wire1, 0);
    let (first_key, first_sign) =
        local_mapping_key(nqubits, wire0, wire1, mapped_first, gate, true);
    let (second_key, second_sign) =
        local_mapping_key(nqubits, wire0, wire1, mapped_second, gate, false);
    let (result, phase) = result.multiply(&first_key, nqubits);
    let (result, second_phase) = result.multiply(&second_key, nqubits);
    let multiplier = first_sign * second_sign * phase_sign(phase) * phase_sign(second_phase);
    (result, multiplier)
}

fn local_mapping_key(
    nqubits: usize,
    wire0: usize,
    wire1: usize,
    code: u8,
    gate: Clifford2,
    first: bool,
) -> (PackedKey, f64) {
    let mut key = PackedKey::identity(nqubits);
    let sign = 1.0;
    match gate {
        Clifford2::Swap => key.set_code(if first { wire1 } else { wire0 }, code),
        Clifford2::Cnot => {
            if first {
                key.set_code(wire0, code);
                if code != 0 {
                    key.set_code(wire1, if code == 3 { 0 } else { 1 });
                }
            } else {
                key.set_code(wire1, code);
                if code != 0 {
                    key.set_code(wire0, if code == 1 { 0 } else { 3 });
                }
            }
        }
        Clifford2::Cz => {
            if first {
                key.set_code(wire0, code);
                if code == 1 || code == 2 {
                    key.set_code(wire1, 3);
                }
            } else {
                key.set_code(wire1, code);
                if code == 1 || code == 2 {
                    key.set_code(wire0, 3);
                }
            }
        }
    }
    // The compact local-code maps above are positive.  `sign` is retained in
    // the return type so the kernel has one common shape for all Clifford maps.
    (key, sign)
}

fn map_cnot_control(code: u8) -> u8 {
    code
}

fn map_cnot_target(code: u8) -> u8 {
    code
}

fn map_cz_first(code: u8) -> u8 {
    code
}

fn map_cz_second(code: u8) -> u8 {
    code
}

fn generator_key(
    nqubits: usize,
    axis: RotationAxis,
    wire0: usize,
    wire1: Option<usize>,
) -> PackedKey {
    let code = match axis {
        RotationAxis::X => 1,
        RotationAxis::Y => 2,
        RotationAxis::Z => 3,
    };
    let mut key = PackedKey::identity(nqubits);
    key.set_code(wire0, code);
    if let Some(wire) = wire1 {
        key.set_code(wire, code);
    }
    key
}

fn resolve_parameter(
    parameter: ParameterRef,
    parameters: &[f64],
) -> Result<(f64, f64), PauliError> {
    match parameter {
        ParameterRef::Static { cos, sin } => Ok((cos, sin)),
        ParameterRef::Slot(slot) => {
            let angle = parameters[slot];
            Ok((angle.cos(), angle.sin()))
        }
    }
}

fn phase_sign_i(phase: PauliPhase) -> f64 {
    match phase {
        PauliPhase::PlusI => -1.0,
        PauliPhase::MinusI => 1.0,
        _ => phase_sign(phase),
    }
}

fn local_index(key: &PackedKey, wire0: usize, wire1: Option<usize>) -> usize {
    match wire1 {
        None => key.code_at(wire0) as usize,
        Some(wire) => 4 * key.code_at(wire0) as usize + key.code_at(wire) as usize,
    }
}

fn local_codes(index: u8, two_qubit: bool) -> (u8, u8) {
    if two_qubit {
        (index / 4, index % 4)
    } else {
        (index, 0)
    }
}

fn expectation_from_terms(terms: &[PauliTerm], state: &ProductState, nqubits: usize) -> f64 {
    terms
        .iter()
        .map(|term| {
            let local = (0..nqubits).fold(1.0, |product, qubit| {
                let code = term.word.code_at(qubit);
                let component = match state {
                    ProductState::Zero => match code {
                        0 | 3 => 1.0,
                        _ => 0.0,
                    },
                    ProductState::ComputationalBasis(bits) => match code {
                        0 => 1.0,
                        3 => {
                            if bits[qubit] == 0 {
                                1.0
                            } else {
                                -1.0
                            }
                        }
                        _ => 0.0,
                    },
                    ProductState::Bloch(vectors) => match code {
                        0 => 1.0,
                        code => vectors[qubit][code as usize - 1],
                    },
                };
                product * component
            });
            term.coefficient.re * local
        })
        .sum()
}

fn check_budget(
    requested: usize,
    limit: Option<u128>,
    _context: &'static str,
) -> Result<(), PauliError> {
    if let Some(limit) = limit {
        if requested as u128 > limit {
            return Err(PauliError::MemoryLimit {
                requested: requested as u128,
                limit,
            });
        }
    }
    Ok(())
}
