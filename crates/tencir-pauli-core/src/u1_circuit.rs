//! Rust-native fixed-particle-number circuit execution.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::mem::size_of;
use std::sync::Arc;

use rayon::prelude::*;

use crate::circuit_ir::{CircuitGate, CircuitProgram};
use crate::error::PauliError;
use crate::operator::PauliOperator;
use crate::scalar::Complex64;
use crate::sector::U1Sector;

const U1_CIRCUIT_PARALLEL_PAIR_THRESHOLD: usize = 1 << 14;

#[derive(Clone, Copy, Debug)]
struct PairIndex {
    zero_one: usize,
    one_zero: usize,
}

#[derive(Clone, Debug)]
enum DiagonalOp {
    Rz {
        wire: usize,
        angle: usize,
    },
    Rzz {
        wire0: usize,
        wire1: usize,
        angle: usize,
    },
    Cz {
        wire0: usize,
        wire1: usize,
        indices: Arc<[usize]>,
    },
    Cphase {
        wire0: usize,
        wire1: usize,
        angle: usize,
        indices: Arc<[usize]>,
    },
    Static {
        wires: Arc<[usize]>,
        payload: Arc<[Complex64]>,
    },
}

impl DiagonalOp {
    fn is_sparse(&self) -> bool {
        matches!(self, Self::Cz { .. } | Self::Cphase { .. })
    }
}

#[derive(Clone, Debug)]
enum CompiledU1Gate {
    Swap {
        pairs: Arc<[PairIndex]>,
    },
    Iswap {
        angle: usize,
        pairs: Arc<[PairIndex]>,
    },
    DiagonalBlock {
        operations: Arc<[DiagonalOp]>,
    },
}

/// Immutable compiled U(1) circuit plan.
#[derive(Clone, Debug)]
pub struct U1CircuitPlan {
    program: CircuitProgram,
    sector: U1Sector,
    dimension: usize,
    gates: Arc<[CompiledU1Gate]>,
    basis_words: Arc<[u64]>,
    word_count: usize,
    max_bytes: Option<u128>,
}

impl U1CircuitPlan {
    pub fn compile(
        program: CircuitProgram,
        sector: U1Sector,
        max_bytes: Option<u128>,
    ) -> Result<Self, PauliError> {
        if program.nqubits() != sector.nqubits() {
            return Err(PauliError::IncompatibleQubitCounts {
                left: program.nqubits(),
                right: sector.nqubits(),
            });
        }
        let dimension = sector.dimension()?;
        let word_count = sector.word_count();
        let basis_count = dimension
            .checked_mul(word_count)
            .ok_or(PauliError::Overflow {
                context: "sizing U1 circuit occupation metadata",
            })?;
        let basis_bytes = (basis_count as u128)
            .checked_mul(size_of::<u64>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating U1 circuit occupation metadata",
            })?;
        check_budget(basis_bytes, max_bytes)?;
        let mut basis_words = vec![0_u64; basis_count];
        for index in 0..dimension {
            sector.unrank_into(
                index as u64,
                &mut basis_words[index * word_count..(index + 1) * word_count],
            )?;
        }

        let mut pair_maps = HashMap::<(usize, usize), Arc<[PairIndex]>>::new();
        let mut diagonal_indices = HashMap::<(usize, usize), Arc<[usize]>>::new();
        let mut gates = Vec::with_capacity(program.operations().len());
        let mut operation_index = 0;
        while operation_index < program.operations().len() {
            if let Some(first) = diagonal_op(
                &program.operations()[operation_index],
                &basis_words,
                word_count,
                dimension,
                &mut diagonal_indices,
                max_bytes,
            )? {
                let mut operations = vec![first];
                operation_index += 1;
                while operation_index < program.operations().len() {
                    if let Some(operation) = diagonal_op(
                        &program.operations()[operation_index],
                        &basis_words,
                        word_count,
                        dimension,
                        &mut diagonal_indices,
                        max_bytes,
                    )? {
                        operations.push(operation);
                        operation_index += 1;
                    } else {
                        break;
                    }
                }
                gates.push(CompiledU1Gate::DiagonalBlock {
                    operations: Arc::from(operations.into_boxed_slice()),
                });
            } else {
                gates.push(compile_non_diagonal_gate(
                    &program.operations()[operation_index],
                    &sector,
                    &mut pair_maps,
                    max_bytes,
                )?);
                operation_index += 1;
            }
        }
        let pair_bytes = pair_maps.values().try_fold(0_u128, |sum, pairs| {
            let bytes = (pairs.len() as u128)
                .checked_mul(size_of::<PairIndex>() as u128)
                .ok_or(PauliError::Overflow {
                    context: "estimating U1 circuit pair-map memory",
                })?;
            sum.checked_add(bytes).ok_or(PauliError::Overflow {
                context: "estimating U1 circuit pair-map memory",
            })
        })?;
        let diagonal_bytes = diagonal_indices.values().try_fold(0_u128, |sum, indices| {
            let bytes = (indices.len() as u128)
                .checked_mul(size_of::<usize>() as u128)
                .ok_or(PauliError::Overflow {
                    context: "estimating U1 circuit diagonal-index memory",
                })?;
            sum.checked_add(bytes).ok_or(PauliError::Overflow {
                context: "estimating U1 circuit diagonal-index memory",
            })
        })?;
        check_budget(
            basis_bytes
                .checked_add(pair_bytes)
                .and_then(|bytes| bytes.checked_add(diagonal_bytes))
                .ok_or(PauliError::Overflow {
                    context: "estimating U1 circuit compiled metadata",
                })?,
            max_bytes,
        )?;
        Ok(Self {
            program,
            sector,
            dimension,
            gates: Arc::from(gates.into_boxed_slice()),
            basis_words: Arc::from(basis_words.into_boxed_slice()),
            word_count,
            max_bytes,
        })
    }

    pub fn sector(&self) -> U1Sector {
        self.sector.clone()
    }

    pub fn nqubits(&self) -> usize {
        self.sector.nqubits()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn nparameters(&self) -> usize {
        self.program.nparameters()
    }

    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }

    pub fn run(
        &self,
        initial_state: &[Complex64],
        parameters: &[f64],
    ) -> Result<Vec<Complex64>, PauliError> {
        self.validate_state(initial_state)?;
        let values = self.program.evaluate_parameters(parameters)?;
        let mut state = initial_state.to_vec();
        for gate in self.gates.iter() {
            self.apply_gate(&mut state, gate, &values, false)?;
        }
        Ok(state)
    }

    pub fn probability(
        &self,
        initial_state: &[Complex64],
        parameters: &[f64],
    ) -> Result<Vec<f64>, PauliError> {
        let state = self.run(initial_state, parameters)?;
        Ok(state.iter().map(|value| value.norm_sqr()).collect())
    }

    pub fn to_dense(
        &self,
        initial_state: &[Complex64],
        parameters: &[f64],
    ) -> Result<Vec<Complex64>, PauliError> {
        let state = self.run(initial_state, parameters)?;
        let full_dimension = checked_full_dimension(self.nqubits(), self.max_bytes)?;
        let mut output = vec![Complex64::default(); full_dimension];
        for (index, amplitude) in state.into_iter().enumerate() {
            let basis = self.basis_integer(index)?;
            output[basis] = amplitude;
        }
        Ok(output)
    }

    pub fn probability_full(
        &self,
        initial_state: &[Complex64],
        parameters: &[f64],
    ) -> Result<Vec<f64>, PauliError> {
        let state = self.run(initial_state, parameters)?;
        let full_dimension = checked_full_dimension(self.nqubits(), self.max_bytes)?;
        let mut output = vec![0.0; full_dimension];
        for (index, amplitude) in state.into_iter().enumerate() {
            let basis = self.basis_integer(index)?;
            output[basis] = amplitude.norm_sqr();
        }
        Ok(output)
    }

    pub fn expectation(
        &self,
        initial_state: &[Complex64],
        observable: &PauliOperator,
        parameters: &[f64],
    ) -> Result<Complex64, PauliError> {
        if observable.nqubits() != self.nqubits() {
            return Err(PauliError::IncompatibleQubitCounts {
                left: observable.nqubits(),
                right: self.nqubits(),
            });
        }
        let state = self.run(initial_state, parameters)?;
        let applied = self.apply_observable(&state, observable)?;
        Ok(inner_product(&state, &applied))
    }

    pub fn value_and_grad(
        &self,
        initial_state: &[Complex64],
        observable: &PauliOperator,
        parameters: &[f64],
    ) -> Result<(f64, Vec<f64>), PauliError> {
        if observable.nqubits() != self.nqubits() {
            return Err(PauliError::IncompatibleQubitCounts {
                left: observable.nqubits(),
                right: self.nqubits(),
            });
        }
        if !observable.is_hermitian(0.0) {
            return Err(PauliError::NonHermitianExpectation);
        }
        self.validate_state(initial_state)?;
        let values = self.program.evaluate_parameters(parameters)?;
        let mut state = initial_state.to_vec();
        for gate in self.gates.iter() {
            self.apply_gate(&mut state, gate, &values, false)?;
        }
        let value_state = self.apply_observable(&state, observable)?;
        let value = inner_product(&state, &value_state).re;
        let mut lambda = value_state;
        let mut node_adjoint = vec![0.0; self.program.parameter_program().len()];
        for gate in self.gates.iter().rev() {
            let mut before = state.clone();
            self.apply_inverse_gate(&mut before, gate, &values)?;
            accumulate_gate_derivative(
                &state,
                &before,
                &lambda,
                gate,
                &values,
                &mut node_adjoint,
                self.word_count,
                &self.basis_words,
            )?;
            self.apply_inverse_gate(&mut lambda, gate, &values)?;
            state = before;
        }
        let gradient = self
            .program
            .reverse_parameter_program(&values, &node_adjoint)?;
        if !value.is_finite() || gradient.iter().any(|entry| !entry.is_finite()) {
            return Err(PauliError::InvalidCircuit {
                context: "circuit value or gradient is non-finite",
            });
        }
        Ok((value, gradient))
    }

    fn validate_state(&self, state: &[Complex64]) -> Result<(), PauliError> {
        if state.len() != self.dimension() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension(),
                actual: state.len(),
            });
        }
        if state
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(PauliError::InvalidCircuit {
                context: "initial state contains a non-finite amplitude",
            });
        }
        let bytes = (state.len() as u128)
            .checked_mul(size_of::<Complex64>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating U1 circuit state memory",
            })?;
        check_budget(bytes, self.max_bytes)
    }

    fn basis_integer(&self, index: usize) -> Result<usize, PauliError> {
        if self.nqubits() >= usize::BITS as usize {
            return Err(PauliError::Overflow {
                context: "converting wide U1 basis state to a dense index",
            });
        }
        let mut value = 0_usize;
        for wire in 0..self.nqubits() {
            if self.bit(index, wire) != 0 {
                value |= 1usize << (self.nqubits() - wire - 1);
            }
        }
        Ok(value)
    }

    fn bit(&self, index: usize, wire: usize) -> u8 {
        let word = self.basis_words[index * self.word_count + wire / 64];
        ((word >> (wire % 64)) & 1) as u8
    }

    fn apply_gate(
        &self,
        state: &mut [Complex64],
        gate: &CompiledU1Gate,
        values: &[f64],
        inverse: bool,
    ) -> Result<(), PauliError> {
        match gate {
            CompiledU1Gate::DiagonalBlock { operations } => {
                if operations.iter().all(DiagonalOp::is_sparse) {
                    for operation in operations.iter() {
                        match operation {
                            DiagonalOp::Cz { indices, .. } => {
                                for index in indices.iter().copied() {
                                    state[index] = -state[index];
                                }
                            }
                            DiagonalOp::Cphase { angle, indices, .. } => {
                                let sign = if inverse { -1.0 } else { 1.0 };
                                let phase = Complex64::from_polar(1.0, sign * values[*angle]);
                                for index in indices.iter().copied() {
                                    state[index] *= phase;
                                }
                            }
                            DiagonalOp::Rz { .. }
                            | DiagonalOp::Rzz { .. }
                            | DiagonalOp::Static { .. } => {
                                return Err(PauliError::InvalidCircuit {
                                    context: "non-sparse gate entered sparse diagonal path",
                                });
                            }
                        }
                    }
                    return Ok(());
                }
                for (index, value) in state.iter_mut().enumerate() {
                    let mut phase = Complex64::new(1.0, 0.0);
                    for operation in operations.iter() {
                        phase *= self.diagonal_phase(index, operation, values);
                    }
                    *value *= if inverse { phase.conj() } else { phase };
                }
            }
            CompiledU1Gate::Swap { pairs } => {
                for pair in pairs.iter() {
                    state.swap(pair.zero_one, pair.one_zero);
                }
            }
            CompiledU1Gate::Iswap { angle, pairs } => {
                let theta = values[*angle] * PI / 2.0 * if inverse { -1.0 } else { 1.0 };
                let cosine = theta.cos();
                let sine = Complex64::new(0.0, theta.sin());
                if pairs.len() >= U1_CIRCUIT_PARALLEL_PAIR_THRESHOLD {
                    // Pair construction emits disjoint endpoints: every sector basis
                    // state with one occupied wire in this pair appears exactly once.
                    // The raw pointer permits Rayon to update those disjoint pairs in
                    // place without allocating a full second state vector.
                    let state_ptr = state.as_mut_ptr() as usize;
                    pairs.par_iter().for_each(|pair| unsafe {
                        let state_ptr = state_ptr as *mut Complex64;
                        let left = *state_ptr.add(pair.zero_one);
                        let right = *state_ptr.add(pair.one_zero);
                        *state_ptr.add(pair.zero_one) = cosine * left + sine * right;
                        *state_ptr.add(pair.one_zero) = cosine * right + sine * left;
                    });
                } else {
                    for pair in pairs.iter() {
                        let left = state[pair.zero_one];
                        let right = state[pair.one_zero];
                        state[pair.zero_one] = cosine * left + sine * right;
                        state[pair.one_zero] = cosine * right + sine * left;
                    }
                }
            }
        }
        Ok(())
    }

    fn diagonal_phase(&self, index: usize, operation: &DiagonalOp, values: &[f64]) -> Complex64 {
        match operation {
            DiagonalOp::Rz { wire, angle } => {
                let z = if self.bit(index, *wire) == 0 {
                    1.0
                } else {
                    -1.0
                };
                Complex64::from_polar(1.0, -0.5 * values[*angle] * z)
            }
            DiagonalOp::Rzz {
                wire0,
                wire1,
                angle,
            } => {
                let z0 = if self.bit(index, *wire0) == 0 {
                    1.0
                } else {
                    -1.0
                };
                let z1 = if self.bit(index, *wire1) == 0 {
                    1.0
                } else {
                    -1.0
                };
                Complex64::from_polar(1.0, -0.5 * values[*angle] * z0 * z1)
            }
            DiagonalOp::Cz { wire0, wire1, .. } => {
                if self.bit(index, *wire0) != 0 && self.bit(index, *wire1) != 0 {
                    Complex64::new(-1.0, 0.0)
                } else {
                    Complex64::new(1.0, 0.0)
                }
            }
            DiagonalOp::Cphase {
                wire0,
                wire1,
                angle,
                ..
            } => {
                if self.bit(index, *wire0) != 0 && self.bit(index, *wire1) != 0 {
                    Complex64::from_polar(1.0, values[*angle])
                } else {
                    Complex64::new(1.0, 0.0)
                }
            }
            DiagonalOp::Static { wires, payload } => {
                let mut local = 0usize;
                for wire in wires.iter().copied() {
                    local = (local << 1) | self.bit(index, wire) as usize;
                }
                payload[local]
            }
        }
    }

    fn apply_inverse_gate(
        &self,
        state: &mut [Complex64],
        gate: &CompiledU1Gate,
        values: &[f64],
    ) -> Result<(), PauliError> {
        self.apply_gate(state, gate, values, true)
    }

    fn apply_observable(
        &self,
        state: &[Complex64],
        observable: &PauliOperator,
    ) -> Result<Vec<Complex64>, PauliError> {
        let mut output = vec![Complex64::default(); state.len()];
        let word_count = self.word_count;
        let mut destination_words = vec![0_u64; word_count];
        for term in observable.terms() {
            let word = &term.word;
            let mut y_phase = Complex64::new(1.0, 0.0);
            let y_count = (0..self.nqubits())
                .filter(|wire| word.code_at(*wire) == 2)
                .count();
            for _ in 0..y_count {
                y_phase *= Complex64::new(0.0, 1.0);
            }
            for (source, source_amplitude) in state.iter().enumerate() {
                let source_words =
                    &self.basis_words[source * word_count..(source + 1) * word_count];
                for limb in 0..word_count {
                    destination_words[limb] = source_words[limb] ^ word.x_words()[limb];
                }
                let destination = match self.sector.rank_words(&destination_words) {
                    Ok(rank) => usize::try_from(rank).map_err(|_| PauliError::Overflow {
                        context: "converting U1 observable destination index",
                    })?,
                    Err(PauliError::InvalidIndex { .. }) => continue,
                    Err(error) => return Err(error),
                };
                let parity = word
                    .z_words()
                    .iter()
                    .zip(source_words)
                    .map(|(left, right)| (left & right).count_ones())
                    .sum::<u32>();
                let phase = if parity % 2 == 0 { y_phase } else { -y_phase };
                output[destination] += term.coefficient * phase * *source_amplitude;
            }
        }
        Ok(output)
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_gate_derivative(
    after: &[Complex64],
    before: &[Complex64],
    lambda: &[Complex64],
    gate: &CompiledU1Gate,
    values: &[f64],
    node_adjoint: &mut [f64],
    word_count: usize,
    basis_words: &[u64],
) -> Result<(), PauliError> {
    let contribution = match gate {
        CompiledU1Gate::DiagonalBlock { operations } => {
            for operation in operations.iter() {
                let (angle, result) = match operation {
                    DiagonalOp::Rz { wire, angle } => {
                        let mut result = 0.0;
                        for index in 0..after.len() {
                            let z = if bit_from_basis(basis_words, word_count, index, *wire) == 0 {
                                1.0
                            } else {
                                -1.0
                            };
                            let derivative = Complex64::new(0.0, -0.5 * z) * after[index];
                            result += 2.0 * (lambda[index].conj() * derivative).re;
                        }
                        (*angle, result)
                    }
                    DiagonalOp::Rzz {
                        wire0,
                        wire1,
                        angle,
                    } => {
                        let mut result = 0.0;
                        for index in 0..after.len() {
                            let z0 = if bit_from_basis(basis_words, word_count, index, *wire0) == 0
                            {
                                1.0
                            } else {
                                -1.0
                            };
                            let z1 = if bit_from_basis(basis_words, word_count, index, *wire1) == 0
                            {
                                1.0
                            } else {
                                -1.0
                            };
                            let derivative = Complex64::new(0.0, -0.5 * z0 * z1) * after[index];
                            result += 2.0 * (lambda[index].conj() * derivative).re;
                        }
                        (*angle, result)
                    }
                    DiagonalOp::Cphase {
                        wire0,
                        wire1,
                        angle,
                        ..
                    } => {
                        let mut result = 0.0;
                        for index in 0..after.len() {
                            if bit_from_basis(basis_words, word_count, index, *wire0) != 0
                                && bit_from_basis(basis_words, word_count, index, *wire1) != 0
                            {
                                let derivative = Complex64::new(0.0, 1.0) * after[index];
                                result += 2.0 * (lambda[index].conj() * derivative).re;
                            }
                        }
                        (*angle, result)
                    }
                    DiagonalOp::Cz { .. } | DiagonalOp::Static { .. } => continue,
                };
                node_adjoint[angle] += result;
            }
            None
        }
        CompiledU1Gate::Iswap { angle, pairs } => {
            let theta = values[*angle] * PI / 2.0;
            let cosine = theta.cos();
            let sine = theta.sin();
            let alpha = PI / 2.0;
            let mut result = 0.0;
            for pair in pairs.iter() {
                let left = before[pair.zero_one];
                let right = before[pair.one_zero];
                let dleft = -alpha * sine * left + Complex64::new(0.0, alpha * cosine) * right;
                let dright = -alpha * sine * right + Complex64::new(0.0, alpha * cosine) * left;
                result += 2.0
                    * (lambda[pair.zero_one].conj() * dleft
                        + lambda[pair.one_zero].conj() * dright)
                        .re;
            }
            Some((*angle, result))
        }
        CompiledU1Gate::Swap { .. } => None,
    };
    if let Some((angle, value)) = contribution {
        node_adjoint[angle] += value;
    }
    Ok(())
}

fn bit_from_basis(basis_words: &[u64], word_count: usize, index: usize, wire: usize) -> u8 {
    ((basis_words[index * word_count + wire / 64] >> (wire % 64)) & 1) as u8
}

fn inner_product(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter()
        .zip(right)
        .fold(Complex64::default(), |sum, (a, b)| sum + a.conj() * b)
}

fn diagonal_op(
    gate: &CircuitGate,
    basis_words: &[u64],
    word_count: usize,
    dimension: usize,
    diagonal_indices: &mut HashMap<(usize, usize), Arc<[usize]>>,
    max_bytes: Option<u128>,
) -> Result<Option<DiagonalOp>, PauliError> {
    let operation = match gate {
        CircuitGate::Rz { wire, angle } => DiagonalOp::Rz {
            wire: *wire,
            angle: *angle,
        },
        CircuitGate::Rzz {
            wire0,
            wire1,
            angle,
        } => DiagonalOp::Rzz {
            wire0: *wire0,
            wire1: *wire1,
            angle: *angle,
        },
        CircuitGate::Cz { wire0, wire1 } => DiagonalOp::Cz {
            wire0: *wire0,
            wire1: *wire1,
            indices: diagonal_index_map(
                basis_words,
                word_count,
                dimension,
                *wire0,
                *wire1,
                diagonal_indices,
                max_bytes,
            )?,
        },
        CircuitGate::Cphase {
            wire0,
            wire1,
            angle,
        } => DiagonalOp::Cphase {
            wire0: *wire0,
            wire1: *wire1,
            angle: *angle,
            indices: diagonal_index_map(
                basis_words,
                word_count,
                dimension,
                *wire0,
                *wire1,
                diagonal_indices,
                max_bytes,
            )?,
        },
        CircuitGate::Diagonal { wires, payload } => DiagonalOp::Static {
            wires: Arc::from(wires.clone().into_boxed_slice()),
            payload: Arc::from(payload.clone().into_boxed_slice()),
        },
        CircuitGate::Swap { .. } | CircuitGate::Iswap { .. } => return Ok(None),
    };
    Ok(Some(operation))
}

fn diagonal_index_map(
    basis_words: &[u64],
    word_count: usize,
    dimension: usize,
    wire0: usize,
    wire1: usize,
    cache: &mut HashMap<(usize, usize), Arc<[usize]>>,
    max_bytes: Option<u128>,
) -> Result<Arc<[usize]>, PauliError> {
    let key = if wire0 < wire1 {
        (wire0, wire1)
    } else {
        (wire1, wire0)
    };
    if let Some(indices) = cache.get(&key) {
        return Ok(indices.clone());
    }
    let mut indices = Vec::new();
    for index in 0..dimension {
        if bit_from_basis(basis_words, word_count, index, wire0) != 0
            && bit_from_basis(basis_words, word_count, index, wire1) != 0
        {
            indices.push(index);
        }
    }
    let bytes = (indices.len() as u128)
        .checked_mul(size_of::<usize>() as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating U1 circuit diagonal-index memory",
        })?;
    check_budget(bytes, max_bytes)?;
    let result: Arc<[usize]> = Arc::from(indices.into_boxed_slice());
    cache.insert(key, result.clone());
    Ok(result)
}

fn compile_non_diagonal_gate(
    gate: &CircuitGate,
    sector: &U1Sector,
    pair_maps: &mut HashMap<(usize, usize), Arc<[PairIndex]>>,
    max_bytes: Option<u128>,
) -> Result<CompiledU1Gate, PauliError> {
    match gate {
        CircuitGate::Swap { wire0, wire1 } => Ok(CompiledU1Gate::Swap {
            pairs: pair_map(sector, *wire0, *wire1, pair_maps, max_bytes)?,
        }),
        CircuitGate::Iswap {
            wire0,
            wire1,
            angle,
        } => Ok(CompiledU1Gate::Iswap {
            angle: *angle,
            pairs: pair_map(sector, *wire0, *wire1, pair_maps, max_bytes)?,
        }),
        CircuitGate::Rz { .. }
        | CircuitGate::Rzz { .. }
        | CircuitGate::Cz { .. }
        | CircuitGate::Cphase { .. }
        | CircuitGate::Diagonal { .. } => Err(PauliError::InvalidCircuit {
            context: "diagonal gate was not compiled into a diagonal block",
        }),
    }
}

fn pair_map(
    sector: &U1Sector,
    wire0: usize,
    wire1: usize,
    cache: &mut HashMap<(usize, usize), Arc<[PairIndex]>>,
    max_bytes: Option<u128>,
) -> Result<Arc<[PairIndex]>, PauliError> {
    let key = if wire0 < wire1 {
        (wire0, wire1)
    } else {
        (wire1, wire0)
    };
    if let Some(pairs) = cache.get(&key) {
        return Ok(pairs.clone());
    }
    let nqubits = sector.nqubits();
    let word_count = sector.word_count();
    let mut remaining = Vec::with_capacity(nqubits.saturating_sub(2));
    for wire in 0..nqubits {
        if wire != wire0 && wire != wire1 {
            remaining.push(wire);
        }
    }
    let occupied = sector.particle_number().saturating_sub(1);
    if occupied > remaining.len() {
        let empty: Arc<[PairIndex]> = Arc::from(Vec::<PairIndex>::new().into_boxed_slice());
        cache.insert(key, empty.clone());
        return Ok(empty);
    }
    let holes = remaining.len().saturating_sub(occupied);
    let enumerate_holes = holes < occupied;
    let choose = if enumerate_holes { holes } else { occupied };
    let pair_count = choose_count(remaining.len(), choose)?;
    let bytes = (pair_count as u128)
        .checked_mul(size_of::<PairIndex>() as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating U1 circuit pair-map memory",
        })?;
    check_budget(bytes, max_bytes)?;
    let mut pairs = Vec::with_capacity(pair_count);
    let mut selected = Vec::with_capacity(choose);
    enumerate_combinations(&remaining, choose, 0, &mut selected, &mut |chosen| {
        let mut words = vec![0_u64; word_count];
        set_bit(&mut words, wire1, true);
        if enumerate_holes {
            for wire in remaining.iter().copied() {
                set_bit(&mut words, wire, true);
            }
            for wire in chosen.iter().copied() {
                set_bit(&mut words, wire, false);
            }
        } else {
            for wire in chosen.iter().copied() {
                set_bit(&mut words, wire, true);
            }
        }
        let zero_one = sector.rank_words(&words).and_then(|rank| {
            usize::try_from(rank).map_err(|_| PauliError::Overflow {
                context: "converting U1 pair-map index",
            })
        });
        set_bit(&mut words, wire1, false);
        set_bit(&mut words, wire0, true);
        let one_zero = sector.rank_words(&words).and_then(|rank| {
            usize::try_from(rank).map_err(|_| PauliError::Overflow {
                context: "converting U1 pair-map index",
            })
        });
        if let (Ok(zero_one), Ok(one_zero)) = (zero_one, one_zero) {
            pairs.push(PairIndex { zero_one, one_zero });
        }
    });
    pairs.sort_unstable_by_key(|pair| pair.zero_one);
    let result: Arc<[PairIndex]> = Arc::from(pairs.into_boxed_slice());
    cache.insert(key, result.clone());
    Ok(result)
}

fn set_bit(words: &mut [u64], wire: usize, value: bool) {
    let mask = 1_u64 << (wire % 64);
    if value {
        words[wire / 64] |= mask;
    } else {
        words[wire / 64] &= !mask;
    }
}

fn enumerate_combinations(
    values: &[usize],
    choose: usize,
    start: usize,
    selected: &mut Vec<usize>,
    callback: &mut impl FnMut(&[usize]),
) {
    if selected.len() == choose {
        callback(selected);
        return;
    }
    let needed = choose - selected.len();
    let last = values.len().saturating_sub(needed);
    for index in start..=last {
        selected.push(values[index]);
        enumerate_combinations(values, choose, index + 1, selected, callback);
        selected.pop();
    }
}

fn choose_count(n: usize, k: usize) -> Result<usize, PauliError> {
    if k > n {
        return Ok(0);
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for index in 1..=k {
        result = result
            .checked_mul(n - k + index)
            .and_then(|value| value.checked_div(index))
            .ok_or(PauliError::Overflow {
                context: "sizing U1 circuit pair map",
            })?;
    }
    Ok(result)
}

fn checked_full_dimension(nqubits: usize, max_bytes: Option<u128>) -> Result<usize, PauliError> {
    if nqubits >= usize::BITS as usize {
        return Err(PauliError::Overflow {
            context: "sizing full U1 circuit output",
        });
    }
    let dimension = 1usize << nqubits;
    let bytes = (dimension as u128)
        .checked_mul(size_of::<Complex64>() as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating full U1 circuit output memory",
        })?;
    check_budget(bytes, max_bytes)?;
    Ok(dimension)
}

fn check_budget(bytes: u128, max_bytes: Option<u128>) -> Result<(), PauliError> {
    if let Some(limit) = max_bytes {
        if bytes > limit {
            return Err(PauliError::MemoryLimit {
                requested: bytes,
                limit,
            });
        }
    }
    Ok(())
}
