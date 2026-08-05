//! Rust-native fixed-particle-number circuit execution.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::mem::size_of;
use std::sync::Arc;

use rayon::prelude::*;

use crate::circuit_ir::{CircuitGate, CircuitProgram, ParameterExprNode};
use crate::error::PauliError;
use crate::operator::PauliOperator;
use crate::scalar::{is_exact_zero, Complex64};
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

#[derive(Clone, Debug)]
enum EvaluatedDiagonalOp {
    OneWire {
        wire: usize,
        phase: [Complex64; 2],
    },
    TwoWire {
        wire0: usize,
        wire1: usize,
        phase: [Complex64; 4],
    },
    General {
        wires: Arc<[usize]>,
        phase: Arc<[Complex64]>,
    },
}

impl DiagonalOp {
    fn is_sparse(&self) -> bool {
        matches!(self, Self::Cz { .. } | Self::Cphase { .. })
    }
}

#[derive(Clone, Debug)]
enum PairMicroOp {
    Swap,
    Iswap { angle: usize },
}

#[derive(Clone, Copy, Debug)]
struct Matrix2 {
    values: [[Complex64; 2]; 2],
}

impl Matrix2 {
    fn identity() -> Self {
        Self {
            values: [
                [Complex64::new(1.0, 0.0), Complex64::default()],
                [Complex64::default(), Complex64::new(1.0, 0.0)],
            ],
        }
    }

    fn multiply(self, right: Self) -> Self {
        let mut values = [[Complex64::default(); 2]; 2];
        for (row, output_row) in values.iter_mut().enumerate() {
            for (column, output) in output_row.iter_mut().enumerate() {
                *output = self.values[row][0] * right.values[0][column]
                    + self.values[row][1] * right.values[1][column];
            }
        }
        Self { values }
    }

    fn adjoint(self) -> Self {
        Self {
            values: [
                [self.values[0][0].conj(), self.values[1][0].conj()],
                [self.values[0][1].conj(), self.values[1][1].conj()],
            ],
        }
    }
}

#[derive(Clone, Debug)]
struct PairBlock {
    pairs: Arc<[PairIndex]>,
    operations: Arc<[PairMicroOp]>,
    static_matrix: Option<Matrix2>,
}

#[derive(Clone, Debug)]
struct ProjectedObservableTerm {
    z_words: Arc<[u64]>,
    coefficient: Complex64,
}

#[derive(Clone, Debug)]
struct ProjectedObservableGroup {
    x_words: Arc<[u64]>,
    x_weight: usize,
    terms: Arc<[ProjectedObservableTerm]>,
}

#[derive(Clone, Debug)]
struct ProjectedObservablePlan {
    groups: Arc<[ProjectedObservableGroup]>,
}

impl ProjectedObservablePlan {
    fn new(observable: &PauliOperator) -> Result<Self, PauliError> {
        let mut lookup = HashMap::<Vec<u64>, usize>::with_capacity(observable.terms().len());
        let mut groups: Vec<(Arc<[u64]>, Vec<ProjectedObservableTerm>)> = Vec::new();
        for term in observable.terms() {
            let x_words = term.word.x_words();
            let group_index = if let Some(index) = lookup.get(x_words) {
                *index
            } else {
                let index = groups.len();
                lookup.insert(x_words.to_vec(), index);
                groups.push((Arc::from(x_words.to_vec().into_boxed_slice()), Vec::new()));
                index
            };
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
                return Err(PauliError::NonFiniteCoefficient { index: group_index });
            }
            groups[group_index].1.push(ProjectedObservableTerm {
                z_words: Arc::from(term.word.z_words().to_vec().into_boxed_slice()),
                coefficient,
            });
        }
        let groups = groups
            .into_iter()
            .map(|(x_words, terms)| ProjectedObservableGroup {
                x_weight: x_words.iter().map(|word| word.count_ones() as usize).sum(),
                x_words,
                terms: Arc::from(terms.into_boxed_slice()),
            })
            .collect::<Vec<_>>();
        Ok(Self {
            groups: Arc::from(groups.into_boxed_slice()),
        })
    }

    fn for_each_transition(
        &self,
        state: &[Complex64],
        basis_words: &[u64],
        word_count: usize,
        sector: &U1Sector,
        mut callback: impl FnMut(usize, Complex64),
    ) -> Result<(), PauliError> {
        let mut destination_words = vec![0_u64; word_count];
        for (source, source_amplitude) in state.iter().copied().enumerate() {
            let source_words = &basis_words[source * word_count..(source + 1) * word_count];
            for group in self.groups.iter() {
                let intersection = group
                    .x_words
                    .iter()
                    .zip(source_words)
                    .map(|(x, source)| (x & source).count_ones() as usize)
                    .sum::<usize>();
                let destination_weight = sector
                    .particle_number()
                    .checked_add(group.x_weight)
                    .and_then(|weight| weight.checked_sub(intersection.checked_mul(2)?))
                    .ok_or(PauliError::Overflow {
                        context: "computing projected observable destination weight",
                    })?;
                if destination_weight != sector.particle_number() {
                    continue;
                }
                let destination = if group.x_weight == 0 {
                    source
                } else {
                    for (destination, (source, x)) in destination_words
                        .iter_mut()
                        .zip(source_words.iter().zip(group.x_words.iter()))
                    {
                        *destination = *source ^ *x;
                    }
                    usize::try_from(sector.rank_words(&destination_words)?).map_err(|_| {
                        PauliError::Overflow {
                            context: "converting projected observable destination index",
                        }
                    })?
                };
                let mut value = Complex64::default();
                for term in group.terms.iter() {
                    let parity = term
                        .z_words
                        .iter()
                        .zip(source_words)
                        .map(|(z, source)| (z & source).count_ones())
                        .sum::<u32>();
                    if parity & 1 == 0 {
                        value += term.coefficient;
                    } else {
                        value -= term.coefficient;
                    }
                }
                if !is_exact_zero(value) {
                    callback(destination, value * source_amplitude);
                }
            }
        }
        Ok(())
    }

    fn apply_into(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
        basis_words: &[u64],
        word_count: usize,
        sector: &U1Sector,
    ) -> Result<(), PauliError> {
        output.fill(Complex64::default());
        self.for_each_transition(
            state,
            basis_words,
            word_count,
            sector,
            |destination, value| {
                output[destination] += value;
            },
        )
    }

    fn expectation(
        &self,
        state: &[Complex64],
        basis_words: &[u64],
        word_count: usize,
        sector: &U1Sector,
    ) -> Result<Complex64, PauliError> {
        let mut value = Complex64::default();
        self.for_each_transition(
            state,
            basis_words,
            word_count,
            sector,
            |destination, term| {
                value += state[destination].conj() * term;
            },
        )?;
        Ok(value)
    }
}

#[derive(Clone, Debug)]
enum CompiledU1Gate {
    PairBlock(PairBlock),
    DiagonalBlock { operations: Arc<[DiagonalOp]> },
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
        let static_values = static_expression_values(&program);
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
                let operations = fold_static_diagonal_operations(operations, &static_values);
                gates.push(CompiledU1Gate::DiagonalBlock {
                    operations: Arc::from(operations.into_boxed_slice()),
                });
            } else {
                let (wire0, wire1) = non_diagonal_pair(&program.operations()[operation_index])
                    .ok_or(PauliError::InvalidCircuit {
                        context: "unsupported non-diagonal gate",
                    })?;
                let key = if wire0 < wire1 {
                    (wire0, wire1)
                } else {
                    (wire1, wire0)
                };
                let mut micro_operations = Vec::new();
                while operation_index < program.operations().len() {
                    let operation = &program.operations()[operation_index];
                    let Some((next_wire0, next_wire1)) = non_diagonal_pair(operation) else {
                        break;
                    };
                    let next_key = if next_wire0 < next_wire1 {
                        (next_wire0, next_wire1)
                    } else {
                        (next_wire1, next_wire0)
                    };
                    if next_key != key {
                        break;
                    }
                    micro_operations.push(pair_micro_operation(operation).ok_or(
                        PauliError::InvalidCircuit {
                            context: "unsupported non-diagonal gate",
                        },
                    )?);
                    operation_index += 1;
                }
                let pairs = pair_map(&sector, wire0, wire1, &mut pair_maps, max_bytes)?;
                let static_matrix = compose_static_pair_matrix(&micro_operations, &static_values);
                gates.push(CompiledU1Gate::PairBlock(PairBlock {
                    pairs,
                    operations: Arc::from(micro_operations.into_boxed_slice()),
                    static_matrix,
                }));
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
        let mut static_bytes = 0_u128;
        for gate in &gates {
            let CompiledU1Gate::DiagonalBlock { operations } = gate else {
                continue;
            };
            for operation in operations.iter() {
                let DiagonalOp::Static { payload, .. } = operation else {
                    continue;
                };
                let bytes = (payload.len() as u128)
                    .checked_mul(size_of::<Complex64>() as u128)
                    .ok_or(PauliError::Overflow {
                        context: "estimating U1 circuit static payload memory",
                    })?;
                static_bytes = static_bytes
                    .checked_add(bytes)
                    .ok_or(PauliError::Overflow {
                        context: "estimating U1 circuit static payload memory",
                    })?;
            }
        }
        let state_bytes = (dimension as u128)
            .checked_mul(size_of::<Complex64>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating U1 circuit state-vector memory",
            })?;
        check_budget(
            basis_bytes
                .checked_add(pair_bytes)
                .and_then(|bytes| bytes.checked_add(diagonal_bytes))
                .and_then(|bytes| bytes.checked_add(static_bytes))
                .and_then(|bytes| bytes.checked_add(state_bytes))
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
        self.probability_from_state(&state)
    }

    pub fn probability_from_state(&self, state: &[Complex64]) -> Result<Vec<f64>, PauliError> {
        self.validate_state(state)?;
        Ok(state.iter().map(|value| value.norm_sqr()).collect())
    }

    pub fn to_dense(
        &self,
        initial_state: &[Complex64],
        parameters: &[f64],
    ) -> Result<Vec<Complex64>, PauliError> {
        let state = self.run(initial_state, parameters)?;
        self.to_dense_from_state(&state)
    }

    pub fn to_dense_from_state(&self, state: &[Complex64]) -> Result<Vec<Complex64>, PauliError> {
        self.validate_state(state)?;
        let full_dimension = checked_full_dimension(self.nqubits(), self.max_bytes)?;
        let mut output = vec![Complex64::default(); full_dimension];
        for (index, amplitude) in state.iter().copied().enumerate() {
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
        self.probability_full_from_state(&state)
    }

    pub fn probability_full_from_state(&self, state: &[Complex64]) -> Result<Vec<f64>, PauliError> {
        self.validate_state(state)?;
        let full_dimension = checked_full_dimension(self.nqubits(), self.max_bytes)?;
        let mut output = vec![0.0; full_dimension];
        for (index, amplitude) in state.iter().copied().enumerate() {
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
        self.expectation_from_state(&state, observable)
    }

    pub fn expectation_from_state(
        &self,
        state: &[Complex64],
        observable: &PauliOperator,
    ) -> Result<Complex64, PauliError> {
        if observable.nqubits() != self.nqubits() {
            return Err(PauliError::IncompatibleQubitCounts {
                left: observable.nqubits(),
                right: self.nqubits(),
            });
        }
        self.validate_state(state)?;
        let projected = ProjectedObservablePlan::new(observable)?;
        projected.expectation(state, &self.basis_words, self.word_count, &self.sector)
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
        self.value_and_grad_from_final_state_with_values(&state, observable, &values)
    }

    /// Evaluate a gradient from a caller-supplied final state.
    ///
    /// The caller must supply the state produced by this plan's forward
    /// evolution from the same initial state and `parameters`. This method
    /// does not rerun the circuit or verify that precondition; use
    /// [`Self::value_and_grad`] when the final state is not already available.
    pub fn value_and_grad_from_state(
        &self,
        state: &[Complex64],
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
        self.validate_state(state)?;
        let values = self.program.evaluate_parameters(parameters)?;
        self.value_and_grad_from_final_state_with_values(state, observable, &values)
    }

    fn value_and_grad_from_final_state_with_values(
        &self,
        state: &[Complex64],
        observable: &PauliOperator,
        values: &[f64],
    ) -> Result<(f64, Vec<f64>), PauliError> {
        let projected = ProjectedObservablePlan::new(observable)?;
        let mut value_state = vec![Complex64::default(); state.len()];
        projected.apply_into(
            state,
            &mut value_state,
            &self.basis_words,
            self.word_count,
            &self.sector,
        )?;
        let value = inner_product(state, &value_state).re;
        let mut lambda = value_state;
        let mut node_adjoint = vec![0.0; self.program.parameter_program().len()];
        let mut state = state.to_vec();
        for gate in self.gates.iter().rev() {
            self.apply_inverse_gate(&mut state, gate, values)?;
            accumulate_gate_derivative(
                &state,
                &lambda,
                gate,
                values,
                &mut node_adjoint,
                self.word_count,
                &self.basis_words,
            )?;
            self.apply_inverse_gate(&mut lambda, gate, values)?;
        }
        let gradient = self
            .program
            .reverse_parameter_program(values, &node_adjoint)?;
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
                let evaluated = operations
                    .iter()
                    .map(|operation| evaluate_diagonal_operation(operation, values, inverse))
                    .collect::<Vec<_>>();
                for (index, value) in state.iter_mut().enumerate() {
                    let mut phase = Complex64::new(1.0, 0.0);
                    for operation in &evaluated {
                        phase *= evaluated_diagonal_phase(
                            index,
                            operation,
                            &self.basis_words,
                            self.word_count,
                        );
                    }
                    *value *= phase;
                }
            }
            CompiledU1Gate::PairBlock(block) => {
                let matrix = pair_block_matrix(block, values, inverse);
                apply_pair_matrix(state, &block.pairs, matrix);
            }
        }
        Ok(())
    }

    fn apply_inverse_gate(
        &self,
        state: &mut [Complex64],
        gate: &CompiledU1Gate,
        values: &[f64],
    ) -> Result<(), PauliError> {
        self.apply_gate(state, gate, values, true)
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_gate_derivative(
    before: &[Complex64],
    lambda: &[Complex64],
    gate: &CompiledU1Gate,
    values: &[f64],
    node_adjoint: &mut [f64],
    word_count: usize,
    basis_words: &[u64],
) -> Result<(), PauliError> {
    match gate {
        CompiledU1Gate::DiagonalBlock { operations } => {
            let evaluated = operations
                .iter()
                .map(|operation| evaluate_diagonal_operation(operation, values, false))
                .collect::<Vec<_>>();
            for index in 0..before.len() {
                let mut total_phase = Complex64::new(1.0, 0.0);
                for operation in &evaluated {
                    total_phase *=
                        evaluated_diagonal_phase(index, operation, basis_words, word_count);
                }
                let after = total_phase * before[index];
                for operation in operations.iter() {
                    let Some((angle, generator)) =
                        diagonal_generator(operation, index, basis_words, word_count)
                    else {
                        continue;
                    };
                    node_adjoint[angle] += 2.0 * (lambda[index].conj() * generator * after).re;
                }
            }
        }
        CompiledU1Gate::PairBlock(block) => {
            let matrices = block
                .operations
                .iter()
                .map(|operation| pair_micro_matrix(operation, values))
                .collect::<Vec<_>>();
            let mut prefix = vec![Matrix2::identity(); matrices.len() + 1];
            for (index, matrix) in matrices.iter().enumerate() {
                prefix[index + 1] = matrix.multiply(prefix[index]);
            }
            let mut suffix = vec![Matrix2::identity(); matrices.len() + 1];
            for index in (0..matrices.len()).rev() {
                suffix[index] = suffix[index + 1].multiply(matrices[index]);
            }
            for (index, operation) in block.operations.iter().enumerate() {
                let PairMicroOp::Iswap { angle } = operation else {
                    continue;
                };
                let theta = values[*angle] * PI / 2.0;
                let derivative = Matrix2 {
                    values: [
                        [
                            Complex64::new(-PI / 2.0 * theta.sin(), 0.0),
                            Complex64::new(0.0, PI / 2.0 * theta.cos()),
                        ],
                        [
                            Complex64::new(0.0, PI / 2.0 * theta.cos()),
                            Complex64::new(-PI / 2.0 * theta.sin(), 0.0),
                        ],
                    ],
                };
                let derivative = suffix[index + 1]
                    .multiply(derivative)
                    .multiply(prefix[index]);
                let mut result = 0.0;
                for pair in block.pairs.iter() {
                    let left = before[pair.zero_one];
                    let right = before[pair.one_zero];
                    let dleft = derivative.values[0][0] * left + derivative.values[0][1] * right;
                    let dright = derivative.values[1][0] * left + derivative.values[1][1] * right;
                    result += 2.0
                        * (lambda[pair.zero_one].conj() * dleft
                            + lambda[pair.one_zero].conj() * dright)
                            .re;
                }
                node_adjoint[*angle] += result;
            }
        }
    }
    Ok(())
}

fn bit_from_basis(basis_words: &[u64], word_count: usize, index: usize, wire: usize) -> u8 {
    ((basis_words[index * word_count + wire / 64] >> (wire % 64)) & 1) as u8
}

fn non_diagonal_pair(gate: &CircuitGate) -> Option<(usize, usize)> {
    match gate {
        CircuitGate::Swap { wire0, wire1 } | CircuitGate::Iswap { wire0, wire1, .. } => {
            Some((*wire0, *wire1))
        }
        CircuitGate::Rz { .. }
        | CircuitGate::Rzz { .. }
        | CircuitGate::Cz { .. }
        | CircuitGate::Cphase { .. }
        | CircuitGate::Diagonal { .. } => None,
    }
}

fn pair_micro_operation(gate: &CircuitGate) -> Option<PairMicroOp> {
    match gate {
        CircuitGate::Swap { .. } => Some(PairMicroOp::Swap),
        CircuitGate::Iswap { angle, .. } => Some(PairMicroOp::Iswap { angle: *angle }),
        _ => None,
    }
}

fn static_expression_values(program: &CircuitProgram) -> Vec<Option<f64>> {
    let mut values: Vec<Option<f64>> = Vec::with_capacity(program.parameter_program().len());
    for node in program.parameter_program() {
        let value = match *node {
            ParameterExprNode::Constant(value) => Some(value),
            ParameterExprNode::Slot(_) => None,
            ParameterExprNode::Neg(child) => values[child].map(|value| -value),
            ParameterExprNode::Add(left, right) => values[left]
                .zip(values[right])
                .map(|(left, right)| left + right),
            ParameterExprNode::Sub(left, right) => values[left]
                .zip(values[right])
                .map(|(left, right)| left - right),
            ParameterExprNode::Mul(left, right) => values[left]
                .zip(values[right])
                .map(|(left, right)| left * right),
            ParameterExprNode::Div(left, right) => values[left]
                .zip(values[right])
                .and_then(|(left, right)| (right != 0.0).then_some(left / right)),
        };
        values.push(value.filter(|value| value.is_finite()));
    }
    values
}

fn static_diagonal_payload(
    operation: &DiagonalOp,
    static_values: &[Option<f64>],
) -> Option<(Arc<[usize]>, Vec<Complex64>)> {
    match operation {
        DiagonalOp::Rz { wire, angle } => {
            let value = static_values.get(*angle)?.as_ref()?;
            Some((
                Arc::from(vec![*wire].into_boxed_slice()),
                vec![
                    Complex64::from_polar(1.0, -0.5 * value),
                    Complex64::from_polar(1.0, 0.5 * value),
                ],
            ))
        }
        DiagonalOp::Rzz {
            wire0,
            wire1,
            angle,
        } => {
            let value = static_values.get(*angle)?.as_ref()?;
            let phases =
                [1.0, -1.0, -1.0, 1.0].map(|sign| Complex64::from_polar(1.0, -0.5 * value * sign));
            Some((
                Arc::from(vec![*wire0, *wire1].into_boxed_slice()),
                phases.into(),
            ))
        }
        DiagonalOp::Cz { wire0, wire1, .. } => Some((
            Arc::from(vec![*wire0, *wire1].into_boxed_slice()),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        )),
        DiagonalOp::Cphase {
            wire0,
            wire1,
            angle,
            ..
        } => {
            let value = static_values.get(*angle)?.as_ref()?;
            Some((
                Arc::from(vec![*wire0, *wire1].into_boxed_slice()),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::from_polar(1.0, *value),
                ],
            ))
        }
        DiagonalOp::Static { wires, payload } => Some((wires.clone(), payload.to_vec())),
    }
}

fn fold_static_diagonal_operations(
    operations: Vec<DiagonalOp>,
    static_values: &[Option<f64>],
) -> Vec<DiagonalOp> {
    let mut folded = Vec::with_capacity(operations.len());
    let mut index = 0;
    while index < operations.len() {
        if operations[index].is_sparse() {
            folded.push(operations[index].clone());
            index += 1;
            continue;
        }
        let Some((wires, mut payload)) = static_diagonal_payload(&operations[index], static_values)
        else {
            folded.push(operations[index].clone());
            index += 1;
            continue;
        };
        let mut end = index + 1;
        while end < operations.len() {
            if operations[end].is_sparse() {
                break;
            }
            let Some((next_wires, next_payload)) =
                static_diagonal_payload(&operations[end], static_values)
            else {
                break;
            };
            if next_wires.as_ref() != wires.as_ref() || next_payload.len() != payload.len() {
                break;
            }
            for (left, right) in payload.iter_mut().zip(next_payload) {
                *left *= right;
            }
            end += 1;
        }
        if end > index + 1 || !matches!(operations[index], DiagonalOp::Static { .. }) {
            folded.push(DiagonalOp::Static {
                wires,
                payload: Arc::from(payload.into_boxed_slice()),
            });
        } else {
            folded.push(operations[index].clone());
        }
        index = end;
    }
    folded
}

fn pair_micro_matrix(operation: &PairMicroOp, values: &[f64]) -> Matrix2 {
    match operation {
        PairMicroOp::Swap => Matrix2 {
            values: [
                [Complex64::default(), Complex64::new(1.0, 0.0)],
                [Complex64::new(1.0, 0.0), Complex64::default()],
            ],
        },
        PairMicroOp::Iswap { angle } => iswap_matrix(values[*angle]),
    }
}

fn iswap_matrix(value: f64) -> Matrix2 {
    let theta = value * PI / 2.0;
    Matrix2 {
        values: [
            [
                Complex64::new(theta.cos(), 0.0),
                Complex64::new(0.0, theta.sin()),
            ],
            [
                Complex64::new(0.0, theta.sin()),
                Complex64::new(theta.cos(), 0.0),
            ],
        ],
    }
}

fn compose_static_pair_matrix(
    operations: &[PairMicroOp],
    static_values: &[Option<f64>],
) -> Option<Matrix2> {
    let mut matrix = Matrix2::identity();
    for operation in operations {
        if let PairMicroOp::Iswap { angle } = operation {
            static_values.get(*angle)?.as_ref()?;
        }
        let operation_matrix = match operation {
            PairMicroOp::Swap => pair_micro_matrix(operation, &[]),
            PairMicroOp::Iswap { angle } => {
                iswap_matrix(static_values.get(*angle)?.as_ref().copied()?)
            }
        };
        matrix = operation_matrix.multiply(matrix);
    }
    Some(matrix)
}

fn pair_block_matrix(block: &PairBlock, values: &[f64], inverse: bool) -> Matrix2 {
    let matrix = block.static_matrix.unwrap_or_else(|| {
        block
            .operations
            .iter()
            .fold(Matrix2::identity(), |total, operation| {
                pair_micro_matrix(operation, values).multiply(total)
            })
    });
    let matrix = if inverse { matrix.adjoint() } else { matrix };
    debug_assert!(
        (matrix.values[0][1] - matrix.values[1][0]).norm() < 1e-12
            && (matrix.values[0][0] - matrix.values[1][1]).norm() < 1e-12,
        "sorted pair-map keys require symmetric pair matrices"
    );
    matrix
}

fn apply_pair_matrix(state: &mut [Complex64], pairs: &[PairIndex], matrix: Matrix2) {
    if pairs.len() >= U1_CIRCUIT_PARALLEL_PAIR_THRESHOLD {
        let state_ptr = state.as_mut_ptr() as usize;
        pairs.par_iter().for_each(|pair| {
            // SAFETY: `state_ptr` comes from the live mutable `state` slice and
            // remains valid for the duration of this parallel section. `pair_map`
            // constructs both endpoints as valid sector indices, and each pair
            // has distinct endpoints. Its construction enumerates every
            // assignment of the other wires exactly once, so endpoints from
            // different pairs are disjoint; parallel workers therefore never
            // alias a read or write location.
            unsafe {
                let state_ptr = state_ptr as *mut Complex64;
                let left = *state_ptr.add(pair.zero_one);
                let right = *state_ptr.add(pair.one_zero);
                *state_ptr.add(pair.zero_one) =
                    matrix.values[0][0] * left + matrix.values[0][1] * right;
                *state_ptr.add(pair.one_zero) =
                    matrix.values[1][0] * left + matrix.values[1][1] * right;
            }
        });
    } else {
        for pair in pairs {
            let left = state[pair.zero_one];
            let right = state[pair.one_zero];
            state[pair.zero_one] = matrix.values[0][0] * left + matrix.values[0][1] * right;
            state[pair.one_zero] = matrix.values[1][0] * left + matrix.values[1][1] * right;
        }
    }
}

fn evaluate_diagonal_operation(
    operation: &DiagonalOp,
    values: &[f64],
    inverse: bool,
) -> EvaluatedDiagonalOp {
    let phase = |angle: f64| {
        let value = Complex64::from_polar(1.0, angle);
        if inverse {
            value.conj()
        } else {
            value
        }
    };
    match operation {
        DiagonalOp::Rz { wire, angle } => EvaluatedDiagonalOp::OneWire {
            wire: *wire,
            phase: [phase(-0.5 * values[*angle]), phase(0.5 * values[*angle])],
        },
        DiagonalOp::Rzz {
            wire0,
            wire1,
            angle,
        } => EvaluatedDiagonalOp::TwoWire {
            wire0: *wire0,
            wire1: *wire1,
            phase: [
                phase(-0.5 * values[*angle]),
                phase(0.5 * values[*angle]),
                phase(0.5 * values[*angle]),
                phase(-0.5 * values[*angle]),
            ],
        },
        DiagonalOp::Cz { wire0, wire1, .. } => EvaluatedDiagonalOp::TwoWire {
            wire0: *wire0,
            wire1: *wire1,
            phase: [
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        },
        DiagonalOp::Cphase {
            wire0,
            wire1,
            angle,
            ..
        } => EvaluatedDiagonalOp::TwoWire {
            wire0: *wire0,
            wire1: *wire1,
            phase: [
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                phase(values[*angle]),
            ],
        },
        DiagonalOp::Static { wires, payload } => {
            let phase = if inverse {
                Arc::from(
                    payload
                        .iter()
                        .map(|value| value.conj())
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                )
            } else {
                payload.clone()
            };
            EvaluatedDiagonalOp::General {
                wires: wires.clone(),
                phase,
            }
        }
    }
}

fn evaluated_diagonal_phase(
    index: usize,
    operation: &EvaluatedDiagonalOp,
    basis_words: &[u64],
    word_count: usize,
) -> Complex64 {
    match operation {
        EvaluatedDiagonalOp::OneWire { wire, phase } => {
            phase[bit_from_basis(basis_words, word_count, index, *wire) as usize]
        }
        EvaluatedDiagonalOp::TwoWire {
            wire0,
            wire1,
            phase,
        } => {
            let local = (bit_from_basis(basis_words, word_count, index, *wire0) as usize) * 2
                + bit_from_basis(basis_words, word_count, index, *wire1) as usize;
            phase[local]
        }
        EvaluatedDiagonalOp::General { wires, phase } => {
            let mut local = 0usize;
            for wire in wires.iter().copied() {
                local =
                    (local << 1) | bit_from_basis(basis_words, word_count, index, wire) as usize;
            }
            phase[local]
        }
    }
}

fn diagonal_generator(
    operation: &DiagonalOp,
    index: usize,
    basis_words: &[u64],
    word_count: usize,
) -> Option<(usize, Complex64)> {
    match operation {
        DiagonalOp::Rz { wire, angle } => {
            let z = if bit_from_basis(basis_words, word_count, index, *wire) == 0 {
                1.0
            } else {
                -1.0
            };
            Some((*angle, Complex64::new(0.0, -0.5 * z)))
        }
        DiagonalOp::Rzz {
            wire0,
            wire1,
            angle,
        } => {
            let z0 = if bit_from_basis(basis_words, word_count, index, *wire0) == 0 {
                1.0
            } else {
                -1.0
            };
            let z1 = if bit_from_basis(basis_words, word_count, index, *wire1) == 0 {
                1.0
            } else {
                -1.0
            };
            Some((*angle, Complex64::new(0.0, -0.5 * z0 * z1)))
        }
        DiagonalOp::Cphase {
            wire0,
            wire1,
            angle,
            ..
        } if bit_from_basis(basis_words, word_count, index, *wire0) != 0
            && bit_from_basis(basis_words, word_count, index, *wire1) != 0 =>
        {
            Some((*angle, Complex64::new(0.0, 1.0)))
        }
        DiagonalOp::Cz { .. } | DiagonalOp::Cphase { .. } | DiagonalOp::Static { .. } => None,
    }
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
    if sector.particle_number() == 0 || occupied > remaining.len() {
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
    let mut words = vec![0_u64; word_count];
    enumerate_combinations(&remaining, choose, 0, &mut selected, &mut |chosen| {
        words.fill(0);
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

#[cfg(test)]
mod tests {
    use super::{
        apply_pair_matrix, pair_map, Matrix2, PairIndex, U1_CIRCUIT_PARALLEL_PAIR_THRESHOLD,
    };
    use crate::scalar::Complex64;
    use crate::sector::U1Sector;
    use std::collections::HashMap;

    #[test]
    fn large_pair_map_parallel_kernel_matches_serial_reference() {
        let sector = U1Sector::new(20, 10).expect("valid sector");
        let dimension = sector.dimension().expect("bounded dimension");
        let mut cache = HashMap::new();
        let pairs = pair_map(&sector, 0, 1, &mut cache, None).expect("pair map");
        assert!(pairs.len() >= U1_CIRCUIT_PARALLEL_PAIR_THRESHOLD);

        let mut seen = vec![false; dimension];
        for pair in pairs.iter() {
            assert!(pair.zero_one < dimension);
            assert!(pair.one_zero < dimension);
            assert_ne!(pair.zero_one, pair.one_zero);
            assert!(!seen[pair.zero_one]);
            assert!(!seen[pair.one_zero]);
            seen[pair.zero_one] = true;
            seen[pair.one_zero] = true;
        }

        let matrix = Matrix2 {
            values: [
                [Complex64::new(0.8, 0.1), Complex64::new(-0.2, 0.3)],
                [Complex64::new(0.4, -0.5), Complex64::new(0.6, 0.2)],
            ],
        };
        let initial = (0..dimension)
            .map(|index| Complex64::new((index % 17) as f64, (index % 11) as f64))
            .collect::<Vec<_>>();
        let mut parallel = initial.clone();
        let mut serial = initial;
        apply_pair_matrix(&mut parallel, &pairs, matrix);
        for PairIndex { zero_one, one_zero } in pairs.iter().copied() {
            let left = serial[zero_one];
            let right = serial[one_zero];
            serial[zero_one] = matrix.values[0][0] * left + matrix.values[0][1] * right;
            serial[one_zero] = matrix.values[1][0] * left + matrix.values[1][1] * right;
        }
        assert_eq!(parallel, serial);
    }
}
