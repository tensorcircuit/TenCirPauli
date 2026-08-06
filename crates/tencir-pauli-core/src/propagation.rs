//! Rust-native Heisenberg Pauli propagation.
//!
//! The engine deliberately keeps its dynamic state separate from the public
//! `PauliOperator`.  Small and medium systems use an inline two-word key; only
//! wider systems fall back to heap-backed masks.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::mem::size_of;
use std::sync::Arc;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::error::PauliError;
use crate::gate::{Clifford1, Clifford2, GateKind, GateOperation, ParameterRef, RotationAxis};
use crate::operator::{PauliOperator, PauliTerm};
use crate::scalar::{is_exact_zero, Complex64};
use crate::word::{packed_word_count, PauliPhase, PauliWord};

const BATCH_PARALLEL_WORK_THRESHOLD: usize = 128;

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

/// A deterministic value and frozen-support reverse gradient.
#[derive(Clone, Debug)]
pub struct PropagationValueAndGradient {
    pub value: f64,
    pub gradient: Vec<f64>,
}

/// Values and row-wise frozen-support gradients for independent observables.
#[derive(Clone, Debug)]
pub struct PropagationBatchValueAndGradient {
    pub values: Vec<f64>,
    pub gradients: Vec<f64>,
}

#[derive(Clone, Debug)]
struct CompiledPropagationProgram {
    nqubits: usize,
    operations: Arc<[GateOperation]>,
    initial_state: ProductState,
    max_weight: Option<usize>,
    max_bytes: Option<u128>,
    nparameters: usize,
    transition_bytes: usize,
}

/// Immutable validated gate tape shared by propagation engines.
#[derive(Clone, Debug)]
pub struct PropagationTape {
    nqubits: usize,
    operations: Arc<[GateOperation]>,
    nparameters: usize,
    transition_bytes: usize,
}

impl PropagationTape {
    pub fn new(
        nqubits: usize,
        operations: Vec<GateOperation>,
        max_bytes: Option<u128>,
    ) -> Result<Arc<Self>, PauliError> {
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
            transition_bytes,
            max_bytes,
            "propagation transition storage",
        )?;
        Ok(Arc::new(Self {
            nqubits,
            operations: Arc::from(operations.into_boxed_slice()),
            nparameters,
            transition_bytes,
        }))
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

    pub fn operations(&self) -> &[GateOperation] {
        &self.operations
    }
}

/// An immutable compiled propagation engine.
#[derive(Clone, Debug)]
pub struct PropagationEngine {
    program: Arc<CompiledPropagationProgram>,
    observable: PauliOperator,
    term_count: usize,
    hermitian: bool,
}

/// Independent observable propagation sharing one immutable compiled program.
#[derive(Clone, Debug)]
pub struct PropagationBatch {
    program: Arc<CompiledPropagationProgram>,
    engines: Vec<PropagationEngine>,
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
        let program = compile_program(nqubits, operations, initial_state, max_weight, max_bytes)?;
        Self::from_program(program, observable)
    }

    /// Build an engine from a previously validated shared gate tape.
    pub fn from_tape(
        tape: Arc<PropagationTape>,
        observable: PauliOperator,
        initial_state: ProductState,
        max_weight: Option<usize>,
        max_bytes: Option<u128>,
    ) -> Result<Self, PauliError> {
        let program = program_from_tape(tape, initial_state, max_weight, max_bytes)?;
        Self::from_program(program, observable)
    }

    fn from_program(
        program: Arc<CompiledPropagationProgram>,
        observable: PauliOperator,
    ) -> Result<Self, PauliError> {
        if observable.nqubits() != program.nqubits {
            return Err(PauliError::IncompatibleQubitCounts {
                left: observable.nqubits(),
                right: program.nqubits,
            });
        }
        let observable_bytes = observable
            .terms()
            .len()
            .checked_mul(dynamic_term_storage_bytes(program.nqubits)?)
            .ok_or(PauliError::Overflow {
                context: "estimating propagation observable storage",
            })?;
        check_budget(
            observable_bytes,
            program.max_bytes,
            "propagation observable storage",
        )?;
        check_budget(
            observable_bytes
                .checked_add(program.transition_bytes)
                .ok_or(PauliError::Overflow {
                    context: "estimating propagation engine storage",
                })?,
            program.max_bytes,
            "propagation engine storage",
        )?;
        let hermitian = observable.is_hermitian(0.0);
        let term_count = observable.terms().len();
        Ok(Self {
            program,
            observable,
            term_count,
            hermitian,
        })
    }

    pub fn nqubits(&self) -> usize {
        self.program.nqubits
    }

    pub fn nparameters(&self) -> usize {
        self.program.nparameters
    }

    pub fn gate_count(&self) -> usize {
        self.program.operations.len()
    }

    pub fn term_count(&self) -> usize {
        self.term_count
    }

    pub fn max_weight(&self) -> Option<usize> {
        self.program.max_weight
    }

    pub fn is_exact(&self) -> bool {
        self.program
            .max_weight
            .is_none_or(|cutoff| cutoff >= self.program.nqubits)
    }

    pub fn is_hermitian_observable(&self) -> bool {
        self.hermitian
    }

    /// Propagate and return a scalar product-state expectation.
    pub fn expectation(&self, parameters: &[f64]) -> Result<f64, PauliError> {
        if !self.hermitian {
            return Err(PauliError::NonHermitianExpectation);
        }
        let result = self.propagate_dynamic(parameters)?;
        Ok(expectation_from_dynamic_terms(
            &result.terms,
            &self.program.initial_state,
            self.program.nqubits,
        ))
    }

    /// Evaluate the executed sparse trace and reverse only its retained edges.
    pub fn value_and_grad(
        &self,
        parameters: &[f64],
        checkpoint_interval: Option<usize>,
    ) -> Result<PropagationValueAndGradient, PauliError> {
        if !self.hermitian {
            return Err(PauliError::NonHermitianExpectation);
        }
        validate_parameters(parameters, self.program.nparameters)?;
        let interval = checkpoint_interval.unwrap_or_else(|| {
            let gates = self.program.operations.len().saturating_add(1);
            (gates as f64).sqrt().ceil() as usize
        });
        if interval == 0 {
            return Err(PauliError::InvalidCheckpointInterval);
        }

        let exact = self.is_exact();
        let cutoff = (!exact).then_some(self.program.max_weight).flatten();
        let initial = initial_dynamic_terms(self.program.nqubits, &self.observable, cutoff);
        let mut checkpoints = vec![(0usize, initial)];
        let mut current = checkpoints[0].1.clone();
        let mut checkpoint_bytes = dynamic_terms_storage_bytes(&current, self.program.nqubits)?;
        check_budget(
            checkpoint_bytes,
            self.program.max_bytes,
            "reverse checkpoint storage",
        )?;
        for (step, operation) in self.program.operations.iter().rev().enumerate() {
            current =
                apply_operation(self.program.nqubits, operation, current, parameters, cutoff)?;
            let boundary = step + 1;
            if boundary % interval == 0 || boundary == self.program.operations.len() {
                checkpoint_bytes = checkpoint_bytes
                    .checked_add(dynamic_terms_storage_bytes(&current, self.program.nqubits)?)
                    .ok_or(PauliError::Overflow {
                        context: "estimating reverse checkpoint storage",
                    })?;
                check_budget(
                    checkpoint_bytes,
                    self.program.max_bytes,
                    "reverse checkpoint storage",
                )?;
                checkpoints.push((boundary, current.clone()));
            }
        }
        let mut value = 0.0;
        let mut lambda = Vec::with_capacity(current.len());
        for term in &current {
            let expectation =
                expectation_of_key(&term.key, &self.program.initial_state, self.program.nqubits);
            value += term.coefficient.re * expectation;
            lambda.push(expectation);
        }
        let gradient_bytes = self
            .program
            .nparameters
            .checked_mul(size_of::<f64>())
            .ok_or(PauliError::Overflow {
                context: "estimating reverse gradient storage",
            })?;
        check_budget(
            gradient_bytes,
            self.program.max_bytes,
            "reverse gradient storage",
        )?;
        let mut gradient = vec![0.0; self.program.nparameters];
        let mut output_indices =
            FxHashMap::with_capacity_and_hasher(current.len(), Default::default());
        for checkpoint_index in (0..checkpoints.len().saturating_sub(1)).rev() {
            let (start, block_start) = &checkpoints[checkpoint_index];
            let (end, _) = &checkpoints[checkpoint_index + 1];
            if *end - *start == 1 {
                let operation =
                    &self.program.operations[self.program.operations.len() - 1 - *start];
                lambda = reverse_frame(
                    self.program.nqubits,
                    operation,
                    block_start,
                    &checkpoints[checkpoint_index + 1].1,
                    parameters,
                    cutoff,
                    &lambda,
                    &mut output_indices,
                    &mut gradient,
                )?;
                continue;
            }
            let mut block_states = Vec::with_capacity(end - start + 1);
            block_states.push(block_start.clone());
            let mut block_bytes = dynamic_terms_storage_bytes(block_start, self.program.nqubits)?;
            let mut state = block_start.clone();
            for step in *start..*end {
                state = apply_operation(
                    self.program.nqubits,
                    &self.program.operations[self.program.operations.len() - 1 - step],
                    state,
                    parameters,
                    cutoff,
                )?;
                block_bytes = block_bytes
                    .checked_add(dynamic_terms_storage_bytes(&state, self.program.nqubits)?)
                    .ok_or(PauliError::Overflow {
                        context: "estimating reverse replay storage",
                    })?;
                check_budget(
                    block_bytes,
                    self.program.max_bytes,
                    "reverse replay storage",
                )?;
                block_states.push(state.clone());
            }

            for local_step in (0..(*end - *start)).rev() {
                let global_step = *start + local_step;
                let operation =
                    &self.program.operations[self.program.operations.len() - 1 - global_step];
                let input = &block_states[local_step];
                let output = &block_states[local_step + 1];
                lambda = reverse_frame(
                    self.program.nqubits,
                    operation,
                    input,
                    output,
                    parameters,
                    cutoff,
                    &lambda,
                    &mut output_indices,
                    &mut gradient,
                )?;
            }
        }
        Ok(PropagationValueAndGradient { value, gradient })
    }

    /// Evaluate already propagated terms against the compiled product state.
    ///
    /// This method is defined for Hermitian observables. Callers that do not
    /// control the source of `terms` should check
    /// [`Self::is_hermitian_observable`] first; the scalar result keeps only
    /// the real coefficient by design.
    pub fn expectation_of_terms(&self, terms: &[PauliTerm]) -> f64 {
        debug_assert!(
            self.hermitian,
            "expectation_of_terms requires a Hermitian observable"
        );
        expectation_from_terms(terms, &self.program.initial_state, self.program.nqubits)
    }

    /// Propagate and return canonical public terms plus structural counters.
    pub fn propagate(&self, parameters: &[f64]) -> Result<PropagationResult, PauliError> {
        let result = self.propagate_dynamic(parameters)?;
        let mut public_terms = Vec::with_capacity(result.terms.len());
        let mut final_weight_counts = vec![0usize; self.program.nqubits.saturating_add(1)];
        for term in result.terms {
            let word = term.key.to_word(self.program.nqubits)?;
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
                gate_count: self.program.operations.len(),
                initial_terms: result.initial_terms,
                final_terms: public_terms.len(),
                peak_terms: result.peak_terms,
                estimated_peak_bytes: result.estimated_peak_bytes,
                final_weight_counts,
            },
            terms: public_terms,
        })
    }

    /// Propagate without materializing canonical public words and return the
    /// scalar expectation together with structural counters.
    pub fn profile_dynamic(
        &self,
        parameters: &[f64],
    ) -> Result<(f64, PropagationStats), PauliError> {
        if !self.hermitian {
            return Err(PauliError::NonHermitianExpectation);
        }
        let result = self.propagate_dynamic(parameters)?;
        let mut final_weight_counts = vec![0usize; self.program.nqubits.saturating_add(1)];
        for term in &result.terms {
            if let Some(count) = final_weight_counts.get_mut(term.key.weight(self.program.nqubits))
            {
                *count += 1;
            }
        }
        let stats = PropagationStats {
            gate_count: self.program.operations.len(),
            initial_terms: result.initial_terms,
            final_terms: result.terms.len(),
            peak_terms: result.peak_terms,
            estimated_peak_bytes: result.estimated_peak_bytes,
            final_weight_counts,
        };
        let value = expectation_from_dynamic_terms(
            &result.terms,
            &self.program.initial_state,
            self.program.nqubits,
        );
        Ok((value, stats))
    }

    fn propagate_dynamic(
        &self,
        parameters: &[f64],
    ) -> Result<DynamicPropagationResult, PauliError> {
        validate_parameters(parameters, self.program.nparameters)?;
        let exact = self.is_exact();
        let cutoff = (!exact).then_some(self.program.max_weight).flatten();
        let mut terms = self
            .observable
            .terms()
            .iter()
            .filter_map(|term| {
                let key = PackedKey::from_word(&term.word);
                (exact || cutoff.is_none_or(|limit| key.weight(self.program.nqubits) <= limit))
                    .then_some(DynamicTerm {
                        key,
                        coefficient: term.coefficient,
                    })
            })
            .collect::<Vec<_>>();
        let initial_terms = terms.len();
        let mut peak_terms = initial_terms;
        let mut estimated_peak_bytes = initial_terms
            .checked_mul(dynamic_term_storage_bytes(self.program.nqubits)?)
            .ok_or(PauliError::Overflow {
                context: "estimating initial propagation storage",
            })?;
        check_budget(
            estimated_peak_bytes,
            self.program.max_bytes,
            "initial propagation storage",
        )?;

        for operation in self.program.operations.iter().rev() {
            let branch_factor = operation_branch_factor(operation);
            let candidate_count =
                terms
                    .len()
                    .checked_mul(branch_factor)
                    .ok_or(PauliError::Overflow {
                        context: "estimating propagation contribution count",
                    })?;
            let candidate_bytes = candidate_count
                .checked_mul(dynamic_term_storage_bytes(self.program.nqubits)?)
                .ok_or(PauliError::Overflow {
                    context: "estimating propagation contribution storage",
                })?;
            estimated_peak_bytes = estimated_peak_bytes.max(candidate_bytes);
            check_budget(
                candidate_bytes,
                self.program.max_bytes,
                "propagation contribution storage",
            )?;
            terms = apply_operation(self.program.nqubits, operation, terms, parameters, cutoff)?;
            peak_terms = peak_terms.max(terms.len());
            let actual_bytes = terms
                .len()
                .checked_mul(dynamic_term_storage_bytes(self.program.nqubits)?)
                .ok_or(PauliError::Overflow {
                    context: "estimating propagated term storage",
                })?;
            estimated_peak_bytes = estimated_peak_bytes.max(actual_bytes);
        }
        Ok(DynamicPropagationResult {
            terms,
            initial_terms,
            peak_terms,
            estimated_peak_bytes,
        })
    }
}

fn compile_program(
    nqubits: usize,
    operations: Vec<GateOperation>,
    initial_state: ProductState,
    max_weight: Option<usize>,
    max_bytes: Option<u128>,
) -> Result<Arc<CompiledPropagationProgram>, PauliError> {
    validate_state(nqubits, &initial_state)?;
    let tape = PropagationTape::new(nqubits, operations, max_bytes)?;
    program_from_tape(tape, initial_state, max_weight, max_bytes)
}

fn program_from_tape(
    tape: Arc<PropagationTape>,
    initial_state: ProductState,
    max_weight: Option<usize>,
    max_bytes: Option<u128>,
) -> Result<Arc<CompiledPropagationProgram>, PauliError> {
    validate_state(tape.nqubits, &initial_state)?;
    Ok(Arc::new(CompiledPropagationProgram {
        nqubits: tape.nqubits,
        operations: Arc::clone(&tape.operations),
        initial_state,
        max_weight,
        max_bytes,
        nparameters: tape.nparameters,
        transition_bytes: tape.transition_bytes,
    }))
}

impl PropagationBatch {
    /// Compile independent observables over one shared tape/state program.
    pub fn new(
        nqubits: usize,
        operations: Vec<GateOperation>,
        observables: Vec<PauliOperator>,
        initial_state: ProductState,
        max_weight: Option<usize>,
        max_bytes: Option<u128>,
    ) -> Result<Self, PauliError> {
        let program = compile_program(nqubits, operations, initial_state, max_weight, max_bytes)?;
        Self::from_program(program, observables)
    }

    /// Build independent observable engines from a previously validated tape.
    pub fn from_tape(
        tape: Arc<PropagationTape>,
        observables: Vec<PauliOperator>,
        initial_state: ProductState,
        max_weight: Option<usize>,
        max_bytes: Option<u128>,
    ) -> Result<Self, PauliError> {
        let program = program_from_tape(tape, initial_state, max_weight, max_bytes)?;
        Self::from_program(program, observables)
    }

    fn from_program(
        program: Arc<CompiledPropagationProgram>,
        observables: Vec<PauliOperator>,
    ) -> Result<Self, PauliError> {
        let observable_bytes = observables.iter().try_fold(0usize, |sum, observable| {
            if observable.nqubits() != program.nqubits {
                return Err(PauliError::IncompatibleQubitCounts {
                    left: observable.nqubits(),
                    right: program.nqubits,
                });
            }
            let bytes = observable
                .terms()
                .len()
                .checked_mul(dynamic_term_storage_bytes(program.nqubits)?)
                .ok_or(PauliError::Overflow {
                    context: "estimating propagation batch observable storage",
                })?;
            sum.checked_add(bytes).ok_or(PauliError::Overflow {
                context: "estimating propagation batch observable storage",
            })
        })?;
        let output_bytes = observables
            .len()
            .checked_mul(size_of::<f64>())
            .and_then(|values| {
                observables
                    .len()
                    .checked_mul(program.nparameters)
                    .and_then(|entries| entries.checked_mul(size_of::<f64>()))
                    .and_then(|gradients| values.checked_add(gradients))
            })
            .ok_or(PauliError::Overflow {
                context: "estimating propagation batch output storage",
            })?;
        let maximum_terms = observables
            .iter()
            .map(|observable| observable.terms().len())
            .max()
            .unwrap_or(0);
        let per_worker_bytes = estimate_batch_worker_bytes(&program, maximum_terms, None)?;
        let base_bytes = observable_bytes
            .checked_add(program.transition_bytes)
            .and_then(|bytes| bytes.checked_add(output_bytes))
            .ok_or(PauliError::Overflow {
                context: "estimating propagation batch shared storage",
            })?;
        let active_workers = allowed_batch_workers(
            observables.len(),
            &program,
            maximum_terms,
            per_worker_bytes,
            base_bytes,
        )?;
        let worker_bytes =
            active_workers
                .checked_mul(per_worker_bytes)
                .ok_or(PauliError::Overflow {
                    context: "estimating propagation batch worker storage",
                })?;
        let shared_bytes = observable_bytes
            .checked_add(program.transition_bytes)
            .and_then(|bytes| bytes.checked_add(output_bytes))
            .and_then(|bytes| bytes.checked_add(worker_bytes))
            .ok_or(PauliError::Overflow {
                context: "estimating propagation batch storage",
            })?;
        check_budget(shared_bytes, program.max_bytes, "propagation batch storage")?;

        let mut engines = Vec::with_capacity(observables.len());
        for observable in observables {
            engines.push(PropagationEngine::from_program(
                program.clone(),
                observable,
            )?);
        }
        Ok(Self { program, engines })
    }

    pub fn nqubits(&self) -> usize {
        self.program.nqubits
    }

    pub fn nparameters(&self) -> usize {
        self.program.nparameters
    }

    pub fn gate_count(&self) -> usize {
        self.program.operations.len()
    }

    pub fn observable_count(&self) -> usize {
        self.engines.len()
    }

    pub fn max_weight(&self) -> Option<usize> {
        self.program.max_weight
    }

    pub fn expectations(&self, parameters: &[f64]) -> Result<Vec<f64>, PauliError> {
        if self.engines.iter().any(|engine| !engine.hermitian) {
            return Err(PauliError::NonHermitianExpectation);
        }
        validate_parameters(parameters, self.program.nparameters)?;
        let results = self.map_observables(|engine| engine.expectation(parameters), None);
        collect_batch_values(results)
    }

    pub fn values_and_gradients(
        &self,
        parameters: &[f64],
        checkpoint_interval: Option<usize>,
    ) -> Result<PropagationBatchValueAndGradient, PauliError> {
        if self.engines.iter().any(|engine| !engine.hermitian) {
            return Err(PauliError::NonHermitianExpectation);
        }
        validate_parameters(parameters, self.program.nparameters)?;
        if checkpoint_interval == Some(0) {
            return Err(PauliError::InvalidCheckpointInterval);
        }
        let interval = checkpoint_interval.unwrap_or_else(|| {
            let gates = self.program.operations.len().saturating_add(1);
            (gates as f64).sqrt().ceil() as usize
        });
        let results = self.map_observables(
            |engine| engine.value_and_grad(parameters, checkpoint_interval),
            Some(interval),
        );
        let mut values = Vec::with_capacity(self.engines.len());
        let gradient_len = self
            .engines
            .len()
            .checked_mul(self.program.nparameters)
            .ok_or(PauliError::Overflow {
                context: "sizing propagation batch gradients",
            })?;
        let mut gradients = Vec::with_capacity(gradient_len);
        for result in results {
            let result = result?;
            values.push(result.value);
            gradients.extend(result.gradient);
        }
        Ok(PropagationBatchValueAndGradient { values, gradients })
    }

    fn map_observables<T, F>(
        &self,
        function: F,
        checkpoint_interval: Option<usize>,
    ) -> Vec<Result<T, PauliError>>
    where
        T: Send,
        F: Fn(&PropagationEngine) -> Result<T, PauliError> + Sync + Send,
    {
        let maximum_terms = self
            .engines
            .iter()
            .map(|engine| engine.observable.terms().len())
            .max()
            .unwrap_or(0);
        let per_worker_bytes =
            estimate_batch_worker_bytes(&self.program, maximum_terms, checkpoint_interval)
                .unwrap_or(usize::MAX);
        let base_bytes = self
            .engines
            .iter()
            .try_fold(0usize, |sum, engine| {
                let bytes = engine
                    .observable
                    .terms()
                    .len()
                    .checked_mul(dynamic_term_storage_bytes(self.program.nqubits)?)
                    .ok_or(PauliError::Overflow {
                        context: "estimating propagation batch observable storage",
                    })?;
                sum.checked_add(bytes).ok_or(PauliError::Overflow {
                    context: "estimating propagation batch observable storage",
                })
            })
            .ok()
            .and_then(|bytes| bytes.checked_add(self.program.transition_bytes))
            .and_then(|bytes| {
                self.engines
                    .len()
                    .checked_mul(size_of::<f64>())
                    .and_then(|values| {
                        self.engines
                            .len()
                            .checked_mul(self.program.nparameters)
                            .and_then(|entries| entries.checked_mul(size_of::<f64>()))
                            .and_then(|gradients| values.checked_add(gradients))
                    })
                    .and_then(|output| bytes.checked_add(output))
            })
            .unwrap_or(usize::MAX);
        let active_workers = allowed_batch_workers(
            self.engines.len(),
            &self.program,
            maximum_terms,
            per_worker_bytes,
            base_bytes,
        )
        .unwrap_or(1);
        if active_workers > 1 && rayon::current_num_threads() > 1 {
            let chunk_size = self.engines.len().div_ceil(active_workers);
            self.engines
                .par_chunks(chunk_size.max(1))
                .map(|chunk| chunk.iter().map(&function).collect::<Vec<_>>())
                .collect::<Vec<_>>()
                .into_iter()
                .flatten()
                .collect()
        } else {
            self.engines.iter().map(function).collect()
        }
    }
}

fn batch_work_units(
    program: &CompiledPropagationProgram,
    observable_count: usize,
    maximum_terms: usize,
) -> usize {
    let operation_work = program
        .operations
        .iter()
        .map(|operation| operation_branch_factor(operation).max(1))
        .fold(0usize, usize::saturating_add)
        .max(1);
    observable_count
        .saturating_mul(maximum_terms.max(1))
        .saturating_mul(operation_work)
}

fn should_parallelize_batch(
    program: &CompiledPropagationProgram,
    observable_count: usize,
    maximum_terms: usize,
) -> bool {
    if observable_count <= 1 {
        return false;
    }
    let row_work = maximum_terms.max(1).saturating_mul(
        program
            .operations
            .iter()
            .map(|operation| operation_branch_factor(operation).max(1))
            .fold(0usize, usize::saturating_add),
    );
    row_work >= 16
        && batch_work_units(program, observable_count, maximum_terms)
            >= BATCH_PARALLEL_WORK_THRESHOLD
}

fn allowed_batch_workers(
    observable_count: usize,
    program: &CompiledPropagationProgram,
    maximum_terms: usize,
    per_worker_bytes: usize,
    base_bytes: usize,
) -> Result<usize, PauliError> {
    if observable_count == 0 {
        return Ok(0);
    }
    let remaining = remaining_budget(base_bytes, program.max_bytes)?;
    if per_worker_bytes > remaining {
        let requested = (base_bytes as u128)
            .checked_add(per_worker_bytes as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating propagation batch worker storage",
            })?;
        return Err(PauliError::MemoryLimit {
            requested,
            limit: program.max_bytes.unwrap_or(u128::MAX),
        });
    }
    let thread_limit = observable_count.min(rayon::current_num_threads());
    if thread_limit <= 1 || !should_parallelize_batch(program, observable_count, maximum_terms) {
        return Ok(1);
    }
    let budget_limit = if per_worker_bytes == 0 {
        thread_limit
    } else {
        remaining.checked_div(per_worker_bytes).unwrap_or(0)
    };
    Ok(thread_limit.min(budget_limit))
}

fn remaining_budget(base_bytes: usize, max_bytes: Option<u128>) -> Result<usize, PauliError> {
    match max_bytes {
        None => Ok(usize::MAX),
        Some(limit) => {
            if base_bytes as u128 > limit {
                return Err(PauliError::MemoryLimit {
                    requested: base_bytes as u128,
                    limit,
                });
            }
            usize::try_from(limit - base_bytes as u128).map_err(|_| PauliError::Overflow {
                context: "converting propagation batch memory budget",
            })
        }
    }
}

fn estimate_batch_worker_bytes(
    program: &CompiledPropagationProgram,
    maximum_terms: usize,
    checkpoint_interval: Option<usize>,
) -> Result<usize, PauliError> {
    let term_bytes = dynamic_term_storage_bytes(program.nqubits)?;
    let initial_terms = maximum_terms.max(1);
    let growth_exponent = program
        .operations
        .iter()
        .map(|operation| branch_exponent(operation_branch_factor(operation)))
        .fold(0usize, usize::saturating_add)
        .min(program.nqubits);
    let propagated_terms = if let Some(max_weight) = program.max_weight {
        let projected_bound = projected_pauli_word_bound(program.nqubits, max_weight)?;
        let projected_growth = checked_pow_two(growth_exponent)
            .ok()
            .and_then(|growth| initial_terms.checked_mul(growth))
            .map_or(projected_bound, |terms| terms.min(projected_bound));
        // Projection is applied after a gate's contributions are aggregated,
        // so the initial observable can temporarily contain more terms than
        // the projected universe itself.
        initial_terms.max(projected_growth)
    } else {
        let growth = checked_pow_two(growth_exponent)?;
        initial_terms
            .checked_mul(growth)
            .ok_or(PauliError::Overflow {
                context: "estimating propagation batch term growth",
            })?
    };
    let maximum_branch = program
        .operations
        .iter()
        .map(|operation| operation_branch_factor(operation).max(1))
        .max()
        .unwrap_or(1);
    let candidate_terms =
        propagated_terms
            .checked_mul(maximum_branch)
            .ok_or(PauliError::Overflow {
                context: "estimating propagation batch candidate storage",
            })?;
    let forward_terms =
        propagated_terms
            .checked_add(candidate_terms)
            .ok_or(PauliError::Overflow {
                context: "estimating propagation batch forward storage",
            })?;
    let mut term_slots = forward_terms;
    if let Some(interval) = checkpoint_interval {
        let interval = interval.max(1);
        let checkpoint_count = program
            .operations
            .len()
            .checked_add(interval - 1)
            .map(|count| count / interval)
            .and_then(|count| count.checked_add(1))
            .ok_or(PauliError::Overflow {
                context: "estimating propagation batch checkpoint count",
            })?;
        let replay_states = interval.checked_add(1).ok_or(PauliError::Overflow {
            context: "estimating propagation batch replay states",
        })?;
        term_slots = term_slots
            .checked_add(
                checkpoint_count
                    .checked_add(replay_states)
                    .and_then(|count| count.checked_add(2))
                    .and_then(|count| count.checked_mul(propagated_terms))
                    .ok_or(PauliError::Overflow {
                        context: "estimating propagation batch reverse storage",
                    })?,
            )
            .ok_or(PauliError::Overflow {
                context: "estimating propagation batch reverse storage",
            })?;
    }
    let term_storage = term_slots
        .checked_mul(term_bytes)
        .ok_or(PauliError::Overflow {
            context: "estimating propagation batch term storage",
        })?;
    let gradient_storage = program
        .nparameters
        .checked_mul(size_of::<f64>())
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or(PauliError::Overflow {
            context: "estimating propagation batch gradient storage",
        })?;
    term_storage
        .checked_add(gradient_storage)
        .ok_or(PauliError::Overflow {
            context: "estimating propagation batch worker storage",
        })
}

/// Return the number of Pauli words with weight at most `max_weight`.
///
/// The bound is `sum_w C(nqubits, w) * 3^w`. Intermediate binomial products
/// use `u128`, which is wide enough for the product of two platform-sized
/// values, and every completed term is required to fit the platform index
/// space before it can influence an allocation estimate.
fn projected_pauli_word_bound(nqubits: usize, max_weight: usize) -> Result<usize, PauliError> {
    let maximum_weight = max_weight.min(nqubits);
    let mut combinations = 1_u128;
    let mut pauli_choices = 1_u128;
    let mut total = 1_u128;
    let platform_limit = usize::MAX as u128;
    for weight in 1..=maximum_weight {
        combinations = combinations
            .checked_mul((nqubits - weight + 1) as u128)
            .and_then(|value| value.checked_div(weight as u128))
            .ok_or(PauliError::Overflow {
                context: "estimating weight-projected Pauli universe",
            })?;
        pauli_choices = pauli_choices.checked_mul(3).ok_or(PauliError::Overflow {
            context: "estimating weight-projected Pauli universe",
        })?;
        let words = combinations
            .checked_mul(pauli_choices)
            .ok_or(PauliError::Overflow {
                context: "estimating weight-projected Pauli universe",
            })?;
        total = total.checked_add(words).ok_or(PauliError::Overflow {
            context: "estimating weight-projected Pauli universe",
        })?;
        if total > platform_limit {
            return Err(PauliError::Overflow {
                context: "estimating weight-projected Pauli universe",
            });
        }
    }
    usize::try_from(total).map_err(|_| PauliError::Overflow {
        context: "estimating weight-projected Pauli universe",
    })
}

fn branch_exponent(branch_factor: usize) -> usize {
    if branch_factor <= 1 {
        0
    } else {
        usize::BITS as usize - (branch_factor - 1).leading_zeros() as usize
    }
}

fn checked_pow_two(exponent: usize) -> Result<usize, PauliError> {
    if exponent >= usize::BITS as usize {
        return Err(PauliError::Overflow {
            context: "estimating propagation batch term growth",
        });
    }
    1usize
        .checked_shl(exponent as u32)
        .ok_or(PauliError::Overflow {
            context: "estimating propagation batch term growth",
        })
}

fn collect_batch_values(results: Vec<Result<f64, PauliError>>) -> Result<Vec<f64>, PauliError> {
    results.into_iter().collect()
}

struct DynamicPropagationResult {
    terms: Vec<DynamicTerm>,
    initial_terms: usize,
    peak_terms: usize,
    estimated_peak_bytes: usize,
}

#[derive(Clone, Debug)]
struct DynamicTerm {
    key: PackedKey,
    coefficient: Complex64,
}

fn initial_dynamic_terms(
    nqubits: usize,
    observable: &PauliOperator,
    cutoff: Option<usize>,
) -> Vec<DynamicTerm> {
    observable
        .terms()
        .iter()
        .filter_map(|term| {
            let key = PackedKey::from_word(&term.word);
            (cutoff.is_none_or(|limit| key.weight(nqubits) <= limit)).then_some(DynamicTerm {
                key,
                coefficient: term.coefficient,
            })
        })
        .collect()
}

fn dynamic_terms_storage_bytes(terms: &[DynamicTerm], nqubits: usize) -> Result<usize, PauliError> {
    terms.iter().try_fold(0usize, |sum, _| {
        sum.checked_add(dynamic_term_storage_bytes(nqubits)?)
            .ok_or(PauliError::Overflow {
                context: "estimating reverse sparse state storage",
            })
    })
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) enum PackedKey {
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
    pub(crate) fn from_word(word: &PauliWord) -> Self {
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

    pub(crate) fn code_at(&self, qubit: usize) -> u8 {
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

    pub(crate) fn set_code(&mut self, qubit: usize, code: u8) {
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
            Ordering::Equal => {
                let word_count = packed_word_count(left_n);
                let (left_x, left_z) = packed_masks(self);
                let (right_x, right_z) = packed_masks(other);
                left_x[..word_count]
                    .cmp(&right_x[..word_count])
                    .then_with(|| left_z[..word_count].cmp(&right_z[..word_count]))
            }
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

fn packed_masks(key: &PackedKey) -> (&[u64], &[u64]) {
    match key {
        PackedKey::Inline { x, z, .. } => (x, z),
        PackedKey::Wide { x, z, .. } => (x, z),
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

pub(crate) fn validate_state(nqubits: usize, state: &ProductState) -> Result<(), PauliError> {
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

fn dynamic_term_storage_bytes(nqubits: usize) -> Result<usize, PauliError> {
    let wide_payload = if nqubits > 128 {
        packed_word_count(nqubits)
            .checked_mul(2)
            .and_then(|words| words.checked_mul(size_of::<u64>()))
            .ok_or(PauliError::Overflow {
                context: "estimating packed wide-key storage",
            })?
    } else {
        0
    };
    size_of::<DynamicTerm>()
        .checked_add(wide_payload)
        .ok_or(PauliError::Overflow {
            context: "estimating packed dynamic-term storage",
        })
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
                let multiplier = apply_clifford1_in_place(&mut term.key, *gate, *wire);
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
                let multiplier = apply_clifford2_in_place(&mut term.key, *gate, *wire0, *wire1);
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
            let mut contributions = Vec::with_capacity(terms.len().saturating_mul(2));
            let generator_code = rotation_code(*axis);
            for term in terms {
                let (product, phase) =
                    multiply_by_generator(&term.key, generator_code, *wire0, *wire1);
                match phase {
                    PauliPhase::PlusI | PauliPhase::MinusI => {
                        contributions.push((term.key, term.coefficient * cosine));
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

#[allow(clippy::too_many_arguments)]
fn reverse_frame(
    nqubits: usize,
    operation: &GateOperation,
    input: &[DynamicTerm],
    output: &[DynamicTerm],
    parameters: &[f64],
    cutoff: Option<usize>,
    output_lambda: &[f64],
    output_indices: &mut FxHashMap<PackedKey, usize>,
    gradient: &mut [f64],
) -> Result<Vec<f64>, PauliError> {
    output_indices.clear();
    output_indices.reserve(output.len());
    for (index, term) in output.iter().enumerate() {
        output_indices.insert(term.key.clone(), index);
    }
    let mut input_lambda = vec![0.0; input.len()];
    for (input_index, term) in input.iter().enumerate() {
        visit_retained_edges(
            nqubits,
            operation,
            term,
            parameters,
            cutoff,
            output_indices,
            |output_index, multiplier, derivative, slot| {
                let output_adjoint = output_lambda[output_index];
                input_lambda[input_index] += multiplier * output_adjoint;
                if let Some(parameter_slot) = slot {
                    gradient[parameter_slot] += term.coefficient.re * derivative * output_adjoint;
                }
            },
        )?;
    }
    Ok(input_lambda)
}

/// Visit the nonzero contribution edges that survived the executed forward
/// support decision. The output map is intentionally supplied by the reverse
/// caller so aggregation and projection are applied exactly once.
fn visit_retained_edges<F>(
    nqubits: usize,
    operation: &GateOperation,
    term: &DynamicTerm,
    parameters: &[f64],
    cutoff: Option<usize>,
    output_indices: &FxHashMap<PackedKey, usize>,
    mut visit: F,
) -> Result<(), PauliError>
where
    F: FnMut(usize, f64, f64, Option<usize>),
{
    let mut emit = |key: PackedKey, multiplier: f64, derivative: f64, slot: Option<usize>| {
        let scaled = term.coefficient * multiplier;
        if is_exact_zero(scaled) || cutoff.is_some_and(|limit| key.weight(nqubits) > limit) {
            return Ok::<(), PauliError>(());
        }
        if let Some(&output_index) = output_indices.get(&key) {
            visit(output_index, multiplier, derivative, slot);
        }
        Ok(())
    };

    match &operation.kind {
        GateKind::Clifford1 { gate, wire } => {
            let (key, multiplier) = map_clifford1(&term.key, *gate, *wire);
            emit(key, multiplier, 0.0, None)?;
        }
        GateKind::Clifford2 { gate, wire0, wire1 } => {
            let (key, multiplier) = map_clifford2(&term.key, *gate, *wire0, *wire1);
            emit(key, multiplier, 0.0, None)?;
        }
        GateKind::Rotation {
            axis,
            wire0,
            wire1,
            parameter,
        } => {
            let (cosine, sine) = resolve_parameter(*parameter, parameters)?;
            let slot = match parameter {
                ParameterRef::Slot(index) => Some(*index),
                ParameterRef::Static { .. } => None,
            };
            let (product, phase) =
                multiply_by_generator(&term.key, rotation_code(*axis), *wire0, *wire1);
            match phase {
                PauliPhase::PlusI | PauliPhase::MinusI => {
                    if cosine != 0.0 {
                        emit(term.key.clone(), cosine, slot.map_or(0.0, |_| -sine), slot)?;
                    }
                    if sine != 0.0 {
                        let sign = phase_sign_i(phase);
                        emit(
                            product,
                            sine * sign,
                            slot.map_or(0.0, |_| cosine * sign),
                            slot,
                        )?;
                    }
                }
                PauliPhase::PlusOne | PauliPhase::MinusOne => {
                    emit(term.key.clone(), 1.0, 0.0, None)?;
                }
            }
        }
        GateKind::CustomPtm {
            wire0,
            wire1,
            transitions,
        } => {
            let input = local_index(&term.key, *wire0, *wire1);
            for &(output, coefficient) in &transitions[input] {
                let mut key = term.key.clone();
                let (first, second) = local_codes(output, wire1.is_some());
                key.set_code(*wire0, first);
                if let Some(second_wire) = wire1 {
                    key.set_code(*second_wire, second);
                }
                emit(key, coefficient, 0.0, None)?;
            }
        }
    }
    Ok(())
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
        if let Some(current) = values.get_mut(&key) {
            *current += coefficient;
        } else {
            values.insert(key, coefficient);
        }
    }
    let mut ordered = values
        .into_iter()
        .map(|(key, coefficient)| DynamicTerm { key, coefficient })
        .collect::<Vec<_>>();
    ordered.sort_unstable_by(|left, right| left.key.cmp(&right.key));
    ordered.retain(|term| {
        !is_exact_zero(term.coefficient)
            && cutoff.is_none_or(|limit| term.key.weight(nqubits) <= limit)
    });
    Ok(ordered)
}

pub(crate) fn map_clifford1(key: &PackedKey, gate: Clifford1, wire: usize) -> (PackedKey, f64) {
    let mut result = key.clone();
    let sign = apply_clifford1_in_place(&mut result, gate, wire);
    (result, sign)
}

pub(crate) fn apply_clifford1_in_place(key: &mut PackedKey, gate: Clifford1, wire: usize) -> f64 {
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
    key.set_code(wire, mapped_code);
    sign
}

pub(crate) fn map_clifford2(
    key: &PackedKey,
    gate: Clifford2,
    wire0: usize,
    wire1: usize,
) -> (PackedKey, f64) {
    let first = key.code_at(wire0);
    let second = key.code_at(wire1);
    let (mapped_first, mapped_second, multiplier) = clifford2_local_map(gate, first, second);
    let mut result = key.clone();
    result.set_code(wire0, mapped_first);
    result.set_code(wire1, mapped_second);
    (result, multiplier)
}

pub(crate) fn apply_clifford2_in_place(
    key: &mut PackedKey,
    gate: Clifford2,
    wire0: usize,
    wire1: usize,
) -> f64 {
    let first = key.code_at(wire0);
    let second = key.code_at(wire1);
    let (mapped_first, mapped_second, multiplier) = clifford2_local_map(gate, first, second);
    key.set_code(wire0, mapped_first);
    key.set_code(wire1, mapped_second);
    multiplier
}

fn clifford2_local_map(gate: Clifford2, first: u8, second: u8) -> (u8, u8, f64) {
    // Each entry is the conjugation of one local two-qubit Pauli word.  The
    // index is `4 * first + second`, matching the public PTM wire order.
    const CNOT: [(u8, u8, f64); 16] = [
        (0, 0, 1.0),
        (0, 1, 1.0),
        (3, 2, 1.0),
        (3, 3, 1.0),
        (1, 1, 1.0),
        (1, 0, 1.0),
        (2, 3, 1.0),
        (2, 2, -1.0),
        (2, 1, 1.0),
        (2, 0, 1.0),
        (1, 3, -1.0),
        (1, 2, 1.0),
        (3, 0, 1.0),
        (3, 1, 1.0),
        (0, 2, 1.0),
        (0, 3, 1.0),
    ];
    const CZ: [(u8, u8, f64); 16] = [
        (0, 0, 1.0),
        (3, 1, 1.0),
        (3, 2, 1.0),
        (0, 3, 1.0),
        (1, 3, 1.0),
        (2, 2, 1.0),
        (2, 1, -1.0),
        (1, 0, 1.0),
        (2, 3, 1.0),
        (1, 2, -1.0),
        (1, 1, 1.0),
        (2, 0, 1.0),
        (3, 0, 1.0),
        (0, 1, 1.0),
        (0, 2, 1.0),
        (3, 3, 1.0),
    ];
    const SWAP: [(u8, u8, f64); 16] = [
        (0, 0, 1.0),
        (1, 0, 1.0),
        (2, 0, 1.0),
        (3, 0, 1.0),
        (0, 1, 1.0),
        (1, 1, 1.0),
        (2, 1, 1.0),
        (3, 1, 1.0),
        (0, 2, 1.0),
        (1, 2, 1.0),
        (2, 2, 1.0),
        (3, 2, 1.0),
        (0, 3, 1.0),
        (1, 3, 1.0),
        (2, 3, 1.0),
        (3, 3, 1.0),
    ];
    match gate {
        Clifford2::Cnot => CNOT[4 * first as usize + second as usize],
        Clifford2::Cz => CZ[4 * first as usize + second as usize],
        Clifford2::Swap => SWAP[4 * first as usize + second as usize],
    }
}

pub(crate) fn rotation_code(axis: RotationAxis) -> u8 {
    match axis {
        RotationAxis::X => 1,
        RotationAxis::Y => 2,
        RotationAxis::Z => 3,
    }
}

pub(crate) fn multiply_by_generator(
    key: &PackedKey,
    generator_code: u8,
    wire0: usize,
    wire1: Option<usize>,
) -> (PackedKey, PauliPhase) {
    let mut result = key.clone();
    let mut phase = PauliPhase::PlusOne;
    for wire in [Some(wire0), wire1].into_iter().flatten() {
        let (code, local_phase) = local_product(generator_code, key.code_at(wire));
        result.set_code(wire, code);
        phase = phase.multiply(local_phase);
    }
    (result, phase)
}

pub(crate) fn generator_transition(
    key: &PackedKey,
    generator_code: u8,
    wire0: usize,
    wire1: Option<usize>,
) -> (PauliPhase, u8, u8) {
    let (first, first_phase) = local_product(generator_code, key.code_at(wire0));
    if let Some(second_wire) = wire1 {
        let (second, second_phase) = local_product(generator_code, key.code_at(second_wire));
        (first_phase.multiply(second_phase), first, second)
    } else {
        (first_phase, first, 0)
    }
}

fn local_product(left: u8, right: u8) -> (u8, PauliPhase) {
    match (left, right) {
        (0, code) | (code, 0) => (code, PauliPhase::PlusOne),
        (1, 1) | (2, 2) | (3, 3) => (0, PauliPhase::PlusOne),
        (1, 2) => (3, PauliPhase::PlusI),
        (2, 1) => (3, PauliPhase::MinusI),
        (1, 3) => (2, PauliPhase::MinusI),
        (3, 1) => (2, PauliPhase::PlusI),
        (2, 3) => (1, PauliPhase::PlusI),
        (3, 2) => (1, PauliPhase::MinusI),
        _ => unreachable!("packed local Pauli codes are always in 0..4"),
    }
}

pub(crate) fn resolve_parameter(
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

pub(crate) fn phase_sign_i(phase: PauliPhase) -> f64 {
    match phase {
        PauliPhase::PlusI => -1.0,
        PauliPhase::MinusI => 1.0,
        _ => unreachable!("rotation sine branch requires an anticommuting phase"),
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
                product * expectation_component(state, term.word.code_at(qubit), qubit)
            });
            term.coefficient.re * local
        })
        .sum()
}

fn expectation_from_dynamic_terms(
    terms: &[DynamicTerm],
    state: &ProductState,
    nqubits: usize,
) -> f64 {
    terms
        .iter()
        .map(|term| term.coefficient.re * expectation_of_key(&term.key, state, nqubits))
        .sum()
}

pub(crate) fn expectation_of_key(key: &PackedKey, state: &ProductState, nqubits: usize) -> f64 {
    match state {
        ProductState::Zero => match key {
            PackedKey::Inline { x, .. } => {
                if (0..packed_word_count(nqubits)).all(|index| x[index] == 0) {
                    1.0
                } else {
                    0.0
                }
            }
            PackedKey::Wide { x, .. } => {
                if x.iter().all(|word| *word == 0) {
                    1.0
                } else {
                    0.0
                }
            }
        },
        _ => (0..nqubits).fold(1.0, |product, qubit| {
            product * expectation_component(state, key.code_at(qubit), qubit)
        }),
    }
}

fn expectation_component(state: &ProductState, code: u8, qubit: usize) -> f64 {
    match state {
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
    }
}

pub(crate) fn check_budget(
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_program(
        operations: Vec<GateOperation>,
        max_bytes: Option<u128>,
    ) -> Arc<CompiledPropagationProgram> {
        compile_program(12, operations, ProductState::Zero, None, max_bytes).unwrap()
    }

    #[test]
    fn batch_scheduler_keeps_light_clifford_rows_serial() {
        let operations = (0..16)
            .map(|wire| GateOperation::clifford1(12, Clifford1::Z, wire % 12).unwrap())
            .collect();
        let program = test_program(operations, None);
        assert!(!should_parallelize_batch(&program, 2, 1));
        assert!(!should_parallelize_batch(&program, 4, 1));
    }

    #[test]
    fn batch_scheduler_parallelizes_rotation_heavy_rows() {
        let operations = (0..48)
            .map(|wire| {
                GateOperation::rotation(
                    12,
                    RotationAxis::Y,
                    wire % 12,
                    None,
                    ParameterRef::Static { cos: 0.9, sin: 0.1 },
                )
                .unwrap()
            })
            .collect();
        let program = test_program(operations, None);
        assert!(should_parallelize_batch(&program, 4, 1));
    }

    #[test]
    fn batch_worker_budget_bounds_active_rows() {
        let operations = (0..48)
            .map(|wire| {
                GateOperation::rotation(
                    12,
                    RotationAxis::Y,
                    wire % 12,
                    None,
                    ParameterRef::Static { cos: 0.9, sin: 0.1 },
                )
                .unwrap()
            })
            .collect();
        let program = test_program(operations, Some(1_000_000));
        let worker_bytes = estimate_batch_worker_bytes(&program, 1, None).unwrap();
        assert!(worker_bytes > 500_000);
        assert_eq!(
            allowed_batch_workers(16, &program, 1, worker_bytes, 0).unwrap(),
            1
        );
    }

    #[test]
    fn local_rotation_products_match_pauli_phase_table() {
        assert_eq!(local_product(1, 2), (3, PauliPhase::PlusI));
        assert_eq!(local_product(2, 1), (3, PauliPhase::MinusI));
        assert_eq!(local_product(2, 3), (1, PauliPhase::PlusI));
        assert_eq!(local_product(3, 2), (1, PauliPhase::MinusI));
    }

    #[test]
    fn two_qubit_local_table_carries_conjugation_signs() {
        assert_eq!(clifford2_local_map(Clifford2::Cnot, 2, 2), (1, 3, -1.0));
        assert_eq!(clifford2_local_map(Clifford2::Cz, 1, 2), (2, 1, -1.0));
        assert_eq!(clifford2_local_map(Clifford2::Swap, 1, 3), (3, 1, 1.0));
    }

    #[test]
    fn in_place_path_updates_match_allocating_maps() {
        for gate in [
            Clifford1::X,
            Clifford1::Y,
            Clifford1::Z,
            Clifford1::H,
            Clifford1::S,
            Clifford1::Sdg,
        ] {
            for code in 0..4 {
                let word = PauliWord::from_codes(1, &[code]).unwrap();
                let key = PackedKey::from_word(&word);
                let (expected, expected_sign) = map_clifford1(&key, gate, 0);
                let mut actual = key.clone();
                let actual_sign = apply_clifford1_in_place(&mut actual, gate, 0);
                assert_eq!(actual, expected);
                assert_eq!(actual_sign, expected_sign);
            }
        }
        for gate in [Clifford2::Cnot, Clifford2::Cz, Clifford2::Swap] {
            for first in 0..4 {
                for second in 0..4 {
                    let word = PauliWord::from_codes(2, &[first, second]).unwrap();
                    let key = PackedKey::from_word(&word);
                    let (expected, expected_sign) = map_clifford2(&key, gate, 0, 1);
                    let mut actual = key.clone();
                    let actual_sign = apply_clifford2_in_place(&mut actual, gate, 0, 1);
                    assert_eq!(actual, expected);
                    assert_eq!(actual_sign, expected_sign);
                }
            }
        }
    }

    #[test]
    fn generator_transition_matches_allocating_product() {
        for generator in 1..=3 {
            for code in 0..4 {
                let word = PauliWord::from_codes(1, &[code]).unwrap();
                let key = PackedKey::from_word(&word);
                let (expected, expected_phase) = multiply_by_generator(&key, generator, 0, None);
                let (phase, mapped_first, mapped_second) =
                    generator_transition(&key, generator, 0, None);
                let mut actual = key.clone();
                actual.set_code(0, mapped_first);
                assert_eq!(actual, expected);
                assert_eq!(phase, expected_phase);
                assert_eq!(mapped_second, 0);
            }
        }
    }

    #[test]
    fn dynamic_aggregation_keeps_ordinary_float_overflow() {
        let word = PauliWord::from_codes(1, &[1]).unwrap();
        let key = PackedKey::from_word(&word);
        let result = aggregate(
            vec![
                (key.clone(), Complex64::new(f64::MAX, 0.0)),
                (key, Complex64::new(f64::MAX, 0.0)),
            ],
            1,
            None,
        );
        assert!(result.unwrap()[0].coefficient.re.is_infinite());
    }
}
