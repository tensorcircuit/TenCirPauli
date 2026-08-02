//! Stochastic Pauli-path value-and-gradient estimation.
//!
//! This module is deliberately separate from deterministic sparse
//! propagation. Paths carry only one packed Pauli word and a reusable list of
//! active local factors; no dynamic operator aggregation is performed.

use std::collections::HashSet;

use rayon::prelude::*;

use crate::error::PauliError;
use crate::gate::{Clifford1, Clifford2, GateKind, GateOperation, ParameterRef};
use crate::operator::PauliOperator;
use crate::propagation::{
    apply_clifford1_in_place, apply_clifford2_in_place, expectation_of_key, generator_transition,
    phase_sign_i, resolve_parameter, rotation_code, validate_state, PackedKey, ProductState,
};

const RANDOM_SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;
const STABLE_PRODUCT_RATIO_THRESHOLD: f64 = 1.0e-12;

/// One SPPS value-and-gradient result.
#[derive(Clone, Debug)]
pub struct SPPSEstimate {
    pub value: f64,
    pub gradient: Vec<f64>,
    pub value_standard_error: f64,
    pub replicates: usize,
    pub samples_per_replicate: Vec<usize>,
    pub total_paths: usize,
    pub seed: u64,
    pub gradient_error_proxy: Option<f64>,
    pub term_gradient_error_proxies: Option<Vec<f64>>,
    pub converged: Option<bool>,
}

/// Immutable stochastic Pauli-path engine.
#[derive(Clone, Debug)]
pub struct SPPSEngine {
    nqubits: usize,
    operations: Vec<SppsOperation>,
    observable: PauliOperator,
    initial_state: ProductState,
    smoothing: f64,
    nparameters: usize,
}

#[derive(Clone, Debug)]
enum SppsOperation {
    Clifford1 {
        gate: Clifford1,
        wire: usize,
    },
    Clifford2 {
        gate: Clifford2,
        wire0: usize,
        wire1: usize,
    },
    Rotation {
        generator_code: u8,
        wire0: usize,
        wire1: Option<usize>,
        parameter: ParameterRef,
        source_index: usize,
    },
}

enum ResolvedOperation {
    Clifford1 {
        gate: Clifford1,
        wire: usize,
    },
    Clifford2 {
        gate: Clifford2,
        wire0: usize,
        wire1: usize,
    },
    Rotation {
        generator_code: u8,
        wire0: usize,
        wire1: Option<usize>,
        cosine: f64,
        sine: f64,
        slot: Option<usize>,
        source_index: usize,
    },
}

#[derive(Clone)]
struct TermStats {
    sum: f64,
    sum_squared: f64,
    gradient_sum: Vec<f64>,
    count: usize,
}

impl TermStats {
    fn new(nparameters: usize) -> Self {
        Self {
            sum: 0.0,
            sum_squared: 0.0,
            gradient_sum: vec![0.0; nparameters],
            count: 0,
        }
    }
}

struct PathScratch {
    ratios: Vec<f64>,
    derivatives: Vec<f64>,
    slots: Vec<Option<usize>>,
    prefix: Vec<f64>,
    suffix: Vec<f64>,
}

impl PathScratch {
    fn new(gate_count: usize) -> Self {
        Self {
            ratios: Vec::with_capacity(gate_count),
            derivatives: Vec::with_capacity(gate_count),
            slots: Vec::with_capacity(gate_count),
            prefix: Vec::with_capacity(gate_count + 1),
            suffix: Vec::with_capacity(gate_count + 1),
        }
    }

    fn clear(&mut self) {
        self.ratios.clear();
        self.derivatives.clear();
        self.slots.clear();
        self.prefix.clear();
        self.suffix.clear();
    }
}

impl SPPSEngine {
    /// Compile a tape and a real Hermitian observable for SPPS.
    pub fn new(
        nqubits: usize,
        operations: Vec<GateOperation>,
        observable: PauliOperator,
        initial_state: ProductState,
        smoothing: f64,
        max_bytes: Option<u128>,
    ) -> Result<Self, PauliError> {
        if observable.nqubits() != nqubits {
            return Err(PauliError::IncompatibleQubitCounts {
                left: observable.nqubits(),
                right: nqubits,
            });
        }
        if !smoothing.is_finite() || smoothing <= 0.0 {
            return Err(PauliError::InvalidSppsSmoothing);
        }
        validate_state(nqubits, &initial_state)?;
        if !observable.is_hermitian(0.0) {
            return Err(PauliError::NonHermitianExpectation);
        }
        if operations
            .iter()
            .any(|operation| matches!(operation.kind, GateKind::CustomPtm { .. }))
        {
            return Err(PauliError::UnsupportedSppsGate);
        }
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
            .checked_mul(nqubits.saturating_add(32))
            .ok_or(PauliError::Overflow {
                context: "estimating SPPS observable storage",
            })?;
        let gradient_bytes = nparameters
            .checked_mul(8)
            .and_then(|bytes| bytes.checked_mul(2))
            .ok_or(PauliError::Overflow {
                context: "estimating SPPS gradient storage",
            })?;
        let estimate = observable_bytes
            .checked_add(gradient_bytes)
            .and_then(|bytes| bytes.checked_add(operations.len().saturating_mul(64)))
            .ok_or(PauliError::Overflow {
                context: "estimating SPPS engine storage",
            })?;
        if let Some(limit) = max_bytes {
            if estimate as u128 > limit {
                return Err(PauliError::MemoryLimit {
                    requested: estimate as u128,
                    limit,
                });
            }
        }
        let path_operations = operations
            .iter()
            .enumerate()
            .rev()
            .map(|(source_index, operation)| match &operation.kind {
                GateKind::Clifford1 { gate, wire } => SppsOperation::Clifford1 {
                    gate: *gate,
                    wire: *wire,
                },
                GateKind::Clifford2 { gate, wire0, wire1 } => SppsOperation::Clifford2 {
                    gate: *gate,
                    wire0: *wire0,
                    wire1: *wire1,
                },
                GateKind::Rotation {
                    axis,
                    wire0,
                    wire1,
                    parameter,
                } => SppsOperation::Rotation {
                    generator_code: rotation_code(*axis),
                    wire0: *wire0,
                    wire1: *wire1,
                    parameter: *parameter,
                    source_index,
                },
                GateKind::CustomPtm { .. } => unreachable!("custom PTM was rejected above"),
            })
            .collect();
        Ok(Self {
            nqubits,
            operations: path_operations,
            observable,
            initial_state,
            smoothing,
            nparameters,
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

    pub fn observable_terms(&self) -> usize {
        self.observable.terms().len()
    }

    pub fn smoothing(&self) -> f64 {
        self.smoothing
    }

    /// Estimate with one independent replicate per observable term.
    pub fn value_and_grad(
        &self,
        parameters: &[f64],
        samples_per_term: usize,
        seed: u64,
    ) -> Result<SPPSEstimate, PauliError> {
        if samples_per_term < 2 {
            return Err(PauliError::InvalidSppsBudget {
                context: "samples_per_term; expected at least 2",
            });
        }
        self.validate_parameters(parameters)?;
        let operations = self.resolve_operations(parameters)?;
        let term_count = self.observable.terms().len();
        if term_count == 0 {
            return Ok(empty_estimate(self.nparameters, seed, 1, false));
        }
        let mut stats = (0..term_count)
            .map(|_| TermStats::new(self.nparameters))
            .collect::<Vec<_>>();
        stats
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(term_index, stats)| {
                let mut scratch = PathScratch::new(self.operations.len());
                let term = &self.observable.terms()[term_index];
                run_samples(
                    &operations,
                    &self.initial_state,
                    self.nqubits,
                    self.smoothing,
                    term,
                    term_index,
                    0,
                    0,
                    samples_per_term,
                    seed,
                    stats,
                    &mut scratch,
                )
            })?;
        let (value, gradient, standard_error) = combine_fixed(&stats, self.nparameters)?;
        Ok(SPPSEstimate {
            value,
            gradient,
            value_standard_error: standard_error,
            replicates: 1,
            samples_per_replicate: vec![samples_per_term; term_count],
            total_paths: samples_per_term
                .checked_mul(term_count)
                .ok_or(PauliError::Overflow {
                    context: "counting SPPS paths",
                })?,
            seed,
            gradient_error_proxy: None,
            term_gradient_error_proxies: None,
            converged: None,
        })
    }

    /// Estimate with two independently seeded, cumulatively doubled
    /// macro-replicates and the Spec's empirical gradient proxy.
    pub fn value_and_grad_adaptive(
        &self,
        parameters: &[f64],
        initial_samples_per_term: usize,
        max_samples_per_term: usize,
        gradient_tolerance: f64,
        seed: u64,
    ) -> Result<SPPSEstimate, PauliError> {
        if initial_samples_per_term < 2 {
            return Err(PauliError::InvalidSppsBudget {
                context: "initial_samples_per_term; expected at least 2",
            });
        }
        if max_samples_per_term < initial_samples_per_term {
            return Err(PauliError::InvalidSppsBudget {
                context: "max_samples_per_term; expected >= initial_samples_per_term",
            });
        }
        if !gradient_tolerance.is_finite() || gradient_tolerance <= 0.0 {
            return Err(PauliError::InvalidSppsBudget {
                context: "gradient_tolerance; expected a finite positive float",
            });
        }
        self.validate_parameters(parameters)?;
        let operations = self.resolve_operations(parameters)?;
        let term_count = self.observable.terms().len();
        if term_count == 0 {
            return Ok(empty_estimate(self.nparameters, seed, 2, true));
        }
        let mut left = (0..term_count)
            .map(|_| TermStats::new(self.nparameters))
            .collect::<Vec<_>>();
        let mut right = (0..term_count)
            .map(|_| TermStats::new(self.nparameters))
            .collect::<Vec<_>>();
        let mut budgets = vec![0usize; term_count];
        let mut needs_more = vec![true; term_count];
        let mut scratch = PathScratch::new(self.operations.len());
        let target = gradient_tolerance / (term_count as f64).sqrt();
        let mut converged = false;
        let mut proxies = vec![f64::INFINITY; term_count];

        loop {
            for term_index in 0..term_count {
                if !needs_more[term_index] {
                    continue;
                }
                let old = budgets[term_index];
                let next = if old == 0 {
                    initial_samples_per_term
                } else {
                    old.saturating_mul(2).min(max_samples_per_term)
                };
                if next > old {
                    let term = &self.observable.terms()[term_index];
                    run_samples_batched(
                        &operations,
                        &self.initial_state,
                        self.nqubits,
                        self.smoothing,
                        term,
                        term_index,
                        0,
                        old,
                        next,
                        seed,
                        &mut left[term_index],
                        &mut scratch,
                    )?;
                    run_samples_batched(
                        &operations,
                        &self.initial_state,
                        self.nqubits,
                        self.smoothing,
                        term,
                        term_index,
                        1,
                        old,
                        next,
                        seed,
                        &mut right[term_index],
                        &mut scratch,
                    )?;
                    budgets[term_index] = next;
                }
            }

            for term_index in 0..term_count {
                proxies[term_index] = gradient_proxy(&left[term_index], &right[term_index])?;
            }
            if proxies.iter().all(|proxy| *proxy <= target) {
                converged = true;
                break;
            }
            let mut can_continue = false;
            for term_index in 0..term_count {
                needs_more[term_index] =
                    proxies[term_index] > target && budgets[term_index] < max_samples_per_term;
                can_continue |= needs_more[term_index];
            }
            if !can_continue {
                break;
            }
        }

        let (value, gradient, standard_error) = combine_adaptive(&left, &right, &budgets)?;
        let global_proxy = proxies
            .iter()
            .map(|proxy| proxy * proxy)
            .sum::<f64>()
            .sqrt();
        let total_paths = budgets.iter().try_fold(0usize, |sum, budget| {
            let paths = budget.checked_mul(2).ok_or(PauliError::Overflow {
                context: "counting adaptive SPPS paths",
            })?;
            sum.checked_add(paths).ok_or(PauliError::Overflow {
                context: "counting adaptive SPPS paths",
            })
        })?;
        Ok(SPPSEstimate {
            value,
            gradient,
            value_standard_error: standard_error,
            replicates: 2,
            samples_per_replicate: budgets,
            total_paths,
            seed,
            gradient_error_proxy: Some(global_proxy),
            term_gradient_error_proxies: Some(proxies),
            converged: Some(converged),
        })
    }

    fn validate_parameters(&self, parameters: &[f64]) -> Result<(), PauliError> {
        if parameters.len() != self.nparameters {
            return Err(PauliError::InvalidParameterLength {
                expected: self.nparameters,
                actual: parameters.len(),
            });
        }
        if let Some(index) = parameters.iter().position(|value| !value.is_finite()) {
            return Err(PauliError::NonFiniteParameter { index });
        }
        Ok(())
    }

    fn resolve_operations(&self, parameters: &[f64]) -> Result<Vec<ResolvedOperation>, PauliError> {
        self.operations
            .iter()
            .map(|operation| match operation {
                SppsOperation::Clifford1 { gate, wire, .. } => Ok(ResolvedOperation::Clifford1 {
                    gate: *gate,
                    wire: *wire,
                }),
                SppsOperation::Clifford2 {
                    gate, wire0, wire1, ..
                } => Ok(ResolvedOperation::Clifford2 {
                    gate: *gate,
                    wire0: *wire0,
                    wire1: *wire1,
                }),
                SppsOperation::Rotation {
                    generator_code,
                    wire0,
                    wire1,
                    parameter,
                    source_index,
                } => {
                    let (cosine, sine) = resolve_parameter(*parameter, parameters)?;
                    let slot = match parameter {
                        ParameterRef::Slot(index) => Some(*index),
                        ParameterRef::Static { .. } => None,
                    };
                    Ok(ResolvedOperation::Rotation {
                        generator_code: *generator_code,
                        wire0: *wire0,
                        wire1: *wire1,
                        cosine,
                        sine,
                        slot,
                        source_index: *source_index,
                    })
                }
            })
            .collect()
    }
}

fn empty_estimate(
    nparameters: usize,
    seed: u64,
    replicates: usize,
    converged: bool,
) -> SPPSEstimate {
    SPPSEstimate {
        value: 0.0,
        gradient: vec![0.0; nparameters],
        value_standard_error: 0.0,
        replicates,
        samples_per_replicate: Vec::new(),
        total_paths: 0,
        seed,
        gradient_error_proxy: (replicates == 2).then_some(0.0),
        term_gradient_error_proxies: (replicates == 2).then_some(Vec::new()),
        converged: (replicates == 2).then_some(converged),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_samples_batched(
    operations: &[ResolvedOperation],
    initial_state: &ProductState,
    nqubits: usize,
    smoothing: f64,
    term: &crate::operator::PauliTerm,
    term_index: usize,
    replicate: usize,
    start: usize,
    end: usize,
    seed: u64,
    stats: &mut TermStats,
    scratch: &mut PathScratch,
) -> Result<(), PauliError> {
    const CHUNK_SIZE: usize = 256;
    if end - start <= CHUNK_SIZE {
        return run_samples(
            operations,
            initial_state,
            nqubits,
            smoothing,
            term,
            term_index,
            replicate,
            start,
            end,
            seed,
            stats,
            scratch,
        );
    }

    let ranges = (start..end)
        .step_by(CHUNK_SIZE)
        .map(|chunk_start| (chunk_start, (chunk_start + CHUNK_SIZE).min(end)))
        .collect::<Vec<_>>();
    let partials = ranges
        .into_par_iter()
        .map(|(chunk_start, chunk_end)| {
            let mut partial = TermStats::new(stats.gradient_sum.len());
            let mut local_scratch = PathScratch::new(operations.len());
            run_samples(
                operations,
                initial_state,
                nqubits,
                smoothing,
                term,
                term_index,
                replicate,
                chunk_start,
                chunk_end,
                seed,
                &mut partial,
                &mut local_scratch,
            )
            .map(|()| partial)
        })
        .collect::<Vec<_>>();
    for partial in partials {
        let partial = partial?;
        stats.sum += partial.sum;
        stats.sum_squared += partial.sum_squared;
        for (total, value) in stats.gradient_sum.iter_mut().zip(partial.gradient_sum) {
            *total += value;
        }
        stats.count = stats
            .count
            .checked_add(partial.count)
            .ok_or(PauliError::Overflow {
                context: "counting SPPS samples",
            })?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_samples(
    operations: &[ResolvedOperation],
    initial_state: &ProductState,
    nqubits: usize,
    smoothing: f64,
    term: &crate::operator::PauliTerm,
    term_index: usize,
    replicate: usize,
    start: usize,
    end: usize,
    seed: u64,
    stats: &mut TermStats,
    scratch: &mut PathScratch,
) -> Result<(), PauliError> {
    for sample_index in start..end {
        scratch.clear();
        let mut current = PackedKey::from_word(&term.word);
        let mut sign = 1.0;
        for operation in operations {
            match operation {
                ResolvedOperation::Clifford1 { gate, wire } => {
                    sign *= apply_clifford1_in_place(&mut current, *gate, *wire);
                }
                ResolvedOperation::Clifford2 { gate, wire0, wire1 } => {
                    sign *= apply_clifford2_in_place(&mut current, *gate, *wire0, *wire1);
                }
                ResolvedOperation::Rotation {
                    generator_code,
                    wire0,
                    wire1,
                    cosine,
                    sine,
                    slot,
                    source_index,
                } => {
                    let (phase, mapped_first, mapped_second) =
                        generator_transition(&current, *generator_code, *wire0, *wire1);
                    if matches!(
                        phase,
                        crate::word::PauliPhase::PlusI | crate::word::PauliPhase::MinusI
                    ) {
                        let q = (cosine.abs() + smoothing)
                            / (cosine.abs() + sine.abs() + 2.0 * smoothing);
                        let random = counter_random(
                            seed,
                            1,
                            term_index,
                            replicate,
                            sample_index,
                            *source_index,
                        );
                        if random < q {
                            if q <= 0.0 {
                                return Err(PauliError::NonFiniteCoefficient {
                                    index: sample_index,
                                });
                            }
                            scratch.ratios.push(cosine / q);
                            scratch.derivatives.push(slot.map_or(0.0, |_| -sine / q));
                            scratch.slots.push(*slot);
                        } else {
                            let probability = 1.0 - q;
                            let local_sign = phase_sign_i(phase);
                            current.set_code(*wire0, mapped_first);
                            if let Some(second_wire) = wire1 {
                                current.set_code(*second_wire, mapped_second);
                            }
                            if probability <= 0.0 {
                                return Err(PauliError::NonFiniteCoefficient {
                                    index: sample_index,
                                });
                            }
                            scratch.ratios.push(local_sign * sine / probability);
                            scratch
                                .derivatives
                                .push(slot.map_or(0.0, |_| local_sign * cosine / probability));
                            scratch.slots.push(*slot);
                        }
                    }
                }
            }
            if !sign.is_finite() {
                return Err(PauliError::NonFiniteCoefficient {
                    index: sample_index,
                });
            }
        }

        let local_expectation = expectation_of_key(&current, initial_state, nqubits);
        let mut nonzero_product = 1.0;
        let mut zero_count = 0usize;
        let mut zero_index = usize::MAX;
        let mut use_stable_products = false;
        for ratio in scratch.ratios.iter().copied() {
            if ratio == 0.0 {
                zero_count += 1;
            } else if ratio.abs() <= STABLE_PRODUCT_RATIO_THRESHOLD {
                use_stable_products = true;
            }
        }
        if zero_count == 0 && use_stable_products {
            scratch.prefix.resize(scratch.ratios.len() + 1, 1.0);
            scratch.suffix.resize(scratch.ratios.len() + 1, 1.0);
            for (index, ratio) in scratch.ratios.iter().copied().enumerate() {
                scratch.prefix[index + 1] = scratch.prefix[index] * ratio;
                if !scratch.prefix[index + 1].is_finite() {
                    return Err(PauliError::NonFiniteCoefficient {
                        index: sample_index,
                    });
                }
            }
            for index in (0..scratch.ratios.len()).rev() {
                scratch.suffix[index] = scratch.ratios[index] * scratch.suffix[index + 1];
                if !scratch.suffix[index].is_finite() {
                    return Err(PauliError::NonFiniteCoefficient {
                        index: sample_index,
                    });
                }
            }
        } else {
            for (index, ratio) in scratch.ratios.iter().copied().enumerate() {
                if ratio == 0.0 {
                    zero_index = index;
                } else {
                    nonzero_product *= ratio;
                    if !nonzero_product.is_finite() {
                        return Err(PauliError::NonFiniteCoefficient {
                            index: sample_index,
                        });
                    }
                }
            }
        }
        let value_factor = term.coefficient.re * local_expectation * sign;
        let path_product = if use_stable_products && zero_count == 0 {
            scratch.prefix[scratch.ratios.len()]
        } else if zero_count == 0 {
            nonzero_product
        } else {
            0.0
        };
        let sample_value = value_factor * path_product;
        if !sample_value.is_finite() {
            return Err(PauliError::NonFiniteCoefficient {
                index: sample_index,
            });
        }
        stats.sum += sample_value;
        stats.sum_squared += sample_value * sample_value;
        if !stats.sum.is_finite() || !stats.sum_squared.is_finite() {
            return Err(PauliError::NonFiniteCoefficient {
                index: sample_index,
            });
        }
        for index in 0..scratch.ratios.len() {
            if let Some(slot) = scratch.slots[index] {
                let product_without_factor = if use_stable_products && zero_count == 0 {
                    scratch.prefix[index] * scratch.suffix[index + 1]
                } else {
                    match zero_count {
                        0 => nonzero_product / scratch.ratios[index],
                        1 if index == zero_index => nonzero_product,
                        _ => 0.0,
                    }
                };
                stats.gradient_sum[slot] +=
                    value_factor * scratch.derivatives[index] * product_without_factor;
                if !stats.gradient_sum[slot].is_finite() {
                    return Err(PauliError::NonFiniteCoefficient {
                        index: sample_index,
                    });
                }
            }
        }
        stats.count = stats.count.checked_add(1).ok_or(PauliError::Overflow {
            context: "counting SPPS samples",
        })?;
    }
    Ok(())
}

fn combine_fixed(
    stats: &[TermStats],
    nparameters: usize,
) -> Result<(f64, Vec<f64>, f64), PauliError> {
    let mut value = 0.0;
    let mut gradient = vec![0.0; nparameters];
    let mut variance = 0.0;
    for stat in stats {
        let count = stat.count as f64;
        let mean = stat.sum / count;
        value += mean;
        let sample_variance = (stat.sum_squared / count - mean * mean).max(0.0);
        variance += sample_variance / count;
        for (output, sum) in gradient.iter_mut().zip(&stat.gradient_sum) {
            *output += sum / count;
        }
    }
    if !value.is_finite() || gradient.iter().any(|entry| !entry.is_finite()) {
        return Err(PauliError::NonFiniteCoefficient { index: 0 });
    }
    Ok((value, gradient, variance.max(0.0).sqrt()))
}

fn gradient_proxy(left: &TermStats, right: &TermStats) -> Result<f64, PauliError> {
    let left_count = left.count as f64;
    let right_count = right.count as f64;
    let squared = left
        .gradient_sum
        .iter()
        .zip(&right.gradient_sum)
        .map(|(a, b)| {
            let difference = a / left_count - b / right_count;
            difference * difference
        })
        .sum::<f64>();
    let result = 0.5 * squared.sqrt();
    result
        .is_finite()
        .then_some(result)
        .ok_or(PauliError::NonFiniteCoefficient { index: 0 })
}

fn combine_adaptive(
    left: &[TermStats],
    right: &[TermStats],
    budgets: &[usize],
) -> Result<(f64, Vec<f64>, f64), PauliError> {
    let nparameters = left.first().map_or(0, |stat| stat.gradient_sum.len());
    let mut value = 0.0;
    let mut gradient = vec![0.0; nparameters];
    let mut variance = 0.0;
    for ((left_stat, right_stat), &budget) in left.iter().zip(right).zip(budgets) {
        let count = budget as f64;
        let left_mean = left_stat.sum / count;
        let right_mean = right_stat.sum / count;
        value += 0.5 * (left_mean + right_mean);
        let left_var = (left_stat.sum_squared / count - left_mean * left_mean).max(0.0);
        let right_var = (right_stat.sum_squared / count - right_mean * right_mean).max(0.0);
        variance += 0.5 * (left_var + right_var) / count;
        for ((output, left_sum), right_sum) in gradient
            .iter_mut()
            .zip(&left_stat.gradient_sum)
            .zip(&right_stat.gradient_sum)
        {
            *output += 0.5 * (left_sum + right_sum) / count;
        }
    }
    if !value.is_finite() || gradient.iter().any(|entry| !entry.is_finite()) {
        return Err(PauliError::NonFiniteCoefficient { index: 0 });
    }
    Ok((value, gradient, variance.max(0.0).sqrt()))
}

fn counter_random(
    seed: u64,
    mode: u64,
    term: usize,
    replicate: usize,
    sample: usize,
    gate: usize,
) -> f64 {
    let mut value = seed
        ^ mode.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (term as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ (replicate as u64).wrapping_mul(0x94d0_49bb_1331_11eb)
        ^ (sample as u64).wrapping_mul(0xd2b7_4407_b1ce_6e93)
        ^ (gate as u64).wrapping_mul(0x632b_be59_bd9b_4e01);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    ((value >> 11) as f64) * RANDOM_SCALE
}
