use std::collections::BTreeMap;

use rayon::prelude::*;

/// One exact, aggregated component-local transition.  The table belongs to a
/// single component and therefore never encodes a selected public basis or a
/// product-space permutation.
#[derive(Clone)]
struct LocalAggregateTransition {
    destination_rank: usize,
    value: Complex64,
}

/// Fast lazy execution for schedules whose complete action is confined to
/// one contiguous factor component.  It is deliberately representation-led:
/// a Qubit, Qudit, or Boson spelling never participates in selection.
struct ContiguousLocalAggregateExecutor {
    component_dimension: usize,
    component_stride: usize,
    transitions: Box<[Box<[LocalAggregateTransition]>]>,
    estimated_bytes: u128,
}

impl ContiguousLocalAggregateExecutor {
    fn apply_into<const ADJOINT: bool>(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), PauliError> {
        if state.len() != output.len()
            || self.component_dimension == 0
            || state.len() % self.component_stride != 0
        {
            return Err(invalid_sector());
        }
        output.fill(Complex64::new(0.0, 0.0));
        for (source, &state_value) in state.iter().enumerate() {
            let source_rank = (source / self.component_stride) % self.component_dimension;
            let component_base = source
                .checked_sub(source_rank.checked_mul(self.component_stride).ok_or(
                    PauliError::Overflow {
                        context: "indexing factorized component source",
                    },
                )?)
                .ok_or(PauliError::Overflow {
                    context: "indexing factorized component source",
                })?;
            for transition in &self.transitions[source_rank] {
                let row = component_base
                    .checked_add(
                        transition
                            .destination_rank
                            .checked_mul(self.component_stride)
                            .ok_or(PauliError::Overflow {
                                context: "indexing factorized component destination",
                            })?,
                    )
                    .ok_or(PauliError::Overflow {
                        context: "indexing factorized component destination",
                    })?;
                if ADJOINT {
                    output[source] += transition.value.conj() * state[row];
                } else {
                    output[row] += transition.value * state_value;
                }
            }
        }
        Ok(())
    }

    fn estimated_bytes(&self) -> u128 {
        self.estimated_bytes
    }
}

#[derive(Clone)]
struct ComponentScheduleTransition {
    destination_rank: usize,
    value: Complex64,
}

type ComponentScheduleTable = Box<[Option<ComponentScheduleTransition>]>;

struct ContiguousScheduleAction {
    /// `None` means the schedule leaves this component unchanged.  A present
    /// table has one entry per source component rank; its inner `None` marks
    /// an inapplicable local action.
    component_actions: Box<[Option<ComponentScheduleTable>]>,
    active_components: Box<[usize]>,
    coefficient: Complex64,
    fermion: Option<FermionMaskDescriptor>,
}

/// Direct factor-product execution for termwise-conserved schedules.  It
/// retains transition tables only for individual components, never for their
/// product, so the retained size is `sum(component_dimension * schedules)`.
struct ContiguousFactorScheduleExecutor {
    component_dimensions: Box<[usize]>,
    component_strides: Box<[usize]>,
    /// Per-component source occupation bits in logical Fermion-mode order.
    /// These are local component caches, not public-sector state maps.
    component_fermion_words: Box<[Box<[u64]>]>,
    actions: Box<[ContiguousScheduleAction]>,
    fermion_word_count: usize,
    has_fermion: bool,
    estimated_bytes: u128,
}

impl ContiguousFactorScheduleExecutor {
    fn apply_into<const ADJOINT: bool>(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), PauliError> {
        if state.len() != output.len()
            || self.component_dimensions.len() != self.component_strides.len()
            || self.component_dimensions.len() != self.component_fermion_words.len()
        {
            return Err(invalid_sector());
        }
        output.fill(Complex64::new(0.0, 0.0));
        let mut source_ranks = vec![0_usize; self.component_dimensions.len()];
        let mut words = vec![0_u64; self.fermion_word_count];
        for (column, &state_value) in state.iter().enumerate() {
            for (component, source_rank) in source_ranks.iter_mut().enumerate() {
                let rank = (column / self.component_strides[component])
                    % self.component_dimensions[component];
                *source_rank = rank;
            }
            if self.has_fermion {
                words.fill(0);
                for (component, &source_rank) in source_ranks.iter().enumerate() {
                    let offset = source_rank.checked_mul(self.fermion_word_count).ok_or(
                        PauliError::Overflow {
                            context: "indexing factorized Fermion source",
                        },
                    )?;
                    let local_words = &self.component_fermion_words[component]
                        [offset..offset + self.fermion_word_count];
                    for (word, &local_word) in words.iter_mut().zip(local_words) {
                        *word |= local_word;
                    }
                }
            }
            for action in &self.actions {
                let mut value = action.coefficient;
                let mut valid = true;
                let mut row = column;
                for &component in &action.active_components {
                    let transitions = action.component_actions[component]
                        .as_ref()
                        .expect("active factorized component has transitions");
                    let Some(transition) = transitions[source_ranks[component]].as_ref() else {
                        valid = false;
                        break;
                    };
                    let source_offset = source_ranks[component]
                        .checked_mul(self.component_strides[component])
                        .ok_or(PauliError::Overflow {
                            context: "indexing factorized schedule source",
                        })?;
                    let destination_offset = transition
                        .destination_rank
                        .checked_mul(self.component_strides[component])
                        .ok_or(PauliError::Overflow {
                            context: "indexing factorized schedule destination",
                        })?;
                    row = if destination_offset >= source_offset {
                        row.checked_add(destination_offset - source_offset)
                    } else {
                        row.checked_sub(source_offset - destination_offset)
                    }
                    .ok_or(PauliError::Overflow {
                        context: "indexing factorized schedule destination",
                    })?;
                    value *= transition.value;
                }
                if !valid {
                    continue;
                }
                if let Some(fermion) = action.fermion.as_ref() {
                    let Some(amplitude) = fermion.amplitude_from_words(&words, value) else {
                        continue;
                    };
                    value = amplitude;
                }
                if ADJOINT {
                    output[column] += value.conj() * state[row];
                } else {
                    output[row] += value * state_value;
                }
            }
        }
        Ok(())
    }

    fn estimated_bytes(&self) -> u128 {
        self.estimated_bytes
    }
}

fn build_contiguous_factor_schedule_executor(
    sector: &ChargeSectorPlan,
    layout: &PreparedChargeTransitionPlanLayout,
    termwise_conserved: bool,
) -> Result<Option<ContiguousFactorScheduleExecutor>, PauliError> {
    if !termwise_conserved {
        return Ok(None);
    }
    let Some(factorized) = sector.factorized_execution_layout() else {
        return Ok(None);
    };
    let Some(component_strides) = factorized.contiguous_component_strides() else {
        return Ok(None);
    };
    let components = factorized.components();
    let axis_count = layout.local_dimensions.len();
    let mut component_axes = Vec::with_capacity(components.len());
    let mut component_indexes = Vec::with_capacity(components.len());
    let mut local_axes = vec![None; axis_count];
    for (component_index, component) in components.iter().enumerate() {
        let axes = component.axes().to_vec().into_boxed_slice();
        for (local_axis, &axis) in axes.iter().enumerate() {
            if axis >= axis_count
                || local_axes[axis]
                    .replace((component_index, local_axis))
                    .is_some()
            {
                return Err(invalid_sector());
            }
        }
        component_axes.push(axes);
        component_indexes.push(component.index().clone());
    }
    if local_axes.iter().any(Option::is_none) {
        return Err(invalid_sector());
    }

    let mut transition_count = 0usize;
    let mut actions = Vec::with_capacity(layout.schedules.len());
    for schedule in &layout.schedules {
        if schedule.zero {
            continue;
        }
        let mut component_actions = Vec::with_capacity(components.len());
        for component_index in 0..components.len() {
            let axes = &component_axes[component_index];
            let index = &component_indexes[component_index];
            let mut source = vec![0_u64; axes.len()];
            let mut destination = vec![0_u64; axes.len()];
            let mut touched = false;
            let mut rows = Vec::with_capacity(index.dimension());
            for source_rank in 0..index.dimension() {
                index.unrank_into(
                    u64::try_from(source_rank).map_err(|_| PauliError::Overflow {
                        context: "converting factorized component rank",
                    })?,
                    &mut source,
                )?;
                destination.copy_from_slice(&source);
                let mut value = Complex64::new(1.0, 0.0);
                let mut valid = true;
                if let Some(fermion) = schedule.fermion.as_ref() {
                    for (mode, &axis) in fermion.mode_axes.iter().enumerate() {
                        let Some((mapped_component, local_axis)) = local_axes[axis] else {
                            return Err(invalid_sector());
                        };
                        if mapped_component != component_index {
                            continue;
                        }
                        let word = mode / 64;
                        let bit = 1_u64 << (mode % 64);
                        let occupation = source[local_axis];
                        if occupation > 1
                            || (fermion.required_occupied[word] & bit != 0 && occupation == 0)
                            || (fermion.required_empty[word] & bit != 0 && occupation != 0)
                        {
                            valid = false;
                            break;
                        }
                        if fermion.flip_mask[word] & bit != 0 {
                            destination[local_axis] = 1 - occupation;
                        }
                    }
                    touched |= fermion.mode_axes.iter().enumerate().any(|(mode, &axis)| {
                        let word = mode / 64;
                        let bit = 1_u64 << (mode % 64);
                        local_axes[axis].is_some_and(|(mapped_component, _)| {
                            mapped_component == component_index
                                && (fermion.required_occupied[word] & bit != 0
                                    || fermion.required_empty[word] & bit != 0
                                    || fermion.flip_mask[word] & bit != 0)
                        })
                    });
                }
                if valid {
                    for &(axis, creation, annihilation) in schedule.boson_blocks.iter().rev() {
                        let Some((mapped_component, local_axis)) = local_axes[axis] else {
                            return Err(invalid_sector());
                        };
                        if mapped_component != component_index {
                            continue;
                        }
                        touched = true;
                        let mut occupation = destination[local_axis];
                        for _ in 0..annihilation {
                            if occupation == 0 {
                                valid = false;
                                break;
                            }
                            value *= (occupation as f64).sqrt();
                            occupation -= 1;
                        }
                        if !valid {
                            break;
                        }
                        for _ in 0..creation {
                            let Some(next) = occupation.checked_add(1) else {
                                valid = false;
                                break;
                            };
                            if next >= layout.local_dimensions[axis] {
                                valid = false;
                                break;
                            }
                            value *= (next as f64).sqrt();
                            occupation = next;
                        }
                        if !valid {
                            break;
                        }
                        destination[local_axis] = occupation;
                    }
                }
                if valid {
                    for &(axis, code) in schedule.qubit_ops.iter().chain(schedule.mapped_ops.iter())
                    {
                        let Some((mapped_component, local_axis)) = local_axes[axis] else {
                            return Err(invalid_sector());
                        };
                        if mapped_component != component_index {
                            continue;
                        }
                        touched = true;
                        let bit = destination[local_axis];
                        if bit > 1 {
                            return Err(invalid_sector());
                        }
                        match code {
                            0 => {}
                            1 => destination[local_axis] = 1 - bit,
                            2 => {
                                destination[local_axis] = 1 - bit;
                                value *= if bit == 0 {
                                    Complex64::new(0.0, 1.0)
                                } else {
                                    Complex64::new(0.0, -1.0)
                                };
                            }
                            3 if bit == 1 => value = -value,
                            3 => {}
                            _ => return Err(invalid_sector()),
                        }
                    }
                }
                if valid {
                    for &(axis, shift, phase_power) in &schedule.qudit_ops {
                        let Some((mapped_component, local_axis)) = local_axes[axis] else {
                            return Err(invalid_sector());
                        };
                        if mapped_component != component_index {
                            continue;
                        }
                        touched = true;
                        let input = destination[local_axis];
                        if input >= layout.qudit_dimension {
                            return Err(invalid_sector());
                        }
                        let exponent = (u128::from(phase_power) * u128::from(input))
                            % u128::from(layout.qudit_dimension);
                        value *= Complex64::from_polar(
                            1.0,
                            2.0 * std::f64::consts::PI * exponent as f64
                                / layout.qudit_dimension as f64,
                        );
                        destination[local_axis] =
                            (input + u64::from(shift)) % layout.qudit_dimension;
                    }
                }
                if !touched {
                    break;
                }
                if !valid || value.re == 0.0 && value.im == 0.0 {
                    rows.push(None);
                    continue;
                }
                let destination_rank = match index.rank(&destination) {
                    Ok(rank) => usize::try_from(rank).map_err(|_| PauliError::Overflow {
                        context: "converting factorized component destination",
                    })?,
                    Err(PauliError::InvalidIndex { .. })
                    | Err(PauliError::InvalidSector { .. }) => {
                        return Err(PauliError::InvalidSector {
                            context: "operator leaks outside the selected charge sector",
                        });
                    }
                    Err(error) => return Err(error),
                };
                rows.push(Some(ComponentScheduleTransition {
                    destination_rank,
                    value,
                }));
            }
            if touched {
                if rows.len() != index.dimension() {
                    return Err(invalid_sector());
                }
                transition_count =
                    transition_count
                        .checked_add(rows.len())
                        .ok_or(PauliError::Overflow {
                            context: "sizing factorized schedule transitions",
                        })?;
                component_actions.push(Some(rows.into_boxed_slice()));
            } else {
                component_actions.push(None);
            }
        }
        actions.push(ContiguousScheduleAction {
            active_components: component_actions
                .iter()
                .enumerate()
                .filter_map(|(component, transitions)| transitions.is_some().then_some(component))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            component_actions: component_actions.into_boxed_slice(),
            coefficient: schedule.coefficient,
            fermion: schedule.fermion.clone(),
        });
    }
    let fermion_descriptor = layout
        .schedules
        .iter()
        .find_map(|schedule| schedule.fermion.as_ref());
    let fermion_word_count =
        fermion_descriptor.map_or(0, |descriptor| descriptor.required_occupied.len());
    let mut component_fermion_words = Vec::with_capacity(components.len());
    for (component_index, index) in component_indexes.iter().enumerate() {
        let entry_count =
            index
                .dimension()
                .checked_mul(fermion_word_count)
                .ok_or(PauliError::Overflow {
                    context: "sizing factorized Fermion component cache",
                })?;
        let mut packed = vec![0_u64; entry_count];
        if let Some(fermion) = fermion_descriptor {
            let mut values = vec![0_u64; component_axes[component_index].len()];
            for rank in 0..index.dimension() {
                index.unrank_into(
                    u64::try_from(rank).map_err(|_| PauliError::Overflow {
                        context: "converting factorized component rank",
                    })?,
                    &mut values,
                )?;
                let offset = rank
                    .checked_mul(fermion_word_count)
                    .ok_or(PauliError::Overflow {
                        context: "indexing factorized Fermion component cache",
                    })?;
                for (mode, &axis) in fermion.mode_axes.iter().enumerate() {
                    let Some((mapped_component, local_axis)) = local_axes[axis] else {
                        return Err(invalid_sector());
                    };
                    if mapped_component == component_index && values[local_axis] != 0 {
                        packed[offset + mode / 64] |= 1_u64 << (mode % 64);
                    }
                }
            }
        }
        component_fermion_words.push(packed.into_boxed_slice());
    }
    let metadata_bytes = (component_axes.iter().map(|axes| axes.len()).sum::<usize>() as u128)
        .checked_mul(std::mem::size_of::<usize>() as u128)
        .and_then(|value| {
            value.checked_add(
                (transition_count as u128).checked_mul(std::mem::size_of::<
                    Option<ComponentScheduleTransition>,
                >() as u128)?,
            )
        })
        .and_then(|value| {
            value.checked_add(
                (component_fermion_words
                    .iter()
                    .map(|words| words.len())
                    .sum::<usize>() as u128)
                    .checked_mul(std::mem::size_of::<u64>() as u128)?,
            )
        })
        .ok_or(PauliError::Overflow {
            context: "estimating factorized schedule transitions",
        })?;
    Ok(Some(ContiguousFactorScheduleExecutor {
        component_dimensions: components
            .iter()
            .map(|component| component.index().dimension())
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        component_strides: component_strides.into_boxed_slice(),
        component_fermion_words: component_fermion_words.into_boxed_slice(),
        actions: actions.into_boxed_slice(),
        fermion_word_count,
        has_fermion: fermion_descriptor.is_some(),
        estimated_bytes: metadata_bytes,
    }))
}

/// Lower a complete schedule set into compact local actions when all action
/// axes belong to one contiguous factor.  Aggregation happens before local
/// membership is checked, retaining the exact cancellation/leakage rule used
/// by General execution.
fn build_contiguous_local_aggregate_executor(
    sector: &ChargeSectorPlan,
    layout: &PreparedChargeTransitionPlanLayout,
) -> Result<Option<ContiguousLocalAggregateExecutor>, PauliError> {
    let Some(factorized) = sector.factorized_execution_layout() else {
        return Ok(None);
    };
    let Some(strides) = factorized.contiguous_component_strides() else {
        return Ok(None);
    };
    if layout
        .schedules
        .iter()
        .any(|schedule| schedule.fermion.is_some())
    {
        return Ok(None);
    }
    let components = factorized.components();
    let mut active_component = None;
    let mut spans_components = false;
    for schedule in &layout.schedules {
        let mut record_axis = |axis: usize| -> Result<(), PauliError> {
            let component = *factorized
                .axis_components()
                .get(axis)
                .ok_or_else(invalid_sector)?;
            if let Some(active) = active_component {
                if active != component {
                    spans_components = true;
                }
            } else {
                active_component = Some(component);
            }
            Ok(())
        };
        for &(axis, _, _) in &schedule.boson_blocks {
            record_axis(axis)?;
        }
        for &(axis, _) in schedule.qubit_ops.iter().chain(schedule.mapped_ops.iter()) {
            record_axis(axis)?;
        }
        for &(axis, _, _) in &schedule.qudit_ops {
            record_axis(axis)?;
        }
    }
    if spans_components {
        return Ok(None);
    }
    let Some(component) = active_component else {
        return Ok(None);
    };
    let component_plan = components.get(component).ok_or_else(invalid_sector)?;
    let component_dimension = component_plan.index().dimension();
    let component_axes = component_plan.axes();
    let mut local_source = vec![0_u64; component_axes.len()];
    let mut source = vec![0_u64; layout.local_dimensions.len()];
    let mut destination = vec![0_u64; layout.local_dimensions.len()];
    let mut words = Vec::new();
    let mut transitions = Vec::with_capacity(component_dimension);
    let mut entry_count = 0usize;
    for source_rank in 0..component_dimension {
        factorized.unrank_component_into(component, source_rank, &mut local_source)?;
        source.fill(0);
        for (&axis, &value) in component_axes.iter().zip(&local_source) {
            source[axis] = value;
        }
        let mut grouped = BTreeMap::<Vec<u64>, Complex64>::new();
        for schedule in &layout.schedules {
            let Some(value) = apply_compiled_charge_schedule(
                schedule,
                &source,
                &mut destination,
                &layout.local_dimensions,
                layout.qudit_dimension,
                &mut words,
            )?
            else {
                continue;
            };
            let key = component_axes
                .iter()
                .map(|&axis| destination[axis])
                .collect::<Vec<_>>();
            *grouped.entry(key).or_default() += value;
        }
        let mut row = Vec::with_capacity(grouped.len());
        for (destination_values, value) in grouped {
            if value.re == 0.0 && value.im == 0.0 {
                continue;
            }
            let destination_rank = match factorized.rank_component(component, &destination_values) {
                Ok(rank) => rank,
                Err(PauliError::InvalidIndex { .. }) | Err(PauliError::InvalidSector { .. }) => {
                    return Err(PauliError::InvalidSector {
                        context: "operator leaks outside the selected charge sector",
                    });
                }
                Err(error) => return Err(error),
            };
            row.push(LocalAggregateTransition {
                destination_rank,
                value,
            });
        }
        entry_count = entry_count
            .checked_add(row.len())
            .ok_or(PauliError::Overflow {
                context: "sizing factorized local transitions",
            })?;
        transitions.push(row.into_boxed_slice());
    }
    let estimated_bytes = (entry_count as u128)
        .checked_mul(std::mem::size_of::<LocalAggregateTransition>() as u128)
        .and_then(|value| {
            value.checked_add(
                (component_dimension as u128)
                    .checked_mul(std::mem::size_of::<Box<[LocalAggregateTransition]>>() as u128)?,
            )
        })
        .ok_or(PauliError::Overflow {
            context: "estimating factorized local transitions",
        })?;
    Ok(Some(ContiguousLocalAggregateExecutor {
        component_dimension,
        component_stride: *strides.get(component).ok_or_else(invalid_sector)?,
        transitions: transitions.into_boxed_slice(),
        estimated_bytes,
    }))
}

/// Apply a one-component contiguous binary fixed-population layout directly
/// in its compact occupation word representation. The layout proof and the
/// shared Fermion descriptor are the only selectors: no physical-family name
/// or Python-derived hint participates. Gathering by destination keeps each
/// output element independent, so the parallel branch has no scatter races.
fn try_apply_contiguous_fixed_population_into(
    plan: &ChargeSectorPlan,
    layout: &PreparedChargeTransitionPlanLayout,
    state: &[Complex64],
    output: &mut [Complex64],
    termwise_conserved: bool,
) -> Result<bool, PauliError> {
    if !termwise_conserved {
        return Ok(false);
    }
    let Some(index) = plan
        .factorized_layout()
        .and_then(FactorizedLayout::contiguous_fixed_population)
    else {
        return Ok(false);
    };
    if index.dimension() != state.len() || index.sites() > usize::BITS as usize {
        return Ok(false);
    }
    if layout.schedules.iter().any(|schedule| {
        !schedule.boson_blocks.is_empty()
            || !schedule.qubit_ops.is_empty()
            || !schedule.mapped_ops.is_empty()
            || !schedule.qudit_ops.is_empty()
            || (!schedule.zero
                && schedule
                    .fermion
                    .as_ref()
                    .is_some_and(|mask| mask.contiguous_public_action().is_none()))
    }) {
        return Ok(false);
    }

    let apply_one = |(destination_rank, destination_value): (usize, &mut Complex64)| {
        let destination = index.unrank_contiguous_binary_word(destination_rank)?;
        let mut total = Complex64::new(0.0, 0.0);
        for schedule in &layout.schedules {
            if schedule.zero {
                continue;
            }
            if let Some(mask) = schedule.fermion.as_ref() {
                let action = mask
                    .contiguous_public_action()
                    .expect("contiguous fixed-population eligibility was checked");
                let source = action.source_for_destination(destination);
                let Some(amplitude) = action.amplitude(source, schedule.coefficient) else {
                    continue;
                };
                let source_rank = index.rank_contiguous_binary_word(source)?;
                total += amplitude * state[source_rank];
            } else {
                total += schedule.coefficient * state[destination_rank];
            }
        }
        *destination_value = total;
        Ok(())
    };
    if state.len() >= 4096 && rayon::current_num_threads() > 1 {
        output.par_iter_mut().enumerate().try_for_each(apply_one)?;
    } else {
        output.iter_mut().enumerate().try_for_each(apply_one)?;
    }
    layout.counters.publish(ChargeApplyStats::default());
    Ok(true)
}

/// Apply two contiguous fixed-population components in the public mixed-radix
/// order. Each component may retain only its bounded local rank/unrank cache;
/// the cross-product is never materialized or permuted.
fn try_apply_contiguous_fixed_population_pair_into(
    plan: &ChargeSectorPlan,
    layout: &PreparedChargeTransitionPlanLayout,
    state: &[Complex64],
    output: &mut [Complex64],
    termwise_conserved: bool,
) -> Result<bool, PauliError> {
    if !termwise_conserved {
        return Ok(false);
    }
    let Some((left, right)) = plan
        .factorized_layout()
        .and_then(FactorizedLayout::contiguous_fixed_population_pair)
    else {
        return Ok(false);
    };
    let total_dimension =
        left.dimension()
            .checked_mul(right.dimension())
            .ok_or(PauliError::Overflow {
                context: "sizing contiguous fixed-population components",
            })?;
    let total_sites = left
        .sites()
        .checked_add(right.sites())
        .ok_or(PauliError::Overflow {
            context: "sizing contiguous fixed-population components",
        })?;
    if total_dimension != state.len() || total_sites > usize::BITS as usize {
        return Ok(false);
    }
    if layout.schedules.iter().any(|schedule| {
        !schedule.boson_blocks.is_empty()
            || !schedule.qubit_ops.is_empty()
            || !schedule.mapped_ops.is_empty()
            || !schedule.qudit_ops.is_empty()
            || (!schedule.zero
                && schedule
                    .fermion
                    .as_ref()
                    .is_some_and(|mask| mask.contiguous_public_action().is_none()))
    }) {
        return Ok(false);
    }
    let right_mask = if right.sites() == usize::BITS as usize {
        usize::MAX
    } else {
        (1_usize << right.sites()) - 1
    };
    let right_dimension = right.dimension();
    let apply_one = |(destination_rank, destination_value): (usize, &mut Complex64)| {
        let left_rank = destination_rank / right_dimension;
        let right_rank = destination_rank % right_dimension;
        let destination = (left.unrank_contiguous_binary_word(left_rank)? << right.sites())
            | right.unrank_contiguous_binary_word(right_rank)?;
        let mut total = Complex64::new(0.0, 0.0);
        for schedule in &layout.schedules {
            if schedule.zero {
                continue;
            }
            if let Some(mask) = schedule.fermion.as_ref() {
                let action = mask
                    .contiguous_public_action()
                    .expect("contiguous fixed-population eligibility was checked");
                let source = action.source_for_destination(destination);
                let Some(amplitude) = action.amplitude(source, schedule.coefficient) else {
                    continue;
                };
                let source_left_rank = left.rank_contiguous_binary_word(source >> right.sites())?;
                let source_right_rank = right.rank_contiguous_binary_word(source & right_mask)?;
                let source_rank = source_left_rank
                    .checked_mul(right_dimension)
                    .and_then(|value| value.checked_add(source_right_rank))
                    .ok_or(PauliError::Overflow {
                        context: "ranking contiguous fixed-population destination",
                    })?;
                total += amplitude * state[source_rank];
            } else {
                total += schedule.coefficient * state[destination_rank];
            }
        }
        *destination_value = total;
        Ok(())
    };
    if state.len() >= 4096 && rayon::current_num_threads() > 1 {
        output.par_iter_mut().enumerate().try_for_each(apply_one)?;
    } else {
        output.iter_mut().enumerate().try_for_each(apply_one)?;
    }
    layout.counters.publish(ChargeApplyStats::default());
    Ok(true)
}

/// Apply a structured operator directly against a reusable charge-sector plan.
///
/// This is the explicitly lazy counterpart to the native CSR compiler. It keeps only the input/output
/// vectors and one source-column destination aggregate in memory; it does not
/// retain the full restricted transition graph. Destination terms are
/// aggregated before sector membership is checked, preserving the same exact
/// cancellation semantics as eager compilation.
#[allow(clippy::too_many_arguments)]
fn apply_charge_mvp_from_prepared_plan<const ADJOINT: bool>(
    plan: &ChargeSectorPlan,
    layout: &PreparedChargeTransitionPlanLayout,
    terms: &[ChargeTransitionTerm],
    local_aggregate: Option<&ContiguousLocalAggregateExecutor>,
    factor_schedule: Option<&ContiguousFactorScheduleExecutor>,
    state: &[Complex64],
    termwise_conserved: bool,
    max_bytes: u128,
) -> Result<Vec<Complex64>, PauliError> {
    let dimension = layout.dimension;
    let output_bytes = dimension
        .checked_mul(std::mem::size_of::<Complex64>())
        .ok_or(PauliError::Overflow {
            context: "estimating charge MVP output",
        })?;
    if output_bytes as u128 > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: output_bytes as u128,
            limit: max_bytes,
        });
    }
    let mut output = vec![Complex64::new(0.0, 0.0); dimension];
    if let Some(executor) = local_aggregate {
        executor.apply_into::<ADJOINT>(state, &mut output)?;
        return Ok(output);
    }
    apply_charge_mvp_from_prepared_plan_into::<ADJOINT>(
        plan,
        layout,
        terms,
        local_aggregate,
        factor_schedule,
        state,
        &mut output,
        termwise_conserved,
        max_bytes,
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn apply_charge_mvp_from_prepared_plan_into<const ADJOINT: bool>(
    plan: &ChargeSectorPlan,
    layout: &PreparedChargeTransitionPlanLayout,
    terms: &[ChargeTransitionTerm],
    local_aggregate: Option<&ContiguousLocalAggregateExecutor>,
    factor_schedule: Option<&ContiguousFactorScheduleExecutor>,
    state: &[Complex64],
    output: &mut [Complex64],
    termwise_conserved: bool,
    max_bytes: u128,
) -> Result<(), PauliError> {
    let dimension = layout.dimension;
    let local_dimensions = &layout.local_dimensions;
    let qudit_dimension = layout.qudit_dimension;
    let schedules = &layout.schedules;
    if dimension != plan.dimension() || state.len() != dimension || output.len() != dimension {
        return Err(invalid_sector());
    }

    let output_bytes = dimension
        .checked_mul(std::mem::size_of::<Complex64>())
        .ok_or(PauliError::Overflow {
            context: "estimating charge MVP output",
        })?;
    if output_bytes as u128 > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: output_bytes as u128,
            limit: max_bytes,
        });
    }
    if let Some(executor) = local_aggregate {
        return executor.apply_into::<ADJOINT>(state, output);
    }
    if !ADJOINT
        && try_apply_contiguous_fixed_population_into(
            plan,
            layout,
            state,
            output,
            termwise_conserved,
        )?
    {
        return Ok(());
    }
    if !ADJOINT
        && try_apply_contiguous_fixed_population_pair_into(
            plan,
            layout,
            state,
            output,
            termwise_conserved,
        )?
    {
        return Ok(());
    }
    if let Some(executor) = factor_schedule {
        return executor.apply_into::<ADJOINT>(state, output);
    }
    let scratch_limit = max_bytes - output_bytes as u128;
    let axis_count = local_dimensions.len();
    let destination_entry_bytes = if layout.full_strides.is_some() {
        std::mem::size_of::<(usize, Complex64)>()
    } else {
        64usize
            .checked_add(axis_count.checked_mul(std::mem::size_of::<u64>()).ok_or(
                PauliError::Overflow {
                    context: "estimating charge MVP destination storage",
                },
            )?)
            .ok_or(PauliError::Overflow {
                context: "estimating charge MVP destination storage",
            })?
    };
    let scratch_bytes = axis_count
        .checked_mul(std::mem::size_of::<u64>())
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_add(terms.len().checked_mul(destination_entry_bytes)?))
        .and_then(|value| value.checked_add(32))
        .ok_or(PauliError::Overflow {
            context: "estimating charge MVP workspace",
        })?;
    check_bytes(1, scratch_bytes, scratch_limit)?;

    output.fill(Complex64::new(0.0, 0.0));
    let mut apply_stats = ChargeApplyStats {
        fallback_invocations: usize::from(!termwise_conserved),
        ..ChargeApplyStats::default()
    };
    let mut source = vec![0_u64; axis_count];
    let mut destination = vec![0_u64; axis_count];
    let mut remaining = vec![0_i128; plan.constraint_count()];
    let mut candidate_remaining = vec![0_i128; plan.constraint_count()];
    let execution_layout = plan.factorized_execution_layout();
    let mut factor_scratch = execution_layout.map(FactorizedLayout::scratch);
    let mut destinations =
        ChargeDestinationAccumulator::new(layout.full_strides.as_deref(), terms.len(), axis_count)?;
    let fermion_word_count = schedules
        .iter()
        .filter_map(|schedule| schedule.fermion.as_ref())
        .map(|descriptor| descriptor.required_occupied.len())
        .max()
        .unwrap_or(0);
    let mut fermion_words = vec![0_u64; fermion_word_count];

    for (column, state_value) in state.iter().enumerate() {
        let column_rank = u64::try_from(column).map_err(|_| PauliError::Overflow {
            context: "indexing charge-sector MVP source",
        })?;
        if let (Some(layout), Some(scratch)) = (execution_layout, factor_scratch.as_mut()) {
            if column == 0 {
                layout.unrank_with_scratch(column_rank, &mut source, scratch)?;
            } else if !layout.advance_public_cursor(&mut source, scratch)? {
                return Err(PauliError::InvalidSector {
                    context: "factorized source cursor ended before sector dimension",
                });
            }
        } else {
            plan.unrank_into_with_scratch(
                column_rank,
                &mut source,
                &mut remaining,
                &mut candidate_remaining,
            )?;
        }
        destinations.clear();
        if termwise_conserved {
            for schedule in schedules {
                let Some(value) = apply_compiled_charge_schedule(
                    schedule,
                    &source,
                    &mut destination,
                    local_dimensions,
                    qudit_dimension,
                    &mut fermion_words,
                )?
                else {
                    continue;
                };
                let row = if destination == source {
                    u64::try_from(column).map_err(|_| PauliError::Overflow {
                        context: "indexing charge-sector MVP destination",
                    })?
                } else {
                    apply_stats.ranker_invocations += 1;
                    if let (Some(layout), Some(scratch)) =
                        (execution_layout, factor_scratch.as_mut())
                    {
                        layout.rank_with_scratch(&destination, scratch)?
                    } else {
                        plan.rank_into(&destination, &mut remaining, &mut candidate_remaining)?
                    }
                };
                let row = usize::try_from(row).map_err(|_| PauliError::Overflow {
                    context: "converting charge-sector MVP destination",
                })?;
                if ADJOINT {
                    output[column] += value.conj() * state[row];
                } else {
                    output[row] += value * *state_value;
                }
            }
        } else {
            for schedule in schedules {
                let Some(value) = apply_compiled_charge_schedule(
                    schedule,
                    &source,
                    &mut destination,
                    local_dimensions,
                    qudit_dimension,
                    &mut fermion_words,
                )?
                else {
                    continue;
                };
                apply_stats.merge(destinations.insert(&destination, value)?);
            }
        }
        destinations.drain(&mut destination, local_dimensions, |destination, value| {
            if value.re == 0.0 && value.im == 0.0 {
                return Ok(());
            }
            apply_stats.ranker_invocations += 1;
            let ranked = if let (Some(layout), Some(scratch)) =
                (execution_layout, factor_scratch.as_mut())
            {
                layout.rank_with_scratch(destination, scratch)
            } else {
                plan.rank_into(destination, &mut remaining, &mut candidate_remaining)
            };
            let row = match ranked {
                Ok(row) => usize::try_from(row).map_err(|_| PauliError::Overflow {
                    context: "converting charge-sector MVP destination",
                })?,
                Err(PauliError::InvalidSector { .. }) => {
                    return Err(PauliError::InvalidSector {
                        context: "operator leaks outside the selected charge sector",
                    });
                }
                Err(error) => return Err(error),
            };
            if ADJOINT {
                output[column] += value.conj() * state[row];
            } else {
                output[row] += value * *state_value;
            }
            Ok(())
        })?;
    }
    layout.counters.publish(apply_stats);
    Ok(())
}
/// Select an execution representation from the sector's native proof.
///
/// Factor analysis is performed once while constructing `ChargeSectorPlan`.
/// No public charge spelling, Python-derived physical hint, or runtime retry
/// is allowed to influence this decision.
fn select_charge_lazy_kernel(sector: &ChargeSectorPlan) -> ChargeLazyKernel {
    let Some(layout) = sector.factorized_execution_layout() else {
        return ChargeLazyKernel::Generic;
    };
    // Validate the compact proof once at selection time.  This uses only
    // retained O(axis-count) metadata, and keeps the component representation
    // exercised even before the specialized traversal enters the hot path.
    let complete = layout.axis_components().len() == sector.local_dimensions().len()
        && layout.components().iter().all(|component| {
            !component.axes().is_empty()
                && component.index().dimension() > 0
                && match component.kind() {
                    FactorComponentKind::Cartesian | FactorComponentKind::FiniteAdditive => true,
                    FactorComponentKind::FixedPopulation { population } => {
                        population <= component.axes().len()
                    }
                }
        });
    if complete {
        ChargeLazyKernel::Factorized
    } else {
        ChargeLazyKernel::Generic
    }
}

/// Return the native representation diagnostic without constructing a lazy
/// owner. Direct eager compilation shares this selector but must not create a
/// `PreparedChargeMvpPlan` solely to read its variant.
pub fn charge_execution_variant(sector: &ChargeSectorPlan) -> &'static str {
    select_charge_lazy_kernel(sector).native_variant()
}

/// Build the single owner of lazy general-charge execution and resolve its
/// immutable native kernel exactly once.
pub fn prepare_charge_mvp_plan(
    sector: Arc<ChargeSectorPlan>,
    layout: PreparedChargeTransitionPlanLayout,
    terms: Box<[ChargeTransitionTerm]>,
    termwise_conserved: bool,
    _max_bytes: u128,
) -> Result<PreparedChargeMvpPlan, PauliError> {
    let local_aggregate = build_contiguous_local_aggregate_executor(&sector, &layout)?;
    let factor_schedule = if local_aggregate.is_none() {
        build_contiguous_factor_schedule_executor(&sector, &layout, termwise_conserved)?
    } else {
        None
    };
    let kernel = select_charge_lazy_kernel(&sector);
    Ok(PreparedChargeMvpPlan::new(
        sector,
        layout,
        terms,
        termwise_conserved,
        kernel,
        local_aggregate,
        factor_schedule,
    ))
}

pub struct PreparedChargeMvpPlan {
    sector: Arc<ChargeSectorPlan>,
    layout: PreparedChargeTransitionPlanLayout,
    terms: Box<[ChargeTransitionTerm]>,
    termwise_conserved: bool,
    kernel: ChargeLazyKernel,
    local_aggregate: Option<ContiguousLocalAggregateExecutor>,
    factor_schedule: Option<ContiguousFactorScheduleExecutor>,
}

impl PreparedChargeMvpPlan {
    /// Apply the conjugate transpose in the retained public charge basis.
    pub fn apply_adjoint(
        &self,
        state: &[Complex64],
        max_bytes: u128,
    ) -> Result<Vec<Complex64>, PauliError> {
        apply_charge_mvp_from_prepared_plan::<true>(
            &self.sector,
            &self.layout,
            &self.terms,
            self.local_aggregate.as_ref(),
            self.factor_schedule.as_ref(),
            state,
            self.termwise_conserved,
            max_bytes,
        )
    }

    fn new(
        sector: Arc<ChargeSectorPlan>,
        layout: PreparedChargeTransitionPlanLayout,
        terms: Box<[ChargeTransitionTerm]>,
        termwise_conserved: bool,
        kernel: ChargeLazyKernel,
        local_aggregate: Option<ContiguousLocalAggregateExecutor>,
        factor_schedule: Option<ContiguousFactorScheduleExecutor>,
    ) -> Self {
        Self {
            sector,
            layout,
            terms,
            termwise_conserved,
            kernel,
            local_aggregate,
            factor_schedule,
        }
    }

    pub fn dimension(&self) -> usize {
        self.sector.dimension()
    }

    pub fn estimated_bytes(&self) -> Result<u128, PauliError> {
        self.sector
            .estimated_bytes()
            .checked_add(estimate_charge_transition_terms_bytes(&self.terms)?)
            .and_then(|value| value.checked_add(self.layout.estimated_bytes()))
            .and_then(|value| value.checked_add(self.kernel.estimated_bytes()))
            .and_then(|value| {
                value.checked_add(
                    self.local_aggregate
                        .as_ref()
                        .map_or(0, ContiguousLocalAggregateExecutor::estimated_bytes),
                )
            })
            .and_then(|value| {
                value.checked_add(
                    self.factor_schedule
                        .as_ref()
                        .map_or(0, ContiguousFactorScheduleExecutor::estimated_bytes),
                )
            })
            .ok_or(PauliError::Overflow {
                context: "estimating prepared charge MVP plan",
            })
    }

    pub fn execution_family(&self) -> &'static str {
        match &self.kernel {
            ChargeLazyKernel::Generic => "generic",
            ChargeLazyKernel::Factorized => "factorized",
        }
    }

    pub fn native_variant(&self) -> &'static str {
        self.kernel.native_variant()
    }

    pub fn schedule_count(&self) -> usize {
        self.terms.len()
    }

    pub fn debug_counters(&self) -> Option<Vec<usize>> {
        self.kernel.debug_counters()
    }

    pub fn generic_debug_counters(&self) -> [usize; 8] {
        let (
            scalar_insertions,
            wide_insertions,
            capacity_growths,
            owned_keys,
            fallbacks,
            rankers,
            validations,
            destination_allocations,
        ) = self.layout.debug_counters();
        [
            scalar_insertions,
            wide_insertions,
            capacity_growths,
            owned_keys,
            fallbacks,
            rankers,
            validations,
            destination_allocations,
        ]
    }

    pub fn apply(
        &self,
        state: &[Complex64],
        max_bytes: u128,
    ) -> Result<Vec<Complex64>, PauliError> {
        apply_charge_mvp_from_prepared_plan::<false>(
            &self.sector,
            &self.layout,
            &self.terms,
            self.local_aggregate.as_ref(),
            self.factor_schedule.as_ref(),
            state,
            self.termwise_conserved,
            max_bytes,
        )
    }

    pub fn apply_into(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
        max_bytes: u128,
    ) -> Result<(), PauliError> {
        apply_charge_mvp_from_prepared_plan_into::<false>(
            &self.sector,
            &self.layout,
            &self.terms,
            self.local_aggregate.as_ref(),
            self.factor_schedule.as_ref(),
            state,
            output,
            self.termwise_conserved,
            max_bytes,
        )
    }

    pub fn compile_eager(&self, max_bytes: u128) -> Result<ChargeCsrResult, PauliError> {
        compile_charge_csr_from_prepared_plan(&self.sector, &self.layout, &self.terms, max_bytes)
    }

    pub fn compile_eager_with_chunk_capacity(
        &self,
        max_bytes: u128,
        chunk_capacity: usize,
    ) -> Result<ChargeCsrResult, PauliError> {
        compile_charge_csr_from_prepared_plan_with_chunk_capacity(
            &self.sector,
            &self.layout,
            &self.terms,
            max_bytes,
            chunk_capacity,
        )
    }
}
use super::*;
