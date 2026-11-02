// Structured row scheduling has a higher crossover than charge CSR on the
// measured workloads; keep small eager graphs on the serial path.
const STRUCTURED_CSR_PARALLEL_THRESHOLD: usize = 1 << 20;

impl StructuredEagerMvpPlan {
    /// Apply the conjugate transpose of the retained CSR graph.
    pub fn apply_adjoint(
        &self,
        state: &[Complex64],
        max_bytes: u128,
    ) -> Result<Vec<Complex64>, PauliError> {
        self.graph.apply_adjoint(state, max_bytes)
    }

    pub fn dimension(&self) -> usize {
        self.graph.dimension()
    }

    pub fn estimated_bytes(&self) -> u128 {
        self.estimated_bytes
    }

    pub fn estimated_construction_peak_bytes(&self) -> u128 {
        self.estimated_construction_peak_bytes
    }

    pub fn source_term_count(&self) -> usize {
        self.source_term_count
    }

    pub fn schedule_count(&self) -> usize {
        self.schedule_count
    }

    pub fn execution_family(&self) -> &'static str {
        self.execution_family.label()
    }

    pub fn native_variant(&self) -> &'static str {
        self.native_variant
    }

    pub fn retained_format(&self) -> &'static str {
        "csr"
    }

    pub fn transition_count(&self) -> usize {
        self.graph.transition_count()
    }

    pub fn csr(&self, max_bytes: u128) -> Result<crate::csr::CsrU64Arrays, PauliError> {
        self.graph.to_u64_arrays(max_bytes)
    }

    /// Move the retained CSR arrays to a native materialization boundary.
    pub fn into_csr(self) -> (usize, Vec<u64>, Vec<u64>, Vec<Complex64>) {
        let dimension = self.graph.dimension();
        let (indptr, columns, values) = self
            .graph
            .into_u64_arrays(u128::MAX)
            .expect("validated CSR graph is always readable");
        (dimension, indptr, columns, values)
    }

    pub fn apply(
        &self,
        state: &[Complex64],
        max_bytes: u128,
    ) -> Result<Vec<Complex64>, PauliError> {
        if state.len() != self.dimension() {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension(),
                actual: state.len(),
            });
        }
        let output_bytes = (self.dimension() as u128)
            .checked_mul(size_of::<Complex64>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating structured eager MVP output",
            })?;
        if output_bytes > max_bytes {
            return Err(PauliError::MemoryLimit {
                requested: output_bytes,
                limit: max_bytes,
            });
        }
        let mut output = vec![Complex64::default(); self.dimension()];
        self.apply_into(state, &mut output, max_bytes)?;
        Ok(output)
    }

    pub fn apply_into(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
        max_bytes: u128,
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
        let output_bytes = (self.dimension() as u128)
            .checked_mul(size_of::<Complex64>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating structured eager MVP output",
            })?;
        if output_bytes > max_bytes {
            return Err(PauliError::MemoryLimit {
                requested: output_bytes,
                limit: max_bytes,
            });
        }
        self.graph.apply_into(
            state,
            output,
            CsrApplyPolicy::parallel(STRUCTURED_CSR_PARALLEL_THRESHOLD),
        )
    }
}

pub(super) fn mixed_radix_strides(local_dimensions: &[usize]) -> Result<Vec<usize>, PauliError> {
    let mut strides = vec![1_usize; local_dimensions.len()];
    let mut stride = 1_usize;
    for position in (0..local_dimensions.len()).rev() {
        strides[position] = stride;
        stride = stride
            .checked_mul(local_dimensions[position])
            .ok_or(PauliError::Overflow {
                context: "computing structured mixed-radix strides",
            })?;
    }
    Ok(strides)
}

fn validate_mvp_operation(
    operation: &StructuredOperation,
    local_dimensions: &[usize],
    fermion_positions: &[usize],
) -> Result<usize, PauliError> {
    let dimension =
        local_dimensions
            .get(operation.axis)
            .copied()
            .ok_or(PauliError::InvalidIndex {
                context: "structured MVP operation axis",
            })?;
    match operation.kind {
        0 => {
            if dimension != 2 {
                return Err(PauliError::InvalidIndex {
                    context: "Pauli operation requires a two-level axis",
                });
            }
            if operation.p > 3 {
                return Err(PauliError::InvalidCode {
                    code: operation.p as u8,
                    index: operation.axis,
                });
            }
            Ok(0)
        }
        1 | 2 => Ok(dimension),
        3 => {
            if dimension != 2 {
                return Err(PauliError::InvalidIndex {
                    context: "Fermion operation requires a two-level axis",
                });
            }
            if !fermion_positions.is_empty() {
                let mode = usize::try_from(operation.p).map_err(|_| PauliError::Overflow {
                    context: "converting structured Fermion mode",
                })?;
                let position =
                    fermion_positions
                        .get(mode)
                        .copied()
                        .ok_or(PauliError::InvalidIndex {
                            context: "structured MVP Fermion mode",
                        })?;
                if position != operation.axis {
                    return Err(PauliError::InvalidIndex {
                        context: "structured MVP Fermion axis",
                    });
                }
            }
            if operation.q > 1 {
                return Err(PauliError::InvalidCode {
                    code: operation.q as u8,
                    index: operation.axis,
                });
            }
            Ok(0)
        }
        _ => Err(PauliError::InvalidCode {
            code: operation.kind,
            index: operation.axis,
        }),
    }
}

pub(super) fn compile_mvp_operations(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    fermion_positions: &[usize],
    strides: &[usize],
) -> Result<Vec<Vec<CompiledStructuredOperation>>, PauliError> {
    let mut compiled_terms = Vec::with_capacity(terms.len());
    for term in terms {
        let mut compiled = Vec::with_capacity(term.len());
        for operation in term {
            let table_len = validate_mvp_operation(operation, local_dimensions, fermion_positions)?;
            let p = usize::try_from(operation.p).map_err(|_| PauliError::Overflow {
                context: "converting structured operation parameter",
            })?;
            let q = usize::try_from(operation.q).map_err(|_| PauliError::Overflow {
                context: "converting structured operation parameter",
            })?;
            let mut local_table = Vec::new();
            if table_len != 0 {
                local_table = vec![Complex64::default(); table_len];
                match operation.kind {
                    1 => {
                        let creation = p;
                        let annihilation = q;
                        for (digit, value) in local_table.iter_mut().enumerate() {
                            if digit < annihilation {
                                continue;
                            }
                            let Some(destination) = digit
                                .checked_sub(annihilation)
                                .and_then(|value| value.checked_add(creation))
                            else {
                                continue;
                            };
                            if destination >= table_len {
                                continue;
                            }
                            let mut ladder_amplitude = 1.0;
                            for offset in 0..annihilation {
                                ladder_amplitude *= ((digit - offset) as f64).sqrt();
                            }
                            let remaining = digit - annihilation;
                            for offset in 0..creation {
                                ladder_amplitude *= ((remaining + offset + 1) as f64).sqrt();
                            }
                            *value = Complex64::new(ladder_amplitude, 0.0);
                        }
                    }
                    2 => {
                        for (digit, value) in local_table.iter_mut().enumerate() {
                            let b_digit = (q as u128 * digit as u128) % table_len as u128;
                            let angle =
                                2.0 * std::f64::consts::PI * b_digit as f64 / table_len as f64;
                            *value = Complex64::from_polar(1.0, angle);
                        }
                    }
                    _ => unreachable!("only Boson and Weyl operations have local tables"),
                }
            }
            compiled.push(CompiledStructuredOperation {
                axis: operation.axis,
                kind: operation.kind,
                p,
                q,
                stride: strides[operation.axis],
                dimension: local_dimensions[operation.axis],
                local_table,
            });
        }
        compiled_terms.push(compiled);
    }
    Ok(compiled_terms)
}

fn update_compiled_destination(
    destination: &mut usize,
    digit: usize,
    new_digit: usize,
    operation: &CompiledStructuredOperation,
) -> Result<(), PauliError> {
    if new_digit >= operation.dimension {
        return Err(PauliError::InvalidIndex {
            context: "structured operation destination",
        });
    }
    if new_digit >= digit {
        let delta =
            (new_digit - digit)
                .checked_mul(operation.stride)
                .ok_or(PauliError::Overflow {
                    context: "computing structured operation destination",
                })?;
        *destination = destination.checked_add(delta).ok_or(PauliError::Overflow {
            context: "computing structured operation destination",
        })?;
    } else {
        let delta =
            (digit - new_digit)
                .checked_mul(operation.stride)
                .ok_or(PauliError::Overflow {
                    context: "computing structured operation destination",
                })?;
        *destination = destination.checked_sub(delta).ok_or(PauliError::Overflow {
            context: "computing structured operation destination",
        })?;
    }
    Ok(())
}

fn compiled_digit(destination: usize, operation: &CompiledStructuredOperation) -> usize {
    (destination / operation.stride) % operation.dimension
}

fn apply_compiled_nonfermion_operation(
    operation: &CompiledStructuredOperation,
    destination: &mut usize,
    amplitude: &mut Complex64,
) -> Result<bool, PauliError> {
    let digit = compiled_digit(*destination, operation);
    match operation.kind {
        0 => match operation.p {
            0 => {}
            1 => update_compiled_destination(destination, digit, 1 - digit, operation)?,
            2 => {
                update_compiled_destination(destination, digit, 1 - digit, operation)?;
                *amplitude *= if digit == 0 {
                    Complex64::new(0.0, 1.0)
                } else {
                    Complex64::new(0.0, -1.0)
                };
            }
            3 => {
                if digit == 1 {
                    *amplitude = -*amplitude;
                }
            }
            _ => unreachable!("MVP Pauli operation was validated during compilation"),
        },
        1 => {
            let factor = operation.local_table[digit];
            if factor.re == 0.0 && factor.im == 0.0 {
                return Ok(false);
            }
            let creation = operation.p;
            let annihilation = operation.q;
            let new_digit = digit
                .checked_sub(annihilation)
                .and_then(|value| value.checked_add(creation))
                .ok_or(PauliError::Overflow {
                    context: "computing structured boson destination",
                })?;
            update_compiled_destination(destination, digit, new_digit, operation)?;
            *amplitude *= factor;
        }
        2 => {
            *amplitude *= operation.local_table[digit];
            let shift = operation.p % operation.dimension;
            let new_digit = if shift == 0 {
                digit
            } else if digit < operation.dimension - shift {
                digit + shift
            } else {
                digit - (operation.dimension - shift)
            };
            update_compiled_destination(destination, digit, new_digit, operation)?;
        }
        _ => unreachable!("MVP operation was validated during compilation"),
    }
    Ok(true)
}

#[cfg(test)]
pub(super) fn apply_compiled_structured_operation(
    operation: &CompiledStructuredOperation,
    local_dimensions: &[usize],
    strides: &[usize],
    fermion_positions: &[usize],
    destination: &mut usize,
    amplitude: &mut Complex64,
) -> Result<bool, PauliError> {
    if operation.kind != 3 {
        return apply_compiled_nonfermion_operation(operation, destination, amplitude);
    }
    let digit = compiled_digit(*destination, operation);
    let mode = operation.p;
    let parity = if fermion_positions.is_empty() {
        (0..operation.axis)
            .map(|axis| (*destination / strides[axis]) % local_dimensions[axis])
            .sum::<usize>()
            & 1
    } else {
        fermion_positions[..mode]
            .iter()
            .map(|&axis| (*destination / strides[axis]) % local_dimensions[axis])
            .sum::<usize>()
            & 1
    };
    if parity != 0 {
        *amplitude = -*amplitude;
    }
    match operation.q {
        0 if digit == 0 => Ok(false),
        0 => {
            update_compiled_destination(destination, digit, 0, operation)?;
            Ok(true)
        }
        1 if digit != 0 => Ok(false),
        1 => {
            update_compiled_destination(destination, digit, 1, operation)?;
            Ok(true)
        }
        _ => unreachable!("MVP Fermion operation was validated during compilation"),
    }
}

impl StructuredMvpPlan {
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn estimated_bytes(&self) -> u128 {
        self.estimated_bytes
    }

    pub fn source_term_count(&self) -> usize {
        self.source_term_count
    }

    pub fn schedule_count(&self) -> usize {
        self.terms.len()
    }

    pub fn execution_family(&self) -> &'static str {
        self.execution_family.label()
    }

    pub fn native_variant(&self) -> &'static str {
        match &self.inner {
            StructuredPlanInner::Pauli(_) => "pauli",
            StructuredPlanInner::Compiled => "compiled",
        }
    }

    pub fn retained_format(&self) -> &'static str {
        "compiled_actions"
    }

    pub fn estimated_construction_peak_bytes(&self) -> u128 {
        self.estimated_bytes
    }

    /// Return test/benchmark counters for the compiled Cartesian executor.
    /// The counters are deliberately diagnostic metadata; they do not affect
    /// the public numerical result.
    pub fn debug_counters(&self) -> (usize, usize, usize, usize) {
        self.counters.snapshot()
    }

    pub fn debug_metadata(&self) -> (usize, usize, usize) {
        (
            self.fermion_schedule_count,
            self.fermion_mask_schedule_count,
            self.descriptor_ineligible_schedule_count,
        )
    }

    /// Measure batched descriptor phases for a raw-Fermion Cartesian plan.
    ///
    /// The sample count is capped to keep this diagnostic bounded; it is not
    /// part of normal plan application or public execution semantics.
    pub fn fermion_mask_phase_profile(
        &self,
        requested_samples: usize,
    ) -> Result<Option<FermionMaskPhaseProfile>, PauliError> {
        let descriptors = self
            .fermion_masks
            .iter()
            .filter_map(Option::as_ref)
            .collect::<Vec<_>>();
        if descriptors.is_empty() || self.dimension == 0 {
            return Ok(None);
        }
        let sample_count = requested_samples.max(1).min(self.dimension).min(65_536);
        let descriptor_words = descriptors
            .iter()
            .map(|descriptor| descriptor.required_occupied.len())
            .collect::<Vec<_>>();
        let contiguous_source_layout = self.contiguous_fermion_source_mode_count.is_some();
        let (word_offsets, total_words, packed_source_unit_count) = if contiguous_source_layout {
            let words = descriptor_words[0];
            (
                vec![0; descriptor_words.len()],
                words
                    .checked_mul(sample_count)
                    .ok_or(PauliError::Overflow {
                        context: "sizing FermionMask profile words",
                    })?,
                sample_count,
            )
        } else {
            let mut word_offsets = Vec::with_capacity(descriptor_words.len());
            let mut total_words = 0usize;
            for &words in &descriptor_words {
                word_offsets.push(total_words);
                total_words =
                    total_words
                        .checked_add(words.checked_mul(sample_count).ok_or(
                            PauliError::Overflow {
                                context: "sizing FermionMask profile words",
                            },
                        )?)
                        .ok_or(PauliError::Overflow {
                            context: "sizing FermionMask profile words",
                        })?;
            }
            (
                word_offsets,
                total_words,
                descriptors.len().saturating_mul(sample_count),
            )
        };
        let mut packed = vec![0_u64; total_words];
        let extraction_start = Instant::now();
        if contiguous_source_layout {
            let words = descriptor_words[0];
            for source in 0..sample_count {
                let start = source * words;
                descriptors[0]
                    .pack_contiguous_binary_index(source, &mut packed[start..start + words])?;
            }
        } else {
            for (descriptor_index, descriptor) in descriptors.iter().enumerate() {
                let words = descriptor_words[descriptor_index];
                let offset = word_offsets[descriptor_index];
                for source in 0..sample_count {
                    let start = offset + source * words;
                    descriptor.pack_index(
                        source,
                        &self.strides,
                        &mut packed[start..start + words],
                    )?;
                }
            }
        }
        let packed_extraction_nanos = extraction_start.elapsed().as_nanos();

        let parity_start = Instant::now();
        for (descriptor_index, descriptor) in descriptors.iter().enumerate() {
            let words = descriptor_words[descriptor_index];
            let offset = word_offsets[descriptor_index];
            for source in 0..sample_count {
                let start = offset + source * words;
                if contiguous_source_layout {
                    black_box(
                        descriptor
                            .amplitude_from_single_word(packed[start], Complex64::new(1.0, -0.25)),
                    );
                } else {
                    black_box(descriptor.amplitude_from_words(
                        &packed[start..start + words],
                        Complex64::new(1.0, -0.25),
                    ));
                }
            }
        }
        let parity_nanos = parity_start.elapsed().as_nanos();

        let index_start = Instant::now();
        for descriptor in &descriptors {
            for source in 0..sample_count {
                if contiguous_source_layout {
                    black_box(descriptor.apply_flip_to_contiguous_binary_index(source)?);
                } else {
                    black_box(descriptor.apply_flip_to_index(source, &self.strides)?);
                }
            }
        }
        let index_update_nanos = index_start.elapsed().as_nanos();

        let state = vec![Complex64::new(0.5, -0.25); sample_count];
        let mut output = vec![Complex64::default(); sample_count];
        let input_output_start = Instant::now();
        for descriptor_index in 0..descriptors.len() {
            for (source, &input) in state.iter().enumerate() {
                let destination = (source + descriptor_index) % sample_count;
                output[destination] += input * Complex64::new(1.0, -0.25);
            }
        }
        black_box(&output);
        let input_output_nanos = input_output_start.elapsed().as_nanos();
        Ok(Some(FermionMaskPhaseProfile {
            descriptor_count: descriptors.len(),
            sample_count,
            packed_source_unit_count,
            schedule_member_evaluations: descriptors.len().saturating_mul(sample_count),
            packed_extraction_nanos,
            parity_nanos,
            index_update_nanos,
            input_output_nanos,
        }))
    }

    pub fn transition_count(&self) -> Option<usize> {
        None
    }

    pub fn apply(
        &self,
        state: &[Complex64],
        max_bytes: u128,
    ) -> Result<Vec<Complex64>, PauliError> {
        self.apply_direction::<false>(state, max_bytes)
    }

    /// Apply the conjugate transpose while retaining the compiled schedules.
    pub fn apply_adjoint(
        &self,
        state: &[Complex64],
        max_bytes: u128,
    ) -> Result<Vec<Complex64>, PauliError> {
        self.apply_direction::<true>(state, max_bytes)
    }

    fn apply_direction<const ADJOINT: bool>(
        &self,
        state: &[Complex64],
        max_bytes: u128,
    ) -> Result<Vec<Complex64>, PauliError> {
        if state.len() != self.dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension,
                actual: state.len(),
            });
        }
        let output_bytes = (self.dimension as u128)
            .checked_mul(size_of::<Complex64>() as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating structured MVP output",
            })?;
        if output_bytes > max_bytes {
            return Err(PauliError::MemoryLimit {
                requested: output_bytes,
                limit: max_bytes,
            });
        }
        let mut output = vec![Complex64::default(); self.dimension];
        self.apply_into_direction::<ADJOINT>(state, &mut output, max_bytes)?;
        Ok(output)
    }

    #[inline]
    fn contiguous_fermion_destination_value<const ADJOINT: bool>(
        &self,
        destination: usize,
        state: &[Complex64],
    ) -> Complex64 {
        let mut value = Complex64::default();
        for group in &self.groups {
            let first_descriptor = self.fermion_masks[group.members[0]]
                .as_ref()
                .expect("contiguous Fermion group has a descriptor");
            let action = first_descriptor
                .contiguous_public_action()
                .expect("contiguous Fermion group has a public-index action");
            let source = action.source_for_destination(destination);
            let mut aggregate = Complex64::default();
            for &term_index in &group.members {
                let descriptor = self.fermion_masks[term_index]
                    .as_ref()
                    .expect("contiguous Fermion member has a descriptor");
                let action = descriptor
                    .contiguous_public_action()
                    .expect("contiguous Fermion member has a public-index action");
                let forward_source = if ADJOINT { destination } else { source };
                let Some(amplitude) =
                    action.amplitude(forward_source, self.coefficients[term_index])
                else {
                    continue;
                };
                aggregate += amplitude;
            }
            if ADJOINT {
                aggregate = aggregate.conj();
            }
            if aggregate.re != 0.0 || aggregate.im != 0.0 {
                value += aggregate * state[source];
            }
        }
        value
    }

    fn apply_contiguous_fermion_gather_into<const ADJOINT: bool>(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), PauliError> {
        let work = self.terms.len().saturating_mul(self.dimension);
        if work >= 1 << 16 {
            output
                .par_iter_mut()
                .enumerate()
                .for_each(|(destination, output_value)| {
                    *output_value =
                        self.contiguous_fermion_destination_value::<ADJOINT>(destination, state);
                });
        } else {
            for (destination, output_value) in output.iter_mut().enumerate() {
                *output_value =
                    self.contiguous_fermion_destination_value::<ADJOINT>(destination, state);
            }
        }
        self.counters.publish(StructuredApplyStats {
            schedule_group_evaluations: self.dimension.saturating_mul(self.groups.len()),
            fermion_mask_applications: self.dimension.saturating_mul(self.terms.len()),
            ..StructuredApplyStats::default()
        });
        Ok(())
    }

    /// Apply the compact plan into caller-owned output storage.
    pub fn apply_into(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
        _max_bytes: u128,
    ) -> Result<(), PauliError> {
        self.apply_into_direction::<false>(state, output, _max_bytes)
    }

    fn apply_into_direction<const ADJOINT: bool>(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
        _max_bytes: u128,
    ) -> Result<(), PauliError> {
        if state.len() != self.dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension,
                actual: state.len(),
            });
        }
        if output.len() != self.dimension {
            return Err(PauliError::InvalidStructureLength {
                expected: self.dimension,
                actual: output.len(),
            });
        }
        if let StructuredPlanInner::Pauli(plan) = &self.inner {
            return plan.apply_into_direction::<ADJOINT>(state, output, _max_bytes);
        }
        output.fill(Complex64::default());
        if self.contiguous_fermion_source_mode_count.is_some() {
            return self.apply_contiguous_fermion_gather_into::<ADJOINT>(state, output);
        }
        let fermion_word_count = self
            .fermion_masks
            .iter()
            .filter_map(Option::as_ref)
            .map(|descriptor| descriptor.required_occupied.len())
            .max()
            .unwrap_or(0);
        let mut fermion_mask_scratch = if self.contiguous_fermion_prefix.is_some() {
            Vec::new()
        } else {
            vec![0_u64; fermion_word_count]
        };
        let mut member_context = CompiledScheduleApplyContext {
            strides: &self.strides,
            fermion_mask_scratch: &mut fermion_mask_scratch,
            stats: StructuredApplyStats::default(),
            contiguous_fermion_prefix: self
                .contiguous_fermion_prefix
                .map(|prefix| prefix.suffix_dimension),
        };
        for group in &self.groups {
            let suffix_stride = group.suffix_stride;
            for block_start in (0..self.dimension).step_by(suffix_stride) {
                member_context.stats.schedule_group_evaluations += 1;
                let mut destination = None;
                let mut aggregate = Complex64::default();
                for &term_index in &group.members {
                    let contribution = apply_compiled_schedule_member(
                        &self.terms[term_index],
                        self.fermion_masks[term_index].as_ref(),
                        self.net_actions[term_index].as_deref(),
                        self.coefficients[term_index],
                        block_start,
                        &mut member_context,
                    )?;
                    let Some((member_destination, member_amplitude)) = contribution else {
                        continue;
                    };
                    if let Some(expected_destination) = destination {
                        if expected_destination != member_destination {
                            // This can only occur for a non-injective fallback
                            // schedule. Keep its members semantically separate
                            // rather than applying a wrong shared segment.
                            for offset in 0..suffix_stride {
                                if ADJOINT {
                                    output[block_start + offset] += member_amplitude.conj()
                                        * state[member_destination + offset];
                                } else {
                                    output[member_destination + offset] +=
                                        member_amplitude * state[block_start + offset];
                                }
                            }
                            continue;
                        }
                    } else {
                        destination = Some(member_destination);
                    }
                    aggregate += member_amplitude;
                }
                let Some(destination) = destination else {
                    continue;
                };
                if aggregate.re == 0.0 && aggregate.im == 0.0 {
                    continue;
                }
                for offset in 0..suffix_stride {
                    if ADJOINT {
                        output[block_start + offset] +=
                            aggregate.conj() * state[destination + offset];
                    } else {
                        output[destination + offset] += aggregate * state[block_start + offset];
                    }
                }
                if suffix_stride > 1 {
                    member_context.stats.segment_applications += 1;
                }
            }
        }
        self.counters.publish(member_context.stats);
        Ok(())
    }
}

fn apply_compiled_nonfermion_fallback(
    term: &[CompiledStructuredOperation],
    destination: &mut usize,
    amplitude: &mut Complex64,
) -> Result<bool, PauliError> {
    debug_assert!(term.iter().all(|operation| operation.kind != 3));
    for operation in term.iter().filter(|operation| operation.kind != 3) {
        if !apply_compiled_nonfermion_operation(operation, destination, amplitude)? {
            return Ok(false);
        }
    }
    Ok(true)
}

struct CompiledScheduleApplyContext<'a> {
    strides: &'a [usize],
    fermion_mask_scratch: &'a mut [u64],
    stats: StructuredApplyStats,
    contiguous_fermion_prefix: Option<usize>,
}

fn apply_compiled_schedule_member(
    term: &[CompiledStructuredOperation],
    descriptor: Option<&FermionMaskDescriptor>,
    actions: Option<&[CompiledNetAction]>,
    coefficient: Complex64,
    source: usize,
    context: &mut CompiledScheduleApplyContext<'_>,
) -> Result<Option<(usize, Complex64)>, PauliError> {
    let (mut destination, mut amplitude) = if let Some(descriptor) = descriptor {
        context.stats.fermion_mask_applications += 1;
        if let Some(suffix_dimension) = context.contiguous_fermion_prefix {
            let fermion_source = source / suffix_dimension;
            let action = descriptor
                .contiguous_public_action()
                .expect("contiguous Fermion prefix has a public-index action");
            let Some(amplitude) = action.amplitude(fermion_source, coefficient) else {
                return Ok(None);
            };
            // XOR by the flip mask is an involution, so the action's
            // destination-to-source operation also yields the destination
            // from a forward source index.
            let fermion_destination = action.source_for_destination(fermion_source);
            let spectator = source % suffix_dimension;
            let destination = fermion_destination
                .checked_mul(suffix_dimension)
                .and_then(|value| value.checked_add(spectator))
                .ok_or(PauliError::Overflow {
                    context: "computing contiguous Fermion prefix destination",
                })?;
            (destination, amplitude)
        } else {
            let words = descriptor.required_occupied.len();
            descriptor.pack_index(
                source,
                context.strides,
                &mut context.fermion_mask_scratch[..words],
            )?;
            let Some(amplitude) = descriptor
                .amplitude_from_words(&context.fermion_mask_scratch[..words], coefficient)
            else {
                return Ok(None);
            };
            (
                descriptor.apply_flip_to_index(source, context.strides)?,
                amplitude,
            )
        }
    } else {
        (source, coefficient)
    };
    let valid = if let Some(actions) = actions {
        let (new_destination, new_amplitude, valid) =
            apply_compiled_net_forward(actions, destination, amplitude)?;
        destination = new_destination;
        amplitude = new_amplitude;
        valid
    } else {
        // A supported non-Fermion local action is normally represented by a
        // compiled net action.  Keep this narrow fallback for a non-injective
        // boundary schedule, but never re-enter the legacy Fermion sign
        // interpreter: FermionMask has already handled the Fermion factor.
        context.stats.nonfermion_fallback_invocations += 1;
        apply_compiled_nonfermion_fallback(term, &mut destination, &mut amplitude)?
    };
    if valid {
        Ok(Some((destination, amplitude)))
    } else {
        Ok(None)
    }
}

/// Build a compact reusable matrix-free plan without enumerating transitions.
pub fn structured_mvp_plan(
    local_dimensions: Vec<usize>,
    terms: Vec<Vec<StructuredOperation>>,
    coefficients: Vec<Complex64>,
    max_bytes: u128,
) -> Result<StructuredMvpPlan, PauliError> {
    structured_mvp_plan_with_fermion_positions(
        local_dimensions,
        terms,
        coefficients,
        &[],
        max_bytes,
    )
}

/// Build a compact MVP plan with an explicit public-axis map for raw Fermion
/// actions. The map is indexed by Fermion mode and stores its public axis.
pub fn structured_mvp_plan_with_fermion_positions(
    local_dimensions: Vec<usize>,
    terms: Vec<Vec<StructuredOperation>>,
    coefficients: Vec<Complex64>,
    fermion_positions: &[usize],
    max_bytes: u128,
) -> Result<StructuredMvpPlan, PauliError> {
    validate_structured_inputs(&local_dimensions, &terms, &coefficients)?;
    let source_term_count = terms.len();
    let (terms, coefficients) = group_structured_terms(&terms, &coefficients, &local_dimensions)?;
    let dimension = mixed_radix_dimension(&local_dimensions)?;
    let strides = mixed_radix_strides(&local_dimensions)?;
    let operation_count: usize = terms.iter().try_fold(0usize, |count, term| {
        count.checked_add(term.len()).ok_or(PauliError::Overflow {
            context: "estimating structured MVP plan",
        })
    })?;
    let table_entries = terms.iter().try_fold(0usize, |count, term| {
        term.iter().try_fold(count, |count, operation| {
            let table_len =
                validate_mvp_operation(operation, &local_dimensions, fermion_positions)?;
            count.checked_add(table_len).ok_or(PauliError::Overflow {
                context: "estimating structured MVP local tables",
            })
        })
    })?;
    let estimated_bytes = (local_dimensions.len() as u128)
        .checked_mul(size_of::<usize>() as u128)
        .and_then(|value| {
            value.checked_add((strides.len() as u128).checked_mul(size_of::<usize>() as u128)?)
        })
        .and_then(|value| {
            value.checked_add(
                (terms.len() as u128)
                    .checked_mul(size_of::<Vec<CompiledStructuredOperation>>() as u128)?,
            )
        })
        .and_then(|value| {
            value.checked_add(
                (operation_count as u128)
                    .checked_mul(size_of::<CompiledStructuredOperation>() as u128)?,
            )
        })
        .and_then(|value| {
            value.checked_add((table_entries as u128).checked_mul(size_of::<Complex64>() as u128)?)
        })
        .and_then(|value| {
            value.checked_add(
                (coefficients.len() as u128).checked_mul(size_of::<Complex64>() as u128)?,
            )
        })
        .and_then(|value| value.checked_add(64))
        .and_then(|value| {
            value.checked_add(
                (fermion_positions.len() as u128).checked_mul(size_of::<usize>() as u128)?,
            )
        })
        .ok_or(PauliError::Overflow {
            context: "estimating structured MVP plan",
        })?;
    if estimated_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: estimated_bytes,
            limit: max_bytes,
        });
    }
    let compiled_terms =
        compile_mvp_operations(&local_dimensions, &terms, fermion_positions, &strides)?;
    let fermion_masks = compiled_terms
        .iter()
        .map(|term| compile_fermion_mask_descriptor(term, fermion_positions))
        .collect::<Vec<_>>();
    let fermion_schedule_count = compiled_terms
        .iter()
        .filter(|term| term.iter().any(|operation| operation.kind == 3))
        .count();
    let fermion_mask_schedule_count = fermion_masks.iter().filter(|mask| mask.is_some()).count();
    let descriptor_ineligible_schedule_count =
        fermion_schedule_count.saturating_sub(fermion_mask_schedule_count);
    let net_actions = terms
        .iter()
        .map(|term| compile_net_actions(term, &local_dimensions, &strides))
        .collect::<Result<Vec<_>, _>>()?;
    let groups = compile_schedule_groups(
        &compiled_terms,
        &net_actions,
        &fermion_masks,
        &strides,
        dimension,
    )?;
    let contiguous_fermion_source_mode_count = fermion_masks
        .iter()
        .find_map(Option::as_ref)
        .and_then(|descriptor| {
            (local_dimensions
                .iter()
                .all(|&local_dimension| local_dimension == 2)
                && fermion_masks.iter().all(Option::is_some)
                && net_actions
                    .iter()
                    .all(|actions| matches!(actions.as_deref(), Some([])))
                && groups.iter().all(|group| group.suffix_stride == 1)
                && descriptor.mode_axes.len() == local_dimensions.len()
                && fermion_masks.iter().all(|mask| {
                    mask.as_ref()
                        .and_then(FermionMaskDescriptor::contiguous_public_action)
                        .is_some()
                }))
            .then_some(descriptor.mode_axes.len())
        });
    let contiguous_fermion_prefix =
        fermion_masks
            .iter()
            .find_map(Option::as_ref)
            .and_then(|descriptor| {
                let mode_count = fermion_positions.len();
                (mode_count > 0
                    && mode_count <= local_dimensions.len()
                    && local_dimensions[..mode_count]
                        .iter()
                        .all(|&dimension| dimension == 2)
                    && fermion_positions
                        .iter()
                        .enumerate()
                        .all(|(mode, &axis)| mode == axis)
                    && descriptor.mode_axes.len() == mode_count
                    && fermion_masks.iter().filter_map(Option::as_ref).all(|mask| {
                        mask.mode_axes == descriptor.mode_axes
                            && mask.contiguous_public_action().is_some()
                    }))
                .then_some(ContiguousFermionPrefix {
                    suffix_dimension: strides[mode_count - 1],
                })
            });
    let net_action_bytes = net_actions.iter().try_fold(0_u128, |total, actions| {
        let action_bytes =
            actions
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .try_fold(0_u128, |subtotal, action| {
                    let table_bytes = (action.dimension as u128)
                        .checked_mul((2 * size_of::<usize>() + size_of::<Complex64>()) as u128)
                        .ok_or(PauliError::Overflow {
                            context: "estimating structured net-action tables",
                        })?;
                    subtotal
                        .checked_add(size_of::<CompiledNetAction>() as u128)
                        .and_then(|value| value.checked_add(table_bytes))
                        .ok_or(PauliError::Overflow {
                            context: "estimating structured net-action tables",
                        })
                })?;
        total.checked_add(action_bytes).ok_or(PauliError::Overflow {
            context: "estimating structured net-action tables",
        })
    })?;
    let fermion_mask_bytes = fermion_masks.iter().try_fold(0_u128, |total, descriptor| {
        let Some(descriptor) = descriptor.as_ref() else {
            return Ok(total);
        };
        let words = descriptor.required_occupied.len();
        let payload = words
            .checked_mul(4)
            .and_then(|value| value.checked_mul(size_of::<u64>()))
            .and_then(|value| {
                descriptor
                    .mode_axes
                    .len()
                    .checked_mul(size_of::<usize>())
                    .and_then(|axes_bytes| value.checked_add(axes_bytes))
            })
            .and_then(|value| value.checked_add(size_of::<FermionMaskDescriptor>()))
            .ok_or(PauliError::Overflow {
                context: "estimating FermionMask descriptors",
            })? as u128;
        total.checked_add(payload).ok_or(PauliError::Overflow {
            context: "estimating FermionMask descriptors",
        })
    })?;
    let estimated_bytes = estimated_bytes
        .checked_add(net_action_bytes)
        .and_then(|value| value.checked_add(fermion_mask_bytes))
        .and_then(|value| {
            value.checked_add(
                groups
                    .iter()
                    .map(|group| {
                        size_of::<StructuredScheduleGroup>()
                            + group.members.len() * size_of::<usize>()
                    })
                    .sum::<usize>() as u128,
            )
        })
        .ok_or(PauliError::Overflow {
            context: "estimating structured MVP net actions",
        })?;
    if estimated_bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: estimated_bytes,
            limit: max_bytes,
        });
    }
    let packed_pauli = compile_packed_pauli_plan(&local_dimensions, &terms, &coefficients)?;
    let (execution_family, inner) = if let Some(plan) = packed_pauli {
        (
            StructuredExecutionFamily::Pauli,
            StructuredPlanInner::Pauli(plan),
        )
    } else {
        (
            classify_structured_execution(
                &local_dimensions,
                &compiled_terms,
                &fermion_masks,
                &net_actions,
            ),
            StructuredPlanInner::Compiled,
        )
    };
    Ok(StructuredMvpPlan {
        dimension,
        strides,
        terms: compiled_terms,
        net_actions,
        fermion_masks,
        groups,
        coefficients,
        source_term_count,
        estimated_bytes,
        execution_family,
        inner,
        counters: Arc::new(StructuredApplyCounters::default()),
        fermion_schedule_count,
        fermion_mask_schedule_count,
        descriptor_ineligible_schedule_count,
        contiguous_fermion_source_mode_count,
        contiguous_fermion_prefix,
    })
}
use super::*;
