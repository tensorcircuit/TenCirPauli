//! Pure Rust occupation-encoding plans and batched Clifford transforms.

use crate::structured::{validate_boson_blocks, HybridBatch, HybridCanonicalResult, HybridLayout};
use crate::{Complex64, PauliError, PauliOperator, PauliPhase, PauliWord};
use rustc_hash::FxHashMap;

const MAPPING_PLAN_FIXED_BYTES: u128 = 256;
const MAPPING_TERM_BYTES: u128 = 16;

fn mapping_plan_native_bytes(n_modes: usize, cnot_count: usize) -> Result<u128, PauliError> {
    let matrix_entries =
        (n_modes as u128)
            .checked_mul(n_modes as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating fermion mapping plan",
            })?;
    let packed_word_count = n_modes.checked_add(63).ok_or(PauliError::Overflow {
        context: "estimating packed fermion mapping plan",
    })? / 64;
    let packed_entries = (n_modes as u128)
        .checked_mul(packed_word_count as u128)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(8))
        .ok_or(PauliError::Overflow {
            context: "estimating packed fermion mapping plan",
        })?;
    matrix_entries
        .checked_mul(2)
        .and_then(|value| value.checked_add(packed_entries))
        .and_then(|value| value.checked_add((cnot_count as u128).checked_mul(16)?))
        .and_then(|value| value.checked_add(MAPPING_PLAN_FIXED_BYTES))
        .ok_or(PauliError::Overflow {
            context: "estimating fermion mapping plan",
        })
}

type HybridMappingKey = (
    bool,
    Vec<u32>,
    Vec<u32>,
    bool,
    Vec<(u32, u32, u32)>,
    Vec<u8>,
    bool,
    Vec<u8>,
    bool,
    Vec<(u32, u32, u32)>,
);

#[derive(Clone, Debug)]
struct PackedBinaryMatrix {
    rows: Vec<Vec<u64>>,
}

impl PackedBinaryMatrix {
    fn from_rows(rows: Vec<Vec<u8>>, transpose: bool) -> Result<Self, PauliError> {
        let n = rows.len();
        if rows.iter().any(|row| row.len() != n) {
            return Err(PauliError::InvalidStructureLength {
                expected: n,
                actual: rows.iter().find(|row| row.len() != n).map_or(0, Vec::len),
            });
        }
        let word_count = n.checked_add(63).ok_or(PauliError::Overflow {
            context: "allocating packed fermion mapping transform",
        })? / 64;
        let mut packed = vec![vec![0_u64; word_count]; n];
        for output in 0..n {
            for input in 0..n {
                let value = if transpose {
                    rows[input][output]
                } else {
                    rows[output][input]
                };
                if value != 0 {
                    packed[output][input / 64] |= 1_u64 << (input % 64);
                }
            }
        }
        Ok(Self { rows: packed })
    }

    fn transform(&self, input: &[u64]) -> Vec<u64> {
        let mut output = vec![0_u64; self.rows.len().div_ceil(64)];
        for (index, row) in self.rows.iter().enumerate() {
            let parity = row.iter().zip(input).fold(0_u32, |value, (&left, &right)| {
                value ^ ((left & right).count_ones() & 1)
            });
            if parity & 1 != 0 {
                output[index / 64] |= 1_u64 << (index % 64);
            }
        }
        output
    }
}

/// A deterministic linear-reversible occupation encoding.
pub struct MappingPlan {
    n_modes: usize,
    encoding: Vec<Vec<u8>>,
    inverse_encoding: Vec<Vec<u8>>,
    cnot_operations: Vec<(usize, usize)>,
    x_transform: PackedBinaryMatrix,
    z_transform: PackedBinaryMatrix,
    estimated_bytes: u128,
}

impl MappingPlan {
    /// Return the number of fermion modes and mapped qubits.
    pub fn n_modes(&self) -> usize {
        self.n_modes
    }

    /// Return the binary occupation encoding matrix.
    pub fn encoding(&self) -> &[Vec<u8>] {
        &self.encoding
    }

    /// Return the inverse binary occupation encoding matrix.
    pub fn inverse_encoding(&self) -> &[Vec<u8>] {
        &self.inverse_encoding
    }

    /// Return the canonical CNOT provenance of the encoding.
    pub fn cnot_operations(&self) -> &[(usize, usize)] {
        &self.cnot_operations
    }

    /// Return the best-effort plan-size estimate.
    pub fn estimated_bytes(&self) -> u128 {
        self.estimated_bytes
    }

    /// Transform and aggregate a complete batch of Pauli words.
    pub fn map_pauli_terms(
        &self,
        structures: &[Vec<u8>],
        coefficients: &[Complex64],
        max_bytes: u128,
    ) -> Result<PauliOperator, PauliError> {
        if structures.len() != coefficients.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: structures.len(),
                actual: coefficients.len(),
            });
        }
        let requested = (structures.len().max(1) as u128)
            .checked_mul(self.n_modes.checked_add(4).ok_or(PauliError::Overflow {
                context: "estimating mapped Pauli output",
            })? as u128)
            .and_then(|value| value.checked_mul(MAPPING_TERM_BYTES))
            .ok_or(PauliError::Overflow {
                context: "estimating mapped Pauli output",
            })?;
        if requested > max_bytes {
            return Err(PauliError::MemoryLimit {
                requested,
                limit: max_bytes,
            });
        }

        let mut transformed = Vec::with_capacity(structures.len());
        let mut transformed_coefficients = Vec::with_capacity(coefficients.len());
        for (index, (structure, &coefficient)) in structures.iter().zip(coefficients).enumerate() {
            if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
                return Err(PauliError::NonFiniteCoefficient { index });
            }
            let (word, phase) = self.transform_codes(structure)?;
            transformed.push(word);
            transformed_coefficients.push(coefficient * phase);
        }
        PauliOperator::from_terms(self.n_modes, &transformed, &transformed_coefficients)
    }

    /// Transform an already-owned Pauli operator without exporting its terms
    /// through the Python boundary.
    pub fn map_pauli_operator(
        &self,
        operator: &PauliOperator,
        max_bytes: u128,
    ) -> Result<PauliOperator, PauliError> {
        if operator.nqubits() != self.n_modes {
            return Err(PauliError::IncompatibleQubitCounts {
                left: operator.nqubits(),
                right: self.n_modes,
            });
        }
        let requested = (operator.terms().len().max(1) as u128)
            .checked_mul((self.n_modes + 4) as u128)
            .and_then(|value| value.checked_mul(MAPPING_TERM_BYTES))
            .ok_or(PauliError::Overflow {
                context: "estimating mapped Pauli output",
            })?;
        if requested > max_bytes {
            return Err(PauliError::MemoryLimit {
                requested,
                limit: max_bytes,
            });
        }
        let mut structures = Vec::with_capacity(operator.terms().len());
        let mut coefficients = Vec::with_capacity(operator.terms().len());
        for (index, term) in operator.terms().iter().enumerate() {
            let (structure, phase) = self.transform_codes(&term.word.codes())?;
            let coefficient = term.coefficient * phase;
            if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
                return Err(PauliError::NonFiniteCoefficient { index });
            }
            structures.push(structure);
            coefficients.push(coefficient);
        }
        PauliOperator::from_terms(self.n_modes, &structures, &coefficients)
    }

    /// Transform only the leading fermion-axis prefix and retain trailing
    /// mixed-domain axes in their original order.
    pub fn map_pauli_operator_prefix(
        &self,
        operator: &PauliOperator,
        prefix_length: usize,
        max_bytes: u128,
    ) -> Result<PauliOperator, PauliError> {
        if prefix_length != self.n_modes || operator.nqubits() < prefix_length {
            return Err(PauliError::InvalidStructureLength {
                expected: self.n_modes,
                actual: prefix_length,
            });
        }
        let requested = (operator.terms().len().max(1) as u128)
            .checked_mul((operator.nqubits() + 4) as u128)
            .and_then(|value| value.checked_mul(MAPPING_TERM_BYTES))
            .ok_or(PauliError::Overflow {
                context: "estimating mapped Pauli output",
            })?;
        if requested > max_bytes {
            return Err(PauliError::MemoryLimit {
                requested,
                limit: max_bytes,
            });
        }
        let mut structures = Vec::with_capacity(operator.terms().len());
        let mut coefficients = Vec::with_capacity(operator.terms().len());
        for (index, term) in operator.terms().iter().enumerate() {
            let codes = term.word.codes();
            let (prefix, phase) = self.transform_codes(&codes[..prefix_length])?;
            let mut structure = prefix;
            structure.extend_from_slice(&codes[prefix_length..]);
            let coefficient = term.coefficient * phase;
            if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
                return Err(PauliError::NonFiniteCoefficient { index });
            }
            structures.push(structure);
            coefficients.push(coefficient);
        }
        PauliOperator::from_terms(operator.nqubits(), &structures, &coefficients)
    }

    fn transform_codes(&self, codes: &[u8]) -> Result<(Vec<u8>, Complex64), PauliError> {
        let word = PauliWord::from_codes(self.n_modes, codes)?;
        let x_words = self.x_transform.transform(word.x_words());
        let z_words = self.z_transform.transform(word.z_words());
        let transformed = PauliWord::from_words(self.n_modes, x_words, z_words)?;
        let input_exponent = word
            .x_words()
            .iter()
            .zip(word.z_words())
            .map(|(&x, &z)| x.count_ones() + z.count_ones() - (x ^ z).count_ones())
            .sum::<u32>()
            / 2;
        let output_exponent = transformed
            .x_words()
            .iter()
            .zip(transformed.z_words())
            .map(|(&x, &z)| x.count_ones() + z.count_ones() - (x ^ z).count_ones())
            .sum::<u32>()
            / 2;
        let phase_exponent = ((input_exponent as i32 - output_exponent as i32).rem_euclid(4)) as u8;
        Ok((
            transformed.codes(),
            PauliPhase::from_exponent(phase_exponent).as_complex(),
        ))
    }

    /// Map canonical Majorana words directly to Pauli words.
    ///
    /// Every input word remains one word throughout this routine.  The
    /// Jordan--Wigner image of each Majorana generator is a single Pauli word,
    /// so multiplying those images only accumulates one packed symplectic
    /// support and one exact four-valued phase.  No fermion expansion is
    /// materialized.
    pub fn map_majorana_terms(
        &self,
        indices: &[Vec<u64>],
        coefficients: &[Complex64],
        max_bytes: u128,
    ) -> Result<PauliOperator, PauliError> {
        if indices.len() != coefficients.len() {
            return Err(PauliError::InvalidStructureLength {
                expected: indices.len(),
                actual: coefficients.len(),
            });
        }
        let word_count = self.n_modes.checked_add(63).ok_or(PauliError::Overflow {
            context: "estimating direct Majorana mapping",
        })? / 64;
        let requested = (indices.len().max(1) as u128)
            .checked_mul((word_count.max(1) as u128).checked_mul(16).ok_or(
                PauliError::Overflow {
                    context: "estimating direct Majorana mapping",
                },
            )?)
            .and_then(|value| value.checked_add((indices.len() as u128).checked_mul(16)?))
            .ok_or(PauliError::Overflow {
                context: "estimating direct Majorana mapping",
            })?;
        if requested > max_bytes {
            return Err(PauliError::MemoryLimit {
                requested,
                limit: max_bytes,
            });
        }
        let generator_count = self.n_modes.checked_mul(2).ok_or(PauliError::Overflow {
            context: "checking direct Majorana mapping indices",
        })?;
        let mut structures = Vec::with_capacity(indices.len());
        let mut mapped_coefficients = Vec::with_capacity(coefficients.len());
        for (term_index, (word, &coefficient)) in indices.iter().zip(coefficients).enumerate() {
            if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
                return Err(PauliError::NonFiniteCoefficient { index: term_index });
            }
            if word.windows(2).any(|window| window[0] >= window[1])
                || word.iter().any(|&index| {
                    usize::try_from(index)
                        .map(|value| value >= generator_count)
                        .unwrap_or(true)
                })
            {
                return Err(PauliError::NonCanonicalTerms { index: term_index });
            }
            let mut x_words = vec![0_u64; word_count];
            let mut z_words = vec![0_u64; word_count];
            let mut local_x = vec![0_u64; word_count];
            let mut local_z = vec![0_u64; word_count];
            let mut phase = PauliPhase::PlusOne;
            for &majorana_index in word {
                let majorana_index =
                    usize::try_from(majorana_index).map_err(|_| PauliError::Overflow {
                        context: "converting Majorana index for direct mapping",
                    })?;
                let mode = majorana_index / 2;
                let odd = majorana_index & 1 != 0;
                local_x.fill(0);
                local_z.fill(0);
                for qubit in 0..mode {
                    local_z[qubit / 64] |= 1_u64 << (qubit % 64);
                }
                local_x[mode / 64] |= 1_u64 << (mode % 64);
                if odd {
                    local_z[mode / 64] |= 1_u64 << (mode % 64);
                }
                phase = phase.multiply(packed_pauli_product_phase(
                    &x_words,
                    &z_words,
                    &local_x,
                    &local_z,
                    self.n_modes,
                ));
                for (left, &right) in x_words.iter_mut().zip(&local_x) {
                    *left ^= right;
                }
                for (left, &right) in z_words.iter_mut().zip(&local_z) {
                    *left ^= right;
                }
            }
            let jordan_wigner_word = PauliWord::from_words(self.n_modes, x_words, z_words)?;
            let (mapped_word, mapping_phase) = self.transform_codes(&jordan_wigner_word.codes())?;
            structures.push(mapped_word);
            mapped_coefficients.push(coefficient * phase.as_complex() * mapping_phase);
        }
        PauliOperator::from_terms(self.n_modes, &structures, &mapped_coefficients)
    }

    /// Transform the mapped-fermion component of a canonical hybrid batch.
    pub fn map_hybrid_terms(
        &self,
        layout: HybridLayout,
        batch: HybridBatch<'_>,
        max_bytes: u128,
    ) -> Result<HybridCanonicalResult, PauliError> {
        let count = batch.coefficients.len();
        if batch.fermion_present.len() != count
            || batch.fermion_creation.len() != count
            || batch.fermion_annihilation.len() != count
            || batch.boson_present.len() != count
            || batch.boson_blocks.len() != count
            || batch.qubit_codes.len() != count
            || batch.mapped_present.len() != count
            || batch.mapped_codes.len() != count
            || batch.qudit_present.len() != count
            || batch.qudit_triples.len() != count
        {
            return Err(PauliError::InvalidStructureLength {
                expected: count,
                actual: batch.coefficients.len(),
            });
        }
        if layout.n_modes != self.n_modes {
            return Err(PauliError::InvalidStructureLength {
                expected: self.n_modes,
                actual: layout.n_modes,
            });
        }
        let requested = (count.max(1) as u128)
            .checked_mul((self.n_modes / 64 + 1) as u128)
            .and_then(|value| value.checked_mul(32))
            .ok_or(PauliError::Overflow {
                context: "estimating mapped hybrid output",
            })?;
        if requested > max_bytes {
            return Err(PauliError::MemoryLimit {
                requested,
                limit: max_bytes,
            });
        }
        let mut aggregate: FxHashMap<HybridMappingKey, Complex64> = FxHashMap::default();
        for index in 0..count {
            if batch.fermion_present[index]
                || !batch.fermion_creation[index].is_empty()
                || !batch.fermion_annihilation[index].is_empty()
            {
                return Err(PauliError::InvalidSector {
                    context: "hybrid mapping expects Jordan-Wigner-expanded fermion terms",
                });
            }
            if batch.qubit_codes[index].len() != layout.nqubits
                || batch.mapped_codes[index].len() != layout.n_modes
            {
                return Err(PauliError::InvalidStructureLength {
                    expected: layout.n_modes,
                    actual: batch.mapped_codes[index].len(),
                });
            }
            validate_boson_blocks(layout.n_bosons, &batch.boson_blocks[index])?;
            if batch.qudit_triples[index]
                .windows(2)
                .any(|pair| pair[0].0 >= pair[1].0)
                || batch.qudit_triples[index].iter().any(|&(site, a, b)| {
                    site as usize >= layout.n_qudit_sites
                        || a as usize >= layout.qudit_dimension
                        || b as usize >= layout.qudit_dimension
                })
            {
                return Err(PauliError::NonCanonicalTerms { index });
            }
            if !batch.coefficients[index].re.is_finite()
                || !batch.coefficients[index].im.is_finite()
            {
                return Err(PauliError::NonFiniteCoefficient { index });
            }
            let (mapped_codes, phase) = if batch.mapped_present[index] {
                self.transform_codes(&batch.mapped_codes[index])?
            } else {
                (Vec::new(), Complex64::new(1.0, 0.0))
            };
            let key = (
                false,
                Vec::new(),
                Vec::new(),
                batch.boson_present[index],
                batch.boson_blocks[index].clone(),
                batch.qubit_codes[index].clone(),
                batch.mapped_present[index],
                mapped_codes,
                batch.qudit_present[index],
                batch.qudit_triples[index].clone(),
            );
            *aggregate.entry(key).or_insert(Complex64::new(0.0, 0.0)) +=
                batch.coefficients[index] * phase;
        }
        let mut result = HybridCanonicalResult {
            fermion_present: Vec::with_capacity(aggregate.len()),
            fermion_creation: Vec::with_capacity(aggregate.len()),
            fermion_annihilation: Vec::with_capacity(aggregate.len()),
            boson_present: Vec::with_capacity(aggregate.len()),
            boson_blocks: Vec::with_capacity(aggregate.len()),
            qubit_codes: Vec::with_capacity(aggregate.len()),
            mapped_present: Vec::with_capacity(aggregate.len()),
            mapped_codes: Vec::with_capacity(aggregate.len()),
            qudit_present: Vec::with_capacity(aggregate.len()),
            qudit_triples: Vec::with_capacity(aggregate.len()),
            coefficients: Vec::with_capacity(aggregate.len()),
        };
        let mut entries: Vec<_> = aggregate.into_iter().collect();
        entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        for (
            (
                _fermion_present,
                _creation,
                _annihilation,
                boson_present,
                boson_blocks,
                qubit_codes,
                mapped_present,
                mapped_codes,
                qudit_present,
                qudit_triples,
            ),
            coefficient,
        ) in entries
        {
            if coefficient.re == 0.0 && coefficient.im == 0.0 {
                continue;
            }
            result.fermion_present.push(false);
            result.fermion_creation.push(Vec::new());
            result.fermion_annihilation.push(Vec::new());
            result.boson_present.push(boson_present);
            result.boson_blocks.push(boson_blocks);
            result.qubit_codes.push(qubit_codes);
            result.mapped_present.push(mapped_present);
            result.mapped_codes.push(mapped_codes);
            result.qudit_present.push(qudit_present);
            result.qudit_triples.push(qudit_triples);
            result.coefficients.push(coefficient);
        }
        Ok(result)
    }
}

/// Construct one of the frozen Phase 7.5 occupation encodings.
pub fn build_mapping_plan(
    mapping: &str,
    n_modes: usize,
    max_bytes: u128,
) -> Result<MappingPlan, PauliError> {
    if !matches!(mapping, "jordan_wigner" | "parity" | "bravyi_kitaev") {
        return Err(PauliError::InvalidSector {
            context: "unsupported fermion-to-qubit mapping",
        });
    }
    let max_cnot_count = (n_modes as u128)
        .checked_mul(n_modes.saturating_sub(1) as u128)
        .and_then(|value| value.checked_div(2))
        .ok_or(PauliError::Overflow {
            context: "estimating fermion mapping plan",
        })?;
    let upper_bound = mapping_plan_native_bytes(
        n_modes,
        usize::try_from(max_cnot_count).map_err(|_| PauliError::Overflow {
            context: "estimating fermion mapping plan",
        })?,
    )?;
    if upper_bound > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: upper_bound,
            limit: max_bytes,
        });
    }

    let encoding = mapping_matrix(mapping, n_modes);
    let inverse_encoding = gf2_inverse(&encoding)?;
    let cnot_operations = canonical_cnot_operations(&encoding)?;
    let x_transform = PackedBinaryMatrix::from_rows(encoding.clone(), false)?;
    let z_transform = PackedBinaryMatrix::from_rows(inverse_encoding.clone(), true)?;
    let estimated_bytes = mapping_plan_native_bytes(n_modes, cnot_operations.len())?;
    Ok(MappingPlan {
        n_modes,
        encoding,
        inverse_encoding,
        cnot_operations,
        x_transform,
        z_transform,
        estimated_bytes,
    })
}

fn mapping_matrix(mapping: &str, n_modes: usize) -> Vec<Vec<u8>> {
    match mapping {
        "jordan_wigner" => (0..n_modes)
            .map(|row| (0..n_modes).map(|column| u8::from(row == column)).collect())
            .collect(),
        "parity" => (0..n_modes)
            .map(|row| (0..n_modes).map(|column| u8::from(column <= row)).collect())
            .collect(),
        "bravyi_kitaev" => (0..n_modes)
            .map(|row| {
                let endpoint = row + 1;
                let lowbit = endpoint & endpoint.wrapping_neg();
                let start = endpoint - lowbit;
                (0..n_modes)
                    .map(|column| u8::from(start <= column && column <= row))
                    .collect()
            })
            .collect(),
        _ => unreachable!("mapping name was validated by build_mapping_plan"),
    }
}

fn gf2_inverse(matrix: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, PauliError> {
    let n_modes = matrix.len();
    if matrix.iter().any(|row| row.len() != n_modes) {
        return Err(PauliError::InvalidStructureLength {
            expected: n_modes,
            actual: matrix
                .iter()
                .find(|row| row.len() != n_modes)
                .map_or(0, Vec::len),
        });
    }
    let augmented_width = n_modes.checked_mul(2).ok_or(PauliError::Overflow {
        context: "constructing fermion mapping inverse",
    })?;
    let mut augmented = vec![vec![0_u8; augmented_width]; n_modes];
    for (row_index, row) in matrix.iter().enumerate() {
        for (column, &value) in row.iter().enumerate() {
            augmented[row_index][column] = value;
        }
        augmented[row_index][n_modes + row_index] = 1;
    }
    for column in 0..n_modes {
        let pivot = (column..n_modes).find(|&row| augmented[row][column] == 1);
        let Some(pivot) = pivot else {
            return Err(PauliError::InvalidSector {
                context: "occupation matrix must be invertible over GF(2)",
            });
        };
        if pivot != column {
            augmented.swap(column, pivot);
        }
        for row in 0..n_modes {
            if row != column && augmented[row][column] == 1 {
                xor_rows(&mut augmented, row, column, augmented_width);
            }
        }
    }
    Ok(augmented
        .into_iter()
        .map(|row| row[n_modes..].to_vec())
        .collect())
}

fn canonical_cnot_operations(matrix: &[Vec<u8>]) -> Result<Vec<(usize, usize)>, PauliError> {
    let n_modes = matrix.len();
    let mut rows = matrix.to_vec();
    let mut reductions = Vec::new();
    for pivot in 0..n_modes {
        if rows[pivot][pivot] != 1 {
            return Err(PauliError::InvalidSector {
                context: "occupation matrix must be unit triangular",
            });
        }
        for target in pivot + 1..n_modes {
            if rows[target][pivot] == 1 {
                xor_rows(&mut rows, target, pivot, n_modes);
                reductions.push((pivot, target));
            }
        }
    }
    for (row, values) in rows.iter().enumerate() {
        for (column, &value) in values.iter().enumerate() {
            if value != u8::from(row == column) {
                return Err(PauliError::InvalidSector {
                    context: "occupation matrix reduction did not reach identity",
                });
            }
        }
    }
    reductions.reverse();
    Ok(reductions)
}

fn xor_rows(rows: &mut [Vec<u8>], target: usize, source: usize, width: usize) {
    debug_assert_ne!(target, source);
    if target < source {
        let (before, after) = rows.split_at_mut(source);
        let target_row = &mut before[target];
        let source_row = &after[0];
        for (target_value, &source_value) in target_row.iter_mut().take(width).zip(source_row) {
            *target_value ^= source_value;
        }
    } else {
        let (before, after) = rows.split_at_mut(target);
        let source_row = &before[source];
        let target_row = &mut after[0];
        for (target_value, &source_value) in target_row.iter_mut().take(width).zip(source_row) {
            *target_value ^= source_value;
        }
    }
}

fn packed_pauli_product_phase(
    left_x: &[u64],
    left_z: &[u64],
    right_x: &[u64],
    right_z: &[u64],
    nqubits: usize,
) -> PauliPhase {
    let mut exponent = 0_u8;
    for qubit in 0..nqubits {
        let left = packed_pauli_code(left_x, left_z, qubit);
        let right = packed_pauli_code(right_x, right_z, qubit);
        exponent = (exponent + pauli_code_product_phase(left, right)) % 4;
    }
    PauliPhase::from_exponent(exponent)
}

fn packed_pauli_code(x_words: &[u64], z_words: &[u64], qubit: usize) -> u8 {
    let x = (x_words[qubit / 64] >> (qubit % 64)) & 1;
    let z = (z_words[qubit / 64] >> (qubit % 64)) & 1;
    match (x, z) {
        (0, 0) => 0,
        (1, 0) => 1,
        (1, 1) => 2,
        (0, 1) => 3,
        _ => unreachable!("packed Pauli bits are binary"),
    }
}

fn pauli_code_product_phase(left: u8, right: u8) -> u8 {
    match (left, right) {
        (0, _) | (_, 0) | (1, 1) | (2, 2) | (3, 3) => 0,
        (1, 2) | (2, 3) | (3, 1) => 1,
        (2, 1) | (1, 3) | (3, 2) => 3,
        _ => unreachable!("Pauli codes are validated before multiplication"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_mapping_plans_have_expected_matrices_and_provenance() {
        let parity = build_mapping_plan("parity", 4, u128::MAX).expect("parity plan");
        assert_eq!(
            parity.encoding,
            vec![
                vec![1, 0, 0, 0],
                vec![1, 1, 0, 0],
                vec![1, 1, 1, 0],
                vec![1, 1, 1, 1],
            ]
        );
        assert_eq!(
            parity.inverse_encoding,
            vec![
                vec![1, 0, 0, 0],
                vec![1, 1, 0, 0],
                vec![0, 1, 1, 0],
                vec![0, 0, 1, 1],
            ]
        );
        assert_eq!(parity.cnot_operations.len(), 6);

        let bk = build_mapping_plan("bravyi_kitaev", 4, u128::MAX).expect("BK plan");
        assert_eq!(
            bk.encoding,
            vec![
                vec![1, 0, 0, 0],
                vec![1, 1, 0, 0],
                vec![0, 0, 1, 0],
                vec![1, 1, 1, 1],
            ]
        );
        assert_eq!(bk.cnot_operations.len(), 4);
    }

    #[test]
    fn mapping_transform_matches_cnot_pauli_conjugation_for_a_batch() {
        let plan = build_mapping_plan("parity", 2, u128::MAX).expect("parity plan");
        let result = plan
            .map_pauli_terms(
                &[vec![1, 0], vec![0, 2]],
                &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
                u128::MAX,
            )
            .expect("mapped Pauli batch");
        assert_eq!(result.terms().len(), 2);
        assert_eq!(result.nqubits(), 2);
    }

    #[test]
    fn direct_majorana_mapping_has_one_native_output_word() {
        let plan = build_mapping_plan("parity", 16, u128::MAX).expect("parity plan");
        let result = plan
            .map_majorana_terms(
                &[(0_u64..16_u64).collect()],
                &[Complex64::new(1.0, 0.0)],
                u128::MAX,
            )
            .expect("direct Majorana mapping");
        assert_eq!(result.terms().len(), 1);
    }
}
