//! Native finite-basis kernels shared by structured Python operators.

use std::collections::{hash_map::Entry, BTreeMap};
use std::mem::size_of;

use crate::{Complex64, PauliError};
use rustc_hash::FxHashMap;

type FermionKey = (Vec<u32>, Vec<u32>);
type BosonKey = Vec<(u32, u32, u32)>;
pub type FermionCanonicalResult = (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<Complex64>);
pub type BosonCanonicalResult = (Vec<Vec<(u32, u32, u32)>>, Vec<Complex64>);

pub struct FermionBatch<'a> {
    pub creation: &'a [Vec<u32>],
    pub annihilation: &'a [Vec<u32>],
    pub coefficients: &'a [Complex64],
}

/// A flattened batch of canonical mixed-domain terms.
///
/// Presence flags distinguish an absent domain factor from its identity word.
/// This keeps the Python-facing representation compact while allowing one
/// coarse-grained Rust multiplication for a complete hybrid operator.
pub struct HybridBatch<'a> {
    pub fermion_present: &'a [bool],
    pub fermion_creation: &'a [Vec<u32>],
    pub fermion_annihilation: &'a [Vec<u32>],
    pub boson_present: &'a [bool],
    pub boson_blocks: &'a [Vec<(u32, u32, u32)>],
    pub qubit_codes: &'a [Vec<u8>],
    pub mapped_present: &'a [bool],
    pub mapped_codes: &'a [Vec<u8>],
    pub qudit_present: &'a [bool],
    pub qudit_triples: &'a [Vec<(u32, u32, u32)>],
    pub coefficients: &'a [Complex64],
}

pub struct HybridRawBatch<'a> {
    pub fermion_factors: &'a [Vec<(usize, u8)>],
    pub boson_factors: &'a [Vec<(usize, u8)>],
    pub qubit_codes: &'a [Vec<u8>],
    pub qudit_present: &'a [bool],
    pub qudit_triples: &'a [Vec<(u32, u32, u32)>],
    pub coefficients: &'a [Complex64],
}

/// Result of a batched mixed-domain symbolic multiplication.
pub struct HybridCanonicalResult {
    pub fermion_present: Vec<bool>,
    pub fermion_creation: Vec<Vec<u32>>,
    pub fermion_annihilation: Vec<Vec<u32>>,
    pub boson_present: Vec<bool>,
    pub boson_blocks: Vec<Vec<(u32, u32, u32)>>,
    pub qubit_codes: Vec<Vec<u8>>,
    pub mapped_present: Vec<bool>,
    pub mapped_codes: Vec<Vec<u8>>,
    pub qudit_present: Vec<bool>,
    pub qudit_triples: Vec<Vec<(u32, u32, u32)>>,
    pub coefficients: Vec<Complex64>,
}

#[derive(Clone, Copy, Debug)]
pub struct HybridLayout {
    pub n_modes: usize,
    pub n_bosons: usize,
    pub n_qubits: usize,
    pub n_qudit_sites: usize,
    pub qudit_dimension: usize,
}

type WeylKey = Vec<(u32, u32, u32)>;
type HybridKey = (
    Option<FermionKey>,
    Option<BosonKey>,
    Vec<u8>,
    Option<Vec<u8>>,
    Option<WeylKey>,
);

/// Multiply complete canonical hybrid operators in one coarse-grained call.
pub fn multiply_hybrid_terms(
    layout: HybridLayout,
    left: HybridBatch<'_>,
    right: HybridBatch<'_>,
    max_bytes: u128,
) -> Result<HybridCanonicalResult, PauliError> {
    validate_hybrid_batch(layout, &left)?;
    validate_hybrid_batch(layout, &right)?;
    let pair_count = left
        .coefficients
        .len()
        .checked_mul(right.coefficients.len())
        .ok_or(PauliError::Overflow {
            context: "estimating hybrid product expansion",
        })?;
    check_structured_bytes(pair_count, max_bytes, "hybrid product expansion")?;

    let mut aggregate: FxHashMap<HybridKey, Vec<Complex64>> = FxHashMap::default();
    for left_index in 0..left.coefficients.len() {
        for right_index in 0..right.coefficients.len() {
            let (fermion_products, boson_products) = (
                hybrid_fermion_products(&left, left_index, &right, right_index),
                hybrid_boson_products(&left, left_index, &right, right_index),
            );
            let (qubit_codes, qubit_phase) = multiply_pauli_codes(
                &left.qubit_codes[left_index],
                &right.qubit_codes[right_index],
            );
            let (mapped_codes, mapped_phase) =
                hybrid_mapped_product(layout.n_modes, &left, left_index, &right, right_index);
            let (qudit_triples, qudit_phase) = hybrid_qudit_product(
                layout.qudit_dimension,
                &left,
                left_index,
                &right,
                right_index,
            );
            let scalar = left.coefficients[left_index]
                * right.coefficients[right_index]
                * qubit_phase
                * mapped_phase
                * qudit_phase;
            for (fermion, fermion_integer) in &fermion_products {
                for (boson, boson_integer) in &boson_products {
                    let value = scalar * (*fermion_integer as f64) * (*boson_integer as f64);
                    push_aggregate(
                        &mut aggregate,
                        (
                            fermion.clone(),
                            boson.clone(),
                            qubit_codes.clone(),
                            mapped_codes.clone(),
                            qudit_triples.clone(),
                        ),
                        value,
                        max_bytes,
                    )?;
                }
            }
        }
    }
    finish_hybrid_aggregate(aggregate)
}

/// Canonicalize raw mixed-domain product specifications in one batch.
pub fn canonicalize_hybrid_terms(
    layout: HybridLayout,
    batch: HybridRawBatch<'_>,
    max_bytes: u128,
) -> Result<HybridCanonicalResult, PauliError> {
    let count = batch.coefficients.len();
    if batch.fermion_factors.len() != count
        || batch.boson_factors.len() != count
        || batch.qubit_codes.len() != count
        || batch.qudit_present.len() != count
        || batch.qudit_triples.len() != count
    {
        return Err(PauliError::InvalidStructureLength {
            expected: count,
            actual: batch.fermion_factors.len(),
        });
    }
    let mut aggregate: FxHashMap<HybridKey, Vec<Complex64>> = FxHashMap::default();
    for index in 0..count {
        validate_coefficient(batch.coefficients[index], index)?;
        if batch.qubit_codes[index].len() != layout.n_qubits {
            return Err(PauliError::InvalidStructureLength {
                expected: layout.n_qubits,
                actual: batch.qubit_codes[index].len(),
            });
        }
        if batch.qubit_codes[index].iter().any(|&code| code > 3) {
            return Err(PauliError::InvalidCode { code: 4, index });
        }
        let fermion_products = if batch.fermion_factors[index].is_empty() {
            vec![(None, 1)]
        } else {
            let sequence = raw_sequence(
                &batch.fermion_factors[index],
                layout.n_modes,
                index,
                "fermion mode",
            )?;
            fermion_rewrite(sequence)
                .into_iter()
                .map(|(key, value)| (Some(key), value))
                .collect()
        };
        let boson_products = if batch.boson_factors[index].is_empty() {
            vec![(None, 1)]
        } else {
            let sequence = raw_sequence(
                &batch.boson_factors[index],
                layout.n_bosons,
                index,
                "boson mode",
            )?;
            boson_rewrite(sequence)
                .into_iter()
                .map(|(key, value)| (Some(key), value))
                .collect()
        };
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
        for (fermion, fermion_integer) in &fermion_products {
            for (boson, boson_integer) in &boson_products {
                push_aggregate(
                    &mut aggregate,
                    (
                        if layout.n_modes != 0 {
                            Some(fermion.clone().unwrap_or_default())
                        } else {
                            None
                        },
                        if layout.n_bosons != 0 {
                            Some(boson.clone().unwrap_or_default())
                        } else {
                            None
                        },
                        batch.qubit_codes[index].clone(),
                        None,
                        if batch.qudit_present[index] {
                            Some(batch.qudit_triples[index].clone())
                        } else {
                            None
                        },
                    ),
                    batch.coefficients[index] * (*fermion_integer as f64) * (*boson_integer as f64),
                    max_bytes,
                )?;
            }
        }
    }
    finish_hybrid_aggregate(aggregate)
}

fn hybrid_fermion_products(
    left: &HybridBatch<'_>,
    left_index: usize,
    right: &HybridBatch<'_>,
    right_index: usize,
) -> Vec<(Option<FermionKey>, i64)> {
    if !left.fermion_present[left_index] && !right.fermion_present[right_index] {
        return vec![(None, 1)];
    }
    let mut sequence = fermion_sequence(
        &left.fermion_creation[left_index],
        &left.fermion_annihilation[left_index],
    );
    sequence.extend(fermion_sequence(
        &right.fermion_creation[right_index],
        &right.fermion_annihilation[right_index],
    ));
    fermion_rewrite(sequence)
        .into_iter()
        .map(|(key, value)| (Some(key), value))
        .collect()
}

fn hybrid_boson_products(
    left: &HybridBatch<'_>,
    left_index: usize,
    right: &HybridBatch<'_>,
    right_index: usize,
) -> Vec<(Option<BosonKey>, i64)> {
    if !left.boson_present[left_index] && !right.boson_present[right_index] {
        return vec![(None, 1)];
    }
    let mut sequence = boson_sequence(&left.boson_blocks[left_index]);
    sequence.extend(boson_sequence(&right.boson_blocks[right_index]));
    boson_rewrite(sequence)
        .into_iter()
        .map(|(key, value)| (Some(key), value))
        .collect()
}

fn hybrid_mapped_product(
    n_modes: usize,
    left: &HybridBatch<'_>,
    left_index: usize,
    right: &HybridBatch<'_>,
    right_index: usize,
) -> (Option<Vec<u8>>, Complex64) {
    if !left.mapped_present[left_index] && !right.mapped_present[right_index] {
        return (None, Complex64::new(1.0, 0.0));
    }
    let identity = vec![0_u8; n_modes];
    let left_codes = if left.mapped_present[left_index] {
        &left.mapped_codes[left_index]
    } else {
        &identity
    };
    let right_codes = if right.mapped_present[right_index] {
        &right.mapped_codes[right_index]
    } else {
        &identity
    };
    let (codes, phase) = multiply_pauli_codes(left_codes, right_codes);
    (Some(codes), phase)
}

fn hybrid_qudit_product(
    dimension: usize,
    left: &HybridBatch<'_>,
    left_index: usize,
    right: &HybridBatch<'_>,
    right_index: usize,
) -> (Option<WeylKey>, Complex64) {
    if !left.qudit_present[left_index] && !right.qudit_present[right_index] {
        return (None, Complex64::new(1.0, 0.0));
    }
    let left_triples: &[(u32, u32, u32)] = if left.qudit_present[left_index] {
        &left.qudit_triples[left_index]
    } else {
        &[]
    };
    let right_triples: &[(u32, u32, u32)] = if right.qudit_present[right_index] {
        &right.qudit_triples[right_index]
    } else {
        &[]
    };
    let mut by_site: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
    let mut phase_exponent = 0_u128;
    let mut left_by_site = BTreeMap::new();
    let mut right_by_site = BTreeMap::new();
    for &(site, a, b) in left_triples {
        left_by_site.insert(site, (a, b));
    }
    for &(site, a, b) in right_triples {
        right_by_site.insert(site, (a, b));
    }
    for site in left_by_site
        .keys()
        .chain(right_by_site.keys())
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
    {
        let (a, b) = left_by_site.get(&site).copied().unwrap_or((0, 0));
        let (c, e) = right_by_site.get(&site).copied().unwrap_or((0, 0));
        phase_exponent =
            (phase_exponent + u128::from(b) * u128::from(c)) % u128::from(dimension as u64);
        let aa = (u128::from(a) + u128::from(c)) % u128::from(dimension as u64);
        let bb = (u128::from(b) + u128::from(e)) % u128::from(dimension as u64);
        if aa != 0 || bb != 0 {
            by_site.insert(site, (aa as u32, bb as u32));
        }
    }
    let triples = by_site
        .into_iter()
        .map(|(site, (a, b))| (site, a, b))
        .collect();
    let angle = 2.0 * std::f64::consts::PI * phase_exponent as f64 / dimension as f64;
    (Some(triples), Complex64::new(angle.cos(), angle.sin()))
}

fn validate_hybrid_batch(layout: HybridLayout, batch: &HybridBatch<'_>) -> Result<(), PauliError> {
    let count = batch.coefficients.len();
    let lengths = [
        batch.fermion_present.len(),
        batch.fermion_creation.len(),
        batch.fermion_annihilation.len(),
        batch.boson_present.len(),
        batch.boson_blocks.len(),
        batch.qubit_codes.len(),
        batch.mapped_present.len(),
        batch.mapped_codes.len(),
        batch.qudit_present.len(),
        batch.qudit_triples.len(),
    ];
    if lengths.iter().any(|&length| length != count) {
        return Err(PauliError::InvalidStructureLength {
            expected: count,
            actual: lengths
                .into_iter()
                .find(|&length| length != count)
                .unwrap_or(count),
        });
    }
    for index in 0..count {
        validate_coefficient(batch.coefficients[index], index)?;
        validate_fermion_arrays(
            layout.n_modes,
            std::slice::from_ref(&batch.fermion_creation[index]),
            std::slice::from_ref(&batch.fermion_annihilation[index]),
            std::slice::from_ref(&batch.coefficients[index]),
        )?;
        validate_boson_blocks(layout.n_bosons, &batch.boson_blocks[index])?;
        if batch.qubit_codes[index].len() != layout.n_qubits {
            return Err(PauliError::InvalidStructureLength {
                expected: layout.n_qubits,
                actual: batch.qubit_codes[index].len(),
            });
        }
        if batch.qubit_codes[index].iter().any(|&code| code > 3) {
            return Err(PauliError::InvalidCode { code: 4, index });
        }
        if batch.mapped_codes[index].len() != layout.n_modes {
            return Err(PauliError::InvalidStructureLength {
                expected: layout.n_modes,
                actual: batch.mapped_codes[index].len(),
            });
        }
        if batch.mapped_codes[index].iter().any(|&code| code > 3) {
            return Err(PauliError::InvalidCode { code: 4, index });
        }
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
    }
    if layout.n_qudit_sites != 0 && layout.qudit_dimension < 3 {
        return Err(PauliError::InvalidStructureLength {
            expected: 3,
            actual: layout.qudit_dimension,
        });
    }
    Ok(())
}

fn finish_hybrid_aggregate(
    aggregate: FxHashMap<HybridKey, Vec<Complex64>>,
) -> Result<HybridCanonicalResult, PauliError> {
    let mut entries: Vec<_> = aggregate.into_iter().collect();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let capacity = entries.len();
    let mut result = HybridCanonicalResult {
        fermion_present: Vec::with_capacity(capacity),
        fermion_creation: Vec::with_capacity(capacity),
        fermion_annihilation: Vec::with_capacity(capacity),
        boson_present: Vec::with_capacity(capacity),
        boson_blocks: Vec::with_capacity(capacity),
        qubit_codes: Vec::with_capacity(capacity),
        mapped_present: Vec::with_capacity(capacity),
        mapped_codes: Vec::with_capacity(capacity),
        qudit_present: Vec::with_capacity(capacity),
        qudit_triples: Vec::with_capacity(capacity),
        coefficients: Vec::with_capacity(capacity),
    };
    for ((fermion, boson, qubit, mapped, qudit), values) in entries {
        let coefficient = deterministic_sum(values);
        validate_coefficient(coefficient, 0)?;
        if is_exact_zero(coefficient) {
            continue;
        }
        if let Some((creation, annihilation)) = fermion {
            result.fermion_present.push(true);
            result.fermion_creation.push(creation);
            result.fermion_annihilation.push(annihilation);
        } else {
            result.fermion_present.push(false);
            result.fermion_creation.push(Vec::new());
            result.fermion_annihilation.push(Vec::new());
        }
        if let Some(blocks) = boson {
            result.boson_present.push(true);
            result.boson_blocks.push(blocks);
        } else {
            result.boson_present.push(false);
            result.boson_blocks.push(Vec::new());
        }
        result.qubit_codes.push(qubit);
        if let Some(codes) = mapped {
            result.mapped_present.push(true);
            result.mapped_codes.push(codes);
        } else {
            result.mapped_present.push(false);
            result.mapped_codes.push(Vec::new());
        }
        if let Some(triples) = qudit {
            result.qudit_present.push(true);
            result.qudit_triples.push(triples);
        } else {
            result.qudit_present.push(false);
            result.qudit_triples.push(Vec::new());
        }
        result.coefficients.push(coefficient);
    }
    Ok(result)
}

/// Canonicalize a batch of raw fermion monomials with CAR rewriting.
pub fn canonicalize_fermion_terms(
    n_modes: usize,
    factors: &[Vec<(usize, u8)>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<FermionCanonicalResult, PauliError> {
    if factors.len() != coefficients.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: factors.len(),
            actual: coefficients.len(),
        });
    }
    let mut aggregate: FxHashMap<FermionKey, Vec<Complex64>> = FxHashMap::default();
    for (index, (raw_factors, &coefficient)) in factors.iter().zip(coefficients).enumerate() {
        validate_coefficient(coefficient, index)?;
        let mut sequence = Vec::with_capacity(raw_factors.len());
        for &(mode, action) in raw_factors {
            if mode >= n_modes {
                return Err(PauliError::InvalidIndex {
                    context: "fermion mode",
                });
            }
            if action > 1 {
                return Err(PauliError::InvalidCode {
                    code: action,
                    index,
                });
            }
            sequence.push((mode, action == 0));
        }
        for (key, integer) in fermion_rewrite(sequence) {
            push_aggregate(&mut aggregate, key, coefficient * integer as f64, max_bytes)?;
        }
    }
    finish_fermion_aggregate(aggregate)
}

/// Multiply two canonical fermion operators in one coarse-grained call.
pub fn multiply_fermion_terms(
    n_modes: usize,
    left: FermionBatch<'_>,
    right: FermionBatch<'_>,
    max_bytes: u128,
) -> Result<FermionCanonicalResult, PauliError> {
    validate_fermion_arrays(n_modes, left.creation, left.annihilation, left.coefficients)?;
    validate_fermion_arrays(
        n_modes,
        right.creation,
        right.annihilation,
        right.coefficients,
    )?;
    let pair_count = left
        .coefficients
        .len()
        .checked_mul(right.coefficients.len())
        .ok_or(PauliError::Overflow {
            context: "estimating fermion product expansion",
        })?;
    check_structured_bytes(pair_count, max_bytes, "fermion product expansion")?;
    let mut aggregate: FxHashMap<FermionKey, Vec<Complex64>> = FxHashMap::default();
    for left_index in 0..left.coefficients.len() {
        for right_index in 0..right.coefficients.len() {
            let mut sequence =
                fermion_sequence(&left.creation[left_index], &left.annihilation[left_index]);
            sequence.extend(fermion_sequence(
                &right.creation[right_index],
                &right.annihilation[right_index],
            ));
            for (key, integer) in fermion_rewrite(sequence) {
                push_aggregate(
                    &mut aggregate,
                    key,
                    left.coefficients[left_index]
                        * right.coefficients[right_index]
                        * integer as f64,
                    max_bytes,
                )?;
            }
        }
    }
    finish_fermion_aggregate(aggregate)
}

/// Map canonical fermion terms to Jordan-Wigner Pauli terms in one batch.
pub fn jordan_wigner_terms(
    n_modes: usize,
    creation: &[Vec<u32>],
    annihilation: &[Vec<u32>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<(Vec<Vec<u8>>, Vec<Complex64>), PauliError> {
    validate_fermion_arrays(n_modes, creation, annihilation, coefficients)?;
    let mut aggregate: FxHashMap<Vec<u8>, Vec<Complex64>> = FxHashMap::default();
    for (index, coefficient) in coefficients.iter().copied().enumerate() {
        let sequence = fermion_sequence(&creation[index], &annihilation[index]);
        let mut current: Vec<(Vec<u8>, Complex64)> =
            vec![(vec![0; n_modes], Complex64::new(1.0, 0.0))];
        for (mode, is_creation) in sequence {
            let y_coefficient = if is_creation {
                Complex64::new(0.0, -0.5)
            } else {
                Complex64::new(0.0, 0.5)
            };
            let expansion_count = current.len().checked_mul(2).ok_or(PauliError::Overflow {
                context: "estimating Jordan-Wigner expansion",
            })?;
            check_structured_bytes(expansion_count, max_bytes, "Jordan-Wigner expansion")?;
            let mut next: FxHashMap<Vec<u8>, Complex64> = FxHashMap::default();
            for (left_word, left_coefficient) in current {
                for (code, right_coefficient) in
                    [(1_u8, Complex64::new(0.5, 0.0)), (2_u8, y_coefficient)]
                {
                    let mut right_word = vec![3_u8; n_modes];
                    right_word[mode..].fill(0);
                    right_word[mode] = code;
                    let (word, phase) = multiply_pauli_codes(&left_word, &right_word);
                    let contribution = left_coefficient * right_coefficient * phase;
                    *next.entry(word).or_default() += contribution;
                }
            }
            current = next
                .into_iter()
                .filter(|(_, value)| !is_exact_zero(*value))
                .collect();
            current.sort_by(|left, right| left.0.cmp(&right.0));
        }
        for (word, value) in current {
            push_pauli_aggregate(&mut aggregate, word, coefficient * value, max_bytes)?;
        }
    }
    let mut entries: Vec<_> = aggregate.into_iter().collect();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let mut words = Vec::with_capacity(entries.len());
    let mut values = Vec::with_capacity(entries.len());
    for (word, contributions) in entries {
        let value = deterministic_sum(contributions);
        if !is_exact_zero(value) {
            words.push(word);
            values.push(value);
        }
    }
    Ok((words, values))
}

/// Map mixed-domain terms while retaining their non-fermionic factors.
pub fn jordan_wigner_hybrid_terms(
    layout: HybridLayout,
    batch: HybridBatch<'_>,
    max_bytes: u128,
) -> Result<HybridCanonicalResult, PauliError> {
    validate_hybrid_batch(layout, &batch)?;
    let mut aggregate: FxHashMap<HybridKey, Vec<Complex64>> = FxHashMap::default();
    for index in 0..batch.coefficients.len() {
        let expansions = if batch.fermion_present[index] {
            jordan_wigner_word_expansion(
                layout.n_modes,
                fermion_sequence(
                    &batch.fermion_creation[index],
                    &batch.fermion_annihilation[index],
                ),
                max_bytes,
            )?
        } else {
            vec![(batch.mapped_codes[index].clone(), Complex64::new(1.0, 0.0))]
        };
        for (mapped_codes, mapped_coefficient) in expansions {
            push_aggregate(
                &mut aggregate,
                (
                    None,
                    if batch.boson_present[index] {
                        Some(batch.boson_blocks[index].clone())
                    } else {
                        None
                    },
                    batch.qubit_codes[index].clone(),
                    if batch.fermion_present[index] || batch.mapped_present[index] {
                        Some(mapped_codes)
                    } else {
                        None
                    },
                    if batch.qudit_present[index] {
                        Some(batch.qudit_triples[index].clone())
                    } else {
                        None
                    },
                ),
                batch.coefficients[index] * mapped_coefficient,
                max_bytes,
            )?;
        }
    }
    finish_hybrid_aggregate(aggregate)
}

fn jordan_wigner_word_expansion(
    n_modes: usize,
    sequence: Vec<(usize, bool)>,
    max_bytes: u128,
) -> Result<Vec<(Vec<u8>, Complex64)>, PauliError> {
    let mut current: Vec<(Vec<u8>, Complex64)> = vec![(vec![0; n_modes], Complex64::new(1.0, 0.0))];
    for (mode, is_creation) in sequence {
        let y_coefficient = if is_creation {
            Complex64::new(0.0, -0.5)
        } else {
            Complex64::new(0.0, 0.5)
        };
        let expansion_count = current.len().checked_mul(2).ok_or(PauliError::Overflow {
            context: "estimating Jordan-Wigner expansion",
        })?;
        check_structured_bytes(expansion_count, max_bytes, "Jordan-Wigner expansion")?;
        let mut next: FxHashMap<Vec<u8>, Complex64> = FxHashMap::default();
        for (left_word, left_coefficient) in current {
            for (code, right_coefficient) in
                [(1_u8, Complex64::new(0.5, 0.0)), (2_u8, y_coefficient)]
            {
                let mut right_word = vec![3_u8; n_modes];
                right_word[mode..].fill(0);
                right_word[mode] = code;
                let (word, phase) = multiply_pauli_codes(&left_word, &right_word);
                *next.entry(word).or_default() += left_coefficient * right_coefficient * phase;
            }
        }
        current = next
            .into_iter()
            .filter(|(_, value)| !is_exact_zero(*value))
            .collect();
        current.sort_by(|left, right| left.0.cmp(&right.0));
    }
    Ok(current)
}

/// Canonicalize a batch of raw boson monomials with CCR rewriting.
pub fn canonicalize_boson_terms(
    n_modes: usize,
    factors: &[Vec<(usize, u8)>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<BosonCanonicalResult, PauliError> {
    if factors.len() != coefficients.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: factors.len(),
            actual: coefficients.len(),
        });
    }
    let mut aggregate: FxHashMap<BosonKey, Vec<Complex64>> = FxHashMap::default();
    for (index, (raw_factors, &coefficient)) in factors.iter().zip(coefficients).enumerate() {
        validate_coefficient(coefficient, index)?;
        let mut sequence = Vec::with_capacity(raw_factors.len());
        for &(mode, action) in raw_factors {
            if mode >= n_modes {
                return Err(PauliError::InvalidIndex {
                    context: "boson mode",
                });
            }
            if action > 1 {
                return Err(PauliError::InvalidCode {
                    code: action,
                    index,
                });
            }
            sequence.push((mode, action == 0));
        }
        for (key, integer) in boson_rewrite(sequence) {
            push_aggregate(&mut aggregate, key, coefficient * integer as f64, max_bytes)?;
        }
    }
    finish_boson_aggregate(aggregate)
}

/// Multiply two canonical boson operators in one coarse-grained call.
pub fn multiply_boson_terms(
    n_modes: usize,
    left_blocks: &[Vec<(u32, u32, u32)>],
    left_coefficients: &[Complex64],
    right_blocks: &[Vec<(u32, u32, u32)>],
    right_coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<BosonCanonicalResult, PauliError> {
    if left_blocks.len() != left_coefficients.len()
        || right_blocks.len() != right_coefficients.len()
    {
        return Err(PauliError::InvalidStructureLength {
            expected: left_blocks.len(),
            actual: left_coefficients.len(),
        });
    }
    for blocks in left_blocks.iter().chain(right_blocks) {
        validate_boson_blocks(n_modes, blocks)?;
    }
    let pair_count = left_coefficients
        .len()
        .checked_mul(right_coefficients.len())
        .ok_or(PauliError::Overflow {
            context: "estimating boson product expansion",
        })?;
    check_structured_bytes(pair_count, max_bytes, "boson product expansion")?;
    let mut aggregate: FxHashMap<BosonKey, Vec<Complex64>> = FxHashMap::default();
    for left in 0..left_blocks.len() {
        for right in 0..right_blocks.len() {
            let mut sequence = boson_sequence(&left_blocks[left]);
            sequence.extend(boson_sequence(&right_blocks[right]));
            for (key, integer) in boson_rewrite(sequence) {
                push_aggregate(
                    &mut aggregate,
                    key,
                    left_coefficients[left] * right_coefficients[right] * integer as f64,
                    max_bytes,
                )?;
            }
        }
    }
    finish_boson_aggregate(aggregate)
}

fn fermion_rewrite(sequence: Vec<(usize, bool)>) -> BTreeMap<FermionKey, i64> {
    for index in 0..sequence.len().saturating_sub(1) {
        let (left_mode, left_creation) = sequence[index];
        let (right_mode, right_creation) = sequence[index + 1];
        let duplicate = left_creation == right_creation && left_mode == right_mode;
        if duplicate {
            return BTreeMap::new();
        }
        let inversion = if left_creation && right_creation {
            left_mode > right_mode
        } else if !left_creation && !right_creation {
            left_mode < right_mode
        } else {
            !left_creation && right_creation
        };
        if !inversion {
            continue;
        }
        let mut swapped = sequence.clone();
        swapped.swap(index, index + 1);
        let mut result = BTreeMap::new();
        merge_integer_maps(&mut result, fermion_rewrite(swapped), -1);
        if !left_creation && right_creation && left_mode == right_mode {
            let mut contracted = sequence.clone();
            contracted.drain(index..=index + 1);
            merge_integer_maps(&mut result, fermion_rewrite(contracted), 1);
        }
        result.retain(|_, value| *value != 0);
        return result;
    }
    let creation = sequence
        .iter()
        .filter_map(|&(mode, is_creation)| is_creation.then_some(mode as u32))
        .collect();
    let annihilation = sequence
        .iter()
        .filter_map(|&(mode, is_creation)| (!is_creation).then_some(mode as u32))
        .collect();
    let mut result = BTreeMap::new();
    result.insert((creation, annihilation), 1);
    result
}

fn boson_rewrite(sequence: Vec<(usize, bool)>) -> BTreeMap<BosonKey, i64> {
    for index in 0..sequence.len().saturating_sub(1) {
        let (left_mode, left_creation) = sequence[index];
        let (right_mode, right_creation) = sequence[index + 1];
        let inversion =
            left_mode > right_mode || (left_mode == right_mode && !left_creation && right_creation);
        if !inversion {
            continue;
        }
        let mut swapped = sequence.clone();
        swapped.swap(index, index + 1);
        let mut result = boson_rewrite(swapped);
        if left_mode == right_mode && !left_creation && right_creation {
            let mut contracted = sequence.clone();
            contracted.drain(index..=index + 1);
            merge_integer_maps(&mut result, boson_rewrite(contracted), 1);
        }
        result.retain(|_, value| *value != 0);
        return result;
    }
    let mut powers: BTreeMap<usize, (u32, u32)> = BTreeMap::new();
    for (mode, is_creation) in sequence {
        let entry = powers.entry(mode).or_default();
        if is_creation {
            entry.0 += 1;
        } else {
            entry.1 += 1;
        }
    }
    let blocks = powers
        .into_iter()
        .filter_map(|(mode, (creation, annihilation))| {
            (creation != 0 || annihilation != 0).then_some((mode as u32, creation, annihilation))
        })
        .collect();
    let mut result = BTreeMap::new();
    result.insert(blocks, 1);
    result
}

fn fermion_sequence(creation: &[u32], annihilation: &[u32]) -> Vec<(usize, bool)> {
    creation
        .iter()
        .map(|&mode| (mode as usize, true))
        .chain(annihilation.iter().map(|&mode| (mode as usize, false)))
        .collect()
}

fn raw_sequence(
    factors: &[(usize, u8)],
    n_modes: usize,
    index: usize,
    context: &'static str,
) -> Result<Vec<(usize, bool)>, PauliError> {
    let mut sequence = Vec::with_capacity(factors.len());
    for &(mode, action) in factors {
        if mode >= n_modes {
            return Err(PauliError::InvalidIndex { context });
        }
        if action > 1 {
            return Err(PauliError::InvalidCode {
                code: action,
                index,
            });
        }
        sequence.push((mode, action == 0));
    }
    Ok(sequence)
}

fn boson_sequence(blocks: &[(u32, u32, u32)]) -> Vec<(usize, bool)> {
    blocks
        .iter()
        .flat_map(|&(mode, creation, annihilation)| {
            std::iter::repeat_n((mode as usize, true), creation as usize).chain(
                std::iter::repeat_n((mode as usize, false), annihilation as usize),
            )
        })
        .collect()
}

fn merge_integer_maps<K: Ord>(
    target: &mut BTreeMap<K, i64>,
    source: BTreeMap<K, i64>,
    factor: i64,
) {
    for (key, value) in source {
        *target.entry(key).or_default() += value * factor;
    }
}

fn validate_coefficient(value: Complex64, index: usize) -> Result<(), PauliError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(PauliError::NonFiniteCoefficient { index })
    }
}

fn validate_fermion_arrays(
    n_modes: usize,
    creation: &[Vec<u32>],
    annihilation: &[Vec<u32>],
    coefficients: &[Complex64],
) -> Result<(), PauliError> {
    if creation.len() != annihilation.len() || creation.len() != coefficients.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: creation.len(),
            actual: coefficients.len(),
        });
    }
    for index in 0..creation.len() {
        validate_coefficient(coefficients[index], index)?;
        if creation[index].windows(2).any(|pair| pair[0] >= pair[1])
            || annihilation[index]
                .windows(2)
                .any(|pair| pair[0] <= pair[1])
            || creation[index]
                .iter()
                .chain(&annihilation[index])
                .any(|&mode| mode as usize >= n_modes)
        {
            return Err(PauliError::NonCanonicalTerms { index });
        }
    }
    Ok(())
}

fn validate_boson_blocks(n_modes: usize, blocks: &[(u32, u32, u32)]) -> Result<(), PauliError> {
    if blocks.windows(2).any(|pair| pair[0].0 >= pair[1].0)
        || blocks.iter().any(|&(mode, _, _)| mode as usize >= n_modes)
    {
        return Err(PauliError::NonCanonicalTerms { index: 0 });
    }
    Ok(())
}

fn check_structured_bytes(
    count: usize,
    max_bytes: u128,
    context: &'static str,
) -> Result<(), PauliError> {
    let requested = (count as u128)
        .checked_mul(192)
        .ok_or(PauliError::Overflow { context })?;
    if requested > max_bytes {
        Err(PauliError::MemoryLimit {
            requested,
            limit: max_bytes,
        })
    } else {
        Ok(())
    }
}

fn push_aggregate<K: Eq + std::hash::Hash>(
    aggregate: &mut FxHashMap<K, Vec<Complex64>>,
    key: K,
    value: Complex64,
    max_bytes: u128,
) -> Result<(), PauliError> {
    validate_coefficient(value, 0)?;
    let value_count = {
        let values = aggregate.entry(key).or_default();
        values.push(value);
        values.len()
    };
    check_structured_bytes(
        aggregate.len().saturating_add(value_count),
        max_bytes,
        "structured aggregation",
    )
}

fn push_pauli_aggregate(
    aggregate: &mut FxHashMap<Vec<u8>, Vec<Complex64>>,
    key: Vec<u8>,
    value: Complex64,
    max_bytes: u128,
) -> Result<(), PauliError> {
    push_aggregate(aggregate, key, value, max_bytes)
}

fn finish_fermion_aggregate(
    aggregate: FxHashMap<FermionKey, Vec<Complex64>>,
) -> Result<FermionCanonicalResult, PauliError> {
    let mut entries: Vec<_> = aggregate.into_iter().collect();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let mut creation = Vec::with_capacity(entries.len());
    let mut annihilation = Vec::with_capacity(entries.len());
    let mut coefficients = Vec::with_capacity(entries.len());
    for ((left, right), values) in entries {
        let value = deterministic_sum(values);
        if !is_exact_zero(value) {
            creation.push(left);
            annihilation.push(right);
            coefficients.push(value);
        }
    }
    Ok((creation, annihilation, coefficients))
}

fn finish_boson_aggregate(
    aggregate: FxHashMap<BosonKey, Vec<Complex64>>,
) -> Result<BosonCanonicalResult, PauliError> {
    let mut entries: Vec<_> = aggregate.into_iter().collect();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let mut blocks = Vec::with_capacity(entries.len());
    let mut coefficients = Vec::with_capacity(entries.len());
    for (key, values) in entries {
        let value = deterministic_sum(values);
        if !is_exact_zero(value) {
            blocks.push(key);
            coefficients.push(value);
        }
    }
    Ok((blocks, coefficients))
}

fn deterministic_sum(mut values: Vec<Complex64>) -> Complex64 {
    values.sort_by_key(|value| (value.re.to_bits(), value.im.to_bits()));
    values
        .into_iter()
        .fold(Complex64::default(), |sum, value| sum + value)
}

fn is_exact_zero(value: Complex64) -> bool {
    value.re == 0.0 && value.im == 0.0
}

fn multiply_pauli_codes(left: &[u8], right: &[u8]) -> (Vec<u8>, Complex64) {
    let mut result = Vec::with_capacity(left.len());
    let mut phase = Complex64::new(1.0, 0.0);
    for (&left_code, &right_code) in left.iter().zip(right) {
        let (code, local_phase) = pauli_code_product(left_code, right_code);
        result.push(code);
        phase *= local_phase;
    }
    (result, phase)
}

fn pauli_code_product(left: u8, right: u8) -> (u8, Complex64) {
    match (left, right) {
        (0, code) | (code, 0) => (code, Complex64::new(1.0, 0.0)),
        (1, 1) | (2, 2) | (3, 3) => (0, Complex64::new(1.0, 0.0)),
        (1, 2) => (3, Complex64::new(0.0, 1.0)),
        (1, 3) => (2, Complex64::new(0.0, -1.0)),
        (2, 1) => (3, Complex64::new(0.0, -1.0)),
        (2, 3) => (1, Complex64::new(0.0, 1.0)),
        (3, 1) => (2, Complex64::new(0.0, 1.0)),
        (3, 2) => (1, Complex64::new(0.0, -1.0)),
        _ => (0, Complex64::new(0.0, 0.0)),
    }
}

/// One local finite-basis operation in a canonical structured term.
/// `kind=0` is Pauli, `kind=1` is a boson block, and `kind=2` is direct Weyl.
#[derive(Clone, Copy, Debug)]
pub struct StructuredOperation {
    pub axis: usize,
    pub kind: u8,
    pub p: u32,
    pub q: u32,
}

pub struct StructuredSparseResult {
    pub dimension: usize,
    pub rows: Vec<u64>,
    pub columns: Vec<u64>,
    pub values: Vec<Complex64>,
}

/// Compile canonical structured terms into deterministic sparse COO arrays.
///
/// This shares the exact transition convention with the dense kernel while
/// avoiding Python basis-state and dictionary loops for larger finite spaces.
pub fn structured_sparse_matrix(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<StructuredSparseResult, PauliError> {
    validate_structured_inputs(local_dimensions, terms, coefficients)?;
    let dimension = mixed_radix_dimension(local_dimensions)?;
    let mut entries: FxHashMap<(usize, usize), Complex64> = FxHashMap::default();
    let mut digits = vec![0usize; local_dimensions.len()];
    let mut output_digits = vec![0usize; local_dimensions.len()];
    for column in 0..dimension {
        decode_index(column, local_dimensions, &mut digits);
        for (term_index, (term, &coefficient)) in terms.iter().zip(coefficients).enumerate() {
            output_digits.copy_from_slice(&digits);
            let mut amplitude = coefficient;
            let mut valid = true;
            for operation in term {
                if !apply_structured_operation(
                    operation,
                    local_dimensions,
                    &mut output_digits,
                    &mut amplitude,
                )? {
                    valid = false;
                    break;
                }
            }
            if !valid {
                continue;
            }
            if !amplitude.re.is_finite() || !amplitude.im.is_finite() {
                return Err(PauliError::NonFiniteCoefficient { index: term_index });
            }
            let row = encode_index(&output_digits, local_dimensions);
            let key = (row, column);
            let next_entry_count = entries.len().saturating_add(1);
            match entries.entry(key) {
                Entry::Occupied(mut entry) => *entry.get_mut() += amplitude,
                Entry::Vacant(entry) => {
                    check_sparse_entry_bytes(next_entry_count, max_bytes)?;
                    entry.insert(amplitude);
                }
            }
        }
    }
    let mut ordered: Vec<_> = entries.into_iter().collect();
    ordered.sort_by_key(|(key, _)| *key);
    let mut rows = Vec::with_capacity(ordered.len());
    let mut columns = Vec::with_capacity(ordered.len());
    let mut values = Vec::with_capacity(ordered.len());
    for ((row, column), value) in ordered {
        if value.re == 0.0 && value.im == 0.0 {
            continue;
        }
        rows.push(u64::try_from(row).map_err(|_| PauliError::Overflow {
            context: "converting structured sparse row index",
        })?);
        columns.push(u64::try_from(column).map_err(|_| PauliError::Overflow {
            context: "converting structured sparse column index",
        })?);
        values.push(value);
    }
    check_sparse_output_bytes(rows.len(), max_bytes)?;
    Ok(StructuredSparseResult {
        dimension,
        rows,
        columns,
        values,
    })
}

/// Compile canonical structured terms into a row-major mixed-radix matrix.
pub fn structured_dense_matrix(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<(usize, Vec<Complex64>), PauliError> {
    validate_structured_inputs(local_dimensions, terms, coefficients)?;
    let dimension = mixed_radix_dimension(local_dimensions)?;
    let entries = dimension
        .checked_mul(dimension)
        .ok_or(PauliError::Overflow {
            context: "computing structured dense matrix entries",
        })?;
    let bytes = (entries as u128)
        .checked_mul(size_of::<Complex64>() as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating structured dense matrix memory",
        })?;
    if bytes > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: bytes,
            limit: max_bytes,
        });
    }
    for (index, &coefficient) in coefficients.iter().enumerate() {
        if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
            return Err(PauliError::NonFiniteCoefficient { index });
        }
    }
    let mut matrix = vec![Complex64::default(); entries];
    let mut digits = vec![0usize; local_dimensions.len()];
    let mut output_digits = vec![0usize; local_dimensions.len()];
    for column in 0..dimension {
        decode_index(column, local_dimensions, &mut digits);
        for (term, &coefficient) in terms.iter().zip(coefficients) {
            output_digits.copy_from_slice(&digits);
            let mut amplitude = coefficient;
            let mut valid = true;
            for operation in term {
                if !apply_structured_operation(
                    operation,
                    local_dimensions,
                    &mut output_digits,
                    &mut amplitude,
                )? {
                    valid = false;
                    break;
                }
            }
            if valid {
                let row = encode_index(&output_digits, local_dimensions);
                matrix[row * dimension + column] += amplitude;
            }
        }
    }
    Ok((dimension, matrix))
}

fn apply_pauli(
    code: u8,
    digit: usize,
    destination: &mut usize,
    amplitude: &mut Complex64,
) -> Result<(), PauliError> {
    match code {
        0 => {}
        1 => *destination = 1 - digit,
        2 => {
            *destination = 1 - digit;
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
        _ => return Err(PauliError::InvalidCode { code, index: 0 }),
    }
    Ok(())
}

fn validate_structured_inputs(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
) -> Result<(), PauliError> {
    if terms.len() != coefficients.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: terms.len(),
            actual: coefficients.len(),
        });
    }
    if local_dimensions.contains(&0) {
        return Err(PauliError::InvalidStructureLength {
            expected: 1,
            actual: 0,
        });
    }
    for (index, &coefficient) in coefficients.iter().enumerate() {
        if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
            return Err(PauliError::NonFiniteCoefficient { index });
        }
    }
    Ok(())
}

fn mixed_radix_dimension(local_dimensions: &[usize]) -> Result<usize, PauliError> {
    local_dimensions.iter().try_fold(1usize, |value, &factor| {
        value.checked_mul(factor).ok_or(PauliError::Overflow {
            context: "computing mixed-radix basis dimension",
        })
    })
}

fn check_sparse_entry_bytes(count: usize, max_bytes: u128) -> Result<(), PauliError> {
    let requested = (count as u128)
        .checked_mul(64)
        .ok_or(PauliError::Overflow {
            context: "estimating structured sparse workspace",
        })?;
    if requested > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested,
            limit: max_bytes,
        });
    }
    Ok(())
}

fn check_sparse_output_bytes(count: usize, max_bytes: u128) -> Result<(), PauliError> {
    let requested = (count as u128)
        .checked_mul(32)
        .ok_or(PauliError::Overflow {
            context: "estimating structured sparse output",
        })?;
    if requested > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested,
            limit: max_bytes,
        });
    }
    Ok(())
}

fn apply_structured_operation(
    operation: &StructuredOperation,
    local_dimensions: &[usize],
    output_digits: &mut [usize],
    amplitude: &mut Complex64,
) -> Result<bool, PauliError> {
    if operation.axis >= local_dimensions.len() {
        return Err(PauliError::InvalidIndex {
            context: "structured operation axis",
        });
    }
    let local_dimension = local_dimensions[operation.axis];
    let digit = output_digits[operation.axis];
    match operation.kind {
        0 => {
            if local_dimension != 2 {
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
            apply_pauli(
                operation.p as u8,
                digit,
                &mut output_digits[operation.axis],
                amplitude,
            )?;
        }
        1 => {
            let annihilation = operation.q as usize;
            let creation = operation.p as usize;
            if digit < annihilation {
                return Ok(false);
            }
            let destination = digit
                .checked_sub(annihilation)
                .and_then(|remaining| remaining.checked_add(creation))
                .ok_or(PauliError::Overflow {
                    context: "computing boson transition destination",
                })?;
            if destination >= local_dimension {
                return Ok(false);
            }
            let mut ladder_amplitude = 1.0;
            for offset in 0..annihilation {
                ladder_amplitude *= (digit - offset) as f64;
            }
            let remaining = digit - annihilation;
            for offset in 0..creation {
                ladder_amplitude *= (remaining + offset + 1) as f64;
            }
            *amplitude *= ladder_amplitude.sqrt();
            output_digits[operation.axis] = destination;
        }
        2 => {
            let a = (operation.p as usize) % local_dimension;
            let b_digit = u128::from(operation.q) * digit as u128;
            output_digits[operation.axis] = (digit + a) % local_dimension;
            let angle = 2.0 * std::f64::consts::PI * b_digit as f64 / local_dimension as f64;
            *amplitude *= Complex64::new(angle.cos(), angle.sin());
        }
        _ => {
            return Err(PauliError::InvalidCode {
                code: operation.kind,
                index: operation.axis,
            });
        }
    }
    Ok(true)
}

fn decode_index(mut index: usize, dimensions: &[usize], digits: &mut [usize]) {
    for position in (0..dimensions.len()).rev() {
        digits[position] = index % dimensions[position];
        index /= dimensions[position];
    }
}

fn encode_index(digits: &[usize], dimensions: &[usize]) -> usize {
    digits
        .iter()
        .zip(dimensions)
        .fold(0usize, |value, (&digit, &dimension)| {
            value * dimension + digit
        })
}

#[cfg(test)]
mod tests {
    use super::{
        canonicalize_boson_terms, canonicalize_fermion_terms, jordan_wigner_terms,
        structured_dense_matrix, structured_sparse_matrix, StructuredOperation,
    };
    use crate::Complex64;

    #[test]
    fn compiles_pauli_boson_and_weyl_operations_in_mixed_radix_order() {
        let (dimension, matrix) = structured_dense_matrix(
            &[2, 3],
            &[
                vec![StructuredOperation {
                    axis: 0,
                    kind: 0,
                    p: 1,
                    q: 0,
                }],
                vec![StructuredOperation {
                    axis: 1,
                    kind: 1,
                    p: 1,
                    q: 0,
                }],
            ],
            &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            u128::MAX,
        )
        .unwrap();
        assert_eq!(dimension, 6);
        assert_eq!(matrix[3], Complex64::new(1.0, 0.0));
        assert_eq!(matrix[dimension], Complex64::new(2.0, 0.0));

        let (dimension, matrix) = structured_dense_matrix(
            &[3],
            &[vec![StructuredOperation {
                axis: 0,
                kind: 2,
                p: 1,
                q: 1,
            }]],
            &[Complex64::new(1.0, 0.0)],
            u128::MAX,
        )
        .unwrap();
        assert_eq!(dimension, 3);
        assert_eq!(matrix[3], Complex64::new(1.0, 0.0));
        assert!(
            (matrix[2 * dimension + 1] - Complex64::new(-0.5, 3.0_f64.sqrt() / 2.0)).norm() < 1e-12
        );
    }

    #[test]
    fn rejects_dense_output_before_allocation() {
        let result = structured_dense_matrix(&[2, 2], &[], &[], 15);
        assert!(matches!(result, Err(crate::PauliError::MemoryLimit { .. })));
    }

    #[test]
    fn native_car_ccr_and_jordan_wigner_kernels_match_frozen_identities() {
        let (creation, annihilation, coefficients) = canonicalize_fermion_terms(
            1,
            &[vec![(0, 1), (0, 0)]],
            &[Complex64::new(1.0, 0.0)],
            u128::MAX,
        )
        .unwrap();
        assert_eq!(creation, vec![vec![], vec![0]]);
        assert_eq!(annihilation, vec![vec![], vec![0]]);
        assert_eq!(
            coefficients,
            vec![Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)]
        );

        let (blocks, coefficients) = canonicalize_boson_terms(
            1,
            &[vec![(0, 1), (0, 0)]],
            &[Complex64::new(1.0, 0.0)],
            u128::MAX,
        )
        .unwrap();
        assert_eq!(blocks, vec![vec![], vec![(0, 1, 1)]]);
        assert_eq!(
            coefficients,
            vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)]
        );

        let (words, coefficients) = jordan_wigner_terms(
            1,
            &[vec![0]],
            &[vec![0]],
            &[Complex64::new(1.0, 0.0)],
            u128::MAX,
        )
        .unwrap();
        assert_eq!(words, vec![vec![0], vec![3]]);
        assert_eq!(coefficients[0], Complex64::new(0.5, 0.0));
        assert_eq!(coefficients[1], Complex64::new(-0.5, 0.0));
    }

    #[test]
    fn sparse_structured_kernel_matches_dense_mixed_radix_output() {
        let dimensions = [2, 3];
        let terms = vec![
            vec![StructuredOperation {
                axis: 0,
                kind: 0,
                p: 1,
                q: 0,
            }],
            vec![StructuredOperation {
                axis: 1,
                kind: 1,
                p: 1,
                q: 0,
            }],
        ];
        let coefficients = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let (dimension, dense) =
            structured_dense_matrix(&dimensions, &terms, &coefficients, u128::MAX).unwrap();
        let sparse =
            structured_sparse_matrix(&dimensions, &terms, &coefficients, u128::MAX).unwrap();
        let mut reconstructed = vec![Complex64::default(); dense.len()];
        for ((&row, &column), &value) in sparse.rows.iter().zip(&sparse.columns).zip(&sparse.values)
        {
            reconstructed[row as usize * dimension + column as usize] = value;
        }
        assert_eq!(reconstructed, dense);
    }
}
