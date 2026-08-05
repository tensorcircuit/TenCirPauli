//! Exact sparse Majorana word and operator kernels.

use crate::structured::{canonicalize_fermion_terms, FermionCanonicalResult};
use crate::{Complex64, PauliError};
use rustc_hash::FxHashMap;

pub struct MajoranaBatch<'a> {
    pub indices: &'a [Vec<u64>],
    pub coefficients: &'a [Complex64],
}

pub type MajoranaCanonicalResult = (Vec<Vec<u64>>, Vec<Complex64>);

/// Packed support for a canonical Majorana word.
///
/// The public representation is a sorted list of generator indices, but the
/// algebraic hot paths only need support XOR and parity queries.  Keeping the
/// support in fixed-width limbs avoids allocating and comparing one index
/// vector for every intermediate aggregate key.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct PackedSupport {
    limbs: Vec<u64>,
}

impl PackedSupport {
    fn empty(n_modes: usize) -> Result<Self, PauliError> {
        let generator_count = n_modes.checked_mul(2).ok_or(PauliError::Overflow {
            context: "allocating packed Majorana support",
        })?;
        let limb_count = generator_count
            .checked_add(63)
            .ok_or(PauliError::Overflow {
                context: "allocating packed Majorana support",
            })?
            / 64;
        Ok(Self {
            limbs: vec![0; limb_count],
        })
    }

    fn toggle(&mut self, index: u64) {
        let index = index as usize;
        self.limbs[index / 64] ^= 1_u64 << (index % 64);
    }

    fn parity_above(&self, index: u64) -> u8 {
        let index = index as usize;
        let limb = index / 64;
        let bit = index % 64;
        let mut parity = if bit == 63 {
            0
        } else {
            ((self.limbs[limb] >> (bit + 1)).count_ones() & 1) as u8
        };
        for &value in &self.limbs[limb + 1..] {
            parity ^= (value.count_ones() & 1) as u8;
        }
        parity
    }

    fn from_canonical(n_modes: usize, word: &[u64]) -> Result<Self, PauliError> {
        validate_canonical(n_modes, word)?;
        let mut support = Self::empty(n_modes)?;
        for &index in word {
            support.toggle(index);
        }
        Ok(support)
    }

    fn to_indices(&self) -> Vec<u64> {
        let capacity = self
            .limbs
            .iter()
            .map(|value| value.count_ones() as usize)
            .sum();
        let mut indices = Vec::with_capacity(capacity);
        for (limb_index, &limb) in self.limbs.iter().enumerate() {
            let mut remaining = limb;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                indices.push((limb_index * 64 + bit) as u64);
                remaining &= remaining - 1;
            }
        }
        indices
    }
}

fn invalid_word() -> PauliError {
    PauliError::InvalidSector {
        context: "invalid Majorana word",
    }
}

fn check_bytes(entries: usize, bytes_per_entry: usize, max_bytes: u128) -> Result<(), PauliError> {
    check_bytes_u128(entries as u128, bytes_per_entry as u128, max_bytes)
}

fn check_bytes_u128(
    entries: u128,
    bytes_per_entry: u128,
    max_bytes: u128,
) -> Result<(), PauliError> {
    let requested = entries
        .checked_mul(bytes_per_entry)
        .ok_or(PauliError::Overflow {
            context: "estimating Majorana expansion",
        })?;
    if requested > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested,
            limit: max_bytes,
        });
    }
    Ok(())
}

fn max_index(n_modes: usize) -> Result<u64, PauliError> {
    let count = n_modes.checked_mul(2).ok_or(PauliError::Overflow {
        context: "checking Majorana index range",
    })?;
    u64::try_from(count).map_err(|_| PauliError::Overflow {
        context: "checking Majorana index range",
    })
}

fn canonicalize_indices(n_modes: usize, raw: &[u64]) -> Result<(PackedSupport, i8), PauliError> {
    let limit = max_index(n_modes)?;
    let mut support = PackedSupport::empty(n_modes)?;
    let mut sign: i8 = 1;
    for &index in raw {
        if index >= limit {
            return Err(invalid_word());
        }
        if support.parity_above(index) != 0 {
            sign = -sign;
        }
        support.toggle(index);
    }
    Ok((support, sign))
}

fn validate_canonical(n_modes: usize, word: &[u64]) -> Result<(), PauliError> {
    let limit = max_index(n_modes)?;
    if word.iter().any(|&index| index >= limit)
        || word.windows(2).any(|window| window[0] >= window[1])
    {
        return Err(PauliError::NonCanonicalTerms { index: 0 });
    }
    Ok(())
}

fn multiply_words(left: &PackedSupport, right: &PackedSupport) -> (PackedSupport, i8) {
    debug_assert_eq!(left.limbs.len(), right.limbs.len());
    let mut inversions = 0_u8;
    for (left_limb_index, &left_limb) in left.limbs.iter().enumerate() {
        for &right_limb in &right.limbs[..left_limb_index] {
            inversions ^= ((left_limb.count_ones() & 1) & (right_limb.count_ones() & 1)) as u8;
        }
        let mut remaining = left_limb;
        while remaining != 0 {
            let bit = remaining.trailing_zeros();
            let lower_mask = if bit == 0 { 0 } else { (1_u64 << bit) - 1 };
            inversions ^= ((right.limbs[left_limb_index] & lower_mask).count_ones() & 1) as u8;
            remaining &= remaining - 1;
        }
    }
    let mut support = left.clone();
    for (left_limb, &right_limb) in support.limbs.iter_mut().zip(&right.limbs) {
        *left_limb ^= right_limb;
    }
    (support, if inversions == 0 { 1 } else { -1 })
}

fn finish(
    aggregate: FxHashMap<PackedSupport, Complex64>,
) -> Result<MajoranaCanonicalResult, PauliError> {
    let mut ordered = Vec::with_capacity(aggregate.len());
    for (word, coefficient) in aggregate {
        if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
            return Err(PauliError::NonFiniteCoefficient { index: 0 });
        }
        if coefficient.re != 0.0 || coefficient.im != 0.0 {
            ordered.push((word.to_indices(), coefficient));
        }
    }
    ordered.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    let (indices, coefficients): (Vec<_>, Vec<_>) = ordered.into_iter().unzip();
    Ok((indices, coefficients))
}

pub fn canonicalize_majorana_terms(
    n_modes: usize,
    indices: &[Vec<u64>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<MajoranaCanonicalResult, PauliError> {
    if indices.len() != coefficients.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: indices.len(),
            actual: coefficients.len(),
        });
    }
    let mut aggregate = FxHashMap::default();
    for (index, (raw, &coefficient)) in indices.iter().zip(coefficients).enumerate() {
        if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
            return Err(PauliError::NonFiniteCoefficient { index });
        }
        let (word, sign) = canonicalize_indices(n_modes, raw)?;
        let value = if sign == 1 { coefficient } else { -coefficient };
        *aggregate.entry(word).or_insert(Complex64::new(0.0, 0.0)) += value;
        check_bytes(aggregate.len(), 192, max_bytes)?;
    }
    finish(aggregate)
}

/// Expand canonical Majorana words into canonical fermion words in one batch.
///
/// The expansion is performed entirely in Rust and then passed directly to
/// the existing CAR canonicalizer.  This keeps the unavoidable branch count,
/// intermediate factors, and contraction work on the native side of the FFI
/// boundary.
pub fn majorana_to_fermion_terms(
    n_modes: usize,
    indices: &[Vec<u64>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<FermionCanonicalResult, PauliError> {
    if indices.len() != coefficients.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: indices.len(),
            actual: coefficients.len(),
        });
    }

    let mut branch_count = 0_u128;
    for (index, (word, &coefficient)) in indices.iter().zip(coefficients).enumerate() {
        if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
            return Err(PauliError::NonFiniteCoefficient { index });
        }
        validate_canonical(n_modes, word)?;
        let degree = u32::try_from(word.len()).map_err(|_| PauliError::Overflow {
            context: "estimating Majorana-to-fermion expansion",
        })?;
        let branches = 1_u128.checked_shl(degree).ok_or(PauliError::Overflow {
            context: "estimating Majorana-to-fermion expansion",
        })?;
        branch_count = branch_count
            .checked_add(branches)
            .ok_or(PauliError::Overflow {
                context: "estimating Majorana-to-fermion expansion",
            })?;
    }
    check_bytes_u128(branch_count, 192, max_bytes)?;
    let branch_capacity = usize::try_from(branch_count).map_err(|_| PauliError::Overflow {
        context: "allocating Majorana-to-fermion expansion",
    })?;
    let mut factors = Vec::with_capacity(branch_capacity);
    let mut expanded_coefficients = Vec::with_capacity(branch_capacity);

    for (word, &coefficient) in indices.iter().zip(coefficients) {
        let branch_count = 1_usize
            .checked_shl(u32::try_from(word.len()).map_err(|_| PauliError::Overflow {
                context: "allocating Majorana-to-fermion expansion",
            })?)
            .ok_or(PauliError::Overflow {
                context: "allocating Majorana-to-fermion expansion",
            })?;
        for mask in 0..branch_count {
            let mut branch = Vec::with_capacity(word.len());
            let mut branch_coefficient = coefficient;
            for (position, &majorana_index) in word.iter().enumerate() {
                let mode_u32 =
                    u32::try_from(majorana_index / 2).map_err(|_| PauliError::Overflow {
                        context: "converting Majorana mode to fermion mode",
                    })?;
                let mode = usize::try_from(mode_u32).map_err(|_| PauliError::Overflow {
                    context: "converting Majorana mode to fermion mode",
                })?;
                let shift =
                    u32::try_from(word.len() - position - 1).map_err(|_| PauliError::Overflow {
                        context: "allocating Majorana-to-fermion expansion",
                    })?;
                let bit = 1_usize.checked_shl(shift).ok_or(PauliError::Overflow {
                    context: "allocating Majorana-to-fermion expansion",
                })?;
                let create = mask & bit == 0;
                branch.push((mode, if create { 0 } else { 1 }));
                if majorana_index & 1 != 0 {
                    branch_coefficient *= if create {
                        Complex64::new(0.0, 1.0)
                    } else {
                        Complex64::new(0.0, -1.0)
                    };
                }
            }
            factors.push(branch);
            expanded_coefficients.push(branch_coefficient);
        }
    }

    canonicalize_fermion_terms(n_modes, &factors, &expanded_coefficients, max_bytes)
}

/// Expand canonical fermion words into Majorana words in one native batch.
pub fn fermion_to_majorana_terms(
    n_modes: usize,
    creation: &[Vec<u32>],
    annihilation: &[Vec<u32>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<MajoranaCanonicalResult, PauliError> {
    if creation.len() != annihilation.len() || creation.len() != coefficients.len() {
        return Err(PauliError::InvalidStructureLength {
            expected: creation.len(),
            actual: annihilation.len(),
        });
    }
    let mut branch_count = 0_u128;
    for (index, ((creates, annihilates), &coefficient)) in creation
        .iter()
        .zip(annihilation)
        .zip(coefficients)
        .enumerate()
    {
        if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
            return Err(PauliError::NonFiniteCoefficient { index });
        }
        for &mode in creates.iter().chain(annihilates) {
            if usize::try_from(mode)
                .map(|mode| mode >= n_modes)
                .unwrap_or(true)
            {
                return Err(PauliError::InvalidIndex {
                    context: "fermion mode in fermion-to-Majorana conversion",
                });
            }
        }
        let degree = creates
            .len()
            .checked_add(annihilates.len())
            .ok_or(PauliError::Overflow {
                context: "estimating fermion-to-Majorana expansion",
            })?;
        let branches = 1_u128
            .checked_shl(u32::try_from(degree).map_err(|_| PauliError::Overflow {
                context: "estimating fermion-to-Majorana expansion",
            })?)
            .ok_or(PauliError::Overflow {
                context: "estimating fermion-to-Majorana expansion",
            })?;
        branch_count = branch_count
            .checked_add(branches)
            .ok_or(PauliError::Overflow {
                context: "estimating fermion-to-Majorana expansion",
            })?;
    }
    check_bytes_u128(branch_count, 192, max_bytes)?;
    let mut aggregate = FxHashMap::default();
    for ((creates, annihilates), &coefficient) in
        creation.iter().zip(annihilation).zip(coefficients)
    {
        let sequence = creates
            .iter()
            .map(|&mode| (mode, true))
            .chain(annihilates.iter().map(|&mode| (mode, false)))
            .collect::<Vec<_>>();
        let branch_count = 1_usize
            .checked_shl(
                u32::try_from(sequence.len()).map_err(|_| PauliError::Overflow {
                    context: "allocating fermion-to-Majorana expansion",
                })?,
            )
            .ok_or(PauliError::Overflow {
                context: "allocating fermion-to-Majorana expansion",
            })?;
        for mask in 0..branch_count {
            let mut raw = Vec::with_capacity(sequence.len());
            let mut branch_coefficient = coefficient;
            for (position, &(mode, is_creation)) in sequence.iter().enumerate() {
                let shift = u32::try_from(sequence.len() - position - 1).map_err(|_| {
                    PauliError::Overflow {
                        context: "allocating fermion-to-Majorana expansion",
                    }
                })?;
                let component_is_odd = mask & (1_usize << shift) != 0;
                let mode = u64::from(mode);
                raw.push(2 * mode + u64::from(component_is_odd));
                branch_coefficient *= Complex64::new(0.5, 0.0);
                if component_is_odd {
                    branch_coefficient *= if is_creation {
                        Complex64::new(0.0, -1.0)
                    } else {
                        Complex64::new(0.0, 1.0)
                    };
                }
            }
            let (word, sign) = canonicalize_indices(n_modes, &raw)?;
            *aggregate.entry(word).or_insert(Complex64::new(0.0, 0.0)) +=
                branch_coefficient * f64::from(sign);
        }
    }
    finish(aggregate)
}

pub fn multiply_majorana_terms(
    n_modes: usize,
    left: MajoranaBatch<'_>,
    right: MajoranaBatch<'_>,
    max_bytes: u128,
) -> Result<MajoranaCanonicalResult, PauliError> {
    if left.indices.len() != left.coefficients.len()
        || right.indices.len() != right.coefficients.len()
    {
        return Err(PauliError::InvalidStructureLength {
            expected: left.indices.len(),
            actual: left.coefficients.len(),
        });
    }
    let pair_count =
        left.indices
            .len()
            .checked_mul(right.indices.len())
            .ok_or(PauliError::Overflow {
                context: "estimating Majorana multiplication",
            })?;
    check_bytes(pair_count, 192, max_bytes)?;
    let mut aggregate = FxHashMap::default();
    let left_words = left
        .indices
        .iter()
        .map(|word| PackedSupport::from_canonical(n_modes, word))
        .collect::<Result<Vec<_>, _>>()?;
    let right_words = right
        .indices
        .iter()
        .map(|word| PackedSupport::from_canonical(n_modes, word))
        .collect::<Result<Vec<_>, _>>()?;
    for (left_index, &left_coefficient) in left.coefficients.iter().enumerate() {
        if !left_coefficient.re.is_finite() || !left_coefficient.im.is_finite() {
            return Err(PauliError::NonFiniteCoefficient { index: left_index });
        }
        for (right_index, &right_coefficient) in right.coefficients.iter().enumerate() {
            if !right_coefficient.re.is_finite() || !right_coefficient.im.is_finite() {
                return Err(PauliError::NonFiniteCoefficient { index: right_index });
            }
            let (word, sign) = multiply_words(&left_words[left_index], &right_words[right_index]);
            let value = left_coefficient * right_coefficient * f64::from(sign);
            *aggregate.entry(word).or_insert(Complex64::new(0.0, 0.0)) += value;
            check_bytes(aggregate.len(), 192, max_bytes)?;
        }
    }
    finish(aggregate)
}

/// Compute a Majorana commutator or anticommutator in one aggregate.
///
/// For supports `A` and `B`, the two ordered products have the same XOR
/// support and differ by the graded sign
/// `(-1)^(|A||B|-|A intersection B|)`.  This lets the fused kernel skip pairs
/// that cancel before inserting anything into the aggregate.
pub fn binary_majorana_terms(
    n_modes: usize,
    left: MajoranaBatch<'_>,
    right: MajoranaBatch<'_>,
    max_bytes: u128,
    reverse_sign: i8,
) -> Result<MajoranaCanonicalResult, PauliError> {
    if left.indices.len() != left.coefficients.len()
        || right.indices.len() != right.coefficients.len()
    {
        return Err(PauliError::InvalidStructureLength {
            expected: left.indices.len(),
            actual: left.coefficients.len(),
        });
    }
    let pair_count =
        left.indices
            .len()
            .checked_mul(right.indices.len())
            .ok_or(PauliError::Overflow {
                context: "estimating Majorana binary expansion",
            })?;
    check_bytes(pair_count, 192, max_bytes)?;
    let left_words = left
        .indices
        .iter()
        .map(|word| PackedSupport::from_canonical(n_modes, word))
        .collect::<Result<Vec<_>, _>>()?;
    let right_words = right
        .indices
        .iter()
        .map(|word| PackedSupport::from_canonical(n_modes, word))
        .collect::<Result<Vec<_>, _>>()?;
    let mut aggregate = FxHashMap::default();
    for (left_index, &left_coefficient) in left.coefficients.iter().enumerate() {
        if !left_coefficient.re.is_finite() || !left_coefficient.im.is_finite() {
            return Err(PauliError::NonFiniteCoefficient { index: left_index });
        }
        let left_degree = left.indices[left_index].len();
        for (right_index, &right_coefficient) in right.coefficients.iter().enumerate() {
            if !right_coefficient.re.is_finite() || !right_coefficient.im.is_finite() {
                return Err(PauliError::NonFiniteCoefficient { index: right_index });
            }
            let overlap = left_words[left_index]
                .limbs
                .iter()
                .zip(&right_words[right_index].limbs)
                .map(|(&left_limb, &right_limb)| (left_limb & right_limb).count_ones() as usize)
                .sum::<usize>();
            let graded_odd = (left_degree * right.indices[right_index].len() - overlap) & 1;
            let reverse_relation = if graded_odd == 0 { 1_i8 } else { -1_i8 };
            let (word, product_sign) =
                multiply_words(&left_words[left_index], &right_words[right_index]);
            let factor = if reverse_sign == 0 {
                product_sign
            } else {
                product_sign * (1 + reverse_sign * reverse_relation)
            };
            if factor == 0 {
                continue;
            }
            let value = left_coefficient * right_coefficient * f64::from(factor);
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(PauliError::NonFiniteCoefficient { index: left_index });
            }
            *aggregate.entry(word).or_insert(Complex64::new(0.0, 0.0)) += value;
            check_bytes(aggregate.len(), 192, max_bytes)?;
        }
    }
    finish(aggregate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn majorana_to_fermion_expansion_preserves_phase_convention() {
        let result =
            majorana_to_fermion_terms(1, &[vec![1]], &[Complex64::new(1.0, 0.0)], u128::MAX)
                .expect("Majorana expansion should succeed");
        assert_eq!(result.0.len(), 2);
        assert!(result.0.iter().any(|word| word == &vec![0]));
        assert!(result.1.iter().any(|word| word == &vec![0]));
        assert!(result
            .2
            .iter()
            .any(|value| { value.re == 0.0 && value.im == 1.0 }));
        assert!(result
            .2
            .iter()
            .any(|value| { value.re == 0.0 && value.im == -1.0 }));
    }

    #[test]
    fn majorana_to_fermion_expansion_checks_branch_budget_before_allocating() {
        let error = majorana_to_fermion_terms(2, &[vec![0, 1]], &[Complex64::new(1.0, 0.0)], 192)
            .expect_err("the branch budget must reject the expansion");
        assert!(matches!(error, PauliError::MemoryLimit { .. }));
    }

    #[test]
    fn packed_support_aggregate_is_sorted_by_public_index_tuples() {
        let result = canonicalize_majorana_terms(
            32,
            &[vec![1], vec![0, 63]],
            &[Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            u128::MAX,
        )
        .expect("packed Majorana canonicalization");
        assert_eq!(result.0, vec![vec![0, 63], vec![1]]);
    }
}
