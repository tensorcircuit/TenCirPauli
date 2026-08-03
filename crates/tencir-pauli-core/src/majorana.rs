//! Exact sparse Majorana word and operator kernels.

use std::collections::BTreeMap;

use crate::{Complex64, PauliError};

pub struct MajoranaBatch<'a> {
    pub indices: &'a [Vec<u64>],
    pub coefficients: &'a [Complex64],
}

pub type MajoranaCanonicalResult = (Vec<Vec<u64>>, Vec<Complex64>);

fn invalid_word() -> PauliError {
    PauliError::InvalidSector {
        context: "invalid Majorana word",
    }
}

fn check_bytes(entries: usize, bytes_per_entry: usize, max_bytes: u128) -> Result<(), PauliError> {
    let requested = (entries as u128)
        .checked_mul(bytes_per_entry as u128)
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

fn canonicalize_indices(n_modes: usize, raw: &[u64]) -> Result<(Vec<u64>, i8), PauliError> {
    let limit = max_index(n_modes)?;
    let mut values = Vec::with_capacity(raw.len());
    let mut sign: i8 = 1;
    for &index in raw {
        if index >= limit {
            return Err(invalid_word());
        }
        let inversions = values.iter().filter(|&&value| value > index).count();
        if inversions & 1 != 0 {
            sign = -sign;
        }
        if let Some(position) = values.iter().position(|&value| value == index) {
            values.remove(position);
        } else {
            let position = values.partition_point(|&value| value < index);
            values.insert(position, index);
        }
    }
    Ok((values, sign))
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

fn multiply_words(left: &[u64], right: &[u64]) -> (Vec<u64>, i8) {
    let inversions = left
        .iter()
        .flat_map(|&left_index| {
            right
                .iter()
                .filter(move |&&right_index| left_index > right_index)
        })
        .count();
    let mut support = Vec::with_capacity(left.len() + right.len());
    let mut left_index = 0;
    let mut right_index = 0;
    while left_index < left.len() || right_index < right.len() {
        match (left.get(left_index), right.get(right_index)) {
            (Some(&left_value), Some(&right_value)) if left_value < right_value => {
                support.push(left_value);
                left_index += 1;
            }
            (Some(&left_value), Some(&right_value)) if right_value < left_value => {
                support.push(right_value);
                right_index += 1;
            }
            (Some(_), Some(_)) => {
                left_index += 1;
                right_index += 1;
            }
            (Some(&value), None) => {
                support.push(value);
                left_index += 1;
            }
            (None, Some(&value)) => {
                support.push(value);
                right_index += 1;
            }
            (None, None) => break,
        }
    }
    (support, if inversions & 1 == 0 { 1 } else { -1 })
}

fn finish(aggregate: BTreeMap<Vec<u64>, Complex64>) -> Result<MajoranaCanonicalResult, PauliError> {
    let mut indices = Vec::with_capacity(aggregate.len());
    let mut coefficients = Vec::with_capacity(aggregate.len());
    for (word, coefficient) in aggregate {
        if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
            return Err(PauliError::NonFiniteCoefficient { index: 0 });
        }
        if coefficient.re != 0.0 || coefficient.im != 0.0 {
            indices.push(word);
            coefficients.push(coefficient);
        }
    }
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
    let mut aggregate = BTreeMap::new();
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
    let mut aggregate = BTreeMap::new();
    for (left_index, &left_coefficient) in left.coefficients.iter().enumerate() {
        if !left_coefficient.re.is_finite() || !left_coefficient.im.is_finite() {
            return Err(PauliError::NonFiniteCoefficient { index: left_index });
        }
        validate_canonical(n_modes, &left.indices[left_index])?;
        for (right_index, &right_coefficient) in right.coefficients.iter().enumerate() {
            if !right_coefficient.re.is_finite() || !right_coefficient.im.is_finite() {
                return Err(PauliError::NonFiniteCoefficient { index: right_index });
            }
            validate_canonical(n_modes, &right.indices[right_index])?;
            let (word, sign) =
                multiply_words(&left.indices[left_index], &right.indices[right_index]);
            let value = left_coefficient * right_coefficient * f64::from(sign);
            *aggregate.entry(word).or_insert(Complex64::new(0.0, 0.0)) += value;
            check_bytes(aggregate.len(), 192, max_bytes)?;
        }
    }
    finish(aggregate)
}
