//! Generic finite charge-sector transition compilation.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::{Complex64, PauliError};

/// One canonical structured term in a finite charge-sector layout.
#[derive(Clone, Debug)]
pub struct ChargeTransitionTerm {
    pub fermion_creation: Vec<u32>,
    pub fermion_annihilation: Vec<u32>,
    pub boson_blocks: Vec<(u32, u32, u32)>,
    pub qubit_codes: Vec<u8>,
    pub mapped_present: bool,
    pub mapped_codes: Vec<u8>,
    pub qudit_present: bool,
    pub qudit_triples: Vec<(u32, u32, u32)>,
    pub coefficient: Complex64,
}

/// Deterministic restricted transition arrays.
pub type ChargeTransitionResult = (Vec<u64>, Vec<u64>, Vec<Complex64>);

/// Immutable layout and memory policy for one restricted compilation.
pub struct ChargeTransitionLayout<'a> {
    pub dimension: usize,
    pub basis: &'a [u64],
    pub local_dimensions: &'a [u64],
    pub fermion_positions: &'a [u64],
    pub boson_positions: &'a [u64],
    pub qubit_positions: &'a [u64],
    pub qudit_positions: &'a [u64],
    pub qudit_dimension: u64,
    pub max_bytes: u128,
}

fn invalid_sector() -> PauliError {
    PauliError::InvalidSector {
        context: "invalid charge-sector transition input",
    }
}

fn check_bytes(entries: usize, bytes_per_entry: usize, limit: u128) -> Result<(), PauliError> {
    let requested = (entries as u128)
        .checked_mul(bytes_per_entry as u128)
        .ok_or(PauliError::Overflow {
            context: "estimating charge-sector transition storage",
        })?;
    if requested > limit {
        return Err(PauliError::MemoryLimit { requested, limit });
    }
    Ok(())
}

fn positions(values: &[u64], axis_count: usize, limit: u128) -> Result<Vec<usize>, PauliError> {
    check_bytes(values.len(), std::mem::size_of::<usize>(), limit)?;
    let mut result = Vec::with_capacity(values.len());
    for &value in values {
        let position = usize::try_from(value).map_err(|_| invalid_sector())?;
        if position >= axis_count {
            return Err(invalid_sector());
        }
        result.push(position);
    }
    if result.iter().collect::<HashSet<_>>().len() != result.len() {
        return Err(invalid_sector());
    }
    Ok(result)
}

fn apply_phase(value: &mut Complex64, phase: Complex64) {
    *value *= phase;
}

fn apply_pauli(
    occupations: &mut [u64],
    codes: &[u8],
    positions: &[usize],
    coefficient: &mut Complex64,
) -> Result<(), PauliError> {
    if codes.len() != positions.len() {
        return Err(invalid_sector());
    }
    for (&code, &position) in codes.iter().zip(positions) {
        let bit = occupations[position];
        if bit > 1 {
            return Err(invalid_sector());
        }
        match code {
            0 => {}
            1 => occupations[position] = 1 - bit,
            2 => {
                occupations[position] = 1 - bit;
                apply_phase(
                    coefficient,
                    if bit == 0 {
                        Complex64::new(0.0, 1.0)
                    } else {
                        Complex64::new(0.0, -1.0)
                    },
                );
            }
            3 => {
                if bit == 1 {
                    *coefficient = -*coefficient;
                }
            }
            _ => return Err(invalid_sector()),
        }
    }
    Ok(())
}

fn apply_fermions(
    occupations: &mut [u64],
    creation: &[u32],
    annihilation: &[u32],
    positions: &[usize],
    coefficient: &mut Complex64,
) -> Result<bool, PauliError> {
    let mut apply = |mode: u32, create: bool| -> Result<bool, PauliError> {
        let mode = usize::try_from(mode).map_err(|_| invalid_sector())?;
        if mode >= positions.len() {
            return Err(invalid_sector());
        }
        let position = positions[mode];
        let occupied = occupations[position];
        if occupied > 1 {
            return Err(invalid_sector());
        }
        let parity = (0..mode)
            .map(|lower| occupations[positions[lower]])
            .sum::<u64>()
            & 1;
        if parity != 0 {
            *coefficient = -*coefficient;
        }
        if create {
            if occupied != 0 {
                return Ok(false);
            }
            occupations[position] = 1;
        } else {
            if occupied == 0 {
                return Ok(false);
            }
            occupations[position] = 0;
        }
        Ok(true)
    };

    for &mode in annihilation.iter().rev() {
        if !apply(mode, false)? {
            return Ok(false);
        }
    }
    for &mode in creation.iter().rev() {
        if !apply(mode, true)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn apply_bosons(
    occupations: &mut [u64],
    blocks: &[(u32, u32, u32)],
    positions: &[usize],
    local_dimensions: &[u64],
    coefficient: &mut Complex64,
) -> Result<bool, PauliError> {
    for &(mode, creation, annihilation) in blocks.iter().rev() {
        let mode = usize::try_from(mode).map_err(|_| invalid_sector())?;
        if mode >= positions.len() {
            return Err(invalid_sector());
        }
        let position = positions[mode];
        for _ in 0..annihilation {
            let occupation = occupations[position];
            if occupation == 0 {
                return Ok(false);
            }
            *coefficient *= (occupation as f64).sqrt();
            occupations[position] = occupation - 1;
        }
        for _ in 0..creation {
            let occupation = occupations[position];
            if occupation
                .checked_add(1)
                .map(|value| value >= local_dimensions[position])
                .unwrap_or(true)
            {
                return Ok(false);
            }
            *coefficient *= ((occupation + 1) as f64).sqrt();
            occupations[position] = occupation + 1;
        }
    }
    Ok(true)
}

fn apply_qudits(
    occupations: &mut [u64],
    triples: &[(u32, u32, u32)],
    positions: &[usize],
    dimension: u64,
    coefficient: &mut Complex64,
) -> Result<(), PauliError> {
    if dimension == 0 {
        return Err(invalid_sector());
    }
    for &(site, a, b) in triples {
        let site = usize::try_from(site).map_err(|_| invalid_sector())?;
        if site >= positions.len() || u64::from(a) >= dimension || u64::from(b) >= dimension {
            return Err(invalid_sector());
        }
        let position = positions[site];
        let input = occupations[position];
        let angle = 2.0 * std::f64::consts::PI * ((u128::from(b) * u128::from(input)) as f64)
            / (dimension as f64);
        apply_phase(coefficient, Complex64::from_polar(1.0, angle));
        occupations[position] = (input + u64::from(a)) % dimension;
    }
    Ok(())
}

/// Compile a structured operator directly in a finite selected basis.
///
/// Terms that reach the same destination are aggregated before the
/// destination is checked against the selected basis. This is required for
/// exact cancellation such as XX + YY in a fixed-particle-number sector.
pub fn compile_charge_transitions(
    layout: ChargeTransitionLayout<'_>,
    terms: &[ChargeTransitionTerm],
) -> Result<ChargeTransitionResult, PauliError> {
    let ChargeTransitionLayout {
        dimension,
        basis,
        local_dimensions,
        fermion_positions,
        boson_positions,
        qubit_positions,
        qudit_positions,
        qudit_dimension,
        max_bytes,
    } = layout;
    let axis_count = local_dimensions.len();
    let expected_basis = dimension
        .checked_mul(axis_count)
        .ok_or(PauliError::Overflow {
            context: "sizing charge-sector basis",
        })?;
    if basis.len() != expected_basis {
        return Err(invalid_sector());
    }
    check_bytes(
        dimension,
        axis_count
            .checked_mul(std::mem::size_of::<u64>())
            .and_then(|value| value.checked_add(32))
            .ok_or(PauliError::Overflow {
                context: "estimating charge-sector basis workspace",
            })?,
        max_bytes,
    )?;
    for &local_dimension in local_dimensions {
        if local_dimension == 0 {
            return Err(invalid_sector());
        }
    }
    let fermion_positions = positions(fermion_positions, axis_count, max_bytes)?;
    let boson_positions = positions(boson_positions, axis_count, max_bytes)?;
    let qubit_positions = positions(qubit_positions, axis_count, max_bytes)?;
    let qudit_positions = positions(qudit_positions, axis_count, max_bytes)?;
    for row in basis.chunks_exact(axis_count.max(1)) {
        for (index, &value) in row.iter().take(axis_count).enumerate() {
            if value >= local_dimensions[index] {
                return Err(invalid_sector());
            }
        }
    }
    for term in terms {
        if term.qubit_codes.len() != qubit_positions.len()
            || term.mapped_codes.len() != fermion_positions.len()
            || (term.mapped_present
                && (!term.fermion_creation.is_empty() || !term.fermion_annihilation.is_empty()))
            || (!term.qudit_present && !term.qudit_triples.is_empty())
        {
            return Err(invalid_sector());
        }
        if !term.coefficient.re.is_finite() || !term.coefficient.im.is_finite() {
            return Err(PauliError::NonFiniteCoefficient { index: 0 });
        }
    }

    let mut basis_index: HashMap<Vec<u64>, u64> = HashMap::with_capacity(dimension);
    for row_index in 0..dimension {
        let start = row_index
            .checked_mul(axis_count)
            .ok_or(PauliError::Overflow {
                context: "indexing charge-sector basis",
            })?;
        let key = basis[start..start + axis_count].to_vec();
        let index = u64::try_from(row_index).map_err(|_| PauliError::Overflow {
            context: "indexing charge-sector transitions",
        })?;
        if basis_index.insert(key, index).is_some() {
            return Err(invalid_sector());
        }
    }

    let mut transitions: BTreeMap<(u64, u64), Complex64> = BTreeMap::new();
    for column in 0..dimension {
        let start = column.checked_mul(axis_count).ok_or(PauliError::Overflow {
            context: "indexing charge-sector basis",
        })?;
        let source = &basis[start..start + axis_count];
        let mut destinations: BTreeMap<Vec<u64>, Complex64> = BTreeMap::new();
        for term in terms {
            let mut destination = source.to_vec();
            let mut value = term.coefficient;
            if !apply_fermions(
                &mut destination,
                &term.fermion_creation,
                &term.fermion_annihilation,
                &fermion_positions,
                &mut value,
            )? {
                continue;
            }
            if !apply_bosons(
                &mut destination,
                &term.boson_blocks,
                &boson_positions,
                local_dimensions,
                &mut value,
            )? {
                continue;
            }
            apply_pauli(
                &mut destination,
                &term.qubit_codes,
                &qubit_positions,
                &mut value,
            )?;
            if term.mapped_present {
                apply_pauli(
                    &mut destination,
                    &term.mapped_codes,
                    &fermion_positions,
                    &mut value,
                )?;
            }
            if term.qudit_present {
                apply_qudits(
                    &mut destination,
                    &term.qudit_triples,
                    &qudit_positions,
                    qudit_dimension,
                    &mut value,
                )?;
            }
            if value.re == 0.0 && value.im == 0.0 {
                continue;
            }
            let entry = destinations
                .entry(destination)
                .or_insert(Complex64::new(0.0, 0.0));
            *entry += value;
            check_bytes(destinations.len(), 64, max_bytes)?;
        }
        for (destination, value) in destinations {
            if value.re == 0.0 && value.im == 0.0 {
                continue;
            }
            let row = basis_index
                .get(&destination)
                .ok_or(PauliError::InvalidSector {
                    context: "operator leaks outside the selected charge sector",
                })?;
            let column_index = u64::try_from(column).map_err(|_| PauliError::Overflow {
                context: "indexing charge-sector transitions",
            })?;
            let entry = transitions
                .entry((*row, column_index))
                .or_insert(Complex64::new(0.0, 0.0));
            *entry += value;
            check_bytes(transitions.len(), 40, max_bytes)?;
        }
    }

    let mut rows = Vec::with_capacity(transitions.len());
    let mut columns = Vec::with_capacity(transitions.len());
    let mut coefficients = Vec::with_capacity(transitions.len());
    for ((row, column), value) in transitions {
        if value.re != 0.0 || value.im != 0.0 {
            rows.push(row);
            columns.push(column);
            coefficients.push(value);
        }
    }
    Ok((rows, columns, coefficients))
}
