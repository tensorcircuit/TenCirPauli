//! Pure Rust occupation-encoding plans and batched Clifford transforms.

use crate::{Complex64, PauliError, PauliOperator, PauliWord};

const MAPPING_PLAN_FIXED_BYTES: u128 = 256;
const MAPPING_TERM_BYTES: u128 = 16;

/// A deterministic linear-reversible occupation encoding.
pub struct MappingPlan {
    n_modes: usize,
    encoding: Vec<Vec<u8>>,
    inverse_encoding: Vec<Vec<u8>>,
    cnot_operations: Vec<(usize, usize)>,
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

    fn transform_codes(&self, codes: &[u8]) -> Result<(Vec<u8>, Complex64), PauliError> {
        let word = PauliWord::from_codes(self.n_modes, codes)?;
        let mut result = word.codes();
        let mut phase = Complex64::new(1.0, 0.0);
        for &(control, target) in &self.cnot_operations {
            let (control_left, control_right) = control_factors(result[control]);
            let (target_left, target_right) = target_factors(result[target]);
            let (new_control, local_control_phase) = pauli_code_product(control_left, target_left);
            let (new_target, local_target_phase) = pauli_code_product(control_right, target_right);
            result[control] = new_control;
            result[target] = new_target;
            phase *= local_control_phase * local_target_phase;
        }
        Ok((result, phase))
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
    let matrix_entries =
        (n_modes as u128)
            .checked_mul(n_modes as u128)
            .ok_or(PauliError::Overflow {
                context: "estimating fermion mapping plan",
            })?;
    let max_cnot_count = (n_modes as u128)
        .checked_mul(n_modes.saturating_sub(1) as u128)
        .and_then(|value| value.checked_div(2))
        .ok_or(PauliError::Overflow {
            context: "estimating fermion mapping plan",
        })?;
    let upper_bound = matrix_entries
        .checked_mul(2)
        .and_then(|value| value.checked_add(max_cnot_count.checked_mul(16)?))
        .and_then(|value| value.checked_add(MAPPING_PLAN_FIXED_BYTES))
        .ok_or(PauliError::Overflow {
            context: "estimating fermion mapping plan",
        })?;
    if upper_bound > max_bytes {
        return Err(PauliError::MemoryLimit {
            requested: upper_bound,
            limit: max_bytes,
        });
    }

    let encoding = mapping_matrix(mapping, n_modes);
    let inverse_encoding = gf2_inverse(&encoding)?;
    let cnot_operations = canonical_cnot_operations(&encoding)?;
    let estimated_bytes = matrix_entries
        .checked_mul(2)
        .and_then(|value| value.checked_add((cnot_operations.len() as u128).checked_mul(16)?))
        .and_then(|value| value.checked_add(MAPPING_PLAN_FIXED_BYTES))
        .ok_or(PauliError::Overflow {
            context: "estimating fermion mapping plan",
        })?;
    Ok(MappingPlan {
        n_modes,
        encoding,
        inverse_encoding,
        cnot_operations,
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

fn control_factors(code: u8) -> (u8, u8) {
    match code {
        0 => (0, 0),
        1 => (1, 1),
        2 => (2, 1),
        3 => (3, 0),
        _ => unreachable!("Pauli codes are validated before conjugation"),
    }
}

fn target_factors(code: u8) -> (u8, u8) {
    match code {
        0 => (0, 0),
        1 => (0, 1),
        2 => (3, 2),
        3 => (3, 3),
        _ => unreachable!("Pauli codes are validated before conjugation"),
    }
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
}
