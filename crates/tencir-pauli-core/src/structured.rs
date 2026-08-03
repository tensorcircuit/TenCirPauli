//! Native finite-basis kernels shared by structured Python operators.

use std::mem::size_of;

use crate::{Complex64, PauliError};

/// One local finite-basis operation in a canonical structured term.
/// `kind=0` is Pauli, `kind=1` is a boson block, and `kind=2` is direct Weyl.
#[derive(Clone, Copy, Debug)]
pub struct StructuredOperation {
    pub axis: usize,
    pub kind: u8,
    pub p: u32,
    pub q: u32,
}

/// Compile canonical structured terms into a row-major mixed-radix matrix.
pub fn structured_dense_matrix(
    local_dimensions: &[usize],
    terms: &[Vec<StructuredOperation>],
    coefficients: &[Complex64],
    max_bytes: u128,
) -> Result<(usize, Vec<Complex64>), PauliError> {
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
    let dimension = local_dimensions.iter().try_fold(1usize, |value, &factor| {
        value.checked_mul(factor).ok_or(PauliError::Overflow {
            context: "computing mixed-radix basis dimension",
        })
    })?;
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
    for column in 0..dimension {
        decode_index(column, local_dimensions, &mut digits);
        for (term, &coefficient) in terms.iter().zip(coefficients) {
            let mut output_digits = digits.clone();
            let mut amplitude = coefficient;
            let mut valid = true;
            for operation in term {
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
                        apply_pauli(
                            operation.p as u8,
                            digit,
                            &mut output_digits[operation.axis],
                            &mut amplitude,
                        )?;
                    }
                    1 => {
                        let annihilation = operation.q as usize;
                        let creation = operation.p as usize;
                        if digit < annihilation {
                            valid = false;
                            break;
                        }
                        let destination = digit - annihilation + creation;
                        if destination >= local_dimension {
                            valid = false;
                            break;
                        }
                        let mut ladder_amplitude = 1.0;
                        for offset in 0..annihilation {
                            ladder_amplitude *= (digit - offset) as f64;
                        }
                        let remaining = digit - annihilation;
                        for offset in 0..creation {
                            ladder_amplitude *= (remaining + offset + 1) as f64;
                        }
                        amplitude *= ladder_amplitude.sqrt();
                        output_digits[operation.axis] = destination;
                    }
                    2 => {
                        let a = (operation.p as usize) % local_dimension;
                        let b = operation.q as usize;
                        output_digits[operation.axis] = (digit + a) % local_dimension;
                        let angle = 2.0 * std::f64::consts::PI * (b * digit) as f64
                            / local_dimension as f64;
                        amplitude *= Complex64::new(angle.cos(), angle.sin());
                    }
                    _ => {
                        return Err(PauliError::InvalidCode {
                            code: operation.kind,
                            index: operation.axis,
                        });
                    }
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
    use super::{structured_dense_matrix, StructuredOperation};
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
}
