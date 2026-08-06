//! Backend-neutral logical circuit representation.
//!
//! Angles are either compile-time constants or indices into one flat runtime
//! angle vector.  Runtime indices are an internal numerical ABI; this module
//! contains no symbolic expression language.

use std::sync::Arc;

use crate::error::PauliError;
use crate::scalar::Complex64;

pub const CIRCUIT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AngleRef {
    Static(f64),
    Slot(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub enum CircuitGate {
    Rz {
        wire: usize,
        angle: usize,
    },
    Rzz {
        wire0: usize,
        wire1: usize,
        angle: usize,
    },
    Cz {
        wire0: usize,
        wire1: usize,
    },
    Cphase {
        wire0: usize,
        wire1: usize,
        angle: usize,
    },
    Swap {
        wire0: usize,
        wire1: usize,
    },
    Iswap {
        wire0: usize,
        wire1: usize,
        angle: usize,
    },
    Diagonal {
        wires: Vec<usize>,
        payload: Vec<Complex64>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct CircuitProgram {
    schema_version: u32,
    nqubits: usize,
    operations: Arc<[CircuitGate]>,
    angles: Arc<[AngleRef]>,
    nparameters: usize,
}

impl CircuitProgram {
    pub fn new(
        schema_version: u32,
        nqubits: usize,
        operations: Vec<CircuitGate>,
        angles: Vec<AngleRef>,
        nparameters: usize,
    ) -> Result<Self, PauliError> {
        if schema_version != CIRCUIT_SCHEMA_VERSION {
            return Err(PauliError::InvalidCircuit {
                context: "unknown schema version",
            });
        }
        validate_angles(&angles, nparameters)?;
        for operation in &operations {
            validate_gate(operation, nqubits, angles.len())?;
        }
        Ok(Self {
            schema_version,
            nqubits,
            operations: Arc::from(operations.into_boxed_slice()),
            angles: Arc::from(angles.into_boxed_slice()),
            nparameters,
        })
    }

    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }
    pub fn nqubits(&self) -> usize {
        self.nqubits
    }
    pub fn operations(&self) -> &[CircuitGate] {
        &self.operations
    }
    pub fn angles(&self) -> &[AngleRef] {
        &self.angles
    }
    pub fn angle_count(&self) -> usize {
        self.angles.len()
    }
    pub fn nparameters(&self) -> usize {
        self.nparameters
    }

    pub fn evaluate_parameters(&self, parameters: &[f64]) -> Result<Vec<f64>, PauliError> {
        if parameters.len() != self.nparameters {
            return Err(PauliError::InvalidParameterLength {
                expected: self.nparameters,
                actual: parameters.len(),
            });
        }
        if let Some(index) = parameters.iter().position(|value| !value.is_finite()) {
            return Err(PauliError::NonFiniteParameter { index });
        }
        self.angles
            .iter()
            .map(|angle| match *angle {
                AngleRef::Static(value) => Ok(value),
                AngleRef::Slot(slot) => Ok(parameters[slot]),
            })
            .collect()
    }

    pub fn gradient_from_angle_adjoint(
        &self,
        angle_adjoint: &[f64],
    ) -> Result<Vec<f64>, PauliError> {
        if angle_adjoint.len() != self.angles.len() {
            return Err(PauliError::InvalidCircuit {
                context: "angle derivative buffer has the wrong length",
            });
        }
        let mut gradient = vec![0.0; self.nparameters];
        for (adjoint, angle) in angle_adjoint.iter().zip(self.angles.iter()) {
            if let AngleRef::Slot(slot) = *angle {
                gradient[slot] += *adjoint;
            }
        }
        Ok(gradient)
    }
}

fn validate_angles(angles: &[AngleRef], nparameters: usize) -> Result<(), PauliError> {
    let mut observed = vec![false; nparameters];
    for angle in angles {
        match *angle {
            AngleRef::Static(value) if !value.is_finite() => {
                return Err(PauliError::InvalidCircuit {
                    context: "static circuit angle is non-finite",
                });
            }
            AngleRef::Static(_) => {}
            AngleRef::Slot(slot) => {
                if slot >= nparameters {
                    return Err(PauliError::InvalidCircuit {
                        context: "circuit angle slot is outside the declared range",
                    });
                }
                observed[slot] = true;
            }
        }
    }
    if observed.iter().any(|value| !value) {
        return Err(PauliError::InvalidCircuit {
            context: "circuit angle slots must cover 0..nparameters-1 without holes",
        });
    }
    Ok(())
}

fn validate_gate(gate: &CircuitGate, nqubits: usize, angle_count: usize) -> Result<(), PauliError> {
    let check_wire = |wire: usize| {
        if wire >= nqubits {
            Err(PauliError::InvalidWire { wire, nqubits })
        } else {
            Ok(())
        }
    };
    let check_pair = |wire0: usize, wire1: usize| {
        check_wire(wire0)?;
        check_wire(wire1)?;
        if wire0 == wire1 {
            return Err(PauliError::DuplicateWire);
        }
        Ok(())
    };
    let check_angle = |angle: usize| {
        if angle >= angle_count {
            Err(PauliError::InvalidCircuit {
                context: "gate angle is outside the circuit angle table",
            })
        } else {
            Ok(())
        }
    };
    match gate {
        CircuitGate::Rz { wire, angle } => {
            check_wire(*wire)?;
            check_angle(*angle)?;
        }
        CircuitGate::Rzz {
            wire0,
            wire1,
            angle,
        }
        | CircuitGate::Cphase {
            wire0,
            wire1,
            angle,
        }
        | CircuitGate::Iswap {
            wire0,
            wire1,
            angle,
        } => {
            check_pair(*wire0, *wire1)?;
            check_angle(*angle)?;
        }
        CircuitGate::Cz { wire0, wire1 } | CircuitGate::Swap { wire0, wire1 } => {
            check_pair(*wire0, *wire1)?;
        }
        CircuitGate::Diagonal { wires, payload } => {
            if wires.is_empty() {
                return Err(PauliError::InvalidCircuit {
                    context: "diagonal gate requires at least one wire",
                });
            }
            let local_dimension =
                1usize
                    .checked_shl(wires.len() as u32)
                    .ok_or(PauliError::Overflow {
                        context: "sizing diagonal gate payload",
                    })?;
            if payload.len() != local_dimension {
                return Err(PauliError::InvalidCircuit {
                    context: "diagonal payload length does not match its arity",
                });
            }
            for (index, wire) in wires.iter().copied().enumerate() {
                check_wire(wire)?;
                if wires[..index].contains(&wire) {
                    return Err(PauliError::DuplicateWire);
                }
            }
            if payload.iter().any(|value| {
                !value.re.is_finite() || !value.im.is_finite() || (value.norm() - 1.0).abs() > 1e-12
            }) {
                return Err(PauliError::InvalidCircuit {
                    context: "diagonal payload must be finite and unit modulus",
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numerical_angle_table_evaluates_static_and_runtime_angles() {
        let program = CircuitProgram::new(
            CIRCUIT_SCHEMA_VERSION,
            2,
            vec![CircuitGate::Rz { wire: 0, angle: 1 }],
            vec![AngleRef::Static(2.0), AngleRef::Slot(0)],
            1,
        )
        .unwrap();
        assert_eq!(program.evaluate_parameters(&[0.5]).unwrap(), vec![2.0, 0.5]);
        assert_eq!(
            program.gradient_from_angle_adjoint(&[3.0, 4.0]).unwrap(),
            vec![4.0]
        );
    }
}
