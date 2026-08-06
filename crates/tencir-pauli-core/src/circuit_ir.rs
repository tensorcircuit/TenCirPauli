//! Backend-neutral logical circuit representation.
//!
//! This module deliberately contains no sector, basis, or state-vector
//! assumptions. Execution backends consume the validated program and attach
//! their own compiled metadata.

use std::sync::Arc;

use crate::error::PauliError;
use crate::scalar::Complex64;

pub const CIRCUIT_SCHEMA_VERSION: u32 = 1;

/// A topologically ordered arithmetic expression node for a real gate angle.
#[derive(Clone, Debug, PartialEq)]
pub enum ParameterExprNode {
    Constant(f64),
    Slot(usize),
    Neg(usize),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
}

/// A supported logical gate. The angle index refers to a node in the shared
/// parameter program; static angles are represented by Constant nodes.
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

/// Validated logical circuit schema shared by future execution backends.
#[derive(Clone, Debug, PartialEq)]
pub struct CircuitProgram {
    schema_version: u32,
    nqubits: usize,
    operations: Arc<[CircuitGate]>,
    parameter_program: Arc<[ParameterExprNode]>,
    nparameters: usize,
}

impl CircuitProgram {
    pub fn new(
        schema_version: u32,
        nqubits: usize,
        operations: Vec<CircuitGate>,
        parameter_program: Vec<ParameterExprNode>,
        nparameters: usize,
    ) -> Result<Self, PauliError> {
        if schema_version != CIRCUIT_SCHEMA_VERSION {
            return Err(PauliError::InvalidCircuit {
                context: "unknown schema version",
            });
        }
        validate_parameter_program(&parameter_program, nparameters)?;
        for operation in &operations {
            validate_gate(operation, nqubits, parameter_program.len())?;
        }
        Ok(Self {
            schema_version,
            nqubits,
            operations: Arc::from(operations.into_boxed_slice()),
            parameter_program: Arc::from(parameter_program.into_boxed_slice()),
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

    pub fn parameter_program(&self) -> &[ParameterExprNode] {
        &self.parameter_program
    }

    pub fn nparameters(&self) -> usize {
        self.nparameters
    }

    /// Evaluate all expression nodes in one topological pass.
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
        let mut values: Vec<f64> = Vec::with_capacity(self.parameter_program.len());
        for node in self.parameter_program.iter() {
            let value = match *node {
                ParameterExprNode::Constant(value) => value,
                ParameterExprNode::Slot(slot) => parameters[slot],
                ParameterExprNode::Neg(child) => -values[child],
                ParameterExprNode::Add(left, right) => values[left] + values[right],
                ParameterExprNode::Sub(left, right) => values[left] - values[right],
                ParameterExprNode::Mul(left, right) => values[left] * values[right],
                ParameterExprNode::Div(left, right) => {
                    if values[right] == 0.0 {
                        return Err(PauliError::InvalidCircuit {
                            context: "parameter expression divides by zero",
                        });
                    }
                    values[left] / values[right]
                }
            };
            values.push(value);
        }
        Ok(values)
    }

    /// Reverse local angle derivatives through the shared expression DAG.
    pub fn reverse_parameter_program(
        &self,
        values: &[f64],
        node_adjoint: &[f64],
    ) -> Result<Vec<f64>, PauliError> {
        if values.len() != self.parameter_program.len()
            || node_adjoint.len() != self.parameter_program.len()
        {
            return Err(PauliError::InvalidCircuit {
                context: "parameter reverse buffers have the wrong length",
            });
        }
        let mut adjoint = node_adjoint.to_vec();
        let mut gradient = vec![0.0; self.nparameters];
        for index in (0..self.parameter_program.len()).rev() {
            let contribution = adjoint[index];
            match self.parameter_program[index] {
                ParameterExprNode::Constant(_) => {}
                ParameterExprNode::Slot(slot) => gradient[slot] += contribution,
                ParameterExprNode::Neg(child) => adjoint[child] -= contribution,
                ParameterExprNode::Add(left, right) => {
                    adjoint[left] += contribution;
                    adjoint[right] += contribution;
                }
                ParameterExprNode::Sub(left, right) => {
                    adjoint[left] += contribution;
                    adjoint[right] -= contribution;
                }
                ParameterExprNode::Mul(left, right) => {
                    adjoint[left] += contribution * values[right];
                    adjoint[right] += contribution * values[left];
                }
                ParameterExprNode::Div(left, right) => {
                    if values[right] == 0.0 {
                        return Err(PauliError::InvalidCircuit {
                            context: "parameter expression divides by zero",
                        });
                    }
                    adjoint[left] += contribution / values[right];
                    adjoint[right] -= contribution * values[left] / values[right].powi(2);
                }
            }
        }
        Ok(gradient)
    }
}

fn validate_parameter_program(
    nodes: &[ParameterExprNode],
    nparameters: usize,
) -> Result<(), PauliError> {
    let mut observed_slots = vec![false; nparameters];
    for (index, node) in nodes.iter().enumerate() {
        let operands = match *node {
            ParameterExprNode::Constant(value) => {
                if !value.is_finite() {
                    return Err(PauliError::InvalidCircuit {
                        context: "parameter expression constant is non-finite",
                    });
                }
                None
            }
            ParameterExprNode::Slot(slot) => {
                if slot >= nparameters {
                    return Err(PauliError::InvalidCircuit {
                        context: "parameter slot is outside the declared range",
                    });
                }
                observed_slots[slot] = true;
                None
            }
            ParameterExprNode::Neg(child) => Some([child, 0]),
            ParameterExprNode::Add(left, right)
            | ParameterExprNode::Sub(left, right)
            | ParameterExprNode::Mul(left, right)
            | ParameterExprNode::Div(left, right) => Some([left, right]),
        };
        if let Some(operands) = operands {
            let count = if matches!(node, ParameterExprNode::Neg(_)) {
                1
            } else {
                2
            };
            if operands[..count].iter().any(|child| *child >= index) {
                return Err(PauliError::InvalidCircuit {
                    context: "parameter expression is not topologically ordered",
                });
            }
        }
    }
    if observed_slots.iter().any(|observed| !observed) {
        return Err(PauliError::InvalidCircuit {
            context: "parameter slots must cover 0..nparameters-1 without holes",
        });
    }
    Ok(())
}

fn validate_gate(
    gate: &CircuitGate,
    nqubits: usize,
    expression_count: usize,
) -> Result<(), PauliError> {
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
        if angle >= expression_count {
            Err(PauliError::InvalidCircuit {
                context: "gate angle node is outside the parameter program",
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
    fn expression_program_evaluates_and_reverses() {
        let program = CircuitProgram::new(
            CIRCUIT_SCHEMA_VERSION,
            4,
            vec![CircuitGate::Rz { wire: 0, angle: 3 }],
            vec![
                ParameterExprNode::Slot(0),
                ParameterExprNode::Constant(2.0),
                ParameterExprNode::Neg(1),
                ParameterExprNode::Add(0, 2),
            ],
            1,
        )
        .unwrap();
        let values = program.evaluate_parameters(&[0.5]).unwrap();
        assert_eq!(values, vec![0.5, 2.0, -2.0, -1.5]);
        let gradient = program
            .reverse_parameter_program(&values, &[0.0, 0.0, 0.0, 1.0])
            .unwrap();
        assert_eq!(gradient, vec![1.0]);
    }
}
