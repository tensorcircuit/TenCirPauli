//! Validated gate-tape building blocks used by Pauli propagation.

use crate::error::PauliError;

/// One-qubit Clifford gate supported by the propagation engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Clifford1 {
    X,
    Y,
    Z,
    H,
    S,
    Sdg,
}

/// Two-qubit Clifford gate supported by the propagation engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Clifford2 {
    Cnot,
    Cz,
    Swap,
}

/// Pauli generator for an analytic rotation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationAxis {
    X,
    Y,
    Z,
}

/// A static angle or a reference to a runtime parameter slot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParameterRef {
    Static { cos: f64, sin: f64 },
    Slot(usize),
}

/// An immutable, validated operation in a compiled tape.
#[derive(Clone, Debug)]
pub struct GateOperation {
    pub(crate) kind: GateKind,
}

#[derive(Clone, Debug)]
pub(crate) enum GateKind {
    Clifford1 {
        gate: Clifford1,
        wire: usize,
    },
    Clifford2 {
        gate: Clifford2,
        wire0: usize,
        wire1: usize,
    },
    Rotation {
        axis: RotationAxis,
        wire0: usize,
        wire1: Option<usize>,
        parameter: ParameterRef,
    },
    CustomPtm {
        wire0: usize,
        wire1: Option<usize>,
        transitions: Vec<Vec<(u8, f64)>>,
    },
}

impl GateOperation {
    /// Construct a validated one-qubit Clifford operation.
    pub fn clifford1(nqubits: usize, gate: Clifford1, wire: usize) -> Result<Self, PauliError> {
        validate_wire(nqubits, wire)?;
        Ok(Self {
            kind: GateKind::Clifford1 { gate, wire },
        })
    }

    /// Construct a validated two-qubit Clifford operation.
    pub fn clifford2(
        nqubits: usize,
        gate: Clifford2,
        wire0: usize,
        wire1: usize,
    ) -> Result<Self, PauliError> {
        validate_two_wires(nqubits, wire0, wire1)?;
        Ok(Self {
            kind: GateKind::Clifford2 { gate, wire0, wire1 },
        })
    }

    /// Construct a validated analytic one- or two-qubit Pauli rotation.
    pub fn rotation(
        nqubits: usize,
        axis: RotationAxis,
        wire0: usize,
        wire1: Option<usize>,
        parameter: ParameterRef,
    ) -> Result<Self, PauliError> {
        validate_wire(nqubits, wire0)?;
        if let Some(second) = wire1 {
            validate_two_wires(nqubits, wire0, second)?;
        }
        if matches!(parameter, ParameterRef::Static { .. }) {
            let angle = match parameter {
                ParameterRef::Static { cos, sin } => (cos, sin),
                ParameterRef::Slot(_) => unreachable!(),
            };
            if !angle.0.is_finite() || !angle.1.is_finite() {
                return Err(PauliError::NonFiniteParameter { index: 0 });
            }
        }
        Ok(Self {
            kind: GateKind::Rotation {
                axis,
                wire0,
                wire1,
                parameter,
            },
        })
    }

    /// Compile a real Pauli-transfer matrix into sparse exact-nonzero rows.
    pub fn custom_ptm(nqubits: usize, wires: &[usize], matrix: &[f64]) -> Result<Self, PauliError> {
        let (wire0, wire1) = match wires {
            [first] => (*first, None),
            [first, second] => (*first, Some(*second)),
            _ => {
                return Err(PauliError::InvalidPtmShape {
                    expected: 1,
                    actual: wires.len(),
                })
            }
        };
        validate_wire(nqubits, wire0)?;
        if let Some(second) = wire1 {
            validate_two_wires(nqubits, wire0, second)?;
        }
        let local_dimension = if wire1.is_some() { 16 } else { 4 };
        let expected = local_dimension * local_dimension;
        if matrix.len() != expected {
            return Err(PauliError::InvalidPtmShape {
                expected,
                actual: matrix.len(),
            });
        }
        if let Some(index) = matrix.iter().position(|value| !value.is_finite()) {
            return Err(PauliError::NonFinitePtm { index });
        }
        let transitions = (0..local_dimension)
            .map(|input| {
                (0..local_dimension)
                    .filter_map(|output| {
                        let value = matrix[output * local_dimension + input];
                        (value != 0.0).then_some((output as u8, value))
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        Ok(Self {
            kind: GateKind::CustomPtm {
                wire0,
                wire1,
                transitions,
            },
        })
    }

    pub(crate) fn parameter_slot(&self) -> Option<usize> {
        match self.kind {
            GateKind::Rotation {
                parameter: ParameterRef::Slot(slot),
                ..
            } => Some(slot),
            _ => None,
        }
    }
}

fn validate_wire(nqubits: usize, wire: usize) -> Result<(), PauliError> {
    if wire >= nqubits {
        return Err(PauliError::InvalidWire { wire, nqubits });
    }
    Ok(())
}

fn validate_two_wires(nqubits: usize, wire0: usize, wire1: usize) -> Result<(), PauliError> {
    validate_wire(nqubits, wire0)?;
    validate_wire(nqubits, wire1)?;
    if wire0 == wire1 {
        return Err(PauliError::DuplicateWire);
    }
    Ok(())
}
