use std::time::Instant;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tencir_pauli_core::{
    Clifford1, Clifford2, GateOperation, ParameterRef, ProductState, PropagationEngine,
    PropagationStats, RotationAxis,
};

use crate::convert::{build_canonical_operator, map_error, CanonicalizeOutput};

type ProfileOutput = (f64, usize, usize, usize, usize, Vec<usize>, f64);

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativePropagationEngine {
    engine: PropagationEngine,
}

#[pymethods]
impl NativePropagationEngine {
    #[getter]
    fn nqubits(&self) -> usize {
        self.engine.nqubits()
    }

    #[getter]
    fn nparameters(&self) -> usize {
        self.engine.nparameters()
    }

    #[getter]
    fn gate_count(&self) -> usize {
        self.engine.gate_count()
    }

    #[getter]
    fn max_weight(&self) -> Option<usize> {
        self.engine.max_weight()
    }

    #[getter]
    fn is_exact(&self) -> bool {
        self.engine.is_exact()
    }

    fn expectation<'py>(
        &self,
        py: Python<'py>,
        parameters: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        let values = parameters
            .as_slice()
            .map_err(|_| PyValueError::new_err("parameters must be C-contiguous"))?;
        py.allow_threads(|| self.engine.expectation(values))
            .map_err(map_error)
    }

    #[pyo3(signature = (parameters, checkpoint_interval=None))]
    fn value_and_grad<'py>(
        &self,
        py: Python<'py>,
        parameters: PyReadonlyArray1<'py, f64>,
        checkpoint_interval: Option<usize>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        let values = parameters
            .as_slice()
            .map_err(|_| PyValueError::new_err("parameters must be C-contiguous"))?;
        let result = py
            .allow_threads(|| self.engine.value_and_grad(values, checkpoint_interval))
            .map_err(map_error)?;
        Ok((result.value, PyArray1::from_vec(py, result.gradient)))
    }

    fn propagate_operator(
        &self,
        py: Python<'_>,
        parameters: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<CanonicalizeOutput> {
        let values = parameters
            .as_slice()
            .map_err(|_| PyValueError::new_err("parameters must be C-contiguous"))?;
        let result = py
            .allow_threads(|| self.engine.propagate(values))
            .map_err(map_error)?;
        Ok(materialize(result.terms))
    }

    fn profile(
        &self,
        py: Python<'_>,
        parameters: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<ProfileOutput> {
        let values = parameters
            .as_slice()
            .map_err(|_| PyValueError::new_err("parameters must be C-contiguous"))?;
        if !self.engine.is_hermitian_observable() {
            return Err(map_error(
                tencir_pauli_core::PauliError::NonHermitianExpectation,
            ));
        }
        let start = Instant::now();
        let result = py
            .allow_threads(|| self.engine.propagate(values))
            .map_err(map_error)?;
        let elapsed = start.elapsed().as_secs_f64();
        let stats: PropagationStats = result.stats;
        Ok((
            self.engine.expectation_of_terms(&result.terms),
            stats.initial_terms,
            stats.final_terms,
            stats.peak_terms,
            stats.estimated_peak_bytes,
            stats.final_weight_counts,
            elapsed,
        ))
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (nqubits, operations, structures, coefficients_re, coefficients_im, state_kind, state_bits, state_values, max_weight=None, max_bytes=None))]
pub(crate) fn pauli_propagation_engine(
    py: Python<'_>,
    nqubits: usize,
    operations: Vec<(u8, usize, usize, i64, f64, Vec<f64>)>,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    state_kind: u8,
    state_bits: Vec<u8>,
    state_values: Vec<f64>,
    max_weight: Option<usize>,
    max_bytes: Option<usize>,
) -> PyResult<NativePropagationEngine> {
    let engine = py.allow_threads(|| {
        let compiled = operations
            .into_iter()
            .map(|(kind, wire0, wire1, parameter, angle, matrix)| {
                compile_operation(nqubits, kind, wire0, wire1, parameter, angle, &matrix)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let operator =
            build_canonical_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
        let state = compile_state(nqubits, state_kind, state_bits, state_values)?;
        PropagationEngine::new(
            nqubits,
            compiled,
            operator,
            state,
            max_weight,
            max_bytes.map(|value| value as u128),
        )
        .map_err(map_error)
    })?;
    Ok(NativePropagationEngine { engine })
}

pub(crate) fn compile_operation(
    nqubits: usize,
    kind: u8,
    wire0: usize,
    wire1: usize,
    parameter: i64,
    angle: f64,
    matrix: &[f64],
) -> PyResult<GateOperation> {
    let result = match kind {
        0 => GateOperation::clifford1(nqubits, Clifford1::X, wire0),
        1 => GateOperation::clifford1(nqubits, Clifford1::Y, wire0),
        2 => GateOperation::clifford1(nqubits, Clifford1::Z, wire0),
        3 => GateOperation::clifford1(nqubits, Clifford1::H, wire0),
        4 => GateOperation::clifford1(nqubits, Clifford1::S, wire0),
        5 => GateOperation::clifford1(nqubits, Clifford1::Sdg, wire0),
        6 => GateOperation::clifford2(nqubits, Clifford2::Cnot, wire0, wire1),
        7 => GateOperation::clifford2(nqubits, Clifford2::Cz, wire0, wire1),
        8 => GateOperation::clifford2(nqubits, Clifford2::Swap, wire0, wire1),
        9..=11 => GateOperation::rotation(
            nqubits,
            match kind {
                9 => RotationAxis::X,
                10 => RotationAxis::Y,
                _ => RotationAxis::Z,
            },
            wire0,
            None,
            parameter_ref(parameter, angle)?,
        ),
        12..=14 => GateOperation::rotation(
            nqubits,
            match kind {
                12 => RotationAxis::X,
                13 => RotationAxis::Y,
                _ => RotationAxis::Z,
            },
            wire0,
            Some(wire1),
            parameter_ref(parameter, angle)?,
        ),
        15 => {
            let wires = if wire1 == usize::MAX {
                vec![wire0]
            } else {
                vec![wire0, wire1]
            };
            GateOperation::custom_ptm(nqubits, &wires, matrix)
        }
        _ => Err(tencir_pauli_core::PauliError::InvalidClifford {
            context: "unknown propagation gate kind",
        }),
    };
    result.map_err(map_error)
}

fn parameter_ref(parameter: i64, angle: f64) -> PyResult<ParameterRef> {
    if parameter == -1 {
        if !angle.is_finite() {
            return Err(PyValueError::new_err("rotation angle must be finite"));
        }
        Ok(ParameterRef::Static {
            cos: angle.cos(),
            sin: angle.sin(),
        })
    } else if parameter >= 0 {
        Ok(ParameterRef::Slot(parameter as usize))
    } else {
        Err(PyValueError::new_err(
            "rotation parameter must be -1 or a non-negative slot",
        ))
    }
}

pub(crate) fn compile_state(
    nqubits: usize,
    kind: u8,
    bits: Vec<u8>,
    values: Vec<f64>,
) -> PyResult<ProductState> {
    match kind {
        0 => Ok(ProductState::Zero),
        1 => Ok(ProductState::ComputationalBasis(bits)),
        2 => {
            let expected = nqubits
                .checked_mul(3)
                .ok_or_else(|| PyValueError::new_err("Bloch state dimension overflow"))?;
            if values.len() != expected {
                return Err(PyValueError::new_err(format!(
                    "expected {expected} Bloch entries, got {}",
                    values.len()
                )));
            }
            let vectors = values
                .chunks_exact(3)
                .map(|chunk| [chunk[0], chunk[1], chunk[2]])
                .collect();
            Ok(ProductState::Bloch(vectors))
        }
        _ => Err(PyValueError::new_err("unknown product-state kind")),
    }
}

fn materialize(terms: Vec<tencir_pauli_core::PauliTerm>) -> CanonicalizeOutput {
    let mut structures = Vec::with_capacity(terms.len());
    let mut real = Vec::with_capacity(terms.len());
    let mut imaginary = Vec::with_capacity(terms.len());
    for term in terms {
        structures.push(term.word.codes());
        real.push(term.coefficient.re);
        imaginary.push(term.coefficient.im);
    }
    (structures, real, imaginary)
}
