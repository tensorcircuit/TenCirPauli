//! Thin PyO3 boundary for the Rust-native U(1) circuit plan.

use numpy::{Complex64 as NumpyComplex128, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tencir_pauli_core::{
    CircuitGate, CircuitProgram, Complex64, ParameterExprNode, U1CircuitPlan, U1Sector,
    CIRCUIT_SCHEMA_VERSION,
};

use crate::convert::{build_canonical_operator, map_error};

type NativeExpressionNode = (u8, usize, usize, f64);
type NativeGate = (u8, usize, usize, usize, Vec<usize>, Vec<f64>, Vec<f64>);

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeU1CircuitPlan {
    plan: U1CircuitPlan,
}

#[pymethods]
impl NativeU1CircuitPlan {
    #[getter]
    fn nqubits(&self) -> usize {
        self.plan.nqubits()
    }

    #[getter]
    fn particle_number(&self) -> usize {
        self.plan.sector().particle_number()
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.plan.dimension()
    }

    #[getter]
    fn nparameters(&self) -> usize {
        self.plan.nparameters()
    }

    #[getter]
    fn gate_count(&self) -> usize {
        self.plan.gate_count()
    }

    fn run<'py>(
        &self,
        py: Python<'py>,
        initial_state: PyReadonlyArray1<'py, NumpyComplex128>,
        parameters: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let initial = initial_state
            .as_slice()
            .map_err(|_| PyValueError::new_err("initial_state must be C-contiguous"))?;
        let parameters = parameters
            .as_slice()
            .map_err(|_| PyValueError::new_err("parameters must be C-contiguous"))?;
        let state = py
            .allow_threads(|| self.plan.run(initial, parameters))
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, state))
    }

    fn probability<'py>(
        &self,
        py: Python<'py>,
        initial_state: PyReadonlyArray1<'py, NumpyComplex128>,
        parameters: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let initial = initial_state
            .as_slice()
            .map_err(|_| PyValueError::new_err("initial_state must be C-contiguous"))?;
        let parameters = parameters
            .as_slice()
            .map_err(|_| PyValueError::new_err("parameters must be C-contiguous"))?;
        let probability = py
            .allow_threads(|| self.plan.probability(initial, parameters))
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, probability))
    }

    fn to_dense<'py>(
        &self,
        py: Python<'py>,
        initial_state: PyReadonlyArray1<'py, NumpyComplex128>,
        parameters: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let initial = initial_state
            .as_slice()
            .map_err(|_| PyValueError::new_err("initial_state must be C-contiguous"))?;
        let parameters = parameters
            .as_slice()
            .map_err(|_| PyValueError::new_err("parameters must be C-contiguous"))?;
        let state = py
            .allow_threads(|| self.plan.to_dense(initial, parameters))
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, state))
    }

    fn probability_full<'py>(
        &self,
        py: Python<'py>,
        initial_state: PyReadonlyArray1<'py, NumpyComplex128>,
        parameters: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let initial = initial_state
            .as_slice()
            .map_err(|_| PyValueError::new_err("initial_state must be C-contiguous"))?;
        let parameters = parameters
            .as_slice()
            .map_err(|_| PyValueError::new_err("parameters must be C-contiguous"))?;
        let probability = py
            .allow_threads(|| self.plan.probability_full(initial, parameters))
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, probability))
    }

    #[pyo3(signature = (initial_state, structures, coefficients_re, coefficients_im, parameters))]
    fn expectation(
        &self,
        py: Python<'_>,
        initial_state: PyReadonlyArray1<'_, NumpyComplex128>,
        structures: Vec<Vec<u8>>,
        coefficients_re: Vec<f64>,
        coefficients_im: Vec<f64>,
        parameters: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<(f64, f64)> {
        let initial = initial_state
            .as_slice()
            .map_err(|_| PyValueError::new_err("initial_state must be C-contiguous"))?;
        let parameters = parameters
            .as_slice()
            .map_err(|_| PyValueError::new_err("parameters must be C-contiguous"))?;
        let observable = build_canonical_operator(
            self.plan.nqubits(),
            &structures,
            &coefficients_re,
            &coefficients_im,
        )?;
        let value = py
            .allow_threads(|| self.plan.expectation(initial, &observable, parameters))
            .map_err(map_error)?;
        Ok((value.re, value.im))
    }

    #[pyo3(signature = (initial_state, structures, coefficients_re, coefficients_im, parameters))]
    fn value_and_grad<'py>(
        &self,
        py: Python<'py>,
        initial_state: PyReadonlyArray1<'py, NumpyComplex128>,
        structures: Vec<Vec<u8>>,
        coefficients_re: Vec<f64>,
        coefficients_im: Vec<f64>,
        parameters: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
        let initial = initial_state
            .as_slice()
            .map_err(|_| PyValueError::new_err("initial_state must be C-contiguous"))?;
        let parameters = parameters
            .as_slice()
            .map_err(|_| PyValueError::new_err("parameters must be C-contiguous"))?;
        let observable = build_canonical_operator(
            self.plan.nqubits(),
            &structures,
            &coefficients_re,
            &coefficients_im,
        )?;
        let (value, gradient) = py
            .allow_threads(|| self.plan.value_and_grad(initial, &observable, parameters))
            .map_err(map_error)?;
        Ok((value, PyArray1::from_vec(py, gradient)))
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (nqubits, particle_number, expression_nodes, gates, max_bytes))]
pub(crate) fn u1_circuit_plan(
    py: Python<'_>,
    nqubits: usize,
    particle_number: usize,
    expression_nodes: Vec<NativeExpressionNode>,
    gates: Vec<NativeGate>,
    max_bytes: usize,
) -> PyResult<NativeU1CircuitPlan> {
    let plan = py.allow_threads(|| {
        let nparameters = expression_nodes
            .iter()
            .filter(|(opcode, _, _, _)| *opcode == 1)
            .map(|(_, slot, _, _)| *slot)
            .max()
            .map_or(0, |slot| slot + 1);
        let expressions = expression_nodes
            .into_iter()
            .map(|(opcode, left, right, constant)| match opcode {
                0 => Ok(ParameterExprNode::Constant(constant)),
                1 => Ok(ParameterExprNode::Slot(left)),
                2 => Ok(ParameterExprNode::Neg(left)),
                3 => Ok(ParameterExprNode::Add(left, right)),
                4 => Ok(ParameterExprNode::Sub(left, right)),
                5 => Ok(ParameterExprNode::Mul(left, right)),
                6 => Ok(ParameterExprNode::Div(left, right)),
                _ => Err(PyValueError::new_err("unknown circuit expression opcode")),
            })
            .collect::<PyResult<Vec<_>>>()?;
        let operations = gates
            .into_iter()
            .map(
                |(opcode, wire0, wire1, angle, wires, payload_re, payload_im)| {
                    let gate = match opcode {
                        0 => CircuitGate::Rz { wire: wire0, angle },
                        1 => CircuitGate::Rzz {
                            wire0,
                            wire1,
                            angle,
                        },
                        2 => CircuitGate::Cz { wire0, wire1 },
                        3 => CircuitGate::Cphase {
                            wire0,
                            wire1,
                            angle,
                        },
                        4 => CircuitGate::Swap { wire0, wire1 },
                        5 => CircuitGate::Iswap {
                            wire0,
                            wire1,
                            angle,
                        },
                        6 => {
                            if payload_re.len() != payload_im.len() {
                                return Err(PyValueError::new_err(
                                    "diagonal payload real/imaginary lengths differ",
                                ));
                            }
                            CircuitGate::Diagonal {
                                wires,
                                payload: payload_re
                                    .into_iter()
                                    .zip(payload_im)
                                    .map(|(re, im)| Complex64::new(re, im))
                                    .collect(),
                            }
                        }
                        _ => return Err(PyValueError::new_err("unknown circuit gate opcode")),
                    };
                    Ok(gate)
                },
            )
            .collect::<PyResult<Vec<_>>>()?;
        let program = CircuitProgram::new(
            CIRCUIT_SCHEMA_VERSION,
            nqubits,
            operations,
            expressions,
            nparameters,
        )
        .map_err(map_error)?;
        let sector = U1Sector::new(nqubits, particle_number).map_err(map_error)?;
        let plan =
            U1CircuitPlan::compile(program, sector, Some(max_bytes as u128)).map_err(map_error)?;
        Ok::<_, PyErr>(plan)
    })?;
    Ok(NativeU1CircuitPlan { plan })
}
