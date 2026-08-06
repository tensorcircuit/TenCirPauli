use pyo3::prelude::*;
use tencir_pauli_core::{build_mapping_plan, MappingPlan};

use crate::convert::map_error;
use crate::majorana::NativeMajoranaOperatorHandle;
use crate::operator::NativePauliOperatorHandle;
use crate::structured::NativeHybridOperatorHandle;

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeMappingPlan {
    plan: MappingPlan,
}

#[pymethods]
impl NativeMappingPlan {
    #[getter]
    fn n_modes(&self) -> usize {
        self.plan.n_modes()
    }

    fn encoding_flat(&self) -> Vec<usize> {
        self.plan
            .encoding()
            .iter()
            .flat_map(|row| row.iter().map(|&value| usize::from(value)))
            .collect()
    }

    fn inverse_encoding_flat(&self) -> Vec<usize> {
        self.plan
            .inverse_encoding()
            .iter()
            .flat_map(|row| row.iter().map(|&value| usize::from(value)))
            .collect()
    }

    #[getter]
    fn cnot_operations(&self) -> Vec<(usize, usize)> {
        self.plan.cnot_operations().to_vec()
    }

    #[getter]
    fn cnot_count(&self) -> usize {
        self.plan.cnot_operations().len()
    }

    #[getter]
    fn estimated_bytes(&self) -> u128 {
        self.plan.estimated_bytes()
    }

    fn encode_occupation(&self, py: Python<'_>, occupation: Vec<u8>) -> PyResult<Vec<u8>> {
        py.allow_threads(|| self.plan.encode_occupation(&occupation))
            .map_err(map_error)
    }

    fn transform_pauli_handle(
        &self,
        py: Python<'_>,
        handle: &NativePauliOperatorHandle,
        max_bytes: u128,
    ) -> PyResult<NativePauliOperatorHandle> {
        let operator = py
            .allow_threads(|| self.plan.map_pauli_operator(handle.core(), max_bytes))
            .map_err(map_error)?;
        Ok(NativePauliOperatorHandle::from_operator(operator))
    }

    fn transform_pauli_handle_prefix(
        &self,
        py: Python<'_>,
        handle: &NativePauliOperatorHandle,
        prefix_length: usize,
        max_bytes: u128,
    ) -> PyResult<NativePauliOperatorHandle> {
        let operator = py
            .allow_threads(|| {
                self.plan
                    .map_pauli_operator_prefix(handle.core(), prefix_length, max_bytes)
            })
            .map_err(map_error)?;
        Ok(NativePauliOperatorHandle::from_operator(operator))
    }

    fn transform_majorana_handle(
        &self,
        py: Python<'_>,
        handle: &NativeMajoranaOperatorHandle,
        max_bytes: u128,
    ) -> PyResult<NativePauliOperatorHandle> {
        let (indices, coefficients) = handle.native_parts();
        let operator = py
            .allow_threads(|| {
                self.plan
                    .map_majorana_terms(indices, coefficients, max_bytes)
            })
            .map_err(map_error)?;
        Ok(NativePauliOperatorHandle::from_operator(operator))
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_hybrid_handle(
        &self,
        py: Python<'_>,
        handle: &NativeHybridOperatorHandle,
        max_bytes: u128,
    ) -> PyResult<NativeHybridOperatorHandle> {
        let layout = handle.layout();
        let result = py
            .allow_threads(|| {
                self.plan
                    .map_hybrid_terms(layout, handle.batch(), max_bytes)
            })
            .map_err(map_error)?;
        Ok(NativeHybridOperatorHandle::from_result(layout, result))
    }
}

#[pyfunction]
pub(crate) fn mapping_plan(
    py: Python<'_>,
    mapping: String,
    n_modes: usize,
    max_bytes: u128,
) -> PyResult<NativeMappingPlan> {
    let plan = py
        .allow_threads(|| build_mapping_plan(&mapping, n_modes, max_bytes))
        .map_err(map_error)?;
    Ok(NativeMappingPlan { plan })
}
