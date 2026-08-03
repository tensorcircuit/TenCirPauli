use numpy::PyArray1;
use pyo3::prelude::*;
use tencir_pauli_core::{build_charge_sector_plan, ChargeSectorPlan};

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeChargeSectorPlan {
    plan: ChargeSectorPlan,
}

#[pymethods]
impl NativeChargeSectorPlan {
    #[getter]
    fn dimension(&self) -> usize {
        self.plan.dimension()
    }

    #[getter]
    fn estimated_bytes(&self) -> u128 {
        self.plan.estimated_bytes()
    }

    fn rank(&self, occupations: Vec<u64>) -> PyResult<u64> {
        self.plan
            .rank(&occupations)
            .map_err(crate::convert::map_error)
    }

    fn unrank(&self, index: u64) -> PyResult<Vec<u64>> {
        self.plan.unrank(index).map_err(crate::convert::map_error)
    }

    fn basis_states<'py>(
        &self,
        py: Python<'py>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let values = py
            .allow_threads(|| self.plan.basis_states(max_bytes))
            .map_err(crate::convert::map_error)?;
        Ok(PyArray1::from_vec(py, values))
    }
}

#[pyfunction]
pub(crate) fn charge_sector_plan(
    py: Python<'_>,
    local_dimensions: Vec<usize>,
    contributions: Vec<Vec<Vec<i128>>>,
    target: Vec<i128>,
    max_bytes: u128,
) -> PyResult<NativeChargeSectorPlan> {
    let plan = py
        .allow_threads(|| {
            build_charge_sector_plan(local_dimensions, contributions, target, max_bytes)
        })
        .map_err(crate::convert::map_error)?;
    Ok(NativeChargeSectorPlan { plan })
}
