use std::sync::Arc;

use numpy::{Complex64 as NumpyComplex128, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;
use tencir_pauli_core::{
    apply_charge_csr_into, apply_charge_mvp_from_prepared_plan,
    apply_charge_mvp_from_prepared_plan_into, build_compact_charge_sector_plan,
    build_fast_fermion_mvp_plan, compile_charge_transitions_from_prepared_plan,
    estimate_charge_transition_terms_bytes, prepare_charge_transition_plan_layout,
    ChargeSectorPlan, ChargeTransitionTerm, Complex64, FastFermionMvpPlan, PauliError,
    PreparedChargeTransitionPlanLayout,
};

use crate::convert::map_error;
use crate::operator::NativePauliOperatorHandle;
use crate::structured::NativeHybridOperatorHandle;

type ChargeCsrArrays<'py> = (
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);
type ChargeCooArrays<'py> = (
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);

fn charge_plan_base_bytes(
    plan: &ChargeSectorPlan,
    terms: &[ChargeTransitionTerm],
) -> Result<u128, PauliError> {
    plan.estimated_bytes()
        .checked_add(estimate_charge_transition_terms_bytes(terms)?)
        .ok_or(PauliError::Overflow {
            context: "estimating native charge MVP plan",
        })
}

fn charge_remaining_budget(max_bytes: u128, base_bytes: u128) -> Result<u128, PauliError> {
    max_bytes
        .checked_sub(base_bytes)
        .ok_or(PauliError::MemoryLimit {
            requested: base_bytes,
            limit: max_bytes,
        })
}

fn charge_terms_from_pauli_handle(handle: &NativePauliOperatorHandle) -> Vec<ChargeTransitionTerm> {
    handle
        .core()
        .terms()
        .iter()
        .map(|term| ChargeTransitionTerm {
            fermion_creation: Vec::new(),
            fermion_annihilation: Vec::new(),
            boson_blocks: Vec::new(),
            qubit_codes: term.word.codes(),
            mapped_present: false,
            mapped_codes: vec![0; 0],
            qudit_present: false,
            qudit_triples: Vec::new(),
            coefficient: term.coefficient,
        })
        .collect()
}

fn charge_terms_from_hybrid_handle(
    handle: &NativeHybridOperatorHandle,
) -> Vec<ChargeTransitionTerm> {
    let batch = handle.batch();
    (0..batch.coefficients.len())
        .map(|index| ChargeTransitionTerm {
            fermion_creation: batch.fermion_creation[index].clone(),
            fermion_annihilation: batch.fermion_annihilation[index].clone(),
            boson_blocks: batch.boson_blocks[index].clone(),
            qubit_codes: batch.qubit_codes[index].clone(),
            mapped_present: batch.mapped_present[index],
            mapped_codes: batch.mapped_codes[index].clone(),
            qudit_present: batch.qudit_present[index],
            qudit_triples: batch.qudit_triples[index].clone(),
            coefficient: batch.coefficients[index],
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn build_native_charge_mvp_plan(
    plan: Arc<ChargeSectorPlan>,
    dimension: usize,
    local_dimensions: Vec<u64>,
    fermion_positions: Vec<u64>,
    boson_positions: Vec<u64>,
    qubit_positions: Vec<u64>,
    qudit_positions: Vec<u64>,
    qudit_dimension: u64,
    terms: Vec<ChargeTransitionTerm>,
    termwise_conserved: bool,
    max_bytes: u128,
    fast_fermion_particles: Option<usize>,
) -> PyResult<NativeChargeMvpPlan> {
    if dimension != plan.dimension() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "charge MVP dimension does not match its sector plan",
        ));
    }
    let base_bytes = charge_plan_base_bytes(&plan, &terms).map_err(map_error)?;
    let layout_budget = charge_remaining_budget(max_bytes, base_bytes).map_err(map_error)?;
    let layout = prepare_charge_transition_plan_layout(
        &plan,
        dimension,
        local_dimensions.clone(),
        &fermion_positions,
        &boson_positions,
        &qubit_positions,
        &qudit_positions,
        qudit_dimension,
        &terms,
        layout_budget,
    )
    .map_err(map_error)?;
    let fast_budget =
        charge_remaining_budget(layout_budget, layout.estimated_bytes()).map_err(map_error)?;
    let fast_fermion_plan = if termwise_conserved {
        build_fast_fermion_mvp_plan(
            &plan,
            &local_dimensions,
            &fermion_positions,
            &boson_positions,
            &qubit_positions,
            &qudit_positions,
            &terms,
            fast_fermion_particles,
            fast_budget,
        )
        .map_err(map_error)?
    } else {
        None
    };
    Ok(NativeChargeMvpPlan {
        plan,
        layout,
        terms,
        termwise_conserved,
        fast_fermion_plan,
    })
}

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeChargeSectorPlan {
    plan: Arc<ChargeSectorPlan>,
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

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        handle,
        dimension,
        local_dimensions,
        fermion_positions,
        boson_positions,
        qubit_positions,
        qudit_positions,
        termwise_conserved,
        max_bytes,
        fast_fermion_particles=None
    ))]
    fn compile_mvp_pauli_handle(
        &self,
        py: Python<'_>,
        handle: &NativePauliOperatorHandle,
        dimension: usize,
        local_dimensions: Vec<u64>,
        fermion_positions: Vec<u64>,
        boson_positions: Vec<u64>,
        qubit_positions: Vec<u64>,
        qudit_positions: Vec<u64>,
        termwise_conserved: bool,
        max_bytes: u128,
        fast_fermion_particles: Option<usize>,
    ) -> PyResult<NativeChargeMvpPlan> {
        let plan = Arc::clone(&self.plan);
        py.allow_threads(move || {
            build_native_charge_mvp_plan(
                plan,
                dimension,
                local_dimensions,
                fermion_positions,
                boson_positions,
                qubit_positions,
                qudit_positions,
                0,
                charge_terms_from_pauli_handle(handle),
                termwise_conserved,
                max_bytes,
                fast_fermion_particles,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        handle,
        dimension,
        local_dimensions,
        fermion_positions,
        boson_positions,
        qubit_positions,
        qudit_positions,
        qudit_dimension,
        termwise_conserved,
        max_bytes,
        fast_fermion_particles=None
    ))]
    fn compile_mvp_hybrid_handle(
        &self,
        py: Python<'_>,
        handle: &NativeHybridOperatorHandle,
        dimension: usize,
        local_dimensions: Vec<u64>,
        fermion_positions: Vec<u64>,
        boson_positions: Vec<u64>,
        qubit_positions: Vec<u64>,
        qudit_positions: Vec<u64>,
        qudit_dimension: u64,
        termwise_conserved: bool,
        max_bytes: u128,
        fast_fermion_particles: Option<usize>,
    ) -> PyResult<NativeChargeMvpPlan> {
        let plan = Arc::clone(&self.plan);
        py.allow_threads(move || {
            build_native_charge_mvp_plan(
                plan,
                dimension,
                local_dimensions,
                fermion_positions,
                boson_positions,
                qubit_positions,
                qudit_positions,
                qudit_dimension,
                charge_terms_from_hybrid_handle(handle),
                termwise_conserved,
                max_bytes,
                fast_fermion_particles,
            )
        })
    }
}

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeChargeMvpPlan {
    plan: Arc<ChargeSectorPlan>,
    layout: PreparedChargeTransitionPlanLayout,
    terms: Vec<ChargeTransitionTerm>,
    termwise_conserved: bool,
    fast_fermion_plan: Option<FastFermionMvpPlan>,
}

#[pymethods]
impl NativeChargeMvpPlan {
    #[getter]
    fn dimension(&self) -> usize {
        self.plan.dimension()
    }

    #[getter]
    fn estimated_bytes(&self) -> u128 {
        self.plan
            .estimated_bytes()
            .saturating_add(
                estimate_charge_transition_terms_bytes(&self.terms).unwrap_or(u128::MAX),
            )
            .saturating_add(self.layout.estimated_bytes())
            .saturating_add(
                self.fast_fermion_plan
                    .as_ref()
                    .map_or(0, FastFermionMvpPlan::estimated_bytes),
            )
    }

    fn apply<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let state_values = state.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge state must be C-contiguous")
        })?;
        if state_values.len() != self.dimension() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "charge state must have shape ({},), got ({},)",
                self.dimension(),
                state_values.len()
            )));
        }
        let result = py
            .allow_threads(|| {
                apply_charge_mvp_from_prepared_plan(
                    &self.plan,
                    &self.layout,
                    &self.terms,
                    state_values,
                    self.termwise_conserved,
                    self.fast_fermion_plan.as_ref(),
                    max_bytes,
                )
            })
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, result))
    }

    fn apply_into<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        mut output: PyReadwriteArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<()> {
        let state_values = state.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge state must be C-contiguous")
        })?;
        let output_values = output.as_slice_mut().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge output must be C-contiguous")
        })?;
        let output_bytes = (self.dimension() as u128)
            .checked_mul(std::mem::size_of::<NumpyComplex128>() as u128)
            .ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge MVP output size overflow")
            })?;
        let effective_max = max_bytes.saturating_add(output_bytes);
        py.allow_threads(|| {
            apply_charge_mvp_from_prepared_plan_into(
                &self.plan,
                &self.layout,
                &self.terms,
                state_values,
                output_values,
                self.termwise_conserved,
                self.fast_fermion_plan.as_ref(),
                effective_max,
            )
        })
        .map_err(map_error)
    }

    fn compile_eager<'py>(
        &self,
        py: Python<'py>,
        max_bytes: u128,
    ) -> PyResult<NativeChargeEagerMvpPlan> {
        py.allow_threads(|| {
            let result = compile_charge_transitions_from_prepared_plan(
                &self.plan,
                &self.layout,
                &self.terms,
                max_bytes,
            )
            .map_err(map_error)?;
            NativeChargeEagerMvpPlan::from_coo(self.dimension(), result, max_bytes)
        })
    }
}

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeChargeEagerMvpPlan {
    dimension: usize,
    indptr: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<Complex64>,
}

impl NativeChargeEagerMvpPlan {
    fn from_coo(
        dimension: usize,
        (rows, columns, values): (Vec<u64>, Vec<u64>, Vec<Complex64>),
        max_bytes: u128,
    ) -> PyResult<Self> {
        if rows.len() != columns.len() || rows.len() != values.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "restricted transition arrays have inconsistent lengths",
            ));
        }
        let pointer_count = dimension.checked_add(1).ok_or_else(|| {
            pyo3::exceptions::PyMemoryError::new_err("charge CSR dimension overflow")
        })?;
        let logical_bytes = (pointer_count as u128)
            .checked_mul(std::mem::size_of::<usize>() as u128)
            .and_then(|value| {
                value.checked_add(
                    (columns.len() as u128).checked_mul(std::mem::size_of::<usize>() as u128)?,
                )
            })
            .and_then(|value| {
                value.checked_add(
                    (values.len() as u128).checked_mul(std::mem::size_of::<Complex64>() as u128)?,
                )
            })
            .ok_or_else(|| pyo3::exceptions::PyMemoryError::new_err("charge CSR size overflow"))?;
        if logical_bytes > max_bytes {
            return Err(pyo3::exceptions::PyMemoryError::new_err(format!(
                "charge eager CSR requires approximately {logical_bytes} bytes, exceeding max_bytes={max_bytes}"
            )));
        }
        let mut indptr = vec![0usize; pointer_count];
        let mut checked_columns = Vec::with_capacity(columns.len());
        for (&row, &column) in rows.iter().zip(&columns) {
            let row = usize::try_from(row).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("restricted row index overflow")
            })?;
            let column = usize::try_from(column).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("restricted column index overflow")
            })?;
            if row >= dimension || column >= dimension {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "restricted transition index is outside the sector dimension",
                ));
            }
            indptr[row + 1] = indptr[row + 1].checked_add(1).ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge CSR offset overflow")
            })?;
            checked_columns.push(column);
        }
        for row in 0..dimension {
            indptr[row + 1] = indptr[row + 1].checked_add(indptr[row]).ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge CSR offset overflow")
            })?;
        }
        Ok(Self {
            dimension,
            indptr,
            columns: checked_columns,
            values,
        })
    }

    fn apply_values(
        &self,
        state: &[Complex64],
        output: &mut [Complex64],
        parallel: bool,
    ) -> PyResult<()> {
        apply_charge_csr_into(
            &self.indptr,
            &self.columns,
            &self.values,
            state,
            output,
            parallel,
        )
        .map_err(map_error)
    }
}

#[pymethods]
impl NativeChargeEagerMvpPlan {
    #[getter]
    fn dimension(&self) -> usize {
        self.dimension
    }

    #[getter]
    fn transition_count(&self) -> usize {
        self.values.len()
    }

    #[getter]
    fn estimated_bytes(&self) -> u128 {
        (self.indptr.len() * std::mem::size_of::<usize>()
            + self.columns.len() * std::mem::size_of::<usize>()
            + self.values.len() * std::mem::size_of::<Complex64>()) as u128
    }

    fn csr<'py>(&self, py: Python<'py>) -> ChargeCsrArrays<'py> {
        let indptr = self.indptr.iter().map(|&value| value as u64).collect();
        let columns = self.columns.iter().map(|&value| value as u64).collect();
        (
            PyArray1::from_vec(py, indptr),
            PyArray1::from_vec(py, columns),
            PyArray1::from_vec(py, self.values.clone()),
        )
    }

    fn coo<'py>(&self, py: Python<'py>, max_bytes: u128) -> PyResult<ChargeCooArrays<'py>> {
        let transition_bytes = (self.values.len() as u128)
            .checked_mul(32)
            .ok_or_else(|| pyo3::exceptions::PyMemoryError::new_err("charge COO size overflow"))?;
        if transition_bytes > max_bytes {
            return Err(pyo3::exceptions::PyMemoryError::new_err(format!(
                "charge COO requires approximately {transition_bytes} bytes, exceeding max_bytes={max_bytes}"
            )));
        }
        let mut rows = Vec::with_capacity(self.values.len());
        for row in 0..self.dimension {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            rows.extend(std::iter::repeat_n(row as u64, end - start));
        }
        let columns = self.columns.iter().map(|&value| value as u64).collect();
        Ok((
            PyArray1::from_vec(py, rows),
            PyArray1::from_vec(py, columns),
            PyArray1::from_vec(py, self.values.clone()),
        ))
    }

    fn dense<'py>(
        &self,
        py: Python<'py>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let entries = (self.dimension as u128)
            .checked_mul(self.dimension as u128)
            .ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge dense size overflow")
            })?;
        let output_bytes = entries
            .checked_mul(std::mem::size_of::<Complex64>() as u128)
            .ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge dense size overflow")
            })?;
        if output_bytes > max_bytes {
            return Err(pyo3::exceptions::PyMemoryError::new_err(format!(
                "charge dense matrix requires approximately {output_bytes} bytes, exceeding max_bytes={max_bytes}"
            )));
        }
        let values = vec![
            Complex64::default();
            usize::try_from(entries).map_err(|_| {
                pyo3::exceptions::PyMemoryError::new_err("charge dense dimension overflow")
            })?
        ];
        let values = py.allow_threads(|| {
            let mut values = values;
            for row in 0..self.dimension {
                for index in self.indptr[row]..self.indptr[row + 1] {
                    values[row * self.dimension + self.columns[index]] = self.values[index];
                }
            }
            values
        });
        Ok(PyArray1::from_vec(py, values))
    }

    fn apply<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let state_values = state.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge state must be C-contiguous")
        })?;
        let output_bytes = (self.dimension as u128)
            .checked_mul(std::mem::size_of::<Complex64>() as u128)
            .ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge output size overflow")
            })?;
        if output_bytes > max_bytes {
            return Err(pyo3::exceptions::PyMemoryError::new_err(format!(
                "charge MVP output requires approximately {output_bytes} bytes, exceeding max_bytes={max_bytes}"
            )));
        }
        let mut output = vec![Complex64::default(); self.dimension];
        py.allow_threads(|| self.apply_values(state_values, &mut output, true))?;
        Ok(PyArray1::from_vec(py, output))
    }

    fn apply_with_parallelism<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        max_bytes: u128,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let state_values = state.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge state must be C-contiguous")
        })?;
        let output_bytes = (self.dimension as u128)
            .checked_mul(std::mem::size_of::<Complex64>() as u128)
            .ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge output size overflow")
            })?;
        if output_bytes > max_bytes {
            return Err(pyo3::exceptions::PyMemoryError::new_err(format!(
                "charge MVP output requires approximately {output_bytes} bytes, exceeding max_bytes={max_bytes}"
            )));
        }
        let mut output = vec![Complex64::default(); self.dimension];
        py.allow_threads(|| self.apply_values(state_values, &mut output, parallel))?;
        Ok(PyArray1::from_vec(py, output))
    }

    fn apply_into<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        mut output: PyReadwriteArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<()> {
        let state_values = state.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge state must be C-contiguous")
        })?;
        let output_values = output.as_slice_mut().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge output must be C-contiguous")
        })?;
        let output_bytes = (self.dimension as u128)
            .checked_mul(std::mem::size_of::<Complex64>() as u128)
            .ok_or_else(|| {
                pyo3::exceptions::PyMemoryError::new_err("charge output size overflow")
            })?;
        if output_bytes > max_bytes {
            return Err(pyo3::exceptions::PyMemoryError::new_err(format!(
                "charge MVP output requires approximately {output_bytes} bytes, exceeding max_bytes={max_bytes}"
            )));
        }
        py.allow_threads(|| self.apply_values(state_values, output_values, true))
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn charge_sector_plan_compact(
    py: Python<'_>,
    local_dimensions: Vec<usize>,
    axis_kinds: Vec<u8>,
    axis_indices: Vec<usize>,
    fermion_weights: Vec<Vec<i128>>,
    boson_weights: Vec<Vec<i128>>,
    qubit_levels: Vec<Vec<(i128, i128)>>,
    target: Vec<i128>,
    max_bytes: u128,
) -> PyResult<NativeChargeSectorPlan> {
    let plan = py
        .allow_threads(|| {
            build_compact_charge_sector_plan(
                local_dimensions,
                axis_kinds,
                axis_indices,
                fermion_weights,
                boson_weights,
                qubit_levels,
                target,
                max_bytes,
            )
        })
        .map_err(crate::convert::map_error)?;
    Ok(NativeChargeSectorPlan {
        plan: Arc::new(plan),
    })
}
