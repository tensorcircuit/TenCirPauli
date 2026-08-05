use std::sync::Arc;

use numpy::{Complex64 as NumpyComplex128, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;
use tencir_pauli_core::{
    apply_charge_csr_into, apply_charge_mvp_from_plan, apply_charge_mvp_from_plan_into,
    build_charge_sector_plan, build_compact_charge_sector_plan, build_fast_fermion_mvp_plan,
    compile_charge_transitions_from_plan, estimate_charge_transition_terms_bytes, ChargeSectorPlan,
    ChargeTransitionPlanLayout, ChargeTransitionTerm, Complex64, FastFermionMvpPlan, PauliError,
};

use crate::convert::{map_error, split_complex};

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

#[allow(clippy::too_many_arguments)]
fn charge_terms_from_inputs(
    coefficients: &[NumpyComplex128],
    fermion_creation: &[Vec<u32>],
    fermion_annihilation: &[Vec<u32>],
    boson_blocks: &[Vec<(u32, u32, u32)>],
    qubit_codes: &[Vec<u8>],
    mapped_present: &[bool],
    mapped_codes: &[Vec<u8>],
    qudit_present: &[bool],
    qudit_triples: &[Vec<(u32, u32, u32)>],
) -> PyResult<Vec<ChargeTransitionTerm>> {
    let term_count = coefficients.len();
    if fermion_creation.len() != term_count
        || fermion_annihilation.len() != term_count
        || boson_blocks.len() != term_count
        || qubit_codes.len() != term_count
        || mapped_present.len() != term_count
        || mapped_codes.len() != term_count
        || qudit_present.len() != term_count
        || qudit_triples.len() != term_count
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "charge transition term arrays have inconsistent lengths",
        ));
    }
    Ok(coefficients
        .iter()
        .enumerate()
        .map(|(index, coefficient)| ChargeTransitionTerm {
            fermion_creation: fermion_creation[index].clone(),
            fermion_annihilation: fermion_annihilation[index].clone(),
            boson_blocks: boson_blocks[index].clone(),
            qubit_codes: qubit_codes[index].clone(),
            mapped_present: mapped_present[index],
            mapped_codes: mapped_codes[index].clone(),
            qudit_present: qudit_present[index],
            qudit_triples: qudit_triples[index].clone(),
            coefficient: Complex64::new(coefficient.re, coefficient.im),
        })
        .collect())
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
        dimension,
        local_dimensions,
        fermion_positions,
        boson_positions,
        qubit_positions,
        qudit_positions,
        fermion_creation,
        fermion_annihilation,
        boson_blocks,
        qubit_codes,
        mapped_present,
        mapped_codes,
        qudit_present,
        qudit_triples,
        coefficients,
        qudit_dimension,
        termwise_conserved,
        max_bytes,
        fast_fermion_particles=None
    ))]
    fn compile_mvp<'py>(
        &self,
        dimension: usize,
        local_dimensions: Vec<u64>,
        fermion_positions: Vec<u64>,
        boson_positions: Vec<u64>,
        qubit_positions: Vec<u64>,
        qudit_positions: Vec<u64>,
        fermion_creation: Vec<Vec<u32>>,
        fermion_annihilation: Vec<Vec<u32>>,
        boson_blocks: Vec<Vec<(u32, u32, u32)>>,
        qubit_codes: Vec<Vec<u8>>,
        mapped_present: Vec<bool>,
        mapped_codes: Vec<Vec<u8>>,
        qudit_present: Vec<bool>,
        qudit_triples: Vec<Vec<(u32, u32, u32)>>,
        coefficients: numpy::PyReadonlyArray1<'py, NumpyComplex128>,
        qudit_dimension: u64,
        termwise_conserved: bool,
        max_bytes: u128,
        fast_fermion_particles: Option<usize>,
    ) -> PyResult<NativeChargeMvpPlan> {
        let coefficient_values = coefficients.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("charge coefficients must be C-contiguous")
        })?;
        let terms = charge_terms_from_inputs(
            coefficient_values,
            &fermion_creation,
            &fermion_annihilation,
            &boson_blocks,
            &qubit_codes,
            &mapped_present,
            &mapped_codes,
            &qudit_present,
            &qudit_triples,
        )?;
        if dimension != self.dimension() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "charge MVP dimension does not match its sector plan",
            ));
        }
        let base_bytes = charge_plan_base_bytes(&self.plan, &terms).map_err(map_error)?;
        let fast_budget = charge_remaining_budget(max_bytes, base_bytes).map_err(map_error)?;
        let fast_fermion_plan = if termwise_conserved {
            build_fast_fermion_mvp_plan(
                &self.plan,
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
            plan: Arc::clone(&self.plan),
            local_dimensions,
            fermion_positions,
            boson_positions,
            qubit_positions,
            qudit_positions,
            qudit_dimension,
            terms,
            termwise_conserved,
            fast_fermion_plan,
        })
    }
}

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeChargeMvpPlan {
    plan: Arc<ChargeSectorPlan>,
    local_dimensions: Vec<u64>,
    fermion_positions: Vec<u64>,
    boson_positions: Vec<u64>,
    qubit_positions: Vec<u64>,
    qudit_positions: Vec<u64>,
    qudit_dimension: u64,
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
                apply_charge_mvp_from_plan(
                    &self.plan,
                    ChargeTransitionPlanLayout {
                        dimension: self.dimension(),
                        local_dimensions: &self.local_dimensions,
                        fermion_positions: &self.fermion_positions,
                        boson_positions: &self.boson_positions,
                        qubit_positions: &self.qubit_positions,
                        qudit_positions: &self.qudit_positions,
                        qudit_dimension: self.qudit_dimension,
                        max_bytes,
                    },
                    &self.terms,
                    state_values,
                    self.termwise_conserved,
                    self.fast_fermion_plan.as_ref(),
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
            apply_charge_mvp_from_plan_into(
                &self.plan,
                ChargeTransitionPlanLayout {
                    dimension: self.dimension(),
                    local_dimensions: &self.local_dimensions,
                    fermion_positions: &self.fermion_positions,
                    boson_positions: &self.boson_positions,
                    qubit_positions: &self.qubit_positions,
                    qudit_positions: &self.qudit_positions,
                    qudit_dimension: self.qudit_dimension,
                    max_bytes: effective_max,
                },
                &self.terms,
                state_values,
                output_values,
                self.termwise_conserved,
                self.fast_fermion_plan.as_ref(),
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
            let result = compile_charge_transitions_from_plan(
                &self.plan,
                ChargeTransitionPlanLayout {
                    dimension: self.dimension(),
                    local_dimensions: &self.local_dimensions,
                    fermion_positions: &self.fermion_positions,
                    boson_positions: &self.boson_positions,
                    qubit_positions: &self.qubit_positions,
                    qudit_positions: &self.qudit_positions,
                    qudit_dimension: self.qudit_dimension,
                    max_bytes,
                },
                &self.terms,
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
        let (real, imaginary) = split_complex(&self.values);
        let values = real
            .into_iter()
            .zip(imaginary)
            .map(|(re, im)| NumpyComplex128::new(re, im))
            .collect();
        (
            PyArray1::from_vec(py, indptr),
            PyArray1::from_vec(py, columns),
            PyArray1::from_vec(py, values),
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
        let (real, imaginary) = split_complex(&self.values);
        let values = real
            .into_iter()
            .zip(imaginary)
            .map(|(re, im)| NumpyComplex128::new(re, im))
            .collect();
        Ok((
            PyArray1::from_vec(py, rows),
            PyArray1::from_vec(py, columns),
            PyArray1::from_vec(py, values),
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
    Ok(NativeChargeSectorPlan {
        plan: Arc::new(plan),
    })
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
