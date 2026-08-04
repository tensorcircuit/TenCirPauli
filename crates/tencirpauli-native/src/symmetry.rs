use numpy::{PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tencir_pauli_core::{find_z2_symmetries, CliffordOperation, U1Sector, Z2TaperingPlan};

use crate::convert::{build_canonical_operator, map_error, operator_output, CanonicalizeOutput};

type U1CsrOutput<'py> = (
    usize,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<numpy::Complex64>>,
);
type U1CooOutput<'py> = (
    usize,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<numpy::Complex64>>,
);
type U1DenseOutput<'py> = (usize, Bound<'py, PyArray1<numpy::Complex64>>);

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeZ2TaperingPlan {
    plan: Z2TaperingPlan,
}

#[pymethods]
impl NativeZ2TaperingPlan {
    #[getter]
    fn nqubits_before(&self) -> usize {
        self.plan.nqubits_before()
    }

    #[getter]
    fn nqubits_after(&self) -> usize {
        self.plan.nqubits_after()
    }

    #[getter]
    fn generators(&self) -> Vec<Vec<u8>> {
        self.plan
            .generators()
            .iter()
            .map(|word| word.codes())
            .collect()
    }

    #[getter]
    fn sector(&self) -> Vec<i8> {
        self.plan.sector().to_vec()
    }

    #[getter]
    fn removed_qubits(&self) -> Vec<usize> {
        self.plan.removed_qubits().to_vec()
    }

    #[getter]
    fn clifford_operations(&self) -> Vec<(u8, usize, usize)> {
        self.plan
            .operations()
            .iter()
            .map(|operation| match operation {
                CliffordOperation::H { qubit } => (0, *qubit, 0),
                CliffordOperation::S { qubit } => (1, *qubit, 0),
                CliffordOperation::Sdg { qubit } => (2, *qubit, 0),
                CliffordOperation::Cnot { control, target } => (3, *control, *target),
            })
            .collect()
    }

    fn transform_operator(
        &self,
        py: Python<'_>,
        nqubits: usize,
        structures: Vec<Vec<u8>>,
        coefficients_re: Vec<f64>,
        coefficients_im: Vec<f64>,
    ) -> PyResult<CanonicalizeOutput> {
        if nqubits != self.plan.nqubits_before() {
            return Err(PyValueError::new_err(format!(
                "expected {} qubits, got {nqubits}",
                self.plan.nqubits_before()
            )));
        }
        let result = py.allow_threads(|| {
            let operator =
                build_canonical_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
            self.plan.transform_operator(&operator).map_err(map_error)
        })?;
        Ok(operator_output(&result))
    }
}

#[pyfunction]
pub(crate) fn pauli_find_z2_symmetries(
    py: Python<'_>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: usize,
) -> PyResult<(Vec<Vec<u8>>, usize)> {
    py.allow_threads(|| {
        let operator =
            build_canonical_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
        let analysis = find_z2_symmetries(&operator, max_bytes as u128).map_err(map_error)?;
        Ok((
            analysis
                .generators
                .iter()
                .map(|word| word.codes())
                .collect(),
            analysis.constraint_rank,
        ))
    })
}

#[pyfunction]
pub(crate) fn pauli_z2_tapering_plan(
    py: Python<'_>,
    nqubits: usize,
    generators: Vec<Vec<u8>>,
    sector: Vec<i8>,
) -> PyResult<NativeZ2TaperingPlan> {
    let plan = py.allow_threads(|| {
        let words = generators
            .iter()
            .map(|codes| tencir_pauli_core::PauliWord::from_codes(nqubits, codes))
            .collect::<Result<Vec<_>, _>>()
            .map_err(map_error)?;
        Z2TaperingPlan::new(nqubits, &words, &sector).map_err(map_error)
    })?;
    Ok(NativeZ2TaperingPlan { plan })
}

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeU1RestrictedOperator {
    operator: tencir_pauli_core::U1RestrictedOperator,
}

#[pymethods]
impl NativeU1RestrictedOperator {
    #[getter]
    fn nqubits(&self) -> usize {
        self.operator.sector().nqubits()
    }

    #[getter]
    fn particle_number(&self) -> usize {
        self.operator.sector().particle_number()
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.operator.dimension()
    }

    fn apply<'py>(
        &self,
        py: Python<'py>,
        state: numpy::PyReadonlyArray1<'py, numpy::Complex64>,
        max_bytes: usize,
    ) -> PyResult<Bound<'py, PyArray1<numpy::Complex64>>> {
        let state_slice = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
        let values = py
            .allow_threads(|| self.operator.apply(state_slice, max_bytes as u128))
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn apply_into<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, numpy::Complex64>,
        mut output: PyReadwriteArray1<'py, numpy::Complex64>,
        max_bytes: usize,
    ) -> PyResult<()> {
        let state_slice = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
        let output_slice = output
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output must be C-contiguous"))?;
        let _ = max_bytes;
        py.allow_threads(|| self.operator.apply_into(state_slice, output_slice))
            .map_err(map_error)
    }

    fn mvp_plan(&self, py: Python<'_>, max_bytes: usize) -> PyResult<NativeU1MvpPlan> {
        let plan = py
            .allow_threads(|| self.operator.mvp_plan(max_bytes as u128))
            .map_err(map_error)?;
        Ok(NativeU1MvpPlan { plan })
    }

    fn dense<'py>(&self, py: Python<'py>, max_bytes: usize) -> PyResult<U1DenseOutput<'py>> {
        let (dimension, values) = py
            .allow_threads(|| self.operator.dense(max_bytes as u128))
            .map_err(map_error)?;
        Ok((dimension, PyArray1::from_vec(py, values)))
    }

    fn coo<'py>(&self, py: Python<'py>, max_bytes: usize) -> PyResult<U1CooOutput<'py>> {
        let matrix = py
            .allow_threads(|| self.operator.coo(max_bytes as u128))
            .map_err(map_error)?;
        Ok((
            matrix.dimension,
            PyArray1::from_vec(py, matrix.rows),
            PyArray1::from_vec(py, matrix.columns),
            PyArray1::from_vec(py, matrix.values),
        ))
    }

    fn csr<'py>(&self, py: Python<'py>, max_bytes: usize) -> PyResult<U1CsrOutput<'py>> {
        let matrix = py
            .allow_threads(|| self.operator.csr(max_bytes as u128))
            .map_err(map_error)?;
        Ok((
            matrix.dimension,
            PyArray1::from_vec(py, matrix.indptr),
            PyArray1::from_vec(py, matrix.columns),
            PyArray1::from_vec(py, matrix.values),
        ))
    }
}

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeU1MvpPlan {
    plan: tencir_pauli_core::U1MvpPlan,
}

#[pymethods]
impl NativeU1MvpPlan {
    #[getter]
    fn nqubits(&self) -> usize {
        self.plan.sector().nqubits()
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
    fn transition_count(&self) -> usize {
        self.plan.transition_count()
    }

    fn apply<'py>(
        &self,
        py: Python<'py>,
        state: numpy::PyReadonlyArray1<'py, numpy::Complex64>,
        max_bytes: usize,
    ) -> PyResult<Bound<'py, PyArray1<numpy::Complex64>>> {
        let state_slice = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
        let values = py
            .allow_threads(|| self.plan.apply(state_slice, max_bytes as u128))
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn apply_into<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, numpy::Complex64>,
        mut output: PyReadwriteArray1<'py, numpy::Complex64>,
        max_bytes: usize,
    ) -> PyResult<()> {
        let state_slice = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
        let output_slice = output
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output must be C-contiguous"))?;
        let _ = max_bytes;
        py.allow_threads(|| self.plan.apply_into(state_slice, output_slice))
            .map_err(map_error)
    }
}

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct NativeU1LazyMvpPlan {
    plan: tencir_pauli_core::U1LazyMvpPlan,
}

#[pymethods]
impl NativeU1LazyMvpPlan {
    #[getter]
    fn nqubits(&self) -> usize {
        self.plan.sector().nqubits()
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
    fn estimated_bytes(&self) -> u128 {
        self.plan.estimated_bytes()
    }

    fn apply<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, numpy::Complex64>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<numpy::Complex64>>> {
        let state_slice = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
        let values = py
            .allow_threads(|| self.plan.apply(state_slice, max_bytes))
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn apply_into<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, numpy::Complex64>,
        mut output: PyReadwriteArray1<'py, numpy::Complex64>,
        max_bytes: u128,
    ) -> PyResult<()> {
        let state_slice = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
        let output_slice = output
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output must be C-contiguous"))?;
        let _ = max_bytes;
        py.allow_threads(|| self.plan.apply_into(state_slice, output_slice))
            .map_err(map_error)
    }
}

#[pyfunction]
pub(crate) fn pauli_restrict_u1(
    py: Python<'_>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    particle_number: usize,
    max_bytes: usize,
) -> PyResult<NativeU1RestrictedOperator> {
    let operator = py.allow_threads(|| {
        let operator =
            build_canonical_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
        let sector = U1Sector::new(nqubits, particle_number).map_err(map_error)?;
        tencir_pauli_core::U1RestrictedOperator::new(&operator, sector, max_bytes as u128)
            .map_err(map_error)
    })?;
    Ok(NativeU1RestrictedOperator { operator })
}

#[pyfunction]
pub(crate) fn pauli_restrict_u1_lazy(
    py: Python<'_>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    particle_number: usize,
    max_bytes: usize,
) -> PyResult<NativeU1LazyMvpPlan> {
    let plan = py.allow_threads(|| {
        let operator =
            build_canonical_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
        let sector = U1Sector::new(nqubits, particle_number).map_err(map_error)?;
        tencir_pauli_core::U1LazyMvpPlan::new(&operator, sector, max_bytes as u128)
            .map_err(map_error)
    })?;
    Ok(NativeU1LazyMvpPlan { plan })
}

#[pyfunction]
pub(crate) fn u1_basis_words<'py>(
    py: Python<'py>,
    nqubits: usize,
    particle_number: usize,
    max_bytes: usize,
) -> PyResult<(usize, usize, Bound<'py, PyArray1<u64>>)> {
    let (dimension, word_count, words) = py.allow_threads(|| {
        let sector = U1Sector::new(nqubits, particle_number).map_err(map_error)?;
        if nqubits <= 64 {
            let basis = sector.basis_words(max_bytes as u128).map_err(map_error)?;
            let words = basis
                .into_iter()
                .map(|word| {
                    u64::try_from(word)
                        .map_err(|_| PyValueError::new_err("basis word exceeds uint64"))
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok::<_, PyErr>((words.len(), 1, words))
        } else {
            let basis = sector
                .basis_words_packed(max_bytes as u128)
                .map_err(map_error)?;
            Ok::<_, PyErr>((
                usize::try_from(basis.dimension)
                    .map_err(|_| PyValueError::new_err("U1 dimension exceeds platform indices"))?,
                basis.word_count,
                basis.words,
            ))
        }
    })?;
    Ok((dimension, word_count, PyArray1::from_vec(py, words)))
}
