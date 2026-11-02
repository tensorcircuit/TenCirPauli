#[pyclass(module = "tencirpauli._native")]
pub(crate) struct StructuredMvpPlan {
    pub(super) plan: CoreStructuredMvpPlan,
}

#[pyclass(module = "tencirpauli._native")]
pub(crate) struct StructuredEagerMvpPlan {
    pub(super) plan: CoreStructuredEagerMvpPlan,
}

#[pymethods]
impl StructuredEagerMvpPlan {
    fn apply_adjoint<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let state_slice = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
        let output = py
            .allow_threads(|| self.plan.apply_adjoint(state_slice, max_bytes))
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, output))
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.plan.dimension()
    }

    #[getter]
    fn estimated_bytes(&self) -> u128 {
        self.plan.estimated_bytes()
    }

    #[getter]
    fn estimated_construction_peak_bytes(&self) -> u128 {
        self.plan.estimated_construction_peak_bytes()
    }

    #[getter]
    fn source_term_count(&self) -> usize {
        self.plan.source_term_count()
    }

    #[getter]
    fn schedule_count(&self) -> usize {
        self.plan.schedule_count()
    }

    #[getter]
    fn execution_family(&self) -> &'static str {
        self.plan.execution_family()
    }

    #[getter]
    fn native_variant(&self) -> &'static str {
        self.plan.native_variant()
    }

    #[getter]
    fn retained_format(&self) -> &'static str {
        self.plan.retained_format()
    }

    #[getter]
    fn transition_count(&self) -> usize {
        self.plan.transition_count()
    }

    fn csr<'py>(
        &self,
        py: Python<'py>,
        max_bytes: u128,
    ) -> PyResult<NumpyStructuredCsrOutput<'py>> {
        let (indptr, columns, values) = py
            .allow_threads(|| self.plan.csr(max_bytes))
            .map_err(map_error)?;
        Ok((
            PyArray1::from_vec(py, indptr),
            PyArray1::from_vec(py, columns),
            PyArray1::from_vec(py, values),
        ))
    }

    fn apply<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let state_slice = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
        if state_slice.len() != self.dimension() {
            return Err(PyValueError::new_err(format!(
                "state must have shape ({},), got ({},)",
                self.dimension(),
                state_slice.len()
            )));
        }
        let output = py
            .allow_threads(|| self.plan.apply(state_slice, max_bytes))
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, output))
    }

    fn apply_into<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        mut output: PyReadwriteArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<()> {
        let state_slice = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
        let output_slice = output
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output must be C-contiguous"))?;
        py.allow_threads(|| self.plan.apply_into(state_slice, output_slice, max_bytes))
            .map_err(map_error)
    }
}

#[pymethods]
impl StructuredMvpPlan {
    fn apply_adjoint<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let state_slice = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
        let output = py
            .allow_threads(|| self.plan.apply_adjoint(state_slice, max_bytes))
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, output))
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.plan.dimension()
    }

    #[getter]
    fn estimated_bytes(&self) -> u128 {
        self.plan.estimated_bytes()
    }

    #[getter]
    fn estimated_construction_peak_bytes(&self) -> u128 {
        self.plan.estimated_construction_peak_bytes()
    }

    #[getter]
    fn source_term_count(&self) -> usize {
        self.plan.source_term_count()
    }

    #[getter]
    fn schedule_count(&self) -> usize {
        self.plan.schedule_count()
    }

    #[getter]
    fn execution_family(&self) -> &'static str {
        self.plan.execution_family()
    }

    #[getter]
    fn native_variant(&self) -> &'static str {
        self.plan.native_variant()
    }

    #[getter]
    fn retained_format(&self) -> &'static str {
        self.plan.retained_format()
    }

    #[getter]
    fn transition_count(&self) -> Option<usize> {
        self.plan.transition_count()
    }

    fn debug_counters(&self) -> (usize, usize, usize, usize) {
        self.plan.debug_counters()
    }

    fn debug_metadata(&self) -> (usize, usize, usize) {
        self.plan.debug_metadata()
    }

    fn debug_fermion_mask_phase_profile(
        &self,
        py: Python<'_>,
        samples: usize,
    ) -> PyResult<FermionMaskPhaseProfileOutput> {
        py.allow_threads(|| {
            self.plan
                .fermion_mask_phase_profile(samples)
                .map(|profile| {
                    profile.map(|profile| {
                        (
                            profile.descriptor_count,
                            profile.sample_count,
                            profile.packed_source_unit_count,
                            profile.schedule_member_evaluations,
                            profile.packed_extraction_nanos,
                            profile.parity_nanos,
                            profile.index_update_nanos,
                            profile.input_output_nanos,
                        )
                    })
                })
        })
        .map_err(map_error)
    }

    fn apply<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let state_slice = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
        if state_slice.len() != self.dimension() {
            return Err(PyValueError::new_err(format!(
                "state must have shape ({},), got ({},)",
                self.dimension(),
                state_slice.len()
            )));
        }
        let output = py
            .allow_threads(|| self.plan.apply(state_slice, max_bytes))
            .map_err(map_error)?;
        Ok(PyArray1::from_vec(py, output))
    }

    fn apply_into<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        mut output: PyReadwriteArray1<'py, NumpyComplex128>,
        max_bytes: u128,
    ) -> PyResult<()> {
        let state_slice = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
        let output_slice = output
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output must be C-contiguous"))?;
        py.allow_threads(|| self.plan.apply_into(state_slice, output_slice, max_bytes))
            .map_err(map_error)
    }
}

#[pyfunction]
pub(crate) fn structured_fermion_canonicalize(
    py: Python<'_>,
    n_modes: usize,
    factors: Vec<Vec<(usize, u8)>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: u128,
) -> PyResult<NativeFermionOperatorHandle> {
    let result = py.allow_threads(|| {
        let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
        canonicalize_fermion_terms(n_modes, &factors, &coefficients, max_bytes).map_err(map_error)
    })?;
    Ok(NativeFermionOperatorHandle::from_result(n_modes, result))
}

#[pyfunction]
pub(crate) fn structured_fermion_integrals(
    py: Python<'_>,
    one_body: Bound<'_, PyAny>,
    two_body: Bound<'_, PyAny>,
    constant_re: f64,
    constant_im: f64,
    max_bytes: u128,
) -> PyResult<NativeFermionOperatorHandle> {
    let one_untyped = one_body
        .downcast::<PyUntypedArray>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("one_body must be a NumPy array"))?;
    if one_untyped.ndim() != 2 || one_untyped.shape()[0] != one_untyped.shape()[1] {
        return Err(PyValueError::new_err("one_body must have a square shape"));
    }
    let n_modes = one_untyped.shape()[0];
    let one_body = integral_array(one_body, &[n_modes, n_modes], "one_body")?;
    let two_body = integral_array(two_body, &[n_modes, n_modes, n_modes, n_modes], "two_body")?;
    let one_values = one_body.values("one_body")?;
    let two_values = two_body.values("two_body")?;
    let result = py.allow_threads(|| -> PyResult<_> {
        let one_body = one_values.complex_values();
        let two_body = two_values.complex_values();
        canonicalize_fermion_integrals(
            FermionIntegralSource::SpinOrbital {
                n_modes,
                one_body: &one_body,
                two_body: &two_body,
            },
            Complex64::new(constant_re, constant_im),
            max_bytes,
        )
        .map_err(map_error)
    })?;
    Ok(NativeFermionOperatorHandle::from_result(n_modes, result))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn structured_fermion_integral_blocks(
    py: Python<'_>,
    n_spatial: usize,
    one_alpha: Bound<'_, PyAny>,
    one_beta: Bound<'_, PyAny>,
    eri_aa: Bound<'_, PyAny>,
    eri_ab: Bound<'_, PyAny>,
    eri_ba: Bound<'_, PyAny>,
    eri_bb: Bound<'_, PyAny>,
    ordering: u8,
    constant_re: f64,
    constant_im: f64,
    max_bytes: u128,
) -> PyResult<NativeFermionOperatorHandle> {
    let expected_one = [n_spatial, n_spatial];
    let expected_eri = [n_spatial, n_spatial, n_spatial, n_spatial];
    let one_alpha = integral_array(one_alpha, &expected_one, "one_alpha")?;
    let one_beta = integral_array(one_beta, &expected_one, "one_beta")?;
    let eri_aa = integral_array(eri_aa, &expected_eri, "eri_aa")?;
    let eri_ab = integral_array(eri_ab, &expected_eri, "eri_ab")?;
    let eri_ba = integral_array(eri_ba, &expected_eri, "eri_ba")?;
    let eri_bb = integral_array(eri_bb, &expected_eri, "eri_bb")?;
    let one_alpha_values = one_alpha.values("one_alpha")?;
    let one_beta_values = one_beta.values("one_beta")?;
    let eri_aa_values = eri_aa.values("eri_aa")?;
    let eri_ab_values = eri_ab.values("eri_ab")?;
    let eri_ba_values = eri_ba.values("eri_ba")?;
    let eri_bb_values = eri_bb.values("eri_bb")?;
    let ordering = match ordering {
        0 => FermionSpinOrdering::Interleaved,
        1 => FermionSpinOrdering::AlphaThenBeta,
        _ => return Err(PyValueError::new_err("unknown spin-orbital ordering")),
    };
    let result = py.allow_threads(|| -> PyResult<_> {
        let one_alpha = one_alpha_values.complex_values();
        let one_beta = one_beta_values.complex_values();
        let eri_aa = eri_aa_values.complex_values();
        let eri_ab = eri_ab_values.complex_values();
        let eri_ba = eri_ba_values.complex_values();
        let eri_bb = eri_bb_values.complex_values();
        canonicalize_fermion_integrals(
            FermionIntegralSource::SpinBlocks(FermionSpinBlocks {
                n_spatial,
                one_alpha: &one_alpha,
                one_beta: &one_beta,
                eri_aa: &eri_aa,
                eri_ab: &eri_ab,
                eri_ba: &eri_ba,
                eri_bb: &eri_bb,
                ordering,
            }),
            Complex64::new(constant_re, constant_im),
            max_bytes,
        )
        .map_err(map_error)
    })?;
    Ok(NativeFermionOperatorHandle::from_result(
        n_spatial
            .checked_mul(2)
            .ok_or_else(|| PyValueError::new_err("spin-orbital mode count overflow"))?,
        result,
    ))
}

#[pyfunction]
pub(crate) fn structured_boson_canonicalize(
    py: Python<'_>,
    n_modes: usize,
    factors: Vec<Vec<(usize, u8)>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: u128,
) -> PyResult<NativeBosonOperatorHandle> {
    let result = py.allow_threads(|| {
        let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
        canonicalize_boson_terms(n_modes, &factors, &coefficients, max_bytes).map_err(map_error)
    })?;
    Ok(NativeBosonOperatorHandle::from_result(n_modes, result))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn structured_hybrid_canonicalize(
    py: Python<'_>,
    n_modes: usize,
    n_bosons: usize,
    nqubits: usize,
    n_qudit_sites: usize,
    qudit_dimension: usize,
    input: HybridRawInput,
    max_bytes: u128,
) -> PyResult<NativeHybridOperatorHandle> {
    let (
        fermion_factors,
        boson_factors,
        qubit_codes,
        qudit_present,
        qudit_triples,
        coefficients_re,
        coefficients_im,
    ) = input;
    let layout = HybridLayout {
        n_modes,
        n_bosons,
        nqubits,
        n_qudit_sites,
        qudit_dimension,
    };
    let result = py.allow_threads(|| {
        let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
        canonicalize_hybrid_terms(
            layout,
            HybridRawBatch {
                fermion_factors: &fermion_factors,
                boson_factors: &boson_factors,
                qubit_codes: &qubit_codes,
                qudit_present: &qudit_present,
                qudit_triples: &qudit_triples,
                coefficients: &coefficients,
            },
            max_bytes,
        )
        .map_err(map_error)
    })?;
    Ok(NativeHybridOperatorHandle::from_result(layout, result))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn structured_hybrid_jordan_wigner(
    py: Python<'_>,
    n_modes: usize,
    n_bosons: usize,
    nqubits: usize,
    n_qudit_sites: usize,
    qudit_dimension: usize,
    input: HybridInput,
    max_bytes: u128,
) -> PyResult<NativeHybridOperatorHandle> {
    let (
        fermion_present,
        fermion_creation,
        fermion_annihilation,
        boson_present,
        boson_blocks,
        qubit_codes,
        mapped_present,
        mapped_codes,
        qudit_present,
        qudit_triples,
        coefficients_re,
        coefficients_im,
    ) = input;
    let layout = HybridLayout {
        n_modes,
        n_bosons,
        nqubits,
        n_qudit_sites,
        qudit_dimension,
    };
    let result = py.allow_threads(|| {
        let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
        jordan_wigner_hybrid_terms(
            layout,
            HybridBatch {
                fermion_present: &fermion_present,
                fermion_creation: &fermion_creation,
                fermion_annihilation: &fermion_annihilation,
                boson_present: &boson_present,
                boson_blocks: &boson_blocks,
                qubit_codes: &qubit_codes,
                mapped_present: &mapped_present,
                mapped_codes: &mapped_codes,
                qudit_present: &qudit_present,
                qudit_triples: &qudit_triples,
                coefficients: &coefficients,
            },
            max_bytes,
        )
        .map_err(map_error)
    })?;
    Ok(NativeHybridOperatorHandle::from_result(layout, result))
}
use super::*;
