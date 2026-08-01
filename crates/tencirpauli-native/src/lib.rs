//! Private PyO3 extension for the public `tencirpauli` Python package.

use numpy::{
    Complex64 as NumpyComplex128, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadwriteArray1,
};
use pyo3::exceptions::{PyMemoryError, PyOverflowError, PyValueError};
use pyo3::prelude::*;
use tencir_pauli_core::{
    compatibility_matrix, group_words, incompatibility_edges, packed_word_count, Complex64,
    GroupingAlgorithm, GroupingMode, MvpPlan, MvpStrategy, PauliError, PauliOperator, PauliWord,
};

type CanonicalizeOutput = (Vec<Vec<u8>>, Vec<f64>, Vec<f64>);
type CanonicalizeBatchOutput = (Vec<Vec<u8>>, Vec<f64>, Vec<f64>, Vec<usize>, Vec<u8>);
type CanonicalizeInput = (Vec<Vec<u8>>, Vec<f64>, Vec<f64>);
type DenseOutput = (usize, Vec<f64>, Vec<f64>);
type CooOutput = (usize, Vec<u64>, Vec<u64>, Vec<f64>, Vec<f64>);
type CsrOutput = (usize, Vec<u64>, Vec<u64>, Vec<f64>, Vec<f64>);
type BackendPlanOutput = (u8, usize, usize, Vec<u64>, Vec<u64>, Vec<f64>, Vec<f64>);
type NumpySparseOutput<'py> = (
    usize,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<NumpyComplex128>>,
);

fn core_complex_slice(values: &[NumpyComplex128]) -> &[Complex64] {
    debug_assert_eq!(
        std::mem::size_of::<NumpyComplex128>(),
        std::mem::size_of::<Complex64>()
    );
    debug_assert_eq!(
        std::mem::align_of::<NumpyComplex128>(),
        std::mem::align_of::<Complex64>()
    );
    // SAFETY: NumPy complex128 is num_complex::Complex<f64>, which is repr(C)
    // with the same two f64 fields as the core's repr(C) Complex64. The input
    // borrow remains alive for the returned slice and no mutable alias is made.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<Complex64>(), values.len()) }
}

fn core_complex_slice_mut(values: &mut [NumpyComplex128]) -> &mut [Complex64] {
    debug_assert_eq!(
        std::mem::size_of::<NumpyComplex128>(),
        std::mem::size_of::<Complex64>()
    );
    debug_assert_eq!(
        std::mem::align_of::<NumpyComplex128>(),
        std::mem::align_of::<Complex64>()
    );
    // SAFETY: See core_complex_slice. The caller holds the exclusive NumPy
    // read-write borrow, so the returned core slice is the only mutable view.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<Complex64>(), values.len()) }
}

fn numpy_complex_array<'py>(
    py: Python<'py>,
    values: Vec<Complex64>,
) -> Bound<'py, PyArray1<NumpyComplex128>> {
    PyArray1::from_vec(
        py,
        values
            .into_iter()
            .map(|value| NumpyComplex128::new(value.re, value.im))
            .collect(),
    )
}

#[pyclass(module = "tencirpauli._native")]
struct NativeMvpPlan {
    plan: MvpPlan,
}

#[pymethods]
impl NativeMvpPlan {
    #[getter]
    fn nqubits(&self) -> usize {
        self.plan.nqubits()
    }

    #[getter]
    fn term_count(&self) -> usize {
        self.plan.term_count()
    }

    #[getter]
    fn strategy(&self) -> &'static str {
        match self.plan.strategy() {
            MvpStrategy::XMaskDiagonal => "x_mask_diagonal",
            MvpStrategy::TermDirect => "term_direct",
        }
    }

    fn apply<'py>(
        &self,
        py: Python<'py>,
        state: PyReadonlyArray1<'py, NumpyComplex128>,
        max_bytes: usize,
    ) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
        let state_slice = state
            .as_slice()
            .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
        let output_array = PyArray1::zeros(py, state_slice.len(), false);
        let mut writable: PyReadwriteArray1<'_, NumpyComplex128> = output_array.readwrite();
        let output_slice = writable
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output must be C-contiguous"))?;
        let state = core_complex_slice(state_slice);
        let output = core_complex_slice_mut(output_slice);
        py.allow_threads(|| self.plan.apply_into(state, output, max_bytes as u128))
            .map_err(map_error)?;
        drop(writable);
        Ok(output_array)
    }
}

fn map_error(error: PauliError) -> PyErr {
    let message = error.to_string();
    match error {
        PauliError::MemoryLimit { .. } => PyMemoryError::new_err(message),
        PauliError::Overflow { .. } => PyOverflowError::new_err(message),
        _ => PyValueError::new_err(message),
    }
}

fn build_word(nqubits: usize, x_words: Vec<u64>, z_words: Vec<u64>) -> PyResult<PauliWord> {
    PauliWord::from_words(nqubits, x_words, z_words).map_err(map_error)
}

fn complex_coefficients(re: Vec<f64>, im: Vec<f64>) -> PyResult<Vec<Complex64>> {
    if re.len() != im.len() {
        return Err(PyValueError::new_err(format!(
            "real and imaginary coefficient lengths differ: {} and {}",
            re.len(),
            im.len()
        )));
    }
    Ok(re
        .into_iter()
        .zip(im)
        .map(|(real, imaginary)| Complex64::new(real, imaginary))
        .collect())
}

fn operator_output(operator: &PauliOperator) -> CanonicalizeOutput {
    let mut result_structures = Vec::with_capacity(operator.terms().len());
    let mut result_re = Vec::with_capacity(operator.terms().len());
    let mut result_im = Vec::with_capacity(operator.terms().len());
    for term in operator.terms() {
        result_structures.push(term.word.codes());
        result_re.push(term.coefficient.re);
        result_im.push(term.coefficient.im);
    }
    (result_structures, result_re, result_im)
}

fn phase_code(phase: tencir_pauli_core::PauliPhase) -> u8 {
    match phase {
        tencir_pauli_core::PauliPhase::PlusOne => 0,
        tencir_pauli_core::PauliPhase::PlusI => 1,
        tencir_pauli_core::PauliPhase::MinusOne => 2,
        tencir_pauli_core::PauliPhase::MinusI => 3,
    }
}

fn split_complex(values: &[Complex64]) -> (Vec<f64>, Vec<f64>) {
    values.iter().map(|value| (value.re, value.im)).unzip()
}

fn build_operator(
    nqubits: usize,
    structures: &[Vec<u8>],
    coefficients_re: &[f64],
    coefficients_im: &[f64],
) -> PyResult<PauliOperator> {
    let coefficients = complex_coefficients(coefficients_re.to_vec(), coefficients_im.to_vec())?;
    PauliOperator::from_terms(nqubits, structures, &coefficients).map_err(map_error)
}

#[pyfunction]
fn pauli_weight(nqubits: usize, x_words: Vec<u64>, z_words: Vec<u64>) -> PyResult<u32> {
    Ok(build_word(nqubits, x_words, z_words)?.weight())
}

#[pyfunction]
fn pauli_support(nqubits: usize, x_words: Vec<u64>, z_words: Vec<u64>) -> PyResult<Vec<usize>> {
    Ok(build_word(nqubits, x_words, z_words)?.support())
}

#[pyfunction]
fn pauli_codes(nqubits: usize, x_words: Vec<u64>, z_words: Vec<u64>) -> PyResult<Vec<u8>> {
    Ok(build_word(nqubits, x_words, z_words)?.codes())
}

#[pyfunction]
fn pauli_from_codes(nqubits: usize, codes: Vec<u8>) -> PyResult<(Vec<u64>, Vec<u64>)> {
    let word = PauliWord::from_codes(nqubits, &codes).map_err(map_error)?;
    Ok((word.x_words().to_vec(), word.z_words().to_vec()))
}

#[pyfunction]
fn pauli_batch_from_codes(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
) -> PyResult<(usize, Vec<u64>, Vec<u64>)> {
    let word_count = packed_word_count(nqubits);
    let mut x_words = Vec::with_capacity(structures.len() * word_count);
    let mut z_words = Vec::with_capacity(structures.len() * word_count);
    for structure in structures {
        let word = PauliWord::from_codes(nqubits, &structure).map_err(map_error)?;
        x_words.extend_from_slice(word.x_words());
        z_words.extend_from_slice(word.z_words());
    }
    Ok((word_count, x_words, z_words))
}

#[pyfunction]
fn pauli_multiply(
    nqubits: usize,
    left_codes: Vec<u8>,
    right_codes: Vec<u8>,
) -> PyResult<(Vec<u8>, u8)> {
    let left = PauliWord::from_codes(nqubits, &left_codes).map_err(map_error)?;
    let right = PauliWord::from_codes(nqubits, &right_codes).map_err(map_error)?;
    let (result, phase) = left.multiply(&right).map_err(map_error)?;
    Ok((
        result.codes(),
        match phase {
            tencir_pauli_core::PauliPhase::PlusOne => 0,
            tencir_pauli_core::PauliPhase::PlusI => 1,
            tencir_pauli_core::PauliPhase::MinusOne => 2,
            tencir_pauli_core::PauliPhase::MinusI => 3,
        },
    ))
}

#[pyfunction]
fn pauli_symplectic_inner_product(
    nqubits: usize,
    x_words_left: Vec<u64>,
    z_words_left: Vec<u64>,
    x_words_right: Vec<u64>,
    z_words_right: Vec<u64>,
) -> PyResult<u8> {
    let left = build_word(nqubits, x_words_left, z_words_left)?;
    let right = build_word(nqubits, x_words_right, z_words_right)?;
    left.symplectic_inner_product(&right).map_err(map_error)
}

#[pyfunction]
fn pauli_commutes(
    nqubits: usize,
    x_words_left: Vec<u64>,
    z_words_left: Vec<u64>,
    x_words_right: Vec<u64>,
    z_words_right: Vec<u64>,
) -> PyResult<bool> {
    let left = build_word(nqubits, x_words_left, z_words_left)?;
    let right = build_word(nqubits, x_words_right, z_words_right)?;
    left.commutes_with(&right).map_err(map_error)
}

#[pyfunction]
fn pauli_canonicalize(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
) -> PyResult<CanonicalizeOutput> {
    let operator = build_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
    Ok(operator_output(&operator))
}

#[pyfunction]
fn pauli_canonicalize_batch(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
) -> PyResult<CanonicalizeBatchOutput> {
    let coefficients = complex_coefficients(coefficients_re, coefficients_im)?;
    let result =
        PauliOperator::canonicalize(nqubits, &structures, &coefficients).map_err(map_error)?;
    let mut result_structures = Vec::with_capacity(result.terms.len());
    let mut result_re = Vec::with_capacity(result.terms.len());
    let mut result_im = Vec::with_capacity(result.terms.len());
    for term in result.terms {
        result_structures.push(term.word.codes());
        result_re.push(term.coefficient.re);
        result_im.push(term.coefficient.im);
    }
    Ok((
        result_structures,
        result_re,
        result_im,
        result.input_to_canonical,
        result
            .phase_multipliers
            .into_iter()
            .map(phase_code)
            .collect(),
    ))
}

#[pyfunction]
fn pauli_operator_binary(
    nqubits: usize,
    left: CanonicalizeInput,
    right: CanonicalizeInput,
    operation: u8,
) -> PyResult<CanonicalizeOutput> {
    let left_operator = build_operator(nqubits, &left.0, &left.1, &left.2)?;
    let right_operator = build_operator(nqubits, &right.0, &right.1, &right.2)?;
    let result = match operation {
        0 => left_operator.add(&right_operator),
        1 => left_operator.multiply(&right_operator),
        2 => left_operator.commutator(&right_operator),
        3 => left_operator.anticommutator(&right_operator),
        _ => return Err(PyValueError::new_err("unknown Pauli operator operation")),
    }
    .map_err(map_error)?;
    Ok(operator_output(&result))
}

#[pyfunction]
fn pauli_operator_scale(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    scalar_re: f64,
    scalar_im: f64,
) -> PyResult<CanonicalizeOutput> {
    let operator = build_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
    let result = operator
        .scale(Complex64::new(scalar_re, scalar_im))
        .map_err(map_error)?;
    Ok(operator_output(&result))
}

#[pyfunction]
fn pauli_operator_adjoint(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
) -> PyResult<CanonicalizeOutput> {
    let operator = build_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
    Ok(operator_output(&operator.adjoint()))
}

#[pyfunction]
fn pauli_operator_is_hermitian(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    tolerance: f64,
) -> PyResult<bool> {
    let operator = build_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(PyValueError::new_err(
            "Hermiticity tolerance must be finite and non-negative",
        ));
    }
    Ok(operator.is_hermitian(tolerance))
}

#[pyfunction]
fn pauli_dense(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: usize,
) -> PyResult<DenseOutput> {
    let operator = build_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
    let (dimension, values) = operator
        .dense_matrix(max_bytes as u128)
        .map_err(map_error)?;
    let (real, imaginary) = split_complex(&values);
    Ok((dimension, real, imaginary))
}

#[pyfunction]
fn pauli_dense_array<'py>(
    py: Python<'py>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: usize,
) -> PyResult<(usize, Bound<'py, PyArray1<NumpyComplex128>>)> {
    let operator = build_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
    let (dimension, values) = operator
        .dense_matrix(max_bytes as u128)
        .map_err(map_error)?;
    Ok((dimension, numpy_complex_array(py, values)))
}

#[pyfunction]
fn pauli_coo(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: usize,
) -> PyResult<CooOutput> {
    let operator = build_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
    let matrix = operator.coo_matrix(max_bytes as u128).map_err(map_error)?;
    let (real, imaginary) = split_complex(&matrix.values);
    Ok((
        matrix.dimension,
        matrix.rows,
        matrix.columns,
        real,
        imaginary,
    ))
}

#[pyfunction]
fn pauli_coo_array<'py>(
    py: Python<'py>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: usize,
) -> PyResult<NumpySparseOutput<'py>> {
    let operator = build_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
    let matrix = operator.coo_matrix(max_bytes as u128).map_err(map_error)?;
    Ok((
        matrix.dimension,
        PyArray1::from_vec(py, matrix.rows),
        PyArray1::from_vec(py, matrix.columns),
        numpy_complex_array(py, matrix.values),
    ))
}

#[pyfunction]
fn pauli_csr(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: usize,
) -> PyResult<CsrOutput> {
    let operator = build_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
    let matrix = operator.csr_matrix(max_bytes as u128).map_err(map_error)?;
    let (real, imaginary) = split_complex(&matrix.values);
    Ok((
        matrix.dimension,
        matrix.indptr,
        matrix.columns,
        real,
        imaginary,
    ))
}

#[pyfunction]
fn pauli_csr_array<'py>(
    py: Python<'py>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: usize,
) -> PyResult<NumpySparseOutput<'py>> {
    let operator = build_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
    let matrix = operator.csr_matrix(max_bytes as u128).map_err(map_error)?;
    Ok((
        matrix.dimension,
        PyArray1::from_vec(py, matrix.indptr),
        PyArray1::from_vec(py, matrix.columns),
        numpy_complex_array(py, matrix.values),
    ))
}

#[pyfunction]
fn pauli_mvp(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    state_re: Vec<f64>,
    state_im: Vec<f64>,
    max_bytes: usize,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let operator = build_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
    let state = complex_coefficients(state_re, state_im)?;
    let values = operator.mvp(&state, max_bytes as u128).map_err(map_error)?;
    Ok(split_complex(&values))
}

#[pyfunction]
fn pauli_mvp_array<'py>(
    py: Python<'py>,
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    state: PyReadonlyArray1<'py, NumpyComplex128>,
    max_bytes: usize,
) -> PyResult<Bound<'py, PyArray1<NumpyComplex128>>> {
    let operator = build_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
    let state_slice = state
        .as_slice()
        .map_err(|_| PyValueError::new_err("state must be C-contiguous"))?;
    let output_array = PyArray1::zeros(py, state_slice.len(), false);
    let mut writable: PyReadwriteArray1<'_, NumpyComplex128> = output_array.readwrite();
    let output_slice = writable
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("output must be C-contiguous"))?;
    let state = core_complex_slice(state_slice);
    let output = core_complex_slice_mut(output_slice);
    py.allow_threads(|| operator.mvp_into(state, output, max_bytes as u128))
        .map_err(map_error)?;
    drop(writable);
    Ok(output_array)
}

#[pyfunction]
fn pauli_mvp_plan(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: usize,
) -> PyResult<NativeMvpPlan> {
    let operator = build_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
    let plan = operator.mvp_plan(max_bytes as u128).map_err(map_error)?;
    Ok(NativeMvpPlan { plan })
}

#[pyfunction]
fn pauli_backend_plan(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    coefficients_re: Vec<f64>,
    coefficients_im: Vec<f64>,
    max_bytes: usize,
) -> PyResult<BackendPlanOutput> {
    let operator = build_operator(nqubits, &structures, &coefficients_re, &coefficients_im)?;
    let plan = operator
        .backend_mvp_plan(max_bytes as u128)
        .map_err(map_error)?;
    let (real, imaginary) = split_complex(&plan.coefficients);
    Ok((
        1,
        plan.nqubits,
        plan.word_count,
        plan.x_words,
        plan.z_words,
        real,
        imaginary,
    ))
}

#[pyfunction]
fn pauli_group(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    mode: u8,
    algorithm: u8,
) -> PyResult<Vec<Vec<usize>>> {
    let words = structures
        .iter()
        .map(|structure| PauliWord::from_codes(nqubits, structure).map_err(map_error))
        .collect::<PyResult<Vec<_>>>()?;
    let grouping_mode = match mode {
        0 => GroupingMode::QubitWise,
        1 => GroupingMode::General,
        _ => return Err(PyValueError::new_err("grouping mode must be 0 or 1")),
    };
    let grouping_algorithm = match algorithm {
        0 => GroupingAlgorithm::LargestFirst,
        1 => GroupingAlgorithm::Dsatur,
        _ => return Err(PyValueError::new_err("grouping algorithm must be 0 or 1")),
    };
    group_words(&words, grouping_mode, grouping_algorithm).map_err(map_error)
}

fn parse_grouping_mode(mode: u8) -> PyResult<GroupingMode> {
    match mode {
        0 => Ok(GroupingMode::QubitWise),
        1 => Ok(GroupingMode::General),
        _ => Err(PyValueError::new_err("grouping mode must be 0 or 1")),
    }
}

fn build_grouping_words(nqubits: usize, structures: &[Vec<u8>]) -> PyResult<Vec<PauliWord>> {
    structures
        .iter()
        .map(|structure| PauliWord::from_codes(nqubits, structure).map_err(map_error))
        .collect()
}

#[pyfunction]
fn pauli_compatibility_matrix(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    mode: u8,
    max_entries: usize,
) -> PyResult<Vec<bool>> {
    let words = build_grouping_words(nqubits, &structures)?;
    compatibility_matrix(&words, parse_grouping_mode(mode)?, max_entries).map_err(map_error)
}

#[pyfunction]
fn pauli_incompatibility_edges(
    nqubits: usize,
    structures: Vec<Vec<u8>>,
    mode: u8,
    max_edges: usize,
) -> PyResult<Vec<(usize, usize)>> {
    let words = build_grouping_words(nqubits, &structures)?;
    incompatibility_edges(&words, parse_grouping_mode(mode)?, max_edges).map_err(map_error)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<NativeMvpPlan>()?;
    module.add_function(wrap_pyfunction!(pauli_weight, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_support, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_codes, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_from_codes, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_batch_from_codes, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_multiply, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_symplectic_inner_product, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_commutes, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_canonicalize, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_canonicalize_batch, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_operator_binary, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_operator_scale, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_operator_adjoint, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_operator_is_hermitian, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_dense, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_dense_array, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_coo, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_coo_array, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_csr, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_csr_array, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_mvp, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_mvp_array, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_mvp_plan, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_backend_plan, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_group, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_compatibility_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(pauli_incompatibility_edges, module)?)?;
    Ok(())
}
