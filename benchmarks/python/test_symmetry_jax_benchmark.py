"""Matched symmetry-aware JAX baselines for reduced-space MVP workloads."""

from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from tencirpauli import PauliOperator, U1Sector


MAX_BYTES = 4 * 1024**3


def make_hopping(nqubits: int = 26) -> PauliOperator:
    terms = []
    for index in range(nqubits - 1):
        prefix = "I" * index
        suffix = "I" * (nqubits - 2 - index)
        terms.extend(((prefix + "XX" + suffix, 0.5), (prefix + "YY" + suffix, 0.5)))
    return PauliOperator.from_terms(nqubits, terms)


def make_tfim(nqubits: int = 20) -> PauliOperator:
    terms = [("X" * nqubits, 0.25)]
    terms.extend(
        ("I" * index + "ZZ" + "I" * (nqubits - 2 - index), -1.0)
        for index in range(nqubits - 1)
    )
    terms.extend(
        ("I" * index + "X" + "I" * (nqubits - 1 - index), -0.2)
        for index in range(nqubits)
    )
    return PauliOperator.from_terms(nqubits, terms)


def python_tfim_taper(nqubits: int, sector: int) -> PauliOperator:
    """Apply the known global-X taper in Python for a favorable baseline."""
    if sector not in (-1, 1):
        raise ValueError("sector must be +1 or -1")
    terms = [("I" * (nqubits - 1), 0.25 * sector)]
    terms.append(("Z" + "I" * (nqubits - 2), -1.0))
    for index in range(1, nqubits - 1):
        terms.append(("I" * (index - 1) + "ZZ" + "I" * (nqubits - 2 - index), -1.0))
    terms.extend(
        ("I" * (index - 1) + "X" + "I" * (nqubits - 1 - index), -0.2)
        for index in range(1, nqubits)
    )
    terms.append(("X" * (nqubits - 1), -0.2))
    return PauliOperator.from_terms(nqubits - 1, terms)


def _transition_table(
    operator: PauliOperator, sector: U1Sector
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Build the exact fixed-sector COO transitions for the JAX baseline."""
    basis = np.asarray(sector.basis_words(), dtype=np.uint64)
    structures, coefficients_re, coefficients_im = operator._arrays()
    basis_index = {int(value): index for index, value in enumerate(basis)}
    terms = []
    nqubits = operator.nqubits
    for structure, real, imaginary in zip(structures, coefficients_re, coefficients_im):
        x_mask = 0
        z_mask = 0
        y_count = 0
        for qubit, code in enumerate(structure):
            matrix_mask = 1 << (nqubits - 1 - qubit)
            if code in (1, 2):
                x_mask |= matrix_mask
            if code in (2, 3):
                z_mask |= matrix_mask
            if code == 2:
                y_count += 1
        weighted = complex(real, imaginary) * (1j**y_count)
        terms.append((x_mask, z_mask, weighted))
    rows = []
    columns = []
    values = []
    for column, source in enumerate(basis):
        aggregate = {}
        source_value = int(source)
        for x_mask, z_mask, weighted in terms:
            destination = source_value ^ x_mask
            sign = -1.0 if (z_mask & source_value).bit_count() & 1 else 1.0
            aggregate[destination] = aggregate.get(destination, 0.0) + weighted * sign
        for destination, value in aggregate.items():
            if value == 0.0:
                continue
            if destination not in basis_index:
                raise ValueError("U1 baseline detected sector leakage")
            rows.append(basis_index[destination])
            columns.append(column)
            values.append(value)
    return (
        np.asarray(rows, dtype=np.int32),
        np.asarray(columns, dtype=np.int32),
        np.asarray(values, dtype=np.complex128),
    )


def make_jax_u1_baseline(
    operator: PauliOperator, sector: U1Sector
) -> Callable[[Any], Any]:
    """Compile a JAX scatter-add MVP over the same fixed-particle sector."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    rows, columns, values = _transition_table(operator, sector)
    j_rows = jnp.asarray(rows)
    j_columns = jnp.asarray(columns)
    j_values = jnp.asarray(values)
    # Device transfers are part of setup; force them complete inside the
    # timed callable so the setup benchmark cannot measure enqueue latency.
    _sync_jax(j_rows)
    _sync_jax(j_columns)
    _sync_jax(j_values)
    dimension = sector.dimension

    def apply(state: Any) -> Any:
        output = jnp.zeros((dimension,), dtype=state.dtype)
        return output.at[j_rows].add(j_values * state[j_columns])

    return jax.jit(apply)


def make_jax_pauli_baseline(operator: PauliOperator) -> Callable[[Any], Any]:
    """Compile a JAX MVP for a supplied tapered Pauli operator."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import lax

    structures, coefficients_re, coefficients_im = operator._arrays()
    nqubits = operator.nqubits
    x_masks = []
    z_masks = []
    coefficients = []
    for structure, real, imaginary in zip(structures, coefficients_re, coefficients_im):
        x_mask = 0
        z_mask = 0
        y_count = 0
        for qubit, code in enumerate(structure):
            matrix_mask = 1 << (nqubits - 1 - qubit)
            if code in (1, 2):
                x_mask |= matrix_mask
            if code in (2, 3):
                z_mask |= matrix_mask
            if code == 2:
                y_count += 1
        x_masks.append(x_mask)
        z_masks.append(z_mask)
        coefficients.append(complex(real, imaginary) * (1j**y_count))
    j_x_masks = tuple(jnp.uint32(mask) for mask in x_masks)
    j_z_masks = tuple(jnp.uint32(mask) for mask in z_masks)
    j_coefficients = tuple(jnp.complex128(value) for value in coefficients)
    dimension = 1 << nqubits

    def apply(state: Any) -> Any:
        columns = jnp.arange(dimension, dtype=jnp.uint32)
        output = jnp.zeros((dimension,), dtype=state.dtype)
        for x_mask, z_mask, coefficient in zip(j_x_masks, j_z_masks, j_coefficients):
            rows = jnp.bitwise_xor(columns, x_mask)
            parity = lax.population_count(jnp.bitwise_and(columns, z_mask)) & 1
            phase = jnp.where(parity == 0, 1.0, -1.0)
            output = output.at[rows].add(coefficient * phase * state)
        return output

    return jax.jit(apply)


def _sync_jax(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return value


@pytest.mark.performance_large
def test_u1_26q_rust_restriction_setup(benchmark: BenchmarkFixture) -> None:
    operator = make_hopping()
    sector = U1Sector(26, 2)
    expected = operator.restrict_u1(sector)
    result = benchmark(operator.restrict_u1, sector)
    assert result.dimension == expected.dimension


@pytest.mark.performance_large
def test_u1_26q_jax_restriction_setup(benchmark: BenchmarkFixture) -> None:
    operator = make_hopping()
    sector = U1Sector(26, 2)
    expected = make_jax_u1_baseline(operator, sector)
    result = benchmark(make_jax_u1_baseline, operator, sector)
    assert callable(result) and callable(expected)


@pytest.mark.performance_large
def test_u1_26q_rust_restricted_mvp(benchmark: BenchmarkFixture) -> None:
    operator = make_hopping()
    sector = U1Sector(26, 2)
    plan = operator.restrict_u1(sector, MAX_BYTES).mvp_plan(max_bytes=MAX_BYTES)
    state = np.arange(plan.dimension, dtype=np.float64) + 1j * np.arange(plan.dimension)
    expected = plan.apply(state, max_bytes=MAX_BYTES)
    result = benchmark.pedantic(
        plan.apply,
        args=(state,),
        kwargs={"max_bytes": MAX_BYTES},
        rounds=10,
        iterations=1,
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.performance_large
def test_u1_26q_jax_restricted_mvp(benchmark: BenchmarkFixture) -> None:
    operator = make_hopping()
    sector = U1Sector(26, 2)
    apply = make_jax_u1_baseline(operator, sector)
    import jax.numpy as jnp

    state = jnp.arange(sector.dimension, dtype=jnp.float64) + 1j * jnp.arange(
        sector.dimension, dtype=jnp.float64
    )
    expected = _sync_jax(apply(state))
    rust_plan = operator.restrict_u1(sector, MAX_BYTES).mvp_plan(max_bytes=MAX_BYTES)
    np.testing.assert_allclose(np.asarray(expected), rust_plan.apply(np.asarray(state)))

    def apply_sync(value: Any) -> Any:
        return _sync_jax(apply(value))

    result = benchmark.pedantic(apply_sync, args=(state,), rounds=10, iterations=1)
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected))


@pytest.mark.performance_large
def test_u1_26q_rust_end_to_end(benchmark: BenchmarkFixture) -> None:
    """Measure Python endpoint -> Rust restriction -> Rust MVP."""
    operator = make_hopping()
    sector = U1Sector(26, 2)
    state = np.arange(sector.dimension, dtype=np.float64) + 1j * np.arange(
        sector.dimension
    )

    def apply_end_to_end() -> Any:
        restricted = operator.restrict_u1(sector, MAX_BYTES)
        return restricted.mvp_plan(max_bytes=MAX_BYTES).apply(
            state, max_bytes=MAX_BYTES
        )

    expected = apply_end_to_end()
    result = benchmark.pedantic(apply_end_to_end, rounds=5, iterations=1)
    np.testing.assert_allclose(result, expected)


@pytest.mark.performance_large
def test_u1_26q_jax_end_to_end(benchmark: BenchmarkFixture) -> None:
    """Measure Python endpoint -> JAX transition setup -> first compiled MVP."""
    operator = make_hopping()
    sector = U1Sector(26, 2)
    import jax.numpy as jnp

    state = jnp.arange(sector.dimension, dtype=jnp.float64) + 1j * jnp.arange(
        sector.dimension, dtype=jnp.float64
    )

    def apply_end_to_end() -> Any:
        apply = make_jax_u1_baseline(operator, sector)
        return _sync_jax(apply(state))

    expected = apply_end_to_end()
    rust_plan = operator.restrict_u1(sector, MAX_BYTES).mvp_plan(max_bytes=MAX_BYTES)
    np.testing.assert_allclose(
        np.asarray(expected), rust_plan.apply(np.asarray(state)), rtol=1e-12, atol=1e-12
    )
    result = benchmark.pedantic(
        apply_end_to_end, rounds=3, iterations=1, warmup_rounds=0
    )
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected))


@pytest.mark.performance_large
def test_z2_20q_rust_tapered_mvp(benchmark: BenchmarkFixture) -> None:
    operator = make_tfim()
    state = np.arange(1 << 19, dtype=np.float64) + 1j * np.arange(1 << 19)
    analysis = operator.find_z2_symmetries()
    tapered = analysis.tapering_plan((1,)).transform_operator(operator)
    plan = tapered.native_mvp_plan(max_bytes=MAX_BYTES)
    expected = plan.apply(state, max_bytes=MAX_BYTES)
    result = benchmark.pedantic(
        plan.apply,
        args=(state,),
        kwargs={"max_bytes": MAX_BYTES},
        rounds=5,
        iterations=1,
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.performance_large
def test_z2_20q_jax_tapered_mvp(benchmark: BenchmarkFixture) -> None:
    operator = make_tfim()
    tapered = python_tfim_taper(operator.nqubits, 1)
    apply = make_jax_pauli_baseline(tapered)
    import jax.numpy as jnp

    state = jnp.arange(1 << 19, dtype=jnp.float64) + 1j * jnp.arange(
        1 << 19, dtype=jnp.float64
    )
    expected = _sync_jax(apply(state))
    rust_tapered = (
        operator.find_z2_symmetries().tapering_plan((1,)).transform_operator(operator)
    )
    rust_expected = rust_tapered.native_mvp_plan(max_bytes=MAX_BYTES).apply(
        np.asarray(state), max_bytes=MAX_BYTES
    )
    np.testing.assert_allclose(
        np.asarray(expected), rust_expected, rtol=1e-10, atol=1e-9
    )

    def apply_sync(value: Any) -> Any:
        return _sync_jax(apply(value))

    result = benchmark.pedantic(apply_sync, args=(state,), rounds=5, iterations=1)
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected))


@pytest.mark.performance_large
def test_z2_20q_rust_end_to_end(benchmark: BenchmarkFixture) -> None:
    """Measure Python endpoint -> generic Rust Z2 taper -> Rust MVP."""
    operator = make_tfim()
    state = np.arange(1 << 19, dtype=np.float64) + 1j * np.arange(1 << 19)

    def apply_end_to_end() -> Any:
        tapered = (
            operator.find_z2_symmetries()
            .tapering_plan((1,))
            .transform_operator(operator)
        )
        return tapered.native_mvp_plan(max_bytes=MAX_BYTES).apply(
            state, max_bytes=MAX_BYTES
        )

    expected = apply_end_to_end()
    result = benchmark.pedantic(
        apply_end_to_end, rounds=3, iterations=1, warmup_rounds=0
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.performance_large
def test_z2_20q_jax_end_to_end(benchmark: BenchmarkFixture) -> None:
    """Measure Python known-symmetry taper -> JAX setup -> first compiled MVP."""
    operator = make_tfim()
    import jax.numpy as jnp

    state = jnp.arange(1 << 19, dtype=jnp.float64) + 1j * jnp.arange(
        1 << 19, dtype=jnp.float64
    )

    def apply_end_to_end() -> Any:
        tapered = python_tfim_taper(operator.nqubits, 1)
        apply = make_jax_pauli_baseline(tapered)
        return _sync_jax(apply(state))

    expected = apply_end_to_end()
    result = benchmark.pedantic(
        apply_end_to_end, rounds=3, iterations=1, warmup_rounds=0
    )
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected))
