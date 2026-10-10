"""Backend MVP callables can be reused across JAX transformation scopes."""

from __future__ import annotations

import numpy as np
import pytest
import tensorcircuit as tc

from tencirpauli import PauliOperator, QuditWeylOperator, backend_mvp


@pytest.mark.parametrize("kind", ["pauli", "weyl"])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_backend_mvp_jit_first_then_eager_retrace_and_grad(
    kind: str, dtype: str
) -> None:
    jax = pytest.importorskip("jax")
    if kind == "pauli":
        operator = PauliOperator.from_terms(2, [("XY", 0.7), ("YZ", -0.2)])
    else:
        operator = QuditWeylOperator.from_terms(
            3, [(((0, 1, 2),), 0.7 - 0.2j), (((1, 2, 1),), 0.3 + 0.1j)], n_sites=2
        )
    dense = operator.compile("dense")
    with jax.enable_x64():
        backend = tc.get_backend("jax")
        apply = backend_mvp(operator.compile("backend_mvp"), backend=backend)
        state = jax.numpy.arange(dense.shape[0], dtype=jax.numpy.float64).astype(dtype)
        expected = dense @ np.asarray(state)
        first = jax.jit(apply)(state)
        eager = apply(state)
        retraced = jax.jit(lambda x: apply(x))(state)
        for result in [first, eager, retraced]:
            assert str(result.dtype) == dtype
            np.testing.assert_allclose(result, expected, atol=2e-5, rtol=2e-5)

        def norm_squared(real_state):
            result = apply(real_state.astype(dtype))
            return jax.numpy.real(jax.numpy.vdot(result, result))

        actual = jax.jit(jax.grad(norm_squared))(state.real)
        expected_gradient = 2 * np.real(dense.conj().T @ expected)
        np.testing.assert_allclose(actual, expected_gradient, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("kind", ["pauli", "weyl"])
def test_backend_mvp_differentiates_replacement_coefficients(kind: str) -> None:
    jax = pytest.importorskip("jax")
    with jax.enable_x64():
        backend = tc.get_backend("jax")
        if kind == "pauli":
            operator = PauliOperator.from_terms(1, [("Z", 1.0)])
            amplitudes = [1.0, 0.0]
        else:
            operator = QuditWeylOperator.from_terms(3, [(((0, 0, 1),), 1.0)], n_sites=1)
            amplitudes = [1.0, 0.0, 0.0]
        plan = operator.compile("backend_mvp")
        state = jax.numpy.asarray(amplitudes, dtype=jax.numpy.complex128)

        def objective(coefficients):
            apply = backend_mvp(plan, coefficients=coefficients, backend=backend)
            return jax.numpy.real(jax.numpy.vdot(state, apply(state)))

        value, gradient = jax.jit(jax.value_and_grad(objective))(jax.numpy.array([0.3]))
        np.testing.assert_allclose(value, 0.3)
        np.testing.assert_allclose(gradient, [1.0])
