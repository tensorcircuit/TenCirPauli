"""JAX boundary tests for scalar circuit expectations."""

from __future__ import annotations

import numpy as np
import pytest

import tencirpauli as tcp
from tencirpauli import advanced


def _jax() -> object:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


def test_propagation_jax_pytree_chain_rule_matches_independent_differences() -> None:
    jax = _jax()
    jnp = jax.numpy
    observable = tcp.PauliOperator.from_terms(1, [("X", 1.0)])

    def objective(tree: dict[str, object]) -> object:
        circuit = tcp.PropagationCircuit(1)
        circuit.ry(0, theta=tree["shared"])
        circuit.ry(0, theta=tree["shared"])
        circuit.rz(0, theta=2.0 * tree["scale"] + jnp.sin(tree["smooth"]))
        return circuit.expectation_jax(observable)

    point = {
        "shared": jnp.asarray(0.13),
        "scale": jnp.asarray(0.2),
        "smooth": jnp.asarray(-0.1),
    }
    eager = jax.value_and_grad(objective)(point)
    compiled = jax.jit(jax.value_and_grad(objective))(point)
    np.testing.assert_allclose(eager[0], compiled[0])
    np.testing.assert_allclose(eager[1]["shared"], compiled[1]["shared"])
    np.testing.assert_allclose(eager[1]["scale"], compiled[1]["scale"])
    np.testing.assert_allclose(eager[1]["smooth"], compiled[1]["smooth"])

    def concrete(shared: float, scale: float, smooth: float) -> float:
        circuit = tcp.PropagationCircuit(1)
        circuit.ry(0, theta=shared)
        circuit.ry(0, theta=shared)
        circuit.rz(0, theta=2.0 * scale + np.sin(smooth))
        return circuit.expectation(observable)

    coordinates = np.asarray([0.13, 0.2, -0.1], dtype=np.float64)
    expected = []
    for index in range(3):
        plus = coordinates.copy()
        minus = coordinates.copy()
        plus[index] += 1.0e-6
        minus[index] -= 1.0e-6
        expected.append((concrete(*plus) - concrete(*minus)) / (2.0e-6))
    np.testing.assert_allclose(
        np.asarray([eager[1]["shared"], eager[1]["scale"], eager[1]["smooth"]]),
        expected,
        atol=3.0e-6,
        rtol=2.0e-6,
    )
    assert np.all(np.abs(expected) > 0.05)


def test_jax_callback_executes_once_and_vjp_does_not_reenter_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jax = _jax()
    jnp = jax.numpy
    observable = tcp.PauliOperator.from_terms(1, [("X", 1.0)])
    calls = 0
    original = advanced.PropagationEngine.value_and_grad

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(advanced.PropagationEngine, "value_and_grad", counted)

    def objective(angle: object) -> object:
        circuit = tcp.PropagationCircuit(1)
        circuit.ry(0, theta=angle)
        return circuit.expectation_jax(observable)

    runner = jax.jit(jax.value_and_grad(objective))
    result = runner(jnp.asarray(0.23, dtype=jnp.float64))
    for leaf in jax.tree_util.tree_leaves(result):
        leaf.block_until_ready()
    assert calls == 1

    calls = 0
    second = runner(jnp.asarray(0.31, dtype=jnp.float64))
    for leaf in jax.tree_util.tree_leaves(second):
        leaf.block_until_ready()
    assert calls == 1


def test_jitted_persistent_circuit_keeps_an_immutable_objective_snapshot() -> None:
    jax = _jax()
    observable = tcp.PauliOperator.from_terms(1, [("X", 1.0)])
    circuit = tcp.PropagationCircuit(1)
    circuit.ry(0, theta=0.2)
    runner = jax.jit(lambda: circuit.expectation_jax(observable))
    snapshot = runner().block_until_ready()

    circuit.rz(0, theta=0.4)
    changed = circuit.expectation(observable)
    replayed = runner().block_until_ready()
    assert changed != pytest.approx(float(snapshot))
    assert replayed == pytest.approx(float(snapshot))


def test_u1_and_spps_jax_gradients_use_occurrence_space() -> None:
    jax = _jax()
    jnp = jax.numpy
    observable = tcp.PauliOperator.from_terms(2, [("XX", 0.4), ("ZI", -0.7)])

    def u1_objective(x: object) -> object:
        circuit = tcp.U1Circuit(2, particle_number=1, occupied=[0])
        circuit.iswap(0, 1, theta=2.0 * x[0] + jnp.sin(x[1]))
        return circuit.expectation_jax(observable)

    point = jnp.asarray([0.13, -0.27])
    value, gradient = jax.value_and_grad(u1_objective)(point)
    native_circuit = tcp.U1Circuit(2, particle_number=1, occupied=[0])
    native_circuit.iswap(0, 1, theta=float(2.0 * point[0] + jnp.sin(point[1])))
    native = native_circuit.value_and_grad(observable)
    np.testing.assert_allclose(value, native.value)
    np.testing.assert_allclose(
        gradient,
        native.gradient[0] * np.asarray([2.0, np.cos(float(point[1]))]),
        atol=2e-12,
    )

    spps_observable = tcp.PauliOperator.from_terms(1, [("Z", 1.0)])

    def spps_objective(x: object) -> object:
        circuit = tcp.SPPSCircuit(1)
        circuit.ry(0, theta=x)
        return circuit.expectation_jax(spps_observable, samples_per_term=64, seed=19)

    spps_value, spps_gradient = jax.jit(jax.value_and_grad(spps_objective))(
        jnp.asarray(0.2)
    )
    native_spps_circuit = tcp.SPPSCircuit(1)
    native_spps_circuit.ry(0, theta=0.2)
    native_spps = native_spps_circuit.value_and_grad(
        spps_observable, samples_per_term=64, seed=19
    )
    np.testing.assert_allclose(spps_value, native_spps.value)
    np.testing.assert_allclose(spps_gradient, native_spps.gradient[0])


def test_u1_jax_pytree_repeated_leaf_scatter_matches_native() -> None:
    jax = _jax()
    jnp = jax.numpy
    observable = tcp.PauliOperator.from_terms(2, [("ZI", 1.0)])

    def objective(tree: dict[str, object]) -> object:
        circuit = tcp.U1Circuit(2, particle_number=1, occupied=[0])
        circuit.iswap(0, 1, theta=tree["shared"])
        circuit.iswap(0, 1, theta=tree["shared"])
        circuit.iswap(
            0,
            1,
            theta=2.0 * tree["scale"] + jnp.sin(tree["smooth"]),
        )
        return circuit.expectation_jax(observable)

    point = {
        "shared": jnp.asarray(0.13),
        "scale": jnp.asarray(0.2),
        "smooth": jnp.asarray(-0.1),
    }
    value, gradient = jax.jit(jax.value_and_grad(objective))(point)
    concrete = tcp.U1Circuit(2, particle_number=1, occupied=[0])
    concrete.iswap(0, 1, theta=0.13)
    concrete.iswap(0, 1, theta=0.13)
    concrete.iswap(0, 1, theta=2.0 * 0.2 + np.sin(-0.1))
    native = concrete.value_and_grad(observable)
    expected_gradient = {
        "shared": native.gradient[0] + native.gradient[1],
        "scale": 2.0 * native.gradient[2],
        "smooth": np.cos(-0.1) * native.gradient[2],
    }
    np.testing.assert_allclose(value, native.value, atol=2e-12, rtol=2e-12)
    for key, expected in expected_gradient.items():
        np.testing.assert_allclose(gradient[key], expected, atol=2e-12, rtol=2e-12)


def test_spps_jax_pytree_repeated_leaf_scatter_matches_native() -> None:
    jax = _jax()
    jnp = jax.numpy
    observable = tcp.PauliOperator.from_terms(1, [("Z", 1.0)])

    def objective(tree: dict[str, object]) -> object:
        circuit = tcp.SPPSCircuit(1)
        circuit.ry(0, theta=tree["shared"])
        circuit.ry(0, theta=tree["shared"])
        return circuit.expectation_jax(observable, samples_per_term=128, seed=37)

    point = {"shared": jnp.asarray(0.17)}
    value, gradient = jax.jit(jax.value_and_grad(objective))(point)
    concrete = tcp.SPPSCircuit(1)
    concrete.ry(0, theta=0.17)
    concrete.ry(0, theta=0.17)
    native = concrete.value_and_grad(observable, samples_per_term=128, seed=37)
    np.testing.assert_allclose(value, native.value, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        gradient["shared"],
        native.gradient[0] + native.gradient[1],
        atol=1e-12,
        rtol=1e-12,
    )


def test_jax_terminal_requires_float64() -> None:
    jax = pytest.importorskip("jax")
    previous = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", False)
    try:
        circuit = tcp.PropagationCircuit(1)
        circuit.ry(0, 0.2)
        observable = tcp.PauliOperator.from_terms(1, [("Z", 1.0)])
        with pytest.raises(ValueError, match="jax_enable_x64"):
            circuit.expectation_jax(observable)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_jax_static_controls_fail_before_callback_staging() -> None:
    _jax()
    nonhermitian = tcp.PauliOperator.from_terms(1, [("X", 1.0j)])
    propagation = tcp.PropagationCircuit(1)
    propagation.ry(0, theta=0.2)
    with pytest.raises(ValueError, match="Hermitian"):
        propagation.expectation_jax(nonhermitian)
    observable = tcp.PauliOperator.from_terms(1, [("Z", 1.0)])
    with pytest.raises(ValueError, match="checkpoint_interval"):
        propagation.expectation_jax(observable, checkpoint_interval=0)

    spps = tcp.SPPSCircuit(1)
    spps.ry(0, theta=0.2)
    with pytest.raises(ValueError, match="samples_per_term"):
        spps.expectation_jax(observable, samples_per_term=1, seed=0)
    with pytest.raises(ValueError, match="seed"):
        spps.expectation_jax(observable, samples_per_term=2, seed=-1)
