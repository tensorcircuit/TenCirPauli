"""Correctness and boundary tests for the deterministic propagation propagation engine."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import pytest
from propagation_reference import product_expectation, propagate_dense
from reference import PAULI_MATRICES, codes_to_dense

import tencirpauli as tcp
from tencirpauli import advanced


def _terms(operator: tcp.PauliOperator) -> dict[tuple[int, ...], complex]:
    return {term.word.to_codes(): term.coefficient for term in operator.terms}


@pytest.mark.parametrize(
    ("name", "operation"),
    [
        ("x", ("x", 0)),
        ("y", ("y", 0)),
        ("z", ("z", 0)),
        ("h", ("h", 0)),
        ("s", ("s", 0)),
        ("sdg", ("sdg", 0)),
    ],
)
def test_one_qubit_cliffords_match_dense_for_all_local_paulis(
    name: str, operation: tuple[object, ...]
) -> None:
    for code in range(4):
        tape = advanced.GateTape(1)
        getattr(tape, name)(0)
        result = advanced.PropagationEngine(
            tape, tcp.PauliOperator(1, [((code,), 1.0)])
        ).propagate_operator([])
        expected = propagate_dense(1, ((code,),), (1.0,), (operation,))
        assert _terms(result) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("name", "operation"),
    [
        ("cnot", ("cnot", 0, 1)),
        ("cz", ("cz", 0, 1)),
        ("swap", ("swap", 0, 1)),
    ],
)
def test_two_qubit_cliffords_match_dense_for_all_local_paulis(
    name: str, operation: tuple[object, ...]
) -> None:
    for left, right in product(range(4), repeat=2):
        tape = advanced.GateTape(2)
        getattr(tape, name)(0, 1)
        result = advanced.PropagationEngine(
            tape, tcp.PauliOperator(2, [((left, right), 1.0)])
        ).propagate_operator([])
        expected = propagate_dense(2, ((left, right),), (1.0,), (operation,))
        assert _terms(result) == pytest.approx(expected)


@pytest.mark.parametrize("name", ("rx", "ry", "rz"))
def test_parameterized_one_qubit_rotation_convention(name: str) -> None:
    theta = 0.371
    for code in range(4):
        tape = advanced.GateTape(1)
        getattr(tape, name)(0, angle=theta)
        result = advanced.PropagationEngine(
            tape, tcp.PauliOperator(1, [((code,), 1.0)])
        ).propagate_operator([])
        expected = propagate_dense(1, ((code,),), (1.0,), ((name, 0, theta),))
        assert _terms(result) == pytest.approx(expected, abs=1e-12)


def test_reverse_heisenberg_order_and_parameter_slots() -> None:
    tape = advanced.GateTape(1)
    tape.h(0)
    tape.rz(0, parameter=0)
    result = advanced.PropagationEngine(
        tape, tcp.PauliOperator(1, [((3,), 1.0)])
    ).propagate_operator([np.pi / 2])
    expected = propagate_dense(1, ((3,),), (1.0,), (("h", 0), ("rz", 0, np.pi / 2)))
    assert _terms(result) == pytest.approx(expected, abs=1e-12)
    assert tape.nparameters == 1


def test_projection_is_initial_and_per_gate_after_aggregation() -> None:
    initial_only = advanced.PropagationEngine(
        advanced.GateTape(1), tcp.PauliOperator(1, [((1,), 1.0)]), max_weight=0
    ).propagate_operator([])
    assert initial_only.terms == ()

    tape = advanced.GateTape(1)
    tape.rz(0, angle=np.pi / 3)
    observable = tcp.PauliOperator(1, [((1,), 1.0), ((2,), 1.0)])
    result = advanced.PropagationEngine(
        tape, observable, max_weight=0
    ).propagate_operator([])
    assert result.terms == ()

    exact = advanced.PropagationEngine(
        tape, observable, max_weight=None
    ).propagate_operator([])
    assert _terms(exact) == pytest.approx(
        propagate_dense(1, ((1,), (2,)), (1.0, 1.0), (("rz", 0, np.pi / 3),)),
        abs=1e-12,
    )


def test_aggregation_cancellation_is_applied_after_collision() -> None:
    collision = np.zeros((4, 4), dtype=np.float64)
    collision[0, 1] = 1.0
    collision[0, 2] = 1.0
    tape = advanced.GateTape(1)
    tape.ptm((0,), collision)
    cancelling = advanced.PropagationEngine(
        tape,
        tcp.PauliOperator(1, [((1,), 1.0), ((2,), -1.0)]),
        max_weight=0,
    ).propagate_operator([])
    assert cancelling.terms == ()


def test_random_seeded_exact_and_projected_cases_match_dense_reference() -> None:
    rng = np.random.default_rng(20260802)
    for _ in range(8):
        nqubits = 3
        tape = advanced.GateTape(nqubits)
        operations: list[tuple[object, ...]] = []
        for _ in range(5):
            kind = int(rng.integers(0, 6))
            wire0 = int(rng.integers(0, nqubits))
            if kind < 3:
                name = ("h", "s", "sdg")[kind]
                getattr(tape, name)(wire0)
                operations.append((name, wire0))
            elif kind == 3:
                angle = float(rng.uniform(-0.7, 0.7))
                tape.rz(wire0, angle=angle)
                operations.append(("rz", wire0, angle))
            else:
                wire1 = (wire0 + int(rng.integers(1, nqubits))) % nqubits
                name = "cnot" if kind == 4 else "cz"
                getattr(tape, name)(wire0, wire1)
                operations.append((name, wire0, wire1))
        structures = [
            [int(value) for value in rng.integers(0, 4, size=nqubits)] for _ in range(4)
        ]
        coefficients = [float(value) for value in rng.uniform(-1.0, 1.0, size=4)]
        observable = tcp.PauliOperator(nqubits, list(zip(structures, coefficients)))
        for max_weight in (None, int(rng.integers(0, nqubits + 1))):
            result = advanced.PropagationEngine(
                tape, observable, max_weight=max_weight
            ).propagate_operator([])
            expected = propagate_dense(
                nqubits,
                structures,
                coefficients,
                operations,
                max_weight=max_weight,
            )
            assert _terms(result) == pytest.approx(expected, abs=1e-10)


def test_finite_projection_matches_independent_every_gate_reference() -> None:
    tape = advanced.GateTape(3)
    tape.rz(0, angle=np.pi / 4)
    tape.cnot(0, 1)
    tape.ry(2, angle=-np.pi / 5)
    operations = (("rz", 0, np.pi / 4), ("cnot", 0, 1), ("ry", 2, -np.pi / 5))
    observable = tcp.PauliOperator(
        3,
        [((1, 1, 1), 0.7), ((3, 0, 3), -0.2), ((0, 2, 0), 0.4)],
    )
    result = advanced.PropagationEngine(
        tape, observable, max_weight=1
    ).propagate_operator([])
    expected = propagate_dense(
        3,
        ((1, 1, 1), (3, 0, 3), (0, 2, 0)),
        (0.7, -0.2, 0.4),
        operations,
        max_weight=1,
    )
    assert _terms(result) == pytest.approx(expected, abs=1e-12)


def test_none_nqubits_and_larger_cutoffs_are_exactly_equal() -> None:
    tape = advanced.GateTape(2)
    tape.rx(0, parameter=0)
    tape.cnot(0, 1)
    observable = tcp.PauliOperator(2, [((1, 0), 0.7), ((0, 3), -0.2)])
    outputs = [
        advanced.PropagationEngine(
            tape, observable, max_weight=cutoff
        ).propagate_operator([0.4])
        for cutoff in (None, 2, 3)
    ]
    assert _terms(outputs[0]) == _terms(outputs[1]) == _terms(outputs[2])


def test_expectation_matches_materialization_for_all_product_state_descriptors() -> (
    None
):
    tape = advanced.GateTape(2)
    tape.ry(0, parameter=0)
    tape.cnot(0, 1)
    observable = tcp.PauliOperator(2, [((1, 0), 0.4), ((3, 3), -0.2)])
    params = [0.27]
    for descriptor, reference_state in (
        (tcp.ZeroState(), "zero"),
        (tcp.ComputationalBasisState((1, 0)), (1, 0)),
        (
            tcp.ProductBlochState(np.array([[0.2, -0.1, 0.7], [0.0, 0.3, 0.4]])),
            np.array([[0.2, -0.1, 0.7], [0.0, 0.3, 0.4]]),
        ),
    ):
        engine = advanced.PropagationEngine(tape, observable, initial_state=descriptor)
        materialized = engine.propagate_operator(params)
        dense = sum(
            (
                term.coefficient * codes_to_dense(term.word.to_codes())
                for term in materialized.terms
            ),
            np.zeros((4, 4), dtype=np.complex128),
        )
        assert engine.expectation(params) == pytest.approx(
            product_expectation(dense, reference_state), abs=1e-12
        )
        profiled = engine.profile(params)
        assert profiled.value == pytest.approx(engine.expectation(params), abs=1e-12)
        assert profiled.profile.final_term_count == len(materialized.terms)


def test_custom_ptm_orientation_negative_entries_and_wire_order() -> None:
    one_qubit = np.diag([1.0, -1.0, 1.0, 1.0])
    tape = advanced.GateTape(2)
    tape.ptm((1,), one_qubit)
    result = advanced.PropagationEngine(
        tape, tcp.PauliOperator(2, [((0, 1), 1.0)])
    ).propagate_operator([])
    assert _terms(result) == {(0, 1): -1.0}

    two_qubit = np.zeros((16, 16), dtype=np.float64)
    two_qubit[4 * 1 + 2, 4 * 3 + 2] = -2.5
    tape = advanced.GateTape(2)
    tape.ptm((1, 0), two_qubit)
    result = advanced.PropagationEngine(
        tape, tcp.PauliOperator(2, [((2, 3), 1.0)])
    ).propagate_operator([])
    assert _terms(result) == {(2, 1): -2.5}


def test_random_unitary_derived_ptm_matches_local_reference() -> None:
    rng = np.random.default_rng(20260802)
    raw = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    unitary, _ = np.linalg.qr(raw)
    ptm = np.zeros((4, 4), dtype=np.float64)
    # Build the PTM directly from the independent public matrices; no native
    # transition compiler participates in this oracle.
    for output in range(4):
        for input_code in range(4):
            transformed = unitary.conj().T @ PAULI_MATRICES[input_code] @ unitary
            ptm[output, input_code] = float(
                np.trace(PAULI_MATRICES[output] @ transformed).real / 2.0
            )
    tape = advanced.GateTape(1)
    tape.ptm((0,), ptm)
    for input_code in range(4):
        result = advanced.PropagationEngine(
            tape, tcp.PauliOperator(1, [((input_code,), 1.0)])
        ).propagate_operator([])
        expected = {
            (output,): ptm[output, input_code]
            for output in range(4)
            if ptm[output, input_code] != 0.0
        }
        assert _terms(result) == pytest.approx(expected, abs=1e-12)


def test_invalid_tape_state_parameter_and_ptm_inputs_fail_explicitly() -> None:
    tape = advanced.GateTape(2)
    with pytest.raises(ValueError):
        tape.rx(0)
    with pytest.raises(ValueError):
        tape.rx(0, angle=0.1, parameter=0)
    with pytest.raises(ValueError):
        tape.cnot(0, 0)
    with pytest.raises(ValueError):
        tape.ptm((0,), np.eye(3, dtype=np.float64))
    with pytest.raises(TypeError):
        tape.ptm((0,), np.eye(4, dtype=np.complex128))
    tape.rx(0, parameter=1)
    with pytest.raises(ValueError, match="parameter slots"):
        advanced.PropagationEngine(tape, tcp.PauliOperator.empty(2))
    tape = advanced.GateTape(1)
    tape.x(0)
    with pytest.raises(ValueError):
        advanced.PropagationEngine(
            tape,
            tcp.PauliOperator(1, [((1,), 1j)]),
        ).expectation([])
    with pytest.raises(ValueError):
        advanced.PropagationEngine(
            tape,
            tcp.PauliOperator(1, [((1,), 1.0)]),
            initial_state=tcp.ProductBlochState(np.array([[2.0, 0.0, 0.0]])),
        )


def test_zero_qubit_identity_and_inline_boundary_structures() -> None:
    tape = advanced.GateTape(0)
    engine = advanced.PropagationEngine(tape, tcp.PauliOperator(0, [((), 1.0)]))
    assert engine.expectation([]) == 1.0
    assert _terms(engine.propagate_operator([])) == {(): 1.0}
    for nqubits in (64, 65, 100, 128, 129):
        tape = advanced.GateTape(nqubits)
        tape.x(nqubits - 1)
        codes = [0] * nqubits
        codes[0] = 1
        observable = tcp.PauliOperator(nqubits, [(codes, 1.0)])
        result = advanced.PropagationEngine(tape, observable).propagate_operator([])
        assert result.terms[0].word.to_codes() == tuple(codes)


def test_wide_key_payload_is_included_in_small_memory_guard() -> None:
    nqubits = 10_000
    with pytest.raises(MemoryError, match="memory limit"):
        advanced.PropagationEngine(
            advanced.GateTape(nqubits),
            tcp.PauliOperator(nqubits, [([1] + [0] * (nqubits - 1), 1.0)]),
            max_bytes=1_000,
        )


def test_engine_snapshots_tape_and_bloch_state_and_supports_concurrent_calls() -> None:
    tape = advanced.GateTape(1)
    tape.rz(0, parameter=0)
    bloch = np.array([[0.2, 0.3, 0.4]], dtype=np.float64)
    state = tcp.ProductBlochState(bloch)
    engine = advanced.PropagationEngine(
        tape, tcp.PauliOperator(1, [((1,), 1.0)]), initial_state=state
    )
    bloch[0, 0] = 0.9
    tape.x(0)
    expected = engine.expectation([0.25])
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: engine.expectation([0.25]), range(8)))
    assert values == pytest.approx([expected] * 8)
