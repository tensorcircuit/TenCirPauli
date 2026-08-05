"""Correctness checks for native-resident lazy Pauli algebra."""

from __future__ import annotations

import threading

import numpy as np

import tencirpauli as tcp


def as_terms(operator: tcp.PauliOperator) -> dict[str, complex]:
    """Return a readable canonical mapping for a small comparison."""
    return {term.word.to_string(): term.coefficient for term in operator.terms}


def test_lazy_bilinear_operations_materialize_like_reference() -> None:
    left = tcp.PauliOperator.from_terms(2, (("XX", 0.7), ("ZI", -0.2j), ("YY", 0.1)))
    right = tcp.PauliOperator.from_terms(2, (("IZ", -0.3), ("XY", 0.4j), ("XX", 0.2)))
    lazy = left.commutator(right).add(left.scale(0.5))
    reference = (
        left.dense() @ right.dense() - right.dense() @ left.dense() + 0.5 * left.dense()
    )

    assert lazy.nqubits == 2
    np.testing.assert_allclose(lazy.dense(), reference)


def test_lazy_scale_and_dense_result_match() -> None:
    operator = tcp.PauliOperator.from_terms(2, (("XY", 0.3), ("ZZ", -0.4j)))
    lazy = operator.scale(1.5 - 0.25j)
    expected = (1.5 - 0.25j) * operator.dense()
    np.testing.assert_allclose(lazy.dense(), expected)


def test_numeric_handle_readback_uses_flat_arrays() -> None:
    operator = tcp.PauliOperator.from_terms(3, (("XYZ", 0.5 - 0.25j),))
    assert operator._native_handle is not None
    term_count, width, codes, coefficients = (
        operator._native_handle.materialize_arrays()
    )
    assert (term_count, width) == (1, 3)
    assert np.asarray(codes).shape == (3,)
    assert np.asarray(coefficients).shape == (1,)
    assert np.asarray(codes).flags.c_contiguous
    assert np.asarray(coefficients).dtype == np.complex128


def test_scalable_native_work_releases_gil() -> None:
    nqubits = 12
    structures = np.asarray(
        [
            [(index // (3**qubit)) % 3 + 1 for qubit in range(nqubits)]
            for index in range(1024)
        ],
        dtype=np.uint8,
    )
    left = tcp.PauliOperator.from_code_arrays(structures, np.ones(1024))
    right = tcp.PauliOperator.from_code_arrays(structures, np.ones(1024))
    started = threading.Event()
    stop = threading.Event()
    progress = [0]

    def observer() -> None:
        started.set()
        while not stop.is_set():
            progress[0] += 1

    thread = threading.Thread(target=observer)
    thread.start()
    started.wait()
    try:
        result = left.multiply(right)
        assert result.term_count > 0
    finally:
        stop.set()
        thread.join()
    assert progress[0] > 0


def test_plain_export_is_lazy_and_avoids_term_objects() -> None:
    left = tcp.PauliOperator.from_terms(2, (("XX", 0.5), ("ZI", -0.25j)))
    right = tcp.PauliOperator.from_terms(2, (("IZ", 0.75),))
    lazy = left.commutator(right)
    assert type(lazy) is tcp.PauliOperator
    assert lazy.to_dict() == {"XY": -0.75j}

    assert isinstance(lazy, tcp.PauliOperator)
    assert lazy.term_count == 1
    assert lazy._terms is None
    assert lazy.to_dict() == {"XY": -0.75j}
    assert lazy._terms is None
    np.testing.assert_allclose(lazy.dense(), left.commutator(right).dense())
    assert lazy._terms is None

    assert tuple((term.word.to_string(), term.coefficient) for term in lazy.terms) == (
        ("XY", -0.75j),
    )
    assert lazy._terms is not None


def test_lazy_operator_rejects_incompatible_operands() -> None:
    left = tcp.PauliOperator.from_terms(1, (("X", 1.0),))
    right = tcp.PauliOperator.from_terms(2, (("XX", 1.0),))
    try:
        left.commutator(right)
    except ValueError as error:
        assert "incompatible qubit counts" in str(error)
    else:
        raise AssertionError("incompatible operators must fail")
