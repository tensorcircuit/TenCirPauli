"""Independent big-int and sparse differential tests for the U1 engine."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

import tencirpauli as tcp


def _basis_values(nqubits: int, particle_number: int) -> list[int]:
    return sorted(
        [
            sum(1 << (nqubits - 1 - qubit) for qubit in occupied)
            for occupied in combinations(range(nqubits), particle_number)
        ]
    )


def _packed(value: int, nqubits: int) -> tuple[int, ...]:
    words = [0] * ((nqubits + 63) // 64)
    for qubit in range(nqubits):
        if (value >> (nqubits - 1 - qubit)) & 1:
            words[qubit // 64] |= 1 << (qubit % 64)
    return tuple(words)


def _local_pauli(nqubits: int, code: str, *qubits: int) -> str:
    result = ["I"] * nqubits
    for qubit in qubits:
        result[qubit] = code
    return "".join(result)


def _hopping_terms(nqubits: int) -> list[tuple[str, float]]:
    terms: list[tuple[str, float]] = []
    for qubit in range(nqubits - 1):
        terms.extend(
            (
                (_local_pauli(nqubits, "X", qubit, qubit + 1), 0.5),
                (_local_pauli(nqubits, "Y", qubit, qubit + 1), 0.5),
            )
        )
    return terms


def _reference_sparse(
    nqubits: int, particle_number: int, terms: list[tuple[str, complex]]
) -> dict[tuple[int, int], complex]:
    basis = _basis_values(nqubits, particle_number)
    lookup = {value: index for index, value in enumerate(basis)}
    result: dict[tuple[int, int], complex] = {}
    for source_index, source in enumerate(basis):
        aggregate: dict[int, complex] = {}
        for pauli, coefficient in terms:
            x_mask = 0
            z_mask = 0
            y_count = 0
            for qubit, code in enumerate(pauli):
                bit = 1 << (nqubits - 1 - qubit)
                if code in "XY":
                    x_mask |= bit
                if code in "YZ":
                    z_mask |= bit
                y_count += code == "Y"
            phase = (1j**y_count) * ((-1) ** bin(z_mask & source).count("1"))
            destination = source ^ x_mask
            aggregate[destination] = (
                aggregate.get(destination, 0j) + coefficient * phase
            )
        for destination, value in aggregate.items():
            if value == 0j:
                continue
            if bin(destination).count("1") != particle_number:
                raise AssertionError(
                    "reference operator leaks from the selected sector"
                )
            result[(lookup[destination], source_index)] = value
    return result


@pytest.mark.parametrize(
    ("nqubits", "particle_number"),
    [(63, 1), (64, 2), (65, 1), (127, 2), (128, 1), (129, 2), (256, 2)],
)
def test_phase5_basis_order_and_packed_padding(
    nqubits: int, particle_number: int
) -> None:
    sector = tcp.U1Sector(nqubits, particle_number)
    expected = _basis_values(nqubits, particle_number)
    assert sector.dimension == len(expected)
    assert sector.rank(expected[0]) == 0
    assert sector.rank(expected[-1]) == len(expected) - 1
    assert sector.unrank(0) == (
        expected[0]
        if nqubits <= 64
        else tuple(
            (expected[0] >> (nqubits - 1 - qubit)) & 1 for qubit in range(nqubits)
        )
    )
    basis = sector.basis_words()
    assert basis.flags.writeable is False
    if nqubits <= 64:
        np.testing.assert_array_equal(basis, np.asarray(expected, dtype=np.uint64))
    else:
        assert basis.shape == (len(expected), (nqubits + 63) // 64)
        np.testing.assert_array_equal(
            basis[: min(32, len(expected))],
            np.asarray(
                [_packed(value, nqubits) for value in expected[:32]], dtype=np.uint64
            ),
        )
        tail_bits = nqubits % 64
        if tail_bits:
            assert np.all(basis[:, -1] < (1 << tail_bits))


@pytest.mark.parametrize(
    ("nqubits", "particle_number"),
    [
        (63, 1),
        (64, 1),
        (65, 2),
        (65, 63),
        (127, 1),
        (128, 2),
        (129, 2),
        (256, 1),
    ],
)
def test_phase5_wide_restriction_matches_big_int_sparse_reference(
    nqubits: int, particle_number: int
) -> None:
    terms = _hopping_terms(nqubits)
    operator = tcp.PauliOperator.from_terms(nqubits, terms)
    restricted = operator.restrict_u1(tcp.U1Sector(nqubits, particle_number))
    reference = _reference_sparse(nqubits, particle_number, terms)
    csr = restricted.csr()
    actual = {
        (int(row), int(column)): complex(value)
        for row in range(csr.shape[0])
        for column, value in zip(
            csr.indices[csr.indptr[row] : csr.indptr[row + 1]],
            csr.data[csr.indptr[row] : csr.indptr[row + 1]],
        )
    }
    assert actual.keys() == reference.keys()
    np.testing.assert_allclose(
        [actual[key] for key in sorted(actual)],
        [reference[key] for key in sorted(reference)],
    )


def test_phase5_leakage_is_checked_after_x_group_aggregation() -> None:
    nqubits = 129
    sector = tcp.U1Sector(nqubits, 1)
    hopping = tcp.PauliOperator.from_terms(
        nqubits,
        (
            (_local_pauli(nqubits, "X", 63, 64), 0.5),
            (_local_pauli(nqubits, "Y", 63, 64), 0.5),
        ),
    )
    assert hopping.restrict_u1(sector).dimension == nqubits
    leaking = tcp.PauliOperator.from_terms(
        nqubits, ((_local_pauli(nqubits, "X", 64), 1.0),)
    )
    with pytest.raises(ValueError, match=r"U\(1\) sector leakage"):
        leaking.restrict_u1(sector)


def test_phase5_complex_directed_hopping_crosses_a_limb_boundary() -> None:
    nqubits = 65
    coefficient = 0.7 + 0.2j
    terms = [
        (_local_pauli(nqubits, "X", 63, 64), coefficient),
        (_local_pauli(nqubits, "Y", 63, 64), coefficient),
    ]
    operator = tcp.PauliOperator.from_terms(nqubits, terms)
    sector = tcp.U1Sector(nqubits, 1)
    restricted = operator.restrict_u1(sector)
    reference = _reference_sparse(nqubits, 1, terms)
    csr = restricted.csr()
    actual = {
        (int(row), int(column)): complex(value)
        for row in range(csr.shape[0])
        for column, value in zip(
            csr.indices[csr.indptr[row] : csr.indptr[row + 1]],
            csr.data[csr.indptr[row] : csr.indptr[row + 1]],
        )
    }
    assert actual.keys() == reference.keys()
    np.testing.assert_allclose(
        [actual[key] for key in sorted(actual)],
        [reference[key] for key in sorted(reference)],
    )
