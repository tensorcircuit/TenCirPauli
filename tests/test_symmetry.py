"""symmetry symmetry and fixed-sector differential tests."""

import numpy as np
import pytest

import tencirpauli as tcp


def _csr_dense(matrix: tcp.CSRMatrix) -> np.ndarray:
    result = np.zeros(matrix.shape, dtype=np.complex128)
    for row in range(matrix.shape[0]):
        start, stop = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
        result[row, matrix.indices[start:stop]] = matrix.data[start:stop]
    return result


def _coo_dense(matrix: tcp.COOMatrix) -> np.ndarray:
    result = np.zeros(matrix.shape, dtype=np.complex128)
    result[matrix.row, matrix.column] = matrix.data
    return result


def test_z2_analysis_and_tapering_preserve_all_sectors() -> None:
    operator = tcp.PauliOperator.from_terms(2, (("XX", 0.7), ("ZZ", 1.2), ("II", 0.3)))
    analysis = operator.find_z2_symmetries()

    assert [word.to_string() for word in analysis.generators] == ["XX", "ZZ"]
    assert analysis.rank == 2
    assert analysis.constraint_rank == 2
    assert all(
        left.commutes_with(right)
        for index, left in enumerate(analysis.generators)
        for right in analysis.generators[:index]
    )

    expected = np.linalg.eigvalsh(operator.dense())
    sector_values = []
    for sector in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        plan = analysis.tapering_plan(sector)
        assert plan.nqubits_after == 0
        assert plan.clifford_operations.flags.writeable is False
        sector_values.extend(
            np.linalg.eigvalsh(plan.transform_operator(operator).dense())
        )
    np.testing.assert_allclose(sorted(sector_values), expected)


def test_z2_tapering_rejects_incompatible_observable() -> None:
    operator = tcp.PauliOperator.from_terms(2, (("XX", 1.0), ("ZZ", 1.0)))
    plan = operator.find_z2_symmetries().tapering_plan((1, 1))
    incompatible = tcp.PauliOperator.from_terms(2, (("XI", 1.0),))
    with pytest.raises(ValueError, match="commute"):
        plan.transform_operator(incompatible)


def test_z2_tapering_preserves_multigenerator_sector_signs() -> None:
    """Compare every tapered sector with an independent dense projector."""
    operator = tcp.PauliOperator.from_terms(3, (("ZYY", 1.0), ("YIZ", 2.0)))
    analysis = operator.find_z2_symmetries()
    assert [word.to_string() for word in analysis.generators] == ["IXZ", "XYX", "YXI"]

    identity = np.eye(2**operator.nqubits, dtype=np.complex128)
    generator_matrices = [
        tcp.PauliOperator.from_terms(
            operator.nqubits, ((word.to_string(), 1.0),)
        ).dense()
        for word in analysis.generators
    ]
    full_matrix = operator.dense()
    for sector in np.ndindex(*(2,) * analysis.rank):
        selected = tuple(1 if value == 0 else -1 for value in sector)
        projector = identity.copy()
        for sign, generator_matrix in zip(selected, generator_matrices):
            projector = projector @ (identity + sign * generator_matrix) / 2
        expected = np.trace(projector @ full_matrix) / np.trace(projector)
        tapered = analysis.tapering_plan(selected).transform_operator(operator).dense()
        np.testing.assert_allclose(tapered, np.asarray([[expected]]))

        for sign, word in zip(selected, analysis.generators):
            transformed = (
                analysis.tapering_plan(selected)
                .transform_operator(
                    tcp.PauliOperator.from_terms(
                        operator.nqubits, ((word.to_string(), 1.0),)
                    )
                )
                .dense()
            )
            np.testing.assert_allclose(transformed, np.asarray([[sign]]))


def _symplectic_rank(words: list[tcp.PauliWord]) -> int:
    basis: dict[int, int] = {}
    nqubits = words[0].nqubits
    for word in words:
        value = 0
        for qubit, code in enumerate(word.to_codes()):
            if code in (1, 2):
                value |= 1 << qubit
            if code in (2, 3):
                value |= 1 << (nqubits + qubit)
        while value:
            pivot = value.bit_length() - 1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    return len(basis)


def test_z2_random_projector_reference_covers_y_generators() -> None:
    """Check independent dense sector actions for random n=4 stabilizers."""
    rng = np.random.default_rng(20260802)
    all_words = [
        tcp.PauliWord.from_codes(codes)
        for codes in np.ndindex(*(4,) * 4)
        if any(code != 0 for code in codes)
    ]
    for _ in range(12):
        selected: list[tcp.PauliWord] = []
        for word in rng.permutation(all_words):
            if all(word.commutes_with(previous) for previous in selected):
                candidate = [*selected, word]
                if _symplectic_rank(candidate) == len(candidate):
                    selected.append(word)
            if len(selected) == 2:
                break
        assert len(selected) == 2
        nqubits = 4
        identity = np.eye(2**nqubits, dtype=np.complex128)
        generator_matrices = [
            tcp.PauliOperator.from_terms(nqubits, ((word.to_string(), 1.0),)).dense()
            for word in selected
        ]
        terms: list[tuple[str, complex]] = [("IIII", 0.3 - 0.2j)]
        terms.extend(
            (word.to_string(), complex(rng.normal(), rng.normal())) for word in selected
        )
        product = selected[0].multiply(selected[1]).word
        terms.append((product.to_string(), complex(rng.normal(), rng.normal())))
        operator = tcp.PauliOperator.from_terms(nqubits, terms)
        analysis = tcp.Z2SymmetryAnalysis(nqubits, tuple(selected), len(selected))
        for sector in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            projector = identity.copy()
            for sign, generator_matrix in zip(sector, generator_matrices):
                projector = projector @ (identity + sign * generator_matrix) / 2
            eigenvalues, eigenvectors = np.linalg.eigh(projector)
            basis = eigenvectors[:, eigenvalues > 0.5]
            expected = basis.conj().T @ operator.dense() @ basis
            tapered = (
                analysis.tapering_plan(sector).transform_operator(operator).dense()
            )
            np.testing.assert_allclose(
                np.linalg.eigvalsh(tapered), np.linalg.eigvalsh(expected)
            )


@pytest.mark.parametrize(
    ("nqubits", "particle_number", "expected"),
    [
        (0, 0, [0]),
        (3, 0, [0]),
        (3, 1, [1, 2, 4]),
        (3, 2, [3, 5, 6]),
        (3, 3, [7]),
    ],
)
def test_u1_basis_order_rank_and_unrank(
    nqubits: int, particle_number: int, expected: list[int]
) -> None:
    sector = tcp.U1Sector(nqubits, particle_number)
    assert sector.dimension == len(expected)
    expected_bits = [
        tuple((value >> (nqubits - 1 - qubit)) & 1 for qubit in range(nqubits))
        for value in expected
    ]
    assert [sector.unrank(index) for index in range(sector.dimension)] == expected_bits
    assert [sector.rank(value) for value in expected] == list(range(len(expected)))
    basis = sector.basis_states()
    np.testing.assert_array_equal(basis, np.asarray(expected_bits, dtype=np.uint8))
    assert basis.flags.writeable is False


def test_u1_restricted_operator_matches_full_projection_and_aggregates_hopping() -> (
    None
):
    operator = tcp.PauliOperator.from_terms(
        3,
        (("XXI", 0.5), ("YYI", 0.5), ("IZZ", -0.25), ("ZII", 0.1)),
    )
    sector = tcp.U1Sector(3, 1)
    restricted = operator.restrict_u1(sector)
    basis = np.asarray(
        [
            sum(bit << (2 - qubit) for qubit, bit in enumerate(sector.unrank(index)))
            for index in range(sector.dimension)
        ]
    )
    expected = operator.dense()[np.ix_(basis, basis)]
    np.testing.assert_allclose(restricted.dense(), expected)
    np.testing.assert_allclose(_coo_dense(restricted.coo()), expected)
    np.testing.assert_allclose(_csr_dense(restricted.csr()), expected)
    state = np.asarray([1.0, 2.0, 3.0], dtype=np.complex128)
    np.testing.assert_allclose(restricted.apply(state), expected @ state)
    np.testing.assert_allclose(restricted.mvp_plan().apply(state), expected @ state)


def test_u1_restriction_rejects_leakage_after_aggregation() -> None:
    sector = tcp.U1Sector(3, 1)
    hopping = tcp.PauliOperator.from_terms(3, (("XXI", 1.0), ("YYI", 1.0)))
    assert hopping.restrict_u1(sector).dimension == 3
    with pytest.raises(ValueError, match="leakage"):
        tcp.PauliOperator.from_terms(3, (("XII", 1.0),)).restrict_u1(sector)


def test_u1_complex_y_properties_match_independent_dense_projection() -> None:
    """Exercise aggregated Y phases and complex coefficients on small sectors."""
    rng = np.random.default_rng(20260802)
    for nqubits in range(1, 5):
        for _ in range(10):
            particle_number = int(rng.integers(0, nqubits + 1))
            terms = [("I" * nqubits, complex(rng.normal(), rng.normal()))]
            for qubit in range(nqubits - 1):
                prefix = "I" * qubit
                suffix = "I" * (nqubits - qubit - 2)
                hopping = complex(rng.normal(), rng.normal())
                terms.extend(
                    (
                        (prefix + "XX" + suffix, hopping),
                        (prefix + "YY" + suffix, hopping),
                    )
                )
            for qubit in range(nqubits):
                terms.append(
                    (
                        "I" * qubit + "Z" + "I" * (nqubits - qubit - 1),
                        complex(rng.normal(), rng.normal()),
                    )
                )
            operator = tcp.PauliOperator.from_terms(nqubits, terms)
            sector = tcp.U1Sector(nqubits, particle_number)
            restricted = operator.restrict_u1(sector)
            basis = np.asarray(
                [
                    sum(
                        bit << (nqubits - 1 - qubit)
                        for qubit, bit in enumerate(sector.unrank(index))
                    )
                    for index in range(sector.dimension)
                ]
            )
            expected = operator.dense()[np.ix_(basis, basis)]
            np.testing.assert_allclose(restricted.dense(), expected)
            state = rng.normal(size=sector.dimension) + 1j * rng.normal(
                size=sector.dimension
            )
            np.testing.assert_allclose(restricted.apply(state), expected @ state)


def test_u1_native_restriction_supports_single_word_width_boundary() -> None:
    nqubits = np.dtype(np.uintp).itemsize * 8
    operator = tcp.PauliOperator.from_terms(nqubits, (("I" * nqubits, 1.0),))
    restricted = operator.restrict_u1(tcp.U1Sector(nqubits, 0))
    np.testing.assert_allclose(restricted.apply(np.asarray([2.0 + 3.0j])), [2.0 + 3.0j])


def test_symmetry_invalid_inputs_fail_explicitly() -> None:
    with pytest.raises(ValueError):
        tcp.U1Sector(2, 3)
    with pytest.raises((ValueError, IndexError)):
        tcp.U1Sector(2, 1).unrank(2)
    with pytest.raises(ValueError, match="sector values"):
        tcp.PauliOperator.from_terms(
            2, (("XX", 1.0),)
        ).find_z2_symmetries().tapering_plan((0,))
