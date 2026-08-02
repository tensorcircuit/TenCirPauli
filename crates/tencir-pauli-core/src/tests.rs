use super::{
    find_z2_symmetries, group_words_bounded, Complex64, GroupingAlgorithm, GroupingMode, MvpPlan,
    MvpStrategy, PauliError, PauliOperator, PauliPhase, PauliWord, U1RestrictedOperator, U1Sector,
    Z2TaperingPlan,
};

fn assert_apply_into_overwrites(
    plan: &MvpPlan,
    state: &[Complex64],
    expected_strategy: MvpStrategy,
    expected_parallel: bool,
) {
    let dimension = state.len();
    let work = plan.term_count().saturating_mul(dimension);
    assert_eq!(plan.strategy(), expected_strategy);
    assert_eq!(work >= 1 << 16, expected_parallel);

    let mut zeroed = vec![Complex64::default(); dimension];
    plan.apply_into(state, &mut zeroed, u128::MAX).unwrap();
    let mut prefilled = vec![Complex64::new(7.0, -11.0); dimension];
    plan.apply_into(state, &mut prefilled, u128::MAX).unwrap();
    assert_eq!(prefilled, zeroed);
}

fn mvp_test_operator(nqubits: usize, term_count: usize) -> PauliOperator {
    let structures = (0..term_count)
        .map(|mask| {
            (0..nqubits)
                .map(|qubit| u8::from(mask & (1 << qubit) != 0))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let coefficients = (0..term_count)
        .map(|index| Complex64::new(index as f64 + 1.0, -(index as f64) * 0.25))
        .collect::<Vec<_>>();
    PauliOperator::from_terms(nqubits, &structures, &coefficients).unwrap()
}

#[test]
fn computes_weight_and_canonicalizes_unused_bits() {
    let word = PauliWord::from_words(2, vec![u64::MAX], vec![0]).unwrap();
    assert_eq!(word.x_words(), &[0b11]);
    assert_eq!(word.weight(), 2);
}

#[test]
fn checks_symplectic_commutation() {
    let x0 = PauliWord::from_words(2, vec![0b01], vec![0]).unwrap();
    let z0 = PauliWord::from_words(2, vec![0], vec![0b01]).unwrap();
    let xx = PauliWord::from_words(2, vec![0b11], vec![0]).unwrap();
    let zz = PauliWord::from_words(2, vec![0], vec![0b11]).unwrap();
    assert!(!x0.commutes_with(&z0).unwrap());
    assert!(xx.commutes_with(&zz).unwrap());
}

#[test]
fn covers_phase_table_and_round_trip() {
    let expected = [
        (1, 2, 3, PauliPhase::PlusI),
        (2, 1, 3, PauliPhase::MinusI),
        (1, 3, 2, PauliPhase::MinusI),
        (3, 1, 2, PauliPhase::PlusI),
        (2, 3, 1, PauliPhase::PlusI),
        (3, 2, 1, PauliPhase::MinusI),
    ];
    for (left, right, result, phase) in expected {
        let left_word = PauliWord::from_codes(1, &[left]).unwrap();
        let right_word = PauliWord::from_codes(1, &[right]).unwrap();
        let (actual, actual_phase) = left_word.multiply(&right_word).unwrap();
        assert_eq!(actual.codes(), vec![result]);
        assert_eq!(actual_phase, phase);
    }
    let word = PauliWord::from_codes(130, &[2; 130]).unwrap();
    assert_eq!(word.codes(), vec![2; 130]);
}

#[test]
fn canonical_operator_algebra_aggregates_exact_zeros() {
    let operator = PauliOperator::from_terms(
        1,
        &[vec![1], vec![1], vec![2]],
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(2.0, 0.0),
        ],
    )
    .unwrap();
    assert_eq!(operator.terms().len(), 1);
    assert!(operator.is_hermitian(0.0));
    let reversed = PauliOperator::from_terms(
        1,
        &[vec![2], vec![1], vec![1]],
        &[
            Complex64::new(2.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    )
    .unwrap();
    let reordered = PauliOperator::from_terms(
        1,
        &[vec![1], vec![2], vec![1]],
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    )
    .unwrap();
    assert_eq!(reversed, reordered);
}

#[test]
fn mvp_apply_into_overwrites_zeroed_and_prefilled_outputs_for_all_paths() {
    let serial_operator = mvp_test_operator(1, 1);
    let serial_state = [Complex64::new(3.0, -2.0), Complex64::new(5.0, 4.0)];
    let serial_direct = MvpPlan::from_operator(&serial_operator).unwrap();
    assert_apply_into_overwrites(
        &serial_direct,
        &serial_state,
        MvpStrategy::TermDirect,
        false,
    );
    let serial_reusable = MvpPlan::from_operator_reusable(&serial_operator, u128::MAX).unwrap();
    assert_apply_into_overwrites(
        &serial_reusable,
        &serial_state,
        MvpStrategy::XMaskDiagonal,
        false,
    );

    let parallel_operator = mvp_test_operator(10, 64);
    let parallel_state = (0..1 << 10)
        .map(|index| Complex64::new(index as f64 * 0.5, 1.0 - index as f64 * 0.125))
        .collect::<Vec<_>>();
    let parallel_direct = MvpPlan::from_operator(&parallel_operator).unwrap();
    assert_apply_into_overwrites(
        &parallel_direct,
        &parallel_state,
        MvpStrategy::TermDirect,
        true,
    );
    let parallel_reusable = MvpPlan::from_operator_reusable(&parallel_operator, u128::MAX).unwrap();
    assert_apply_into_overwrites(
        &parallel_reusable,
        &parallel_state,
        MvpStrategy::XMaskDiagonal,
        true,
    );
}

#[test]
fn scale_preserves_finite_nonzero_canonical_terms() {
    let operator = PauliOperator::from_terms(
        1,
        &[vec![1], vec![2]],
        &[
            Complex64::new(2.0, -3.0),
            Complex64::new(f64::from_bits(1), 0.0),
        ],
    )
    .unwrap();

    let zero = operator.scale(Complex64::new(0.0, 0.0)).unwrap();
    assert!(zero.terms().is_empty());

    let underflowed = operator.scale(Complex64::new(0.5, 0.0)).unwrap();
    assert_eq!(underflowed.terms().len(), 1);
    assert_eq!(
        underflowed.terms()[0].coefficient,
        Complex64::new(1.0, -1.5)
    );

    let complex = PauliOperator::from_terms(1, &[vec![1]], &[Complex64::new(2.0, -3.0)])
        .unwrap()
        .scale(Complex64::new(4.0, 5.0))
        .unwrap();
    assert_eq!(complex.terms()[0].coefficient, Complex64::new(23.0, -2.0));

    let overflow = PauliOperator::from_terms(1, &[vec![1]], &[Complex64::new(2.0, 0.0)])
        .unwrap()
        .scale(Complex64::new(f64::MAX, 0.0));
    assert_eq!(overflow, Err(PauliError::NonFiniteCoefficient { index: 0 }));
}

#[test]
fn canonical_fast_path_validates_its_contract() {
    let structures = vec![vec![0, 0], vec![3, 1]];
    let coefficients = vec![Complex64::new(2.0, 0.0), Complex64::new(1.0, -0.25)];
    let fast = PauliOperator::from_canonical_terms(2, &structures, &coefficients).unwrap();
    let reduced = PauliOperator::from_terms(2, &structures, &coefficients).unwrap();
    assert_eq!(fast, reduced);

    let duplicate = PauliOperator::from_canonical_terms(
        2,
        &[vec![0, 0], vec![0, 0]],
        &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
    );
    assert_eq!(duplicate, Err(PauliError::NonCanonicalTerms { index: 1 }));
    let zero = PauliOperator::from_canonical_terms(2, &[vec![0, 0]], &[Complex64::new(0.0, 0.0)]);
    assert_eq!(zero, Err(PauliError::NonCanonicalTerms { index: 0 }));

    let overflow = PauliOperator::from_terms(
        1,
        &[vec![1], vec![1]],
        &[Complex64::new(f64::MAX, 0.0), Complex64::new(f64::MAX, 0.0)],
    );
    assert!(matches!(
        overflow,
        Err(PauliError::NonFiniteCoefficient { .. })
    ));
}

#[test]
fn bounded_grouping_rejects_the_dense_matrix_before_allocation() {
    let words = (0..4)
        .map(|index| PauliWord::from_codes(1, &[u8::try_from(index).unwrap()]).unwrap())
        .collect::<Vec<_>>();
    let result = group_words_bounded(
        &words,
        GroupingMode::QubitWise,
        GroupingAlgorithm::LargestFirst,
        15,
    );
    assert_eq!(
        result,
        Err(PauliError::MemoryLimit {
            requested: 16,
            limit: 15,
        })
    );
}

#[test]
fn z2_analysis_and_tapering_are_deterministic() {
    let operator = PauliOperator::from_terms(
        2,
        &[vec![1, 1], vec![3, 3], vec![0, 0]],
        &[
            Complex64::new(0.7, 0.0),
            Complex64::new(1.2, 0.0),
            Complex64::new(0.3, 0.0),
        ],
    )
    .unwrap();
    let analysis = find_z2_symmetries(&operator, 1 << 20).unwrap();
    assert_eq!(
        analysis
            .generators
            .iter()
            .map(|word| word.codes())
            .collect::<Vec<_>>(),
        vec![vec![1, 1], vec![3, 3]]
    );
    let plan = Z2TaperingPlan::new(2, &analysis.generators, &[1, 1]).unwrap();
    assert_eq!(plan.nqubits_after(), 0);
    let tapered = plan.transform_operator(&operator).unwrap();
    assert_eq!(tapered.nqubits(), 0);
    assert_eq!(tapered.terms().len(), 1);
    assert_eq!(tapered.terms()[0].coefficient, Complex64::new(2.2, 0.0));
}

fn binary_rank(words: &[PauliWord], nqubits: usize) -> usize {
    let mut rows = words
        .iter()
        .map(|word| {
            let mut row = vec![0_u64; (2 * nqubits).div_ceil(64)];
            for (qubit, code) in word.codes().into_iter().enumerate() {
                if matches!(code, 1 | 2) {
                    row[qubit / 64] ^= 1_u64 << (qubit % 64);
                }
                if matches!(code, 2 | 3) {
                    let index = nqubits + qubit;
                    row[index / 64] ^= 1_u64 << (index % 64);
                }
            }
            row
        })
        .collect::<Vec<_>>();
    let mut rank = 0;
    for column in 0..2 * nqubits {
        let Some(pivot) =
            (rank..rows.len()).find(|&row| rows[row][column / 64] & (1_u64 << (column % 64)) != 0)
        else {
            continue;
        };
        rows.swap(rank, pivot);
        let (prefix, suffix) = rows.split_at_mut(rank + 1);
        let pivot_row = &prefix[rank];
        for row in suffix {
            if row[column / 64] & (1_u64 << (column % 64)) != 0 {
                for (value, pivot_value) in row.iter_mut().zip(pivot_row) {
                    *value ^= pivot_value;
                }
            }
        }
        rank += 1;
    }
    rank
}

fn visit_commuting_generator_sets(
    words: &[PauliWord],
    nqubits: usize,
    start: usize,
    selected: &mut Vec<PauliWord>,
) {
    if !selected.is_empty() {
        let coefficients = (0..selected.len())
            .map(|index| Complex64::new((3_usize.pow(index as u32)) as f64, 0.0))
            .collect::<Vec<_>>();
        let structures = selected.iter().map(PauliWord::codes).collect::<Vec<_>>();
        let operator = PauliOperator::from_terms(nqubits, &structures, &coefficients).unwrap();
        for sector_bits in 0..(1_usize << selected.len()) {
            let sector = (0..selected.len())
                .map(|index| {
                    if sector_bits & (1 << index) == 0 {
                        1_i8
                    } else {
                        -1_i8
                    }
                })
                .collect::<Vec<_>>();
            let plan = Z2TaperingPlan::new(nqubits, selected, &sector).unwrap();
            let tapered = plan.transform_operator(&operator).unwrap();
            let expected = coefficients
                .iter()
                .zip(&sector)
                .map(|(coefficient, sign)| *coefficient * f64::from(*sign))
                .sum::<Complex64>();
            assert_eq!(tapered.terms().len(), 1);
            assert_eq!(tapered.terms()[0].coefficient, expected);
        }
    }
    if selected.len() == nqubits {
        return;
    }
    for index in start..words.len() {
        let candidate = &words[index];
        if selected
            .iter()
            .all(|word| word.commutes_with(candidate).unwrap())
        {
            selected.push(candidate.clone());
            if binary_rank(selected, nqubits) == selected.len() {
                visit_commuting_generator_sets(words, nqubits, index + 1, selected);
            }
            selected.pop();
        }
    }
}

#[test]
fn z2_exhaustive_small_commuting_sets_preserve_every_sector_sign() {
    for nqubits in 1..=3 {
        let words = (1..4_usize.pow(nqubits as u32))
            .map(|mut value| {
                let mut codes = vec![0_u8; nqubits];
                for code in &mut codes {
                    *code = (value % 4) as u8;
                    value /= 4;
                }
                PauliWord::from_codes(nqubits, &codes).unwrap()
            })
            .collect::<Vec<_>>();
        visit_commuting_generator_sets(&words, nqubits, 0, &mut Vec::new());
    }
}

#[test]
fn u1_basis_and_restricted_hopping_use_aggregated_transitions() {
    let sector = U1Sector::new(3, 1).unwrap();
    assert_eq!(sector.dimension().unwrap(), 3);
    assert_eq!(sector.basis_words(u128::MAX).unwrap(), vec![1, 2, 4]);
    assert_eq!(sector.rank(2).unwrap(), 1);
    assert_eq!(sector.unrank(2).unwrap(), 4);

    let operator = PauliOperator::from_terms(
        3,
        &[vec![1, 1, 0], vec![2, 2, 0]],
        &[Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
    )
    .unwrap();
    let restricted = U1RestrictedOperator::new(&operator, sector, 1 << 20).unwrap();
    let (dimension, dense) = restricted.dense(u128::MAX).unwrap();
    assert_eq!(dimension, 3);
    assert_eq!(dense[5], Complex64::new(2.0, 0.0));
    assert_eq!(dense[7], Complex64::new(2.0, 0.0));
    let coo = restricted.coo(u128::MAX).unwrap();
    assert_eq!(coo.rows, vec![1, 2]);
    assert_eq!(coo.columns, vec![2, 1]);
    assert_eq!(coo.values, vec![Complex64::new(2.0, 0.0); 2]);
    assert_eq!(
        restricted
            .apply(
                &[
                    Complex64::new(3.0, 0.0),
                    Complex64::new(5.0, 0.0),
                    Complex64::new(7.0, 0.0)
                ],
                u128::MAX
            )
            .unwrap(),
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(14.0, 0.0),
            Complex64::new(10.0, 0.0)
        ]
    );
}

#[test]
fn u1_packed_rank_and_unrank_are_inverse_for_particle_and_hole_paths() {
    for nqubits in 0..=8 {
        for particle_number in 0..=nqubits {
            let sector = U1Sector::new(nqubits, particle_number).unwrap();
            let mut words = vec![0_u64; sector.word_count()];
            for index in 0..sector.dimension().unwrap() {
                sector.unrank_into(index as u64, &mut words).unwrap();
                assert_eq!(
                    words
                        .iter()
                        .map(|word| word.count_ones() as usize)
                        .sum::<usize>(),
                    particle_number
                );
                assert_eq!(sector.rank_words(&words).unwrap(), index as u64);
            }
        }
    }
}

#[test]
fn u1_packed_rank_crosses_three_limb_boundaries() {
    let sector = U1Sector::new(129, 2).unwrap();
    let mut words = vec![0_u64; sector.word_count()];
    for index in [0_u64, 1, 63, 64, 127, 128, 8255] {
        sector.unrank_into(index, &mut words).unwrap();
        assert_eq!(sector.rank_words(&words).unwrap(), index);
        assert_eq!(words[2] & !1_u64, 0);
    }
}

#[test]
fn u1_apply_into_reports_the_failing_buffer_length() {
    let sector = U1Sector::new(3, 1).unwrap();
    let operator = PauliOperator::from_terms(
        3,
        &[vec![1, 1, 0], vec![2, 2, 0]],
        &[Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
    )
    .unwrap();
    let plan = U1RestrictedOperator::new(&operator, sector, u128::MAX)
        .unwrap()
        .mvp_plan(u128::MAX)
        .unwrap();
    let state = vec![Complex64::default(); 2];
    let mut output = vec![Complex64::default(); 3];
    assert_eq!(
        plan.apply_into(&state, &mut output),
        Err(PauliError::InvalidStructureLength {
            expected: 3,
            actual: 2,
        })
    );
    let state = vec![Complex64::default(); 3];
    let mut output = vec![Complex64::default(); 2];
    assert_eq!(
        plan.apply_into(&state, &mut output),
        Err(PauliError::InvalidStructureLength {
            expected: 3,
            actual: 2,
        })
    );
}
