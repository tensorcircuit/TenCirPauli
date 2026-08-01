use super::{
    group_words_bounded, Complex64, GroupingAlgorithm, GroupingMode, MvpPlan, MvpStrategy,
    PauliError, PauliOperator, PauliPhase, PauliWord,
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
