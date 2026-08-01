use std::mem::size_of;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tencir_pauli_core::{group_words, GroupingAlgorithm, GroupingMode, PauliOperator, PauliWord};

const SIZES: [usize; 3] = [64, 1_024, 16_384];

fn make_word(nqubits: usize, x_pattern: u64, z_pattern: u64) -> PauliWord {
    let nwords = nqubits.div_ceil(64);
    PauliWord::from_words(nqubits, vec![x_pattern; nwords], vec![z_pattern; nwords])
        .expect("benchmark words have valid dimensions")
}

fn benchmark_weight(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("pauli_word/weight");
    for nqubits in SIZES {
        let word = make_word(nqubits, 0xAAAA_AAAA_AAAA_AAAA, 0x1111_1111_1111_1111);
        let bytes = 2 * word.x_words().len() * size_of::<u64>();
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(nqubits),
            &word,
            |bencher, input| bencher.iter(|| black_box(input).weight()),
        );
    }
    group.finish();
}

fn benchmark_commutation(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("pauli_word/commutes_with");
    for nqubits in SIZES {
        let left = make_word(nqubits, 0xAAAA_AAAA_AAAA_AAAA, 0x1111_1111_1111_1111);
        let right = make_word(nqubits, 0xCCCC_CCCC_CCCC_CCCC, 0x0101_0101_0101_0101);
        left.commutes_with(&right)
            .expect("benchmark words have equal qubit counts");

        let bytes = 4 * left.x_words().len() * size_of::<u64>();
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(nqubits),
            &(&left, &right),
            |bencher, (left_input, right_input)| {
                bencher.iter(|| {
                    black_box(left_input)
                        .commutes_with(black_box(right_input))
                        .expect("benchmark words have equal qubit counts")
                });
            },
        );
    }
    group.finish();
}

fn benchmark_conversion_and_multiplication(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("pauli_word/algebra");
    for nqubits in [6_usize, 64, 256] {
        let left_codes = (0..nqubits)
            .map(|index| (index % 4) as u8)
            .collect::<Vec<_>>();
        let right_codes = (0..nqubits)
            .map(|index| ((index + 1) % 4) as u8)
            .collect::<Vec<_>>();
        group.bench_with_input(
            BenchmarkId::new("codes_round_trip", nqubits),
            &left_codes,
            |bencher, codes| {
                bencher.iter(|| {
                    let word = PauliWord::from_codes(nqubits, black_box(codes)).unwrap();
                    black_box(word.codes())
                });
            },
        );
        let left = PauliWord::from_codes(nqubits, &left_codes).unwrap();
        let right = PauliWord::from_codes(nqubits, &right_codes).unwrap();
        group.bench_with_input(
            BenchmarkId::new("multiply", nqubits),
            &(&left, &right),
            |bencher, (left_input, right_input)| {
                bencher.iter(|| {
                    black_box(left_input)
                        .multiply(black_box(right_input))
                        .unwrap()
                });
            },
        );
    }
    group.finish();
}

fn benchmark_canonicalization(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("pauli_operator/canonicalize");
    for count in [1_000_usize, 10_000, 100_000] {
        let structures = (0..count)
            .map(|index| {
                (0..8)
                    .map(|qubit| ((index + qubit) % 4) as u8)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let coefficients = (0..count)
            .map(|index| {
                tencir_pauli_core::Complex64::new(
                    (index % 7) as f64 - 3.0,
                    (index % 5) as f64 - 2.0,
                )
            })
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &(&structures, &coefficients),
            |bencher, (structures_input, coefficients_input)| {
                bencher.iter(|| {
                    black_box(PauliOperator::from_terms(
                        8,
                        black_box(structures_input),
                        black_box(coefficients_input),
                    ))
                });
            },
        );
    }
    group.finish();
}

fn benchmark_grouping(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("grouping/qwc");
    for count in [128_usize, 1_024] {
        let words = (0..count)
            .map(|index| {
                PauliWord::from_codes(
                    8,
                    &(0..8)
                        .map(|qubit| ((index / (qubit + 1)) % 4) as u8)
                        .collect::<Vec<_>>(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &words,
            |bencher, words_input| {
                bencher.iter(|| {
                    black_box(group_words(
                        black_box(words_input),
                        GroupingMode::QubitWise,
                        GroupingAlgorithm::LargestFirst,
                    ))
                });
            },
        );
    }
    group.finish();
}

fn benchmark_hamiltonian(criterion: &mut Criterion) {
    let structures = (0..32)
        .map(|index| {
            (0..8)
                .map(|qubit| ((index + qubit) % 4) as u8)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let coefficients = (0..32)
        .map(|index| tencir_pauli_core::Complex64::new(index as f64 + 1.0, 0.0))
        .collect::<Vec<_>>();
    let operator = PauliOperator::from_terms(8, &structures, &coefficients).unwrap();
    let state = (0..256)
        .map(|index| tencir_pauli_core::Complex64::new(index as f64, 0.0))
        .collect::<Vec<_>>();
    let mut group = criterion.benchmark_group("hamiltonian/targets");
    group.bench_function("dense", |bencher| {
        bencher.iter(|| black_box(operator.dense_matrix(u128::MAX).unwrap()))
    });
    group.bench_function("coo", |bencher| {
        bencher.iter(|| black_box(operator.coo_matrix(u128::MAX).unwrap()))
    });
    group.bench_function("mvp", |bencher| {
        bencher.iter(|| black_box(operator.mvp(black_box(&state), u128::MAX).unwrap()))
    });
    group.bench_function("backend_plan", |bencher| {
        bencher.iter(|| black_box(operator.backend_mvp_plan(u128::MAX).unwrap()))
    });
    group.finish();
}

fn benchmark_hamiltonian_scaling(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("hamiltonian/scaling");
    for (nqubits, count) in [(10_usize, 64_usize), (16, 256)] {
        let structures = (0..count)
            .map(|index| {
                (0..nqubits)
                    .map(|qubit| ((index / 4_usize.pow(qubit as u32)) % 4) as u8)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let coefficients = (0..count)
            .map(|index| tencir_pauli_core::Complex64::new(1.0 + index as f64 / 100.0, 0.0))
            .collect::<Vec<_>>();
        let operator = PauliOperator::from_terms(nqubits, &structures, &coefficients).unwrap();
        let plan = operator.mvp_plan(u128::MAX).unwrap();
        let state = (0..(1_usize << nqubits))
            .map(|index| {
                tencir_pauli_core::Complex64::new((index as f64).sin(), (index as f64).cos())
            })
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements((count * (1_usize << nqubits)) as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "reusable_plan_construction",
                format!("{nqubits}q_{count}terms"),
            ),
            &operator,
            |bencher, operator_input| {
                bencher.iter(|| black_box(operator_input.mvp_plan(u128::MAX).unwrap()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("operator_mvp", format!("{nqubits}q_{count}terms")),
            &(&operator, &state),
            |bencher, (operator_input, state_input)| {
                bencher.iter(|| {
                    black_box(
                        operator_input
                            .mvp(black_box(state_input), u128::MAX)
                            .unwrap(),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("reusable_plan_apply", format!("{nqubits}q_{count}terms")),
            &(&plan, &state),
            |bencher, (plan_input, state_input)| {
                bencher.iter(|| {
                    black_box(plan_input.apply(black_box(state_input), u128::MAX).unwrap())
                });
            },
        );
        if nqubits <= 12 {
            group.bench_with_input(
                BenchmarkId::new("coo", format!("{nqubits}q_{count}terms")),
                &operator,
                |bencher, operator_input| {
                    bencher.iter(|| black_box(operator_input.coo_matrix(u128::MAX).unwrap()));
                },
            );
            group.bench_with_input(
                BenchmarkId::new("csr", format!("{nqubits}q_{count}terms")),
                &operator,
                |bencher, operator_input| {
                    bencher.iter(|| black_box(operator_input.csr_matrix(u128::MAX).unwrap()));
                },
            );
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(50);
    targets = benchmark_weight, benchmark_commutation, benchmark_conversion_and_multiplication,
        benchmark_canonicalization, benchmark_grouping, benchmark_hamiltonian,
        benchmark_hamiltonian_scaling
}
criterion_main!(benches);
