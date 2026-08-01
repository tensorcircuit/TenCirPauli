use std::mem::size_of;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tencir_pauli_core::PauliWord;

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

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(50);
    targets = benchmark_weight, benchmark_commutation
}
criterion_main!(benches);
