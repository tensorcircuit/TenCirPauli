use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tencir_pauli_core::{
    find_z2_symmetries, Complex64, PauliOperator, U1RestrictedOperator, U1Sector,
};

fn make_hopping(nqubits: usize) -> PauliOperator {
    let mut structures = Vec::with_capacity(2 * nqubits.saturating_sub(1));
    let mut coefficients = Vec::with_capacity(structures.capacity());
    for index in 0..nqubits.saturating_sub(1) {
        let mut xx = vec![0_u8; nqubits];
        let mut yy = vec![0_u8; nqubits];
        xx[index] = 1;
        xx[index + 1] = 1;
        yy[index] = 2;
        yy[index + 1] = 2;
        structures.push(xx);
        structures.push(yy);
        coefficients.push(Complex64::new(0.5, 0.0));
        coefficients.push(Complex64::new(0.5, 0.0));
    }
    PauliOperator::from_terms(nqubits, &structures, &coefficients).unwrap()
}

fn make_long_range_duplicate_x(nqubits: usize) -> PauliOperator {
    let mut structures = Vec::new();
    let mut coefficients = Vec::new();
    for index in 0..nqubits {
        let mut z = vec![0_u8; nqubits];
        z[index] = 3;
        structures.push(z);
        coefficients.push(Complex64::new(0.01, 0.0));
    }
    for left in 0..nqubits {
        for right in (left + 1)..nqubits {
            let mut zz = vec![0_u8; nqubits];
            zz[left] = 3;
            zz[right] = 3;
            structures.push(zz);
            coefficients.push(Complex64::new(0.001, 0.0));
        }
    }
    for (left, right) in [(0, nqubits / 2), (1, nqubits - 2), (2, nqubits - 1)] {
        let mut xx = vec![0_u8; nqubits];
        let mut yy = vec![0_u8; nqubits];
        xx[left] = 1;
        xx[right] = 1;
        yy[left] = 2;
        yy[right] = 2;
        structures.push(xx);
        structures.push(yy);
        coefficients.push(Complex64::new(0.5, 0.0));
        coefficients.push(Complex64::new(0.5, 0.0));
    }
    PauliOperator::from_terms(nqubits, &structures, &coefficients).unwrap()
}

fn make_tfim(nqubits: usize) -> PauliOperator {
    let mut structures = Vec::with_capacity(2 * nqubits);
    let mut coefficients = Vec::with_capacity(structures.capacity());
    let mut global_x = vec![1_u8; nqubits];
    structures.push(std::mem::take(&mut global_x));
    coefficients.push(Complex64::new(0.25, 0.0));
    for index in 0..nqubits.saturating_sub(1) {
        let mut zz = vec![0_u8; nqubits];
        zz[index] = 3;
        zz[index + 1] = 3;
        structures.push(zz);
        coefficients.push(Complex64::new(-1.0, 0.0));
    }
    for index in 0..nqubits {
        let mut x = vec![0_u8; nqubits];
        x[index] = 1;
        structures.push(x);
        coefficients.push(Complex64::new(-0.2, 0.0));
    }
    PauliOperator::from_terms(nqubits, &structures, &coefficients).unwrap()
}

fn benchmark_z2(criterion: &mut Criterion) {
    let operator = make_tfim(8);
    let analysis = find_z2_symmetries(&operator, u128::MAX).unwrap();
    let plan = tencir_pauli_core::Z2TaperingPlan::new(
        operator.nqubits(),
        &analysis.generators,
        &vec![1_i8; analysis.generators.len()],
    )
    .unwrap();
    let mut group = criterion.benchmark_group("symmetry/z2");
    group.bench_function("analysis_8q", |bencher| {
        bencher.iter(|| black_box(find_z2_symmetries(&operator, u128::MAX).unwrap()))
    });
    group.bench_function("taper_transform_8q", |bencher| {
        bencher.iter(|| black_box(plan.transform_operator(&operator).unwrap()))
    });
    group.finish();
}

fn benchmark_u1(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("symmetry/u1");
    for (nqubits, particle_number) in [
        (12_usize, 2_usize),
        (16, 8),
        (63, 2),
        (64, 2),
        (65, 2),
        (128, 2),
        (129, 2),
        (128, 126),
        (256, 1),
        (256, 2),
        (512, 2),
    ] {
        let operator = make_hopping(nqubits);
        let sector = U1Sector::new(nqubits, particle_number).unwrap();
        let restricted = U1RestrictedOperator::new(&operator, sector.clone(), u128::MAX).unwrap();
        let plan = restricted.mvp_plan(u128::MAX).unwrap();
        let state = (0..plan.dimension())
            .map(|index| Complex64::new(index as f64, -(index as f64)))
            .collect::<Vec<_>>();
        group.throughput(Throughput::Elements(plan.dimension() as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "restriction_setup",
                format!("{nqubits}q_k{particle_number}"),
            ),
            &operator,
            |bencher, operator_input| {
                bencher.iter(|| {
                    black_box(
                        U1RestrictedOperator::new(operator_input, sector.clone(), u128::MAX)
                            .unwrap(),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("mvp_apply", format!("{nqubits}q_k{particle_number}")),
            &(&plan, &state),
            |bencher, (plan_input, state_input)| {
                bencher.iter(|| black_box(plan_input.apply(state_input, u128::MAX).unwrap()));
            },
        );
        if nqubits == 12 || (nqubits == 128 && particle_number == 2) {
            group.bench_with_input(
                BenchmarkId::new(
                    "csr_materialization",
                    format!("{nqubits}q_k{particle_number}"),
                ),
                &restricted,
                |bencher, restricted_input| {
                    bencher.iter(|| black_box(restricted_input.csr(u128::MAX).unwrap()));
                },
            );
            group.bench_with_input(
                BenchmarkId::new(
                    "coo_materialization",
                    format!("{nqubits}q_k{particle_number}"),
                ),
                &restricted,
                |bencher, restricted_input| {
                    bencher.iter(|| black_box(restricted_input.coo(u128::MAX).unwrap()));
                },
            );
        }
        if nqubits == 12 {
            group.bench_with_input(
                BenchmarkId::new(
                    "dense_materialization",
                    format!("{nqubits}q_k{particle_number}"),
                ),
                &restricted,
                |bencher, restricted_input| {
                    bencher.iter(|| black_box(restricted_input.dense(u128::MAX).unwrap()));
                },
            );
        }
    }
    let nqubits = 129;
    let particle_number = 2;
    let operator = make_long_range_duplicate_x(nqubits);
    let sector = U1Sector::new(nqubits, particle_number).unwrap();
    group.bench_function(
        "restriction_setup_long_range_duplicate_x_129q_k2",
        |bencher| {
            bencher.iter(|| {
                black_box(U1RestrictedOperator::new(&operator, sector.clone(), u128::MAX).unwrap())
            });
        },
    );
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1))
        .sample_size(30);
    targets = benchmark_z2, benchmark_u1
}
criterion_main!(benches);
