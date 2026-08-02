use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tencir_pauli_core::{
    Clifford1, Clifford2, Complex64, GateOperation, ParameterRef, PauliOperator, ProductState,
    PropagationEngine, RotationAxis, SPPSEngine,
};

fn observable(nqubits: usize, count: usize) -> PauliOperator {
    let structures = (0..count)
        .map(|index| {
            let mut codes = vec![0_u8; nqubits];
            codes[index % nqubits] = 1;
            codes
        })
        .collect::<Vec<_>>();
    let coefficients = (0..count)
        .map(|index| Complex64::new(0.1 + index as f64 * 0.001, 0.0))
        .collect::<Vec<_>>();
    PauliOperator::from_terms(nqubits, &structures, &coefficients).unwrap()
}

fn tape(nqubits: usize, layers: usize) -> Vec<GateOperation> {
    let mut operations = Vec::new();
    for layer in 0..layers {
        for wire in 0..nqubits {
            operations.push(
                GateOperation::clifford1(
                    nqubits,
                    if layer % 2 == 0 {
                        Clifford1::H
                    } else {
                        Clifford1::S
                    },
                    wire,
                )
                .unwrap(),
            );
        }
        for wire in (layer % 2..nqubits.saturating_sub(1)).step_by(2) {
            operations
                .push(GateOperation::clifford2(nqubits, Clifford2::Cnot, wire, wire + 1).unwrap());
        }
    }
    operations
}

fn deep_near_clifford_tape(nqubits: usize, layers: usize) -> Vec<GateOperation> {
    let mut operations = Vec::new();
    for layer in 0..layers {
        for wire in (0..nqubits).step_by(2) {
            operations.push(GateOperation::clifford1(nqubits, Clifford1::H, wire).unwrap());
            operations.push(GateOperation::clifford1(nqubits, Clifford1::S, wire + 1).unwrap());
            operations
                .push(GateOperation::clifford2(nqubits, Clifford2::Cnot, wire, wire + 1).unwrap());
        }
        for wire in (1..nqubits.saturating_sub(1)).step_by(2) {
            operations
                .push(GateOperation::clifford2(nqubits, Clifford2::Cz, wire, wire + 1).unwrap());
        }
        for wire in (layer % 8..nqubits).step_by(8) {
            let angle = 0.031 + 0.0001 * wire as f64 + 0.002 * layer as f64;
            operations.push(
                GateOperation::rotation(
                    nqubits,
                    RotationAxis::Z,
                    wire,
                    None,
                    ParameterRef::Static {
                        cos: angle.cos(),
                        sin: angle.sin(),
                    },
                )
                .unwrap(),
            );
        }
    }
    operations
}

fn benchmark_local_kernels(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("propagation/local");
    let observable_word = observable(2, 1);
    let rotation = GateOperation::rotation(
        2,
        RotationAxis::X,
        0,
        Some(1),
        ParameterRef::Static {
            cos: 0.91,
            sin: 0.41,
        },
    )
    .unwrap();
    let engine = PropagationEngine::new(
        2,
        vec![rotation],
        observable_word,
        ProductState::Zero,
        None,
        None,
    )
    .unwrap();
    group.bench_function("rotation_branch_2q", |bencher| {
        bencher.iter(|| black_box(engine.propagate(&[]).unwrap()))
    });
    group.finish();
}

fn benchmark_tapes(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("propagation/tape");
    for (nqubits, layers) in [(12_usize, 3_usize), (100, 4)] {
        let engine = PropagationEngine::new(
            nqubits,
            tape(nqubits, layers),
            observable(nqubits, 4),
            ProductState::Zero,
            None,
            None,
        )
        .unwrap();
        group.throughput(Throughput::Elements(engine.gate_count() as u64));
        group.bench_with_input(
            BenchmarkId::new("exact", format!("{nqubits}q_{layers}layers")),
            &engine,
            |bencher, engine_input| {
                bencher.iter(|| black_box(engine_input.expectation(&[]).unwrap()))
            },
        );
    }
    let deep_engine = PropagationEngine::new(
        128,
        deep_near_clifford_tape(128, 12),
        observable(128, 4),
        ProductState::Zero,
        Some(4),
        None,
    )
    .unwrap();
    group.throughput(Throughput::Elements(deep_engine.gate_count() as u64));
    group.bench_with_input(
        BenchmarkId::new("weight_projected", "128q_12layers"),
        &deep_engine,
        |bencher, engine_input| bencher.iter(|| black_box(engine_input.expectation(&[]).unwrap())),
    );
    group.finish();
}

fn benchmark_value_and_gradient(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("propagation/value_and_gradient");
    let mut operations = Vec::new();
    for layer in 0..3 {
        for wire in 0..12 {
            operations.push(
                GateOperation::rotation(
                    12,
                    RotationAxis::Y,
                    wire,
                    None,
                    ParameterRef::Slot(wire % 2),
                )
                .unwrap(),
            );
        }
        for wire in (layer % 2..11).step_by(2) {
            operations.push(GateOperation::clifford2(12, Clifford2::Cnot, wire, wire + 1).unwrap());
        }
    }
    let engine = PropagationEngine::new(
        12,
        operations,
        observable(12, 12),
        ProductState::Zero,
        Some(3),
        None,
    )
    .unwrap();
    for interval in [1_usize, 4, 16] {
        group.bench_with_input(
            BenchmarkId::new("deterministic", format!("12q_interval_{interval}")),
            &interval,
            |bencher, interval| {
                bencher.iter(|| {
                    black_box(
                        engine
                            .value_and_grad(&[0.13, -0.21], Some(*interval))
                            .unwrap(),
                    )
                })
            },
        );
    }

    let spps = SPPSEngine::new(
        12,
        vec![
            GateOperation::rotation(12, RotationAxis::Y, 0, None, ParameterRef::Slot(0)).unwrap(),
            GateOperation::rotation(12, RotationAxis::Z, 1, None, ParameterRef::Slot(1)).unwrap(),
        ],
        observable(12, 12),
        ProductState::Zero,
        0.01,
        None,
    )
    .unwrap();
    group.throughput(Throughput::Elements(256));
    group.bench_function("spps_12q_12terms_256paths", |bencher| {
        bencher.iter(|| black_box(spps.value_and_grad(&[0.13, -0.21], 256, 7).unwrap()))
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(1))
        .sample_size(30);
    targets = benchmark_local_kernels, benchmark_tapes, benchmark_value_and_gradient
}
criterion_main!(benches);
