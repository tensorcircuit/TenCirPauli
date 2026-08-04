# Phase 6 implementation review（2026-08-02）

## Scope and verdict

审查范围为提交 `cbbf45c`、`9df99fe`、`02c6f0a`，基线为 `1172ea6`，验收合同为 `docs/vibe/phase-6-spec.md`。本次仅新增本评审报告，没有修改任何实现源码、测试或基准文件。

结论：这些提交已经形成一个有实际价值的 Phase 6 checkpoint，主路径 gate/state/observable/adjoint-gradient 数值结果基本可信，Rust core、PyO3 和 Python facade 的分层也正确；但目前不能按冻结规格认定 Phase 6 完成。没有发现 `CRITICAL` 级数值错误，存在 5 个应在阶段验收前处理的 `MAJOR` 问题，主要是 required fusion 未实现、diagonal 热路径仍做逐 amplitude 三角函数、facade cache/observable reducer 未落地、gradient 每 gate 克隆 state，以及 P0–P4 验收矩阵和 benchmark/documentation handoff 明显不完整。

建议状态：可以保留并继续迭代，不建议回滚；在 M1–M5 修复并补齐验收证据前，不应把 Phase 6 标记为 complete。

## Closure note（2026-08-04）

后续 Phase 6 remediation 已覆盖 M1–M5 的实现与本地验证；本报告保留为历史审查记录，Phase 6 的历史 acceptance scope 已为 0.2 release 收口，机器相关 benchmark handoff 不作为发布门禁。

## Compliance checklist

| Check | Status | Evidence |
|---|---|---|
| Pure-Rust core 与 PyO3/TensorCircuit 边界隔离 | PASS | 新逻辑分别位于 core、native binding、Python facade 和 optional integration；core 没有 Python 依赖。 |
| Terminal 使用 coarse-grained FFI，长运行释放 GIL | PASS | plan construction 和 run/expectation/gradient 均为单次 native 调用，并使用 `allow_threads`。 |
| Required gate、basis ordering 和基本 projected-observable 数值语义 | PASS | release build 下本地测试通过；锁定的 TensorCircuit 1.8.0/reference commit 差分为 8 passed、2 skipped。 |
| RZ/RZZ/CPhase/iSWAP 与参数表达式的 adjoint 数值正确性 | PASS | 额外混合电路有限差分最大误差为 `2.5e-11`；但正式测试覆盖仍不足，见 M5。 |
| Arbitrary-width execution 不重新引入 64-bit ceiling | PASS | 129-qubit k=1/k=2 tests 通过；完整 63/64/65、127/128/129、256 和 low-hole acceptance matrix 仍缺失，计入 M5。 |
| Required same-pair SWAP/iSWAP fusion 与 static/runtime block precomputation | FAIL | 每个 non-diagonal logical gate 仍独立生成 compiled gate；diagonal phase 仍在 amplitude 内层计算。见 M1、M2。 |
| Facade exact-parameter native final-state cache 被所有 terminals 复用 | FAIL | 只有 `state()`/Python `probability()` 使用 NumPy state cache，其他 terminals 从 initial state 重跑。见 M3。 |
| Private projected-observable compilation/reduction path | FAIL | 每次 expectation 都执行 term × basis 的 XOR、combinatorial rank 和 parity。见 M3。 |
| Gradient reverse 复用预分配 scratch，不进行 per-gate full-state clone | FAIL | reverse loop 每个 compiled gate 都执行 `state.clone()`。见 M4。 |
| Phase 6 correctness、concurrency、memory 和 benchmark acceptance matrix | FAIL | 当前 tests/benchmarks 只覆盖合同的一小部分，且 matched benchmark 的 native/JAX 参数化不一致。见 M5。 |
| Repository standard quality gate | PASS | release `maturin develop` 后 `python scripts/check.py` 成功；相关 Phase 6 tests 为 14 passed/9 skipped，使用本地锁定 TensorCircuit 后为 8 passed/2 skipped。 |

## CRITICAL

None found.

## MAJOR

### M1. Required same-pair SWAP/iSWAP fusion is not implemented

位置：`crates/tencir-pauli-core/src/u1_circuit.rs:150-157` 对每个 non-diagonal operation 单独调用 `compile_non_diagonal_gate`，而 `crates/tencir-pauli-core/src/u1_circuit.rs:781-807` 只返回单个 `Swap` 或 `Iswap`；这与 `docs/vibe/phase-6-spec.md:377-383` 要求的 maximal same-unordered-pair block fusion 不符。

实际影响：20 个连续同-pair iSWAP 编译后仍报告 20 个 compiled gates。release-mode 40q/k=5、32 个同-pair iSWAP 与一个等价合成 iSWAP 的结果完全一致，但 steady run 分别约 `4.05 ms` 和 `0.82 ms`，前者慢约 `4.9×`。当前 benchmark workload 在每个 iSWAP 后插入 CPhase，恰好绕开了 required repeated-same-pair case，因此没有暴露这个缺口。

处理建议：增加 private `PairBlock`，收集 maximal same-pair SWAP/iSWAP run；每次 run 先在 constant-size local matrix 上按原顺序 compose，再遍历一次 pair map。gradient reverse 保留 ordered micro-operations，并用 local prefix/suffix 或等价常数空间 VJP，不保存 full-state intermediates。

### M2. DiagonalBlock only reduces outer traversals; trigonometry remains inside the amplitude × operation loop

位置：`crates/tencir-pauli-core/src/u1_circuit.rs:420-425` 对每个 amplitude 遍历 block 内所有 operation，`crates/tencir-pauli-core/src/u1_circuit.rs:463-506` 又为每个 RZ/RZZ/CPhase 调用 `from_polar`。因此 compiled `gate_count == 1` 并不代表 phase work 已融合或预计算，违反 `docs/vibe/phase-6-spec.md:377-383` 的 static trigonometric/block precomputation 要求。

实际影响：release-mode 20q/k=10、32 个同-wire `rz(0, 0.01)` 与单个等价 `rz(0, 0.32)` 的输出最大误差为 `3.4e-16`，运行时间约 `19.85 ms` 对 `1.03 ms`，前者慢约 `19.2×`。这会直接打击 diagonal-heavy、深度较大的目标 workload。

处理建议：compile 时折叠纯 static diagonal micro-ops；run 时每个 dynamic angle 只计算一次所需 phase table，然后 amplitude loop 只做 bit lookup 和复乘。对 repeated identical support 可进一步 compose 成小型 lookup table，但无需引入跨 non-diagonal gate 的全局优化。

### M3. The facade cache and projected-observable path do not satisfy the terminal contract

位置：`python/tencirpauli/u1_circuit.py:420-436` 只缓存已经传回 Python 的 restricted state；`python/tencirpauli/u1_circuit.py:438-508` 的 `to_dense()`、`probability_full()`、`expectation_*()` 和 `value_and_grad()` 都从 `_initial_state` 重新调用 plan。与此同时，`crates/tencir-pauli-core/src/u1_circuit.rs:528-568` 每次 observable reduction 都对每个 term、每个 basis source 重建 destination words 并调用 `rank_words`，没有规格要求的 private projected-observable compilation path。

实际影响：先调用并缓存 `state()` 后，同参数的 repeated `expectation_z()` 仍稳定在约 `10 ms`，没有复用最终 state；40q/k=5 空线路的 identity expectation 约 `6.88 ms`，而 state copy 约 `0.63 ms`，慢约 `11×`。`probability()` 还在 Python 对已传回的 state 做 `abs(...) ** 2`，与“native final state 上直接 reduction”的合同不一致。

处理建议：在 mutable facade 内缓存一个 private native final-state handle 或等价 owned native state，并让 probability/full scatter/expectation reducers直接消费该 state；参数 key 继续使用 exact IEEE bytes。为 observable 增加 private compiled projected reducer，至少为 I/Z-only 使用直接 diagonal reduction，为 conserving flip patterns复用组合枚举/transition metadata；不需要公开 persistent observable plan。

### M4. Adjoint reverse allocates and copies a full state for every compiled gate

位置：`crates/tencir-pauli-core/src/u1_circuit.rs:315-330` 的 reverse loop 每次执行 `let mut before = state.clone()`，随后 inverse、计算 derivative，再用 `state = before` 替换。这不造成 `depth × dimension` 的峰值驻留，但会造成 `O(depth × dimension)` 的额外复制和 allocator traffic，违反 `docs/vibe/phase-6-spec.md:388-390` 的 scratch reuse 要求；M1 未融合又会放大这个问题。

处理建议：预分配一个 scratch state 并在两个 buffer 间 swap，或对当前 gate 类型改写 derivative 使 forward state 可先原位 inverse 后再用 local data 计算；实现 PairBlock 后在 block 内完成 local reverse/VJP。应增加 allocation-sensitive benchmark，而不必为了精确 RSS 引入复杂 allocator instrumentation。

### M5. The acceptance and performance handoff is incomplete, and the current JAX comparison is not fully matched

位置：`tests/test_circuit_ir.py` 只有 38 行，`tests/test_u1_circuit.py` 只有 217 行；缺少 frozen spec 要求的 malformed native IR/schema tests、k=0/k=n/low-hole、完整 width matrix、cache invalidation across all terminals、all-gate/shared-slot gradient、1-thread/multi-thread determinism、concurrent calls 和 memory guard。`benchmarks/python/test_u1_circuit_benchmark.py:38-48` 的 native circuit 使用常量 angles，而 `benchmarks/python/test_u1_circuit_benchmark.py:77-93` 的 JAX circuit 使用 runtime parameter array，不是规格要求的 matched parameterized comparison；metadata 在 `benchmarks/python/test_u1_circuit_benchmark.py:100-128` 只记录 logical gate count、state bytes 和 process-wide peak RSS，没有 before/after fusion、pair-map/scratch bytes、accuracy 或 gradient data。仓库没有 Phase 6 的 recorded `.benchmarks` result，也没有新增 Rust microbenchmark。

交付状态同样未完成：`README.md:100` 仍写着 lazy U1 circuit “planned in Phase 6”，`docs/vibe/implementation-status.md:160-164` 仍把 Phase 6 列为 future/frozen，`CHANGELOG.md` 没有记录新 API。这些不是文字洁癖，而是当前实现是否已经可发布、是否满足阶段验收的直接矛盾证据。

处理建议：按 spec Section 10/11 建立一个最小但完整的 acceptance matrix；native/JAX 两侧都使用相同 runtime parameter vector；补 repeated-pair、diagonal-heavy、observable、gradient、wide/low-hole、facade-vs-plan benchmark 和至少一个 core Criterion hot-path benchmark；记录一次同机 release result 后再同步 README、CHANGELOG、implementation status 和 typing/docs。当前额外实测的 20q/k=2/160-gate case 中 native steady 约 `5.0 us`、JAX warm-JIT 约 `278 us`，最大 state 误差 `1.6e-16`，已经足以证明 native plan 有明确用途，但不能替代完整验收。

## MINOR

### N1. `expectation_ps` silently accepts mutually exclusive or duplicate friendly inputs

位置：`python/tencirpauli/u1_circuit.py:137-158` 在 `ps is not None` 时直接返回并忽略同时传入的 `x/y/z`；同一个 x/y/z list 内的重复 index 也被当作合法输入。已复现 `expectation_ps(x=[0], ps=[0, 0])` 静默返回 identity expectation，`expectation_ps(x=[0, 0])` 也不报错。这违反 `docs/vibe/phase-6-spec.md:290-292` 的 fail-fast 输入合同，可能掩盖调用者错误。

处理建议：`ps` 与任一 x/y/z 同时非 `None` 时抛 `ValueError`，并分别检查每个 index sequence 内无重复。

### N2. Native common-IR validation does not reject parameter holes, and the transport is not actually schema-versioned

位置：`crates/tencirpauli-native/src/u1_circuit.rs:179-186` 的 native constructor 没有 schema-version 参数，`crates/tencirpauli-native/src/u1_circuit.rs:252-258` 总是强制当前常量版本；`crates/tencir-pauli-core/src/circuit_ir.rs:212-253` 只检查 slot 小于推断的 `nparameters`，不检查所有 slots 都出现。已直接构造并接受只有 `Slot(2)` 的 native IR，plan 报告 3 个参数。public Python program 正常不会生成这种输入，因此当前风险低于 MAJOR，但 P0/P1 malformed-IR acceptance 未满足。

处理建议：transport 显式携带 schema version 和 declared `nparameters`，native validation 比较 observed slot set 是否严格等于 `0..nparameters-1`；再补 unknown schema/opcode、holes、dead/malformed nodes tests。是否立即改成 NumPy fixed-width arrays可由 profile 决定，但版本字段和 holes rejection 不应省略。

### N3. Pair-map construction performs one small heap allocation per enumerated pair

位置：`crates/tencir-pauli-core/src/u1_circuit.rs:850-879` 在 combination callback 内执行 `vec![0_u64; word_count]`。40q/k=5 的单 pair 约有 73,815 个组合；当前实测额外 compile 时间约 `1.4 ms`，尚未成为主瓶颈，但这是明确的 hot-path allocation anti-pattern，在 many-distinct-pair 或更宽 word count 下会放大。

处理建议：在 callback 外复用一个 words scratch，或让组合枚举直接维护 packed occupation buffer。优先级低于 M1–M4，不建议先为它引入复杂抽象。

## OBSERVATIONS

- 主路径 gate semantics、MSB qubit ordering、129-qubit cross-limb execution和 projected Pauli expectation 的现有 tests 都通过；锁定版 TensorCircuit state/probability/observable 差分也通过，没有发现 global-phase 掩盖或明显 ordering 错误。
- 自定义混合 RZ/RZZ/CPhase/iSWAP、非线性 `+ - * /` expression 的 adjoint gradient 与 central finite difference 最大误差约 `2.5e-11`，说明核心 VJP 公式本身是可信的；问题主要是测试覆盖和执行代价。
- `unsafe` Rayon pair update 有局部 disjoint-endpoint safety argument，并有跨并行阈值的 40q/k=5 round-trip test；本次没有发现 unsafe correctness defect。
- Python gate append 每次重建 tuple 并重新验证完整 tape，理论上是 `O(depth^2)` setup；现有 depth/计时证据尚不足以把它列为实际瓶颈，暂不建议为此重构。
- `max_bytes` 的 aggregate metadata check 在所有 pair/index maps 已经构建后才执行（`crates/tencir-pauli-core/src/u1_circuit.rs:160-188`），严格说会晚于目标 guard；默认限制下暂无实测问题，可在处理 M1/M3 的 metadata redesign 时顺手改为增量累计，不建议单独过度工程化。

## IMPLEMENTATION SKETCHES FOR THE HARDEST FIXES

以下 sketch 不是新增公开 API 设计，而是给后续实现者的最小内部 handoff。应保持现有 `U1Circuit`/`U1CircuitPlan` public signatures、TensorCircuit gate convention、strict `restrict_u1()` 语义和 deterministic output 不变；不要借修复之机引入 public general U(2) gate、persistent observable plan、full-state backend 或 exact-RSS accounting。

### S1. M1 pair-block fusion and exact VJP

#### Core representation

在 `crates/tencir-pauli-core/src/u1_circuit.rs` 中用一个 private block 替换逐 gate 的 `Swap`/`Iswap` variants。矩阵约定固定为 column vector，local basis order固定为 `(|01>, |10>)`：

```rust
type Matrix2 = [[Complex64; 2]; 2];

#[derive(Clone, Debug)]
enum PairMicroOp {
    Swap,
    Iswap { angle: usize },
}

#[derive(Clone, Debug)]
struct PairBlock {
    pairs: Arc<[PairIndex]>,
    operations: Arc<[PairMicroOp]>,
    // Some only when every angle is constant after expression folding.
    static_total: Option<Matrix2>,
}

enum CompiledU1Gate {
    PairBlock(PairBlock),
    DiagonalBlock { operations: Arc<[DiagonalOp]> },
}
```

当前 required SWAP 和 iSWAP matrices 都对交换 wire orientation 对称，因此 pair map 可以继续按 unordered `(min_wire, max_wire)` 共享；实现时仍应把 block 的 canonical pair key 固定下来，并增加 reversed-wire tests，避免未来 private block 扩展后误用该性质。

#### Compiler scan

现有 compile loop 遇到第一个 non-diagonal gate 后，不再立即 push 单 gate，而是取它的 unordered pair key，继续消费后续连续、同 unordered pair 的 SWAP/iSWAP，遇到 diagonal 或不同 pair 立即停止。不得跨任何中间 gate 融合，即使该 gate 数值上是 identity。

```text
while operation_index < logical_ops.len():
    if next op begins a diagonal run:
        compile maximal diagonal run
    else:
        pair_key = unordered_pair(next op)
        micro_ops = consume maximal consecutive SWAP/iSWAP with pair_key
        pairs = get_or_build_shared_pair_map(pair_key)
        static_total = compose once iff every referenced angle node is Constant
        push PairBlock(pairs, micro_ops, static_total)
```

矩阵按原 gate order compose。若 `psi' = U_j psi`，循环应使用 `total = U_j * total`；最终 `total = U_(m-1) ... U_1 U_0`。SWAP matrix 为 `[[0, 1], [1, 0]]`，iSWAP 令 `phi = theta * pi / 2`，matrix 为 `[[cos(phi), i sin(phi)], [i sin(phi), cos(phi)]]`。不要把 SWAP 偷换成某个 iSWAP angle，因为两者相差 local phases。

#### Forward and inverse

每次 run 只对 dynamic block compose 一次 `Matrix2`，然后对 shared pair map 做一次 traversal。现有 disjoint-endpoint Rayon kernel可抽成 `apply_pair_matrix(state, pairs, matrix)`；inverse 直接使用 conjugate transpose，不重新逆序逐 micro-op 扫 state。

#### Exact block VJP

不要为 block 内每个 micro-op 保存 full restricted state。对长度 `m` 的 block，每次 reverse 只构造 `O(m)` 个 2×2 matrices：

```text
prefix[0] = I
prefix[j + 1] = U_j * prefix[j]

suffix[m] = I
suffix[j] = suffix[j + 1] * U_j

dTotal_j = suffix[j + 1] * dU_j * prefix[j]
```

其中 iSWAP 对用户 `theta` 的显式导数为

```text
alpha = pi / 2
dU/dtheta = [[-alpha sin(phi), i alpha cos(phi)],
             [i alpha cos(phi), -alpha sin(phi)]]
```

对每个 dynamic micro-op `j`，按 pair map 的固定排序累加

```text
df/dtheta_j += 2 Re[lambda_after_pair^dagger * dTotal_j * psi_before_pair]
```

然后加到对应 expression node adjoint。为了与未融合 reverse 的 floating accumulation order保持稳定，block 内 micro-ops 按 reverse gate order `m-1..0` 处理；pair reduction保持顺序 serial，或使用固定 logical chunks并按 chunk index merge，不能直接做 schedule-dependent parallel sum。SWAP 没有 derivative，但仍参与 prefix/suffix。

#### Required tests

- `SWAP`、iSWAP、混合序列、reversed wire order和 special angles 的 fused state 对 independent dense reference。
- 20 个同-pair gate 必须编译为一个 block；插入 diagonal identity 后必须形成两个 blocks，证明没有跨边界重排。
- shared slot、compound expression、SWAP+iSWAP mixed block 的 analytic gradient 对 finite difference和 TensorCircuit JAX-x64。
- fused 与人为 barrier 阻止 fusion 的等价线路直接比较 state/value/gradient，不做 global-phase alignment。
- 1-thread/multi-thread bitwise一致；保留 repeated-pair release benchmark并记录 logical/compiled gate counts。

### S2. M2 diagonal evaluation without per-amplitude trigonometry

保留 `DiagonalBlock` ordered micro-operations，但在进入 amplitude loop 前先把每个 operation变成只含 small lookup 的 evaluated form：

```rust
enum EvaluatedDiagonalOp<'a> {
    OneWire { wire: usize, phase: [Complex64; 2] },
    TwoWire { wire0: usize, wire1: usize, phase: [Complex64; 4] },
    Static { wires: &'a [usize], payload: &'a [Complex64] },
}
```

RZ、RZZ 和 CPhase 的 `sin/cos/from_polar` 每个 operation 每次 run 只执行一次；CZ lookup固定。compile 时把 consecutive、相同 ordered support 的 pure-static lookup按 gate order逐 entry相乘，因此 repeated same-wire static RZ 会变成一个 2-entry lookup。不同 support 不强行合成 dimension-length phase vector，以免用额外 `O(dimension)` metadata换取小 workload 的收益。

Amplitude loop 只做 bit extraction、lookup 和 complex multiply。gradient reverse 可以复用同一 evaluated block：对 index 计算 forward total phase，得到 `psi_after = total_phase * psi_before`，再按现有 RZ/RZZ/CPhase generator公式累计，不需要重新调用三角函数。

Required tests应覆盖 static/dynamic mixed run、不同 support、相同 support folding、static arbitrary diagonal、inverse和所有 dynamic diagonal gradients；release benchmark至少保留 repeated same-wire RZ 与 many-support diagonal-heavy 两类，避免只优化评审中的合成案例。

### S3. M4 reverse without a full-state clone per gate

在 S1/S2 后，把 local derivative contract统一改成接收 `psi_before` 和 `lambda_after`。由于所有 accepted gates严格 unitary，reverse loop可以先原位恢复 pre-gate state，再计算 derivative：

```rust
for gate in self.gates.iter().rev() {
    self.apply_inverse_gate(&mut state, gate, &values)?;
    // state is psi_before; lambda is still lambda_after.
    self.accumulate_gate_derivative(
        &state,
        &lambda,
        gate,
        &values,
        &mut node_adjoint,
    )?;
    self.apply_inverse_gate(&mut lambda, gate, &values)?;
}
```

PairBlock 使用 S1 的 `dTotal_j * psi_before_pair`；DiagonalBlock 使用 S2 的 total phase重建局部 `psi_after`。这样可以完全删除 `state.clone()` 和 state scratch，而不仅是把 allocator call换成 `copy_from_slice`。若实现者希望分两步迁移，先复用一个 scratch buffer是安全的中间状态，但最终 benchmark应确认 reverse 不再有 per-block full-state copy。

Required tests除现有 gradient differential外，还应加入 depth scaling和 allocation-sensitive benchmark；不要设置 wall-time CI gate，只记录同机 release 数据和 `O(dimension)` peak workspace。

### S4. M3 projected observable reuse and native final-state cache

M3 应拆成两个独立 commit：先复用 Phase 5 restricted compiler得到正确且确定性的 projected reducer，再加入 private native final-state handle。不要在一个大 commit 中同时重写 observable algebra、PyO3 ownership和 facade cache。

#### S4.1 Reuse Phase 5 term grouping without materializing a per-call CSR

`crates/tencir-pauli-core/src/sector.rs:862-1085` 已经提供正确的 X-mask grouping、duplicate aggregation、Y phase、Z parity、arbitrary-width active-basis iteration和 rank helpers。应复用这些语义内核，但不要让每次 circuit expectation都走 `crates/tencir-pauli-core/src/sector.rs:356-440` 的完整 two-pass CSR materialization：observable plan不是 persistent public object，为 identity/Z objective临时构建 `O(dimension)` CSR 会制造额外 setup 和 memory，实际可能比当前实现更慢。

先把 `aggregate_source()` 的 leakage behavior参数化，同时保持现有 strict constructor固定使用 `Reject`：

```rust
#[derive(Clone, Copy)]
pub(crate) enum U1LeakagePolicy {
    Reject,
    Project,
}
```

唯一语义分支放在 `aggregate_source()` 当前 `actual_weight != particle_number` 的位置：`Reject` 保持现有 `SectorLeakage` error；`Project` 直接 `continue`，即丢弃 `P_k O (1-P_k)`/`(1-P_k) O P_k` transitions。`U1RestrictedOperator::new()` 必须继续固定传 `Reject`，circuit private projected observable compiler传 `Project`。其余 grouping、aggregation、phase和 rank逻辑完全共享，从结构上保证不会弱化 public strict restriction。

在 `sector.rs` 内增加 crate-private、metadata-only projected plan；放在同一模块可避免把 `CompiledU1Terms` 等内部细节扩大为 public API：

```rust
pub(crate) struct U1ProjectedObservablePlan {
    sector: U1Sector,
    terms: CompiledU1Terms,
}

impl U1ProjectedObservablePlan {
    pub(crate) fn new(
        operator: &PauliOperator,
        sector: U1Sector,
        max_bytes: u128,
    ) -> Result<Self, PauliError>;

    fn apply_into(&self, state: &[Complex64], output: &mut [Complex64]) -> Result<(), PauliError>;
    fn expectation(&self, state: &[Complex64]) -> Result<Complex64, PauliError>;
}
```

`new()` 只执行 `compile_terms()`、丢弃 odd-X 或组合计数证明不可能保 sector 的 groups，并估算 term metadata；它不生成 `indptr/columns/values`。`apply_into()` 使用一个 `U1BasisIterator` 按 source order单次扫描：每个 source 调 `aggregate_source(..., Project)`，然后执行 `output[destination] += value * state[source]`。`expectation()` 可以在同一 traversal直接固定顺序累加 `conj(state[destination]) * value * state[source]`，不需要 output buffer。该 source-major serial baseline天然 deterministic；只有 profile证明 reducer占主导时才考虑 fixed-chunk parallelism。

X mask为空时 destination就是 source，应直接使用 `source_index`，不要再次调用 combinatorial rank。这样 identity/Z-only objective退化为一次 `O(dimension × diagonal_term_groups)` scan，不会建立临时 CSR；多个 terms共享同一 X mask时只 rank一次 destination，也明显优于当前 term × source rank。`value_and_grad` compile一次 plan并用 `apply_into()` 生成 `lambda_final = O psi_final`。

必须增加一个成对 regression：同一 leaking operator经 public `restrict_u1()` 仍抛错，而 circuit expectation返回 dense `P_k O P_k` 结果。single X/Y 应得到零，XX、XX+YY和 duplicate-aggregation cancellation必须覆盖。

#### S4.2 Private native final-state handle

core 保持纯 Rust；cache ownership放在 native/Python 两层。将 native plan包装改为 shared immutable ownership，并新增 private pyclass：

```rust
#[pyclass(module = "tencirpauli._native")]
struct NativeU1CircuitPlan {
    plan: Arc<U1CircuitPlan>,
}

#[pyclass(module = "tencirpauli._native")]
struct NativeU1FinalState {
    plan: Arc<U1CircuitPlan>,
    state: Arc<[Complex64]>,
    parameters: Arc<[f64]>,
}
```

`NativeU1CircuitPlan.run_cached(initial, parameters)` 执行一次 forward并返回 handle；handle 提供 private `state_array()`、`probability()`、`to_dense()`、`probability_full()`、`expectation(...)` 和 `value_and_grad(...)`。为此将 core terminals机械拆成 `run()` 与 `*_from_state()` helpers：public immutable plan methods仍保持 stateless并沿用原 signatures，facade才使用 cached handle。`value_and_grad_from_final()` 从 handle保存的 final state和 parameters开始 reverse，不再重复 forward。

Python facade 统一通过一个 helper取 cache：

```python
def _final_state(self, parameters: object) -> tuple[np.ndarray, NativeU1FinalState]:
    values = _parameter_array(parameters, self.nparameters)
    key = (self._generation, values.tobytes())
    if self._state_cache is None or self._state_cache.key != key:
        native = self.compile()._native.run_cached(self._initial_state, values)
        self._state_cache = _FinalCache(key=key, native=native, array=None)
    if self._state_cache.array is None:
        self._state_cache.array = _readonly(self._state_cache.native.state_array())
    return self._state_cache.array, self._state_cache.native
```

实际实现可拆成 `_cached_final()`（不强制复制 NumPy state）和 `state()`（按需 materialize array），使 expectation/probability在用户未请求 state 时完全不回传 amplitudes。所有 facade terminals必须先取得同一 exact-bit key 的 handle；append继续使 cache失效。`+0.0`/`-0.0` 由现有 `tobytes()` 区分，NaN/Inf仍在建 key 前拒绝。

Memory guard至少计入 native final state；若 `state()` 同时缓存 NumPy owned output，再计入第二个 state-sized allocation。无需追踪 Python object overhead或 allocator transient。handle持有 `Arc<U1CircuitPlan>`，所以旧 plan即使 facade append后也不会悬空；facade只丢弃引用，不需要 mutable native global state。

#### S4.3 Cache and concurrency tests

- 用 fake/private counting plan验证 `state -> probability -> expectation -> to_dense` 在相同 parameter bits下只 forward一次；不同参数、append和 `+0.0/-0.0` 分别触发新 run。
- 只调用 expectation 时不 materialize Python state array；调用 state后 array必须 owned、C-contiguous、read-only。
- immutable `U1CircuitPlan.run()` 仍可并发且不共享 final state；两个 facade instances不共享 cache。
- cached 与 stateless plan terminals逐项 bitwise/数值比较；gradient cache path必须与 fresh forward path一致。
- 在 memory-limit regression中分别覆盖 native handle、NumPy state materialization、projected plan和 adjoint lambda workspace。

### S5. Suggested commit order

为了降低返工和 review 面积，建议按以下独立 commits落地：

1. `fix phase 6 input/native IR validation`：N1/N2 和 regression tests。
2. `optimize phase 6 diagonal evaluation`：S2，不改 pair/gradient结构。
3. `add phase 6 fused pair blocks`：S1 forward、inverse和 compiled metadata。
4. `complete fused pair adjoint and in-place reverse`：S1 VJP + S3。
5. `factor projected U1 observable compiler`：S4.1，证明 strict/projected 分离。
6. `add native final-state cache terminals`：S4.2/S4.3。
7. `complete phase 6 acceptance and benchmark handoff`：M5 docs、tests、Criterion/Python records。

每个 commit先过 small dense/TensorCircuit correctness，再做 release benchmark；不要把全部变化压成一个难以二分的性能 commit。

## RECOMMENDED IMPROVEMENTS

1. 先实现 M1、M2：它们是最明确且已有 release 数据支持的 forward hot-path瓶颈，并且是冻结规格的 required fusion。
2. 再实现 M3：统一 native final-state cache 与 projected observable reducers，避免 terminal 重跑和 term × basis combinatorial ranking。
3. 在 PairBlock 结构稳定后处理 M4，复用 reverse scratch 并保留 exact adjoint semantics。
4. 修复 N1/N2 的 fail-fast 行为并补最小 regression tests；N3 只做简单 scratch reuse，不扩展抽象层。
5. 最后完成 M5 的 matched benchmark、full acceptance matrix 和 release documentation handoff，再将 Phase 6 状态改为 complete。
