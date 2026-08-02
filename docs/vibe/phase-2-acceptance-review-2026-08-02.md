# Phase II 审查报告

审查基线：`main` at `96fc171`，审查对象为该基线之上的全部当前工作树改动。审查日期：2026-08-02。结论：**Phase II 当前不应合并或标记为完成**；存在 1 个会返回错误物理 sector 结果的 CRITICAL 正确性缺陷，以及性能实现、正确性 gate 和验收证据方面的 MAJOR 问题。原 M1、M2、M6 已在 owner 复核后分别转为 best-effort memory scope、已知 64+ qubit 限制并排入 Phase 5，以及 serialization de-scope，不再作为当前 Phase II 实现阻断项。

Owner decisions recorded on 2026-08-02：

- `max_bytes` 只保留为可廉价估算 major allocation 的 best-effort guard，不追求完整 FFI/transient/allocator peak accounting，也不承诺避免操作系统 OOM。
- 当前 native U1 restricted Hamiltonian/MVP/CSR 明确限制为 `nqubits < usize::BITS`；64+ qubit packed multiword U1 Hamiltonian 已排入 roadmap Phase 5。
- TensorCircuit-style `U1Circuit` fixed-particle-number circuit execution 和含时演化已排入 roadmap Phase 6，不属于当前 static restricted Hamiltonian slice。
- 稳定 pickle/JSON/跨进程 Z2 plan serialization 没有当前用户需求，从 Phase II 验收范围移除；主要工作流是在 Python 进程内直接调用 sparse matrix、MVP plan 和函数。

## COMPLIANCE CHECKLIST

| 检查项 | 状态 | 证据与说明 |
|---|---|---|
| Rust core 保持纯 Rust，TensorCircuit 仍为可选集成 | PASS | 新 core 模块只依赖 core 类型；TensorCircuit import 边界未被改变。 |
| PyO3 调用保持粗粒度并在长计算时释放 GIL | PASS | `crates/tencirpauli-native/src/symmetry.rs:72-90,95-116,159-177,181-203,235-267` 均按整批 operator/sector 调用并使用 `allow_threads`。 |
| 确定性输出与聚合后 U1 leakage 判断 | PASS | Z2 elimination/pivot 顺序确定；U1 在 `crates/tencir-pauli-core/src/sector.rs:178-205` 先按 destination 聚合再判断 leakage。额外 4,000 组 n=1..4 随机小系统差分未发现 U1 action/leakage 错误。 |
| Z2 tapering 的 sector、phase 和谱正确性 | FAIL | 见 CRITICAL C1；自动发现的多生成元例子把 `(+,+,+)` sector 的能量从 `+3` 错变为 `-3`。 |
| `max_bytes` 严格覆盖所有 allocation/scratch/output | SCOPED OUT | Owner 已决定它只是 cheap best-effort guard；checked dimensions/overflow 仍必须保留，但 exact peak accounting 不再是验收项。 |
| U1 API 的 qubit-count 合同一致 | KNOWN LIMIT | 当前 native restriction 明确只支持 `< usize::BITS`；64+ qubit multiword Hamiltonian 已排入 Phase 5。 |
| 热路径符合连续存储、预计算、少分配和可复用 scratch 规范 | FAIL | 见 MAJOR M3。 |
| Phase II property/differential 测试足以保护 phase、sector 和 ordering | FAIL | 见 MAJOR M4；现有 Z2 测试未逐 sector 对 projector/reference，因而漏过 C1。 |
| Phase II 性能结论可由保存的 release benchmark、内存和 scaling 证据复核 | FAIL | 见 MAJOR M5。 |
| `clifford_operations` 已版本化并可稳定序列化/恢复 | SCOPED OUT | 当前需求仅为 Python 进程内复用；stable persistence 已从 Phase II 合同移除。 |
| 通用格式、lint、typing、构建与现有测试 | PASS | `python scripts/check.py --benchmark skip` 通过：rustfmt、Black、Clippy `-D warnings`、Ruff、strict mypy、10 个 Rust tests、66 passed/2 skipped Python tests。 |

## CRITICAL

### C1. 多生成元 tapering 丢失 Clifford 行符号，导致 sector 被错误映射

位置：`crates/tencir-pauli-core/src/symmetry.rs:184-215,394-405`。`rows` 的第二个 `i8` 在初始化后从未更新；`apply_gate()` 只更新独立的 `row_signs`。但消去已用 pivot 时，代码在 `symmetry.rs:212` 乘的是始终为 `+1` 的 `rows[previous].1`，而不是实际的 `row_signs[previous]`。因此，若前一生成元在 Clifford conjugation 中获得负号，后续 row combination 会丢失该负号。

可复现例：

```python
h = tcp.PauliOperator.from_terms(3, (("ZYY", 1.0), ("YIZ", 2.0)))
a = h.find_z2_symmetries()
# generators == (IXZ, XYX, YXI)
p = a.tapering_plan((1, 1, 1))
```

对原空间三个 generator 构造 `Π_i (I + S_i)/2` 后，`Tr(PH)/Tr(P) = +3`；当前 `p.transform_operator(h).dense()[0, 0] = -3`。同一 plan 还把第三个原 generator `YXI` 在其 `+1` sector 中变成标量 `-1`。这会直接产生错误的 sector Hamiltonian、能量和 observable，属于发布阻断问题。

## MAJOR

### M1. [SCOPE RESOLVED] `max_bytes` 不是完整 peak-allocation gate

位置：`crates/tencir-pauli-core/src/symmetry.rs:62-104,136-313,340-390,655-718`，`crates/tencir-pauli-core/src/sector.rs:379-407`，`python/tencirpauli/symmetry.py:79-86`。Z2 analysis 只计算 packed payload，却同时保留 `Vec<Vec<_>>` headers、null basis、`candidates`、`candidate_bits`、selected vectors 和 elimination clones；例如 3-qubit、2-term operator 在 `max_bytes=64` 时通过，但仅两个受计 payload 已用满 64 bytes，实际 live allocation 明显更大。U1 CSR 只检查最终 `(indptr, columns, values)`，随后又分配完整 `by_destination` scratch；3-dimensional/2-nnz 例子在 `max_bytes=80` 时成功，虽然输出本身已是 80 bytes，构建期间还至少同时持有约 120 bytes 的 row headers/entry scratch。`Z2TaperingPlan::new()` 与 `transform_operator()` 则没有 `max_bytes`，后者还在每个 term 内复制 `removed_qubits` 并建立 code vectors。

原 spec 把它描述为完整 scratch/output gate，因此该实现与原合同不一致。Owner 已决定不投入复杂 peak accounting：`max_bytes` 现在只表示对 major output/workspace 的廉价 best-effort estimate，允许遗漏 allocator overhead、FFI conversion 和 transient scratch，也不保证避免 OOM。相应 AGENTS/README/spec 已调整；本项不再要求实现 patch。

### M2. [KNOWN LIMIT / PHASE 5] U1 native restriction 当前不支持 64+ qubits

位置：`python/tencirpauli/symmetry.py:132-182`，`crates/tencir-pauli-core/src/sector.rs:75-143,417-455,477-483`。Python `U1Sector` 明确为 `nqubits > 64` 返回 tuple/packed multiword basis，但 native `rank`、`unrank` 和 `apply_term` 使用单个 `usize`，并在 `nqubits >= usize::BITS` 时返回 overflow。可复现：64-qubit identity operator、`U1Sector(64, 0)` 的 `basis_words()` 成功，但 `restrict_u1()` 抛出 `OverflowError: integer overflow while representing a computational basis integer`。这不是维度或内存超限，dimension 仅为 1。

Owner 已选择当前 Phase II 明确限制 native restriction `< usize::BITS`，而不是立即扩大实现。文档必须区分“Python `U1Sector` combinatorial/basis helper 支持宽 bitstring”和“native restricted Hamiltonian 仅支持单字 basis index”；当前入口应给出清楚的 unsupported-width error。完整 packed multiword transition/rank/lookup、MVP/CSR 和 64/65/128-qubit low-k tests/benchmarks 已排入 Phase 5；U1Circuit/time evolution 已排入 Phase 6。

### M3. U1/Z2 热路径存在系统性的重复解码、碎片化分配和整计划深拷贝

位置：`crates/tencir-pauli-core/src/sector.rs:18-29,178-205,228-230,343-376,417-455`，`crates/tencir-pauli-core/src/symmetry.rs:361-390,423-506,655-768`。U1 transition table 使用 `Vec<Vec<(usize, Complex64)>>`，为每个 source basis state 新建一个预留到 term count 的 `FxHashMap`，并对每个 `(state, term)` 重新遍历全部 qubits 生成 x/z masks；随后 `mvp_plan()` 深拷贝整个 rows table。apply 使用 source-major scatter，无法直接 row-parallel，CSR/COO 又重新转置并复制全部 entries。Z2 transform 对每个 gate/term 反复 materialize codes 或 clone packed words，在每个 term 内还复制 `removed_qubits` 并执行线性 `contains`；GF(2) elimination 在 inner loop 中重复 clone pivot rows。特别是 `symmetry.rs:476-477` 的 CNOT 明确为 x/z 各执行一次 `to_vec()`，与 `docs/vibe/implementation-status.md:91` 所称的 “allocation-free CNOT updates” 不一致。

这些实现与项目要求的 compact contiguous storage、precomputed masks、reusable scratch、避免 per-term/per-state allocation 和 data-layout-first optimization 不一致。当前 k=2 微小 reduced space 的好结果不能证明 central sectors 或更高 generator rank 的 scaling。

### M4. 正确性测试没有建立 Phase II 要求的独立 sector reference/property gate

位置：`tests/test_symmetry.py:23-53,56-113`，`crates/tencir-pauli-core/src/tests.rs:235-304`。唯一 Z2 谱测试是 `XX + ZZ + I` 的 2-qubit 完全可对易特例，只把所有 tapered sector eigenvalues 合并后与 full spectrum 比较；它没有逐 sector 构造 projector，也没有验证每个 original generator 在所选 sector 中变成指定标量。因此 C1 即使把 sector 标签/符号交换也不会被发现。U1 测试主要是 3-qubit real `XX+YY`，没有 tracked randomized/property coverage、complex/Y phase cases、multiword boundary 或 budget peak cases。

这不满足 `AGENTS.md` 的 property/differential-test 要求及 `docs/vibe/phase-2-spec.md:210-228,242-246` 的 independent GF(2)/dense-sector reference 验收要求。手工“随机 smoke”写在状态文档中不能替代可重复的 regression tests。

### M5. 性能验收证据不完整，部分 JAX setup timing 未同步

位置：`docs/vibe/benchmarking.md:7-26`，`crates/tencir-pauli-core/Cargo.toml:19-24`，`benchmarks/python/test_symmetry_benchmark.py:38-128`，`benchmarks/python/test_symmetry_jax_benchmark.py:103-121,175-190`，`docs/vibe/implementation-status.md:89-92`。项目规范要求每个 material hot path 同时有 Rust kernel 与 Python/FFI benchmark，并报告 peak memory、allocation/scaling 和保存的同机 release 结果；当前 Rust Criterion 仍只注册 `pauli_word`，Phase II benchmark 只在 Python 层，源码不测 peak memory/allocations/固定线程 scaling，U1 大 workload 只有 `k=2`。当前 `.benchmarks/` 中也没有 Phase II record/manifest；最新保存结果仍对应 Phase I 基线 `96fc171`，所以状态文档中的 Phase II 数字无法按 `benchmarks/run.py:158-193` 的记录模型复核。

此外，JAX restriction setup benchmark 计时 `make_jax_u1_baseline()`，其中 `jnp.asarray()` 创建 device arrays 后直接返回 jitted callable，没有在 timed callable 内 `block_until_ready()`；这违反 `docs/vibe/benchmarking.md:26`，setup 数字可能只包含异步 enqueue。steady/end-to-end apply 已同步，此问题主要影响 setup 对比。

### M6. [SCOPED OUT] 公开 tapering plan 不提供稳定持久化

位置：`docs/vibe/phase-2-spec.md:107-110`，`crates/tencirpauli-native/src/symmetry.rs:58-69`，`python/tencirpauli/symmetry.py:40-86`。原 spec 曾要求 versioned/serializable plan，而当前对象只支持进程内 native reuse，默认 pickle 会失败。Owner 的实际需求是同一 Python 进程内调用 sparse matrices、MVP plans 和 functions，不需要把 tapering plan 写入磁盘、跨进程发送或跨库版本恢复，因此稳定 serialization 已正式 de-scope；当前只需保证 runtime plan 能反复 transform compatible operators。

## MINOR

### m1. Null-space conversion 静默吞掉理论上不应发生的错误

位置：`crates/tencir-pauli-core/src/symmetry.rs:90-93`。`filter_map(|bits| bits_to_word(...).ok())` 会丢弃 conversion error，而项目要求 unsupported/invalid state fail fast。这里应 `collect::<Result<Vec<_>, _>>()?`，使 invariant 破坏可见。

### m2. 完成状态与后续动作自相矛盾

位置：`docs/vibe/implementation-status.md:48-52,89-99`。文档一方面声明 Phase II 已完成并列出 benchmark coverage，另一方面 next action 仍是“Add Phase 2 release-quality benchmark workloads”；在 C1、M3-M5 修复和证据记录完成前，状态应改为 remediation/review failed，而不是 completed。

### m3. `apply_into` 的长度错误会丢失具体失败参数

位置：`crates/tencir-pauli-core/src/sector.rs:343-353`。当 state/output 任一长度错误时，`actual` 使用两者的 `max`，可能报告一个恰好正确的长度，降低诊断质量。应分别验证并报告 state 或 output 的实际长度。

## OBSERVATIONS

- 全套常规工具链检查通过，说明问题不是格式、lint、typing 或构建失败，而是现有测试没有覆盖到语义和资源合同。
- 额外执行的 4,000 组随机 U1 小系统差分均通过，覆盖 complex coefficients、Y phases、接受/拒绝 leakage、dense projection 与 MVP；当前 U1 的小系统数学 action 看起来可靠，主要风险集中在支持范围、memory contract 和 scaling。
- Phase II benchmark smoke harness 可运行：7 passed、12 个 `performance_large` cases deselected。
- 首轮审查没有修改实现；owner decision 更新仅修改 `AGENTS.md`、README 和 `docs/vibe/*.md`/本报告，没有修改 `.rs`、`.py`、`.toml`、测试或基准源码。

## RECOMMENDED IMPROVEMENTS

| 对应问题 | 明确下一步 | 完成判据 |
|---|---|---|
| C1 | 统一 row sign 的单一存储源；row elimination 合并前一行时传播真实 `row_signs[previous]`，并审计 CNOT/H/S/Sdg 的 signed conjugation。 | 上述 `ZYY + 2·YIZ` regression 在每个 sector 与 dense projector 完全一致；n≤3 全部 independent commuting generator sets 的 generator-eigenvalue exhaustive test 通过；随机 n=4 projector/action tests 通过。 |
| M1 | 不实现 exact peak accounting；保留 checked dimensions/overflow 和 cheap major-allocation guards，并在 API 文档明确其 best-effort 性质。 | 文档不再声称 hard peak limit；低预算仍可拒绝明显超大的主要输出；代码不为 exact accounting 引入复杂 FFI/allocator 模型。 |
| M2 | 当前入口明确拒绝 `nqubits >= usize::BITS` 的 native restriction并说明限制；Phase 5 实现 packed multiword transitions/rank/lookup/MVP/CSR。 | 当前宽度错误清楚且在大分配前失败；Phase 5 保留 64/65/128-qubit low-k correctness/performance acceptance。 |
| M3 | 预编译每个 Pauli term 的 packed x/z mask 与 fixed phase；使用 flat CSR/CSC/SoA storage、可复用聚合 scratch 和 destination-major gather；让 restricted operator 与 MVP plan 共享 immutable storage；对 Z2 使用 packed in-place gate kernels 和预计算 removal map。 | release profiling 证明主要 allocation/decoder bottleneck 消失；n/k/term-count scaling、peak RSS 与 throughput benchmark 保留；结果继续通过 differential/property gate。 |
| M4 | 把本报告的反例加入 Rust/Python regression，并增加独立 dense Clifford/projector reference、generator independence/rank properties、complex/Y U1、current-width boundary、dimension/overflow 和 cheap-guard properties。 | C1 类 sector-label/sign mutation 会稳定使测试失败；测试不依赖被测 taper implementation 来生成 expected sector result。 |
| M5 | 新增 symmetry/sector Rust Criterion benches；Python benchmarks 增加 central-sector、scaling、peak/output memory和固定线程配置；同步所有 JAX timed outputs/device arrays；通过 runner 保存 Phase II label/manifest。 | `.benchmarks/runs/<phase2-label>.json` 为 complete，Rust/Python结果均可 compare，文档每个数字都能映射到保存 case 和配置。 |
| M6 | 不实现稳定 plan persistence；删除 versioned/serializable 完成声明，只保留 process-local reusable plan。 | README/spec/status 与 runtime capability 一致，不再把 pickle/JSON round-trip 当作 Phase II 验收项。 |

## IMPLEMENTATION HANDOFF

### 修复难度与改动范围

| 问题 | 难度/范围 | 是否需要 owner 决策 | 可否由另一模型直接执行 |
|---|---|---|---|
| C1 row-sign bug | 小改动、高语义风险；core 局部重构加穷举测试 | 否 | 可以；必须先写 failing regression，再改实现。 |
| M4 correctness gate | 中等；主要新增独立 reference/property tests | 否 | 可以；测试合同已在下文冻结。 |
| M1 memory contract | 已按 owner 决策降为 best-effort scope；只需维护 cheap guards/文档 | 否，D1 已决 | 不需要 exact peak-accounting patch。 |
| M2 64+ qubit U1 | 当前限制是小型文档/error patch；Phase 5 multiword engine 是大项目 | 否，D2 已决 | 当前只需清楚 fail fast；完整实现排入 Phase 5。 |
| M3 U1/Z2 性能结构 | 中到大；可分“必要去重”与“flat transition engine”两层 | 暂不需要，先 profile | 不要和 C1 同一 patch；用 central-sector profile 决定范围。 |
| M5 benchmark/evidence | 中等；依赖 C1、M1-M3 稳定后执行 | 否 | 可以，但必须最后做。 |
| M6 plan serialization | 已正式 de-scope | 否，D4 已决 | 不实现 schema；保持 runtime reuse。 |
| m1-m3 | 小 | 否 | 可以随对应模块 patch 修复。 |

### C1 的精确根因与修复形状

每个 elimination row 的真实代数对象是 `s_i P_i`，其中 `P_i` 是 phase-free `PauliWord`，`s_i ∈ {+1,-1}`。执行 Clifford gate 后，`P_i` 可能得到一个额外符号，因此代码用 `row_signs[i]` 跟踪 `s_i`。清除前一 pivot 时执行 row multiplication：

```text
(s_i P_i) <- (s_i P_i)(s_j P_j)
new_sign = s_i * s_j * phase(P_i P_j)
new_word = phase_free_word(P_i P_j)
```

当前 `Z2TaperingPlan::new()` 同时维护了两个 sign 容器：`rows: Vec<(PauliWord, i8)>` 的 tuple sign 和单独的 `row_signs: Vec<i8>`。tuple sign 在 `symmetry.rs:186-190` 全部初始化为 `+1`，而 `apply_gate()` 在 `symmetry.rs:394-405` 只更新 `row_signs`，从不更新 tuple sign。随后 row multiplication 在 `symmetry.rs:212` 使用 `rows[previous].1`，所以公式中的 `s_j` 永远被当成 `+1`。这就是只有部分多生成元/Clifford 路径出错、简单 `XX/ZZ` 用例却通过的原因。

推荐不要只把一行机械改成 `row_signs[previous]` 后结束，而是消除双重状态，避免以后再次分叉：

```rust
let mut rows = generators.to_vec();
let mut row_signs = vec![1_i8; generator_count];

let (word, phase) = rows[row_index].multiply(&rows[previous])?;
rows[row_index] = word;
let previous_sign = row_signs[previous];
row_signs[row_index] *= previous_sign;
row_signs[row_index] *= phase_sign(phase)?;
```

相应地把 `apply_gate(rows: &mut [(PauliWord, i8)], ...)` 改成只接收 `&mut [PauliWord]` 与唯一的 `row_signs`。这属于局部重构，不改变 public API。修复前先提交 `ZYY + 2·YIZ` regression；修复后增加两个互补 gate：其一，n≤3 穷举所有 independent commuting generator bases 和所有 sectors，验证每个 original generator 被变成 `sector[i] * I`；其二，对自动 analysis 结果逐 sector 构造独立 dense projector `P_s = ∏_i (I+s_iS_i)/2`，比较 `P_s H P_s` 与 tapered operator 的 action/spectrum，而不是只比较所有 sector 谱的并集。

### 建议实施顺序

1. **Correctness patch**：只处理 C1、m1 和新增 Z2 exhaustive/projector tests；先让反例失败，再修复，运行全套检查。不要在此 patch 混入性能重构。
2. **U1 correctness/boundary patch**：补 complex/Y randomized properties、current-width boundary、dimension/overflow tests，并对 64+ native restriction给出清楚 unsupported error；完整 multiword engine由 Phase 5 实施。
3. **Plan/data-layout patch**：按尚未决定的 D3 预编译 term masks、消除 plan 深拷贝，并决定是否直接落地 flat destination-major storage；重新跑 correctness gate 后再 profile。
4. **Benchmark and documentation patch**：最后补 Rust/Python benchmarks、storage/scaling、同步 JAX setup、保存 Phase II record，然后才把 status 改回 completed。`max_bytes` exact accounting和 stable plan serialization 不再创建实现 patch。

### OWNER DECISIONS / TRADEOFFS

#### D1. `max_bytes` 到底约束什么 — OWNER SELECTED BEST-EFFORT

决定：保留当前 cheap estimates 作为 best-effort guard；不计算完整 public-call peak，不为 Python contiguous copy、PyO3 conversion、allocator overhead 或全部 transient scratch 建复杂模型，OOM 视为可接受的系统级失败。仍必须 checked dimension/arithmetic overflow，并应保留对明显超大 dense/sparse output 的廉价拒绝。README/spec/AGENTS 已同步，M1 不再是 Phase II blocker。

#### D2. Phase II 是否承诺 64+ qubit U1 restriction — OWNER SELECTED CURRENT CAP + PHASE 5

决定：当前 `U1Sector` combinatorial/basis helper 可以继续处理 Python wide integers/multiword basis，但 `restrict_u1()`、restricted MVP/CSR 只承诺 `nqubits < usize::BITS` 并清楚 fail fast。64+ qubit low-particle-number U1 Hamiltonian 的 packed XOR/popcount、multiword rank/lookup、transition storage、MVP/CSR 和 benchmarks 已排入 architecture roadmap Phase 5；TensorCircuit-style U1 circuit/time evolution 排入 Phase 6。

#### D3. 性能重构做到哪一层

- **选项 A（最低可接受）**：在 restriction setup 前一次性把每个 Pauli term 编译成 `x_mask/z_mask/weighted_phase`，复用 aggregation scratch，并用 `Arc` 让 `U1RestrictedOperator` 与 `U1MvpPlan` 共享同一 transition data；暂时保留 `Vec<Vec<_>>`。改动中等，可立即消除最明显的 `O(D*T*n)` 重复解码和 plan 深拷贝。
- **选项 B（推荐）**：同时改为 flat destination-major CSR-like storage（`indptr`, `source_indices`, `values`），restricted apply 变为按 destination gather，可安全 Rayon parallel；`.csr()` 可直接复制/导出 canonical arrays，不再构造 `by_destination` scratch，M1 与 M3 一次解决。改动较大，但避免先修一套 source-major representation 后再次迁移。

当前不要求 owner 预先选择 A/B。先修 C1 并记录 central-sector setup/apply/CSR profile：如果主要成本是重复 term decode 和 plan clone，做 A 后复测；只有 source-major scatter、CSR transpose 或 row fragmentation 仍是主瓶颈时再做 B。这样既不放弃性能目标，也避免在没有 profile 证据时启动最大的数据布局重写。

#### D4. 是否现在承诺稳定 plan serialization — OWNER SELECTED DE-SCOPE

决定：当前主要需求是在 Python 进程内调用 sparse matrix、MVP plan 和函数，不需要把 tapering plan 存到磁盘、发送到另一进程/机器或在升级库版本后恢复，因此不定义 schema/version，也不实现 pickle/JSON round-trip。runtime plan 仍必须可在同一进程反复应用于 compatible operators；如果未来出现 checkpoint/distributed/cache 用例，再单独设计持久化合同。
