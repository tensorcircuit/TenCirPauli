# Phase 1 实现规格

状态：可执行。`semantics.md` 中 S1–S4 已冻结。目标范围为架构路线中的阶段零与阶段一。

## 1. 目标与完成定义

Phase 1 交付一个可独立安装、可从 Rust 和 Python 使用、具有明确 TensorCircuit 兼容语义的 Pauli algebra、measurement grouping 和 Hamiltonian engine。正确性是硬门槛；在正确性覆盖之后，每个核心热路径必须具有本地 benchmark，但架构文档中的性能倍数是 go/no-go 评估指标，不是让 Agent 无限优化的完成条件。

Phase 1 只有在所有 REQUIRED checklist、测试、文档和 benchmark workload 完成后才能声明完成。性能未达到目标时应记录 profile、瓶颈和实测差距，不得用占位实现宣称完成，也不得因未达到 aspirational speedup 永久停留在同一优化循环。

## 2. Source of truth

实现时的规范优先级为：`AGENTS.md` > 已冻结的 `semantics.md` > 本文 > tests/reference vectors > `architecture.md` > 当前 bootstrap 实现。发现会改变公开语义的冲突时，完成不受影响的工作并记录 blocker，不能自行选择一种新语义。

## 3. Phase 1 非目标

- 不实现 Z2/U(1) symmetry、tapering 或 restricted sector；这些属于 Phase 2。
- 不实现 GateTape、Clifford/dynamic/weight-truncated propagation；这些属于 Phase 3。
- 不实现 native gradients、checkpointing 或 JAX custom call；这些属于 Phase 4 或独立项目。
- 不实现 fixed-buffer、top-k 或 TensorCircuit `SparsePauliPropagationEngine` 的 Rust 版本。
- 不发布 PyPI、不修改 TensorCircuit 主仓库，也不让 TensorCircuit 基础安装依赖 TenCirPauli。
- 不为尚无 workload 的抽象提前拆分更多 crate。

## 4. 里程碑 P0：语义冻结与 reference

### REQUIRED deliverables

- [x] Owner 确认 S1–S4，`semantics.md` 状态改为“已冻结”。
- [x] 建立独立的小系统 NumPy dense reference，覆盖 code/string/symplectic/matrix 互转，不调用被测 Rust 实现生成 expected result。
- [x] 建立固定 regression vectors：单比特 multiplication table、首尾 qubit ordering、Y phase、duplicate cancellation、empty/identity operator 和无效输入。
- [x] 为随机测试固定 seed，并记录 tolerance、dtype 与最大 reference qubit 数。
- [x] 创建 scaffold 初始 Git commit；之后 benchmark label 必须包含真实 commit id。

### Acceptance gate

所有 reference vectors 在文档中有明确来源，现有 bootstrap `PauliWord` 行为与冻结语义一致或已迁移。Phase 1 的后续代码不得绕过这些 reference。

## 5. 里程碑 P1：PauliWord 与批量结构转换

### REQUIRED deliverables

- [x] Rust `PauliWord` 完成 construction、code/string conversion、weight、support、commutation、symplectic inner product、multiplication、adjoint 和 stable ordering。
- [x] 批量结构转换在一次 Rust 调用中处理二维 code arrays 或 packed buffers；禁止逐 term PyO3 调用成为推荐路径。
- [x] Python `PauliWord` 提供稳定、typed、documented public API，`_native` 仍为 private implementation detail。
- [x] Typed Rust errors 与 Python exception mapping 覆盖 shape、code、nqubits、overflow 和 incompatible operand。
- [x] Rust property tests、Python tests 和 NumPy dense differential tests全部通过。
- [x] Criterion 与 pytest-benchmark 覆盖单项 kernel、batch conversion 和完整 Python boundary。

### Acceptance gate

对随机 `n<=6` Pauli words，所有代数结果与 dense reference 一致；公开输出在重复运行中 byte-for-byte deterministic。Batch API 相比逐项 Python 调用有可观察的端到端优势，实际结果记录但不设置硬倍数门槛。

## 6. 里程碑 P2：PauliOperator 与 canonicalization

### REQUIRED deliverables

- [x] Rust `PauliOperator` 实现 canonical terms、coefficient storage、add、scale、multiply、commutator、anticommutator、adjoint 和 Hermiticity validation。
- [x] Batch canonicalization 返回 canonical keys、aggregated coefficients、`input_to_canonical` 和 phase multipliers。
- [x] Native static operator 与 backend structural plan 分离 exact-zero 和 parameter-dependent coefficient 语义。
- [x] Python 支持从 code arrays、strings 和 packed arrays 构造 operator，并返回稳定排序的 terms。
- [x] 重复项、相消项、complex phase、NaN/Inf、empty operator 和超大输入预估均有测试。
- [x] Benchmark 覆盖 `10^3`、`10^4`、`10^5` term 的 parse/canonicalize/deduplicate，并记录输入转换、kernel 与输出转换。

### Acceptance gate

随机小系统 operator algebra 与 NumPy dense matrix 一致；canonicalization 与输入顺序无关；重复运行返回相同 key order、mapping 和 coefficients。

## 7. 里程碑 P3：Commutation 与 measurement grouping

### REQUIRED deliverables

- [x] 批量 commutation/QWC compatibility kernel 支持 packed input，并明确 full matrix 与 streaming/edge-list 模式的内存成本。
- [x] 实现确定性的 largest-first greedy；DSATUR 在 largest-first 正确性稳定后实现。
- [x] QWC groups 返回逐 qubit basis、term membership、coefficient mapping 和 bitstring eigenvalue reconstruction masks。
- [x] General commuting 按冻结的 S3 决策实现，类型和文档不能把 algebraic partition 与 measurement-ready plan 混淆。
- [x] 图 coloring 正确性、deterministic tie-break、identity term、duplicate term 和 adversarial graph 均有测试。
- [x] Benchmark 报告 term 数、qubit 数、edge density、group count、basis depth、运行时间和峰值内存。

### Acceptance gate

每个返回 group 满足所声明 compatibility；QWC reconstruction 在随机小系统 shot bitstrings 上与逐 term expectation reconstruction 一致；同一输入在不同 hash seed/thread count 下保持公开结果稳定。

## 8. 里程碑 P4：Hamiltonian compiler

### REQUIRED deliverables

- [x] Dense target 与 NumPy Kronecker reference 一致，并具有严格的小系统/memory guard。
- [x] COO 直接生成、聚合重复 entry、按 `(row,column)` 排序；CSR 由确定性聚合结果构造。
- [x] Native MVP 接受一维 complex state，验证 shape/dtype/overflow，不物化完整 matrix。
- [x] Backend MVP plan 按冻结的 S4 范围实现版本化 arrays schema 和 reference executor。
- [x] 所有物化 target 在分配前估算 dimension、nonzeros 与 bytes，并通过明确的 memory limit fail fast。
- [x] Rust/Python differential tests 比较 dense、COO、CSR、native MVP、backend plan 和 Hermiticity。
- [x] Benchmark 分开记录 plan construction、first apply、steady apply、input/output conversion 和 peak memory。

### Acceptance gate

对随机 `n<=10` operator，全部 target 在冻结 tolerance 下逐元素或逐向量一致；超限请求在分配前失败；大系统 API 引导使用 MVP 而不是尝试危险分配。

## 9. 里程碑 P5：Public API、TensorCircuit adapter 与交付

### REQUIRED deliverables

- [x] `tencirpauli` 顶层只导出稳定 public classes/functions；所有 `_native` symbol 均通过 typed Python facade 使用。
- [x] PyO3 热路径为 batch API，长计算释放 GIL；不存在 Python callback 进入 Rust hot loop。
- [x] Optional TensorCircuit adapter 按 S4 范围完成 ordering、dtype、MVP 和缺失依赖行为测试。
- [x] README、Python docstrings、typing stub、examples、CHANGELOG 和 `docs/vibe/implementation-status.md` 同步。
- [x] Linux/macOS/Windows CI 完成 Rust/Python correctness 与 packaging smoke；性能仍只在本机记录，不增加 CI 门禁。
- [x] 本机保存一个包含所有 Phase 1 workload 的最终 benchmark label，并记录关键瓶颈与 go/no-go 数据。

### Acceptance gate

新环境可以通过 wheel 或 sdist 安装并运行 public example；所有 quality/test commands 通过；没有 placeholder、未解释 TODO、静默 fallback 或未记录的语义偏差。

## 10. Public API 最小形状

Phase 1 public Python surface 至少包含 `PauliWord`、`PauliOperator`、明确区分 QWC/general 的 grouping result，以及 Hamiltonian dense/sparse/native/backend-plan 构造入口。具体函数签名允许在实现中简化，但必须满足：typed、batch-first、deterministic、无 `_native` 泄漏、错误可预测、结构与动态 coefficient 可分离。

Rust core public surface至少提供 phase-free/phaseful 选择所需的代数结果类型、canonical key/operator、grouping result 和 Hamiltonian plan。不要把 Python/NumPy/TensorCircuit object 放入 core type。

## 11. 每个里程碑的执行循环

1. 阅读 status 与本里程碑规范，确认没有 owner blocker。
2. 先建立或扩充独立 reference 与失败测试。
3. 实现最小纵向切片：Rust core → batch PyO3 → Python facade。
4. 运行 format/lint/type/test gates，修复真实问题而不是 suppress。
5. 为热路径增加或保持稳定 benchmark，记录当前 label并与最近同机 baseline 手动比较。
6. 更新 `implementation-status.md` 的证据、限制、下一步；如已授权，在里程碑边界创建本地 commit。
7. 立即进入下一个未完成 REQUIRED item，直到 Phase 1 completion checklist 全部满足。

## 12. Phase 1 completion checklist

- [x] P0–P5 所有 REQUIRED 项和 acceptance gate 完成。
- [x] `cargo fmt --check` 通过。
- [x] `cargo clippy --workspace --all-targets --all-features -- -D warnings` 通过。
- [x] `cargo test --workspace` 通过。
- [x] Black、Ruff、mypy 通过。
- [x] `maturin develop --release` 与 pytest 通过。
- [x] Phase 1 本地 benchmark 已记录并人工检查，无未解释的重大回归。
- [x] Public docs、typing、CHANGELOG、status 与实现一致。
- [x] 没有 Phase 2–4 的越界实现或对 TensorCircuit 主仓库的未授权修改。
