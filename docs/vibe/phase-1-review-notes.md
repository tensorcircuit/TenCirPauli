# Phase 1 审核意见与改进候选

状态：持续审核记录（2026-08-01）。本文是对当前实现和要求的审查记录，不是新的语义合同，也不改变已冻结的 S1–S4。表中已解决项目保留原始问题和解决证据，供 owner 复核。

## 1. 审查范围

审查对象是基于 `0ae1973` 演进到 `0a546a6` 的当前本地 Phase 1 实现、`phase-1-spec.md`、`implementation-status.md`、release-mode Rust benchmark、public Python boundary benchmark，以及同机 TensorCircuit/JAX 对照。性能代码和文档已提交到本地 focused commits；TensorCircuit 仓库只读、未修改；TenCirPauli 仓库未配置 Git remote，也没有执行 push。

## 2. 已确认的问题

| 优先级 | 问题 | 证据 | 建议 |
| --- | --- | --- | --- |
| P0 | 旧记录中 `phase-1-spec.md` 的 P0–P5 checklist 曾与 status 文档漂移。 | 当前 checklist 已同步为 `[x]`，status 仍保留历史证据和最终 clean-label 收尾状态。 | 已解决；后续只允许在统一检查和 benchmark 记录后更新 completion record。 |
| P0 | 规格要求的 batch canonicalization `input_to_canonical` 与 phase multipliers 在旧 public/native API 中没有对应输出。 | 旧 `_native.pyi` 的 `pauli_canonicalize` 只返回 canonical structures 和 real/imag coefficients。 | 已解决：新增 `_native.pauli_canonicalize_batch` 与 typed `PauliOperator.canonicalize_batch()`；dynamic result 保留 exact-zero keys，返回 mapping 和 exact `PauliPhase` multipliers；static constructor 保留 fast path。 |
| P0 | “Rust MVP 性能”存在 kernel、PyO3 boundary、public Python API 三种口径，当前文档容易把它们混为一谈。 | Criterion 的 `19.905 µs` 是小型 Rust core workload；public `PauliOperator.mvp` 每次都会重新组装输入并转换 state。 | 所有 benchmark 名称显式标注 `rust-core`、`native-ffi`、`python-public` 或 `backend-jax`，并在表格中分别报告。 |
| P0 | 当前 public native MVP 的默认一次性调用仍会重新编译 operator；此前缺少可复用 plan 的问题已修复。 | 新增 `PauliOperator.native_mvp_plan()`、NumPy zero-copy PyO3 input/output 和 reusable Rust plan；一次性 `PauliOperator.mvp()` 仍保留直接路径。 | 后续 benchmark 必须区分一次性 native MVP 和重复 apply 的 reusable plan；重复 workload 应使用 reusable plan。 |
| P1 | Rust MVP inner loop 的临时 `codes()` 扫描问题已修复，但仍需持续 profile。 | 当前 term 只预计算 packed X/Z masks 和 Y phase；reusable plan 按 X mask 预计算 diagonal。 | 已解决当前 slice：NumPy boundary zero-copy complex buffers、Rust output direct-fill、Rayon row-parallel apply；`/usr/bin/sample` 已确认热路径为 `MvpPlan::apply_into`/Rayon。只有 profile 证明必要时再评估更底层 SIMD。 |
| P1 | `BackendMVPPlan` 的 NumPy executor 和 TensorCircuit adapter 仍在 Python 中逐 term 组织 mask、flip 和累加。 | `hamiltonian.py` 和 `integrations/tensorcircuit.py` 都有 term/qubit Python loops；同机 JAX warm 性能没有明显优于 TensorCircuit 原生 MVP。 | 将它定位为 portable reference/plan adapter；若要成为性能路径，应批量化 masks/indices，并单独 benchmark setup、compile、warm apply。 |
| P1 | sparse matrix 的跨实现 benchmark 曾尚未进入最终持久化性能证据。 | 当前实现已按 X mask 分组后生成 contiguous candidate entries；同 workload 的 release 补测已明显快于 TensorCircuit NumPy 和 JAX raw construction。 | 已解决：clean label `20260801T101504Z_dc089491dd43` 包含 TenCirPauli public COO、TensorCircuit NumPy COO construction、JAX BCOO first/warm construction、first/warm `sum_duplicates()` 和 warm matvec。 |
| P1 | 当前 benchmark workload 与 JAX 对照 workload 不完全一致。 | Phase 1 Rust Hamiltonian benchmark 使用 duplicate-heavy 小系统；JAX 对照使用 10/16 qubits 的 unique terms。 | 保留历史 benchmark，同时新增一套同结构、同 canonical term count、同 dtype 的 cross-implementation workload。 |
| P1 | 性能原则要求 profiling，但当前交付记录主要是 timing，没有 allocation/peak-memory/profile evidence。 | `implementation-status.md` 记录 Criterion/pytest-benchmark 数字，但没有 profiler 或 allocation breakdown。 | 对 native MVP、COO/CSR 和 grouping 至少保存一次本机 profile 摘要；不要把 profile 文件提交进仓库，只提交 workload、命令和结论。 |
| P2 | TensorCircuit adapter 的 setup 成本可能抵消 backend plan 的收益。 | 同机 complex128 测量中，10/16 qubits 的 TenCirPauli adapter setup 约 35/62 ms；TensorCircuit 原生 MVP setup 约 1/4 ms。 | 若 adapter 保留，预构造并缓存 backend-friendly masks；或者明确它只保证语义/plan reuse，不承诺一次性 setup 加速。 |
| P2 | completion record 的依赖环境叙述不够统一。 | status 同时记录“默认环境 41 passed, 2 skipped（未安装 TC/JAX）”和“只读源码环境 42 passed, 1 skipped”。 | 将 optional dependency matrix 单独列出 backend、版本、命令和 skip 原因，避免读者误解为同一次环境运行。 |

## 3. 当前性能判断

同机 Darwin arm64、Python 3.11、JAX 0.10.2、complex128、相同 unique Pauli workload 的 warm 对照显示：10 qubits/64 terms 时 TenCirPauli reusable native plan 约 `0.008 ms`，TensorCircuit 原生 MVP+JAX 约 `0.032 ms`；16 qubits/256 terms 时约 `0.408 ms` 对 `2.4 ms`。当前 release Rust plan construction 约为 `0.034/1.64 ms`，TensorCircuit JAX 首次编译调用约为 `100/585 ms`。这个结果来自按 X mask 分组、预计算 diagonal 的 reusable native plan；一次性 `PauliOperator.mvp()` 和 reusable apply 不能混为一个数字。

TenCirPauli backend plan 加 JAX 仍然是另一条 backend/AD 路径；native reusable plan 的优势来自 Rust CPU 预计算，不应与 backend plan 的 JAX warm 数字混淆。backend plan 继续用于需要 TensorCircuit backend、JAX JIT 或 AD 的工作流。

本次 sparse 补测使用相同的 unique terms、complex128 和 CPU；结果来自 release-mode `pytest-benchmark` clean label `20260801T101504Z_dc089491dd43`：

| workload | TenCirPauli COO | TensorCircuit NumPy COO | JAX BCOO first / warm construction | JAX first / warm `sum_duplicates()` |
| --- | ---: | ---: | ---: | ---: |
| 8q / 32 terms | 0.057 ms | 3.51 ms | 219 ms / 0.794 ms | 453 ms / 1.668 ms |
| 10q / 64 terms | 0.200 ms | 9.20 ms | 165 ms / 1.337 ms | 427 ms / 4.981 ms |
| 12q / 64 terms | 0.715 ms | 20.91 ms | 160 ms / 1.457 ms | 450 ms / 20.739 ms |

TenCirPauli COO 相对 JAX raw warm construction 约快 `14.0x/6.7x/2.0x`，相对 JAX warm duplicate canonicalization 约快 `29x/25x/29x`；在 `n=8` 上跨实现 dense reconstruction 的最大误差小于 `5e-15`。

Sparse 输出不能只按 construction 时间比较：TensorCircuit JAX BCOO 的 raw `nse` 为 `8192/65536/262144`，即 `terms * 2**n`，且 `unique_indices=False`；调用 `sum_duplicates()` 后它变成 `unique_indices=True` 的 padded BCOO，`nse` 为 `2048/8192/32768`，exact nonzero data count 为 `1984/7680/30720`，而 TenCirPauli canonical COO exact nnz 为 `1984/7296/29184`。JAX 的 `sum_duplicates()` 可能因浮点重复求和留下约 `1e-16` 的 cancellation residual，因此“精确非零数”和阈值后的数学 support 也要分开记录；JAX raw 和 padded canonical storage 仍都大于 TenCirPauli 的 exact canonical COO storage。

重复 entry 并不会让 JAX BCOO 的基本矩阵乘法失效：当前 n=8 对照中 `unique_indices=False`、`nse=8192`，直接执行 `sparse @ state` 与 `sparse.todense() @ state` 的最大误差仍约为 `7.3e-15`。但 raw BCOO 不是 canonical COO；如果后续算子要求 unique/sorted indices、稳定 nnz、低内存或 exact aggregation，就必须在 plan 阶段生成 canonical structure，或者显式承担 JAX `sum_duplicates()` 的首次编译和 warm canonicalization 成本。

## 3.1 Rust sparse/MVP 性能的根因

旧 Rust COO 的核心循环是 `term -> column -> BTreeMap<(row,column), value>`。该设计虽正确，却有树查找、比较和节点分配开销。当前实现已改为按 MSB X mask 分组，在 contiguous candidate buffer 中聚合同一 permutation，再做 exact-zero filter 和一次稳定 row-major sort；CSR 直接复用 sorted entries，不先物化 public COO arrays。TensorCircuit 的 NumPy 路径则先用向量化 bit operations 生成整列 indices/values，再交给 SciPy 的 C 实现合并；JAX 路径进一步用 `vmap`/XLA 执行，所以不能用“Rust 语言”本身解释差距。

旧 Rust `matrix_x_mask` 和 `matrix_phase` 会反复调用 `word.codes()`；当前 term 已预计算 packed X/Z masks 和 Y phase，reusable plan 再按 X mask 预计算 diagonal。当前 Python native API 的一次性路径仍会重新 `build_operator`，这是明确的 setup 成本；重复 state-vector workload 应使用 reusable plan。NumPy complex128 state/output 已在 PyO3 boundary 直接映射和填充，消除了旧的 Python tuple/实部/虚部复制。这些是实现策略，不是算法下界。

下一步只在新的 workload/profile 证明必要时评估 SIMD 或更专门的 sparse kernel；当前 Phase 1 已经覆盖 mask/phase precompute、X-mask grouped COO、direct CSR、reusable plan、zero-copy NumPy boundary 和 Rayon row parallel。任何更深层优化都必须保留现有 dense differential、deterministic ordering 和 memory guard。

## 4. 建议 owner 审核的决策

1. `input_to_canonical`、phase multipliers 和动态 coefficient reduction 已按 Phase 1 REQUIRED 补齐；code-array 输入的 phase multipliers 明确全部为 exact `+1`，后续 phaseful input 不能复用该 API 而静默改变语义。

2. reusable native MVP plan/apply 已作为 Phase 1 性能修正固化在本地 commits `ff02ae8`/`4b10598`，clean benchmark label 为 `20260801T093405Z_0a546a696d38`。

3. 是否接受 Rust native MVP 只作为当前正确性实现，暂不承诺超过 JAX warm kernel？当前固定 workload 的 reusable native plan 已超过 TensorCircuit JAX warm；仍需在随机高 X-mask cardinality workload 上验证 memory fallback 和 scaling，不能扩大结论到所有 Hamiltonian。

4. COO/CSR 与 TensorCircuit NumPy/JAX sparse 对照已加入 benchmark harness 并进入 clean label；JAX BCOO 的 raw `nse`、padded canonical `nse`、exact data count、thresholded support 和 storage bytes 必须继续分开报告。

5. `phase-1-spec.md` checklist 已与实现同步为 `[x]`；最终 completion record 仍以 unified check、clean benchmark label 和本地 commits 为准。

## 5. 明确不建议现在做的事

- 不因为 MVP benchmark 不理想而引入新的 crate、GPU kernel、JAX custom call 或 speculative abstraction。
- 不用 fixed-buffer top-k propagation 替换当前 Rust native MVP；那属于明确排除的 Phase 3 语义。
- 不把 TensorCircuit 的 `PauliPropagationEngine` 或 `SparsePauliPropagationEngine` 伪装成 TenCirPauli 已实现功能。
- 不把 JAX warm benchmark 与 Rust cold benchmark，或 Rust core kernel 与 Python boundary，直接做未经说明的倍数比较。

## 6. 审核后的执行顺序

执行记录：canonicalization mapping 语义已明确并实现；native reusable MVP plan 已公开策略；native MVP 已完成 release/profile 优化；COO/CSR/JAX sparse cross-implementation benchmark 已加入并写入 clean status label，且已补齐 JAX BCOO construction 与 duplicate canonicalization 的 cold/warm 对照。Phase 1 closeout 已完成。
