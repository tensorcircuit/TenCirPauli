# Phase 4 实现与性能验收 Review（2026-08-02）

状态：**主体实现完成，但暂不建议按 `phase-4-spec.md` 宣告完整验收通过。** Deterministic frozen-support reverse 的实现、公开 API 和代表性性能基本达到阶段目标；SPPS 的核心公式与 API 已落地，性能也有明显价值，但 fixed-budget 并行粒度、执行期内存约束、adaptive standard error 以及若干真正不同语义分支的正确性证据仍需补齐。

本 review 只检查会影响数值正确性、可复现性、真实工作负载性能、内存可用性或阶段完成声明的事项。没有要求穷举 qubit 数、sample budget、`max_weight` 或 checkpoint interval；仅改变尺寸而不进入新代码路径的重复测试不视为验收缺口。除本报告及 `docs/vibe/README.md` 索引外，没有修改任何实现源文件。

## 审核范围与证据

- 权威合同：`docs/vibe/phase-4-spec.md`，重点核对第 4–12 节和第 14 节一次性验收标准。
- 实现：`crates/tencir-pauli-core/src/propagation.rs`、`crates/tencir-pauli-core/src/spps.rs`、两层 PyO3 binding、Python facade 和 TensorCircuit adapter。
- 测试与 reference：`tests/test_phase4_gradient.py`、`tests/test_spps.py`、`tests/spps_reference.py` 及相关 Phase 1–3 regression。
- 性能证据：Criterion、`benchmarks/python/test_phase4_benchmark.py`、large matched comparison、`implementation-status.md` 的历史 profile/benchmark 记录，以及本 review 的两个定向实验。
- 当前质量门：`.conda/bin/python scripts/check.py --benchmark smoke` 完整通过，包括 Rustfmt、Clippy `-D warnings`、17 个 Rust tests、Black、Ruff、strict mypy、release maturin build、106 passed/4 skipped Python tests、全部 Criterion smoke 和 73 passed/41 skipped benchmark smoke。通过本地 TensorCircuit source 运行 adapter tests 得到 3 passed/2 skipped；跳过项需要额外 JAX/SymPy 环境。

## Compliance checklist

| 项目 | 结论 | 说明 |
|---|---|---|
| 两条独立 Rust-native gradient 路线 | PASS | `PropagationEngine` 与 `SPPSEngine` 是独立内核和公开类型，没有用 mode flag 混合。 |
| Deterministic frozen-support 语义 | PASS | Forward exact-zero/projection 与 retained-edge reverse 一致；聚合后删除的 output 不进入反向；shared slot、PTM transpose 和 checkpoint replay 均已实现。 |
| Public API、只读 gradient、粗粒度 FFI、GIL release | PASS | 两个 hot call 各使用一次主要 PyO3 调用，参数批量传入、gradient 一次返回，长计算位于 `allow_threads` 内。 |
| SPPS proposal、importance reweighting、zero-factor PAD | PASS（实现） | 代码按正 smoothing proposal 采样，proposal 不参与求导，零 factor 使用不除零的 product 逻辑，三角函数按 call 预解析。 |
| 正确性验收矩阵 | FAIL | 现有 Phase 4 tests 未覆盖若干会进入不同相位、梯度归并、terminal expectation 或并行归约逻辑的语义分支，当前证据不足以支撑状态文档中的完整验收声明。 |
| Seeded replay 与跨线程一致性 | FAIL（证据） | 同进程重复 seed 已测，但没有 1/2/4-thread bitwise replay test；fixed-budget 单 term 实际上也没有 sample-level 并行。 |
| `max_bytes` 对主要 workspace 的 best-effort guard | FAIL | SPPS call 的 `O(N_terms * N_parameters)` per-term gradient storage 未估算，低 limit 可被轻易绕过并产生远高于 limit 的主要分配。 |
| Adaptive `value_standard_error` | FAIL | 两个独立 replicate 均值再取平均时使用了错误的方差系数，典型情况下 standard error 高估 `sqrt(2)`。 |
| Release 性能与 profile-backed 优化 | PASS（代表性 latency） | 现有 profile 导致了 in-place path update、generator-only transition、快速 product reduction 和 term-level Rayon；本机代表性延迟与 matched JAX 对比均显示实际收益。 |
| SPPS parallel scaling 与 memory 性能证据 | FAIL | 已有 budget/size scaling，但没有证明单 term path throughput 随线程扩展；定向实验确认当前 fixed-budget 单 term 不能扩展。 |
| Optional TensorCircuit 边界 | PASS | import 保持 lazy，adapter 位于 integration module，未修改 TensorCircuit source；numeric QIR 当前环境通过。 |

## CRITICAL

无。

## MAJOR

### M1. SPPS 正确性证据没有覆盖真正不同的数学分支，完成声明过强

`tests/test_spps.py:35-46` 只用一个单 qubit `RY`、单项 observable、`ZeroState` 做大样本近似；`tests/test_spps.py:49-69` 补了 zero/near-zero factor；`tests/spps_reference.py:129-155` 虽能枚举路径，但当前只返回完整路径和的 exact value/gradient，没有逐路径或按 proposal 加权地对照 native sample kernel。Deterministic tests 同样只直接差分了单 qubit `RX/RY/RZ`（`tests/test_phase4_gradient.py:13-39`）。

建议只补下面五组小系统测试；它们各自进入不同逻辑，不是重复尺寸扫描：

| 最小测试 | Fixture 与断言 | 实际覆盖的风险 |
|---|---|---|
| D1 local VJP table | generic angle `0.37`；`RX/RY/RZ` 遍历 4 个 one-qubit Pauli，`RXX/RYY/RZZ` 遍历 16 个 two-qubit Pauli；native 对照独立 frozen recurrence或中心差分 | 六种 gate 的 commute/anticommute、two-qubit phase sign和local derivative；总计仅 60 个微型 case。 |
| D2 deleted-support reverse | 一个 static PTM collision 使两个不同输入贡献精确抵消；另一个 `max_weight=1` case 让 rotation 生成的 weight-2 output 被删除；两者都断言 deleted output 不贡献 gradient | exact aggregate cancellation 与 projection deletion 是 frozen-support 合同中尚未直接验证的两条控制流。 |
| D3 deterministic concurrency | 同一 immutable engine 用 4 个 Python threads 同时执行两个参数向量，逐项等于各自串行结果 | `value_and_grad` 自身的 concurrent-call/scratch 隔离；现有并发测试只覆盖 Phase 3 forward。 |
| S1 composite exact-path | n=2，包含 interleaved Clifford、一个 static rotation、一个 two-qubit slot rotation和两个 occurrence 共享同一 slot；枚举每条 legal path，逐路径核对 probability、value contribution和slot gradient，再核对加权总和 | 多 active factor product、static factor、two-qubit sign、shared-slot scatter和proposal不求导，可由一个 fixture 同时覆盖。 |
| S2 terminal/reduction/replay | 一个 Bloch state 使 X/Y/Z expectation 均非零，observable 含一正一负 coefficient；另用 1-thread/4-thread local Rayon pool及两个并发 public calls断言同 seed bitwise identical | 非 `ZeroState` terminal reduction、多 term线性组合、thread-independent counter RNG与归约顺序。 |

这些 case 全部使用 n≤2；除 fixed-seed replay 外不需要大 Monte Carlo budget，预计总运行时间远低于一秒。Rust unit tests 虽检查了 two-qubit Clifford/local product table，但没有检查 two-qubit parameter derivative 与 SPPS importance/PAD 的组合，因此不能替代 D1/S1。

影响：当前实现很可能在常用单 qubit 路径上正确，但对 Phase 4 宣称的完整 gate/state/shared-parameter 合同缺少足够证据；后续优化 path kernel 时也缺少能阻止相位或 scatter 回归的有效门禁。

### M2. Fixed-budget SPPS 只按 observable term 并行，单 term 或少 term workload 无法利用多核

这里的 fixed-budget 是指调用者指定每个 observable term 恰好采样 `B` 条 path；相对地，adaptive mode 会按 A/B proxy 将 `B` 逐步翻倍。`fixed` 只固定样本数量，不要求串行执行。每条 path 已由 `(seed, term, sample index, gate index)` 独立决定，因此可以把 `0..B` 切成固定 chunks 并行计算，再按 chunk index 顺序归约，而不改变 budget、path choices或public result。

当前 `SPPSEngine::value_and_grad` 在 `crates/tencir-pauli-core/src/spps.rs:281-304` 只对 term 使用 `par_iter_mut()`，每个 term 内直接调用串行 `run_samples`。真正按 256-sample chunk 并行的 `run_samples_batched` 位于 `crates/tencir-pauli-core/src/spps.rs:534-607`，只被 adaptive 路径调用。这样 term 数较少、sample budget 较大时，大部分 CPU 核心空闲；它也没有达到规格第 6.6、8.3、10.3 节要求的 fixed sample-index chunk batching。

定向 release 实测使用 12 qubits、72 rotations、8 parameters、1 observable term、50,000 paths：`RAYON_NUM_THREADS=1` median 22.216 ms，`RAYON_NUM_THREADS=4` median 22.058 ms，没有可测 speedup。这个问题不会从更多 qubit-size benchmark 中自然暴露，必须用单/少 term 的 thread-scaling case 检查。

影响：SPPS 的复杂度主要随 path budget 增长，而不是一定随 observable term 数增长；当前并行策略会直接限制单项 observable、少项局域 observable以及高精度 budget 的吞吐量，属于阶段性能目标的实际缺口。

### M3. SPPS `max_bytes` 漏掉主要的 `N_terms * N_parameters` 执行期分配

构造期估算在 `crates/tencir-pauli-core/src/spps.rs:177-203` 只计入约 `2 * N_parameters * 8` 的 gradient bytes；fixed call 随后在 `crates/tencir-pauli-core/src/spps.rs:281-283` 为每个 observable term 创建一个包含完整 `gradient_sum: Vec<f64>` 的 `TermStats`，adaptive call 在 `crates/tencir-pauli-core/src/spps.rs:355-360` 创建两份。因此主要 workspace 实际为 fixed `O(N_terms * N_parameters)`、adaptive `O(2 * N_terms * N_parameters)`，而 engine 不保存 `max_bytes`，call 前也没有 guard。

定向实测使用 6 qubits、1,000 terms、1,000 parameter slots、`samples_per_term=2` 和 `max_bytes=150,000`：engine construction 与 call 均成功，call 的进程 peak RSS 增量约 10,043,392 bytes。这里不是要求精确约束 RSS；漏掉的 per-term gradient vectors 是尺寸完全已知、可廉价估算的主要 workspace，属于项目规则明确要求覆盖的对象。

影响：大型 VQE Hamiltonian 与高参数量 tape 的乘积会快速放大内存，当前 public guard 会给出错误安全感，并可能在进入 Rust hot call 后触发系统级内存压力。

### M4. Adaptive `value_standard_error` 的方差系数错误

Adaptive 最终值为 `0.5 * (mean_A + mean_B)`，见 `crates/tencir-pauli-core/src/spps.rs:843-847`。若两个独立 replicate 各有 `B` 个样本、单样本方差分别为 `var_A` 和 `var_B`，则 `Var(mean_A)=var_A/B`、`Var(mean_B)=var_B/B`。平均值整体乘了 `1/2`，方差必须乘系数平方 `1/4`，所以正确结果是 `(var_A + var_B)/(4B)`。实现却在 `crates/tencir-pauli-core/src/spps.rs:848-850` 使用 `0.5 * (var_A + var_B)/B`，相当于只平均了两边的方差，却漏掉“两个均值再次取平均”的平方系数。两边方差相近时，reported variance 正好大两倍，standard error 因为是方差的平方根而高估 `sqrt(2)`。`tests/test_spps.py:46` 只对 fixed mode 检查一个宽松上界，没有验证 adaptive standard error 的定义。

影响：这不改变 value、gradient 或 adaptive gradient proxy，但会系统性误报公开 `SPPSEstimate.value_standard_error`，影响调用者判断采样精度。修复成本很低，且应配一个可手算的 two-replicate moment test。

## MINOR

无需要单独阻止验收的 minor issue。规格中更完整的 size/budget/max-weight 笛卡尔扫描、逐层 FFI conversion 拆时和 exhaustive thread-count matrix 对当前正确性没有额外贡献，不建议为了形式完整而扩张。

## OBSERVATIONS

### O1. Deterministic 路线整体达到实际可用水平

`value_and_grad` 复用了 forward recurrence，checkpoint interval 只改变保存/replay 策略；reverse 通过 canonical output index 只访问 retained outputs。现有 checkpoint bitwise equality、shared slot、PTM、zero-support 和 non-Hermitian tests 加上完整 regression，使这条路线的风险显著低于 SPPS。没有发现会阻止当前 deterministic API 使用的 correctness bug。

### O2. 代表性性能良好，但应区分 latency 成绩与 scaling 完成度

本 review 的当前 release public benchmark 得到：12q deterministic median 约 0.487 ms（checkpoint 1）和 0.766 ms（interval 4），100q near-Clifford gradient 约 23.8 us；SPPS 12q/12-term median约 86 us（128 paths/term）与 242 us（1,024 paths/term），rotation-heavy 23-term case 约 0.414 ms。`implementation-status.md` 记录的大型 matched case 中，deterministic 对 TensorCircuit/JAX warm gradient 约 18.4x–43.6x，SPPS 在 12q/16q case 约 6.5x–12.4x。绝对时间和端到端边界都具有实际价值，但其中的 SPPS 成绩主要依赖多 term 并行，不能作为单 term sample-level scaling 已完成的证据。

### O3. 规格与状态文档的主语义描述基本一致

README 正确区分 frozen-support reverse 与完整 basis derivative，也没有把 adaptive proxy 写成 confidence interval；public typing、只读数组、optional dependency 和不修改 TensorCircuit 上游的边界清楚。完成状态需要在 M1–M4 修复前改为“主体完成、验收整改中”，避免把缺少证据的 thread replay、memory 和完整 correctness matrix 写成已闭环。

## RECOMMENDED IMPROVEMENTS

1. 对 M1 实现该节列出的 D1–D3、S1–S2 五组小系统测试；不增加 qubit/budget 尺寸矩阵。
2. 对 M2 让 fixed-budget 也使用固定 sample-index chunks，并按 chunk index稳定归约。保留单 term `RAYON_NUM_THREADS=1/4` throughput case和两个 local Rayon pools 的 same-seed bitwise equality test；不需要扩展为所有 qubit/budget/thread 组合。
3. 对 M3 在 call 前估算 fixed/adaptive `TermStats`、worker gradient、chunk result和 path scratch 的主要 storage，并优先把 per-term full gradient 改为 worker-local或分块归并，使 workspace 接近 `O(threads * N_parameters)`。加入一个低 `max_bytes`、高 `N_terms * N_parameters` 的 regression。
4. 对 M4 将 adaptive term variance 改为 `0.25 * (left_var + right_var) / count`，并用固定样本 moments 精确验证 `value_standard_error`。是否对 fixed mode采用 Bessel correction应另行明确，但不应阻塞这次确定的系数修复。
5. 完成上述四项后重跑 `scripts/check.py --benchmark smoke`，再补一组 release 单 term thread scaling和内存记录即可重新验收；无需重跑所有历史尺寸或制造新的全量性能矩阵。

## 最终判定

Phase 4 的功能主体和代表性性能已经具有实际价值，特别是 deterministic frozen-support reverse 可以视为完成。SPPS 也不是原型占位，而是可运行且在多 term workload 上很快；但 M1–M4 涉及正确性证据、公开统计量、主要内存和核心并行吞吐，不能归类为过度保护或未来才可能需要的工程。建议将阶段状态设为“implementation complete, acceptance remediation required”，修复后再宣告 Phase 4 完整完成。

## 整改结果（2026-08-02）

本次 review 指出的 M1–M4 已完成并纳入实现与测试：deterministic 路线增加了六种 rotation 的 one-/two-qubit local VJP 表、aggregate cancellation、projection deletion 和并发 gradient regression；SPPS fixed mode 改为固定 sample-index chunk 执行，chunk 结果按稳定顺序归约，因此单 term 也能进入 Rayon sample-level path；engine 保存 `max_bytes` 并在 fixed/adaptive call 前 checked 估算 term-gradient、moment、chunk result、resolved operation 和 path scratch workspace；adaptive 两个独立 replicate 的均值方差修正为 `(left_var + right_var) / (4 * count)`。

新增证据包括 Rust local Rayon 1/4-thread fixed replay、Python SPPS concurrent-call replay、单 term 1024-path chunk replay、低 `max_bytes` 与高 `N_terms * N_parameters` 回归、六类 rotation 的 60-case dense VJP table、deterministic aggregate/projection deletion、composite exact legal-path enumeration、Bloch terminal reduction 以及 adaptive standard-error moment unit test。release public workload（12q、72 rotations、8 parameter slots、1 observable term、50,000 paths）稳定调用平均为 8.29 ms（1 thread）和 2.25 ms（4 threads）。最终 `scripts/check.py --benchmark smoke` 通过 19 个 Rust tests、114 个 Python tests（4 optional skips）和 73 个 benchmark tests（41 optional skips）；历史 review 中的 FAIL 表格保留为整改前审计记录。

整改后阶段状态为：**implementation complete; acceptance remediation complete**。
