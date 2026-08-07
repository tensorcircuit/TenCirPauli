# Phase Experimental A：Zixy 启发的算符数据平面性能实验

状态：实验设计提案。本文不授权任何生产实现，也不预设任何候选方案最终会被采用。

## 目标

本实验阶段用于验证从 Zixy 源码中发现的若干底层架构假设，判断它们是否能在 TenCirPauli 的真实端到端工作负载上带来足够稳定的性能或内存收益。每个候选方向都必须先完成 A/B 对照实验，只有在正确性、接口语义、资源合同和代表性性能证据全部通过后，才允许进入单独的实现决策。

本阶段的核心问题不是“哪种数据结构看起来更先进”，而是当前实现是否已经在真实成本中被 allocation、cache locality、重复 canonicalization、中间 operator materialization 或 packed representation 往返所主导。如果候选方案只改善孤立 kernel、只在人工极端输入上有效，或者端到端收益不足以覆盖复杂度，则应保留当前实现并关闭实验。

## 非目标

- 不在本阶段改变公共 Python API、operator 语义、canonical ordering、phase convention、qubit ordering 或 `max_bytes` 合同。
- 不因为 Zixy 使用某种表示就直接迁移该表示，不维护未经证据支持的两套长期生产 storage。
- 不把 microbenchmark 的改善写成库级性能结论；所有晋级证据都必须包含输入转换、Rust/Python boundary 和最终结果 materialization 的端到端路径。
- 不在本阶段扩展新的 operator family、mapping 类型、measurement grouping 算法或 chemistry workflow。

## 统一实验规则

### A/B 定义

A 是当前 release-mode 实现，保持现有公共接口和算法语义不变。B 是仅用于实验的候选实现，可以通过 benchmark-only feature、独立 native kernel 或临时分支接入，但不得把未通过证据门槛的 B 方案混入默认生产路径。

每个 B 方案必须与 A 使用相同输入、相同输出 materialization、相同线程数、相同 `max_bytes`、相同数据类型和相同误差要求。若 B 需要改变内部 representation，benchmark harness 可以增加转换成本，但不能将转换成本从端到端计时中删除。

### 正确性门槛

所有实验必须先通过现有数值和语义测试，再增加针对性 differential test。至少需要覆盖 Pauli multiplication phase、canonical ordering、duplicate aggregation、mapping output、Hermiticity、qubit ordering、structured operator round-trip 和 dense small-system reference。任何正确性差异、确定性差异、资源限制绕过或梯度差异都会使该 B 方案暂不晋级，无论其速度提升多大。

### 性能证据门槛

实验使用 optimized release build，固定机器、Rust 版本、Python 环境和线程配置，分别记录冷启动、steady-state、输入转换、native execution、输出 materialization、峰值内存和可获得的 allocation 指标。每个 workload 至少重复多次并报告中位数、离散程度和相对 A 的变化；不得只报告一次最快结果。

默认的晋级参考线是：B 在至少三类代表性 workload 中的两类上取得不低于 15% 的端到端中位运行时间改善，或者取得不低于 20% 的峰值内存改善，同时其他代表性 workload 的端到端回归不超过 5%。这只是实验决策的默认参考线，不替代对 workload 重要性、复杂度、可维护性和长期收益的判断。

若收益只出现在内部 kernel、只出现在 term 数极小或极大的非代表性输入、只来自减少输出 materialization，或者需要保留高维护成本的双重生产表示，则不应晋级为默认实现。实验结果应记录为采用、继续观察或淘汰，并保留 workload、环境和原始结果。

## 代表性 workload 矩阵

所有实验应尽量覆盖以下 workload，而不是为单一优化挑选有利输入。

| 类别 | 代表性输入 | 主要观察 |
| --- | --- | --- |
| Pauli canonicalization | 32–256 qubit、随机项、重复项比例分别为低/中/高 | allocation、hash、排序、聚合、cache locality |
| Operator product | 中等 term 数的 Pauli product、commutator、anticommutator | pair expansion、临时对象、duplicate aggregation |
| Chemistry mapping | PySCF/积分导出的实际规模 Fermion Hamiltonian 和二体项 | 中间 operator、mapping throughput、峰值内存 |
| Mapping reuse | 同一 mapping plan 重复映射多个 operator | plan construction、workspace reuse、steady-state throughput |
| Grouping/symmetry | QWC/general commuting、Z2 和 charge-sector analysis | 按 qubit 扫描、word traversal、输入布局 |
| Propagation | 动态 Pauli propagation、weight projection、多 observable batch | packed key、coefficient update、scratch reuse |
| Structured operator | fermion、Majorana、hybrid operator 的构造和乘法 | nested allocation、flat payload、跨 family 转换 |

## 实验 A：连续 SoA Pauli storage

### 假设

当前 `PauliOperator -> Vec<PauliTerm> -> PauliWord -> x_words/z_words Vec` 的层层 owning storage 可能造成每个 term 的多次 allocation、指针追踪和低 cache locality。Zixy 的 component/coefficient 分离和 borrowed term view 可能在 term 数较大时显著降低内存和遍历成本。

### A/B 方案

- A：当前 `Vec<PauliTerm>` 和每个 `PauliWord` 独立拥有 packed masks 的实现。
- B：benchmark-only 的连续 `x_words`、`z_words`、`coefficients` storage；固定宽度 Pauli word 使用 term-major 或 component-major 的一种布局，并提供不分配的 borrowed `TermView`。

B 方案不能同时长期保留两份生产 operator representation。实验中可以提供一次性转换，但转换时间和内存必须纳入端到端结果。

### 重点指标

- canonicalization 的端到端时间和峰值内存；
- operator multiplication、commutator 和 coefficient-only operation 的吞吐；
- term 数扩展时的 scaling；
- allocation 次数、L1/L2 cache miss 或可获得的等价硬件计数；
- B 的转换成本是否在重复操作中被摊薄。

### 晋级条件

只有当 B 在 canonicalization、operator product 或 mapping reuse 中至少两类真实 workload 达到统一性能门槛，并且没有破坏 lazy handle、确定性排序和 read-back ABI，才进入单独的生产实现评审。

## 实验 B：TermSet 式原地唯一化聚合

### 假设

当前 canonicalization 和 multiplication 使用 `HashMap<PauliWord, Vec<Complex64>>`，可能为每个 collision key 分配临时 coefficient vector，再排序并归约。Zixy 的 `Terms`、`TermSet`、`TermSum` 分层提示可以把重复输入 batch、唯一 key set 和线性组合明确分开，并在 native 层对唯一 key 直接聚合。

### A/B 方案

- A：当前按 key 保存 `Vec<Complex64>`，最后排序和 fold 的 deterministic reduction。
- B：benchmark-only 的唯一 key accumulator，每个 key 直接维护聚合状态；如果必须保持浮点聚合确定性，则使用固定 chunk reduction 或稳定 finalization，而不是无序地改变加法顺序。

B 不能为了速度放弃当前的确定性 coefficient policy，也不能把 hash iteration order 暴露为 public ordering。

### 重点指标

- duplicate ratio 从 0% 到高碰撞时的 canonicalization 时间和峰值内存；
- operator multiplication、fermion CAR expansion 和 hybrid multiplication；
- 临时 Vec 数量、总 allocation bytes 和 aggregation workspace；
- 不同输入排列是否得到完全相同的 canonical output。

### 晋级条件

B 必须在 duplicate-heavy 和 ordinary chemistry 两类 workload 中都取得可重复的端到端收益；如果只在极高重复率的合成输入上有效，则只记录为局部优化，不进入默认路径。

## 实验 C：mapping workspace 与 Fermion-to-Pauli 融合

### 假设

当前 pure Fermion mapping 先生成完整 Jordan–Wigner Pauli operator，再执行 parity/BK 的 Pauli transform。Zixy 的 mapping workspace 会缓存单 mode image，在一个 native pass 中直接组合 ladder product 并写入最终 term set，可能减少中间 operator、二次 canonicalization 和峰值内存。

### A/B 方案

- A：`FermionOperator -> Jordan-Wigner PauliOperator -> mapping plan transform -> final PauliOperator`。
- B：native `MappingWorkspace` 预计算 creation/annihilation 的单 mode image，复用 scratch buffers，在一次 coarse-grained native call 中直接产生最终 mapping 的 canonical accumulator。

B 必须支持现有的资源预估和失败语义。不能因为 workspace 复用而允许未经估算的指数 ladder expansion，也不能把映射中间态偷偷转回 Python。

### 重点指标

- 实际 chemistry Hamiltonian 的端到端 mapping 时间；
- mapping peak RSS 和 native allocation bytes；
- 单个 plan 重复映射多个 operator 时的 steady-state throughput；
- JW、parity、BK 三种 mapping 的 scaling；
- 长 ladder product 和二体项的扩展峰值。

### 晋级条件

B 必须在真实 chemistry workload 和 mapping reuse workload 中至少一类达到性能门槛，并证明输出与当前路径逐项一致。若 B 只改善 plan construction 而不改善重复 mapping，需单独评估是否值得增加 workspace 生命周期和维护复杂度。

## 实验 D：packed transform、phase 和 ordering kernel

### 假设

当前 Pauli word 已经有 packed X/Z masks，但部分路径仍然在 packed representation 与逐 qubit `u8` codes 之间往返。`PauliWord::multiply` 的结果使用 limb XOR，而 phase 仍逐 qubit调用 `code_at()`；mapping 还会对每个 term 执行 `codes()`、`from_codes()` 和再次 materialization。Zixy 的 packed phase/component 操作提示这些路径可以完全保持 limb-level。

### A/B 方案

- A：现有 `codes()`、`from_codes()`、逐 qubit phase 和逐 qubit ordering 路径。
- B：直接接收和产生 packed `x_words/z_words`，使用 bitwise/popcount 计算 phase，并使用 packed comparator 保持相同的 public canonical order。

B 只允许在内部 kernel 使用 packed representation；Python 显式请求 codes 时仍然可以 materialize flat array。

### 重点指标

- Pauli word multiplication、hash/canonical sort、mapping transform 的单项吞吐；
- 64、128、256、512 qubit 下的 scaling；
- `codes()` 往返次数和临时 allocation；
- packed phase 与逐项参考的数值一致性；
- 单 kernel 收益在完整 operator workflow 中是否仍然存在。

### 晋级条件

B 若只带来 kernel 级收益但端到端收益低于门槛，则不得单独引入复杂 packed abstraction；若它与实验 A 或 C 组合后才有效，必须重新进行包含组合转换成本的 A/B 测试。

## 实验 E：mapper-specific compact plan

### 假设

当前 `MappingPlan` 同时保存 dense `encoding`、`inverse_encoding`、CNOT provenance 以及 packed X/Z transforms。Zixy 的 update/parity/rho 集合和 BK 的 Fenwick 风格结构提示 JW、parity、BK 可以使用各自的紧凑 runtime plan，避免长期保存多套相关矩阵。

### A/B 方案

- A：当前 dense matrix、GF(2) inverse、packed transform 和 CNOT provenance 的组合。
- B：mapper-specific runtime plan；JW 使用 prefix masks，parity 使用对应的 prefix/suffix 结构，BK 使用 Fenwick/update/parity/rho 集合；公共 encoding metadata 仅在显式请求时 materialize。

B 不得通过删除现有公开 metadata 改变 API；如果需要保留 metadata，应把它作为按需生成的 view 或缓存，并将生成成本单独计入相应 API 的端到端测量。

### 重点指标

- mapping plan build time 和峰值内存；
- plan 的常驻内存；
- 重复 Pauli transform 和 occupation transform throughput；
- JW、parity、BK 在宽 mode count 下的 scaling；
- metadata materialization 对用户可见路径的影响。

### 晋级条件

B 主要在大 mode count 或大量 plan reuse 场景中取得稳定收益时才有意义。若普通 chemistry 规模下差异很小，则保持当前通用 plan，避免为理论上的大规模输入引入多套实现。

## 实验 F：structured operator flat payload

### 假设

structured operator 的 `Vec<Vec<...>>` batch 表示可能带来与 Pauli operator 相同的 per-term allocation 和 clone 成本。Zixy 的 component/coefficient view 思路可以推广为 offsets 加 flat payload，用 slice view 执行 CAR rewrite、hybrid multiplication 和 mapping。

### A/B 方案

- A：当前 `Vec<Vec<u32>>`、`Vec<Vec<u8>>` 和 `Vec<Vec<(u32, u32, u32)>>` 表示。
- B：benchmark-only 的 flat payload 加 offsets 表示，term view 只携带 payload slice 和 coefficient index。

B 必须支持 variable-length fermion words、boson blocks、qudit triples 和 mixed-domain optional factors，不得为了布局整齐而把这些 family 错误地降级成 Pauli-only representation。

### 重点指标

- Fermion/Majorana/hybrid construction 和 multiplication；
- CAR rewrite、mapping 前后的临时 allocation；
- term 数和平均 factor length 的 scaling；
- Python 输入转换、FFI 传输和 native 输出 materialization 的总成本。

### 晋级条件

B 只有在 structured operator 的真实工作流中取得明显收益，并且不会引入第二套长期 Python storage 时才进入后续设计。若收益仅来自 benchmark-only 输入重排，则应淘汰。

## 实验之间的组合规则

实验 A、B、C、D 可能互相放大，也可能把转换成本叠加起来。不得只测试“所有 B 同时打开”的结果，因为这样无法知道收益来自哪一个假设；应先分别做单因素 A/B，再对通过单因素门槛的候选做少量组合实验。

实验 E 和 F 改变的对象生命周期与数据布局更大，必须在 A、B、C、D 有证据支持后再开始组合，避免同时改变 storage、mapping 和 structured input 导致结果不可归因。

## 结果记录格式

每个实验应在本文件或关联 benchmark record 中记录：实验编号、A/B commit、机器与软件环境、workload 输入摘要、线程数、数据类型、max_bytes、正确性结果、cold/steady 时间、输入转换时间、输出 materialization 时间、峰值内存、allocation 指标、相对变化、离散程度、是否通过门槛以及下一步决定。

结果只能标记为以下三种状态之一：`adopt` 表示证据支持进入独立实现规格；`observe` 表示有局部信号但不足以改变架构；`reject` 表示收益不足、回归过大、复杂度不合理或正确性门槛未通过。

## 决策边界

本实验阶段完成后，不得直接把 `adopt` 解释为已经实现。每个通过的方向都必须另行形成小范围实现规格，明确最终 storage contract、兼容策略、迁移成本、删除的旧路径、测试向量和 release-mode benchmark。没有达到统一性能门槛的候选方案继续留在实验记录中，不应因为“理论上更 cache-friendly”而进入生产代码。

## 参考

- [Rust quantum operator competitor research](competitor-research.md)
- [Zixy term containers](https://github.com/Quantinuum/zixy/blob/main/zixy-py/zixy/container/terms.py)
- [Zixy native term views](https://github.com/Quantinuum/zixy/blob/main/zixy/src/container/word_iters/term_set.rs)
- [Zixy TermData](https://github.com/Quantinuum/zixy/blob/main/zixy-py/zixy/container/data.py)
- [Zixy mapping workspace](https://github.com/Quantinuum/zixy/blob/main/zixy/src/fermion/mappings/operators.rs)
- [TenCirPauli Pauli word](../../crates/tencir-pauli-core/src/word.rs)
- [TenCirPauli Pauli operator](../../crates/tencir-pauli-core/src/operator.rs)
- [TenCirPauli mapping plan](../../crates/tencir-pauli-core/src/mapping.rs)
