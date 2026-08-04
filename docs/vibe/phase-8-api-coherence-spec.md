# Phase 8 API Coherence Specification

状态：已实现并通过本地 API contract、集成、Rust 和 Python 质量检查。本文记录 TenCirPauli 在正式用户出现之前进行的一次 breaking-change API 整理；Phase 8 的接口合同优先于早期阶段中与之冲突的示例性 API 描述，早期文档作为历史记录保留。

## 1. 目标

本阶段不增加新的量子算法或新的 Rust kernel，目标是把现有 Python public API 收敛成少量、可组合、容易被 Agent 发现的入口。相同概念在不同对象之间必须使用相同的字段名、参数名和执行协议；语义确实不同的对象则明确保留差异，而不是通过继承或别名伪装成相同接口。

本阶段允许 breaking change。当前没有需要维护的外部用户，因此不保留仅为兼容旧版本而存在的名称、属性或构造路径。代码、测试、examples、README、docstring 和 API 文档应在同一次变更中迁移到新合同。

## 2. 总体设计原则

### 2.1 普通入口与 advanced 入口分层

普通用户和 Agent 的默认路径应尽量短：构造 canonical operator，调用统一的分析/变换方法，再调用 `compile()` 或 circuit facade 的 value terminal。数组 canonicalization、raw engine、native plan、内部 result container 和 QIR 恢复属于 advanced API；它们可以继续存在，但不能与普通构造器在同一层级上暗示为等价主路径。

推荐的普通入口是：`PauliWord.from_string()`、`PauliWord.from_codes()`、`PauliOperator.from_terms()`、`operator.compile(target=...)`、`PropagationCircuit`、`SPPSCircuit`、`U1Circuit`、`OperatorSpace`、`AdditiveCharge` 和 `FermionQubitMapping.from_name()`。

### 2.2 统一命名，保留领域必要的维度名称

`nqubits`、`n_modes`、`nparameters` 和 `dimension` 是有明确领域含义的名称，应保留。表示同一个计数或执行属性的名称不得因为模块不同而变化。

### 2.3 不用一个过度抽象的基类强行统一语义

Pauli MVP plan、restricted-operator plan 和 circuit plan 都是可复用对象，但它们不是同一种执行对象。统一应通过最小 Python protocol、共同字段和一致的 docstring 完成，而不是建立一个包含所有 terminal 的万能基类。

### 2.4 只统一真正相同的语义

共同名称必须对应共同语义，而不只是相似的数据形状。代数 term、restricted transition、matrix nonzero entry、observable 和 circuit gate 分别计数；不得为了让字段看起来一致而把 transition 或 matrix entry 称为 term。统一 protocol 只包含所有实现都能无歧义满足的属性，领域特有的 provenance 和执行统计通过附加字段表达。

## 3. 必须修正的行为问题

### 3.1 移除 SPPS 对 deterministic propagation API 的继承泄漏

`SPPSCircuit` 不再继承 `PropagationCircuit`。两者改为共同继承私有的 `_CircuitBuilder`。该私有基类只提供 operation storage、wire validation、parameter expression、cache invalidation、QIR serialization 和 conversion 的内部机制，不通过继承自动扩大任一 facade 的 public gate 或 terminal surface。

Propagation 与 SPPS 共同支持的 public gates 固定为 `x`、`y`、`z`、`h`、`s`、`sdg`、`cnot`、`cz`、`swap`、`rx`、`ry`、`rz`、`rxx`、`ryy` 和 `rzz`。任意实 Pauli-transfer matrix 的 `ptm()` 只属于 `PropagationCircuit`；`SPPSCircuit` 不暴露 `ptm()`，`SPPSCircuit.from_qir()` 和 `SPPSCircuit.from_circuit()` 也必须在 conversion boundary 明确拒绝 PTM，而不是等到 native SPPS plan construction 才失败。

`from_circuit()` 和 `from_qir()` 使用 `Self` 或等价的精确 subclass typing；从 `SPPSCircuit` 调用时静态和运行时返回类型都必须是 `SPPSCircuit`。共享 conversion 只接受目标 facade 真正支持的 gate set。

`PropagationCircuit` 保留 deterministic 专属 terminal：`expectation()`、`value_and_grad()`、`propagate_operator()` 和 `profile()`。

`SPPSCircuit` 只提供 stochastic 专属 terminal：`expectation()`、`value_and_grad()` 和 `value_and_grad_adaptive()`。SPPS 不再暴露 `propagate_operator()` 或 `profile()`，因为随机路径 plan 没有这两个语义。对应的 `SPPSCircuitPlan` 也只提供相同的 stochastic terminal。

这不是把两个执行器合并；它只是消除当前继承关系造成的无效方法。必须增加回归测试，确认 `SPPSCircuit` 不再拥有 `propagate_operator()`、`profile()` 或 `ptm()`，并确认真正共享的 gate、parameter、QIR 和 conversion 行为仍然一致。

### 3.2 明确 Hermitian 与 value/gradient 合同

所有 deterministic propagation 和 SPPS 的 scalar estimator 都要求 exact Hermitian `PauliOperator`，这里的 public execution contract 使用 `is_hermitian(tolerance=0.0)`，不静默容忍或丢弃小虚部。U1 的 `expectation()` 可以计算一般 complex expectation，但 U1 的 `value_and_grad()` 仍要求 exact Hermitian observable 并返回 real `value`。

Hermitian validation 的时机固定如下：

- `PropagationCircuit.compile()` 和 `PropagationEngine` construction 允许 non-Hermitian operator，因为 `propagate_operator()` 对它仍有定义；deterministic `expectation()`、`value_and_grad()` 和带 scalar value 的 `profile()` 在执行前检查并对 non-Hermitian 输入抛出 `ValueError`。
- `SPPSCircuit.compile()` 和 `SPPSEngine` construction 可以直接拒绝 non-Hermitian observable，因为 SPPS 的所有 public terminals 都是 scalar estimators。
- `U1Circuit.compile()` 与 `U1Circuit.expectation()` 不要求 Hermitian；`U1Circuit.value_and_grad()` 及对应 plan terminal 对 non-Hermitian observable 抛出 `ValueError`。
- `PropagationBatch` 的 scalar/batch-gradient terminal 在任一 observable non-Hermitian 时整体抛出 `ValueError`，不返回部分结果。

所有相关 public docstring 必须明确 `expectation()` 的返回类型、`value_and_grad()` 的 Hermitian 前置条件、上述异常类型，以及 SPPS 结果是 estimate 而不是 exact scalar。Python facade 负责统一 Python 类型和公开错误合同，native 层继续作为最终语义防线。不得为了 Python precheck 在每次 hot execution 中增加一次额外的全 operator 扫描或 PyO3 round trip；exact Hermiticity 应在 immutable operator/compiled handle 中缓存，或复用 construction 时已经得到的标志。

### 3.3 统一 Pauli 输入验证

所有 Pauli code 输入都使用同一条规则：必须支持 Python `__index__`/`operator.index()` 语义且归一化后属于 `0, 1, 2, 3`；因此内置 `int` 和 NumPy integer scalar 可接受，`bool` 与 `numpy.bool_` 必须显式拒绝。不能解释为 integer-like 的 scalar 抛出 `TypeError`，integer-like 但超出 `0..4` 半开区间的 scalar 抛出 `ValueError`。

`PauliWord.from_codes()`、`batch_from_codes()`、mapping 和 structured builder 使用同一套 scalar normalization helper。`PauliOperator.from_code_arrays()` 对 integer-dtype NumPy/array-like 输入使用等价的向量化规则，拒绝 bool、float 和 object dtype；不得为了字面上的“共享 validator”在大批量 array path 中增加 Python per-element loop。所有路径的错误文本统一使用枚举 `0, 1, 2, 3` 或半开区间 `0..4`，不再使用含义错误的 `0..3`。

## 4. 统一计数和结果字段

以下名称是 Phase 8 的 canonical vocabulary：

| 概念 | canonical 名称 | 处理方式 |
| --- | --- | --- |
| 一个 operator 的非零 canonical algebraic term 数 | `term_count` | 所有 operator 使用；删除 `nterms` |
| MVP plan 编译所消费的 canonical algebraic term 数 | `term_count` | 等于该 plan 的 `plan_term_count`；不表示 transition/NNZ 数 |
| grouping 产生的 group 数 | `group_count` | 两种 grouping result 都提供 |
| grouping 覆盖的 operator term 数 | `term_count` | 两种 grouping result 都提供 |
| batch 中独立 observable 数 | `observable_count` | `PropagationBatch` 保留，因为它不是 term 数 |
| 传播 profile 的首末项数 | `initial_term_count`、`final_term_count`、`peak_term_count` | 替换 `initial_terms`、`final_terms`、`peak_terms` |
| mapping/structured lowering 后交给 MVP compiler 的 canonical term 数 | `plan_term_count` | 与 MVP plan 的 `term_count` 相同 |
| 映射前 source term 数 | `source_term_count` | 只在 mapping/structured plan provenance 中使用 |
| restricted plan 中实际保存的聚合 transition 数 | `transition_count` | U1/charge 等 transition plan 使用；不得称为 term count |

`source_term_count` 和 `plan_term_count` 都在各自阶段完成 canonicalization 和 exact-zero removal 后计数。一个 algebraic term 即使在特定 restricted sector 中作用为零，仍计入 consumed `plan_term_count`；lowering 后真正保存的聚合稀疏作用由 `transition_count` 表示。

`PauliOperator`、`MajoranaOperator` 和所有 structured operator container 都提供 `term_count` 与 `__len__()`，并满足 `len(operator) == operator.term_count`。因此 canonical zero operator 的 Python truth value 为 false；这一行为是正式 public contract。MVP plans、grouping results 和 single-observable engines 使用相同的 `term_count` 名称，但不因为拥有该 metadata 而实现 `__len__()`。`SPPSEngine.observable_terms` 改为 `term_count`，deterministic single-observable engine 也提供对应 metadata。`nqubits`、`n_modes`、`nparameters` 和 `dimension` 不改成 `count` 形式。

## 5. Grouping result 合同

`QWCGroupingResult` 和 `GeneralCommutingGroupingResult` 都提供 `groups`、`group_count`、`term_count`、`term_to_group`、`mode`、`algorithm` 和 `measurement_ready`。`group_count` 等于 `len(groups)`；每个 canonical term index 必须在 `groups` 中恰好出现一次，`term_count` 是覆盖的 term 总数。

`QWCGroupingResult` 继续提供 `bases`、`reconstruction_masks` 和 `reconstruct()`。两种 result 当前重复保存 `groups` 的 `coefficient_mapping` 都删除，改为真正的 immutable `term_to_group: tuple[int, ...]`；其长度固定为 `term_count`，位置是 canonical term index，值是对应 group index。`term_to_group` 在 result construction 时一次生成，不在每次属性访问时重复扫描 groups。

`mode` 和 `measurement_ready` 是 result type 的不变量而不是可由调用者任意覆盖的普通 dataclass constructor 参数：QWC 固定为 `mode="qubit_wise"`、`measurement_ready=True`，general 固定为 `mode="general"`、`measurement_ready=False`。所有 result arrays/tuples 与 reconstruction metadata 构造后不可变。

`group_commuting()`、`group_operator()`、`compatibility_matrix()` 和 `incompatibility_edges()` 的 `mode` 默认统一为 `"qubit_wise"`。这是 measurement-safe、相对保守的默认值：它可能少报告只能 general commute 的 pair，但不会把缺少 tensor-product measurement basis 的 group 标成 measurement-ready。需要 algebraic general commuting 时必须显式传入 `mode="general"`，避免相邻 helper 使用不同默认语义。

## 6. MVP plan 的统一协议

所有公开的 matrix-free operator plan 都遵循以下最小协议：

```python
class MVPPlan(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def term_count(self) -> int: ...

    @property
    def estimated_bytes(self) -> int: ...

    @property
    def basis_ordering(self) -> str: ...

    @property
    def target(self) -> Literal["native_mvp", "backend_mvp"]: ...

    def apply(
        self,
        state: Sequence[complex],
        *,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
    ) -> np.ndarray: ...

    def __call__(self, state: Sequence[complex]) -> np.ndarray: ...
```

`NativeMVPPlan`、`BackendMVPPlan`、`U1MvpPlan` 和 eager `ChargeMvpPlan` 都实现这个最小协议。`term_count` 严格使用第 4 节的 algebraic-term 定义；`U1MvpPlan` 和 eager `ChargeMvpPlan` 另外提供 `transition_count`，表示 sector lowering 与 duplicate aggregation 后实际保存的 sparse transitions。native restricted plans 的 target 为 `"native_mvp"`，backend plan 的 target 为 `"backend_mvp"`。Charge restriction additionally accepts explicit `storage="lazy"`; its lazy native MVP plan keeps sector metadata and term metadata, applies one flat vector at a time, and does not provide sparse materialization or a transition count.

共同 `apply()` contract 只接受 shape `(dimension,)` 的 flat complex state，并始终返回 shape `(dimension,)`。direct-Weyl backend plan 当前接受 mixed-radix tensor shape 并保留输入 rank 的 convenience 从共同 `apply()` 中删除；调用者可依据 `local_dimensions` 自行 reshape，reshape 不改变 basis ordering 或底层数据。若未来需要 `apply_tensor()`，它必须作为独立 advanced convenience 审查，不能改变 `MVPPlan.apply()` 的 flat-vector contract。

`estimated_bytes` 只表示 plan 自身拥有的 immutable storage 的 best-effort estimate，不包含传入 state、`apply()` 输出、每次执行的 temporary workspace、Python wrapper 或 allocator overhead。每次 `apply(..., max_bytes=...)` 使用该次调用自己的 execution budget；创建 plan 时传给 factory/`compile()` 的 `max_bytes` 只约束 plan construction 和 plan-owned major storage，不成为后续执行的隐藏默认值。`plan(state)` 精确定义为 `plan.apply(state)`，因此使用全局 `DEFAULT_MAX_BYTES`；需要不同 execution budget 时必须显式调用 keyword-only `apply(state, max_bytes=...)`。

所有 `apply()` 的 `max_bytes` 都改为 keyword-only。输入通过一次 array conversion 归一化为 C-contiguous `complex128` flat vector；所有 MVP `apply()` 返回 owned、C-contiguous、shape `(dimension,)` 的 `complex128` NumPy vector。输出可写，plan 自身携带的公开结构数组必须 read-only。所有 plan 的公开 metadata 在构造后不可重新赋值。内部可使用不改变 public semantics 的 cache，但必须保证并发只读执行安全。

`PropagationCircuitPlan` 和 `SPPSCircuitPlan` 是 circuit plan，不要求实现 MVP protocol，但必须满足同样的 public immutability：公开属性不可重新赋值、公开数组 read-only，并支持多个线程并发读取；不能只在 docstring 中声称 immutable。builder/circuit facade 本身仍是 mutable，不承诺并发 mutation 安全。

native plan 和 backend plan 不再允许用户直接使用内部 native handle、packed mask 或 generic entry 构造。它们由 `compile()`、`native_mvp_plan()`、`backend_mvp_plan()`、restricted-operator factory 或 mapping factory 创建；内部构造器改为 private/factory-only。本阶段不新增通用 plan deserialization API；未来若需要跨进程恢复 versioned backend plan，应增加经过 schema 和 array invariant 验证的独立 `from_arrays()`/`load()` advanced factory，而不是重新公开 generic constructor。

## 7. Circuit facade 合同

三个 facade 都继续使用 `theta=`、`Parameter`、`ParameterExpr`、一维 runtime parameter vector、`expectation()`、`value_and_grad()`、`compile()` 和 `to_qir()`，但 compile 语义必须写清楚：

| facade | `compile()` 固定的内容 | 主要 terminal |
| --- | --- | --- |
| `PropagationCircuit` | observable、initial state、weight projection 和 propagation engine | expectation、value/gradient、propagated operator、profile |
| `SPPSCircuit` | observable、initial state、smoothing 和 stochastic engine | value estimate、value/gradient estimate、adaptive estimate |
| `U1Circuit` | circuit program、particle-number sector 和 restricted-state engine | restricted/full state、probability、expectation、value/gradient |

U1 的 `compile()` 不接 observable 是刻意保留的语义差异，不改成与 propagation 相同。observable 在 U1 terminal 中传入，并通过 final-state cache 复用 circuit execution。

U1 构造器固定为以下形状：

```python
U1Circuit(
    nqubits: int,
    particle_number: int | None = None,
    *,
    occupied: Sequence[int] | None = None,
    initial_state: Sequence[complex] | np.ndarray | None = None,
    max_bytes: int | None = DEFAULT_MAX_BYTES,
)
```

删除 `k`、`filled` 和 `inputs`。`particle_number` 与 `U1Sector.particle_number` 对齐；`occupied` 是初始 computational-basis occupation 的 qubit index 序列；`initial_state` 是 restricted-sector complex vector。参数组合规则固定如下：

- `occupied` 可以单独给出并推导 `particle_number=len(occupied)`；同时给出两者时必须一致。
- `occupied` 必须由互不重复的 in-range qubit indices 组成，内部按升序 canonicalize，输入顺序不改变物理初态。
- `occupied` 与 `initial_state` 都指定初态，因此互斥。
- 使用 `initial_state` 时必须显式提供 `particle_number`，因为 restricted vector 的长度不能在所有 sector 间唯一反推出粒子数。
- 只提供 `particle_number` 时保留当前确定性默认初态，即 `occupied=tuple(range(particle_number))`。
- `initial_state` 必须是一维、finite、长度为 `sector.dimension` 的 complex vector；本阶段延续现有语义，不额外强制单位归一化。

facade 公开 `particle_number`、`sector` 和 `dimension`，不保留 `k` alias；`occupied` 只是 basis-state initialization 输入，不作为对任意 vector 初态都有意义的持久属性。

U1 的 canonical restricted-state terminal 是 facade 的 `state()` 和 plan 的 `run()`；两者都返回 restricted-sector vector。删除仅为 `state()` alias 的 `wavefunction()`。当前 `to_dense()` 实际会把 restricted amplitudes 扩展到 full computational basis，并非重复 terminal，因此不删除该能力，而是将 facade 和 plan 的 `to_dense()` 都改名为语义明确的 `state_full()`。最终成对保留 `state()`/`probability()` 与 `state_full()`/`probability_full()`；带 `_full` 的 terminals 返回长度 `2**nqubits` 的 full-space vector，并受保存的 resource budget 保护。

U1 的 `expectation_z()`、`expectation_ps()` 和 `expectation_pss()` 删除，不作为普通入口保留。Agent 应构造 `PauliOperator` 后调用统一的 `expectation()`；如果后续真实工作流证明单 word convenience 有明确价值，再通过新的设计审查决定是否增加 `expectation_word()`，不在本阶段预留 alias。

所有高层 facade 和 facade plan 的 `parameters` 都统一为默认 `None`，包括 `value_and_grad()`；无参数 circuit 可以省略，参数化 circuit 在执行时得到统一的 shape/finite validation。`None` 只在 `nparameters == 0` 时等价于空 vector，在参数化 circuit 上抛出 `ValueError`。SPPS plan 当前缺少默认值的 `parameters` 参数和 U1 plan 当前使用空 tuple 默认值的 terminals 都必须迁移。

## 8. U1 basis 与 state 返回类型

`U1Sector.unrank()` 的 public 返回类型固定为 `tuple[int, ...]`，长度为 `nqubits`，元素为 `0` 或 `1`，按 public qubit order 从 qubit zero 到 qubit `nqubits - 1` 排列；不再根据 `nqubits <= 64` 改变为 integer 或 tuple。`rank(unrank(i)) == i` 必须对所有合法 index 成立。

普通 materialization terminal 改名为 `basis_states()`，稳定返回 shape `(dimension, nqubits)`、dtype `uint8` 的 C-contiguous array；每一行与同 index 的 `unrank()` 完全一致。`nqubits == 0` 时 shape 固定为 `(1, 0)`。这个 ordinary representation 比 packed limbs 最多多约八倍 payload memory，但语义直接、无机器字宽度分支，并继续受 keyword-only `max_bytes` guard 约束。

原 packed representation 保留为 advanced `basis_words_packed()`，稳定返回 shape `(dimension, ceil(nqubits / 64))`、dtype `uint64` 的二维 array，即使只有一个 limb 也不降为一维。packed contract 固定为 qubit `q` 位于 limb `q // 64` 的 bit `q % 64`；未使用的 tail bits 为零。普通 API 不返回 platform-width integer，也不暴露一维/二维 shape switch。

所有 U1 circuit/sector 的 state、probability、gradient 和 basis arrays 都返回 owned、C-contiguous、read-only NumPy arrays，并在 docstring 中写明 dtype、shape、restricted/full-space distinction 和 ordering。`state_full()` 和 `probability_full()` 的 shape 都固定为 `(2**nqubits,)`，并继续受 circuit construction/plan compilation 时保存的 resource budget 保护。

## 9. Mapping 与 structured operator 命名

`FermionQubitMapping` 的公共名称属性统一为 read-only `name`；删除重复的 `mapping_name` 和 `mapping` 属性，内部 helper 和 plan provenance 也从 `plan.name` 读取。所有接受 fermion-to-qubit mapping 的 public method 和 `compile()` signature 都统一使用 keyword `mapping=`，包括 `HybridOperator.compile()`；删除 `fermion_mapping=`，不保留兼容 alias。`NativeMVPPlan.mapping`/`BackendMVPPlan.mapping` 是编译 provenance 字符串，不是 `FermionQubitMapping` 对象上的重复名称属性，因此保留。

以下方向性方法保留，但返回类型必须在 docstring 中明确：`FermionOperator.map_fermions()` 始终返回 `PauliOperator`；混合空间的 `map_fermions()` 在仍有 boson/qudit axes 时返回 `HybridOperator`，只有完全 qubit-compatible 的空间才可以降为 `PauliOperator`。

`MajoranaOperator.to_fermion()`、`FermionOperator.to_majorana()` 和 `FermionQubitMapping.map_*()` 的方向性命名保留，因为它们描述的是不同的表示边界；不再引入含义重叠的 generic `convert()`。

## 10. 构造器和顶层导出分层

普通构造器只暴露语义完整的输入：`PauliWord.from_string/from_codes`、`PauliOperator.from_terms`、`U1Sector`、`OperatorSpace`、`AdditiveCharge` 和 named mapping factories。operator words/terms、facades 和普通 immutable value/result types 可以继续作为顶层 public 类型，但 quickstart 不把 result container 当作构造入口展示。

以下类型保留为返回值类型和 advanced import，但不再是普通构造路径：`NativeMVPPlan`、`BackendMVPPlan`、`U1MvpPlan`、`ChargeMvpPlan`、`ChargeLazyMvpPlan`、`Z2TaperingPlan`、`U1RestrictedOperator`、`ChargeRestrictedOperator`、`PropagationCircuitPlan`、`SPPSCircuitPlan`、`U1CircuitPlan`、`GateTape`、`PropagationEngine`、`SPPSEngine`、`OperatorBuilder`、array canonicalization result 以及携带 native/internal buffers 的 structured result containers。

Phase 8 新增 public `tencirpauli.advanced` namespace，集中导出上一段中的 raw engine、concrete plan、QIR/packed-array 和 internal-buffer-facing 类型；这些 advanced concrete types 不再从顶层 `tencirpauli.__all__` 导出。顶层继续导出普通入口、operator algebra 类型、`MVPPlan` typing protocol 和常用 immutable value/result 类型，方便类型标注与 `isinstance`。README、首页、quickstart 和 ordinary docstring 只使用顶层普通入口；advanced namespace 的完整签名和 stability boundary 由单独 API reference 页面说明。`tencirpauli._native` 始终是 private module，不能作为 advanced namespace 的替代品。

实现前先在 contract test 中提交一份显式 top-level 与 `tencirpauli.advanced` symbol manifest；后续新增顶层 export 必须经过 API review，避免文档分层与实际 autocomplete surface 再次漂移。

## 11. 构造入口的文档分层

`PauliOperator.from_terms()` 是普通构造主入口，`from_strings()` 是只处理字符串的 convenience，`from_code_arrays()` 是大批量 array-facing convenience，`canonicalize_batch()`、`canonicalize_code_arrays()` 和 `canonicalize_code_arrays_numpy()` 是 backend/advanced canonicalization API。这一项属于文档和 import/export 层面的整理：这些方法的计算语义不需要改变，也不需要为了减少入口而删除；只需在 docstring、API reference 和首页中明确推荐层级。

`compile(target=...)` 是所有 operator family 的统一概念入口。Phase 8 不要求为每种 structured operator 复制所有快捷方法；现有显式 shortcuts 是 `PauliOperator` convenience，其精确对应关系为：

| `PauliOperator` convenience | canonical target/行为 | 等价要求 |
| --- | --- | --- |
| `.dense()` | `.compile(target="dense")` | 返回类型、内容和 ordering 等价 |
| `.coo()` | `.compile(target="coo")` | 返回类型、内容和 ordering 等价 |
| `.csr()` | `.compile(target="csr")` | 返回类型、内容和 ordering 等价 |
| `.native_mvp_plan()` | `.compile(target="native_mvp")` | 返回同类 reusable native plan |
| `.backend_mvp_plan()` | `.compile(target="backend_mvp")` | 返回同类 reusable backend plan |
| `.mvp(state)` | one-shot native MVP | 只要求与 `.compile(target="native_mvp").apply(state)` 数值和 error semantics 等价；不要求相同 setup cost、kernel strategy 或返回协议 |

因此 `.mvp()` 不是第六种 compile target，也不与 `compile()` 返回值一一对应。文档和 benchmark 必须明确区分 one-shot `.mvp(state)`、reusable native plan construction 和 repeated `plan.apply(state)`。所有 public resource budget 参数 `max_bytes` 统一为 keyword-only，包括 `dense/coo/csr/mvp`、plan factories、`compile()`、basis materialization 和 plan `apply()`；`target` 与 `state` 等主要业务参数保持 positional-or-keyword，除非各自合同另有规定。

Phase 0–7.5 文档继续作为历史设计记录保留，不批量重写其中的当时 API 示例；但所有仍含旧 public 名称的阶段规范必须在开头增加统一提示，说明其 API spelling 已被本规范取代并链接到本文。当前 README、quickstart、examples、docstrings 和 API reference 不得继续出现旧名称。这样既保留历史决策，也避免 repository search 和 Agent retrieval 把旧接口误当成现行合同。

## 12. 测试迁移与验收

实现时先增加一组 API contract tests，再批量迁移现有测试、examples 和 docstrings。contract tests 至少覆盖：

- 所有 operator container 的 `term_count`、`len(operator)`、zero-operator truth value、canonical term counts 和统一 code validation，包括 NumPy integer、bool、float、object-array 和大批量 vectorized path；
- 两种 grouping result 的不变量、`term_count`、`group_count`、`term_to_group`、mode defaults 和 QWC reconstruction；
- 四种 MVP plan 的共同 `apply`、`dimension`、algebraic `term_count`、`estimated_bytes`、`target` 和 `__call__` 行为，以及 restricted plan 的独立 `transition_count`；
- construction budget 与 execution budget 的独立作用域、`__call__` 使用全局默认 budget、keyword-only override 和 output ownership/contiguity；
- `SPPSCircuit` 不暴露 deterministic-only terminal 或 `ptm()`，SPPS conversion 在 boundary 拒绝 PTM，Propagation/SPPS 真正共享的 builder 行为不回归；
- 三种 circuit 及 facade plan 的 `parameters=None`、exact Hermitian error contract、validation timing、result types、concurrent-read 与 immutable plan behavior；
- U1 的新构造参数组合、稳定的 `unrank()`/`basis_states()`/`basis_words_packed()` shape、state/probability ordering，以及 `state_full()`/`probability_full()`；
- mapping 的 `name` 与统一 `mapping=` keyword，以及 structured operator 的 conditional return type；
- explicit top-level/advanced export manifest、factory-only plan construction，以及 `compile(target=...)` 与 shortcut 映射表规定的等价性。

完成 API 迁移后运行完整检查：`python scripts/check.py`、`maturin develop --release`、`pytest`、Black、Ruff、strict mypy、Rust fmt/clippy/test，以及所有 doctest。不得只更新测试断言而遗漏 README、examples、docs/vibe 交叉引用和 `_native.pyi`。

## 13. 实施顺序

1. 抽出 capability-aware `_CircuitBuilder`，修复 SPPS terminal 与 PTM 泄漏，先让 facade 的共享结构和 gate-boundary tests 通过。
2. 引入统一 validator、计数属性和 grouping result 字段，迁移所有 Python tests。
3. 收敛 MVP plan 的 algebraic/transition metadata、construction/execution budget、keyword-only `max_bytes`、factory-only constructors 和真实 immutability。
4. 迁移 U1 constructor、restricted/full state 与 sector basis API，再迁移 mapping keyword/property 和 structured return-type docstrings。
5. 增加 `tencirpauli.advanced`、冻结 export manifest 并更新 ordinary/advanced 文档分层；canonicalization 方法本身不改变计算语义。
6. 最后删除旧名称和旧测试引用，执行完整质量门禁，并由独立 reviewer 审查本 spec 与实现的逐项对应关系。

## 14. 非目标

本阶段不增加新的 mapping、不增加 generic operator trait、不引入参数化 Hamiltonian binding、不把 native plan 改造成 JAX primitive，也不把三个 circuit executor 合并成一个 Rust 实现。API coherence 的目标是减少选择和歧义，不是隐藏真实的执行合同。
