# Phase 4 实现规格：Frozen-support reverse gradient 与 SPPS

状态：可执行。Phase 1–3 已完成；本文冻结 2026-08-02 owner 讨论确认的 Phase 4 范围、两类梯度合同、公开接口、随机性语义、性能要求和验收边界。

## 1. 目标与完成定义

Phase 4 在 Phase 3 的 `GateTape`、`PropagationEngine`、packed Pauli key、product-state expectation 和 parameter-slot 基础上交付两个相互独立的 Rust-native value-and-gradient 路径。

第一条路径扩展现有动态稀疏 propagation，提供 deterministic frozen-support reverse gradient。它只对本次前向实际保留的非零 sparse trace 做手写 reverse mode；前向未生成、聚合后严格为零或被 Pauli-weight projection 删除的 term 不进入反向。该路径的目标是低开销地复现当前动态稀疏程序的局部解析反向，不声称等于 fixed-basis dense AD 在 support-change point 的导数。

第二条路径实现 arXiv:2607.17804 的 stochastic Pauli-path simulator（SPPS）。它按 observable Pauli term 独立采样完整 legal path space，使用平滑 proposal、importance reweighting 和 numerically stable path automatic differentiation（PAD）同时估计 value 与全部 parameter-slot gradients。SPPS 不使用 `max_weight`，也不复用 deterministic engine 的 sparse-term aggregation hot representation。

本阶段是框架实现与性能阶段，不承担 weight truncation bias、gradient-direction error、optimization-trajectory deviation 或物理规律研究。完成定义是：两个合同均有独立 reference 和稳定测试，Rust/Python public API 可重复使用，长计算释放 GIL，随机执行可按 seed 重放，release benchmark/profile 覆盖主要路径，并完成本文 REQUIRED 文档与 optional-dependency TensorCircuit adapter 边界。

## 2. Source of truth 与已冻结 owner 决策

实现优先级为：`AGENTS.md` > 已冻结的 `semantics.md` > 本文 > `phase-3-spec.md` > tests > `architecture.md` > 当前实现。本文是 Phase 4 的权威合同，并替代 `architecture.md` 中关于“确定性梯度必须对完整 weight-projected recurrence 精确求导、保留零系数 derivative path、研究相对 exact objective bias”的旧 roadmap 描述。发现其他会改变公开语义的冲突时，继续完成不受影响的工作并在 `implementation-status.md` 记录 blocker，不能由实现 Agent 临时选择新语义。

以下 owner 决策已经冻结：

1. **两条独立路线均为 REQUIRED**：现有 `PropagationEngine` 增加 deterministic frozen-support reverse；新增独立 `SPPSEngine` 实现 stochastic value-and-gradient。二者不能通过一个 mode flag 混成同一内核。
2. **Deterministic target 是 executed sparse trace**：反向只覆盖给定参数下 Phase 3 前向实际生成并保留的 term/contribution。局部 multiplier 或 aggregated coefficient 严格等于 IEEE `0.0` 时沿用前向 exact-zero removal；不为导数额外保留该路径。
3. **Support decisions 不求导**：exact-zero branch skipping、exact-zero aggregated-term removal 和 `max_weight` projection 均作为本次执行的 frozen control flow。返回的是该 frozen trace 上连续乘加的解析反向，不是 support-change point 的 dense/fixed-basis derivative。
4. **手写 local VJP**：内置 Pauli rotations 使用解析 `sin/cos` derivative 和 reverse accumulation；不引入通用 Rust autodiff crate，不在生产 Rust 中实现 parameter shift 或 forward sensitivity。
5. **Parameter shift 仅为外部测试 helper**：Python reference 可在一般非零、远离 support-change 的测试点用 gate-occurrence parameter shift 交叉验证；它不是 public API、Rust fallback 或 performance path。
6. **Checkpointing 属于正式能力**：deterministic reverse 必须支持 block checkpoint/recomputation；checkpoint 选择只改变时间和内存，不得改变 frozen trace value 或 gradient reduction order。
7. **Static custom PTM 边界**：deterministic reverse 支持 Phase 3 的 static real one-/two-qubit PTM，反向使用固定线性 map 的 transpose/VJP；PTM 没有 parameter gradient。Phase 4 不增加 parameterized PTM 或 derivative-PTM public API。
8. **SPPS 严格保留零值 derivative-sensitive branch**：SPPS 的 smoothing 必须满足 `a > 0`。即使 `sin(theta)` 或 `cos(theta)` 为零，对应 branch 仍有非零 proposal probability，并可通过 stable PAD 贡献非零 gradient。
9. **SPPS fixed-budget 保证**：对受支持 gate set，fixed-budget estimator 按论文公式实现，对完整 legal Pauli-path expansion 的 value 和 parameter gradient 在采样期望上无偏。Adaptive A/B 是实用停止 proxy，不表述为 theorem-level confidence interval。
10. **首版 smoothing 是显式固定标量**：public `smoothing` 是 finite positive `float64`，默认 `0.01`。论文中的跨 optimization-step term-wise adaptive smoothing 不属于本阶段。
11. **首版多项 observable 按 term 独立采样**：不做 observable-term sampling、correlated sampling、common-random-number optimization 或 coefficient-dependent term allocation。
12. **Seed 定义随机执行**：每个 SPPS public call 必须显式接收 non-negative 64-bit `seed`。同一 engine、parameters、budget、seed 和 library version 在受支持 thread configurations 下返回可重放结果；线程调度不能改变 path choices 或 reduction order。
13. **SPPS gate set**：支持 Phase 3 fixed Clifford gates 和 `RX/RY/RZ/RXX/RYY/RZZ`，包括 static angles 与 parameter slots。任何 custom PTM 在 `SPPSEngine` construction 时明确失败。
14. **Public hot calls 粗粒度执行**：deterministic 或 SPPS 的一次 value-and-gradient 各自只进行一次主要 PyO3 call，参数一次传入，value/gradient/metadata 一次返回；禁止 per-gate、per-term 或 per-sample FFI。
15. **不做框架级 AD 嵌入**：JAX custom call、custom VJP、PyTorch autograd Function、TensorFlow custom gradient 和任意 Python callback 都不属于 Phase 4。
16. **不做误差规律研究**：本阶段不要求扫描 deterministic gradient 相对未截断 objective 的 bias，不要求优化轨迹实验，也不以论文式物理 benchmark 结论作为验收项。

## 3. 可直接复用的 Phase 3 基础

- 复用 `GateTape` 的 Schrödinger append order、reverse Heisenberg traversal、static/slot parameter reference、连续 slot coverage 和 shared-slot semantics。
- 复用 `PackedKey` 的 inline 128-qubit representation、wide fallback、local Clifford lookup、Pauli generator multiplication、weight popcount 和 product-state expectation。
- 复用 Phase 3 rotation convention `R_P(theta) = exp(-i theta P / 2)`，以及反对易时 `cos(theta) Q + sin(theta) i P Q` 的 exact word/sign rule。
- 复用 Phase 3 canonical input、deterministic aggregation、exact-zero removal、projection-after-aggregation、16 GiB best-effort `max_bytes` 和 checked arithmetic。
- 复用 `PropagationEngine` immutable snapshot、concurrent call safety、GIL release和一次性 native construction boundary。
- 复用 `tests/propagation_reference.py`、matched JAX k-local reference、Criterion propagation target和现有 12q/16q/100q/128q workloads；reference 可以扩展，但不能调用被测 Rust VJP 或 SPPS kernel 生成 expected result。

## 4. Deterministic frozen-support reverse 语义

### 4.1 Forward trace

令 `s_r` 为逆序处理第 `r` 个 gate 前的 canonical sparse state。Deterministic value-and-gradient 必须使用与 `PropagationEngine.expectation()` 相同的 forward recurrence、浮点 dtype、gate order、aggregation order、exact-zero removal 和 projection order。

对每个 gate，forward trace 只记录实际生成并存活的 contributions：

- Clifford gate 对每个 input term 产生一个 signed output；scaled coefficient 严格为零时不保留。
- 反对易 rotation 的 cosine contribution 仅在 `cos(theta) != 0.0` 时生成，sine contribution 仅在 `sin(theta) != 0.0` 时生成；对易 rotation 只生成 unchanged contribution。
- Custom PTM 只遍历 compile time 保存的 exact-nonzero transitions；scaled coefficient 严格为零时不保留。
- 相同 output word 的 contributions 按 Phase 3 的确定性顺序聚合；aggregated coefficient 严格为零时删除。
- 有限 `max_weight` 在完整 aggregation 后删除超重 output word。

`value_and_grad(params).value` 必须由这一次 forward trace 直接计算，并与相同 engine 的 `expectation(params)` 在相同执行配置下相等。实现不得为梯度偷偷改用不同的 zero/pruning 或 projection 语义。

### 4.2 Frozen-support derivative contract

把本次 forward trace 中保留的 contribution edges 记为 `T(params)`。在反向中，`T(params)` 被视为常量 control flow，只对每条 retained edge 的 coefficient arithmetic 求导。

若一个 retained rotation edge 的 local multiplier 为 `m(theta)`，其 derivative 为：

~~~text
cosine branch: m(theta) = cos(theta),  dm/dtheta = -sin(theta)
sine branch:   m(theta) = sign * sin(theta),  dm/dtheta = sign * cos(theta)
commuting path: m(theta) = 1,  dm/dtheta = 0
~~~

其中 `sign` 来自 exact `i P Q` phase rule。对于 slot `p`，反向累计

~~~text
gradient[p] += input_coefficient * dm/dtheta * output_adjoint.
~~~

多个 gate occurrences 引用同一 slot 时，贡献按 reverse gate order 和 canonical input order确定性累加。Static rotations只传播 adjoint，不写 parameter gradient。

如果某个 local branch 因 multiplier 严格为零未进入 `T(params)`，它的 `dm/dtheta` 不参与反向。如果多个 contributions 聚合后严格抵消并删除，该 output 及其 incoming edges 不参与反向。该行为是本阶段明确合同，不作为 bug 或隐藏 coefficient cutoff。

因此在 `sin(theta)=0`、`cos(theta)=0`、exact cancellation、underflow-to-zero 或其他 sparse support 变化点，返回值可以不同于固定完整 basis 上的数学导数，也可以不同于相邻参数极限。文档和 error messages 不得把该结果称为“对完整 projected recurrence 的精确导数”；公开名称统一使用 frozen-support reverse gradient。

### 4.3 Reverse recurrence

对固定 trace，最终 expectation 是 retained final coefficients 的线性 reduction。终点 adjoint由相同 product-state descriptor 对 retained final words 的 expectation 初始化。

反向逐 gate 遍历 retained contribution edges：

~~~text
lambda_input += edge_multiplier * lambda_output
gradient[slot] += input_coefficient * edge_derivative * lambda_output
~~~

Aggregation 的 VJP 将一个 canonical output adjoint分发到所有 retained parent contributions。Projection 已在 forward trace 中决定 retained outputs；反向不为已投影 word 建立 adjoint。Static custom PTM 的 edge derivative 为零，`lambda_input` 使用 `R[out, in]` 对应的 transpose action。

实现可以记录 compact edge/index frames，也可以在 reverse block 内从 checkpointed input state 确定性重建同一 edges。不得保存 Python objects 或在 Rust hot loop 中回调 Python。

### 4.4 Checkpoint 与 recomputation

`checkpoint_interval` 表示相邻保存 forward boundary states 之间的最大 gate 数：

- `checkpoint_interval=1` 保存每个 gate boundary，最少 recomputation、最高 checkpoint memory。
- 任意正整数 `c` 保存 block boundaries，并在反向到达该 block 时从 boundary replay forward，临时保存 block 内 frames。
- `checkpoint_interval=None` 使用 library-chosen deterministic auto strategy。具体 heuristic 不是稳定 public semantic，可以随 profile 改进。

负数、零、bool 或非整数明确失败。空 tape 和无 parameter slot tape 合法，返回 shape `(0,)` 或相应长度的全零 gradient。

不同合法 checkpoint interval 必须生成相同 forward trace、value、gradient 和 canonical reduction order。Best-effort memory estimate至少覆盖 checkpoints、当前 block frames、adjoint buffers、gradient buffer和主要 aggregation workspace；`max_bytes=None` 关闭 guard但不关闭 overflow checks。

### 4.5 Hermiticity、dtype 与异常

Deterministic value-and-gradient只接受 Phase 3 `expectation()` 已接受的 exactly Hermitian observable，并返回 `float64` value 与 contiguous `float64` gradient。一般 complex operator仍可使用 `propagate_operator()`，但不能调用 gradient API。

Runtime parameter vector shape 必须严格为 `(nparameters,)` 且所有值 finite。任何 coefficient/gradient accumulation 产生 NaN 或 Inf 时明确失败，不允许静默 clip、replace 或继续返回部分结果。

## 5. Deterministic public Python API

在现有 `python/tencirpauli/propagation.py` 增加：

~~~python
@dataclass(frozen=True)
class PropagationValueAndGradient:
    value: float
    gradient: np.ndarray


class PropagationEngine:
    def value_and_grad(
        self,
        parameters: Sequence[float] | np.ndarray,
        *,
        checkpoint_interval: int | None = None,
    ) -> PropagationValueAndGradient: ...
~~~

`gradient` shape 固定为 `(engine.nparameters,)`，dtype为 `float64`、C-contiguous且只读。API 不额外提供 `gradient()` convenience method，避免为同一计算建立第二条路径。`expectation()`、`propagate_operator()` 和 `profile()` 的 Phase 3 行为保持不变。

首版不公开 reverse frames、adjoint operators、per-gate gradients、observable-coefficient gradients或 checkpoint contents。性能诊断通过 benchmark/internal stats 完成；若未来需要 public gradient profile，另行设计结果类型，不能改变 `PropagationValueAndGradient` 字段含义。

## 6. SPPS 数学与数值语义

### 6.1 支持的 objective

`SPPSEngine` 接受 exactly Hermitian canonical `PauliOperator` 和 Phase 3 product-state descriptor。Observable coefficients 必须为 finite real `float64`；每个 canonical nonzero Pauli term独立采样，最后线性组合。

SPPS 不接受 `max_weight`，不应用 Pauli-weight projection，也不使用 deterministic forward的 exact-zero path pruning。它估计由完整 Clifford/Pauli-rotation tape 定义的 legal Pauli-path expansion。Custom PTM、一般 channel、任意 unitary、dynamic measurement 和 Python callback 在 construction 时失败。

Static rotation参与 path sampling和value weight，但不产生 parameter gradient。多个 parameterized rotation occurrences 共享同一 slot 时，每条 path 上各 occurrence 的 PAD contribution累加到同一 gradient entry。

### 6.2 Sequential path sampling

对一个 observable Pauli term 和一条 sample path，从 observable word开始逆序遍历 `GateTape`：

- Clifford gate确定性映射 current word并累计 exact sign。
- 若 current word 与 rotation generator commute，则 word不变、branch label为0、conditional probability为1。
- 若 anti-commute，cosine branch label为`+1`，sine branch label为`-1`，并使用

~~~text
q(theta) = (abs(cos(theta)) + a)
           / (abs(cos(theta)) + abs(sin(theta)) + 2*a)
Pr(cosine) = q(theta)
Pr(sine) = 1 - q(theta)
~~~

其中 `a = smoothing > 0`。Cosine branch保持word，local factor为`cos(theta)`；sine branch更新为 exact `i P Q` word/sign，local factor为`sign * sin(theta)`。

即使 local factor严格为零，sampled branch仍保留在 path record中。Proposal probability不得因 factor 为零而变成零；这是 SPPS 与 deterministic frozen-support engine 的有意差异。

### 6.3 Importance-reweighted value

对 path `omega`，令 `Psi_omega` 为全部 active local factors的乘积，`Pr(omega)` 为 conditional probabilities的乘积，`e(P_omega)` 为最终 product-state expectation，`c_m` 为 observable term coefficient。单路径 value sample为

~~~text
h_tilde = c_m * Psi_omega * e(P_omega) / Pr(omega).
~~~

Fixed-budget mode 对每个 observable term独立生成相同数量 `B` 的 paths，分别求均值后按 term求和。不同 observable terms和不同 path samples使用相互独立的 counter domains。

Proposal `Pr(omega)` 只用于重要性校正；PAD 不对 `q(theta)` 或 sampling decision 求导。实现不得把 proposal derivative混入 parameter gradient。

### 6.4 Stable PAD

对 path 中第 `j` 个 active rotation occurrence，局部 factor和derivative为：

~~~text
cosine branch: psi_j = cos(theta_j),          dpsi_j = -sin(theta_j)
sine branch:   psi_j = sign * sin(theta_j),   dpsi_j = sign * cos(theta_j)
~~~

若 occurrence引用 slot `p`，单路径 gradient contribution为

~~~text
g_tilde[p] += c_m * e(P_omega) / Pr(omega)
              * dpsi_j * product(psi_k for k != j).
~~~

实现可以在 active factor安全远离零时使用等价 score form，但在零点或内部稳定阈值附近必须使用 prefix/suffix product，不能计算 `0 * inf`、`tan`/`cot` overflow 或用 epsilon 改写 estimator。稳定阈值是内部数值策略，不得改变 branch probability、path set或数学 estimator。

所有 active factors，包括 static rotation factors，都参与 prefix/suffix product；只有 parameter-slot occurrences写 gradient。若最终 product-state expectation严格为零，可以在不改变结果的前提下跳过该 path 的 value/gradient accumulation。

若 importance weight 或 accumulator 超出 finite `float64`，调用明确失败并包含 term/sample context；不允许 clipping、winsorization或静默丢弃 sample。

### 6.5 Fixed-budget 与 adaptive A/B

Fixed-budget mode的 `samples_per_term=B` 必须是至少为2的整数且bool不合法。它使用一个 replicate，每个 term恰好生成 `B` 条 paths。返回的 value 与 gradient来自同一组 samples；二者可以相关。

Adaptive mode 对每个 term维护两个独立 macro-replicates `A/B`，每个 replicate从 `initial_samples_per_term` 开始。令 `g_m,A` 与 `g_m,B` 已包含 observable coefficient `c_m`，term proxy为

~~~text
Delta_m = 0.5 * norm(g_m,A - g_m,B, ord=2).
~~~

若 canonical observable包含 `N` 个 nonzero terms，term target为 `gradient_tolerance / sqrt(N)`。未通过的 term将两个 replicate的累计 sample budget翻倍，直到通过或达到 `max_samples_per_term`；最后一次不足整倍时取max值。新预算通过追加独立 samples扩展已有 replicate，不丢弃已完成 samples。

最终 adaptive value/gradient使用每个 term的 `A/B` replicate均值。Observable-level proxy为` sqrt(sum(Delta_m**2))`；只有所有 terms在到达max之前通过时 `converged=True`。该 proxy是paper-style empirical stopping statistic，不是confidence interval。Fixed-budget estimator具有无偏性合同；adaptive stopping结果只承诺遵循上述算法，不额外声明optional-stopping theorem。

`initial_samples_per_term` 至少为2，`max_samples_per_term >= initial_samples_per_term`，`gradient_tolerance` 必须finite且大于0。Empty observable合法并立即返回 value 0、zero gradient、zero paths和converged状态。

### 6.6 Seeded deterministic replay 与并行归约

每个 random variate必须由稳定坐标派生，坐标至少包含：public seed、mode、observable term canonical index、macro-replicate index、sample index和reverse gate index。实现可以使用counter-based generator或从该坐标构造独立 stream，但不能依赖Rayon worker identity、任务抢占顺序或前一条path消费了多少random numbers。

Paths按固定 sample-index chunks并行执行；每个chunk使用worker-local value、second-moment和gradient accumulators。Chunk结果按chunk index固定顺序归约。相同seed/config在1/2/4/固定最大线程配置下必须产生相同path choices和public result；若浮点bitwise equality在特定平台不可实现，任何放宽到tolerance的变更必须先取得新的owner decision。

## 7. SPPS public Python API

新增 `python/tencirpauli/spps.py`，顶层导出稳定类型：

~~~python
@dataclass(frozen=True)
class SPPSEstimate:
    value: float
    gradient: np.ndarray
    value_standard_error: float
    replicates: int
    samples_per_replicate: tuple[int, ...]
    total_paths: int
    seed: int
    gradient_error_proxy: float | None
    term_gradient_error_proxies: tuple[float, ...] | None
    converged: bool | None


class SPPSEngine:
    def __init__(
        self,
        tape: GateTape,
        observable: PauliOperator,
        *,
        initial_state: ZeroState | ComputationalBasisState | ProductBlochState | str = ZeroState(),
        smoothing: float = 0.01,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
    ) -> None: ...

    def value_and_grad(
        self,
        parameters: Sequence[float] | np.ndarray,
        *,
        samples_per_term: int,
        seed: int,
    ) -> SPPSEstimate: ...

    def value_and_grad_adaptive(
        self,
        parameters: Sequence[float] | np.ndarray,
        *,
        initial_samples_per_term: int,
        max_samples_per_term: int,
        gradient_tolerance: float,
        seed: int,
    ) -> SPPSEstimate: ...
~~~

`samples_per_replicate`按canonical observable term order返回：fixed mode中`replicates=1`且每项为`B`；adaptive mode中`replicates=2`且每项为该term每个macro-replicate的最终累计budget。`total_paths = replicates * sum(samples_per_replicate)`。

`gradient`是shape `(nparameters,)` 的只读 contiguous `float64` array。`value_standard_error`由独立 term sample moments按线性组合规则估算，每个 term/replicate 使用至少两个样本的 `N-1` sample-variance denominator；empty observable返回0。Fixed mode的proxy/converged fields为`None`；adaptive mode返回global/term proxies和bool。

Engine construction获取tape、observable和state的immutable snapshot。Public properties至少包括 `nqubits`、`nparameters`、`gate_count`、`observable_terms` 和 `smoothing`。本阶段不提供SPPS propagated operator materialization、raw sampled paths、per-sample gradients或持久化 RNG state。

## 8. Rust core、PyO3 与数据模型

### 8.1 Module boundary

Pure Rust算法继续位于`tencir-pauli-core`。建议把Phase 3的gate-local mapping、packed key和product expectation调整为crate-private shared kernels，并将deterministic reverse与SPPS分别放入清晰模块；不得复制两套word/sign convention。

`tencirpauli-native`只负责batched conversion、immutable handle、NumPy gradient allocation/result transfer、GIL release和typed error mapping。Public validation、dataclasses和optional TensorCircuit adapter位于Python package。

### 8.2 Deterministic reverse storage

Deterministic path可使用canonical sparse frames：sorted unique keys、real coefficients、retained transition/output indices和可选local derivative metadata。实现应优先在reverse block内重建可廉价重建的edge metadata，避免跨全部tape保存重复full-width keys。

Adjoint storage只覆盖forward trace中retained canonical keys。Custom PTM VJP可以遍历forward input transitions并gather output adjoint，不要求另存完整dense transpose。所有hash iteration在公开reduction前转为canonical deterministic order。

### 8.3 SPPS path/batch storage

每次public execution必须先为每个rotation occurrence解析一次angle、`sin`、`cos`和proposal `q`，static angle结果可在engine construction预计算；这些值供全部observable terms和samples复用，禁止在per-sample loop重复执行trigonometric functions。每条SPPS sample至少携带一个packed current word、累计sign/probability信息和active factor records。不得为每条sample分配长度`nparameters`的gradient vector；按fixed-size batch使用worker-local gradient accumulator，并把active occurrences直接scatter-add到slot。

总sample budget应stream/batch执行，使主要path workspace与batch size和thread count相关，而不与全部`B * N_terms`线性常驻。`max_bytes`估计至少覆盖observable/tape snapshot、worker gradients、path records、prefix/suffix scratch和deterministic chunk results。

### 8.4 Concurrency 与 panic boundary

两个engine均保持immutable、`Send + Sync`兼容设计；并发public calls不得共享可变scratch、RNG stream或长持有global lock。长调用释放GIL。Rust panic不得穿过PyO3 boundary；invalid budget、seed overflow、unsupported gate、nonfinite arithmetic、allocation overflow和memory guard必须映射为清晰Python异常。

## 9. 独立 reference 与 correctness matrix

### 9.1 Deterministic reference

扩展Python reference以明确复刻frozen-support trace：相同reverse gate order、local branch skipping、deterministic duplicate aggregation、exact-zero deletion和per-gate weight projection。Reference不能调用native VJP。

Parameter-shift helper只用于一般测试点：所有被shifted occurrences的`abs(sin)`和`abs(cos)`远离零，fixture不得发生exact cancellation或support change；shared slot通过逐gate occurrence shift并累加。它不进入public package、benchmark或fallback。

至少覆盖：

- 每个RX/RY/RZ/RXX/RYY/RZZ在one-/two-qubit local Pauli basis上的retained-edge value/VJP。
- mixed static/slot rotations、多个gate共享slot、unused effect due commuting path和empty parameter vector。
- exact与finite `max_weight`，包括Clifford改变weight后删除output。
- duplicate aggregation、static custom PTM transpose VJP和product Bloch state。
- `checkpoint_interval=1`、多个block sizes和auto strategy返回相同结果。
- `value_and_grad().value == expectation()`，重复执行和并发执行确定性。
- 零角度或exact cancellation fixture明确验证frozen-support行为，而不是与dense AD比较。

### 9.2 SPPS exact path reference

对小系统/短tape建立独立legal-path enumerator，枚举每个anti-commuting rotation的cosine/sine choices，直接计算每条path的probability、importance-reweighted value和PAD gradient，并对proposal distribution精确求和。该reference用于无flaky Monte Carlo地验证fixed-budget single-sample estimator expectation。

至少覆盖：

- 单个与多个anti-commuting rotations、interleaved Clifford、one-/two-qubit rotations和shared slots。
- static rotation factors参与PAD product但不写gradient。
- `sin(theta)=0`、`cos(theta)=0`及邻近点，验证`a>0`、zero-factor path和prefix/suffix finite结果。
- Zero/computational/Bloch product states、positive/negative multi-term observable coefficients和empty observable。
- explicit seed golden replay、不同thread counts相同输出和并发calls互不污染。
- fixed/adaptive budget validation、budget doubling、max-budget stop、proxy aggregation和reported path counts。
- custom PTM和所有unsupported inputs在construction时明确失败。

允许增加固定seed Monte Carlo end-to-end smoke，但statistical test必须使用预先固定seed/budget/tolerance且避免偶发CI失败。Exact path enumeration是数学正确性的主要gate。

### 9.3 Quality gate

Phase 1–3 regression必须全部通过。Phase 4仍执行Rustfmt、Clippy workspace/all-targets/all-features `-D warnings`、workspace tests、Black、Ruff、strict mypy、`maturin develop --release --locked`和pytest。新增依赖必须保持MSRV 1.85兼容并在lockfile中固定。

## 10. 性能设计与 benchmark 合同

### 10.1 Timed boundaries

每个public benchmark至少分开记录：engine construction、first call、steady value-and-gradient、Python-to-Rust parameter conversion、Rust-to-NumPy gradient return、major estimated/observed memory和result validation。不得把只返回scalar的Phase 3 timing当作gradient timing。

Deterministic benchmark额外记录checkpoint interval、replayed gates、initial/final/peak retained terms和gradient length。SPPS benchmark额外记录observable terms、samples per replicate、total paths、active rotations/path、paths per second、thread count、smoothing、proxy和value/gradient validation。

### 10.2 REQUIRED deterministic workloads

1. **Local VJP microbench**：one-/two-qubit rotation retained edge、shared-slot accumulation、Clifford/PTM adjoint和product expectation terminal reduction。
2. **12q TFIM-style PPE**：复用Phase 3参数化workload，`max_weight=2/3/4`，比较scalar forward、frozen reverse和matched JAX warm gradient at generic nonsingular parameters。
3. **2D Heisenberg/Trotter**：4x4 RXX/RYY/RZZ tape，覆盖two-qubit rotations、aggregation和checkpoint replay。
4. **100q near-Clifford**：少量parameterized rotations、local observable sum和finite `max_weight`，验证large packed keys与小gradient vector的端到端latency。
5. **Duplicate/PTM synthetic**：大量retained collisions和static dense/sparse PTM，定位aggregation VJP和transpose gather成本。
6. **Checkpoint scaling**：固定workload扫描gate depth和checkpoint intervals，报告runtime/estimated peak memory，不设置wall-time CI gate。

### 10.3 REQUIRED SPPS workloads

1. **Fixed-budget throughput**：12q multi-term和100q near-Clifford circuits，扫描budget、observable term count、parameter count和1/2/4/固定最大threads。
2. **Rotation-heavy path**：足够多anti-commuting rotations以覆盖active-record、prefix/suffix和importance arithmetic热点。
3. **Adaptive A/B**：固定seed/tolerance/min/max budgets，记录term budgets、total paths、proxy和max-budget cases。
4. **Zero-factor stable PAD**：包含exact trigonometric zeros的bounded workload，只验证finite correctness和kernel overhead，不把它扩展为误差规律研究。
5. **Scaling**：扫描qubits、tape depth、terms和sample budget，报告paths/s与peak workspace；所有本机case运行前遵守`AGENTS.local.md`完整peak不超过16 GiB的限制。

### 10.4 性能策略

性能不设置预先固定的speedup pass/fail倍数。Correctness gate通过后，必须对representative release workload profile，优先消除per-path allocation、full-width key cloning、per-sample gradient allocation、hash rebuild、checkpoint over-retention、RNG contention、non-deterministic merge和FFI materialization成本。

Deterministic主要对照是语义可对齐、generic nonsingular parameters下的matched JAX CPU warm-JIT value-and-gradient；setup/cold compile另报。SPPS没有可直接等价的JAX性能基线时，以独立Python path reference、Rust single-thread control和parallel scaling为主，不制造不公平speedup claim。

每个material optimization保留Criterion或pytest-benchmark source、同机release compare和profile结论；`.benchmarks/`与raw profiles不提交。性能结果informational，不加入wall-time CI gate。

## 11. TensorCircuit optional-dependency integration

Phase 4最后增加TenCirPauli侧的lazy optional adapter，不修改相邻TensorCircuit源码，也不要求TensorCircuit基础安装依赖Rust wheel。

建议public shape位于`tencirpauli.integrations.tensorcircuit`：

~~~python
@dataclass(frozen=True)
class TensorCircuitTapeConversion:
    tape: GateTape
    parameters: tuple[Any, ...]


def gate_tape_from_circuit(
    circuit: Any,
    *,
    parameter_order: Sequence[Any] | None = None,
) -> TensorCircuitTapeConversion: ...
~~~

Adapter通过`circuit.to_qir()`读取supported gates。普通numeric `Circuit`的angles编译为static；`SymbolCircuit`中直接的`sympy.Symbol`映射为slots，重复symbol复用slot。若未给`parameter_order`，按QIR首次出现顺序确定；显式order必须恰好覆盖全部symbols且无重复。

首版只接受单个direct symbol或finite numeric angle；symbolic expressions、affine combinations、backend tracer、channels、measurements、controls、unsupported gate names和arbitrary matrix gates明确失败。Adapter返回symbol tuple供调用者构造runtime parameter vector，不尝试把native result接入`tc.backend.grad/jit`。

Integration tests固定受支持TensorCircuit版本范围，覆盖gate naming、angle convention、wire order、QIR order、shared symbols和missing optional dependency。TensorCircuit不可用时core package和两个native engines仍可导入使用。

## 12. 实现切片

### P0：合同、reference 与 API skeleton

- 增加frozen-support Python reference、bounded parameter-shift test helper和SPPS exact path enumerator。
- 增加private Python result/API skeleton和失败测试；完成对应native纵向切片后再顶层导出。
- 固定seed/budget/proxy fixtures和benchmark workload definitions。
- 同步`architecture.md`、`implementation-status.md`和docs index，不再保留旧bias-study任务。

Acceptance gate：references独立；zero-support行为、proposal/PAD公式、shared slots和adaptive counters都有明确expected vectors；Phase 1–3 tests保持通过。

### P1：Deterministic frozen-support reverse

- 在pure Rust core实现retained trace frames、terminal adjoint、local analytic VJP、aggregation reverse、shared-slot accumulation和static PTM transpose action。
- 先完成`checkpoint_interval=1` correctness纵向切片，再贯通batched PyO3和typed `PropagationValueAndGradient`。
- 覆盖exact/finite-weight、zero/cancellation contract、concurrency和nonfinite errors。
- 增加local VJP Criterion和public steady benchmark。

Acceptance gate：value与Phase 3 expectation一致；gradient与frozen-support reference和generic parameter-shift fixtures一致；没有per-gate FFI或通用AD依赖。

### P2：Checkpoint、memory 与 deterministic performance

- 实现block boundary checkpoints、deterministic replay和auto strategy。
- 完成cheap major allocation estimates、best-effort guard和overflow cases。
- 对12q/16q/100q workloads profile并至少优化一个实际dominant bottleneck。
- 验证不同checkpoint intervals与thread configs结果一致。

Acceptance gate：checkpoint显著改变memory/time选择而不改变结果；release benchmark和profile记录timed boundary、peak estimate和优化证据。

### P3：SPPS fixed-budget engine

- 实现immutable SPPS tape validation、term-wise path sampling、importance-reweighted value、stable PAD和shared-slot scatter。
- 实现explicit seed/counter domains、streamed batches、worker-local accumulators和deterministic chunk reduction。
- 贯通single-call PyO3与typed `SPPSEstimate` fixed mode。
- 对exact path enumerator验证value/gradient estimator，覆盖trigonometric zeros。

Acceptance gate：fixed-budget数学合同、seed replay、thread independence、product states和unsupported PTM failure全部通过；hot path无per-sample allocation/FFI。

### P4：Adaptive A/B、SPPS performance 与 diagnostics

- 实现two-replicate cumulative budgets、term-wise doubling、weighted proxy、max-budget stop和value standard error。
- 增加adaptive boundary/error tests、parallel scaling和rotation-heavy benchmarks。
- Profile RNG、path key updates、active-factor storage、prefix/suffix和gradient merge，完成至少一轮profile-backed优化。
- 文档明确fixed unbiasedness与adaptive empirical proxy边界。

Acceptance gate：reported budgets/path counts/proxies与reference一致；同seed跨线程可重放；release paths/s、memory和scaling evidence完整。

### P5：Optional-dependency TensorCircuit adapter 与public交付

- 实现numeric Circuit/direct-symbol SymbolCircuit QIR adapter和compatibility tests。
- 顶层导出`PropagationValueAndGradient`、`SPPSEngine`、`SPPSEstimate`；adapter类型保持integration module边界。
- 更新README、CHANGELOG、public docstrings、typing、docs index和implementation status。
- 运行完整quality workflow、packaging smoke和最终benchmark record/compare。

Acceptance gate：新环境release安装后可运行deterministic与SPPS examples；optional TensorCircuit缺失时明确失败且不影响core import；全部文档与实际public surface一致。

## 13. 非目标

- 不实现对完整weight-projected fixed-basis recurrence的严格导数，不额外保留forward coefficient为零的deterministic paths。
- 不研究deterministic gradient相对exact circuit的bias、direction error、optimization trajectory或物理规律。
- 不在Rust/public API实现parameter shift、finite difference、forward sensitivity或通用automatic differentiation。
- 不提供observable coefficients、initial-state components、custom PTM entries或gate-generator coefficients的gradient。
- 不支持parameterized custom PTM、custom SPPS transition rule、arbitrary unitary、channel或measurement。
- 不实现adaptive smoothing、observable-term sampling、correlated sampling、control variates或跨optimization-step stateful sampler。
- 不实现JAX/PyTorch/TensorFlow native gradient bridge、GPU kernel或distributed sampling。
- 不修改TensorCircuit上游源码，不新增`tc.pauli`顶层入口。
- 不把SPPS A/B proxy表述成confidence interval，也不承诺任意circuit/sample budget上的低方差。

## 14. 一次性验收标准

Phase 4完成时至少证明：

- Deterministic value-and-gradient严格遵循本文frozen-support contract，value与现有expectation一致，gradient与独立reference一致。
- Zero multiplier、exact cancellation和projection deletion不被deterministic reverse重新引入；文档不把该结果误称为dense projected AD。
- Local analytic VJP、shared slots、static PTM transpose、checkpoint replay、memory guard、nonfinite failure和concurrent calls均有tests。
- SPPS fixed-budget estimator按论文公式执行，proposal不参与求导，`a>0`零值branch和stable PAD有exact path enumeration证据。
- SPPS fixed/adaptive APIs的seed、budgets、replicates、total paths、standard error、proxy和converged fields语义稳定且可重放。
- 同seed/config在受支持thread counts下返回相同public results；long calls释放GIL且无global mutable RNG/scratch。
- 两个hot APIs各只做一次主要FFI call，gradient一次返回NumPy，SPPS无per-path FFI。
- Rust/Python full quality workflow和Phase 1–3 regression全部通过。
- Criterion/pytest-benchmark覆盖本文REQUIRED workloads；release profile定位dominant bottleneck并有至少一轮correctness-gated优化。
- 最终本机benchmark label、commands、thread config、workload、主要结果、known limitations和下一步写入`implementation-status.md`。
- README、CHANGELOG、typing、docstrings、architecture/status/index和optional TensorCircuit compatibility与实现一致。

## 15. 给下一位实现模型的执行顺序

从P0开始按“独立reference/失败测试 → pure Rust kernel → batched PyO3 → typed Python facade → focused correctness → release benchmark/profile → status evidence”的纵向切片推进。同一时间只把最早未完成slice标为active；不要先铺设所有public placeholders。

Deterministic实现先复用Phase 3 exact sparse trace，不得擅自恢复旧roadmap的zero-path structural support。SPPS实现不得为了复用deterministic kernel而跳过零值sampled branch、加入`max_weight`或把proposal derivative混进PAD。

性能优化必须保持本文语义。任何unsafe、新hash/RNG依赖、scratch pool、Rayon threshold或specialized key layout都需要dedicated tests和representative end-to-end release evidence。每完成一个slice，更新active milestone、精确verification结果、benchmark/profile结论、known limitations和下一步。
