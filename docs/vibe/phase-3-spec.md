# Phase 3 实现规格：Rust-native Pauli propagation

状态：可执行。Phase 1 与 Phase 2 已完成；本文冻结 2026-08-02 owner 讨论确认的 Phase 3 范围、公开接口、数值语义、性能方向和验收合同。

## 1. 目标与完成定义

Phase 3 交付一个可从 Python 端直接使用、核心计算完全在 Rust 中执行的动态 Pauli propagation engine。用户显式构造 `GateTape`，将一个 canonical `PauliOperator` 按 Heisenberg picture 传播，并选择不做结构投影的 exact recurrence，或在每个 gate 后应用 Pauli-weight projection。默认热路径直接在 Rust 中计算 product-state expectation，只返回一个标量；只有用户明确请求时才把完整传播后算符跨 FFI 物化回 Python。

传播由一个统一 recurrence 定义。Clifford gate 自然走不分支的 exact fast path；`max_weight=None` 或 `max_weight >= nqubits` 得到 exact propagation；有限 `max_weight` 在初始 canonical operator 和每个 gate 的全部贡献完成聚合后保留 weight 不超过阈值的 Pauli words。

正确性是硬门槛，性能是本阶段的核心工程目标。性能不设置“达到某个倍数即可停止”的数字门槛，也不因初版落后就取消路线；实现必须持续对同步后的 Python/JAX warm-JIT steady runtime 做等价语义比较，profile 真实瓶颈，优先消除算法复杂度、数据布局、per-term allocation、scratch 重建、并行调度和 FFI materialization 成本，并保留每个 material hot path 的 release benchmark。

Phase 3 的完成定义包括本文所有 REQUIRED deliverables、独立 reference、Rust/Python correctness、public typing/docs、release benchmark、profile/optimization evidence、warm-JIT 对照、100-qubit packed-key 路径、custom real PTM、product-state expectation、operator materialization 和完整 handoff 状态。

## 2. Source of truth 与已冻结 owner 决策

实现优先级为：`AGENTS.md` > 已冻结的 `semantics.md` > 本文 > `reference-vectors.md` 与新增 Phase 3 reference > tests > `architecture.md` > 当前实现。发现会改变公开语义的冲突时，继续完成不受影响的工作并在 `implementation-status.md` 记录 blocker，不能由实现 Agent 临时选择新语义。

以下 Phase 3 owner 决策已经冻结，不是实现过程中的开放选项：

1. **统一 recurrence**：公开结构参数是 `max_weight`。`None` 或不小于 `nqubits` 表示 exact；有限 cutoff 表示相同 recurrence 中的 Pauli-weight projection。
2. **projection 顺序**：初始 operator 先完成 canonical aggregation，再应用 projection；每个 gate 产生的相同 Pauli word 全部完成聚合后，再按 weight 保留 canonical terms。Operator structure 只由 Pauli weight 决定。
3. **首批内置 gates**：固定 Clifford gates 为 `X/Y/Z/H/S/Sdg/CNOT/CZ/SWAP`；参数化 gates 为 `RX/RY/RZ/RXX/RYY/RZZ`。
4. **显式 GateTape**：Phase 3 使用 typed Python builder 和 parameter slots 作为 canonical circuit input。
5. **Real PTM**：custom PTM 使用 Hermitian Pauli basis，公开输入为 finite `float64` real arrays，允许任意正负实数 entry。
6. **初态范围**：支持 `|0...0>`、任意 computational-basis product state，以及 pure/mixed tensor-product single-qubit Bloch vectors。Expectation 在 Rust 中按 product structure 计算。
7. **Rust 内 expectation**：默认性能路径在 Rust 中完成传播与 expectation，只返回 `float64`；完整 operator 使用显式、单独计时的 materialization 路径。
8. **统一 16 GiB 默认预算**：所有现有和新增 public `max_bytes` 默认改为 `16 * 1024**3`，语义是 major output/workspace 的 best-effort guard；调用者可提高或显式传 `None` 关闭，checked dimension/arithmetic overflow 始终保留，系统 OOM 仍可能发生。
9. **持续性能优化**：主要对照是同步后的 JAX CPU warm-JIT steady runtime；cold compile、setup、first execution、memory 与 accuracy 另行报告。每个 material hot path 都保留 profile 和 release benchmark evidence。
10. **后续梯度兼容性**：GateTape parameter slots、local analytic rules 和 native handle 需要同时支撑 projected-recurrence deterministic gradient 与 SPPS stochastic gradient 的后续实现。

上述语义已冻结；实现细节根据 profile 选择，并保持这些公开合同不变。

## 3. 可直接复用的基础

- 直接复用 Phase 1 的 phase-free `PauliWord`、exact `PauliPhase`、canonical `PauliOperator`、complex128 coefficient、deterministic aggregation、packed qubit-zero-is-LSB internal representation 和 TensorCircuit qubit-zero-is-MSB matrix boundary。
- Phase 1 的 `from_code_arrays()`、private native array boundary 和 exact-zero static canonicalization直接服务于一次性 propagation setup。
- Phase 2 已验证的 Z2/U1 模块和既有 tapering row-sign、U1 width boundary、best-effort memory、runtime-only plan serialization 语义保持稳定。
- Core 继续保持 pure Rust；native crate 只做 batched PyO3 转换；public API 继续位于 `python/tencirpauli/`，`_native` 是 private implementation detail。
- Phase 1–2 的 format、Clippy、test、Black、Ruff、strict mypy、maturin、pytest、benchmark manifest 和 packaging 规则继续有效。本 Spec 只列 Phase 3 新增证据。

## 4. 数学与传播语义

### 4.1 GateTape 顺序与 Heisenberg direction

`GateTape` 按 Schrödinger circuit 的执行顺序 append gates。若用户依次 append `G0, G1, ..., G(L-1)`，state evolution 为 `U = G(L-1) ... G1 G0`。Propagation 从 observable `O` 开始逆序遍历 tape，计算

~~~text
O_final = U† O U
        = G0† ... G(L-1)† O G(L-1) ... G0.
~~~

因此最后 append 的 gate 最先作用于 Heisenberg observable。任何 TensorCircuit differential reference 必须使用同一 convention，不能把 forward tape iteration 与 reverse Heisenberg iteration混淆。

### 4.2 Pauli rotation convention

所有参数化 gate 使用

~~~text
R_P(theta) = exp(-i theta P / 2).
~~~

当当前 Pauli word `Q` 与 generator `P` 对易时，`R_P(theta)† Q R_P(theta) = Q`。反对易时，

~~~text
R_P(theta)† Q R_P(theta)
    = cos(theta) Q + sin(theta) i P Q.
~~~

`i P Q` 通过现有 exact `PauliPhase` multiplication 转成一个 canonical Hermitian Pauli word 和实数符号。每个 rotation gate 的 `sin`/`cos` 在每次 engine execution 中计算一次并供全部 terms 复用。

`RX/RY/RZ` 分别使用单 qubit `X/Y/Z` generator；`RXX/RYY/RZZ` 使用两个指定 wires 上的对应二体 Pauli generator。Rotation angle 不做周期归一化，必须是 finite `float64`。

### 4.3 单一 recurrence 与 weight projection

令 `A0` 为输入 `PauliOperator` 完成 canonical duplicate aggregation 后的结果，`M_r` 为逆序遍历到的第 `r` 个 local gate map，`Pi_k` 为只保留 Pauli weight 不超过 `k` 的结构投影。Phase 3 recurrence 定义为

~~~text
s0 = A0                              if max_weight is None or >= nqubits
s0 = Pi_k(A0)                        otherwise

s(r+1) = Aggregate(M_r(s(r)))        if exact
s(r+1) = Pi_k(Aggregate(M_r(s(r))))  otherwise.
~~~

`Aggregate` 包括相同 canonical word 的确定性 coefficient reduction 和 aggregated exact-zero removal。Projection 在完整 aggregation 结果上按 word weight执行，且不依赖 coefficient 数值。有限 `max_weight=0` 合法，只保留 identity；任意非负 `max_weight >= nqubits` 与 `None` 语义相同。负数、bool 或非整数返回明确输入错误。

Clifford gate 仍属于同一 recurrence。它对单个 Pauli word 产生唯一 word/sign，可能改变 Pauli weight，因此有限 `max_weight` 时同样在该 gate 后应用 projection。Exact tape 中若所有 gates 都是 Clifford，term 数保持不因branching增长，runtime自动使用specialized fast path。

### 4.4 Determinism 与浮点 reduction

公开 `propagate_operator()` 结果按现有 canonical external code order 排序，coefficients 在相同输入、参数和 library version 下对受支持 thread configurations 可复现。公开 arrays、operator terms和structural stats均由canonical order生成；显式 profiling 返回的wall-clock timing作为运行环境测量值处理。

并行 aggregation 必须定义 deterministic merge/reduction order。允许 worker-local maps、sorted contribution buffers 或其他结构，但不能让线程完成顺序改变公开 coefficient。若为性能采用不同 thread count 后只保证数值 tolerance 而非 bitwise coefficient 相等，必须先取得新的 owner decision；默认合同是 public deterministic result。

### 4.5 Custom real PTM

Phase 3 custom PTM 支持 one- or two-qubit static local maps。单 qubit basis 顺序为 `[I, X, Y, Z]`；对 wires `(q0, q1)`，two-qubit local basis index 为 `4 * code(q0) + code(q1)`，即 `[II, IX, IY, IZ, XI, ..., ZZ]`，wire 顺序严格使用调用者给出的顺序。

PTM orientation 固定为

~~~text
R[out, in] = 2**(-m) Tr[P_out E(P_in)],
new_local_coefficients = R @ old_local_coefficients,
~~~

其中 `E(P) = U† P U` 对 unitary gate 成立，`m` 为 local wire 数。单比特 shape 为 `(4, 4)`，双比特 shape 为 `(16, 16)`。输入合同是real `float64`、C-contiguous或可一次性转换为contiguous的finite array，以及互异且在范围内的wires。

Hermitian Pauli basis 下，Hermiticity-preserving map 的 PTM 为实数，负 entry 完全合法。Phase 3 将 custom matrix 解释为调用者提供的 real linear Pauli map，并执行 dtype、shape、wire和finite-value validation。

Custom PTM 在 engine construction 时复制到 native immutable storage，并按每个 input local Pauli code 预编译全部 exact-nonzero output transitions，包括数值很小的非零 entries。Built-in Clifford 和 rotation gates 使用 specialized lookup/analytic rules，以保持主要路径性能。

### 4.6 Product-state expectation

`expectation()` 面向 Hermitian observable，并返回 Python `float`/native `f64`。在 Hermitian Pauli basis 中，Hermitian operator 的 canonical coefficients 为实数；expectation compilation 使用现有 Hermiticity validation 的显式 `atol=0.0` 合同。`propagate_operator()` 接受一般 complex128 `PauliOperator`，覆盖通用operator propagation。

三类 initial-state descriptor 的 expectation 规则为：

- `ZeroState`：每个 local `I/Z` expectation 为 `1`，`X/Y` 为 `0`。
- `ComputationalBasisState(bits)`：`bits` 长度必须等于 `nqubits` 且每项是整数 `0` 或 `1`；qubit `q` 的 `Z` expectation 为 `(-1)**bits[q]`，`X/Y` 为 `0`；`bits[0]` 对应 qubit 0，不按 packed integer bit order反转。
- `ProductBlochState(bloch)`：`bloch` shape 为 `(nqubits, 3)`，列顺序为 `(x, y, z)`，每个 Pauli word expectation 是其非 identity local Bloch components 的乘积。

Bloch entries 必须是 finite `float64`。每个 vector norm 的validation boundary为 `1 + 1e-12`；norm小于1表示mixed single-qubit state，norm等于1表示pure state。输入值保持原样，不做renormalization。`nqubits=0` 使用空 product，identity expectation 为1。

Expectation reduction 在 Rust 中直接遍历最终 native terms并返回scalar。Canonical sorting、Python `PauliWord` construction和timing instrumentation只在对应的operator materialization或profile调用中发生。

## 5. 公开 Python API

新增 public API 放在 `python/tencirpauli/propagation.py`，由顶层 `tencirpauli` 导出稳定类型。Python 负责 friendly validation 和一次性 contiguous conversion；完整 tape compilation、operator storage、propagation、projection、expectation 与 optional profiling 在一次粗粒度 native call/handle 中完成。

### 5.1 Initial-state descriptors

建议 public shape：

~~~python
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class ZeroState:
    pass

@dataclass(frozen=True)
class ComputationalBasisState:
    bits: tuple[int, ...]

@dataclass(frozen=True)
class ProductBlochState:
    bloch: np.ndarray
~~~

允许字符串便利输入 `initial_state="zero"`，但其他 state 必须通过 typed descriptor，避免把长度为 `nqubits` 的 bits 与 `(nqubits,3)` Bloch arrays 混淆。Descriptor 构造后对 NumPy storage 使用 immutable snapshot；调用者后续修改原 array 不得改变已编译 engine。

### 5.2 GateTape builder

建议 public shape：

~~~python
tape = tcp.GateTape(nqubits=100)

tape.h(0)
tape.cnot(0, 1)
tape.sdg(2)

tape.rx(0, parameter=0)
tape.ry(1, angle=0.125)
tape.rzz(0, 1, parameter=1)

tape.ptm((2,), one_qubit_ptm, name="custom_1q")
tape.ptm((3, 4), two_qubit_ptm, name="custom_2q")

tape.nqubits
tape.nparameters
len(tape)
~~~

REQUIRED gate methods：

~~~python
class GateTape:
    def x(self, wire: int) -> None: ...
    def y(self, wire: int) -> None: ...
    def z(self, wire: int) -> None: ...
    def h(self, wire: int) -> None: ...
    def s(self, wire: int) -> None: ...
    def sdg(self, wire: int) -> None: ...
    def cnot(self, control: int, target: int) -> None: ...
    def cz(self, wire0: int, wire1: int) -> None: ...
    def swap(self, wire0: int, wire1: int) -> None: ...

    def rx(self, wire: int, *, angle: float | None = None, parameter: int | None = None) -> None: ...
    def ry(self, wire: int, *, angle: float | None = None, parameter: int | None = None) -> None: ...
    def rz(self, wire: int, *, angle: float | None = None, parameter: int | None = None) -> None: ...
    def rxx(self, wire0: int, wire1: int, *, angle: float | None = None, parameter: int | None = None) -> None: ...
    def ryy(self, wire0: int, wire1: int, *, angle: float | None = None, parameter: int | None = None) -> None: ...
    def rzz(self, wire0: int, wire1: int, *, angle: float | None = None, parameter: int | None = None) -> None: ...

    def ptm(self, wires: Sequence[int], matrix: np.ndarray, *, name: str | None = None) -> None: ...
~~~

每个 rotation 必须且只能给 `angle` 或 `parameter` 之一。Static angle 是 finite float64；parameter slot 是 non-negative integer，bool 不合法。多个 gates 可以引用同一 slot，Phase 4 梯度会确定性累加这些贡献。Engine compilation 时所有使用过的 slots 必须恰好覆盖 `0..nparameters-1`，有 hole 则失败；runtime parameter vector shape 必须严格为 `(nparameters,)`。

`CNOT(control,target)` 保持方向；所有 two-qubit gates 使用不同 wires。Tape 是 mutable Python builder，`PropagationEngine` construction 获取 immutable native snapshot；之后修改 builder 不影响已有 engine。

### 5.3 Reusable PropagationEngine

建议 REQUIRED API：

~~~python
engine = tcp.PropagationEngine(
    tape,
    observable,
    initial_state=tcp.ZeroState(),
    max_weight=3,
    max_bytes=tcp.DEFAULT_MAX_BYTES,
)

value = engine.expectation(params)          # float, Rust-only hot result
operator = engine.propagate_operator(params) # PauliOperator, explicit materialization
profiled = engine.profile(params)            # value + lightweight metadata

engine.nqubits
engine.nparameters
engine.max_weight
engine.is_exact
~~~

建议结果类型：

~~~python
@dataclass(frozen=True)
class PropagationProfile:
    gate_count: int
    initial_terms: int
    final_terms: int
    peak_terms: int
    estimated_peak_bytes: int
    final_weight_counts: tuple[int, ...]
    kernel_seconds: float

@dataclass(frozen=True)
class ProfiledExpectation:
    value: float
    profile: PropagationProfile
~~~

`expectation()` 是主性能路径，只返回 scalar。`profile()` 是显式诊断调用并启用 timing instrumentation；`final_weight_counts[w]` 描述最终 operator 中 weight `w` 的 canonical term 数。

`PropagationEngine` construction 一次性完成 tape validation/compilation、custom PTM transition compression、observable native snapshot、initial projection、state descriptor snapshot 和主要 scratch capacity planning。Steady `expectation(params)` 只把 contiguous float64 parameter buffer传入 Rust并取回 scalar。

`propagate_operator()` 返回现有 public `PauliOperator`；native 侧直接生成 packed/code arrays 与 complex128 buffer，再通过已有 private array constructor 一次性建立结果。该路径的 conversion/materialization 成本与 scalar expectation 分开 benchmark。

同一 engine object 的并发调用通过 per-call scratch 或明确的内部同步保证结果隔离与内存安全。长时间 execution 释放 GIL。

### 5.4 `max_bytes` migration

Phase 3 开始时把 `python/tencirpauli/hamiltonian.py` 的 `DEFAULT_MAX_BYTES` 改为 `16 * 1024**3`，并同步所有 public signatures、validation、tests、README 和 typing。历史 acceptance/benchmark 报告中的“当时为 4 GiB”事实不回写；当前文档与代码使用 16 GiB。

所有 public `max_bytes` 接受 positive integer 或 `None`。`None` 表示关闭 best-effort byte guard，但不关闭 checked dimension、index、capacity 和 arithmetic overflow。Native boundary 可以用 `Option<usize>` 或 private sentinel 表示 unbounded，不能把 Python `None` 错当成零预算。

Propagation 的 estimate 覆盖当前/下一 canonical term storage、custom transition table 和可廉价估算的主要 contribution/scratch buffers。该值是best-effort major-allocation estimate；hash-table overhead、allocator fragmentation、Rayon worker-local transient storage、PyO3 conversion和OS RSS由运行环境承担，系统OOM仍可能发生。

## 6. Rust core、PyO3 与数据模型

### 6.1 Module boundary

在现有core crate增加清晰模块，例如`gate.rs`与`propagation.rs`；native crate增加对应`propagation.rs` binding；Python增加`propagation.py`。Core保持pure Rust，PyO3、NumPy和TensorCircuit边界留在既有native/Python层。

建议 core types：

~~~rust
enum ParameterRef {
    Static(f64),
    Slot(u32),
}

enum GateOperation {
    Clifford1 { gate: Clifford1, wire: u32 },
    Clifford2 { gate: Clifford2, wire0: u32, wire1: u32 },
    PauliRotation { generator: LocalPauliGenerator, parameter: ParameterRef },
    CustomPtm { wires: LocalWires, transitions: LocalTransitions },
}

struct GateTape { /* immutable compiled operations */ }
struct PropagationConfig { max_weight: Option<u32>, max_bytes: Option<usize> }
enum ProductState { Zero, ComputationalBasis(/* packed bits */), Bloch(/* contiguous f64 */) }
struct PropagationStats { /* lightweight counters */ }
~~~

具体命名可调整，但 parameter structure、compiled operations、numeric parameter buffer 和 scratch 必须分离，以便同一 engine 重复执行不同参数，不重建结构。

### 6.2 Propagation-specific packed key

Phase 3 hot container为`nqubits <= 64`和`nqubits <= 128`提供inline packed key representation，使100-qubit key的equality、hash、local bit update、commutation和weight不触发per-term heap allocation。更宽系统使用语义一致的multiword fallback。

实现可以使用propagation-private key enum、fixed inline arrays或经过benchmark证明的其他zero-/single-allocation layout；allocation/profile evidence需要覆盖100-qubit representative workload。

### 6.3 Local kernels

- Clifford one-/two-qubit conjugation使用预计算code/sign lookup table或等价bit kernel。
- Pauli rotation先用 packed symplectic inner product判断 commute；反对易时只生成 unchanged cosine branch 与 exact `i P Q` sine branch。
- Custom PTM 在 compile time按 input local code压缩 exact-nonzero transitions；runtime 只遍历对应 row list。
- 有限 `max_weight` 的 weight 通过 packed popcount或经验证的 local delta更新；exact path 应完全跳过不需要的 weight/projection工作。
- Static angle 的 `sin/cos` 可在 engine construction 预计算；parameter slot 的 `sin/cos` 每次 execution 每个 gate计算一次并复用。

### 6.4 Dynamic aggregation 与 scratch reuse

Engine围绕reusable current/next term storage、预分配capacity和reusable worker scratch设计。Exact Clifford-only segments使用不分支fast path；branching rotations/PTM进入aggregation machinery。

优先评估以下实现：unique term vector作为稳定current storage，Clifford segment原地或写入reusable vector，branching gate写入worker-local contribution buffers/maps，再按canonical key deterministic merge。若profile支持double-buffer hash aggregation，则以deterministic iteration、capacity reuse和final-sort成本作为选择依据。

并行化以term数和branch factor为依据设置threshold，小workload走sequential kernel，大workload按chunk使用worker-local scratch。线程数和oversubscription处理遵循现有benchmark/AGENTS规则；所有parallel speedup同时给出固定单线程control。

### 6.5 Error mapping 与 panic boundary

Core 使用 typed errors 覆盖 invalid wire、duplicate two-qubit wire、invalid parameter slot、parameter shape、non-finite angle/parameter/PTM/Bloch value、non-real PTM dtype、invalid PTM shape、non-Hermitian expectation input、invalid `max_weight`、checked overflow、allocation estimate 和 incompatible nqubits。PyO3 稳定映射到 `TypeError`、`ValueError`、`OverflowError` 或 `MemoryError`，并把panic boundary保持在library内部。

## 7. 独立 reference 与 correctness matrix

### 7.1 Dense reference

新增独立 NumPy reference，不调用被测 Rust gate tables、PTM compiler 或 propagation code生成 expected result。对 `n<=5`，用显式 I/X/Y/Z matrices、`np.kron` 和 gate matrices构造 `U† O U`；从 dense matrix直接计算 product-state expectation。

Weight-projected reference 对 `n<=4` 在每个 reverse gate 后将 dense operator完整分解到 `4**n` Hermitian Pauli basis，聚合 coefficients，再把 weight 超过 `max_weight` 的 basis coefficients置零。该 reference 用于验证“每 gate projection”和“aggregation before projection”，不能只在最终结果上做一次 projection。

Custom PTM reference 直接按本文 `R[out,in]` convention作用于 local Pauli coefficients，并与由随机 one-/two-qubit unitary计算出的 real PTM交叉验证。Reference 必须包含 negative PTM entries 和 complex-input rejection。

### 7.2 Fixed regression vectors

至少固定以下 regression：

- 每个内置 one-/two-qubit gate 对全部 local Pauli basis elements 的 conjugation word/sign。
- `RZ(theta)† X RZ(theta)` 等 rotation sign/convention，覆盖 `theta=0, ±pi/2, pi` 和非特殊随机角。
- CNOT control/target 方向、SWAP wires、首尾 qubit ordering 和跨 packed-word boundary wires `63/64`、`64/65`。
- 初始 operator 含 weight 超限 term 时的 initial projection。
- 两个或更多 input contributions 在一个 gate 后生成相同 word，先聚合再 exact-zero removal/projection。
- `max_weight=None`、`max_weight=nqubits` 和大于 `nqubits` 返回同一 exact result。
- Clifford-only tape 不产生 branch term growth，但有限 cutoff仍可因 Clifford weight growth删除 term。
- zero-qubit identity、empty operator、identity-only observable 和 empty tape。
- repeated parameter slot、static/slot mixture、slot hole、错误 parameter length 和 non-finite parameter。
- computational bits 的 qubit-0 ordering、pure/mixed Bloch product states 和 invalid Bloch norm。
- real PTM negative entries、dense two-qubit PTM、wrong shape、complex dtype、NaN/Inf 和 duplicate wires。

### 7.3 Differential/property tests

- 随机 `n<=5` Clifford/rotation tapes 的 exact operator 与 dense reference一致。
- 随机 `n<=4` tapes、observables 和 `max_weight` 的每步 projected recurrence与 full Pauli decomposition reference一致。
- `expectation()` 与显式 `propagate_operator()` 后在同一 product state上求值一致，两条路径使用各自独立的public implementation和timed boundary。
- Hermitian input经 real PTM/内置 gates后保持 Hermitian；non-Hermitian operator可 materialize但不能调用 physical expectation。
- 同一 tape/observable/params重复运行、不同 hash seed和受支持 thread configurations返回 deterministic public operator/stats。
- 65、100、128 qubit structural/property cases覆盖 multiword bit operations，不使用 dense reference；与独立 Python packed-word rules比较 local updates、commutation、weight和 product-state expectation。
- Memory guard用小预算触发，不实际申请接近 16 GiB；`None`关闭 guard但 overflow仍失败。
- 长计算释放 GIL，并验证并发调用不会 data race、deadlock 或污染 engine state。

## 8. 性能设计与 benchmark 合同

### 8.1 衡量原则

所有性能数据使用 optimized release build。Debug timing、未同步 JAX enqueue、只测 Rust 内核却宣称 Python API speedup、或把 operator materialization排除在声称返回 operator 的计时之外，均不构成证据。

性能工作采用持续profile和优化策略。代表性warm-JIT workload中的kernel、allocation、parallelism和FFI结果决定下一轮结构性优化；每轮改变后重测correctness与同机release benchmark。超出Phase 3算法、依赖或语义范围的后续优化以具体证据进入handoff。

### 8.2 必须分开的时间边界

每个 Python benchmark至少分开记录：

1. friendly Python input/observable/tape construction；
2. native engine construction与custom PTM compilation；
3. first `expectation(params)`；
4. synchronized steady `expectation(params)`，包含 Python→Rust参数边界与scalar返回；
5. `propagate_operator(params)` 的完整 packed output与Python `PauliOperator` materialization；
6. explicit `profile(params)` instrumentation overhead；
7. major output/storage、estimated peak和有条件时的OS peak RSS；
8. result error、final/peak term counts和thread count。

JAX timed callable必须在内部 `block_until_ready()`。Warm-JIT comparison在双方都完成 reusable setup后进行：Rust复用 native tape/observable handle，JAX复用已编译 function和固定结构；不能让一侧重复setup而另一侧复用。

### 8.3 Baselines

- **Matched JAX reference**：benchmark目录内维护基于全局 k-local basis、但数学上等价于有限 `max_weight` recurrence 的 complex128 JAX实现，使用相同 gate convention和product-state expectation，用于主要 warm-JIT comparison。它是benchmark/reference代码，不进入public package；exact dynamic cases只在结构和内存都可公平覆盖的小规模上比较。
- **TensorCircuit ecosystem baseline**：对只支持 `|0...0>` 且语义可对齐的cases，使用现有 `tensorcircuit.pauliprop.PauliPropagationEngine`，记录其global k-local dense representation、complex64内部dtype和同步warm-JIT性能。由于dtype与representation不同，结果必须明确标注，不能作为唯一matched baseline。
- **Python/NumPy reference**：用于correctness与non-JIT端到端对照，但不是主要性能门槛。

### 8.4 REQUIRED workloads

Workload名称、seed、gate顺序、observable、`max_weight`、dtype和thread配置写入benchmark源码并保持稳定。至少覆盖：

1. **Clifford-heavy 100q**：100 qubits，多层 H/S/Sdg/CNOT/CZ/SWAP brick-wall，局部和多项 observable；验证无branch fast path、multiword key和scalar expectation吞吐。
2. **12q TFIM-style PPE**：hardware-efficient RX/RY/RZ+CNOT tape，TFIM local Hamiltonian，`max_weight` 取 2/3/4 与 exact-small control；对照 matched JAX warm-JIT。
3. **2D Heisenberg/Trotter**：至少 4x4 qubits 的 nearest-neighbor RXX/RYY/RZZ layers，`max_weight` 取 3/4，覆盖两比特rotation、term growth和aggregation。
4. **100q near-Clifford rotation workload**：100 qubits，Clifford brick-wall中插入少量 Pauli rotations，local observable sum，覆盖目标大系统动态结构。
5. **Duplicate-heavy synthetic**：构造大量不同parent contributions汇入相同canonical words，测aggregation、exact cancellation和deterministic merge。
6. **Custom PTM**：sparse one-qubit、sparse two-qubit和dense two-qubit real PTM，单独报告inherent branch factor与transition compilation成本。
7. **Operator materialization**：选择能生成大final operator的workload，分开测Rust propagation、packed return和Python public object construction。
8. **Scaling**：固定物理结构下扫描qubits、layers、input terms、`max_weight`、final/peak terms和1/2/4/固定最大线程数，报告throughput与memory。

所有大workload在运行前按`AGENTS.local.md`估计完整peak；当前机器不得启动估计超过16 GiB的case，即使public API允许用户关闭guard。

### 8.5 Rust microbench 与 profiling

新增Criterion target，例如 `crates/tencir-pauli-core/benches/propagation.rs`，至少覆盖：inline key hash/equality/weight、Clifford local update、rotation commute/branch、custom PTM transition apply、duplicate aggregation、finite projection、product-state expectation，以及完整gate-tape kernel。

Python benchmark新增公开API setup/first/steady/materialization cases，并纳入 `benchmarks/run.py` record/compare。每个material optimization保留前后同机release记录；profile可以在仓库外保存，但`implementation-status.md`必须记录工具、workload、主瓶颈、采取的改变和结果。`.benchmarks/`与profile原始输出不得提交。

## 9. 实现切片

### P0：语义/reference skeleton 与 16 GiB migration

- 建立独立dense和per-gate Pauli-decomposition reference、固定gate/PTM/ordering vectors和失败测试。
- 增加private Python类型/API skeleton和typing；完成对应native纵向切片后再加入顶层public exports。
- 将所有public `DEFAULT_MAX_BYTES`与signatures迁移到16 GiB、支持显式`None`，同步README/tests；历史报告保持原文。
- 为Phase 3 benchmark加入stable workload definitions和matched JAX reference skeleton，先记录当前Python/TensorCircuit baseline。

Acceptance gate：所有新reference独立于被测实现；memory migration通过existing Phase 1–2 regression；baseline在timed callable内完成同步，dtype/语义差异被明确记录。

### P1：GateTape 与 specialized local rules

- 在pure Rust core实现compiled `GateTape`、wire/parameter validation、fixed Clifford lookup和RX/RY/RZ/RXX/RYY/RZZ analytic branch rules。
- 覆盖static angle、contiguous/repeated parameter slots和reverse Heisenberg order。
- Rust unit/property tests对全部local Pauli basis与dense matrices一致。
- 增加Criterion local-kernel benchmark并完成首轮profile。

Acceptance gate：每个built-in gate的word/sign/coefficient与independent dense reference一致；100-qubit tape/key operation全部在Rust批量路径中完成。

### P2：统一 dynamic recurrence 与 exact propagation

- 实现propagation-private inline packed key、dynamic term storage、deterministic aggregation、exact-zero removal和reusable scratch。
- `max_weight=None`路径完成exact dynamic propagation；Clifford-only segments使用不分支fast path。
- 覆盖empty/identity、n=0、64/65/100/128边界、term growth、exact cancellation和memory estimates。
- 增加完整Rust gate-tape benchmark，profile allocation/hash/merge热点并优化。

Acceptance gate：随机小系统operator逐元素与dense `U†OU`一致；公开/serialized output canonical deterministic；100-qubit representative path的allocation profile满足inline-key设计目标。

### P3：Pauli-weight projection

- 实现initial projection和每gate `Aggregate`之后的finite `max_weight` projection。
- Exact path跳过weight/projection开销；finite path覆盖Clifford weight growth和rotation/PTM branching。
- 用full Pauli decomposition reference验证每一步recurrence，而非只比较最终近似值。
- 增加`max_weight`/term-growth/duplicate-heavy scaling benchmark。

Acceptance gate：`None`、`nqubits`和更大cutoff与exact相同；有限cutoff严格匹配逐gate aggregation与weight projection reference。

### P4：Product-state expectation、PyO3 与 reusable Python engine

- 实现Zero/computational/Bloch product-state descriptors和Rust scalar expectation。
- 建立一次性batched native engine construction；steady call只接收params并返回scalar。
- 实现显式packed `propagate_operator()` materialization和`profile()` metadata。
- Release GIL，保证concurrent call安全，补齐public docstrings、typing与error mapping。
- 增加Python setup/first/steady/materialization benchmark并对照matched JAX warm-JIT。

Acceptance gate：expectation与explicit operator求值一致；默认hot path只传递parameter buffer并返回scalar；public API通过typed facade和一次性batched native handle执行。

### P5：Custom real PTM

- 实现one-/two-qubit real float64 PTM validation、orientation、wire-order contract和exact-nonzero transition compilation。
- Built-in gates继续走specialized kernel；custom PTM覆盖negative、sparse和dense real entries。
- 增加random unitary-derived PTM differential、real-dtype/shape/value validation tests和sparse/dense PTM benchmark。

Acceptance gate：custom map与independent local coefficient reference一致；real float64输入合同完整验证；全部exact-nonzero entries参与传播；dense PTM的branch/memory成本被如实记录。

### P6：性能工程与完整 benchmark conclusion

- 对REQUIRED workloads运行release Rust/Python benchmark、固定thread scaling和同步warm-JIT对照。
- 至少完成针对实际dominant bottleneck的一轮结构性优化；每次优化后重跑correctness gate和同机compare。
- 检查inline key、scratch reuse、sincos caching、Clifford fast path、aggregation layout、parallel threshold、GIL和operator boundary。
- 对Rust与warm-JIT baseline的全部差异记录workload、dtype、原因、profile和下一步。

Acceptance gate：每个material hot path有稳定benchmark和profile-backed结论；所有数据来自release build、同步JAX、批量FFI和等价语义timed boundary。

### P7：Public交付与Phase 4 handoff

- 顶层导出GateTape、state descriptors、PropagationEngine和profile types；更新README、CHANGELOG、typing stub、docs/vibe index与implementation status。
- 完成Linux/macOS/Windows correctness/package smoke所需修改；性能记录继续使用本机release benchmark体系。
- 记录最终benchmark label、workload结果、known limitations和Phase 4 prerequisites。
- 确认parameter slots、Pauli-rotation local rules和engine handles形成稳定的后续gradient基础。

Acceptance gate：新环境release安装后可运行public example；完整quality workflow通过；文档准确描述Phase 3已经交付的public surface和Phase 4 handoff。

## 10. 一次性验收标准

Phase 3最终验收至少证明：

- 单一recurrence语义、gate conventions、reverse tape order、PTM orientation和product-state ordering均有独立reference。
- Exact与finite-weight recurrence在小系统分别匹配dense/full-Pauli references；projection只发生在aggregation之后。
- 所有首批built-in gates和real custom PTM通过local basis exhaustive tests。
- Rust scalar expectation、explicit materialized operator和dense state expectation一致。
- Python hot call复用native tape/observable handle，只传params并返回scalar；完整operator conversion单独显式发生。
- 64/65/100/128-qubit packed paths、n=0、invalid inputs、overflow、16 GiB default和`max_bytes=None`均有tests。
- Public output deterministic，long computation释放GIL，并发调用无data race。
- Rustfmt、Clippy `-D warnings`、workspace tests、Black、Ruff、strict mypy、`maturin develop --release --locked`和pytest全部通过。
- Criterion与pytest-benchmark包含本文REQUIRED workloads；JAX计时同步，warm-JIT是主要性能对照，setup/cold/materialization/memory/accuracy分开报告。
- 至少一个最终本机benchmark label与profile-backed optimization记录写入`implementation-status.md`；`.benchmarks/`不提交。
- README、public docstrings、typing、CHANGELOG、architecture/status与实际实现一致。

Phase 3性能验收材料包括matched warm-JIT comparison、profile、代表性100-qubit workload和针对主瓶颈的优化证据；这些材料与correctness evidence共同构成完成记录。

## 11. 给下一位实现模型的执行顺序

从P0开始按纵向切片推进，每片都遵循“独立reference/失败测试 → pure Rust core → batched native boundary → typed Python facade → correctness gate → release benchmark/profile → status evidence”。同一时间只把一个最早未完成切片标为active，并在纵向能力贯通后导出对应public API。

性能优化贯穿P1–P6，不是最后才运行benchmark。每次发现热点先确认timed boundary与语义公平，再优先处理asymptotic/data-layout/allocation问题，最后才做micro-optimization。任何unsafe、specialized hashing、new dependency、architecture-specific SIMD或并行复杂化都必须有代表性end-to-end收益和专门tests；没有profile证据时保持safe Rust和最小设计。

每完成一个有证据的切片，更新`implementation-status.md`的active milestone、精确命令/结果、benchmark label、profile结论、known limitations和下一步。Phase 3最终handoff包含稳定的GateTape、parameter slots、local analytic rules和native handles，供下一阶段直接扩展gradient能力。
