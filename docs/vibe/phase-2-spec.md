# Phase 2 Spike：Symmetry analysis 与 sector reduction

状态：已实现并完成首轮本地检查；2026-08-02 acceptance review 发现多生成元 tapering row-sign blocker，修复并通过独立 projector/property regression 前不得标记为最终验收完成。Phase 1 remediation 和 Rust module split 已完成，不属于本 Spike 的待办。MSRV 1.85 CI 不在本阶段实施。

## 1. 这一阶段到底要解决什么

Phase 2 不负责“生成一个具有 Z2 或 U(1) 对称性的 Hamiltonian”。它接收用户已有的 `PauliOperator`，回答两个更实用的问题：这个 Hamiltonian 是否存在可利用的对称性，以及选定某个对称 sector 后，能否把后续矩阵、MVP、求本征值或 VQE 计算放到更小的空间中。

本阶段要求交付两个端到端能力：第一，自动发现 Pauli 型 Z2 symmetries，并把选定 sector 的 Hamiltonian taper 成更少 qubit 的 `PauliOperator`；第二，在用户明确给出粒子数 `k` 时验证 U(1) 粒子数守恒，并构造维度为 `C(n,k)` 的 restricted Hamiltonian plan。

这两个能力的价值不只是“报告 symmetry 存在”，而是实际减少后续计算的状态空间：每个独立 Z2 symmetry 通常可以 taper 掉一个 qubit；U(1) sector 则把完整的 `2**n` 维 Hilbert space 限制到 fixed-Hamming-weight 的 `C(n,k)` 维子空间。

## 2. 背景概念

### 2.1 什么是 Z2 symmetry analysis

如果一个 Pauli operator `S` 满足 `S²=I` 且与 Hamiltonian 对易，即 `[H,S]=0`，那么 `S` 的本征值只能是 `+1` 或 `-1`。系统因此分成两个互不混合的 sector，这就是一个 Z2 symmetry。若有 `r` 个线性独立、彼此对易的 generators，则 sector 由 `r` 个正负号标记。

例如 transverse-field Ising Hamiltonian `H=-Σ Zi Zi+1-hΣ Xi` 与全局 spin flip `S=X0 X1 ... Xn-1` 对易。求基态时可以只研究 `S=+1` 或 `S=-1` 的 sector，而不必同时保留两者。

这里的 analysis 是：把 Hamiltonian 的 canonical Pauli terms 转换成 binary-symplectic constraints，在 GF(2) 上求解所有与这些 terms 对易的 Pauli candidates，再选出一组确定性的、线性独立且彼此对易的 generators。返回前必须用完整 Hamiltonian 再验证每个 generator 的 exact commutation，不能把未经验证的 null-space candidate 暴露给用户。

第一版只发现逐 term 对易的 Pauli symmetries。这是清楚、稳定且可精确验证的合同；依赖不同 Hamiltonian terms 之间 coefficient cancellation 才成立的更一般 symmetry 不属于本 Spike。

### 2.2 什么是 tapering

Symmetry analysis 本身只告诉用户“存在 `S`”。Tapering 才把这个结果转化成更小的问题：构造一个 Clifford change of basis，将每个 generator 映射成某个单 qubit `Z`；用户选择该 generator 的 sector eigenvalue `s∈{+1,-1}` 后，用 `Z=s` 代入并移除这个 qubit。

例如一个 8-qubit Hamiltonian 有两个可独立利用的 Z2 generators，选择 sector `(+1,-1)` 后，tapered Hamiltonian 通常只剩 6 qubits。变换必须同时适用于 Hamiltonian 和用户之后要测量的 observables，因此公开结果应是可复用的 `Z2TaperingPlan`，而不只是一次性返回缩小后的 Hamiltonian。

本 Spike 选择实现最小但完整的 tapering，而不是只返回 generators 或 projector。没有可复用 Clifford/provenance plan 的 analysis-only 结果不算 Phase 2 Z2 能力完成。

### 2.3 什么是 U(1) particle-number sector

U(1) symmetry 在这里具体指总粒子数 `N` 守恒。用户明确指定 `particle_number=k` 后，只保留恰好有 `k` 个占据位的 computational basis states。其维度从 `2**n` 降为 `C(n,k)`；例如 `n=20,k=2` 时从 1,048,576 降为 190。

第一版不自动猜测任意连续 symmetry，也不从 Hamiltonian 自动决定 `k`。它只接受显式 `k`，验证 Hamiltonian 不会把 weight-`k` basis state 映射到其他 particle-number sector，然后生成 restricted MVP/CSR。验证针对聚合后的完整 operator，而不是逐 Pauli term 下结论：例如 number-conserving hopping 中的 `XX+YY` 依靠两项抵消 sector-changing amplitudes，不能因为单独的 `XX` 或 `YY` 看似会改变粒子数而误报。若完整 Hamiltonian 仍含有泄漏，例如额外的单独 `X0`，必须明确失败，不能静默投影或删除该贡献。

## 3. 已完成基础，不要重复实现

- Phase 1 的 canonical `PauliWord`、`PauliOperator`、deterministic aggregation、dense/COO/CSR/MVP、memory limits 和 TensorCircuit qubit ordering 直接复用。
- Core 已拆成 `error.rs`、`scalar.rs`、`word.rs`、`operator.rs`、`grouping.rs`、`hamiltonian.rs`；native binding 已拆成 `convert.rs`、`word.rs`、`operator.rs`、`grouping.rs`、`hamiltonian.rs`。下一位 Agent 不再做 module split。
- Phase 1 的 correctness、format、lint、packaging 和 benchmark 规则继续由 `AGENTS.md`、`semantics.md`、`benchmarking.md` 和 `scripts/check.py` 管理。本 Spec 不重复列出这些通用命令。

## 4. 建议的公开 Python API

新 API 放在 `python/tencirpauli/symmetry.py`，并由顶层 `tencirpauli` 导出稳定类型。`_native` 仍是 private implementation detail；Rust core 不依赖 Python、NumPy 或 TensorCircuit。

### 4.1 Z2 analysis

~~~python
analysis = h.find_z2_symmetries(max_bytes=1 << 30)

analysis.nqubits
analysis.generators       # tuple[PauliWord, ...]
analysis.rank             # len(generators)
analysis.constraint_rank  # GF(2) constraint rank，诊断信息
~~~

建议结果类型：

~~~python
@dataclass(frozen=True)
class Z2SymmetryAnalysis:
    nqubits: int
    generators: tuple[PauliWord, ...]
    rank: int
    constraint_rank: int

    def tapering_plan(self, sector: Sequence[int]) -> "Z2TaperingPlan": ...
~~~

`generators` 的顺序必须 canonical 且可复现。不同 GF(2) elimination pivot、hash seed 或线程数不能返回不同但等价的 generator bases。恒等算符不作为 generator；找不到非平凡 symmetry 时返回空 analysis，而不是报错。

### 4.2 Z2 tapering plan

~~~python
analysis = h.find_z2_symmetries()
plan = analysis.tapering_plan(sector=(+1, -1))

h_small = plan.transform_operator(h)
observable_small = plan.transform_operator(observable)

plan.nqubits_before
plan.nqubits_after
plan.generators
plan.sector
plan.removed_qubits
plan.clifford_operations
~~~

建议类型：

~~~python
@dataclass(frozen=True)
class Z2TaperingPlan:
    nqubits_before: int
    nqubits_after: int
    generators: tuple[PauliWord, ...]
    sector: tuple[int, ...]
    removed_qubits: tuple[int, ...]
    clifford_operations: np.ndarray

    def transform_operator(self, operator: PauliOperator) -> PauliOperator: ...
~~~

`sector` 长度必须等于 generator 数量，每个值只能是 `+1` 或 `-1`。`transform_operator()` 必须验证输入 operator 与所选 generators 兼容；不对易的 observable 明确失败。变换后的 coefficient、Pauli phase 和 qubit ordering 必须与对原矩阵执行同一个 Clifford transform 并投影到 sector 的 dense reference 一致。

`clifford_operations` 是当前进程内可检查的紧凑 gate/bit-operation provenance，不保存 Python callable 或 backend object。public plan 必须保留 generator、sector、removed-qubit 和 forward transform provenance，使同一 runtime plan 能稳定应用到多个 observables。稳定的 pickle/JSON/跨进程 plan serialization 不是 Phase 2 用户需求或验收项；需要持久化时应在真实用例出现后另行定义 schema/version。

可以在上述基础上增加便利方法 `h.taper_z2(sector=...)`，但它只是 `find_z2_symmetries()`、`tapering_plan()` 和 `transform_operator()` 的组合，不替代显式的可复用接口。

### 4.3 U1 sector

~~~python
sector = tcp.U1Sector(nqubits=20, particle_number=2)

sector.dimension          # 190
sector.rank(bitstring)    # full-space basis state -> restricted index
sector.unrank(index)      # restricted index -> full-space basis state
sector.basis_words()      # packed read-only uint64 arrays
~~~

建议类型：

~~~python
@dataclass(frozen=True)
class U1Sector:
    nqubits: int
    particle_number: int

    @property
    def dimension(self) -> int: ...
    def rank(self, bitstring: int | Sequence[int]) -> int: ...
    def unrank(self, index: int) -> int | tuple[int, ...]: ...
    def basis_words(self, *, max_bytes: int = DEFAULT_MAX_BYTES) -> np.ndarray: ...
~~~

Basis 顺序固定为 TensorCircuit computational-basis integer 的升序，其中 qubit 0 是最高有效位，只筛选 Hamming weight 等于 `k` 的 states。`rank()`/`unrank()` 应使用 combinatorial indexing，不应为了查询一个 index 就物化整个 basis。`basis_words()` 才显式分配完整 packed basis，并受 `max_bytes` 保护。

### 4.4 U1 restricted Hamiltonian

~~~python
sector = tcp.U1Sector(nqubits=h.nqubits, particle_number=k)
restricted = h.restrict_u1(sector, max_bytes=1 << 30)

restricted.dimension
restricted.sector
restricted.apply(state_in_sector)
restricted.mvp_plan()
restricted.csr()
~~~

建议类型：

~~~python
class U1RestrictedOperator:
    sector: U1Sector
    dimension: int

    def apply(self, state: np.ndarray, *, max_bytes: int = DEFAULT_MAX_BYTES) -> np.ndarray: ...
    def mvp_plan(self, *, max_bytes: int = DEFAULT_MAX_BYTES) -> U1MvpPlan: ...
    def dense(self, *, max_bytes: int = DEFAULT_MAX_BYTES) -> np.ndarray: ...
    def coo(self, *, max_bytes: int = DEFAULT_MAX_BYTES) -> COOMatrix: ...
    def csr(self, *, max_bytes: int = DEFAULT_MAX_BYTES) -> CSRMatrix: ...

class U1MvpPlan:
    sector: U1Sector
    dimension: int

    def apply(self, state: np.ndarray) -> np.ndarray: ...
~~~

`U1MvpPlan` 是新类型，因为现有 `NativeMvpPlan` 的状态空间固定为完整 `2**n` basis，不能让两个不同 dimension 合同共用一个名字。`restrict_u1()` 首先对完整聚合 operator 验证 sector preservation，再构造 restricted plan。MVP 是大问题的默认路径；CSR 用于可安全物化的规模。`apply()` 的输入长度必须是 `C(n,k)`，结果必须等于 full-space action 后投影回 fixed-particle basis，但实现不能偷偷分配 full `2**n` state 或 full Hamiltonian。

当前 Phase 2 restricted Hamiltonian/MVP/CSR 使用单机 `usize` computational-basis index，只承诺 `nqubits < usize::BITS`；`U1Sector` 的 Python combinatorial rank/basis helper 可以覆盖更宽的 multiword bitstrings，但不能据此推断 restricted native operator 已支持 64+ qubits。64+ qubit low-particle-number U1 Hamiltonian restriction 已明确排入 roadmap 阶段五；TensorCircuit-style U1 circuit/time evolution 排入阶段六。

## 5. 典型用户流程

### 5.1 Z2 示例

~~~python
import tencirpauli as tcp

h = make_transverse_field_ising_hamiltonian(nqubits=8)
analysis = h.find_z2_symmetries()
print([word.to_string() for word in analysis.generators])

plan = analysis.tapering_plan(sector=(+1,))
h_even = plan.transform_operator(h)
energy = smallest_eigenvalue(h_even.csr())
~~~

用户得到的是“原 Hamiltonian 的 `+1` symmetry sector 中等价的更小 Hamiltonian”，不是新生成的物理模型。

### 5.2 U1 示例

~~~python
h = make_number_conserving_hopping_hamiltonian(nqubits=20)
sector = tcp.U1Sector(nqubits=20, particle_number=2)
restricted = h.restrict_u1(sector)

state = np.zeros(sector.dimension, dtype=np.complex128)
state[0] = 1.0
next_state = restricted.apply(state)
~~~

如果把 `X0` 加入 `h`，`restrict_u1()` 应报告该 term 会把粒子数 `k` 映射到 `k±1`，而不是返回一个看似可用但物理语义错误的结果。

## 6. 实现切片

### P0：独立 reference 与 API skeleton

先用小系统 NumPy/GF(2) reference 固定 Z2 commutation、generator span、Clifford transform、sector projection、U1 basis order、rank/unrank 和 `P†HP`。建立上述 Python dataclasses/type hints，但未完成 native kernel 的方法应保持 private，不发布空壳 public API。

### P1：Z2 symmetry analysis

在 core 增加 packed GF(2) elimination 和 Z2 constraint construction；native 用一次 batched call 返回 packed generators；Python 暴露 `find_z2_symmetries()`。这一切片完成时，用户可以可靠地发现和查看 generators，但还不能声称已经完成降维。

### P2：Z2 tapering

实现确定性的 Clifford mapping、sector substitution 和 reusable `Z2TaperingPlan.transform_operator()`。用 small dense matrices 验证 transformed Hamiltonian/observable 与原 sector action 和 spectrum 一致。这一切片完成后，Z2 才形成端到端功能。

### P3：U1 sector basis

实现 `U1Sector.dimension/rank/unrank/basis_words`，固定 TensorCircuit-compatible ordering，并覆盖 `k=0`、`k=n`、中心 sector、`n=0`、invalid `k` 和组合数溢出。

### P4：U1 restricted operator

实现 sector-leakage validation、restricted reusable MVP 和 bounded CSR。先保证与 `P†HP` 的 action 一致，再优化 basis lookup、neighbor generation、allocation 和 parallelism。

### P5：集成与性能结论

补齐 README examples、typing、CHANGELOG、TensorCircuit read-only differential smoke 和 benchmark workloads。性能结论必须同时报告 symmetry analysis/restriction setup 成本，以及降维后 MVP、CSR 或 eigensolver 的端到端收益；不能只展示 GF(2) kernel 很快。

## 7. 实现约束

- Core 保持 pure Rust；Z2/U1 新实现分别进入清晰模块，例如 `gf2.rs`、`symmetry.rs` 和 `sector.rs`，不再改回单文件。
- 每次 analysis、plan construction、operator transform 或 restricted apply 使用一次粗粒度 FFI；不逐 term、generator 或 basis state 调用 Python。
- 所有输出确定性；组合数、matrix dimension 和 arithmetic overflow 必须 checked。`max_bytes` 只对可廉价估算的主要 output/workspace 提供 best-effort guard，不是精确 peak-RSS 合同，不要求为 allocator overhead、FFI conversion 或所有 transient scratch 建立复杂模型。
- U1 不守恒、非法 sector、不兼容 observable、溢出和超限必须明确失败；sector-leakage 判断必须在相同 transition contributions 聚合后进行，任何逐 term 误判、自动 projection、silent fallback 或隐藏 tolerance 都是错误。
- Phase 1 的 complex128、exact-zero aggregation、qubit ordering、GIL release 和 public/private package 边界保持不变。

## 8. 一次性验收标准

通用 format/lint/test/package 命令直接遵循 `AGENTS.md` 和 `scripts/check.py`，不在每个切片重复抄写。Phase 2 额外需要证明：所有 Z2 generators exact commute、independent、pairwise commute 且顺序稳定；tapered operator 与原 sector 的 dense action/spectrum 一致；U1 rank/unrank 与 basis order 一致；restricted MVP/CSR 与 `P†HP` 一致；invalid dimension/arithmetic overflow 和可廉价识别的超大主要输出明确失败；公开 Python 调用没有逐元素 FFI。

至少保留四组端到端 workload：有全局 parity 的 TFIM、具有多个 Z2 generators 的小模型、number-conserving XX/YY hopping 或 XXZ 模型，以及明确破坏 U1 的反例。Benchmark 同时记录 setup、steady apply、主要 output/storage、结果误差以及有诊断价值时的 peak memory；本地结果仍保存在被忽略的 `.benchmarks/` 中，不设置共享 CI wall-time gate。

## 9. 非目标

- 不自动生成 TFIM、Hubbard 或其他具有 symmetry 的模型；这些只作为 examples/fixtures。
- 不自动发现一般连续 symmetry 或替用户选择 particle number。
- 不实现任意 non-Pauli symmetry、non-Abelian symmetry 或多个 U1 charges。
- 不进入 GateTape、Pauli propagation、native gradient、JAX custom call 或 general-commuting measurement diagonalization。
- 不实现 stable Z2 plan serialization/persistence；当前 reusable plan 只保证进程内使用。
- 不实现 64+ qubit native U1 restricted Hamiltonian/MVP/CSR；该能力属于 roadmap 阶段五。
- 不实现 TensorCircuit-style `U1Circuit` circuit execution 或含时演化；该能力属于 roadmap 阶段六。
- 不新增 crate，不修改 TensorCircuit 主仓库。

## 10. 给下一位 Agent 的执行边界

从 P0 开始逐片实现，每片都完成 core → batched native → typed Python → differential test，再进入下一片。`implementation-status.md` 记录唯一当前切片和证据；本文件描述目标与接口，不充当完成状态清单。不要重复 module split，不要在 P1 只有 analysis 时提前宣称支持 tapering，也不要在 U1 validation 尚未完成时通过丢弃 offending terms 伪造 restricted operator。
