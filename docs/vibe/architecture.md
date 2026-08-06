# TenCirPauli 设计方案

状态：概念设计，建议进入原型验证。项目名称为 TenCirPauli，Python distribution 与 import package 均为 `tencirpauli`。

## 1. 决策摘要

TenCirPauli 是 TensorCircuit 的 Rust-native companion，围绕 Pauli 代数、Pauli 算符、Hamiltonian 生成、对称性分析和 Pauli propagation 提供必需的 Python runtime package。它不替代 TensorCircuit 的 tensor backend，也不把通用量子线路数值计算搬到 Rust。Rust 负责离散、bit-packed、CPU 密集且适合批量执行的结构化工作；JAX、TensorFlow、PyTorch 和 NumPy 继续负责需要 backend tensor、自动微分或加速器执行的数值工作。

项目采用两种互补执行模式：Rust-native 模式在 CPU 上完成动态 Pauli operator propagation，并在 Phase 4 同时提供两类原生梯度，即只对当次非零 sparse forward trace 求导的 deterministic frozen-support reverse，以及从完整 Pauli path 空间采样得到的 SPPS 随机 value-and-gradient；backend-plan 模式由 Rust 生成稳定的代数、Hamiltonian 和 measurement plan，再由 `tc.backend` 执行需要多 backend、JIT 或加速器的数值计算。两类梯度针对不同执行合同，均为 REQUIRED 能力，不能互相替代。

建议首先实现 Pauli 核心表示、算符规范化、measurement grouping、Hamiltonian matrix/MVP plan 和基准体系，然后实现 forward-only Rust-native Pauli propagation。对称性分析和两类原生梯度在核心语义稳定后增加。性能工作不设置固定倍数的停止或淘汰门槛；每个阶段都必须以同步后的 Python/JAX warm-JIT steady runtime 为主要对照持续 profile 和优化，同时分别保留 setup、cold execution、memory 与 correctness evidence。

## 2. 背景与现状

TensorCircuit 当前已经包含三组相关能力：`tensorcircuit/quantum.py` 中的 `PauliStringSum2MVP`、`PauliStringSum2COO` 和 dense/sparse Hamiltonian 构造；`tensorcircuit/pauliprop.py` 中的 dense k-local 与 sparse bit-packed Pauli propagation；`tensorcircuit/u1circuit.py` 中的固定粒子数子空间模拟和 Pauli observable 支持。

这些实现已经验证了 Pauli 表示对 VQE、time evolution、ODE、局部 observable 和百比特近似传播的价值，但也暴露出适合 Rust 的瓶颈：组合 basis 构造和 neighbor map 依赖多层 Python 循环；传播过程中 Pauli term 集合会动态增长、合并并跨 weight sector 移动，而 JAX 更适合静态 shape 的 dense/fixed-buffer 计算；bit-packed 数据在 backend 中受 dtype 和 JIT 语义约束；大型 Hamiltonian 的规范化、重复项合并、commutation 分组和稀疏矩阵生成缺少统一的高性能抽象。

TenCirPauli 不移植当前 fixed-buffer `SparsePauliPropagationEngine`，也不在 Rust 中实现 top-k sparsity truncation 或 coefficient-magnitude cutoff。Native propagation 使用一个统一的动态 Pauli operator recurrence：先精确聚合相同 Pauli word，再在 `max_weight` 有限时按 Pauli weight/locality `w(P) <= max_weight` 做结构性投影；`max_weight=None` 或 `max_weight >= nqubits` 自动给出 exact recurrence。传播按 Heisenberg picture 逆序应用 gates，现有 one- and two-qubit gate 支持范围之外的 gate 必须明确失败。

## 3. 项目定位与技术壁垒

TenCirPauli 的核心技术壁垒不是单个 bit operation，而是一套与 TensorCircuit 语义一致的端到端 Pauli operator pipeline：同一 canonical representation 同时服务算符代数、measurement grouping、Hamiltonian 生成、Z2/U(1) 对称性处理、weight-truncated propagation、TensorCircuit backend plan 和可验证的近似截断。

相比独立的 Pauli 工具包，TenCirPauli 的差异化价值是：直接接受和生成 TensorCircuit 的 observable、Hamiltonian 与 gate tape；同时覆盖 exact/static preprocessing 与 approximate propagation；明确处理 qubit ordering、gate convention、complex dtype、gradient 和截断语义；能在 native CPU 与 backend-compatible 模式之间共享同一结构计划。

## 4. 设计目标

- 为任意 qubit 数提供紧凑、确定性、可 hash 的 Pauli word 和 Pauli operator 表示。
- 批量完成 Pauli multiplication、commutation、support、weight、canonicalization、deduplication 和 grouping。
- 快速生成 dense、COO/CSR、matrix-free MVP 和 symmetry-restricted Hamiltonian plan。
- 支持 Z2 Pauli symmetry 的发现、验证、sector 选择和 tapering；为显式给定粒子数守恒的 U(1) sector 提供 restricted basis 与 operator plan。
- 提供一个统一的动态 Pauli propagation recurrence；Clifford gate 自然走不分支的 exact fast path，`max_weight` 决定是否在聚合后应用 Pauli-weight projection。
- 提供无需 JAX tracing 的 Rust-native forward 路径，并在后续阶段同时提供 deterministic frozen-support reverse gradient 与 SPPS stochastic value-and-gradient。
- 提供 backend-plan 路径，使结构计算离开 JIT hot path，同时保留 `tc.backend` 数值执行和自动微分。
- TensorCircuit 是 Python distribution 的必需运行依赖；Rust wheel 或 source build 是 TenCirPauli 的执行依赖。Rust core 仍保持纯 Rust，不依赖 TensorCircuit。

## 5. 非目标

- 不实现通用 statevector、density-matrix、MPS 或 tensor-network simulator。
- 不替代 JAX、TensorFlow、PyTorch、NumPy、XLA 或 TensorCircuit backend abstraction。
- 不捕获或编译任意 Python 函数，也不建立通用 Function IR。
- 第一阶段不支持任意多比特 unitary、任意 channel、动态 measurement 或量子经典反馈。
- 第一阶段不承诺在 GPU 上执行 Rust-native propagation。
- 不假设 full Hamiltonian matrix 可以突破 `2**n` 的存储下界；大系统默认使用 MVP、sector-restricted 或 Pauli-native 表示。
- 不在第一阶段自动发现任意连续 Lie symmetry。U(1) 首先采用显式 charge 与 sector，自动检测仅作为后续能力。
- 不在 Rust 中复刻当前 fixed-buffer、top-k `SparsePauliPropagationEngine`。

## 6. 总体架构

~~~text
TensorCircuit / Python API
          │
          ├── Pauli strings / structures / coefficients
          ├── TensorCircuit QIR or explicit GateTape
          └── symmetry, measurement, and propagation configuration
          │
          ▼
PyO3 facade
          │  one batched call per construction, analysis, or propagation
          ▼
Rust Pauli core
          ├── canonical Pauli algebra
          ├── PauliOperator and coefficient mapping
          ├── Hamiltonian compiler
          ├── symmetry analyzer
          └── propagation engine
          │
          ├── native value / value_and_grad
          ├── NumPy COO/CSR or dense arrays
          ├── matrix-free native MVP
          └── backend execution plan
                    │
                    ▼
              tc.backend runtime
              JIT / AD / accelerator
~~~

Rust 与 Python 的边界必须是粗粒度的。禁止每个 gate、Pauli term 或 matrix element 单独跨 PyO3。构造一个算符、编译一个 plan、传播完整 gate tape 或计算一次 value-and-gradient 各自只进行一次主要 FFI 调用。

## 7. 核心数据模型

### 7.1 PauliWord

Rust 内部使用 binary symplectic representation：每个 phase-free `PauliWord` 由 `x_words: Vec<u64>`、`z_words: Vec<u64>` 和 `nqubits` 组成，只表示 canonical Hermitian `I/X/Y/Z` tensor product。乘法额外返回精确的四值 `PauliPhase={+1,+i,-1,-i}`；当 word 属于 `PauliTerm` 或 `PauliOperator` 时，该 phase 吸收到 complex coefficient。内部 qubit `q` 固定映射到 word 中的 bit `q % 64`，矩阵 basis ordering 在输入输出边界显式转换，不让内部 bit ordering 隐式决定 TensorCircuit 的 ket ordering。

`PauliWord` 必须支持 equality、stable ordering、hash、weight、support、adjoint、multiplication、commutation 和 symplectic inner product；multiplication result 同时包含 canonical word 和 `PauliPhase`。所有公开序列化必须包含 schema version、qubit count 和 ordering 标记。

### 7.2 PauliKey 与 coefficient

`PauliKey` 只包含 canonical `(x, z)`，phase 在规范化时吸收到 coefficient。这样相同算符可以可靠聚合。Native 模式初期支持 `complex64` 和 `complex128` coefficient；结构计划必须允许 coefficient 与结构分离，使 Python 端可在运行时提供 backend tensor weights。

重复项规范化返回 `input_to_canonical`、phase multiplier 和 canonical keys，而不只返回合并后的数值。这一映射允许 backend-plan 模式用 `tc.backend` 对动态或可微 weights 做 segment reduction。

### 7.3 PauliOperator

`PauliOperator` 是有序 canonical terms 与 coefficient buffer 的组合。核心操作包括 add、scale、multiply、commutator、anticommutator、adjoint、truncate-by-tolerance、filter-by-weight、grouping 和 Hermiticity validation。

对浮点 coefficient 不使用其数值作为 structural hash。结构 hash 只依赖 canonical keys、qubit count、ordering 和编译配置；包含静态 coefficient 的 artifact 另有 content hash。

### 7.4 GateTape

Rust-native propagation 不直接解析任意 Python function。`GateTape` 是受支持 gate 的紧凑序列，每个 operation 包含 gate enum、one or two wires、parameter slot 或 static parameter，以及可选 source index。参数化 tape 将结构与连续参数向量分离，因此同一 tape 可以重复计算新参数。

Python facade 可以从受支持的 TensorCircuit QIR 构造 fixed-parameter tape。对于可训练线路，第一版提供显式 `GateTape` builder；未来可以让 `SymbolCircuit` 或 TensorCircuit QIR 记录 parameter reference，但这不是 Rust core 的前置条件。

## 8. 功能模块

### 8.1 Pauli algebra

该模块提供单项和批量 Pauli 运算：字符串、整数 structure 与 symplectic word 的互转；乘法和 phase；commutation matrix；support/weight；重复项合并；deterministic sort；commuting 与 qubit-wise-commuting grouping。

Measurement grouping 区分 qubit-wise commuting（QWC）和 general commuting。QWC group 输出逐 qubit measurement basis，只需局部 basis rotation；general commuting group 还必须输出 Clifford diagonalization 或等价的 joint-measurement plan，不能把整体 commute 错当成可用单比特旋转共同测量。

Grouping 将 Pauli terms 作为图顶点，以 measurement incompatibility 作为边。第一版提供确定性的 largest-first greedy 和 DSATUR，并输出 group membership、basis-change plan、bitstring eigenvalue reconstruction masks 和 coefficient mapping。后续可以增加 multi-start coloring、hardware-depth constraints 和 coefficient/variance-aware shot allocation。Benchmark 必须同时报告 grouping time、group count、basis-change depth 和总 shot variance，避免只优化组数。

### 8.2 Hamiltonian compiler

Hamiltonian compiler 接收 canonical Pauli structure 和 coefficient，支持以下目标：

- `dense`：仅用于小系统验证和算法原型。
- `coo` / `csr`：直接生成 NumPy/SciPy-compatible indices、indptr 和 values，聚合重复 matrix entries。
- `native_mvp`：Rust CPU matrix-free application，避免物化 `2**n × 2**n` matrix。
- `backend_mvp_plan`：生成 bit masks、permutation 和 phase plan，由 `tc.backend` 构造可 JIT、可微的 MVP callable。
- `sector_csr` / `sector_mvp`：在给定 symmetry sector 或 fixed-Hamming-weight basis 中生成 restricted operator。

Pauli word 对 computational basis 的作用采用 XOR permutation 加 phase evaluation。矩阵构造按 term 和 basis block 并行，使用 Rayon；最终 COO/CSR 聚合必须确定性。系统过大时应根据估计的 rows、nonzeros 和 bytes 直接拒绝 dense/CSR target，并建议 MVP，而不是尝试分配后 OOM。

All CPU-native MVP plans default to compact lazy storage; eager dimension-scale diagonals or transition graphs require an explicit eager-plan request or an explicit restricted materialization such as CSR, COO, or dense output. A fixed plan never changes storage. A charge-restricted facade may build and retain a synchronized eager transition cache after materialization is requested, then reuse it for later facade execution without changing previously returned plans. This keeps the ordinary MVP path memory-safe while making materialization natural and amortizing its cost.

### 8.3 Symmetry engine

第一类能力是 Z2 Pauli symmetry。给定 Hamiltonian support，构造 binary symplectic commutation constraints，求 GF(2) null space，提取线性独立且彼此 commuting 的 symmetry generators，验证每个 generator 与完整 Hamiltonian commute，并允许用户选择 eigenvalue sector。Tapering 必须返回显式 Clifford transform、移除 qubits、sector signs 和可逆 provenance。

第二类能力是显式 U(1) particle-number sector。第一阶段不声称从任意 Pauli sum 自动发现 U(1)，而是接受已知 number operator 或 `particle_number=k` 配置，构造 fixed-Hamming-weight basis、rank/unrank map 和 restricted Hamiltonian plan，并验证目标 Hamiltonian 在数值容限内没有 sector leakage。普通 operator restriction 统一通过 `restrict_charge()`；eligible pure-qubit fixed-weight sectors dispatch to the optimized packed U1 backend, while spinful fermion and general multi-charge layouts retain their own specialized or generic backends. `U1Sector` and `U1Circuit` remain public basis/circuit contracts, but the separate `restrict_u1()` operator entry point is deprecated according to Phase 8.5.

后续可以扩展 parity、spin sectors、multiple commuting charges 和自动 symmetry suggestion，但所有自动建议必须经过 exact commutator validation 才能用于降维。

### 8.4 Pauli propagation engine

Propagation 采用 Heisenberg picture，对 GateTape 逆序执行。公开语义只有一个动态 recurrence，不要求用户选择 `clifford_exact`、`exact_dynamic` 或 `weight_truncated` mode。Clifford gate 对每个 Pauli word 自然只产生一个输出 word；非 Clifford local map 可以产生多个贡献。每次 local expansion 后必须先聚合相同 Pauli word，再在 `max_weight` 有限时删除 Pauli weight 超过该值的项。`max_weight=None` 或 `max_weight >= nqubits` 不执行 projection，因此恢复 exact propagation；任意有限 cutoff 都不设置 fixed buffer，也不执行 top-k。

Dynamic operator 默认使用 hash aggregation，并在公开输出或序列化前按 canonical key 确定性排序。并行传播可以为每个 worker 建立局部 map，再执行 deterministic merge，避免对全局 map 的高争用。`max_weight` 是 Pauli word 的结构属性，不依赖连续参数值，因此 weight projection 可以作为传播递推中的固定线性投影参与求导。Clifford fast path、exact propagation 和 weight-projected propagation 是同一 recurrence 的自然特例，而不是三套公开执行模式。

Phase 3 不提供 coefficient-magnitude cutoff。Phase 4 deterministic reverse显式冻结当次动态稀疏forward trace，并沿用exact-zero removal；这不是可调coefficient cutoff，也不声称等于support-change point的fixed-basis AD。结构性近似仍只使用公开的Pauli-weight projection。

第一批 native gate 支持为固定 Clifford gates，以及 `rx`、`ry`、`rz`、`rxx`、`ryy`、`rzz`。Static one- or two-qubit real PTM可以通过显式输入进入forward recurrence，并以固定transpose/VJP进入deterministic reverse；它没有parameter gradient，也不能自动进入SPPS。

PTM 使用 Hermitian Pauli basis `I/X/Y/Z`。若 local map 保持 Hermiticity，则 `R_ab = 2**(-m) Tr[P_a E(P_b)]` 必为实数；负值是正常的实数系数，不意味着需要 complex PTM。Phase 3 公开 custom PTM API 只接受 finite real `float64` arrays，complex dtype 明确失败。Complex entries 只对应非 Hermiticity-preserving 的一般线性 Pauli map、非 Hermitian basis，或数值噪声；未来若有真实需求，必须另设不宣称物理 PTM 的 complex linear-Pauli-map API。

初始态 expectation 第一阶段支持 `|0...0>`、computational basis product state 和 tensor-product single-qubit Bloch vector。stabilizer state、mixed product state 和一般 MPS 可以后续通过显式 expectation callback plan 增加，但不能让 Rust hot loop 回调 Python。

## 9. 两种执行模式

### 9.1 Rust-native 模式

Native 模式接受完整 GateTape、参数、初始 Pauli operator、initial-state descriptor 和 `max_weight` 配置，在 Rust 内完成传播与 expectation。它适合 CPU、动态线路规模、冷启动敏感、JAX tracing 成本高或不需要嵌入其他 JAX tensor program 的场景。

Native 模式的优势是可以使用动态 hash maps、Rayon、宽松的 best-effort memory guard 和无需静态 shape 的 Pauli weight projection。其限制是不能自动参与 `tc.backend.jit/grad/vmap`。Python API 必须明确命名为 native execution，不能伪装成普通 backend tensor primitive。

### 9.2 Backend-plan 模式

Backend-plan 模式由 Rust 完成 basis、canonicalization、transition table、duplicate mapping、symmetry basis 和 Hamiltonian masks 的生成，然后导出普通 integer/float arrays。Python facade 使用 `tc.backend` 构造纯 tensor callable。

该模式保留多 backend 和 AD，主要用于 Hamiltonian MVP、measurement reconstruction、固定 k-local basis 和 symmetry-restricted operator。它的价值是把组合结构、Python dict、basis generation 和稳定 mapping 移出 trace，并让相同 plan 在多次 JIT 与多组参数之间复用。动态 weight-truncated propagation 以 Rust-native 模式为主，不要求在 backend 中复刻同一容器算法。

## 10. 自动微分策略

### 10.1 第一阶段

第一阶段 native propagation 只保证 value correctness。需要梯度的用户继续使用 backend-plan 或现有 `tensorcircuit.pauliprop`。这样可以先验证 Rust 数据模型、传播语义和性能，不把 AD 实现变成项目启动的阻塞条件。

### 10.2 Deterministic frozen-support reverse gradient

第一类 native gradient 对受支持 Pauli rotations 使用解析 local derivative，并对给定参数下实际执行的dynamic sparse trace手写reverse mode。Forward未生成、local multiplier严格为零、聚合后严格抵消或被`max_weight`删除的term/contribution不进入反向；support decisions作为frozen control flow，不求导。

对retained edge，reverse mode保存或checkpoint/replay输入sparse states，计算`gradient[p] += input_coefficient * edge_derivative * output_adjoint`和`lambda_input += edge_multiplier * lambda_output`。多个gates引用同一parameter slot时确定性累加。Static custom PTM使用固定transpose/VJP且不产生parameter gradient。

该结果在一般固定support区域等价于相应sparse arithmetic的解析反向，但在trigonometric zero、exact cancellation、underflow-to-zero等support-change point不声称等于dense/fixed-basis AD或相邻参数极限。Phase 4不实现production parameter shift、forward sensitivity、通用Rust AD，也不承担相对exact objective的bias或optimization-trajectory研究；完整合同见`phase-4-spec.md`。

### 10.3 SPPS unbiased stochastic value-and-gradient

第二类 native gradient 实现 arXiv:2607.17804 的 stochastic Pauli-path simulator（SPPS）。SPPS 不构造或截断完整传播树，而是对每个 observable Pauli term 从完整 legal path space 逐 gate 采样一条路径；Clifford/commuting steps 确定性前进，反对易 Pauli rotation 按平滑后的 cosine/sine branch probability 采样。每条路径通过 importance reweighting 贡献无偏 value estimate，并通过 numerically stable path automatic differentiation（PAD）从同一路径贡献所有 active parameter gradients。

SPPS 必须支持 fixed sample budget 与基于两个独立 macro-replicates 的 adaptive A/B gradient-error proxy，提供显式 random seed、可复现的确定性调度、最大 sample budget、并行 path batching，以及零点附近采用 prefix/suffix products 的稳定 PAD。对多项 observable，第一版按 Pauli term 独立采样并线性组合；observable-term sampling 或 correlated sampling 不属于Phase 4。

SPPS 首版只承诺 Clifford gates 与具有论文所需二分 trigonometric branch rule 的参数化 Pauli rotations。任意 custom PTM 不会仅因 forward recurrence 可用而自动获得 SPPS 支持；它还必须显式提供合法 transition sampling distribution、importance weight、parameter derivative 和稳定 local PAD rule。

Deterministic frozen-support reverse与SPPS都为REQUIRED。前者提供确定、可重复且高性能的executed sparse-trace反向；后者的fixed-budget mode提供相对完整path expansion的无偏随机估计但带有sampling variance。测试、文档和benchmark必须分别验证各自合同、runtime和memory，不能把一者描述为另一者的替代品。

初期不使用 Python callback 把 Rust native engine 强行包装进 JAX `jit`。只有在 native gradient 稳定且存在明确需求时，才评估 JAX custom call/custom VJP；这属于独立工程，不作为本项目成功的必要条件。

## 11. 建议 Python API

~~~python
import tencirpauli as tcp

h = tcp.PauliOperator.from_terms(
    nqubits=100,
    terms=[("ZZ" + "I" * 98, 1.0), ("X" + "I" * 99, 0.5)],
)

groups = h.group_commuting(mode="qubit_wise")
symmetries = h.find_z2_symmetries()
mvp_plan = h.compile(target="backend_mvp")
~~~

~~~python
from tencirpauli import advanced

tape = advanced.GateTape(100)
tape.rxx(0, 1, parameter=0)
tape.ryy(0, 1, parameter=1)
tape.rzz(0, 1, parameter=2)

engine = advanced.PropagationEngine(
    tape,
    max_weight=3,
)

value = engine.expectation(h, params, initial_state="zero")
operator_result = engine.propagate_operator(h, params)

projected = engine.value_and_grad(h, params, initial_state="zero")
spps = tcp.StochasticPauliPathEngine(tape, sample_budget=4096, seed=7)
stochastic = spps.value_and_grad(h, params, initial_state="zero")
~~~

长期集成 API 可以由 TensorCircuit 暴露为 `tc.pauli`，但第一版独立 distribution 应使用明确的 `tencirpauli` import，避免核心包在 native extension 缺失时发生隐式行为变化。若用户显式请求 native engine 而扩展未安装，应直接给出安装指引并失败，不做静默 fallback。

## 12. 独立仓库、开发环境与发布

### 12.1 仓库建议

建议建立独立仓库和独立 Python distribution，例如仓库 `TenCirPauli`、Python package `tencirpauli`。不要在第一阶段把 Cargo workspace 放进 TensorCircuit 主仓库，也不要让 TensorCircuit 的基础安装依赖 Rust wheel。

独立仓库的优势是 Rust/maturin 发布周期、CI 平台矩阵和版本号可以独立管理；Pauli core 也可以被其他量子框架或纯 Rust 程序复用。与 TensorCircuit 的耦合集中在一个薄 adapter：TensorCircuit QIR/GateTape 转换、qubit ordering、gate convention 和 backend-plan loading。集成测试在 TenCirPauli CI 中固定一组受支持的 TensorCircuit 版本。

原型阶段可以把新仓库放在 TensorCircuit 仓库的相邻目录，并把当前 TensorCircuit 以 editable mode 安装到同一 Conda 环境。等 API 稳定后，再决定是否在 TensorCircuit 中增加 `tc.pauli` lazy adapter。

### 12.2 Rust workspace

第一版只建议两个 crate：

~~~text
TenCirPauli/
├── Cargo.toml
├── crates/
│   ├── tencir-pauli-core/      # algebra, Hamiltonian, symmetry, propagation
│   └── tencirpauli-native/    # PyO3 and NumPy facade
└── python/                 # Python package, typing, high-level adapters
~~~

在模块边界和维护团队扩大前，不提前把 algebra、symmetry 和 propagation 拆成更多 crate。核心 crate 不依赖 Python，便于 fuzz/property testing 和未来 CLI/WASM 复用；Python crate 使用 PyO3、numpy bindings 和 maturin。发布优先评估 `abi3`，目标覆盖项目支持的 Python 版本以及 Linux、macOS x86_64/arm64 和 Windows wheels。

TensorCircuit 是 TenCirPauli Python distribution 的必需 runtime dependency；Rust extension 是 TenCirPauli 执行路径的必需组件，不改变 TensorCircuit 当前 setuptools 构建。TensorCircuit-facing adapter 仍然只位于 Python 边界，Rust core 不直接导入 framework。

### 12.3 开发环境

原型开发使用独立 Conda 环境即可。最简单的隔离方案是在环境中安装 Python、Rust/Cargo 和 maturin：

~~~bash
conda create -n tencirpauli-dev python=3.11 pip
conda activate tencirpauli-dev
conda install -c conda-forge rust maturin

rustc --version
cargo --version
maturin --version
~~~

在 TenCirPauli 仓库中运行 `maturin develop --release`，即可把当前 Rust extension 构建并安装到激活的 Conda 环境。TensorCircuit 使用 editable install 指向相邻源码仓库，从而同时测试 Python adapter 与 Rust extension。

长期维护更推荐用官方 `rustup` 管理 Rust stable toolchain、`rustfmt`、`clippy` 和 cross-compilation targets，同时仍在 Conda 环境中管理 Python 与 maturin。两种方式只选一种 Rust toolchain，避免 Conda Rust 与 rustup Rust 在 `PATH` 中混用。macOS 如果缺少 linker，需要先具备 Xcode Command Line Tools；Linux wheel 发布使用 manylinux CI，而不是依赖开发机环境。

## 13. FFI、并发与内存规则

- NumPy 输入按批量二维 arrays 或 packed buffers 传递，能借用只读内存时避免复制。
- Rust 返回 arrays、plan capsule 或不可变 Python object，不返回包含 Python callable 的 Rust state。
- 所有长时间 Rust 计算释放 GIL。
- Rayon thread count 可由显式 option 或专用环境变量控制，避免和 BLAS、JAX CPU thread pool 形成不可控 oversubscription。
- plan 和 operator object 在并发读取时必须线程安全；带 mutable scratch buffer 的 engine 不允许无保护共享。
- 所有公开 API 的统一默认 memory budget 为 16 GiB。对可廉价估算的主要目标输出和 workspace 提供 best-effort `max_bytes` guard，允许调用者提高或显式关闭该 guard，并始终 checked dimension/arithmetic overflow；不把该 guard 描述成包含 allocator overhead、FFI conversion 和所有 transient scratch 的精确 peak-RSS limit，也不承诺避免操作系统 OOM。估算只应使用随维度、term/transition 数量直接可得的宽松 major-buffer 上界；不得为了把 `max_bytes` 算准而增加额外 dry run、逐元素记账、allocator 查询或可感知的热路径运行时间。
- artifact serialization 使用版本化、确定性的格式；不序列化 Python object 地址、hash seed 或 backend device object。

## 14. 正确性与验证

### 14.1 Algebra property tests

- Pauli multiplication 与小规模 dense matrix reference 一致，包括 phase。
- multiplication associativity、adjoint involution、commutation symmetry 和 canonical round trip。
- batch canonicalization 与 Python reference 对重复项、零项和 complex coefficient 的结果一致。
- qubit ordering 对 `X/Y/Z` 在首尾 qubit 的 matrix action 与 TensorCircuit 一致。

### 14.2 Hamiltonian tests

- dense、COO、CSR、native MVP 和 backend MVP 在小系统逐元素一致。
- real Hermitian Pauli sum 生成 Hermitian matrix。
- duplicate matrix entries 正确聚合。
- restricted-sector operator 与 full matrix projection `P H P` 一致。
- 超出 memory limit 或 index width 的 target 明确失败。

### 14.3 Symmetry tests

- 每个报告的 generator 都与 Hamiltonian 精确 commute。
- generators 线性独立且彼此 commute。
- tapering 前后目标 sector spectrum 一致。
- U(1) restricted plan 与现有 `U1Circuit` 和 full-state reference 一致。
- 对不守恒 Hamiltonian，sector validation 必须失败而不是静默投影。

### 14.4 Propagation tests

- Clifford propagation 与 Stim/TensorCircuit small-circuit reference 一致。
- `max_weight=None` 或 `max_weight >= nqubits` 的统一 recurrence 与 dense state expectation 一致。
- 有限 `max_weight` 的统一 recurrence 严格执行 duplicate aggregation 后再应用 Pauli weight projection，不包含 fixed buffer、top-k 或 coefficient cutoff。
- Rust 与现有 `PauliPropagationEngine` 在相同 Pauli weight cutoff 下比较 value；现有 `SparsePauliPropagationEngine` 只作为独立性能参考，不要求复制其 truncation 语义。
- deterministic frozen-support reverse与独立sparse-trace reference在小系统比较；generic nonsingular fixtures可用外部Python parameter-shift helper交叉验证。
- SPPS value/gradient在可exact枚举的小系统上验证proposal-weighted estimator、fixed-seed reproducibility、A/B proxy behavior和trigonometric zero附近的stable PAD。
- 零角度Pauli rotation对deterministic与SPPS分别验证各自不同合同；共享parameter slots和weight boundary transitions有专门tests。
- fuzz 随机 Pauli sums、随机 Clifford/rotation tapes 和随机 qubit ordering。

## 15. Benchmark 设计与性能策略

所有 benchmark 分开记录 input conversion、plan construction、first execution、steady execution、gradient、peak host memory 和 result error。JAX 对照必须分别报告 tracing/compilation 和 warm execution，不能只把 Rust cold call 与 JAX warm call 比较，反过来也不可以。

### 15.1 Algebra 与 Hamiltonian

测试 20–1000 qubits、`10**3`–`10**6` Pauli terms 的 parse、canonicalization、deduplication、commutation matrix 和 grouping；测试 10–24 qubits、不同 term counts 的 COO/CSR 构造；测试更大系统的 MVP plan construction。

性能不设置固定倍数的 pass/fail 门槛。每个 material hot path 都必须保留可复现的 release benchmark、与最佳适用 Python/TensorCircuit/JAX baseline 的等价语义对照、profile 证据和优化前后结果；在 correctness gate 通过后持续消除已识别的主要瓶颈，而不是达到某个倍数后停止。

### 15.2 Propagation

使用现有 12-qubit TFIM dense PPE、2D Heisenberg 和 100-qubit Pauli propagation 示例的线路与 Hamiltonian 作为起点，但统一改用 `max_weight` 配置。增加 Clifford-heavy、Pauli-rotation-heavy、duplicate-heavy 和不同 weight-growth profile 的 workload。

传播性能的主要基线是同步后的 JAX CPU warm-JIT steady runtime；非 JIT Python、JAX trace/compile、first execution、one-shot end-to-end、peak memory 和 result validation 分开记录，但不能用 cold compile 优势替代 steady-runtime 对比。Python 调 Rust 的 steady benchmark 必须复用 native tape/operator handle，只在热调用中传递参数和返回 scalar 或 gradient；完整 operator 只在显式请求时跨 FFI。Deterministic frozen-support reverse在generic fixed-support fixtures对照matched JAX/reference；SPPS报告sample budget、paths/s、proxy和parallel scaling，不要求optimization-level研究。

### 15.3 Symmetry

使用 TFIM、XXZ、Heisenberg、Hubbard 映射和随机 Pauli Hamiltonian，报告 symmetry analysis time、减少的 qubit/sector dimension、matrix/MVP memory 和端到端 eigensolver/VQE 收益。对称性模块的主要指标不是单独分析速度，而是可靠降维后的端到端时间与内存。

## 16. 分阶段路线

### 阶段零：语义冻结和 baseline

定义 Pauli phase、qubit ordering、dtype、duplicate aggregation、Pauli weight projection、GateTape 和 error behavior。明确 propagation 不提供 coefficient cutoff。整理当前 Python/JAX correctness 与性能 baseline。此阶段不发布 wheel。

### 阶段一：Pauli core 与 Hamiltonian compiler

实现 PauliWord、PauliOperator、批量 canonicalization、commutation/grouping、dense/COO/CSR、native MVP、backend MVP plan 和 PyO3 facade。完成 cross-platform wheel smoke test。

### 阶段二：Symmetry 与 sector plan

实现 Z2 generator analysis、sector validation、基础 tapering，以及显式 U(1) fixed-particle basis/restricted Hamiltonian plan。该阶段的 native restricted Hamiltonian 使用单字 `usize` basis index，只覆盖 `nqubits < usize::BITS`；阶段二不包含 64+ qubit multiword restriction 或 circuit/time-evolution execution。

### 阶段三：Rust-native propagation

实现一个统一的 Rust-native dynamic propagation recurrence：Clifford gate 自动使用不分支 fast path，`max_weight=None` 或足够大时自动 exact，有限 `max_weight` 时在聚合后应用 Pauli-weight projection。支持首批 rotation gates、显式 custom local PTM、Rust 内 expectation、按需 operator materialization、可复用 Python native handles，以及 initial/final/peak term count 与可选 timing metadata。该阶段不实现 fixed-buffer sparse engine、top-k、coefficient cutoff、discarded-norm error estimate 或 native gradient。

### 阶段四：双梯度引擎与 TensorCircuit integration

同时实现两类REQUIRED gradient。第一类为deterministic frozen-support reverse：对受支持rotation gates实现analytic local VJP、reverse mode和checkpointing，只反传当次forward实际保留的nonzero sparse trace。第二类为arXiv:2607.17804 SPPS：实现sequential path sampling、importance reweighting、stable PAD、fixed/adaptive sample budgets、A/B proxy、seeded reproducibility和parallel path batching。完成TenCirPauli侧 TensorCircuit QIR/SymbolCircuit adapter、文档和examples；不开发JAX custom call，也不加入bias/optimization-trajectory研究。

### 阶段五：任意宽 multiword U1 Hamiltonian engine

已完成：full-space U1 occupation word 使用任意数量的 packed `u64` limbs，不在128 qubits处建立新边界。Restricted-space logical index和公开sparse index保持有界`u64`，native内存寻址前执行checked `usize`/NumPy `intp` gate；wide low-particle/low-hole source/destination transitions、combinatorial rank/lookup、sector-preservation validation、restricted MVP和CSR已贯通，并保持现有Python `U1Sector`/`U1RestrictedOperator`语义兼容。

该阶段以63/64/65、127/128/129和256 qubits的low-k/low-hole differential tests为correctness gate，并记录word count、term/X-group count、particle number、sector dimension、setup、steady MVP、CSR storage和scaling。它不包含circuit execution或time evolution；这些属于阶段六。完整冻结合同见`phase-5-spec.md`。

### 阶段五点五：多 Observable deterministic propagation（已实现）

在不改变现有单个`PauliOperator` sum expectation/frozen-support gradient语义的前提下，可增加一个独立`PropagationBatch`。多个observables共享immutable compiled GateTape、state和parameter metadata，但每个observable完整复用现有forward/reverse内核，并只在observable维度由Rayon并行；不同observables不共享aggregation、projection或frozen support。

该阶段已按 spec 完成：`PropagationBatch` 共享一个 immutable compiled program，执行空/单/多 observable 的 expectations 与 row-wise frozen-support values/gradients，并在工作量达到 private threshold 时只沿 observable 维度使用 Rayon；小 workload 走串行路径。coefficient-batched keys、Clifford frame、inner term parallelism和batch SPPS均不属于本阶段。完整合同与验证证据见`phase-5.5-spec.md`与`implementation-status.md`。

### 阶段六：common circuit IR 与 Rust-native U1Circuit

在 GateTape/adapter 边界和 multiword U1 engine 完成后，backend-neutral common circuit layer 统一 typed gate semantics、实际 `theta` 值、私有 occurrence-angle slots 和 deterministic Python/Rust serialization。Common IR 不计算 `2**n`，不包含 U1 sector/rank/pair map，也不承诺 public 通用模拟器；未来 execution mode 可以消费同一 logical representation，而不重复 gate 协议。用户参数的共享、算术关系和 PyTree 由 JAX 或调用方处理，native 层只接收 flat `f64` angle arrays。

Phase 6唯一实现的execution backend与TensorCircuit `U1Circuit`常用构造、gate名称、basis ordering和observable语义对齐。Python gate methods只记录common typed operations；`state()`、`expectation()`、`value_and_grad()`和`expectation_jax()`以一次coarse native call消费私有编译计划，由U1 compiler完成sector validation、fusion、pair-map construction、restricted-state execution和reduction。Required gate set为RZ、RZZ、CZ、CPhase、SWAP、TensorCircuit-convention iSWAP和bounded static diagonal，并支持任意宽low-k/low-hole sectors。通用full-state和tensor-network simulator都不属于本阶段。

Phase 6实现普通restricted statevector的精确adjoint gradient：forward只保留final state，reverse通过unitary inverses同时重建pre-gate state和传播adjoint state。Required bounded terminals包含`state_full()`和`probability_full()`；static diagonal严格幺正；Givens/fSim/public general block不进入首版。它不包含time-evolution solver、noise、sampling/RDM/entropy、automatic Trotter、JAX custom call或GPU。完整冻结合同见`phase-6-spec.md`。

### 阶段六点五：generic Rust-native matrix-free time evolution（deferred proposal）

Phase 6.5与U1语义解耦，接受已经在Rust中的full-space `MvpPlan`、restricted `U1MvpPlan`或兼容native operator handle，在一次coarse call中完成Taylor expm-multiply、Hermitian Krylov/Lanczos或Chebyshev exponential action。所有重复MVP、vector recurrence、error checks和scratch reuse留在Rust；若未来恢复，首版只做time-independent Hermitian real-time forward evolution，可选返回state、selected-time trajectory、native observable reductions和显式time derivative。

该阶段当前搁置，不是Phase 6之后的自动下一里程碑，也不得为其提前引入general linear algebra abstraction、small eigensolver或Bessel dependency。重新启动必须有真实workload、matched baseline、dependency/accuracy spike和新的owner decision。它不实现time-dependent Hamiltonian、Python callback、ODE、automatic Trotter、general autodiff、JAX custom call或accelerator integration。完整deferred proposal见`phase-6.5-spec.md`。

### 阶段七：structured Hamiltonian algebra and compiler（frozen future contract）

Phase 7 is expanded from the earlier qudit-only roadmap into a structured Hamiltonian algebra and compilation milestone. It covers Pauli-compatible downstream compilation plus separate fermion, infinite-Fock symbolic boson, finite mixed-basis hybrid, and uniform-dimension direct-Weyl operator domains. The frozen implementation contract, scope, non-goals, finite-basis ordering, TensorCircuit boundary, and Python construction API are recorded in `docs/vibe/phase-7-spec.md`.

The contract uses the existing best-effort 16 GiB default and a unified `compile()` target vocabulary across current Pauli and new operator types. Fermions use Jordan-Wigner only. Bosons use exact symbolic CCR canonicalization followed by explicit inclusive-cutoff `P O P` compilation with an open boundary; no public finite-boson algebra is added. Hybrid mixed-dimension execution is native-only, while TensorCircuit backend MVP remains required for Pauli-compatible and uniform-qudit plans. Direct Weyl words store modular phase exponents before complex128 conversion. The Python construction contract uses a compact default `OperatorSpace`, readable factories, overloaded `+`/`-`/`*`, explicit `tensor_product()`, low-level `from_terms()`, and a batched `OperatorBuilder` without exposing a universal algebra trait or a large public limits object.

## 17. 主要风险与控制措施

### 17.1 Term explosion

非 Clifford gate 会让 Pauli term 数指数增长。Rust 只能降低常数，不能改变复杂度。Deterministic forward/reverse的公开结构控制只使用`max_weight`与宽松的best-effort 16 GiB memory budget，不使用可调coefficient cutoff；它显式沿用exact-zero sparse removal和frozen-support反向。SPPS不构造完整传播树，但其sampling variance和所需sample count仍可随有效branching factor快速增长。

### 17.2 Native gradient 实现复杂度

Rust 不具备现成的 TensorCircuit backend AD graph。Deterministic frozen-support reverse必须维护retained-edge local VJP、parameter-slot accumulation、checkpoint和term aggregation的反向映射；控制措施是复用实际forward trace、只对Pauli rotations求parameter derivative，并让static PTM仅使用固定transpose/VJP。SPPS必须另外维护sampling distribution、importance weights、stable PAD、randomness和proxy语义；custom PTM不进入Phase 4 SPPS。

### 17.3 SPPS variance 与复现性

SPPS 的无偏性不等于低方差或任意线路上的高效率。控制措施是同时提供fixed budget与adaptive A/B proxy、显式最大sample budget、seeded deterministic replay和独立exact small-system path-enumeration tests；性能记录budget、paths/s、memory和parallel scaling，不扩展为variance规律研究。并行实现不得因线程调度改变同一deterministic-replay配置的结果。

### 17.4 与 TensorCircuit 语义漂移

gate angle、qubit ordering、global phase、dtype 和 QIR gate naming 变化都可能破坏结果。控制措施是稳定 adapter 层、版本 compatibility matrix 和与 TensorCircuit tests 的 differential suite；Rust core 不直接依赖 Python callable 名称。

### 17.5 Native extension 发布成本

PyO3 wheel 增加平台矩阵和维护门槛。控制措施是独立可选 distribution、`abi3` 可行性验证、最小 crate 数量和纯 Python core 安装不受影响。

### 17.6 Rust 与 backend 重复实现

同一算法可能出现 native 与 backend 两套逻辑。控制措施是共享 canonical plan、固定语义测试向量和明确职责：Rust-native 负责统一 dynamic recurrence 下的 exact/finite-weight CPU propagation；backend-plan 负责 Hamiltonian、measurement 和固定 basis 的 AD/JIT/accelerator execution。两者不要求逐行同构，但对共同支持的语义必须给出一致结果。

## 18. 性能决策策略

项目不以预设 speedup 倍数决定是否停止某条 REQUIRED 路线，也不因阶段三初版性能不足而取消阶段四的两类 gradient。每个阶段必须提供 correctness、warm-JIT steady comparison、cold/setup、memory、profile 和 scaling evidence；发现 Rust 落后时继续定位表示、算法、allocation、parallelism 或 FFI bottleneck，并用同机 release benchmark 记录改进。任何具体优化仍需以 correctness gate 和代表性 end-to-end evidence 为前提。

## 19. 关键开放问题

- Phase 4之后是否需要上游TensorCircuit增加`tc.pauli`入口；TenCirPauli侧先提供 TensorCircuit-facing QIR/SymbolCircuit adapter。
- Deterministic frozen-support reverse的auto checkpoint heuristic可按profile演进，但显式interval和结果语义已由Phase 4 spec冻结。
- SPPS后续是否增加adaptive smoothing、observable-term sampling或correlated sampling；这些均不属于Phase 4。
- Z2 tapering 是否第一版实现完整 Clifford transform，还是先只提供 symmetry generators 与 sector projector。
- Backend MVP plan 的 portable integer dtype 如何兼容 JAX 默认关闭 int64、TensorFlow 和 PyTorch。
- Phase 6.5需要通过P0 spike冻结Krylov orthogonalization和small symmetric eigensolver dependency、Chebyshev Bessel implementation、Taylor schedule defaults以及spectral-bound safety margin。

## 20. 推荐结论

TenCirPauli 值得进入原型阶段。它与 TensorCircuit 的 Hamiltonian、VQE、time evolution、U(1) 和 Heisenberg-picture 能力直接相连，同时把 Rust 放在其最擅长且不会破坏 backend abstraction 的位置。

推荐在独立仓库中从阶段零和阶段一开始，不承诺完整 native autodiff。第一交付物应该是一个可 benchmark、可验证、面向 TensorCircuit 用户发布的 Pauli algebra、measurement grouping 与 Hamiltonian engine；第二交付物是 forward-only native weight-truncated propagation。每个阶段以真实 TensorCircuit workload 的端到端结果决定是否继续。
