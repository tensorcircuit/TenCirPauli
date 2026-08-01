# TenCirPauli 设计方案

状态：概念设计，建议进入原型验证。项目名称为 TenCirPauli，Python distribution 与 import package 均为 `tencirpauli`。

## 1. 决策摘要

TenCirPauli 围绕 Pauli 代数、Pauli 算符、Hamiltonian 生成、对称性分析和 Pauli propagation 建立一个可选的 Rust 原生扩展。它不替代 TensorCircuit 的 tensor backend，也不把通用量子线路数值计算搬到 Rust。Rust 负责离散、bit-packed、CPU 密集且适合批量执行的结构化工作；JAX、TensorFlow、PyTorch 和 NumPy 继续负责需要 backend tensor、自动微分或加速器执行的数值工作。

项目采用两种互补执行模式：Rust-native 模式在 CPU 上完成动态 Pauli operator propagation，并逐步提供原生梯度接口，用于规避 Pauli term 集合变化、重复项聚合和结构增长在 JAX 中引起的 tracing、静态 shape 与编译开销；backend-plan 模式由 Rust 生成稳定的代数、Hamiltonian 和 measurement plan，再由 `tc.backend` 执行需要多 backend、JIT 或加速器的数值计算。

建议首先实现 Pauli 核心表示、算符规范化、measurement grouping、Hamiltonian matrix/MVP plan 和基准体系，然后实现 forward-only Rust-native weight-truncated propagation。对称性分析和原生梯度在核心语义稳定后增加。Rust-native propagation 是否成为默认推荐路径，必须由端到端 benchmark 决定。

## 2. 背景与现状

TensorCircuit 当前已经包含三组相关能力：`tensorcircuit/quantum.py` 中的 `PauliStringSum2MVP`、`PauliStringSum2COO` 和 dense/sparse Hamiltonian 构造；`tensorcircuit/pauliprop.py` 中的 dense k-local 与 sparse bit-packed Pauli propagation；`tensorcircuit/u1circuit.py` 中的固定粒子数子空间模拟和 Pauli observable 支持。

这些实现已经验证了 Pauli 表示对 VQE、time evolution、ODE、局部 observable 和百比特近似传播的价值，但也暴露出适合 Rust 的瓶颈：组合 basis 构造和 neighbor map 依赖多层 Python 循环；传播过程中 Pauli term 集合会动态增长、合并并跨 weight sector 移动，而 JAX 更适合静态 shape 的 dense/fixed-buffer 计算；bit-packed 数据在 backend 中受 dtype 和 JIT 语义约束；大型 Hamiltonian 的规范化、重复项合并、commutation 分组和稀疏矩阵生成缺少统一的高性能抽象。

TenCirPauli 不移植当前 fixed-buffer `SparsePauliPropagationEngine`，也不在 Rust 中实现 top-k sparsity truncation。Native propagation 使用动态 Pauli operator 容器，先精确聚合相同 Pauli word，再按 Pauli weight/locality `w(P) <= k` 做结构性投影。可选的 coefficient-magnitude cutoff 只作为显式 forward 策略，第一版不纳入自动微分保证。传播按 Heisenberg picture 逆序应用 gates，现有 one- and two-qubit gate 支持范围之外的 gate 必须明确失败。

## 3. 项目定位与技术壁垒

TenCirPauli 的核心技术壁垒不是单个 bit operation，而是一套与 TensorCircuit 语义一致的端到端 Pauli operator pipeline：同一 canonical representation 同时服务算符代数、measurement grouping、Hamiltonian 生成、Z2/U(1) 对称性处理、weight-truncated propagation、TensorCircuit backend plan 和可验证的近似截断。

相比独立的 Pauli 工具包，TenCirPauli 的差异化价值是：直接接受和生成 TensorCircuit 的 observable、Hamiltonian 与 gate tape；同时覆盖 exact/static preprocessing 与 approximate propagation；明确处理 qubit ordering、gate convention、complex dtype、gradient 和截断语义；能在 native CPU 与 backend-compatible 模式之间共享同一结构计划。

## 4. 设计目标

- 为任意 qubit 数提供紧凑、确定性、可 hash 的 Pauli word 和 Pauli operator 表示。
- 批量完成 Pauli multiplication、commutation、support、weight、canonicalization、deduplication 和 grouping。
- 快速生成 dense、COO/CSR、matrix-free MVP 和 symmetry-restricted Hamiltonian plan。
- 支持 Z2 Pauli symmetry 的发现、验证、sector 选择和 tapering；为显式给定粒子数守恒的 U(1) sector 提供 restricted basis 与 operator plan。
- 提供 exact Clifford propagation、有限 gate set 的 exact Pauli-transfer propagation，以及基于 Pauli weight/locality cutoff 的动态 propagation。
- 提供无需 JAX tracing 的 Rust-native forward 路径，并在后续阶段提供受支持 gate set 的 `value_and_grad`。
- 提供 backend-plan 路径，使结构计算离开 JIT hot path，同时保留 `tc.backend` 数值执行和自动微分。
- 保持核心 TensorCircuit 纯 Python 安装可用；Rust 扩展作为显式可选依赖发布。

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

### 8.3 Symmetry engine

第一类能力是 Z2 Pauli symmetry。给定 Hamiltonian support，构造 binary symplectic commutation constraints，求 GF(2) null space，提取线性独立且彼此 commuting 的 symmetry generators，验证每个 generator 与完整 Hamiltonian commute，并允许用户选择 eigenvalue sector。Tapering 必须返回显式 Clifford transform、移除 qubits、sector signs 和可逆 provenance。

第二类能力是显式 U(1) particle-number sector。第一阶段不声称从任意 Pauli sum 自动发现 U(1)，而是接受已知 number operator 或 `particle_number=k` 配置，构造 fixed-Hamming-weight basis、rank/unrank map 和 restricted Hamiltonian plan，并验证目标 Hamiltonian 在数值容限内没有 sector leakage。该能力应与现有 `U1Circuit` 的 basis ordering 和 observable 语义对齐，而不是建立第二套不兼容约定。

后续可以扩展 parity、spin sectors、multiple commuting charges 和自动 symmetry suggestion，但所有自动建议必须经过 exact commutator validation 才能用于降维。

### 8.4 Pauli propagation engine

Propagation 采用 Heisenberg picture，对 GateTape 逆序执行。内部提供三种策略：

- `clifford_exact`：一个 Pauli word 在 Clifford gate 下映射为一个 Pauli word 和 phase。
- `exact_dynamic`：使用动态 Pauli operator 容器传播全部生成项，只受显式 hard memory limit 约束。
- `weight_truncated`：每次 local PTM expansion 后先聚合相同 Pauli word，再删除 Pauli weight 超过 `max_weight` 的项，不设置 fixed buffer，也不执行 top-k。

Dynamic operator 默认使用 hash aggregation，并在公开输出或序列化前按 canonical key 确定性排序。并行传播可以为每个 worker 建立局部 map，再执行 deterministic merge，避免对全局 map 的高争用。`max_weight` 是 Pauli word 的结构属性，不依赖连续参数值，因此 weight projection 可以作为传播递推中的固定线性投影参与求导。

可选的 coefficient cutoff 在聚合后删除 `abs(c_P) < epsilon` 的项，用于 forward-only 内存控制。它默认关闭，并且不能出现在第一版 `value_and_grad` 路径中。若未来支持其梯度语义，必须作为单独模式设计，不能影响基于 Pauli weight 的默认传播。

第一批 native gate 支持为固定 Clifford gates，以及 `rx`、`ry`、`rz`、`rxx`、`ryy`、`rzz` 和 TensorCircuit 中语义明确的 one- and two-qubit Pauli rotations。通用 unitary 可以通过显式 PTM 输入进入 forward-only 路径，但在没有 derivative rule 时不能进入 native gradient API。

初始态 expectation 第一阶段支持 `|0...0>`、computational basis product state 和 tensor-product single-qubit Bloch vector。stabilizer state、mixed product state 和一般 MPS 可以后续通过显式 expectation callback plan 增加，但不能让 Rust hot loop 回调 Python。

## 9. 两种执行模式

### 9.1 Rust-native 模式

Native 模式接受完整 GateTape、参数、初始 Pauli operator、initial-state descriptor 和 `max_weight` 配置，在 Rust 内完成传播与 expectation。它适合 CPU、动态线路规模、冷启动敏感、JAX tracing 成本高或不需要嵌入其他 JAX tensor program 的场景。

Native 模式的优势是可以使用动态 hash maps、Rayon、精确内存控制和无需静态 shape 的 Pauli weight projection。其限制是不能自动参与 `tc.backend.jit/grad/vmap`。Python API 必须明确命名为 native execution，不能伪装成普通 backend tensor primitive。

### 9.2 Backend-plan 模式

Backend-plan 模式由 Rust 完成 basis、canonicalization、transition table、duplicate mapping、symmetry basis 和 Hamiltonian masks 的生成，然后导出普通 integer/float arrays。Python facade 使用 `tc.backend` 构造纯 tensor callable。

该模式保留多 backend 和 AD，主要用于 Hamiltonian MVP、measurement reconstruction、固定 k-local basis 和 symmetry-restricted operator。它的价值是把组合结构、Python dict、basis generation 和稳定 mapping 移出 trace，并让相同 plan 在多次 JIT 与多组参数之间复用。动态 weight-truncated propagation 以 Rust-native 模式为主，不要求在 backend 中复刻同一容器算法。

## 10. 自动微分策略

### 10.1 第一阶段

第一阶段 native propagation 只保证 value correctness。需要梯度的用户继续使用 backend-plan 或现有 `tensorcircuit.pauliprop`。这样可以先验证 Rust 数据模型、传播语义和性能，不把 AD 实现变成项目启动的阻塞条件。

### 10.2 Native value-and-gradient

后续 native gradient 对受支持 Pauli rotations 使用解析 PTM derivative。Hamiltonian coefficient 的梯度利用线性结构直接计算。多参数线路优先实现 reverse mode，并允许 checkpoint interval 控制保存中间 Pauli operators 与重计算之间的权衡；参数很少时可以提供 forward sensitivity mode。

令一次反向线路传播写成 `s[r + 1] = Projection_k(M_r(theta_r) s[r])`，其中 `Projection_k` 只根据 Pauli weight 删除 `w(P) > k` 的 basis terms。由于该投影与参数值无关，truncated model 对参数仍然可以正常求导。Reverse mode 保存或重算 `s[r]`，并使用 local PTM derivative 计算 `dE/dtheta_r = lambda[r + 1]^T Projection_k(dM_r/dtheta_r s[r])`，再传播 `lambda[r] = M_r(theta_r)^T Projection_k(lambda[r + 1])`。

对 `rx`、`ry`、`rz`、`rxx`、`ryy`、`rzz` 等 Pauli rotations，直接实现解析 conjugation 和 derivative rule，不通过通用 Rust autodiff crate 对 hash map 程序求导。多个 gates 引用同一 parameter slot 时，其梯度贡献确定性累加。即使某个角度当前为零，也不能因为对应 branch coefficient 为零而删除其 derivative path；只有 parameter-independent 的 Pauli weight projection 可以进入保证正确的 gradient 路径。

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
tape = tcp.GateTape(100)
tape.rxx(0, 1, parameter=0)
tape.ryy(0, 1, parameter=1)
tape.rzz(0, 1, parameter=2)

engine = tcp.PropagationEngine(
    tape,
    max_weight=3,
    mode="weight_truncated",
)

result = engine.value_and_grad(h, params, initial_state="zero")
value = result.value
gradient = result.gradient
discarded_weight_norm = result.discarded_weight_norm
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

Rust 扩展保持可选，不改变 TensorCircuit 当前 setuptools 构建。TensorCircuit core 中只增加轻量 adapter 和类型协议，并且只在用户调用 native API 时导入扩展。

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
- 所有目标分配在执行前估算 bytes，并提供 hard memory limit。
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
- `exact_dynamic` propagation 与 dense state expectation 一致。
- `weight_truncated` engine 严格执行 duplicate aggregation 后再应用 Pauli weight projection，不包含 fixed buffer 或 top-k。
- Rust 与现有 `PauliPropagationEngine` 在相同 Pauli weight cutoff 下比较 value；现有 `SparsePauliPropagationEngine` 只作为独立性能参考，不要求复制其 truncation 语义。
- native gradient 与 TensorCircuit/JAX 对同一个 weight-truncated recurrence 的 gradient 在小系统比较。
- 零角度 Pauli rotation、共享 parameter slots 和 weight boundary transitions 均有专门 gradient tests。
- fuzz 随机 Pauli sums、随机 Clifford/rotation tapes 和随机 qubit ordering。

## 15. Benchmark 设计与验收门槛

所有 benchmark 分开记录 input conversion、plan construction、first execution、steady execution、gradient、peak host memory 和 result error。JAX 对照必须分别报告 tracing/compilation 和 warm execution，不能只把 Rust cold call 与 JAX warm call 比较，反过来也不可以。

### 15.1 Algebra 与 Hamiltonian

测试 20–1000 qubits、`10**3`–`10**6` Pauli terms 的 parse、canonicalization、deduplication、commutation matrix 和 grouping；测试 10–24 qubits、不同 term counts 的 COO/CSR 构造；测试更大系统的 MVP plan construction。

原型进入产品阶段的建议门槛是：大批量结构操作相对当前 Python 路径至少 5 倍加速或至少 2 倍峰值内存改善；Hamiltonian matrix/MVP construction 至少在两个真实 workload 上获得 3 倍以上 setup speedup，并且不改变数值结果。

### 15.2 Propagation

使用现有 12-qubit TFIM dense PPE、2D Heisenberg 和 100-qubit Pauli propagation 示例的线路与 Hamiltonian 作为起点，但统一改用 `max_weight` 配置。增加 Clifford-heavy、Pauli-rotation-heavy、duplicate-heavy 和不同 weight-growth profile 的 workload。

Rust-native forward 的建议门槛是：相对非 JIT Python/NumPy 路径至少 3 倍 steady speedup；相对 JAX CPU 路径在包含 compile 的 cold end-to-end 时间上至少 3 倍加速，或在动态 term structure workload 上显示稳定优势；peak memory 受 hard memory limit 控制。Native `value_and_grad` 应显著优于逐参数 parameter-shift，并将 value 与 gradient 的误差分别对照同一 weight-truncated reference。

### 15.3 Symmetry

使用 TFIM、XXZ、Heisenberg、Hubbard 映射和随机 Pauli Hamiltonian，报告 symmetry analysis time、减少的 qubit/sector dimension、matrix/MVP memory 和端到端 eigensolver/VQE 收益。对称性模块的主要指标不是单独分析速度，而是可靠降维后的端到端时间与内存。

## 16. 分阶段路线

### 阶段零：语义冻结和 baseline

定义 Pauli phase、qubit ordering、dtype、duplicate aggregation、Pauli weight projection、GateTape 和 error behavior。明确 coefficient cutoff 不进入第一版 gradient mode。整理当前 Python/JAX correctness 与性能 baseline。此阶段不发布 wheel。

### 阶段一：Pauli core 与 Hamiltonian compiler

实现 PauliWord、PauliOperator、批量 canonicalization、commutation/grouping、dense/COO/CSR、native MVP、backend MVP plan 和 PyO3 facade。完成 cross-platform wheel smoke test。

### 阶段二：Symmetry 与 sector plan

实现 Z2 generator analysis、sector validation、基础 tapering，以及显式 U(1) fixed-particle basis/restricted Hamiltonian plan。与 `U1Circuit` 对齐。

### 阶段三：Rust-native propagation

实现 Clifford exact、`exact_dynamic` 和 `weight_truncated` forward engine，支持首批 rotation gates，输出各 Pauli weight 的 term count、被 weight projection 删除的 coefficient norm、peak terms 和 timing metadata。该阶段不实现 fixed-buffer sparse engine。

### 阶段四：Native gradient 与 TensorCircuit integration

为受支持 rotation gates 实现 analytic derivative、reverse mode 和 checkpointing；稳定后增加 `tc.pauli` adapter、文档、examples 和迁移指南。是否开发 JAX custom call 另行立项。

### 阶段五：Qudit generalized Pauli/Weyl 与 Hamiltonian compiler

在 qubit API 和语义稳定后，增加统一局域维数 `d>2` 的 generalized Pauli/Weyl 表示。每个 qudit site 用指数对 `(a,b)` 表示 `X^a Z^b`，其中 `a,b` 按 `d` 取模，`X|j⟩=|j+1 mod d⟩`，`Z|j⟩=ω^j|j⟩`，`ω=exp(2πi/d)`。第一版提供 `QuditPauliWord`、`QuditPauliOperator`、乘法相位、adjoint、commutation、canonicalization 和 deterministic aggregation，并让 Hamiltonian compiler 支持 `d**n` basis 上的 bounded dense、COO/CSR、native MVP 与 backend plan。

该阶段优先支持所有 sites 使用相同 local dimension 的模型，并保持 qudit 0 的 computational-basis ordering 与 TensorCircuit adapter 明确一致。Mixed local dimensions、任意 composite-d stabilizer/symmetry 算法和 qudit propagation 不自动包含在首个 qudit slice 中。Qudit Hamiltonian generation 在这里指从 generalized Pauli/Weyl sums 编译矩阵或 MVP，而不是内置生成特定物理模型；常见 clock/shift、Potts 或 Bose-Hubbard fixtures 可以作为 examples 和 benchmarks。

## 17. 主要风险与控制措施

### 17.1 Term explosion

非 Clifford gate 会让 Pauli term 数指数增长。Rust 只能降低常数，不能改变复杂度。控制措施是 `max_weight`、hard memory limit、可选的 forward-only coefficient tolerance、symmetry sector 和详细的 discarded-weight diagnostics；不得用性能语言掩盖近似误差。

### 17.2 Native gradient 实现复杂度

Rust 不具备现成的 TensorCircuit backend AD graph，必须维护 local VJP、parameter-slot accumulation、checkpoint 和 term aggregation 的反向映射。控制措施是只支持有解析 derivative rule 的 Pauli rotations；把 parameter-independent 的 Pauli weight projection 纳入明确递推；对通用 PTM 要求同时提供 derivative PTM，否则拒绝 gradient mode。

### 17.3 与 TensorCircuit 语义漂移

gate angle、qubit ordering、global phase、dtype 和 QIR gate naming 变化都可能破坏结果。控制措施是稳定 adapter 层、版本 compatibility matrix 和与 TensorCircuit tests 的 differential suite；Rust core 不直接依赖 Python callable 名称。

### 17.4 Native extension 发布成本

PyO3 wheel 增加平台矩阵和维护门槛。控制措施是独立可选 distribution、`abi3` 可行性验证、最小 crate 数量和纯 Python core 安装不受影响。

### 17.5 Rust 与 backend 重复实现

同一算法可能出现 native 与 backend 两套逻辑。控制措施是共享 canonical plan、固定语义测试向量和明确职责：Rust-native 负责动态 weight-truncated CPU propagation；backend-plan 负责 Hamiltonian、measurement 和固定 basis 的 AD/JIT/accelerator execution。两者不要求逐行同构，但对共同支持的模式必须给出一致结果。

## 18. Go/No-Go 决策

阶段一完成后，如果 Pauli canonicalization、Hamiltonian construction 和 MVP plan 在真实 workload 上没有达到可观察的 setup 或内存收益，则停止 native 扩展路线，只保留算法和测试改进。阶段三完成后，如果 Rust-native propagation 仅比 Python 小幅加速，并且 JAX cold compile 不是实际用户瓶颈，则不继续投入 native gradient。

只有在 forward engine 显示明确端到端优势、截断误差可诊断、用户工作流能够接受 GateTape/explicit native API 时，才进入 native value-and-gradient 和深度 TensorCircuit 集成。

## 19. 关键开放问题

- 参数化 GateTape 由显式 builder、`SymbolCircuit` adapter 还是新的 QIR parameter reference 生成。
- Native gradient 优先实现 reverse mode、forward mode，还是先提供 parameter-shift baseline。
- Z2 tapering 是否第一版实现完整 Clifford transform，还是先只提供 symmetry generators 与 sector projector。
- Backend MVP plan 的 portable integer dtype 如何兼容 JAX 默认关闭 int64、TensorFlow 和 PyTorch。
- Weight projection 的误差报告采用 discarded L1/L2 coefficient norm、observable-specific bound，还是同时提供多项指标。
- Qudit canonical basis 采用直接 `X^a Z^b` 还是带中心相位的 Weyl-normalized `τ^(ab)X^a Z^b`；该选择会影响 multiplication phase、adjoint、Hermiticity 和 `d=2` 与现有 `PauliWord` 的兼容方式，必须在实现前冻结。
- 首个 qudit slice 是支持任意整数 `d>=2`，还是先限定 prime/prime-power dimension；GF(d) symmetry 方法不能在 composite `d` 上未经说明地复用。
- Qudit 首版是否只支持 uniform local dimension，mixed-radix systems 后续再做；公开 serialization 必须无歧义记录每个 site 的 dimension 和 exponent ordering。

## 20. 推荐结论

TenCirPauli 值得进入原型阶段。它与 TensorCircuit 的 Hamiltonian、VQE、time evolution、U(1) 和 Heisenberg-picture 能力直接相连，同时把 Rust 放在其最擅长且不会破坏 backend abstraction 的位置。

推荐在独立仓库中从阶段零和阶段一开始，不承诺完整 native autodiff。第一交付物应该是一个可 benchmark、可验证、可选安装的 Pauli algebra、measurement grouping 与 Hamiltonian engine；第二交付物是 forward-only native weight-truncated propagation。每个阶段以真实 TensorCircuit workload 的端到端结果决定是否继续。
