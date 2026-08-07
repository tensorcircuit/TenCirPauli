# Rust 量子算符与 Pauli 生态竞品调研

调研日期：2026-08-06。本文面向开源项目维护者，比较 Rust 生态中与量子算符、Pauli/Fermion algebra、Hamiltonian 表示、mapping、symmetry、measurement 和 circuit-level propagation 相关的项目。重点项目包括 [Zixy](https://github.com/Quantinuum/zixy)、[Struqture](https://github.com/HQSquantumsimulations/struqture)、[pauli_tracker](https://github.com/taeruh/pauli_tracker)、[qoqo/roqoqo](https://github.com/HQSquantumsimulations/qoqo) 和 [qsym-rs](https://github.com/QudeLeap/qsym-rs)；其中 qoqo/roqoqo 和 qsym-rs 分别属于相邻生态参考和小型 baseline，并非与 TenCirPauli 完全同层次的直接竞品。结论基于各项目公开仓库、文档、源码结构和示例的源码级阅读；没有把未经 matched release benchmark 验证的性能差异写成定量结论。

## 总结

Rust 生态中没有一个项目同时覆盖 TenCirPauli 当前关注的完整组合：Pauli algebra、结构化 fermion/Majorana/boson/qudit/hybrid operator、measurement grouping、Z2 tapering、U1/general additive charge sectors、可复用 mapping plan、native MVP/COO/CSR/backend plan，以及 TensorCircuit-facing Python API。

Zixy 是最直接的底层性能和 Pauli/Fermion algebra 竞品；Struqture 是成熟的通用领域对象、系数类型、序列化和 Python binding 参考；pauli_tracker 是 Clifford Pauli propagation、Pauli frame 和紧凑 bit-vector 后端的参考；qoqo/roqoqo 更适合参考 circuit IR、序列化和 backend 生态边界；qsym-rs 则是一个功能较小的 Pauli-string 教学型 baseline。

因此，TenCirPauli 的差异化不应只表述为“另一个 Rust Pauli 库”，而应表述为：面向 TensorCircuit 的、native-handle 驱动的算符数据平面，把结构化 Hamiltonian、mapping、symmetry/sector reduction、measurement grouping、matrix-free execution preparation 和传播路径接到同一套可验证的接口上。

## 竞品总览

下表是能力边界的快速索引。“部分”表示源码或文档中存在相关 primitive，但尚未形成与该列标题相同完整度的公开工作流；“未见”表示本次源码范围内没有找到对应能力，不等于项目在所有分支或私有代码中绝对没有该能力。

| 项目 | 核心定位 | Rust core | Python/其他绑定 | Pauli/Fermion algebra | Fermion mapping | Hamiltonian/matrix | Grouping/symmetry | Propagation/circuit | 与 TenCirPauli 的关系 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [TenCirPauli](../../) | TensorCircuit-facing native 算符与执行准备层 | 有 | Python + PyO3 | Pauli、structured fermion/Majorana/boson/qudit/hybrid | JW/parity/BK 与可复用 mapping 路径 | dense、COO、CSR、MVP、backend plans | QWC/general commuting、Z2、U1/additive charge sectors | dynamic Pauli propagation、weight projection、native gradients | 本项目基准 |
| [Zixy](https://github.com/Quantinuum/zixy) | 高性能 Pauli/Fermion algebra、低层 mapping 和 sparse state/operator algebra | 有 | Python + PyO3 | Pauli、term containers、fermion ladder products | Rust core 有 JW/parity/BK；Python 当前主要暴露 JW | sparse matrix、state action、subspace projection | compatibility/commutation、centralizer-like filtering、Hamming-weight checks | Clifford conjugation；不是通用 propagation engine | 最直接的底层重合对象 |
| [Struqture](https://github.com/HQSquantumsimulations/struqture) | 面向量子模拟的成熟领域算符对象和系数/序列化生态 | 有 | `struqture-py` | Pauli、boson、fermion、mixed/open-system operator | 不是以 fermion-to-qubit mapping 为核心 | sparse operator/matrix 转换、symbolic coefficient、序列化 | 有算符结构和相关分析，但不是 TenCirPauli 式 grouping/tapering 主线 | 与 qoqo 生态配合；非专门传播引擎 | 通用算符对象和 API 设计参考 |
| [pauli_tracker](https://github.com/taeruh/pauli_tracker) | Clifford circuit 中的 Pauli tracking、Pauli frame 和 measurement tracking | 有 | Python + C interface | 单 Pauli/Pauli frame 语义，不是通用 Hamiltonian linear combination | 不以 fermion mapping 为目标 | 非 Hamiltonian/matrix 主线 | measurement tracking；不提供通用 grouping/tapering | Clifford propagation 是核心，含 bit-vector/SIMD backend | propagation 数据结构参考 |
| [qoqo/roqoqo](https://github.com/HQSquantumsimulations/qoqo) | circuit/program IR、序列化、measurement runtime 和 backend interface | 有 | Python 等绑定 | 不是 Pauli algebra 主库 | 未见作为核心 mapping API | circuit measurement/runtime；不是 operator matrix 库 | 依赖生态组件，不是 grouping 主库 | circuit representation 和 backend execution | IR、序列化、生态边界参考 |
| [qsym-rs](https://github.com/QudeLeap/qsym-rs) | 小型 Pauli string 教学/实验项目 | 有 | CLI/库 | single Pauli、Pauli string、phase-aware multiplication、commutes、weight | 未见 | 未见大规模 Hamiltonian/matrix 工作流 | 未见 | 未见通用 circuit propagation | 最小正确性 baseline |

### 非 Rust 生态参照

OpenFermion、Qiskit Nature、PennyLane 和 TensorCircuit 等项目仍然是接口和用户工作流的重要参照，但它们不是本表的 Rust 直接竞品。尤其 OpenFermion 更像 fermionic/qubit operator interchange 和 chemistry workflow 参照，PySCF 更像分子积分与 SCF 数据源；TenCirPauli 的 Phase 10 合同计划提供可选 PySCF import，但不能把该计划描述成已经完成的当前功能。

## 项目逐项分析

### Struqture

Struqture（HQS Quantum Simulations）是本次比较中最成熟的通用算符对象参考之一。它由 Rust crate `struqture` 和 Python binding `struqture-py` 组成，核心对象包括 `PauliProduct`、`PauliOperator`、`PauliHamiltonian`，并延伸到 boson、fermion、mixed system 和 open-system 表示。它的重点不是某一种量子化学输入格式，而是把不同物理系统的 operator、系数、加法/乘法、稀疏矩阵转换和序列化组织成可复用的领域对象。

Struqture 对 TenCirPauli 最有价值的参考是“对象语义先于后端实现”：不同 operator family 有明确的 term 类型和维度语义，数值系数与 symbolic coefficient 可以共存于统一的高层表达，Rust core 和 Python API 之间也有相对清晰的边界。对于 TenCirPauli，这提示 structured operator 的 public contract、系数 promotion、序列化 schema 和跨 family 转换应当是显式设计点，而不是把所有对象都降级成一组 Pauli strings。

它与 TenCirPauli 的主要差异在于目标中心不同。Struqture 更像通用的量子模拟 operator model 和 HQS 生态基础库；本次阅读没有把它视为专门的 measurement grouping、Z2 tapering、U1 sector compiler 或 dynamic Pauli propagation engine。TenCirPauli 则更强调从算符构造一路到 grouping、symmetry/sector reduction、MVP/CSR preparation 和 TensorCircuit integration 的执行闭环。

### pauli_tracker

`pauli_tracker` 是 Rust library、Python package 和 C interface 组成的 Clifford-oriented 工具。它维护 Pauli frame、追踪 Clifford circuit 对 Pauli 的作用，并将 measurement tracking、bit-vector 表示和 SIMD-friendly backend 放在核心路径上。它解决的是“电路执行过程中如何低成本更新和解释 Pauli 信息”，不是“如何表示任意 Hamiltonian 的线性组合并做 fermion-to-qubit mapping”。

它的设计对 TenCirPauli 的直接启发是传播数据结构和批量更新策略：如果 Pauli propagation 的主路径是 Clifford 或有限 gate alphabet，可以把 symplectic bits、sign/frame metadata 和 measurement record 分离，并让更新在连续 bit-vector 上进行。TenCirPauli 仍需保留自己的 dynamic operator、weight projection、coefficient accumulation 和 native gradient 语义，因为这些超出了 pauli_tracker 的定位。

### qoqo/roqoqo

qoqo/roqoqo 是 HQS 生态中的 Rust-native circuit/program IR 与 Python interface，关注 circuit representation、序列化、measurement runtime 和 backend interfaces。它和 Struqture 在生态上有关联，但不是直接的 Pauli operator 或 fermionic algebra 竞品。把它列入调研的原因是：用户最终需要把 operator-level preparation 与 circuit-level execution 接起来，而 qoqo 展示了如何将 circuit、measurement、backend capability 和序列化拆成相互独立的层。

对 TenCirPauli 而言，qoqo/roqoqo 更适合作为接口架构参考：公共 circuit IR 可以保持 backend-neutral，运行时能力通过明确的 plan/capability 描述，序列化和执行不必绑定到某一个 Python framework。它不能替代 TenCirPauli 的 operator core，也不直接提供本项目需要的 grouping、charge-sector restriction 或 Pauli propagation 算法。

### qsym-rs

`qsym-rs` 是小型的 Pauli 代数项目，覆盖 single Pauli、Pauli string、phase-aware multiplication、commutation 和 Pauli weight，并提供 CLI/库形态。它的价值不是功能广度或性能规模，而是作为最小正确性 baseline：Pauli multiplication 的相位、对易判断和 weight 语义都可以用很小的实现独立复核。

它不具备本次调研关注的大规模 operator/Hamiltonian 容器、fermion mapping、matrix materialization、grouping、symmetry reduction 或 circuit propagation 能力。因此它更适合用于测试向量和 API 复杂度对照，不适合作为 TenCirPauli 的架构模板。

### 横向判断

| 维度 | Zixy | Struqture | pauli_tracker | qoqo/roqoqo | qsym-rs | TenCirPauli 的机会 |
| --- | --- | --- | --- | --- | --- | --- |
| 算符对象 | Pauli/Fermion term containers 很强 | 多物理体系 operator model 最完整 | 以 Pauli frame 为中心 | circuit/program object | 最小 Pauli string | 统一 structured operator 与执行计划，但保持 native lazy data plane |
| Mapping | Rust core JW/parity/BK | 非核心定位 | 非核心定位 | 非核心定位 | 未见 | mapping 与 chemistry ingestion、symmetry reduction 连成可复用 plan |
| 数值执行 | sparse matrix、state/subspace algebra | operator/matrix conversion | Clifford tracking 高效 | circuit/backend runtime | 小型代数 | MVP、COO/CSR、restricted backend 与 TensorCircuit 组合 |
| Grouping/symmetry | primitive 较少 | 通用 operator 生态 | measurement tracking | measurement/runtime | 未见 | QWC/general commuting、Z2 tapering、U1/additive charge sectors |
| 公开 Python 形态 | container/view 较丰富 | 领域对象和 binding 成熟 | tracker/frame API | circuit/backend API | 简单 | Python 薄封装、opaque native handles、flat read-back ABI |

本次源码阅读没有发现某个项目在表现力或性能上存在一个足以让 TenCirPauli 整体失去竞争力的“本质性缺口”。更准确的说法是，各项目在不同轴上有局部优势：Zixy 的 Pauli/Fermion container 和 mapping workspace 值得做 matched benchmark，Struqture 的 operator-family 设计和序列化更成熟，pauli_tracker 的 Clifford tracking 数据布局更专门化，qoqo/roqoqo 的 circuit/runtime 分层更完整。是否存在大幅性能差距，必须由相同语义、相同输入转换和相同 Rust/Python boundary 成本的 release-mode benchmark 验证，不能仅凭源码体量判断。

## Zixy

Zixy 是本次调研中与 TenCirPauli 在底层 Pauli/Fermion algebra 和 mapping 层最直接重合的项目。下面保留它的源码级分析，以便把它与上面的横向项目区分开来。

### Formal fermion mapping

#### Rust core 中的 mapping

Zixy 的 Rust core 在 `zixy/src/fermion/mappings/` 中定义了统一的 `UpdateParityRho` formalism。源码注释明确对应 Seeley、Richard 和 Love 的 update/parity/rho 集合形式，并用这三个集合构造 fermionic creation/annihilation operators 的 Pauli 展开。

当前 core 包含以下 mapper：

- Jordan–Wigner mapping，文件为 `jw.rs`；
- parity mapping，文件为 `parity.rs`；
- Bravyi–Kitaev mapping，文件为 `bk.rs`，使用 Fenwick-tree 风格的集合计算；
- `ParapartiularMapper`，当前更像一个内部占位或实验性 mapper，不应视为成熟公开能力。

核心的 `Operators` workspace 接受一个有序的 ladder-operator product，例如 `[(0, true), (1, false)]` 表示 (a_0^\dagger a_1)，然后把它展开成若干 Pauli strings 及其离散相位/系数。它支持 real 和 complex coefficient accumulation，也会在内部合并相同 Pauli word。

#### Python public API 的实际暴露范围

当前 `zixy-py/src/lib.rs` 只向 Python module 注册了 `JordanWignerMapper`。因此，虽然 Rust core 已经有 parity 和 Bravyi–Kitaev mapper，当前 Python 公开层主要只有 Jordan–Wigner。

Python 侧的 `JordanWignerMapper` 接口包括：

- `encode(fermion_ops)`：编码任意给定的 creation/annihilation product；
- `encode_ca(c, a)`：编码 (a_c^\dagger a_a)；
- `encode_n(i)`：编码 number operator (n_i=a_i^\dagger a_i)；
- `encode_caca(...)`：编码两个 creation-annihilation product；
- `encode_nn(i, j)`：编码 (n_i n_j)；
- `encode_ccaa(...)`：编码两个 creation followed by two annihilation operators；
- `mode_ordering`：允许显式指定 fermionic mode 到 qubit 的顺序。

`RealTermSum.from_fermionic(...)` 可以把一批 `[(ladder_operator_product, coefficient)]` 直接积累成 Pauli `RealTermSum`。Python 还提供 `UnorderedFermionOpReal`，其本质是 `[(list[(mode, is_creation)], coefficient)]` 的薄封装；它不是一个带有丰富语义、规范化、积分接口或化学对象元数据的完整 `FermionOperator` 类型。

这里的“formal”主要体现在 ladder-operator product 的 algebraic encoding 和 mapping formula，而不是完整的 quantum-chemistry operator interchange format。

### 是否支持从 PySCF 或 OpenFermion 读取算符

#### PySCF

当前没有直接 PySCF 支持。Zixy 的 Python 项目依赖只有 NumPy、SciPy、SymPy、Pandas 和 typing helpers；源码和文档中没有 `pyscf` import、SCF adapter、molecular geometry parser、AO/MO integral transformation、nuclear-repulsion ingestion 或 active-space interface。

Zixy 的 chemistry notebook 使用如下两种输入方式：

1. 直接把已经完成 Jordan–Wigner 变换的 H2 STO-3G Pauli Hamiltonian 写成 sparse string；
2. 手动调用 `JordanWignerMapper.encode_*()` 构造若干费米子项，再将它们加入 `RealTermSum`。

因此，用户可以在外部用 PySCF 生成积分，再自行转换成 Zixy 所需的 ladder-operator list 或 Pauli term list，但这属于调用方的桥接代码，不是 Zixy 自己提供的 PySCF 读取能力。

#### OpenFermion

当前也没有 OpenFermion adapter 或 `FermionOperator`/`QubitOperator` 兼容入口。Zixy 可以表达一部分与 OpenFermion 类似的对象，但没有看到从 OpenFermion Python object 直接导入、导出或复用 OpenFermion mapping API 的实现。

#### 文本和内部序列化

Zixy 支持自己的 Pauli sparse-string 表达，例如带系数的 `(coefficient, X0 Y1 Z3)`，也支持 Python iterable、dict、DataFrame 和部分二进制内部对象保存/加载。这些属于 Zixy 自己的 operator representation，不等同于 FCIDUMP、PySCF object、OpenFermion JSON 或通用分子 Hamiltonian interchange format。

### Chemistry notebook 实际展示的能力

Zixy 的 chemistry 示例已经能完成一个小型电子 Hamiltonian 的后续数值工作流：

- 读取或构造 Pauli 线性组合；
- Jordan–Wigner 编码一体和二体 fermion products；
- 检查 Hamiltonian 是否守恒总 Hamming weight 或奇数位 Hamming weight；
- 转换为 little-endian 或 big-endian SciPy sparse matrix；
- 对正交和非正交 subspace 做投影，并返回 overlap matrix；
- 构造 computational-basis state 和稀疏 state linear combination；
- 计算 operator-on-state、inner product、matrix element 和 expectation value；
- 做 operator multiplication、commutator、Hermiticity-related real/imaginary separation；
- 对 Pauli terms 做 lexicographic 或 numerical sorting；
- 对 Clifford gate list 做 Pauli conjugation；
- 用 number operator commutation 检查 Hamming-weight conservation；
- 支持 real、complex 和 Python/SymPy symbolic coefficient container。

这些能力足以支持“外部量子化学程序已经给出 Hamiltonian 后，在 Pauli/operator 层继续做矩阵、子空间和代数计算”的工作流，但不包含分子数据准备本身。

### Centralizer、symmetry 和 grouping 的边界

Zixy 提供 `compatibility_matrix()`、`commutes_with()` 和 `centralizer_and_remainder()`。其中 `centralizer_and_remainder()` 会把输入 Pauli array 中“与该列表其他元素全部对易”的项分出来；它不是在完整 Pauli 空间中求出 centralizer 的一组基，也不是 Z2 tapering。

Zixy 还提供 Hamming-weight conservation 检查，但当前没有看到 TenCirPauli 式的完整 Z2 symmetry discovery、independent isotropic generator selection、Clifford tapering plan 或 selected-sector transform。也没有看到 QWC measurement basis/reconstruction 或一般 commuting measurement plan。

## TenCirPauli 与 Zixy 的定位比较

| 能力层次 | Zixy | TenCirPauli |
| --- | --- | --- |
| Pauli algebra | 高性能 Rust core，Python term containers | Rust core，Python-first lazy operator API |
| Fermion formal mapping | Rust core 有 JW/parity/BK；Python 当前主要暴露 JW | 已有 JW/parity/BK mapping 与 formal structured fermion operator 路径 |
| 外部化学输入 | 未见 PySCF/OpenFermion adapter | PySCF 适配器有单独的 chemistry/scientific-interop 合同，OpenFermion 不作为运行时依赖 |
| Hamiltonian 数值工作流 | sparse matrix、state、subspace projection、expectation | dense/COO/CSR/MVP、backend plan、restricted sector 与 TensorCircuit integration |
| Measurement grouping | 有 commutation/compatibility primitive，未见完整 grouping plan | QWC/general commuting grouping，含确定性分组和 QWC reconstruction |
| Symmetry | Hamming-weight conservation 与 centralizer-like filtering | Z2 null-space symmetry、tapering、U(1)/charge sector restriction |
| Circuit propagation | Clifford conjugation | dynamic Pauli propagation、weight projection、native gradients 和 TensorCircuit gate adapters |
| Framework边界 | 独立 Python/Rust algebra package | TensorCircuit-facing Rust-native companion |

最重要的判断是：Zixy 与 TenCirPauli 在“Pauli/Fermion algebra + low-level mapping”层面存在直接重合，但 Zixy 当前没有覆盖从 PySCF 分子对象到 canonical fermion Hamiltonian、再到 Pauli mapping、symmetry reduction、TensorCircuit execution 的完整端到端链路。

## 对 TenCirPauli 的可借鉴点

- mapping 内部可以采用统一的 update/parity/rho formalism，并把 JW、parity、BK 共享到同一个 batched ladder-product expansion engine；
- Zixy 的 component/coefficient 分离和 `Sign`/`ComplexSign` 专用存储值得作为 Pauli phase 和 SoA 布局的 benchmark 参考；
- `RealTermSum`、`ComplexTermSum`、`SymbolicTermSum` 的类型分层说明“数值快速路径”和“符号表达路径”可以分离，而不必让符号能力污染 Rust 数值核心；
- chemistry notebook 的正交/非正交 subspace projection、overlap matrix 和 state algebra 是 TenCirPauli 的 PySCF/科学计算互操作层可以参考的下游接口；
- Zixy 没有直接 PySCF adapter，反而支持 TenCirPauli 以“标准化 PySCF ingestion + native fermion operator + reusable mapping plan”作为明确差异化能力。

## 源码级可借鉴点

### 1. 明确区分 Terms、TermSet 和 TermSum

Zixy 的 `Terms` 是允许重复、保留顺序的 array-like collection；`TermSet` 以 component 为 key、保证唯一性；`TermSum` 在 `TermSet` 之上提供线性组合语义。这个区分把“输入批次”“canonical unique operator”和“可做加法的 operator”拆开了，避免所有路径都被迫使用同一个容器。

TenCirPauli 当前的 `PauliOperator` 更接近 canonical `TermSum`。后续如果扩展 `OperatorBuilder` 或大批量 canonicalization，可以借鉴这个语义分层：输入 batch 保留原始 index/duplicate，canonical operator 使用唯一 key，显式的 linear-combination kernel 直接消费两个 native views。这个思想比把三个公开 Python 类原样搬过来更适合 TenCirPauli 的 lazy handle 架构。

源码参考：[Zixy Python term containers](https://github.com/Quantinuum/zixy/blob/main/zixy-py/zixy/container/terms.py)、[Zixy native term views](https://github.com/Quantinuum/zixy/blob/main/zixy/src/container/word_iters/term_set.rs)。

### 2. Component 与 coefficient 的 SoA 分离

Zixy 用 `TermData` 把 component storage 和 coefficient storage 分开，并让二者具有相同长度；Rust 侧的线性组合 view 只借用两块连续数组。这个设计适合批量乘法、排序、相位更新和 coefficient-only 操作，也使 `ComplexSign` 等特殊系数可以使用独立的 packed vector。

TenCirPauli 已经有 packed Pauli representation 和独立 coefficient arrays，但可以继续借鉴两点：一是把 phase-only、real coefficient、complex coefficient 的 storage contract 明确分开；二是在 native algebra kernels 中显式传递 borrowed component/coefficient views，避免为了调用一个 operation 复制完整 operator。

源码参考：[Zixy `TermData`](https://github.com/Quantinuum/zixy/blob/main/zixy-py/zixy/container/data.py)、[Zixy coefficient traits](https://github.com/Quantinuum/zixy/blob/main/zixy/src/container/coeffs/traits.rs)、[packed `ComplexSignVec`](https://github.com/Quantinuum/zixy/blob/main/zixy/src/container/coeffs/complex_sign.rs)。

### 3. 用专用 packed storage 表示离散相位

Zixy 把 `{+1, -1}` 存成 bit vector，把 `{+1, +i, -1, -i}` 存成 two-bit vector，并把相位乘法实现成 bit-level operation。这是正确的数值方向：Pauli phase 不应该在 hot path 中反复转成 `Complex64`。

TenCirPauli 已经使用离散 `PauliPhase`，因此不需要复制 Zixy 的类型体系；值得继续验证的是大批量 canonicalization、mapping 和 propagation 中 phase 是否可以使用 packed phase arrays，而不是每条边都携带完整 enum/struct。

### 4. Mapping workspace 缓存单算符展开

Zixy 的 `Operators` 在构造时先缓存每个 fermionic mode 的 real/imaginary Pauli string，随后 `load_product()` 只做这些单算符的组合，并复用一个 `work` buffer；`contribute_real()`/`contribute_complex()` 再把结果批量写入目标 `TermSet` view。

这个模式很适合 TenCirPauli 的 mapping plan：mapping plan 负责一次性准备单 mode transform，重复调用只接收 ladder products 和 coefficients，最终一次性聚合到 native output。TenCirPauli 应继续保留 memory guard 和 checked expansion estimate，因为 Zixy 的 ladder product workspace 会按 operator length 指数扩张，源码中没有 TenCirPauli 同等层次的公共资源合同。

源码参考：[Zixy mapping workspace](https://github.com/Quantinuum/zixy/blob/main/zixy/src/fermion/mappings/operators.rs)。

### 5. 虚拟 qubit register 与物理 storage 解耦

Zixy 的 `Qubits`、`relabel()` 和 `standardize()` 把“当前对象使用哪些 qubit labels”和“内部 bit storage 如何排列”分开。`relabel()` 可以只改变虚拟 register，`standardize()` 才执行物理重排。这对嵌入、子系统抽取、mode ordering 和 endian conversion 很有用。

TenCirPauli 已经有严格的 qubit ordering 和 embedding 语义，但可以考虑在 operator-level API 中更明确地区分 `relabel`、`permute`、`embed` 和 `standardize`。尤其是映射、tensor product 和 restricted-sector plan 不应把标签变化和实际 bit permutation 混为一个操作。

源码参考：[Zixy qubit traits](https://github.com/Quantinuum/zixy/blob/main/zixy/src/qubit/traits.rs)、[Zixy qubit register](https://github.com/Quantinuum/zixy/blob/main/zixy/src/qubit/mode.rs)。

### 6. Python view/ownership 模型可以借鉴，但不应整体复制

Zixy 的 Python `Cmpnt`、`Terms` 和 `TermData` 支持 owning object、single-element view、slice view，并用 `requires_ownership` 禁止在非 owning view 上做 resize 或 append。这使 term slicing 和 coefficient mutation 很自然。

TenCirPauli 可以借鉴其中的 ownership invariant 和“view 不能改变 shape”的规则，但不建议把大量 Python view object 引入 public lazy operator API。TenCirPauli 的主路径更适合保持 opaque native handle、flat NumPy read-back 和 Rust 内部 borrowed view；否则 Python view graph 会重新引入 aliasing、生命周期和 materialization 复杂度。

源码参考：[Zixy ownership/view base classes](https://github.com/Quantinuum/zixy/blob/main/zixy-py/zixy/container/base.py)。

### 7. Component-major 与 transposed batch layout 的思路值得 benchmark

Zixy 的 Pauli core 主要使用 component-major 的 packed symplectic storage，同时保留了转置后的 mode-major `Array`。前者适合单个 word 的 multiplication、phase 和 Clifford update，后者适合按 qubit/mode 批量访问。当前 mode-major 的一部分旧实现已经被注释掉，因此这更适合作为数据布局假设来 benchmark，而不是直接复制两套长期存储。

TenCirPauli 应在 profile 显示 grouping、symplectic constraints 或 mapping plan 被按 qubit 扫描主导时，再评估 transient transposed layout；默认不保留两份长期 operator representation。

### 8. API 与验证方式有几项值得吸收

Zixy 将 `big_endian` 作为 sparse matrix API 的显式参数，在 `Qubits`/mode ordering 上也有专门的 relabel/standardize 测试；其 Python 测试覆盖 coefficient conversion、非法 view mutation、Pauli phase、term multiplication、state action 和 matrix differential；Rust 还有 Criterion tableau benchmark 和 mapping unit tests。

TenCirPauli 可以继续吸收三类实践：每个 matrix/sector/mapping target 都显式携带 ordering metadata；对 coefficient promotion 和 phase representability 写数值测试；对 mapping、grouping、restricted compile 同时保留 termwise reference 和 dense differential，而不是只测 term count 或 output shape。

## 不建议直接照搬的部分

- 不要整体复制 Zixy 的泛型 coefficient hierarchy。它同时服务 `Unity`、`Sign`、`ComplexSign`、real、complex 和 Python symbolic coefficients；TenCirPauli 的 Rust core 以数值 `f64`/`complex128` 和结构化 algebra contract 为主，直接引入这套体系会扩大编译和维护面。
- 不要直接复制 Zixy 的 Python `Term`/`TermSet` object graph。Zixy 的高层 `TermSum` 仍有 Python 循环逐项累加的路径；TenCirPauli 的 scalable algebra 应继续坚持 batch native call 和 handle-to-handle pipeline。
- 不要无条件维护 component-major 和 mode-major 两份长期 storage。先用 representative grouping、mapping 和 canonicalization profile 证明转置成本值得支付，再引入 transient layout。
- 不要照搬 Zixy 的直接 `bincode` file API 作为稳定 public interchange format。公共持久化需要 schema/version/endian/error contract；内部 benchmark asset 可以使用二进制，但不应把 `unwrap()` 式文件读写当作用户接口。
- 不要照搬 mapping workspace 的无界指数扩张。费米子 ladder product 的 Pauli branch 数会随长度增长，TenCirPauli 必须在扩展前做 checked resource estimate，并对超限输入明确失败。

总体建议是：优先吸收 Zixy 的“容器语义分层、SoA views、packed phase、mapping workspace 和 register relabeling”这些底层思想；不要复制它的整个 Python object model，也不要因为存在第二种 layout 就提前引入平行生产表示。

## 证据链接

- [Zixy repository](https://github.com/Quantinuum/zixy)
- [Zixy README](https://github.com/Quantinuum/zixy/blob/main/README.md)
- [Zixy Rust fermion mappings](https://github.com/Quantinuum/zixy/tree/main/zixy/src/fermion/mappings)
- [Zixy Python mapping wrapper](https://github.com/Quantinuum/zixy/blob/main/zixy-py/zixy/fermion/mappings.py)
- [Zixy Python binding registration](https://github.com/Quantinuum/zixy/blob/main/zixy-py/src/lib.rs)
- [Zixy chemistry example](https://github.com/Quantinuum/zixy/blob/main/zixy-py/docs/examples/chem.ipynb)
- [Struqture repository](https://github.com/HQSquantumsimulations/struqture)
- [Struqture Rust documentation](https://docs.rs/struqture/latest/struqture/)
- [pauli_tracker repository](https://github.com/taeruh/pauli_tracker)
- [pauli_tracker Rust documentation](https://docs.rs/pauli_tracker/latest/pauli_tracker/)
- [qoqo repository](https://github.com/HQSquantumsimulations/qoqo)
- [qsym-rs repository](https://github.com/QudeLeap/qsym-rs)
- [TenCirPauli fermion mapping module](../../python/tencirpauli/mapping.py)
- [TenCirPauli structured fermion operators](../../python/tencirpauli/structured.py)
- [TenCirPauli chemistry interoperability contract](phase-10-spec.md)
