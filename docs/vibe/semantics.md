# TenCirPauli 核心语义

状态：已冻结。S1–S4 已由项目 owner 确认，可作为 Phase 1 自主实现的语义合同。

本文档定义跨 Rust core、PyO3、Python API、Hamiltonian 输出和 TensorCircuit adapter 必须一致的语义。实现细节可以优化，公开语义不得由实现 Agent 临时猜测或静默改变。

## 1. Pauli code 与 canonical word

外部整数编码与 TensorCircuit 一致：`0=I`、`1=X`、`2=Y`、`3=Z`。外部结构数组 `structures[t, q]` 的位置 `q` 表示 qubit `q`；字符串形式同样令第 `q` 个字符表示 qubit `q`，因此 `"XI"` 在两比特系统中表示 `X0`，不是 `X1`。

内部 binary symplectic mapping 为 `I=(x=0,z=0)`、`X=(1,0)`、`Y=(1,1)`、`Z=(0,1)`。qubit `q` 存在 packed word 的 bit `q % 64`，word index 为 `q // 64`。`nqubits` 不是 64 的倍数时，最后一个 word 中未使用的高位在构造时清零；输入 word 数量不匹配时直接报错。`nqubits=0` 合法，表示零比特 identity 空间。

S1 已确认：采用 phase-free `PauliWord`。它只表示 Hermitian basis element `I/X/Y/Z` 的 tensor product，hash、equality 和 canonical key 不含 phase。独立的 word 不要求附带 coefficient；weight、support、commutation、serialization 和 grouping 都只需要 word。只有乘法会产生额外 phase，因此 word multiplication 返回 canonical word 与一个精确的四值 `PauliPhase={+1,+i,-1,-i}`，不使用浮点复数计算 phase。例如 `X*Y` 返回 `(Z,+i)`，`Y*X` 返回 `(Z,-i)`。当结果进入 `PauliTerm` 或 `PauliOperator` 时，再把该离散 phase 乘进 coefficient；两个 terms 相乘时得到 `new_coefficient=c1*c2*phase` 与 canonical result word。这样同一个物理 term 不会因 phase 存储位置不同产生多个 hash key，也不强迫所有单独的 `PauliWord` 携带无用权重。

## 2. Qubit ordering 与矩阵 basis

TensorCircuit 的 qubit 0 是 statevector basis index 的最高有效位；例如两比特 `X0|00⟩=|10⟩`。TenCirPauli 内部 packed key 则令 qubit 0 为最低有效 packed bit，以便快速 bit operation。两种 ordering 都是固定语义，不能混为一谈。

Hamiltonian dense/COO/CSR/MVP 边界将 qubit `q` 映射到 computational-basis integer 的 bit `nqubits-1-q`。字符串、结构数组、measurement basis 和 TensorCircuit adapter 保持 qubit index 不变；只有 matrix basis mask 的构造执行该显式换位。每个矩阵 target 都必须以首尾 qubit 上的 `X/Y/Z` 非对称测试防止 bit reversal。

## 3. Coefficient、dtype 与 phase

S2 已确认：Phase 1 的 native numeric operator 只使用 IEEE complex128 语义，即实部和虚部均为 `f64`；Python 接受可安全转换为 NumPy `complex128` 的标量或一维数组。Rust `num-complex` 中对应类型名是 `Complex64`（两个 `f64`），不要将它与 NumPy `complex64`（两个 `float32`）混淆。结构 plan 与 coefficient buffer 分离，backend-plan 不把 coefficient 固定为 Rust dtype，允许 TensorCircuit backend 在执行时提供自己的 float/complex tensor。

Phase 1 不做隐式近似截断。重复 key 的 coefficient 先按确定性顺序求和；静态 native operator 可以删除结果严格等于复数零的 term，但不得使用隐藏 epsilon。任何 tolerance-based drop、Hermiticity tolerance 或 coefficient cutoff 都必须由公开参数显式给出，并且不进入结构 hash。`NaN` 和无穷 coefficient 默认拒绝，除非未来以独立模式明确支持。

Python 输入若为 complex64 可以无损提升存储精度但不能静默降精度；输出 native dense/COO/CSR/MVP 的默认数值 dtype 为 complex128。后续若增加 complex64 fast path，必须作为显式 dtype 并与 complex128 reference 差分测试。

## 4. Canonicalization 与确定性

Canonical public order 建议按每个 term 的外部 Pauli code tuple `(p0, p1, ..., p[n-1])` 做 lexicographic ordering，其中 `I<X<Y<Z`。内部容器可以使用更快的 hash/key order，但公开 arrays、序列化、group membership 和测试结果必须转换为该稳定顺序。

Batch canonicalization 输入相同 `nqubits` 的 structures 与 coefficients，输出 canonical keys、聚合后的 coefficients、`input_to_canonical` 和每个输入 term 的 phase multiplier。Backend-plan 使用 mapping 在 backend tensor 上执行 reduction，不得根据当次 parameter value 删除结构。静态 native `PauliOperator` 可以在聚合完成后删除 exact-zero term。

Hash 只依赖 `nqubits` 与 canonical `(x,z)` masks。包含静态 coefficients 的 content hash 必须与 structure hash 分离。任何依赖 hash-map iteration order 的输出都属于 bug。

## 5. Algebra 与验证

`PauliWord` 的 equality、weight、support、commutation、symplectic inner product、multiplication 和 adjoint 必须同时通过代数 property tests 与小系统 dense-matrix differential tests。`PauliOperator` 的 add、scale、multiply、commutator、anticommutator、adjoint 和 Hermiticity validation 必须以 complex128 dense reference 验证。

所有公开 API 对不兼容 `nqubits`、非法 code、错误 shape、越界 qubit、非有限 coefficient、整数溢出和无法满足的内存请求直接返回明确错误。Rust core 使用 typed error；PyO3 转换为稳定的 Python `ValueError`、`OverflowError` 或 `MemoryError`，不能 panic 或静默修正语义错误。

## 6. Hamiltonian target 语义

Dense、COO、CSR 和 native MVP 表示同一个按 TensorCircuit basis ordering 构造的线性算符。COO 使用 `(row, column, value)`，按 row-major `(row,column)` 排序并聚合重复 matrix entry；CSR 从已聚合的确定性 COO 构造。index 对外使用可安全表示矩阵维度的 64-bit integer，在转换到平台 `usize` 或 SciPy index dtype 前检查溢出。

任何物化 target 在分配前估算 dimension、nonzero upper bound 和 bytes，并应用明确的 memory limit。超限时返回包含估计值与建议 MVP target 的错误。Native MVP 不物化 `2**n × 2**n` matrix，但仍检查 state length、dtype 和地址空间溢出。

## 7. Measurement grouping 语义

Qubit-wise commuting（QWC）要求任意两个 term 在每个 qubit 上相同或至少一个为 identity；QWC group 可以由单比特 basis rotation 共同测量。General commuting 只要求 symplectic inner product 为零，通常需要 entangling Clifford diagonalization。

S3 已确认：Phase 1 完整交付 QWC grouping、逐 qubit measurement basis 和 bitstring eigenvalue reconstruction。General-commuting grouping 先实现一个小型、确定性的原型 partition；它必须明确标注为 algebraic grouping，并设置 `measurement_ready=false`。在实现 Clifford diagonalization 与 reconstruction plan 前，不得声称该 prototype 是可直接共同测量的 plan，也不得复用 QWC plan 类型。Prototype 以正确性、确定性和可扩展数据结构为目标，不要求第一版在 coloring quality 或大规模性能上达到最终水平。

确定性 grouping 的 tie-break 使用 canonical term order。Largest-first greedy 是 REQUIRED baseline；DSATUR 在同一语义稳定后实现。返回结果包含 canonical group membership、算法名、QWC/general mode、basis/reconstruction metadata，以及是否 measurement-ready 的显式标记。

## 8. Backend plan 与 TensorCircuit 边界

S4 已确认：Phase 1 要求 Rust 生成版本化、纯 arrays 的 backend MVP plan，并提供 NumPy reference executor；TensorCircuit adapter 只做 plan 到 `tc.backend` tensor operation 的薄转换。Phase 1 验收至少覆盖 TensorCircuit NumPy 与 JAX backend 的 differential smoke test，不要求此阶段同时优化 TensorFlow、PyTorch、CuPy 的性能。

Rust core 不导入或调用 Python/TensorCircuit。Plan 不保存 Python callable、device object 或 backend tensor；序列化包含 schema version、nqubits、ordering、integer width 和 required operations。Adapter 缺少 TensorCircuit 时应给出安装错误，不做 native 或 NumPy 静默 fallback。

## 9. Owner 待确认决策

| ID | 推荐方案 | 主要替代方案 | 影响 |
|---|---|---|---|
| S1（已确认） | `PauliWord` phase-free，乘法单独返回 phase，operator coefficient 吸收 phase | `PauliWord` 内含 `phase: u8` | 保持唯一 canonical key 和精确离散 phase |
| S2（已确认） | Phase 1 native coefficient 统一 complex128；结构 plan dtype-independent | 同时实现 complex64/complex128 两套 native kernel | 第一阶段避免 dtype dispatch、双套 FFI 与测试矩阵 |
| S3（已确认） | Phase 1 完整 QWC measurement；general commuting 实现 `measurement_ready=false` 的小型 deterministic prototype | Phase 1 同时实现 general commuting Clifford diagonalization | Prototype 不冒充可直接测量 plan，后续可独立演进 |
| S4（已确认） | Phase 1 交付 plan schema、NumPy executor 和 TC NumPy/JAX smoke；其他 backend 后续扩展 | Phase 1 要求全部 TensorCircuit backend | 控制 adapter 测试矩阵与完成时间 |

S1–S4 均已冻结。后续若修改这些决策，必须增加 changelog/decision note、迁移说明和新旧语义的回归测试。
