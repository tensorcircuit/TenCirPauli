# Phase 5 Spec：任意宽 packed U1 restricted Hamiltonian engine

状态：设计已冻结，可进入实现。Owner 已确认使用多个固定宽度整数而不是 arbitrary-precision big integer；不得在 128 qubits 处建立新的硬边界。Phase 5 只扩展 static U1 restricted Hamiltonian engine，不包含 U1 circuit、time evolution 或 gradient。

> API note: this historical specification predates the breaking Phase 8 API contract; current public names and signatures are defined in [`phase-8-api-coherence-spec.md`](phase-8-api-coherence-spec.md).

## 1. 阶段目标

Phase 5 将阶段二只支持 `nqubits < usize::BITS` 的 native U1 restriction 扩展到任意数量的 64-bit packed words。目标 workload 是 64、65、128、129、256 qubits 等宽系统中的低粒子数或低空穴数 sector，其中完整 Hilbert space 的 computational-basis state 已无法放入单个整数，但 restricted dimension `C(n,k)` 仍可安全表示和物化。

阶段完成后，现有 Python 调用应直接工作：

```python
sector = tcp.U1Sector(nqubits=129, particle_number=2)
restricted = h.restrict_u1(sector)
plan = restricted.mvp_plan()
out = plan.apply(state)
csr = restricted.csr()
```

用户不需要选择 `u64`、`u128` 或 wide mode。实现根据 `word_count = ceil(nqubits / 64)` 自动选择内部路径，公开语义与 qubit 数无关。

## 2. 已冻结的核心决策

1. **全空间 bitstring 使用任意宽 packed limbs**：一个 computational-basis occupation word 使用 `word_count` 个 `u64`，qubit `q` 位于 `words[q / 64]` 的 bit `q % 64`。通用路径必须覆盖 `word_count > 2`，不能把 `u128` 当作最终表示。
2. **restricted-space index 不使用 multiword**：restricted basis、CSR row/column 和 state-vector position 的逻辑 index 是有界整数。公开 sparse arrays 和 FFI index 使用 `u64`；Rust 对实际内存寻址执行 checked `u64 -> usize` 转换，Python/NumPy 还必须满足 `np.intp` shape/index 上限。
3. **不引入 BigInt/BigUint**：Phase 5 不增加 arbitrary-precision integer 依赖。固定宽度 limbs 更紧凑、更可预测，也允许直接使用 XOR、AND 和 popcount。
4. **现有 Python API 保持兼容**：`U1Sector`、`PauliOperator.restrict_u1()`、`U1RestrictedOperator` 和 `U1MvpPlan` 的公开方法、basis ordering、dtype 和结果语义不变。Phase 5 主要移除 native width rejection。
5. **完整 Hilbert-space targets 不扩展**：普通 `PauliOperator.dense()/coo()/csr()/mvp()` 仍受 `2**n` state dimension 限制。Phase 5 只扩展 `C(n,k)` restricted engine，不能让文档暗示 129-qubit full statevector 已可物化。
6. **聚合后验证 U1 conservation**：相同 source/destination 的全部 Pauli contributions 必须先以确定顺序求和并执行 exact-zero removal，再判断 sector leakage。`XX+YY` 等依靠项间抵消实现守恒的 Hamiltonian 必须继续成功。
7. **首版继续使用预编译 transition plan**：`restrict_u1()` 生成 destination-major flat plan，steady MVP 复用它。On-the-fly/direct combinatorial MVP 不是首版 REQUIRED 能力；只有 profile 证明 plan storage 或 setup 是代表性瓶颈时才增加内部 strategy。
8. **Phase 6 边界不前移**：不实现 number-conserving gate tape、U1Circuit、Trotter、含时演化、backend/JIT execution 或 gradient。

## 3. 当前实现边界

可直接复用的现有能力：

- `PauliWord` 已使用 `Vec<u64>` 保存 binary-symplectic X/Z masks，Pauli algebra、grouping 和 Z2 symmetry 已经是 multiword。
- Phase 3/4 propagation 已有 `<=128` inline key 和 `>128` wide fallback；Phase 5 不重写 propagation key。
- `U1MvpPlan` 已使用 destination-major flat `indptr/columns/values`，restricted operator 与 MVP plan 通过 `Arc` 共享 immutable storage，steady apply 已是 destination gather。
- Python `U1Sector.rank()/unrank()/basis_words()` 已定义 TensorCircuit-compatible basis ordering，宽系统 helper 已有 Python 语义。

必须替换的单字假设集中在 `crates/tencir-pauli-core/src/sector.rs`：

- `U1Sector::rank()`、`unrank()` 和 `basis_words()` 的 Rust 路径以单个 `usize` 表示完整 basis state。
- `U1Term.x_mask/z_mask` 是 `usize`。
- source、destination、XOR、phase parity、Hamming weight 和 leakage diagnostic 假定完整 basis state 可装入单字。
- `ensure_native_width()` 对 `nqubits >= usize::BITS` 直接失败。

`U1MvpPlan` 编译完成后只处理 restricted indices 和 complex values，因此 steady apply、dense/COO/CSR materialization 的核心数学不依赖 full-space bit width。主要改动应限制在 sector combinatorics、term compilation 和 transition-plan construction，不应无故重写已验证的 apply storage。

## 4. 表示、ordering 与 index 合同

### 4.1 Packed occupation words

内部 full-space occupation 使用 little-endian-by-qubit packed words：

```text
word_count = nqubits / 64 + (nqubits % 64 != 0)
qubit q    -> words[q / 64], bit (q % 64)
```

该表示与 `PauliWord.x_words()/z_words()` 一致。Pauli action 不再把 qubit `q` 转换成单整数的 bit `nqubits - 1 - q`；只在 combinatorial rank/unrank 中显式维护 TensorCircuit basis ordering。

最后一个 limb 中 `nqubits` 之外的 padding bits 必须始终为零。构造、XOR、complement、Hamming weight、equality 和公开 packed output 都必须应用 tail mask。Padding bit 不得影响 hash、rank、weight、leakage 或 deterministic ordering。

实现不得为每个 source、destination 或 transition 分配独立 `Vec<u64>`。允许并推荐的布局是：

- 单字路径使用现有 scalar fast path。
- 两字路径可以使用 `[u64; 2]` 或等价 inline scratch，但它只是优化，不能成为语义上限。
- 通用路径使用 flat contiguous limb arrays、borrowed slices 和每个 worker 一组可复用 source/destination scratch。
- Compiled X/Z masks 按 SoA 或 flat row-major layout 存储，不能使用 transition 数量级的 heap-backed word objects。

Phase 5 不要求把 `PauliWord`、propagation `PackedKey` 和 U1 occupation 强行统一成一个 public generic bitset。可以增加 sector-private packed helper；若抽取共享内部模块，必须证明没有给现有 algebra/propagation 热路径带来回退。

### 4.2 TensorCircuit basis ordering

Restricted basis 顺序保持阶段二合同：筛选 Hamming weight 为 `k` 的 computational-basis integers，并按整数升序排列，其中 qubit 0 是 full-space computational integer 的最高有效位。

内部 packed bit `q` 仍表示 qubit `q`，因此 rank 必须按 qubit `0..nqubits` 的 public 顺序扫描，而不是按 limb 数值升序解释 packed words。对于 occupied qubits `q_0 < q_1 < ... < q_{k-1}`，rank 可写为：

```text
rank = sum_i C(nqubits - q_i - 1, particle_number - i)
```

Unrank 是该映射的精确逆。任何 limb boundary、particle-hole optimization 或 combination iterator 都不能改变 basis order。

### 4.3 Restricted index 与 dimension

Full-space occupation 和 restricted index 是不同类型，不得混用：

- Full-space occupation：`[u64; word_count]`，可超过 64、128 qubits。
- Restricted logical index：`u64`，范围为 `0..C(n,k)`。
- Rust slice/vector offset：checked `usize`。
- Python array shape/index：还需满足平台 `np.intp` 上限。

`U1Sector` construction、native plan construction 和 materialization 必须在大分配前检查：

```text
C(n, k) <= u64::MAX
C(n, k) <= usize::MAX       # native addressability
C(n, k) <= np.intp.max      # Python array boundary, when applicable
```

CSR `indptr` 的最终 nnz、COO/CSR row/column values 和所有 prefix sums也必须 checked。即使 sector dimension 可表示，若 transition count、dense `dimension**2` 或 output bytes 溢出或超过 `max_bytes`，对应 target 仍须明确失败。

Rust 内部为了 apply 性能可以在上述 dimension gate 后继续把 `indptr`/`columns` 保存为 `usize`；公开 core sparse output、PyO3 和 NumPy arrays 继续使用 `u64`。禁止在每个 steady-MVP transition 上重复执行 fallible conversion。

### 4.4 Python helper compatibility

本阶段不借机重设计已发布的 basis helper：

- `U1Sector.rank(bitstring)` 继续接受非负 Python `int` 或长度为 `nqubits` 的 bit sequence。
- `U1Sector.unrank(index)` 继续对 `nqubits <= 64` 返回 Python `int`，对更宽系统返回 public-qubit-order bit tuple。
- `basis_words()` 对 `nqubits <= 64` 继续返回只读一维 `uint64` computational integers；对更宽系统继续返回只读二维 packed `uint64` array，shape 为 `(dimension, word_count)`，qubit `q` 位于 column `q // 64` 的 bit `q % 64`。
- Rows 始终按 restricted index/TensorCircuit computational integer 顺序排列。

宽系统 `basis_words()` 的实现应从 Python per-bit/per-row loops 移到一次 native batched call，但公开 shape、dtype、writeability 和 ordering 不变。若未来希望 `unrank()` 对所有宽度统一返回 Python big integer，必须另设兼容性 proposal/deprecation，不与 Phase 5 混合。

## 5. Core 数据结构建议

下列名字可以在实现中小幅调整，但存储和所有权合同是 REQUIRED。

### 5.1 Sector combinatorics

```rust
struct U1Combinatorics {
    nqubits: usize,
    particle_number: usize,
    active_number: usize, // min(k, n-k)
    complement: bool,
    dimension: u64,
    choose: Vec<u64>,     // flat checked table or equivalent
}
```

`active_number = min(k, n-k)` 使低粒子数和低空穴数路径具有对称复杂度。若使用 complement，weight-`k` word 的 rank 与 weight-`n-k` complement rank 的关系必须通过 property tests 固定，不能依赖未经证明的 bit-order直觉。

Binomial computation 必须避免中间乘法造成“结果本可表示但中间值溢出”的假 overflow。可使用 checked Pascal recurrence、约分后的 multiplicative recurrence 或等价精确算法。Table 的主要存储纳入 `max_bytes` cheap estimate。

### 5.2 Compiled X-mask groups

```rust
struct U1XGroup {
    x_offset: usize,
    term_start: usize,
    term_end: usize,
}

struct CompiledU1Terms {
    word_count: usize,
    x_words: Vec<u64>,
    z_words: Vec<u64>,
    weighted_coefficients: Vec<Complex64>,
    groups: Vec<U1XGroup>,
}
```

每个 canonical Pauli term 编译一次：

```text
x = term.word.x_words
z = term.word.z_words
weighted_coefficient = coefficient * i**popcount(x & z)
```

相同 X mask 的 terms 进入同一 group。Group 顺序和 group 内 term 顺序必须由 canonical input 唯一决定；浮点加法顺序不能依赖 hash seed、Rayon scheduling 或 unstable equal-key sorting。

### 5.3 Restricted plan storage

现有 destination-major形状继续使用：

```rust
struct U1MvpPlan {
    sector: U1Sector,
    indptr: Arc<[usize]>,
    columns: Arc<[usize]>,
    values: Arc<[Complex64]>,
}
```

`U1RestrictedOperator` 与 `U1MvpPlan` 继续共享同一 immutable storage，不因 `.mvp_plan()` 深拷贝 transition arrays。Python `.coo()`/`.csr()` 输出 `uint64` indices，`.dense()`/`.apply()` 输出 `complex128`。

不要求 plan 在完成后保留 full-space basis words、compiled term masks 或 combinatorial table；如果这些数据只用于 setup，应及时释放。若未来 direct strategy 需要保留，必须由 benchmark/profile 证明。

## 6. Transition 算法

### 6.1 Pauli action

对 packed source occupation `b`，一个 compiled Pauli term 的作用是：

```text
destination = b XOR x
amplitude   = weighted_coefficient * (-1)**parity(popcount(z AND b))
```

XOR、AND、parity 和 Hamming weight跨全部 limbs执行。`Y` phase 来自 `i**popcount(x & z)`，不得逐 qubit 重建 Pauli codes。所有 complex aggregation保持现有 complex128-compatible `Complex64`（两个`f64`）语义。

### 6.2 按 X mask 聚合

对固定 source，同一 X group 的所有 terms 产生同一 destination。算法必须按 group 内稳定顺序累加 amplitude：

```text
group_amplitude(source) = sum_j c_j * (-1)**parity(z_j & source)
```

若 aggregate 严格等于复数零，删除该 transition。只有非零 aggregate 才检查 destination Hamming weight：

- `weight(destination) == k`：计算 restricted destination rank 并保留。
- `weight(destination) != k`：返回 typed sector-leakage error。

不得使用 tolerance、magnitude cutoff、逐 term leakage 判断、自动 projection 或 silent drop。若聚合产生非有限结果，也必须明确失败。

### 6.3 Rank/lookup

Restricted destination lookup 首选 combinatorial rank，而不是 materialized basis hash map。低 `k` 路径应通过 set-bit iteration 或占据位置计算 rank，使主要复杂度接近 `O(min(k,n-k))`；通用实现允许按 limbs扫描，但必须避免 `O(nqubits)` 次重复 binomial computation。

Plan construction 不得构造完整 `2**n` lookup、statevector 或 matrix。显式 `basis_words()` 仍可按用户请求物化 `dimension * word_count` packed words，并单独受 `max_bytes` 保护。

### 6.4 Plan construction

推荐的确定性基线是两遍 source-ordered construction：

1. 预计算 combinatorics、compiled X groups 和 cheap allocation estimates。
2. 按 restricted source index 升序生成/迭代 packed source。
3. 第一遍按 X group 聚合，验证 leakage，并统计每个 destination row 的 entry count。
4. 对 row counts 执行 checked prefix sum，分配精确大小的 `columns/values`。
5. 第二遍以相同 source/group/term 顺序重新计算 transition，填入 destination-major storage。
6. 每个 CSR row 的 columns 必须按 source restricted index 升序排列；公开 COO/CSR 因而天然确定性。

Source generation 可以使用 `unrank_into()` 基线实现，也可以使用保持相同顺序的 incremental combination iterator。增量 iterator属于性能优化，必须通过对所有小 `n,k` 的 rank/unrank/basis-order exhaustive test 后才能替换基线。

Plan setup 的第一版可以串行，以建立清晰 reference和确定性。若 release profile 证明 setup 是代表性瓶颈，可使用 source ranges、thread-local row counts/scratch 和固定顺序 merge 并行化；禁止在 shared global hash map 或浮点 accumulator上使用调度相关的并发更新。

### 6.5 Transition upper bound

不要继续只用 `dimension * term_count` 估计主要 transition storage。按 X group 可以得到更紧的 sector-preserving candidate upper bound：若 X mask weight 为 `r`，只有 source 在这 `r` 个位置恰有 `r/2` 个 occupied bits 时 XOR 后仍保持 weight `k`，因此：

```text
candidate_count(x) = 0                                             if r is odd
candidate_count(x) = C(r, r/2) * C(n-r, k-r/2)                    otherwise
```

非法组合数参数视为零。对所有 distinct X groups 求和得到 plan nnz 的 cheap upper bound；实际 exact-zero cancellation 只会减少 entries。该 estimate 用于 checked arithmetic、capacity planning 和 `max_bytes` guard，但仍不声称是 exact peak RSS。

注意：这个 bound 不能替代 leakage validation。一个 group 仍可能对 sector source 产生非零 sector-changing amplitude，必须执行聚合后的显式验证。

## 7. Rust core API 迁移

现有单字 API不应立即删除。推荐增加 multiword primitives，并让 restricted engine内部只依赖新路径：

```rust
impl U1Sector {
    pub fn word_count(&self) -> usize;
    pub fn rank_words(&self, words: &[u64]) -> Result<u64, PauliError>;
    pub fn unrank_into(&self, index: u64, output: &mut [u64]) -> Result<(), PauliError>;
    pub fn basis_words_packed(&self, max_bytes: u128) -> Result<PackedU1Basis, PauliError>;
}

pub struct PackedU1Basis {
    pub dimension: u64,
    pub word_count: usize,
    pub words: Vec<u64>,
}
```

现有 `rank(usize) -> usize`、`unrank(usize) -> usize` 和单字 `basis_words()` 可以保留为 narrow convenience API，在不满足单字 width 时继续给出明确错误。它们不再被 `U1RestrictedOperator::new()` 调用。

若维护者希望直接修改尚处于 `0.1.x` 的 Rust core API，也必须在 CHANGELOG 中明确记录；Python public兼容仍是 REQUIRED。不要只把旧 `usize` 返回类型机械替换成 `u128`，因为这既破坏 API 又没有解决任意宽 full-space state。

Sector leakage error 不再能携带完整 wide input/output integer。Typed diagnostic 至少应包含可表示的 `source_index: u64`、expected particle number 和 actual destination weight；Python message必须保留明确的 `U(1) sector leakage` 文本。不得为打印错误而构造 Python big integer或巨型 bit string。

## 8. PyO3 与 Python API

### 8.1 Public Python API

以下签名保持不变：

```python
sector = U1Sector(nqubits, particle_number)
sector.dimension
sector.rank(bitstring)
sector.unrank(index)
sector.basis_words(max_bytes=...)

restricted = operator.restrict_u1(sector, max_bytes=...)
restricted.apply(state, max_bytes=...)
restricted.mvp_plan(max_bytes=...)
restricted.dense(max_bytes=...)
restricted.coo(max_bytes=...)
restricted.csr(max_bytes=...)
plan.apply(state, max_bytes=...)
```

`U1RestrictedOperator` 当前 docstring 中关于 `< usize::BITS` 的限制必须在实现完成时删除。README 中“multiword native restriction is planned for Phase 5”也必须改成准确的已实现说明，但不能在实现完成前提前修改成完成态。

### 8.2 Private native boundary

`pauli_restrict_u1()` 继续是一次粗粒度调用：Python 传入完整 canonical operator和 sector配置，Rust 完成 term compilation、validation和plan construction。禁止逐 basis state或逐 term跨 PyO3。

`u1_basis_words()` 应返回一个 contiguous `uint64` payload及 `dimension/word_count` metadata。Python wrapper只做 reshape、legacy narrow-format adaptation和read-only flag，不再用 Python loop逐 row/bit生成 wide basis。

所有长时间 restriction setup、basis materialization、MVP和sparse materialization必须释放GIL。Native `_native.pyi` 与 public typing必须同步更新，但 `_native` 仍是 private implementation detail。

## 9. Error、overflow 与内存合同

以下情况必须 fail fast：

- `particle_number > nqubits`。
- Packed word count、flat mask offset、binomial table或output byte arithmetic overflow。
- Packed input长度不等于 `word_count`，或tail padding bits非零。
- `C(n,k)`不能表示为restricted `u64` index，或不能转换成平台`usize`/NumPy `intp`以执行请求的plan/materialization。
- CSR nnz/prefix sum、COO/CSR index或dense entry count overflow。
- Operator与sector `nqubits`不兼容。
- 聚合后sector leakage、非有限coefficient或非有限aggregate。
- State shape不等于restricted dimension。
- 可廉价估算的major output/workspace超过`max_bytes`。

`max_bytes`继续遵守项目全局合同：默认16 GiB、允许`None`表示不设public guard、只对可廉价估计的主要output/workspace提供best-effort保护，不是exact peak-RSS保证。Phase 5 estimate至少覆盖compiled masks/groups、combinatorial table、row counts/prefix arrays、最终transition storage、显式packed basis output和dense/COO/CSR主要output。

Restriction setup不得隐式物化完整packed basis。每worker scratch是`O(word_count + group_count)`或更小；若某项优化引入`O(dimension * word_count)`常驻basis cache，必须有代表性benchmark证明并在profile/内存结果中单独报告。

## 10. Correctness reference 与测试矩阵

### 10.1 独立 reference

新增 Python big-int/combinatorial oracle，使用 Python arbitrary-width integers表示full-space basis state，仅作为test/reference代码。Oracle必须独立实现：

- TensorCircuit basis order和rank/unrank。
- Pauli X/Z/Y action与complex phase。
- 相同destination aggregation和exact-zero removal。
- Sector leakage判断。
- Restricted dense/COO/CSR/MVP expected结果。

64+ qubit reference不得构造`2**n` state、full dense matrix或调用被测native rank/transition kernel生成expected结果。低-k sector可枚举`C(n,k)` states。

### 10.2 REQUIRED boundary cases

至少覆盖：

- `nqubits = 0, 1, 63, 64, 65, 127, 128, 129, 256`。
- `k = 0, 1, 2, n-2, n-1, n`，在dimension安全时覆盖中间sector回归。
- Wires和Pauli support跨越`62/63`、`63/64`、`64/65`、`126/127`、`127/128`边界。
- Tail word不是64整数倍时的padding mask。
- Identity、diagonal Z、single hopping、nearest-neighbor chain、long-range hopping、XX/YY cancellation和明确泄漏的single X/Y反例。
- 包含Y和complex coefficients的守恒operator；至少一个non-Hermitian但sector-preserving directed hopping case。
- 不同canonical terms具有相同X mask、不同Z parity并发生exact cancellation的case。
- Empty operator、非法sector/index请求，以及one-dimensional `k=0`/`k=n` sector。
- Dimension、nnz、offset、bytes和packed length overflow/error mapping。

### 10.3 Property tests

Rust和Python tests至少证明：

- `rank_words(unrank_into(i)) == i`。
- 所有生成word的weight等于`k`且padding为零。
- Basis order严格匹配Python big-int computational integer升序。
- Particle-hole path与直接weight-k reference给出相同rank/order。
- Multiword XOR/parity/popcount与Python integer reference一致。
- Restricted MVP、dense、COO和CSR在可物化的小dimension上完全一致。
- Public outputs在重复调用、不同hash seed和受支持thread count下保持确定性。
- 63/64/65边界的旧窄结果不发生语义变化。

## 11. Benchmark 与 profiling

### 11.1 Stable workloads

Rust Criterion和Python/FFI benchmark至少保留以下固定workloads：

1. **63/64/65q k=2 boundary chain**：相同nearest-neighbor XX+YY+Z Hamiltonian，显示单字到双字边界的setup/apply变化。
2. **128q k=2 chain**：覆盖恰好两个limbs、跨63/64 hopping和代表性low-k规模。
3. **129q k=2 chain**：强制进入`word_count > 2`通用路径，防止实现只支持`u128`。
4. **256q k=1/k=2 chain**：验证任意宽limbs、combinatorial scaling和plan storage。
5. **Low-hole control**：例如128q k=126，验证particle-hole优化与k=2具有一致的combinatorial复杂度趋势。
6. **Long-range/duplicate-X workload**：覆盖跨limb X masks、term grouping和aggregation热点。

每个case记录：`nqubits`、`particle_number`、`word_count`、canonical term count、distinct X-group count、sector dimension、actual nnz、setup time、steady MVP time、COO/CSR materialization time、plan/output bytes、thread count和numerical error。

### 11.2 Performance process

所有性能结论使用release build。先保存Phase 2现有12q/k2、16q/k8和26q/k2基线，确保Phase 5没有明显破坏窄路径；再profile 128/129/256q representative setup和steady MVP。

优化优先级：

1. 消除per-source/per-transition heap allocation和逐qubit term decode。
2. 使用X-mask grouping、flat masks和reusable scratch。
3. 优化combinatorial rank/iteration和binomial lookup。
4. 只有profile确认后才增加setup并行、two-word专用kernel、SIMD或direct MVP strategy。

不设置固定倍数speedup或wall-time CI gate。完成态必须包含可复现benchmark label和至少一次代表性profile记录；仅通过correctness tests而没有64+ release benchmark不能宣称Phase 5完成。

## 12. 实现切片

### P0：Reference、边界回归和失败用例

- 增加独立Python big-int U1 reference。
- 先用测试记录当前64q native rejection这一待移除边界，并增加实现前会失败的64/65/128/129q expected-success cases。
- 固定basis order、wide packing、Y phase、leakage-after-aggregation和particle-hole properties。

Acceptance gate：reference不调用native transition/rank；现有窄tests继续通过；新增wide restriction tests在实现前以预期width错误失败。

### P1：Multiword sector primitives

- 实现checked combinatorics、`word_count`、`rank_words`、`unrank_into`和packed basis materialization。
- 实现tail mask、multiword XOR/AND/popcount/parity和complement helpers。
- 保留现有narrow convenience API，native restriction不再依赖单字rank/unrank。

Acceptance gate：0/63/64/65/128/129/256q rank-unrank、basis order和padding properties全部匹配独立reference；overflow在分配前失败。

### P2：Compiled X groups 与wide leakage validation

- 直接从`PauliWord` packed masks构造flat compiled terms。
- 按相同X mask稳定分组，预计算Y phase。
- 实现multiword source action、stable aggregate、exact-zero removal和聚合后leakage判断。
- 增加基于X-weight combinatorics的transition upper bound。

Acceptance gate：XX+YY跨63/64和127/128边界成功；single X/Y反例失败；complex/Y和directed hopping匹配reference。

### P3：Restricted plan、MVP 与 sparse targets

- 以两遍deterministic construction生成destination-major flat plan。
- 保持restricted operator/MVP plan共享storage和steady destination gather。
- 让apply、dense、COO和CSR在wide systems上工作，保留u64 public indices和checked native offsets。
- 释放不再需要的setup-only wide buffers。

Acceptance gate：64/65/128/129/256q安全dimension cases的MVP/CSR匹配reference；同一row的columns有序；重复调用byte-for-byte deterministic。

### P4：PyO3、Python facade 和文档兼容

- 移除native width rejection。
- 把wide `basis_words()`迁入一次batched native call。
- 更新`_native.pyi`、public docstrings、README limitation和examples。
- 确认long setup/apply/materialization释放GIL并可并发调用。

Acceptance gate：现有Python调用无需修改；narrow和wide helper shape/dtype/order保持合同；无逐state/term FFI。

### P5：Profiling 与优化

- 添加并运行required Rust/Python benchmarks。
- profile 128q和129/256q setup、rank、aggregation和steady MVP。
- 优先修复实际主瓶颈；任何two-word specialization或parallel construction都必须保持reference和determinism gates。
- 保存clean release benchmark label并更新implementation status。

Acceptance gate：全套quality checks通过；窄路径无未解释的代表性回退；wide path没有per-transition allocation；benchmark metadata和profile结论可复核。

## 13. 非目标

- 不实现full-space 64+ qubit statevector、dense/COO/CSR或MVP。
- 不引入BigInt/BigUint或把full-space state序列化成十进制大整数。
- 不实现自动U1 symmetry discovery、多个U1 charges、non-Abelian symmetry或一般charge sector。
- 不实现U1Circuit、state preparation gate、number-conserving circuit execution、time evolution、Trotter、backend/JIT adapter或gradient。
- 不修改TensorCircuit主仓库。
- 不增加coefficient tolerance、silent projection或approximate leakage policy。
- 不把`max_bytes`升级成exact peak-RSS合同。
- 不在没有profile证据时重写现有destination-major steady MVP或增加public strategy selector。

## 14. 最终验收清单

- Full-space occupation使用任意数量`u64` limbs，129/256q实际走通，不存在64或128硬上限。
- Restricted logical indices和public sparse indices保持`u64`；native/NumPy addressability执行checked gate。
- TensorCircuit basis order、qubit mapping、Y phase、complex coefficients和exact-zero aggregation与independent reference一致。
- Leakage只在相同destination contributions聚合后判断。
- `U1Sector`、`restrict_u1()`、restricted apply/MVP/dense/COO/CSR公开API兼容。
- Restriction和MVP不物化`2**n` state、full matrix或完整basis lookup。
- Plan storage flat、deterministic、由restricted operator和MVP plan共享；hot setup无per-transition heap allocation。
- 0/63/64/65/127/128/129/256q以及low-particle/low-hole properties和differential tests通过。
- Rustfmt、Clippy `-D warnings`、workspace tests、Black、Ruff、strict mypy、`maturin develop --release --locked`和pytest通过。
- Required Criterion/Python release benchmarks、storage/scaling metadata和representative profile已保存并写入`implementation-status.md`。
- README、architecture、typing、CHANGELOG和限制说明与实际实现同步，Phase 6边界未被提前扩大。

## 15. 实施交接规则

按P0→P5纵向切片推进，每片完成core→batched native→typed Python→independent differential test后再进入下一片。Correctness patch、data-layout patch和profile-driven optimization尽量分开提交；不要在wide reference尚未建立时先写专用fast path。

`implementation-status.md`只在有测试证据后记录已完成切片。Phase 5开始实施时把它设为active milestone；完成前README仍应描述当前native width限制。若实现者发现必须改变public basis helper返回类型、加入direct MVP public selector或扩大到U1Circuit/time evolution，必须停止并提交新的owner decision，不能自行扩大本Spec。
