# TenCirPauli 代码库审查报告

审查范围：当前 `main` 工作区的 Rust core、PyO3 binding、Python public API、TensorCircuit boundary、测试/benchmark/CI 与 `docs/vibe` 状态文档。审查重点覆盖实现正确性、接口合同、内存与 GIL、性能路径、确定性、自洽性和可维护性。

## 结论摘要

当前代码库的基础质量较好：Rust/Python 分层清楚，核心 Pauli 代数、Hamiltonian target、U(1) sector、deterministic propagation、SPPS 和 U1 circuit 均有实现，现有测试与静态检查通过。审查中没有发现会在常规小规模输入下普遍破坏主流程的算法错误，但发现了一些真实缺陷、接口边界问题和需要基准才能决定是否值得投入的性能方向。

本次重新排序的原则是：实际会影响正确性、OOM 风险、并发吞吐或 API 可预测性的，且改动范围清楚的，标为 `FIX_NOW`；真实但适合下次触及时顺手处理的，标为 `FIX_WITH_NEXT_TOUCH`；只有极端输入、低概率对象生命周期或需要新 API/新内部表示的，标为 `DEFER` 或 `OBSERVATION`。因此，报告保留所有发现，但不再把所有发现都当作当前迭代的必修项。

建议当前迭代修复 M1、M2、M3、M4、M6、N1、N2、N3 以及 D1–D3。M5、M7、N4、N5 暂不建议为了“完整”而立即扩展设计；其中 M5 是低概率 cache 生命周期问题，M7/N4/N5 则需要 workload、profile 或更大的 API/表示层设计来证明投入值得。

`unsafe` 不是当前必须消除的缺陷。U1 pair kernel 的性能动机合理，应该补齐局部 safety contract 和针对真实并行阈值的 invariant/differential test；不建议为了追求零 `unsafe` 而引入锁、复制或更复杂的抽象。

## 处置标签与执行总表

| 标签 | 含义 | 本报告中的项目 |
| --- | --- | --- |
| `FIX_NOW` | 真实影响明确，改动边界清楚，当前修复收益高于维护成本。 | M1、M2、M3、M4、M6、N1、N2、N3、D1、D2、D3 |
| `FIX_WITH_NEXT_TOUCH` | 问题真实，但适合在相关代码下一次修改时一起处理，不值得单独扩大范围。 | 本轮没有必须单列的项目；M6 的测试可与下一次 U1 kernel 修改一起扩充。 |
| `DEFER` | 风险存在但低频/极端，或修复会引入新的公共接口、缓存策略或核心表示设计。 | M5、M7、N4 |
| `OBSERVATION` | 先保留问题和测量计划，只有代表性 workload 证明它是瓶颈后再改。 | N5 |

`MAJOR` 表示问题的潜在影响，不表示必须在当前迭代全部修复；最终优先级以这里的处置标签为准。

## 可复现验证结果

- `cargo fmt --all -- --check`：通过。
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`：通过。
- `cargo test --workspace`：26 passed。
- `black --check python tests benchmarks scripts examples`：通过。
- `ruff check python tests benchmarks scripts examples`：通过。
- `mypy`：14 个 Python source files，0 issues。
- `pytest -q`：184 passed，1 skipped；`python scripts/check.py --benchmark smoke` 全流程也通过，benchmark harness 为 133 passed、77 deselected。
- 工作区初始状态干净；审查没有修改任何源文件，仅新增本报告。

## 合规检查

| 检查项 | 状态 | 说明 |
| --- | --- | --- |
| Rust core 不依赖 Python/PyO3/TensorCircuit | PASS | 当前模块边界符合架构目标。 |
| 主要 native 长计算释放 GIL | FAIL | U1 observable canonicalization 有路径在 `allow_threads` 之外执行。 |
| 输出确定性和 canonical ordering | PASS | 现有实现有排序和回归测试，未发现常规路径的 hash iteration 泄漏。 |
| 数值/phase/ordering 基础语义 | PASS（常规范围） | dense/reference 与现有测试覆盖良好，但极端 tolerance 边界失败。 |
| `max_bytes` 主要输出/workspace guard | FAIL | MVP 输出先分配；batch worker guard 存在低估。 |
| 公开输入显式拒绝错误类型 | FAIL | grouping reconstruction 会把非二值浮点静默转换为 0/1。 |
| unsafe 局部安全论证与 dedicated test | FAIL | U1 parallel pair kernel 的 unsafe 只有一般性注释。 |
| 代码、README、状态文档相互一致 | FAIL | Phase 6、默认内存上限和 U1 状态存在互相冲突的陈述。 |

## MAJOR 问题

### M1：Hermiticity tolerance 可能因平方溢出把明显非 Hermitian 算符判为 Hermitian【FIX_NOW】

位置：`crates/tencir-pauli-core/src/operator.rs:339-347`，Python 入口为 `python/tencirpauli/pauli.py:477-489`。

实现使用 `difference.norm_sqr() <= tolerance * tolerance`。当 coefficient 和 tolerance 都是合法 finite float、但平方溢出为 `inf` 时，`inf <= inf` 会返回真。已复现：`PauliOperator.from_terms(1, [("I", 1.7e308j)]).is_hermitian(1e308)` 返回 `True`，而 tolerance 为 0 时返回 `False`。这会让显式 tolerance API 的结果失真，并可能使依赖该判断的上层路径接受不应接受的 observable。

这是一个极端输入缺陷，但修复不需要复杂数值防御：用 `difference.re.hypot(difference.im) <= tolerance` 或等价的稳定 Euclidean norm 替代平方比较即可，保持原有 tolerance 语义并避免 `inf <= inf`。加上一个接近 `f64` 上限、一个阈值内和一个阈值外测试，成本很小，属于应顺手修复的边界 bug；不需要额外建立 finite-value 体系、allocator 统计或大范围数值策略。

### M2：MVP `max_bytes` 检查发生在输出 buffer 分配之后【FIX_NOW】

位置：`crates/tencir-pauli-core/src/hamiltonian.rs:160-195`、`crates/tencirpauli-native/src/hamiltonian.rs:38-58` 和 `:189-213`。

`MvpPlan::apply()` 先执行 `vec![...]`，之后才进入 `apply_into()` 的预算检查；PyO3 `NativeMvpPlan.apply` 和 one-shot `pauli_mvp_array` 也先创建 NumPy output，再调用 Rust guard。因此 `max_bytes` 过小时仍可能先分配完整的 `2**n` complex output，才抛 `MemoryError`。这违背了“分配 major output 前 fail fast”的设计意图，且在接近内存上限的进程中可能触发真正的 OOM。

建议把 dimension、state length 和 output bytes 检查提取为无副作用的 preflight，在 `Vec`/NumPy allocation 之前调用；`apply_into` 继续保留 caller-owned buffer 的长度检查。增加一个大维度、低预算的回归测试即可，不需要为了精确统计 RSS 引入 allocator instrumentation。

### M3：PropagationBatch 的预算计算会把“至少一个 worker 也放不下”错误地截断为剩余预算【FIX_NOW】

位置：`crates/tencir-pauli-core/src/propagation.rs:739-756`、`:526-552`。

`allowed_batch_workers()` 用 `.max(1)` 强制至少一个 worker；随后构造函数在 `:541` 使用 `per_worker_bytes.min(remaining_budget(...))`，把实际 worker storage 截断成剩余预算，从而可能通过构造期检查。已复现：12 qubit、48 个 rotation、16 行 observable、`max_bytes=100000` 可以构造 `PropagationBatch`，但首次 `expectations()` 才报 `requested 147456 bytes exceeds memory limit 100000`。

这不是要求精确 RSS，而是当前代码已经知道的 worker workspace 被低估，导致 API 的失败时机不稳定。建议若 `remaining < per_worker_bytes` 直接在 construction 或 execution preflight 返回 `MemoryLimit`；只有在一个完整 worker 能放下时才允许 serial path，且 `shared_bytes` 必须使用实际 `active_workers * per_worker_bytes`，不能使用 `min`。补充“预算小于一个 worker”和“可运行的 serial path”两组测试即可；多 worker 情况可在已有预算测试中覆盖。`map_observables()` 中 `.unwrap_or(1)` 的 overflow fallback 如果没有可复现输入，可先不扩大范围，等相关代码触及时再改为显式错误。

### M4：U1 observable canonicalization 在持有 GIL 时执行，破坏高层长调用的并发性【FIX_NOW】

位置：`crates/tencirpauli-native/src/u1_circuit.rs:159-203`、`:236-275`。

`NativeU1CircuitPlan.expectation/value_and_grad` 与 `NativeU1FinalState.expectation/value_and_grad` 在调用 `py.allow_threads` 前执行 `build_canonical_operator(...)`。大 observable 的 canonical validation、term conversion 和分配会阻塞所有 Python threads；同一仓库其他 Hamiltonian、propagation、SPPS 和 symmetry 入口已经把相同工作放进 `allow_threads`，因此这是新 U1 路径的回归。

建议将纯 Rust 的 `build_canonical_operator` 和后续 `plan.*` 一并放进一个 `allow_threads` closure，只在 closure 返回后创建 NumPy result。这里不需要新增复杂的通用抽象；现有 binding 模式足以完成修复。并发 smoke test 可作为后续回归补充，不必为了测试而建立严格的 wall-time 门槛。

### M5：`PropagationCircuit`/`SPPSCircuit` 的 cache key 只保存 `id(...)`，存在对象回收后的 id 复用风险【DEFER】

位置：`python/tencirpauli/propagation_circuit.py:356-400` 和 `python/tencirpauli/spps_circuit.py:125-143`。

cache key 使用 `id(observable)`、`id(initial_state)`，而 cached tuple 没有保留对应 Python object；底层 native plan 也只保留已编译内容。CPython 允许对象回收后复用地址，长生命周期 circuit 在这种情况下可能把新 observable/state 误认为旧 plan，返回 stale result。风险低频但属于缓存导致的数值错误，且没有必要承担。

这是理论上可能造成 stale result 的设计风险，但触发概率低；保存 strong reference 又会延长大 observable/state 的生命周期，引入新的内存成本，generation/content token 则会扩大 public object 设计。暂不单独改 cache。若出现实际复现，或下一次重做 cache，再选择“持有对象并用 `is` 比较”或显式 token，并补 GC/id-reuse 回归测试；在此之前不要为了理论风险增加 cache 层次。

### M6：U1 circuit 的 parallel pair kernel 使用 raw pointer，但没有满足项目要求的局部安全论证和 dedicated invariant test【FIX_NOW】

位置：`crates/tencir-pauli-core/src/u1_circuit.rs:1089-1101`。

实现把 `state.as_mut_ptr()` 转成整数后在 Rayon 中并行写入。当前注释声称 pair endpoints disjoint，但 unsafe block 内没有 `// SAFETY:` 级别的局部证明，也没有专门验证所有 `PairIndex` 在 bounds 内且 endpoints 两两不重叠的测试；现有并发测试主要覆盖小规模调用，不足以锁定触发 `U1_CIRCUIT_PARALLEL_PAIR_THRESHOLD` 的实际 kernel invariant。一旦 pair map 构造或融合逻辑改变，可能产生未定义行为。

建议先做最小修复：在 unsafe block 紧邻处补完整 `// SAFETY:` 说明，明确 slice lifetime、每个 pair 的两个 index 合法、pair 间无重复写入，并增加一个超过 `16384` pairs 的 safe-reference differential test 或 invariant test。不要仅为了消除 `unsafe` 而重写成锁或复制版本；只有 profile 证明该 kernel 不是瓶颈且 safe layout 成本可接受时，才评估更大的结构调整。

### M7：Pauli operator 乘法没有 public allocation budget，且 commutator/anticommutator 会重复构造完整乘积【DEFER】

位置：`python/tencirpauli/pauli.py:457-467`、`:741-746`，核心实现为 `crates/tencir-pauli-core/src/operator.rs:266-321`。

`multiply()` 的候选项数量是 `L_left * L_right`，每个结果还进入 `FxHashMap<PauliWord, Vec<Complex64>>`；没有 `max_bytes`、candidate-count guard 或按预算拒绝机制。`commutator()`/`anticommutator()` 又分别计算 `A*B` 和 `B*A`，随后再合并，峰值和重复工作都接近两次乘法以上。问题真实，但当前建议不把它直接变成新的公共预算接口：需要定义估算误差、错误类型、`max_terms` 与 `max_bytes` 的关系，并设计 fused algebra API。

暂不修复。后续若 profile 显示代数运算是实际热点，先以代表性 term count、qubit count 和峰值 workspace 建 benchmark，再考虑内部 fused path；只有确实需要用户控制超大输入时，才增加 public budget。不要仅因存在理论上的大输入放大，就先引入一套新的 allocation accounting。

## MINOR 问题

### N1：QWC reconstruction 会静默把非二值浮点 bitstrings 转成 int8【FIX_NOW】

位置：`python/tencirpauli/grouping.py:32-51`。`np.asarray(bitstrings, dtype=np.int8)` 在校验前执行，因此 `[[0.5]]` 变成 `[[0]]`、`[[1.9]]` 变成 `[[1]]`，当前 API 会正常返回而不是拒绝非法测量结果。

建议先保留原始 dtype 做二维、整数性和 `{0,1}` 校验，再转换为紧凑整数数组；明确是否接受 bool，若接受应专门记录并测试。改动局部，且能避免调用者得到看似成功但语义错误的重构结果。

### N2：grouping 的 `max_matrix_entries`/`max_edges` 缺少统一的非负整数校验【FIX_NOW】

位置：`python/tencirpauli/grouping.py:75-100` 以及 `python/tencirpauli/pauli.py:550-580`。例如 `max_matrix_entries=-1` 对空 operator 也走到 `MemoryError`，而其他非法类型可能在 PyO3 参数转换处产生不一致的异常。

建议复用一个小型公共 `_validate_nonnegative_int`，在 Python boundary 统一拒绝 bool、负数和非整数，并在 docstring 中说明限制单位是 entries 还是 bytes。不要为此引入更复杂的配置对象。

### N3：Backend MVP plan 暴露的 NumPy arrays 可被修改，和 immutable/frozen plan 语义不一致【FIX_NOW】

位置：`python/tencirpauli/hamiltonian.py:111-156`、构造位置 `python/tencirpauli/pauli.py:672-698`。已确认 `plan.coefficients.flags.writeable` 和 `plan.x_words.flags.writeable` 都是 `True`，调用者可以直接改变已编译 plan 的数值或结构。

建议在 plan 构造时将 `x_words`、`z_words`、`coefficients` 设为 C-contiguous、read-only，并加入 mutation regression test；这是低成本地兑现 frozen plan 语义。如果计划设计上允许修改，再另行改成显式 `with_coefficients()`/recompile API，不要同时引入两套语义。

### N4：常规 PauliWord 单项运算仍是细粒度 FFI/对象往返，限制热路径吞吐【DEFER】

位置：`python/tencirpauli/pauli.py:156-188`。`multiply()` 每次先把两个 packed word 转成 codes，再跨 PyO3，再将结果 codes 重新构造成新的 `PauliWord`；`weight/support/codes` 也各自是独立 native call。对用户逐项处理大量 terms 的场景，这会把 Rust 优势消耗在 Python/FFI 和临时对象上。

这是合理的性能方向，但需要 batch API 或 packed native handle，属于表示层设计，不是单点小修。暂不扩展公共 API；若 benchmark 证明逐项操作是实际瓶颈，再设计 batch multiply/commutation/support，并同时评估对象 materialization 和 FFI 边界成本。单项便利 API 继续保留。

### N5：BackendMVPPlan 的纯 NumPy executor 每个 term 都分配完整 state-sized phase/row 临时数组【OBSERVATION】

位置：`python/tencirpauli/hamiltonian.py:125-156`。每个 term 创建 `rows`、`phase`，且每个 qubit 的 Y/Z 处理还创建 `np.where` 中间数组，复杂度和临时分配约为 `O(T*n*2**n)`；这条路径不适合大 term-count，且与 native MVP plan 的预计算策略重复。

静态上看有额外临时分配，但是否值得改取决于该 fallback 的真实使用规模和 backend/JAX 主路径占比。先把 `BackendMVPPlan.apply` 明确定位为小规模 NumPy fallback，并记录 T、n scaling benchmark；只有 profile 显示它占用显著时间或内存，再考虑预计算 mask、复用 scratch 或 batched gather/scatter，避免维护两套未经证明的优化实现。

## 设计与文档自洽性

### D1：`docs/vibe/implementation-status.md` 同时把 Phase 6 写成完成、under acceptance 和 future【FIX_NOW】

位置：`docs/vibe/implementation-status.md:7` 写“Phase 1–6 已完成”；`:158` 写 Phase 6 仍 under acceptance review；`:176-178` 又把 Phase 6 写成 scheduled future phase。另有 `:144` 仍描述 Phase 5 不实现 U1Circuit，而 `:162`、`:168-172` 又描述 U1Circuit 已实现。

建议指定一个 active status section，明确当前 commit、阶段状态（implemented / under acceptance / deferred）和剩余 gate；历史 review 保持只读归档，不把过去的“future”段落留在当前状态章节。发布前让 README、CHANGELOG、phase spec、implementation-status 和 benchmark manifest 使用同一状态表。

### D2：默认内存上限的文档已过期【FIX_NOW】

当前实现 `python/tencirpauli/hamiltonian.py:12-15` 为 16 GiB，测试 `tests/test_hamiltonian.py:145-146` 也锁定 16 GiB，但 `docs/vibe/implementation-status.md:29` 仍写默认值为 4 GiB。建议一次性更新所有历史引用；若要保留历史事实，标注为“Phase 1 historical value”，避免用户据此估计当前内存行为。

### D3：性能证据与验收状态没有统一入口【FIX_NOW】

当前 `scripts/check.py --benchmark smoke` 只验证 benchmark harness 能运行，不记录可比较的 steady runtime；`docs/vibe/implementation-status.md:172` 明确 Phase 6 的 matched native/JAX 和完整 handoff matrix 尚未记录，但文件开头又给出“Phase 1–6 已完成”。当前先统一状态文档的措辞和入口：明确 smoke 只代表 harness 可运行，不把它写成性能验收。只有在确实需要发布级性能证据时，再增加包含 commit、命令、输入规模和 accuracy 的简短 benchmark manifest，不要为了形式建立复杂的外部 benchmark 服务。

## `unsafe` 政策建议

`unsafe` 不是这类库的必然组成部分，但在追求高吞吐的 Rust 数值库里很常见，尤其是 Rayon 对不规则、互不重叠的索引并行写入时。当前 U1 pair kernel 使用 raw pointer 的理由是绕开锁、原子操作或额外复制；如果 pair-map invariant 确实成立，这种使用可以是合理的。真正需要审查的是 invariant 是否能被证明和持续验证，而不是 `unsafe` 这个词本身是否出现。

不建议在 `AGENTS.md` 中写“禁止任何 `unsafe`”。绝对禁用的好处是规则简单、UB 风险和审查范围更小；代价是可能迫使实现采用锁、原子操作、重复分配或复制，直接损害本项目的吞吐和内存；也可能把同样的复杂性藏到依赖或更难审查的 safe wrapper 后面。对于当前这类单一、局部、可由索引不重叠性支撑的 kernel，零 `unsafe` 目标的收益不足以抵消这种约束。

更合适的是保留“默认 safe Rust，例外允许 `unsafe`”的政策，并把现有规则具体化为：每个 `unsafe` block 都必须紧邻一段 `// SAFETY:` 说明；说明必须覆盖 bounds、aliasing、线程间写入不重叠、slice lifetime 和必要的初始化条件；unsafe 范围保持最小，不把整个函数包进 unsafe；至少有一个针对 invariant 的回归或 differential test，涉及并行阈值时要覆盖实际并行路径；如果存在明显的 safe alternative，只有在 release benchmark 证明其性能/内存代价不可接受后才保留 unsafe。这里的“至少一个”应按风险匹配测试，不需要为每个 block 建立复杂的验证框架。

因此本仓库的建议措辞不是“禁止 unsafe”，而是“允许经过证明的局部 unsafe，并将其视为需要额外审查的性能实现细节”。如果未来 unsafe 数量增加、跨越多个模块或无法写出清楚的 safety contract，再考虑收紧为模块级白名单或要求单独 review；当前不需要先建立这类制度。

## 建议执行顺序

1. 先修 M1、M2、M3、M4、N1、N2、N3 和 D1–D3；这些修改边界小，分别改善极端数值边界、预算失败时机、并发吞吐、输入语义、plan 不可变性和发布判断。
2. 修 M6 的局部 `SAFETY` 说明，并补一个覆盖真实并行阈值的 invariant/differential test；不改写为零 `unsafe` 实现。
3. M5、M7、N4、N5 保留在 backlog：遇到相关代码触及时顺手处理；涉及性能的项目先用代表性 term count、qubit count、重复执行次数和峰值 workspace 做 release-mode benchmark，再按 profile 决定是否投入。
4. 不要把复杂的极端浮点 hardening、cache 重构、代数新预算 API、batch FFI 表示和 NumPy fallback 重写混入当前小修批次；这些都是可能合理、但需要独立设计决策的工作。

## 最终声明

本次审查未修改任何源代码、测试、配置或文档源文件；仅新增 `REVIEW_REPORT.md` 作为审查输出。

## 本轮处置记录（2026-08-03）

本报告作为 Phase Alpha Review Report 归档。报告中标记为 `FIX_NOW` 的 M1、M2、M3、M4、M6、N1、N2、N3 和 D1–D3 已完成修复；M5、M7、N4、N5 按报告建议保留为 deferred/backlog，没有扩大本轮公共 API 或内部表示设计。

修复内容包括稳定 Hermiticity norm、MVP 输出分配前的 memory preflight、PropagationBatch 的完整 worker 预算检查、U1 observable canonicalization 的 GIL 范围收缩、U1 pair kernel 的局部 `SAFETY` contract 与实际并行阈值 differential test、QWC bitstring 与非负整数边界校验，以及 read-only backend MVP plan arrays。状态文档已统一 Phase Alpha、Phase 6、Phase 6.5、Phase 7、16 GiB 默认上限和 benchmark smoke 的验收含义。

最终验证命令和结果以本次提交后的 `docs/vibe/implementation-status.md` 为准；该归档保留审查当时的原始发现与处置标签，后附本节记录 remediation outcome。
