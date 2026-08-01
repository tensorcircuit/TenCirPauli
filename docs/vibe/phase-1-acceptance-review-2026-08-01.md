# TenCirPauli Phase 1 验收报告

审查日期：2026-08-01

归档状态：Phase 1 初次验收与同日 remediation 记录。初次结论针对 commit `44ae2858d2440164f82b5be31dd40525613bc13b`；文末 remediation 结论基于修复后的本地工作树和完整复验。

审查对象：commit `44ae2858d2440164f82b5be31dd40525613bc13b`，覆盖 Rust core、PyO3 binding、Python public API、TensorCircuit adapter、正确性测试、质量门禁和 release-mode benchmark。

## 验收结论

结论：**暂不建议正式签署 Phase 1 验收通过**。功能范围、分层架构、Hamiltonian 数值结果、确定性 sparse 输出、打包流程和多数性能目标已经达到较高完成度；但 Rust public API 仍有一个可复现的错误结果问题，Python public API 还有 canonical invariant、有限系数 invariant 和逐 term FFI 三类硬规范违例。修复下述 `CRITICAL` 与 `MAJOR` 正确性/规范项后，再做一次收口验收即可，不需要推倒当前架构。

本次审查没有修改 `.rs`、`.py`、测试、benchmark 或其他源码；只新增本报告。复现 Rust core 缺陷时使用了 `/private/tmp` 下的临时 probe，不属于仓库源码。

## 合规检查表

| 检查项 | 状态 | 结论 |
| --- | --- | --- |
| Pure Rust core 与 PyO3/Python/TensorCircuit 隔离 | PASS | core 仅依赖 Rayon；TensorCircuit 直接 import 只存在于 adapter。 |
| Phase 1 范围控制 | PASS | 未越界实现 symmetry、propagation、native gradients 或 fixed-buffer top-k。 |
| Pauli phase、qubit ordering、dense/COO/CSR/MVP 数值语义 | PASS | 独立 NumPy differential tests 与首尾 qubit ordering tests 通过。 |
| Public canonical representation 与有限系数 invariant | FAIL | Python `PauliWord` 可保存非 canonical packed bits；`scale(0)` 保留零项，有限乘数也可产生 `inf`。 |
| Rust public MVP API 正确性 | FAIL | `MvpPlan::apply_into` 的串行分支不覆盖调用者缓冲区。 |
| Coarse-grained FFI | FAIL | `PauliOperator` 构造和 `_arrays()` 会逐 term 调用 PyO3，然后 native 端再次 canonicalize。 |
| 长计算释放 GIL | FAIL | 仅 MVP apply 释放 GIL；canonicalization、grouping、dense/COO/CSR 和 plan construction 均未释放。 |
| 分配前完整峰值内存 guard | FAIL | core guard 没有覆盖 Rust-to-NumPy complex buffer 复制等 end-to-end 峰值。 |
| Deterministic public output | PASS | Rust canonicalization 和 sparse/grouping 输出使用稳定顺序；但非 canonical Python `PauliWord` 的 equality/hash 需要修复。 |
| Format/lint/type/test/package 门禁 | PASS | `scripts/check.py --benchmark smoke` 全部通过；默认环境 45 passed/2 skipped，可选 TensorCircuit 源码环境 46 passed/1 skipped。 |
| Hamiltonian release 性能 | PASS | reusable native MVP 与 canonical sparse 路径在已对齐 workload 上达到或超过阶段目标。 |
| Algebra/grouping go/no-go 性能证据 | FAIL | 有 native 和 public timing，但没有匹配的 Python/TensorCircuit baseline 与完整 allocation/peak-memory 对照，不能确认所有结构路径达到目标倍数。 |

## CRITICAL

### C1. `MvpPlan::apply_into` 对非零输出缓冲区返回错误结果

证据：并行分支先在局部变量中累加并以 `*output = value` 覆盖写入（`crates/tencir-pauli-core/src/lib.rs:607`），串行分支却直接从调用者原值开始使用 `*output += ...`（`crates/tencir-pauli-core/src/lib.rs:628`）。函数文档承诺“Apply the plan into caller-owned storage”，并未要求输入缓冲区预先清零（`crates/tencir-pauli-core/src/lib.rs:578`）。临时 Rust probe 对算符 `2I`、state `[3, 5]` 和预填充 output `[7, 11]` 调用该 API，应得到 `[6, 10]`，实际得到 `[13, 21]`。

影响：这是 Rust core public API 的静默数值错误，并且结果随 workload 是否越过并行阈值而改变。Python binding 当前用 zero-filled NumPy output，因而掩盖了该缺陷，但独立 Rust 用户和未来 scratch-buffer reuse 会直接受影响。

建议：让串行分支与并行分支都覆盖写入，或在进入 kernel 前统一清零。新增 direct/reusable 两种 strategy、串行/并行阈值两侧、zero/nonzero output buffer 的 Rust regression tests。

## MAJOR

### M1. Python `PauliWord` 构造器没有建立 canonical invariant

证据：public constructor 仅检查整数范围，然后原样保存 `x_words`/`z_words`（`python/tencirpauli/pauli.py:56`）；长度与尾部未使用 bits 只会在后续 native method 中被检查或遮蔽。Rust canonical constructor 会主动 mask 尾部 bits（`crates/tencir-pauli-core/src/lib.rs:221`）。实测 `PauliWord(1, (2,), (0,))` 保存 `x_words=(2,)`，但 `to_codes()` 返回 `I`，同时它与 canonical `I` 的 equality/hash 都不同；`PauliWord(1, (), ())` 也能成功构造，只在之后调用方法时才失败。现有测试甚至明确把 incompatible packed length 的失败延迟到 `.weight`（`tests/test_pauli_word.py:21`）。

影响：相同数学对象可具有不同 equality/hash，且 `__lt__` 使用 canonical codes、dataclass equality 使用原始 fields，破坏有序性与哈希语义一致性；这与 canonical binary symplectic representation 和 fail-fast 要求冲突。

建议：在 `__init__` 中立即验证 packed length 并 mask 尾部 bits，或调用一次真正 canonical 的 packed batch constructor。新增 equality/hash、尾部 bits、零 qubit、错误长度的 constructor-level tests。

### M2. `PauliOperator.scale` 可破坏 canonical 与有限系数 invariant

证据：Rust `scale` clone 后原地相乘并直接返回（`crates/tencir-pauli-core/src/lib.rs:999`），没有删除 exact-zero terms，也没有验证乘法结果仍为 finite。实测 `2X.scale(0)` 返回一个 coefficient 为 `0j` 的 `X` term；`2X.scale(1e308)` 返回 coefficient 为 `inf` 的 term。Python `_from_native` 直接接收该结果，不再 canonicalize（`python/tencirpauli/pauli.py:320`）。

影响：public class 文档声称是 exact-zero-aggregated static operator（`python/tencirpauli/pauli.py:228`），后续算法也假设 coefficients finite、terms nonzero。该 invariant 被 scale 静默破坏。

建议：zero scalar 直接返回 `empty(nqubits)`；其他 scalar 对每个结果做 finite check，并保持 exact-zero removal。新增 zero、underflow-to-zero、overflow-to-inf 和 complex scalar regression tests。

### M3. Python operator 热路径逐 term 跨 PyO3，并反复 canonicalize

证据：`PauliOperator.__init__` 逐 term 调用 `_coerce_word`，code/string 输入经 `PauliWord.from_codes` 进入 native，再立即通过 `word.to_codes()` 第二次进入 native（`python/tencirpauli/pauli.py:241`、`python/tencirpauli/pauli.py:78`、`python/tencirpauli/pauli.py:137`），最后才发起一次 batch canonicalization。已经 canonical 的 operator 每次执行 `_arrays()` 又逐 term 调用 `to_codes()`（`python/tencirpauli/pauli.py:331`），native `build_operator` 随后再次 canonicalize（`crates/tencirpauli-native/src/lib.rs:170`）。这直接违反“Never cross PyO3 once per Pauli term in a hot path”和“eliminate repeated canonicalization”。当前 benchmark 中 100k-term core canonicalization 约 `13.38 ms`，public constructor 约 `226.06 ms`，public boundary 比 core kernel 慢约 17 倍。

影响：Python facade 抵消了 Rust core 的大部分结构处理收益，并让 add/multiply/grouping/Hamiltonian one-shot compile 重复支付 object materialization、FFI 和 canonicalization 成本。

建议：短期先在 Python 中把 string/code 输入直接规范化为一个二维 code buffer，避免为每 term 构造临时 `PauliWord`；`PauliWord` 输入用一次 packed batch conversion。中期让 `PauliOperator` 持有 canonical packed arrays 或 private immutable native handle，并缓存 coefficients；`terms` 只在用户读取时 materialize。PyO3 接口优先接收 contiguous `uint8/uint64/complex128` NumPy buffers，而不是 `Vec<Vec<u8>> + real list + imag list`。

### M4. 多个长时间 native 路径没有释放 GIL

证据：binding 中只有 reusable MVP apply（`crates/tencirpauli-native/src/lib.rs:94`）和 one-shot MVP kernel（`crates/tencirpauli-native/src/lib.rs:495`）调用 `py.allow_threads`。dense、COO、CSR 分别直接执行 core 计算（`crates/tencirpauli-native/src/lib.rs:386`、`crates/tencirpauli-native/src/lib.rs:422`、`crates/tencirpauli-native/src/lib.rs:461`），MVP plan construction 与 grouping 也在持有 GIL 时运行（`crates/tencirpauli-native/src/lib.rs:523`、`crates/tencirpauli-native/src/lib.rs:559`）。Phase 1 checklist 明确把“长计算释放 GIL”标成已完成（`docs/vibe/phase-1-spec.md:103`）。

影响：100 ms 级 sparse construction、large grouping 和 plan construction 会阻塞同一 Python process 的其他线程；文档完成状态与实现不一致。

建议：Python 参数解析和 NumPy object 创建保留在 GIL 内，完成 owned/borrowed Rust input 后，用 `py.allow_threads` 包住 canonicalization、grouping、dense/COO/CSR 和 plan construction。加入一个并发 smoke test或至少对所有 long-running binding 做结构性检查。

### M5. `max_bytes` 不是 end-to-end 峰值内存上限

证据：core 只按 native output/working buffers检查 allocation，例如 dense 只检查 matrix entries（`crates/tencir-pauli-core/src/lib.rs:1073`）。返回 Python 时 `numpy_complex_array` 将整个 `Vec<Complex64>` map/collect 成第二个 complex vector（`crates/tencirpauli-native/src/lib.rs:56`），因此转换期间两个 value buffers 同时存在。reusable plan 在 memory check 前还 clone 并 group terms（`crates/tencir-pauli-core/src/lib.rs:500`）。TensorCircuit adapter 也按 term 构造完整 `2**n` mask，且没有独立 memory guard（`python/tencirpauli/integrations/tensorcircuit.py:52`）。

影响：公开 4 GiB budget 可以在 boundary conversion 中产生接近额外一个完整 complex value buffer，实际峰值明显高于调用者指定值，存在 OOM 风险；这与“估算 complete peak”和 allocation fail-fast 规则不符。

建议：让 core 与 NumPy 共用 `num_complex::Complex64` 或提供 caller-owned output APIs，避免全量 complex copy；memory estimator 纳入 input canonicalization、grouped terms、Rayon scratch、Rust-to-NumPy handoff 和 backend masks。对高预算 case 用 RSS/allocation instrumentation 验证 guard，而不只验证 source-level estimated bytes。

### M6. Grouping 的算法与数据布局存在明显扩展瓶颈

证据：core `group_words` 无 memory-limit 参数，直接分配 `Vec<Vec<bool>>` 的完整 `N×N` adjacency matrix（`crates/tencir-pauli-core/src/lib.rs:715`）；Python 只是以默认 `10_000_000` entries 在 facade 侧拒绝（`python/tencirpauli/grouping.py:75`），约 3162 terms 已触顶。每个 QWC pair 又通过两次 `codes()` 分配完整 code vectors（`crates/tencir-pauli-core/src/lib.rs:347`）。largest-first 在 sort comparator 中反复扫描整行 degree（`crates/tencir-pauli-core/src/lib.rs:741`），DSATUR 每轮重算 saturation，且用线性 `Vec::contains` 去重 colors（`crates/tencir-pauli-core/src/lib.rs:752`、`crates/tencir-pauli-core/src/lib.rs:862`）。

影响：当前 1024-term QWC 约 `22.0 ms`，但内存和算法复杂度会很快主导，无法支持架构文档中更大的结构 workload；Rust core 独立调用还缺少 fail-fast guard。

建议：先把 QWC compatibility 改为 packed bitwise conflict test，消除 pair-wise code allocations；预计算 degree，并增量维护 DSATUR saturation。若目标超过几千 terms，使用 bitset adjacency、bounded edge representation 或 on-demand coloring，并把显式 memory budget下沉到 core public API。

### M7. Algebra/grouping 性能验收证据不足，public boundary 已显示明显损耗

证据：canonicalization 与 grouping 的 Python benchmarks只测 TenCirPauli public path（`benchmarks/python/test_pauli_operator_benchmark.py:24`、`benchmarks/python/test_grouping_benchmark.py:22`），没有语义匹配的 pure-Python/TensorCircuit baseline；当前记录也没有这些路径的 peak RSS/allocation breakdown。另一方面，Hamiltonian workload 已有对齐的 TensorCircuit/JAX 数值、同步和 storage-contract 说明。

影响：可以确认 Hamiltonian 的性能目标达成，但不能据现有证据确认“大批量结构操作至少 5x 或 2x peak-memory 改善”的 go/no-go 指标。把 Phase 1 所有性能都概括为达标会超出证据。

建议：为 canonicalization/dedup、QWC/general compatibility、largest-first/DSATUR 各增加相同输入、相同输出合同的 Python/TensorCircuit baseline；分别报告 input conversion、kernel、output conversion、peak RSS/allocations 和结果一致性。修复 M3 后再做比较，否则主要测到的是 facade 开销。

## MINOR

### N1. `unsafe` layout bridge 有安全注释，但缺少专门的 layout regression/compile-time assertion

证据：两个 slice cast 的 safety argument是清楚的（`crates/tencirpauli-native/src/lib.rs:27`），但 size/alignment 只使用 `debug_assert_eq!`，release build 不检查，测试中也没有专门锁定 layout contract。项目规则要求每个 `unsafe` 有 dedicated tests。

建议：最好直接统一为同一个 `num_complex::Complex64` 类型并删除 cast；若保留自定义 scalar，则加入 compile-time size/alignment assertion 与专门 round-trip test。

### N2. MSRV 已声明但 CI 只跑 stable

证据：workspace 声明 `rust-version = "1.85"`（`Cargo.toml:11`），CI Rust job 使用 `dtolnay/rust-toolchain@stable`（`.github/workflows/ci.yml:12`）。

建议：增加一个只运行 `cargo check --locked` 或核心 tests 的 Rust 1.85 job，避免 dependency update 在未察觉时抬高 MSRV。

## 性能复核

本次在同一台 Darwin arm64 机器、Python 3.11.15、rustc 1.97.1 下运行 `python benchmarks/run.py compare 20260801T104116Z_a872af7f8e5b`。以下数字为当前 HEAD 的 release Criterion/Python median 或 Criterion point estimate；小幅变化需结合置信区间解释。

| Workload | 当前结果 | 相对 clean baseline | 判断 |
| --- | ---: | ---: | --- |
| Rust canonicalize 100k | 13.38 ms | 约 +1.5% | 轻微回退/接近噪声阈值；public path 仍为 226.06 ms。 |
| Rust QWC grouping 1024 | 22.04 ms | 约 +0.9% | 无显著变化。 |
| Reusable MVP apply 10q/64 | 7.50 µs | 约 +1.1% | 稳定。 |
| Reusable MVP apply 16q/256 | 0.441 ms | 约 -3.3% | 稳定到略有改善。 |
| Reusable plan construction 10q/64 | 36.04 µs | 约 +8.7% | 统计上回退，建议复测并 profile。 |
| One-shot MVP 10q/64 | 64.74 µs | 约 +6.1% | 统计上回退；重复 workload 应继续使用 reusable plan。 |
| Rust COO/CSR 10q/64 | 69.25/68.16 µs | 约 -50.6%/-55.1% | 明显改善。 |
| Public COO/CSR 20q/3 terms | 6.36/5.00 ms | 基线 85.15/87.05 ms | 约 13x/17x 改善。 |

保存的同 workload 证据表明 reusable native MVP 在 10q/64 和 16q/256 上约比 TensorCircuit/JAX warm MVP 快 4.0x 和 5.7x（`docs/vibe/implementation-status.md:65`）；aligned canonical sparse end-to-end workload 约快 48–61x（`docs/vibe/implementation-status.md:71`）。20q local Heisenberg MVP 只有约 1.1–1.2x 优势（`docs/vibe/implementation-status.md:70`），所以不应把某个 sparse workload 的 50x 扩大成所有 Hamiltonian/MVP 的普遍结论。

总体性能判断：**Hamiltonian sparse 与 reusable native MVP 已达到 Phase 1 价值证明；operator construction、canonicalization public boundary、grouping scalability 和 backend adapter setup 仍有非常明确的优化空间。**

## Rust 单文件与模块拆分判断

Rust 没有“库就应该写在单个 `lib.rs`”的惯例。小型 crate 用一个文件很常见；`mod` 拆分只是同一个 crate 内的源码组织，不会增加运行时开销、FFI 次数或发布包数量，也不等于拆成更多 crates。

当前单文件形态可以由 Phase 1 的“最小纵向切片、避免 speculative abstraction”和“暂不拆更多 crate”解释，但“不要拆更多 crate”不等于“不要拆 modules”。现在 core `lib.rs` 已有 1674 行，同时包含 scalar/error、PauliWord、operator、grouping、Hamiltonian sparse、MVP 和 tests；native `lib.rs` 也有 652 行。职责边界已经稳定，继续单文件会增加 review 冲突、让 invariant 难以定位，并使 Phase 2/3 扩展更危险。**建议现在拆 module，但仍保持现有两个 crates。**

建议的最小结构如下：

```text
crates/tencir-pauli-core/src/
├── lib.rs              # public re-exports，保持现有 API
├── scalar.rs           # Complex64 / PauliPhase
├── error.rs            # PauliError
├── word.rs             # PauliWord 与 packed helpers
├── operator.rs         # PauliTerm / PauliOperator / canonicalization
├── grouping.rs         # compatibility 与 coloring
└── hamiltonian.rs      # dense/COO/CSR/MVP plans；规模继续增长时再建子模块

crates/tencirpauli-native/src/
├── lib.rs              # #[pymodule] 与注册
├── convert.rs          # NumPy/complex/error bridge
├── word.rs
├── operator.rs
├── grouping.rs
└── hamiltonian.rs
```

拆分原则：只移动代码并用 `pub use` 保持 public paths，不在同一次提交改变算法；随后分别修复 correctness 和 performance。模块内部 helper 默认 private 或 `pub(crate)`，不要为了跨文件方便而扩大 public API。

## RECOMMENDED IMPROVEMENTS

1. **Release blocker correctness patch**：修复 C1、M1、M2，并增加能在修复前失败的 Rust/Python regression tests。
2. **FFI/data-model patch**：消除 M3 的 per-term round trips 和 repeated canonicalization；以 public 100k canonicalization 从约 226 ms 接近 core 13 ms 为直接目标，同时保留端到端 benchmark。
3. **Concurrency/memory patch**：为所有长 native 路径释放 GIL，修复 M5 的 boundary peak-memory accounting，并测试高预算/低预算失败行为。
4. **Grouping patch**：先做 packed QWC predicate、degree cache 和 core memory guard，再决定是否需要 bitset/streaming coloring。
5. **Mechanical module split**：保持两个 crates 和全部 public APIs，不把拆文件与行为修改混在一个 commit。
6. **最终复验**：重跑 `scripts/check.py --benchmark smoke`、可选 TensorCircuit NumPy/JAX tests、完整 release benchmark compare，并新增 matched algebra/grouping baseline 与峰值内存记录。

## REMEDIATION RESULT（2026-08-01）

初次验收的阻断项已完成修复，本地 Phase 1 复验结论更新为 **PASS**。该结论覆盖源码、默认测试环境、只读 TensorCircuit/JAX 集成环境、benchmark smoke 和完整 release baseline compare；尚未替代远端 CI 结果或创建 Git commit。

本次 remediation 的本地 benchmark 记录标签为 `phase1-acceptance-remediation-20260801`；记录保存在被 Git 忽略的 `.benchmarks/` 中，不属于归档文档或发布产物。对照用的修复前 clean baseline 为 `20260801T104116Z_a872af7f8e5b`。

| Finding | 状态 | 修复与证据 |
| --- | --- | --- |
| C1 `apply_into` 非零输出错误 | RESOLVED | 串行与并行路径均覆盖写入；新增 direct/reusable、串行/并行、zero/prefilled output Rust regressions。 |
| M1 非 canonical Python `PauliWord` | RESOLVED | constructor 立即验证 packed 长度并 mask tail bits；equality/hash/ordering/fail-fast tests 已加入。 |
| M2 `scale` 破坏 invariant | RESOLVED | zero/underflow 删除 exact-zero terms，overflow 显式失败；duplicate aggregation overflow 也新增检查。 |
| M3 per-term FFI/re-canonicalization | RESOLVED | Python 输入在单次 batch call 前规范化；operator 缓存 canonical arrays；native 下游使用 validated canonical fast path；新增 contiguous NumPy plan API。 |
| M4 长计算持有 GIL | RESOLVED | canonicalization、operator algebra、grouping、dense/COO/CSR、MVP 和 plan construction 均在 owned/borrowed input 建立后释放 GIL。 |
| M5 end-to-end memory guard | RESOLVED | core/NumPy 统一 `num_complex::Complex64`，删除 unsafe cast 与整块 complex output copy；reusable-plan construction 纳入临时 term/group memory；NumPy/TC backend executors 增加 budget guard。 |
| M6 grouping scalability | RESOLVED | QWC 改为 packed-bit predicate，largest-first degree 预计算，DSATUR 增量维护 neighbor colors，core 增加 bounded grouping API。 |
| M7 matched performance evidence | RESOLVED | 新增同输入/输出合同的 Python tuple/dict canonicalization 与 QWC largest-first baselines。 |
| N1 unsafe layout bridge | RESOLVED | 统一 scalar type后删除两处 unsafe slice cast。 |
| N2 MSRV CI | OPEN / NON-BLOCKING | `rust-version=1.85` 仍只声明未单独在 CI job 验证，保留为后续维护项。 |

最终验证结果：`scripts/check.py --benchmark smoke` 全部通过；Rust core tests `8 passed`；默认 Python 环境 `56 passed, 2 skipped`；只读 TensorCircuit/JAX 环境 `57 passed, 1 skipped`；benchmark harness `48 passed, 36 skipped`。

完整 release compare 相对 clean baseline `20260801T104116Z_a872af7f8e5b`：Rust 100k canonicalization `13.18 ms → 4.23 ms`（约快 68%）；Rust 1024-term QWC `21.84 ms → 1.16 ms`（约快 94.7%）；public 100k friendly-term canonicalization `223.67 ms → 42.83 ms`（约快 5.2x）；public 1024-term QWC `12.95 ms → 0.979 ms`（约快 13.2x）；reusable MVP apply 为 `7.31 µs → 4.94 µs`（10q/64）和 `457.7 µs → 329.0 µs`（16q/256）。新增 matched baselines 中，100k contiguous canonicalization 为约 `45.5 ms` 对 Python tuple/dict `315.0 ms`（约 6.9x），1024-input QWC 为约 `0.979 ms` 对 Python `107.4 ms`（约 110x）。

保留的非阻断性能信号：小型 Rust `hamiltonian/targets/mvp` 比 clean baseline 慢约 3.8%，10q/64 one-shot MVP 慢约 7.3%，而相同 workload 的 reusable apply 快约 32%；重复调用仍应使用 reusable plan，后续只在 profile 证明必要时优化 one-shot setup。
