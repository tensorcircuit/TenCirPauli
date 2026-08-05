# Phase 8.5 二审与 Deep Audit Round 2 收口复核

复核日期：2026-08-05

复核基线：`aa5bf0e90b5440ff6a4eba6f72c4063b5966ead4`，重点审查 `01e92b1`（`fix: close phase 8.5 second-round review`）与 `aa5bf0e`（`chore: checkpoint current workspace`）对 `phase-8.5-second-round-review-2026-08-04.md` 的 SR1–SR3 和 `audit-report-2026-08-04-round2.md` 的 R2-1–R2-17 的实际闭环情况。

审查视角：优先验证热点路径的数值正确性、高可用性、端到端性能和可复现证据；不追究对主路径无实际影响的精确 allocator accounting、极端错误消息统一或没有 profile 支持的抽象重构。

本次审查没有修改生产代码、测试或 benchmark 源，只新增本报告及其 `docs/vibe/README.md` 索引项。

## 总体结论

当前实现可以继续用于主要科学计算和性能开发，Phase 8.5 的核心 MVP 路径在正确性、内存行为和当前机器上的性能方向上均有较强证据。Spinful Hubbard lazy plan 已缓存组合索引和紧凑 term descriptor，并直接写入 caller-owned output；generic eager charge 已使用 native destination-major CSR；大 CSR 的并行 gather 有明确收益；SR1–SR3 均已真实关闭。

但是，当前文档中“Deep Audit Round 2 的 17 项全部关闭”和“Phase 8.5 完全收口”的表述略早。R2-4 的修复只处理了第一次 estimate overflow，后续仍把 `usize::MAX` 放入 checked arithmetic，宽量子比 weight-projected batch 仍稳定报 `OverflowError`。此外，generic charge lazy plan 每次 apply 仍重新转换 position vectors 并重复 term validation，违反 frozen specification 的一次性 plan compilation 要求；这不是数值错误，但确实是反复执行热点中的可避免分配和 O(T) 前置成本。

推荐状态：**有条件接受（conditional acceptance）**。主 Pauli、packed U1、spinful Hubbard 和 eager CSR 路径可接受；在修复 R2-4、固化 spinful differential coverage、完成 generic lazy plan 的一次性预编译并为当前 clean commit 生成 release record 之前，不应继续宣称整个 Round 2 与 Phase 8.5 已无保留地闭环。

## 验证结果

- `conda run -p ./.conda python scripts/check.py --benchmark smoke` 通过：Rust format、Clippy `-D warnings`、Black、Ruff、strict mypy、`git diff --check`、release maturin build、41 个 Rust tests、331 个 Python tests、10 个 doctests、3 组 Criterion smoke 和 297 个非大型 Python benchmark smoke 全部通过。
- `conda run -p ./.conda mkdocs build --strict` 通过。
- 当前 release build 上的定向热点 benchmark 通过，固定 `RAYON_NUM_THREADS=4`、BLAS/OpenMP 单线程：4x3 spinful lazy `apply_into` median 约 `199.205 ms`；2x4 约 `0.740 ms`；large generic CSR serial/parallel median 约 `674.146/361.375 us`，并行约 `1.87x`；generic non-termwise aggregation约 `65.875 us`。
- R2-4 最小复现仍失败：64 qubits、64 个 rotation、`max_weight=1`、单一 weight-1 observable、`max_bytes=None` 的 `PropagationBatch` 构造报 `OverflowError: integer overflow while estimating propagation batch storage`。该 workload 的投影后 Pauli 空间是多项式规模，不应由未投影的 `2^64` estimate 拒绝。
- 工作树审查开始时 clean，当前分支相对 `origin/main` ahead 6；`01e92b1`、`aa5bf0e` 和 `7d7fbfc..HEAD` 均通过 `git show/diff --check`。

## Phase 8.5 二审闭环矩阵

| 项目 | 结果 | 复核结论 |
| --- | --- | --- |
| SR1 generic eager compilation 释放 GIL | PASS | `NativeChargeMvpPlan::compile_eager` 在 `crates/tencirpauli-native/src/charge_sector.rs:339-362` 将 transition compilation 和 CSR conversion 一并放入 `py.allow_threads`。 |
| SR2 materialization 在 eager build 前预检 | PASS | Generic charge 在 `python/tencirpauli/charge.py:1289-1336` 先扣除 lazy retained bytes 和 target floor，dense 使用精确 target bytes；U1 sibling 在 `python/tencirpauli/symmetry.py:400-448` 使用同样顺序。失败请求不会发布 cache，retry tests 通过。 |
| SR3 generic aggregation benchmark 不再误走 packed U1 | PASS | `benchmarks/python/test_native_mvp_resources_benchmark.py:285-308` 使用 boson spectator 阻止 U1 dispatch，并断言 `strategy == "term_direct"`。 |
| Structured eager/lazy contract note | PARTIAL / DOC ONLY | 不保留无收益 cache 的性能决策正确，但 frozen spec 与返回 metadata 仍不一致，见 MINOR-1。 |

## Deep Audit Round 2 闭环矩阵

| Finding | 结果 | 当前证据 |
| --- | --- | --- |
| R2-1 parallel CSR correctness branch | PASS | `tests/test_native_mvp_resources.py:121-141` 构造超过 `1 << 19` transitions 的 eager plan，并 bitwise 比较 serial/parallel。 |
| R2-2 empty qudit identity canonical form | PASS | `crates/tencir-pauli-core/src/structured.rs:375-385` 将 empty triples 规范化为 `None`；Rust regression test 通过。 |
| R2-3 large-dimension Weyl phase reduction | PASS | `crates/tencir-pauli-core/src/structured.rs:1931-1936` 在 float conversion 前使用 `u128` 取模；范围边界 regression test 通过。 |
| R2-4 projected batch estimate overflow | **FAIL** | `crates/tencir-pauli-core/src/propagation.rs:572-600` 仍在后续 `checked_mul/checked_add` 中合并 `usize::MAX`；稳定复现 `OverflowError`。 |
| R2-5 value-only propagation discarded Jacobian | PASS | `python/tencirpauli/propagation_circuit.py:170-228` 的 value-only terminals 使用 `_native_values`。 |
| R2-6 SPPS native/public parameter-space mismatch | PASS AS DOCUMENTED LIMIT | `python/tencirpauli/spps_circuit.py:103-155` 明确说明 proxy/converged 仍属于 native-angle space。 |
| R2-7 dead charge FFI | PASS | 旧 `crates/tencirpauli-native/src/charge.rs` 和对应 registration/stub 已删除。 |
| R2-8 hybrid mapping validation parity | PASS | `crates/tencir-pauli-core/src/mapping.rs:343-371` 检查 boson blocks 和 qudit triples。 |
| R2-9 U1 pair-map symmetry invariant | PASS | `crates/tencir-pauli-core/src/u1_circuit.rs:1107-1122` 在生成 pair matrix 时 debug-assert symmetry。 |
| R2-10 U1 static payload budget | PASS | `crates/tencir-pauli-core/src/u1_circuit.rs:443-479` 计入 static payload；`tests/test_u1_circuit.py` 有 allocation regression。 |
| R2-11 Clifford1 table duplication | PASS | `crates/tencir-pauli-core/src/propagation.rs:1505-1509` 委托给 in-place authoritative kernel。 |
| R2-12 lowercase Pauli builder input | PASS | `python/tencirpauli/structured.py:2545-2552` 先 uppercase。 |
| R2-13 ndarray state diagnostic | PASS | `python/tencirpauli/propagation.py:602-619` 先进行 type-aware zero-state 判断。 |
| R2-14 eager charge/U1 apply_into budget | PASS | Generic charge 和 U1 native plans 都在写 output 前检查 output bytes，失败时 output 保持不变；focused regression 通过。 |
| R2-15 U1 lazy plan budget test | PASS | `tests/test_native_mvp_resources.py:195-206` 覆盖 `mvp_plan(max_bytes=0)` 和 lazy scratch failure。 |
| R2-16 private stub shapes | PASS | `_native.pyi` 已收紧；strict mypy 通过。 |
| R2-17 dead apply_lazy stub default | MOOT | 对应 dead FFI 已按 R2-7 删除，正确地不再单独修 stub。 |

## CRITICAL

无。未发现会让主要 MVP 路径产生静默错误结果、破坏 phase/ordering、发生 data race 或引入 state-sized worker multiplier 的新问题。

## MAJOR

### MAJOR-1 — R2-4 实际未关闭：宽 weight-projected batch 仍被 estimate overflow 拒绝

位置：`crates/tencir-pauli-core/src/propagation.rs:567-600, 802-818, 838-920`；对应 audit finding R2-4。

当前修改将 `estimate_batch_worker_bytes(...)?` 改为 `.unwrap_or(usize::MAX)`，但 `active_workers * per_worker_bytes` 随后仍得到 `usize::MAX`，再与 nonzero base bytes 做 checked add 时必然 overflow。即使 `max_bytes=None`，构造也在进入真实 weight-projected kernel 前失败。

这不是纯边界问题：weight-projected Heisenberg propagation 正是宽系统的核心用例，`max_weight=1/2` 时可达 Pauli words 数量分别是 `1 + 3n` 和 `1 + 3n + 9*C(n,2)` 量级，远小于未投影 branching estimate。

建议修复：不要继续用 `usize::MAX` 穿过普通 checked arithmetic。对 `max_weight=Some(k)` 计算保守的 projected Pauli-universe bound `sum_{w=0..min(k,n)} C(n,w) * 3^w`，使用 checked/saturating arithmetic，并以它截断 propagated-term upper bound；candidate storage 仍乘真实 maximum branch factor。只有 projected bound 本身也不可表示或确实超过显式 `max_bytes` 时才拒绝。对 `max_weight=None` 保留现有保守失败行为。

必须增加 regression：64q/64 rotations、`max_weight=1` 的 batch 应可构造并与 scalar engine 相等；再加 128q、`max_weight=2` 的 construction-only case，以及显式小 `max_bytes` 仍必须 `MemoryError`。该修复收益高、风险低，建议 fix-now。

### MAJOR-2 — Generic charge lazy 固定 plan 仍在每次 apply 重建 positions 并重复结构校验

位置：`crates/tencirpauli-native/src/charge_sector.rs:223-233`；`crates/tencir-pauli-core/src/charge.rs:1409-1425`；contract：`phase-8.5-spec.md` §6 和 §16。

`NativeChargeMvpPlan` 仍保存 `Vec<u64>` positions。每次 generic lazy `apply/apply_into` 都调用 `positions()` 生成四个新的 `Vec<usize>`，然后遍历所有 terms 重做 code-length、mapped/raw exclusivity、qudit canonical 和 finite coefficient validation。Fast spinful path在这些步骤前返回，因此 4x4 Hubbard 不受影响；generic term-direct repeated execution 则没有完全实现 frozen contract 所要求的“一次转换、一次校验、重复调用只执行 kernel”。

建议修复：在 `compile_mvp` 内将 positions 转成并保留 `usize`，完成 term/layout validation，并把经过验证的 owned layout 与 descriptors 放进 immutable native handle；apply 只保留 state/output/budget validation 和 call-owned scratch。不要为此增加新的 public abstraction，也不需要改变 generic aggregation 算法。

需要 A/B：以当前 `test_generic_charge_aggregation_steady_apply` 为代表 case，再增加一个小 dimension/中等 term count 的 repeated `apply_into` case以放大固定前置成本；记录 runtime 和 native allocation count。若收益仅在很小 case 可见，仍应保留该改动，因为它同时关闭明确 contract gap 且减少每次分配；不应顺带重写 destination aggregation。建议 fix-now，但可与 R2-4 分开提交以便归因。

### MAJOR-3 — Spinful fast path 的广覆盖 differential 是审查时 probe，不是持久 regression gate

位置：`tests/test_charge.py:290-346`、`tests/test_native_mvp_resources.py:104-118`；contract：`phase-8.5-spec.md` §12。

现有提交测试覆盖一个 2-site half-filled Hamiltonian 的 fast/generic/eager comparison，以及一个 16-site cache-budget construction case。二审报告记录的 2–6 sites、全部非零 fillings、高-hole、complex hopping、generic quartic 和 fallback 的广覆盖 differential 是一次性 probe，没有进入当前 tests。当前代码在这些 probes 中曾通过，因此这里不是已知数值 bug；问题是 fast descriptor、rank/unrank 和 sign convention 的后续修改缺少足够持久的回归保护。

建议修复：加入确定性参数化 differential，覆盖 sites 2–6 的 low/high filling、real/complex hopping、density、受支持 generic quartic、zero/cancellation，并分别强制 combination/rank cache enabled 与 combinatorial fallback；对照 mixed-domain spectator 强制走 generic backend。控制每个 fixture 的 dimension，默认 test suite 总增量应保持在秒级。该项是 test-only、风险极低，建议 fix-now，并作为任何 spinful optimization 的前置 gate。

### MAJOR-4 — 当前 clean commit 缺少完整 release record，文档证据锚点早于收口提交

位置：`docs/vibe/implementation-status.md:7,19`、`docs/vibe/phase-8.5-review-2026-08-04.md` post-review table；本机 ignored `.benchmarks/runs/`。

文档声称当前实现已有 complete local release recordings，但本机最新完整 artifacts 仍锚定 `2ca1d72...-dirty`，时间早于 `01e92b1` 和 `aa5bf0e`。本次 current-commit 定向 benchmark 显示没有观察到回退，甚至 4x3 spinful 从文档约 `203 ms` 到约 `199 ms`，large CSR parallel 仍显著快于 serial；因此性能方向可信，但“当前 clean commit 的完整可复现记录”这一证据声明不成立。

建议修复：完成上述代码/test 收口后，在 clean commit、固定线程环境上运行 `python benchmarks/run.py record`，保存 label、commit、accuracy metadata 和完整 Python/Rust matrix；用 `compare` 对比现有 Phase 8.5 baseline。`.benchmarks/` 继续 ignored，不应提交机器结果。4x4 QuSpin A/B 只在 spinful kernel 有生产改动或 4x3 出现显著变化时重跑；否则现有 1.73x/2.16x 单机结果可保留为方向性证据，避免无收益地重复 5–12 GiB 级工作负载。

## MINOR

### MINOR-1 — Structured eager/lazy 的 owner decision 尚未写回 frozen specification

位置：`docs/vibe/phase-8.5-spec.md` §5.2；`tests/test_native_mvp_resources.py:220-229`；`phase-8.5-second-round-review-2026-08-04.md` 的 structured disposition。

拒绝无实测收益的 stride cache 是正确的，不应为了 metadata 差异制造无意义分配。但 spec 仍写着只有 retained data 不同时才报告 distinct storage metadata，而实现返回 `storage="eager"`、相同 strategy 和相同 estimated bytes。收口报告称这是明确 owner decision，却没有修改 frozen contract。

建议只做文档决策：明确 structured 的 `storage="eager"` 在当前版本是 accepted alias，保留请求 metadata 但共享 profile-selected compact representation；或者将返回 metadata 规范化为 lazy。前者兼容性更好，推荐采用。同时把 `test_structured_eager_retains_a_real_bounded_cache` 改成准确描述 profile-selected compact representation 的名称。不要重新引入已证实无收益的 cache。

## OBSERVATIONS

### 热点性能仍有清晰的下一步，但应先 A/B

当前 spinful descriptor 在 `crates/tencir-pauli-core/src/charge.rs:632-648` 仍保存通用 mode lists，inner loop 在 `:676-723` 对每个 state、每个 term 进入 enum match，hopping 又在 `:872-935` 动态构造 occupation/parity 操作。Hubbard workload 的 term family 很固定，这里是最值得继续优化的单核热点。

优先 A/B 方案是把 diagonal term 预编译为 occupied bit mask，把 quadratic hopping 预编译为 required-occupied/required-empty mask、flip mask 和 parity mask，并把 diagonal/hopping/generic 分开存放以移除内层 enum dispatch；generic quartic 保留已有 validated fallback。该方案只增加 O(T) 紧凑 metadata，不引入 graph 或 state-sized scratch，符合 Phase 8.5 trade-off。先在 2x4 与 4x3 做至少 5 个 process-level repeats；若 4x3 中位数改善明显超过噪声且 differential 全过，再运行一次 4x4 matched A/B。

第二优先级才是 spinful destination-major Rayon gather。它可以让 worker 拥有互不重叠的 output rows，不需要 source-parallel state-sized buffers，但实现和 sign/order验证成本高于 descriptor mask 优化。只有在 mask optimization 后 profile 仍显示 kernel CPU-bound，且 4x3 serial baseline 足够大时再做；必须保留 serial/parallel differential 和固定线程 scaling record。

Generic eager CSR 当前不建议继续改：当前提交实测 parallel median 约 `361 us` 对 serial `674 us`，thresholded row parallelism 已有约 `1.87x` 收益，且 correctness branch 已进入默认 tests。除非新 profile 显示 materialization copy 成为真实瓶颈，不要扩展 public tuning knobs 或添加另一种 CSR storage。

## 推荐执行顺序

1. 修复 R2-4，并加入宽 weight-projected batch regression。
2. 将 spinful 广覆盖 probes 固化为默认 differential tests。
3. 把 generic lazy positions/term validation 移到 plan construction，运行 scoped A/B。
4. 更新 structured storage 的 frozen contract，不实现无收益 cache。
5. 生成当前 clean commit 的完整 release record；确认 2x4、4x3、generic aggregation、large CSR 和 packed U1 无回退。
6. 只有在 correctness gate 与 clean baseline 固定后，实施 spinful bit-mask descriptor A/B；destination-major parallelism保持 profile-gated。

## 最终验收建议

正确性方面，当前主要 scientific outputs 有充分的 dense/generic/eager differential、canonical/Weyl boundary tests 和完整 quality gate 支撑，未发现需要停止使用主 MVP 路径的 blocker。R2-4 是明确的 availability bug而非静默数值错误，spinful 的问题是持久覆盖不足而不是已复现错误。

性能方面，当前版本的 4x3 spinful 与 large CSR 定向 release 数据维持或优于已有记录，4x4 对 QuSpin 的单机优势也与实现结构一致；但在 clean commit 全量 record 完成前，应表述为“当前机器上的正向证据”，不应升级为普适保证。

完成推荐顺序中的 1–5 后，可以把 Phase 8.5 与 Deep Audit Round 2 标记为无保留关闭。第 6 项属于下一轮性能提升，不是本轮 acceptance blocker。

## Post-review remediation status

更新日期：2026-08-05。以下记录是审查后的 working-tree remediation，不改写上述基线结论；形成 clean commit 后仍需补全 release record。

| Finding | 状态 | Remediation evidence |
| --- | --- | --- |
| MAJOR-1 / R2-4 projected batch estimate | CLOSED IN WORKING TREE | Worker estimate 以 `sum_{w=0..min(k,n)} C(n,w) 3^w` 截断投影后 term growth，并保留真实 maximum branch factor。Regression 覆盖 64q/64 rotations/`max_weight=1` 的 batch-scalar equality、128q/`max_weight=2` construction 和显式小预算 `MemoryError`。 |
| MAJOR-2 generic lazy repeated preparation | CLOSED IN WORKING TREE | Native plan 现在持有一次转换、一次校验的 prepared layout；apply 不再重建四个 position vectors 或逐 term 重做结构校验，retained layout bytes 纳入 `estimated_bytes` 和 construction budget。对 `aa5bf0e` 的同机 fixed-thread release A/B：新增 63-term/dimension-20 `apply_into` median 约 `32.2 -> 29.6 us`，原 aggregation case 约 `66.7 -> 65.4 us`。 |
| MAJOR-3 spinful durable differential | CLOSED IN WORKING TREE | 默认 tests 覆盖 sites 2–6 的全部非零 filling、low/high-hole、complex hopping、density、generic quartic、zero/cancellation、fast/generic/eager 以及 caller-owned output；21-site one-particle-per-spin case 固化 combinatorial rank fallback。 |
| MINOR-1 structured eager alias contract | CLOSED IN WORKING TREE | Frozen spec 明确 `storage="eager"` 可作为 profile-selected compact representation 的兼容性 alias，`strategy`/`estimated_bytes` 可与 lazy 相同；regression 名称同步为该 contract。 |
| MAJOR-4 current clean release record | PENDING CLEAN COMMIT | Working tree 已通过 `python scripts/check.py --benchmark smoke`：41 Rust tests、352 Python tests、10 doctests、3 组 Criterion smoke、298 个非大型 Python benchmark smoke；`mkdocs build --strict` 通过。完整 `benchmarks/run.py record` 与 compare 必须锚定后续 clean remediation commit，ignored 机器结果不提交。 |

因此当前代码、测试和文档整改已完成，但 acceptance 状态仍保持 conditional，唯一剩余项是 clean-commit release record；不得用 dirty working-tree timing 代替该证据。
