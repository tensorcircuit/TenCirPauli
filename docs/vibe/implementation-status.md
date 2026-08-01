# Implementation Status

状态日期：2026-08-01。该文件是长时间 Agent 工作的持久状态，不替代 Git history 或规范。

## Current objective

完成 `phase-1-spec.md` 中的 P0–P5。当前 active milestone 是 P5：Public API、TensorCircuit adapter 与交付；实现、最终 release benchmark、质量检查和本地提交均已完成。

## Completed foundation

- 独立 Cargo workspace、pure Rust core crate、PyO3 native crate 和单一 Python package 已建立。
- P0 NumPy dense reference 与固定 regression vectors 已建立；reference 独立使用 I/X/Y/Z matrices、`np.kron` 和显式 local product table，不调用被测实现。
- P0 acceptance gate 已满足：来源、seed、dtype、tolerance 和最大 reference system 已记录在 `docs/vibe/reference-vectors.md`；P0 commit 为 `829221e`。
- P1 Rust `PauliWord` 已完成 code/string/packed conversion、weight、support、symplectic inner product、commutation、exact four-valued multiplication phase、adjoint 和 canonical ordering。
- P1 batch conversion 已通过单次 native call 处理二维 code structures；Python facade 已提供 typed `PauliWord.from_codes`, `from_string`, `batch_from_codes` 和 `PauliProduct`。
- P1 typed error paths 已覆盖 invalid code/shape, incompatible qubit count, packed word length and non-negative nqubit validation。
- P2 `PauliOperator` 已完成 deterministic canonical terms、complex128 coefficient storage、add、scale、multiply、commutator、anticommutator、adjoint 和 explicit-tolerance Hermiticity validation。
- P2 static canonicalization 只删除 exact-zero aggregated terms；duplicate contributions 按 IEEE bit pattern deterministic reduction，结构与 coefficients 保持分离，未引入 parameter-dependent cutoff。
- P3 Rust core/native grouping 已实现 QWC 与 general symplectic compatibility、largest-first greedy 和 DSATUR；公开 QWC result 提供 canonical membership、basis、coefficient mapping、reconstruction masks 和 `measurement_ready=True`。
- P3 general grouping 使用独立 `GeneralCommutingGroupingResult`，明确 `measurement_ready=False`，不复用 QWC measurement plan。Dense compatibility matrix 与 bounded streaming incompatibility edge-list 两条路径均已提供。
- P4 Rust/PyO3/Python Hamiltonian compiler 已完成 dense、COO、CSR、native matrix-free MVP 和 schema-versioned backend MVP plan；matrix action 明确采用 TensorCircuit qubit-zero-is-MSB ordering，packed plan 保持 qubit-zero-is-LSB 并在 executor 边界转换。
- P2 batch canonicalization 现在提供 `PauliOperator.canonicalize_batch()`，返回 canonical structures、aggregated coefficients、`input_to_canonical` 和 exact `PauliPhase` multipliers；动态结构结果保留 exact-zero keys，静态 `from_terms` 仍使用无 mapping 的 fast path 并删除 exact-zero terms。
- Native NumPy boundary 使用 `complex128` contiguous array 直接映射到 `repr(C)` Rust complex buffers；MVP 输出由 Rust 直接填充 NumPy allocation，避免逐元素 tuple/实部/虚部转换。COO/CSR/dense 也提供 private NumPy-array bindings，public facade 不再从两个 Python float lists 重建 sparse values。
- Reusable native MVP plan 公开 `strategy`：`x_mask_diagonal` 表示已预计算每个 X permutation mask 的 diagonal，`term_direct` 表示由于显式 memory limit 选择逐 term direct kernel；这不是语义 fallback，两个策略共享同一 Rust recurrence 和 differential tests。
- COO/CSR 先按 X mask 聚合 term contributions，再生成按 row 分块的 contiguous entries、exact-zero filter 和稳定 row-major sort；大 workload 使用 Rayon row parallel，CSR 直接从每行计数构造 row pointer、columns 和 values。每个 X mask 只有一个 term 且不存在行内抵消时，COO/CSR 直接写最终数组，跳过候选缓冲区和二次拆分。
- P4 物化 target 与 MVP output 都在分配前估算 dimension/bytes；默认 public limit `DEFAULT_MAX_BYTES` 为 4 GiB，可通过每次调用的 `max_bytes` 显式降低或提高，超限映射为 `MemoryError`，dimension overflow 映射为 `OverflowError`。
- P5 顶层 `tencirpauli` 仅导出 typed public classes/results/targets；private `_native` symbols remain behind Python facades. README、docstrings、typing stub、examples、CHANGELOG 和 CI packaging smoke 已同步。
- P5 TensorCircuit adapter 已迁移到 lazy optional boundary：缺失 `tensorcircuit-ng` 明确失败；backend plan 使用 TensorCircuit backend operations，支持 NumPy/JAX smoke when those optional dependencies are installed。
- Minimal phase-free `PauliWord` weight/commutation 路径已贯通 Rust、PyO3、Python 和 tests；S1 已确认该 phase-free 方向。
- Linux/macOS/Windows correctness/package CI 与 GitHub Release/PyPI workflow 已建立。
- 本地 Criterion + pytest-benchmark 记录/比较基础设施已建立；性能结果不进入 CI 门禁。
- Rustfmt、Clippy、Rust tests、maturin build、Python tests、sdist/wheel smoke 已通过。
- Black/Ruff/strict-mypy quality gate 与完整本地 pre-commit runner 已建立；tracked hook 已在当前 clone 启用。
- Scaffold 初始 Git commit 已建立；后续实现从该公开基线演进。

## Frozen owner decisions

- S1 已确认：`PauliWord` 保持 phase-free，multiplication 返回 canonical word 与精确的四值 `PauliPhase`，operator coefficient 吸收 phase。
- S2 已确认：Phase 1 native coefficient 统一 complex128。
- S3 已确认：完整 QWC measurement；general commuting 先交付 `measurement_ready=false` 的小型 deterministic prototype。
- S4 已确认：Phase 1 backend plan 范围为 schema、NumPy executor 和 TensorCircuit NumPy/JAX differential smoke。

S1–S4 已全部冻结，不再存在 owner 语义阻塞。实现必须遵循 `semantics.md`；任何修改均需新的 owner decision 与迁移测试。

## Verification evidence

- P0 targeted tests：`python -m pytest tests/test_numpy_reference.py`，13 passed。
- P1 targeted tests：`conda run -p .conda pytest tests`，21 passed；Rust unit/doc tests 4 passed。
- P2 targeted/full tests：`conda run -p .conda pytest tests`，25 passed；Rust unit/doc tests 4 passed。
- P3 targeted/full tests：`conda run -p .conda pytest tests`，30 passed；QWC reconstruction、identity, adversarial XX/ZZ graph, deterministic DSATUR and memory-bound matrix/edge paths covered。
- P4 targeted/full tests：`conda run -p .conda pytest tests`，40 passed；dense/COO/CSR/MVP/backend plan 与独立 NumPy reference 全部通过，覆盖 n=0/首尾 qubit/invalid state/memory guard/overflow。
- P5 targeted/full tests：默认项目环境 `conda run -p .conda pytest -q` 为 45 passed, 2 skipped；随后在不修改 TensorCircuit 源码、仅用只读 `PYTHONPATH` 加上本地 optional dependencies 的环境中，NumPy/JAX backend differential cases 均通过，missing-dependency branch 仍明确失败而不 fallback。Rust/Python quality and benchmark smoke all passed。
- Rust format：通过。
- Rust Clippy `-D warnings`：通过。
- Rust unit/doc tests：2 passed。
- Python package tests：3 passed。
- Black、Ruff、strict mypy：通过。
- `scripts/check.py --benchmark smoke`：完整通过，包括 Rust/Python benchmark harness。
- P1 benchmark workloads：Rust Criterion 已加入 code round-trip/multiply（6/64/256 qubits），Python pytest-benchmark 已加入 1,024-term batch conversion；smoke harness 全部通过。
- P2 benchmark workloads：Rust Criterion 与 Python pytest-benchmark 已覆盖 1,000、10,000、100,000-term duplicate-heavy canonicalization；smoke harness 全部通过。历史 baseline 为 `p1-2e0f154`，最新 all-workload label 为 `20260801T104116Z_a872af7f8e5b`。
- P3 benchmark workloads：Rust Criterion 与 Python pytest-benchmark 已加入 QWC grouping（128/1,024 terms）；smoke harness 全部通过。历史 baseline 为 `p2-6b90270`，最新 all-workload label 为 `20260801T104116Z_a872af7f8e5b`。
- P4 benchmark workloads：Rust Criterion 与 Python pytest-benchmark 已加入 dense/COO/MVP/backend-plan construction/apply；smoke harness 全部通过。历史 baseline 为 `p3-acf5c60`，最新 all-workload label 为 `20260801T104116Z_a872af7f8e5b`。
- P5 packaging/integration evidence：`maturin develop --release --locked`、public example tests and optional adapter tests pass; `maturin build --release --locked` produced a macOS abi3 wheel and `maturin sdist` produced a source archive under `/private/tmp` (not tracked). P4 hook benchmark was recorded at commit `9c11117`.
- Local benchmark：`p0-829221e` 已在 clean commit 上完成 Rust/Python record；Rust weight kernel 为 1.02 ns (64 qubits)、3.00 ns (1024)、41.52 ns (16384)，commutation 为 2.15 ns、4.97 ns、62.93 ns；Python public-path workload mean 为 174.4 µs。该结果是本机 informational baseline，不构成 CI 门禁。
- Final clean optimization evidence (`20260801T104116Z_a872af7f8e5b`)：release Rust Criterion measured `hamiltonian/scaling` at approximately 33.8 µs plan construction/59.9 µs one-shot MVP/7.31 µs reusable apply/141.0 µs COO/155.0 µs CSR for 10q/64 terms, and 1.58 ms plan construction/2.35 ms one-shot MVP/457.7 µs reusable apply for 16q/256 terms. Public Python warm reusable native medians were approximately 7.96 µs and 0.415 ms; public sparse COO medians were approximately 0.057/0.198/0.696 ms for 8q/32, 10q/64 and 12q/64。
- Same-workload TensorCircuit evidence in clean label `20260801T104116Z_a872af7f8e5b`：complex128 JAX warm MVP medians were approximately 31.9 µs and 2.35 ms for 10q/64 and 16q/256, versus TenCirPauli reusable native approximately 7.96 µs and 0.415 ms (about 4.0x and 5.7x faster). TensorCircuit NumPy sparse construction medians were approximately 3.50/8.97/19.96 ms on 8q/32, 10q/64 and 12q/64, versus TenCirPauli approximately 0.057/0.198/0.696 ms (about 62x/45x/29x faster). JAX BCOO warm sparse matvec medians were approximately 0.160/0.286 ms for 8q/32 and 10q/64。
- JAX sparse construction is now measured with synchronization inside the timed callable：first raw BCOO construction was approximately 179/170/164 ms; warm raw construction was 0.799/1.342/2.606 ms; first `sum_duplicates()` was 421/449/464 ms; and warm `sum_duplicates()` was 1.429/5.898/21.841 ms for 8q/32, 10q/64 and 12q/64. The previous unsynchronized warm values were invalid enqueue timings and are retired。
- Large 20-qubit evidence in the same clean label uses 64 full-width random terms for MVP and 3 full-width random terms for materialized sparse targets：the historical native MVP plan construction/apply was approximately 0.084/8.06 ms, versus JAX first/warm MVP at 1.192 s/20.12 ms；the historical native COO/CSR was approximately 85.15/87.05 ms, versus JAX raw BCOO first/warm construction at 259.6/17.87 ms and first/warm `sum_duplicates()` at 543.6/208.9 ms. After the row-parallel direct-output optimization, the same native 20q/3-term COO/CSR path measured approximately 5.73/4.06 ms in a targeted release Python benchmark. The 20q/3-term COO values and indices each occupy about 100.7 MB; with the current 4 GiB default, this 20q scale is no longer rejected solely by the default budget, while larger requests still fail fast unless the caller raises `max_bytes`.
- Local Heisenberg benchmark coverage now includes 20q nearest-neighbor (57 terms) and nearest-plus-next-nearest (111 terms) MVP, 16q COO/CSR construction under the default budget, and 20q COO/CSR construction with the default 4 GiB budget. A 20q nearest-neighbor chain measured approximately 6.82 ms MVP, 107.9 ms COO and 103.1 ms CSR; the next-nearest chain measured approximately 13.38 ms MVP, 214.5 ms COO and 199.4 ms CSR. The corresponding explicit output storage was about 352 MB COO/273 MB CSR for nearest-neighbor and 652 MB COO/497 MB CSR for nearest-plus-next-nearest.
- In the same local TensorCircuit 1.8.0/JAX 0.10.0 complex128 environment, 20q Heisenberg MVP measured approximately 6.84/14.21 ms for nearest/next-nearest versus native approximately 5.70/13.37 ms. For 16q sparse construction, JAX raw BCOO measured approximately 12.4/26.8 ms and warm `sum_duplicates()` approximately 207/414 ms for nearest/next-nearest; TenCirPauli canonical COO/CSR measured approximately 5.7/4.8 ms and 9.6/9.0 ms. All JAX values were synchronized inside the timed callable; raw and canonical storage contracts remain distinct.
- Post-optimization full Python benchmark record `20260801T111756Z_96ab8a52ae97-dirty` measured native sparse construction at approximately 42.4/130.6/316.1 µs for 8q/32, 10q/64 and 12q/64 COO workloads, and 7.49/5.77 ms for the 20q/3-term COO/CSR workload. The run passed 32 benchmark tests with 25 optional TensorCircuit/JAX skips; the dirty suffix records that the optimization had not yet been committed when the record was created。
- JAX sparse storage has a different contract：raw `nse` is `8192/65536/262144` (`terms * 2**n`), with `unique_indices=False`; after `sum_duplicates()` the BCOO `nse` is padded to `2048/8192/32768` and `unique_indices=True`, while exact nonzero data counts are `1984/7680/30720` and `|value|>1e-12` counts are `1920/6912/27648`. TenCirPauli canonical COO exact nnz is `1984/7296/29184`, with values plus row/column storage of `63488/233472/933888` bytes; JAX raw BCOO uses `262144/2097152/8388608` bytes and its padded post-`sum_duplicates()` storage uses `65536/262144/1048576` bytes. JAX duplicate entries are numerically valid for basic matvec, but padded `nse`, floating cancellation residuals and noncanonical raw storage must not be compared as if they were the same COO format。
- Profiling evidence：macOS `/usr/bin/sample` on the 16q/256 reusable public workload showed the dominant cost in Rust `MvpPlan::apply_into` and Rayon row-parallel execution；Python/NumPy borrow and allocation bookkeeping was a small boundary component。A single-thread control regressed reusable 16q apply from approximately 0.44 ms to 3.03 ms，confirming that Rayon parallelism is material for this workload。Profile output remains outside the repository。
- Public-file/local-secret audit：通过；`.conda/`、`.benchmarks/`、`AGENTS.local.md`、build artifacts 均被忽略。

## Next actions

1. No Phase 1 REQUIRED work remains. Future work must begin from a new milestone and must not add symmetry, GateTape, propagation, or native-gradient scope here。
2. Keep benchmark artifacts, local environments and machine-specific profile output untracked；the clean label is reproducible through `python benchmarks/run.py compare 20260801T104116Z_a872af7f8e5b` on this machine。

## Phase 1 completion record

- P0–P5 REQUIRED items and acceptance gates are implemented in local commits `829221e`, `2e0f154`, `6b90270`, `acf5c60`, `9c11117`, `c7d18c3`, `ff02ae8`, `4b10598`, `0a546a6`, `dc08949`, `5384191` and `a872af7`。
- Final quality evidence: `python scripts/check.py --fix --benchmark smoke` passed on the final implementation；the clean code commit hook passed full Rust/Python checks, and the optional read-only TensorCircuit environment passed 46 tests with 1 missing-dependency skip。No TensorCircuit source was modified。
- Final clean benchmark label `20260801T104116Z_a872af7f8e5b`：Rust canonicalization 130.89 µs/1k, 1.3095 ms/10k, 13.182 ms/100k；QWC grouping 345.13 µs/128 and 21.837 ms/1024；dense/COO/MVP/backend-plan kernels 9.457/11.328/1.073/56.90 µs/ns；reusable plan construction/apply 33.79/7.31 µs at 10q/64 terms and 1.579/457.7 µs at 16q/256 terms。Python/TensorCircuit full-boundary medians are recorded above；numerical error remained within documented differential tolerances and benchmark results are informational, not CI gates。
- Known limitation is explicit and intentional: the TensorCircuit adapter remains optional and its missing-dependency branch is only exercised when TensorCircuit is absent. The adapter is lazy, explicit, and never silently falls back to NumPy or native execution; the available NumPy/JAX differential smoke was run against the read-only local TensorCircuit source.

## Update protocol

每完成一个有测试证据的纵向切片，更新 active milestone、completed items、精确验证命令/结果、benchmark label、已知限制和下一步。不要把计划写成已完成，不要删除历史 blocker；解决后将其移动到 decision record 并注明日期。若上下文压缩或 Agent 更换，首先阅读本文件，然后从最早未完成 REQUIRED item 恢复。
