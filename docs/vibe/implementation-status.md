# Implementation Status

状态日期：2026-08-01。该文件是长时间 Agent 工作的持久状态，不替代 Git history 或规范。

## Current objective

完成 `phase-1-spec.md` 中的 P0–P5。当前 active milestone 是 P2：PauliOperator 与 canonicalization。

## Completed foundation

- 独立 Cargo workspace、pure Rust core crate、PyO3 native crate 和单一 Python package 已建立。
- P0 NumPy dense reference 与固定 regression vectors 已建立；reference 独立使用 I/X/Y/Z matrices、`np.kron` 和显式 local product table，不调用被测实现。
- P0 acceptance gate 已满足：来源、seed、dtype、tolerance 和最大 reference system 已记录在 `docs/vibe/reference-vectors.md`；P0 commit 为 `829221e`。
- P1 Rust `PauliWord` 已完成 code/string/packed conversion、weight、support、symplectic inner product、commutation、exact four-valued multiplication phase、adjoint 和 canonical ordering。
- P1 batch conversion 已通过单次 native call 处理二维 code structures；Python facade 已提供 typed `PauliWord.from_codes`, `from_string`, `batch_from_codes` 和 `PauliProduct`。
- P1 typed error paths 已覆盖 invalid code/shape, incompatible qubit count, packed word length and non-negative nqubit validation。
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
- Rust format：通过。
- Rust Clippy `-D warnings`：通过。
- Rust unit/doc tests：2 passed。
- Python package tests：3 passed。
- Black、Ruff、strict mypy：通过。
- `scripts/check.py --benchmark smoke`：完整通过，包括 Rust/Python benchmark harness。
- P1 benchmark workloads：Rust Criterion 已加入 code round-trip/multiply（6/64/256 qubits），Python pytest-benchmark 已加入 1,024-term batch conversion；smoke harness 全部通过。
- Local benchmark：`p0-829221e` 已在 clean commit 上完成 Rust/Python record；Rust weight kernel 为 1.02 ns (64 qubits)、3.00 ns (1024)、41.52 ns (16384)，commutation 为 2.15 ns、4.97 ns、62.93 ns；Python public-path workload mean 为 174.4 µs。该结果是本机 informational baseline，不构成 CI 门禁。
- Public-file/local-secret audit：通过；`.conda/`、`.benchmarks/`、`AGENTS.local.md`、build artifacts 均被忽略。

## Next actions

1. 在 P1 commit 后记录 clean benchmark label，并手动比较 batch path 结果。
2. 完成 P2：PauliOperator canonicalization、operator algebra 和 structural/dynamic coefficient separation。

## Update protocol

每完成一个有测试证据的纵向切片，更新 active milestone、completed items、精确验证命令/结果、benchmark label、已知限制和下一步。不要把计划写成已完成，不要删除历史 blocker；解决后将其移动到 decision record 并注明日期。若上下文压缩或 Agent 更换，首先阅读本文件，然后从最早未完成 REQUIRED item 恢复。
