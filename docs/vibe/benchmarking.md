# 本地 Benchmark 方案

状态：已采用的初始方案。

## 决策

TenCirPauli 不在共享 CI runner 上设置性能门禁，也不因 benchmark 波动阻止合并。性能测量在开发者本机手动运行；仓库提交稳定、可复现的 workload 和运行工具，本机结果保存在被 Git 忽略的 `.benchmarks/` 中。

Rust core 使用 Criterion 进行统计微基准，当前覆盖 bit-packed `PauliWord` 的 weight 与 commutation。Python 层使用 pytest-benchmark 测量公开 wrapper、PyO3 转换、FFI 和小型端到端 workload。后续新增 canonicalization、grouping、Hamiltonian、symmetry 和 propagation 时，应分别增加 Rust kernel 与公开 Python API 两层 benchmark。

## 记录模型

`python benchmarks/run.py record` 为每次运行生成唯一 label，并保存 Git commit、dirty 状态、UTC 时间、平台与工具版本。Criterion baseline、pytest-benchmark JSON 和 manifest 使用同一 label；`python benchmarks/run.py compare <label>` 在当前代码上重新测量并分别输出 Rust 与 Python 的历史对比。

提交 hook 默认调用 `python scripts/check.py --benchmark smoke`：Rust benchmark 只执行 `cargo bench -- --test` 的 harness/build 检查，Python benchmark 排除 `performance_large` 标记，因此不会在每次 commit 中重复完整 release measurement。完整 release record 是显式的性能检查，使用 `python scripts/check.py --benchmark record`；若希望某一次 commit 同时执行，可使用 `TENCIRPAULI_PRE_COMMIT_BENCHMARK=record git commit ...`。`TENCIRPAULI_PRE_COMMIT_BENCHMARK=skip` 或 `scripts/check.py --benchmark skip` 仅跳过 benchmark，不跳过格式、lint、typing、Rust/Python tests 和 release extension build；hook 只接受 `smoke`、`record`、`skip` 三种值。

结果默认不提交，因为不同机器、温度、功耗模式和后台负载不能形成可移植基线。需要长期保留时，应备份整个 `.benchmarks/`，并始终在相同机器上比较。性能结论必须结合统计区间、绝对时间、吞吐量、峰值内存和等价数值精度，不能只看单次百分比。

## Workload 规则

- Benchmark 名称、输入生成器、随机种子、规模与语义一经建立应保持稳定；语义变化时新增 case。
- 正确性验证放在 timed loop 之外，但每个 benchmark 必须验证结果没有被优化器消除或算法改动破坏。
- Rust 使用 release benchmark profile；Python benchmark 前重新执行 release 模式的 maturin develop。
- 分开测量纯 Rust kernel、Python/FFI 路径和端到端 workload，避免把跨层收益错误归因于单个 kernel。
- 对并行算法分别记录单线程与固定线程数 scaling；不要在未知线程池配置下比较结果。
- U1Circuit 的 `40q-k5` compressed sector（`C(40,5)=658008`）默认只跑 native benchmark；TensorCircuit JAX JIT 的首次编译必须显式设置 `TENCIRPAULI_ALLOW_HEAVY_JAX=1`，因为同机实测 peak RSS 约 3.4 GB，不能把 JAX 编译内存误当作 10 MB state storage。
- Sparse COO 对照必须拆分 TensorCircuit/JAX BCOO 的 first construction（含 shape-specialized compile）、warm raw construction、first/warm `sum_duplicates()` 和 warm matvec；同时报告 raw/padded `nse`、实际 data count、unique/sorted flags 与 values/indices storage，不能把 duplicate BCOO 当作 TenCirPauli canonical COO。
- 任何宣称两个公开 sparse 接口性能可比的结果，都必须另设 canonical end-to-end workload：从 Python 公开调用开始，计时到两侧 sparse 对象及其 data/indices 完成；如果目标语义要求 unique/sorted/aggregated entries，则两侧都必须在计时内完成相应 canonicalization。TensorCircuit `PauliStringSum2COO()` 的 raw BCOO 不自动满足这一合同，因此 benchmark 必须显式调用并同步 `sum_duplicates()`，不能把单独的 canonicalization 时间冒充完整接口时间，也不能把 raw-only 时间冒充 canonical 结果。
- JAX 异步 backend 的 warm benchmark 必须在 timed callable 内对结果及 sparse `data`/`indices` 调用 `block_until_ready()`；只在 timed loop 外同步会把 enqueue latency 错当执行时间。
