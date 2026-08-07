# Local Benchmarks

TenCirPauli benchmarks are intentionally local and informational. They do not run as CI performance gates. The committed benchmark code defines stable workloads, while machine-specific measurements live under the ignored `.benchmarks/` directory.

The standard automated benchmark suite lives under `benchmarks/python/`. The algebra suites include eager Pauli BCH, native-backed lazy Pauli BCH, native Fermion/Boson BCH, plain export, explicit Python term materialization, and eager/materialized Structured BCH as separate cases. The handle-boundary suite records flat construction, handle-native mapping and conversion, and dense/COO/CSR/native-MVP terminal compilation with input/output term metadata. Scripts under `benchmarks/manual/` are opt-in workloads that may require an external TensorCircuit checkout, a release build, or a deliberately chosen large input; they are not part of the default pytest collection or commit smoke check.

The manual U1 execution A/B cases are defined in `benchmarks/manual/u1_execution_ab.py`. Run `conda run -p .conda python benchmarks/manual/u1_execution_ab.py --output /private/tmp/u1_execution_ab.json` after a release extension build; it covers repeated same-pair blocks, diagonal-heavy static runs, adjoint gradients, facade cache versus stateless terminals, and grouped projected observables. Keep before/after outputs on the same machine and compare numerical metadata alongside medians.

The actual-angle circuit boundary benchmark is `benchmarks/manual/circuit_differentiation_ab.py`. It records construction, private plan creation, public forward expectation, public occurrence-space native value-and-gradient, the private native endpoint, first execution, synchronized warm JAX `jit(value_and_grad)` for Propagation/U1/SPPS, and an applicable TensorCircuit/JAX circuit baseline. JAX timing synchronizes every value-and-gradient leaf; SPPS records observable terms, angle count, sample budget, gradient workspace, callback bytes, and process peak RSS. Its forward tape uses zero runtime parameters while its gradient tape uses one private slot per angle occurrence, making the performance boundary explicit.

The formal U1 parameterized-circuit cases are in `benchmarks/python/test_u1_circuit_benchmark.py`. The 16-qubit, four-particle, three-layer workload records native and JAX first/steady forward expectation plus value-and-gradient timings, and checks every result against the native numerical oracle. Native forward calls construct a fresh circuit per timed invocation so facade state caching cannot be mistaken for execution throughput.

The repeated-MVP decision benchmark is `benchmarks/manual/mvp_scratch_ab.py`. It covers conserved and aggregate-cancelled generic charge plans, low/medium/wide U1-lazy sectors, and finite fermion/boson/hybrid structured plans. Each case records plan construction, first and steady caller-owned `apply_into`, allocating `apply`, first-versus-steady ratio, four concurrent independent calls, output/plan bytes, numerical error, and retained scratch bytes. The current evidence keeps retained scratch at zero and defers scratch reuse until a measured representative hotspot clears the roughly 10% owner threshold.

## Setup

Install the benchmark extra in the active development environment, then make sure Rust, Cargo, and maturin come from the same selected toolchain:

```bash
python -m pip install -e ".[benchmark]"
```

## Record a run

Record both the Rust Criterion microbenchmarks and Python/PyO3 integration benchmarks:

```bash
python benchmarks/run.py record
```

The generated label combines UTC time, the current Git commit, and a `dirty` suffix when the worktree is not clean. Use `--label NAME` to choose a stable name and `--suite rust` or `--suite python` to run only one layer.

The runner stores Criterion baselines, pytest-benchmark JSON, and a metadata manifest containing the commit, dirty state, platform, and tool versions. It deliberately omits usernames, hostnames, and absolute project paths.

The repository pre-commit hook runs `python scripts/check.py --benchmark smoke` by default, after formatting, linting, typing, correctness tests, and doctests pass. Set `TENCIRPAULI_PRE_COMMIT_BENCHMARK=record` when a full benchmark record is wanted, or use `python scripts/check.py --benchmark skip` when benchmarking is intentionally handled separately.

当前面向科学计算用户的性能覆盖包括：创建和检查 Pauli 项（权重、支集、对易性、乘法）；把大量 Pauli 项合并成确定性的 Pauli Hamiltonian；把 Hamiltonian 生成 dense、COO 或 CSR 矩阵；不生成矩阵而直接计算 `H|ψ⟩`；重复使用同一个 Hamiltonian 的 native MVP plan；把 Pauli 项分成可共同测量的 QWC groups；Z2 analysis/tapering setup；U1 restriction setup、restricted MVP steady apply 和 CSR materialization；以及在 TensorCircuit/JAX 中执行 Hamiltonian MVP 和 sparse matrix。当前还覆盖了 20-qubit 的随机与局域 Heisenberg Hamiltonian/MVP、26-qubit full-space native MVP（约 1 GiB complex128 statevector）、26-qubit `k=2` restricted MVP、16 GiB 默认预算下的显式 materialization，以及真正超过默认预算时的明确拒绝。deterministic propagation 还覆盖 GateTape setup、Rust scalar first/steady expectation、operator materialization、100-qubit Clifford/near-Clifford tape、128-qubit/12-layer native-only near-Clifford scaling、2D Heisenberg rotations、custom PTM 和同步的 matched complex128 JAX warm reference。general-commuting 分组、完整 operator algebra 吞吐、backend plan 的实际执行和更大规模 propagation 仍是后续 benchmark 候选，不把它们写成已经有性能结论的功能。
当前面向科学计算用户的性能覆盖包括：创建和检查 Pauli 项（权重、支集、对易性、乘法）；把大量 Pauli 项合并成确定性的 Pauli Hamiltonian；把 Hamiltonian 生成 dense、COO 或 CSR 矩阵；不生成矩阵而直接计算 `H|ψ⟩`；重复使用同一个 Hamiltonian 的 native MVP plan；把 Pauli 项分成可共同测量的 QWC groups；Z2 analysis/tapering setup；U1 restriction setup、restricted MVP steady apply 和 CSR materialization；以及在 TensorCircuit/JAX 中执行 Hamiltonian MVP 和 sparse matrix。当前还覆盖了 20-qubit 的随机与局域 Heisenberg Hamiltonian/MVP、26-qubit full-space native MVP（约 1 GiB complex128 statevector）、26-qubit `k=2` restricted MVP、16 GiB 默认预算下的显式 materialization，以及真正超过默认预算时的明确拒绝。deterministic propagation 还覆盖 GateTape setup、Rust scalar first/steady expectation、operator materialization、100-qubit Clifford/near-Clifford tape、128-qubit/12-layer native-only near-Clifford scaling、2D Heisenberg rotations、custom PTM 和同步的 matched complex128 JAX warm reference。stochastic propagation 新增 deterministic value-and-gradient 的 checkpoint/gradient-length cases、SPPS 的 12q/100q fixed-budget、adaptive、zero-factor 与 sample-budget scaling cases，并在可选 TensorCircuit 环境中从 Python 端对照 `examples/spps_pauli_path_vqe.py` 和 `PauliPropagationEngine` + JAX autodiff。general-commuting 分组、完整 operator algebra 吞吐、backend plan 的实际执行和更大规模 propagation 仍是后续 benchmark 候选，不把它们写成已经有性能结论的功能。
新增的 symmetry-aware JAX benchmark 使用相同 fixed-Hamming-weight transition table 和相同 tapered Pauli semantics，分别报告 setup、JIT steady apply、first-call/end-to-end 和 Rust native 结果；它不把对称性降维后的 Rust 结果与 full-space JAX 结果做不公平比较。

structured benchmarks also cover direct uniform-Weyl qudit Hamiltonians at dimensions 3, 5, and 7, timing COO, CSR, and native MVP construction with dense reconstruction or MVP differential checks. The same suite now measures Jordan–Wigner, parity, and Bravyi–Kitaev fermion mappings separately for COO, CSR, and native MVP compilation on a bounded Hubbard workload; each mapping is validated against its own dense encoded-basis result.

stochastic propagation 大规模 matched runner 位于 `benchmarks/manual/large_tensorcircuit_compare.py`，默认运行 12q/16q 的 TensorCircuit/JAX 与 native 对照；可用 `--only deterministic --case 20,3,256`、`--only deterministic --case 24,4,128` 和 `--only deterministic --case 32,3,128` 扩展 deterministic locality-3 workload，用 `--native-only --only spps --case 20,3,1024` 测量更高 SPPS budget，并用 `--only spps --case 20,1,64` 测量 TensorCircuit 示例仍可完成编译的较大 SPPS case。输出同时报告 first-call（包含 JAX compile）和 steady median；SPPS 的 Rust seed 与 TensorCircuit 外部 uniforms 不同，因此随机 value 不作逐样本相等断言。`test_propagation_engines_benchmark.py` 另外保留代表性的 `max_weight=2/4` 和 rotation-heavy SPPS cases，不展开完整参数笛卡尔积。

symmetry 的 Rust Criterion suite 同时覆盖 Z2 analysis/taper transform、U1 restriction setup、低粒子数与 central-sector restricted MVP，以及 CSR materialization；Python suite 记录相同 workload 的 public/FFI boundary、restricted output storage 和 central-sector scaling。运行前可显式设置 `RAYON_NUM_THREADS=1` 或目标线程数，runner 会把该配置写入本地 manifest；JAX setup 会在计时 callable 内同步 device arrays。

## Compare with an earlier run

List available local records and compare the current checkout with one baseline:

```bash
python benchmarks/run.py list
python benchmarks/run.py compare <baseline-label>
```

Criterion prints statistical change estimates and writes its HTML report below `.benchmarks/rust-target/criterion/`. Pytest-benchmark prints the current-versus-baseline table using the JSON stored below `.benchmarks/python/`.

## Measurement discipline

- Compare runs on the same machine, power mode, toolchain, feature set, and thread configuration.
- Close competing CPU-heavy processes and allow the machine to reach a stable thermal state.
- Treat a single small delta as noise. Repeat suspicious results and inspect confidence intervals, absolute time, throughput, and memory together.
- Keep benchmark names and workload semantics stable. Add a new case instead of silently changing an old baseline.
- Validate outputs outside the timed region and use release builds for all performance claims.
- Do not commit `.benchmarks/`; archive it separately if long-term machine history matters.
