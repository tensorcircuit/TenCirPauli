# Phase 6.5 Spec：generic Rust-native matrix-free time evolution

状态：deferred research proposal，尚未冻结，也不是Phase 6完成后的默认下一里程碑，禁止据此直接启动完整实现或标记完成。Owner认为本阶段需要新的数值线性代数、small eigensolver、Bessel和误差控制选择，与当前Pauli/U1专用核心的实现风格和依赖边界存在明显差异；只有出现明确matrix-free evolution workload、matched baseline和dependency/accuracy spike后才重新评估。若未来恢复，本阶段不属于U1-specific circuit能力：算法接受已经存在于Rust中的matrix-free Hamiltonian handle，对full-space `MvpPlan`、restricted `U1MvpPlan`和未来兼容native operators使用同一核心。

> API note: this historical specification predates the breaking Phase 8 API contract; current public names and signatures are defined in [`phase-8-api-coherence-spec.md`](phase-8-api-coherence-spec.md).

## 1. Goal and scope

Phase 6.5 实现完整留在Rust中的exponential action：

```text
psi(t) = exp(-i t H) psi(0)
```

Python只在plan construction、一次evolve/trajectory/expectation request和最终结果返回时跨PyO3。Taylor、Krylov/Lanczos或Chebyshev内部的每次Hamiltonian MVP、AXPY、dot、orthogonalization、error check和scratch reuse都在Rust执行；禁止Python循环逐次调用`plan.apply()`。

本阶段的性能目标是降低matrix-free repeated-MVP workload的Python/FFI overhead并提供可复现的CPU implementation。Rust相对Python loops应有明显结构优势，但相对warm-JIT JAX CPU、SciPy compiled kernels或accelerator execution不预先保证胜出；所有claim由matched release benchmark和accuracy gate决定。

## 2. Confirmed decisions

1. **Generic rather than U1-specific**：算法只依赖dimension和native `apply_into`；U1只是一个operator来源。
2. **Three forward algorithms**：分切片实现scaling-and-Taylor expm-multiply、Hermitian Krylov/Lanczos和Chebyshev expansion。
3. **No time-dependent Hamiltonian**：首版没有Python `H(t)` callback、adaptive ODE、Magnus或coefficient-function DSL。
4. **No automatic Trotter**：Trotter gate schedule由Phase 6 U1 circuit或调用者显式构造；本阶段不把任意Pauli terms自动指数化。
5. **Forward first**：不提供一般Hamiltonian coefficient autodiff或JAX AD。可以提供廉价且数学明确的time derivative。
6. **No JAX integration yet**：custom call、custom VJP、GPU和TensorCircuit backend integration留给后续阶段。
7. **Coarse outputs/reducers**：完整state、selected-time states或Rust内compiled observable reductions可返回；不在每个time step调用Python callback。
8. **Deferred after Phase 6**：Phase 6验收不会自动触发Phase 6.5；不得为了准备本阶段提前给core增加general linear algebra abstraction或新数值依赖。

## 2.1 Re-entry gate

重新启动Phase 6.5前必须先有一份独立owner decision，至少包含：

1. 一个真实forward-only workload，说明为什么现有SciPy、TensorCircuit/JAX或外部solver与native MVP组合不能满足需求。
2. 同机baseline，分离Python-loop/FFI、warm-JIT JAX CPU、SciPy compiled path和native MVP kernel成本。
3. 依赖spike，验证MSRV、wheel平台、license、binary size、small symmetric eigensolver和Bessel accuracy。
4. 明确先实施哪一个algorithm；保留三种候选不等于恢复时必须同时实现全部三种。
5. 更新本spec状态为active并冻结accuracy、error、memory和public API合同。

## 3. Supported operator sources

Python首版只接受已在Rust中编译并具有compatible complex128 semantics的operator handles：

- Full-space `NativeMVPPlan`/core `MvpPlan`。
- Restricted `U1MvpPlan`。
- 由canonical Hermitian `PauliOperator`或strict Hermitian `U1RestrictedOperator`构造的evolution plan。

不接受Python callable、TensorCircuit backend callable、SciPy `LinearOperator` callback或每次MVP跨FFI的对象。Dense/CSR input若未来支持，必须一次复制/验证到native handle并遵循相同`apply_into` contract。

Core可以使用private/internal trait或enum统一：

```rust
trait NativeLinearOperator: Send + Sync {
    fn dimension(&self) -> usize;
    fn apply_into(&self, input: &[Complex64], output: &mut [Complex64])
        -> Result<(), PauliError>;
    fn trace(&self) -> Option<Complex64>;
    fn norm_bound(&self) -> Option<f64>;
}
```

本阶段不因算法复用自动承诺公开、object-safe、用户可实现的Rust trait；优先让现有两个plan共享zero-allocation internal interface。若独立Rust用户确有自定义operator需求，再把trait稳定化。

## 4. Scientific contract

### 4.1 Hamiltonian requirements

Real-time evolution只接受exactly Hermitian Hamiltonian。由 `PauliOperator` 构造时在canonical coefficients上检查Hermiticity；由U1 operator构造时先使用strict sector-preserving restriction，再检查restricted operator Hermitian。不得把leaking Hamiltonian静默替换成`P_k H P_k`进行演化。

Initial state必须为shape `(dimension,)`的finite complex vector。算法是线性的，不强制输入norm为1也不自动normalize；diagnostics报告initial/final norm和drift。`t`必须finite real。`t=0`返回input copy或明确只读等价结果，不执行MVP。

### 4.2 Approximation and errors

输出是指定algorithm/tolerance下的近似，不声称bitwise exact。每个result至少可返回：

```python
@dataclass(frozen=True)
class EvolutionDiagnostics:
    method: str
    mvp_count: int
    accepted_steps: int
    rejected_steps: int
    estimated_error: float
    initial_norm: float
    final_norm: float
    norm_drift: float
    kernel_seconds: float
    estimated_peak_bytes: int
```

若在`max_mvp`、maximum degree/subspace/restarts或memory limit内不能满足requested tolerance，明确失败并附diagnostic context；不能返回未标记的不收敛state。

## 5. Public Python API discussion draft

```python
class TimeEvolutionPlan:
    def __init__(
        self,
        operator: NativeMVPPlan | U1MvpPlan,
        *,
        method: Literal["taylor", "krylov", "chebyshev"],
        atol: float = 1e-12,
        rtol: float = 1e-10,
        max_mvp: int | None = None,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
        options: TaylorOptions | KrylovOptions | ChebyshevOptions | None = None,
    ) -> None: ...

    def evolve(
        self,
        state: Sequence[complex] | np.ndarray,
        time: float,
        *,
        return_diagnostics: bool = False,
    ) -> np.ndarray | EvolvedState: ...

    def evolve_times(
        self,
        state: Sequence[complex] | np.ndarray,
        times: Sequence[float] | np.ndarray,
        *,
        return_diagnostics: bool = False,
    ) -> np.ndarray | EvolvedTrajectory: ...

    def expectations_at_times(
        self,
        state: Sequence[complex] | np.ndarray,
        times: Sequence[float] | np.ndarray,
        observables: Sequence[NativeMVPPlan | U1ProjectedObservablePlan],
    ) -> np.ndarray: ...
```

首版不提供`method="auto"`。调用者显式选择算法，避免在没有跨workload evidence时固化一个脆弱heuristic。实现完成并有profile后可以另提auto-selection proposal。

`evolve_times`语义固定为每行 `exp(-i times[j] H) psi0`，所有时间相对同一个initial state；内部可以按sorted absolute times复用work，但返回顺序必须匹配输入。Duplicate times和negative times合法。完整output shape为`(ntimes, dimension)`并在分配前受`max_bytes`保护。

### 5.1 Rust-native observable reducers

`expectations_at_times`不是Python callback。它接受提前编译且dimension compatible的native observable handles，在每个requested time state上计算`<psi(t)|O_i|psi(t)>`，只返回shape `(ntimes, nobservables)`的small array。它的收益是避免物化/跨FFI返回巨大的trajectory，并支持用户实际只关心observable dynamics的工作流。

这与Phase 3/4 Pauli propagation不重复：Pauli propagation在Heisenberg picture动态传播operator并可做Pauli-weight projection；Phase 6.5在Schrödinger picture对一个state执行matrix-free exponential action，误差来自numerical exponential algorithm。两者应在small exact systems上互相cross-check，但不能共享或混淆误差合同。

首版可以把observable reducer放在最后一个implementation slice；base `evolve()` correctness和performance不能依赖它。Arbitrary user Python callback明确不支持。

## 6. Shared Rust execution design

三个algorithm共享：

- Borrowed immutable operator handle。
- `apply_into(input, output)`，不得每次MVP分配output。
- Reusable contiguous complex scratch vectors。
- Checked dimension/byte/MVP arithmetic。
- GIL-free native loop。
- Deterministic scalar reductions；parallel dot/reduction若改变floating order，必须记录tolerance contract。
- Optional trace shift、norm/spectral metadata和profile counters。

MVP plan内部可能使用Rayon；outer algorithm默认不再对同一MVP建立第二层parallelism。多state/batch evolution不进入首版，避免nested oversubscription。Vector AXPY/dot/norm可以在profile证明后使用Rayon或SIMD，但小dimension保留serial path。

## 7. Scaling-and-Taylor expm-multiply

### 7.1 Algorithm

实现Al-Mohy--Higham风格的scaling-and-truncated-Taylor exponential action。对trace shift `mu = trace(H)/dimension`：

```text
exp(-i t H) psi = exp(-i t mu) exp(-i t (H - mu I)) psi
```

使用conservative operator norm bound选择scaling count `s`和maximum Taylor degree `m`。Rust不是JIT graph，可以在每个scaling step内根据term norm与running sum执行可靠early stop，同时受public `max_degree`和`max_mvp`限制。Fixed schedule options可用于benchmark/reproducibility，但default不能只凭hard-coded degree无error check。

### 7.2 Metadata

Hermitian CSR/destination-major plans可以从row absolute sums得到`infinity norm`，并利用Hermitian `||H||_1 = ||H||_infinity`。Trace从diagonal transitions稳定累加。若operator handle没有可信norm bound，construction要求调用者显式提供bound或拒绝automatic schedule；不能在Python端通过反复MVP动态估计。

### 7.3 Expected profile

Cost约为`m * s`次MVP，working memory为少量state-sized vectors。它是首个实现基线，因为不需要large Krylov basis或small eigensolver，且容易与SciPy/ TensorCircuit `expm_multiply_evol`做matched comparison。

## 8. Hermitian Krylov/Lanczos

### 8.1 Algorithm

使用Hermitian Lanczos构造`m`维Krylov basis，将H投影为real symmetric tridiagonal `T_m`，计算 `exp(-i t T_m) e_1`并回投。必须处理happy breakdown、loss of orthogonality、residual error和long-time restart/step splitting。

Plain three-term recurrence在finite precision下可能丢失orthogonality。首版必须选择并测试一种明确策略：full reorthogonalization、selective reorthogonalization或带可靠breakdown/restart的方案；不能只因为small tests通过就忽略ghost eigenvalues。Krylov basis memory约`O(m * dimension)`并受`max_bytes`先验限制。

### 8.2 Small projected exponential

Projected tridiagonal eigendecomposition需要稳定Rust实现。P0 spike比较最小依赖方案，例如成熟的symmetric eigensolver crate；不得手写未经验证的general eigensolver。新增dependency必须兼容workspace MSRV、wheel platforms和license，并在small dense reference上覆盖clustered/repeated eigenvalues。

### 8.3 Multiple times

同一个initial state和Hamiltonian的Krylov basis可以服务一个受error bound覆盖的time interval；超出interval必须重建或分步，不能无条件复用。Diagnostics记录subspace dimension、restarts、orthogonalization passes和residual estimate。

## 9. Chebyshev expansion

### 9.1 Algorithm

给定可信spectral bounds `[E_min, E_max]`，把H仿射映射到`[-1,1]`，使用

```text
exp(-i t H) psi = exp(-i b t) sum_j (2-delta_j0) (-i)^j J_j(a t) T_j(H_tilde) psi
```

和three-vector Chebyshev recurrence。Working memory为常数个state vectors；MVP次数由series degree决定。Coefficient tail和norm提供停止/误差诊断。

### 9.2 Spectral bounds and Bessel coefficients

调用者可以显式提供bounds；automatic bounds可以使用独立Lanczos estimate加safety margin，但estimate failure或invalid bounds必须明确失败。Bessel `J_j`计算需要稳定且跨平台的Rust实现/依赖；P0必须验证large order/argument、underflow和complex128 accuracy。错误spectral bounds会破坏收敛，因此plan diagnostics必须回报实际使用的bounds和degree。

Chebyshev不是因为API完整就默认优于Taylor/Krylov；只有long-time、known-spectrum或many-time workload benchmark显示收益时才推荐使用。

## 10. Differentiation policy

Phase 6.5首版不提供一般autodiff，也不声称PyO3结果可被JAX追踪。特别排除：

- Hamiltonian Pauli coefficients的gradient。
- Spectral bounds、Taylor schedule或Krylov basis的gradient。
- Arbitrary time-dependent controls。
- Initial-state VJP/JVP public API。
- JAX custom VJP/custom call。

对time-independent H，final state对time的导数具有精确形式：

```text
d psi(t) / dt = -i H psi(t)
```

因此可在P4增加 `evolve_with_time_derivative()`，只需对final state额外执行一次MVP。Hermitian observable `O`的time derivative也可通过state、`H psi`和`O psi`计算。该能力是显式analytic derivative，不扩张为general AD framework。

## 11. Memory and output policy

`max_bytes`至少cheap-estimate：input/output states、all algorithm scratch、Krylov basis和orthogonalization workspace、projected matrices/eigensolver workspace、Chebyshev/Taylor vectors、trajectory output、observable reducers和parallel worker scratch。它仍非exact RSS，但不得遗漏已知的`m * dimension` major storage。

默认API返回single final state。Trajectory是显式large output；observable-only dynamics优先使用native reducers。Plan不得隐藏保留全部历史states。Repeated `evolve()`复用immutable operator metadata和scratch sizing，但每个concurrent call使用独立scratch。

## 12. Correctness tests

每种algorithm至少覆盖：

1. 1x1/2x2 analytic Hamiltonians、diagonal、zero、identity和global trace shift。
2. Random small Hermitian dense matrices对SciPy `expm_multiply`/eigendecomposition。
3. Full-space Pauli `MvpPlan`与U1 restricted `U1MvpPlan` matched cases。
4. Positive/negative/zero time、duplicate/unsorted times和multiple-time outputs。
5. Complex Hermitian hopping、degenerate spectra、near-breakdown Krylov和tight/loose spectral bounds。
6. Norm preservation、time reversal、composition `U(t1)U(t2)`和energy conservation within tolerance。
7. Tolerance monotonicity、max-MVP nonconvergence、invalid bounds/options和memory failures。
8. 1-thread/multi-thread repeatability、concurrent plan calls和FFI coarse-grain instrumentation。
9. `expectations_at_times`对returned-state postprocessing和small exact Heisenberg propagation cross-check。
10. Time derivative对analytic formula和finite differences。

## 13. Benchmarks and performance claims

Release benchmarks必须分别记录operator setup、evolution plan setup、first call、steady call、MVP count/time、vector-kernel time、total runtime、peak/estimated memory、state norm/error和thread count。至少覆盖：

1. Full-space 12-20q local Pauli Hamiltonians，dimension和memory安全时比较native/JAX/SciPy。
2. U1 64/65/128/129/256q k=1/k=2 hopping/XXZ restricted plans。
3. Dimension约8k、32k、128k的matched operator sparsity点。
4. Short/medium/long time和single/many time points。
5. Taylor不同`m*s`、Krylov dimensions/restarts、Chebyshev degrees/bounds。
6. Final-state output与observable-only output，展示避免trajectory FFI的收益。
7. 1-thread和fixed multi-thread，防止MVP/outer algorithm oversubscription。

主要baseline：SciPy `expm_multiply` CPU、TensorCircuit/JAX warm-JIT `expm_multiply_evol`或matched JAX MVP循环、small dense eigendecomposition reference。JAX compile/first run与warm steady分开；GPU不作为native CPU完成门槛。

不设预定speedup倍数。若Rust不快，必须profile确认瓶颈是MVP layout、vector operations、small eigensolver、Bessel、allocation、threading或FFI，并保留可复现benchmark。只实现三个method但没有accuracy/performance evidence不能标记完成。

## 14. Implementation slices

以下切片仅在re-entry gate通过并由owner把本spec从deferred改为active后生效。

### P0：Common operator contract and baselines

- 抽取existing `MvpPlan`/`U1MvpPlan` zero-allocation `apply_into` interface。
- 建立SciPy/dense/TensorCircuit JAX references和benchmark workloads。
- Spike stable symmetric eigensolver与Bessel dependencies。

Acceptance gate：没有per-MVP FFI或allocation；dependencies/MSRV/license/platform选择有记录。

### P1：Taylor expm-multiply

- 实现trace/norm metadata、schedule、scaling、early stop和diagnostics。
- 支持single/multiple times和full/U1 operators。
- 完成small dense与SciPy differential。

Acceptance gate：tolerance/error和memory contracts成立；representative release benchmark完成。

### P2：Krylov/Lanczos

- 实现basis、orthogonalization、projected exponential、residual和restart。
- 增加near-breakdown/degenerate/long-time tests。
- 比较Taylor crossover和many-time reuse。

Acceptance gate：无未标记nonconvergence；basis memory先验受guard。

### P3：Chebyshev

- 实现spectral rescaling、Bessel coefficients、recurrence和tail diagnostics。
- 支持explicit和validated estimated bounds。
- 比较long-time/known-spectrum workloads。

Acceptance gate：错误bounds明确失败或被diagnostics发现；accuracy与norm gate通过。

### P4：Reducers, time derivative and handoff

- 增加native `expectations_at_times`和optional time derivative。
- 完成thread/memory/profile/FFI benchmarks。
- 更新README、typing、CHANGELOG、architecture和implementation status。

Acceptance gate：observable-only path有真实end-to-end用途；全套quality/correctness gates和clean manual benchmark record通过。

## 15. Non-goals

- 不实现time-dependent H、adaptive ODE、Magnus、control schedules或Python callbacks。
- 不实现automatic Trotter/Suzuki decomposition或Pauli-term exponentiation compiler。
- 不实现imaginary-time normalization flow、ground-state solver或non-Hermitian exponential。
- 不实现general autodiff、Hamiltonian coefficient gradients、JAX/TF/PyTorch integration或GPU。
- 不允许算法loop逐MVP跨PyO3。
- 不把Pauli propagation的`max_weight`、frozen-support gradient或SPPS semantics带入state evolution。

## 16. Remaining decisions before freeze

1. Public plan最终命名采用`TimeEvolutionPlan`还是更明确的`NativeTimeEvolutionPlan`。
2. Taylor default tolerance/schedule和是否允许advanced users显式`m/s`。
3. Krylov采用哪种orthogonalization策略及small symmetric eigensolver dependency。
4. Chebyshev采用哪种Bessel implementation和automatic spectral-bound safety margin。
5. `U1ProjectedObservablePlan`是否作为Phase 6公开类型供Phase 6.5 reducer接受。
