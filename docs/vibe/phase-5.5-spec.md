# Phase 5.5 Spec：多 Observable deterministic Pauli propagation

状态：已实现 P0–P4，作为 Phase 5 与 Phase 6 之间的可选增量阶段完成。实现只做 observable-level parallelism：每个 observable 完整复用现有 deterministic propagation/value-and-gradient 内核，不做跨 observable coefficient batching、共享 aggregation map 或新的传播算法。

## 1. 背景与现有能力

当前 `PropagationEngine` 已支持一个 canonical `PauliOperator`，因此一个 observable 可以是任意多个 Pauli strings 的加权和。`expectation()` 返回这个和的一个 scalar expectation；`value_and_grad()` 返回这个和的一个 scalar value 与 shape 为 `(nparameters,)` 的 frozen-support reverse gradient。传播过程中，同一 observable 内的 Pauli contributions 会按既有确定顺序聚合、执行 exact-zero removal，并在有限 `max_weight` 下执行 weight projection。

这已经覆盖 Hamiltonian expectation、单目标 VQE objective 和“先把 Pauli strings 求和，再求总梯度”的主要场景。Phase 5.5 不替代或重定义这条路径。

尚未支持的是多个相互独立 observable 的 vector-valued execution。例如输入 `B` 个 Pauli strings 或 `B` 个 `PauliOperator`，用户需要保留每个 observable 的身份，并得到：

```text
values.shape    == (B,)
gradients.shape == (B, nparameters)
```

当前只能构造 `B` 个 `PropagationEngine` 并逐个调用，或由 Python 调用者自行建立线程池。这样会重复 tape/native-handle setup、parameter validation 和 FFI 调用，也没有库内统一的并行阈值、内存估算与 deterministic batch result contract。

## 2. 核心技术决策

1. **现有 Pauli-sum 路径保持不变**：一个 `PauliOperator` 仍表示一个 observable；其 sum expectation 与 sum gradient 继续由 `PropagationEngine` 计算。
2. **Batch 表示多个独立目标，不表示一个和**：batch 第 `i` 行必须等于为第 `i` 个 observable 单独构造现有 engine 的结果。不同 observable 不互相聚合、抵消或共享 frozen support。
3. **只做 observable-level parallelism**：每个 observable 使用独立 current/next sparse terms、aggregation、checkpoints、adjoints 和 gradient buffer。Rayon 只在 observable 维度调度。
4. **共享 immutable compiled program**：GateTape operations、custom PTM transitions、parameter-slot metadata、product-state descriptor、`max_weight` 和静态配置通过 `Arc` 或等价 immutable storage 共享；不得为每个 observable 深拷贝完整 schedule。
5. **小 workload 自动串行**：batch size 或预测工作量不足时直接按输入顺序串行运行现有内核，避免 Rayon 调度成本。阈值是 private heuristic，由 release benchmark 决定，不成为 public API。
6. **不承诺所有 batch 加速**：性能目标是降低足够大、足够重的独立 observable batch 的端到端 latency，并提高 throughput；batch size 1 和轻量 Pauli string 可以与单 engine 持平或略有固定开销。
7. **不做 nested parallelism**：Phase 5.5 内每个 deterministic observable kernel保持现有串行执行。若未来单 observable 内部增加并行，必须重新设计避免与 observable-level Rayon oversubscription。

## 3. 科学与梯度语义

### 3.1 Sum 与 batch 的区别

设输入 observables 为 `O_0, ..., O_(B-1)`。Batch 返回向量

```text
v_i(theta) = expectation(propagate(O_i, theta))
J[i, p]    = frozen_support_d(v_i) / d(theta_p)
```

现有 sum engine 对 `O = sum_i c_i O_i` 返回一个 value 和一行 gradient。Batch 则保留全部 `v_i` 和 `J[i, :]`。如果用户只需要加权总目标，继续使用一个 `PauliOperator` 的现有 engine，避免物化 `B * nparameters` Jacobian。

Phase 4 frozen-support reverse在 trigonometric zero、exact aggregate cancellation、underflow-to-zero和其他 support-change point不等同于固定完整 basis AD。由于 sum engine 在整个 sum 聚合后冻结 support，而 batch 对每个 observable 独立冻结 support，本阶段不承诺“先 batch 再线性组合 gradients”在这些边界点与“先构造 Pauli sum 再调用单 engine”bitwise或数学等价。两条路径各自以现有单-engine合同为准。

### 3.2 Row-wise oracle

对每个合法 batch index `i`：

```text
batch.expectations(params)[i]
    == PropagationEngine(tape, observables[i], ...).expectation(params)

batch.values_and_gradients(params).values[i]
    == PropagationEngine(tape, observables[i], ...).value_and_grad(params).value

batch.values_and_gradients(params).gradients[i, :]
    == PropagationEngine(tape, observables[i], ...).value_and_grad(params).gradient
```

相等性沿用单 engine 当前可提供的确定性/bitwise合同。不同 Rayon worker 的完成顺序不得改变输入顺序、单 observable reduction order 或输出 bits。

### 3.3 Observable 合法性

`expectations()` 与 `values_and_gradients()` 只接受与当前 scalar API 相同的 exactly Hermitian observable。Batch 中任意一个 observable 非法、qubit 数不匹配或执行产生非有限结果时，整个 call 明确失败，不返回部分成功结果。

一般 complex `PauliOperator` 的批量 propagated-operator materialization不属于首个 Phase 5.5 slice；现有单 engine `propagate_operator()` 保持可用。

## 4. 公共 Python API

建议新增独立类型，避免扩大现有 `PropagationEngine` 构造函数的输入联合类型：

```python
@dataclass(frozen=True)
class PropagationBatchValueAndGradient:
    values: np.ndarray      # float64, shape (B,), C-contiguous, read-only
    gradients: np.ndarray   # float64, shape (B, P), C-contiguous, read-only


class PropagationBatch:
    def __init__(
        self,
        tape: GateTape,
        observables: Sequence[PauliOperator],
        *,
        initial_state: ZeroState | ComputationalBasisState | ProductBlochState | str = "zero",
        max_weight: int | None = None,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
    ) -> None: ...

    def expectations(self, parameters: Sequence[float] | np.ndarray) -> np.ndarray: ...

    def values_and_gradients(
        self,
        parameters: Sequence[float] | np.ndarray,
        *,
        checkpoint_interval: int | None = None,
    ) -> PropagationBatchValueAndGradient: ...
```

一个 Pauli string batch 使用单项 `PauliOperator` 作为每个元素。首版不增加混合 `PauliWord | PauliOperator | str` 构造输入；若真实 workload 表明 Python 单项 operator 构造成为端到端瓶颈，再增加独立的 `from_code_arrays` convenience constructor，不让 native hot path出现逐 observable FFI。

空 batch 合法：`expectations()` 返回 shape `(0,)`，`values_and_gradients()` 返回 `(0,)` values 和 `(0, nparameters)` gradients。Batch size 1 必须与现有 engine 相同，不要求用户迁移原 API。

## 5. Rust core 结构

### 5.1 Shared compiled program

将当前 `PropagationEngine` 中与 observable 无关的 immutable 数据抽为 private 或 public-core-internal program，例如：

```rust
struct CompiledPropagationProgram {
    nqubits: usize,
    operations: Arc<[GateOperation]>,
    initial_state: ProductState,
    max_weight: Option<usize>,
    max_bytes: Option<u128>,
    nparameters: usize,
}

struct PropagationEngine {
    program: Arc<CompiledPropagationProgram>,
    observable: PauliOperator,
    hermitian: bool,
}

struct PropagationBatch {
    program: Arc<CompiledPropagationProgram>,
    engines: Vec<PropagationEngine>,
}
```

具体名字不是 public contract。现有 `PropagationEngine::new()` 签名和 Python API保持兼容；它可以内部构造一个 shared program。Batch 构造只编译一次 tape/state/slot metadata，然后为每个 observable建立轻量 engine state。

这一 refactor 必须由 batch size 1 与现有版本的逐项 regression保护。不得借 Phase 5.5 重写 `PackedKey`、gate recurrence、aggregation、checkpoint/replay 或 local VJP。

### 5.2 Execution

Batch call先一次性验证参数 shape/finiteness和 `checkpoint_interval`，然后按固定 observable index range执行：

```text
if batch_work < private_parallel_threshold:
    for i in 0..B:
        results[i] = run_existing_kernel(engine[i], params)
else:
    results.par_iter_mut().enumerate().for_each(|(i, output)|
        *output = run_existing_kernel(engine[i], params)
    )
```

输出预先按 batch index分配。Worker不写共享 floating accumulator；每行 gradient由对应 observable独占写入。最终只做固定位置组装，不做跨行 reduction，因此 worker scheduling不影响数值顺序。

参数 slot的 `sin/cos` 可以在一次 batch call中预解析并供所有 engine读取，但这只是低风险内部优化。首版若为保持现有内核完全复用而继续在每个 observable内解析参数，也不影响语义；是否抽取由 profile决定。

### 5.3 Error collection

并行执行中的任意错误使整个 batch call失败。为了 deterministic diagnostics，若多个 observable失败，公开错误应报告最小 input observable index，而不是最先完成的 worker。不得依赖 Rayon scheduling选择错误。

## 6. PyO3 与数据布局

Batch construction 使用一次粗粒度 native call。多个 canonical observables按输入顺序展平为：

- `observable_offsets: uint64/usize[B+1]`；
- flat Pauli structures；
- flat real/imaginary coefficients。

Offsets必须checked、单调且最后一个值等于flat term count。Python wrapper不逐 observable调用PyO3。

`expectations()` 返回一个contiguous `float64[B]` NumPy array。`values_and_gradients()` 返回contiguous row-major `float64[B]` values与`float64[B * P]` gradients，并在Python facade reshape为 `(B, P)`。长时间construction和execution必须释放GIL。

Phase 5.5不引入ragged propagated-operator output；该输出需要额外offsets、canonical materialization和更大的FFI流量，没有当前batch-gradient需求支撑。

## 7. Memory 与并行边界

Batch-level `max_bytes` 至少cheap-estimate：

- shared compiled program与全部input observable storage；
- values和完整`B * P` gradient output；
- 每个active worker的主要current/next terms、checkpoint/replay states、adjoints和单行gradient workspace；
- batch result/error staging。

该估算仍是项目统一的best-effort guard，不是exact peak RSS。Implementation可以根据budget降低active observable workers或退回串行；不能为了并行在低budget下无条件同时启动全部observables。`max_bytes=None`关闭public guard但不关闭checked arithmetic。

现有Rayon线程配置继续作为执行线程来源，Phase 5.5不新增public thread-count参数。Benchmark必须记录实际线程数，并分别测1-thread与默认/固定多线程。调用者同时使用Python线程、JAX或BLAS线程池时仍需自行避免oversubscription；库不得在每次call创建新的OS thread pool。

## 8. 性能预期与限制

Observable-level parallelism不会减少单 observable的Pauli branching、aggregation或reverse checkpoint复杂度。对`B`个相近工作量observable，理想总work仍约为`B`倍；收益来自：

- tape/state/slot metadata只构造和保存一次；
- 一次parameter validation和一次PyO3 call；
- 多个独立observable在CPU cores上并行；
- 输出一次连续分配和组装。

当每个observable很轻时，Rayon调度和结果组装可能让batch不快于简单串行循环。实现必须保留串行阈值，并把“batch size 1 overhead”和“crossover batch/work size”作为benchmark结果，而不是宣称无条件线性speedup。

如果用户只需要sum value/gradient，现有单`PauliOperator`路径通常更省计算和内存：它只返回一个gradient向量，并允许同一observable内的重复Pauli words聚合。Phase 5.5主要服务多目标、逐项诊断、多个Hamiltonian/response component或必须保留每个observable结果的工作流。

## 9. Correctness tests

首版至少覆盖：

1. Batch size `0/1/4/16`，逐行对照现有单 engine。
2. 单项 Pauli strings 与多项 PauliOperator混合的batch。
3. Exact recurrence和有限`max_weight`，包括aggregation后projection。
4. Static/slot rotations、shared parameter slots、Clifford gates与custom PTM。
5. Zero/computational-basis/product-Bloch states。
6. Empty observable、identity observable、同batch重复observable和不同observable内部exact cancellation。
7. Frozen-support边界：`sin(theta)=0`、`cos(theta)=0`和aggregate exact-zero；只要求每行等于对应单 engine，不要求batch线性组合等于sum engine。
8. `nqubits=0`和64/65/128/129-qubit packed-key边界。
9. 1-thread与多线程结果、重复调用和并发调用的bitwise稳定性。
10. 非Hermitian element、错误qubit数、invalid parameters、nonfinite arithmetic和memory guard；并行多错误时验证最小observable index。

独立dense reference只需用于小系统确认vector values/Jacobian；主要batch oracle必须是现有单 engine逐行结果，因为Phase 5.5的目标是批量执行等价而不是改变数学模型。

## 10. Benchmark 与 go/no-go

Benchmark必须比较完整公开边界：

1. 构造`B`个现有`PropagationEngine`与构造一个`PropagationBatch`。
2. Python串行调用`B`个engine与一次batch call。
3. 1-thread batch与固定多线程batch，记录parallel efficiency。
4. `expectations()`与`values_and_gradients()`分开测量。

固定case至少覆盖batch size `1/4/16/64`、12q rotation-heavy、100q near-Clifford、轻量单-Pauli-string与较重多项observable，并记录每个observable的initial/peak/final term statistics或其分布、parameter count、checkpoint interval、runtime、throughput、output bytes、estimated peak bytes、thread count和numerical error。

Phase 5.5是可选增量，不因已有文档自动进入实现。开始实现前至少应有一个真实使用场景需要保留独立observable rows，并且串行多engine调用在该流程中是可见成本。若用户只需要sum objective，或prototype显示batch长期处于调度开销区间，则保留现有API并推迟本阶段。

完成不设置固定speedup倍数，但必须证明：代表性heavy batch在多线程下相对同一批单engine串行调用有稳定端到端收益；batch size 1无显著回退；内存随active workers而不是无界随B同时增长。若这些条件不成立，不应仅为API完整性合并batch实现。

## 11. 实现切片

### P0：Baseline 与 row-wise contract

- 增加只使用现有engine的串行reference helper和公开边界benchmark。
- 固定sum-vs-batch frozen-support语义、输出shape、错误index和empty batch行为。
- 记录轻/重observable的parallel crossover。

Acceptance gate：确认至少一个真实workload达到go/no-go条件；否则Phase 5.5保持deferred，不继续实现。

### P1：Shared compiled program

- 抽取immutable shared program并让现有单engine复用。
- 保持现有Rust/Python constructor与全部单engine结果不变。
- 增加batch size 1 bitwise regression与setup/storage benchmark。

Acceptance gate：现有Phase 3/4 regression全部通过；单engine没有未解释的correctness或代表性性能回退。

### P2：Batch scalar expectations

- 增加ragged observable input、native batch handle和contiguous values output。
- 实现observable-level serial/parallel自动选择与deterministic error index。
- 增加GIL/concurrency、memory和thread-count tests。

Acceptance gate：每行等于单engine；heavy batch显示稳定端到端收益；轻量batch不会因强制并行明显退化。

### P3：Batch frozen-support reverse

- 复用现有`value_and_grad`内核生成row-major Jacobian。
- 估算`B * P` output与active-worker checkpoint/replay workspace。
- 增加不同checkpoint interval、support boundary和shared-slot tests。

Acceptance gate：values与gradient rows逐项等于单engine；1-thread/多线程bitwise一致；budget不足时明确失败或降低并发而不是OOM。

### P4：Release benchmark 与交接

- 运行公开Python/FFI与Rust core release benchmark。
- 记录setup、steady、parallel scaling、memory和crossover。
- 更新README、typing、CHANGELOG、architecture与implementation status。

Acceptance gate：只在有真实收益和完整correctness evidence时标记Phase 5.5完成。

## 12. 非目标

- 不改变现有sum-of-Pauli `PropagationEngine`语义或API。
- 不实现跨observable coefficient-batched Pauli keys、dense coefficient lanes、shared aggregation map或observable-id global map。
- 不实现inner term-level parallelism或两层Rayon并行。
- 不实现compiled Clifford frame、stabilizer tableau、measurement sampling、noise或Stim替代品。
- 不实现batch SPPS、correlated sampling、common random numbers或observable-term sampling。
- 不实现batched propagated-operator materialization、per-gate gradients、observable-coefficient gradients、Hessian或JAX custom VJP/vmap。
- 不新增public strategy selector、parallel threshold或thread-count knob。

## 13. 最终判断

该方案在架构上是低破坏性的：现有single/sum engine保持权威内核，batch只增加共享immutable program、粗粒度FFI和observable-level调度。主要实施风险集中在内部program refactor、并行期best-effort memory估算和deterministic error reporting，而不是传播数学或VJP本身。

相对地，跨observable共享Pauli propagation state会显著扩大语义、数据布局和数值风险，且在当前需求下缺少收益证据，因此明确排除。Phase 5.5应保持可选；如果sum objective已满足主要用户需求，没有必要仅为“API看起来完整”而提前实施。

## 14. Implementation outcome — 2026-08-02

P0–P4 are implemented. The Rust core now shares an immutable compiled program through `Arc`, the PyO3 boundary accepts flattened observable offsets in one coarse-grained call, and the Python facade exposes `PropagationBatch` plus contiguous read-only values and Jacobian arrays. The existing scalar engine remains the row oracle and no cross-observable aggregation or nested parallelism was added.

Correctness evidence includes batch sizes 0/1/4/16, exact and finite weight projection, static and parameterized gates, custom PTMs, product states, repeated calls, invalid rows/parameters, memory guards, and row-wise deterministic gradients. Release benchmarks include batch and serial B-engine controls at B=1/4/16/64; the full contract is now accepted for this optional phase, with unconditional speedup intentionally not promised.
