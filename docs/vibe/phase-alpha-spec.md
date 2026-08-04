# Phase Alpha：统一 Python Circuit Facade

状态：已实现的 Phase Alpha public contract。本文冻结 Python 用户层的调用形状；U(1)、deterministic Pauli propagation 和 SPPS 的 native executor 继续保持各自实现，不要求共享 Rust kernel、内存布局或编译器。

> API note: this historical specification predates the breaking Phase 8 API contract; current public names and signatures are defined in [`phase-8-api-coherence-spec.md`](phase-8-api-coherence-spec.md).

## 1. 目标

Phase Alpha 的目标不是增加新的量子算法，而是把已有的 U(1) state execution、deterministic Heisenberg Pauli propagation 和 stochastic Pauli-path execution 组织成一套一致的 Python circuit facade，使 Agent 可以用同一套规则构造线路、声明参数、编译线路，并对 Hamiltonian/observable 请求 value、gradient 或 estimator。

TenCirPauli 是 TensorCircuit 的 native companion package。Python distribution 将 TensorCircuit 作为必需运行依赖；Rust core 仍然保持纯 Rust，不依赖 Python、PyO3、NumPy 或 TensorCircuit。TensorCircuit 的 backend/JAX graph 与 TenCirPauli 的 host-side Rust execution 是两个明确的执行边界。

## 2. 非目标

- 不统一 U(1)、deterministic propagation 和 SPPS 的内部执行器。
- 不要求三个 executor 使用同一个 Rust IR、同一个稀疏存储或同一个 scratch buffer。
- 不把 native `value_and_grad` 伪装成 JAX-traceable primitive。
- 不把 `PropagationCircuit` 或 `SPPSCircuit` 设计成通用 statevector simulator。
- 不在 Phase Alpha 实现 Phase 6.5 time evolution、Phase 7 新算法或新的物理模型 factory。

## 3. 用户层统一合同

三个 public circuit facade 都遵循以下生命周期：

```text
construct circuit → append gates → compile/cache → evaluate observable
```

高层调用形状统一为：

```python
circuit = tcp.<CircuitType>(nqubits=..., ...)
circuit.<gate>(..., theta=...)
value = circuit.expectation(observable, parameters=...)
result = circuit.value_and_grad(observable, parameters=...)
```

其中 `<CircuitType>` 为 `U1Circuit`、`PropagationCircuit` 或 `SPPSCircuit`。`observable` 使用 public `PauliOperator`；Hamiltonian 只是 VQE 场景下对 observable 的常用称呼。

所有 facade 都应提供 `nqubits`、`nparameters`、`to_qir()`、`compile()`、`expectation()` 和 `value_and_grad()`；具体 compiled-plan 类型和额外 terminal 由 executor 决定。`expectation()` 只执行 value path，不计算 gradient。重复调用同一结构的 `expectation()` 或 `value_and_grad()` 必须复用可复用的 compiled native handle。

## 4. Parameter 与 theta 合同

所有 circuit facade 的参数入口统一使用 `theta=`。静态数值、`Parameter` 和 `ParameterExpr` 是结构层面的 angle 表达式；实际数值通过一维 parameter vector 传入。

```python
p0 = tcp.Parameter(0)
p1 = tcp.Parameter(1)

circuit.rz(0, theta=p0)
circuit.rzz(0, 1, theta=p0)
circuit.iswap(1, 2, theta=2.0 * p1 + 0.1)
```

同一个 `Parameter(slot)` 可以出现在任意多个 gate 中。其 gradient 是这些 gate 对同一个 slot 的总导数。`ParameterExpr` 的 chain rule 属于 Python facade 合同；native executor 可以使用自己的表达式编译方式，或者在 facade 层将 local gate derivatives 映射回 public slots。

运行时参数接受 Python sequence、NumPy array 和可转换为 host NumPy `float64` 的 concrete backend array，包括 concrete JAX array。入口统一执行 finite、shape、contiguous 和 `float64` conversion，然后把一份 host buffer 交给 Rust。JAX tracer、`jax.jit` 内的动态 tracer 和要求 native call 保持在 JAX graph 中的用法不属于 Phase Alpha 合同；需要 JAX tracing 的计算使用 backend MVP 路径。

如果 gate 在构造时使用静态 `theta=0.2`，它不是可微 parameter。只有 `Parameter`/`ParameterExpr` gate 才会在 `value_and_grad()` 中产生对应的 gradient 分量。

## 5. U1Circuit

`U1Circuit` 是 fixed-particle-number Schrödinger state facade。它继续提供 TensorCircuit-compatible 的 U(1) gate subset：`rz`、`rzz`、`cz`、`cphase`、`swap`、TensorCircuit-convention `iswap` 和静态 `diagonal`。

```python
circuit = tcp.U1Circuit(nqubits=8, k=2, filled=[0, 1])
circuit.iswap(0, 1, theta=tcp.Parameter(0))

result = circuit.value_and_grad(hamiltonian, parameters=[0.25])
energy = circuit.expectation(hamiltonian, parameters=[0.25])
state = circuit.state([0.25])
probability = circuit.probability([0.25])
```

U1-specific terminals 包括 `state()`、`wavefunction()`、`probability()`、`probability_full()`、`to_dense()`、`expectation_z()`、`expectation_ps()` 和 `expectation_pss()`。这些 terminals 不要求被其他 circuit facade 复制；`expectation(observable, parameters)` 是统一的 PauliOperator value-only 入口。

`U1Circuit.compile()` 主要编译 circuit structure、sector map、gate schedule 和 restricted-state kernel；observable 可以在高层 expectation/value-and-gradient call 中传入。对于重复执行同一 circuit，compiled plan 和 final-state cache 应保持可复用。

## 6. PropagationCircuit

`PropagationCircuit` 是 deterministic Heisenberg Pauli propagation 的用户层 circuit facade。它不返回 Schrödinger state；其主要 terminals 是 `expectation()`、`value_and_grad()`、`propagate_operator()` 和 `profile()`。

```python
p0 = tcp.Parameter(0)
p1 = tcp.Parameter(1)

circuit = tcp.PropagationCircuit(
    nqubits=8,
    initial_state=tcp.ZeroState(),
)
circuit.h(0)
circuit.cnot(0, 1)
circuit.rz(1, theta=p0)
circuit.rxx(2, 3, theta=p1)

energy = circuit.expectation(
    hamiltonian,
    parameters=[0.25, -0.12],
)
result = circuit.value_and_grad(
    hamiltonian,
    parameters=[0.25, -0.12],
)
```

支持的 gate set 继续由 deterministic propagation executor 决定，初始范围为 `X/Y/Z/H/S/Sdg/CNOT/CZ/SWAP`、`RX/RY/RZ/RXX/RYY/RZZ` 和有限 real local PTM。U1 gate subset 与 generic propagation gate set 不需要相同。

`PropagationCircuit.compile(observable, *, initial_state=None, max_weight=None, max_bytes=...)` 返回 propagation-specific compiled plan。observable 是 propagation setup 的结构输入，因此显式 plan 可以把 observable 固定；高层 `value_and_grad(observable, parameters=...)` 负责自动创建和缓存该 plan。

低层 `GateTape` 和 `PropagationEngine` 继续保留，供需要显式控制 native tape、initial-state descriptor、profile 或批量 observable 的 Agent 使用，但不再是 README 中的主要 VQE 入口。

## 7. SPPSCircuit

`SPPSCircuit` 与 `PropagationCircuit` 共享 circuit construction、parameter 和 observable 入口，但返回随机 estimator，而不是 deterministic scalar contract。

```python
circuit = tcp.SPPSCircuit(nqubits=8, initial_state=tcp.ZeroState())
circuit.h(0)
circuit.cnot(0, 1)
circuit.rz(1, theta=tcp.Parameter(0))

estimate = circuit.value_and_grad(
    hamiltonian,
    parameters=[0.25],
    samples_per_term=256,
    seed=7,
)
```

SPPS 同时提供 value-only：

```python
estimate = circuit.expectation(
    hamiltonian,
    parameters=[0.25],
    samples_per_term=256,
    seed=7,
)
```

SPPS value-only 返回不包含 gradient 的 stochastic estimate；其统计字段至少包括 `value`、`value_standard_error`、`replicates`、`samples_per_replicate`、`total_paths` 和 `seed`。`value_and_grad()` 返回 `SPPSEstimate` 的完整字段：`value`、`gradient`、`value_standard_error`、`replicates`、`samples_per_replicate`、`total_paths`、`seed`、`gradient_error_proxy`、`term_gradient_error_proxies` 和 `converged`。统一的是调用形状，不是数值确定性、误差合同或停止准则。

SPPS 的 `value_and_grad_adaptive()` 仍然使用独立的 adaptive keyword 参数和 empirical proxy；不能把它包装成 deterministic `PropagationValueAndGradient`。

## 8. TensorCircuit circuit conversion

TensorCircuit 是必需 package，但用户层函数名不重复包含框架名称。目标类型 classmethod 使用：

```python
native_u1 = tcp.U1Circuit.from_circuit(tc_u1_circuit)
native_prop = tcp.PropagationCircuit.from_circuit(tc_circuit)
```

`from_qir()` 保留为低层、schema-directed restore API。`from_circuit()` 读取 TensorCircuit circuit 的 QIR 和 circuit parameters，并把 numeric angle、direct symbol 和 supported gate 映射到统一的 public parameter contract。numeric QIR 产生静态 gate；需要梯度时，circuit 必须保留可识别的 symbol/parameter reference，不能在每个 optimizer step 重新编译。

TensorCircuit 的 gate object 不进入 TenCirPauli logical QIR。`diagonal` 在 adapter boundary 被转成静态 contiguous complex array。Phase Alpha 的 canonical logical QIR 使用 `name`、`index`、`parameters` 和 `diagonal` payload；adapter 可以兼容 TensorCircuit 当前的 gate-object payload 和历史 `diag` spelling，但 public native serialization 只保留一个 canonical spelling。

## 9. Backend MVP

MVP 是 TensorCircuit backend-facing path，但函数名不需要重复 `TensorCircuit`：

```python
plan = hamiltonian.backend_mvp_plan()
mvp = tcp.backend_mvp(plan)
```

该路径可以接受 TensorCircuit NumPy/JAX backend tensor，并保留 backend execution、JIT 和 autodiff 的能力。它与 `PropagationCircuit`/`SPPSCircuit` 的 host-side native gradient 是不同执行合同。

## 10. 结果接口

`expectation(observable, parameters=...)` 是 value-only 接口。对于 deterministic circuits，它返回与同一次执行配置下 `value_and_grad(...).value` 一致的标量，但不分配或计算 gradient。对于 Hermitian VQE observable，public return type 为 Python `float`；支持一般 complex observable 的 executor 可以返回 `complex`，并必须在 docstring 中明确该行为。

确定性 circuit 的 value-and-gradient result 至少提供：

```python
result.value
result.gradient
```

`gradient` 是 contiguous、read-only `float64` array，shape 为 `(nparameters,)`。U1 和 deterministic propagation 可以保留不同的具体 result dataclass，但字段和 shape 必须一致。SPPS 额外提供 estimator metadata，不强行复用 deterministic result type。

## 11. Phase Alpha acceptance

- `U1Circuit`、`PropagationCircuit` 和 `SPPSCircuit` 都使用 `theta=`、`Parameter`、runtime parameter vector 和 `value_and_grad(observable, parameters=...)` 的共同形状。
- `U1Circuit`、`PropagationCircuit` 和 `SPPSCircuit` 都提供 `expectation(observable, parameters=...)` value-only 入口；deterministic value 与对应 `value_and_grad(...).value` 一致。
- 同一 parameter 在多个 gate 中复用，并有独立测试验证总梯度。
- static numeric gate 不生成 gradient；parameterized gate 产生对应 slot gradient。
- concrete NumPy、Python list 和 concrete JAX array 都能进入 host-side native call。
- `jax.jit`/JAX tracer 不被误报为 supported native autodiff。
- `from_circuit()` 是用户级 TensorCircuit conversion 入口；`from_qir()` 是低层 schema 入口。
- `diagonal` method、logical QIR 和 TensorCircuit adapter 的 payload 语义一致。
- three native executor 的 correctness、performance、memory 和 stochastic contracts 分别验证，不因 facade 统一而合并 benchmark 或 oracle。
- README、examples、CI 和 public docstrings 只描述本合同中已实现的 API；后续未实现能力必须进入新的 spec，不得把本合同的 target 重新标成 pending。

Phase Alpha 已完成。后续工作可以独立决定是否进入新的 high-level facade；本阶段不启动 Phase 6.5 或 Phase 7。
