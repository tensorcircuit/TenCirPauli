# Phase 6 Spec：common circuit IR and Rust-native U1Circuit

状态：设计已冻结，可直接交付实现。Phase 6先建立backend-neutral logical circuit IR，但只实现fixed-particle-number U1 execution、observable evaluation和精确native adjoint gradient，不实现通用full-state simulator。Phase 6.5的MVP evolution proposal已搁置，不是本阶段完成后的自动下一里程碑。P0–P4的语义、范围、接口、错误行为和acceptance gates均以本文为准；实现中发现需要改变公开语义时必须先更新本文，不得自行留下隐式分叉。

## 1. 目标与边界

Phase 6先建立一个与execution mode无关的logical circuit层，统一typed gates、parameter slots、parameter-expression DAG、线路变换和Python/Rust transport serialization。该层不模拟量子态，也不假设full-state `2**n` indexing、U1 sector、combinatorial rank或packed occupation representation；它只是未来不同circuit execution backend都可消费的语义输入。

第一个且本阶段唯一的execution backend是在现有`U1Sector`、任意宽packed occupation、restricted rank/lookup和Python/TensorCircuit integration基础上的精确Schrödinger-picture U1 statevector engine。公开Python facade尽量与TensorCircuit `U1Circuit`的常用构造、gate名称、参数名和terminal方法保持可替换；底层执行模型不复制TensorCircuit的逐gate backend dispatch，而是lazy record全部gates，在`compile()`、`state()`、`expectation_*()`、`probability()`或gradient terminal首次触发时一次性跨PyO3，将完整logical circuit编译为U1-specific plan并在Rust内执行。

本阶段的核心价值是restricted dimension `C(n,k)`上的高性能CPU circuit execution、粗粒度FFI、可复用compiled plan、精确observable evaluation和显式adjoint gradient。公共logical layer是为避免把参数、gate和serialization重复绑定到U1而建立的最小共享边界，不构成通用模拟器承诺。Phase 6不是TensorCircuit `AbstractCircuit`的完整重实现，也不包含full-state simulation、noise、density matrix、sampling、RDM、entropy、time evolution solver、automatic Trotter compiler、JAX custom call或accelerator backend。

## 2. 已确认的 owner decisions

1. **拆分 Phase 6/6.5**：Phase 6 只做 U1 circuit；Phase 6.5 是与 U1 无关的通用 Rust-native matrix-free time evolution。
2. **lazy terminal execution**：Python gate methods只记录 typed operations，不逐 gate 调 native。Terminal call 或显式 `compile()` 才跨边界。
3. **语义兼容优先**：支持的 TensorCircuit `U1Circuit` 名称、参数名、qubit ordering、basis ordering和angle convention尽量相同；未支持能力在Python层明确失败。
4. **TensorCircuit iSWAP 兼容**：`iswap(i, j, theta)` 使用 TensorCircuit 的 `theta * pi / 2` mixing convention。额外物理角度 gate 使用独立名称，不能改变 `iswap` 语义。
5. **全 Rust execution**：一次 run、expectation 或 gradient 的 circuit kernel、state更新和reduction全部留在Rust；只返回最终state或小型结果。
6. **compiled fusion**：Rust compiler必须生成融合后的执行schedule并复用wire/pair metadata；不得按Python gate对象逐个回调。
7. **精确 adjoint gradient**：Phase 6 gradient是restricted statevector的普通光滑导数，不沿用Phase 4 frozen-support语义。Reverse同时逆演forward state和adjoint state，避免保存全部中间state。
8. **任意宽 occupation**：不得重新引入TensorCircuit当前`nqubits < 64`限制。只要`C(n,k)`、native indexing和memory可安全表示，64/65/128/129+ qubit low-k/low-hole circuit必须可执行。
9. **native CPU first**：不在本阶段实现JAX JIT、custom call、custom VJP、GPU或TensorFlow graph execution；这些留给后续integration阶段。
10. **通用 parameterized circuit IR**：公开类型固定为与U1无关的`Parameter(slot)`和immutable `ParameterExpr`，不另建`U1Parameter`。Gate angle接受finite numeric constant或由base slots组成的表达式；首版表达式只支持constant、slot、unary negation和`+ - * /`。Slot identity和expression topology必须可跨Python/Rust边界无损往返；表达式在Rust中一次求值并通过reverse expression pass把gate-angle adjoints累加到base slots。
11. **QIR只在Python层**：TensorCircuit QIR dict、gate name、`append()`、`inverse()`和parameterized circuit transformations属于Python logical tape；Rust只接收compact typed gate/expression IR，不实现TensorCircuit serialization语义。
12. **projected observable内部化**：`expectation_ps/pss`内部编译`P_k O P_k`，但首版不新增公开projected-observable类型，也不改变strict `restrict_u1()`。
13. **公共层、专用执行**：gate、parameter、expression和logical serialization属于backend-neutral common circuit layer；sector validation、basis indexing、pair maps、fusion schedule、state kernels、observable reduction和adjoint gate VJP属于U1 backend。Phase 6不先实现通用full-state simulator，也不让U1热路径调用full-state kernel。
14. **不提前铺设空 backend abstraction**：公共IR只包含本阶段实际需要的gate和payload，不预定义GPU、density-matrix、tensor-network或arbitrary-unitary execution traits。未来backend扩展IR时使用显式schema evolution；当前U1 compiler对unknown或non-number-conserving gate明确失败。

## 3. 与 TensorCircuit 的兼容级别

兼容目标是调用形状和科学语义，而不是继承 TensorCircuit class 或复用其backend graph。以下首版能力应尽量保持相同名称和参数：

```python
U1Circuit(nqubits, k=None, filled=None, inputs=None)
c.rz(i, theta=...)
c.rzz(i, j, theta=...)
c.cz(i, j)
c.cphase(i, j, theta=...)
c.swap(i, j)
c.iswap(i, j, theta=1.0)
c.diagonal(*indices, diag=...)
c.state()
c.wavefunction()
c.probability()
c.to_dense()
c.probability_full()
c.expectation_z(i)
c.expectation_ps(x=None, y=None, z=None, ps=None)
c.expectation_pss(ps_list, coefficients)
```

TenCirPauli增加keyword-only `parameters`、constructor-level `max_bytes`、`compile()`、`value_and_grad()`和显式plan API；已有TensorCircuit风格的numeric circuit调用不因不必要的参数重命名而改变。

首版明确不承诺TensorCircuit的`sample()`、`measure()`、`reduced_density_matrix()`、`entanglement_entropy()`、任意`AbstractCircuit` gate、backend tracing或跨全部TensorCircuit gate的QIR round-trip。`to_qir()`、`from_qir()`、`append()`、`inverse()`和parameter remapping只在Python logical tape层覆盖本spec的supported gate/parameter-expression set；QIR parameter values可以保留TenCirPauli `Parameter/ParameterExpr` Python objects供自身round-trip，但这不宣称TensorCircuit backend能执行compound expressions。Rust core不解析或返回TensorCircuit QIR dict。`to_dense()`和`probability_full()`是required bounded terminals，分别将restricted amplitudes或probabilities scatter到TensorCircuit ordering的长度`2**nqubits`输出；它们必须在checked dimension与`max_bytes`验证后分配，不能用作full-state gate execution fallback。

Optional TensorCircuit integration沿用现有adapter风格，增加`u1_circuit_from_tensorcircuit(circuit, *, parameter_order=None, max_bytes=DEFAULT_MAX_BYTES) -> TensorCircuitU1Conversion`。Conversion result包含`circuit: U1Circuit`和`parameters: tuple[object, ...]`。Adapter lazy-import TensorCircuit，读取source `U1Circuit.to_qir()`与`circuit_param`，只接受本spec gates以及finite numeric或direct SymPy-symbol angles；compound SymPy expressions、backend tracers和unsupported gates明确失败。默认slot order按symbol在QIR中的first appearance，显式`parameter_order`必须无重复且exactly cover全部symbols。TensorCircuit缺失时给出现有optional-dependency错误，不fallback。

对 `x/y/h/rx/ry/cnot` 等已知不守恒gate，facade应提供清楚的Python错误或在QIR import时fail fast，不能静默忽略、投影或回退到full-state simulation。

Semantic baseline固定为TensorCircuit `1.8.0`中reference commit `55fd1630448b04ba29deebbc25743422d62de8b9`的`tensorcircuit/u1circuit.py`和`tensorcircuit/gates.py`。Qubit `0`是full computational-basis integer的most-significant bit；restricted basis按该integer升序，与现有`U1Sector`一致。所有gate、dense conversion和Pauli differential fixtures都直接使用该ordering，不允许通过最终global-phase alignment掩盖公式差异。

Constructor接受non-negative integer `nqubits`和`0 <= k <= nqubits`。若`k is None`，必须由`filled`长度推断；`k`与`filled`同时提供时长度必须一致。`filled`必须是无重复、in-range integer indices；`filled is None`且已给`k`时默认`range(k)`。`inputs`提供时仍需由`k`或`filled`确定sector，并覆盖one-hot initialization；否则`filled`决定初始computational basis state。`k`和`filled`同时缺失、bool冒充integer或任何shape/index不合法均在Python层失败。

## 4. Public Python API

### 4.0 Common logical circuit layer

Phase 6不新增一个可执行的public `Circuit` class。Python内部使用backend-neutral `_CircuitProgram`或等价typed structure作为`U1Circuit`的logical tape；未来若增加full-state、tensor-network或其他execution mode，可以复用该structure并提供新的public facade和compiler，而不改变现有parameter或gate serialization。

Common program只拥有：

- `nqubits`和ordered typed gate operations；
- immutable parameter-expression DAG及base-slot count；
- gate wires、angle-node references和bounded static payload；
- deterministic logical schema version。

它不拥有U1 particle number、restricted initial state、basis ranks、pair maps、state buffers、observable plans或compiled fusion schedule。`U1Circuit`额外拥有sector与initial-state configuration，并把common program交给U1 compiler。Common program不得计算`2**nqubits`，不得用single-word bitstring限制wire count，也不得因未来full-state backend的dimension ceiling拒绝64/65/128/129+ qubit logical circuits。

Schema version固定从`1`开始，只定义Phase 6实际使用的RZ、RZZ、CZ、CPhase、SWAP、iSWAP和static diagonal operations。Gate opcode表达科学语义而不是Python method object或backend kernel；particle-number conservation由U1 compiler验证。Version 1 transport使用fixed-width unsigned integer arrays存放opcodes、wire/node indices和offsets，`float64`存放constants，`complex128`存放static payload；Python integer在编码前执行checked conversion。未来增加gate variant必须递增或兼容扩展schema version并增加对应target validation，不改变Parameter identity；unknown version或opcode明确失败。

### 4.1 Parameter slots and expressions

Public parameter primitives与U1无关，可在未来其他compiled circuit中复用：

```python
p0 = Parameter(0)
p1 = Parameter(1)
angle = 2.0 * p0 - p1 / 3.0

c = U1Circuit(12, k=2, filled=[0, 3])
c.rz(0, theta=angle)
c.iswap(0, 1, theta=p0)
value = c.expectation_ps(z=[0], parameters=[0.2, -0.4])
```

`Parameter(slot)`是轻量identity token而不是SymPy symbol，也不归属于某一个circuit instance；同一slot number跨append、serialization和native round-trip保持同一参数身份。`ParameterExpr`是immutable arithmetic DAG，支持finite `float` constants、slot leaves、unary negation、addition、subtraction、multiplication和division。首版不支持power、sin/cos/exp、comparisons、conditionals、Python callbacks或一般symbolic simplification；gate kernel本身负责对最终angle执行所需trigonometric functions。

`Parameter` constructor只接受non-negative Python integer且拒绝bool；两个`Parameter` objects只要slot相同就表示同一leaf。`Parameter`和`ParameterExpr`实现unary `-`以及双向`+ - * /` operator overload，另一operand只能是finite real scalar、`Parameter`或`ParameterExpr`；complex、bool和nonfinite constants立即失败。Expression objects不暴露mutable child lists，Python encode按operand order保留floating evaluation order。

Slots必须覆盖`0..nparameters-1`且无洞。相同slot或相同expression可被多个gates引用并在gradient中自动合并贡献。Numeric `theta`保持TensorCircuit普通调用语义并作为static constant，可被constant-fold/fuse；只有expression中实际出现的slots需要runtime `parameters`。

Compiler把expression DAG拓扑排序、执行constant folding、dead-node removal和exact structural common-subexpression reuse，但不得进行会改变floating evaluation order的algebraic reassociation。Runtime每次run先用一个linear pass计算所有expression nodes；division-by-zero或任何nonfinite intermediate明确失败。

### 4.2 Parameterized circuit transformations

Python logical tape保留typed gates和expression DAG，因此以下操作返回新的parameterized `U1Circuit`而不执行simulation：

```python
bound = circuit.bind_parameters({0: 0.25})
remapped = circuit.remap_parameters({0: 1, 1: 0})
inverse = circuit.inverse()
combined = circuit.append(other, parameter_map={0: 1})
```

Required signatures为：

```python
def bind_parameters(self, values: Mapping[int, float]) -> U1Circuit: ...
def remap_parameters(self, mapping: Mapping[int, int]) -> U1Circuit: ...
def inverse(self) -> U1Circuit: ...
def append(
    self,
    other: U1Circuit,
    *,
    parameter_map: Mapping[int, int] | None = None,
) -> U1Circuit: ...
def to_qir(self) -> list[dict[str, object]]: ...
@classmethod
def from_qir(
    cls,
    qir: Sequence[Mapping[str, object]],
    circuit_params: Mapping[str, object],
    *,
    max_bytes: int | None = DEFAULT_MAX_BYTES,
) -> U1Circuit: ...
```

`bind_parameters`用constants替换指定slots并constant-fold，然后把remaining slots按旧slot升序确定性compact到`0..nparameters-1`；该升序规则就是公开的old-to-new mapping，不增加额外provenance对象。`remap_parameters`允许多个旧slot映射到同一新slot以显式共享参数，但结果slot集合必须恰为无洞的`0..nparameters-1`，否则失败。`inverse()`逆序gates并对RZ/RZZ/CPhase/iSWAP angle expressions取负、保持CZ/SWAP自逆、对static diagonal取共轭。`append()`默认保留slot numbers，因此相同slot表示有意共享；`parameter_map`只作用于被append circuit，未列出的slot保持原编号，最终slot集合也必须无洞。所有QIR import/export和这些transformations停留在Python层。

Python与Rust之间使用language-neutral compact parameterized-circuit IR：schema version、`nqubits`、topologically ordered expression opcode/operand arrays、typed gate opcodes、wires、angle-node indices和static payload。Encoding必须deterministic，并具有无损的Python encode/decode路径，保留slot numbers、expression node references和gate order，使未来Rust transformation可以返回新的parameterized circuit并由Python重建。Phase 6验证IR round-trip、malformed-input rejection和native validation；它不是承诺跨release永久兼容的公开磁盘格式，也不要求Rust实现TensorCircuit QIR或任意circuit synthesis。Compiled/fused U1 schedule保持opaque，不要求反编译回logical tape。

### 4.3 Lazy facade

```python
class U1Circuit:
    def __init__(
        self,
        nqubits: int,
        k: int | None = None,
        filled: Sequence[int] | None = None,
        inputs: Sequence[complex] | np.ndarray | None = None,
        *,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
    ) -> None: ...

    @property
    def nparameters(self) -> int: ...

    def compile(self) -> U1CircuitPlan: ...

    def state(
        self,
        parameters: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray: ...

    def wavefunction(
        self,
        parameters: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray: ...

    def probability(
        self,
        parameters: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray: ...

    def to_dense(
        self,
        parameters: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray: ...

    def probability_full(
        self,
        parameters: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray: ...

    def expectation_ps(
        self,
        x: Sequence[int] | None = None,
        y: Sequence[int] | None = None,
        z: Sequence[int] | None = None,
        ps: Sequence[int] | None = None,
        *,
        parameters: Sequence[float] | np.ndarray | None = None,
    ) -> complex: ...

    def expectation_z(
        self,
        i: int,
        *,
        parameters: Sequence[float] | np.ndarray | None = None,
    ) -> float: ...

    def expectation_pss(
        self,
        ps_list: Sequence[object],
        coefficients: Sequence[complex] | np.ndarray,
        *,
        parameters: Sequence[float] | np.ndarray | None = None,
    ) -> complex: ...

    def value_and_grad(
        self,
        observable: PauliOperator,
        *,
        parameters: Sequence[float] | np.ndarray,
    ) -> U1CircuitValueAndGradient: ...
```

Gate calls只修改Python-side typed tape generation并使cached plan/run失效。`compile()`进行一次coarse native compilation并返回immutable plan。Terminal method若尚未compile则隐式compile；同一generation的plan必须复用。Mutable facade必须保留至多一个native final-state cache，key为logical generation、initial-state generation和全部runtime parameter的exact IEEE bit patterns；相同key的state/probability/observable terminals复用该state，不同key替换旧entry。NaN/Inf参数在进入cache前失败，`+0.0`与`-0.0`按不同bit pattern处理。Immutable plan methods保持stateless并发语义，不共享该facade cache。

在terminal之后继续append gate是合法的；它使之前的final-state cache失效，并在下一terminal从原始initial state按完整新tape重新执行。首版不要求增量native append或只执行delta gates。

### 4.4 Immutable execution plan

```python
class U1CircuitPlan:
    @property
    def sector(self) -> U1Sector: ...
    @property
    def dimension(self) -> int: ...
    @property
    def nparameters(self) -> int: ...

    def run(
        self,
        initial_state: Sequence[complex] | np.ndarray,
        parameters: Sequence[float] | np.ndarray = (),
    ) -> np.ndarray: ...

    def probability(
        self,
        initial_state: Sequence[complex] | np.ndarray,
        parameters: Sequence[float] | np.ndarray = (),
    ) -> np.ndarray: ...

    def to_dense(
        self,
        initial_state: Sequence[complex] | np.ndarray,
        parameters: Sequence[float] | np.ndarray = (),
    ) -> np.ndarray: ...

    def probability_full(
        self,
        initial_state: Sequence[complex] | np.ndarray,
        parameters: Sequence[float] | np.ndarray = (),
    ) -> np.ndarray: ...

    def expectation(
        self,
        initial_state: Sequence[complex] | np.ndarray,
        observable: PauliOperator,
        parameters: Sequence[float] | np.ndarray = (),
    ) -> complex: ...

    def value_and_grad(
        self,
        initial_state: Sequence[complex] | np.ndarray,
        observable: PauliOperator,
        parameters: Sequence[float] | np.ndarray,
    ) -> U1CircuitValueAndGradient: ...
```

Facade的`filled`初态在compile/run边界转换为restricted basis index one-hot；`inputs`必须可转换为shape `(C(n,k),)`的complex128 vector，并在constructor中复制为owned contiguous snapshot，后续调用者mutation不改变circuit。首版不自动归一化输入；它验证shape、finite values和major allocation。所有state/probability/gradient array outputs均为owned、C-contiguous、read-only NumPy arrays。`U1CircuitValueAndGradient`是frozen result object，字段为real `value: float`和read-only `gradient: np.ndarray`，gradient shape固定为`(nparameters,)`。Unitary gates应在floating tolerance内保持输入norm。

## 5. Observable 语义

### 5.1 Evolution/circuit observable不是strict restricted Hamiltonian

现有 `PauliOperator.restrict_u1()` 验证完整operator保持sector；该语义不能改变。Circuit expectation需要的是

```text
<psi_k | P_k O P_k | psi_k>
```

即使 `O` 单独泄漏sector，这个投影expectation仍然有定义。例如单独`XX`可将一部分basis state移出sector，但其sector内matrix element仍可用于expectation；odd single-X/Y string的projected expectation为零。

因此Phase 6必须增加private native projected-observable编译路径。它可以丢弃目标不在sector的matrix elements，因为这是显式`P_k O P_k`语义；它不得复用或弱化strict `restrict_u1()`的leakage error。首版只通过`U1Circuit`/`U1CircuitPlan` terminal暴露expectation，不增加容易与strict evolution Hamiltonian混淆的public projected-operator类型。

`expectation_ps()`和`expectation_pss()`统一返回Python `complex`，即使数学结果为实数；`expectation_z()`返回Python `float`。`value_and_grad()`只接受canonical `PauliOperator`中每个coefficient的imaginary component按IEEE comparison严格等于`0.0`的Hermitian objective，否则明确失败；其value返回real `float`。

`expectation_ps()`中`ps`与`x/y/z` forms互斥；`ps`长度必须等于`nqubits`且entries只能为`0/1/2/3`，而`x/y/z`中的indices必须分别无重复、彼此disjoint且in range。`expectation_pss()`在Python层把每行规范化为同一Pauli encoding，要求`len(ps_list) == len(coefficients)`、finite coefficients和一致的`nqubits`，然后通过一次native call计算一个sum objective。Empty sum返回`0j`。这些friendly-input checks不拆成per-term FFI。

### 5.2 Coarse terminal reductions

`state()`与`wavefunction()`返回相同的完整restricted state语义，shape为`(C(n,k),)`；`probability()`直接在native final state上生成同shape的`float64`概率。`to_dense()`将amplitudes直接scatter到TensorCircuit computational-basis ordering的shape `(2**nqubits,)` complex128 output；`probability_full()`直接scatter probabilities到float64 output，不先分配一个full complex state。两种full-shaped terminal均执行checked exponentiation、NumPy index-width检查和constructor `max_bytes` guard，不能因wide logical circuit发生shift overflow或先分配后失败。

`expectation_ps()`和`expectation_pss()`在native final state上直接reduction，不先把state复制到Python再计算。Phase 6只实现单个Pauli string和单个Pauli-sum objective；multi-observable batch、persistent public observable plan和Phase 5.5-style batched gradients均不进入本阶段。

## 6. Gate semantics

### 6.1 Required TensorCircuit-compatible gates

- `rz(i, theta)`：`exp(-i theta Z_i / 2)`。
- `rzz(i, j, theta)`：`exp(-i theta Z_i Z_j / 2)`。
- `cz(i, j)`：`|11>`分量乘`-1`。
- `cphase(i, j, theta)`：`|11>`分量乘`exp(i theta)`。
- `swap(i, j)`：交换两个occupation bits。
- `iswap(i, j, theta=1.0)`：在`|01>,|10>`子空间使用TensorCircuit `cos(theta*pi/2)`与`+i sin(theta*pi/2)` convention。
- `diagonal(*indices, diag)`：对至少一个指定qubit的computational configuration应用static unitary diagonal；`indices[0]`是`diag`索引中的most-significant local bit，输入长度必须为`2**len(indices)`并受checked arithmetic和`max_bytes`保护。

Wires必须distinct且in range。Angles必须为finite numeric、`Parameter`或`ParameterExpr`。`diagonal`只接受可转换为contiguous complex128的static vector；每个entry必须finite且满足`abs(abs(entry) - 1.0) <= 1e-12`，不满足时直接失败且不自动归一化。该unit-modulus合同保证inverse使用complex conjugate并保证adjoint reverse可重建forward state。Parameter-dependent arbitrary diagonal和其gradient不进入Phase 6。

### 6.2 Internal fused block and excluded public extensions

Phase 6不公开`givens()`、`fsim()`或general U1 block。Rust compiled schedule必须具有private two-qubit number-conserving block representation，在local basis `|00>,|01>,|10>,|11>`中表示为`phase00`、作用于`|01>,|10>`的`U(2)` block和`phase11`；它只用于required gates的static/adjacent fusion，不进入common logical schema或Python API。未来若新增public gate，必须另行定义名称、公式、parameter convention和VJP，不能改变TensorCircuit-compatible `iswap`。

## 7. Common IR and U1 compiler

### 7.1 Backend-neutral logical program

Pure-Rust core必须具有以下语义字段；private Rust type与field名称可以不同，但不得合并common logical program和U1 execution metadata：

```rust
pub struct CircuitProgram {
    schema_version: u32,
    nqubits: usize,
    operations: Arc<[CircuitGate]>,
    parameter_program: Arc<[ParameterExprNode]>,
    nparameters: usize,
}

pub struct U1CircuitPlan {
    sector: U1Sector,
    operations: Arc<[CompiledU1Gate]>,
    parameter_program: Arc<[ParameterExprNode]>,
    pair_maps: Arc<[CompiledPairMap]>,
    nparameters: usize,
    max_bytes: Option<u128>,
}
```

`CircuitProgram`是语义层，不含execution buffers或basis-specific metadata。`U1CircuitPlan::compile(program, sector, max_bytes)`验证所有operations保持particle number，然后生成U1-specific pair maps、occupation metadata和fused schedule。Logical gate order和expression nodes在compile后不再参与Python callbacks；plan可保留执行和gradient所需的compact parameter program，但不需要保留可反编译的logical object graph。

Rust core不得依赖PyO3、NumPy或TensorCircuit。Python facade把完整common program serialization、sector和static payload一次性flatten到native constructor；binding/native boundary先重建并验证`CircuitProgram`，随后在同一次coarse call中编译`U1CircuitPlan`。Long compile/run/expectation/gradient释放GIL。

Module placement固定为core `circuit_ir.rs`承载`CircuitProgram/CircuitGate/ParameterExprNode`与validation，core `u1_circuit.rs`承载U1 compiler、pair maps、execution、observable和adjoint kernels，native `u1_circuit.rs`承载thin PyO3 conversion，Python `circuit.py`承载public `Parameter/ParameterExpr`和private logical program，Python `u1_circuit.py`承载public facade/plan/result。顶层`python/tencirpauli/__init__.py`导出`Parameter`、`ParameterExpr`、`U1Circuit`、`U1CircuitPlan`和`U1CircuitValueAndGradient`；`_native` handles保持private。

Parameter expression opcode set固定为：

```rust
enum ParameterExprNode {
    Constant(f64),
    Slot(usize),
    Neg(usize),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
}
```

Operands只能引用更早node，使evaluation和reverse都为linear topological passes。Node count、depth、operand indices和bytes执行checked validation；malformed DAG在native plan construction时失败。

Common IR validation只检查schema、wire范围、expression topology、finite constants、payload offsets和checked sizes。Particle-number conservation、restricted dimension、pair counts和sector compatibility只能由U1 compiler检查；不得把这些规则写进通用serializer。

### 7.2 Pair maps and arbitrary width

Mixing gates只作用于两个wire occupation不同的basis pairs。对固定 `(i,j)`，需要更新的pair数为 `C(n-2,k-1)`。Compiler应直接enumerate其余`k-1` occupations并通过combinatorial rank得到两个restricted indices，而不是为每个gate构造full packed basis或对每个amplitude重新unrank。

每个distinct unordered wire pair最多缓存一份compact disjoint index-pair array；相同pair的SWAP/iSWAP复用。不得按gate实例保存`dimension`长度mapping。对low-hole sector使用active-hole symmetry，保持复杂度接近`min(k,n-k)`。

Diagonal gates可用occupation feature或compact support计算phase。Compiler不得为每个single-qubit RZ保存一个`dimension`长度mask；常用single/pair occupation metadata按distinct wire/support共享。

### 7.3 Fusion requirements

Compiler必须实现以下确定性fusion：

1. Maximal consecutive diagonal runs（RZ、RZZ、CZ、CPhase和static diagonal）编译为一个ordered `DiagonalBlock`并在一次state traversal中执行；block保留parameterized micro-operation metadata供reverse gradient使用。
2. Maximal consecutive SWAP/iSWAP runs只有在unordered wire pair相同时才编译为一个private number-conserving pair block；runtime参数求值后先在constant-size local matrices上按原顺序compose，再遍历restricted pair map。Reverse在local block内用ordered micro-operations和constant-size prefix/suffix products恢复每个angle contribution，不保存full-state intermediates。
3. Repeated pair maps和occupation metadata去重。
4. 预计算static trigonometric/block entries；runtime parameter expressions和各gate angle只解析一次。

不得跨run边界重排gate，也不得把被non-diagonal gate隔开的diagonal operations合并。Phase 6不实现wire-permutation tracking、跨non-diagonal block的global fusion、explicit SIMD或architecture-specific kernels；这些只能在后续profile驱动的spec中增加。Fusion前后state、expectation和gradient必须在本文tolerance内一致。

### 7.4 Execution kernels

Diagonal gate原位逐amplitude乘phase。Two-mode gate按disjoint restricted-index pairs原位执行2x2 complex update；不同pairs无写冲突，可在work足够大时Rayon并行。线程阈值必须按实际pair/amplitude数量和gate类型决定，不能重复Phase 5.5只按gate count触发的轻任务断崖。

Plan执行预分配并复用scratch；普通unitary gate path不应每gate clone整个state。Public state输出需要一个owned array；native internal run handle保留至多一个final state供多个terminal reductions。Expectation和gradient reductions使用与Rayon pool width无关的fixed logical chunks并按chunk index顺序merge，因此同一platform/build、相同input bits在1-thread和multi-thread下返回bitwise-identical public scalars/gradients；worker scheduling不能改变加法顺序。

## 8. Exact adjoint gradient

对real scalar objective

```text
f(theta) = <psi(theta) | O | psi(theta)>
```

forward只需得到final state `psi_L`。计算 `lambda_L = O psi_L` 后逆序遍历compiled gates，同时维护：

```text
psi_(j-1)    = U_j(theta)^dagger psi_j
lambda_(j-1) = U_j(theta)^dagger lambda_j
df/dtheta_p += 2 Re <lambda_j | (dU_j/dtheta_p) psi_(j-1)>
```

由于所有required gates都是unitary，reverse可从final state精确重建每个pre-gate state，不需要保存全部forward intermediates或checkpoint tape。Memory主要是final/current state、adjoint state、observable workspace和gradient vector，约`O(dimension + nparameters)`。

Adjoint gate reverse先得到每个dynamic gate angle node的scalar contribution `df/dphi_g`。随后parameter-expression DAG执行一次reverse pass：Neg、Add、Sub、Mul和Div使用显式local derivatives，把所有fan-out和shared-expression贡献按固定node/gate顺序累加到base slots。最终gradient shape为`(nparameters,)`，表示对用户传入slot vector的导数。

每个parameterized gate必须有显式local derivative：RZ、RZZ、CPhase和iSWAP分别测试。Shared parameter slots与nonlinear expressions通过expression reverse确定性累加。CZ、SWAP和static diagonal没有parameter gradient；private fused block的VJP只能由这些required gate rules按原顺序组成，不能引入数值parameter shift。

Gradient tests对照dense reference、finite differences和TensorCircuit JAX AD。Phase 6 gradient是exact restricted-state recurrence的导数，不使用coefficient cutoff、support freezing或Pauli-weight projection。

## 9. Memory, concurrency and errors

`max_bytes`继续是cheap best-effort major-allocation guard，不是exact RSS。至少估算input/final states、native cached state、pair maps、occupation metadata、diagonal payload、observable plan、adjoint state、gradient和batch outputs。Plan compile不能因为每个active worker单独低于limit而让aggregate workspace无界增长。

所有dimension/offset/pair-count arithmetic checked。显然过大的`2**arity` diagonal、state output和pair map在分配前失败。Long compile/run释放GIL；同一immutable plan允许并发run，每个call使用独立state/scratch。Mutable facade的cache不承诺并发mutation安全。

Runtime parameter vector长度必须严格等于`nparameters`；static circuit接受`None`或空vector，dynamic circuit收到`None`时失败。Unsupported gate、invalid wire、duplicate wire、invalid `k/filled`、wrong sector/state dimension、nonfinite input/angle/expression intermediate、division by zero、malformed expression DAG、parameter holes、nonunitary或wrong-length diagonal和non-Hermitian gradient objective全部明确失败。

## 10. Correctness tests

首版至少覆盖：

1. Common IR：deterministic encode、lossless decode/re-encode、bind/append/inverse/remap后slot identity、unknown schema/opcode和malformed offsets/nodes的明确错误。
2. Backend separation：common program在129/256 qubits上不计算full-state dimension；U1-specific sector、rank和pair metadata不进入logical serialization。
3. Initial states：`k` default filled、explicit filled、arbitrary restricted vector、k=0/k=n、low-k/low-hole。
4. Basis ordering：与现有`U1Sector`、TensorCircuit和small full-state reference一致。
5. 每个required gate的single-step dense differential，包括angle 0、special angles、negative angles和global phase。
6. Mixed multi-layer circuits、repeated wires/pairs、shared slots、shared subexpressions、nonlinear `+ - * /` expressions和static/parameterized混合。
7. 63/64/65、127/128/129和256 qubits的safe low-k cases。
8. Lazy semantics：gate append不触发native run；compile cache复用；append-after-terminal失效并给出正确state。
9. `state()`、`probability()`、`expectation_z/ps/pss`与full-state or independent restricted reference一致。
10. Projected observable与strict restricted Hamiltonian语义分离，包括single X/Y、XX、XX+YY和明确leaking Hamiltonian。
11. Adjoint gradient对TensorCircuit JAX、dense derivative和finite differences；shared slot、expression chain rule和all supported parameter gates。
12. 1-thread/multi-thread bitwise一致、deterministic parallel repeats、并发plan calls、memory guard和unsupported gates。

所有reference comparisons使用complex128/float64。State、probability和expectation differential tests直接比较，不做global-phase alignment，默认`atol=1e-11, rtol=1e-10`；analytic/dense/JAX-x64 gradient默认`atol=1e-9, rtol=1e-8`；central finite difference只作独立diagnostic，默认`atol=1e-6, rtol=1e-5`。若单个documented stress case因depth需要更宽tolerance，必须在该test旁解释并仍检查norm drift，不能全局放宽。

## 11. Benchmarks and acceptance

所有结论使用release build并分别记录setup/compile、first terminal、steady run、expectation、gradient、memory和accuracy。至少保留：

1. 12q k=2/k=6 mixed diagonal+iSWAP circuit，depth 20/100/500。
2. 20-32q low-k nearest-neighbor parameterized iSWAP+CPhase layers，比较TensorCircuit NumPy和warm-JIT JAX CPU。
3. 64/65/128/129/256q k=1/k=2 wide cases，验证无single-word ceiling。
4. Repeated same-pair和many-distinct-pair workloads，分离pair-map setup、fusion和steady execution。
5. Static diagonal-heavy、parameterized mixing-heavy和gradient workloads。
6. Facade end-to-end与precompiled plan steady path，确保Python gate recording和FFI成本被如实计入。

记录dimension、gate counts before/after fusion、distinct wires/pairs、parameter count、state/pair-map/scratch bytes、thread count、runtime、amplitude throughput、norm error、state/expectation/gradient error。Rust相对warm-JIT JAX CPU的性能不得预先保证；若代表性路径落后，先profile data layout、fusion、allocation、parallel threshold和FFI，再决定优化。

Full performance recording保持manual opt-in并通过现有`python benchmarks/run.py` workflow执行，不加入Git commit hook、默认`pytest`或wall-time CI gate。Commit前仍运行项目标准`python scripts/check.py`；benchmark smoke只验证harness和小型correctness path，不能替代release-mode representative measurements。

## 12. Implementation slices

### P0：Reference fixtures and contract vectors

- 将Section 3固定的TensorCircuit commit、basis/qubit ordering和required gate formulas固化为fixtures。
- 建立independent NumPy restricted/full-state oracle。
- 为已冻结的backend-neutral gate/parameter-expression schema、deterministic transport serialization、terminals和private projected-observable semantics建立contract vectors。

Acceptance gate：所有gate公式、iSWAP scale、lazy mutation semantics、parameter-expression evaluation/derivatives、logical IR round-trip和strict/projected operator区别有deterministic vectors；common IR不含full-state或U1 basis assumptions。

### P1：Pair maps and forward plan

- 实现common `CircuitProgram`、compact parameter-expression IR、constant folding和runtime evaluation。
- 实现logical parameterized-circuit IR的无损Python encode/decode、deterministic serialization与native validation；compiled schedule无需反编译。
- 实现从common program到U1 plan的target validation、arbitrary-width pair enumeration、shared metadata和required gates。
- 实现immutable plan、state run和Python lazy facade。
- 增加single/multi-gate/wide differential tests。

Acceptance gate：forward state正确、无per-gate FFI、无`depth * dimension` mapping storage；除U1 compiler/executor外的common layer不计算sector dimension或basis ranks。

### P2：Fusion and terminal observables

- 实现required low-risk fusion和native final-state cache。
- 实现state/wavefunction、restricted/full probability、bounded `to_dense()`和expectation_z/ps/pss。
- 增加projected observable plan而不改变strict `restrict_u1()`。

Acceptance gate：fusion前后结果满足Section 10 tolerance；observable terminal不回传state再由Python计算；`probability_full()`不分配intermediate full complex state；oversized full-shaped output在分配前失败。

### P3：Exact adjoint gradient

- 实现reverse state reconstruction、adjoint state和local derivative rules。
- 支持expression reverse、shared slots和Hermitian Pauli-sum objective。
- 增加TensorCircuit JAX/dense/finite-difference tests。

Acceptance gate：values/gradients匹配reference，memory不随depth存储全部states。

### P4：Integration and performance

- 在Python层实现`u1_circuit_from_tensorcircuit()`与`TensorCircuitU1Conversion`，覆盖required U1 gates并按first-appearance或显式`parameter_order`把direct symbols映射为parameter slots；Rust core不接触QIR dict。
- 运行setup/steady/gradient/wide benchmarks并profile。
- 更新README、typing、CHANGELOG、architecture和implementation status。

Acceptance gate：完整quality gates通过；至少一个代表性CPU workload显示native plan的明确用途；不以无证据的speedup宣称完成。

## 13. Non-goals

- Phase 6不实现public通用`Circuit` simulator、full-state `2**n` execution backend或tensor-network backend；common circuit IR本身不是模拟器。
- Phase 6不实现Taylor/Krylov/Chebyshev、ODE、time-dependent Hamiltonian或automatic Trotter compiler。
- 不实现noise channels、density matrix、mid-circuit measurement、sampling、RDM或entropy。
- 不实现full-state fallback、silent sector projection或non-conserving gate approximation。
- 不公开Givens、fSim、general U1 block、parameter-dependent arbitrary diagonal或multi-observable batch API。
- 不实现JAX custom call/VJP、TensorFlow op、GPU、distributed execution或backend tracing。
- 不要求复制TensorCircuit全部`AbstractCircuit` inheritance或所有QIR gates。
- 不改变现有`U1Sector` ordering、strict `restrict_u1()`或Phase 3/4 propagation语义。
- 不在Phase 6重写现有propagation `GateTape/ParameterRef`；common circuit IR作为新线路执行边界落地，后续只有在独立兼容迁移中才能统一旧API。

## 14. Closed decisions and implementation handoff

Phase 6没有剩余owner decision或optional implementation branch。Required terminal包含bounded `to_dense()`和`probability_full()`；static `diagonal()`严格要求unit modulus；Givens、fSim和public general block不进入首版；common layer只统一logical gate/parameter/serialization，不实现通用模拟器；execution只走U1-specific compiler和kernels。

实现者按P0→P4顺序推进。每个slice必须满足自己的acceptance gate并保留相应tests/benchmarks；不得用full-state fallback、per-gate FFI、Python callback、未记录的gate convention或forward-only nonunitary special case绕过合同。若依赖版本变化暴露新的TensorCircuit语义冲突，应暂停相关slice并以具体differential fixture更新本文，而不是在代码中静默选择另一语义。
