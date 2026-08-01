# P0 Reference Vectors

状态：已实现。本文档记录 Phase 1 所有小系统 differential tests 使用的独立 oracle 和固定回归边界。

`tests/reference.py` 是唯一 reference implementation。它只使用 IEEE complex128 的 I/X/Y/Z 2×2 matrices、`numpy.kron` 和显式 local product table；它不导入 `tencirpauli`、PyO3 或 Rust，因此不会从被测实现生成 expected values。

固定语义来源是 [`semantics.md`](semantics.md)：外部 code 为 `0=I, 1=X, 2=Y, 3=Z`，结构位置 `q` 表示 qubit `q`，矩阵中 qubit 0 是 MSB，独立 reference packed conversion 中 qubit 0 是 LSB。`tests/test_numpy_reference.py` 固定覆盖完整 16 项单比特乘法表、`X*Y=iZ`、`Y*X=-iZ`、adjoint、commutation、support、首尾 qubit 的 X/Y/Z、零比特 identity、duplicate aggregation、exact cancellation、非法 code/shape/nqubits/mask overflow，以及 seed `20260801` 下的 `n<=6` 随机 cases。

数值比较默认使用 `np.complex128`；exact regression vectors 使用 `np.testing.assert_array_equal`，随机 differential tests 使用 `np.testing.assert_allclose`，当前 reference 最大系统为 6 qubits。reference matrix 的 dimension 检查遵守 `2**nqubits`，负 qubit 数和超出 mask 的 bit 直接抛出 `ValueError`，不静默截断。
