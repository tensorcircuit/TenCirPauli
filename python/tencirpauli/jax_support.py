"""Lazy JAX callback/VJP bridge for scalar circuit objectives."""

from __future__ import annotations

import importlib
from typing import Any, Iterable, Optional, cast

import numpy as np


def _require_jax() -> Any:
    try:
        jax = cast(Any, importlib.import_module("jax"))
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "expectation_jax() requires the optional 'jax' dependency"
        ) from error
    if not bool(jax.config.read("jax_enable_x64")):
        raise ValueError(
            "expectation_jax() requires jax_enable_x64=True for float64 callbacks"
        )
    return jax


def native_expectation_jax(
    angles: Iterable[object],
    engine: Any,
    *,
    checkpoint_interval: Optional[int] = None,
) -> Any:
    """Stage one native value-and-gradient callback with a custom VJP."""
    jax = _require_jax()
    jnp = jax.numpy
    angle_values = tuple(angles)
    count = len(angle_values)
    if count:
        values = jnp.stack(
            [jnp.asarray(angle, dtype=jnp.float64) for angle in angle_values], axis=0
        )
    else:
        values = jnp.zeros((0,), dtype=jnp.float64)
    value_spec = jax.ShapeDtypeStruct((), jnp.float64)
    gradient_spec = jax.ShapeDtypeStruct((count,), jnp.float64)

    def callback(runtime_angles: Any) -> tuple[np.float64, np.ndarray[Any, Any]]:
        concrete = np.ascontiguousarray(np.asarray(runtime_angles, dtype=np.float64))
        if checkpoint_interval is None:
            result = engine.value_and_grad(concrete)
        else:
            result = engine.value_and_grad(
                concrete, checkpoint_interval=checkpoint_interval
            )
        gradient = np.ascontiguousarray(
            np.asarray(result.gradient, dtype=np.float64).reshape((count,))
        )
        return np.float64(result.value), gradient

    def call_native(runtime_angles: Any) -> tuple[Any, Any]:
        return cast(
            tuple[Any, Any],
            jax.pure_callback(
                callback,
                (value_spec, gradient_spec),
                runtime_angles,
            ),
        )

    @jax.custom_vjp  # type: ignore[untyped-decorator]
    def native_scalar(runtime_angles: Any) -> Any:
        value, _ = call_native(runtime_angles)
        return value

    def forward(runtime_angles: Any) -> tuple[Any, Any]:
        value, gradient = call_native(runtime_angles)
        return value, gradient

    def backward(gradient: Any, cotangent: Any) -> tuple[Any]:
        return (cotangent * gradient,)

    native_scalar.defvjp(forward, backward)
    return native_scalar(values)


__all__ = ["native_expectation_jax"]
