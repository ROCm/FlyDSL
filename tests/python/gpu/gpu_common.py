"""Shared helpers for GPU python tests (softmax/layernorm/rmsnorm).

Keep this file **pure Python** (host-side only). Importing it must not change
kernel codegen or performance; it only de-duplicates common utilities.
"""

from __future__ import annotations

import numpy as np
import _mlir.extras.types as T


# Small helper: unwrap MLIR wrapper values into ir.Value
def unwrap(v):
    # Prefer the public `.value` interface used by FLIR wrapper types.
    if hasattr(v, "value"):
        return v.value
    if hasattr(v, "result"):
        return v.result
    return v


# Default epsilon used by norm operators.
EPS: float = 1e-5


# bf16 host packing helpers (store bf16 as uint16 payload)
def bf16_to_fp32_cpu(arr_bf16_uint16: np.ndarray) -> np.ndarray:
    arr_u32 = arr_bf16_uint16.astype(np.uint32) << 16
    return np.frombuffer(arr_u32.tobytes(), dtype=np.float32).reshape(arr_bf16_uint16.shape)


def fp32_to_bf16_rne_cpu(arr_fp32: np.ndarray) -> np.ndarray:
    """FP32 -> BF16 payload with round-to-nearest-even (RNE)."""
    u32 = np.frombuffer(arr_fp32.astype(np.float32).tobytes(), dtype=np.uint32).reshape(arr_fp32.shape)
    lsb = (u32 >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    u32_rounded = u32 + rounding_bias
    return (u32_rounded >> 16).astype(np.uint16)


def fp32_to_bf16_trunc_cpu(arr_fp32: np.ndarray) -> np.ndarray:
    """FP32 -> BF16 payload by truncation (drop low 16 bits)."""
    arr_u32 = np.frombuffer(arr_fp32.astype(np.float32).tobytes(), dtype=np.uint32)
    return (arr_u32 >> 16).astype(np.uint16).reshape(arr_fp32.shape)


def next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def dtype_to_elem_type(dtype_str: str):
    """Map string dtype to MLIR element type used by layernorm/rmsnorm tests."""
    if dtype_str == "f32":
        return T.f32()
    if dtype_str == "f16":
        return T.f16()
    if dtype_str == "bf16":
        return T.bf16()
    raise ValueError(f"unsupported dtype: {dtype_str}")


