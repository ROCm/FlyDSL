"""Python ctypes wrapper for the OPUS GQA attention kernels.

Inputs use the same layout as the benchmark harness:
    q: [B, N, H, D]
    k: [B, N, H_KV, D]
    v: [B, N, H_KV, D]
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_LIB_PATH = _THIS_DIR / "build" / "libopus_gqa_attn.so"


def _load_lib():
    if not _LIB_PATH.is_file():
        raise RuntimeError(
            f"{_LIB_PATH} not found. Build/install it with: "
            f"cd {_THIS_DIR} && ./install_python.sh"
        )

    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.opus_gqa_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.opus_gqa_forward.restype = ctypes.c_int
    lib.opus_gqa_hip_error_string.argtypes = [ctypes.c_int]
    lib.opus_gqa_hip_error_string.restype = ctypes.c_char_p
    return lib


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = _load_lib()
    return _LIB


def _check_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA/ROCm tensor")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be torch.bfloat16")
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [B, N, H, D]")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def forward_out(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """Launch the OPUS GQA kernel into ``out`` and return ``out``."""
    _check_tensor("q", q)
    _check_tensor("k", k)
    _check_tensor("v", v)
    _check_tensor("out", out)

    if q.device != k.device or q.device != v.device or q.device != out.device:
        raise ValueError("q, k, v, and out must be on the same device")
    if q.shape != out.shape:
        raise ValueError(f"out shape {tuple(out.shape)} must match q shape {tuple(q.shape)}")
    if k.shape != v.shape:
        raise ValueError(f"k shape {tuple(k.shape)} must match v shape {tuple(v.shape)}")

    B, N, H, D = q.shape
    Bk, Nk, H_KV, Dk = k.shape
    if (Bk, Nk, Dk) != (B, N, D):
        raise ValueError("q/k/v must share B, N, and D")
    if H % H_KV != 0:
        raise ValueError(f"H ({H}) must be divisible by H_KV ({H_KV})")
    if D not in (128, 512):
        raise ValueError(f"OPUS GQA supports D=128 or D=512, got D={D}")

    err = _lib().opus_gqa_forward(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        int(B),
        int(N),
        int(H),
        int(H_KV),
        int(D),
        int(bool(causal)),
    )
    if err != 0:
        msg = _lib().opus_gqa_hip_error_string(err).decode("utf-8", errors="replace")
        raise RuntimeError(f"opus_gqa_forward failed: {msg} ({err})")
    return out


def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
    """Return ``OPUS_GQA(q, k, v)`` for bf16 contiguous tensors."""
    out = torch.empty_like(q)
    return forward_out(q, k, v, out, causal=causal)
