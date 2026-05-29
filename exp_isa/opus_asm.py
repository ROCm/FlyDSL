"""Python wrapper for exp_isa/flash_attn_opus.v1.s.

Inputs use the benchmark harness layout:
    q: [B, S, H, D]
    k: [B, S, H_KV, D]
    v: [B, S, H_KV, D]
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_CODE_OBJECT = _THIS_DIR / "flash_attn_opus.v1.co"


def _load_ext():
    if not _CODE_OBJECT.is_file():
        raise RuntimeError(
            f"{_CODE_OBJECT} not found. Build it with: cd {_THIS_DIR} && ./build.sh"
        )
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    return importlib.import_module("opus_asm_ext")


_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = _load_ext()
    return _EXT


def _check_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA/ROCm tensor")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be torch.bfloat16")
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [B, S, H, D]")
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
    """Launch the assembly kernel into ``out`` and return ``out``."""
    if not causal:
        raise ValueError("flash_attn_opus.v1.s is the causal kernel variant")

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

    b, s, h, d = q.shape
    bk, sk, h_kv, dk = k.shape
    if (bk, sk, dk) != (b, s, d):
        raise ValueError("q/k/v must share B, S, and D")
    if (h, h_kv, d) != (64, 64, 128):
        raise ValueError(
            f"flash_attn_opus.v1.s supports H=64, H_KV=64, D=128; got H={h}, H_KV={h_kv}, D={d}"
        )
    if s % 256 != 0:
        raise ValueError(f"seq_len must be divisible by 256, got {s}")

    _ext().forward_out(q, k, v, out, str(_CODE_OBJECT))
    return out


def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
    """Return ``flash_attn_opus.v1.s(q, k, v)`` for bf16 contiguous tensors."""
    out = torch.empty_like(q)
    return forward_out(q, k, v, out, causal=causal)
