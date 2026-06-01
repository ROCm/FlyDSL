"""Python wrapper for MI350 256x64 FMHA assembly kernels.

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
_CAUSAL_CODE_OBJECT = _THIS_DIR / "fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0.co"
_NONCAUSAL_CODE_OBJECT = _THIS_DIR / "fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk0_gm0.co"


def _code_object(causal: bool) -> Path:
    return _CAUSAL_CODE_OBJECT if causal else _NONCAUSAL_CODE_OBJECT


def _load_ext():
    missing = [path for path in (_CAUSAL_CODE_OBJECT, _NONCAUSAL_CODE_OBJECT) if not path.is_file()]
    if missing:
        missing_s = ", ".join(str(path) for path in missing)
        raise RuntimeError(f"{missing_s} not found. Build them with: cd {_THIS_DIR} && ./build.sh")
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    return importlib.import_module("fmha_asm_ext")


_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = _load_ext()
    return _EXT


def _check_tensor(name: str, tensor: torch.Tensor, dtype: torch.dtype) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA/ROCm tensor")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}")
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
    """Launch the causal or non-causal assembly kernel into ``out`` and return ``out``."""
    _check_tensor("q", q, torch.bfloat16)
    _check_tensor("k", k, torch.bfloat16)
    _check_tensor("v", v, torch.bfloat16)
    _check_tensor("out", out, torch.bfloat16)

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or out.ndim != 4:
        raise ValueError("q, k, v, and out must be 4D tensors")
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
    if d != 128 or h_kv <= 0 or h % h_kv != 0:
        raise ValueError(
            f"MI350 fmha asm supports H % H_KV == 0, D=128; got H={h}, H_KV={h_kv}, D={d}"
        )
    if causal and h % 8 != 0:
        raise ValueError(f"causal MI350 fmha asm requires H to be a multiple of 8, got H={h}")
    if s % 256 != 0:
        raise ValueError(f"seq_len must be divisible by 256, got {s}")

    lse = torch.empty((b, h, s), dtype=torch.float32, device=q.device)
    _ext().forward_out(q, k, v, out, lse, bool(causal), str(_code_object(causal)))
    return out


def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
    """Return the assembly FMHA output for bf16 contiguous tensors."""
    out = torch.empty_like(q)
    return forward_out(q, k, v, out, causal=causal)
