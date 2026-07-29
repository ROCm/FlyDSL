# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared helpers for the FlyDSL puzzles.

Kept intentionally small and dependency-light so each puzzle file is
self-contained: it only needs ``torch`` (for reference outputs and result
checking) plus ``flydsl``.
"""

from __future__ import annotations

import torch

# All puzzle reference solutions target CDNA MFMA (gfx942 / gfx950), wave size 64.
WAVE_SIZE = 64


def cuda() -> torch.device:
    """The ROCm/HIP device torch exposes as 'cuda'."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU visible to torch. The FlyDSL puzzles require an AMD CDNA GPU "
            "(gfx942/gfx950)."
        )
    return torch.device("cuda")


def new_stream() -> torch.cuda.Stream:
    return torch.cuda.Stream()


def check(got: torch.Tensor, ref: torch.Tensor, *, atol=1e-2, rtol=1e-2, name="result") -> bool:
    """Print a PASS/FAIL report and return whether the tensors match.

    Uses relaxed tolerances by default because MFMA accumulation and fast-math
    ``exp2`` differ slightly from torch's fp32 reference.
    """
    got_f = got.float()
    ref_f = ref.float()
    ok = torch.allclose(got_f, ref_f, atol=atol, rtol=rtol)
    if ok:
        print(f"PASS: {name} matches reference (atol={atol}, rtol={rtol})")
        return True

    diff = (got_f - ref_f).abs()
    max_diff = diff.max().item()
    n_bad = (~torch.isclose(got_f, ref_f, atol=atol, rtol=rtol)).sum().item()
    total = ref_f.numel()
    print(f"FAIL: {name} differs from reference")
    print(f"  max |diff| = {max_diff:.4e}")
    print(f"  mismatched = {n_bad}/{total} ({100.0 * n_bad / total:.2f}%)")
    print(f"  got  (flat[:8]) = {got_f.flatten()[:8].tolist()}")
    print(f"  ref  (flat[:8]) = {ref_f.flatten()[:8].tolist()}")
    return ok


# ---------------------------------------------------------------------------
# Reference implementations (torch) for each puzzle.
# ---------------------------------------------------------------------------


def ref_copy(a: torch.Tensor) -> torch.Tensor:
    return a.clone()


def ref_vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


def ref_scale_bias(a: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    return a * alpha + beta


def ref_transpose(a: torch.Tensor) -> torch.Tensor:
    return a.t().contiguous()


def ref_row_sum(a: torch.Tensor) -> torch.Tensor:
    return a.sum(dim=1)


def ref_softmax(a: torch.Tensor) -> torch.Tensor:
    return torch.softmax(a.float(), dim=1)


def ref_rmsnorm(a: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a_f = a.float()
    var = a_f.pow(2).mean(dim=1, keepdim=True)
    out = a_f * torch.rsqrt(var + eps) * gamma.float()
    return out


def ref_gemm(a: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
    """C = A @ B^T where B is stored row-major as (N, K)."""
    return a.float() @ b_t.float().t()


def ref_flash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    """Reference scaled dot-product attention for one head.

    q: (S, D), k: (S, D), v: (S, D). Returns (S, D).
    """
    d = q.shape[-1]
    scale = 1.0 / (d ** 0.5)
    scores = (q.float() @ k.float().t()) * scale
    if causal:
        s = scores.shape[0]
        mask = torch.triu(torch.ones(s, s, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return probs @ v.float()


def ref_conv2d(
    x: torch.Tensor, w: torch.Tensor, stride: int = 1, padding: int = 0
) -> torch.Tensor:
    """Reference 2D convolution. x: (N,C,H,W), w: (K,C,R,S). Returns (N,K,Ho,Wo)."""
    return torch.nn.functional.conv2d(x.float(), w.float(), stride=stride, padding=padding)
