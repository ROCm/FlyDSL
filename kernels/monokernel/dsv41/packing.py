# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Host-side layouts of the V4.1 weights the gfx1250 mono kernels read, and the
checks that refuse a tensor laid out any other way.

The MI350 mono kernels read the model's parameters in place and checked only
shape and dtype, so a weight in another layout was read wrong without an error.
Here every packed tensor carries its layout id, and ``validate_*`` decodes tiles
back against the logical tensor.

Dense FP8 (group32: E4M3 with one E8M0 code a 32 x 32 block): the weight in
``shuffle_weight_gfx1250`` order (``dsv41.gemv``), the block scales row-major
(N / 32, K / 32) bytes as the checkpoint stores them.
"""

from dataclasses import dataclass

import torch

FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu
BLOCK = 32

DENSE_FP8_GFX1250 = "dense_fp8_g32/shuffle_gfx1250/v1"


def shuffle_gfx1250(w: torch.Tensor) -> torch.Tensor:
    """(N, K) bytes -> (N / 16, K * 16): ``view(N/16, 16, K/32, 2, 16)
    .permute(0, 2, 3, 1, 4)``, i.e. ``aiter.ops.shuffle.shuffle_weight_gfx1250``."""
    n, k = w.shape
    if n % 16 or k % 32:
        raise ValueError(f"shuffle_gfx1250 needs N % 16 == 0 and K % 32 == 0, got {n} x {k}")
    b = w.view(torch.uint8)
    return b.view(n // 16, 16, k // 32, 2, 16).permute(0, 2, 3, 1, 4).contiguous().view(n // 16, k * 16)


def unshuffle_gfx1250(p: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Inverse of ``shuffle_gfx1250`` -> (N, K) bytes."""
    return p.view(torch.uint8).view(n // 16, k // 32, 2, 16, 16).permute(0, 3, 1, 2, 4).contiguous().view(n, k)


@dataclass(frozen=True)
class DenseFp8:
    """A packed group32 FP8 linear: ``weight`` (N / 16, K * 16) uint8 and ``scale``
    (N / 32, K / 32) uint8 E8M0 codes."""

    n: int
    k: int
    weight: torch.Tensor
    scale: torch.Tensor
    layout: str = DENSE_FP8_GFX1250


def pack_dense_fp8(weight: torch.Tensor, scale: torch.Tensor) -> DenseFp8:
    """A checkpoint group32 linear (``weight`` (N, K) E4M3, ``scale`` (N / 32,
    K / 32) E8M0) -> the gfx1250 GEMV layout."""
    n, k = weight.shape
    if weight.dtype not in (FP8, torch.uint8):
        raise TypeError(f"dense FP8 weight dtype {weight.dtype}")
    if tuple(scale.shape) != (n // BLOCK, k // BLOCK):
        raise ValueError(f"scale {tuple(scale.shape)} for a {n} x {k} weight, expected {(n // BLOCK, k // BLOCK)}")
    if k % 128 and k % 32:
        raise ValueError(f"K {k} is not a multiple of 32")
    return DenseFp8(
        n, k, shuffle_gfx1250(weight.contiguous()), scale.contiguous().view(torch.uint8).clone()
    )


class LayoutError(ValueError):
    """A packed tensor not in the layout the kernels read."""


# ---------------------------------------------------------------- MoE MXFP4
MOE_FP4_GFX1250 = "moe_mxfp4/aiter_gfx1250_shuffle16x16_n32k4/v1"


def interleave_gate_up(t: torch.Tensor) -> torch.Tensor:
    """(E, 2 I, ...) gate rows then up rows -> GUGU: row 2 j gate j, 2 j + 1 up j
    (aiter ``interleave_gate_up_rows``)."""
    i = t.shape[1] // 2
    return torch.stack([t[:, :i], t[:, i:]], dim=2).flatten(1, 2)


def shuffle_moe_fp4(codes: torch.Tensor) -> torch.Tensor:
    """(E, N, K) E2M1 nibbles -> (E, N * K / 2) bytes: packed two a byte (even K
    low), then aiter ``shuffle_weight`` (16, 16): ``view(E, N/16, 16, Kb/32, 2,
    16).permute(0, 1, 3, 4, 2, 5)``."""
    e, n, k = codes.shape
    if n % 16 or k % 64:
        raise ValueError(f"MoE FP4 needs N % 16 == 0 and K % 64 == 0, got {n} x {k}")
    b = (codes[..., 0::2] & 0xF) | ((codes[..., 1::2] & 0xF) << 4)
    kb = k // 2
    return b.view(e, n // 16, 16, kb // 32, 2, 16).permute(0, 1, 3, 4, 2, 5).contiguous().view(e, n * kb)


def unshuffle_moe_fp4(p: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Inverse of ``shuffle_moe_fp4`` -> (E, N, K) nibbles."""
    e, kb = p.shape[0], k // 2
    b = p.view(e, n // 16, kb // 32, 2, 16, 16).permute(0, 1, 4, 2, 3, 5).contiguous().view(e, n, kb)
    out = torch.empty(e, n, k, dtype=torch.uint8, device=p.device)
    out[..., 0::2], out[..., 1::2] = b & 0xF, b >> 4
    return out


def shuffle_scale_n32k4(s: torch.Tensor) -> torch.Tensor:
    """(E, N, K / 32) E8M0 -> (E, N / 32 * K): aiter ``shuffle_scale_n32k4``,
    ``view(E, N/32, 32, ks/4, 4).permute(0, 1, 3, 2, 4)``: a row's four codes of a
    128-K step one dword, a 32-row block's dwords of a step together."""
    e, n, ks = s.shape
    if n % 32 or ks % 4:
        raise ValueError(f"n32k4 needs N % 32 == 0 and (K / 32) % 4 == 0, got {n} x {ks}")
    return s.view(e, n // 32, 32, ks // 4, 4).permute(0, 1, 3, 2, 4).contiguous().view(e, n * ks)


def unshuffle_scale_n32k4(p: torch.Tensor, n: int, ks: int) -> torch.Tensor:
    e = p.shape[0]
    return p.view(e, n // 32, ks // 4, 32, 4).permute(0, 1, 3, 2, 4).contiguous().view(e, n, ks)


@dataclass(frozen=True)
class MoeFp4:
    """Packed routed-expert weights: ``weight`` (E, N K / 2) uint8, ``scale``
    (E, N K / 32) uint8; ``gugu``: rows gate / up interleaved (w13)."""

    n: int
    k: int
    gugu: bool
    weight: torch.Tensor
    scale: torch.Tensor
    layout: str = MOE_FP4_GFX1250


def pack_moe_fp4(codes: torch.Tensor, scales: torch.Tensor, gugu: bool) -> MoeFp4:
    """Logical (E, N, K) nibbles and (E, N, K / 32) codes (w13: gate rows then up
    rows) -> aiter's gfx1250 A8W4 layout (``moe_shuffle_weight`` /
    ``moe_shuffle_scale`` with ``is_guinterleave=gugu``)."""
    e, n, k = codes.shape
    if tuple(scales.shape) != (e, n, k // BLOCK):
        raise ValueError(f"scales {tuple(scales.shape)} for {e} x {n} x {k}")
    c, s = codes.view(torch.uint8), scales.view(torch.uint8)
    if gugu:
        c, s = interleave_gate_up(c), interleave_gate_up(s)
    return MoeFp4(n, k, gugu, shuffle_moe_fp4(c.contiguous()), shuffle_scale_n32k4(s.contiguous()))


def validate_moe_fp4(p: MoeFp4, codes: torch.Tensor | None = None, scales: torch.Tensor | None = None,
                     experts: int = 2) -> None:
    """Refuse ``p`` unless it is in ``MOE_FP4_GFX1250``; with the logical tensors,
    decode ``experts`` random experts back and compare."""
    if p.layout != MOE_FP4_GFX1250:
        raise LayoutError(f"layout {p.layout!r}, kernels read {MOE_FP4_GFX1250!r}")
    e = p.weight.shape[0]
    if p.weight.dtype != torch.uint8 or tuple(p.weight.shape) != (e, p.n * p.k // 2):
        raise LayoutError(f"weight {p.weight.dtype} {tuple(p.weight.shape)}")
    if p.scale.dtype != torch.uint8 or tuple(p.scale.shape) != (e, p.n * p.k // 32):
        raise LayoutError(f"scale {p.scale.dtype} {tuple(p.scale.shape)}")
    if not (p.weight.is_contiguous() and p.scale.is_contiguous()):
        raise LayoutError("packed tensors must be contiguous")
    if codes is None:
        return
    gen = torch.Generator().manual_seed(0)
    for x in torch.randint(0, e, (experts,), generator=gen).tolist():
        w = unshuffle_moe_fp4(p.weight[x : x + 1].cpu(), p.n, p.k)
        s = unshuffle_scale_n32k4(p.scale[x : x + 1].cpu(), p.n, p.k // 32)
        want_c, want_s = codes[x : x + 1].cpu().view(torch.uint8), scales[x : x + 1].cpu().view(torch.uint8)
        if p.gugu:
            want_c, want_s = interleave_gate_up(want_c), interleave_gate_up(want_s)
        if not torch.equal(w, want_c & 0xF) or not torch.equal(s, want_s):
            raise LayoutError(f"expert {x} does not decode to the logical weight")


def validate_dense_fp8(p: DenseFp8, weight: torch.Tensor | None = None, scale: torch.Tensor | None = None,
                       samples: int = 8) -> None:
    """Refuse ``p`` unless it is in ``DENSE_FP8_GFX1250`` with consistent shapes,
    dtypes and contiguity; with the logical tensors, also decode ``samples`` random
    16-row tiles back and compare bytes."""
    if p.layout != DENSE_FP8_GFX1250:
        raise LayoutError(f"layout {p.layout!r}, kernels read {DENSE_FP8_GFX1250!r}")
    if p.weight.dtype != torch.uint8 or tuple(p.weight.shape) != (p.n // 16, p.k * 16):
        raise LayoutError(f"weight {p.weight.dtype} {tuple(p.weight.shape)}, expected uint8 {(p.n // 16, p.k * 16)}")
    if p.scale.dtype != torch.uint8 or tuple(p.scale.shape) != (p.n // BLOCK, p.k // BLOCK):
        raise LayoutError(f"scale {p.scale.dtype} {tuple(p.scale.shape)}")
    if not (p.weight.is_contiguous() and p.scale.is_contiguous()):
        raise LayoutError("packed tensors must be contiguous")
    if p.weight.is_cuda and p.weight.data_ptr() % 128:
        raise LayoutError("packed weight must be 128-byte aligned (TDM direct copy, b128 loads)")
    if weight is None:
        return
    logical = weight.contiguous().view(torch.uint8)
    gen = torch.Generator().manual_seed(0)
    for rg in torch.randint(0, p.n // 16, (samples,), generator=gen).tolist():
        got = p.weight[rg].cpu().view(p.k // 32, 2, 16, 16).permute(2, 0, 1, 3).reshape(16, p.k)
        if not torch.equal(got, logical[16 * rg : 16 * rg + 16].cpu()):
            raise LayoutError(f"row group {rg} does not decode to the logical weight")
    if scale is not None and not torch.equal(p.scale.cpu(), scale.contiguous().view(torch.uint8).cpu()):
        raise LayoutError("block scales differ from the logical scales")
