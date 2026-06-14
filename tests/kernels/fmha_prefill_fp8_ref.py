# SPDX-License-Identifier: Apache-2.0
"""Reference + paged-cache packing helpers for the FP8 FMHA prefill port.

This module is self-contained (torch only, runs on CPU) so the reference and the
physical paged-cache packing can be validated independently of FlyDSL / the GPU.

Feature parity with the source PyISA kernel
``f8_fmha_prefill_gfx942_hd128_qkptph_vph_paged_vkcolv``:

* FP8 e4m3 **FNUZ** Q/K/V, head dim 128 (QK and V), bf16 output, causal mask.
* GQA: ``nhead_q = gqa * nhead_k``; q head ``h`` uses kv head ``h // gqa``.
* Per-token-per-head Q/K descale; per-head V descale.
* Paged KV with ``vec_k_col_v`` physical layout:
  - K pool: ``[pages, nhead_k, hd/16, page_size, 16]``
  - V pool: ``[pages, page_size, nhead_k, hd]`` (row-major within page)
  block table ``[batch, pages_per_batch]`` of physical page ids + ``kv_indptr``.
* p_scale: in the kernel this only rescales the fp8 probabilities P to keep them in
  range, and cancels exactly between the softmax numerator and denominator, so it
  does NOT change the math the full-precision reference computes (intentionally omitted
  here; the kernel applies and divides it back out internally).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

VEC_X = 16  # fp8 vector width for vec_k_col_v K layout
FP8_DTYPE = torch.float8_e4m3fnuz
NEG = float("-3.4e38")


# ---------------------------------------------------------------------------
# Quantization (logical float -> per-token/head fp8 + scales)
# ---------------------------------------------------------------------------
def _fp8_absmax() -> float:
    return float(torch.finfo(FP8_DTYPE).max)


def quantize_per_token_head(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x[b, s, h, d]`` to fp8 with a per-(b,s,h) scale.

    Returns ``(x_fp8[b,s,h,d], scale[b,h,s])`` where ``descale = scale`` is the
    multiplier that maps fp8 back to the original magnitude:
    ``x ~= x_fp8.float() * scale[b,h,s]``.
    """
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)  # [b,s,h,1]
    scale = amax / _fp8_absmax()  # [b,s,h,1]
    x_fp8 = (x / scale).to(FP8_DTYPE)
    descale = scale.squeeze(-1).permute(0, 2, 1).contiguous()  # [b,h,s]
    return x_fp8, descale


def quantize_per_head(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x[b, s, h, d]`` to fp8 with a per-(b,h) scale (used for V).

    Returns ``(x_fp8[b,s,h,d], scale[b,h])``.
    """
    amax = x.abs().amax(dim=(1, 3)).clamp_min(1e-8)  # [b,h]
    scale = amax / _fp8_absmax()  # [b,h]
    x_fp8 = (x / scale[:, None, :, None]).to(FP8_DTYPE)
    return x_fp8, scale.contiguous()


# ---------------------------------------------------------------------------
# Paged physical layout packing
# ---------------------------------------------------------------------------
@dataclass
class PagedCache:
    k_pool: torch.Tensor  # uint8 [total_pages, nk, hd/16, page_size, 16]
    v_pool: torch.Tensor  # uint8 [total_pages, page_size, nk, hd]
    block_table: torch.Tensor  # int32 [batch, pages_per_batch]
    kv_indptr: torch.Tensor  # int32 [batch+1]  (prefix sum of pages per batch)
    page_ids: torch.Tensor  # int32 [total_pages]  flat LTD (physical page per slot)
    page_size: int
    k_page_stride: int  # elements per K page = page_size * nk * hd
    v_page_stride: int  # elements per V page = page_size * nk * hd


def pack_paged_cache(
    k_fp8: torch.Tensor,  # [b, sk, nk, hd]
    v_fp8: torch.Tensor,  # [b, sk, nk, hd]
    page_size: int,
    scatter: bool = False,
    seed: int = 42,
    v_col: bool = False,
) -> PagedCache:
    """Pack contiguous per-batch K/V fp8 into the kernel's physical paged pools.

    Mirrors the host packing in ``tests/fmha/src/bench/fmha_bench.cpp``:
      * K is rearranged to ``vec_k_col_v``: ``[pages, nk, hd/16, page_size, 16]``.
      * V default: row-major within a page ``[pages, page_size, nk, hd]``.
      * ``v_col=True``: **column-major V** ``[pages, nk, hd, page_size]`` (kv/page_size
        contiguous per (head, d)) — matches CK-Tile's true ``vec_k_col_v`` V, which is
        already GEMM2-ready (the contraction dim ``kv`` is contiguous) so the kernel needs
        **no V transpose**. This is what lets CK avoid the gfx942 LDS-transpose DS-wait.
      * Pages for batch ``b`` occupy LTD slots ``[b*ppb, (b+1)*ppb)``; ``scatter``
        shuffles which physical pool page each slot maps to (gather-via-LTD).
    """
    b, sk, nk, hd = k_fp8.shape
    assert hd % VEC_X == 0 and v_fp8.shape == (b, sk, nk, hd)
    ppb = (sk + page_size - 1) // page_size  # pages per batch
    total_pages = b * ppb

    kv_indptr = torch.arange(0, b + 1, dtype=torch.int32) * ppb

    # LTD: slot -> physical page id. Identity, optionally shuffled per batch.
    page_ids = torch.arange(total_pages, dtype=torch.int32)
    if scatter:
        g = torch.Generator().manual_seed(seed)
        for bi in range(b):
            seg = page_ids[bi * ppb : (bi + 1) * ppb]
            page_ids[bi * ppb : (bi + 1) * ppb] = seg[torch.randperm(ppb, generator=g)]

    k_u8 = k_fp8.view(torch.uint8)
    v_u8 = v_fp8.view(torch.uint8)
    k_pool = torch.zeros(total_pages, nk, hd // VEC_X, page_size, VEC_X, dtype=torch.uint8)
    if v_col:
        v_pool = torch.zeros(total_pages, nk, hd, page_size, dtype=torch.uint8)
    else:
        v_pool = torch.zeros(total_pages, page_size, nk, hd, dtype=torch.uint8)

    for bi in range(b):
        for t in range(sk):
            slot = bi * ppb + t // page_size
            phys = int(page_ids[slot])
            tok = t % page_size
            # K: [nk, hd/16, 16] for this token -> [phys, nk, cg, tok, xi]
            kt = k_u8[bi, t]  # [nk, hd]
            k_pool[phys, :, :, tok, :] = kt.view(nk, hd // VEC_X, VEC_X)
            if v_col:
                # column-major V: [phys, nk, hd, tok]
                v_pool[phys, :, :, tok] = v_u8[bi, t]  # [nk, hd]
            else:
                # V row-major: [phys, tok, nk, hd]
                v_pool[phys, tok] = v_u8[bi, t]

    block_table = torch.empty(b, ppb, dtype=torch.int32)
    for bi in range(b):
        block_table[bi] = page_ids[bi * ppb : (bi + 1) * ppb]

    return PagedCache(
        k_pool=k_pool,
        v_pool=v_pool,
        block_table=block_table,
        kv_indptr=kv_indptr,
        page_ids=page_ids,
        page_size=page_size,
        k_page_stride=page_size * nk * hd,
        v_page_stride=page_size * nk * hd,
    )


def gather_from_paged(cache: PagedCache, b: int, sk: int, nk: int, hd: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`pack_paged_cache`: gather logical fp8 K/V back.

    Returns ``(k_fp8[b,sk,nk,hd], v_fp8[b,sk,nk,hd])``. Used to round-trip-test the
    packer and to drive the reference from exactly what the kernel will read.
    """
    ps = cache.page_size
    k = torch.zeros(b, sk, nk, hd, dtype=torch.uint8)
    v = torch.zeros(b, sk, nk, hd, dtype=torch.uint8)
    ppb = cache.block_table.shape[1]
    for bi in range(b):
        for t in range(sk):
            phys = int(cache.block_table[bi, t // ps])
            tok = t % ps
            k[bi, t] = cache.k_pool[phys, :, :, tok, :].reshape(nk, hd)
            v[bi, t] = cache.v_pool[phys, tok]
    return k.view(FP8_DTYPE), v.view(FP8_DTYPE)


# ---------------------------------------------------------------------------
# Reference attention (full precision, from dequantized fp8)
# ---------------------------------------------------------------------------
def fmha_prefill_reference(
    q_fp8: torch.Tensor,  # [b, sq, nq, hd]
    k_fp8: torch.Tensor,  # [b, sk, nk, hd]
    v_fp8: torch.Tensor,  # [b, sk, nk, hd]
    q_descale: torch.Tensor,  # [b, nq, sq]
    k_descale: torch.Tensor,  # [b, nk, sk]
    v_descale: torch.Tensor,  # [b, nk]
    sm_scale: float,
    causal: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Golden causal FP8 prefill attention. Returns ``O[b, sq, nq, hd]``."""
    b, sq, nq, hd = q_fp8.shape
    _, sk, nk, _ = k_fp8.shape
    gqa = nq // nk

    q = q_fp8.float() * q_descale.permute(0, 2, 1).unsqueeze(-1)  # [b,sq,nq,hd]
    k = k_fp8.float() * k_descale.permute(0, 2, 1).unsqueeze(-1)  # [b,sk,nk,hd]
    v = v_fp8.float() * v_descale[:, None, :, None]  # [b,sk,nk,hd]

    k = k.repeat_interleave(gqa, dim=2)  # [b,sk,nq,hd]
    v = v.repeat_interleave(gqa, dim=2)

    # scores [b, nq, sq, sk]
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * sm_scale
    if causal:
        qpos = torch.arange(sq).unsqueeze(1)
        kpos = torch.arange(sk).unsqueeze(0)
        mask = qpos + (sk - sq) < kpos  # disallow key in the future
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), NEG)

    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", probs, v)
    return out.to(out_dtype)
