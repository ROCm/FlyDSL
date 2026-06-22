# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""High-level FlyDSL Flash Attention API for gfx950 / gfx942.

Wraps ``flash_attn_generic.build_flash_attn_func_module`` (gfx942-compatible,
dense self/cross-attention) and ``flash_attn_gfx950.build_flash_attn_dualwave_swp_module``
(gfx950 DUALWAVE_SWP, varlen + split-K) behind a single function:

    ``flydsl_flash_attn_func(q, k, v, ...)``

Key features vs calling build_* directly:
- ``@functools.lru_cache`` on the build call so repeated invocations with the
  same (static) config compile only once per process.
- Explicit ``max_seqlen_q`` / ``cross_seqlen`` controls for varlen builds.
- split-K fp32 workspace allocation, zeroing, and the 4 GiB descriptor guard.
- Unified device / stream context (``torch.cuda.device`` + current stream).
- Validates shapes, dtypes, and arch before compiling.
- Accepts ``debug_counts`` tensor to enable the lazy-rescale branch counter
  (gfx950 DUALWAVE_SWP dualwave_swp_debug_lazy_counts=True path).
"""

from __future__ import annotations

import functools
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: F401  (imported for callers' convenience)

# Re-export so callers only need to import from this module.
from kernels.flash_attn_gfx950 import dualwave_splitk_workspace_elems  # noqa: F401

__all__ = ["flydsl_flash_attn_func", "dualwave_splitk_workspace_elems"]

_DTYPE_MAP = {torch.bfloat16: "bf16", torch.float16: "f16"}


def _dtype_str(t: torch.Tensor) -> str:
    s = _DTYPE_MAP.get(t.dtype)
    if s is None:
        raise ValueError(f"flydsl_flash_attn_func only supports bf16/f16, got {t.dtype!r}")
    return s


def _gpu_arch(device: torch.device) -> str:
    try:
        return torch.cuda.get_device_properties(device.index).gcnArchName.split(":")[0]
    except Exception:
        return ""


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _paged_kv_page_size(k: torch.Tensor, kv_cache_layout: str) -> int:
    if kv_cache_layout == "linear":
        return int(k.shape[1])
    if kv_cache_layout == "linear3d":
        return 1
    if kv_cache_layout == "vectorized":
        return int(k.shape[3])
    raise ValueError(f"flydsl_flash_attn_func: unsupported kv_cache_layout={kv_cache_layout}")


def _logical_kv_from_pages(
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    kv_cache_layout: str,
    kv_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kv_cache_layout in ("linear", "linear3d"):
        kb = k_pages.reshape(-1, k_pages.shape[-2], k_pages.shape[-1])
        vb = v_pages.reshape(-1, v_pages.shape[-2], v_pages.shape[-1])
    elif kv_cache_layout == "vectorized":
        kb = (
            k_pages.permute(0, 3, 1, 2, 4)
            .contiguous()
            .reshape(-1, k_pages.shape[1], k_pages.shape[2] * k_pages.shape[4])
        )
        vb = v_pages.permute(0, 2, 4, 1, 3).contiguous().reshape(-1, v_pages.shape[1], v_pages.shape[3])
    else:
        raise ValueError(f"flydsl_flash_attn_func: unsupported kv_cache_layout={kv_cache_layout}")
    return kb[:kv_len].contiguous(), vb[:kv_len].contiguous()


def _materialize_paged_kv_for_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor],
    kv_indptr: Optional[torch.Tensor],
    kv_indices: Optional[torch.Tensor],
    kv_last_page_lens: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    seqlen_k: Optional[torch.Tensor],
    kv_cache_layout: str,
    kv_lookup_table: str,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[int]]:
    """Temporary host-side ABI bridge until a native paged-KV FlyDSL kernel exists."""
    page_size = _paged_kv_page_size(k, kv_cache_layout)
    dense_q = q.dim() == 4
    batch = q.shape[0] if dense_q else int(cu_seqlens_q.numel() - 1)

    k_batches: list[torch.Tensor] = []
    v_batches: list[torch.Tensor] = []
    kv_lens: list[int] = []
    for b in range(batch):
        if kv_lookup_table == "vllm":
            if block_table is None or seqlen_k is None:
                raise ValueError("flydsl_flash_attn_func: vllm paged KV requires block_table and seqlen_k")
            kv_len = int(seqlen_k[b].item())
            page_ids = block_table[b, : _ceil_div(kv_len, page_size)].to(device=k.device, dtype=torch.long)
        elif kv_lookup_table == "sglang":
            if kv_indptr is None or kv_indices is None:
                raise ValueError("flydsl_flash_attn_func: sglang paged KV requires kv_indptr and kv_indices")
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            page_ids = kv_indices[start:end].to(device=k.device, dtype=torch.long)
            if kv_last_page_lens is None:
                kv_len = int(page_ids.numel()) * page_size
            else:
                kv_len = (int(page_ids.numel()) - 1) * page_size + int(kv_last_page_lens[b].item())
        else:
            raise ValueError(f"flydsl_flash_attn_func: unsupported kv_lookup_table={kv_lookup_table}")
        kb, vb = _logical_kv_from_pages(k[page_ids], v[page_ids], kv_cache_layout, kv_len)
        k_batches.append(kb)
        v_batches.append(vb)
        kv_lens.append(kv_len)

    if dense_q:
        if len(set(kv_lens)) != 1:
            raise ValueError("flydsl_flash_attn_func: dense paged-KV fallback requires equal seqlen_k per batch")
        return torch.stack(k_batches, dim=0), torch.stack(v_batches, dim=0), None, kv_lens[0]

    cu_kv = [0]
    for kv_len in kv_lens:
        cu_kv.append(cu_kv[-1] + kv_len)
    cu_seqlens_kv = torch.tensor(cu_kv, dtype=torch.int32, device=q.device)
    return torch.cat(k_batches, dim=0), torch.cat(v_batches, dim=0), cu_seqlens_kv, max(kv_lens)


# ── build-cache helpers ────────────────────────────────────────────────────


@functools.lru_cache(maxsize=256)
def _build_dense(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    cross_seqlen: bool,
    waves_per_eu: int,
    daz: bool,
    lazy_rescale: bool,
    setprio: bool,
    debug_lazy_counts: bool,
    enable_stagger: bool,
):
    """Build (and cache) a dense-mode launcher via the generic dispatch."""
    from kernels.flash_attn_generic import build_flash_attn_func_module

    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
        cross_seqlen=cross_seqlen,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=lazy_rescale,
        dualwave_swp_setprio=setprio,
        dualwave_swp_debug_lazy_counts=debug_lazy_counts,
        dualwave_swp_enable_stagger=enable_stagger,
    )


@functools.lru_cache(maxsize=256)
def _build_varlen(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    cross_seqlen: bool,
    waves_per_eu: int,
    daz: bool,
    lazy_rescale: bool,
    setprio: bool,
    debug_lazy_counts: bool,
    enable_stagger: bool,
):
    """Build (and cache) a varlen-mode launcher (gfx950 DUALWAVE_SWP, varlen=True)."""
    from kernels.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module

    return build_flash_attn_dualwave_swp_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
        varlen=True,
        cross_seqlen=cross_seqlen,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=lazy_rescale,
        dualwave_swp_setprio=setprio,
        dualwave_swp_debug_lazy_counts=debug_lazy_counts,
        dualwave_swp_enable_stagger=enable_stagger,
    )


@functools.lru_cache(maxsize=256)
def _build_splitk(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    num_kv_splits: int,
    waves_per_eu: int,
    daz: bool,
    lazy_rescale: bool,
    setprio: bool,
    enable_stagger: bool,
):
    """Build (and cache) a split-K launcher (gfx950 DUALWAVE_SWP, num_kv_splits>1)."""
    from kernels.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module

    return build_flash_attn_dualwave_swp_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
        num_kv_splits=num_kv_splits,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=lazy_rescale,
        dualwave_swp_setprio=setprio,
        dualwave_swp_enable_stagger=enable_stagger,
    )


# ── public API ─────────────────────────────────────────────────────────────


def flydsl_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    num_kv_heads: Optional[int] = None,
    # Varlen (packed cu_seqlens): pass both to enable the varlen path.
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    # Max per-batch Q seqlen (varlen only). Required for varlen to size grid_y
    # without synchronizing on cu_seqlens_q.
    max_seqlen_q: Optional[int] = None,
    # Max per-batch KV seqlen (varlen cross-attn only). Used to size the KV grid
    # when seqlen_q != seqlen_kv per batch.
    max_seqlen_kv: Optional[int] = None,
    # Whether per-batch Sq and Skv can differ. Dense mode infers this from shapes;
    # varlen mode requires it explicitly to choose the correct build variant.
    cross_seqlen: Optional[bool] = None,
    # Paged KV cache ABI (interface only for now; paged FlyDSL kernel support is
    # wired in a future implementation). When provided, k/v are physical KV cache
    # tensors and lookup-table metadata describes logical sequence order.
    kv_indptr: Optional[torch.Tensor] = None,
    kv_indices: Optional[torch.Tensor] = None,
    kv_last_page_lens: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    seqlen_k: Optional[torch.Tensor] = None,
    kv_cache_layout: str = "linear",
    kv_lookup_table: str = "vllm",
    # Split-K (gfx950 only, seq_len >= 384, D=128, bf16/f16).
    num_kv_splits: int = 1,
    # Output tensor; allocated if None.
    out: Optional[torch.Tensor] = None,
    # Kernel build options.
    waves_per_eu: int = 2,
    daz: bool = True,
    dualwave_swp_lazy_rescale: bool = True,
    dualwave_swp_setprio: bool = True,
    dualwave_swp_enable_stagger: bool = True,
    # Debug: pass a pre-allocated float32[2] tensor to enable the lazy-rescale
    # branch counter (dualwave_swp_debug_lazy_counts=True). Only for dense mode.
    debug_counts: Optional[torch.Tensor] = None,
    # CUDA/HIP stream; defaults to the current stream for q.device.
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Run FlyDSL Flash Attention (gfx950 DUALWAVE_SWP / gfx942 generic fallback).

    Args:
        q: Query tensor. Dense: ``[B, Sq, H, D]`` (BSHD).
           Varlen: ``[total_q, H, D]`` (packed, cu_seqlens_q required).
        k: Key tensor. Dense: ``[B, Skv, Hkv, D]``.
           Varlen: ``[total_kv, Hkv, D]``.
        v: Value tensor, same shape as k.
           Paged KV cache (future ABI): physical K/V cache tensors. Supported
           ``kv_cache_layout`` values:
           - ``linear``: 4D paged K/V, ``[NumBlocks, PageSize, NumKVHeads, HeadDim]``.
           - ``linear3d``: page_size=1 special case,
             ``[NumBlocks, NumKVHeads, HeadDim]``.
           - ``vectorized``: aiter-style 5D K/V, where
             ``K = [NumBlocks, NumKVHeads, HeadDim / kVectorSize, PageSize, kVectorSize]``
             and
             ``V = [NumBlocks, NumKVHeads, PageSize / kVectorSize, HeadDim, kVectorSize]``.
             Here ``kVectorSize = 16 / element_size`` (bf16/fp16: 8, fp8: 16);
             page_size and head_dim must be divisible by it.
        causal: Bottom-right aligned causal mask when True.
        num_kv_heads: KV head count for GQA/MQA; defaults to q num_heads (MHA).
        cu_seqlens_q: Int32 ``[B+1]`` cumulative Q token counts (varlen).
        cu_seqlens_kv: Int32 ``[B+1]`` cumulative KV token counts (varlen).
        max_seqlen_q: Maximum per-batch Q seqlen (varlen). Required in varlen mode.
        max_seqlen_kv: Maximum per-batch KV seqlen (varlen cross-attn). Required when
            seqlen_q != seqlen_kv per batch.
        cross_seqlen: Whether seqlen_q and seqlen_kv differ. Required in varlen mode;
            dense mode infers it from ``q.shape[1] != k.shape[1]``.
        kv_indptr / kv_indices: SGLang-style 1D page table metadata.
        block_table / seqlen_k: vLLM-style 2D block table metadata.
        kv_last_page_lens: Per-batch valid token count in the final KV page.
        kv_lookup_table: ``"sglang"`` or ``"vllm"``.
        num_kv_splits: Split-K factor (>1: gfx950 only, D=128, bf16/f16, seq>=384).
        out: Optional pre-allocated output tensor (same shape/dtype as q).
        waves_per_eu: Kernel occupancy hint.
        daz: Enable denormals-are-zero.
        dualwave_swp_lazy_rescale: Enable lazy online softmax rescale.
        dualwave_swp_setprio: Enable s_setprio scheduling hints.
        dualwave_swp_enable_stagger: Enable wave-group phase stagger.
        debug_counts: Float32[2] tensor; when given, counts lazy-rescale branches
            (debug_counts[0] = all-below-true, debug_counts[1] = all-below-false).
        stream: CUDA/HIP stream to launch on.

    Returns:
        Output tensor with same shape and dtype as q.
    """
    # ── validation ──────────────────────────────────────────────────────────
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_func: q/k/v must be CUDA tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(f"flydsl_flash_attn_func: q/k/v must share device; got {q.device}/{k.device}/{v.device}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"flydsl_flash_attn_func: q/k/v must share dtype; got {q.dtype}/{k.dtype}/{v.dtype}")

    paged_kv = any(x is not None for x in (kv_indptr, kv_indices, kv_last_page_lens, block_table, seqlen_k))
    if paged_kv:
        if kv_cache_layout not in ("linear", "linear3d", "vectorized"):
            raise ValueError(f"flydsl_flash_attn_func: unsupported kv_cache_layout={kv_cache_layout}")
        if kv_lookup_table not in ("vllm", "sglang"):
            raise ValueError(f"flydsl_flash_attn_func: unsupported kv_lookup_table={kv_lookup_table}")
        # Future native paged-KV FlyDSL kernel call shape:
        # return flydsl_flash_attn_paged_func(
        #     q,
        #     k,
        #     v,
        #     cu_seqlens_q=cu_seqlens_q,
        #     kv_indptr=kv_indptr,
        #     kv_indices=kv_indices,
        #     kv_last_page_lens=kv_last_page_lens,
        #     block_table=block_table,
        #     seqlen_k=seqlen_k,
        #     max_seqlen_q=max_seqlen_q,
        #     max_seqlen_kv=max_seqlen_kv,
        #     kv_cache_layout=kv_cache_layout,
        #     kv_lookup_table=kv_lookup_table,
        #     out=out,
        #     stream=stream,
        #     # plus kernel build options as needed:
        #     # waves_per_eu=waves_per_eu,
        #     # daz=daz,
        #     # dualwave_swp_lazy_rescale=dualwave_swp_lazy_rescale,
        #     # dualwave_swp_setprio=dualwave_swp_setprio,
        #     # dualwave_swp_enable_stagger=dualwave_swp_enable_stagger,
        #     # debug_counts=debug_counts,
        # )
        #
        # Temporary bridge: materialize the logical K/V sequence from the physical
        # paged cache, then call the existing dense/varlen FlyDSL kernels.
        k_logical, v_logical, fallback_cu_kv, fallback_max_kv = _materialize_paged_kv_for_fallback(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_lens=kv_last_page_lens,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_cache_layout=kv_cache_layout,
            kv_lookup_table=kv_lookup_table,
        )
        return flydsl_flash_attn_func(
            q,
            k_logical,
            v_logical,
            causal=causal,
            num_kv_heads=num_kv_heads,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=fallback_cu_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=fallback_max_kv,
            cross_seqlen=True if fallback_cu_kv is not None else cross_seqlen,
            num_kv_splits=num_kv_splits,
            out=out,
            waves_per_eu=waves_per_eu,
            daz=daz,
            dualwave_swp_lazy_rescale=dualwave_swp_lazy_rescale,
            dualwave_swp_setprio=dualwave_swp_setprio,
            dualwave_swp_enable_stagger=dualwave_swp_enable_stagger,
            debug_counts=debug_counts,
            stream=stream,
        )

    dtype_str = _dtype_str(q)
    varlen = cu_seqlens_q is not None

    if varlen and cu_seqlens_kv is None:
        raise ValueError("flydsl_flash_attn_func: cu_seqlens_kv required when cu_seqlens_q is given")
    if not varlen and cu_seqlens_kv is not None:
        raise ValueError("flydsl_flash_attn_func: cu_seqlens_q required when cu_seqlens_kv is given")
    if varlen and num_kv_splits > 1:
        raise ValueError("flydsl_flash_attn_func: varlen + split-K (num_kv_splits>1) is not supported")

    # ── shape inference ─────────────────────────────────────────────────────
    if varlen:
        if q.dim() != 3:
            raise ValueError(f"flydsl_flash_attn_func: varlen q must be 3D [total,H,D], got {q.dim()}D")
        _total_q, H, D = q.shape
        Hkv = k.shape[1]
        B = cu_seqlens_q.numel() - 1
        if max_seqlen_q is None:
            raise ValueError("flydsl_flash_attn_func: max_seqlen_q is required in varlen mode")
        if cross_seqlen is None:
            raise ValueError("flydsl_flash_attn_func: cross_seqlen is required in varlen mode")
        Sq = int(max_seqlen_q)
        cross = bool(cross_seqlen)
        if cross and max_seqlen_kv is None:
            raise ValueError("flydsl_flash_attn_func: max_seqlen_kv is required when varlen cross_seqlen=True")
    else:
        if q.dim() != 4:
            raise ValueError(f"flydsl_flash_attn_func: dense q must be 4D [B,Sq,H,D], got {q.dim()}D")
        B, Sq, H, D = q.shape
        Skv = k.shape[1]
        Hkv = k.shape[2]
        cross = Sq != Skv if cross_seqlen is None else bool(cross_seqlen)

    if num_kv_heads is None:
        num_kv_heads = Hkv
    if H % num_kv_heads != 0:
        raise ValueError(f"flydsl_flash_attn_func: num_heads ({H}) must be divisible by num_kv_heads ({num_kv_heads})")
    if D < 64 or D % 32 != 0:
        raise ValueError(f"flydsl_flash_attn_func: head_dim ({D}) must be >= 64 and a multiple of 32")

    splitk = num_kv_splits > 1

    # ── split-K eligibility guard (SKIP analogous to run_splitk_config) ────
    if splitk:
        if D != 128 or dtype_str not in ("bf16", "f16") or Sq < 384:
            raise ValueError(
                f"flydsl_flash_attn_func: split-K requires D=128, dtype bf16/f16, seq_len>=384; "
                f"got D={D}, dtype={dtype_str}, seq_len={Sq}"
            )
        from kernels.flash_attn_gfx950 import dualwave_splitk_workspace_elems

        ws_elems = dualwave_splitk_workspace_elems(B, H, Sq, int(num_kv_splits), head_dim=D)
        if ws_elems * 4 >= 0xFFFFFFFF:
            raise ValueError(
                f"flydsl_flash_attn_func: split-K workspace would exceed 4 GiB "
                f"({ws_elems * 4} bytes); use fewer splits or a smaller shape"
            )

    # ── build (cached) ──────────────────────────────────────────────────────
    debug_lazy = debug_counts is not None

    with torch.cuda.device(q.device.index):
        launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream

        if splitk:
            exe = _build_splitk(
                num_heads=H,
                num_kv_heads=num_kv_heads,
                head_dim=D,
                causal=causal,
                dtype_str=dtype_str,
                num_kv_splits=int(num_kv_splits),
                waves_per_eu=waves_per_eu,
                daz=daz,
                lazy_rescale=dualwave_swp_lazy_rescale,
                setprio=dualwave_swp_setprio,
                enable_stagger=dualwave_swp_enable_stagger,
            )
        elif varlen:
            exe = _build_varlen(
                num_heads=H,
                num_kv_heads=num_kv_heads,
                head_dim=D,
                causal=causal,
                dtype_str=dtype_str,
                cross_seqlen=cross,
                waves_per_eu=waves_per_eu,
                daz=daz,
                lazy_rescale=dualwave_swp_lazy_rescale,
                setprio=dualwave_swp_setprio,
                debug_lazy_counts=debug_lazy,
                enable_stagger=dualwave_swp_enable_stagger,
            )
        else:
            exe = _build_dense(
                num_heads=H,
                num_kv_heads=num_kv_heads,
                head_dim=D,
                causal=causal,
                dtype_str=dtype_str,
                cross_seqlen=cross,
                waves_per_eu=waves_per_eu,
                daz=daz,
                lazy_rescale=dualwave_swp_lazy_rescale,
                setprio=dualwave_swp_setprio,
                debug_lazy_counts=debug_lazy,
                enable_stagger=dualwave_swp_enable_stagger,
            )

        # ── allocate output ─────────────────────────────────────────────────
        if out is None:
            out = torch.empty_like(q)
        q_flat = q.contiguous().reshape(-1)
        k_flat = k.contiguous().reshape(-1)
        v_flat = v.contiguous().reshape(-1)
        o_flat = out.reshape(-1)

        # ── launch ──────────────────────────────────────────────────────────
        if splitk:
            _ws = torch.empty(ws_elems, dtype=torch.float32, device=q.device)
            exe(q_flat, k_flat, v_flat, o_flat, B, Sq, workspace=_ws, stream=launch_stream)
        elif varlen:
            kwargs = dict(cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, stream=launch_stream)
            if cross:
                kwargs["seq_len_kv"] = int(max_seqlen_kv)
            if debug_lazy:
                exe(q_flat, k_flat, v_flat, o_flat, B, Sq, debug_counts=debug_counts, **kwargs)
            else:
                exe(q_flat, k_flat, v_flat, o_flat, B, Sq, **kwargs)
        else:
            kwargs: dict = dict(stream=launch_stream)
            if cross:
                kwargs["seq_len_kv"] = Skv
            if debug_lazy:
                kwargs["debug_counts"] = debug_counts
            exe(q_flat, k_flat, v_flat, o_flat, B, Sq, **kwargs)

    return out
