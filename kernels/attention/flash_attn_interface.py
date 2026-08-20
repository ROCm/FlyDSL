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
from kernels.attention.flash_attn_utils import (
    DUALWAVE_SWP_BLOCK_M,
    MIN_Q_BLOCKS_XCD_SWIZZLE,
    NUM_XCD_GFX950,
    bias_addressing_error,
    dualwave_splitk_workspace_elems,
)

__all__ = ["flydsl_flash_attn_func", "dualwave_splitk_workspace_elems"]

_DTYPE_MAP = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float8_e4m3fn: "fp8"}

# Short varlen/paged cases use the lightweight generic path.
_VARLEN_LIGHT_MAX_SEQ = 256
# Largest flat element count the fp8 C-ABI can address; see the split below.
_FP8_MAX_FLAT_ELEMS = 2**31
_DENSE_LIGHT_CU_FALLBACK = 256
_DENSE_DUALWAVE_MIN_SEQ = 256
_DENSE_DUALWAVE_LARGE_BATCH = 8
_DENSE_DUALWAVE_MIN_SEQ_LARGE_BATCH = 192
_DENSE_M256_MIN_TOKENS = 4096


def _dtype_str(t: torch.Tensor) -> str:
    s = _DTYPE_MAP.get(t.dtype)
    if s is None:
        raise ValueError(f"flydsl_flash_attn_func only supports bf16/f16/fp8, got {t.dtype!r}")
    return s


def _gpu_arch(device: torch.device) -> str:
    try:
        return torch.cuda.get_device_properties(device.index).gcnArchName.split(":")[0]
    except Exception:
        return ""


def _dense_routes_to_dualwave(batch: int, seq_len: int) -> bool:
    if batch >= _DENSE_DUALWAVE_LARGE_BATCH:
        return seq_len >= _DENSE_DUALWAVE_MIN_SEQ_LARGE_BATCH
    return seq_len >= _DENSE_DUALWAVE_MIN_SEQ


def _dense_light_cu(device: torch.device) -> int:
    try:
        return int(torch.cuda.get_device_properties(device.index).multi_processor_count)
    except Exception:
        return _DENSE_LIGHT_CU_FALLBACK


def _dense_generic_tile(batch: int, seq_len: int, num_heads: int, head_dim: int, dtype_str: str, device: torch.device):
    if head_dim in (64, 128) and dtype_str in ("bf16", "f16"):
        main_blocks = batch * num_heads * ((seq_len + 127) // 128)
        if main_blocks < _dense_light_cu(device):
            return 64, 128, "N32"
    if num_heads >= 32 and batch * seq_len >= _DENSE_M256_MIN_TOKENS:
        return 256, 512, "auto"
    return 128, 256, "auto"


# ── build-cache helpers ────────────────────────────────────────────────────


@functools.lru_cache(maxsize=256)
def _build_dense(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    cross_seqlen: bool,
    block_m: int,
    flat_work_group_size: int,
    path_tag: str,
    waves_per_eu: int,
    daz: bool,
    return_lse: bool = False,
):
    """Build (and cache) one dense generic launcher variant."""
    from kernels.attention.flash_attn_generic import build_flash_attn_func_module

    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
        cross_seqlen=cross_seqlen,
        block_m=block_m,
        flat_work_group_size=flat_work_group_size,
        path_tag=path_tag,
        waves_per_eu=waves_per_eu,
        daz=daz,
        return_lse=return_lse,
    )


@functools.lru_cache(maxsize=256)
def _build_dense_dualwave(
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
    return_lse: bool = False,
    has_bias: bool = False,
    has_alibi: bool = False,
    has_sink: bool = False,
    xcd_swizzle: bool = False,
):
    """Build (and cache) the dense gfx950 DUALWAVE_SWP launcher."""
    from kernels.attention.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module

    return build_flash_attn_dualwave_swp_module(
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
        return_lse=return_lse,
        has_bias=has_bias,
        has_alibi=has_alibi,
        has_sink=has_sink,
        _xcd_swizzle=xcd_swizzle,
    )


@functools.lru_cache(maxsize=128)
def _build_dense_fp8(
    num_heads: int,
    num_kv_heads: int,
    causal: bool,
    waves_per_eu: int,
    daz: bool,
    lazy_rescale: bool,
    setprio: bool,
    enable_stagger: bool,
):
    """Build (and cache) the dense gfx950 fp8 launcher."""
    from kernels.attention.flash_attn_fp8_gfx950 import build_flash_attn_dualwave_swp_fp8_module

    return build_flash_attn_dualwave_swp_fp8_module(
        num_heads=num_heads,
        head_dim=128,
        causal=causal,
        dtype_str="fp8",
        num_kv_heads=num_kv_heads,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=lazy_rescale,
        dualwave_swp_setprio=setprio,
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
    return_lse: bool = False,
    has_bias: bool = False,
    has_alibi: bool = False,
    has_sink: bool = False,
):
    """Build (and cache) a varlen-mode launcher (gfx950 DUALWAVE_SWP, varlen=True)."""
    from kernels.attention.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module

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
        return_lse=return_lse,
        has_bias=has_bias,
        has_alibi=has_alibi,
        has_sink=has_sink,
    )


@functools.lru_cache(maxsize=256)
def _build_varlen_light(
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
    return_lse: bool = False,
):
    """Build a lightweight packed-varlen launcher for short attention."""
    from kernels.attention.flash_attn_generic import build_flash_attn_func_module

    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
        cross_seqlen=cross_seqlen,
        varlen=True,
        block_m=64,
        flat_work_group_size=128,
        waves_per_eu=waves_per_eu,
        daz=daz,
        return_lse=return_lse,
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
    return_lse: bool = False,
    has_bias: bool = False,
    has_alibi: bool = False,
    has_sink: bool = False,
):
    """Build (and cache) a split-K launcher (gfx950 DUALWAVE_SWP, num_kv_splits>1)."""
    from kernels.attention.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module

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
        return_lse=return_lse,
        has_bias=has_bias,
        has_alibi=has_alibi,
        has_sink=has_sink,
    )


@functools.lru_cache(maxsize=256)
def _build_paged(
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
    enable_stagger: bool,
    num_kv_splits: int = 1,
    varlen: bool = False,
    kv_cache_layout: str = "linear",
    return_lse: bool = False,
    has_bias: bool = False,
):
    """Build (and cache) a paged-KV launcher (gfx950 DUALWAVE_SWP, paged=True).

    ``num_kv_splits > 1`` builds the paged + split-K variant (KV dimension split
    across grid_z = B*num_kv_splits workgroups + a combine pass), which fills the
    GPU for low-occupancy shapes (small B / few heads).

    ``varlen=True`` builds the packed-Q (cu_seqlens) + paged-KV variant: Q/O are
    ``[total_q, H, D]`` and K/V are the physical page cache, looked up via the
    block table per kv-tile. Mutually exclusive with split-K.

    ``kv_cache_layout`` selects the physical page layout: "linear"
    [NumBlocks,PageSize,Hkv,D] or "vectorized" (aiter 5D).
    """
    from kernels.attention.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module

    return build_flash_attn_dualwave_swp_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
        paged=True,
        varlen=varlen,
        num_kv_splits=num_kv_splits,
        cross_seqlen=cross_seqlen,
        kv_cache_layout=kv_cache_layout,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=lazy_rescale,
        dualwave_swp_setprio=setprio,
        dualwave_swp_enable_stagger=enable_stagger,
        return_lse=return_lse,
        has_bias=has_bias,
    )


# ── paged-KV native path ────────────────────────────────────────────────────

# gfx950 dualwave paged-KV currently supports exactly one configuration.
_PAGED_PAGE_SIZE = 64
_PAGED_BT_LDS_SIZE = 2048


def _flydsl_flash_attn_paged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    num_kv_heads: Optional[int],
    bias: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    seqlen_k: Optional[torch.Tensor],
    max_seqlen_kv: Optional[int],
    kv_cache_layout: str,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_kv: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    cross_seqlen: Optional[bool],
    num_kv_splits: int,
    out: Optional[torch.Tensor],
    waves_per_eu: int,
    daz: bool,
    dualwave_swp_lazy_rescale: bool,
    dualwave_swp_setprio: bool,
    dualwave_swp_enable_stagger: bool,
    stream,
) -> torch.Tensor:
    """Native paged-KV attention on the gfx950 dualwave kernel.

    Supported config ONLY (anything else raises): linear/vectorized cache layout
    [NumBlocks, PageSize=64, NumKVHeads, HeadDim], vLLM lookup (block_table +
    seqlen_k), causal, D=64/128, dtype bf16/f16.
    - Dense 4D Q ``[B, Sq, H, D]``: split-K (num_kv_splits>1) supported (seq_len>=384).
    - Varlen packed Q ``[total_q, H, D]`` (cu_seqlens_q given): paged K/V looked up
      per kv-tile via block_table; split-K not supported (matches dense varlen).
    """
    if kv_cache_layout not in ("linear", "vectorized"):
        raise NotImplementedError(
            f"flydsl_flash_attn_func: native paged KV supports kv_cache_layout in ('linear','vectorized'), "
            f"got {kv_cache_layout!r}"
        )
    if block_table is None or seqlen_k is None:
        raise ValueError("flydsl_flash_attn_func: native paged KV (vllm) requires block_table and seqlen_k")
    vectorized = kv_cache_layout == "vectorized"
    if vectorized:
        # aiter 5D: K [NumBlocks, Hkv, D/kVS, PageSize, kVS], V [NumBlocks, Hkv, PageSize/kVS, D, kVS].
        if k.dim() != 5 or v.dim() != 5:
            raise ValueError(f"flydsl_flash_attn_func: vectorized paged K/V must be 5D, got K{k.dim()}D V{v.dim()}D")
    elif k.dim() != 4:
        raise ValueError(
            f"flydsl_flash_attn_func: linear paged K/V must be 4D [NumBlocks,PageSize,Hkv,D], got {k.dim()}D"
        )

    varlen = cu_seqlens_q is not None
    dtype_str = _dtype_str(q)
    if varlen:
        # Packed varlen Q: [total_q, H, D]. Per-batch ranges come from cu_seqlens
        # inside the kernel; grid_y is sized by max_seqlen_q.
        if cu_seqlens_kv is None:
            raise ValueError("flydsl_flash_attn_func: varlen paged KV requires cu_seqlens_kv")
        if max_seqlen_q is None:
            raise ValueError("flydsl_flash_attn_func: varlen paged KV requires max_seqlen_q")
        if num_kv_splits > 1:
            raise NotImplementedError("flydsl_flash_attn_func: varlen paged KV does not support split-K")
        if q.dim() != 3:
            raise ValueError(f"flydsl_flash_attn_func: varlen paged q must be 3D [total_q,H,D], got {q.dim()}D")
        _total_q, H, D = q.shape
        B = cu_seqlens_q.numel() - 1
        Sq = int(max_seqlen_q)
    else:
        if q.dim() != 4:
            raise ValueError(f"flydsl_flash_attn_func: paged dense q must be 4D [B,Sq,H,D], got {q.dim()}D")
        B, Sq, H, D = q.shape
    if vectorized:
        kvs = 16 // q.element_size()
        Hkv = int(k.shape[1])
        page_size = int(k.shape[3])
        k_head_dim = int(k.shape[2]) * int(k.shape[4])  # (D/kVS) * kVS
        if int(k.shape[4]) != kvs:
            raise ValueError(f"flydsl_flash_attn_func: vectorized K last dim ({k.shape[4]}) must equal kVS={kvs}")
    else:
        page_size = int(k.shape[1])
        Hkv = int(k.shape[2])
        k_head_dim = int(k.shape[3])
    if page_size != _PAGED_PAGE_SIZE:
        raise NotImplementedError(
            f"flydsl_flash_attn_func: native paged KV supports page_size={_PAGED_PAGE_SIZE} only, got {page_size}"
        )
    if D not in (64, 128):
        raise NotImplementedError(f"flydsl_flash_attn_func: native paged KV supports head_dim=64 or 128, got {D}")
    if k_head_dim != D:
        raise ValueError(f"flydsl_flash_attn_func: paged K head_dim ({k_head_dim}) must match q head_dim ({D})")

    if num_kv_heads is None:
        num_kv_heads = Hkv
    if H % num_kv_heads != 0:
        raise ValueError(f"flydsl_flash_attn_func: num_heads ({H}) must be divisible by num_kv_heads ({num_kv_heads})")

    # Split-K (paged, dense only): split the KV dimension across grid_z = B*num_kv_splits
    # workgroups + a combine pass. Fills the GPU for low-occupancy shapes (small B / few
    # heads), where single-split paged underutilizes the device.
    splitk = num_kv_splits > 1
    if splitk and (D not in (64, 128) or dtype_str not in ("bf16", "f16") or Sq < 384):
        raise ValueError(
            f"flydsl_flash_attn_func: paged split-K requires D=64/128, dtype bf16/f16, seq_len>=384; "
            f"got D={D}, dtype={dtype_str}, seq_len={Sq}"
        )

    # Per-batch KV lengths differ in general → bottom-right cross-length masking. Varlen
    # paged always uses cross masking (per-batch seqlen_q/seqlen_kv come from cu_seqlens).
    _kv_lens = seqlen_k.reshape(-1).tolist() if max_seqlen_kv is None or (bias is not None and not varlen) else None
    skv = int(max_seqlen_kv) if max_seqlen_kv is not None else int(max(_kv_lens))
    max_kv_pages = (skv + page_size - 1) // page_size
    max_pages_per_split = (max_kv_pages + int(num_kv_splits) - 1) // int(num_kv_splits)
    if max_pages_per_split > _PAGED_BT_LDS_SIZE:
        max_supported_kv = _PAGED_BT_LDS_SIZE * int(num_kv_splits) * page_size
        raise NotImplementedError(
            f"flydsl_flash_attn_func: paged KV length {skv} exceeds block-table LDS window "
            f"({_PAGED_BT_LDS_SIZE} pages/split, max_kv_len={max_supported_kv} for "
            f"num_kv_splits={num_kv_splits}, page_size={page_size})"
        )
    if varlen:
        cross = bool(cross_seqlen) if cross_seqlen is not None else True
    else:
        cross = skv != Sq
    if bias is not None:
        if not varlen:
            if min(_kv_lens) != max(_kv_lens):
                raise NotImplementedError(
                    f"flydsl_flash_attn_func: dense paged bias requires uniform seqlen_k, got lengths in "
                    f"[{min(_kv_lens)}, {max(_kv_lens)}]; the dense paged kernel receives only "
                    f"max_seqlen_kv. Use the varlen paged path (cu_seqlens_q/cu_seqlens_kv) for "
                    f"ragged KV lengths."
                )
        # Same convention as non-paged: rows are q tokens, columns are batch-local
        # logical key positions (the block table only redirects the K/V fetch).
        _bias_rows = int(q.shape[0]) if varlen else Sq
        if bias.dim() != 2:
            raise ValueError(f"flydsl_flash_attn_func: paged bias must be 2D, got {bias.dim()}D")
        if bias.shape[0] != _bias_rows:
            raise ValueError(
                f"flydsl_flash_attn_func: paged bias must have {_bias_rows} rows "
                f"({'total_q' if varlen else 'seq_len_q'}), got {tuple(bias.shape)}"
            )
        if bias.shape[1] < skv:
            raise ValueError(
                f"flydsl_flash_attn_func: paged bias needs >= max_seqlen_kv={skv} columns, got {bias.shape[1]}"
            )

    block_table_stride = int(block_table.shape[1])
    # Flatten so the kernel's flat row-major index addresses block_table correctly.
    block_table_i32 = (
        (block_table if block_table.dtype == torch.int32 else block_table.to(torch.int32)).contiguous().reshape(-1)
    )

    with torch.cuda.device(q.device.index):
        launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
        # Short paged attention uses generic light; unsupported cases stay on dualwave.
        _arch = _gpu_arch(q.device)
        _paged_light_ok = (
            (num_kv_splits <= 1)
            and bias is None  # the light paged kernel has no bias path
            and D in (64, 128)
            and dtype_str in ("bf16", "f16")
            and (not _arch.startswith("gfx950") or Sq <= _VARLEN_LIGHT_MAX_SEQ)
        )
        if _paged_light_ok:
            exe = _build_paged_light(
                num_heads=H,
                num_kv_heads=num_kv_heads,
                head_dim=D,
                causal=causal,
                dtype_str=dtype_str,
                cross_seqlen=cross,
                varlen=varlen,
                kv_cache_layout=kv_cache_layout,
                waves_per_eu=waves_per_eu,
                daz=daz,
                lazy_rescale=dualwave_swp_lazy_rescale,
                setprio=dualwave_swp_setprio,
                debug_lazy_counts=False,
                enable_stagger=dualwave_swp_enable_stagger,
            )
        else:
            exe = _build_paged(
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
                enable_stagger=dualwave_swp_enable_stagger,
                num_kv_splits=int(num_kv_splits),
                varlen=varlen,
                kv_cache_layout=kv_cache_layout,
                has_bias=bias is not None,
            )
        if out is None:
            out = torch.empty_like(q)
        # Keep tensors in natural shape; flattening can overflow int32 C-ABI dims.
        # The paged kernel rebuilds per-batch/page descriptors from base pointers.
        q_flat = q.contiguous()
        k_flat = k.contiguous()
        v_flat = v.contiguous()
        o_flat = out.contiguous()
        kwargs = dict(block_table=block_table_i32, block_table_stride=block_table_stride, stream=launch_stream)
        if bias is not None:
            kwargs["bias"] = bias
        if varlen:
            kwargs["cu_seqlens_q"] = cu_seqlens_q
            kwargs["cu_seqlens_kv"] = cu_seqlens_kv
        if cross:
            kwargs["seq_len_kv"] = skv
        if splitk:
            ws_elems = dualwave_splitk_workspace_elems(B, H, Sq, int(num_kv_splits), head_dim=D)
            _ws = torch.empty(ws_elems, dtype=torch.float32, device=q.device)
            kwargs["workspace"] = _ws
        exe(q_flat, k_flat, v_flat, o_flat, B, Sq, **kwargs)

    return out


@functools.lru_cache(maxsize=256)
def _build_paged_light(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    cross_seqlen: bool,
    varlen: bool,
    kv_cache_layout: str,
    waves_per_eu: int,
    daz: bool,
    lazy_rescale: bool,
    setprio: bool,
    debug_lazy_counts: bool,
    enable_stagger: bool,
    return_lse: bool = False,
):
    """Build a lightweight paged-varlen launcher for short attention."""
    from kernels.attention.flash_attn_generic import build_flash_attn_func_module

    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
        cross_seqlen=cross_seqlen,
        varlen=varlen,
        paged=True,
        kv_cache_layout=kv_cache_layout,
        block_m=64,
        flat_work_group_size=128,
        path_tag="N32",
        waves_per_eu=waves_per_eu,
        daz=daz,
        return_lse=return_lse,
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
    # Paged KV cache ABI: vLLM-style block_table + seqlen_k.
    block_table: Optional[torch.Tensor] = None,
    seqlen_k: Optional[torch.Tensor] = None,
    kv_cache_layout: str = "linear",
    # Split-K (gfx950 only, seq_len >= 384, D=64/128, bf16/f16).
    num_kv_splits: int = 1,
    # Additive attention bias, folded into the scores after sm_scale and before
    # masking. gfx950 DUALWAVE_SWP only (dense / varlen / split-K / paged KV).
    bias: Optional[torch.Tensor] = None,
    # Per-head ALiBi slope table, computed analytically into the scores. Same
    # path as `bias` but no paged-KV support; may be combined with `bias`.
    alibi_slopes: Optional[torch.Tensor] = None,
    # Per-head attention-sink logit: one extra softmax denominator term with no
    # matching V row. Same path and restrictions as `alibi_slopes`; combinable.
    sink: Optional[torch.Tensor] = None,
    # fp8 dense ABI: per-tensor descales for pre-quantized e4m3fn Q/K/V.
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    # Output tensor; allocated if None.
    out: Optional[torch.Tensor] = None,
    # Also return per-row LSE = ln(sum_j exp(sm_scale * q_i.k_j)); fp32
    # [B, num_heads, Sq]. Needed by backward; not supported for fp8.
    return_lse: bool = False,
    # Kernel build options.
    waves_per_eu: int = 2,
    daz: bool = True,
    dualwave_swp_lazy_rescale: Optional[bool] = None,
    dualwave_swp_setprio: bool = True,
    dualwave_swp_enable_stagger: bool = True,
    # Re-derive (head, q_block) with head as the slow axis so one head's q-blocks
    # stay on one XCD instead of every XCD re-streaming that head's K/V. None
    # auto-selects on the shapes it helps; True/False force it. Dense non-fp8 only.
    dualwave_swp_xcd_swizzle: Optional[bool] = None,
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
        block_table / seqlen_k: vLLM-style 2D block table metadata. Enables the
            native paged-KV path, which supports ``bias`` but not
            ``alibi_slopes``, ``sink``, ``return_lse``, or fp8.
        num_kv_splits: Split-K factor (>1: gfx950 only, D=64/128, bf16/f16, seq>=384).
        bias: Additive attention bias with the same dtype as q, folded in as
            ``softmax(q @ k^T * sm_scale + bias)`` -- after the scale, before the
            causal/padding mask. Dense: ``[Sq, Skv]``, broadcast over batch and
            head. Varlen: ``[total_q, max_seqlen_kv]``, where the row is the
            *global* packed q token index and the column is the *per-batch-local*
            key index, broadcast over head. Varlen self-attention leaves
            ``max_seqlen_kv`` unset, so its column bound is ``max_seqlen_q``.
            Routes to the gfx950 DUALWAVE_SWP kernel; fp8 raises
            NotImplementedError rather than silently dropping the bias.
            Paged KV is supported (dense, varlen, and paged split-K) with the
            same row/column convention: rows are ``seq_len_q`` (dense) or
            ``total_q`` (varlen) q tokens, columns are batch-local key indices
            and must number at least ``max_seqlen_kv``. Dense paged
            additionally requires a uniform ``seqlen_k`` across the batch --
            the dense paged launch only receives ``max_seqlen_kv``, so ragged
            lengths would address the wrong bias columns and raise
            ``NotImplementedError``; use the varlen paged path
            (``cu_seqlens_q``/``cu_seqlens_kv``) for ragged KV.
        alibi_slopes: fp32 ALiBi slope table, ``[H]`` (broadcast over batch) or
            ``[B, H]``, values positive. Adds
            ``-slope * |i + seqlen_kv - seqlen_q - j|`` to the scores after the
            1/sqrt(D) scaling (the slope is not divided by it), bottom-right
            aligned like the causal mask. Positions are measured *within* the
            sequence, so varlen does not offset by the packed-token base. Same
            kernel path as ``bias`` and may be combined with it, but unlike
            ``bias`` it is not supported with paged KV (raises
            NotImplementedError), nor with fp8.
        sink: fp32 ``[H]`` per-head attention-sink logit -- one extra softmax
            denominator term that has no matching V row::

                O = sum_j exp(s_j - m) v_j / (exp(sink - m) + sum_j exp(s_j - m))

            Consumed verbatim (no host-side scaling), so it lives in the same
            post-sm_scale logit space as the scores. Applied in the epilogue, so
            it touches no score element; under split-K the per-split partials
            stay sink-free and the combine pass folds it in exactly once. Same
            kernel path and restrictions as ``alibi_slopes`` -- not supported
            with paged KV or fp8 -- but freely combinable with ``bias`` and
            ``alibi_slopes``.
        q_descale / k_descale / v_descale: fp32 shape-[1] descales required
            for dense fp8 e4m3fn inputs.
        out: Optional pre-allocated output tensor. For fp8, output is bf16;
            otherwise it has the same dtype as q.
        waves_per_eu: Kernel occupancy hint.
        daz: Enable denormals-are-zero.
        dualwave_swp_lazy_rescale: Enable lazy online softmax rescale. Defaults to
            True for bf16/fp16 and False for fp8, where the lazy path costs
            accuracy; pass explicitly to override.
        dualwave_swp_setprio: Enable s_setprio scheduling hints.
        dualwave_swp_enable_stagger: Enable wave-group phase stagger.
        debug_counts: Float32[2] tensor; when given, counts lazy-rescale branches
            (debug_counts[0] = all-below-true, debug_counts[1] = all-below-false).
        stream: CUDA/HIP stream to launch on.

    Returns:
        Output tensor with the same shape as q. The dtype is bf16 for fp8
        inputs, otherwise the same dtype as q. When ``return_lse=True`` returns
        ``(out, lse)`` where ``lse`` is fp32 ``[B, num_heads, Sq]`` (varlen:
        ``[B, num_heads, max_seqlen_q]``, padded) holding the per-row
        natural-log, scale-folded log-sum-exp.
    """
    # ── validation ──────────────────────────────────────────────────────────
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_func: q/k/v must be CUDA tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(f"flydsl_flash_attn_func: q/k/v must share device; got {q.device}/{k.device}/{v.device}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"flydsl_flash_attn_func: q/k/v must share dtype; got {q.dtype}/{k.dtype}/{v.dtype}")

    dtype_str = _dtype_str(q)
    if dualwave_swp_lazy_rescale is None:
        # fp8 defaults to the eager rescale: the lazy path cannot lift P far
        # enough to keep the softmax tail out of e4m3's flush-to-zero range.
        dualwave_swp_lazy_rescale = dtype_str != "fp8"
    if return_lse and dtype_str == "fp8":
        raise NotImplementedError("flydsl_flash_attn_func: return_lse is not supported for fp8")
    paged_kv = any(x is not None for x in (block_table, seqlen_k))
    if dtype_str == "fp8" and paged_kv:
        raise NotImplementedError("flydsl_flash_attn_func: fp8 flash_attn does not support paged KV")

    # The fp8 path flattens Q/K/V/O to 1-D and the C-ABI packs a dynamic dim as
    # int32, so a launch aborts once B*S*H*D reaches 2**31 (S >= 131072 at
    # D=128, H=64). Batch entries are independent and a leading slice of a
    # contiguous tensor is still contiguous, so one launch per entry divides the
    # flat dim by B at no copy. bf16 passes the natural 4-D shape and is exempt.
    if (
        dtype_str == "fp8"
        and not paged_kv
        and cu_seqlens_q is None
        and cu_seqlens_kv is None
        and q.dim() == 4
        and q.shape[0] > 1
        and q.numel() >= _FP8_MAX_FLAT_ELEMS
    ):
        kw = dict(
            causal=causal,
            num_kv_heads=num_kv_heads,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            cross_seqlen=cross_seqlen,
            kv_cache_layout=kv_cache_layout,
            num_kv_splits=num_kv_splits,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            waves_per_eu=waves_per_eu,
            daz=daz,
            dualwave_swp_lazy_rescale=dualwave_swp_lazy_rescale,
            dualwave_swp_setprio=dualwave_swp_setprio,
            dualwave_swp_enable_stagger=dualwave_swp_enable_stagger,
            debug_counts=debug_counts,
            stream=stream,
        )
        parts = []
        for i in range(q.shape[0]):
            sl = slice(i, i + 1)
            o = flydsl_flash_attn_func(
                q[sl].contiguous(), k[sl].contiguous(), v[sl].contiguous(), out=None if out is None else out[sl], **kw
            )
            parts.append(o[0] if isinstance(o, (tuple, list)) else o)
        # Each call already wrote into its own out[sl] view; concatenating would
        # copy several GB again and return a different tensor than was passed.
        return out if out is not None else torch.cat(parts, dim=0)
    if return_lse and paged_kv:
        raise NotImplementedError("flydsl_flash_attn_func: return_lse is not supported for paged KV")
    has_bias = bias is not None
    has_alibi = alibi_slopes is not None
    has_sink = sink is not None
    for _name, _t in (("bias", bias), ("alibi_slopes", alibi_slopes), ("sink", sink)):
        if _t is None:
            continue
        if paged_kv and _name != "bias":
            raise NotImplementedError(f"flydsl_flash_attn_func: {_name} is not supported for paged KV")
        if dtype_str == "fp8":
            raise NotImplementedError(f"flydsl_flash_attn_func: {_name} is not supported for fp8")
        if not _t.is_cuda or _t.device != q.device:
            raise ValueError(f"flydsl_flash_attn_func: {_name} must be a CUDA tensor on {q.device}, got {_t.device}")
    if has_bias:
        if bias.dtype != q.dtype:
            raise ValueError(f"flydsl_flash_attn_func: bias dtype must match q dtype {q.dtype}, got {bias.dtype}")
        if bias.dim() != 2:
            raise ValueError(f"flydsl_flash_attn_func: bias must be 2D, got {bias.dim()}D")
        _bias_err = bias_addressing_error(bias.shape[0] * bias.shape[1], bias.element_size())
        if _bias_err is not None:
            raise ValueError(f"flydsl_flash_attn_func: bias {tuple(bias.shape)} {_bias_err}")
    if has_alibi:
        if alibi_slopes.dtype != torch.float32:
            raise ValueError(f"flydsl_flash_attn_func: alibi_slopes must be float32, got {alibi_slopes.dtype}")
        if alibi_slopes.dim() not in (1, 2):
            raise ValueError(f"flydsl_flash_attn_func: alibi_slopes must be [H] or [B, H], got {alibi_slopes.dim()}D")
    if has_sink:
        if sink.dtype != torch.float32:
            raise ValueError(f"flydsl_flash_attn_func: sink must be float32, got {sink.dtype}")
        if sink.dim() != 1:
            raise ValueError(f"flydsl_flash_attn_func: sink must be 1D [H], got {sink.dim()}D")
    if paged_kv:
        return _flydsl_flash_attn_paged(
            q,
            k,
            v,
            causal=causal,
            num_kv_heads=num_kv_heads,
            bias=bias,
            block_table=block_table,
            seqlen_k=seqlen_k,
            max_seqlen_kv=max_seqlen_kv,
            kv_cache_layout=kv_cache_layout,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            cross_seqlen=cross_seqlen,
            num_kv_splits=num_kv_splits,
            out=out,
            waves_per_eu=waves_per_eu,
            daz=daz,
            dualwave_swp_lazy_rescale=dualwave_swp_lazy_rescale,
            dualwave_swp_setprio=dualwave_swp_setprio,
            dualwave_swp_enable_stagger=dualwave_swp_enable_stagger,
            stream=stream,
        )

    varlen = cu_seqlens_q is not None

    if dtype_str == "fp8":
        if varlen:
            raise NotImplementedError("flydsl_flash_attn_func: fp8 flash_attn does not support varlen")
        if num_kv_splits > 1:
            raise NotImplementedError("flydsl_flash_attn_func: fp8 flash_attn does not support split-K")
        if debug_counts is not None:
            raise NotImplementedError("flydsl_flash_attn_func: fp8 flash_attn does not support debug_counts")
        if any(x is None for x in (q_descale, k_descale, v_descale)):
            raise ValueError("flydsl_flash_attn_func: fp8 requires q_descale, k_descale, and v_descale")
        for name, scale in (("q_descale", q_descale), ("k_descale", k_descale), ("v_descale", v_descale)):
            if not scale.is_cuda:
                raise ValueError(f"flydsl_flash_attn_func: {name} must be a CUDA tensor")
            if scale.device != q.device:
                raise ValueError(f"flydsl_flash_attn_func: {name} must be on {q.device}, got {scale.device}")
            if scale.dtype != torch.float32 or scale.numel() != 1:
                raise ValueError(f"flydsl_flash_attn_func: {name} must be a shape-[1] float32 tensor")

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

    if has_bias:
        # Bias rows are indexed by q token, columns by the per-batch-local key.
        if varlen:
            if bias.shape[0] != q.shape[0]:
                raise ValueError(
                    f"flydsl_flash_attn_func: varlen bias must be [total_q, max_seqlen_kv] with "
                    f"total_q={q.shape[0]}, got {tuple(bias.shape)}"
                )
            _bias_cols_min = int(max_seqlen_kv) if cross else Sq
            if bias.shape[1] < _bias_cols_min:
                _bound = "max_seqlen_kv" if cross else "max_seqlen_q, the self-attention KV maximum"
                raise ValueError(
                    f"flydsl_flash_attn_func: varlen bias needs >= {_bound}={_bias_cols_min} "
                    f"columns, got {bias.shape[1]}"
                )
        elif tuple(bias.shape) != (Sq, Skv):
            raise ValueError(f"flydsl_flash_attn_func: dense bias must be [{Sq}, {Skv}], got {tuple(bias.shape)}")

    if has_alibi:
        if alibi_slopes.shape[-1] != H:
            raise ValueError(
                f"flydsl_flash_attn_func: alibi_slopes last dim must be num_heads={H}, "
                f"got {tuple(alibi_slopes.shape)}"
            )
        if alibi_slopes.dim() == 2 and alibi_slopes.shape[0] != B:
            raise ValueError(
                f"flydsl_flash_attn_func: 2D alibi_slopes must be [batch={B}, num_heads={H}], "
                f"got {tuple(alibi_slopes.shape)}"
            )

    if has_sink and sink.shape[0] != H:
        raise ValueError(f"flydsl_flash_attn_func: sink must be [num_heads={H}], got {tuple(sink.shape)}")

    splitk = num_kv_splits > 1

    # ── split-K eligibility guard (SKIP analogous to run_splitk_config) ────
    if splitk:
        if D not in (64, 128) or dtype_str not in ("bf16", "f16") or Sq < 384:
            raise ValueError(
                f"flydsl_flash_attn_func: split-K requires D=64/128, dtype bf16/f16, seq_len>=384; "
                f"got D={D}, dtype={dtype_str}, seq_len={Sq}"
            )
        ws_elems = dualwave_splitk_workspace_elems(B, H, Sq, int(num_kv_splits), head_dim=D)

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
                return_lse=return_lse,
                has_bias=has_bias,
                has_alibi=has_alibi,
                has_sink=has_sink,
            )
        elif varlen:
            # Short varlen attention uses generic light; long/debug stays on dualwave.
            _arch = _gpu_arch(q.device)
            _prefer_light = (
                (not debug_lazy)
                # The light (generic) kernel folds in neither bias nor ALiBi.
                and (not has_bias)
                and (not has_alibi)
                and (not has_sink)
                and D in (64, 128)
                and dtype_str in ("bf16", "f16")
                and (not _arch.startswith("gfx950") or Sq <= _VARLEN_LIGHT_MAX_SEQ)
            )
            if _prefer_light:
                exe = _build_varlen_light(
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
                    return_lse=return_lse,
                )
            else:
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
                    return_lse=return_lse,
                    has_bias=has_bias,
                    has_alibi=has_alibi,
                    has_sink=has_sink,
                )
        else:
            _arch = _gpu_arch(q.device)
            if dtype_str == "fp8":
                if not _arch.startswith("gfx950"):
                    raise ValueError(f"flydsl_flash_attn_func: fp8 requires gfx950, got '{_arch or 'unknown'}'")
                exe = _build_dense_fp8(
                    num_heads=H,
                    num_kv_heads=num_kv_heads,
                    causal=causal,
                    waves_per_eu=waves_per_eu,
                    daz=daz,
                    lazy_rescale=dualwave_swp_lazy_rescale,
                    setprio=dualwave_swp_setprio,
                    enable_stagger=dualwave_swp_enable_stagger,
                )
            else:
                can_dualwave = D in (64, 128) and dtype_str in ("bf16", "f16") and _arch.startswith("gfx950")
                if debug_lazy and not can_dualwave:
                    raise NotImplementedError(
                        "flydsl_flash_attn_func: debug_counts requires the gfx950 DUALWAVE_SWP path"
                    )
                if (has_bias or has_alibi or has_sink) and not can_dualwave:
                    _term = "bias" if has_bias else ("alibi_slopes" if has_alibi else "sink")
                    raise NotImplementedError(
                        f"flydsl_flash_attn_func: {_term} requires the gfx950 DUALWAVE_SWP path "
                        f"(D=64/128, bf16/f16, gfx950); got D={D}, dtype={dtype_str}, arch='{_arch or 'unknown'}'"
                    )
                # bias/ALiBi force dualwave: the generic dense kernel folds in neither.
                if (
                    debug_lazy
                    or has_bias
                    or has_alibi
                    or has_sink
                    or (can_dualwave and _dense_routes_to_dualwave(B, Sq))
                ):
                    # Workgroups map to XCDs as linear_id % 8, and linear_id is
                    # bx + by*H + bz*H*nqb, so with H % 8 == 0 a head-fast grid pins
                    # head h to XCD h % 8. The ~256 resident workgroups span all H
                    # heads within one batch, leaving each XCD to juggle H/8 K/V
                    # streams against its L2 slice. The head-slow remap in
                    # _init_dualwave_thread_mapping puts the resident window inside a
                    # single head instead: 1 stream. Measured penalty for leaving it
                    # off tracks H/8 (-6% at 8 streams, -3% at 4, nil at <=2), not the
                    # hit rate and not traffic volume. Bijective, so output is
                    # unchanged. NB: that function's own comment states the opposite
                    # rationale ("scatter across all XCDs") and is wrong.
                    num_q_blocks = -(-int(Sq) // DUALWAVE_SWP_BLOCK_M)
                    if dualwave_swp_xcd_swizzle is None:
                        xcd_swizzle = (
                            not causal and H % NUM_XCD_GFX950 == 0 and num_q_blocks >= MIN_Q_BLOCKS_XCD_SWIZZLE
                        )
                    else:
                        xcd_swizzle = dualwave_swp_xcd_swizzle
                    exe = _build_dense_dualwave(
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
                        return_lse=return_lse,
                        has_bias=has_bias,
                        has_alibi=has_alibi,
                        has_sink=has_sink,
                        xcd_swizzle=xcd_swizzle,
                    )
                else:
                    block_m, flat_work_group_size, path_tag = _dense_generic_tile(B, Sq, H, D, dtype_str, q.device)
                    exe = _build_dense(
                        num_heads=H,
                        num_kv_heads=num_kv_heads,
                        head_dim=D,
                        causal=causal,
                        dtype_str=dtype_str,
                        cross_seqlen=cross,
                        block_m=block_m,
                        flat_work_group_size=flat_work_group_size,
                        path_tag=path_tag,
                        waves_per_eu=waves_per_eu,
                        daz=daz,
                        return_lse=return_lse,
                    )

        # ── allocate output ─────────────────────────────────────────────────
        if out is None:
            out_dtype = torch.bfloat16 if dtype_str == "fp8" else q.dtype
            out = torch.empty(q.shape, dtype=out_dtype, device=q.device)
        elif dtype_str == "fp8" and out.dtype != torch.bfloat16:
            raise ValueError(f"flydsl_flash_attn_func: fp8 output must be bf16, got {out.dtype}")
        elif dtype_str != "fp8" and out.dtype != q.dtype:
            raise ValueError(f"flydsl_flash_attn_func: output dtype must match q dtype {q.dtype}, got {out.dtype}")
        # Keep natural shape; flattening can overflow int32 C-ABI dims.
        # Kernels rebuild per-batch descriptors from base pointers and strides.
        if dtype_str == "fp8":
            # The fp8 gfx950 module preserves the original dense ABI from 711.diff:
            # flattened Q/K/V/O tensors plus descale kwargs.
            q_flat = q.contiguous().view(-1)
            k_flat = k.contiguous().view(-1)
            v_flat = v.contiguous().view(-1)
            o_flat = out.contiguous().view(-1)
        else:
            q_flat = q.contiguous()
            k_flat = k.contiguous()
            v_flat = v.contiguous()
            o_flat = out.contiguous()

        # ── allocate LSE (fp32 [B, num_heads, Sq]) ───────────────────────────
        lse = torch.empty((B, H, Sq), dtype=torch.float32, device=q.device) if return_lse else None

        # ── launch ──────────────────────────────────────────────────────────
        _bias_kw = {}
        if has_bias:
            _bias_kw["bias"] = bias
        if has_alibi:
            _bias_kw["alibi_slopes"] = alibi_slopes
        if has_sink:
            _bias_kw["sink"] = sink
        if splitk:
            _ws = torch.empty(ws_elems, dtype=torch.float32, device=q.device)
            exe(q_flat, k_flat, v_flat, o_flat, B, Sq, workspace=_ws, lse=lse, stream=launch_stream, **_bias_kw)
        elif varlen:
            kwargs = dict(
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, lse=lse, stream=launch_stream, **_bias_kw
            )
            if cross:
                kwargs["seq_len_kv"] = int(max_seqlen_kv)
            if debug_lazy:
                exe(q_flat, k_flat, v_flat, o_flat, B, Sq, debug_counts=debug_counts, **kwargs)
            else:
                exe(q_flat, k_flat, v_flat, o_flat, B, Sq, **kwargs)
        else:
            kwargs: dict = dict(stream=launch_stream, **_bias_kw)
            # fp8 has no LSE path (guarded above) and its launcher takes no `lse` arg.
            if dtype_str != "fp8":
                kwargs["lse"] = lse
            if cross:
                kwargs["seq_len_kv"] = Skv
            if debug_lazy:
                kwargs["debug_counts"] = debug_counts
            if dtype_str == "fp8":
                kwargs.update(q_descale=q_descale, k_descale=k_descale, v_descale=v_descale)
            exe(q_flat, k_flat, v_flat, o_flat, B, Sq, **kwargs)

    if return_lse:
        return out, lse
    return out
