# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tuning support for the mixed (fp4/fp8 x fp4) MoE 2-stage GEMM kernels.

This module holds host-side, pre-compile tooling for the MXFP4 MoE tuning
campaign.  Nothing here changes kernel behavior; it mirrors the legality checks
that ``compile_mixed_moe_gemm1`` / ``compile_mixed_moe_gemm2`` already enforce so
that a tile-config search can reject illegal candidates *before* spending GPU
time on a compile that the kernel would refuse.

The single entry point is :func:`check_tile_config`, which returns a
:class:`TileCheck` describing whether a ``(stage, tile_m, tile_n, tile_k, ...)``
candidate is legal and, when it is not, a machine-readable reason.

The constraints encoded here are a faithful copy of the ones in
``kernels/mixed_moe_gemm_2stage.py`` (stage1: ``tile_k_bytes % 64``,
``tile_m*tile_k*elem_bytes % total_threads``, split-K divisibility, the LDS
sizing / arch limit; stage2: ``model_dim % tile_n``, ``inter_dim % tile_k``,
``sort_block_m % tile_m``, ``tile_m*tile_k % 256``, the LDS sizing) plus the
MX-FP4 layout requirements (``tile_m % 32``, ``tile_m >= 32``, ``tile_k >= 256``).
Keep the two files in sync: if a constraint changes in the kernel builder, update
the matching check below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# gfx -> total LDS bytes available to a single workgroup.  Matches the
# ``_lds_limit`` dict in compile_mixed_moe_gemm1 / 2.
LDS_LIMIT_BYTES = {"gfx950": 163840, "gfx942": 65536}

# Element byte width of the activation operand, keyed by a_dtype.  fp4 and fp8
# both occupy 1 byte in the kernel's sizing math (fp4 is vector-packed 2:1 via
# a_elem_vec_pack, handled separately); fp16 is 2 bytes.
_A_ELEM_BYTES = {"fp8": 1, "fp4": 1, "int8": 1, "fp16": 2}

# Activation vector pack factor (fp4 packs two logical elements per byte).
_A_ELEM_VEC_PACK = {"fp4": 2}


@dataclass
class TileCheck:
    """Result of a legality check for one tile candidate.

    ``legal`` is True iff the kernel builder would accept the candidate.  When
    illegal, ``reason`` is a short machine-readable token (e.g.
    ``"tile_k_bytes_not_div_64"``) and ``detail`` is a human-readable message.
    ``lds_bytes`` is the computed LDS footprint when it could be evaluated.
    """

    legal: bool
    stage: int
    reason: str = ""
    detail: str = ""
    lds_bytes: Optional[int] = None
    params: dict = field(default_factory=dict)

    def as_record(self) -> dict:
        """Flat dict suitable for JSONL/CSV logging of a rejected candidate."""
        rec = {
            "stage": self.stage,
            "legal": self.legal,
            "reason": self.reason,
            "detail": self.detail,
            "lds_bytes": self.lds_bytes,
        }
        rec.update(self.params)
        return rec


def _align(ptr: int, align: int) -> int:
    """Round ``ptr`` up to a multiple of ``align`` (mirrors SmemAllocator._align)."""
    if ptr % align == 0:
        return ptr
    return (ptr + align - 1) // align * align


def _a_elem_bytes(a_dtype: str) -> int:
    if a_dtype not in _A_ELEM_BYTES:
        raise ValueError(f"a_dtype must be one of {sorted(_A_ELEM_BYTES)}, got {a_dtype!r}")
    return _A_ELEM_BYTES[a_dtype]


def stage1_lds_bytes(
    *,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a_dtype: str,
    out_dtype: str = "f16",
    waves_per_eu: int = 4,
    use_cshuffle_epilog: bool = True,
    gpu_arch: str = "gfx950",
) -> int:
    """LDS bytes used by a stage1 config, mirroring compile_mixed_moe_gemm1.

    Follows the ping/pong allocator walk: pong holds max(input, lds_out)+tid,
    ping holds input, with the lds_out auto-split when the standard layout would
    overflow the arch limit, plus the waves_per_eu minimum-LDS padding.
    """
    a_elem_bytes = _a_elem_bytes(a_dtype)
    # FLIR_CK_LDS128 defaults on -> pad_k = 0.
    lds_stride = tile_k
    # NOTE: stage1 sizes the LDS A tile from the FULL lds_stride; unlike stage2 it
    # does NOT divide by a_elem_vec_pack for fp4 here.  The fp4 vec-pack stride
    # halving only applies, conditionally, to an inner async-copy buffer in the
    # kernel body, not to this top-level ping/pong allocation.  See
    # compile_mixed_moe_gemm1: ``_single_x_bytes = tile_m * lds_stride * a_elem_bytes``.

    out_s = str(out_dtype).strip().lower()
    out_is_f32 = out_s in ("f32", "fp32", "float")
    need_quant = out_s in ("fp4", "fp8")
    if need_quant:
        use_cshuffle_epilog = True

    single_x_bytes = tile_m * lds_stride * a_elem_bytes
    cshuffle_elem_bytes = 4 if need_quant else (4 if out_is_f32 else 2)
    lds_out_bytes = cshuffle_elem_bytes * tile_m * tile_n if use_cshuffle_epilog else 0
    lds_tid_bytes = tile_m * 4
    num_waves = min(4, tile_n // 32) if tile_n >= 32 else 0

    global_align = 1024
    std_pong = max(single_x_bytes, lds_out_bytes) + lds_tid_bytes
    std_ping = single_x_bytes
    std_pong_aligned = _align(std_pong, 128)
    std_total = _align(std_pong_aligned, global_align) + _align(std_ping, 128)
    lds_limit = LDS_LIMIT_BYTES.get(gpu_arch, 0)

    split_lds_out = lds_limit > 0 and lds_out_bytes > 0 and std_total > lds_limit and num_waves >= 2

    if split_lds_out:
        half_out_bytes = cshuffle_elem_bytes * tile_m * (tile_n // 2)
        pong_buffer_bytes = max(single_x_bytes, half_out_bytes)
        ping_buffer_bytes = max(single_x_bytes, half_out_bytes)
    else:
        pong_buffer_bytes = max(single_x_bytes, lds_out_bytes)
        ping_buffer_bytes = single_x_bytes

    # Allocator walk: pong = align16(0)+pong_buf, then align4()+tid.
    pong_ptr = _align(0, 16) + pong_buffer_bytes
    pong_ptr = _align(pong_ptr, 4) + lds_tid_bytes
    ping_ptr = _align(0, 16) + ping_buffer_bytes

    if waves_per_eu is not None and waves_per_eu >= 1:
        total_cu_lds = 160 * 1024
        min_lds = total_cu_lds // (waves_per_eu + 1) + 1
        pong_sz = _align(pong_ptr, 128)
        ping_sz = _align(ping_ptr, 128)
        cur_lds = pong_sz + ping_sz
        if cur_lds < min_lds:
            ping_ptr += min_lds - cur_lds

    # Final footprint uses the same global/128 alignment as _std_total.
    return _align(_align(pong_ptr, 128), global_align) + _align(ping_ptr, 128)


def stage2_lds_bytes(
    *,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a_dtype: str,
    use_cshuffle_epilog: bool = True,
) -> int:
    """LDS bytes used by a stage2 config, mirroring compile_mixed_moe_gemm2.

    Stage2 has no lds_out auto-split and no waves_per_eu padding.
    """
    a_elem_bytes = _a_elem_bytes(a_dtype)
    vec_pack = _A_ELEM_VEC_PACK.get(a_dtype, 1)
    lds_stride = tile_k  # pad_k = 0 with FLIR_CK_LDS128 default.
    eff_lds_stride = lds_stride // vec_pack if vec_pack > 1 else lds_stride

    single_x_bytes = tile_m * eff_lds_stride * a_elem_bytes
    cshuffle_elem_bytes = 2  # stage2 f16/bf16
    lds_out_bytes = cshuffle_elem_bytes * tile_m * tile_n if use_cshuffle_epilog else 0
    lds_tid_bytes = tile_m * 4

    pong_buffer_bytes = max(single_x_bytes, lds_out_bytes)
    ping_buffer_bytes = single_x_bytes

    pong_ptr = _align(0, 16) + pong_buffer_bytes
    pong_ptr = _align(pong_ptr, 4) + lds_tid_bytes
    ping_ptr = _align(0, 16) + ping_buffer_bytes
    return pong_ptr + ping_ptr


def _check_stage1(
    *,
    model_dim: int,
    inter_dim: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a_dtype: str,
    out_dtype: str,
    k_batch: int,
    waves_per_eu: int,
    gpu_arch: str,
    params: dict,
) -> TileCheck:
    a_elem_bytes = _a_elem_bytes(a_dtype)

    # MX-FP4 layout requirements (fp4/fp8 weight path).
    if tile_m < 32:
        return TileCheck(
            False, 1, "tile_m_lt_32", f"tile_m={tile_m} < 32 (MX-FP4 layout requires tile_m>=32)", params=params
        )
    if tile_m % 32 != 0:
        return TileCheck(
            False, 1, "tile_m_not_div_32", f"tile_m={tile_m} not divisible by 32 (MX-FP4 layout)", params=params
        )
    if tile_k < 256:
        return TileCheck(
            False, 1, "tile_k_lt_256", f"tile_k={tile_k} < 256 (MX-FP4 layout requires tile_k>=256)", params=params
        )

    if tile_n < 32 or tile_n % 32 != 0:
        return TileCheck(
            False, 1, "tile_n_not_mult_32", f"tile_n={tile_n} must be a positive multiple of 32", params=params
        )

    # tile_k_bytes % 64 (kernel raises otherwise).
    tile_k_bytes = tile_k * a_elem_bytes
    if tile_k_bytes % 64 != 0:
        return TileCheck(
            False, 1, "tile_k_bytes_not_div_64", f"tile_k_bytes={tile_k_bytes} not divisible by 64", params=params
        )

    # total_threads = min(4, tile_n // 32) * 64
    num_waves = min(4, tile_n // 32)
    total_threads = num_waves * 64
    bytes_x_per_tile = tile_m * tile_k * a_elem_bytes
    if bytes_x_per_tile % total_threads != 0:
        return TileCheck(
            False,
            1,
            "tile_load_not_div_total_threads",
            f"tile_m*tile_k*elem_bytes={bytes_x_per_tile} not divisible by total_threads={total_threads}",
            params=params,
        )

    # K-loop coverage: model_dim must be divisible by tile_k (implicit but required).
    if model_dim % tile_k != 0:
        return TileCheck(
            False,
            1,
            "model_dim_not_div_tile_k",
            f"model_dim={model_dim} not divisible by tile_k={tile_k}",
            params=params,
        )

    # Split-K divisibility.
    if k_batch > 1:
        if model_dim % k_batch != 0:
            return TileCheck(
                False,
                1,
                "model_dim_not_div_k_batch",
                f"model_dim={model_dim} not divisible by k_batch={k_batch}",
                params=params,
            )
        k_per_batch = model_dim // k_batch
        if k_per_batch % tile_k != 0:
            return TileCheck(
                False,
                1,
                "k_per_batch_not_div_tile_k",
                f"(model_dim//k_batch)={k_per_batch} not divisible by tile_k={tile_k}",
                params=params,
            )

    # LDS fits the arch limit.
    lds = stage1_lds_bytes(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        a_dtype=a_dtype,
        out_dtype=out_dtype,
        waves_per_eu=waves_per_eu,
        gpu_arch=gpu_arch,
    )
    limit = LDS_LIMIT_BYTES.get(gpu_arch, 0)
    if limit and lds > limit:
        return TileCheck(
            False, 1, "lds_over_limit", f"stage1 LDS {lds} > {gpu_arch} limit {limit}", lds_bytes=lds, params=params
        )

    return TileCheck(True, 1, lds_bytes=lds, params=params)


def _check_stage2(
    *,
    model_dim: int,
    inter_dim: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a_dtype: str,
    sort_block_m: int,
    gpu_arch: str,
    params: dict,
) -> TileCheck:
    a_elem_bytes = _a_elem_bytes(a_dtype)

    # MX-FP4 layout requirements.
    if tile_m < 32:
        return TileCheck(
            False, 2, "tile_m_lt_32", f"tile_m={tile_m} < 32 (MX-FP4 layout requires tile_m>=32)", params=params
        )
    if tile_m % 32 != 0:
        return TileCheck(
            False, 2, "tile_m_not_div_32", f"tile_m={tile_m} not divisible by 32 (MX-FP4 layout)", params=params
        )
    if tile_k < 256:
        return TileCheck(
            False, 2, "tile_k_lt_256", f"tile_k={tile_k} < 256 (MX-FP4 layout requires tile_k>=256)", params=params
        )

    # model_dim % 16 (kernel asserts) and the N-tile coverage model_dim % tile_n.
    if model_dim % 16 != 0:
        return TileCheck(False, 2, "model_dim_not_div_16", f"model_dim={model_dim} not divisible by 16", params=params)
    if model_dim % tile_n != 0:
        return TileCheck(
            False,
            2,
            "model_dim_not_div_tile_n",
            f"model_dim={model_dim} not divisible by tile_n={tile_n}",
            params=params,
        )

    # inter_dim (= stage2 K) must be divisible by tile_k.
    if inter_dim % tile_k != 0:
        return TileCheck(
            False,
            2,
            "inter_dim_not_div_tile_k",
            f"inter_dim={inter_dim} not divisible by tile_k={tile_k}",
            params=params,
        )

    # tile_k_bytes % 64.
    tile_k_bytes = tile_k * a_elem_bytes
    if tile_k_bytes % 64 != 0:
        return TileCheck(
            False, 2, "tile_k_bytes_not_div_64", f"tile_k_bytes={tile_k_bytes} not divisible by 64", params=params
        )

    # total_threads is a fixed 256 in stage2.
    bytes_x_per_tile = tile_m * tile_k * a_elem_bytes
    if bytes_x_per_tile % 256 != 0:
        return TileCheck(
            False,
            2,
            "tile_load_not_div_256",
            f"tile_m*tile_k*elem_bytes={bytes_x_per_tile} not divisible by 256",
            params=params,
        )
    # gmem load mapping: bytes_per_thread must be divisible by 4.
    if (bytes_x_per_tile // 256) % 4 != 0:
        return TileCheck(
            False,
            2,
            "bytes_per_thread_not_div_4",
            f"bytes_per_thread_x={bytes_x_per_tile // 256} not divisible by 4",
            params=params,
        )

    # sort_block_m must be a multiple of tile_m (0 -> equals tile_m, always legal).
    eff_sort_block_m = tile_m if sort_block_m <= 0 else sort_block_m
    if eff_sort_block_m != tile_m and eff_sort_block_m % tile_m != 0:
        return TileCheck(
            False,
            2,
            "sort_block_m_not_mult_tile_m",
            f"sort_block_m={eff_sort_block_m} not a multiple of tile_m={tile_m}",
            params=params,
        )

    # LDS fits the arch limit.
    lds = stage2_lds_bytes(tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, a_dtype=a_dtype)
    limit = LDS_LIMIT_BYTES.get(gpu_arch, 0)
    if limit and lds > limit:
        return TileCheck(
            False, 2, "lds_over_limit", f"stage2 LDS {lds} > {gpu_arch} limit {limit}", lds_bytes=lds, params=params
        )

    return TileCheck(True, 2, lds_bytes=lds, params=params)


def check_tile_config(
    *,
    stage: int,
    model_dim: int,
    inter_dim: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a_dtype: str = "fp4",
    out_dtype: str = "f16",
    k_batch: int = 1,
    waves_per_eu: int = 4,
    sort_block_m: int = 0,
    gpu_arch: str = "gfx950",
) -> TileCheck:
    """Check whether a single tile candidate is legal for ``stage`` (1 or 2).

    Mirrors the pre-compile constraints in ``compile_mixed_moe_gemm1`` /
    ``compile_mixed_moe_gemm2`` so the candidate never reaches a compile the
    kernel would reject.  ``a_dtype`` is ``"fp4"`` for a4w4 and ``"fp8"`` for
    a8w4 (the activation operand); the weight operand is fp4 in both cases.

    Returns a :class:`TileCheck`; ``.legal`` is the accept/reject decision and
    ``.reason`` is a machine-readable token on rejection.
    """
    params = {
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "a_dtype": a_dtype,
        "out_dtype": out_dtype,
        "k_batch": k_batch,
        "waves_per_eu": waves_per_eu,
        "sort_block_m": sort_block_m,
        "gpu_arch": gpu_arch,
    }
    if a_dtype not in _A_ELEM_BYTES:
        return TileCheck(False, stage, "bad_a_dtype", f"a_dtype={a_dtype!r} not supported", params=params)

    if stage == 1:
        return _check_stage1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            a_dtype=a_dtype,
            out_dtype=out_dtype,
            k_batch=k_batch,
            waves_per_eu=waves_per_eu,
            gpu_arch=gpu_arch,
            params=params,
        )
    if stage == 2:
        return _check_stage2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            a_dtype=a_dtype,
            sort_block_m=sort_block_m,
            gpu_arch=gpu_arch,
            params=params,
        )
    return TileCheck(False, stage, "bad_stage", f"stage must be 1 or 2, got {stage}", params=params)


def enumerate_legal_configs(
    *,
    stage: int,
    model_dim: int,
    inter_dim: int,
    a_dtype: str,
    tile_m_choices,
    tile_n_choices,
    tile_k_choices,
    out_dtype: str = "f16",
    k_batch_choices=(1,),
    waves_per_eu_choices=(4,),
    sort_block_m_choices=(0,),
    gpu_arch: str = "gfx950",
    rejected_log: Optional[list] = None,
):
    """Yield every legal tile candidate from the cross product of the choices.

    Rejected candidates are appended (as ``TileCheck.as_record()`` dicts) to
    ``rejected_log`` when provided, so the search never silently drops a
    candidate without a machine-readable reason.
    """
    legal = []
    for tile_m in tile_m_choices:
        for tile_n in tile_n_choices:
            for tile_k in tile_k_choices:
                for k_batch in (k_batch_choices if stage == 1 else (1,)):
                    for waves_per_eu in (waves_per_eu_choices if stage == 1 else (4,)):
                        for sort_block_m in (sort_block_m_choices if stage == 2 else (0,)):
                            res = check_tile_config(
                                stage=stage,
                                model_dim=model_dim,
                                inter_dim=inter_dim,
                                tile_m=tile_m,
                                tile_n=tile_n,
                                tile_k=tile_k,
                                a_dtype=a_dtype,
                                out_dtype=out_dtype,
                                k_batch=k_batch,
                                waves_per_eu=waves_per_eu,
                                sort_block_m=sort_block_m,
                                gpu_arch=gpu_arch,
                            )
                            if res.legal:
                                legal.append(res)
                            elif rejected_log is not None:
                                rejected_log.append(res.as_record())
    return legal
