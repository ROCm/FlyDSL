# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Layout-API MXFP4 MoE gemm (2-stage), opus-sort only.

This is the layout-API replacement for ``mixed_moe_gemm_2stage`` for the MXFP4
a4w4 / a8w4 surface. It consumes the standard (opus-style) sort contract emitted
by ``moe_sorting_kernel`` -- ``sorted_token_ids`` packed ``(topk<<24)|token_id``
with sentinel ``(topk<<24)|M`` -- and needs NO fused-sort extras:

  * gemm1 gathers its A rows straight from ``sorted_token_ids & 0xFFFFFF``
    (the reference ``mixed_moe_gemm_2stage`` gather; padding rows carry M -> the
    buffer-bounds load returns 0).
  * gemm2 uses the atomic bf16 epilogue, scattering per sorted row into the
    output via ``global.atomic.fadd`` weighted by ``sorted_weights`` -- so there
    is no inverse-permutation (``reverse_sorted``) dependency.

Covered surface (BM=32):
  * gemm1: a4w4 + a8w4 (fp8 act), interleave + separated gate, nt/cached B-load,
    out fp4 / fp8.
  * gemm2: atomic epilog, a4w4 + a8w4 (fp8 intermediate).

The MMA + B / B-scale data movement run through the FlyDSL layout API
(``mxfp4_moe_layout``: ``fx.copy`` + ``fx.gemm``); the A-side LDS staging, the
e8m0 scale math, and the atomic epilogue are raw (shared via
``mxfp4_moe_common``).
"""

import flydsl.compiler as flyc
from flydsl._mlir import ir

from .mxfp4_moe_gemm1 import compile_gemm1_a4w4_port, gemm1_grid
from .mxfp4_moe_gemm2 import compile_gemm2_a4w4_port

__all__ = [
    "compile_gemm1_a4w4_port",
    "compile_gemm2_a4w4_port",
    "gemm1_grid",
    "mxfp4_moe_gemm1",
    "mxfp4_moe_gemm2",
]

# -- launcher cache + dispatch (compile once per config, fast-dispatch after) ---
_G1_CACHE = {}
_G2_CACHE = {}


def _run_compiled(exe, args):
    """First call: flyc.compile (compiles + executes + caches the CompiledFunction)
    on ``exe._cf``. Subsequent calls: fast dispatch via the cached function."""
    cf = getattr(exe, "_cf", None)
    if cf is not None:
        cf(*args)
        return
    try:
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    except Exception:
        # JitFunction.__call__ leaks ir.Context on compile failure; clean up so a
        # later call doesn't take the wrong (no-CompilationContext) code path.
        try:
            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        except Exception:
            pass
        raise


def _get_g1(BM, use_nt, inline_quant, D_HIDDEN, D_INTER, NE, topk, interleave,
            a_dtype, out_dtype):
    key = (BM, use_nt, inline_quant, D_HIDDEN, D_INTER, NE, topk, interleave,
           a_dtype, out_dtype)
    launch = _G1_CACHE.get(key)
    if launch is None:
        launch = compile_gemm1_a4w4_port(
            BM=BM, use_nt=use_nt, inline_quant=inline_quant, D_HIDDEN=D_HIDDEN,
            D_INTER=D_INTER, NE=NE, TOPK=topk, interleave=interleave,
            a_dtype=a_dtype, out_dtype=out_dtype,
        )
        _G1_CACHE[key] = launch
    return launch


def _get_g2(BM, use_nt, NE, D_HIDDEN, epilog, D_INTER, D_INTER_REAL, a_dtype):
    key = (BM, use_nt, NE, D_HIDDEN, epilog, D_INTER, D_INTER_REAL, a_dtype)
    launch = _G2_CACHE.get(key)
    if launch is None:
        launch = compile_gemm2_a4w4_port(
            BM=BM, use_nt=use_nt, NE=NE, N_OUT=D_HIDDEN, epilog=epilog,
            D_INTER=D_INTER, D_INTER_REAL=D_INTER_REAL, a_dtype=a_dtype,
        )
        _G2_CACHE[key] = launch
    return launch


def mxfp4_moe_gemm1(
    *,
    a_quant,
    a_scale_sorted_shuffled,
    w1_u8,
    w1_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    sorted_token_ids,
    inter_sorted_quant,
    inter_sorted_shuffled_scale,
    hidden_states,
    n_tokens,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    BM=32,
    use_nt=True,
    inline_quant=False,
    interleave=True,
    a_dtype="fp4",
    out_dtype="fp4",
    stream=None,
):
    """Stage-1 up/gate gemm: A_q x w1 -> inter (packed MXFP4 / MXFP8, sorted layout).

    Buffers are pre-allocated by the caller. w1_u8 / w1_scale_u8 must be uint8
    views. ``sorted_token_ids`` is the opus-sort output (gemm1 masks it to the
    token id internally).
    """
    import torch

    launch = _get_g1(BM, use_nt, inline_quant, D_HIDDEN, D_INTER, NE, topk,
                     interleave, a_dtype, out_dtype)
    grid = gemm1_grid(n_tokens, BM, NE=NE, TOPK=topk, INTER=D_INTER)
    _run_compiled(
        launch,
        (
            a_quant.data_ptr(),
            a_scale_sorted_shuffled.data_ptr(),
            w1_u8.data_ptr(),
            w1_scale_u8.data_ptr(),
            sorted_expert_ids.data_ptr(),
            cumsum_tensor.data_ptr(),
            sorted_token_ids.data_ptr(),
            n_tokens,
            grid,
            inter_sorted_quant.data_ptr(),
            inter_sorted_shuffled_scale.data_ptr(),
            hidden_states.data_ptr(),
            torch.cuda.current_stream() if stream is None else stream,
        ),
    )
    return inter_sorted_quant, inter_sorted_shuffled_scale


def mxfp4_moe_gemm2(
    *,
    inter_sorted_quant,
    inter_sorted_shuffled_scale,
    w2_u8,
    w2_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    sorted_token_ids,
    sorted_weights,
    out,
    M_logical,
    max_sorted,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    BM=32,
    use_nt=False,
    a_dtype="fp4",
    D_INTER_REAL=None,
    stream=None,
):
    """Stage-2 down-proj gemm (atomic bf16 epilog): scatters per sorted row into
    ``out`` via weighted ``global.atomic.fadd`` (opus-sort only, no reverse_sorted).

    ``out`` MUST be pre-zeroed ([M, D_HIDDEN] bf16) -- the opus sort zeroes its
    ``moe_buf`` for exactly this accumulation.
    """
    import torch

    launch = _get_g2(BM, use_nt, NE, D_HIDDEN, "atomic", D_INTER, D_INTER_REAL,
                     a_dtype)
    max_m_blocks = (max_sorted + BM - 1) // BM
    out_scale = out  # unused by the atomic epilog; any valid device ptr is fine
    _run_compiled(
        launch,
        (
            inter_sorted_quant.data_ptr(),
            inter_sorted_shuffled_scale.data_ptr(),
            w2_u8.data_ptr(),
            w2_scale_u8.data_ptr(),
            sorted_expert_ids.data_ptr(),
            cumsum_tensor.data_ptr(),
            sorted_token_ids.data_ptr(),
            sorted_weights.data_ptr(),
            M_logical,
            max_m_blocks,
            out.data_ptr(),
            out_scale.data_ptr(),
            torch.cuda.current_stream() if stream is None else stream,
        ),
    )
    return out
