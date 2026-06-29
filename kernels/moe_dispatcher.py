# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Compile + launch dispatch for the layout-API MXFP4 MoE gemm (BM32, opus-sort); a4w4/a8w4 entry point."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import _to_raw as _raw
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int8, T

from .mxmoe_gemm_v2 import (
    BK,
    BN,
    H_DEFAULT,
    INTER_DEFAULT,
    MAX_M,
    NE,
    TOPK_DEFAULT,
    gemm1_body_v2,
    gemm2_body_v2,
    global_typed_ptr,
    issue_a_load_lds_dt,
    kStages,
    lds_bytes_for,
)
from .tensor_shim import _run_compiled as run_compiled

__all__ = [
    "compile_gemm1_a4w4_port",
    "compile_gemm2_a4w4_port",
    "gemm1_grid",
    "mxfp4_moe_gemm1",
    "mxfp4_moe_gemm2",
]


# ---- gemm1 (up/gate-proj) compile ----
def gemm1_grid(n_tokens, BM=32, NE=NE, TOPK=TOPK_DEFAULT, INTER=INTER_DEFAULT):
    """Host-side grid size (BM=32 active-experts bound)."""
    active = min(n_tokens * TOPK, NE)
    max_m_blocks = (n_tokens * TOPK + active * (BM - 1) + BM - 1) // BM
    return max_m_blocks * ((2 * INTER) // 256)


def compile_gemm1_a4w4_port(
    BM=32,
    use_nt=True,
    inline_quant=False,
    D_HIDDEN=H_DEFAULT,
    D_INTER=INTER_DEFAULT,
    NE=NE,
    interleave=True,
    a_dtype="fp4",
    out_dtype="fp4",
):
    # use_nt IS the B-load cache policy: True -> non-temporal, False -> cached.
    b_nontemporal = use_nt
    if (BM, inline_quant) != (32, False):
        raise AssertionError(
            f"mxfp4_moe_gemm1 supports only (BM=32, inline_quant=False); " f"got (BM={BM}, inline_quant={inline_quant})"
        )
    if a_dtype not in ("fp4", "fp8"):
        raise AssertionError(f"a_dtype must be 'fp4' or 'fp8', got {a_dtype!r}")
    if out_dtype not in ("fp4", "fp8"):
        raise AssertionError(f"out_dtype must be 'fp4' or 'fp8', got {out_dtype!r}")

    K, INTER, NE = D_HIDDEN, D_INTER, NE
    assert K % BK == 0, f"D_HIDDEN (K) must be a multiple of {BK}, got {K}"
    N_OUT = 2 * INTER
    assert N_OUT % BN == 0, f"2*D_INTER (N_OUT) must be a multiple of {BN}, got {N_OUT}"

    KH_TILE_A = BK // (1 if a_dtype == "fp8" else 2)
    lds_bytes = lds_bytes_for(K // BK, KH_TILE_A)  # K_TILES_TOTAL

    gu_tag = "il" if interleave else "sep"
    bnt_tag = "nt" if b_nontemporal else "cached"
    a_tag = "a8" if a_dtype == "fp8" else "a4"
    o_tag = "o8" if out_dtype == "fp8" else "o4"
    name_suffix = f"h{K}_i{INTER}_ne{NE}_bm{BM}_{bnt_tag}_{gu_tag}_{a_tag}{o_tag}_v2"

    @fx.struct
    class SharedStorage:
        buf: fx.Array[Int8, lds_bytes, 16]

    @flyc.kernel(name=f"gemm1_a4w4_port_{name_suffix}", known_block_size=[256, 1, 1])
    def gemm1_kernel(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_sti: fx.Int64,
        i32_ntok: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
    ):
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = arith.index_cast(T.i32, tx)
        bx_i32 = arith.index_cast(T.i32, bx)
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))
        cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(N_OUT // 256)  # * NUM_N_BLOCKS
        if fx.Int32(bx_i32) < bound:
            gemm1_body_v2(
                lds_base_i32,
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_sti,
                arg_aqout,
                arg_ascaleout,
                bx_i32,
                lane,
                wave,
                i32_ntok,
                total_m_blocks,
                K=K,
                INTER=INTER,
                NE=NE,
                interleave=interleave,
                b_nontemporal=b_nontemporal,
                a_dtype=a_dtype,
                out_dtype=out_dtype,
            )

    @flyc.jit
    def launch_gemm1(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_sti: fx.Int64,
        i32_ntok: fx.Int32,
        i32_grid: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
        stream: fx.Stream,
    ):
        grid_x = arith.index_cast(T.index, i32_grid)
        gemm1_kernel(
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_sti,
            i32_ntok,
            arg_aqout,
            arg_ascaleout,
            arg_hidden,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm1


# ---- gemm2 (down-proj) compile ----
def compile_gemm2_a4w4_port(
    BM=32,
    use_nt=False,
    NE=NE,
    N_OUT=H_DEFAULT,
    MAX_M=MAX_M,
    epilog="atomic",
    D_INTER=INTER_DEFAULT,
    D_INTER_REAL=None,
    a_dtype="fp4",
):
    """Compile the gemm2 a4w4 down-proj; only (BM=32, atomic) supported, D_INTER a multiple of BK(256)."""
    if (BM, epilog) != (32, "atomic"):
        raise AssertionError(
            f"mxfp4_moe_gemm2 supports only (BM=32, epilog='atomic'); " f"got (BM={BM}, epilog={epilog})"
        )
    if D_INTER_REAL is not None and D_INTER_REAL != D_INTER:
        raise AssertionError(
            f"mxfp4_moe_gemm2 does not support D_INTER_REAL padding "
            f"(D_INTER_REAL={D_INTER_REAL}, D_INTER={D_INTER})"
        )
    if a_dtype not in ("fp4", "fp8"):
        raise AssertionError(f"a_dtype must be 'fp4' or 'fp8', got {a_dtype!r}")
    K = D_INTER
    assert K % BK == 0, f"D_INTER (gemm2 contraction K = inter_dim) must be a multiple of {BK}, got {K}"
    is_f8 = a_dtype == "fp8"
    KH_TILE_A = BK // (1 if is_f8 else 2)  # A LDS K-tile bytes (fp8 256, fp4 128)
    K_BYTES = K // (1 if is_f8 else 2)  # A row stride bytes (fp8 K, fp4 K//2)
    slot_bytes = BM * KH_TILE_A
    K_TILES_TOTAL = K // BK
    aStages = kStages if K_TILES_TOTAL <= kStages else 3
    lds_bytes = max(BM * BN * 4, aStages * slot_bytes)
    num_n_blocks = N_OUT // 256

    atag = "_a8" if is_f8 else ""
    tag = f"ne{NE}_h{N_OUT}_i{K}_bm{BM}{'_nt' if use_nt else ''}_atomic{atag}_v2"
    name = f"gemm2_a4w4_port_{tag}"

    @fx.struct
    class SharedStorage:
        buf: fx.Array[Int8, lds_bytes, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def gemm2_kernel(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,  # unused (atomic epilog); kept for signature parity
    ):
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = fx.Int32(tx)
        bx_i32 = fx.Int32(bx)
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))

        aq_num = arith.index_cast(T.index, _raw(i32_max_m_blocks)) * fx.Index(BM * K_BYTES)
        aq_rsrc = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_aq)), num_records_bytes=aq_num)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))

        # Preload the first kStages K-tiles (all tiles for the K_TILES<=2 fast path; else the prologue).
        def issue_all_a_loads(m_row0):
            for slot in range_constexpr(kStages):
                issue_a_load_lds_dt(
                    arg_aq,
                    aq_num,
                    lds_base_i32,
                    slot,
                    slot,
                    m_row0,
                    wave,
                    lane,
                    is_f8,
                    KH_TILE_A,
                    K_BYTES,
                )

        # One-shot grid (atomic): issue A->LDS before the cumsum load so HBM latency overlaps the bound check.
        issue_all_a_loads((bx_i32 // num_n_blocks) * fx.Int32(BM))
        rocdl.sched_barrier(0)

        cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
        total_m_blocks = cumsum0 // BM
        bound = total_m_blocks * fx.Int32(num_n_blocks)

        if fx.Int32(bx_i32) < bound:
            gemm2_body_v2(
                lds_base_i32,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_stids,
                arg_sweights,
                i32_M,
                i32_max_m_blocks,
                arg_out,
                bx_i32,
                lane,
                wave,
                aq_rsrc,
                arg_aq,
                use_nt=use_nt,
                NE=NE,
                N_OUT=N_OUT,
                D_INTER=K,
                aStages=aStages,
                a_dtype=a_dtype,
            )

    @flyc.jit
    def launch_gemm2(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,
        stream: fx.Stream,
    ):
        grid_x = arith.index_cast(T.index, i32_max_m_blocks) * fx.Index(num_n_blocks)
        gemm2_kernel(
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            i32_M,
            i32_max_m_blocks,
            arg_out,
            arg_out_scale,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2


# ---- launcher cache + dispatch (compile once per config, fast-dispatch after) ----
G1_CACHE = {}
G2_CACHE = {}


def get_g1(BM, use_nt, inline_quant, D_HIDDEN, D_INTER, NE, topk, interleave, a_dtype, out_dtype):
    # topk is host-grid-only (gemm1_grid); it does not enter the compiled kernel, so it is not a cache-key dim.
    key = (BM, use_nt, inline_quant, D_HIDDEN, D_INTER, NE, interleave, a_dtype, out_dtype)
    launch = G1_CACHE.get(key)
    if launch is None:
        launch = compile_gemm1_a4w4_port(
            BM=BM,
            use_nt=use_nt,
            inline_quant=inline_quant,
            D_HIDDEN=D_HIDDEN,
            D_INTER=D_INTER,
            NE=NE,
            interleave=interleave,
            a_dtype=a_dtype,
            out_dtype=out_dtype,
        )
        G1_CACHE[key] = launch
    return launch


def get_g2(BM, use_nt, NE, D_HIDDEN, epilog, D_INTER, D_INTER_REAL, a_dtype):
    key = (BM, use_nt, NE, D_HIDDEN, epilog, D_INTER, D_INTER_REAL, a_dtype)
    launch = G2_CACHE.get(key)
    if launch is None:
        launch = compile_gemm2_a4w4_port(
            BM=BM,
            use_nt=use_nt,
            NE=NE,
            N_OUT=D_HIDDEN,
            epilog=epilog,
            D_INTER=D_INTER,
            D_INTER_REAL=D_INTER_REAL,
            a_dtype=a_dtype,
        )
        G2_CACHE[key] = launch
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
    """Stage-1 up/gate gemm: A_q x w1 -> inter (packed MXFP4/MXFP8, sorted); buffers pre-allocated by caller."""
    import torch

    launch = get_g1(BM, use_nt, inline_quant, D_HIDDEN, D_INTER, NE, topk, interleave, a_dtype, out_dtype)
    grid = gemm1_grid(n_tokens, BM, NE=NE, TOPK=topk, INTER=D_INTER)
    run_compiled(
        launch,
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
    """Stage-2 down-proj gemm (atomic bf16 epilog): weighted atomic.fadd into pre-zeroed out (opus-sort only)."""
    import torch

    launch = get_g2(BM, use_nt, NE, D_HIDDEN, "atomic", D_INTER, D_INTER_REAL, a_dtype)
    max_m_blocks = (max_sorted + BM - 1) // BM
    out_scale = out  # unused by the atomic epilog; any valid device ptr is fine
    run_compiled(
        launch,
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
    )
    return out
