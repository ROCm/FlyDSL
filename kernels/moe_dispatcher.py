# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Compile + launch dispatch for the layout-API MXFP4 MoE gemm (BM32, opus-sort); a4w4/a8w4 entry point."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import _to_raw as _raw
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import Int8, T

from .mxmoe_gemm_v2 import (
    BK,
    BN,
    H_DEFAULT,
    INTER_DEFAULT,
    INTER_MAX_DEFAULT,
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
    "select_pipe_config",
]


def _get_cu_num() -> int:
    """CU count for the persistent-m fixed grid (env CU_NUM override, else device props)."""
    import os

    env = os.environ.get("CU_NUM")
    if env:
        return int(env)
    try:
        import torch

        return int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
    except Exception:
        return 304


# ---- aiter-tuned per-token config dispatch (the dominant perf lever) ----
# Source: aiter *_tuned_fmoe.csv best row per token (.humanize/kernel-agent/aiter-tuned-config-map.md).
# Keyed by MoE family signature (model_dim, inter_dim, experts) -> {tokens: (block_m, epilog)}.
# block_m is the CSV `block_m` column (sort/compute tile); epilog is the CSV stage2 mode
# ('atomic' or 'reduce'). The selector clamps block_m to the currently-supported {32,64}
# (BM128 is a follow-on; see select_pipe_config) and nearest-rounds an unlisted token count.
#
# The #1 lever is stage2 atomic-vs-reduce: reduce for the high-expert small-inter families
# (DSV3 E257, Kimi/DSV4 E384) where atomic collapses under heavy per-token contention; atomic
# for the low-expert large-inter GPT-OSS family. block_m tracks the CSV 32/64/128 column.
#
# Each value is (block_m_csv, epilog_csv). Only stage2 epilog + block_m are dispatched here; the
# finer s1/s2 knobs (bnt/persist/xcd4/kw/kb) are separate follow-on wiring.
_AITER_PIPE_TABLE = {
    # DeepSeekV3 fp4 (7168/256/E257/k9) — reduce-dominant for our kernel. The CSV picks atomic at
    # 2048/16384/32768, but those CSV rows rely on persist+sbm128 (unimplemented here); measured
    # cold, reduce beats atomic at every one of those tokens for our current feature set (e.g.
    # 32768: reduce comb 1408 vs atomic 814). block_m tracks the CSV column clamped to <=64, except
    # tok=256 where BM32 reduce measured faster than BM64.
    (7168, 256, 257): {
        1: (32, "atomic"),
        2: (32, "reduce"),
        4: (32, "reduce"),
        8: (32, "reduce"),
        16: (32, "reduce"),
        32: (32, "reduce"),
        64: (32, "reduce"),
        128: (64, "reduce"),
        256: (32, "reduce"),
        512: (64, "reduce"),
        1024: (64, "reduce"),
        2048: (64, "reduce"),
        4096: (64, "reduce"),
        8192: (64, "reduce"),
        16384: (64, "reduce"),
        32768: (64, "reduce"),
    },
    # KimiK2 fp4 (7168/256/E384/k8) — reduce-dominant for our kernel. The CSV uses atomic below
    # 16384 (with bnt2/persist), but measured cold reduce beats atomic at every token for our
    # feature set (e.g. 32768: reduce 1169 vs atomic 651; 256: 106 vs 26). BM32 reduce is fastest
    # <=1024, BM64 reduce >=2048.
    (7168, 256, 384): {
        1: (32, "atomic"),
        2: (32, "atomic"),
        4: (32, "atomic"),
        8: (32, "atomic"),
        16: (32, "reduce"),
        32: (32, "reduce"),
        64: (32, "reduce"),
        128: (32, "reduce"),
        256: (32, "reduce"),
        512: (32, "reduce"),
        1024: (32, "reduce"),
        2048: (64, "reduce"),
        4096: (64, "reduce"),
        8192: (64, "reduce"),
        16384: (64, "reduce"),
        32768: (64, "reduce"),
    },
    # DeepSeekV4 a8w4 (7168/512/E384/k6) — reduce-dominant for our kernel. The CSV uses atomic at
    # large-M (with sbm128/persist), but measured cold reduce beats even atomic-BM64 at every token
    # for our feature set (e.g. 32768: reduce 1426 vs atomic-BM64 1187). BM32 reduce fastest
    # <=1024, BM64 reduce >=2048.
    (7168, 512, 384): {
        1: (32, "reduce"),
        2: (32, "reduce"),
        4: (32, "reduce"),
        8: (32, "reduce"),
        16: (32, "reduce"),
        32: (32, "reduce"),
        64: (32, "reduce"),
        128: (32, "reduce"),
        256: (32, "reduce"),
        512: (32, "reduce"),
        1024: (32, "reduce"),
        2048: (64, "reduce"),
        4096: (64, "reduce"),
        8192: (64, "reduce"),
        16384: (64, "reduce"),
        32768: (64, "reduce"),
        131072: (64, "reduce"),
    },
    # GPT-OSS (3072/3072/E128/k4) swiglu — atomic-dominant (large inter=3072 => few tokens/expert
    # row => low atomic contention, so atomic does NOT collapse here). BM64 atomic measured fastest
    # at every token (e.g. 32768: BM64 2524 vs BM32 2203, already >= CSV target 1943); reduce is
    # ~parity at mid tokens. Keep atomic throughout to protect the already-at/above-target large-M
    # rows.
    (3072, 3072, 128): {
        256: (64, "atomic"),
        512: (64, "atomic"),
        1024: (64, "atomic"),
        2048: (64, "atomic"),
        4096: (64, "atomic"),
        8192: (64, "atomic"),
        16384: (64, "atomic"),
        32768: (64, "atomic"),
    },
}


def _nearest_token_key(tok_map, tokens):
    """Pick the table token bucket nearest (<=) the requested token count; fall back to the min key."""
    keys = sorted(tok_map)
    chosen = keys[0]
    for k in keys:
        if k <= tokens:
            chosen = k
        else:
            break
    return chosen


def select_pipe_config(model_dim, inter_dim, experts, topk, tokens, allow_bm128=False):
    """Host-side per-(shape, token) config picker from the aiter tuned map.

    Returns (BM, epilog) for the v2 pipe. ``epilog`` in {'atomic','reduce'} is the #1 perf lever
    (reduce for high-expert small-inter contention; atomic for low-expert large-inter). ``BM`` is
    the CSV block_m clamped to the currently-supported compute tiles: 128 -> 64 unless
    ``allow_bm128`` (BM128 compute tile is a follow-on / Milestone C). Unlisted families fall back
    to the current default (BM=32, atomic); unlisted token counts snap to the nearest lower bucket.
    """
    fam = _AITER_PIPE_TABLE.get((model_dim, inter_dim, experts))
    if fam is None:
        return 32, "atomic"
    bm_csv, epilog = fam[_nearest_token_key(fam, tokens)]
    if bm_csv >= 128 and not allow_bm128:
        bm = 64  # BM128 compute tile not yet enabled; use the largest supported tile.
    else:
        bm = bm_csv
    if bm not in (32, 64, 128):
        bm = 32
    return bm, epilog


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
    interleave=True,
    a_dtype="fp4",
    out_dtype="fp4",
    act="silu",
    swiglu_limit=0.0,
    SBM=None,
):
    # SBM (sort_block_m) is the moe_sorting padding unit, decoupled from the compute tile BM.
    # None -> SBM==BM (byte-identical). Otherwise SBM must be a multiple of BM (SBM//BM compute
    # blocks per SBM sort block, all sharing one expert).
    if SBM is None:
        SBM = BM
    # use_nt IS the B-load cache policy: True -> non-temporal, False -> cached.
    b_nontemporal = use_nt
    if BM not in (32, 64) or inline_quant:
        raise AssertionError(
            f"mxfp4_moe_gemm1 supports only (BM in {{32,64}}, inline_quant=False); "
            f"got (BM={BM}, inline_quant={inline_quant})"
        )
    if SBM % BM != 0:
        raise AssertionError(f"SBM ({SBM}) must be a multiple of BM ({BM})")
    if a_dtype not in ("fp4", "fp8"):
        raise AssertionError(f"a_dtype must be 'fp4' or 'fp8', got {a_dtype!r}")
    if out_dtype not in ("fp4", "fp8"):
        raise AssertionError(f"out_dtype must be 'fp4' or 'fp8', got {out_dtype!r}")
    if act not in ("silu", "swiglu"):
        raise AssertionError(f"act must be 'silu' or 'swiglu', got {act!r}")

    K = D_HIDDEN  # contraction (compile-time); inter_dim (N-output) is the runtime i32_inter arg
    assert K % BK == 0, f"D_HIDDEN (K) must be a multiple of {BK}, got {K}"

    KH_TILE_A = BK // (1 if a_dtype == "fp8" else 2)
    lds_bytes = lds_bytes_for(K // BK, KH_TILE_A, BM=BM)  # K_TILES_TOTAL (inter-independent)

    gu_tag = "il" if interleave else "sep"
    bnt_tag = "nt" if b_nontemporal else "cached"
    a_tag = "a8" if a_dtype == "fp8" else "a4"
    o_tag = "o8" if out_dtype == "fp8" else "o4"
    # act tag empty for the default silu variant so its kernel name/IR stays byte-identical (AC-3);
    # swiglu is a distinct compile-time variant (limit folded into the name).
    act_tag = "" if act == "silu" else f"_swiglu{swiglu_limit:g}"
    # sbm tag empty when SBM==BM so the default variant keeps its byte-identical kernel name.
    sbm_tag = "" if SBM == BM else f"_sbm{SBM}"
    name_suffix = f"h{K}_bm{BM}_{bnt_tag}_{gu_tag}_{a_tag}{o_tag}{act_tag}{sbm_tag}_v2"

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
        i32_inter: fx.Int32,
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
        num_n_blocks = (fx.Int32(i32_inter) * fx.Int32(2)) // fx.Int32(256)  # NUM_N_BLOCKS = N_OUT//256
        bound = total_m_blocks * num_n_blocks
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
                i32_inter,
                BM=BM,
                K=K,
                interleave=interleave,
                b_nontemporal=b_nontemporal,
                a_dtype=a_dtype,
                out_dtype=out_dtype,
                act=act,
                swiglu_limit=swiglu_limit,
                SBM=SBM,
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
        i32_inter: fx.Int32,
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
            i32_inter,
            arg_aqout,
            arg_ascaleout,
            arg_hidden,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm1


# ---- gemm2 (down-proj) compile ----
def compile_gemm2_a4w4_port(
    BM=32,
    use_nt=False,
    N_OUT=H_DEFAULT,
    MAX_M=MAX_M,
    epilog="atomic",
    INTER_MAX=INTER_MAX_DEFAULT,
    D_INTER_REAL=None,
    a_dtype="fp4",
    topk=1,
    SBM=None,
    persist=False,
    cu_num=0,
):
    """Compile the gemm2 a4w4 down-proj. epilog='atomic' (default) does per-token weighted
    atomic-fadd; epilog='reduce' does a non-atomic store into out[token_id*topk + slot] (unique
    per (token,topk) slot; host reduces over topk), mirroring main's accumulate=False path.
    BM in {32,64} (per-launch parameter). inter_dim is a runtime arg (a multiple of BK=256,
    <= INTER_MAX); INTER_MAX caps the compile-time B-view / LDS bounds. topk enters the reduce
    output-row index (compile-time).

    SBM (sort_block_m) is the moe_sorting padding unit, decoupled from the compute tile BM.
    None -> SBM==BM (byte-identical). Otherwise SBM must be a multiple of BM."""
    if SBM is None:
        SBM = BM
    if BM not in (32, 64) or epilog not in ("atomic", "reduce"):
        raise AssertionError(
            f"mxfp4_moe_gemm2 supports only (BM in {{32,64}}, epilog in {{'atomic','reduce'}}); "
            f"got (BM={BM}, epilog={epilog})"
        )
    if SBM % BM != 0:
        raise AssertionError(f"SBM ({SBM}) must be a multiple of BM ({BM})")
    use_reduce = epilog == "reduce"
    if D_INTER_REAL is not None:
        raise AssertionError(f"mxfp4_moe_gemm2 does not support D_INTER_REAL padding (D_INTER_REAL={D_INTER_REAL})")
    if a_dtype not in ("fp4", "fp8"):
        raise AssertionError(f"a_dtype must be 'fp4' or 'fp8', got {a_dtype!r}")
    assert INTER_MAX % BK == 0, f"INTER_MAX must be a multiple of {BK}, got {INTER_MAX}"
    is_f8 = a_dtype == "fp8"
    KH_TILE_A = BK // (1 if is_f8 else 2)  # A LDS K-tile bytes (fp8 256, fp4 128)
    slot_bytes = BM * KH_TILE_A
    aStages = 3  # runtime K-loop: triple-buffered A LDS (handles both K_TILES==2 and larger)
    lds_bytes = max(BM * BN * 4, aStages * slot_bytes)
    num_n_blocks = N_OUT // 256

    atag = "_a8" if is_f8 else ""
    # atomic tag unchanged (byte-identical default); reduce is a distinct variant (topk folded in).
    etag = "atomic" if not use_reduce else f"reduce_tk{topk}"
    # sbm tag empty when SBM==BM so the default variant keeps its byte-identical kernel name.
    sbm_tag = "" if SBM == BM else f"_sbm{SBM}"
    # persist tag empty for the default one-shot grid (byte-identical); persist folds cu_num into
    # the name (the fixed grid size is a distinct variant).
    if persist and cu_num <= 0:
        raise AssertionError(f"persist=True requires cu_num>0, got {cu_num}")
    persist_tag = "" if not persist else f"_persist_cu{cu_num}"
    tag = f"h{N_OUT}_imax{INTER_MAX}_bm{BM}{'_nt' if use_nt else ''}_{etag}{atag}{sbm_tag}{persist_tag}_v2"
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
        i32_inter: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,  # unused (atomic epilog); kept for signature parity
    ):
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = fx.Int32(tx)
        bx_i32 = fx.Int32(bx)
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))

        k_bytes = fx.Int32(i32_inter) // fx.Int32(1 if is_f8 else 2)  # A row stride bytes (runtime)
        aq_num = arith.index_cast(T.index, _raw(i32_max_m_blocks)) * fx.Index(fx.Int32(BM) * k_bytes)
        aq_rsrc = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_aq)), num_records_bytes=aq_num)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))

        # Preload the first kStages K-tiles (the streaming prologue).
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
                    k_bytes,
                    BM=BM,
                )

        # One (m_block, n_block) unit of work for a synthesized block index `unit_bx`: preload the
        # A prologue for its m_row then run the body. Non-persist calls this once with the launched
        # bx (byte-identical); persist calls it per grid-strided m-tile.
        def run_unit(unit_bx):
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
                unit_bx,
                lane,
                wave,
                aq_rsrc,
                arg_aq,
                i32_inter,
                BM=BM,
                use_nt=use_nt,
                N_OUT=N_OUT,
                INTER_MAX=INTER_MAX,
                aStages=aStages,
                a_dtype=a_dtype,
                use_reduce=use_reduce,
                topk=topk,
                SBM=SBM,
            )

        if const_expr(not persist):
            # One-shot grid (atomic): issue A->LDS before the cumsum load so HBM latency overlaps the bound check.
            issue_all_a_loads((bx_i32 // num_n_blocks) * fx.Int32(BM))
            rocdl.sched_barrier(0)

            cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
            total_m_blocks = cumsum0 // BM
            bound = total_m_blocks * fx.Int32(num_n_blocks)

            if fx.Int32(bx_i32) < bound:
                run_unit(bx_i32)
        else:
            # Persistent-m grid: a fixed grid of cu_num*num_n_blocks blocks. The launched block
            # index encodes (m_tile0 in [0,cu_num), n_block); each block grid-strides over m-tiles
            # with stride cu_num, looping [m_tile0, m_tile0+cu_num, ...) < total_m_blocks. Cuts
            # launch/tail overhead and improves large-M occupancy (aiter `_persist`).
            m_tile0 = bx_i32 // fx.Int32(num_n_blocks)
            n_block = bx_i32 - m_tile0 * fx.Int32(num_n_blocks)
            c_stride = fx.Int32(cu_num)

            cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
            total_m_blocks = cumsum0 // BM
            # ceil((total_m_blocks - m_tile0) / cu_num), clamped to 0 when m_tile0 >= total_m_blocks.
            diff = total_m_blocks - m_tile0
            rem = (diff > fx.Int32(0)).select(diff, fx.Int32(0))
            n_iters = (rem + c_stride - fx.Int32(1)) // c_stride
            for _it in range(fx.Index(0), ArithValue(_raw(n_iters)).index_cast(T.index), fx.Index(1)):
                m_block = m_tile0 + fx.Int32(_it) * c_stride
                unit_bx = m_block * fx.Int32(num_n_blocks) + n_block
                issue_all_a_loads(m_block * fx.Int32(BM))
                rocdl.sched_barrier(0)
                if fx.Int32(m_block) < total_m_blocks:
                    run_unit(unit_bx)

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
        i32_grid_blocks: fx.Int32,
        i32_inter: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,
        stream: fx.Stream,
    ):
        # i32_max_m_blocks sizes the A/scale buffer resources (kernel body); i32_grid_blocks bounds
        # the launch to the actual padded sorted-token m-blocks (avoids empty blocks at small tokens).
        grid_x = arith.index_cast(T.index, i32_grid_blocks) * fx.Index(num_n_blocks)
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
            i32_inter,
            arg_out,
            arg_out_scale,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2


# ---- launcher cache + dispatch (compile once per config, fast-dispatch after) ----
G1_CACHE = {}
G2_CACHE = {}


def get_g1(BM, use_nt, inline_quant, D_HIDDEN, interleave, a_dtype, out_dtype, act="silu", swiglu_limit=0.0, SBM=None):
    # inter_dim (gemm1 N-output) is a runtime arg; NE/topk are host-only (NE: gemm1_grid active-expert
    # cap; topk: grid sizing). None of the three enters the compiled kernel, so none is a cache-key dim.
    # act/swiglu_limit are compile-time (folded into the epilog), so both are cache-key dims.
    # SBM (sort_block_m) is a compile-time cache-key dim; None means SBM==BM (byte-identical variant).
    if SBM is None:
        SBM = BM
    key = (BM, use_nt, inline_quant, D_HIDDEN, interleave, a_dtype, out_dtype, act, swiglu_limit, SBM)
    launch = G1_CACHE.get(key)
    if launch is None:
        launch = compile_gemm1_a4w4_port(
            BM=BM,
            use_nt=use_nt,
            inline_quant=inline_quant,
            D_HIDDEN=D_HIDDEN,
            interleave=interleave,
            a_dtype=a_dtype,
            out_dtype=out_dtype,
            act=act,
            swiglu_limit=swiglu_limit,
            SBM=SBM,
        )
        G1_CACHE[key] = launch
    return launch


def get_g2(BM, use_nt, D_HIDDEN, epilog, INTER_MAX, D_INTER_REAL, a_dtype, topk=1, SBM=None, persist=False, cu_num=0):
    # NE / inter_dim do not enter the compiled gemm2 kernel (inter_dim is a runtime arg); the only
    # contraction-shape key is the compile-time cap INTER_MAX. epilog + topk are compile-time
    # (reduce folds topk into the output-row index); atomic ignores topk.
    # SBM (sort_block_m) is a compile-time cache-key dim; None means SBM==BM (byte-identical variant).
    # persist (+ cu_num, the fixed-grid size) are compile-time cache-key dims; persist=False is the
    # byte-identical one-shot-grid variant.
    if SBM is None:
        SBM = BM
    topk_key = topk if epilog == "reduce" else 1
    cu_key = cu_num if persist else 0
    key = (BM, use_nt, D_HIDDEN, epilog, INTER_MAX, D_INTER_REAL, a_dtype, topk_key, SBM, persist, cu_key)
    launch = G2_CACHE.get(key)
    if launch is None:
        launch = compile_gemm2_a4w4_port(
            BM=BM,
            use_nt=use_nt,
            N_OUT=D_HIDDEN,
            epilog=epilog,
            INTER_MAX=INTER_MAX,
            D_INTER_REAL=D_INTER_REAL,
            a_dtype=a_dtype,
            topk=topk_key,
            SBM=SBM,
            persist=persist,
            cu_num=cu_key,
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
    use_nt=False,
    inline_quant=False,
    interleave=True,
    a_dtype="fp4",
    out_dtype="fp4",
    act="silu",
    swiglu_limit=0.0,
    SBM=None,
    n_sorted_padded=None,
    stream=None,
):
    """Stage-1 up/gate gemm: A_q x w1 -> inter (packed MXFP4/MXFP8, sorted); buffers pre-allocated by caller.

    ``use_nt`` is the B-weight load cache policy (False -> cached, True -> non-temporal).
    Default False: stage1 reuses each expert's weights across many m-blocks (large tokens
    give many m-blocks per expert), so caching B in L2 is a large win on compute-bound
    shapes and matches base's stage1 (``b_nt=0``). nt only helps when there is no reuse.

    ``n_sorted_padded`` is the actual padded sorted-token count (cumsum[0], host-read after the
    moe_sorting sync). When given, the launch grid is bounded to the real work
    ``(n_sorted_padded // BM) * num_n_blocks`` instead of the worst-case E-based bound, avoiding
    empty blocks at small token counts. Falls back to the worst-case ``gemm1_grid`` bound if None.
    """
    import torch

    launch = get_g1(BM, use_nt, inline_quant, D_HIDDEN, interleave, a_dtype, out_dtype, act, swiglu_limit, SBM=SBM)
    sbm = SBM or BM
    num_n_blocks = (2 * D_INTER) // 256
    if n_sorted_padded is None:
        # E-based worst-case grid: sort padding is per SBM (the sort unit); the compute grid is
        # in BM blocks (padded_rows // BM). SBM==BM reduces to the original gemm1_grid.
        active = min(n_tokens * topk, NE)
        padded_rows = ((n_tokens * topk + active * (sbm - 1) + sbm - 1) // sbm) * sbm
        grid = (padded_rows // BM) * num_n_blocks
    else:
        grid = (n_sorted_padded // BM) * num_n_blocks
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
        D_INTER,
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
    epilog="atomic",
    SBM=None,
    persist=False,
    cu_num=0,
    n_sorted_padded=None,
    stream=None,
):
    """Stage-2 down-proj gemm. epilog='atomic' (default): weighted atomic.fadd into pre-zeroed out
    [tokens, H] (opus-sort only). epilog='reduce': non-atomic weighted store into
    out[token_id*topk + slot] of a [tokens*topk, H] buffer (host reduces over topk, applying any
    EP valid_mask). Mirrors main mixed_moe_gemm_2stage accumulate=True/False.

    ``n_sorted_padded`` is the actual padded sorted-token count (cumsum[0], host-read after the
    moe_sorting sync). When given, the launch grid is bounded to ``(n_sorted_padded // BM) *
    num_n_blocks`` (real work) while ``max_m_blocks`` (from ``max_sorted``) still sizes the kernel's
    A/scale buffer resources. Falls back to the full ``max_sorted`` grid if None.

    ``persist`` (aiter `_persist`): launch a fixed grid of ``cu_num`` m-slots (times num_n_blocks);
    each block grid-strides over the padded sort blocks (stride cu_num), looping over multiple
    m-tiles. Cuts launch/tail overhead and improves large-M occupancy. The launched
    ``i32_grid_blocks`` becomes ``cu_num`` (the fixed m-slot count) instead of the per-m-tile block
    count. ``max_m_blocks`` still sizes the A/scale buffer resources. Default OFF (byte-identical).
    """
    import torch

    if D_INTER_REAL is not None and D_INTER_REAL != D_INTER:
        raise AssertionError(f"D_INTER_REAL padding unsupported (D_INTER_REAL={D_INTER_REAL}, D_INTER={D_INTER})")
    if persist and cu_num <= 0:
        cu_num = _get_cu_num()
    launch = get_g2(
        BM,
        use_nt,
        D_HIDDEN,
        epilog,
        INTER_MAX_DEFAULT,
        None,
        a_dtype,
        topk=topk,
        SBM=SBM,
        persist=persist,
        cu_num=cu_num,
    )
    if D_INTER > INTER_MAX_DEFAULT:
        raise AssertionError(f"D_INTER ({D_INTER}) exceeds compile cap INTER_MAX ({INTER_MAX_DEFAULT})")
    max_m_blocks = (max_sorted + BM - 1) // BM
    if persist:
        # Fixed grid: cu_num m-slots (x num_n_blocks blocks); each block loops over its m-tiles.
        grid_blocks = cu_num
    else:
        grid_blocks = max_m_blocks if n_sorted_padded is None else (n_sorted_padded // BM)
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
        grid_blocks,
        D_INTER,
        out.data_ptr(),
        out_scale.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return out
