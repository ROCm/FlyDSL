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
# Each value is (block_m_csv, epilog_csv). stage2 epilog + block_m come from this table; the
# stage1-only BM128 tile and gemm2 persist are layered on top per-family (see
# _STAGE1_BM128_MIN_TOK / _PERSIST_MIN_TOK and select_pipe_config). The remaining finer knobs
# (bnt/xcd4/kw/kb) are separate follow-on wiring.
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


def _norm_sbm(SBM, BM):
    """Resolve SBM (sort_block_m): None -> SBM==BM (byte-identical). Both stages / the cache key
    and the compile path share this normalization so None and BM map to the same variant."""
    return BM if SBM is None else SBM


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


# ---- stage1-only BM128 + persist wiring (evidence: aiter-tuned-config-map.md) ----
# The high-expert small-inter families (DSV3 E257, Kimi/DSV4 E384) run the aiter CSV stage1 at
# tile_m=128 (``t128x...``) from ~4096 tokens up while their stage2 stays at tile_m=64 with
# sbm128 (BM128 regresses stage2/GPT-OSS per F1 — it is a stage1-only lever). We give gemm1 a
# BM128 compute tile and keep gemm2 at its measured-optimal <=64 tile; the shared sort unit
# becomes SBM=lcm(bm_stage1, bm_stage2)=128 so both stages agree on padding/expert-id lookup.
# Keyed by family signature -> min token count to enable stage1 BM128. Gated by allow_bm128.
_STAGE1_BM128_MIN_TOK = {
    (7168, 256, 257): 4096,  # DeepSeekV3 fp4
    (7168, 256, 384): 4096,  # KimiK2 fp4
    (7168, 512, 384): 4096,  # DeepSeekV4 a8w4
}

# persist (aiter `_persist`, gemm2 fixed-grid grid-stride): ON for the high-expert large-M reduce
# rows (DSV3/Kimi fp4) where it cuts launch/tail overhead and lifts large-M occupancy (F2:
# +5-17%); OFF for GPT-OSS (E128 large-inter — no benefit, keep the byte-identical one-shot grid).
# Keyed by family signature -> min token count to enable gemm2 persist.
#
# NOTE: the a8w4/fp8-A gemm2 persist path is a KNOWN-BROKEN F2 combo (produces cos=0 at large M,
# reproduces on rlcr/moe-persist-sbm alone — the multi-iteration grid-stride corrupts the fp8-A
# accumulator/LDS state). The fp4 persist path is correct at all tokens (validated to 32768).
# The dispatcher therefore enables persist ONLY for the fp4 families; DeepSeekV4 (a8w4, sig
# (7168,512,384)) is deliberately excluded. The knob stays manually selectable (MXFP4_PERSIST=1)
# but a8w4+persist is guarded fail-fast in compile_gemm2_a4w4_port so it can never silently ship
# garbage. Re-add DSV4 here once the fp8-A persist path is fixed.
_PERSIST_MIN_TOK = {
    (7168, 256, 257): 4096,  # DeepSeekV3 fp4
    (7168, 256, 384): 4096,  # KimiK2 fp4
}

# ---- tiny-M gemm1 BN=64 + k_wave=4 dispatch (evidence: bn64-kw4.md) ----
# The high-expert small-inter fp4 families (DSV3 E257, Kimi E384; inter=256 -> N_OUT=512) are
# block-count bound at tiny M: with BN=256 there are only N_OUT//256 = 2 N-blocks, so few blocks
# cover the GPU. BN=64 gives N_OUT//64 = 8 N-blocks (4x coverage) and pairs with k_wave=4 (nnw=1)
# to keep the K reduction busy -> DSV3 fp4 m=2 gemm1 ~1.5x (7.99 vs 12.58us). The coverage win ends
# by m~=4 (enough M-blocks to fill the GPU), so BN=256+kw1 otherwise. Keyed by family signature ->
# max token count (inclusive) to enable BN=64+kw4. gemm2 is BN-independent (unchanged).
_TINYM_BN64_MAX_TOK = {
    (7168, 256, 257): 2,  # DeepSeekV3 fp4
    (7168, 256, 384): 2,  # KimiK2 fp4
}


def select_pipe_config(model_dim, inter_dim, experts, topk, tokens, allow_bm128=False):
    """Host-side per-(shape, token) config picker from the aiter tuned map.

    Returns ``(BM, epilog, bm_stage1, persist, bn, k_wave)`` for the v2 pipe:

    * ``BM`` -- the stage2/compute tile, the CSV block_m clamped to the currently-supported
      compute tiles <=64 (128 -> 64). BM128 is a stage1-only lever (it regresses stage2), so the
      stage2 tile never exceeds 64 here.
    * ``epilog`` in {'atomic','reduce'} -- the #1 perf lever (reduce for high-expert small-inter
      contention; atomic for low-expert large-inter).
    * ``bm_stage1`` -- the gemm1 compute tile: 128 for the high-expert small-inter families at
      large M (``_STAGE1_BM128_MIN_TOK``) when ``allow_bm128``, else == ``BM``. When
      ``bm_stage1 != BM`` the caller must use SBM = lcm(bm_stage1, BM) (=128) as the shared sort
      unit so both stages agree on padding / expert-id lookup.
    * ``persist`` -- enable the gemm2 persistent-m grid for the high-expert large-M fp4 families
      (``_PERSIST_MIN_TOK``, DSV3/Kimi); OFF for GPT-OSS (byte-identical one-shot grid) and for the
      a8w4/fp8-A families (the fp8-A persist path is a known-broken F2 combo; see _PERSIST_MIN_TOK).
    * ``bn, k_wave`` -- the gemm1 fused gate|up N-tile and intra-block K-slice. (64, 4) for the
      high-expert small-inter fp4 families at tiny M (``_TINYM_BN64_MAX_TOK``, block-count bound),
      (256, 1) otherwise (the shipped default, byte-identical).

    Unlisted families fall back to the current default (BM=32, atomic, bm_stage1=32, persist=off,
    bn=256, k_wave=1); unlisted token counts snap to the nearest lower bucket.
    """
    sig = (model_dim, inter_dim, experts)
    fam = _AITER_PIPE_TABLE.get(sig)
    if fam is None:
        return 32, "atomic", 32, False, 256, 1
    bm_csv, epilog = fam[_nearest_token_key(fam, tokens)]
    # stage2/compute tile: BM128 is stage1-only -> stage2 caps at 64.
    bm = 64 if bm_csv >= 128 else bm_csv
    if bm not in (32, 64):
        bm = 32
    # stage1 tile: BM128 for the listed high-expert families at large M (allow_bm128-gated).
    bm_stage1 = bm
    s1_min = _STAGE1_BM128_MIN_TOK.get(sig)
    if allow_bm128 and s1_min is not None and tokens >= s1_min:
        bm_stage1 = 128
    # persist: high-expert large-M families only.
    p_min = _PERSIST_MIN_TOK.get(sig)
    persist = p_min is not None and tokens >= p_min
    # tiny-M gemm1: BN=64 + k_wave=4 for the block-count-bound high-expert fp4 families at m<=2.
    t_max = _TINYM_BN64_MAX_TOK.get(sig)
    if t_max is not None and tokens <= t_max:
        bn, k_wave = 64, 4
    else:
        bn, k_wave = 256, 1
    return bm, epilog, bm_stage1, persist, bn, k_wave


# ---- gemm1 (up/gate-proj) compile ----
def gemm1_grid(n_tokens, BM=32, NE=NE, TOPK=TOPK_DEFAULT, INTER=INTER_DEFAULT):
    """Host-side grid size (BM=32 active-experts bound)."""
    active = min(n_tokens * TOPK, NE)
    max_m_blocks = (n_tokens * TOPK + active * (BM - 1) + BM - 1) // BM
    return max_m_blocks * ((2 * INTER) // 256)


def compile_gemm1_a4w4_port(
    BM=32,
    use_nt=True,
    D_HIDDEN=H_DEFAULT,
    interleave=True,
    a_dtype="fp4",
    out_dtype="fp4",
    act="silu",
    swiglu_limit=0.0,
    SBM=None,
    k_wave=1,
    BN=BN,
    has_pad=False,
):
    # SBM (sort_block_m) is the moe_sorting padding unit, decoupled from the compute tile BM.
    # None -> SBM==BM (byte-identical); otherwise SBM must be a multiple of BM (SBM//BM compute
    # blocks per SBM sort block, all sharing one expert).
    SBM = _norm_sbm(SBM, BM)
    # use_nt IS the B-load cache policy: True -> non-temporal, False -> cached.
    b_nontemporal = use_nt
    if BM not in (16, 32, 64, 128):
        raise AssertionError(f"mxfp4_moe_gemm1 supports only BM in {{16,32,64,128}}, got BM={BM}")
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

    # k_wave (intra-block K-slice): split the K contraction across k_wave cooperating waves within
    # the 256-thread (4-wave) block; each wave accumulates its K-slice, then the k_wave partials are
    # reduced in LDS before the shared silu+quant epilogue. Guards: k_wave in {1,2,4} (<=4 waves so
    # num_n_waves=4//k_wave >= 1 -> <=256 threads, well under the 512 cap); K%k_wave==0; the k_wave
    # per-K-wave A regions + reduction slabs fit LDS (gfx950 160KB). (aiter's literal 4*tile_n<=tile_k
    # scratch guard is for its per-group reduction; this port reduces full [BM,BN] slabs sized below.)
    LDS_LIMIT = 160 * 1024  # gfx950
    if k_wave not in (1, 2, 4):
        raise AssertionError(f"k_wave must be in {{1,2,4}} (4-wave block), got {k_wave}")
    if k_wave > 1:
        if a_dtype != "fp4":
            raise AssertionError("k_wave>1 is fp4-only (fp8 A path not ported)")
        if not interleave:
            raise AssertionError("k_wave>1 requires interleave gate mode")
        if (K // BK) % k_wave != 0:
            raise AssertionError(f"K/BK ({K // BK}) must be divisible by k_wave ({k_wave})")
    # BN (fused gate|up N-tile) in {64, 256}. BN=64 gives N_OUT//64 = 4x more N-blocks for tiny-M
    # block-count coverage. In interleave mode each N-wave must hold >=2 j-tiles (a gate+up pair):
    # NJ = (BN//num_n_waves)//16 must be even, i.e. BN//num_n_waves >= 32. num_n_waves = 4//k_wave, so
    # BN=64 requires num_n_waves <= 2 -> k_wave in {2,4} (nnw=1 for the k_wave=4 tiny-M fix). BN=64 +
    # k_wave=1 (nnw=4 -> 16 cols/wave = 1 j-tile, no gate/up pair) is not expressible in this scheme.
    if BN not in (64, 256):
        raise AssertionError(f"BN must be in {{64, 256}}, got {BN}")
    if BN != 256:
        if not interleave:
            raise AssertionError("BN != 256 requires interleave gate mode")
        if a_dtype != "fp4":
            raise AssertionError("BN != 256 is fp4-only (fp8 A path not ported)")
        num_n_waves = 4 // k_wave
        nj = (BN // num_n_waves) // 16
        if nj < 2 or nj % 2 != 0:
            raise AssertionError(
                f"BN={BN} with k_wave={k_wave} (num_n_waves={num_n_waves}) yields NJ={nj}; each N-wave "
                f"needs an even NJ>=2 (gate+up pair). BN=64 needs k_wave in {{2,4}}."
            )

    KH_TILE_A = BK // (1 if a_dtype == "fp8" else 2)
    lds_bytes = lds_bytes_for(K // BK, KH_TILE_A, BM=BM, k_wave=k_wave, BN=BN)  # K_TILES_TOTAL (inter-independent)
    if lds_bytes > LDS_LIMIT:
        raise AssertionError(f"k_wave LDS {lds_bytes} > {LDS_LIMIT} (BM={BM}, k_wave={k_wave})")

    gu_tag = "il" if interleave else "sep"
    bnt_tag = "nt" if b_nontemporal else "cached"
    a_tag = "a8" if a_dtype == "fp8" else "a4"
    o_tag = "o8" if out_dtype == "fp8" else "o4"
    # act tag empty for the default silu variant so its kernel name/IR stays byte-identical (AC-3);
    # swiglu is a distinct compile-time variant (limit folded into the name).
    act_tag = "" if act == "silu" else f"_swiglu{swiglu_limit:g}"
    # sbm tag empty when SBM==BM so the default variant keeps its byte-identical kernel name.
    sbm_tag = "" if SBM == BM else f"_sbm{SBM}"
    # kw tag empty at k_wave=1 so the default variant keeps its byte-identical kernel name (AC-3).
    kw_tag = "" if k_wave == 1 else f"_kw{k_wave}"
    # bn tag empty at BN=256 so the default variant keeps its byte-identical kernel name (AC-3).
    bn_tag = "" if BN == 256 else f"_bn{BN}"
    # pad tag empty when has_pad=False so the default variant keeps its byte-identical kernel name
    # AND has no i32_kpad kernarg (AC-3). has_pad=True is a distinct compile variant that adds the
    # runtime pad kernarg + weight-OOB pad-skip.
    pad_tag = "_pad" if has_pad else ""
    name_suffix = f"h{K}_bm{BM}_{bnt_tag}_{gu_tag}_{a_tag}{o_tag}{act_tag}{sbm_tag}{kw_tag}{bn_tag}{pad_tag}_v2"

    @fx.struct
    class SharedStorage:
        buf: fx.Array[Int8, lds_bytes, 16]

    @flyc.jit
    def _gemm1_kernel_body(
        arg_aq,
        arg_ascale,
        arg_bq,
        arg_bscale,
        arg_eids,
        arg_cumsum,
        arg_sti,
        arg_aqout,
        arg_ascaleout,
        bx_i32,
        lane,
        wave,
        i32_ntok,
        i32_inter,
        i32_kpad,
        i32_npad,
    ):
        # Shared kernel body for both has_pad variants (@flyc.jit so the @flyc.kernel AST rewriter
        # recurses into its scf `if` dispatch, like gemm1_body_v2). i32_kpad (K/contraction pad) and
        # i32_npad (N/inter-output pad) are fx.Int32(0) (compile-time constants, no kernarg) in the
        # default variant -> has_pad=False folds the pad math away (byte-identical; AC-3). Only the
        # has_pad variant threads the runtime kpad/npad kernargs.
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))
        cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
        total_m_blocks = cumsum0 // fx.Int32(BM)
        num_n_blocks = (fx.Int32(i32_inter) * fx.Int32(2)) // fx.Int32(BN)  # NUM_N_BLOCKS = N_OUT//BN
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
                i32_kpad,
                i32_npad,
                BM=BM,
                K=K,
                interleave=interleave,
                b_nontemporal=b_nontemporal,
                a_dtype=a_dtype,
                out_dtype=out_dtype,
                act=act,
                swiglu_limit=swiglu_limit,
                has_pad=has_pad,
                SBM=SBM,
                k_wave=k_wave,
                BN=BN,
            )

    if not has_pad:

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
            _gemm1_kernel_body(
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_cumsum,
                arg_sti,
                arg_aqout,
                arg_ascaleout,
                bx_i32,
                lane,
                wave,
                i32_ntok,
                i32_inter,
                fx.Int32(0),
                fx.Int32(0),
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

    else:

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
            i32_kpad: fx.Int32,
            i32_npad: fx.Int32,
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
            _gemm1_kernel_body(
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_cumsum,
                arg_sti,
                arg_aqout,
                arg_ascaleout,
                bx_i32,
                lane,
                wave,
                i32_ntok,
                i32_inter,
                i32_kpad,
                i32_npad,
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
            i32_kpad: fx.Int32,
            i32_npad: fx.Int32,
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
                i32_kpad,
                i32_npad,
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
    a_dtype="fp4",
    topk=1,
    SBM=None,
    persist=False,
    cu_num=0,
    has_pad=False,
):
    """Compile the gemm2 a4w4 down-proj. epilog='atomic' (default) does per-token weighted
    atomic-fadd; epilog='reduce' does a non-atomic store into out[token_id*topk + slot] (unique
    per (token,topk) slot; host reduces over topk), mirroring main's accumulate=False path.
    BM in {32,64} (per-launch parameter). inter_dim is a runtime arg (a multiple of BK=256,
    <= INTER_MAX); INTER_MAX caps the compile-time B-view / LDS bounds. topk enters the reduce
    output-row index (compile-time).

    SBM (sort_block_m) is the moe_sorting padding unit, decoupled from the compute tile BM.
    None -> SBM==BM (byte-identical); otherwise SBM must be a multiple of BM."""
    SBM = _norm_sbm(SBM, BM)
    if BM not in (16, 32, 64, 128) or epilog not in ("atomic", "reduce"):
        raise AssertionError(
            f"mxfp4_moe_gemm2 supports only (BM in {{16,32,64,128}}, epilog in {{'atomic','reduce'}}); "
            f"got (BM={BM}, epilog={epilog})"
        )
    if SBM % BM != 0:
        raise AssertionError(f"SBM ({SBM}) must be a multiple of BM ({BM})")
    use_reduce = epilog == "reduce"
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
    if persist and is_f8:
        # KNOWN-BROKEN F2 combo: the fp8-A gemm2 persist multi-iteration grid-stride corrupts the
        # accumulator/LDS state and yields cos=0 at large M (reproduces on rlcr/moe-persist-sbm
        # alone). fp4 persist is correct. Fail fast rather than silently shipping garbage.
        raise AssertionError(
            "a8w4/fp8-A gemm2 persist is not supported (known-broken F2 path: cos=0 at large M). "
            "Use persist only with a_dtype='fp4', or run a8w4 with persist=False."
        )
    persist_tag = "" if not persist else f"_persist_cu{cu_num}"
    # pad tag empty when has_pad=False so the default keeps its byte-identical kernel name AND no
    # i32_kpad kernarg (AC-3). has_pad=True adds the runtime pad kernarg + weight-OOB pad-skip.
    pad_tag = "_pad" if has_pad else ""
    tag = f"h{N_OUT}_imax{INTER_MAX}_bm{BM}{'_nt' if use_nt else ''}_{etag}{atag}{sbm_tag}{persist_tag}{pad_tag}_v2"
    name = f"gemm2_a4w4_port_{tag}"

    @fx.struct
    class SharedStorage:
        buf: fx.Array[Int8, lds_bytes, 16]

    @flyc.jit
    def _gemm2_kernel_body(
        arg_aq,
        arg_ascale,
        arg_bq,
        arg_bscale,
        arg_eids,
        arg_cumsum,
        arg_stids,
        arg_sweights,
        arg_out,
        bx_i32,
        lane,
        wave,
        i32_M,
        i32_max_m_blocks,
        i32_inter,
        i32_kpad,
    ):
        # Shared kernel body for both has_pad variants (@flyc.jit so the @flyc.kernel AST rewriter
        # recurses into its scf `if` / grid-stride dispatch, like gemm2_body_v2). i32_kpad is
        # fx.Int32(0) (compile-time constant, no kernarg) in the default variant -> has_pad=False
        # folds the pad math away (byte-identical; AC-3). Only has_pad threads a runtime kpad kernarg.
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
                i32_kpad,
                BM=BM,
                use_nt=use_nt,
                N_OUT=N_OUT,
                INTER_MAX=INTER_MAX,
                aStages=aStages,
                a_dtype=a_dtype,
                use_reduce=use_reduce,
                topk=topk,
                has_pad=has_pad,
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

    if not has_pad:

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
            _gemm2_kernel_body(
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_cumsum,
                arg_stids,
                arg_sweights,
                arg_out,
                bx_i32,
                lane,
                wave,
                i32_M,
                i32_max_m_blocks,
                i32_inter,
                fx.Int32(0),
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

    else:

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
            i32_kpad: fx.Int32,
            arg_out: fx.Int64,
            arg_out_scale: fx.Int64,  # unused (atomic epilog); kept for signature parity
        ):
            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            tx_i32 = fx.Int32(tx)
            bx_i32 = fx.Int32(bx)
            lane = tx_i32 % fx.Int32(64)
            wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
            _gemm2_kernel_body(
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_cumsum,
                arg_stids,
                arg_sweights,
                arg_out,
                bx_i32,
                lane,
                wave,
                i32_M,
                i32_max_m_blocks,
                i32_inter,
                i32_kpad,
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
            i32_grid_blocks: fx.Int32,
            i32_inter: fx.Int32,
            i32_kpad: fx.Int32,
            arg_out: fx.Int64,
            arg_out_scale: fx.Int64,
            stream: fx.Stream,
        ):
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
                i32_kpad,
                arg_out,
                arg_out_scale,
            ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2


# ---- launcher cache + dispatch (compile once per config, fast-dispatch after) ----
G1_CACHE = {}
G2_CACHE = {}


def get_g1(
    BM,
    use_nt,
    D_HIDDEN,
    interleave,
    a_dtype,
    out_dtype,
    act="silu",
    swiglu_limit=0.0,
    SBM=None,
    k_wave=1,
    BN=BN,
    has_pad=False,
):
    # inter_dim (gemm1 N-output) is a runtime arg; NE/topk are host-only (NE: gemm1_grid active-expert
    # cap; topk: grid sizing). None of the three enters the compiled kernel, so none is a cache-key dim.
    # act/swiglu_limit are compile-time (folded into the epilog), so both are cache-key dims.
    # SBM (sort_block_m) is a compile-time cache-key dim; None means SBM==BM (byte-identical variant).
    # k_wave (intra-block K-slice) is a compile-time cache-key dim; k_wave==1 is the byte-identical
    # default variant. BN (fused gate|up N-tile) is a compile-time cache-key dim; BN==256 is the
    # byte-identical default variant.
    SBM = _norm_sbm(SBM, BM)
    # has_pad is a compile-time cache-key dim; has_pad=False is the byte-identical default variant
    # (no i32_kpad kernarg). has_pad=True is a distinct variant with the runtime pad kernarg.
    key = (BM, use_nt, D_HIDDEN, interleave, a_dtype, out_dtype, act, swiglu_limit, SBM, k_wave, BN, has_pad)
    launch = G1_CACHE.get(key)
    if launch is None:
        launch = compile_gemm1_a4w4_port(
            BM=BM,
            use_nt=use_nt,
            D_HIDDEN=D_HIDDEN,
            interleave=interleave,
            a_dtype=a_dtype,
            out_dtype=out_dtype,
            act=act,
            swiglu_limit=swiglu_limit,
            SBM=SBM,
            k_wave=k_wave,
            BN=BN,
            has_pad=has_pad,
        )
        G1_CACHE[key] = launch
    return launch


def get_g2(BM, use_nt, D_HIDDEN, epilog, INTER_MAX, a_dtype, topk=1, SBM=None, persist=False, cu_num=0, has_pad=False):
    # NE / inter_dim do not enter the compiled gemm2 kernel (inter_dim is a runtime arg); the only
    # contraction-shape key is the compile-time cap INTER_MAX. epilog + topk are compile-time
    # (reduce folds topk into the output-row index); atomic ignores topk.
    # SBM (sort_block_m) is a compile-time cache-key dim; None means SBM==BM (byte-identical variant).
    # persist (+ cu_num, the fixed-grid size) are compile-time cache-key dims; persist=False is the
    # byte-identical one-shot-grid variant.
    SBM = _norm_sbm(SBM, BM)
    topk_key = topk if epilog == "reduce" else 1
    cu_key = cu_num if persist else 0
    # has_pad is a compile-time cache-key dim; has_pad=False is the byte-identical default variant.
    key = (BM, use_nt, D_HIDDEN, epilog, INTER_MAX, a_dtype, topk_key, SBM, persist, cu_key, has_pad)
    launch = G2_CACHE.get(key)
    if launch is None:
        launch = compile_gemm2_a4w4_port(
            BM=BM,
            use_nt=use_nt,
            N_OUT=D_HIDDEN,
            epilog=epilog,
            INTER_MAX=INTER_MAX,
            a_dtype=a_dtype,
            topk=topk_key,
            SBM=SBM,
            persist=persist,
            cu_num=cu_key,
            has_pad=has_pad,
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
    interleave=True,
    a_dtype="fp4",
    out_dtype="fp4",
    act="silu",
    swiglu_limit=0.0,
    SBM=None,
    k_wave=1,
    BN=BN,
    n_sorted_padded=None,
    model_dim_pad=0,
    inter_dim_pad=0,
    stream=None,
):
    """Stage-1 up/gate gemm: A_q x w1 -> inter (packed MXFP4/MXFP8, sorted); buffers pre-allocated by caller.

    ``model_dim_pad`` (>0): D_HIDDEN is the PADDED model_dim (contraction K); the trailing
    model_dim_pad columns are host zero-pad. Enables the has_pad weight-OOB pad-skip variant, whose
    kernel sizes the per-16N-tile B-weight buffer resource to the REAL K (= D_HIDDEN - model_dim_pad)
    so the fully-pad 128-K weight halves buffer-load OOB -> 0 (no HBM fetch). 0 -> byte-identical
    default (no pad kernarg).

    ``inter_dim_pad`` (>0): D_INTER is the PADDED per-half inter (gemm1 N-output); the trailing
    inter_dim_pad cols of EACH of the fused gate|up halves are pad. Enables the N-output pad-skip: a
    16-N weight tile whose logical-inter base is >= (D_INTER - inter_dim_pad) has its buffer sized to
    0 records so its weight loads OOB -> 0 (no HBM fetch, ~2*inter_dim_pad/N_OUT of the w1 N bytes).
    Correctness-safe: pad-N output feeds gemm2's pad-K input, already OOB-skipped. Either pad>0
    enables the shared has_pad variant; both fold to the byte-identical default at 0 (AC-3).

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

    has_pad = model_dim_pad > 0 or inter_dim_pad > 0
    launch = get_g1(
        BM,
        use_nt,
        D_HIDDEN,
        interleave,
        a_dtype,
        out_dtype,
        act,
        swiglu_limit,
        SBM=SBM,
        k_wave=k_wave,
        BN=BN,
        has_pad=has_pad,
    )
    sbm = _norm_sbm(SBM, BM)
    num_n_blocks = (2 * D_INTER) // BN
    if n_sorted_padded is None:
        # E-based worst-case grid: sort padding is per SBM (the sort unit); the compute grid is
        # in BM blocks (padded_rows // BM). SBM==BM reduces to the original gemm1_grid.
        active = min(n_tokens * topk, NE)
        padded_rows = ((n_tokens * topk + active * (sbm - 1) + sbm - 1) // sbm) * sbm
        grid = (padded_rows // BM) * num_n_blocks
    else:
        grid = (n_sorted_padded // BM) * num_n_blocks
    # has_pad variant threads the runtime i32_kpad (K = D_HIDDEN, kpad = model_dim_pad) and i32_npad
    # (N-output per-half inter pad = inter_dim_pad) kernargs right after i32_inter, matching the launch
    # signature; default has no such kernargs (AC-3). Either pad may be 0 while the other is set.
    pad_args = (int(model_dim_pad), int(inter_dim_pad)) if has_pad else ()
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
        *pad_args,
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
    epilog="atomic",
    SBM=None,
    persist=False,
    cu_num=0,
    n_sorted_padded=None,
    inter_dim_pad=0,
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

    if persist and cu_num <= 0:
        cu_num = _get_cu_num()
    has_pad = inter_dim_pad > 0
    launch = get_g2(
        BM,
        use_nt,
        D_HIDDEN,
        epilog,
        INTER_MAX_DEFAULT,
        a_dtype,
        topk=topk,
        SBM=SBM,
        persist=persist,
        cu_num=cu_num,
        has_pad=has_pad,
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
    # has_pad variant threads the runtime i32_kpad kernarg (K = inter_dim, kpad = inter_dim_pad)
    # right after i32_inter, matching the launch signature; default has no such kernarg (AC-3).
    pad_args = (int(inter_dim_pad),) if has_pad else ()
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
        *pad_args,
        out.data_ptr(),
        out_scale.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return out
