# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Compile + launch dispatch for the layout-API MXFP4 MoE gemm (BM32, opus-sort).

Public entry point for the a4w4 / a8w4 surface. Consumes the opus sort contract
from ``moe_sorting_kernel`` (``sorted_token_ids`` = (topk<<24)|token_id, sentinel
(topk<<24)|M); no fused-sort extras. gemm2's atomic epilogue scatters into the
pre-zeroed output, so there is no reverse-permutation dependency.

The ``compile_*`` builders wrap the ``moegemm`` device bodies (@flyc.jit) in the
@flyc.kernel entry + @flyc.jit launch; basics come from ``utils``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .moegemm import (
    _gemm1_body_v2,
    _gemm2_body_v2,
    _issue_a_load_lds_dt,
    _lds_bytes_for,
)
from .utils import (
    BK,
    BN,
    INTER_DEFAULT,
    K_DEFAULT,
    MAX_M,
    N_OUT,
    NE,
    NE_DEFAULT,
    TOPK_DEFAULT,
    K,
    _global_ptr1,
    _raw,
    _udiv,
    k_tiles_total_for,
    kStages,
    num_n_blocks_for,
)

__all__ = [
    "compile_gemm1_a4w4_port",
    "compile_gemm2_a4w4_port",
    "gemm1_grid",
    "mxfp4_moe_gemm1",
    "mxfp4_moe_gemm2",
]


# ===========================================================================
# gemm1 (up/gate-proj) compile
# ===========================================================================
def gemm1_grid(n_tokens, BM=32, NE=NE_DEFAULT, TOPK=TOPK_DEFAULT, INTER=INTER_DEFAULT):
    """Host-side grid size (BM=32 active-experts bound)."""
    active = min(n_tokens * TOPK, NE)
    max_m_blocks = (n_tokens * TOPK + active * (BM - 1) + BM - 1) // BM
    return max_m_blocks * ((2 * INTER) // 256)


def compile_gemm1_a4w4_port(
    BM=32,
    use_nt=True,
    inline_quant=False,
    D_HIDDEN=K_DEFAULT,
    D_INTER=INTER_DEFAULT,
    NE=NE_DEFAULT,
    TOPK=TOPK_DEFAULT,
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

    _K, _INTER, _NE = D_HIDDEN, D_INTER, NE
    assert _K % BK == 0, f"D_HIDDEN (K) must be a multiple of {BK}, got {_K}"
    _N_OUT = 2 * _INTER
    assert _N_OUT % BN == 0, f"2*D_INTER (N_OUT) must be a multiple of {BN}, got {_N_OUT}"

    _KH_TILE_A = BK // (1 if a_dtype == "fp8" else 2)
    lds_bytes = _lds_bytes_for(_K // BK, _KH_TILE_A)  # K_TILES_TOTAL

    gu_tag = "il" if interleave else "sep"
    bnt_tag = "nt" if b_nontemporal else "cached"
    a_tag = "a8" if a_dtype == "fp8" else "a4"
    o_tag = "o8" if out_dtype == "fp8" else "o4"
    name_suffix = f"h{_K}_i{_INTER}_ne{_NE}_bm{BM}_{bnt_tag}_{gu_tag}_{a_tag}{o_tag}_v2"

    allocator = SmemAllocator(None, arch="gfx950", global_sym_name=f"gemm1port_v2_smem_{name_suffix}")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + lds_bytes

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
        cumsum0 = llvm.load(T.i32, _global_ptr1(arg_cumsum, fx.Int32(0)))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(_N_OUT // 256)  # * NUM_N_BLOCKS
        if fx.Int32(bx_i32) < bound:
            _gemm1_body_v2(
                allocator,
                lds_off,
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
                K=_K,
                INTER=_INTER,
                NE=_NE,
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
        from flydsl.compiler.kernel_function import CompilationContext

        ctx = CompilationContext.get_current()
        allocator.finalized = False
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
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


# ===========================================================================
# gemm2 (down-proj) compile
# ===========================================================================
def compile_gemm2_a4w4_port(
    BM=32,
    use_nt=False,
    NE=NE,
    N_OUT=N_OUT,
    MAX_M=MAX_M,
    epilog="atomic",
    D_INTER=K,
    D_INTER_REAL=None,
    a_dtype="fp4",
):
    """Compile the gemm2 a4w4 layout-API down-proj. Only (BM=32, epilog="atomic") is
    supported; D_INTER (= contraction K = inter_dim) must be a multiple of BK(256)
    (512 keeps the fully-unrolled fast path; >512 uses the streaming K-loop).
    a_dtype="fp8" reads an mxfp8 intermediate (gemm1 out_dtype="fp8"). The
    D_INTER_REAL pad-tail (unpadded non-256-aligned shards) is not supported."""
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
    _K = D_INTER
    assert _K % BK == 0, f"D_INTER (gemm2 contraction K = inter_dim) must be a multiple of {BK}, got {_K}"
    _is_f8 = a_dtype == "fp8"
    _KH_TILE_A = BK // (1 if _is_f8 else 2)  # A LDS K-tile bytes (fp8 256, fp4 128)
    _K_BYTES = _K // (1 if _is_f8 else 2)  # A row stride bytes (fp8 K, fp4 K//2)
    _slot_bytes = BM * _KH_TILE_A
    _K_TILES_TOTAL = k_tiles_total_for(_K)
    _aStages = kStages if _K_TILES_TOTAL <= kStages else 3
    _lds_bytes = max(BM * BN * 4, _aStages * _slot_bytes)
    _num_n_blocks = num_n_blocks_for(N_OUT)

    _atag = "_a8" if _is_f8 else ""
    _tag = f"ne{NE}_h{N_OUT}_i{_K}_bm{BM}{'_nt' if use_nt else ''}_atomic{_atag}_v2"
    _name = f"gemm2_a4w4_port_{_tag}"

    allocator = SmemAllocator(None, arch="gfx950", global_sym_name=f"gemm2port_v2_smem_{_tag}")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + _lds_bytes

    @flyc.kernel(name=_name, known_block_size=[256, 1, 1])
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

        _aq_num = arith.index_cast(T.index, _raw(i32_max_m_blocks)) * fx.Index(BM * _K_BYTES)
        aq_rsrc = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_aq)), num_records_bytes=_aq_num)
        saq = SmemPtr(allocator.get_base(), lds_off, T.i8, shape=(_aStages * _slot_bytes,))

        # Preload the first kStages K-tiles (all tiles for the K_TILES<=2 fast path;
        # the prologue for the streaming path). slot == kt for the preload.
        def _issue_all_a_loads(m_row0):
            for slot in range_constexpr(kStages):
                _issue_a_load_lds_dt(
                    aq_rsrc,
                    saq,
                    slot,
                    slot,
                    m_row0,
                    wave,
                    lane,
                    _is_f8,
                    _KH_TILE_A,
                    _K_BYTES,
                )

        # One-shot grid (atomic). Issue A->LDS BEFORE the cumsum load so the HBM
        # latency overlaps the cumsum + bound check (A->LDS depends only on bx/lane).
        _issue_all_a_loads(_udiv(bx_i32, _num_n_blocks) * fx.Int32(BM))
        rocdl.sched_barrier(0)

        cumsum0 = llvm.load(T.i32, _global_ptr1(arg_cumsum, fx.Int32(0)))
        total_m_blocks = _udiv(cumsum0, BM)
        bound = total_m_blocks * fx.Int32(_num_n_blocks)

        if fx.Int32(bx_i32) < bound:
            _gemm2_body_v2(
                allocator,
                lds_off,
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
                use_nt=use_nt,
                NE=NE,
                N_OUT=N_OUT,
                D_INTER=_K,
                aStages=_aStages,
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
        from flydsl.compiler.kernel_function import CompilationContext

        ctx = CompilationContext.get_current()
        allocator.finalized = False
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        grid_x = arith.index_cast(T.index, i32_max_m_blocks) * fx.Index(_num_n_blocks)
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


# ===========================================================================
# launcher cache + dispatch (compile once per config, fast-dispatch after)
# ===========================================================================
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


def _get_g1(BM, use_nt, inline_quant, D_HIDDEN, D_INTER, NE, topk, interleave, a_dtype, out_dtype):
    key = (BM, use_nt, inline_quant, D_HIDDEN, D_INTER, NE, topk, interleave, a_dtype, out_dtype)
    launch = _G1_CACHE.get(key)
    if launch is None:
        launch = compile_gemm1_a4w4_port(
            BM=BM,
            use_nt=use_nt,
            inline_quant=inline_quant,
            D_HIDDEN=D_HIDDEN,
            D_INTER=D_INTER,
            NE=NE,
            TOPK=topk,
            interleave=interleave,
            a_dtype=a_dtype,
            out_dtype=out_dtype,
        )
        _G1_CACHE[key] = launch
    return launch


def _get_g2(BM, use_nt, NE, D_HIDDEN, epilog, D_INTER, D_INTER_REAL, a_dtype):
    key = (BM, use_nt, NE, D_HIDDEN, epilog, D_INTER, D_INTER_REAL, a_dtype)
    launch = _G2_CACHE.get(key)
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

    launch = _get_g1(BM, use_nt, inline_quant, D_HIDDEN, D_INTER, NE, topk, interleave, a_dtype, out_dtype)
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

    launch = _get_g2(BM, use_nt, NE, D_HIDDEN, "atomic", D_INTER, D_INTER_REAL, a_dtype)
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
