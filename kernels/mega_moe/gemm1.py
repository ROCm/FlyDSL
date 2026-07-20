# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

# FUSED-OPERATOR stage1 dispatch⊕GEMM builder for the MoE megakernel (compile_fused_moe_gemm1).

"""MoE stage-1 fused dispatch⊕GEMM kernel builder (FlyDSL MFMA, CDNA4 / MI355X).

`compile_fused_moe_gemm1` builds the persistent sparse-tile gate/up+silu group-GEMM with the
fixedslot in-kernel dispatch prologue. See docs/moe_stage1_mega.md.
"""

import functools
import os

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, gpu, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator
from kernels.common.kernels_common import _guard, validate_moe_dtypes
from kernels.common.layout_utils import get as layout_get
from kernels.common.layout_utils import idx2crd
from kernels.common.mma.mfma_preshuffle_pipeline import (
    tile_chunk_coord_i32,
)

# GateMode is re-exported from moe_common (must be the SAME enum object for `is` comparisons).
from kernels.moe.moe_common import GateMode  # single source of truth (re-exported below)

from .dispatch import emit_dispatch_prologue  # noqa: E402
from .epilogue import (
    Activation,
    Gemm1Epilogue,
    OutputQuant,
    Scatter,
    reduce_slice_k_partials,
    stage_srcmap_to_lds,
)
from .kloop import (
    BLoader,
    G2SLoaderX,
    KLoop,
    KScaleLoader,
    MfmaScaleGU,
    S2RLoaderA,
)
from .utils import (  # noqa: E402
    AGatherAddresser,
    ExpertOperandBases,
    TileScheduler,
    barrier,
    build_lds_views,
    build_pipe_schedule,
    epk,
    finalize_lds,
    make_layouts,
    make_rearrange_scales,
    persistent_tile_loop,
    plan_lds,
    resolve_wave_tiling,
    tensor_base_addr_index,
)


@functools.lru_cache(maxsize=None)
def compile_fused_moe_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "f16",
    use_async_copy: bool = False,
    waves_per_eu: int = 4,
    slice_k: int = 1,
    b_nt: int = 0,
    gate_mode: GateMode = GateMode.SEPARATED,
    xcd_swizzle: int = 0,
    rank: int = 0,
    experts_per_rank: int = 0,
    fuse_dispatch: str = "fixedslot",
    fuse_npes: int = 0,
    fuse_topk: int = 0,
    fuse_cap: int = 0,
    fuse_mtpr: int = 0,
    fuse_scale_dim: int = 0,
    fuse_scale_type_size: int = 1,
    compact_dispatch: bool = False,
):
    """Compile stage1 gate+up+silu kernel; the a2 OUTPUT is written to the ATOM logical row
    via the dispatch srcmap (atom_contract). GEMM: silu(X @ W_gate.T) * (X @ W_up.T) ->
    [tokens*topk, inter_dim].
    """
    gpu_arch = get_hip_arch()
    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")

    validate_moe_dtypes(a_dtype, b_dtype)
    # fused gemm1 supports fp8/fp4 activation x fp4 weight -> quantized fp8/fp4 output only.
    if a_dtype not in ("fp8", "fp4"):
        raise ValueError(f"fused gemm1 requires a_dtype in ('fp8', 'fp4'), got {a_dtype!r}")
    if b_dtype != "fp4":
        raise ValueError(f"fused gemm1 requires b_dtype='fp4', got {b_dtype!r}")
    if out_dtype not in ("fp8", "fp4"):
        raise ValueError(f"fused gemm1 requires a quantized out_dtype in ('fp8', 'fp4'), got {out_dtype!r}")

    is_f8_a = a_dtype == "fp8"
    is_f4_a = a_dtype == "fp4"

    sort_block_m = max(32, tile_m)
    # slice_k = intra-CTA K-slice count: total_waves = num_waves*slice_k groups each partial-sum a K-slice, reduced in LDS before the epilogue; slice_k==1 is identity.
    num_waves, total_waves, total_threads, pack_M, n_per_wave, pack_N = resolve_wave_tiling(
        tile_m=tile_m, tile_n=tile_n, slice_k=slice_k
    )
    if slice_k > 1:
        if (model_dim % slice_k) != 0:
            raise ValueError(f"slice_k={slice_k} must divide model_dim={model_dim}")
        if ((model_dim // slice_k) % tile_k) != 0:
            raise ValueError(f"model_dim//slice_k ({model_dim // slice_k}) must be divisible by tile_k={tile_k}")
    group_threads = total_threads // slice_k
    k_slice = model_dim // slice_k  # per-group K extent (== model_dim when slice_k==1)
    pack_K = 2
    scale_mn_pack = 2
    elem_bytes = 1
    a_elem_bytes = 1
    b_elem_bytes = 1
    tile_k_bytes = int(tile_k) * int(a_elem_bytes)
    a_elem_vec_pack = 2 if is_f4_a else 1
    cbsz = 0 if is_f8_a else 4
    blgp = 4

    if (tile_k_bytes % 64) != 0:
        raise ValueError(f"tile_k_bytes must be divisible by 64, got {tile_k_bytes}")

    out_s = str(out_dtype).strip().lower()

    def w_elem_type():
        return T.i8

    gate_up_interleave = gate_mode is GateMode.INTERLEAVE

    # Per-group K extent: each K-slice wave group runs the mainloop over its own [wid_k*_k_slice, (wid_k+1)*_k_slice).
    k_dim = k_slice

    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(a_elem_bytes)
    # X gather per K-slice group: group_threads threads load the FULL tile_m*tile_k of their K-slice (== total_threads at slice_k==1).
    if bytes_x_per_tile % group_threads != 0:
        raise ValueError(f"tile_m*tile_k*elem_bytes must be divisible by {group_threads}")
    bytes_per_thread_x = bytes_x_per_tile // group_threads

    use_lds128 = os.environ.get("FLIR_CK_LDS128", "1") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    pad_k = 0 if use_lds128 else 8
    lds_stride = tile_k + pad_k

    need_fp4 = out_dtype == "fp4"
    need_fp8 = out_dtype == "fp8"
    need_sort = True
    # CK-style hot-loop interleave (mode 3, rate 3 MFMAs/interleaved load): hides B/A load latency.
    isched = 3
    ck_rate = 3
    cu_num = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if fuse_dispatch != "fixedslot":
        raise ValueError(f"fuse_dispatch={fuse_dispatch!r} unsupported; only 'fixedslot' is supported.")
    compact = bool(compact_dispatch)
    ca = bool(compact_dispatch)  # compact+atom: compact DISPATCH but atom-logical a2 OUTPUT
    static_tiles = bool(not compact_dispatch)  # P-static: derive (expert,k) from fixed-slot layout
    # Fused fixedslot dispatch-prologue host consts.
    fz_npes, fz_epr, fz_k = int(fuse_npes), int(experts_per_rank), int(fuse_topk)
    fz_cap, fz_mtpr, fz_rank = int(fuse_cap), int(fuse_mtpr), int(rank)
    fz_tile_m = int(sort_block_m)  # dispatch tile granularity == GEMM M-tile
    assert fz_cap % fz_tile_m == 0, f"fuse_cap({fz_cap}) %% tile_m({fz_tile_m}) != 0"
    fz_total_experts = fz_npes * fz_epr
    fz_n_i32, fz_nbytes = (model_dim // 8, model_dim // 2) if is_f4_a else (model_dim // 4, model_dim)
    fz_scale_bytes = int(fuse_scale_dim) * int(fuse_scale_type_size)
    fz_scale_n_i32 = (fz_scale_bytes + 3) // 4 if fz_scale_bytes > 0 else 0
    fz_enable_scales = fz_scale_bytes > 0
    fz_safe_end_i32 = (fz_n_i32 // 512) * 512
    # co-resident grid: gx (N-tiles) * gy <= cu_num keeps the dispatch-prologue grid barrier deadlock-free.
    fz_inter_in = 2 * inter_dim
    fz_gx_static = (
        (fz_inter_in + tile_n - 1) // tile_n if gate_up_interleave else (fz_inter_in + 2 * tile_n - 1) // tile_n // 2
    )
    fz_gy = max(1, cu_num // max(1, fz_gx_static))

    def get_module_name():
        fp4q, fp8q = ("_fp4q" if need_fp4 else ""), ("_fp8q" if need_fp8 else "")
        sort, asy = ("_sort" if need_sort else ""), ("_async" if use_async_copy else "")
        gui = "_gui" if gate_up_interleave else ""
        xcd = f"_xcd{xcd_swizzle}" if xcd_swizzle > 0 else ""
        isd = f"_is{isched}_r{ck_rate}"
        fuse = f"_fz{fuse_dispatch}{fuse_npes}x{fuse_cap}x{fuse_topk}"
        fuse += "_cptag" if compact else ""
        sk = f"_sk{slice_k}" if slice_k > 1 else ""  # empty at slice_k==1 -> unchanged symbol
        bnt = f"_bnt{b_nt}" if b_nt != 0 else ""  # B-load cache modifier (empty at 0 -> unchanged symbol)
        return (
            f"mfma_fmoe1_silu_mul_a{a_dtype}_w{b_dtype}_{out_s}_t{tile_m}x{tile_n}x{tile_k}"
            f"_persist_cu{cu_num}{fp4q}{fp8q}{sort}{asy}{gui}{xcd}_ci_ras_spt{isd}{fuse}{sk}{bnt}_w1l{experts_per_rank}_v37"
        ).replace("-", "_")

    module_name = get_module_name()

    # LDS arena plan (X ping/pong + cshuffle-out + tid + raw_a_scale + compact count) -> plan_lds.
    lds = plan_lds(
        allocator_pong,
        allocator_ping,
        gpu_arch=gpu_arch,
        tile_m=tile_m,
        tile_n=tile_n,
        num_waves=num_waves,
        pack_M=pack_M,
        model_dim=model_dim,
        total_threads=total_threads,
        waves_per_eu=waves_per_eu,
        lds_stride=lds_stride,
        a_elem_bytes=a_elem_bytes,
        a_elem_vec_pack=a_elem_vec_pack,
        use_async_copy=use_async_copy,
        compact_ag=compact,
        fz_total_experts=fz_total_experts,
        slice_k=slice_k,
    )
    # offsets/shape consumed by build_lds_views(lds=_lds); only these fields are read elsewhere:
    sc_cp_vec = lds.sc_cp_vec
    sc_cp_iters = lds.sc_cp_iters
    cnt_lds_off = lds.cnt_lds_off
    lds_data_bytes = lds.data_bytes
    lds_scale_bytes = lds.scale_bytes
    lds_total_bytes = lds.total_bytes

    kpack_bytes = 16

    e_vec_s1 = max(2, min(tile_n // 32, 8))
    num_threads_per_quant_blk_s1 = 32 // e_vec_s1
    shuffle_dists_s1 = []
    sh_val = 1
    while sh_val < num_threads_per_quant_blk_s1:
        shuffle_dists_s1.append(sh_val)
        sh_val *= 2
    num_shuffle_steps_s1 = len(shuffle_dists_s1)

    sched = build_pipe_schedule(
        tile_m=tile_m,
        tile_k_bytes=tile_k_bytes,
        pack_K=pack_K,
        pack_N=pack_N,
        n_per_wave=n_per_wave,
        gate_up_interleave=gate_up_interleave,
    )
    pp_mfma = sched.pp_mfma
    pp_a_reads = sched.pp_a_reads
    pp_b_loads = sched.pp_b_loads
    pp_has_scale = sched.pp_has_scale
    pipe_n_phases = sched.n_phases

    fp4_ratio = 2 if a_dtype == "fp4" else 1
    gui_ratio = 1 if gate_up_interleave else 2
    vmcnt_before_barrier = tile_m // 32 // fp4_ratio + tile_n // 32 * gui_ratio

    @flyc.kernel(known_block_size=[total_threads, 1, 1])
    def moe_gemm1(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        arg_bias: fx.Tensor,
        arg_out_scale_sorted: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        addr_payload_done: fx.Int64,  # [epr] i32 per-expert post-write counter (overlap gate)
        addr_expected_real: fx.Int64,  # [epr] i32 per-expert expected real count (overlap gate)
        addr_disp: fx.Int64,  # FUSED: ptr to FIXED-pointer table (op bufs + p2p; built once)
        i32_cur_tok: fx.Int32,  # FUSED: this-rank token count this launch
        addr_in_tok: fx.Int64,  # FUSED: per-step input pointers passed as SCALAR launch args
        addr_in_idx: fx.Int64,  # (NOT via the table) so CUDAGraph capture bakes the graph-
        addr_in_wts: fx.Int64,  # stable quant-output addresses -> no illegal in-capture H2D.
        addr_in_sc: fx.Int64,
    ):

        # FUSED dispatch PROLOGUE (phase P0): cooperative all-to-all distributed over all total_waves block warps.
        emit_dispatch_prologue(
            num_waves=total_waves,
            sort_block_m=sort_block_m,
            fz_npes=fz_npes,
            fz_epr=fz_epr,
            fz_k=fz_k,
            fz_cap=fz_cap,
            fz_mtpr=fz_mtpr,
            fz_rank=fz_rank,
            fz_tile_m=fz_tile_m,
            fz_total_experts=fz_total_experts,
            fz_nbytes=fz_nbytes,
            fz_n_i32=fz_n_i32,
            fz_safe_end_i32=fz_safe_end_i32,
            fz_scale_n_i32=fz_scale_n_i32,
            fz_enable_scales=fz_enable_scales,
            fuse_fs=True,
            compact=compact,
            static_tiles=static_tiles,
            allocator_pong=allocator_pong,
            cnt_lds_off=cnt_lds_off,
            addr_disp=addr_disp,
            i32_cur_tok=i32_cur_tok,
            addr_in_tok=addr_in_tok,
            addr_in_idx=addr_in_idx,
            addr_in_wts=addr_in_wts,
            addr_in_sc=addr_in_sc,
        )

        tokens_in = arith.index_cast(T.index, i32_tokens_in.ir_value())
        n_in = arith.index_cast(T.index, i32_n_in.ir_value())
        k_in = arith.index_cast(T.index, i32_k_in.ir_value())
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in.ir_value())

        x_elem = T.f8
        f32 = T.f32
        i64 = T.i64
        vec4_f32 = T.vec(4, f32)
        vec16_elems = 16 if a_elem_bytes == 1 else 8
        vec16_x = T.vec(vec16_elems, x_elem)
        vec2_i64 = T.vec(2, i64)

        acc_init = arith.constant_vector(0.0, vec4_f32)

        # Stage1 dims: X[tokens, model_dim] (M=tokens, K=model_dim), W gate N=inter_dim, Out[tokens*topk, inter_dim].

        # w1/scale span only this rank's experts_per_rank experts (LOCAL slice; expert_off_idx is LOCAL).
        assert experts_per_rank > 0, "fused gemm1 requires experts_per_rank>0 (local-w1 layout)"
        w_experts = experts_per_rank
        # Preshuffle B / A-scale / B-scale data layouts -> mega_moe_groupgemm_utils.make_layouts.
        layout_b, layout_a_scale, layout_b_scale, sorted_m = make_layouts(
            arith,
            w_experts=w_experts,
            inter_dim=inter_dim,
            k_in=k_in,
            pack_K=pack_K,
            kpack_bytes=kpack_bytes,
            b_elem_bytes=b_elem_bytes,
            size_expert_ids_in=size_expert_ids_in,
            sort_block_m=sort_block_m,
            model_dim=model_dim,
        )

        eff_lds_stride = lds_stride
        eff_tile_k_bytes = tile_k_bytes
        if const_expr(use_async_copy and a_elem_vec_pack > 1):
            eff_lds_stride = lds_stride // a_elem_vec_pack
            eff_tile_k_bytes = tile_k_bytes // a_elem_vec_pack

        shape_lds = fx.make_shape(tile_m, eff_lds_stride)
        stride_lds = fx.make_stride(eff_lds_stride, 1)
        layout_lds = fx.make_layout(shape_lds, stride_lds)

        tx = gpu.thread_id("x")
        by = gpu.block_id("x")  # tile along inter_dim (N)
        bx_persist = gpu.block_id("y")  # persistent WG index

        sched_mode = "static" if static_tiles else "sparse"
        sched = TileScheduler(
            mode=sched_mode,
            xcd_swizzle=xcd_swizzle,
            gate_up_interleave=gate_up_interleave,
            n_in=n_in,
            tile_n=tile_n,
            sort_block_m=sort_block_m,
            skip_gemm=False,
            rank=rank,
            experts_per_rank=experts_per_rank,
            experts=experts,
            fz_epr=fz_epr,
            fz_cap=fz_cap,
            fz_tile_m=fz_tile_m,
            fz_rank=fz_rank,
            addr_expected_real=addr_expected_real,
            tx=tx,
            epk=epk,
        )
        bx_persist, by = sched.swizzled_wg(bx_persist, by)

        by_n = by * arith.constant(tile_n, index=True)

        k_base_idx = arith.index(0)

        k_blocks16 = arith.constant(eff_tile_k_bytes // 16, index=True)
        # tx -> (wave_id, lane): wave_id spans all total_waves = num_waves*slice_k block warps.
        layout_tx_wave_lane = fx.make_layout((total_waves, 64), stride=(64, 1))
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

        # Per-block LDS memref views -> mega_moe_groupgemm_utils.build_lds_views.
        v = build_lds_views(allocator_pong, allocator_ping, lds=lds, tile_m=tile_m, tile_n=tile_n)
        lds_x_pong, lds_x_ping = v.x_pong, v.x_ping
        lds_out, lds_out_B = v.out, v.out_B
        lds_tid, lds_a_scale = v.tid, v.a_scale

        # Buffer resources
        c_a_pack = arith.constant(int(a_elem_vec_pack), index=True)
        c_elem_bytes = arith.constant(int(a_elem_bytes), index=True)

        # X: [tokens, model_dim]
        x_nbytes_idx = (tokens_in * k_in * c_elem_bytes) / c_a_pack
        x_nbytes_i32 = arith.index_cast(T.i32, x_nbytes_idx)
        x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False, num_records_bytes=x_nbytes_i32)

        # W: [experts, 2*inter_dim, model_dim]; fp4 packs 2 elements per byte.
        w_nbytes_s1 = (w_experts * (2 * inter_dim) * model_dim) // 2
        w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False, num_records_bytes=w_nbytes_s1)

        # Out: [tokens*topk, inter_dim]
        numids_rsrc = buffer_ops.create_buffer_resource(
            arg_num_valid_ids,
            max_size=False,
            num_records_bytes=arith.constant(4, type=T.i32),
        )
        num_valid_i32 = buffer_ops.buffer_load(numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=T.i32)

        # A scale: [sorted_size, model_dim/32] pre-scattered by caller
        c32 = arith.constant(32, index=True)
        kblk = k_in / c32
        sx_nbytes_idx = sorted_m * kblk
        sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
        sx_rsrc = buffer_ops.create_buffer_resource(arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_i32)

        c32 = arith.constant(32, index=True)
        kblk_w = k_in / c32
        mn_w = arith.constant(w_experts * (2 * inter_dim), index=True)
        sw_nbytes_idx = mn_w * kblk_w
        sw_nbytes_i32 = arith.index_cast(T.i32, sw_nbytes_idx)
        sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False, num_records_bytes=sw_nbytes_i32)

        sorted_nbytes_idx = size_expert_ids_in * arith.constant(sort_block_m * 4, index=True)
        sorted_nbytes_i32 = arith.index_cast(T.i32, sorted_nbytes_idx)
        sorted_rsrc = buffer_ops.create_buffer_resource(
            arg_sorted_token_ids,
            max_size=False,
            num_records_bytes=sorted_nbytes_i32,
        )
        sorted_w_rsrc = buffer_ops.create_buffer_resource(
            arg_sorted_weights, max_size=False, num_records_bytes=sorted_nbytes_i32
        )

        eid_nbytes_idx = size_expert_ids_in * arith.constant(4, index=True)
        eid_nbytes_i32 = arith.index_cast(T.i32, eid_nbytes_idx)
        expert_rsrc = buffer_ops.create_buffer_resource(
            arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes_i32
        )
        # Sorted-scale buffer resource for fused mxfp4 quantization
        sorted_scale_cols = inter_dim // 32
        sorted_scale_cols_i32 = arith.constant(sorted_scale_cols, type=T.i32)
        sorted_scale_rsrc = None
        if const_expr(need_sort):
            sort_rows_idx = size_expert_ids_in * arith.constant(sort_block_m, index=True)
            sort_padded_rows = (
                (sort_rows_idx + arith.constant(255, index=True))
                / arith.constant(256, index=True)
                * arith.constant(256, index=True)
            )
            sort_padded_cols = arith.constant(((sorted_scale_cols + 7) // 8) * 8, index=True)
            sort_scale_nbytes = arith.index_cast(T.i32, sort_padded_rows * sort_padded_cols)
            sorted_scale_rsrc = buffer_ops.create_buffer_resource(
                arg_out_scale_sorted, max_size=False, num_records_bytes=sort_scale_nbytes
            )

        # Persistent round-robin tile loop (grid_y CTAs, bounded by num_valid); bounds+decode from TileScheduler.
        c0_p, tiles_per_block, c1_p = sched.tiles_per_block(num_valid_i32)
        pstate = sched.setup_persistent()
        with persistent_tile_loop(c0_p, tiles_per_block, c1_p) as mi_p:
            tc = sched.decode(bx_persist, mi_p, pstate, sorted_rsrc=sorted_rsrc, expert_rsrc=expert_rsrc)
            bx = tc.bx
            bx_m = tc.bx_m
            expert_idx = tc.expert_idx
            blk_valid = tc.blk_valid
            exp_valid = tc.exp_valid
            kf = tc.kf
            cnt_ef = tc.cnt_ef

            def moe_gemm1_body():
                # Gate expert offset: first inter_dim rows of each expert's 2*inter_dim block
                expert_off_idx = expert_idx * arith.constant(2 * inter_dim, index=True)
                atom_sw_out_addr = buffer_ops.buffer_load(
                    buffer_ops.create_buffer_resource_from_addr(addr_disp), arith.constant(20), vec_width=1, dtype=T.i64
                )
                atom_sw_out_rsrc = buffer_ops.create_buffer_resource_from_addr(atom_sw_out_addr)
                wts_sorted_addr = buffer_ops.buffer_load(
                    buffer_ops.create_buffer_resource_from_addr(addr_disp),
                    arith.constant(42 if ca else 25),
                    vec_width=1,
                    dtype=T.i64,
                )
                wts_sorted_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    wts_sorted_addr, num_records_bytes=sorted_nbytes_i32
                )

                # X loading: X row = token_id only.
                x_load_bytes = 16
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4

                c_k_div4 = ((k_in / c_a_pack) * arith.constant(int(a_elem_bytes), index=True)) / arith.index(4)
                tile_k_dwords = (int(tile_k) * int(a_elem_bytes)) // (4 * int(a_elem_vec_pack))
                layout_x_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), stride=(tile_k_dwords, 1))
                c_chunk_i32 = arith.constant(chunk_i32, index=True)
                # X gather per K-slice group: index by group_tid over group_threads (== tx at slice_k==1).
                if const_expr(slice_k > 1):
                    group_tid = tx % arith.constant(group_threads, index=True)
                else:
                    group_tid = tx
                tx_i32_base = group_tid * c_chunk_i32

                topk_i32 = arith.constant(topk)
                tokens_i32 = arith.index_cast(T.i32, tokens_in)

                def x_tile_chunk_coord_i32(i: int):
                    return tile_chunk_coord_i32(
                        arith,
                        tx_i32_base=tx_i32_base,
                        i=i,
                        total_threads=group_threads,
                        layout_tile_div4=layout_x_tile_div4,
                        chunk_i32=chunk_i32,
                    )

                # Decode sorted token ids (stage1 X row = token_id); A-gather row bases via AGatherAddresser.
                a_gather = AGatherAddresser(
                    num_x_loads=num_x_loads,
                    x_tile_chunk_coord_i32=x_tile_chunk_coord_i32,
                    c_k_div4=c_k_div4,
                )
                x_row_base_div4, x_col_local_i32, x_row_local = a_gather.row_bases(bx_m)

                # Wave/lane decomposition.
                coord_wl = idx2crd(tx, layout_tx_wave_lane)
                wave_id = layout_get(coord_wl, 0)
                lane_id = layout_get(coord_wl, 1)
                coord_l16 = idx2crd(lane_id, layout_lane16)
                lane_div_16 = layout_get(coord_l16, 0)
                lane_mod_16 = layout_get(coord_l16, 1)
                row_a_lds = lane_mod_16
                col_offset_base = lane_div_16 * arith.constant(16, index=True)

                num_acc_n = n_per_wave // 16
                c_n_per_wave = arith.constant(n_per_wave, index=True)
                wave_n_id = wave_id % arith.constant(num_waves, index=True)
                n_tile_base = wave_n_id * c_n_per_wave

                # slice_k K-slice decode: wid_k = wave_id // num_waves; each group gathers its A K-slice into its own X ping/pong (lds_group_base), runs the mainloop over [_k_base, _k_base+_k_slice), then reduces partials in LDS; slice_k==1 is identity.
                if const_expr(slice_k > 1):
                    wid_k = wave_id / arith.constant(num_waves, index=True)
                    lds_group_base = wid_k * arith.constant(int(lds.input_elems), index=True)
                    async_wave = wave_n_id
                    k_base = wid_k * arith.constant(k_slice, index=True)
                else:
                    wid_k = None
                    lds_group_base = None
                    async_wave = wave_id
                    k_base = k_base_idx

                # Per-expert gate/up N-tile blocks + scale bases (ExpertOperandBases); inter_idx also read below.
                op_bases = ExpertOperandBases(
                    gate_up_interleave=gate_up_interleave,
                    num_acc_n=num_acc_n,
                    n_tile_base=n_tile_base,
                    lane_mod_16=lane_mod_16,
                    w_experts=w_experts,
                    inter_dim=inter_dim,
                    pack_N=pack_N,
                    lane_div_16=lane_div_16,
                    layout_b_scale=layout_b_scale,
                    layout_a_scale=layout_a_scale,
                    sort_block_m=sort_block_m,
                    scale_mn_pack=scale_mn_pack,
                )
                (
                    gate_n_blk_list,
                    gate_n_intra_list,
                    up_n_blk_list,
                    up_n_intra_list,
                    col_g_list,
                    inter_idx,
                ) = op_bases.gate_up_n_blocks(expert_off_idx, by_n)

                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 128
                k_unroll_packed = k_unroll // pack_K
                m_repeat_packed = m_repeat // pack_M
                num_acc_n_packed = num_acc_n // pack_N

                # Per-K-step B (gate/up) VMEM->reg loader; per-tile N-tile lists via for_tile.
                b_loader = BLoader(
                    w_rsrc=w_rsrc,
                    layout_b=layout_b,
                    w_elem_type=w_elem_type,
                    vec2_i64=vec2_i64,
                    b_elem_bytes=b_elem_bytes,
                    kpack_bytes=kpack_bytes,
                    b_nt=b_nt,
                    lane_div_16=lane_div_16,
                    num_acc_n=num_acc_n,
                    k_unroll=k_unroll,
                    gate_up_interleave=gate_up_interleave,
                )
                b_loader.for_tile(gate_n_blk_list, gate_n_intra_list, up_n_blk_list, up_n_intra_list)

                # Scale base elem idx (K-loop invariant): base = mni*stride_n0 + lane_elem; k_elem = ku*stride_k0.
                scale_lane_elem = op_bases.scale_lane_elem()

                gate_scale_bases, up_scale_bases = op_bases.gate_up_scale_bases(
                    expert_off_idx, by_n, inter_idx, scale_lane_elem, num_acc_n_packed
                )

                # raw_a_scale: A-scale staged to LDS once in the prologue (row-major scale_em), read coalesced in K-loop.
                raw_sni = model_dim // 128  # scale_em i32 cols/row (= sblocks/4)
                raw_klane_sh = arith.index_cast(T.i32, lane_div_16) * arith.constant(8, type=T.i32)

                rearrange_a_scale, rearrange_b_scale = make_rearrange_scales(
                    pack_M=pack_M,
                    pack_N=pack_N,
                    scale_mn_pack=scale_mn_pack,
                    bx_m=bx_m,
                    n_tile_base=n_tile_base,
                )

                k_scale = KScaleLoader(
                    sw_rsrc=sw_rsrc,
                    layout_b_scale=layout_b_scale,
                    lds_a_scale=lds_a_scale,
                    lane_mod_16=lane_mod_16,
                    m_repeat_packed=m_repeat_packed,
                    num_acc_n_packed=num_acc_n_packed,
                    gate_up_interleave=gate_up_interleave,
                    raw_sni=raw_sni,
                    raw_klane_sh=raw_klane_sh,
                    sx_rsrc=sx_rsrc,
                    sc_cp_vec=sc_cp_vec,
                    sc_cp_iters=sc_cp_iters,
                    total_threads=group_threads,
                    tx=group_tid,
                )
                k_scale.for_tile(
                    gate_scale_bases=gate_scale_bases,
                    up_scale_bases=up_scale_bases,
                    rearrange_a_scale=rearrange_a_scale,
                    rearrange_b_scale=rearrange_b_scale,
                )

                def prefetch_ab_scale_tile(base_k, ku_packed_limit=k_unroll_packed):
                    return k_scale.prefetch_ab_scale_tile(base_k, ku_packed_limit)

                g2s_x = G2SLoaderX(
                    x_rsrc=x_rsrc,
                    x_elem=x_elem,
                    vec16_elems=vec16_elems,
                    vec16_x=vec16_x,
                    a_elem_bytes=a_elem_bytes,
                    elem_bytes=elem_bytes,
                    c_a_pack=c_a_pack,
                    num_x_loads=num_x_loads,
                    x_load_bytes=x_load_bytes,
                    k_blocks16=k_blocks16,
                    layout_lds=layout_lds,
                    total_threads=group_threads,
                    wave_id=async_wave,
                    use_async_copy=use_async_copy,
                    eff_lds_stride=eff_lds_stride,
                    tile_m=tile_m,
                    lds_group_base=lds_group_base,
                )
                g2s_x.for_tile(x_row_base_div4, x_col_local_i32, x_row_local)
                load_x_tile = g2s_x.load_tile
                store_x_tile_to_lds = g2s_x.store_to_lds
                if const_expr(use_async_copy):
                    prefetch_x_to_lds = g2s_x.prefetch_to_lds

                s2r_a = S2RLoaderA(
                    vec16_x=vec16_x,
                    vec2_i64=vec2_i64,
                    elem_bytes=elem_bytes,
                    a_elem_vec_pack=a_elem_vec_pack,
                    is_f8_a=is_f8_a,
                    k_blocks16=k_blocks16,
                    layout_lds=layout_lds,
                    row_a_lds=row_a_lds,
                    col_offset_base=col_offset_base,
                    m_repeat=m_repeat,
                    k_unroll=k_unroll,
                    lds_group_base=lds_group_base,
                )

                single_b_pipe = gate_up_interleave

                mfma_gu = MfmaScaleGU(
                    is_f8_a=is_f8_a,
                    cbsz=cbsz,
                    blgp=blgp,
                    pack_M=pack_M,
                    pack_N=pack_N,
                    pack_K=pack_K,
                    m_repeat=m_repeat,
                    m_repeat_packed=m_repeat_packed,
                    num_acc_n=num_acc_n,
                    num_acc_n_packed=num_acc_n_packed,
                    k_unroll=k_unroll,
                    vec4_f32=vec4_f32,
                    gate_up_interleave=gate_up_interleave,
                )
                mfma_gu.for_tile()

                # KLoop (mega_moe_kloop): CUTLASS-style collective mainloop (prime -> steady -> drain) in run().
                kloop = KLoop(
                    g2s_x=g2s_x,
                    s2r_a=s2r_a,
                    b_loader=b_loader,
                    k_scale=k_scale,
                    mfma=mfma_gu,
                    lds_x_ping=lds_x_ping,
                    lds_x_pong=lds_x_pong,
                    sw_rsrc=sw_rsrc,
                    layout_b_scale=layout_b_scale,
                    barrier=barrier,
                    acc_init=acc_init,
                    k_base_idx=k_base,
                    k_dim=k_dim,
                    tile_k=tile_k,
                    pack_K=pack_K,
                    pack_N=pack_N,
                    k_unroll=k_unroll,
                    k_unroll_packed=k_unroll_packed,
                    m_repeat=m_repeat,
                    m_repeat_packed=m_repeat_packed,
                    num_acc_n=num_acc_n,
                    num_acc_n_packed=num_acc_n_packed,
                    single_b_pipe=single_b_pipe,
                    use_async_copy=use_async_copy,
                    isched=isched,
                    ck_rate=ck_rate,
                    vmcnt_before_barrier=vmcnt_before_barrier,
                    pipe_n_phases=pipe_n_phases,
                    pp_mfma=pp_mfma,
                    pp_a_reads=pp_a_reads,
                    pp_b_loads=pp_b_loads,
                    pp_has_scale=pp_has_scale,
                    gate_n_blk_list=gate_n_blk_list,
                    gate_n_intra_list=gate_n_intra_list,
                    up_n_blk_list=up_n_blk_list,
                    up_n_intra_list=up_n_intra_list,
                    gate_scale_bases=gate_scale_bases,
                    up_scale_bases=up_scale_bases,
                    rearrange_a_scale=rearrange_a_scale,
                    rearrange_b_scale=rearrange_b_scale,
                )

                # Pipeline (split ping/pong allocators)
                rocdl.sched_barrier(0)

                # raw_a_scale: stage this tile's full A-scale to LDS once before the K-loop (KScaleLoader).
                k_scale.stage_a_scale_to_lds(bx_m)

                k0 = k_base
                if const_expr(use_async_copy):
                    prefetch_x_to_lds(k0, lds_x_pong)
                else:
                    x_regs0 = load_x_tile(k0)
                    store_x_tile_to_lds(x_regs0, lds_x_pong)
                rocdl.sched_barrier(0)
                k0_scale = k_base // arith.constant(pack_K * 128, index=True)
                a_scale_pong, gate_bs_pong, up_bs_pong = prefetch_ab_scale_tile(k0_scale)
                # atom_contract: emit dispatch srcmap -> lds_tid (+ compact/static tile meta) via stage_srcmap_to_lds.
                stage_srcmap_to_lds(
                    tc=tc,
                    addr_disp=addr_disp,
                    tx=tx,
                    lds_tid=lds_tid,
                    sorted_rsrc=sorted_rsrc,
                    expert_rsrc=expert_rsrc,
                    sorted_w_rsrc=sorted_w_rsrc,
                    wts_sorted_rsrc=wts_sorted_rsrc,
                    atom_sw_out_rsrc=atom_sw_out_rsrc,
                    tile_m=tile_m,
                    sort_block_m=sort_block_m,
                    ca=ca,
                    static_tiles=static_tiles,
                    fz_epr=fz_epr,
                    fz_npes=fz_npes,
                    fz_mtpr=fz_mtpr,
                    fz_k=fz_k,
                    fz_tile_m=fz_tile_m,
                    fz_rank=fz_rank,
                )

                # Deep-pipeline mainloop: prime -> ping/pong steady loop -> drain (see KLoop.run).
                acc_gate, acc_up, epilogue_pf = kloop.run(a_scale_pong, gate_bs_pong, up_bs_pong)

                # slice_k: reduce per-K-slice partials across wave groups in LDS and broadcast, BEFORE activation, so silu(sum_gate)*sum_up is the correct full-K result.
                if const_expr(slice_k > 1):
                    acc_gate = reduce_slice_k_partials(
                        acc_gate,
                        lds_scratch=lds_out,
                        wid_k=wid_k,
                        local_tid=group_tid,
                        slice_k=slice_k,
                        vec4_f32=vec4_f32,
                    )
                    if const_expr(not gate_up_interleave):
                        acc_up = reduce_slice_k_partials(
                            acc_up,
                            lds_scratch=lds_out,
                            wid_k=wid_k,
                            local_tid=group_tid,
                            slice_k=slice_k,
                            vec4_f32=vec4_f32,
                        )

                epi_act = Activation(
                    gate_up_interleave=gate_up_interleave,
                    num_acc_n=num_acc_n,
                    m_repeat=m_repeat,
                    pack_N=pack_N,
                    vec4_f32=vec4_f32,
                    f32=f32,
                )
                epi_acc, epi_gui_out_n = epi_act.merge(acc_gate, acc_up)

                # Epilogue: CShuffle + direct store; out[(t*topk+s)*inter_dim + col] = silu(gate) * up.
                mask24_i32 = arith.constant(0xFFFFFF)
                out_base_idx = tensor_base_addr_index(arg_out)

                e_vec = e_vec_s1
                cshuffle_nlane = min(32, tile_n // e_vec)

                out_row_stride = inter_dim // 2 if need_fp4 else inter_dim

                # OutputQuant ctor materialises the epilogue constant locals (preserves trace op order).
                epi_quant = OutputQuant(
                    need_fp4=need_fp4,
                    need_fp8=need_fp8,
                    need_sort=need_sort,
                    e_vec=e_vec,
                    num_shuffle_steps=num_shuffle_steps_s1,
                    shuffle_dists=shuffle_dists_s1,
                    sorted_scale_cols_i32=sorted_scale_cols_i32,
                    sorted_scale_rsrc=sorted_scale_rsrc,
                    sort_block_m=sort_block_m,
                )
                epi_scatter = Scatter(
                    contiguous_io=True,
                    out_base_idx=out_base_idx,
                    out_row_stride=out_row_stride,
                    lds_tid=lds_tid,
                    mask24_i32=mask24_i32,
                    num_valid_i32=num_valid_i32,
                    topk_i32_v=topk_i32,
                    tokens_i32_v=tokens_i32,
                    topk=topk,
                    fz_npes=fz_npes,
                    fz_mtpr=fz_mtpr,
                    fz_k=fz_k,
                    static_tiles=static_tiles,
                    fz_epr=fz_epr,
                    fz_tile_m=fz_tile_m,
                    cnt_ef=cnt_ef,
                    kf=kf,
                    bx=bx,
                    quant=epi_quant,
                )

                frag_elem = T.f32

                # slice_k>1: reduction broadcasts the summed acc to every group, so each runs the cshuffle epilogue over its OWN group_threads/group_tid; idempotent writes keep control flow uniform.
                epi = Gemm1Epilogue(activation=epi_act, quant=epi_quant, scatter=epi_scatter)
                epi.run(
                    acc=epi_acc,
                    gui_out_n=epi_gui_out_n,
                    gate_up_interleave=gate_up_interleave,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    e_vec=e_vec,
                    cshuffle_nlane=cshuffle_nlane,
                    total_threads=group_threads,
                    m_repeat=m_repeat,
                    num_acc_n=num_acc_n,
                    tx=group_tid,
                    lane_div_16=lane_div_16,
                    lane_mod_16=lane_mod_16,
                    bx_m=bx_m,
                    by_n=by_n,
                    n_tile_base=n_tile_base,
                    lds_out=lds_out,
                    lds_out_B=lds_out_B,
                    frag_elem=frag_elem,
                )

            with _guard(blk_valid):
                with _guard(exp_valid):
                    moe_gemm1_body()

            gpu.barrier()

    # -- Host launcher --
    cache_tag = (
        module_name,
        a_dtype,
        b_dtype,
        out_dtype,
        tile_m,
        tile_n,
        tile_k,
        use_async_copy,
        waves_per_eu,
        gate_mode,
        xcd_swizzle,
    )

    # Fused launcher.  addr_payload_done/addr_expected_real are 0 for fixedslot (strict-phase).
    @flyc.jit
    def launch_fused_moe_gemm1(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        arg_bias: fx.Tensor,
        arg_out_scale_sorted: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        addr_payload_done: fx.Int64,
        addr_expected_real: fx.Int64,
        addr_disp: fx.Int64,
        i32_cur_tok: fx.Int32,
        addr_in_tok: fx.Int64,
        addr_in_idx: fx.Int64,
        addr_in_wts: fx.Int64,
        addr_in_sc: fx.Int64,
        stream: fx.Stream,
    ):
        _ = cache_tag
        finalize_lds(allocator_pong, allocator_ping)

        inter_in = arith.index_cast(T.index, i32_inter_in.ir_value())
        tile_n_index = arith.constant(tile_n, index=True)
        if const_expr(gate_up_interleave):
            gx = (inter_in + tile_n_index - 1) / tile_n_index
        else:
            gx = (inter_in + 2 * tile_n_index - 1) / tile_n_index / arith.constant(2, index=True)
        # co-resident grid: total = gx*gy <= cu_num keeps the dispatch-prologue grid-sync deadlock-free.
        gy = arith.constant(fz_gy, index=True)

        moe_gemm1(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_max_token_ids,
            arg_bias,
            arg_out_scale_sorted,
            i32_tokens_in,
            i32_inter_in,
            i32_k_in,
            i32_size_expert_ids_in,
            addr_payload_done,
            addr_expected_real,
            addr_disp,
            i32_cur_tok,
            addr_in_tok,
            addr_in_idx,
            addr_in_wts,
            addr_in_sc,
        ).launch(grid=(gx, gy, 1), block=(total_threads, 1, 1), stream=stream)

    # Occupancy-gate sizing attrs (host-only): lds_total_bytes (post-floor) / lds_data_bytes / lds_scale_bytes.
    launch_fused_moe_gemm1.lds_total_bytes = int(lds_total_bytes)
    launch_fused_moe_gemm1.lds_data_bytes = int(lds_data_bytes)
    launch_fused_moe_gemm1.lds_scale_bytes = int(lds_scale_bytes)
    launch_fused_moe_gemm1.raw_a_scale = True
    return launch_fused_moe_gemm1
