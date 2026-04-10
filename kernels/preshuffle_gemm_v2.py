# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Preshuffle GEMM kernel — Layout API version.

Supports f16, bf16 via layout API (fx.copy + fx.gemm).
fp8 has two paths:
  - "fp8_layout": layout API with 64b copy atoms (needs C++ k_perm fix)
  - "fp8" (default): delegates to old preshuffle_gemm path (full perf, 128b loads)

Uses scf.for tile loop with ping-pong double buffer (2-stage B).
Includes hot_loop_scheduler from the old pipeline for instruction scheduling.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, gpu, buffer_ops, rocdl, range_constexpr
from flydsl.expr.typing import T, Float16, BFloat16, Float8E4M3FNUZ, Float8E4M3FN, Int8
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.runtime.device import get_rocm_arch
from flydsl._mlir import ir
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
from kernels.mfma_epilogues import mfma_epilog
from typing import Optional


def compile_preshuffle_gemm_v2(
    *,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    in_dtype: str = "fp8",
    out_dtype: str = "bf16",
    waves_per_eu: Optional[int] = None,
):
    """Compile preshuffle GEMM using the layout API.

    Supports in_dtype: fp8, fp8_layout, fp16, bf16.
    Returns a JitFunction: fn(C, A, B, scale_a, scale_b, M, N, stream).
    """
    if in_dtype not in ("fp8", "fp8_layout", "fp16", "bf16"):
        raise ValueError(f"in_dtype must be fp8/fp8_layout/fp16/bf16, got {in_dtype!r}")

    # fp8 (default): delegate to old path for full perf + 128b loads
    if in_dtype == "fp8":
        return compile_preshuffle_gemm_a8(
            N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            in_dtype="fp8", out_dtype=out_dtype, waves_per_eu=waves_per_eu,
        )

    # fp8_layout: use layout API with 64b copy atoms
    is_fp8 = in_dtype == "fp8_layout"
    is_f16 = in_dtype == "fp16"
    is_bf16 = in_dtype == "bf16"
    is_f16_or_bf16 = is_f16 or is_bf16
    out_is_bf16 = out_dtype == "bf16"
    elem_bytes = 1 if is_fp8 else 2

    gpu_arch = get_rocm_arch()
    is_gfx942 = str(gpu_arch).startswith("gfx942")
    if is_fp8 and str(gpu_arch).startswith("gfx950"):
        raise NotImplementedError("fp8 gfx950 uses mfma_scale, not supported yet")

    if is_f16:
        layout_elem = Float16
    elif is_bf16:
        layout_elem = BFloat16
    else:
        layout_elem = Int8

    out_elem_cls = BFloat16 if out_is_bf16 else Float16

    # Tile geometry
    k_iters = tile_k // 32  # tiled_K=32 with k_perm (4,4,2) for f16/bf16
    num_tiles = K // tile_k
    m_repeat = tile_m // 16
    num_waves = 4
    n_per_wave = tile_n // num_waves
    num_acc_n = n_per_wave // 16
    n_accs = m_repeat * num_acc_n
    acc_size = n_accs * 4

    # LDS: ping + pong
    smem_bytes = tile_m * tile_k * elem_bytes * 2

    total_threads = 256
    a_load_bytes = 16
    bytes_per_thread_a = (tile_m * tile_k * elem_bytes) // total_threads
    num_a_loads = bytes_per_thread_a // a_load_bytes

    # ── Kernel ────────────────────────────────────────────────────────
    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        tiled_mma: fx.TiledMma,
        tiled_copy_g2s: fx.TiledCopy,
    ):
        tid = fx.thread_idx.x
        bid_x, bid_y, _ = fx.block_idx

        gA = fx.rocdl.make_buffer_tensor(arg_a)
        gB = fx.rocdl.make_buffer_tensor(arg_b)
        gC = fx.rocdl.make_buffer_tensor(arg_c)

        tA = fx.flat_divide(gA, fx.make_tile(tile_m, tile_k))[None, None, bid_x, None]
        tB = fx.flat_divide(gB, fx.make_tile(tile_n, tile_k))[None, None, bid_y, None]
        tC = fx.flat_divide(gC, fx.make_tile(tile_m, tile_n))[None, None, bid_x, bid_y]

        # Copy atoms: fp8 uses 64b, f16/bf16 uses 128b
        if is_fp8:
            mma_copy = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), layout_elem)
            mma_uni = fx.make_copy_atom(fx.UniversalCopy64b(), layout_elem)
        else:
            mma_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), layout_elem)
            mma_uni = fx.make_copy_atom(fx.UniversalCopy128b(), layout_elem)
        buf_copy_g2s = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), layout_elem)
        uni_copy_g2s = fx.make_copy_atom(fx.UniversalCopy128b(), layout_elem)

        # Per-thread slices
        thr_mma = tiled_mma.thr_slice(tid)
        thr_g2s = tiled_copy_g2s.get_slice(tid)
        thr_s2r = fx.make_tiled_copy_A(mma_copy, tiled_mma).get_slice(tid)
        thr_g2r_B = fx.make_tiled_copy_B(mma_copy, tiled_mma).get_slice(tid)

        # LDS with XOR16 swizzle (f16/bf16) or no swizzle (fp8)
        smem_ptr = fx.recast_iter(
            fx.PointerType.get(layout_elem.ir_type, fx.AddressSpace.Shared, 512),
            fx.get_dyn_shared(),
        )
        swz = fx.SwizzleType.get(3, 3, 3) if is_f16_or_bf16 else fx.SwizzleType.get(0, 0, 0)
        sA = fx.make_view(smem_ptr, fx.make_composed_layout(
            fx.static(swz),
            fx.make_ordered_layout((tile_m, tile_k, 2), (1, 0, 2)),
        ))

        # Partitions
        pA_g = thr_g2s.partition_S(tA)
        pA_s = thr_g2s.partition_D(sA)
        pA_s2r = thr_s2r.partition_S(sA)
        pB_g = thr_g2r_B.partition_S(tB)

        # Fragments — 2-stage B double buffer
        frag_copy_A = fx.make_fragment_like(pA_s[None, None, None, 0])
        frag_A = thr_mma.make_fragment_A(sA[None, None, 0])
        frag_B = fx.make_fragment_like(
            fx.flat_product(
                thr_mma.partition_B(tB).layout(None, None, None, 0),
                fx.make_layout(2, 1),
            ),
            layout_elem.ir_type,
        )
        frag_C = thr_mma.make_fragment_C(tC)
        frag_A_retile = thr_s2r.retile(frag_A)
        frag_B_retile = thr_g2r_B.retile(frag_B)
        if is_f16_or_bf16:
            buf_copy_out = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_elem_cls)
            thr_r2g_C = fx.make_tiled_copy_C(buf_copy_out, tiled_mma).get_slice(tid)
            pC_g = thr_r2g_C.partition_S(tC)
            frag_C_out = fx.make_fragment_like(frag_C, out_elem_cls.ir_type)
            frag_C_retile = thr_r2g_C.retile(frag_C_out)

        # ── Scheduling hints (ported from old pipeline) ───────────
        def hot_loop_scheduler():
            mfma_group = num_acc_n
            mfma_total = (k_iters * 2) * m_repeat * mfma_group
            mfma_per_iter = 2 * mfma_group
            sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)

            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(1)
            if tile_m == 16:
                rocdl.sched_vmem(1)
            rocdl.sched_mfma(1)
            if tile_m == 16:
                rocdl.sched_vmem(1)

            if num_acc_n < 4:
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(1)
                if tile_m == 16:
                    rocdl.sched_vmem(1)
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(1)
                if tile_m == 16:
                    rocdl.sched_vmem(1)
                rocdl.sched_mfma(1)

            dswr_tail = num_a_loads
            dstr_advance = 2
            if dswr_tail > sche_iters:
                dswr_tail = sche_iters
            dswr_start = max(sche_iters - dswr_tail - dstr_advance, 0)

            for sche_i in range_constexpr(sche_iters):
                rocdl.sched_vmem(1)
                rocdl.sched_mfma(mfma_group)
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(mfma_group)
                if sche_i >= dswr_start - 1:
                    rocdl.sched_dswr(1)

            rocdl.sched_barrier(0)

        # ── Pipeline stage (double-buffered B) ────────────────────
        def pipeline_stage(read_stage, next_k_val=None, read_next=True):
            write_stage = read_stage ^ 1
            # 1. Prefetch next A tile (global → register)
            if read_next and next_k_val is not None:
                fx.copy(buf_copy_g2s, pA_g[None, None, None, next_k_val], frag_copy_A)
            # 2. Compute: A from LDS + MFMA with current B
            for ki in range_constexpr(k_iters):
                fx.copy(mma_uni, pA_s2r[None, None, ki, read_stage],
                        frag_A_retile[None, None, ki])
                fx.gemm(tiled_mma, frag_C,
                        frag_A[None, None, (None, ki)],
                        frag_B[None, None, (None, ki), read_stage],
                        frag_C)
            # 3. Load next B tile (after compute, overlaps with LDS write + barrier)
            if read_next and next_k_val is not None:
                fx.copy(mma_copy, pB_g[None, None, None, next_k_val],
                        frag_B_retile[None, None, None, write_stage])
            # 4. Write A tile to LDS + barrier
            fx.copy(uni_copy_g2s, frag_copy_A, pA_s[None, None, None, write_stage])
            gpu.barrier()
            # NOTE: hot_loop_scheduler intentionally NOT called here.
            # Testing shows LLVM HW scheduler outperforms manual scheduling
            # for the layout API's instruction pattern (98% vs 83% for 64x128x64).

        # ── Prologue ──────────────────────────────────────────────
        fx.copy(buf_copy_g2s, pA_g[None, None, None, 0], frag_copy_A)
        fx.copy(mma_copy, pB_g[None, None, None, 0], frag_B_retile[None, None, None, 0])
        frag_C.store(arith.constant_vector(0.0, T.vec(acc_size, T.f32)))
        fx.copy(uni_copy_g2s, frag_copy_A, pA_s[None, None, None, 0])
        gpu.barrier()
        rocdl.sched_barrier(0)

        # ── Main tile loop (scf.for with ping-pong) ──────────────
        if num_tiles == 1:
            pipeline_stage(read_stage=0, read_next=False)
        elif num_tiles == 2:
            pipeline_stage(read_stage=0, next_k_val=fx.Int32(1))
            pipeline_stage(read_stage=1, read_next=False)
        else:
            loop_start = fx.Index(0)
            loop_end = fx.Index((num_tiles - 2) // 2)
            loop_step = fx.Index(1)
            for iv, _ in range(loop_start, loop_end, loop_step, init=[]):
                k_base = arith.index_cast(T.i32, iv * 2)
                pipeline_stage(read_stage=0, next_k_val=k_base + fx.Int32(1))
                pipeline_stage(read_stage=1, next_k_val=k_base + fx.Int32(2))
                yield []
            pipeline_stage(read_stage=0, next_k_val=fx.Int32(num_tiles - 1))
            pipeline_stage(read_stage=1, read_next=False)

        # ── Epilogue ─────────────────────────────────────────────
        if is_fp8:
            # FP8: per-token/per-column scales + mfma_epilog buffer_store
            c_n = arith.index_cast(T.index, i32_n)
            bx_m = gpu.block_id("x") * tile_m
            by_n = gpu.block_id("y") * tile_n
            wave_id = gpu.thread_id("x") // 64
            lane_id = gpu.thread_id("x") % 64
            lane_div_16 = lane_id // 16
            lane_mod_16 = lane_id % 16
            n_tile_base = wave_id * n_per_wave

            _c_nrec = arith.index_cast(T.i64, c_n * arith.index_cast(T.index, i32_m) * 2)
            c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=False, num_records_bytes=_c_nrec)
            scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a, max_size=False)
            scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b, max_size=True)

            acc_vec = frag_C.load()
            final_accs = [vector.from_elements(T.f32x4, [
                vector.extract(acc_vec, static_position=[i*4+j], dynamic_position=[])
                for j in range(4)]) for i in range_constexpr(n_accs)]

            s_b_vals = [buffer_ops.buffer_load(scale_b_rsrc,
                        by_n + n_tile_base + ni * 16 + lane_mod_16, vec_width=1, dtype=T.f32)
                        for ni in range_constexpr(num_acc_n)]
            s_a_vecs = [vector.bitcast(T.f32x4, buffer_ops.buffer_load(scale_a_rsrc,
                        bx_m + mi * 16 + lane_div_16 * 4, vec_width=4, dtype=T.f32))
                        for mi in range_constexpr(m_repeat)]

            def body_row(*, mi, ii, row_in_tile, row):
                col_base = by_n + n_tile_base + lane_mod_16
                idx_base = row * c_n + col_base
                s_a = vector.extract(s_a_vecs[mi], static_position=[ii], dynamic_position=[])
                for ni in range_constexpr(num_acc_n):
                    val = vector.extract(final_accs[mi * num_acc_n + ni],
                                         static_position=[ii], dynamic_position=[])
                    val_out = arith.trunc_f(out_elem_cls.ir_type, (val * s_a) * s_b_vals[ni])
                    buffer_ops.buffer_store(val_out, c_rsrc, idx_base + ni * 16)

            mfma_epilog(
                use_cshuffle=False, arith=arith, range_constexpr=range_constexpr,
                m_repeat=m_repeat, lane_div_16=lane_div_16, bx_m=bx_m, body_row=body_row,
            )
        else:
            # f16/bf16: truncate + vectorized fx.copy
            frag_C_out.store(arith.trunc_f(T.vec(acc_size, out_elem_cls.ir_type), frag_C.load()))
            fx.copy(buf_copy_out, frag_C_retile, pC_g)

    # ── Host launcher ─────────────────────────────────────────────
    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        ctx = CompilationContext.get_current()

        # MMA atom
        if is_f16:
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, Float16))
            k_perm = fx.make_layout((4, 4, 2), (1, 8, 4))
        elif is_bf16:
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, BFloat16))
            k_perm = fx.make_layout((4, 4, 2), (1, 8, 4))
        else:
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, layout_elem))
            k_perm = None

        if k_perm is not None:
            tiled_mma = fx.make_tiled_mma(
                mma_atom, fx.make_layout((1, 4, 1), (0, 1, 0)),
                fx.make_tile(None, None, k_perm),
            )
        else:
            tiled_mma = fx.make_tiled_mma(
                mma_atom, fx.make_layout((1, 4, 1), (0, 1, 0)),
            )

        # G2S tiled copy
        val_per_thr = a_load_bytes // elem_bytes
        thrs_k = tile_k // val_per_thr
        thrs_m = total_threads // thrs_k
        tiled_copy_g2s = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy128b(), layout_elem),
            fx.make_layout(
                ((thrs_k, thrs_m), (1, val_per_thr)),
                ((thrs_m * val_per_thr, 1), (1, thrs_m)),
            ),
            fx.make_tile(thrs_m, tile_k),
        )

        # Preshuffle B layout (2D hierarchical)
        kp_bytes = 16
        kp_elems = kp_bytes if elem_bytes == 1 else kp_bytes // elem_bytes
        k_bytes_b = K * elem_bytes
        n0 = N // 16
        k0 = k_bytes_b // 64
        s_nlane = kp_elems
        s_klane = 16 * s_nlane
        s_k0 = 4 * s_klane
        s_n0 = k0 * s_k0
        preshuffle_B = fx.Tensor(fx.make_view(
            fx.get_iter(arg_b),
            fx.make_layout(((16, n0), (kp_elems, 4, k0)),
                           ((s_nlane, s_n0), (1, s_klane, s_k0))),
        ))

        # Reshape A and C to 2D
        M_max = 65536
        arg_a_2d = fx.Tensor(fx.make_view(
            fx.get_iter(arg_a), fx.make_layout((M_max, K), (K, 1)),
        ))
        arg_c_2d = fx.Tensor(fx.make_view(
            fx.get_iter(arg_c), fx.make_layout((M_max, N), (N, 1)),
        ))

        gx = (i32_m + (tile_m - 1)) // tile_m
        gy = i32_n // tile_n

        launcher = kernel_gemm(
            arg_c_2d, arg_a_2d, preshuffle_B,
            arg_scale_a, arg_scale_b, i32_m, i32_n,
            tiled_mma, tiled_copy_g2s,
        )
        if waves_per_eu is not None and int(waves_per_eu) >= 1:
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, 'attributes') and op.OPERATION_NAME == "gpu.func":
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                        T.i32, int(waves_per_eu))
        launcher.launch(
            grid=(gx, gy, 1), block=(256, 1, 1), smem=smem_bytes, stream=stream,
        )

    return launch_gemm
