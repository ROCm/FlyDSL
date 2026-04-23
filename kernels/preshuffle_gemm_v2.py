# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Preshuffle GEMM kernel — Layout API version.

Supports f16, bf16, fp8 via layout API (fx.copy + fx.gemm).
Uses scf.for tile loop with ping-pong double buffer (2-stage B).
Includes hot_loop_scheduler from the old pipeline for instruction scheduling.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import vector, gpu, rocdl, range_constexpr
from flydsl.expr.typing import T, Float16, Float32, BFloat16, Float8E4M3FNUZ, Float8E4M3FN, Int8, Vector as Vec
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.runtime.device import get_rocm_arch
from flydsl._mlir import ir
from kernels.preshuffle_gemm import _get_preload
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
    enable_scheduler: bool = True,
):
    """Compile preshuffle GEMM using the layout API.

    Supports in_dtype: fp8, fp16, bf16.
    Returns a JitFunction: fn(C, A, B, scale_a, scale_b, M, N, stream).
    """
    if in_dtype not in ("fp8", "fp16", "bf16"):
        raise ValueError(f"in_dtype must be fp8/fp16/bf16, got {in_dtype!r}")

    is_fp8 = in_dtype == "fp8"
    is_f16 = in_dtype == "fp16"
    is_bf16 = in_dtype == "bf16"
    is_f16_or_bf16 = is_f16 or is_bf16
    out_is_bf16 = out_dtype == "bf16"
    elem_bytes = 1 if is_fp8 else 2

    gpu_arch = get_rocm_arch()
    is_gfx942 = str(gpu_arch).startswith("gfx942")
    is_gfx950 = str(gpu_arch).startswith("gfx950")
    # TODO: enable when CDNA4 MFMA_Scale works through layout API (fly.mma_atom_call)
    use_mfma_scale_128 = False  # is_fp8 and is_gfx950
    use_mfma_k32 = is_f16_or_bf16 and is_gfx950
    if use_mfma_scale_128:
        if tile_k % 128 != 0:
            raise ValueError(f"tile_k must be divisible by 128 for gfx950 fp8, got {tile_k}")

    if is_f16:
        layout_elem = Float16
    elif is_bf16:
        layout_elem = BFloat16
    elif is_gfx950:
        layout_elem = Float8E4M3FN
    else:
        layout_elem = Float8E4M3FNUZ

    out_elem_cls = BFloat16 if out_is_bf16 else Float16

    # Tile geometry
    # k_perm groups atoms: 32 for f16/bf16 K=16 (2 atoms), 32 for K=32 (1 atom),
    # 128 for gfx950 fp8 (1×K=128), 64 for gfx942 fp8 (2×K=32)
    tile_K_perm = 128 if use_mfma_scale_128 else (64 if is_fp8 else 32)
    k_iters = tile_k // tile_K_perm
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
    num_b_loads = (tile_n * tile_k * elem_bytes) // total_threads // 16
    num_ds_load = (tile_m * tile_k * elem_bytes) // 64 // 16  # A LDS reads per wave
    num_gmem_loads = num_a_loads + num_b_loads
    if is_fp8 and is_gfx950:
        dsrd_preload, dvmem_preload = _get_preload(tile_m, tile_n, tile_k)
    else:
        dsrd_preload, dvmem_preload = (0, 0)

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

        # Copy atoms: 128b for all dtypes (matches old path's buffer_load_dwordx4 / ds_read_b128)
        mma_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), layout_elem)
        mma_uni = fx.make_copy_atom(fx.UniversalCopy128b(), layout_elem)
        buf_copy_g2s = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), layout_elem)
        uni_copy_g2s = fx.make_copy_atom(fx.UniversalCopy128b(), layout_elem)

        # Per-thread slices
        thr_mma = tiled_mma.thr_slice(tid)
        thr_g2s = tiled_copy_g2s.get_slice(tid)
        thr_s2r = fx.make_tiled_copy_A(mma_copy, tiled_mma).get_slice(tid)
        thr_g2r_B = fx.make_tiled_copy_B(mma_copy, tiled_mma).get_slice(tid)

        # LDS: XOR swizzle for f16/bf16 to avoid bank conflicts, identity for fp8
        smem_ptr = fx.recast_iter(
            fx.PointerType.get(layout_elem.ir_type, fx.AddressSpace.Shared, 512),
            fx.get_dyn_shared(),
        )
        if is_fp8:
            sA = fx.make_view(smem_ptr,
                fx.make_ordered_layout((tile_m, tile_k, 2), (1, 0, 2)))
        else:
            swz = fx.SwizzleType.get(3, 3, 3)
            sA = fx.make_view(smem_ptr, fx.make_composed_layout(
                fx.static(swz),
                fx.make_ordered_layout((tile_m, tile_k, 2), (1, 0, 2)),
            ))

        # Partitions
        pA_g = thr_g2s.partition_S(tA)
        pA_s = thr_g2s.partition_D(sA)
        pA_s2r = thr_s2r.partition_S(sA)
        pB_g = thr_g2r_B.partition_S(tB)

        # Fragments — 2 separate B fragments (split double buffer for VGPR lifetime)
        frag_copy_A = fx.make_fragment_like(pA_s[None, None, None, 0])
        frag_A = thr_mma.make_fragment_A(sA[None, None, 0])
        frag_B_single_layout = thr_mma.partition_B(tB).layout(None, None, None, 0)
        frag_B_0 = fx.make_fragment_like(frag_B_single_layout, layout_elem.ir_type)
        frag_B_1 = fx.make_fragment_like(frag_B_single_layout, layout_elem.ir_type)
        frag_B_stages = [frag_B_0, frag_B_1]
        frag_C = thr_mma.make_fragment_C(tC)
        frag_A_retile = thr_s2r.retile(frag_A)
        frag_B_0_retile = thr_g2r_B.retile(frag_B_0)
        frag_B_1_retile = thr_g2r_B.retile(frag_B_1)
        frag_B_retile_stages = [frag_B_0_retile, frag_B_1_retile]
        buf_copy_out = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_elem_cls)
        thr_r2g_C = fx.make_tiled_copy_C(buf_copy_out, tiled_mma).get_slice(tid)
        pC_g = thr_r2g_C.partition_S(tC)
        frag_C_out = fx.make_fragment_like(frag_C, out_elem_cls.ir_type)
        frag_C_retile = thr_r2g_C.retile(frag_C_out)

        # ── Scheduling hints (ported from old pipeline) ───────────
        def build_scheduler(numer: int, denom: int):
            if denom <= 0:
                return []
            if numer <= 0:
                return [0] * denom
            out = []
            prev = 0
            for i in range_constexpr(denom):
                cur = ((i + 1) * numer + (denom - 1)) // denom
                out.append(cur - prev)
                prev = cur
            return out

        def hot_loop_scheduler():
            mfma_group = num_acc_n

            if is_gfx942:
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
            else:
                # gfx950 path: distribute vmem/dsrd across MFMA slots
                if use_mfma_k32:
                    element_k_per_mfma = 32
                elif is_fp8:
                    element_k_per_mfma = 128  # mfma_scale_f32_16x16x128
                else:
                    element_k_per_mfma = 16
                num_mfma_per_tile_k = tile_k // element_k_per_mfma
                mfma_total = num_mfma_per_tile_k * m_repeat * mfma_group
                dswr_tail = num_a_loads
                dstr_advance = 2
                if dswr_tail > mfma_total:
                    dswr_tail = mfma_total
                dsrd_preload_eff = min(int(dsrd_preload), num_ds_load)
                dvmem_preload_eff = min(int(dvmem_preload), num_gmem_loads)
                vmem_remaining = num_gmem_loads - dvmem_preload_eff
                dsrd_remaining = num_ds_load - dsrd_preload_eff
                if vmem_remaining > 0 and vmem_remaining < mfma_total:
                    vmem_schedule = (build_scheduler(vmem_remaining, vmem_remaining)
                                     + [0] * (mfma_total - vmem_remaining))
                else:
                    vmem_schedule = build_scheduler(vmem_remaining, mfma_total)
                dsrd_schedule = build_scheduler(dsrd_remaining, mfma_total)
                dswr_start = max(mfma_total - dswr_tail - dstr_advance, 0)
                last_dsrd_mfma_idx = -1
                for sched_idx in range_constexpr(mfma_total):
                    if dsrd_schedule[sched_idx]:
                        last_dsrd_mfma_idx = sched_idx
                dswr_start = max(dswr_start, last_dsrd_mfma_idx + 1)
                idx_ds_read = dsrd_preload_eff
                idx_gmem_load = dvmem_preload_eff
                idx_ds_write = 0
                if dvmem_preload_eff:
                    rocdl.sched_vmem(dvmem_preload_eff)
                if dsrd_preload_eff:
                    rocdl.sched_dsrd(dsrd_preload_eff)
                for mfma_idx in range_constexpr(mfma_total):
                    rocdl.sched_mfma(1)
                    n_dsrd = dsrd_schedule[mfma_idx]
                    if n_dsrd and (idx_ds_read < num_ds_load):
                        if idx_ds_read + n_dsrd > num_ds_load:
                            n_dsrd = num_ds_load - idx_ds_read
                        if n_dsrd:
                            rocdl.sched_dsrd(n_dsrd)
                            idx_ds_read += n_dsrd
                    n_vmem = vmem_schedule[mfma_idx]
                    if n_vmem and (idx_gmem_load < num_gmem_loads):
                        if idx_gmem_load + n_vmem > num_gmem_loads:
                            n_vmem = num_gmem_loads - idx_gmem_load
                        if n_vmem:
                            rocdl.sched_vmem(n_vmem)
                            idx_gmem_load += n_vmem
                    if (idx_ds_write < dswr_tail) and (mfma_idx >= dswr_start):
                        rocdl.sched_dswr(1)
                        idx_ds_write += 1
                if idx_ds_write < num_a_loads:
                    rocdl.sched_dswr(num_a_loads - idx_ds_write)

            rocdl.sched_barrier(0)

        # ── Pipeline stage (double-buffered B via split fragments) ─
        def pipeline_stage(read_stage, next_k_val=None, read_next=True):
            write_stage = read_stage ^ 1
            cur_frag_B = frag_B_stages[read_stage]
            # 1. Prefetch next A tile (global → register)
            if read_next and next_k_val is not None:
                fx.copy(buf_copy_g2s, pA_g[None, None, None, next_k_val], frag_copy_A)
            # 2. Load next B tile (before compute — matches v1 pipeline order,
            #    all vmem available for scheduler interleaving with MFMAs)
            if read_next and next_k_val is not None:
                fx.copy(mma_copy, pB_g[None, None, None, next_k_val],
                        frag_B_retile_stages[write_stage])
            # 3. Compute: A from LDS + MFMA with current B
            for ki in range_constexpr(k_iters):
                fx.copy(mma_uni, pA_s2r[None, None, ki, read_stage],
                        frag_A_retile[None, None, ki])
                # K=128 or K=32 (1 atom): frag K dim is flat k_iters → coord = ki
                # K=16 gfx942 (2 atoms): frag K dim is (atoms, k_iters) → coord = (None, ki)
                k_coord = ki if (use_mfma_scale_128 or use_mfma_k32) else (None, ki)
                fx.gemm(tiled_mma, frag_C,
                        frag_A[None, None, k_coord],
                        cur_frag_B[None, None, k_coord],
                        frag_C)
            # 4. Write A tile to LDS + barrier
            fx.copy(uni_copy_g2s, frag_copy_A, pA_s[None, None, None, write_stage])
            if enable_scheduler:
                hot_loop_scheduler()
            gpu.barrier()

        # ── Prologue ──────────────────────────────────────────────
        fx.copy(buf_copy_g2s, pA_g[None, None, None, 0], frag_copy_A)
        fx.copy(mma_copy, pB_g[None, None, None, 0], frag_B_retile_stages[0])
        frag_C.store(Vec.filled(acc_size, 0.0, Float32))
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
            # Loop-carried values:
            #   bf16/f16: acc + B stage 0 (B alloca types don't match for SROA)
            #   fp8: acc only (B alloca has uniform i64 types → SROA promotes it)
            acc_init = frag_C.load()
            if is_fp8:
                for iv, state in range(loop_start, loop_end, loop_step,
                                       init=[acc_init]):
                    frag_C.store(state[0])
                    k_base = fx.Int32(iv * 2)
                    pipeline_stage(read_stage=0, next_k_val=k_base + fx.Int32(1))
                    pipeline_stage(read_stage=1, next_k_val=k_base + fx.Int32(2))
                    results = yield [frag_C.load()]
                frag_C.store(results)
            else:
                b0_init = frag_B_stages[0].load()
                for iv, state in range(loop_start, loop_end, loop_step,
                                       init=[acc_init, b0_init]):
                    frag_C.store(state[0])
                    frag_B_stages[0].store(state[1])
                    k_base = fx.Int32(iv * 2)
                    pipeline_stage(read_stage=0, next_k_val=k_base + fx.Int32(1))
                    pipeline_stage(read_stage=1, next_k_val=k_base + fx.Int32(2))
                    results = yield [frag_C.load(), frag_B_stages[0].load()]
                frag_C.store(results[0])
                frag_B_stages[0].store(results[1])
            pipeline_stage(read_stage=0, next_k_val=fx.Int32(num_tiles - 1))
            pipeline_stage(read_stage=1, read_next=False)

        # ── Epilogue ─────────────────────────────────────────────
        if is_fp8:
            # FP8: inline scale multiply via layout API buffer loads
            # Accumulator layout: [mi*num_acc_n*4 + ni*4 + ii]
            #   scale_a depends on row (mi, ii), scale_b depends on col (ni)
            bx_m = gpu.block_id("x") * tile_m
            by_n = gpu.block_id("y") * tile_n
            wave_id = gpu.thread_id("x") // 64
            lane_id = gpu.thread_id("x") % 64
            lane_div_16 = lane_id // 16
            lane_mod_16 = lane_id % 16
            n_tile_base = wave_id * n_per_wave

            # Scale buffer tensors + scalar copy atom
            scale_a_buf = fx.rocdl.make_buffer_tensor(arg_scale_a, max_size=True)
            scale_b_buf = fx.rocdl.make_buffer_tensor(arg_scale_b, max_size=True)
            scale_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
            scale_reg_ty = fx.MemRefType.get(T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
            scale_reg_lay = fx.make_layout(1, 1)
            scale_a_div = fx.logical_divide(scale_a_buf, fx.make_layout(1, 1))
            scale_b_div = fx.logical_divide(scale_b_buf, fx.make_layout(1, 1))

            def load_scale(div_tensor, index):
                r = fx.memref_alloca(scale_reg_ty, scale_reg_lay)
                fx.copy_atom_call(scale_copy, fx.slice(div_tensor, (None, fx.Int32(index))), r)
                return Vec(fx.memref_load_vec(r))[0]

            # Load per-column scales: 1 scalar per N-block
            s_b_vals = [load_scale(scale_b_div, by_n + n_tile_base + ni * 16 + lane_mod_16)
                        for ni in range_constexpr(num_acc_n)]
            # Load per-row scales: 1 scalar per row per thread
            s_a_vals = [[load_scale(scale_a_div, bx_m + mi * 16 + lane_div_16 * 4 + ii)
                         for ii in range_constexpr(4)]
                        for mi in range_constexpr(m_repeat)]

            # Build scaled accumulator inline
            acc_vec = Vec(frag_C.load())
            scaled_elems = []
            for mi in range_constexpr(m_repeat):
                for ni in range_constexpr(num_acc_n):
                    for ii in range_constexpr(4):
                        idx = mi * num_acc_n * 4 + ni * 4 + ii
                        val = acc_vec[idx]
                        s_a = s_a_vals[mi][ii]
                        scaled_val = (val * s_a) * s_b_vals[ni]
                        scaled_elems.append(scaled_val.to(out_elem_cls))

            out_vec = vector.from_elements(
                T.vec(acc_size, out_elem_cls.ir_type), scaled_elems)
            frag_C_out.store(out_vec)
            fx.copy(buf_copy_out, frag_C_retile, pC_g)
        else:
            # f16/bf16: truncate + vectorized fx.copy
            frag_C_out.store(Vec(frag_C.load()).to(out_elem_cls))
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

        # MMA atom — layout_elem carries the dtype (Float16/BFloat16/Float8E4M3FN/etc)
        if use_mfma_k32:
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, layout_elem))
            k_perm = fx.make_layout((8, 4), (1, 8))
        elif is_f16_or_bf16:
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, layout_elem))
            k_perm = fx.make_layout((4, 4, 2), (1, 8, 4))
        elif use_mfma_scale_128:
            mma_atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, layout_elem))
            k_perm = fx.make_layout((32, 4), (1, 32))
        else:
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, layout_elem))
            k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))

        tiled_mma = fx.make_tiled_mma(
            mma_atom, fx.make_layout((1, 4, 1), (0, 1, 0)),
            fx.make_tile(None, None, k_perm),
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
