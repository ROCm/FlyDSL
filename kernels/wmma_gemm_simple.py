"""WMMA GEMM kernel for gfx1250."""

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from kernels.layout_utils import crd2idx, idx2crd, get as layout_get

WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
WAVE_SIZE = 32


def compile_wmma_gemm(
    *,
    M: int = 0,
    N: int = 0,
    K: int,
    tile_m: int = 64,
    tile_n: int = 128,
    tile_k: int = WMMA_K,
    in_dtype: str = "fp16",
    block_threads: int = 128,
):
    """Compile a WMMA GEMM kernel using the @flyc.kernel API.

    Returns a JitFunction that auto-compiles and executes when called.
    Signature:  launch_fn(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, M, N, stream)
    
    Compile-time constants: K, tile_m/n/k, in_dtype (determine loop structure).
    Runtime parameters: M, N (passed as i32 kernel args).
    """
    _ = (M, N)
    if in_dtype not in ("fp16", "bf16"):
        raise ValueError(f"in_dtype must be 'fp16' or 'bf16', got {in_dtype!r}")
    is_fp4 = in_dtype == "fp4"
    is_int4 = in_dtype == "int4"
    is_int8 = (in_dtype == "int8") or is_int4
    is_f16 = in_dtype == "fp16"
    is_bf16 = in_dtype == "bf16"
    is_f16_or_bf16 = is_f16 or is_bf16
    elem_bytes = 1 if (in_dtype in ("fp8", "int8", "int4", "fp4")) else 2

    if K % tile_k != 0:
        raise ValueError(f"K must be divisible by tile_k={tile_k}, got K={K}")
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")

    waves_per_block = block_threads // WAVE_SIZE
    if tile_n % (waves_per_block * WMMA_N) != 0:
        raise ValueError(
            f"tile_n must be a multiple of waves_per_block*{WMMA_N}={waves_per_block * WMMA_N}, got {tile_n}"
        )

    gpu_arch = str(get_hip_arch(timeout_s=300))
    assert gpu_arch.startswith("gfx1250"), f"Expected a gfx1250 architecture, got {gpu_arch}"

    wmma_op = rocdl.wmma_f32_16x16x32_f16 if is_f16 else rocdl.wmma_f32_16x16x32_bf16
    k_wmma_steps = tile_k // WMMA_K

    def _elem_type():
        return T.f16 if is_f16 else T.bf16

    warp_tile_n = tile_n // waves_per_block
    wmma_m_rep = tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_accs = wmma_m_rep * wmma_n_rep

    lds_a_elems = tile_m * tile_k
    lds_b_elems = tile_k * tile_n
    lds_a_offset = 0
    lds_b_offset = lds_a_elems * elem_bytes

    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="wmma_gemm_smem")
    allocator.ptr = lds_b_offset + lds_b_elems * elem_bytes

    total_vec_a = tile_m * (tile_k // 4)
    total_vec_b = tile_k * (tile_n // 4)
    if total_vec_a % block_threads != 0 or total_vec_b % block_threads != 0:
        raise ValueError(
            f"vectorized copy requires vec slots divisible by block_threads: "
            f"A={total_vec_a}, B={total_vec_b}, block_threads={block_threads}"
        )
    vec_iters_a = total_vec_a // block_threads
    vec_iters_b = total_vec_b // block_threads

    @flyc.kernel
    def kernel_wmma_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")

        n_stride = arith.index_cast(T.index, i32_n.ir_value())
        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)

        layout_thr = fx.make_layout((waves_per_block, WAVE_SIZE), (WAVE_SIZE, 1))
        layout_lane = fx.make_layout((2, 16), (16, 1))
        layout_lds_a = fx.make_layout((tile_m, tile_k), (tile_k, 1))
        layout_lds_b = fx.make_layout((tile_k, tile_n), (tile_n, 1))
        layout_vec_a = fx.make_layout((tile_m, tile_k // 4), (tile_k // 4, 1))
        layout_vec_b = fx.make_layout((tile_k, tile_n // 4), (tile_n // 4, 1))

        thr = idx2crd(tx, layout_thr)
        wave_id = layout_get(thr, 0)
        lane = layout_get(thr, 1)

        lc = idx2crd(lane, layout_lane)
        lane_kgrp = layout_get(lc, 0)  # 0/1
        lane16 = layout_get(lc, 1)  # 0..15
        warp_n_off = wave_id * arith.index(warp_tile_n)

        elem_ty = _elem_type()
        base_ptr = allocator.get_base()
        lds_a = SmemPtr(base_ptr, lds_a_offset, elem_ty, shape=(lds_a_elems,))
        lds_b = SmemPtr(base_ptr, lds_b_offset, elem_ty, shape=(lds_b_elems,))
        lds_a_mem = lds_a.get()
        lds_b_mem = lds_b.get()

        a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=True)
        vec4_elem_ty = T.vec(4, elem_ty)

        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))
        accs = [acc_zero] * n_accs

        for kblk in range_constexpr(K // tile_k):
            k_base = arith.index(kblk * tile_k)

            for t in range_constexpr(vec_iters_a):
                vec_idx = tx + arith.index(t * block_threads)
                a_crd = idx2crd(vec_idx, layout_vec_a)
                a_m = layout_get(a_crd, 0)
                a_kv = layout_get(a_crd, 1)
                a_k = a_kv * arith.index(4)

                g_off = (blk_m + a_m) * arith.index(K) + (k_base + a_k)
                v_i16 = buffer_ops.buffer_load(a_rsrc, g_off, vec_width=4, dtype=T.i16)
                v = vector.bitcast(vec4_elem_ty, v_i16)
                lds_off = crd2idx((a_m, a_k), layout_lds_a)
                vector.store(v, lds_a_mem, [lds_off])

            for t in range_constexpr(vec_iters_b):
                vec_idx = tx + arith.index(t * block_threads)
                b_crd = idx2crd(vec_idx, layout_vec_b)
                b_k = layout_get(b_crd, 0)
                b_nv = layout_get(b_crd, 1)
                b_n = b_nv * arith.index(4)

                g_off = (k_base + b_k) * n_stride + (blk_n + b_n)
                v_i16 = buffer_ops.buffer_load(b_rsrc, g_off, vec_width=4, dtype=T.i16)
                v = vector.bitcast(vec4_elem_ty, v_i16)
                lds_off = crd2idx((b_k, b_n), layout_lds_b)
                vector.store(v, lds_b_mem, [lds_off])

            gpu.barrier()

            for ks in range_constexpr(k_wmma_steps):
                k_step = arith.index(ks * WMMA_K)

                b_frags = []
                for wn in range_constexpr(wmma_n_rep):
                    n_off = warp_n_off + arith.index(wn * WMMA_N)
                    vals = []
                    for k0 in range_constexpr(2):
                        for k1 in range_constexpr(8):
                            kk = k_step + (arith.index(k0 * 2) + lane_kgrp) * arith.index(8) + arith.index(k1)
                            off = crd2idx((kk, n_off + lane16), layout_lds_b)
                            vals.append(lds_b.load([off]))
                    b_frags.append(vector.from_elements(T.vec(16, elem_ty), vals))

                for wm in range_constexpr(wmma_m_rep):
                    m_off = arith.index(wm * WMMA_M)
                    a_vals = []
                    for k0 in range_constexpr(2):
                        for k1 in range_constexpr(8):
                            kk = k_step + (arith.index(k0 * 2) + lane_kgrp) * arith.index(8) + arith.index(k1)
                            off = crd2idx((m_off + lane16, kk), layout_lds_a)
                            a_vals.append(lds_a.load([off]))
                    a_frag = vector.from_elements(T.vec(16, elem_ty), a_vals)

                    for wn in range_constexpr(wmma_n_rep):
                        acc_idx = wm * wmma_n_rep + wn
                        accs[acc_idx] = wmma_op(
                            T.vec(8, T.f32),
                            a_frag,
                            b_frags[wn],
                            accs[acc_idx],
                            signA=False,
                            signB=False,
                            modC=0,
                            reuseA=False,
                            reuseB=False,
                        ).result

            gpu.barrier()

        for wm in range_constexpr(wmma_m_rep):
            for wn in range_constexpr(wmma_n_rep):
                acc_idx = wm * wmma_n_rep + wn
                m_base = blk_m + arith.index(wm * WMMA_M)
                n_base = blk_n + warp_n_off + arith.index(wn * WMMA_N)
                for mi in range_constexpr(8):
                    row = m_base + lane_kgrp * arith.index(8) + arith.index(mi)
                    col = n_base + lane16
                    c_off = row * n_stride + col
                    c_val = vector.extract(accs[acc_idx], static_position=[mi], dynamic_position=[])
                    fx.memref_store(c_val, arg_c, c_off)

    cache_tag = (in_dtype, K, tile_m, tile_n, tile_k, block_threads)

    @flyc.jit
    def launch_wmma_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        idx_m = arith.index_cast(T.index, i32_m.ir_value())
        idx_n = arith.index_cast(T.index, i32_n.ir_value())
        gx = _raw((idx_m + arith.index(tile_m - 1)) / arith.index(tile_m))
        gy = _raw((idx_n + arith.index(tile_n - 1)) / arith.index(tile_n))

        kernel_wmma_gemm(arg_c, arg_a, arg_b, i32_m, i32_n).launch(
            grid=(gx, gy, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_wmma_gemm


__all__ = ["compile_wmma_gemm"]
