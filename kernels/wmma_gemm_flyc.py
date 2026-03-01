"""WMMA GEMM kernel for gfx1250. """

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir import ir
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


def compile_wmma_gemm(
    *,
    M: int = 0,
    N: int = 0,
    K: int,
    tile_m: int = 16,
    tile_n: int = 16,
    tile_k: int = 32,
    in_dtype: str = "fp16",
    block_threads: int = 32,
):
    """Compile a WMMA GEMM kernel using the @flyc.kernel API.

    Returns a JitFunction that auto-compiles and executes when called.
    Signature:  launch_fn(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, M, N, stream)
    
    Compile-time constants: K, tile_m/n/k, in_dtype (determine loop structure).
    Runtime parameters: M, N (passed as i32 kernel args).
    """
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

    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected a gfx1250 architecture, got {gpu_arch}"

    lds_a_offset = 0
    lds_a_elems = tile_m * tile_k
    lds_b_offset = lds_a_offset + lds_a_elems * elem_bytes
    lds_b_elems = tile_k * tile_n

    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="wmma_gemm_smem")
    allocator.ptr = lds_b_offset + lds_b_elems * elem_bytes

    def _elem_type():
        return T.f16 if is_f16 else T.bf16

    def _wmma_op():
        return rocdl.wmma_f32_16x16x32_f16 if is_f16 else rocdl.wmma_f32_16x16x32_bf16

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

        lane = tx
        n_idx = arith.index_cast(T.index, i32_n.ir_value())

        tile_m_base = bx * arith.index(tile_m)
        tile_n_base = by * arith.index(tile_n)

        base_ptr = allocator.get_base()
        lds_a_ptr = SmemPtr(base_ptr, lds_a_offset, _elem_type(), shape=(lds_a_elems,))
        lds_b_ptr = SmemPtr(base_ptr, lds_b_offset, _elem_type(), shape=(lds_b_elems,))

        c_frag = arith.constant_vector(0.0, T.vec(8, T.f32))

        for kblk in range_constexpr(K // tile_k):
            k_base = arith.index(kblk * tile_k)

            for t in range_constexpr((tile_m * tile_k) // 32):
                idx = lane + arith.index(t * 32)
                lm = idx // arith.index(tile_k)
                lk = idx % arith.index(tile_k)
                g_row = tile_m_base + lm
                g_col = k_base + lk
                g_off = g_row * arith.index(K) + g_col
                v = fx.memref_load(arg_a, g_off)
                l_off = lm * arith.index(tile_k) + lk
                lds_a_ptr.store(v, [l_off])

            for t in range_constexpr((tile_k * tile_n) // 32):
                idx = lane + arith.index(t * 32)
                lk = idx // arith.index(tile_n)
                ln = idx % arith.index(tile_n)
                g_row = k_base + lk
                g_col = tile_n_base + ln
                g_off = g_row * n_idx + g_col
                v = fx.memref_load(arg_b, g_off)
                l_off = lk * arith.index(tile_n) + ln
                lds_b_ptr.store(v, [l_off])

            gpu.barrier()

            lane16 = lane % arith.index(16)
            lane_kgrp = lane // arith.index(16)
            a_vals = []
            b_vals = []
            for k0 in range_constexpr(2):
                for k1 in range_constexpr(8):
                    kk = (arith.index(k0 * 2) + lane_kgrp) * arith.index(8) + arith.index(k1)
                    a_off = lane16 * arith.index(tile_k) + kk
                    b_off = kk * arith.index(tile_n) + lane16
                    a_vals.append(lds_a_ptr.load([a_off]))
                    b_vals.append(lds_b_ptr.load([b_off]))

            a_frag = fx.vector.from_elements(T.vec(16, _elem_type()), a_vals)
            b_frag = fx.vector.from_elements(T.vec(16, _elem_type()), b_vals)

            c_frag = _wmma_op()(
                T.vec(8, T.f32),
                a_frag,
                b_frag,
                c_frag,
                signA=False,
                signB=False,
                modC=0,
                reuseA=False,
                reuseB=False,
            ).result

            gpu.barrier()

        lane_n = lane % arith.index(16)
        lane_mgrp = lane // arith.index(16)
        for mi in range_constexpr(8):
            row = tile_m_base + lane_mgrp * arith.index(8) + arith.index(mi)
            col = tile_n_base + lane_n
            c_off = row * n_idx + col
            c_val = fx.vector.extract(c_frag, static_position=[mi], dynamic_position=[])
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
