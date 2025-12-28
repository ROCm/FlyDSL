#!/usr/bin/env python3
"""
MFMA FP8 GEMM Test using Flir with B preshuffle.

- When M < 32: use the m16-specialized implementation (dynamic waves + double-buffer LDS pipeline).
- When M >= 32: use the m1024 implementation (fixed 256 threads + K32 micro-step with on-the-fly B loads).
"""

import os
import sys
import logging
import ctypes

logging.basicConfig(level=logging.INFO)

import pyflir
from pyflir.runtime.device import get_rocm_arch
import pyflir.dialects.ext.flir as flir
from pyflir.dialects.ext.python_control_flow import range_constexpr
from pyflir.utils import SmemAllocator, SmemPtr
from tests.utils import compile_to_hsaco, pertoken_quant, shuffle_weight
from tests.test_common import verify_output, run_perftest
import torch
import torch.nn.functional as F
import pytest
from _mlir import ir
from _mlir.dialects import vector
from pyflir.dialects.ext import arith, gpu, buffer_ops, memref
from _mlir.dialects import arith as _arith_mlir
import _mlir.dialects.rocdl as rocdl
import _mlir.extras.types as T

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

try:
    import aiter

    HAS_AITER = True
except ImportError:
    print("Warning: Aiter not found, skipping comparison")
    HAS_AITER = False

RUN_AITER_BENCH = os.environ.get("FLIR_RUN_AITER_BENCH", "0") == "1"


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    """
    Torch reference implementation (from aiter project).
    Dequantize FP8 inputs and compute FP32 matmul.
    """
    x = x.to(torch.float32) * x_scale
    weight = weight.to(torch.float32) * w_scale
    out = F.linear(x, weight)
    if bias is not None:
        out = out.to(bias.dtype) + bias
    return out.to(dtype)


def unwrap(v):
    if isinstance(v, int):
        return arith.constant(v, index=True).value
    while hasattr(v, "value") or hasattr(v, "_value"):
        if hasattr(v, "_value"):
            v = v._value
        elif hasattr(v, "value"):
            v = v.value
    return v


def _hip_check(result):
    """Check HIP API call result and raise exception on error (minimal local helper)."""
    from hip import hip  # type: ignore

    if isinstance(result, tuple):
        err = result[0]
        ret_val = result[1] if len(result) == 2 else result[1:]
    else:
        err = result
        ret_val = None

    if err != hip.hipError_t.hipSuccess:
        try:
            error_str = hip.hipGetErrorString(err)
        except Exception:
            error_str = str(err)
        raise RuntimeError(f"HIP Error: {error_str}")
    return ret_val


def _pad_fp8_storage(fp8_tensor: torch.Tensor, pad_elems: int = 64) -> torch.Tensor:
    if fp8_tensor.is_contiguous():
        flat = fp8_tensor.view(-1)
    else:
        flat = fp8_tensor.contiguous().view(-1)
    storage = torch.empty(
        flat.numel() + pad_elems, device=fp8_tensor.device, dtype=fp8_tensor.dtype
    )
    storage[: flat.numel()] = flat
    return storage[: flat.numel()].view(fp8_tensor.shape)


def _prepare_inputs(M: int, N: int, K: int, device: torch.device):
    torch.manual_seed(42)
    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.randn(N, K, device=device, dtype=torch.float32)  # (N, K) weight

    a_q_fp8, scale_a = pertoken_quant(a_fp32, quant_dtype=torch.float8_e4m3fnuz)
    b_q_fp8, scale_b = pertoken_quant(b_fp32_t, quant_dtype=torch.float8_e4m3fnuz)

    a_q_fp8 = _pad_fp8_storage(a_q_fp8)
    b_q_fp8 = _pad_fp8_storage(b_q_fp8)

    b_shuffled = shuffle_weight(b_q_fp8)
    c_ref = run_torch(
        a_q_fp8, b_q_fp8, scale_a, scale_b, bias=None, dtype=torch.float32
    )
    return a_q_fp8, b_q_fp8, b_shuffled, scale_a, scale_b, c_ref


def _maybe_run_aiter(
    a_q_fp8, b_shuffled, scale_a, scale_b, c_ref, flops, bytes_moved, us_flir
):
    if not HAS_AITER:
        return
    if not RUN_AITER_BENCH:
        print("-" * 40)
        print("Skipping Aiter benchmark (set FLIR_RUN_AITER_BENCH=1 to enable)")
        print("-" * 40)
        return

    print("-" * 40)
    print("Running Aiter Benchmark...")

    def launch_aiter(a, b, sa, sb):
        return aiter.gemm_a8w8_bpreshuffle(a, b, sa, sb, None, torch.float16)

    c_aiter, us1 = run_perftest(launch_aiter, a_q_fp8, b_shuffled, scale_a, scale_b)
    verify_output(c_aiter.to(torch.float32), c_ref, rtol=0.1, atol=0.1)

    tflops_aiter = flops / (us1 / 1e6) / 1e12
    bw_aiter = bytes_moved / 1e9 / (us1 / 1e6)
    tflops_flir = flops / (us_flir / 1e6) / 1e12
    print(
        f"Aiter Throughput: {us1:.1f} us, {tflops_aiter:.2f} TFLOPS, BW: {bw_aiter:.2f} GB/s"
    )
    print(
        f"Speedup vs Aiter: {tflops_flir / tflops_aiter:.2f}x, us {us1:.1f} vs {us_flir:.1f}"
    )
    print("-" * 40)


def _run_impl_m16(M, N, K, tile_m, tile_n, tile_k):
    """
    m16 specialization: dynamic waves (based on tile_n) + double-buffered LDS pipeline.
    Kernel body is lifted from test_mfma_gemm_fp8_rocir_preshuffle_m16.py with minimal edits.
    """
    gpu_arch = get_rocm_arch()

    def _f8():
        return ir.Float8E4M3FNType.get()

    def _f32():
        return ir.F32Type.get()

    size_c = M * N
    size_a = M * K
    size_b = N * K

    # Dynamic waves based on tile_n
    num_waves = 2
    assert (
        tile_n % num_waves == 0
    ), f"tile_n({tile_n}) must be divisible by num_waves({num_waves})"
    assert (tile_n // num_waves) % 16 == 0, "n_per_wave must be a multiple of 16"
    total_threads = num_waves * 64
    elems_a_per_tile = tile_m * tile_k
    elems_per_thread_a = elems_a_per_tile // total_threads
    bytes_per_thread_a = elems_per_thread_a

    pad_k = 0
    lds_stride = tile_k + pad_k

    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    class _MFMA_M16(flir.MlirModule):
        GPU_MODULE_NAME = "mfma_mod_m16"
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            _state["lds_a_decl"] = allocator.allocate_array(
                _f8(), 2 * tile_m * lds_stride
            )
            allocator.finalize()

        @flir.kernel
        def kernel_fixed(
            self: flir.T.i64,
            arg_c: lambda: T.memref(size_c, T.f16()),
            arg_a: lambda: T.memref(size_a, _f8()),
            arg_b: lambda: T.memref(size_b, _f8()),
            arg_scale_a: lambda: T.memref(M, T.f32()),
            arg_scale_b: lambda: T.memref(N, T.f32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            f8 = _f8()
            f32 = _f32()
            c_m = m_in
            c_n = n_in
            c_k = k_in
            c0 = arith.constant(0, index=True)
            c1 = arith.constant(1, index=True)
            c_tile_k = arith.constant(tile_k, index=True)

            i32_type = ir.IntegerType.get_signless(32)
            index_type = ir.IndexType.get()
            vec4_f32 = ir.VectorType.get([4], f32)
            vec8_f8 = ir.VectorType.get([8], f8)
            vec16_f8 = ir.VectorType.get([16], f8)
            vec2_i64 = ir.VectorType.get([2], ir.IntegerType.get_signless(64))
            vec4_i32 = ir.VectorType.get([4], i32_type)

            vec_a_load_len = bytes_per_thread_a

            # Use low-level ConstantOp for index constants (avoids std::bad_cast in some envs).
            def const_index(i: int):
                return _arith_mlir.ConstantOp(
                    index_type, ir.IntegerAttr.get(index_type, i)
                ).result

            zero_attr = ir.DenseElementsAttr.get_splat(
                vec4_f32, ir.FloatAttr.get(f32, 0.0)
            )
            acc_init = _arith_mlir.ConstantOp(vec4_f32, zero_attr).result

            layout_a = flir.make_layout((c_m, c_k), stride=(c_k, 1))
            layout_c = flir.make_layout((c_m, c_n), stride=(c_n, 1))

            c0_i32 = arith.i32(0)
            c4 = arith.constant(4, index=True)
            c8 = arith.constant(8, index=True)
            c16 = arith.constant(16, index=True)
            c64 = arith.constant(64, index=True)
            c1024 = arith.constant(1024, index=True)

            def swizzle_idx(row_idx, col_idx):
                c16 = arith.constant(16, index=True)
                k_blocks16 = arith.constant(tile_k // 16, index=True)
                row_mod = _arith_mlir.RemUIOp(
                    unwrap(row_idx), unwrap(k_blocks16)
                ).result
                xor_mask = _arith_mlir.MulIOp(unwrap(row_mod), unwrap(c16)).result
                return _arith_mlir.XOrIOp(unwrap(col_idx), unwrap(xor_mask)).result

            c_k0 = _arith_mlir.DivUIOp(unwrap(c_k), unwrap(c64)).result
            stride_n0 = _arith_mlir.MulIOp(unwrap(c_k0), unwrap(c1024)).result
            stride_b = (
                stride_n0,
                c1024,
                arith.constant(256, index=True),
                c16,
                arith.constant(1, index=True),
            )
            layout_b = flir.make_layout(
                (
                    _arith_mlir.DivUIOp(unwrap(c_n), unwrap(c16)).result,
                    c_k0,
                    arith.constant(4, index=True),
                    c16,
                    c16,
                ),
                stride=stride_b,
            )

            shape_lds = flir.make_shape(tile_m, tile_k)
            stride_lds = flir.make_stride(lds_stride, 1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            by = gpu.block_id("y")

            base_ptr = allocator.get_base()
            lds_a_ptr = _state["lds_a_decl"](base_ptr)
            lds_a = lds_a_ptr.get()

            a_rsrc = buffer_ops.create_buffer_resource(arg_a)
            b_rsrc = buffer_ops.create_buffer_resource(arg_b)
            c_rsrc = buffer_ops.create_buffer_resource(arg_c)
            scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a)
            scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b)

            tx_idx = unwrap(tx)
            vec_len_val = arith.constant(vec_a_load_len, index=True)
            linear_id = _arith_mlir.MulIOp(unwrap(tx_idx), unwrap(vec_len_val)).result

            c_tile_k_val = arith.constant(tile_k, index=True)
            row_a_local = _arith_mlir.DivUIOp(
                unwrap(linear_id), unwrap(c_tile_k_val)
            ).result
            col_a_local = _arith_mlir.RemUIOp(
                unwrap(linear_id), unwrap(c_tile_k_val)
            ).result

            bx_m = _arith_mlir.MulIOp(
                unwrap(bx), unwrap(arith.constant(tile_m, index=True))
            ).result
            row_a_global = _arith_mlir.AddIOp(unwrap(bx_m), unwrap(row_a_local)).result
            by_n = _arith_mlir.MulIOp(
                unwrap(by), unwrap(arith.constant(tile_n, index=True))
            ).result

            wave_id = tx / 64
            lane_id = tx % 64
            lane_mod_16 = lane_id % 16
            lane_div_16 = lane_id / 16

            row_a_lds = lane_mod_16
            col_offset_base = _arith_mlir.MulIOp(
                unwrap(lane_div_16), unwrap(c16)
            ).result
            row_b_lds = lane_mod_16

            coord_a_base = flir.make_coord(unwrap(row_a_global), unwrap(col_a_local))
            idx_a_base = flir.crd2idx(coord_a_base, layout_a)
            idx_a_base_div4 = _arith_mlir.DivUIOp(unwrap(idx_a_base), unwrap(c4)).result

            m_repeat = tile_m // 16
            k_unroll = tile_k // 64

            # --- Dynamic tiling ---
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16
            c_n_per_wave = arith.constant(n_per_wave, index=True)
            c_num_waves = arith.constant(num_waves, index=True)
            wave_mod = _arith_mlir.RemUIOp(unwrap(wave_id), unwrap(c_num_waves)).result
            n_tile_base = _arith_mlir.MulIOp(
                unwrap(wave_mod), unwrap(c_n_per_wave)
            ).result

            n_intra_list = []
            n_blk_list = []
            for i in range_constexpr(num_acc_n):
                offset = i * 16
                c_offset = const_index(offset)
                tmp1 = _arith_mlir.AddIOp(unwrap(by_n), unwrap(n_tile_base)).result
                tmp2 = _arith_mlir.AddIOp(unwrap(tmp1), unwrap(c_offset)).result
                global_n = _arith_mlir.AddIOp(unwrap(tmp2), unwrap(row_b_lds)).result
                n_intra = _arith_mlir.RemUIOp(unwrap(global_n), unwrap(c16)).result
                n_blk = _arith_mlir.DivUIOp(unwrap(global_n), unwrap(c16)).result
                n_intra_list.append(n_intra)
                n_blk_list.append(n_blk)

            acc_inits = [acc_init] * (num_acc_n * m_repeat)

            # --- B load ---
            def load_b_tile(base_k):
                b_packs = []
                k0_base = _arith_mlir.DivUIOp(unwrap(base_k), unwrap(c64)).result
                k1 = lane_div_16
                k2_base = c0
                vec2_i64 = ir.VectorType.get([2], ir.IntegerType.get_signless(64))
                for i in range_constexpr(num_acc_n):
                    n_intra = n_intra_list[i]
                    n_blk = n_blk_list[i]
                    for ki_step in range_constexpr(k_unroll):
                        k0 = _arith_mlir.AddIOp(
                            unwrap(k0_base), unwrap(arith.constant(ki_step, index=True))
                        ).result
                        coord_b = flir.make_coord(n_blk, k0, k1, n_intra, k2_base)
                        idx_bytes = flir.crd2idx(coord_b, layout_b)
                        idx_i32 = _arith_mlir.DivUIOp(
                            unwrap(idx_bytes), unwrap(c4)
                        ).result
                        b16 = buffer_ops.buffer_load(
                            b_rsrc, idx_i32, vec_width=4, dtype=i32_type
                        )
                        b_vec128 = vector.BitCastOp(vec2_i64, b16).result
                        b0_pack = unwrap(
                            vector.ExtractOp(
                                b_vec128, static_position=[0], dynamic_position=[]
                            ).result
                        )
                        b1_pack = unwrap(
                            vector.ExtractOp(
                                b_vec128, static_position=[1], dynamic_position=[]
                            ).result
                        )
                        b_packs.append(b0_pack)
                        b_packs.append(b1_pack)
                return b_packs

            # Split A loads logic
            max_bytes_per_load = 16
            num_a_loads = (
                bytes_per_thread_a + max_bytes_per_load - 1
            ) // max_bytes_per_load
            vec_a_parts_lens = []
            remaining_bytes = bytes_per_thread_a
            for _ in range_constexpr(num_a_loads):
                curr_bytes = min(remaining_bytes, max_bytes_per_load)
                vec_a_parts_lens.append(curr_bytes)
                remaining_bytes -= curr_bytes

            def load_a_split(idx_div4):
                parts = []
                curr_off_i32 = 0
                for i in range_constexpr(num_a_loads):
                    curr_bytes = vec_a_parts_lens[i]
                    curr_idx = idx_div4
                    if curr_off_i32 > 0:
                        curr_idx = _arith_mlir.AddIOp(
                            unwrap(idx_div4),
                            unwrap(arith.constant(curr_off_i32, index=True)),
                        ).result
                    val = buffer_ops.buffer_load(
                        a_rsrc, curr_idx, vec_width=4, dtype=i32_type
                    )
                    parts.append(val)
                    curr_off_i32 += curr_bytes // 4
                return parts

            def store_a_to_lds(vec_a_in_parts, buffer_idx):
                buffer_offset_val = _arith_mlir.MulIOp(
                    unwrap(buffer_idx),
                    unwrap(arith.constant(tile_m * lds_stride, index=True)),
                ).result
                curr_store_off = 0
                for i in range_constexpr(num_a_loads):
                    val_vec = vec_a_in_parts[i]
                    curr_bytes = vec_a_parts_lens[i]
                    curr_i32 = curr_bytes // 4

                    col_0 = _arith_mlir.AddIOp(
                        unwrap(col_a_local),
                        unwrap(arith.constant(curr_store_off, index=True)),
                    ).result
                    col_swizzled_0 = swizzle_idx(row_a_local, col_0)
                    coord_store_0 = flir.make_coord(
                        unwrap(row_a_local), unwrap(col_swizzled_0)
                    )
                    idx_0 = flir.crd2idx(coord_store_0, layout_lds)
                    idx_0_final = _arith_mlir.AddIOp(
                        unwrap(idx_0), unwrap(buffer_offset_val)
                    ).result

                    if curr_i32 == 4:
                        val_16 = vector.BitCastOp(vec16_f8, val_vec).result
                        vector.StoreOp(val_16, lds_a, [unwrap(idx_0_final)])
                    elif curr_i32 == 2:
                        val_2_i32 = vector.ShuffleOp(val_vec, val_vec, [0, 1]).result
                        val_8 = vector.BitCastOp(vec8_f8, val_2_i32).result
                        vector.StoreOp(val_8, lds_a, [unwrap(idx_0_final)])
                    else:
                        vec_f8 = ir.VectorType.get([curr_bytes], f8)
                        if curr_bytes <= 4:
                            val_1_i32 = vector.ShuffleOp(val_vec, val_vec, [0]).result
                            val_f8 = vector.BitCastOp(vec_f8, val_1_i32).result
                        else:
                            val_2_i32 = vector.ShuffleOp(
                                val_vec, val_vec, [0, 1]
                            ).result
                            val_f8 = vector.BitCastOp(vec_f8, val_2_i32).result
                        vector.StoreOp(val_f8, lds_a, [unwrap(idx_0_final)])
                    curr_store_off += curr_bytes

            # Prolog: tile0 into buffer0
            vec_a_cur = load_a_split(idx_a_base_div4)
            b_vals_cur = load_b_tile(c0)
            store_a_to_lds(vec_a_cur, c0)
            gpu.barrier()

            assert K % tile_k == 0, f"K({K}) must be divisible by tile_k({tile_k})"
            num_tiles = K // tile_k
            assert num_tiles >= 1

            if num_tiles > 1:
                k1 = arith.constant(tile_k, index=True)
                k1_div4 = _arith_mlir.DivUIOp(unwrap(k1), unwrap(c4)).result
                idx_a_1_div4 = _arith_mlir.AddIOp(
                    unwrap(idx_a_base_div4), unwrap(k1_div4)
                ).result
                vec_a_next = load_a_split(idx_a_1_div4)
                b_vals_next = load_b_tile(k1)
            else:
                vec_a_next = vec_a_cur
                b_vals_next = b_vals_cur

            accs = list(acc_inits)
            buf_read = c0

            def mfma_tile(accs_in, b_vals_in, buffer_read):
                current_accs_list = list(accs_in)
                buffer_offset_val = _arith_mlir.MulIOp(
                    unwrap(buffer_read),
                    unwrap(arith.constant(tile_m * lds_stride, index=True)),
                ).result
                for ki_step in range_constexpr(k_unroll):
                    b0_packs = []
                    b1_packs = []
                    for ni in range_constexpr(num_acc_n):
                        base = ni * (2 * k_unroll)
                        b0_packs.append(b_vals_in[base + (2 * ki_step + 0)])
                        b1_packs.append(b_vals_in[base + (2 * ki_step + 1)])

                    ki = ki_step * 64
                    ki_val = arith.constant(ki, index=True)
                    col_lds0 = _arith_mlir.AddIOp(
                        unwrap(col_offset_base), unwrap(ki_val)
                    ).result
                    for mi in range_constexpr(m_repeat):
                        mi_val = arith.constant(mi * 16, index=True)
                        curr_row_a_lds = _arith_mlir.AddIOp(
                            unwrap(row_a_lds), unwrap(mi_val)
                        ).result
                        col_lds0_swizzled = swizzle_idx(curr_row_a_lds, col_lds0)
                        coord_a0 = flir.make_coord(
                            unwrap(curr_row_a_lds), unwrap(col_lds0_swizzled)
                        )
                        idx_a0 = flir.crd2idx(coord_a0, layout_lds)
                        idx_a0 = _arith_mlir.AddIOp(
                            unwrap(idx_a0), unwrap(buffer_offset_val)
                        ).result
                        idx_a0_idx = unwrap(idx_a0)
                        loaded_a16 = vector.LoadOp(
                            vec16_f8, lds_a, [unwrap(idx_a0_idx)]
                        ).result
                        a_vec128 = vector.BitCastOp(vec2_i64, loaded_a16).result
                        a0_pack = vector.ExtractOp(
                            a_vec128, static_position=[0], dynamic_position=[]
                        ).result
                        a1_pack = vector.ExtractOp(
                            a_vec128, static_position=[1], dynamic_position=[]
                        ).result
                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            curr_acc = current_accs_list[acc_idx]
                            b0_pack = b0_packs[ni]
                            b1_pack = b1_packs[ni]
                            acc0 = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                vec4_f32,
                                [
                                    unwrap(a0_pack),
                                    unwrap(b0_pack),
                                    unwrap(curr_acc),
                                    unwrap(c0_i32),
                                    unwrap(c0_i32),
                                    unwrap(c0_i32),
                                ],
                            ).result
                            acc1 = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                vec4_f32,
                                [
                                    unwrap(a1_pack),
                                    unwrap(b1_pack),
                                    unwrap(acc0),
                                    unwrap(c0_i32),
                                    unwrap(c0_i32),
                                    unwrap(c0_i32),
                                ],
                            ).result
                            current_accs_list[acc_idx] = acc1
                return current_accs_list

            for t in range_constexpr(num_tiles - 1):
                accs = mfma_tile(accs, b_vals_cur, buf_read)
                buf_write = _arith_mlir.XOrIOp(unwrap(buf_read), unwrap(c1)).result
                store_a_to_lds(vec_a_next, buf_write)
                gpu.barrier()
                buf_read = buf_write
                b_vals_cur = b_vals_next
                if (t + 2) < num_tiles:
                    k2 = arith.constant((t + 2) * tile_k, index=True)
                    k2_div4 = _arith_mlir.DivUIOp(unwrap(k2), unwrap(c4)).result
                    idx_a_2_div4 = _arith_mlir.AddIOp(
                        unwrap(idx_a_base_div4), unwrap(k2_div4)
                    ).result
                    vec_a_next = load_a_split(idx_a_2_div4)
                    b_vals_next = load_b_tile(k2)

            accs = mfma_tile(accs, b_vals_cur, buf_read)

            # Prefetch scales
            s_b_vals = []
            for ni in range_constexpr(num_acc_n):
                offset = ni * 16
                c_offset = arith.constant(offset, index=True)
                tmp1 = _arith_mlir.AddIOp(unwrap(by_n), unwrap(n_tile_base)).result
                tmp2 = _arith_mlir.AddIOp(unwrap(tmp1), unwrap(c_offset)).result
                col_g = _arith_mlir.AddIOp(unwrap(tmp2), unwrap(lane_mod_16)).result
                val = buffer_ops.buffer_load(
                    scale_b_rsrc, col_g, vec_width=1, dtype=f32
                )
                s_b_vals.append(val)

            s_a_vecs = []
            row_off_base = lane_div_16 * 4
            for mi in range_constexpr(m_repeat):
                row_base_m = _arith_mlir.AddIOp(
                    unwrap(bx_m), unwrap(arith.constant(mi * 16, index=True))
                ).result
                row_g_base = _arith_mlir.AddIOp(
                    unwrap(row_base_m), unwrap(row_off_base)
                ).result
                s_a_vec = buffer_ops.buffer_load(
                    scale_a_rsrc, row_g_base, vec_width=4, dtype=f32
                )
                s_a_vec4 = vector.BitCastOp(vec4_f32, s_a_vec).result
                s_a_vecs.append(s_a_vec4)

            for mi in range_constexpr(m_repeat):
                row_base_m = _arith_mlir.AddIOp(
                    unwrap(bx_m), unwrap(arith.constant(mi * 16, index=True))
                ).result
                s_a_vec4 = s_a_vecs[mi]
                for i in range_constexpr(4):
                    row_off = (lane_div_16 * 4) + i
                    row_g = _arith_mlir.AddIOp(
                        unwrap(row_base_m), unwrap(row_off)
                    ).result
                    s_a = vector.ExtractOp(
                        s_a_vec4, static_position=[i], dynamic_position=[]
                    ).result
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        acc = accs[acc_idx]
                        val = vector.ExtractOp(acc, [], [i]).result
                        offset = ni * 16
                        c_offset = arith.constant(offset, index=True)
                        tmp1 = _arith_mlir.AddIOp(
                            unwrap(by_n), unwrap(n_tile_base)
                        ).result
                        tmp2 = _arith_mlir.AddIOp(unwrap(tmp1), unwrap(c_offset)).result
                        col_g = _arith_mlir.AddIOp(
                            unwrap(tmp2), unwrap(lane_mod_16)
                        ).result
                        s_b = s_b_vals[ni]
                        val_s = _arith_mlir.MulFOp(unwrap(val), unwrap(s_a)).result
                        val_s = _arith_mlir.MulFOp(unwrap(val_s), unwrap(s_b)).result
                        val_f16 = _arith_mlir.TruncFOp(T.f16(), unwrap(val_s)).result
                        idx = flir.crd2idx(
                            flir.make_coord(unwrap(row_g), unwrap(col_g)), layout_c
                        )
                        buffer_ops.buffer_store(val_f16, c_rsrc, idx)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_c: lambda: T.memref(size_c, T.f16()),
            arg_a: lambda: T.memref(size_a, _f8()),
            arg_b: lambda: T.memref(size_b, _f8()),
            arg_scale_a: lambda: T.memref(M, T.f32()),
            arg_scale_b: lambda: T.memref(N, T.f32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            c1 = arith.constant(1, index=True).value
            bdx = arith.constant(total_threads, index=True).value
            gx = arith.constant(M // tile_m, index=True).value
            gy = arith.constant(N // tile_n, index=True).value
            flir.gpu_ext.LaunchFuncOp(
                ["mfma_mod_m16", "kernel_fixed"],
                grid_size=(gx, gy, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[
                    unwrap(arg_c),
                    unwrap(arg_a),
                    unwrap(arg_b),
                    unwrap(arg_scale_a),
                    unwrap(arg_scale_b),
                    unwrap(m_in),
                    unwrap(n_in),
                    unwrap(k_in),
                ],
            )

    m = _MFMA_M16()
    # Optional: bypass ExecutionEngine host stub and launch via HIP driver API.
    use_hip_driver = os.environ.get("FLIR_USE_HIP_DRIVER", "0") == "1"
    exe = None
    kernel_func = None
    if use_hip_driver:
        try:
            from hip import hip  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "FLIR_USE_HIP_DRIVER=1 requested but `hip` python package is not importable."
            ) from e

        hsaco = compile_to_hsaco(
            m.module,
            kernel_name=f"mfma_m16_t{tile_m}x{tile_n}x{tile_k}_w{num_waves}_driver",
            waves_per_eu=2,
        )
        hip_module = _hip_check(hip.hipModuleLoadData(hsaco))
        kernel_func = _hip_check(hip.hipModuleGetFunction(hip_module, b"kernel_fixed"))
        print("✓ Compiled (m16 hsaco + hip driver)")
    else:
        exe = pyflir.compile(m)
        print("✓ Compiled (m16 impl)")

    device = torch.device("cuda")
    a_q_fp8, b_q_fp8, b_shuffled, scale_a, scale_b, c_ref = _prepare_inputs(
        M, N, K, device
    )
    c_out_raw = torch.zeros((M, N), dtype=torch.float16, device=device)

    def launch_kernel(c, a, b, sa, sb):
        if use_hip_driver:
            assert kernel_func is not None
            from hip import hip  # type: ignore

            arg_ptrs = [
                ctypes.c_void_p(int(c.data_ptr())),
                ctypes.c_void_p(int(a.data_ptr())),
                ctypes.c_void_p(int(b.data_ptr())),
                ctypes.c_void_p(int(sa.data_ptr())),
                ctypes.c_void_p(int(sb.data_ptr())),
                ctypes.c_long(int(M)),
                ctypes.c_long(int(N)),
                ctypes.c_long(int(K)),
            ]
            args_array = (ctypes.c_void_p * len(arg_ptrs))(
                *[ctypes.addressof(p) for p in arg_ptrs]
            )
            _hip_check(
                hip.hipModuleLaunchKernel(
                    kernel_func,
                    M // tile_m,
                    N // tile_n,
                    1,
                    total_threads,
                    1,
                    1,
                    0,
                    0,
                    args_array,
                    None,
                )
            )
        else:
            assert exe is not None
            exe(c, a, b, sa, sb, M, N, K)

    _, us = run_perftest(
        launch_kernel, c_out_raw, a_q_fp8, b_shuffled, scale_a, scale_b
    )
    c_out_scaled = c_out_raw.to(torch.float32)
    assert verify_output(c_out_scaled, c_ref, rtol=0.1, atol=0.1)

    size_c = M * N
    size_a = M * K
    size_b = N * K
    bytes_moved = size_a + size_b + size_c * 2 + (M + N) * 4
    flops = 2 * M * N * K
    tflops = flops / (us / 1e6) / 1e12
    bw = bytes_moved / 1e9 / (us / 1e6)
    print(f"Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {bw:.2f} GB/s")
    _maybe_run_aiter(
        a_q_fp8, b_shuffled, scale_a, scale_b, c_ref, flops, bytes_moved, us
    )


def _run_impl_m1024(M, N, K, tile_m, tile_n, tile_k):
    """
    m1024 implementation: fixed 256 threads + K32 micro-step with on-the-fly B loads.
    Kernel body is lifted from test_mfma_gemm_fp8_rocir_preshuffle_m1024.py with minimal edits.
    """
    gpu_arch = get_rocm_arch()

    def _f8():
        return ir.Float8E4M3FNType.get()

    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    size_c = M * N
    size_a = M * K
    size_b = N * K

    total_threads = 256
    elems_a_per_tile = tile_m * tile_k
    elems_per_thread_a = elems_a_per_tile // total_threads
    bytes_per_thread_a = elems_per_thread_a

    # Make LDS row-stride 16B-aligned so LDS vector loads can be 16B-aligned
    # and AMDGPU can select `ds_read_b128` (instead of splitting/using read2).
    pad_k = 16
    # Make LDS row-stride 16B-aligned so LDS vector loads can be 16B-aligned
    # and AMDGPU can select `ds_read_b128` (instead of splitting/using read2).
    pad_k = 16
    lds_stride = tile_k + pad_k

    class _MFMA_M1024(flir.MlirModule):
        GPU_MODULE_NAME = "mfma_mod_m1024"
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            # Double-buffered LDS A (ping-pong) to support pipelined k-tiles.
            _state["lds_a_decl"] = allocator.allocate_array(
                _f8(), 2 * tile_m * lds_stride
            )
            allocator.finalize()

        @flir.kernel
        def kernel_fixed(
            self: flir.T.i64,
            arg_c: lambda: T.memref(size_c, T.f16()),
            arg_a: lambda: T.memref(size_a, _f8()),
            arg_b: lambda: T.memref(size_b, _f8()),
            arg_scale_a: lambda: T.memref(M, T.f32()),
            arg_scale_b: lambda: T.memref(N, T.f32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            f8 = _f8()
            f32 = ir.F32Type.get()

            c_m = m_in
            c_n = n_in
            c_k = k_in
            c0 = arith.constant(0, index=True)
            c_tile_k = arith.constant(tile_k, index=True)

            i32_type = ir.IntegerType.get_signless(32)
            vec4_f32 = ir.VectorType.get([4], f32)
            vec8_f8 = ir.VectorType.get([8], f8)
            vec16_f8 = ir.VectorType.get([16], f8)
            vec1_i64 = ir.VectorType.get([1], ir.IntegerType.get_signless(64))
            vec2_i64 = ir.VectorType.get([2], ir.IntegerType.get_signless(64))
            vec4_i32 = ir.VectorType.get([4], i32_type)

            vec_a_load_len = bytes_per_thread_a

            zero_attr = ir.DenseElementsAttr.get_splat(
                vec4_f32, ir.FloatAttr.get(f32, 0.0)
            )
            acc_init = _arith_mlir.ConstantOp(vec4_f32, zero_attr).result

            layout_a = flir.make_layout((c_m, c_k), stride=(c_k, 1))
            layout_c = flir.make_layout((c_m, c_n), stride=(c_n, 1))

            c0_i32 = arith.i32(0)
            c1 = arith.constant(1, index=True)
            c4 = arith.constant(4, index=True)
            c16 = arith.constant(16, index=True)
            c256 = arith.constant(256, index=True)
            c1024 = arith.constant(1024, index=True)

            c_k0 = c_k / 64
            c_n0 = c_n / 16
            stride_n0 = c_k0 * 1024

            stride_b = (stride_n0, c1024, c256, c16, c1)
            layout_b = flir.make_layout((c_n0, c_k0, c4, c16, c16), stride=stride_b)

            shape_lds = flir.make_shape(tile_m, tile_k)
            stride_lds = flir.make_stride(lds_stride, 1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            by = gpu.block_id("y")

            base_ptr = allocator.get_base()
            lds_a_ptr = _state["lds_a_decl"](base_ptr)
            lds_a = lds_a_ptr.get()

            # A second view of the same LDS buffer, but typed as 16B blocks.
            # This makes the addressing naturally 16B-granular, allowing LLVM to
            # attach a 16-byte alignment to the underlying `llvm.load` and enabling
            # AMDGPU to select `ds_read_b128` instead of `ds_read2_b64`.
            lds_a16 = SmemPtr(
                lds_a_ptr.base_memref,
                lds_a_ptr.byte_offset,
                vec16_f8,
                shape=((2 * tile_m * lds_stride) // 16,),
            ).get()

            a_rsrc = buffer_ops.create_buffer_resource(arg_a)
            b_rsrc = buffer_ops.create_buffer_resource(arg_b)
            c_rsrc = buffer_ops.create_buffer_resource(arg_c)
            scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a)
            scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b)

            tx_idx = unwrap(tx)
            vec_len_val = arith.constant(vec_a_load_len, index=True)
            linear_id = tx_idx * vec_len_val

            c_tile_k_val = arith.constant(tile_k, index=True)
            row_a_local = linear_id / c_tile_k_val
            col_a_local = linear_id % c_tile_k_val

            bx_m = bx * tile_m
            row_a_global = bx_m + row_a_local
            by_n = by * tile_n

            wave_id = tx / 64
            lane_id = tx % 64
            lane_mod_16 = lane_id % 16
            lane_div_16 = lane_id / 16

            row_a_lds = lane_mod_16
            col_offset_base = lane_div_16 * 16
            row_b_lds = lane_mod_16

            coord_a_base = flir.make_coord(unwrap(row_a_global), unwrap(col_a_local))
            idx_a_base = flir.crd2idx(coord_a_base, layout_a)
            idx_a_base_div4 = idx_a_base / 4

            m_repeat = tile_m // 16
            k_unroll = tile_k // 32

            num_waves = 4
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16

            c_n_per_wave = arith.constant(n_per_wave, index=True)
            wave_mod_4 = wave_id % 4
            n_tile_base = wave_mod_4 * c_n_per_wave

            n_intra_list = []
            n_blk_list = []
            for i in range_constexpr(num_acc_n):
                offset = i * 16
                c_offset = arith.constant(offset, index=True)
                global_n = by_n + n_tile_base + c_offset + row_b_lds
                n_intra = global_n % 16
                n_blk = global_n / 16
                n_intra_list.append(n_intra)
                n_blk_list.append(n_blk)

            acc_inits = [acc_init] * (num_acc_n * m_repeat)

            def load_b_pack(base_k, ki_step, ni):
                k0_base = base_k / 64
                k0 = k0_base + (ki_step // 2)
                k1 = lane_div_16
                half = ki_step % 2
                k2_base = arith.constant(half * 8, index=True)

                n_intra = n_intra_list[ni]
                n_blk = n_blk_list[ni]
                coord_b = flir.make_coord(n_blk, k0, k1, n_intra, k2_base)
                idx_bytes = flir.crd2idx(coord_b, layout_b)
                idx_i32 = idx_bytes / 4
                b8 = buffer_ops.buffer_load(
                    b_rsrc, idx_i32, vec_width=2, dtype=i32_type
                )
                b_vec64 = vector.BitCastOp(vec1_i64, b8).result
                b_pack = unwrap(
                    vector.ExtractOp(
                        b_vec64, static_position=[0], dynamic_position=[]
                    ).result
                )
                return b_pack

            # Split A loads logic
            max_bytes_per_load = 16
            num_a_loads = (
                bytes_per_thread_a + max_bytes_per_load - 1
            ) // max_bytes_per_load
            vec_a_parts_lens = []
            remaining_bytes = bytes_per_thread_a
            for _ in range_constexpr(num_a_loads):
                curr_bytes = min(remaining_bytes, max_bytes_per_load)
                vec_a_parts_lens.append(curr_bytes)
                remaining_bytes -= curr_bytes

            def load_a_split(idx_div4):
                parts = []
                curr_off_i32 = 0
                for i in range_constexpr(num_a_loads):
                    curr_bytes = vec_a_parts_lens[i]
                    curr_idx = idx_div4
                    if curr_off_i32 > 0:
                        curr_idx = idx_div4 + curr_off_i32
                    val = buffer_ops.buffer_load(
                        a_rsrc, curr_idx, vec_width=4, dtype=i32_type
                    )
                    parts.append(val)
                    curr_off_i32 += curr_bytes // 4
                return parts

            # --- Double-buffered LDS A (ping-pong) ---
            c1 = arith.constant(1, index=True)

            def swizzle_idx(row_idx, col_idx):
                k_blocks16 = arith.constant(tile_k // 16, index=True)
                row_mod = row_idx % k_blocks16
                xor_mask = row_mod * 16
                return _arith_mlir.XOrIOp(unwrap(col_idx), unwrap(xor_mask)).result

            def store_a_to_lds(vec_a_in_parts, buffer_idx):
                buffer_offset_val = _arith_mlir.MulIOp(
                    unwrap(buffer_idx),
                    unwrap(arith.constant(tile_m * lds_stride, index=True)),
                ).result
                curr_store_off = 0
                for i in range_constexpr(num_a_loads):
                    val_vec = vec_a_in_parts[i]
                    curr_bytes = vec_a_parts_lens[i]
                    curr_i32 = curr_bytes // 4
                    col_0 = col_a_local + curr_store_off
                    col_swizzled_0 = swizzle_idx(row_a_local, col_0)
                    coord_store_0 = flir.make_coord(
                        unwrap(row_a_local), unwrap(col_swizzled_0)
                    )
                    idx_0 = flir.crd2idx(coord_store_0, layout_lds)
                    idx_0_final = _arith_mlir.AddIOp(
                        unwrap(idx_0), unwrap(buffer_offset_val)
                    ).result
                    if curr_i32 == 4:
                        val_16 = vector.BitCastOp(vec16_f8, val_vec).result
                        vector.StoreOp(val_16, lds_a, [unwrap(idx_0_final)])
                    elif curr_i32 == 2:
                        val_2_i32 = vector.ShuffleOp(val_vec, val_vec, [0, 1]).result
                        val_8 = vector.BitCastOp(vec8_f8, val_2_i32).result
                        vector.StoreOp(val_8, lds_a, [unwrap(idx_0_final)])
                    else:
                        vec_f8 = ir.VectorType.get([curr_bytes], f8)
                        if curr_bytes <= 4:
                            val_1_i32 = vector.ShuffleOp(val_vec, val_vec, [0]).result
                            val_f8 = vector.BitCastOp(vec_f8, val_1_i32).result
                        else:
                            val_2_i32 = vector.ShuffleOp(val_vec, val_vec, [0, 1]).result
                            val_f8 = vector.BitCastOp(vec_f8, val_2_i32).result
                        vector.StoreOp(val_f8, lds_a, [unwrap(idx_0_final)])
                    curr_store_off += curr_bytes

            def mfma_tile(accs_in, k_iv, buffer_read):
                buffer_offset_val = _arith_mlir.MulIOp(
                    unwrap(buffer_read),
                    unwrap(arith.constant(tile_m * lds_stride, index=True)),
                ).result
                current_accs_list = list(accs_in)
                for ki_step in range_constexpr(k_unroll):
                    b_packs = []
                    for ni in range_constexpr(num_acc_n):
                        b = load_b_pack(k_iv, ki_step, ni)
                        b_packs.append(b)

                    ki64 = (ki_step // 2) * 64
                    ki64_val = arith.constant(ki64, index=True)
                    half = ki_step % 2
                    col_base = col_offset_base + ki64_val

                    for mi in range_constexpr(m_repeat):
                        mi_val = arith.constant(mi * 16, index=True)
                        curr_row_a_lds = row_a_lds + mi_val
                        col_base_swizzled = swizzle_idx(curr_row_a_lds, col_base)
                        coord_a = flir.make_coord(
                            unwrap(curr_row_a_lds), unwrap(col_base_swizzled)
                        )
                        idx_a = flir.crd2idx(coord_a, layout_lds)
                        idx_a_final = _arith_mlir.AddIOp(
                            unwrap(idx_a), unwrap(buffer_offset_val)
                        ).result
                        # Force a single 16B LDS read so AMDGPU selects ds_read_b128.
                        # We then pick the desired 64-bit half (half=0/1) in registers.
                        idx_a_blk = (arith.ArithValue(idx_a_final) // c16).value
                        loaded_a16 = unwrap(memref.load(lds_a16, [idx_a_blk]))
                        a_vec128 = vector.BitCastOp(vec2_i64, loaded_a16).result
                        a_pack = vector.ExtractOp(
                            a_vec128, static_position=[half], dynamic_position=[]
                        ).result
                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            curr_acc = current_accs_list[acc_idx]
                            b_pack = b_packs[ni]
                            acc0 = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                vec4_f32,
                                [
                                    unwrap(a_pack),
                                    unwrap(b_pack),
                                    unwrap(curr_acc),
                                    unwrap(c0_i32),
                                    unwrap(c0_i32),
                                    unwrap(c0_i32),
                                ],
                            ).result
                            current_accs_list[acc_idx] = acc0
                return current_accs_list

            assert K % tile_k == 0, f"K({K}) must be divisible by tile_k({tile_k})"
            num_tiles = K // tile_k
            assert num_tiles >= 1

            # Prolog: tile0 into buffer0
            vec_a_cur_parts = load_a_split(idx_a_base_div4)
            store_a_to_lds(vec_a_cur_parts, c0)
            gpu.barrier()
            accs = list(acc_inits)
            buf_read = c0

            if num_tiles > 1:
                k1 = arith.constant(tile_k, index=True)
                k1_div4 = k1 / 4
                idx_a_1_div4 = idx_a_base_div4 + k1_div4
                vec_a_next_parts = load_a_split(idx_a_1_div4)
            else:
                vec_a_next_parts = vec_a_cur_parts

            for t in range_constexpr(num_tiles - 1):
                k_iv = arith.constant(t * tile_k, index=True)
                accs = mfma_tile(accs, k_iv, buf_read)
                buf_write = _arith_mlir.XOrIOp(unwrap(buf_read), unwrap(c1)).result
                store_a_to_lds(vec_a_next_parts, buf_write)
                gpu.barrier()
                buf_read = buf_write
                if (t + 2) < num_tiles:
                    k2 = arith.constant((t + 2) * tile_k, index=True)
                    k2_div4 = k2 / 4
                    idx_a_2_div4 = idx_a_base_div4 + k2_div4
                    vec_a_next_parts = load_a_split(idx_a_2_div4)

            # Last tile
            k_last = arith.constant((num_tiles - 1) * tile_k, index=True)
            final_accs = mfma_tile(accs, k_last, buf_read)

            # Load scales (done after MFMA for clarity; correctness-equivalent).
            s_b_vals = []
            for ni in range_constexpr(num_acc_n):
                offset = ni * 16
                c_offset = arith.constant(offset, index=True)
                col_g = by_n + n_tile_base + c_offset + lane_mod_16
                val = buffer_ops.buffer_load(scale_b_rsrc, col_g, vec_width=1, dtype=f32)
                s_b_vals.append(val)

            s_a_vecs = []
            row_off_base = lane_div_16 * 4
            for mi in range_constexpr(m_repeat):
                row_base_m = bx_m + (mi * 16)
                row_g_base = row_base_m + row_off_base
                s_a_vec = buffer_ops.buffer_load(
                    scale_a_rsrc, row_g_base, vec_width=4, dtype=f32
                )
                s_a_vec4 = vector.BitCastOp(vec4_f32, s_a_vec).result
                s_a_vecs.append(s_a_vec4)

            for mi in range_constexpr(m_repeat):
                row_base_m = bx_m + (mi * 16)
                s_a_vec4 = s_a_vecs[mi]
                for i in range_constexpr(4):
                    row_off = (lane_div_16 * 4) + i
                    row_g = row_base_m + row_off
                    s_a = vector.ExtractOp(
                        s_a_vec4, static_position=[i], dynamic_position=[]
                    ).result
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        acc = final_accs[acc_idx]
                        val = vector.ExtractOp(acc, [], [i]).result
                        offset = ni * 16
                        c_offset = arith.constant(offset, index=True)
                        col_g = by_n + n_tile_base + c_offset + lane_mod_16
                        s_b = s_b_vals[ni]
                        val_s = val * s_a
                        val_s = val_s * s_b
                        val_f16 = _arith_mlir.TruncFOp(T.f16(), unwrap(val_s)).result
                        idx = flir.crd2idx(
                            flir.make_coord(unwrap(row_g), unwrap(col_g)), layout_c
                        )
                        buffer_ops.buffer_store(val_f16, c_rsrc, idx)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_c: lambda: T.memref(size_c, T.f16()),
            arg_a: lambda: T.memref(size_a, _f8()),
            arg_b: lambda: T.memref(size_b, _f8()),
            arg_scale_a: lambda: T.memref(M, T.f32()),
            arg_scale_b: lambda: T.memref(N, T.f32()),
            m_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            c1 = arith.constant(1, index=True).value
            bdx = arith.constant(256, index=True).value
            gx = arith.constant(M // tile_m, index=True).value
            gy = arith.constant(N // tile_n, index=True).value
            flir.gpu_ext.LaunchFuncOp(
                ["mfma_mod_m1024", "kernel_fixed"],
                grid_size=(gx, gy, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[
                    unwrap(arg_c),
                    unwrap(arg_a),
                    unwrap(arg_b),
                    unwrap(arg_scale_a),
                    unwrap(arg_scale_b),
                    unwrap(m_in),
                    unwrap(n_in),
                    unwrap(k_in),
                ],
            )

    m = _MFMA_M1024()
    # Optional: bypass ExecutionEngine host stub and launch via HIP driver API.
    use_hip_driver = os.environ.get("FLIR_USE_HIP_DRIVER", "0") == "1"
    exe = None
    kernel_func = None
    if use_hip_driver:
        try:
            from hip import hip  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "FLIR_USE_HIP_DRIVER=1 requested but `hip` python package is not importable."
            ) from e

        hsaco = compile_to_hsaco(
            m.module,
            kernel_name=f"mfma_m1024_t{tile_m}x{tile_n}x{tile_k}_driver",
            waves_per_eu=2,
        )
        hip_module = _hip_check(hip.hipModuleLoadData(hsaco))
        kernel_func = _hip_check(hip.hipModuleGetFunction(hip_module, b"kernel_fixed"))
        print("✓ Compiled (m1024 hsaco + hip driver)")
    else:
        exe = pyflir.compile(m)
        print("✓ Compiled (m1024 impl)")

    device = torch.device("cuda")
    a_q_fp8, b_q_fp8, b_shuffled, scale_a, scale_b, c_ref = _prepare_inputs(
        M, N, K, device
    )
    c_out_raw = torch.zeros((M, N), dtype=torch.float16, device=device)

    def launch_kernel(c, a, b, sa, sb):
        if use_hip_driver:
            assert kernel_func is not None
            from hip import hip  # type: ignore

            arg_ptrs = [
                ctypes.c_void_p(int(c.data_ptr())),
                ctypes.c_void_p(int(a.data_ptr())),
                ctypes.c_void_p(int(b.data_ptr())),
                ctypes.c_void_p(int(sa.data_ptr())),
                ctypes.c_void_p(int(sb.data_ptr())),
                ctypes.c_long(int(M)),
                ctypes.c_long(int(N)),
                ctypes.c_long(int(K)),
            ]
            args_array = (ctypes.c_void_p * len(arg_ptrs))(
                *[ctypes.addressof(p) for p in arg_ptrs]
            )
            _hip_check(
                hip.hipModuleLaunchKernel(
                    kernel_func,
                    M // tile_m,
                    N // tile_n,
                    1,
                    total_threads,
                    1,
                    1,
                    0,
                    0,
                    args_array,
                    None,
                )
            )
        else:
            assert exe is not None
            exe(c, a, b, sa, sb, M, N, K)

    _, us = run_perftest(
        launch_kernel, c_out_raw, a_q_fp8, b_shuffled, scale_a, scale_b
    )
    torch.cuda.synchronize()
    c_out_scaled = c_out_raw.to(torch.float32)
    assert verify_output(c_out_scaled, c_ref, rtol=0.1, atol=0.1)

    bytes_moved = size_a + size_b + size_c * 2 + (M + N) * 4
    flops = 2 * M * N * K
    tflops = flops / (us / 1e6) / 1e12
    bw = bytes_moved / 1e9 / (us / 1e6)
    print(f"Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {bw:.2f} GB/s")
    _maybe_run_aiter(
        a_q_fp8, b_shuffled, scale_a, scale_b, c_ref, flops, bytes_moved, us
    )


def test_mfma_fp8_rocir_preshuffle(M, N, K, tile_m, tile_n, tile_k):
    print("=" * 80)
    print(f"MFMA FP8 GEMM Test ({M}x{N}x{K} Tile: {tile_m}x{tile_n}x{tile_k}) [Merged]")
    print("=" * 80)
    if M < 32:
        _run_impl_m16(M, N, K, tile_m, tile_n, tile_k)
    else:
        _run_impl_m1024(M, N, K, tile_m, tile_n, tile_k)


if __name__ == "__main__":
    torch.set_default_device("cuda")
    print("Running merged tiling tests...")
    # Small-M (m16) case
    # test_mfma_fp8_rocir_preshuffle(16, 7168, 2048, tile_m=16, tile_n=32, tile_k=512)
    # Large-M (m1024) case
    # test_mfma_fp8_rocir_preshuffle(1024, 9216, 4096, tile_m=64, tile_n=256, tile_k=128)
    # test_mfma_fp8_rocir_preshuffle(4096, 7168, 256, tile_m=64, tile_n=256, tile_k=128)
    test_mfma_fp8_rocir_preshuffle(4096, 4608, 7168, tile_m=64, tile_n=256, tile_k=128)
    # test_mfma_fp8_rocir_preshuffle(16384, 8192, 1024, tile_m=64, tile_n=256, tile_k=128)

    # Work around a known finalization crash (seen in some environments).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
