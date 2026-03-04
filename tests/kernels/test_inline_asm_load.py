#!/usr/bin/env python3
"""Test llvm.inline_asm for batched global loads on gfx1201.

Goal: Force the LLVM backend to use distinct VGPRs for each global_load_b128,
preventing the VGPR reuse that serializes loads in the WMMA GEMM kernel.

Approach:
  1. Convert memref base address to flat pointer via extract_aligned_pointer_as_index
  2. Compute per-thread byte offset
  3. Use llvm.inline_asm to emit global_load_b128 with explicit VGPR constraints
"""

import sys
import os
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import flydsl
from flydsl.dialects.ext import flir, arith, memref, vector, gpu
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch
from flydsl.lang.ir.types import T
from _mlir import ir
from _mlir.dialects import llvm as _llvm
from _mlir.dialects import arith as _std_arith
from _mlir.dialects import memref as _std_memref
import _mlir.extras.types as Textra

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

gpu_arch = get_rocm_arch()
if not gpu_arch.startswith("gfx12"):
    pytest.skip(f"Test requires gfx12xx, got {gpu_arch}", allow_module_level=True)


def _unwrap(v):
    """Unwrap ArithValue to raw MLIR Value."""
    while hasattr(v, "_value"):
        v = v._value
    return v


def create_inline_asm_copy_kernel():
    """Minimal kernel: copy 8 bf16 values using inline asm global_load_b128.

    Each thread loads 8 bf16 (= 16 bytes = 128 bits) from src and stores to dst.
    This tests the full pipeline: memref -> flat ptr -> inline asm load -> store.
    """
    S = ir.ShapedType.get_dynamic_size()

    class _InlineAsmTest(flir.MlirModule):
        GPU_MODULE_NAME = "inline_asm_test"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        @flir.kernel
        def inline_asm_kernel(
            self: flir.T.i64,
            src: lambda: Textra.memref(S, Textra.bf16()),
            dst: lambda: Textra.memref(S, Textra.bf16()),
        ):
            tid = flir.thread_idx("x")

            # Each thread handles 8 bf16 elements = 16 bytes
            elem_offset = tid * arith.index(8)
            byte_offset = tid * arith.index(16)

            # --- Get flat pointer for src ---
            # 1. Extract base pointer as index
            src_raw = _unwrap(src)
            src_ptr_idx = _unwrap(
                _std_memref.ExtractAlignedPointerAsIndexOp(src_raw).result
            )

            # 2. Convert index -> i64
            i64_ty = ir.IntegerType.get_signless(64)
            src_base_i64 = _unwrap(_std_arith.IndexCastOp(i64_ty, src_ptr_idx).result)

            # 3. Convert byte offset to i64
            byte_off_raw = _unwrap(arith.unwrap(byte_offset))
            byte_off_i64 = _unwrap(_std_arith.IndexCastOp(i64_ty, byte_off_raw).result)

            # 4. Add base + offset
            addr_i64 = _unwrap(_std_arith.AddIOp(src_base_i64, byte_off_i64).result)

            # 5. Convert to llvm.ptr
            ptr_ty = ir.Type.parse("!llvm.ptr")
            addr_ptr = _unwrap(_llvm.IntToPtrOp(ptr_ty, addr_i64).result)

            # --- Use inline asm to do global_load_b128 ---
            # global_load_b128 loads 128 bits (16 bytes = 4xi32) from flat address
            v4i32_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))

            loaded = _llvm.inline_asm(
                v4i32_ty,  # result: vector<4xi32>
                [addr_ptr],  # operands: the flat pointer
                "global_load_b128 $0, $1, off\ns_wait_loadcnt 0x0",  # asm
                "=&v,v",  # constraints: output=vgpr (early-clobber), input=vgpr
                has_side_effects=True,
            )

            # --- Bitcast v4i32 -> v8bf16 and store ---
            v8bf16_ty = ir.VectorType.get([8], ir.BF16Type.get())
            loaded_bf16 = vector.bitcast(v8bf16_ty, loaded)

            # Store to dst using regular vector.store
            vector.store(loaded_bf16, dst, [elem_offset])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            src: lambda: Textra.memref(S, Textra.bf16()),
            dst: lambda: Textra.memref(S, Textra.bf16()),
        ):
            c1 = arith.index(1)
            c32 = arith.index(32)
            flir.gpu_ext.LaunchFuncOp(
                ["inline_asm_test", "inline_asm_kernel"],
                grid_size=(c1, c1, c1),
                block_size=(c32, c1, c1),
                kernel_operands=[src, dst],
            )

    return _InlineAsmTest()


def test_inline_asm_single_load():
    """Test that a single inline asm global_load_b128 works correctly."""
    print(f"\n{'=' * 60}")
    print(f"Inline ASM single load test - {gpu_arch}")
    print(f"{'=' * 60}")

    N = 256  # 32 threads * 8 elements each
    m = create_inline_asm_copy_kernel()
    exe = flydsl.compile(m)

    src = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    dst = torch.zeros(N, device="cuda", dtype=torch.bfloat16)

    exe(src, dst)
    torch.cuda.synchronize()

    # Verify copy
    error = torch.max(torch.abs(src.float() - dst.float())).item()
    print(f"Max error: {error:.2e}")
    assert error < 1e-6, f"Inline asm copy failed: error={error:.2e}"
    print("PASS - single inline asm load works!")
    return True


def create_batched_load_kernel(num_loads=2):
    """Kernel that uses inline asm for multiple batched global_load_b128.

    Tests that we can issue multiple loads in a single asm block with
    s_clause prefix to force the hardware to batch them.
    """
    S = ir.ShapedType.get_dynamic_size()

    class _BatchedLoadTest(flir.MlirModule):
        GPU_MODULE_NAME = "batched_load_test"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        @flir.kernel
        def batched_load_kernel(
            self: flir.T.i64,
            src: lambda: Textra.memref(S, Textra.bf16()),
            dst: lambda: Textra.memref(S, Textra.bf16()),
        ):
            tid = flir.thread_idx("x")
            i64_ty = ir.IntegerType.get_signless(64)
            ptr_ty = ir.Type.parse("!llvm.ptr")
            v4i32_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
            v8bf16_ty = ir.VectorType.get([8], ir.BF16Type.get())

            # Get base pointer
            src_raw = _unwrap(src)
            src_ptr_idx = _unwrap(
                _std_memref.ExtractAlignedPointerAsIndexOp(src_raw).result
            )
            src_base_i64 = _unwrap(_std_arith.IndexCastOp(i64_ty, src_ptr_idx).result)

            # Compute addresses for num_loads chunks, each 16 bytes apart per thread
            # Thread t loads from: base + t*16 + chunk*32*16
            # (32 threads, 16 bytes each = 512 bytes per chunk)
            chunk_stride = 32 * 16  # bytes per chunk

            addrs = []
            for i in range_constexpr(num_loads):
                byte_off_raw = _unwrap(
                    arith.unwrap(tid * arith.index(16) + arith.index(i * chunk_stride))
                )
                byte_off_i64 = _unwrap(
                    _std_arith.IndexCastOp(i64_ty, byte_off_raw).result
                )
                addr_i64 = _unwrap(_std_arith.AddIOp(src_base_i64, byte_off_i64).result)
                addr_ptr = _llvm.IntToPtrOp(ptr_ty, addr_i64).result
                addrs.append(addr_ptr)

            if num_loads == 2:
                # 2 loads with s_clause 1
                # Result type: struct of 2 x v4i32
                # Use llvm.StructType for the result
                struct_ty = _llvm.StructType.get_literal([v4i32_ty, v4i32_ty])

                result = _llvm.inline_asm(
                    struct_ty,
                    addrs,
                    "s_clause 1\n"
                    "global_load_b128 $0, $2, off\n"
                    "global_load_b128 $1, $3, off\n"
                    "s_wait_loadcnt 0x0",
                    "=&v,=&v,v,v",  # early-clobber outputs to force distinct VGPRs
                    has_side_effects=True,
                )

                # Extract results from struct
                loaded_0 = _llvm.ExtractValueOp(v4i32_ty, result, [0]).result
                loaded_1 = _llvm.ExtractValueOp(v4i32_ty, result, [1]).result
                loaded_list = [loaded_0, loaded_1]
            else:
                # Fallback: individual loads
                loaded_list = []
                for i in range_constexpr(num_loads):
                    loaded = _llvm.inline_asm(
                        v4i32_ty,
                        [addrs[i]],
                        "global_load_b128 $0, $1, off",
                        "=v,v",
                        has_side_effects=True,
                    )
                    loaded_list.append(loaded)

            # Store results
            for i in range_constexpr(num_loads):
                elem_offset = tid * arith.index(8) + arith.index(i * 32 * 8)
                loaded_bf16 = vector.bitcast(v8bf16_ty, loaded_list[i])
                vector.store(loaded_bf16, dst, [elem_offset])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            src: lambda: Textra.memref(S, Textra.bf16()),
            dst: lambda: Textra.memref(S, Textra.bf16()),
        ):
            c1 = arith.index(1)
            c32 = arith.index(32)
            flir.gpu_ext.LaunchFuncOp(
                ["batched_load_test", "batched_load_kernel"],
                grid_size=(c1, c1, c1),
                block_size=(c32, c1, c1),
                kernel_operands=[src, dst],
            )

    return _BatchedLoadTest()


def test_batched_load_2():
    """Test 2 batched loads with s_clause 1."""
    print(f"\n{'=' * 60}")
    print(f"Inline ASM batched load (2 loads) test - {gpu_arch}")
    print(f"{'=' * 60}")

    N = 2 * 32 * 8  # 2 chunks * 32 threads * 8 elements
    m = create_batched_load_kernel(num_loads=2)
    exe = flydsl.compile(m)

    src = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    dst = torch.zeros(N, device="cuda", dtype=torch.bfloat16)

    exe(src, dst)
    torch.cuda.synchronize()

    error = torch.max(torch.abs(src.float() - dst.float())).item()
    print(f"Max error: {error:.2e}")
    assert error < 1e-6, f"Batched load failed: error={error:.2e}"
    print("PASS - batched 2-load inline asm works!")


if __name__ == "__main__":
    test_inline_asm_single_load()
    test_batched_load_2()
