#!/usr/bin/env python3
"""
Backend Correctness Tests - Compare ASM backend vs execution_engine backend.

Each kernel tests multiple operations to maximize coverage with minimal test time.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

import flydsl
from flydsl.dialects.ext import flir, arith
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch
from _mlir import ir
import _mlir.extras.types as T

DTYPE_FP32 = torch.float32


def compare_results(ref, test, atol=1e-5, rtol=1e-5, name=""):
    """Compare two tensors and report differences."""
    abs_diff = (ref.float() - test.float()).abs()
    max_diff = abs_diff.max().item()
    is_close = torch.allclose(ref.float(), test.float(), atol=atol, rtol=rtol)
    
    if not is_close:
        mask = ~torch.isclose(ref.float(), test.float(), atol=atol, rtol=rtol)
        num_mismatch = mask.sum().item()
        percent = num_mismatch / ref.numel() * 100
        print(f"\n{name} MISMATCH: max_diff={max_diff:.6e}, {num_mismatch}/{ref.numel()} ({percent:.1f}%)")
    
    return is_close, max_diff


# =============================================================================
# Test: Combined Arithmetic Ops (add, sub, mul, fma, scale)
# =============================================================================

class TestArithmeticOps:
    """Test arithmetic operations: add, sub, mul, fma in one kernel."""
    
    @pytest.mark.parametrize("size", [64, 256])
    def test_arith_combined(self, size):
        """Test combined arithmetic: out = (a + b) * (a - b) + a * scale."""
        gpu_arch = get_rocm_arch()
        SIZE = size
        THREADS = 64
        TILE = 1  # 1 element per thread for simplicity
        TILE_ELEMS = THREADS * TILE
        SCALE = 2.5
        S = ir.ShapedType.get_dynamic_size()
        
        class _ArithCombined(flir.MlirModule):
            GPU_MODULE_NAME = "arith_combined"
            GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']
            
            @flir.kernel
            def kernel(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                B: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                tid = flir.thread_idx("x")
                bid = flir.block_idx("x")
                bdim = flir.block_dim("x")
                tid_linear = tid
                
                thr_layout = flir.make_ordered_layout((THREADS,), order=(0,))
                val_layout = flir.make_ordered_layout((TILE,), order=(0,))
                copy_atom = flir.make_copy_atom(T.f32(), vector_size=1)
                
                tiled_copy = flir.make_tiled_copy_tv(
                    copy_atom, thr_layout, val_layout,
                    thr_shape=(THREADS,), val_shape=(TILE,),
                )
                
                tensor_A = flir.make_tensor(A, shape=(n,), strides=(1,))
                tensor_B = flir.make_tensor(B, shape=(n,), strides=(1,))
                tensor_C = flir.make_tensor(C, shape=(n,), strides=(1,))
                
                tile_shape = (TILE_ELEMS,)
                gA = flir.zipped_divide(tensor_A, tile_shape)
                gB = flir.zipped_divide(tensor_B, tile_shape)
                gC = flir.zipped_divide(tensor_C, tile_shape)
                idC = flir.make_identity_tensor((n,))
                cC = flir.zipped_divide(idC, tile_shape)
                
                blk_coord = (bid,)
                blkA, blkB, blkC = gA[blk_coord], gB[blk_coord], gC[blk_coord]
                blkCrd = cC[blk_coord]
                
                thr_copy = tiled_copy.get_slice(tid_linear)
                thrA = thr_copy.partition_S(blkA)
                thrB = thr_copy.partition_S(blkB)
                thrC = thr_copy.partition_S(blkC)
                thrCrd = thr_copy.partition_S(blkCrd)
                
                val_shape = tiled_copy.val_shape
                frgA = flir.make_fragment_like(thrA, T.f32())
                frgB = flir.make_fragment_like(thrB, T.f32())
                frgC = flir.make_fragment_like(thrC, T.f32())
                
                pred_ty = ir.IntegerType.get_signless(1)
                frgPred = flir.make_rmem_tensor(val_shape, pred_ty)
                for linear in range(val_shape[0]):
                    lin_idx = flir.const_index(linear)
                    coords = thrCrd.coords_from_linear(lin_idx)
                    pred_val = flir.elem_less(coords, (n,))
                    frgPred[tuple(frgPred.offsets_from_linear(lin_idx))] = pred_val
                
                flir.copy(tiled_copy, thrA, frgA, pred=frgPred)
                flir.copy(tiled_copy, thrB, frgB, pred=frgPred)
                
                # Combined: out = (a + b) * (a - b) + a * SCALE
                # Tests: add, sub, mul, fma pattern, constant mul
                for i in range_constexpr(TILE):
                    idx = flir.const_index(i)
                    a = frgA[(idx,)]
                    b = frgB[(idx,)]
                    
                    t1 = a + b        # add
                    t2 = a - b        # sub  
                    t3 = t1 * t2      # mul
                    t4 = a * SCALE    # constant mul
                    out = t3 + t4     # add (fma-like)
                    
                    frgC[(idx,)] = out
                
                flir.copy(tiled_copy, frgC, thrC, pred=frgPred)
            
            @flir.jit
            def __call__(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                B: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                c1 = arith.index(1)
                c_tile = arith.index(TILE_ELEMS)
                gx = (n + c_tile - c1) // c_tile
                bx = arith.index(THREADS)
                flir.gpu_ext.LaunchFuncOp(
                    [self.GPU_MODULE_NAME, "kernel"],
                    grid_size=(gx, c1, c1),
                    block_size=(bx, c1, c1),
                    kernel_operands=[A, B, C, n],
                )
        
        for backend in ["execution_engine", "asm"]:
            a = torch.randn(SIZE, dtype=DTYPE_FP32, device="cuda")
            b = torch.randn(SIZE, dtype=DTYPE_FP32, device="cuda")
            c_out = torch.zeros(SIZE, dtype=DTYPE_FP32, device="cuda")
            c_ref = (a + b) * (a - b) + a * SCALE
            
            try:
                m = _ArithCombined()
                exe = flydsl.compile(m, backend=backend)
                exe(a, b, c_out, SIZE)
                torch.cuda.synchronize()
                
                is_close, max_diff = compare_results(c_ref, c_out, atol=1e-5, name=f"arith_{backend}")
                assert is_close, f"arith_combined ({backend}) failed: max_diff={max_diff}"
                print(f"  arith_combined size={SIZE} {backend}: PASS")
            except Exception as e:
                pytest.fail(f"arith_combined ({backend}) failed: {e}")


# =============================================================================
# Test: Max/Min Operations
# =============================================================================

class TestMinMaxOps:
    """Test max and min operations."""
    
    @pytest.mark.parametrize("size", [64, 256])
    def test_minmax(self, size):
        """Test: out = max(a, b) - min(a, b) (range)."""
        gpu_arch = get_rocm_arch()
        SIZE = size
        THREADS = 64
        TILE = 1
        TILE_ELEMS = THREADS * TILE
        S = ir.ShapedType.get_dynamic_size()
        
        class _MinMax(flir.MlirModule):
            GPU_MODULE_NAME = "minmax"
            GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']
            
            @flir.kernel
            def kernel(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                B: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                tid = flir.thread_idx("x")
                bid = flir.block_idx("x")
                tid_linear = tid
                
                thr_layout = flir.make_ordered_layout((THREADS,), order=(0,))
                val_layout = flir.make_ordered_layout((TILE,), order=(0,))
                copy_atom = flir.make_copy_atom(T.f32(), vector_size=1)
                
                tiled_copy = flir.make_tiled_copy_tv(
                    copy_atom, thr_layout, val_layout,
                    thr_shape=(THREADS,), val_shape=(TILE,),
                )
                
                tensor_A = flir.make_tensor(A, shape=(n,), strides=(1,))
                tensor_B = flir.make_tensor(B, shape=(n,), strides=(1,))
                tensor_C = flir.make_tensor(C, shape=(n,), strides=(1,))
                
                tile_shape = (TILE_ELEMS,)
                gA = flir.zipped_divide(tensor_A, tile_shape)
                gB = flir.zipped_divide(tensor_B, tile_shape)
                gC = flir.zipped_divide(tensor_C, tile_shape)
                idC = flir.make_identity_tensor((n,))
                cC = flir.zipped_divide(idC, tile_shape)
                
                blk_coord = (bid,)
                blkA, blkB, blkC = gA[blk_coord], gB[blk_coord], gC[blk_coord]
                blkCrd = cC[blk_coord]
                
                thr_copy = tiled_copy.get_slice(tid_linear)
                thrA = thr_copy.partition_S(blkA)
                thrB = thr_copy.partition_S(blkB)
                thrC = thr_copy.partition_S(blkC)
                thrCrd = thr_copy.partition_S(blkCrd)
                
                val_shape = tiled_copy.val_shape
                frgA = flir.make_fragment_like(thrA, T.f32())
                frgB = flir.make_fragment_like(thrB, T.f32())
                frgC = flir.make_fragment_like(thrC, T.f32())
                
                pred_ty = ir.IntegerType.get_signless(1)
                frgPred = flir.make_rmem_tensor(val_shape, pred_ty)
                for linear in range(val_shape[0]):
                    lin_idx = flir.const_index(linear)
                    coords = thrCrd.coords_from_linear(lin_idx)
                    pred_val = flir.elem_less(coords, (n,))
                    frgPred[tuple(frgPred.offsets_from_linear(lin_idx))] = pred_val
                
                flir.copy(tiled_copy, thrA, frgA, pred=frgPred)
                flir.copy(tiled_copy, thrB, frgB, pred=frgPred)
                
                # out = max(a, b) - min(a, b)
                for i in range_constexpr(TILE):
                    idx = flir.const_index(i)
                    a = frgA[(idx,)]
                    b = frgB[(idx,)]
                    
                    mx = arith.maximum(a, b)
                    mn = arith.minimum(a, b)
                    out = mx - mn
                    
                    frgC[(idx,)] = out
                
                flir.copy(tiled_copy, frgC, thrC, pred=frgPred)
            
            @flir.jit
            def __call__(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                B: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                c1 = arith.index(1)
                c_tile = arith.index(TILE_ELEMS)
                gx = (n + c_tile - c1) // c_tile
                bx = arith.index(THREADS)
                flir.gpu_ext.LaunchFuncOp(
                    [self.GPU_MODULE_NAME, "kernel"],
                    grid_size=(gx, c1, c1),
                    block_size=(bx, c1, c1),
                    kernel_operands=[A, B, C, n],
                )
        
        for backend in ["execution_engine", "asm"]:
            a = torch.randn(SIZE, dtype=DTYPE_FP32, device="cuda")
            b = torch.randn(SIZE, dtype=DTYPE_FP32, device="cuda")
            c_out = torch.zeros(SIZE, dtype=DTYPE_FP32, device="cuda")
            c_ref = torch.maximum(a, b) - torch.minimum(a, b)
            
            try:
                m = _MinMax()
                exe = flydsl.compile(m, backend=backend)
                exe(a, b, c_out, SIZE)
                torch.cuda.synchronize()
                
                is_close, max_diff = compare_results(c_ref, c_out, atol=1e-6, name=f"minmax_{backend}")
                assert is_close, f"minmax ({backend}) failed: max_diff={max_diff}"
                print(f"  minmax size={SIZE} {backend}: PASS")
            except Exception as e:
                pytest.fail(f"minmax ({backend}) failed: {e}")


# =============================================================================
# Test: Division and Negation
# =============================================================================

class TestDivNeg:
    """Test division and negation."""
    
    @pytest.mark.parametrize("size", [64, 256])
    def test_div_neg(self, size):
        """Test: out = -a / b."""
        gpu_arch = get_rocm_arch()
        SIZE = size
        THREADS = 64
        TILE = 1
        TILE_ELEMS = THREADS * TILE
        S = ir.ShapedType.get_dynamic_size()
        
        class _DivNeg(flir.MlirModule):
            GPU_MODULE_NAME = "divneg"
            GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']
            
            @flir.kernel
            def kernel(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                B: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                tid = flir.thread_idx("x")
                bid = flir.block_idx("x")
                tid_linear = tid
                
                thr_layout = flir.make_ordered_layout((THREADS,), order=(0,))
                val_layout = flir.make_ordered_layout((TILE,), order=(0,))
                copy_atom = flir.make_copy_atom(T.f32(), vector_size=1)
                
                tiled_copy = flir.make_tiled_copy_tv(
                    copy_atom, thr_layout, val_layout,
                    thr_shape=(THREADS,), val_shape=(TILE,),
                )
                
                tensor_A = flir.make_tensor(A, shape=(n,), strides=(1,))
                tensor_B = flir.make_tensor(B, shape=(n,), strides=(1,))
                tensor_C = flir.make_tensor(C, shape=(n,), strides=(1,))
                
                tile_shape = (TILE_ELEMS,)
                gA = flir.zipped_divide(tensor_A, tile_shape)
                gB = flir.zipped_divide(tensor_B, tile_shape)
                gC = flir.zipped_divide(tensor_C, tile_shape)
                idC = flir.make_identity_tensor((n,))
                cC = flir.zipped_divide(idC, tile_shape)
                
                blk_coord = (bid,)
                blkA, blkB, blkC = gA[blk_coord], gB[blk_coord], gC[blk_coord]
                blkCrd = cC[blk_coord]
                
                thr_copy = tiled_copy.get_slice(tid_linear)
                thrA = thr_copy.partition_S(blkA)
                thrB = thr_copy.partition_S(blkB)
                thrC = thr_copy.partition_S(blkC)
                thrCrd = thr_copy.partition_S(blkCrd)
                
                val_shape = tiled_copy.val_shape
                frgA = flir.make_fragment_like(thrA, T.f32())
                frgB = flir.make_fragment_like(thrB, T.f32())
                frgC = flir.make_fragment_like(thrC, T.f32())
                
                pred_ty = ir.IntegerType.get_signless(1)
                frgPred = flir.make_rmem_tensor(val_shape, pred_ty)
                for linear in range(val_shape[0]):
                    lin_idx = flir.const_index(linear)
                    coords = thrCrd.coords_from_linear(lin_idx)
                    pred_val = flir.elem_less(coords, (n,))
                    frgPred[tuple(frgPred.offsets_from_linear(lin_idx))] = pred_val
                
                flir.copy(tiled_copy, thrA, frgA, pred=frgPred)
                flir.copy(tiled_copy, thrB, frgB, pred=frgPred)
                
                # out = -a / b  (using 0 - a for negation)
                for i in range_constexpr(TILE):
                    idx = flir.const_index(i)
                    a = frgA[(idx,)]
                    b = frgB[(idx,)]
                    
                    neg_a = 0.0 - a  # negation via subtraction
                    out = neg_a / b
                    
                    frgC[(idx,)] = out
                
                flir.copy(tiled_copy, frgC, thrC, pred=frgPred)
            
            @flir.jit
            def __call__(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                B: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                c1 = arith.index(1)
                c_tile = arith.index(TILE_ELEMS)
                gx = (n + c_tile - c1) // c_tile
                bx = arith.index(THREADS)
                flir.gpu_ext.LaunchFuncOp(
                    [self.GPU_MODULE_NAME, "kernel"],
                    grid_size=(gx, c1, c1),
                    block_size=(bx, c1, c1),
                    kernel_operands=[A, B, C, n],
                )
        
        for backend in ["execution_engine", "asm"]:
            a = torch.randn(SIZE, dtype=DTYPE_FP32, device="cuda")
            b = torch.randn(SIZE, dtype=DTYPE_FP32, device="cuda").abs() + 0.1  # avoid div by small
            c_out = torch.zeros(SIZE, dtype=DTYPE_FP32, device="cuda")
            c_ref = -a / b
            
            try:
                m = _DivNeg()
                exe = flydsl.compile(m, backend=backend)
                exe(a, b, c_out, SIZE)
                torch.cuda.synchronize()
                
                is_close, max_diff = compare_results(c_ref, c_out, atol=1e-5, name=f"divneg_{backend}")
                assert is_close, f"divneg ({backend}) failed: max_diff={max_diff}"
                print(f"  divneg size={SIZE} {backend}: PASS")
            except Exception as e:
                pytest.fail(f"divneg ({backend}) failed: {e}")


# =============================================================================
# Test: Copy (identity, basic memory)
# =============================================================================

class TestCopy:
    """Test basic memory copy."""
    
    @pytest.mark.parametrize("size", [64, 256])
    def test_copy(self, size):
        """Test: out = a (identity copy)."""
        gpu_arch = get_rocm_arch()
        SIZE = size
        THREADS = 64
        TILE = 1
        TILE_ELEMS = THREADS * TILE
        S = ir.ShapedType.get_dynamic_size()
        
        class _Copy(flir.MlirModule):
            GPU_MODULE_NAME = "copy"
            GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']
            
            @flir.kernel
            def kernel(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                tid = flir.thread_idx("x")
                bid = flir.block_idx("x")
                tid_linear = tid
                
                thr_layout = flir.make_ordered_layout((THREADS,), order=(0,))
                val_layout = flir.make_ordered_layout((TILE,), order=(0,))
                copy_atom = flir.make_copy_atom(T.f32(), vector_size=1)
                
                tiled_copy = flir.make_tiled_copy_tv(
                    copy_atom, thr_layout, val_layout,
                    thr_shape=(THREADS,), val_shape=(TILE,),
                )
                
                tensor_A = flir.make_tensor(A, shape=(n,), strides=(1,))
                tensor_C = flir.make_tensor(C, shape=(n,), strides=(1,))
                
                tile_shape = (TILE_ELEMS,)
                gA = flir.zipped_divide(tensor_A, tile_shape)
                gC = flir.zipped_divide(tensor_C, tile_shape)
                idC = flir.make_identity_tensor((n,))
                cC = flir.zipped_divide(idC, tile_shape)
                
                blk_coord = (bid,)
                blkA, blkC = gA[blk_coord], gC[blk_coord]
                blkCrd = cC[blk_coord]
                
                thr_copy = tiled_copy.get_slice(tid_linear)
                thrA = thr_copy.partition_S(blkA)
                thrC = thr_copy.partition_S(blkC)
                thrCrd = thr_copy.partition_S(blkCrd)
                
                val_shape = tiled_copy.val_shape
                frgA = flir.make_fragment_like(thrA, T.f32())
                
                pred_ty = ir.IntegerType.get_signless(1)
                frgPred = flir.make_rmem_tensor(val_shape, pred_ty)
                for linear in range(val_shape[0]):
                    lin_idx = flir.const_index(linear)
                    coords = thrCrd.coords_from_linear(lin_idx)
                    pred_val = flir.elem_less(coords, (n,))
                    frgPred[tuple(frgPred.offsets_from_linear(lin_idx))] = pred_val
                
                flir.copy(tiled_copy, thrA, frgA, pred=frgPred)
                flir.copy(tiled_copy, frgA, thrC, pred=frgPred)
            
            @flir.jit
            def __call__(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                c1 = arith.index(1)
                c_tile = arith.index(TILE_ELEMS)
                gx = (n + c_tile - c1) // c_tile
                bx = arith.index(THREADS)
                flir.gpu_ext.LaunchFuncOp(
                    [self.GPU_MODULE_NAME, "kernel"],
                    grid_size=(gx, c1, c1),
                    block_size=(bx, c1, c1),
                    kernel_operands=[A, C, n],
                )
        
        for backend in ["execution_engine", "asm"]:
            a = torch.randn(SIZE, dtype=DTYPE_FP32, device="cuda")
            c_out = torch.zeros(SIZE, dtype=DTYPE_FP32, device="cuda")
            c_ref = a.clone()
            
            try:
                m = _Copy()
                exe = flydsl.compile(m, backend=backend)
                exe(a, c_out, SIZE)
                torch.cuda.synchronize()
                
                is_close, max_diff = compare_results(c_ref, c_out, atol=0, name=f"copy_{backend}")
                assert is_close, f"copy ({backend}) failed: max_diff={max_diff}"
                print(f"  copy size={SIZE} {backend}: PASS")
            except Exception as e:
                pytest.fail(f"copy ({backend}) failed: {e}")


# =============================================================================
# Test: Vector Load/Store with wider vector width
# =============================================================================

class TestVectorLoadStore:
    """Test vector load/store with vec_width > 1."""
    
    @pytest.mark.parametrize("size", [256, 512])
    def test_vec4_copy(self, size):
        """Test vec4 load/store."""
        gpu_arch = get_rocm_arch()
        SIZE = size
        THREADS = 64
        VEC_WIDTH = 4
        TILE = VEC_WIDTH
        TILE_ELEMS = THREADS * TILE
        S = ir.ShapedType.get_dynamic_size()
        
        class _Vec4Copy(flir.MlirModule):
            GPU_MODULE_NAME = "vec4copy"
            GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']
            
            @flir.kernel
            def kernel(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                B: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                tid = flir.thread_idx("x")
                bid = flir.block_idx("x")
                tid_linear = tid
                
                thr_layout = flir.make_ordered_layout((THREADS,), order=(0,))
                val_layout = flir.make_ordered_layout((TILE,), order=(0,))
                copy_atom = flir.make_copy_atom(T.f32(), vector_size=VEC_WIDTH)
                
                tiled_copy = flir.make_tiled_copy_tv(
                    copy_atom, thr_layout, val_layout,
                    thr_shape=(THREADS,), val_shape=(TILE,),
                )
                
                tensor_A = flir.make_tensor(A, shape=(n,), strides=(1,))
                tensor_B = flir.make_tensor(B, shape=(n,), strides=(1,))
                tensor_C = flir.make_tensor(C, shape=(n,), strides=(1,))
                
                tile_shape = (TILE_ELEMS,)
                gA = flir.zipped_divide(tensor_A, tile_shape)
                gB = flir.zipped_divide(tensor_B, tile_shape)
                gC = flir.zipped_divide(tensor_C, tile_shape)
                idC = flir.make_identity_tensor((n,))
                cC = flir.zipped_divide(idC, tile_shape)
                
                blk_coord = (bid,)
                blkA, blkB, blkC = gA[blk_coord], gB[blk_coord], gC[blk_coord]
                blkCrd = cC[blk_coord]
                
                thr_copy = tiled_copy.get_slice(tid_linear)
                thrA = thr_copy.partition_S(blkA)
                thrB = thr_copy.partition_S(blkB)
                thrC = thr_copy.partition_S(blkC)
                thrCrd = thr_copy.partition_S(blkCrd)
                
                val_shape = tiled_copy.val_shape
                frgA = flir.make_fragment_like(thrA, T.f32())
                frgB = flir.make_fragment_like(thrB, T.f32())
                frgC = flir.make_fragment_like(thrC, T.f32())
                
                pred_ty = ir.IntegerType.get_signless(1)
                frgPred = flir.make_rmem_tensor(val_shape, pred_ty)
                for linear in range(val_shape[0]):
                    lin_idx = flir.const_index(linear)
                    coords = thrCrd.coords_from_linear(lin_idx)
                    pred_val = flir.elem_less(coords, (n,))
                    frgPred[tuple(frgPred.offsets_from_linear(lin_idx))] = pred_val
                
                flir.copy(tiled_copy, thrA, frgA, pred=frgPred)
                flir.copy(tiled_copy, thrB, frgB, pred=frgPred)
                
                # vec add
                for i in range_constexpr(TILE):
                    idx = flir.const_index(i)
                    frgC[(idx,)] = frgA[(idx,)] + frgB[(idx,)]
                
                flir.copy(tiled_copy, frgC, thrC, pred=frgPred)
            
            @flir.jit
            def __call__(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                B: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                c1 = arith.index(1)
                c_tile = arith.index(TILE_ELEMS)
                gx = (n + c_tile - c1) // c_tile
                bx = arith.index(THREADS)
                flir.gpu_ext.LaunchFuncOp(
                    [self.GPU_MODULE_NAME, "kernel"],
                    grid_size=(gx, c1, c1),
                    block_size=(bx, c1, c1),
                    kernel_operands=[A, B, C, n],
                )
        
        for backend in ["execution_engine", "asm"]:
            a = torch.randn(SIZE, dtype=DTYPE_FP32, device="cuda")
            b = torch.randn(SIZE, dtype=DTYPE_FP32, device="cuda")
            c_out = torch.zeros(SIZE, dtype=DTYPE_FP32, device="cuda")
            c_ref = a + b
            
            try:
                m = _Vec4Copy()
                exe = flydsl.compile(m, backend=backend)
                exe(a, b, c_out, SIZE)
                torch.cuda.synchronize()
                
                is_close, max_diff = compare_results(c_ref, c_out, atol=1e-6, name=f"vec4_{backend}")
                assert is_close, f"vec4copy ({backend}) failed: max_diff={max_diff}"
                print(f"  vec4copy size={SIZE} {backend}: PASS")
            except Exception as e:
                pytest.fail(f"vec4copy ({backend}) failed: {e}")


# =============================================================================
# Test: Math Operations (sqrt, rsqrt, exp2)
# =============================================================================

class TestMathOps:
    """Test math operations: sqrt, rsqrt, exp2."""
    
    @pytest.mark.parametrize("size", [64, 256])
    def test_math_combined(self, size):
        """Test: out = sqrt(a) * rsqrt(b) + exp2(a * 0.1)."""
        gpu_arch = get_rocm_arch()
        SIZE = size
        THREADS = 64
        TILE = 1
        TILE_ELEMS = THREADS * TILE
        S = ir.ShapedType.get_dynamic_size()
        
        from flydsl.dialects.ext import math as math_ops
        
        class _MathCombined(flir.MlirModule):
            GPU_MODULE_NAME = "math_combined"
            GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']
            
            @flir.kernel
            def kernel(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                B: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                tid = flir.thread_idx("x")
                bid = flir.block_idx("x")
                tid_linear = tid
                
                thr_layout = flir.make_ordered_layout((THREADS,), order=(0,))
                val_layout = flir.make_ordered_layout((TILE,), order=(0,))
                copy_atom = flir.make_copy_atom(T.f32(), vector_size=1)
                
                tiled_copy = flir.make_tiled_copy_tv(
                    copy_atom, thr_layout, val_layout,
                    thr_shape=(THREADS,), val_shape=(TILE,),
                )
                
                tensor_A = flir.make_tensor(A, shape=(n,), strides=(1,))
                tensor_B = flir.make_tensor(B, shape=(n,), strides=(1,))
                tensor_C = flir.make_tensor(C, shape=(n,), strides=(1,))
                
                tile_shape = (TILE_ELEMS,)
                gA = flir.zipped_divide(tensor_A, tile_shape)
                gB = flir.zipped_divide(tensor_B, tile_shape)
                gC = flir.zipped_divide(tensor_C, tile_shape)
                idC = flir.make_identity_tensor((n,))
                cC = flir.zipped_divide(idC, tile_shape)
                
                blk_coord = (bid,)
                blkA, blkB, blkC = gA[blk_coord], gB[blk_coord], gC[blk_coord]
                blkCrd = cC[blk_coord]
                
                thr_copy = tiled_copy.get_slice(tid_linear)
                thrA = thr_copy.partition_S(blkA)
                thrB = thr_copy.partition_S(blkB)
                thrC = thr_copy.partition_S(blkC)
                thrCrd = thr_copy.partition_S(blkCrd)
                
                val_shape = tiled_copy.val_shape
                frgA = flir.make_fragment_like(thrA, T.f32())
                frgB = flir.make_fragment_like(thrB, T.f32())
                frgC = flir.make_fragment_like(thrC, T.f32())
                
                pred_ty = ir.IntegerType.get_signless(1)
                frgPred = flir.make_rmem_tensor(val_shape, pred_ty)
                for linear in range(val_shape[0]):
                    lin_idx = flir.const_index(linear)
                    coords = thrCrd.coords_from_linear(lin_idx)
                    pred_val = flir.elem_less(coords, (n,))
                    frgPred[tuple(frgPred.offsets_from_linear(lin_idx))] = pred_val
                
                flir.copy(tiled_copy, thrA, frgA, pred=frgPred)
                flir.copy(tiled_copy, thrB, frgB, pred=frgPred)
                
                # out = sqrt(a) * rsqrt(b) + exp2(a * 0.1)
                for i in range_constexpr(TILE):
                    idx = flir.const_index(i)
                    a = frgA[(idx,)]
                    b = frgB[(idx,)]
                    
                    sqrt_a = flir.math.sqrt(arith.as_value(a))
                    rsqrt_b = flir.math.rsqrt(arith.as_value(b))
                    t1 = sqrt_a * rsqrt_b
                    scaled = a * 0.1
                    exp2_scaled = flir.math.exp2(arith.as_value(scaled))
                    out = t1 + exp2_scaled
                    
                    frgC[(idx,)] = out
                
                flir.copy(tiled_copy, frgC, thrC, pred=frgPred)
            
            @flir.jit
            def __call__(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                B: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                c1 = arith.index(1)
                c_tile = arith.index(TILE_ELEMS)
                gx = (n + c_tile - c1) // c_tile
                bx = arith.index(THREADS)
                flir.gpu_ext.LaunchFuncOp(
                    [self.GPU_MODULE_NAME, "kernel"],
                    grid_size=(gx, c1, c1),
                    block_size=(bx, c1, c1),
                    kernel_operands=[A, B, C, n],
                )
        
        for backend in ["execution_engine", "asm"]:
            # Use positive values for sqrt/rsqrt
            a = torch.rand(SIZE, dtype=DTYPE_FP32, device="cuda") + 0.1
            b = torch.rand(SIZE, dtype=DTYPE_FP32, device="cuda") + 0.1
            c_out = torch.zeros(SIZE, dtype=DTYPE_FP32, device="cuda")
            c_ref = torch.sqrt(a) * torch.rsqrt(b) + torch.pow(2.0, a * 0.1)
            
            try:
                m = _MathCombined()
                exe = flydsl.compile(m, backend=backend)
                exe(a, b, c_out, SIZE)
                torch.cuda.synchronize()
                
                is_close, max_diff = compare_results(c_ref, c_out, atol=1e-4, rtol=1e-4, name=f"math_{backend}")
                assert is_close, f"math_combined ({backend}) failed: max_diff={max_diff}"
                print(f"  math_combined size={SIZE} {backend}: PASS")
            except Exception as e:
                pytest.fail(f"math_combined ({backend}) failed: {e}")


# =============================================================================
# Test: FMA (Fused Multiply-Add) pattern
# =============================================================================

class TestFMA:
    """Test FMA pattern: a * b + c."""
    
    @pytest.mark.parametrize("size", [64, 256])
    def test_fma_chain(self, size):
        """Test FMA chain: out = ((a * b + c) * d + e)."""
        gpu_arch = get_rocm_arch()
        SIZE = size
        THREADS = 64
        TILE = 1
        TILE_ELEMS = THREADS * TILE
        S = ir.ShapedType.get_dynamic_size()
        
        class _FMAChain(flir.MlirModule):
            GPU_MODULE_NAME = "fma_chain"
            GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']
            
            @flir.kernel
            def kernel(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                B: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                tid = flir.thread_idx("x")
                bid = flir.block_idx("x")
                tid_linear = tid
                
                thr_layout = flir.make_ordered_layout((THREADS,), order=(0,))
                val_layout = flir.make_ordered_layout((TILE,), order=(0,))
                copy_atom = flir.make_copy_atom(T.f32(), vector_size=1)
                
                tiled_copy = flir.make_tiled_copy_tv(
                    copy_atom, thr_layout, val_layout,
                    thr_shape=(THREADS,), val_shape=(TILE,),
                )
                
                tensor_A = flir.make_tensor(A, shape=(n,), strides=(1,))
                tensor_B = flir.make_tensor(B, shape=(n,), strides=(1,))
                tensor_C = flir.make_tensor(C, shape=(n,), strides=(1,))
                
                tile_shape = (TILE_ELEMS,)
                gA = flir.zipped_divide(tensor_A, tile_shape)
                gB = flir.zipped_divide(tensor_B, tile_shape)
                gC = flir.zipped_divide(tensor_C, tile_shape)
                idC = flir.make_identity_tensor((n,))
                cC = flir.zipped_divide(idC, tile_shape)
                
                blk_coord = (bid,)
                blkA, blkB, blkC = gA[blk_coord], gB[blk_coord], gC[blk_coord]
                blkCrd = cC[blk_coord]
                
                thr_copy = tiled_copy.get_slice(tid_linear)
                thrA = thr_copy.partition_S(blkA)
                thrB = thr_copy.partition_S(blkB)
                thrC = thr_copy.partition_S(blkC)
                thrCrd = thr_copy.partition_S(blkCrd)
                
                val_shape = tiled_copy.val_shape
                frgA = flir.make_fragment_like(thrA, T.f32())
                frgB = flir.make_fragment_like(thrB, T.f32())
                frgC = flir.make_fragment_like(thrC, T.f32())
                
                pred_ty = ir.IntegerType.get_signless(1)
                frgPred = flir.make_rmem_tensor(val_shape, pred_ty)
                for linear in range(val_shape[0]):
                    lin_idx = flir.const_index(linear)
                    coords = thrCrd.coords_from_linear(lin_idx)
                    pred_val = flir.elem_less(coords, (n,))
                    frgPred[tuple(frgPred.offsets_from_linear(lin_idx))] = pred_val
                
                flir.copy(tiled_copy, thrA, frgA, pred=frgPred)
                flir.copy(tiled_copy, thrB, frgB, pred=frgPred)
                
                # out = ((a * b + 1.5) * a + b) - complex FMA chain
                for i in range_constexpr(TILE):
                    idx = flir.const_index(i)
                    a = frgA[(idx,)]
                    b = frgB[(idx,)]
                    
                    t1 = a * b + 1.5    # FMA pattern
                    t2 = t1 * a + b     # Another FMA
                    out = t2 * 0.5 - a  # Scale and subtract
                    
                    frgC[(idx,)] = out
                
                flir.copy(tiled_copy, frgC, thrC, pred=frgPred)
            
            @flir.jit
            def __call__(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                B: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                c1 = arith.index(1)
                c_tile = arith.index(TILE_ELEMS)
                gx = (n + c_tile - c1) // c_tile
                bx = arith.index(THREADS)
                flir.gpu_ext.LaunchFuncOp(
                    [self.GPU_MODULE_NAME, "kernel"],
                    grid_size=(gx, c1, c1),
                    block_size=(bx, c1, c1),
                    kernel_operands=[A, B, C, n],
                )
        
        for backend in ["execution_engine", "asm"]:
            a = torch.randn(SIZE, dtype=DTYPE_FP32, device="cuda")
            b = torch.randn(SIZE, dtype=DTYPE_FP32, device="cuda")
            c_out = torch.zeros(SIZE, dtype=DTYPE_FP32, device="cuda")
            t1 = a * b + 1.5
            t2 = t1 * a + b
            c_ref = t2 * 0.5 - a
            
            try:
                m = _FMAChain()
                exe = flydsl.compile(m, backend=backend)
                exe(a, b, c_out, SIZE)
                torch.cuda.synchronize()
                
                is_close, max_diff = compare_results(c_ref, c_out, atol=1e-5, name=f"fma_{backend}")
                assert is_close, f"fma_chain ({backend}) failed: max_diff={max_diff}"
                print(f"  fma_chain size={SIZE} {backend}: PASS")
            except Exception as e:
                pytest.fail(f"fma_chain ({backend}) failed: {e}")


# =============================================================================
# Test: Clamp pattern (compare + select)
# =============================================================================

class TestClamp:
    """Test clamp pattern using max/min."""
    
    @pytest.mark.parametrize("size", [64, 256])
    def test_clamp(self, size):
        """Test clamp: out = clamp(a, -1.0, 1.0) = min(max(a, -1.0), 1.0)."""
        gpu_arch = get_rocm_arch()
        SIZE = size
        THREADS = 64
        TILE = 1
        TILE_ELEMS = THREADS * TILE
        S = ir.ShapedType.get_dynamic_size()
        
        class _Clamp(flir.MlirModule):
            GPU_MODULE_NAME = "clamp"
            GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']
            
            @flir.kernel
            def kernel(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                tid = flir.thread_idx("x")
                bid = flir.block_idx("x")
                tid_linear = tid
                
                thr_layout = flir.make_ordered_layout((THREADS,), order=(0,))
                val_layout = flir.make_ordered_layout((TILE,), order=(0,))
                copy_atom = flir.make_copy_atom(T.f32(), vector_size=1)
                
                tiled_copy = flir.make_tiled_copy_tv(
                    copy_atom, thr_layout, val_layout,
                    thr_shape=(THREADS,), val_shape=(TILE,),
                )
                
                tensor_A = flir.make_tensor(A, shape=(n,), strides=(1,))
                tensor_C = flir.make_tensor(C, shape=(n,), strides=(1,))
                
                tile_shape = (TILE_ELEMS,)
                gA = flir.zipped_divide(tensor_A, tile_shape)
                gC = flir.zipped_divide(tensor_C, tile_shape)
                idC = flir.make_identity_tensor((n,))
                cC = flir.zipped_divide(idC, tile_shape)
                
                blk_coord = (bid,)
                blkA, blkC = gA[blk_coord], gC[blk_coord]
                blkCrd = cC[blk_coord]
                
                thr_copy = tiled_copy.get_slice(tid_linear)
                thrA = thr_copy.partition_S(blkA)
                thrC = thr_copy.partition_S(blkC)
                thrCrd = thr_copy.partition_S(blkCrd)
                
                val_shape = tiled_copy.val_shape
                frgA = flir.make_fragment_like(thrA, T.f32())
                frgC = flir.make_fragment_like(thrC, T.f32())
                
                pred_ty = ir.IntegerType.get_signless(1)
                frgPred = flir.make_rmem_tensor(val_shape, pred_ty)
                for linear in range(val_shape[0]):
                    lin_idx = flir.const_index(linear)
                    coords = thrCrd.coords_from_linear(lin_idx)
                    pred_val = flir.elem_less(coords, (n,))
                    frgPred[tuple(frgPred.offsets_from_linear(lin_idx))] = pred_val
                
                flir.copy(tiled_copy, thrA, frgA, pred=frgPred)
                
                # clamp(a, -1.0, 1.0) = min(max(a, -1.0), 1.0)
                for i in range_constexpr(TILE):
                    idx = flir.const_index(i)
                    a = frgA[(idx,)]
                    
                    lo = arith.constant(-1.0, type=T.f32())
                    hi = arith.constant(1.0, type=T.f32())
                    t1 = arith.maximum(arith.as_value(a), lo)
                    out = arith.minimum(t1, hi)
                    
                    frgC[(idx,)] = out
                
                flir.copy(tiled_copy, frgC, thrC, pred=frgPred)
            
            @flir.jit
            def __call__(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                c1 = arith.index(1)
                c_tile = arith.index(TILE_ELEMS)
                gx = (n + c_tile - c1) // c_tile
                bx = arith.index(THREADS)
                flir.gpu_ext.LaunchFuncOp(
                    [self.GPU_MODULE_NAME, "kernel"],
                    grid_size=(gx, c1, c1),
                    block_size=(bx, c1, c1),
                    kernel_operands=[A, C, n],
                )
        
        for backend in ["execution_engine", "asm"]:
            a = torch.randn(SIZE, dtype=DTYPE_FP32, device="cuda") * 3  # range roughly -3 to 3
            c_out = torch.zeros(SIZE, dtype=DTYPE_FP32, device="cuda")
            c_ref = torch.clamp(a, -1.0, 1.0)
            
            try:
                m = _Clamp()
                exe = flydsl.compile(m, backend=backend)
                exe(a, c_out, SIZE)
                torch.cuda.synchronize()
                
                is_close, max_diff = compare_results(c_ref, c_out, atol=1e-6, name=f"clamp_{backend}")
                assert is_close, f"clamp ({backend}) failed: max_diff={max_diff}"
                print(f"  clamp size={SIZE} {backend}: PASS")
            except Exception as e:
                pytest.fail(f"clamp ({backend}) failed: {e}")


# =============================================================================
# Test: Type Conversion (f32 <-> i32)
# =============================================================================

class TestTypeConversion:
    """Test type conversions."""
    
    @pytest.mark.parametrize("size", [64, 256])
    def test_f32_to_i32_round(self, size):
        """Test: out = float(int(a * 100)) / 100 (quantize pattern)."""
        gpu_arch = get_rocm_arch()
        SIZE = size
        THREADS = 64
        TILE = 1
        TILE_ELEMS = THREADS * TILE
        S = ir.ShapedType.get_dynamic_size()
        
        class _Quantize(flir.MlirModule):
            GPU_MODULE_NAME = "quantize"
            GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']
            
            @flir.kernel
            def kernel(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                tid = flir.thread_idx("x")
                bid = flir.block_idx("x")
                tid_linear = tid
                
                thr_layout = flir.make_ordered_layout((THREADS,), order=(0,))
                val_layout = flir.make_ordered_layout((TILE,), order=(0,))
                copy_atom = flir.make_copy_atom(T.f32(), vector_size=1)
                
                tiled_copy = flir.make_tiled_copy_tv(
                    copy_atom, thr_layout, val_layout,
                    thr_shape=(THREADS,), val_shape=(TILE,),
                )
                
                tensor_A = flir.make_tensor(A, shape=(n,), strides=(1,))
                tensor_C = flir.make_tensor(C, shape=(n,), strides=(1,))
                
                tile_shape = (TILE_ELEMS,)
                gA = flir.zipped_divide(tensor_A, tile_shape)
                gC = flir.zipped_divide(tensor_C, tile_shape)
                idC = flir.make_identity_tensor((n,))
                cC = flir.zipped_divide(idC, tile_shape)
                
                blk_coord = (bid,)
                blkA, blkC = gA[blk_coord], gC[blk_coord]
                blkCrd = cC[blk_coord]
                
                thr_copy = tiled_copy.get_slice(tid_linear)
                thrA = thr_copy.partition_S(blkA)
                thrC = thr_copy.partition_S(blkC)
                thrCrd = thr_copy.partition_S(blkCrd)
                
                val_shape = tiled_copy.val_shape
                frgA = flir.make_fragment_like(thrA, T.f32())
                frgC = flir.make_fragment_like(thrC, T.f32())
                
                pred_ty = ir.IntegerType.get_signless(1)
                frgPred = flir.make_rmem_tensor(val_shape, pred_ty)
                for linear in range(val_shape[0]):
                    lin_idx = flir.const_index(linear)
                    coords = thrCrd.coords_from_linear(lin_idx)
                    pred_val = flir.elem_less(coords, (n,))
                    frgPred[tuple(frgPred.offsets_from_linear(lin_idx))] = pred_val
                
                flir.copy(tiled_copy, thrA, frgA, pred=frgPred)
                
                # out = float(int(a * 100)) / 100
                for i in range_constexpr(TILE):
                    idx = flir.const_index(i)
                    a = frgA[(idx,)]
                    
                    scaled = a * 100.0
                    as_int = arith.fptosi(T.i32(), arith.as_value(scaled))
                    back_float = arith.sitofp(T.f32(), as_int)
                    out = back_float / 100.0
                    
                    frgC[(idx,)] = out
                
                flir.copy(tiled_copy, frgC, thrC, pred=frgPred)
            
            @flir.jit
            def __call__(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                c1 = arith.index(1)
                c_tile = arith.index(TILE_ELEMS)
                gx = (n + c_tile - c1) // c_tile
                bx = arith.index(THREADS)
                flir.gpu_ext.LaunchFuncOp(
                    [self.GPU_MODULE_NAME, "kernel"],
                    grid_size=(gx, c1, c1),
                    block_size=(bx, c1, c1),
                    kernel_operands=[A, C, n],
                )
        
        for backend in ["execution_engine", "asm"]:
            a = torch.randn(SIZE, dtype=DTYPE_FP32, device="cuda")
            c_out = torch.zeros(SIZE, dtype=DTYPE_FP32, device="cuda")
            c_ref = (a * 100).to(torch.int32).float() / 100.0
            
            try:
                m = _Quantize()
                exe = flydsl.compile(m, backend=backend)
                exe(a, c_out, SIZE)
                torch.cuda.synchronize()
                
                is_close, max_diff = compare_results(c_ref, c_out, atol=1e-6, name=f"quantize_{backend}")
                assert is_close, f"quantize ({backend}) failed: max_diff={max_diff}"
                print(f"  quantize size={SIZE} {backend}: PASS")
            except Exception as e:
                pytest.fail(f"quantize ({backend}) failed: {e}")


# =============================================================================
# Test: Complex Expression (polynomial evaluation)
# =============================================================================

class TestPolynomial:
    """Test polynomial evaluation pattern."""
    
    @pytest.mark.parametrize("size", [64, 256])
    def test_polynomial(self, size):
        """Test Horner's method: out = ((c3*x + c2)*x + c1)*x + c0."""
        gpu_arch = get_rocm_arch()
        SIZE = size
        THREADS = 64
        TILE = 1
        TILE_ELEMS = THREADS * TILE
        # Polynomial coefficients: approximation of exp(x) for small x
        C0, C1, C2, C3 = 1.0, 1.0, 0.5, 0.166667
        S = ir.ShapedType.get_dynamic_size()
        
        class _Polynomial(flir.MlirModule):
            GPU_MODULE_NAME = "polynomial"
            GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']
            
            @flir.kernel
            def kernel(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                tid = flir.thread_idx("x")
                bid = flir.block_idx("x")
                tid_linear = tid
                
                thr_layout = flir.make_ordered_layout((THREADS,), order=(0,))
                val_layout = flir.make_ordered_layout((TILE,), order=(0,))
                copy_atom = flir.make_copy_atom(T.f32(), vector_size=1)
                
                tiled_copy = flir.make_tiled_copy_tv(
                    copy_atom, thr_layout, val_layout,
                    thr_shape=(THREADS,), val_shape=(TILE,),
                )
                
                tensor_A = flir.make_tensor(A, shape=(n,), strides=(1,))
                tensor_C = flir.make_tensor(C, shape=(n,), strides=(1,))
                
                tile_shape = (TILE_ELEMS,)
                gA = flir.zipped_divide(tensor_A, tile_shape)
                gC = flir.zipped_divide(tensor_C, tile_shape)
                idC = flir.make_identity_tensor((n,))
                cC = flir.zipped_divide(idC, tile_shape)
                
                blk_coord = (bid,)
                blkA, blkC = gA[blk_coord], gC[blk_coord]
                blkCrd = cC[blk_coord]
                
                thr_copy = tiled_copy.get_slice(tid_linear)
                thrA = thr_copy.partition_S(blkA)
                thrC = thr_copy.partition_S(blkC)
                thrCrd = thr_copy.partition_S(blkCrd)
                
                val_shape = tiled_copy.val_shape
                frgA = flir.make_fragment_like(thrA, T.f32())
                frgC = flir.make_fragment_like(thrC, T.f32())
                
                pred_ty = ir.IntegerType.get_signless(1)
                frgPred = flir.make_rmem_tensor(val_shape, pred_ty)
                for linear in range(val_shape[0]):
                    lin_idx = flir.const_index(linear)
                    coords = thrCrd.coords_from_linear(lin_idx)
                    pred_val = flir.elem_less(coords, (n,))
                    frgPred[tuple(frgPred.offsets_from_linear(lin_idx))] = pred_val
                
                flir.copy(tiled_copy, thrA, frgA, pred=frgPred)
                
                # Horner's method: ((c3*x + c2)*x + c1)*x + c0
                for i in range_constexpr(TILE):
                    idx = flir.const_index(i)
                    x = frgA[(idx,)]
                    
                    # Step by step for clarity
                    t1 = C3 * x + C2   # c3*x + c2
                    t2 = t1 * x + C1   # (c3*x + c2)*x + c1
                    out = t2 * x + C0  # ((c3*x + c2)*x + c1)*x + c0
                    
                    frgC[(idx,)] = out
                
                flir.copy(tiled_copy, frgC, thrC, pred=frgPred)
            
            @flir.jit
            def __call__(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                c1 = arith.index(1)
                c_tile = arith.index(TILE_ELEMS)
                gx = (n + c_tile - c1) // c_tile
                bx = arith.index(THREADS)
                flir.gpu_ext.LaunchFuncOp(
                    [self.GPU_MODULE_NAME, "kernel"],
                    grid_size=(gx, c1, c1),
                    block_size=(bx, c1, c1),
                    kernel_operands=[A, C, n],
                )
        
        for backend in ["execution_engine", "asm"]:
            a = torch.randn(SIZE, dtype=DTYPE_FP32, device="cuda") * 0.5  # small values for polynomial approx
            c_out = torch.zeros(SIZE, dtype=DTYPE_FP32, device="cuda")
            # Reference: Horner's method
            t1 = C3 * a + C2
            t2 = t1 * a + C1
            c_ref = t2 * a + C0
            
            try:
                m = _Polynomial()
                exe = flydsl.compile(m, backend=backend)
                exe(a, c_out, SIZE)
                torch.cuda.synchronize()
                
                is_close, max_diff = compare_results(c_ref, c_out, atol=1e-5, name=f"poly_{backend}")
                assert is_close, f"polynomial ({backend}) failed: max_diff={max_diff}"
                print(f"  polynomial size={SIZE} {backend}: PASS")
            except Exception as e:
                pytest.fail(f"polynomial ({backend}) failed: {e}")


# =============================================================================
# Test: Abs and Sign pattern
# =============================================================================

class TestAbsSign:
    """Test abs operations."""
    
    @pytest.mark.parametrize("size", [64, 256])
    def test_abs_sign(self, size):
        """Test: out = abs(a) + abs(b)."""
        gpu_arch = get_rocm_arch()
        SIZE = size
        THREADS = 64
        TILE = 1
        TILE_ELEMS = THREADS * TILE
        S = ir.ShapedType.get_dynamic_size()
        
        from flydsl.dialects.ext import math as math_ops
        
        class _AbsSign(flir.MlirModule):
            GPU_MODULE_NAME = "abs_sign"
            GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']
            
            @flir.kernel
            def kernel(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                B: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                tid = flir.thread_idx("x")
                bid = flir.block_idx("x")
                tid_linear = tid
                
                thr_layout = flir.make_ordered_layout((THREADS,), order=(0,))
                val_layout = flir.make_ordered_layout((TILE,), order=(0,))
                copy_atom = flir.make_copy_atom(T.f32(), vector_size=1)
                
                tiled_copy = flir.make_tiled_copy_tv(
                    copy_atom, thr_layout, val_layout,
                    thr_shape=(THREADS,), val_shape=(TILE,),
                )
                
                tensor_A = flir.make_tensor(A, shape=(n,), strides=(1,))
                tensor_B = flir.make_tensor(B, shape=(n,), strides=(1,))
                tensor_C = flir.make_tensor(C, shape=(n,), strides=(1,))
                
                tile_shape = (TILE_ELEMS,)
                gA = flir.zipped_divide(tensor_A, tile_shape)
                gB = flir.zipped_divide(tensor_B, tile_shape)
                gC = flir.zipped_divide(tensor_C, tile_shape)
                idC = flir.make_identity_tensor((n,))
                cC = flir.zipped_divide(idC, tile_shape)
                
                blk_coord = (bid,)
                blkA, blkB, blkC = gA[blk_coord], gB[blk_coord], gC[blk_coord]
                blkCrd = cC[blk_coord]
                
                thr_copy = tiled_copy.get_slice(tid_linear)
                thrA = thr_copy.partition_S(blkA)
                thrB = thr_copy.partition_S(blkB)
                thrC = thr_copy.partition_S(blkC)
                thrCrd = thr_copy.partition_S(blkCrd)
                
                val_shape = tiled_copy.val_shape
                frgA = flir.make_fragment_like(thrA, T.f32())
                frgB = flir.make_fragment_like(thrB, T.f32())
                frgC = flir.make_fragment_like(thrC, T.f32())
                
                pred_ty = ir.IntegerType.get_signless(1)
                frgPred = flir.make_rmem_tensor(val_shape, pred_ty)
                for linear in range(val_shape[0]):
                    lin_idx = flir.const_index(linear)
                    coords = thrCrd.coords_from_linear(lin_idx)
                    pred_val = flir.elem_less(coords, (n,))
                    frgPred[tuple(frgPred.offsets_from_linear(lin_idx))] = pred_val
                
                flir.copy(tiled_copy, thrA, frgA, pred=frgPred)
                flir.copy(tiled_copy, thrB, frgB, pred=frgPred)
                
                # out = abs(a) + abs(b)  (test absf only)
                for i in range_constexpr(TILE):
                    idx = flir.const_index(i)
                    a = frgA[(idx,)]
                    b = frgB[(idx,)]
                    
                    abs_a = flir.math.absf(arith.as_value(a))
                    abs_b = flir.math.absf(arith.as_value(b))
                    out = abs_a + abs_b
                    
                    frgC[(idx,)] = out
                
                flir.copy(tiled_copy, frgC, thrC, pred=frgPred)
            
            @flir.jit
            def __call__(
                self: flir.T.i64,
                A: lambda: T.memref(S, T.f32()),
                B: lambda: T.memref(S, T.f32()),
                C: lambda: T.memref(S, T.f32()),
                n: lambda: T.index(),
            ):
                c1 = arith.index(1)
                c_tile = arith.index(TILE_ELEMS)
                gx = (n + c_tile - c1) // c_tile
                bx = arith.index(THREADS)
                flir.gpu_ext.LaunchFuncOp(
                    [self.GPU_MODULE_NAME, "kernel"],
                    grid_size=(gx, c1, c1),
                    block_size=(bx, c1, c1),
                    kernel_operands=[A, B, C, n],
                )
        
        for backend in ["execution_engine", "asm"]:
            a = torch.randn(SIZE, dtype=DTYPE_FP32, device="cuda")
            b = torch.randn(SIZE, dtype=DTYPE_FP32, device="cuda")
            c_out = torch.zeros(SIZE, dtype=DTYPE_FP32, device="cuda")
            c_ref = torch.abs(a) + torch.abs(b)
            
            try:
                m = _AbsSign()
                exe = flydsl.compile(m, backend=backend)
                exe(a, b, c_out, SIZE)
                torch.cuda.synchronize()
                
                is_close, max_diff = compare_results(c_ref, c_out, atol=1e-6, name=f"abs_sign_{backend}")
                assert is_close, f"abs_sign ({backend}) failed: max_diff={max_diff}"
                print(f"  abs_sign size={SIZE} {backend}: PASS")
            except Exception as e:
                pytest.fail(f"abs_sign ({backend}) failed: {e}")


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
