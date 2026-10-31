#!/usr/bin/env python3
"""Three if-condition cases — compare generated IR:
  Case 1: const_expr(cond)       → native Python if, NO scf.if
  Case 2: Python-traceable cond  → scf_if_dispatch → _is_dynamic=False → short-circuit, NO scf.if
  Case 3: dynamic cond (tid)     → scf_if_dispatch → _is_dynamic=True  → scf.IfOp in IR

Run with:
  FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 python examples/test_if_three_cases.py
Then compare the 00_origin.mlir files for each kernel.
"""

import os, sys
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx

VEC = 4
BLOCK = 64
TILE = BLOCK * VEC


# ============================================================
# Case 1: const_expr — AST rewriter keeps native Python if
# ============================================================
@flyc.kernel
def kernel_case1_constexpr(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    use_add: fx.Constexpr[bool],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    tile_elems = block_dim * vec_width

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    tA = fx.logical_divide(A, fx.make_layout(tile_elems, 1))
    tB = fx.logical_divide(B, fx.make_layout(tile_elems, 1))
    tC = fx.logical_divide(C, fx.make_layout(tile_elems, 1))
    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))
    tC = fx.slice(tC, (None, bid))
    tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
    tB = fx.logical_divide(tB, fx.make_layout(vec_width, 1))
    tC = fx.logical_divide(tC, fx.make_layout(vec_width, 1))

    reg_ty = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(vec_width, 1), fx.AddressSpace.Register)
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    rA = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))
    rB = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))
    rC = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))

    fx.copy_atom_call(copy_atom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copy_atom, fx.slice(tB, (None, tid)), rB)
    vA = fx.memref_load_vec(rA)
    vB = fx.memref_load_vec(rB)

    # const_expr: native Python if, only one branch traced
    if const_expr(use_add):
        vOut = fx.arith.addf(vA, vB)
    else:
        vOut = fx.arith.subf(vA, vB)

    fx.memref_store_vec(vOut, rC)
    fx.copy_atom_call(copy_atom, rC, fx.slice(tC, (None, tid)))


# ============================================================
# Case 2: Python-traceable — use_add is Constexpr[bool]
#         but NOT wrapped in const_expr, so visit_If rewrites
#         it to scf_if_dispatch.
#         At runtime: _is_dynamic(bool) → False → short-circuit.
# ============================================================
@flyc.kernel
def kernel_case2_python_traceable(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    use_add: fx.Constexpr[bool],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    tile_elems = block_dim * vec_width

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    tA = fx.logical_divide(A, fx.make_layout(tile_elems, 1))
    tB = fx.logical_divide(B, fx.make_layout(tile_elems, 1))
    tC = fx.logical_divide(C, fx.make_layout(tile_elems, 1))
    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))
    tC = fx.slice(tC, (None, bid))
    tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
    tB = fx.logical_divide(tB, fx.make_layout(vec_width, 1))
    tC = fx.logical_divide(tC, fx.make_layout(vec_width, 1))

    reg_ty = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(vec_width, 1), fx.AddressSpace.Register)
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    rA = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))
    rB = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))
    rC = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))

    fx.copy_atom_call(copy_atom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copy_atom, fx.slice(tB, (None, tid)), rB)
    vA = fx.memref_load_vec(rA)
    vB = fx.memref_load_vec(rB)
    vOut = fx.arith.addf(vA, vB)  # initialize before if

    # use_add is a plain Python bool at tracing time,
    # NOT wrapped in const_expr.
    if use_add:
        vOut = fx.arith.addf(vA, vB)
    else:
        vOut = fx.arith.subf(vA, vB)

    fx.memref_store_vec(vOut, rC)
    fx.copy_atom_call(copy_atom, rC, fx.slice(tC, (None, tid)))


# ============================================================
# Case 3: Dynamic — tid is ir.Value → scf.IfOp
# ============================================================
@flyc.kernel
def kernel_case3_dynamic(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    tile_elems = block_dim * vec_width

    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

    tA = fx.logical_divide(A, fx.make_layout(tile_elems, 1))
    tB = fx.logical_divide(B, fx.make_layout(tile_elems, 1))
    tC = fx.logical_divide(C, fx.make_layout(tile_elems, 1))
    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))
    tC = fx.slice(tC, (None, bid))
    tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
    tB = fx.logical_divide(tB, fx.make_layout(vec_width, 1))
    tC = fx.logical_divide(tC, fx.make_layout(vec_width, 1))

    reg_ty = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(vec_width, 1), fx.AddressSpace.Register)
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    rA = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))
    rB = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))
    rC = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))

    fx.copy_atom_call(copy_atom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copy_atom, fx.slice(tB, (None, tid)), rB)
    vA = fx.memref_load_vec(rA)
    vB = fx.memref_load_vec(rB)
    vOut = fx.arith.addf(vA, vB)  # initialize before if

    # tid is dynamic (ir.Value) → scf.IfOp
    if (tid % 2) == 0:
        vOut = fx.arith.addf(vA, vB)
    else:
        vOut = fx.arith.subf(vA, vB)

    fx.memref_store_vec(vOut, rC)
    fx.copy_atom_call(copy_atom, rC, fx.slice(tC, (None, tid)))


# ============================================================
# Launchers
# ============================================================
@flyc.jit
def launch_case1(A: fx.Tensor, B: fx.Tensor, C, n: fx.Int32,
                 block_dim: fx.Constexpr[int], vec_width: fx.Constexpr[int],
                 use_add: fx.Constexpr[bool], stream: fx.Stream = fx.Stream(None)):
    a = fx.arith.index(0)
    # b = fx.Int32(a)
    c = fx.Index(a)
    # tile = block_dim * vec_width
    # grid_x = (n + tile - 1) // tile
    # kernel_case1_constexpr(A, B, C, block_dim, vec_width, use_add).launch(
    #     grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream)

@flyc.jit
def launch_case2(A: fx.Tensor, B: fx.Tensor, C, n: fx.Int32,
                 block_dim: fx.Constexpr[int], vec_width: fx.Constexpr[int],
                 use_add: fx.Constexpr[bool], stream: fx.Stream = fx.Stream(None)):
    tile = block_dim * vec_width
    grid_x = (n + tile - 1) // tile
    kernel_case2_python_traceable(A, B, C, block_dim, vec_width, use_add).launch(
        grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream)

@flyc.jit
def launch_case3(A: fx.Tensor, B: fx.Tensor, C, n: fx.Int32,
                 block_dim: fx.Constexpr[int], vec_width: fx.Constexpr[int],
                 stream: fx.Stream = fx.Stream(None)):
    tile = block_dim * vec_width
    grid_x = (n + tile - 1) // tile
    kernel_case3_dynamic(A, B, C, block_dim, vec_width).launch(
        grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream)


if __name__ == "__main__":
    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

    num_blocks = 4
    size = TILE * num_blocks

    a = torch.randn(size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)
    tA = flyc.from_dlpack(a).mark_layout_dynamic(leading_dim=0, divisibility=VEC)
    stream = torch.cuda.Stream()

    print("=" * 60)
    print("Case 1: const_expr(use_add=True) — expect NO scf.if")
    print("=" * 60)
    launch_case1(tA, b, c, size, BLOCK, VEC, True, stream=stream)
    torch.cuda.synchronize()

    # print()
    # print("=" * 60)
    # print("Case 2: Python-traceable (use_add=True, no const_expr)")
    # print("=" * 60)
    # c.zero_()
    # launch_case2(tA, b, c, size, BLOCK, VEC, True, stream=stream)
    # torch.cuda.synchronize()

    # print()
    # print("=" * 60)
    # print("Case 3: Dynamic cond (tid % 2 == 0) — expect scf.if")
    # print("=" * 60)
    # c.zero_()
    # launch_case3(tA, b, c, size, BLOCK, VEC, stream=stream)
    # torch.cuda.synchronize()

    print()
    print("Done. Check FLYDSL_DUMP_IR output for IR files.")
