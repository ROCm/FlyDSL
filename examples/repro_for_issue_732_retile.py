"""Repro for issue #732: retile() crashes when load/store use different copy atoms.

load atom = UniversalCopy128b (4 fp32 per op), store atom = UniversalAtomicAdd
(1 fp32 per op). Re-viewing the load-partitioned fragment into the store's value
layout is the tricky part.

Usage (run from FlyDSL repo root):
    python examples/repro_for_issue_732_retile.py case1   # error: unsupported cmpxchg
    python examples/repro_for_issue_732_retile.py case2   # IntTupleUtils.h:865 assert abort
    python examples/repro_for_issue_732_retile.py case3   # works, B == A.sum()
"""
import sys
sys.path.insert(0, "build-fly/python_packages")
sys.path.insert(1, ".")

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

CASE = sys.argv[1] if len(sys.argv) > 1 else "case2"


@flyc.kernel
def sum_kernel(A: fx.Tensor, B: fx.Tensor, tileM: fx.Constexpr[int], tileN: fx.Constexpr[int]):
    bx = fx.block_idx.x
    by = fx.block_idx.y
    tid = fx.thread_idx.x

    A = A[None, bx, by]

    tv_tilemn = (32, 64)
    tv_layout = fx.make_layout(((8, 32), 8), ((256, 1), 32))  # each thread loads 8 fp32

    load_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
    store_atom = fx.make_copy_atom(fx.UniversalAtomicAdd(fx.Float32), fx.Float32)

    tiled_load = fx.make_tiled_copy(load_atom, tv_layout, tv_tilemn)
    tiled_store = fx.make_tiled_copy(store_atom, tv_layout, tv_tilemn)

    broadcasted_B = fx.composition(B, fx.make_layout((tileM, tileN), (0, 0)))

    part_A = tiled_load.get_slice(tid).partition_S(A)

    ts = tiled_store.get_slice(tid)
    part_B = ts.partition_D(broadcasted_B)

    for bm in fx.range_constexpr(tileM // tv_tilemn[0]):
        for bn in fx.range_constexpr(tileN // tv_tilemn[1]):
            frag = fx.make_fragment_like(part_A[None, bm, bn])
            fx.copy(load_atom, part_A[None, bm, bn], frag)

            # frag: Tensor<f32, register, (4,2):(1,4)>
            # part_B[None, bm, bn]: Tensor<f32, global, (1,8):(0,0)>
            print("before: ", frag, part_B[None, bm, bn])

            if CASE == "case1":
                # no retile -> compiler crashes: error: unsupported cmpxchg
                pass
            elif CASE == "case2":
                # retile -> abort: IntTupleUtils.h:865
                #   Assertion `tRank >= guideRank && "Mismatched ranks in intTupleZip2By"'
                frag = ts.retile(frag)
            elif CASE == "case3":
                # hard-coded view works correctly
                frag = fx.make_view(fx.get_iter(frag), fx.make_layout((1, 8), (0, 1)))

            fx.copy(store_atom, frag, part_B[None, bm, bn])


@flyc.jit
def sum(A: fx.Tensor, B: fx.Tensor, tileM: fx.Constexpr[int], tileN: fx.Constexpr[int]):
    A = fx.tiled_divide(A, (tileM, tileN))
    grid_x = fx.get_scalar(A.shape[1])
    grid_y = fx.get_scalar(A.shape[2])
    sum_kernel(A, B, tileM, tileN).launch(
        grid=(grid_x, grid_y, 1),
        block=(256, 1, 1),
    )


torch.manual_seed(0)
A = torch.randn((64, 64), device="cuda", dtype=torch.float32)
B = torch.zeros(1, device="cuda", dtype=torch.float32)
print(f"=== running {CASE} ===")
sum(A, B, 64, 64)
print("ref A.sum() =", A.sum())
print("B =", B)
