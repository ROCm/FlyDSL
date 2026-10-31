#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Minimal repro in flyc style: host + @flyc.jit + @flyc.kernel.

Focus:
- Keep nested function inside kernel.
- Replace closure access with explicit parameter passing.
- Verify runtime branch result is correct.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def pick_with_dynamic_if(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x

    A = fx.rocdl.make_buffer_tensor(A)
    tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
    tB = fx.logical_divide(B, fx.make_layout(block_dim, 1))
    tC = fx.logical_divide(C, fx.make_layout(block_dim, 1))

    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))
    tC = fx.slice(tC, (None, bid))
    tA = fx.logical_divide(tA, fx.make_layout(1, 1))
    tB = fx.logical_divide(tB, fx.make_layout(1, 1))
    tC = fx.logical_divide(tC, fx.make_layout(1, 1))

    reg_ty = fx.MemRefType.get(
        fx.T.f32(),
        fx.LayoutType.get(1, 1),
        fx.AddressSpace.Register,
    )

    copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    copy_atom_buffer = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

    rA = fx.memref_alloca(reg_ty, fx.make_layout(1, 1))
    rB = fx.memref_alloca(reg_ty, fx.make_layout(1, 1))
    rC = fx.memref_alloca(reg_ty, fx.make_layout(1, 1))

    fx.copy_atom_call(copy_atom_buffer, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copy_atom, fx.slice(tB, (None, tid)), rB)

    va = fx.memref_load_vec(rA)
    vb = fx.memref_load_vec(rB)

    def choose_value(x, lane, rhs):
        # Explicit parameter rhs avoids implicit closure capture in nested function.
        out = x
        if lane == fx.Int32(0):
            out = rhs
        return out

    vc = choose_value(va, tid, vb)
    fx.memref_store_vec(vc, rC)
    fx.copy_atom_call(copy_atom, rC, fx.slice(tC, (None, tid)))


@flyc.jit
def run_pick_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C,
    n: fx.Int32,
    const_n: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    block_dim = 64
    grid_x = (n + block_dim - 1) // block_dim
    pick_with_dynamic_if(A, B, C, block_dim).launch(
        grid=(grid_x, 1, 1),
        block=[block_dim, 1, 1],
        stream=stream,
    )


def host():
    n = 128
    A = torch.arange(n, dtype=torch.float32, device="cuda")
    B = torch.arange(n, dtype=torch.float32, device="cuda") + 1000.0
    C = torch.zeros(n, dtype=torch.float32, device="cuda")

    tA = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=0, divisibility=4)
    tB = flyc.from_dlpack(B).mark_layout_dynamic(leading_dim=0, divisibility=4)
    stream = torch.cuda.Stream()

    run_pick_kernel(tA, tB, C, n, n, stream=stream)
    torch.cuda.synchronize()

    block_dim = 64
    expect = A.clone()
    expect[0::block_dim] = B[0::block_dim]
    equal_mask = C.eq(expect)
    ok = bool(equal_mask.all().item())
    print("Result correct:", ok)
    if not ok:
        mismatch_idx = torch.nonzero(~equal_mask, as_tuple=False).flatten()
        print("Mismatch count:", int(mismatch_idx.numel()))
        print("Mismatch indices:", mismatch_idx.cpu().tolist())
        print("C[mismatch] =", C[mismatch_idx].cpu().tolist())
        print("Expect[mismatch] =", expect[mismatch_idx].cpu().tolist())
    assert ok, "Kernel result mismatch."


if __name__ == "__main__":
    host()
