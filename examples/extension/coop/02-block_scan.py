# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Stream compaction with ``fx.coop.BlockScan``.

Keeping the positive elements of an array, in order and with no holes, is the
problem an *exclusive* scan exists to solve: a thread that wants to append needs
to know how many elements every thread in front of it is appending, and that is
exactly ``exclusive`` over the per-element keep flags.

  1. **Exclusive, not inclusive.** ``inclusive`` would count the thread's own
     element, which is the slot after the one it wants.
  2. **A ``Vector`` scans as a run of consecutive elements.** Thread ``t`` owns
     items ``t * ITEMS .. t * ITEMS + ITEMS - 1`` of the block's sequence, so a
     single call gives every one of them its own output slot -- the thread-local
     scan and the block-wide one compose into one result.
  3. **``BlockScan`` is specialized on what it folds**, which here is the keep
     flags rather than the elements they belong to.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

BLOCK = 256
ITEMS = 4
N = BLOCK * ITEMS


@flyc.kernel
def compact_positive(A: fx.Pointer, Out: fx.Tensor):
    block_scan = fx.coop.BlockScan[fx.Int32, fx.known_block_size()]
    storage = fx.SharedAllocator().allocate(block_scan.SharedStorage).peek()

    items = (A + fx.thread_idx.x * ITEMS).load(fx.Int32x4)
    flags = (items > 0).select(1, 0)

    # slots[i] == how many elements before this one are kept == where it goes.
    slots = block_scan.exclusive(flags, fx.ReductionOp.ADD, storage=storage)

    for i in fx.range_constexpr(ITEMS):
        if items[i] > 0:
            Out[slots[i]] = items[i]


@flyc.jit
def compact(A: fx.Pointer, Out: fx.Tensor):
    compact_positive(A, Out).launch(grid=(1, 1, 1), block=(BLOCK, 1, 1))


A = torch.randint(-8, 8, (N,), dtype=torch.int32, device="cuda")
Out = torch.zeros(N, dtype=torch.int32, device="cuda")

compact(flyc.from_c_void_p(fx.Int32, A.data_ptr()), Out)
torch.cuda.synchronize()

expected = A.cpu()[A.cpu() > 0]
if torch.equal(Out.cpu()[: expected.numel()], expected):
    print(f"PASS ({expected.numel()} of {N} kept)")
else:
    print("FAIL")
    raise SystemExit(1)
