# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Wide-row softmax: split each row across blocks.

The one-row-per-block kernel register-buffers a whole row. Its grid holds only
``M`` blocks, so a decode-shaped call (a handful of wide rows) leaves the GPU
almost idle, and past ~98k columns the row also spills to scratch (1.2--2.1 KB
per lane at 128k--164k).

This module covers exactly that regime with two streaming passes over the row and
no row-sized register buffer:

  ``partial_kernel``  one block per (row, chunk): the chunk's max and its exp-sum
                      relative to that max, into a small fp32 workspace.
  ``combine_kernel``  one block per (row, chunk): fold the row's chunk partials
                      (``chunks`` values, cheaper than a third dispatch), then
                      normalise its own chunk of the row. The fold is warp-local,
                      so it costs no LDS staging and no block barrier.

Both kernels stream, so the chunk size is a parallelism knob rather than a
register-pressure one. ``N`` and the chunk geometry are build-time constants; the
row count and the grid stay runtime values. Traffic is one read of ``A`` for the
partials and one read plus one write for the normalising pass.

The partials live in a caller-provided fp32 workspace of
``split_softmax_workspace_elems(M, N, dtype_str)`` elements, as in the split-K
FlashAttention launchers. ``softmax_autotuned`` allocates it and sends the shapes
``use_split_softmax`` picks here; ``build_softmax_module`` keeps the
register-buffered kernel at every width.

A single-dispatch variant was measured on gfx950 and is not used. Its per-row
barrier (publish a chunk partial, take an arrival ticket, poll, fold) costs
1.5--4 us of pure latency: two dependent L2 round trips plus the wait for the
slowest chunk. That held whether every block repeated the fold, whether chunk 0
published the fold once, and whether the fold stayed warp-local, and it is the
same order as the ~2.5 us a second dispatch costs -- so the fused path is a wash
on the widest decode rows and 1.5--3x worse once several rows are in flight.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp
from kernels.common.kernels_common import LOG2E, dtype_to_elem_type, get_warp_size

# The split path reads each row from global memory once more than the register-
# buffered kernel, so it wins only where that kernel is not bandwidth-bound.
# Measured on gfx950 (MI355X, cold cache):
#   - up to WIDE_ROW_MIN_N columns it loses at every row count (0.60-0.83x at
#     8193-32768 columns, 0.80x at 1024 x 65536 f32);
#   - past it, a call of at most DECODE_MAX_ROWS rows leaves most CUs idle under
#     one block per row, and the split path is 1.2-3.5x at 1-64 rows;
#   - with more rows it wins from ALL_ROWS_SPLIT_N columns on: 1.04-1.11x at
#     106496 f32 over 256-8192 rows, where the register kernel spills 364 B per
#     lane (0.91-0.98x at 8192 rows up to 104448), and 1.38-1.96x at 77824
#     bf16/f16 over 128-8192 rows (0.83-0.95x at 73728).
WIDE_ROW_MIN_N = 65536
DECODE_MAX_ROWS = 64
ALL_ROWS_SPLIT_N = {"f32": 106496, "f16": 77824, "bf16": 77824}


def use_split_softmax(M: int, N: int, dtype_str: str) -> bool:
    """Whether an ``M x N`` softmax should take the split kernel (thresholds above)."""
    return N >= ALL_ROWS_SPLIT_N[dtype_str] or (N > WIDE_ROW_MIN_N and M <= DECODE_MAX_ROWS)


# Vectors per lane per chunk. Two keeps a block's loads overlapped while leaving
# enough blocks per row (≈N/(2·BLOCK_THREADS·VEC)) for a one-row call to fill the
# machine. Measured within 3% over 1/2/4/8 on gfx950: one vector wins on the
# one-row calls by ~4% and loses the same on the 24-row ones, two is the tie.
VECS_PER_THREAD = 2
MAX_VECS_PER_THREAD = 8


def _elem_bits(dtype_str: str) -> int:
    return 32 if dtype_str == "f32" else 16


def _vec_width(dtype_str: str) -> int:
    """Columns per 128-bit transaction: the buffer copy moves one per lane."""
    return 128 // _elem_bits(dtype_str)


def chunk_geometry(N: int, dtype_str: str, BLOCK_THREADS: int) -> tuple[int, int]:
    """(vectors per lane, chunks per row) for a wide row of ``N`` columns."""
    vw = _vec_width(dtype_str)
    vecs_total = N // vw
    if vecs_total <= 0:
        return 1, 1
    vecs_per_thread = max(1, min(MAX_VECS_PER_THREAD, VECS_PER_THREAD))
    vecs_per_chunk = BLOCK_THREADS * vecs_per_thread
    return vecs_per_thread, -(-vecs_total // vecs_per_chunk)


def split_softmax_workspace_elems(M: int, N: int, dtype_str: str, BLOCK_THREADS: int = 256) -> int:
    """fp32 elements of the workspace a launch over ``M`` rows needs: a max and a sum per chunk."""
    _, chunks = chunk_geometry(N, dtype_str, BLOCK_THREADS)
    return 2 * chunks * M


def _load_vec(copy_atom, vec_width, elem_dtype, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.copy(copy_atom, fx.slice(div_tensor, (None, idx)), r)
    return fx.memref_load_vec(r)


def _store_vec(copy_atom, vec_width, elem_dtype, val, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.memref_store_vec(val, r)
    fx.copy(copy_atom, r, fx.slice(div_tensor, (None, idx)))


def build_split_softmax_module(N: int, dtype_str: str, BLOCK_THREADS: int = 256):
    """Build a two-pass wide-row softmax launcher ``launch(A, C, W, m_in, stream)``.

    ``W`` is an fp32 workspace of at least ``split_softmax_workspace_elems(m_in, N,
    dtype_str, BLOCK_THREADS)`` elements; nothing is carried in it between calls.
    ``m_in`` is a runtime grid value.
    """
    vecs_per_thread, chunks = chunk_geometry(N, dtype_str, BLOCK_THREADS)
    vw = _vec_width(dtype_str)
    elem_bits = _elem_bits(dtype_str)
    elem_dtype = dtype_to_elem_type(dtype_str)
    vecs_total = N // vw
    tail = N - vecs_total * vw
    vecs_per_chunk = BLOCK_THREADS * vecs_per_thread
    WARP = get_warp_size()
    n_fold_iters = -(-chunks // WARP)

    BlockReduce = fx.coop.BlockReduce[fx.Float32, BLOCK_THREADS]
    SharedStorage = BlockReduce.SharedStorage

    fm_fast = arith.FastMathFlags.fast
    c_zero_f = fx.Float32(0.0)
    c_neg_inf = fx.Float32(float("-inf"))

    @flyc.kernel
    def partial_kernel(A: fx.Tensor, W: fx.Tensor, MIn: fx.Int32):
        row = fx.block_idx.y
        cid = fx.block_idx.x
        tid = fx.thread_idx.x

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        def block_reduce(val, op):
            with fx.fastmath(fm_fast if op == ReductionOp.ADD else arith.FastMathFlags.none):
                return BlockReduce(val, op, storage=lds)

        A_buf = fx.rocdl.make_buffer_tensor(A)
        W_buf = fx.rocdl.make_buffer_tensor(W)
        row_a = fx.slice(A_buf, (row, None))
        a_div = fx.logical_divide(row_a, fx.make_layout(vw, 1))
        load_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

        # Pass 1: this chunk's max, from the vectors this lane owns and (in the
        # last chunk) the scalar tail. Out-of-range vectors read lane 0 and are
        # masked out of the reduction.
        base = cid * vecs_per_chunk
        thread_max = c_neg_inf
        row_buffer = []
        for j in range_constexpr(vecs_per_thread):
            idx = base + j * BLOCK_THREADS + tid
            is_valid = idx < vecs_total
            idx_safe = is_valid.select(idx, 0)
            x = _load_vec(load_atom, vw, elem_dtype, a_div, idx_safe).to(fx.Float32)
            row_buffer.append(x)
            red_max = x.reduce(ReductionOp.MAX)
            thread_max = fx.max(thread_max, is_valid.select(red_max, c_neg_inf))

        tail_x = c_zero_f
        if const_expr(tail > 0):
            if cid == chunks - 1:
                if tid < tail:
                    x_t = row_a[vecs_total * vw + tid]
                    tail_x = x_t if dtype_str == "f32" else x_t.to(fx.Float32)
                    thread_max = fx.max(thread_max, tail_x)

        block_max = block_reduce(thread_max, ReductionOp.MAX)

        # Pass 2: this chunk's exp-sum relative to its own max, so the pair
        # (block_max, block_sum) carries no assumption about the other chunks.
        thread_sum = c_zero_f
        for j in range_constexpr(vecs_per_thread):
            idx = base + j * BLOCK_THREADS + tid
            is_valid = idx < vecs_total
            exp_val = fmath.exp2((row_buffer[j] - block_max) * LOG2E, fastmath=fm_fast)
            red_sum = exp_val.reduce(ReductionOp.ADD, fastmath=fm_fast)
            thread_sum = thread_sum + is_valid.select(red_sum, c_zero_f)

        if const_expr(tail > 0):
            if cid == chunks - 1:
                if tid < tail:
                    exp_t = fmath.exp2((tail_x - block_max) * LOG2E, fastmath=fm_fast)
                    thread_sum = thread_sum + exp_t

        block_sum = block_reduce(thread_sum, ReductionOp.ADD)

        if tid == 0:
            W_buf[row * chunks + cid] = block_max
            W_buf[MIn * chunks + row * chunks + cid] = block_sum

    @flyc.kernel
    def combine_kernel(A: fx.Tensor, C: fx.Tensor, W: fx.Tensor, MIn: fx.Int32):
        row = fx.block_idx.y
        cid = fx.block_idx.x
        tid = fx.thread_idx.x

        A_buf = fx.rocdl.make_buffer_tensor(A)
        C_buf = fx.rocdl.make_buffer_tensor(C)
        W_buf = fx.rocdl.make_buffer_tensor(W)
        row_a = fx.slice(A_buf, (row, None))
        row_c = fx.slice(C_buf, (row, None))
        a_div = fx.logical_divide(row_a, fx.make_layout(vw, 1))
        c_div = fx.logical_divide(row_c, fx.make_layout(vw, 1))
        load_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

        # Fold this row's chunk partials. Every block of the row repeats the fold,
        # and every wave within the block repeats it too: the partials are one
        # coalesced load each way, so folding per wave costs no LDS staging, no
        # block barrier and no broadcast, and each lane ends up holding the result.
        # Measured 3.4 -> 2.7 us on the combine kernel of a 1x129280 f32 call.
        m_base = row * chunks
        s_base = MIn * chunks + m_base
        lane = tid % WARP
        thread_max = c_neg_inf
        for j in range_constexpr(n_fold_iters):
            i = j * WARP + lane
            if i < chunks:
                thread_max = fx.max(thread_max, W_buf[m_base + i])
        row_max = fx.coop.warp_reduce(thread_max, ReductionOp.MAX, width=WARP)

        thread_sum = c_zero_f
        for j in range_constexpr(n_fold_iters):
            i = j * WARP + lane
            if i < chunks:
                scale = fmath.exp2((W_buf[m_base + i] - row_max) * LOG2E, fastmath=fm_fast)
                thread_sum = thread_sum + W_buf[s_base + i] * scale
        with fx.fastmath(fm_fast):
            row_sum = fx.coop.warp_reduce(thread_sum, ReductionOp.ADD, width=WARP)
        inv_sum = 1.0 / row_sum

        base = cid * vecs_per_chunk
        for j in range_constexpr(vecs_per_thread):
            idx = base + j * BLOCK_THREADS + tid
            if idx < vecs_total:
                x = _load_vec(load_atom, vw, elem_dtype, a_div, idx).to(fx.Float32)
                norm = fmath.exp2((x - row_max) * LOG2E, fastmath=fm_fast) * inv_sum
                out_e = norm if dtype_str == "f32" else norm.to(elem_dtype)
                _store_vec(store_atom, vw, elem_dtype, out_e, c_div, idx)

        if const_expr(tail > 0):
            if cid == chunks - 1:
                if tid < tail:
                    x_t = row_a[vecs_total * vw + tid]
                    x_f = x_t if dtype_str == "f32" else x_t.to(fx.Float32)
                    norm = fmath.exp2((x_f - row_max) * LOG2E, fastmath=fm_fast) * inv_sum
                    row_c[vecs_total * vw + tid] = norm if dtype_str == "f32" else norm.to(elem_dtype)

    @flyc.jit
    def launch_split(A: fx.Tensor, C: fx.Tensor, W: fx.Tensor, m_in: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        # Two dispatches on the caller's stream; the combine pass reads every
        # chunk partial the first one wrote.
        partial_kernel(A, W, m_in).launch(grid=(chunks, m_in, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)
        combine_kernel(A, C, W, m_in).launch(grid=(chunks, m_in, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_split
