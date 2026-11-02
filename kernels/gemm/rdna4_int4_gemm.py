#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""INT4 WMMA GEMM for RDNA4 (gfx120*, wave32).

Computes ``C[M, N] = A[M, K] @ B_T[N, K].T`` with signed INT4 inputs and an
exact INT32 accumulator. A and B_T are packed 8 nibbles per int32,
little-endian, with shapes ``[M, K/8]`` and ``[N, K/8]``.

A workgroup computes one ``BLOCK_M x BLOCK_N`` output tile with several
waves. Each wave owns ``reg_m x reg_n`` of the gfx120 IU4 16x16x32 atom.
Packed K-slices of A and B sit in double-buffered LDS: a packed word is
loaded from global memory once per K-tile and reused across the waves and
repeats that share it. ``fx.Int4`` is not Storable, so LDS holds int32 words
and each lane bitcasts its pair at ``mma_atom_call_ssa``.

LDS row stride is padded so one wave's 16x4 dword read touches each of the
32 LDS banks twice, which is the minimum for a 64-bit read from every lane.
The next K-tile's global loads are issued across the current tile's WMMA
issues, and land in the other LDS buffer.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.runtime.device import get_rocm_arch

WMMA_M = 16
WMMA_N = 16
WMMA_K = 32
WAVE_SIZE = 32
LDS_BYTES = 64 * 1024
_MAX_ATOM = 8  # 128 rows or columns per workgroup


def _padded_stride(packed_k):
    """Smallest row stride >= packed_k whose 16x4 dword read is 2-way banked.

    Bank = (row * stride + col) % 32. A stride congruent to 4 (mod 8) makes
    the 16 rows alias in pairs and the 4 consecutive columns cover every bank
    twice.
    """
    pad = (4 - packed_k % 8) % 8
    stride = packed_k + pad
    return stride


def _pipeline_schedule(ncopy, k_iters, inflight=2):
    """Interleave next-tile copies with this tile's WMMA K-steps.

    Each op is ``(kind, arg, slot)``. ``load``/``store`` arg is the copy
    index; ``slot`` is the register buffer it occupies. At most ``inflight``
    global vectors are live, and every load is issued before the WMMA group
    that can hide it when there are enough K-steps.
    """
    if ncopy < 1 or k_iters < 1:
        raise ValueError(f"need at least one copy and one K-step, got {ncopy}, {k_iters}")
    inflight = max(1, min(inflight, ncopy))
    ops = []
    issued = 0
    stored = 0

    def _load():
        nonlocal issued
        ops.append(("load", issued, issued % inflight))
        issued += 1

    def _store():
        nonlocal stored
        ops.append(("store", stored, stored % inflight))
        stored += 1

    while issued < ncopy and issued - stored < inflight:
        _load()
    for kt in range(k_iters):
        ops.append(("wmma", kt, 0))
        for _ in range(inflight):
            outstanding = issued - stored
            if stored < issued and (issued == ncopy or outstanding >= inflight):
                _store()
            if issued < ncopy and issued - stored < inflight:
                _load()
    while issued < ncopy or stored < ncopy:
        if issued < ncopy and issued - stored < inflight:
            _load()
        elif stored < issued:
            _store()
        else:
            break

    return tuple(ops), inflight


def choose_int4_tile(M, N, K):
    """Pick a workgroup tile that divides ``M, N, K`` and fits in 64KB of LDS.

    Returns ``(block_m, block_n, block_k, waves_m, waves_n, reg_m, reg_n)``.
    """
    m_atoms = M // WMMA_M
    n_atoms = N // WMMA_N
    best = None
    best_key = None
    for waves_m in (2, 4, 3, 1):
        for reg_m in (4, 3, 2, 1):
            block_m_atoms = waves_m * reg_m
            if block_m_atoms > _MAX_ATOM or m_atoms % block_m_atoms:
                continue
            for waves_n in (2, 4, 3, 1):
                for reg_n in (4, 3, 2, 1):
                    block_n_atoms = waves_n * reg_n
                    if block_n_atoms > _MAX_ATOM or n_atoms % block_n_atoms:
                        continue
                    block_m = WMMA_M * block_m_atoms
                    block_n = WMMA_N * block_n_atoms
                    for block_k in (256, 128, 64, 32):
                        if K % block_k:
                            continue
                        stride = _padded_stride(block_k // 8)
                        lds_bytes = 2 * (block_m + block_n) * stride * 4
                        if lds_bytes > LDS_BYTES:
                            continue
                        # Prefer larger output and K tiles, then waves in both
                        # dimensions, register reuse, and total wave count.
                        key = (
                            block_m * block_n,
                            block_k,
                            (waves_m > 1) + (waves_n > 1),
                            reg_m * reg_n,
                            waves_m * waves_n,
                        )
                        if best_key is None or key > best_key:
                            best_key = key
                            best = (block_m, block_n, block_k, waves_m, waves_n, reg_m, reg_n)
    if best is None:
        raise ValueError(f"no INT4 tile fits {M}x{N}x{K}")
    return best


def create_wmma_int4_gemm_module(M: int, N: int, K: int):
    """Specialize a gfx120 signed-INT4 GEMM for one static shape."""
    if M % WMMA_M or N % WMMA_N or K % WMMA_K:
        raise ValueError(f"INT4 GEMM needs M%{WMMA_M}==0, N%{WMMA_N}==0, K%{WMMA_K}==0; got {M}x{N}x{K}")

    BLOCK_M, BLOCK_N, BLOCK_K, WAVES_M, WAVES_N, REG_M, REG_N = choose_int4_tile(M, N, K)
    THREADS = WAVES_M * WAVES_N * WAVE_SIZE
    PACKED_K = K // 8
    PK = BLOCK_K // 8
    STRIDE = _padded_stride(PK)
    VECS_PER_ROW = PK // 4
    NVEC_A = BLOCK_M * VECS_PER_ROW
    NVEC_B = BLOCK_N * VECS_PER_ROW
    ITERS_A = (NVEC_A + THREADS - 1) // THREADS
    ITERS_B = (NVEC_B + THREADS - 1) // THREADS
    NCOPY = ITERS_A + ITERS_B
    A_ELEMS = BLOCK_M * STRIDE
    B_ELEMS = BLOCK_N * STRIDE
    STAGE_ELEMS = A_ELEMS + B_ELEMS
    LDS_ELEMS = 2 * STAGE_ELEMS
    K_ITERS = BLOCK_K // WMMA_K
    NUM_K_TILES = K // BLOCK_K
    N_ACC = REG_M * REG_N
    if NUM_K_TILES > 1:
        SCHED, INFLIGHT = _pipeline_schedule(NCOPY, K_ITERS)
    else:
        SCHED, INFLIGHT = (), 1

    gpu_arch = str(get_rocm_arch() or "")
    if not gpu_arch.startswith("gfx120"):
        raise RuntimeError(f"rdna4_int4_gemm requires gfx120* (RDNA4 v8 IU4 K=32); current arch is {gpu_arch!r}.")

    @fx.struct
    class _Lds:
        data: fx.Array[fx.Int32, LDS_ELEMS, 16]

    @flyc.kernel
    def wmma_int4_gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        wave = tid // WAVE_SIZE
        lane = tid % WAVE_SIZE
        l16 = lane % 16
        g = lane // 16
        wave_m = wave // WAVES_N
        wave_n = wave % WAVES_N
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y

        a2d = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(A)),
            fx.make_layout((M, PACKED_K), (PACKED_K, 1)),
        )
        b2d = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(B)),
            fx.make_layout((N, PACKED_K), (PACKED_K, 1)),
        )
        c2d = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(C)),
            fx.make_layout((M, N), (N, 1)),
        )
        a_iter = fx.get_iter(a2d)
        b_iter = fx.get_iter(b2d)
        lds = fx.make_view(
            fx.recast_iter(fx.Int32, fx.SharedAllocator().allocate(_Lds).peek().data.ptr),
            fx.make_layout((LDS_ELEMS,), (1,)),
        )
        lds_iter = fx.get_iter(lds)
        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, fx.Int4, fx.Int32, sign_a=True, sign_b=True))
        acc_ty = fx.Vector.make_type(8, fx.Int32)

        def _sync():
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()

        def _load_global(src, bid, n_rows, k_tile, vec_id):
            row = vec_id // VECS_PER_ROW
            col = (vec_id % VECS_PER_ROW) * 4
            ptr = fx.add_offset(src, (bid * n_rows + row) * PACKED_K + k_tile * PK + col)
            return fx.make_view(ptr, fx.make_layout(4, 1)).load()

        def _store_lds(stage_base, vec_id, vec):
            row = vec_id // VECS_PER_ROW
            col = (vec_id % VECS_PER_ROW) * 4
            ptr = fx.add_offset(lds_iter, stage_base + row * STRIDE + col)
            fx.make_view(ptr, fx.make_layout(4, 1)).store(vec)

        def _copy_src(copy_id):
            if const_expr(copy_id < ITERS_A):
                return a_iter, bid_m, BLOCK_M, NVEC_A, copy_id, 0
            return b_iter, bid_n, BLOCK_N, NVEC_B, copy_id - ITERS_A, A_ELEMS

        def _load_copy(copy_id, k_tile):
            src, bid, n_rows, nvec, local_i, _base = _copy_src(copy_id)
            vec_id = tid + local_i * THREADS
            if const_expr(nvec % THREADS == 0):
                vec = _load_global(src, bid, n_rows, k_tile, vec_id)
            else:
                vec = fx.Vector.filled(4, 0, fx.Int32)
                if vec_id < nvec:
                    vec = _load_global(src, bid, n_rows, k_tile, vec_id)
            return vec

        def _store_copy(copy_id, stage, vec):
            _src, _bid, _rows, nvec, local_i, base = _copy_src(copy_id)
            vec_id = tid + local_i * THREADS
            stage_base = stage * STAGE_ELEMS + base
            if const_expr(nvec % THREADS == 0):
                _store_lds(stage_base, vec_id, vec)
            else:
                if vec_id < nvec:
                    _store_lds(stage_base, vec_id, vec)

        def _copy_tile(stage, k_tile):
            for copy_id in range_constexpr(NCOPY):
                _store_copy(copy_id, stage, _load_copy(copy_id, k_tile))

        def _lds_i4(elem_off):
            ptr = fx.add_offset(lds_iter, elem_off)
            return fx.make_view(ptr, fx.make_layout(2, 1)).load().bitcast(fx.Int4)

        def _wmma(a_vec, b_vec, acc):
            return fx.Vector(
                fly.mma_atom_call_ssa(
                    [acc_ty],
                    mma_atom,
                    a_vec.ir_value(),
                    b_vec.ir_value(),
                    acc.ir_value(),
                )
            )

        def _as_acc(value):
            return fx.Vector(value)

        def _wmma_kt(stage, kt, accs):
            a_base = stage * STAGE_ELEMS
            b_base = a_base + A_ELEMS
            col = 4 * kt + 2 * g
            new = [_as_acc(acc) for acc in accs]
            b_vecs = []
            for rn in range_constexpr(REG_N):
                brow = wave_n * (REG_N * WMMA_N) + rn * WMMA_N + l16
                b_vecs.append(_lds_i4(b_base + brow * STRIDE + col))
            a_row = wave_m * (REG_M * WMMA_M) + l16
            a_vec = _lds_i4(a_base + a_row * STRIDE + col)
            for rm in range_constexpr(REG_M):
                a_cur = a_vec
                if const_expr(rm + 1 < REG_M):
                    a_vec = _lds_i4(a_base + (a_row + (rm + 1) * WMMA_M) * STRIDE + col)
                for rn in range_constexpr(REG_N):
                    new[rm * REG_N + rn] = _wmma(a_cur, b_vecs[rn], new[rm * REG_N + rn])
            return new

        def _compute_stage(stage, accs):
            new = [_as_acc(acc) for acc in accs]
            for kt in range_constexpr(K_ITERS):
                new = _wmma_kt(stage, kt, new)
            return new

        def _compute_prefetch(stage, next_stage, k_tile, accs):
            new = [_as_acc(acc) for acc in accs]
            held = [None] * INFLIGHT
            for step in range_constexpr(len(SCHED)):
                kind, arg, slot = SCHED[step]
                if const_expr(kind == "load"):
                    held[slot] = _load_copy(arg, k_tile)
                elif const_expr(kind == "wmma"):
                    new = _wmma_kt(stage, arg, new)
                else:
                    _store_copy(arg, next_stage, held[slot])
            return new

        zero = fx.Vector.filled(8, 0, fx.Int32)
        init_accs = [zero for _ in range_constexpr(N_ACC)]
        _copy_tile(fx.Int32(0), fx.Int32(0))
        _sync()
        if const_expr(NUM_K_TILES == 1):
            accs = _compute_stage(fx.Int32(0), init_accs)
        else:
            for iv, state in range(0, NUM_K_TILES - 1, 1, init=init_accs):
                stage = fx.Int32(iv) % 2
                accs = _compute_prefetch(stage, fx.Int32(1) - stage, fx.Int32(iv) + 1, state)
                _sync()
                results = yield accs
            if const_expr(N_ACC == 1):
                results = [results]
            accs = _compute_stage(fx.Int32((NUM_K_TILES - 1) % 2), results)

        for rm in range_constexpr(REG_M):
            for rn in range_constexpr(REG_N):
                acc = _as_acc(accs[rm * REG_N + rn])
                row0 = bid_m * BLOCK_M + wave_m * (REG_M * WMMA_M) + rm * WMMA_M + 8 * g
                col = bid_n * BLOCK_N + wave_n * (REG_N * WMMA_N) + rn * WMMA_N + l16
                for value_idx in range_constexpr(8):
                    c2d[row0 + value_idx, col] = acc[value_idx]

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        wmma_int4_gemm_kernel(A, B, C).launch(
            grid=(M // BLOCK_M, N // BLOCK_N, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    return launch
