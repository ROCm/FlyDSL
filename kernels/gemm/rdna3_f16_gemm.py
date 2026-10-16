#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""WMMA GEMM kernel for RDNA3 / RDNA3.5 (gfx11*, wave32).

Ported from rdna_f16_gemm.py (gfx120x). Same algorithm (4-warp double-
buffered LDS ping-pong, 128x128x32 tiles, swizzled grid mapping) but
adapted for the legacy v16-operand WMMA ABI used by RDNA3/RDNA3.5:

  * Input operands (A, B) are vector<16> instead of vector<8>; each
    lane carries 16 contiguous K-elements of one M (or N) row. Lanes
    0-15 carry distinct rows; lanes 16-31 carry duplicates of the same
    rows lanes 0-15 read. We just have all lanes do the LDS loads —
    duplicate loads are wasted bandwidth but simpler than a wave-half
    broadcast.
    TODO(perf): lanes 16-31 could ``ds_swizzle_b32`` XOR 16 broadcast
    from lanes 0-15 to halve LDS read bandwidth.

  * Accumulator (C/D) is still vector<8>, but the per-lane row mapping
    differs from gfx12: lane L holds D[2*si + (L/16)][L%16], i.e. even
    rows in lanes 0-15 and odd rows in lanes 16-31.

Computes C[M,N] = A[M,K] @ B_T[N,K]^T (same interface as
``rdna_f16_gemm.create_wmma_gemm_module``).

The block tile is a parameter and defaults to 128x128x32. Deciding it from the
shape -- which is worth up to 3.0x on shapes too small to fill the grid -- is
the job of ``rdna3_f16_gemm_autotune``; this module only builds what it is told.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch
from kernels.common.kernels_common import cvt_sr_f32_to_bf16

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16
WAVE_SIZE = 32

# K-padding for the "pad" LDS layout; "kblock" needs none.
K_PAD = 8

DTYPE_BYTES = {"bf16": 2, "f16": 2, "f32": 4}


def _sched_plan(reg_m, reg_n, reg_k, g2s_chunks):
    """Issue-order hints for the k-tile body as (group, count) pairs."""
    per_rk_wmma = reg_m * reg_n
    per_rk_dsrd = 2 * (reg_m + reg_n)  # each v16 operand is two 128-bit reads
    chunk = max(1, reg_n)

    plan = [("vmem", g2s_chunks), ("dsrd", per_rk_dsrd)]
    dsrd_left = per_rk_dsrd * (reg_k - 1)
    dsrd_step = chunk
    dswr_left = g2s_chunks
    dswr_chunk = max(1, g2s_chunks // 2)
    for _ in range(reg_k):
        issued = 0
        while issued < per_rk_wmma:
            n = min(chunk, per_rk_wmma - issued)
            plan.append(("mfma", n))
            issued += n
            if dsrd_left:
                take = min(dsrd_step, dsrd_left)
                plan.append(("dsrd", take))
                dsrd_left -= take
            elif dswr_left:
                take = min(dswr_chunk, dswr_left)
                plan.append(("dswr", take))
                dswr_left -= take
    if dswr_left:
        plan.append(("dswr", dswr_left))
    return tuple(plan)


def _group_width(grid_m, group_m):
    """Largest grouping width <= group_m that divides grid_m.

    ``_swizzle_tile_id`` requires a group width that divides grid_m.
    """
    return max(d for d in range(1, min(group_m, grid_m) + 1) if grid_m % d == 0)


def _swizzle_tile_id(pid, grid_n, group_width):
    """Linear workgroup id -> (bid_m, bid_n).

    Walks group_width tiles down M before stepping in N, so workgroups that run
    concurrently share B tiles in L2. Plain integer arithmetic, so it evaluates
    the same on a host int as on the kernel's block id.
    """
    num_pid_in_group = group_width * grid_n
    group_id = pid // num_pid_in_group
    pid_in_group = pid % num_pid_in_group
    return group_id * group_width + (pid_in_group % group_width), pid_in_group // group_width


def create_wmma_gemm_module(
    M: int,
    N: int,
    K: int,
    in_dtype="bf16",
    out_dtype="bf16",
    *,
    rounding="rn",  # "rn" (round to nearest) or "rs" (stochastic rounding)
    reg_m=4,
    reg_n=4,
    reg_k=2,
    waves_m=2,
    waves_n=2,
    group_m=8,
    lds_layout="pad",
    sched_hint=False,
    stagger=0,
    persistent_wgs=0,
    lda=None,
    ldb=None,
    ldc=None,
    m_valid=None,
    n_valid=None,
):
    # M and N may be rounded-up tile extents over ``m_valid`` real rows of A/C
    # and ``n_valid`` real rows of B_T; the strides describe the real operands.
    m_rows = M if m_valid is None else int(m_valid)
    n_cols = N if n_valid is None else int(n_valid)
    ld_a = K if lda is None else int(lda)
    ld_b = K if ldb is None else int(ldb)
    ld_c = n_cols if ldc is None else int(ldc)
    if ld_a < K or ld_b < K or ld_c < n_cols:
        raise ValueError(
            f"leading dimensions must cover the operands: lda={ld_a}, ldb={ld_b} "
            f"need K={K}, ldc={ld_c} needs N={n_cols}"
        )
    vec_elems = 16 // (2 if in_dtype in ("bf16", "f16") else 4)
    if ld_a % vec_elems or ld_b % vec_elems:
        raise ValueError(
            f"lda={ld_a} and ldb={ld_b} must be multiples of {vec_elems} to keep "
            f"each row 16-byte aligned for the 128-bit copy"
        )

    gpu_arch = str(get_rocm_arch() or "")
    if not gpu_arch.startswith("gfx11"):
        raise RuntimeError(
            f"rdna3_f16_gemm requires gfx11* (RDNA3 / RDNA3.5); current arch is {gpu_arch!r}. "
            "Use rdna_f16_gemm.create_wmma_gemm_module on gfx120* (RDNA4)."
        )

    BLOCK_M = WMMA_M * reg_m * waves_m
    BLOCK_N = WMMA_N * reg_n * waves_n
    BLOCK_K = WMMA_K * reg_k
    NUM_WAVES = waves_m * waves_n
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE

    assert reg_k >= 1
    assert rounding in ("rn", "rs"), f"rounding must be 'rn' or 'rs', got {rounding!r}"
    if rounding == "rs":
        assert out_dtype == "bf16", "stochastic rounding currently supports bf16 output only"

    LOAD_VEC = 8  # 8 bf16 = 128-bit GMEM/LDS load
    # G2S thread geometry: thread (tk, tm) moves the 128-bit chunk at columns
    # [tk*LOAD_VEC, +LOAD_VEC) of row tm, repeated over M to cover the tile.
    THRS_K = BLOCK_K // LOAD_VEC
    THRS_M = THREADS_PER_BLOCK // THRS_K
    G2S_CHUNKS = (BLOCK_M + BLOCK_N) * BLOCK_K // THREADS_PER_BLOCK // LOAD_VEC
    SCHED_PLAN = _sched_plan(reg_m, reg_n, reg_k, G2S_CHUNKS) if sched_hint else ()
    assert THRS_K * THRS_M == THREADS_PER_BLOCK
    assert BLOCK_M % THRS_M == 0 and BLOCK_N % THRS_M == 0

    # "pad": row-major with K_PAD; "kblock": k-major, no pad (fits 256x256x32 in 64 KB).
    assert lds_layout in ("pad", "kblock")
    if lds_layout == "kblock":
        assert BLOCK_K % LOAD_VEC == 0
    k_pad = 0 if lds_layout == "kblock" else K_PAD
    ROW_STRIDE_A = BLOCK_K + k_pad
    ROW_STRIDE_B = BLOCK_K + k_pad
    LDS_A_SIZE = BLOCK_M * ROW_STRIDE_A
    LDS_B_SIZE = BLOCK_N * ROW_STRIDE_B
    LDS_ONE_BUF = LDS_A_SIZE + LDS_B_SIZE
    LDS_TOTAL = 2 * LDS_ONE_BUF

    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    # Rows past the bound are suppressed by the hardware: zero on load, which a
    # product ignores, and dropped on store. Columns get no such free edge --
    # past the end of a row is a valid offset into the next one -- so the C
    # columns past ``n_cols`` are the one thing here that needs a predicate.
    if not M - BLOCK_M < m_rows <= M:
        raise ValueError(f"m_valid={m_rows} must round up to M={M} with BLOCK_M={BLOCK_M}")
    if not N - BLOCK_N < n_cols <= N:
        raise ValueError(f"n_valid={n_cols} must round up to N={N} with BLOCK_N={BLOCK_N}")
    a_bound_bytes = m_rows * ld_a * DTYPE_BYTES[in_dtype] if m_rows < M else None
    b_bound_bytes = n_cols * ld_b * DTYPE_BYTES[in_dtype] if n_cols < N else None
    c_bound_bytes = m_rows * ld_c * DTYPE_BYTES[out_dtype] if m_rows < M else None

    if lds_layout == "pad" and (BLOCK_K + k_pad) % LOAD_VEC:
        raise ValueError(
            f"K_PAD={k_pad} leaves an LDS row of {BLOCK_K + k_pad} elements, "
            f"which is not a multiple of the {LOAD_VEC}-element vector store"
        )

    num_k_tiles = K // BLOCK_K
    if num_k_tiles < 2:
        raise ValueError(f"Need at least 2 K-tiles for prefetch pipeline; got K={K}, BLOCK_K={BLOCK_K}")

    grid_m = M // BLOCK_M
    grid_n = N // BLOCK_N

    group_width = _group_width(grid_m, group_m)

    assert stagger >= 0
    stagger_step = int(stagger) if num_k_tiles & (num_k_tiles - 1) == 0 else 0

    num_tiles = grid_m * grid_n
    persist_wgs = int(persistent_wgs)
    if persist_wgs not in (0, num_tiles):
        raise ValueError(
            f"persistent_wgs must be 0 (plain grid) or num_tiles={num_tiles} "
            f"(whole-tile persistent), got {persist_wgs}"
        )
    persist_rot_step = int(stagger) if persist_wgs else 0
    if persist_wgs:
        stagger_step = 0

    is_bf16 = in_dtype == "bf16"

    def _wmma_op(a_vec, b_vec, acc):
        if is_bf16:
            a_i16 = a_vec.bitcast(fx.Int16)
            b_i16 = b_vec.bitcast(fx.Int16)
            return rocdl.wmma_f32_16x16x16_bf16(acc.type, a_i16, b_i16, acc).result
        return rocdl.wmma_f32_16x16x16_f16(acc.type, a_vec, b_vec, acc).result

    elem_dtype = fx.BFloat16 if is_bf16 else fx.Float16
    out_elem_cls = {"bf16": fx.BFloat16, "f16": fx.Float16, "f32": fx.Float32}[out_dtype]
    acc_size = 8 * reg_m * reg_n  # accumulator f32 VGPRs per thread

    # One flat array holding both LDS buffers, aligned to 16 bytes so that the
    # 8-element vector loads are legal.
    @fx.struct
    class _SharedStorage:
        lds: fx.Array[elem_dtype, LDS_TOTAL, 16]

    @flyc.kernel
    def wmma_gemm_kernel(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_bt: fx.Tensor,
        tiled_mma: fx.TiledMma,
        tiled_copy_g2s: fx.TiledCopy,
        sr_seed: fx.Int32,  # runtime seed; only read on the stochastic-rounding path
    ):
        lds_storage = fx.SharedAllocator().allocate(_SharedStorage).peek()
        lds_ptr = lds_storage.lds.ptr  # i8-base aliased as elem_dtype*

        def _v8_load(v8_idx):
            """One 8-element vector out of the flat LDS array."""
            elem_off = fx.Int32(v8_idx * 8)
            ptr_off = fx.add_offset(lds_ptr, fx.make_int_tuple(elem_off))
            typed_ptr = fx.recast_iter(elem_dtype, ptr_off)
            return fx.make_view(typed_ptr, fx.make_layout(8, 1)).load()

        tid = gpu.thread_id("x")
        pid = gpu.block_id("x")

        wave_id = tid // 32
        lane = tid % 32
        lane16 = lane % 16  # lanes 16-31 mirror 0-15, so the row is lane % 16

        if const_expr(stagger_step):
            k_first = fx.Int32(pid) * stagger_step % num_k_tiles
        else:
            k_first = 0

        def _k_tile(step, rot=None, n_iter=None):
            """Global k-tile index for the ``step``-th tile of one output-tile visit."""
            if const_expr(stagger_step):
                return (k_first + fx.Int32(step)) % num_k_tiles
            if const_expr(persist_rot_step):
                kk = rot + fx.Int32(step)
                return (kk >= n_iter).select(kk - n_iter, kk)
            return step

        wave_m = wave_id // waves_n
        wave_n = wave_id % waves_n

        # ── GMEM -> registers -> LDS, through the tiled copy ──
        thr_g2s = tiled_copy_g2s.get_slice(tid)
        thr_mma = tiled_mma.thr_slice(tid)
        copy_out = fx.make_copy_atom(fx.rocdl.BufferCopy(out_elem_cls.width), out_elem_cls)
        thr_r2g_C = fx.make_tiled_copy_C(copy_out, tiled_mma).get_slice(tid)

        def _bounded(tensor, bound_bytes):
            """Buffer view whose descriptor stops at the last real row."""
            if const_expr(bound_bytes is None):
                return fx.rocdl.make_buffer_tensor(tensor)
            return fx.rocdl.make_buffer_tensor(tensor, max_size=False, num_records_bytes=bound_bytes)

        def _tile_operands(bid_m, bid_n):
            """The A and B row bands this tile reads, partitioned per thread."""
            tA = fx.flat_divide(_bounded(arg_a, a_bound_bytes), fx.make_tile(BLOCK_M, BLOCK_K))[None, None, bid_m, None]
            tB = fx.flat_divide(_bounded(arg_bt, b_bound_bytes), fx.make_tile(BLOCK_N, BLOCK_K))[
                None, None, bid_n, None
            ]
            return thr_g2s.partition_S(tA), thr_g2s.partition_S(tB)

        buf_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
        uni_copy = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)

        # The destination buffer alternates on ``iv % 2``, a loop-carried value,
        # so it cannot be picked from a list of per-stage views at trace time;
        # the view is built from the running element offset instead.
        def _lds_dst(buf_offset, base, rows, row_stride):
            ptr = fx.add_offset(lds_ptr, fx.make_int_tuple(buf_offset + base))
            if const_expr(lds_layout == "kblock"):
                layout = fx.make_layout(
                    (rows, (LOAD_VEC, BLOCK_K // LOAD_VEC)),
                    (LOAD_VEC, (1, rows * LOAD_VEC)),
                )
            else:
                layout = fx.make_layout((rows, BLOCK_K), (row_stride, 1))
            view = fx.make_view(fx.recast_iter(elem_dtype, ptr), layout)
            return thr_g2s.partition_D(view)[None, None, None]

        def _pA_s(buf_offset):
            return _lds_dst(buf_offset, 0, BLOCK_M, ROW_STRIDE_A)

        def _pB_s(buf_offset):
            return _lds_dst(buf_offset, LDS_A_SIZE, BLOCK_N, ROW_STRIDE_B)

        def _lds_elem(rows, row_stride, row, col):
            """Element index of (row, col) inside one A- or B-tile of the buffer."""
            if const_expr(lds_layout == "kblock"):
                return (col // LOAD_VEC * rows + row) * LOAD_VEC + col % LOAD_VEC
            return row * row_stride + col

        frag_copy_A = fx.make_fragment_like(_pA_s(0))
        frag_copy_B = fx.make_fragment_like(_pB_s(0))

        def _gmem_load(pA_g, pB_g, k_tile):
            fx.copy(buf_copy, pA_g[None, None, None, k_tile], frag_copy_A)
            fx.copy(buf_copy, pB_g[None, None, None, k_tile], frag_copy_B)

        def _lds_store(buf_offset):
            fx.copy(uni_copy, frag_copy_A, _pA_s(buf_offset))
            fx.copy(uni_copy, frag_copy_B, _pB_s(buf_offset))

        # ── LDS reads: a v16 operand is two contiguous v8 chunks, at
        # ``col_lo = 16*rk`` and ``col_hi = 16*rk + 8`` of the row ──
        _concat16_mask = list(range(16))  # shuffle mask for v8 ++ v8 -> v16

        def _load_b_from_lds(rk, buf_offset):
            vecs = []
            col_lo = 16 * rk
            col_hi = 16 * rk + 8
            for rn in range_constexpr(reg_n):
                row = wave_n * (reg_n * WMMA_N) + 16 * rn + lane16
                lds_idx_lo = buf_offset + LDS_A_SIZE + _lds_elem(BLOCK_N, ROW_STRIDE_B, row, col_lo)
                lds_idx_hi = buf_offset + LDS_A_SIZE + _lds_elem(BLOCK_N, ROW_STRIDE_B, row, col_hi)
                v_lo = _v8_load(lds_idx_lo // 8)
                v_hi = _v8_load(lds_idx_hi // 8)
                vecs.append(v_lo.shuffle(v_hi, _concat16_mask))
            return vecs

        def _load_a_single_from_lds(rk, rm_val, buf_offset):
            col_lo = 16 * rk
            col_hi = 16 * rk + 8
            row = wave_m * (reg_m * WMMA_M) + 16 * rm_val + lane16
            lds_idx_lo = buf_offset + _lds_elem(BLOCK_M, ROW_STRIDE_A, row, col_lo)
            lds_idx_hi = buf_offset + _lds_elem(BLOCK_M, ROW_STRIDE_A, row, col_hi)
            v_lo = _v8_load(lds_idx_lo // 8)
            v_hi = _v8_load(lds_idx_hi // 8)
            return v_lo.shuffle(v_hi, _concat16_mask)

        def _barrier():
            # gfx11 barrier — split signal/wait and s_wait_dscnt are gfx12+.
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_waitcnt lgkmcnt(0)\ns_barrier",
                constraints="",
                has_side_effects=True,
            )

        def _do_compute_rk(accs_in, rk, buf_offset, b_vecs):
            new_accs = list(accs_in)
            a_next = _load_a_single_from_lds(rk, 0, buf_offset)
            for rm in range_constexpr(reg_m):
                a_vec = a_next
                if const_expr(rm + 1 < reg_m):
                    a_next = _load_a_single_from_lds(rk, rm + 1, buf_offset)
                for rn in range_constexpr(reg_n):
                    idx = rm * reg_n + rn
                    new_accs[idx] = _wmma_op(
                        a_vec,
                        b_vecs[rn],
                        new_accs[idx],
                    )
            return new_accs

        def _compute_k_tile(accs_in, buf_offset):
            """All reg_k WMMA steps over one LDS buffer."""
            new_accs = list(accs_in)
            for rk in range_constexpr(reg_k):
                new_accs = _do_compute_rk(new_accs, rk, buf_offset, _load_b_from_lds(rk, buf_offset))
            return new_accs

        def _sched_k_tile():
            emit = {
                "vmem": rocdl.sched_vmem,
                "mfma": rocdl.sched_mfma,
                "dsrd": rocdl.sched_dsrd,
                "dswr": rocdl.sched_dswr,
            }
            for group, count in SCHED_PLAN:
                emit[group](count)

        zero_acc = fx.full(8, 0.0, fx.Float32)
        n_acc = reg_m * reg_n
        c_lds_buf_stride = LDS_ONE_BUF

        def _one_k_tile(pA_g, pB_g, s_accs, read_off, write_off, load_tile):
            """Prefetch the next k-tile, consume this one, hand over, barrier."""
            _gmem_load(pA_g, pB_g, load_tile)
            s_accs = _compute_k_tile(s_accs, read_off)
            _lds_store(write_off)
            if const_expr(sched_hint):
                _sched_k_tile()
            _barrier()
            return s_accs

        def _carry_list(state):
            """The k-loop carry as a list.

            scf.for hands back a bare value rather than a one-element sequence
            when the loop carries one accumulator, as a 16x16 tile does.
            """
            return list(state) if isinstance(state, (list, tuple)) else [state]

        def _accumulate(pA_g, pB_g, rot=None):
            """Run the double-buffered pipeline over all k-tiles of one output tile."""
            if const_expr(persist_wgs):
                _barrier()
            _gmem_load(pA_g, pB_g, _k_tile(fx.Int32(0), rot, num_k_tiles))
            _lds_store(0)
            _barrier()

            init_state = [zero_acc for _ in range_constexpr(n_acc)]

            for iv, state in range(0, num_k_tiles - 1, 1, init=init_state):
                s_accs = _carry_list(state)[:n_acc]
                s_accs = _one_k_tile(
                    pA_g,
                    pB_g,
                    s_accs,
                    iv % 2 * c_lds_buf_stride,
                    (1 - iv % 2) * c_lds_buf_stride,
                    _k_tile(iv + 1, rot, num_k_tiles),
                )
                results = yield list(s_accs)

            return _compute_k_tile(_carry_list(results)[:n_acc], ((num_k_tiles - 1) % 2) * c_lds_buf_stride)

        # ── Store results to GMEM through the tiled copy ──
        def _store_C(accs, bid_m, bid_n, seed_slot):
            # The accumulator layout and the wave banding are both encoded in
            # the tiled_mma; what is left here is the value transform, which no
            # copy atom can express. frag_C flattens as si + 8*(rm + reg_m*rn),
            # so each run of 8 elements is one atom's accumulator.
            tC = fx.flat_divide(_bounded(arg_c, c_bound_bytes), fx.make_tile(BLOCK_M, BLOCK_N))[
                None, None, bid_m, bid_n
            ]
            frag_C = thr_mma.make_fragment_C(tC)
            pC_g = thr_r2g_C.partition_S(tC)
            if const_expr(n_cols < N):
                cC = fx.flat_divide(
                    fx.make_view((0, 0), fx.make_identity_layout((M, N))),
                    fx.make_tile(BLOCK_M, BLOCK_N),
                )[None, None, bid_m, bid_n]
                thr_cC = thr_r2g_C.partition_S(cC)
                pred_C = fx.make_fragment_like(thr_cC, dtype=fx.Boolean)
                for a in range_constexpr(fx.size(thr_cC.shape).unpack()):
                    pred_C[a] = fx.elem_less(thr_cC[a], (M, n_cols))
            else:
                pred_C = None
            if const_expr(out_elem_cls is fx.Float32):
                frag_C_out = frag_C
            else:
                frag_C_out = fx.make_fragment_like(frag_C, out_elem_cls.ir_type)
            frag_C_retile = thr_r2g_C.retile(frag_C_out)

            ordered_accs = [accs[rm * reg_n + rn] for rn in range_constexpr(reg_n) for rm in range_constexpr(reg_m)]
            if const_expr(rounding == "rs"):
                # Each of the 8 values takes a distinct 16-bit slice of the 4
                # random words, so the store is unbiased without a per-element
                # draw. Keying on the thread's slot rather than the output
                # coordinate keeps it independent of where the fragment lands.
                out_elems = []
                for g, acc in enumerate(ordered_accs):
                    base_off = (seed_slot * THREADS_PER_BLOCK + tid) * acc_size + 8 * g
                    words = fx.random.randint4x(fx.Uint32(sr_seed), fx.Uint32(base_off))
                    for si in range_constexpr(8):
                        word = words[si // 2]
                        rbits = word if si % 2 == 0 else (word >> fx.Uint32(16))
                        out_elems.append(cvt_sr_f32_to_bf16(acc[si], rbits))
            elif const_expr(out_elem_cls is fx.Float32):
                out_elems = [acc[si] for acc in ordered_accs for si in range_constexpr(8)]
            else:
                out_elems = [acc[si].to(out_elem_cls) for acc in ordered_accs for si in range_constexpr(8)]

            frag_C_out.store(fx.Vector.from_elements(out_elems, out_elem_cls))
            fx.copy(copy_out, frag_C_retile, pC_g, pred=pred_C)

        if const_expr(not persist_wgs):
            bid_m, bid_n = _swizzle_tile_id(pid, grid_n, group_width)
            pA_g, pB_g = _tile_operands(bid_m, bid_n)
            accs = _accumulate(pA_g, pB_g)
            _store_C(accs, bid_m, bid_n, pid)
        else:
            pid32 = fx.Int32(pid)
            t_first = pid32 * num_tiles // persist_wgs
            t_last = (pid32 + 1) * num_tiles // persist_wgs - 1

            for t, _carry in range(t_first, t_last + 1, 1, init=[fx.Int32(0)]):
                t32 = fx.Int32(t)
                bid_m, bid_n = _swizzle_tile_id(t32, grid_n, group_width)
                pA_g, pB_g = _tile_operands(bid_m, bid_n)
                if const_expr(persist_rot_step):
                    rot = pid32 * persist_rot_step % fx.Int32(num_k_tiles)
                else:
                    rot = None
                accs = _accumulate(pA_g, pB_g, rot)
                _store_C(accs, bid_m, bid_n, t32)
                _ = yield [fx.Int32(0)]

    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_bt: fx.Tensor,
        stream: fx.Stream,
        sr_seed: fx.Int32 = 0,
    ):
        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, elem_dtype, fx.Float32))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((waves_m, waves_n, 1), (waves_n, 1, 0)),
            permutation=(
                fx.make_layout((WMMA_M, waves_m, reg_m), (1, WMMA_M * reg_m, WMMA_M)),
                fx.make_layout((WMMA_N, waves_n, reg_n), (1, WMMA_N * reg_n, WMMA_N)),
                WMMA_K,
            ),
        )
        tiled_copy_g2s = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype),
            fx.make_layout(
                ((THRS_K, THRS_M), (1, LOAD_VEC)),
                ((THRS_M * LOAD_VEC, 1), (1, THRS_M)),
            ),
            fx.make_tile(THRS_M, BLOCK_K),
        )

        arg_a_2d = fx.make_view(fx.get_iter(arg_a), fx.make_layout((M, K), (ld_a, 1)))
        arg_bt_2d = fx.make_view(fx.get_iter(arg_bt), fx.make_layout((N, K), (ld_b, 1)))
        arg_c_2d = fx.make_view(fx.get_iter(arg_c), fx.make_layout((M, N), (ld_c, 1)))

        c1 = 1
        total_blocks = persist_wgs if persist_wgs else grid_m * grid_n
        bk = THREADS_PER_BLOCK

        launcher = wmma_gemm_kernel(arg_c_2d, arg_a_2d, arg_bt_2d, tiled_mma, tiled_copy_g2s, sr_seed)
        launcher.launch(
            grid=(total_blocks, c1, c1),
            block=(bk, c1, c1),
            stream=stream,
        )

    def launch(arg_c, arg_a, arg_bt, stream, sr_seed=0):
        return launch_gemm(arg_c, arg_a, arg_bt, stream, sr_seed)

    return launch, BLOCK_M, BLOCK_N, BLOCK_K
