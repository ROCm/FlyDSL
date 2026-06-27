# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""4-wave MXFP4 matmul for AMD CDNA4 (gfx950 / MI355X).

C[M,N] = A[M,K] @ B[N,K]^T with per-32-block E8M0 scales on both A and B,
bf16 output.

Structure is copied from ``kernels/fp8_gemm_4wave.py``: 1 block = 256 threads =
4 waves in a 2x2 layout; each wave owns a 128x128 quadrant computed as a 2x2 of
64x64 (c00/c01/c10/c11); 8-buffer LDS ping-pong with a depth-2 K pipeline; an
``_interleaved_cluster`` interleaving MFMAs with global->LDS and LDS->reg loads.

FP4 specifics:
  * MFMA = ``mfma_scale_f32_16x16x128_f8f6f4`` cbsz=4 blgp=4; per-32-block E8M0
    scale applied INSIDE the MFMA (epilogue only converts acc->bf16).
  * One LDS K-step row = 128 bytes = 256 fp4 = TWO MFMA K=128 blocks. The fp8
    data-movement (G2SLoader / swizzle / S2RLoader) is reused treating fp4 as
    bytes; the S2R i32x8 (32 B/lane) is split into two 16-B fp4 operands.
  * pack_M=pack_N=pack_K=2: per wave-quadrant there is exactly one M-pair and
    one N-pair (n_tiles=2), so one A-scale i32 and one B-scale i32 hold the four
    E8M0 sub-fields selected by opsel = k_sub*2 + tile_in_pair.

A: row-major fp4 (uint8, 2 fp4/byte). B: ``shuffle_weight_w4(b_q, 16)``.
Scales: ``shuffle_scale_w4(scale, 1, False)``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr import buffer_ops as _buffer_ops
from flydsl.expr.typing import Vector as Vec
from kernels.fp8_gemm_utils import (
    G2SLoader,
    ceildiv,
    compute_global_swizzle,
    divmod,
    make_fp8_buffer_tensor,
    swizzle_128,
    wait_barrier,
)


class S2RLoaderFp4:
    """fp4 S2R LDS->reg loader. Unlike the fp8 loader it does NOT pack the two
    K=64 halves into an i32x8 fragment -- it returns the two i32x4 halves as-is,
    one per fp4 MFMA K=128 sub-block. This avoids the pack_i32x4_i32x8 (S2R) +
    _split_i32x8 + _pack_fp4_operand (MFMA) round-trip that the fp8-derived path
    forced, which created ~64 VGPR of split temporaries on top of the i32x8
    fragments and pushed arch VGPR to 256 (full) -> scale spilled. Each tile's
    value is [i32x4_ksub0, i32x4_ksub1]."""

    def __init__(self, wave_idx, n_tiles):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles

    def _vec_load_16xf8(self, lds_src, offset):
        off_tup = fx.make_int_tuple(offset)
        ptr_off = fx.add_offset(lds_src.ptr, off_tup)
        i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
        view = fx.make_view(i8_iter, fx.make_layout(16, 1))
        return view.load()

    def _offset(self, i, step, preshuffled):
        row = self.wave_idx * (self.n_tiles * 16) + i * 16 + self.lane_id % 16
        col = (self.lane_id // 16) * 16 + step * 64
        if const_expr(preshuffled):
            return (row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128
        row_swz, col_swz = swizzle_128(row, col)
        return row_swz * 128 + col_swz

    def load(self, lds_src, preshuffled=False):
        frag = []
        for i in range_constexpr(self.n_tiles):
            halves = []
            for step in range_constexpr(2):
                v = self._vec_load_16xf8(lds_src, self._offset(i, step, preshuffled))
                halves.append(v.bitcast(fx.Int32))  # i32x4, the K=128 MFMA operand
            frag.append(halves)  # [ksub0_i32x4, ksub1_i32x4]
        return frag

    def load_one(self, lds_src, i, ksub, preshuffled=False):
        """One i32x4 (tile i, K=128 sub-block ksub) -- the interleave granularity."""
        v = self._vec_load_16xf8(lds_src, self._offset(i, ksub, preshuffled))
        return v.bitcast(fx.Int32)


def _flat_frag(frag):
    """fragment [tile][ksub] -> flat list of raw i32x4 ir.Values (2*n_tiles).
    scf.for loop-carried args must be raw ir.Values (the dispatch reads .type),
    so unwrap Vec/ArithValue via arith._to_raw."""
    out = []
    for t in frag:
        out.append(arith._to_raw(t[0]))
        out.append(arith._to_raw(t[1]))
    return out


def _unflat_frag(flat, n_tiles):
    return [[flat[2 * i], flat[2 * i + 1]] for i in range(n_tiles)]


def _g2s_thunks(g2s, dst, gl_off, n_steps):
    """Module-level (so the @kernel AST rewriter doesn't turn the `range` into
    scf.for): list of thunks, each issuing one g2s.load_one step."""
    return [lambda s=s: g2s.load_one(dst, gl_off, s) for s in range(n_steps)]


def _s2r_thunks(s2r, src, holder, n, pre):
    """List of thunks, each issuing one s2r.load_one (tile i, ksub) into holder[i]."""
    ts = []
    for i in range(n):
        for ks in range(_FP4_PACK):

            def f(i=i, ks=ks):
                if holder[i] is None:
                    holder[i] = [None, None]
                holder[i][ks] = s2r.load_one(src, i, ks, preshuffled=pre)

            ts.append(f)
    return ts


def _min(a, b):
    return arith.select(a < b, a, b)


def _xcd_swizzle(num_pid_m, num_pid_n):
    NUM_XCDS = 8
    WGM = 4
    NUM_CUS = 32 * NUM_XCDS
    SWIZZLE_THRESHOLD = 4 * NUM_CUS

    wgid = fx.block_idx.x
    num_wg = num_pid_m * num_pid_n
    simple_m, simple_n = divmod(wgid, num_pid_n)

    intra_xcd, xcd = divmod(wgid, NUM_XCDS)
    wgid_remap = xcd * (num_wg // NUM_XCDS) + intra_xcd
    num_wgid_in_group = WGM * num_pid_n
    group_id, intra_group = divmod(wgid_remap, num_wgid_in_group)
    first_pid_m = group_id * WGM
    group_size_m = _min(num_pid_m - first_pid_m, WGM)
    pid_n, intra_group_m = divmod(intra_group, group_size_m)
    pid_m = first_pid_m + intra_group_m

    use_simple = (num_wg < SWIZZLE_THRESHOLD) | (num_wg % NUM_XCDS != 0)
    return (arith.select(use_simple, simple_m, pid_m), arith.select(use_simple, simple_n, pid_n))


# ── FP4 scaled MFMA ──────────────────────────────────────────────────────────
_FP4_CBSZ = 4
_FP4_BLGP = 4
_FP4_PACK = 2  # pack_M = pack_N = pack_K = 2


def _split_i32x8(v):
    return v.shuffle(v, [0, 1, 2, 3]), v.shuffle(v, [4, 5, 6, 7])


def _pack_fp4_operand(i32x4):
    """i32x4 (16 B = K=128 fp4) -> a128 i32x8 = (i64_0, i64_1, 0, 0)."""
    i64x2 = i32x4.bitcast(fx.Int64)
    z = fx.Int64(0)
    return Vec.from_elements([i64x2[0], i64x2[1], z, z], fx.Int64).bitcast(fx.Int32)


class Mfma16x16x128Fp4:
    """fp4 16x16x128 scaled MFMA. ``call_one`` runs the two K=128 fp4 sub-blocks
    packed into the 32-byte S2R operand, accumulating into one f32x4 acc.

    a/b operands are i32x8 (the full 256-fp4 K-step, two K=128 sub-blocks).
    sa/sb are i32 packed-E8M0 scales (4 e8m0 each); opsel selects the field
    ``k_sub * pack + tile_in_pair`` where tile_in_pair = i % pack / j % pack.
    """

    def __init__(self, n_tiles_a, n_tiles_b):
        assert n_tiles_a % _FP4_PACK == 0 and n_tiles_b % _FP4_PACK == 0
        self.accum_type = Vec.make_type(4, fx.Float32)
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.res_ty = Vec.make_type(4, fx.Float32)

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def call(self, a, b, c, sa, sb, interleave=None):
        """``sa`` / ``sb`` are lists (len n_groups) of packed-E8M0 i32 scales
        (4 sub-fields each, one full K=256 step for a 32-row pack-group).

        The accumulator is PINNED IN AGPR via inline asm (constraint ``=a,...,0``),
        mirroring fp8's Mfma16x16x128AGPR. The plain ssa-lowered mfma_scale let the
        compiler spill accumulators to arch VGPR and shuffle them with
        v_accvgpr_mov/read (ISA: 1679 such ops, arch VGPR -> 256, scale spilled).
        Pinning keeps the f32x4 acc in-place in AGPR -> arch VGPR drops, no spill.

        opsel is a COMPILE-TIME byte-select baked into the asm string:
          opsel_a = ksub*2 + (i%2), opsel_b = ksub*2 + (j%2).
        AMD encoding: low bit -> op_sel[lane], high bit (=ksub) -> op_sel_hi[lane].
        So op_sel=[i%2, j%2, 0], op_sel_hi=[ksub, ksub, 0]."""
        # a[i] / b[j] are [i32x4_ksub0, i32x4_ksub1] from S2RLoaderFp4.
        # ``interleave`` is an optional list of zero-arg thunks (ds_read / buffer_load
        # for the NEXT quad); one is issued after each MFMA so the load co-issues in
        # the MFMA's execute shadow (fp4 MFMA: 4-cyc issue, ~16-cyc execute -> a
        # ds_read/buffer_load fits free between MFMAs). Mirrors fp8 _interleaved_cluster.
        thunks = list(interleave) if interleave else []
        nth = [0]  # python-level counter (compile-time), not loop-carried
        for ksub in range_constexpr(_FP4_PACK):
            for i in range_constexpr(self.n_tiles_a):
                a_op = a[i][ksub]
                sa_v = sa[i // _FP4_PACK]
                ia = i % _FP4_PACK
                for j in range_constexpr(self.n_tiles_b):
                    b_op = b[j][ksub]
                    sb_v = sb[j // _FP4_PACK]
                    jb = j % _FP4_PACK
                    c[self.idx(i, j)] = self._mfma_agpr(a_op, b_op, c[self.idx(i, j)], sa_v, sb_v, ksub, ia, jb)
                    if nth[0] < len(thunks):
                        thunks[nth[0]]()
                        nth[0] += 1
        while nth[0] < len(thunks):
            thunks[nth[0]]()
            nth[0] += 1
        return c

    def _mfma_agpr(self, a_op, b_op, acc, sa_v, sb_v, ksub, ia, jb):
        # Build the op_sel / op_sel_hi suffix (compile-time). op_sel[2]/hi[2]=0.
        opsel = f"op_sel:[{ia},{jb},0]"
        opsel_hi = f"op_sel_hi:[{ksub},{ksub},0]"
        asm = "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $3, $4 " f"{opsel} {opsel_hi} cbsz:4 blgp:4"
        return _llvm.inline_asm(
            self.res_ty,
            [
                arith._to_raw(a_op),
                arith._to_raw(b_op),
                arith._to_raw(sa_v),
                arith._to_raw(sb_v),
                arith._to_raw(acc),
            ],
            asm,
            "=a,v,v,v,v,0",
            has_side_effects=True,
        )


class ScaleLoader:
    """Loads ``shuffle_scale_w4``-PRESHUFFLED per-1x32 E8M0 scales.

    The shuffled layout (gate_up=False) packs the e8m0 as
    ``[N1, K1, K_Lane, N_Lane, K_Pack, N_Pack]`` (K_Lane=4, N_Lane=16,
    K_Pack=N_Pack=2), so the 4 e8m0 selected by a lane's opsel values 0..3 are
    exactly the 4 contiguous bytes of one i32. Per lane the element offset is::

        group * (K1 * 64) + kstep * 64 + lane_div_16 * 16 + lane_mod_16   (i32)

    where ``group = base_tile // 32`` is the N1 index (one pack-group = the 2
    tiles n_tiles=2), ``K1 = K // 256``, and the i32 holds [K_Pack, N_Pack].
    The MFMA opsel ``ksub*2 + tile_in_pair`` selects the byte at runtime-free.
    One buffer_load per K-step feeds all (tile, ksub) MFMAs of the group.
    """

    def __init__(self, scale_arg, n_tiles, K, lane_id):
        assert n_tiles % _FP4_PACK == 0
        self.n_tiles = n_tiles
        self.n_groups = n_tiles // _FP4_PACK  # pack-groups of 2 tiles (32 rows)
        self.K1 = K // 256
        self.row_stride = self.K1 * 64  # i32 elems per N1 group
        self.lane_off = (lane_id // 16) * 16 + (lane_id % 16)
        self.rsrc = _buffer_ops.create_buffer_resource(scale_arg, max_size=True)

    def load_step(self, kstep, base_tile):
        """list[n_groups] of packed i32 (K=256 step for each 32-row pack-group).
        All addends cast to Int32 so a runtime fx ``kstep`` (scf.for loop var)
        doesn't trip arith.addi's same-type requirement (base_tile is Index)."""
        base_group = fx.Int32(base_tile // 32)
        kterm = fx.Int32(kstep) * fx.Int32(64)
        lane = fx.Int32(self.lane_off)
        out = []
        for g in range_constexpr(self.n_groups):
            i32_off = (base_group + fx.Int32(g)) * fx.Int32(self.row_stride) + kterm + lane
            out.append(_buffer_ops.buffer_load(self.rsrc, i32_off, vec_width=1, dtype=fx.Int32))
        return out


class StoreCFp4:
    """Epilogue: acc(f32x4) -> bf16, no scale mul (scale was applied in MFMA)."""

    def __init__(self, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b, mn_aligned=False):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.mn_aligned = mn_aligned
        c_nbytes = c_rows * c_cols * 2
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False, num_records_bytes=c_nbytes)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        self.reg_bf16_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.BFloat16)

    def _store_bf16(self, value_bf16, c_index):
        fx.memref_store_vec(Vec.filled(1, value_bf16, fx.BFloat16), self.reg_bf16_1)
        fx.copy(self.out_atom_1, self.reg_bf16_1, fx.slice(self.c_div, (None, fx.Int32(c_index))))

    def store(self, c_frag, base_row, base_col):
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + (self.lane_id // 16) * 4
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + tj * 16 + self.lane_id % 16
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                if const_expr(self.mn_aligned):
                    # M/N aligned to BLOCK -> every store in-bounds, no select.
                    for i in range_constexpr(4):
                        scaled = vec_f32[i].to(fx.BFloat16)
                        self._store_bf16(scaled, (row + i) * self.c_cols + col)
                else:
                    # arbitrary M/N: guard each store; OOB redirected to a sentinel
                    # index the bounded buffer resource drops.
                    col_valid = col < self.c_cols
                    oob = fx.Int32(self.c_rows * self.c_cols)
                    for i in range_constexpr(4):
                        scaled = vec_f32[i].to(fx.BFloat16)
                        c_index = (row + i) * self.c_cols + col
                        self._store_bf16(scaled, arith.select(col_valid, c_index, oob))


def compile_fp4_gemm_4w(
    *,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    use_xcd_remap: bool = True,
    mn_aligned: bool = False,
):
    # mn_aligned: caller asserts M % BLOCK_M == 0 and N % BLOCK_N == 0, so every
    # epilogue store is in-bounds -> skip the per-store col-bounds select (saves
    # 256 v_cmp+v_cndmask/wave). Leave False for arbitrary M/N (correctness via the
    # explicit bounds select). Common alignment-fast-path optimization.
    # 256 fp4 per LDS K-step row = 128 bytes; reuse fp8's 128-byte LDS layout.
    BLOCK_K = 256  # fp4 elements
    BLOCK_K_BYTES = BLOCK_K // 2  # 128 bytes / row
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

    assert BLOCK_M % 64 == 0 and BLOCK_N % 64 == 0
    assert K % BLOCK_K == 0

    K_ITERS = K // BLOCK_K
    N_TILES_A = BLOCK_M // 4 // 16
    N_TILES_B = BLOCK_N // 4 // 16
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

    N_LDS_ROUNDS = max(N_TILES_A, N_TILES_B)

    a_lds_size = LDS_BLOCK_M * BLOCK_K_BYTES
    b_lds_size = LDS_BLOCK_N * BLOCK_K_BYTES

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Int8, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Int8, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Int8, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Int8, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Int8, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Int8, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Int8, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Int8, b_lds_size, 16]

    @flyc.kernel
    def kernel_gemm(
        A: fx.Tensor, B_T: fx.Tensor, C: fx.Tensor, A_scale: fx.Tensor, B_scale: fx.Tensor, c_m: fx.Int32, c_n: fx.Int32
    ):
        I8_IR_t = fx.Int8.ir_type

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64

        n_blocks = ceildiv(c_n, BLOCK_N)
        if const_expr(use_xcd_remap):
            tile_i, tile_j = _xcd_swizzle(ceildiv(c_m, BLOCK_M), n_blocks)
        else:
            tile_i, tile_j = divmod(fx.block_idx.x, n_blocks)

        wave_i = wave_id // 2
        wave_j = wave_id % 2

        # Global byte offsets (fp4 packed: K bytes = K // 2).
        K_BYTES = K // 2
        A0_gl_offset = (tile_i * BLOCK_M) * K_BYTES
        A1_gl_offset = (tile_i * BLOCK_M + LDS_BLOCK_M) * K_BYTES
        A_K_STEP = BLOCK_K_BYTES
        B0_gl_offset = (tile_j * BLOCK_N) * K_BYTES
        B1_gl_offset = (tile_j * BLOCK_N + LDS_BLOCK_N) * K_BYTES
        # B is preshuffled (16,16): one N-16 row-block spans 2*1024 bytes per
        # K-step (same constant fp8_gemm_4wave uses for b_preshuffled).
        B_K_STEP = 2 * 1024

        gA = make_fp8_buffer_tensor(A, I8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, I8_IR_t)
        ga_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        gb_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        mfma = Mfma16x16x128Fp4(N_TILES_A, N_TILES_B)

        # One i32 scale per K-step (256 fp4) per M/N-pair; K-step index = k.
        a_scale_ld = ScaleLoader(A_scale, N_TILES_A, K, lane_id)
        b_scale_ld = ScaleLoader(B_scale, N_TILES_B, K, lane_id)

        base_row = tile_i * BLOCK_M + wave_i * (N_TILES_A * 16)
        base_col = tile_j * BLOCK_N + wave_j * (N_TILES_B * 16)
        sa_R0 = base_row
        sa_R1 = base_row + LDS_BLOCK_M
        sb_C0 = base_col
        sb_C1 = base_col + LDS_BLOCK_N

        def _a_sc(k, base):
            return a_scale_ld.load_step(k, base)

        def _b_sc(k, base):
            return b_scale_ld.load_step(k, base)

        def _load_scales(k):
            """All four packed-i32 scales for K-step ``k`` (prefetchable). ``k`` may
            be a compile-time int OR a runtime fx index (load_step's kstep*64 offset
            arithmetic and buffer_load both accept a runtime value)."""
            return (
                _a_sc(k, sa_R0),
                _a_sc(k, sa_R1),
                _b_sc(k, sb_C0),
                _b_sc(k, sb_C1),
            )

        _load_scales_rt = _load_scales  # runtime-k alias used inside the scf.for body

        # Accumulators: 2x2 64x64 quadrants per wave.
        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K_BYTES, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K_BYTES, N_LDS_ROUNDS, preshuffled=True)

        a_g2s = G2SLoader(ga_div, gl_off_a, N_TILES_A, I8_IR_t, wave_id)
        b_g2s = G2SLoader(gb_div, gl_off_b, N_TILES_B, I8_IR_t, wave_id)
        a_s2r = S2RLoaderFp4(wave_i, N_TILES_A)
        b_s2r = S2RLoaderFp4(wave_j, N_TILES_B)
        store_c = StoreCFp4(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B, mn_aligned=mn_aligned)

        # Prologue.
        a_g2s.load(a_cur0, A0_gl_offset + 0 * A_K_STEP)
        b_g2s.load(b_cur0, B0_gl_offset + 0 * B_K_STEP)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * A_K_STEP)

        a_g2s.load(a_next0, A0_gl_offset + 1 * A_K_STEP)
        b_g2s.load(b_next0, B0_gl_offset + 1 * B_K_STEP)
        b_g2s.load(b_next1, B1_gl_offset + 1 * B_K_STEP)
        a_g2s.load(a_next1, A1_gl_offset + 1 * A_K_STEP)

        def _do_quad(a_frag, b_frag, c_frag, sa_ksub, sb_ksub):
            return mfma.call(a_frag, b_frag, c_frag, sa_ksub, sb_ksub)

        wait_barrier((3 * N_TILES_A) + (4 * N_TILES_B))
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier((3 * N_TILES_A) + (3 * N_TILES_B))
        b0_frag = b_s2r.load(b_cur0, preshuffled=True)

        # Step-0 scale (consumed in iter 0); each iter prefetches step k+1.
        saR0, saR1, sbC0, sbC1 = _load_scales(0)

        # ---- Main K-loop as scf.for, unrolled by 2 ------------------------------
        # Why scf.for (not range_constexpr full unroll): fully unrolling all 30 main
        # steps blew .text to ~59KB > 32KB I-cache -> periodic instruction-fetch
        # stalls. Rolling into an scf.for (body = 2 unrolled steps) keeps .text small.
        # Unroll-2 is chosen because the buffer ping-pong swaps the cur<->next LDS
        # pointers exactly twice per body -> identity, so the LDS pointers need NOT
        # be loop-carried. Carried state = the 4 accumulator groups + a0/b0 fragment
        # + the 4 prefetched scales.
        #
        # ``buf`` arg names below are fixed (cur0/cur1/next0/next1); a single step
        # mutates which physical buffer is "cur" via the pointer-pair swap, so the
        # step body is parameterized by the current pointer set passed in.
        def _one_step(kc, a0f, b0f, sc, accs, bufs):
            # bufs = (a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1)
            ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
            saR0, saR1, sbC0, sbC1 = sc
            c00f, c01f, c10f, c11f = accs
            kc_i = fx.Int32(kc)  # Int32 for scale load_step (matches lane_off Int32)
            saR0_n, saR1_n, sbC0_n, sbC1_n = _load_scales_rt(kc_i + 1)

            _b1 = [None] * N_TILES_B
            _a1 = [None] * N_TILES_A
            _a0n = [None] * N_TILES_A
            _b0n = [None] * N_TILES_B
            # This step prefetches K-step (kc+2). g2s offsets fully in Int32
            # (A*_gl_offset is Index; kc_i is the Int32 loop var), so arith.addi
            # operands match. a*_off = base + (kc+2)*A_K_STEP.
            ak = (kc_i + fx.Int32(2)) * fx.Int32(A_K_STEP)
            bk = (kc_i + fx.Int32(2)) * fx.Int32(B_K_STEP)
            a0_off = fx.Int32(A0_gl_offset) + ak
            a1_off = fx.Int32(A1_gl_offset) + ak
            b0_off = fx.Int32(B0_gl_offset) + bk
            b1_off = fx.Int32(B1_gl_offset) + bk

            wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))
            il = _g2s_thunks(a_g2s, ac0, a0_off, N_TILES_A) + _s2r_thunks(b_s2r, bc1, _b1, N_TILES_B, True)
            c00f = mfma.call(a0f, b0f, c00f, saR0, sbC0, interleave=il)
            b1f = _b1

            il = _g2s_thunks(b_g2s, bc0, b0_off, N_TILES_A) + _s2r_thunks(a_s2r, ac1, _a1, N_TILES_A, False)
            c01f = mfma.call(a0f, b1f, c01f, saR0, sbC1, interleave=il)
            a1f = _a1

            wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))
            il = _g2s_thunks(b_g2s, bc1, b1_off, N_TILES_A) + _s2r_thunks(a_s2r, an0, _a0n, N_TILES_A, False)
            c10f = mfma.call(a1f, b0f, c10f, saR1, sbC0, interleave=il)
            a0nf = _a0n

            il = _g2s_thunks(a_g2s, ac1, a1_off, N_TILES_A) + _s2r_thunks(b_s2r, bn0, _b0n, N_TILES_B, True)
            c11f = mfma.call(a1f, b1f, c11f, saR1, sbC1, interleave=il)
            b0nf = _b0n

            new_bufs = (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)  # swap cur<->next
            return a0nf, b0nf, (saR0_n, saR1_n, sbC0_n, sbC1_n), (c00f, c01f, c10f, c11f), new_bufs

        bufs0 = (a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1)
        n_a = 2 * N_TILES_A
        n_b = 2 * N_TILES_B
        _R = arith._to_raw
        n_ga = N_TILES_A // _FP4_PACK  # scale groups per A-scale (=len(saR0))
        n_gb = N_TILES_B // _FP4_PACK

        def _flat_sc(sc4):
            # sc4 = (saR0, saR1, sbC0, sbC1); each is a list of n_g i32
            out = []
            for s in sc4:
                for v in s:
                    out.append(_R(v))
            return out

        def _unflat_sc(flat):
            o = 0
            saR0 = list(flat[o : o + n_ga])
            o += n_ga
            saR1 = list(flat[o : o + n_ga])
            o += n_ga
            sbC0 = list(flat[o : o + n_gb])
            o += n_gb
            sbC1 = list(flat[o : o + n_gb])
            o += n_gb
            return saR0, saR1, sbC0, sbC1

        n_sc = 2 * n_ga + 2 * n_gb
        init_state = (
            _flat_frag(a0_frag)
            + _flat_frag(b0_frag)
            + _flat_sc((saR0, saR1, sbC0, sbC1))
            + [_R(x) for x in c00_frag]
            + [_R(x) for x in c01_frag]
            + [_R(x) for x in c10_frag]
            + [_R(x) for x in c11_frag]
        )
        for kk, state in range(0, K_ITERS - 2, 2, init=init_state):
            off = 0
            a0f = _unflat_frag(state[off : off + n_a], N_TILES_A)
            off += n_a
            b0f = _unflat_frag(state[off : off + n_b], N_TILES_B)
            off += n_b
            sc = _unflat_sc(state[off : off + n_sc])
            off += n_sc
            c00f = list(state[off : off + N_ACCUMS])
            off += N_ACCUMS
            c01f = list(state[off : off + N_ACCUMS])
            off += N_ACCUMS
            c10f = list(state[off : off + N_ACCUMS])
            off += N_ACCUMS
            c11f = list(state[off : off + N_ACCUMS])
            off += N_ACCUMS
            accs = (c00f, c01f, c10f, c11f)

            # step kk
            a0f, b0f, sc, accs, bufs = _one_step(kk, a0f, b0f, sc, accs, bufs0)
            # step kk+1 (pointers swapped once; swap again -> back to bufs0 at exit)
            a0f, b0f, sc, accs, bufs = _one_step(kk + 1, a0f, b0f, sc, accs, bufs)

            new_state = (
                _flat_frag(a0f)
                + _flat_frag(b0f)
                + _flat_sc(sc)
                + [_R(x) for x in accs[0]]
                + [_R(x) for x in accs[1]]
                + [_R(x) for x in accs[2]]
                + [_R(x) for x in accs[3]]
            )
            state = yield new_state

        # unpack final state back into the named vars the tail uses
        off = 0
        a0_frag = _unflat_frag(state[off : off + n_a], N_TILES_A)
        off += n_a
        b0_frag = _unflat_frag(state[off : off + n_b], N_TILES_B)
        off += n_b
        saR0, saR1, sbC0, sbC1 = _unflat_sc(state[off : off + n_sc])
        off += n_sc
        c00_frag = list(state[off : off + N_ACCUMS])
        off += N_ACCUMS
        c01_frag = list(state[off : off + N_ACCUMS])
        off += N_ACCUMS
        c10_frag = list(state[off : off + N_ACCUMS])
        off += N_ACCUMS
        c11_frag = list(state[off : off + N_ACCUMS])
        off += N_ACCUMS

        # Tail step K_ITERS - 2 (scale carried from loop's last prefetch).
        wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))
        b1_frag = b_s2r.load(b_cur1, preshuffled=True)
        c00_frag = _do_quad(a0_frag, b0_frag, c00_frag, saR0, sbC0)
        a1_frag = a_s2r.load(a_cur1)
        c01_frag = _do_quad(a0_frag, b1_frag, c01_frag, saR0, sbC1)
        wait_barrier((1 * N_TILES_A) + (1 * N_TILES_B))
        a0_frag = a_s2r.load(a_next0)
        c10_frag = _do_quad(a1_frag, b0_frag, c10_frag, saR1, sbC0)
        b0_frag = b_s2r.load(b_next0, preshuffled=True)
        c11_frag = _do_quad(a1_frag, b1_frag, c11_frag, saR1, sbC1)

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Tail step K_ITERS - 1.
        saR0, saR1, sbC0, sbC1 = _load_scales(K_ITERS - 1)
        wait_barrier(0)
        b1_frag = b_s2r.load(b_cur1, preshuffled=True)
        a1_frag = a_s2r.load(a_cur1)
        c00_frag = _do_quad(a0_frag, b0_frag, c00_frag, saR0, sbC0)
        c01_frag = _do_quad(a0_frag, b1_frag, c01_frag, saR0, sbC1)
        c10_frag = _do_quad(a1_frag, b0_frag, c10_frag, saR1, sbC0)
        c11_frag = _do_quad(a1_frag, b1_frag, c11_frag, saR1, sbC1)

        store_c.store(c00_frag, sa_R0, sb_C0)
        store_c.store(c01_frag, sa_R0, sb_C1)
        store_c.store(c10_frag, sa_R1, sb_C0)
        store_c.store(c11_frag, sa_R1, sb_C1)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm
