# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""8-wave FP8 matmul with row-wise scaling for AMD CDNA4.

Algorithm derived from HipKittens FP8_8wave
(https://github.com/HazyResearch/HipKittens/blob/7782744ba1fd259a377a99e2ea8f71384cc80e55/kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu#L1)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from kernels.gemm.fp8_gemm_utils import (
    G2SLoader,
    S2RLoader,
    StoreC,
    ceildiv,
    compute_global_swizzle,
    divmod,
    make_fp8_buffer_tensor,
    wait_barrier,
)


def _layout_global_swizzle(lane_id, wave_id, K, n_rounds, block_dim_x):
    """Layout-API form of ``compute_global_swizzle(preshuffled=False)``.

    Carries the XOR bank-avoidance swizzle in a
    ``make_composed_layout(SwizzleType.get(3, 4, 4), ...)`` and reads the
    swizzled column back with ``crd2idx`` on concrete (compile-time) coords
    (``get_scalar`` -> a constant), instead of the hand-written ``swizzle_128``
    bit math. ``swizzle_128`` was verified equal to ``SwizzleType.get(3, 4, 4)``
    and this offset list is byte-identical to the helper's, so the emitted
    ``buffer_load_lds`` addressing is unchanged.
    """
    swz_layout = fx.make_composed_layout(
        fx.static(fx.SwizzleType.get(3, 4, 4)),
        fx.make_ordered_layout((16, 128), (1, 0)),
    )
    n_waves = block_dim_x // 64
    offsets = []
    for rnd in range_constexpr(n_rounds):
        row = lane_id // 8 + wave_id * 8 + rnd * (n_waves * 8)
        col = (lane_id % 8) * 16
        # swizzle_128 keeps r == row (the XOR only permutes columns), so the
        # global element is row*K + swizzled_col, with swizzled_col periodic in
        # 16 rows -> take (row % 16, col) through the composed layout.
        swz_col = fx.get_scalar(fx.crd2idx((row % 16, col), swz_layout)) % 128
        offsets.append(row * K + swz_col)
    return offsets


class LayoutS2R:
    """Shared->register reader with the XOR swizzle carried in a composed layout.

    Mirrors ``S2RLoader.load(preshuffled=False)`` but computes the swizzled LDS
    byte offset with ``crd2idx`` over
    ``make_composed_layout(SwizzleType.get(3, 4, 4), ...)`` (``get_scalar`` ->
    compile-time constant) instead of the hand-written ``swizzle_128`` math.
    ``swizzle_128`` was verified equal to ``SwizzleType.get(3, 4, 4)`` and the
    resulting offsets are byte-identical, so the emitted ``ds_read`` addressing
    is unchanged. The split-16@64 i32x8 packing (``pack_i32x4_i32x8``) is kept
    as-is -- it is the MFMA operand ABI. Preshuffled B stays on ``S2RLoader``
    (a different affine, non-XOR LDS map).
    """

    def __init__(self, wave_idx, n_tiles):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles
        self.swz_layout = fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(3, 4, 4)),
            fx.make_ordered_layout((128, 128), (1, 0)),
        )

    def _vec_load_16xf8(self, lds_src, offset):
        ptr_off = fx.add_offset(lds_src.ptr, fx.make_int_tuple(offset))
        i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
        return fx.make_view(i8_iter, fx.make_layout(16, 1)).load()

    def load(self, lds_src, preshuffled=False):
        assert not preshuffled, "LayoutS2R only handles the non-preshuffled (XOR) LDS layout"
        frag = []
        for i in range_constexpr(self.n_tiles):
            halves = []
            row = self.wave_idx * (self.n_tiles * 16) + i * 16 + self.lane_id % 16
            for step in range_constexpr(2):
                col = (self.lane_id // 16) * 16 + step * 64
                offset = fx.get_scalar(fx.crd2idx((row, col), self.swz_layout))
                v = self._vec_load_16xf8(lds_src, offset)
                halves.append(v.bitcast(fx.Int32))
            frag.append(_pack_i32x4_i32x8(halves[0], halves[1]))
        return frag


def _pack_i32x4_i32x8(lo, hi):
    return lo.shuffle(hi, list(range(8)))


class TiledMmaDriver:
    """MMA driver in the example-04 layout-API idiom.

    Replaces ``Mfma16x16x128``'s bare-atom ``fx.gemm(atom, ...)`` calls with
    ``fx.gemm(tiled_mma, ...)`` over a ``make_tiled_mma``. The 16x16x128 fp8
    operands stay the split-16@64-packed i32x8 / f32x4 register fragments
    produced by ``S2RLoader`` and consumed by ``StoreC``; only the MMA
    construction moves to the layout API.
    """

    def __init__(self, tiled_mma, n_tiles_a, n_tiles_b):
        self.tiled_mma = tiled_mma
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _make_operand_frag(self, value):
        frag = fx.make_rmem_tensor(8, fx.Int32)
        frag.store(Vec(value))
        return frag

    def _make_accum_frag(self, value):
        frag = fx.make_rmem_tensor(4, fx.Float32)
        frag.store(Vec(value))
        return frag

    def call(self, a, b, c, *, set_prio=True):
        assert len(a) == self.n_tiles_a
        assert len(b) == self.n_tiles_b
        assert len(c) == self.n_tiles_a * self.n_tiles_b

        a_frags = [self._make_operand_frag(a[idx]) for idx in range_constexpr(self.n_tiles_a)]
        b_frags = [self._make_operand_frag(b[idx]) for idx in range_constexpr(self.n_tiles_b)]
        c_frags = [self._make_accum_frag(c[idx]) for idx in range_constexpr(self.n_tiles_a * self.n_tiles_b)]
        if const_expr(set_prio):
            rocdl.s_setprio(1)
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                cf = c_frags[self.idx(i, j)]
                fx.gemm(self.tiled_mma, cf, a_frags[i], b_frags[j], cf)
        if const_expr(set_prio):
            rocdl.s_setprio(0)
            rocdl.s_barrier()
        return [c_frags[idx].load().ir_value() for idx in range_constexpr(self.n_tiles_a * self.n_tiles_b)]


def compile_fp8_gemm_8w(*, K: int, BLOCK_M: int = 256, BLOCK_N: int = 256, b_preshuffled: bool = False):
    BLOCK_K = 128

    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0

    K_ITERS = K // BLOCK_K

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

    # half size
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type

        # Layout-API MMA construction (example-04 idiom): a single 16x16x128 fp8
        # atom, one MFMA per wave sub-tile call (no wave/permutation
        # replication). Built in-kernel because the operands are the raw
        # split-16@64-packed i32x8 register fragments from ``S2RLoader`` rather
        # than TV-partitioned ``make_fragment_*`` tensors, so ``fx.gemm`` needs
        # the concrete ``tiled_mma`` value to take the tiled lowering path.
        tiled_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN)),
            fx.make_layout((1, 1, 1), (0, 0, 0)),
        )

        n_blocks = ceildiv(c_n, BLOCK_N)

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
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        block_m, block_n = divmod(fx.block_idx.x, n_blocks)

        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        B_K_STEP = (2 * 1024) if b_preshuffled else BLOCK_K
        B0_gl_offset = (block_n * BLOCK_N) * K
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = _layout_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, fx.block_dim.x)
        if const_expr(b_preshuffled):
            gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=True)
        else:
            gl_off_b = _layout_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, fx.block_dim.x)

        mfma = TiledMmaDriver(tiled_mma, N_TILES_A, N_TILES_B)

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = LayoutS2R(wave_m, N_TILES_A)
        if const_expr(b_preshuffled):
            b_s2r = S2RLoader(wave_n, N_TILES_B)
        else:
            b_s2r = LayoutS2R(wave_n, N_TILES_B)
        store_c = StoreC(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        # 2x2 config of 4x2 (instead of 4x4 in 4wave) 16x16 sub-tiles
        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        b_g2s.load(b_cur0, B0_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * B_K_STEP)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * B_K_STEP)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        for k in range_constexpr(K_ITERS - 2):
            b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

            b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP)
            rocdl.s_barrier()

            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * B_K_STEP)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

            # Swap cur and next
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 2
        k = K_ITERS - 2
        b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
        rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)

        a1_frag = a_s2r.load(a_cur1)
        # Main loop prefetches a_next1 one step behind; issue the final
        # K_ITERS - 1 tile here, otherwise c10 / c11 read stale A1 data.
        a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
        rocdl.s_barrier()

        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)

        b0_frag = b_s2r.load(b_next0, preshuffled=b_preshuffled)
        rocdl.s_barrier()

        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        # Swap cur and next
        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 1
        k = K_ITERS - 1
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier(0)

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)

        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
        rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)

        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, set_prio=False)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, set_prio=False)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        # Scale and store back to gmem
        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

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
            value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "512,512"},
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_gemm
