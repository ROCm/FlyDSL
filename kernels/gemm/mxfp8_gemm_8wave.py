# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""8-wave MXFP8 matmul for AMD CDNA4 (gfx950 / MI355X)."""

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
    xcd_remap_pid,
)

# 1 block = 512 threads = 8 waves (2 in M x 4 in N); the LDS budget below only
# closes for this shape, see ``compile_mxfp8_gemm_8w``.
BLOCK_K = 128
LDS_LIMIT_BYTES = 160 * 1024


class ScalePreshuffledS2R:
    """Coalesced reader for ``shuffle_scale_w4``-packed E8M0 -- no LDS staging.

    That layout flattens to ``i32[(n1 * K1 + k1) * 64 + lane]`` with
    ``lane = k_lane * 16 + n_lane``, i.e. one dword per lane and **64 consecutive
    dwords per wave** (256 B, fully coalesced), against the raw layout's 64
    separate cache lines. The lane split is the same one this kernel already
    uses: ``n_lane = lane % 16`` is the operand row and ``k_lane = lane // 16``
    the MX block inside the K-step.

    The four bytes of that dword are two 16-row tiles (``n_pack``) x two K-steps
    (``k_pack``), both compile-time indices here, so the byte is selected by the
    atom's ``opsel`` and no shift is emitted. One load therefore feeds four
    MFMAs, and consecutive tiles of a pair share it.
    """

    def __init__(self, scale_arg, rows, K, n_tiles):
        assert n_tiles % 2 == 0, "shuffle_scale_w4 pairs tiles two at a time"
        self.n_pairs = n_tiles // 2
        self.k1_stride = K // 256  # i32 groups of 64 per 32-row super-row
        self.lane = fx.thread_idx.x % 64
        # Same byte count as the raw layout, just permuted.
        t_i8 = fx.rocdl.make_buffer_tensor(
            scale_arg, max_size=False, num_records_bytes=fx.Int64(rows) * fx.Int64(K // 32)
        )
        i32_ptr = fx.PointerType.get(
            elem_ty=fx.Int32.ir_type, address_space=fx.rocdl.TargetAddressSpace.BufferDesc, alignment=4
        )
        iter_i32 = fx.recast_iter(i32_ptr, fx.get_iter(t_i8))
        n_i32 = fx.Int32(rows) * fx.Int32(K // 128)
        self.g_div = fx.logical_divide(
            fx.Tensor(fx.make_view(iter_i32, fx.make_layout(n_i32, 1))), fx.make_layout(1, 1)
        )
        self.atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        # Two register sets: the caller prefetches the next K-pair while the
        # current one is still feeding MFMAs.
        self.regs = [
            [fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32) for _ in range_constexpr(self.n_pairs)]
            for _ in range_constexpr(2)
        ]

    def read(self, row_base16, k):
        """``n_tiles`` scale operands for K-step ``k``; tiles of a pair share one."""
        k1 = k // 2  # compile-time; k % 2 is the k_pack the opsel encodes
        regs = self.regs[k1 % 2]
        words = []
        for p in range_constexpr(self.n_pairs):
            # row_base16 is even (wave offsets are multiples of 32 rows), so the
            # 32-row super-row is row_base16 // 2 + p and n_pack is the tile parity.
            n1 = row_base16 // 2 + p
            base = fx.rocdl.readfirstlane(fx.Int32.ir_type, (n1 * self.k1_stride + k1) * 64)
            fx.copy(self.atom, fx.slice(self.g_div, (None, fx.Int32(base) + self.lane)), regs[p])
            w = fx.Int32(regs[p].load()[0])
            words += [w, w]
        return words


class MxMfma:
    """16x16x128 scaled-MFMA driver (bare atom: ``TiledMma`` has no ``set_value``).

    One ``(opsel_a, opsel_b)`` atom per byte pair: in the ``shuffle_scale_w4``
    layout a single i32 carries the E8M0 of two 16-row tiles x two K-steps, and
    the byte is picked by ``opsel`` -- a compile-time atom field -- so the hot
    loop emits no byte-select instructions at all.
    """

    def __init__(self, n_tiles_a, n_tiles_b):
        # opsel = k_pack * 2 + tile_in_pair, so both operands share k_pack.
        self.atoms = {
            (kp * 2 + ia, kp * 2 + jb): fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN, opsel_a=kp * 2 + ia, opsel_b=kp * 2 + jb)
            )
            for kp in range_constexpr(2)
            for ia in range_constexpr(2)
            for jb in range_constexpr(2)
        }
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _operand(self, value):
        frag = fx.make_rmem_tensor(8, fx.Int32)
        frag.store(Vec(value))
        return frag

    def _accum(self, value):
        frag = fx.make_rmem_tensor(4, fx.Float32)
        frag.store(Vec(value))
        return frag

    def _atom_for(self, k_pack, i, j):
        return self.atoms[(k_pack * 2 + i % 2, k_pack * 2 + j % 2)]

    def call(self, a, b, c, sa, sb, *, k_pack, set_prio=True):
        assert len(a) == self.n_tiles_a and len(sa) == self.n_tiles_a
        assert len(b) == self.n_tiles_b and len(sb) == self.n_tiles_b
        assert len(c) == self.n_tiles_a * self.n_tiles_b

        a_frags = [self._operand(a[i]) for i in range_constexpr(self.n_tiles_a)]
        b_frags = [self._operand(b[j]) for j in range_constexpr(self.n_tiles_b)]
        c_frags = [self._accum(c[i]) for i in range_constexpr(self.n_tiles_a * self.n_tiles_b)]
        if const_expr(set_prio):
            rocdl.s_setprio(1)
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                cf = c_frags[self.idx(i, j)]
                atom = self._atom_for(k_pack, i, j)
                fx.gemm(atom, cf, a_frags[i], b_frags[j], cf, scale_a=sa[i], scale_b=sb[j])
        if const_expr(set_prio):
            rocdl.s_setprio(0)
            rocdl.s_barrier()
        return [c_frags[i].load().ir_value() for i in range_constexpr(self.n_tiles_a * self.n_tiles_b)]


def compile_mxfp8_gemm_8w(
    *,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    b_preshuffled: bool = False,
    xcd_swizzle: int = 0,
):
    """Build the MXFP8 launcher."""
    # A 256x256 tile already spends 128 KB of the 160 KB LDS; every other
    # BLOCK_M/BLOCK_N either overflows it or breaks the 2-threads-per-scale-row
    # staging split. Kept as an assert rather than dead generality.
    assert BLOCK_M == 256 and BLOCK_N == 256, "MXFP8 8-wave is specialized to a 256x256 block tile"
    assert K % 256 == 0, f"K must be a multiple of 256 (MX scale chunk staging), got {K}"

    K_ITERS = K // BLOCK_K
    # Scale words are addressed by K-pair, so K must contain a whole number of
    # them (K % 256 above already guarantees it).
    assert K_ITERS % 2 == 0

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B

    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    A_GRP_ROWS = N_TILES_A * 16  # rows one wave_m owns inside one LDS half
    B_GRP_ROWS = N_TILES_B * 16

    lds_bytes = 4 * (a_lds_size + b_lds_size)
    assert lds_bytes <= LDS_LIMIT_BYTES, f"LDS {lds_bytes} B exceeds {LDS_LIMIT_BYTES} B"

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
        if const_expr(xcd_swizzle > 0):
            block_m, block_n = xcd_remap_pid(ceildiv(c_m, BLOCK_M), n_blocks, group_m=xcd_swizzle)
        else:
            block_m, block_n = divmod(fx.block_idx.x, n_blocks)

        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        # ``preshuffle_b`` keeps a row's identity (it only reorders K within a
        # 16-row x 64-K brick), so a row's base offset is still row * K and the
        # raw per-row E8M0 scale needs no host-side change; only the K step and
        # the in-tile K order differ.
        B_K_STEP = (2 * 1024) if b_preshuffled else BLOCK_K
        B0_gl_offset = (block_n * BLOCK_N) * K
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=b_preshuffled)

        mfma = MxMfma(N_TILES_A, N_TILES_B)

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        store_c = StoreC(None, None, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        a_sc = ScalePreshuffledS2R(A_scale, c_m, K, N_TILES_A)
        b_sc = ScalePreshuffledS2R(B_scale, c_n, K, N_TILES_B)
        # 16-row tile index of each wave's first row, per LDS half.
        a_base16 = [(block_m * BLOCK_M + h * LDS_BLOCK_M + wave_m * A_GRP_ROWS) // 16 for h in range_constexpr(2)]
        b_base16 = [(block_n * BLOCK_N + h * LDS_BLOCK_N + wave_n * B_GRP_ROWS) // 16 for h in range_constexpr(2)]

        sc_pf = {}

        def scale_prefetch(k0):
            """Issue the scale loads for the K-pair starting at ``k0``."""
            if const_expr(k0 < K_ITERS and k0 // 2 not in sc_pf):
                sc_pf[k0 // 2] = {
                    w: (a_sc if w[0] == "a" else b_sc).read((a_base16 if w[0] == "a" else b_base16)[int(w[1])], k0)
                    for w in ("a0", "a1", "b0", "b1")
                }

        def scale_read(k, which):
            """The prefetched E8M0 operands for K-step ``k``."""
            return sc_pf[k // 2][which]

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        scale_prefetch(0)

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
            if const_expr(k % 2 == 1):
                scale_prefetch(k + 1)
            sa0 = scale_read(k, "a0")
            sb0 = scale_read(k, "b0")
            b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2)

            sb1 = scale_read(k, "b1")
            b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP)
            rocdl.s_barrier()

            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2)

            sa1 = scale_read(k, "a1")
            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0, k_pack=k % 2)

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * B_K_STEP)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1, k_pack=k % 2)

            # Swap cur and next
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 2
        k = K_ITERS - 2
        sa0 = scale_read(k, "a0")
        sb0 = scale_read(k, "b0")
        b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2)

        sb1 = scale_read(k, "b1")
        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
        rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2)

        sa1 = scale_read(k, "a1")
        a1_frag = a_s2r.load(a_cur1)
        # Main loop prefetches a_next1 one step behind; issue the final
        # K_ITERS - 1 tile here, otherwise c10 / c11 read stale A1 data.
        a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
        rocdl.s_barrier()

        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0, k_pack=k % 2)

        b0_frag = b_s2r.load(b_next0, preshuffled=b_preshuffled)
        rocdl.s_barrier()

        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1, k_pack=k % 2)
        # Swap cur and next
        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 1
        k = K_ITERS - 1
        sa0 = scale_read(k, "a0")
        sb0 = scale_read(k, "b0")
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier(0)

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2)

        sb1 = scale_read(k, "b1")
        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
        rocdl.s_barrier()

        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2)

        sa1 = scale_read(k, "a1")
        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0, k_pack=k % 2, set_prio=False)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1, k_pack=k % 2, set_prio=False)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        # Accumulators are already scaled by the MFMA: convert and store.
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
