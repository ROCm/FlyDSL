# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Narrow-tile MXFP8 matmul for AMD CDNA4 (gfx950 / MI355X).

The 256x256 / 8-wave kernel already runs its CU at MFMA peak, so problems that
only fill a few dozen of the 256 CUs cannot go faster on that tile: they need
more workgroups, which means a smaller tile.  This module is the family of
narrow tiles the 8-wave entry point dispatches to -- a 128x128 one and a 64x64
one today -- parameterised by ``(BLOCK_M, BLOCK_N)`` and a wave/slab grid so
both come out of one code path.

Four things differ from the 8-wave kernel beyond the tile:

* the wave grid is ``CFGS[cfg]`` -- the column tile has to keep every wave's
  first 16-row tile index even, because ``shuffle_scale_w4`` packs the two
  16-row halves of a 32-row super-row into one word and the MFMA ``opsel`` bit
  that picks the half is a compile-time constant per fragment tile;
* the LDS staging is a ``stages``-deep ring with one rendezvous per step instead
  of the two-stage ping-pong, because a smaller tile leaves less compute per
  step to hide the global-load latency behind;
* the MX scale words are issued two steps before they are read, so their load
  latency lands under the MFMAs in between;
* the wave grid, the LDS slab count and the tile are all derived from the
  requested ``(BLOCK_M, BLOCK_N)``, so the caller picks the shape that matches
  the problem's workgroup count.

The XOR-swizzled LDS layout, the coalesced global read pattern and the MFMA
driver are shared with the 8-wave kernel.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch
from kernels.common.kernels_common import get_warp_size
from kernels.gemm.fp8_gemm_utils import (
    StoreC,
    ceildiv,
    compute_global_swizzle,
    divmod,
    make_fp8_buffer_tensor,
    pack_i32x4_i32x8,
    swizzle_128,
    xcd_remap_pid,
)
from kernels.gemm.mxfp8_gemm_8wave import MxMfma

BLOCK_K = 128
LDS_LIMIT_BYTES = 160 * 1024

# BLOCK_M, BLOCK_N -> (waves in M, waves in N, LDS slabs in M, LDS slabs in N,
# ring stages).  A wave owns 16 * N_TILES_A rows of each M slab and
# 16 * N_TILES_B columns of each N slab; both tile counts must be even for the
# paired MX scale words, and the slab has to fill in whole copy rounds (a round
# is ``n_waves * 8`` rows).
#
# These two are the shapes ``compile_mxfp8_gemm_8w`` dispatches to.  Others were
# measured and lost: halving one side only (256x128, 128x64, 64x128, 256x64) is
# worse than the square of the same area because it costs workgroups without
# buying back enough reuse, and a 256x256 ring loses by 25% to the 8-wave
# kernel's two-stage ping-pong -- one ring slot refilled per K-step cannot hide a
# 256-row tile's load latency the way the ping-pong's split halves do.
CFGS = {
    (128, 128): (2, 4, 2, 1, 4),
    (64, 64): (1, 2, 1, 1, 4),
}


def wait_vmcnt(count):
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=f"s_waitcnt vmcnt({count})",
        constraints="",
        has_side_effects=True,
    )


class StageG2S:
    """``G2SLoader`` over a byte offset into one flat LDS array (the ring)."""

    def __init__(self, gl_src, gl_offsets, n_load_steps, lds_dtype):
        self.g2lds_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.LdsPtr_t = fx.PointerType.get(lds_dtype, 2, 512)
        self.gl_src = gl_src
        self.gl_offsets = gl_offsets
        self.n_load_steps = n_load_steps
        self.wave_id = fx.thread_idx.x // 64
        self.n_waves = fx.block_dim.x // 64
        self.base_i32 = None

    def set_base(self, lds):
        self.base_i32 = fx.Int32(fx.ptrtoint(lds.ptr))

    def load(self, base_off, k_offset):
        for step in range_constexpr(self.n_load_steps):
            src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[step])))
            off = base_off + self.wave_id * 1024 + step * (self.n_waves * 1024)
            dst = fx.make_view(fx.inttoptr(self.LdsPtr_t, self.base_i32 + fx.Int32(off)), fx.make_layout(1, 1))
            fx.copy(self.g2lds_atom, src, dst, soffset=fx.Int32(k_offset))


class StageS2R:
    """``S2RLoader`` over a byte offset into one flat LDS array (the ring)."""

    def __init__(self, lds, wave_idx, n_tiles):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles
        self.ptr = lds.ptr

    def _vec_load_16xf8(self, offset):
        ptr_off = fx.add_offset(self.ptr, fx.make_int_tuple(offset))
        i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
        return fx.make_view(i8_iter, fx.make_layout(16, 1)).load()

    def load(self, base_off, preshuffled=False):
        frag = []
        for i in range_constexpr(self.n_tiles):
            halves = []
            row = self.wave_idx * (self.n_tiles * 16) + i * 16 + self.lane_id % 16
            for step in range_constexpr(2):
                col = (self.lane_id // 16) * 16 + step * 64
                if const_expr(preshuffled):
                    offset = (row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128
                else:
                    row_swz, col_swz = swizzle_128(row, col)
                    offset = row_swz * 128 + col_swz
                v = self._vec_load_16xf8(base_off + offset)
                halves.append(v.bitcast(fx.Int32))
            frag.append(pack_i32x4_i32x8(halves[0], halves[1]))
        return frag


class ScaleRing:
    """``shuffle_scale_w4`` E8M0 words, issued before the steps that read them.

    Same coalesced 32-row-super-row read as the 8-wave kernel's
    ``ScalePreshuffledS2R``, but split into ``issue`` / ``value`` so the load
    latency lands under the MFMAs of the steps in between instead of on the
    dependency that consumes it.
    """

    def __init__(self, scale_arg, row_base16, K, n_tiles, rows):
        assert n_tiles % 2 == 0, "shuffle_scale_w4 pairs tiles two at a time"
        self.n_pairs = n_tiles // 2
        self.k1_stride = K // 256  # i32 groups of 64 per 32-row super-row
        self.row_base16 = row_base16
        self.lane = fx.thread_idx.x % 64
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
        self.regs = [
            [fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32) for _ in range_constexpr(self.n_pairs)]
            for _ in range_constexpr(2)
        ]

    def issue(self, k):
        regs = self.regs[(k // 2) % 2]
        for p in range_constexpr(self.n_pairs):
            # row_base16 is even (wave offsets are multiples of 32 rows), so the
            # 32-row super-row is row_base16 // 2 + p and n_pack is the tile parity.
            n1 = self.row_base16 // 2 + p
            base = fx.rocdl.readfirstlane(fx.Int32.ir_type, (n1 * self.k1_stride + k // 2) * 64)
            fx.copy(self.atom, fx.slice(self.g_div, (None, fx.Int32(base) + self.lane)), regs[p])

    def value(self, k):
        regs = self.regs[(k // 2) % 2]
        words = []
        for p in range_constexpr(self.n_pairs):
            w = fx.Int32(regs[p].load()[0])
            words += [w, w]
        return words


def describe(block_m, block_n, stages=None):
    """Derived geometry of the ``(block_m, block_n)`` tile, or ``None``."""
    cfg = CFGS.get((block_m, block_n))
    if cfg is None:
        return None
    n_waves_m, n_waves_n, n_slab_m, n_slab_n, default_stages = cfg
    if stages is None:
        stages = default_stages
    n_waves = n_waves_m * n_waves_n
    lds_bm = block_m // n_slab_m
    lds_bn = block_n // n_slab_n
    if lds_bm % (n_waves_m * 16) or lds_bn % (n_waves_n * 16):
        return None
    n_tiles_a = lds_bm // (n_waves_m * 16)
    n_tiles_b = lds_bn // (n_waves_n * 16)
    if n_tiles_a % 2 or n_tiles_b % 2:
        return None
    if lds_bm % (n_waves * 8) or lds_bn % (n_waves * 8):
        return None
    a_slab = lds_bm * BLOCK_K
    b_slab = lds_bn * BLOCK_K
    if stages * (n_slab_m * a_slab + n_slab_n * b_slab) > LDS_LIMIT_BYTES:
        return None
    return {
        "n_waves": n_waves,
        "n_waves_m": n_waves_m,
        "n_waves_n": n_waves_n,
        "n_slab_m": n_slab_m,
        "n_slab_n": n_slab_n,
        "stages": stages,
        "lds_bm": lds_bm,
        "lds_bn": lds_bn,
        "n_tiles_a": n_tiles_a,
        "n_tiles_b": n_tiles_b,
        "n_steps_a": lds_bm // (n_waves * 8),
        "n_steps_b": lds_bn // (n_waves * 8),
        "a_slab": a_slab,
        "b_slab": b_slab,
    }


def compile_mxfp8_gemm_tile(
    *,
    K: int,
    BLOCK_M: int,
    BLOCK_N: int,
    b_preshuffled: bool = False,
    xcd_swizzle: int = 0,
    stages: "int | None" = None,
):
    """Build the ``BLOCK_M x BLOCK_N`` MXFP8 launcher for ``shuffle_scale_w4`` scales."""
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx950"), f"MXFP8 tile GEMM requires gfx950 (CDNA4), got {arch}"
    assert get_warp_size() == 64, f"MXFP8 tile GEMM assumes wave64, got {get_warp_size()}"
    assert K % 256 == 0, f"K must be a multiple of 256 (MX scale chunk staging), got {K}"

    K_ITERS = K // BLOCK_K
    assert K_ITERS % 2 == 0

    geo = describe(BLOCK_M, BLOCK_N, stages)
    assert geo is not None, f"no feasible wave grid for a {BLOCK_M}x{BLOCK_N} tile"
    n_waves = geo["n_waves"]
    n_waves_n = geo["n_waves_n"]
    n_slab_m = geo["n_slab_m"]
    n_slab_n = geo["n_slab_n"]
    stages = geo["stages"]
    LDS_BLOCK_M = geo["lds_bm"]
    LDS_BLOCK_N = geo["lds_bn"]
    N_TILES_A = geo["n_tiles_a"]
    N_TILES_B = geo["n_tiles_b"]
    N_LDS_STEPS_A = geo["n_steps_a"]
    N_LDS_STEPS_B = geo["n_steps_b"]
    a_slab = geo["a_slab"]
    b_slab = geo["b_slab"]
    n_threads = 64 * n_waves

    A_GRP_ROWS = N_TILES_A * 16
    B_GRP_ROWS = N_TILES_B * 16

    OPS_PER_STEP = n_slab_m * N_LDS_STEPS_A + n_slab_n * N_LDS_STEPS_B
    SCALE_OPS_PER_PAIR = n_slab_m * (N_TILES_A // 2) + n_slab_n * (N_TILES_B // 2)
    N_PAIRS = K_ITERS // 2

    # Compile-time issue schedule.  ``pre_wait[k]`` is how many VMEM ops this wave
    # may still have in flight once everything K-step ``k`` consumes is complete.
    # The ring is filled ``PIPE = stages - 1`` steps ahead, so the stage refilled
    # at step ``k`` is the one every wave finished reading at step ``k - 1``.
    #
    # K-step ``k`` consumes two things: its own staged tile (``load`` event ``k``)
    # and MX scale pair ``k // 2`` (``scale`` event, issued two steps earlier).
    # The scale loads are issued *after* the step-(k-2) tile loads, so a wait
    # counted only from the tile loads leaves them in flight -- hence the explicit
    # per-event positions below rather than a rolling counter.
    PIPE = stages - 1
    assert PIPE >= 1, "the ring needs at least one step of prefetch"
    # (issued_at_step, n_ops, name); step -1 is the prologue.
    issue_log = [(-1, SCALE_OPS_PER_PAIR, ("scale", 0))]
    for k in range(min(PIPE, K_ITERS)):
        issue_log.append((-1, OPS_PER_STEP, ("load", k)))
    for k in range(K_ITERS):
        if k % 2 == 0 and k // 2 + 1 < N_PAIRS:
            issue_log.append((k, SCALE_OPS_PER_PAIR, ("scale", k // 2 + 1)))
        if k + PIPE < K_ITERS:
            issue_log.append((k, OPS_PER_STEP, ("load", k + PIPE)))
    # Cumulative op index each event finishes at (1-based: index of its last op).
    done_at = {}
    total = 0
    for _, n_ops, name in issue_log:
        total += n_ops
        done_at[name] = total
    issued_before = []
    running = sum(n for step, n, _ in issue_log if step < 0)
    for k in range(K_ITERS):
        issued_before.append(running)
        running += sum(n for step, n, _ in issue_log if step == k)
    pre_wait = []
    for k in range(K_ITERS):
        need = max(done_at[("load", k)], done_at[("scale", k // 2)])
        pre_wait.append(issued_before[k] - need)
        assert pre_wait[k] >= 0
    del total

    @fx.struct
    class SharedStorage:
        A_lds: fx.Array[fx.Float8E4M3FN, stages * n_slab_m * a_slab, 16]
        B_lds: fx.Array[fx.Float8E4M3FN, stages * n_slab_n * b_slab, 16]

    @flyc.kernel(known_block_size=[n_threads, 1, 1])
    def kernel_gemm_tile(
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

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // n_waves_n
        wave_n = wave_id % n_waves_n
        if const_expr(xcd_swizzle > 0):
            block_m, block_n = xcd_remap_pid(ceildiv(c_m, BLOCK_M), n_blocks, group_m=xcd_swizzle)
        else:
            block_m, block_n = divmod(fx.block_idx.x, n_blocks)

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_STEPS_A, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_STEPS_B, preshuffled=b_preshuffled)

        mfma = MxMfma(N_TILES_A, N_TILES_B, paired_tiles=True)

        a_g2s = StageG2S(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t)
        b_g2s = StageG2S(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t)
        a_g2s.set_base(lds.A_lds)
        b_g2s.set_base(lds.B_lds)
        a_s2r = StageS2R(lds.A_lds, wave_m, N_TILES_A)
        b_s2r = StageS2R(lds.B_lds, wave_n, N_TILES_B)
        store_c = StoreC(None, None, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        # 16-row tile index of each wave's first row, per LDS slab.
        a_base16 = [
            (block_m * BLOCK_M + h * LDS_BLOCK_M + wave_m * A_GRP_ROWS) // 16 for h in range_constexpr(n_slab_m)
        ]
        b_base16 = [
            (block_n * BLOCK_N + h * LDS_BLOCK_N + wave_n * B_GRP_ROWS) // 16 for h in range_constexpr(n_slab_n)
        ]
        a_sc = [ScaleRing(A_scale, a_base16[h], K, N_TILES_A, c_m) for h in range_constexpr(n_slab_m)]
        b_sc = [ScaleRing(B_scale, b_base16[h], K, N_TILES_B, c_n) for h in range_constexpr(n_slab_n)]

        def a_off(stage, slab):
            return (stage * n_slab_m + slab) * a_slab

        def b_off(stage, slab):
            return (stage * n_slab_n + slab) * b_slab

        # Row-major [rows, K] bytes either way: ``shuffle_weight_w4`` only
        # permutes inside a row block, so a K step advances by 2 KB there.
        b_k_stride = 2 * 1024 if b_preshuffled else BLOCK_K

        def load_step(k, stage):
            """Stage K-step ``k``: every LDS slab of both operands."""
            for h in range_constexpr(n_slab_m):
                a_g2s.load(a_off(stage, h), (block_m * BLOCK_M + h * LDS_BLOCK_M) * K + k * BLOCK_K)
            for h in range_constexpr(n_slab_n):
                b_g2s.load(
                    b_off(stage, h),
                    (block_n * BLOCK_N + h * LDS_BLOCK_N) * K + k * b_k_stride,
                )

        def issue_scales(k):
            for h in range_constexpr(n_slab_m):
                a_sc[h].issue(k)
            for h in range_constexpr(n_slab_n):
                b_sc[h].issue(k)

        n_accums = N_TILES_A * N_TILES_B
        c_frag = [
            [([mfma.zero_value] * n_accums) for _ in range_constexpr(n_slab_n)] for _ in range_constexpr(n_slab_m)
        ]

        issue_scales(0)
        for k in range_constexpr(min(PIPE, K_ITERS)):
            load_step(k, k % stages)

        for k in range_constexpr(K_ITERS):
            stage = k % stages
            # Every op issued after this step's own loads may stay in flight.
            wait_vmcnt(pre_wait[k])
            # One rendezvous per step.  It publishes this stage's global reads to
            # the whole workgroup, and -- because each wave drains its own LDS
            # reads first -- it also retires the ring slot the refill below
            # overwrites: ``PIPE = stages - 1``, so that refill targets the slot
            # every wave finished reading at step ``k - 1``.  The slot this step
            # reads is a different one, so the refill cannot race the loads below.
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()

            a_frags = [a_s2r.load(a_off(stage, h)) for h in range_constexpr(n_slab_m)]
            b_frags = [b_s2r.load(b_off(stage, h), preshuffled=b_preshuffled) for h in range_constexpr(n_slab_n)]
            sa = [a_sc[h].value(k) for h in range_constexpr(n_slab_m)]
            sb = [b_sc[h].value(k) for h in range_constexpr(n_slab_n)]

            # Scales first: they are then older than this step's tile loads, so
            # the rendezvous two steps from now already guarantees them.
            if const_expr(k % 2 == 0 and k // 2 + 1 < N_PAIRS):
                issue_scales(k + 2)
            if const_expr(k + PIPE < K_ITERS):
                load_step(k + PIPE, (k + PIPE) % stages)

            rocdl.s_setprio(1)
            for i in range_constexpr(n_slab_m):
                for j in range_constexpr(n_slab_n):
                    c_frag[i][j] = mfma.call(
                        a_frags[i], b_frags[j], c_frag[i][j], sa[i], sb[j], k_pack=k % 2, set_prio=False
                    )
            rocdl.s_setprio(0)

        base_row = block_m * BLOCK_M + wave_m * A_GRP_ROWS
        base_col = block_n * BLOCK_N + wave_n * B_GRP_ROWS
        for i in range_constexpr(n_slab_m):
            for j in range_constexpr(n_slab_n):
                store_c.store(c_frag[i][j], base_row + i * LDS_BLOCK_M, base_col + j * LDS_BLOCK_N)

    def launch(A, B_T, C, A_scale, B_scale, c_m, c_n, stream):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_gemm_tile(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs={
                "rocdl.waves_per_eu": max(1, n_waves // 4),
                "rocdl.flat_work_group_size": f"{n_threads},{n_threads}",
            },
        ).launch(grid=(grid_x, 1, 1), block=(n_threads, 1, 1), stream=stream)

    return kernel_gemm_tile, launch
