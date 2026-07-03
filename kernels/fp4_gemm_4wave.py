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
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr import buffer_ops as _buffer_ops
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec
from kernels.fp8_gemm_utils import (
    ceildiv,
    compute_global_swizzle,
    divmod,
    swizzle_128,
    wait_barrier,
)

_N_WAVES__4 = 4  # block is always 256 threads -> 4 waves (compile-time constant)


class _Buf:
    """LDS sub-buffer handle over the ONE contiguous LDS array.

    Holds the shared base pointer plus this buffer's compile-time byte offset
    SEPARATELY (not pre-added). Keeping the buffer offset as an int lets the ds_read
    path build the address as ``(base + dynamic_swizzle) + const(buffer_off+tile)``
    so the constant outer GEP folds into the ds_read 16-bit offset: field. (Pre-adding
    buffer_off into .ptr made the dynamic swizzle the outermost term -> ptrtoint +
    int-add -> unfoldable, leaving ~24 address VGPRs materialized.)

    ``.ptr`` (base + buffer_off) is still provided for G2SLoaderAsm, which needs the
    actual per-buffer LDS base for m0.
    """

    def __init__(self, base_ptr, byte_off):
        self.base_ptr = base_ptr
        self.byte_off = byte_off

    @property
    def ptr(self):
        return fx.add_offset(self.base_ptr, self.byte_off)


_gep = _buffer_ops.get_element_ptr


def _lds_ptr_t():
    return _ir.Type.parse("!llvm.ptr<3>")


def _asm_void(operands, asm_string, constraints):
    """Side-effecting void inline asm (LLVM sees no memory op -> no waitcnt added)."""
    _llvm.inline_asm(None, operands, asm_string, constraints, has_side_effects=True)


def _uniform_i32(value):
    """Cast to i32 and force a wave-uniform SGPR value for scalar inline-asm operands."""
    raw = arith._to_raw(value) if not isinstance(value, _ir.Value) else value
    if raw.type != _T.i32:
        raw = arith._to_raw(fx.Int32(raw))
    return _rocdl.readfirstlane(_T.i32, raw)


class G2SLoaderAsm:
    """Global->LDS DMA via INLINE-ASM ``buffer_load_dwordx4 ... lds`` instead of the
    BufferCopyLDS128b copy atom.

    Mirrors the mla_fwd_decode trick: with the 8 LDS buffers merged into ONE
    symbol (so ds_read cross-buffer offsets fold into the 16-bit imm), LLVM can no
    longer prove the g2s LDS writes don't alias the ds_reads, and would insert a
    spurious ``s_waitcnt vmcnt(0)`` before every ds_read. Emitting the load as
    opaque inline asm means LLVM sees no LDS write at all, so it adds no drain --
    vmcnt ordering is managed entirely by our explicit ``wait_barrier`` (which is
    itself inline-asm ``s_waitcnt vmcnt(N); s_barrier``). The inline-asm
    buffer_load IS still counted toward vmcnt by hardware, so the manual counts in
    wait_barrier stay correct.

    Each ``buffer_load_dwordx4 vN, rsrc, soffset offen lds`` writes 16 bytes to the
    LDS address in m0; m0 holds the per-step LDS byte base (wave-uniform via
    readfirstlane), the per-lane swizzle is the voffset VGPR (loop-invariant), and
    the K-step offset is the scalar soffset operand (hardware-free add).
    """

    def __init__(self, rsrc, gl_offsets, n_load_steps__4, wave_id):
        self.rsrc = arith._to_raw(rsrc)
        self.gl_offsets = gl_offsets
        self.n_load_steps__4 = n_load_steps__4
        self.wave_id = wave_id
        self.n_waves__0_4 = fx.block_dim.x // 64

    @property
    def _step_stride(self):
        # m0 (LDS byte) advance between consecutive steps of one load. Must be a
        # Python int (baked into the asm string), so use the compile-time wave count
        # (block is always 256 -> 4 waves) rather than the runtime block_dim value.
        return _N_WAVES__4 * 1024

    def _lds_base_sgpr(self, lds_dst):
        # Step-0 LDS byte base (wave-uniform). Later steps add _step_stride to m0.
        lds_base = fx.Int32(fx.ptrtoint(lds_dst.ptr)) + fx.Int32(self.wave_id * 1024)
        return _uniform_i32(lds_base)

    def _voffset(self, step):
        # Per-lane global byte offset = swizzle only (loop-invariant). The K-step
        # offset is NOT added here -- it goes in the buffer instruction's scalar
        # soffset field (hardware-free add), so voffset is a constant VGPR reused
        # across all K iterations instead of an `v_add k_offset` every step.
        return arith._to_raw(fx.Int32(self.gl_offsets[step]))

    def _emit(self, lds_dst, k_offset, step):
        # m0 idiom (gcnasm async_copy): set m0 once for step 0, then advance it with
        # s_add for later steps instead of recomputing readfirstlane+s_mov per step.
        # The N_TILES steps of one load are issued back-to-back (interleaved into one
        # MFMA cluster, in order), so the m0 add-chain stays coherent; s_add m0 is
        # volatile inline-asm so the compiler can't reorder across it. Cuts the
        # per-step v_readfirstlane (32->8/main-loop) and turns s_mov into s_add,
        # freeing scalar-issue slots so the MFMAs pack tighter (toward cyc/mfma 16).
        voff = self._voffset(step)
        soff = _uniform_i32(k_offset)  # scalar soffset (K-step), folded by hardware
        stride = self._step_stride
        if step == 0:
            m0 = self._lds_base_sgpr(lds_dst)
            asm = "s_mov_b32 m0, $0\nbuffer_load_dwordx4 $1, $2, $3 offen lds"
            _asm_void([m0, voff, self.rsrc, soff], asm, "s,v,s,s")
        else:
            asm = f"s_add_u32 m0, {stride}, m0\nbuffer_load_dwordx4 $0, $1, $2 offen lds"
            _asm_void([voff, self.rsrc, soff], asm, "v,s,s")

    def load(self, lds_dst, k_offset):
        for step in range_constexpr(self.n_load_steps__4):
            self._emit(lds_dst, k_offset, step)

    def load_one(self, lds_dst, k_offset, step):
        self._emit(lds_dst, k_offset, step)


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

    def _vec_load_16xf8(self, lds_src, dyn_offset, const_offset):
        # Plain LLVM ds_read (NOT inline-asm). The single-symbol LDS layout would
        # normally make the compiler insert an s_waitcnt vmcnt(0) drain before each
        # ds_read (it can't prove the g2s global->LDS DMA writes don't alias) -- but
        # because g2s is now inline-asm (G2SLoaderAsm), the LDS WRITE is invisible to
        # the compiler, so no drain is emitted. And being a REAL LDS load, the
        # compiler tracks it with fine-grained lgkmcnt (matching the 8-symbol
        # baseline's sync) instead of the conservative per-MFMA VMEM vmcnt it would
        # insert for an opaque inline-asm block.
        #
        # Address folding: the per-lane vaddr VGPR carries only the DYNAMIC part
        # (symbol base + 64KB-window half + lane swizzle); the constant (buffer +
        # tile) byte offset folds into the ds_read 16-bit offset: immediate via the
        # inttoptr + GEP(static) below. ds offset is 16-bit (max 0xFFFF=64KB) but the
        # 8 buffers span 0x1d800 (>64KB), so split into two windows: buffers 0-3
        # (base+0) and 4-7 (base+0x10000). imm = buffer_off - window_base + tile*2048
        # stays <= 0xc000+0x1800 = 0xd800 < 0xFFFF. The window base (0 or 0x10000) is
        # the only buffer-dependent term left in the dynamic vaddr, so all 8 buffers
        # collapse to 2 base VGPRs per (operand, step) -- 4 total in the hot loop.
        total_off = lds_src.byte_off + const_offset
        window_base = (total_off // 0x10000) * 0x10000
        imm = total_off - window_base
        assert 0 <= imm <= 0xFFFF
        vaddr = fx.Int32(fx.ptrtoint(lds_src.base_ptr)) + fx.Int32(window_base + dyn_offset)
        lds_ptr = _llvm.inttoptr(_lds_ptr_t(), arith._to_raw(vaddr))
        if imm != 0:
            lds_ptr = _gep(lds_ptr, static_byte_offset=imm)
        vec4_i32 = _ir.VectorType.get([4], fx.Int32.ir_type)
        raw = _llvm.LoadOp(vec4_i32, lds_ptr, alignment=16).result
        return Vec(raw)

    def _dyn_offset(self, step, preshuffled):
        # Lane-dependent part (tile i=0); the per-tile i*2048 is added as a constant.
        # Verified: _offset(i,step) - _offset(0,step) == i*2048 on both paths.
        row = self.wave_idx * (self.n_tiles * 16) + self.lane_id % 16
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
                dyn = self._dyn_offset(step, preshuffled)
                v = self._vec_load_16xf8(lds_src, dyn, i * 2048)
                halves.append(v.bitcast(fx.Int32))  # i32x4, the K=128 MFMA operand
            frag.append(halves)  # [ksub0_i32x4, ksub1_i32x4]
        return frag

    def load_one(self, lds_src, i, ksub, preshuffled=False):
        """One i32x4 (tile i, K=128 sub-block ksub) -- the interleave granularity."""
        dyn = self._dyn_offset(ksub, preshuffled)
        v = self._vec_load_16xf8(lds_src, dyn, i * 2048)
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

    def call(self, a, b, c, sa, sb, interleave=None, interleave_stride=1):
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
        mth = [0]  # MFMA counter, for spacing thunks every `interleave_stride` MFMAs
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
                    # Issue a thunk only every `interleave_stride` MFMAs so a load gets
                    # >1 MFMA execute-shadow to hide behind (vs bunched after the first
                    # few MFMAs). Spreads buffer_load/ds_read across the quad.
                    if nth[0] < len(thunks) and (mth[0] % interleave_stride) == 0:
                        thunks[nth[0]]()
                        nth[0] += 1
                    mth[0] += 1
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


# Scale-LDS geometry. Each wave needs 4 scale blocks (256 B each) per operand:
# blocks {R0g0, R0g1, R1g0, R1g1} for A, {C0g0, C0g1, C1g0, C1g1} for B. One
# ``buffer_load_dwordx4 ... lds`` (64 lanes x 16 B = 1024 B/wave) gathers all 4
# blocks of one operand. A region (1024 B) + B region (1024 B) = 2048 B/wave;
# x4 waves = 8192 B per pipeline slot. Slots are indexed kstep%_SCALE_SLOTS.
# With depth-2 prefetch AND an unroll-by-2 main loop, up to 4 K-steps are live at
# once (read[kk], read[kk+1], gather[kk+2], gather[kk+3]), so 4 slots are needed
# to avoid one being overwritten before its read (3 slots -> nan). 4*8192 = 32 KB.
_SCALE_WAVE_BYTES__2048 = 2048
_SCALE_SLOT_BYTES__8192 = _N_WAVES__4 * _SCALE_WAVE_BYTES__2048  # 8192
_SCALE_SLOTS__4 = 4
_SCALE_LDS_BYTES__32768 = _SCALE_SLOTS__4 * _SCALE_SLOT_BYTES__8192  # 32 KB
_SCALE_A_REGION = 0
_SCALE_B_REGION = 1024


class ScaleLoaderLDS:
    """Loads ``shuffle_scale_w4``-PRESHUFFLED per-1x32 E8M0 scales via a single
    ``buffer_load_dwordx4 ... lds`` per operand per K-step, into a scale LDS
    region, then per-lane ``ds_read_b32`` -- replacing the 8 per-step
    ``buffer_load_dword`` (each with a dur-6 voffset v_add, the #1 exposed
    hot-loop cost).

    Layout (gate_up=False): per (N1 group, K-step) the e8m0 form a 64-i32
    (256 B) block ``[K_Lane(4), N_Lane(16)]`` in which lane L's MFMA scale is
    element L. A wave's 4 blocks are groups ``{G, G+1, G+4, G+5}`` (G+4 == the
    second M/N half's group, since LDS_BLOCK/32 == 4). The dwordx4 gather has
    lane g fetch 4 contiguous i32 ``(g%16)*4..+3`` of block ``g//16`` and write
    them to ``m0 + g*16``; that lands i32 j of block blk at LDS byte
    ``blk*256 + j*4`` (natural order), so the read is ``region + blk*256 + L*4``.
    """

    def __init__(self, scale_arg, n_tiles, K, lane_id, wave_id, lds_base_ptr, region_off):
        assert n_tiles % _FP4_PACK == 0
        self.n_groups = n_tiles // _FP4_PACK  # pack-groups per M/N half (=2)
        self.K1 = K // 256
        self.row_i32 = self.K1 * 64  # i32 per N1 group
        self.lane_id = lane_id
        self.wave_id = wave_id
        self.region_off = region_off
        self.rsrc = arith._to_raw(_buffer_ops.create_buffer_resource(scale_arg, max_size=True))
        # Gather per-lane block / within-block index (loop-invariant).
        self._blk = lane_id // 16  # 0..3 -> which of the 4 blocks
        self._in16 = lane_id % 16  # 0..15 -> which 4-i32 chunk within the block
        self._lds_base = fx.Int32(fx.ptrtoint(lds_base_ptr))

    def _slot_wave_byte(self, slot):
        return self._lds_base + fx.Int32(slot) * fx.Int32(_SCALE_SLOT_BYTES__8192) + fx.Int32(
            self.wave_id * _SCALE_WAVE_BYTES__2048 + self.region_off
        )

    def set_wave_base(self):
        """Precompute the wave-uniform LDS base (lds_base + wave_id*2048 + region)
        into an SGPR ONCE via a single readfirstlane. gather() then computes m0 =
        wave_base_s + slot*8192 with scalar arithmetic, so the "s" constraint on m0
        needs NO per-gather readfirstlane (was 4/iter: one per A/B gather per step)."""
        wave_base = self._lds_base + fx.Int32(self.wave_id * _SCALE_WAVE_BYTES__2048 + self.region_off)
        self._wave_base_s = _rocdl.readfirstlane(_T.i32, arith._to_raw(wave_base))
        # soffset=0 as a wave-uniform SGPR (readfirstlane'd once, reused every gather).
        self._soff0 = _uniform_i32(fx.Int32(0))

    def gather(self, kstep, slot, base_tile):
        """Issue ONE buffer_load_dwordx4...lds gathering all 4 blocks of this
        operand into scale-LDS ``slot``. ``base_tile`` = the M/N-half-0 base row
        (half 1 is reached via group +4). Inline-asm so LLVM emits no LDS-write
        drain; vmcnt accounting is owned by wait_barrier (hardware counts it)."""
        # LDS write (layout A): lane L writes its 16 B to m0 + L*16. So lane L must
        # READ from global the data destined for LDS slot L*16 = block (L//16),
        # i32 chunk (L%16)*4..+3. block g//16 -> group G + (blk//2)*4 + (blk%2).
        G = fx.Int32(base_tile // 32)
        grp = G + (self._blk // 2) * fx.Int32(4) + (self._blk % 2)
        i32_off = grp * fx.Int32(self.row_i32) + fx.Int32(kstep) * fx.Int32(64) + self._in16 * fx.Int32(4)
        voff = arith._to_raw(i32_off * fx.Int32(4))  # bytes
        # m0 = precomputed wave base (SGPR) + slot*8192 (scalar): no readfirstlane.
        m0 = arith._to_raw(fx.Int32(self._wave_base_s) + fx.Int32(slot) * fx.Int32(_SCALE_SLOT_BYTES__8192))
        asm = "s_mov_b32 m0, $0\nbuffer_load_dwordx4 $1, $2, $3 offen lds"
        _asm_void([m0, voff, self.rsrc, self._soff0], asm, "s,v,s,s")

    def read_half(self, slot, half):
        """Per-lane ds_read of ONE half (2 blocks) -> list[n_groups] of i32.
        Split from read() so half-1 can be issued as a thunk in an MFMA shadow
        (half-0 feeds c00/c01, half-1 feeds c10/c11 -- two MFMA clusters later)."""
        # lane-contiguous write: lane g's 4 i32 at m0 + g*16. block-elem e of block
        # blk was written by lane (blk*16 + e//4) as its (e%4)-th i32 -> LDS byte
        # blk*256 + (e//4)*16 + (e%4)*4. MFMA lane L wants block-elem L.
        L = self.lane_id
        base = self._slot_wave_byte(slot) + fx.Int32((L // 4) * 16 + (L % 4) * 4)
        grp_list = []
        for gi in range_constexpr(self.n_groups):
            blk = half * 2 + gi
            vaddr = base + fx.Int32(blk * 256)
            lds_ptr = _llvm.inttoptr(_lds_ptr_t(), arith._to_raw(vaddr))
            raw = _llvm.LoadOp(fx.Int32.ir_type, lds_ptr, alignment=4).result
            grp_list.append(fx.Int32(raw))
        return grp_list

    def read(self, slot):
        """Per-lane ds_read of the 4 blocks -> (half0, half1), each list[n_groups]
        of i32 (the same shape the MFMA consumes: sa[i//2] / sb[j//2])."""
        return self.read_half(slot, 0), self.read_half(slot, 1)


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
    BLOCK_M__256: int = 256,
    BLOCK_N__256: int = 256,
    use_xcd_remap: bool = True,
    mn_aligned: bool = False,
):
    # mn_aligned: caller asserts M % BLOCK_M == 0 and N % BLOCK_N == 0, so every
    # epilogue store is in-bounds -> skip the per-store col-bounds select (saves
    # 256 v_cmp+v_cndmask/wave). Leave False for arbitrary M/N (correctness via the
    # explicit bounds select). Common alignment-fast-path optimization.
    # 256 fp4 per LDS K-step row = 128 bytes; reuse fp8's 128-byte LDS layout.
    BLOCK_K__256 = 256  # fp4 elements
    BLOCK_K_BYTES__128 = BLOCK_K__256 // 2  # 128 bytes / row
    LDS_BLOCK_M__128 = BLOCK_M__256 // 2
    LDS_BLOCK_N__128 = BLOCK_N__256 // 2

    assert BLOCK_M__256 % 64 == 0 and BLOCK_N__256 % 64 == 0
    assert K % BLOCK_K__256 == 0

    K_ITERS = K // BLOCK_K__256
    N_TILES_A__4 = BLOCK_M__256 // 4 // 16
    N_TILES_B__4 = BLOCK_N__256 // 4 // 16
    N_ACCUMS__16 = N_TILES_A__4 * N_TILES_B__4
    assert N_ACCUMS__16 > 0

    N_LDS_ROUNDS__4 = max(N_TILES_A__4, N_TILES_B__4)

    a_lds_size__16384 = LDS_BLOCK_M__128 * BLOCK_K_BYTES__128
    b_lds_size__16384 = LDS_BLOCK_N__128 * BLOCK_K_BYTES__128

    # One contiguous LDS array (single static `make_ptr` -> single `@__shared_alloc`
    # symbol), NOT 8 independent leaf fields. With one base symbol every cross-buffer
    # offset is a compile-time constant against the same pointer, so the compiler
    # folds it into the 16-bit ds_read offset: field instead of materializing ~24
    # per-(buffer,tile) address VGPRs (252 -> ~228 arch VGPR). The catch: a single
    # LDS symbol breaks the compiler's alias analysis between the g2s global->LDS DMA
    # writes and the ds_reads, so it would insert a spurious `s_waitcnt vmcnt(0)`
    # before every ds_read -- which is why g2s uses inline-asm buffer_load (see
    # G2SLoaderAsm) so LLVM sees no LDS write and emits no drain.
    assert a_lds_size__16384 == b_lds_size__16384
    _lds_buf__16384 = a_lds_size__16384  # 16KB; 8 buffers laid out contiguously # 128KB

    @fx.struct
    class SharedStorage:
        all_lds: fx.Array[fx.Int8, 8 * _lds_buf__16384, 16]
        scale_lds: fx.Array[fx.Int8, _SCALE_LDS_BYTES__32768, 16]

    @flyc.kernel
    def kernel_gemm(
        A: fx.Tensor, B_T: fx.Tensor, C: fx.Tensor, A_scale: fx.Tensor, B_scale: fx.Tensor, c_m: fx.Int32, c_n: fx.Int32
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # 8 buffers as compile-time offsets off ONE base pointer (single symbol).
        _base_ptr = lds.all_lds.ptr

        def _buf(idx):
            return _Buf(_base_ptr, idx * _lds_buf__16384)

        a_cur0 = _buf(0)
        a_cur1 = _buf(1)
        a_next0 = _buf(2)
        a_next1 = _buf(3)
        b_cur0 = _buf(4)
        b_cur1 = _buf(5)
        b_next0 = _buf(6)
        b_next1 = _buf(7)

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64

        n_blocks = ceildiv(c_n, BLOCK_N__256)
        if const_expr(use_xcd_remap):
            tile_i, tile_j = _xcd_swizzle(ceildiv(c_m, BLOCK_M__256), n_blocks)
        else:
            tile_i, tile_j = divmod(fx.block_idx.x, n_blocks)

        wave_i__0_1 = wave_id // 2
        wave_j__0_1 = wave_id % 2

        # Global byte offsets (fp4 packed: K bytes = K // 2).
        K_BYTES = K // 2
        A0_gl_offset = (tile_i * BLOCK_M__256) * K_BYTES
        A1_gl_offset = (tile_i * BLOCK_M__256 + LDS_BLOCK_M__128) * K_BYTES
        A_K_STEP = BLOCK_K_BYTES__128
        B0_gl_offset = (tile_j * BLOCK_N__256) * K_BYTES
        B1_gl_offset = (tile_j * BLOCK_N__256 + LDS_BLOCK_N__128) * K_BYTES
        # B is preshuffled (16,16): one N-16 row-block spans 2*1024 bytes per
        # K-step (same constant fp8_gemm_4wave uses for b_preshuffled).
        B_K_STEP = 2 * 1024

        mfma = Mfma16x16x128Fp4(N_TILES_A__4, N_TILES_B__4)

        # Scale via dwordx4...lds gather + ds_read (see ScaleLoaderLDS). One gather
        # per operand per K-step into a triple-buffered scale-LDS slot; read[kc] /
        # gather[kc+2] use slots kc%3 / (kc+2)%3 (3 distinct -> race-free).
        _scale_base_ptr = lds.scale_lds.ptr
        a_scale_ld = ScaleLoaderLDS(A_scale, N_TILES_A__4, K, lane_id, wave_id, _scale_base_ptr, _SCALE_A_REGION)
        b_scale_ld = ScaleLoaderLDS(B_scale, N_TILES_B__4, K, lane_id, wave_id, _scale_base_ptr, _SCALE_B_REGION)
        # Precompute each loader's wave-uniform LDS base into an SGPR once, so the
        # per-gather m0 needs no readfirstlane (was 4/iter -> exposed ~19% in ATT).
        a_scale_ld.set_wave_base()
        b_scale_ld.set_wave_base()

        base_row = tile_i * BLOCK_M__256 + wave_i__0_1 * (N_TILES_A__4 * 16)
        base_col = tile_j * BLOCK_N__256 + wave_j__0_1 * (N_TILES_B__4 * 16)
        sa_R0 = base_row
        sa_R1 = base_row + LDS_BLOCK_M__128
        sb_C0 = base_col
        sb_C1 = base_col + LDS_BLOCK_N__128

        def _slot(k):
            return fx.Int32(k) % fx.Int32(_SCALE_SLOTS__4)

        def _gather_scales(k, slot):
            """Issue the 2 dwordx4...lds gathers (A, B) for K-step ``k`` into LDS
            ``slot``. base_row/base_col are the M/N half-0 bases; half-1 is reached
            via group +4 inside gather()."""
            a_scale_ld.gather(k, slot, base_row)
            b_scale_ld.gather(k, slot, base_col)

        def _gather_scale_thunks(k, slot):
            """gather (A,B) as thunks so they co-issue in the MFMA execute shadow
            instead of two back-to-back buffer_load after all MFMAs (ATT showed the
            end-of-step gather pair exposed, not hidden)."""
            return [
                lambda: a_scale_ld.gather(k, slot, base_row),
                lambda: b_scale_ld.gather(k, slot, base_col),
            ]

        # Accumulators: 2x2 64x64 quadrants per wave.
        c00_frag = [mfma.zero_value] * N_ACCUMS__16
        c01_frag = [mfma.zero_value] * N_ACCUMS__16
        c10_frag = [mfma.zero_value] * N_ACCUMS__16
        c11_frag = [mfma.zero_value] * N_ACCUMS__16

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K_BYTES, N_LDS_ROUNDS__4, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K_BYTES, N_LDS_ROUNDS__4, preshuffled=True)

        # Inline-asm g2s (see G2SLoaderAsm): needs the raw buffer resource. Build it
        # once from the i8 buffer tensor (max_size OOB check; all addresses in-bounds).
        a_rsrc = _buffer_ops.create_buffer_resource(A, max_size=True) # why max size here...
        b_rsrc = _buffer_ops.create_buffer_resource(B_T, max_size=True) # why???
        a_g2s = G2SLoaderAsm(a_rsrc, gl_off_a, N_TILES_A__4, wave_id)
        b_g2s = G2SLoaderAsm(b_rsrc, gl_off_b, N_TILES_B__4, wave_id)
        a_s2r = S2RLoaderFp4(wave_i__0_1, N_TILES_A__4)
        b_s2r = S2RLoaderFp4(wave_j__0_1, N_TILES_B__4)
        store_c = StoreCFp4(C, c_m, c_n, mfma.idx, N_TILES_A__4, N_TILES_B__4, mn_aligned=mn_aligned)

        # Prologue. Scale gathers for step 0/1/2 go FIRST (before the 32 g2s) so they
        # are the OLDEST outstanding VMEM -- the main loop's wait_barrier(17) then
        # drains them naturally (vs issuing them last, where vmcnt(17) can't reach
        # them past the 32 newer g2s -> step-0 read got un-landed LDS -> big-K nan).
        # DEPTH-3: scale[0] feeds the initial VGPR carry, scale[1]/scale[2] are read
        # (into carry) during step 0/1; step kc then gathers scale[kc+3].
        _gather_scales(0, _slot(0))
        _gather_scales(1, _slot(1))
        _gather_scales(2, _slot(2))

        a_g2s.load(a_cur0, A0_gl_offset + 0 * A_K_STEP) # 4个load.
        b_g2s.load(b_cur0, B0_gl_offset + 0 * B_K_STEP) # 4个...
        b_g2s.load(b_cur1, B1_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * A_K_STEP)

        a_g2s.load(a_next0, A0_gl_offset + 1 * A_K_STEP)
        b_g2s.load(b_next0, B0_gl_offset + 1 * B_K_STEP)
        b_g2s.load(b_next1, B1_gl_offset + 1 * B_K_STEP)
        a_g2s.load(a_next1, A1_gl_offset + 1 * A_K_STEP)

        def _do_quad(a_frag, b_frag, c_frag, sa_ksub, sb_ksub):
            return mfma.call(a_frag, b_frag, c_frag, sa_ksub, sb_ksub)

        wait_barrier((3 * N_TILES_A__4) + (4 * N_TILES_B__4))
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier((3 * N_TILES_A__4) + (3 * N_TILES_B__4))
        b0_frag = b_s2r.load(b_cur0, preshuffled=True)

        # Initial VGPR scale carry = scale[0] (gathered first in prologue, landed by
        # the barriers above). The loop carries scale[kc] in VGPR; each step reads
        # scale[kc+1] in the MFMA shadow. saR0/saR1/sbC0/sbC1 each list[n_groups] i32.
        sc0_saR0, sc0_saR1 = a_scale_ld.read(_slot(0))
        sc0_sbC0, sc0_sbC1 = b_scale_ld.read(_slot(0))
        sc0 = (sc0_saR0, sc0_saR1, sc0_sbC0, sc0_sbC1)


        # Main-loop wait_barrier vmcnt. g2s and scale gathers are inline-asm so the
        # compiler uses our literal vmcnt verbatim. With depth-2 end-of-step scale,
        # the scale[kc] gather (issued 2 steps ago) is the oldest outstanding VMEM at
        # the step top; vmcnt(17) drains exactly it while keeping the g2s window (16)
        # in flight, so the ds_read of scale[kc] sees landed LDS.
        _MAIN_VMCNT = 17

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
        def _read_scale_thunks(kc_idx, holder):
            """4 thunks, each doing one half-read of scale[kc_idx] into holder
            (holder = [saR0, saR1, sbC0, sbC1]). Co-issued in an MFMA shadow so the
            read of NEXT step's scale is fully hidden (no exposed lgkmcnt at loop top;
            scale lives in VGPR carry)."""
            s = _slot(kc_idx)

            def _r(dst, ld, half, _s=s):
                holder[dst] = ld.read_half(_s, half)

            return [
                lambda: _r(0, a_scale_ld, 0),
                lambda: _r(1, a_scale_ld, 1),
                lambda: _r(2, b_scale_ld, 0),
                lambda: _r(3, b_scale_ld, 1),
            ]

        def _one_step(kc, a0f, b0f, sc, accs, bufs):
            # bufs = (a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1)
            ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
            # DEPTH-3 scale + VGPR carry: ``sc`` = scale[kc] ALREADY in VGPR (read in
            # the prior step's MFMA shadow), so there is NO scale ds_read / lgkmcnt
            # wait at the top of this step. This step (a) reads scale[kc+1] into the
            # next carry inside the MFMA shadow, and (b) gathers scale[kc+3] (depth-3,
            # so scale[kc+1] -- gathered at step kc-2 -- is landed at the barrier here).
            saR0, saR1, sbC0, sbC1 = sc
            c00f, c01f, c10f, c11f = accs
            kc_i = fx.Int32(kc)

            _b1 = [None] * N_TILES_B__4
            _a1 = [None] * N_TILES_A__4
            _a0n = [None] * N_TILES_A__4
            _b0n = [None] * N_TILES_B__4
            # This step prefetches K-step (kc+2) for g2s. a*_off = base + (kc+2)*STEP.
            ak = (kc_i + fx.Int32(2)) * fx.Int32(A_K_STEP)
            bk = (kc_i + fx.Int32(2)) * fx.Int32(B_K_STEP)
            a0_off = fx.Int32(A0_gl_offset) + ak
            a1_off = fx.Int32(A1_gl_offset) + ak
            b0_off = fx.Int32(B0_gl_offset) + bk
            b1_off = fx.Int32(B1_gl_offset) + bk

            # Next-step scale carry: read scale[kc+1] in the MFMA shadow (4 thunks).
            _scn = [None, None, None, None]  # saR0, saR1, sbC0, sbC1 for kc+1
            _rd_scn = _read_scale_thunks(kc_i + 1, _scn)
            # Scale[kc+3] gather (depth-3), co-issued in the MFMA shadow (2 thunks).
            # Clamp to K_ITERS-1: the last loop step would gather scale[K_ITERS] (OOB
            # -> memory fault); the clamped extra gather re-reads scale[K_ITERS-1]
            # into the same slot (idempotent, its result is never consumed).
            _gk = _min(kc_i + fx.Int32(3), fx.Int32(K_ITERS - 1))
            _sc_gather = _gather_scale_thunks(_gk, _slot(_gk))

            wait_barrier(_MAIN_VMCNT)
            il = (
                _g2s_thunks(a_g2s, ac0, a0_off, N_TILES_A__4)
                + _s2r_thunks(b_s2r, bc1, _b1, N_TILES_B__4, True)
                + _rd_scn[:2]
            )
            c00f = mfma.call(a0f, b0f, c00f, saR0, sbC0, interleave=il, interleave_stride=2)
            b1f = _b1

            il = (
                _g2s_thunks(b_g2s, bc0, b0_off, N_TILES_A__4)
                + _s2r_thunks(a_s2r, ac1, _a1, N_TILES_A__4, False)
                + _rd_scn[2:]
            )
            c01f = mfma.call(a0f, b1f, c01f, saR0, sbC1, interleave=il, interleave_stride=2)
            a1f = _a1

            wait_barrier(_MAIN_VMCNT)
            il = (
                _g2s_thunks(b_g2s, bc1, b1_off, N_TILES_A__4)
                + _s2r_thunks(a_s2r, an0, _a0n, N_TILES_A__4, False)
                + _sc_gather[:1]
            )
            c10f = mfma.call(a1f, b0f, c10f, saR1, sbC0, interleave=il, interleave_stride=2)
            a0nf = _a0n

            il = (
                _g2s_thunks(a_g2s, ac1, a1_off, N_TILES_A__4)
                + _s2r_thunks(b_s2r, bn0, _b0n, N_TILES_B__4, True)
                + _sc_gather[1:]
            )
            c11f = mfma.call(a1f, b1f, c11f, saR1, sbC1, interleave=il, interleave_stride=2)
            b0nf = _b0n

            sc_next = (_scn[0], _scn[1], _scn[2], _scn[3])
            new_bufs = (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)  # swap cur<->next
            return a0nf, b0nf, sc_next, (c00f, c01f, c10f, c11f), new_bufs

        bufs0 = (a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1)
        n_a = 2 * N_TILES_A__4
        n_b = 2 * N_TILES_B__4
        n_ga = N_TILES_A__4 // _FP4_PACK  # scale groups per A half (=len(saR0))
        n_gb = N_TILES_B__4 // _FP4_PACK
        n_sc = 2 * n_ga + 2 * n_gb  # sc = (saR0,saR1,sbC0,sbC1) flattened
        _R = arith._to_raw

        def _flat_sc(sc):
            saR0, saR1, sbC0, sbC1 = sc
            return [_R(v) for v in saR0] + [_R(v) for v in saR1] + [_R(v) for v in sbC0] + [_R(v) for v in sbC1]

        def _unflat_sc(flat):
            o = 0
            saR0 = list(flat[o : o + n_ga]); o += n_ga
            saR1 = list(flat[o : o + n_ga]); o += n_ga
            sbC0 = list(flat[o : o + n_gb]); o += n_gb
            sbC1 = list(flat[o : o + n_gb]); o += n_gb
            return (saR0, saR1, sbC0, sbC1)

        # Carry = a0/b0 fragments + VGPR scale carry (scale[kc]) + 4 accumulator groups.
        init_state = (
            _flat_frag(a0_frag)
            + _flat_frag(b0_frag)
            + _flat_sc(sc0)
            + [_R(x) for x in c00_frag]
            + [_R(x) for x in c01_frag]
            + [_R(x) for x in c10_frag]
            + [_R(x) for x in c11_frag]
        )
        for kk, state in range(0, K_ITERS - 2, 2, init=init_state):
            off = 0
            a0f = _unflat_frag(state[off : off + n_a], N_TILES_A__4)
            off += n_a
            b0f = _unflat_frag(state[off : off + n_b], N_TILES_B__4)
            off += n_b
            sc = _unflat_sc(state[off : off + n_sc])
            off += n_sc
            c00f = list(state[off : off + N_ACCUMS__16])
            off += N_ACCUMS__16
            c01f = list(state[off : off + N_ACCUMS__16])
            off += N_ACCUMS__16
            c10f = list(state[off : off + N_ACCUMS__16])
            off += N_ACCUMS__16
            c11f = list(state[off : off + N_ACCUMS__16])
            off += N_ACCUMS__16
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
        a0_frag = _unflat_frag(state[off : off + n_a], N_TILES_A__4)
        off += n_a
        b0_frag = _unflat_frag(state[off : off + n_b], N_TILES_B__4)
        off += n_b
        # VGPR carry at loop exit = scale[K_ITERS-2] (each iter advances sc by 2).
        sc = _unflat_sc(state[off : off + n_sc])
        off += n_sc
        c00_frag = list(state[off : off + N_ACCUMS__16])
        off += N_ACCUMS__16
        c01_frag = list(state[off : off + N_ACCUMS__16])
        off += N_ACCUMS__16
        c10_frag = list(state[off : off + N_ACCUMS__16])
        off += N_ACCUMS__16
        c11_frag = list(state[off : off + N_ACCUMS__16])
        off += N_ACCUMS__16

        # Tail step K_ITERS - 2: scale[K_ITERS-2] is the carried VGPR sc (no read).
        # Read scale[K_ITERS-1] into the next carry in the c00 MFMA shadow.
        saR0, saR1, sbC0, sbC1 = sc
        _scn = [None, None, None, None]
        _rd_scn = _read_scale_thunks(fx.Int32(K_ITERS - 1), _scn)
        _b1 = [None] * N_TILES_B__4
        _a1 = [None] * N_TILES_A__4
        wait_barrier((2 * N_TILES_A__4) + (2 * N_TILES_B__4))
        il = (
            _s2r_thunks(b_s2r, b_cur1, _b1, N_TILES_B__4, True)
            + _s2r_thunks(a_s2r, a_cur1, _a1, N_TILES_A__4, False)
            + _rd_scn
        )
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, saR0, sbC0, interleave=il, interleave_stride=2)
        b1_frag = _b1
        a1_frag = _a1
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, saR0, sbC1)
        _a0n = [None] * N_TILES_A__4
        _b0n = [None] * N_TILES_B__4
        wait_barrier((1 * N_TILES_A__4) + (1 * N_TILES_B__4))
        il = _s2r_thunks(a_s2r, a_next0, _a0n, N_TILES_A__4, False) + _s2r_thunks(b_s2r, b_next0, _b0n, N_TILES_B__4, True)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, saR1, sbC0, interleave=il, interleave_stride=2)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, saR1, sbC1)
        a0_frag = _a0n
        b0_frag = _b0n

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Tail step K_ITERS - 1: scale[K_ITERS-1] read into the carry above.
        _b1 = [None] * N_TILES_B__4
        _a1 = [None] * N_TILES_A__4
        wait_barrier(0)
        saR0, saR1, sbC0, sbC1 = (_scn[0], _scn[1], _scn[2], _scn[3])
        il = _s2r_thunks(b_s2r, b_cur1, _b1, N_TILES_B__4, True) + _s2r_thunks(a_s2r, a_cur1, _a1, N_TILES_A__4, False)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, saR0, sbC0, interleave=il, interleave_stride=2)
        b1_frag = _b1
        a1_frag = _a1
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, saR0, sbC1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, saR1, sbC0)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, saR1, sbC1)

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
        grid_x = ceildiv(c_m, BLOCK_M__256) * ceildiv(c_n, BLOCK_N__256)
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
