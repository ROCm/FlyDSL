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

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec
from kernels.common import buffer_ops as _buffer_ops
from kernels.gemm.fp8_gemm_utils import (
    ceildiv,
    divmod,
    swizzle_128,
)

_N_WAVES = 4


def _global_swizzle(lane_id, wave_id, K, n_rounds, preshuffled):
    """``fp8_gemm_utils.compute_global_swizzle`` with the wave count pinned."""
    offsets = []
    for round in range_constexpr(n_rounds):
        if const_expr(preshuffled):
            row = lane_id % 8 + wave_id * 8 + round * (_N_WAVES * 8)
            col = (lane_id // 8) * 16
            offsets.append(
                (row // 16) * (K * 16) + (row % 16) * 16 + (col // 64) * 1024 + ((col % 64) // 16) * 256 + (col % 16)
            )
        else:
            row = lane_id // 8 + wave_id * 8 + round * (_N_WAVES * 8)
            col = (lane_id % 8) * 16
            r, c = swizzle_128(row, col)
            offsets.append(r * K + c)
    return offsets


BLOCK_M = 256
BLOCK_N = 256
N_TILES_A = 256 // 4 // 16
N_TILES_B = 256 // 4 // 16


class _Buf:
    def __init__(self, base_ptr, byte_off):
        self.base_ptr = base_ptr
        self.byte_off = byte_off

    @property
    def ptr(self):
        return fx.add_offset(self.base_ptr, self.byte_off)


_gep = _buffer_ops.get_element_ptr


def _lds_ptr_t():
    return _ir.Type.parse("!llvm.ptr<3>")


def _asm_void(operands, asm_string, constraints, clobbers=""):
    """Side-effecting void inline asm (LLVM sees no memory op -> no waitcnt added)."""
    if clobbers:
        constraints = f"{constraints},{clobbers}"
    _llvm.inline_asm(None, operands, asm_string, constraints, has_side_effects=True)


def _cvt_pk_bf16(a, b):
    """same as rocdl.cvt_pk_bf16_f32, but no inline asm to give compiler more freedom"""
    v2f32 = _ir.VectorType.get([2], fx.Float32.ir_type)
    vec = Vec.from_elements([fx.Float32(a), fx.Float32(b)], fx.Float32)
    src = fx.as_ir_value(vec)
    if src.type != v2f32:
        src = fx.arith.bitcast(v2f32, src)
    v2bf16 = _ir.VectorType.get([2], fx.BFloat16.ir_type)
    # llvm.bitcast, not arith.bitcast: the latter requires operand and result to
    # have the same shape, and this one is <2xbf16> -> i32.
    return _llvm.BitcastOp(fx.Int32.ir_type, fx.arith.trunc_f(v2bf16, src)).result


# for FP4_DMA_INTRINSIC=1 path, don't let compiler reorder the m0 set
_M0_CLOBBER = "~{m0}"


def wait_barrier(count):
    """``s_waitcnt vmcnt(count) lgkmcnt(0)`` + ``s_barrier``"""
    _rocdl.s_waitcnt(vmcnt=count, lgkmcnt=0)
    _rocdl.s_barrier()


# use intrinsic not the inline asm
_USE_DMA_INTRINSIC = os.environ.get("FP4_DMA_INTRINSIC", "0") == "1"
_LDS_BUF_BYTES = 16384  # one of the 8 A/B tile buffers; asserted against a_lds_size
_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "fp4_gemm_4wave.lds">'
# 10 lds scope names
_LDS_SCOPE_NAMES = [f"buf.{i}" for i in range(8)] + ["Asc", "Bsc"]
_SC_ASC = 8
_SC_BSC = 9


def _lds_scopes():
    return [
        _ir.Attribute.parse(f'#llvm.alias_scope<id = "fp4_gemm_4wave.{n}", domain = {_LDS_DOMAIN}>')
        for n in _LDS_SCOPE_NAMES
    ]


def _tag_alias(op, scopes, slot):
    """ascale & bscale lds loads use this wave0/1 for ascale & wave2/3 for bscale"""
    slots = slot if isinstance(slot, tuple) else (slot,)
    op = getattr(op, "owner", op)
    op.attributes["alias_scopes"] = _ir.ArrayAttr.get([scopes[s] for s in slots])
    op.attributes["noalias_scopes"] = _ir.ArrayAttr.get([sc for i, sc in enumerate(scopes) if i not in slots])


def _uniform_i32(value):
    """Cast to i32 and force a wave-uniform SGPR value for scalar inline-asm operands."""
    raw = fx.as_ir_value(value) if not isinstance(value, _ir.Value) else value
    if raw.type != _T.i32:
        raw = fx.as_ir_value(fx.Int32(raw))
    return _rocdl.readfirstlane(_T.i32, raw)


class G2SLoaderAsm:
    def __init__(self, rsrc, gl_offsets, n_load_steps, wave_id, scopes=None, base_ptr=None):
        self.rsrc = fx.as_ir_value(rsrc)
        self.gl_offsets = gl_offsets
        self.n_load_steps = n_load_steps
        self.wave_id = wave_id
        self.scopes = scopes
        self.base_ptr = base_ptr

    @property
    def _step_stride(self):
        # m0 (LDS byte) advance per step.
        return _N_WAVES * 1024

    def set_wave_base(self, base_ptr):
        # The wave-uniform LDS base, readfirstlane'd into an SGPR ONCE.
        wb = fx.Int32(fx.ptrtoint(base_ptr)) + fx.Int32(self.wave_id * 1024)
        self._wave_base_s = _rocdl.readfirstlane(_T.i32, fx.as_ir_value(wb))

    def _lds_base_sgpr(self, lds_dst):
        m0 = fx.Int32(self._wave_base_s) + fx.Int32(lds_dst.byte_off)
        return fx.as_ir_value(m0)

    def _voffset(self, step):
        # Swizzle only: the K-step goes in the scalar soffset field, so this VGPR is
        # loop-invariant instead of needing a `v_add k_offset` every step.
        return fx.as_ir_value(fx.Int32(self.gl_offsets[step]))

    def _emit(self, lds_dst, k_offset, step):
        # m0 idiom (gcnasm async_copy): set m0 for step 0, then s_add for the rest.
        voff = self._voffset(step)
        soff = _uniform_i32(k_offset)  # scalar soffset (K-step)
        stride = self._step_stride
        if self.scopes is not None:
            slot = lds_dst.byte_off // _LDS_BUF_BYTES
            # Off the readfirstlane'd wave base, not ptrtoint(base)+wave_id*1024:
            # wave_id is a VGPR, which would cost a v_readfirstlane per load.
            addr = fx.Int64(fx.Int32(self._wave_base_s) + fx.Int32(lds_dst.byte_off + step * stride))
            lds_ptr = _llvm.inttoptr(_lds_ptr_t(), fx.as_ir_value(addr))
            dma = _rocdl.raw_ptr_buffer_load_lds(self.rsrc, lds_ptr, fx.Int32(16), voff, soff, fx.Int32(0), fx.Int32(0))
            _tag_alias(dma, self.scopes, slot)
            return
        if step == 0:
            m0 = self._lds_base_sgpr(lds_dst)
            asm = "s_mov_b32 m0, $0\nbuffer_load_dwordx4 $1, $2, $3 offen lds"
            _asm_void([m0, voff, self.rsrc, soff], asm, "s,v,s,s")
        else:
            asm = f"s_add_u32 m0, {stride}, m0\nbuffer_load_dwordx4 $0, $1, $2 offen lds"
            _asm_void([voff, self.rsrc, soff], asm, "v,s,s")

    def load(self, lds_dst, k_offset):
        for step in range_constexpr(self.n_load_steps):
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

    def __init__(self, wave_idx, n_tiles, scopes=None):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles
        # Tag each ds_read with the buffer it reads (see _lds_scopes). Only needed
        # when g2s emits the real DMA intrinsic; with inline-asm g2s the write is
        # invisible and there is nothing to disambiguate against.
        self.scopes = scopes

    def _vec_load_16xf8(self, lds_src, dyn_offset, const_offset):
        total_off = lds_src.byte_off + const_offset
        # cute to two windows cause imm is 16-bit
        window_base = (total_off // 0x10000) * 0x10000
        imm = total_off - window_base
        assert 0 <= imm <= 0xFFFF
        vaddr = fx.Int32(fx.ptrtoint(lds_src.base_ptr)) + fx.Int32(window_base + dyn_offset)
        lds_ptr = _llvm.inttoptr(_lds_ptr_t(), fx.as_ir_value(vaddr))
        if imm != 0:
            lds_ptr = _gep(lds_ptr, static_byte_offset=imm)
        vec4_i32 = _ir.VectorType.get([4], fx.Int32.ir_type)
        load = _llvm.LoadOp(vec4_i32, lds_ptr, alignment=16)
        if self.scopes is not None:
            _tag_alias(load, self.scopes, lds_src.byte_off // _LDS_BUF_BYTES)
        raw = load.result
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
    so unwrap Vec/ArithValue via as_ir_value."""
    out = []
    for t in frag:
        out.append(fx.as_ir_value(t[0]))
        out.append(fx.as_ir_value(t[1]))
    return out


def _unflat_frag(flat, n_tiles):
    return [[flat[2 * i], flat[2 * i + 1]] for i in range(n_tiles)]


def _g2s_thunks(g2s, dst, gl_off, n_steps):
    """Module-level (so the @kernel AST rewriter doesn't turn the `range` into
    scf.for): list of thunks, each issuing one g2s.load_one step."""
    return [lambda s=s: g2s.load_one(dst, gl_off, s) for s in range(n_steps)]


def _riffle(glb, lds):
    """Interleave the global and LDS thunk lists proportionally like aiter's asm"""
    if not glb or not lds:
        return list(glb) + list(lds)
    out = []
    step = len(lds) / len(glb)
    li = 0
    for gi, t in enumerate(glb):
        out.append(t)
        upto = int(round((gi + 1) * step))
        out += lds[li:upto]
        li = upto
    return out + lds[li:]


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
    return fx.arith.select(a < b, a, b)


def _divmod_nonneg(a, b):
    """``divmod(a, b)`` where ``a >= 0`` is known and ``b`` may be a constant."""
    if const_expr(isinstance(b, int) and b > 0 and (b & (b - 1)) == 0):
        sh = b.bit_length() - 1
        return (a >> sh, a & (b - 1)) if const_expr(sh > 0) else (a, 0)
    return divmod(a, b)


def _xcd_swizzle(num_pid_m, num_pid_n):
    NUM_XCDS = 8
    WGM = 4
    NUM_CUS = 32 * NUM_XCDS
    SWIZZLE_THRESHOLD = 4 * NUM_CUS

    wgid = fx.block_idx.x
    num_wg = num_pid_m * num_pid_n
    simple_m, simple_n = _divmod_nonneg(wgid, num_pid_n)

    intra_xcd, xcd = _divmod_nonneg(wgid, NUM_XCDS)
    wgid_remap = xcd * (num_wg // NUM_XCDS) + intra_xcd
    num_wgid_in_group = WGM * num_pid_n
    group_id, intra_group = _divmod_nonneg(wgid_remap, num_wgid_in_group)
    first_pid_m = group_id * WGM
    if const_expr(isinstance(num_pid_m, int) and num_pid_m % WGM == 0):
        group_size_m = WGM
    else:
        group_size_m = _min(num_pid_m - first_pid_m, WGM)
    pid_n, intra_group_m = _divmod_nonneg(intra_group, group_size_m)
    pid_m = first_pid_m + intra_group_m

    use_simple = (num_wg < SWIZZLE_THRESHOLD) | (num_wg % NUM_XCDS != 0)
    if const_expr(isinstance(use_simple, bool)):
        return (simple_m, simple_n) if use_simple else (pid_m, pid_n)
    return (fx.arith.select(use_simple, simple_m, pid_m), fx.arith.select(use_simple, simple_n, pid_n))


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

    def __init__(self):
        assert N_TILES_A % _FP4_PACK == 0 and N_TILES_B % _FP4_PACK == 0
        self.accum_type = Vec.make_type(4, fx.Float32)
        self.res_ty = Vec.make_type(4, fx.Float32)

    def idx(self, i, j):
        return i * N_TILES_B + j

    def _order(self):
        """xdl buffer emission order learn from aiter's asm"""
        order = []
        j0s = list(range(0, N_TILES_B, 2))
        for n, i0 in enumerate(range(0, N_TILES_A, 2)):
            for j0 in reversed(j0s) if n % 2 else j0s:
                order += [(i0 + di, j0 + dj) for di in range(2) for dj in range(2)]
        return order

    def call(self, a, b, c, sa, sb, interleave=None, zero_acc=False):
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
        mth = [0]  # MFMA counter, indexes into ``slots``
        order = self._order()
        n_mfma = _FP4_PACK * len(order)
        slots = {(t * n_mfma) // len(thunks) for t in range(len(thunks))} if thunks else set()
        for ksub in range_constexpr(_FP4_PACK):
            for i, j in order:
                a_op = a[i][ksub]
                sa_v = sa[i // _FP4_PACK]
                ia = i % _FP4_PACK
                b_op = b[j][ksub]
                sb_v = sb[j // _FP4_PACK]
                jb = j % _FP4_PACK
                if zero_acc and ksub == 0:
                    # First K-sub of the first K-step: src2 is the inline constant 0,
                    # so the accumulator needs no v_accvgpr_write pre-initialization.
                    c[self.idx(i, j)] = self._mfma_agpr(a_op, b_op, None, sa_v, sb_v, ksub, ia, jb)
                else:
                    c[self.idx(i, j)] = self._mfma_agpr(a_op, b_op, c[self.idx(i, j)], sa_v, sb_v, ksub, ia, jb)
                if nth[0] < len(thunks) and mth[0] in slots:
                    thunks[nth[0]]()
                    nth[0] += 1
                mth[0] += 1
        while nth[0] < len(thunks):
            thunks[nth[0]]()
            nth[0] += 1
        return c

    def _mfma_agpr(self, a_op, b_op, acc, sa_v, sb_v, ksub, ia, jb):
        # Build the op_sel / op_sel_hi suffix (compile-time). op_sel[2]/hi[2]=0.
        #
        # feeds (B, A) instead of (A, B). Since C^T = B^T A^T,
        a_op, b_op = b_op, a_op
        sa_v, sb_v = sb_v, sa_v
        ia, jb = jb, ia
        opsel = f"op_sel:[{ia},{jb},0]"
        opsel_hi = f"op_sel_hi:[{ksub},{ksub},0]"
        src2 = "$0" if acc is not None else "0"
        asm = f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {src2}, $3, $4 {opsel} {opsel_hi} cbsz:4 blgp:4"
        ops = [
            fx.as_ir_value(a_op),
            fx.as_ir_value(b_op),
            fx.as_ir_value(sa_v),
            fx.as_ir_value(sb_v),
        ]
        cons = "=a,v,v,v,v"
        if acc is not None:
            ops.append(fx.as_ir_value(acc))
            cons += ",0"
        return _llvm.inline_asm(self.res_ty, ops, asm, cons, has_side_effects=True)


_SCALE_QUARTER_BYTES = 1024  # one gather = 4 blocks = one wave's share of one operand
_SCALE_REGION_BYTES = 2 * _SCALE_QUARTER_BYTES  # all of A (or all of B) for one K-step
_SCALE_SLOT_BYTES = _N_WAVES * _SCALE_QUARTER_BYTES  # 4096
_SCALE_SLOTS = 4
_SCALE_LDS_BYTES = _SCALE_SLOTS * _SCALE_SLOT_BYTES  # 16 KB
_SCALE_A_REGION = 0
_SCALE_B_REGION = _SCALE_REGION_BYTES


class ScaleGatherLDS:

    def __init__(self, a_scale, b_scale, K, lane_id, wave_id, lds_base_ptr, a_rows, b_rows, scopes=None):
        self.row_i32 = (K // 256) * 64
        self.wave_id = wave_id
        self.scopes = scopes
        # Exact num_records so the hardware OOB check is real: one e8m0 per 32
        # elements, i.e. K//32 bytes per row.
        row_bytes = K // 32
        self.a_rsrc = fx.as_ir_value(
            fx.rocdl.get_buffer_rsrc(
                _buffer_ops.create_buffer_resource(a_scale, max_size=False, num_records_bytes=a_rows * row_bytes)
            )
        )
        self.b_rsrc = fx.as_ir_value(
            fx.rocdl.get_buffer_rsrc(
                _buffer_ops.create_buffer_resource(b_scale, max_size=False, num_records_bytes=b_rows * row_bytes)
            )
        )
        # Per-lane block / within-block index (loop-invariant).
        self._blk = lane_id // 16  # 0..3 -> which of the 4 blocks
        self._in16 = lane_id % 16  # 0..15 -> which 4-i32 chunk within the block
        self._lds_base = fx.Int32(fx.ptrtoint(lds_base_ptr))

    def set_wave_base(self, a_base_tile, b_base_tile):

        wid = fx.Int32(_rocdl.readfirstlane(_T.i32, fx.as_ir_value(self.wave_id)))

        self._wave_base_s = fx.as_ir_value(self._lds_base + wid * fx.Int32(_SCALE_QUARTER_BYTES))

        is_a = wid < fx.Int32(2)
        q = wid % fx.Int32(2)
        base_tile = fx.arith.select(is_a, a_base_tile + q * fx.Int32(64), b_base_tile + q * fx.Int32(64))
        self._G = _uniform_i32(base_tile // fx.Int32(32))
        self._rsrc = fx.arith.select(is_a, self.a_rsrc, self.b_rsrc)
        # soffset=0 as a wave-uniform SGPR (readfirstlane'd once, reused every gather).
        self._soff0 = _uniform_i32(fx.Int32(0))

    def gather(self, kstep, slot):
        grp = fx.Int32(self._G) + (self._blk // 2) * fx.Int32(4) + (self._blk % 2)
        i32_off = grp * fx.Int32(self.row_i32) + fx.Int32(kstep) * fx.Int32(64) + self._in16 * fx.Int32(4)
        voff = fx.as_ir_value(i32_off * fx.Int32(4))  # bytes
        # m0 = precomputed wave quarter (SGPR) + slot*4096 (scalar): no readfirstlane.
        addr = fx.Int32(self._wave_base_s) + fx.Int32(slot) * fx.Int32(_SCALE_SLOT_BYTES)
        if self.scopes is not None:
            lds_ptr = _llvm.inttoptr(_lds_ptr_t(), fx.as_ir_value(fx.Int64(addr)))
            dma = _rocdl.raw_ptr_buffer_load_lds(
                self._rsrc, lds_ptr, fx.Int32(16), voff, self._soff0, fx.Int32(0), fx.Int32(0)
            )
            # Which of the two scale regions this lands in is wave-dependent, so
            # claim both; the point is that it misses all 8 tile buffers.
            _tag_alias(dma, self.scopes, (_SC_ASC, _SC_BSC))
            return
        asm = "s_mov_b32 m0, $0\nbuffer_load_dwordx4 $1, $2, $3 offen lds"
        _asm_void([fx.as_ir_value(addr), voff, self._rsrc, self._soff0], asm, "s,v,s,s", _M0_CLOBBER)


class ScaleLoaderLDS:

    def __init__(self, n_tiles, lane_id, quarter, lds_base_ptr, region_off, scopes=None, slot_id=None):
        assert n_tiles % _FP4_PACK == 0
        self.n_groups = n_tiles // _FP4_PACK
        self.lane_id = lane_id
        self.scopes = scopes
        self.slot_id = slot_id
        self._region_base = (
            fx.Int32(fx.ptrtoint(lds_base_ptr)) + fx.Int32(region_off) + quarter * fx.Int32(_SCALE_QUARTER_BYTES)
        )

    def _slot_wave_byte(self, slot):
        return self._region_base + fx.Int32(slot) * fx.Int32(_SCALE_SLOT_BYTES)

    def read_half(self, slot, half):
        L = self.lane_id
        base = self._slot_wave_byte(slot) + fx.Int32((L // 4) * 16 + (L % 4) * 4)
        grp_list = []
        for gi in range_constexpr(self.n_groups):
            blk = half * 2 + gi
            vaddr = base + fx.Int32(blk * 256)
            lds_ptr = _llvm.inttoptr(_lds_ptr_t(), fx.as_ir_value(vaddr))
            load = _llvm.LoadOp(fx.Int32.ir_type, lds_ptr, alignment=4)
            if self.scopes is not None:
                _tag_alias(load, self.scopes, self.slot_id)
            grp_list.append(fx.Int32(load.result))
        return grp_list

    def read(self, slot):
        """Per-lane ds_read of the 4 blocks -> (half0, half1), each list[n_groups]
        of i32 (the same shape the MFMA consumes: sa[i//2] / sb[j//2])."""
        return self.read_half(slot, 0), self.read_half(slot, 1)


class StoreCFp4:
    def __init__(
        self,
        C,
        c_rows,
        c_cols,
        c_idx_fn,
    ):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        # swapped_operand here, so the accumulator is
        # transposed: lane L holds C[L%16, 4 consecutive cols] rather than
        # C[4 consecutive rows, L%16]. Row/col roles below flip accordingly.
        c_nbytes = c_rows * c_cols * 2
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False, num_records_bytes=c_nbytes)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.out_atom_8 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        self.reg_bf16_8 = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)

    def _store_one(self, c_frag, base_row, base_col, ti, tj):
        vec_lo = Vec(c_frag[self.c_idx_fn(ti, tj)])
        vec_hi = Vec(c_frag[self.c_idx_fn(ti, tj + 1)])
        a0 = _cvt_pk_bf16(vec_lo[0], vec_lo[1])
        a1 = _cvt_pk_bf16(vec_lo[2], vec_lo[3])
        b0 = _cvt_pk_bf16(vec_hi[0], vec_hi[1])
        b1 = _cvt_pk_bf16(vec_hi[2], vec_hi[3])

        def _permlane16_swap(d_a, d_b):
            pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
            res = _rocdl.permlane16_swap(pair_ty, fx.as_ir_value(d_a), fx.as_ir_value(d_b), False, False)
            return _llvm.extractvalue(_T.i32, res, [0]), _llvm.extractvalue(_T.i32, res, [1])

        a0, b0 = _permlane16_swap(a0, b0)
        a1, b1 = _permlane16_swap(a1, b1)
        g = self.lane_id // 16
        row = base_row + ti * 16 + self.lane_id % 16
        col = base_col + (tj + g % 2) * 16 + (g // 2) * 8
        pack = Vec.from_elements([fx.Int32(a0), fx.Int32(a1), fx.Int32(b0), fx.Int32(b1)], fx.Int32).bitcast(
            fx.BFloat16
        )
        fx.memref_store_vec(pack, self.reg_bf16_8)
        c_index = row * self.c_cols + col
        fx.copy(self.out_atom_8, self.reg_bf16_8, fx.slice(self.c_div, (None, fx.Int32(c_index))))

    def thunks(self, c_frag, base_row, base_col):
        return [
            (lambda ti=ti, tj=tj: self._store_one(c_frag, base_row, base_col, ti, tj))
            for ti in range_constexpr(N_TILES_A)
            for tj in range_constexpr(0, N_TILES_B, 2)
        ]

    def store(self, c_frag, base_row, base_col):
        for t in self.thunks(c_frag, base_row, base_col):
            t()


def compile_fp4_gemm_4w(
    *,
    K: int,
    use_xcd_remap: bool = True,
    MN: tuple = None,
):
    """``MN=(M, N)`` bakes the output shape in as a compile-time constant.
    miss it is ok for this kernel.
    """
    BLOCK_K = 256
    BLOCK_K_BYTES = BLOCK_K // 2
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

    assert K % BLOCK_K == 0

    K_ITERS = K // BLOCK_K
    # not fully unroll, cause full unroll -> 111KB > 32KB I-cache.
    UNROLL = 4 if (K_ITERS - 4) % 4 == 0 else 2
    assert (K_ITERS - 4) % UNROLL == 0, K_ITERS
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

    N_LDS_ROUNDS = max(N_TILES_A, N_TILES_B)

    a_lds_size = LDS_BLOCK_M * BLOCK_K_BYTES
    b_lds_size = LDS_BLOCK_N * BLOCK_K_BYTES

    # One contiguous LDS array will save ~24 address VGPRs.
    assert a_lds_size == b_lds_size
    assert a_lds_size == _LDS_BUF_BYTES
    _lds_buf = a_lds_size

    @fx.struct
    class SharedStorage:
        all_lds: fx.Array[fx.Int8, 8 * _lds_buf, 16]
        scale_lds: fx.Array[fx.Int8, _SCALE_LDS_BYTES, 16]

    @flyc.kernel
    def kernel_gemm(
        A: fx.Tensor, B_T: fx.Tensor, C: fx.Tensor, A_scale: fx.Tensor, B_scale: fx.Tensor, c_m: fx.Int32, c_n: fx.Int32
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        _base_ptr = lds.all_lds.ptr

        def _buf(idx):
            return _Buf(_base_ptr, idx * _lds_buf)

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

        # Shadow the runtime c_m/c_n with compile-time constants so every divide
        # below folds away. See compile_fp4_gemm_4w's docstring.
        if const_expr(MN is not None):
            c_m, c_n = const_expr(MN[0]), const_expr(MN[1])
        n_blocks = ceildiv(c_n, BLOCK_N)
        if const_expr(use_xcd_remap):
            tile_i, tile_j = _xcd_swizzle(ceildiv(c_m, BLOCK_M), n_blocks)
        else:
            tile_i, tile_j = divmod(fx.block_idx.x, n_blocks)

        wave_i = wave_id // 2
        wave_j = wave_id % 2

        K_BYTES = K // 2
        A0_gl_offset = (tile_i * BLOCK_M) * K_BYTES
        A1_gl_offset = (tile_i * BLOCK_M + LDS_BLOCK_M) * K_BYTES
        A_K_STEP = BLOCK_K_BYTES
        B0_gl_offset = (tile_j * BLOCK_N) * K_BYTES
        B1_gl_offset = (tile_j * BLOCK_N + LDS_BLOCK_N) * K_BYTES
        # B is preshuffled (16,16): one N-16 row-block spans 2*1024 bytes per K-step
        B_K_STEP = 2 * 1024

        mfma = Mfma16x16x128Fp4()

        _scale_base_ptr = lds.scale_lds.ptr
        _sc = _lds_scopes() if _USE_DMA_INTRINSIC else None
        # One gather per wave covering the whole block-tile (see the geometry note
        # above); every wave reads its own wave_i / wave_j quarter back out.
        scale_gather = ScaleGatherLDS(A_scale, B_scale, K, lane_id, wave_id, _scale_base_ptr, c_m, c_n, _sc)
        scale_gather.set_wave_base(tile_i * BLOCK_M, tile_j * BLOCK_N)
        a_scale_ld = ScaleLoaderLDS(N_TILES_A, lane_id, wave_i, _scale_base_ptr, _SCALE_A_REGION, _sc, _SC_ASC)
        b_scale_ld = ScaleLoaderLDS(N_TILES_B, lane_id, wave_j, _scale_base_ptr, _SCALE_B_REGION, _sc, _SC_BSC)

        base_row = tile_i * BLOCK_M + wave_i * (N_TILES_A * 16)
        base_col = tile_j * BLOCK_N + wave_j * (N_TILES_B * 16)
        sa_R0 = base_row
        sa_R1 = base_row + LDS_BLOCK_M
        sb_C0 = base_col
        sb_C1 = base_col + LDS_BLOCK_N

        def _slot(k):
            return fx.Int32(k) % fx.Int32(_SCALE_SLOTS)

        def _gather_scales(k, slot):
            """Issue this wave's ONE dwordx4...lds for K-step ``k`` into LDS ``slot``.
            The 4 waves together cover the block-tile's A and B scales."""
            scale_gather.gather(k, slot)

        def _gather_scale_thunks(k, slot):
            """The gather as a thunk so it co-issues in the MFMA execute shadow
            instead of a bare buffer_load after all MFMAs (ATT showed the
            end-of-step gather exposed, not hidden)."""
            return [lambda: scale_gather.gather(k, slot)]

        # Accumulators: 2x2 64x64 quadrants per wave. They are NOT zero-initialized --
        # K-step 0 runs with ``zero_acc`` so its ksub-0 MFMAs write C = A*B directly.

        gl_off_a = _global_swizzle(lane_id, wave_id, K_BYTES, N_LDS_ROUNDS, False)
        gl_off_b = _global_swizzle(lane_id, wave_id, K_BYTES, N_LDS_ROUNDS, True)

        a_rsrc = fx.rocdl.get_buffer_rsrc(
            _buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=c_m * K_BYTES)
        )
        b_rsrc = fx.rocdl.get_buffer_rsrc(
            _buffer_ops.create_buffer_resource(B_T, max_size=False, num_records_bytes=c_n * K_BYTES)
        )
        a_g2s = G2SLoaderAsm(a_rsrc, gl_off_a, N_TILES_A, wave_id, scopes=_sc, base_ptr=_base_ptr)
        b_g2s = G2SLoaderAsm(b_rsrc, gl_off_b, N_TILES_B, wave_id, scopes=_sc, base_ptr=_base_ptr)
        # Precompute the g2s wave-uniform LDS base into SGPR once (all 8 buffers share
        # _base_ptr; per-buffer byte_off is compile-time) -> no per-load readfirstlane.
        a_g2s.set_wave_base(_base_ptr)
        b_g2s.set_wave_base(_base_ptr)
        a_s2r = S2RLoaderFp4(wave_i, N_TILES_A, scopes=_sc)
        b_s2r = S2RLoaderFp4(wave_j, N_TILES_B, scopes=_sc)

        _gather_scales(0, _slot(0))
        _gather_scales(1, _slot(1))
        _gather_scales(2, _slot(2))

        a_g2s.load(a_cur0, A0_gl_offset + 0 * A_K_STEP)
        b_g2s.load(b_cur0, B0_gl_offset + 0 * B_K_STEP)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * A_K_STEP)

        a_g2s.load(a_next0, A0_gl_offset + 1 * A_K_STEP)
        b_g2s.load(b_next0, B0_gl_offset + 1 * B_K_STEP)
        b_g2s.load(b_next1, B1_gl_offset + 1 * B_K_STEP)
        a_g2s.load(a_next1, A1_gl_offset + 1 * A_K_STEP)

        wait_barrier((3 * N_TILES_A) + (4 * N_TILES_B))
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier((3 * N_TILES_A) + (3 * N_TILES_B))
        b0_frag = b_s2r.load(b_cur0, preshuffled=True)
        # b1 is carried across steps (see _one_step); seed the carry here.
        b1_frag = b_s2r.load(b_cur1, preshuffled=True)

        # Initial VGPR scale carry = scale[0] (gathered first in prologue, landed by
        # the barriers above). The loop carries scale[kc] in VGPR; each step reads
        # scale[kc+1] in the MFMA shadow. saR0/saR1/sbC0/sbC1 each list[n_groups] i32.
        sc0_saR0, sc0_saR1 = a_scale_ld.read(_slot(0))
        sc0_sbC0, sc0_sbC1 = b_scale_ld.read(_slot(0))
        sc0 = (sc0_saR0, sc0_saR1, sc0_sbC0, sc0_sbC1)

        _MAIN_VMCNT = 16  # maybe higher here?
        _SEG2_VMCNT = 12

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

        def _one_step(kc, a0f, b0f, b1f_in, sc, accs, bufs, zero_acc=False):
            # bufs = (a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1)
            ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
            saR0, saR1, sbC0, sbC1 = sc
            c00f, c01f, c10f, c11f = accs
            kc_i = fx.Int32(kc)

            _a1 = [None] * N_TILES_A
            _a0n = [None] * N_TILES_A
            _b0n = [None] * N_TILES_B
            _b1n = [None] * N_TILES_B
            # This step prefetches K-step (kc+2) for g2s. a*_off = base + (kc+2)*STEP.
            ak = (kc_i + fx.Int32(2)) * fx.Int32(A_K_STEP)
            bk = (kc_i + fx.Int32(2)) * fx.Int32(B_K_STEP)
            a0_off = fx.Int32(A0_gl_offset) + ak
            a1_off = fx.Int32(A1_gl_offset) + ak
            b0_off = fx.Int32(B0_gl_offset) + bk
            b1_off = fx.Int32(B1_gl_offset) + bk

            _scn = [None, None, None, None]  # saR0, saR1, sbC0, sbC1 for kc+1
            _rd_scn = _read_scale_thunks(kc_i + 1, _scn)
            _gk = _min(kc_i + fx.Int32(3), fx.Int32(K_ITERS - 1))
            _sc_gather = _gather_scale_thunks(_gk, _slot(_gk))

            wait_barrier(_MAIN_VMCNT)
            il = (
                _riffle(_g2s_thunks(a_g2s, ac0, a0_off, N_TILES_A), _s2r_thunks(a_s2r, ac1, _a1, N_TILES_A, False))
                + _rd_scn[:2]
            )
            c00f = mfma.call(a0f, b0f, c00f, saR0, sbC0, interleave=il, zero_acc=zero_acc)

            il = _riffle(_g2s_thunks(b_g2s, bc0, b0_off, N_TILES_A), _rd_scn[2:])
            c01f = mfma.call(a0f, b1f_in, c01f, saR0, sbC1, interleave=il, zero_acc=zero_acc)
            a1f = _a1

            wait_barrier(_SEG2_VMCNT)
            il = (
                _riffle(_g2s_thunks(b_g2s, bc1, b1_off, N_TILES_A), _s2r_thunks(a_s2r, an0, _a0n, N_TILES_A, False))
                + _sc_gather
            )
            c10f = mfma.call(a1f, b0f, c10f, saR1, sbC0, interleave=il, zero_acc=zero_acc)
            a0nf = _a0n

            il = _riffle(
                _g2s_thunks(a_g2s, ac1, a1_off, N_TILES_A),
                _s2r_thunks(b_s2r, bn0, _b0n, N_TILES_B, True) + _s2r_thunks(b_s2r, bn1, _b1n, N_TILES_B, True),
            )
            c11f = mfma.call(a1f, b1f_in, c11f, saR1, sbC1, interleave=il, zero_acc=zero_acc)
            b0nf = _b0n
            b1nf = _b1n

            sc_next = (_scn[0], _scn[1], _scn[2], _scn[3])
            new_bufs = (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)  # swap cur<->next
            return a0nf, b0nf, b1nf, sc_next, (c00f, c01f, c10f, c11f), new_bufs

        bufs0 = (a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1)

        def _swap_bufs(bufs):
            """Same cur<->next swap ``_one_step`` returns; used by the peeled steps."""
            ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
            return (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)

        n_a = 2 * N_TILES_A
        n_b = 2 * N_TILES_B
        n_ga = N_TILES_A // _FP4_PACK  # scale groups per A half (=len(saR0))
        n_gb = N_TILES_B // _FP4_PACK
        n_sc = 2 * n_ga + 2 * n_gb  # sc = (saR0,saR1,sbC0,sbC1) flattened
        _R = fx.as_ir_value

        def _flat_sc(sc):
            saR0, saR1, sbC0, sbC1 = sc
            return [_R(v) for v in saR0] + [_R(v) for v in saR1] + [_R(v) for v in sbC0] + [_R(v) for v in sbC1]

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
            return (saR0, saR1, sbC0, sbC1)

        _accs0 = ([None] * N_ACCUMS, [None] * N_ACCUMS, [None] * N_ACCUMS, [None] * N_ACCUMS)
        a0f, b0f, b1f, sc, accs, _ = _one_step(0, a0_frag, b0_frag, b1_frag, sc0, _accs0, bufs0, zero_acc=True)
        a0f, b0f, b1f, sc, accs, _ = _one_step(1, a0f, b0f, b1f, sc, accs, _swap_bufs(bufs0))

        # Carry = a0/b0/b1 fragments + VGPR scale carry (scale[kc]) + 4 accumulator groups.
        init_state = (
            _flat_frag(a0f)
            + _flat_frag(b0f)
            + _flat_frag(b1f)
            + _flat_sc(sc)
            + [_R(x) for x in accs[0]]
            + [_R(x) for x in accs[1]]
            + [_R(x) for x in accs[2]]
            + [_R(x) for x in accs[3]]
        )
        for kk, state in range(2, K_ITERS - 2, UNROLL, init=init_state):
            off = 0
            a0f = _unflat_frag(state[off : off + n_a], N_TILES_A)
            off += n_a
            b0f = _unflat_frag(state[off : off + n_b], N_TILES_B)
            off += n_b
            b1f = _unflat_frag(state[off : off + n_b], N_TILES_B)
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

            # UNROLL steps. The pointer pair swaps once per step and UNROLL is even
            # -> back to bufs0 at body exit, so the LDS pointers stay loop-invariant
            # and need not be carried.
            bufs = bufs0
            for u in range_constexpr(UNROLL):
                a0f, b0f, b1f, sc, accs, bufs = _one_step(kk + u, a0f, b0f, b1f, sc, accs, bufs)

            new_state = (
                _flat_frag(a0f)
                + _flat_frag(b0f)
                + _flat_frag(b1f)
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
        b1_frag = _unflat_frag(state[off : off + n_b], N_TILES_B)
        off += n_b
        # VGPR carry at loop exit = scale[K_ITERS-2] (each iter advances sc by UNROLL).
        sc = _unflat_sc(state[off : off + n_sc])
        off += n_sc
        c00_frag = list(state[off : off + N_ACCUMS])
        off += N_ACCUMS
        c01_frag = list(state[off : off + N_ACCUMS])
        off += N_ACCUMS
        c10_frag = list(state[off : off + N_ACCUMS])
        off += N_ACCUMS
        c11_frag = list(state[off : off + N_ACCUMS])
        off += N_ACCUMS

        # Tail step K_ITERS - 2: scale[K_ITERS-2] is the carried VGPR sc (no read).
        # Read scale[K_ITERS-1] into the next carry in the c00 MFMA shadow.
        saR0, saR1, sbC0, sbC1 = sc
        _scn = [None, None, None, None]
        _rd_scn = _read_scale_thunks(fx.Int32(K_ITERS - 1), _scn)
        _a1 = [None] * N_TILES_A
        wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))
        il = _s2r_thunks(a_s2r, a_cur1, _a1, N_TILES_A, False) + _rd_scn
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, saR0, sbC0, interleave=il)
        a1_frag = _a1
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, saR0, sbC1)
        _a0n = [None] * N_TILES_A
        _b0n = [None] * N_TILES_B
        _b1n = [None] * N_TILES_B
        # One g2s batch of slack: b_next1 was issued by the last loop step's call3,
        # so only call4's a_cur1 batch may still be in flight (was 2 batches when
        # b1 was not read here).
        wait_barrier(1 * N_TILES_A)
        il = (
            _s2r_thunks(a_s2r, a_next0, _a0n, N_TILES_A, False)
            + _s2r_thunks(b_s2r, b_next0, _b0n, N_TILES_B, True)
            + _s2r_thunks(b_s2r, b_next1, _b1n, N_TILES_B, True)
        )
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, saR1, sbC0, interleave=il)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, saR1, sbC1)
        a0_frag = _a0n
        b0_frag = _b0n
        b1_frag = _b1n

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Tail step K_ITERS - 1: scale[K_ITERS-1] read into the carry above.
        _a1 = [None] * N_TILES_A
        wait_barrier(0)
        saR0, saR1, sbC0, sbC1 = (_scn[0], _scn[1], _scn[2], _scn[3])
        store_c = StoreCFp4(
            C=C,
            c_rows=c_m,
            c_cols=c_n,
            c_idx_fn=mfma.idx,
        )
        il = _s2r_thunks(a_s2r, a_cur1, _a1, N_TILES_A, False)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, saR0, sbC0, interleave=il)
        a1_frag = _a1
        _rocdl.sched_barrier(0)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, saR0, sbC1)
        _rocdl.sched_barrier(0)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, saR1, sbC0, interleave=store_c.thunks(c00_frag, sa_R0, sb_C0))
        _rocdl.sched_barrier(0)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, saR1, sbC1, interleave=store_c.thunks(c01_frag, sa_R0, sb_C1))
        _rocdl.sched_barrier(0)
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
        # should assert c_m % BLOCK_M == 0 and c_n % BLOCK_N == 0 here. but can't find a way to add it now.
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
