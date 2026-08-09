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
from flydsl._mlir.dialects import arith as _arith_d
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec
from kernels.common import buffer_ops as _buffer_ops
from kernels.gemm.fp8_gemm_utils import (
    ceildiv,
    divmod,
    swizzle_128,
)

_N_WAVES = 4  # block is always 256 threads -> 4 waves (compile-time constant)


def _global_swizzle(lane_id, wave_id, K, n_rounds, preshuffled):
    """``fp8_gemm_utils.compute_global_swizzle`` with the wave count pinned.

    Identical arithmetic; the only difference is that ``n_waves`` is the
    compile-time ``_N_WAVES`` instead of ``fx.block_dim.x // 64``. The runtime read
    is a kernarg ``s_load`` of hidden_group_size_x, and because it lands after the
    other kernarg loads it costs a SECOND ``s_waitcnt lgkmcnt(0)`` that every g2s
    address VGPR then waits behind.
    """
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


def _asm_void(operands, asm_string, constraints, clobbers=""):
    """Side-effecting void inline asm (LLVM sees no memory op -> no waitcnt added)."""
    if clobbers:
        constraints = f"{constraints},{clobbers}"
    _llvm.inline_asm(None, operands, asm_string, constraints, has_side_effects=True)


def _cvt_pk_bf16(a, b):
    """Pack two f32 into 2xbf16 (i32), as ``arith.truncf <2xf32> -> <2xbf16>``.

    Selects to the same single ``v_cvt_pk_bf16_f32`` on gfx950 as
    ``rocdl.cvt_pk_bf16_f32``, but that helper is inline asm (ROCDL has no op for
    this instruction -- see expr/rocdl/inline_asm.py), which plants an
    ASMSTART/ASMEND wall the machine scheduler cannot move across. 128 of them in
    the epilogue. Going through arith.truncf lets the backend select it as a
    normal SSA value: identical 128 v_cvt_pk_bf16_f32, identical 2578 instrs and
    440/256/66 VGPR/AGPR/SGPR, only epilogue scheduling differs.
    """
    v2f32 = _ir.VectorType.get([2], fx.Float32.ir_type)
    vec = Vec.from_elements([fx.Float32(a), fx.Float32(b)], fx.Float32)
    src = arith._to_raw(vec)
    if src.type != v2f32:
        src = _llvm.BitcastOp(v2f32, src).result
    v2bf16 = _ir.VectorType.get([2], fx.BFloat16.ir_type)
    return _llvm.BitcastOp(fx.Int32.ir_type, _arith_d.TruncFOp(v2bf16, src).result).result


# Any inline asm that writes m0 must say so. Only the FP4_DMA_INTRINSIC=0 fallback
# still has such asm, but the omission was a live bug: once the BACKEND materializes
# m0 for each buffer_load...lds it hoists those s_mov_b32 m0 freely, an undeclared
# writer stomps one, and the DMA lands at the wrong LDS address (cos=nan).
_M0_CLOBBER = "~{m0}"


def _enc_waitcnt_gfx9(vm, lgkm=15, exp=7):
    """gfx9/CDNA ``s_waitcnt`` SIMM16 encoding: ``vmcnt[3:0] | expcnt[6:4] | lgkmcnt[11:8] | vmcnt[15:14]``.

    The split-field mess is why gfx12 replaced it with s_wait_loadcnt/dscnt/storecnt,
    but those intrinsics are gfx12+ only ("Cannot select" on gfx950).
    """
    return (vm & 0xF) | ((exp & 0x7) << 4) | ((lgkm & 0xF) << 8) | (((vm >> 4) & 0x3) << 14)


def wait_barrier(count):
    """``s_waitcnt vmcnt(count) lgkmcnt(0)`` + ``s_barrier``.

    The lgkmcnt(0) is not redundant with the barrier: it makes the barrier also
    drain every outstanding ds_read, which lets the backend drop the standalone
    per-consumer lgkmcnt waits (60 -> 8 s_waitcnt in the hot loop). It only pays
    off because every s2r fragment is read one segment before it is consumed --
    see the b1 carry in ``_one_step``.

    Same semantics as ``fp8_gemm_utils.wait_barrier`` but emitted as ROCDL ops
    instead of inline asm: inline asm plants hard ``ASMSTART``/``ASMEND`` walls
    the machine scheduler cannot move instructions across, costing us prologue
    scheduling freedom. Hot-loop ISA is unchanged.
    """
    _rocdl.s_waitcnt(_enc_waitcnt_gfx9(count, lgkm=0))
    _rocdl.s_barrier()


# Emit g2s AND the scale gather as inline asm rather than the real
# rocdl.raw.ptr.buffer.load.lds intrinsic. DEFAULT ON; FP4_DMA_INTRINSIC=1 gets
# the intrinsic + LLVM-alias-scope path back.
#
# gfx9 `buffer_load ... lds` has no LDS-address field -- the destination can ONLY
# come from m0. The asm sets m0 once per load group and advances it with
# s_add_u32; the intrinsic hands m0 to the backend, which materializes one
# s_mov_b32 m0 per load unconditionally. Hot loop: s_mov_b32 20 vs 68 (49
# s_add_u32 replacing them), 911 vs 915 instrs. Measured +0.9% at 16384^3
# (5475/5484/5470 vs 5432/5415/5427, 3 alternating pairs).
#
# The intrinsic path was the default for a while, deliberately eating that cost:
# inline asm is opaque to the scheduler and hides real dependencies (a missing m0
# clobber in the scale gather silently corrupted every DMA once the backend
# started managing m0 -- see _M0_CLOBBER). Reverted because ~1% is too much to
# pay for it here. Closing the gap needs the backend to amortize m0 across loads.
#
# What the intrinsic path needs to stay viable, if it is ever revived: the alias
# scopes. Without them LLVM cannot prove the DMA misses the ds_reads and drops a
# `s_waitcnt vmcnt(0)` in front of every one. See _lds_scopes.
_USE_DMA_INTRINSIC = os.environ.get("FP4_DMA_INTRINSIC", "0") == "1"
_LDS_BUF_BYTES = 16384  # one of the 8 A/B tile buffers; asserted against a_lds_size
_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "fp4_gemm_4wave.lds">'
# One scope per disjoint LDS region: the 8 A/B tile buffers, then the two scale
# gather regions (separate allocation entirely).
_LDS_SCOPE_NAMES = [f"buf.{i}" for i in range(8)] + ["Asc", "Bsc"]
_SC_ASC = 8
_SC_BSC = 9


def _lds_scopes():
    """One ``#llvm.alias_scope`` per disjoint LDS region, in a shared domain.

    si-insert-waitcnts asks, for every LDS access, whether it may alias any
    outstanding LDS DMA. With alias info it checks them one at a time and waits
    only on the overlapping ones; without it, it conservatively waits on all of
    them, which for this loop means a full ``s_waitcnt vmcnt(0)`` immediately
    ahead of every ds_read. Our 8 buffers ping-pong so a ds_read never overlaps
    an in-flight DMA -- but every address is ptrtoint arithmetic off ONE 128 KB
    addrspace(3) symbol, which the backend cannot see through.
    """
    return [
        _ir.Attribute.parse(f'#llvm.alias_scope<id = "fp4_gemm_4wave.{n}", domain = {_LDS_DOMAIN}>')
        for n in _LDS_SCOPE_NAMES
    ]


def _tag_alias(op, scopes, slot):
    """Mark ``op`` as touching only LDS region(s) ``slot``, and no other.

    ``slot`` may be a tuple when the region is not statically known -- the scale
    gather writes A's or B's region depending on wave_id, so it claims both. What
    matters is that it still promises to miss all 8 tile buffers.
    """
    if os.environ.get("FP4_NO_ALIAS"):
        return
    slots = slot if isinstance(slot, tuple) else (slot,)
    op = getattr(op, "owner", op)
    op.attributes["alias_scopes"] = _ir.ArrayAttr.get([scopes[s] for s in slots])
    op.attributes["noalias_scopes"] = _ir.ArrayAttr.get([sc for i, sc in enumerate(scopes) if i not in slots])


def _uniform_i32(value):
    """Cast to i32 and force a wave-uniform SGPR value for scalar inline-asm operands."""
    raw = arith._to_raw(value) if not isinstance(value, _ir.Value) else value
    if raw.type != _T.i32:
        raw = arith._to_raw(fx.Int32(raw))
    return _rocdl.readfirstlane(_T.i32, raw)


class G2SLoaderAsm:
    """Global->LDS DMA via ``buffer_load_dwordx4 ... lds`` instead of the
    BufferCopyLDS128b copy atom.

    The problem this class exists to solve: with the 8 LDS buffers merged into ONE
    symbol (so ds_read cross-buffer offsets fold into the 16-bit imm), LLVM cannot
    prove the g2s LDS writes don't alias the ds_reads, and inserts a spurious
    ``s_waitcnt vmcnt(0)`` before every ds_read.

    Two ways out, selected by ``scopes``:

    * DEFAULT (``scopes`` set, _USE_DMA_INTRINSIC): the real
      rocdl.raw.ptr.buffer.load.lds intrinsic, tagged with per-buffer alias scopes
      so LLVM *can* prove the DMA misses the ds_reads. LLVM sees the write and
      still emits no drain.
    * FALLBACK (``scopes`` None): opaque inline asm, so LLVM sees no LDS write at
      all and has nothing to be conservative about.

    Either way vmcnt ordering is owned by our explicit ``wait_barrier``, and either
    way hardware still counts the load toward vmcnt, so the manual counts stay
    correct.

    Each ``buffer_load_dwordx4 vN, rsrc, soffset offen lds`` writes 16 bytes to the
    LDS address in m0; m0 holds the per-step LDS byte base (wave-uniform via
    readfirstlane), the per-lane swizzle is the voffset VGPR (loop-invariant), and
    the K-step offset is the scalar soffset operand (hardware-free add).
    """

    def __init__(self, rsrc, gl_offsets, n_load_steps, wave_id, scopes=None, base_ptr=None):
        self.rsrc = arith._to_raw(rsrc)
        self.gl_offsets = gl_offsets
        self.n_load_steps = n_load_steps
        self.wave_id = wave_id
        # When `scopes` is given, emit the real rocdl.raw.ptr.buffer.load.lds
        # intrinsic tagged with per-buffer alias scopes instead of opaque inline
        # asm: the compiler then sees the LDS write, but the scopes prove it does
        # not alias the ds_reads, so it still emits no drain -- while gaining the
        # freedom to schedule the loads (inline asm is an immovable barrier).
        self.scopes = scopes
        self.base_ptr = base_ptr

    @property
    def _step_stride(self):
        # m0 (LDS byte) advance between consecutive steps of one load. Must be a
        # Python int (baked into the asm string), so use the compile-time wave count
        # (block is always 256 -> 4 waves) rather than the runtime block_dim value.
        return _N_WAVES * 1024

    def set_wave_base(self, base_ptr):
        # Precompute the wave-uniform LDS base (ptrtoint(base) + wave_id*1024) into an
        # SGPR ONCE. _lds_base_sgpr then adds the per-buffer compile-time byte_off with
        # scalar arithmetic, so the step-0 m0 needs NO per-load readfirstlane (was 8/
        # iter: the m0 base came from wave_id, a VGPR, forcing readfirstlane).
        wb = fx.Int32(fx.ptrtoint(base_ptr)) + fx.Int32(self.wave_id * 1024)
        self._wave_base_s = _rocdl.readfirstlane(_T.i32, arith._to_raw(wb))

    def _lds_base_sgpr(self, lds_dst):
        # Step-0 LDS byte base (wave-uniform) = precomputed wave base (SGPR) + this
        # buffer's compile-time byte_off. Scalar -> no readfirstlane. Later steps add
        # _step_stride to m0.
        m0 = fx.Int32(self._wave_base_s) + fx.Int32(lds_dst.byte_off)
        return arith._to_raw(m0)

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
        if self.scopes is not None:
            # Intrinsic form: the LDS destination is an explicit addrspace(3) pointer
            # (no m0 juggling -- the backend materializes m0 itself), and the alias
            # scope tells si-insert-waitcnts this DMA touches only buffer `slot`.
            slot = lds_dst.byte_off // _LDS_BUF_BYTES
            # Build off the ONE readfirstlane'd wave base (set_wave_base) rather than
            # ptrtoint(base)+wave_id*1024: wave_id is a VGPR, so the latter makes the
            # backend emit a v_readfirstlane per load to get m0 into an SGPR.
            addr = fx.Int64(fx.Int32(self._wave_base_s) + fx.Int32(lds_dst.byte_off + step * stride))
            lds_ptr = _llvm.inttoptr(_lds_ptr_t(), arith._to_raw(addr))
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
        # Plain LLVM ds_read (NOT inline-asm). The single-symbol LDS layout would
        # normally make the compiler insert an s_waitcnt vmcnt(0) drain before each
        # ds_read (it can't prove the g2s global->LDS DMA writes don't alias) -- see
        # G2SLoaderAsm for the two ways that is avoided (alias scopes by default).
        # And being a REAL LDS load, the
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


def _riffle(glb, lds):
    """Interleave the global and LDS thunk lists proportionally instead of
    concatenating them. Concatenated, the 4 g2s issue back-to-back and each stalls
    on a full TA queue (ATT: buffer_load ARBITER_WIN_EX 53%); riffled, LDS reads sit
    between consecutive global loads so the TA queue drains. aiter's loop is built
    the same way -- its sequence alternates global and LDS, never 4 global in a row."""
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
    return arith.select(a < b, a, b)


def _divmod_nonneg(a, b):
    """``divmod(a, b)`` where ``a >= 0`` is known and ``b`` may be a constant.

    Everything in ``_xcd_swizzle`` derives from ``block_idx.x``, so it is always
    non-negative -- but its type is signed i32, so plain ``//`` and ``%`` lower to
    ``floordivsi``/``floormodsi``, and those carry a sign-correction chain
    (``s_ashr 31`` / ``s_lshr`` / ``s_add`` / ``s_ashr``, plus a ``s_cmp`` +
    ``s_cselect`` + ``s_subb`` fixup) that is dead code here. For a power-of-two
    divisor the whole thing collapses to one shift and one mask.
    """
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
        # group_id < num_pid_m/WGM, so first_pid_m <= num_pid_m - WGM and the min is
        # always WGM. Worth special-casing: a variable group_size_m makes the divmod
        # below a RUNTIME divide, and gfx950 has no scalar divider -- it becomes a
        # ~45-instruction v_rcp_iflag_f32 Newton sequence sitting in front of every
        # address, hence in front of the first buffer_load.
        group_size_m = WGM
    else:
        group_size_m = _min(num_pid_m - first_pid_m, WGM)
    pid_n, intra_group_m = _divmod_nonneg(intra_group, group_size_m)
    pid_m = first_pid_m + intra_group_m

    use_simple = (num_wg < SWIZZLE_THRESHOLD) | (num_wg % NUM_XCDS != 0)
    if const_expr(isinstance(use_simple, bool)):
        # num_pid_m/n are compile-time (M/N pinned), so the whole predicate folds
        # and only one of the two mappings needs to be emitted.
        return (simple_m, simple_n) if use_simple else (pid_m, pid_n)
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

    def __init__(self):
        assert N_TILES_A % _FP4_PACK == 0 and N_TILES_B % _FP4_PACK == 0
        self.accum_type = Vec.make_type(4, fx.Float32)
        self.res_ty = Vec.make_type(4, fx.Float32)

    def idx(self, i, j):
        return i * N_TILES_B + j

    def _order(self):
        """(i, j) emission order for one ksub.

        The plain ``for i: for j:`` sweep holds one operand fixed across a whole
        row and cycles the other, so the cycled side never hits: the XDL operand
        buffer is 8 registers per side = 2 slots of v[n:n+3], and a row of
        n_tiles_b=4 distinct operands evicts each entry before it is reused.

        A 2x2 tile touches exactly 2 distinct i and 2 distinct j, which is what
        those 2 slots hold, so of the 4 MFMAs only the first fetches on either
        side::

            (i,j) (i,j+1) (i+1,j) (i+1,j+1)   ->  2 A fetches + 2 B fetches
            row sweep, 4 wide                 ->  4 A fetches + 1 B fetch

        The j0 sweep is serpentine (0,2 then 2,0) rather than restarting at 0 for
        every i0 row: at the row turn the previous row's last 2x2 tile holds the
        same B pair the new row starts with, so that tile's two B fetches become
        hits, and the same carry then survives across the call boundary into the
        next quadrant. Plain 2x2 restarts at j0=0 and evicts it.

        Modelled over the full 512-MFMA body: 640 (row sweep) -> 384 (plain 2x2)
        -> 320 (serpentine), which is the floor for pure reordering under the
        four-quadrant call structure. Order only; the (i,j) set and each
        accumulator's operands are unchanged.
        """
        order = []
        j0s = list(range(0, N_TILES_B, 2))
        for n, i0 in enumerate(range(0, N_TILES_A, 2)):
            for j0 in (reversed(j0s) if n % 2 else j0s):
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
        # Thunk placement. A fixed stride (the previous scheme) packs every thunk
        # into the first stride*len(thunks) MFMAs and leaves the rest of the call a
        # bare back-to-back MFMA tail -- ATT showed those tail MFMAs stalling 12 cyc
        # each while the interleaved region cost a few hundred cycles in total.
        # Spread thunks evenly over ALL n_mfma slots so every one gets the same
        # execute-shadow and no tail is left uncovered: thunk t goes after MFMA
        # floor(t * n_mfma / n_thunks). Max MFMA run drops 8 -> 3.
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
        # ``swap_operands`` feeds (B, A) instead of (A, B). Since C^T = B^T A^T,
        # the accumulator then holds the transpose: lane L owns C[L%16, 4 cols]
        # -- row-contiguous -- instead of C[4 rows, L%16]. That is what lets the
        # epilogue store wide (dwordx4) instead of one bf16 at a time.
        #
        # MFMA's two source operands have symmetric lane layouts (M/N from
        # lane%16, K from lane//16), so swapping needs no change to the S2R
        # loaders. Only the op_sel byte-select and the scale operands follow the
        # data to the other side.
        a_op, b_op = b_op, a_op
        sa_v, sb_v = sb_v, sa_v
        ia, jb = jb, ia
        opsel = f"op_sel:[{ia},{jb},0]"
        opsel_hi = f"op_sel_hi:[{ksub},{ksub},0]"
        # acc=None -> src2 is the literal 0 (C = A*B, not A*B + C). Used for the
        # first K-sub of the first K-step so the 256 v_accvgpr_write_b32 that
        # would zero the accumulators disappear entirely.
        src2 = "$0" if acc is not None else "0"
        asm = f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {src2}, $3, $4 " f"{opsel} {opsel_hi} cbsz:4 blgp:4"
        ops = [
            arith._to_raw(a_op),
            arith._to_raw(b_op),
            arith._to_raw(sa_v),
            arith._to_raw(sb_v),
        ]
        cons = "=a,v,v,v,v"
        if acc is not None:
            ops.append(arith._to_raw(acc))
            cons += ",0"
        return _llvm.inline_asm(self.res_ty, ops, asm, cons, has_side_effects=True)


# Scale-LDS geometry. A block's scales for one K-step are 8 blocks (256 B) for A
# (BLOCK_M/32) and 8 for B, i.e. 2048 B each -- and one buffer_load_dwordx4...lds
# is 64 lanes x 16 B = 1024 B = 4 blocks. So the whole block-tile's scales are
# exactly FOUR gathers, and the 4 waves take one each:
#
#   wave 0 -> A groups {G,G+1,G+4,G+5}   wave 2 -> B, same shape
#   wave 1 -> A groups {G+2,G+3,G+6,G+7} wave 3 -> B, same shape
#
# (each quarter is one wave_i/wave_j's 64 rows of BOTH M/N halves, i.e. the same
# 4 blocks that wave_i would have fetched for itself before.)
#
# Every wave then reads all four quarters (its own A half + its own B half),
# which the existing wait_barrier already orders -- same cross-wave visibility
# the 8 tile buffers rely on. Having each wave gather its own A *and* B copy
# instead would fetch each quarter twice (waves 0/1 share wave_i, 2/3 share
# wave_j) and cost 4 extra buffer_load + 4 s_mov m0 per K-step.
#
# Layout per slot: [A 2048][B 2048], each half two 1024 B wave quarters, so the
# gather destination is just slot_base + wave_id*1024. Slots are kstep%_SCALE_SLOTS;
# 4 K-steps are live at once (carry[kc], read[kc+1], gather[kc+2], gather[kc+3])
# so 4 slots are needed or one is overwritten before its read (3 -> nan).
_SCALE_QUARTER_BYTES = 1024  # one gather = 4 blocks = one wave's share of one operand
_SCALE_REGION_BYTES = 2 * _SCALE_QUARTER_BYTES  # all of A (or all of B) for one K-step
_SCALE_SLOT_BYTES = _N_WAVES * _SCALE_QUARTER_BYTES  # 4096
_SCALE_SLOTS = 4
_SCALE_LDS_BYTES = _SCALE_SLOTS * _SCALE_SLOT_BYTES  # 16 KB
_SCALE_A_REGION = 0
_SCALE_B_REGION = _SCALE_REGION_BYTES


class ScaleGatherLDS:
    """The block-tile's whole scale gather for one K-step: FOUR
    ``buffer_load_dwordx4 ... lds``, one per wave, no duplication.

    Which quarter a wave fetches is picked by wave_id -- operand from ``wave_id
    // 2`` and 64-row group from ``wave_id % 2``. Both selectors are
    wave-uniform, so this is scalar select (SGPR), NOT divergent control flow:
    every wave still executes exactly one straight-line DMA, which is what lets
    the gather stay a thunk inside the MFMA interleave.
    """

    def __init__(self, a_scale, b_scale, K, lane_id, wave_id, lds_base_ptr, scopes=None):
        self.row_i32 = (K // 256) * 64  # i32 per N1 group
        self.wave_id = wave_id
        self.scopes = scopes
        self.a_rsrc = arith._to_raw(_buffer_ops.create_buffer_resource(a_scale, max_size=True))
        self.b_rsrc = arith._to_raw(_buffer_ops.create_buffer_resource(b_scale, max_size=True))
        # Per-lane block / within-block index (loop-invariant).
        self._blk = lane_id // 16  # 0..3 -> which of the 4 blocks
        self._in16 = lane_id % 16  # 0..15 -> which 4-i32 chunk within the block
        self._lds_base = fx.Int32(fx.ptrtoint(lds_base_ptr))

    def set_wave_base(self, a_base_tile, b_base_tile):
        """Resolve this wave's share ONCE into SGPRs: the LDS quarter it writes,
        the operand resource it reads, and that operand's base row/col. Keeps
        gather() free of both readfirstlane and any re-selection."""
        # readfirstlane FIRST: wave_id descends from thread_idx, so without this the
        # uniformity analysis calls the resource select divergent and wraps every
        # gather in a readfirstlane waterfall loop (s_and_saveexec / s_cbranch_execnz)
        # -- a branch inside the MFMA block, which destroys the thunk interleave.
        wid = fx.Int32(_rocdl.readfirstlane(_T.i32, arith._to_raw(self.wave_id)))
        # LDS: quarter wave_id of the slot -- [A q0][A q1][B q0][B q1].
        self._wave_base_s = arith._to_raw(self._lds_base + wid * fx.Int32(_SCALE_QUARTER_BYTES))
        # Operand: waves 0/1 -> A, waves 2/3 -> B. Quarter q = wave_id % 2 selects
        # the 64-row group, matching the wave_i / wave_j the readers use.
        is_a = wid < fx.Int32(2)
        q = wid % fx.Int32(2)
        base_tile = arith.select(is_a, a_base_tile + q * fx.Int32(64), b_base_tile + q * fx.Int32(64))
        self._G = _uniform_i32(base_tile // fx.Int32(32))
        self._rsrc = _llvm.SelectOp(arith._to_raw(is_a), self.a_rsrc, self.b_rsrc).result
        # soffset=0 as a wave-uniform SGPR (readfirstlane'd once, reused every gather).
        self._soff0 = _uniform_i32(fx.Int32(0))

    def gather(self, kstep, slot):
        """ONE buffer_load_dwordx4...lds writing this wave's quarter of scale-LDS
        ``slot``. vmcnt accounting is owned by wait_barrier."""
        # The LDS side is not addressable: lane L's 16 B always lands at m0 + L*16.
        # So lane L must READ the global data destined for there: block (L//16),
        # i32 chunk (L%16)*4..+3. block blk -> group G + (blk//2)*4 + (blk%2).
        grp = fx.Int32(self._G) + (self._blk // 2) * fx.Int32(4) + (self._blk % 2)
        i32_off = grp * fx.Int32(self.row_i32) + fx.Int32(kstep) * fx.Int32(64) + self._in16 * fx.Int32(4)
        voff = arith._to_raw(i32_off * fx.Int32(4))  # bytes
        # m0 = precomputed wave quarter (SGPR) + slot*4096 (scalar): no readfirstlane.
        addr = fx.Int32(self._wave_base_s) + fx.Int32(slot) * fx.Int32(_SCALE_SLOT_BYTES)
        if self.scopes is not None:
            lds_ptr = _llvm.inttoptr(_lds_ptr_t(), arith._to_raw(fx.Int64(addr)))
            dma = _rocdl.raw_ptr_buffer_load_lds(
                self._rsrc, lds_ptr, fx.Int32(16), voff, self._soff0, fx.Int32(0), fx.Int32(0)
            )
            # Which of the two scale regions this lands in is wave-dependent, so
            # claim both; the point is that it misses all 8 tile buffers.
            _tag_alias(dma, self.scopes, (_SC_ASC, _SC_BSC))
            return
        asm = "s_mov_b32 m0, $0\nbuffer_load_dwordx4 $1, $2, $3 offen lds"
        _asm_void([arith._to_raw(addr), voff, self._rsrc, self._soff0], asm, "s,v,s,s", _M0_CLOBBER)


class ScaleLoaderLDS:
    """Per-lane ``ds_read_b32`` of one operand's ``shuffle_scale_w4``-PRESHUFFLED
    per-1x32 E8M0 scales out of scale LDS. (Replaced 8 per-step
    ``buffer_load_dword``, each with a dur-6 voffset v_add -- the #1 exposed
    hot-loop cost.) ScaleGatherLDS fills the LDS.

    Layout (gate_up=False): per (N1 group, K-step) the e8m0 form a 64-i32
    (256 B) block ``[K_Lane(4), N_Lane(16)]`` in which lane L's MFMA scale is
    element L. A wave's 4 blocks are groups ``{G, G+1, G+4, G+5}`` (G+4 == the
    second M/N half's group, since LDS_BLOCK/32 == 4) -- exactly the quarter
    some wave gathered, so the read needs no cross-quarter addressing.
    """

    def __init__(self, n_tiles, lane_id, quarter, lds_base_ptr, region_off, scopes=None, slot_id=None):
        assert n_tiles % _FP4_PACK == 0
        self.n_groups = n_tiles // _FP4_PACK  # pack-groups per M/N half (=2)
        self.lane_id = lane_id
        self.scopes = scopes
        self.slot_id = slot_id
        # This operand's quarter: wave_i for A, wave_j for B. The gather wave that
        # wrote it (wave_id = operand*2 + quarter) is a different wave in general;
        # wait_barrier is what makes the write visible.
        self._region_base = (
            fx.Int32(fx.ptrtoint(lds_base_ptr)) + fx.Int32(region_off) + quarter * fx.Int32(_SCALE_QUARTER_BYTES)
        )

    def _slot_wave_byte(self, slot):
        return self._region_base + fx.Int32(slot) * fx.Int32(_SCALE_SLOT_BYTES)

    def read_half(self, slot, half):
        """Per-lane ds_read of ONE half (2 blocks) -> list[n_groups] of i32.
        Split from read() so half-1 can be issued as a thunk in an MFMA shadow
        (half-0 feeds c00/c01, half-1 feeds c10/c11 -- two MFMA clusters later)."""
        # Inverse of gather's m0 + L*16 write: block-elem e of block blk sits at
        # LDS byte blk*256 + (e//4)*16 + (e%4)*4. MFMA lane L wants elem L.
        L = self.lane_id
        base = self._slot_wave_byte(slot) + fx.Int32((L // 4) * 16 + (L % 4) * 4)
        grp_list = []
        for gi in range_constexpr(self.n_groups):
            blk = half * 2 + gi
            vaddr = base + fx.Int32(blk * 256)
            lds_ptr = _llvm.inttoptr(_lds_ptr_t(), arith._to_raw(vaddr))
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
            res = _rocdl.permlane16_swap(pair_ty, arith._to_raw(d_a), arith._to_raw(d_b), False, False)
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
        """One zero-arg thunk per ``buffer_store_dwordx4`` (8 for a quadrant).

        Handed to ``Mfma16x16x128Fp4.call(interleave=...)`` so a later quadrant's
        MFMAs sit between this quadrant's stores. Back-to-back stores queue up on
        L1 rather than on bandwidth, so spacing them out is what pays -- the MFMAs
        are free filler, and the accumulator reads / bf16 converts that each thunk
        drags along get hidden in the MFMA execute shadow too.

        The body reads ``c_frag`` when the thunk RUNS, not when it is built, but
        the caller always builds a quadrant's thunks after that quadrant's
        ``mfma.call`` has returned, so the values are final either way.
        """
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

    The kernel still takes c_m/c_n as arguments (the signature is unchanged); they
    are simply ignored, so the caller must pass the same M/N it compiled for.

    Worth doing because c_m/c_n only ever feed integer DIVISIONS -- ceildiv to a
    block count, then four divmods in ``_xcd_swizzle`` -- and a runtime scalar
    divide is a ~20-instruction software sequence with no divider in hardware.
    That whole dependency chain sits in front of the first buffer_load, so the
    wave issues nothing to memory until it finishes: 272 instructions (199 SALU)
    before the first global load, vs 168 (107 SALU) with M/N pinned.

    +1.0% at 8192x8192x4096 (4662-4688 vs 4607-4633 TFLOPS, non-overlapping over
    4 alternating pairs); neutral at 16384^3, where the fixed cost is amortized.
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
        scale_gather = ScaleGatherLDS(A_scale, B_scale, K, lane_id, wave_id, _scale_base_ptr, _sc)
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

        # g2s (see G2SLoaderAsm): needs the raw buffer resource. Build it
        # once from the i8 buffer tensor (max_size OOB check; all addresses in-bounds).
        a_rsrc = _buffer_ops.create_buffer_resource(A, max_size=True)
        b_rsrc = _buffer_ops.create_buffer_resource(B_T, max_size=True)
        a_g2s = G2SLoaderAsm(a_rsrc, gl_off_a, N_TILES_A, wave_id, scopes=_sc, base_ptr=_base_ptr)
        b_g2s = G2SLoaderAsm(b_rsrc, gl_off_b, N_TILES_B, wave_id, scopes=_sc, base_ptr=_base_ptr)
        # Precompute the g2s wave-uniform LDS base into SGPR once (all 8 buffers share
        # _base_ptr; per-buffer byte_off is compile-time) -> no per-load readfirstlane.
        a_g2s.set_wave_base(_base_ptr)
        b_g2s.set_wave_base(_base_ptr)
        a_s2r = S2RLoaderFp4(wave_i, N_TILES_A, scopes=_sc)
        b_s2r = S2RLoaderFp4(wave_j, N_TILES_B, scopes=_sc)

        # Prologue. Scale gathers for step 0/1/2 go FIRST (before the 32 g2s) so they
        # are the OLDEST outstanding VMEM -- the main loop's wait_barrier then
        # drains them naturally (vs issuing them last, where the vmcnt cannot reach
        # them past the 32 newer g2s -> step-0 read got un-landed LDS -> big-K nan).
        # DEPTH-3: scale[0] feeds the initial VGPR carry, scale[1]/scale[2] are read
        # (into carry) during step 0/1; step kc then gathers scale[kc+3].
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

        # A step issues 16 g2s + 1 gather = 17 VMEM, in cluster order
        #   [4 g2s][4 g2s][4 g2s + gather][4 g2s]
        # and consumes what step kc-2 wrote (g2s prefetches kc+2; scale[X] is
        # gathered at step X-3 and read at X-1). vmcnt(16) leaves at most 16
        # outstanding at the top of a step, i.e. one short of a full step, so
        # everything through step kc-2 -- gather included -- has landed.
        # (Was 17 of 18 when every wave gathered both operands: same one-load
        # margin, since dropping the duplicate gather shrank the step by one.)
        _MAIN_VMCNT = int(os.environ.get("FP4_MAIN_VMCNT", "16"))
        # The seg-2 barrier needs a tighter count than seg-1: it guards the b1
        # fragment read for step kc+1, whose g2s went out in the PREVIOUS step's
        # call3 and so has only 12 newer VMEM ops behind it (vs 21/17 for a0n/b0n,
        # issued in call1/call2). 16 would let those 4 loads still be in flight.
        _SEG2_VMCNT = int(os.environ.get("FP4_SEG2_VMCNT", "12"))

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
            # DEPTH-3 scale + VGPR carry: ``sc`` = scale[kc] ALREADY in VGPR (read in
            # the prior step's MFMA shadow), so there is NO scale ds_read / lgkmcnt
            # wait at the top of this step. This step (a) reads scale[kc+1] into the
            # next carry inside the MFMA shadow, and (b) gathers scale[kc+3] (depth-3,
            # so scale[kc+1] -- gathered at step kc-2 -- is landed at the barrier here).
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

            # Next-step scale carry: read scale[kc+1] in the MFMA shadow (4 thunks).
            _scn = [None, None, None, None]  # saR0, saR1, sbC0, sbC1 for kc+1
            _rd_scn = _read_scale_thunks(kc_i + 1, _scn)
            # Scale[kc+3] gather (depth-3), co-issued in the MFMA shadow (2 thunks).
            # Clamp to K_ITERS-1: the last loop step would gather scale[K_ITERS] (OOB
            # -> memory fault); the clamped extra gather re-reads scale[K_ITERS-1]
            # into the same slot (idempotent, its result is never consumed).
            _gk = _min(kc_i + fx.Int32(3), fx.Int32(K_ITERS - 1))
            _sc_gather = _gather_scale_thunks(_gk, _slot(_gk))

            # b1 is a loop carry (read in the PREVIOUS step's seg 2), exactly like
            # a0/b0. That is what keeps every s2r read separated from its consumer
            # by a barrier: with b1 read here in seg 1 and consumed by c01 in the
            # same segment, the backend had to emit a standalone lgkmcnt wait for
            # it (20 per hot loop). All four fragments now cross the barrier's
            # lgkmcnt(0) instead, so those waits disappear.
            #
            # Only a1 is still read-and-used inside one segment, but its read sits
            # in seg 1 and its use in seg 2 -- a barrier apart.
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
        _R = arith._to_raw

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

        # Steps 0 and 1 are peeled out of the scf.for so step 0 can run with
        # ``zero_acc``: its ksub-0 MFMAs take the literal 0 as src2 (C = A*B), which
        # removes the 256 v_accvgpr_write_b32 that would otherwise zero the
        # accumulators before the loop. Two steps are peeled (not one) to keep the
        # LDS ping-pong identity -- the pointer pairs swap twice, so the loop body
        # still sees ``bufs0``.
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
        # Each quadrant's 8 stores ride in a LATER quadrant's MFMA slots. Issuing
        # all 32 buffer_stores back to back queues them on L1 (not on bandwidth),
        # so the win is spacing them out; the MFMAs are free filler and the
        # accvgpr_read / cvt_pk_bf16 each thunk drags along hide in the MFMA
        # execute shadow.
        #
        # The sched_barrier(0) after each call is load-bearing, not a hint.
        # ``_mfma_agpr`` emits the MFMA as inline asm, so GCNHazardRecognizer
        # cannot see the AGPR write and inserts no nops for the MFMA ->
        # v_accvgpr_read RAW hazard. Without the barrier the machine scheduler
        # sinks a c00 MFMA to 2 instructions before the thunk that reads that same
        # accumulator, and the epilogue reads stale data (cos 0.999999 ->
        # 0.999512). The barrier pins every quadrant's MFMAs ahead of the stores
        # that read them, leaving a full 32-MFMA call in between.
        #
        # Stores go two calls back, not one: one call back still let a tail MFMA
        # land inside the hazard window (cos 0.999877). c10/c11 have no later call
        # to hide in.
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
