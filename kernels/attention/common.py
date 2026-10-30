# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Arch-neutral device helpers shared by the attention kernels.

Ported from AOTriton's `fmha_common_gfx1201.py`, keeping only the subset the
gfx950 kernels use. Nothing here names a GPU architecture: masked axes, the
VarlenBits decode, LSE row addressing, sliding-window resolution and the Philox
seed/offset plumbing are properties of the attention ABI, not of a target.

**Why the branching helpers are written as they are.** The rewrite from Python's
`if` to `scf.if` is lexical per `@flyc.kernel`/`@flyc.jit` function, so a plain
module-level function gets a branch only by building the `scf.IfOp` itself --
which is what `cond_load` and `_load_u64_or_zero` do. A null-pointer guard has
to be a real `scf.if`, not a select: a select evaluates both arms and would
fault on the null. `philox_report` and `_store_u64_if_nonnull` are `@flyc.jit`
helpers instead, so they can use a Python `if`.

One trap remains for kernel-side code. `ast_rewriter._collect_assigned_vars`
counts `name.method(...)` under a dynamic `if` as a use of carried state and
assigns `name` back after the region. If `name` came from an *enclosing* scope
that makes it a local of the inner function, unbound on a sibling path that
skips the `if`. Put the code in a module-level function, call a free function
so the base name is a module, or bind a local alias before the branch.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf as _scf
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from kernels.common.utils import sdiv_rd_pow2, smax, smin, ssel

__all__ = [
    "MASK_SAFE_FASTMATH",
    "lse_row_addressing",
    "lse_row_step_per_head",
    "bitcast_i32",
    "MaskedAxis",
    "cond_load",
    "seqinfo_addr",
    "decode_addressing",
    "lse_token_pitch",
    "WINDOW_TOPLEFT",
    "WINDOW_BOTRIGHT",
    "resolve_window",
    "CausalRegions",
    "decompose_causal_regions",
    "philox_offset_base",
    "philox_seed_value",
    "philox_report",
]

# Fast-math flags for the few sites that must not carry `ninf`/`nnan`. Under the
# ambient `fast` compile hint LLVM may delete a -inf mask or fold `log(0)` to
# poison: the bias add, the dK/dV bias chain, the dQ LSE floor and the
# masked-row LSE log/select. Use as `with fx.fastmath(MASK_SAFE_FASTMATH):`
# around operator expressions, or as `fastmath=MASK_SAFE_FASTMATH` on
# `fx.log` / `fx.maxnumf`. One name keeps every exemption greppable, and keeps
# FMA contraction.
MASK_SAFE_FASTMATH = fx.FastMathFlags.contract | fx.FastMathFlags.reassoc  # no ninf/nnan/afn

# No fast-math flags, for the two operations that form a softmax exponent: the score scale and the row-max subtract.
# With `contract` on both, LLVM fuses them into `exp2(fma(s, c, -m))` -- an unrounded product against a max that was
# rounded -- whose residue reaches +-2**11 at scores near 5e10, so `exp2` gives inf or 0 and the output NaN (aotriton
# issue 54, `test_large_bf16_nan_values`). This is not a speed trade to revisit: the FMA form must not be used.
SOFTMAX_EXPONENT_FASTMATH = fx.FastMathFlags.none


def lse_row_addressing(varlen_bits, batch, head, num_head_q, tokens, row_off):
    """`(base, pitch)` for a row-wise f32 side input -- logsumexp, and delta.

    The element for absolute query row `r` is at `base + r * pitch`. Factored
    that way because `base` is loop-invariant while `r` is not: the four
    callers that had open-coded this each recomputed the whole offset per row.

    Two layouts, one select, decoded from VarlenBits bits 17:16:

        _HT   (H, T), T contiguous -- AOTriton's and this kernel's default.
              base = (batch * H + head) * tokens + row_off,   pitch = 1
        _TH   (T, H), H contiguous -- Transformer Engine's.
              base = (batch * tokens + row_off) * H + head,   pitch = H

    `tokens` is `lse_token_pitch`'s answer, not `max_seqlen_q`: a stacked
    layout runs to the batch total rather than padding each row-group.

    **Delta is required to share LSE's layout.** It is produced beside it by
    the same caller, and giving it its own decode would double the work for no
    expressiveness -- so both side inputs use this one function.

    Four callers of one fact: the forward *writes* LSE through it, and all
    three backward kernels *read* LSE and delta through it. It was two
    spellings and a third that had already factored out the base, which is the
    form kept here.
    """
    tok = fx.Index(tokens)
    nhq = fx.Index(num_head_q)
    base_ht = (batch * nhq + head) * tok + row_off
    base_th = (batch * tok + row_off) * nhq + head
    is_th = ((fx.Int32(varlen_bits) >> fx.Int32(16)) & fx.Int32(3)) != fx.Int32(0)
    base = fx.Index(is_th.select(fx.Index(base_th), fx.Index(base_ht)))
    pitch = fx.Index(is_th.select(nhq, fx.Index(1)))
    return base, pitch


def lse_row_step_per_head(varlen_bits, tokens):
    """How far `lse_row_addressing`'s `base` moves when `head` rises by one.

    `base` is affine in `head` under both layouts -- `_HT` has it inside the
    `* tokens` product, `_TH` outside it -- so a caller walking a GQA group can
    add this per head instead of re-running the decode. That is not a peephole:
    it is what stops `varlen_bits`, `batch` and `row_off` from being live across
    the walk, and on gfx950 that liveness is what drives the backward kernel's
    scalar spilling. See `BwdDkDvKernelContext._init_q_head_invariants`.

    Exactly the mirror of the `pitch` select above -- `_HT` pitches by 1 and
    steps by `tokens`, `_TH` pitches by `H` and steps by 1 -- so the two are
    written the same way, from the same bits, and cannot drift apart.

    `num_head_q` is deliberately not a parameter: it does not appear in either
    step, and taking it would suggest the caller must keep it live.
    """
    tok = fx.Index(tokens)
    is_th = ((fx.Int32(varlen_bits) >> fx.Int32(16)) & fx.Int32(3)) != fx.Int32(0)
    return fx.Index(is_th.select(fx.Index(1), tok))


def bitcast_i32(value):
    return fx.Float32(value).bitcast(fx.Int32)


class MaskedAxis:
    """One axis of a tile whose index can run past the real extent.

    Out-of-range indices always need two things, and keeping them together is
    the point: an address that is *safe to issue*, and a way to *discard what
    it returns*. Issuing the access unconditionally and throwing the value away
    beats branching around it, but only once the address has been redirected
    somewhere legal -- element 0 of the axis, which always exists.

    **One class covers rows and columns**, because the apparent difference
    between them is not about the axes. An access reads `width` contiguous
    elements *along one axis*; for that axis the extent boundary can fall
    inside the access, so validity is per element. For every other axis the
    index is a single scalar and the whole access stands or falls together.
    In this kernel the vector runs along the column axis, which is why columns
    look "per element" and rows "whole" -- but that is a property of the
    access, not of the axis, and `valid(idx)` is exactly `mask(idx, 1)`.

    `active=False` compiles the masking away, for an axis whose extent is known
    to be a multiple of the access width.
    """

    __slots__ = ("extent", "active", "elem_dtype", "bitmask")

    def __init__(self, extent, active=True, elem_dtype=None, bitmask=False):
        self.extent = extent
        self.active = active
        # See `discard`. Off by default: it trades in-loop instructions for
        # live registers, and only the caller knows whether that is affordable.
        self.bitmask = bitmask
        # Only `discard` needs this, so the row axis leaves it unset. It is a
        # property of the tensor and every access along an axis shares it.
        #
        # `width` is deliberately *not* bound here. It belongs to the access,
        # not the axis, and the two accesses along the QK column axis agree
        # only by coincidence: the cooperative loads are `VEC_WIDTH` wide while
        # the Q preload is 8 wide because `load_global_v8f16` is tied to the
        # WMMA operand shape. Both are 8 today from unrelated definitions, and
        # binding one would make the Q preload silently follow `VEC_WIDTH`.
        self.elem_dtype = elem_dtype

    def _bound(self):
        """The extent as a traced value.

        Resolved lazily so the object can be built on the host when the extent
        is a `const_expr` int.
        """
        return fx.Index(self.extent) if isinstance(self.extent, int) else self.extent

    def valid(self, idx):
        """`fx.Boolean`: is this index inside the extent?

        A signed compare, on every axis. Signedness is a property of the
        *type* -- `_make_binop` reads each operand's class-level `signed` --
        so `fx.Int64` is what makes `<` emit `slt` here, and it is also why
        this cannot simply compare the `fx.Index` values it is handed:
        `fx.Index` is unsigned and would give `ult`. The answer agrees, since
        every index and extent is non-negative, but the two are not the same
        code and mixing them per axis was a difference with no reason behind
        it.

        Spelled with the operator rather than `arith.cmpi(CmpIPredicate.slt,
        ...)` because `cmpi` is stable while `CmpIPredicate` is not, and
        because the `fx.Boolean` this returns has a stable `select`. The
        `index_cast` that `fx.Int64` inserts folds away -- index is already
        64-bit here -- which was measured, not assumed.
        """
        return fx.Int64(idx) < fx.Int64(self._bound())

    def mask(self, idx, width):
        """i1 vector, element j set iff `idx + j` is inside the extent.

        Built from a loop-invariant index at every current caller, so it hoists
        out of the KV loop and costs one vector select per access inside it.
        """
        return Vec.from_elements(
            [self.valid(idx + fx.Index(j)) for j in range_constexpr(width)],
            fx.Boolean,
        )

    def safe(self, idx, addressed=None):
        """`addressed` if `idx` is inside the extent, else 0.

        `addressed` defaults to `idx`, which is the column case. Rows need the
        two to differ: the bound is on the *absolute* row, `start_q + ...`,
        while the address is built from the row's offset *within the tile*, so
        the tested and the redirected quantity are not the same value.
        """
        if addressed is None:
            addressed = idx
        if not self.active:
            return addressed
        return fx.Index(self.valid(idx).select(addressed, fx.Index(0)))

    def discard(self, vec, idx, width):
        """Zero the elements of `vec` whose index is past the extent.

        **AND with a precomputed bit mask, not a per-element select**, whenever
        the elements are 16-bit and pair up evenly into dwords.

        The obvious spelling -- `mask(idx, width).select(vec, zeros)` -- asks
        for a *per-element* choice over 16-bit lanes that live packed two to a
        VGPR, so the backend has to take them apart and put them back. Measured
        on a head_dim 32 build in the 64 tile, against an unpadded 64:

            v_cndmask_b32   +419
            v_lshrrev_b32   +208     <- pure repacking
            v_perm_b32      +192     <- pure repacking
            total          +1022 instructions (3585 against 2563)

        Two thirds of that is the register file being shuffled rather than any
        masking work. A dword whose two halves are `0xFFFF` or `0` expresses the
        same choice with one `v_and_b32` and no repacking, and the mask itself
        is loop-invariant -- it depends on the extent and a loop-invariant
        column base -- so it hoists out and costs nothing inside the loop.

        Exact for any extent, including one that splits a pair: the two halves
        of a dword carry independent masks.

        **Opt-in, because it is not free and not always a win.** The masks are
        loop-invariant, so they hoist -- and then stay *live* for the whole
        loop, `width/2` registers per masked access. Measured on the gfx950
        parity kernel, that is 16 registers in the 64-wide tile and 32 in the
        128-wide one, and the 128 build already sits at 238 of 256:

            tile    spills   TFLOP/s at the padded rungs
             64        0     +21% (head_dim 16/32/48)
            128       61     -43% (head_dim 80/96/112)

        So the caller decides. `bitmask=False` keeps the per-element select,
        which is also the path the fp8 and odd-width callers were verified on.
        """
        if not self.active:
            return vec
        if not self.bitmask or self.elem_dtype is None or self.elem_dtype.width != 16 or width % 2:
            zeros = Vec.filled(width, 0.0, self.elem_dtype)
            return self.mask(idx, width).select(Vec(vec), zeros).ir_value()

        keep_lo = fx.Int32(0x0000FFFF)
        keep_hi = fx.Int32(0xFFFF0000)
        zero_i32 = fx.Int32(0)
        dwords = [
            self.valid(idx + fx.Index(2 * d)).select(keep_lo, zero_i32)
            | self.valid(idx + fx.Index(2 * d + 1)).select(keep_hi, zero_i32)
            for d in range_constexpr(width // 2)
        ]
        raw = Vec(vec).bitcast(fx.Int32)
        return (raw & Vec.from_elements(dwords, fx.Int32)).bitcast(self.elem_dtype).ir_value()

    def gate(self, idx, addressed=None):
        """`(valid(idx), safe(idx, addressed))` -- the two halves together.

        An out-of-range index needs both, always, and returning them as a pair
        is what stops a caller taking the address and forgetting the flag.
        """
        return self.valid(idx), self.safe(idx, addressed)


def cond_load(cond, addr, default):
    """Load i32 from `addr` when `cond`, else `default`. The load is skipped.

    A real `scf.if`, not a select, and that is the point: the sequence-info
    pointers are **null** whenever their mode is off, so a select -- which
    evaluates both arms -- would fault. Inside the region the load is never
    issued; verified against a null pointer.

    Built as an explicit `IfOp` rather than Python's `if`, which is what lets
    this live in a module at all: the rewrite from `if` to `scf.if` is lexical
    per `@flyc.kernel` function, but an `IfOp` written out needs no rewriting.

    `addr` is computed by the caller and may be derived from a null pointer --
    address arithmetic touches no memory.
    """
    if_op = _scf.IfOp(fx.as_ir_value(cond), results_=[T.i32], has_else=True)
    with ir.InsertionPoint(if_op.then_block):
        _scf.YieldOp([fx.as_ir_value(fx.ptr_load(addr, fx.Int32))])
    with ir.InsertionPoint(if_op.else_block):
        _scf.YieldOp([fx.as_ir_value(default)])
    return fx.Int32(if_op.results[0])


def philox_offset_base(offset1, offset2):
    """`offset2 + *offset1` -- the Philox counter, split the way torch splits it.

    `at::cuda::PhiloxCudaState` carries the offset two ways. Outside a graph
    capture it is an immediate. Under capture the counter has to advance
    between replays, so it lives in device memory that the graph re-reads,
    while the per-call increment stays baked in as an immediate;
    `at::cuda::philox::unpack` is `*offset_.ptr + offset_intragraph_`. Passing
    one pre-summed scalar instead is what freezes a captured graph onto a
    single dropout mask, because the sum is done once at capture time and the
    replay never sees the counter move.

    AOTriton spells the pair `philox_offset1` (the pointer) and
    `philox_offset2` (the immediate), and this is the same ABI so that the two
    are drop-in for each other.

    A null `offset1` is the uncaptured case. The load is *skipped*, not
    selected away -- `select` evaluates both arms and would fault. Same
    explicit `IfOp` and the same reason as `cond_load`, at i64: written out
    rather than as a Python `if` because the rewrite to `scf.if` is lexical
    per `@flyc.kernel`, and this is module level.
    """
    return fx.Int64(offset2) + _load_u64_or_zero(offset1)


def philox_seed_value(seed_ptr):
    """`*seed_ptr`, or 0 when null. The seed side of the same graph story.

    A captured graph must see the seed move too, so torch keeps it in device
    memory exactly as it keeps the offset counter, and AOTriton takes it as
    `philox_seed_ptr` rather than a value. Splitting the offset but leaving the
    seed an immediate would give a replay a moving counter under a frozen key,
    which is not the same stream torch's own RNG would have produced.

    Null reads as 0 rather than faulting, matching AOTriton's `dropout_rng`.
    """
    return _load_u64_or_zero(seed_ptr)


@flyc.jit
def philox_report(seed_output, offset_output, seed, offset_base):
    """Write back the `(seed, offset)` this launch actually drew from.

    Only the backward can say why this exists. It has to regenerate the
    forward's stream, and under graph capture the effective offset is
    `*offset1 + offset2` -- a sum formed *on the device*, from a counter the
    host cannot read without synchronising. So the forward records what it
    used and the backward is handed that, instead of both sides trying to
    re-derive it and being wrong in different ways.

    One workgroup stores, not all of them: every workgroup computed the same
    two values, so the rest would be writing the same bytes to the same two
    addresses for no reason. `block_idx` raw rather than the flipped
    `q_tile_idx`, matching AOTriton's `program_id` guard -- which workgroup is
    designated does not matter, only that exactly one is.

    Either output may be null, which is how a caller says it does not want the
    value; both are skipped independently.
    """
    first = (
        (fx.Index(gpu.block_idx.x) == fx.Index(0))
        & (fx.Index(gpu.block_idx.y) == fx.Index(0))
        & (fx.Index(gpu.block_idx.z) == fx.Index(0))
    )
    if first:
        _store_u64_if_nonnull(seed_output, seed)
        _store_u64_if_nonnull(offset_output, offset_base)


def _load_u64_or_zero(ptr):
    """`*ptr` as an i64, or 0 when `ptr` is null.

    The load is *skipped*, not selected away -- `select` evaluates both arms
    and would fault on the null. Same explicit `IfOp` and the same reason as
    `cond_load`, at i64: written out rather than as a Python `if` because the
    rewrite to `scf.if` is lexical per `@flyc.kernel`, and this is module
    level.
    """
    nonnull = fx.Int64(fx.ptrtoint(ptr)) != fx.Int64(0)
    if_op = _scf.IfOp(fx.as_ir_value(nonnull), results_=[T.i64], has_else=True)
    with ir.InsertionPoint(if_op.then_block):
        _scf.YieldOp([fx.as_ir_value(fx.ptr_load(fx.recast_iter(_i64_global_ptr_ty(), ptr), fx.Int64))])
    with ir.InsertionPoint(if_op.else_block):
        _scf.YieldOp([fx.as_ir_value(fx.Int64(0))])
    return fx.Int64(if_op.results[0])


@flyc.jit
def _store_u64_if_nonnull(ptr, value):
    nonnull = fx.Int64(fx.ptrtoint(ptr)) != fx.Int64(0)
    if nonnull:
        fx.ptr_store(fx.Int64(value), fx.recast_iter(_i64_global_ptr_ty(), ptr))


def _i64_global_ptr_ty():
    """A u64 counter in global memory. Alignment 8, for the same reason as i32."""
    return fx.PointerType.get(
        elem_ty=fx.Int64.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=8,
    )


def _i32_global_ptr_ty():
    """An i32 pointer into global memory, alignment 4.

    Spelled out rather than `fx.recast_iter(fx.Int32, ptr)`, which inherits the
    source pointer's alignment: a kernel argument arrives as `u8` with
    alignment 1, and the shorthand then raises "alignment must be a positive
    multiple of element byte size (4), got 1". Same construction as
    `kernels/moe/moe_a8w4_mxscale_gfx1250.py` and
    `kernels/gemm/mxfp4_preshuffle.py`.

    A function, not a module constant: `PointerType.get` needs an MLIR context,
    and at import time there is none.
    """
    return fx.PointerType.get(
        elem_ty=fx.Int32.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=4,
    )


def seqinfo_addr(ptr, index):
    """`&ptr[index]` for an i32 sequence-info array. No memory is touched.

    A typed `!fly.ptr`, not an `!llvm.ptr`: that is what lets `cond_load` read
    it with the stable `fx.ptr_load` instead of a raw `llvm.LoadOp`.
    """
    return fx.recast_iter(_i32_global_ptr_ty(), ptr) + fx.Int64(index)


def decode_addressing(varlen_bits, bits_shift, max_seqlen, s0, s1, z):
    """One side of VarlenBits: where this workgroup's sequence lives.

    Returns `(seqlen, row_off, batch)` -- how long this sequence is, which row
    it starts at, and which batch index to use. Called once for Q and once for
    K. The axes are STACKED (bit 0),
    LENGTH (bits 2:1) and POSITION (bits 4:3).

    The LSE token pitch is *not* here, though it decodes from the same bits: it
    describes the logsumexp output rather than where Q or K live, and only the
    Q side needs it. See `lse_token_pitch`.

    Every load goes through `cond_load`, so the shape is flat -- fetch what
    each mode might need, then select. `s0[z]` serves both length modes and,
    under REUSE, the position too, which is why three loads cover five modes.
    """
    bits = fx.Int32(varlen_bits) >> fx.Int32(bits_shift)
    stacked = (bits & fx.Int32(1)) != fx.Int32(0)
    lenmode = (bits >> fx.Int32(1)) & fx.Int32(3)
    posmode = (bits >> fx.Int32(3)) & fx.Int32(3)

    cumulative = lenmode == fx.Int32(1)
    individual = lenmode == fx.Int32(2)
    reuse = posmode == fx.Int32(1)  # position already read as `cur`
    array = posmode == fx.Int32(2)  # position from its own array
    zero = fx.Int32(0)

    cur = cond_load(lenmode != zero, seqinfo_addr(s0, z), zero)
    nxt = cond_load(cumulative, seqinfo_addr(s0, z + fx.Int32(1)), zero)
    pos = cond_load(array, seqinfo_addr(s1, z), zero)

    seqlen = ssel(
        cumulative,
        nxt - cur,
        ssel(individual, cur, fx.Int32(max_seqlen)),
    )
    row_off = ssel(
        array,
        pos,
        ssel(
            reuse,
            cur,
            ssel(stacked, z * fx.Int32(max_seqlen), zero),
        ),
    )
    # **An empty sequence gets row 0 of the tensor, not row 0 of itself.**
    #
    # A zero-length sequence has no row to point at. Under a packed layout a
    # *trailing* one puts `row_off` at exactly `total_tokens`, so even row 0 of
    # that sequence is one past the last row of the tensor -- and every clamp
    # downstream is powerless, because `MaskedAxis.safe` redirects to element 0
    # *of the axis* and an empty axis has none. Three kernels read past the end
    # of Q, dO and K/V that way; two of the three were reproduced as HIP
    # memory-access faults whose address was bit-for-bit the end of the tensor.
    #
    # Redirecting the offset itself is what there always is a legal answer for:
    # row 0 of the whole tensor exists whenever any workgroup runs at all, since
    # a zero-row tensor makes the grid empty.
    #
    # Nothing else needs to change, because a workgroup on an empty sequence
    # already does no work: every loop is zero-trip (`decompose_causal_regions`
    # inverts its range when `alive` is false) and every store is guarded. The
    # prologue preloads were the one access that ran regardless, and their
    # results are discarded -- so making the address legal is the whole fix,
    # and skipping the transaction as well would buy nothing.
    #
    # Inert for every non-empty sequence, and for the padded layouts where
    # `row_off` is already zero.
    row_off = ssel(seqlen > zero, row_off, zero)
    batch = ssel(stacked, zero, z)
    return seqlen, row_off, batch


def lse_token_pitch(varlen_bits, bits_shift, max_seqlen, s0, s1, num_seqlens):
    """Row pitch of the logsumexp output, in tokens. Q side only.

    Batched layouts pad every row-group to `max_seqlen`; stacked ones run to
    the batch total, which lives in slot [N] of whichever array supplies
    positions -- the prefix-sum assumption, asserted host
    side.

    Derived from the bits rather than passed because the logsumexp tensor,
    alone among the tensors here, is always compact: its strides are a function
    of the bits, and passing them would be a second source of truth for one
    fact.
    """
    bits = fx.Int32(varlen_bits) >> fx.Int32(bits_shift)
    stacked = (bits & fx.Int32(1)) != fx.Int32(0)
    posmode = (bits >> fx.Int32(3)) & fx.Int32(3)
    reuse = posmode == fx.Int32(1)
    array = posmode == fx.Int32(2)
    zero = fx.Int32(0)

    total_s0 = cond_load(stacked & reuse, seqinfo_addr(s0, num_seqlens), zero)
    total_s1 = cond_load(stacked & array, seqinfo_addr(s1, num_seqlens), zero)
    return ssel(
        stacked,
        ssel(
            reuse,
            total_s0,
            ssel(array, total_s1, fx.Int32(num_seqlens) * fx.Int32(max_seqlen)),
        ),
        fx.Int32(max_seqlen),
    )


WINDOW_TOPLEFT = -2147483647  # 0x80000001


WINDOW_BOTRIGHT = -2147483646  # 0x80000002


def resolve_window(window_left, window_right, seqlen_q, seqlen_k):
    """`(window_left, window_right)` with the causal sentinels resolved.

    `Window_left` / `Window_right` may carry `WINDOW_TOPLEFT` or
    `WINDOW_BOTRIGHT` instead of a literal bound, and they are resolved
    against *this sequence's* lengths rather than on the host. That is the
    whole reason the sentinels exist: host resolution works only when there is
    one length to resolve against, and under varlen bottom-right needs
    `seqlen_k[z] - seqlen_q[z]`, which differs per sequence. Matches
    AOTriton's `parse_window`.

    Both sentinels give an unbounded left edge -- no row reaches further back
    than the start of its own sequence -- so they differ only in the right one.

    **Everything derived from a window stays i32.** Window bounds go negative;
    that is what a sentinel and a leading masked region are. `fx.Int32` is
    signed, so `<`/`>` emit `slt`/`sgt`, while `fx.Index` is unsigned and
    64-bit -- widening any of these even once makes the same comparison
    unsigned and a negative bound comes out enormous.
    """
    left = fx.Int32(window_left)
    right = fx.Int32(window_right)
    left_is_sentinel = (left == fx.Int32(WINDOW_TOPLEFT)) | (left == fx.Int32(WINDOW_BOTRIGHT))
    left = ssel(left_is_sentinel, seqlen_q, left)
    right = ssel(right == fx.Int32(WINDOW_TOPLEFT), fx.Int32(0), right)
    right = ssel(
        fx.Int32(window_right) == fx.Int32(WINDOW_BOTRIGHT),
        seqlen_k - seqlen_q,
        right,
    )
    return left, right


@dataclass(frozen=True, slots=True)
class CausalRegions:
    """The three contiguous KV block runs a causal/windowed Q block walks.

    Every field is a traced `fx.Int32`, not a Python int -- these are values
    the kernel computes per workgroup. Signed, because `right_col0` goes
    negative when the window admits no key at all; see
    `decompose_causal_regions`.

    A dataclass rather than a `NamedTuple` to match `Philox` next door, and
    because no caller destructures it positionally -- the kernel reads the
    seven fields by name. Being a Python object it is subject to the usual
    rule: do not let one live across a dynamic `if` (see "How to hand a helper
    object to kernel code"). The kernel unpacks it immediately, which is why
    it is safe here.
    """

    n_left: fx.Int32  # masked tiles before the full run
    n_full: fx.Int32  # tiles with no mask at all
    n_right: fx.Int32  # masked tiles after it
    left_col0: fx.Int32  # first KV column of each run
    full_col0: fx.Int32
    right_col0: fx.Int32
    masked_col0: fx.Int32  # first column of the masked run, whichever side


def decompose_causal_regions(start_q, q_len, k_len, window_left, window_right, block_m, block_n, alive):
    """Cut this Q block's visited KV range into `[masked][full][masked]`.

    **Three regions, not two.** A left window kills columns at the *start* of
    the range as well as the end, so masked tiles are a prefix as well as a
    suffix and tile 0 is not automatically live. A negative `window_left` is
    the sharpest case: it pushes the whole band right of the diagonal, so the
    leading masked run can span several tiles rather than clipping one. Do not
    carry the non-causal two-region intuition in here.

    The three are contiguous and non-overlapping *by construction*, because
    they are derived by cutting one visited range rather than intersected as
    three independent intervals. That collapses two of the three special cases
    (two of the usual special cases): a window narrower than a block leaves
    the full region empty, which is detected once and turns the other two into
    a single masked run, and an irregular `seqlen_q` needs no special handling
    because `q_hi` already bounds the rows.

    A column c is live for row i iff `i - window_left <= c <= i + window_right`,
    so over the block the live columns span
    `[start_q - window_left, (q_hi - 1) + window_right]`, and a tile is *fully*
    live iff every one of its columns is live for every row -- worst case the
    largest row on the left and the smallest on the right.

    `alive` is false for a workgroup whose rows all sit past `q_len`, which the
    varlen grid dispatches because its Q extent is sized from `Max_seqlen_q`.
    The kernel is one single-exit trace and cannot return out of those, so
    the visited range is *inverted* instead and every region
    count falls to zero. Dropping this makes those workgroups walk real tiles.

    Everything here is i32 and signed, deliberately: `left_col0` and friends
    go negative when the window admits no key at all. See `resolve_window`.
    """
    one = fx.Int32(1)
    zero = fx.Int32(0)
    bn = fx.Int32(block_n)

    q_start = fx.Int32(start_q)
    q_hi = smin(q_start + fx.Int32(block_m), q_len)
    q_last = q_hi - one

    # Blocks that exist at all, and the last block that is *whole*. Splitting
    # these: a ragged seqlen_k leaves a partial final
    # tile, which must be masked rather than counted as full.
    blk_last = sdiv_rd_pow2(k_len - one, block_n)
    blk_last_whole = sdiv_rd_pow2(k_len, block_n) - one

    # The visited range: outside it every column is dead for every row in this
    # Q block, so those tiles are not walked at all.
    v_lo = smax(sdiv_rd_pow2(q_start - window_left, block_n), zero)
    v_hi = smin(blk_last, sdiv_rd_pow2(q_last + window_right, block_n))
    v_hi = ssel(alive, v_hi, v_lo - one)

    # Rounded *up* on the left: a block is fully live only once its first
    # column clears the leftmost row's window. Rounding down would send a
    # partly-masked tile through the unmasked loop body -- invisible to a
    # tolerance test, not to the bitwise one.
    l_first_full = sdiv_rd_pow2(q_last - window_left + fx.Int32(block_n - 1), block_n)
    r_first_mask = sdiv_rd_pow2(q_start + window_right + one, block_n)

    fb_lo = smax(l_first_full, v_lo)
    fb_hi = smin(smin(r_first_mask - one, blk_last_whole), v_hi)
    fb_empty = fb_lo > fb_hi

    # Cut [v_lo, v_hi] at the full region. With no full region the whole range
    # becomes one masked run, the window narrower than a
    # block, falling out for free.
    lb_hi = ssel(fb_empty, v_hi, fb_lo - one)
    rb_lo = ssel(fb_empty, v_hi + one, fb_hi + one)

    n_left = smax(lb_hi - v_lo + one, zero)
    n_full = smax(fb_hi - fb_lo + one, zero)
    n_right = smax(v_hi - rb_lo + one, zero)

    left_col0 = v_lo * bn
    right_col0 = rb_lo * bn
    full_col0 = fb_lo * bn
    # First tile of the masked run, which is also what the full loop's last
    # prefetch must fetch: the two loops are adjacent only when the left run is
    # empty. Clamped, because with a window admitting no key at all every run
    # is empty and `rb_lo` sits below zero -- and this value still reaches the
    # prologue's address computation.
    masked_col0 = smax(ssel(n_left > zero, left_col0, right_col0), zero)
    return CausalRegions(n_left, n_full, n_right, left_col0, full_col0, right_col0, masked_col0)
