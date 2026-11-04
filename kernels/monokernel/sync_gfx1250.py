# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Memory scopes, wait counters and the tagged mailbox on gfx1250.

gfx12 encodes a memory instruction's cache policy as TH[2:0] | SCOPE[4:3]. The
gfx95x values the gfx950 mono kernels use mean something else here: 17 (SC0|SC1,
"system") is SCOPE_DEV | TH_NT, and 2 (NT) is TH_HT. A store's ``STOREcnt``
reaches zero once the write is at the level its own SCOPE names, and the WGP
cache is not coherent: every store another CTA reads, and every load of such
data, carries SCOPE_DEV (SYS across GPUs).

The combined ``s_waitcnt`` does not exist on gfx1250; each counter is waited on
by its own instruction, and it covers only the issuing wave.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from kernels.common import buffer_ops as bo

# --- gfx12 cache policy (CPol) -------------------------------------------------
TH_RT = 0
TH_NT = 1  # nontemporal: weights a layer streams once
TH_LU = 3  # last use (loads): the line may go once read -- a GL2-prefetched weight's demand load
SCOPE_CU = 0
SCOPE_SE = 8
SCOPE_DEV = 16  # visible to every CTA of this GPU
SCOPE_SYS = 24  # visible to peer GPUs and the host

POLL_MAX = 12  # mailbox pairs polled a batch (bounds the live registers)
DIAG_WORDS = 8  # a CTA's debug wait record
DIAG_POLLS = 1 << 22  # a debug wait gives up after this many re-polls (seconds: each is a memory round trip)


def traced(fn):
    """flydsl's AST rewriting for a helper, so its ``while`` / ``if`` over traced
    values lower to scf ops."""
    return ASTRewriter.transform(fn)


# --- wait counters -------------------------------------------------------------
def wait_load(n=0):
    rocdl.s_wait_loadcnt(n)


def wait_store(n=0):
    rocdl.s_wait_storecnt(n)


def wait_ds(n=0):
    rocdl.s_wait_dscnt(n)


def lds_barrier():
    """A CTA barrier ordering LDS only: this wave's LDS ops done, then the raw
    split barrier. ``gpu.barrier()``'s workgroup release first waits every
    global load and store in flight (a peer push's trip, a load issued ahead);
    use this where the stores before it are tagged mailbox pairs (their reader
    checks the tag, no order needed) or already drained (``wait_store``).

    The split-barrier intrinsics are memory-free to LLVM, so an empty asm with a
    memory clobber on each side keeps the compiler from moving LDS accesses
    across (no instruction, and no wait on the global counters)."""
    compiler_fence()
    wait_ds(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)
    compiler_fence()


def compiler_fence():
    llvm.InlineAsmOp(None, [], "", "~{memory}", has_side_effects=True)


def wave_lds_fence():
    """Order a wave's LDS accesses of a region only it uses (its stores before its
    later loads, its loads before its later stores): the wave's DS ops done and
    the compiler kept from moving LDS accesses across -- no CTA barrier, so the
    other waves (their loads in flight) never wait on this one."""
    compiler_fence()
    wait_ds(0)
    compiler_fence()


def wait_async(n=0):
    rocdl.s_wait_asynccnt(n)


def wait_tensor(n=0):
    rocdl.s_wait_tensorcnt(n)


def wait_km():
    """s_wait_kmcnt 0: scalar memory and message returns (no dialect op)."""
    llvm.InlineAsmOp(None, [], "s_wait_kmcnt 0", "", has_side_effects=True)


def realtime():
    """The fixed 100 MHz REALTIME counter, unlike shader cycles: comparable across a
    GPU's CTAs, not across GPUs (each its own constant offset, tens of ms:
    ``probe_pingpong`` 2026-10-29). gfx12 has no s_memrealtime."""
    op = llvm.InlineAsmOp(
        res=ir.IntegerType.get_signless(64),
        operands_=[],
        asm_string="s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\n\ts_wait_kmcnt 0",
        constraints="=&s",
        has_side_effects=True,
        is_align_stack=False,
    )
    return fx.Int64(op.res)


def rsrc(addr, nbytes=None):
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


# --- named regions -------------------------------------------------------------
class Space(Enum):
    SCRATCH = "scratch"  # this GPU's workspace: device scope
    PEER = "peer"  # a symmetric buffer another GPU reads or writes: system scope


SCOPE_OF = {Space.SCRATCH: SCOPE_DEV, Space.PEER: SCOPE_SYS}


@dataclass(frozen=True)
class Addr:
    """A mailbox region: its name, its space (which sets the scope) and address."""

    region: str
    space: Space
    value: object


def sreg(base, offset, region):
    return Addr(region, Space.SCRATCH, base + fx.Int64(offset))


def preg(base, offset, region):
    return Addr(region, Space.PEER, base + fx.Int64(offset))


def shift(a, delta):
    return Addr(a.region, a.space, a.value + delta)


def store_words(a, i, words):
    """Plain 32-bit words at word ``i`` of region ``a``, at the region's scope: data
    another CTA (or GPU) reads after a ``publish``."""
    bo.buffer_store(
        fx.Vector.from_elements([fx.Int32(w) for w in words], fx.Int32),
        rsrc(a.value),
        i,
        cache_modifier=SCOPE_OF[a.space],
    )


def load_words(a, i, n=1):
    """``n`` 32-bit words (1, 2 or 4) at word ``i`` of region ``a``, read past the
    WGP cache at the region's scope."""
    return bo.buffer_load(rsrc(a.value), i, vec_width=n, dtype=T.i32, cache_modifier=SCOPE_OF[a.space])


# --- spinning ------------------------------------------------------------------
@traced
def _spin(load_all, pending):
    """Re-issue ``load_all`` until ``pending`` clears. ``s_sleep`` keeps the loads in
    the loop and yields the SIMD; it is bounded, so it needs no ``s_wait_xcnt``
    (only sleep-forever does, SPG BUG #11)."""
    v = load_all()
    while pending(v):
        rocdl.s_sleep(1)
        v = load_all()
    return v


@traced
def _spin_bounded(load_all, pending, diag, region_id, index, tag):
    """``_spin`` giving up after DIAG_POLLS re-polls; the CTA's first give-up of the
    step is recorded in ``diag`` (DIAG_WORDS words a CTA: 1, region, pair, tag
    wanted, tag seen, CTA, clock, polls) and the kernel runs on. Once a CTA has
    given up, its later waits give up at once, so a broken launch ends in seconds.
    The bound counts polls, not REALTIME ticks: a clock deadline gave up at once
    on MI455X B0 (2026-10-21), so the clock is only recorded."""
    bid = fx.block_idx.x
    seen0 = fx.Int32(bo.buffer_load(rsrc(diag), bid * DIAG_WORDS, vec_width=1, dtype=T.i32, cache_modifier=SCOPE_SYS))
    v = load_all()
    polls = (seen0 != 0).select(fx.Int32(DIAG_POLLS), fx.Int32(0))
    while pending(v) & (polls < DIAG_POLLS):
        rocdl.s_sleep(1)
        v = load_all()
        polls = polls + 1
    if pending(v):
        seen = bo.buffer_load(rsrc(diag), bid * DIAG_WORDS, vec_width=1, dtype=T.i32, cache_modifier=SCOPE_SYS)
        if fx.Int32(seen) == 0:
            at = fx.Int32(realtime() & 0x7FFFFFFF)
            record = [1, region_id, index, tag, v[1], bid, at, polls]
            for h in range_constexpr(2):
                bo.buffer_store(
                    fx.Vector.from_elements([fx.Int32(w) for w in record[4 * h : 4 * h + 4]], fx.Int32),
                    rsrc(diag),
                    bid * DIAG_WORDS + 4 * h,
                    cache_modifier=SCOPE_SYS,
                )
    return v


def f32_pinned(x):
    """f32 ``x`` behind an opaque move: the value rounded to fp32 here. The gfx1250
    backend otherwise fuses an fma (or mul / add) with a following f32 -> bf16
    conversion into one v_fma_mix*_bf16, rounding the exact result straight to
    bf16 -- which differs from fp32-then-bf16 exactly on a bf16 tie."""
    bits = fx.Float32(x).bitcast(fx.Int32)
    out = fx.Int32(llvm.inline_asm(T.i32, [bits.ir_value()], "v_mov_b32 $0, $1", "=v,v", has_side_effects=False))
    return out.bitcast(fx.Float32)


def to_bf16(x):
    """``x`` rounded to fp32, then to bf16 (RNE): the reference's two roundings."""
    return f32_pinned(x).to(fx.BFloat16)


def _bf16_pair(a, b):
    return fx.Vector.from_elements([f32_pinned(a), f32_pinned(b)], fx.Float32).to(fx.BFloat16).bitcast(fx.Float32)[0]


class Mailbox:
    """Tagged-pair hand-off between CTAs (SCOPE_DEV) and GPUs (SCOPE_SYS).

    Every 32-bit value is stored next to the launch's tag in one 8- or 16-byte
    store; a consumer re-polls until every tag matches, so a hand-off costs one
    round trip and no flag. The tag is ``layer + 1`` and every mailbox is zeroed at
    a step's start, so a pair from an earlier step never carries a live tag.
    """

    def __init__(self, layer, diag=None, region_ids=None):
        self.tag = layer + 1
        self.diag = diag
        self.region_ids = region_ids or {}

    def _store(self, a, word_index, vals):
        bo.buffer_store(
            fx.Vector.from_elements(vals, fx.Int32),
            rsrc(a.value),
            word_index,
            cache_modifier=SCOPE_OF[a.space],
        )

    def put(self, a, i, v):
        bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
        self._store(a, i * 2, [bits, fx.Int32(self.tag)])

    def put_bf(self, a, i, vs):
        """Elements i .. i+len(vs) (2 or 4, i aligned) as packed bf16 pairs."""
        words = []
        for j in range_constexpr(len(vs) // 2):
            words += [_bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), fx.Int32(self.tag)]
        self._store(a, i, words)

    def put_words(self, a, i, words):
        """Raw 32-bit words (1 or 2) at pairs i.. as (word, tag); integer payloads
        come this way (an f32 path would not keep NaN bit patterns)."""
        vals = []
        for w in words:
            vals += [fx.Int32(w), fx.Int32(self.tag)]
        self._store(a, i * 2, vals)

    def poll(self, specs, batch=POLL_MAX):
        """Batched poll of ``specs`` = [(Addr, pair index, npairs in {1, 2})], each at
        its own space's scope. Returns the value words, a list per spec."""
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):
            return self.poll(specs[:batch], batch) + self.poll(specs[batch:], batch)
        tag = self.tag

        def load_all():
            words = []
            for a, i, n in specs:
                w = fx.Vector(
                    bo.buffer_load(
                        rsrc(a.value),
                        fx.Int32(i) * 2,
                        vec_width=2 * n,
                        dtype=T.i32,
                        cache_modifier=SCOPE_OF[a.space],
                    )
                )
                words += [w[e] for e in range(2 * n)]
            return fx.Vector.from_elements(words, fx.Int32)

        nw = sum(2 * n for _, _, n in specs)

        def pending(v):
            bad = v[1] != tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (v[e] != tag)
            return bad

        if const_expr(self.diag is None):
            v = _spin(load_all, pending)
        else:
            rid = self.region_ids.get(specs[0][0].region, -1)
            v = _spin_bounded(load_all, pending, self.diag, rid, specs[0][1], tag)
        outs, e = [], 0
        for _, _, n in specs:
            outs.append([v[e + 2 * q] for q in range(n)])
            e += 2 * n
        return outs


@traced
def publish(put, a, i, v, who):
    """Announce this CTA's plain stores (made with ``store_words``, at scope >= DEV)
    with the mailbox pair (``a``, i) := v, written by thread ``who``.

    Every wave drains its own stores (STOREcnt is per wave) before the barrier,
    so the flag cannot overtake data another wave wrote. No agent-scope fence:
    it lowers to a full L2 writeback."""
    wait_store(0)
    lds_barrier()
    if who:
        put(a, i, v)
