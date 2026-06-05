# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 FlyDSL Project Contributors

"""Pinned VGPR helpers for hand-allocated register layouts.

A ``PinnedRange`` is a closed interval ``[lo, hi]`` of VGPR numbers that
the kernel author guarantees no LLVM-allocated SSA value will land in
(typically ``v >= 64`` when paired with
``__attribute__((amdgpu_num_vgpr(64)))``).

Operations on pinned data go through inline-asm wrappers that name the
register numbers explicitly (``v[N:M]``) - LLVM cannot rename them.  SSA values flow in via ``v_mov`` trampolines from ``"v"`` -
constrained inputs; results flow out via ``v_mov`` trampolines into
``"=v"`` - constrained outputs.

Changing the pinned layout = edit the ``PinnedRange`` declarations at
the top of your kernel; every helper call below picks up the new
numbers.

Typical kernel-level usage::

    from flydsl.expr import pinned_vgpr as pv

    layout = pv.PinnedLayout()
    q_lds    = layout.declare(64, 71,   name="q_lds")
    q_vgpr   = layout.declare(72, 103,  name="q_vgpr")
    pv_v_aux = layout.declare(104, 111, name="pv_v_aux")
    kv       = layout.declare(112, 119, name="kv")
    p_comp   = layout.declare(120, 127, name="p_comp")
    p_mfma   = layout.declare(120, 123, name="p_mfma")  # overlay - merged
    oaccu    = layout.declare(128, 255, name="oaccu")

    q_tiles                  = q_vgpr.split(4)
    kv_top, kv_bot           = kv.split(4)
    kv_alt_top, kv_alt_bot   = pv_v_aux.split(4)
    oaccu_subs               = oaccu.split(4)

Then somewhere near the entry of the kernel::

    layout.emit_clobber()   # single union clobber over everything declared
"""

from typing import Iterable, List, Optional, Sequence, Tuple, Union

__all__ = [
    "PinnedRange",
    "PinnedLayout",
    "write_pinned",
    "read_pinned",
    "pinned_mfma_fp8_fp8",
    "pinned_mfma_fp8_fp8_init",
    "pinned_mfma_bf16",
    "pinned_mfma_bf16_init",
    "pinned_cvt_scalef32_pk_bf16_fp8",
    "pinned_cvt_pk_bf16_f32",
    "pinned_cvt_pk_bf16_f32_from_pinned",
    "pinned_cvt_pk_bf16_f32_from_pinned_to_ssa",
    "pinned_v_mul_f32",
    "pinned_v_mul_f32_pair",
    "pinned_softmax_exp_block",
    "pinned_inline_asm",
]


# ---------------------------------------------------------------------------
# Lazy MLIR/IR access (avoid circular import at module-load time)
# ---------------------------------------------------------------------------
def _ir_mod():
    from .._mlir import ir as _ir

    return _ir


def _llvm_mod():
    from .._mlir.dialects import llvm as _llvm

    return _llvm


def _to_ir(v):
    """Coerce a FlyDSL Numeric / Vector into a raw ``ir.Value``."""
    ir = _ir_mod()
    if isinstance(v, ir.Value):
        return v
    if hasattr(v, "ir_value"):
        return v.ir_value()
    # Last resort: try the arith._to_raw shim used elsewhere in flydsl.
    from .arith import _to_raw  # type: ignore

    return _to_raw(v)


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------
class PinnedRange:
    """Inclusive closed interval ``[lo, hi]`` of pinned VGPR numbers.

    A ``PinnedRange`` is a pure value object - it carries no SSA state.
    Helpers that operate on pinned data take SSA inputs ("v"-constrained)
    and route them through ``v_mov`` trampolines into the named pinned
    registers.
    """

    __slots__ = ("lo", "hi", "name")

    def __init__(self, lo: int, hi: int, *, name: str = ""):
        if not (0 <= lo <= hi <= 255):
            raise ValueError(f"PinnedRange out of bounds: [{lo}, {hi}]")
        self.lo = int(lo)
        self.hi = int(hi)
        self.name = name

    # ---- introspection ------------------------------------------------
    @property
    def size(self) -> int:
        return self.hi - self.lo + 1

    @property
    def asm_name(self) -> str:
        """Decimal-form asm operand string: ``v[N]`` or ``v[N:M]``.

        The LLVM AMDGPU assembler requires decimal vgpr numbers inside
        inline-asm operands (hex like ``v[0x48]`` is rejected as 'invalid
        operand').  HK's pinned kernels use the decimal form throughout.
        """
        if self.size == 1:
            return f"v[{self.lo}]"
        return f"v[{self.lo}:{self.hi}]"

    def vgpr_numbers(self) -> List[int]:
        return list(range(self.lo, self.hi + 1))

    # ---- slicing / splitting -----------------------------------------
    def __getitem__(self, idx) -> "PinnedRange":
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.size)
            if step != 1:
                raise ValueError("PinnedRange slicing only supports step=1")
            if start >= stop:
                raise ValueError(f"empty PinnedRange slice [{start}:{stop}]")
            return PinnedRange(
                self.lo + start,
                self.lo + stop - 1,
                name=f"{self.name}[{start}:{stop}]" if self.name else "",
            )
        i = int(idx)
        if not (0 <= i < self.size):
            raise IndexError(f"PinnedRange index {i} out of range [0, {self.size})")
        return PinnedRange(
            self.lo + i, self.lo + i,
            name=f"{self.name}[{i}]" if self.name else "",
        )

    def split(self, regs_per_tile: int) -> List["PinnedRange"]:
        """Partition into equal sub-ranges of ``regs_per_tile`` vgprs each.

        Mirrors ``hkdart::split_many_t<list, N>``: requires the range size
        to be a multiple of ``regs_per_tile``.
        """
        if regs_per_tile <= 0:
            raise ValueError("regs_per_tile must be positive")
        if self.size % regs_per_tile != 0:
            raise ValueError(
                f"PinnedRange {self!r} (size {self.size}) is not divisible "
                f"by tile size {regs_per_tile}"
            )
        n = self.size // regs_per_tile
        return [self[i * regs_per_tile:(i + 1) * regs_per_tile] for i in range(n)]

    def __repr__(self) -> str:
        tag = f" {self.name!r}" if self.name else ""
        return f"PinnedRange[{self.lo}..{self.hi}]{tag}"


# Module-level "currently active layout" -- helpers consult this for
# per-call clobber tokens.  Caller installs via ``PinnedLayout.set_active()``
# at kernel entry (once).  When ``_ACTIVE_LAYOUT is None`` (the default),
# per-call clobbers are empty -- the helpers stay backward-compatible.
_ACTIVE_LAYOUT: Optional["PinnedLayout"] = None


def _active_clobbers() -> Tuple[str, ...]:
    """Return the active layout's clobber tuple, or () if no layout is active."""
    return _ACTIVE_LAYOUT.clobbers if _ACTIVE_LAYOUT is not None else ()


class PinnedLayout:
    """Accumulates declared pinned ranges and emits the union clobber once.

    Overlapping declarations are allowed - the layout tracks the *union*
    of all declared intervals.  This lets aliasing views (``p_mfma``
    overlaying ``p_comp``) be declared as first-class ``PinnedRange``
    objects without double-counting in the clobber list.

    After declaring all ranges, call ``set_active()`` (and optionally
    ``emit_clobber()``) at kernel entry.  ``set_active`` makes the
    layout's clobber tuple visible to every subsequent pinned-asm helper
    call -- each call appends the full union clobber to its constraint
    list, forcing LLVM to treat every pinned vgpr as 'invalidated by
    every pinned-asm op'.  This is what prevents allocator leaks of SSA
    values into the pinned range across multiple asm ops (the
    function-wide ``emit_clobber()`` alone is NOT sufficient; see
    auto-memory ``[[llvm-emit-clobber-not-reservation]]``).
    """

    def __init__(self):
        # List of inclusive intervals, kept normalised (sorted, non-overlapping).
        self._union: List[Tuple[int, int]] = []
        # Diagnostic-only record of every declared range.
        self._declared: List[PinnedRange] = []
        self._clobber_emitted: bool = False
        # Cache: clobber-token tuple is computed on first .clobbers access.
        self._clobber_cache: Optional[Tuple[str, ...]] = None

    # ---- declaration --------------------------------------------------
    def declare(self, lo: int, hi: int, *, name: str = "") -> PinnedRange:
        """Register a new pinned range and merge it into the union."""
        r = PinnedRange(lo, hi, name=name)
        self._declared.append(r)
        self._merge_in(r.lo, r.hi)
        return r

    def _merge_in(self, lo: int, hi: int) -> None:
        merged: List[Tuple[int, int]] = []
        placed = False
        for a, b in self._union:
            if b + 1 < lo:
                merged.append((a, b))
            elif hi + 1 < a:
                if not placed:
                    merged.append((lo, hi))
                    placed = True
                merged.append((a, b))
            else:
                lo = min(lo, a)
                hi = max(hi, b)
        if not placed:
            merged.append((lo, hi))
        merged.sort()
        self._union = merged

    # ---- output -------------------------------------------------------
    @property
    def declared(self) -> Tuple[PinnedRange, ...]:
        return tuple(self._declared)

    @property
    def union_intervals(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(self._union)

    @property
    def clobbers(self) -> Tuple[str, ...]:
        """All ``~{vN}`` clobber tokens over the declared union.

        Cached after first access -- helpers call this once per asm op,
        so rebuilding the 192-string tuple every call would dominate
        Python overhead during trace.
        """
        if self._clobber_cache is None:
            out: List[str] = []
            for lo, hi in self._union:
                for v in range(lo, hi + 1):
                    out.append(f"~{{v{v}}}")
            self._clobber_cache = tuple(out)
        return self._clobber_cache

    def set_active(self) -> None:
        """Make this layout's clobber tuple available to all pinned helpers.

        Call once at kernel entry (after declaring all ranges).  Subsequent
        ``pinned_mfma_*``, ``pinned_cvt_*``, ``write_pinned`` etc. calls
        will append ``self.clobbers`` to their constraint string -- this
        is what actually keeps LLVM from allocating pinned vgprs as SSA
        scratch across pinned-asm ops.

        Idempotent if called multiple times with the same instance.
        """
        global _ACTIVE_LAYOUT
        _ACTIVE_LAYOUT = self

    @classmethod
    def clear_active(cls) -> None:
        """Reset the module-level active layout (for test isolation)."""
        global _ACTIVE_LAYOUT
        _ACTIVE_LAYOUT = None

    def emit_clobber(self) -> None:
        """Emit a no-op inline-asm clobber at kernel entry + install as active.

        - Emits one inline-asm op with empty asm string and the full union
          clobber list.  This marks every pinned vgpr as 'invalidated' at
          kernel entry so LLVM doesn't try to read prior values from them.
        - Also calls ``set_active()`` so subsequent pinned-asm helpers
          attach the same clobber tuple to their own asm ops.  Without
          this per-call attachment, LLVM can still allocate SSA values
          into pinned vgprs (the entry-time clobber alone is not enough --
          see ``[[llvm-emit-clobber-not-reservation]]``).
        """
        if not self._union:
            return
        if self._clobber_emitted:
            raise RuntimeError("PinnedLayout.emit_clobber() called twice")
        self._clobber_emitted = True
        _llvm_mod().inline_asm(
            None, [], "", ",".join(self.clobbers), has_side_effects=True,
        )
        self.set_active()


# ---------------------------------------------------------------------------
# Helpers shared by the wrappers
# ---------------------------------------------------------------------------
def _split_input_to_dwords(src, dst_size: int) -> List:
    """Decompose ``src`` into ``dst_size`` SSA i32 dwords.

    Accepts a single i32 (when ``dst_size==1``), a single i64
    (``dst_size==2``), a Vector (any size), or a tuple/list of i32s.
    """
    from .arith import _to_raw, ArithValue  # local import to break cycles
    from .typing import T

    ir = _ir_mod()

    if isinstance(src, (list, tuple)):
        if len(src) != dst_size:
            raise ValueError(
                f"src length {len(src)} does not match dst size {dst_size}"
            )
        return [_to_ir(v) for v in src]

    from .numeric import Int64  # local
    raw = _to_ir(src)
    if raw.type == T.i32:
        if dst_size != 1:
            raise ValueError(f"i32 src but dst_size={dst_size}")
        return [raw]
    if raw.type == T.i64:
        if dst_size != 2:
            raise ValueError(f"i64 src but dst_size={dst_size}")
        c32 = Int64(32)
        lo = ArithValue(raw).trunci(T.i32)
        hi = ArithValue(ArithValue(raw) >> c32).trunci(T.i32)
        return [_to_raw(lo), _to_raw(hi)]
    # Fall back: treat as Vector-like
    try:
        from .vector import Vector  # type: ignore

        vec = Vector(raw)
        elems = [_to_raw(ArithValue(vec[i]).bitcast(T.i32)) for i in range(dst_size)]
        return elems
    except Exception as e:
        raise TypeError(f"unsupported src for pinned write: {raw.type}: {e}")


def _emit_inline_asm(result_type, srcs: Sequence, asm: str,
                     out_constraints: Sequence[str],
                     in_constraints: Sequence[str],
                     *, has_side_effects: bool = True):
    """Emit a pinned-VGPR inline asm op.

    Appends the active layout's clobber tuple to the constraint list so
    LLVM treats every pinned vgpr as invalidated by this op -- prevents
    allocator-managed SSA values from landing in the pinned range
    across multiple asm ops.  See ``[[llvm-emit-clobber-not-reservation]]``.
    """
    parts = list(out_constraints) + list(in_constraints)
    return _llvm_mod().inline_asm(
        result_type, [_to_ir(s) for s in srcs], asm, ",".join(parts),
        has_side_effects=has_side_effects,
    )


# ---------------------------------------------------------------------------
# v_mov trampolines (atomic building blocks)
# ---------------------------------------------------------------------------
def write_pinned(dst: PinnedRange, src) -> None:
    """Move ``src`` into the pinned vgprs of ``dst``.

    Used at the start of pinned-DST asm blocks to land a fresh SSA value
    in the named registers.  ``src`` may be a single i32 (``dst.size==1``),
    a single i64 (``dst.size==2``), a Vector, or a sequence of i32s.
    """
    dws = _split_input_to_dwords(src, dst.size)
    lines = [f"v_mov_b32 v[{dst.lo + i}], ${i}" for i in range(dst.size)]
    in_constraints = ["v"] * dst.size
    _emit_inline_asm(
        None, dws, "\n".join(lines), [], in_constraints,
    )


def read_pinned(src: PinnedRange, *,
                as_i64: bool = False):
    """Move ``src``'s pinned vgprs into fresh SSA values.

    Returns an i32 (size==1), i64 (size==2 and ``as_i64=True``), or a
    tuple of i32s.  Used at the end of pinned-DST asm blocks to surface
    the result to LLVM SSA.
    """
    from .typing import T
    ir = _ir_mod()

    clobbers = list(_active_clobbers())
    if src.size == 1:
        return _llvm_mod().inline_asm(
            T.i32, [], f"v_mov_b32 $0, v[{src.lo}]",
            ",".join(["=v"] + clobbers),
            has_side_effects=True,
        )
    if src.size == 2 and as_i64:
        return _llvm_mod().inline_asm(
            T.i64, [], f"v_mov_b64 $0, v[{src.lo}:{src.hi}]",
            ",".join(["=v"] + clobbers),
            has_side_effects=True,
        )
    # Default: tuple of i32 dwords
    outs = []
    for i in range(src.size):
        outs.append(
            _llvm_mod().inline_asm(
                T.i32, [], f"v_mov_b32 $0, v[{src.lo + i}]",
                ",".join(["=v"] + clobbers),
                has_side_effects=True,
            )
        )
    return tuple(outs)


# ---------------------------------------------------------------------------
# Pinned mfma wrappers
# ---------------------------------------------------------------------------
def _split_src_to_dwords(src, size: int):
    """Decompose ``src`` (i32 / i64 / Vector / sequence) into ``size`` SSA i32 dwords."""
    from .arith import _to_raw, ArithValue  # local
    from .numeric import Int64
    from .typing import T
    from .vector import Vector

    if isinstance(src, (list, tuple)):
        if len(src) != size:
            raise ValueError(f"src has {len(src)} dwords; expected {size}")
        return [_to_ir(v) for v in src]
    raw = _to_ir(src)
    if size == 1:
        if raw.type == T.i32:
            return [raw]
        return [_to_raw(ArithValue(Vector(raw)[0]).bitcast(T.i32))]
    if size == 2 and raw.type == T.i64:
        c32 = Int64(32)
        lo = ArithValue(raw).trunci(T.i32)
        hi = ArithValue(ArithValue(raw) >> c32).trunci(T.i32)
        return [_to_raw(lo), _to_raw(hi)]
    # Treat as Vector.
    vec = Vector(raw)
    return [_to_raw(ArithValue(vec[i]).bitcast(T.i32)) for i in range(size)]


def _f32x4_from_pair(lo_i64, hi_i64):
    """Re-pack two SSA i64 into f32x4."""
    from .arith import _to_raw, ArithValue
    from .numeric import Int64, Float32
    from .typing import T
    from .vector import Vector
    c32 = Int64(32)
    lo_av = ArithValue(lo_i64)
    hi_av = ArithValue(hi_i64)
    words = [
        lo_av.trunci(T.i32),
        ArithValue(lo_av >> c32).trunci(T.i32),
        hi_av.trunci(T.i32),
        ArithValue(hi_av >> c32).trunci(T.i32),
    ]
    elems = [ArithValue(w).bitcast(T.f32) for w in words]
    return _to_raw(Vector.from_elements(elems, Float32))


def _emit_pinned_mfma(opcode: str, a: PinnedRange, b: PinnedRange, d: PinnedRange,
                      a_src, b_src, acc_src, *,
                      return_ssa: Optional[bool] = None):
    """Pinned-VGPR mfma accumulate form ``opcode d, a, b, d``.

    ``a`` and ``b`` may be 2-vgpr (fp8: i64 SSA) or 4-vgpr (bf16/f16:
    f32x4 / i32x4 SSA).  ``d`` is always 4 vgprs (f32x4 accumulator).

    ``a_src=None`` / ``b_src=None`` / ``acc_src=None`` skip the
    corresponding ``v_mov`` trampoline -- the pinned slot is read in
    place.  The caller is responsible for keeping that data alive
    across the call.

    ``return_ssa``:
      - ``True`` -> emit ``v_mov_b64 $0/$1`` trampolines to extract d
                    as an SSA f32x4; return that value.  Use when the
                    next consumer can't read d in place (e.g. SSA-side
                    softmax math).
      - ``False`` -> skip the output extraction; leave the result in
                     pinned d for a subsequent pinned-asm op to read.
                     Avoids the 2 ``v_mov_b64 =v`` trampolines that
                     consume scratch vgprs -- critical when v0..v63 is
                     saturated and LLVM would otherwise spill into the
                     pinned range and corrupt it.
      - ``None`` (default) -> auto: True if ``acc_src is not None``,
                              else False.  Preserves prior behaviour.
    """
    from .typing import T

    if d.size != 4:
        raise ValueError(f"pinned mfma dst must be size 4; got {d.size}")
    if return_ssa is None:
        return_ssa = acc_src is not None

    a_dws = _split_src_to_dwords(a_src, a.size) if a_src is not None else []
    b_dws = _split_src_to_dwords(b_src, b.size) if b_src is not None else []
    acc_dws = _split_src_to_dwords(acc_src, 4) if acc_src is not None else []

    lines: List[str] = []
    operand_idx = 0
    out_constraints: List[str] = []
    if return_ssa:
        out_constraints = ["=v", "=v"]
        operand_idx = 2
    if a_src is not None:
        for i in range(a.size):
            lines.append(f"v_mov_b32 v[{a.lo + i}], ${operand_idx + i}")
        operand_idx += a.size
    if b_src is not None:
        for i in range(b.size):
            lines.append(f"v_mov_b32 v[{b.lo + i}], ${operand_idx + i}")
        operand_idx += b.size
    if acc_src is not None:
        for i in range(4):
            lines.append(f"v_mov_b32 v[{d.lo + i}], ${operand_idx + i}")
    lines.append(f"{opcode} {d.asm_name}, {a.asm_name}, {b.asm_name}, {d.asm_name}")
    if return_ssa:
        lines.append(f"v_mov_b64 $0, v[{d.lo + 0}:{d.lo + 1}]")
        lines.append(f"v_mov_b64 $1, v[{d.lo + 2}:{d.lo + 3}]")

    n_a = a.size if a_src is not None else 0
    n_b = b.size if b_src is not None else 0
    n_acc = 4 if acc_src is not None else 0
    constraints = out_constraints + ["v"] * (n_a + n_b + n_acc)
    ir = _ir_mod()
    if return_ssa:
        pair = _llvm_mod().inline_asm(
            ir.Type.parse("!llvm.struct<(i64, i64)>"),
            [*a_dws, *b_dws, *acc_dws],
            "\n".join(lines),
            ",".join(constraints),
            has_side_effects=True,
        )
        lo = _llvm_mod().extractvalue(T.i64, pair, [0])
        hi = _llvm_mod().extractvalue(T.i64, pair, [1])
        return _f32x4_from_pair(lo, hi)
    else:
        _llvm_mod().inline_asm(
            None, [*a_dws, *b_dws, *acc_dws],
            "\n".join(lines), ",".join(constraints),
            has_side_effects=True,
        )
        return None


def _emit_pinned_mfma_init(opcode: str, a: PinnedRange, b: PinnedRange, d: PinnedRange,
                           a_src, b_src, *, return_ssa: bool = True):
    """3-arg init form ``opcode d, a, b, d`` with d pre-zeroed.

    ``a_src=None`` / ``b_src=None``: reference pinned A/B in place.
    ``return_ssa=False``: leave the result in pinned d, return None
    (useful for iterated accumulators where d is read back via another
    pinned-asm op).
    """
    from .typing import T

    if d.size != 4:
        raise ValueError(f"pinned mfma dst must be size 4; got {d.size}")
    a_dws = _split_src_to_dwords(a_src, a.size) if a_src is not None else []
    b_dws = _split_src_to_dwords(b_src, b.size) if b_src is not None else []

    lines: List[str] = []
    operand_idx = 2 if return_ssa else 0
    out_constraints = ["=v", "=v"] if return_ssa else []
    if a_src is not None:
        for i in range(a.size):
            lines.append(f"v_mov_b32 v[{a.lo + i}], ${operand_idx + i}")
        operand_idx += a.size
    if b_src is not None:
        for i in range(b.size):
            lines.append(f"v_mov_b32 v[{b.lo + i}], ${operand_idx + i}")
        operand_idx += b.size
    # Use the 3-arg implicit-zero-accumulator encoding (literal `0` in 4th
    # operand position).  Avoids the SIMD-write -> MFMA-read RAW hazard that
    # `v_mov_b32 v[d], 0; v_mfma ... v[d:d+3], v[d:d+3]` exposes -- the v_mov
    # writes through the SIMD pipeline and may not retire before MFMA reads C.
    # HK uses this same encoding for init mfmas (see store_kv_tile_step and
    # _do_qk_gemm's first pair).
    lines.append(f"{opcode} {d.asm_name}, {a.asm_name}, {b.asm_name}, 0")
    if return_ssa:
        lines.append(f"v_mov_b64 $0, v[{d.lo + 0}:{d.lo + 1}]")
        lines.append(f"v_mov_b64 $1, v[{d.lo + 2}:{d.lo + 3}]")

    n_a = a.size if a_src is not None else 0
    n_b = b.size if b_src is not None else 0
    constraints = out_constraints + ["v"] * (n_a + n_b)
    ir = _ir_mod()
    if return_ssa:
        pair = _llvm_mod().inline_asm(
            ir.Type.parse("!llvm.struct<(i64, i64)>"),
            [*a_dws, *b_dws],
            "\n".join(lines), ",".join(constraints),
            has_side_effects=True,
        )
        lo = _llvm_mod().extractvalue(T.i64, pair, [0])
        hi = _llvm_mod().extractvalue(T.i64, pair, [1])
        return _f32x4_from_pair(lo, hi)
    else:
        _llvm_mod().inline_asm(
            None, [*a_dws, *b_dws],
            "\n".join(lines), ",".join(constraints),
            has_side_effects=True,
        )
        return None


def pinned_mfma_fp8_fp8(a, b, d, a_src, b_src, acc_src):
    """``v_mfma_f32_16x16x32_fp8_fp8 d, a, b, d``.

    A/B operands are size 2 vgprs each (i64 fp8 fragments).  D is size 4
    (f32x4 accumulator).  Returns SSA f32x4 of the new accumulator.
    """
    if a.size != 2 or b.size != 2:
        raise ValueError(f"fp8/fp8 mfma needs A/B size 2; got {a.size}/{b.size}")
    return _emit_pinned_mfma("v_mfma_f32_16x16x32_fp8_fp8",
                             a, b, d, a_src, b_src, acc_src)


def pinned_mfma_fp8_fp8_init(a, b, d, a_src, b_src):
    """3-arg init form of fp8/fp8 mfma (C=0).  For ``kIsFirstIter``."""
    if a.size != 2 or b.size != 2:
        raise ValueError(f"fp8/fp8 mfma needs A/B size 2; got {a.size}/{b.size}")
    return _emit_pinned_mfma_init("v_mfma_f32_16x16x32_fp8_fp8",
                                  a, b, d, a_src, b_src)


def pinned_mfma_bf16(a, b, d, a_src, b_src, acc_src, *,
                     return_ssa: Optional[bool] = None):
    """``v_mfma_f32_16x16x32_bf16 d, a, b, d`` (QK + PV paths in v40).

    A/B operands are size 4 vgprs each (bf16 fragments, 8 elems/lane).
    D is size 4 (f32x4 accumulator).  See ``_emit_pinned_mfma`` for the
    semantics of ``a_src/b_src/acc_src=None`` and ``return_ssa``.
    """
    if a.size != 4 or b.size != 4:
        raise ValueError(f"bf16 mfma needs A/B size 4; got {a.size}/{b.size}")
    return _emit_pinned_mfma("v_mfma_f32_16x16x32_bf16",
                             a, b, d, a_src, b_src, acc_src,
                             return_ssa=return_ssa)


def pinned_mfma_bf16_init(a, b, d, a_src, b_src, *, return_ssa: bool = True):
    if a.size != 4 or b.size != 4:
        raise ValueError(f"bf16 mfma needs A/B size 4; got {a.size}/{b.size}")
    return _emit_pinned_mfma_init("v_mfma_f32_16x16x32_bf16",
                                  a, b, d, a_src, b_src,
                                  return_ssa=return_ssa)


# ---------------------------------------------------------------------------
# Pinned cvt wrappers
# ---------------------------------------------------------------------------
def pinned_cvt_scalef32_pk_bf16_fp8(dst: PinnedRange, src_dw, scale_dw, *, opsel: bool):
    """``v_cvt_scalef32_pk_bf16_fp8`` with destination pinned to ``dst``.

    Reads one of the two fp8 pairs (low/high half) of ``src_dw``,
    multiplies by ``scale_dw`` (f32 form of the E8M0 scale) and writes 2
    bf16 packed into ``dst`` (one dword).

    Per spec Ch. 13.1: the template-int form (``v_cvt... v[N]``) silently
    corrupts; the pinned-DST form encodes the register number in the asm
    string directly.  An ``s_nop 0`` is appended to close the
    VALU->MFMA RAW hazard window across the opaque inline-asm boundary.
    """
    if dst.size != 1:
        raise ValueError(f"pinned cvt dst must be size 1; got {dst.size}")
    op_sel_str = "op_sel:[1,0,0]" if opsel else ""
    asm = "\n".join([
        f"v_cvt_scalef32_pk_bf16_fp8 {dst.asm_name}, $0, $1 {op_sel_str}",
        "s_nop 0",
    ])
    constraints = ["v", "v"]
    _llvm_mod().inline_asm(
        None, [_to_ir(src_dw), _to_ir(scale_dw)], asm,
        ",".join(constraints), has_side_effects=True,
    )


def pinned_cvt_pk_bf16_f32_from_pinned_to_ssa(src_pair: PinnedRange):
    """``v_cvt_pk_bf16_f32 dst, v[SRC], v[SRC+1]`` reading 2 pinned f32 dwords
    and returning the packed bf16x2 as an SSA i32 value (LLVM picks the
    destination register).

    Use this when the dst can be allocator-managed scratch (e.g. OMgr V3
    bf16 packs that get immediately ds_write'd) rather than a fixed pinned
    slot.  Avoids the "pack-into-oaccu" pitfall where consecutive packs
    clobber each other's sources.
    """
    if src_pair.size != 2:
        raise ValueError(f"src must be size 2; got {src_pair.size}")
    from .typing import T
    return _llvm_mod().inline_asm(
        T.i32, [],
        f"v_cvt_pk_bf16_f32 $0, v[{src_pair.lo}], v[{src_pair.lo + 1}]",
        ",".join(["=v"]), has_side_effects=True,
    )


def pinned_cvt_pk_bf16_f32_from_pinned(dst: PinnedRange, src_pair: PinnedRange):
    """``v_cvt_pk_bf16_f32 v[DST], v[SRC], v[SRC+1]``: read 2 pinned f32
    dwords, pack to bf16x2 in 1 pinned dword.

    ``src_pair`` must be a 2-vgpr pinned range holding the f32 pair.
    Mirrors HK's ``pack_2f32_to_bf16_pair_pinned<DST, SRC>()``.  Used by
    OMgr V3 to pack oaccu (pinned f32 v128..v255) into bf16 dwords for
    the LDS bounce write, without ever materialising an SSA carrier
    (oaccu is too large to thread through SSA across all 32 sub-tiles).
    """
    if dst.size != 1 or src_pair.size != 2:
        raise ValueError(
            f"pinned cvt from pinned needs dst.size=1, src.size=2; "
            f"got dst={dst.size}, src={src_pair.size}"
        )
    asm = f"v_cvt_pk_bf16_f32 {dst.asm_name}, v[{src_pair.lo}], v[{src_pair.lo + 1}]"
    _llvm_mod().inline_asm(None, [], asm, ",".join(list(_active_clobbers())), has_side_effects=True)


def pinned_cvt_pk_bf16_f32(dst: PinnedRange, src_lo, src_hi):
    """``v_cvt_pk_bf16_f32 dst, src_lo, src_hi`` writing into pinned ``dst``.

    The caller MUST emit overlay packs in low-to-high order: when
    ``p_mfma`` (v120..v123) overlays ``p_comp`` (v120..v127), the cvt
    reads its src f32s BEFORE writing dst, so writing dst[0] first is
    safe (src[0,1] are still fp32 in v120..v121 at that moment); writing
    dst[1] reads src[2,3] from v122..v123 which is still fp32 - and so
    on.  Out-of-order writes would clobber not-yet-read sources.
    """
    if dst.size != 1:
        raise ValueError(f"pinned cvt dst must be size 1; got {dst.size}")
    asm = f"v_cvt_pk_bf16_f32 {dst.asm_name}, $0, $1"
    constraints = ["v", "v"]
    _llvm_mod().inline_asm(
        None, [_to_ir(src_lo), _to_ir(src_hi)], asm,
        ",".join(constraints), has_side_effects=True,
    )


# ---------------------------------------------------------------------------
# Pinned ALU
# ---------------------------------------------------------------------------
def pinned_v_mul_f32(dst: PinnedRange, factor_src):
    """``v_mul_f32_e32 v[dst], factor, v[dst]`` -- in-place per-vgpr multiply.

    Iterates over every vgpr in ``dst`` (typically a wide pinned range
    like oaccu).  Each vgpr gets one ``v_mul_f32`` referencing the same
    SSA scalar ``factor_src``.

    Used by OMgr V3 to normalize oaccu by ``1/row_sum_e`` before the
    epilogue's bf16 pack.  HK calls this ``hk::mul_vgpr(oaccu, oaccu,
    reci_row_sum_e)`` over the full 128-vgpr tile -- 128 instructions.
    """
    asm_lines = [
        f"v_mul_f32_e32 v[{dst.lo + i}], $0, v[{dst.lo + i}]"
        for i in range(dst.size)
    ]
    # Per-call clobbers force LLVM to order this mul AFTER any prior
    # pinned-asm that wrote into the active layout (e.g. PV mfmas writing
    # oaccu in place).  Without these, has_side_effects alone is not
    # enough -- LLVM treats two side-effecting asm ops with disjoint
    # operand/clobber lists as reorderable.
    _llvm_mod().inline_asm(
        None, [_to_ir(factor_src)], "\n".join(asm_lines),
        ",".join(["v"]), has_side_effects=True,
    )


def pinned_v_mul_f32_pair(dst: PinnedRange, factor_pair_src):
    """``v_pk_mul_f32 dst, dst, factor_pair`` - in-place on a 2-vgpr pair.

    ``factor_pair_src`` must be an SSA i64 (or 2-vgpr Vector) holding
    ``{factor, factor}`` as a packed-fp32 pair.  Pack via e.g.
    ``ArithValue(bits).extui(T.i64) | (ArithValue(bits).extui(T.i64) << 32)``.

    Notation note: ``v_pk_mul_f32 d, d, [s, s]`` (replicating a single
    vgpr) needs *two consecutive* vgprs in the operand list, so an SSA
    value must be aligned as an i64 pair before reaching this helper -
    the assembler rejects ``[v10, v10]`` (same vgpr twice) with
    'registers in a list must have consecutive indices'.
    """
    if dst.size != 2:
        raise ValueError(f"pinned v_pk_mul_f32 needs size 2; got {dst.size}")
    asm = f"v_pk_mul_f32 {dst.asm_name}, {dst.asm_name}, $0"
    _llvm_mod().inline_asm(
        None, [_to_ir(factor_pair_src)], asm,
        ",".join(["v"]), has_side_effects=True,
    )


def pinned_softmax_exp_block(p_comp: PinnedRange, neg_max_pair_src,
                             log2e_pk_src):
    """Spec Ch. 9.5 fused asm block (4 v_pk_add + 4 v_pk_mul + 8 v_exp_f32).

    Operates in-place on ``p_comp`` (8 fp32 / lane).
    ``neg_max_pair_src`` is i64 packing ``{-m, -m}``; ``log2e_pk_src``
    is i64 packing ``{log2e, log2e}``.  Both must be packed pairs for the
    same reason ``pinned_v_mul_f32_pair`` requires it -- v_pk_add /
    v_pk_mul vgpr-pair-list operands must be consecutive.
    """
    if p_comp.size != 8:
        raise ValueError(f"softmax_exp_block needs size 8; got {p_comp.size}")
    pairs = [p_comp[i:i + 2] for i in range(0, 8, 2)]
    lines = []
    for pr in pairs:
        lines.append(f"v_pk_add_f32 {pr.asm_name}, {pr.asm_name}, $0")
    for pr in pairs:
        lines.append(f"v_pk_mul_f32 {pr.asm_name}, {pr.asm_name}, $1")
    for i in range(8):
        v = p_comp.lo + i
        lines.append(f"v_exp_f32 v[{v}], v[{v}]")
    asm = "\n".join(lines)
    _llvm_mod().inline_asm(
        None, [_to_ir(neg_max_pair_src), _to_ir(log2e_pk_src)], asm,
        ",".join(["v", "v"]), has_side_effects=True,
    )


# ---------------------------------------------------------------------------
# Generic escape hatch
# ---------------------------------------------------------------------------
def pinned_inline_asm(asm: str, *,
                      result_type=None, srcs: Sequence = (),
                      in_constraints: Sequence[str] = (),
                      out_constraints: Sequence[str] = (),
                      has_side_effects: bool = True):
    """One-off pinned-VGPR inline asm.  ``asm`` may reference pinned
    registers by their ``PinnedRange.asm_name`` and SSA inputs by ``$N``.

    Use sparingly: if a pattern recurs, add a dedicated helper above.
    """
    parts = list(out_constraints) + list(in_constraints)
    return _llvm_mod().inline_asm(
        result_type, [_to_ir(s) for s in srcs], asm, ",".join(parts),
        has_side_effects=has_side_effects,
    )
