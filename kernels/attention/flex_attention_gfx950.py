# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Independent flex-attention forward on the FlyDSL layout API (gfx950).

This is an attention kernel written on the FlyDSL layout API
(``fx.make_tiled_mma`` / ``make_fragment_{A,B,C}`` / ``fx.copy`` /
swizzled LDS views).

One workgroup computes ``num_groups`` independent ``[BLOCK_M, D]`` query tiles:
load Q resident, loop over KV ``[BLOCK_N, D]`` tiles doing GEMM1 (S = Q@K^T),
online softmax, the C->B bridge (scores packed as MFMA B operand), then
GEMM2 (O += P@V with V=A, P=B); epilogue normalizes O by the row sum and
stores it. The KV loop is the n64 4-cluster deferred-softmax pipeline, or the
8-cluster schedule when ``long_seq_8c`` is on.

Target arch: gfx950 (CDNA4). Uses the cdna4 LDS transpose-read atom and the
gfx950 LDS swizzles; it is NOT expected to run on gfx942.

Score and mask modifiers
------------------------
Pass PyTorch FlexAttention-style callables into
``flydsl_flex_attention_layout`` / ``flydsl_flex_attention_layout_paged``:

    mask_mod(b, h, q_idx, kv_idx) -> bool   # True = keep this Q/KV pair
    score_mod(score, b, h, q_idx, kv_idx) -> score

``None`` is dense (all positions visible, logits unmodified). The same
callables can be used with ``torch.nn.attention.flex_attention``.

They are traced into the kernel, so they must be Python functions with no
closures and no global/attribute loads (no ``torch.tanh``, no tensor
lookups). Bind extra scalars as positional defaults:

    def causal(b, h, q_idx, kv_idx, offset=0):
        return kv_idx <= q_idx + offset

    def alibi(score, b, h, q_idx, kv_idx, slope=0.125):
        return score + slope * (kv_idx - q_idx)

    def quadratic(score, b, h, q_idx, kv_idx, inv_var=1e-4):
        dist = q_idx - kv_idx
        return score - dist * dist * inv_var

Named functions are rewritten to a constexpr-friendly lambda clone; the
math is unchanged. Combine both kwargs when a mask and a score apply
together (document + ALiBi, causal + quadratic, ...).

Host inspection does not replace the formulas. It only classifies layout
or an equivalent score lowering:

- ``mask_mod`` matching ``kv <= q + offset`` (offset 0 or Skv-Sq; paged
  forces 0) enables packed vec2 masking, reverse Q grid, skip-dead-Q, and
  a 4c diagonal tail under 8c. A sliding-window band or prefix-LM union
  keeps those structure opts with their own kv_range. Anything else stays
  generic: the callable still runs per element, plus a host-inferred
  per-workgroup KV envelope so fully masked tiles can be skipped.
- ``score_mod`` matching ``score + slope*(kv-q)`` is applied as a packed
  log2-space add (no natural-log round trip). Other scores, including
  quadratic bias, run the callable after converting pipeline logits to
  natural log and back. Disable inference with ``infer_score_mod=False``,
  ``score_mod.flex_infer = False``, or
  ``score_mod.flex_score_kind = "exact"``.
"""

import dis
import math
import struct
import types
from functools import lru_cache
from typing import Callable, Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ReplaceIfWithDispatch
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.utils.arith import _to_raw as as_mlir_value
from flydsl.runtime.device import get_rocm_arch

scf_if_dispatch = ReplaceIfWithDispatch.scf_if_dispatch

if not hasattr(fx, "max"):
    from flydsl._mlir.dialects import arith as _arith_mlir

    def _fx_max(a, b):
        if hasattr(a, "maximumf"):
            return a.maximumf(b)
        return type(a)(_arith_mlir.maxsi(a.ir_value(), b.ir_value()))

    def _fx_min(a, b):
        if hasattr(a, "minimumf"):
            return a.minimumf(b)
        return type(a)(_arith_mlir.minsi(a.ir_value(), b.ir_value()))

    fx.max = _fx_max
    fx.min = _fx_min


def _i32_min(a, b):
    from flydsl._mlir.dialects import arith as _am

    def _v(x):
        return x.ir_value() if hasattr(x, "ir_value") else x

    return fx.Int32(_am.minsi(_v(a), _v(b)))


def _i32_max(a, b):
    from flydsl._mlir.dialects import arith as _am

    def _v(x):
        return x.ir_value() if hasattr(x, "ir_value") else x

    return fx.Int32(_am.maxsi(_v(a), _v(b)))


def _f32_max(a, b):
    if hasattr(a, "maximumf"):
        return a.maximumf(b)
    return fx.Float32(a).maximumf(fx.Float32(b))


def pipeline_stagger_enabled(*, depth: int, num_groups: int, m_waves: int) -> bool:
    return depth >= 2 and num_groups >= 2 and m_waves >= 2


class _InfraContext:
    stagger_i32: object = None


try:
    from flydsl.expr.rocdl.universal import make_buffer_ptr as _make_buffer_ptr
except ImportError:
    from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
    from flydsl.expr import buffer_ops

    def _make_buffer_ptr(ptr, num_records_bytes=None):
        if num_records_bytes is None:
            num_records_bytes = fx.Int64(0xFFFFFFFF)
        elif not isinstance(num_records_bytes, fx.Int64):
            num_records_bytes = fx.Int64(num_records_bytes)
        buf_ptr_ty = fx.PointerType.get(
            elem_ty=ptr.element_type.ir_type,
            address_space=TargetAddressSpace.BufferDesc,
            alignment=ptr.alignment,
        )
        return fx.make_ptr(
            buf_ptr_ty,
            [
                ptr,
                fx.Int16(0).ir_value(),
                num_records_bytes.ir_value(),
                fx.Int32(buffer_ops._get_buffer_flags()).ir_value(),
            ],
        )


GFX950_WAVE_SIZE = 64
GFX950_DMA_BYTES = 16
_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3


def _waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7)."""
    val = (n & _VMCNT_LO_MASK) | _LGKMCNT_EXPCNT_BASE | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
    rocdl.s_waitcnt(val)


def _read_exec_i64():
    """Read the current wave exec mask, matching Clang's builtin lowering."""
    true_i1 = fx.Boolean(True).ir_value()
    return rocdl.ballot(T.i64, true_i1)


def _bitcast_i32(value):
    return as_mlir_value(fx.Float32(value).bitcast(fx.Int32).ir_value())


def _bitcast_f32(value):
    return as_mlir_value(fx.Int32(value).bitcast(fx.Float32).ir_value())


def _attn_mask_vec2_imm(rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32):
    """Pair mask: 2 compares followed by 2 cndmasks."""
    asm_str = (
        f"v_cmp_lt_i32_e64 $0, $6, {int(thr_x)}\n\t"
        f"v_cmp_lt_i32_e64 $1, $6, {int(thr_y)}\n\t"
        "v_cndmask_b32_e64 $2, $4, $7, $0\n\t"
        "v_cndmask_b32_e64 $3, $5, $7, $1"
    )
    ret_struct_ty = ir.Type.parse("!llvm.struct<(i64, i64, i32, i32)>")
    ret = llvm.inline_asm(
        ret_struct_ty,
        [
            as_mlir_value(x_ref_i32),
            as_mlir_value(y_ref_i32),
            as_mlir_value(rel_i32),
            as_mlir_value(neg_inf_i32),
        ],
        asm_str,
        "=s,=s,=v,=v,2,3,v,v,~{vcc}",
        has_side_effects=True,
    )
    return llvm.extractvalue(T.i32, ret, [2]), llvm.extractvalue(T.i32, ret, [3])


def _swap_halves(dw):
    pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(pair_i32_ty, as_mlir_value(dw), as_mlir_value(dw), False, False)
    lo_res = llvm.extractvalue(T.i32, swapped, [0])
    hi_res = llvm.extractvalue(T.i32, swapped, [1])
    return lo_res, hi_res


def _fused_o_128_dwords(lane_div_32, d0_a, d1_a, d0_b, d1_b):
    is_hi_half = lane_div_32 != fx.Index(0)
    y0_a_lo, y0_a_hi = _swap_halves(d0_a)
    y1_a_lo, y1_a_hi = _swap_halves(d1_a)
    y0_b_lo, y0_b_hi = _swap_halves(d0_b)
    y1_b_lo, y1_b_hi = _swap_halves(d1_b)
    y0_a, y1_a = is_hi_half.select(y0_a_lo, y0_a_hi), is_hi_half.select(y1_a_lo, y1_a_hi)
    y0_b, y1_b = is_hi_half.select(y0_b_lo, y0_b_hi), is_hi_half.select(y1_b_lo, y1_b_hi)
    w0 = is_hi_half.select(y0_b, as_mlir_value(d0_a))
    w1 = is_hi_half.select(y1_b, as_mlir_value(d1_a))
    w2 = is_hi_half.select(as_mlir_value(d0_b), y0_a)
    w3 = is_hi_half.select(as_mlir_value(d1_b), y1_a)
    return w0, w1, w2, w3


def _stagger_extra_barrier_if_zero(stagger_i32):
    """Emit `s_barrier;` only when stagger == 0."""
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [stagger_i32],
        ("s_cmp_eq_u32 $0, 0\n\ts_cbranch_scc0 1f\n\ts_barrier\n\t1:"),
        "s",
        has_side_effects=True,
    )


@flyc.jit
def _stagger_extra_barrier_if_one(stagger_i32):
    """Emit `sched_barrier(0); s_barrier;` only when stagger == 1."""
    if fx.Int32(stagger_i32) != fx.Int32(0):
        rocdl.sched_barrier(0)
        rocdl.s_barrier()


# Lazy-rescale: keep the old max while every lane's tile max is within
# this log2 headroom, then skip O/l rebasing.
_LAZY_RESCALE_THRESHOLD = 8.0
# s_waitcnt: lgkmcnt=0, vmcnt=63, expcnt=7. Drain LDS without killing in-flight DMA.
_LGKMCNT_0_ONLY = 0xC07F
FLEX_DTYPE_BF16 = 2
FLEX_DTYPE_FP16 = 3

_LOG2E = 1.4426950408889634
_MAX_BUFFER_BYTES = 0x7FFFFFFF

# Compile-time 8-cluster cutoff. Masked paths skip KV tiles, so the 8c body
# only pays once the tensor Skv is long enough for the last Q tiles.
_LONG_SEQ_8C_SKV_DENSE = 768
_LONG_SEQ_8C_SKV_MASKED = 2048
# Lower-tri 8c is branch-free: only pairs wholly below the diagonal
# use C0..C7. The diagonal band is a 4c tail with packed attn_mask_vec2_imm.
_MASK_NEG_INF_F32_BITS = struct.unpack("<I", struct.pack("<f", -1e9))[0]
# Split-K host cap: keep a split only while each partition has at least
# six n64 tiles. The kernel then folds a leftover live range shorter than
# four tiles into the previous partition so residual workgroups stay empty.
_SPLITK_MIN_CHUNK_TILES = 6
_SPLITK_MIN_LIVE_TILES = 4

_IDENTITY_SCORE_MOD = lambda score, b, h, q, kv: score
_VISIBLE_MASK_MOD = lambda b, h, q, kv: q == q


def _prepare_constexpr_callable(fn, fallback):
    if fn is None:
        return fallback
    prepared = fn
    # FlyDSL constexprs key anonymous functions by bytecode. Preserve a normal
    # Python function's exact code/defaults while presenting it through that
    # supported callable representation.
    if isinstance(fn, types.FunctionType) and fn.__name__ != "<lambda>":
        prepared = types.FunctionType(
            fn.__code__,
            fn.__globals__,
            name="<lambda>",
            argdefs=fn.__defaults__,
            closure=fn.__closure__,
        )
        prepared.__kwdefaults__ = fn.__kwdefaults__
    try:
        Constexpr.value_signature(prepared)
    except TypeError as exc:
        raise TypeError(
            "score_mod/mask_mod must be a Python function without closures or "
            "global references; bind scalar parameters as positional defaults"
        ) from exc
    return prepared


class _InspectedMod:
    """Callable spec plus host-derived equivalent math/layout lowerings."""

    def __init__(
        self,
        score_mod,
        mask_mod,
        *,
        has_score=False,
        has_mask=False,
        lower_tri=False,
        banded=False,
        has_prefix=False,
        q_offset=0,
        window=0,
        prefix_len=0,
        affine_score=False,
        score_slope=0.0,
    ):
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.has_score = has_score
        self.has_mask = has_mask
        self.needs_safe_norm = has_mask
        self.lower_tri = lower_tri
        self.banded = banded
        self.has_prefix = has_prefix
        self.q_offset = q_offset
        self.window = window
        self.prefix_len = prefix_len
        self.affine_score = affine_score
        self.score_slope = score_slope
        self.score_slope_log2 = score_slope * _LOG2E
        self.lpt_q_grid = lower_tri
        self.packed_lower_tri_mask = lower_tri and not banded and not has_prefix
        # Predictable masks derive their bounds arithmetically in kv_range().
        # Generic masks instead use a host-inferred per-Q-workgroup envelope.
        self.needs_kv_bounds = has_mask and not lower_tri
        self._device_kv_bounds = {}

    def get_kv_bounds(
        self,
        *,
        seqlen_q,
        seqlen_kv,
        num_batches,
        num_heads,
        rows_per_wg,
        block_n,
        device,
    ):
        key = (
            seqlen_q,
            seqlen_kv,
            num_batches,
            num_heads,
            rows_per_wg,
            block_n,
            device.type,
            device.index,
        )
        if key not in self._device_kv_bounds:
            host_bounds = _infer_generic_kv_bounds(
                self.mask_mod,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                num_batches=num_batches,
                num_heads=num_heads,
                rows_per_wg=rows_per_wg,
                block_n=block_n,
            )
            self._device_kv_bounds[key] = host_bounds.to(device=device)
        return self._device_kv_bounds[key]

    def kv_range(self, q_min_wg, q_max_wg, n_kv_tiles, block_n):
        if const_expr(self.lower_tri):
            q_hi = q_max_wg
            if const_expr(self.has_prefix):
                q_hi = _i32_max(q_hi, fx.Int32(self.prefix_len - 1))
            raw_hi = (q_hi + fx.Int32(block_n)) // fx.Int32(block_n)
            kv_hi = _i32_min(raw_hi, fx.Int32(n_kv_tiles))
            if const_expr(self.banded):
                raw_lo = (q_min_wg - fx.Int32(self.window)) // fx.Int32(block_n)
                return _i32_max(raw_lo, fx.Int32(0)), kv_hi
            return fx.Int32(0), kv_hi
        return fx.Int32(0), fx.Int32(n_kv_tiles)

    def tile_needs_mask(self, kv_tile_idx, q_idx, block_n):
        kv_tile_end = kv_tile_idx * fx.Int32(block_n) + fx.Int32(block_n - 1)
        needs_mask = kv_tile_end > q_idx
        if const_expr(self.banded):
            kv_tile_start = kv_tile_idx * fx.Int32(block_n)
            needs_mask = needs_mask | ((q_idx - kv_tile_start) > fx.Int32(self.window))
        return needs_mask

    def apply_mask(self, score, b, h, q_idx, kv_idx):
        visible = self.mask_mod(b, h, q_idx, kv_idx)
        return visible.select(score, fx.Float32(-1e9))

    def apply_score(self, score, b, h, q_idx, kv_idx):
        if const_expr(self.affine_score):
            # The pipeline stores logits in log2 space. The inspected
            # score + slope*(kv-q) form can be applied directly without the
            # generic natural-log round trip.
            relative = fx.Float32(kv_idx - q_idx)
            return fx.Float32(score) + relative * fx.Float32(self.score_slope_log2)
        # The pipeline stores logits in log2 space. Present the PyTorch
        # callable with its natural-log score and convert its result back.
        natural_score = fx.Float32(score) / fx.Float32(_LOG2E)
        modified = self.score_mod(natural_score, b, h, q_idx, kv_idx)
        return fx.Float32(modified) * fx.Float32(_LOG2E)


class _ScoreExpr:
    """Small expression tree used to recognize affine score modifiers."""

    def __init__(self, op, *args):
        self.op = op
        self.args = args

    @staticmethod
    def variable(name):
        return _ScoreExpr("var", name)

    @staticmethod
    def coerce(value):
        if isinstance(value, _ScoreExpr):
            return value
        if isinstance(value, (int, float)):
            return _ScoreExpr("const", float(value))
        raise TypeError(f"unsupported symbolic score value {type(value)!r}")

    def __add__(self, other):
        return _ScoreExpr("add", self, self.coerce(other))

    def __radd__(self, other):
        return _ScoreExpr("add", self.coerce(other), self)

    def __sub__(self, other):
        return _ScoreExpr("sub", self, self.coerce(other))

    def __rsub__(self, other):
        return _ScoreExpr("sub", self.coerce(other), self)

    def __mul__(self, other):
        return _ScoreExpr("mul", self, self.coerce(other))

    def __rmul__(self, other):
        return _ScoreExpr("mul", self.coerce(other), self)

    def __neg__(self):
        return _ScoreExpr("neg", self)


def _affine_score_tree(expr):
    """Return ({variable: coefficient}, constant) or raise for non-affine trees."""
    expr = _ScoreExpr.coerce(expr)
    if expr.op == "const":
        return {}, expr.args[0]
    if expr.op == "var":
        return {expr.args[0]: 1.0}, 0.0
    if expr.op == "neg":
        coeffs, constant = _affine_score_tree(expr.args[0])
        return {name: -value for name, value in coeffs.items()}, -constant
    if expr.op in ("add", "sub"):
        lhs, lhs_c = _affine_score_tree(expr.args[0])
        rhs, rhs_c = _affine_score_tree(expr.args[1])
        sign = 1.0 if expr.op == "add" else -1.0
        coeffs = dict(lhs)
        for name, value in rhs.items():
            coeffs[name] = coeffs.get(name, 0.0) + sign * value
        return coeffs, lhs_c + sign * rhs_c
    if expr.op == "mul":
        lhs, lhs_c = _affine_score_tree(expr.args[0])
        rhs, rhs_c = _affine_score_tree(expr.args[1])
        if lhs and rhs:
            raise TypeError("non-affine score product")
        if lhs:
            return {name: value * rhs_c for name, value in lhs.items()}, lhs_c * rhs_c
        if rhs:
            return {name: value * lhs_c for name, value in rhs.items()}, rhs_c * lhs_c
        return {}, lhs_c * rhs_c
    raise TypeError(f"unsupported symbolic score op {expr.op!r}")


def _infer_affine_score(score_mod, *, enabled=True):
    """Recognize score + slope*(kv-q), then verify it on numeric holdouts."""
    if score_mod is None or not enabled:
        return False, 0.0
    if getattr(score_mod, "flex_infer", True) is False:
        return False, 0.0
    if getattr(score_mod, "flex_score_kind", None) == "exact":
        return False, 0.0

    unsupported = (
        "LOAD_GLOBAL",
        "LOAD_ATTR",
        "CALL",
        "JUMP",
        "FOR_ITER",
    )
    for inst in dis.get_instructions(score_mod):
        if any(inst.opname.startswith(prefix) for prefix in unsupported):
            return False, 0.0

    try:
        result = score_mod(
            _ScoreExpr.variable("score"),
            _ScoreExpr.variable("b"),
            _ScoreExpr.variable("h"),
            _ScoreExpr.variable("q"),
            _ScoreExpr.variable("kv"),
        )
        coeffs, constant = _affine_score_tree(result)
    except (TypeError, ValueError, AttributeError):
        return False, 0.0

    slope = coeffs.get("kv", 0.0)
    if not math.isclose(coeffs.get("score", 0.0), 1.0):
        return False, 0.0
    if not math.isclose(coeffs.get("q", 0.0), -slope):
        return False, 0.0
    if not math.isclose(coeffs.get("b", 0.0), 0.0):
        return False, 0.0
    if not math.isclose(coeffs.get("h", 0.0), 0.0):
        return False, 0.0
    if not math.isclose(constant, 0.0):
        return False, 0.0
    if any(name not in ("score", "b", "h", "q", "kv") for name in coeffs):
        return False, 0.0

    q = torch.tensor([0, 1, 7, 31, 127], dtype=torch.float64).reshape(-1, 1)
    kv = torch.tensor([0, 2, 13, 63, 191], dtype=torch.float64).reshape(1, -1)
    score = torch.linspace(-0.75, 0.75, q.numel() * kv.numel(), dtype=torch.float64).reshape(q.numel(), kv.numel())
    try:
        actual = torch.as_tensor(score_mod(score, 0, 0, q, kv), dtype=torch.float64)
        expected = score + slope * (kv - q)
    except Exception:
        return False, 0.0
    if not torch.allclose(actual, expected, rtol=1e-7, atol=1e-9):
        return False, 0.0
    return True, float(slope)


@lru_cache(maxsize=256)
def _infer_generic_kv_bounds(
    mask_mod,
    *,
    seqlen_q,
    seqlen_kv,
    num_batches,
    num_heads,
    rows_per_wg,
    block_n,
):
    """Infer a conservative [lo, hi) KV-tile envelope for each Q workgroup.

    This is only used for masks that the inspector cannot represent with a
    predictable lower-tri/band/prefix formula. Interior holes remain in the
    envelope and are still handled by the original mask callable.
    """
    n_q_tiles = (seqlen_q + rows_per_wg - 1) // rows_per_wg
    n_kv_tiles = (seqlen_kv + block_n - 1) // block_n
    bounds = torch.empty((num_batches, num_heads, n_q_tiles, 2), dtype=torch.int32)
    kv_values = list(range(seqlen_kv))

    def infer_one(b, h):
        bh_bounds = torch.empty((n_q_tiles, 2), dtype=torch.int32)
        for q_tile in range(n_q_tiles):
            q_lo = q_tile * rows_per_wg
            q_hi = min(q_lo + rows_per_wg, seqlen_q)
            visible_cols = _mask_values(mask_mod, list(range(q_lo, q_hi)), kv_values, b=b, h=h).any(dim=0)
            visible = visible_cols.nonzero().flatten()
            if len(visible):
                kv_lo = int(visible[0]) // block_n
                kv_hi = int(visible[-1]) // block_n + 1
            else:
                # The deferred-softmax pipeline requires at least one
                # iteration. Keep one tile and let mask_mod produce the
                # existing all-masked safe-normalization result.
                kv_lo, kv_hi = 0, min(1, n_kv_tiles)
            bh_bounds[q_tile, 0] = kv_lo
            bh_bounds[q_tile, 1] = kv_hi
        return bh_bounds

    # Avoid repeating the potentially large Q×KV host evaluation only when
    # bytecode proves that the callable never reads its B/H arguments.
    arg_names = mask_mod.__code__.co_varnames[: mask_mod.__code__.co_argcount]
    batch_arg = arg_names[0] if len(arg_names) > 0 else None
    head_arg = arg_names[1] if len(arg_names) > 1 else None
    used_args = {inst.argval for inst in dis.get_instructions(mask_mod) if inst.opname == "LOAD_FAST"}
    batch_head_invariant = batch_arg not in used_args and head_arg not in used_args
    if batch_head_invariant:
        shared = infer_one(0, 0)
        bounds.copy_(shared)
    else:
        for b in range(num_batches):
            for h in range(num_heads):
                bounds[b, h].copy_(infer_one(b, h))
    return bounds


def _mask_values(mask_mod, q_values, kv_values, b=0, h=0):
    q = torch.tensor(q_values, dtype=torch.int64).reshape(-1, 1)
    kv = torch.tensor(kv_values, dtype=torch.int64).reshape(1, -1)
    try:
        values = mask_mod(
            torch.full_like(q, b),
            torch.full_like(q, h),
            q,
            kv,
        )
        values = torch.as_tensor(values, dtype=torch.bool)
        return values.expand(len(q_values), len(kv_values))
    except Exception:
        return torch.tensor(
            [[bool(mask_mod(b, h, qv, kvv)) for kvv in kv_values] for qv in q_values],
            dtype=torch.bool,
        )


@lru_cache(maxsize=256)
def inspect_flex_mods(
    score_mod,
    mask_mod,
    *,
    seqlen_q,
    seqlen_kv,
    num_batches=1,
    num_heads=1,
    paged=False,
    infer_score_mod=True,
):
    """Classify mask/score structure for equivalent optimized lowerings.

    ``infer_score_mod=False``, ``score_mod.flex_infer=False``, or
    ``score_mod.flex_score_kind="exact"`` disables score-math inference.
    The original callable remains stored on this mod in every case.
    """
    score_callable = _prepare_constexpr_callable(score_mod, _IDENTITY_SCORE_MOD)
    mask_callable = _prepare_constexpr_callable(mask_mod, _VISIBLE_MASK_MOD)
    affine_score, score_slope = _infer_affine_score(score_mod, enabled=infer_score_mod)
    score_kwargs = {
        "has_score": score_mod is not None,
        "affine_score": affine_score,
        "score_slope": score_slope,
    }
    if mask_mod is None:
        return _InspectedMod(score_callable, mask_callable, **score_kwargs)

    q_values = {round(i * (seqlen_q - 1) / 16) for i in range(17)}
    q_values.update(min(boundary, seqlen_q - 1) for boundary in (0, 1, 31, 32, 63, 64, 127, 128, 255, 256))
    q_values = sorted(q_values)
    kv_values = list(range(seqlen_kv))
    actual = _mask_values(mask_callable, q_values, kv_values)
    # Layout skipping is legal only when the predicate is batch/head invariant.
    for b, h in ((max(num_batches - 1, 0), 0), (0, max(num_heads - 1, 0))):
        if not torch.equal(actual, _mask_values(mask_callable, q_values, kv_values, b, h)):
            return _InspectedMod(
                score_callable,
                mask_callable,
                has_mask=True,
                **score_kwargs,
            )

    offsets = [0] if paged else list(dict.fromkeys((0, seqlen_kv - seqlen_q)))
    q_grid = torch.tensor(q_values, dtype=torch.int64).reshape(-1, 1)
    kv_grid = torch.arange(seqlen_kv, dtype=torch.int64).reshape(1, -1)
    for offset in offsets:
        upper = q_grid + offset
        lower_tri = kv_grid <= upper
        if torch.equal(actual, lower_tri):
            return _InspectedMod(
                score_callable,
                mask_callable,
                has_mask=True,
                lower_tri=True,
                q_offset=offset,
                **score_kwargs,
            )

        # A band has the same upper edge and one constant inclusive width.
        visible = actual & lower_tri
        if torch.equal(actual, visible):
            first_visible = []
            for row in actual:
                indices = row.nonzero().flatten()
                if len(indices) == 0:
                    first_visible.append(None)
                else:
                    first_visible.append(int(indices[0]))
            windows = [
                qv + offset - first for qv, first in zip(q_values, first_visible) if first is not None and first > 0
            ]
            for window in set(windows):
                expected = lower_tri & (kv_grid >= upper - window)
                if torch.equal(actual, expected):
                    return _InspectedMod(
                        score_callable,
                        mask_callable,
                        has_mask=True,
                        lower_tri=True,
                        banded=True,
                        q_offset=offset,
                        window=window,
                        **score_kwargs,
                    )

        # Prefix-LM is a lower triangle union a fixed leading prefix.
        prefix_candidates = set()
        for qv, row in zip(q_values, actual):
            indices = row.nonzero().flatten()
            if len(indices) and int(indices[-1]) > qv + offset:
                prefix_candidates.add(int(indices[-1]) + 1)
        for prefix_len in prefix_candidates:
            expected = lower_tri | (kv_grid < prefix_len)
            if torch.equal(actual, expected):
                return _InspectedMod(
                    score_callable,
                    mask_callable,
                    has_mask=True,
                    lower_tri=True,
                    has_prefix=True,
                    q_offset=offset,
                    prefix_len=prefix_len,
                    **score_kwargs,
                )

    return _InspectedMod(
        score_callable,
        mask_callable,
        has_mask=True,
        **score_kwargs,
    )


def _make_k_lds_layout(block_n, head_dim):
    # GEMM make_transposed_lds_layout XOR swizzle on D-contiguous storage.
    # Keep (block_n, head_dim) shape so QK MFMA fragment A matches loop_m=block_n.
    base_layout = fx.make_layout((block_n, head_dim), (head_dim, 1))
    if const_expr(head_dim == 128):
        k_swizzle = fx.static(fx.SwizzleType.get(3, 3, 3))
        return fx.make_composed_layout(k_swizzle, base_layout)
    return base_layout


@fx.struct
class FlexAttnParam:
    dtype_id: fx.Constexpr[int]
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    head_dim: fx.Constexpr[int]
    num_heads_q: fx.Constexpr[int]
    num_heads_kv: fx.Constexpr[int]
    # wave tiling
    m_waves: fx.Constexpr[int]
    n_waves: fx.Constexpr[int]
    # num_groups independent query subtiles per workgroup, all sharing the same KV
    # loop. Each group runs the validated 32-row body on rows
    # [group*block_m : (group+1)*block_m); K/V are loaded once and reused across all
    # groups (strategy A). Total query rows per workgroup = num_groups*block_m.
    # Default 8: fills all 8 SIMDs/CU (8 groups × 1 wave × 64 threads = 512) and
    # enables wave-group stagger for overlapping DMA with compute.
    num_groups: fx.Constexpr[int]
    # mma shape
    mma_m: fx.Constexpr[int]
    mma_n: fx.Constexpr[int]
    mma_k: fx.Constexpr[int]
    # derived
    group_threads: fx.Constexpr[int]  # threads per group = m_waves*n_waves*wave_size
    block_threads: fx.Constexpr[int]  # = num_groups * group_threads
    gqa_group: fx.Constexpr[int]
    in_data_bytes: fx.Constexpr[int]
    n_kv_tiles: fx.Constexpr[int]  # seqlen_kv // block_n
    # Dedicated 8-cluster schedule for long n64 sequences.
    long_seq_8c: fx.Constexpr[bool]
    pipe_depth: fx.Constexpr[int]  # 1 = monolithic, 2 = decomposed pipeline
    pipe_stages: fx.Constexpr[int]  # deprecated: stagger follows num_groups/pipe_depth/m_waves
    # True = exact per-row softmax; False = approximate column softmax (mma_m=32 only)
    accurate_softmax: fx.Constexpr[bool]
    has_mask: fx.Constexpr[bool]
    has_score: fx.Constexpr[bool]
    affine_score: fx.Constexpr[bool]
    score_slope_log2: fx.Constexpr[float]
    lower_tri: fx.Constexpr[bool]
    banded: fx.Constexpr[bool]
    has_prefix: fx.Constexpr[bool]
    packed_lower_tri_mask: fx.Constexpr[bool]
    lpt_q_grid: fx.Constexpr[bool]
    has_kv_bounds: fx.Constexpr[bool]
    q_offset: fx.Constexpr[int]
    mask_window: fx.Constexpr[int]
    mask_prefix_len: fx.Constexpr[int]
    num_kv_splits: fx.Constexpr[int]  # split-K: partition KV range across this many WGs (1=disabled)
    paged: fx.Constexpr[bool]  # True = paged KV cache, False = contiguous


_PAGED_BT_LDS_SIZE = 2048


def _effective_lower_tri_kv_splits(
    *,
    requested_splits: int,
    batch: int,
    seqlen_q: int,
    seqlen_kv: int,
    num_heads_q: int,
    rows_per_wg: int,
    block_n: int,
    num_cus: int,
) -> int:
    """Cap lower-tri split-K to splits that can add useful device parallelism.

    A split is useful only while the unsplit lower-tri grid underfills the CUs,
    and only when the largest lower-tri KV range leaves a substantial n64 chunk
    per split.  Six tiles matches the minimum useful dual-tile pipeline chunk
    and avoids paying workspace/combine overhead for very short partitions.
    """
    if requested_splits <= 1:
        return 1
    if min(batch, seqlen_q, seqlen_kv, num_heads_q, rows_per_wg, block_n, num_cus) <= 0:
        return 1

    num_q_tiles = (seqlen_q + rows_per_wg - 1) // rows_per_wg
    dead_q_rows = max(seqlen_q - seqlen_kv, 0)
    first_live_q_tile = dead_q_rows // rows_per_wg
    active_q_tiles = max(num_q_tiles - first_live_q_tile, 0)
    base_wgs = batch * num_heads_q * active_q_tiles
    if base_wgs <= 0 or base_wgs >= num_cus:
        return 1

    occupancy_splits = (num_cus + base_wgs - 1) // base_wgs
    effective = min(requested_splits, occupancy_splits)
    n_kv_tiles = (seqlen_kv + block_n - 1) // block_n
    while effective > 1:
        chunk = (n_kv_tiles + effective - 1) // effective
        chunk = ((chunk + 1) // 2) * 2
        if chunk >= _SPLITK_MIN_CHUNK_TILES:
            break
        effective -= 1
    return effective


def make_flex_attn_param(
    seqlen_kv: int,
    dtype_id: int = FLEX_DTYPE_BF16,
    block_m: int = 32,
    block_n: int = 64,
    head_dim: int = 128,
    num_heads_q: int = 8,
    num_heads_kv: int = 8,
    m_waves: int = 1,
    n_waves: int = 1,
    num_groups: int = 8,
    mma_m: int = 32,
    mma_n: int = 32,
    mma_k: int = 16,
    pipe_depth: int = 1,
    pipe_stages: int = 1,
    accurate_softmax: bool = True,
    flex_mod: Optional[_InspectedMod] = None,
    num_kv_splits: int = 1,
    paged: bool = False,
    long_seq_8c: Optional[bool] = None,
    seqlen_q: Optional[int] = None,
    has_kv_bounds: bool = False,
) -> FlexAttnParam:
    if flex_mod is None:
        flex_mod = inspect_flex_mods(
            None,
            None,
            seqlen_q=seqlen_q if seqlen_q is not None else seqlen_kv,
            seqlen_kv=seqlen_kv,
            paged=paged,
        )
    if dtype_id not in (FLEX_DTYPE_BF16, FLEX_DTYPE_FP16):
        raise ValueError(f"unsupported dtype_id={dtype_id}")
    if block_m <= 0 or block_n <= 0 or head_dim <= 0:
        raise ValueError("block_m, block_n, head_dim must be positive")
    if (mma_m, mma_n, mma_k) != (32, 32, 16):
        raise ValueError("gfx950 flex is MFMA 32x32x16 only")
    if block_m % (m_waves * mma_m) != 0:
        raise ValueError(f"block_m ({block_m}) must be divisible by m_waves*mma_m ({m_waves * mma_m})")
    if block_n % (n_waves * mma_n) != 0:
        raise ValueError(f"block_n ({block_n}) must be divisible by n_waves*mma_n ({n_waves * mma_n})")
    if n_waves != 1:
        raise ValueError("n_waves must be 1 (softmax row reduction requires all N-lanes in one wave)")
    if block_n != 64 or head_dim != 128:
        raise ValueError("gfx950 flex is n64 d128 only (4c/8c)")
    if num_groups < 1:
        raise ValueError("num_groups must be >= 1")
    if num_heads_q % num_heads_kv != 0:
        raise ValueError("num_heads_q must be divisible by num_heads_kv (GQA)")
    if head_dim % mma_k != 0:
        raise ValueError(f"head_dim ({head_dim}) must be divisible by mma_k ({mma_k})")
    if seqlen_kv % block_n != 0:
        raise ValueError(f"seqlen_kv ({seqlen_kv}) must be a multiple of block_n ({block_n})")
    if pipe_stages not in (1, 2):
        raise ValueError("pipe_stages must be 1 or 2")
    if pipe_stages >= 2 and pipe_depth < 2:
        raise ValueError("pipe_stages=2 requires pipe_depth>=2 (decomposed pipeline)")
    if pipe_depth >= 2 and not pipeline_stagger_enabled(
        depth=pipe_depth,
        num_groups=num_groups,
        m_waves=m_waves,
    ):
        raise ValueError(
            "pipe_depth>=2 requires pipeline stagger: num_groups>=2 and m_waves>=2 "
            f"(got num_groups={num_groups}, m_waves={m_waves})"
        )

    in_dbytes = 2

    group_threads = m_waves * n_waves * GFX950_WAVE_SIZE
    block_threads = num_groups * group_threads
    _max_waves = 8
    if block_threads > _max_waves * GFX950_WAVE_SIZE:
        raise ValueError(
            f"block_threads ({block_threads}) exceeds {_max_waves} SIMDs/CU limit "
            f"({_max_waves * GFX950_WAVE_SIZE} threads); reduce num_groups or m_waves"
        )
    if block_threads != 512:
        raise ValueError(
            f"gfx950 flex requires a 512-thread workgroup for the 4c/8c path " f"(got block_threads={block_threads})"
        )

    return FlexAttnParam(
        dtype_id=dtype_id,
        block_m=block_m,
        block_n=block_n,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        m_waves=m_waves,
        n_waves=n_waves,
        num_groups=num_groups,
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
        group_threads=group_threads,
        block_threads=block_threads,
        gqa_group=num_heads_q // num_heads_kv,
        in_data_bytes=in_dbytes,
        n_kv_tiles=seqlen_kv // block_n,
        long_seq_8c=(
            (
                (
                    seqlen_kv >= (_LONG_SEQ_8C_SKV_MASKED if flex_mod.has_mask else _LONG_SEQ_8C_SKV_DENSE)
                    # Short-Q split-K benefits from the lower-power,
                    # finer-grained 4c body even when the full Skv would
                    # normally select 8c. Long-Q remains throughput-bound
                    # and keeps 8c.
                    and (num_kv_splits == 1 or seqlen_q is None or seqlen_q > 512)
                )
                if long_seq_8c is None
                else bool(long_seq_8c)
            )
            and not paged
        ),
        pipe_depth=pipe_depth,
        pipe_stages=pipe_stages,
        accurate_softmax=accurate_softmax,
        has_mask=flex_mod.has_mask,
        has_score=flex_mod.has_score,
        affine_score=flex_mod.affine_score,
        score_slope_log2=flex_mod.score_slope_log2,
        lower_tri=flex_mod.lower_tri,
        banded=flex_mod.banded,
        has_prefix=flex_mod.has_prefix,
        packed_lower_tri_mask=flex_mod.packed_lower_tri_mask,
        lpt_q_grid=flex_mod.lpt_q_grid,
        has_kv_bounds=has_kv_bounds,
        q_offset=flex_mod.q_offset,
        mask_window=flex_mod.window,
        mask_prefix_len=flex_mod.prefix_len,
        num_kv_splits=num_kv_splits,
        paged=paged,
    )


def _flex_stagger_divisor(block_threads: int) -> int:
    """Waves per stagger half (``wave_id // N`` for 32×32 MFMA)."""
    total_waves = block_threads // GFX950_WAVE_SIZE
    return max(1, total_waves // 2)


def make_flex_attn_kernel_name(param: FlexAttnParam) -> str:
    dtype_str = "fp16" if param.dtype_id == FLEX_DTYPE_FP16 else "bf16"
    name = f"flex_attn_{dtype_str}_m{param.block_m}n{param.block_n}d{param.head_dim}"
    name += f"_w{param.m_waves}x{param.n_waves}g{param.num_groups}"
    name += "_dense"
    name += "_rsm" if param.accurate_softmax else "_csm"
    name += f"_pd{param.pipe_depth}"
    if bool(param.long_seq_8c):
        name += "_8c"
    name += "_stg"
    return name


_FM = fx.arith.FastMathFlags.fast


def _elem_dtype(dtype_id):
    if dtype_id == FLEX_DTYPE_FP16:
        return fx.Float16
    return fx.BFloat16


def _size_scalar(shape) -> int:
    s = fx.size(shape)
    if hasattr(s, "unpack"):
        return s.unpack()
    if hasattr(s, "is_static") and s.is_static:
        v = s.to_py_value()
        if isinstance(v, tuple):
            return int(v[0]) if len(v) == 1 else int(v)
        return int(v)
    raise TypeError(f"cannot get static size from {type(s)!r}")


def _to_elem(val, elem_ty):
    if hasattr(val, "to"):
        return val.to(elem_ty)
    return fx.Float32(val).to(elem_ty)


def _hw_exp2(x):
    return fx.Float32(rocdl.exp2(T.f32, fx.Float32(x).ir_value()))


def _permlane32_reduce(x, mode):
    """Cross-half-wave reduce via permlane32_swap (1 instruction)."""
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import arith as _arith
    from flydsl._mlir.dialects import llvm

    v_i32 = fx.Int32(_arith.bitcast(T.i32, fx.Float32(x).ir_value()))
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(pair_ty, v_i32.ir_value(), v_i32.ir_value(), False, True)
    lhs_i32 = llvm.extractvalue(T.i32, swapped, [0])
    rhs_i32 = llvm.extractvalue(T.i32, swapped, [1])
    lhs = fx.Float32(_arith.bitcast(T.f32, lhs_i32))
    rhs = fx.Float32(_arith.bitcast(T.f32, rhs_i32))
    if mode == "max":
        return _f32_max(lhs, rhs)
    else:
        return lhs.addf(rhs, fastmath=_FM)


def _mfma_acc(a, b, c, mma_atom):
    """Single MFMA call: C += A × B. Returns updated accumulator."""
    from flydsl._mlir.dialects import fly

    acc_ty = c.type
    return fly.mma_atom_call_ssa([acc_ty], mma_atom, a, b, c)


def _idx_to_i32(idx):
    return fx.Int32(arith.index_cast(T.i32, idx))


@flyc.kernel
def flex_attn_fwd_gfx950_kernel(
    o: fx.Tensor,  # [B, Sq, Hq, D]
    q: fx.Tensor,  # [B, Sq, Hq, D]
    k: fx.Tensor,  # [B, Skv, Hkv, D]
    v: fx.Tensor,  # [B, Skv, Hkv, D]
    seqlen_q: fx.Int32,
    seqlen_kv: fx.Int32,
    num_batches: fx.Int32,
    scale: fx.Float32,
    tiled_mma_qk: fx.TiledMma,
    param: FlexAttnParam,
    score_mod: fx.Constexpr[Callable],
    mask_mod: fx.Constexpr[Callable],
    ws_o: fx.Tensor = fx.Tensor,
    ws_ml: fx.Tensor = fx.Tensor,
    block_table: fx.Tensor = fx.Tensor,  # [B * max_pages_per_seq] i32, flat
    block_table_stride: fx.Int32 = fx.Int32(0),
    context_lens: fx.Tensor = fx.Tensor,  # [B] i32
    kv_bounds: fx.Tensor = fx.Tensor,  # [B, Hq, Q-workgroups, 2] i32, flat
):
    block_m = param.block_m
    block_n = param.block_n
    head_dim = param.head_dim
    elem_dtype = _elem_dtype(param.dtype_id)
    _paged = bool(param.paged)

    tid = fx.thread_idx.x
    # Strategy A: num_groups independent 1-wave query subtiles per workgroup, all
    # driving the SAME KV loop so K/V (staged in LDS) is reused across groups. Each
    # group runs the 64-thread (m_waves=1) body via local_tid; group g owns query
    # rows [(q_tile*num_groups + g)*block_m : +block_m).
    num_groups = param.num_groups
    group_threads = param.group_threads  # 64 (m_waves*n_waves*wave_size)
    group = tid // group_threads
    local_tid = tid % group_threads
    # One SGPR wave id for the whole kernel. Deriving wave from `tid` inside
    # each buffer_load_lds makes LLVM emit v_readfirstlane per DMA (~66/wave).
    _wave_id_uni_s = rocdl.readfirstlane(
        fx.Int32.ir_type,
        fx.Int32(tid // GFX950_WAVE_SIZE).ir_value(),
    )
    _wave_id_uni = fx.Int32(_wave_id_uni_s)
    # Head-fast grid so linear_id % 8 (XCD) is the head, not the Q tile.
    # gfx950 L2 is private per XCD: Q tiles of one head then replay the same KV
    # on one XCD.  Lower-tri masks reverse q_tile on grid.y so the longest
    # KV loops of every head enqueue first.  grid.z = batch (or batch * splits).
    _SPLITK = int(param.num_kv_splits) > 1
    _num_kv_splits = int(param.num_kv_splits)
    _lower_tri_grid = bool(param.lpt_q_grid)
    h_idx = fx.block_idx.x
    if const_expr(_lower_tri_grid):
        _num_q_tiles = (seqlen_q + fx.Int32(num_groups * block_m - 1)) // fx.Int32(num_groups * block_m)
        q_tile = fx.Index(arith.index_cast(T.index, _num_q_tiles - fx.Int32(1) - _idx_to_i32(fx.block_idx.y)))
    else:
        q_tile = fx.block_idx.y
    if const_expr(_SPLITK):
        b_idx = fx.block_idx.z // fx.Index(_num_kv_splits)
        split_idx = _idx_to_i32(fx.block_idx.z % fx.Index(_num_kv_splits))
    else:
        b_idx = fx.block_idx.z
    kv_head = h_idx // param.gqa_group

    q_start = (q_tile * num_groups + group) * block_m
    # Bottom-right lower-tri alignment for contiguous
    # cross-seqlen inputs.  When Skv < Sq, the first Sq-Skv query rows are
    # fully masked; the launcher removes whole dead workgroups and the output
    # store below zeros any dead rows in the first partially-live workgroup.
    # Paged KV keeps its existing top-left convention until it has a per-batch
    # active guard (context lengths may differ across batches).
    _bottom_right_lower_tri = bool(param.packed_lower_tri_mask) and int(param.q_offset) < 0 and not _paged
    _layout_q_offset = fx.Int32(int(param.q_offset))

    if const_expr(_paged):
        _ctx_len_it = fx.recast_iter(fx.Int32, fx.get_iter(context_lens))
        _ctx_len = fx.Int32(fx.ptr_load(_ctx_len_it + _idx_to_i32(b_idx)))
        n_kv_tiles = (_ctx_len + fx.Int32(block_n - 1)) // fx.Int32(block_n)
    else:
        n_kv_tiles = param.n_kv_tiles

    # Build the mod and split bounds before loading Q. Empty split workgroups still
    # publish an (m=-inf,l=0) sentinel, but do not need any query data.
    flex_mod = _InspectedMod(
        score_mod,
        mask_mod,
        has_score=bool(param.has_score),
        affine_score=bool(param.affine_score),
        score_slope=float(param.score_slope_log2) / _LOG2E,
        has_mask=bool(param.has_mask),
        lower_tri=bool(param.lower_tri),
        banded=bool(param.banded),
        has_prefix=bool(param.has_prefix),
        q_offset=int(param.q_offset),
        window=int(param.mask_window),
        prefix_len=int(param.mask_prefix_len),
    )
    _q_min_wg = _idx_to_i32(q_tile) * fx.Int32(num_groups * block_m) + _layout_q_offset
    _q_max_wg = _q_min_wg + fx.Int32(num_groups * block_m - 1)
    if const_expr(bool(param.has_kv_bounds)):
        _num_q_tiles = (seqlen_q + fx.Int32(num_groups * block_m - 1)) // fx.Int32(num_groups * block_m)
        _bounds_idx = (
            (_idx_to_i32(b_idx) * fx.Int32(int(param.num_heads_q)) + _idx_to_i32(h_idx)) * _num_q_tiles
            + _idx_to_i32(q_tile)
        ) * fx.Int32(2)
        _bounds_it = fx.recast_iter(fx.Int32, fx.get_iter(kv_bounds))
        _kv_lo = fx.Int32(fx.ptr_load(_bounds_it + _bounds_idx))
        _kv_hi = fx.Int32(fx.ptr_load(_bounds_it + _bounds_idx + fx.Int32(1)))
    else:
        _kv_lo, _kv_hi = flex_mod.kv_range(_q_min_wg, _q_max_wg, n_kv_tiles, block_n)
    if const_expr(_SPLITK):
        _kv_hi_full = _kv_hi
        _total_tiles = _kv_hi - _kv_lo
        _chunk = (_total_tiles + fx.Int32(_num_kv_splits - 1)) // fx.Int32(_num_kv_splits)
        _chunk = ((_chunk + fx.Int32(1)) // fx.Int32(2)) * fx.Int32(2)
        _chunk = _i32_max(_chunk, fx.Int32(_SPLITK_MIN_CHUNK_TILES))
        _kv_lo_candidate = _kv_lo + split_idx * _chunk
        _kv_hi = _i32_min(_kv_lo_candidate + _chunk, _kv_hi_full)
        _kv_hi = ((_kv_hi_full - _kv_hi) < fx.Int32(_SPLITK_MIN_LIVE_TILES)).select(_kv_hi_full, _kv_hi)
        _split_nonempty = (_kv_lo_candidate < _kv_hi_full) & (
            (split_idx == fx.Int32(0)) | (_kv_lo_candidate + fx.Int32(_SPLITK_MIN_LIVE_TILES) <= _kv_hi_full)
        )
        _split_tiles = _split_nonempty.select(_kv_hi - _kv_lo_candidate, fx.Int32(0))
        _empty_fallback = _i32_max(_kv_hi_full - fx.Int32(1), fx.Int32(0))
        _kv_lo = _split_nonempty.select(_kv_lo_candidate, _empty_fallback)

    kv_tile_elems = block_n * head_dim
    _lds_ring_slots = max(2, int(param.pipe_depth))
    # Shared K/V LDS, padded wave-linear (n64, 512 threads):
    # 8 N-reps × 2 D-reps, +16B pad per K line and +64B pad per V line.
    _lds_n_rpt = 8
    _lds_d_rpt = 2
    _k_lds_line = 520  # 512 bf16 payload + 8 bf16 (16B) pad
    _v_lds_line = 544  # 512 bf16 payload + 32 bf16 (64B) pad
    _k_storage_elems = _lds_n_rpt * _lds_d_rpt * _k_lds_line
    _v_storage_elems = _lds_n_rpt * _lds_d_rpt * _v_lds_line

    if const_expr(_paged):

        @fx.struct
        class SharedStorage:
            k_lds_0: fx.Array[elem_dtype, _k_storage_elems, 16]
            k_lds_1: fx.Array[elem_dtype, _k_storage_elems, 16]
            v_lds_0: fx.Array[elem_dtype, _v_storage_elems, 16]
            v_lds_1: fx.Array[elem_dtype, _v_storage_elems, 16]
            p: fx.Array[elem_dtype, num_groups * block_m * block_n, 16]
            bt: fx.Array[fx.Int32, _PAGED_BT_LDS_SIZE, 16]

    else:

        @fx.struct
        class SharedStorage:
            k_lds_0: fx.Array[elem_dtype, _k_storage_elems, 16]
            k_lds_1: fx.Array[elem_dtype, _k_storage_elems, 16]
            v_lds_0: fx.Array[elem_dtype, _v_storage_elems, 16]
            v_lds_1: fx.Array[elem_dtype, _v_storage_elems, 16]
            p: fx.Array[elem_dtype, num_groups * block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    sK_ptr = [storage.k_lds_0.peek().ptr, storage.k_lds_1.peek().ptr]
    sV_ptr = [storage.v_lds_0.peek().ptr, storage.v_lds_1.peek().ptr]

    # Fragment overlay only. Payload addressing is the padded wave-linear map.
    _k_base_layout = _make_k_lds_layout(block_n, head_dim)
    sK = [fx.make_view(sK_ptr[i], _k_base_layout) for i in range_constexpr(_lds_ring_slots)]
    # QK C-fragment template. After the operand swap (K=A, Q=B) the score tile is
    # [M=block_n score, N=block_m query], so the view must be (block_n, block_m) --
    # square only when block_n==block_m. No data flows through here: PV packs P
    # straight from registers, so this exists purely to derive the C layout.
    sP = fx.make_view(
        storage.p.peek().ptr + group * fx.Int32(block_m * block_n),
        fx.make_layout((block_n, block_m), (block_m, 1)),
    )

    # BSHD element (b,s,h,d); Q/O start at q_start, K/V share this head's base.
    hq = param.num_heads_q
    hkv = param.num_heads_kv
    q_off = b_idx * seqlen_q * hq * head_dim + h_idx * head_dim + q_start * hq * head_dim
    o_off = q_off
    k_off = b_idx * seqlen_kv * hkv * head_dim + kv_head * head_dim

    # BufferCopy128b can overshoot head_dim on the last K-group; bound the
    # descriptor to the full tensor so hardware clamps OOB reads to 0.
    _q_total_bytes = num_batches * seqlen_q * fx.Int32(hq * head_dim * param.in_data_bytes)
    q_it = _make_buffer_ptr(
        fx.recast_iter(elem_dtype, fx.get_iter(q)),
        num_records_bytes=_q_total_bytes,
    )
    gQ = fx.make_view(q_it + fx.Int32(q_off), fx.make_layout((block_m, head_dim), (hq * head_dim, 1)))

    thr_qk = tiled_mma_qk.thr_slice(local_tid)

    ca = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    uca = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)

    # Q resident: load once into the GEMM1 B-fragment (reused every KV tile).
    # QK uses K=A, Q=B so C's M-rows = score indices, allowing register C→B pack for PV.
    tcB_q = fx.make_tiled_copy_B(ca, tiled_mma_qk).get_slice(local_tid)
    frag_Q = thr_qk.make_fragment_B(gQ)
    n_q = _size_scalar(frag_Q.shape)
    _scale_log2e_f32 = scale * fx.Float32(_LOG2E)

    def _load_and_scale_q():
        fx.copy(ca, tcB_q.partition_S(gQ), tcB_q.retile(frag_Q))
        for qi in range_constexpr(n_q):
            frag_Q[qi] = _to_elem(
                _to_elem(frag_Q[qi], fx.Float32) * _scale_log2e_f32,
                elem_dtype,
            )

    if const_expr(_SPLITK):

        @flyc.jit
        def _load_q_if_nonempty():
            if _split_nonempty:
                _load_and_scale_q()

        _load_q_if_nonempty()
    else:
        _load_and_scale_q()

    # Persistent O accumulator: 4 × v16f32 (one per D-chunk).
    # With V=A, P=B PV GEMM: each v16f32 has 16 D-values at 1 query-row per lane.
    Vec = fx.Vector
    _n_d_chunks = head_dim // 32
    # One PV MFMA consumes mma_k score positions; n64 needs 4 packs (16 MFMAs).
    _pv_packs = block_n // int(param.mma_k)
    o_accs_init = [Vec.filled(16, 0.0, fx.Float32).ir_value() for _ in range_constexpr(_n_d_chunks)]

    n_c = _size_scalar(thr_qk.partition_C(sP).shape)
    # After QK operand swap (K=A, Q=B), C's M-rows = score indices, N-cols = query.
    # 32×32×16, one wave: each lane has 16 score values at 1 query column.
    # npair=1: single max/sum per lane.
    npair = 1

    # m_i lives in log2-scaled space (Q is pre-multiplied by scale*log2e) so exp2
    # in the softmax hot loop is just subtract + exp2 with no per-element multiply.
    _M_NEG_FLOOR_SCALED = -60.0 * _LOG2E
    m_i = [fx.Float32(_M_NEG_FLOOR_SCALED) for _ in range_constexpr(npair)]
    l_i = [fx.Float32(0.0) for _ in range_constexpr(npair)]

    # K LDS read: padded wave-linear payload; sK/tcA are MFMA A overlay only.
    tcA_k_lds = [fx.make_tiled_copy_A(uca, tiled_mma_qk).get_slice(local_tid) for _ in range_constexpr(_lds_ring_slots)]
    frag_K = [thr_qk.make_fragment_A(sK[i]) for i in range_constexpr(_lds_ring_slots)]

    _v_tr_atom = fx.make_copy_atom(rocdl.cdna4.LDSReadTrans16_64b(), elem_dtype)
    block_threads = param.block_threads
    _dma_bytes = GFX950_DMA_BYTES
    _kv_tile_bytes = kv_tile_elems * param.in_data_bytes
    _dma_ops_per_thread = _kv_tile_bytes // (block_threads * _dma_bytes)
    dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
    _k_row_stride_bytes = hkv * head_dim * param.in_data_bytes
    # Paged KV cache: [num_blocks, block_n, Hkv, D] — page stride and head offset.
    _page_byte_stride = block_n * hkv * head_dim * param.in_data_bytes
    _kv_head_byte_offset = _idx_to_i32(kv_head) * fx.Int32(head_dim * param.in_data_bytes)

    def _stage_kv_lds(tile_idx, buf, do_k, do_v):
        """Non-vectorized wave-linear padded K/V DMA map."""
        lane = fx.Int32(tid % GFX950_WAVE_SIZE)
        wave = _wave_id_uni
        n_in_wave = lane // fx.Int32(8)
        d_bucket = lane % fx.Int32(8)
        tile_row = n_in_wave * fx.Int32(8) + wave
        if const_expr(_paged):
            page_id = _load_page_id(tile_idx)
            src_base = page_id * fx.Int32(_page_byte_stride) + _kv_head_byte_offset
        else:
            src_base = tile_idx * fx.Int32(block_n * _k_row_stride_bytes)
            src_base = src_base + _idx_to_i32(k_off) * fx.Int32(param.in_data_bytes)
        for d in range_constexpr(_lds_d_rpt):
            d_col = d_bucket * fx.Int32(8) + fx.Int32(d * 64)
            src_byte = src_base + tile_row * fx.Int32(_k_row_stride_bytes) + d_col * fx.Int32(param.in_data_bytes)
            if const_expr(do_k):
                # buffer_load_lds adds lane*16B; m0 is one padded line per wave.
                k_m0 = wave * fx.Int32(_k_lds_line * param.in_data_bytes)
                k_m0 = k_m0 + fx.Int32(d * _lds_n_rpt * _k_lds_line * param.in_data_bytes)
                k_dst = fx.add_offset(sK_i8[buf], k_m0)
                fx.copy(
                    dma_atom,
                    fx.slice(k_div, (None, fx.Int32(src_byte))),
                    fx.make_view(k_dst, fx.make_layout(1, 1)),
                )
            if const_expr(do_v):
                v_m0 = wave * fx.Int32(_v_lds_line * param.in_data_bytes)
                v_m0 = v_m0 + fx.Int32(d * _lds_n_rpt * _v_lds_line * param.in_data_bytes)
                v_dst = fx.add_offset(sV_i8[buf], v_m0)
                fx.copy(
                    dma_atom,
                    fx.slice(v_div, (None, fx.Int32(src_byte))),
                    fx.make_view(v_dst, fx.make_layout(1, 1)),
                )

    gK_flat = fx.rocdl.make_buffer_tensor(
        fx.Tensor(fx.make_view(fx.recast_iter(fx.Int8, fx.get_iter(k)), fx.make_layout(_MAX_BUFFER_BYTES, 1))),
        max_size=True,
    )
    gV_flat = fx.rocdl.make_buffer_tensor(
        fx.Tensor(fx.make_view(fx.recast_iter(fx.Int8, fx.get_iter(v)), fx.make_layout(_MAX_BUFFER_BYTES, 1))),
        max_size=True,
    )
    k_div = fx.logical_divide(gK_flat, fx.make_layout(1, 1))
    v_div = fx.logical_divide(gV_flat, fx.make_layout(1, 1))
    sK_i8 = [fx.recast_iter(fx.Int8, sK_ptr[i]) for i in range_constexpr(_lds_ring_slots)]
    sV_i8 = [fx.recast_iter(fx.Int8, sV_ptr[i]) for i in range_constexpr(_lds_ring_slots)]

    def load_kv(tile_idx, slot):
        _stage_kv_lds(tile_idx, slot, True, True)
        return []

    def load_k(tile_idx, slot):
        """DMA only K into LDS[slot]. Used in the V-read mem cluster (C2)."""
        _stage_kv_lds(tile_idx, slot, True, False)
        return []

    def load_v(tile_idx, slot):
        """DMA only V into LDS[slot]. Used in the K-read mem cluster (C0)."""
        _stage_kv_lds(tile_idx, slot, False, True)
        return []

    # V LDS: ds_read_tr16_b64 then shuffle to v8 MFMA A. Half-wave (local_tid//32)
    # selects score rows 0..3 vs 4..7.
    _v_tr_layout = fx.make_layout(4, 1)  # dst/src tile: 4 bf16 (64b) per lane per copy
    _k_frag_retile_0 = tcA_k_lds[0].retile(frag_K[0])
    _k_frag_retile_1 = tcA_k_lds[1].retile(frag_K[1])

    _qk_mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype))

    def _frag_reps(tensor, mode):
        return fx.size(fx.get_shape(tensor)[mode]).to_py_value()

    _qk_k_reps = _frag_reps(frag_K[0], 2)
    _qk_a_m_reps = _frag_reps(frag_K[0], 1)
    _qk_b_n_reps = _frag_reps(frag_Q, 1)

    def gemm1_qk_mfma(frag_S_acc, frag_Q_in, frag_K_in, ki):
        """All M×N MFMAs for one K-group ki. Caller controls ki scheduling."""
        for m in range_constexpr(_qk_a_m_reps):
            for n in range_constexpr(_qk_b_n_reps):
                fx.mma_atom_call(
                    _qk_mma_atom,
                    frag_S_acc[None, m, n],
                    frag_K_in[None, m, ki],
                    frag_Q_in[None, n, ki],
                    frag_S_acc[None, m, n],
                )

    def gemm1_qk_unrolled(frag_Q_in, frag_K_in):
        """QK GEMM with explicit per-ki MFMA calls (register-only, no bank concerns)."""
        frag_S_out = thr_qk.make_fragment_C(sP)
        frag_S_out.fill(0.0)
        for ki in range_constexpr(_qk_k_reps):
            gemm1_qk_mfma(frag_S_out, frag_Q_in, frag_K_in, ki)
        return [frag_S_out]

    def read_k_lds(slot):
        """Read full K using padded wave-linear LDS addresses."""
        lane = fx.Int32(local_tid % GFX950_WAVE_SIZE)
        lane_mod_32 = lane % fx.Int32(32)
        lane_div_32 = lane // fx.Int32(32)
        lane_base = (lane_mod_32 % fx.Int32(8)) * fx.Int32(_k_lds_line)
        lane_base = lane_base + (lane_mod_32 // fx.Int32(8)) * fx.Int32(64)
        lane_base = lane_base + lane_div_32 * fx.Int32(8)
        pack_ty = Vec.make_type(8, elem_dtype)
        for ki in range_constexpr(_qk_k_reps):
            ki_off = (ki // 4) * (_lds_n_rpt * _k_lds_line) + (ki % 4) * 16
            lo_ptr = fx.add_offset(sK_ptr[slot], fx.make_int_tuple(lane_base + fx.Int32(ki_off)))
            hi_ptr = fx.add_offset(lo_ptr, fx.make_int_tuple(fx.Int32(256)))
            lo = Vec(fx.ptr_load(lo_ptr, result_type=pack_ty))
            hi = Vec(fx.ptr_load(hi_ptr, result_type=pack_ty))
            if const_expr(slot == 0):
                _k_frag_retile_0[None, 0, ki].store(lo)
                _k_frag_retile_0[None, 1, ki].store(hi)
            else:
                _k_frag_retile_1[None, 0, ki].store(lo)
                _k_frag_retile_1[None, 1, ki].store(hi)
        return frag_K[slot]

    # Score/mask indices after K=A,Q=B: q = q_start + local_tid%32;
    # kv_in_tile(e) = 8*(e//4) + e%4 + 4*(local_tid//32).
    mod_has_score = flex_mod.has_score
    mod_has_mask = flex_mod.has_mask
    _mod_apply_score = flex_mod.apply_score
    _mod_apply_mask = flex_mod.apply_mask
    b_i32 = _idx_to_i32(b_idx)
    h_i32 = _idx_to_i32(h_idx)
    q_idx_mod = _idx_to_i32(q_start) + fx.Int32(local_tid % 32)
    lane_group_off = fx.Int32((local_tid // 32) * 4)
    kv_offsets = [8 * (e // 4) + (e % 4) for e in range(n_c)]
    # Mask if any row in the WG can see this tile, not this group's q_start.
    # Uniform across the 8 groups.
    _lower_tri_wg_q_min = _idx_to_i32(q_tile) * fx.Int32(num_groups * block_m) + _layout_q_offset
    _mask_neg_inf_i32 = fx.Int32(_MASK_NEG_INF_F32_BITS)
    _packed_pair_thresholds = [(kv_offsets[i], kv_offsets[i + 1]) for i in range(0, n_c, 2)]

    def apply_score_mods(frag_S_in, kv_tile_idx):
        kv_base = kv_tile_idx * fx.Int32(block_n) + lane_group_off
        for e in range_constexpr(n_c):
            kv_idx = kv_base + fx.Int32(kv_offsets[e])
            frag_S_in[e] = _mod_apply_score(frag_S_in[e], b_i32, h_i32, q_idx_mod, kv_idx)

    def apply_mask_mods(frag_S_in, kv_tile_idx):
        kv_base = kv_tile_idx * fx.Int32(block_n) + lane_group_off
        for e in range_constexpr(n_c):
            kv_idx = kv_base + fx.Int32(kv_offsets[e])
            frag_S_in[e] = _mod_apply_mask(frag_S_in[e], b_i32, h_i32, q_idx_mod, kv_idx)

    def _mask_scores(s_in, kv_tile_idx):
        kv_base = kv_tile_idx * fx.Int32(block_n) + lane_group_off
        return [
            _mod_apply_mask(s_in[e], b_i32, h_i32, q_idx_mod, kv_base + fx.Int32(kv_offsets[e]))
            for e in range_constexpr(n_c)
        ]

    def _mask_scores_packed(s_in, kv_tile_idx):
        """Packed vec2 mask on this C-fragment's KV offsets."""
        kv_base = kv_tile_idx * fx.Int32(block_n) + lane_group_off
        rel_i32 = q_idx_mod + _layout_q_offset - kv_base
        s_out = [s_in[e] for e in range_constexpr(n_c)]
        for p in range_constexpr(len(_packed_pair_thresholds)):
            thr_x, thr_y = _packed_pair_thresholds[p]
            idx_x = p * 2
            idx_y = p * 2 + 1
            new_x, new_y = _attn_mask_vec2_imm(
                rel_i32,
                _mask_neg_inf_i32,
                thr_x,
                thr_y,
                _bitcast_i32(s_out[idx_x]),
                _bitcast_i32(s_out[idx_y]),
            )
            s_out[idx_x] = _bitcast_f32(new_x)
            s_out[idx_y] = _bitcast_f32(new_y)
        return s_out

    def apply_mods(frag_S_in, kv_tile_idx):
        if const_expr(mod_has_score):
            apply_score_mods(frag_S_in, kv_tile_idx)
        if const_expr(mod_has_mask):
            if const_expr(bool(param.packed_lower_tri_mask)):
                s_out = [frag_S_in[e] for e in range_constexpr(n_c)]
                # Mask if any row in the WG can see this tile.
                needs_mask = flex_mod.tile_needs_mask(kv_tile_idx, _lower_tri_wg_q_min, block_n)
                if needs_mask:
                    s_out = _mask_scores_packed(s_out, kv_tile_idx)
                for e in range_constexpr(n_c):
                    frag_S_in[e] = s_out[e]
            elif const_expr(bool(param.banded)):
                # Interior band tiles need no mask. Use q_start for the lower-tri
                # edge and the last row in the group for the left window edge.
                s_out = [frag_S_in[e] for e in range_constexpr(n_c)]
                q_lo = _idx_to_i32(q_start) + _layout_q_offset
                q_hi = q_lo + fx.Int32(int(block_m) - 1)
                kv_tile_end = kv_tile_idx * fx.Int32(block_n) + fx.Int32(block_n - 1)
                kv_tile_start = kv_tile_idx * fx.Int32(block_n)
                too_far = kv_tile_end > q_lo
                out_of_window = (q_hi - kv_tile_start) > fx.Int32(int(param.mask_window))
                needs_mask = too_far | out_of_window
                if needs_mask:
                    s_out = _mask_scores(s_out, kv_tile_idx)
                for e in range_constexpr(n_c):
                    frag_S_in[e] = s_out[e]
            else:
                apply_mask_mods(frag_S_in, kv_tile_idx)

    def _scale_o_vec(o_accs_in, scale_scalar):
        """Vectorized O rescale: broadcast scalar to vec16, multiply per D-chunk."""
        scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(16)
        o_out = []
        for dc in range_constexpr(_n_d_chunks):
            o_vec = Vec(o_accs_in[dc])
            o_out.append((o_vec * scale_vec).ir_value())
        return o_out

    def _scale_o_if_needed(o_accs_in, scale_scalar):
        """Wave-uniform fast path for the overwhelmingly common corr == 1 case."""

        @flyc.jit
        def _run(o_accs, corr):
            is_identity = fx.Float32(corr) == fx.Float32(1.0)
            identity_lanes = rocdl.ballot(T.i64, as_mlir_value(is_identity))
            all_identity = arith.cmpi(
                arith.CmpIPredicate.eq,
                as_mlir_value(identity_lanes),
                _read_exec_i64(),
            )
            all_identity = llvm.intr_expect(
                all_identity,
                arith.constant(1, type=ir.IntegerType.get_signless(1)),
            )
            state = [as_mlir_value(o_accs[dc]) for dc in range(_n_d_chunks)]
            names = tuple(f"_lazy_o{dc}" for dc in range(_n_d_chunks))

            def _rescale(_n, *_st):
                scaled = _scale_o_vec(o_accs, corr)
                return [as_mlir_value(scaled[dc]) for dc in range(_n_d_chunks)]

            return list(
                scf_if_dispatch(
                    all_identity,
                    lambda *_a: None,
                    _rescale,
                    state_names=names,
                    state_values=state,
                )
            )

        return _run(o_accs_in, scale_scalar)

    def softmax_start(frag_S_in, m_i_in):
        s_out = [frag_S_in[i] for i in range_constexpr(n_c)]
        tile_max = s_out[0]
        for i in range_constexpr(1, n_c):
            tile_max = _f32_max(tile_max, s_out[i])
        tile_max = _permlane32_reduce(tile_max, "max")
        m_new = _f32_max(m_i_in[0], tile_max)
        corr_scalar = _hw_exp2(m_i_in[0] - m_new)
        return corr_scalar, s_out, m_new

    def softmax_finish(s_scaled, m_i_in, l_i_in, o_accs_in, corr_scalar):
        m_new = m_i_in[0]
        p_elems = [_hw_exp2(s_scaled[i] - m_new) for i in range_constexpr(n_c)]
        p_vec = Vec.from_elements(p_elems, fx.Float32)
        local_sum = p_vec.reduce("add", init_val=fx.Float32(0.0), fastmath=_FM)
        local_sum = _permlane32_reduce(local_sum, "sum")
        corr = [corr_scalar]
        l_new = fx.Float32(fx.fma(l_i_in[0], corr_scalar, local_sum, fastmath=_FM))
        l_i_out = [l_new] + [l_i_in[r] for r in range_constexpr(1, npair)]
        o_accs_out = _scale_o_vec(o_accs_in, corr_scalar)
        return [p_elems, m_i_in, l_i_out, o_accs_out, corr]

    # PV GEMM (V=A, P=B): pack 16 f32 scores to 2×v8 in registers.
    _pv_mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype))

    _is_bf16 = int(param.dtype_id) == FLEX_DTYPE_BF16

    def _pack_8_f32_to_v8elem(vals_8):
        """Pack 8 f32 values into v8 of elem_dtype (bf16 or f16)."""
        if const_expr(_is_bf16):
            pairs = []
            for j in range_constexpr(4):
                pairs.append(rocdl.cvt_pk_bf16_f32(vals_8[j * 2], vals_8[j * 2 + 1]))
            return Vec.from_elements(pairs, fx.Int32).bitcast(fx.BFloat16).ir_value()
        else:
            elems = []
            for j in range_constexpr(8):
                elems.append(fx.Float32(vals_8[j]).to(elem_dtype))
            return Vec.from_elements(elems, elem_dtype).ir_value()

    def _pack_p_b(frag_P_in):
        """Pack the C fragment into one v8elem MFMA B pack per PV step."""
        return [_pack_8_f32_to_v8elem([frag_P_in[pk * 8 + i] for i in range(8)]) for pk in range_constexpr(_pv_packs)]

    def _p_packs_to_vec(p_packs):
        return Vec.from_elements(
            [Vec(p_packs[pk])[i] for pk in range_constexpr(_pv_packs) for i in range_constexpr(8)],
            elem_dtype,
        ).ir_value()

    def _p_vec_to_packs(p_all):
        p_vec = Vec(p_all, (n_c,), elem_dtype)
        return [p_vec.shuffle(p_vec, [pk * 8 + i for i in range(8)]).ir_value() for pk in range_constexpr(_pv_packs)]

    def _scale_p_packs(p_packs, corr):
        p_all = _p_packs_to_vec(p_packs)
        p_f32_ty = Vec.make_type(n_c, fx.Float32)
        p_elem_ty = Vec.make_type(n_c, elem_dtype)
        fm_fast = ir.Attribute.parse("#llvm.fastmath<fast>")
        p_f32_op = llvm.FPExtOp(p_f32_ty, as_mlir_value(p_all))
        p_f32_op.operation.attributes["fastmathFlags"] = fm_fast
        corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(n_c)
        p_scaled = as_mlir_value(corr_vec * Vec(p_f32_op.result))
        p_elem_op = llvm.FPTruncOp(p_elem_ty, p_scaled)
        p_elem_op.operation.attributes["fastmathFlags"] = fm_fast
        return _p_vec_to_packs(p_elem_op.result)

    def _anchor_p_packs(p_packs):
        p_all = as_mlir_value(_p_packs_to_vec(p_packs))
        anchored = llvm.inline_asm(
            p_all.type,
            [p_all],
            "",
            "=v,0",
            has_side_effects=True,
        )
        return _p_vec_to_packs(anchored)

    def _c3_lazy_rescale(o_accs, m_i, l_i, tile_max, p_packs):
        """Wave-uniform rebase of O/l and pending packed P."""

        @flyc.jit
        def _run(o_in, m_row, l_row, tile_max_i, p_in):
            below = (fx.Float32(tile_max_i) - fx.Float32(m_row)) <= fx.Float32(_LAZY_RESCALE_THRESHOLD)
            below_lanes = rocdl.ballot(T.i64, as_mlir_value(below))
            all_below = arith.cmpi(
                arith.CmpIPredicate.eq,
                as_mlir_value(below_lanes),
                _read_exec_i64(),
            )
            all_below = llvm.intr_expect(
                all_below,
                arith.constant(1, type=ir.IntegerType.get_signless(1)),
            )
            p_all = _p_packs_to_vec(p_in)
            state = [as_mlir_value(o_in[dc]) for dc in range(_n_d_chunks)]
            state += [
                as_mlir_value(p_all),
                as_mlir_value(l_row),
                as_mlir_value(m_row),
            ]
            names = tuple(f"_c3_lr{i}" for i in range(_n_d_chunks + 3))

            def _rescale(_n, *_st):
                m_new = _f32_max(m_row, tile_max_i)
                corr = _hw_exp2(fx.Float32(m_row) - m_new)
                o_scaled = _scale_o_vec(o_in, corr)
                p_scaled = _scale_p_packs(p_in, corr)
                out = [as_mlir_value(o_scaled[dc]) for dc in range(_n_d_chunks)]
                out += [
                    as_mlir_value(_p_packs_to_vec(p_scaled)),
                    as_mlir_value(fx.Float32(l_row) * corr),
                    as_mlir_value(m_new),
                ]
                return out

            result = scf_if_dispatch(
                all_below,
                lambda *_a: None,
                _rescale,
                state_names=names,
                state_values=state,
            )
            o_out = list(result[:_n_d_chunks])
            p_out = _p_vec_to_packs(result[_n_d_chunks])
            l_out = result[_n_d_chunks + 1]
            m_out = result[_n_d_chunks + 2]
            return o_out, p_out, l_out, m_out

        o_out, p_out, l_out, m_out = _run(o_accs, m_i[0], l_i[0], tile_max, p_packs)
        return (
            o_out,
            [fx.Float32(m_out)] + [m_i[r] for r in range_constexpr(1, npair)],
            [fx.Float32(l_out)] + [l_i[r] for r in range_constexpr(1, npair)],
            p_out,
        )

    def pv_gemm_register_packs(frag_P_in, v_regs, o_accs):
        """PV GEMM for flattened V packs (n64 padded LDS)."""
        p_packs = _pack_p_b(frag_P_in)
        for pk in range_constexpr(_pv_packs):
            for dc in range_constexpr(_n_d_chunks):
                o_accs[dc] = _mfma_acc(v_regs[pk * _n_d_chunks + dc], p_packs[pk], o_accs[dc], _pv_mma_atom)

    def read_v_mfma_step(slot, pk):
        """One PV pack of V from LDS: 2 transpose steps × all D-chunks."""
        lane = fx.Int32(local_tid % GFX950_WAVE_SIZE)
        lane_div_32 = lane // fx.Int32(32)
        lane_base = lane_div_32 * fx.Int32(2176)
        lane_base = lane_base + ((lane % fx.Int32(16)) // fx.Int32(4)) * fx.Int32(_v_lds_line)
        lane_base = lane_base + ((lane // fx.Int32(16)) % fx.Int32(2)) * fx.Int32(16)
        lane_base = lane_base + (lane % fx.Int32(4)) * fx.Int32(4)
        base_ptr = fx.add_offset(sV_ptr[slot], fx.make_int_tuple(lane_base))
        v_regs = [None] * _n_d_chunks
        for dc in range_constexpr(_n_d_chunks):
            # Non-vectorized V: k-substep stride 128, D pair stride
            # 8*544, inner D-chunk stride 32, transpose pair stride 64.
            off_a = pk * 128 + (dc // 2) * (_lds_n_rpt * _v_lds_line) + (dc % 2) * 32
            off_b = off_a + 64
            src_a = fx.make_view(fx.add_offset(base_ptr, fx.make_int_tuple(off_a)), _v_tr_layout)
            src_b = fx.make_view(fx.add_offset(base_ptr, fx.make_int_tuple(off_b)), _v_tr_layout)
            dst_a = fx.make_rmem_tensor(_v_tr_layout, elem_dtype)
            dst_b = fx.make_rmem_tensor(_v_tr_layout, elem_dtype)
            fx.copy(_v_tr_atom, src_a, dst_a)
            fx.copy(_v_tr_atom, src_b, dst_b)
            v_regs[dc] = Vec(dst_a.load()).shuffle(Vec(dst_b.load()), list(range(8))).ir_value()
        return v_regs

    def read_v_lds(slot):
        """Read all four PV packs using the padded V LDS map."""
        v_regs = [None] * (_pv_packs * _n_d_chunks)
        for pk in range_constexpr(_pv_packs):
            pack = read_v_mfma_step(slot, pk)
            for dc in range_constexpr(_n_d_chunks):
                v_regs[pk * _n_d_chunks + dc] = pack[dc]
        return v_regs

    def cluster_sync(cluster_index):
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

    def sched_interleave_lds_math(groups, *, ds, trans=0, valu=0, sync_id=0):
        """Alternate LDS reads with softmax arithmetic in a memory cluster.

        The AMDGPU scheduler counts transcendentals (`v_exp_f32`) in their own
        group, so exp2 has to be asked for separately from plain VALU.
        """
        for _ in range_constexpr(groups):
            rocdl.sched_group_barrier(0x100, ds, sync_id)
            if const_expr(trans):
                rocdl.sched_group_barrier(0x400, trans, sync_id)
            if const_expr(valu):
                rocdl.sched_group_barrier(0x002, valu, sync_id)
        rocdl.sched_barrier(0)

    def sched_interleave_qk_sum(groups, *, mfma, valu):
        """Distribute the deferred P reduction through the independent QK GEMM."""
        for _ in range_constexpr(groups):
            rocdl.sched_group_barrier(0x008, mfma, 2)
            rocdl.sched_group_barrier(0x002, valu, 2)

    def sched_interleave_pv_softmax(groups, *, mfma, valu=0, trans=0):
        """Hide the P[n] softmax chain in the PV GEMM's MFMA shadow."""
        for _ in range_constexpr(groups):
            rocdl.sched_group_barrier(0x008, mfma, 3)
            if const_expr(valu):
                rocdl.sched_group_barrier(0x002, valu, 3)
            if const_expr(trans):
                rocdl.sched_group_barrier(0x400, trans, 3)

    def sched_mfma_pairs(pairs, *, valu=0, trans=0, group):
        """MFMA/VALU-or-EXP scheduling recipe for 8c pairs."""
        for _ in range_constexpr(pairs):
            rocdl.sched_group_barrier(0x008, 1, group)
            if const_expr(valu):
                rocdl.sched_group_barrier(0x002, valu, group)
            if const_expr(trans):
                rocdl.sched_group_barrier(0x400, trans, group)

    _stagger_div = _flex_stagger_divisor(int(param.block_threads))
    infra = _InfraContext()
    infra.stagger_i32 = arith.divsi(_wave_id_uni_s, as_mlir_value(fx.Int32(_stagger_div)))

    # Paged: stage the block table in LDS.
    if const_expr(_paged):
        _bt_lds_ptr = storage.bt.peek().ptr
        _bt_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        _bt_flat = fx.rocdl.make_buffer_tensor(
            fx.Tensor(
                fx.make_view(fx.recast_iter(fx.Int32, fx.get_iter(block_table)), fx.make_layout(_MAX_BUFFER_BYTES, 1))
            ),
            max_size=True,
        )
        _bt_div = fx.logical_divide(_bt_flat, fx.make_layout(1, 1))
        _bt_batch_off = _idx_to_i32(b_idx) * block_table_stride
        _bt_entries = n_kv_tiles
        for _bt_pass in range_constexpr((_PAGED_BT_LDS_SIZE + block_threads - 1) // block_threads):
            _bt_local = fx.Int32(_bt_pass) * fx.Int32(block_threads) + fx.Int32(tid)
            _bt_in_range = _bt_local < _bt_entries
            _bt_global = _bt_batch_off + _bt_local
            _bt_lds_dst = fx.add_offset(fx.recast_iter(fx.Int32, _bt_lds_ptr), fx.make_int_tuple(_bt_local))
            if _bt_in_range:
                fx.copy(
                    _bt_copy_atom,
                    fx.slice(_bt_div, (None, fx.Int32(_bt_global))),
                    fx.make_view(_bt_lds_dst, fx.make_layout(1, 1)),
                )
            else:
                fx.ptr_store(fx.Int32(0), _bt_lds_dst)
        rocdl.s_waitcnt(0)
        rocdl.s_barrier()

        def _load_page_id(tile_idx):
            _lds_i32 = fx.recast_iter(fx.Int32, _bt_lds_ptr)
            _raw = fx.ptr_load(
                fx.add_offset(_lds_i32, fx.make_int_tuple(tile_idx)),
                result_type=fx.Int32,
            )
            rocdl.s_waitcnt(lgkmcnt=0)
            return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, _raw))

    else:

        def _load_page_id(tile_idx):
            return fx.Int32(0)

    _lower_tri_full_tile_8c = bool(param.packed_lower_tri_mask) and bool(param.long_seq_8c)
    if const_expr(_lower_tri_full_tile_8c):
        # Tile t is fully visible to every row in the WG iff
        # t*block_n + block_n-1 <= q_min.  Therefore the number of full tiles
        # before the diagonal is floor((q_min+1)/block_n).  Clamp it to this
        # WG's live range; _kv_hi already excludes wholly masked future tiles.
        _kv_full_hi_raw = (_q_min_wg + fx.Int32(1)) // fx.Int32(block_n)
        _kv_full_hi = _i32_min(
            _i32_max(_kv_full_hi_raw, _kv_lo),
            _kv_hi,
        )
    else:
        _kv_full_hi = _kv_hi
    rocdl.s_barrier()
    rocdl.s_barrier()

    # Unrolled-by-2 KV loop with split LDS globals for compile-time slot
    # selection.  Each iteration does: read K/V → QK GEMM → softmax →
    # DMA next tile → PV GEMM.  DMA and LDS reads target separate per-slot
    # globals so LLVM can prove non-aliasing.

    o_accs = o_accs_init
    _o = 2 * npair

    _p_half = n_c // 2

    def _apply_mods_and_mask(s_raw, kv_i32, *, invalid=False, valid=None):
        if const_expr(mod_has_score or mod_has_mask):
            apply_mods(s_raw, kv_i32)
        if const_expr(invalid):
            neg_inf = fx.Float32(-1e9)
            s_raw = [valid.select(s_raw[i], neg_inf) for i in range_constexpr(n_c)]
        return s_raw

    def _finish_deferred_exp(p_mixed_vec):
        """C0: first half is P; second half still holds (S-m)."""
        p_mixed = Vec(p_mixed_vec)
        p_out = [p_mixed[i] for i in range_constexpr(_p_half)]
        p_out += [_hw_exp2(p_mixed[_p_half + i]) for i in range_constexpr(n_c - _p_half)]
        return p_out

    def _add_deferred_sum(p_prev, l_i):
        """C1: add sum(P[n-1]); l was already rescaled in prior C2."""
        p_vec = Vec.from_elements(p_prev, fx.Float32)
        local_sum = p_vec.reduce("add", init_val=fx.Float32(0.0), fastmath=_FM)
        local_sum = _permlane32_reduce(local_sum, "sum")
        l_new = l_i[0] + local_sum
        return [l_new] + [l_i[r] for r in range_constexpr(1, npair)]

    def _rescale_for_current(s_raw, m_i, l_i, o_accs, p_prev):
        """C2: scale S[n]; O/l rebase happens in C3 after PV pack 0."""
        s_scaled = [s_raw[i] for i in range_constexpr(n_c)]
        return s_scaled, m_i, l_i, o_accs, p_prev, fx.Float32(1.0)

    def _pin_in_cluster(values):
        """Give pure VALU results a use here so they cannot sink downstream."""
        vals = [as_mlir_value(v) for v in values]
        llvm.inline_asm(
            None,
            vals,
            "",
            ",".join("v" for _ in vals),
            has_side_effects=True,
        )

    def _make_deferred_p(s_scaled, m_i):
        """C3: begin P[n], leaving its second half for next C0."""
        m_new = m_i[0]
        shifted = [s_scaled[i] - m_new for i in range_constexpr(n_c)]
        p_head = [_hw_exp2(shifted[i]) for i in range_constexpr(_p_half)]
        p_mixed = p_head + [shifted[i] for i in range_constexpr(_p_half, n_c)]
        return Vec.from_elements(p_mixed, fx.Float32).ir_value(), p_head

    def _deferred_prologue(kv_i32, m_i, l_i, o_accs):
        """Prime S/P for tile 0; K0/V0 are already resident in slot 0."""
        rocdl.s_waitcnt(vmcnt=0)
        rocdl.s_barrier()
        k_regs = read_k_lds(0)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        cluster_sync(0)

        (frag_S,) = gemm1_qk_unrolled(frag_Q, k_regs)
        s_raw = [frag_S[i] for i in range_constexpr(n_c)]
        s_raw = _apply_mods_and_mask(s_raw, kv_i32)
        cluster_sync(1)

        has_next = (kv_i32 + fx.Int32(1)) < _kv_hi
        if has_next:
            load_k(kv_i32 + fx.Int32(1), 1)
        # There is no pending P yet; only establish tile-0 max/state.
        corr_scalar, s_scaled, m_new = softmax_start(s_raw, m_i)
        m_i = [m_new] + [m_i[r] for r in range_constexpr(1, npair)]
        l_i = [l_i[0] * corr_scalar] + [l_i[r] for r in range_constexpr(1, npair)]
        # K[1] must be complete before this cross-wave barrier publishes slot
        # 1. Leaving its two VMEM operations outstanding makes the first cold
        # launch race the first steady-step LDS read.
        rocdl.s_waitcnt(vmcnt=0)
        cluster_sync(2)

        p_mixed, p_head = _make_deferred_p(s_scaled, m_i)
        _pin_in_cluster(p_head)
        cluster_sync(3)
        return m_i, l_i, o_accs, p_mixed, corr_scalar

    def _deferred_step(kv_i32, cur_slot, prev_slot, m_i, l_i, o_accs, p_mixed_prev, corr_pending):
        """Steady tile n: QK/P preparation for n while consuming PV[n-1]."""
        # C0: launch V[n], issue K[n] LDS, then fill LDS latency with exp(P[n-1]).
        rocdl.s_waitcnt(vmcnt=0)
        load_v(kv_i32, cur_slot)
        k_regs = read_k_lds(cur_slot)
        p_prev = _finish_deferred_exp(p_mixed_prev)
        sched_interleave_lds_math(_qk_k_reps, ds=2, trans=1, sync_id=0)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(_dma_ops_per_thread)
        cluster_sync(0)

        # C1: independent QK[n] and sum(P[n-1]).
        (frag_S,) = gemm1_qk_unrolled(frag_Q, k_regs)
        s_raw = [frag_S[i] for i in range_constexpr(n_c)]
        s_raw = _apply_mods_and_mask(s_raw, kv_i32)
        l_i = _add_deferred_sum(p_prev, l_i)
        # One requested VALU expands to roughly two ISA ops, spreading the
        # reduction across all 16 QK MFMAs without leaving an MFMA-only tail.
        sched_interleave_qk_sum(16, mfma=1, valu=1)
        cluster_sync(1)

        # C2: launch K[n+1], issue V[n-1] LDS, then max/rescale for S[n].
        has_next = (kv_i32 + fx.Int32(1)) < _kv_hi
        if has_next:
            load_k(kv_i32 + fx.Int32(1), prev_slot)
        v_prev = read_v_lds(prev_slot)
        s_scaled, m_i, l_i, o_accs, p_prev, corr_scalar = _rescale_for_current(s_raw, m_i, l_i, o_accs, p_prev)
        sched_interleave_lds_math(_pv_packs * 2, ds=_n_d_chunks, valu=2, sync_id=1)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(_dma_ops_per_thread)
        cluster_sync(2)

        # C3: consume P[n-1] at the old max, then lazily rescale the combined
        # accumulator in the next C0 (or here, just before PV).
        rocdl.s_setprio(1)
        p_packs = _pack_p_b(p_prev)
        for dc in range_constexpr(_n_d_chunks):
            o_accs[dc] = _mfma_acc(v_prev[dc], p_packs[0], o_accs[dc], _pv_mma_atom)
        tile_max = s_scaled[0]
        for i in range_constexpr(1, n_c):
            tile_max = _f32_max(tile_max, s_scaled[i])
        tile_max = _permlane32_reduce(tile_max, "max")
        o_accs, m_i, l_i, p_packs = _c3_lazy_rescale(o_accs, m_i, l_i, tile_max, p_packs)
        for pk in range_constexpr(1, _pv_packs):
            for dc in range_constexpr(_n_d_chunks):
                o_accs[dc] = _mfma_acc(
                    v_prev[pk * _n_d_chunks + dc],
                    p_packs[pk],
                    o_accs[dc],
                    _pv_mma_atom,
                )
        p_mixed, p_head = _make_deferred_p(s_scaled, m_i)
        sched_interleave_pv_softmax(_pv_packs * _n_d_chunks, mfma=1, trans=1)
        _pin_in_cluster(p_head)
        rocdl.s_setprio(0)
        cluster_sync(3)
        return m_i, l_i, o_accs, p_mixed, corr_scalar

    def _deferred_step_8c(kv_odd, m_i, l_i, o_accs, p_mixed_prev):
        """C0..C7 pipeline for two consecutive KV tiles."""
        kv_even = kv_odd + fx.Int32(1)

        # C0: V[odd] DMA and resident K[odd] read. K[even] is already in slot 0.
        llvm.inline_asm(None, [], "s_nop 7", "", has_side_effects=True)
        rocdl.sched_barrier(0)
        load_v(kv_odd, 1)
        # K in slot 1 was staged earlier; leave only this cluster's V DMA
        # outstanding before issuing its LDS read.
        _waitcnt_vm_n(_dma_ops_per_thread)
        k_odd = read_k_lds(1)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(2 * _dma_ops_per_thread)
        cluster_sync(0)

        # C1: QK[odd], finish/sum P[odd-1], and carry packed P into C3.
        (frag_s_odd,) = gemm1_qk_unrolled(frag_Q, k_odd)
        s_odd = [frag_s_odd[i] for i in range_constexpr(n_c)]
        if const_expr(_lower_tri_full_tile_8c and mod_has_score):
            apply_score_mods(s_odd, kv_odd)
        elif const_expr(not _lower_tri_full_tile_8c):
            s_odd = _apply_mods_and_mask(s_odd, kv_odd)
        p_prev = _finish_deferred_exp(p_mixed_prev)
        l_i = _add_deferred_sum(p_prev, l_i)
        p_prev_packs = _anchor_p_packs(_pack_p_b(p_prev))
        sched_mfma_pairs(6, trans=3, group=1)
        sched_mfma_pairs(10, valu=5, group=1)
        cluster_sync(1)

        # C2: prefetch K[odd+2] into the consumed odd slot and read V[odd-1].
        llvm.inline_asm(None, [], "s_nop 7", "", has_side_effects=True)
        rocdl.sched_barrier(0)
        if (kv_odd + fx.Int32(2)) < _kv_hi:
            load_k(kv_odd + fx.Int32(2), 1)
        v_prev = read_v_lds(0)
        s_odd, m_i, l_i, o_accs, _p_prev, _corr = _rescale_for_current(s_odd, m_i, l_i, o_accs, p_prev)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(2 * _dma_ops_per_thread)
        cluster_sync(2)

        # C3: PV[odd-1], lazy rebase from S[odd], then begin P[odd].
        rocdl.s_setprio(1)
        for dc in range_constexpr(_n_d_chunks):
            o_accs[dc] = _mfma_acc(v_prev[dc], p_prev_packs[0], o_accs[dc], _pv_mma_atom)
        tile_max_odd = s_odd[0]
        for i in range_constexpr(1, n_c):
            tile_max_odd = _f32_max(tile_max_odd, s_odd[i])
        tile_max_odd = _permlane32_reduce(tile_max_odd, "max")
        sched_mfma_pairs(4, valu=6, group=2)
        o_accs, m_i, l_i, p_prev_packs = _c3_lazy_rescale(o_accs, m_i, l_i, tile_max_odd, p_prev_packs)
        for pk in range_constexpr(1, _pv_packs):
            for dc in range_constexpr(_n_d_chunks):
                o_accs[dc] = _mfma_acc(
                    v_prev[pk * _n_d_chunks + dc],
                    p_prev_packs[pk],
                    o_accs[dc],
                    _pv_mma_atom,
                )
        p_mixed_odd, p_head_odd = _make_deferred_p(s_odd, m_i)
        sched_mfma_pairs(6, valu=6, group=2)
        sched_mfma_pairs(6, trans=3, group=2)
        _pin_in_cluster(p_head_odd)
        rocdl.s_setprio(0)
        cluster_sync(3)

        # C4: V[even] DMA and resident K[even] read.
        llvm.inline_asm(None, [], "s_nop 7", "", has_side_effects=True)
        rocdl.sched_barrier(0)
        load_v(kv_even, 0)
        # K in slot 0 was staged earlier; leave only this cluster's V DMA
        # outstanding before issuing its LDS read.
        _waitcnt_vm_n(_dma_ops_per_thread)
        k_even = read_k_lds(0)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(2 * _dma_ops_per_thread)
        cluster_sync(4)

        # C5: QK[even], finish/sum P[odd], and carry packed P into C7.
        (frag_s_even,) = gemm1_qk_unrolled(frag_Q, k_even)
        s_even = [frag_s_even[i] for i in range_constexpr(n_c)]
        if const_expr(_lower_tri_full_tile_8c and mod_has_score):
            apply_score_mods(s_even, kv_even)
        elif const_expr(not _lower_tri_full_tile_8c):
            s_even = _apply_mods_and_mask(s_even, kv_even)
        p_odd = _finish_deferred_exp(p_mixed_odd)
        l_i = _add_deferred_sum(p_odd, l_i)
        p_odd_packs = _anchor_p_packs(_pack_p_b(p_odd))
        sched_mfma_pairs(6, trans=3, group=3)
        sched_mfma_pairs(10, valu=5, group=3)
        cluster_sync(5)

        # C6: prefetch K[even+2] into the consumed even slot and read V[odd].
        llvm.inline_asm(None, [], "s_nop 7", "", has_side_effects=True)
        rocdl.sched_barrier(0)
        if (kv_even + fx.Int32(2)) < _kv_hi:
            load_k(kv_even + fx.Int32(2), 0)
        v_odd = read_v_lds(1)
        s_even, m_i, l_i, o_accs, _p_odd, corr_scalar = _rescale_for_current(s_even, m_i, l_i, o_accs, p_odd)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(2 * _dma_ops_per_thread)
        cluster_sync(6)

        # C7: PV[odd], lazy rebase from S[even], then begin P[even].
        rocdl.s_setprio(1)
        for dc in range_constexpr(_n_d_chunks):
            o_accs[dc] = _mfma_acc(v_odd[dc], p_odd_packs[0], o_accs[dc], _pv_mma_atom)
        tile_max_even = s_even[0]
        for i in range_constexpr(1, n_c):
            tile_max_even = _f32_max(tile_max_even, s_even[i])
        tile_max_even = _permlane32_reduce(tile_max_even, "max")
        sched_mfma_pairs(4, valu=6, group=4)
        o_accs, m_i, l_i, p_odd_packs = _c3_lazy_rescale(o_accs, m_i, l_i, tile_max_even, p_odd_packs)
        for pk in range_constexpr(1, _pv_packs):
            for dc in range_constexpr(_n_d_chunks):
                o_accs[dc] = _mfma_acc(
                    v_odd[pk * _n_d_chunks + dc],
                    p_odd_packs[pk],
                    o_accs[dc],
                    _pv_mma_atom,
                )
        p_mixed_even, p_head_even = _make_deferred_p(s_even, m_i)
        sched_mfma_pairs(6, valu=5, group=4)
        sched_mfma_pairs(6, trans=3, group=4)
        _pin_in_cluster(p_head_even)
        rocdl.s_setprio(0)
        cluster_sync(7)
        return m_i, l_i, o_accs, p_mixed_even, corr_scalar

    def _deferred_epilogue(last_slot, m_i, l_i, o_accs, p_mixed, corr_pending):
        """Drain the final pending P/V tile."""
        p_last = _finish_deferred_exp(p_mixed)
        cluster_sync(0)
        l_i = _add_deferred_sum(p_last, l_i)
        cluster_sync(1)
        v_last = read_v_lds(last_slot)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        cluster_sync(2)
        pv_gemm_register_packs(p_last, v_last, o_accs)
        cluster_sync(3)
        return m_i, l_i, o_accs

    # Pad tile count to pairs: ceil(range / 2). When range is odd the
    # last pair's odd tile lands past _kv_hi — has_next=False skips the
    # DMA, and the QK/softmax on stale LDS is harmless because the NEXT
    # iteration (which would consume those scores) never runs.
    # For a 1-tile range the loop executes once: even is the real tile,
    # odd is a no-op past _kv_hi whose scores never feed a subsequent PV.
    if const_expr(_SPLITK):
        _kv_range = _split_tiles
    else:
        _kv_range = _kv_hi - _kv_lo
    _kv_pairs = (_kv_range + fx.Int32(1)) // fx.Int32(2)

    # n64 4c deferred softmax; 8c when long_seq_8c.
    # Cross-tile pipeline. Each main step consumes PV[n-1] while preparing
    # scores/P[n], distributing softmax across both LDS and MFMA clusters.
    if const_expr(_SPLITK):
        _prologue_state = (
            [as_mlir_value(m_i[r]) for r in range_constexpr(npair)]
            + [as_mlir_value(l_i[r]) for r in range_constexpr(npair)]
            + [as_mlir_value(o_accs[dc]) for dc in range_constexpr(_n_d_chunks)]
            + [
                as_mlir_value(Vec.filled(n_c, 0.0, fx.Float32)),
                as_mlir_value(fx.Float32(1.0)),
            ]
        )
        _prologue_names = tuple(
            [f"_split_m{r}" for r in range(npair)]
            + [f"_split_l{r}" for r in range(npair)]
            + [f"_split_o{dc}" for dc in range(_n_d_chunks)]
            + ["_split_p", "_split_corr"]
        )

        def _run_nonempty_prologue(_n, *_st):
            load_kv(_kv_lo, 0)
            rocdl.s_waitcnt(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            _stagger_extra_barrier_if_one(infra.stagger_i32)
            _m, _l, _o, _p, _corr = _deferred_prologue(_kv_lo, m_i, l_i, o_accs)
            return (
                [as_mlir_value(_m[r]) for r in range_constexpr(npair)]
                + [as_mlir_value(_l[r]) for r in range_constexpr(npair)]
                + [as_mlir_value(_o[dc]) for dc in range_constexpr(_n_d_chunks)]
                + [as_mlir_value(_p), as_mlir_value(_corr)]
            )

        _prologue_result = list(
            scf_if_dispatch(
                as_mlir_value(_split_nonempty),
                _run_nonempty_prologue,
                lambda *_a: None,
                state_names=_prologue_names,
                state_values=_prologue_state,
            )
        )
        m_i = [_prologue_result[r] for r in range_constexpr(npair)]
        l_i = [_prologue_result[npair + r] for r in range_constexpr(npair)]
        o_accs = [_prologue_result[2 * npair + dc] for dc in range_constexpr(_n_d_chunks)]
        p_mixed = _prologue_result[2 * npair + _n_d_chunks]
        corr_pending = _prologue_result[2 * npair + _n_d_chunks + 1]
    else:
        load_kv(_kv_lo, 0)
        rocdl.s_waitcnt(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)
        _stagger_extra_barrier_if_one(infra.stagger_i32)
        m_i, l_i, o_accs, p_mixed, corr_pending = _deferred_prologue(_kv_lo, m_i, l_i, o_accs)
    if const_expr(param.long_seq_8c):
        # Keep K two tiles ahead: K0/K1 are primed by the prologue,
        # and K2 is launched before C0 starts consuming K1.
        # For lower-tri full-tile splitting, only prime K2 when an 8c pair
        # actually exists.  Otherwise the first 4c tail step will stage it.
        if (_kv_lo + fx.Int32(2)) < _kv_full_hi:
            load_k(_kv_lo + fx.Int32(2), 0)
            # Retire the prologue's K[lo+1] before the first C0 reads it,
            # leaving only this K[lo+2] in flight. Later trips are ordered
            # by the memory clusters' own closing waits. A shorter range
            # than this leaves _steady_pairs at 0, so no C0 read happens.

    # Dense keeps the old all-8c range.  Bottom-right lower-tri stops 8c at
    # _kv_full_hi, so C1/C5 never need a mask predicate.  The following
    # 4c loop consumes the diagonal band without draining p_mixed.
    _full_remaining = _kv_full_hi - (_kv_lo + fx.Int32(1))
    _full_remaining = _i32_max(_full_remaining, fx.Int32(0))
    _steady_pairs = _full_remaining // fx.Int32(2)
    _p_arg = _o + _n_d_chunks
    _corr_arg = _p_arg + 1
    init_args = (
        [m_i[r] for r in range_constexpr(npair)]
        + [l_i[r] for r in range_constexpr(npair)]
        + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
        + [p_mixed, corr_pending]
    )
    pair_results = init_args
    for pair_i, pair_args in range(
        fx.Int32(0),
        _steady_pairs,
        fx.Int32(1),
        init=init_args,
    ):
        m_i = [pair_args[r] for r in range_constexpr(npair)]
        l_i = [pair_args[npair + r] for r in range_constexpr(npair)]
        o_accs = [pair_args[_o + dc] for dc in range_constexpr(_n_d_chunks)]
        p_mixed = pair_args[_p_arg]
        corr_pending = pair_args[_corr_arg]
        kv_odd = _kv_lo + fx.Int32(1) + _idx_to_i32(pair_i) * fx.Int32(2)
        if const_expr(param.long_seq_8c):
            m_i, l_i, o_accs, p_mixed, corr_pending = _deferred_step_8c(kv_odd, m_i, l_i, o_accs, p_mixed)
        else:
            m_i, l_i, o_accs, p_mixed, corr_pending = _deferred_step(
                kv_odd, 1, 0, m_i, l_i, o_accs, p_mixed, corr_pending
            )
            m_i, l_i, o_accs, p_mixed, corr_pending = _deferred_step(
                kv_odd + fx.Int32(1), 0, 1, m_i, l_i, o_accs, p_mixed, corr_pending
            )
        pair_results = yield (
            [m_i[r] for r in range_constexpr(npair)]
            + [l_i[r] for r in range_constexpr(npair)]
            + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
            + [p_mixed, corr_pending]
        )

    m_i = [pair_results[r] for r in range_constexpr(npair)]
    l_i = [pair_results[npair + r] for r in range_constexpr(npair)]
    o_accs = [pair_results[_o + dc] for dc in range_constexpr(_n_d_chunks)]
    p_mixed = pair_results[_p_arg]
    corr_pending = pair_results[_corr_arg]

    # 8c pairs leave the pending even tile in slot 0 and the next K in
    # slot 1.  Consume the rest (an optional unpaired full tile followed by
    # the lower-tri diagonal) as ordinary 4c pairs.  apply_mods cheaply skips
    # the optional full tile and masks each diagonal tile before softmax.
    _tail_start = _kv_lo + fx.Int32(1) + _steady_pairs * fx.Int32(2)
    _tail_remaining = _kv_hi - _tail_start
    _tail_pairs = _tail_remaining // fx.Int32(2)
    tail_pair_init = (
        [m_i[r] for r in range_constexpr(npair)]
        + [l_i[r] for r in range_constexpr(npair)]
        + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
        + [p_mixed, corr_pending]
    )
    tail_pair_results = tail_pair_init
    for tail_pair_i, tail_pair_args in range(
        fx.Int32(0),
        _tail_pairs,
        fx.Int32(1),
        init=tail_pair_init,
    ):
        m_i = [tail_pair_args[r] for r in range_constexpr(npair)]
        l_i = [tail_pair_args[npair + r] for r in range_constexpr(npair)]
        o_accs = [tail_pair_args[_o + dc] for dc in range_constexpr(_n_d_chunks)]
        p_mixed = tail_pair_args[_p_arg]
        corr_pending = tail_pair_args[_corr_arg]
        kv_tail_even = _tail_start + _idx_to_i32(tail_pair_i) * fx.Int32(2)
        m_i, l_i, o_accs, p_mixed, corr_pending = _deferred_step(
            kv_tail_even, 1, 0, m_i, l_i, o_accs, p_mixed, corr_pending
        )
        m_i, l_i, o_accs, p_mixed, corr_pending = _deferred_step(
            kv_tail_even + fx.Int32(1),
            0,
            1,
            m_i,
            l_i,
            o_accs,
            p_mixed,
            corr_pending,
        )
        tail_pair_results = yield (
            [m_i[r] for r in range_constexpr(npair)]
            + [l_i[r] for r in range_constexpr(npair)]
            + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
            + [p_mixed, corr_pending]
        )

    m_i = [tail_pair_results[r] for r in range_constexpr(npair)]
    l_i = [tail_pair_results[npair + r] for r in range_constexpr(npair)]
    o_accs = [tail_pair_results[_o + dc] for dc in range_constexpr(_n_d_chunks)]
    p_mixed = tail_pair_results[_p_arg]
    corr_pending = tail_pair_results[_corr_arg]

    # An odd tail leaves one tile in slot 1. Process and drain it.
    _tail_count = _tail_remaining - _tail_pairs * fx.Int32(2)
    tail_init = (
        [m_i[r] for r in range_constexpr(npair)]
        + [l_i[r] for r in range_constexpr(npair)]
        + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
        + [p_mixed, corr_pending]
    )
    tail_results = tail_init
    for _tail_i, tail_args in range(
        fx.Int32(0),
        _tail_count,
        fx.Int32(1),
        init=tail_init,
    ):
        m_i = [tail_args[r] for r in range_constexpr(npair)]
        l_i = [tail_args[npair + r] for r in range_constexpr(npair)]
        o_accs = [tail_args[_o + dc] for dc in range_constexpr(_n_d_chunks)]
        p_mixed = tail_args[_p_arg]
        corr_pending = tail_args[_corr_arg]
        kv_tail = _tail_start + _tail_pairs * fx.Int32(2)
        m_i, l_i, o_accs, p_mixed, corr_pending = _deferred_step(kv_tail, 1, 0, m_i, l_i, o_accs, p_mixed, corr_pending)
        m_i, l_i, o_accs = _deferred_epilogue(1, m_i, l_i, o_accs, p_mixed, corr_pending)
        tail_results = yield (
            [m_i[r] for r in range_constexpr(npair)]
            + [l_i[r] for r in range_constexpr(npair)]
            + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
            + [p_mixed, corr_pending]
        )

    m_i = [tail_results[r] for r in range_constexpr(npair)]
    l_i = [tail_results[npair + r] for r in range_constexpr(npair)]
    o_accs = [tail_results[_o + dc] for dc in range_constexpr(_n_d_chunks)]
    p_mixed = tail_results[_p_arg]
    corr_pending = tail_results[_corr_arg]

    # No tail means the final pending tile is in slot 0 (including 1 tile).
    if const_expr(_SPLITK):
        _drain_pending = _split_nonempty & (_tail_count == fx.Int32(0))
    else:
        _drain_pending = _tail_count == fx.Int32(0)
    _no_tail_count = _drain_pending.select(fx.Int32(1), fx.Int32(0))
    drain_init = (
        [m_i[r] for r in range_constexpr(npair)]
        + [l_i[r] for r in range_constexpr(npair)]
        + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
        + [p_mixed, corr_pending]
    )
    drain_results = drain_init
    for _drain_i, drain_args in range(
        fx.Int32(0),
        _no_tail_count,
        fx.Int32(1),
        init=drain_init,
    ):
        m_i = [drain_args[r] for r in range_constexpr(npair)]
        l_i = [drain_args[npair + r] for r in range_constexpr(npair)]
        o_accs = [drain_args[_o + dc] for dc in range_constexpr(_n_d_chunks)]
        p_mixed = drain_args[_p_arg]
        corr_pending = drain_args[_corr_arg]
        m_i, l_i, o_accs = _deferred_epilogue(0, m_i, l_i, o_accs, p_mixed, corr_pending)
        drain_results = yield (
            [m_i[r] for r in range_constexpr(npair)]
            + [l_i[r] for r in range_constexpr(npair)]
            + [o_accs[dc] for dc in range_constexpr(_n_d_chunks)]
            + [p_mixed, corr_pending]
        )

    m_i = [drain_results[r] for r in range_constexpr(npair)]
    l_i = [drain_results[npair + r] for r in range_constexpr(npair)]
    o_accs = [drain_results[_o + dc] for dc in range_constexpr(_n_d_chunks)]

    if const_expr(_SPLITK):

        @flyc.jit
        def _close_stagger_if_nonempty():
            if _split_nonempty:
                _stagger_extra_barrier_if_zero(infra.stagger_i32)

        _close_stagger_if_nonempty()
    else:
        _stagger_extra_barrier_if_zero(infra.stagger_i32)
    rocdl.s_waitcnt(0)
    rocdl.s_barrier()

    def _normalize_partial(_n, *_st):
        l0 = l_i[0]
        if const_expr(flex_mod.needs_safe_norm):
            l0 = _f32_max(l0, fx.Float32(1e-12))
        l0_ssa = l0.ir_value() if hasattr(l0, "ir_value") else l0
        inv_l = fx.Float32(rocdl.rcp(T.f32, l0_ssa))
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(16)
        return [as_mlir_value((Vec(o_accs[dc]) * inv_l_vec).ir_value()) for dc in range_constexpr(_n_d_chunks)]

    if const_expr(_SPLITK):
        o_accs = list(
            scf_if_dispatch(
                as_mlir_value(_split_nonempty),
                _normalize_partial,
                lambda *_a: None,
                state_names=tuple(f"_split_norm_o{dc}" for dc in range(_n_d_chunks)),
                state_values=[as_mlir_value(o_accs[dc]) for dc in range_constexpr(_n_d_chunks)],
            )
        )
    else:
        o_accs = list(_normalize_partial(0))

    # O store: each thread writes 16 D-values at 1 query-row per D-chunk.
    # C fragment layout (M=D, N=query): lane L has query_row = L%32.
    # Elements [4k..4k+3] map to 4 contiguous D columns at offset 8k,
    # so each group of 4 can be stored as one 64-bit buffer store.
    _qrow = fx.Int32(local_tid % 32)
    _group_d_base = fx.Int32((local_tid // 32) * 4)
    _o_row_stride = hq * head_dim
    _out_elem_dtype = elem_dtype

    if const_expr(_SPLITK):
        # Split-K: write normalized partial O in the output dtype plus f32
        # (m, l).  Packed partials halve workspace traffic versus f32 and use
        # the same fused half-wave stores as the normal output path.
        _ws_o_store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), _out_elem_dtype)
        ws_o_reg = fx.make_rmem_tensor(fx.make_layout(8, 1), _out_elem_dtype)
        ws_o_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(
                        fx.recast_iter(_out_elem_dtype, fx.get_iter(ws_o)),
                        fx.make_layout(_MAX_BUFFER_BYTES, 1),
                    )
                ),
                max_size=True,
            ),
            fx.make_layout(1, 1),
        )
        # ws_o layout: [num_splits, B, Hq, Sq, D] — compute flat offset.
        _ws_sq = seqlen_q
        _ws_o_row_stride = fx.Int32(head_dim)
        _ws_o_head_stride = _ws_sq * _ws_o_row_stride
        _ws_o_batch_stride = hq * _ws_o_head_stride
        _ws_o_split_stride = num_batches * _ws_o_batch_stride
        _ws_o_base = (
            split_idx * _ws_o_split_stride
            + b_idx * _ws_o_batch_stride
            + _idx_to_i32(h_idx) * _ws_o_head_stride
            + (q_start + _qrow) * _ws_o_row_stride
        )
        _lane_div_32 = fx.Index(local_tid // 32)
        for dc in range_constexpr(_n_d_chunks):
            o_vec = Vec(o_accs[dc])
            for g in range_constexpr(2):
                dwords = []
                for sg in range_constexpr(2):
                    k = 2 * g + sg
                    elems = [o_vec[k * 4 + e] for e in range_constexpr(4)]
                    packed = Vec.from_elements(elems, fx.Float32).to(_out_elem_dtype).bitcast(fx.Int32)
                    dwords.extend([packed[0], packed[1]])
                fused = _fused_o_128_dwords(
                    _lane_div_32,
                    dwords[0],
                    dwords[1],
                    dwords[2],
                    dwords[3],
                )
                packed_o = Vec.from_elements([fx.Int32(w) for w in fused], fx.Int32).bitcast(_out_elem_dtype)
                col = dc * 32 + fx.Int32((2 * g) * 8) + fx.Int32(local_tid // 32) * fx.Int32(8)
                off = _ws_o_base + col
                fx.memref_store_vec(packed_o, ws_o_reg)

                @flyc.jit
                def _store_ws_o_if_nonempty():
                    if _split_nonempty:
                        fx.copy(
                            _ws_o_store_atom,
                            ws_o_reg,
                            fx.slice(ws_o_div, (None, fx.Int32(off))),
                        )

                _store_ws_o_if_nonempty()

        # Store (m, l) per query row. Each lane in the wave has the same m_i/l_i
        # (permlane32 reduced), so only one lane per row writes.
        _ws_ml_store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Float32)
        ws_ml_reg = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Float32)
        ws_ml_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(fx.recast_iter(fx.Float32, fx.get_iter(ws_ml)), fx.make_layout(_MAX_BUFFER_BYTES, 1))
                ),
                max_size=True,
            ),
            fx.make_layout(1, 1),
        )
        _ws_ml_row_stride = fx.Int32(2)
        _ws_ml_head_stride = _ws_sq * _ws_ml_row_stride
        _ws_ml_batch_stride = hq * _ws_ml_head_stride
        _ws_ml_split_stride = num_batches * _ws_ml_batch_stride
        _ws_ml_base = (
            split_idx * _ws_ml_split_stride
            + b_idx * _ws_ml_batch_stride
            + _idx_to_i32(h_idx) * _ws_ml_head_stride
            + (q_start + _qrow) * _ws_ml_row_stride
        )
        _ml_m = _split_nonempty.select(m_i[0], fx.Float32(-1e30))
        _ml_l = _split_nonempty.select(l_i[0], fx.Float32(0.0))
        ml_vec = Vec.from_elements([_ml_m, _ml_l], fx.Float32)
        fx.memref_store_vec(ml_vec, ws_ml_reg)
        fx.copy(_ws_ml_store_atom, ws_ml_reg, fx.slice(ws_ml_div, (None, fx.Int32(_ws_ml_base))))
    else:
        # Fuse the complementary 32-lane halves of each wave so every lane
        # issues eight contiguous 128-bit writes instead of sixteen 64-bit
        # writes.  The two halves own alternating 4-element D groups for the
        # same query row, so permlane32 can assemble each contiguous 8-tuple
        # without LDS or a workgroup synchronization.
        _o_store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), _out_elem_dtype)
        o_store_reg = fx.make_rmem_tensor(fx.make_layout(8, 1), _out_elem_dtype)
        o_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(fx.recast_iter(_out_elem_dtype, fx.get_iter(o)), fx.make_layout(_MAX_BUFFER_BYTES, 1))
                ),
                max_size=True,
            ),
            fx.make_layout(1, 1),
        )
        o_base = o_off + _qrow * _o_row_stride
        _lower_tri_row_active = (_idx_to_i32(q_start) + _qrow + _layout_q_offset) >= fx.Int32(0)
        _lane_div_32 = fx.Index(local_tid // 32)
        for dc in range_constexpr(_n_d_chunks):
            o_vec = Vec(o_accs[dc])
            if const_expr(_bottom_right_lower_tri):
                o_vec = Vec.from_elements(
                    [_lower_tri_row_active.select(o_vec[e], fx.Float32(0.0)) for e in range_constexpr(16)]
                )
            for g in range_constexpr(2):
                dwords = []
                for sg in range_constexpr(2):
                    k = 2 * g + sg
                    elems = [o_vec[k * 4 + e] for e in range_constexpr(4)]
                    packed = Vec.from_elements(elems, fx.Float32).to(_out_elem_dtype).bitcast(fx.Int32)
                    dwords.extend([packed[0], packed[1]])
                fused = _fused_o_128_dwords(
                    _lane_div_32,
                    dwords[0],
                    dwords[1],
                    dwords[2],
                    dwords[3],
                )
                vbf = Vec.from_elements([fx.Int32(w) for w in fused], fx.Int32).bitcast(_out_elem_dtype)
                col = dc * 32 + fx.Int32((2 * g) * 8) + fx.Int32(local_tid // 32) * fx.Int32(8)
                off = o_base + col
                fx.memref_store_vec(vbf, o_store_reg)
                fx.copy(_o_store_atom, o_store_reg, fx.slice(o_div, (None, fx.Int32(off))))


_COMBINE_BLOCK = 256


@flyc.kernel(known_block_size=[_COMBINE_BLOCK, 1, 1])
def flex_splitk_combine_kernel(
    o: fx.Tensor,  # [B, Sq, Hq, D] output (bf16/f16)
    ws_o: fx.Tensor,  # [num_splits, B, Hq, Sq, D] bf16/f16 partial O
    ws_ml: fx.Tensor,  # [num_splits, B, Hq, Sq, 2] f32 (m, l) per split
    num_splits: fx.Constexpr[int],
    head_dim: fx.Constexpr[int],
    out_dtype_id: fx.Constexpr[int],
    first_live_q: fx.Int32,
    combine_partials: fx.Constexpr[bool] = True,
):
    """Combine split-K partials, and/or GPU-zero a bottom-right lower-tri prefix.

    Each thread handles 4 D-values at one query row. Block covers
    head_dim/4 lanes × (256 / (head_dim/4)) rows per workgroup.

    ``combine_partials=False`` is the unsplit default: the grid covers only
    ``first_live_q`` fully-masked leading rows and stores zeros. Live rows
    were already written by the main kernel.
    """
    from flydsl.expr.primitive import const_expr, range_constexpr

    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    _lanes_per_row = head_dim // 4
    _rows_per_block = _COMBINE_BLOCK // _lanes_per_row
    lane_in_row = tid % _lanes_per_row
    row_in_block = tid // _lanes_per_row

    compact_row = _idx_to_i32(bid) * fx.Int32(_rows_per_block) + fx.Int32(row_in_block)

    _B = fx.Int32(fx.get_scalar(o.shape[0]))
    _Sq = fx.Int32(fx.get_scalar(o.shape[1]))
    _Hq = fx.Int32(fx.get_scalar(o.shape[2]))
    if const_expr(combine_partials):
        _total_rows = _B * _Hq * _Sq
        _b = compact_row // (_Hq * _Sq)
        _rem = compact_row % (_Hq * _Sq)
        _h = _rem // _Sq
        _sq = _rem % _Sq
    else:
        _q_span = _i32_max(first_live_q, fx.Int32(1))
        _total_rows = _B * _Hq * first_live_q
        _b = compact_row // (_Hq * _q_span)
        _rem = compact_row % (_Hq * _q_span)
        _h = _rem // _q_span
        _sq = _rem % _q_span
    global_row = (_b * _Hq + _h) * _Sq + _sq
    _split_o_stride = _B * _Hq * _Sq * fx.Int32(head_dim)
    _split_ml_stride = _B * _Hq * _Sq * fx.Int32(2)

    d_col = fx.Int32(lane_in_row) * fx.Int32(4)
    _o_off = _b * _Sq * _Hq * fx.Int32(head_dim) + _sq * _Hq * fx.Int32(head_dim) + _h * fx.Int32(head_dim) + d_col
    _out_elem = fx.BFloat16 if const_expr(out_dtype_id == FLEX_DTYPE_BF16) else fx.Float16
    o_it = fx.recast_iter(_out_elem, fx.get_iter(o))
    _in_range = compact_row < _total_rows
    _is_dead_prefix = _sq < first_live_q

    def _store_o_vals(vals):
        for e in range_constexpr(4):
            fx.ptr_store(vals[e].to(_out_elem), o_it + fx.Int32(_o_off + fx.Int32(e)))

    def _combine_live(_n, *_st):
        ws_ml_it = fx.recast_iter(fx.Float32, fx.get_iter(ws_ml))
        _partial_elem = fx.BFloat16 if const_expr(out_dtype_id == FLEX_DTYPE_BF16) else fx.Float16
        ws_o_it = fx.recast_iter(_partial_elem, fx.get_iter(ws_o))
        _M_NEG_INF = fx.Float32(-1e30)
        m_max = _M_NEG_INF
        for s in range_constexpr(num_splits):
            ml_off = fx.Int32(s) * _split_ml_stride + global_row * fx.Int32(2)
            m_s = fx.Float32(fx.ptr_load(ws_ml_it + fx.Int32(ml_off)))
            m_max = _f32_max(m_max, m_s)

        Vec = fx.Vector
        acc = Vec.filled(4, 0.0, fx.Float32)
        den = fx.Float32(0.0)
        for s in range_constexpr(num_splits):
            ml_off = fx.Int32(s) * _split_ml_stride + global_row * fx.Int32(2)
            m_s = fx.Float32(fx.ptr_load(ws_ml_it + fx.Int32(ml_off)))
            l_s = fx.Float32(fx.ptr_load(ws_ml_it + fx.Int32(ml_off + fx.Int32(1))))
            w_s = _hw_exp2(m_s - m_max) * l_s
            o_off = fx.Int32(s) * _split_o_stride + global_row * fx.Int32(head_dim) + d_col
            _zero_o = [as_mlir_value(fx.Float32(0.0)) for _ in range_constexpr(4)]

            def _load_live_o(_n2, *_st2):
                return [
                    as_mlir_value(fx.ptr_load(ws_o_it + fx.Int32(o_off + fx.Int32(e))).to(fx.Float32))
                    for e in range_constexpr(4)
                ]

            o_vals = list(
                scf_if_dispatch(
                    as_mlir_value(l_s > fx.Float32(0.0)),
                    _load_live_o,
                    lambda *_a: None,
                    state_names=tuple(f"_split{s}_o{e}" for e in range(4)),
                    state_values=_zero_o,
                )
            )
            o_vec = Vec.from_elements(o_vals, fx.Float32)
            w_vec = Vec.from_elements([w_s], fx.Float32).broadcast_to(4)
            acc = acc + o_vec * w_vec
            den = den + w_s

        inv_den = fx.Float32(1.0) / _f32_max(den, fx.Float32(1e-12))
        inv_vec = Vec.from_elements([inv_den], fx.Float32).broadcast_to(4)
        result = acc * inv_vec
        _store_o_vals([result[e] for e in range_constexpr(4)])
        return []

    def _store_dead(_n, *_st):
        _store_o_vals([fx.Float32(0.0) for _ in range_constexpr(4)])
        return []

    def _dispatch_row(_n, *_st):
        if const_expr(combine_partials):
            scf_if_dispatch(
                as_mlir_value(_is_dead_prefix),
                _store_dead,
                _combine_live,
                state_names=(),
                state_values=[],
            )
        else:
            _store_dead(0)
        return []

    scf_if_dispatch(
        as_mlir_value(_in_range),
        _dispatch_row,
        lambda *_a: None,
        state_names=(),
        state_values=[],
    )


@flyc.jit
def launch_flex_attn_gfx950(
    o: fx.Tensor,
    q: fx.Tensor,
    k: fx.Tensor,
    v: fx.Tensor,
    scale: fx.Float32,
    param: FlexAttnParam,
    score_mod: fx.Constexpr[Callable],
    mask_mod: fx.Constexpr[Callable],
    stream: fx.Stream = fx.Stream(None),
    ws_o: fx.Tensor = fx.Tensor,
    ws_ml: fx.Tensor = fx.Tensor,
    block_table: fx.Tensor = fx.Tensor,
    block_table_stride: fx.Int32 = fx.Int32(0),
    context_lens: fx.Tensor = fx.Tensor,
    max_seqlen_kv: fx.Int32 = fx.Int32(0),
    kv_bounds: fx.Tensor = fx.Tensor,
):
    b = fx.Int32(fx.get_scalar(q.shape[0]))
    seqlen_q = fx.Int32(fx.get_scalar(q.shape[1]))
    hq = fx.Int32(fx.get_scalar(q.shape[2]))
    _paged = bool(param.paged)
    if const_expr(_paged):
        seqlen_kv = max_seqlen_kv
    else:
        seqlen_kv = fx.Int32(fx.get_scalar(k.shape[1]))

    elem_dtype = _elem_dtype(param.dtype_id)
    _SPLITK = int(param.num_kv_splits) > 1
    _num_kv_splits = int(param.num_kv_splits)

    wave_layout = fx.make_layout((param.m_waves, param.n_waves, 1), (param.n_waves, 1, 0))
    mma_atom_qk = fx.make_mma_atom(fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype))
    tiled_mma_qk = fx.make_tiled_mma(mma_atom_qk, wave_layout)

    rows_per_wg = param.block_m * param.num_groups
    num_q_tiles = (seqlen_q + rows_per_wg - 1) // rows_per_wg

    flex_attn_fwd_gfx950_kernel._known_block_size = [param.block_threads, 1, 1]
    flex_attn_fwd_gfx950_kernel._func.__name__ = make_flex_attn_kernel_name(param)
    _total_waves = int(param.block_threads) // GFX950_WAVE_SIZE
    _waves_per_eu = max(1, _total_waves // 4)

    if const_expr(_SPLITK):
        grid_z = b * fx.Int32(_num_kv_splits)
    else:
        grid_z = b
    _lower_tri_grid = bool(param.lpt_q_grid)
    _first_live_q = fx.Int32(0)
    if const_expr(_lower_tri_grid):
        if const_expr(bool(param.packed_lower_tri_mask) and int(param.q_offset) < 0 and not _paged):
            # Bottom-right lower-tri has max(Sq-Skv, 0) fully-masked leading
            # query rows.  Drop every workgroup wholly contained in that
            # prefix.  Kernel-side reverse mapping still starts from the last
            # full-grid q_tile, so reducing grid.y naturally selects the live
            # tail without another kernel argument.
            _dead_q_rows = seqlen_q - seqlen_kv
            _dead_q_rows = (_dead_q_rows > fx.Int32(0)).select(_dead_q_rows, fx.Int32(0))
            _first_live_q = _dead_q_rows
            _first_live_q_tile = _dead_q_rows // fx.Int32(rows_per_wg)
            _active_q_tiles = num_q_tiles - _first_live_q_tile
            _grid = (hq, _active_q_tiles, grid_z)
        else:
            _grid = (hq, num_q_tiles, grid_z)
    else:
        # Dense (and other non-lower-tri masks): same head-fast mapping as
        # lower-tri so Q tiles of one head share an XCD and replay KV from its L2.
        _grid = (hq, num_q_tiles, grid_z)

    flex_attn_fwd_gfx950_kernel(
        o,
        q,
        k,
        v,
        seqlen_q,
        seqlen_kv,
        b,
        scale,
        tiled_mma_qk,
        param,
        score_mod,
        mask_mod,
        ws_o,
        ws_ml,
        block_table,
        block_table_stride,
        context_lens,
        kv_bounds,
        value_attrs={
            "rocdl.waves_per_eu": _waves_per_eu,
            "rocdl.flat_work_group_size": f"{param.block_threads},{param.block_threads}",
        },
    ).launch(
        grid=_grid,
        block=(param.block_threads, 1, 1),
        stream=stream,
    )

    _lower_tri_prefix_zero = bool(param.packed_lower_tri_mask) and int(param.q_offset) < 0 and not _paged
    if const_expr(_SPLITK) or const_expr(_lower_tri_prefix_zero):
        _head_dim = int(param.head_dim)
        _lanes_per_row = _head_dim // 4
        _rows_per_block = _COMBINE_BLOCK // _lanes_per_row
        _combine_partials = bool(_SPLITK)
        if const_expr(_combine_partials):
            _total_rows = b * hq * seqlen_q
        else:
            _total_rows = b * hq * _first_live_q
        _combine_blocks = (_total_rows + fx.Int32(_rows_per_block - 1)) // fx.Int32(_rows_per_block)
        # HIP rejects a 0-block grid (square lower-tri, no dead prefix).
        _combine_blocks = _i32_max(_combine_blocks, fx.Int32(1))
        flex_splitk_combine_kernel(
            o,
            ws_o,
            ws_ml,
            _num_kv_splits,
            _head_dim,
            int(param.dtype_id),
            _first_live_q,
            _combine_partials,
        ).launch(
            grid=(_combine_blocks, fx.Index(1), fx.Index(1)),
            block=(_COMBINE_BLOCK, 1, 1),
            stream=stream,
        )


# fast_fp_math breaks pipe_depth=2 when seqlen_kv == block_n (single KV tile); omit it.
_flex_attn_compile_hints = {
    "waves_per_eu": 2,
    "unsafe_fp_math": True,
    "llvm_options": {
        "enable-post-misched": False,
        "lsr-drop-solution": True,
    },
}
launch_flex_attn_gfx950.compile_hints = dict(_flex_attn_compile_hints)


def flydsl_flex_attention_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
    num_kv_heads: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    block_m: int = 32,
    block_n: int = 64,
    num_groups: int = 8,
    pipe_depth: int = 1,
    pipe_stages: int = 1,
    accurate_softmax: bool = True,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    infer_score_mod: bool = True,
    num_kv_splits: int = 1,
    stream: Optional[torch.cuda.Stream] = None,
    long_seq_8c: Optional[bool] = None,
) -> torch.Tensor:
    """Flex-attention forward on the layout API (gfx950) with score/mask mods.

    q/k/v: ``[B, S, H, D]`` (BSHD), bf16/f16. Returns ``[B, Sq, Hq, D]``.

    ``score_mod`` remains the source of truth. By default, a callable exactly
    matching ``score + slope * (kv_idx - q_idx)`` may be represented inside
    ``_InspectedMod`` by an equivalent direct log2-space affine lowering.
    Set ``infer_score_mod=False``, ``score_mod.flex_infer=False``, or
    ``score_mod.flex_score_kind="exact"`` to force the callable's exact traced
    operations. PyTorch ignores attributes attached to the function object.
    Non-affine score mods (for example a quadratic relative-position bias)
    always use the exact callable path. Callables must be arithmetic on the
    ``(score, b, h, q, kv)`` arguments; they cannot load globals.

    The KV loop is n64 only: 4-cluster deferred softmax, or 8-cluster
    when ``long_seq_8c`` (dense Skv>=768, masked Skv>=2048). Requires
    ``block_n=64``, ``head_dim=128``, and a 512-thread workgroup.
    """
    arch = get_rocm_arch()
    if not arch.startswith("gfx950"):
        raise RuntimeError(f"flex_attention_layout targets gfx950; got {arch!r}")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q/k/v must be GPU tensors")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q/k/v must share dtype")
    if q.dim() != 4:
        raise ValueError(f"q must be 4D [B,S,H,D], got {q.dim()}D")

    dtype_id = FLEX_DTYPE_FP16 if q.dtype is torch.float16 else FLEX_DTYPE_BF16
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {q.dtype}")

    B, Sq, Hq, D = q.shape
    Skv, Hkv = k.shape[1], k.shape[2]
    if num_kv_heads is not None and num_kv_heads != Hkv:
        raise ValueError(f"num_kv_heads {num_kv_heads} != k head count {Hkv}")
    rows_per_wg = block_m * num_groups
    if Sq % rows_per_wg != 0:
        raise ValueError(f"seqlen_q ({Sq}) must be a multiple of block_m*num_groups ({rows_per_wg})")
    if scale is None:
        scale = 1.0 / (D**0.5)
    flex_mod = inspect_flex_mods(
        score_mod,
        mask_mod,
        seqlen_q=Sq,
        seqlen_kv=Skv,
        num_batches=B,
        num_heads=Hq,
        infer_score_mod=infer_score_mod,
    )
    use_kv_bounds = flex_mod.needs_kv_bounds
    if use_kv_bounds:
        host_kv_bounds = _infer_generic_kv_bounds(
            flex_mod.mask_mod,
            seqlen_q=Sq,
            seqlen_kv=Skv,
            num_batches=B,
            num_heads=Hq,
            rows_per_wg=rows_per_wg,
            block_n=block_n,
        )
        if long_seq_8c is None:
            max_kv_tiles = int((host_kv_bounds[..., 1] - host_kv_bounds[..., 0]).max().item())
            long_seq_8c = max_kv_tiles * block_n >= _LONG_SEQ_8C_SKV_MASKED
        kv_bounds = flex_mod.get_kv_bounds(
            seqlen_q=Sq,
            seqlen_kv=Skv,
            num_batches=B,
            num_heads=Hq,
            rows_per_wg=rows_per_wg,
            block_n=block_n,
            device=q.device,
        )
    else:
        kv_bounds = torch.empty(1, dtype=torch.int32, device=q.device)

    if stream is None:
        stream = torch.cuda.current_stream()
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if pipe_depth >= 2 and num_groups < 2:
        raise ValueError("pipe_depth>=2 requires num_groups>=2 (Strategy A staggered pipeline)")

    effective_kv_splits = int(num_kv_splits)
    if flex_mod.packed_lower_tri_mask and effective_kv_splits > 1:
        effective_kv_splits = _effective_lower_tri_kv_splits(
            requested_splits=effective_kv_splits,
            batch=B,
            seqlen_q=Sq,
            seqlen_kv=Skv,
            num_heads_q=Hq,
            rows_per_wg=rows_per_wg,
            block_n=block_n,
            num_cus=torch.cuda.get_device_properties(q.device).multi_processor_count,
        )

    param = make_flex_attn_param(
        seqlen_kv=Skv,
        dtype_id=dtype_id,
        block_m=block_m,
        block_n=block_n,
        head_dim=D,
        num_heads_q=Hq,
        num_heads_kv=Hkv,
        num_groups=num_groups,
        pipe_depth=pipe_depth,
        pipe_stages=pipe_stages,
        accurate_softmax=accurate_softmax,
        flex_mod=flex_mod,
        num_kv_splits=effective_kv_splits,
        long_seq_8c=long_seq_8c,
        seqlen_q=Sq,
        has_kv_bounds=use_kv_bounds,
    )

    if effective_kv_splits > 1:
        # Every non-empty split writes its complete partial. Avoid hundreds of
        # MiB of memset/fill traffic on every host call.
        ws_o = torch.empty(effective_kv_splits, B, Hq, Sq, D, dtype=q.dtype, device=q.device)
        ws_ml = torch.empty(effective_kv_splits, B, Hq, Sq, 2, dtype=torch.float32, device=q.device)
    else:
        ws_o = torch.empty(1, dtype=torch.float32, device=q.device)
        ws_ml = torch.empty(1, dtype=torch.float32, device=q.device)

    _dummy_bt = torch.empty(1, dtype=torch.int32, device=q.device)
    _dummy_ctx = torch.empty(1, dtype=torch.int32, device=q.device)
    launch_flex_attn_gfx950(
        out.contiguous(),
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        fx.Float32(scale),
        param,
        flex_mod.score_mod,
        flex_mod.mask_mod,
        stream,
        ws_o=ws_o,
        ws_ml=ws_ml,
        block_table=_dummy_bt,
        block_table_stride=fx.Int32(0),
        context_lens=_dummy_ctx,
        max_seqlen_kv=fx.Int32(Skv),
        kv_bounds=kv_bounds,
    )
    return out


def flydsl_flex_attention_layout_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    scale: Optional[float] = None,
    num_kv_heads: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    block_m: int = 32,
    num_groups: int = 8,
    accurate_softmax: bool = True,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    infer_score_mod: bool = True,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Paged-KV-cache flex attention forward (gfx950).

    q: ``[B, Sq, Hq, D]`` bf16/f16.
    k_cache/v_cache: ``[num_blocks, page_size, Hkv, D]`` bf16/f16 (linear layout).
    block_table: ``[B, max_pages_per_seq]`` i32 physical page IDs.
    context_lens: ``[B]`` i32 per-sequence KV context length.
    Returns ``[B, Sq, Hq, D]``.
    """
    arch = get_rocm_arch()
    if not arch.startswith("gfx950"):
        raise RuntimeError(f"flex_attention_layout_paged targets gfx950; got {arch!r}")
    if not (q.is_cuda and k_cache.is_cuda and v_cache.is_cuda):
        raise ValueError("q/k_cache/v_cache must be GPU tensors")
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("q/k_cache/v_cache must share dtype")
    if q.dim() != 4:
        raise ValueError(f"q must be 4D [B,Sq,H,D], got {q.dim()}D")
    if k_cache.dim() != 4:
        raise ValueError(f"k_cache must be 4D [num_blocks,page_size,Hkv,D], got {k_cache.dim()}D")

    dtype_id = FLEX_DTYPE_FP16 if q.dtype is torch.float16 else FLEX_DTYPE_BF16
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {q.dtype}")

    B, Sq, Hq, D = q.shape
    page_size = k_cache.shape[1]
    Hkv = k_cache.shape[2]
    if page_size != 64:
        raise ValueError(f"paged gfx950 flex requires page_size=64 (got {page_size})")
    block_n = page_size

    if num_kv_heads is not None and num_kv_heads != Hkv:
        raise ValueError(f"num_kv_heads {num_kv_heads} != k_cache head count {Hkv}")
    rows_per_wg = block_m * num_groups
    if Sq % rows_per_wg != 0:
        raise ValueError(f"seqlen_q ({Sq}) must be a multiple of block_m*num_groups ({rows_per_wg})")
    if scale is None:
        scale = 1.0 / (D**0.5)
    if stream is None:
        stream = torch.cuda.current_stream()
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)

    max_ctx = int(context_lens.max().item())
    max_seqlen_kv = ((max_ctx + block_n - 1) // block_n) * block_n
    flex_mod = inspect_flex_mods(
        score_mod,
        mask_mod,
        seqlen_q=Sq,
        seqlen_kv=max_seqlen_kv,
        num_batches=B,
        num_heads=Hq,
        paged=True,
        infer_score_mod=infer_score_mod,
    )

    bt_i32 = block_table.to(torch.int32).contiguous().reshape(-1)
    bt_stride = block_table.shape[1]
    ctx_i32 = context_lens.to(torch.int32).contiguous()

    param = make_flex_attn_param(
        seqlen_kv=max_seqlen_kv,
        dtype_id=dtype_id,
        block_m=block_m,
        block_n=block_n,
        head_dim=D,
        num_heads_q=Hq,
        num_heads_kv=Hkv,
        num_groups=num_groups,
        accurate_softmax=accurate_softmax,
        flex_mod=flex_mod,
        paged=True,
        seqlen_q=Sq,
        has_kv_bounds=False,
    )

    ws_o = torch.empty(1, dtype=torch.float32, device=q.device)
    ws_ml = torch.empty(1, dtype=torch.float32, device=q.device)
    _dummy_kv_bounds = torch.empty(1, dtype=torch.int32, device=q.device)

    launch_flex_attn_gfx950(
        out.contiguous(),
        q.contiguous(),
        k_cache.contiguous(),
        v_cache.contiguous(),
        fx.Float32(scale),
        param,
        flex_mod.score_mod,
        flex_mod.mask_mod,
        stream,
        ws_o=ws_o,
        ws_ml=ws_ml,
        block_table=bt_i32,
        block_table_stride=fx.Int32(bt_stride),
        context_lens=ctx_i32,
        max_seqlen_kv=fx.Int32(max_seqlen_kv),
        kv_bounds=_dummy_kv_bounds,
    )
    return out
