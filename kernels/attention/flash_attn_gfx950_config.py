# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Metadata, knobs and traits for the gfx950 attention kernels (forward, dQ, dK/dV).

**This module imports no flydsl and no torch** (stdlib only, enforced by a test):
AOTriton's generator calls it to describe a build before any compiler exists.

Three objects, and the vocabulary is the point:

* **`FmhaInputMetadata`** directly reflects the *inputs*: the real head dims, the
  dtype, and which optional inputs (window, bias, dropout, ALiBi, sink, a paged
  cache) are present. Set by the caller, never by policy.
* **Knobs** (`Gfx950FwdKnobs`, `Gfx950DqKnobs`, `Gfx950DkdvKnobs`) are the
  *constexpr build parameters derived from the inputs*: the padded tile width,
  the tiles, waves and schedule. `None` means "`resolve` decides"; any knob can
  be pinned, by a user, a sweep or AOT, and **`resolve` validates every pin** so
  a pinned knob can be slow but never silently wrong. The derivation and tuning
  rules live in `resolve*` and nowhere else.
* **Traits** (`Gfx950Traits` and friends) are internal constants derived from
  the metadata and the knobs: layout strides, hardware numbers, derived floors.
  Never user-facing, never pinned.

Naming policy: lower_snake means a runtime value or an input description, and
UPPER_CASE means baked into the kernel. The exceptions are the knobs that are
compiler or launch attributes (`num_warps`, `waves_per_eu`, `daz`).
A third class, Leading_upper_snake_case (`Window_left`, `Max_seqlen_q`; clang-tidy's
`readability-identifier-naming` style, AOTriton's `constexpr_or_i32`), is for kernel
parameters that are constexpr in a JIT build but a real argument in an AOT build: one
parameter whose annotation is `fx.Constexpr` when its `STATIC_*` knob is on and `fx.Int32`
otherwise (so at the default knobs, every AOT build, it is a kernarg).

Usage::

    meta = FmhaInputMetadata(dtype_str="bf16", head_dim=100, window=True)
    knobs = fwd_knobs("gfx950", STAGGER=False).resolve(meta)
    traits = fwd_traits(meta, knobs)       # what the builder is made from
    knobs.as_psels()                        # flat scalars, for AOTriton

--- Unsupported combination: bias with a causal or window mask ---------------

**Rejected, by design, everywhere** (`resolve`, the traits functions, the builders and the
interface). A bias already *is* an attention mask: a large negative or `-inf` entry is how
a caller says "do not attend here". A causal or (g)SWA mask on top says the same thing
twice in two vocabularies, with no rule for which wins where they disagree, so the
combination has no defined meaning. AOTriton disables it, PyTorch's math SDPA raises on it
and gfx1201 rejects it in the same words. It is not a missing feature: with the check
removed the kernel produces finite, plausible numbers, which is exactly why the check
stays (a plausible answer to an undefined question is the worst outcome). Fold the causal
pattern into the bias tensor, or drop the bias.

--- Policy versus geometry --------------------------------------------------

Two kinds of thing live here, and the distinction decides whether a change needs
a benchmark or a correctness argument:

* **Policy** (`LADDER`, the family tables, the `_GEOMETRY` table, the occupancy
  rule) is a measured choice; any of it could change without making a build
  incorrect.
* **Geometry** (`tile_width_for`, `staging_shape`, the divisibility rules the
  traits constructor enforces) computes what is *legal*, and changing one can
  make a build invalid.

**The ladder is the design, and the granule is a knob within it.** A head_dim
between two rungs is served by compiling the next rung up and passing the real
extent as a runtime argument, which is what `PADDED_HEAD` records. What decides
the rungs is the *staging granule*: how many D elements one DMA issue covers.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

__all__ = [
    "LADDER",
    "FWD_KERNEL_NAME",
    "DQ_KERNEL_NAME",
    "DKDV_KERNEL_NAME",
    "STAGGER_GROUP_WAVES",
    "GRID_AXIS_HEAD_FASTEST",
    "GRID_AXIS_TILE_FASTEST",
    "FmhaInputMetadata",
    "FmhaHints",
    "Gfx950Traits",
    "Gfx950DqTraits",
    "Gfx950DkdvTraits",
    "Gfx950FwdKnobs",
    "Gfx950DqKnobs",
    "Gfx950DkdvKnobs",
    "fwd_knobs",
    "dq_knobs",
    "dkdv_knobs",
    "fwd_traits",
    "dq_traits",
    "dkdv_traits",
    "tile_width_for",
    "rung_below",
    "traits_cache_key",
    "build_cache_key",
]

# Names the compiler driver looks up in the exported object (one kernel per launcher).
FWD_KERNEL_NAME = "flash_attn_func_gfx950_kernel"
DQ_KERNEL_NAME = "fmha_bwd_dq_gfx950_kernel"
DKDV_KERNEL_NAME = "fmha_bwd_dkdv_gfx950_kernel"

# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

# Compiled tile widths, all measured at `B=4 H=8 S=4096` bf16 non-causal on
# real FLOPs (not the padded tile):
#
#   tile  waves  BLOCK_M  gran  stages  shards   AGPR  spills    LDS   TFLOP/s
#     32    4      128     32     1       1        -     0     17 KB     618
#     64    8      256     64     1       1        0     0     33 KB     889
#    128    8      256     64     1       1        0     0     67 KB    1117
#    160    4      128     32     1       1        -     0     83 KB     917
#    192    4      128     64     1       1      174     0    100 KB     936
#    224    4      128     32     1       1        -     0    116 KB     939
#    256    4      128     64     1       1      192     0    133 KB     940
#    384    4      128     64     2       1      112     0    100 KB     803
#    512    4       64     64     2       2       91     0    133 KB     479
#
# Three families. **A** (granule 64, 8 waves) serves 64 and 128; **B/S**
# (4 waves) serve the rest up to 256, with granule 32 where the width is not a
# multiple of 64; **W** stages the D axis (`D_STAGES`) and shards it
# (`VO_SHARDS`) for 384 and 512, on the separate wide body.
#
# The break at 384 is not a tuning preference, it is LDS: two KV tiles in flight
# need `2 * BLOCK_N * head_dim * ~8.3 B`, which is 199.5 KB at 384 and 266 KB at
# 512 against a 163840 B cap.
#
# **96 is a rung** even though its odd granule-32 shape once computed the wrong
# answer in both bodies; that is fixed, and the test suite covers it.
LADDER = (32, 64, 96, 128, 160, 192, 224, 256, 384, 512)

# The D-axis staging granule when nothing says otherwise: how many bf16 elements
# of one token a single DMA issue covers (a wave moves 512 per issue).
DEFAULT_HEAD_DIM_GRANULE = 64

# ---------------------------------------------------------------------------
# The grid axis order
# ---------------------------------------------------------------------------
#
# **A consumer that computes the launch grid itself needs this, and cannot see
# it.** AOTriton dispatches the hsaco directly and computes the grid in C++, so
# every decision a launcher makes about which quantity lands on which grid axis
# has to travel in the knob set.
#
#   0  HEAD_FASTEST  grid = (head, tile_blocks, batch_or_seq)
#   1  TILE_FASTEST  grid = (tile_blocks, head, batch_or_seq)
#
# "head" is the q head for the forward and dQ, the **kv** head for dK/dV (the GQA
# fold made it so). All three gfx950 launchers are HEAD_FASTEST; gfx1201's dK/dV
# is TILE_FASTEST. gfx950 measured the other order and rejected it: KV-fastest
# was 12-15% slower at every rung, because MI355X's eight XCDs make this an
# L2-locality lever.
GRID_AXIS_HEAD_FASTEST = 0
GRID_AXIS_TILE_FASTEST = 1

# Stagger runs group B one pipeline phase behind group A, which pays only when
# both groups share a SIMD, i.e. at 8 waves. The grouping is `wave_id // 4`
# (not `wave_id // (NUM_WAVES // 2)`), so a 4-wave rung never shifts.
STAGGER_GROUP_WAVES = 4

# Not a knob: trades precision for speed; see FWD-32. Lazy rescale lets P = exp2(S - stale m) exceed 1 before the
# PV MFMA, roughly doubling the error, so it is never on by default (a precision policy, like TF32 for FP32). Read
# through the traits constructor so the FWD-32 hook test can flip it.
_LAZY_RESCALE = False

# The PV MFMA is `v_mfma_f32_32x32x16`, whose output is 32 D columns wide, so
# `D_CHUNKS = head_dim / 32` cannot go below 1 -- an instruction limit.
PV_MFMA_N = 32

# Hardware constants (traits, never knobs): a build that changed one would not be
# this algorithm.
WARP_SIZE = 64
DMA_BYTES = 16
BF16_BYTES = 2
VEC_KV = 8  # bf16 elements one lane moves per DMA issue (16 B)
MFMA_LANE_K = 8
K_STEP_QK = 16  # MFMA K extent
MFMA_M = 32  # MFMA M extent: what pins ROWS_PER_WAVE, whatever BLOCK_M says
D_CHUNK = 32  # PV MFMA N extent: the O accumulator's width
PV_K_STEP = 16
K_SUB_N = 32
LDS_CAP_BYTES = 163840
PAGED_BT_LDS_SIZE = 2048
SCHED_MFMA_MASK = 0x008
SCHED_VALU_MASK = 0x002
SCHED_EXP_MASK = 0x400
NEG_INF_F32_BITS = 0xFF800000
LGKMCNT_0_ONLY = 0xC07F
_RESCALE_THRESHOLD = 8.0  # moot while lazy rescale is off

_RETURN_LSE_MODES = ("runtime", "always", "never")
_KV_CACHE_LAYOUTS = ("linear", "vectorized")
_DTYPES = ("bf16", "f16")


def rung_below(block_dmodel):
    """The widest rung strictly narrower than `block_dmodel`, or 0.

    `tile_width_for` rounds *up* to the first rung that fits, so a build it chose
    serves exactly the half-open range `(rung_below(R), R]`. That lower bound is
    what lets the kernel skip masking the D columns it knows are real. It is a
    property of the ladder rather than of any one build: adding a rung silently
    tightens every wider build's floor, and that is the correct behaviour.
    """
    below = [r for r in LADDER if r < block_dmodel]
    return max(below) if below else 0


def tile_width_for(head_dim):
    """The compiled tile width serving `head_dim` (the first rung that fits), or raise."""
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    for rung in LADDER:
        if head_dim <= rung:
            return rung
    raise ValueError(f"head_dim {head_dim} exceeds the widest tile ({max(LADDER)})")


# ---------------------------------------------------------------------------
# What to compute
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FmhaInputMetadata:
    """The inputs. Set by the caller; never by policy, never by arch."""

    dtype_str: str  # "bf16" | "f16"
    head_dim: int  # the real QK head dim (Triton's head_dim; at runtime: hdim_qk)
    head_dim_v: int | None = None  # the real V head dim; None = head_dim (at runtime: hdim_vo)
    window: bool = False  # a (window_left, window_right) mask is given (causal is a window)
    bias: bool = False  # a (B,H,Sq,Sk) bias tensor is given (+ dB in the backward)
    dropout: bool = False  # dropout is requested
    alibi: bool = False  # ALiBi slopes are given (fp32 [H] / [B,H])
    sink: bool = False  # per-head sink logits are given (fp32 [H])
    paged: bool = False  # K/V come as a block-table cache (page 64)
    kv_cache_layout: str = "linear"  # paged layout: "linear" | "vectorized"; inert unless paged

    @property
    def head_dim_v_real(self):
        return self.head_dim if self.head_dim_v is None else self.head_dim_v


@dataclass(frozen=True)
class FmhaHints:
    """Optimisation inputs, not semantics; 0 means unknown.

    Reserved for shape-aware policy (split-K, XCD swizzle, tile choice). `resolve`
    accepts them so callers do not change when a rule starts using one; no current
    default depends on them.
    """

    seqlen_q: int = 0
    seqlen_k: int = 0
    num_heads: int = 0
    batch: int = 0

    def __post_init__(self):
        for f in fields(self):
            if getattr(self, f.name) < 0:
                raise ValueError(f"FmhaHints.{f.name} must be >= 0 (0 = unknown), got {getattr(self, f.name)}")


def _check_meta(meta):
    """Refuse inputs no gfx950 build serves, at the decision rather than at an address."""
    if meta.dtype_str not in _DTYPES:
        raise NotImplementedError(f"gfx950 attention builds bf16 and f16, got {meta.dtype_str!r}")
    if meta.head_dim <= 0 or (meta.head_dim_v is not None and meta.head_dim_v <= 0):
        raise ValueError(f"head dims must be positive, got head_dim={meta.head_dim} head_dim_v={meta.head_dim_v}")
    if meta.kv_cache_layout not in _KV_CACHE_LAYOUTS:
        raise ValueError(f"kv_cache_layout must be one of {_KV_CACHE_LAYOUTS}, got {meta.kv_cache_layout!r}")
    if meta.bias and meta.window:
        # Undefined, not unimplemented: causal is an attention mask with a fixed pattern, and a bias
        # *is* an attention mask supplied directly (a -inf entry is how a caller spells "do not attend
        # here"). Asking for both asks which wins where they disagree. AOTriton disables the
        # combination and PyTorch's math backend raises on it.
        raise ValueError(
            "bias and window/causal masking are mutually exclusive: a bias already is an attention mask, "
            "so combining it with a positional one has no defined meaning. Fold the causal pattern into "
            "the bias tensor, or drop the bias"
        )


# ---------------------------------------------------------------------------
# Traits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gfx950Traits:
    """Compile-time tile/layout constants for the gfx950 dual-wave kernels.

    Carries every field the `flash_attn_utils` helpers read (by duck typing),
    including `HEAD_DIM_V`, plus the fields the parity kernels add. A standalone
    dataclass (not a subclass of the helpers' own) so that this module needs no
    flydsl.

    Roles in the dK/dV kernel are swapped: see `Gfx950DkdvTraits`.
    """

    # tile shape and waves
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_N_OUT: int
    K_SUB_N: int
    WARP_SIZE: int
    NUM_WAVES: int
    BLOCK_SIZE: int
    ROWS_PER_WAVE: int
    HEAD_DIM: int
    HEAD_DIM_V: int
    K_STEP_QK: int
    K_STEPS_QK: int
    D_CHUNK: int
    D_CHUNKS: int
    PV_K_STEP: int
    PV_K_STEPS: int
    MFMA_LANE_K: int
    # head counts: legacy fields of the shared helpers. The kernels take runtime head counts, so they
    # are fixed dummies here and no trait the kernel body reads is a head count (CFG-12).
    NUM_HEADS_Q: int
    NUM_HEADS_KV: int
    GQA_GROUP_SIZE: int
    # modes
    CAUSAL: bool
    DTYPE_STR: str
    WAVES_PER_EU: int
    DAZ: bool
    DUALWAVE_SWP_LAZY_RESCALE: bool
    DUALWAVE_SWP_SETPRIO: bool
    DUALWAVE_SWP_DEBUG_LAZY_COUNTS: bool
    DUALWAVE_SWP_ENABLE_STAGGER: bool
    NUM_KV_SPLITS: int
    SPLITK: bool
    PAGED: bool
    VARLEN: bool
    CROSS_SEQLEN: bool
    KV_CACHE_LAYOUT: str
    KV_VECTORIZED: bool
    DEFAULT_STRIDE_Q_N: int
    DEFAULT_STRIDE_KV_N: int
    # staging layout
    DMA_BYTES: int
    BF16_BYTES: int
    D_128B_SIZE: int
    VEC_KV: int
    SMEM_LINEAR_WAVE: int
    SMEM_N_PER_WAVE: int
    SMEM_N_RPT: int
    SMEM_D_RPT: int
    SMEM_K_PAD: int
    SMEM_V_PAD: int
    SMEM_K_LINE_STRIDE: int
    SMEM_V_LINE_STRIDE: int
    SMEM_K_TILE_ELEMS: int
    SMEM_V_TILE_ELEMS: int
    NUM_PREFETCH_K: int
    DUALWAVE_SWP_KV_PER_BUFFER: int
    LDS_KV_TOTAL_SIZE: int
    DUALWAVE_SWP_K_BUF_BASE: tuple[int, int]
    DUALWAVE_SWP_V_BUF_BASE: tuple[int, int]
    K_LDS_TO_REG_N_STRIP_STRIDE: int
    K_LDS_TO_REG_KSTEP_INNER_STRIDE: int
    K_LDS_TO_REG_KSTEP_OUTER_STRIDE: int
    V_LDS_TO_REG_HALF_WAVE_STRIDE: int
    V_LDS_TO_REG_LANE_QUAD_STRIDE: int
    V_LDS_TO_REG_N_GROUP_STRIDE: int
    V_LDS_TO_REG_LANE_IN_QUAD_STRIDE: int
    V_LDS_TO_REG_K_SUBSTEP_STRIDE: int
    V_LDS_TO_REG_DCHUNK_PAIR_STRIDE: int
    V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE: int
    V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE: int
    PAGED_BT_LDS_SIZE: int
    DUALWAVE_SWP_RESCALE_THRESHOLD: float
    KV_VEC_SIZE: int
    VEC_V_ROW_STRIDE: int
    SCHED_MFMA_MASK: int
    SCHED_VALU_MASK: int
    SCHED_EXP_MASK: int
    LDS_SCOPE_NAMES: tuple[str, str, str, str]
    NEG_INF_F32_BITS: int
    LGKMCNT_0_ONLY: int
    RETURN_LSE: bool  # the LSE is stored (always or when non-null); False only for RETURN_LSE="never"
    XCD_SWIZZLE: bool
    # --- parity fields ---
    # How many passes the KV tile's D axis is staged through LDS in. LDS is
    # `BLOCK_N * head_dim * ~8.3 B` against a 163840 B cap, so 384 and 512 do not fit in one pass.
    D_STAGES: int
    # QK_SHARDS is not implemented: fixed at 1.
    QK_SHARDS: int
    # How many waves split the *output* D axis of one Q tile. Shards write disjoint columns and never
    # have to agree on anything; the price is that QK is recomputed per shard.
    VO_SHARDS: int
    STAGE_DIM: int  # head_dim // D_STAGES
    K_STEPS_PER_STAGE: int
    D_CHUNKS_PER_STAGE: int
    D_CHUNKS_PER_STAGE_SHARD: int
    Q_TILES: int  # NUM_WAVES // VO_SHARDS
    # How a granule subdivides, for the sites that hardcoded granule 64.
    SMEM_D_BUCKETS: int
    K_STEPS_PER_BAND: int
    D_CHUNKS_PER_BAND: int
    # A window is causal plus a left bound. The bounds are runtime i32 (or baked, see STATIC_WINDOW).
    WINDOW: bool
    STATIC_WINDOW: bool
    # The sequence lengths are baked (dense, JIT-only): `CROSS_SEQLEN` and the KV-tail mask resolve at trace time.
    STATIC_SEQLEN: bool
    BIAS_TYPE: int
    ENABLE_DROPOUT: bool
    ALIBI: bool
    SINK: bool
    # Dispatch causal Q blocks longest-first: a bijection, so bit-identical output.
    LPT_TILE_ORDER: bool
    # A compile-time lower bound on the runtime hdim_qk, exclusive: columns at or below it are real
    # data and are not masked. The host rejects `hdim <= floor`. A correctness contract.
    HDIM_QK_FLOOR: int
    # The null-LSE-pointer branch is emitted (RETURN_LSE="runtime"); "always" drops it.
    LSE_NULL_GUARD: bool = True

    @property
    def cache_tag(self):
        """Every field, named. The JIT cache must distinguish every trait (a bias build once got the
        bias-free binary because the old tag omitted 46 of the fields)."""
        return traits_cache_key(self)


@dataclass(frozen=True)
class Gfx950DqTraits(Gfx950Traits):
    """The forward's traits plus what the dQ kernel alone needs."""

    # Write `dB = dS`. The store is per element, so a build that does not want it must not pay for it.
    # Whether it *runs* is a runtime question: all-zero dB strides mean no store.
    STORE_DB: bool = False
    # The `hdim_vo` counterpart of HDIM_QK_FLOOR, for the V tile's own padded-head mask.
    HDIM_VO_FLOOR: int = 0
    # (M, N, K) of the one MFMA this body issues; the three coincidences that hold at 32x32x16 and
    # none of which survives 16 rows are named separately.
    MFMA_M: int = 32
    MFMA_N: int = 32
    MFMA_K: int = 16

    @property
    def ACC_ELEMS(self):
        """f32 accumulator elements one lane holds for one MFMA (16 at 32x32x16, 4 at 16-row)."""
        return self.MFMA_M * self.MFMA_N // self.WARP_SIZE

    @property
    def SCORE_MSTEPS(self):
        """MFMA steps along the KV-token axis of one score tile."""
        return self.BLOCK_N // self.MFMA_M

    @property
    def OPERAND_LANE_ELEMS(self):
        """bf16 values one lane holds of an A or B operand, for one MFMA."""
        return self.MFMA_M * self.MFMA_K // self.WARP_SIZE


@dataclass(frozen=True)
class Gfx950DkdvTraits(Gfx950Traits):
    """The forward's traits, read with the dK/dV kernel's role names.

    dK/dV is the forward loop transposed: K and V stay resident in registers and Q
    and dO stream through LDS. Three trait names mean the other thing here:
    `BLOCK_M` is KV rows per workgroup, `BLOCK_N` is Q rows per streamed tile and
    `VO_SHARDS` is waves splitting the dK/dV D axis. The properties below are the
    kernel body's names for them; they add no new derived state.
    """

    # How many stream buffers the body cycles. 2 prefetches one tile ahead; 1 is the same body with
    # the prefetch distance collapsed to zero, which is what 384 and 512 buy their LDS with.
    NUM_STREAM_BUFFERS: int = 2
    # KV rows one wave owns, and so which MFMA family the body is: 32 or 16.
    MFMA_ROWS: int = 32
    # Trade instruction-level parallelism for live registers (32-row body only).
    TIGHT_REGISTERS: bool = False

    @property
    def BLOCK_KV(self):
        """KV rows one workgroup owns, resident in registers across the Q loop."""
        return self.BLOCK_M

    @property
    def BLOCK_Q(self):
        """Q rows one streamed tile carries through LDS."""
        return self.BLOCK_N

    @property
    def DKV_SHARDS(self):
        """Waves splitting the D axis of the dK and dV accumulators (`VO_SHARDS`)."""
        return self.VO_SHARDS

    @property
    def KV_BLOCKS_PER_WG(self):
        """Distinct 32-row KV blocks in one workgroup: `num_waves / shards`."""
        return self.Q_TILES

    @property
    def D_CHUNKS_PER_SHARD(self):
        """Output chunks one shard owns. Even, which the LDS offset needs."""
        return self.D_CHUNKS_PER_STAGE_SHARD

    @property
    def STREAM_LINE_STRIDE(self):
        """LDS elements per staged line. **Both streamed tiles use the V line stride**: the transpose
        read path is only validated against it, and the row-major read does not care."""
        return self.SMEM_V_LINE_STRIDE

    @property
    def STREAM_TILE_ELEMS(self):
        """LDS elements one staged tile occupies."""
        return self.SMEM_V_TILE_ELEMS

    @property
    def LDS_STREAM_TOTAL_SIZE(self):
        """LDS elements for both tensors across all buffers."""
        return 2 * self.NUM_STREAM_BUFFERS * self.STREAM_TILE_ELEMS


def traits_cache_key(traits):
    """`((name, value), ...)` over every field of `traits`, in declaration order."""
    return tuple((f.name, getattr(traits, f.name)) for f in fields(traits))


def _make_traits(
    *,
    head_dim,
    head_dim_v,
    num_waves,
    block_m,
    block_n,
    granule,
    d_stages=1,
    vo_shards=1,
    v_half_wave=None,
    v_n_group=None,
    v_k_substep=None,
    v_dc_in_pair=None,
    window=False,
    static_window=False,
    static_seqlen=False,
    bias=False,
    dropout=False,
    alibi=False,
    sink=False,
    lpt_tile_order=False,
    dtype_str="bf16",
    waves_per_eu=2,
    daz=True,
    setprio=True,
    stagger=True,
    num_kv_splits=1,
    paged=False,
    kv_cache_layout="linear",
    return_lse="runtime",
    xcd_swizzle=False,
    hdim_qk_floor=0,
):
    """The traits for an arbitrary (waves, BLOCK_M, BLOCK_N, granule); the one place geometry is checked.

    `head_dim` and `head_dim_v` are the compiled *tile* widths. Every derivation is a transcription of
    the original dual-wave constructor at family A's numbers, with the tile geometry opened up.
    """
    if num_waves % vo_shards:
        raise ValueError(f"VO_SHARDS {vo_shards} must divide num_warps {num_waves}")
    # With `vo_shards` waves sharing one Q tile, the workgroup covers `num_waves // vo_shards` tiles.
    # Rows per wave is *not* what shrinks (it is pinned at 32 by the MFMA's M extent): BLOCK_M falls
    # instead, and what each wave saves is D columns of O.
    q_tiles = num_waves // vo_shards
    if block_m % q_tiles:
        raise ValueError(f"BLOCK_M {block_m} does not divide across {q_tiles} Q tiles")
    if head_dim % granule:
        raise ValueError(f"head_dim tile {head_dim} is not a multiple of the granule {granule}")
    if head_dim % D_CHUNK:
        raise ValueError(f"head_dim tile {head_dim} is not a multiple of the PV MFMA width {D_CHUNK}")

    block_size = num_waves * WARP_SIZE
    rows_per_wave = block_m // q_tiles
    if rows_per_wave > MFMA_M:
        # **An invariant, enforced.** A wave holds at most the MFMA's M extent in rows, so BLOCK_M is
        # really `q_tiles * MFMA_M`; a larger one builds a kernel whose helpers address rows its
        # accumulator does not have. It does not fail: the sweep that found this had twelve such points,
        # each returning finite garbage at 0.15 to 0.28 relative error. Fewer rows per wave is legal and
        # merely proportionally slower.
        raise ValueError(
            f"BLOCK_M {block_m} over {q_tiles} Q tiles gives {rows_per_wave} rows per wave, but the MFMA's "
            f"M extent caps it at {MFMA_M}. BLOCK_M is derived: pass {q_tiles * MFMA_M} for "
            f"num_warps={num_waves}, VO_SHARDS={vo_shards}"
        )

    k_steps_qk = head_dim // K_STEP_QK
    d_chunks = head_dim // D_CHUNK
    pv_k_steps = K_SUB_N // PV_K_STEP

    # The D axis is cut two independent ways: `d_stages` splits it in time (one LDS residency per pass)
    # and `vo_shards` across waves. Validated together so an illegal pair fails at the decision.
    if d_stages < 1 or head_dim % d_stages:
        raise ValueError(f"head_dim tile {head_dim} is not a multiple of D_STAGES {d_stages}")
    stage_dim = head_dim // d_stages
    if stage_dim % granule:
        raise ValueError(f"stage extent {stage_dim} (tile/{d_stages}) is not a multiple of granule {granule}")
    if k_steps_qk % d_stages or d_chunks % d_stages:
        raise ValueError(
            f"D_STAGES {d_stages} must divide both K_STEPS_QK {k_steps_qk} and D_CHUNKS {d_chunks}; "
            "a stage that splits an MFMA step has no meaning"
        )
    if vo_shards < 1 or d_chunks % (d_stages * vo_shards):
        raise ValueError(
            f"VO_SHARDS {vo_shards} x D_STAGES {d_stages} must divide D_CHUNKS {d_chunks}; "
            "each (stage, shard) owns a whole number of 32-column PV output chunks"
        )
    d_chunks_per_stage_shard = d_chunks // (d_stages * vo_shards)
    if vo_shards > 1 and d_chunks_per_stage_shard % 2:
        raise ValueError(
            f"D_CHUNKS per (stage, shard) is {d_chunks_per_stage_shard}, which must be even once sharded: the "
            "LDS offset is folded into `urv_base`, and `_swizzled_v_dc_off` only decomposes that way "
            "when the shard starts on an even chunk"
        )
    k_steps_per_stage = k_steps_qk // d_stages
    d_chunks_per_stage = d_chunks // d_stages

    # One DMA issue per wave moves `smem_linear_wave` elements; the granule decides how that splits
    # into (tokens, D).
    smem_linear_wave = WARP_SIZE * DMA_BYTES // BF16_BYTES
    smem_n_per_wave = smem_linear_wave // granule
    if block_n % smem_n_per_wave:
        raise ValueError(f"BLOCK_N {block_n} is not a multiple of {smem_n_per_wave} tokens per DMA issue")
    smem_n_rpt = block_n // smem_n_per_wave
    # `stage_dim`, not `head_dim`: the sole term through which D_STAGES reaches LDS. At D_STAGES == 1
    # it is the tile width and every derived number is unchanged.
    smem_d_rpt = stage_dim // granule
    if smem_n_rpt == 0:
        raise ValueError(f"BLOCK_N {block_n} is not a multiple of {smem_n_per_wave} tokens per DMA issue")
    if smem_n_rpt % num_waves:
        raise ValueError(f"{smem_n_rpt} KV tile lines do not divide across {num_waves} waves")

    smem_k_pad = DMA_BYTES // BF16_BYTES
    smem_v_pad = 64 // BF16_BYTES
    smem_k_line_stride = smem_linear_wave + smem_k_pad
    smem_v_line_stride = smem_linear_wave + smem_v_pad
    smem_k_tile_elems = smem_n_rpt * smem_d_rpt * smem_k_line_stride
    smem_v_tile_elems = smem_n_rpt * smem_d_rpt * smem_v_line_stride

    num_prefetch_k = 2
    kv_per_buffer = smem_k_tile_elems + smem_v_tile_elems
    lds_kv_total_size = num_prefetch_k * kv_per_buffer

    # K LDS->VGPR. `n_strip_stride` is the offset from a lane's lo pack to its hi pack: half a wave of
    # lanes further on, each holding VEC_KV elements.
    k_lds_to_reg_n_strip_stride = (WARP_SIZE // 2) * VEC_KV
    k_lds_to_reg_kstep_inner_stride = K_STEP_QK
    k_lds_to_reg_kstep_outer_stride = smem_n_rpt * smem_k_line_stride

    # V LDS->VGPR. Three of these are "advance the KV token by t". A line holds `512 // granule` token
    # slots; slot s, line n is token `s * SMEM_N_RPT + n`, so advancing t tokens moves `t // N_RPT`
    # slots and `t % N_RPT` lines:
    #
    #     tok_off(t) = (t // N_RPT) * granule + (t % N_RPT) * line
    #
    #                    t   granule 64 (n_rpt 8)   granule 32 (n_rpt 4)
    #   half_wave        4   0*64 + 4*line = 2176   1*32 + 0      =   32
    #   transpose_pair   8   1*64 + 0      =   64   2*32 + 0      =   64
    #   k_substep       16   2*64 + 0      =  128   4*32 + 0      =  128
    #
    # At granule 64 that reproduces `4 * line`, `granule` and `2 * granule`, which is why those literals
    # survived and why each was wrong in a different way at granule 32. `n_group` is a D offset, not a
    # token one: 16 is the one the MFMA wants (lane 16 must receive D 16, not 8).
    def tok_off(t):
        return (t // smem_n_rpt) * granule + (t % smem_n_rpt) * smem_v_line_stride

    v_half_wave_stride = tok_off(4) if v_half_wave is None else v_half_wave
    v_n_group_stride = (2 * VEC_KV) if v_n_group is None else v_n_group
    v_k_substep_stride = tok_off(16) if v_k_substep is None else v_k_substep
    v_dc_in_pair_stride = D_CHUNK if v_dc_in_pair is None else v_dc_in_pair

    # No head counts reach the kernel as traits: they are runtime kernargs. These legacy fields exist
    # for the shared helpers, fixed at MHA with one head.
    num_heads = num_kv_heads = 1

    return Gfx950Traits(
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_N_OUT=block_n,
        K_SUB_N=K_SUB_N,
        WARP_SIZE=WARP_SIZE,
        NUM_WAVES=num_waves,
        BLOCK_SIZE=block_size,
        ROWS_PER_WAVE=rows_per_wave,
        HEAD_DIM=head_dim,
        HEAD_DIM_V=head_dim_v,
        K_STEP_QK=K_STEP_QK,
        K_STEPS_QK=k_steps_qk,
        D_CHUNK=D_CHUNK,
        D_CHUNKS=d_chunks,
        PV_K_STEP=PV_K_STEP,
        PV_K_STEPS=pv_k_steps,
        MFMA_LANE_K=MFMA_LANE_K,
        NUM_HEADS_Q=num_heads,
        NUM_HEADS_KV=num_kv_heads,
        GQA_GROUP_SIZE=num_heads // num_kv_heads,
        # A window build is the causal path with a left bound; a plain-causal build no longer exists.
        CAUSAL=bool(window),
        DTYPE_STR=dtype_str,
        WAVES_PER_EU=waves_per_eu,
        DAZ=bool(daz),
        DUALWAVE_SWP_LAZY_RESCALE=_LAZY_RESCALE,
        DUALWAVE_SWP_SETPRIO=bool(setprio),
        DUALWAVE_SWP_DEBUG_LAZY_COUNTS=False,
        DUALWAVE_SWP_ENABLE_STAGGER=bool(stagger),
        NUM_KV_SPLITS=num_kv_splits,
        SPLITK=num_kv_splits > 1,
        PAGED=bool(paged),
        # **Constants, not choices.** This arch decodes `varlen_bits` at runtime, so there is no dense
        # build to distinguish; the two fields survive only because the shared helpers read them.
        # `VARLEN=True` makes `compute_active_guard` return the unconditional `q_start < seqlen_q_v`.
        # `CROSS_SEQLEN` follows `CAUSAL`: Q and K lengths arrive at runtime from independent arrays, so
        # no build knows whether they match, and where `seqlen_k < seqlen_q` bottom-right causal leaves
        # leading Q blocks with no live key, which must be *written* as zero, not skipped.
        VARLEN=True,
        CROSS_SEQLEN=bool(window),
        KV_CACHE_LAYOUT=kv_cache_layout,
        KV_VECTORIZED=bool(paged and kv_cache_layout == "vectorized"),
        DEFAULT_STRIDE_Q_N=num_heads * head_dim,
        DEFAULT_STRIDE_KV_N=num_kv_heads * head_dim,
        DMA_BYTES=DMA_BYTES,
        BF16_BYTES=BF16_BYTES,
        D_128B_SIZE=granule,
        VEC_KV=VEC_KV,
        SMEM_LINEAR_WAVE=smem_linear_wave,
        SMEM_N_PER_WAVE=smem_n_per_wave,
        SMEM_N_RPT=smem_n_rpt,
        SMEM_D_RPT=smem_d_rpt,
        SMEM_K_PAD=smem_k_pad,
        SMEM_V_PAD=smem_v_pad,
        SMEM_K_LINE_STRIDE=smem_k_line_stride,
        SMEM_V_LINE_STRIDE=smem_v_line_stride,
        SMEM_K_TILE_ELEMS=smem_k_tile_elems,
        SMEM_V_TILE_ELEMS=smem_v_tile_elems,
        NUM_PREFETCH_K=num_prefetch_k,
        DUALWAVE_SWP_KV_PER_BUFFER=kv_per_buffer,
        LDS_KV_TOTAL_SIZE=lds_kv_total_size,
        DUALWAVE_SWP_K_BUF_BASE=(0, kv_per_buffer),
        DUALWAVE_SWP_V_BUF_BASE=(smem_k_tile_elems, smem_k_tile_elems + kv_per_buffer),
        K_LDS_TO_REG_N_STRIP_STRIDE=k_lds_to_reg_n_strip_stride,
        K_LDS_TO_REG_KSTEP_INNER_STRIDE=k_lds_to_reg_kstep_inner_stride,
        K_LDS_TO_REG_KSTEP_OUTER_STRIDE=k_lds_to_reg_kstep_outer_stride,
        V_LDS_TO_REG_HALF_WAVE_STRIDE=v_half_wave_stride,
        V_LDS_TO_REG_LANE_QUAD_STRIDE=smem_v_line_stride,
        V_LDS_TO_REG_N_GROUP_STRIDE=v_n_group_stride,
        V_LDS_TO_REG_LANE_IN_QUAD_STRIDE=4,
        V_LDS_TO_REG_K_SUBSTEP_STRIDE=v_k_substep_stride,
        V_LDS_TO_REG_DCHUNK_PAIR_STRIDE=smem_n_rpt * smem_v_line_stride,
        V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE=v_dc_in_pair_stride,
        V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE=tok_off(8),
        PAGED_BT_LDS_SIZE=PAGED_BT_LDS_SIZE,
        DUALWAVE_SWP_RESCALE_THRESHOLD=_RESCALE_THRESHOLD,
        KV_VEC_SIZE=DMA_BYTES // BF16_BYTES,
        VEC_V_ROW_STRIDE=smem_v_line_stride,
        SCHED_MFMA_MASK=SCHED_MFMA_MASK,
        SCHED_VALU_MASK=SCHED_VALU_MASK,
        SCHED_EXP_MASK=SCHED_EXP_MASK,
        LDS_SCOPE_NAMES=("lds_k0", "lds_k1", "lds_v0", "lds_v1"),
        NEG_INF_F32_BITS=NEG_INF_F32_BITS,
        LGKMCNT_0_ONLY=LGKMCNT_0_ONLY,
        RETURN_LSE=return_lse != "never",
        XCD_SWIZZLE=bool(xcd_swizzle),
        D_STAGES=d_stages,
        QK_SHARDS=1,
        VO_SHARDS=vo_shards,
        STAGE_DIM=stage_dim,
        K_STEPS_PER_STAGE=k_steps_per_stage,
        D_CHUNKS_PER_STAGE=d_chunks_per_stage,
        D_CHUNKS_PER_STAGE_SHARD=d_chunks_per_stage_shard,
        Q_TILES=q_tiles,
        SMEM_D_BUCKETS=granule // VEC_KV,
        K_STEPS_PER_BAND=granule // K_STEP_QK,
        D_CHUNKS_PER_BAND=granule // D_CHUNK,
        WINDOW=bool(window),
        STATIC_WINDOW=bool(static_window),
        STATIC_SEQLEN=bool(static_seqlen),
        BIAS_TYPE=1 if bias else 0,
        ENABLE_DROPOUT=bool(dropout),
        ALIBI=bool(alibi),
        SINK=bool(sink),
        LPT_TILE_ORDER=bool(lpt_tile_order),
        HDIM_QK_FLOOR=int(hdim_qk_floor),
        LSE_NULL_GUARD=return_lse == "runtime",
    )


# ---------------------------------------------------------------------------
# Knobs: the shared pipeline
# ---------------------------------------------------------------------------

# Geometries whose *address helpers* are known correct, which is a stricter set than the ones
# `_make_traits` can describe. BLOCK_M is in the tuple but does not affect KV addressing at all
# (`SMEM_N_RPT` follows BLOCK_N and the granule). (8, 128, 64, 64) is family W at 2 shards and reuses
# family A's staging exactly.
_FWD_SUPPORTED_GEOMETRIES = (
    (8, 256, 64, 64),
    (4, 128, 64, 64),
    (8, 128, 64, 64),
    (4, 64, 64, 64),
    (8, 64, 64, 64),
    (4, 128, 64, 32),  # family S: granule 32, for widths off the 64 grid
)

# dQ adds the two-wave points and the 16-row family. Not a relaxation of the check: it exists because
# `_k_dma_m0_base` assumes one DMA issue per wave, and the staging generalisation lifts it; every entry
# is *run* by the geometry-agreement test.
_DQ_SUPPORTED_GEOMETRIES = _FWD_SUPPORTED_GEOMETRIES + (
    (2, 64, 64, 64),
    (2, 64, 64, 32),
    # 16-row family: BLOCK_M is `4 waves * 16 rows`, BLOCK_N is 32, so a KV tile is four lines.
    (4, 64, 32, 64),
)


class _Knobs:
    """Behaviour shared by every knob class: pin merging, serialisation and the width step."""

    def merge(self, other):
        """`other`'s set fields win; its `None`s leave this one's alone."""
        if other is None:
            return self
        set_fields = {f.name: getattr(other, f.name) for f in fields(other) if getattr(other, f.name) is not None}
        return replace(self, **set_fields)

    def as_psels(self) -> dict[str, Any]:
        """A flat dict of str/int/bool scalars, one per knob. Raises if `resolve` has not run (a `None`)."""
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                raise ValueError(f"knob {f.name} is unresolved; call .resolve(meta) before .as_psels()")
            out[f.name] = value
        return out

    def _with_widths(self, meta):
        """Decide BLOCK_DMODEL, BLOCK_DMODEL_V and PADDED_HEAD, validating every pin.

        The HDIM_QK_FLOOR trait follows from the rung (`_floor_for`): only a *derived* tile carries the
        ladder's guarantee, so a caller that pins BLOCK_DMODEL wider than the ladder's own rung for
        `head_dim` claims no floor and the kernel masks everything.
        """
        head_dim_v = meta.head_dim_v_real
        own = tile_width_for(meta.head_dim)
        block = self.BLOCK_DMODEL
        if block is None:
            block = own
        elif block not in LADDER:
            raise ValueError(f"BLOCK_DMODEL must be one of the built rungs {LADDER}, got {block}")
        if meta.head_dim > block:
            raise ValueError(f"head_dim {meta.head_dim} does not fit the pinned BLOCK_DMODEL {block}")
        if head_dim_v > block:
            raise ValueError(f"head_dim_v {head_dim_v} does not fit BLOCK_DMODEL {block}")
        block_v = block if self.BLOCK_DMODEL_V is None else self.BLOCK_DMODEL_V
        if block_v != block:
            # Both bodies size the tile, the staging and the O accumulator from one width. A second tile
            # width would give the V reads and the output store a different D_CHUNKS from QK.
            raise NotImplementedError(
                f"BLOCK_DMODEL_V {block_v} != BLOCK_DMODEL {block}: the kernels stage one tile width for Q, K, V "
                "and O, so a separate V rung has nothing to describe"
            )
        needs_pad = (meta.head_dim != block) or (head_dim_v != block_v)
        padded = needs_pad if self.PADDED_HEAD is None else bool(self.PADDED_HEAD)
        if needs_pad and not padded:
            raise ValueError(
                f"PADDED_HEAD=False requires head_dim == BLOCK_DMODEL and head_dim_v == BLOCK_DMODEL_V; got "
                f"head_dim {meta.head_dim} / head_dim_v {head_dim_v} for tile {block}. An unmasked wider tile "
                "reduces over the caller's padding and returns a finite wrong answer"
            )
        return replace(self, BLOCK_DMODEL=block, BLOCK_DMODEL_V=block_v, PADDED_HEAD=padded)

    def _check_grid_axis_order(self, derived):
        """GRID_AXIS_ORDER is an output. Pinning the value `resolve` would pick is a no-op (so `resolve`
        is idempotent); pinning any other is an error."""
        if self.GRID_AXIS_ORDER is not None and self.GRID_AXIS_ORDER != derived:
            raise ValueError(
                f"GRID_AXIS_ORDER is an output of resolve (it is the launcher's contract with the C++ grid "
                f"calculator), not an input: got {self.GRID_AXIS_ORDER}, this kernel's is {derived}"
            )
        return replace(self, GRID_AXIS_ORDER=derived)

    def _check_geometry_pin(self, names):
        """The geometry tuple pins together or not at all, because only some tuples have verified helpers."""
        pinned = tuple(getattr(self, n) for n in names)
        if any(x is not None for x in pinned) and not all(x is not None for x in pinned):
            raise ValueError(f"pin {', '.join(names)} together or not at all, got {dict(zip(names, pinned))}")
        return all(x is not None for x in pinned)


# ---------------------------------------------------------------------------
# Forward knobs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gfx950FwdKnobs(_Knobs):
    """The forward's constexpr build parameters. `None` means "`resolve` decides"."""

    # widths: the rung head_dim is padded to, and whether the runtime head dim may be narrower
    BLOCK_DMODEL: int | None = None
    BLOCK_DMODEL_V: int | None = None
    PADDED_HEAD: bool | None = None
    # wave geometry: pinned as a set of four
    BLOCK_M: int | None = None
    BLOCK_N: int | None = None
    num_warps: int | None = None
    HEAD_DIM_GRANULE: int | None = None
    # the two D-axis splits: per-width policy a sweep wants to vary on its own
    D_STAGES: int | None = None
    VO_SHARDS: int | None = None
    # compiler / launch attributes
    waves_per_eu: int | None = None
    daz: bool | None = None
    # schedule
    SETPRIO: bool | None = None
    STAGGER: bool | None = None
    LPT_TILE_ORDER: bool | None = None
    # outputs and window baking
    RETURN_LSE: str | None = None  # "runtime" (store unless the LSE pointer is null) | "always" | "never"
    STATIC_WINDOW: bool | None = None
    # JIT-only, dense-only, opt-in: bakes `Max_seqlen_q`/`Max_seqlen_k` (call data, a compile per length pair). Never for AOT.
    STATIC_SEQLEN: bool | None = None
    # feature knobs, off by default; AOTriton never sets them
    XCD_SWIZZLE: bool | None = None
    NUM_KV_SPLITS: int | None = None
    # output only: which quantity lands on which grid axis
    GRID_AXIS_ORDER: int | None = None

    def resolve(self, meta: FmhaInputMetadata, hints: FmhaHints = FmhaHints()) -> "Gfx950FwdKnobs":
        """The complete build configuration for `meta`. Idempotent: every derived field is recomputed
        from `meta` and the pinned fields rather than read back."""
        _check_meta(meta)
        return (
            _FWD_FALLBACK.merge(self)
            ._checked_modes(meta)
            ._with_policy_defaults(meta, hints)
            ._with_widths(meta)
            ._with_wave_geometry()
            ._with_occupancy_target(meta)
            ._checked_against_traits(meta)
        )

    # -- steps ------------------------------------------------------------

    def _checked_modes(self, meta):
        if self.RETURN_LSE not in _RETURN_LSE_MODES:
            raise ValueError(f"RETURN_LSE must be one of {_RETURN_LSE_MODES}, got {self.RETURN_LSE!r}")
        if self.NUM_KV_SPLITS < 1:
            raise ValueError(f"NUM_KV_SPLITS must be >= 1, got {self.NUM_KV_SPLITS}")
        if self.STATIC_WINDOW and not meta.window:
            raise ValueError("STATIC_WINDOW bakes the window bounds, so it requires meta.window")
        return self

    def _with_policy_defaults(self, meta, hints):
        # Longest-first dispatch is a bijection that pays for every windowed build (measured +1.4% to
        # +40%, see the A/B record), so it is on exactly there.
        lpt = meta.window if self.LPT_TILE_ORDER is None else self.LPT_TILE_ORDER
        return replace(self, LPT_TILE_ORDER=bool(lpt))

    def _with_d_axis_splits(self):
        """D_STAGES and VO_SHARDS from the tile width; both stay 1 through 256.

        - D_STAGES answers **LDS**: one pass of 384 needs 199.5 KB against a 160 KB cap.
        - VO_SHARDS answers **registers**: at 512 a wave's O accumulator is the whole AGPR file.

        2 stages is the least that fits LDS, and measurement says least is also best: past 2 the
        allocator stops using AGPRs altogether and spills to scratch, and at 512 the register allocator
        itself runs away (more than 480 s to build). 384's O is 192 VGPRs and fits unsharded.
        """
        d_stages, vo_shards = self.D_STAGES, self.VO_SHARDS
        if d_stages is None:
            d_stages = 2 if self.BLOCK_DMODEL > 256 else 1
        if vo_shards is None:
            vo_shards = 2 if self.BLOCK_DMODEL > 384 else 1
        return replace(self, D_STAGES=d_stages, VO_SHARDS=vo_shards)

    def _with_wave_geometry(self):
        """Wave geometry and staging granule from the tile width.

        | family | tile width | waves | BLOCK_M | BLOCK_N | granule |
        |---|---|---|---|---|---|
        | S | off the 64 grid | 4 | 128 | 64 | 32 |
        | A | <= 128 | 8 | 256 | 64 | 64 |
        | B | 129..256 | 4 | 128 | 64 | 64 |
        | W | > 256 | 4 | (4 / VO_SHARDS) * 32 | 64 | 64 |

        **A** is measured saturated at 128 (248 of 256 VGPRs, zero spills), so **B** halves the wave
        count to double the per-lane register file. At 8 waves the allocator stops using AGPRs and spills
        to scratch; at 4 it puts the O accumulator where it belongs. For **W** the effect is the single
        biggest lever measured: 384 at 4 waves is 579 TFLOP/s against 372 at 8, and no amount of extra
        sharding recovers it. `ROWS_PER_WAVE` stays 32 (the MFMA's M extent), so BLOCK_M is
        `Q_TILES * 32` and the shards eat the waves rather than the rows.
        """
        me = self._with_d_axis_splits()
        if me._check_geometry_pin(("num_warps", "BLOCK_M", "BLOCK_N", "HEAD_DIM_GRANULE")):
            return me
        if me.BLOCK_DMODEL % 64:
            return replace(me, num_warps=4, BLOCK_M=128, BLOCK_N=64, HEAD_DIM_GRANULE=32)  # family S
        if me.BLOCK_DMODEL <= 128:
            return replace(me, num_warps=8, BLOCK_M=256, BLOCK_N=64, HEAD_DIM_GRANULE=64)  # family A
        if me.BLOCK_DMODEL <= 256:
            return replace(me, num_warps=4, BLOCK_M=128, BLOCK_N=64, HEAD_DIM_GRANULE=64)  # family B
        return replace(me, num_warps=4, BLOCK_M=(4 // me.VO_SHARDS) * 32, BLOCK_N=64, HEAD_DIM_GRANULE=64)  # W

    def _with_occupancy_target(self, meta):
        """Two waves per EU only where two workgroups can exist.

        A workgroup of 4 waves is one wave on each of a CU's 4 SIMDs, so two waves per EU means two
        *co-resident workgroups*, and LDS decides whether that is possible before the register allocator
        is consulted. Past 128 it is not close (160 stages 85 KB, so the pair wants 170 KB against the 160 KB
        cap), and the backend says so once per build ("desired occupancy was 2, final occupancy is 1", on 120
        of the 216 shipped builds).

        **Dropping a refused request costs nothing, which was checked rather than assumed**: the 120
        refused builds were built both ways and 118 were byte-identical (the 2 that differ are noise:
        this backend is not bit-reproducible, and rebuilding all 216 at unchanged settings also gives 2).
        **Where the hint is granted it is doing real work**: at 96, 17 of 24 builds change when it is
        dropped, 13 of them raising VGPRs. The rule is LDS rather than a head_dim list, because the
        granule, BLOCK_N and D_STAGES all move the footprint. A pinned `waves_per_eu` outranks this.
        """
        if self.waves_per_eu is not None:
            return self
        probe = replace(self, waves_per_eu=1)
        lds_bytes = fwd_traits(meta, probe).LDS_KV_TOTAL_SIZE * BF16_BYTES
        return replace(self, waves_per_eu=2 if 2 * lds_bytes <= LDS_CAP_BYTES else 1)

    def _checked_against_traits(self, meta):
        """`resolve`'s last step: prove the traits are buildable. The traits object is built and thrown
        away; only its verdict is kept, because every check names the knob to move."""
        fwd_traits(meta, self)
        return self._check_grid_axis_order(GRID_AXIS_HEAD_FASTEST)


# Defaults the policy has no shape-dependent opinion about. `waves_per_eu` is absent so that
# `_with_occupancy_target` can see that nobody pinned it.
#
# **`daz=True`** is flash-attention's usual setting, with no perf or accuracy change measured in the DAZ fix; it
# is a numerics change from AOTriton 0.14's effective IEEE denormals.
_FWD_FALLBACK = Gfx950FwdKnobs(
    daz=True,
    SETPRIO=True,
    STAGGER=True,
    RETURN_LSE="runtime",
    STATIC_WINDOW=False,
    STATIC_SEQLEN=False,
    XCD_SWIZZLE=False,
    NUM_KV_SPLITS=1,
)


def fwd_traits(meta: FmhaInputMetadata, knobs: Gfx950FwdKnobs) -> Gfx950Traits:
    """The traits `knobs` (resolved against `meta`) imply. The one knob->traits map; what the forward
    builder is made from. Needs the geometry and widths decided, which is why `resolve` calls it last."""
    _check_meta(meta)
    if knobs.num_warps is None or knobs.BLOCK_DMODEL is None:
        raise ValueError("knobs are not resolved: call `fwd_knobs(...).resolve(meta)` first")
    if knobs.VO_SHARDS is None or knobs.D_STAGES is None:
        raise ValueError("D_STAGES / VO_SHARDS are not resolved")
    _check_geometry_supported(knobs, ("num_warps", "BLOCK_M", "BLOCK_N", "HEAD_DIM_GRANULE"), _FWD_SUPPORTED_GEOMETRIES)
    if knobs.D_STAGES > 2:
        # The wide body's `s_waitcnt vmcnt` bookkeeping retires "the stage just issued, plus the next tile's first K
        # stage", which is right for two stages and, from the third stage on, lets the stage being read race its own DMA:
        # 384 with 6 stages and 512 with 4 return finite wrong values in the last D chunks (O error 0.17 and 0.46 against a
        # floor of 2e-3; 3 stages happens to survive the race). More stages were never better anyway: past two the
        # allocator stops using AGPRs and spills, and at 512 it runs away (more than 480 s to build).
        raise NotImplementedError(
            f"D_STAGES={knobs.D_STAGES}: the wide body's DMA wait counts are derived for at most 2 stages; "
            "beyond that the stage being read can race its own DMA and the answer is finite and wrong"
        )
    traits = _make_traits(
        head_dim=knobs.BLOCK_DMODEL,
        head_dim_v=knobs.BLOCK_DMODEL_V,
        num_waves=knobs.num_warps,
        block_m=knobs.BLOCK_M,
        block_n=knobs.BLOCK_N,
        granule=knobs.HEAD_DIM_GRANULE,
        d_stages=knobs.D_STAGES,
        vo_shards=knobs.VO_SHARDS,
        window=meta.window,
        static_window=bool(knobs.STATIC_WINDOW),
        static_seqlen=bool(knobs.STATIC_SEQLEN),
        bias=meta.bias,
        dropout=meta.dropout,
        alibi=meta.alibi,
        sink=meta.sink,
        lpt_tile_order=knobs.LPT_TILE_ORDER,
        dtype_str=meta.dtype_str,
        waves_per_eu=knobs.waves_per_eu,
        daz=knobs.daz,
        setprio=knobs.SETPRIO,
        stagger=knobs.STAGGER,
        num_kv_splits=knobs.NUM_KV_SPLITS,
        paged=meta.paged,
        kv_cache_layout=meta.kv_cache_layout,
        return_lse=knobs.RETURN_LSE,
        xcd_swizzle=knobs.XCD_SWIZZLE,
        hdim_qk_floor=_floor_for(meta, knobs),
    )
    _check_lds(traits.LDS_KV_TOTAL_SIZE, knobs, "KV staging")
    return traits


def _check_geometry_supported(knobs, names, supported):
    """Refuse a geometry the kernel's addressing cannot actually serve.

    `_make_traits` takes the geometry as parameters, so it will happily *describe* families the
    addressing has not caught up with. The gap is specific: `_k_dma_m0_base` places a tile line per
    wave per d-band (assuming `SMEM_N_RPT == NUM_WAVES` unless the staging generalisation applies),
    `init_dma_thread_offsets` splits a lane as `lane // VEC_KV` tokens by `lane % VEC_KV` D-buckets, and the
    K/V read bases fold constants that are `SMEM_N_RPT` and `granule // K_STEP_QK` at family A's
    numbers. Failing here keeps the diagnosis at the level of the decision: a geometry that builds and
    runs but addresses the wrong LDS produces plausible numbers.
    """
    geom = tuple(getattr(knobs, n) for n in names)
    if geom not in supported:
        raise NotImplementedError(
            f"geometry ({', '.join(names)}) = {geom} is describable but not yet addressable: the DMA and "
            f"LDS-read helpers assume particular staging shapes. Supported: {supported}"
        )
    if knobs.BLOCK_DMODEL < PV_MFMA_N:
        raise ValueError(f"BLOCK_DMODEL {knobs.BLOCK_DMODEL} is narrower than the PV MFMA's {PV_MFMA_N}-column output")


def _check_lds(elems, knobs, what):
    lds_bytes = elems * BF16_BYTES
    if lds_bytes > LDS_CAP_BYTES:
        raise ValueError(
            f"{what} needs {lds_bytes} B of LDS, over the {LDS_CAP_BYTES} B cap, for BLOCK_DMODEL "
            f"{knobs.BLOCK_DMODEL} at BLOCK_N {knobs.BLOCK_N}. Raise D_STAGES (LDS scales as 1/D_STAGES) or "
            "lower BLOCK_N. Left to the compiler this surfaces as 'local memory (N) exceeds limit' with no "
            "indication of which knob to move."
        )


def _floor_for(meta, knobs):
    """HDIM_QK_FLOOR for a resolved knob set (a pure function of `meta` and BLOCK_DMODEL)."""
    block = knobs.BLOCK_DMODEL
    return rung_below(block) if block == tile_width_for(meta.head_dim) else 0


# ---------------------------------------------------------------------------
# dQ knobs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gfx950DqKnobs(_Knobs):
    """The dQ (and dB) kernel's build parameters.

    Shares the forward's common knobs. `D_STAGES`, `VO_SHARDS` and `QK_SHARDS` are **traits fixed at 1**,
    not knobs: `D_STAGES=2` is a silent wrong answer in dQ (`qk` becomes a one-stage reduction and `pv`
    writes half the accumulator, finite and wrong). `STORE_DB` is derived from `meta.bias`, with the
    runtime zero-stride gate covering "bias without dB". The forward-only knobs are absent.
    """

    BLOCK_DMODEL: int | None = None
    BLOCK_DMODEL_V: int | None = None
    PADDED_HEAD: bool | None = None
    BLOCK_M: int | None = None
    BLOCK_N: int | None = None
    num_warps: int | None = None
    HEAD_DIM_GRANULE: int | None = None
    # JIT-only, opt-in: bake call data (`Window_left`/`Window_right`, `Max_seqlen_q`/`Max_seqlen_k`) as `Constexpr`s, one
    # compile per value. Off for every AOT build, where they are real `Int32` kernargs.
    STATIC_WINDOW: bool | None = None
    STATIC_SEQLEN: bool | None = None
    # 32 or 16: the MFMA's N extent, i.e. the query rows one wave owns. The MFMA shape and the KV tile
    # are derived together from it, because at 16 rows BLOCK_N is not free.
    MFMA_ROWS: int | None = None
    waves_per_eu: int | None = None
    daz: bool | None = None
    SETPRIO: bool | None = None
    LPT_TILE_ORDER: bool | None = None
    GRID_AXIS_ORDER: int | None = None

    def resolve(self, meta: FmhaInputMetadata, hints: FmhaHints = FmhaHints()) -> "Gfx950DqKnobs":
        _check_bwd_meta(meta, "dQ")
        return _DQ_FALLBACK.merge(self)._with_widths(meta)._with_wave_geometry()._checked_against_traits(meta)

    def _hdim_vo_floor(self, meta):
        """The V tile's counterpart of HDIM_QK_FLOOR, exclusive.

        The floor exists so the padded-head mask can skip the D columns the dispatcher guarantees are
        real (it is what keeps a padded build from paying the 27-54% the forward measured for masking
        every K-step). Both extents live in the same compiled tile, so the vo floor comes from the same
        rung. Dropped to 0 when `head_dim_v` is at or below it: an asymmetric call like (128, 40) promises
        nothing about `hdim_vo`, so the V mask must cover every step.
        """
        floor = _floor_for(meta, self)
        return floor if meta.head_dim_v_real > floor else 0

    def _with_wave_geometry(self):
        """dQ's own geometry table. Measured, not inherited.

        Measured at `B=2 H=8 S=2048` bf16 non-causal, TFLOP/s on `6*S^2*d`, over the whole
        `(num_warps, waves_per_eu)` grid:

            head_dim   w2/e1  w2/e2  w4/e1  w4/e2  w8/e1  w8/e2
                32       335    334    423    420      -      -
                64       501    500    582    580    436    435
                96       599    597    678    681      -      -
               128       715    622    685    702    496    505
               160       665    313    645    429      -      -
               192       672    260    657    350    311    316
               224       682    667    677    171      -      -
               256       712    694    675    169    162    162
               384       339    336    335    338    338    336
               512       115    113    113    117    115    114

        (`-` is `SMEM_N_RPT % NUM_WAVES != 0`: a granule-32 tile has four KV lines, so eight waves
        cannot each own one.)

        **`waves_per_eu = 1` unconditionally**: never worse than 2 and worth up to 4x (256 at four waves is
        675 against 169; the slow build reports 0 AGPRs and 191 spills against 460 VGPR + 204 AGPR with
        zero spills: it was forbidden from using half the registers). **Two waves from 128 to 256, four
        elsewhere.** 384 and 512 are flat across the grid: they are not occupancy-limited.
        """
        me = self
        if me.waves_per_eu is None:
            me = replace(me, waves_per_eu=1)
        if me._check_geometry_pin(("num_warps", "BLOCK_M", "BLOCK_N", "HEAD_DIM_GRANULE")):
            if me.MFMA_ROWS is None:
                raise ValueError("a pinned geometry must pin MFMA_ROWS too")
            return me
        # **256 and up take the 16-row family.** Measured at `B=2 H=8 S=4096`, 16 rows against 32: 512 is
        # 4.23x (546 spills -> 0), 384 is 1.69x, 256 is 1.04x, 192 level, 128 and 64 worse (0.94x, 0.86x).
        # 256 at 1.04x is inside the band a sweep cannot settle; correctness moved it: the 32-row build
        # at 256 with dropout lands on 507 VGPR + 251 AGPR, five registers short of the file, and the
        # allocator spills the 32 LDS-DMA address operands into AGPRs and rematerialises nine of them
        # through a single VGPR that is overwritten between consecutive `buffer_load ... offen lds`
        # issues. That build races: it poisons one wave's 32x32 accumulator tile in one or two
        # (batch, head, q-block) instances per launch, non-deterministically. The 16-row family at 256
        # is 232 VGPR, 0 AGPR, 0 spills and half the LDS (564 TF against 550 without dropout, 527 against 509 with).
        rows = me.MFMA_ROWS if me.MFMA_ROWS is not None else (16 if me.BLOCK_DMODEL >= 256 else 32)
        if rows == 16:
            # BLOCK_N 32 because the score tile must stay two MFMA steps, and four waves because the
            # wide rungs need one wave per SIMD to see all 512 registers.
            return replace(me, num_warps=4, BLOCK_M=4 * 16, BLOCK_N=32, HEAD_DIM_GRANULE=64, MFMA_ROWS=16)
        waves = 2 if 128 <= me.BLOCK_DMODEL <= 256 else 4
        return replace(
            me,
            MFMA_ROWS=32,
            num_warps=waves,
            # `ROWS_PER_WAVE` is pinned at the MFMA's M extent, so BLOCK_M is derived, not chosen.
            BLOCK_M=waves * 32,
            BLOCK_N=64,
            HEAD_DIM_GRANULE=32 if me.BLOCK_DMODEL % 64 else 64,
        )

    def _checked_against_traits(self, meta):
        if self.STATIC_WINDOW and not meta.window:
            raise ValueError("STATIC_WINDOW bakes the window bounds, so it requires meta.window")
        dq_traits(meta, self)
        return self._check_grid_axis_order(GRID_AXIS_HEAD_FASTEST)


# No measurement for the dQ tile walk order: LPT is off unless asked for.
_DQ_FALLBACK = Gfx950DqKnobs(daz=True, SETPRIO=True, LPT_TILE_ORDER=False, STATIC_WINDOW=False, STATIC_SEQLEN=False)

# The 16-row family's transpose read folds `tok_off(4 * group)` into `group * granule`, which holds only
# when `SMEM_N_RPT` divides 4: true at granule 64 and not at granule 32 where it is 2.


def _check_bwd_meta(meta, which):
    _check_meta(meta)
    for name in ("alibi", "sink", "paged"):
        if getattr(meta, name):
            raise NotImplementedError(
                f"{name}=True is not implemented by the backward {which} kernel: the backward serves the "
                "forward's training inputs (window, bias, dropout) only"
            )


def dq_traits(meta: FmhaInputMetadata, knobs: Gfx950DqKnobs) -> Gfx950DqTraits:
    """The dQ traits `knobs` (resolved against `meta`) imply."""
    _check_bwd_meta(meta, "dQ")
    floor = _floor_for(meta, knobs)
    if knobs.MFMA_ROWS is None or knobs.num_warps is None:
        raise ValueError("knobs are not resolved: call `dq_knobs(...).resolve(meta)` first")
    if knobs.MFMA_ROWS not in (16, 32):
        raise ValueError(f"MFMA_ROWS must be 16 or 32, got {knobs.MFMA_ROWS}")
    shape = (16, 16, 32) if knobs.MFMA_ROWS == 16 else (32, 32, 16)
    if knobs.MFMA_ROWS == 16 and knobs.BLOCK_DMODEL % 64:
        # Before the traits constructor, which would otherwise raise first with a message about the
        # granule and leave the caller to work out that the granule came from the family.
        raise NotImplementedError(
            f"the 16-row family is built for head_dim tiles that are multiples of 64, not {knobs.BLOCK_DMODEL}: "
            "its transpose read assumes the granule-64 staging shape, and the off-grid rungs are all served by "
            "the 32-row family anyway"
        )
    if knobs.BLOCK_DMODEL_V != knobs.BLOCK_DMODEL:
        raise NotImplementedError("dQ's output width is the *qk* extent; a second tile width has nothing to describe")
    _check_geometry_supported(knobs, ("num_warps", "BLOCK_M", "BLOCK_N", "HEAD_DIM_GRANULE"), _DQ_SUPPORTED_GEOMETRIES)
    if knobs.MFMA_ROWS == 16 and knobs.BLOCK_N != 2 * 16:
        raise ValueError(
            f"the 16-row family needs BLOCK_N {2 * 16}, got {knobs.BLOCK_N}: the score tile must stay two MFMA "
            "steps or the (s_lo, s_hi) pair in the shared helpers stops describing it"
        )
    base = _make_traits(
        head_dim=knobs.BLOCK_DMODEL,
        head_dim_v=knobs.BLOCK_DMODEL_V,
        num_waves=knobs.num_warps,
        block_m=knobs.BLOCK_M,
        block_n=knobs.BLOCK_N,
        granule=knobs.HEAD_DIM_GRANULE,
        # D_STAGES / VO_SHARDS / QK_SHARDS are fixed at 1: refused as knobs, not defaulted away.
        window=meta.window,
        static_window=bool(knobs.STATIC_WINDOW),
        static_seqlen=bool(knobs.STATIC_SEQLEN),
        bias=meta.bias,
        dropout=meta.dropout,
        lpt_tile_order=bool(knobs.LPT_TILE_ORDER),
        dtype_str=meta.dtype_str,
        waves_per_eu=knobs.waves_per_eu,
        daz=knobs.daz,
        setprio=knobs.SETPRIO,
        stagger=False,  # neither backward kernel uses stagger
        num_kv_splits=1,
        paged=False,
        kv_cache_layout=meta.kv_cache_layout,
        return_lse="never",
        hdim_qk_floor=floor,
    )
    m, n, k = shape
    if n != base.ROWS_PER_WAVE:
        raise ValueError(f"MFMA N extent {n} must equal ROWS_PER_WAVE {base.ROWS_PER_WAVE}: the query row is N")
    # **One K-pitch region plus one V-pitch region**, i.e. `DUALWAVE_SWP_KV_PER_BUFFER`. The forward sizes
    # its allocation for `NUM_PREFETCH_K` KV tiles in flight; dQ keeps one. At 512, BLOCK_N 64:
    # 66560 + 69632 = 136192 B against the 163840 B cap. A three-slot layout needed 199 KB there.
    carried = {f.name: getattr(base, f.name) for f in fields(base) if f.name != "LDS_KV_TOTAL_SIZE"}
    traits = Gfx950DqTraits(
        **carried,
        STORE_DB=bool(meta.bias),
        HDIM_VO_FLOOR=knobs._hdim_vo_floor(meta),
        LDS_KV_TOTAL_SIZE=base.DUALWAVE_SWP_KV_PER_BUFFER,
        MFMA_M=m,
        MFMA_N=n,
        MFMA_K=k,
    )
    _check_lds(traits.LDS_KV_TOTAL_SIZE, knobs, "KV staging")
    return traits


# ---------------------------------------------------------------------------
# dK/dV knobs
# ---------------------------------------------------------------------------

# `(num_warps, waves_per_eu, DKV_SHARDS, MFMA_ROWS, BLOCK_M)` by tile width. `BLOCK_N` (KV rows) follows
# as `MFMA_ROWS * num_warps / DKV_SHARDS` and is not stored beside them.
#
# **Four levers, found in this order, and each reframed the last.**
#
# 1. **The AGPR cliff.** At 8 waves (2 per SIMD) a wave may address 256 registers *in total*, so the allocator
#    cannot reach the AGPR file and spills instead. At 128, 8 waves gave 0 AGPRs and 118 spills at 444
#    TFLOP/s; 2 waves gave 108 AGPRs, no spills, 788. At 4 waves the only thing between 403 and 721 was
#    the occupancy *hint*: sweep `(num_warps, waves_per_eu)` as a pair, never the wave count alone.
# 2. **BLOCK_N (KV rows) is a bandwidth lever.** Every workgroup streams the whole of Q and dO for its head,
#    so the read traffic is `seqlen / BLOCK_N` copies of that slab. More waves at one wave per SIMD is free
#    traffic relief (160: 408 -> 690 TF, 224: 486 -> 723). It reframes DKV_SHARDS, which *divides* BLOCK_N.
# 3. **16 rows per wave.** The loop invariant is `0.75 * d` rather than `1.5 * d`, which fits at 512
#    unsharded. It must be `16x16x32`: `16x16x16` is half rate (a family on it measured 280 TF at 64
#    against 713).
# 4. **BLOCK_M (Q rows per streamed tile).** The 16-row transposed operand spans 32 q rows, so it can take
#    either, and at the wide rungs 32 halves the live transposed reads and buys the second stream buffer back
#    (512: 280 TF with 36 spills at 64, 416 with none at 32).
#
# Measured at `B=2 H=8 S=4096` bf16 non-causal, nominal FLOPs:
#
#   head_dim  rows  waves  wpe  BLOCK_M  BLOCK_N  AGPR  spills   TFLOP/s
#      32      32     4     2      64      128        0      0       504
#      64      32     4     1      64      128        0      0       735  (**)
#      96      16     4     1      64       64        0      0       733
#     128      16     4     1      64       64        0      0       775
#     160      32     4     1      64      128      100      0       735  (*)
#     192      16     4     1      32       64        0      0       849
#     224      32     4     1      64      128      230      0       799
#     256      16     4     1      32       64        0      0       743
#     384      16     4     1      32       64       96      0       423
#     512      16     4     1      32       64      214      0       429
#
# **(**) 64's non-causal bias-free build takes `waves_per_eu=2` from `_FEATURE_OVERRIDES`**: the 735 above
# predates the runtime logsumexp layout, after which the rung wants 276 registers and stops reaching two waves
# per SIMD on its own.
# **(*) 160 asks for 1, because 2 was never granted**: the workgroup takes 87040 B, so two want 174080 B against
# the cap. An unmet hint is not automatically inert (it is still a register budget), so the 24 builds of this
# rung were compared at 1 and 2: byte-identical, 24 of 24. It is a build-log change, not a performance change.
# 96, 160 and 224 are the granule-32 rungs; at granule 32 a staged tile has `SMEM_N_RPT = 4` lines, so the wave
# count cannot exceed 4 and the 16-row family's BLOCK_N is capped at 64 against the 32-row family's 128.
_DKDV_GEOMETRY = {
    # head_dim: (waves, waves_per_eu, shards, mfma_rows, block_m)
    32: (4, 2, 1, 32, 64),
    64: (4, 1, 1, 32, 64),
    96: (4, 1, 1, 16, 64),
    128: (4, 1, 1, 16, 64),
    160: (4, 1, 1, 32, 64),
    192: (4, 1, 1, 16, 32),
    224: (4, 1, 1, 32, 64),
    256: (4, 1, 1, 16, 32),
    384: (4, 1, 1, 16, 32),
    512: (4, 1, 1, 16, 32),
}

# Whether the **32-row** body is written for minimum live registers rather than maximum instruction-level
# parallelism. The 16-row body has no such knob: a q group's live set there is two 16x16 accumulators, a
# quarter of what the 32-row body holds. Measured at `B=4 H=8 S=4096` on the 32-row family:
#
#   head_dim   loose            tight
#      32      472 TF, 0 sp     511 TF, 0 sp
#      64      712 TF, 0 sp     703 TF, 0 sp
#     128      744 TF, 0 sp     606 TF, 0 sp
#     224      723 TF, 0 sp     645 TF, 0 sp
#     384      197 TF, 368 sp   283 TF,  31 sp
#
# 32 is the odd one at the narrow end and not a register story: with `D_CHUNKS == 1` there is barely any
# independent work for the loose arm to overlap, so all it does is lengthen live ranges.
_DKDV_TIGHT_REGISTERS = {
    32: True,
    64: False,
    96: False,
    128: False,
    160: False,
    192: False,
    224: False,
    256: False,
    384: True,
    512: True,
}

# **A feature's register cost lands on whichever rung was already at the cap**, and 224 at 32 rows is the
# one: 486 VGPR of 512 with no spills in a plain dense build. A window's mask adds about 19 live registers
# and tips it (512 VGPR, 53 spills at 777 TF loose; 0 spills at 1199 TF tight). **Bias needed four more**,
# and the reason is not register pressure at three of them: a bias read is one scalar `buffer_load` per
# accumulator element, so the 32-row family's 32 dependent loads per tile land in a tile whose MFMA work is
# small, and the fix is a different geometry (64 with bias: 149 TF loose, 520 TF tight, 3.5x).
# **GQA gets no entry**: that is a measurement, not an omission. The right geometry there is a function of the
# group size, the batch and the sequence length, which this table cannot express.
# **No varlen axis in the key**: the decode is unconditional, so there is nothing to twin.
_DKDV_FEATURE_OVERRIDES = {
    # (head_dim, window, bias): (waves, waves_per_eu, shards, mfma_rows, block_m, tight)
    (224, True, False): (4, 1, 1, 32, 64, True),
    # Bias excludes the window by construction, so these can never collide with the one above.
    (32, False, True): (4, 2, 1, 16, 64, False),
    (64, False, True): (4, 1, 1, 32, 64, True),
    (160, False, True): (4, 1, 1, 16, 64, False),
    (224, False, True): (4, 1, 1, 32, 64, True),
    # Three rungs the runtime logsumexp layout moved: the row read now emits both arms and picks once outside
    # the tile loop, and the 32-row family (32 accumulator elements per lane against the 16-row's 4) gives way.
    #   head_dim  64 window        policy  917   16-row      1017
    #   head_dim 224 non-window    policy  473   32 tight     723
    (64, True, False): (4, 1, 1, 16, 64, False),
    (224, False, False): (4, 1, 1, 32, 64, True),
    # **This one asks for the occupancy, not a geometry**: a dense build reaches two waves per SIMD on its own at
    # 234 VGPRs, ours wants 276 and, told `waves_per_eu=1`, LLVM has a 512-register budget and no reason to stop.
    # It buys the second wave for 24 scratch slots: the spills cost 2%, the wave is worth 15%. Unlike 160 this 2
    # *is* grantable; an `amdgpu-waves-per-eu` warning on a 64 dK/dV build means it should go.
    (64, False, False): (4, 2, 1, 32, 64, False),
}


def _granule_for(block_dmodel):
    """The D-axis staging granule for a tile width: 64 on the 64 grid, else 32 (the PV MFMA's 32-column
    output is the floor, so 16 is not available at any width)."""
    return 64 if block_dmodel % 64 == 0 else 32


@dataclass(frozen=True)
class Gfx950DkdvKnobs(_Knobs):
    """The dK/dV kernel's build parameters.

    `BLOCK_N` tiles seqlen_k (KV rows one workgroup owns, resident in registers) and `BLOCK_M` walks Q (the
    rows one streamed tile carries), as in ATI's Triton `bwd_kernel_dk_dv`. C++ therefore reads `BLOCK_N` for the
    grid. `num_warps` and `HEAD_DIM_GRANULE` pin as a set with the tiles. Neither backward kernel uses stagger.
    """

    BLOCK_DMODEL: int | None = None
    BLOCK_DMODEL_V: int | None = None
    PADDED_HEAD: bool | None = None
    BLOCK_M: int | None = None
    BLOCK_N: int | None = None
    num_warps: int | None = None
    HEAD_DIM_GRANULE: int | None = None
    # Per-width policy a sweep wants to vary on its own, so not part of the pinned set.
    DKV_SHARDS: int | None = None
    MFMA_ROWS: int | None = None
    NUM_STREAM_BUFFERS: int | None = None
    # JIT-only, opt-in: bake call data (`Window_left`/`Window_right`, `Max_seqlen_q`/`Max_seqlen_k`) as `Constexpr`s, one
    # compile per value. Off for every AOT build, where they are real `Int32` kernargs.
    STATIC_WINDOW: bool | None = None
    STATIC_SEQLEN: bool | None = None
    TIGHT_REGISTERS: bool | None = None
    waves_per_eu: int | None = None
    daz: bool | None = None
    GRID_AXIS_ORDER: int | None = None

    def resolve(self, meta: FmhaInputMetadata, hints: FmhaHints = FmhaHints()) -> "Gfx950DkdvKnobs":
        _check_bwd_meta(meta, "dK/dV")
        return (
            _DKDV_FALLBACK.merge(self)
            ._with_widths(meta)
            ._with_geometry(meta)
            ._with_buffers()
            ._with_register_pressure(meta)
            ._checked_against_traits(meta)
        )

    def _geometry_for(self, meta):
        """`(waves, waves_per_eu, shards, rows, block_m, tight)` for a build."""
        key = (self.BLOCK_DMODEL, bool(meta.window), bool(meta.bias))
        if key in _DKDV_FEATURE_OVERRIDES:
            return _DKDV_FEATURE_OVERRIDES[key]
        return _DKDV_GEOMETRY[self.BLOCK_DMODEL] + (_DKDV_TIGHT_REGISTERS[self.BLOCK_DMODEL],)

    def _with_geometry(self, meta):
        """Waves, BLOCK_N, the granule and the shard count from the tile width.

        `_DKDV_GEOMETRY` is the whole policy. **BLOCK_N is derived rather than pinned beside the wave count**:
        a wave owns exactly the MFMA's row extent and shards split the waves, not the rows, so
        `BLOCK_N = MFMA_ROWS * num_warps / DKV_SHARDS` and the two cannot disagree.
        """
        if self._check_geometry_pin(("num_warps", "BLOCK_N", "BLOCK_M", "HEAD_DIM_GRANULE")):
            if self.DKV_SHARDS is None or self.MFMA_ROWS is None:
                raise ValueError("a pinned geometry must pin DKV_SHARDS and MFMA_ROWS too; they decide BLOCK_N")
            if self.waves_per_eu is None:
                # The occupancy hint is a per-width table value, not part of the geometry tuple: a pinned geometry
                # takes the table's (as dQ's takes 1) rather than leaving the build unresolved.
                return replace(self, waves_per_eu=self._geometry_for(meta)[1])
            return self
        waves, wpe, table_shards, table_rows, table_bm, _tight = self._geometry_for(meta)
        shards = self.DKV_SHARDS if self.DKV_SHARDS is not None else table_shards
        rows = self.MFMA_ROWS if self.MFMA_ROWS is not None else table_rows
        return replace(
            self,
            num_warps=waves,
            waves_per_eu=self.waves_per_eu if self.waves_per_eu is not None else wpe,
            DKV_SHARDS=shards,
            MFMA_ROWS=rows,
            BLOCK_N=rows * (waves // shards),
            BLOCK_M=table_bm,
            HEAD_DIM_GRANULE=_granule_for(self.BLOCK_DMODEL),
        )

    def _stream_slot_bytes(self):
        """Bytes one staged tile occupies, without building the traits (the buffer count is decided
        before they exist; `_dkdv_traits_for` asserts the two agree)."""
        granule = self.HEAD_DIM_GRANULE
        smem_n_rpt = self.BLOCK_M // (512 // granule)
        line = 512 + 32
        return smem_n_rpt * (self.BLOCK_DMODEL // granule) * line * 2

    def _with_buffers(self):
        """Two stream buffers if LDS allows, one if it does not.

        **The second buffer is the only thing LDS ever costs this kernel**, which is why there is no
        `D_STAGES` here. A staged slot is `68 * head_dim` elements, so two tensors double-buffered are
        `544 * head_dim` bytes: 139264 at 256 and 278528 at 512 against a 163840 cap. Single-buffered they
        are `272 * head_dim`, 139264 at 512, so 512 fits with a whole tile of each tensor resident. What is
        lost at one buffer is the prefetch distance; the body is otherwise identical.
        """
        if self.NUM_STREAM_BUFFERS is not None:
            return self
        return replace(self, NUM_STREAM_BUFFERS=2 if 4 * self._stream_slot_bytes() <= LDS_CAP_BYTES else 1)

    def _with_register_pressure(self, meta):
        """Trade instruction-level parallelism for live registers, or not (`_DKDV_TIGHT_REGISTERS`)."""
        if self.TIGHT_REGISTERS is not None:
            return self
        return replace(self, TIGHT_REGISTERS=self._geometry_for(meta)[5])

    def _checked_against_traits(self, meta):
        if self.STATIC_WINDOW and not meta.window:
            raise ValueError("STATIC_WINDOW bakes the window bounds, so it requires meta.window")
        dkdv_traits(meta, self)
        return self._check_grid_axis_order(GRID_AXIS_HEAD_FASTEST)


_DKDV_FALLBACK = Gfx950DkdvKnobs(daz=True, STATIC_WINDOW=False, STATIC_SEQLEN=False)


def dkdv_traits(meta: FmhaInputMetadata, knobs: Gfx950DkdvKnobs) -> Gfx950DkdvTraits:
    """The dK/dV traits `knobs` (resolved against `meta`) imply."""
    _check_bwd_meta(meta, "dK/dV")
    floor = _floor_for(meta, knobs)
    # dK/dV's two extents share one mask floor, so both must sit above it: an asymmetric call whose V width is at or below
    # the floor promises nothing about it, and the build masks every column (as dQ's V tile does, with its own floor).
    if meta.head_dim_v_real <= floor:
        floor = 0
    if knobs.MFMA_ROWS is None or knobs.num_warps is None or knobs.NUM_STREAM_BUFFERS is None:
        raise ValueError("knobs are not resolved: call `dkdv_knobs(...).resolve(meta)` first")
    if knobs.MFMA_ROWS not in (16, 32):
        raise ValueError(f"MFMA_ROWS must be 16 or 32, got {knobs.MFMA_ROWS}")
    if knobs.MFMA_ROWS == 16 and knobs.DKV_SHARDS != 1:
        # The 16-row family exists so that sharding is unnecessary, and it keeps `a16_chunk_offset` a
        # compile-time immediate by not having a runtime shard origin.
        raise ValueError(
            f"MFMA_ROWS 16 with DKV_SHARDS {knobs.DKV_SHARDS}: the 16-row family does not shard. Its loop "
            "invariant is 0.75*d, which fits unsharded at every rung. Pass DKV_SHARDS=1."
        )
    # `_make_traits` is the forward's, called with block_m=KV rows, block_n=Q rows and vo_shards=DKV_SHARDS.
    # Those three slots change meaning; nothing else does. Reusing VO_SHARDS rather than adding a field is what
    # gets the shard validation (the even-chunk rule the LDS offset decomposition needs) for free.
    base = _make_traits(
        head_dim=knobs.BLOCK_DMODEL,
        head_dim_v=knobs.BLOCK_DMODEL_V,
        num_waves=knobs.num_warps,
        block_m=knobs.BLOCK_N,
        block_n=knobs.BLOCK_M,
        granule=knobs.HEAD_DIM_GRANULE,
        vo_shards=knobs.DKV_SHARDS,
        window=meta.window,
        static_window=bool(knobs.STATIC_WINDOW),
        static_seqlen=bool(knobs.STATIC_SEQLEN),
        bias=meta.bias,
        dropout=meta.dropout,
        lpt_tile_order=False,
        dtype_str=meta.dtype_str,
        waves_per_eu=knobs.waves_per_eu,
        daz=knobs.daz,
        setprio=True,
        stagger=False,  # neither backward kernel uses stagger
        num_kv_splits=1,
        paged=False,
        kv_cache_layout=meta.kv_cache_layout,
        return_lse="never",
        hdim_qk_floor=floor,
    )
    traits = Gfx950DkdvTraits(
        **{f.name: getattr(base, f.name) for f in fields(base)},
        NUM_STREAM_BUFFERS=knobs.NUM_STREAM_BUFFERS,
        MFMA_ROWS=knobs.MFMA_ROWS,
        TIGHT_REGISTERS=bool(knobs.TIGHT_REGISTERS),
    )
    # **The rows-per-wave ceiling's other half.** A wave holds exactly the MFMA's row extent in KV rows: more
    # would address rows the accumulator does not have (`_make_traits` rejects that), fewer would run a full MFMA
    # and store rows the workgroup does not own.
    if traits.ROWS_PER_WAVE != knobs.MFMA_ROWS:
        raise ValueError(
            f"BLOCK_N {knobs.BLOCK_N} over {knobs.num_warps} waves at {knobs.DKV_SHARDS} shards gives "
            f"{traits.ROWS_PER_WAVE} KV rows per wave, and this build's family serves exactly {knobs.MFMA_ROWS}. "
            f"Pass BLOCK_N={knobs.MFMA_ROWS * (knobs.num_warps // knobs.DKV_SHARDS)}."
        )
    if traits.STREAM_TILE_ELEMS * traits.BF16_BYTES != knobs._stream_slot_bytes():
        raise AssertionError(
            f"`_stream_slot_bytes` says {knobs._stream_slot_bytes()} B per slot but the traits derive "
            f"{traits.STREAM_TILE_ELEMS * traits.BF16_BYTES}; the buffer-count decision was made against a "
            "stale copy of the LDS derivation"
        )
    lds_bytes = traits.LDS_STREAM_TOTAL_SIZE * BF16_BYTES
    if lds_bytes > LDS_CAP_BYTES:
        raise ValueError(
            f"Q + dO staging needs {lds_bytes} B of LDS, over the {LDS_CAP_BYTES} B cap, for BLOCK_DMODEL "
            f"{knobs.BLOCK_DMODEL} at BLOCK_M {knobs.BLOCK_M} with {knobs.NUM_STREAM_BUFFERS} buffers. "
            "Drop to one buffer, or lower BLOCK_M."
        )
    return traits


# ---------------------------------------------------------------------------
# Factories and the cache key
# ---------------------------------------------------------------------------


def _factory(cls, kind):
    def make(arch: str = "gfx950", **overrides):
        base = arch.split(":")[0].lower() if arch else ""
        if not base.startswith("gfx95"):
            raise ValueError(f"the gfx950 {kind} kernel serves gfx95x only, got arch {arch!r}")
        known = {f.name for f in fields(cls)}
        unknown = set(overrides) - known
        if unknown:
            raise TypeError(f"unknown {cls.__name__} field(s): {sorted(unknown)}")
        return cls(**overrides)

    make.__name__ = make.__qualname__ = f"{kind}_knobs"
    make.__doc__ = f"The `{cls.__name__}` for `arch` (a gcnArchName prefix match) with `overrides` pinned."
    return make


fwd_knobs = _factory(Gfx950FwdKnobs, "fwd")
dq_knobs = _factory(Gfx950DqKnobs, "dq")
dkdv_knobs = _factory(Gfx950DkdvKnobs, "dkdv")


def build_cache_key(traits, knobs):
    """The JIT cache key of a build: every trait field **and** every knob (the knobs that are not traits
    -- RETURN_LSE, STATIC_WINDOW, daz, ... -- change the binary too). Keyed by name, so a reordering or an
    added field cannot alias two builds."""
    return (type(traits).__name__, traits_cache_key(traits), tuple(sorted(knobs.as_psels().items())))
