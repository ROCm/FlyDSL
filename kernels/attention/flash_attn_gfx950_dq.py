# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Flash-attention **backward dQ (and dB)** for gfx950 -- AOTriton's `bwd_kernel_dq`.

B2 and B3 of `sdpa-bwd-plan-gfx950.md`. Dense, non-causal, bf16, the full
`LADDER` from head_dim 32 to 512, with the 8xD input contract, padded heads and
asymmetric `hdim_qk`/`hdim_vo`. The dK/dV half is B1 and lives in its own file;
there is deliberately no fused kernel (plan section 4).

    P  = exp2(qk_scale * S - lse2)        S  = Q.K^T,  qk_scale = sm_scale*log2e
    dP = dO . V^T
    dS = P * (dP - delta)                 delta = rowsum(dO*O), a host argument
    dQ = sum_j dS . K * sm_scale
    dB = dS                               (bias-gradient builds only)

--- The whole design, in one observation --------------------------------------

**Every one of the three GEMMs is a GEMM the forward already emits**, once the
right tensor is staged in the right LDS layout. Nothing below derives a new
lane map, and that is the point: the forward's `V` read path is validated end
to end, and plan section 3 says to reuse it rather than re-derive it.

| this kernel | forward equivalent | A operand (M axis) | B operand (N axis) |
|---|---|---|---|
| `S  = Q.K^T`  | `qk` | K, `load_k` | Q, `ParityQLoader` |
| `dP = dO.V^T` | `qk` | V, `load_k` | dO, same loader |
| `dQ = dS.K`   | `pv` | K^T, `load_v` | dS, packed like P |

The score-shaped accumulators are all `[kv][q]` -- M is the KV token, N is the
query row -- which is exactly the forward's `v_s`, so `seq_pad_mask_if_needed`,
`sub_m`, `exp2` and `cast_p` apply unchanged. `dS` reaches the third GEMM
through `cast_p`, so it carries the *same* K permutation `P` does, which is why
`load_v`'s transpose read lands K in the operand registers the MFMA wants with
no further shuffle.

--- Two LDS tiles, and the read that makes it two ------------------------------

GEMM1 wants K with the KV token on the MFMA's M axis and `d` contracted; GEMM3
wants K with `d` on M and the KV token contracted. Those are the forward's K
path and its V path, and the two stagings differ **only in the line padding**
(`SMEM_K_PAD` 8 elements against `SMEM_V_PAD` 32). B2 answered that by staging
K twice. B3 cannot: three slots is 199 KB at head_dim 512 against a 163840 B
cap, so the double staging is what stops the ladder rather than merely costing
DMA.

So the K tile is staged **once, in the V layout**, and read two ways:

    (V, buf 0) <- K   -> load_v(0)          GEMM3, the stock transpose read
                      -> load_k_packs()     GEMM1, the K path on V-pitch lines
    (K, buf 0) <- V   -> load_k_packs()     GEMM2, the stock K path
    buf 1                unused, and not allocated

`LDS_KV_TOTAL_SIZE` drops to one buffer -- `SMEM_K_TILE_ELEMS +
SMEM_V_TILE_ELEMS`, 133 KB at head_dim 512 -- which is a single trait field
(`BwdDqTraits`, `make_bwd_dq_traits`).

**The K tile is the one in the V region, not the other way round, and that is
deliberate.** One of the two readers has to be re-pointed at the other pitch,
and the K path is a plain `llvm.LoadOp` whose whole addressing is two
expressions; the transpose path is `ds_read_b64_tr_b16` with alias scopes, an
even-VGPR-pair constraint and two open hazards against it (`sdpa_lore_gfx950`).
Leave the fragile one stock. `BwdDqKvLdsToVgprLoader` re-points the other by
scoping two trait fields, so the *formula* is still the shared one.

The alias scopes stay truthful: the K tile is read under `lds_v0` by both of
its readers and the V tile under `lds_k0`, so no scope claims two regions are
disjoint when they are the same memory.

--- Conventions this kernel must not get wrong --------------------------------

Three, and each is a *silent* wrong answer if broken, which is why the tests
check them against our own forward rather than only against torch:

- **`qk_scale = sm_scale * log2e` is applied to the f32 scores, `sm_scale`
  alone to the dQ accumulator at the end.** AOTriton spells them
  `p = exp2(qk_scale*qk - l_i)` and `dq *= sm_scale`. The forward used to fold
  `qk_scale` into Q and round the product back to bf16 while this kernel
  deliberately did not; `BwdDqSoftmaxHelper.scale_and_sub_lse` has the
  measurement that says why, and the forward has since been moved onto the
  same arithmetic (`ParityGemmHelper.scale_scores`). The two now agree.
- **Neither Q nor `dO` is pre-scaled.** `ParityQLoader.scale_all` now raises
  rather than scaling, so the Q half of this is enforced; `dO` is not, and
  pre-scaling it multiplies every gradient by `qk_scale` with nothing checking
  shapes.
- **LSE is read in natural units and converted here.** The forward writes
  `m_row*ln2 + ln(l)`, so `lse2 = lse * log2e` (AOTriton's `l_i = ... *
  RCP_LN2`). `common.lse_row_addressing` owns the layout for both LSE and delta.

--- The two head extents are crossed, and that is not a detail ----------------

In the forward, the tensor read through the K register path has the *qk*
extent and the tensor written through the O store has the *vo* one. **Both are
the other way round here**, because GEMM2 reads V through the K path and the
store writes dQ, which is Q-shaped:

- two `BwdDqKvLdsToVgprLoader` instances, one per tile, differing in the LDS
  pitch *and* in which extent their padded-head mask is written against
  (plus `HDIM_VO_FLOOR`, the vo counterpart of `HDIM_QK_FLOOR`);
- `BwdDqStoreHelper`, which points the inherited store at dQ's descriptor,
  stride, sentinel and `hdim_qk` width.

They coincide in every symmetric build, so only `test_asymmetric_hdim` can tell
the fix from its absence.

--- B3.5: what a 16-row family still needs, after the shape is named -----------

`BwdDqTraits` now names `MFMA_M/N/K` and derives `ACC_ELEMS`,
`SCORE_MSTEPS` and `OPERAND_LANE_ELEMS` from them, and every literal in this
file that meant one of those is now written as one. That change is a no-op
today -- the ISA is byte-identical, same 260 VGPRs at head_dim 128 -- so what
is left is exactly the list below, and nothing else.

**Two findings that do not need the lane-map probe**, both in
`flash_attn_gfx950_config`:

- `v_mfma_f32_16x16x16_bf16` is **4 passes for 8192 flops** against
  `32x32x16`'s **8 for 32768** (`SISchedule.td`, `SIDPGFX950FullSpeedModel`).
  It is *half rate*, so a family built on it cannot beat the 32-row one
  anywhere the 32-row one fits. `16x16x32` is 4 passes for 16384 -- full rate,
  16 rows.
- `register_demand` says 16 rows takes head_dim 384 from 144 registers over the
  file to **fitting**, and 512 from 336 over to **40 over**. So 16 rows
  finishes 384 and leaves 512 needing one more lever.

**16x16x32 is also structurally the cheaper port**, and that is a third
argument for it: `OPERAND_LANE_ELEMS` stays 8, so `_pack_p_v8_slices`,
`_bf16_trunc_pack_v8` and the v8 shape of every LDS read are unchanged. At
16x16x16 all of them become 4-wide.

What still has to be re-derived, in dependency order:

1. **`SCORE_MSTEPS == 2` is structural, not parametric.** The softmax path is
   written as an explicit `(s_lo, s_hi)` *pair* throughout
   `flash_attn_utils` -- `_score_pair_to_lists`, `_reduce_score_pair`,
   `_sub_score_pair`, `_exp2_score_slice`, `_pack_p_v8_slices`,
   `seq_pad_mask_inplace`. At 16 rows and `BLOCK_N` 64 there are four M steps.
   **`BLOCK_N` 32 keeps the pair, and it is also what closes head_dim 512.**
   `register_demand` at `(16 rows, BLOCK_N 32)` is 404 at head_dim 512 and 308
   at 384, against 552 and 424 at `BLOCK_N` 64 -- so the cheapest port and the
   register fix are the same choice, and there is no need to generalise the
   pair. Settle this before anything else.
2. `_seq_pad_score_threshold` -- the element -> KV column map, derived from the
   32x32 accumulator layout. The KV tail mask and the dB store both read it,
   so they stay consistent for free once it is right. `flash_attn_gfx950_helpers`
   gives the replacement: a lane's four accumulator elements are four
   *contiguous* KV tokens at `4 * (lane // 16)`, where at 32 rows they are the
   scattered `8*(i//4) + 4*(lane//32) + (i%4)`.
3. The `permlane32_swap` O store (`_fused_o_128_dwords`): it transposes the
   `[d][q]` accumulator for a 128-bit row-major store, and `permlane32_swap` is
   a 32-lane primitive.
4. `_init_dualwave_q_row`'s `q_row = ... + lane_mod_32`, which the LSE/delta
   row load reads. At 16 rows it is `lane % 16`; the two row inputs stay
   per-lane scalars, because the query row is the accumulator's *n* axis here
   (it is dK/dV, where the query row is *m*, that gets the `dwordx4`).
5. The transpose read for GEMM3. **Answered by the probe, and favourably**
   (`flash_attn_gfx950_helpers`): it serves both 16-row shapes from *one* staged tile,
   so the four-LDS-tile risk does not materialise. Better than that for this
   kernel -- **the 16-row maps carry no permutation on the contraction index**,
   where the 32-row one does. GEMM3 currently works because `cast_p`'s pack
   order happens to reproduce the 32-row read's `k` permutation; at 16 rows
   `dS` only has to be in plain order, so that coincidence stops being
   load-bearing.

Items 2-4 live in `flash_attn_utils.py`, which is imported and never edited, so
each becomes an override here.

--- f16 and bf16, and where the operand dtype actually reaches ----------------

Both are supported; `BwdDqKnobs.build_traits` refuses anything else by name,
matching gfx1201's `assert dtype_str in ("f16", "bf16")`.

**The operand dtype reaches only two places**, and knowing that is what makes
f16 cheap here: the `dS` pack that feeds GEMM3's B operand
(`_bf16_trunc_pack_v8`, which already branches on `DTYPE_STR`) and the dQ
store. Everything with an exponent in it -- the score, `lse2`, `delta`, the
softmax, `dP`, the dropout survivor scale, the bias add -- is **f32 in both
builds**. So f16's narrower range (65504 against bf16's fp32 range) is exposed
at those two narrowings and nowhere else.

Two consequences worth stating rather than rediscovering:

- The `lse2` floor and the dropout ordering were both derived under bf16, and
  both carry, for the same reason: they are about f32 intermediates. See
  `BwdDqKernelContext.load_row_scalars`.
- **The dropout survivor scale is applied to the f32 `dP`, not folded into
  `dO`.** B6 chose that on precision grounds -- folding would round `dO`
  through the operand type a second time -- and it is also the f16-safe
  choice, because `1/(1-p)` on an f16 operand is a multiply into a 65504
  ceiling where on f32 it is not.

What is *not* guarded, deliberately and in line with AOTriton and gfx1201: a
`dS` or `dQ` magnitude beyond 65504 saturates in an f16 build where a bf16
build would carry it. That needs inputs far outside anything these tests or a
trained model produce, and clamping it would cost a per-element `min` on the
hot path to change one implausible case into a different wrong answer.

--- Not implemented, deliberately ---------------------------------------------

Causal, windows, varlen, dropout, bias *input*, split-K, paged, `D_STAGES`,
d-axis sharding. `BwdDqKnobs.build_traits` refuses each by name rather than
ignoring it -- every one of them would otherwise build, run and return a
correctly-shaped wrong answer, and `D_STAGES` nearly did: the forward's knob
policy turns it on above head_dim 256 and the inherited GEMM helpers then
reduce over one stage of the head dim while this loop never advances the stage.

--- Tensor argument order is the ABI ------------------------------------------

    q, k, v, b, do, dq, db, lse, delta

Four groups, and the grouping is the mnemonic: the **forward's inputs**
(`q, k, v, b`), then the **backward's input** (`do`), then this kernel's
**outputs**, then the **lower-rank** tensors (`lse`, `delta`, both rank 2).

It is also AOTriton's order -- `bwd_kernel_dq(Q, K, V, B, sm_scale, DO, DQ, DB,
L, D, ...)` and `bwd_kernel_dk_dv(Q, K, V, B, sm_scale, DO, DK, DV, L, D, ...)`
-- so a reader moving between the Triton reference and this file does not have
to re-derive the mapping, and neither does anyone eventually dispatching the
compiled hsaco directly.

`b` sits in the forward-input group even though no build here reads it yet: the
slot is held so adding bias in B6 does not move the wire format. It was
initially placed after `delta`, at the end, which is where an unused argument
naturally lands and is exactly why the grouping is written down rather than
left to accrete.
"""

from dataclasses import replace

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value
from kernels.attention import abi, common
from kernels.attention import flash_attn_gfx950 as fwd_module
from kernels.attention import flash_attn_gfx950_config as config
from kernels.attention import flash_attn_utils as dualwave
from kernels.attention.flash_attn_gfx950_helpers import (
    MFMA16_M,
    ParityGemmHelper,
    ParityKernelContext,
    ParityKvGmemToLdsLoader,
    ParityKvLdsToVgprLoader,
    ParityQLoader,
    ParitySoftmaxHelper,
    ParityStoreHelper,
    _bias_slab_num_records_bytes,
    _score_column_runs,
    _slab_span_elems,
    exp2_wait_state,
    fadd,
    fmax,
    fmul,
    fsub,
    mfma_operand_wait_state,
    wire_ptr,
    wire_view,
)
from kernels.attention.flash_attn_gfx950_helpers import _k_read_base as _parity_k_read_base
from kernels.attention.flash_attn_gfx950_helpers import _ks_offset as _parity_ks_offset
from kernels.attention.philox import Philox
from kernels.common import buffer_ops

_ds_read_tr_v4f16_imm = dualwave._ds_read_tr_v4f16_imm

KERNEL_NAME = config.DQ_KERNEL_NAME

__all__ = [
    "KERNEL_NAME",
    "BwdDbStoreHelper",
    "BwdDqKernelContext",
    "BwdDqKvGmemToLdsLoader",
    "BwdDqKvLdsToVgprLoader",
    "BwdDqSoftmaxHelper",
    "BwdDqStoreHelper",
    "BwdRowInputLoader",
    "BwdSecondaryQLoader",
    "build_flash_attn_gfx950_dq",
]


# ============================================================================
# The 16-row dQ body (MFMA 16x16)
# ============================================================================
#
# The dQ backward body at **16 rows per wave**, on `v_mfma_f32_16x16x32_bf16`.
#
# B3.5. A second MFMA family beside the 32-row one in `fmha_bwd_dq_gfx950.py`,
# not a replacement: the 32-row family wins below head_dim 256 and this one
# exists for the top of the ladder, where B3 left dQ at 303 TF (384) and 110
# (512) against ~730 in the middle.
#
# --- Why 16 rows, and why this shape ------------------------------------------
#
# `register_demand` in `flash_attn_gfx950_config` states the problem exactly. At
# 32 rows the loop-invariant Q + dO + dQ is `1.0 * head_dim` registers per lane,
# which is the whole 512-register file at head_dim 512 *before a single operand*.
# At 16 rows it is `0.5 * head_dim`. Nothing else moves that term -- it is
# loop-carried, so no amount of staging or scheduling touches it.
#
# The shape follows from the gfx950 scheduling model rather than from the lane
# map (`SISchedule.td`, `SIDPGFX950FullSpeedModel`): `16x16x16` is 8192 flops in
# 4 passes and `32x32x16` is 32768 in 8, so **`16x16x16` is half rate**.
# `16x16x32` is 16384 in 4 -- 16 rows at full rate. It is also the cheaper port,
# because a lane still holds 8 bf16 of an operand, so `_bf16_trunc_pack_v8` and
# the v8 shape of every LDS read carry over.
#
# `BLOCK_N` is **32**, and that is two decisions at once. It closes the last 40
# registers at head_dim 512 (552 -> 404 predicted), and it keeps
# `BLOCK_N / MFMA_M == 2`, so the score tile is still the `(s_lo, s_hi)` *pair*
# that `flash_attn_utils` is written around -- which matters because that file is
# shared production code and must not be edited.
#
# --- What is reused, and what could not be ------------------------------------
#
# **The LDS staging is untouched.** `BwdDqKvGmemToLdsLoader` stages K into the V
# region and V into the K region exactly as it does for the 32-row family; only
# the register reads change. So is the context, the descriptors, the padded-head
# contract and the dB store's column map.
#
# What could not be reused is everything shaped by the accumulator, because
# `flash_attn_utils` writes `range_constexpr(16)` for "elements in a score half"
# and here it is 4: `_score_pair_to_lists`, `_sub_score_pair`,
# `_exp2_score_slice`, `_pack_p_v8_slices`. Each is a handful of lines and is
# re-spelled below against `ACC_ELEMS`.
#
# --- The one permutation, and where it was put --------------------------------
#
# The score accumulator gives lane `j` (group `g = j // 16`) the KV tokens
# `{4g+e}` from `s_lo` and `{16+4g+e}` from `s_hi`. GEMM3's B operand wants 8
# contraction values per lane, and the *published* transpose map
# (`flash_attn_gfx950_helpers`) delivers `k = 8g + i` -- a different set. One of the two
# has to bend.
#
# **The transpose read bends**, exactly as it does in the 32-row family, and the
# alternative was considered and rejected: permuting GEMM1/GEMM2's token order
# instead would leave the accumulator holding a scrambled token map, which the KV
# tail mask and the dB store both read. Bending one read keeps the natural map
# everywhere else.
#
# The required order is `T(g, 4r + i) = 4g + 16r + i`, which is separable, so it
# costs nothing: the group term becomes `4 * (lane // 16)` tokens instead of
# `8 * (lane // 16)`, and the second read sits at `+16` tokens instead of `+4`.
# `_kt_read_base` below. The result was checked against a host reference before
# anything else was written.

_s_barrier = dualwave._s_barrier
_s_waitcnt = dualwave._s_waitcnt
_sched_barrier = dualwave._sched_barrier

# The MFMA's contraction extent. 32 here; the 8 bf16 a lane holds of an operand
# are `MFMA_K * MFMA_M / WARP_SIZE`, which is the same 8 the 32-row family uses
# and the reason its packing helpers carry over.
MFMA16_K = 32
LANE_ELEMS = 8


def acc_elems(traits):
    """f32 the accumulator gives one lane per MFMA. 4 here, 16 at 32 rows."""
    return MFMA16_M * MFMA16_M // traits.WARP_SIZE


def score_msteps(traits):
    """MFMA steps along the KV axis of one score tile. **Must be 2.**

    The `(s_lo, s_hi)` pair is not a parameter of `flash_attn_utils`, it is its
    shape -- so `BLOCK_N` is chosen to keep this at 2 rather than the softmax
    path being generalised. Asserted rather than commented for the reason P7
    gives about the rows-per-wave ceiling.
    """
    steps = traits.BLOCK_N // MFMA16_M
    if steps != 2:
        raise ValueError(
            f"BLOCK_N {traits.BLOCK_N} over a {MFMA16_M}-row MFMA gives {steps} score steps, and the "
            "softmax path is written as an (s_lo, s_hi) pair. Pass block_n=32."
        )
    return steps


def k_steps16(traits):
    """MFMA steps along the head dim, for the two d-contracted GEMMs."""
    return traits.HEAD_DIM // MFMA16_K


def d_chunks16(traits):
    """16-wide output chunks of dQ one wave accumulates."""
    return traits.HEAD_DIM // MFMA16_M


class M16DqReader:
    """The three LDS reads this body needs, all from the two staged tiles.

    Instantiated per tile, because the tiles differ in *both* the LDS line
    pitch and the padded-head extent -- the same two-instance split the 32-row
    family uses, and for the same reason.
    """

    def __init__(self, ctx, *, region, line_stride, hdim, hdim_floor):
        self.ctx = ctx
        self.traits = ctx.traits
        self.region = region  # "k" or "v": which of the two staged regions
        self.line = line_stride
        self.hdim = hdim
        self.hdim_floor = int(hdim_floor)
        self.base_elems = (
            dualwave._v_buf_base(self.traits, 0) if region == "v" else dualwave._k_buf_base(self.traits, 0)
        )
        self.scope = dualwave._dualwave_lds_scope(region, 0)

    # -- addressing ---------------------------------------------------------

    def _tok_off(self, t):
        """`flash_attn_gfx950_helpers.tok_off`, over *this* region's pitch.

        The published helper is written against `SMEM_V_LINE_STRIDE` because
        the forward only ever transposes the V region. This kernel reads a
        K-pitch region the same way, so the pitch is a parameter here.
        """
        n = self.traits.SMEM_N_RPT
        return (t // n) * self.traits.D_128B_SIZE + (t % n) * self.line

    def _tok_off_dyn(self, t):
        n = fx.Index(self.traits.SMEM_N_RPT)
        return (t // n) * fx.Index(self.traits.D_128B_SIZE) + (t % n) * fx.Index(self.line)

    def _d_off(self, d):
        g = self.traits.D_128B_SIZE
        return (d // g) * self.traits.SMEM_N_RPT * self.line + (d % g)

    # -- GEMM1 / GEMM2: A operand with the KV token on M --------------------

    def a_token_base(self, lane):
        """Per-lane base for an A operand whose `m` axis is the KV token.

        `A[m][k]` with `m` the token and `k` the head dim, so lane `l` holds
        token `l % 16` and the eight d columns `8 * (l // 16) + i`. The token
        term is `tok_off(l % 16)` written out: with `SMEM_N_RPT == 4` at
        `BLOCK_N` 32 that is `((l%16)//4) * granule + (l%4) * line`, and the
        M-step then adds a whole `16` tokens, which is `4 * granule` and
        constant.
        """
        n = fx.Index(self.traits.SMEM_N_RPT)
        lane16 = lane % fx.Index(MFMA16_M)
        return (lane16 // n) * fx.Index(self.traits.D_128B_SIZE) + (lane16 % n) * fx.Index(self.line)

    def load_a_tokens(self, lane, step, ks):
        """One A-operand pack for (M-step, k-step): 8 contiguous d out of LDS."""
        d0 = LANE_ELEMS * (lane // fx.Index(MFMA16_M)) + fx.Index(ks * MFMA16_K)
        idx = (
            fx.Index(self.base_elems)
            + self.a_token_base(lane)
            + fx.Index(self._tok_off(step * MFMA16_M))
            + self._d_off_dyn(d0)
        )
        return self._lds_pack(idx)

    def _d_off_dyn(self, d):
        g = fx.Index(self.traits.D_128B_SIZE)
        return (d // g) * fx.Index(self.traits.SMEM_N_RPT * self.line) + (d % g)

    def _lds_pack(self, elem_idx):
        traits = self.traits
        ptr = buffer_ops.get_element_ptr(
            self.ctx.lds_kv_base_ptr, byte_offset=elem_idx * traits.BF16_BYTES, elem_type=T.i8
        )
        return llvm.LoadOp(
            self.ctx.kv_mfma_pack_type,
            ptr,
            alignment=16,
            alias_scopes=dualwave._dualwave_lds_alias_scopes(self.scope),
            noalias_scopes=dualwave._dualwave_lds_noalias_scopes(self.scope, traits.LDS_SCOPE_NAMES),
        ).result

    def load_a_all(self, lane):
        """Every A pack for one tile: `[step][ks]`, padded-head masked.

        Materialised rather than streamed. `register_demand` says one live A
        tile is `BLOCK_N * d / 128` = 128 registers at head_dim 512 with
        `BLOCK_N` 32, which the 16-row budget has room for -- that is the whole
        point of the smaller tile.

        The mask is the 32-row loader's argument transposed onto this lane map:
        masking the *other* operand (Q, dO) is enough for a finite pad because
        `0 * x == 0`, and not enough for a pad holding NaN, which the caller's
        D-axis slack is entitled to contain. Steps whose columns all lie at or
        below `HDIM_QK_FLOOR` are skipped -- the dispatcher guarantees they are
        real data -- which is what keeps a padded build off the 27-54% the
        forward measured for masking every step.
        """
        traits = self.traits
        steps = k_steps16(traits)
        packs = [[self.load_a_tokens(lane, s, ks) for ks in range_constexpr(steps)] for s in (0, 1)]
        if const_expr(not self.ctx.PADDED_HEAD):
            return packs
        masked = [ks for ks in range(steps) if (ks + 1) * MFMA16_K > self.hdim_floor]
        if const_expr(not masked):
            return packs
        cols = common.MaskedAxis(
            fx.Index(self.hdim), elem_dtype=self.ctx.elem_dtype, bitmask=len(masked) * (LANE_ELEMS // 2) <= 16
        )
        for ks in masked:
            col0 = LANE_ELEMS * (lane // fx.Index(MFMA16_M)) + fx.Index(ks * MFMA16_K)
            for st in (0, 1):
                packs[st][ks] = cols.discard(packs[st][ks], col0, LANE_ELEMS)
        return packs

    # -- GEMM3: A operand with the head dim on M, through the transpose -----

    def _kt_read_base(self, lane):
        """Per-lane base for the transposed A operand, in the accumulator's token order.

        The published `a16_read_base` puts `quad * (lane // 16)` tokens in the
        group term, which delivers `k = 8g + i`. This body needs
        `k = 4g + 16r + i`, because that is the order the score accumulator
        already holds (see the module docstring). Both terms are separable and
        `tok_off` is additive across them here -- `(4g + c) // 4 == g` and
        `(4g + c) % 4 == c` for `c < 4 == SMEM_N_RPT` -- so the change is the
        group multiplier and nothing else.
        """
        traits = self.traits
        g = lane // fx.Index(MFMA16_M)
        c = (lane % fx.Index(MFMA16_M)) // fx.Index(4)
        return (
            g * fx.Index(traits.D_128B_SIZE)  # tok_off(4 * g), with SMEM_N_RPT == 4
            + c * fx.Index(self.line)  # tok_off(c)
            + (lane % fx.Index(4)) * fx.Index(4)  # the four d columns this lane fetches
        )

    def load_kt(self, lane):
        """`d_chunks16` transposed A packs, one per 16-wide output chunk."""
        traits = self.traits
        base = fx.Index(self.base_elems) + self._kt_read_base(lane)
        # `+16` tokens, which is `MFMA16_M`, not `2 * MFMA16_M`: the required
        # order is `T(g, 4r + i) = 4g + 16r + i`, so read `r = 1` is one M-step
        # of tokens further on. Getting this wrong reads past the tile and
        # returns finite garbage -- it did.
        pair_bytes = self._tok_off(MFMA16_M) * traits.BF16_BYTES
        packs = []
        for c in range_constexpr(d_chunks16(traits)):
            imm = self._d_off(c * MFMA16_M) * traits.BF16_BYTES

            def read(off):
                return _ds_read_tr_v4f16_imm(
                    base,
                    off,
                    lds_kv_base_idx=self.ctx.lds_kv_base_idx,
                    v_lds_read_vec4_type=self.ctx.v_lds_read_vec4_type,
                    scope_name=self.scope,
                    scope_names=traits.LDS_SCOPE_NAMES,
                )

            lo, hi = read(imm), read(imm + pair_bytes)
            packs.append(Vec(lo).shuffle(Vec(hi), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value())
        return packs


class M16QLoader:
    """Q or dO as a B operand at 16 rows.

    `B[k][n]`: lane holds `n = q row = lane % 16` and the eight head-dim
    columns `8 * (lane // 16) + i`, so one 128-bit `buffer_load` per k-step --
    the same shape as the 32-row loader, with both lane terms re-based from 32
    to 16.
    """

    def __init__(self, ctx, div, stride_seq_v, gmem_elem_offset, hdim):
        self.ctx = ctx
        self.traits = ctx.traits
        self.div = div
        self.stride = stride_seq_v
        self.gmem = gmem_elem_offset
        self.hdim = hdim

    def load_all(self, q_row_in_block, lane):
        traits = self.traits
        packs = []
        for ks in range_constexpr(k_steps16(traits)):
            col = LANE_ELEMS * (lane // fx.Index(MFMA16_M)) + fx.Index(ks * MFMA16_K)
            raw = dualwave._buffer_load_128(
                self.gmem + q_row_in_block * self.stride + col,
                _load_atom_128=self.ctx.load_atom_128,
                q_div=self.div,
                q_load_i32x4_type=self.ctx.q_load_i32x4_type,
            )
            pack = Vec(raw, (4,), fx.Int32).bitcast(self.ctx.elem_dtype).ir_value()
            if const_expr(self.ctx.PADDED_HEAD):
                # Same argument as the 32-row loader's: a zero in the reduction
                # operand annihilates whatever the other side holds at that
                # column, and the pad's contents are the caller's business.
                pack = common.MaskedAxis(fx.Index(self.hdim), elem_dtype=self.ctx.elem_dtype, bitmask=True).discard(
                    pack, col, LANE_ELEMS
                )
            packs.append(pack)
        return packs


class M16DqGemmHelper:
    """The three GEMMs, on `v_mfma_f32_16x16x32_bf16`."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.traits = ctx.traits
        self.atom = fx.make_mma_atom(fx.rocdl.MFMA(MFMA16_M, MFMA16_M, MFMA16_K, ctx.elem_dtype))
        self.acc_ty = Vec.make_type(acc_elems(ctx.traits), fx.Float32)
        self.zero = Vec.filled(acc_elems(ctx.traits), 0.0, fx.Float32)

    def _mfma(self, a, b, c):
        return dualwave._mfma_acc(a, b, c, self.atom, self.acc_ty)

    def qk(self, a_packs, b_packs):
        """`S = A . B^T` over the head dim, for both KV M-steps.

        GEMM1 with `(K, Q)` and GEMM2 with `(V, dO)` are the same call: the
        operands differ, the contraction does not.
        """
        out = []
        for step in (0, 1):
            acc = self.zero
            for ks in range_constexpr(k_steps16(self.traits)):
                acc = self._mfma(a_packs[step][ks], b_packs[ks], acc)
            out.append(acc)
        # The head_dim 96 workaround the 32-row family carries. Same producer
        # (the QK burst) and same near-zero cost; kept because the underlying
        # defect is latent rather than fixed. See `sdpa_lore_gfx950.md`.
        dualwave._s_nop(1)
        return tuple(out)

    def pv(self, ds_pack, kt_packs, v_dq):
        """`dQ += K^T . dS`, one MFMA per 16-wide output chunk.

        One k-step: `BLOCK_N` is 32 and the MFMA contracts 32, so the whole KV
        tile is a single step -- where the 32-row family needs four substeps.
        """
        for c in range_constexpr(d_chunks16(self.traits)):
            v_dq[c] = self._mfma(kt_packs[c], ds_pack, v_dq[c])
        return v_dq


class M16DqSoftmax:
    """`P`, `dS` and the KV tail mask over a 4-element score half.

    Every method here is a `flash_attn_utils` helper re-spelled against
    `ACC_ELEMS`; that file hardcodes 16 and is production code that must not be
    edited. They are short because the pair structure is preserved -- only the
    element count changed.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.traits = ctx.traits
        self.n = acc_elems(ctx.traits)
        score_msteps(ctx.traits)  # the (s_lo, s_hi) precondition, checked once

    def to_lists(self, v_s):
        return ([Vec(v_s[0])[r] for r in range_constexpr(self.n)], [Vec(v_s[1])[r] for r in range_constexpr(self.n)])

    def to_vecs(self, lists):
        return tuple(Vec.from_elements([as_mlir_value(v) for v in half], fx.Float32).ir_value() for half in lists)

    def scale_and_sub_lse(self, v_s, qk_scale, lse2):
        """`qk_scale * S - lse2`, one FMA per element.

        B2's finding, unchanged: folding `qk_scale` into Q and rounding back to
        bf16 puts `|S| * 2^-8` into the *exponent*, which `dQ` does not
        normalise away. Measured at 10.9x the error ratio at `sm_scale` 1.0.
        """
        fm = self.ctx.fm_fast
        scale_v = Vec.from_elements([fx.Float32(qk_scale)], fx.Float32).broadcast_to(self.n)
        neg = fsub(self.ctx.c_zero_f, lse2, fm)
        neg_v = Vec.from_elements([fx.Float32(neg)], fx.Float32).broadcast_to(self.n)
        return tuple(as_mlir_value(fmath.fma(Vec(h), scale_v, neg_v, fastmath=fm)) for h in v_s)

    def kv_col(self, tile_idx, lane, step, i):
        """The KV column of score element `(step, i)`.

        A lane's four accumulator elements are four **contiguous** tokens at
        `4 * (lane // 16)` -- the 16-row map from `flash_attn_gfx950_helpers`, and
        much simpler than the 32-row one, whose four runs of four are why
        `_score_column_runs` exists. The KV tail mask and the dB store both
        read this, so they cannot disagree.
        """
        return (
            fx.Int32(tile_idx * self.traits.BLOCK_N)
            + fx.Int32(step * MFMA16_M + i)
            + fx.Int32(4 * (lane // fx.Index(MFMA16_M)))
        )

    def add_bias(self, v_s, tile_idx, lane, q_row):
        """`S += bias * log2e`, from this family's own column map.

        A lane's four accumulator elements are four **contiguous** KV columns
        at `4 * (lane // 16)`, so each half is a single 4-wide `buffer_load`.
        The 32-row path needs `_score_column_runs` because its sixteen are four
        scattered spans; reusing that table here would be a transcription.

        **Not `fm_fast`.** That carries `ninf`, a licence to assume no
        infinities reach the operation -- and a bias entry of `-inf` is how a
        caller spells "never attend here", so this is the first thing in the
        kernel that puts a real infinity into *arithmetic* rather than into a
        select. Two operations per element, one by a constant, so getting the
        flag right costs nothing measurable.

        The descriptor is bounded at the (batch, head) slab, so a read past the
        last row returns zero rather than faulting -- which is also the right
        bias for a padded row -- and a column past `seqlen_kv` picks up the
        next row's entry, which the KV tail mask below then overwrites with
        `-inf`. That is the forward's argument, unchanged.

        **2-byte reads, one column at a time, not one `dwordx2` per run.**
        gfx950 range-checks a multi-dword buffer op per dword and drops any
        dword not wholly inside `num_records`, and the slab ends exactly on the
        last valid element -- so a wide read of the last row with odd
        `seqlen_kv` loses column `seqlen_kv-1` to the out-of-range column
        `seqlen_kv` sharing its dword. The forward narrows only on its
        tail-masked tiles; here every tile takes the mask (see the call at the
        `M16TileBody` loop), so there is no wide case to keep.
        """
        ctx = self.ctx
        fm = arith.FastMathFlags.contract | arith.FastMathFlags.reassoc
        log2e = fx.Float32(dualwave._LOG2E)
        row_base = q_row * fx.Index(ctx.stride_b_seq_q)
        lists = self.to_lists(v_s)
        for step in range_constexpr(2):
            col0 = self.kv_col(tile_idx, lane, step, 0)
            for i in range_constexpr(self.n):
                one = buffer_ops.buffer_load(
                    ctx.bias_rsrc,
                    as_mlir_value(fx.Int32(row_base + fx.Index(col0 + i))),
                    vec_width=1,
                    dtype=ctx.elem_dtype,
                )
                # `vec_width=1` returns a **scalar**, not a one-lane vector.
                b = fx.Float32(ctx.elem_dtype(one).to(fx.Float32))
                lists[step][i] = fadd(lists[step][i], fmul(b, log2e, fm), fm)
        return self.to_vecs(lists)

    def seq_pad_mask(self, v_s, tile_idx, lane):
        """`-inf` past `seqlen_kv`, on the tiles that need it.

        Guarded by a traced `if` rather than applied to every tile: the
        forward measured 27-54% for an unconditional per-step mask, and this
        one is only ever live on the last tile.
        """
        traits = self.traits
        seqlen_kv_i32 = self.ctx.seqlen_kv_i32
        neg_inf = self.ctx.c_neg_inf
        to_lists, to_vecs, kv_col, n = self.to_lists, self.to_vecs, self.kv_col, self.n

        @flyc.jit
        def _mask_if_needed(v_s, tile_idx):
            s_lo, s_hi = v_s
            if fx.Int32((tile_idx + fx.Index(1)) * traits.BLOCK_N) > seqlen_kv_i32:
                lists = to_lists(v_s)
                for step in range_constexpr(2):
                    for i in range_constexpr(n):
                        live = kv_col(tile_idx, lane, step, i) < seqlen_kv_i32
                        lists[step][i] = live.select(fx.Float32(lists[step][i]), neg_inf)
                s_lo, s_hi = to_vecs(lists)
            return s_lo, s_hi

        return _mask_if_needed(v_s, tile_idx)

    def causal_mask(self, v_s, tile_idx, lane, q_row):
        """The causal right bound, and a window's left one, at 16 rows.

        **Derived from this family's own lane map**, not transcribed from the
        32-row one: `kv_col` is the single place the element -> column map
        lives here, and the KV tail mask and the dB store read the same
        function. The 32-row path cannot be reused at all -- its
        `_causal_pair_thresholds` table describes a 16-element accumulator half
        whose columns are four scattered runs, where this one holds four
        contiguous tokens.

        The guard is the forward's two-sided test, over the *wave's* row range
        rather than the workgroup's, because a wave owns 16 rows here:

        - a column can overrun the right bound only if the **lowest** row's
          bound lands inside the tile;
        - a column can fall behind the left bound only if the **highest** row's
          edge is still above `kv_start`.

        Both terms are wave-uniform (`wave_id_uni` is `readfirstlane`d), so the
        branch is scalar and **EXEC is untouched** -- which is what keeps the
        transpose read below it legal under CDNA4 11.4. A divergent condition
        here would be a correctness bug, not a slow one.

        Plain `select`s rather than the 32-row path's paired inline asm: that
        exists because the causal mask is on the innermost path of every
        forward build, and reaching for it here before a measurement asks would
        be copying a decision rather than its reason.
        """
        traits = self.traits
        ctx = self.ctx
        neg_inf = ctx.c_neg_inf
        delta_i32 = ctx.delta_i32
        to_lists, to_vecs, kv_col, n = self.to_lists, self.to_vecs, self.kv_col, self.n
        windowed = const_expr(traits.WINDOW)
        window_left_i32 = ctx.window_left_i32 if windowed else None
        # The wave's first row, uniform. `q_row` is this lane's and is not.
        qs_i32 = fx.Int32(ctx.q_start) + fx.Int32(ctx.wave_id_uni) * fx.Int32(MFMA16_M)
        q_row_i32 = fx.Int32(q_row)

        @flyc.jit
        def _mask_if_needed(v_s, tile_idx):
            s_lo, s_hi = v_s
            kv_end = fx.Int32((tile_idx + fx.Index(1)) * traits.BLOCK_N)
            need = qs_i32 + delta_i32 < kv_end
            if windowed:
                kv_start = fx.Int32(tile_idx * traits.BLOCK_N)
                need = need | (kv_start < qs_i32 + fx.Int32(MFMA16_M - 1) - window_left_i32)
            if need:
                lists = to_lists(v_s)
                for step in range_constexpr(2):
                    for i in range_constexpr(n):
                        col = kv_col(tile_idx, lane, step, i)
                        keep = col <= q_row_i32 + delta_i32
                        if windowed:
                            keep = keep & (col >= q_row_i32 - window_left_i32)
                        lists[step][i] = keep.select(fx.Float32(lists[step][i]), neg_inf)
                s_lo, s_hi = to_vecs(lists)
            return s_lo, s_hi

        return _mask_if_needed(v_s, tile_idx)

    def exp2(self, v_s):
        """The `exp2` batch, held one slot away from its consumers.

        `exp2_wait_state` carries the argument; the 32-row family's `exp2`
        carries the measurement for dQ, which is that no dQ build scanned comes
        within 8 instructions of the hazard. Imported as a property of the
        instruction pair rather than of today's schedule.

        Unlike the 32-row path this one returns the lists directly, so it wraps
        rather than needing the body copied.
        """
        return tuple(
            exp2_wait_state([dualwave.rocdl.exp2(T.f32, as_mlir_value(Vec(h)[r])) for r in range_constexpr(self.n)])
            for h in v_s
        )

    def dropout_dp(self, dp_lists, tile_idx, lane, q_row):
        """`dP <- keep ? dP * (1/(1-p)) : 0`, at 16 rows.

        Same placement and same reasoning as the 32-row family's
        `BwdDqSoftmaxHelper.dropout_dp` -- the mask is on the gradient of the
        dropped output, `P` stays undropped, and `delta` already carries the
        factor through the forward's `O`.

        The span is simpler here and that is the family's own map, not a
        transcription: a lane's four accumulator elements are four
        **contiguous** KV columns starting at `4 * (lane // 16)`, which is a
        multiple of `randoms_per_offset`, so each half is exactly one Philox
        call with no partial draw. The 32-row path needs
        `_score_column_runs` because its four are scattered.
        """
        ctx = self.ctx
        rng = ctx.philox_rng
        scale = fx.Float32(ctx.dropout_scale_arg)
        zero = ctx.c_zero_f
        for step in range_constexpr(2):
            first = rng.grid_offset(
                ctx.philox_plane_base,
                ctx.philox_row_stride,
                fx.Int64(q_row),
                fx.Int64(self.kv_col(tile_idx, lane, step, 0)),
            )
            keep = rng.keep_span(ctx.philox_seed, first, self.n, ctx.idropout_p)
            for i in range_constexpr(self.n):
                kept = fmul(dp_lists[step][i], scale, ctx.fm_fast)
                dp_lists[step][i] = keep[i].select(fx.Float32(kept), zero)
        return dp_lists

    def dscores(self, p_lists, dp, delta):
        """`dS = P * (dP - delta)`, elementwise. `dp` is already a list pair."""
        fm = self.ctx.fm_fast
        return tuple(
            [fmul(p_lists[h][r], fsub(dp[h][r], delta, fm), fm) for r in range_constexpr(self.n)] for h in (0, 1)
        )

    def pack_ds(self, ds_lists):
        """One v8 bf16 B operand out of the two 4-element halves.

        `[lo0..lo3, hi0..hi3]`, in that order, because slot `4r + i` of the
        operand must hold the token the transpose read put at `4g + 16r + i` --
        which is accumulator half `r`, element `i`. The whole permutation
        argument reduces to this concatenation being in the obvious order.

        The pack leaves through `mfma_operand_wait_state` for the same reason
        dK/dV's does; that docstring has the measurement. dQ has not been seen
        to fail this way, but it builds the operand with the same instruction
        and hands it to the same MFMA shape, and the barrier is free.
        """
        vals = list(ds_lists[0]) + list(ds_lists[1])
        return mfma_operand_wait_state(dualwave._bf16_trunc_pack_v8(self.traits, vals, elem_dtype=self.ctx.elem_dtype))


class M16DqStore:
    """dQ, four contiguous columns per lane per chunk.

    **No cross-lane shuffle.** The 32-row family's O store runs the
    accumulator through `permlane32_swap` (`_fused_o_128_dwords`) because a
    lane's 16 values are scattered `[d]` for one `[q]` and a row-major store
    needs them gathered. At 16 rows a lane's four accumulator elements are four
    *contiguous* d columns of one query row, so the store is one 8-byte write.

    Suppression past `hdim_qk` is by address, as everywhere else here: pushing
    the offset past the descriptor's `num_records` makes the hardware drop it.
    A run may write up to three columns into the caller's pad, which the 8xD
    contract guarantees exists -- and is stricter than the 32-row path, whose
    128-bit store can overrun by eight.

    The descriptor is bounded at the last row's last real element rather than
    at `rows * stride`, which is what makes that pad claim true for the *last*
    (batch, head) slab of a BSHD dQ, where the row stride is `num_heads * hdim`
    and the pad it would otherwise write into is off the end of the tensor.
    See `_slab_span_elems`. `oob` stays the untightened span: a sentinel only
    has to sit at or past `num_records`, and shrinking the bound moves it
    further out of range rather than back into it.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.traits = ctx.traits
        self.oob = ctx.dq_oob_off
        # `hdim_qk`, not `hdim_vo`: dQ is Q-shaped, as `store` says again.
        span = _slab_span_elems(ctx.seqlen_q_v, ctx.stride_dq_seq, ctx.hdim_qk)
        self.rsrc = buffer_ops.create_buffer_resource(
            ctx.DQ,
            max_size=False,
            num_records_bytes=as_mlir_value(span * fx.Index(self.traits.BF16_BYTES)),
            base_byte_offset=as_mlir_value(
                ctx._slab_byte_base(
                    ctx.stride_dq_batch, ctx.stride_dq_head, ctx.stride_dq_seq, ctx.q_row_off, ctx.q_head_idx
                )
            ),
        )

    def store(self, v_dq, q_row, lane):
        ctx, traits = self.ctx, self.traits
        n = acc_elems(traits)
        row_base = q_row * ctx.stride_dq_seq_v
        in_row = q_row < ctx.seqlen_q_v
        col_lane = fx.Index(4) * (lane // fx.Index(MFMA16_M))
        for c in range_constexpr(d_chunks16(traits)):
            col = fx.Index(c * MFMA16_M) + col_lane
            vals = [fx.Float32(Vec(v_dq[c])[r]).to(ctx.elem_dtype) for r in range_constexpr(n)]
            pack = Vec.from_elements(vals, ctx.elem_dtype)
            live = in_row
            if const_expr(ctx.PADDED_HEAD):
                # `hdim_qk`, not `hdim_vo`: dQ is Q-shaped. The two extents are
                # used the other way round in this kernel than in the forward.
                live = live & (col < fx.Index(ctx.hdim_qk))
            off = fx.Index(live.select(row_base + col, self.oob))
            buffer_ops.buffer_store(as_mlir_value(pack), self.rsrc, as_mlir_value(fx.Int32(off)))


def make_m16_dq_body(
    ctx, traits, *, kv_gmem_to_lds, db_store, hdim_qk, hdim_vo, hdim_qk_floor, hdim_vo_floor, store_db
):
    """The traced 16-row body. A factory, for `make_wide_body`'s reason.

    The `for ... init=[...]` / `yield` protocol only exists after the AST
    rewrite, so a module-level function called from the kernel is compiled by
    Python as written and `range(..., init=...)` is a `TypeError`. Closing over
    the helpers keeps them out of the traced signature, where only values
    belong.
    """
    lane = ctx.lane
    # 16 rows per wave, so the wave's row block is 16 wide and the lane's row
    # is `lane % 16` -- `_init_dualwave_q_row`'s `lane_mod_32` is the 32-row
    # spelling of this and is not reusable.
    q_row_in_block = ctx.wave_id * fx.Index(MFMA16_M) + lane % fx.Index(MFMA16_M)
    q_row = ctx.q_start + q_row_in_block

    k_reader = M16DqReader(
        ctx, region="v", line_stride=traits.SMEM_V_LINE_STRIDE, hdim=hdim_qk, hdim_floor=hdim_qk_floor
    )
    v_reader = M16DqReader(
        ctx, region="k", line_stride=traits.SMEM_K_LINE_STRIDE, hdim=hdim_vo, hdim_floor=hdim_vo_floor
    )
    q_loader = M16QLoader(ctx, ctx.q_div, ctx.stride_q_seq_v, ctx.q_gmem_elem_offset, hdim_qk)
    do_loader = M16QLoader(ctx, ctx.do_div, ctx.stride_do_seq_v, ctx.do_gmem_elem_offset, hdim_vo)
    gemm = M16DqGemmHelper(ctx)
    softmax = M16DqSoftmax(ctx)
    store = M16DqStore(ctx)
    n_chunks = d_chunks16(traits)

    @flyc.jit
    def _m16_body():
        """One KV tile of 32 per iteration; three GEMMs and one accumulator."""
        q_packs = q_loader.load_all(q_row_in_block, lane)
        do_packs = do_loader.load_all(q_row_in_block, lane)
        lse2, delta = ctx.load_row_scalars(q_row)

        init_args = [gemm.zero for _ in range_constexpr(n_chunks)]
        loop_results = init_args
        for j, loop_args in range(ctx.split_tile(0), ctx.split_t_end, fx.Index(1), init=init_args):
            v_dq = [loop_args[i] for i in range_constexpr(n_chunks)]
            tile_start = ctx.tile_start(j)

            _s_barrier()
            kv_gmem_to_lds.stage_k(tile_start)
            kv_gmem_to_lds.stage_v(tile_start)
            _s_waitcnt(0)
            _sched_barrier(0)
            _s_barrier()

            v_s = gemm.qk(k_reader.load_a_all(lane), q_packs)
            v_s = softmax.scale_and_sub_lse(v_s, ctx.c_sm_scale_log2e, lse2)
            if const_expr(traits.BIAS_TYPE):
                v_s = softmax.add_bias(v_s, j, lane, q_row)
            v_s = softmax.seq_pad_mask(v_s, j, lane)
            if const_expr(traits.CAUSAL):
                # Same placement argument as the 32-row body's: this touches
                # `v_s` only, and every transpose read is below it. The guard
                # is wave-uniform, so the branch is scalar and EXEC stays all
                # ones across `load_kt`.
                v_s = softmax.causal_mask(v_s, j, lane, q_row)
            v_dp = gemm.qk(v_reader.load_a_all(lane), do_packs)

            dp_lists = softmax.to_lists(v_dp)
            if const_expr(traits.ENABLE_DROPOUT):
                dp_lists = softmax.dropout_dp(dp_lists, j, lane, q_row)
            ds = softmax.dscores(softmax.exp2(v_s), dp_lists, delta)
            ds_pack = softmax.pack_ds(ds)
            if const_expr(store_db):
                db_store.store_tile_m16(ds_pack, j, q_row, lane)
            v_dq = gemm.pv(ds_pack, k_reader.load_kt(lane), v_dq)

            loop_results = yield v_dq

        v_dq = [loop_results[i] for i in range_constexpr(n_chunks)]
        scale = Vec.from_elements([ctx.c_sm_scale], fx.Float32).broadcast_to(acc_elems(traits))
        for c in range_constexpr(n_chunks):
            v_dq[c] = fmul(Vec(v_dq[c]), scale, ctx.fm_fast)
        _s_barrier()
        store.store(v_dq, q_row, lane)

    return _m16_body


# ============================================================================
# The dQ kernel
# ============================================================================


def _carried(values, count):
    """An `scf.for`'s carried values as a list of `count`, however many it hands back.

    **A loop with exactly one carried value hands it back unwrapped**, because
    an `scf.for` with one result is a value rather than a tuple. This kernel
    carries `D_CHUNKS` accumulators and nothing else, so head_dim 32 -- the one
    rung where `D_CHUNKS == 1` -- is the only place that bites, and it bites
    two frames away: `loop_results[0]` on a `vector<16xf32>` returns the *first
    f32 element*, and the failure surfaces inside `_scale_o_accs` as "Cannot
    cast type to VectorType". The forward never sees this because it always
    carries `m_row` and `l_row` alongside.
    """
    if const_expr(isinstance(values, (list, tuple))):
        return list(values)
    if const_expr(count != 1):
        raise AssertionError(f"expected {count} carried values, got one unwrapped value")
    return [values]


class BwdDqKernelContext(ParityKernelContext):
    """The forward's parity context plus dQ, dO, dB and the two row inputs.

    Subclassed rather than ported, per the contract: the strides, the padded
    head, the varlen decode and the descriptor machinery are all inherited.
    What is added is this kernel's output (`dQ`), one more Q-shaped input
    (`dO`), one score-shaped output (`dB`), and the LSE/delta pair.

    **There is no `O` here.** dQ used to be handed to the base class's `O`
    slot, which made `stride_o_*` dQ's strides, `o_div` dQ's descriptor and
    `hdim_vo` -- the head dim of V or O -- the bound on a tensor that is
    Q-shaped and therefore `hdim_qk` wide. At an asymmetric head dim that
    bound is simply the wrong number, and the D-tail stores it was supposed to
    admit were dropped instead. dQ is spelled `DQ` here, with `stride_dq_*`,
    `dq_div` and `dq_oob_off`; the base's `O` stays None and its O view is not
    built. `BwdDqStoreHelper` is the one place upstream's `o_*` store
    vocabulary is bound to them.
    """

    def __init__(
        self,
        traits,
        *,
        dq_strides,
        do_strides,
        db_strides=(0, 0, 0),
        DQ=None,
        DO=None,
        DB=None,
        Delta=None,
        **kwargs,
    ):
        super().__init__(traits, **kwargs)
        self.stride_dq_batch, self.stride_dq_head, self.stride_dq_seq = dq_strides
        self.stride_do_batch, self.stride_do_head, self.stride_do_seq = do_strides
        # `_seq_q` for the reason `ParityKernelContext` gives for the bias
        # input's: dB is `(batch, head, seqlen_q, seqlen_k)` and a bare `_seq`
        # does not say which of the two it is.
        self.stride_db_batch, self.stride_db_head, self.stride_db_seq_q = db_strides
        self.DQ = DQ
        self.DO = DO
        self.DB = DB
        self.Delta = Delta

    def init_runtime_indices(self, **kwargs):
        super().init_runtime_indices(**kwargs)
        self.stride_dq_seq_v = fx.Index(self.stride_dq_seq)
        self.stride_do_seq_v = fx.Index(self.stride_do_seq)

    def init_descriptors(self, **kwargs):
        """The forward's four views, plus dQ's, dO's and dB's.

        `dO` gets exactly Q's treatment -- same slab shape, same row origin,
        same `seqlen_q` bound -- because it *is* Q-shaped: (batch, head, q row,
        d), with the vo head dim rather than the qk one. The bound is what
        makes a row past `seqlen_q` read zero instead of faulting, which is the
        whole reason the ragged tail needs no branch.

        Each of the three carries the width of *its own* rows, which is the
        point of `_slab_span_elems`: `stride_seq` is the distance between rows
        and is only the width of one when the tensor is contiguous in
        `(.., seqlen, hdim)` order. dQ and dO differ in that width -- dQ is
        Q-shaped and `hdim_qk` wide, dO is O-shaped and `hdim_vo` wide -- and
        an asymmetric build is where the difference shows.
        """
        traits = self.traits
        super().init_descriptors(**kwargs)
        # dQ is Q-shaped -- (batch, head, q row, d) with the *qk* head dim --
        # so it gets Q's slab treatment against `hdim_qk`. Not `hdim_vo`: at
        # an asymmetric head dim the two are different numbers and only one of
        # them is dQ's row width.
        self.dq_div, self.dq_oob_off = self.q_row_slab(
            self.DQ, self.stride_dq_batch, self.stride_dq_head, self.stride_dq_seq, self.hdim_qk
        )
        self.do_div = self._slab_view(
            self.DO,
            self.stride_do_batch,
            self.stride_do_head,
            self.stride_do_seq,
            self.q_row_off,
            self.q_head_idx,
            self.seqlen_q_v,
            self.hdim_vo,
        )
        self.do_gmem_elem_offset = self.q_start * self.stride_do_seq_v
        if const_expr(traits.STORE_DB):
            # **Whether dB is wanted is a runtime question, not a build one.**
            # `STORE_DB` above only decides whether the store *exists*; a build
            # that has it still has to serve a caller who passes a bias but does
            # not want its gradient, which AOTriton spells as an all-zero dB
            # stride triple over a null DB (`bwd_kernel_dq.py`'s `store_db`).
            # PyTorch takes that path on every bool-masked SDPA: the mask becomes
            # an additive bias that is not a leaf, so attention_backward.cu hands
            # over `empty_t4` for DB.
            #
            # Not currently a fault here, unlike gfx1201's flat store: the buffer
            # bound below collapses to 0 records when `stride_db_seq_q` is 0, so
            # the hardware drops every store and the null base is never touched.
            # That is an accident of the `num_records` arithmetic rather than
            # anything this kernel says, and it does not cover a null DB carried
            # with a non-zero row stride. Say it instead.
            self.db_store_enabled = (
                (fx.Int64(self.stride_db_batch) != fx.Int64(0))
                | (fx.Int64(self.stride_db_head) != fx.Int64(0))
                | (fx.Int64(self.stride_db_seq_q) != fx.Int64(0))
            )
            # Same slab shape as the forward's bias descriptor -- dB is indexed
            # (batch, head, q row, kv col) and the KV axis is contractually
            # contiguous -- and a raw resource for the same reason: the stores
            # are per-lane at an address the lane computes, which is
            # `buffer_ops.buffer_store`'s shape and not the copy atom's.
            # First element past the *untightened* span. A store redirected
            # here is dropped by the hardware bound; see `BwdDbStoreHelper`.
            # It stays untightened deliberately: a sentinel only has to sit at
            # or past `num_records`, and the bound below only shrinks, which
            # moves it further out of range rather than back into it.
            self.db_oob_off = self.seqlen_q_v * fx.Index(self.stride_db_seq_q)
            # Bounded through the forward's own helper, and for its reason: a
            # BSHD caller's `stride_db_seq_q` is `num_heads * seqlen_k`, so
            # `seqlen_q * stride_db_seq_q` overshoots this head's slab by the
            # other heads of the last row -- which for the last (batch, head)
            # is off the end of dB. Storing there would corrupt a neighbouring
            # head's gradient, or fault.
            #
            # The load-side caveat in that helper's docstring -- a wide read
            # losing the last column to the per-dword range check at an odd
            # `seqlen_k` -- has no store-side counterpart here: this is a
            # *2-byte* store per element, already predicated on
            # `col < seqlen_k`, so every live one lands inside the tight bound
            # whatever the parity.
            self.db_rsrc = buffer_ops.create_buffer_resource(
                self.DB,
                max_size=False,
                num_records_bytes=_bias_slab_num_records_bytes(
                    self.seqlen_q_v, self.seqlen_kv_v, self.stride_db_seq_q, traits.BF16_BYTES
                ),
                base_byte_offset=as_mlir_value(
                    self._slab_byte_base(
                        self.stride_db_batch,
                        self.stride_db_head,
                        self.stride_db_seq_q,
                        self.q_row_off,
                        self.q_head_idx,
                    )
                ),
            )

    def init_philox(self):
        """The forward's philox prologue, minus the report.

        **The backward *consumes* the reproducibility contract; it does not
        publish one.** `ParityKernelContext.init_philox` ends by calling
        `common.philox_report`, which writes the seed and offset back for a
        later backward to pick up -- correct there, meaningless here, and it
        would need two output pointers on the wire that nothing reads.

        Everything else is inherited verbatim, and that is the whole point:
        the mask has to be regenerated *bit-identically*, and it is because
        both kernels call the same `Philox.grid_plane` with the same
        `(seed, offset)` and the same plane -- not because two transcriptions
        of one formula happened to agree.

        The consequence worth naming: `grid_plane` is given `max_seqlen_q` and
        `max_seqlen_k`, never `BLOCK_M`/`BLOCK_N`, so the mask is a function of
        element coordinates only and **the tile geometry may differ between
        the two kernels**. What may *not* differ is the pair of max lengths or
        the plane index, and the plane index is `(sequence, q head)` -- the raw
        grid `z`, kept as `seq_idx_i32`, and **not** `batch_idx`, which the
        varlen decode resolves to 0 for every sequence of a stacked layout. The
        forward reads the same field for the same reason.
        """
        if const_expr(not self.traits.ENABLE_DROPOUT):
            return
        self.philox_rng = Philox.for_arch("gfx950")
        plane = self.seq_idx_i32 * fx.Int32(self.num_head_q) + fx.Int32(self.q_head_idx)
        seed = common.philox_seed_value(self.philox_seed_ptr)
        offset = common.philox_offset_base(self.philox_offset1, self.philox_offset2)
        self.philox_seed = seed
        self.philox_plane_base, self.philox_row_stride = self.philox_rng.grid_plane(
            offset, plane, self.seq_len_v, self.seq_len_kv_v
        )

    def compute_active_guard(self):
        """`q_start < seqlen_q`, and **not** the cross-sequence term.

        The base adds `causal_end_raw_i32 > 0` under `CAUSAL and CROSS_SEQLEN`,
        which skips a Q block whose causal region is empty -- correct for the
        forward, which then zeroes `O` for those rows through
        `zero_o_block_if_needed`. Inheriting it here would leave **dQ
        unwritten** for the same rows, which is a garbage read for the caller
        rather than a zero.

        Dropping the term is not a workaround: this loop derives its tile count
        from the same `causal_end_raw_i32`, so an empty causal region walks
        **zero tiles**, leaves the accumulator at its zero seed and stores
        zeros. The block does the right thing by running rather than by being
        skipped, and no second zeroing path is needed in either family.

        The `q_start` term is kept, and it is the one that saves work: the grid
        is sized from `max_seqlen_q`, so a short sequence dispatches Q blocks
        with no real rows at all. A dense call has `seqlen_q_v == max_seqlen_q`
        and the predicate is true for every block the grid dispatches, which is
        why it needs no build-time case -- it is a scalar compare that a dense
        launch simply never fails.
        """
        if const_expr(self.traits.SPLITK):
            return self.split_nonempty
        return self.q_start < self.seqlen_q_v

    def init_tile_bounds(self, **kwargs):
        """The inherited bounds, re-tightened for a **one-tile** loop.

        `ParityKernelContext` resolves the window and re-points `delta_i32`,
        and the base class then derives `max_num_tiles` from it -- all of which
        this body wants. What it does not want is the two lines after: the base
        rounds the count up to even and floors it at 4, because the dual-wave
        pipeline consumes two tiles per iteration and its prologue plus
        epilogue need four to exist. This loop consumes one and needs none.

        Left alone it is still *correct* -- the extra tiles are fully masked and
        contribute nothing -- which is exactly the problem the contract warns
        about: **a dead tile is a no-op, so correctness cannot show the cut
        works.** Up to three dead tiles per Q block is also enough to hide the
        cut from a timing test, which is the only test that can see it.
        """
        super().init_tile_bounds(**kwargs)
        traits = self.traits
        if const_expr(traits.CAUSAL):
            # `causal_end_raw_i32` is `q_start + BLOCK_M + delta_i32`, and
            # `delta_i32` is the resolved *right* bound under a window -- which
            # is why re-pointing it generalises the tile count as well as the
            # mask.
            end_i32 = fx.Int32((self.causal_end_raw_i32 > fx.Int32(0)).select(self.causal_end_raw_i32, fx.Int32(0)))
            n = (fx.Index(end_i32) + fx.Index(traits.BLOCK_N - 1)) // fx.Index(traits.BLOCK_N)
            n = fx.Index((n < self.num_kv_tiles).select(n, self.num_kv_tiles))
        else:
            n = self.num_kv_tiles
        self.max_num_tiles = n
        # **Not a literal 0.** P3 found four literal tile bases in the forward,
        # and a window moves this one: `_skip_dead_leading_tiles` below is the
        # only thing that writes it, and spelling the non-window arm as 0 here
        # would be correct and would throw the left-bound saving away.
        self.split_t0 = fx.Index(0)
        self.split_t_end = n
        if const_expr(traits.WINDOW):
            self._skip_dead_leading_tiles()

    def _skip_dead_leading_tiles(self):
        """Start the KV walk at the window's left edge. Exact, for a one-tile loop.

        The inherited version rounds the base down to even and caps it at
        `split_t_end - 4`, both because the dual-wave pipeline needs an even
        segment with four tiles in it. Neither applies here, and both would
        blunt the cut -- the cap in particular pins the base to zero for any Q
        block whose live range is short, which is the case a left bound is for.

        The lowest column any row of this Q block can reach is
        `q_start - window_left`, since `q_start` is the smallest row. Clamped
        before the divide rather than after: `fx.Index` is unsigned, so a
        negative would come out enormous and skip the whole range.
        """
        traits = self.traits
        first_col_i32 = fx.Int32(self.q_start) - self.window_left_i32
        first_col_i32 = fx.Int32((first_col_i32 > fx.Int32(0)).select(first_col_i32, fx.Int32(0)))
        t0 = fx.Index(first_col_i32) // fx.Index(traits.BLOCK_N)
        self.split_t0 = fx.Index((t0 < self.split_t_end).select(t0, self.split_t_end))

    def init_row_inputs(self):
        """Descriptors and the shared row addressing for LSE and delta.

        **One addressing for both**, which is `common.lse_row_addressing`'s
        stated contract: delta is produced beside LSE by the same caller and
        giving it its own decode would double the work for no expressiveness.
        Called with batch 0 because the descriptors below already fold the
        batch in, exactly as the forward's varlen LSE store does.
        """
        tokens = fx.Index(self.lse_tokens_i32)
        per_batch = fx.Index(self.num_head_q) * tokens
        per_batch_bytes = per_batch * fx.Index(4)
        self.row_base, self.row_pitch = common.lse_row_addressing(
            self.varlen_bits_arg,
            fx.Index(0),
            self.q_head_idx,
            self.num_head_q,
            tokens,
            self.q_row_off,
        )
        # The sentinel a row past `seqlen_q` is redirected to. Without it the
        # offset would run into the *next head's* rows, which is in bounds and
        # returns a plausible LSE for the wrong row -- finite, and wrong in a
        # way no shape check sees. The row is discarded at the store either
        # way, but a zero keeps the arithmetic in between boring.
        self.row_oob_off = per_batch
        self.lse_rsrc = dualwave._make_ws_rsrc(
            fx.Int64(fx.ptrtoint(fx.get_iter(self.LSE))), self.batch_idx * per_batch_bytes, per_batch_bytes
        )
        self.delta_rsrc = dualwave._make_ws_rsrc(
            fx.Int64(fx.ptrtoint(fx.get_iter(self.Delta))), self.batch_idx * per_batch_bytes, per_batch_bytes
        )

    def load_row_scalars(self, q_row):
        """`(lse2, delta)` for one query row. Per-lane scalars in both families.

        The query row is the *n* axis of GEMM1 and GEMM3, so a lane owns
        exactly one of them whatever the MFMA's row count -- which is why this
        is shape-independent and the dK/dV kernel's counterpart is not (there
        the query row is *m*, and a lane holds four).
        """
        off = self.row_base + q_row * self.row_pitch
        off = fx.Index((q_row < self.seqlen_q_v).select(off, self.row_oob_off))
        off_i32 = as_mlir_value(fx.Int32(off))
        lse = buffer_ops.buffer_load(self.lse_rsrc, off_i32, vec_width=1, dtype=fx.Float32)
        delta = buffer_ops.buffer_load(self.delta_rsrc, off_i32, vec_width=1, dtype=fx.Float32)
        # The forward writes LSE in natural units (`m_row*ln2 + ln(l)`), and
        # every exponent downstream is `exp2`, so the conversion happens once
        # here rather than per element. AOTriton's `l_i = tl.load(...) *
        # RCP_LN2` is the same fold.
        lse2 = fmul(fx.Float32(lse), fx.Float32(dualwave._LOG2E), self.fm_fast)
        # **Floored, because `LSE` can legitimately be `-inf`** and this is an
        # *input*. A causal row with no live key -- every row below
        # `seqlen_q - seqlen_kv` under bottom-right alignment, which is every
        # `Sq > Sk` varlen sequence -- makes the forward's `l_row` zero and its
        # `m_row*ln2 + log(l_row)` therefore `-inf`. Our own forward writes it.
        #
        # **This is not fixing an observed NaN, and saying so matters.**
        # Measured with the floor removed, the varlen causal oracle still
        # passes bitwise: `scale_and_sub_lse` computes `fma(S, qk_scale,
        # +inf)`, and the KV/causal mask that runs *after* it overwrites every
        # column of such a row with `-inf` before `exp2` sees it. The rescue is
        # real and it is why the mask is ordered after the scale.
        #
        # The floor is kept because that rescue leans on an infinity flowing
        # through an FMA carrying `fastmath<fast>`, whose `ninf` is a licence
        # to assume infinities are absent. P5 recorded exactly that licence
        # being taken up later by a different pass and silently deleting a KV
        # tail mask on gfx1201. One `max` per kernel removes the case.
        #
        # After the `log2e` conversion, not before: `-3e38 * log2e` overflows
        # back to `-inf`. Same device `ParitySoftmaxHelper` uses to keep the
        # forward's running max off `-inf`, applied to an input.
        #
        # **Re-derived for f16 rather than assumed to carry.** The floor is
        # `-3.0e38`, which is an f32 magnitude and would be `-inf` in f16 --
        # but nothing here is f16. `lse` arrives f32, `log2e` is f32, the `max`
        # is f32, and the FMA it protects in `scale_and_sub_lse` is f32. The
        # operand dtype reaches this kernel at exactly two places, the `dS`
        # pack and the dQ store, and neither is upstream of this. So the
        # argument is about `fastmath<fast>`'s `ninf` on an f32 FMA and is
        # dtype-independent.
        lse2 = fmax(fx.Float32(lse2), self.c_neg_floor, self.fm_fast)
        return lse2, fx.Float32(delta)


class BwdRowInputLoader(ParityStoreHelper):
    """`lse2` and `delta` for this lane's query row, once per kernel.

    A thin wrapper on `BwdDqKernelContext.load_row_scalars`, kept because the
    32-row body reaches it through a helper object like everything else. The
    work is on the context because the 16-row body needs the same two values
    at a different row map (`lane % 16`), and one copy of the addressing is
    the point.
    """

    def load(self, q_row):
        return self.ctx_ref.load_row_scalars(q_row)


class BwdDqSoftmaxHelper(ParitySoftmaxHelper):
    """The forward's softmax, minus the one place it trades accuracy for speed."""

    def scale_and_sub_lse(self, v_s, qk_scale, lse2):
        """`qk_scale * S - lse2`, one FMA per score element.

        **This is where the backward stopped matching the forward, and it was
        worth 10x at a large `sm_scale`.** The forward folded
        `sm_scale * log2e` into Q and rounded the product back to bf16, which
        saves a multiply per score; the error that introduces is `|S| * 2^-8`
        in the *exponent*, so `P = exp2(S - lse2)` inherits it as a relative
        error. The forward was thought to tolerate that because `O` is a
        normalised average and the error largely cancels, while
        `dS = P * (dP - delta)` does not normalise and `dQ` sums it over the
        whole key axis. **The forward did not tolerate it either** -- the
        measurement below transfers, and `ParityQLoader.scale_all` records what
        it cost there -- so the fold is gone from both and the table is now the
        history of a fixed bug rather than a live divergence.

        Measured at `B=1 H=4 S=512 d=64`, max error against an fp64 reference:

            sm_scale   Q pre-scaled (the old fold)     scaled here
              0.05            1.8e-4                     1.6e-4
              0.25            4.3e-2                     2.0e-2
              1.00            6.9e-1                     6.8e-2

        A host model of both variants reproduces the kernel's own numbers to
        two digits, which is what identifies the Q rounding rather than
        anything else as the cause. AOTriton scales after the dot in *both*
        directions (`qk += Qk_scale * tl.dot(q, k)`), so this is its
        arithmetic, not a new choice.

        Not `dualwave._scale_sub_score_pair`: that one derives the offset from
        an *unscaled* row max (`fma(s, scale, -scale*m)`), and `lse2` is
        already in the scaled base-2 domain. Passing it there would need a
        divide by `sm_scale` to undo a multiply.

        The KV tail mask runs **after** this, not before, so no infinity ever
        reaches the FMA -- `qk_scale * -inf` is correct but puts a real
        infinity into arithmetic under `fastmath<fast>`, which plan section 5
        records as the thing that silently deleted a mask on gfx1201.
        """
        s_lo, s_hi = v_s
        acc = self.traits.ACC_ELEMS
        scale_v = Vec.from_elements([fx.Float32(qk_scale)], fx.Float32).broadcast_to(acc)
        neg_lse2 = fsub(self.c_zero_f, lse2, self.fm_fast)
        neg_v = Vec.from_elements([fx.Float32(neg_lse2)], fx.Float32).broadcast_to(acc)
        lo = fmath.fma(Vec(s_lo), scale_v, neg_v, fastmath=self.fm_fast)
        hi = fmath.fma(Vec(s_hi), scale_v, neg_v, fastmath=self.fm_fast)
        return as_mlir_value(lo), as_mlir_value(hi)

    def cast_p(self, v_p, tile_idx=None):
        """Pack to bf16 with **no dropout**, unlike the forward's `cast_p`.

        `ParitySoftmaxHelper.cast_p` is where the *forward* applies its mask --
        after the row sum, before the O accumulation. Inheriting it here would
        drop `dS` on the way into GEMM3 while `dropout_dp` has already dropped
        `dP`, i.e. **apply the mask twice**: once correctly and once to a
        quantity that must not carry it, since `P` is the undropped softmax
        the undropped `lse` defines.

        It fails loudly today only by accident -- the forward's block reads
        `tile_idx`, which this call site does not pass. That is a bad reason to
        be safe, so the override is explicit.

        **Every pack leaves through `mfma_operand_wait_state`, and here that is
        load-bearing rather than prophylactic.** `dS` is GEMM3's B operand, and
        at `BLOCK_DMODEL=64, PADDED_HEAD=False, bf16` the scheduler put all
        four packs in one register quad and issued the *first* `D_CHUNKS`
        accumulator's MFMA in the slot right after the last
        `v_cvt_pk_bf16_f32` of each pack -- one intervening vector-pipe
        instruction at the best site and zero at the worst:

            v_cvt_pk_bf16_f32 v33, v42, v43
            s_waitcnt lgkmcnt(2)                        <- scalar, no wait state
            v_mfma_f32_32x32x16_bf16 v[0:15], v[52:55], v[32:35], v[0:15]

        The second chunk's MFMA reads the same quad one slot later and is
        shielded by the first one, so the damage is confined to head-dim
        columns `[0, 32)`: measured 39% relative L2 on dQ against 7% on
        `[32, 64)`, deterministic, dQ only, and invisible in an f16 build,
        whose `v_cvt_f16_f32` packs the scheduler happened to hoist 8-36
        instructions clear. That is the `v_cvt_pk_bf16_f32` -> MFMA hazard with
        a wrong answer attached rather than a suspicion, and
        `fmha_bwd_dq_m16_gfx950.pack_ds` already carried the barrier on the
        strength of the suspicion alone.
        """
        p_lo_packs, p_hi_packs = dualwave.DualwaveSoftmaxHelper.cast_p(self, v_p)
        return (
            [mfma_operand_wait_state(p) for p in p_lo_packs],
            [mfma_operand_wait_state(p) for p in p_hi_packs],
        )

    def exp2(self, v_s, start, length):
        """`_exp2_score_slice` with the gfx950 `v_exp_f32` wait state.

        The body is `dualwave._exp2_score_slice` verbatim apart from the one
        call, and copied rather than wrapped for the reason the forward's
        equivalent gives: on the `start == 0` path the results are consumed
        *inside* it, by `Vec.from_elements`, so there is no return value a
        wrapper could interpose on.

        See `exp2_wait_state` for why a bare `_s_nop` is not this.

        **Not reproduced in dQ, and imported on the argument.** The same scan
        measured the gap from each `v_exp_f32` to its first VALU consumer over
        ten dQ builds: the minimum anywhere was 8, with per-build minima of 8,
        9, 26, 50 and 54, against the *exactly 1* that AOTriton's dK/dV scan
        reports and that phase 1 confirmed here. dQ is not near this cliff, and
        saying otherwise would overstate it.

        Imported anyway, because "not near the cliff" is a fact about one
        schedule and not about the kernel. AOTriton's live case became live
        exactly that way -- issue 9's fix raised register pressure, the
        schedule moved, and a latent gap became a zero-gap wrong answer. The
        cost here is measured rather than assumed; see the plan's outcome
        section for the register and ISA comparison.
        """
        if const_expr(start == 0):
            s_lo = [Vec(v_s[0])[r] for r in range_constexpr(16)]
            lo_partial = exp2_wait_state(
                [dualwave.rocdl.exp2(T.f32, as_mlir_value(s_lo[r])) for r in range_constexpr(16)]
            )
            return Vec.from_elements(lo_partial, fx.Float32).ir_value(), v_s[1]
        lo_partial = [Vec(v_s[0])[r] for r in range_constexpr(16)]
        hi_full = exp2_wait_state(
            [dualwave.rocdl.exp2(T.f32, as_mlir_value(Vec(v_s[1])[r])) for r in range_constexpr(16)]
        )
        return lo_partial, hi_full

    def dropout_dp(self, dp_lists, tile_idx, q_row):
        """`dP <- keep ? dP * (1/(1-p)) : 0`, on the **dP** rather than on P.

        AOTriton's `bwd_inner_dq` applies the mask here and this follows it,
        because that is what the chain rule says: the forward's output is
        `sum_j (P_j * keep_j * scale) V_j`, so the gradient with respect to the
        *undropped* `P_j` carries the same `keep_j * scale` factor. `P` itself
        stays undropped -- it is `exp2(qk_scale*S - lse2)` and `lse2` is the
        undropped logsumexp the forward wrote.

        **`delta` needs no adjustment**, and that is not luck:
        `delta = rowsum(dO * O)` with `O` the *dropped* output is exactly
        `sum_j P_j keep_j scale (dO . V_j)`, which is the reduction term the
        softmax backward wants against a masked `dP`. The host already computes
        it from the forward's own `O`.

        **The survivor scale cannot be folded the way the forward folds it.**
        There it becomes one multiply on `1/l` per output row, because a row
        sum exists to fold into; here the row sum is an *input*. Folding it
        into `dO` before GEMM2 would be one multiply instead of one per
        element -- and would round `dO` through bf16 a second time, which is
        the mistake B2 measured at 10.9x on the Q pre-scale. It stays an f32
        multiply on the scores.

        The column runs are the bias and forward-dropout ones: a lane's 16
        scores are a few contiguous spans starting at multiples of
        `randoms_per_offset`, so each is a whole number of Philox calls with no
        partial draw.
        """
        traits = self.traits
        ctx = self.ctx_ref
        rng = ctx.philox_rng
        lo, hi = dp_lists
        lane_n_off = 8 if traits.KV_VECTORIZED else 4
        col_base = fx.Int64(tile_idx * traits.BLOCK_N) + fx.Int64(self.lane_div_32 * fx.Index(lane_n_off))
        scale = fx.Float32(ctx.dropout_scale_arg)
        zero = fx.Float32(0.0)
        for half, values in ((0, lo), (1, hi)):
            for elem0, col_off, width in _score_column_runs(traits.KV_VECTORIZED):
                first = rng.grid_offset(
                    ctx.philox_plane_base,
                    ctx.philox_row_stride,
                    fx.Int64(q_row),
                    col_base + fx.Int64(col_off + half * 32),
                )
                keep = rng.keep_span(ctx.philox_seed, first, width, ctx.idropout_p)
                for j in range_constexpr(width):
                    kept = fmul(values[elem0 + j], scale, self.fm_fast)
                    values[elem0 + j] = keep[j].select(fx.Float32(kept), zero)
        return (lo, hi)


class BwdSecondaryQLoader(ParityQLoader):
    """A second Q-shaped loader, for dO.

    `ParityQLoader` reads three attributes to find its tensor -- the buffer
    view, the sequence stride and the tile's element origin -- and everything
    else about it is the lane map, which dO shares with Q exactly. So this
    rebinds those three and inherits the loop, rather than being a second copy
    of an addressing scheme.

    `hdim_qk` is rebound too: the padded-head mask inside `load_pack` is
    against the *reduction* extent for Q, and dO's is `hdim_vo` (it contracts
    with V's D axis, not K's). Inert while `PADDED_HEAD` is off, which is B2 --
    set correctly now so B3 does not have to find it.
    """

    def __init__(self, ctx, div, stride_seq_v, gmem_elem_offset, hdim):
        super().__init__(ctx)
        self.q_div = div
        self.stride_q_n_v = stride_seq_v
        self.q_gmem_elem_offset = gmem_elem_offset
        self.hdim_qk = hdim


class BwdDqKvGmemToLdsLoader(ParityKvGmemToLdsLoader):
    """One staging routine, two (tensor, region) pairs.

    The inherited `load_k` / `load_v` pair hardcodes which tensor goes to which
    region, and this kernel puts **K in the V region**. `_stage` is the
    inherited body with the tensor, the stride and the `m0` table all handed
    in, so the two calls below differ only in their arguments.

    The `m0` table *is* the pitch: `k_dma_m0` addresses lines at
    `SMEM_K_LINE_STRIDE` and `v_dma_m0` at `SMEM_V_LINE_STRIDE`, and each has to
    match the reader that later walks the region. Pairing them wrongly reads a
    tile written with a different pitch -- plausible numbers, no fault.

    Setting `stride_kv_n_v` before delegating is the inherited trick and is
    safe for the inherited reason: tracing is eager, so the attribute is read
    while the call below runs and no branch is open across the swap.
    """

    def _stage(self, tile_start, src_div, stride_seq_v, dma_m0, buf_id):
        self.stride_kv_n_v = stride_seq_v
        self.dma_stage = 0
        self._async_load_kv_linear(
            dma_m0,
            buf_id,
            src_div,
            self.kv_gmem_elem_offset,
            tile_start * stride_seq_v,
            self.NUM_DMA_K,
        )

    def stage_k(self, tile_start):
        """K into the V region: GEMM1's and GEMM3's A operand, one copy."""
        self._stage(tile_start, self.k_div, self.stride_k_seq_v, self.ctx_ref.v_dma_m0, 0)

    def stage_v(self, tile_start):
        """V into the K region: GEMM2's A operand."""
        self._stage(tile_start, self.v_div, self.stride_v_seq_v, self.ctx_ref.k_dma_m0, 0)


class BwdDqKvLdsToVgprLoader(ParityKvLdsToVgprLoader):
    """The forward's two readers, with the K path re-pointable at the V pitch.

    One instance per tile, because the *padded-head extent* differs between
    them as much as the pitch does: the K tile's D axis is `hdim_qk` and the V
    tile's is `hdim_vo`. B2 had one instance and would have masked V's columns
    against Q's extent the moment a padded head arrived -- one of the two
    crossed sites the plan's B2 outcome names.

    `v_layout` is a Python bool fixed at construction, so every `const_expr`
    below folds and neither instance emits the other's code.
    """

    def __init__(self, ctx, *, v_layout, hdim, hdim_floor):
        super().__init__(ctx)
        self.v_layout = bool(v_layout)
        # `load_k`'s mask reads these two by these names. Rebinding them is
        # what makes one class serve both extents.
        self.hdim_qk = hdim
        self.HDIM_QK_FLOOR = int(hdim_floor)

    def _lds_pack(self, elem_idx, scope_name):
        """One 8xbf16 MFMA operand pack out of LDS, under a chosen alias scope.

        `dualwave._load_k_pack_aligned` is the same four lines, except that it
        derives the scope as `lds_k{buf_id}` -- and this kernel's K tile lives
        in the `v0` region. A scope that names the wrong region is not a
        cosmetic problem: `ROCDL_LDS_Read_Tr_IntrOp` and this load are told
        they cannot alias, and here they read the same bytes.
        """
        traits = self.traits
        ptr = buffer_ops.get_element_ptr(self.lds_kv_base_ptr, byte_offset=elem_idx * traits.BF16_BYTES, elem_type=T.i8)
        return llvm.LoadOp(
            self.kv_mfma_pack_type,
            ptr,
            alignment=16,
            alias_scopes=dualwave._dualwave_lds_alias_scopes(scope_name),
            noalias_scopes=dualwave._dualwave_lds_noalias_scopes(scope_name, traits.LDS_SCOPE_NAMES),
        ).result

    def _read_k_packs(self, buf_id, urk_base):
        """The K path, over whichever region this instance owns.

        The `v_layout` arm re-points three things and no more: the line pitch,
        the k-step outer stride (which is `SMEM_N_RPT` lines) and the region
        base. Everything else -- `_parity_k_read_base`, `_parity_ks_offset`,
        the `N_STRIP` lo/hi split -- is the shared formula, evaluated against a
        `replace`d traits object rather than transcribed. The two pitches are
        the *only* difference between the layouts, which is what makes the
        substitution total.

        `urk_base` is ignored on this arm: `load_k` computes it from
        `k_lds_read_base_per_lane`, which the context derived at the K pitch.
        """
        if const_expr(not self.v_layout):
            return super()._read_k_packs(buf_id, urk_base)
        full = self.traits
        self.traits = replace(
            full,
            SMEM_K_LINE_STRIDE=full.SMEM_V_LINE_STRIDE,
            K_LDS_TO_REG_KSTEP_OUTER_STRIDE=full.SMEM_N_RPT * full.SMEM_V_LINE_STRIDE,
        )
        try:
            traits = self.traits
            scope = dualwave._dualwave_lds_scope("v", buf_id)
            base = dualwave._v_buf_base(traits, buf_id) + _parity_k_read_base(
                traits, self.lane_mod_32, self.lane_div_32
            )
            k_lo = [None] * traits.K_STEPS_QK
            k_hi = [None] * traits.K_STEPS_QK
            for ks in range_constexpr(traits.K_STEPS_QK):
                idx = base + _parity_ks_offset(traits, ks)
                k_lo[ks] = self._lds_pack(idx, scope)
                k_hi[ks] = self._lds_pack(idx + traits.K_LDS_TO_REG_N_STRIP_STRIDE, scope)
            return k_lo, k_hi
        finally:
            self.traits = full


class BwdDqStoreHelper(ParityStoreHelper):
    """The inherited store, pointed at dQ.

    Upstream's `DualwaveStoreHelper` has exactly one output vocabulary and it
    is spelled `o_*`: `o_div` is the descriptor it writes through,
    `stride_o_seq_v` the row pitch it addresses with, `o_oob_off` the sentinel
    it redirects suppressed chunks to, and `hdim_vo` the width it suppresses
    past. This kernel's output is dQ, so those four names are bound to dQ's
    here -- in one block, at the boundary with the code that owns them. The
    context below spells the tensor `DQ`, its strides `stride_dq_*` and its
    descriptor `dq_div`, so nothing outside this block has to know that the
    inherited store calls it O.

    `hdim_qk`, not `hdim_vo`, for the width: dQ is Q-shaped. The two coincide
    in every symmetric build and cross the moment they do not -- the second of
    the two sites the plan's B2 outcome names.

    Rebinding attributes rather than overriding methods: the store helper is a
    dict copy of the context, and the suppression is one comparison inside a
    method that also computes the address, so a copy of it would be a copy of
    both.
    """

    def __init__(self, ctx):
        super().__init__(ctx)
        self.o_div = ctx.dq_div
        self.o_oob_off = ctx.dq_oob_off
        self.stride_o_seq_v = ctx.stride_dq_seq_v
        self.hdim_vo = ctx.hdim_qk


class BwdDbStoreHelper(ParityStoreHelper):
    """`dB = dS` for one KV tile, one element per store.

    **Per element, and that is a correctness decision rather than an
    oversight.** A lane's 16 scores are four runs of four contiguous KV
    columns (`_score_column_runs`), so a 4-wide vector store is available and
    is what the bias *load* uses. It is not available here: a run straddling
    `seqlen_k` would have to be partially written, and a 128-bit store is
    all-or-nothing. Suppressing the whole run instead is exact only when
    `seqlen_k` is a multiple of 4, and silently corrupting the next row of dB
    when it is not is precisely the failure mode this phase is written to
    avoid. So the tail is paid for in full, on a path that is off by default.

    Suppression is by *address*, not by a branch -- pushing the offset past the
    descriptor's `num_records` makes the hardware drop the store -- which is
    the same device `ParityStoreHelper._final_o_global` and `_store_lse_row`
    use.

    It costs 2-5.5x, measured across the ladder, and it is a *cost* rather than
    a ceiling: every rung including 512 builds and is correct with it on. The
    numbers and the register accounting are in the B3 outcome section of
    `sdpa-bwd-plan-gfx950.md`. A vectorised version needs *two* things and not
    one: a runtime second arm for the tile containing `seqlen_k`, and a
    `stride_db_seq_q` divisible by 4, since an 8-byte store off a row pitch of,
    say, 201 elements is 2-byte aligned.

    --- The source is the packed bf16, not the f32 -----------------------------

    **This shortens a live range rather than shrinking a value**, and at the
    top of the ladder that is the difference that matters. `dS` exists twice:
    as 32 f32 scores, and -- after `cast_p` -- as 4 packed `v8` bf16 vectors,
    16 VGPRs against 32. Storing from the f32 list keeps *both* alive across
    the 32-store sequence and its address arithmetic, at exactly the point
    where head_dim 384 and 512 have no registers left. Reading from the packs
    lets the f32 form die at `cast_p`.

    The values are identical: `dB` is a bf16 tensor either way, so the f32 ->
    bf16 rounding happens once regardless -- this only moves *where*. The
    element map is unchanged, because `_pack_p_v8_slices` packs list element
    `pks*8 + j` into lane slot `j` of pack `pks`, in order.
    """

    def store_tile(self, ds_packs, tile_idx, q_row):
        traits = self.traits
        ctx = self.ctx_ref
        lo_packs, hi_packs = ds_packs
        row_base = q_row * fx.Index(ctx.stride_db_seq_q)
        col_base = dualwave._seq_pad_col_base(traits, tile_idx, lane_div_32=self.lane_div_32)
        # `db_store_enabled` folded in here, once per tile, rather than into
        # `live` once per element: it is uniform, and the per-element form would
        # cost an instruction per score on the rungs that have no registers to
        # spare. A disabled store lands on `db_oob_off` and the hardware drops it,
        # which is the suppression device this whole helper is built on.
        in_row = (q_row < self.seqlen_q_v) & ctx.db_store_enabled
        per_pack = traits.OPERAND_LANE_ELEMS
        for half, packs in ((0, lo_packs), (1, hi_packs)):
            for pks in range_constexpr(len(packs)):
                vec = Vec(packs[pks], (per_pack,), self.elem_dtype)
                for j in range_constexpr(per_pack):
                    # The same element -> column map the KV tail mask uses.
                    # Derived once, in `flash_attn_utils`, and read here rather
                    # than transcribed: the mask and this store must agree
                    # about which column an element is, or dB is a permutation
                    # of itself.
                    r = pks * per_pack + j
                    col_i32 = col_base + fx.Int32(dualwave._seq_pad_score_threshold(traits, r) + half * traits.MFMA_M)
                    live = in_row & (col_i32 < self.seqlen_kv_i32)
                    off = fx.Index(live.select(row_base + fx.Index(col_i32), ctx.db_oob_off))
                    buffer_ops.buffer_store(as_mlir_value(vec[j]), ctx.db_rsrc, as_mlir_value(fx.Int32(off)))

    def store_tile_m16(self, ds_pack, tile_idx, q_row, lane):
        """`dB = dS` for the 16-row family.

        Same device as `store_tile` -- per element, address-suppressed, read
        out of the packed bf16 so the f32 form can die -- over the 16-row
        column map instead of `_seq_pad_score_threshold`'s. A lane's four
        accumulator elements are four *contiguous* KV tokens at
        `4 * (lane // 16)`, so the map is an addition rather than a table.

        Slot `4r + i` of the pack is half `r`, element `i`, which is the same
        concatenation GEMM3's B operand relies on: if this indexing is wrong,
        dB is a permutation of itself and dQ is simply wrong, so the two check
        each other.
        """
        traits = self.traits
        ctx = self.ctx_ref
        n = acc_elems(traits)
        row_base = q_row * fx.Index(ctx.stride_db_seq_q)
        in_row = (q_row < self.seqlen_q_v) & ctx.db_store_enabled  # see store_tile
        tok_lane = fx.Int32(4) * fx.Int32(lane // fx.Index(MFMA16_M))
        vec = Vec(ds_pack, (n * 2,), self.elem_dtype)
        for r in range_constexpr(2):
            for i in range_constexpr(n):
                col_i32 = fx.Int32(tile_idx * traits.BLOCK_N) + fx.Int32(r * MFMA16_M + i) + tok_lane
                live = in_row & (col_i32 < self.seqlen_kv_i32)
                off = fx.Index(live.select(row_base + fx.Index(col_i32), ctx.db_oob_off))
                buffer_ops.buffer_store(as_mlir_value(vec[r * n + i]), ctx.db_rsrc, as_mlir_value(fx.Int32(off)))


def build_flash_attn_gfx950_dq(meta, knobs):
    """Build the backward dQ kernel for a resolved (meta, knobs) pair.

    Same shape as the forward's builder and for the same reason: `meta` is what
    the caller asked for, `knobs` is what the tuning policy answered, and
    nothing here falls back to a policy.
    """
    if knobs.BLOCK_DMODEL is None:
        raise ValueError("knobs must be resolved: call `dq_knobs(arch, ...).resolve(meta)` first")
    traits = config.dq_traits(meta, knobs)

    BLOCK_DMODEL = knobs.BLOCK_DMODEL
    PADDED_HEAD = knobs.PADDED_HEAD
    HDIM_QK_FLOOR = traits.HDIM_QK_FLOOR
    # f32 accumulator elements per lane per MFMA -- 16 at 32x32x16. Every
    # `16` the body used to spell over a score half was this, and each was a
    # different quarter of the kernel. See `BwdDqTraits.ACC_ELEMS`.
    ACC_ELEMS = traits.ACC_ELEMS
    STORE_DB = traits.STORE_DB
    # Which MFMA family this build is. `MFMA_N` is the query rows a wave owns,
    # so it is the discriminator rather than a width threshold -- the same
    # shape of choice `WIDE` is in the forward.
    M16 = traits.MFMA_N == 16

    # Everything the build depends on, so two builds differing anywhere cannot collide in the JIT disk cache: every
    # trait field and every knob, and the compile hints.
    CACHE_TAG = (config.build_cache_key(traits, knobs), tuple(sorted(fwd_module._COMPILE_HINTS_KEY.items())))

    lds_elem_dtype = dualwave.dtype_to_elem_type(traits.DTYPE_STR)
    STATIC_WINDOW = knobs.STATIC_WINDOW
    STATIC_SEQLEN = knobs.STATIC_SEQLEN
    # Leading_upper_snake_case parameters (AOTriton's `constexpr_or_i32`): ONE parameter whose annotation is chosen per
    # build, `fx.Constexpr` when the governing knob is on (a JIT build bakes the value) and `fx.Int32` otherwise, so
    # every AOT build, which has the knobs at their defaults, still has a real kernarg.
    WL_ANN = fx.Constexpr if STATIC_WINDOW else fx.Int32
    SEQ_ANN = fx.Constexpr if STATIC_SEQLEN else fx.Int32

    @fx.struct
    class SharedStorage:
        kv: fx.Array[lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]

    @flyc.kernel(known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def fmha_bwd_dq_gfx950_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        B: fx.Pointer,
        DO: fx.Pointer,
        DQ: fx.Pointer,
        DB: fx.Pointer,
        LSE: fx.Pointer,
        Delta: fx.Pointer,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        varlen_bits: fx.Int32,
        num_seqlens: fx.Int32,
        Max_seqlen_q: SEQ_ANN,
        Max_seqlen_k: SEQ_ANN,
        Window_left: WL_ANN,
        Window_right: WL_ANN,
        philox_seed_ptr: fx.Pointer,
        philox_offset1: fx.Pointer,
        philox_offset2: fx.Int64,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        sm_scale: fx.Float32,
        stride_q_batch: fx.Int64,
        stride_q_head: fx.Int64,
        stride_q_seq: fx.Int64,
        stride_k_batch: fx.Int64,
        stride_k_head: fx.Int64,
        stride_k_seq: fx.Int64,
        stride_v_batch: fx.Int64,
        stride_v_head: fx.Int64,
        stride_v_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        stride_db_batch: fx.Int64,
        stride_db_head: fx.Int64,
        stride_db_seq_q: fx.Int64,
    ):
        # Nine pointers in, nine nominal views out -- `wire_view` in
        # `flash_attn_gfx950_helpers.py` has the argument. Everything below this line
        # is unchanged by the move, because every extent the kernel bounds a
        # descriptor with was already built from the strides and seqlens on the
        # wire rather than read off a tensor.
        Q = wire_view(Q)
        K = wire_view(K)
        V = wire_view(V)
        B = wire_view(B)
        DO = wire_view(DO)
        DQ = wire_view(DQ)
        DB = wire_view(DB)
        LSE = wire_view(LSE)
        Delta = wire_view(Delta)
        # dQ's three strides, named once. They reach the context twice, and
        # the second of the two is a residue with a retirement condition:
        #
        # - `dq_strides` is the real one. It builds `dq_div`, bounded by
        #   `hdim_qk`, which is what `BwdDqStoreHelper` writes through.
        # - `o_strides` fills the base class's `O` slot, which this kernel has
        #   no tensor for. It is still needed because
        #   `ParityKernelContext.init_descriptors` builds an O view
        #   unconditionally; the view it builds here is dead -- the store reads
        #   `dq_div` -- and folds away, being pure descriptor arithmetic.
        #
        # Retire the second, along with `O=DQ` below, once the shared context
        # grows the `if self.O is not None` guard that dK/dV's split wants
        # anyway. That file has another owner, so this stays a duplicated
        # argument rather than a cross-file change.
        ctx = BwdDqKernelContext(
            traits,
            strides=(
                stride_q_batch,
                stride_q_head,
                stride_q_seq,
                stride_k_batch,
                stride_k_head,
                stride_k_seq,
                stride_v_batch,
                stride_v_head,
                stride_v_seq,
            ),
            dq_strides=(stride_dq_batch, stride_dq_head, stride_dq_seq),
            do_strides=(stride_do_batch, stride_do_head, stride_do_seq),
            db_strides=(stride_db_batch, stride_db_head, stride_db_seq_q),
            sm_scale=sm_scale,
            num_head_q=num_head_q,
            num_head_k=num_head_k,
            hdim_qk=hdim_qk,
            hdim_vo=hdim_vo,
            padded_head=PADDED_HEAD,
            hdim_qk_floor=HDIM_QK_FLOOR,
            window_left=Window_left,
            window_right=Window_right,
            seqinfo=(seqinfo_q0, seqinfo_q1, seqinfo_k0, seqinfo_k1),
            varlen_bits=varlen_bits,
            num_seqlens=num_seqlens,
            Q=Q,
            K=K,
            V=V,
            DQ=DQ,
            DO=DO,
            DB=DB,
            Delta=Delta,
            Bias=B,
            bias_strides=(stride_b_batch, stride_b_head, stride_b_seq_q),
            philox=(philox_seed_ptr, philox_offset1, philox_offset2, None, None),
            idropout_p=idropout_p,
            dropout_scale=dropout_scale,
            seq_len=Max_seqlen_q,
            seq_len_kv=Max_seqlen_k,
            stride_q_n=stride_q_seq,
            stride_kv_n=stride_k_seq,
            head_dim_runtime=hdim_qk,
            block_table_stride=0,
            LSE=LSE,
        )
        ctx.init_types_and_constants()
        ctx.init_runtime_indices()
        ctx.init_lds(SharedStorage)
        ctx.init_thread_mapping()
        ctx.init_sequence_lengths()
        ctx.init_descriptors()
        ctx.init_workspace()
        ctx.init_philox()
        ctx.init_atoms_and_lds_ptrs()
        ctx.init_dma_thread_offsets()
        ctx.init_tile_bounds()
        ctx.init_active_guard()
        ctx.init_lds_read_bases()
        ctx.init_dma_m0_tables()
        ctx.init_q_row()
        ctx.init_row_inputs()

        kv_gmem_to_lds = BwdDqKvGmemToLdsLoader(ctx)
        # Two reader instances, and they differ in *both* the LDS pitch and the
        # padded-head extent. `k_reader` also serves GEMM3 through the stock
        # `load_v(0)`, since that reads the same region it does.
        k_reader = BwdDqKvLdsToVgprLoader(ctx, v_layout=True, hdim=hdim_qk, hdim_floor=HDIM_QK_FLOOR)
        v_reader = BwdDqKvLdsToVgprLoader(ctx, v_layout=False, hdim=hdim_vo, hdim_floor=traits.HDIM_VO_FLOOR)
        q_loader = ParityQLoader(ctx)
        do_loader = BwdSecondaryQLoader(ctx, ctx.do_div, ctx.stride_do_seq_v, ctx.do_gmem_elem_offset, hdim_vo)
        gemm_helper = ParityGemmHelper(ctx)
        softmax_helper = BwdDqSoftmaxHelper(ctx)
        output_store = BwdDqStoreHelper(ctx)
        row_inputs = BwdRowInputLoader(ctx)
        db_store = BwdDbStoreHelper(ctx)

        def main_body():
            """One KV tile per iteration; three GEMMs and one accumulator."""
            # Neither operand is pre-scaled: `qk_scale` is applied to the f32
            # scores below, where it costs the same FMA the `lse2` subtract
            # needed anyway and does not round Q through bf16 a second time.
            # See `BwdDqSoftmaxHelper.scale_and_sub_lse`.
            q_all_bf16 = q_loader.load_all()
            do_all_bf16 = do_loader.load_all()
            lse2, delta = row_inputs.load(ctx.q_row)

            init_args = [ctx.c_zero_v16f32 for _ in range_constexpr(traits.D_CHUNKS)]
            loop_results = init_args
            for j, loop_args in range(ctx.split_tile(0), ctx.split_t_end, fx.Index(1), init=init_args):
                v_dq = _carried(loop_args, traits.D_CHUNKS)
                tile_start = ctx.tile_start(j)

                # Closes the previous iteration's readers before the DMA
                # overwrites what they were reading. One tile in flight, so
                # there is no second buffer to hide behind.
                _s_barrier()
                kv_gmem_to_lds.stage_k(tile_start)
                kv_gmem_to_lds.stage_v(tile_start)
                _s_waitcnt(0)
                _sched_barrier(0)
                _s_barrier()  # every wave's DMA has landed, not just mine

                # -- GEMM1. S, raw. The scale and the LSE subtract follow.
                v_s = gemm_helper.qk(k_reader.load_k(0), q_all_bf16)
                v_s = softmax_helper.scale_and_sub_lse(v_s, ctx.c_sm_scale_log2e, lse2)
                # **This also adds the bias**, on the 32-row family:
                # `ParitySoftmaxHelper.seq_pad_mask_if_needed` folds
                # `_add_bias_inplace` in ahead of the tail mask, which is
                # exactly the order wanted -- `bias * log2e` onto the scaled
                # score, then `-inf` for the columns that do not exist. Adding
                # it again here is the B6 `cast_p` mistake in a second costume,
                # and it is *not* benign: the backward has no softmax
                # renormalisation, so `P = exp2(S - lse2)` does not absorb a
                # uniform shift the way the forward's does. Measured at 3.3
                # relative error, and even a row-*constant* bias failed --
                # which is what identified it, since shift invariance would
                # have hidden a double add in the forward.
                #
                # Columns past `seqlen_kv` read zero from the buffer bound,
                # which is a *score of zero*, not an absent key: without the
                # mask `exp2(0 - lse2)` contributes a spurious P. After the
                # scale, so the `-inf` it writes never reaches an FMA.
                #
                # **Kept under causal too**, where the forward drops it. Under
                # plain causal it is redundant -- row `i`'s bound is
                # `i + seqlen_kv - seqlen_q <= seqlen_kv - 1`, so the causal
                # mask already kills every column past the sequence -- but a
                # *window* re-points `delta_i32` at an arbitrary right bound
                # and that argument stops holding. Both guards fire only on the
                # last tile, so keeping the cheap one unconditionally costs
                # nothing and removes the case.
                v_s = softmax_helper.seq_pad_mask_if_needed(v_s, j)
                if const_expr(traits.CAUSAL):
                    # The forward's mask and the forward's two-sided guard,
                    # inherited whole: `_causal_mask_inplace` applies the right
                    # bound through `delta_i32` and, under `WINDOW`, the left
                    # one. **No transpose read is inside this region** -- it
                    # touches `v_s` only, and `load_v` sits below. CDNA4 11.4
                    # requires EXEC all 1s across `ds_read_b64_tr_b16`, and
                    # B4 is the first phase where a branch exists to violate
                    # it; `test_no_transpose_read_under_a_restricted_exec`
                    # is what keeps that true.
                    v_s = softmax_helper.causal_mask_prologue_if_needed(v_s, j, kv_end_tile=j + fx.Index(1))

                # -- GEMM2. dP. V is read through the K path, so this is
                #    GEMM1's code with the two operands substituted -- and with
                #    the *other* reader instance, which owns the K-pitch region
                #    and masks against `hdim_vo`.
                v_dp = gemm_helper.qk(v_reader.load_k(0), do_all_bf16)

                # P = exp2(qk_scale*S - lse2). `exp2` is the forward's, split
                # into two halves there for the pipeline and simply adjacent
                # here.
                v_p = softmax_helper.exp2(v_s, 0, ACC_ELEMS)
                v_p = softmax_helper.exp2(v_p, ACC_ELEMS, ACC_ELEMS)

                # dS = P * (dP - delta), elementwise over the 32 scores a lane
                # holds. `delta` is a per-row scalar and a lane owns one row.
                dp_lo, dp_hi = softmax_helper.v_s_vec_to_lists(v_dp)
                if const_expr(traits.ENABLE_DROPOUT):
                    # **On dP, before dS.** See `dropout_dp`: the mask belongs
                    # on the gradient of the dropped output, not on `P`, and
                    # `delta` already carries it through the forward's `O`.
                    dp_lo, dp_hi = softmax_helper.dropout_dp((dp_lo, dp_hi), j, ctx.q_row)
                ds_lo = [None] * ACC_ELEMS
                ds_hi = [None] * ACC_ELEMS
                for r in range_constexpr(ACC_ELEMS):
                    ds_lo[r] = fmul(v_p[0][r], fsub(dp_lo[r], delta, ctx.fm_fast), ctx.fm_fast)
                    ds_hi[r] = fmul(v_p[1][r], fsub(dp_hi[r], delta, ctx.fm_fast), ctx.fm_fast)

                # -- GEMM3. dQ += dS . K, with K read transposed out of the V
                #    slot. `cast_p` gives dS the same K permutation P has, and
                #    the transpose read is built against that permutation, so
                #    the two line up with no further shuffle.
                v_ds = softmax_helper.cast_p((ds_lo, ds_hi))

                if const_expr(STORE_DB):
                    # **After `cast_p`, before `pv`, and reading the packs.**
                    # `dB = dS`, so nothing here is ordered by *arithmetic* --
                    # the store need only precede `sm_scale`, which is after
                    # the loop. It is ordered entirely by register pressure,
                    # and three placements were measured at `B=2 H=8 S=2048`
                    # (TFLOP/s, on the rungs that spill):
                    #
                    #     variant                        256    384    512
                    #     f32 lists, before cast_p       280    149     56
                    #     packs, after cast_p (this)     315    136     83
                    #     packs, after pv                305    146     73
                    #
                    # Reading the packs is the lever: `dS` exists as 32 f32
                    # *and*, after `cast_p`, as 16 packed bf16, and storing
                    # from the f32 form keeps both alive across the 32-store
                    # sequence. Below head_dim 256 it changes nothing -- the
                    # counts are register-identical, so the allocator was
                    # already sinking the f32 form -- and at 512 it is 1.49x.
                    #
                    # 384 prefers the f32 form by 9%, which the lore says a
                    # sweep cannot settle (it wants interleaved A/B), and it is
                    # an allocator outcome rather than a mechanism: the spill
                    # counts are not monotone with the rate in any of the three
                    # columns. This variant is kept because its one *decisive*
                    # measurement -- 1.49x at 512 -- agrees with the mechanism.
                    db_store.store_tile(v_ds, j, ctx.q_row)

                v_dq = gemm_helper.pv(v_ds, k_reader.load_v(0), v_dq)

                loop_results = yield v_dq

            v_dq = _carried(loop_results, traits.D_CHUNKS)
            # `sm_scale` once on the accumulator rather than on every dS: the
            # softmax input is `sm_scale * Q.K^T`, so the factor is linear in
            # the whole sum. AOTriton's `composed_mul_lhs(dq, sm_scale)`.
            softmax_helper.scale_o(v_dq, ctx.c_sm_scale)
            _s_barrier()
            output_store.store_final_o(v_dq, ctx.q_row)

        # head_dim 384 and 512 run a second MFMA family, in its own file: 16
        # rows per wave on `16x16x32`, which halves the loop-invariant Q + dO +
        # dQ that is the whole 512-register file at 32 rows. See
        # `fmha_bwd_dq_m16_gfx950` for why that shape and why `BLOCK_N` 32.
        if const_expr(M16):
            main_body = make_m16_dq_body(
                ctx,
                traits,
                kv_gmem_to_lds=kv_gmem_to_lds,
                db_store=db_store,
                hdim_qk=hdim_qk,
                hdim_vo=hdim_vo,
                hdim_qk_floor=HDIM_QK_FLOOR,
                hdim_vo_floor=traits.HDIM_VO_FLOOR,
                store_db=STORE_DB,
            )

        active = ctx.active
        if active is None:
            main_body()
        else:

            @flyc.jit
            def run_body_if_active():
                if active:
                    main_body()

            run_body_if_active()

    @flyc.jit
    def launch_flash_attn_gfx950_dq(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        B: fx.Pointer,
        DO: fx.Pointer,
        DQ: fx.Pointer,
        DB: fx.Pointer,
        LSE: fx.Pointer,
        Delta: fx.Pointer,
        batch_size: fx.Int32,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        varlen_bits: fx.Int32,
        num_seqlens: fx.Int32,
        Max_seqlen_q: SEQ_ANN,
        Max_seqlen_k: SEQ_ANN,
        Window_left: WL_ANN,
        Window_right: WL_ANN,
        philox_seed_ptr: fx.Pointer,
        philox_offset1: fx.Pointer,
        philox_offset2: fx.Int64,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        sm_scale: fx.Float32,
        stride_q_batch: fx.Int64,
        stride_q_head: fx.Int64,
        stride_q_seq: fx.Int64,
        stride_k_batch: fx.Int64,
        stride_k_head: fx.Int64,
        stride_k_seq: fx.Int64,
        stride_v_batch: fx.Int64,
        stride_v_head: fx.Int64,
        stride_v_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        stride_db_batch: fx.Int64,
        stride_db_head: fx.Int64,
        stride_db_seq_q: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        # Make the build configuration visible to the JIT cache key.
        _ = CACHE_TAG
        # **Sequences, not batches.** A packed `(1, H, T, D)` call holding N
        # sequences is `batch_size=1, num_seqlens=N`; using the batch extent
        # would launch one program for all N. The forward's expression.
        bs_idx = fx.Index(num_seqlens if num_seqlens != fx.Int32(0) else batch_size)
        num_q_blocks = (fx.Index(Max_seqlen_q) + traits.BLOCK_M - 1) // traits.BLOCK_M

        # `llvm.denormal_fpenv` is what reaches the hardware (`.amdhsa_float_denorm_mode_32`); the old
        # `denormal-fp-math-f32` passthrough string never did. The two fast-math passthroughs carry no `no-infs`.
        passthrough_entries = (
            [
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(traits.DAZ)
            else None
        )
        # Head-fastest, the forward's order. Not a free choice there -- it is
        # an L2-locality lever on MI355X's 8 XCDs -- and this kernel streams
        # the same K/V per (batch, head), so it inherits the reasoning.
        fmha_bwd_dq_gfx950_kernel(
            Q,
            K,
            V,
            B,
            DO,
            DQ,
            DB,
            LSE,
            Delta,
            seqinfo_q0,
            seqinfo_q1,
            seqinfo_k0,
            seqinfo_k1,
            varlen_bits,
            num_seqlens,
            Max_seqlen_q,
            Max_seqlen_k,
            Window_left,
            Window_right,
            philox_seed_ptr,
            philox_offset1,
            philox_offset2,
            idropout_p,
            dropout_scale,
            num_head_q,
            num_head_k,
            hdim_qk,
            hdim_vo,
            sm_scale,
            stride_q_batch,
            stride_q_head,
            stride_q_seq,
            stride_k_batch,
            stride_k_head,
            stride_k_seq,
            stride_v_batch,
            stride_v_head,
            stride_v_seq,
            stride_do_batch,
            stride_do_head,
            stride_do_seq,
            stride_dq_batch,
            stride_dq_head,
            stride_dq_seq,
            stride_b_batch,
            stride_b_head,
            stride_b_seq_q,
            stride_db_batch,
            stride_db_head,
            stride_db_seq_q,
            value_attrs={
                "rocdl.waves_per_eu": traits.WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{traits.BLOCK_SIZE},{traits.BLOCK_SIZE}",
                "passthrough": passthrough_entries,
                "llvm.denormal_fpenv": (dualwave.daz_denormal_attr() if const_expr(traits.DAZ) else None),
            },
        ).launch(
            grid=(num_head_q, num_q_blocks, bs_idx),
            block=(traits.BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    def host_args(
        Q,
        K,
        V,
        DO,
        DQ,
        LSE,
        Delta,
        batch_size,
        seqlen_q,
        seqlen_k=None,
        scale=None,
        db=None,
        bias=None,
        window=None,
        varlen=None,
        num_seqlens=0,
        dropout_p=None,
        philox_seed=None,
        philox_offset1=None,
        philox_offset2=0,
        stream=None,
    ):
        """Every kernel argument but the stream, in launch order.

        One place that turns tensors into the wire format, so `launch` and
        `compile_launcher` cannot drift apart -- the same shape, and the same
        reason, as the forward's `host_args`.
        """
        seqlen_k = seqlen_q if seqlen_k is None else seqlen_k
        ptrs, shape_meta, st = abi.prep_tensors(
            [("Q", Q), ("K", K), ("V", V), ("DO", DO), ("DQ", DQ)],
            q_heads=("DO", "DQ"),
        )
        # Used for the checks and the strides, not the pointers: `prep_tensors`
        # builds those as `fx.Uint8`, which carries alignment 1. `wire_ptr`
        # types each operand from its own tensor instead -- see its docstring.
        del ptrs
        num_head_q, num_head_k, hdim_qk, hdim_vo = shape_meta

        # **A build without `padded_head` promises the tile *is* the extent**,
        # on both axes, and this is the only place that can be checked. Found
        # by a test of this file's own making: `build(head_dim=128)` handed a V
        # of width 64 resolves to `padded_head=False`, emits no mask at all,
        # and reduces `dP` over the caller's D-axis slack. The answer is finite
        # and 0.70 relative error. A real caller can make the same mistake, so
        # the guard belongs here rather than in the test that found it.
        if not PADDED_HEAD and not (hdim_qk == hdim_vo == BLOCK_DMODEL):
            raise ValueError(
                f"this build is not compiled for a padded head, so it requires hdim_qk == hdim_vo == "
                f"{BLOCK_DMODEL}; got hdim_qk {hdim_qk}, hdim_vo {hdim_vo}. Pass head_dim_v to the "
                "builder so the D-axis masks are emitted."
            )

        # `abi.varlen_args` is gfx1201's, reused unedited: it encodes the same
        # wire format the forward uses, and it is where the two host-side
        # checks live that no kernel can make -- `batch_size` must be the
        # tensor's batch extent whatever the layout, and a packed
        # `num_seqlens` must agree with the length array. Passing the sequence
        # count where the batch extent belongs launches N programs over a
        # 1-batch tensor and every one of them addresses a plausible row.
        # `varlen=None` is not an error and never was a separate build: it is
        # `bits == 0`, which the decode answers with dense addressing.
        vl = abi.varlen_args(varlen, seqlen_q, seqlen_k, Q, batch_size, num_seqlens)
        if STATIC_SEQLEN and varlen is not None:
            # The build cannot know the layout, so a baked length is dense-only.
            raise ValueError(
                "this build has STATIC_SEQLEN=True and bakes (seqlen_q, seqlen_k); a varlen call has per-sequence "
                "lengths. Build without STATIC_SEQLEN for varlen"
            )

        # **`(batch * heads, tokens)`, and the shape is shared with dK/dV.**
        # Both backward kernels take the same two row tensors and read them
        # with the same `common.lse_row_addressing`, so the host check is the
        # shared `abi.row_tensor_arg` rather than a second spelling of it.
        # This is the layout the forward writes LSE in -- `q_head_idx *
        # seq_len_v + q_row` inside a per-batch slab -- viewed at rank 2; a
        # caller holding `(B, H, S)` passes `lse.view(-1, S)`, which is free
        # on a contiguous tensor.
        #
        # **Varlen moves the token pitch**, and this is the one input where
        # that is invisible from the tensor alone: the kernel derives the
        # pitch from `lse_token_pitch` -- the batch total for a stacked Q side,
        # `max_seqlen_q` otherwise -- rather than reading a stride, so a
        # caller who sized LSE by `max_seqlen_q` under a packed layout gets a
        # plausible value for the wrong row. `row_tensor_arg` checks the
        # declared layout against the bits, which is the only place that can
        # be done. The forward's P4 got this wrong in a way only `_TH`
        # exposed.
        rows_expected = int(num_seqlens or batch_size) * num_head_q
        for name, t in (("logsumexp", LSE), ("delta", Delta)):
            if t is None:
                raise ValueError(f"{name} is required: the backward reads it, it is not recomputed")
            abi.row_tensor_arg(t, name, num_head_q, seqlen_q, varlen)
            if varlen is None and t.shape[0] != rows_expected:
                raise ValueError(f"{name} must be ({rows_expected}, {seqlen_q}); got {tuple(t.shape)}")

        # A dB build must be handed a tensor and a build without dB must not
        # be: silently ignoring one returns gradients that are the right shape
        # with the bias gradient missing, and it is only ever passed by a
        # caller who believes it is being written.
        if STORE_DB and db is None:
            raise ValueError("this build has store_db=True and requires a (batch, num_heads_q, seqlen_q, seqlen_k) dB")
        if db is not None and not STORE_DB:
            raise ValueError("this build was not compiled for dB; pass bias=True in FmhaInputMetadata")
        if db is not None:
            # **dB follows Q's batch and row layout, not a dense one.** Its
            # descriptor is rebased at the same `(batch, head, row origin)` dQ
            # uses, so under a packed Q the rows are packed too and the batch
            # extent is Q's. Only the KV axis is sized independently, to
            # `max_seqlen_k`, because a row's live columns are a per-sequence
            # prefix of it.
            want = (int(Q.shape[0]), num_head_q, int(Q.shape[2]), int(seqlen_k))
            if tuple(db.shape) != want:
                raise ValueError(
                    f"dB must follow Q's batch and row layout with a seqlen_k column axis, {want}; "
                    f"got {tuple(db.shape)}"
                )
            if db.stride(3) != 1:
                raise ValueError(f"dB needs a contiguous seqlen_k axis; strides are {tuple(db.stride())}")
            if db.dtype != Q.dtype:
                raise ValueError(f"dB must match Q's dtype ({Q.dtype}), got {db.dtype}")
        # **The seed and offset the forward *reported*, not what its caller
        # passed.** Under a captured graph the caller's offset is a pointer the
        # forward re-reads, so the two need not be equal -- and the mask has to
        # be regenerated bit-identically or the gradients are quietly wrong.
        # `abi.dropout_args` is gfx1201's: it turns the probability into the
        # i32 threshold the raw random is compared against and the `1/(1-p)`
        # survivor scale, once per call, and keeps the counter as the
        # (pointer, immediate) pair torch splits it into.
        *dp, dp_keepalive = abi.dropout_args(
            bool(traits.ENABLE_DROPOUT),
            dropout_p,
            philox_seed,
            philox_offset1,
            philox_offset2,
            device=Q.device,
            stream=stream,
        )
        if dropout_p is not None and not traits.ENABLE_DROPOUT:
            raise ValueError("this build was not compiled for dropout; pass dropout=True in FmhaInputMetadata")

        # The window pair rides the wire as signed i32 for every build, so every build shares one ABI; a *sentinel*
        # rather than a bound goes on the wire for the fixed alignments, because the kernel resolves it against each
        # sequence's own lengths (under varlen there is more than one pair of lengths to resolve against).
        if not traits.WINDOW:
            if window is not None:
                raise ValueError("this build was not compiled for a window; pass window=True in FmhaInputMetadata")
            wl, wr = 0, 0
        else:
            if window is None:
                raise ValueError(
                    "a window build requires window=(left, right); pass "
                    "(common.WINDOW_TOPLEFT, common.WINDOW_TOPLEFT) for top-left causal or "
                    "(common.WINDOW_BOTRIGHT, common.WINDOW_BOTRIGHT) for bottom-right"
                )
            wl, wr = int(window[0]), int(window[1])

        # **The bias *input* is a different path from the dB *output***, and
        # having written dB is the part that gets mistaken for done. `B` is
        # added to the scores before the softmax; `dB` is the gradient with
        # respect to it. A build can read one, write the other, both or
        # neither.
        if traits.BIAS_TYPE and bias is None:
            raise ValueError("this build has bias=True and requires a (batch, num_heads_q, seqlen_q, seqlen_k) tensor")
        if bias is not None and not traits.BIAS_TYPE:
            # Silently ignoring it returns dense attention's gradient: right
            # shape, finite, wrong. It is only ever passed by a caller who
            # believes it is being applied.
            raise ValueError("this build was not compiled for bias; pass bias=True in FmhaInputMetadata")
        if bias is not None:
            want = (int(Q.shape[0]), num_head_q, int(Q.shape[2]), int(seqlen_k))
            if tuple(bias.shape) != want:
                raise ValueError(f"bias must follow Q's batch and row layout, {want}; got {tuple(bias.shape)}")
            if bias.stride(3) != 1:
                raise ValueError(f"bias needs a contiguous seqlen_k axis; strides are {tuple(bias.stride())}")
        bias_t = bias if bias is not None else DQ
        # `_seq_q` is the *query* axis, matching `stride_db_seq_q` two lines
        # down and the 24 other occurrences the rename settled on. `_seq`
        # alone does not say which sequence; `_seq0` would reintroduce the
        # numeric slots the stride rename removed, and leaves `_seq_k` with
        # nothing to be called if the k axis ever stops being contiguous.
        bias_st = tuple(int(x) for x in bias.stride()[:3]) if bias is not None else (0, 0, 0)

        db_t = db if db is not None else DQ
        db_st = tuple(int(x) for x in db.stride()[:3]) if db is not None else (0, 0, 0)

        return (
            # Nine pointers, not nine tensors. LSE and Delta are pinned to f32
            # rather than read off the tensor: both are required and both are
            # already checked f32 by `abi.row_tensor_arg`, so naming the type
            # here keeps the slot's type a property of the ABI rather than of
            # the call.
            wire_ptr(Q),
            wire_ptr(K),
            wire_ptr(V),
            wire_ptr(bias_t),
            wire_ptr(DO),
            wire_ptr(DQ),
            wire_ptr(db_t),
            wire_ptr(LSE, fx.Float32),
            wire_ptr(Delta, fx.Float32),
            int(batch_size),
            vl[1],
            vl[2],
            vl[3],
            vl[4],
            vl[0],  # varlen_bits
            int(num_seqlens),
            *(
                (int(seqlen_q), int(seqlen_k)) if STATIC_SEQLEN else vl[5:]
            ),  # Max_seqlen_q/k (the decode's MAX fallback)
            wl,
            wr,
            dp[0],
            dp[1],
            dp[2],
            dp[3],  # idropout_p
            dp[4],  # dropout_scale
            num_head_q,
            num_head_k,
            hdim_qk,
            hdim_vo,
            abi.resolve_scale(Q, scale, PADDED_HEAD, 1.0 / (BLOCK_DMODEL**0.5)),
            *st,
            *bias_st,
            *db_st,
        ), (stream, dp_keepalive)

    # The compile hints ride on the launcher, so a direct `jf(*args)` (AOTriton's driver) and the JIT compile
    # identically. `flyc.compile[hints](launcher)` attaches them and returns the launcher.
    flyc.compile[fwd_module._COMPILE_HINTS](launch_flash_attn_gfx950_dq)
    COMPILED = {}

    def launch(*args, **kwargs):
        packed, (stream, dp_keepalive) = host_args(*args, **kwargs)
        # The compiled function is per signature, and a baked length or window pair is part of it (`Constexpr`
        # values): one cache per value, or the second would reuse the first one's binary.
        baked = (packed[16], packed[17]) if STATIC_SEQLEN else ()
        baked += (packed[18], packed[19]) if STATIC_WINDOW else ()
        result = abi.run_compiled(
            COMPILED.setdefault(baked, abi.new_compiled_cache()),
            launch_flash_attn_gfx950_dq,
            *packed,
            stream if stream is not None else fx.Stream(None),
        )
        del dp_keepalive  # held across the launch call
        return result

    def compile_launcher(*args, **kwargs):
        packed, extras = host_args(*args, **kwargs)
        return flyc.compile(launch_flash_attn_gfx950_dq, *packed, fx.Stream(extras[0]))

    launch.compile = compile_launcher
    launch.traits = traits
    launch.knobs = knobs
    launch.launcher = launch_flash_attn_gfx950_dq
    launch.host_args = host_args
    return launch
