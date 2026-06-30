# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL intra-node ``mega_moe`` mega-kernel (dispatch phase).

This is the device side of the FlyDSL port of DeepGEMM's
``sm100_fp8_fp4_mega_moe`` (see
``DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh``
and ``.../layout/mega_moe.cuh`` / ``.../scheduler/mega_moe.cuh``). It is a
single fused, warp-specialised kernel. This milestone implements the
dispatch role plus the **L1 grouped GEMM** compute phase (run after a grid
barrier, all warps, no warp specialisation yet); the remaining compute
roles (SwiGLU / L2 GEMM / combine) are still stubbed, mirroring the
DeepGEMM warp-role split.

Parity notes (AMD gfx950 vs NVIDIA SM100):

* Wavefronts are 64 lanes (not 32). All lane-strided loops use step 64.
* NVIDIA-specific machinery is replaced by portable equivalents:
  - TMA load/store pulls  -> ``buffer_load`` / ``buffer_store`` (P2P).
  - ``__reduce_*_sync`` round-robin rank peeling over ``kNumRanks`` lanes
    -> a scalar loop over the (compile-time, <= 8) ranks, computed
    redundantly on every lane (no warp shuffle needed).
  - NVLink barriers + packed ``expert_send_count`` arrival counting ->
    a grid barrier (atomic counter + ``mori`` wait) plus the packed
    ``expert_recv_count_sum`` high-bits "all ranks arrived" spin, exactly
    like DeepGEMM's ``fetch_expert_recv_count``.
  - Per-block LDS counting of ``expert_token_count`` is dropped; each
    token-topk does one global atomic slot-allocation on
    ``expert_send_count`` directly (simpler, correct; the LDS aggregation
    was a throughput optimisation and is deferred).

The byte layout is identical to the host op (``mega_moe_intranode_op.py``)
so the same ``Workspace`` / ``Buffer`` offset math applies. Every remote
segment address is ``peer_base[pe] + offset``.

Launcher contract (called by ``FlyDSLMegaMoEIntraNodeOp``)::

    fn(addr_local_base, addr_peer_base, addr_cumulative_stats,
       num_tokens, stream)

* ``addr_local_base``    : i64, this rank's symmetric-buffer head.
* ``addr_peer_base``     : i64, device array ``i64[world_size]`` of peer
                           buffer heads (``addr_peer_base[rank] ==
                           addr_local_base``).
* ``addr_cumulative_stats``: i64, ``int32[num_experts_per_rank]`` or 0.
* ``num_tokens``         : i32, this rank's current token count.
"""

from __future__ import annotations

import os

import mori.ir.flydsl as mori_shmem

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl._mlir.dialects import memref as _memref_d
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import T, arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.arith import CmpIPredicate
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from flydsl.expr.typing import Stream

from .mega_moe_intranode_op import (
    _align,
    _num_max_pool_tokens,
    _num_padded_sf_pool_tokens,
    _KCANDIDATE_BLOCK_M,
)
from .mfma_preshuffle_pipeline import _buffer_load_vec, crd2idx, make_preshuffle_b_layout
from .dpp_utils import dpp_xor_f32
from flydsl.expr.typing import T as _Tt
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

# Bump when generated kernel shape changes.
_MEGA_MOE_JIT_SCHEMA_VERSION = "v10-dispatch+l1ws+fuseswiglu+l2ws+overlap+fusecombine+fp8combine+tailreset-2026-06"


# --------------------------------------------------------------------------
# Low-level device helpers (mirrors dispatch_combine_intranode_kernel.py).
# --------------------------------------------------------------------------
def _to_i64(v):
    """Zero-extend an i32 value to i64 ``ArithValue``."""
    return arith.extui(T.i64(), arith.unwrap(v))


def _to_i32(v):
    """Truncate an i64 value to i32 ``ArithValue``."""
    return arith.trunci(T.i32(), arith.unwrap(v))


def _c64(x):
    """i64 constant from a Python int."""
    return arith.constant(x, type=T.i64())


def _to_ptr_global(v):
    """Cast an i64 address to ``!llvm.ptr<1>`` (global address space)."""
    return _llvm_d.IntToPtrOp(_llvm_d.PointerType.get(address_space=1), arith.unwrap(v)).result


def store_i32_system(addr_i64, val):
    """System-scope release i32 store at ``addr_i64``."""
    gptr = _to_ptr_global(addr_i64)
    _llvm_d.StoreOp(arith.unwrap(val), gptr, alignment=4,
                    ordering=_llvm_d.AtomicOrdering.release, syncscope="one-as")


def store_i64_system(addr_i64, val):
    """System-scope release i64 store at ``addr_i64``."""
    gptr = _to_ptr_global(addr_i64)
    _llvm_d.StoreOp(arith.unwrap(val), gptr, alignment=8,
                    ordering=_llvm_d.AtomicOrdering.release, syncscope="one-as")


def store_vec_plain(addr_i64, vec_val, align):
    """Plain (non-atomic) vector global store at ``addr_i64``.

    Ordering / cross-agent visibility is NOT provided per-store; the caller
    must issue a single ``fence_system_release()`` after a batch of these
    before the completion signal (release/acquire contract).  This is the
    bulk-transfer fast path: one wide ``global_store_dwordx4`` instead of N
    per-word system-release stores (which each carry a cache-flush).
    """
    gptr = _to_ptr_global(addr_i64)
    _llvm_d.StoreOp(arith.unwrap(vec_val), gptr, alignment=align)


def fence_system_acquire():
    """System-scope acquire fence."""
    _llvm_d.FenceOp(_llvm_d.AtomicOrdering.acquire, syncscope="one-as")


def fence_system_release():
    """System-scope release fence."""
    _llvm_d.FenceOp(_llvm_d.AtomicOrdering.release, syncscope="one-as")


def fence_agent_acquire():
    """Agent (single-device) scope acquire fence.  Sufficient for intra-rank
    handoffs where producer + consumer run on the SAME GPU; far cheaper than a
    system-scope fence (no cross-device/system coherence flush)."""
    _llvm_d.FenceOp(_llvm_d.AtomicOrdering.acquire, syncscope="agent")


def fence_agent_release():
    """Agent (single-device) scope release fence (see ``fence_agent_acquire``)."""
    _llvm_d.FenceOp(_llvm_d.AtomicOrdering.release, syncscope="agent")


def atomic_add_i32_release_agent(addr_i64, val):
    """Agent-scope release 32-bit ``atomic fetch-and-add`` (intra-rank signal)."""
    ptr = _to_ptr_global(addr_i64)
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add, ptr, arith.unwrap(val),
        _llvm_d.AtomicOrdering.release, syncscope="agent").res


def load_i32_relaxed_agent(addr_i64):
    """Agent-scope relaxed (monotonic) i32 load for the intra-rank barrier spin."""
    gptr = _to_ptr_global(addr_i64)
    return _llvm_d.LoadOp(T.i32(), gptr, alignment=4,
                          ordering=_llvm_d.AtomicOrdering.monotonic,
                          syncscope="agent").result


def atomic_add_i32(addr_i64, val):
    """Monotonic global 32-bit ``atomic fetch-and-add``; returns old value."""
    ptr = _to_ptr_global(addr_i64)
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add, ptr, arith.unwrap(val),
        _llvm_d.AtomicOrdering.monotonic).res


def atomic_add_i32_release_sys(addr_i64, val):
    """System-scope release 32-bit ``atomic fetch-and-add`` (cross-rank signal)."""
    ptr = _to_ptr_global(addr_i64)
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add, ptr, arith.unwrap(val),
        _llvm_d.AtomicOrdering.release, syncscope="one-as").res


def atomic_add_u64_release_sys(addr_i64, val_i64):
    """System-scope release 64-bit ``atomic add`` (cross-rank signal)."""
    ptr = _to_ptr_global(addr_i64)
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add, ptr, arith.unwrap(val_i64),
        _llvm_d.AtomicOrdering.release, syncscope="one-as").res


def load_i32_relaxed_sys(addr_i64):
    """System-scope relaxed (monotonic) i32 load; matches mori's spin load.

    Correctness comes from a single ``fence_system_acquire()`` issued after
    the spin loop, not from per-iteration acquire ordering -- identical to
    ``mori`` ``ShmemTypeWaitUntilEquals`` (``while (AtomicLoadRelaxedSystem
    (addr) != val) {}``).
    """
    gptr = _to_ptr_global(addr_i64)
    return _llvm_d.LoadOp(T.i32(), gptr, alignment=4,
                          ordering=_llvm_d.AtomicOrdering.monotonic,
                          syncscope="one-as").result


# Sense-reversal (phase-flipping) grid barrier tag, mirroring DeepGEMM's
# ``kFinishSumTag`` (``comm/barrier.cuh``). Each grid barrier round adds
# exactly ``_FINISH_TAG`` to its counter, so the counter only ever toggles
# its top bit (low bits net to zero): reset-free and overflow-immune (u32
# wrap preserves the toggle). ``_SIGN_BIT`` is the same bit as a signed-i32
# constant, used as the mask in the wait predicate.
_FINISH_TAG = 0x80000000
_SIGN_BIT = -0x80000000

# ``s_waitcnt`` immediate encoding for ``lgkmcnt(0)`` only (vmcnt / expcnt left
# at max).  Used to drain LDS (DS) traffic before an ``s_barrier`` so the LDS
# producer/consumer ring is ordered across waves (mirrors flash_attn_gfx950).
_LGKMCNT_0_ONLY = 0xC07F

# ``s_waitcnt`` immediate for ``vmcnt(0)`` only (expcnt / lgkmcnt left at max).
# Drains in-flight VMEM (global) traffic so a producer's pool store is visible
# to the SAME wave's later pool loads (used by the fused-pull "pull once"
# write-through: chunk-0 stores the pulled token/SF into the local L1 pool,
# later chunks read it back instead of re-pulling from the peer).  Encoded as
# vmcnt=0 (lo bits[3:0]=0, hi bits[15:14]=0), expcnt=7 (bits[6:4]), lgkmcnt=63
# (bits[13:8]) -> 0 | (7<<4) | (63<<8) = 0x3F70.
_VMCNT_0_ONLY = 0x3F70


# --------------------------------------------------------------------------
# Kernel builder.
# --------------------------------------------------------------------------
def make_mega_moe_kernel(
    *,
    rank: int,
    world_size: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_max_tokens_per_rank: int,
    layout,
    block_m: int,
    num_sms: int,
    num_warps_per_block: int,
    num_dispatch_warps: int,
    enable_l1_compute: bool = False,
    enable_l2_compute: bool = False,
    enable_combine: bool = False,
    enable_l1_ws: bool = False,
    enable_l2_ws: bool = False,
    enable_dispatch_overlap: bool = False,
    enable_fuse_swiglu: bool = False,
    enable_fuse_combine: bool = False,
    enable_fp8_combine: bool = False,
    num_l1_load_warps: int = 2,
    num_l1_stages: int = 3,
    l1_n16_per_block: int = 16,
):
    """Build the warp-specialised mega_moe ``@flyc.kernel``.

    ``layout`` is the host ``_MegaMoELayout`` (byte offsets). Geometry is
    fully baked in; the kernel only takes runtime base pointers + token
    count.

    The kernel is always built as one monolithic launch (dispatch + L1 GEMM +
    fused SwiGLU + L2 GEMM + combine + tail reset); ``enable_dispatch`` /
    ``enable_l1_gemm`` / ``enable_l2_gemm`` / ``enable_reset`` are therefore
    constant ``True``.

    Returns ``(kernel, allocator)``; ``allocator`` is the LDS
    ``SmemAllocator`` (``None`` when ``enable_l1_ws`` is False) and must be
    ``finalize()``-d inside the launcher's ``gpu_module_body``.
    """
    # Monolithic single-launch: every phase is compiled into one kernel.
    enable_dispatch = True
    enable_l1_gemm = True
    enable_l2_gemm = True
    enable_reset = True
    # Removed experimental paths, pinned to their production-off constants:
    #   * merged scheduler (L1/L2 wave interleave) + its expert-per-wave split,
    #   * fused pull (peer-direct WS loadA) + its pull-once mirror,
    #   * producer/consumer SM split for dispatch overlap,
    #   * the L2-readiness gate (only ever turned on by the merged scheduler),
    #   * the measurement-only PROBE_SKIP_* / PROBE_LOCAL_SCATTER diagnostics.
    enable_merged_scheduler = False
    num_experts_per_wave = 0
    num_producer_blocks = 0
    enable_fused_pull = False
    enable_fused_pull_once = False
    enable_l2_readiness = False
    probe_skip_l2gemm = False
    probe_skip_combine = False
    probe_skip_fused_swiglu = False
    npes = world_size
    epr = num_experts // world_size
    m = num_max_tokens_per_rank
    nmre = npes * m  # num_max_recv_tokens_per_expert (== src_token_topk slab)
    block_threads = num_warps_per_block * 64
    global_warp_num = num_sms * num_warps_per_block

    # Pool sizing (replicates host helpers so the kernel and the host views
    # agree on strides). ``padded_sf`` is the *max* over candidate BLOCK_M,
    # matching the host's M-major SF view stride.
    num_max_pool_tokens = _num_max_pool_tokens(npes, m, num_topk, epr)
    padded_sf = max(_num_padded_sf_pool_tokens(num_max_pool_tokens, bm) for bm in _KCANDIDATE_BLOCK_M)
    sf_block_m = _align(block_m, 128)

    # Dispatch lane mapping (DeepGEMM ``kNumTokensPerWarp`` adapted to 64).
    tokens_per_warp = max(1, 64 // num_topk)
    activate_lanes = tokens_per_warp * num_topk

    # Token / SF chunking.
    n_i32 = hidden // 4               # fp8 token bytes / 4
    sf_uint32 = hidden // 128         # K-major SF words per token

    # Byte offsets (compile-time).
    off_barrier = layout.barrier_off
    off_send = layout.expert_send_count_off
    off_recv = layout.expert_recv_count_off
    off_sum = layout.expert_recv_count_sum_off
    off_l1arr = layout.l1_arrival_count_off
    off_l2mask = layout.l2_arrival_mask_off
    off_srctt = layout.src_token_topk_idx_off
    off_tsm = layout.token_src_metadata_off
    off_inp_tok = layout.input_token_off
    off_inp_sf = layout.input_sf_off
    off_inp_idx = layout.input_topk_idx_off
    off_inp_wts = layout.input_topk_weights_off
    off_l1_tok = layout.l1_token_off
    off_l1_sf = layout.l1_sf_off
    off_l1_wts = layout.l1_topk_weights_off
    off_l2_tok = layout.l2_token_off
    off_l2_sf = layout.l2_sf_off
    off_combine = layout.combine_token_off
    off_combine_scale = layout.combine_scale_off
    _fp8_combine = bool(enable_fp8_combine)
    _probe_local_scatter = False
    # fp8-combine staging strides (bytes).  fp8 data reuses ``off_combine``
    # with a halved per-row stride (1 byte/elem); the per-16-channel bf16
    # scales live in their own ``off_combine_scale`` region.
    _comb_fp8_slab = m * hidden          # bytes per topk slot (1 byte/elem)
    _comb_fp8_row = hidden               # bytes per token row
    _comb_sc_slab = m * (hidden // 16) * 2   # bytes per topk slot (bf16 scale)
    _comb_sc_row = (hidden // 16) * 2        # bytes per token row

    # Cross-rank "all ranks arrived" threshold: sum high 32 bits == npes.
    sum_arrival_threshold = npes * (1 << 32) - 1
    _INF = 1 << 30  # signed-safe sentinel for min-reduction

    # -----------------------------------------------------------------
    # L1 warp-specialisation (producer/consumer + LDS pipeline) layout.
    # The block's warps split into loadA / loadB producers and mma+epilogue
    # consumers; a (expert, m16, n_block) persistent scheduler hands one
    # 16-row x (l1_n16_per_block*16)-col output region to the whole block,
    # streamed over K through a ``num_l1_stages`` LDS ring.  Producers
    # pre-stage the exact per-lane MFMA operands (a128 / b16 / SF), so the
    # consumer's MFMA is bit-identical to the non-WS path.
    # -----------------------------------------------------------------
    # L1 warp-specialisation (producer/consumer + LDS pipeline) layout.
    # The block's warps split into loadA / loadB producers and mma+epilogue
    # consumers; the fused L1 epilogue produces the L2 fp8 pool, then a global
    # L1->L2 grid barrier separates the L1 and L2 GEMM phases.
    # -----------------------------------------------------------------
    _ws_alloc = None
    _ws_any = enable_l1_ws or enable_l2_ws
    if _ws_any:
        if num_l1_load_warps < 1:
            raise ValueError("num_l1_load_warps must be >= 1 for warp specialisation")
        if num_warps_per_block - num_l1_load_warps < 1:
            raise ValueError(
                f"need >=1 consumer warp: num_warps_per_block={num_warps_per_block} "
                f"num_l1_load_warps={num_l1_load_warps}")
        if num_l1_stages < 2:
            raise ValueError("num_l1_stages must be >= 2 for the ping-pong ring")
        _ws_num_consumer_warps = num_warps_per_block - num_l1_load_warps

        def _pick_n16pb(n16_total, requested=None):
            # Clamp n16-per-block to what divides the total and the consumer
            # count (largest valid value <= ``requested``, default l1 knob).
            n = l1_n16_per_block if requested is None else requested
            while n > 1 and (n16_total % n != 0
                            or n % _ws_num_consumer_warps != 0):
                n -= 1
            if n < 1 or n16_total % n != 0 or n % _ws_num_consumer_warps != 0:
                raise ValueError(
                    f"cannot pick n16_per_block dividing n16_total={n16_total} "
                    f"and consumer_warps={_ws_num_consumer_warps}")
            return n

        # Per-phase N16 tiling: L1 has N = 2*ih, L2 has N = hidden.
        _n16_total_l1 = (2 * intermediate_hidden) // 16
        _n16_total_l2 = hidden // 16
        if enable_l1_ws and enable_fuse_swiglu:
            # Fused per-chunk pairing: chunk c uses gate N-block c + up N-block
            # c+n_half, so n16pb MUST divide the gate tile count (ih/16); also
            # cap it (<=8) so the bf16 activation staging (16 rows x 2*n16pb*16
            # cols) stays small enough that ring+staging keeps 2 blocks/CU.
            _gate_tiles = intermediate_hidden // 16
            _n = min(l1_n16_per_block, 8)
            while _n > 1 and (_gate_tiles % _n != 0 or _n % _ws_num_consumer_warps != 0):
                _n -= 1
            if _gate_tiles % _n != 0 or _n % _ws_num_consumer_warps != 0:
                raise ValueError(
                    f"fused SwiGLU: cannot pick n16pb<=8 dividing ih/16={_gate_tiles} "
                    f"and consumer_warps={_ws_num_consumer_warps}")
            _ws_n16pb = _n
        elif enable_l1_ws:
            _ws_n16pb = _pick_n16pb(_n16_total_l1)
        else:
            _ws_n16pb = 1
        _ws_n_owned = (_ws_n16pb // _ws_num_consumer_warps) if enable_l1_ws else 0
        _ws_n_blocks = (_n16_total_l1 // _ws_n16pb) if enable_l1_ws else 0
        # L2 N16-per-block is independent of L1.  L2 has N = hidden (huge) and a
        # short K = ih, so a WIDE N tile leaves too few N-blocks to fill the
        # persistent grid (and inflates per-block regs) -- measured 7168/2048:
        # n16pb=16 -> 3.6 ms vs n16pb=8 -> 1.1 ms.  Default to 8 (narrower =
        # more blocks = better occupancy); env can override.
        _l2_n16pb_req = int(os.environ.get("MEGA_MOE_L2_N16PB", "8"))
        _ws_n16pb_l2 = _pick_n16pb(_n16_total_l2, _l2_n16pb_req) if enable_l2_ws else 1
        _ws_n_owned_l2 = (_ws_n16pb_l2 // _ws_num_consumer_warps) if enable_l2_ws else 0
        _ws_n_blocks_l2 = (_n16_total_l2 // _ws_n16pb_l2) if enable_l2_ws else 0

        # Single LDS ring, reused across phases (L1 and L2 run in disjoint
        # grid-barrier-separated phases), sized to the max n16pb requested.
        _ws_n16pb_max = max(_ws_n16pb if enable_l1_ws else 1,
                            _ws_n16pb_l2 if enable_l2_ws else 1)
        # M-batching: a single scheduler super-tile owns MB*16 M-rows and
        # streams ONE B-ring residency across all of them, so each weight tile
        # is reused MB*16 rows deep (vs 16) -- restoring aiter-style M=64 reuse
        # without growing the (dominant) B ring.  Only the non-fused GEMM path
        # (L1-no-fuse + L2) uses it; the A ring grows x MB.  L1 and L2 are
        # independently tunable (MB / stages); clamp so MB*16 divides block_m
        # (all subtiles share one expert) and the pool tiles.
        def _clamp_m16pb(v):
            v = int(v)
            while v > 1 and ((block_m // 16) % v != 0
                             or (num_max_pool_tokens // 16) % v != 0):
                v -= 1
            return v
        _ws_m16pb = _clamp_m16pb(os.environ.get("MEGA_MOE_GEMM_M16PB", "1"))
        _ws_m16pb_l2 = _clamp_m16pb(os.environ.get("MEGA_MOE_L2_M16PB", str(_ws_m16pb)))
        # Independent software-pipeline depth per phase (ring sized to the max).
        _ws_l1_stages = num_l1_stages
        _ws_l2_stages = max(2, int(os.environ.get("MEGA_MOE_L2_STAGES", str(num_l1_stages))))
        _ws_m16pb_max = max(_ws_m16pb, _ws_m16pb_l2)
        _ws_stages_max = max(_ws_l1_stages, _ws_l2_stages)
        # LDS ring sizes, in i32 elements (sized to the max over phases).
        _ws_a_words = _ws_stages_max * _ws_m16pb_max * 64 * 8   # a128 (vec8 i32) per lane x MB
        _ws_asf_words = _ws_stages_max * _ws_m16pb_max * 64     # sa (i32) per lane x MB
        _ws_b_words = _ws_stages_max * _ws_n16pb_max * 64 * 4   # b16 (vec4 i32)
        _ws_bsf_words = _ws_stages_max * _ws_n16pb_max * 64     # sb (i32)
        _ws_alloc = SmemAllocator(None, arch="gfx950", global_sym_name="mega_moe_l1_ws_smem")
        _ws_off_a = _ws_alloc._align(_ws_alloc.ptr, 32)
        _ws_alloc.ptr = _ws_off_a + _ws_a_words * 4
        _ws_off_asf = _ws_alloc._align(_ws_alloc.ptr, 16)
        _ws_alloc.ptr = _ws_off_asf + _ws_asf_words * 4
        _ws_off_b = _ws_alloc._align(_ws_alloc.ptr, 16)
        _ws_alloc.ptr = _ws_off_b + _ws_b_words * 4
        _ws_off_bsf = _ws_alloc._align(_ws_alloc.ptr, 16)
        _ws_alloc.ptr = _ws_off_bsf + _ws_bsf_words * 4
        # Fused-SwiGLU activation staging: per-CHUNK gate||up tile, in clean
        # [16 rows x 2*(n16pb*16)] bf16 layout.  The consumer de-swizzles gate
        # N-block c AND up N-block c+n_half into this small buffer, then a block
        # barrier + SwiGLU + mxfp8 quant sweep reads gate[ch]/up[ch] pairs and
        # writes straight to the L2 fp8 pool -- no f32 global round-trip, no grid
        # barrier #3.  Keeping it per-chunk (and bf16) is what lets ring+staging
        # stay <=80KB so the kernel keeps 2 blocks/CU (vs full-row 64KB -> 1/CU).
        # Element count is in i16 units.  Only allocated when fused.  With
        # M-batching the chunk stages MB*16 rows (one B-ring residency reused
        # across all of them), so the staging grows x MB.
        _ws_act_words = (_ws_m16pb * 16 * 2 * (_ws_n16pb * 16)) if enable_fuse_swiglu else 0
        _ws_off_act = _ws_alloc._align(_ws_alloc.ptr, 16)
        if enable_fuse_swiglu:
            _ws_alloc.ptr = _ws_off_act + _ws_act_words * 2
        # fp8 combine LDS-staged epilogue: stage the L2 output tile (MB*16 rows x
        # n16pb*16 fp8 cols + per-16ch bf16 scales) in row-major LDS, then issue
        # WIDE coalesced P2P bursts per token row (one contiguous burst per row
        # instead of n16pb scattered 16-byte dword stores + scattered 2-byte
        # scale stores).  xGMI is transaction-size bound here, so coalescing is
        # what makes fp8's half-byte payload actually beat bf16.  Only viable when
        # the consumer-thread/row/dword geometry divides cleanly (see check).
        def _comb_lds_check(MB, n16pb, ncw):
            if not (enable_l2_ws and _fp8_combine):
                return False
            # TEMP toggle to A/B the LDS-staged vs scattered fp8 epilogue.  REMOVE.
            if os.environ.get("MEGA_MOE_COMB_LDS", "1") in ("0", "", "false", "False"):
                return False
            n_rows = MB * 16
            n_ct = ncw * 64
            row_dwords = n16pb * 4
            if n_rows == 0 or n_ct % n_rows != 0:
                return False
            tpr = n_ct // n_rows
            if tpr == 0 or row_dwords % tpr != 0:
                return False
            if (row_dwords // tpr) not in (1, 2, 4):
                return False
            if n16pb % 2 != 0 or (n16pb // 2) not in (1, 2, 4):
                return False
            return True
        _comb_lds_ok = _comb_lds_check(_ws_m16pb_l2, _ws_n16pb_l2,
                                       _ws_num_consumer_warps)
        _ws_comb_data_words = (_ws_m16pb_l2 * 16 * _ws_n16pb_l2 * 4) if _comb_lds_ok else 0
        _ws_comb_sc_i16 = (_ws_m16pb_l2 * 16 * _ws_n16pb_l2) if _comb_lds_ok else 0
        _ws_off_comb_data = _ws_alloc._align(_ws_alloc.ptr, 16)
        if _comb_lds_ok:
            _ws_alloc.ptr = _ws_off_comb_data + _ws_comb_data_words * 4
        _ws_off_comb_sc = _ws_alloc._align(_ws_alloc.ptr, 16)
        if _comb_lds_ok:
            _ws_alloc.ptr = _ws_off_comb_sc + _ws_comb_sc_i16 * 2
        _ws_total_bytes = _ws_alloc._align(_ws_alloc.ptr, 128)
        # gfx950: 163840 B LDS/CU.  Static per-block LDS caps co-resident
        # blocks/CU = floor(cap / total) (occupancy is also bounded by VGPR,
        # not modelled here -- this is the LDS-only upper bound).
        _CU_LDS = 163840
        _blocks_per_cu = max(1, _CU_LDS // _ws_total_bytes) if _ws_total_bytes > 0 else 0
        if os.environ.get("MEGA_MOE_DEBUG_LDS", "0") not in ("0", "", "false", "False"):
            import sys as _sys
            _warn = "  !! <2/CU" if _blocks_per_cu < 2 else ""
            print(f"[LDS] fuse={enable_fuse_swiglu} "
                  f"L1(n16pb={_ws_n16pb} m16pb={_ws_m16pb} stages={_ws_l1_stages}) "
                  f"L2(n16pb={_ws_n16pb_l2} m16pb={_ws_m16pb_l2} stages={_ws_l2_stages}) "
                  f"n16pb_max={_ws_n16pb_max} m16pb_max={_ws_m16pb_max} stages_max={_ws_stages_max} "
                  f"ring={_ws_total_bytes - _ws_act_words*2}B act={_ws_act_words*2}B "
                  f"total={_ws_total_bytes}B -> {_blocks_per_cu} block(s)/CU "
                  f"(LDS-bound, cap={_CU_LDS}){_warn}", file=_sys.stderr, flush=True)
        if _ws_total_bytes > 163840:
            raise ValueError(
                f"L1 WS LDS request {_ws_total_bytes} B exceeds gfx950 cap 163840 B; "
                f"lower num_l1_stages ({num_l1_stages}) or l1_n16_per_block ({_ws_n16pb}).")

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def mega_moe_intranode(
        addr_local_base: fx.Int64,
        addr_peer_base: fx.Int64,
        addr_cumulative_stats: fx.Int64,
        num_tokens: fx.Int32,
        addr_l1_w: fx.Int64,
        addr_l1_ws: fx.Int64,
        addr_l1_out: fx.Int64,
        addr_l2_w: fx.Int64,
        addr_l2_ws: fx.Int64,
        addr_l2_out: fx.Int64,
        addr_y: fx.Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & 63
        warp = tid >> 6
        global_warp_id = bid * num_warps_per_block + warp

        _FALSE = arith.constant(0) == arith.constant(1)
        _TRUE = arith.constant(0) == arith.constant(0)

        def _and(a, b):
            return arith.select(a, b, _FALSE)

        def _or(a, b):
            return arith.select(a, _TRUE, b)

        def _not(a):
            return arith.select(a, _FALSE, _TRUE)

        def _min(a, b):
            return arith.select(a < b, a, b)

        _r_peer = create_buffer_resource_from_addr(addr_peer_base)

        def _peer_base(pe):
            return buffer_load(_r_peer, pe, vec_width=1, dtype=T.i64())

        def _scat_base(pe):
            # Combine-scatter destination base; the local-scatter diagnostic
            # redirects it to this rank's own buffer.
            if const_expr(_probe_local_scatter):
                return addr_local_base
            return _peer_base(pe)

        # Intra-rank grid barriers (dispatch->L1, L1->L2, ... handoffs that
        # both produce and consume in THIS rank's local pool) only need
        # agent-scope coherence, not the system-scope cache flush a one-as
        # fence forces.  System-scope grid barriers were the dominant fixed
        # cost of the fused kernel (~8 of them, each w/ 2 system fences).
        # Cross-rank barriers (``_nvlink_barrier`` epilogue) keep system scope.
        # Intra-rank grid barriers only need agent-scope coherence; opt-in via
        # MEGA_MOE_GRIDBAR_SCOPE=agent.  Measured perf-neutral at 7168/2048
        # (the fused-kernel floor is register-spill bound, not fence bound),
        # so the conservative default keeps system scope.  Cross-rank barriers
        # (``_nvlink_barrier`` epilogue) always keep system scope.
        _gridbar_agent = os.environ.get("MEGA_MOE_GRIDBAR_SCOPE", "system") == "agent"

        def _grid_bar(slot_off, cross_rank=False):
            # DeepGEMM-style sense-reversal grid barrier on a dedicated
            # counter slot (``comm/barrier.cuh::grid_sync``). Block 0 adds
            # ``_FINISH_TAG - (num_sms - 1)`` and every other block adds 1,
            # so each round adds exactly ``_FINISH_TAG``: only the counter's
            # top bit toggles. Each thread-0 waits until the top bit flips
            # relative to its own fetched ``old``, so no per-call reset and
            # no meaningful overflow. The spin uses a relaxed load + a single
            # trailing acquire fence (mori-equivalent).
            _agent = const_expr(_gridbar_agent and not cross_rank)
            if _agent:
                fence_agent_release()
            else:
                fence_system_release()
            fx.barrier()
            if tid == 0:
                _caddr = addr_local_base + _c64(slot_off)
                _delta = arith.select(
                    bid == 0,
                    arith.constant(_FINISH_TAG - (num_sms - 1)),
                    arith.constant(1))
                if _agent:
                    _old = atomic_add_i32_release_agent(_caddr, _delta)
                    _newv = load_i32_relaxed_agent(_caddr)
                    while arith.andi(arith.xori(_newv, _old),
                                     arith.constant(_SIGN_BIT)) == arith.constant(0):
                        _newv = load_i32_relaxed_agent(_caddr)
                else:
                    _old = atomic_add_i32_release_sys(_caddr, _delta)
                    _newv = load_i32_relaxed_sys(_caddr)
                    while arith.andi(arith.xori(_newv, _old),
                                     arith.constant(_SIGN_BIT)) == arith.constant(0):
                        _newv = load_i32_relaxed_sys(_caddr)
            if _agent:
                fence_agent_acquire()
            else:
                fence_system_acquire()
            fx.barrier()

        def _nvlink_barrier(cnt_off, sig0_off, sig1_off, epilogue_gridsync_off=None):
            # DeepGEMM-style cross-rank NVLink sense-reversal barrier
            # (``comm/barrier.cuh::nvlink_barrier``), block 0 only.  A 2-bit
            # counter holds ``phase`` (which of the two signal slots) and
            # ``sign`` (this round adds +1 or -1); each peer gets exactly one
            # signal add, so the local signal bounces 0 <-> npes with no reset.
            # Thread 0 advances the counter and waits the alternating target.
            # All threads acquire (via the trailing release/acquire of the
            # optional epilogue grid barrier) so peers' writes are visible.
            # Single-rank (npes == 1) compiles to nothing (no peer to signal).
            if const_expr(npes > 1):
                if bid == 0:
                    _cnt_addr = addr_local_base + _c64(cnt_off)
                    _status = arith.andi(load_i32_relaxed_sys(_cnt_addr),
                                         arith.constant(3))
                    _phase = arith.andi(_status, arith.constant(1))
                    _sign = arith.shrui(_status, arith.constant(1))
                    _sig_off = arith.select(_phase == arith.constant(0),
                                            _c64(sig0_off), _c64(sig1_off))
                    _addend = arith.select(_sign == arith.constant(0),
                                           arith.constant(1), arith.constant(-1))
                    # Publish this rank's writes to peers before signalling.
                    fence_system_release()
                    if tid < npes:               # one thread per peer
                        atomic_add_i32_release_sys(_peer_base(tid) + _sig_off, _addend)
                    if tid == 0:
                        atomic_add_i32(_cnt_addr, arith.constant(1))  # advance phase/sign
                        _target = arith.select(_sign == arith.constant(0),
                                               arith.constant(npes), arith.constant(0))
                        mori_shmem.int32_wait_until_equals(
                            addr_local_base + _sig_off, _target)
                # Optional epilogue grid barrier: propagate block 0's completion
                # to every block (and acquire peers' writes) before downstream
                # reads.  Omitted at the kernel tail, where nothing follows.
                if const_expr(epilogue_gridsync_off is not None):
                    # Cross-rank: must system-acquire peers' writes here.
                    _grid_bar(epilogue_gridsync_off, cross_rank=True)

        def _reset_round():
            # DeepGEMM-style kernel-tail workspace reset
            # (``sm100_fp8_fp4_mega_moe.cuh`` :587-642).  Runs on ALL warps at a
            # phase tail, replacing the host-side per-call counter zeroing.  The
            # host reset raced with peers' Phase C cross-rank writes into our
            # ``expert_recv_count_sum`` (a fast rank's next-call Phase C add
            # could be wiped by a slow rank's host clear -> Phase D spins
            # forever -> multi-rank hang).  Here every rank clears its own
            # counters on-device after consuming them, then a cross-rank NVLink
            # barrier guarantees no rank starts the next round's cross-rank
            # writes until all ranks have cleared.
            #
            # Layout: a local grid sync (all SMs done reading counts) -> SM 0
            # clears ``expert_send_count``; the other SMs split the local
            # experts, clearing ``expert_recv_count`` / ``expert_recv_count_sum``
            # (+ ``red_add`` into ``cumulative_local_expert_recv_stats``) and,
            # when overlap is on, the per-block ``l1_arrival_count`` /
            # ``l2_arrival_mask`` -> a tail NVLink barrier.
            _i32_t = _Tt.i32
            _i64_t = _Tt.i64
            _idx_r = ir.IndexType.get()
            c_bm_r = arith.constant(block_m, index=True)
            c_bmm1_r = arith.constant(block_m - 1, index=True)
            c0_i32_r = arith.constant(0, type=_i32_t)
            c0_i64_r = arith.constant(0, type=_i64_t)

            _grid_bar(off_barrier + 40)

            if bid == 0:
                # SM 0: clear this rank's expert_send_count (full i64/expert).
                for e in range(tid, num_experts, block_threads):
                    store_i64_system(
                        addr_local_base + _c64(off_send) + _to_i64(e) * 8, c0_i64_r)
            else:
                # Other SMs split the local experts (stride num_sms-1).
                _rsum = create_buffer_resource_from_addr(addr_local_base + _c64(off_sum))
                _has_stats = addr_cumulative_stats != _c64(0)
                for i in range(bid - 1, epr, num_sms - 1):
                    num_recv = buffer_load(_rsum, i * 2, vec_width=1, dtype=_i32_t)
                    if tid == 0:
                        store_i64_system(
                            addr_local_base + _c64(off_sum) + _to_i64(i) * 8, c0_i64_r)
                        if _has_stats:
                            atomic_add_i32(
                                addr_cumulative_stats + _to_i64(i) * 4, num_recv)
                    # NOTE: ``expert_recv_count`` (per-rank) is NOT cleared: Phase
                    # C overwrites every (rank, expert) slot with a plain store
                    # each call, so stale values are harmless, and keeping them
                    # lets the host reconstruct the per-call recv counts after
                    # the kernel (the count_sum accumulator is the only one that
                    # must be zeroed, since Phase C atomic-ADDs into it).
                    if const_expr(enable_dispatch_overlap or enable_l2_readiness):
                        # Per-block arrival counters: clear this expert's blocks
                        # [pool_block_off, pool_block_off + m_blocks).
                        # (l1_arrival_count is only consumed by dispatch overlap;
                        # clearing it when only readiness/merged is on is a
                        # harmless extra store.  l2_arrival_mask MUST be cleared
                        # for the readiness / merged-scheduler L1->L2 handoff so
                        # the next call starts from 0.)
                        _i_idx = arith.index_cast(_idx_r, arith.unwrap(i))
                        pbo = arith.constant(0, index=True)
                        for e in range_constexpr(epr):
                            cnt_e = arith.index_cast(
                                _idx_r, buffer_load(_rsum, e * 2, vec_width=1, dtype=_i32_t))
                            nb_e = (cnt_e + c_bmm1_r) / c_bm_r
                            is_before = arith.constant(e, index=True) < _i_idx
                            pbo = arith.select(is_before, pbo + nb_e, pbo)
                        num_recv_idx = arith.index_cast(_idx_r, num_recv)
                        m_blocks_i = arith.index_cast(
                            _i32_t, (num_recv_idx + c_bmm1_r) / c_bm_r)
                        for b in range(tid, m_blocks_i, block_threads):
                            _blk_idx = pbo + arith.index_cast(_idx_r, arith.unwrap(b))
                            _blk_i = arith.index_cast(_i32_t, _blk_idx)
                            store_i32_system(
                                addr_local_base + _c64(off_l1arr) + _to_i64(_blk_i) * 4,
                                c0_i32_r)
                            store_i64_system(
                                addr_local_base + _c64(off_l2mask) + _to_i64(_blk_i) * 8,
                                c0_i64_r)

            # Cross-rank: wait for all ranks to finish clearing before any rank
            # exits (so the next call's cross-rank writes hit cleared counters).
            _nvlink_barrier(off_barrier + 44, off_barrier + 48, off_barrier + 52)

        def _wait_block_ready(pool_block, chosen_cnt, chosen_estart):
            # dispatch<->compute overlap: per-pool-block readiness gate that
            # REPLACES the global dispatch->L1 grid barrier (``off_barrier+4``).
            # Phase E increments ``l1_arrival_count[pool_block]`` once per token
            # pulled into the L1 pool (after a system release fence, line ~824),
            # so a consumer that observes the full per-block count + an acquire
            # fence sees all of that block's pooled token / SF writes.  Gating
            # per block (instead of one global barrier) lets a block that has
            # finished its own pull share start computing already-ready pool
            # blocks while other warps/blocks are still pulling later ones.
            #
            # Expected count for this block = min(BLOCK_M, cnt - b*BLOCK_M)
            # where ``b`` is the block index within the chosen expert; producer
            # increments exactly that many times, so we wait for value > exp-1.
            # ``l1_arrival_count`` is zeroed in-kernel at the previous call's
            # tail (``_reset_round``) behind a cross-rank barrier, so the count
            # starts from 0 (and the first call sees the construction-time zero).
            _c_bm = arith.constant(block_m, index=True)
            _b_in_e = pool_block - chosen_estart
            _rem = chosen_cnt - _b_in_e * _c_bm
            _exp = arith.select(_rem < _c_bm, _rem, _c_bm)   # min(BLOCK_M, rem)
            _exp_i32 = arith.index_cast(_Tt.i32, _exp)
            _pb_i32 = arith.index_cast(_Tt.i32, pool_block)
            _addr = addr_local_base + _c64(off_l1arr) + _to_i64(_pb_i32) * 4
            mori_shmem.int32_wait_until_greater_than(
                _addr, _exp_i32 - arith.constant(1, type=_Tt.i32))
            fence_system_acquire()

        def _ws_gemm(*, bid, warp, lm16, ld16, g16, scale_lane, a_byte_sh,
                     x_rsrc, sf_rsrc, sum_rsrc, w_rsrc, ws_rsrc, out_rsrc,
                     b_layout, _i32, _i64, _idx_t,
                     vec4_f32, vec8_i32, vec4_i32, vec2_i64, vec4_i64,
                     c16, c4, c32, c64, c2i,
                     c_hidden, c_N, c_bm, c_sn0,
                     c0_i64, c8i, c16_i32, c_ff, i32_0,
                     _c_not127, _c31, _c3, _c5, _c4_i, _c_bm_i, _c_sbm_i,
                     _n_ktiles, ws_n16pb, ws_ncw, ws_n_owned, ws_n_blocks,
                     m16pb=1, n_stages=None,
                     fused_pull=False, pull_once=False, fuse_swiglu=False,
                     combine_scatter=False,
                     l2_signal=False, l2_wait_expected=0,
                     return_executor=False,
                     probe_skip_fused_swiglu=False):
            # Generic warp-specialised grouped GEMM, shared by L1 and L2.  The
            # caller supplies the per-phase activation/weight/scale resources,
            # the K (``c_hidden``) / N (``c_N``) dims, the preshuffle B layout,
            # and the WS tiling shape (``ws_n16pb`` n16-cols per block,
            # ``ws_ncw`` consumer warps, ``ws_n_owned`` n16-tiles per consumer,
            # ``ws_n_blocks`` N-blocks).  The LDS ring / stage count are shared
            # closure state (sized to the max over phases).  Warps split into
            # loadA / loadB producers (warps [0, num_l1_load_warps)) and
            # mma+epilogue consumers; a persistent (m16, n_block) scheduler
            # hands one 16-row x (n16pb*16)-col output region to the whole
            # block, K streamed through a ``num_l1_stages`` LDS ring.  Producers
            # pre-stage the exact per-lane MFMA operands (a128 / b16 / SF) so
            # the consumer's MFMA is bit-identical to the non-WS path.
            S = num_l1_stages if n_stages is None else n_stages
            n16pb = ws_n16pb
            ncw = ws_ncw
            n_owned = ws_n_owned
            n_blocks = ws_n_blocks
            cbsz, blgp = 0, 4
            m_tiles16_max = num_max_pool_tokens // 16
            total_tiles = m_tiles16_max * n_blocks

            _base = _ws_alloc.get_base()
            A_view = SmemPtr(_base, _ws_off_a, _i32, shape=(_ws_a_words,)).get()
            Asf_view = SmemPtr(_base, _ws_off_asf, _i32, shape=(_ws_asf_words,)).get()
            B_view = SmemPtr(_base, _ws_off_b, _i32, shape=(_ws_b_words,)).get()
            Bsf_view = SmemPtr(_base, _ws_off_bsf, _i32, shape=(_ws_bsf_words,)).get()

            lane_v = scale_lane  # full lane index 0..63 (== ld16*16 + lm16)
            is_loadA = warp == 0
            is_loadB = warp == (1 if num_l1_load_warps >= 2 else 0)
            is_producer = warp < num_l1_load_warps
            is_consumer = warp >= num_l1_load_warps
            cw = (arith.index_cast(_idx_t, arith.unwrap(warp))
                  - arith.constant(num_l1_load_warps, index=True))
            c_n16pb = arith.constant(n16pb, index=True)
            c256 = arith.constant(256, index=True)
            c8idx = arith.constant(8, index=True)

            MB = m16pb

            def a_at(stage, lane_idx, ms=0):
                return arith.constant((stage * MB + ms) * 64 * 8, index=True) + lane_idx * c8idx

            def asf_at(stage, lane_idx, ms=0):
                return arith.constant((stage * MB + ms) * 64, index=True) + lane_idx

            def b_at(stage, jb_idx, lane_idx):
                x = arith.constant(stage * n16pb, index=True) + jb_idx
                return x * c256 + lane_idx * c4

            def bsf_at(stage, jb_idx, lane_idx):
                x = arith.constant(stage * n16pb, index=True) + jb_idx
                return x * c64 + lane_idx

            def _sts(view, idx, val):
                _memref_d.store(arith.unwrap(val), view, [arith.unwrap(idx)])

            def _lds_scalar(view, idx):
                return _memref_d.load(view, [arith.unwrap(idx)])

            def fill_A(stage, kt, a_row, sf_row_idx, ms=0, first_touch=None):
                base_k = arith.constant(kt * 128, index=True)
                col0 = base_k + g16
                col1 = base_k + g16 + c64
                a0_i32 = (a_row * c_hidden + col0) / c4
                a1_i32 = (a_row * c_hidden + col1) / c4
                a_word = sf_row_idx + arith.constant(kt * padded_sf, index=True)

                def _stage_a(a_c0, a_c1, a_raw):
                    # Common tail: pack the 4 i64 halves into the per-lane MFMA
                    # a128 operand, stage to the LDS ring, and extract the fp8
                    # block scale (byte selected by ld16 via ``a_byte_sh``).
                    a_c0_64 = vector.bitcast(vec2_i64, a_c0)
                    a_c1_64 = vector.bitcast(vec2_i64, a_c1)
                    a0 = vector.extract(a_c0_64, static_position=[0], dynamic_position=[])
                    a1 = vector.extract(a_c0_64, static_position=[1], dynamic_position=[])
                    a2 = vector.extract(a_c1_64, static_position=[0], dynamic_position=[])
                    a3 = vector.extract(a_c1_64, static_position=[1], dynamic_position=[])
                    a128 = vector.bitcast(vec8_i32, vector.from_elements(vec4_i64, [a0, a1, a2, a3]))
                    vector.store(a128, A_view, [a_at(stage, lane_v, ms)])
                    sa = arith.andi(arith.shrui(a_raw, a_byte_sh), c_ff)
                    _sts(Asf_view, asf_at(stage, lane_v, ms), sa)

                def _remote_read():
                    # Read this lane's A row (token fp8) + SF DIRECTLY from the
                    # peer rank via token_src_metadata.  Reading the peer token at
                    # the same column and the peer SF word ``kt`` (byte selected by
                    # ld16) reproduces bit-identical MFMA operands vs the pool copy.
                    _tsm = create_buffer_resource_from_addr(addr_local_base + _c64(off_tsm))
                    _row_i = arith.index_cast(_i32, a_row)
                    _rank = buffer_load(_tsm, _row_i * 3 + 0, vec_width=1, dtype=_i32)
                    _stok = buffer_load(_tsm, _row_i * 3 + 1, vec_width=1, dtype=_i32)
                    _tok_base = (_peer_base(_rank) + _c64(off_inp_tok)
                                 + _to_i64(_stok) * hidden)
                    _r_tok = create_buffer_resource_from_addr(_tok_base)
                    a_c0 = buffer_load(_r_tok, col0 / c4, vec_width=4, dtype=_i32)
                    a_c1 = buffer_load(_r_tok, col1 / c4, vec_width=4, dtype=_i32)
                    _sf_base = (_peer_base(_rank) + _c64(off_inp_sf)
                                + _to_i64(_stok) * (hidden // 32))
                    _r_sf = create_buffer_resource_from_addr(_sf_base)
                    a_raw = buffer_load(_r_sf, arith.constant(kt, type=_i32),
                                        vec_width=1, dtype=_i32)
                    return a_c0, a_c1, a_raw

                def _local_read():
                    a_c0 = buffer_load(x_rsrc, a0_i32, vec_width=4, dtype=_i32)
                    a_c1 = buffer_load(x_rsrc, a1_i32, vec_width=4, dtype=_i32)
                    a_raw = buffer_load(sf_rsrc, a_word, vec_width=1, dtype=_i32)
                    return a_c0, a_c1, a_raw

                if const_expr(fused_pull and pull_once and (first_touch is not None)):
                    # "Pull once" write-through: on the FIRST chunk pull from the
                    # peer AND mirror the verbatim token / SF word into the local
                    # L1 pool (the column/word the local-read path expects); later
                    # chunks reuse the cheap HBM pool copy instead of re-pulling the
                    # same token N_blocks times over the slow interconnect.  The
                    # stored SF word is the raw peer word ``kt`` (same shift recovers
                    # the scale), so the reuse is bit-identical to the remote path.
                    if first_touch:
                        a_c0, a_c1, a_raw = _remote_read()
                        buffer_store(a_c0, x_rsrc, a0_i32)
                        buffer_store(a_c1, x_rsrc, a1_i32)
                        buffer_store(a_raw, sf_rsrc, a_word)
                        _stage_a(a_c0, a_c1, a_raw)
                    else:
                        a_c0, a_c1, a_raw = _local_read()
                        _stage_a(a_c0, a_c1, a_raw)
                elif const_expr(fused_pull):
                    a_c0, a_c1, a_raw = _remote_read()
                    _stage_a(a_c0, a_c1, a_raw)
                else:
                    a_c0, a_c1, a_raw = _local_read()
                    _stage_a(a_c0, a_c1, a_raw)

            def fill_B(stage, kt, brow_expert_base, n_block_v):
                b_k0 = arith.constant(kt, type=_i32)
                b_k1 = arith.index_cast(_i32, ld16)
                k_off = arith.constant((kt // 2) * 64, index=True)
                for jb in range_constexpr(n16pb):
                    jb_c = arith.constant(jb, index=True)
                    n16_abs = n_block_v * c_n16pb + jb_c
                    n_col_base = n16_abs * c16
                    b_row = brow_expert_base + n_col_base + lm16
                    b_n_blk = arith.index_cast(_i32, b_row / c16)
                    b_n_intra = arith.index_cast(_i32, b_row % c16)
                    b_row_tile = brow_expert_base + n_col_base
                    mni_b = b_row_tile / c32
                    n_half = arith.index_cast(_i32, (b_row_tile / c16) % c2i)
                    b_sh = n_half * c8i
                    b_sh_hi = b_sh + c16_i32
                    b_scale_base = mni_b * c_sn0 + scale_lane
                    b_idx = crd2idx([b_n_blk, b_k0, b_k1, b_n_intra, i32_0], b_layout)
                    b16 = _buffer_load_vec(
                        buffer_ops, vector, w_rsrc, b_idx,
                        elem_type=_Tt.i8, vec_elems=16, elem_bytes=1, offset_in_bytes=True,
                    )
                    b16_i32 = vector.bitcast(vec4_i32, b16)
                    vector.store(b16_i32, B_view, [b_at(stage, jb_c, lane_v)])
                    sb_raw = buffer_load(ws_rsrc, b_scale_base + k_off, vec_width=1, dtype=_i32)
                    b_k0s = arith.andi(arith.shrui(sb_raw, b_sh), c_ff)
                    b_k1s = arith.andi(arith.shrui(sb_raw, b_sh_hi), c_ff)
                    sb = arith.ori(b_k0s, arith.shli(b_k1s, c8i))
                    _sts(Bsf_view, bsf_at(stage, jb_c, lane_v), sb)

            # True warp specialisation: producers ONLY stage the LDS ring and
            # consumers ONLY issue MFMA.  Because the MFMA now lives entirely in
            # the consumer region, the accumulators never cross the
            # producer/consumer split, so there is no SSA carry to fight.
            #
            # Block sync uses the raw ``s_barrier`` idiom from
            # ``flash_attn_gfx950`` (``sched_barrier(0)`` scheduler fence +
            # ``s_waitcnt lgkmcnt(0)`` LDS drain + ``s_barrier``) rather than
            # ``fx.barrier()``.  On gfx9 ``s_barrier`` is a workgroup-wide count:
            # producer- and consumer-region barriers pair up as long as every
            # wave executes the same total (here ``1 + _n_ktiles``), which they
            # do -- so divergent placement across the warp roles is safe.
            def _lds_bar():
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
                rocdl.s_barrier()

            def consume(accs, cur, kt):
                opsel_b = kt % 2
                a128 = vector.load_op(vec8_i32, A_view, [a_at(cur, lane_v)])
                sa = _lds_scalar(Asf_view, asf_at(cur, lane_v))
                out = []
                for t in range_constexpr(n_owned):
                    j = cw + arith.constant(t * ncw, index=True)
                    b16_i32 = vector.load_op(vec4_i32, B_view, [b_at(cur, j, lane_v)])
                    b16_64 = vector.bitcast(vec2_i64, b16_i32)
                    gb0 = vector.extract(b16_64, static_position=[0], dynamic_position=[])
                    gb1 = vector.extract(b16_64, static_position=[1], dynamic_position=[])
                    b128 = vector.bitcast(vec8_i32, vector.from_elements(vec4_i64, [gb0, gb1, c0_i64, c0_i64]))
                    sb = _lds_scalar(Bsf_view, bsf_at(cur, j, lane_v))
                    out.append(rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        vec4_f32, [a128, b128, accs[t], cbsz, blgp, 0, sa, opsel_b, sb]))
                return out

            def consume_mb(accs2d, cur, kt):
                # M-batched consume: load each owned B tile ONCE and reuse it
                # across all MB A subtiles (B-reuse = MB*16 rows deep).
                opsel_b = kt % 2
                a128s = []
                sa_s = []
                for ms in range_constexpr(MB):
                    a128s.append(vector.load_op(vec8_i32, A_view, [a_at(cur, lane_v, ms)]))
                    sa_s.append(_lds_scalar(Asf_view, asf_at(cur, lane_v, ms)))
                out = [[None] * n_owned for _ in range_constexpr(MB)]
                for t in range_constexpr(n_owned):
                    j = cw + arith.constant(t * ncw, index=True)
                    b16_i32 = vector.load_op(vec4_i32, B_view, [b_at(cur, j, lane_v)])
                    b16_64 = vector.bitcast(vec2_i64, b16_i32)
                    gb0 = vector.extract(b16_64, static_position=[0], dynamic_position=[])
                    gb1 = vector.extract(b16_64, static_position=[1], dynamic_position=[])
                    b128 = vector.bitcast(vec8_i32, vector.from_elements(vec4_i64, [gb0, gb1, c0_i64, c0_i64]))
                    sb = _lds_scalar(Bsf_view, bsf_at(cur, j, lane_v))
                    for ms in range_constexpr(MB):
                        out[ms][t] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            vec4_f32, [a128s[ms], b128, accs2d[ms][t], cbsz, blgp, 0, sa_s[ms], opsel_b, sb])
                return out

            if const_expr(fuse_swiglu):
                # ===== Fused L1 epilogue: SwiGLU + mxfp8 quant (no f32 global
                # round-trip, no grid barrier #3). =====
                # PER-CHUNK pairing: gate N-block c is paired with up N-block
                # c+n_half (n_half == gate-block count).  For each chunk we run
                # TWO pipelined MMA passes (gate then up) and de-swizzle each
                # consumer's accumulators into a SMALL ``Act_view``
                # [16 rows x 2*(n16pb*16)] bf16 buffer -- gate in the low half,
                # up in the high half.  One block barrier later, a SwiGLU +
                # mxfp8 quant sweep over just this chunk's channels reads
                # gate[ch]/up[ch] pairs from LDS and writes straight to the L2
                # fp8 pool (off_l2_tok) + M-major SF (off_l2_sf), then a barrier
                # frees the buffer for the next chunk.  Keeping the staging
                # per-chunk + bf16 is what holds ring+staging <=80KB so the
                # kernel keeps 2 blocks/CU (a full-row 64KB stage -> 1/CU).
                _ih = intermediate_hidden
                _f32 = _Tt.f32
                _i16t = _Tt.i16
                chunk_cols = n16pb * 16
                chunk_stride = 2 * chunk_cols
                sfw_per_chunk = chunk_cols // 128
                n_half_blk = n_blocks // 2
                Act_view = SmemPtr(_base, _ws_off_act, _i16t,
                                   shape=(_ws_act_words,)).get()

                neg_log2e = arith.constant(-1.4426950408889634, type=_f32)
                c1_f32 = arith.constant(1.0, type=_f32)
                c_inv448 = arith.constant(1.0 / 448.0, type=_f32)
                c_448 = arith.constant(448.0, type=_f32)
                c_neg448 = arith.constant(-448.0, type=_f32)
                c0_i32q = arith.constant(0, type=_i32)
                c1q = arith.constant(1, type=_i32)
                c16q = arith.constant(16, type=_i32)
                c23q = arith.constant(23, type=_i32)
                c_7fffq = arith.constant(0x7FFF, type=_i32)
                c_0xffffq = arith.constant(0xFFFF, type=_i32)
                c_0x400000 = arith.constant(0x400000, type=_i32)
                c_0xFF800000 = arith.constant(0xFF800000 & 0xFFFFFFFF, type=_i32)
                c254q = arith.constant(254, type=_i32)
                c_padq = arith.constant(padded_sf, type=_i32)
                c_ih_i = arith.constant(_ih, index=True)
                c_chunk_stride = arith.constant(chunk_stride, index=True)
                c_chunk_cols = arith.constant(chunk_cols, index=True)
                c128i = arith.constant(128, index=True)
                c_sfwpc = arith.constant(sfw_per_chunk, index=True)
                c_n16pb_i = arith.constant(n16pb, index=True)
                c1q_idx = arith.constant(1, index=True)
                c2q_idx = arith.constant(2, index=True)
                c3q_idx = arith.constant(3, index=True)

                def _scaledq(av_j, recip):
                    s = arith.mulf(av_j, recip)
                    s = arith.minimumf(s, c_448)
                    return arith.maximumf(s, c_neg448)

                def _f32_to_bf16(vf):
                    # round-to-nearest-even f32 -> bf16 (i16).
                    vi32 = vf.bitcast(_i32)
                    lsb = arith.andi(arith.shrui(vi32, c16q), c1q)
                    rounded = arith.addi(arith.addi(vi32, c_7fffq), lsb)
                    return arith.trunci(_i16t, arith.shrui(rounded, c16q))

                def _bf16_to_f32(b16):
                    bi = arith.shli(arith.extui(_i32, b16), c16q)
                    return bi.bitcast(_f32)

                wts_rsrc = create_buffer_resource_from_addr(
                    addr_local_base + _c64(off_l1_wts))
                _l2tok_bytes = num_max_pool_tokens * _ih
                l2tok_rsrc = create_buffer_resource_from_addr(
                    addr_local_base + _c64(off_l2_tok), num_records_bytes=_l2tok_bytes)

                m_super_tiles = m_tiles16_max // MB
                c_mb16 = arith.constant(MB * 16, index=True)

                # Per-tile executor for the fused-SwiGLU L1 path: one work item
                # = one m_super super-tile (all N handled internally via the
                # chunk/gate-up pairing).  Hoisted out of the persistent loop so
                # the merged scheduler can drive it directly; behaviour matches
                # the inlined loop body.
                def _fused_tile(m_super):
                    m_row_base = m_super * c_mb16
                    pool_block = m_row_base / c_bm

                    blocks = arith.constant(0, index=True)
                    chosen_e = arith.constant(0)
                    chosen_cnt = arith.constant(0, index=True)
                    chosen_estart = arith.constant(0, index=True)
                    found = arith.constant(0) == arith.constant(1)
                    for e in range_constexpr(epr):
                        cnt = arith.index_cast(_idx_t, buffer_load(sum_rsrc, e * 2, vec_width=1, dtype=_i32))
                        nb = (cnt + arith.constant(block_m - 1, index=True)) / c_bm
                        here = arith.select(found, arith.constant(0) == arith.constant(1),
                                            arith.andi(pool_block >= blocks, pool_block < blocks + nb))
                        chosen_e = arith.select(here, arith.constant(e), chosen_e)
                        chosen_cnt = arith.select(here, cnt, chosen_cnt)
                        chosen_estart = arith.select(here, blocks, chosen_estart)
                        found = arith.select(here, arith.constant(0) == arith.constant(0), found)
                        blocks = blocks + nb

                    if found:
                        if const_expr(enable_dispatch_overlap):
                            _wait_block_ready(pool_block, chosen_cnt, chosen_estart)
                        expert_idx = arith.index_cast(_idx_t, chosen_e)
                        brow_expert_base = expert_idx * c_N
                        a_rows = []
                        sf_row_idxs = []
                        for ms in range_constexpr(MB):
                            a_row = m_row_base + arith.constant(ms * 16, index=True) + lm16
                            a_row_i = arith.index_cast(_i32, a_row)
                            pbo = arith.divui(a_row_i, _c_bm_i)
                            idx_in = arith.remui(a_row_i, _c_bm_i)
                            sf_within = arith.addi(
                                arith.addi(arith.andi(idx_in, _c_not127),
                                           arith.muli(arith.andi(idx_in, _c31), _c4_i)),
                                arith.andi(arith.shrui(idx_in, _c5), _c3))
                            sf_row = arith.addi(arith.muli(pbo, _c_sbm_i), sf_within)
                            a_rows.append(a_row)
                            sf_row_idxs.append(arith.index_cast(_idx_t, sf_row))

                        # ---- per-chunk: 2 pipelined MMA passes (gate, up) ----
                        # ROLLED chunk loop (scf.for): emit the chunk body ONCE
                        # instead of unrolling n_half_blk copies -- keeps the
                        # (already-huge, x MB) fused GEMM IR compilable at the
                        # 7168 shape.  Side-effect only (accs recomputed per
                        # chunk, Act staged to LDS), so no scf iter_args; the
                        # inner K-loop stays unrolled so MFMA opsel_b stays an
                        # immediate.
                        for c_chunk in range(n_half_blk):
                            c_chunk_idx = arith.index_cast(_idx_t, arith.unwrap(c_chunk))
                            # "Pull once": chunk 0 (both halves) pulls from the peer
                            # AND mirrors into the local pool; chunks >0 read the
                            # pool.  Predicate is uniform (c_chunk is the loop
                            # induction), so the divergent loads / drain are safe.
                            first_touch_pred = (
                                (c_chunk_idx == arith.constant(0, index=True))
                                if const_expr(fused_pull and pull_once) else None)
                            for half in range_constexpr(2):
                                n_block_v = c_chunk_idx + arith.constant(half * n_half_blk, index=True)
                                c_half_off = arith.constant(half * chunk_cols, index=True)
                                if is_producer:
                                    if is_loadA:
                                        for ms in range_constexpr(MB):
                                            fill_A(0, 0, a_rows[ms], sf_row_idxs[ms], ms,
                                                   first_touch=first_touch_pred)
                                    if is_loadB:
                                        fill_B(0, 0, brow_expert_base, n_block_v)
                                    _lds_bar()
                                    for kt in range_constexpr(_n_ktiles):
                                        if const_expr(kt + 1 < _n_ktiles):
                                            nxt = (kt + 1) % S
                                            if is_loadA:
                                                for ms in range_constexpr(MB):
                                                    fill_A(nxt, kt + 1, a_rows[ms], sf_row_idxs[ms], ms,
                                                           first_touch=first_touch_pred)
                                            if is_loadB:
                                                fill_B(nxt, kt + 1, brow_expert_base, n_block_v)
                                        _lds_bar()
                                    if const_expr(fused_pull and pull_once):
                                        if is_loadA:
                                            # Drain the chunk-0 pool stores so the
                                            # SAME wave's later-chunk pool loads
                                            # observe them (global store->load order).
                                            if first_touch_pred:
                                                rocdl.s_waitcnt(_VMCNT_0_ONLY)
                                if is_consumer:
                                    accs = [[arith.constant_vector(0.0, vec4_f32)
                                             for _t in range_constexpr(n_owned)]
                                            for _ms in range_constexpr(MB)]
                                    _lds_bar()
                                    for kt in range_constexpr(_n_ktiles):
                                        cur = kt % S
                                        accs = consume_mb(accs, cur, kt)
                                        _lds_bar()
                                    # de-swizzle accs -> Act half [MB*16 x chunk_cols] bf16
                                    for ms in range_constexpr(MB):
                                        for t in range_constexpr(n_owned):
                                            j = cw + arith.constant(t * ncw, index=True)
                                            col = c_half_off + j * c16 + lm16
                                            for ii in range_constexpr(4):
                                                row = (arith.constant(ms * 16, index=True)
                                                       + ld16 * c4 + arith.constant(ii, index=True))
                                                val = vector.extract(accs[ms][t], static_position=[ii], dynamic_position=[])
                                                _sts(Act_view, row * c_chunk_stride + col, _f32_to_bf16(val))
                            # ---- chunk staged: one block barrier ----
                            _lds_bar()
                            # ---- SwiGLU + mxfp8 quant sweep over THIS chunk ----
                            c_chunk_v = c_chunk_idx
                            _fused_units = 0 if probe_skip_fused_swiglu else (MB * 16 * sfw_per_chunk)
                            for u in range(tid, _fused_units, block_threads):
                                u_i = arith.index_cast(_idx_t, arith.unwrap(u))
                                lrow = u_i / c_sfwpc
                                kw_local = u_i % c_sfwpc
                                kw = c_chunk_v * c_sfwpc + kw_local
                                pool_row = m_row_base + lrow
                                w_val = buffer_load(wts_rsrc,
                                                    arith.index_cast(_i32, pool_row),
                                                    vec_width=1, dtype=_f32)
                                lds_row_base = lrow * c_chunk_stride
                                lds_col_base = kw_local * c128i
                                out_byte_row = pool_row * c_ih_i
                                packed_sf = c0_i32q

                                def _act_at(ic):
                                    # SwiGLU(gate)*up*topk_w for one channel, read
                                    # fresh from LDS.  Recomputed (not cached in
                                    # regs) to keep the sweep's live VGPR set tiny
                                    # -- mxfp8 needs amax over 32 ch BEFORE scaling,
                                    # and holding 32 f32 live halved occupancy for
                                    # the WHOLE kernel (slowing the L2 GEMM phase).
                                    gv = _bf16_to_f32(_lds_scalar(Act_view, lds_row_base + ic))
                                    uv = _bf16_to_f32(_lds_scalar(Act_view, lds_row_base + c_chunk_cols + ic))
                                    tt = arith.mulf(gv, neg_log2e)
                                    emu = _llvm_d.call_intrinsic(_f32, "llvm.amdgcn.exp2.f32", [tt], [], [])
                                    den = arith.addf(c1_f32, emu)
                                    sig = _llvm_d.call_intrinsic(_f32, "llvm.amdgcn.rcp.f32", [den], [], [])
                                    return arith.mulf(arith.mulf(arith.mulf(gv, sig), uv), w_val)

                                for gsub in range_constexpr(4):
                                    gcol0 = lds_col_base + arith.constant(gsub * 32, index=True)
                                    # pass 1: amax over the 32-ch group (no act kept live)
                                    amax = arith.constant(0.0, type=_f32)
                                    for ch in range_constexpr(32):
                                        ic = gcol0 + arith.constant(ch, index=True)
                                        av = _llvm_d.call_intrinsic(_f32, "llvm.fabs.f32", [_act_at(ic)], [], [])
                                        amax = arith.maximumf(amax, av)
                                    y = arith.mulf(amax, c_inv448).bitcast(_i32)
                                    e8m0 = arith.shrui(arith.andi(arith.addi(y, c_0x400000), c_0xFF800000), c23q)
                                    recip = arith.shli(arith.subi(c254q, e8m0), c23q).bitcast(_f32)
                                    out_byte_base = out_byte_row + kw * c128i + arith.constant(gsub * 32, index=True)
                                    # pass 2: recompute 4 acts at a time, scale + pack
                                    for wwd in range_constexpr(8):
                                        b4 = gcol0 + arith.constant(wwd * 4, index=True)
                                        pk = c0_i32q
                                        pk = rocdl.cvt_pk_fp8_f32(
                                            _i32, _scaledq(_act_at(b4), recip),
                                            _scaledq(_act_at(b4 + c1q_idx), recip), pk, 0)
                                        pk = rocdl.cvt_pk_fp8_f32(
                                            _i32, _scaledq(_act_at(b4 + c2q_idx), recip),
                                            _scaledq(_act_at(b4 + c3q_idx), recip), pk, 1)
                                        word_off = out_byte_base + arith.constant(wwd * 4, index=True)
                                        buffer_store(pk, l2tok_rsrc, word_off, offset_is_bytes=True)
                                    packed_sf = arith.ori(
                                        packed_sf, arith.shli(e8m0, arith.constant(gsub * 8, type=_i32)))
                                # M-major SF word (row permutation within block; kw K-word).
                                row_iq = arith.index_cast(_i32, pool_row)
                                pboq = arith.divui(row_iq, _c_bm_i)
                                idx_inq = arith.remui(row_iq, _c_bm_i)
                                sf_withinq = arith.addi(
                                    arith.addi(arith.andi(idx_inq, _c_not127),
                                               arith.muli(arith.andi(idx_inq, _c31), _c4_i)),
                                    arith.andi(arith.shrui(idx_inq, _c5), _c3))
                                sf_rowq = arith.addi(arith.muli(pboq, _c_sbm_i), sf_withinq)
                                kw_iq = arith.index_cast(_i32, kw)
                                wordidxq = arith.addi(sf_rowq, arith.muli(kw_iq, c_padq))
                                l2sf_rsrc_i = create_buffer_resource_from_addr(
                                    addr_local_base + _c64(off_l2_sf))
                                buffer_store(packed_sf, l2sf_rsrc_i,
                                             arith.index_cast(_idx_t, arith.muli(wordidxq, _c4_i)),
                                             offset_is_bytes=True)
                            # ---- block barrier before reusing the chunk buffer ----
                            _lds_bar()

                        if const_expr(l2_signal):
                            # Publish this m_super's L2-pool writes to the system
                            # coherence point, then bump the per-pool-block
                            # readiness counter once.  An L2 tile for this expert
                            # waits for ``block_m // (MB*16)`` such bumps (one per
                            # m_super sub-block of the pool block) before reading,
                            # so the global L1->L2 grid barrier can be dropped.
                            fence_system_release()
                            _lds_bar()
                            if tid == 0:
                                _pb_i32 = arith.index_cast(_i32, pool_block)
                                atomic_add_i32(
                                    addr_local_base + _c64(off_l2mask) + _to_i64(_pb_i32) * 8,
                                    arith.constant(1, type=_i32))

                # Merged-loop mode: hand the per-tile executor back so the
                # caller's unified scheduler can drive it (plan todo 3c).
                if const_expr(return_executor):
                    return _fused_tile
                for it_m in range(bid, m_super_tiles, num_sms):
                    _fused_tile(arith.index_cast(_idx_t, arith.unwrap(it_m)))
                return

            c_nblk = arith.constant(n_blocks, index=True)
            # M-batched scheduler: each super-tile spans MB*16 M-rows (one B-ring
            # residency reused across all of them).  MB*16 divides block_m so all
            # subtiles map to the same expert / pool_block.
            m_super_tiles = m_tiles16_max // MB
            total_super = m_super_tiles * n_blocks
            c_mb16 = arith.constant(MB * 16, index=True)
            # Per-tile executor (one work item = one (m_super, n_block) tile).
            # Hoisted out of the persistent loop so the merged scheduler (plan
            # todo restructure-body) can drive it directly; behaviour is
            # identical to the inlined loop body.
            def _nf_tile(it_i):
                m_super = it_i / c_nblk
                n_block_v = it_i % c_nblk
                m_row_base = m_super * c_mb16
                pool_block = m_row_base / c_bm

                blocks = arith.constant(0, index=True)
                chosen_e = arith.constant(0)
                chosen_cnt = arith.constant(0, index=True)
                chosen_estart = arith.constant(0, index=True)
                found = arith.constant(0) == arith.constant(1)
                for e in range_constexpr(epr):
                    cnt = arith.index_cast(_idx_t, buffer_load(sum_rsrc, e * 2, vec_width=1, dtype=_i32))
                    nb = (cnt + arith.constant(block_m - 1, index=True)) / c_bm
                    here = arith.select(found, arith.constant(0) == arith.constant(1),
                                        arith.andi(pool_block >= blocks, pool_block < blocks + nb))
                    chosen_e = arith.select(here, arith.constant(e), chosen_e)
                    chosen_cnt = arith.select(here, cnt, chosen_cnt)
                    chosen_estart = arith.select(here, blocks, chosen_estart)
                    found = arith.select(here, arith.constant(0) == arith.constant(0), found)
                    blocks = blocks + nb

                if found:
                    if const_expr(enable_dispatch_overlap):
                        _wait_block_ready(pool_block, chosen_cnt, chosen_estart)
                    if const_expr(l2_wait_expected > 0):
                        # L2 readiness gate: spin until the fused L1 epilogue has
                        # signalled all ``l2_wait_expected`` m_super sub-blocks of
                        # this pool block, then acquire so the pooled fp8/SF
                        # writes are visible.  Replaces the global grid barrier.
                        _pb_i32 = arith.index_cast(_i32, pool_block)
                        mori_shmem.int32_wait_until_greater_than(
                            addr_local_base + _c64(off_l2mask) + _to_i64(_pb_i32) * 8,
                            arith.constant(l2_wait_expected - 1, type=_i32))
                        fence_system_acquire()
                    expert_idx = arith.index_cast(_idx_t, chosen_e)
                    brow_expert_base = expert_idx * c_N
                    # Per-subtile A row + M-major SF row (all in one pool_block).
                    a_rows = []
                    sf_row_idxs = []
                    for ms in range_constexpr(MB):
                        a_row = m_row_base + arith.constant(ms * 16, index=True) + lm16
                        a_row_i = arith.index_cast(_i32, a_row)
                        pbo = arith.divui(a_row_i, _c_bm_i)
                        idx_in = arith.remui(a_row_i, _c_bm_i)
                        sf_within = arith.addi(
                            arith.addi(arith.andi(idx_in, _c_not127),
                                       arith.muli(arith.andi(idx_in, _c31), _c4_i)),
                            arith.andi(arith.shrui(idx_in, _c5), _c3))
                        sf_row = arith.addi(arith.muli(pbo, _c_sbm_i), sf_within)
                        a_rows.append(a_row)
                        sf_row_idxs.append(arith.index_cast(_idx_t, sf_row))

                    # --- Producer region: stage the K ring only (no MFMA). ---
                    # Prologue stages K-step 0, then each loop step stages the
                    # next K-step (one ahead) into the ring.  Barrier count:
                    # 1 (prologue) + _n_ktiles (loop) == 1 + _n_ktiles.
                    if is_producer:
                        if is_loadA:
                            for ms in range_constexpr(MB):
                                fill_A(0, 0, a_rows[ms], sf_row_idxs[ms], ms)
                        if is_loadB:
                            fill_B(0, 0, brow_expert_base, n_block_v)
                        _lds_bar()
                        for kt in range_constexpr(_n_ktiles):
                            if const_expr(kt + 1 < _n_ktiles):
                                nxt = (kt + 1) % S
                                if is_loadA:
                                    for ms in range_constexpr(MB):
                                        fill_A(nxt, kt + 1, a_rows[ms], sf_row_idxs[ms], ms)
                                if is_loadB:
                                    fill_B(nxt, kt + 1, brow_expert_base, n_block_v)
                            _lds_bar()
                        # Balanced partner for the consumer's fp8-combine LDS
                        # staging barrier (workgroup ``s_barrier`` pairs by COUNT;
                        # every wave must execute the same total).  Gated by the
                        # same constexpr as the consumer side so counts stay equal.
                        if const_expr(combine_scatter and _fp8_combine and _comb_lds_ok):
                            _lds_bar()

                    # --- Consumer region: MFMA from the ring + epilogue. ---
                    # Matching barrier count: 1 (prologue, wait stage 0) +
                    # _n_ktiles (one per consumed K-step).
                    if is_consumer:
                        accs = [[arith.constant_vector(0.0, vec4_f32)
                                 for _t in range_constexpr(n_owned)]
                                for _ms in range_constexpr(MB)]
                        _lds_bar()
                        for kt in range_constexpr(_n_ktiles):
                            cur = kt % S
                            accs = consume_mb(accs, cur, kt)
                            _lds_bar()

                        if const_expr(combine_scatter):
                            # ---- TIER-1: fused combine scatter ----
                            # Instead of writing the f32 L2 output to the global
                            # ``l2_out`` pool (later re-read + converted + P2P
                            # scattered by the standalone combine phase), the L2
                            # consumer epilogue converts each output element to
                            # bf16 and P2P-stores it DIRECTLY into the source
                            # rank's combine staging slot.  Removes the f32 l2_out
                            # round-trip AND the L2->combine grid barrier; the
                            # cross-rank NVLink barrier + reduce still follow.
                            c1_i32q = arith.constant(1, type=_i32)
                            c7fff_i32q = arith.constant(0x7FFF, type=_i32)
                            _i16w = _Tt.i16
                            _tsm_sc = create_buffer_resource_from_addr(
                                addr_local_base + _c64(off_tsm))
                            c_rowbytes_sc = _c64(hidden * 2)
                            c_slabbytes_sc = _c64(m * hidden * 2)
                            # exclusive upper bound of valid pool rows for this expert
                            valid_hi = chosen_estart * c_bm + chosen_cnt

                            def _bf16_i16(valf):
                                vi = valf.bitcast(_i32)
                                lsb = arith.andi(arith.shrui(vi, c16_i32), c1_i32q)
                                rnd = arith.addi(arith.addi(vi, c7fff_i32q), lsb)
                                return arith.trunci(_i16w, arith.shrui(rnd, c16_i32))

                            if const_expr(_fp8_combine):
                                # ---- fp8 combine: e4m3 data + per-16ch bf16 scale ----
                                # Each n16 block (16 lm16 lanes of one token row)
                                # shares a scale = absmax/448 (DPP butterfly over
                                # the 16-lane DPP row).  Adjacent groups of 4 lanes
                                # pack their e4m3 bytes into one dword (DPP gather).
                                _f32q8 = _Tt.f32
                                _c448q8 = arith.constant(448.0, type=_f32q8)
                                _cn448q8 = arith.constant(-448.0, type=_f32q8)
                                _cinv448q8 = arith.constant(1.0 / 448.0, type=_f32q8)
                                _c0fq8 = arith.constant(0.0, type=_f32q8)
                                _c0iq8 = arith.constant(0, type=_i32)
                                _c3iq8 = arith.constant(3, type=_i32)
                                lm16_i = arith.index_cast(_i32, lm16)
                                lm4 = arith.andi(lm16_i, _c3iq8)
                                c_fp8_row = _c64(_comb_fp8_row)
                                c_fp8_slab = _c64(_comb_fp8_slab)
                                c_sc_row = _c64(_comb_sc_row)
                                c_sc_slab = _c64(_comb_sc_slab)

                                def _quant_fp8(_vq8):
                                    # per-16ch absmax butterfly -> e4m3 scaled byte +
                                    # bf16 scale; pack 4 adjacent lanes into one dword.
                                    _afq8 = arith.maximumf(_vq8, _c0fq8 - _vq8)
                                    _afq8 = arith.maximumf(_afq8, dpp_xor_f32(_afq8, 1))
                                    _afq8 = arith.maximumf(_afq8, dpp_xor_f32(_afq8, 2))
                                    _afq8 = arith.maximumf(_afq8, dpp_xor_f32(_afq8, 4))
                                    _afq8 = arith.maximumf(_afq8, dpp_xor_f32(_afq8, 8))
                                    _rcq8 = (_afq8 > _c0fq8).select(_c448q8 / _afq8, _c0fq8)
                                    _scq8 = _afq8 * _cinv448q8
                                    _sdq8 = arith.minimumf(_vq8 * _rcq8, _c448q8)
                                    _sdq8 = arith.maximumf(_sdq8, _cn448q8)
                                    _s1q8 = dpp_xor_f32(_sdq8, 1)
                                    _s2q8 = dpp_xor_f32(_sdq8, 2)
                                    _s3q8 = dpp_xor_f32(_s2q8, 1)
                                    _pk = rocdl.cvt_pk_fp8_f32(_i32, _sdq8, _s1q8, _c0iq8, 0)
                                    _pk = rocdl.cvt_pk_fp8_f32(_i32, _s2q8, _s3q8, _pk, 1)
                                    return _pk, _scq8

                                if const_expr(_comb_lds_ok):
                                    # LDS-staged coalesced epilogue: quantise the
                                    # whole tile into row-major LDS, then issue ONE
                                    # wide contiguous P2P burst per token row (data)
                                    # + one coalesced scale burst.  Replaces the
                                    # scattered 16-byte dword + 2-byte scale stores
                                    # that made fp8 transaction-bound on xGMI.
                                    _cd_view = SmemPtr(_base, _ws_off_comb_data, _i32,
                                                       shape=(_ws_comb_data_words,)).get()
                                    _cs_view_w = SmemPtr(_base, _ws_off_comb_sc, _i16w,
                                                         shape=(_ws_comb_sc_i16,)).get()
                                    _cs_view_r = SmemPtr(_base, _ws_off_comb_sc, _i32,
                                                         shape=(_ws_comb_sc_i16 // 2,)).get()
                                    _rowdw = n16pb * 4
                                    _c_rowdw = arith.constant(_rowdw, index=True)
                                    _dlane = lm16 / c4   # dword within n16 block (0..3)
                                    # ---- Phase 1: quantise accs -> LDS tile ----
                                    for ms in range_constexpr(MB):
                                        for t in range_constexpr(n_owned):
                                            _jL = cw + arith.constant(t * ncw, index=True)
                                            for ii in range_constexpr(4):
                                                _trow = (arith.constant(ms * 16, index=True)
                                                         + ld16 * c4 + arith.constant(ii, index=True))
                                                _vq8 = vector.extract(accs[ms][t],
                                                                      static_position=[ii], dynamic_position=[])
                                                _pkq8, _scq8 = _quant_fp8(_vq8)
                                                if lm4 == _c0iq8:
                                                    _didx = _trow * _c_rowdw + _jL * c4 + _dlane
                                                    _sts(_cd_view, _didx, _pkq8)
                                                if lm16_i == _c0iq8:
                                                    _sidx = _trow * c_n16pb + _jL
                                                    _sts(_cs_view_w, _sidx, _bf16_i16(_scq8))
                                    # ---- barrier: publish LDS tile to all warps ----
                                    _lds_bar()
                                    # ---- Phase 2: coalesced P2P bursts per row ----
                                    _n_rows = MB * 16
                                    _n_ct = ncw * 64
                                    _tpr = _n_ct // _n_rows
                                    _dpt = _rowdw // _tpr
                                    _sc_w = n16pb // 2
                                    _c_tpr = arith.constant(_tpr, index=True)
                                    _c_dpt = arith.constant(_dpt, index=True)
                                    _tidc = cw * c64 + lane_v
                                    _myrow = _tidc / _c_tpr
                                    _sub = _tidc % _c_tpr
                                    _prow = m_row_base + _myrow
                                    if _prow < valid_hi:
                                        _pp = arith.index_cast(_i32, _prow)
                                        _r2 = buffer_load(_tsm_sc, _pp * 3 + 0, vec_width=1, dtype=_i32)
                                        _t2 = buffer_load(_tsm_sc, _pp * 3 + 1, vec_width=1, dtype=_i32)
                                        _k2 = buffer_load(_tsm_sc, _pp * 3 + 2, vec_width=1, dtype=_i32)
                                        _nb = arith.index_cast(_i32, n_block_v)
                                        _dbase = (_scat_base(_r2) + _c64(off_combine)
                                                  + _to_i64(_k2) * c_fp8_slab
                                                  + _to_i64(_t2) * c_fp8_row
                                                  + _to_i64(_nb) * _c64(n16pb * 16))
                                        _dr = create_buffer_resource_from_addr(_dbase)
                                        _ldoff = _myrow * _c_rowdw + _sub * _c_dpt
                                        _vd = vector.load_op(_Tt.vec(_dpt, _i32), _cd_view, [_ldoff])
                                        _doff = arith.index_cast(_i32, _sub * _c_dpt)
                                        buffer_store(_vd, _dr, _doff)
                                        if _sub == arith.constant(0, index=True):
                                            _sbase = (_scat_base(_r2) + _c64(off_combine_scale)
                                                      + _to_i64(_k2) * c_sc_slab
                                                      + _to_i64(_t2) * c_sc_row
                                                      + _to_i64(_nb) * _c64(n16pb * 2))
                                            _sr = create_buffer_resource_from_addr(_sbase)
                                            _scoff = _myrow * arith.constant(_sc_w, index=True)
                                            _vs = vector.load_op(_Tt.vec(_sc_w, _i32), _cs_view_r, [_scoff])
                                            buffer_store(_vs, _sr, arith.constant(0, type=_i32))
                                else:
                                  for ms in range_constexpr(MB):
                                    out_row_base = (m_row_base + arith.constant(ms * 16, index=True)
                                                    + ld16 * c4)
                                    for ii in range_constexpr(4):
                                        p_row = out_row_base + arith.constant(ii, index=True)
                                        if p_row < valid_hi:
                                            p_i = arith.index_cast(_i32, p_row)
                                            r_sc = buffer_load(_tsm_sc, p_i * 3 + 0, vec_width=1, dtype=_i32)
                                            t_sc = buffer_load(_tsm_sc, p_i * 3 + 1, vec_width=1, dtype=_i32)
                                            k_sc = buffer_load(_tsm_sc, p_i * 3 + 2, vec_width=1, dtype=_i32)
                                            data_row = (_scat_base(r_sc) + _c64(off_combine)
                                                        + _to_i64(k_sc) * c_fp8_slab
                                                        + _to_i64(t_sc) * c_fp8_row)
                                            sc_row = (_scat_base(r_sc) + _c64(off_combine_scale)
                                                      + _to_i64(k_sc) * c_sc_slab
                                                      + _to_i64(t_sc) * c_sc_row)
                                            _data_rsrc = create_buffer_resource_from_addr(data_row)
                                            _sc_rsrc = create_buffer_resource_from_addr(sc_row)
                                            for t in range_constexpr(n_owned):
                                                _jq8 = cw + arith.constant(t * ncw, index=True)
                                                _n16q8 = n_block_v * c_n16pb + _jq8
                                                _colq8 = _n16q8 * c16 + lm16
                                                _vq8 = vector.extract(accs[ms][t],
                                                                      static_position=[ii], dynamic_position=[])
                                                _pkq8, _scq8 = _quant_fp8(_vq8)
                                                _coliq8 = arith.index_cast(_i32, _colq8)
                                                _n16iq8 = arith.index_cast(_i32, _n16q8)
                                                if lm4 == _c0iq8:
                                                    buffer_store(_pkq8, _data_rsrc, _coliq8,
                                                                 offset_is_bytes=True)
                                                if lm16_i == _c0iq8:
                                                    buffer_store(_bf16_i16(_scq8), _sc_rsrc, _n16iq8)
                            else:
                              for ms in range_constexpr(MB):
                                out_row_base = (m_row_base + arith.constant(ms * 16, index=True)
                                                + ld16 * c4)
                                for ii in range_constexpr(4):
                                    p_row = out_row_base + arith.constant(ii, index=True)
                                    if p_row < valid_hi:
                                        p_i = arith.index_cast(_i32, p_row)
                                        r_sc = buffer_load(_tsm_sc, p_i * 3 + 0, vec_width=1, dtype=_i32)
                                        t_sc = buffer_load(_tsm_sc, p_i * 3 + 1, vec_width=1, dtype=_i32)
                                        k_sc = buffer_load(_tsm_sc, p_i * 3 + 2, vec_width=1, dtype=_i32)
                                        dst_row = (_scat_base(r_sc) + _c64(off_combine)
                                                   + _to_i64(k_sc) * c_slabbytes_sc
                                                   + _to_i64(t_sc) * c_rowbytes_sc)
                                        _dst_rsrc = create_buffer_resource_from_addr(dst_row)
                                        for t in range_constexpr(n_owned):
                                            j = cw + arith.constant(t * ncw, index=True)
                                            n16_abs = n_block_v * c_n16pb + j
                                            out_col = n16_abs * c16 + lm16
                                            val = vector.extract(accs[ms][t], static_position=[ii], dynamic_position=[])
                                            buffer_store(_bf16_i16(val), _dst_rsrc,
                                                         arith.index_cast(_i32, out_col))
                        else:
                            for ms in range_constexpr(MB):
                                out_row_base = (m_row_base + arith.constant(ms * 16, index=True)
                                                + ld16 * c4)
                                for t in range_constexpr(n_owned):
                                    j = cw + arith.constant(t * ncw, index=True)
                                    n16_abs = n_block_v * c_n16pb + j
                                    out_col = n16_abs * c16 + lm16
                                    for ii in range_constexpr(4):
                                        out_row = out_row_base + arith.constant(ii, index=True)
                                        out_idx = out_row * c_N + out_col
                                        val = vector.extract(accs[ms][t], static_position=[ii], dynamic_position=[])
                                        buffer_store(val, out_rsrc, out_idx)

            if const_expr(return_executor):
                return _nf_tile
            for it_b in range(bid, total_super, num_sms):
                _nf_tile(arith.index_cast(_idx_t, arith.unwrap(it_b)))

        # =================================================================
        # Warp role: DISPATCH
        # -----------------------------------------------------------------
        # NOTE: this milestone runs dispatch on every warp
        # (``num_dispatch_warps == num_warps_per_block``). The role gate is
        # kept so the compute roles can slot in later without reshaping the
        # kernel; with the default config the predicate is always true.
        #
        # ``enable_dispatch`` (split-launch back kernel passes False) compiles
        # the whole dispatch phase out: the back launch reuses the counts /
        # pool / token-src-metadata the front launch already produced.
        # =================================================================
        if enable_dispatch and warp < num_dispatch_warps:
            # -------------------------------------------------------------
            # Phase A+B: slot-allocate per (token, topk) and push the
            # source token-topk index to the destination rank.
            #
            # DeepGEMM counts into LDS first, then does one packed atomic
            # per expert; here each valid token-topk does a single global
            # atomic on ``expert_send_count[expert]`` (low 32 bits). The
            # returned old value is this token's slot within
            # ``(dst_local_expert, src_rank)``; the running total ends up
            # in ``expert_send_count`` for Phase C.
            # -------------------------------------------------------------
            _r_idx = create_buffer_resource_from_addr(addr_local_base + _c64(off_inp_idx))
            for i in range(global_warp_id * tokens_per_warp, num_tokens, global_warp_num * tokens_per_warp):
                tok = i + lane // num_topk
                if lane < activate_lanes:
                    if tok < num_tokens:
                        ttk = i * num_topk + lane          # flat (token*topk + topk_slot)
                        expert = _to_i32(buffer_load(_r_idx, ttk, vec_width=1, dtype=T.i64()))
                        if expert >= 0:
                            dst_pe = expert // epr
                            local_e = expert % epr
                            send_addr = addr_local_base + _c64(off_send) + _to_i64(expert) * 8
                            slot = atomic_add_i32(send_addr, arith.constant(1))
                            # src_token_topk_idx[local_e][rank][slot] on dst.
                            idx = local_e * (npes * nmre) + rank * nmre + slot
                            dst_addr = (_peer_base(dst_pe) + _c64(off_srctt)
                                        + _to_i64(idx) * 4)
                            store_i32_system(dst_addr, ttk)

            # -------------------------------------------------------------
            # Grid barrier #1: all blocks have finished pushing.
            # -------------------------------------------------------------
            _grid_bar(off_barrier)

            # -------------------------------------------------------------
            # Phase C (block 0 only): publish per-rank receive counts and
            # accumulate the packed ``expert_recv_count_sum`` on every
            # destination rank. The packed high bits (1 << 32 per rank)
            # let the pull phase detect "all ranks arrived".
            # -------------------------------------------------------------
            if bid == 0:
                _r_send_i32 = create_buffer_resource_from_addr(addr_local_base + _c64(off_send))
                for e in range(tid, num_experts, block_threads):
                    c_low = buffer_load(_r_send_i32, e * 2, vec_width=1, dtype=T.i32())
                    dst_pe = e // epr
                    local_e = e % epr
                    # recv_count[rank][local_e] (low word) on dst.
                    rc_addr = (_peer_base(dst_pe) + _c64(off_recv)
                               + _to_i64(rank * epr + local_e) * 8)
                    store_i32_system(rc_addr, c_low)
                    # recv_count_sum[local_e] += (1 << 32) | c_low on dst.
                    sum_addr = _peer_base(dst_pe) + _c64(off_sum) + _to_i64(local_e) * 8
                    packed = _c64(1 << 32) + _to_i64(c_low)
                    atomic_add_u64_release_sys(sum_addr, packed)

            # -------------------------------------------------------------
            # Phase D: wait until every rank has published its counts
            # (cross-rank sync). Mirrors ``fetch_expert_recv_count``.
            # -------------------------------------------------------------
            for e in range(lane, epr, 64):
                mori_shmem.uint64_wait_until_greater_than(
                    addr_local_base + _c64(off_sum) + _to_i64(e) * 8, sum_arrival_threshold)
            fence_system_acquire()
            fx.barrier()

            # -------------------------------------------------------------
            # Phase E: pull tokens / SF / weights from source ranks into
            # the shared L1 pool, write source metadata, bump the per-block
            # arrival count.
            #
            # The flattened receive-token index ``flat_idx`` enumerates the
            # per-expert concatenation (no BLOCK_M padding). For each one we
            # recover (expert, token-in-expert, pool block offset) by
            # scanning experts, then (source rank, token-in-rank) by the
            # round-robin min-peeling DeepGEMM uses.
            # -------------------------------------------------------------
            _r_sum_i32 = create_buffer_resource_from_addr(addr_local_base + _c64(off_sum))
            _r_recv_i32 = create_buffer_resource_from_addr(addr_local_base + _c64(off_recv))
            _r_srctt = create_buffer_resource_from_addr(addr_local_base + _c64(off_srctt))
            _r_l1wts = create_buffer_resource_from_addr(addr_local_base + _c64(off_l1_wts))
            _r_tsm = create_buffer_resource_from_addr(addr_local_base + _c64(off_tsm))

            def _count_sum(e):
                return buffer_load(_r_sum_i32, e * 2, vec_width=1, dtype=T.i32())

            def _rank_count(r, e):
                return buffer_load(_r_recv_i32, (r * epr + e) * 2, vec_width=1, dtype=T.i32())

            # dispatch<->compute concurrency (true overlap): when a producer
            # subset is configured, ONLY the first ``num_producer_blocks`` blocks
            # pull (restrided among themselves); the remaining (consumer) blocks
            # start the loop past the end (0 iters) and head straight into the
            # L1 GEMM, where the per-pool-block readiness gates make them spin
            # only on the blocks the producers have not filled yet.  So while a
            # few producer blocks are still pulling across GPUs, the consumer
            # blocks already compute the ready pool blocks -- genuine overlap.
            if const_expr(enable_dispatch_overlap and 0 < num_producer_blocks < num_sms):
                _pull_start = arith.select(
                    bid < num_producer_blocks, global_warp_id,
                    arith.constant(num_max_pool_tokens))
                _pull_stride = num_producer_blocks * num_warps_per_block
            else:
                _pull_start = global_warp_id
                _pull_stride = global_warp_num
            for flat_idx in range(_pull_start, num_max_pool_tokens, _pull_stride):
                # --- Scan experts: find expert + token-in-expert + pool
                #     block offset for this flat receive index. ---
                start = arith.constant(0)
                blocks = arith.constant(0)
                chosen_e = arith.constant(0)
                tie = arith.constant(0)               # token_idx_in_expert
                pool_block_off = arith.constant(0)
                efound = _FALSE
                for e in range_constexpr(epr):
                    cnt = _count_sum(e)
                    here = _and(_not(efound), _and(flat_idx >= start, flat_idx < start + cnt))
                    chosen_e = arith.select(here, arith.constant(e), chosen_e)
                    tie = arith.select(here, flat_idx - start, tie)
                    pool_block_off = arith.select(here, blocks, pool_block_off)
                    efound = _or(efound, here)
                    start = start + cnt
                    blocks = blocks + (cnt + (block_m - 1)) // block_m
                valid = efound

                if valid:
                    # --- Round-robin min-peeling over ranks (scalar; runs
                    #     identically on every lane). ---
                    rem = [_rank_count(r, chosen_e) for r in range_constexpr(npes)]
                    offset = arith.constant(0)
                    slot = tie
                    chosen_rank = arith.constant(0)
                    tir = arith.constant(0)            # token_idx_in_rank
                    found = _FALSE
                    for _round in range_constexpr(npes):
                        fp = found
                        num_active = arith.constant(0)
                        minv = arith.constant(_INF)
                        for r in range_constexpr(npes):
                            act = rem[r] > 0
                            num_active = num_active + arith.select(act, arith.constant(1), arith.constant(0))
                            minv = arith.select(_and(act, rem[r] < minv), rem[r], minv)
                        div_safe = arith.select(num_active > 0, num_active, arith.constant(1))
                        num_round = minv * num_active
                        hit = _and(_and(_not(fp), num_active > 0), slot < num_round)
                        slot_in_round = slot % div_safe
                        seen = arith.constant(0)
                        for r in range_constexpr(npes):
                            act = rem[r] > 0
                            is_this = _and(_and(hit, act), seen == slot_in_round)
                            chosen_rank = arith.select(is_this, arith.constant(r), chosen_rank)
                            seen = seen + arith.select(act, arith.constant(1), arith.constant(0))
                        tir = arith.select(hit, offset + slot // div_safe, tir)
                        found = _or(fp, hit)
                        adv = _and(_not(fp), _not(hit))
                        slot = arith.select(adv, slot - num_round, slot)
                        offset = arith.select(adv, offset + minv, offset)
                        for r in range_constexpr(npes):
                            rem[r] = arith.select(adv, rem[r] - _min(rem[r], minv), rem[r])

                    # --- Decode the source token-topk index (written by the
                    #     source rank's Phase B into our workspace). ---
                    src_slot = (chosen_e * (npes * nmre) + chosen_rank * nmre + tir)
                    ttk_src = buffer_load(_r_srctt, src_slot, vec_width=1, dtype=T.i32())
                    src_token = ttk_src // num_topk
                    src_topk = ttk_src % num_topk

                    pool_token_idx = pool_block_off * block_m + tie

                    # Fused pull: SKIP the token + SF copy into the local pool.
                    # The WS L1 loadA reads them directly from the peer via the
                    # token_src_metadata written below, eliminating the pool
                    # roundtrip.  Only the (cheap) metadata / weights / arrival
                    # count are produced here, so dispatch becomes metadata-only.
                    if const_expr(not enable_fused_pull):
                        # --- Pull the token embedding (fp8) from the source
                        #     rank's input buffer into the L1 pool. ---
                        src_tok_addr = (_peer_base(chosen_rank) + _c64(off_inp_tok)
                                        + _to_i64(src_token) * hidden)
                        dst_tok_addr = (addr_local_base + _c64(off_l1_tok)
                                        + _to_i64(pool_token_idx) * hidden)
                        _r_src_tok = create_buffer_resource_from_addr(src_tok_addr)
                        _r_dst_tok = create_buffer_resource_from_addr(dst_tok_addr)
                        for off in range(lane * 4, n_i32, 256):
                            vec = buffer_load(_r_src_tok, off, vec_width=4, dtype=T.i32())
                            buffer_store(vec, _r_dst_tok, off)

                        # --- Pull SF, K-major source -> M-major pool (UTCCP
                        #     4x32 transpose within each 128-elem group). ---
                        idx128 = tie % block_m
                        sf_within = ((tie // block_m) * sf_block_m
                                     + (idx128 & 0xFFFFFF80)
                                     + (idx128 & 31) * 4
                                     + ((idx128 >> 5) & 3))
                        sf_pool_idx = pool_block_off * sf_block_m + sf_within
                        src_sf_addr = (_peer_base(chosen_rank) + _c64(off_inp_sf)
                                       + _to_i64(src_token) * (hidden // 32))
                        _r_src_sf = create_buffer_resource_from_addr(src_sf_addr)
                        _r_dst_sf = create_buffer_resource_from_addr(addr_local_base + _c64(off_l1_sf))
                        for j in range(lane, sf_uint32, 64):
                            sf_word = buffer_load(_r_src_sf, j, vec_width=1, dtype=T.i32())
                            buffer_store(sf_word, _r_dst_sf, j * padded_sf + sf_pool_idx)

                    # --- Weights + source metadata + arrival count
                    #     (single lane per token). ---
                    if lane == 0:
                        wt_addr = (_peer_base(chosen_rank) + _c64(off_inp_wts)
                                   + _to_i64(ttk_src) * 4)
                        _r_src_wt = create_buffer_resource_from_addr(wt_addr)
                        wt = buffer_load(_r_src_wt, 0, vec_width=1, dtype=T.f32())
                        buffer_store(wt, _r_l1wts, pool_token_idx)

                        buffer_store(chosen_rank, _r_tsm, pool_token_idx * 3 + 0)
                        buffer_store(src_token, _r_tsm, pool_token_idx * 3 + 1)
                        buffer_store(src_topk, _r_tsm, pool_token_idx * 3 + 2)

                        fence_system_release()
                        l1arr_addr = (addr_local_base + _c64(off_l1arr)
                                      + _to_i64(pool_block_off + tie // block_m) * 4)
                        atomic_add_i32(l1arr_addr, arith.constant(1))

        # =================================================================
        # Grid barrier #2 + L1 grouped GEMM over the pool. Compiled in only
        # when L1 weights are supplied (``enable_l1_compute``); dispatch-only
        # callers (e.g. the dispatch parity test) compile-time skip it via the
        # const_expr early return below (no compute IR is emitted).
        # =================================================================
        if const_expr(not enable_l1_compute):
            if const_expr(False):
                _ = (addr_l1_w, addr_l1_ws, addr_l1_out,
                     addr_l2_w, addr_l2_ws, addr_l2_out)
            # Tail reset: dispatch-only path, after Phase E consumed the counts.
            if const_expr(enable_reset):
                _reset_round()
            return

        # dispatch->L1 sync: the global dispatch->L1 grid barrier is replaced by
        # per-pool-block readiness gates (``_wait_block_ready``) inside the L1
        # tile loop (dispatch<->compute overlap), so a block can begin computing
        # already-pulled pool blocks while later ones are still in flight.

        # --- L1 grouped GEMM over the pool (validated body from
        #     mega_moe_l1_gemm_kernel; inlined so the AST transform handles
        #     the dynamic persistent loop + per-tile expert predicate). ---
        _i32 = _Tt.i32
        _i64 = _Tt.i64
        _idx_t = ir.IndexType.get()
        vec4_f32 = _Tt.vec(4, _Tt.f32)
        vec8_i32 = _Tt.vec(8, _i32)
        vec4_i32 = _Tt.vec(4, _i32)
        vec2_i64 = _Tt.vec(2, _i64)
        vec4_i64 = _Tt.vec(4, _i64)

        _N = 2 * intermediate_hidden
        _K = hidden
        _n_ktiles = _K // 128
        _hidden_half = _K // 2
        _n_tiles16 = _N // 16
        _m_tiles16_max = num_max_pool_tokens // 16
        _cbsz, _blgp = 0, 4

        _lane_idx = arith.index_cast(_idx_t, arith.unwrap(lane))
        c16 = arith.constant(16, index=True)
        c4 = arith.constant(4, index=True)
        c32 = arith.constant(32, index=True)
        c64 = arith.constant(64, index=True)
        lm16 = _lane_idx % c16
        ld16 = _lane_idx / c16

        x_rsrc = create_buffer_resource_from_addr(addr_local_base + _c64(off_l1_tok))
        sf_rsrc = create_buffer_resource_from_addr(addr_local_base + _c64(off_l1_sf))
        sum_rsrc = create_buffer_resource_from_addr(addr_local_base + _c64(off_sum))
        w_rsrc = create_buffer_resource_from_addr(addr_l1_w)
        ws_rsrc = create_buffer_resource_from_addr(addr_l1_ws)
        out_rsrc = create_buffer_resource_from_addr(addr_l1_out)

        c_hidden = arith.constant(_K, index=True)
        c_N = arith.constant(_N, index=True)
        c_bm = arith.constant(block_m, index=True)
        c_nt16 = arith.constant(_n_tiles16, index=True)
        c0_i64 = arith.constant(0, type=_i64)
        c8i = arith.constant(8, type=_i32)
        c16_i32 = arith.constant(16, type=_i32)
        c_ff = arith.constant(0xFF, type=_i32)
        c2i = arith.constant(2, index=True)
        # stride_n0 = ceil(K/256)*64 (single 128-K tile still strides one block).
        c_sn0 = arith.constant(((_K + 255) // 256) * 64, index=True)
        scale_lane = ld16 * c16 + lm16
        a_byte_sh = arith.index_cast(_i32, ld16) * c8i
        g16 = ld16 * c16

        b_layout = make_preshuffle_b_layout(
            arith, c_n=arith.constant(epr * _N, index=True),
            c_k=arith.constant(_hidden_half, index=True),
            kpack_bytes=16, elem_bytes=1,
        ).layout_b
        i32_0 = arith.constant(0, type=_i32)
        _c_not127 = arith.constant(~127 & 0xFFFFFFFF, type=_i32)
        _c31 = arith.constant(31, type=_i32)
        _c3 = arith.constant(3, type=_i32)
        _c5 = arith.constant(5, type=_i32)
        _c4_i = arith.constant(4, type=_i32)
        _c_bm_i = arith.constant(block_m, type=_i32)
        _c_sbm_i = arith.constant(sf_block_m, type=_i32)

        if const_expr(enable_l1_gemm and enable_l1_ws):
            _l1_exec = _ws_gemm(
                bid=bid, warp=warp, lm16=lm16, ld16=ld16, g16=g16,
                scale_lane=scale_lane, a_byte_sh=a_byte_sh,
                x_rsrc=x_rsrc, sf_rsrc=sf_rsrc, sum_rsrc=sum_rsrc,
                w_rsrc=w_rsrc, ws_rsrc=ws_rsrc, out_rsrc=out_rsrc,
                b_layout=b_layout,
                _i32=_i32, _i64=_i64, _idx_t=_idx_t,
                vec4_f32=vec4_f32, vec8_i32=vec8_i32, vec4_i32=vec4_i32,
                vec2_i64=vec2_i64, vec4_i64=vec4_i64,
                c16=c16, c4=c4, c32=c32, c64=c64, c2i=c2i,
                c_hidden=c_hidden, c_N=c_N, c_bm=c_bm, c_sn0=c_sn0,
                c0_i64=c0_i64, c8i=c8i, c16_i32=c16_i32, c_ff=c_ff, i32_0=i32_0,
                _c_not127=_c_not127, _c31=_c31, _c3=_c3, _c5=_c5,
                _c4_i=_c4_i, _c_bm_i=_c_bm_i, _c_sbm_i=_c_sbm_i,
                _n_ktiles=_n_ktiles,
                ws_n16pb=_ws_n16pb, ws_ncw=_ws_num_consumer_warps,
                ws_n_owned=_ws_n_owned, ws_n_blocks=_ws_n_blocks,
                m16pb=_ws_m16pb, n_stages=_ws_l1_stages,
                fused_pull=enable_fused_pull, pull_once=enable_fused_pull_once,
                fuse_swiglu=enable_fuse_swiglu,
                l2_signal=enable_l2_readiness,
                return_executor=True,
                probe_skip_fused_swiglu=probe_skip_fused_swiglu,
            )
            # External persistent loop driving the returned per-tile executor.
            # In merged-scheduler mode the L1 loop is NOT run here: the unified
            # wave-interleave loop below (after the L2 executor is built) drives
            # both _l1_exec and the L2 executor per wave.
            if const_expr(not enable_merged_scheduler):
                _m_super_tiles_l1 = (num_max_pool_tokens // 16) // _ws_m16pb
                if const_expr(enable_fuse_swiglu):
                    for it_m in range(bid, _m_super_tiles_l1, num_sms):
                        _l1_exec(arith.index_cast(_idx_t, arith.unwrap(it_m)))
                else:
                    _total_super_l1 = _m_super_tiles_l1 * _ws_n_blocks
                    for it_b in range(bid, _total_super_l1, num_sms):
                        _l1_exec(arith.index_cast(_idx_t, arith.unwrap(it_b)))

        if const_expr(not enable_l2_compute):
            if const_expr(False):
                _ = (addr_l2_w, addr_l2_ws, addr_l2_out)
            # Tail reset: dispatch+L1 path, after the L1 GEMM consumed the
            # counts (sum_rsrc expert scan).  Split-launch front passes
            # enable_reset=False -> the counts/pool survive for the back launch.
            if const_expr(enable_reset):
                _reset_round()
            return

        # Split-launch front boundary: the L1 WS epilogue (enable_fuse_swiglu)
        # already produced the fp8 L2 pool + SF in the symmetric buffer, so the
        # front launch stops HERE -- the Python ``return`` ends tracing, so none
        # of the L2-GEMM / combine IR (the register hogs that spill the
        # monolithic build) is emitted.  The pool + dispatch counts persist for
        # the back launch (front passes enable_reset=False).  Monolithic and
        # back launches pass enable_l2_gemm=True and fall through to the L2 GEMM.
        if const_expr(not enable_l2_gemm):
            if const_expr(enable_reset):
                _reset_round()
            return

        # =================================================================
        # Grid barrier #3 + SwiGLU + MXFP8 quant.
        #   l1_out[pool, 2*ih] (f32, at addr_l1_out) -> l2_acts (fp8 [pool, ih],
        #   off_l2_tok) + l2_acts_sf (M-major, off_l2_sf).  Validated body from
        #   mega_moe_swiglu_quant_kernel; grid-stride over 32-channel groups.
        #   topk_weight is applied here (combine is then a plain sum).
        # =================================================================
        # Producer-side release (folded into _grid_bar): flush this block's L1
        # GEMM stores to addr_l1_out (plain device memory) to the system
        # coherence point *before* signalling, so the SwiGLU consumer's
        # acquire actually sees them.
        #
        # Fused-SwiGLU path: the L1 WS epilogue already produced the L2 fp8 pool
        # + SF in-kernel (no f32 l1_out round-trip), so this standalone SwiGLU
        # pass AND grid barrier #3 are skipped entirely -- only barrier #4
        # (below) is needed before L2 to publish the fused pool writes.
        if const_expr(not enable_fuse_swiglu):
            _grid_bar(off_barrier + 8)

        _ih = intermediate_hidden
        _groups_per_row = _ih // 32
        _f32q = _Tt.f32
        # NOTE: bound the *store* resources to their exact pool-region byte
        # size.  The SwiGLU grid-stride loop sweeps the whole pool (incl.
        # padding rows whose M-major SF index is large); without a real
        # num_records the hardware OOB check is disabled (default ~4GB), so
        # padding-row stores scribble outside the region and corrupt the
        # workspace.  The standalone kernel got this for free from the
        # tensor-backed resource's max_size.
        _l2tok_bytes = num_max_pool_tokens * _ih
        _l2sf_bytes = padded_sf * (_ih // 32)
        wts_rsrc = create_buffer_resource_from_addr(addr_local_base + _c64(off_l1_wts))
        # Re-materialise the L1-output read resource: the descriptor created in
        # the L1 GEMM phase (out_rsrc) is a cold SGPR group that gets spilled /
        # clobbered across the grid barriers + heavy SwiGLU register pressure,
        # so the gate/up reads silently return 0.  Recreate it fresh here.
        out_rsrc = create_buffer_resource_from_addr(addr_l1_out)
        l2tok_rsrc = create_buffer_resource_from_addr(
            addr_local_base + _c64(off_l2_tok), num_records_bytes=_l2tok_bytes)
        l2sf_rsrc = create_buffer_resource_from_addr(
            addr_local_base + _c64(off_l2_sf), num_records_bytes=_l2sf_bytes)

        c_ih = arith.constant(_ih, index=True)
        c2ih = arith.constant(2 * _ih, index=True)
        c128 = arith.constant(128, index=True)
        vec4_f32q = _Tt.vec(4, _f32q)
        c1_f32 = arith.constant(1.0, type=_f32q)
        neg_log2e = arith.constant(-1.4426950408889634, type=_f32q)
        c_inv448 = arith.constant(1.0 / 448.0, type=_f32q)
        c_448 = arith.constant(448.0, type=_f32q)
        c_neg448 = arith.constant(-448.0, type=_f32q)
        c0_i32q = arith.constant(0, type=_i32)
        c23q = arith.constant(23, type=_i32)
        c_0x400000 = arith.constant(0x400000, type=_i32)
        c_0xFF800000 = arith.constant(0xFF800000 & 0xFFFFFFFF, type=_i32)
        c254q = arith.constant(254, type=_i32)

        # Each thread owns one M-major SF int32 word == 4 contiguous 32-channel
        # quant groups (128 channels) of one pool row.  Emitting the SF as one
        # packed i32 avoids sub-dword buffer_store, which is silently dropped
        # by from-addr buffer resources on this target.
        _sfw_per_row = _ih // 128
        c_sfwpr = arith.constant(_sfw_per_row, index=True)
        # Fused path produced the pool in the L1 epilogue -> this standalone
        # SwiGLU sweep does nothing (zero-trip grid-stride loop).
        _total_sfwords = 0 if enable_fuse_swiglu else (num_max_pool_tokens * _sfw_per_row)
        _gt_stride = num_sms * block_threads
        gthread0 = bid * block_threads + tid
        c_padq = arith.constant(padded_sf, type=_i32)

        def _scaledq(av_j, recip):
            s = av_j * recip
            s = arith.minimumf(s, c_448)
            return arith.maximumf(s, c_neg448)

        for ww in range(gthread0, _total_sfwords, _gt_stride):
            ww_i = arith.index_cast(_idx_t, arith.unwrap(ww))
            row = ww_i / c_sfwpr
            kw = ww_i % c_sfwpr
            w_val = buffer_load(wts_rsrc, row, vec_width=1, dtype=_f32q)
            gate_row = row * c2ih
            up_row = gate_row + c_ih
            base_col = kw * c128
            packed_sf = c0_i32q
            for gsub in range_constexpr(4):
                gcol0 = base_col + arith.constant(gsub * 32, index=True)
                gbase = gate_row + gcol0
                ubase = up_row + gcol0
                act_vals = []
                for ch in range_constexpr(8):
                    offc = arith.constant(ch * 4, index=True)
                    g4 = vector.bitcast(
                        vec4_f32q, buffer_load(out_rsrc, gbase + offc, vec_width=4, dtype=_i32))
                    u4 = vector.bitcast(
                        vec4_f32q, buffer_load(out_rsrc, ubase + offc, vec_width=4, dtype=_i32))
                    for vi in range_constexpr(4):
                        gv = vector.extract(g4, static_position=[vi], dynamic_position=[])
                        uv = vector.extract(u4, static_position=[vi], dynamic_position=[])
                        t = gv * neg_log2e
                        emu = _llvm_d.call_intrinsic(_f32q, "llvm.amdgcn.exp2.f32", [t], [], [])
                        den = c1_f32 + emu
                        sig = _llvm_d.call_intrinsic(_f32q, "llvm.amdgcn.rcp.f32", [den], [], [])
                        act_vals.append(gv * sig * uv * w_val)
                amax = arith.constant(0.0, type=_f32q)
                for vi in range_constexpr(32):
                    av = _llvm_d.call_intrinsic(_f32q, "llvm.fabs.f32", [act_vals[vi]], [], [])
                    amax = arith.maximumf(amax, av)
                y = (amax * c_inv448).bitcast(_i32)
                e8m0 = arith.shrui(arith.andi(arith.addi(y, c_0x400000), c_0xFF800000), c23q)
                recip = arith.shli(arith.subi(c254q, e8m0), c23q).bitcast(_f32q)

                out_byte_base = row * c_ih + gcol0
                for wwd in range_constexpr(8):
                    pk = c0_i32q
                    pk = rocdl.cvt_pk_fp8_f32(
                        _i32, _scaledq(act_vals[wwd * 4 + 0], recip),
                        _scaledq(act_vals[wwd * 4 + 1], recip), pk, 0)
                    pk = rocdl.cvt_pk_fp8_f32(
                        _i32, _scaledq(act_vals[wwd * 4 + 2], recip),
                        _scaledq(act_vals[wwd * 4 + 3], recip), pk, 1)
                    word_off = out_byte_base + arith.constant(wwd * 4, index=True)
                    buffer_store(pk, l2tok_rsrc, word_off, offset_is_bytes=True)
                # pack this group's e8m0 into byte ``gsub`` of the SF word.
                packed_sf = arith.ori(
                    packed_sf, arith.shli(e8m0, arith.constant(gsub * 8, type=_i32)))

            # M-major SF word index (row permutation within block; kw K-word).
            row_iq = arith.index_cast(_i32, row)
            pboq = arith.divui(row_iq, _c_bm_i)
            idx_inq = arith.remui(row_iq, _c_bm_i)
            sf_withinq = arith.addi(
                arith.addi(arith.andi(idx_inq, _c_not127),
                           arith.muli(arith.andi(idx_inq, _c31), _c4_i)),
                arith.andi(arith.shrui(idx_inq, _c5), _c3))
            sf_rowq = arith.addi(arith.muli(pboq, _c_sbm_i), sf_withinq)
            kw_iq = arith.index_cast(_i32, kw)
            wordidxq = arith.addi(sf_rowq, arith.muli(kw_iq, c_padq))
            # Re-materialise the SF resource inline: the SwiGLU loop has heavy
            # register pressure and the cold ``l2sf_rsrc`` descriptor (used once
            # per iter) is otherwise spilled/clobbered, silently dropping the
            # store.  Recreating from the raw address is just a few SGPR moves.
            l2sf_rsrc_i = create_buffer_resource_from_addr(
                addr_local_base + _c64(off_l2_sf))
            buffer_store(packed_sf, l2sf_rsrc_i,
                         arith.index_cast(_idx_t, arith.muli(wordidxq, _c4_i)),
                         offset_is_bytes=True)

        # =================================================================
        # Grid barrier #4 + L2 grouped GEMM: l2_acts[pool, ih] @ W2 -> addr_l2_out.
        #   Same validated fragment body as L1 with K=ih, N=hidden, A-scale
        #   adapt-on-load (M-major), B = preshuffled fp4 W2 + CK e8m0 scale.
        # =================================================================
        # Release the SwiGLU pool writes (l2_acts / l2_sf) before signalling
        # (release fence folded into _grid_bar).  When the per-pool-block L2
        # readiness gate is on, each L2 tile waits on ``l2_arrival_mask`` (set
        # by the fused L1 epilogue) instead, so this global barrier is dropped.
        if const_expr(not enable_l2_readiness):
            _grid_bar(off_barrier + 12)

        # TIER-1 fused combine: the L2 WS epilogue P2P-scatters bf16 straight
        # into peers' combine staging (no f32 l2_out round-trip, no L2->combine
        # grid barrier).  Requires the WS L2 path + combine + L2 GEMM in the
        # same (monolithic) launch.
        _l2_fuse_combine = (bool(enable_fuse_combine) and bool(enable_combine)
                            and bool(enable_l2_ws) and bool(enable_l2_gemm)
                            and not probe_skip_combine and not probe_skip_l2gemm)

        _N2 = hidden
        _K2 = _ih
        _n_ktiles2 = _K2 // 128
        _hidden_half2 = _K2 // 2
        _n_tiles16_2 = _N2 // 16
        c_N2 = arith.constant(_N2, index=True)
        c_hidden2 = arith.constant(_K2, index=True)
        c_nt16_2 = arith.constant(_n_tiles16_2, index=True)
        c_sn0_2 = arith.constant(((_K2 + 255) // 256) * 64, index=True)
        b_layout2 = make_preshuffle_b_layout(
            arith, c_n=arith.constant(epr * _N2, index=True),
            c_k=arith.constant(_hidden_half2, index=True),
            kpack_bytes=16, elem_bytes=1,
        ).layout_b
        w2_rsrc = create_buffer_resource_from_addr(addr_l2_w)
        ws2_rsrc = create_buffer_resource_from_addr(addr_l2_ws)
        l2out_rsrc = create_buffer_resource_from_addr(addr_l2_out)
        # Re-materialise the pool A-side resources for L2 (the L1-phase
        # descriptors may have been spilled across the inlined SwiGLU block).
        sum_rsrc2 = create_buffer_resource_from_addr(addr_local_base + _c64(off_sum))
        l2a_rsrc = create_buffer_resource_from_addr(addr_local_base + _c64(off_l2_tok))
        l2asf_rsrc = create_buffer_resource_from_addr(addr_local_base + _c64(off_l2_sf))

        # Warp-specialised L2 GEMM: same producer/consumer pipeline as L1, with
        # A = l2_acts pool, B = W2, K = ih, N = hidden, scale = l2_sf.
        _ws_gemm(
            bid=bid, warp=warp, lm16=lm16, ld16=ld16, g16=g16,
            scale_lane=scale_lane, a_byte_sh=a_byte_sh,
            x_rsrc=l2a_rsrc, sf_rsrc=l2asf_rsrc, sum_rsrc=sum_rsrc2,
            w_rsrc=w2_rsrc, ws_rsrc=ws2_rsrc, out_rsrc=l2out_rsrc,
            b_layout=b_layout2,
            _i32=_i32, _i64=_i64, _idx_t=_idx_t,
            vec4_f32=vec4_f32, vec8_i32=vec8_i32, vec4_i32=vec4_i32,
            vec2_i64=vec2_i64, vec4_i64=vec4_i64,
            c16=c16, c4=c4, c32=c32, c64=c64, c2i=c2i,
            c_hidden=c_hidden2, c_N=c_N2, c_bm=c_bm, c_sn0=c_sn0_2,
            c0_i64=c0_i64, c8i=c8i, c16_i32=c16_i32, c_ff=c_ff, i32_0=i32_0,
            _c_not127=_c_not127, _c31=_c31, _c3=_c3, _c5=_c5,
            _c4_i=_c4_i, _c_bm_i=_c_bm_i, _c_sbm_i=_c_sbm_i,
            _n_ktiles=_n_ktiles2,
            ws_n16pb=_ws_n16pb_l2, ws_ncw=_ws_num_consumer_warps,
            ws_n_owned=_ws_n_owned_l2, ws_n_blocks=_ws_n_blocks_l2,
            m16pb=_ws_m16pb_l2, n_stages=_ws_l2_stages,
            combine_scatter=_l2_fuse_combine,
            l2_wait_expected=0,
        )

        # =================================================================
        # Grid barrier #5/#6 + COMBINE scatter-reduce -> y.
        #   Scatter: each valid L2 pool row p reads token_src_metadata[p] =
        #   (rank r, token t, topk k) and P2P-writes bf16(l2_out[p, :hidden])
        #   into source rank r's combine staging slot [k][t].  Reduce: each
        #   local token t sums its <=num_topk bf16 partials -> y[t] bf16.
        #   ``topk_weight`` is already folded into the activations in SwiGLU,
        #   so combine is a plain f32 sum.
        # NOTE: the two grid barriers below are *per-rank* (local counter).
        #   Single-rank (ws=1) is fully correct; multi-rank combine still
        #   needs a cross-rank barrier (peers write our staging) -- tracked in
        #   M1g.4.
        # =================================================================
        if const_expr(enable_combine and not probe_skip_combine):
            _H = hidden
            _Hhalf = _H // 2
            _f32c = _Tt.f32
            c_H = arith.constant(_H, index=True)
            c2idx = arith.constant(2, index=True)
            c_0xffff = arith.constant(0xFFFF, type=_i32)
            c16i = arith.constant(16, type=_i32)
            c1i = arith.constant(1, type=_i32)
            c_7fff = arith.constant(0x7FFF, type=_i32)
            c_rowbytes = _c64(_H * 2)        # bf16 token-row stride
            c_slabbytes = _c64(m * _H * 2)   # per-topk-slot slab stride
            vec2_i32c = _Tt.vec(2, _i32)
            vec4_i32t = _Tt.vec(4, _i32)
            c8idx2 = arith.constant(8, index=True)
            c4idx2 = arith.constant(4, index=True)

            def _f32_to_bf16(vi32):
                lsb = arith.andi(arith.shrui(vi32, c16i), c1i)
                rounded = arith.addi(arith.addi(vi32, c_7fff), lsb)
                return arith.andi(arith.shrui(rounded, c16i), c_0xffff)

            def _bf16_to_f32(b16):
                return arith.shli(arith.andi(b16, c_0xffff), c16i).bitcast(_f32c)

            # ----- release L2 GEMM output + grid barrier #5 -----
            # TIER-1: when the L2 epilogue already P2P-scattered, there is no
            # f32 l2_out to release and no standalone scatter to feed, so the
            # L2->combine grid barrier is dropped.
            if const_expr(not _l2_fuse_combine):
                _grid_bar(off_barrier + 16)

            _skip_scatter = False
            _skip_reduce = False
            # ----- scatter: valid pool row -> source-rank combine staging -----
            # (skipped under TIER-1: the L2 epilogue already scattered directly.)
            tsm_rsrc = create_buffer_resource_from_addr(addr_local_base + _c64(off_tsm))
            l2o_rsrc = create_buffer_resource_from_addr(addr_l2_out)
            sumc_rsrc = create_buffer_resource_from_addr(addr_local_base + _c64(off_sum))
            _scatter_n = 0 if (_skip_scatter or _l2_fuse_combine) else num_max_pool_tokens
            for p in range(global_warp_id, _scatter_n, global_warp_num):
                p_idx = arith.index_cast(_idx_t, arith.unwrap(p))
                # validity: p in [base_e, base_e + cnt_e) for some expert e
                valid = arith.constant(0) == arith.constant(1)
                blk = arith.constant(0, index=True)
                for e in range_constexpr(epr):
                    cnt = arith.index_cast(
                        _idx_t, buffer_load(sumc_rsrc, e * 2, vec_width=1, dtype=_i32))
                    base = blk * c_bm
                    here = arith.andi(p_idx >= base, p_idx < base + cnt)
                    valid = arith.select(here, arith.constant(0) == arith.constant(0), valid)
                    nb = (cnt + arith.constant(block_m - 1, index=True)) / c_bm
                    blk = blk + nb
                if valid:
                    r = buffer_load(tsm_rsrc, p_idx * 3, vec_width=1, dtype=_i32)
                    t = buffer_load(tsm_rsrc, p_idx * 3 + arith.constant(1, index=True),
                                    vec_width=1, dtype=_i32)
                    k = buffer_load(tsm_rsrc, p_idx * 3 + arith.constant(2, index=True),
                                    vec_width=1, dtype=_i32)
                    dst0 = (_peer_base(r) + _c64(off_combine)
                            + _to_i64(k) * c_slabbytes + _to_i64(t) * c_rowbytes)
                    src_elem0 = p_idx * c_H
                    # Each lane owns a contiguous 4-word (16 B) chunk of the
                    # bf16 token row: load 8 f32, pack -> vec4 i32, one wide
                    # plain store.  Across lanes a warp-iter writes 64*16 B
                    # contiguous + coalesced.  Visibility is flushed once by the
                    # fence_system_release() below (not per word).
                    for c in range(lane, _Hhalf // 4, 64):
                        c_idx = arith.index_cast(_idx_t, arith.unwrap(c))
                        f0 = src_elem0 + c_idx * c8idx2
                        lo4 = buffer_load(l2o_rsrc, f0, vec_width=4, dtype=_i32)
                        hi4 = buffer_load(l2o_rsrc, f0 + c4idx2, vec_width=4, dtype=_i32)
                        pk = []
                        for q in range_constexpr(4):
                            src = lo4 if q < 2 else hi4
                            qq = (q % 2) * 2
                            a = vector.extract(src, static_position=[qq], dynamic_position=[])
                            b = vector.extract(src, static_position=[qq + 1], dynamic_position=[])
                            pk.append(arith.ori(_f32_to_bf16(a),
                                                arith.shli(_f32_to_bf16(b), c16i)))
                        v4 = vector.from_elements(vec4_i32t, pk)
                        dst = dst0 + _to_i64(arith.index_cast(_i32, c_idx)) * _c64(16)
                        store_vec_plain(dst, v4, 16)


            # ----- flush all scatter stores to system scope (single release
            # fence per thread; replaces the old per-word release stores) -----
            if const_expr(not _skip_scatter):
                fence_system_release()

            # ----- grid barrier #6a (local): all my blocks done scattering -----
            _grid_bar(off_barrier + 20)

            # ----- grid barrier #6b (cross-rank): every peer has finished
            # writing into *this* rank's combine staging.  DeepGEMM-style
            # NVLink sense-reversal (``comm/barrier.cuh::nvlink_barrier``):
            # block 0 only.  A 2-bit counter holds ``phase`` (which of the two
            # signal slots) and ``sign`` (this round adds +1 or -1); each peer
            # gets one signal add, so the local signal bounces 0 <-> npes with
            # no reset.  Thread 0 advances the counter and waits the alternating
            # target.  All threads acquire so peers' writes are visible before
            # the reduce reads them.  Skipped for single-rank.
            # Prologue grid barrier #6a already ran above (all local blocks
            # finished scattering).  The NVLink barrier (block 0) makes every
            # peer's combine-staging writes visible; its epilogue grid barrier
            # propagates that to all blocks before the reduce reads them.
            _nvlink_barrier(off_barrier + 24, off_barrier + 28, off_barrier + 32,
                            epilogue_gridsync_off=off_barrier + 36)

            # ----- reduce: sum <=num_topk partials -> y[t] bf16 -----
            cidx_rsrc = create_buffer_resource_from_addr(addr_local_base + _c64(off_inp_idx))
            y_rsrc = create_buffer_resource_from_addr(addr_y)
            c_Hhalf = arith.constant(_Hhalf, index=True)
            c_ntopk = arith.constant(num_topk, index=True)
            if const_expr(_fp8_combine):
                # fp8 partials: dword (4 e4m3) + per-16ch bf16 scale -> dequant -> sum.
                _Hq = _H // 4
                _Hsc = _H // 16
                comb_fp8_rsrc = create_buffer_resource_from_addr(addr_local_base + _c64(off_combine))
                comb_sc_rsrc = create_buffer_resource_from_addr(addr_local_base + _c64(off_combine_scale))
                c_Hq = arith.constant(_Hq, index=True)
                c_Hsc = arith.constant(_Hsc, index=True)
                c1idx = arith.constant(1, index=True)
                vec2_f32c = _Tt.vec(2, _f32c)
                _i16t = _Tt.i16
                for tt in range(global_warp_id, (0 if _skip_reduce else num_tokens), global_warp_num):
                    t_idx = arith.index_cast(_idx_t, arith.unwrap(tt))
                    for w in range(lane, _Hq, 64):
                        w_idx = arith.index_cast(_idx_t, arith.unwrap(w))
                        grp = w_idx / c4idx2  # 4 dwords (16 cols) share one scale
                        acc0 = arith.constant(0.0, type=_f32c)
                        acc1 = arith.constant(0.0, type=_f32c)
                        acc2 = arith.constant(0.0, type=_f32c)
                        acc3 = arith.constant(0.0, type=_f32c)
                        for kk in range_constexpr(num_topk):
                            e = _to_i32(buffer_load(
                                cidx_rsrc, t_idx * c_ntopk + arith.constant(kk, index=True),
                                vec_width=1, dtype=_i64))
                            isval = e >= arith.constant(0)
                            delem = (arith.constant(kk * m * _Hq, index=True)
                                     + t_idx * c_Hq + w_idx)
                            word = buffer_load(comb_fp8_rsrc, delem, vec_width=1, dtype=_i32)
                            scelem = (arith.constant(kk * m * _Hsc, index=True)
                                      + t_idx * c_Hsc + grp)
                            sc_i16 = buffer_load(comb_sc_rsrc, scelem, vec_width=1, dtype=_i16t)
                            sc_f = _bf16_to_f32(sc_i16.extui(_i32))
                            lo2 = rocdl.cvt_pk_f32_fp8(vec2_f32c, word, 0)
                            hi2 = rocdl.cvt_pk_f32_fp8(vec2_f32c, word, 1)
                            v0 = vector.extract(lo2, static_position=[0], dynamic_position=[]) * sc_f
                            v1 = vector.extract(lo2, static_position=[1], dynamic_position=[]) * sc_f
                            v2 = vector.extract(hi2, static_position=[0], dynamic_position=[]) * sc_f
                            v3 = vector.extract(hi2, static_position=[1], dynamic_position=[]) * sc_f
                            acc0 = arith.select(isval, acc0 + v0, acc0)
                            acc1 = arith.select(isval, acc1 + v1, acc1)
                            acc2 = arith.select(isval, acc2 + v2, acc2)
                            acc3 = arith.select(isval, acc3 + v3, acc3)
                        w2 = w_idx * c2idx  # 4 fp8 cols -> 2 bf16 y-words
                        yb0 = _f32_to_bf16(acc0.bitcast(_i32))
                        yb1 = _f32_to_bf16(acc1.bitcast(_i32))
                        yb2 = _f32_to_bf16(acc2.bitcast(_i32))
                        yb3 = _f32_to_bf16(acc3.bitcast(_i32))
                        ypk0 = arith.ori(yb0, arith.shli(yb1, c16i))
                        ypk1 = arith.ori(yb2, arith.shli(yb3, c16i))
                        buffer_store(ypk0, y_rsrc, t_idx * c_Hhalf + w2)
                        buffer_store(ypk1, y_rsrc, t_idx * c_Hhalf + w2 + c1idx)
            else:
                comb_rsrc = create_buffer_resource_from_addr(addr_local_base + _c64(off_combine))
                for tt in range(global_warp_id, (0 if _skip_reduce else num_tokens), global_warp_num):
                    t_idx = arith.index_cast(_idx_t, arith.unwrap(tt))
                    for w in range(lane, _Hhalf, 64):
                        w_idx = arith.index_cast(_idx_t, arith.unwrap(w))
                        acc0 = arith.constant(0.0, type=_f32c)
                        acc1 = arith.constant(0.0, type=_f32c)
                        for kk in range_constexpr(num_topk):
                            e = _to_i32(buffer_load(
                                cidx_rsrc, t_idx * c_ntopk + arith.constant(kk, index=True),
                                vec_width=1, dtype=_i64))
                            isval = e >= arith.constant(0)
                            welem = (arith.constant(kk * m * _Hhalf, index=True)
                                     + t_idx * c_Hhalf + w_idx)
                            word = buffer_load(comb_rsrc, welem, vec_width=1, dtype=_i32)
                            lo = arith.andi(word, c_0xffff)
                            hi = arith.shrui(word, c16i)
                            acc0 = arith.select(isval, acc0 + _bf16_to_f32(lo), acc0)
                            acc1 = arith.select(isval, acc1 + _bf16_to_f32(hi), acc1)
                        yb0 = _f32_to_bf16(acc0.bitcast(_i32))
                        yb1 = _f32_to_bf16(acc1.bitcast(_i32))
                        ypk = arith.ori(yb0, arith.shli(yb1, c16i))
                        buffer_store(ypk, y_rsrc, t_idx * c_Hhalf + w_idx)

        # Tail reset: full path, after L2 GEMM + combine consumed the counts.
        if const_expr(enable_reset):
            _reset_round()

        # =================================================================
        # Warp roles: COMBINE  (legacy notes)
        # -----------------------------------------------------------------
        # L1 GEMM now runs above (all warps, post grid-barrier #2, output to
        # the debug ``addr_l1_out`` pool). The remaining compute phases
        # (SwiGLU activation + L2 GEMM + combine reduction) and the in-kernel
        # workspace reset are not implemented yet; ``y`` is left untouched and
        # the host zeroes the workspace counters between calls. The eventual
        # warp-specialised design (load/mma/epilogue split, persistent
        # (expert, m_block, n_block) scheduler) is sketched below.
        #
        # M1+ design (kept here so the scaffold is explicit; ``block_m`` is
        # now threaded from the host config -> make_mega_moe_jit, so dispatch
        # packing and the future compute tile share one value):
        #   * Warp partition (per block, warp = tid >> 6):
        #       [0, num_dispatch_warps)            -> DISPATCH (above)
        #       num_dispatch_warps    (loadA)      -> buffer_load pooled fp8
        #                                             A-tile + M-major SF
        #       num_dispatch_warps+1  (loadB)      -> fp4 weights + SF -> LDS
        #       num_dispatch_warps+2  (mma)        -> scaled MFMA -> acc regs
        #       >= num_dispatch_warps+kNumMMAWarps -> EPILOGUE/COMBINE
        #   * Scheduler (scheduler/mega_moe.cuh parity): persistent block
        #     claims (expert, m_block, n_block) via ``block_idx += num_sms``;
        #     per-expert pool-block offset is the prefix sum of
        #     ceil_div(recv_count_sum[e], block_m); L1 N-blocks =
        #     (2*intermediate_hidden)//BLOCK_N, L2 N-blocks = hidden//BLOCK_N
        #     (BLOCK_N = 128). loadA gates on ``l1_arrival_count`` (L1) /
        #     ``l2_arrival_mask`` (L2), both already produced by dispatch.
        #   * SMEM budget (gfx950): smem_a[stages][LOAD_BLOCK_M x BLOCK_K] +
        #     smem_b[stages][BLOCK_N x BLOCK_K] + SF tiles + smem_d store +
        #     amax-reduction scratch; ``num_stages`` sized to the LDS cap.
        # =================================================================
        if const_expr(False):
            _ = (addr_cumulative_stats,)

    return mega_moe_intranode, _ws_alloc


def make_mega_moe_jit(
    *,
    rank: int,
    world_size: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_max_tokens_per_rank: int,
    layout,
    block_m: int = 128,
    num_sms: int = 16,
    num_warps_per_block: int = 4,
    num_dispatch_warps: int = None,
    enable_l1_compute: bool = False,
    enable_l2_compute: bool = False,
    enable_combine: bool = False,
    num_l1_load_warps: int = 2,
    num_l1_stages: int = 3,
    l1_n16_per_block: int = 16,
):
    """Build the mega_moe JIT launcher.

    ``block_m`` must match the compute kernel's per-expert BLOCK_M; it drives
    the L1 pool token packing and the SF M-major transpose. ``num_sms`` is the
    persistent grid size used by the grid barrier and MUST be <= the number of
    co-resident blocks (otherwise the barrier deadlocks).

    The fused production config is fixed: warp-specialized L1/L2 GEMM, fused
    SwiGLU+mxfp8 quant L1 epilogue, fused fp8 combine, and dispatch<->compute
    overlap are all enabled whenever the relevant compute phase is compiled in
    (derived from ``enable_l1_compute`` / ``enable_l2_compute`` / ``enable_combine``).
    """
    if num_dispatch_warps is None:
        num_dispatch_warps = num_warps_per_block

    # Fixed production config (derived from the compiled-in compute phases):
    #   * L1 / L2 warp specialisation on whenever the phase is compiled in,
    #   * dispatch<->compute overlap on whenever L1 compute is compiled in,
    #   * fused SwiGLU+mxfp8 quant L1 epilogue on when both L1 and L2 compute in,
    #   * fused fp8 combine on when the L2 WS GEMM + combine run in one launch.
    enable_l1_ws = bool(enable_l1_compute)
    enable_l2_ws = bool(enable_l2_compute)
    enable_dispatch_overlap = bool(enable_l1_compute)
    enable_fuse_swiglu = bool(enable_l1_compute) and bool(enable_l2_compute)
    enable_fuse_combine = bool(enable_l2_compute) and bool(enable_combine)
    enable_fp8_combine = enable_fuse_combine

    # Optional numeric tuning knobs (perf only, correctness-neutral).
    #
    # ``num_sms`` is the persistent grid size: it MUST be <= the number of
    # co-resident blocks or the grid barrier deadlocks.  The op auto-detects
    # the device CU count and passes it here; empty / "0" means "keep the
    # passed-in value" so a 0 can never zero the grid.
    _env_num_sms = os.environ.get("MEGA_MOE_NUM_SMS")
    if _env_num_sms not in (None, "", "0"):
        num_sms = int(_env_num_sms)
    num_l1_load_warps = int(os.environ.get("MEGA_MOE_L1_LOAD_WARPS", num_l1_load_warps))
    num_l1_stages = int(os.environ.get("MEGA_MOE_L1_STAGES", num_l1_stages))
    l1_n16_per_block = int(os.environ.get("MEGA_MOE_L1_N16PB", l1_n16_per_block))
    num_warps_per_block = int(os.environ.get("MEGA_MOE_WARPS_PER_BLOCK", num_warps_per_block))
    if "MEGA_MOE_WARPS_PER_BLOCK" in os.environ:
        num_dispatch_warps = num_warps_per_block

    # The fused L1 compute consumes the CK microscale B layout, whose
    # n0-stride is ``(hidden//256)*64``; that layout requires hidden to be a
    # multiple of 256 (a K-block-pair).  Dispatch alone only needs %128, so the
    # guard only applies when the L1 compute phase is compiled in.
    if enable_l1_compute and hidden % 256 != 0:
        raise ValueError(
            f"hidden={hidden} must be a multiple of 256 for the fused L1 "
            "compute phase (CK B-scale layout)."
        )

    # ``block_m`` drives the L1 pool block packing and the SF M-major
    # transpose; the host SF pool (``padded_sf``) is sized as the max over
    # the candidate BLOCK_M set, so a non-candidate value can overflow it.
    if block_m not in _KCANDIDATE_BLOCK_M:
        raise ValueError(
            f"block_m={block_m} must be one of the candidate BLOCK_M values "
            f"{_KCANDIDATE_BLOCK_M}"
        )

    kernel, _ws_alloc = make_mega_moe_kernel(
        rank=rank,
        world_size=world_size,
        num_experts=num_experts,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        layout=layout,
        block_m=block_m,
        num_sms=num_sms,
        num_warps_per_block=num_warps_per_block,
        num_dispatch_warps=num_dispatch_warps,
        enable_l1_compute=enable_l1_compute,
        enable_l2_compute=enable_l2_compute,
        enable_combine=enable_combine,
        enable_l1_ws=enable_l1_ws,
        enable_l2_ws=enable_l2_ws,
        enable_dispatch_overlap=enable_dispatch_overlap,
        enable_fuse_swiglu=enable_fuse_swiglu,
        enable_fuse_combine=enable_fuse_combine,
        enable_fp8_combine=enable_fp8_combine,
        num_l1_load_warps=num_l1_load_warps,
        num_l1_stages=num_l1_stages,
        l1_n16_per_block=l1_n16_per_block,
    )

    block_threads = num_warps_per_block * 64

    # Closure vars that key the JIT cache: every input that affects the
    # emitted IR must appear here so a config change cannot reuse a stale
    # hsaco.
    _key = (
        rank, world_size, num_experts, num_topk, hidden, intermediate_hidden,
        num_max_tokens_per_rank, block_m, num_sms, num_warps_per_block,
        num_dispatch_warps, enable_l1_compute, enable_l2_compute, enable_combine,
        enable_l1_ws, enable_l2_ws, enable_dispatch_overlap,
        enable_fuse_swiglu, enable_fuse_combine, enable_fp8_combine,
        num_l1_load_warps, num_l1_stages, l1_n16_per_block,
        layout.total_bytes, _MEGA_MOE_JIT_SCHEMA_VERSION,
    )

    @flyc.jit
    def mega_moe_launch(
        addr_local_base: fx.Int64,
        addr_peer_base: fx.Int64,
        addr_cumulative_stats: fx.Int64,
        num_tokens: fx.Int32,
        addr_l1_w: fx.Int64,
        addr_l1_ws: fx.Int64,
        addr_l1_out: fx.Int64,
        addr_l2_w: fx.Int64,
        addr_l2_ws: fx.Int64,
        addr_l2_out: fx.Int64,
        addr_y: fx.Int64,
        stream: Stream = Stream(None),
    ):
        _ = _key
        if _ws_alloc is not None:
            _ws_alloc.finalized = False
            _ctx = CompilationContext.get_current()
            with ir.InsertionPoint(_ctx.gpu_module_body):
                _ws_alloc.finalize()
        kernel(
            addr_local_base,
            addr_peer_base,
            addr_cumulative_stats,
            num_tokens,
            addr_l1_w,
            addr_l1_ws,
            addr_l1_out,
            addr_l2_w,
            addr_l2_ws,
            addr_l2_out,
            addr_y,
        ).launch(
            grid=(num_sms, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return mega_moe_launch
