# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Python wrapper for the FlyDSL intra-node DispatchCombine op.

Mori-parity surface: symmetric I/O (``combine()`` writes back in
``cfg.data_type``; ``quant_type='fp8_direct_cast'`` keeps the caller
dtype bf16 but switches the wire dtype to fp8, equivalent to mori
``UseFp8DirectCast``).  Optional caps ``max_total_recv_tokens`` and
``max_token_type_size`` shrink / re-target shmem allocation -- see
the dataclass fields below for the exact semantics.

Buffer-sizing design note (intentional drift from mori):
  ``shmem_comb_inp_{tok,wts}`` stay sized to the worst case
  ``ws * M`` even when ``max_total_recv_tokens`` shrinks the
  dispatch-side recv cap, because combine Stage 1 P2P-scatters into
  slot ``rank * M + dest_lid`` with ``dest_lid ∈ [0, M)`` -- the
  max reachable index is ``ws*M - 1`` independent of cap, so
  shrinking these buffers (as mori does for ``combineInp``) would
  OOB-write Stage 1.  Dispatch-side buffers (indexed by
  ``dest_tok_id ∈ [0, effective_max_recv)``) DO shrink with the cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import mori.shmem as ms
import torch
from mori.shmem import mori_shmem_create_tensor

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch

from .dispatch_combine_intranode_kernel import (
    make_combine_jit,
    make_dispatch_jit,
)

# Supported token dtypes for the intranode kernel.  Kept in lock-step with
# mori's ``_DTYPE_SUFFIX`` table; any dtype outside this set is rejected at
# wrapper construction time rather than failing deep inside JIT codegen.
_SUPPORTED_TOK_DTYPES = (
    torch.bfloat16,
    torch.float32,
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float4_e2m1fn_x2,
)

_SUPPORTED_QUANT_TYPES = ("none", "fp8_direct_cast")

# Intranode topology cap: this op is the IntraNode variant, i.e. all
# peers must live inside the same single node (one HIP P2P domain).
# Standard MI300/MI325/MI350 nodes ship with 8 GPUs per node, which is
# also mori's ``gpu_per_node`` default for the intranode kernel.  Larger
# world_size requires the inter-node variants instead.
_MAX_INTRANODE_NPES = 8

# Vector-load granularity in the kernel: token embeddings are streamed
# through ``buffer_load(vec_width=4, dtype=i32)`` (16-byte chunks).
# This sets the alignment contract on ``token_bytes``.
_TOK_BYTES_ALIGN = 16

_DEFAULT_DISPATCH_BLOCK_NUM = 128
_DEFAULT_DISPATCH_WARP_NUM = 4
_DEFAULT_COMBINE_BLOCK_NUM = 128
_DEFAULT_COMBINE_WARP_NUM = 8


def _dtype_elem_size(dt):
    """Raw storage size in bytes.  fp4x2 returns 1 (two fp4 per byte);
    callers handle the fp4 ``hidden_dim // 2`` row-stride elsewhere.
    """
    return torch.tensor([], dtype=dt).element_size()


def _is_fp4_dtype(dt):
    return dt == torch.float4_e2m1fn_x2


def _token_bytes_for(dt, hidden_dim):
    """Per-row payload bytes for a given dtype + hidden_dim.

    Mirrors ``FlyDSLDispatchCombineConfig.token_bytes`` but parametrised
    on a launch-time dtype so dispatch / combine can use independent
    dtypes (mori parity).
    """
    if _is_fp4_dtype(dt):
        return hidden_dim // 2
    return hidden_dim * _dtype_elem_size(dt)


def _token_view_dim_for(dt, hidden_dim):
    """Per-row torch view trailing dim for a given dtype + hidden_dim
    (fp4 is packed 2-elements-per-byte, so the storage view exposes
    ``hidden_dim // 2`` columns)."""
    return hidden_dim // 2 if _is_fp4_dtype(dt) else hidden_dim


@dataclass
class FlyDSLDispatchCombineConfig:
    rank: int
    world_size: int
    hidden_dim: int
    max_num_inp_token_per_rank: int
    num_experts_per_rank: int
    num_experts_per_token: int
    data_type: torch.dtype = torch.bfloat16
    # Legacy shared geometry knobs (kept for compatibility with callers
    # still passing these fields). They are not used by launch selection
    # now that dispatch/combine each has dedicated defaults.
    warp_num_per_block: int = _DEFAULT_COMBINE_WARP_NUM
    block_num: int = _DEFAULT_COMBINE_BLOCK_NUM
    # Per-phase launch geometry defaults (no shared fallback).
    dispatch_warp_num_per_block: int = _DEFAULT_DISPATCH_WARP_NUM
    dispatch_block_num: int = _DEFAULT_DISPATCH_BLOCK_NUM
    combine_warp_num_per_block: int = _DEFAULT_COMBINE_WARP_NUM
    combine_block_num: int = _DEFAULT_COMBINE_BLOCK_NUM
    scale_dim: int = 0
    scale_type_size: int = 0
    enable_std_moe: bool = False
    use_external_inp_buf: bool = True
    quant_type: str = "none"
    # Cap on total receive tokens across peers (mori
    # ``maxTotalRecvTokens``).  ``0`` -> worst-case ``ws * M``.
    # Per-rank slots = ``ceil(cap/ws)``, clamped to ``M``.  Over-cap
    # tokens are folded into the dup-sentinel codepath (no OOB) but
    # are silently dropped -- caller owns routing staying in cap.
    max_total_recv_tokens: int = 0
    # Token-element upper bound in bytes for shmem allocation (mori
    # ``maxTokenTypeSize``).  ``0`` derives from ``data_type``; set
    # to keep one op alive across dtype switches without re-alloc.
    max_token_type_size: int = 0
    # Target ROCm chip ID, forwarded to JIT compile_hints (mostly
    # consumed by the fused gemm2+combine path's resident-block
    # estimator).  Default mirrors the dispatch_combine deployment.
    chip: str = "gfx950"
    # D-flag C-1: enable the fused gemm2+combine per-token cross-card
    # flag-sync codepath.  ``False`` keeps the const_expr DCE branch
    # (behaviour identical to baseline).  ``True`` requires a follow-up
    # commit that allocates real shmem buffers and ports the dispatch
    # kernel grid-stride memset of comb_flag plus the combine kernel
    # stage-3 spin wait.
    use_token_flag_sync: bool = False

    @property
    def is_fp4(self):
        return self.data_type == torch.float4_e2m1fn_x2

    @property
    def elem_size(self):
        return _dtype_elem_size(self.data_type)

    @property
    def token_bytes(self):
        if self.is_fp4:
            return self.hidden_dim // 2
        return self.hidden_dim * self.elem_size

    @property
    def token_view_dim(self):
        if self.is_fp4:
            return self.hidden_dim // 2
        return self.hidden_dim

    @property
    def block_dim(self):
        return self.warp_num_per_block * 64

    @property
    def zero_copy(self) -> bool:
        """Zero-copy combine mode switch.

        ``True`` means combine reads/writes through the registered shared
        staging buffer.
        """
        return not self.use_external_inp_buf

    @property
    def dispatch_warp_num_per_block_eff(self):
        return self.dispatch_warp_num_per_block

    @property
    def dispatch_block_num_eff(self):
        return self.dispatch_block_num

    @property
    def combine_warp_num_per_block_eff(self):
        return self.combine_warp_num_per_block

    @property
    def combine_block_num_eff(self):
        return self.combine_block_num

    @property
    def max_recv(self):
        """Worst-case ``ws * M`` -- the upper bound the kernel reasons
        about (sentinel encoding, sender-side ``atomic_add`` allocation).
        Independent of ``max_total_recv_tokens``; buffers may use the
        tighter ``effective_max_recv`` when the cap is set.
        """
        return self.world_size * self.max_num_inp_token_per_rank

    @property
    def effective_max_recv_per_rank(self):
        """Per-sender slot share on each dest PE (mori
        ``MaxNumTokensToRecvPerRank``): ``ceil(cap/ws)`` clamped to
        ``M``; ``cap == 0`` reverts to ``M``.
        """
        if self.max_total_recv_tokens <= 0:
            return self.max_num_inp_token_per_rank
        per_rank = (self.max_total_recv_tokens + self.world_size - 1) // self.world_size
        return min(per_rank, self.max_num_inp_token_per_rank)

    @property
    def effective_max_recv(self):
        """Total recv-slot count on each dest PE (mori
        ``MaxNumTokensToRecv``).  Used as the ``max_recv`` kernel
        parameter (dispatch sentinel encoding, Stage 3 addressing).
        """
        return self.world_size * self.effective_max_recv_per_rank

    @property
    def max_token_bytes(self):
        """Per-row shmem allocation bound (mori ``maxTokenTypeSize``):
        ``max_token_type_size > 0`` reserves room for a dtype larger
        than ``data_type``; ``0`` derives from ``token_bytes``.
        """
        if self.max_token_type_size > 0:
            return self.hidden_dim * self.max_token_type_size
        return self.token_bytes

    @property
    def scale_bytes(self):
        return self.scale_dim * self.scale_type_size


class FlyDSLDispatchCombineIntraNodeOp:

    def __init__(self, config):
        self.cfg = config
        # Validate BEFORE any GPU allocation so misconfigurations
        # surface as a clear ValueError, not an opaque HIP/MLIR error.
        self._check_config()
        self._dev = torch.device("cuda", config.rank)
        r = config.rank

        self._alloc_buffers()
        ms.shmem_barrier_all()

        npes = config.world_size
        self._p2p_tok_off = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_tis = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_out_wts = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_out_idx = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_out_tok = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_recv_num = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_out_scales = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        for pe in range(npes):
            self._p2p_tok_off[pe] = ms.shmem_ptr_p2p(self.shmem_tok_off.data_ptr(), r, pe)
            self._p2p_tis[pe] = ms.shmem_ptr_p2p(self.shmem_tok_id_to_src.data_ptr(), r, pe)
            self._p2p_out_wts[pe] = ms.shmem_ptr_p2p(self.shmem_disp_out_wts.data_ptr(), r, pe)
            self._p2p_out_idx[pe] = ms.shmem_ptr_p2p(self.shmem_disp_out_idx.data_ptr(), r, pe)
            self._p2p_out_tok[pe] = ms.shmem_ptr_p2p(self.shmem_disp_out_tok.data_ptr(), r, pe)
            self._p2p_recv_num[pe] = ms.shmem_ptr_p2p(self.shmem_recv_tok_num.data_ptr(), r, pe)
            self._p2p_out_scales[pe] = ms.shmem_ptr_p2p(self.shmem_out_scales.data_ptr(), r, pe)

        self._p2p_comb_inp = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_comb_inp_wts = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        self._p2p_xdb_mem = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        # D-flag C-1: per-token flag P2P pointer table.  ``dest_pe -> remote
        # base + flag offset`` lookup feeds the cross-card system-scope
        # ``atomic_add_global_at`` in the fused gemm2 epilogue's last
        # N-tile completion path.
        self._p2p_comb_flag = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        for pe in range(npes):
            self._p2p_comb_inp[pe] = ms.shmem_ptr_p2p(self.shmem_comb_inp_tok.data_ptr(), r, pe)
            self._p2p_comb_inp_wts[pe] = ms.shmem_ptr_p2p(self.shmem_comb_inp_wts.data_ptr(), r, pe)
            self._p2p_xdb_mem[pe] = ms.shmem_ptr_p2p(self.shmem_xdev_bar_mem.data_ptr(), r, pe)
            self._p2p_comb_flag[pe] = ms.shmem_ptr_p2p(self.shmem_comb_token_flag.data_ptr(), r, pe)

        # Dispatch (encode) and combine (decode, Stage 3) must agree on this.
        self._effective_max_recv = config.effective_max_recv

        # Launch-time JIT caches (mori parity): each ``op.dispatch(input)``
        # / ``op.combine(input)`` selects the kernel specialization by
        # ``input.dtype`` and lazy-compiles a fresh entry on cache miss.
        # ``_disp_jit_cache`` keyed by dispatch input dtype; combine cache
        # keyed by ``(combine_dtype, zero_copy, enable_weights,
        # fp8_direct_cast)`` because all four switches change the
        # generated kernel.
        self._disp_jit_cache: Dict[torch.dtype, Any] = {}
        self._disp_compiled_cache: Dict[torch.dtype, Any] = {}
        self._comb_jit_cache: Dict[Tuple[torch.dtype, bool, bool, bool], Any] = {}
        self._comb_compiled_cache: Dict[Tuple[torch.dtype, bool, bool, bool], Any] = {}

        # Must start at 1: a zero-init flag would satisfy the first
        # ``wait_until_equals(slot, 0)`` immediately and skip the sync.
        self._xdev_flag = torch.ones(1, dtype=torch.int64, device=self._dev)

        self._fx_out_tok = fx.Int64(self.shmem_disp_out_tok.data_ptr())
        self._fx_out_idx = fx.Int64(self.shmem_disp_out_idx.data_ptr())
        self._fx_tok_off = fx.Int64(self.shmem_tok_off.data_ptr())
        self._fx_recv_num = fx.Int64(self.shmem_recv_tok_num.data_ptr())
        self._fx_dest_ctr = fx.Int64(self.dest_pe_ctr.data_ptr())
        self._fx_disp_bar = fx.Int64(self.disp_bar.data_ptr())
        self._fx_tok_map = fx.Int64(self.dest_tok_map.data_ptr())
        self._fx_out_shmem_tok_id_to_src = fx.Int64(self.shmem_tok_id_to_src.data_ptr())
        self._fx_out_total_recv = fx.Int64(self.total_recv.data_ptr())
        self._fx_comb_inp = fx.Int64(self.shmem_comb_inp_tok.data_ptr())
        self._fx_comb_out = fx.Int64(self.shmem_comb_out_tok.data_ptr())
        self._fx_xdb_mem = fx.Int64(self.shmem_xdev_bar_mem.data_ptr())
        self._fx_xdev_flag = fx.Int64(self._xdev_flag.data_ptr())

        self._fx_comb_bar = fx.Int64(self.comb_bar.data_ptr())
        self._fx_trecv = fx.Int64(self.total_recv.data_ptr())
        self._fx_p2p_tok_off = fx.Int64(self._p2p_tok_off.data_ptr())
        self._fx_p2p_out_tok_id_to_src = fx.Int64(self._p2p_tis.data_ptr())
        self._fx_p2p_out_wts = fx.Int64(self._p2p_out_wts.data_ptr())
        self._fx_p2p_out_idx = fx.Int64(self._p2p_out_idx.data_ptr())
        self._fx_p2p_out_tok = fx.Int64(self._p2p_out_tok.data_ptr())
        self._fx_p2p_recv_num = fx.Int64(self._p2p_recv_num.data_ptr())
        self._fx_p2p_out_scales = fx.Int64(self._p2p_out_scales.data_ptr())
        self._fx_out_scales = fx.Int64(self.shmem_out_scales.data_ptr())
        self._fx_p2p_comb_inp = fx.Int64(self._p2p_comb_inp.data_ptr())
        self._fx_p2p_comb_inp_wts = fx.Int64(self._p2p_comb_inp_wts.data_ptr())
        self._fx_p2p_xdb_mem = fx.Int64(self._p2p_xdb_mem.data_ptr())
        self._fx_comb_inp_wts = fx.Int64(self.shmem_comb_inp_wts.data_ptr())
        self._fx_comb_out_wts = fx.Int64(self.shmem_comb_out_wts.data_ptr())
        self._fx_packed_recv_count = fx.Int64(self.packed_recv_count.data_ptr())
        self._fx_packed_recv_src_info = fx.Int64(self.packed_recv_src_info.data_ptr())
        self._fx_disp_tok_map = fx.Int64(self.disp_tok_to_ep_slot_map.data_ptr())
        self._fx_disp_grid_bar = fx.Int64(self.disp_grid_bar.data_ptr())
        self._fx_disp_out_wts = fx.Int64(self.shmem_disp_out_wts.data_ptr())

        # Surface aliases / wrappers for the fused GEMM2+combine launcher
        # (matches the attribute names used in fused_gemm2_combine).
        # ``_fx_tis`` aliases the existing tok_id_to_src wrapper; the
        # comb_flag / local_counter / p2p_comb_flag triple is the D-flag
        # C-1 per-token sync surface backed by real shmem / device
        # buffers (see ``_alloc_buffers`` and the P2P table block above).
        # ``use_token_flag_sync=False`` callers pay only the buffer
        # allocation cost; const_expr DCE elides every kernel-side load
        # and store.
        self._fx_tis = self._fx_out_shmem_tok_id_to_src
        self._fx_comb_flag = fx.Int64(self.shmem_comb_token_flag.data_ptr())
        self._fx_local_counter = fx.Int64(self.device_local_counter.data_ptr())
        self._fx_p2p_comb_flag = fx.Int64(self._p2p_comb_flag.data_ptr())

        # D-flag C-1: cached pointer to the most recent ``dispatch()``
        # input weights.  Stage 3b in ``combine_no_stage1`` reads weights
        # directly from this local buffer when ``use_token_flag_sync`` is
        # ON, bypassing the Stage 1 weight P2P scatter.  Populated on
        # every ``dispatch()`` call; 0 until the first dispatch.
        self._raw_input_wts_ptr = 0

        # Lazy skip_stage1 variant for the fused GEMM2+combine path
        # (Stage 2 + Stage 3 only, Stage 1 P2P scatter pre-staged
        # upstream).  Cache entries keyed by ``(input.dtype,
        # enable_weights)`` so a runtime dtype switch (e.g. fp4 dispatch
        # + bf16 combine) does not silently reuse a stale specialization.
        self._comb_no_s1_fn: Dict[Tuple[torch.dtype, bool], Any] = {}
        self._comb_no_s1_compiled: Dict[Tuple[torch.dtype, bool], Any] = {}

    def barrier(self):
        """Cross-rank shmem barrier (fused launcher API surface)."""
        ms.shmem_barrier_all()

    def reset(self):
        """Reset op state between graph captures.  Currently a barrier
        is sufficient because all per-launch counters are device-side
        ticket counters that derive their epoch from atomic_add."""
        self.barrier()

    def _alloc_buffers(self):
        cfg = self.cfg
        npes = cfg.world_size
        k = cfg.num_experts_per_token
        mt = cfg.max_num_inp_token_per_rank
        # Two recv-slot caps (mori parity):
        #   mr       = effective_max_recv  -- dispatch-side per-PE cap;
        #                                     shrinks with the user cap.
        #   mr_worst = max_recv = ws * M   -- combine-input cap; cannot
        #              shrink because Stage 1 indexes into slot
        #              ``rank * M + dest_lid``, dest_lid in [0, M).
        # Both reduce to ``ws * M`` when cap == 0.
        mr = cfg.effective_max_recv
        mr_worst = cfg.max_recv

        tb_max = cfg.max_token_bytes
        tok_i16_mr = (mr * tb_max + 1) // 2
        tok_i16_mr_worst = (mr_worst * tb_max + 1) // 2
        tok_i16_mt = (mt * tb_max + 1) // 2

        # All shmem buffers below are P2P-accessed by peer PEs except
        # shmem_comb_out_{tok,wts} (local-only writes today; stay on
        # the shmem heap for mori-layout parity and for future fused
        # ops that may P2P-fetch them).
        self.shmem_disp_out_tok = mori_shmem_create_tensor((tok_i16_mr,), torch.int16)
        self.shmem_disp_out_wts = mori_shmem_create_tensor((mr * k,), torch.float32)
        self.shmem_disp_out_idx = mori_shmem_create_tensor((mr * k,), torch.int32)
        scale_total = mr * cfg.scale_bytes if cfg.scale_bytes > 0 else 1
        self.shmem_out_scales = mori_shmem_create_tensor((scale_total,), torch.int8)
        self.shmem_tok_off = mori_shmem_create_tensor((1,), torch.int32)
        self.shmem_recv_tok_num = mori_shmem_create_tensor((npes,), torch.int32)
        self.shmem_tok_id_to_src = mori_shmem_create_tensor((mr,), torch.int32)
        self.shmem_comb_inp_tok = mori_shmem_create_tensor((tok_i16_mr_worst,), torch.int16)
        self.shmem_comb_inp_wts = mori_shmem_create_tensor((mr_worst * k,), torch.float32)
        self.shmem_comb_out_tok = mori_shmem_create_tensor((tok_i16_mt,), torch.int16)
        self.shmem_comb_out_wts = mori_shmem_create_tensor((mt * k,), torch.float32)
        self.shmem_xdev_bar_mem = mori_shmem_create_tensor((npes,), torch.int64)
        # D-flag C-1 per-token flag buffer: fused gemm2 epilogue's last
        # N-tile completion thread bumps remote ``flag[token_id] += 1``
        # via system-scope atomic_add; combine kernel stage 3 entry
        # spin-waits ``flag[tok_id] >= topk``.  Sized for the worst-case
        # recv slot count (mr_worst = ws * M); dispatch entry resets the
        # used prefix [0, cur_tok) at the start of each chain via
        # grid-stride memset.  ``use_token_flag_sync=False`` callers
        # pay only the ~32KB allocation; const_expr DCE elides every
        # load/store.
        self.shmem_comb_token_flag = mori_shmem_create_tensor((mr_worst,), torch.int32)

        # shmem_malloc returns uninitialized memory; zero the buffers
        # combine reads to keep degenerate slots harmless:
        #   - tok_id_to_src[unwritten] decodes to (pe=0,lid=0) so the
        #     Stage 1 P2P scatter collapses to a duplicate write;
        #   - xdev_bar_mem starts at 0 so the first CrossDeviceBarrier
        #     wait never observes stale data (cur_flag is monotonic);
        #   - comb_inp_{tok,wts} start at 0 so combine_no_stage1's
        #     Stage 3 accum never folds garbage into the result;
        #   - shmem_comb_token_flag starts at 0 so the very first chain
        #     (before any dispatch reset) sees a clean slate; thereafter
        #     the dispatch kernel grid-stride memset takes over.
        self.shmem_tok_id_to_src.zero_()
        self.shmem_comb_inp_tok.zero_()
        self.shmem_comb_inp_wts.zero_()
        self.shmem_xdev_bar_mem.zero_()
        self.shmem_comb_token_flag.zero_()

        self.dest_pe_ctr = torch.zeros(npes, dtype=torch.int32, device=self._dev)
        self.disp_bar = torch.zeros(1, dtype=torch.int32, device=self._dev)
        self.comb_bar = torch.zeros(1, dtype=torch.int32, device=self._dev)
        self.total_recv = torch.zeros(1, dtype=torch.int32, device=self._dev)
        sentinel = cfg.world_size * mr
        self.dest_tok_map = torch.full((mt * k,), sentinel, dtype=torch.int32, device=self._dev)

        if cfg.enable_std_moe:
            epr = cfg.num_experts_per_rank
            # Expert-side capacity is independent from
            # ``max_total_recv_tokens`` (mori treats them as separate
            # dims), so keep the worst-case bound.
            max_tok_per_expert = cfg.max_recv
            self.packed_recv_count = torch.zeros(epr, dtype=torch.int32, device=self._dev)
            self.packed_recv_src_info = torch.zeros(epr * max_tok_per_expert, dtype=torch.int32, device=self._dev)
            self.disp_tok_to_ep_slot_map = torch.full((mr * k,), -1, dtype=torch.int64, device=self._dev)
            # i64 ticket counter for the StdMoE Phase 4 in-kernel grid
            # barrier: never host-reset; each launch derives its epoch
            # from ``atomic_add``'s returned ticket.  i64 prevents
            # wraparound (>290k years at 1 M launches/s).
            self.disp_grid_bar = torch.zeros(1, dtype=torch.int64, device=self._dev)
        else:
            self.packed_recv_count = torch.zeros(1, dtype=torch.int32, device=self._dev)
            self.packed_recv_src_info = torch.zeros(1, dtype=torch.int32, device=self._dev)
            self.disp_tok_to_ep_slot_map = torch.zeros(1, dtype=torch.int64, device=self._dev)
            self.disp_grid_bar = torch.zeros(1, dtype=torch.int64, device=self._dev)

        # D-flag C-1 device-local counter (not symmetric).  The fused
        # gemm2 epilogue uses a device-scope ``atomic_add`` to count
        # N-tile completions per ``(target_token, j)`` slot; the last
        # N-tile completion thread issues the cross-card system-scope
        # atomic_add against the remote flag and ``atomicrmw xchg``-
        # resets the local slot to 0 for the next chain iteration.
        # Length ``mr_worst * topk`` covers the worst-case routing where
        # every recv slot receives ``topk`` distinct partials; the
        # ``use_token_flag_sync=False`` path const_expr-elides every
        # access on this buffer.
        self.device_local_counter = torch.zeros(
            mr_worst * k, dtype=torch.int32, device=self._dev,
        )

    # ------------------------------------------------------------------
    # Config / runtime contract checks (mori parity): fail fast on
    # misuse instead of OOB-writing or aborting deep in JIT codegen.
    # ------------------------------------------------------------------
    def _check_config(self):
        """Static check of ``self.cfg``; runs before any GPU alloc."""
        cfg = self.cfg

        if not isinstance(cfg.rank, int) or cfg.rank < 0:
            raise ValueError(f"rank must be a non-negative int, got {cfg.rank!r}")
        if not isinstance(cfg.world_size, int) or cfg.world_size <= 0:
            raise ValueError(f"world_size must be a positive int, got {cfg.world_size!r}")
        if cfg.rank >= cfg.world_size:
            raise ValueError(f"rank({cfg.rank}) must be < world_size({cfg.world_size})")
        if cfg.world_size > _MAX_INTRANODE_NPES:
            raise ValueError(
                f"world_size={cfg.world_size} exceeds intranode limit "
                f"_MAX_INTRANODE_NPES={_MAX_INTRANODE_NPES} (single-node GPU count); "
                "use an inter-node dispatch/combine op for world_size > 8"
            )

        if cfg.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {cfg.hidden_dim}")
        if cfg.max_num_inp_token_per_rank <= 0:
            raise ValueError(f"max_num_inp_token_per_rank must be positive, got {cfg.max_num_inp_token_per_rank}")
        if cfg.num_experts_per_rank <= 0:
            raise ValueError(f"num_experts_per_rank must be positive, got {cfg.num_experts_per_rank}")
        if cfg.num_experts_per_token <= 0:
            raise ValueError(f"num_experts_per_token must be positive, got {cfg.num_experts_per_token}")
        # k <= 64 is a hard kernel constraint (``ballot`` covers only
        # the first 64 warp lanes); production configs use k <= 16.
        if cfg.num_experts_per_token > 64:
            raise ValueError(f"num_experts_per_token={cfg.num_experts_per_token} exceeds the warp-lane budget (64)")

        if cfg.data_type not in _SUPPORTED_TOK_DTYPES:
            raise ValueError(f"data_type={cfg.data_type} not supported. Supported: {_SUPPORTED_TOK_DTYPES}")
        if cfg.quant_type not in _SUPPORTED_QUANT_TYPES:
            raise ValueError(f"quant_type={cfg.quant_type!r} not supported. Supported: {_SUPPORTED_QUANT_TYPES}")
        if cfg.quant_type == "fp8_direct_cast" and cfg.data_type != torch.bfloat16:
            raise ValueError(
                f"quant_type='fp8_direct_cast' requires data_type=bfloat16 " f"(external dtype), got {cfg.data_type}"
            )
        # std-MoE Stage 1/3 uses ``_weighted_accum_experts`` which has
        # not been retrofitted for asymmetric I/O dtypes, so the two
        # switches are mutually exclusive.
        if cfg.quant_type == "fp8_direct_cast" and cfg.enable_std_moe:
            raise NotImplementedError(
                "quant_type='fp8_direct_cast' is not yet supported together with "
                "enable_std_moe=True (the std-MoE Stage 1 / Stage 3 path uses "
                "weighted-accum helpers that have not been retrofitted for "
                "asymmetric I/O dtypes)"
            )

        # Kernel streams 16B (v4i32) per lane; Stage 3 writes back
        # ``data_type`` so one stride check on ``token_bytes`` suffices.
        if cfg.token_bytes % _TOK_BYTES_ALIGN != 0:
            raise ValueError(
                f"token row bytes ({cfg.token_bytes}) must be a multiple of "
                f"{_TOK_BYTES_ALIGN} for v4i32 vector loads; check hidden_dim "
                f"({cfg.hidden_dim}) and data_type ({cfg.data_type})"
            )

        # ``max_total_recv_tokens`` cap (mori parity): 0 disables;
        # positive values must give every rank >= 1 slot; over-cap is
        # clamped to the worst case (matches mori
        # ``dispatch_combine.cpp:145-151``) -- clamp is implicit in
        # ``effective_max_recv_per_rank``, we only warn here.
        if cfg.max_total_recv_tokens < 0:
            raise ValueError(f"max_total_recv_tokens must be non-negative, got {cfg.max_total_recv_tokens}")
        if cfg.max_total_recv_tokens > 0:
            lo = cfg.world_size
            hi = cfg.world_size * cfg.max_num_inp_token_per_rank
            if cfg.max_total_recv_tokens < lo:
                raise ValueError(
                    f"max_total_recv_tokens={cfg.max_total_recv_tokens} < "
                    f"world_size={lo}; every rank must receive at least one slot"
                )
            if cfg.max_total_recv_tokens > hi:
                import warnings

                warnings.warn(
                    f"max_total_recv_tokens={cfg.max_total_recv_tokens} exceeds the "
                    f"worst case {hi} (= world_size * max_num_inp_token_per_rank); "
                    f"clamping to {hi}.  effective_max_recv_per_rank will be "
                    f"{cfg.max_num_inp_token_per_rank} (M).",
                    stacklevel=2,
                )

        # ``max_token_type_size``: 0 disables; otherwise must cover
        # the dispatch element size (mori-parity dtype-switch knob).
        if cfg.max_token_type_size < 0:
            raise ValueError(f"max_token_type_size must be non-negative, got {cfg.max_token_type_size}")
        if cfg.max_token_type_size > 0:
            inp_es = 1 if cfg.is_fp4 else cfg.elem_size
            if cfg.max_token_type_size < inp_es:
                raise ValueError(
                    f"max_token_type_size={cfg.max_token_type_size} is smaller than the "
                    f"dispatch element size ({inp_es} bytes from data_type="
                    f"{cfg.data_type})"
                )

        if cfg.scale_dim < 0 or cfg.scale_type_size < 0:
            raise ValueError(
                f"scale_dim/scale_type_size must be non-negative, got " f"({cfg.scale_dim}, {cfg.scale_type_size})"
            )
        if (cfg.scale_dim == 0) != (cfg.scale_type_size == 0):
            raise ValueError(
                "scale_dim and scale_type_size must be both zero or both "
                f"positive, got ({cfg.scale_dim}, {cfg.scale_type_size})"
            )

        if cfg.warp_num_per_block <= 0:
            raise ValueError(f"warp_num_per_block must be positive, got {cfg.warp_num_per_block}")
        if cfg.block_num <= 0:
            raise ValueError(f"block_num must be positive, got {cfg.block_num}")
        if cfg.dispatch_warp_num_per_block <= 0:
            raise ValueError(f"dispatch_warp_num_per_block must be positive, got {cfg.dispatch_warp_num_per_block}")
        if cfg.dispatch_block_num <= 0:
            raise ValueError(f"dispatch_block_num must be positive, got {cfg.dispatch_block_num}")
        if cfg.combine_warp_num_per_block <= 0:
            raise ValueError(f"combine_warp_num_per_block must be positive, got {cfg.combine_warp_num_per_block}")
        if cfg.combine_block_num <= 0:
            raise ValueError(f"combine_block_num must be positive, got {cfg.combine_block_num}")

        # expert_id = dest_pe * num_experts_per_rank + local_expert_id
        # must fit i32 to avoid overflow in the ``divui`` decoding.
        total_experts = cfg.world_size * cfg.num_experts_per_rank
        if total_experts > (1 << 31) - 1:
            raise ValueError(
                f"total experts ({cfg.world_size} * {cfg.num_experts_per_rank} = {total_experts}) "
                "exceeds int32 range"
            )

        # LDS budget pre-flight (clearer error than the equivalent
        # ``check_smem_capacity`` failure deep in JIT codegen).
        self._check_lds_capacity()

    def _check_lds_capacity(self):
        """Reject configs whose combine-kernel LDS overflows the GPU."""
        from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP

        cfg = self.cfg

        # Mirror ``make_combine_jit``'s layout: two 8B-aligned
        # i64[npes] tables (tokens + weights), arena padded to 128B.
        # The weight table is always present since ``enable_weights=True``.
        def _align(p, a):
            return (p + a - 1) // a * a

        ptr = 0
        ptr = _align(ptr, 8) + cfg.world_size * 8
        ptr = _align(ptr, 8) + cfg.world_size * 8
        lds_bytes = max(_align(ptr, 128), 128)

        arch = get_rocm_arch()
        limit = SMEM_CAPACITY_MAP.get(arch)
        if limit is not None and lds_bytes > limit:
            raise RuntimeError(
                f"combine kernel LDS layout needs {lds_bytes} bytes "
                f"(2 x i64[world_size={cfg.world_size}] P2P tables + "
                f"128-byte arena alignment), but device {arch} provides "
                f"only {limit} bytes -- reduce world_size or run on a "
                "larger-LDS GPU"
            )

    def _check_tensor_device(self, name, t):
        if not torch.is_tensor(t):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t)}")
        if not t.is_cuda:
            raise ValueError(f"{name} must live on CUDA, got device={t.device}")
        if t.device.index != self.cfg.rank:
            raise ValueError(
                f"{name}.device={t.device} does not match cfg.rank={self.cfg.rank} " f"(expected cuda:{self.cfg.rank})"
            )

    def _check_dispatch_inputs(self, input, weights, scales, indices, packed_recv_x):
        cfg = self.cfg
        self._check_tensor_device("input", input)
        self._check_tensor_device("weights", weights)
        self._check_tensor_device("indices", indices)

        # input: (cur_tok, hidden_dim) or (cur_tok, hidden_dim//2) for fp4.
        if input.dim() != 2:
            raise ValueError(f"input must be 2-D (cur_tok, hidden_dim), got shape {tuple(input.shape)}")
        cur_tok = input.shape[0]
        if cur_tok > cfg.max_num_inp_token_per_rank:
            raise ValueError(
                f"input rows={cur_tok} exceeds cfg.max_num_inp_token_per_rank="
                f"{cfg.max_num_inp_token_per_rank} (would OOB-write into shmem)"
            )
        # Mori-parity launch-time dtype: ``input.dtype`` is no longer
        # tied to ``cfg.data_type`` (which is now a buffer-capacity hint
        # + back-compat default); accept any supported token dtype that
        # fits the per-row shmem allocation budget.
        if input.dtype not in _SUPPORTED_TOK_DTYPES:
            raise ValueError(f"input.dtype={input.dtype} not in supported set {_SUPPORTED_TOK_DTYPES}")
        expected_hdim = _token_view_dim_for(input.dtype, cfg.hidden_dim)
        if input.shape[1] != expected_hdim:
            raise ValueError(
                f"input.shape[1]={input.shape[1]} != expected {expected_hdim} "
                f"(hidden_dim={cfg.hidden_dim}, dtype={input.dtype})"
            )
        # Per-row payload must fit ``max_token_bytes`` (shmem allocation
        # bound).  Bump ``cfg.max_token_type_size`` to lift this when
        # mixing dtypes (e.g. fp4 dispatch + bf16 combine in one op).
        inp_token_bytes = _token_bytes_for(input.dtype, cfg.hidden_dim)
        if inp_token_bytes > cfg.max_token_bytes:
            raise ValueError(
                f"dispatch input.dtype={input.dtype} needs {inp_token_bytes}B/token "
                f"but shmem buffers are sized for {cfg.max_token_bytes}B/token; "
                f"set cfg.max_token_type_size to cover the largest dtype the op handles."
            )

        # weights: (cur_tok, k) f32.
        if weights.dim() != 2:
            raise ValueError(f"weights must be 2-D (cur_tok, k), got shape {tuple(weights.shape)}")
        if weights.shape != (cur_tok, cfg.num_experts_per_token):
            raise ValueError(
                f"weights.shape={tuple(weights.shape)} != expected " f"({cur_tok}, {cfg.num_experts_per_token})"
            )
        if weights.dtype != torch.float32:
            raise ValueError(f"weights.dtype={weights.dtype} must be torch.float32")

        # indices: (cur_tok, k), any signed-int dtype; the wrapper itself
        # already casts to int32 for the kernel.
        if indices.dim() != 2:
            raise ValueError(f"indices must be 2-D (cur_tok, k), got shape {tuple(indices.shape)}")
        if indices.shape != (cur_tok, cfg.num_experts_per_token):
            raise ValueError(
                f"indices.shape={tuple(indices.shape)} != expected " f"({cur_tok}, {cfg.num_experts_per_token})"
            )
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"indices.dtype={indices.dtype} must be int32 or int64")

        # scales: only valid when configured; shape (cur_tok, scale_bytes) bytes.
        if scales is not None:
            self._check_tensor_device("scales", scales)
            if cfg.scale_bytes <= 0:
                raise ValueError(
                    f"scales provided but cfg.scale_dim={cfg.scale_dim} / "
                    f"cfg.scale_type_size={cfg.scale_type_size} disable the scales path"
                )
            if scales.dim() != 2:
                raise ValueError(f"scales must be 2-D, got shape {tuple(scales.shape)}")
            row_bytes = scales.shape[1] * scales.element_size()
            if scales.shape[0] != cur_tok or row_bytes != cfg.scale_bytes:
                raise ValueError(
                    f"scales row-bytes={row_bytes} (shape={tuple(scales.shape)}, "
                    f"elem={scales.element_size()}B) does not match cfg.scale_bytes="
                    f"{cfg.scale_bytes}; expected ({cur_tok}, ...) totalling "
                    f"{cfg.scale_bytes}B per row"
                )

        # packed_recv_x: only meaningful under StdMoE.
        if packed_recv_x is not None:
            self._check_tensor_device("packed_recv_x", packed_recv_x)
            if not cfg.enable_std_moe:
                raise ValueError("packed_recv_x is only consumed when cfg.enable_std_moe=True")
            expected_rows = cfg.num_experts_per_rank * cfg.max_recv
            if packed_recv_x.shape[0] != expected_rows:
                raise ValueError(
                    f"packed_recv_x.shape[0]={packed_recv_x.shape[0]} != "
                    f"num_experts_per_rank * max_recv = {expected_rows}"
                )

    def _check_combine_inputs(self, input, weights, indices, packed_recv_x, strict_input_dtype: bool = True):
        cfg = self.cfg
        self._check_tensor_device("input", input)

        # Combine input is the dispatch out_tok buffer reshaped to
        # (max_recv, hidden_or_packed).  Only the leading dim varies (it
        # depends on ``cur_rank_num_token`` stored in total_recv); the
        # hidden_dim must still match the configured one.
        if input.dim() != 2:
            raise ValueError(f"combine input must be 2-D, got shape {tuple(input.shape)}")
        if strict_input_dtype and input.dtype not in _SUPPORTED_TOK_DTYPES:
            raise ValueError(f"combine input.dtype={input.dtype} not in supported set {_SUPPORTED_TOK_DTYPES}")
        # Mori-parity launch-time dtype: combine accepts any supported
        # token dtype; the row-stride is derived from ``input.dtype``
        # (fp4 packs 2 elements per byte and exposes ``hidden_dim // 2``
        # storage columns).
        view_dtype = (
            input.dtype
            if strict_input_dtype
            else (input.dtype if input.dtype in _SUPPORTED_TOK_DTYPES else cfg.data_type)
        )
        expected_hdim = _token_view_dim_for(view_dtype, cfg.hidden_dim)
        if input.shape[1] != expected_hdim:
            raise ValueError(
                f"combine input.shape[1]={input.shape[1]} != expected "
                f"{expected_hdim} (hidden_dim={cfg.hidden_dim}, dtype={view_dtype})"
            )
        if input.shape[0] > cfg.max_recv:
            raise ValueError(f"combine input rows={input.shape[0]} exceeds max_recv={cfg.max_recv}")
        if strict_input_dtype:
            inp_token_bytes = _token_bytes_for(input.dtype, cfg.hidden_dim)
            if inp_token_bytes > cfg.max_token_bytes:
                raise ValueError(
                    f"combine input.dtype={input.dtype} needs {inp_token_bytes}B/token "
                    f"but shmem buffers are sized for {cfg.max_token_bytes}B/token; "
                    f"set cfg.max_token_type_size to cover the largest dtype the op handles."
                )

        if weights is not None:
            self._check_tensor_device("weights", weights)
            if weights.dim() != 2 or weights.shape[1] != cfg.num_experts_per_token:
                raise ValueError(
                    f"combine weights must be (max_recv, {cfg.num_experts_per_token}), "
                    f"got shape {tuple(weights.shape)}"
                )
            if weights.dtype != torch.float32:
                raise ValueError(f"combine weights.dtype={weights.dtype} must be torch.float32")

        if indices is not None:
            self._check_tensor_device("indices", indices)
            if indices.dim() != 2 or indices.shape[1] != cfg.num_experts_per_token:
                raise ValueError(
                    f"combine indices must be (max_recv, {cfg.num_experts_per_token}), "
                    f"got shape {tuple(indices.shape)}"
                )
            if indices.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"combine indices.dtype={indices.dtype} must be int32/int64")

        if packed_recv_x is not None:
            self._check_tensor_device("packed_recv_x", packed_recv_x)
            if not cfg.enable_std_moe:
                raise ValueError("packed_recv_x is only consumed when cfg.enable_std_moe=True")

    def _get_dispatch_jit(self, d_dtype):
        """Lazy-jit a dispatch kernel specialized to ``d_dtype`` (mori
        parity).  Cache miss compiles a fresh entry; subsequent calls
        with the same dtype reuse it.  The flyc.compile second-stage
        cache is also keyed by dtype to keep cudagraphs from different
        dtypes from clobbering each other."""
        if d_dtype not in self._disp_jit_cache:
            cfg = self.cfg
            self._disp_jit_cache[d_dtype] = make_dispatch_jit(
                rank=cfg.rank,
                npes=cfg.world_size,
                experts_per_rank=cfg.num_experts_per_rank,
                experts_per_token=cfg.num_experts_per_token,
                hidden_dim=cfg.hidden_dim,
                max_tok_per_rank=cfg.max_num_inp_token_per_rank,
                block_num=cfg.dispatch_block_num_eff,
                warp_num_per_block=cfg.dispatch_warp_num_per_block_eff,
                data_type=d_dtype,
                scale_dim=cfg.scale_dim,
                scale_type_size=cfg.scale_type_size,
                enable_std_moe=cfg.enable_std_moe,
                max_recv=self._effective_max_recv,
                use_token_flag_sync=cfg.use_token_flag_sync,
            )
        return self._disp_jit_cache[d_dtype]

    def dispatch(self, input, weights, scales, indices, packed_recv_x=None):
        self._check_dispatch_inputs(input, weights, scales, indices, packed_recv_x)
        cfg = self.cfg
        d_dtype = input.dtype
        inp_cur_tok = input.shape[0]
        # Stash for ``combine()``'s default ``cur_tok`` (mori
        # ``args.curRankNumToken``: dispatch input count, NOT the
        # receive-side buffer shape).
        self._last_inp_cur_tok = inp_cur_tok
        stream = torch.cuda.current_stream()
        inp_c = input if input.is_contiguous() else input.contiguous()
        wts_c = weights if weights.is_contiguous() else weights.contiguous()
        idx_c = (
            indices
            if (indices.dtype == torch.int32 and indices.is_contiguous())
            else indices.to(torch.int32).contiguous()
        )

        # D-flag C-1: stash the contiguous input-weights pointer for the
        # Stage 3b local-read path inside ``combine_no_stage1`` (bypasses
        # the Stage 1 weight P2P scatter when ``use_token_flag_sync`` is
        # ON).  ``wts_c`` is contiguous and pinned to ``[max_tok_per_rank,
        # topk] f32`` layout so ``(src_tok, lane)`` indexing matches the
        # mori out_wts contract.
        self._raw_input_wts_ptr = wts_c.data_ptr()

        sc_ptr = scales.data_ptr() if scales is not None else 0
        prx_ptr = packed_recv_x.data_ptr() if packed_recv_x is not None else 0

        if cfg.enable_std_moe:
            self.packed_recv_count.zero_()
            # ``disp_grid_bar`` is intentionally NOT reset: the kernel
            # uses an ``atomic_add`` ticket to derive each launch's
            # epoch (see StdMoE Phase 4 in ``make_dispatch_kernel``).

        # _std_args layout MUST match the trailing 8 ``addr_*`` params
        # of ``dispatch_launch`` (JIT slots 17..24 in order):
        #   17 addr_shmem_tok
        #   18 addr_shmem_idx
        #   19 addr_shmem_tok_id_to_src
        #   20 addr_out_packed_recv_x
        #   21 addr_out_packed_recv_count
        #   22 addr_out_packed_recv_src_info
        #   23 addr_out_disp_tok_map
        #   24 addr_disp_grid_bar
        _std_args = (
            self._fx_out_tok,
            self._fx_out_idx,
            self._fx_out_shmem_tok_id_to_src,
            fx.Int64(prx_ptr),
            self._fx_packed_recv_count if cfg.enable_std_moe else fx.Int64(0),
            self._fx_packed_recv_src_info,
            self._fx_disp_tok_map,
            self._fx_disp_grid_bar,
        )

        disp_fn = self._get_dispatch_jit(d_dtype)
        disp_compiled = self._disp_compiled_cache.get(d_dtype)
        if disp_compiled is None:
            args = (
                fx.Int64(inp_c.data_ptr()),
                fx.Int64(idx_c.data_ptr()),
                fx.Int64(wts_c.data_ptr()),
                self._fx_tok_map,
                self._fx_tok_off,
                self._fx_dest_ctr,
                self._fx_disp_bar,
                self._fx_recv_num,
                self._fx_out_total_recv,
                self._fx_p2p_tok_off,
                self._fx_p2p_out_tok,
                self._fx_p2p_out_tok_id_to_src,
                self._fx_p2p_out_idx,
                self._fx_p2p_out_wts,
                self._fx_p2p_recv_num,
                fx.Int64(sc_ptr),
                self._fx_p2p_out_scales,
                *_std_args,
                self._fx_comb_flag,
                inp_cur_tok,
                stream,
            )
            disp_compiled = flyc.compile(disp_fn, *args)
            self._disp_compiled_cache[d_dtype] = disp_compiled
        else:
            disp_compiled(
                inp_c.data_ptr(),
                idx_c.data_ptr(),
                wts_c.data_ptr(),
                self._fx_tok_map,
                self._fx_tok_off,
                self._fx_dest_ctr,
                self._fx_disp_bar,
                self._fx_recv_num,
                self._fx_out_total_recv,
                self._fx_p2p_tok_off,
                self._fx_p2p_out_tok,
                self._fx_p2p_out_tok_id_to_src,
                self._fx_p2p_out_idx,
                self._fx_p2p_out_wts,
                self._fx_p2p_recv_num,
                sc_ptr,
                self._fx_p2p_out_scales,
                *_std_args,
                self._fx_comb_flag,
                inp_cur_tok,
                stream,
            )

        # Match the allocation size in ``_alloc_buffers`` (tighter
        # than ``max_recv`` when the cap is set).  Output view uses
        # launch-time dtype (mori parity), not ``cfg.data_type``.
        mr = cfg.effective_max_recv
        k = cfg.num_experts_per_token

        out_token_bytes = _token_bytes_for(d_dtype, cfg.hidden_dim)
        out_view_dim = _token_view_dim_for(d_dtype, cfg.hidden_dim)
        out_tok = self.shmem_disp_out_tok.view(torch.int8)[: mr * out_token_bytes].view(d_dtype).view(mr, out_view_dim)
        out_wts = self.shmem_disp_out_wts.view(mr, k)
        out_idx = self.shmem_disp_out_idx.view(mr, k)
        out_scales = None
        if cfg.scale_bytes > 0:
            out_scales = self.shmem_out_scales[: mr * cfg.scale_bytes].view(mr, cfg.scale_dim * cfg.scale_type_size)

        result = (out_tok, out_wts, out_scales, out_idx, self.total_recv)
        if cfg.enable_std_moe:
            epr = cfg.num_experts_per_rank
            result = result + (
                self.packed_recv_count[:epr],
                self.packed_recv_src_info,
            )
        return result

    def _get_combine_jit(self, c_dtype, zero_copy, enable_weights_flag, fp8_dc):
        """Lazy-jit a combine kernel specialized to the launch-time
        ``(c_dtype, zero_copy, enable_weights, fp8_direct_cast)`` tuple
        (mori parity).  Zero-copy mode hard-wires ``skip_stage1=True`` --
        the caller is contractually responsible for staging into
        ``shmem_comb_inp_tok`` upstream, so the kernel must NOT
        double-write that buffer."""
        key = (c_dtype, bool(zero_copy), bool(enable_weights_flag), bool(fp8_dc))
        if key not in self._comb_jit_cache:
            cfg = self.cfg
            self._comb_jit_cache[key] = make_combine_jit(
                rank=cfg.rank,
                npes=cfg.world_size,
                experts_per_token=cfg.num_experts_per_token,
                hidden_dim=cfg.hidden_dim,
                max_tok_per_rank=cfg.max_num_inp_token_per_rank,
                block_num=cfg.combine_block_num_eff,
                warp_num_per_block=cfg.combine_warp_num_per_block_eff,
                data_type=c_dtype,
                enable_weights=bool(enable_weights_flag),
                enable_std_moe=cfg.enable_std_moe,
                zero_copy=bool(zero_copy),
                # Mori-parity zero-copy: caller has pre-staged token bytes
                # into ``shmem_comb_inp_tok`` via
                # :meth:`get_registered_combine_input_buffer` (contract
                # enforced above).  ``skip_stage1=True`` removes the
                # kernel-side Stage 1 token copy; the kernel still copies
                # ``shmem_inp_wts -> shmem_comb_inp_wts`` because mori
                # keeps that block (``intranode.hpp:297-306``) and Stage
                # 3b reads weights via zero-copy by ``recv_tok_id``.  The
                # Stage 3 fused-upstream layout is gated on
                # ``not zero_copy`` in the kernel, so this combination
                # keeps mori's regular ``(dest_pe, dest_lid)`` decode.
                skip_stage1=bool(zero_copy),
                fp8_direct_cast=bool(fp8_dc),
                max_recv=self._effective_max_recv,
                use_token_flag_sync=cfg.use_token_flag_sync,
            )
        return self._comb_jit_cache[key], key

    def combine(
        self,
        input,
        weights,
        indices,
        packed_recv_x=None,
        cur_tok=None,
    ):
        """Intranode combine entry point.

        Launch-time dtype (mori parity): ``input.dtype`` selects the
        kernel specialization on each call; the op maintains a JIT
        cache keyed by ``(input.dtype, zero_copy, enable_weights,
        fp8_direct_cast)``.

        Zero-copy mode (``cfg.zero_copy=True``) requires the
        caller to write data into the buffer returned by
        ``get_registered_combine_input_buffer(combine_dtype)`` BEFORE
        calling ``combine()`` -- the kernel runs ``skip_stage1=True``
        and the local staging copy is removed entirely (mori parity).
        Passing a non-shmem tensor in this mode raises ``ValueError``.

        Launch geometry and the zero-copy switch remain frozen by
        ``self.cfg``; rebuild the op (``EpDispatchCombineOp(new_cfg)``)
        to change them. Geometry can be split per phase via
        ``dispatch_block_num``/``dispatch_warp_num_per_block`` and
        ``combine_block_num``/``combine_warp_num_per_block`` (all
        optional, defaulting to legacy shared values).
        """
        self._check_combine_inputs(input, weights, indices, packed_recv_x)
        cfg = self.cfg
        stream = torch.cuda.current_stream()

        c_dtype = input.dtype
        zero_copy = cfg.zero_copy
        # ``enable_weights`` mirrors the historical default at op
        # construction; combine_no_stage1 is the only entry point that
        # exposes weight-free combines today.
        enable_weights_flag = True
        # fp8_direct_cast is now a launch-time decision: caller may pass
        # bf16 even when the op was configured with quant_type='none', so
        # the gate fires only when both the config knob asks for it AND
        # the launch dtype is bf16 (mori UseFp8DirectCast parity).
        fp8_dc = cfg.quant_type == "fp8_direct_cast" and c_dtype == torch.bfloat16
        if zero_copy and fp8_dc:
            # In zero-copy mode Stage 1 is removed; there is no in-kernel
            # cast site for the bf16->fp8 conversion that fp8_direct_cast
            # relies on, so the two switches are mutually exclusive.
            raise ValueError(
                "fp8_direct_cast is incompatible with zero_copy=True "
                "(zero_copy mode skips Stage 1, where the bf16->fp8 cast lives)."
            )
        if zero_copy and input.data_ptr() != self.shmem_comb_inp_tok.data_ptr():
            # Mori-parity caller contract: in zero-copy mode the kernel
            # peer-reads ``shmem_comb_inp_tok``; the caller MUST write
            # data into the buffer returned by
            # ``get_registered_combine_input_buffer(combine_dtype)``
            # before calling ``combine``.  Any other tensor pointer is
            # a silent correctness bug (peers would read stale shmem).
            raise ValueError(
                "zero_copy mode requires the caller to "
                "write data into the buffer returned by "
                "op.get_registered_combine_input_buffer(combine_dtype) and pass "
                "that view as the combine input.  Got input.data_ptr()="
                f"{input.data_ptr():#x} but shmem_comb_inp_tok.data_ptr()="
                f"{self.shmem_comb_inp_tok.data_ptr():#x}."
            )

        inp_c = input if input.is_contiguous() else input.contiguous()

        # ``cur_tok`` = Stage 3 iteration bound (mori parity:
        # ``args.curRankNumToken``).  Default to the last
        # ``dispatch()`` input count -- the same dispatch wrote
        # ``dest_tok_map[0:cur_tok*k)``, so using a larger value would
        # read tail residue from a prior (larger) dispatch.  Callers
        # bypassing dispatch (fused GEMM2+combine) must pass it.
        if cur_tok is None:
            stashed = getattr(self, "_last_inp_cur_tok", None)
            if stashed is None:
                raise ValueError(
                    "combine() requires either an explicit cur_tok or a "
                    "preceding dispatch() call on the same op (cur_tok "
                    "defaults to the dispatch input.shape[0])."
                )
            _cur_tok = stashed
        else:
            _cur_tok = cur_tok
        if _cur_tok < 0 or _cur_tok > cfg.max_num_inp_token_per_rank:
            raise ValueError(
                f"cur_tok={_cur_tok} out of range " f"[0, max_num_inp_token_per_rank={cfg.max_num_inp_token_per_rank}]"
            )

        wts_ptr = self.shmem_disp_out_wts.data_ptr() if weights is None else weights.data_ptr()

        _prx_ref = None
        if fp8_dc and packed_recv_x is not None:
            # std-MoE expert-major buffer is bf16 upstream but Stage 1
            # reads it in fp8 dtype; cast here (independent from the
            # main combine input path).
            _prx_ref = packed_recv_x.view(torch.bfloat16).to(torch.float8_e4m3fn).contiguous()
            prx_ptr = _prx_ref.data_ptr()
        else:
            prx_ptr = packed_recv_x.data_ptr() if packed_recv_x is not None else 0

        _std_args_comb = (
            fx.Int64(prx_ptr),
            self._fx_disp_tok_map,
            self._fx_disp_out_wts,
        )

        comb_fn, comb_key = self._get_combine_jit(c_dtype, zero_copy, enable_weights_flag, fp8_dc)
        comb_compiled = self._comb_compiled_cache.get(comb_key)
        if comb_compiled is None:
            args = (
                fx.Int64(inp_c.data_ptr()),
                self._fx_comb_inp,
                self._fx_comb_out,
                self._fx_xdb_mem,
                self._fx_xdev_flag,
                self._fx_tok_map,
                self._fx_comb_bar,
                self._fx_trecv,
                self._fx_out_shmem_tok_id_to_src,
                self._fx_p2p_comb_inp,
                self._fx_p2p_xdb_mem,
                fx.Int64(wts_ptr),
                self._fx_comb_inp_wts,
                self._fx_comb_out_wts,
                self._fx_p2p_comb_inp_wts,
                *_std_args_comb,
                self._fx_comb_flag,
                _cur_tok,
                stream,
            )
            comb_compiled = flyc.compile(comb_fn, *args)
            self._comb_compiled_cache[comb_key] = comb_compiled
        else:
            comb_compiled(
                inp_c.data_ptr(),
                self._fx_comb_inp,
                self._fx_comb_out,
                self._fx_xdb_mem,
                self._fx_xdev_flag,
                self._fx_tok_map,
                self._fx_comb_bar,
                self._fx_trecv,
                self._fx_out_shmem_tok_id_to_src,
                self._fx_p2p_comb_inp,
                self._fx_p2p_xdb_mem,
                wts_ptr,
                self._fx_comb_inp_wts,
                self._fx_comb_out_wts,
                self._fx_p2p_comb_inp_wts,
                prx_ptr,
                self._fx_disp_tok_map,
                self._fx_disp_out_wts,
                self._fx_comb_flag,
                _cur_tok,
                stream,
            )

        mt = cfg.max_num_inp_token_per_rank
        k = cfg.num_experts_per_token

        out_token_bytes = _token_bytes_for(c_dtype, cfg.hidden_dim)
        out_view_dim = _token_view_dim_for(c_dtype, cfg.hidden_dim)
        out_tok = self.shmem_comb_out_tok.view(torch.int8)[: mt * out_token_bytes].view(c_dtype).view(mt, out_view_dim)
        out_wts = self.shmem_comb_out_wts.view(mt, k)

        return out_tok, out_wts

    # Gate for the reserved ``combine_no_stage1`` API: only the fused
    # GEMM2+combine path meets the contract (pre-populated
    # ``shmem_comb_inp_tok``, caller-supplied ``cur_tok``, upstream IPC
    # ordering vs. the weight P2P).  Default-open on this branch
    # (``fused_gemm2_combine_new``) because the fused launcher is the
    # primary consumer; a dispatch_combine-only deployment can still
    # gate this off by overriding the class attribute on a subclass.
    _ENABLE_COMBINE_NO_STAGE1 = True

    def combine_no_stage1(self, input, weights, indices, packed_recv_x=None, cur_tok=None, enable_weights: bool = True):
        """Skip-Stage1 combine (reserved for fused GEMM2+combine path).

        Bypasses the P2P scatter (upstream fused kernel has already
        populated ``shmem_comb_inp[_wts]``) and runs only Stage 2
        (CrossDeviceBarrier) + Stage 3 (local weighted-accum).  Gated
        by ``_ENABLE_COMBINE_NO_STAGE1``; raises ``NotImplementedError``
        otherwise.

        ``cur_tok``: real Stage 3 iteration count (not ``self.total_recv``).
        ``enable_weights=True`` keeps the Stage 1 weight scatter +
        Stage 3b inside combine (the upstream GEMM2 epilogue cannot
        emit them: 16B narrow stores get dropped by the ROCm IPC
        fabric when they race the upstream token P2P).
        ``False`` DCEs both weight steps for weight-free fused paths
        (saves ~3-5 us); cached separately by ``enable_weights``.

        Contract (caller-owned):
          - ``shmem_comb_inp_tok`` holds every token this PE consumes
            (max_tok_per_rank slots);
          - ``shmem_comb_inp_wts`` holds matching weights when
            ``enable_weights=True``;
          - ``cur_tok`` matches the real receive count.
        """
        if not type(self)._ENABLE_COMBINE_NO_STAGE1:
            raise NotImplementedError(
                "FlyDSLDispatchCombineIntraNodeOp.combine_no_stage1 is a "
                "reserved API for the fused GEMM2+combine production path "
                "(see yanbo/fused_gemm2_combine branch).  The standalone "
                "intranode dispatch/combine wrapper does not yet validate "
                "this entry point — invoke combine(...) instead.  To enable "
                "from the fused branch, set "
                "FlyDSLDispatchCombineIntraNodeOp._ENABLE_COMBINE_NO_STAGE1 "
                "= True after auditing the upstream IPC-ordering contract "
                "documented in combine_no_stage1.__doc__."
            )

        # Kernel doesn't actually read ``input`` (Stage 1 bypassed), so
        # we only check shape/device and relax the strict dtype contract.
        self._check_combine_inputs(input, weights, indices, packed_recv_x, strict_input_dtype=False)
        cfg = self.cfg
        stream = torch.cuda.current_stream()

        # Launch-time fp8_direct_cast gate (mori parity): only fires
        # when the config asked for it AND the caller-passed dtype is
        # bf16 (the dtype combination the wire-fp8 path was built for).
        fp8_dc = cfg.quant_type == "fp8_direct_cast" and input.dtype == torch.bfloat16
        # ``input`` is unread under skip_stage1; the Python-level fp8
        # cast would still sit on the cudagraph critical path (~12us
        # for nothing), so the fused caller is expected to have
        # CV-casted in the GEMM2 epilogue already.
        if fp8_dc and input.dtype != torch.float8_e4m3fn:
            inp_c = input.to(torch.float8_e4m3fn).contiguous()
        else:
            inp_c = input if input.is_contiguous() else input.contiguous()
        # Combine kernel is parametrised on ``data_type`` (fp4/fp8/bf16/...);
        # under fp8_direct_cast the wire dtype is fp8 even when the
        # caller passes bf16, otherwise it tracks the input dtype.
        c_dtype = torch.float8_e4m3fn if fp8_dc else input.dtype

        # ``input`` is a placeholder under skip_stage1, so we can't
        # infer cur_tok from its shape.  Preference: explicit arg >
        # stashed dispatch input count > error.
        if cur_tok is not None:
            _cur_tok = cur_tok
        else:
            stashed = getattr(self, "_last_inp_cur_tok", None)
            if stashed is None:
                raise ValueError(
                    "combine_no_stage1() requires an explicit cur_tok "
                    "when no prior dispatch() has been issued on this op "
                    "(the fused GEMM2+combine path populates "
                    "shmem_comb_inp directly, so dispatch is bypassed)."
                )
            _cur_tok = stashed
        if _cur_tok < 0 or _cur_tok > cfg.max_num_inp_token_per_rank:
            raise ValueError(
                f"cur_tok={_cur_tok} out of range " f"[0, max_num_inp_token_per_rank={cfg.max_num_inp_token_per_rank}]"
            )

        # D-flag C-1: when ``use_token_flag_sync`` is ON Stage 3b reads
        # weights directly from the local raw input weights buffer
        # ``[max_tok_per_rank, topk] f32`` instead of the P2P-scattered
        # ``shmem_disp_out_wts`` -- so the launcher swaps in the cached
        # ``_raw_input_wts_ptr`` (populated by the most-recent
        # ``dispatch()`` call).  Caller-provided ``weights`` always wins
        # if present (matches baseline semantics: e.g. the fused tester
        # passes a fresh weight tensor for verification runs).
        if cfg.use_token_flag_sync and weights is None:
            if self._raw_input_wts_ptr == 0:
                raise RuntimeError(
                    "combine_no_stage1(use_token_flag_sync=True) requires a "
                    "preceding dispatch() so ``_raw_input_wts_ptr`` is "
                    "populated; no dispatch has been observed on this op yet."
                )
            wts_ptr = self._raw_input_wts_ptr
        else:
            wts_ptr = self.shmem_disp_out_wts.data_ptr() if weights is None else weights.data_ptr()

        _prx_ref = None
        if fp8_dc and packed_recv_x is not None:
            _prx_ref = packed_recv_x.view(torch.bfloat16).to(torch.float8_e4m3fn).contiguous()
            prx_ptr = _prx_ref.data_ptr()
        else:
            prx_ptr = packed_recv_x.data_ptr() if packed_recv_x is not None else 0

        # Cache key (mori parity): a runtime dtype switch (e.g. fp4 ->
        # bf16) must select a fresh specialization rather than reuse
        # the first-seen kernel.
        no_s1_key = (c_dtype, bool(enable_weights))

        if no_s1_key not in self._comb_no_s1_fn:
            from .dispatch_combine_intranode_kernel import make_combine_jit

            self._comb_no_s1_fn[no_s1_key] = make_combine_jit(
                rank=cfg.rank,
                npes=cfg.world_size,
                experts_per_token=cfg.num_experts_per_token,
                hidden_dim=cfg.hidden_dim,
                max_tok_per_rank=cfg.max_num_inp_token_per_rank,
                block_num=cfg.combine_block_num_eff,
                warp_num_per_block=cfg.combine_warp_num_per_block_eff,
                data_type=c_dtype,
                enable_weights=bool(enable_weights),
                enable_std_moe=cfg.enable_std_moe,
                zero_copy=cfg.zero_copy,
                skip_stage1=True,
                fp8_direct_cast=bool(fp8_dc),
                # Must match dispatch's encoding stride so the
                # tok_map sentinel + (peer_pe, dest_lid) decode line up
                # when the cap shrinks the recv buffer.
                max_recv=self._effective_max_recv,
                use_token_flag_sync=cfg.use_token_flag_sync,
            )

        if no_s1_key not in self._comb_no_s1_compiled:
            args = (
                fx.Int64(inp_c.data_ptr()),
                self._fx_comb_inp,
                self._fx_comb_out,
                self._fx_xdb_mem,
                self._fx_xdev_flag,
                self._fx_tok_map,
                self._fx_comb_bar,
                self._fx_trecv,
                self._fx_out_shmem_tok_id_to_src,
                self._fx_p2p_comb_inp,
                self._fx_p2p_xdb_mem,
                fx.Int64(wts_ptr),
                self._fx_comb_inp_wts,
                self._fx_comb_out_wts,
                self._fx_p2p_comb_inp_wts,
                fx.Int64(prx_ptr),
                self._fx_disp_tok_map,
                self._fx_disp_out_wts,
                self._fx_comb_flag,
                _cur_tok,
                stream,
            )
            self._comb_no_s1_compiled[no_s1_key] = flyc.compile(self._comb_no_s1_fn[no_s1_key], *args)
        else:
            self._comb_no_s1_compiled[no_s1_key](
                inp_c.data_ptr(),
                self._fx_comb_inp,
                self._fx_comb_out,
                self._fx_xdb_mem,
                self._fx_xdev_flag,
                self._fx_tok_map,
                self._fx_comb_bar,
                self._fx_trecv,
                self._fx_out_shmem_tok_id_to_src,
                self._fx_p2p_comb_inp,
                self._fx_p2p_xdb_mem,
                wts_ptr,
                self._fx_comb_inp_wts,
                self._fx_comb_out_wts,
                self._fx_p2p_comb_inp_wts,
                prx_ptr,
                self._fx_disp_tok_map,
                self._fx_disp_out_wts,
                self._fx_comb_flag,
                _cur_tok,
                stream,
            )

        mt = cfg.max_num_inp_token_per_rank
        k = cfg.num_experts_per_token

        out_token_bytes = _token_bytes_for(c_dtype, cfg.hidden_dim)
        out_view_dim = _token_view_dim_for(c_dtype, cfg.hidden_dim)
        out_tok = self.shmem_comb_out_tok.view(torch.int8)[: mt * out_token_bytes].view(c_dtype).view(mt, out_view_dim)
        out_wts = self.shmem_comb_out_wts.view(mt, k)

        return out_tok, out_wts

    def get_dispatch_src_token_pos(self):
        torch.cuda.synchronize()
        n = int(self.total_recv[0].item())
        return self.shmem_tok_id_to_src[:n].clone()

    def get_registered_combine_input_buffer(self, dtype=None, hidden_dim=-1):
        """Return the shmem ``comb_inp_tok`` buffer viewed as ``dtype``
        (mori parity).  ``dtype=None`` falls back to ``cfg.data_type``;
        ``hidden_dim<=0`` derives the trailing dim from the dtype
        (fp4 packs 2 elements per byte and exposes ``hidden_dim // 2``).

        In zero-copy mode (``cfg.zero_copy=True``) the caller
        MUST write its combine input into this view before invoking
        ``op.combine(...)``; the kernel skips Stage 1 entirely and peers
        read this buffer directly.
        """
        cfg = self.cfg
        dt = dtype if dtype is not None else cfg.data_type
        if dt not in _SUPPORTED_TOK_DTYPES:
            raise ValueError(
                f"get_registered_combine_input_buffer: dtype={dt} not in " f"supported set {_SUPPORTED_TOK_DTYPES}"
            )
        # Capacity guard: dtype must fit the per-row shmem allocation.
        row_bytes = _token_bytes_for(dt, cfg.hidden_dim)
        if row_bytes > cfg.max_token_bytes:
            raise ValueError(
                f"get_registered_combine_input_buffer: dtype={dt} needs "
                f"{row_bytes}B/token but the buffer is sized for "
                f"{cfg.max_token_bytes}B/token; bump cfg.max_token_type_size."
            )
        h = hidden_dim if hidden_dim > 0 else _token_view_dim_for(dt, cfg.hidden_dim)
        return self.shmem_comb_inp_tok.view(torch.int8).view(dt).view(-1, h)
