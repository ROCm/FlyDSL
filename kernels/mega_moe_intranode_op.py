# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Python host wrapper for the FlyDSL intra-node ``mega_moe`` op.

This is the host-side surface of the FlyDSL port of DeepGEMM's
``fp8_fp4_mega_moe`` (NVIDIA SM100) onto AMD MI350 (gfx950/CDNA4). It
mirrors DeepGEMM's symmetric-buffer byte layout exactly (see
``DeepGEMM/csrc/apis/mega.hpp`` and
``DeepGEMM/deep_gemm/include/deep_gemm/layout/mega_moe.cuh``) so the
kernel can reuse the same ``Workspace`` / ``Buffer`` offset math, but the
symmetric allocation and P2P peer addressing go through mori shmem
instead of ``torch.distributed._symmetric_memory``.

The whole symmetric region is a single mori int8 allocation; every
sub-segment (workspace counters, input pools, L1/L2 activation pools,
combine staging) is a byte offset into it. A peer therefore only needs
one base-pointer table (``shmem_ptr_p2p`` of the buffer head per PE), and
any remote segment address is ``peer_base[pe] + offset``.

Milestone scope: this round implements the host only -- buffer layout,
views, peer pointers, validation, and a temporary dispatch-output debug
interface. ``transform_weights_for_mega_moe`` is stubbed and the kernel
launch lazily imports the (not-yet-written) dispatch kernel builder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import mori.shmem as ms
import torch
from mori.shmem import mori_shmem_create_tensor

import flydsl.compiler as flyc
import flydsl.expr as fx

# Candidate per-expert BLOCK_M values; mirror
# ``layout::kCandidateBlockM`` in mega_moe.cuh. The pool token count is
# aligned to their LCM so any BLOCK_M divides it evenly.
_KCANDIDATE_BLOCK_M = (8, 16, 32, 64, 96, 128, 192)
_KMAX_CANDIDATE_BLOCK_M = 192
_KMIN_CANDIDATE_BLOCK_M = 8
_KLCM_CANDIDATE_BLOCK_M = 384  # token alignment (get_token_alignment_for_mega_moe)

# Barrier head (DeepGEMM sense-reversal scheme), as i32 slots from
# ``barrier_off``:
#   +0,+4,+8,+12,+16,+20  six per-site grid-sync counters (#1-#6a)
#   +24                   NVLink cross-rank barrier counter (low 2 bits)
#   +28,+32               NVLink signal slots (phase 0 / phase 1)
#   +36                   combine epilogue grid-sync counter (#6b epilogue)
#   +40                   kernel-tail reset grid-sync counter
#   +44                   kernel-tail reset NVLink barrier counter (low 2 bits)
#   +48,+52               kernel-tail reset NVLink signal slots (phase 0 / 1)
#   +56,+60               spare/padding
# All sense-reversal / phase-flipping, so they are reset-free across calls.
_NUM_BARRIER_SIGNAL_BYTES = 64
# ``TokenSrcMetadata`` = 3 x uint32 (rank_idx, token_idx, topk_idx).
_TOKEN_SRC_METADATA_BYTES = 12

_MAX_INTRANODE_NPES = 8

# Wire dtype for dispatched tokens (OCP FP8 E4M3; gfx950 native).
_FP8_DTYPE = torch.float8_e4m3fn

_SUPPORTED_ACTIVATIONS = ("swiglu",)


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _align(x: int, y: int) -> int:
    return _ceil_div(x, y) * y


def get_token_alignment_for_mega_moe() -> int:
    """Parity with ``_C.get_token_alignment_for_mega_moe()``."""
    return _KLCM_CANDIDATE_BLOCK_M


def _num_max_pool_tokens(num_ranks: int, num_max_tokens_per_rank: int, num_topk: int,
                         num_experts_per_rank: int) -> int:
    """Mirror ``layout::get_num_max_pool_tokens``: worst-case received
    tokens plus per-expert BLOCK_M padding, aligned to the candidate LCM."""
    num_max_recv_tokens = num_ranks * num_max_tokens_per_rank
    num_max_experts_per_token = min(num_topk, num_experts_per_rank)
    return _align(
        num_max_recv_tokens * num_max_experts_per_token
        + num_experts_per_rank * (_KMAX_CANDIDATE_BLOCK_M - 1),
        _KLCM_CANDIDATE_BLOCK_M,
    )


def _num_padded_sf_pool_tokens(num_max_pool_tokens: int, block_m: int) -> int:
    """Mirror ``layout::get_num_padded_sf_pool_tokens``."""
    return (num_max_pool_tokens // block_m) * _align(block_m, 128)


@dataclass
class MegaMoEIntraNodeConfig:
    """Static configuration for the intra-node mega_moe op.

    ``num_max_tokens_per_rank`` is the raw request; the op aligns it to
    ``get_token_alignment_for_mega_moe()`` (384) before sizing buffers,
    matching ``get_symm_buffer_for_mega_moe``.
    """

    rank: int
    world_size: int
    num_experts: int
    num_topk: int
    hidden: int
    intermediate_hidden: int
    num_max_tokens_per_rank: int
    use_fp8_dispatch: bool = True
    activation: str = "swiglu"
    # Per-expert pool tile (one value baked into the kernel; mirrors
    # DeepGEMM's single JIT-time BLOCK_M, which is NOT user-facing -- it is
    # chosen by a heuristic, see get_block_config_for_mega_moe). Kept as an
    # internal config knob so dispatch (pool packing + M-major SF transpose)
    # and the (future) compute tile stay in lockstep. The auto-select
    # heuristic lands later; until then 64 is the default -- measured fastest
    # on gfx950 across the default decode/prefill shapes (~3.5% over 128;
    # DeepGEMM's NV-tuned table picks larger values that are NOT optimal here,
    # see the gfx950 block_m sweep). Override per-shape via this field.
    block_m: int = 64

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_experts // self.world_size

    @property
    def aligned_num_max_tokens_per_rank(self) -> int:
        return _align(self.num_max_tokens_per_rank, _KLCM_CANDIDATE_BLOCK_M)

    @property
    def num_max_recv_tokens_per_expert(self) -> int:
        return self.world_size * self.aligned_num_max_tokens_per_rank

    @property
    def num_max_pool_tokens(self) -> int:
        return _num_max_pool_tokens(
            self.world_size, self.aligned_num_max_tokens_per_rank,
            self.num_topk, self.num_experts_per_rank,
        )

    @property
    def num_max_pool_blocks(self) -> int:
        return self.num_max_pool_tokens // _KMIN_CANDIDATE_BLOCK_M

    @property
    def num_max_padded_sf_pool_tokens(self) -> int:
        return max(
            _num_padded_sf_pool_tokens(self.num_max_pool_tokens, bm)
            for bm in _KCANDIDATE_BLOCK_M
        )


@dataclass(frozen=True)
class _MegaMoELayout:
    """Byte offsets and sizes of every symmetric-buffer segment.

    Built once per op from the (aligned) config. Workspace segments come
    first (mirroring ``Workspace::get_num_bytes`` accumulation order),
    followed by the input / L1 / L2 / combine data buffers (mirroring the
    ``Buffer`` chain in ``get_symm_buffer_size_for_mega_moe``).
    """

    # Workspace segment offsets.
    barrier_off: int
    expert_send_count_off: int
    expert_recv_count_off: int
    expert_recv_count_sum_off: int
    l1_arrival_count_off: int
    l2_arrival_mask_off: int
    src_token_topk_idx_off: int
    token_src_metadata_off: int
    workspace_bytes: int

    # Data buffer offsets.
    input_token_off: int
    input_sf_off: int
    input_topk_idx_off: int
    input_topk_weights_off: int
    l1_token_off: int
    l1_sf_off: int
    l1_topk_weights_off: int
    l2_token_off: int
    l2_sf_off: int
    combine_token_off: int
    combine_scale_off: int

    total_bytes: int

    @staticmethod
    def build(cfg: "MegaMoEIntraNodeConfig") -> "_MegaMoELayout":
        num_ranks = cfg.world_size
        num_experts = cfg.num_experts
        epr = cfg.num_experts_per_rank
        m = cfg.aligned_num_max_tokens_per_rank
        hidden = cfg.hidden
        ih = cfg.intermediate_hidden
        num_topk = cfg.num_topk

        num_max_recv_per_expert = cfg.num_max_recv_tokens_per_expert
        pool_tokens = cfg.num_max_pool_tokens
        pool_blocks = cfg.num_max_pool_blocks
        padded_sf_pool = cfg.num_max_padded_sf_pool_tokens

        # --- Workspace (counters / signals), accumulation order matches
        #     Workspace::get_num_bytes ---
        off = 0
        barrier_off = off
        off += _NUM_BARRIER_SIGNAL_BYTES
        expert_send_count_off = off
        off += num_experts * 8
        expert_recv_count_off = off
        off += num_experts * 8
        expert_recv_count_sum_off = off
        off += epr * 8
        l1_arrival_count_off = off
        off += _align(pool_blocks, 2) * 4
        l2_arrival_mask_off = off
        off += pool_blocks * 8
        src_token_topk_idx_off = off
        off += epr * num_ranks * num_max_recv_per_expert * 4
        token_src_metadata_off = off
        off += pool_tokens * _TOKEN_SRC_METADATA_BYTES
        workspace_bytes = _align(off, 16)

        # --- Data buffers (Buffer chain) ---
        off = workspace_bytes
        input_token_off = off
        off += m * hidden                      # fp8 tokens
        input_sf_off = off
        off += m * (hidden // 32)              # SF bytes (= hidden/128 int32)
        input_topk_idx_off = off
        off += m * num_topk * 8                # int64
        input_topk_weights_off = off
        off += m * num_topk * 4                # float32

        l1_token_off = off
        off += pool_tokens * hidden
        l1_sf_off = off
        off += padded_sf_pool * (hidden // 32)
        l1_topk_weights_off = off
        off += pool_tokens * 4

        l2_token_off = off
        off += pool_tokens * ih
        l2_sf_off = off
        off += padded_sf_pool * (ih // 32)

        combine_token_off = off
        off += num_topk * m * (hidden * 2)     # bf16 combine staging
                                               # (also holds fp8 data, hidden*1,
                                               #  under MEGA_MOE_FP8_COMBINE)

        # fp8-combine per-16-channel bf16 scales (one per n16 block of each
        # token-partial row).  Always reserved so the symmetric buffer is sized
        # for the opt-in fp8 path; the bf16 combine path leaves it unused.
        combine_scale_off = off
        off += num_topk * m * (hidden // 16) * 2

        total_bytes = off

        return _MegaMoELayout(
            barrier_off=barrier_off,
            expert_send_count_off=expert_send_count_off,
            expert_recv_count_off=expert_recv_count_off,
            expert_recv_count_sum_off=expert_recv_count_sum_off,
            l1_arrival_count_off=l1_arrival_count_off,
            l2_arrival_mask_off=l2_arrival_mask_off,
            src_token_topk_idx_off=src_token_topk_idx_off,
            token_src_metadata_off=token_src_metadata_off,
            workspace_bytes=workspace_bytes,
            input_token_off=input_token_off,
            input_sf_off=input_sf_off,
            input_topk_idx_off=input_topk_idx_off,
            input_topk_weights_off=input_topk_weights_off,
            l1_token_off=l1_token_off,
            l1_sf_off=l1_sf_off,
            l1_topk_weights_off=l1_topk_weights_off,
            l2_token_off=l2_token_off,
            l2_sf_off=l2_sf_off,
            combine_token_off=combine_token_off,
            combine_scale_off=combine_scale_off,
            total_bytes=total_bytes,
        )


def get_symm_buffer_bytes_for_mega_moe(world_size: int, num_experts: int,
                                       num_max_tokens_per_rank: int, num_topk: int,
                                       hidden: int, intermediate_hidden: int,
                                       use_fp8_dispatch: bool = True,
                                       activation: str = "swiglu") -> int:
    """Parity helper for ``_C.get_symm_buffer_size_for_mega_moe`` (size only).

    Returns the total symmetric-buffer byte count for the given geometry.
    """
    cfg = MegaMoEIntraNodeConfig(
        rank=0, world_size=world_size, num_experts=num_experts,
        num_topk=num_topk, hidden=hidden, intermediate_hidden=intermediate_hidden,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        use_fp8_dispatch=use_fp8_dispatch, activation=activation,
    )
    return _MegaMoELayout.build(cfg).total_bytes


class FlyDSLMegaMoEIntraNodeOp:
    """Intra-node mega_moe op (host surface).

    Owns the mori symmetric buffer and exposes DeepGEMM-parity input
    views (``x``, ``x_sf``, ``topk_idx``, ``topk_weights``) plus the
    L1/L2 activation pools. The caller copies inputs into the views, then
    calls ``fp8_fp4_mega_moe(...)``.
    """

    def __init__(self, config: MegaMoEIntraNodeConfig):
        self.cfg = config
        self._check_config()
        self._dev = torch.device("cuda", config.rank)
        self._layout = _MegaMoELayout.build(config)

        self._alloc_buffer()
        ms.shmem_barrier_all()
        self._build_peer_base()
        self._build_views()

        # Lazy two-stage JIT caches for the single fused mega_moe kernel
        # (geometry is baked into the builder, so a single "mega_moe" key
        # suffices); first stage caches the jit fn, second the compiled.
        self._mega_moe_jit_cache: Dict[Any, Any] = {}
        self._mega_moe_compiled_cache: Dict[Any, Any] = {}

        # L1 compute-phase scratch (debug interface; populated per call).
        self._l1_out: Optional[torch.Tensor] = None
        self._l1_w: Optional[torch.Tensor] = None
        self._l1_ws: Optional[torch.Tensor] = None
        # L2 compute-phase scratch (debug interface; populated per call).
        self._l2_out: Optional[torch.Tensor] = None
        self._l2_w: Optional[torch.Tensor] = None
        self._l2_ws: Optional[torch.Tensor] = None

        # fx handles reused across launches.
        self._fx_local_base = fx.Int64(self.buffer.data_ptr())
        self._fx_peer_base = fx.Int64(self._peer_base.data_ptr())

    # ---- allocation / wiring ------------------------------------------
    def _alloc_buffer(self):
        # Single int8 symmetric region; all segments are offsets into it.
        # mori_shmem_create_tensor returns uninitialized memory, so zero
        # the counters/signals the kernel reads before its first write.
        self.buffer = mori_shmem_create_tensor((self._layout.total_bytes,), torch.int8)
        self.buffer.zero_()

    def _build_peer_base(self):
        npes = self.cfg.world_size
        r = self.cfg.rank
        self._peer_base = torch.zeros(npes, dtype=torch.int64, device=self._dev)
        head = self.buffer.data_ptr()
        for pe in range(npes):
            self._peer_base[pe] = ms.shmem_ptr_p2p(head, r, pe)

    def _seg(self, off: int, nbytes: int, dtype: torch.dtype, shape):
        """Contiguous typed view of a byte segment of ``self.buffer``."""
        sub = self.buffer[off:off + nbytes]
        if dtype != torch.int8:
            sub = sub.view(dtype)
        return sub.view(*shape)

    def _seg_sf_mmajor(self, off: int, rows: int, cols: int):
        """M-major (column-major) SF view: element [t, k] lives at
        ``base + t + k * rows`` (int32 units). Mirrors DeepGEMM's
        ``l1_acts_sf`` / ``l2_acts_sf`` strides ``{1, padded_pool}``."""
        nbytes = rows * cols * 4
        flat = self.buffer[off:off + nbytes].view(torch.int32)
        return torch.as_strided(flat, size=(rows, cols), stride=(1, rows))

    def _build_views(self):
        cfg = self.cfg
        L = self._layout
        m = cfg.aligned_num_max_tokens_per_rank
        hidden = cfg.hidden
        ih = cfg.intermediate_hidden
        k = cfg.num_topk
        pool = cfg.num_max_pool_tokens
        padded_sf = cfg.num_max_padded_sf_pool_tokens

        # Input buffers (DeepGEMM slice order: x, x_sf, topk_idx,
        # topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf).
        # NOTE: x_sf is K-major (contiguous), l1/l2 SF are M-major.
        self.x = self._seg(L.input_token_off, m * hidden, _FP8_DTYPE, (m, hidden))
        self.x_sf = self._seg(L.input_sf_off, m * (hidden // 32), torch.int32, (m, hidden // 128))
        self.topk_idx = self._seg(L.input_topk_idx_off, m * k * 8, torch.int64, (m, k))
        self.topk_weights = self._seg(L.input_topk_weights_off, m * k * 4, torch.float32, (m, k))

        self.l1_acts = self._seg(L.l1_token_off, pool * hidden, _FP8_DTYPE, (pool, hidden))
        self.l1_acts_sf = self._seg_sf_mmajor(L.l1_sf_off, padded_sf, hidden // 128)
        self.l1_topk_weights = self._seg(L.l1_topk_weights_off, pool * 4, torch.float32, (pool,))
        self.l2_acts = self._seg(L.l2_token_off, pool * ih, _FP8_DTYPE, (pool, ih))
        self.l2_acts_sf = self._seg_sf_mmajor(L.l2_sf_off, padded_sf, ih // 128)

        # Workspace counter views (kernel-internal; also read by the
        # debug interface). uint64 counts surface as int64 (torch lacks
        # uint64); the values fit positive int64.
        ne = cfg.num_experts
        epr = cfg.num_experts_per_rank
        self.expert_send_count = self._seg(L.expert_send_count_off, ne * 8, torch.int64, (ne,))
        self.expert_recv_count = self._seg(L.expert_recv_count_off, ne * 8, torch.int64, (ne,))
        self.expert_recv_count_sum = self._seg(L.expert_recv_count_sum_off, epr * 8, torch.int64, (epr,))
        self.token_src_metadata = self._seg(
            L.token_src_metadata_off, pool * _TOKEN_SRC_METADATA_BYTES, torch.int32, (pool, 3)
        )

    # ---- validation ----------------------------------------------------
    def _check_config(self):
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
                f"_MAX_INTRANODE_NPES={_MAX_INTRANODE_NPES}; use an inter-node op"
            )
        if cfg.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {cfg.num_experts}")
        if cfg.num_experts % cfg.world_size != 0:
            raise ValueError(
                f"num_experts({cfg.num_experts}) must be divisible by "
                f"world_size({cfg.world_size})"
            )
        if cfg.num_topk <= 0:
            raise ValueError(f"num_topk must be positive, got {cfg.num_topk}")
        if cfg.hidden <= 0 or cfg.intermediate_hidden <= 0:
            raise ValueError(
                f"hidden/intermediate_hidden must be positive, got "
                f"({cfg.hidden}, {cfg.intermediate_hidden})"
            )
        # SF granularity (gran_k=128) and UTCCP SF transpose require these.
        if cfg.hidden % 128 != 0 or cfg.intermediate_hidden % 128 != 0:
            raise ValueError(
                f"hidden({cfg.hidden}) and intermediate_hidden({cfg.intermediate_hidden}) "
                "must be multiples of 128"
            )
        if cfg.num_max_tokens_per_rank <= 0:
            raise ValueError(f"num_max_tokens_per_rank must be positive, got {cfg.num_max_tokens_per_rank}")
        if not cfg.use_fp8_dispatch:
            raise ValueError("only use_fp8_dispatch=True is supported (DeepGEMM parity)")
        if cfg.activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"activation={cfg.activation!r} not supported. Supported: {_SUPPORTED_ACTIVATIONS}"
            )
        # SF pool token count must be a multiple of 4 (DeepGEMM assert).
        if cfg.num_max_padded_sf_pool_tokens % 4 != 0:
            raise ValueError(
                f"num_max_padded_sf_pool_tokens={cfg.num_max_padded_sf_pool_tokens} "
                "must be a multiple of 4"
            )
        # block_m must be a candidate BLOCK_M: the SF pool is sized as the max
        # over the candidate set, so a non-candidate value can overflow it.
        if cfg.block_m not in _KCANDIDATE_BLOCK_M:
            raise ValueError(
                f"block_m={cfg.block_m} must be one of the candidate BLOCK_M "
                f"values {_KCANDIDATE_BLOCK_M}"
            )

    def _check_tensor_device(self, name, t):
        if not torch.is_tensor(t):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t)}")
        if not t.is_cuda:
            raise ValueError(f"{name} must live on CUDA, got device={t.device}")
        if t.device.index != self.cfg.rank:
            raise ValueError(
                f"{name}.device={t.device} does not match cfg.rank={self.cfg.rank} "
                f"(expected cuda:{self.cfg.rank})"
            )

    def _check_call_inputs(self, y, l1_weights, l2_weights,
                           cumulative_local_expert_recv_stats, recipe, activation,
                           activation_clamp):
        cfg = self.cfg
        self._check_tensor_device("y", y)
        if y.dim() != 2:
            raise ValueError(f"y must be 2-D (num_tokens, hidden), got shape {tuple(y.shape)}")
        num_tokens = y.shape[0]
        if num_tokens > cfg.aligned_num_max_tokens_per_rank:
            raise ValueError(
                f"y rows={num_tokens} exceeds aligned num_max_tokens_per_rank="
                f"{cfg.aligned_num_max_tokens_per_rank}"
            )
        if y.shape[1] != cfg.hidden:
            raise ValueError(f"y.shape[1]={y.shape[1]} != hidden={cfg.hidden}")

        rm, rn, rk = recipe
        if (rm, rn, rk) != (1, 1, 32):
            raise ValueError(f"recipe must be (1, 1, 32), got {recipe}")
        if activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(f"activation={activation!r} not supported. Supported: {_SUPPORTED_ACTIVATIONS}")
        # DeepGEMM parity: activation_clamp defaults to +inf (disabled) and
        # must be non-negative when provided.
        if activation_clamp is not None and activation_clamp < 0:
            raise ValueError(f"activation_clamp must be non-negative, got {activation_clamp}")

        for name, wt in (("l1_weights", l1_weights), ("l2_weights", l2_weights)):
            if not (isinstance(wt, (tuple, list)) and len(wt) == 2):
                raise ValueError(f"{name} must be a (weights, weights_sf) tuple, got {type(wt)}")
            self._check_tensor_device(f"{name}[0]", wt[0])
            self._check_tensor_device(f"{name}[1]", wt[1])

        if cumulative_local_expert_recv_stats is not None:
            s = cumulative_local_expert_recv_stats
            self._check_tensor_device("cumulative_local_expert_recv_stats", s)
            if s.dtype != torch.int32:
                raise ValueError(f"cumulative_local_expert_recv_stats.dtype={s.dtype} must be int32")
            if s.numel() != cfg.num_experts_per_rank:
                raise ValueError(
                    f"cumulative_local_expert_recv_stats.numel()={s.numel()} != "
                    f"num_experts_per_rank={cfg.num_experts_per_rank}"
                )

    # ---- public API (DeepGEMM parity) ---------------------------------
    @staticmethod
    def transform_weights_for_mega_moe(l1_weights, l2_weights):
        """Parity stub for ``transform_weights_for_mega_moe``.

        DeepGEMM does two things: (1) interleave gate/up at granularity 8,
        and (2) transpose the weight SF for UTCCP. Step (2) targets the
        NVIDIA TMEM/UTCCP layout and does NOT apply to gfx950: the FlyDSL
        grouped GEMM rearranges scales in-kernel at load time (see
        ``mixed_moe_gemm_2stage.py`` ``_rearrange_a_scale`` /
        ``_rearrange_b_scale`` and its ``gate_up_interleave`` mode), so
        the host-side weight/SF layout must follow whatever the gfx950
        mega compute kernel ends up consuming.

        That kernel is still stubbed, so this stays a stub until the
        compute-phase weight/SF layout is finalized (avoids committing to
        a wrong, NV-specific transform). The dispatch phase does not
        consume transformed weights.
        """
        raise NotImplementedError(
            "transform_weights_for_mega_moe is not implemented yet: the "
            "gfx950 weight/SF layout is decided by the (still-stubbed) "
            "compute kernel, and DeepGEMM's UTCCP SF transpose does not "
            "apply on AMD. This milestone covers the host surface + dispatch."
        )

    def fp8_fp4_mega_moe(self, y: torch.Tensor,
                         l1_weights: Tuple[torch.Tensor, torch.Tensor],
                         l2_weights: Tuple[torch.Tensor, torch.Tensor],
                         cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                         recipe: Tuple[int, int, int] = (1, 1, 32),
                         activation: str = "swiglu",
                         activation_clamp: Optional[float] = None,
                         fast_math: bool = True):
        """Run the mega_moe op. Signature mirrors DeepGEMM's
        ``deep_gemm.mega.fp8_fp4_mega_moe`` (minus ``sym_buffer``, which
        the op owns).

        Milestone scope: only the dispatch phase runs; compute warps are
        stubbed and ``y`` is not yet populated with final results.
        """
        self._check_call_inputs(
            y, l1_weights, l2_weights,
            cumulative_local_expert_recv_stats, recipe, activation,
            activation_clamp,
        )
        num_tokens = int(y.shape[0])
        # Per-call workspace prep.  The dispatch COUNT accumulators
        # (expert_send/recv/recv_count_sum + l1_arrival) are now cleared
        # IN-KERNEL at the tail (``_reset_round`` in the kernel), guarded by a
        # cross-rank NVLink barrier so a peer's next-call Phase C write can no
        # longer race a host-side clear (the multi-rank repeated-call hang).
        # Only the fused-pull metadata padding still needs a host zero.
        self._prep_call_workspace()

        # Enable the fused L1 compute only when real L1 weights are supplied
        # (preshuffled fp4 ``[epr*N, hidden//2]`` + CK e8m0 scale).  The
        # dispatch parity test passes a 1-element placeholder -> dispatch-only.
        epr = self.cfg.num_experts_per_rank
        ih = self.cfg.intermediate_hidden
        hidden = self.cfg.hidden
        N = 2 * ih
        l1_w, l1_ws = l1_weights
        l2_w, l2_ws = l2_weights
        enable_l1 = l1_w.numel() >= epr * N * (hidden // 2)
        if enable_l1 and hidden % 256 != 0:
            raise ValueError(
                f"hidden={hidden} must be a multiple of 256 for the "
                "fused L1 compute phase (CK B-scale layout)."
            )
        # The L2 compute phase additionally needs real L2 weights
        # (preshuffled fp4 ``[epr*hidden, ih//2]`` + CK e8m0 scale).  Require an
        # exact size match so an L1-weights placeholder (different shape) does
        # not spuriously enable the L2 phase.
        enable_l2 = enable_l1 and l2_w.numel() == epr * hidden * (ih // 2)

        if enable_l1:
            # B is consumed in the CK/aiter preshuffled fp4 layout with
            # CK-microscale e8m0 scales (caller-transformed for now).  Stash
            # device pointers for the kernel + a zeroed debug output pool.
            self._l1_w = l1_w.contiguous()
            self._l1_ws = l1_ws.contiguous().view(torch.int32)
            if self._l1_out is None:
                self._l1_out = torch.zeros(
                    (self.cfg.num_max_pool_tokens, N),
                    dtype=torch.float32, device=self._dev,
                )
            else:
                self._l1_out.zero_()
            fx_l1_w = fx.Int64(self._l1_w.data_ptr())
            fx_l1_ws = fx.Int64(self._l1_ws.data_ptr())
            fx_l1_out = fx.Int64(self._l1_out.data_ptr())
        else:
            fx_l1_w = fx_l1_ws = fx_l1_out = fx.Int64(0)

        if enable_l2:
            self._l2_w = l2_w.contiguous()
            self._l2_ws = l2_ws.contiguous().view(torch.int32)
            if self._l2_out is None:
                self._l2_out = torch.zeros(
                    (self.cfg.num_max_pool_tokens, hidden),
                    dtype=torch.float32, device=self._dev,
                )
            else:
                self._l2_out.zero_()
            fx_l2_w = fx.Int64(self._l2_w.data_ptr())
            fx_l2_ws = fx.Int64(self._l2_ws.data_ptr())
            fx_l2_out = fx.Int64(self._l2_out.data_ptr())
        else:
            fx_l2_w = fx_l2_ws = fx_l2_out = fx.Int64(0)
        ms.shmem_barrier_all()

        stream = torch.cuda.current_stream()
        stats_ptr = (
            cumulative_local_expert_recv_stats.data_ptr()
            if cumulative_local_expert_recv_stats is not None else 0
        )

        # Combine writes the final ``y`` (bf16 [num_tokens, hidden]); it runs
        # whenever the L2 phase does.  Pass a 0 pointer otherwise so the arg is
        # always present (the kernel only dereferences it when enable_combine).
        if enable_l2:
            y_c = y if y.is_contiguous() else y.contiguous()
            fx_y = fx.Int64(y_c.data_ptr())
        else:
            y_c = None
            fx_y = fx.Int64(0)

        # Compile args are identical across every phase (same pointers); the
        # phase only changes which kernel code is compiled in.
        _compile_args = (
            self._fx_local_base,
            self._fx_peer_base,
            fx.Int64(stats_ptr),
            num_tokens,
            fx_l1_w,
            fx_l1_ws,
            fx_l1_out,
            fx_l2_w,
            fx_l2_ws,
            fx_l2_out,
            fx_y,
            stream,
        )

        # Single monolithic launch (dispatch + L1 + fused SwiGLU + L2 +
        # combine + tail reset).  Production config is baked into the kernel
        # builder (WS / fused-swiglu / fused-combine / dispatch overlap all on).
        cache_key = ("mega_moe", bool(enable_l1), bool(enable_l2))
        compiled = self._mega_moe_compiled_cache.get(cache_key)
        if compiled is None:
            fn = self._get_mega_moe_jit(enable_l1, enable_l2)
            compiled = flyc.compile(fn, *_compile_args)
            self._mega_moe_compiled_cache[cache_key] = compiled
        else:
            compiled(
                self._fx_local_base, self._fx_peer_base, stats_ptr, num_tokens,
                fx_l1_w, fx_l1_ws, fx_l1_out, fx_l2_w, fx_l2_ws, fx_l2_out,
                fx_y, stream,
            )
        # Copy back if we had to densify a non-contiguous caller tensor.
        if y_c is not None and y_c is not y:
            y.copy_(y_c)

    def _prep_call_workspace(self):
        """Per-call host workspace prep.

        The dispatch COUNT accumulators (``expert_send_count`` /
        ``expert_recv_count`` / ``expert_recv_count_sum`` / ``l1_arrival_count``
        / ``l2_arrival_mask``) are cleared IN-KERNEL at the tail
        (``_reset_round`` in ``mega_moe_intranode_kernel.py``, mirroring DeepGEMM
        ``sm100_fp8_fp4_mega_moe.cuh:587-642``), behind a device-side cross-rank
        NVLink barrier.  That removes the multi-rank repeated-call hang.  The
        first call sees zeroed counters from the one-shot ``buffer.zero_()`` at
        construction.

        ``cumulative_local_expert_recv_stats`` is likewise NOT zeroed per call:
        the kernel ``red_add``-accumulates token counts into it on device
        (DeepGEMM parity), so the caller owns its lifetime.

        There is no remaining host-side prep.
        """

    def _resolve_num_sms(self) -> int:
        """Persistent grid size for the mega-kernel.

        Defaults to the device CU count (one block per CU) so the persistent
        grid covers the whole GPU instead of the scaffold default of 16, while
        staying within the grid barrier's co-residency requirement.  An
        ``MEGA_MOE_NUM_SMS`` env override lets the occupancy A/B benchmark
        sweep it; the value also keys the JIT caches below."""
        env = os.environ.get("MEGA_MOE_NUM_SMS")
        if env not in (None, "", "0"):
            return int(env)
        return torch.cuda.get_device_properties(self._dev).multi_processor_count

    def _get_mega_moe_jit(self, enable_l1: bool = False, enable_l2: bool = False):
        """Lazy-jit the monolithic mega_moe kernel (geometry baked in).

        Variants cached by ``(enable_l1, enable_l2, num_sms)``: dispatch-only,
        dispatch + fused L1 GEMM, and the full dispatch + L1 + SwiGLU/quant +
        L2 GEMM + combine chain.  The production fused config (warp-specialized
        L1/L2, fused SwiGLU, fused fp8 combine, dispatch<->compute overlap) is
        baked into the builder.  The builder lives in
        ``mega_moe_intranode_kernel`` and is imported lazily so the host module
        loads even before the kernel is written."""
        num_sms = self._resolve_num_sms()
        key = ("mega_moe", bool(enable_l1), bool(enable_l2), num_sms)
        if key not in self._mega_moe_jit_cache:
            try:
                from .mega_moe_intranode_kernel import make_mega_moe_jit
            except ImportError as exc:
                raise NotImplementedError(
                    "mega_moe kernel is not implemented yet "
                    "(host-only milestone). Buffer layout, views, and "
                    "debug_get_dispatch_output() are available without it."
                ) from exc

            cfg = self.cfg
            self._mega_moe_jit_cache[key] = make_mega_moe_jit(
                rank=cfg.rank,
                world_size=cfg.world_size,
                num_experts=cfg.num_experts,
                num_topk=cfg.num_topk,
                hidden=cfg.hidden,
                intermediate_hidden=cfg.intermediate_hidden,
                num_max_tokens_per_rank=cfg.aligned_num_max_tokens_per_rank,
                layout=self._layout,
                block_m=cfg.block_m,
                num_sms=num_sms,
                enable_l1_compute=enable_l1,
                enable_l2_compute=enable_l2,
                enable_combine=enable_l2,
            )
        return self._mega_moe_jit_cache[key]

    # ---- temporary single-phase validation interface ------------------
    def debug_get_dispatch_output(self) -> Dict[str, torch.Tensor]:
        """Snapshot the dispatch pull results for single-phase checking.

        Returns clones (host sync first) of the L1 token pool, its
        M-major SF, per-token weights, source metadata, and the
        per-expert receive counts. The test harness slices these against
        a torch reference dispatcher.
        """
        torch.cuda.synchronize()
        epr = self.cfg.num_experts_per_rank
        ne = self.cfg.num_experts
        nranks = ne // epr
        # ``expert_recv_count_sum`` is zeroed in-kernel at the tail (it is an
        # atomic-ADD accumulator that must start each call from 0), so it no
        # longer carries this call's counts after the kernel.  ``expert_recv_count``
        # (per-rank) IS still valid: Phase C overwrites it with a plain store
        # each call.  Reconstruct the per-local-expert sum from it (low 32 bits
        # hold the token count contributed by each rank).
        recv_count = self.expert_recv_count.clone()                     # (ne,) i64
        per_rank = (recv_count.view(nranks, epr) & 0xFFFFFFFF)          # (nranks, epr)
        recv_sum = per_rank.sum(dim=0)                                  # (epr,) i64
        return {
            "l1_acts": self.l1_acts.clone(),
            "l1_acts_sf": self.l1_acts_sf.clone(),
            "l1_topk_weights": self.l1_topk_weights.clone(),
            "token_src_metadata": self.token_src_metadata.clone(),
            "expert_recv_count_sum": recv_sum,
            "expert_recv_count": recv_count,
            # Per-call recv counts (== this dispatch's cumulative stats); the
            # real device-side ``cumulative_local_expert_recv_stats`` is an
            # ACROSS-call accumulator owned by the caller.
            "cumulative_local_expert_recv_stats": recv_sum.to(torch.int32),
        }

    def debug_get_l1_output(self) -> torch.Tensor:
        """Snapshot the L1 grouped-GEMM output pool (raw gate&up, no act).

        Shape ``[num_max_pool_tokens, 2*intermediate_hidden]`` f32; only rows
        for valid pool tokens are meaningful (padding rows read 0).
        """
        torch.cuda.synchronize()
        return self._l1_out.clone()

    def debug_get_l2_output(self) -> torch.Tensor:
        """Snapshot the L2 grouped-GEMM output pool (post-SwiGLU/quant @ W2).

        Shape ``[num_max_pool_tokens, hidden]`` f32; only rows for valid pool
        tokens are meaningful (padding rows read 0).  Available only when real
        L2 weights were supplied (otherwise ``None``).
        """
        torch.cuda.synchronize()
        return None if self._l2_out is None else self._l2_out.clone()

    def destroy(self):
        # Drop every view that references the symmetric buffer storage so
        # it can be freed once ``self.buffer`` is released.
        self.buffer = None
        self._peer_base = None
        self.x = self.x_sf = self.topk_idx = self.topk_weights = None
        self.l1_acts = self.l1_acts_sf = self.l1_topk_weights = None
        self.l2_acts = self.l2_acts_sf = None
        self.expert_send_count = self.expert_recv_count = None
        self.expert_recv_count_sum = self.token_src_metadata = None
