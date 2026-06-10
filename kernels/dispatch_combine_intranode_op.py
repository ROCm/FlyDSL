# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Python wrapper for the FlyDSL intra-node DispatchCombine op.

Mori-parity surface: symmetric I/O. ``quant_type='fp8_direct_cast'``
keeps caller dtype bf16 but switches wire dtype to fp8 (mori
``UseFp8DirectCast``). Optional caps ``max_total_recv_tokens`` and
``max_token_type_size`` shrink / re-target shmem allocation.

Buffer-sizing drift from mori: ``shmem_comb_inp_{tok,wts}`` stay at
worst-case ``ws * M`` even when the recv cap shrinks, because combine
Stage 1 P2P-scatters into slot ``rank * M + dest_lid`` with
``dest_lid in [0, M)`` -- shrinking would OOB-write. Dispatch-side
buffers (indexed by ``dest_tok_id``) DO shrink with the cap.
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

# Supported token dtypes; reject outside this set at wrapper construction
# rather than failing deep in JIT codegen.
_SUPPORTED_TOK_DTYPES = (
    torch.bfloat16,
    torch.float32,
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float4_e2m1fn_x2,
)

_SUPPORTED_QUANT_TYPES = ("none", "fp8_direct_cast")

_MAX_INTRANODE_NPES = 8

# Kernel streams 16-byte chunks (v4i32); sets ``token_bytes`` alignment.
_TOK_BYTES_ALIGN = 16

_DEFAULT_DISPATCH_BLOCK_NUM = 128
_DEFAULT_DISPATCH_WARP_NUM = 4
_DEFAULT_COMBINE_BLOCK_NUM = 128
_DEFAULT_COMBINE_WARP_NUM = 8


def _dtype_elem_size(dt):
    """Raw storage size in bytes. fp4x2 returns 1 (two fp4 per byte)."""
    return torch.tensor([], dtype=dt).element_size()


def _is_fp4_dtype(dt):
    return dt == torch.float4_e2m1fn_x2


def _token_bytes_for(dt, hidden_dim):
    """Per-row payload bytes; parametrised on launch-time dtype so dispatch
    / combine can use independent dtypes (mori parity)."""
    if _is_fp4_dtype(dt):
        return hidden_dim // 2
    return hidden_dim * _dtype_elem_size(dt)


def _token_view_dim_for(dt, hidden_dim):
    """Per-row torch view trailing dim (fp4 packs 2 elements per byte)."""
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
    # Legacy shared geometry knobs; kept for back-compat but no longer
    # used for launch selection.
    warp_num_per_block: int = _DEFAULT_COMBINE_WARP_NUM
    block_num: int = _DEFAULT_COMBINE_BLOCK_NUM
    # Per-phase launch geometry.
    dispatch_warp_num_per_block: int = _DEFAULT_DISPATCH_WARP_NUM
    dispatch_block_num: int = _DEFAULT_DISPATCH_BLOCK_NUM
    combine_warp_num_per_block: int = _DEFAULT_COMBINE_WARP_NUM
    combine_block_num: int = _DEFAULT_COMBINE_BLOCK_NUM
    scale_dim: int = 0
    scale_type_size: int = 0
    enable_std_moe: bool = False
    use_external_inp_buf: bool = True
    quant_type: str = "none"
    # Cap on total receive tokens across peers (mori ``maxTotalRecvTokens``).
    # ``0`` => worst-case ``ws * M``. Per-rank slots = ``ceil(cap/ws)``
    # clamped to ``M``. Over-cap tokens fold into the dup-sentinel path
    # (no OOB) but are silently dropped.
    max_total_recv_tokens: int = 0
    # Per-element upper bound in bytes for shmem allocation (mori
    # ``maxTokenTypeSize``). ``0`` derives from ``data_type``; set to
    # keep one op alive across dtype switches without re-alloc.
    max_token_type_size: int = 0

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
    def zero_copy(self) -> bool:
        """``True`` => combine reads/writes through the registered shared
        staging buffer (no caller-side input copy)."""
        return not self.use_external_inp_buf

    @property
    def max_recv(self):
        """Worst-case ``ws * M`` -- kernel sentinel encoding and
        sender-side ``atomic_add`` allocation use this regardless of
        ``max_total_recv_tokens``."""
        return self.world_size * self.max_num_inp_token_per_rank

    @property
    def effective_max_recv_per_rank(self):
        """Per-sender slot share on each dest PE (mori
        ``MaxNumTokensToRecvPerRank``)."""
        if self.max_total_recv_tokens <= 0:
            return self.max_num_inp_token_per_rank
        per_rank = (self.max_total_recv_tokens + self.world_size - 1) // self.world_size
        return min(per_rank, self.max_num_inp_token_per_rank)

    @property
    def effective_max_recv(self):
        """Total recv-slot count on each dest PE (mori
        ``MaxNumTokensToRecv``); passed to the kernel as ``max_recv``."""
        return self.world_size * self.effective_max_recv_per_rank

    @property
    def max_token_bytes(self):
        """Per-row shmem allocation bound (mori ``maxTokenTypeSize``)."""
        if self.max_token_type_size > 0:
            return self.hidden_dim * self.max_token_type_size
        return self.token_bytes

    @property
    def scale_bytes(self):
        return self.scale_dim * self.scale_type_size


class FlyDSLDispatchCombineIntraNodeOp:

    def __init__(self, config):
        self.cfg = config
        # Validate BEFORE GPU alloc so misconfigs raise ValueError, not
        # an opaque HIP/MLIR error.
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
        for pe in range(npes):
            self._p2p_comb_inp[pe] = ms.shmem_ptr_p2p(self.shmem_comb_inp_tok.data_ptr(), r, pe)
            self._p2p_comb_inp_wts[pe] = ms.shmem_ptr_p2p(self.shmem_comb_inp_wts.data_ptr(), r, pe)
            self._p2p_xdb_mem[pe] = ms.shmem_ptr_p2p(self.shmem_xdev_bar_mem.data_ptr(), r, pe)

        # Dispatch (encode) and combine (decode, Stage 3) must agree on this.
        self._effective_max_recv = config.effective_max_recv

        # Launch-time JIT caches (mori parity): each ``op.dispatch(input)``
        # / ``op.combine(input)`` picks a specialization by ``input.dtype``.
        # Combine cache key includes ``(zero_copy, enable_weights,
        # fp8_direct_cast)`` because all four switches affect codegen.
        self._disp_jit_cache: Dict[torch.dtype, Any] = {}
        self._disp_compiled_cache: Dict[torch.dtype, Any] = {}
        self._comb_jit_cache: Dict[Tuple[torch.dtype, bool, bool, bool], Any] = {}
        self._comb_compiled_cache: Dict[Tuple[torch.dtype, bool, bool, bool], Any] = {}

        # Start at 1: a zero-init flag would satisfy the first
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

        # Lazy skip_stage1 variant for the fused GEMM2+combine path.
        # Key includes dtype so a runtime dtype switch picks a fresh kernel.
        self._comb_no_s1_fn: Dict[Tuple[torch.dtype, bool], Any] = {}
        self._comb_no_s1_compiled: Dict[Tuple[torch.dtype, bool], Any] = {}

    def _alloc_buffers(self):
        cfg = self.cfg
        npes = cfg.world_size
        k = cfg.num_experts_per_token
        mt = cfg.max_num_inp_token_per_rank
        # Two recv-slot caps:
        #   mr       = effective dispatch-side cap (shrinks with user cap)
        #   mr_worst = ws * M (combine-input cap, cannot shrink because
        #              Stage 1 indexes slot ``rank * M + dest_lid``).
        mr = cfg.effective_max_recv
        mr_worst = cfg.max_recv

        tb_max = cfg.max_token_bytes
        tok_i16_mr = (mr * tb_max + 1) // 2
        tok_i16_mr_worst = (mr_worst * tb_max + 1) // 2
        tok_i16_mt = (mt * tb_max + 1) // 2

        # All shmem buffers below are P2P-accessed by peers except
        # ``shmem_comb_out_{tok,wts}`` (local-only; kept on the shmem
        # heap for mori layout parity).
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

        # shmem_malloc is uninitialized; zero the buffers combine reads
        # so degenerate (unwritten) slots are harmless.
        self.shmem_tok_id_to_src.zero_()
        self.shmem_comb_inp_tok.zero_()
        self.shmem_comb_inp_wts.zero_()
        self.shmem_xdev_bar_mem.zero_()

        self.dest_pe_ctr = torch.zeros(npes, dtype=torch.int32, device=self._dev)
        self.disp_bar = torch.zeros(1, dtype=torch.int32, device=self._dev)
        self.comb_bar = torch.zeros(1, dtype=torch.int32, device=self._dev)
        self.total_recv = torch.zeros(1, dtype=torch.int32, device=self._dev)
        sentinel = cfg.world_size * mr
        self.dest_tok_map = torch.full((mt * k,), sentinel, dtype=torch.int32, device=self._dev)

        if cfg.enable_std_moe:
            epr = cfg.num_experts_per_rank
            # Expert-side capacity is independent of max_total_recv_tokens
            # (mori treats them as separate dims).
            max_tok_per_expert = cfg.max_recv
            self.packed_recv_count = torch.zeros(epr, dtype=torch.int32, device=self._dev)
            self.packed_recv_src_info = torch.zeros(epr * max_tok_per_expert, dtype=torch.int32, device=self._dev)
            self.disp_tok_to_ep_slot_map = torch.full((mr * k,), -1, dtype=torch.int64, device=self._dev)
            # i64 ticket counter for the StdMoE Phase 4 in-kernel grid
            # barrier; never host-reset, i64 prevents wraparound.
            self.disp_grid_bar = torch.zeros(1, dtype=torch.int64, device=self._dev)
        else:
            self.packed_recv_count = torch.zeros(1, dtype=torch.int32, device=self._dev)
            self.packed_recv_src_info = torch.zeros(1, dtype=torch.int32, device=self._dev)
            self.disp_tok_to_ep_slot_map = torch.zeros(1, dtype=torch.int64, device=self._dev)
            self.disp_grid_bar = torch.zeros(1, dtype=torch.int64, device=self._dev)

    # Config / runtime contract checks: fail fast on misuse instead of
    # OOB-writing or aborting deep in JIT codegen.
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
        # k <= 64 is a hard kernel constraint (``ballot`` only covers
        # 64 warp lanes).
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
        # not been retrofitted for asymmetric I/O dtypes.
        if cfg.quant_type == "fp8_direct_cast" and cfg.enable_std_moe:
            raise NotImplementedError(
                "quant_type='fp8_direct_cast' is not supported with "
                "enable_std_moe=True (asymmetric I/O dtypes not yet wired)"
            )
        # zero_copy (use_external_inp_buf=False) skips Stage 1, which is
        # exactly where the bf16->fp8 cast lives, so fp8_direct_cast and
        # zero_copy are mutually exclusive.  Both are static cfg, so reject
        # the combination here instead of on the first combine() call.
        if cfg.quant_type == "fp8_direct_cast" and cfg.zero_copy:
            raise ValueError(
                "quant_type='fp8_direct_cast' is incompatible with "
                "zero_copy=True (use_external_inp_buf=False): zero_copy "
                "skips Stage 1 where the bf16->fp8 cast lives."
            )

        # Kernel streams 16 B (v4i32) per lane; one stride check on
        # ``token_bytes`` covers both Stage 1 reads and Stage 3 writes.
        if cfg.token_bytes % _TOK_BYTES_ALIGN != 0:
            raise ValueError(
                f"token row bytes ({cfg.token_bytes}) must be a multiple of "
                f"{_TOK_BYTES_ALIGN} for v4i32 vector loads; check hidden_dim "
                f"({cfg.hidden_dim}) and data_type ({cfg.data_type})"
            )

        # max_total_recv_tokens: 0 disables; positive values must give
        # every rank >= 1 slot; over-cap is clamped to the worst case.
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

        # max_token_type_size: 0 disables; otherwise must cover the
        # dispatch element size.
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
        # must fit i32 (used by the kernel's ``divui`` decoding).
        total_experts = cfg.world_size * cfg.num_experts_per_rank
        if total_experts > (1 << 31) - 1:
            raise ValueError(
                f"total experts ({cfg.world_size} * {cfg.num_experts_per_rank} = {total_experts}) "
                "exceeds int32 range"
            )

        # LDS budget pre-flight (clearer error than the equivalent JIT-time
        # capacity failure).
        self._check_lds_capacity()

    def _check_lds_capacity(self):
        """Reject configs whose combine-kernel LDS overflows the GPU."""
        from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP

        cfg = self.cfg

        # Mirror ``make_combine_jit``'s layout: two 8B-aligned i64[npes]
        # tables (tokens + weights), arena padded to 128 B.
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
                f"combine kernel LDS needs {lds_bytes} B "
                f"(2 x i64[world_size={cfg.world_size}] P2P tables + 128 B arena), "
                f"but device {arch} provides only {limit} B"
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
        # Launch-time dtype (mori parity): ``input.dtype`` is not tied
        # to ``cfg.data_type``; any supported dtype that fits the shmem
        # allocation budget is accepted.
        if input.dtype not in _SUPPORTED_TOK_DTYPES:
            raise ValueError(f"input.dtype={input.dtype} not in supported set {_SUPPORTED_TOK_DTYPES}")
        expected_hdim = _token_view_dim_for(input.dtype, cfg.hidden_dim)
        if input.shape[1] != expected_hdim:
            raise ValueError(
                f"input.shape[1]={input.shape[1]} != expected {expected_hdim} "
                f"(hidden_dim={cfg.hidden_dim}, dtype={input.dtype})"
            )
        # Per-row payload must fit ``max_token_bytes``; bump
        # ``cfg.max_token_type_size`` when mixing dtypes.
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

        # indices: (cur_tok, k); wrapper casts to int32 below.
        if indices.dim() != 2:
            raise ValueError(f"indices must be 2-D (cur_tok, k), got shape {tuple(indices.shape)}")
        if indices.shape != (cur_tok, cfg.num_experts_per_token):
            raise ValueError(
                f"indices.shape={tuple(indices.shape)} != expected " f"({cur_tok}, {cfg.num_experts_per_token})"
            )
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"indices.dtype={indices.dtype} must be int32 or int64")

        # scales: all-or-none contract -- pass either all three
        # (scales tensor + cfg.scale_dim>0 + cfg.scale_type_size>0) or
        # none. Mixed states cause silent OOB / dropped writes.
        scales_configured = cfg.scale_bytes > 0
        if (scales is not None) != scales_configured:
            raise ValueError(
                "dispatch scales all-or-none contract violated: "
                f"scales={'provided' if scales is not None else 'None'} but "
                f"cfg.scale_dim={cfg.scale_dim} / cfg.scale_type_size={cfg.scale_type_size} "
                f"{'enable' if scales_configured else 'disable'} the scales path. "
                "Pass all three (scales tensor + cfg.scale_dim>0 + cfg.scale_type_size>0) "
                "or none of them."
            )
        if scales is not None:
            self._check_tensor_device("scales", scales)
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
        # (cur_rank_num_token, hidden_or_packed); leading dim varies,
        # hidden_dim must match the configured one.
        if input.dim() != 2:
            raise ValueError(f"combine input must be 2-D, got shape {tuple(input.shape)}")
        if strict_input_dtype and input.dtype not in _SUPPORTED_TOK_DTYPES:
            raise ValueError(f"combine input.dtype={input.dtype} not in supported set {_SUPPORTED_TOK_DTYPES}")
        # Launch-time dtype (mori parity); row-stride derived from input.dtype.
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
        parity). flyc.compile second-stage cache is also dtype-keyed."""
        if d_dtype not in self._disp_jit_cache:
            cfg = self.cfg
            self._disp_jit_cache[d_dtype] = make_dispatch_jit(
                rank=cfg.rank,
                npes=cfg.world_size,
                experts_per_rank=cfg.num_experts_per_rank,
                experts_per_token=cfg.num_experts_per_token,
                hidden_dim=cfg.hidden_dim,
                max_tok_per_rank=cfg.max_num_inp_token_per_rank,
                block_num=cfg.dispatch_block_num,
                warp_num_per_block=cfg.dispatch_warp_num_per_block,
                data_type=d_dtype,
                scale_dim=cfg.scale_dim,
                scale_type_size=cfg.scale_type_size,
                enable_std_moe=cfg.enable_std_moe,
                max_recv=self._effective_max_recv,
            )
        return self._disp_jit_cache[d_dtype]

    def dispatch(self, input, weights, scales, indices, packed_recv_x=None):
        self._check_dispatch_inputs(input, weights, scales, indices, packed_recv_x)
        cfg = self.cfg
        d_dtype = input.dtype
        inp_cur_tok = input.shape[0]
        # Stash dispatch input count for ``combine()``'s default
        # ``cur_tok`` (mori ``args.curRankNumToken``).
        self._last_inp_cur_tok = inp_cur_tok
        stream = torch.cuda.current_stream()
        inp_c = input if input.is_contiguous() else input.contiguous()
        wts_c = weights if weights.is_contiguous() else weights.contiguous()
        idx_c = (
            indices
            if (indices.dtype == torch.int32 and indices.is_contiguous())
            else indices.to(torch.int32).contiguous()
        )

        sc_ptr = scales.data_ptr() if scales is not None else 0
        prx_ptr = packed_recv_x.data_ptr() if packed_recv_x is not None else 0

        if cfg.enable_std_moe:
            self.packed_recv_count.zero_()
            # ``disp_grid_bar`` is intentionally NOT reset; the kernel
            # uses an ``atomic_add`` ticket to derive each launch's epoch.

        # _std_args layout MUST match the trailing 8 ``addr_*`` params
        # of ``dispatch_launch`` in order:
        #   shmem_tok, shmem_idx, shmem_tok_id_to_src, out_packed_recv_x,
        #   out_packed_recv_count, out_packed_recv_src_info,
        #   out_disp_tok_map, disp_grid_bar.
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
                inp_cur_tok,
                stream,
            )

        # Match the allocation size in ``_alloc_buffers``. Output view
        # uses the launch-time dtype, not ``cfg.data_type``.
        mr = cfg.effective_max_recv
        k = cfg.num_experts_per_token

        out_token_bytes = _token_bytes_for(d_dtype, cfg.hidden_dim)
        out_view_dim = _token_view_dim_for(d_dtype, cfg.hidden_dim)
        out_tok = self.shmem_disp_out_tok.view(torch.int8)[: mr * out_token_bytes].view(d_dtype).view(mr, out_view_dim)
        out_wts = self.shmem_disp_out_wts.view(mr, k)
        out_idx = self.shmem_disp_out_idx.view(mr, k)
        out_scales = None
        if cfg.scale_bytes > 0:
            # Mirror caller's scales layout: typed (fp32/fp16/bf16) or
            # uint8 byte-stream view, both must round-trip out of the
            # int8 shmem buffer with the same dtype + per-row dim.
            # ``scales`` is guaranteed non-None here (all-or-none input
            # contract enforced in _check_dispatch_inputs).
            out_scales = self.shmem_out_scales[: mr * cfg.scale_bytes].view(scales.dtype).view(mr, scales.shape[1])

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
        ``(c_dtype, zero_copy, enable_weights, fp8_direct_cast)`` tuple.
        Zero-copy hard-wires ``skip_stage1=True`` -- the caller has
        already staged into ``shmem_comb_inp_tok``."""
        key = (c_dtype, bool(zero_copy), bool(enable_weights_flag), bool(fp8_dc))
        if key not in self._comb_jit_cache:
            cfg = self.cfg
            self._comb_jit_cache[key] = make_combine_jit(
                rank=cfg.rank,
                npes=cfg.world_size,
                experts_per_token=cfg.num_experts_per_token,
                hidden_dim=cfg.hidden_dim,
                max_tok_per_rank=cfg.max_num_inp_token_per_rank,
                block_num=cfg.combine_block_num,
                warp_num_per_block=cfg.combine_warp_num_per_block,
                data_type=c_dtype,
                enable_weights=bool(enable_weights_flag),
                enable_std_moe=cfg.enable_std_moe,
                zero_copy=bool(zero_copy),
                # Zero-copy => caller pre-staged tokens into
                # shmem_comb_inp_tok; skip_stage1 removes the kernel's
                # Stage 1 token copy (weight copy is still emitted so
                # Stage 3b can read via recv_tok_id).
                skip_stage1=bool(zero_copy),
                fp8_direct_cast=bool(fp8_dc),
                max_recv=self._effective_max_recv,
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

        ``input.dtype`` selects the kernel specialization at launch time
        (mori parity); the JIT cache is keyed by
        ``(input.dtype, zero_copy, enable_weights, fp8_direct_cast)``.

        Zero-copy mode (``cfg.zero_copy=True``) requires the caller to
        write into the buffer returned by
        ``get_registered_combine_input_buffer(combine_dtype)`` before
        invoking ``combine()``; passing a non-shmem tensor raises
        ``ValueError``.

        Launch geometry and the zero-copy switch are frozen by
        ``self.cfg``; rebuild the op to change them.
        """
        self._check_combine_inputs(input, weights, indices, packed_recv_x)
        cfg = self.cfg
        stream = torch.cuda.current_stream()

        c_dtype = input.dtype
        zero_copy = cfg.zero_copy
        # combine_no_stage1 is the only entry point exposing weight-free
        # combines today; this path always carries weights.
        enable_weights_flag = True
        # fp8_direct_cast fires only when both the cfg knob is set AND
        # the launch dtype is bf16 (mori UseFp8DirectCast parity).
        fp8_dc = cfg.quant_type == "fp8_direct_cast" and c_dtype == torch.bfloat16
        # NOTE: the static zero_copy + fp8_direct_cast conflict is rejected
        # up-front in _check_config(); only the per-call buffer-pointer
        # contract is validated here.
        if zero_copy and input.data_ptr() != self.shmem_comb_inp_tok.data_ptr():
            # Zero-copy contract: peers read ``shmem_comb_inp_tok``;
            # any other pointer = silent correctness bug.
            raise ValueError(
                "zero_copy mode requires the caller to write into the buffer "
                "returned by op.get_registered_combine_input_buffer(combine_dtype). "
                f"Got input.data_ptr()={input.data_ptr():#x} but "
                f"shmem_comb_inp_tok.data_ptr()={self.shmem_comb_inp_tok.data_ptr():#x}."
            )

        inp_c = input if input.is_contiguous() else input.contiguous()

        # cur_tok = Stage 3 iteration bound. Defaults to the last
        # dispatch()'s input count; callers bypassing dispatch (fused
        # GEMM2+combine) must pass it explicitly.
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
            # reads fp8; cast here (independent from main input path).
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

    # Gated reserved API: only the fused GEMM2+combine path on
    # ``yanbo/fused_gemm2_combine`` meets the pre-populated-buffer
    # contract.
    _ENABLE_COMBINE_NO_STAGE1 = False

    def combine_no_stage1(self, input, weights, indices, packed_recv_x=None, cur_tok=None, enable_weights: bool = True):
        """Skip-Stage1 combine (reserved for fused GEMM2+combine path).

        Runs only Stage 2 + Stage 3; the upstream fused kernel must have
        populated ``shmem_comb_inp[_wts]``. Gated by
        ``_ENABLE_COMBINE_NO_STAGE1``.

        ``enable_weights=True`` keeps the Stage 1 weight scatter + Stage 3b
        in combine (upstream GEMM2 16B writes get dropped under IPC
        contention). ``False`` DCEs both weight steps (~3-5 us savings).

        Caller-owned contract:
          - ``shmem_comb_inp_tok`` holds every consumed token;
          - ``shmem_comb_inp_wts`` holds matching weights when enabled;
          - ``cur_tok`` matches the real receive count.
        """
        if not type(self)._ENABLE_COMBINE_NO_STAGE1:
            raise NotImplementedError(
                "combine_no_stage1 is reserved for the fused GEMM2+combine "
                "path. Use combine(...) instead, or set "
                "FlyDSLDispatchCombineIntraNodeOp._ENABLE_COMBINE_NO_STAGE1=True "
                "after auditing the upstream IPC-ordering contract."
            )

        # Kernel does not read ``input`` (Stage 1 bypassed); only check
        # shape/device, relax strict dtype contract.
        self._check_combine_inputs(input, weights, indices, packed_recv_x, strict_input_dtype=False)
        cfg = self.cfg
        stream = torch.cuda.current_stream()

        # fp8_direct_cast fires only when cfg asks for it AND caller
        # passes bf16.
        fp8_dc = cfg.quant_type == "fp8_direct_cast" and input.dtype == torch.bfloat16
        # ``input`` is unread under skip_stage1; a Python-level fp8 cast
        # here would still sit on the cudagraph critical path, so the
        # fused caller is expected to have CV-casted in GEMM2 epilogue.
        if fp8_dc and input.dtype != torch.float8_e4m3fn:
            inp_c = input.to(torch.float8_e4m3fn).contiguous()
        else:
            inp_c = input if input.is_contiguous() else input.contiguous()
        # Wire dtype is fp8 under fp8_direct_cast even when caller
        # passes bf16; otherwise track input dtype.
        c_dtype = torch.float8_e4m3fn if fp8_dc else input.dtype

        # ``input`` is a placeholder under skip_stage1 so cur_tok cannot
        # be inferred from its shape: explicit > stashed > error.
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

        wts_ptr = self.shmem_disp_out_wts.data_ptr() if weights is None else weights.data_ptr()

        _prx_ref = None
        if fp8_dc and packed_recv_x is not None:
            _prx_ref = packed_recv_x.view(torch.bfloat16).to(torch.float8_e4m3fn).contiguous()
            prx_ptr = _prx_ref.data_ptr()
        else:
            prx_ptr = packed_recv_x.data_ptr() if packed_recv_x is not None else 0

        # Key includes dtype so a runtime dtype switch picks a fresh
        # specialization instead of the first-seen kernel.
        no_s1_key = (c_dtype, bool(enable_weights))

        if no_s1_key not in self._comb_no_s1_fn:
            from .dispatch_combine_intranode_kernel import make_combine_jit

            self._comb_no_s1_fn[no_s1_key] = make_combine_jit(
                rank=cfg.rank,
                npes=cfg.world_size,
                experts_per_token=cfg.num_experts_per_token,
                hidden_dim=cfg.hidden_dim,
                max_tok_per_rank=cfg.max_num_inp_token_per_rank,
                block_num=cfg.combine_block_num,
                warp_num_per_block=cfg.combine_warp_num_per_block,
                data_type=c_dtype,
                enable_weights=bool(enable_weights),
                enable_std_moe=cfg.enable_std_moe,
                zero_copy=cfg.zero_copy,
                skip_stage1=True,
                fp8_direct_cast=bool(fp8_dc),
                # Must match dispatch's encoding stride so tok_map
                # sentinel + (peer_pe, dest_lid) decode line up.
                max_recv=self._effective_max_recv,
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
        """Return ``shmem_comb_inp_tok`` viewed as ``dtype`` (mori parity).

        In zero-copy mode the caller MUST write into this view before
        ``op.combine(...)``; the kernel skips Stage 1 and peers read
        this buffer directly.
        """
        cfg = self.cfg
        dt = dtype if dtype is not None else cfg.data_type
        if dt not in _SUPPORTED_TOK_DTYPES:
            raise ValueError(
                f"get_registered_combine_input_buffer: dtype={dt} not in supported set {_SUPPORTED_TOK_DTYPES}"
            )
        row_bytes = _token_bytes_for(dt, cfg.hidden_dim)
        if row_bytes > cfg.max_token_bytes:
            raise ValueError(
                f"get_registered_combine_input_buffer: dtype={dt} needs "
                f"{row_bytes}B/token but buffer is sized for "
                f"{cfg.max_token_bytes}B/token; bump cfg.max_token_type_size."
            )
        h = hidden_dim if hidden_dim > 0 else _token_view_dim_for(dt, cfg.hidden_dim)
        return self.shmem_comb_inp_tok.view(torch.int8).view(dt).view(-1, h)
