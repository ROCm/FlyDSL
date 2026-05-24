# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Python wrapper for the FlyDSL intra-node DispatchCombine op."""

from __future__ import annotations

from dataclasses import dataclass

import mori.shmem as ms
import torch
from mori.shmem import mori_shmem_create_tensor

import flydsl.compiler as flyc
import flydsl.expr as fx

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


@dataclass
class FlyDSLDispatchCombineConfig:
    rank: int
    world_size: int
    hidden_dim: int
    max_num_inp_token_per_rank: int
    num_experts_per_rank: int
    num_experts_per_token: int
    data_type: torch.dtype = torch.bfloat16
    warp_num_per_block: int = 16
    block_num: int = 80
    scale_dim: int = 0
    scale_type_size: int = 0
    enable_std_moe: bool = False
    use_external_inp_buf: bool = True
    quant_type: str = "none"

    @property
    def is_fp4(self):
        return self.data_type == torch.float4_e2m1fn_x2

    @property
    def elem_size(self):
        return torch.tensor([], dtype=self.data_type).element_size()

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
    def max_recv(self):
        return self.world_size * self.max_num_inp_token_per_rank

    @property
    def scale_bytes(self):
        return self.scale_dim * self.scale_type_size


class FlyDSLDispatchCombineIntraNodeOp:

    def __init__(self, config):
        self.cfg = config
        # Static contract check on the config itself (task 2).
        # Run BEFORE any GPU buffer allocation so that misconfigurations
        # surface as a clear ValueError instead of an opaque downstream
        # HIP/MLIR error during ``_alloc_buffers`` or JIT compilation.
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

        self._disp_fn = make_dispatch_jit(
            rank=r,
            npes=config.world_size,
            experts_per_rank=config.num_experts_per_rank,
            experts_per_token=config.num_experts_per_token,
            hidden_dim=config.hidden_dim,
            max_tok_per_rank=config.max_num_inp_token_per_rank,
            block_num=config.block_num,
            warp_num_per_block=config.warp_num_per_block,
            data_type=config.data_type,
            scale_dim=config.scale_dim,
            scale_type_size=config.scale_type_size,
            enable_std_moe=config.enable_std_moe,
        )

        # Mixed-dtype Stage 1 (mori UseFp8DirectCast equivalent): when
        # ``fp8_direct_cast`` is on, the user feeds bf16 input to
        # ``combine()`` and the kernel performs an inline bf16 → fp8 cast
        # in Stage 1 before the P2P scatter.  This avoids an extra ~12μs
        # ``input.to(fp8).contiguous()`` PyTorch elementwise kernel that
        # would otherwise sit on the cudagraph critical path.  The
        # external input/output dtype seen by the caller stays bf16
        # (i.e. ``data_type=cfg.data_type``); only the on-wire transport
        # dtype switches to fp8 inside the kernel.
        _use_fp8_cast = config.quant_type == "fp8_direct_cast" and config.data_type == torch.bfloat16
        self._comb_fn = make_combine_jit(
            rank=r,
            npes=config.world_size,
            experts_per_token=config.num_experts_per_token,
            hidden_dim=config.hidden_dim,
            max_tok_per_rank=config.max_num_inp_token_per_rank,
            block_num=config.block_num,
            warp_num_per_block=config.warp_num_per_block,
            data_type=config.data_type,
            enable_weights=True,
            enable_std_moe=config.enable_std_moe,
            use_p2p_read=not config.use_external_inp_buf,
            fp8_direct_cast=_use_fp8_cast,
        )
        self._use_fp8_cast = _use_fp8_cast

        # The cross-device barrier flag must start at 1; otherwise the very
        # first wait_until_equals(slot, 0) would be satisfied immediately by
        # the zero-initialized memory and skip the actual synchronization.
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

        self._disp_compiled = None
        self._comb_compiled = None
        # Lazy-compiled skip_stage1 combine variant used by the fused
        # GEMM2-combine path: the upstream fused kernel has already P2P-
        # scattered tokens / weights into shmem_comb_inp[_wts], so the
        # combine kernel only runs Stage 2 (CrossDeviceBarrier) + Stage 3
        # (local weighted-accum).
        self._comb_no_s1_fn = None
        self._comb_no_s1_compiled = None

    def _alloc_buffers(self):
        cfg = self.cfg
        npes = cfg.world_size
        k = cfg.num_experts_per_token
        mt = cfg.max_num_inp_token_per_rank
        mr = cfg.max_recv  # npes * mt

        tb = cfg.token_bytes
        tok_i16_mr = (mr * tb + 1) // 2
        tok_i16_mt = (mt * tb + 1) // 2

        # Symmetric shmem buffers. All buffers below are P2P-accessed by
        # peer PEs inside dispatch/combine kernels, except
        # shmem_comb_out_{tok,wts} which are local-only writes today —
        # they stay on the shmem heap to match mori's allocation layout
        # and to leave room for future fused ops that P2P-fetch them.
        self.shmem_disp_out_tok = mori_shmem_create_tensor((tok_i16_mr,), torch.int16)
        self.shmem_disp_out_wts = mori_shmem_create_tensor((mr * k,), torch.float32)
        self.shmem_disp_out_idx = mori_shmem_create_tensor((mr * k,), torch.int32)
        scale_total = mr * cfg.scale_bytes if cfg.scale_bytes > 0 else 1
        self.shmem_out_scales = mori_shmem_create_tensor((scale_total,), torch.int8)
        self.shmem_tok_off = mori_shmem_create_tensor((1,), torch.int32)
        self.shmem_recv_tok_num = mori_shmem_create_tensor((npes,), torch.int32)
        self.shmem_tok_id_to_src = mori_shmem_create_tensor((mr,), torch.int32)
        self.shmem_comb_inp_tok = mori_shmem_create_tensor((tok_i16_mr,), torch.int16)
        self.shmem_comb_out_tok = mori_shmem_create_tensor((tok_i16_mt,), torch.int16)
        self.shmem_comb_inp_wts = mori_shmem_create_tensor((mr * k,), torch.float32)
        self.shmem_comb_out_wts = mori_shmem_create_tensor((mt * k,), torch.float32)
        self.shmem_xdev_bar_mem = mori_shmem_create_tensor((npes,), torch.int64)

        # mori_shmem_create_tensor goes through shmem_malloc, which returns
        # uninitialized raw memory.  In the fused MoE-GEMM2 + EP-Combine
        # path the GEMM2 epilogue decodes (dest_pe, dest_lid) from
        # shmem_tok_id_to_src; out-of-bounds garbage there would trigger
        # LDS OOB and corrupt arbitrary global state, so every symmetric
        # buffer that combine touches directly is zeroed up-front to
        # guarantee:
        #   - shmem_tok_id_to_src[t] for slots never written by dispatch
        #     decodes to (pe=0, lid=0), making the P2P scatter degenerate
        #     into a harmless duplicate write into a single slot;
        #   - shmem_xdev_bar_mem starts at 0 so the first
        #     CrossDeviceBarrier wait never observes stale data (the
        #     protocol relies on cur_flag monotonic increase);
        #   - shmem_comb_inp_{tok,wts} start at 0 so combine_no_stage1's
        #     Stage 3 accumulation never folds garbage into the result.
        self.shmem_tok_id_to_src.zero_()
        self.shmem_comb_inp_tok.zero_()
        self.shmem_comb_inp_wts.zero_()
        self.shmem_xdev_bar_mem.zero_()

        # Local device buffers
        self.dest_pe_ctr = torch.zeros(npes, dtype=torch.int32, device=self._dev)
        self.disp_bar = torch.zeros(1, dtype=torch.int32, device=self._dev)
        self.comb_bar = torch.zeros(1, dtype=torch.int32, device=self._dev)
        self.total_recv = torch.zeros(1, dtype=torch.int32, device=self._dev)
        sentinel = cfg.world_size * mr
        self.dest_tok_map = torch.full((mt * k,), sentinel, dtype=torch.int32, device=self._dev)

        # StdMoE buffers
        if cfg.enable_std_moe:
            epr = cfg.num_experts_per_rank
            max_tok_per_expert = mr  # world_size * max_num_inp_token_per_rank
            self.packed_recv_count = torch.zeros(epr, dtype=torch.int32, device=self._dev)
            self.packed_recv_src_info = torch.zeros(epr * max_tok_per_expert, dtype=torch.int32, device=self._dev)
            self.disp_tok_to_ep_slot_map = torch.full((mr * k,), -1, dtype=torch.int64, device=self._dev)
            # i64 ticket counter for the in-kernel grid barrier
            # (StdMoE Phase 4): the kernel does NOT reset the counter,
            # it uses ``atomic_add``'s returned ticket to derive the
            # current launch's epoch and wait target.  See the
            # ``in-kernel grid barrier`` block in the dispatch kernel
            # for the protocol.  i64 prevents wraparound (>290k years
            # at 1 M launches/s).
            self.disp_grid_bar = torch.zeros(1, dtype=torch.int64, device=self._dev)
        else:
            self.packed_recv_count = torch.zeros(1, dtype=torch.int32, device=self._dev)
            self.packed_recv_src_info = torch.zeros(1, dtype=torch.int32, device=self._dev)
            self.disp_tok_to_ep_slot_map = torch.zeros(1, dtype=torch.int64, device=self._dev)
            self.disp_grid_bar = torch.zeros(1, dtype=torch.int64, device=self._dev)

    def barrier(self):
        ms.shmem_barrier_all()

    def reset(self):
        self.barrier()

    # ------------------------------------------------------------------
    # Config + runtime input contracts (task 2): keep the wrapper's
    # accepted config / shapes / dtypes in sync with mori's intranode
    # dispatch/combine, fail fast on misuse instead of silently
    # triggering OOB writes inside the kernel or opaque downstream
    # HIP / MLIR errors during JIT compilation.
    # ------------------------------------------------------------------
    def _check_config(self):
        """Static contract check on ``self.cfg``.

        Called once from ``__init__`` right after ``self.cfg`` is bound
        and BEFORE any GPU memory allocation, so that the user sees the
        actual config violation up-front.  Mirrors the constraints
        encoded in mori's intranode ``EpDispatchCombineConfig`` so the
        two ops remain drop-in interchangeable in tests/benchmarks.
        """
        cfg = self.cfg

        # Type sanity.
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
        # ``ballot`` over k_slot lanes only covers the first 64 lanes per
        # warp, so k <= 64 is a hard kernel constraint.  In practice k
        # is tiny (<=16) but we keep the upper bound explicit.
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
        # fp8_direct_cast pre-condition lifted out of make_combine_kernel:
        # the std-MoE Stage 1 / Stage 3 path uses ``_weighted_accum_experts``
        # which has not been retrofitted for asymmetric I/O dtypes, so the
        # two switches are mutually exclusive.
        if cfg.quant_type == "fp8_direct_cast" and cfg.enable_std_moe:
            raise NotImplementedError(
                "quant_type='fp8_direct_cast' is not yet supported together with "
                "enable_std_moe=True (the std-MoE Stage 1 / Stage 3 path uses "
                "weighted-accum helpers that have not been retrofitted for "
                "asymmetric I/O dtypes)"
            )

        # Token-row byte alignment: kernel streams 16B (v4i32) per lane.
        if cfg.token_bytes % _TOK_BYTES_ALIGN != 0:
            raise ValueError(
                f"token row bytes ({cfg.token_bytes}) must be a multiple of "
                f"{_TOK_BYTES_ALIGN} for v4i32 vector loads; check hidden_dim "
                f"({cfg.hidden_dim}) and data_type ({cfg.data_type})"
            )

        # Scales consistency.
        if cfg.scale_dim < 0 or cfg.scale_type_size < 0:
            raise ValueError(
                f"scale_dim/scale_type_size must be non-negative, got " f"({cfg.scale_dim}, {cfg.scale_type_size})"
            )
        if (cfg.scale_dim == 0) != (cfg.scale_type_size == 0):
            raise ValueError(
                "scale_dim and scale_type_size must be both zero or both "
                f"positive, got ({cfg.scale_dim}, {cfg.scale_type_size})"
            )

        # Launch geometry.
        if cfg.warp_num_per_block <= 0:
            raise ValueError(f"warp_num_per_block must be positive, got {cfg.warp_num_per_block}")
        if cfg.block_num <= 0:
            raise ValueError(f"block_num must be positive, got {cfg.block_num}")

        # Local expert id encoding contract: ``expert_id = dest_pe *
        # num_experts_per_rank + local_expert_id`` requires
        # ``num_experts_per_rank`` to be representable as a positive i32
        # and the total num_experts to fit i32 without overflow during
        # divui by num_experts_per_rank.
        total_experts = cfg.world_size * cfg.num_experts_per_rank
        if total_experts > (1 << 31) - 1:
            raise ValueError(
                f"total experts ({cfg.world_size} * {cfg.num_experts_per_rank} = {total_experts}) "
                "exceeds int32 range"
            )

        # LDS budget pre-flight: the combine kernel reserves an LDS-resident
        # table of P2P base pointers (``i64[npes]`` for tokens, plus a
        # second ``i64[npes]`` for weights when enable_weights), 8-byte
        # aligned, both padded out to 128 bytes by ``SmemAllocator``.
        # Reject configs that would exceed the device LDS limit here —
        # the same check fires again inside the JIT (``check_smem_capacity``)
        # but surfacing it at op construction gives a much clearer error
        # than the opaque downstream MLIR failure.
        self._check_lds_capacity()

    def _check_lds_capacity(self):
        """Reject configs whose combine-kernel LDS layout overflows the
        running GPU's shared-memory budget.
        """
        from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP

        from .dispatch_combine_intranode_kernel import _detect_gpu_arch

        cfg = self.cfg

        # Mirror the layout in ``make_combine_jit``: two i64[npes] tables,
        # 8-byte aligned, then the whole arena rounded up to 128 bytes
        # (matching ``SmemAllocator.finalize``'s ``_align(ptr, 128)``).
        def _align(p, a):
            return (p + a - 1) // a * a

        # Always include the weight table because ``make_combine_jit`` is
        # invoked with ``enable_weights=True`` from the wrapper -- both
        # tables exist whenever the wrapper is used.
        ptr = 0
        ptr = _align(ptr, 8) + cfg.world_size * 8  # token P2P base table
        ptr = _align(ptr, 8) + cfg.world_size * 8  # weight P2P base table
        lds_bytes = max(_align(ptr, 128), 128)

        # ``_detect_gpu_arch`` is the same lookup the kernel uses; it
        # avoids spawning ``rocm_agent_enumerator`` so it's safe to
        # call inside an 8-way torch.multiprocessing.spawn worker.
        arch = _detect_gpu_arch()
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
        expected_hdim = cfg.hidden_dim // 2 if cfg.is_fp4 else cfg.hidden_dim
        if input.shape[1] != expected_hdim:
            raise ValueError(
                f"input.shape[1]={input.shape[1]} != expected {expected_hdim} "
                f"(hidden_dim={cfg.hidden_dim}, is_fp4={cfg.is_fp4})"
            )
        if input.dtype != cfg.data_type:
            raise ValueError(f"input.dtype={input.dtype} != cfg.data_type={cfg.data_type}")

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
        expected_hdim = cfg.hidden_dim // 2 if cfg.is_fp4 else cfg.hidden_dim
        if input.shape[1] != expected_hdim:
            raise ValueError(
                f"combine input.shape[1]={input.shape[1]} != expected "
                f"{expected_hdim} (hidden_dim={cfg.hidden_dim}, is_fp4={cfg.is_fp4})"
            )
        if input.shape[0] > cfg.max_recv:
            raise ValueError(f"combine input rows={input.shape[0]} exceeds max_recv={cfg.max_recv}")
        if strict_input_dtype:
            # External dtype contract: caller feeds bf16 under fp8_direct_cast,
            # otherwise the configured token dtype.
            expected_inp_dt = (
                torch.bfloat16
                if (cfg.quant_type == "fp8_direct_cast" and cfg.data_type == torch.bfloat16)
                else cfg.data_type
            )
            if input.dtype != expected_inp_dt:
                raise ValueError(
                    f"combine input.dtype={input.dtype} != expected {expected_inp_dt} " f"(quant_type={cfg.quant_type})"
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

    def dispatch(self, input, weights, scales, indices, packed_recv_x=None):
        self._check_dispatch_inputs(input, weights, scales, indices, packed_recv_x)
        cfg = self.cfg
        inp_cur_tok = input.shape[0]
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
            # ``disp_grid_bar`` is intentionally NOT reset here: the
            # kernel itself is CUDAGraph-replay safe via the in-kernel
            # ticket-based grid barrier (see StdMoE Phase 4 in
            # ``make_dispatch_kernel``).  The counter grows
            # monotonically across replays and every block derives the
            # current launch's wait target from its own atomic-add
            # ticket -- no host-side memset required.

        # _std_args layout MUST match the trailing 8 ``addr_*`` params of
        # ``dispatch_launch`` (kernel side, in order):
        #   17 addr_shmem_tok                  <- self._fx_out_tok
        #   18 addr_shmem_idx                  <- self._fx_out_idx
        #   19 addr_shmem_tok_id_to_src        <- self._fx_out_shmem_tok_id_to_src
        #   20 addr_out_packed_recv_x          <- prx_ptr (runtime)
        #   21 addr_out_packed_recv_count      <- self._fx_packed_recv_count (or 0)
        #   22 addr_out_packed_recv_src_info   <- self._fx_packed_recv_src_info
        #   23 addr_out_disp_tok_map           <- self._fx_disp_tok_map
        #   24 addr_disp_grid_bar              <- self._fx_disp_grid_bar
        # Earlier this list was missing ``addr_out_packed_recv_x``; the
        # ``prx_ptr`` was inserted ahead of _std_args (at JIT slot 17 =
        # ``addr_shmem_tok``) instead, which shifted slots 17..20 by one
        # and silently disabled StdMoE Phase 4 (packed_recv_x stayed all
        # zero, packed_recv_count stayed zero) without any runtime
        # diagnostic.
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

        if self._disp_compiled is None:
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
            self._disp_compiled = flyc.compile(self._disp_fn, *args)
        else:
            self._disp_compiled(
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

        mr = cfg.max_recv
        k = cfg.num_experts_per_token

        out_tok = (
            self.shmem_disp_out_tok.view(torch.int8)[: mr * cfg.token_bytes]
            .view(cfg.data_type)
            .view(mr, cfg.token_view_dim)
        )
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

    def combine(
        self,
        input,
        weights,
        indices,
        packed_recv_x=None,
        cur_tok=None,
        block_num=-1,
        rdma_block_num=-1,
        warp_per_block=-1,
        use_external_inp_buf=-1,
        call_reset=False,
    ):
        self._check_combine_inputs(input, weights, indices, packed_recv_x)
        cfg = self.cfg
        stream = torch.cuda.current_stream()

        # In _use_fp8_cast mode, the combine kernel does the bf16 → fp8 cast
        # inline in Stage 1 (mori UseFp8DirectCast equivalent), so the wrapper
        # passes bf16 input straight through.  Skipping the PyTorch-level
        # ``.to(fp8).contiguous()`` saves ~12μs per iter on the cudagraph
        # critical path.
        inp_c = input if input.is_contiguous() else input.contiguous()
        _cur_tok = cur_tok if cur_tok is not None else cfg.max_num_inp_token_per_rank

        wts_ptr = self.shmem_disp_out_wts.data_ptr() if weights is None else weights.data_ptr()

        _prx_ref = None
        if self._use_fp8_cast and packed_recv_x is not None:
            # std-MoE expert-major buffer (`packed_recv_x`) is produced in bf16
            # by the upstream pipeline; downstream Stage 1 reads it in fp8
            # dtype, so we still cast here.  This branch is independent from
            # the regular combine input path above.
            _prx_ref = packed_recv_x.view(torch.bfloat16).to(torch.float8_e4m3fn).contiguous()
            prx_ptr = _prx_ref.data_ptr()
        else:
            prx_ptr = packed_recv_x.data_ptr() if packed_recv_x is not None else 0

        _std_args_comb = (
            fx.Int64(prx_ptr),
            self._fx_disp_tok_map,
            self._fx_disp_out_wts,
        )

        if self._comb_compiled is None:
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
            self._comb_compiled = flyc.compile(self._comb_fn, *args)
        else:
            self._comb_compiled(
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

        # fp8_direct_cast contract: external dtype is bf16 on both ends; the
        # combine kernel itself writes bf16 to ``shmem_comb_out_tok`` (Stage 3
        # _from_accum casts v4f32 → v4bf16 inline), so we view the buffer as
        # bf16 directly with no extra PyTorch-level cast on the critical path.
        out_tok = (
            self.shmem_comb_out_tok.view(torch.int8)[: mt * cfg.token_bytes]
            .view(cfg.data_type)
            .view(mt, cfg.token_view_dim)
        )
        out_wts = self.shmem_comb_out_wts.view(mt, k)

        if call_reset:
            self.reset()
        return out_tok, out_wts

    # Class-level gate for the reserved ``combine_no_stage1`` entry point.
    # Default ``False``: the intranode dispatch/combine baseline does NOT
    # expose this API to end users — the contract (pre-populated
    # ``shmem_comb_inp_tok``, caller-supplied ``cur_tok``, upstream IPC
    # ordering vs. the weight P2P) is only met by the fused
    # GEMM2+combine production path on the ``yanbo/fused_gemm2_combine``
    # branch.  That branch flips this flag after staging its end-to-end
    # validation.  Anyone toggling this in isolation must read the full
    # contract in ``combine_no_stage1.__doc__`` and own correctness.
    _ENABLE_COMBINE_NO_STAGE1 = False

    def combine_no_stage1(
        self, input, weights, indices, packed_recv_x=None, cur_tok=None, call_reset=False, enable_weights: bool = True
    ):
        """Skip-Stage1 variant of ``combine`` (reserved API, not yet exposed).

        Semantics: bypass the P2P scatter (the upstream fused kernel has
        already populated shmem_comb_inp[_wts]) and only run Stage 2
        (CrossDeviceBarrier) + Stage 3 (local weighted-accum).

        Status
        ------
        Reserved for the fused GEMM2+combine path.  The standalone
        intranode wrapper does not currently expose this entry point;
        invoking it raises ``NotImplementedError`` unless the class-level
        flag ``_ENABLE_COMBINE_NO_STAGE1`` is explicitly set to ``True``
        (intended for the fused-path integration on the
        ``yanbo/fused_gemm2_combine`` branch).

        Parameters
        ----------
        cur_tok
            **Caller-supplied actual token count** consumed by Stage 3.
            Stage 3 iterates ``range(global_warp_id, cur_rank_num_token,
            global_warp_num)`` where ``cur_rank_num_token`` is this
            argument (NOT ``self.total_recv``).  When ``None`` it falls
            back to ``cfg.max_num_inp_token_per_rank``, which over-
            iterates harmlessly only when the unrouted slots are
            zero-initialized (see ``shmem_comb_inp_tok.zero_()`` in
            ``_alloc_buffers``).  The fused path MUST pass the real
            count to avoid wasted Stage 3 iterations.
        enable_weights
            ``True`` (default) keeps the Stage 1 weight scatter and the
            Stage 3b weight accumulate inside the combine kernel.  The
            weight scatter is intentionally kept here (instead of being
            folded into the upstream fused GEMM2 epilogue) because the
            16B narrow stores are silently dropped by the ROCm IPC
            fabric when they race with the upstream token P2P, so they
            must be issued by the combine kernel on the static fabric.
            ``False`` is for weight-free fused paths (the upstream fused
            MoE has already collapsed the weights, so the combine side
            does not need ``out_wts``): both the weight scatter and
            Stage 3b are completely DCE'd, saving ~3-5 us.  The two
            variants use distinct JIT caches.

        Contract: before invocation the fused kernel must guarantee:
              - shmem_comb_inp_tok already contains every token this PE
                will consume (laid out in max_tok_per_rank slots);
              - shmem_comb_inp_wts already contains the matching weights
                (only when enable_weights=True);
              - ``cur_tok`` matches the actual number of received tokens
                consumed by Stage 3 (NOT read from ``self.total_recv``).
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

        # Stage-1 is bypassed: the kernel does not actually read ``input``,
        # so any contiguous dtype works (the upstream fused GEMM2 epilogue
        # has already populated ``shmem_comb_inp_tok``).  We still check
        # shape/device but relax the strict dtype contract.
        self._check_combine_inputs(input, weights, indices, packed_recv_x, strict_input_dtype=False)
        cfg = self.cfg
        stream = torch.cuda.current_stream()

        # When skip_stage1=True (the only mode this method ever compiles for),
        # the combine kernel does NOT read inp_c — Stage 1 is bypassed and the
        # kernel reads from shmem_comb_inp_tok directly (already populated by
        # the upstream fused GEMM2 epilogue P2P scatter).  So skip the
        # potentially-expensive Python-level fp8 cast (.to(fp8) + .contiguous())
        # if the caller gave us a fp8 input or even a placeholder bf16: the
        # cast is a ~12us elementwise kernel that gets captured by cudagraph
        # and ends up serially on the chain critical path for nothing.
        # Caller (fused op wrapper) already CV-casted in the GEMM2 epilogue.
        if self._use_fp8_cast and input.dtype != torch.float8_e4m3fn:
            inp_c = input.to(torch.float8_e4m3fn).contiguous()
        else:
            inp_c = input if input.is_contiguous() else input.contiguous()
        _cur_tok = cur_tok if cur_tok is not None else cfg.max_num_inp_token_per_rank

        wts_ptr = self.shmem_disp_out_wts.data_ptr() if weights is None else weights.data_ptr()

        _prx_ref = None
        if self._use_fp8_cast and packed_recv_x is not None:
            _prx_ref = packed_recv_x.view(torch.bfloat16).to(torch.float8_e4m3fn).contiguous()
            prx_ptr = _prx_ref.data_ptr()
        else:
            prx_ptr = packed_recv_x.data_ptr() if packed_recv_x is not None else 0

        # Two JIT'd variants are cached, keyed by enable_weights.
        if not isinstance(self._comb_no_s1_fn, dict):
            self._comb_no_s1_fn = {}
            self._comb_no_s1_compiled = {}

        if enable_weights not in self._comb_no_s1_fn:
            from .dispatch_combine_intranode_kernel import make_combine_jit

            # Same fp8_direct_cast mixed-dtype contract as in ``combine``
            # above: external dtype is the configured ``cfg.data_type``
            # (bf16 under fp8_direct_cast), transport dtype switches to
            # fp8 inside the kernel; Stage 3 ``_from_accum`` casts
            # f32 → bf16 inline so the kernel writes bf16 straight to
            # ``shmem_comb_out_tok``.
            #
            # See the ``enable_weights`` doc above for the rationale of
            # the two variants: ``False`` lets const_expr DCE the weight
            # scatter + Stage 3b (~3-5 us); ``True`` keeps the weight
            # scatter inside combine to dodge the IPC-fabric race against
            # the upstream token P2P.
            self._comb_no_s1_fn[enable_weights] = make_combine_jit(
                rank=cfg.rank,
                npes=cfg.world_size,
                experts_per_token=cfg.num_experts_per_token,
                hidden_dim=cfg.hidden_dim,
                max_tok_per_rank=cfg.max_num_inp_token_per_rank,
                block_num=cfg.block_num,
                warp_num_per_block=cfg.warp_num_per_block,
                data_type=cfg.data_type,
                enable_weights=bool(enable_weights),
                enable_std_moe=cfg.enable_std_moe,
                use_p2p_read=not cfg.use_external_inp_buf,
                skip_stage1=True,
                fp8_direct_cast=self._use_fp8_cast,
            )

        if enable_weights not in self._comb_no_s1_compiled:
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
            self._comb_no_s1_compiled[enable_weights] = flyc.compile(self._comb_no_s1_fn[enable_weights], *args)
        else:
            self._comb_no_s1_compiled[enable_weights](
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

        # fp8_direct_cast contract: combine kernel writes bf16 to
        # ``shmem_comb_out_tok`` directly (see ``combine`` above for details).
        out_tok = (
            self.shmem_comb_out_tok.view(torch.int8)[: mt * cfg.token_bytes]
            .view(cfg.data_type)
            .view(mt, cfg.token_view_dim)
        )
        out_wts = self.shmem_comb_out_wts.view(mt, k)

        if call_reset:
            self.reset()
        return out_tok, out_wts

    def get_dispatch_src_token_pos(self):
        torch.cuda.synchronize()
        n = int(self.total_recv[0].item())
        return self.shmem_tok_id_to_src[:n].clone()

    def get_registered_combine_input_buffer(self, dtype, hidden_dim=-1):
        h = hidden_dim if hidden_dim > 0 else self.cfg.token_view_dim
        dt = dtype if dtype is not None else self.cfg.data_type
        return self.shmem_comb_inp_tok.view(torch.int8).view(dt).view(-1, h)
