# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused MoE Stage2 GEMM + EP combine operator wrapper.

Wraps the host-side launch of :func:`compile_fused_moe_gemm2_combine` and
reuses :class:`FlyDSLDispatchCombineIntraNodeOp`'s shmem buffers, P2P
tables, and xdev barrier.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

from .dispatch_combine_intranode_op import (
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)
from .tmp_mega_gemm2_combine_fused import compile_fused_moe_gemm2_combine


__all__ = ["FlyDSLMoeGemm2CombineOp"]


class FlyDSLMoeGemm2CombineOp:
    """Fused MoE-GEMM2 + EP-Combine operator.

    The GEMM2 epilogue inlines combine Stage 1's P2P token scatter; the
    host then launches a trimmed combine (``combine_no_stage1``) for
    Stage 2/3. This is the only supported mode.

    Parameters
    ----------
    comb_cfg, comb_op
        Shared dispatch-op config and the already-allocated shmem / P2P
        infrastructure.
    inter_dim
        GEMM2 input dim (GEMM1 output); output dim = ``comb_cfg.hidden_dim``.
    tile_m, tile_n, tile_k, persist_m
        GEMM2 tile scheduling parameters.
    a_dtype, b_dtype
        GEMM2 input / weight dtypes.
    """

    # Probed by test scripts via getattr(cls, 'READY', False) to graceful-skip.
    READY: bool = True

    def __init__(
        self,
        *,
        comb_cfg: FlyDSLDispatchCombineConfig,
        comb_op: FlyDSLDispatchCombineIntraNodeOp,
        inter_dim: int,
        tile_m: int = 32, tile_n: int = 128, tile_k: int = 256,
        persist_m: int = 4,
        sort_block_m: int = 0,
        b_nt: int = 2,
        a_dtype: str = "fp8",
        b_dtype: str = "fp4",
        xcd_swizzle: int = 0,
        use_token_flag_sync: bool = False,
        doweight_fused: bool = True,
        force_mode: Optional[str] = None,
    ):
        if force_mode not in (None, "stage1_only"):
            raise ValueError(f"unsupported force_mode={force_mode!r}; only 'stage1_only' is wired")
        self.comb_cfg    = comb_cfg
        self.comb_op     = comb_op
        self.inter_dim   = inter_dim
        self.tile_m      = tile_m
        self.tile_n      = tile_n
        self.tile_k      = tile_k
        self.persist_m   = persist_m
        self.sort_block_m = sort_block_m
        self.b_nt        = b_nt
        self.a_dtype     = a_dtype
        self.b_dtype     = b_dtype
        self.xcd_swizzle = xcd_swizzle
        self.use_token_flag_sync = bool(use_token_flag_sync)
        # MegaMoE.forward consumes only combine_no_stage1's token output. The
        # combine kernel reports out_wts separately but does not apply them to
        # out_tok, so the fused GEMM2 epilogue must apply routing weights by
        # default. The env var remains as a debugging override.
        if os.getenv("FLYDSL_MEGA_STAGE2_DOWEIGHT_FUSED"):
            doweight_fused = os.getenv("FLYDSL_MEGA_STAGE2_DOWEIGHT_FUSED") == "1"
        self.doweight_fused = bool(doweight_fused)

        # fp8_direct_cast: GEMM2 epilogue casts + 1B/elem P2P scatters inline,
        # avoiding the baseline's input.to(fp8) copy.
        self._fp8_cast = (
            getattr(comb_cfg, "quant_type", "none") == "fp8_direct_cast"
            and comb_cfg.data_type == torch.bfloat16
        )
        if self._fp8_cast:
            self._out_dtype_str = "fp8e4m3fn"
        else:
            self._out_dtype_str = "bf16" if comb_cfg.data_type == torch.bfloat16 else "f16"

        self._launch_fn = None
        self._compiled = None
        self._alloc_dummy_tensors()

    def _alloc_dummy_tensors(self):
        """Pre-allocate all placeholder tensors at ctor time (torch.zeros /
        empty would trigger hipMalloc errors inside a cudagraph capture)."""
        comb_cfg = self.comb_cfg
        dev = torch.device("cuda", comb_cfg.rank)
        max_recv = comb_cfg.world_size * comb_cfg.max_num_inp_token_per_rank

        # arg_out is unused on the fused path but the launcher signature
        # still needs a valid tensor.
        self._dummy_out = torch.zeros(1, dtype=comb_cfg.data_type, device=dev)
        self._dummy_bias = torch.empty(0, dtype=comb_cfg.data_type, device=dev)

        # Allocate the combine_no_stage1 input directly in fp8 under fp8_cast
        # to skip combine's internal bf16->fp8 copy.
        inp_dtype = torch.float8_e4m3fn if self._fp8_cast else comb_cfg.data_type
        self._dummy_inp = torch.zeros(
            max_recv, comb_cfg.hidden_dim, dtype=inp_dtype, device=dev
        )

    def info(self) -> dict:
        return dict(
            tile=(self.tile_m, self.tile_n, self.tile_k),
            persist_m=self.persist_m,
            sort_block_m=self.sort_block_m,
            b_nt=self.b_nt,
            xcd_swizzle=self.xcd_swizzle,
            a_dtype=self.a_dtype, b_dtype=self.b_dtype,
            out_dtype=self._out_dtype_str,
            fp8_cast=self._fp8_cast,
            inter_dim=self.inter_dim,
        )

    def run(
        self,
        *,
        a2: torch.Tensor,
        w2: torch.Tensor,
        a2_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        sorted_weights: torch.Tensor,
        num_valid_ids: torch.Tensor,
        wts_buf: Optional[torch.Tensor] = None,
        cur_tok: Optional[int] = None,
        bias: Optional[torch.Tensor] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        """Run the fused GEMM2+combine.

        ``wts_buf`` is the f32[max_recv*k] weight buffer dispatch writes for
        combine; if None we fall back to ``comb_op.shmem_disp_out_wts``.
        ``cur_tok`` is this rank's local token count (combine Stage 3 output
        rows). The ``comb_op.dispatch`` path stashes it internally, but a fused
        stage-1 (which writes ``comb_op``'s routing tables directly without
        calling dispatch) must pass it explicitly or ``combine_no_stage1`` will
        fail to find the stash.

        Returns ``(out_tok, out_wts)``, mirroring ``comb_op.combine``'s outputs.
        """
        return self._run_stage1_only(
            a2=a2, w2=w2, a2_scale=a2_scale, w2_scale=w2_scale,
            sorted_token_ids=sorted_token_ids,
            sorted_expert_ids=sorted_expert_ids,
            sorted_weights=sorted_weights,
            num_valid_ids=num_valid_ids,
            wts_buf=wts_buf,
            cur_tok=cur_tok,
            bias=bias,
            stream=stream,
        )

    def _ensure_launch_fn(self):
        if self._launch_fn is not None:
            return
        comb_cfg = self.comb_cfg
        # Plan B writes into shmem_comb_inp_tok's [0, mt*k) sub-range; k <= npes
        # keeps it in-bounds (mt*k <= mr).
        k = comb_cfg.num_experts_per_token
        if k > comb_cfg.world_size:
            raise ValueError(
                f"FlyDSLMoeGemm2CombineOp (Plan B) requires k <= npes; "
                f"got k={k}, npes={comb_cfg.world_size}."
            )
        self._launch_fn = compile_fused_moe_gemm2_combine(
            model_dim=comb_cfg.hidden_dim,
            inter_dim=self.inter_dim,
            experts=comb_cfg.num_experts_per_rank,
            topk=k,
            tile_m=self.tile_m, tile_n=self.tile_n, tile_k=self.tile_k,
            persist_m=self.persist_m,
            sort_block_m=self.sort_block_m,
            b_nt=self.b_nt,
            a_dtype=self.a_dtype, b_dtype=self.b_dtype,
            out_dtype=self._out_dtype_str,
            rank=comb_cfg.rank, npes=comb_cfg.world_size,
            max_tok_per_rank=comb_cfg.max_num_inp_token_per_rank,
            experts_per_token=k,
            xcd_swizzle=self.xcd_swizzle,
            use_token_flag_sync=self.use_token_flag_sync,
            doweight_fused=self.doweight_fused,
        )

    def _run_stage1_only(self, *, a2, w2, a2_scale, w2_scale,
                         sorted_token_ids, sorted_expert_ids, sorted_weights,
                         num_valid_ids, wts_buf=None, cur_tok=None, bias=None, stream=None):
        """fused GEMM2 (epilogue P2P scatter) + combine_no_stage1 (Stage 2/3)."""
        self._ensure_launch_fn()

        comb_cfg = self.comb_cfg
        comb_op  = self.comb_op
        if stream is None:
            stream = torch.cuda.current_stream()
        s_fx = fx.Stream(stream.cuda_stream)

        # tokens_in MUST be max_recv (not a2.shape[0]); it drives the epilogue
        # row_valid early-exit, otherwise padding rows OOB-read addr_tis and the
        # P2P scatter corrupts combine state.
        tokens_in       = comb_cfg.world_size * comb_cfg.max_num_inp_token_per_rank
        n_in            = comb_cfg.hidden_dim
        k_in            = self.inter_dim
        size_expert_ids = sorted_expert_ids.numel()

        # No upfront zero_() on shmem_comb_inp_tok: a local zero races (cross-
        # device unordered) against peers' fused_gemm2 P2P writes to the same
        # HBM. Safe because every (tok_id, j) in [0, total_recv*k) is fully
        # overwritten by some rank (standard MoE routing sends all k experts);
        # token dropout would instead need a Stage-3 validity mask, not a zero.
        if os.getenv("FLYDSL_MEGA_CLEAR_COMB_INPUT", "0") == "1":
            comb_op.shmem_comb_inp_tok.zero_()
            comb_op.shmem_comb_inp_wts.zero_()
            comb_op.barrier()

        # Default weight source is shmem_disp_out_wts (f32[max_recv*k], indexed
        # as wts[t*k + s]). A caller-supplied wts_buf must keep the same layout.
        addr_wts_buf = (
            fx.Int64(wts_buf.data_ptr())
            if wts_buf is not None
            else comb_op._fx_disp_out_wts
        )

        bias_t = bias if bias is not None else self._dummy_bias

        # First call passes fx-typed scalars so flyc infers the signature;
        # cached calls reuse self._compiled with raw ints.
        common_args = (
            self._dummy_out, a2, w2, a2_scale, w2_scale,
            sorted_token_ids, sorted_expert_ids, sorted_weights,
            num_valid_ids, bias_t,
            comb_op._fx_tis,
            comb_op._fx_p2p_comb_inp,
            addr_wts_buf,
            comb_op._fx_p2p_comb_inp_wts,
            comb_op._fx_local_counter,
            comb_op._fx_p2p_comb_flag,
            comb_op._fx_out_total_recv,
        )
        if self._compiled is None:
            self._compiled = flyc.compile(
                self._launch_fn,
                *common_args,
                fx.Int32(tokens_in), fx.Int32(n_in), fx.Int32(k_in),
                fx.Int32(size_expert_ids),
                s_fx,
            )
        else:
            self._compiled(
                *common_args,
                tokens_in, n_in, k_in, size_expert_ids,
                s_fx,
            )

        # When GEMM2 epilogue does not multiply routing weights, pass the
        # fused stage1 atom-contract weight buffer into combine_no_stage1 so
        # Stage 3b reads the current launch's weights. Falling back to
        # shmem_disp_out_wts would be stale here because dispatch() is bypassed.
        enable_combine_weights = not self.doweight_fused
        combine_weights = None
        if enable_combine_weights and wts_buf is not None:
            combine_weights = wts_buf.view(comb_cfg.max_recv, comb_cfg.num_experts_per_token)
        return comb_op.combine_no_stage1(
            self._dummy_inp,
            combine_weights,
            None,
            cur_tok=cur_tok,
            enable_weights=enable_combine_weights,
        )
