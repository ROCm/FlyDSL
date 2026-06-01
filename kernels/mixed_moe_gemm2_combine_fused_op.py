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
from .mixed_moe_gemm2_combine_fused import (
    compile_fused_moe_gemm2_combine,
    estimate_max_resident_blocks,
)


__all__ = ["FlyDSLMoeGemm2CombineOp"]


_FORCE_MODE_ENV = "FLYDSL_GEMM2_COMBINE_FORCE_MODE"
_VALID_MODES = ("auto", "full", "stage1_only")


def _resolve_force_mode(force_mode: str) -> str:
    """Normalize force_mode: env overrides; ``fallback`` aliases ``stage1_only``."""
    env = os.environ.get(_FORCE_MODE_ENV, "").strip().lower()
    mode = (env or force_mode).strip().lower()
    if mode == "fallback":
        mode = "stage1_only"
    if mode not in _VALID_MODES:
        raise ValueError(
            f"force_mode must be one of {_VALID_MODES} (or 'fallback'), got {mode!r}"
        )
    return mode


class FlyDSLMoeGemm2CombineOp:
    """Fused MoE-GEMM2 + EP-Combine operator.

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
    force_mode
        ``auto`` / ``full`` / ``stage1_only`` (``fallback`` is an alias).
        Also overridable via the ``FLYDSL_GEMM2_COMBINE_FORCE_MODE`` env var.
    """

    # Compatibility hook: upstream test scripts probe ``getattr(cls,
    # 'READY', False)`` to decide whether to graceful-skip; always True
    # once the fused path is fully on.
    READY: bool = True

    def __init__(
        self,
        *,
        comb_cfg: FlyDSLDispatchCombineConfig,
        comb_op: FlyDSLDispatchCombineIntraNodeOp,
        inter_dim: int,
        tile_m: int = 32, tile_n: int = 128, tile_k: int = 128,
        persist_m: int = 4,
        a_dtype: str = "fp8",
        b_dtype: str = "fp4",
        force_mode: str = "auto",
        xcd_swizzle: int = 0,
        use_token_flag_sync: bool = False,
    ):
        self.comb_cfg    = comb_cfg
        self.comb_op     = comb_op
        self.inter_dim   = inter_dim
        self.tile_m      = tile_m
        self.tile_n      = tile_n
        self.tile_k      = tile_k
        self.persist_m   = persist_m
        self.a_dtype     = a_dtype
        self.b_dtype     = b_dtype
        self.xcd_swizzle = xcd_swizzle
        self.use_token_flag_sync = bool(use_token_flag_sync)
        self.force_mode  = _resolve_force_mode(force_mode)

        # fp8_direct_cast: GEMM2 epilogue does cvt_pk_fp8_f32 + 1B/elem P2P
        # scatter inline, avoiding the baseline's input.to(fp8) copy so the
        # fused / baseline byte streams are directly comparable.
        self._fp8_cast = (
            getattr(comb_cfg, "quant_type", "none") == "fp8_direct_cast"
            and comb_cfg.data_type == torch.bfloat16
        )
        if self._fp8_cast:
            self._out_dtype_str = "fp8e4m3fn"
        else:
            self._out_dtype_str = "bf16" if comb_cfg.data_type == torch.bfloat16 else "f16"

        self.chosen_mode = self._select_mode()

        self._launch_fn = None
        self._compiled = None
        self._alloc_dummy_tensors()

    def _select_mode(self) -> str:
        """auto: pick ``full`` when the GEMM2 natural grid is fully resident,
        otherwise ``stage1_only``.

        Note: ``persist_m <= 0`` triggers GEMM2 ``persistent`` mode (grid_y = cu_num,
        CTA round-robin over M tiles). In that mode ``nat_blocks`` is bounded by
        cu_num by construction; but ``_run_full`` (single-kernel fusion) is still
        a PR2 stub, so we hard-fall back to ``stage1_only`` to avoid hitting
        ``NotImplementedError("_run_full ...")``.
        """
        if self.force_mode != "auto":
            return self.force_mode
        if self.persist_m <= 0:
            return "stage1_only"
        comb_cfg = self.comb_cfg
        max_resident = estimate_max_resident_blocks(chip=comb_cfg.chip, block_dim=256)
        max_recv = comb_cfg.world_size * comb_cfg.max_num_inp_token_per_rank
        per_e_blocks = (max_recv + self.tile_m - 1) // self.tile_m
        nat_blocks = (comb_cfg.hidden_dim // self.tile_n) * (
            (per_e_blocks * comb_cfg.num_experts_per_rank + self.persist_m - 1)
            // self.persist_m
        )
        return "full" if nat_blocks <= max_resident else "stage1_only"

    def _alloc_dummy_tensors(self):
        """cudagraph-compatible: all placeholder tensors must be allocated
        at ctor time.

        Inside capture, ``torch.zeros`` / ``torch.empty`` cannot be called
        because they trigger hipMalloc errors.
        """
        comb_cfg = self.comb_cfg
        dev = torch.device("cuda", comb_cfg.rank)
        max_recv = comb_cfg.world_size * comb_cfg.max_num_inp_token_per_rank

        # arg_out is unread/unwritten on the fused path; the launcher signature
        # still requires a valid tensor.
        self._dummy_out = torch.zeros(1, dtype=comb_cfg.data_type, device=dev)
        self._dummy_bias = torch.empty(0, dtype=comb_cfg.data_type, device=dev)

        # combine_no_stage1 input placeholder: under fp8_cast we allocate
        # directly in fp8 to avoid the combine-internal bf16->fp8 .to() +
        # .contiguous() copy (~12us / chain).
        inp_dtype = torch.float8_e4m3fn if self._fp8_cast else comb_cfg.data_type
        self._dummy_inp = torch.zeros(
            max_recv, comb_cfg.hidden_dim, dtype=inp_dtype, device=dev
        )

    def info(self) -> dict:
        return dict(
            chosen_mode=self.chosen_mode,
            force_mode=self.force_mode,
            tile=(self.tile_m, self.tile_n, self.tile_k),
            persist_m=self.persist_m,
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
        dispatch_total_recv: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        """Run the fused GEMM2+combine.

        ``wts_buf`` is the f32[max_recv*k] weight buffer dispatch writes for
        combine; if None we fall back to ``comb_op.shmem_disp_out_wts``.
        ``dispatch_total_recv`` is consumed only by ``full`` mode and ignored
        under ``stage1_only``.

        Returns ``(out_tok, out_wts)``, mirroring ``comb_op.combine``'s outputs.
        """
        del dispatch_total_recv  # reserved for full mode
        if self.chosen_mode == "full":
            return self._run_full(
                a2, w2, a2_scale, w2_scale,
                sorted_token_ids, sorted_expert_ids, sorted_weights,
                num_valid_ids, bias,
            )
        return self._run_stage1_only(
            a2=a2, w2=w2, a2_scale=a2_scale, w2_scale=w2_scale,
            sorted_token_ids=sorted_token_ids,
            sorted_expert_ids=sorted_expert_ids,
            sorted_weights=sorted_weights,
            num_valid_ids=num_valid_ids,
            wts_buf=wts_buf,
            bias=bias,
            stream=stream,
        )

    def _ensure_launch_fn(self):
        if self._launch_fn is not None:
            return
        comb_cfg = self.comb_cfg
        # Plan B: topk matches dispatch experts_per_token (s field used as
        # j_global). baseline shmem_comb_inp_tok size stays at mr; Plan B
        # writes into the [0, mt*k) sub-range, guaranteed in-bounds by the
        # ctor-time k <= npes check.
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
            a_dtype=self.a_dtype, b_dtype=self.b_dtype,
            out_dtype=self._out_dtype_str,
            rank=comb_cfg.rank, npes=comb_cfg.world_size,
            max_tok_per_rank=comb_cfg.max_num_inp_token_per_rank,
            experts_per_token=k,
            mode=self.chosen_mode,
            xcd_swizzle=self.xcd_swizzle,
            use_token_flag_sync=self.use_token_flag_sync,
        )

    def _run_stage1_only(self, *, a2, w2, a2_scale, w2_scale,
                         sorted_token_ids, sorted_expert_ids, sorted_weights,
                         num_valid_ids, wts_buf=None, bias=None, stream=None):
        """fused GEMM2 (epilogue P2P scatter) + combine_no_stage1 (Stage 2/3)."""
        self._ensure_launch_fn()

        comb_cfg = self.comb_cfg
        comb_op  = self.comb_op
        if stream is None:
            stream = torch.cuda.current_stream()
        s_fx = fx.Stream(stream.cuda_stream)

        # tokens_in MUST be max_recv = world_size * max_num_inp_token_per_rank,
        # matching baseline run_moe_stage2 / _build_gemm2_static_inputs. Mis-
        # passing a2.shape[0] (= max_recv * inter_dim under the 1D view) breaks
        # the row_valid early-exit; padding sentinels then look valid, the
        # epilogue OOB-reads addr_tis, P2P flies to arbitrary remote addresses
        # and corrupts combine control state, deadlocking the xdev barrier.
        tokens_in       = comb_cfg.world_size * comb_cfg.max_num_inp_token_per_rank
        n_in            = comb_cfg.hidden_dim
        k_in            = self.inter_dim
        size_expert_ids = sorted_expert_ids.numel()

        # Plan B does NOT need an upfront zero_() on shmem_comb_inp_tok.
        #
        # Historical: this site once ran `shmem_comb_inp_tok[:mt*k].zero_()`
        # to clear stale data. That zero turned out to be the root cause of
        # a P0 race: the local zero (HBM write) and the remote peer's
        # fused_gemm2 P2P writes (same HBM region) are not cross-device
        # ordered. When the local zero arrived after the peer's first GEMM2
        # wave, whole blocks got clobbered (~5-10% intermittent fail).
        #
        # Plan B's invariant: every fused_gemm2 epilogue fully overwrites
        # [0, total_recv*k) of shmem_comb_inp_tok for tok_id < total_recv,
        # and combine_no_stage1 only reads that range. Standard MoE routing
        # always sends k experts per token (no dropout), so all (tok_id, j)
        # are written by some R-rank — no stale-data risk.
        # The k <= npes ctor assert guarantees mt*k <= mr.
        #
        # Note: token-level dropout (some j not sent) would require a
        # (tok_id, j) validity mask in combine Stage 3, NOT a fallback zero_().
        #
        # Debug knob: FLYDSL_FUSED_ZERO_MODE=normal force-enables the zero
        # (only useful to reproduce the historical race).
        if os.environ.get("FLYDSL_FUSED_ZERO_MODE", "").strip().lower() == "normal":
            _mt_k_bytes = (comb_cfg.max_num_inp_token_per_rank
                           * comb_cfg.num_experts_per_token
                           * comb_cfg.token_bytes)
            _mt_k_i16   = (_mt_k_bytes + 1) // 2
            comb_op.shmem_comb_inp_tok[:_mt_k_i16].zero_()

        # Default weight source is shmem_disp_out_wts (f32[max_recv*k], indexed
        # as wts[t*k + s]). A caller-supplied wts_buf must keep the same layout.
        addr_wts_buf = (
            fx.Int64(wts_buf.data_ptr())
            if wts_buf is not None
            else comb_op._fx_disp_out_wts
        )

        bias_t = bias if bias is not None else self._dummy_bias

        # First call uses fx-wrapped typed scalars so flyc can infer the
        # launcher signature; cached self._compiled calls below pass raw ints.
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

        # Weight handling matches baseline: combine_no_stage1 still runs
        # Stage 1 weight-only P2P scatter (preserved via
        # skip_stage1_keep_wts=True when skip_stage1=True) + Stage 3b accum.
        # End-to-end (out_tok, out_wts) is identical to baseline so verify
        # can diff both outputs directly. fused only removes the token P2P
        # traffic; weight traffic (~k*4B/token) is negligible vs token
        # traffic (~k*hidden_bytes).
        return comb_op.combine_no_stage1(
            self._dummy_inp, None, None, enable_weights=True,
        )

    def _run_full(self, a2, w2, a2_scale, w2_scale,
                  sti, eids, sw, num_valid_ids, bias):
        """Single-kernel end-to-end GEMM2 + Stage1/2/3."""
        raise NotImplementedError("_run_full: pending kernel implementation (PR2)")
