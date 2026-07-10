# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""End-to-end intra-node EP-MoE single operator ``MegaMoE`` (a full MoE layer).

Consolidates the previously scattered host-side logic into one place:

  * stage-1 (dispatch + GEMM1, ATOM contract)  -> ``MegaMoeStage1``
  * stage-2 (GEMM2 + EP combine)               -> ``MegaMoeStage2`` +
    ``compile_fused_moe_gemm2_combine`` (thin builder, inlined here)
  * megastage1 / megagemm2 tune-table helpers

The expert-major dispatch symmetric buffers are owned by the communication op
``FlyDSLDispatchCombineIntraNodeOp`` (when ``enable_group_major=True``) via its
internal ``FlyDSLDispatchGroupMajorOp`` (``comb_op._gm``). ``total_recv`` is a
single buffer shared with combine (the fused dispatch prologue writes it and
combine reads it -- no cross-op bridge).

Main entry ``MegaMoE.forward(x_bf16, wts, topk_ids)``: internal quantize ->
stage1 -> stage2 -> bf16 output. ``enable_fused_stage1`` / ``enable_fused_stage2``
select fused/non-fused paths (non-fused is a stub for now).
"""
from __future__ import annotations

import functools
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms

from .mega_moe_gemm1 import compile_fused_moe_gemm1
from .mixed_moe_gemm_2stage import compile_mixed_moe_gemm1, compile_mixed_moe_gemm2
from kernels.comm.flydsl_dispatch_combine_intranode_op import (
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)

try:
    import aiter
    from aiter import dtypes as _adt
except Exception:  # noqa: BLE001
    aiter = None
    _adt = None


def _mx_quant():
    """Lazy handle to aiter's MXFP8/FP4 activation quant (a8w4/a4w4 only). Imported at the call
    site, NOT at module load: some aiter branches (moe_v2_api / customer LQQ) ship the smoothquant
    path but not ``per_1x32_mx_quant_hip``, and the a8w4smooth/w8a8smooth delivery path never needs
    it -- so importing this module must not depend on it being present."""
    from aiter.ops.quant import per_1x32_mx_quant_hip
    return per_1x32_mx_quant_hip


__all__ = ["MegaMoE", "MegaMoeStage1", "MegaMoeStage2", "Stage1Output",
           "compile_fused_moe_gemm2_combine"]


def _is_fp4(dt):
    return dt == torch.float4_e2m1fn_x2


@dataclass(frozen=True)
class Stage1Output:
    """Stage-1 -> stage-2 handoff contract.

    Field names match ``MegaMoeStage2.run`` parameters. All tensors are stable
    buffers allocated once in ``MegaMoeStage1.__init__``; each ``forward``
    overwrites their contents in place (the objects themselves never change).
    """
    a2: torch.Tensor                 # GEMM1 output (raw buffer; stage-2 views it)
    a2_scale: torch.Tensor           # e8m0 sorted scale
    sorted_token_ids: torch.Tensor
    sorted_expert_ids: torch.Tensor
    sorted_weights: torch.Tensor     # per-sorted-row routing weights [nvm]
    num_valid_ids: torch.Tensor
    wts_buf: torch.Tensor            # combine weight buffer f32[max_recv*k]


# ======================================================================================
# Tune-table helpers (megastage1 tiles + megagemm2 per-M tiles from the FlyDSL JSON)
# ======================================================================================
def _mega_default_tile(inter_dim):
    """Generic validated fallback tile used when a shape is intentionally untuned."""
    return 64, (256 if (int(inter_dim) % 256 == 0) else 128)


_MEGA_TUNING_DIR = Path(__file__).resolve().parent.parent / "comm" / "mega_moe_tuning_config"
# megastage1 weights are always w4(fp4)/int8; the JSON ``dtype`` is the ACTIVATION quant.
# int8 covers a8w4smooth (packed-int4 W) and w8a8smooth (full int8 W) -- both run int8 activations
# through the same int8 K64 GEMM, so they share one "int8" tune-table dtype.
_MEGA_QUANT_TO_DTYPE = {"a4w4": "fp4", "a8w4": "fp8_ocp", "a8w4smooth": "int8", "w8a8smooth": "int8"}


def _convert_aiter_lqq_to_megamoe(u4, scale_u8, zero_u8):
    """Host-side (load-time) re-layout of an aiter LQQ ``lqq_1x64_quant`` RAW output into the
    megamoe a8w4smooth weight layout. The dequant MATH is identical on both sides --
    ``int8 = (u4*scale_u8 + zero_u8) ^ 0x80`` per 64-K group -- so ONLY the byte layout changes:

      * weight: base(16,16) int4 preshuffle -> K64 interleave -> 4-bit pack (mirrors
        ``shuffle_weight(use_int4=True, interleave_k64=True)`` + megamoe nibble pack).
      * qparams: aiter ``[E, rows, K//64]`` u8 -> megamoe ``[E, nb, g256, 16]`` int32, packing the
        four K64 groups of a 256-K tile into one dword (little-endian byte order).

    Args:
        u4:       ``[E, rows, K]`` uint8 (values 0..15) -- LQQ 4-bit weight, pre-shuffle.
        scale_u8: ``[E, rows, K//64]`` uint8 -- per-64-K scale.
        zero_u8:  ``[E, rows, K//64]`` uint8 -- per-64-K zero point.

    Returns:
        ``(w_packed_i8, qscale_i32, qzero_i32)`` ready to feed MegaMoE's a8w4smooth path as
        ``w1``/``w1_lqq_scale``/``w1_lqq_zero`` (or the w2 trio).
    """
    if u4.dim() != 3:
        raise ValueError(f"aiter LQQ u4 must be [E, rows, K], got {tuple(u4.shape)}")
    E, M, K = u4.shape
    if K % 256 != 0 or M % 16 != 0:
        raise ValueError(f"aiter LQQ convert needs K%256==0 and rows%16==0, got rows={M}, K={K}")
    nb, g256 = M // 16, K // 256
    dev = u4.device
    # ---- weight: base(16,16) int4 preshuffle (inner K=32, BK=32) + K64 interleave ----
    # Faithful to tests.utils.shuffle_weight(layout=(16,16), use_int4=True, interleave_k64=True).
    x = (u4.view(torch.uint8) & 0xF).to(torch.uint8)
    BN, Kinner, BKb = 16, 32, 32
    xs = x.view(E, M // BN, BN, K // BKb, BKb // Kinner, Kinner).permute(0, 1, 3, 4, 2, 5).contiguous().view(E, M, K)
    x128 = xs.view(E, M, K // 128, 128)
    y128 = torch.empty_like(x128)
    y128[..., 0::2] = x128[..., :64]
    y128[..., 1::2] = x128[..., 64:]
    xs = y128.view(E, M, K)
    # nibble pack: [v0..v7] -> b0=(v1<<4)|v0, ...
    u = (xs & 0xF).reshape(-1, 8)
    wp = torch.empty((u.shape[0], 4), device=dev, dtype=torch.uint8)
    wp[:, 0] = u[:, 0] | (u[:, 1] << 4)
    wp[:, 1] = u[:, 2] | (u[:, 3] << 4)
    wp[:, 2] = u[:, 4] | (u[:, 5] << 4)
    wp[:, 3] = u[:, 6] | (u[:, 7] << 4)
    w_packed = wp.view(-1).view(torch.int8).contiguous()

    # ---- qparams: [E, M, K//64] u8 -> [E, nb, g256, 16] int32 (4 K64 groups packed LE) ----
    def _to_i32(q):
        q5 = q.view(E, nb, 16, g256, 4).permute(0, 1, 3, 2, 4).contiguous().to(torch.int32)
        return (q5[..., 0] | (q5[..., 1] << 8) | (q5[..., 2] << 16) | (q5[..., 3] << 24)).contiguous()

    return w_packed, _to_i32(scale_u8), _to_i32(zero_u8)


def _as_i32_contig(t):
    """Contiguous int32 view of a packed4 LQQ qparam tensor. The megamoe qparams are already
    int32-sized (four K64 bytes packed per dword); reinterpret to int32 only if a differently
    typed (e.g. uint8) tensor was passed."""
    t = t.contiguous()
    return t if t.dtype == torch.int32 else t.view(torch.int32)


def _detect_gpu_model_name(device_index=0):
    """GPU model substring (e.g. ``"mi355x"``) for tuning-file selection."""
    try:
        name = torch.cuda.get_device_properties(device_index).name.lower()
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"\bmi\d+\w*", name)
    return m.group(0) if m else None


@functools.lru_cache(maxsize=8)
def _load_mega_tuning_rows(ep_size: int, gpu_model):
    """megastage1 rows from the best-matching ``flydsl_*_MegaStage1_ep{n}.json``
    (preferring a ``gpu_model`` name match). Returns a hashable tuple of dicts,
    ``()`` on any miss. Call ``.cache_clear()`` after editing the JSON."""
    if not _MEGA_TUNING_DIR.is_dir():
        return ()
    suffix = f"_MegaStage1_ep{ep_size}.json"
    cands = [p for p in _MEGA_TUNING_DIR.glob(f"flydsl_*{suffix}") if p.is_file()]
    if not cands:
        return ()
    cands.sort(key=lambda p: (1 if (gpu_model and gpu_model in p.name) else 0, p.name),
               reverse=True)
    try:
        with open(cands[0], "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return ()
    return tuple(raw.get("megastage1", []))


def _mega_tuned_tile(model_dim, inter_dim, experts, topk, quant, mtpr, ep_size, gpu_model):
    """megastage1 config from the tune JSON, keyed by GEMM shape. The token
    bucket rounds up to the smallest ``num_tokens >= mtpr`` (largest on
    overflow). Returns the tile dict or ``None`` on a miss."""
    dtype = _MEGA_QUANT_TO_DTYPE.get(quant)
    if dtype is None:
        return None
    rows = _load_mega_tuning_rows(int(ep_size), gpu_model)
    if not rows:
        return None

    def _match(r):
        try:
            return (r.get("dtype") == dtype
                    and int(r["model_dim"]) == int(model_dim)
                    and int(r["inter_dim"]) == int(inter_dim)
                    and int(r["expert"]) == int(experts)
                    and int(r["topk"]) == int(topk))
        except (KeyError, ValueError, TypeError):
            return False

    buckets = {int(r["num_tokens"]): r for r in rows if _match(r)}
    if not buckets:
        return None
    if int(mtpr) in buckets:
        chosen = buckets[int(mtpr)]
    else:
        ge = [k for k in buckets if k >= int(mtpr)]
        chosen = buckets[min(ge)] if ge else buckets[max(buckets)]
    return dict(
        tile_m=int(chosen["tile_m"]),
        tile_n=int(chosen["tile_n"]),
        tile_k=int(chosen["tile_k"]),
        waves_per_eu=int(chosen.get("waves_per_eu", 4)),
        use_async_copy=bool(chosen.get("use_async_copy", True)),
    )


@functools.lru_cache(maxsize=8)
def _load_mega_gemm2_rows(ep_size: int, gpu_model):
    """megagemm2 rows from the best-matching ``flydsl_*_MegaGemm2_ep{n}.json``
    (symmetric with ``_load_mega_tuning_rows``). ``()`` on any miss."""
    if not _MEGA_TUNING_DIR.is_dir():
        return ()
    suffix = f"_MegaGemm2_ep{ep_size}.json"
    cands = [p for p in _MEGA_TUNING_DIR.glob(f"flydsl_*{suffix}") if p.is_file()]
    if not cands:
        return ()
    cands.sort(key=lambda p: (1 if (gpu_model and gpu_model in p.name) else 0, p.name),
               reverse=True)
    try:
        with open(cands[0], "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return ()
    return tuple(raw.get("megagemm2", []))


def _mega_gemm2_tuned_table(model_dim, inter_dim, experts, topk, ep_size, gpu_model,
                            quant=None):
    """Per-M fused gemm2 config table ``{num_tokens: {tile_m, tile_n, tile_k,
    [xcd_swizzle], [persist_m], [b_nt]}}`` from the tune JSON, keyed by GEMM2 shape.
    ``experts`` is the LOCAL per-rank expert count. Returns the dict, or ``None`` on a
    miss (op falls back to its ctor defaults).

    Tile is required per row; ``xcd_swizzle`` / ``persist_m`` / ``b_nt`` are OPTIONAL
    (like aiter's kernel-name-encoded knobs) and only override the op ctor default when
    present. RAW tiles -- the op clamps tile_m to a divisor of sort_block_m at run time.
    ``quant`` ("a8w4"/"a4w4") filters rows by the row's ``dtype`` (the activation quant;
    weights are always fp4); passing None keeps the legacy dtype-agnostic match."""
    rows = _load_mega_gemm2_rows(int(ep_size), gpu_model)
    if not rows:
        return None
    want_dtype = _MEGA_QUANT_TO_DTYPE.get(quant) if quant is not None else None

    def _match(r):
        try:
            if want_dtype is not None and r.get("dtype") != want_dtype:
                return False
            return (int(r["model_dim"]) == int(model_dim)
                    and int(r["inter_dim"]) == int(inter_dim)
                    and int(r["expert"]) == int(experts)
                    and int(r["topk"]) == int(topk))
        except (KeyError, ValueError, TypeError):
            return False

    table = {}
    for r in rows:
        if _match(r):
            try:
                entry = {"tile_m": int(r["tile_m"]), "tile_n": int(r["tile_n"]),
                         "tile_k": int(r["tile_k"])}
            except (KeyError, ValueError):
                continue
            for opt in ("xcd_swizzle", "persist_m", "b_nt"):  # optional per-bucket knobs
                if r.get(opt) is not None:
                    try:
                        entry[opt] = int(r[opt])
                    except (ValueError, TypeError):
                        pass
            table[int(r["num_tokens"])] = entry
    return table or None


# ======================================================================================
# stage-2 fused GEMM2+combine builder (thin wrapper over compile_mixed_moe_gemm2)
# ======================================================================================
def compile_fused_moe_gemm2_combine(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    persist_m: int = 4,
    sort_block_m: int = 0,
    b_nt: int = 2,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    rank: int,
    npes: int,
    max_tok_per_rank: int,
    experts_per_token: int = 1,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    xcd_swizzle: int = 0,
    use_token_flag_sync: bool = False,
    doweight_fused: bool = False,
):
    """Compile the fused GEMM2+combine kernel and return its host launcher.

    The GEMM2 epilogue's local store_pair is rewritten into a remote P2P
    buffer_store (combine Stage-1 P2P scatter inlined); the host then launches a
    trimmed combine (``combine_no_stage1``) for Stage 2/3.
    """
    # Plan B slot = dest_lid * k + s; reusing the baseline shmem_comb_inp_tok
    # (size mr) requires mt*k <= mr, i.e. topk <= npes.
    if topk > npes:
        raise ValueError(
            f"fused GEMM2+combine (Plan B) requires topk <= npes; "
            f"got topk={topk}, npes={npes}."
        )
    if topk != experts_per_token:
        raise ValueError(
            f"topk ({topk}) must equal experts_per_token ({experts_per_token}) "
            "in Plan B; sorted_token_ids' s field is used directly as j_global."
        )
    if model_dim % tile_n != 0:
        raise ValueError(f"model_dim={model_dim} must be divisible by tile_n={tile_n}")
    if inter_dim % tile_k != 0:
        raise ValueError(f"inter_dim={inter_dim} must be divisible by tile_k={tile_k}")
    if (max_tok_per_rank & (max_tok_per_rank - 1)) != 0:
        # Epilogue decodes dest_enc -> (dest_pe, dest_lid) via shift+mask.
        raise ValueError(f"max_tok_per_rank={max_tok_per_rank} must be a power of two.")

    out_s = str(out_dtype).strip().lower()
    fp8_cast = out_s in ("fp8", "fp8e4m3", "fp8e4m3fn", "f8e4m3fn", "float8_e4m3fn")
    if not fp8_cast and out_s not in ("bf16", "bfloat16", "f16", "fp16"):
        raise ValueError(f"fused mode output must be bf16/f16 or fp8e4m3fn, got {out_dtype!r}")

    # fused_cfg[3]=enable_weights is pinned False: in-kernel weight P2P scatter
    # (~16B writes) is unreliable under fabric saturation, so weights are handled
    # by the subsequent combine_no_stage1 Stage-1 instead. Under fp8_cast, GEMM2
    # stays bf16 internally and store_pair does the bf16->fp32->fp8 cast (1B/elem).
    fused_cfg = (
        int(npes),
        int(rank),
        int(max_tok_per_rank),
        False,
        int(experts_per_token),
        bool(fp8_cast),
    )
    gemm2_out_dtype = "bf16" if fp8_cast else out_dtype

    return compile_mixed_moe_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        persist_m=persist_m,
        sort_block_m=sort_block_m,
        b_nt=b_nt,
        a_dtype=a_dtype, b_dtype=b_dtype, out_dtype=gemm2_out_dtype,
        accumulate=False,
        doweight_stage2=bool(doweight_fused),
        enable_bias=False,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        use_cshuffle_epilog=True,
        xcd_swizzle=xcd_swizzle,
        fused_p2p_scatter=fused_cfg,
        use_token_flag_sync=use_token_flag_sync,
    )


# ======================================================================================
# stage-2 operator: fused MoE-GEMM2 + EP-combine
# ======================================================================================
class MegaMoeStage2:
    """Fused MoE-GEMM2 + EP-combine operator.

    The GEMM2 epilogue inlines combine Stage-1's P2P token scatter; the host then
    launches a trimmed combine (``combine_no_stage1``) for Stage 2/3. This is the
    only supported mode.
    """

    # Probed by test scripts via getattr(cls, 'READY', False) to graceful-skip.
    READY: bool = True

    def __init__(
        self,
        *,
        comb_cfg: FlyDSLDispatchCombineConfig,
        comb_op: FlyDSLDispatchCombineIntraNodeOp,
        inter_dim: int,
        tile_m: int = 32, tile_n: int = 128, tile_k: int = 128,
        persist_m: int = -1,   # <=0 -> persistent mode (grid_y=cu_num), the default; >0 = legacy N-tile
        sort_block_m: int = 0,
        b_nt: int = 2,
        a_dtype: str = "fp8",
        b_dtype: str = "fp4",
        xcd_swizzle: int = 0,
        use_token_flag_sync: bool = False,
        doweight_fused: bool = True,
        gemm2_tile_table: dict | None = None,
        force_mode: Optional[str] = None,
        # a8w4smooth (a_dtype='int8'): packed4 W2 qscale/qzero (this rank's LOCAL epr experts).
        w2_lqq_scale: Optional[torch.Tensor] = None,
        w2_lqq_zero: Optional[torch.Tensor] = None,
    ):
        # force_mode is accepted for back-compat only (some callers pass
        # "stage1_only"); run() always uses the fused stage1-only path.
        if force_mode not in (None, "stage1_only"):
            raise ValueError(f"MegaMoeStage2: unsupported force_mode={force_mode!r} "
                             "(only 'stage1_only' / None supported)")
        self.comb_cfg    = comb_cfg
        self.comb_op     = comb_op
        # MegaMoE is the audited consumer of the fused GEMM2+combine path, so enable the
        # combine_no_stage1 gate main ships default-off (it reduces the P2P-scattered tokens
        # our fused GEMM2 epilogue writes; Stage-1 scatter is done in-kernel, not by combine).
        FlyDSLDispatchCombineIntraNodeOp._ENABLE_COMBINE_NO_STAGE1 = True
        self.inter_dim   = inter_dim
        self.tile_m      = tile_m
        self.tile_n      = tile_n
        self.tile_k      = tile_k
        self.persist_m   = persist_m
        self.sort_block_m = sort_block_m
        self.b_nt        = b_nt
        self.a_dtype     = a_dtype
        self.b_dtype     = b_dtype
        self._is_int8    = (a_dtype == "int8")
        # w8a8smooth: FULL int8 W2 (b_dtype='int8'), per-output-row f32 scale, NO packed4 LQQ
        # qscale/qzero (gemm2 is_int8_full B-loader). a8w4smooth uses b_dtype='int4' + packed4 qparams.
        self._is_w8a8    = self._is_int8 and (b_dtype == "int8")
        # a8w4smooth: packed4 W2 qscale/qzero addresses for the fused gemm2's in-kernel int4->int8
        # dequant. int32-viewed contiguous (kernel reads via create_buffer_resource_from_addr).
        self._qs_w2 = None
        self._qz_w2 = None
        if self._is_int8 and not self._is_w8a8:
            assert w2_lqq_scale is not None and w2_lqq_zero is not None, (
                "MegaMoeStage2(a8w4smooth) requires w2_lqq_scale/w2_lqq_zero (packed4 qparams)")
            self._qs_w2 = _as_i32_contig(w2_lqq_scale)
            self._qz_w2 = _as_i32_contig(w2_lqq_zero)
        self.xcd_swizzle = xcd_swizzle
        self.use_token_flag_sync = bool(use_token_flag_sync)
        # Apply routing weights in the GEMM2 epilogue; combine_no_stage1 reduces
        # tokens unweighted, so this is on by default for correctness.
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

        # Per-config JIT caches: the gemm2 kernel bakes (tile_m,tile_n,tile_k,persist_m,
        # b_nt,xcd_swizzle) as compile-time constants, so we compile/cache one kernel per
        # distinct config and pick by run_tokens at run() (mirrors aiter's per-M selection).
        # The heavy comb_op shmem is shared -> no extra symmetric-heap cost.
        self._launch_by_tile = {}
        self._compiled_by_tile = {}
        # Optional {pow2_token: {tile_m,tile_n,tile_k,[xcd_swizzle],[persist_m],[b_nt]}}
        # table; None -> ctor defaults. Legacy 3-tuple values are normalised to dicts.
        if gemm2_tile_table:
            self._tile_table = {}
            for _tok, _v in dict(gemm2_tile_table).items():
                if isinstance(_v, dict):
                    self._tile_table[int(_tok)] = dict(_v)
                else:  # legacy (tile_m, tile_n, tile_k)
                    _tm, _tn, _tk = _v
                    self._tile_table[int(_tok)] = {
                        "tile_m": int(_tm), "tile_n": int(_tn), "tile_k": int(_tk)}
        else:
            self._tile_table = None
        # With a per-M table active, stage1 lays sorted_expert_ids at a FIXED block
        # (MegaMoE: 32). gemm2 must read at that block (tile_m only sub-tiles within
        # it), so pin sort_block_m=32 (default sort_block_m<=0 assumes == tile_m,
        # which mismatches stage1 once tile_m != 32). Table tile_m is pre-clamped.
        if self._tile_table is not None and int(sort_block_m) <= 0:
            self.sort_block_m = 32
        # Back-compat single-tile handles (unused when a table is active).
        self._launch_fn = None
        self._compiled = None
        self._alloc_dummy_tensors()

    @staticmethod
    def _next_pow2(n: int) -> int:
        n = int(n)
        if n <= 1:
            return 1
        return 1 << (n - 1).bit_length()

    def _clamp_tile_m(self, tm):
        """Safety net: gemm2 sub-tiles stage1's sorted_expert_ids blocks, so tile_m
        MUST divide sort_block_m or the kernel reads the wrong expert's W2 (silent
        garbage). Halve tile_m to the nearest power-of-two divisor (floored at 16)
        if a table hands us a bad value. No-op when sort_block_m<=0."""
        sbm = int(self.sort_block_m)
        if sbm <= 0:
            return int(tm)
        tm = int(tm)
        if tm > sbm:
            tm = sbm
        while tm > 16 and (sbm % tm) != 0:
            tm //= 2
        return tm

    def _select_tile(self, run_tokens):
        """Pick the full gemm2 config ``(tile_m, tile_n, tile_k, persist_m, b_nt,
        xcd_swizzle)`` for this forward's run_tokens via the host-static table
        (nextPow2 bucket, clamped to the table range), mirroring aiter's get_2stage_cfgs.
        Per-bucket ``persist_m``/``b_nt``/``xcd_swizzle`` override the ctor default only
        when present in the table row; tile falls back to the ctor tile with no table.
        The chosen tile_m is always clamped to a divisor of sort_block_m."""
        _tk_force = 256 if self._is_int8 else None   # int8 gemm2 is K64-packed -> tile_k MUST be 256
        if not self._tile_table or run_tokens is None:
            return (self._clamp_tile_m(self.tile_m), self.tile_n, (_tk_force or self.tile_k),
                    self.persist_m, self.b_nt, self.xcd_swizzle)
        keys = sorted(self._tile_table)
        b = self._next_pow2(run_tokens)
        if b < keys[0]:
            b = keys[0]
        elif b > keys[-1]:
            b = keys[-1]
        if b not in self._tile_table:
            ge = [k for k in keys if k >= b]
            b = ge[0] if ge else keys[-1]
        row = self._tile_table[b]
        return (
            self._clamp_tile_m(row["tile_m"]), int(row["tile_n"]), (_tk_force or int(row["tile_k"])),
            int(row.get("persist_m", self.persist_m)),
            int(row.get("b_nt", self.b_nt)),
            int(row.get("xcd_swizzle", self.xcd_swizzle)),
        )

    def _alloc_dummy_tensors(self):
        """Pre-allocate placeholder tensors at ctor time (torch.zeros/empty inside
        a cudagraph capture would trigger hipMalloc errors)."""
        comb_cfg = self.comb_cfg
        dev = torch.device("cuda", comb_cfg.rank)
        max_recv = comb_cfg.world_size * comb_cfg.max_num_inp_token_per_rank

        # arg_out is unused on the fused path but the launcher signature needs it.
        self._dummy_out = torch.zeros(1, dtype=comb_cfg.data_type, device=dev)
        self._dummy_bias = torch.empty(0, dtype=comb_cfg.data_type, device=dev)

        # Allocate the combine_no_stage1 input directly in fp8 under fp8_cast to
        # skip combine's internal bf16->fp8 copy.
        inp_dtype = torch.float8_e4m3fn if self._fp8_cast else comb_cfg.data_type
        self._dummy_inp = torch.zeros(max_recv, comb_cfg.hidden_dim, dtype=inp_dtype, device=dev)

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
        run_tokens: Optional[int] = None,
    ):
        """Run the fused GEMM2+combine.

        ``wts_buf`` is the f32[max_recv*k] weight buffer dispatch writes for
        combine (falls back to ``comb_op.shmem_disp_out_wts`` if None). ``cur_tok``
        is this rank's local token count (combine Stage-3 output rows); a fused
        stage-1 that writes the routing tables directly (without calling dispatch)
        must pass it explicitly. Returns ``(out_tok, out_wts)``.
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
            run_tokens=run_tokens,
        )

    def _ensure_launch_fn(self, cfg=None):
        """Build (once) and cache the launch_fn for ``cfg`` (defaults to the ctor
        config), keyed by distinct (tile_m, tile_n, tile_k, persist_m, b_nt, xcd_swizzle)
        -- all baked as compile-time constants."""
        if cfg is None:
            cfg = (self.tile_m, self.tile_n, self.tile_k,
                   self.persist_m, self.b_nt, self.xcd_swizzle)
        cfg = tuple(int(v) for v in cfg)
        tile_m, tile_n, tile_k, persist_m, b_nt, xcd_swizzle = cfg
        # Divisibility contract: tile_m must divide the real sort block stage1
        # emitted sorted_expert_ids at, else the kernel silently produces garbage.
        # sort_block_m<=0 => kernel assumes == tile_m (always OK).
        if int(self.sort_block_m) > 0 and (int(self.sort_block_m) % tile_m) != 0:
            raise ValueError(
                f"MegaMoeStage2: gemm2 tile_m={tile_m} must divide "
                f"sort_block_m={self.sort_block_m} (stage1 sorted_expert_ids block). "
                f"Pick tile_m in divisors of {self.sort_block_m}."
            )
        lf = self._launch_by_tile.get(cfg)
        if lf is not None:
            return lf
        comb_cfg = self.comb_cfg
        # Plan B writes into shmem_comb_inp_tok's [0, mt*k) sub-range; k <= npes
        # keeps it in-bounds (mt*k <= mr).
        k = comb_cfg.num_experts_per_token
        if k > comb_cfg.world_size:
            raise ValueError(
                f"MegaMoeStage2 (Plan B) requires k <= npes; "
                f"got k={k}, npes={comb_cfg.world_size}."
            )
        lf = compile_fused_moe_gemm2_combine(
            model_dim=comb_cfg.hidden_dim,
            inter_dim=self.inter_dim,
            experts=comb_cfg.num_experts_per_rank,
            topk=k,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            persist_m=persist_m,
            sort_block_m=self.sort_block_m,
            b_nt=b_nt,
            a_dtype=self.a_dtype, b_dtype=self.b_dtype,
            out_dtype=self._out_dtype_str,
            rank=comb_cfg.rank, npes=comb_cfg.world_size,
            max_tok_per_rank=comb_cfg.max_num_inp_token_per_rank,
            experts_per_token=k,
            xcd_swizzle=xcd_swizzle,
            use_token_flag_sync=self.use_token_flag_sync,
            doweight_fused=self.doweight_fused,
        )
        self._launch_by_tile[cfg] = lf
        return lf

    def _run_stage1_only(self, *, a2, w2, a2_scale, w2_scale,
                         sorted_token_ids, sorted_expert_ids, sorted_weights,
                         num_valid_ids, wts_buf=None, cur_tok=None, bias=None, stream=None,
                         run_tokens=None):
        """fused GEMM2 (epilogue P2P scatter) + combine_no_stage1 (Stage 2/3)."""
        # Per-M config selection keys on the un-expanded forward token count (NOT the
        # topk-expanded gemm rows); host-static so the kernel is baked into each graph.
        cfg = self._select_tile(run_tokens)
        launch_fn = self._ensure_launch_fn(cfg)

        comb_cfg = self.comb_cfg
        comb_op  = self.comb_op
        if stream is None:
            stream = torch.cuda.current_stream()
        s_fx = fx.Stream(stream.cuda_stream)

        # tokens_in MUST be max_recv (not a2.shape[0]); it drives the epilogue
        # row_valid early-exit, else padding rows OOB-read addr_tis and the P2P
        # scatter corrupts combine state.
        tokens_in       = comb_cfg.world_size * comb_cfg.max_num_inp_token_per_rank
        n_in            = comb_cfg.hidden_dim
        k_in            = self.inter_dim
        size_expert_ids = sorted_expert_ids.numel()

        # No upfront zero_() on shmem_comb_inp_tok: a local zero races (cross-device
        # unordered) against peers' fused_gemm2 P2P writes to the same HBM. Safe
        # because every (tok_id, j) in [0, total_recv*k) is fully overwritten by
        # some rank (standard MoE routing sends all k experts). Debug knob
        # FLYDSL_FUSED_ZERO_MODE=normal force-enables the zero.
        if os.environ.get("FLYDSL_FUSED_ZERO_MODE", "").strip().lower() == "normal":
            _mt_k_bytes = (comb_cfg.max_num_inp_token_per_rank
                           * comb_cfg.num_experts_per_token
                           * comb_cfg.token_bytes)
            _mt_k_i16   = (_mt_k_bytes + 1) // 2
            comb_op.shmem_comb_inp_tok[:_mt_k_i16].zero_()

        # Default weight source is shmem_disp_out_wts (f32[max_recv*k], indexed as
        # wts[t*k + s]); a caller-supplied wts_buf must keep the same layout.
        addr_wts_buf = (
            fx.Int64(wts_buf.data_ptr())
            if wts_buf is not None
            else comb_op._fx_disp_out_wts
        )

        bias_t = bias if bias is not None else self._dummy_bias

        # First call passes fx-typed scalars so flyc infers the signature; cached
        # calls reuse the compiled kernel with raw ints.
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
        # a8w4smooth (int8): the fused gemm2 launcher takes trailing packed4 W2 qscale/qzero addresses
        # (after the i32 shape scalars, before the stream) for the in-kernel int4->int8 B-dequant.
        # w8a8smooth (FULL int8 W2): no LQQ qparams -> pass 0/0 (gemm2 is_int8_full skips them).
        _qp = ()
        if self._is_int8:
            _qs2 = fx.Int64(self._qs_w2.data_ptr()) if self._qs_w2 is not None else fx.Int64(0)
            _qz2 = fx.Int64(self._qz_w2.data_ptr()) if self._qz_w2 is not None else fx.Int64(0)
            _qp = (_qs2, _qz2)
        compiled = self._compiled_by_tile.get(cfg)
        if compiled is None:
            compiled = flyc.compile(
                launch_fn,
                *common_args,
                fx.Int32(tokens_in), fx.Int32(n_in), fx.Int32(k_in),
                fx.Int32(size_expert_ids),
                *_qp,
                s_fx,
            )
            self._compiled_by_tile[cfg] = compiled
        else:
            compiled(
                *common_args,
                tokens_in, n_in, k_in, size_expert_ids,
                *_qp,
                s_fx,
            )

        # Weights still go through combine_no_stage1's Stage-1 weight-only P2P
        # scatter + Stage-3b accum (negligible ~k*4B/token traffic), so the output
        # matches baseline bit-for-bit. Fused only removes the heavy token P2P.
        return comb_op.combine_no_stage1(
            self._dummy_inp, None, None, cur_tok=cur_tok, enable_weights=True,
        )


# ======================================================================================
# stage-1 config resolution + driver: single-launch fused dispatch + GEMM1
# ======================================================================================
def _resolve_stage1_config(*, model_dim, inter_dim, experts, topk, quant,
                           max_tok_per_rank, world_size, rank,
                           tune_tokens=None, unit_size=-1, tile_n=-1, tile_k=256,
                           waves_per_eu=4, use_async_copy=True):
    """Resolve the fused stage-1 GEMM tile + layout (compact) BEFORE the combine op
    allocates its group-major buffers (which need unit_size/compact). Returns a
    dict of resolved values consumed by both the combine op (gm_*) and stage-1."""
    assert quant in ("a4w4", "a8w4", "a8w4smooth", "w8a8smooth")
    if experts % world_size != 0:
        raise ValueError(f"experts={experts} must divide world_size={world_size}")
    epr = experts // world_size
    if quant in ("a8w4smooth", "w8a8smooth"):
        # int8 activations + packed-int4 (a8w4smooth) or FULL int8 (w8a8smooth) weights. The
        # group-major dispatch payload is int8 (row_bytes = model_dim, like fp8) and the activation
        # scale is ONE f32 per (token,slot) (scale_dim=1, type_size=4 -> the comb_op sizes scale_em
        # as int32-per-row). tile_k is pinned to 256 (K64 packs / kpack=16), matching the standalone
        # a8w4 gemm1. The activation datapath is IDENTICAL; only the weight kind differs downstream.
        data_type = torch.int8
        a_dtype = "int8"
        scale_dim = 1
        tile_k = 256
    else:
        data_type = torch.float8_e4m3fn if quant == "a8w4" else torch.float4_e2m1fn_x2
        a_dtype = "fp8" if quant == "a8w4" else "fp4"
        scale_dim = model_dim // 32

    # Two layouts, switched purely by buffer size (no env, not customer-visible): fixedslot below
    # the 3GB wrap, compact above it. a8w4smooth/w8a8smooth use the SAME size-based rule as the fp
    # a8w4/a4w4 path (unified) -- so DECODE (small max_tok_per_rank -> buffer <3GB) auto-picks the
    # fixedslot dispatch (low fixed overhead, ~20% faster at small batch) and PREFILL/large-batch
    # (>= ~2181 tok/rank -> buffer >=3GB) auto-picks compact. fixedslot was verified bit-for-bit
    # correctness-equivalent to compact at EP1 AND EP8 (identical relL2, deterministic across
    # launches; the old P-static ll_count race no longer manifests).
    _row_b = int(model_dim) if quant in ("a8w4", "a8w4smooth", "w8a8smooth") else (int(model_dim) // 2)
    _max_buf = epr * world_size * int(max_tok_per_rank) * max(_row_b, int(inter_dim) * 2)
    compact = _max_buf >= 3_000_000_000

    tune_tokens = int(max_tok_per_rank if tune_tokens is None else tune_tokens)
    _tuned = None
    if int(unit_size) <= 0:
        _tuned = _mega_tuned_tile(
            model_dim, inter_dim, experts, topk, quant, tune_tokens,
            world_size, _detect_gpu_model_name(rank))
    if _tuned is not None:
        unit_size = _tuned["tile_m"]
        if int(tile_n) <= 0:
            tile_n = _tuned["tile_n"]
        tile_k = _tuned["tile_k"]
        waves_per_eu = _tuned["waves_per_eu"]
        use_async_copy = _tuned["use_async_copy"]
    else:
        _tm_t, _tn_t = _mega_default_tile(int(inter_dim))
        unit_size = int(unit_size) if int(unit_size) > 0 else _tm_t
        tile_n = int(tile_n) if int(tile_n) > 0 else _tn_t
    if quant in ("a8w4smooth", "w8a8smooth"):
        tile_k = 256   # int8 gemm1 is K64-packed (kpack=16) -> tile_k MUST be 256, whatever the JSON says
    unit_size = int(unit_size)
    sort_block_m = max(32, unit_size)

    # XCD swizzle correctness guard: disable when the WGM group would straddle XCD
    # boundaries (gx not a divisor of 8, or cu % gx != 0) -> would corrupt/lose
    # output tiles, and gives no XCD-alignment benefit anyway.
    xcd = 4
    if xcd > 0:
        try:
            from aiter.jit.utils.chip_info import get_cu_num
            _cu_g = get_cu_num()
        except Exception:  # noqa: BLE001
            _cu_g = 256
        _gx_g = max(1, (inter_dim * 2 + 2 * int(tile_n) - 1) // int(tile_n) // 2)
        if _cu_g % _gx_g != 0 or (8 % _gx_g) != 0:
            import warnings
            warnings.warn(
                f"XCD swizzle DISABLED: gx={_gx_g} is not a divisor of 8 (or cu="
                f"{_cu_g} % gx != 0).")
            xcd = 0

    return dict(
        epr=epr, data_type=data_type, a_dtype=a_dtype, scale_dim=scale_dim,
        compact=compact, unit_size=unit_size, tile_n=int(tile_n), tile_k=int(tile_k),
        waves_per_eu=int(waves_per_eu), use_async_copy=bool(use_async_copy),
        sort_block_m=sort_block_m, xcd=xcd, tune_tokens=tune_tokens,
    )


class MegaMoeStage1:
    """Single-launch fused dispatch + GEMM1 megakernel (fixedslot decode).

    The expert-major dispatch buffers live in the combine op (``comm_op._gm``, a
    ``FlyDSLDispatchGroupMajorOp``); this driver compiles the fused kernel, builds
    the dispatch-arg pointer table over those buffers, and runs forward.
    """

    def __init__(self, *, comm_op, cfg, rank, world_size, model_dim, inter_dim,
                 experts, topk, quant, w1, w1_scale, max_tok_per_rank,
                 w1_lqq_scale=None, w1_lqq_zero=None, fc1_smooth=None,
                 scheme="fixedslot", warp_num_per_block=4, out_dtype="auto"):
        assert quant in ("a4w4", "a8w4", "a8w4smooth", "w8a8smooth")
        # a8w4smooth (LQQ int8 x packed-int4 + smoothquant): int8 activations, packed-int4 weights,
        # per-(token,slot) f32 activation scale (vs mxscale's fp8/fp4 + e8m0). The send-side front
        # smoothquant emits per-slot int8; the dispatch prologue scatters it PER-SLOT (see the
        # is_int8 branch in mega_moe_gemm1).
        # w8a8smooth: int8 activations x FULL int8 W1 (per-output-row f32 scale, NO packed-int4 / NO
        # LQQ qscale-qzero). Shares a8w4smooth's int8-activation dispatch/smoothquant; only the weight
        # datapath differs (b_dtype='int8' -> gemm1's is_int8_full B-loader).
        self._is_int8 = quant in ("a8w4smooth", "w8a8smooth")
        self._is_w8a8 = (quant == "w8a8smooth")
        assert scheme == "fixedslot", f"scheme={scheme!r}: only 'fixedslot' supported (handshake removed)"
        assert out_dtype in ("auto", "f16", "fp8", "fp4"), out_dtype
        self.scheme = scheme
        # The megakernel ALWAYS emits a2 in the ATOM contract.
        self.atom_contract = True
        self.rank = rank
        self.world_size = world_size
        self.model_dim = model_dim
        self.inter_dim = inter_dim
        self.experts = experts
        self.epr = cfg["epr"]
        self.topk = topk
        self.quant = quant
        self.dev = torch.device("cuda", rank)
        self.data_type = cfg["data_type"]
        self.a_dtype = cfg["a_dtype"]
        if self._is_int8:
            self.a_dtype = "int8"        # activations are int8 (front smoothquant output)
        if out_dtype == "auto":
            # int8's a_dtype isn't a valid OUTPUT dtype -> stage-1 a2 is f16 (bf16 handoff); the
            # fused-stage2 int8 requant (T2.7) reads it back. mxscale keeps out==a_dtype.
            out_dtype = "f16" if self._is_int8 else self.a_dtype
        self.scale_dim = cfg["scale_dim"]
        self.compact = cfg["compact"]
        self.unit_size = int(cfg["unit_size"])
        self.sort_block_m = cfg["sort_block_m"]
        self.tile_n = int(cfg["tile_n"])
        tile_k = int(cfg["tile_k"])
        waves_per_eu = cfg["waves_per_eu"]
        use_async_copy = cfg["use_async_copy"]
        self._xcd = cfg["xcd"]
        self.mtpr = int(max_tok_per_rank)
        self.tune_tokens = int(cfg["tune_tokens"])

        # w1/w1_scale are this rank's `epr` expert rows ONLY (ATOM local convention).
        self.w1 = w1.contiguous()
        self.w1_scale = w1_scale.contiguous()
        # a8w4smooth: packed4 qscale/qzero (this rank's LOCAL epr experts) for the in-kernel
        # int4->int8 B-dequant, plus the GLOBAL per-expert smooth (the SEND-side front quant sees
        # every expert a local token may route to, so smooth must cover ALL experts, not just epr).
        if self._is_int8:
            assert fc1_smooth is not None, (
                "a8w4smooth/w8a8smooth fused stage-1 requires fc1_smooth (GLOBAL [experts, model_dim])")
            if self._is_w8a8:
                # FULL int8 W1: no LQQ packed4 qscale/qzero (gemm1 is_int8_full B-loader).
                self._qs_w = None
                self._qz_w = None
            else:
                assert w1_lqq_scale is not None and w1_lqq_zero is not None, (
                    "a8w4smooth fused stage-1 requires w1_lqq_scale/w1_lqq_zero (packed4 qparams)")
                self._qs_w = _as_i32_contig(w1_lqq_scale)
                self._qz_w = _as_i32_contig(w1_lqq_zero)
            self._fc1_smooth = fc1_smooth.to(torch.float32).contiguous()
            # Front smoothquant scratch (token-major per-slot): row = src_tok*topk + slot == _wk,
            # exactly the index the dispatch prologue reads for the int8 payload + f32 scale.
            # Pre-allocated once (fixed mtpr) so forward is allocation-free on this path.
            self._fq_i8 = torch.zeros((self.mtpr * topk, model_dim), dtype=torch.int8, device=self.dev)
            self._fq_scale = torch.zeros((self.mtpr * topk,), dtype=torch.float32, device=self.dev)
            # Cache the aiter front-smoothquant fn once (no per-forward import in the hot loop).
            from aiter.ops.quant import smooth_per_token_scaled_quant as _sptsq
            self._sptsq = _sptsq

        # group-major dispatch buffers owned by the combine op.
        self.comm_op = comm_op
        self.op = comm_op._gm
        assert self.op is not None, "combine op was built without enable_group_major"
        self.nvm = self.op.num_valid_max
        self.max_blocks = self.op.max_blocks
        self.cap = self.op.ll_cap
        # metadata-ready flag (monotonic, NEVER reset -> CUDAGraph-safe).
        self._meta = torch.zeros(1, dtype=torch.int32, device=self.dev)
        # total_recv is UNIFIED with combine (op.total_recv aliases comb_op.total_recv).
        self._trecv_buf = None

        # a8w4smooth (int8) datapath knobs: packed-int4 B, tile_k=256 (K64 packs, kpack=16), and a
        # per-(token,slot) f32 activation scale (fuse_scale_dim=1, type_size=4) vs mxscale's e8m0.
        _b_dtype = ("int8" if self._is_w8a8 else "int4") if self._is_int8 else "fp4"
        _tile_k_c = 256 if self._is_int8 else tile_k
        _fuse_sdim = 1 if self._is_int8 else self.scale_dim
        _fuse_stsz = 4 if self._is_int8 else 1
        # Compile the megakernel (serialize across ranks to bound peak memory).
        self.mega = None
        for pe in range(world_size):
            if rank == pe:
                self.mega = compile_fused_moe_gemm1(
                    model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=1,
                    tile_m=self.unit_size, tile_n=self.tile_n, tile_k=_tile_k_c, doweight_stage1=False,
                    a_dtype=self.a_dtype, b_dtype=_b_dtype, out_dtype=out_dtype, act="silu",
                    waves_per_eu=waves_per_eu, use_async_copy=use_async_copy,
                    use_cshuffle_epilog=None, contiguous_io=True, dedup_gather=False,
                    atom_contract=self.atom_contract,
                    sparse_tiles=True, persist_m=-1,
                    raw_a_scale=True, xcd_swizzle=self._xcd,
                    fuse_dispatch=scheme, fuse_npes=world_size, fuse_topk=topk,
                    fuse_cap=self.cap, fuse_mtpr=self.mtpr,
                    fuse_scale_dim=_fuse_sdim, fuse_scale_type_size=_fuse_stsz,
                    rank=rank, experts_per_rank=self.epr, compact_dispatch=self.compact,
                    compact_allgather=True)
            if dist.is_initialized():
                dist.barrier()

        # Views into op buffers (A-input + raw A-scale + metadata the GEMM consumes).
        v = self.op._ll_views()
        self._rx = v["rx_em"]
        self._scale_i32 = v["scale_em_i32"]
        self._trb = self.op.tile_row_base
        self._se = self.op.sorted_expert_ids
        self._nv = self.op.num_valid

        # Output / scratch.
        self.out_dtype = out_dtype
        _nf4 = out_dtype == "fp4"
        _nf8 = out_dtype == "fp8"
        self.out_is_quant = _nf4 or _nf8
        self._a2rows = world_size * self.mtpr * topk
        if _nf4:
            self._out = torch.zeros((self._a2rows, inter_dim // 2), dtype=torch.uint8, device=self.dev)
        elif _nf8:
            self._out = torch.zeros((self._a2rows, inter_dim), dtype=torch.float8_e4m3fn, device=self.dev)
        else:
            self._out = torch.zeros((self._a2rows, 1, inter_dim), dtype=torch.float16, device=self.dev)
        self._bias = torch.empty((0,), dtype=torch.float32, device=self.dev)
        # COMPACT sorted_token_ids[compact_row] = (k<<24)|src_global into a [nvm] buffer.
        self._sti = torch.zeros(self.nvm, dtype=torch.int32, device=self.dev)
        # sorted_expert_ids at 32-row SUB-TILE granularity -> nvm/32 entries.
        self._se_atom = torch.zeros(self.nvm // 32 + 8, dtype=torch.int32, device=self.dev)
        # combine wts_buf (f32, LOGICAL layout wts[t*topk+s], t=src_global via identity tis).
        self._sw_atom = torch.zeros(world_size * self.mtpr * topk, dtype=torch.float32, device=self.dev)
        # sorted-row routing weights (f32, [nvm], parallel to _sti).
        self._wts_sorted = torch.zeros(self.nvm, dtype=torch.float32, device=self.dev)
        # e8m0 sorted-scale buffer (a2_scale).
        if self.out_is_quant:
            _sbm = max(32, self.unit_size)
            _prows = ((self.max_blocks * _sbm + 255) // 256) * 256
            _pcols = (((inter_dim // 32) + 7) // 8) * 8
            self._osd = torch.zeros(_prows * _pcols + inter_dim, dtype=torch.uint8, device=self.dev)
        else:
            self._osd = torch.empty((0,), dtype=torch.uint8, device=self.dev)
        self.out_scale = self._osd

        # dispatch-arg FIXED-pointer table (op bufs + p2p); built ONCE.
        self._build_disp_table()

        # stage-2 handoff contract (stable buffers; forward writes them in place).
        self.output = Stage1Output(
            a2=self._out, a2_scale=self._osd,
            sorted_token_ids=self._sti, sorted_expert_ids=self._se_atom,
            sorted_weights=self._wts_sorted, num_valid_ids=self._nv,
            wts_buf=self._sw_atom,
        )

    def _build_disp_table(self):
        op = self.op
        if self.scheme == "fixedslot":
            tbl = [0, 0, 0, 0,                                   # 0-3 inputs (lazy)
                   op.gb1.data_ptr(), op.running.data_ptr(), op.done2.data_ptr(),
                   op.ll_count.data_ptr(),                       # 4-7
                   op.p2p_rx_em.data_ptr(), op.p2p_scale_em.data_ptr(), op.p2p_idx_em.data_ptr(),
                   op.p2p_wts_em.data_ptr(), op.p2p_srcmap_em.data_ptr(), op.p2p_running.data_ptr(),
                   op.p2p_done2.data_ptr(),                      # 8-14
                   op.sorted_expert_ids.data_ptr(), op.tile_row_base.data_ptr(),
                   op.num_valid.data_ptr(),                      # 15-17
                   self._meta.data_ptr()]                        # 18 metadata-ready flag
            if self.compact:
                # 19-35: compact count-first extras
                tbl += [op.compact_base.data_ptr(), op.p2p_compact_base.data_ptr(),  # 19-20
                        op.gb_cnt.data_ptr(), op.done2c.data_ptr(), op.p2p_done2c.data_ptr(),  # 21-23
                        op.meta2.data_ptr(),                                          # 24 payload-ready flag
                        op.write_cursor.data_ptr(), op.p2p_write_cursor.data_ptr(),   # 25-26
                        op.done2cb.data_ptr(), op.p2p_done2cb.data_ptr(),             # 27-28 cross-PE#1b
                        op.local_hist.data_ptr(), op.bigcnt.data_ptr(), op.p2p_bigcnt.data_ptr(),  # 29-31
                        op.cnt_done.data_ptr(), op.p2p_cnt_done.data_ptr(),           # 32-33
                        op.my_base.data_ptr(), op.local_cursor.data_ptr()]            # 34-35
                tbl[19] = op.srcmap_em.data_ptr()                # 19 LOCAL srcmap (override compact_base)
                tbl[20] = self._sw_atom.data_ptr()               # 20 compact sorted_weights out
                tbl += [(self._trecv_buf if self._trecv_buf is not None else op.total_recv).data_ptr(),
                        op.dest_ctr.data_ptr(),                  # 36 total_recv | 37 dest_ctr(local)
                        op.recv_num.data_ptr(), op.p2p_recv_num.data_ptr(),  # 38 recv_num | 39 p2p_recv_num
                        self._sti.data_ptr(), self._se_atom.data_ptr(),      # 40 _sti | 41 _se_atom
                        self._wts_sorted.data_ptr()]                         # 42 sorted-row weights out
            else:
                tbl += [op.srcmap_em.data_ptr(),                 # 19 LOCAL srcmap (k_slot<<24|src_global)
                        self._sw_atom.data_ptr(),                # 20 compact sorted_weights out
                        (self._trecv_buf if self._trecv_buf is not None else op.total_recv).data_ptr(),
                        op.dest_ctr.data_ptr(),                              # 21 total_recv | 22 dest_ctr(local)
                        op.recv_num.data_ptr(), op.p2p_recv_num.data_ptr(),  # 23 recv_num(local) | 24 p2p_recv_num
                        self._wts_sorted.data_ptr()]                         # 25 sorted-row weights out
        self._disp = torch.tensor(tbl, dtype=torch.int64, device=self.dev)
        self._disp_host = torch.tensor(tbl, dtype=torch.int64)

    def _agv(self, t):
        return t.view(torch.uint8) if self.a_dtype == "fp4" else t

    def impl_for(self, tokens) -> str:
        return "mega_" + self.scheme

    def reset_counters(self, tokens=None):
        return

    def forward(self, x, wts, scales, topk_ids, stream=None) -> "Stage1Output":
        """Run the WHOLE stage-1 (dispatch + group-GEMM) in ONE launch.

        Returns the stable ``Stage1Output`` (its buffers are overwritten here).
        """
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream())
        cur_tok = int(x.shape[0])
        assert cur_tok <= self.mtpr, f"[mega] cur_tok={cur_tok} > max_tok_per_rank={self.mtpr}"
        xc = x.contiguous(); wc = wts.contiguous()
        ic = topk_ids.to(torch.int32).contiguous(); sc = scales.contiguous()

        # a8w4smooth: send-side per-slot front smoothquant -> token-major int8 + per-slot f32 scale
        # (the fused dispatch prologue then scatters each slot's row PER-SLOT). This replaces the raw
        # bf16/fp8 payload+scale pointers with the pre-quantized int8 A + f32 scale.
        if self._is_int8:
            # Front smoothquant via aiter's device `smooth_per_token_scaled_quant` -- the SAME kernel
            # AG/RS's fused_moe uses -- in ONE graph-safe launch: token-major [T,topk,K] int8 + f32
            # scale (row = t*topk+slot) that the dispatch scatters per-slot. x is broadcast over slots
            # (expand, stride-0); per-(token,slot) expert = topk_ids drives fc1_smooth[expert].
            _T = int(xc.shape[0])
            self._sptsq(self._fq_i8[:_T * self.topk].view(_T, self.topk, self.model_dim),
                        xc.view(_T, 1, self.model_dim).expand(_T, self.topk, self.model_dim),
                        self._fq_scale[:_T * self.topk].view(_T, self.topk, 1),
                        self._fc1_smooth, ic, smooth_scale_map_hash=None, enable_ps=True)
            _a_in_ptr = self._fq_i8.data_ptr()
            _sc_in_ptr = self._fq_scale.data_ptr()
        else:
            _a_in_ptr = xc.data_ptr()
            _sc_in_ptr = sc.data_ptr()

        pd_ptr = 0
        # total_recv accumulates in-kernel each launch -> zero first (graph-safe).
        (self._trecv_buf if self._trecv_buf is not None else self.op.total_recv).zero_()
        er_ptr = (self.op.ll_count.data_ptr()
                  if (self.scheme == "fixedslot" and not self.compact)
                  else 0)
        if self.compact:
            self.op.local_hist.zero_()
            self.op.local_cursor.zero_()
        a_mat = self._agv(self._rx)
        if self.compact:
            _sorted_arg = self._trb
            _se_arg = self._se
        else:
            _sorted_arg = self._sti
            _se_arg = self._se_atom
        _wt_arg = self.op.wts_em
        # a8w4smooth/w8a8smooth thread two trailing packed4 W1 qscale/qzero addresses for the
        # in-kernel int4->int8 B-dequant (w8a8smooth is FULL int8 W1 -> pass 0/0; gemm1 is_int8_full
        # skips them). mxscale (fp8/fp4) has no such trailing args. The int8 dispatch also reads the
        # pre-quantized per-slot int8 A (_a_in_ptr) + f32 scale (_sc_in_ptr) set above.
        _int8_qp = ()
        if self._is_int8:
            _qs_ptr = fx.Int64(self._qs_w.data_ptr()) if self._qs_w is not None else fx.Int64(0)
            _qz_ptr = fx.Int64(self._qz_w.data_ptr()) if self._qz_w is not None else fx.Int64(0)
            _int8_qp = (_qs_ptr, _qz_ptr)
        self.mega(self._out, a_mat, self.w1, self._scale_i32, self.w1_scale,
                  _sorted_arg, _se_arg, _wt_arg, self._nv, self._bias, self._osd,
                  fx.Int32(self.nvm), fx.Int32(self.inter_dim * 2), fx.Int32(self.model_dim),
                  fx.Int32(self.max_blocks),
                  fx.Int64(pd_ptr), fx.Int64(er_ptr),
                  fx.Int64(self._disp.data_ptr()), fx.Int32(cur_tok),
                  fx.Int64(_a_in_ptr), fx.Int64(ic.data_ptr()),
                  fx.Int64(wc.data_ptr()), fx.Int64(_sc_in_ptr),
                  *_int8_qp,
                  stream=stream)
        return self.output


# ======================================================================================
# End-to-end EP-MoE single operator
# ======================================================================================
class MegaMoE:
    """End-to-end EP-MoE single operator (a full MoE layer).

    ``enable_fused_stage1`` / ``enable_fused_stage2`` pick fused vs non-fused per stage at init:
      * stage-1 fused    : single-launch dispatch + GEMM1 megakernel (``MegaMoeStage1``).
      * stage-1 non-fused: bf16 dispatch (comb_op) + moe_sorting + ``compile_mixed_moe_gemm1``
                           (requires ``forward(x_bf16, ...)``; ``forward_prequant`` is fused-only).
      * stage-2 fused    : fused GEMM2 (in-epilogue P2P scatter) + ``combine_no_stage1`` (MegaMoeStage2).
      * stage-2 non-fused: standalone ``compile_mixed_moe_gemm2`` + separate ``comb_op.combine``.

    Supported combos: (fused, fused), (non-fused, non-fused), (non-fused, fused). The
    (fused stage-1, non-fused stage-2) combo is rejected at init: the fused stage-1 bypasses
    ``dispatch()`` and only establishes the ``combine_no_stage1`` handoff (identity tok_id_to_src
    + total_recv), not the full dispatch bookkeeping (shmem_disp_out_idx / disp_tok_map / per-token
    flags) that the STANDARD ``combine`` (used by a non-fused stage-2) needs for its Stage-1
    gather+scatter. (The a2/_sti layout itself is fine -- a standalone GEMM2 decodes
    ``t = sti & 0xFFFFFF = src_global``; the mismatch is purely at the combine boundary.)
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        model_dim: int,
        inter_dim: int,
        experts: int,
        topk: int,
        quant: str,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        max_tok_per_rank: int,
        # ---- a8w4smooth (LQQ W4A8 + smoothquant) extra weights; None for a8w4/a4w4 ----
        w1_lqq_scale: Optional[torch.Tensor] = None,
        w1_lqq_zero: Optional[torch.Tensor] = None,
        w2_lqq_scale: Optional[torch.Tensor] = None,
        w2_lqq_zero: Optional[torch.Tensor] = None,
        fc1_smooth_scale: Optional[torch.Tensor] = None,
        fc2_smooth_scale: Optional[torch.Tensor] = None,
        expert_mask: Optional[torch.Tensor] = None,
        local_expert_hash: Optional[torch.Tensor] = None,
        tune_tokens: Optional[int] = None,
        network: Optional[str] = None,
        mega_scheme: str = "fixedslot",
        tile_m: int = -1,
        tile_n: int = -1,
        tile_k: int = 256,
        gemm2_tile_m: int = -1,
        gemm2_tile_n: int = -1,
        gemm2_tile_k: int = -1,
        gemm2_persist_m: int = -1,
        warp_num_per_block: int = 4,
        waves_per_eu: int = 4,
        use_async_copy: bool = True,
        block_num: Optional[int] = None,
        xcd_swizzle: int = 0,
        gemm2_tile_table: Optional[dict] = None,
        enable_fused_stage1: bool = True,
        enable_fused_stage2: bool = True,
        weight_format: str = "megamoe",
    ):
        assert quant in ("a8w4", "a4w4", "a8w4smooth", "w8a8smooth"), quant
        assert weight_format in ("megamoe", "aiter_lqq"), weight_format
        # a8w4smooth (LQQ int8 x packed-int4 + qscale/qzero + smoothquant) is the AG/RS repro's
        # datatype. FUSED-only: front smoothquant + per-slot int8 dispatch + int8 gemm1 (MegaMoeStage1)
        # -> recv requant -> int8 gemm2 + combine (MegaMoeStage2).
        self._is_a8w4smooth = (quant == "a8w4smooth")
        # w8a8smooth (W8A8 + smoothquant): int8 activation x FULL int8 weight (per-output-row f32
        # scale, NO packed-int4 / NO LQQ qscale/qzero). Shares a8w4smooth's fused int8-activation
        # datapath; ONLY the weight datapath differs (b_dtype='int8' baked into stage1/_g2).
        self._is_w8a8smooth = (quant == "w8a8smooth")
        # int8-smooth = a8w4smooth (packed-int4 W + LQQ) OR w8a8smooth (FULL int8 W, per-row f32 scale).
        # Both ride the SAME fused megakernels (front smoothquant -> per-slot int8 dispatch -> int8
        # gemm1 -> requant -> int8 gemm2 + combine); ONLY the weight datapath differs.
        self._is_int8_smooth = self._is_a8w4smooth or self._is_w8a8smooth
        # int8 smoothquant is FUSED-ONLY (the bf16-dispatch non-fused reference paths were validation
        # scaffolding, removed from the delivery). Require (fused, fused).
        if self._is_int8_smooth and not (bool(enable_fused_stage1) and bool(enable_fused_stage2)):
            raise ValueError(
                "quant in {a8w4smooth,w8a8smooth} is fused-only; "
                "set enable_fused_stage1=enable_fused_stage2=True.")
        self._a8_fused = self._is_a8w4smooth
        self._w8a8_fused = self._is_w8a8smooth
        self._int8_fused = self._is_int8_smooth
        if not self._is_a8w4smooth and not self._is_w8a8smooth:
            try:
                _mx_quant()   # a8w4/a4w4 (mxscale) need the fp8/fp4 quant kernel; smoothquant does not
            except Exception as _e:  # noqa: BLE001
                raise RuntimeError(
                    "MegaMoE(a8w4/a4w4) needs aiter.ops.quant.per_1x32_mx_quant_hip") from _e
        if (self._is_a8w4smooth or self._is_w8a8smooth) and (
                aiter is None or not hasattr(aiter, "moe_smooth_per_token_scaled_quant")):
            raise RuntimeError(
                "MegaMoE(a8w4smooth/w8a8smooth) needs aiter.moe_smooth_per_token_scaled_quant + moe_sorting_fwd")
        if (max_tok_per_rank & (max_tok_per_rank - 1)) != 0:
            raise ValueError(f"max_tok_per_rank={max_tok_per_rank} must be a power of two")
        # weight_format='aiter_lqq': w1/w2 are the aiter LQQ ``lqq_1x64_quant`` RAW outputs
        # (u4 [E,rows,K] uint8 + per-64-K scale/zero [E,rows,K//64] uint8). Re-lay-out them into
        # megamoe's packed4 + interleave_k64 layout ONCE here (dequant math is identical), so the
        # rest of __init__ sees ordinary megamoe a8w4smooth weights.
        self.weight_format = weight_format
        if weight_format == "aiter_lqq":
            if not self._is_a8w4smooth:
                raise ValueError("weight_format='aiter_lqq' is only valid for quant='a8w4smooth'")
            if any(t is None for t in (w1_lqq_scale, w1_lqq_zero, w2_lqq_scale, w2_lqq_zero)):
                raise ValueError(
                    "weight_format='aiter_lqq' needs w1/w2 (u4 [E,rows,K] uint8) and "
                    "w1/w2_lqq_scale + w1/w2_lqq_zero (u8 [E,rows,K//64]) from lqq_1x64_quant")
            w1, w1_lqq_scale, w1_lqq_zero = _convert_aiter_lqq_to_megamoe(w1, w1_lqq_scale, w1_lqq_zero)
            w2, w2_lqq_scale, w2_lqq_zero = _convert_aiter_lqq_to_megamoe(w2, w2_lqq_scale, w2_lqq_zero)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.model_dim = int(model_dim)
        self.inter_dim = int(inter_dim)
        self.experts = int(experts)
        self.epr = experts // world_size
        self.topk = int(topk)
        self.quant = quant
        self.mtpr = int(max_tok_per_rank)
        self.enable_fused_stage1 = bool(enable_fused_stage1)
        self.enable_fused_stage2 = bool(enable_fused_stage2)
        # The blocker is the COMBINE boundary, not the a2 layout: the standalone GEMM2 masks
        # t = sti & 0xFFFFFF = src_global and writes gemm2_out[src_global] fine, so it CAN consume
        # fused stage-1's a2/_sti. But the fused stage-1 megakernel bypasses dispatch() and only sets
        # up the combine_no_stage1 handoff (tok_id_to_src=identity + total_recv) -- NOT the full
        # dispatch bookkeeping (shmem_disp_out_idx / disp_tok_map / per-token flags) that the STANDARD
        # comb_op.combine's Stage-1 gather+scatter reads. Fused stage-2 sidesteps this (its GEMM2
        # epilogue P2P-scatters directly, combine_no_stage1 only reduces); a non-fused stage-2 would
        # feed the standard combine fused stage-1's stale dispatch tables -> wrong result. Disallow.
        if self.enable_fused_stage1 and not self.enable_fused_stage2:
            raise ValueError(
                "enable_fused_stage1=True with enable_fused_stage2=False is unsupported: the fused "
                "stage-1 sets up only the combine_no_stage1 handoff (identity tok_id_to_src + "
                "total_recv), not the dispatch bookkeeping the standard combine needs for a non-fused "
                "stage-2. Supported combos: (fused,fused), (non-fused,non-fused), (non-fused,fused). "
                "Set enable_fused_stage1=False for a non-fused stage-2.")
        self.dev = torch.device("cuda", rank)
        self._is_fp4 = (quant == "a4w4")
        if self._int8_fused:
            # a8w4smooth/w8a8smooth fused: no mxscale per_1x32 quant (front smoothquant runs inside stage-1).
            self._qd = None
            self._stp = None
        else:
            self._qd = _adt.fp4x2 if self._is_fp4 else _adt.fp8
            self._stp = None if self._is_fp4 else _adt.fp8_e8m0

        # Resolve stage-1 tile / layout (tiles used by both fused & non-fused gemm1; compact sizing).
        self._s1cfg = _resolve_stage1_config(
            model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
            quant=quant, max_tok_per_rank=max_tok_per_rank, world_size=world_size,
            rank=rank, tune_tokens=tune_tokens, unit_size=int(tile_m), tile_n=int(tile_n),
            tile_k=int(tile_k), waves_per_eu=int(waves_per_eu),
            use_async_copy=bool(use_async_copy))
        self.a2_dtype = self._s1cfg["a_dtype"]  # "fp8" | "fp4"
        self.mega_scheme = mega_scheme

        # Combine op (bf16). group-major dispatch buffers only for the fused stage-1 megakernel;
        # the non-fused stage-1 uses the op's standard (token-major) bf16 dispatch instead.
        self.comb_cfg = FlyDSLDispatchCombineConfig(
            rank=rank, world_size=world_size, hidden_dim=model_dim,
            max_num_inp_token_per_rank=self.mtpr, num_experts_per_rank=self.epr,
            num_experts_per_token=topk, data_type=torch.bfloat16,
            scale_dim=0, scale_type_size=0, enable_std_moe=False,
            enable_group_major=self.enable_fused_stage1,
            gm_data_type=self._s1cfg["data_type"],
            gm_unit_size=self._s1cfg["unit_size"],
            gm_scale_dim=self._s1cfg["scale_dim"],
            gm_scale_type_size=(4 if self._is_int8_smooth else 1),  # a8w4smooth/w8a8smooth: f32 per-(token,slot) scale
            gm_scheme=mega_scheme,
            gm_compact=self._s1cfg["compact"],
        )
        self.comb_op = FlyDSLDispatchCombineIntraNodeOp(self.comb_cfg)
        torch.cuda.synchronize(); ms.shmem_barrier_all()

        self.w2 = w2 if w2.is_contiguous() else w2.contiguous()
        self.w2_scale = w2_scale if w2_scale.is_contiguous() else w2_scale.contiguous()
        self.max_recv = world_size * self.mtpr

        # ---- stage-1 ----
        if self.enable_fused_stage1:
            # Fused dispatch + GEMM1 megakernel (uses comb_op._gm buffers).
            # a8w4smooth: thread packed4 W1 qparams + GLOBAL fc1_smooth (front smoothquant is inside
            # MegaMoeStage1.forward, so it needs every expert a local token may route to).
            if self._a8_fused:
                _s1_a8 = dict(w1_lqq_scale=w1_lqq_scale, w1_lqq_zero=w1_lqq_zero, fc1_smooth=fc1_smooth_scale)
            elif self._w8a8_fused:
                # w8a8smooth: FULL int8 W1 (no LQQ packed4 qparams); still needs GLOBAL fc1_smooth.
                _s1_a8 = dict(fc1_smooth=fc1_smooth_scale)
            else:
                _s1_a8 = {}
            self.stage1 = MegaMoeStage1(
                comm_op=self.comb_op, cfg=self._s1cfg,
                rank=rank, world_size=world_size, model_dim=model_dim, inter_dim=inter_dim,
                experts=experts, topk=topk, quant=quant, w1=w1, w1_scale=w1_scale,
                max_tok_per_rank=max_tok_per_rank, scheme=mega_scheme,
                warp_num_per_block=int(warp_num_per_block), out_dtype="auto", **_s1_a8)
            self.sort_block_m = int(self.stage1.sort_block_m)
            # megav1 _sti encodes src_global=dest -> combine tok_id_to_src is identity (const).
            self.comb_op.shmem_tok_id_to_src.copy_(
                torch.arange(self.max_recv, device=self.dev, dtype=torch.int32))
            torch.cuda.synchronize(); ms.shmem_barrier_all()
        else:
            # Non-fused stage-1: bf16 dispatch (comb_op) + aiter moe_sorting + compile_mixed_moe_gemm1.
            self.stage1 = None
            self._build_nonfused_stage1(w1, w1_scale)
            torch.cuda.synchronize(); ms.shmem_barrier_all()

        # ---- stage-2 ----
        _s1_sbm = int(self.sort_block_m)
        if self.enable_fused_stage2:
            # fused GEMM2 ghost-gate = constant max_recv (megav1 self-contained; relies on the
            # fused stage-1's identity routing). For non-fused stage-1 the op keeps real total_recv.
            if self.enable_fused_stage1:
                self._gemm2_gate_bound = torch.tensor(
                    [self.max_recv], dtype=torch.int32, device=self.dev)
                self.comb_op._fx_out_total_recv = fx.Int64(self._gemm2_gate_bound.data_ptr())
                torch.cuda.synchronize(); ms.shmem_barrier_all()
            _eff_g2_tm = int(gemm2_tile_m) if int(gemm2_tile_m) > 0 else 32  # op default tile_m=32
            if _s1_sbm % _eff_g2_tm != 0:
                raise ValueError(
                    f"gemm2 tile_m={_eff_g2_tm} does not divide stage1 sort_block_m={_s1_sbm}; "
                    f"set gemm2_tile_m to a divisor of {_s1_sbm} (e.g. 32).")
            if gemm2_tile_table is None:
                try:
                    gemm2_tile_table = _mega_gemm2_tuned_table(
                        int(model_dim), int(inter_dim),
                        int(self.experts) // int(self.world_size), int(topk),
                        int(self.world_size), _detect_gpu_model_name(int(self.rank)),
                        quant=self.quant)
                    if int(self.rank) == 0:
                        print(f"[MegaGemm2] auto-config tile table (sort_block_m={_s1_sbm}): "
                              f"{'<miss -> default tile>' if not gemm2_tile_table else gemm2_tile_table}",
                              flush=True)
                except Exception:  # noqa: BLE001 -- fallback to default tile
                    gemm2_tile_table = None
            # persist_m / xcd_swizzle are per-launch scalars, ORTHOGONAL to the tile table (which
            # only holds (tile_m,tile_n,tile_k)). Forward them unconditionally so tuning them does
            # NOT require also pinning tile_m (which would clobber the tune table's optimal prefill
            # tile_m=64). tile_m/n/k stay conditional -> only an explicit gemm2_tile_m>0 overrides
            # the table. gemm2_persist_m<=0 (default -1) -> MegaMoeStage2's persistent mode
            # (grid_y=cu_num, matches aiter's dsv4/EP gemm2 `_persist`); >0 opts into legacy N-tile.
            # a8w4smooth: gemm2 is int8 x packed-int4 (a2_dtype=='int8' from _s1cfg); thread W2 qparams.
            g2_kwargs = dict(
                comb_cfg=self.comb_cfg, comb_op=self.comb_op, inter_dim=int(inter_dim),
                a_dtype=self.a2_dtype,
                b_dtype=("int8" if self._w8a8_fused else ("int4" if self._a8_fused else "fp4")),
                sort_block_m=_s1_sbm,
                gemm2_tile_table=gemm2_tile_table,
                xcd_swizzle=int(xcd_swizzle),
            )
            if int(gemm2_persist_m) > 0:
                g2_kwargs["persist_m"] = int(gemm2_persist_m)
            if int(gemm2_tile_m) > 0:
                g2_kwargs.update(
                    tile_m=int(gemm2_tile_m), tile_n=int(gemm2_tile_n), tile_k=int(gemm2_tile_k))
            if self._int8_fused:
                # int8 gemm2 needs tile_k=256 (K64 packs / kpack=16). Priority: explicit gemm2_tile_m param
                # > JSON MegaGemm2 tune table (int8 rows, dtype-mapped) > default 32x64. tile_k is ALWAYS
                # forced to 256 (MegaMoeStage2 re-forces it for int8, so table rows can't break it).
                if int(gemm2_tile_m) > 0:
                    g2_kwargs.update(tile_m=int(gemm2_tile_m), tile_n=int(gemm2_tile_n), tile_k=256,
                                     gemm2_tile_table=None)
                elif gemm2_tile_table:
                    pass  # keep the int8 tune table; MegaMoeStage2 picks per-bucket tiles (tile_k->256)
                else:
                    g2_kwargs.update(tile_m=32, tile_n=64, tile_k=256, gemm2_tile_table=None)
                if self._a8_fused:
                    # a8w4smooth: thread W2 packed4 qparams (w8a8smooth full-int8 W2 has none).
                    g2_kwargs.update(w2_lqq_scale=w2_lqq_scale, w2_lqq_zero=w2_lqq_zero)
            self._g2 = MegaMoeStage2(**g2_kwargs)
            if self._int8_fused:
                # LOCAL fc2_smooth slice [epr, inter]: the v2 requant indexes smooth_scale by the LOCAL
                # sorted expert id (0..epr-1) that stage-1 emits. Preallocated int8 a2 + f32 a2_scale
                # scratch (ATOM row = token*topk+slot), fed to the fused int8 gemm2.
                _lo = self.rank * self.epr
                self._a8f_fc2_smooth = fc2_smooth_scale.to(torch.float32)[_lo:_lo + self.epr].contiguous()
                _a2rows = world_size * self.mtpr * topk
                self._a8f_a2i8 = torch.zeros((_a2rows, inter_dim), dtype=torch.int8, device=self.dev)
                self._a8f_a2scale = torch.zeros((_a2rows,), dtype=torch.float32, device=self.dev)
                # Hoist the fused-forward's per-step host junk into __init__ so the hot loop is just
                # kernel launches: preallocated (unused) stage-1 `scales` placeholder, cached host
                # constants, and the cached aiter requant fn (no per-forward import).
                self._a8f_dummy_sc = torch.zeros((self.mtpr, 1), dtype=torch.float32, device=self.dev)
                self._a8f_G = int(world_size * self.mtpr)
                self._a8f_g2_sbm = int(self._g2.sort_block_m)
                from aiter import moe_smooth_per_token_scaled_quant as _msptsq
                self._a8f_msptsq = _msptsq
        else:
            # Non-fused stage-2: standalone compile_mixed_moe_gemm2 + comb_op.combine.
            self._g2 = None
            self._build_nonfused_stage2(
                int(gemm2_tile_m), int(gemm2_tile_n), int(gemm2_tile_k), int(gemm2_persist_m))

    # ------------------------------------------------------------------ non-fused stage-1
    def _build_nonfused_stage1(self, w1, w1_scale):
        """Buffers + gemm1 for the non-fused stage-1 (bf16 dispatch -> moe_sorting -> GEMM1)."""
        dev, epr, topk = self.dev, self.epr, self.topk
        model_dim, inter_dim, experts = self.model_dim, self.inter_dim, self.experts
        max_recv = self.max_recv
        tm, tn1, tk = 32, 128, 256
        self._nf_tm = tm
        self.sort_block_m = max(32, tm)
        _max_pad = max_recv * topk + experts * tm
        _max_blocks = (_max_pad + tm - 1) // tm
        self._nf_max_blocks = _max_blocks
        _scaleN_pad = ((model_dim // 32 + 7) // 8) * 8
        self._nf_a_st = torch.empty(_max_pad, dtype=torch.int32, device=dev)
        self._nf_a_sw = torch.empty(_max_pad, dtype=torch.float32, device=dev)
        self._nf_a_se = torch.empty(_max_blocks, dtype=torch.int32, device=dev)
        self._nf_a_se_local = torch.empty(_max_blocks, dtype=torch.int32, device=dev)
        self._nf_a_nv = torch.zeros(2, dtype=torch.int32, device=dev)
        self._nf_a_mbuf = torch.empty((max_recv, model_dim), dtype=torch.float16, device=dev)
        self._nf_a1s = torch.empty(((_max_pad + 31) // 32 * 32, _scaleN_pad),
                                   dtype=_adt.fp8_e8m0, device=dev)
        self._nf_recv_topk = torch.empty((max_recv, topk), dtype=torch.int32, device=dev)
        self._nf_sentinel = torch.full((max_recv, topk), experts, dtype=torch.int32, device=dev)
        if self._is_fp4:
            self._nf_a2 = torch.zeros((max_recv * topk, inter_dim // 2), dtype=torch.uint8, device=dev)
        else:
            self._nf_a2 = torch.zeros((max_recv * topk, inter_dim), dtype=torch.float8_e4m3fn, device=dev)
        _sbm = self.sort_block_m
        _pr = ((_max_blocks * _sbm + 255) // 256) * 256
        _pc = (((inter_dim // 32) + 7) // 8) * 8
        self._nf_a2s = torch.zeros(_pr * _pc + inter_dim, dtype=torch.uint8, device=dev)
        self._nf_bias = torch.empty((0,), dtype=torch.float32, device=dev)
        self._nf_w1 = w1.contiguous()
        self._nf_w1s = w1_scale.contiguous()
        self._nf_gemm1 = compile_mixed_moe_gemm1(
            model_dim=model_dim, inter_dim=inter_dim, experts=epr, topk=topk,
            tile_m=tm, tile_n=tn1, tile_k=tk, doweight_stage1=False,
            a_dtype=self.a2_dtype, b_dtype="fp4", out_dtype=self.a2_dtype,
            act="silu", waves_per_eu=4, use_async_copy=True)
        self._nf_trc = None  # cached total_recv (fixed routing) so forward stays CUDAGraph-safe

    def _run_nonfused_stage1(self, x_bf16, wc, ic) -> "Stage1Output":
        """bf16 dispatch (comb_op) -> recv-quant -> moe_sorting -> GEMM1 -> Stage1Output."""
        epr, rank, topk = self.epr, self.rank, self.topk
        model_dim, inter_dim, experts = self.model_dim, self.inter_dim, self.experts
        max_recv, tm = self.max_recv, self._nf_tm
        self._nf_a2.zero_()
        self.comb_op.total_recv.zero_()
        _bt, _ow, _, _oidx, _ = self.comb_op.dispatch(x_bf16, wc, None, ic)
        if self._nf_trc is None:
            self._nf_trc = max(1, int(self.comb_op.total_recv.item()))
        trc = self._nf_trc
        _oi = _oidx[:trc].to(torch.int32)
        _loc = (_oi >= rank * epr) & (_oi < (rank + 1) * epr)
        self._nf_recv_topk[:trc].copy_(torch.where(_loc, _oi, self._nf_sentinel[:trc]))
        aiter.moe_sorting_fwd(self._nf_recv_topk[:trc], _ow[:trc], self._nf_a_st, self._nf_a_sw,
                              self._nf_a_se, self._nf_a_nv, self._nf_a_mbuf[:trc],
                              int(experts), int(tm), None, None, 0)
        _q = _mx_quant()
        if self._is_fp4:
            _a1q, _a1sp = _q(_bt[:trc].contiguous(), quant_dtype=self._qd)
        else:
            _a1q, _a1sp = _q(_bt[:trc].contiguous(), quant_dtype=self._qd,
                                                scale_type=self._stp)
        aiter.mxfp4_moe_sort_hip(self._nf_a1s, _a1sp, self._nf_a_st, self._nf_a_nv,
                                 int(trc), int(model_dim))
        self._nf_a_se_local.copy_(self._nf_a_se - rank * epr)
        a_mat = _a1q.view(torch.uint8) if self.a2_dtype == "fp4" else _a1q
        self._nf_gemm1(self._nf_a2.view(max_recv, topk, self._nf_a2.shape[-1]), a_mat,
                       self._nf_w1, self._nf_a1s.view(torch.uint8), self._nf_w1s,
                       self._nf_a_st, self._nf_a_se_local, self._nf_a_sw, self._nf_a_nv,
                       self._nf_bias, self._nf_a2s, fx.Int32(trc),
                       fx.Int32(inter_dim * 2), fx.Int32(model_dim), fx.Int32(int(self._nf_max_blocks)),
                       stream=fx.Stream(torch.cuda.current_stream()))
        return Stage1Output(
            a2=self._nf_a2, a2_scale=self._nf_a2s,
            sorted_token_ids=self._nf_a_st, sorted_expert_ids=self._nf_a_se_local,
            sorted_weights=self._nf_a_sw, num_valid_ids=self._nf_a_nv,
            wts_buf=None)  # None -> combine uses shmem_disp_out_wts (written by the bf16 dispatch)

    # ------------------------------------------------------------------ non-fused stage-2
    def _build_nonfused_stage2(self, g2_tm, g2_tn, g2_tk, g2_pm):
        """Standalone GEMM2 (compile_mixed_moe_gemm2, doweight, token-level accumulate) + buffer."""
        tm2 = g2_tm if g2_tm > 0 else 32
        tn2 = g2_tn if g2_tn > 0 else 128
        tk2 = g2_tk if g2_tk > 0 else 256
        pm = g2_pm if g2_pm > 0 else -1
        sbm = int(self.sort_block_m)
        if sbm % tm2 != 0:
            raise ValueError(f"gemm2 tile_m={tm2} must divide sort_block_m={sbm}")
        self._nfg2_exe = compile_mixed_moe_gemm2(
            model_dim=self.model_dim, inter_dim=self.inter_dim, experts=self.epr, topk=self.topk,
            tile_m=tm2, tile_n=tn2, tile_k=tk2, doweight_stage2=True,
            a_dtype=self.a2_dtype, b_dtype="fp4", out_dtype="bf16",
            accumulate=True, persist_m=pm, sort_block_m=sbm)
        self._nfg2_out = torch.zeros(self.max_recv, self.model_dim, dtype=torch.bfloat16, device=self.dev)
        self._nfg2_bias = torch.empty((0,), dtype=torch.float32, device=self.dev)
        self._nfg2_blocks = (self.max_recv * self.topk + self.experts * sbm + sbm - 1) // sbm
        self._nfg2_c = {}

    def _run_nonfused_stage2(self, s1, run_tokens):
        """GEMM2 -> gemm2_out; then comb_op.combine (standard, does the Stage-1 scatter)."""
        self._nfg2_out.zero_()
        _ga = (self._nfg2_out, s1.a2.view(-1), self.w2, s1.a2_scale, self.w2_scale,
               s1.sorted_token_ids, s1.sorted_expert_ids, s1.sorted_weights, s1.num_valid_ids,
               self._nfg2_bias, self.max_recv, self.model_dim, self.inter_dim,
               int(self._nfg2_blocks), torch.cuda.current_stream())
        if self._nfg2_c.get("c") is None:
            self._nfg2_c["c"] = flyc.compile(self._nfg2_exe, *_ga)
        else:
            self._nfg2_c["c"](*_ga)
        _idx = self.comb_op.shmem_disp_out_idx.view(self.max_recv, self.topk)
        # wts_buf None (non-fused stage-1) -> combine reads shmem_disp_out_wts; fused stage-1 hands a
        # flat [max_recv*topk] buffer -> combine wants [max_recv, topk].
        _w = None if s1.wts_buf is None else s1.wts_buf.view(self.max_recv, self.topk)
        _r = self.comb_op.combine(self._nfg2_out, _w, _idx, cur_tok=run_tokens)
        return _r[0] if isinstance(_r, (tuple, list)) else _r

    def _forward_smoothquant_fused(self, x_bf16, wts, topk_ids, *, stream=None, slice_output=True):
        """FUSED int8 smoothquant (a8w4smooth / w8a8smooth): front-smoothquant + per-slot int8 dispatch
        + int8 gemm1 (MegaMoeStage1) -> recv requant(fc2_smooth) -> int8 gemm2 + combine (MegaMoeStage2).
        Both GEMMs are BIT-EXACT to the standalone kernels; this assembles them over the (fused,fused)
        plumbing. The packed4-vs-full-int8 weight datapath is baked into stage1/_g2 at construction
        (b_dtype='int4' for a8w4smooth, 'int8' for w8a8smooth)."""
        # Optional phase-boundary CUDA events (diagnostic; OFF by default -> no graph/runtime impact).
        # Splits wall-clock into stage1[front-quant+dispatch+gemm1] / requant / stage2[gemm2+combine].
        # (dispatch and gemm1 are ONE fused megakernel, as are gemm2 and the P2P scatter, so those
        # cannot be split at a host boundary -- use the per-kernel profiler for the rest.)
        _ph = getattr(self, "_phase_timing", False)
        if _ph:
            _pe = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            _pe[0].record()
        run_tokens = int(x_bf16.shape[0])
        ic = topk_ids.to(torch.int32).contiguous()
        wc = wts.contiguous()
        # MegaMoeStage1 (int8) runs its own front smoothquant internally; the `scales` arg is unused --
        # feed the preallocated dummy (sliced, no per-forward alloc/fill).
        s1 = self.stage1.forward(x_bf16[:run_tokens].contiguous(), wc,
                                 self._a8f_dummy_sc[:run_tokens], ic, stream=stream)
        if _ph:
            _pe[1].record()
        # Intermediate requant: a2 (f16, ATOM [_G,topk,inter]) -> per-(token,slot) int8 + f32 a2_scale,
        # via aiter's moe_smooth_per_token_scaled_quant v2 path -- the SAME call the customer AG/RS
        # repro uses. v2 consumes stage-1's sorted metadata DIRECTLY: it decodes each valid row's ATOM
        # position (t*topk+s, where t=src_global, s=slot) and its per-block LOCAL expert internally,
        # then writes a2_i8/a2_scale at that ATOM row. No separate per-row expert map / scatter is
        # needed (no all-gather, no ~10-op emap build) -- the routing already crossed during dispatch
        # and lives in stage-1's sorted_token_ids/sorted_expert_ids. topk_ids is unused by v2.
        _G = self._a8f_G
        self._a8f_msptsq(self._a8f_a2i8.view(_G, self.topk, self.inter_dim),
                         s1.a2.view(_G, self.topk, self.inter_dim),
                         self._a8f_a2scale, self._a8f_fc2_smooth, ic,
                         s1.sorted_token_ids, s1.sorted_expert_ids, s1.num_valid_ids,
                         self._a8f_g2_sbm, None, False, False)
        if _ph:
            _pe[2].record()
        # fused int8 gemm2 (+ in-epilogue P2P scatter) + combine_no_stage1.
        ret = self._g2.run(
            a2=self._a8f_a2i8, w2=self.w2,
            a2_scale=self._a8f_a2scale, w2_scale=self.w2_scale,
            sorted_token_ids=s1.sorted_token_ids, sorted_expert_ids=s1.sorted_expert_ids,
            sorted_weights=s1.sorted_weights, num_valid_ids=s1.num_valid_ids, wts_buf=s1.wts_buf,
            cur_tok=run_tokens, run_tokens=run_tokens, stream=stream)
        if _ph:
            _pe[3].record()
            self._phase_evs = _pe
        out_tok = ret[0] if isinstance(ret, (tuple, list)) else ret
        return out_tok[:run_tokens] if slice_output else out_tok

    def quantize(self, x_bf16):
        """Quantize bf16 activation -> (fp8/fp4 payload, e8m0 scale as uint8)."""
        _q = _mx_quant()
        if self._is_fp4:
            mq, msq = _q(x_bf16.contiguous(), quant_dtype=self._qd)
        else:
            mq, msq = _q(x_bf16.contiguous(), quant_dtype=self._qd, scale_type=self._stp)
        return mq, msq.view(torch.uint8)

    def _run_stage2(self, s1, run_tokens, stream, slice_output):
        """stage-2 (fused MegaMoeStage2 or non-fused gemm2+combine) -> bf16 out."""
        if self.enable_fused_stage2:
            ret = self._g2.run(
                a2=s1.a2.view(-1), w2=self.w2,
                a2_scale=s1.a2_scale, w2_scale=self.w2_scale,
                sorted_token_ids=s1.sorted_token_ids,
                sorted_expert_ids=s1.sorted_expert_ids,
                sorted_weights=s1.sorted_weights,
                num_valid_ids=s1.num_valid_ids,
                wts_buf=s1.wts_buf,
                cur_tok=run_tokens, stream=stream,
                run_tokens=run_tokens,
            )
            out_tok = ret[0] if isinstance(ret, (tuple, list)) else ret
        else:
            out_tok = self._run_nonfused_stage2(s1, run_tokens)

        if out_tok is None:
            cfg = self.comb_cfg
            out_tok = (self.comb_op.shmem_comb_out_tok.view(torch.int8)[: self.mtpr * cfg.token_bytes]
                       .view(cfg.data_type).view(self.mtpr, cfg.token_view_dim))
        return out_tok[:run_tokens] if slice_output else out_tok

    def forward(self, x_bf16, wts, topk_ids, *, stream=None, slice_output: bool = True):
        """Full MoE layer: bf16 activation in -> stage1 (fused or non-fused) -> stage2 -> bf16 out.

        Fused stage-1 quantizes internally then runs the megakernel; non-fused stage-1 bf16-dispatches
        then recv-quantizes + moe_sorting + GEMM1.
        """
        run_tokens = int(x_bf16.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        if self._is_int8_smooth:
            # int8 smoothquant (a8w4smooth / w8a8smooth) is fused-only: front-smoothquant + per-slot
            # int8 dispatch + int8 gemm1 (MegaMoeStage1) -> recv requant -> int8 gemm2 + combine
            # (MegaMoeStage2). The packed4-vs-full-int8 weight datapath is baked into stage1/_g2 at
            # construction (b_dtype='int4' for a8w4smooth, 'int8' for w8a8smooth).
            return self._forward_smoothquant_fused(x_bf16, wts, topk_ids, stream=stream, slice_output=slice_output)
        wc = wts.contiguous()
        ic = topk_ids.to(torch.int32).contiguous()
        if self.enable_fused_stage1:
            x_q, scales = self.quantize(x_bf16[:run_tokens])
            s1 = self.stage1.forward(x_q, wc, scales, ic, stream=stream)
        else:
            s1 = self._run_nonfused_stage1(x_bf16[:run_tokens].contiguous(), wc, ic)
        return self._run_stage2(s1, run_tokens, stream, slice_output)

    # bf16 in, bf16 out; primary entry.
    forward_bf16 = forward

    def forward_prequant(self, x_q, scales, wts, topk_ids, *, stream=None, slice_output: bool = True):
        """Pre-quantized fast path (fp8/fp4 + e8m0 scale in). Requires fused stage-1 (which consumes
        x_q directly); for non-fused stage-1 use ``forward(x_bf16, ...)`` (bf16 dispatch)."""
        if not self.enable_fused_stage1:
            raise ValueError(
                "forward_prequant requires enable_fused_stage1=True (pre-quantized input feeds the "
                "fused megakernel); use forward(x_bf16, ...) for the non-fused (bf16-dispatch) stage-1.")
        run_tokens = int(x_q.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        wc = wts.contiguous()
        ic = topk_ids.to(torch.int32).contiguous()
        s1 = self.stage1.forward(x_q, wc, scales, ic, stream=stream)
        return self._run_stage2(s1, run_tokens, stream, slice_output)

    __call__ = forward
