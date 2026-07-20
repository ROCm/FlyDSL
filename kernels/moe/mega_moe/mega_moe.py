# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""End-to-end intra-node EP-MoE single operator ``MegaMoE`` (a full MoE layer).

``MegaMoE`` folds both stages in-house: fused stage-1 (dispatch + GEMM1 megakernel,
``_build_fused_stage1`` / ``_run_fused_stage1``) and fused stage-2 (GEMM2 + EP combine,
``_build_fused_stage2`` / ``_run_fused_stage2`` over ``compile_fused_moe_gemm2_combine``),
with non-fused fallbacks for each. The expert-major dispatch symmetric buffers are owned by
``FlyDSLDispatchCombineIntraNodeOp`` (``comb_op._gm``, enabled by ``enable_group_major``);
``total_recv`` is shared with combine (no bridge). Main entry
``MegaMoE.forward(x_bf16, wts, topk_ids)``: quantize -> stage1 -> stage2 -> bf16.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from typing import Optional

import mori.shmem as ms
import torch
import torch.distributed as dist

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.comm.flydsl_dispatch_combine_intranode_op import (
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
    _detect_gpu_model,
    resolve_tuning_config_path,
)

# activation MX quant + sorted-scale scatter: native FlyDSL kernels (both stage-1
# paths are aiter-free; these replace aiter per_1x32_mx_quant_hip / mxfp4_moe_sort_hip).
from ..mixed_moe_gemm_2stage import compile_mixed_moe_gemm1, compile_mixed_moe_gemm2
from ..moe_sorting_kernel import moe_sorting_flydsl, moe_sorting_get_workspace_size
from .gemm1 import GateMode, compile_fused_moe_gemm1
from .quant import mxfp4_moe_scale_sort
from .quant import per_1x32_mx_quant as _flydsl_mx_quant

__all__ = ["MegaMoE", "Stage1Output", "compile_fused_moe_gemm2_combine"]


def is_fp4(dt):
    return dt == torch.float4_e2m1fn_x2


@dataclass(frozen=True)
class Stage1Output:
    """Stage-1 -> stage-2 handoff (field names match ``_run_fused_stage2`` inputs).

    All tensors are stable buffers allocated once in ``_build_fused_stage1`` (fused) or
    ``_build_nonfused_stage1`` (non-fused); each stage-1 run overwrites them in place.
    """

    a2: torch.Tensor  # GEMM1 output (raw buffer; stage-2 views it)
    a2_scale: torch.Tensor  # e8m0 sorted scale
    sorted_token_ids: torch.Tensor
    sorted_expert_ids: torch.Tensor
    sorted_weights: torch.Tensor  # per-sorted-row routing weights [nvm]
    num_valid_ids: torch.Tensor
    wts_buf: torch.Tensor  # combine weight buffer f32[max_recv*k]


# ---- Tune-table helpers (megastage1 tiles + megagemm2 per-M tiles from the FlyDSL JSON) ----
def mega_default_tile(inter_dim):
    """Generic validated fallback tile used when a shape is intentionally untuned."""
    return 64, (256 if (int(inter_dim) % 256 == 0) else 128)


# megastage1 weights are always w4(fp4); the JSON ``dtype`` is the ACTIVATION quant.
MEGA_QUANT_TO_DTYPE = {"a4w4": "fp4", "a8w4": "fp8_ocp"}


def _shape_matches(r, model_dim, inter_dim, experts, topk, dtype=None):
    """True if tune-row ``r`` matches the GEMM shape (and ``dtype`` when given)."""
    try:
        if dtype is not None and r.get("dtype") != dtype:
            return False
        return (
            int(r["model_dim"]) == int(model_dim)
            and int(r["inter_dim"]) == int(inter_dim)
            and int(r["expert"]) == int(experts)
            and int(r["topk"]) == int(topk)
        )
    except (KeyError, ValueError, TypeError):
        return False


@functools.lru_cache(maxsize=16)
def _load_mega_rows(ep_size: int, gpu_model, kernel_tag: str, json_key: str):
    """Rows under ``json_key`` from the best-matching ``flydsl_*_{kernel_tag}_ep{n}.json``
    (shared file resolver ``resolve_tuning_config_path``; gpu_arch="" -> gpu_model-only match).
    Returns a hashable tuple of dicts, ``()`` on any miss. ``.cache_clear()`` after editing JSON."""
    path = resolve_tuning_config_path(int(ep_size), kernel_type=kernel_tag, gpu_arch="", gpu_model=gpu_model)
    if path is None:
        return ()
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return ()
    return tuple(raw.get(json_key, []))


def mega_tuned_tile(model_dim, inter_dim, experts, topk, quant, mtpr, ep_size, gpu_model):
    """megastage1 config from the tune JSON, keyed by GEMM shape. The token
    bucket rounds up to the smallest ``num_tokens >= mtpr`` (largest on
    overflow). Returns the tile dict or ``None`` on a miss."""
    dtype = MEGA_QUANT_TO_DTYPE.get(quant)
    if dtype is None:
        return None
    rows = _load_mega_rows(int(ep_size), gpu_model, "MegaStage1", "megastage1")
    if not rows:
        return None

    buckets = {
        int(r["num_tokens"]): r
        for r in rows
        if _shape_matches(r, model_dim, inter_dim, experts, topk, dtype)
    }
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
        # intra-CTA slice-K (default 1 = off); opt-in per bucket for small-M/large-K decode.
        slice_k=int(chosen.get("slice_k", 1)),
        # gemm1 B-load cache modifier (default 0 = cached); tuned per bucket (3 = stream, decode win).
        b_nt=int(chosen.get("b_nt", 0)),
    )


def mega_gemm2_tuned_table(model_dim, inter_dim, experts, topk, ep_size, gpu_model, quant=None):
    """Per-M fused gemm2 config table ``{num_tokens: {tile_m, tile_n, tile_k, [xcd_swizzle],
    [persist_m], [b_nt]}}`` from the tune JSON (``experts`` is the LOCAL per-rank count); ``None``
    on a miss. Optional per-row knobs override the ctor default only when present; ``quant``
    filters rows by activation dtype (None keeps the dtype-agnostic match)."""
    rows = _load_mega_rows(int(ep_size), gpu_model, "MegaGemm2", "megagemm2")
    if not rows:
        return None
    want_dtype = MEGA_QUANT_TO_DTYPE.get(quant) if quant is not None else None

    table = {}
    for r in rows:
        if _shape_matches(r, model_dim, inter_dim, experts, topk, want_dtype):
            try:
                entry = {"tile_m": int(r["tile_m"]), "tile_n": int(r["tile_n"]), "tile_k": int(r["tile_k"])}
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


# ---- stage-2 fused GEMM2+combine builder (thin wrapper over compile_mixed_moe_gemm2) ----
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

    The GEMM2 epilogue rewrites its local store_pair into a remote P2P buffer_store
    (combine Stage-1 scatter inlined); the host then runs a trimmed ``combine_no_stage1``.
    """
    # Plan B: sorted_token_ids' s field is used directly as j_global, so topk must equal k.
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

    # fused_cfg[3]=enable_weights pinned False (combine_no_stage1 does the weight P2P instead).
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
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        persist_m=persist_m,
        sort_block_m=sort_block_m,
        b_nt=b_nt,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=gemm2_out_dtype,
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


# ---- stage-1 config resolution: single-launch fused dispatch + GEMM1 ----
def resolve_stage1_config(
    *,
    model_dim,
    inter_dim,
    experts,
    topk,
    quant,
    max_tok_per_rank,
    world_size,
    rank,
    tune_tokens=None,
    unit_size=-1,
    tile_n=-1,
    tile_k=256,
    waves_per_eu=4,
    use_async_copy=True,
    slice_k=1,
    gate_mode=GateMode.SEPARATED,
):
    """Resolve the fused stage-1 GEMM tile + layout (compact) BEFORE the combine op
    allocates its group-major buffers (which need unit_size/compact). Returns a
    dict of resolved values consumed by both the combine op (gm_*) and stage-1."""
    assert quant in ("a4w4", "a8w4")
    if experts % world_size != 0:
        raise ValueError(f"experts={experts} must divide world_size={world_size}")
    epr = experts // world_size
    data_type = torch.float8_e4m3fn if quant == "a8w4" else torch.float4_e2m1fn_x2
    a_dtype = "fp8" if quant == "a8w4" else "fp4"
    scale_dim = model_dim // 32

    # compact when buffers cross ~4GB, or epr>64 (fixedslot decode is a <=64-lane readlane scan).
    row_b = int(model_dim) if quant == "a8w4" else (int(model_dim) // 2)
    max_buf = epr * world_size * int(max_tok_per_rank) * max(row_b, int(inter_dim) * 2)
    compact = (max_buf >= 3_000_000_000) or (epr > 64)

    tune_tokens = int(max_tok_per_rank if tune_tokens is None else tune_tokens)
    b_nt = 0  # gemm1 B-load cache modifier; overridden by the tuned bucket (untuned -> 0 = cached)
    tuned = None
    if int(unit_size) <= 0:
        tuned = mega_tuned_tile(
            model_dim, inter_dim, experts, topk, quant, tune_tokens, world_size, _detect_gpu_model(rank)
        )
    if tuned is not None:
        unit_size = tuned["tile_m"]
        if int(tile_n) <= 0:
            tile_n = tuned["tile_n"]
        tile_k = tuned["tile_k"]
        waves_per_eu = tuned["waves_per_eu"]
        use_async_copy = tuned["use_async_copy"]
        slice_k = tuned.get("slice_k", slice_k)
        b_nt = int(tuned.get("b_nt", 0))
    else:
        tm_t, tn_t = mega_default_tile(int(inter_dim))
        unit_size = int(unit_size) if int(unit_size) > 0 else tm_t
        tile_n = int(tile_n) if int(tile_n) > 0 else tn_t
    unit_size = int(unit_size)
    sort_block_m = max(32, unit_size)

    # INTERLEAVE halves tile_n in the epilog; bump to 256 when inter_dim allows (else ~15% slower).
    if gate_mode is GateMode.INTERLEAVE and int(tile_n) < 256 and int(inter_dim) % 256 == 0:
        tile_n = 256

    # XCD swizzle guard: disable unless the gx N-tile grid divides cleanly into the 8 XCDs.
    xcd = 4
    if xcd > 0:
        cu_g = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        if gate_mode is GateMode.INTERLEAVE:
            gx_g = max(1, (inter_dim * 2 + int(tile_n) - 1) // int(tile_n))
        else:
            gx_g = max(1, (inter_dim * 2 + 2 * int(tile_n) - 1) // int(tile_n) // 2)
        gx_xcd_ok = (cu_g % gx_g == 0) and (8 % gx_g == 0 or gx_g % 8 == 0)
        if not gx_xcd_ok:
            import warnings

            warnings.warn(f"XCD swizzle DISABLED: gx={gx_g} is not a divisor of 8 (or cu=" f"{cu_g} % gx != 0).")
            xcd = 0

    return dict(
        epr=epr,
        data_type=data_type,
        a_dtype=a_dtype,
        scale_dim=scale_dim,
        compact=compact,
        unit_size=unit_size,
        tile_n=int(tile_n),
        tile_k=int(tile_k),
        waves_per_eu=int(waves_per_eu),
        use_async_copy=bool(use_async_copy),
        slice_k=int(slice_k),
        sort_block_m=sort_block_m,
        xcd=xcd,
        tune_tokens=tune_tokens,
        b_nt=b_nt,  # from the tuned bucket (see MegaStage1 tune JSON "b_nt"); 0 when untuned
    )


def _next_pow2(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# ---- End-to-end EP-MoE single operator ----
class MegaMoE:
    """End-to-end EP-MoE single operator (a full MoE layer).

    ``enable_fused_stage1`` / ``enable_fused_stage2`` pick fused vs non-fused per stage.
    Supported combos: (fused, fused), (non-fused, non-fused), (non-fused, fused); the
    (fused stage-1, non-fused stage-2) combo is rejected at init (see the ValueError below).

    Both fused stages are folded in as methods: fused stage-1 (``_build_fused_stage1`` +
    ``_run_fused_stage1``, ``_s1_*`` state) is the single-launch dispatch+GEMM1 megakernel
    over the group-major buffers in ``comb_op._gm``; fused stage-2 (``_build_fused_stage2`` +
    ``_run_fused_stage2``, ``_g2_*`` state) is the GEMM2 epilogue P2P-scatter + trimmed
    ``combine_no_stage1``.
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
        tune_tokens: Optional[int] = None,
        mega_scheme: str = "fixedslot",
        tile_m: int = -1,
        tile_n: int = -1,
        tile_k: int = 256,
        gemm2_tile_m: int = -1,
        gemm2_tile_n: int = -1,
        gemm2_tile_k: int = -1,
        gemm2_persist_m: int = -1,
        waves_per_eu: int = 4,
        use_async_copy: bool = True,
        slice_k: int = 1,
        xcd_swizzle: int = 0,
        gemm2_tile_table: Optional[dict] = None,
        enable_fused_stage1: bool = True,
        enable_fused_stage2: bool = True,
        gate_mode=None,
    ):
        if quant not in ("a8w4", "a4w4"):
            raise ValueError(f"MegaMoE quant must be 'a8w4' or 'a4w4', got {quant!r}")
        if (max_tok_per_rank & (max_tok_per_rank - 1)) != 0:
            raise ValueError(f"max_tok_per_rank={max_tok_per_rank} must be a power of two")
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
        # gate-up mode default: INTERLEAVE(g1u1) for a8w4, SEPARATED for a4w4 (the caller's
        # w1/w1_scale shuffle MUST match; INTERLEAVE -> shuffle_weight_w4/shuffle_scale_w4 gate_up=True).
        self.gate_mode = (
            (GateMode.INTERLEAVE if quant == "a8w4" else GateMode.SEPARATED)
            if gate_mode is None
            else GateMode(gate_mode)
        )
        # Fused stage-1 only sets up the combine_no_stage1 handoff (identity tok_id_to_src +
        # total_recv), not the dispatch bookkeeping the standard combine needs -> reject fused/non-fused.
        if self.enable_fused_stage1 and not self.enable_fused_stage2:
            raise ValueError(
                "enable_fused_stage1=True with enable_fused_stage2=False is unsupported: the fused "
                "stage-1 sets up only the combine_no_stage1 handoff (identity tok_id_to_src + "
                "total_recv), not the dispatch bookkeeping the standard combine needs for a non-fused "
                "stage-2. Supported combos: (fused,fused), (non-fused,non-fused), (non-fused,fused). "
                "Set enable_fused_stage1=False for a non-fused stage-2."
            )
        self.dev = torch.device("cuda", rank)
        self._is_fp4 = quant == "a4w4"

        # Resolve stage-1 tile / layout (tiles used by both fused & non-fused gemm1; compact sizing).
        self._s1cfg = resolve_stage1_config(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            quant=quant,
            max_tok_per_rank=max_tok_per_rank,
            world_size=world_size,
            rank=rank,
            tune_tokens=tune_tokens,
            unit_size=int(tile_m),
            tile_n=int(tile_n),
            tile_k=int(tile_k),
            waves_per_eu=int(waves_per_eu),
            use_async_copy=bool(use_async_copy),
            slice_k=int(slice_k),
            gate_mode=self.gate_mode,
        )
        self.a2_dtype = self._s1cfg["a_dtype"]  # "fp8" | "fp4"
        self.mega_scheme = mega_scheme

        # Combine op (bf16). dispatch_dtype/scale reflect the active path: fused stage-1 uses the
        # group-major quantized dispatch (fp8/fp4 + scales); non-fused uses token-major bf16 dispatch.
        gm_on = self.enable_fused_stage1
        self.comb_cfg = FlyDSLDispatchCombineConfig(
            rank=rank,
            world_size=world_size,
            hidden_dim=model_dim,
            max_num_inp_token_per_rank=self.mtpr,
            num_experts_per_rank=self.epr,
            num_experts_per_token=topk,
            combine_dtype=torch.bfloat16,
            dispatch_dtype=(self._s1cfg["data_type"] if gm_on else torch.bfloat16),
            scale_dim=(self._s1cfg["scale_dim"] if gm_on else 0),
            scale_type_size=(1 if gm_on else 0),
            enable_std_moe=False,
            enable_group_major=gm_on,
            gm_unit_size=self._s1cfg["unit_size"],
            gm_scheme=mega_scheme,
            gm_compact=self._s1cfg["compact"],
        )
        self.comb_op = FlyDSLDispatchCombineIntraNodeOp(self.comb_cfg)
        torch.cuda.synchronize()
        ms.shmem_barrier_all()

        self.w2 = w2 if w2.is_contiguous() else w2.contiguous()
        self.w2_scale = w2_scale if w2_scale.is_contiguous() else w2_scale.contiguous()
        self.max_recv = world_size * self.mtpr

        # ---- stage-1 ----
        if self.enable_fused_stage1:
            # Fused dispatch + GEMM1 megakernel (uses comb_op._gm buffers); sets self.sort_block_m.
            self._build_fused_stage1(w1, w1_scale)
            # megav1 _sti encodes src_global=dest -> combine tok_id_to_src is identity (const).
            self.comb_op.shmem_tok_id_to_src.copy_(torch.arange(self.max_recv, device=self.dev, dtype=torch.int32))
            torch.cuda.synchronize()
            ms.shmem_barrier_all()
        else:
            # Non-fused stage-1: bf16 dispatch (comb_op) + FlyDSL moe_sorting + compile_mixed_moe_gemm1.
            self._build_nonfused_stage1(w1, w1_scale)
            torch.cuda.synchronize()
            ms.shmem_barrier_all()

        # ---- stage-2 ----
        s1_sbm = int(self.sort_block_m)
        if self.enable_fused_stage2:
            # fused stage-1: pin the GEMM2 gate to constant max_recv (identity routing).
            if self.enable_fused_stage1:
                self._gemm2_gate_bound = torch.tensor([self.max_recv], dtype=torch.int32, device=self.dev)
                self.comb_op._fx_out_total_recv = fx.Int64(self._gemm2_gate_bound.data_ptr())
                torch.cuda.synchronize()
                ms.shmem_barrier_all()
            eff_g2_tm = int(gemm2_tile_m) if int(gemm2_tile_m) > 0 else 32  # op default tile_m=32
            if s1_sbm % eff_g2_tm != 0:
                raise ValueError(
                    f"gemm2 tile_m={eff_g2_tm} does not divide stage1 sort_block_m={s1_sbm}; "
                    f"set gemm2_tile_m to a divisor of {s1_sbm} (e.g. 32)."
                )
            if gemm2_tile_table is None:
                try:
                    gemm2_tile_table = mega_gemm2_tuned_table(
                        int(model_dim),
                        int(inter_dim),
                        int(self.experts) // int(self.world_size),
                        int(topk),
                        int(self.world_size),
                        _detect_gpu_model(int(self.rank)),
                        quant=self.quant,
                    )
                    if int(self.rank) == 0:
                        print(
                            f"[MegaGemm2] auto-config tile table (sort_block_m={s1_sbm}): "
                            f"{'<miss -> default tile>' if not gemm2_tile_table else gemm2_tile_table}",
                            flush=True,
                        )
                except Exception:  # noqa: BLE001 -- fallback to default tile
                    gemm2_tile_table = None
            # tile_m/n/k only override the table when gemm2_tile_m>0; persist_m/xcd forwarded always.
            g2_kwargs = dict(
                a_dtype=self.a2_dtype,
                b_dtype="fp4",
                sort_block_m=s1_sbm,
                gemm2_tile_table=gemm2_tile_table,
                xcd_swizzle=int(xcd_swizzle),
            )
            if int(gemm2_persist_m) > 0:
                g2_kwargs["persist_m"] = int(gemm2_persist_m)
            if int(gemm2_tile_m) > 0:
                g2_kwargs.update(tile_m=int(gemm2_tile_m), tile_n=int(gemm2_tile_n), tile_k=int(gemm2_tile_k))
            self._build_fused_stage2(**g2_kwargs)
        else:
            # Non-fused stage-2: standalone compile_mixed_moe_gemm2 + comb_op.combine.
            self._build_nonfused_stage2(int(gemm2_tile_m), int(gemm2_tile_n), int(gemm2_tile_k), int(gemm2_persist_m))

    # ------------------------------------------------------------------ fused stage-1
    def _build_fused_stage1(self, w1, w1_scale):
        """Single-launch fused dispatch + GEMM1 megakernel build (fixedslot decode).

        The expert-major dispatch buffers live in ``comb_op._gm``; this compiles the fused
        kernel, builds the dispatch-arg pointer table over those buffers, and allocates the
        stable ``Stage1Output`` buffers. ``self.sort_block_m`` is set here.
        """
        cfg = self._s1cfg
        assert self.mega_scheme == "fixedslot", (
            f"scheme={self.mega_scheme!r}: only 'fixedslot' supported (handshake removed)"
        )
        out_dtype = self.a2_dtype  # out_dtype "auto" -> activation a_dtype
        self._s1_data_type = cfg["data_type"]
        self._s1_scale_dim = cfg["scale_dim"]
        self._s1_compact = cfg["compact"]
        self._s1_unit_size = int(cfg["unit_size"])
        self.sort_block_m = cfg["sort_block_m"]
        self._s1_tile_n = int(cfg["tile_n"])
        tile_k = int(cfg["tile_k"])
        waves_per_eu = cfg["waves_per_eu"]
        use_async_copy = cfg["use_async_copy"]
        self._s1_slice_k = int(cfg.get("slice_k", 1))
        self._s1_xcd = cfg["xcd"]
        self._s1_b_nt = int(cfg.get("b_nt", 0))
        self._s1_tune_tokens = int(cfg["tune_tokens"])

        # w1/w1_scale are this rank's `epr` expert rows ONLY (ATOM local convention).
        self._s1_w1 = w1.contiguous()
        self._s1_w1_scale = w1_scale.contiguous()

        # group-major dispatch buffers owned by the combine op.
        op = self.comb_op._gm
        assert op is not None, "combine op was built without enable_group_major"
        self._s1_op = op
        self._s1_nvm = op.num_valid_max
        self._s1_max_blocks = op.max_blocks
        self._s1_cap = op.ll_cap
        # metadata-ready flag (monotonic, NEVER reset -> CUDAGraph-safe).
        self._s1_meta = torch.zeros(1, dtype=torch.int32, device=self.dev)
        # total_recv is UNIFIED with combine (op.total_recv aliases comb_op.total_recv).
        self._s1_trecv_buf = None

        # Compile the megakernel (serialize across ranks to bound peak memory).
        self._s1_mega = None
        for pe in range(self.world_size):
            if self.rank == pe:
                self._s1_mega = compile_fused_moe_gemm1(
                    model_dim=self.model_dim,
                    inter_dim=self.inter_dim,
                    experts=self.experts,
                    topk=1,
                    tile_m=self._s1_unit_size,
                    tile_n=self._s1_tile_n,
                    tile_k=tile_k,
                    a_dtype=self.a2_dtype,
                    b_dtype="fp4",
                    out_dtype=out_dtype,
                    gate_mode=self.gate_mode,
                    waves_per_eu=waves_per_eu,
                    use_async_copy=use_async_copy,
                    b_nt=self._s1_b_nt,
                    slice_k=self._s1_slice_k,
                    xcd_swizzle=self._s1_xcd,
                    fuse_dispatch=self.mega_scheme,
                    fuse_npes=self.world_size,
                    fuse_topk=self.topk,
                    fuse_cap=self._s1_cap,
                    fuse_mtpr=self.mtpr,
                    fuse_scale_dim=self._s1_scale_dim,
                    fuse_scale_type_size=1,
                    rank=self.rank,
                    experts_per_rank=self.epr,
                    compact_dispatch=self._s1_compact,
                )
            if dist.is_initialized():
                dist.barrier()

        # Views into op buffers (A-input + raw A-scale + metadata the GEMM consumes).
        v = op._ll_views()
        self._s1_rx = v["rx_em"]
        self._s1_scale_i32 = v["scale_em_i32"]
        self._s1_trb = op.tile_row_base
        self._s1_se = op.sorted_expert_ids
        self._s1_nv = op.num_valid

        # Output / scratch.
        self._s1_out_dtype = out_dtype
        nf4 = out_dtype == "fp4"
        nf8 = out_dtype == "fp8"
        self._s1_out_is_quant = nf4 or nf8
        inter_dim = self.inter_dim
        self._s1_a2rows = self.world_size * self.mtpr * self.topk
        if nf4:
            self._s1_out = torch.zeros((self._s1_a2rows, inter_dim // 2), dtype=torch.uint8, device=self.dev)
        elif nf8:
            self._s1_out = torch.zeros((self._s1_a2rows, inter_dim), dtype=torch.float8_e4m3fn, device=self.dev)
        else:
            self._s1_out = torch.zeros((self._s1_a2rows, 1, inter_dim), dtype=torch.float16, device=self.dev)
        self._s1_bias = torch.empty((0,), dtype=torch.float32, device=self.dev)
        # COMPACT sorted_token_ids[compact_row] = (k<<24)|src_global into a [nvm] buffer.
        self._s1_sti = torch.zeros(self._s1_nvm, dtype=torch.int32, device=self.dev)
        # sorted_expert_ids at 32-row SUB-TILE granularity -> nvm/32 entries.
        self._s1_se_atom = torch.zeros(self._s1_nvm // 32 + 8, dtype=torch.int32, device=self.dev)
        # combine wts_buf (f32, LOGICAL layout wts[t*topk+s], t=src_global via identity tis).
        self._s1_sw_atom = torch.zeros(self.world_size * self.mtpr * self.topk, dtype=torch.float32, device=self.dev)
        # sorted-row routing weights (f32, [nvm], parallel to _s1_sti).
        self._s1_wts_sorted = torch.zeros(self._s1_nvm, dtype=torch.float32, device=self.dev)
        # e8m0 sorted-scale buffer (a2_scale).
        if self._s1_out_is_quant:
            sbm = max(32, self._s1_unit_size)
            prows = ((self._s1_max_blocks * sbm + 255) // 256) * 256
            pcols = (((inter_dim // 32) + 7) // 8) * 8
            self._s1_osd = torch.zeros(prows * pcols + inter_dim, dtype=torch.uint8, device=self.dev)
        else:
            self._s1_osd = torch.empty((0,), dtype=torch.uint8, device=self.dev)

        # dispatch-arg FIXED-pointer table (op bufs + p2p); built ONCE.
        self._build_disp_table()

        # stage-2 handoff contract (stable buffers; run overwrites them in place).
        self._s1_output = Stage1Output(
            a2=self._s1_out,
            a2_scale=self._s1_osd,
            sorted_token_ids=self._s1_sti,
            sorted_expert_ids=self._s1_se_atom,
            sorted_weights=self._s1_wts_sorted,
            num_valid_ids=self._s1_nv,
            wts_buf=self._s1_sw_atom,
        )

    def _build_disp_table(self):
        op = self._s1_op
        tbl = [
            0,
            0,
            0,
            0,  # 0-3 inputs (lazy)
            op.gb1.data_ptr(),
            op.running.data_ptr(),
            op.done2.data_ptr(),
            op.ll_count.data_ptr(),  # 4-7
            op.p2p_rx_em.data_ptr(),
            op.p2p_scale_em.data_ptr(),
            op.p2p_idx_em.data_ptr(),
            op.p2p_wts_em.data_ptr(),
            op.p2p_srcmap_em.data_ptr(),
            op.p2p_running.data_ptr(),
            op.p2p_done2.data_ptr(),  # 8-14
            op.sorted_expert_ids.data_ptr(),
            op.tile_row_base.data_ptr(),
            op.num_valid.data_ptr(),  # 15-17
            self._s1_meta.data_ptr(),
        ]  # 18 metadata-ready flag
        if self._s1_compact:
            # 19-35: compact count-first extras
            tbl += [
                op.compact_base.data_ptr(),
                op.p2p_compact_base.data_ptr(),  # 19-20
                op.gb_cnt.data_ptr(),
                op.done2c.data_ptr(),
                op.p2p_done2c.data_ptr(),  # 21-23
                op.meta2.data_ptr(),  # 24 payload-ready flag
                op.write_cursor.data_ptr(),
                op.p2p_write_cursor.data_ptr(),  # 25-26
                op.done2cb.data_ptr(),
                op.p2p_done2cb.data_ptr(),  # 27-28 cross-PE#1b
                op.local_hist.data_ptr(),
                op.bigcnt.data_ptr(),
                op.p2p_bigcnt.data_ptr(),  # 29-31
                op.cnt_done.data_ptr(),
                op.p2p_cnt_done.data_ptr(),  # 32-33
                op.my_base.data_ptr(),
                op.local_cursor.data_ptr(),
            ]  # 34-35
            tbl[19] = op.srcmap_em.data_ptr()  # 19 LOCAL srcmap (override compact_base)
            tbl[20] = self._s1_sw_atom.data_ptr()  # 20 compact sorted_weights out
            tbl += [
                (self._s1_trecv_buf if self._s1_trecv_buf is not None else op.total_recv).data_ptr(),
                op.dest_ctr.data_ptr(),  # 36 total_recv | 37 dest_ctr(local)
                op.recv_num.data_ptr(),
                op.p2p_recv_num.data_ptr(),  # 38 recv_num | 39 p2p_recv_num
                self._s1_sti.data_ptr(),
                self._s1_se_atom.data_ptr(),  # 40 _sti | 41 _se_atom
                self._s1_wts_sorted.data_ptr(),
            ]  # 42 sorted-row weights out
        else:
            tbl += [
                op.srcmap_em.data_ptr(),  # 19 LOCAL srcmap (k_slot<<24|src_global)
                self._s1_sw_atom.data_ptr(),  # 20 compact sorted_weights out
                (self._s1_trecv_buf if self._s1_trecv_buf is not None else op.total_recv).data_ptr(),
                op.dest_ctr.data_ptr(),  # 21 total_recv | 22 dest_ctr(local)
                op.recv_num.data_ptr(),
                op.p2p_recv_num.data_ptr(),  # 23 recv_num(local) | 24 p2p_recv_num
                self._s1_wts_sorted.data_ptr(),
            ]  # 25 sorted-row weights out
        self._s1_disp = torch.tensor(tbl, dtype=torch.int64, device=self.dev)
        self._s1_disp_host = torch.tensor(tbl, dtype=torch.int64)

    def _s1_agv(self, t):
        return t.view(torch.uint8) if self.a2_dtype == "fp4" else t

    def _run_fused_stage1(self, x, wts, scales, topk_ids, stream=None) -> "Stage1Output":
        """Run the WHOLE fused stage-1 (dispatch + group-GEMM) in ONE launch.

        Returns the stable ``Stage1Output`` (its buffers are overwritten here).
        """
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream())
        cur_tok = int(x.shape[0])
        assert cur_tok <= self.mtpr, f"[mega] cur_tok={cur_tok} > max_tok_per_rank={self.mtpr}"
        xc = x.contiguous()
        wc = wts.contiguous()
        ic = topk_ids.to(torch.int32).contiguous()
        sc = scales.contiguous()
        op = self._s1_op

        pd_ptr = 0
        # total_recv accumulates in-kernel each launch -> zero first (graph-safe).
        (self._s1_trecv_buf if self._s1_trecv_buf is not None else op.total_recv).zero_()
        er_ptr = op.ll_count.data_ptr() if (self.mega_scheme == "fixedslot" and not self._s1_compact) else 0
        if self._s1_compact:
            op.local_hist.zero_()
            op.local_cursor.zero_()
        a_mat = self._s1_agv(self._s1_rx)
        if self._s1_compact:
            sorted_arg = self._s1_trb
            se_arg = self._s1_se
        else:
            sorted_arg = self._s1_sti
            se_arg = self._s1_se_atom
        wt_arg = op.wts_em
        self._s1_mega(
            self._s1_out,
            a_mat,
            self._s1_w1,
            self._s1_scale_i32,
            self._s1_w1_scale,
            sorted_arg,
            se_arg,
            wt_arg,
            self._s1_nv,
            self._s1_bias,
            self._s1_osd,
            fx.Int32(self._s1_nvm),
            fx.Int32(self.inter_dim * 2),
            fx.Int32(self.model_dim),
            fx.Int32(self._s1_max_blocks),
            fx.Int64(pd_ptr),
            fx.Int64(er_ptr),
            fx.Int64(self._s1_disp.data_ptr()),
            fx.Int32(cur_tok),
            fx.Int64(xc.data_ptr()),
            fx.Int64(ic.data_ptr()),
            fx.Int64(wc.data_ptr()),
            fx.Int64(sc.data_ptr()),
            stream=stream,
        )
        return self._s1_output

    # ------------------------------------------------------------------ fused stage-2
    def _build_fused_stage2(
        self,
        *,
        tile_m: int = 32,
        tile_n: int = 128,
        tile_k: int = 128,
        persist_m: int = -1,  # <=0 -> persistent mode (grid_y=cu_num), the default; >0 = legacy N-tile
        sort_block_m: int = 0,
        b_nt: int = 2,
        a_dtype: str = "fp8",
        b_dtype: str = "fp4",
        xcd_swizzle: int = 0,
        use_token_flag_sync: bool = False,
        doweight_fused: bool = True,
        gemm2_tile_table: dict | None = None,
    ):
        """Fused MoE-GEMM2 + EP-combine build. The GEMM2 epilogue inlines combine Stage-1's
        P2P token scatter; the run then launches a trimmed combine (``combine_no_stage1``)."""
        # audited consumer -> opt into the default-off combine_no_stage1 gate.
        FlyDSLDispatchCombineIntraNodeOp._ENABLE_COMBINE_NO_STAGE1 = True
        self._g2_tile_m = tile_m
        self._g2_tile_n = tile_n
        self._g2_tile_k = tile_k
        self._g2_persist_m = persist_m
        self._g2_sort_block_m = sort_block_m
        self._g2_b_nt = b_nt
        self._g2_a_dtype = a_dtype
        self._g2_b_dtype = b_dtype
        self._g2_xcd_swizzle = xcd_swizzle
        self._g2_use_token_flag_sync = bool(use_token_flag_sync)
        # weight the GEMM2 epilogue (combine_no_stage1 reduces unweighted) -> on by default.
        self._g2_doweight_fused = bool(doweight_fused)

        comb_cfg = self.comb_cfg
        # fp8_direct_cast: GEMM2 epilogue casts + scatters fp8 inline (skips input.to(fp8) copy).
        self._g2_fp8_cast = (
            getattr(comb_cfg, "combine_quant_type", "none") == "fp8_direct_cast"
            and comb_cfg.combine_dtype == torch.bfloat16
        )
        if self._g2_fp8_cast:
            self._g2_out_dtype_str = "fp8e4m3fn"
        else:
            self._g2_out_dtype_str = "bf16" if comb_cfg.combine_dtype == torch.bfloat16 else "f16"

        # per-config JIT caches (tile config is a compile-time constant; pick by run_tokens at run()).
        self._g2_launch_by_tile = {}
        self._g2_compiled_by_tile = {}
        # optional {pow2_token: {tile_m,tile_n,tile_k,...}}; legacy 3-tuple normalised to dict.
        if gemm2_tile_table:
            self._g2_tile_table = {}
            for tok, val in dict(gemm2_tile_table).items():
                if isinstance(val, dict):
                    self._g2_tile_table[int(tok)] = dict(val)
                else:  # legacy (tile_m, tile_n, tile_k)
                    tm, tn, tk = val
                    self._g2_tile_table[int(tok)] = {"tile_m": int(tm), "tile_n": int(tn), "tile_k": int(tk)}
        else:
            self._g2_tile_table = None
        # per-M table active -> pin sort_block_m=32 to match stage1's fixed 32-row block.
        if self._g2_tile_table is not None and int(sort_block_m) <= 0:
            self._g2_sort_block_m = 32
        self._g2_alloc_dummy_tensors()

    def _g2_clamp_tile_m(self, tm):
        """Halve tile_m to the nearest power-of-two divisor of sort_block_m (floored at 16);
        tile_m must divide sort_block_m or gemm2 reads the wrong expert's W2. No-op if sbm<=0."""
        sbm = int(self._g2_sort_block_m)
        if sbm <= 0:
            return int(tm)
        tm = int(tm)
        if tm > sbm:
            tm = sbm
        while tm > 16 and (sbm % tm) != 0:
            tm //= 2
        return tm

    def _g2_select_tile(self, run_tokens):
        """Pick ``(tile_m, tile_n, tile_k, persist_m, b_nt, xcd_swizzle)`` for run_tokens via the
        host-static table (nextPow2 bucket, clamped to range); ctor defaults when no table.
        tile_m is always clamped to a divisor of sort_block_m."""
        if not self._g2_tile_table or run_tokens is None:
            return (
                self._g2_clamp_tile_m(self._g2_tile_m),
                self._g2_tile_n,
                self._g2_tile_k,
                self._g2_persist_m,
                self._g2_b_nt,
                self._g2_xcd_swizzle,
            )
        keys = sorted(self._g2_tile_table)
        b = _next_pow2(run_tokens)
        if b < keys[0]:
            b = keys[0]
        elif b > keys[-1]:
            b = keys[-1]
        if b not in self._g2_tile_table:
            ge = [k for k in keys if k >= b]
            b = ge[0] if ge else keys[-1]
        row = self._g2_tile_table[b]
        return (
            self._g2_clamp_tile_m(row["tile_m"]),
            int(row["tile_n"]),
            int(row["tile_k"]),
            int(row.get("persist_m", self._g2_persist_m)),
            int(row.get("b_nt", self._g2_b_nt)),
            int(row.get("xcd_swizzle", self._g2_xcd_swizzle)),
        )

    def _g2_alloc_dummy_tensors(self):
        """Pre-allocate placeholders at ctor time (torch.zeros inside a cudagraph capture errors)."""
        comb_cfg = self.comb_cfg
        dev = torch.device("cuda", comb_cfg.rank)
        max_recv = comb_cfg.world_size * comb_cfg.max_num_inp_token_per_rank

        # arg_out is unused on the fused path but the launcher signature needs it.
        self._g2_dummy_out = torch.zeros(1, dtype=comb_cfg.combine_dtype, device=dev)
        self._g2_dummy_bias = torch.empty(0, dtype=comb_cfg.combine_dtype, device=dev)

        # Under fp8_cast, allocate the combine_no_stage1 input in fp8 to skip its bf16->fp8 copy.
        inp_dtype = torch.float8_e4m3fn if self._g2_fp8_cast else comb_cfg.combine_dtype
        self._g2_dummy_inp = torch.zeros(max_recv, comb_cfg.hidden_dim, dtype=inp_dtype, device=dev)

    def _g2_ensure_launch_fn(self, cfg):
        """Build+cache the launch_fn for ``cfg``, keyed by the
        (tile_m, tile_n, tile_k, persist_m, b_nt, xcd_swizzle) compile-time constants."""
        cfg = tuple(int(v) for v in cfg)
        tile_m, tile_n, tile_k, persist_m, b_nt, xcd_swizzle = cfg
        # tile_m must divide the sort block stage1 emitted sorted_expert_ids at (else garbage).
        if int(self._g2_sort_block_m) > 0 and (int(self._g2_sort_block_m) % tile_m) != 0:
            raise ValueError(
                f"MegaMoE gemm2 tile_m={tile_m} must divide "
                f"sort_block_m={self._g2_sort_block_m} (stage1 sorted_expert_ids block). "
                f"Pick tile_m in divisors of {self._g2_sort_block_m}."
            )
        lf = self._g2_launch_by_tile.get(cfg)
        if lf is not None:
            return lf
        comb_cfg = self.comb_cfg
        # Plan B writes into shmem_comb_inp_tok's [0, mt*k) sub-range; the combine op sizes it to
        # max(npes, topk)*mtpr so any topk stays in-bounds.
        k = comb_cfg.num_experts_per_token
        lf = compile_fused_moe_gemm2_combine(
            model_dim=comb_cfg.hidden_dim,
            inter_dim=self.inter_dim,
            experts=comb_cfg.num_experts_per_rank,
            topk=k,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            persist_m=persist_m,
            sort_block_m=self._g2_sort_block_m,
            b_nt=b_nt,
            a_dtype=self._g2_a_dtype,
            b_dtype=self._g2_b_dtype,
            out_dtype=self._g2_out_dtype_str,
            rank=comb_cfg.rank,
            npes=comb_cfg.world_size,
            max_tok_per_rank=comb_cfg.max_num_inp_token_per_rank,
            experts_per_token=k,
            xcd_swizzle=xcd_swizzle,
            use_token_flag_sync=self._g2_use_token_flag_sync,
            doweight_fused=self._g2_doweight_fused,
        )
        self._g2_launch_by_tile[cfg] = lf
        return lf

    def _run_fused_stage2(self, s1, run_tokens, stream=None):
        """fused GEMM2 (epilogue P2P scatter) + combine_no_stage1 (Stage 2/3) -> (out_tok, out_wts).

        ``run_tokens`` is this rank's local token count; keys the per-M tile config (host-static).
        """
        cfg = self._g2_select_tile(run_tokens)
        launch_fn = self._g2_ensure_launch_fn(cfg)

        comb_cfg = self.comb_cfg
        comb_op = self.comb_op
        if stream is None:
            stream = torch.cuda.current_stream()
        s_fx = fx.Stream(stream.cuda_stream)

        a2 = s1.a2.view(-1)
        a2_scale = s1.a2_scale
        sorted_token_ids = s1.sorted_token_ids
        sorted_expert_ids = s1.sorted_expert_ids
        sorted_weights = s1.sorted_weights
        num_valid_ids = s1.num_valid_ids

        # tokens_in MUST be max_recv (drives the epilogue row_valid early-exit; a2.shape[0] would OOB).
        tokens_in = comb_cfg.world_size * comb_cfg.max_num_inp_token_per_rank
        n_in = comb_cfg.hidden_dim
        k_in = self.inter_dim
        size_expert_ids = sorted_expert_ids.numel()

        # No upfront zero_() on shmem_comb_inp_tok (races peers' P2P; every slot is fully overwritten).
        # FLYDSL_FUSED_ZERO_MODE=normal force-enables it for debugging.
        if os.environ.get("FLYDSL_FUSED_ZERO_MODE", "").strip().lower() == "normal":
            mt_k_bytes = (
                comb_cfg.max_num_inp_token_per_rank * comb_cfg.num_experts_per_token * comb_cfg.combine_token_bytes
            )
            mt_k_i16 = (mt_k_bytes + 1) // 2
            comb_op.shmem_comb_inp_tok[:mt_k_i16].zero_()

        # Weight source: shmem_disp_out_wts (f32[max_recv*k], wts[t*k+s]) unless s1.wts_buf overrides.
        wts_buf = s1.wts_buf
        addr_wts_buf = fx.Int64(wts_buf.data_ptr()) if wts_buf is not None else comb_op._fx_disp_out_wts

        # First call passes fx-typed scalars (flyc infers the signature); cached calls use raw ints.
        common_args = (
            self._g2_dummy_out,
            a2,
            self.w2,
            a2_scale,
            self.w2_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids,
            self._g2_dummy_bias,
            comb_op._fx_tis,
            comb_op._fx_p2p_comb_inp,
            addr_wts_buf,
            comb_op._fx_p2p_comb_inp_wts,
            comb_op._fx_local_counter,
            comb_op._fx_p2p_comb_flag,
            comb_op._fx_out_total_recv,
        )
        compiled = self._g2_compiled_by_tile.get(cfg)
        if compiled is None:
            compiled = flyc.compile(
                launch_fn,
                *common_args,
                fx.Int32(tokens_in),
                fx.Int32(n_in),
                fx.Int32(k_in),
                fx.Int32(size_expert_ids),
                s_fx,
            )
            self._g2_compiled_by_tile[cfg] = compiled
        else:
            compiled(
                *common_args,
                tokens_in,
                n_in,
                k_in,
                size_expert_ids,
                s_fx,
            )

        # Weights still go through combine_no_stage1 (fused only removed the token P2P).
        return comb_op.combine_no_stage1(
            self._g2_dummy_inp,
            None,
            None,
            cur_tok=run_tokens,
            enable_weights=True,
        )

    # ------------------------------------------------------------------ non-fused stage-1
    def _build_nonfused_stage1(self, w1, w1_scale):
        """Buffers + gemm1 for the non-fused stage-1 (bf16 dispatch + FlyDSL sort)."""
        dev, epr, topk = self.dev, self.epr, self.topk
        model_dim, inter_dim, experts = self.model_dim, self.inter_dim, self.experts
        max_recv = self.max_recv
        # INTERLEAVE (g1u1) halves tile_n in the epilog -> use tn256 for a full 128-wide epilog.
        tm, tk = 32, 256
        tn1 = 256 if (self.gate_mode is GateMode.INTERLEAVE and inter_dim % 256 == 0) else 128
        self._nf_tm = tm
        self.sort_block_m = max(32, tm)
        max_pad = max_recv * topk + experts * tm
        max_blocks = (max_pad + tm - 1) // tm
        self._nf_max_blocks = max_blocks
        scaleN_pad = ((model_dim // 32 + 7) // 8) * 8
        self._nf_a_st = torch.empty(max_pad, dtype=torch.int32, device=dev)
        self._nf_a_sw = torch.empty(max_pad, dtype=torch.float32, device=dev)
        self._nf_a_se = torch.empty(max_blocks, dtype=torch.int32, device=dev)
        self._nf_a_nv = torch.zeros(2, dtype=torch.int32, device=dev)
        self._nf_a_mbuf = torch.empty((max_recv, model_dim), dtype=torch.float16, device=dev)
        self._nf_a1s = torch.empty(((max_pad + 31) // 32 * 32, scaleN_pad), dtype=torch.float8_e8m0fnu, device=dev)
        self._nf_recv_topk = torch.empty((max_recv, topk), dtype=torch.int32, device=dev)
        # EP mask: enable only this rank's local experts. The FlyDSL sort then
        # emits sorted_expert_ids compacted to local index [0, epr) directly.
        self._nf_expert_mask = torch.zeros(experts, dtype=torch.int32, device=dev)
        self._nf_expert_mask[self.rank * epr : (self.rank + 1) * epr] = 1
        # Pre-allocated multiphase sort workspace (keeps forward CUDAGraph-safe).
        self._nf_sort_ws = torch.empty(
            max(1, moe_sorting_get_workspace_size(max_recv, experts, topk, tm)), dtype=torch.int32, device=dev
        )
        if self._is_fp4:
            self._nf_a2 = torch.zeros((max_recv * topk, inter_dim // 2), dtype=torch.uint8, device=dev)
        else:
            self._nf_a2 = torch.zeros((max_recv * topk, inter_dim), dtype=torch.float8_e4m3fn, device=dev)
        sbm = self.sort_block_m
        pr = ((max_blocks * sbm + 255) // 256) * 256
        pc = (((inter_dim // 32) + 7) // 8) * 8
        self._nf_a2s = torch.zeros(pr * pc + inter_dim, dtype=torch.uint8, device=dev)
        self._nf_bias = torch.empty((0,), dtype=torch.float32, device=dev)
        self._nf_w1 = w1.contiguous()
        self._nf_w1s = w1_scale.contiguous()
        self._nf_gemm1 = compile_mixed_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=epr,
            topk=topk,
            tile_m=tm,
            tile_n=tn1,
            tile_k=tk,
            doweight_stage1=False,
            a_dtype=self.a2_dtype,
            b_dtype="fp4",
            out_dtype=self.a2_dtype,
            act="silu",
            waves_per_eu=4,
            use_async_copy=True,
            # w1/w1_scale layout must follow the op's gate_mode (else gemm1 misreads interleaved w1).
            gate_mode=self.gate_mode,
        )
        self._nf_trc = None  # cached total_recv (fixed routing) so forward stays CUDAGraph-safe

    def _run_nonfused_stage1(self, x_bf16, wc, ic) -> "Stage1Output":
        """bf16 dispatch -> recv-quant -> moe_sorting -> GEMM1 -> Stage1Output."""
        topk = self.topk
        model_dim, inter_dim, experts = self.model_dim, self.inter_dim, self.experts
        max_recv, tm = self.max_recv, self._nf_tm
        self._nf_a2.zero_()
        self.comb_op.total_recv.zero_()
        bt, ow, _, oidx, _ = self.comb_op.dispatch(x_bf16, wc, None, ic)
        if self._nf_trc is None:
            self._nf_trc = max(1, int(self.comb_op.total_recv.item()))
        trc = self._nf_trc
        self._nf_recv_topk[:trc].copy_(oidx[:trc].to(torch.int32))
        moe_sorting_flydsl(
            self._nf_recv_topk[:trc],
            ow[:trc],
            self._nf_a_st,
            self._nf_a_sw,
            self._nf_a_se,
            self._nf_a_nv,
            self._nf_a_mbuf[:trc],
            int(experts),
            int(tm),
            expert_mask=self._nf_expert_mask,
            workspace=self._nf_sort_ws,
        )
        a1q, a1sp = _flydsl_mx_quant(bt[:trc], quant_mode=("fp4" if self._is_fp4 else "fp8"))
        mxfp4_moe_scale_sort(self._nf_a1s, a1sp, self._nf_a_st, self._nf_a_nv, int(trc), int(model_dim))
        a_mat = a1q.view(torch.uint8) if self.a2_dtype == "fp4" else a1q
        self._nf_gemm1(
            self._nf_a2.view(max_recv, topk, self._nf_a2.shape[-1]),
            a_mat,
            self._nf_w1,
            self._nf_a1s.view(torch.uint8),
            self._nf_w1s,
            self._nf_a_st,
            self._nf_a_se,
            self._nf_a_sw,
            self._nf_a_nv,
            self._nf_bias,
            self._nf_a2s,
            fx.Int32(trc),
            fx.Int32(inter_dim * 2),
            fx.Int32(model_dim),
            fx.Int32(int(self._nf_max_blocks)),
            stream=fx.Stream(torch.cuda.current_stream()),
        )
        return Stage1Output(
            a2=self._nf_a2,
            a2_scale=self._nf_a2s,
            sorted_token_ids=self._nf_a_st,
            sorted_expert_ids=self._nf_a_se,
            sorted_weights=self._nf_a_sw,
            num_valid_ids=self._nf_a_nv,
            wts_buf=None,
        )  # None -> combine uses shmem_disp_out_wts (written by the bf16 dispatch)

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
            model_dim=self.model_dim,
            inter_dim=self.inter_dim,
            experts=self.epr,
            topk=self.topk,
            tile_m=tm2,
            tile_n=tn2,
            tile_k=tk2,
            doweight_stage2=True,
            a_dtype=self.a2_dtype,
            b_dtype="fp4",
            out_dtype="bf16",
            accumulate=True,
            persist_m=pm,
            sort_block_m=sbm,
        )
        self._nfg2_out = torch.zeros(self.max_recv, self.model_dim, dtype=torch.bfloat16, device=self.dev)
        self._nfg2_bias = torch.empty((0,), dtype=torch.float32, device=self.dev)
        self._nfg2_blocks = (self.max_recv * self.topk + self.experts * sbm + sbm - 1) // sbm
        self._nfg2_c = {}

    def _run_nonfused_stage2(self, s1, run_tokens):
        """GEMM2 -> gemm2_out; then comb_op.combine (standard, does the Stage-1 scatter)."""
        self._nfg2_out.zero_()
        ga = (
            self._nfg2_out,
            s1.a2.view(-1),
            self.w2,
            s1.a2_scale,
            self.w2_scale,
            s1.sorted_token_ids,
            s1.sorted_expert_ids,
            s1.sorted_weights,
            s1.num_valid_ids,
            self._nfg2_bias,
            self.max_recv,
            self.model_dim,
            self.inter_dim,
            int(self._nfg2_blocks),
            torch.cuda.current_stream(),
        )
        if self._nfg2_c.get("c") is None:
            self._nfg2_c["c"] = flyc.compile(self._nfg2_exe, *ga)
        else:
            self._nfg2_c["c"](*ga)
        idx = self.comb_op.shmem_disp_out_idx.view(self.max_recv, self.topk)
        # wts_buf None (non-fused stage-1) -> combine reads shmem_disp_out_wts; fused stage-1 hands a
        # flat [max_recv*topk] buffer -> combine wants [max_recv, topk].
        w = None if s1.wts_buf is None else s1.wts_buf.view(self.max_recv, self.topk)
        r = self.comb_op.combine(self._nfg2_out, w, idx, cur_tok=run_tokens)
        return r[0] if isinstance(r, (tuple, list)) else r

    def quantize(self, x_bf16):
        """bf16 activation -> (fp8/fp4 payload, e8m0 u8 scale) via FlyDSL MX quant."""
        return _flydsl_mx_quant(x_bf16, quant_mode=("fp4" if self._is_fp4 else "fp8"))

    def _run_stage2(self, s1, run_tokens, stream, slice_output):
        """stage-2 (fused GEMM2+combine or non-fused gemm2+combine) -> bf16 out."""
        if self.enable_fused_stage2:
            ret = self._run_fused_stage2(s1, run_tokens, stream)
            out_tok = ret[0] if isinstance(ret, (tuple, list)) else ret
        else:
            out_tok = self._run_nonfused_stage2(s1, run_tokens)

        if out_tok is None:
            cfg = self.comb_cfg
            out_tok = (
                self.comb_op.shmem_comb_out_tok.view(torch.int8)[: self.mtpr * cfg.combine_token_bytes]
                .view(cfg.combine_dtype)
                .view(self.mtpr, cfg.combine_token_view_dim)
            )
        return out_tok[:run_tokens] if slice_output else out_tok

    def forward(self, x_bf16, wts, topk_ids, *, stream=None, slice_output: bool = True):
        """Full MoE layer: bf16 activation in -> stage1 (fused or non-fused) -> stage2 -> bf16 out.

        Fused stage-1 quantizes internally then runs the megakernel; non-fused stage-1 bf16-dispatches
        then recv-quantizes + moe_sorting + GEMM1.
        """
        run_tokens = int(x_bf16.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        wc = wts.contiguous()
        ic = topk_ids.to(torch.int32).contiguous()
        if self.enable_fused_stage1:
            x_q, scales = self.quantize(x_bf16[:run_tokens])
            s1 = self._run_fused_stage1(x_q, wc, scales, ic, stream=stream)
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
                "fused megakernel); use forward(x_bf16, ...) for the non-fused (bf16-dispatch) stage-1."
            )
        run_tokens = int(x_q.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        wc = wts.contiguous()
        ic = topk_ids.to(torch.int32).contiguous()
        s1 = self._run_fused_stage1(x_q, wc, scales, ic, stream=stream)
        return self._run_stage2(s1, run_tokens, stream, slice_output)

    __call__ = forward
