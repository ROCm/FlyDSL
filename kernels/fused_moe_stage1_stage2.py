# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""端到端节点内 EP-MoE 单算子 ``MegaMoE``：一个完整的 MoE 层算子。

本文件把原先分散的 host 侧逻辑聚拢到一处：

  * stage-1（dispatch ⊕ GEMM1，atom 契约）—— ``FusedMoEMegaStage1``；
  * stage-2（GEMM2 + EP combine）—— ``FlyDSLMoeGemm2CombineOp`` +
    ``compile_fused_moe_gemm2_combine``（薄 builder，已内联到本文件）；
  * megastage1 / megagemm2 的调优表助手（原 fused_moe_megakernel.py）。

dispatch 的 expert-major symmetric buffer 现由通信算子
``FlyDSLDispatchCombineIntraNodeOp``（``enable_group_major=True`` 时）经其内部的
``FlyDSLDispatchGroupMajorOp``(``comb_op._gm``) 拥有；``total_recv`` 与 combine 侧
统一为同一 buffer（fused dispatch prologue 写、combine 读，零桥接）。

对外主入口 ``MegaMoE.forward(x_bf16, wts, topk_ids)``：内部量化 → stage1 → stage2 →
bf16 输出。``enable_fused_stage1`` / ``enable_fused_stage2`` 两个 init 开关选择融合/
非融合路径（非融合分支暂为占位）。
"""
from __future__ import annotations

import functools
import json
import os
import re
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms

from .fused_moe_gemm_2stage import compile_fused_moe_gemm1
from .mixed_moe_gemm_2stage import compile_mixed_moe_gemm2
from .dispatch_combine_intranode_op import (
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)

try:
    from aiter import dtypes as _adt
    from aiter.ops.quant import per_1x32_mx_quant_hip
    _HAS_AITER_QUANT = True
except Exception:  # noqa: BLE001
    _adt = None
    per_1x32_mx_quant_hip = None
    _HAS_AITER_QUANT = False


__all__ = ["MegaMoE", "FusedMoEMegaStage1", "FlyDSLMoeGemm2CombineOp",
           "compile_fused_moe_gemm2_combine"]


def _is_fp4(dt):
    return dt == torch.float4_e2m1fn_x2


# ======================================================================================
# Megastage1 / MegaGemm2 tune-table helpers (moved from fused_moe_megakernel.py)
# ======================================================================================
# Tuned tiles live in the FlyDSL JSON table below.  When a shape is intentionally
# untuned, keep only the generic validated default instead of a second
# hand-maintained host-side tune table.
def _mega_default_tile(inter_dim):
    return 64, (256 if (int(inter_dim) % 256 == 0) else 128)


_MEGA_TUNING_DIR = Path(__file__).resolve().parent / "tuning_configs"
# Megastage1 weights are always w4(fp4); the JSON ``dtype`` is the ACTIVATION quant.
_MEGA_QUANT_TO_DTYPE = {"a4w4": "fp4", "a8w4": "fp8_ocp"}


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
    """All ``megastage1`` rows from the best-matching MegaStage1 tune JSON
    (``flydsl_*_MegaStage1_ep{n}.json``), preferring a ``gpu_model`` name match.
    Returns a tuple of dicts (hashable cache value); ``()`` on any miss. Call
    ``_load_mega_tuning_rows.cache_clear()`` after editing the JSON to reload."""
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
    """Precise megastage1 config from the FlyDSL tune JSON, keyed by GEMM shape;
    the token bucket rounds up to the smallest ``num_tokens >= mtpr`` (largest on
    overflow), mirroring the gemm2+combine ``GeometryTuningTable.lookup``. Returns
    ``{tile_m, tile_n, tile_k, waves_per_eu, use_async_copy}`` or ``None`` on a
    miss (caller then falls back to the generic default tile)."""
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
    """All ``megagemm2`` rows from the best-matching MegaGemm2 tune JSON
    (``flydsl_*_MegaGemm2_ep{n}.json``), preferring a ``gpu_model`` name match.
    Returns a tuple of dicts; ``()`` on any miss. Symmetric with
    ``_load_mega_tuning_rows`` (gemm1). Call ``.cache_clear()`` after editing."""
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


def _mega_gemm2_tuned_table(model_dim, inter_dim, experts, topk, ep_size, gpu_model):
    """Per-M fused gemm2 tile table ``{num_tokens: (tile_m,tile_n,tile_k)}`` from
    the FlyDSL MegaGemm2 tune JSON, keyed by GEMM2 shape. ``experts`` is the
    LOCAL per-rank expert count (how the gemm2 tiles were tuned). Returns the
    dict, or ``None`` on a miss so the op falls back to its single default tile.
    RAW tiles -- the op clamps tile_m to a divisor of sort_block_m at run time.
    Default entry point for ``MegaMoE`` (no env gate; mirrors gemm1's auto-tune)."""
    rows = _load_mega_gemm2_rows(int(ep_size), gpu_model)
    if not rows:
        return None

    def _match(r):
        try:
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
                table[int(r["num_tokens"])] = (
                    int(r["tile_m"]), int(r["tile_n"]), int(r["tile_k"]))
            except (KeyError, ValueError):
                continue
    return table or None


# ======================================================================================
# stage-2 fused GEMM2+combine builder (inlined from mixed_moe_gemm2_combine_fused.py)
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

    Rewrites the GEMM2 epilogue's local store_pair into a remote P2P buffer_store,
    inlining combine Stage 1's P2P scatter. The host then launches a trimmed combine
    (``combine_no_stage1``) to run Stage 2/3.
    """
    # Plan B slot = dest_lid * k + s; reusing baseline shmem_comb_inp_tok
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
        raise ValueError(
            f"max_tok_per_rank={max_tok_per_rank} must be a power of two."
        )

    out_s = str(out_dtype).strip().lower()
    fp8_cast = out_s in (
        "fp8", "fp8e4m3", "fp8e4m3fn", "f8e4m3fn", "float8_e4m3fn"
    )
    if not fp8_cast and out_s not in ("bf16", "bfloat16", "f16", "fp16"):
        raise ValueError(
            f"fused mode output must be bf16/f16 or fp8e4m3fn, got {out_dtype!r}"
        )

    # fused_cfg[3]=enable_weights is pinned False: in-kernel weight P2P
    # scatter (~16B writes) is unreliable under fabric saturation, so weights
    # are handled by the subsequent combine_no_stage1 Stage 1 instead.
    # Under fp8_cast GEMM2 stays bf16 internally; store_pair does the
    # bf16->fp32->cvt_pk_fp8 conversion and writes 1B/elem.
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
# stage-2 fused GEMM2+combine operator (moved from mixed_moe_gemm2_combine_fused_op.py)
# ======================================================================================
class FlyDSLMoeGemm2CombineOp:
    """Fused MoE-GEMM2 + EP-Combine operator.

    The GEMM2 epilogue inlines combine Stage 1's P2P token scatter; the
    host then launches a trimmed combine (``combine_no_stage1``) for
    Stage 2/3. This is the only supported mode.
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
        persist_m: int = 4,
        sort_block_m: int = 0,
        b_nt: int = 2,
        a_dtype: str = "fp8",
        b_dtype: str = "fp4",
        xcd_swizzle: int = 0,
        use_token_flag_sync: bool = False,
        doweight_fused: bool = True,
        gemm2_tile_table: dict | None = None,
        force_mode: Optional[str] = None,
    ):
        # force_mode is accepted for back-compat (some callers pass
        # "stage1_only"); run() always uses the fused stage1-only path, so it is
        # a no-op flag here.
        if force_mode not in (None, "stage1_only"):
            raise ValueError(f"FlyDSLMoeGemm2CombineOp: unsupported force_mode={force_mode!r} "
                             "(only 'stage1_only' / None supported)")
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
        # Apply routing weights in the GEMM2 epilogue; combine_no_stage1
        # reduces tokens unweighted, so this is on by default for correctness.
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

        # Per-tile JIT caches: the gemm2 kernel bakes (tile_m,tile_n,tile_k) as
        # compile-time constants, so to mirror aiter's per-M tile selection we
        # compile/cache one kernel per distinct tile and pick by run_tokens at
        # run(). The heavy comb_op shmem is shared across tiles (only the gemm2
        # kernel binary varies) -> no extra symmetric-heap cost.
        self._launch_by_tile = {}
        self._compiled_by_tile = {}
        # Optional {pow2_token: (tile_m, tile_n, tile_k)} table (built host-side
        # from aiter get_2stage_cfgs by the caller, cktile buckets pre-aligned
        # to the nearest FlyDSL bucket). None -> always use the default tile.
        self._tile_table = dict(gemm2_tile_table) if gemm2_tile_table else None
        # When a per-M table is active, the sorted_expert_ids fed by stage1 are
        # laid out at a FIXED block (MegaMoE: 32). gemm2 must read at that block
        # (its tile_m only sub-tiles within it), so pin sort_block_m=32 instead
        # of the default (sort_block_m<=0 -> kernel assumes == tile_m, which
        # mismatches stage1 once tile_m != 32). tile_m in the table is already
        # clamped to a divisor of 32 by the builder.
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
        """Safety net: gemm2 sub-tiles the stage1 sorted_expert_ids blocks, so
        tile_m MUST divide self.sort_block_m or the kernel reads the wrong
        expert's W2 (silent garbage) / trips the _ensure_launch_fn assert. If a
        (mis)built table ever hands us a tile_m that does not divide it, halve
        tile_m down to the nearest power-of-two divisor (floored at 16) instead
        of crashing. No-op when sort_block_m<=0 (kernel assumes == tile_m)."""
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
        """Pick (tile_m, tile_n, tile_k) for this forward's run_tokens via the
        host-static table (nextPow2 bucket, clamped to the table's range),
        mirroring aiter's get_2stage_cfgs(get_padded_M(token_num)). Falls back
        to the ctor default tile when no table or no run_tokens is given. The
        chosen tile_m is always clamped to a divisor of sort_block_m (safety)."""
        if not self._tile_table or run_tokens is None:
            return (self._clamp_tile_m(self.tile_m), self.tile_n, self.tile_k)
        keys = sorted(self._tile_table)
        b = self._next_pow2(run_tokens)
        if b < keys[0]:
            b = keys[0]
        elif b > keys[-1]:
            b = keys[-1]
        if b not in self._tile_table:
            # nearest bucket >= b, else largest
            ge = [k for k in keys if k >= b]
            b = ge[0] if ge else keys[-1]
        tm, tn, tk = self._tile_table[b]
        return (self._clamp_tile_m(tm), int(tn), int(tk))

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
        run_tokens: Optional[int] = None,
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
            run_tokens=run_tokens,
        )

    def _ensure_launch_fn(self, tile=None):
        """Build (once) and return the launch_fn for ``tile`` (defaults to the
        ctor tile). Cached per distinct (tile_m,tile_n,tile_k)."""
        if tile is None:
            tile = (self.tile_m, self.tile_n, self.tile_k)
        tile = tuple(int(v) for v in tile)
        # Divisibility contract: gemm2 sub-tiles the sorted_expert_ids blocks,
        # so tile_m must divide the real sort block (the granularity stage1
        # emitted sorted_expert_ids at). sort_block_m<=0 => kernel assumes
        # == tile_m (always OK). Catch mismatches early with a clear error
        # instead of silently producing garbage (the 0.22 gsm8k failure mode).
        if int(self.sort_block_m) > 0 and (int(self.sort_block_m) % tile[0]) != 0:
            raise ValueError(
                f"FlyDSLMoeGemm2CombineOp: gemm2 tile_m={tile[0]} must divide "
                f"sort_block_m={self.sort_block_m} (stage1 sorted_expert_ids block). "
                f"Pick tile_m in divisors of {self.sort_block_m}."
            )
        lf = self._launch_by_tile.get(tile)
        if lf is not None:
            return lf
        comb_cfg = self.comb_cfg
        # Plan B writes into shmem_comb_inp_tok's [0, mt*k) sub-range; k <= npes
        # keeps it in-bounds (mt*k <= mr).
        k = comb_cfg.num_experts_per_token
        if k > comb_cfg.world_size:
            raise ValueError(
                f"FlyDSLMoeGemm2CombineOp (Plan B) requires k <= npes; "
                f"got k={k}, npes={comb_cfg.world_size}."
            )
        lf = compile_fused_moe_gemm2_combine(
            model_dim=comb_cfg.hidden_dim,
            inter_dim=self.inter_dim,
            experts=comb_cfg.num_experts_per_rank,
            topk=k,
            tile_m=tile[0], tile_n=tile[1], tile_k=tile[2],
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
        self._launch_by_tile[tile] = lf
        return lf

    def _run_stage1_only(self, *, a2, w2, a2_scale, w2_scale,
                         sorted_token_ids, sorted_expert_ids, sorted_weights,
                         num_valid_ids, wts_buf=None, cur_tok=None, bias=None, stream=None,
                         run_tokens=None):
        """fused GEMM2 (epilogue P2P scatter) + combine_no_stage1 (Stage 2/3)."""
        # Per-M tile selection (aiter-aligned: key on the un-expanded forward
        # token count, NOT the topk-expanded gemm rows). Host-static per
        # cudagraph capture -> the chosen kernel is baked into each graph.
        tile = self._select_tile(run_tokens)
        launch_fn = self._ensure_launch_fn(tile)

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
        # Debug knob FLYDSL_FUSED_ZERO_MODE=normal force-enables the zero.
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
        compiled = self._compiled_by_tile.get(tile)
        if compiled is None:
            compiled = flyc.compile(
                launch_fn,
                *common_args,
                fx.Int32(tokens_in), fx.Int32(n_in), fx.Int32(k_in),
                fx.Int32(size_expert_ids),
                s_fx,
            )
            self._compiled_by_tile[tile] = compiled
        else:
            compiled(
                *common_args,
                tokens_in, n_in, k_in, size_expert_ids,
                s_fx,
            )

        # Weights still go through combine_no_stage1's Stage 1 weight-only
        # P2P scatter + Stage 3b accum (negligible ~k*4B/token traffic), so
        # end-to-end output matches baseline bit-for-bit. fused only removes
        # the heavy token P2P traffic.
        return comb_op.combine_no_stage1(
            self._dummy_inp, None, None, cur_tok=cur_tok, enable_weights=True,
        )


# ======================================================================================
# stage-1 driver: single-launch fused dispatch ⊕ GEMM1 (moved from fused_moe_megakernel.py)
# ======================================================================================
def _mega_default_tile_wrap(inter_dim):  # kept name compat / small indirection
    return _mega_default_tile(inter_dim)


def _resolve_stage1_config(*, model_dim, inter_dim, experts, topk, quant,
                           max_tok_per_rank, world_size, rank,
                           tune_tokens=None, unit_size=-1, tile_n=-1, tile_k=256,
                           waves_per_eu=4, use_async_copy=True):
    """Resolve the fused stage-1 GEMM tile + layout (compact) BEFORE the combine
    op allocates its group-major buffers (which need unit_size/compact). Mirrors
    the resolution the former FusedMoEMegaStage1.__init__ did inline. Returns a
    dict of resolved values consumed by both the combine op (gm_*) and stage-1."""
    assert quant in ("a4w4", "a8w4")
    if experts % world_size != 0:
        raise ValueError(f"experts={experts} must divide world_size={world_size}")
    epr = experts // world_size
    data_type = torch.float8_e4m3fn if quant == "a8w4" else torch.float4_e2m1fn_x2
    a_dtype = "fp8" if quant == "a8w4" else "fp4"
    scale_dim = model_dim // 32

    # Two layouts, switched purely by buffer size (no env): fixedslot below the
    # 4GB wrap, compact above it.
    _row_b = int(model_dim) if quant == "a8w4" else (int(model_dim) // 2)
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
    unit_size = int(unit_size)
    sort_block_m = max(32, unit_size)

    # ── XCD swizzle correctness guard (mirrors former FusedMoEMegaStage1) ──
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
                f"{_cu_g} % gx != 0) -> XCD WGM group straddles XCD boundaries and would corrupt/"
                f"lose output tiles; gx={_gx_g} gives no XCD-alignment benefit anyway.")
            xcd = 0

    return dict(
        epr=epr, data_type=data_type, a_dtype=a_dtype, scale_dim=scale_dim,
        compact=compact, unit_size=unit_size, tile_n=int(tile_n), tile_k=int(tile_k),
        waves_per_eu=int(waves_per_eu), use_async_copy=bool(use_async_copy),
        sort_block_m=sort_block_m, xcd=xcd, tune_tokens=tune_tokens,
    )


class FusedMoEMegaStage1:
    """Single-launch fused dispatch⊕GEMM megakernel (fixedslot decode strict-phase).

    The expert-major dispatch buffers live in the combine op (``comm_op._gm``,
    a ``FlyDSLDispatchGroupMajorOp``); this driver compiles the fused kernel,
    builds the dispatch-arg pointer table over those buffers, and runs forward.
    """

    def __init__(self, *, comm_op, cfg, rank, world_size, model_dim, inter_dim,
                 experts, topk, quant, w1, w1_scale, max_tok_per_rank,
                 scheme="fixedslot", warp_num_per_block=4, out_dtype="auto"):
        assert quant in ("a4w4", "a8w4")
        assert scheme == "fixedslot", f"scheme={scheme!r}: only 'fixedslot' supported (handshake removed)"
        assert out_dtype in ("auto", "f16", "fp8", "fp4"), out_dtype
        self.scheme = scheme
        # The megakernel ALWAYS emits a2 in the ATOM contract; the GEMM is always
        # compiled with atom_contract=True.
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
        if out_dtype == "auto":
            out_dtype = self.a_dtype
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

        # ---- group-major dispatch buffers owned by the combine op ----
        self.comm_op = comm_op
        self.op = comm_op._gm
        assert self.op is not None, "combine op was built without enable_group_major"
        self.nvm = self.op.num_valid_max
        self.max_blocks = self.op.max_blocks
        self.cap = self.op.ll_cap
        # metadata-ready flag (monotonic, NEVER reset -> CUDAGraph-safe).
        self._meta = torch.zeros(1, dtype=torch.int32, device=self.dev)
        # total_recv is UNIFIED with combine (op.total_recv aliases comb_op.total_recv);
        # no separate bridge buffer.
        self._trecv_buf = None

        # ---- compile the megakernel (serialize compile across ranks to bound peak memory) ----
        self.mega = None
        for pe in range(world_size):
            if rank == pe:
                self.mega = compile_fused_moe_gemm1(
                    model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=1,
                    tile_m=self.unit_size, tile_n=self.tile_n, tile_k=tile_k, doweight_stage1=False,
                    a_dtype=self.a_dtype, b_dtype="fp4", out_dtype=out_dtype, act="silu",
                    waves_per_eu=waves_per_eu, use_async_copy=use_async_copy,
                    use_cshuffle_epilog=None, contiguous_io=True, dedup_gather=False,
                    atom_contract=self.atom_contract,
                    sparse_tiles=True, persist_m=-1,
                    raw_a_scale=True, xcd_swizzle=self._xcd,
                    fuse_dispatch=scheme, fuse_npes=world_size, fuse_topk=topk,
                    fuse_cap=self.cap, fuse_mtpr=self.mtpr,
                    fuse_scale_dim=self.scale_dim, fuse_scale_type_size=1,
                    rank=rank, experts_per_rank=self.epr, compact_dispatch=self.compact,
                    compact_allgather=True)
            if dist.is_initialized():
                dist.barrier()

        # ---- views into op buffers (A-input + raw A-scale + metadata the GEMM consumes) ----
        v = self.op._ll_views()
        self._rx = v["rx_em"]
        self._scale_i32 = v["scale_em_i32"]
        self._trb = self.op.tile_row_base
        self._se = self.op.sorted_expert_ids
        self._nv = self.op.num_valid

        # ---- output / scratch ----
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
        # COMPACT sorted_token_ids[compact_row] = (k<<24)|src_global into a dedicated [nvm] buffer.
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

        # ---- dispatch-arg FIXED-pointer table (op bufs + p2p); built ONCE. ----
        self._build_disp_table()

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
                # 19-26: compact count-first extras
                tbl += [op.compact_base.data_ptr(), op.p2p_compact_base.data_ptr(),  # 19-20
                        op.gb_cnt.data_ptr(), op.done2c.data_ptr(), op.p2p_done2c.data_ptr(),  # 21-23
                        op.meta2.data_ptr(),                                          # 24 payload-ready flag
                        op.write_cursor.data_ptr(), op.p2p_write_cursor.data_ptr(),   # 25-26
                        op.done2cb.data_ptr(), op.p2p_done2cb.data_ptr(),             # 27-28 cross-PE#1b
                        op.local_hist.data_ptr(), op.bigcnt.data_ptr(), op.p2p_bigcnt.data_ptr(),  # 29-31
                        op.cnt_done.data_ptr(), op.p2p_cnt_done.data_ptr(),           # 32-33
                        op.my_base.data_ptr(), op.local_cursor.data_ptr()]            # 34-35
                tbl[19] = op.srcmap_em.data_ptr()                 # 19 LOCAL srcmap (override compact_base)
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

    def forward(self, x, wts, scales, topk_ids, stream=None):
        """Runs the WHOLE stage-1 (dispatch + group-GEMM) in ONE launch."""
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream())
        cur_tok = int(x.shape[0])
        assert cur_tok <= self.mtpr, f"[mega] cur_tok={cur_tok} > max_tok_per_rank={self.mtpr}"
        xc = x.contiguous(); wc = wts.contiguous()
        ic = topk_ids.to(torch.int32).contiguous(); sc = scales.contiguous()

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
        self.mega(self._out, a_mat, self.w1, self._scale_i32, self.w1_scale,
                  _sorted_arg, _se_arg, _wt_arg, self._nv, self._bias, self._osd,
                  fx.Int32(self.nvm), fx.Int32(self.inter_dim * 2), fx.Int32(self.model_dim),
                  fx.Int32(self.max_blocks),
                  fx.Int64(pd_ptr), fx.Int64(er_ptr),
                  fx.Int64(self._disp.data_ptr()), fx.Int32(cur_tok),
                  fx.Int64(xc.data_ptr()), fx.Int64(ic.data_ptr()),
                  fx.Int64(wc.data_ptr()), fx.Int64(sc.data_ptr()),
                  stream=stream)
        return dict(out=self._out, srcmap_em=self.op.srcmap_em, num_valid=self._nv,
                    sorted_expert_ids=self._se_atom,
                    sorted_token_ids=self._sti, sorted_weights=self._sw_atom,
                    sorted_weights_row=self._wts_sorted,
                    a2_scale=self._osd, impl="mega_" + self.scheme)


# ======================================================================================
# End-to-end EP-MoE single operator
# ======================================================================================
class MegaMoE:
    """端到端 EP-MoE 单算子（完整 MoE 层）。

    ``enable_fused_stage1`` / ``enable_fused_stage2``：init 固定选择融合/非融合路径。
    本期仅融合路径已接线；非融合分支占位（NotImplementedError）。
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
        stage2_mode: Optional[str] = None,
    ):
        assert quant in ("a8w4", "a4w4"), quant
        # Back-compat: stage2_mode="fused"/"nonfused" maps to enable_fused_stage2.
        if stage2_mode is not None:
            assert stage2_mode in ("fused", "nonfused"), stage2_mode
            enable_fused_stage2 = (stage2_mode == "fused")
        if not _HAS_AITER_QUANT:
            raise RuntimeError("MegaMoE needs aiter (per_1x32_mx_quant_hip)")
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
        self.dev = torch.device("cuda", rank)
        self._is_fp4 = (quant == "a4w4")
        self._qd = _adt.fp4x2 if self._is_fp4 else _adt.fp8
        self._stp = None if self._is_fp4 else _adt.fp8_e8m0

        if not self.enable_fused_stage1:
            raise NotImplementedError(
                "enable_fused_stage1=False (non-fused stage1: dispatch + moe_sorting + "
                "compile_mixed_moe_gemm1) is not yet wired; use enable_fused_stage1=True.")
        if not self.enable_fused_stage2:
            raise NotImplementedError(
                "enable_fused_stage2=False (non-fused stage2: compile_mixed_moe_gemm2 + "
                "comb_op.combine) is not yet wired; use enable_fused_stage2=True.")

        # ---- resolve stage-1 tile / layout (needed BEFORE the combine op allocs group-major) ----
        self._s1cfg = _resolve_stage1_config(
            model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk,
            quant=quant, max_tok_per_rank=max_tok_per_rank, world_size=world_size,
            rank=rank, tune_tokens=tune_tokens, unit_size=int(tile_m), tile_n=int(tile_n),
            tile_k=int(tile_k), waves_per_eu=int(waves_per_eu),
            use_async_copy=bool(use_async_copy))
        self.a2_dtype = self._s1cfg["a_dtype"]  # "fp8" | "fp4"

        # ---- unified stage-2 combine op + group-major dispatch buffers (one op) ----
        try:
            _cu = int(torch.cuda.get_device_properties(self.dev).multi_processor_count)
        except Exception:  # noqa: BLE001
            _cu = 256
        self.comb_cfg = FlyDSLDispatchCombineConfig(
            rank=rank, world_size=world_size, hidden_dim=model_dim,
            max_num_inp_token_per_rank=self.mtpr, num_experts_per_rank=self.epr,
            num_experts_per_token=topk, data_type=torch.bfloat16,
            scale_dim=0, scale_type_size=0, enable_std_moe=False,
            # group-major (fused stage-1) dispatch buffers owned by this op:
            enable_group_major=True,
            gm_data_type=self._s1cfg["data_type"],
            gm_unit_size=self._s1cfg["unit_size"],
            gm_scale_dim=self._s1cfg["scale_dim"],
            gm_scale_type_size=1,
            gm_scheme=mega_scheme,
            gm_compact=self._s1cfg["compact"],
        )
        self.comb_op = FlyDSLDispatchCombineIntraNodeOp(self.comb_cfg)
        torch.cuda.synchronize(); ms.shmem_barrier_all()

        # ---- stage-1: fused dispatch ⊕ GEMM1 (uses comb_op._gm buffers) ----
        self.stage1 = FusedMoEMegaStage1(
            comm_op=self.comb_op, cfg=self._s1cfg,
            rank=rank, world_size=world_size, model_dim=model_dim, inter_dim=inter_dim,
            experts=experts, topk=topk, quant=quant, w1=w1, w1_scale=w1_scale,
            max_tok_per_rank=max_tok_per_rank, scheme=mega_scheme,
            warp_num_per_block=int(warp_num_per_block), out_dtype="auto")

        self.w2 = w2 if w2.is_contiguous() else w2.contiguous()
        self.w2_scale = w2_scale if w2_scale.is_contiguous() else w2_scale.contiguous()
        self.max_recv = world_size * self.mtpr
        # megav1 _sti encodes src_global=dest -> combine tok_id_to_src is identity (const);
        # set once at construction (combine reads, stage1 never touches it).
        self.comb_op.shmem_tok_id_to_src.copy_(
            torch.arange(self.max_recv, device=self.dev, dtype=torch.int32))
        torch.cuda.synchronize(); ms.shmem_barrier_all()

        # ---- fused GEMM2 ghost-gate boundary = constant max_recv (mega self-contained) ----
        # gemm2 ghost gate reads comb_op._fx_out_total_recv; repoint it to a constant max_recv
        # so real rows (src_global in [0, npes*mtpr)) are all kept and padding (==max_recv)
        # dropped. combine_no_stage1 still reads the real total_recv via _fx_trecv.
        self._gemm2_gate_bound = torch.tensor(
            [self.max_recv], dtype=torch.int32, device=self.dev)
        self.comb_op._fx_out_total_recv = fx.Int64(self._gemm2_gate_bound.data_ptr())
        torch.cuda.synchronize(); ms.shmem_barrier_all()

        # ---- stage-2 backend (fused GEMM2+combine) ----
        _s1_sbm = int(self.stage1.sort_block_m)
        _eff_g2_tm = int(gemm2_tile_m) if int(gemm2_tile_m) > 0 else 32  # op default tile_m=32
        if _s1_sbm % _eff_g2_tm != 0:
            raise ValueError(
                f"gemm2 tile_m={_eff_g2_tm} does not divide stage1 sort_block_m={_s1_sbm} "
                f"(stage1 unit_size={self.stage1.unit_size}); set gemm2_tile_m to a divisor "
                f"of {_s1_sbm} (e.g. 32), or raise stage1's tile_m.")
        if gemm2_tile_table is None:
            try:
                gemm2_tile_table = _mega_gemm2_tuned_table(
                    int(model_dim), int(inter_dim),
                    int(self.experts) // int(self.world_size), int(topk),
                    int(self.world_size), _detect_gpu_model_name(int(self.rank)))
                if int(self.rank) == 0:
                    print(f"[MegaGemm2] auto-config tile table "
                          f"(sort_block_m={_s1_sbm}): "
                          f"{'<miss -> default tile>' if not gemm2_tile_table else gemm2_tile_table}",
                          flush=True)
            except Exception:  # noqa: BLE001 -- fallback to default tile
                gemm2_tile_table = None
        g2_kwargs = dict(
            comb_cfg=self.comb_cfg, comb_op=self.comb_op, inter_dim=int(inter_dim),
            a_dtype=self.a2_dtype, b_dtype="fp4", sort_block_m=_s1_sbm,
            gemm2_tile_table=gemm2_tile_table,
        )
        if int(gemm2_tile_m) > 0:
            g2_kwargs.update(
                tile_m=int(gemm2_tile_m), tile_n=int(gemm2_tile_n),
                tile_k=int(gemm2_tile_k), persist_m=int(gemm2_persist_m),
                xcd_swizzle=int(xcd_swizzle),
            )
        self._g2 = FlyDSLMoeGemm2CombineOp(**g2_kwargs)

    def quantize(self, x_bf16):
        """便捷量化：bf16 激活 -> (fp8/fp4 payload, e8m0 scale uint8)。"""
        if self._is_fp4:
            mq, msq = per_1x32_mx_quant_hip(x_bf16.contiguous(), quant_dtype=self._qd)
        else:
            mq, msq = per_1x32_mx_quant_hip(x_bf16.contiguous(), quant_dtype=self._qd,
                                            scale_type=self._stp)
        return mq, msq.view(torch.uint8)

    def forward(self, x_bf16, wts, topk_ids, *, stream=None, slice_output: bool = True):
        """完整 MoE 层：bf16 激活进 -> 内部量化 -> stage1 -> stage2 -> bf16 出。"""
        x_q, scales = self.quantize(x_bf16)
        return self.forward_prequant(x_q, scales, wts, topk_ids,
                                     stream=stream, slice_output=slice_output)

    # bf16 in, bf16 out; primary entry.
    forward_bf16 = forward

    def forward_prequant(self, x_q, scales, wts, topk_ids, *, stream=None, slice_output: bool = True):
        """已量化激活入口（fp8/fp4 + e8m0 scale 进，bf16 出）。"""
        run_tokens = int(x_q.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        wc = wts.contiguous()
        ic = topk_ids.to(torch.int32).contiguous()

        # ---- stage-1: fused dispatch ⊕ GEMM1 (writes combine op's total_recv + atom-contract a2) ----
        self.stage1.forward(x_q, wc, scales, ic, stream=stream)

        # ---- stage-2: fused GEMM2 + combine ----
        ret = self._g2.run(
            a2=self.stage1._out.view(-1), w2=self.w2,
            a2_scale=self.stage1._osd, w2_scale=self.w2_scale,
            sorted_token_ids=self.stage1._sti,
            sorted_expert_ids=self.stage1._se_atom,
            sorted_weights=self.stage1._wts_sorted,
            num_valid_ids=self.stage1._nv,
            wts_buf=self.stage1._sw_atom,
            cur_tok=run_tokens, stream=stream,
            run_tokens=run_tokens,
        )
        out_tok = ret[0] if isinstance(ret, (tuple, list)) else ret

        if out_tok is None:
            cfg = self.comb_cfg
            out_tok = (self.comb_op.shmem_comb_out_tok.view(torch.int8)[: self.mtpr * cfg.token_bytes]
                       .view(cfg.data_type).view(self.mtpr, cfg.token_view_dim))
        return out_tok[:run_tokens] if slice_output else out_tok

    __call__ = forward
