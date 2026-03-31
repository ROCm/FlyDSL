#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import argparse
import logging
import math
import os
import sys
import time
from typing import Tuple, Optional, List

import pytest
import torch

# -----------------------------------------------------------------------------
# Ensure we use the repo-local `flydsl` when running this file directly.
#
# Some environments have another `flydsl` (e.g. from a sibling checkout) earlier
# on `sys.path`, which can miss newer ROCDL wrappers (notably atomic fadd / MFMA).
# -----------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYTHON_CANDIDATES = [
    os.path.join(_REPO_ROOT, "build", "python_packages"),
    _REPO_ROOT,
]
for _p in reversed(_PYTHON_CANDIDATES):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from tests.kernels.test_ref import torch_moe_gemm1, torch_moe_gemm2
from tests.utils import pertoken_quant, shuffle_weight, shuffle_scale_for_int4
from tests.test_common import verify_output, run_perftest
from flydsl.runtime.device import get_rocm_arch
from tests.kernels.utils import fp4_utils

ARCH = get_rocm_arch()
# GFX950 (MI350) and newer typically use OCP standard float8_e4m3fn
# GFX940/941/942 (MI300) use float8_e4m3fnuz
if "gfx95" in ARCH:
    DTYPE_FP8 = torch.float8_e4m3fn
else:
    DTYPE_FP8 = torch.float8_e4m3fnuz

def _pack_shuffled_int8_to_packed_int4_no_perm(x_shuf_i8: torch.Tensor) -> torch.Tensor:
    """Pack a preshuffled int8 tensor (values in [-8, 7]) into packed int4 bytes.

    Each contiguous 8-value block [v0..v7] -> 4 bytes:
      b0=(v4<<4)|v0, b1=(v5<<4)|v1, b2=(v6<<4)|v2, b3=(v7<<4)|v3.

    This matches the 7-op in-kernel unpack sequence and avoids any v_perm.
    """
    flat = x_shuf_i8.contiguous().view(-1).to(torch.int16)
    assert flat.numel() % 8 == 0
    u = (flat & 0xF).to(torch.uint8).view(-1, 8)
    out = torch.empty((u.shape[0], 4), device=u.device, dtype=torch.uint8)
    out[:, 0] = u[:, 0] | (u[:, 4] << 4)
    out[:, 1] = u[:, 1] | (u[:, 5] << 4)
    out[:, 2] = u[:, 2] | (u[:, 6] << 4)
    out[:, 3] = u[:, 3] | (u[:, 7] << 4)
    return out.view(-1).to(torch.int8)

# Optional: use aiter's exact routing/sorting implementation (matches `aiter/op_tests/test_moe_2stage.py`).
# Some environments ship aiter python but miss required JIT .so dependencies; we fall back gracefully.
try:
    import aiter
    from aiter.fused_moe import moe_sorting as aiter_moe_sorting

    HAS_AITER = True
except Exception:
    HAS_AITER = False

# Kernel implementations live under `kernels/`; this test file is the harness.
from kernels.moe_gemm_2stage_gfx1250 import (
    compile_moe_gemm1,
    compile_moe_gemm2,
    compile_moe_gemm2_ex,
    MoeGemm2Mode,
)

logging.basicConfig(level=logging.INFO)

# Reduce noisy aiter log spam (e.g. "type hints mismatch, override to --> ...") so test output
# stays readable. You can override via env: FLYDSL_AITER_LOG_LEVEL=INFO/WARNING/ERROR.
_aiter_level = os.environ.get("FLYDSL_AITER_LOG_LEVEL", "ERROR").upper().strip()
try:
    logging.getLogger("aiter").setLevel(getattr(logging, _aiter_level, logging.ERROR))
except Exception:
    # Best-effort only; never break tests due to logging configuration.
    pass


def _use_serial_fp4_debug(in_dtype: str) -> bool:
    dt = str(in_dtype).strip().lower()
    if dt not in ("fp4", "fp8", "fp16"):
        return False
    if dt == "fp4":
        serial_key = "MOE_FP4_SERIAL"
        debug_key = "MOE_FP4_DEBUG"
    elif dt == "fp8":
        serial_key = "MOE_FP8_SERIAL"
        debug_key = "MOE_FP8_DEBUG"
    else:
        serial_key = "MOE_FP16_SERIAL"
        debug_key = "MOE_FP16_DEBUG"
    serial = str(os.environ.get(serial_key, "0")).strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    dbg = str(os.environ.get(debug_key, "0")).strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return serial or dbg


def _use_serial_fp4_perf(in_dtype: str, debug_serial: bool) -> bool:
    dt = str(in_dtype).strip().lower()
    if dt not in ("fp4", "fp8", "fp16"):
        return False
    if bool(debug_serial):
        return True
    # Default to safe mode for fp4/fp8/fp16 to avoid run_perftest rotate-worker segfaults.
    if dt == "fp4":
        safe_key = "MOE_FP4_SAFE_PERF"
    elif dt == "fp8":
        safe_key = "MOE_FP8_SAFE_PERF"
    else:
        safe_key = "MOE_FP16_SAFE_PERF"
    safe = str(os.environ.get(safe_key, "1")).strip().lower()
    return safe in {"1", "true", "t", "yes", "y", "on"}

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

if not str(ARCH).startswith("gfx1250"):
    pytest.skip(f"MoE 2stage gfx1250 tests require gfx1250, got {ARCH}", allow_module_level=True)

def moe_sorting_torch_native(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    block_size: int,
    expert_mask: Optional[torch.Tensor] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch reference for aiter's moe_sorting.

    Returns:
      - sorted_ids[int32]: fused (topk_slot<<24 | token_id)
      - sorted_weights[fp32]: aligned with sorted_ids
      - sorted_expert_ids[int32]: one expert id per M-block (size = num_blocks)
      - num_tokens_post_pad[int32]: [0]=total padded tokens, [1]=num_tokens (logical)

    Notes:
      - This function intentionally mirrors `aiter/op_tests/test_moe_sorting.py::moe_sorting_native`.
    """
    assert topk_ids.is_cuda and topk_weights.is_cuda
    device = topk_ids.device
    M, topk = topk_ids.shape
    topk = topk_ids.shape[1]

    # Upper bound allocation (matches aiter op_tests; not strictly required but keeps shapes predictable).
    max_num_tokens_padded = int(topk_ids.numel() + int(num_experts) * int(block_size) - int(topk))
    max_num_m_blocks = int((max_num_tokens_padded + int(block_size) - 1) // int(block_size))

    init_val = (int(topk) << 24) | int(M)
    sorted_ids = torch.full((max_num_tokens_padded,), init_val, dtype=torch.int32, device=device)
    sorted_weights = torch.empty((max_num_tokens_padded,), dtype=torch.float32, device=device)
    sorted_expert_ids = torch.full((max_num_m_blocks,), -1, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((2,), dtype=torch.int32, device=device)

    if num_local_tokens is not None:
        topk_ids = topk_ids[: num_local_tokens.item()]

    sorted_ids_begin = 0
    sorted_expert_ids_begin = 0
    skip_expert_num = 0
    for expertId in range(int(num_experts)):
        if expert_mask is not None and int(expert_mask[expertId].item()) == 0:
            skip_expert_num += 1
            continue
        token_id, topk_id = torch.where(topk_ids == expertId)
        tokensNum = int(token_id.numel())
        sorted_expert_ids_num = int((tokensNum + int(block_size) - 1) // int(block_size))
        tokensNumPad = int(sorted_expert_ids_num * int(block_size))
        sorted_ids[sorted_ids_begin : sorted_ids_begin + tokensNum] = (
            (topk_id.to(torch.int32) << 24) | token_id.to(torch.int32)
        )
        sorted_weights[sorted_ids_begin : sorted_ids_begin + tokensNum] = topk_weights[
            token_id, topk_id
        ].to(torch.float32)
        sorted_ids_begin = int(sorted_ids_begin + tokensNumPad)
        sorted_expert_ids[
            sorted_expert_ids_begin : sorted_expert_ids_begin + sorted_expert_ids_num
        ] = int(expertId - skip_expert_num)
        sorted_expert_ids_begin = int(sorted_expert_ids_begin + sorted_expert_ids_num)

    num_tokens_post_pad[0] = int(sorted_ids_begin)
    num_tokens_post_pad[1] = int(topk_ids.shape[0])

    return sorted_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad


@pytest.mark.parametrize(
    "tokens,model_dim,inter_dim,experts,topk,doweight_stage1",
    [
        (256, 1024, 256, 4, 2, False),
    ],
)
def _maybe_aiter_moe_sorting(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    model_dim: int,
    block_m: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids) or None."""
    if not HAS_AITER:
        return None
    try:
        # aiter expects i32 ids and fp32 weights
        topk_ids_i32 = topk_ids.to(torch.int32)
        topk_w_f32 = topk_weights.to(torch.float32)
        sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids, _moe_buf = aiter_moe_sorting(
            topk_ids_i32,
            topk_w_f32,
            num_experts,
            model_dim,
            torch.float16,
            block_m,
        )
        # `num_valid_ids` is documented as [1]; some builds allocate [2]. Keep the first element.
        if num_valid_ids.numel() > 1:
            num_valid_ids = num_valid_ids[:1].contiguous()
        return sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids
    except Exception:
        return None


RoutingBuffers = Tuple[
    torch.Tensor,  # sorted_token_ids
    torch.Tensor,  # sorted_weights
    torch.Tensor,  # sorted_expert_ids
    torch.Tensor,  # num_valid_ids (shape [1], i32)
    int,  # sorted_size
    int,  # blocks
]


def get_topk_valid_mask(topk_ids: torch.Tensor, expert_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Build valid_mask [tokens, topk] for (optional) EP-style masking.

    Mirrors `aiter.fused_moe.get_topk_valid_mask` semantics:
    - If expert_mask is None: all slots are valid (all ones)
    - Else: valid_mask[t, k] = expert_mask[topk_ids[t, k]] (cast to int8)
    """
    if expert_mask is None:
        return torch.ones(topk_ids.shape, dtype=torch.int8, device=topk_ids.device)
    return expert_mask[topk_ids].to(torch.int8)


def build_routing_buffers(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    experts: int,
    model_dim: int,
    tile_m: int,
    moe_sort_mode: Optional[str] = None,
) -> RoutingBuffers:
    """Build routing buffers once, reusable across stage1 + stage2.

    NOTE:
    - `moe_sort_mode="aiter"` aligns with `aiter/aiter/test_moe_flydsl.py` (swap path):
    - Use aiter's `moe_sorting` output directly (no host trim/pad of sorted buffers)
    - Launch full expert-block range; kernels use `num_valid_ids` to early-exit extra blocks
    - `moe_sort_mode="torch"` is a portable fallback when aiter isn't available:
      - Mirrors `aiter/op_tests/test_moe_sorting.py::moe_sorting_native` for consistent semantics
    """
    device = topk_ids.device
    default_mode = "aiter" if HAS_AITER else "torch"
    sort_mode = str(moe_sort_mode or os.environ.get("flydsl_MOE_SORT_MODE", default_mode)).lower().strip()
    if sort_mode not in ("aiter", "torch"):
        raise ValueError(f"invalid moe_sort_mode={sort_mode!r} (expected 'aiter' or 'torch')")

    if sort_mode == "torch":
        sorted_token_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad = moe_sorting_torch_native(
            topk_ids=topk_ids.to(torch.int32),
            topk_weights=topk_weights.to(torch.float32),
            num_experts=int(experts),
            block_size=int(tile_m),
        )
        # num_valid_ids[0] == total padded rows; kernels use this for early-exit.
        num_valid_ids = num_tokens_post_pad[:1].contiguous()
        sorted_size = int(sorted_token_ids.numel())
        blocks = int(sorted_expert_ids.numel())
        return (
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            sorted_size,
            blocks,
        )

    # aiter mode
    if not HAS_AITER:
        raise RuntimeError("aiter is not available; cannot build routing buffers (moe_sort_mode='aiter').")

    res = _maybe_aiter_moe_sorting(
        topk_ids,
        topk_weights,
        num_experts=experts,
        model_dim=model_dim,
        block_m=tile_m,
    )
    if res is None:
        raise RuntimeError("aiter moe_sorting failed/unavailable; cannot build routing buffers.")
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids = res

    # Keep moe_sorting outputs as-is (no host trim/pad). Launch full expert-block range.
    sorted_token_ids = sorted_token_ids.contiguous()
    sorted_weights = sorted_weights.contiguous()
    sorted_expert_ids = sorted_expert_ids.contiguous()
    sorted_size = int(sorted_token_ids.numel())
    blocks = int(sorted_expert_ids.numel())
    return (
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        sorted_size,
        blocks,
    )


# ---- Stage1/Stage2 runners (helpers; NOT pytest tests) ----
def run_moe_stage1(
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    *,
    in_dtype: str = "fp8",
    group_size: int = -1,
    seed: int = 0,
    num_iters: int = 5,
    num_warmup: int = 2,
    compare_aiter_ck: Optional[bool] = None,
    moe_sort_mode: Optional[str] = None,
    # Optional overrides (used by the 2-stage runner to avoid duplicated setup/sorting).
    x_fp32_in: Optional[torch.Tensor] = None,
    w1_fp32_in: Optional[torch.Tensor] = None,
    w2_fp32_in: Optional[torch.Tensor] = None,
    topk_ids_in: Optional[torch.Tensor] = None,
    topk_weights_in: Optional[torch.Tensor] = None,
    routing_in: Optional[RoutingBuffers] = None,
    return_outputs: bool = False,
    skip_ref: bool = False,
    w_fp4_kernel: bool = False,
    test_graph: bool = False,
    waves_per_eu: Optional[int] = None,
    debug_serial: bool = False,
    # Optional override for pre-built groupwise scale tensor [E, K//group_size, 2*inter_dim] (Opt 0 layout).
    scale_w1_groups_in: Optional[torch.Tensor] = None,
):
    assert model_dim % 64 == 0
    assert model_dim % tile_k == 0
    assert inter_dim % tile_n == 0

    device = torch.device("cuda")
    torch.manual_seed(int(seed))

    # Data: input and weights (aiter shapes)
    x_fp32 = (
        x_fp32_in
        if x_fp32_in is not None
        else torch.randn((tokens, model_dim), device=device, dtype=torch.float32)
    )
    w1_fp32 = (
        w1_fp32_in
        if w1_fp32_in is not None
        else torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32)
    )
    # w2 is required by aiter API even for stage1; keep it allocated to avoid null ptr.
    # Stage1 kernels should not touch it, but we allocate a correct-shape tensor for safety.
    w2_fp32 = (
        w2_fp32_in
        if w2_fp32_in is not None
        else torch.randn((experts, model_dim, inter_dim), device=device, dtype=torch.float32)
    )

    # Routing: aiter uses fused_topk; we use torch topk+softmax for portability/determinism.
    if topk_ids_in is None or topk_weights_in is None:
        score = torch.randn((tokens, experts), device=device, dtype=torch.float32)
        topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
        topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)
    else:
        topk_ids = topk_ids_in
        topk_weights = topk_weights_in

    routing = (
        routing_in
        if routing_in is not None
        else build_routing_buffers(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            experts=experts,
            model_dim=model_dim,
            tile_m=tile_m,
            moe_sort_mode=moe_sort_mode,
        )
    )
    (
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        sorted_size,
        blocks,
    ) = routing

    if in_dtype not in ("fp4", "fp8", "fp16", "bf16", "int8", "int8smooth", "int4", "int4_bf16"):
        raise ValueError(
            f"in_dtype must be one of ('fp4','fp8','fp16','bf16','int8','int8smooth','int4','int4_bf16'), got {in_dtype!r}"
        )
    is_fp4 = in_dtype == "fp4"
    is_int4 = in_dtype == "int4"
    is_int4_bf16 = in_dtype == "int4_bf16"  # W4A16: bf16 activations, packed int4 weights
    is_int8 = in_dtype in ("int8", "int8smooth", "int4")
    is_int8smooth = in_dtype == "int8smooth"

    # Quantize inputs / weights.
    if in_dtype == "fp4":
        x_q = fp4_utils.random_fp4_packed(tokens, model_dim, device=device)
        w1_q = fp4_utils.random_fp4_packed(experts * (2 * inter_dim), model_dim, device=device).view(
            experts, 2 * inter_dim, model_dim // 2
        )
        w2_q = fp4_utils.random_fp4_packed(experts * model_dim, inter_dim, device=device).view(
            experts, model_dim, inter_dim // 2
        )
        scale_x = fp4_utils.random_e8m0(tokens, model_dim // 32, device=device, low_exp=124, high_exp=126)
        scale_w1 = fp4_utils.random_e8m0(experts * (2 * inter_dim), model_dim // 32, device=device, low_exp=124, high_exp=126).view(
            experts, 2 * inter_dim, model_dim // 32
        )
    elif in_dtype == "fp8":
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=DTYPE_FP8)  # [tokens,K], [tokens,1]
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=DTYPE_FP8)  # [E,2*inter,K], [E,2*inter,1]
    # w2 is not used by our kernel, but required by aiter stage1 API
        w2_q, _scale_w2_unused = pertoken_quant(w2_fp32, quant_dtype=DTYPE_FP8)
    elif in_dtype == "fp16":
        x_q = x_fp32.to(torch.float16)
        w1_q = w1_fp32.to(torch.float16)
        w2_q = w2_fp32.to(torch.float16)
        scale_x = None
        scale_w1 = None
    elif in_dtype == "bf16":
        x_q = x_fp32.to(torch.bfloat16)
        w1_q = w1_fp32.to(torch.bfloat16)
        w2_q = w2_fp32.to(torch.bfloat16)
        scale_x = None
        scale_w1 = None
    elif in_dtype == "int8":
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.int8)
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.int8)
        w2_q, _scale_w2_unused = pertoken_quant(w2_fp32, quant_dtype=torch.int8)
    elif in_dtype == "int8smooth":
        # "SmoothQuant" emulation for MoE stage1:
        # - Create a per-expert smooth scale S[e, k] (k=model_dim)
        # - Expand X into per-route rows: X_route[t, slot, k] = X[t, k] * S[topk_ids[t,slot], k]
        # - Per-(t,slot) dynamic quant: int8 + scale_x[t,slot]
        #
        # This matches the kernel contract in kernels/moe_gemm_2stage.py for in_dtype="int8smooth",
        # where X/scale_x are indexed by (t*topk + slot).
        smooth_scale = (0.75 + 0.5 * torch.rand((experts, model_dim), device=device, dtype=torch.float32))
        x_route = x_fp32[:, None, :].expand(tokens, topk, model_dim)
        x_route = x_route * smooth_scale[topk_ids.to(torch.int64)]
        amax = torch.amax(torch.abs(x_route), dim=-1, keepdim=True)
        scale_x = amax / 127.0
        scale_x[scale_x == 0] = 1.0
        x_q = (x_route / scale_x).to(torch.int8)
        # Match moe_smoothquant layout: slot-major [topk*tokens, K].
        x_q = x_q.permute(1, 0, 2).contiguous()
        scale_x = scale_x.permute(1, 0, 2).contiguous()
        # W quantization is unchanged for this harness (same as aiter perf tests: smooth scales
        # exercise the API rather than implementing the exact SQ calibration workflow).
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.int8)
        w2_q, _scale_w2_unused = pertoken_quant(w2_fp32, quant_dtype=torch.int8)
    elif in_dtype == "int4_bf16":
        # W4A16: X is bf16 (no quant), W is int4 packed (host packs from int8 values in [-8,7]).
        x_q = x_fp32.to(torch.bfloat16)
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.int8, dtypeMax=7)
        w2_q, _scale_w2_unused = pertoken_quant(w2_fp32, quant_dtype=torch.int8, dtypeMax=7)
        scale_x = None
    else:
        # W4A8: X is int8, W is int4 packed (host packs from int8 values in [-8,7]).
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.int8)
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.int8, dtypeMax=7)
        w2_q, _scale_w2_unused = pertoken_quant(w2_fp32, quant_dtype=torch.int8, dtypeMax=7)

    # --- Groupwise scale for W4A16 ---
    use_groupwise_scale = is_int4_bf16 and group_size > 0
    scale_w1_groups = None  # [E, K//group_size, 2*inter_dim] for kernel (Opt 0 layout)
    if use_groupwise_scale:
        N_total = 2 * inter_dim
        num_groups_s1 = model_dim // group_size
        if scale_w1_groups_in is not None:
            scale_w1_groups = scale_w1_groups_in
        else:
            # Generate random groupwise scale [E, num_groups, N] (Opt 0: cache-friendly).
            scale_w1_groups = (
                torch.rand(experts, num_groups_s1, N_total, device=device, dtype=torch.float32)
                * 0.05 + 0.005
            )
        # Prepare scale for kernel (no-op shuffle, returns contiguous).
        scale_w1_prepared = shuffle_scale_for_int4(scale_w1_groups, group_size=group_size)
        # Override per-row scale (kernel uses groupwise scale instead).
        scale_w1 = None

    # Preshuffle weights on the *unpacked* tensor.
    w1_shuffled = w1_q if is_fp4 else shuffle_weight(w1_q)
    w2_shuffled = w2_q if is_fp4 else (shuffle_weight(w2_q) if in_dtype == "fp8" else None)

    # Flatten W1 for our FlyDSL kernel (treat expert dim as part of N).
    if is_fp4:
        w1_shuffled_flat = w1_shuffled.view(experts * (2 * inter_dim), model_dim // 2)
        w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim // 2)
        scale_w1_flat = None if scale_w1 is None else scale_w1.view(experts * (2 * inter_dim), model_dim // 32)
    else:
        w1_shuffled_flat = w1_shuffled.view(experts * (2 * inter_dim), model_dim)
        w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim)
        scale_w1_flat = None if scale_w1 is None else scale_w1.view(experts * (2 * inter_dim), 1)

    # No host-side padding: keep tensors contiguous and rely on kernel-side resource sizes / early-exit.
    x_q = (
        x_q.contiguous().view(tokens * topk, model_dim)
        if is_int8smooth
        else (
            x_q.contiguous().view(tokens, model_dim // 2)
            if is_fp4
            else x_q.contiguous().view(tokens, model_dim)
        )
    )
    # Pack weights for int4 variants (W4A8 and W4A16).
    # Both use the same interleaved packing: [ (v4<<4)|v0, (v5<<4)|v1, (v6<<4)|v2, (v7<<4)|v3 ]
    use_packed_int4 = is_int4 or is_int4_bf16
    w_kernel = (
        _pack_shuffled_int8_to_packed_int4_no_perm(w1_shuffled_flat) if use_packed_int4 else w1_shuffled_flat
    ).contiguous()
    if is_fp4:
        w_kernel = w_kernel.view(experts * (2 * inter_dim), model_dim // 2)
    elif not use_packed_int4:
        w_kernel = w_kernel.view(experts * (2 * inter_dim), model_dim)

    # Flatten scales to 1D memrefs (fp16 path uses 0-sized scale tensors; kernel ignores them).
    if scale_x is None:
        scale_x_1d = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        scale_x_1d = scale_x.view(-1).contiguous()
    if use_groupwise_scale:
        # Groupwise scale: flatten [E, num_groups, N] -> 1D for kernel memref.
        scale_w1_1d = scale_w1_prepared.view(-1).contiguous()
    elif scale_w1_flat is None:
        scale_w1_1d = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        scale_w1_1d = scale_w1_flat.view(-1).contiguous()
    sorted_weights_1d = sorted_weights.contiguous().view(-1)  # [sorted_size]

    # Output: [tokens, topk, inter_dim] fp16
    out = torch.empty((tokens, topk, inter_dim), device=device, dtype=torch.float16)

    from kernels.moe_gemm_2stage_gfx1250 import compile_moe_gemm1
    exe = compile_moe_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        in_dtype=in_dtype,
        group_size=group_size,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=bool(doweight_stage1),
        use_cshuffle_epilog=False,
        waves_per_eu=waves_per_eu,
    )

    def launch(o, x, w, sx, sw, st, eids, sw_sorted):
        stream = torch.cuda.current_stream()
        exe(
            o,
            x,
            w,
            sx,
            sw,
            st,
            eids,
            sw_sorted,
            num_valid_ids,
            tokens,
            inter_dim,
            model_dim,
            int(blocks),
            stream,
        )

    if _use_serial_fp4_perf(in_dtype, bool(debug_serial)):
        mode = "debug" if bool(debug_serial) else "safe-perf"
        print(f"[debug] using serial launch path for {in_dtype} stage1 ({mode})", flush=True)
        for _ in range(int(num_warmup)):
            launch(out, x_q, w_kernel, scale_x_1d, scale_w1_1d, sorted_token_ids, sorted_expert_ids, sorted_weights_1d)
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(int(max(1, num_iters))):
            launch(out, x_q, w_kernel, scale_x_1d, scale_w1_1d, sorted_token_ids, sorted_expert_ids, sorted_weights_1d)
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        us = (t1 - t0) * 1e6 / float(max(1, int(num_iters)))
    else:
        _, us = run_perftest(
            launch,
            out,
            x_q,
            w_kernel,
            scale_x_1d,
            scale_w1_1d,
            sorted_token_ids,
            sorted_expert_ids,
            sorted_weights_1d,
            num_iters=int(num_iters),
            num_warmup=int(num_warmup),
            testGraph=test_graph,
        )
    torch.cuda.synchronize()

    if not bool(skip_ref):
        if is_int8smooth:
            # x_q is slot-major [topk, tokens, K]; convert to [tokens, topk, K] for ref.
            x_ref = x_q.view(topk, tokens, model_dim).permute(1, 0, 2).contiguous()
            sx_ref = scale_x.view(topk, tokens, 1).permute(1, 0, 2).contiguous()
        else:
            x_ref = x_q
            sx_ref = scale_x
        ref = torch_moe_gemm1(
            x_ref,
            w1_q_flat,
            sx_ref,
            scale_w1_flat,
            topk_ids.to(torch.int64),
            topk_weights,
            inter_dim=inter_dim,
            doweight_stage1=doweight_stage1,
            group_size=group_size,
            scale_w1_groups=scale_w1_groups,
        )

        rtol = 0.5 if (is_int4 or is_int4_bf16) else 0.25
        atol = 0.5 if (is_int4 or is_int4_bf16) else 0.25
        assert verify_output(out.to(torch.float32), ref, rtol=rtol, atol=atol)

    # Note: kernel launches full expert-block range; effective work is gated by num_valid_ids.
    flops = 2 * tokens * topk * (2 * inter_dim) * model_dim
    tflops = flops / (us / 1e6) / 1e12

    # Rough bytes-moved accounting (same spirit as GEMM tests: count each tensor once).
    bytes_moved = 0
    x_elem_bytes = 2 if (is_int4_bf16 or in_dtype in ("bf16", "fp16")) else 1  # bf16/fp16 activations
    bytes_moved += (tokens * topk if is_int8smooth else tokens) * model_dim * x_elem_bytes  # x (bf16 for W4A16, else fp8/int8)
    bytes_moved += (experts * (2 * inter_dim) * model_dim) // (2 if use_packed_int4 else 1)  # w (packed for int4)
    bytes_moved += tokens * topk * inter_dim * 2  # out fp16 (logical)
    bytes_moved += (tokens * topk if is_int8smooth else tokens) * 4  # scale_x f32 (1D)
    bytes_moved += experts * (2 * inter_dim) * 4  # scale_w f32 (1D)
    bytes_moved += int(sorted_weights.numel()) * 4  # sorted_weights f32
    bytes_moved += int(sorted_token_ids.numel()) * 4  # sorted_token_ids i32
    bytes_moved += int(sorted_expert_ids.numel()) * 4  # sorted_expert_ids i32
    tbps = bytes_moved / 1e12 / (us / 1e6)

    print(
        f"FlyDSL MoE stage1[{in_dtype}]: "
        f"{us:.1f} us, "
        f"{tflops:.2f} TFLOPS(logical, M={tokens*topk}), "
        f"{tbps:.3f} TB/s (doweight_stage1={doweight_stage1})"
    )
    # Compare + benchmark vs aiter stage1 (optional; enabled by default when aiter is runnable).
    if compare_aiter_ck is None:
        compare_ck = os.environ.get("COMPARE_AITER_CK", "1" if HAS_AITER else "0") == "1"
    else:
        compare_ck = bool(compare_aiter_ck)
    # aiter paths are fp8-only in our setup.
    compare_ck = compare_ck and (in_dtype == "fp8")
    if compare_ck:
        if not HAS_AITER:
            pytest.skip("aiter not available; cannot compare to aiter moe stage1.", allow_module_level=False)
        try:
            from aiter.ops.moe_op import ck_moe_stage1_fwd
            from aiter.ops.enum import QuantType, ActivationType

            out_ck = torch.empty((tokens, topk, inter_dim), device=device, dtype=torch.float16)

            # aiter expects w1/w2 with expert dimension preserved.
            w1_ck = w1_shuffled
            w2_ck = w2_shuffled
            w1_scale_ck = scale_w1.contiguous()

            def launch_ck(o, x, w1_, w2_, sorted_ids_, sorted_eids_, num_valid_, w1_scale_, a1_scale_, sorted_w_):
                ck_moe_stage1_fwd(
                    hidden_states=x,
                    w1=w1_,
                    w2=w2_,
                    sorted_token_ids=sorted_ids_,
                    sorted_expert_ids=sorted_eids_,
                    num_valid_ids=num_valid_,
                    out=o,
                    topk=topk,
                    kernelName="",
                    w1_scale=w1_scale_,
                    a1_scale=a1_scale_,
                    block_m=tile_m,
                    sorted_weights=sorted_w_ if doweight_stage1 else None,
                    quant_type=QuantType.per_Token,
                    activation=ActivationType.Silu,
                    dst_type=o.dtype,
                )

            # Benchmark aiter stage1
            # Align with aiter swap rules:
            # - aiter takes quantized activations (fp8) + per-token scale
            # - routing buffers are used as-is (no host trim/pad); launch range is sorted_eids.numel()
            _, us_ck = run_perftest(
                launch_ck,
                out_ck,
                x_q,  # fp8 activations
                w1_ck,
                w2_ck,
                sorted_token_ids,
                sorted_expert_ids,
                num_valid_ids,
                w1_scale_ck,
                scale_x_1d,  # [tokens]
                sorted_weights,
                num_iters=int(num_iters),
                num_warmup=int(num_warmup),
                testGraph=test_graph,
            )

            # Correctness: FlyDSL vs aiter
            assert verify_output(out.to(torch.float32), out_ck.to(torch.float32), rtol=0.25, atol=0.25, msg="FlyDSL vs aiter:")

            # Perf print: use the same flop model for both
            flops = 2 * tokens * topk * (2 * inter_dim) * model_dim
            tflops_ck = flops / (us_ck / 1e6) / 1e12
            print(f"[aiter] stage1: {us_ck:.1f} us, {tflops_ck:.2f} TFLOPS, FlyDSL vs aiter speedups: {tflops / tflops_ck:.2f}x")
        except Exception as e:
            # Treat aiter compare as best-effort: many environments can import `aiter` but can't load
            # the full JIT .so dependency chain. Don't fail the FlyDSL test suite for that.
            logging.warning(f"Skipping aiter moe stage1 compare (not runnable here): {e}")
    if return_outputs:
        return out, us
    return None


def run_moe_stage2(
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    *,
    in_dtype: str = "fp8",
    out_dtype: str = "f16",
    group_size: int = -1,
    seed: int = 0,
    num_iters: int = 5,
    num_warmup: int = 2,
    compare_aiter_ck: Optional[bool] = None,
    moe_sort_mode: Optional[str] = None,
    # Optional overrides (used by the 2-stage runner to avoid duplicated setup/sorting).
    x_fp32_in: Optional[torch.Tensor] = None,
    w1_fp32_in: Optional[torch.Tensor] = None,
    w2_fp32_in: Optional[torch.Tensor] = None,
    topk_ids_in: Optional[torch.Tensor] = None,
    topk_weights_in: Optional[torch.Tensor] = None,
    routing_in: Optional[RoutingBuffers] = None,
    a2_fp8_in: Optional[torch.Tensor] = None,
    a2_scale_in: Optional[torch.Tensor] = None,
    return_outputs: bool = False,
    skip_ref: bool = False,
    init_scale: float = 0.2,
    # Custom compile function for kernel comparison (default: compile_moe_gemm2).
    compile_fn=None,
    # Kernel name for logging (default: "moe_gemm2").
    kernel_name: str = "moe_gemm2",
    # Use reduce mode (accumulate=False) instead of atomic mode.
    use_reduce: bool = False,
    # Use valid mask for optimizationwhen reduce or not
    use_valid_mask: bool = False,
    # graph mode
    test_graph: bool = False,
    waves_per_eu: Optional[int] = None,
    debug_serial: bool = False,
    # Optional override for pre-built groupwise scale tensor [E, inter_dim//group_size, model_dim] (Opt 0 layout).
    scale_w2_groups_in: Optional[torch.Tensor] = None,
):
    """MoE stage2 (gemm2): out2[t] = sum_{slot} ( out1[t,slot] @ W2[expert]^T ) with optional routed weight."""

    # Parameter sanity checks with actionable hints (avoid bare AssertionError).
    if model_dim % tile_n != 0:
        raise ValueError(
            f"Invalid stage2 tiling: model_dim ({model_dim}) must be divisible by tile_n2 ({tile_n})."
        )
    if inter_dim % tile_k != 0:
        # Stage2 tile_k is tile_k2 (K dimension = inter_dim).
        # In kernels/moe_gemm_2stage.py, total_threads is fixed to 256 and there is no K-tail.
        raise ValueError(
            "Invalid stage2 tiling: inter_dim ({inter_dim}) must be divisible by tile_k2 ({tile_k}). "
            "Try setting `--tile_k2` to a divisor of inter_dim. "
            "Tip: stage2 splits A2 loads across 256 threads; if you want smaller tile_k2, you may need a larger tile_m so (tile_m*tile_k2) stays divisible by 1024."
            .format(inter_dim=inter_dim, tile_k=tile_k)
        )
    # Enforce the kernel's stage2 gmem->reg load mapping constraints.
    # See: kernels/moe_gemm_2stage.py::compile_moe_gemm2 (x_load_bytes selection).
    if (tile_m * tile_k) % 256 != 0:
        raise ValueError(
            f"Invalid stage2 tiling: tile_m*tile_k2 must be divisible by 256 (total_threads=256). "
            f"Got tile_m={tile_m}, tile_k2={tile_k} -> tile_m*tile_k2={tile_m * tile_k}."
        )
    bytes_per_thread_x = (tile_m * tile_k) // 256  # 1B elements
    if bytes_per_thread_x % 4 != 0:
        raise ValueError(
            f"Invalid stage2 tiling for gmem loads: bytes_per_thread_x ((tile_m*tile_k2)/256) must be divisible by 4. "
            f"Got tile_m={tile_m}, tile_k2={tile_k} -> bytes_per_thread_x={bytes_per_thread_x}. "
        )

    # Default compile function.
    if compile_fn is None:
        if use_reduce:
            compile_fn = _make_reduce_mode_compile_fn(use_flydsl_reduce=True, use_valid_mask=bool(use_valid_mask))
        else:
            compile_fn = compile_moe_gemm2

    device = torch.device("cuda")
    torch.manual_seed(int(seed))

    s = float(init_scale)

    # Data: input and weights (aiter shapes)
    x_fp32 = (
        x_fp32_in
        if x_fp32_in is not None
        else torch.rand((tokens, model_dim), device=device, dtype=torch.float32) * s
    )
    w1_fp32 = (
        w1_fp32_in
        if w1_fp32_in is not None
        else torch.rand((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * (s / math.sqrt(model_dim))
    )
    w2_fp32 = (
        w2_fp32_in
        if w2_fp32_in is not None
        else torch.rand((experts, model_dim, inter_dim), device=device, dtype=torch.float32) * (s / math.sqrt(inter_dim))
    )

    # Routing: deterministic torch topk + softmax.
    if topk_ids_in is None or topk_weights_in is None:
        score = torch.rand((tokens, experts), device=device, dtype=torch.float32)
        topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
        topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)
    else:
        topk_ids = topk_ids_in
        topk_weights = topk_weights_in

    routing = (
        routing_in
        if routing_in is not None
        else build_routing_buffers(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            experts=experts,
            model_dim=model_dim,
            tile_m=tile_m,
            moe_sort_mode=moe_sort_mode,
        )
    )
    (
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        sorted_size,
        blocks,
    ) = routing
    # NOTE: routing uses `moe_sorting` output directly (no host trim/pad). Extra launched blocks
    # are gated by `num_valid_ids` inside the kernels.

    if in_dtype not in ("fp4", "fp8", "fp16", "bf16", "int8", "int8smooth", "int4", "int4_bf16"):
        raise ValueError(
            f"in_dtype must be one of ('fp4','fp8','fp16','bf16','int8','int8smooth','int4','int4_bf16'), got {in_dtype!r}"
        )
    is_fp4 = in_dtype == "fp4"
    is_int4 = in_dtype == "int4"
    is_int4_bf16 = in_dtype == "int4_bf16"  # W4A16: bf16 activations, packed int4 weights
    is_int8 = in_dtype in ("int8", "int8smooth", "int4")
    is_int8smooth = in_dtype == "int8smooth"

    # Quantize inputs / weights.
    if in_dtype == "fp4":
        x_q = fp4_utils.random_fp4_packed(tokens, model_dim, device=device)
        w1_q = fp4_utils.random_fp4_packed(experts * (2 * inter_dim), model_dim, device=device).view(
            experts, 2 * inter_dim, model_dim // 2
        )
        w2_q = fp4_utils.random_fp4_packed(experts * model_dim, inter_dim, device=device).view(
            experts, model_dim, inter_dim // 2
        )
        scale_x = fp4_utils.random_e8m0(tokens, model_dim // 32, device=device, low_exp=124, high_exp=126)
        scale_w1 = fp4_utils.random_e8m0(experts * (2 * inter_dim), model_dim // 32, device=device, low_exp=124, high_exp=126).view(
            experts, 2 * inter_dim, model_dim // 32
        )
        scale_w2 = fp4_utils.random_e8m0(experts * model_dim, inter_dim // 32, device=device, low_exp=124, high_exp=126).view(
            experts, model_dim, inter_dim // 32
        )
    elif in_dtype == "fp8":
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=DTYPE_FP8)
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=DTYPE_FP8)
        w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=DTYPE_FP8)
    elif in_dtype == "fp16":
        x_q = x_fp32.to(torch.float16)
        w1_q = w1_fp32.to(torch.float16)
        w2_q = w2_fp32.to(torch.float16)
        scale_x = None
        scale_w1 = None
        scale_w2 = None
    elif in_dtype == "bf16":
        x_q = x_fp32.to(torch.bfloat16)
        w1_q = w1_fp32.to(torch.bfloat16)
        w2_q = w2_fp32.to(torch.bfloat16)
        scale_x = None
        scale_w1 = None
        scale_w2 = None
    elif in_dtype == "int8":
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.int8)
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.int8)
        w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=torch.int8)
    elif in_dtype == "int8smooth":
        # Stage2 uses token-slot activations (A2) already, so we keep the same W8A8 kernel path.
        # We optionally apply a per-expert smooth scale to A2 *before* quantization below.
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.int8)
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.int8)
        w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=torch.int8)
    elif in_dtype == "int4_bf16":
        # W4A16: X is bf16 (no quant), W is int4 packed (host packs from int8 values in [-8,7]).
        x_q = x_fp32.to(torch.bfloat16)
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.int8, dtypeMax=7)
        w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=torch.int8, dtypeMax=7)
        scale_x = None
    else:
        # W4A8: A2 is int8, W2 is int4 packed (host packs from int8 values in [-8,7]).
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.int8)
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.int8, dtypeMax=7)
        w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=torch.int8, dtypeMax=7)

    # --- Groupwise scale for W4A16 (stage 2) ---
    use_groupwise_scale = is_int4_bf16 and group_size > 0
    scale_w2_groups = None  # [E, inter_dim//group_size, model_dim] Opt 0 layout
    if use_groupwise_scale:
        num_groups_s2 = inter_dim // group_size
        if scale_w2_groups_in is not None:
            scale_w2_groups = scale_w2_groups_in
        else:
            # Generate random groupwise scale [E, num_groups, N] (Opt 0: cache-friendly).
            scale_w2_groups = (
                torch.rand(experts, num_groups_s2, model_dim, device=device, dtype=torch.float32)
                * 0.05 + 0.005
            )
        scale_w2_prepared = shuffle_scale_for_int4(scale_w2_groups, group_size=group_size)
        # Override per-row scale (kernel uses groupwise scale instead).
        scale_w2 = None

    # Preshuffle weights on the *unpacked* tensor.
    w1_shuffled = w1_q if is_fp4 else shuffle_weight(w1_q)
    w2_shuffled = w2_q if is_fp4 else shuffle_weight(w2_q)

    # Stage2 input (A2): either provided (gemm1->quantize chaining) or built from stage1 reference.
    # For int4_bf16, A2 is bf16 (same as fp16 for scale handling).
    if a2_fp8_in is not None and (a2_scale_in is not None or in_dtype in ("fp16", "bf16", "int4_bf16")):
        a2_q = a2_fp8_in
        a2_scale = a2_scale_in
    else:
        if is_fp4:
            w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim // 2)
            scale_w1_flat = None if scale_w1 is None else scale_w1.view(experts * (2 * inter_dim), model_dim // 32)
        else:
            w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim)
            scale_w1_flat = None if scale_w1 is None else scale_w1.view(experts * (2 * inter_dim), 1)
        # Build stage2 input via reference stage1 only when correctness is enabled.
        if bool(skip_ref):
            raise RuntimeError(
                "run_moe_stage2(skip_ref=True) requires providing a2_fp8_in and a2_scale_in "
                "(so we don't have to run the huge torch reference stage1)."
            )
        out1_ref = torch_moe_gemm1(
            x_q,
            w1_q_flat,
            scale_x,
            scale_w1_flat,
            topk_ids.to(torch.int64),
            topk_weights,
            inter_dim=inter_dim,
            doweight_stage1=bool(doweight_stage1),
        )  # [tokens, topk, inter] fp32
        if in_dtype == "fp8":
            a2_q, a2_scale = pertoken_quant(out1_ref, quant_dtype=DTYPE_FP8)
        elif in_dtype == "fp4":
            a2_q = fp4_utils.random_fp4_packed(tokens * topk, inter_dim, device=device)
            a2_scale = fp4_utils.random_e8m0(tokens * topk, inter_dim // 32, device=device, low_exp=124, high_exp=126)
        elif in_dtype == "fp16":
            a2_q = out1_ref.to(torch.float16)
            a2_scale = None
        elif in_dtype == "bf16":
            a2_q = out1_ref.to(torch.bfloat16)
            a2_scale = None
        elif in_dtype == "int4_bf16":
            # W4A16: A2 is bf16 (no quant).
            a2_q = out1_ref.to(torch.bfloat16)
            a2_scale = None
        else:
            if is_int8smooth:
                # Apply a per-expert smooth scale to A2 before W8A8 quantization.
                smooth_scale2 = (0.75 + 0.5 * torch.rand((experts, inter_dim), device=device, dtype=torch.float32))
                out1_ref = out1_ref * smooth_scale2[topk_ids.to(torch.int64)]
            a2_q, a2_scale = pertoken_quant(out1_ref, quant_dtype=torch.int8)

    # Flatten weights/scales for the kernel.
    if is_fp4:
        w2_shuffled_flat = w2_shuffled.view(experts * model_dim, inter_dim // 2)
        scale_w2_flat = None if scale_w2 is None else scale_w2.view(experts * model_dim, inter_dim // 32)
    else:
        w2_shuffled_flat = w2_shuffled.view(experts * model_dim, inter_dim)
        scale_w2_flat = None if scale_w2 is None else scale_w2.view(experts * model_dim, 1)

    # For W4A8 and W4A16, pack preshuffled int8 weights into packed int4 bytes.
    # Both use the same interleaved packing: [ (v4<<4)|v0, (v5<<4)|v1, (v6<<4)|v2, (v7<<4)|v3 ]
    use_packed_int4 = is_int4 or is_int4_bf16
    w2_kernel = w2_shuffled_flat
    if use_packed_int4:
        w2_kernel = _pack_shuffled_int8_to_packed_int4_no_perm(w2_shuffled_flat)

    w2_flat = w2_kernel.contiguous().view(-1)
    w2_kernel = w2_flat
    if is_fp4:
        w2_kernel = w2_kernel.view(experts * model_dim, inter_dim // 2)
    elif not use_packed_int4:
        w2_kernel = w2_kernel.view(experts * model_dim, inter_dim)

    # Flatten scales to 1D memrefs (fp16 path uses 0-sized scale tensors; kernel ignores them).
    if a2_scale is None:
        a2_scale_1d = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        a2_scale_1d = a2_scale.view(-1).contiguous()
    if use_groupwise_scale:
        w2_scale_1d = scale_w2_prepared.view(-1).contiguous()
    elif scale_w2_flat is None:
        w2_scale_1d = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        w2_scale_1d = scale_w2_flat.view(-1).contiguous()
    sorted_weights_1d = sorted_weights.contiguous().view(-1)  # [sorted_size]

    out_s = str(out_dtype).strip().lower()
    if out_s in ("f16", "fp16", "half"):
        out_torch_dtype = torch.float16
    elif out_s in ("f32", "fp32", "float"):
        out_torch_dtype = torch.float32
    else:
        raise ValueError(f"out_dtype must be 'f16' or 'f32', got {out_dtype!r}")

    out = torch.zeros((tokens, model_dim), device=device, dtype=out_torch_dtype)
    out_perf = torch.zeros_like(out)

    doweight_stage2 = not bool(doweight_stage1)
    compile_kwargs = dict(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        group_size=group_size,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=bool(doweight_stage2),
    )
    if waves_per_eu is not None:
        compile_kwargs["waves_per_eu"] = waves_per_eu
    try:
        exe = compile_fn(**compile_kwargs)
    except TypeError:
        # Some wrapper compile_fns (e.g. local reduce wrappers) may not expose
        # waves_per_eu; retry without it to keep compatibility.
        compile_kwargs.pop("waves_per_eu", None)
        exe = compile_fn(**compile_kwargs)
    is_reduce_exe = (getattr(exe, "mode", None) == MoeGemm2Mode.REDUCE) or bool(use_reduce)

    def launch(o, x, w, sx, sw, st, eids, sw_sorted):
        stream = torch.cuda.current_stream()
        valid_mask = None
        if is_reduce_exe and bool(use_valid_mask):
            # Default: non-EP (all ones). EP mode can be emulated by passing expert_mask.
            valid_mask = get_topk_valid_mask(topk_ids, expert_mask=None).contiguous()
        if is_reduce_exe:
            exe(
                o,
                x,
                w,
                sx,
                sw,
                st,
                eids,
                sw_sorted,
                num_valid_ids,
                tokens,
                model_dim,
                inter_dim,
                int(blocks),
                valid_mask,
                stream,
            )
        else:
            # Atomic mode does not take valid_mask.
            exe(
                o,
                x,
                w,
                sx,
                sw,
                st,
                eids,
                sw_sorted,
                num_valid_ids,
                tokens,
                model_dim,
                inter_dim,
                int(blocks),
                stream,
            )
 
    # NOTE: stage2 uses atomic-add into `out`, so we cannot reuse the same output buffer
    # across perf iterations for correctness. Time into a dedicated buffer, then run
    # a single clean launch for correctness verification below.
    if _use_serial_fp4_perf(in_dtype, bool(debug_serial)):
        mode = "debug" if bool(debug_serial) else "safe-perf"
        print(f"[debug] using serial launch path for {in_dtype} stage2 ({mode})", flush=True)
        for _ in range(int(num_warmup)):
            out_perf.zero_()
            launch(
                out_perf,
                a2_q.view(-1),
                w2_kernel.view(-1),
                a2_scale_1d,
                w2_scale_1d,
                sorted_token_ids,
                sorted_expert_ids,
                sorted_weights_1d,
            )
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(int(max(1, num_iters))):
            out_perf.zero_()
            launch(
                out_perf,
                a2_q.view(-1),
                w2_kernel.view(-1),
                a2_scale_1d,
                w2_scale_1d,
                sorted_token_ids,
                sorted_expert_ids,
                sorted_weights_1d,
            )
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        us = (t1 - t0) * 1e6 / float(max(1, int(num_iters)))
    else:
        _, us = run_perftest(
            launch,
            out_perf,
            a2_q.view(-1),
            w2_kernel.view(-1),
            a2_scale_1d,
            w2_scale_1d,
            sorted_token_ids,
            sorted_expert_ids,
            sorted_weights_1d,
            num_iters=int(num_iters),
            num_warmup=int(num_warmup),
            testGraph=test_graph,
        )
    torch.cuda.synchronize()

    # Correctness run (single launch into a clean zeroed output).
    out.zero_()
    launch(
        out,
        a2_q.view(-1),
        w2_kernel.view(-1),
        a2_scale_1d,
        w2_scale_1d,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights_1d,
    )
    torch.cuda.synchronize()

    if not bool(skip_ref):
        ref2 = torch_moe_gemm2(
            a2_q,
            w2_q,
            a2_scale,
            scale_w2,
            topk_ids.to(torch.int64),
            topk_weights,
            model_dim=model_dim,
            doweight_stage2=doweight_stage2,
            group_size=group_size,
            scale_w2_groups=scale_w2_groups,
        )
        assert verify_output(out.to(torch.float32), ref2, rtol=0.5, atol=0.5)

    # Launches full expert-block range; effective work is gated by num_valid_ids.
    flops = 2 * tokens * topk * model_dim * inter_dim
    tflops = flops / (us / 1e6) / 1e12

    bytes_moved = 0
    a2_elem_bytes = 2 if in_dtype in ("int4_bf16", "bf16", "fp16") else 1  # bf16/fp16 activations
    bytes_moved += tokens * topk * inter_dim * a2_elem_bytes  # a2 (logical)
    bytes_moved += (experts * model_dim * inter_dim) // (2 if is_int4 else 1)  # w2 (packed for int4)
    bytes_moved += tokens * model_dim * (2 if out_torch_dtype == torch.float16 else 4)  # out
    bytes_moved += tokens * topk * 4  # a2_scale f32 (logical)
    bytes_moved += experts * model_dim * 4  # w2_scale f32 (1D)
    bytes_moved += int(sorted_weights.numel()) * 4
    bytes_moved += int(sorted_token_ids.numel()) * 4
    bytes_moved += int(sorted_expert_ids.numel()) * 4
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(
        f"FlyDSL MoE stage2 [{kernel_name}] {in_dtype} {'reduce' if use_reduce else 'atomic'} | "
        f"{model_dim}x{inter_dim}, E={experts}, K={topk}, M_eff={tokens*topk} | "
        f"{us:.1f} us, {tflops:.2f} TFLOPS, {tbps:.3f} TB/s"
    )
    # Optional compare vs aiter stage2.
    if compare_aiter_ck is None:
        compare_ck = os.environ.get("COMPARE_AITER_CK", "1" if HAS_AITER else "0") == "1"
    else:
        compare_ck = bool(compare_aiter_ck)
    # aiter paths are fp8-only in our setup.
    compare_ck = compare_ck and (in_dtype == "fp8")
    if compare_ck:
        if not HAS_AITER:
            pytest.skip("aiter not available; cannot compare to aiter moe stage2.", allow_module_level=False)
        try:
            from aiter.ops.moe_op import ck_moe_stage2_fwd
            from aiter.ops.enum import QuantType, ActivationType

            # aiter stage2 output type is fp16 in many builds; keep fp16 for compatibility.
            # (Some environments don't accept fp32 output tensors here.)
            out_ck = torch.zeros((tokens, model_dim), device=device, dtype=torch.float16)
            out_ck_perf = torch.zeros_like(out_ck)

            def launch_ck(o, a2_, w1_, w2_, sorted_ids_, sorted_eids_, num_valid_, w2_scale_, a2_scale_, sorted_w_):
                ck_moe_stage2_fwd(
                    inter_states=a2_,
                    w1=w1_,
                    w2=w2_,
                    sorted_token_ids=sorted_ids_,
                    sorted_expert_ids=sorted_eids_,
                    num_valid_ids=num_valid_,
                    out=o,
                    topk=topk,
                    kernelName="",
                    w2_scale=w2_scale_,
                    a2_scale=a2_scale_,
                    block_m=tile_m,
                    sorted_weights=sorted_w_ if doweight_stage2 else None,
                    quant_type=QuantType.per_Token,
                    activation=ActivationType.Silu,
                )

            _, us_ck = run_perftest(
                launch_ck,
                out_ck_perf,
                a2_q,
                w1_shuffled,  # stage2 signature includes w1; provide preshuffled tensor
                w2_shuffled,
                sorted_token_ids,
                sorted_expert_ids,
                num_valid_ids,
                scale_w2.contiguous(),
                a2_scale.contiguous(),
                sorted_weights,
                num_iters=int(num_iters),
                num_warmup=int(num_warmup),
                testGraph=test_graph,
            )

            # Perf print (report both executed vs logical FLOPs, same convention as FlyDSL).
            flops = 2 * tokens * topk * model_dim * inter_dim
            tflops_ck = flops / (us_ck / 1e6) / 1e12
            print(
                f"[aiter] stage2: {us_ck:.1f} us, "
                f"{tflops_ck:.2f} TFLOPS(logical, M={tokens*topk}), FlyDSL vs aiter speedups: {tflops / tflops_ck:.2f}x"
            )

            # Correctness run (best-effort; do not fail perf comparison if aiter diverges).
            out_ck.zero_()
            launch_ck(
                out_ck,
                a2_q,
                w1_shuffled,
                w2_shuffled,
                sorted_token_ids,
                sorted_expert_ids,
                num_valid_ids,
                scale_w2.contiguous(),
                a2_scale.contiguous(),
                sorted_weights,
            )
            torch.cuda.synchronize()
            if not verify_output(out.to(torch.float32), out_ck.to(torch.float32), rtol=0.5, atol=0.5, msg="[aiter] stage2:"):
                    logging.warning("[aiter] stage2 correctness mismatch vs FlyDSL (continuing; perf numbers still printed).")
        except Exception as e:
            logging.warning(f"Skipping aiter moe stage2 compare (not runnable here): {e}")

    # Print profile breakdown if the executor supports it
    if hasattr(exe, 'print_profile_stats'):
        exe.print_profile_stats()

    if return_outputs:
        return out, us
    return None


@pytest.mark.parametrize("waves_per_eu", [1, 2], ids=["wpe1", "wpe2"])
def test_moe_2stage_waves_per_eu_smoke(waves_per_eu: int):
    """Smoke test for stage1/stage2 waves_per_eu plumbing on gfx1250."""
    tokens = 32
    model_dim = 256
    inter_dim = 128
    experts = 4
    topk = 2
    tile_m = 16
    tile_n1 = 64
    tile_k1 = 128
    tile_n2 = 64
    tile_k2 = 128
    stage1_out, _ = run_moe_stage1(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n1,
        tile_k=tile_k1,
        doweight_stage1=False,
        in_dtype="fp16",
        num_iters=1,
        num_warmup=1,
        compare_aiter_ck=False,
        return_outputs=True,
        skip_ref=True,
        waves_per_eu=waves_per_eu,
    )

    stage2_out, _ = run_moe_stage2(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n2,
        tile_k=tile_k2,
        doweight_stage1=False,
        in_dtype="fp16",
        out_dtype="f16",
        num_iters=1,
        num_warmup=1,
        compare_aiter_ck=False,
        return_outputs=True,
        skip_ref=True,
        a2_fp8_in=stage1_out.to(torch.float16),
        a2_scale_in=None,
        waves_per_eu=waves_per_eu,
    )

    assert not torch.isnan(stage1_out).any(), "stage1 output contains NaN (kernel computation error)"
    assert not torch.isnan(stage2_out).any(), "stage2 output contains NaN (kernel computation error)"


@pytest.mark.parametrize("use_reduce", [False, True], ids=["atomic", "reduce"])
def test_moe_2stage_fp4_smoke(use_reduce: bool):
    """Smoke test for gfx1250 fp4 stage1/stage2 path."""
    tokens = 32
    model_dim = 256
    inter_dim = 128
    experts = 4
    topk = 2
    tile_m = 16
    tile_n1 = 64
    tile_k1 = 128
    tile_n2 = 64
    tile_k2 = 128

    stage1_out, _ = run_moe_stage1(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n1,
        tile_k=tile_k1,
        doweight_stage1=False,
        in_dtype="fp4",
        num_iters=1,
        num_warmup=1,
        compare_aiter_ck=False,
        return_outputs=True,
        skip_ref=True,
    )

    # Stage2 fp4 expects packed activations + e8m0 scales.
    a2_fp4 = fp4_utils.random_fp4_packed(tokens * topk, inter_dim, device=stage1_out.device)
    a2_scale = fp4_utils.random_e8m0(tokens * topk, inter_dim // 32, device=stage1_out.device)
    stage2_out, _ = run_moe_stage2(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n2,
        tile_k=tile_k2,
        doweight_stage1=False,
        in_dtype="fp4",
        out_dtype="f16",
        num_iters=1,
        num_warmup=1,
        compare_aiter_ck=False,
        return_outputs=True,
        skip_ref=True,
        a2_fp8_in=a2_fp4,
        a2_scale_in=a2_scale,
        use_reduce=bool(use_reduce),
    )

    assert not torch.isnan(stage1_out).any(), "stage1 output contains NaN (kernel computation error)"
    assert not torch.isnan(stage2_out).any(), "stage2 output contains NaN (kernel computation error)"


@pytest.mark.parametrize(
    "tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n1, tile_k1, tile_n2, tile_k2, doweight_stage1",
    [
        # Small smoke (fast compile + run) for all in_dtype.
        pytest.param(64, 256, 128, 4, 2, 16, 64, 128, 64, 128, False, id="S"),
        # Medium (more realistic) for all in_dtype (skip_ref will auto-enable).
        pytest.param(129, 1024, 256, 8, 2, 32, 128, 128, 128, 128, False, id="M"),
        # Large (aiter-style) mainly for perf smoke; reference is too expensive here.
        pytest.param(333, 4096, 2048, 17, 9, 64, 128, 128, 256, 128, False, id="L", marks=pytest.mark.large_shape),
    ],
)
@pytest.mark.parametrize("in_dtype", ["fp4", "fp8", "fp16", "bf16", "int8", "int8smooth", "int4", "int4_bf16"])
@pytest.mark.parametrize("out_dtype", ["f16", "f32"], ids=["out_f16", "out_f32"])
@pytest.mark.parametrize("use_reduce", [False, True], ids=["atomic", "reduce"])
@pytest.mark.parametrize("use_valid_mask", [False, True], ids=["nomask", "mask"])
@pytest.mark.parametrize("test_graph", [
    pytest.param(False, id="eager"),
    pytest.param(True, id="graph"),
])
@pytest.mark.parametrize("group_size", [-1, 32], ids=["perrow", "g32"])
def test_moe_gemm_2stage(
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n1: int,
    tile_k1: int,
    tile_n2: int,
    tile_k2: int,
    doweight_stage1: bool,
    in_dtype: str,
    out_dtype: str,
    use_reduce: bool,
    use_valid_mask: bool,
    test_graph: bool,
    group_size: int,
    *,
    seed: int = 0,
    num_iters: int = 5,
    num_warmup: int = 2,
    moe_sort_mode: Optional[str] = None,
    compare_aiter_ck: Optional[bool] = None,
    init_scale: float = 1.0,
    skip_ref: bool = False,
    w_fp4_kernel: bool = False,
):
    """Single 2-stage test: gemm1 -> quantize -> gemm2, with routing built once.

    When in_dtype='int4_bf16' and group_size>0, uses groupwise scale (W4A16 with per-group dequant).
    """
    if (not bool(use_reduce)) and bool(use_valid_mask):
        pytest.skip("valid_mask is only used in reduce mode (atomic mode ignores it).")
    if in_dtype == "fp4":
        # Keep fp4 in the main parameterized matrix, but only run the small shape
        # to avoid excessive compile/runtime during broad CI sweeps.
        is_small_shape = (
            tokens == 64
            and model_dim == 256
            and inter_dim == 128
            and experts == 4
            and topk == 2
        )
        if not is_small_shape:
            pytest.skip("fp4 in main matrix is enabled only for the small shape.")
    out_s = str(out_dtype).strip().lower()
    if in_dtype == "fp4" and out_s in ("f32", "fp32", "float"):
        pytest.skip("fp4 stage2 kernel does not support out_dtype='f32'.")
    if bool(use_reduce) and out_s in ("f32", "fp32", "float"):
        pytest.skip("reduce mode does not support out_dtype='f32' (compile_moe_gemm2(accumulate=False) forbids it).")
    if group_size > 0 and in_dtype != "int4_bf16":
        pytest.skip("groupwise scale only applies to int4_bf16")
    device = torch.device("cuda")
    # torch.manual_seed(int(seed))

    # Keep inputs tame by default; fp16 paths are less robust to overflow.
    # (Callers can still override via pytest param / direct invocation.)
    if init_scale == 1.0:
        init_scale = 0.2
    s = float(init_scale)
    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32) * s
    # x_fp32 = torch.ones((tokens, model_dim), device=device, dtype=torch.float32) * s
    # fan_in = model_dim for W1: [E, 2*inter, model]
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * s #* (s / math.sqrt(model_dim))
    # w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * 0.2
    # w1_fp32 = torch.ones((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * s
    # fan_in = inter_dim for W2: [E, model, inter]
    w2_fp32 = torch.randn((experts, model_dim, inter_dim), device=device, dtype=torch.float32) * (s / math.sqrt(inter_dim))

    score = torch.rand((tokens, experts), device=device, dtype=torch.float32)
    topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)

    routing = build_routing_buffers(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        experts=experts,
        model_dim=model_dim,
        tile_m=tile_m,
        moe_sort_mode=moe_sort_mode,
    )

    # Default routing + comparison knobs for test stability:
    # - Use torch routing (no aiter dependency for sorting).
    # - Only compare aiter for fp8, and only when explicitly requested.
    if moe_sort_mode is None:
        moe_sort_mode = "torch"
    if compare_aiter_ck is None:
        compare_aiter_ck = False

    fp4_skip_ref = (in_dtype == "fp4") or bool(skip_ref)
    out1_fp16, _us1 = run_moe_stage1(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        in_dtype=in_dtype,
        group_size=group_size,
        tile_m=tile_m,
        tile_n=tile_n1,
        tile_k=tile_k1,
        doweight_stage1=bool(doweight_stage1),
        seed=seed,
        num_iters=num_iters,
        num_warmup=num_warmup,
        compare_aiter_ck=bool(compare_aiter_ck) and (in_dtype == "fp8"),
        moe_sort_mode=moe_sort_mode,
        x_fp32_in=x_fp32,
        w1_fp32_in=w1_fp32,
        w2_fp32_in=w2_fp32,
        topk_ids_in=topk_ids,
        topk_weights_in=topk_weights,
        routing_in=routing,
        return_outputs=True,
        skip_ref=fp4_skip_ref,
        w_fp4_kernel=w_fp4_kernel,
        test_graph=test_graph,
    )

    if w_fp4_kernel:
        if in_dtype == "fp4":
            # Stage2 FP4 kernel expects packed FP4 + E8M0 scale input.
            # For benchmark-only mode (skip_ref), use synthetic A2 quant tensors.
            a2_q = fp4_utils.random_fp4_packed(tokens * topk, inter_dim, device=device).view(tokens, topk, inter_dim // 2)
            a2_scale = fp4_utils.random_e8m0(tokens * topk, inter_dim // 32, device=device, low_exp=124, high_exp=126).view(tokens, topk, inter_dim // 32)
        else:
            a2_q = out1_fp16.to(torch.float32)
            # a2_q = torch.ones_like(out1_fp16, dtype=torch.float32) / 5
            # w2_fp32 = torch.ones_like(w2_fp32, dtype=torch.float32) / 10
            a2_scale = None
    elif in_dtype == "fp8":
        out1_fp32 = out1_fp16.to(torch.float32)
        a2_q, a2_scale = pertoken_quant(out1_fp32, quant_dtype=DTYPE_FP8)
    elif in_dtype == "fp16":
        a2_q = out1_fp16
        a2_scale = None
    elif in_dtype == "bf16":
        a2_q = out1_fp16.to(torch.bfloat16)
        a2_scale = None
    elif in_dtype == "int4_bf16":
        # W4A16: A2 is bf16 (no quant)
        a2_q = out1_fp16.to(torch.bfloat16)
        a2_scale = None
    else:
        out1_fp32 = out1_fp16.to(torch.float32)
        if in_dtype == "int8smooth":
            smooth_scale2 = (
                0.75
                + 0.5
                * torch.rand((experts, inter_dim), device=out1_fp32.device, dtype=torch.float32)
            )
            out1_fp32 = out1_fp32 * smooth_scale2[topk_ids.to(torch.int64)]
        a2_q, a2_scale = pertoken_quant(out1_fp32, quant_dtype=torch.int8)

    _out2_fp32, _us2 = run_moe_stage2(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        group_size=group_size,
        tile_m=tile_m,
        tile_n=tile_n2,
        tile_k=tile_k2,
        doweight_stage1=bool(doweight_stage1),
        seed=seed,
        num_iters=num_iters,
        num_warmup=num_warmup,
        compare_aiter_ck=compare_aiter_ck,
        moe_sort_mode=moe_sort_mode,
        x_fp32_in=x_fp32,
        w1_fp32_in=w1_fp32,
        w2_fp32_in=w2_fp32,
        topk_ids_in=topk_ids,
        topk_weights_in=topk_weights,
        routing_in=routing,
        a2_fp8_in=a2_q,
        a2_scale_in=a2_scale,
        return_outputs=True,
        skip_ref=fp4_skip_ref,
        use_reduce=bool(use_reduce),
        use_valid_mask=use_valid_mask,
        test_graph=test_graph,
    )


# Test Helpers for MoE GEMM2 Mode Comparison
def _make_reduce_mode_compile_fn(use_flydsl_reduce: bool = True, use_valid_mask: bool = False):
    """Create a compile function that forces reduce mode.
    
    Args:
        use_flydsl_reduce: If True, use FlyDSL reduce kernel.
                          If False, use torch.sum (for baseline comparison).
    """
    def _compile(
        *,
        model_dim: int,
        inter_dim: int,
        experts: int,
        topk: int,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        doweight_stage2: bool,
        in_dtype: str = "fp8",
        group_size: int = -1,
        out_dtype: str = "f16",
    ):
        if use_flydsl_reduce:
            return compile_moe_gemm2_ex(
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=experts,
                topk=topk,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                doweight_stage2=doweight_stage2,
                in_dtype=in_dtype,
                group_size=group_size,
                out_dtype=out_dtype,
                valid_mask=(True if bool(use_valid_mask) else None),
                mode=MoeGemm2Mode.REDUCE,
                zero_intermediate=False, # test non-zeroed performance
            )
        else:
            gemm2_exe = compile_moe_gemm2(
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=experts,
                topk=topk,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                doweight_stage2=doweight_stage2,
                in_dtype=in_dtype,
                group_size=group_size,
                out_dtype=out_dtype,
                accumulate=False,
            )
            return _TorchReduceWrapper(gemm2_exe, topk, model_dim)
    return _compile


class _TorchReduceWrapper:
    """Wrapper for GEMM2 (accumulate=False) with torch.sum reduction.
    
    For baseline comparison only. Production code should use compile_moe_gemm2_ex.
    """

    def __init__(self, gemm2_exe, topk: int, model_dim: int):
        self._exe = gemm2_exe
        self._topk = topk
        self._model_dim = model_dim
        self._intermediate = None
        self._mode = MoeGemm2Mode.REDUCE

    def __call__(
        self,
        arg_out,
        arg_x,
        arg_w,
        arg_scale_x,
        arg_scale_w,
        arg_sorted_token_ids,
        arg_expert_ids,
        arg_sorted_weights,
        arg_num_valid_ids,
        tokens_in,
        n_in,
        k_in,
        size_expert_ids_in,
        valid_mask,
        stream,
    ):
        # Lazy allocate intermediate buffer
        needed = tokens_in * self._topk * self._model_dim
        if self._intermediate is None or self._intermediate.numel() < needed:
            self._intermediate = torch.empty(
                tokens_in * self._topk, self._model_dim,
                device=arg_out.device, dtype=arg_out.dtype
            )

        intermediate = self._intermediate[:tokens_in * self._topk, :]
        self._exe(
            intermediate.view(-1),
            arg_x, arg_w, arg_scale_x, arg_scale_w,
            arg_sorted_token_ids, arg_expert_ids, arg_sorted_weights,
            arg_num_valid_ids, tokens_in, n_in, k_in, size_expert_ids_in,
            stream,
        )
        X = intermediate.view(tokens_in, self._topk, self._model_dim)
        if valid_mask is not None:
            X = X * valid_mask.view(tokens_in, self._topk, 1).to(dtype=X.dtype)
        torch.sum(X, dim=1, out=arg_out)

    @property
    def mode(self) -> str:
        return self._mode


@pytest.mark.parametrize(
    "tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n, tile_k",
    [
        pytest.param(8192, 7168, 256, 128, 8, 64, 256, 128, id="DS-TP8-prefill-S", marks=pytest.mark.large_shape),
        pytest.param(16384, 7168, 256, 256, 8, 64, 256, 128, id="DS-TP8-prefill-M", marks=pytest.mark.large_shape),
        pytest.param(32768, 7168, 256, 256, 8, 64, 256, 128, id="DS-TP8-prefill-L", marks=pytest.mark.large_shape),
        pytest.param(1, 7168, 256, 256, 8, 16, 256, 128, id="DS-TP8-decode-bs1"),
        pytest.param(8, 7168, 256, 256, 8, 32, 256, 128, id="DS-TP8-decode-bs8"),
        pytest.param(1666, 5120, 1536, 64, 6, 64, 256, 128, id="EP-K6-prefill", marks=pytest.mark.large_shape),
        pytest.param(32768, 5120, 1536, 64, 6, 64, 256, 128, id="EP-K6-prefill-L", marks=pytest.mark.large_shape),
        pytest.param(1, 5120, 1536, 16, 6, 16, 128, 256, id="EP-K6-decode-bs1"),
        pytest.param(8, 5120, 1536, 16, 6, 64, 128, 128, id="EP-K6-decode-bs8"),
    ],
)
@pytest.mark.parametrize("in_dtype", ["fp8"])
def test_moe_stage2_standalone(
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    in_dtype: str,
    *,
    seed: int = 0,
    num_iters: int = 10,
    num_warmup: int = 3,
):
    """Standalone stage2 test comparing atomic vs reduce modes.

    Tests:
    1. Atomic mode: direct accumulation with atomics
    2. Reduce mode (torch): GEMM2 + torch.sum reduction
    3. Reduce mode (FlyDSL): GEMM2 + FlyDSL reduce kernel
    """
    # Common args
    common_args = dict(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=False,  # Apply weight in stage2
        in_dtype=in_dtype,
        seed=seed,
        num_iters=num_iters,
        num_warmup=num_warmup,
        moe_sort_mode="torch",
        compare_aiter_ck=False,
        skip_ref=False,
    )

    # Run baseline stage2 (atomic accumulation)
    run_moe_stage2(**common_args, kernel_name="moe_gemm2_atomic")

    # Run reduce mode with torch.sum (baseline comparison)
    run_moe_stage2(
        **common_args,
        compile_fn=_make_reduce_mode_compile_fn(use_flydsl_reduce=False),
        kernel_name="moe_gemm2_reduce_torch",
    )

    # Run reduce mode with FlyDSL kernel (production path)
    run_moe_stage2(
        **common_args,
        use_reduce=True,
        kernel_name="moe_gemm2_reduce_flydsl",
    )

    # Run reduce mode and use valid mask with FlyDSL kernel
    run_moe_stage2(
        **common_args,
        use_reduce=True,
        use_valid_mask=True,
        kernel_name="moe_gemm2_reduce_flydsl_valid_mask",
    )


if __name__ == "__main__":
    torch.set_default_device("cuda")
    # CLI (mirrors key knobs from aiter/op_tests/test_moe_2stage.py, stage1 subset)
    def _str2bool(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"invalid bool: {v} (use t/f, true/false, 1/0)")

    def _str2tuple_dim(v: str) -> Tuple[int, int]:
        # aiter uses "-dim 6144,4096" meaning (model_dim, inter_dim)
        s = str(v).strip()
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"invalid -dim {v!r}; expected 'model_dim,inter_dim' e.g. 6144,4096")
        return int(parts[0]), int(parts[1])

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MoE 2-stage (FlyDSL MFMA FP8) test/benchmark (argparse subset aligned with aiter test_moe_2stage.py)",
    )
    parser.add_argument(
        "--in_dtype",
        type=str,
        default="fp8",
        choices=["fp4", "fp8", "fp16", "bf16", "int8", "int8smooth", "int4", "int4_bf16", "all"],
        help="Kernel input dtype: fp4 / fp8 / fp16 / int8 / int8smooth / int4 / int4_bf16 / all (default: fp8). "
        "int8smooth expands X to [tokens*topk, K] with per-(token,slot) scales. "
        "int4 means W4A8: A int8, W packed int4. "
        "int4_bf16 means W4A16: A bf16, W packed int4.",
    )
    parser.add_argument("-d", "--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Input init dtype (currently data is quantized to FP8 per-token; init dtype mainly affects RNG range).")
    parser.add_argument("-dim", type=_str2tuple_dim, default=(6144, 4096), help="Model dimension: model_dim,inter_dim (e.g. -dim 6144,4096)")
    parser.add_argument("-t", "--tokenNum", type=int, default=32, help="Number of tokens (e.g. -t 1024)")
    parser.add_argument("-e", "--expert", type=int, default=8, help="Number of experts (e.g. -e 8)")
    parser.add_argument("-k", "--topk", type=int, default=2, help="Top-k (e.g. -k 2)")
    parser.add_argument("-s", "--doweight_stage1", type=_str2bool, nargs="?", const=True, default=False, help="Whether to multiply routed weight in stage1 (t/f).")

    # Stage1-specific kernel tiling knobs
    parser.add_argument("--tile_m", type=int, default=32, help="Tile M / block_m (routing block size).")
    parser.add_argument("--tile_n", type=int, default=128, help="Tile N (inter dim tile).")
    parser.add_argument("--tile_k", type=int, default=256, help="Tile K (model dim tile).")
    parser.add_argument("--tile_n2", type=int, default=None, help="Stage2 tile N (model dim tile). Default: 2*tile_n.")
    parser.add_argument("--tile_k2", type=int, default=None, help="Stage2 tile K (inter dim tile). Default: tile_k.")

    # Sorting / comparison knobs
    parser.add_argument("--moe_sort_mode", type=str, default=None, choices=["aiter", "torch"], help="Routing buffer build mode (aiter moe_sorting vs torch fallback).")
    parser.add_argument("--compare_aiter_ck", type=_str2bool, nargs="?", const=True, default=None, help="Override COMPARE_AITER_CK (t/f). Default: env or HAS_AITER.")
    parser.add_argument("--skip_ref", type=_str2bool, nargs="?", const=True, default=False, help="Skip torch reference correctness checks (benchmark-only).")
    parser.add_argument(
        "--gemm2_mode",
        type=str,
        default="both",
        choices=["both", "atomic", "reduce"],
        help="Stage2 accumulation mode: 'atomic', 'reduce', or 'both' (default: both).",
    )
    parser.add_argument(
        "--out_dtype",
        type=str,
        default="f16",
        choices=["f16", "f32"],
        help="Stage2 output dtype: f16 (half2 atomics) or f32 (scalar fp32 atomics).",
    )
    parser.add_argument("--use_valid_mask", type=_str2bool, nargs="?", const=True, default=False, help="Use valid mask for optimization when reduce or not.")

    # Benchmark knobs
    parser.add_argument("--seed", type=int, default=0, help="torch.manual_seed(seed)")
    parser.add_argument("--num_iters", type=int, default=2, help="Benchmark iters")
    parser.add_argument("--num_warmup", type=int, default=1, help="Benchmark warmup iters")

    # graph mode test
    parser.add_argument(
        "--test_graph",
        "-tg",
        action="store_true",
        default=False,
        help="test with graph mode.",
    )

    # w fp4 moe kernel
    parser.add_argument(
        "--wfp4",
        action="store_true",
        default=False,
        help="Use weight fp4 gemm.",
    )

    # Groupwise scale for W4A16
    parser.add_argument(
        "--group_size",
        type=int,
        default=-1,
        help="Group size for W4A16 groupwise scale (-1 = per-row, 32 = group_size=32).",
    )
    parser.add_argument(
        "--debug_stage",
        type=str,
        default="both",
        choices=["both", "stage1", "stage2"],
        help="Debug split switch: run both stages (default), stage1 only, or stage2 only.",
    )
    parser.add_argument(
        "--debug_single_block",
        action="store_true",
        default=False,
        help="Debug switch: force a single routing block by using tokenNum=min(tokenNum,tile_m) and topk=1.",
    )
    parser.add_argument(
        "--debug_single_expert",
        action="store_true",
        default=False,
        help="Debug switch: force single expert by overriding experts=1 and topk=min(topk,1).",
    )

    args = parser.parse_args()

    model_dim, inter_dim = args.dim

    tile_n2 = int(args.tile_n2) if args.tile_n2 is not None else int(args.tile_n) * 2
    tile_k2 = int(args.tile_k2) if args.tile_k2 is not None else args.tile_k

    dbg_tokens = int(args.tokenNum)
    dbg_experts = int(args.expert)
    dbg_topk = int(args.topk)
    if bool(args.debug_single_expert):
        dbg_experts = 1
        dbg_topk = min(dbg_topk, 1)
    if bool(args.debug_single_block):
        dbg_tokens = min(dbg_tokens, max(1, int(args.tile_m)))
        dbg_topk = 1
    if bool(args.debug_single_expert) or bool(args.debug_single_block):
        print(
            f"[debug] effective shape override: tokens={dbg_tokens}, experts={dbg_experts}, topk={dbg_topk} "
            f"(stage={args.debug_stage}, single_block={args.debug_single_block}, single_expert={args.debug_single_expert})"
            ,
            flush=True,
        )

    # Determine which gemm2 modes to run.
    if args.gemm2_mode == "both":
        reduce_flags = [False, True]
    elif args.gemm2_mode == "reduce":
        reduce_flags = [True]
    else:  # "atomic"
        reduce_flags = [False]

    def run_one(dt: str, use_reduce: bool):
        out_s = str(args.out_dtype).strip().lower()
        if bool(use_reduce) and out_s in ("f32", "fp32", "float"):
            print("[skip] reduce mode does not support out_dtype='f32'")
            return
        if (not bool(use_reduce)) and bool(args.use_valid_mask):
            print("[skip] valid_mask is only used in reduce mode (atomic ignores it)")
            return
        if args.debug_stage == "stage1":
            print(f"[debug] run stage1 only (dtype={dt}, reduce={use_reduce})", flush=True)
            run_moe_stage1(
                tokens=dbg_tokens,
                model_dim=int(model_dim),
                inter_dim=int(inter_dim),
                experts=dbg_experts,
                topk=dbg_topk,
                in_dtype=dt,
                group_size=int(args.group_size),
                tile_m=int(args.tile_m),
                tile_n=int(args.tile_n),
                tile_k=int(args.tile_k),
                doweight_stage1=bool(args.doweight_stage1),
                seed=int(args.seed),
                num_iters=int(args.num_iters),
                num_warmup=int(args.num_warmup),
                compare_aiter_ck=bool(args.compare_aiter_ck) and (dt == "fp8"),
                moe_sort_mode=args.moe_sort_mode,
                return_outputs=False,
                skip_ref=bool(args.skip_ref),
                w_fp4_kernel=bool(args.wfp4),
                test_graph=bool(args.test_graph),
                debug_serial=True,
            )
            print(f"PASSED: stage1 dtype={dt} reduce={use_reduce}")
            return
        if args.debug_stage == "stage2":
            print(f"[debug] run stage2 only (dtype={dt}, reduce={use_reduce})", flush=True)
            a2_in = None
            a2_scale_in = None
            if bool(args.skip_ref):
                dev = torch.device("cuda")
                if dt == "fp4":
                    a2_in = fp4_utils.random_fp4_packed(dbg_tokens * dbg_topk, int(inter_dim), device=dev)
                    a2_scale_in = fp4_utils.random_e8m0(dbg_tokens * dbg_topk, int(inter_dim) // 32, device=dev)
                elif dt == "fp8":
                    a2_src = torch.randn((dbg_tokens, dbg_topk, int(inter_dim)), device=dev, dtype=torch.float32)
                    a2_in, a2_scale_in = pertoken_quant(a2_src, quant_dtype=DTYPE_FP8)
                elif dt in ("fp16", "bf16", "int4_bf16"):
                    dt_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "int4_bf16": torch.bfloat16}
                    a2_in = torch.randn((dbg_tokens, dbg_topk, int(inter_dim)), device=dev, dtype=dt_map[dt])
                    a2_scale_in = None
                else:
                    a2_src = torch.randn((dbg_tokens, dbg_topk, int(inter_dim)), device=dev, dtype=torch.float32)
                    a2_in, a2_scale_in = pertoken_quant(a2_src, quant_dtype=torch.int8)
            run_moe_stage2(
                tokens=dbg_tokens,
                model_dim=int(model_dim),
                inter_dim=int(inter_dim),
                experts=dbg_experts,
                topk=dbg_topk,
                in_dtype=dt,
                out_dtype=str(args.out_dtype),
                group_size=int(args.group_size),
                tile_m=int(args.tile_m),
                tile_n=tile_n2,
                tile_k=tile_k2,
                doweight_stage1=bool(args.doweight_stage1),
                seed=int(args.seed),
                num_iters=int(args.num_iters),
                num_warmup=int(args.num_warmup),
                compare_aiter_ck=args.compare_aiter_ck,
                moe_sort_mode=args.moe_sort_mode,
                a2_fp8_in=a2_in,
                a2_scale_in=a2_scale_in,
                return_outputs=False,
                skip_ref=bool(args.skip_ref),
                use_reduce=bool(use_reduce),
                use_valid_mask=bool(args.use_valid_mask),
                test_graph=bool(args.test_graph),
                debug_serial=True,
            )
            print(f"PASSED: stage2 dtype={dt} reduce={use_reduce}")
            return
        test_moe_gemm_2stage(
            tokens=dbg_tokens,
            model_dim=int(model_dim),
            inter_dim=int(inter_dim),
            experts=dbg_experts,
            topk=dbg_topk,
            tile_m=int(args.tile_m),
            tile_n1=int(args.tile_n),
            tile_k1=int(args.tile_k),
            tile_n2=tile_n2,
            tile_k2=tile_k2,
            doweight_stage1=bool(args.doweight_stage1),
            in_dtype=dt,
            out_dtype=str(args.out_dtype),
            group_size=int(args.group_size),
            seed=int(args.seed),
            num_iters=int(args.num_iters),
            num_warmup=int(args.num_warmup),
            moe_sort_mode=args.moe_sort_mode,
            compare_aiter_ck=args.compare_aiter_ck,
            skip_ref=bool(args.skip_ref),
            w_fp4_kernel=args.wfp4,
            use_reduce=use_reduce,
            use_valid_mask=bool(args.use_valid_mask),
            test_graph=bool(args.test_graph),
        )
        print(f"PASSED: both dtype={dt} reduce={use_reduce}")

    # Run 2-stage (gemm1 -> quantize -> gemm2) aiter-style test/benchmark.
    # Expand "all" to all supported dtypes.
    in_dtypes = args.in_dtype.split(",")
    if "all" in in_dtypes:
        in_dtypes = ["fp4", "fp8", "fp16", "bf16", "int8", "int4", "int4_bf16"]
    for dt in in_dtypes:
        for use_reduce in reduce_flags:
            run_one(dt, use_reduce)
