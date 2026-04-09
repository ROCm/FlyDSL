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
import torch.nn.functional as F

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
from tests.utils import pertoken_quant, shuffle_weight, shuffle_scale_for_int4, get_dtype_max
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

SCALE_BLOCK = 32


def _per_1x32_fp8_quant(x: torch.Tensor):
    """Quantize fp32 tensor to raw FP8/E4M3 bytes with one E8M0 scale per 32-wide K block."""
    if x.shape[-1] % SCALE_BLOCK != 0:
        raise ValueError(f"Last dim must be divisible by {SCALE_BLOCK}, got {x.shape[-1]}")
    shape_original = x.shape
    x2d = x.reshape(-1, shape_original[-1]).to(torch.float32)
    m, n = x2d.shape
    x_blk = x2d.view(-1, SCALE_BLOCK)
    x_blk = torch.nan_to_num(x_blk, nan=0.0, posinf=0.0, neginf=0.0)
    max_abs = torch.amax(torch.abs(x_blk), dim=1)
    dtype_max = float(get_dtype_max(DTYPE_FP8))
    scale_e8m0 = fp4_utils.f32_to_e8m0(max_abs / dtype_max)
    scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0)
    scale_f32 = torch.nan_to_num(scale_f32, nan=1.0, posinf=1.0, neginf=1.0)
    scale_f32[scale_f32 == 0] = 1.0
    y_f32 = x_blk / scale_f32.view(-1, 1)
    # Clamp before casting to float8 to avoid generating NaN payloads.
    y_f32 = torch.clamp(y_f32, min=-dtype_max, max=dtype_max)
    y = fp4_utils._f32_to_floatx_unpacked(y_f32.contiguous().view(-1), 4, 3).view(torch.uint8)
    y = y.view(*shape_original)
    scale = scale_e8m0.view(m, n // SCALE_BLOCK).view(torch.uint8)
    return y, scale


def _dequant_blockscale_fp8(x_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if scale.dim() == x_q.dim() - 1:
        scale = scale.view(*x_q.shape[:-1], scale.shape[-1])
    scale_f32 = fp4_utils.e8m0_to_f32(scale.view(torch.uint8))
    scale_expanded = scale_f32.repeat_interleave(SCALE_BLOCK, dim=-1)[..., : x_q.shape[-1]]
    return fp4_utils.fp8_e4m3_to_f32(x_q.view(torch.uint8)) * scale_expanded


def _dequant_blockscale_fp4(x_q: torch.Tensor, scale: torch.Tensor, k_dim: int) -> torch.Tensor:
    if scale.dim() == x_q.dim() - 1:
        scale = scale.view(*x_q.shape[:-1], scale.shape[-1])
    scale_f32 = fp4_utils.e8m0_to_f32(scale.view(torch.uint8))
    scale_expanded = scale_f32.repeat_interleave(SCALE_BLOCK, dim=-1)[..., :k_dim]
    return fp4_utils.mxfp4_to_f32(x_q.view(torch.uint8))[..., :k_dim] * scale_expanded


def _bench_kernel_us(run_fn, warmup=10, iters=50, flush_l2=True, prep_fn=None):
    """Per-iteration CUDA events timer with optional L2 flush and median latency."""
    flush_buf = None
    if flush_l2:
        l2_bytes = getattr(
            torch.cuda.get_device_properties(torch.cuda.current_device()),
            "L2_cache_size", 4 * 1024 * 1024)
        alloc_bytes = max(l2_bytes * 2, 8 * 1024 * 1024)
        flush_buf = torch.empty(alloc_bytes, dtype=torch.uint8, device="cuda")

    for _ in range(warmup):
        if flush_buf is not None:
            flush_buf.zero_()
        if prep_fn is not None:
            prep_fn()
        run_fn()
    torch.cuda.synchronize()

    start_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        if flush_buf is not None:
            flush_buf.zero_()
        if prep_fn is not None:
            prep_fn()
        start_ev[i].record()
        run_fn()
        end_ev[i].record()

    torch.cuda.synchronize()
    latencies = sorted(start_ev[i].elapsed_time(end_ev[i]) * 1e3 for i in range(iters))

    n = len(latencies)
    if n >= 8:
        q1, q3 = latencies[n // 4], latencies[3 * n // 4]
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filtered = [x for x in latencies if lo <= x <= hi]
        if filtered:
            latencies = filtered

    del flush_buf
    return latencies[len(latencies) // 2]


def _torch_moe_gemm1_a8w4(
    x_fp8: torch.Tensor,
    w1_fp4_flat: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w1_flat: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    inter_dim: int,
    doweight_stage1: bool,
) -> torch.Tensor:
    topk = topk_ids.shape[1]
    tokens, model_dim = x_fp8.shape
    experts = int(w1_fp4_flat.shape[0] // (2 * inter_dim))
    x = _dequant_blockscale_fp8(x_fp8, scale_x)
    w1 = _dequant_blockscale_fp4(w1_fp4_flat, scale_w1_flat, model_dim).view(experts, 2 * inter_dim, model_dim)
    out = torch.zeros((tokens, topk, inter_dim), device=x.device, dtype=torch.float32)
    for e in range(experts):
        mask = topk_ids == e
        idx = mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            continue
        t_idx = idx[:, 0]
        s_idx = idx[:, 1]
        y2 = F.linear(x[t_idx, :], w1[e, :, :])
        gate = y2[:, :inter_dim]
        up = y2[:, inter_dim:]
        y = F.silu(gate) * up
        if doweight_stage1:
            y = y * topk_weights[t_idx, s_idx].unsqueeze(-1)
        out[t_idx, s_idx, :] = y
    return out


def _torch_moe_gemm2_a8w4(
    a2_fp8: torch.Tensor,
    w2_fp4: torch.Tensor,
    scale_a2: torch.Tensor,
    scale_w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    model_dim: int,
    doweight_stage2: bool,
) -> torch.Tensor:
    tokens, topk, inter_dim = a2_fp8.shape
    experts = int(w2_fp4.shape[0]) if w2_fp4.dim() == 3 else int(w2_fp4.shape[0] // model_dim)
    a2 = _dequant_blockscale_fp8(a2_fp8, scale_a2)
    w2 = _dequant_blockscale_fp4(w2_fp4, scale_w2, inter_dim).view(experts, model_dim, inter_dim)
    out = torch.zeros((tokens, model_dim), device=a2.device, dtype=torch.float32)
    for e in range(experts):
        mask = topk_ids == e
        idx = mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            continue
        t_idx = idx[:, 0]
        s_idx = idx[:, 1]
        y = F.linear(a2[t_idx, s_idx, :], w2[e, :, :])
        if doweight_stage2:
            y = y * topk_weights[t_idx, s_idx].unsqueeze(-1)
        out.index_add_(0, t_idx, y)
    return out


def preshuffle_e8m0_scale(scale: torch.Tensor, warp_tile: int,
                          scale_k_per_tile: int = 4,
                          WMMA_DIM: int = 16,
                          byte_swap: bool = False) -> torch.Tensor:
    """Preshuffle E8M0 scale for ds_load_b128-based B-scale access."""
    _, K_scale = scale.shape
    assert K_scale % 4 == 0, f"K_scale must be divisible by 4, got {K_scale}"

    if byte_swap:
        grouped = scale.view(-1, K_scale // 4, 4)
        scale = grouped[:, :, [0, 2, 1, 3]].contiguous().view(-1, K_scale)

    scales_per_wmma = 4
    wmma_rep = warp_tile // WMMA_DIM
    k_groups = K_scale // scale_k_per_tile
    k_wmma_steps = scale_k_per_tile // scales_per_wmma
    g = scale.view(-1, wmma_rep, WMMA_DIM, k_groups, k_wmma_steps, scales_per_wmma)
    g = g.permute(0, 2, 3, 4, 1, 5).contiguous()
    return g.reshape(-1, k_groups * k_wmma_steps * wmma_rep * scales_per_wmma)


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
    _pick_fp16_single_launch_shape,
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


def _debug_tensor_health(name: str, tensor: Optional[torch.Tensor]) -> None:
    if tensor is None:
        print(f"[debug] {name}: <None>", flush=True)
        return
    tensor_for_check = tensor
    if tensor.dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    ):
        tensor_for_check = tensor.to(torch.float32)
    finite = torch.isfinite(tensor_for_check)
    finite_count = int(finite.sum().item())
    total = int(finite.numel())
    print(
        f"[debug] {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"finite={finite_count}/{total}",
        flush=True,
    )
    if finite_count != total:
        bad_idx = (finite == 0).nonzero(as_tuple=False)
        if bad_idx.numel() > 0:
            first_bad = tuple(int(v) for v in bad_idx[0].tolist())
            print(f"[debug] {name}: first_non_finite_idx={first_bad}", flush=True)
            try:
                print(f"[debug] {name}: first_non_finite_val={tensor_for_check[first_bad]}", flush=True)
            except Exception:
                pass


def _maybe_debug_a8w4_stage1_failure(
    *,
    in_dtype: str,
    out: torch.Tensor,
    ref: torch.Tensor,
    x_ref: torch.Tensor,
    sx_ref: Optional[torch.Tensor],
    w1_q_flat: torch.Tensor,
    scale_w1_ref_flat: Optional[torch.Tensor],
) -> None:
    flag = str(os.environ.get("A8W4_STAGE1_DEBUG", "0")).strip().lower()
    if in_dtype != "a8w4" or flag not in {"1", "true", "t", "yes", "y", "on"}:
        return
    print("[debug] A8W4 stage1 failure diagnostics", flush=True)
    _debug_tensor_health("out", out)
    _debug_tensor_health("ref", ref)
    _debug_tensor_health("x_ref", x_ref)
    _debug_tensor_health("sx_ref", sx_ref)
    _debug_tensor_health("w1_q_flat", w1_q_flat)
    _debug_tensor_health("scale_w1_ref_flat", scale_w1_ref_flat)


def _use_serial_fp4_debug(in_dtype: str) -> bool:
    dt = str(in_dtype).strip().lower()
    if dt not in ("fp4", "fp8", "fp16", "a8w4"):
        return False
    if dt == "fp4":
        serial_key = "MOE_FP4_SERIAL"
        debug_key = "MOE_FP4_DEBUG"
    elif dt in ("fp8", "a8w4"):
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
    if dt not in ("fp4", "fp8", "fp16", "a8w4"):
        return False
    if bool(debug_serial):
        return True
    # Default to safe mode for fp4/fp8/fp16 to avoid run_perftest rotate-worker segfaults.
    if dt == "fp4":
        safe_key = "MOE_FP4_SAFE_PERF"
    elif dt in ("fp8", "a8w4"):
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
    benchmark_mode: bool = False,
    flush_l2: bool = True,
    num_buffers: int = 1,
    use_tdm_gather: bool = True,
    use_tdm_store: bool = False,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
    expert_sched_mode: bool = True,
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

    if in_dtype not in ("fp4", "fp8", "a8w4", "fp16", "bf16", "int8", "int8smooth", "int4", "int4_bf16"):
        raise ValueError(
            f"in_dtype must be one of ('fp4','fp8','a8w4','fp16','bf16','int8','int8smooth','int4','int4_bf16'), got {in_dtype!r}"
        )
    is_fp4 = in_dtype == "fp4"
    is_a8w4 = in_dtype == "a8w4"
    is_int4 = in_dtype == "int4"
    is_int4_bf16 = in_dtype == "int4_bf16"  # W4A16: bf16 activations, packed int4 weights
    is_int8 = in_dtype in ("int8", "int8smooth", "int4")
    is_int8smooth = in_dtype == "int8smooth"

    # Quantize inputs / weights.
    if in_dtype == "fp4":
        # Quantize fp32 inputs to fp4+e8m0 to keep values in a range that avoids
        # fp16 overflow after the silu(gate)*up epilogue.
        x_fp4, scale_x_raw, _ = fp4_utils.per_1x32_f4_quant(x_fp32)
        x_q = x_fp4.view(torch.uint8)
        scale_x = scale_x_raw.view(torch.uint8).view(tokens, model_dim // 32)
        w1_fp4, scale_w1_raw, _ = fp4_utils.per_1x32_f4_quant(w1_fp32.view(-1, model_dim))
        w1_q = w1_fp4.view(torch.uint8).view(experts, 2 * inter_dim, model_dim // 2)
        scale_w1 = scale_w1_raw.view(torch.uint8).view(experts, 2 * inter_dim, model_dim // 32)
        w2_fp4, _, _ = fp4_utils.per_1x32_f4_quant(w2_fp32.view(-1, inter_dim))
        w2_q = w2_fp4.view(torch.uint8).view(experts, model_dim, inter_dim // 2)
    elif in_dtype == "fp8":
        # gfx1250 WMMA_SCALE requires per-block-of-32 e8m0 microscaling, not per-row scales.
        x_q, scale_x = _per_1x32_fp8_quant(x_fp32)        # [tokens,K] uint8, [tokens, K//32] uint8
        w1_q, scale_w1 = _per_1x32_fp8_quant(w1_fp32)      # [E,2*inter,K] uint8, [E,2*inter, K//32] uint8
        w2_q, _scale_w2_unused = _per_1x32_fp8_quant(w2_fp32)
    elif in_dtype == "a8w4":
        x_q, scale_x = _per_1x32_fp8_quant(x_fp32)
        w1_q, scale_w1, _ = fp4_utils.per_1x32_f4_quant(w1_fp32)
        w2_q, _scale_w2_unused, _ = fp4_utils.per_1x32_f4_quant(w2_fp32)
        w1_q = w1_q.view(torch.uint8)
        scale_w1 = scale_w1.view(torch.uint8)
        w2_q = w2_q.view(torch.uint8)
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

    scale_w1_kernel = scale_w1

    # Preshuffle weights on the *unpacked* tensor.
    uses_fp4_weight_layout = is_fp4 or is_a8w4
    # gfx1250 native kernels (fp4, fp8, a8w4, fp16) use TDM/transpose-load and
    # handle weight layout internally — skip the MFMA-style shuffle_weight.
    uses_native_gfx1250 = in_dtype in ("fp4", "fp8", "a8w4", "fp16", "bf16")
    w1_shuffled = w1_q if (uses_fp4_weight_layout or uses_native_gfx1250) else shuffle_weight(w1_q)
    w2_shuffled = w2_q if (uses_fp4_weight_layout or uses_native_gfx1250) else (shuffle_weight(w2_q) if in_dtype == "fp8" else None)
    if in_dtype in ("fp8", "a8w4"):
        _single_tile_m, _single_tile_n, _single_m_warp, _single_n_warp = _pick_fp16_single_launch_shape(tile_m, tile_n)
        _warp_tile_n = _single_tile_n // _single_n_warp
        w1_rows = experts * (2 * inter_dim)
        w1_cols = model_dim // 2 if is_a8w4 else model_dim
        w1_shuffled = fp4_utils.preshuffle_b_16x16(
            w1_q.contiguous().view(w1_rows, w1_cols),
            w1_rows,
            w1_cols,
        ).view_as(w1_q)

    # Flatten W1 for our FlyDSL kernel (treat expert dim as part of N).
    uses_block_scale = in_dtype in ("fp8", "a8w4", "fp4")
    if uses_fp4_weight_layout:
        w1_shuffled_flat = w1_shuffled.view(experts * (2 * inter_dim), model_dim // 2)
        w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim // 2)
        scale_w1_ref_flat = None if scale_w1 is None else scale_w1.view(experts * (2 * inter_dim), model_dim // 32)
        scale_w1_kernel_flat = None if scale_w1_kernel is None else scale_w1_kernel.view(experts * (2 * inter_dim), model_dim // 32)
    else:
        w1_shuffled_flat = w1_shuffled.view(experts * (2 * inter_dim), model_dim)
        w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim)
        scale_cols = model_dim // 32 if uses_block_scale else 1
        scale_w1_ref_flat = None if scale_w1 is None else scale_w1.view(experts * (2 * inter_dim), scale_cols)
        scale_w1_kernel_flat = None if scale_w1_kernel is None else scale_w1_kernel.view(experts * (2 * inter_dim), scale_cols)

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
    if uses_fp4_weight_layout:
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
    elif scale_w1_kernel_flat is None:
        scale_w1_1d = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        scale_w1_1d = scale_w1_kernel_flat.view(-1).contiguous()
    sorted_weights_1d = sorted_weights.contiguous().view(-1)  # [sorted_size]

    # Output: [tokens, topk, inter_dim] fp16
    out = torch.zeros((tokens, topk, inter_dim), device=device, dtype=torch.float16)

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
        num_buffers=int(num_buffers),
        use_tdm_gather=bool(use_tdm_gather),
        use_tdm_store=bool(use_tdm_store),
        inst_prefetch=bool(inst_prefetch),
        wave_specialized_tdm=bool(wave_specialized_tdm),
        cluster_m=int(cluster_m),
        cluster_n=int(cluster_n),
        expert_sched_mode=bool(expert_sched_mode),
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
    elif bool(benchmark_mode) and not bool(test_graph):
        def _prep_stage1():
            out.zero_()

        def _run_stage1():
            launch(out, x_q, w_kernel, scale_x_1d, scale_w1_1d, sorted_token_ids, sorted_expert_ids, sorted_weights_1d)

        us = _bench_kernel_us(
            _run_stage1,
            warmup=int(num_warmup),
            iters=int(max(1, num_iters)),
            flush_l2=bool(flush_l2),
            prep_fn=_prep_stage1,
        )
        torch.cuda.synchronize()
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
        elif in_dtype == "fp8":
            x_ref = _dequant_blockscale_fp8(x_q.view(tokens, model_dim), scale_x.view(tokens, model_dim // 32))
            sx_ref = None
        elif is_fp4:
            x_ref = _dequant_blockscale_fp4(x_q.view(tokens, model_dim // 2), scale_x.view(tokens, model_dim // 32), model_dim)
            sx_ref = None
        else:
            x_ref = x_q
            sx_ref = scale_x
        if is_a8w4:
            ref = _torch_moe_gemm1_a8w4(
                x_ref,
                w1_q_flat,
                sx_ref,
                scale_w1_ref_flat,
                topk_ids.to(torch.int64),
                topk_weights,
                inter_dim=inter_dim,
                doweight_stage1=doweight_stage1,
            )
        elif in_dtype == "fp8":
            w_ref_f32 = _dequant_blockscale_fp8(w1_q_flat, scale_w1_ref_flat)
            ref = torch_moe_gemm1(
                x_ref,
                w_ref_f32,
                sx_ref,
                None,
                topk_ids.to(torch.int64),
                topk_weights,
                inter_dim=inter_dim,
                doweight_stage1=doweight_stage1,
            )
        elif is_fp4:
            w_ref_f32 = _dequant_blockscale_fp4(w1_q_flat, scale_w1_ref_flat, model_dim)
            ref = torch_moe_gemm1(
                x_ref,
                w_ref_f32,
                sx_ref,
                None,
                topk_ids.to(torch.int64),
                topk_weights,
                inter_dim=inter_dim,
                doweight_stage1=doweight_stage1,
            )
        else:
            ref = torch_moe_gemm1(
                x_ref,
                w1_q_flat,
                sx_ref,
                scale_w1_ref_flat,
                topk_ids.to(torch.int64),
                topk_weights,
                inter_dim=inter_dim,
                doweight_stage1=doweight_stage1,
                group_size=group_size,
                scale_w1_groups=scale_w1_groups,
            )

        rtol = 0.5 if (is_int4 or is_int4_bf16 or is_a8w4 or is_fp4) else 0.25
        atol = 0.5 if (is_int4 or is_int4_bf16 or is_a8w4) else 0.25
        _maybe_debug_a8w4_stage1_failure(
            in_dtype=in_dtype,
            out=out.to(torch.float32),
            ref=ref,
            x_ref=x_ref,
            sx_ref=sx_ref,
            w1_q_flat=w1_q_flat,
            scale_w1_ref_flat=scale_w1_ref_flat,
        )
        assert verify_output(out.to(torch.float32), ref, rtol=rtol, atol=atol)

    # Note: kernel launches full expert-block range; effective work is gated by num_valid_ids.
    flops = 2 * tokens * topk * (2 * inter_dim) * model_dim
    tflops = flops / (us / 1e6) / 1e12

    # Rough bytes-moved accounting (same spirit as GEMM tests: count each tensor once).
    bytes_moved = 0
    x_elem_bytes = 2 if (is_int4_bf16 or in_dtype in ("bf16", "fp16")) else 1  # bf16/fp16 activations
    bytes_x = (tokens * topk if is_int8smooth else tokens) * model_dim * x_elem_bytes
    bytes_w = (experts * (2 * inter_dim) * model_dim) // (2 if (use_packed_int4 or is_a8w4) else 1)
    bytes_out = tokens * topk * inter_dim * 2
    bytes_scale_x = (tokens * topk if is_int8smooth else tokens) * 4
    bytes_scale_w = experts * (2 * inter_dim) * 4
    bytes_route = (
        int(sorted_weights.numel()) * 4
        + int(sorted_token_ids.numel()) * 4
        + int(sorted_expert_ids.numel()) * 4
    )
    bytes_moved += bytes_x + bytes_w + bytes_out + bytes_scale_x + bytes_scale_w + bytes_route
    tbps = bytes_moved / 1e12 / (us / 1e6)
    read_bytes = bytes_x + bytes_w + bytes_scale_x + bytes_scale_w + bytes_route
    write_bytes = bytes_out
    read_bw_gbs = read_bytes / 1e9 / (us / 1e6)
    write_bw_gbs = write_bytes / 1e9 / (us / 1e6)

    if bool(benchmark_mode):
        print(
            f"FlyDSL MoE stage1[{in_dtype}] benchmark | "
            f"shape=({tokens},{model_dim},{inter_dim}), E={experts}, K={topk}, "
            f"tile=({tile_m},{tile_n},{tile_k})"
        )
        print(
            f"  kernel: {us:.1f} us ({us / 1e3:.4f} ms) | "
            f"{tflops:.2f} TFLOPS(logical, M={tokens*topk}) | {tbps:.3f} TB/s"
        )
        print(
            f"  bandwidth: read {read_bw_gbs:.1f} GB/s + write {write_bw_gbs:.1f} GB/s | "
            f"bytes: x={bytes_x/1e6:.1f}MB w={bytes_w/1e6:.1f}MB "
            f"sx={bytes_scale_x/1e6:.1f}MB sw={bytes_scale_w/1e6:.1f}MB "
            f"route={bytes_route/1e6:.1f}MB out={bytes_out/1e6:.1f}MB"
        )
    else:
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
    benchmark_mode: bool = False,
    flush_l2: bool = True,
    num_buffers: int = 1,
    use_tdm_gather: bool = True,
    use_tdm_store: bool = False,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
    expert_sched_mode: bool = True,
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

    if in_dtype not in ("fp4", "fp8", "a8w4", "fp16", "bf16", "int8", "int8smooth", "int4", "int4_bf16"):
        raise ValueError(
            f"in_dtype must be one of ('fp4','fp8','a8w4','fp16','bf16','int8','int8smooth','int4','int4_bf16'), got {in_dtype!r}"
        )
    is_fp4 = in_dtype == "fp4"
    is_a8w4 = in_dtype == "a8w4"
    is_int4 = in_dtype == "int4"
    is_int4_bf16 = in_dtype == "int4_bf16"  # W4A16: bf16 activations, packed int4 weights
    is_int8 = in_dtype in ("int8", "int8smooth", "int4")
    is_int8smooth = in_dtype == "int8smooth"

    # Quantize inputs / weights.
    if in_dtype == "fp4":
        x_fp4, scale_x_raw, _ = fp4_utils.per_1x32_f4_quant(x_fp32)
        x_q = x_fp4.view(torch.uint8)
        scale_x = scale_x_raw.view(torch.uint8).view(tokens, model_dim // 32)
        w1_fp4, scale_w1_raw, _ = fp4_utils.per_1x32_f4_quant(w1_fp32.view(-1, model_dim))
        w1_q = w1_fp4.view(torch.uint8).view(experts, 2 * inter_dim, model_dim // 2)
        scale_w1 = scale_w1_raw.view(torch.uint8).view(experts, 2 * inter_dim, model_dim // 32)
        w2_fp4, scale_w2_raw, _ = fp4_utils.per_1x32_f4_quant(w2_fp32.view(-1, inter_dim))
        w2_q = w2_fp4.view(torch.uint8).view(experts, model_dim, inter_dim // 2)
        scale_w2 = scale_w2_raw.view(torch.uint8).view(experts, model_dim, inter_dim // 32)
    elif in_dtype == "fp8":
        # gfx1250 WMMA_SCALE requires per-block-of-32 e8m0 microscaling.
        x_q, scale_x = _per_1x32_fp8_quant(x_fp32)
        w1_q, scale_w1 = _per_1x32_fp8_quant(w1_fp32)
        w2_q, scale_w2 = _per_1x32_fp8_quant(w2_fp32)
    elif in_dtype == "a8w4":
        x_q, scale_x = _per_1x32_fp8_quant(x_fp32)
        w1_q, scale_w1, _ = fp4_utils.per_1x32_f4_quant(w1_fp32)
        w2_q, scale_w2, _ = fp4_utils.per_1x32_f4_quant(w2_fp32)
        w1_q = w1_q.view(torch.uint8)
        scale_w1 = scale_w1.view(torch.uint8)
        w2_q = w2_q.view(torch.uint8)
        scale_w2 = scale_w2.view(torch.uint8)
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

    scale_w2_kernel = scale_w2

    # Preshuffle weights on the *unpacked* tensor.
    uses_fp4_weight_layout = is_fp4 or is_a8w4
    uses_native_gfx1250 = in_dtype in ("fp4", "fp8", "a8w4", "fp16", "bf16")
    w1_shuffled = w1_q if (uses_fp4_weight_layout or uses_native_gfx1250) else shuffle_weight(w1_q)
    w2_shuffled = w2_q if (uses_fp4_weight_layout or uses_native_gfx1250) else shuffle_weight(w2_q)
    if in_dtype in ("fp8", "a8w4"):
        _single_tile_m, _single_tile_n, _single_m_warp, _single_n_warp = _pick_fp16_single_launch_shape(tile_m, tile_n)
        _warp_tile_n = _single_tile_n // _single_n_warp
        w2_rows = experts * model_dim
        w2_cols = inter_dim // 2 if is_a8w4 else inter_dim
        w2_shuffled = fp4_utils.preshuffle_b_16x16(
            w2_q.contiguous().view(w2_rows, w2_cols),
            w2_rows,
            w2_cols,
        ).view_as(w2_q)

    # Stage2 input (A2): either provided (gemm1->quantize chaining) or built from stage1 reference.
    # For int4_bf16, A2 is bf16 (same as fp16 for scale handling).
    if a2_fp8_in is not None and (a2_scale_in is not None or in_dtype in ("fp16", "bf16", "int4_bf16")):
        a2_q = a2_fp8_in
        a2_scale = a2_scale_in
    else:
        if is_fp4:
            w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim // 2)
            scale_w1_flat = None if scale_w1 is None else scale_w1.view(experts * (2 * inter_dim), model_dim // 32)
        elif is_a8w4:
            w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim // 2)
            scale_w1_flat = None if scale_w1 is None else scale_w1.view(experts * (2 * inter_dim), model_dim // 32)
        elif in_dtype == "fp8":
            w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim)
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
        if in_dtype == "fp8":
            # fp8 block-scale: dequantize for reference computation.
            x_dequant = _dequant_blockscale_fp8(x_q.view(-1, model_dim), scale_x.reshape(-1, model_dim // 32))
            w1_dequant = _dequant_blockscale_fp8(w1_q_flat, scale_w1_flat)
            out1_ref = torch_moe_gemm1(
                x_dequant, w1_dequant, None, None,
                topk_ids.to(torch.int64), topk_weights,
                inter_dim=inter_dim, doweight_stage1=bool(doweight_stage1),
            )
        else:
            out1_ref = torch_moe_gemm1(
                x_q,
                w1_q_flat,
                scale_x,
                scale_w1_flat,
                topk_ids.to(torch.int64),
                topk_weights,
                inter_dim=inter_dim,
                doweight_stage1=bool(doweight_stage1),
            )
        # [tokens, topk, inter] fp32
        if in_dtype == "fp8":
            a2_q, a2_scale = _per_1x32_fp8_quant(out1_ref)
        elif in_dtype == "a8w4":
            a2_q, a2_scale = _per_1x32_fp8_quant(out1_ref)
        elif in_dtype == "fp4":
            a2_q = fp4_utils.random_fp4_packed(tokens * topk, inter_dim, device=device)
            a2_scale = fp4_utils.random_e8m0(tokens * topk, inter_dim // 32, device=device)
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
    uses_block_scale_s2 = in_dtype in ("fp8", "a8w4", "fp4")
    if uses_fp4_weight_layout:
        w2_shuffled_flat = w2_shuffled.view(experts * model_dim, inter_dim // 2)
        scale_w2_ref_flat = None if scale_w2 is None else scale_w2.view(experts * model_dim, inter_dim // 32)
        scale_w2_kernel_flat = None if scale_w2_kernel is None else scale_w2_kernel.view(experts * model_dim, inter_dim // 32)
    else:
        w2_shuffled_flat = w2_shuffled.view(experts * model_dim, inter_dim)
        scale_cols_s2 = inter_dim // 32 if uses_block_scale_s2 else 1
        scale_w2_ref_flat = None if scale_w2 is None else scale_w2.view(experts * model_dim, scale_cols_s2)
        scale_w2_kernel_flat = None if scale_w2_kernel is None else scale_w2_kernel.view(experts * model_dim, scale_cols_s2)

    # For W4A8 and W4A16, pack preshuffled int8 weights into packed int4 bytes.
    # Both use the same interleaved packing: [ (v4<<4)|v0, (v5<<4)|v1, (v6<<4)|v2, (v7<<4)|v3 ]
    use_packed_int4 = is_int4 or is_int4_bf16
    w2_kernel = w2_shuffled_flat
    if use_packed_int4:
        w2_kernel = _pack_shuffled_int8_to_packed_int4_no_perm(w2_shuffled_flat)

    w2_flat = w2_kernel.contiguous().view(-1)
    w2_kernel = w2_flat
    if uses_fp4_weight_layout:
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
    elif scale_w2_kernel_flat is None:
        w2_scale_1d = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        w2_scale_1d = scale_w2_kernel_flat.view(-1).contiguous()
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
        num_buffers=int(num_buffers),
        use_tdm_gather=bool(use_tdm_gather),
        use_tdm_store=bool(use_tdm_store),
        inst_prefetch=bool(inst_prefetch),
        wave_specialized_tdm=bool(wave_specialized_tdm),
        cluster_m=int(cluster_m),
        cluster_n=int(cluster_n),
        expert_sched_mode=bool(expert_sched_mode),
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
    elif bool(benchmark_mode) and not bool(test_graph):
        def _prep_stage2():
            out_perf.zero_()

        def _run_stage2():
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

        us = _bench_kernel_us(
            _run_stage2,
            warmup=int(num_warmup),
            iters=int(max(1, num_iters)),
            flush_l2=bool(flush_l2),
            prep_fn=_prep_stage2,
        )
        torch.cuda.synchronize()
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
        if is_a8w4:
            ref2 = _torch_moe_gemm2_a8w4(
                a2_q,
                w2_q,
                a2_scale,
                scale_w2,
                topk_ids.to(torch.int64),
                topk_weights,
                model_dim=model_dim,
                doweight_stage2=doweight_stage2,
            )
        elif in_dtype == "fp8":
            a2_dequant = _dequant_blockscale_fp8(a2_q.view(-1, inter_dim), a2_scale.reshape(-1, inter_dim // 32))
            a2_dequant = a2_dequant.view(a2_q.shape[0], a2_q.shape[1], inter_dim)
            w2_dequant = _dequant_blockscale_fp8(
                w2_q.view(experts * model_dim, inter_dim),
                scale_w2.view(experts * model_dim, inter_dim // 32),
            )
            ref2 = torch_moe_gemm2(
                a2_dequant, w2_dequant.view_as(w2_q), None, None,
                topk_ids.to(torch.int64), topk_weights,
                model_dim=model_dim, doweight_stage2=doweight_stage2,
            )
        elif is_fp4:
            a2_dequant = _dequant_blockscale_fp4(
                a2_q.view(-1, inter_dim // 2), a2_scale.reshape(-1, inter_dim // 32), inter_dim
            )
            a2_dequant = a2_dequant.view(a2_q.shape[0], a2_q.shape[1], inter_dim)
            w2_dequant = _dequant_blockscale_fp4(
                w2_q.view(experts * model_dim, inter_dim // 2),
                scale_w2.view(experts * model_dim, inter_dim // 32),
                inter_dim,
            )
            ref2 = torch_moe_gemm2(
                a2_dequant, w2_dequant.view(experts, model_dim, inter_dim), None, None,
                topk_ids.to(torch.int64), topk_weights,
                model_dim=model_dim, doweight_stage2=doweight_stage2,
            )
        else:
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
    bytes_a2 = tokens * topk * inter_dim * a2_elem_bytes
    bytes_w2 = (experts * model_dim * inter_dim) // (2 if (is_int4 or is_a8w4) else 1)
    bytes_out = tokens * model_dim * (2 if out_torch_dtype == torch.float16 else 4)
    bytes_scale_a2 = tokens * topk * 4
    bytes_scale_w2 = experts * model_dim * 4
    bytes_route = (
        int(sorted_weights.numel()) * 4
        + int(sorted_token_ids.numel()) * 4
        + int(sorted_expert_ids.numel()) * 4
    )
    bytes_moved += bytes_a2 + bytes_w2 + bytes_out + bytes_scale_a2 + bytes_scale_w2 + bytes_route
    tbps = bytes_moved / 1e12 / (us / 1e6)
    read_bytes = bytes_a2 + bytes_w2 + bytes_scale_a2 + bytes_scale_w2 + bytes_route
    write_bytes = bytes_out
    read_bw_gbs = read_bytes / 1e9 / (us / 1e6)
    write_bw_gbs = write_bytes / 1e9 / (us / 1e6)
    if bool(benchmark_mode):
        print(
            f"FlyDSL MoE stage2[{kernel_name}] {in_dtype} {'reduce' if use_reduce else 'atomic'} benchmark | "
            f"shape=({tokens},{model_dim},{inter_dim}), E={experts}, K={topk}, "
            f"tile=({tile_m},{tile_n},{tile_k})"
        )
        print(
            f"  kernel: {us:.1f} us ({us / 1e3:.4f} ms) | "
            f"{tflops:.2f} TFLOPS(logical, M={tokens*topk}) | {tbps:.3f} TB/s"
        )
        print(
            f"  bandwidth: read {read_bw_gbs:.1f} GB/s + write {write_bw_gbs:.1f} GB/s | "
            f"bytes: a2={bytes_a2/1e6:.1f}MB w2={bytes_w2/1e6:.1f}MB "
            f"sa2={bytes_scale_a2/1e6:.1f}MB sw2={bytes_scale_w2/1e6:.1f}MB "
            f"route={bytes_route/1e6:.1f}MB out={bytes_out/1e6:.1f}MB"
        )
    else:
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

    assert torch.isfinite(stage1_out).all()
    assert torch.isfinite(stage2_out).all()


def test_moe_stage1_use_tdm_gather_smoke():
    """Stage1 should support toggling A TDM gather without changing numerics on padded routes."""
    tokens = 5
    model_dim = 256
    inter_dim = 128
    experts = 1
    topk = 1
    tile_m = 16
    tile_n = 64
    tile_k = 128

    device = torch.device("cuda")
    torch.manual_seed(0)
    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32)
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32)
    w2_fp32 = torch.randn((experts, model_dim, inter_dim), device=device, dtype=torch.float32)
    topk_ids = torch.zeros((tokens, topk), device=device, dtype=torch.int64)
    topk_weights = torch.ones((tokens, topk), device=device, dtype=torch.float32)
    routing = build_routing_buffers(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        experts=experts,
        model_dim=model_dim,
        tile_m=tile_m,
        moe_sort_mode="torch",
    )

    common_args = dict(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=False,
        in_dtype="fp8",
        seed=123,
        num_iters=1,
        num_warmup=1,
        compare_aiter_ck=False,
        moe_sort_mode="torch",
        x_fp32_in=x_fp32,
        w1_fp32_in=w1_fp32,
        w2_fp32_in=w2_fp32,
        topk_ids_in=topk_ids,
        topk_weights_in=topk_weights,
        routing_in=routing,
        return_outputs=True,
        skip_ref=True,
    )

    out_scalar, _ = run_moe_stage1(
        **common_args,
        use_tdm_gather=False,
    )
    out_gather, _ = run_moe_stage1(
        **common_args,
        use_tdm_gather=True,
    )

    assert torch.isfinite(out_scalar).all()
    assert torch.isfinite(out_gather).all()
    assert verify_output(out_gather.to(torch.float32), out_scalar.to(torch.float32), rtol=0.5, atol=0.5)


def test_moe_stage2_use_tdm_gather_smoke():
    """Stage2 should support toggling A TDM gather without changing numerics."""
    tokens = 32
    model_dim = 256
    inter_dim = 128
    experts = 4
    topk = 2
    tile_m = 16
    tile_n = 64
    tile_k = 128

    torch.manual_seed(0)
    a2_fp32 = torch.randn((tokens, topk, inter_dim), device="cuda", dtype=torch.float32)
    a2_q, a2_scale = _per_1x32_fp8_quant(a2_fp32)

    common_args = dict(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=False,
        in_dtype="fp8",
        out_dtype="f16",
        seed=123,
        num_iters=1,
        num_warmup=1,
        compare_aiter_ck=False,
        return_outputs=True,
        skip_ref=True,
        a2_fp8_in=a2_q,
        a2_scale_in=a2_scale,
    )

    out_scalar, _ = run_moe_stage2(
        **common_args,
        use_tdm_gather=False,
    )
    out_gather, _ = run_moe_stage2(
        **common_args,
        use_tdm_gather=True,
    )

    assert torch.isfinite(out_scalar).all()
    assert torch.isfinite(out_gather).all()
    assert verify_output(out_gather.to(torch.float32), out_scalar.to(torch.float32), rtol=0.5, atol=0.5)


def test_moe_stage2_use_tdm_gather_padding_smoke():
    """Stage2 gather should match scalar load on deterministic padding-heavy routing."""
    tokens = 5
    model_dim = 256
    inter_dim = 128
    experts = 1
    topk = 2
    tile_m = 16
    tile_n = 64
    tile_k = 128

    device = torch.device("cuda")
    torch.manual_seed(1)
    a2_fp32 = torch.randn((tokens, topk, inter_dim), device=device, dtype=torch.float32)
    a2_q, a2_scale = _per_1x32_fp8_quant(a2_fp32)
    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32)
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32)
    w2_fp32 = torch.randn((experts, model_dim, inter_dim), device=device, dtype=torch.float32)
    topk_ids = torch.zeros((tokens, topk), device=device, dtype=torch.int64)
    topk_weights = torch.tensor([[0.75, 0.25]], device=device, dtype=torch.float32).expand(tokens, topk).contiguous()
    routing = build_routing_buffers(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        experts=experts,
        model_dim=model_dim,
        tile_m=tile_m,
        moe_sort_mode="torch",
    )

    common_args = dict(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=False,
        in_dtype="fp8",
        out_dtype="f16",
        seed=321,
        num_iters=1,
        num_warmup=1,
        compare_aiter_ck=False,
        moe_sort_mode="torch",
        x_fp32_in=x_fp32,
        w1_fp32_in=w1_fp32,
        w2_fp32_in=w2_fp32,
        topk_ids_in=topk_ids,
        topk_weights_in=topk_weights,
        routing_in=routing,
        a2_fp8_in=a2_q,
        a2_scale_in=a2_scale,
        return_outputs=True,
        skip_ref=True,
    )

    out_scalar, _ = run_moe_stage2(
        **common_args,
        use_tdm_gather=False,
    )
    out_gather, _ = run_moe_stage2(
        **common_args,
        use_tdm_gather=True,
    )

    assert torch.isfinite(out_scalar).all()
    assert torch.isfinite(out_gather).all()
    assert verify_output(out_gather.to(torch.float32), out_scalar.to(torch.float32), rtol=0.5, atol=0.5)


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

    assert torch.isfinite(stage1_out).all()
    assert torch.isfinite(stage2_out).all()


def test_moe_2stage_fp4_wfp4_reduce_reference():
    """wfp4 correctness path should still use stage1-derived fp4 A2 input."""
    tokens = 64
    model_dim = 256
    inter_dim = 128
    experts = 4
    topk = 2
    tile_m = 16
    tile_n = 64
    tile_k = 128
    seed = 0
    test_moe_gemm_2stage(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n1=tile_n,
        tile_k1=tile_k,
        tile_n2=tile_n,
        tile_k2=tile_k,
        doweight_stage1=False,
        in_dtype="fp4",
        out_dtype="f16",
        group_size=-1,
        seed=seed,
        num_iters=1,
        num_warmup=1,
        compare_aiter_ck=False,
        moe_sort_mode="torch",
        skip_ref=False,
        use_reduce=True,
        use_tdm_store=False,
        w_fp4_kernel=True,
        test_graph=False,
        use_valid_mask=False,
    )


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
@pytest.mark.parametrize("in_dtype", ["fp4", "fp8", "a8w4", "fp16", "bf16", "int8", "int8smooth", "int4", "int4_bf16"])
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
    benchmark_mode: bool = False,
    flush_l2: bool = True,
    num_buffers: int = 1,
    use_tdm_store: bool = False,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
):
    """Single 2-stage test: gemm1 -> quantize -> gemm2, with routing built once.

    When in_dtype='int4_bf16' and group_size>0, uses groupwise scale (W4A16 with per-group dequant).
    """
    if (not bool(use_reduce)) and bool(use_valid_mask):
        pytest.skip("valid_mask is only used in reduce mode (atomic mode ignores it).")
    if out_dtype in ("f32", "fp32", "float") and in_dtype in ("fp4", "fp8", "a8w4", "fp16", "bf16"):
        pytest.skip(f"gfx1250 {in_dtype} kernels only support out_dtype f16/bf16, not f32.")
    if in_dtype in ("fp4", "a8w4") and os.environ.get("FLYDSL_SKIP_SHAPE_GUARD", "0") != "1":
        is_small_shape = (
            tokens == 64
            and model_dim == 256
            and inter_dim == 128
            and experts == 4
            and topk == 2
        )
        if not is_small_shape:
            pytest.skip(f"{in_dtype} in main matrix is enabled only for the small shape.")
    out_s = str(out_dtype).strip().lower()
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
        skip_ref=bool(skip_ref),
        w_fp4_kernel=w_fp4_kernel,
        test_graph=test_graph,
        benchmark_mode=bool(benchmark_mode),
        flush_l2=bool(flush_l2),
        num_buffers=int(num_buffers),
        use_tdm_store=bool(use_tdm_store),
        inst_prefetch=bool(inst_prefetch),
        wave_specialized_tdm=bool(wave_specialized_tdm),
        cluster_m=int(cluster_m),
        cluster_n=int(cluster_n),
    )

    if w_fp4_kernel:
        if in_dtype == "fp4":
            # Stage2 FP4 kernel expects packed FP4 + E8M0 scale input.
            # For correctness mode, quantize stage1 output so stage2 checks the
            # real two-stage path. Keep synthetic A2 only for benchmark-only
            # runs that explicitly skip the reference.
            if bool(skip_ref):
                a2_q = fp4_utils.random_fp4_packed(tokens * topk, inter_dim, device=device).view(
                    tokens, topk, inter_dim // 2
                )
                a2_scale = fp4_utils.random_e8m0(tokens * topk, inter_dim // 32, device=device).view(
                    tokens, topk, inter_dim // 32
                )
            else:
                out1_fp32 = out1_fp16.to(torch.float32)
                a2_fp4, a2_scale_raw, _ = fp4_utils.per_1x32_f4_quant(out1_fp32.view(-1, inter_dim))
                a2_q = a2_fp4.view(torch.uint8).view(tokens, topk, inter_dim // 2)
                a2_scale = a2_scale_raw.view(torch.uint8).view(tokens, topk, inter_dim // 32)
        else:
            a2_q = out1_fp16.to(torch.float32)
            # a2_q = torch.ones_like(out1_fp16, dtype=torch.float32) / 5
            # w2_fp32 = torch.ones_like(w2_fp32, dtype=torch.float32) / 10
            a2_scale = None
    elif in_dtype == "fp4":
        out1_fp32 = out1_fp16.to(torch.float32)
        a2_fp4, a2_scale_raw, _ = fp4_utils.per_1x32_f4_quant(out1_fp32.view(-1, inter_dim))
        a2_q = a2_fp4.view(torch.uint8).view(tokens, topk, inter_dim // 2)
        a2_scale = a2_scale_raw.view(torch.uint8).view(tokens, topk, inter_dim // 32)
    elif in_dtype in ("fp8", "a8w4"):
        out1_fp32 = out1_fp16.to(torch.float32)
        a2_q, a2_scale = _per_1x32_fp8_quant(out1_fp32)
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
        skip_ref=bool(skip_ref),
        use_reduce=bool(use_reduce),
        use_valid_mask=use_valid_mask,
        test_graph=test_graph,
        benchmark_mode=bool(benchmark_mode),
        flush_l2=bool(flush_l2),
        num_buffers=int(num_buffers),
        use_tdm_store=bool(use_tdm_store),
        inst_prefetch=bool(inst_prefetch),
        wave_specialized_tdm=bool(wave_specialized_tdm),
        cluster_m=int(cluster_m),
        cluster_n=int(cluster_n),
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
        waves_per_eu: Optional[int] = None,
        expert_sched_mode: bool = True,
        num_buffers: int = 1,
        use_tdm_gather: bool = True,
        use_tdm_store: bool = False,
        inst_prefetch: bool = False,
        wave_specialized_tdm: bool = False,
        cluster_m: int = 1,
        cluster_n: int = 1,
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
                waves_per_eu=waves_per_eu,
                valid_mask=(True if bool(use_valid_mask) else None),
                mode=MoeGemm2Mode.REDUCE,
                zero_intermediate=False, # test non-zeroed performance
                expert_sched_mode=bool(expert_sched_mode),
                num_buffers=int(num_buffers),
                use_tdm_gather=bool(use_tdm_gather),
                use_tdm_store=bool(use_tdm_store),
                inst_prefetch=bool(inst_prefetch),
                wave_specialized_tdm=bool(wave_specialized_tdm),
                cluster_m=int(cluster_m),
                cluster_n=int(cluster_n),
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
                waves_per_eu=waves_per_eu,
                expert_sched_mode=bool(expert_sched_mode),
                num_buffers=int(num_buffers),
                use_tdm_gather=bool(use_tdm_gather),
                use_tdm_store=bool(use_tdm_store),
                inst_prefetch=bool(inst_prefetch),
                wave_specialized_tdm=bool(wave_specialized_tdm),
                cluster_m=int(cluster_m),
                cluster_n=int(cluster_n),
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


# ── Benchmark Mode ─────────────────────────────────────────────────────────
# Integrated from bench_moe_gfx1250.py: sweeps model configs × dtypes × token counts.

BENCH_WARMUP = 5
BENCH_ITERS = 20

BENCH_MODEL_CONFIGS = [
    # name,      model_dim, inter_dim, experts, topk
    ("DeepSeek-TP", 7168,    256,   257,  9),
    ("DeepSeek-EP", 7168,   2048,    32,  8),
    ("GPToss",      2880,   2880,   128,  4),
]

BENCH_DTYPE_TARGET_TILES = {
    # dtype: (tile_m, target_n, target_k, wmma_k)
    "fp4":  (16, 256, 512, 128),
    "fp8":  (16, 256, 512, 128),
    "a8w4": (16, 256, 512, 128),
    "fp16": (32,  64,  64,  32),
    "bf16": (32,  64,  64,  32),
}


def _best_tile(target, dim, align):
    """Largest value <= target that divides dim and is a multiple of align."""
    for v in range(target, 0, -align):
        if dim % v == 0:
            return v
    return None


def _resolve_tiles(in_dtype, model_dim, inter_dim):
    """Compute the largest valid (tile_m, tile_n1, tile_k1, tile_n2, tile_k2)
    for a given dtype and model shape, falling back from the target when
    dimensions don't divide evenly."""
    tile_m, target_n, target_k, wmma_k = BENCH_DTYPE_TARGET_TILES[in_dtype]

    tile_n1 = _best_tile(target_n, inter_dim, 16)
    tile_k1 = _best_tile(target_k, model_dim, wmma_k)
    tile_n2 = _best_tile(target_n, model_dim, 16)

    tile_k2 = None
    for k in range(target_k, 0, -wmma_k):
        if inter_dim % k != 0:
            continue
        total = tile_m * k
        if total % 256 == 0 and (total // 256) % 4 == 0:
            tile_k2 = k
            break

    if any(v is None for v in (tile_n1, tile_k1, tile_n2, tile_k2)):
        return None
    return (tile_m, tile_n1, tile_k1, tile_n2, tile_k2)

BENCH_DEFAULT_TOKEN_SWEEP = [1, 4, 8, 32, 64, 128, 256]
_BENCH_SCALE_GROUP = 32


def _bench_dtype_bpe(in_dtype):
    """Return (a_bpe, w_bpe, w_scale_bpg) for bandwidth accounting."""
    if in_dtype == "fp4":
        return 0.5, 0.5, 1
    if in_dtype == "a8w4":
        return 1, 0.5, 1
    if in_dtype == "fp8":
        return 1, 1, 1
    if in_dtype in ("fp16", "bf16"):
        return 2, 2, 0
    return 1, 1, 1


def _bench_bytes_moved_stage1(tokens, topk, model_dim, inter_dim, experts, in_dtype):
    a_bpe, w_bpe, w_scale_bpg = _bench_dtype_bpe(in_dtype)
    aE = min(tokens * topk, experts)
    b = 0
    b += tokens * model_dim * a_bpe
    b += aE * (2 * inter_dim) * model_dim * w_bpe
    b += aE * (2 * inter_dim) * math.ceil(model_dim / _BENCH_SCALE_GROUP) * w_scale_bpg
    b += tokens * topk * inter_dim * 2
    return int(b)


def _bench_bytes_moved_stage2(tokens, topk, model_dim, inter_dim, experts, in_dtype):
    a_bpe, w_bpe, w_scale_bpg = _bench_dtype_bpe(in_dtype)
    aE = min(tokens * topk, experts)
    b = 0
    b += tokens * topk * inter_dim * a_bpe
    b += aE * model_dim * inter_dim * w_bpe
    b += aE * model_dim * math.ceil(inter_dim / _BENCH_SCALE_GROUP) * w_scale_bpg
    b += tokens * topk * model_dim * 2
    return int(b)


def _bench_setup_data(tokens, model_dim, inter_dim, experts, topk, tile_m, seed=42):
    device = torch.device("cuda")
    torch.manual_seed(seed)
    s = 0.2
    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32) * s
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * s
    w2_fp32 = torch.randn((experts, model_dim, inter_dim), device=device, dtype=torch.float32) * (s / math.sqrt(inter_dim))
    score = torch.rand((tokens, experts), device=device, dtype=torch.float32)
    topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)
    routing = build_routing_buffers(
        topk_ids=topk_ids, topk_weights=topk_weights,
        experts=experts, model_dim=model_dim, tile_m=tile_m,
    )
    return x_fp32, w1_fp32, w2_fp32, topk_ids, topk_weights, routing


def _bench_prepare_a2(out1_fp16, tokens, topk, inter_dim, in_dtype):
    if in_dtype == "fp4":
        f32 = out1_fp16.to(torch.float32)
        a2_fp4, a2_scale_raw, _ = fp4_utils.per_1x32_f4_quant(f32.view(-1, inter_dim))
        a2_q = a2_fp4.view(torch.uint8).view(tokens, topk, inter_dim // 2)
        a2_scale = a2_scale_raw.view(torch.uint8).view(tokens, topk, inter_dim // 32)
    elif in_dtype in ("fp8", "a8w4"):
        a2_q, a2_scale = _per_1x32_fp8_quant(out1_fp16.to(torch.float32))
    elif in_dtype in ("fp16", "bf16"):
        a2_q = out1_fp16
        a2_scale = None
    else:
        raise ValueError(f"Unsupported bench in_dtype={in_dtype}")
    return a2_q, a2_scale


def _bench_print_banner(text):
    print(f"\n{'=' * 110}")
    print(f"  {text}")
    print(f"{'=' * 110}")


def _bench_print_stage_header():
    print(f"{'Tokens':>7} {'M_eff':>7} {'Latency(us)':>12} {'TFLOPS':>9} "
          f"{'BW(TB/s)':>10} {'Util%':>7} {'Status':>8}")
    print("-" * 110)


def _bench_print_stage_row(tokens, m_eff, us, tflops, tbps, util_pct, status):
    print(f"{tokens:>7} {m_eff:>7} {us:>10.1f}   {tflops:>8.2f} "
          f"{tbps:>9.3f}  {util_pct:>6.1f}% {status:>8}")


def bench_config(name, model_dim, inter_dim, experts, topk,
                 in_dtype, token_list, check_ref, peak_tflops,
                 warmup=BENCH_WARMUP, iters=BENCH_ITERS,
                 use_tdm_store=False, inst_prefetch=False,
                 wave_specialized_tdm=False):
    """Benchmark a single (model, dtype) configuration across all token counts."""
    tiles = _resolve_tiles(in_dtype, model_dim, inter_dim)
    if tiles is None:
        _bench_print_banner(f"{name}  |  {in_dtype}  |  dim={model_dim}  inter={inter_dim}")
        print(f"  SKIP: no valid tile for this shape (WMMA_K alignment)")
        return
    tile_m, tile_n1, tile_k1, tile_n2, tile_k2 = tiles

    _bench_print_banner(f"{name}  |  {in_dtype}  |  dim={model_dim}  inter={inter_dim}  E={experts}  K={topk}")
    print(f"  Tiles: stage1=({tile_m},{tile_n1},{tile_k1})  stage2=({tile_m},{tile_n2},{tile_k2})")
    print(f"  Warmup={warmup}  Iters={iters}  RefCheck={'ON' if check_ref else 'OFF'}")
    print(
        f"  Knobs: use_tdm_store={bool(use_tdm_store)}  "
        f"inst_prefetch={bool(inst_prefetch)}  "
        f"wave_specialized_tdm={bool(wave_specialized_tdm)}"
    )
    if peak_tflops > 0:
        print(f"  Peak compute reference: {peak_tflops:.0f} TFLOPS")

    # ── Stage 1 ──
    print(f"\n  ── Stage 1 (gate+up: [{model_dim}] -> [{2*inter_dim}]) ──")
    _bench_print_stage_header()

    s1_results = []
    for tok in token_list:
        torch.cuda.empty_cache()
        data_pack = _bench_setup_data(tok, model_dim, inter_dim, experts, topk, tile_m)
        x, w1, w2, ids, wts, routing = data_pack
        try:
            out1, us1 = run_moe_stage1(
                tokens=tok, model_dim=model_dim, inter_dim=inter_dim,
                experts=experts, topk=topk, in_dtype=in_dtype, group_size=-1,
                tile_m=tile_m, tile_n=tile_n1, tile_k=tile_k1,
                doweight_stage1=False, seed=0,
                num_iters=iters, num_warmup=warmup,
                x_fp32_in=x, w1_fp32_in=w1, w2_fp32_in=w2,
                topk_ids_in=ids, topk_weights_in=wts, routing_in=routing,
                return_outputs=True, skip_ref=(not check_ref),
                use_tdm_store=bool(use_tdm_store),
                inst_prefetch=bool(inst_prefetch),
                wave_specialized_tdm=bool(wave_specialized_tdm),
            )
            status = "PASS" if check_ref else "OK"
        except Exception as e:
            status = "FAIL"
            us1 = 0.0
            out1 = torch.zeros((tok, topk, inter_dim), device="cuda", dtype=torch.float16)
            print(f"  [{type(e).__name__}] tokens={tok}: {e}")

        m_eff = tok * topk
        flops = 2 * tok * topk * (2 * inter_dim) * model_dim
        tflops = flops / (us1 / 1e6) / 1e12 if us1 > 0 else 0
        bm = _bench_bytes_moved_stage1(tok, topk, model_dim, inter_dim, experts, in_dtype)
        tbps = bm / 1e12 / (us1 / 1e6) if us1 > 0 else 0
        util = (tflops / peak_tflops * 100) if (peak_tflops > 0 and tflops > 0) else 0

        _bench_print_stage_row(tok, m_eff, us1, tflops, tbps, util, status)
        s1_results.append((tok, m_eff, us1, tflops, tbps, status, out1))

    # ── Stage 2 atomic ──
    print(f"\n  ── Stage 2 atomic (down: [{inter_dim}] -> [{model_dim}]) ──")
    _bench_print_stage_header()

    for tok, m_eff, _, _, _, s1_status, out1 in s1_results:
        torch.cuda.empty_cache()
        data_pack = _bench_setup_data(tok, model_dim, inter_dim, experts, topk, tile_m)
        x, w1, w2, ids, wts, routing = data_pack
        a2_q, a2_scale = _bench_prepare_a2(out1, tok, topk, inter_dim, in_dtype)
        try:
            _, us2 = run_moe_stage2(
                tokens=tok, model_dim=model_dim, inter_dim=inter_dim,
                experts=experts, topk=topk, in_dtype=in_dtype, out_dtype="f16",
                group_size=-1, tile_m=tile_m, tile_n=tile_n2, tile_k=tile_k2,
                doweight_stage1=False, seed=0,
                num_iters=iters, num_warmup=warmup,
                x_fp32_in=x, w1_fp32_in=w1, w2_fp32_in=w2,
                topk_ids_in=ids, topk_weights_in=wts, routing_in=routing,
                a2_fp8_in=a2_q, a2_scale_in=a2_scale,
                return_outputs=True, skip_ref=(not check_ref),
                use_reduce=False,
                use_tdm_store=bool(use_tdm_store),
                inst_prefetch=bool(inst_prefetch),
                wave_specialized_tdm=bool(wave_specialized_tdm),
            )
            status = "PASS" if check_ref else "OK"
        except Exception as e:
            status = "FAIL"
            us2 = 0.0
            print(f"  [{type(e).__name__}] tokens={tok}: {e}")

        flops = 2 * tok * topk * model_dim * inter_dim
        tflops = flops / (us2 / 1e6) / 1e12 if us2 > 0 else 0
        bm = _bench_bytes_moved_stage2(tok, topk, model_dim, inter_dim, experts, in_dtype)
        tbps = bm / 1e12 / (us2 / 1e6) if us2 > 0 else 0
        util = (tflops / peak_tflops * 100) if (peak_tflops > 0 and tflops > 0) else 0

        _bench_print_stage_row(tok, m_eff, us2, tflops, tbps, util, status)

    # ── Stage 2 reduce ──
    print(f"\n  ── Stage 2 reduce (down: [{inter_dim}] -> [{model_dim}]) ──")
    _bench_print_stage_header()

    for tok, m_eff, _, _, _, s1_status, out1 in s1_results:
        torch.cuda.empty_cache()
        data_pack = _bench_setup_data(tok, model_dim, inter_dim, experts, topk, tile_m)
        x, w1, w2, ids, wts, routing = data_pack
        a2_q, a2_scale = _bench_prepare_a2(out1, tok, topk, inter_dim, in_dtype)
        try:
            _, us2r = run_moe_stage2(
                tokens=tok, model_dim=model_dim, inter_dim=inter_dim,
                experts=experts, topk=topk, in_dtype=in_dtype, out_dtype="f16",
                group_size=-1, tile_m=tile_m, tile_n=tile_n2, tile_k=tile_k2,
                doweight_stage1=False, seed=0,
                num_iters=iters, num_warmup=warmup,
                x_fp32_in=x, w1_fp32_in=w1, w2_fp32_in=w2,
                topk_ids_in=ids, topk_weights_in=wts, routing_in=routing,
                a2_fp8_in=a2_q, a2_scale_in=a2_scale,
                return_outputs=True, skip_ref=(not check_ref),
                use_reduce=True,
                use_tdm_store=bool(use_tdm_store),
                inst_prefetch=bool(inst_prefetch),
                wave_specialized_tdm=bool(wave_specialized_tdm),
            )
            status = "PASS" if check_ref else "OK"
        except Exception as e:
            status = "FAIL"
            us2r = 0.0
            print(f"  [{type(e).__name__}] tokens={tok}: {e}")

        flops = 2 * tok * topk * model_dim * inter_dim
        tflops = flops / (us2r / 1e6) / 1e12 if us2r > 0 else 0
        bm = _bench_bytes_moved_stage2(tok, topk, model_dim, inter_dim, experts, in_dtype)
        tbps = bm / 1e12 / (us2r / 1e6) if us2r > 0 else 0
        util = (tflops / peak_tflops * 100) if (peak_tflops > 0 and tflops > 0) else 0

        _bench_print_stage_row(tok, m_eff, us2r, tflops, tbps, util, status)

    del s1_results
    torch.cuda.empty_cache()


def bench_main(args):
    """Entry point for --bench mode: sweep model configs x dtypes x token counts."""
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "1"

    warmup = args.bench_warmup
    iters = args.bench_iters

    timing_mode = args.bench_timing
    if timing_mode == "events":
        os.environ["MOE_TIMING"] = "events"
        os.environ["MOE_FP8_SAFE_PERF"] = "1"
        os.environ["MOE_FP4_SAFE_PERF"] = "1"
        os.environ["MOE_FP16_SAFE_PERF"] = "1"
    elif timing_mode == "serial":
        os.environ.pop("MOE_TIMING", None)
        os.environ["MOE_FP8_SAFE_PERF"] = "1"
        os.environ["MOE_FP4_SAFE_PERF"] = "1"
        os.environ["MOE_FP16_SAFE_PERF"] = "1"

    dtypes = args.bench_dtype.split(",") if args.bench_dtype else list(BENCH_DTYPE_TARGET_TILES.keys())
    token_list = (
        [int(t) for t in args.bench_tokens.split(",")]
        if args.bench_tokens
        else BENCH_DEFAULT_TOKEN_SWEEP
    )
    check_ref = not args.bench_no_ref

    timing_label = {"events": "CUDA-Events", "serial": "Serial(sync)", "both": "Both"}[timing_mode]
    print("=" * 110)
    print("  AMD gfx1250 MOE GEMM Kernel Performance Benchmark")
    print(f"  PyTorch {torch.__version__}")
    print(f"  Device:  {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  CUs:     {props.multi_processor_count}")
    print(f"  Memory:  {props.total_memory / 1024**3:.0f} GB")
    print(f"  Warmup={warmup}  Iters={iters}  RefCheck={'ON' if check_ref else 'OFF'}  Timing={timing_label}")
    print(f"  Dtypes:  {dtypes}")
    print(f"  Tokens:  {token_list}")
    print("=" * 110)

    def _set_timing_env(mode):
        if mode == "events":
            os.environ["MOE_TIMING"] = "events"
        else:
            os.environ.pop("MOE_TIMING", None)
        os.environ["MOE_FP8_SAFE_PERF"] = "1"
        os.environ["MOE_FP4_SAFE_PERF"] = "1"
        os.environ["MOE_FP16_SAFE_PERF"] = "1"

    modes = ["events", "serial"] if timing_mode == "both" else [timing_mode]

    t_start = time.time()
    for mode in modes:
        _set_timing_env(mode)
        if timing_mode == "both":
            mode_label = "CUDA-Events" if mode == "events" else "Serial(sync)"
            _bench_print_banner(f"Timing mode: {mode_label}")
        for cfg_name, mdim, idim, exp, topk in BENCH_MODEL_CONFIGS:
            if args.bench_config and args.bench_config not in cfg_name:
                continue
            for dt in dtypes:
                if dt not in BENCH_DTYPE_TARGET_TILES:
                    print(f"\n  [SKIP] Unknown dtype: {dt}")
                    continue
                try:
                    bench_config(
                        cfg_name, mdim, idim, exp, topk,
                        dt, token_list, check_ref, args.bench_peak_tflops,
                        warmup=warmup, iters=iters,
                        use_tdm_store=bool(args.use_tdm_store),
                        inst_prefetch=bool(args.inst_prefetch),
                        wave_specialized_tdm=bool(args.wave_specialized_tdm),
                    )
                except Exception as e:
                    print(f"\n  [ERROR] {cfg_name}/{dt}: {e}")
                    import traceback; traceback.print_exc()

    elapsed = time.time() - t_start
    _bench_print_banner(f"Done in {elapsed:.1f}s")


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
        choices=["fp4", "fp8", "a8w4", "fp16", "bf16", "int8", "int8smooth", "int4", "int4_bf16", "all"],
        help="Kernel input dtype: fp4 / fp8 / a8w4 / fp16 / int8 / int8smooth / int4 / int4_bf16 / all (default: fp8). "
        "int8smooth expands X to [tokens*topk, K] with per-(token,slot) scales. "
        "a8w4 uses FP8 activation with blockscale + FP4 packed weight. "
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
    parser.add_argument("--tile_m", type=int, default=16, help="Tile M / block_m (routing block size).")
    parser.add_argument("--tile_n", type=int, default=256, help="Tile N (inter dim tile).")
    parser.add_argument("--tile_k", type=int, default=512, help="Tile K (model dim tile).")
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
    parser.add_argument("--benchmark", action="store_true", default=False,
                        help="Use GEMM-style per-iteration event timing with optional L2 flush.")
    parser.add_argument("--no_flush_l2", action="store_true", default=False,
                        help="Disable L2 flush in benchmark mode.")
    parser.add_argument("--num_buffers", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Requested MXScale pipeline buffers for gfx1250 MoE kernels.")
    parser.add_argument("--use_tdm_store", type=_str2bool, nargs="?", const=True, default=False,
                        help="Requested TDM store epilogue for gfx1250 MoE kernels.")
    parser.add_argument("--inst_prefetch", type=_str2bool, nargs="?", const=True, default=False,
                        help="Enable instruction prefetch for gfx1250 MoE kernels.")
    parser.add_argument("--wave_specialized_tdm", type=_str2bool, nargs="?", const=True, default=False,
                        help="Enable wave-specialized TDM loading for gfx1250 MoE kernels.")
    parser.add_argument("--cluster_m", type=int, default=1,
                        help="Requested cluster_m for gfx1250 MoE kernels.")
    parser.add_argument("--cluster_n", type=int, default=1,
                        help="Requested cluster_n for gfx1250 MoE kernels.")
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

    # ── Benchmark sweep mode (--bench) ──
    bench_group = parser.add_argument_group("benchmark sweep", "Options for --bench mode (sweep model configs × dtypes × token counts)")
    bench_group.add_argument("--bench", action="store_true", default=False,
                             help="Run benchmark sweep mode instead of normal test mode.")
    bench_group.add_argument("--bench-dtype", type=str, default=None,
                             help="Comma-separated dtypes for bench (default: all of fp4,fp8,a8w4,fp16,bf16).")
    bench_group.add_argument("--bench-tokens", type=str, default=None,
                             help="Comma-separated token counts for bench (default: 1,4,8,32,64,128,256).")
    bench_group.add_argument("--bench-config", type=str, default=None,
                             help="Config name filter for bench (DeepSeek-TP, DeepSeek-EP, GPToss).")
    bench_group.add_argument("--bench-no-ref", action="store_true", default=False,
                             help="Skip correctness reference check in bench mode (pure perf).")
    bench_group.add_argument("--bench-warmup", type=int, default=BENCH_WARMUP,
                             help=f"Warmup iterations for bench (default: {BENCH_WARMUP}).")
    bench_group.add_argument("--bench-iters", type=int, default=BENCH_ITERS,
                             help=f"Measurement iterations for bench (default: {BENCH_ITERS}).")
    bench_group.add_argument("--bench-peak-tflops", type=float, default=0,
                             help="Peak TFLOPS for utilization calculation in bench mode.")
    bench_group.add_argument("--bench-timing", type=str, default="events",
                             choices=["events", "serial", "both"],
                             help="Timing method for bench: 'events' (CUDA events, default), "
                                  "'serial' (host time + sync), 'both'.")

    args = parser.parse_args()

    # ── Bench sweep mode: run and exit ──
    if args.bench:
        bench_main(args)
        sys.exit(0)

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
                benchmark_mode=bool(args.benchmark),
                flush_l2=not bool(args.no_flush_l2),
                num_buffers=int(args.num_buffers),
                use_tdm_store=bool(args.use_tdm_store),
                inst_prefetch=bool(args.inst_prefetch),
                wave_specialized_tdm=bool(args.wave_specialized_tdm),
                cluster_m=int(args.cluster_m),
                cluster_n=int(args.cluster_n),
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
                elif dt == "a8w4":
                    a2_src = torch.randn((dbg_tokens, dbg_topk, int(inter_dim)), device=dev, dtype=torch.float32)
                    a2_in, a2_scale_in = _per_1x32_fp8_quant(a2_src)
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
                benchmark_mode=bool(args.benchmark),
                flush_l2=not bool(args.no_flush_l2),
                num_buffers=int(args.num_buffers),
                use_tdm_store=bool(args.use_tdm_store),
                inst_prefetch=bool(args.inst_prefetch),
                wave_specialized_tdm=bool(args.wave_specialized_tdm),
                cluster_m=int(args.cluster_m),
                cluster_n=int(args.cluster_n),
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
            benchmark_mode=bool(args.benchmark),
            flush_l2=not bool(args.no_flush_l2),
            num_buffers=int(args.num_buffers),
            use_tdm_store=bool(args.use_tdm_store),
            inst_prefetch=bool(args.inst_prefetch),
            wave_specialized_tdm=bool(args.wave_specialized_tdm),
            cluster_m=int(args.cluster_m),
            cluster_n=int(args.cluster_n),
        )
        print(f"PASSED: both dtype={dt} reduce={use_reduce}")

    # Run 2-stage (gemm1 -> quantize -> gemm2) aiter-style test/benchmark.
    # Expand "all" to all supported dtypes.
    in_dtypes = args.in_dtype.split(",")
    if "all" in in_dtypes:
        in_dtypes = ["fp4", "fp8", "a8w4", "fp16", "bf16", "int8", "int4", "int4_bf16"]
    for dt in in_dtypes:
        for use_reduce in reduce_flags:
            run_one(dt, use_reduce)
