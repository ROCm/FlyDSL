#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE GEMM tests for MXScale data types (fp4/fp8/a8w4) on gfx1250."""
import math
import os
import sys
from typing import Optional

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
from tests.utils import get_dtype_max
from tests.test_common import verify_output, run_perftest
from flydsl.runtime.device import get_rocm_arch
from tests.kernels.utils import fp4_utils
from tests.kernels.benchmark_common import (
    bench_kernel_us as _bench_kernel_us,
)

ARCH = get_rocm_arch()
# GFX950 (MI350) and newer typically use OCP standard float8_e4m3fn
# GFX940/941/942 (MI300) use float8_e4m3fnuz
if "gfx95" in ARCH:
    DTYPE_FP8 = torch.float8_e4m3fn
else:
    DTYPE_FP8 = torch.float8_e4m3fnuz

SCALE_BLOCK = 32

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

if not str(ARCH).startswith("gfx1250"):
    pytest.skip(f"MoE 2stage gfx1250 tests require gfx1250, got {ARCH}", allow_module_level=True)



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


# Reuse routing utilities from the base MoE GEMM test harness.
from tests.kernels.test_moe_gemm import (
    build_routing_buffers,
    get_topk_valid_mask,
    RoutingBuffers,
)

# Kernel implementations live under `kernels/`; this test file is the harness.
from kernels.moe_gemm_2stage_mxscale_gfx1250 import (
    compile_moe_gemm1,
    compile_moe_gemm2,
    compile_moe_gemm2_ex,
    MoeGemm2Mode,
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
    seed: int = 0,
    num_iters: int = 5,
    num_warmup: int = 2,
    moe_sort_mode: Optional[str] = None,
    # Optional overrides (used by the 2-stage runner to avoid duplicated setup/sorting).
    x_fp32_in: Optional[torch.Tensor] = None,
    w1_fp32_in: Optional[torch.Tensor] = None,
    topk_ids_in: Optional[torch.Tensor] = None,
    topk_weights_in: Optional[torch.Tensor] = None,
    routing_in: Optional[RoutingBuffers] = None,
    return_outputs: bool = False,
    skip_ref: bool = False,
    test_graph: bool = False,
    waves_per_eu: Optional[int] = None,
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
):
    assert model_dim % 64 == 0
    assert inter_dim % tile_n == 0

    device = torch.device("cuda")
    torch.manual_seed(int(seed))

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

    if in_dtype not in ("fp4", "fp8", "a8w4"):
        raise ValueError(f"in_dtype must be one of ('fp4','fp8','a8w4'), got {in_dtype!r}")
    is_fp4 = in_dtype == "fp4"
    is_a8w4 = in_dtype == "a8w4"

    # Quantize inputs / weights (stage1 does not use W2).
    if in_dtype == "fp4":
        x_fp4, scale_x_raw, _ = fp4_utils.per_1x32_f4_quant(x_fp32)
        x_q = x_fp4.view(torch.uint8)
        scale_x = scale_x_raw.view(torch.uint8).view(tokens, model_dim // 32)
        w1_fp4, scale_w1_raw, _ = fp4_utils.per_1x32_f4_quant(w1_fp32.view(-1, model_dim))
        w1_q = w1_fp4.view(torch.uint8).view(experts, 2 * inter_dim, model_dim // 2)
        scale_w1 = scale_w1_raw.view(torch.uint8).view(experts, 2 * inter_dim, model_dim // 32)
    elif in_dtype == "fp8":
        x_q, scale_x = _per_1x32_fp8_quant(x_fp32)
        w1_q, scale_w1 = _per_1x32_fp8_quant(w1_fp32)
    else:  # a8w4
        x_q, scale_x = _per_1x32_fp8_quant(x_fp32)
        w1_q, scale_w1, _ = fp4_utils.per_1x32_f4_quant(w1_fp32)
        w1_q = w1_q.view(torch.uint8)
        scale_w1 = scale_w1.view(torch.uint8)

    # --- K-dimension padding for non-aligned model_dim ---
    _orig_model_dim = model_dim
    if model_dim % tile_k != 0:
        _pad_k = ((model_dim + tile_k - 1) // tile_k) * tile_k - model_dim
        if is_fp4:
            x_q = F.pad(x_q, (0, _pad_k // 2))
        else:
            x_q = F.pad(x_q, (0, _pad_k))
        scale_x = F.pad(scale_x, (0, _pad_k // 32))
        if is_fp4 or is_a8w4:
            w1_q = F.pad(w1_q, (0, _pad_k // 2))
        else:
            w1_q = F.pad(w1_q, (0, _pad_k))
        if scale_w1 is not None:
            scale_w1 = F.pad(scale_w1, (0, _pad_k // 32))
        model_dim = model_dim + _pad_k

    # Preshuffle weights — gfx1250 native kernels handle layout internally.
    uses_fp4_weight_layout = is_fp4 or is_a8w4
    w1_shuffled = w1_q
    if in_dtype in ("fp8", "a8w4"):
        w1_rows = experts * (2 * inter_dim)
        w1_cols = model_dim // 2 if is_a8w4 else model_dim
        w1_shuffled = fp4_utils.preshuffle_b_16x16(
            w1_q.contiguous().view(w1_rows, w1_cols),
            w1_rows,
            w1_cols,
        ).view_as(w1_q)

    # Flatten W1 for our FlyDSL kernel (treat expert dim as part of N).
    if uses_fp4_weight_layout:
        w1_shuffled_flat = w1_shuffled.view(experts * (2 * inter_dim), model_dim // 2)
        w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim // 2)
        scale_w1_ref_flat = None if scale_w1 is None else scale_w1.view(experts * (2 * inter_dim), model_dim // 32)
    else:
        w1_shuffled_flat = w1_shuffled.view(experts * (2 * inter_dim), model_dim)
        w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim)
        scale_w1_ref_flat = None if scale_w1 is None else scale_w1.view(experts * (2 * inter_dim), model_dim // 32)

    x_q = (
        x_q.contiguous().view(tokens, model_dim // 2)
        if is_fp4
        else x_q.contiguous().view(tokens, model_dim)
    )
    w_kernel = w1_shuffled_flat.contiguous()
    if uses_fp4_weight_layout:
        w_kernel = w_kernel.view(experts * (2 * inter_dim), model_dim // 2)
    else:
        w_kernel = w_kernel.view(experts * (2 * inter_dim), model_dim)

    scale_x_1d = scale_x.view(-1).contiguous()
    if scale_w1 is None:
        scale_w1_1d = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        scale_w1_1d = scale_w1.view(-1).contiguous()
    sorted_weights_1d = sorted_weights.contiguous().view(-1)  # [sorted_size]

    # Output: [tokens, topk, inter_dim] fp16
    out = torch.zeros((tokens, topk, inter_dim), device=device, dtype=torch.float16)

    exe = compile_moe_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        in_dtype=in_dtype,
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

    if bool(benchmark_mode) and not bool(test_graph):
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
        if in_dtype == "fp8":
            x_ref = _dequant_blockscale_fp8(x_q.view(tokens, model_dim), scale_x.view(tokens, model_dim // 32))
            sx_ref = None
        elif is_fp4:
            x_ref = _dequant_blockscale_fp4(x_q.view(tokens, model_dim // 2), scale_x.view(tokens, model_dim // 32), model_dim)
            sx_ref = None
        else:  # a8w4
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
        else:  # fp4
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

        rtol = 0.5 if (is_a8w4 or is_fp4) else 0.25
        atol = 0.5 if is_a8w4 else 0.25
        assert verify_output(out.to(torch.float32), ref, rtol=rtol, atol=atol)

    # Note: kernel launches full expert-block range; effective work is gated by num_valid_ids.
    flops = 2 * tokens * topk * (2 * inter_dim) * _orig_model_dim
    tflops = flops / (us / 1e6) / 1e12

    # Rough bytes-moved accounting (same spirit as GEMM tests: count each tensor once).
    bytes_moved = 0
    bytes_x = tokens * _orig_model_dim  # 1B elements (fp4/fp8/a8w4)
    bytes_w = (experts * (2 * inter_dim) * _orig_model_dim) // (2 if is_a8w4 else 1)
    bytes_out = tokens * topk * inter_dim * 2
    bytes_scale_x = tokens * 4
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
    seed: int = 0,
    num_iters: int = 5,
    num_warmup: int = 2,
    moe_sort_mode: Optional[str] = None,
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
    compile_fn=None,
    kernel_name: str = "moe_gemm2",
    use_reduce: bool = False,
    use_valid_mask: bool = False,
    test_graph: bool = False,
    waves_per_eu: Optional[int] = None,
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
):
    """MoE stage2 (gemm2): out2[t] = sum_{slot} ( out1[t,slot] @ W2[expert]^T ) with optional routed weight."""

    # Parameter sanity checks with actionable hints (avoid bare AssertionError).
    if model_dim % tile_n != 0:
        raise ValueError(
            f"Invalid stage2 tiling: model_dim ({model_dim}) must be divisible by tile_n2 ({tile_n})."
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

    if in_dtype not in ("fp4", "fp8", "a8w4"):
        raise ValueError(f"in_dtype must be one of ('fp4','fp8','a8w4'), got {in_dtype!r}")
    is_fp4 = in_dtype == "fp4"
    is_a8w4 = in_dtype == "a8w4"

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
        x_q, scale_x = _per_1x32_fp8_quant(x_fp32)
        w1_q, scale_w1 = _per_1x32_fp8_quant(w1_fp32)
        w2_q, scale_w2 = _per_1x32_fp8_quant(w2_fp32)
    else:  # a8w4
        x_q, scale_x = _per_1x32_fp8_quant(x_fp32)
        w1_q, scale_w1, _ = fp4_utils.per_1x32_f4_quant(w1_fp32)
        w2_q, scale_w2, _ = fp4_utils.per_1x32_f4_quant(w2_fp32)
        w1_q = w1_q.view(torch.uint8)
        scale_w1 = scale_w1.view(torch.uint8)
        w2_q = w2_q.view(torch.uint8)
        scale_w2 = scale_w2.view(torch.uint8)

    # --- K-dimension padding for non-aligned inter_dim (stage2 K=inter_dim) ---
    _orig_inter_dim = inter_dim
    if inter_dim % tile_k != 0:
        _pad_k2 = ((inter_dim + tile_k - 1) // tile_k) * tile_k - inter_dim
        if is_fp4 or is_a8w4:
            w2_q = F.pad(w2_q, (0, _pad_k2 // 2))
        else:
            w2_q = F.pad(w2_q, (0, _pad_k2))
        if scale_w2 is not None:
            scale_w2 = F.pad(scale_w2, (0, _pad_k2 // 32))
        inter_dim = inter_dim + _pad_k2

    # Preshuffle W2 — gfx1250 native kernels handle layout internally.
    uses_fp4_weight_layout = is_fp4 or is_a8w4
    w2_shuffled = w2_q
    if in_dtype in ("fp8", "a8w4"):
        w2_rows = experts * model_dim
        w2_cols = inter_dim // 2 if is_a8w4 else inter_dim
        w2_shuffled = fp4_utils.preshuffle_b_16x16(
            w2_q.contiguous().view(w2_rows, w2_cols),
            w2_rows,
            w2_cols,
        ).view_as(w2_q)

    # Stage2 input (A2): either provided (gemm1->quantize chaining) or built from stage1 reference.
    if a2_fp8_in is not None and a2_scale_in is not None:
        a2_q = a2_fp8_in
        a2_scale = a2_scale_in
    else:
        if is_fp4 or is_a8w4:
            w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim // 2)
            scale_w1_flat = None if scale_w1 is None else scale_w1.view(experts * (2 * inter_dim), model_dim // 32)
        else:  # fp8
            w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim)
            scale_w1_flat = None if scale_w1 is None else scale_w1.view(experts * (2 * inter_dim), model_dim // 32)
        if bool(skip_ref):
            raise RuntimeError(
                "run_moe_stage2(skip_ref=True) requires providing a2_fp8_in and a2_scale_in "
                "(so we don't have to run the huge torch reference stage1)."
            )
        if in_dtype == "fp8":
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
        if in_dtype in ("fp8", "a8w4"):
            a2_q, a2_scale = _per_1x32_fp8_quant(out1_ref)
        else:  # fp4
            a2_q = fp4_utils.random_fp4_packed(tokens * topk, inter_dim, device=device)
            a2_scale = fp4_utils.random_e8m0(tokens * topk, inter_dim // 32, device=device)

    # Pad A2 activation for non-aligned inter_dim (stage2 K-padding).
    if _orig_inter_dim != inter_dim:
        _pad_k2 = inter_dim - _orig_inter_dim
        if is_fp4:
            a2_q = F.pad(a2_q, (0, _pad_k2 // 2))
        else:
            a2_q = F.pad(a2_q, (0, _pad_k2))
        if a2_scale is not None:
            a2_scale = F.pad(a2_scale, (0, _pad_k2 // 32))

    # Flatten weights/scales for the kernel.
    if uses_fp4_weight_layout:
        w2_shuffled_flat = w2_shuffled.view(experts * model_dim, inter_dim // 2)
        scale_w2_ref_flat = None if scale_w2 is None else scale_w2.view(experts * model_dim, inter_dim // 32)
    else:
        w2_shuffled_flat = w2_shuffled.view(experts * model_dim, inter_dim)
        scale_w2_ref_flat = None if scale_w2 is None else scale_w2.view(experts * model_dim, inter_dim // 32)

    w2_flat = w2_shuffled_flat.contiguous().view(-1)
    w2_kernel = w2_flat
    if uses_fp4_weight_layout:
        w2_kernel = w2_kernel.view(experts * model_dim, inter_dim // 2)
    else:
        w2_kernel = w2_kernel.view(experts * model_dim, inter_dim)

    a2_scale_1d = a2_scale.view(-1).contiguous()
    if scale_w2 is None:
        w2_scale_1d = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        w2_scale_1d = scale_w2.view(-1).contiguous()
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
 
    if bool(benchmark_mode) and not bool(test_graph):
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
        else:  # fp4
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
        assert verify_output(out.to(torch.float32), ref2, rtol=0.5, atol=0.5)

    # Launches full expert-block range; effective work is gated by num_valid_ids.
    flops = 2 * tokens * topk * model_dim * _orig_inter_dim
    tflops = flops / (us / 1e6) / 1e12

    bytes_moved = 0
    bytes_a2 = tokens * topk * _orig_inter_dim  # 1B elements (fp4/fp8/a8w4)
    bytes_w2 = (experts * model_dim * _orig_inter_dim) // (2 if is_a8w4 else 1)
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
    # Print profile breakdown if the executor supports it
    if hasattr(exe, 'print_profile_stats'):
        exe.print_profile_stats()

    if return_outputs:
        return out, us
    return None


def run_moe_gemm_2stage(
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
    *,
    seed: int = 0,
    num_iters: int = 5,
    num_warmup: int = 2,
    moe_sort_mode: Optional[str] = None,
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
    """Single 2-stage test: gemm1 -> quantize -> gemm2, with routing built once."""
    if (not bool(use_reduce)) and bool(use_valid_mask):
        pytest.skip("valid_mask is only used in reduce mode (atomic mode ignores it).")
    if out_dtype in ("f32", "fp32", "float"):
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
    device = torch.device("cuda")

    if init_scale == 1.0:
        init_scale = 0.2
    s = float(init_scale)
    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32) * s
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * s
    w2_fp32 = torch.randn((experts, model_dim, inter_dim), device=device, dtype=torch.float32) * (s / math.sqrt(inter_dim))

    score = torch.rand((tokens, experts), device=device, dtype=torch.float32)
    topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)

    if moe_sort_mode is None:
        moe_sort_mode = "torch"

    routing = build_routing_buffers(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        experts=experts,
        model_dim=model_dim,
        tile_m=tile_m,
        moe_sort_mode=moe_sort_mode,
    )

    out1_fp16, _us1 = run_moe_stage1(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        in_dtype=in_dtype,
        tile_m=tile_m,
        tile_n=tile_n1,
        tile_k=tile_k1,
        doweight_stage1=bool(doweight_stage1),
        seed=seed,
        num_iters=num_iters,
        num_warmup=num_warmup,
        moe_sort_mode=moe_sort_mode,
        x_fp32_in=x_fp32,
        w1_fp32_in=w1_fp32,
        topk_ids_in=topk_ids,
        topk_weights_in=topk_weights,
        routing_in=routing,
        return_outputs=True,
        skip_ref=bool(skip_ref),
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

    a2_q, a2_scale = _prepare_a2_from_stage1(
        out1_fp16, in_dtype, tokens, topk, inter_dim,
        w_fp4_kernel=w_fp4_kernel, skip_ref=bool(skip_ref),
        topk_ids=topk_ids, experts=experts, device=device,
    )

    _out2_fp32, _us2 = run_moe_stage2(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        tile_m=tile_m,
        tile_n=tile_n2,
        tile_k=tile_k2,
        doweight_stage1=bool(doweight_stage1),
        seed=seed,
        num_iters=num_iters,
        num_warmup=num_warmup,
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



def _prepare_a2_from_stage1(out1_fp16, in_dtype, tokens, topk, inter_dim,
                            *, w_fp4_kernel=False, skip_ref=False,
                            topk_ids=None, experts=None, device=None):
    """Convert stage1 fp16 output to appropriate stage2 activation input."""
    if w_fp4_kernel:
        if in_dtype == "fp4":
            dev = device or out1_fp16.device
            if skip_ref:
                a2_q = fp4_utils.random_fp4_packed(tokens * topk, inter_dim, device=dev).view(
                    tokens, topk, inter_dim // 2)
                a2_scale = fp4_utils.random_e8m0(tokens * topk, inter_dim // 32, device=dev).view(
                    tokens, topk, inter_dim // 32)
            else:
                f32 = out1_fp16.to(torch.float32)
                a2_fp4, a2_scale_raw, _ = fp4_utils.per_1x32_f4_quant(f32.view(-1, inter_dim))
                a2_q = a2_fp4.view(torch.uint8).view(tokens, topk, inter_dim // 2)
                a2_scale = a2_scale_raw.view(torch.uint8).view(tokens, topk, inter_dim // 32)
        else:
            a2_q = out1_fp16.to(torch.float32)
            a2_scale = None
        return a2_q, a2_scale

    if in_dtype == "fp4":
        f32 = out1_fp16.to(torch.float32)
        a2_fp4, a2_scale_raw, _ = fp4_utils.per_1x32_f4_quant(f32.view(-1, inter_dim))
        a2_q = a2_fp4.view(torch.uint8).view(tokens, topk, inter_dim // 2)
        a2_scale = a2_scale_raw.view(torch.uint8).view(tokens, topk, inter_dim // 32)
    elif in_dtype in ("fp8", "a8w4"):
        a2_q, a2_scale = _per_1x32_fp8_quant(out1_fp16.to(torch.float32))
    else:
        raise ValueError(f"in_dtype must be one of ('fp4','fp8','a8w4'), got {in_dtype!r}")
    return a2_q, a2_scale



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
        moe_sort_mode="torch",
        x_fp32_in=x_fp32,
        w1_fp32_in=w1_fp32,
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


# ---------------------------------------------------------------------------
# FP4 smoke tests
# ---------------------------------------------------------------------------

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
        return_outputs=True,
        skip_ref=True,
    )

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
    run_moe_gemm_2stage(
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
        seed=seed,
        num_iters=1,
        num_warmup=1,
        moe_sort_mode="torch",
        skip_ref=False,
        use_reduce=True,
        use_tdm_store=False,
        w_fp4_kernel=True,
        test_graph=False,
        use_valid_mask=False,
    )


# ---------------------------------------------------------------------------
# Main parametrized 2-stage test — MXScale dtypes only
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n1, tile_k1, tile_n2, tile_k2, doweight_stage1",
    [
        pytest.param(64, 256, 128, 4, 2, 16, 64, 128, 64, 128, False, id="S"),
        pytest.param(129, 1024, 256, 8, 2, 32, 128, 128, 128, 128, False, id="M"),
        pytest.param(333, 4096, 2048, 17, 9, 64, 128, 128, 256, 128, False, id="L", marks=pytest.mark.large_shape),
    ],
)
@pytest.mark.parametrize("in_dtype", ["fp4", "fp8", "a8w4"])
@pytest.mark.parametrize("out_dtype", ["f16"], ids=["out_f16"])
@pytest.mark.parametrize("use_reduce", [False, True], ids=["atomic", "reduce"])
@pytest.mark.parametrize("use_valid_mask", [False, True], ids=["nomask", "mask"])
@pytest.mark.parametrize("test_graph", [
    pytest.param(False, id="eager"),
    pytest.param(True, id="graph"),
])
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
):
    run_moe_gemm_2stage(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n1=tile_n1,
        tile_k1=tile_k1,
        tile_n2=tile_n2,
        tile_k2=tile_k2,
        doweight_stage1=doweight_stage1,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        use_reduce=use_reduce,
        use_valid_mask=use_valid_mask,
        test_graph=test_graph,
    )


# ---------------------------------------------------------------------------
# Standalone stage2 test (atomic vs reduce comparison)
# ---------------------------------------------------------------------------

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
        in_dtype=in_dtype,
        seed=seed,
        num_iters=num_iters,
        num_warmup=num_warmup,
        moe_sort_mode="torch",
        skip_ref=False,
    )

    run_moe_stage2(**common_args, kernel_name="moe_gemm2_atomic")

    run_moe_stage2(
        **common_args,
        compile_fn=_make_reduce_mode_compile_fn(use_flydsl_reduce=False),
        kernel_name="moe_gemm2_reduce_torch",
    )

    run_moe_stage2(
        **common_args,
        use_reduce=True,
        kernel_name="moe_gemm2_reduce_flydsl",
    )

    run_moe_stage2(
        **common_args,
        use_reduce=True,
        use_valid_mask=True,
        kernel_name="moe_gemm2_reduce_flydsl_valid_mask",
    )
