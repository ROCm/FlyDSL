#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE GEMM tests for WMMA data types (fp16/bf16) on gfx1250.
"""
import argparse
import logging
import math
import os
import sys
from typing import Tuple, Optional

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
from tests.test_common import verify_output, run_perftest
from flydsl.runtime.device import get_rocm_arch
from tests.kernels.benchmark_common import (
    bench_kernel_us as _bench_kernel_us,
    add_moe_bench_args,
    moe_bench_main,
)

ARCH = get_rocm_arch()

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

if not str(ARCH).startswith("gfx1250"):
    pytest.skip(f"MoE 2stage gfx1250 tests require gfx1250, got {ARCH}", allow_module_level=True)


# Optional: use aiter's exact routing/sorting implementation (matches `aiter/op_tests/test_moe_2stage.py`).
# Some environments ship aiter python but miss required JIT .so dependencies; we fall back gracefully.
try:
    import aiter
    from aiter.fused_moe import moe_sorting as aiter_moe_sorting

    HAS_AITER = True
except Exception:
    HAS_AITER = False

# Kernel implementations live under `kernels/`; this test file is the harness.
from kernels.moe_gemm_2stage_wmma_gfx1250 import (
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
    in_dtype: str = "fp16",
    seed: int = 0,
    num_iters: int = 5,
    num_warmup: int = 2,
    moe_sort_mode: Optional[str] = None,
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
    assert model_dim % tile_k == 0, f"model_dim={model_dim} must be divisible by tile_k={tile_k}"
    assert inter_dim % tile_n == 0

    if in_dtype not in ("fp16", "bf16"):
        raise ValueError(f"in_dtype must be 'fp16' or 'bf16', got {in_dtype!r}")

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

    cast = torch.float16 if in_dtype == "fp16" else torch.bfloat16
    x_q = x_fp32.to(cast)
    w1_q = w1_fp32.to(cast)

    w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim)
    x_q = x_q.contiguous().view(tokens, model_dim)
    w_kernel = w1_q_flat.contiguous()

    scale_x_1d = torch.empty((0,), device=device, dtype=torch.float32)
    scale_w1_1d = torch.empty((0,), device=device, dtype=torch.float32)
    sorted_weights_1d = sorted_weights.contiguous().view(-1)  # [sorted_size]

    # Output: [tokens, topk, inter_dim] fp16
    out = torch.zeros((tokens, topk, inter_dim), device=device, dtype=torch.float16)

    from kernels.moe_gemm_2stage_wmma_gfx1250 import compile_moe_gemm1
    exe = compile_moe_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        in_dtype=in_dtype,
        group_size=-1,
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
        ref = torch_moe_gemm1(
            x_q,
            w1_q_flat,
            None,
            None,
            topk_ids.to(torch.int64),
            topk_weights,
            inter_dim=inter_dim,
            doweight_stage1=doweight_stage1,
        )
        assert verify_output(out.to(torch.float32), ref, rtol=0.25, atol=0.25)

    # Note: kernel launches full expert-block range; effective work is gated by num_valid_ids.
    flops = 2 * tokens * topk * (2 * inter_dim) * model_dim
    tflops = flops / (us / 1e6) / 1e12

    # Rough bytes-moved accounting (same spirit as GEMM tests: count each tensor once).
    x_elem_bytes = 2
    bytes_x = tokens * model_dim * x_elem_bytes
    bytes_w = experts * (2 * inter_dim) * model_dim
    bytes_out = tokens * topk * inter_dim * 2
    bytes_scale_x = tokens * 4
    bytes_scale_w = experts * (2 * inter_dim) * 4
    bytes_route = (
        int(sorted_weights.numel()) * 4
        + int(sorted_token_ids.numel()) * 4
        + int(sorted_expert_ids.numel()) * 4
    )
    bytes_moved = bytes_x + bytes_w + bytes_out + bytes_scale_x + bytes_scale_w + bytes_route
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
    in_dtype: str = "fp16",
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
    if in_dtype not in ("fp16", "bf16"):
        raise ValueError(f"in_dtype must be 'fp16' or 'bf16', got {in_dtype!r}")

    # Parameter sanity checks with actionable hints (avoid bare AssertionError).
    if model_dim % tile_n != 0:
        raise ValueError(
            f"Invalid stage2 tiling: model_dim ({model_dim}) must be divisible by tile_n2 ({tile_n})."
        )
    if inter_dim % tile_k != 0:
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

    cast = torch.float16 if in_dtype == "fp16" else torch.bfloat16
    x_q = x_fp32.to(cast)
    w1_q = w1_fp32.to(cast)
    w2_q = w2_fp32.to(cast)

    if a2_fp8_in is not None:
        a2_q = a2_fp8_in
        a2_scale = a2_scale_in
    else:
        w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim)
        if bool(skip_ref):
            raise RuntimeError(
                "run_moe_stage2(skip_ref=True) requires providing a2_fp8_in "
                "(so we don't have to run the huge torch reference stage1)."
            )
        out1_ref = torch_moe_gemm1(
            x_q,
            w1_q_flat,
            None,
            None,
            topk_ids.to(torch.int64),
            topk_weights,
            inter_dim=inter_dim,
            doweight_stage1=bool(doweight_stage1),
        )
        if in_dtype == "fp16":
            a2_q = out1_ref.to(torch.float16)
            a2_scale = None
        else:
            a2_q = out1_ref.to(torch.bfloat16)
            a2_scale = None

    w2_shuffled_flat = w2_q.view(experts * model_dim, inter_dim)
    w2_kernel = w2_shuffled_flat.contiguous().view(-1)

    a2_scale_1d = torch.empty((0,), device=device, dtype=torch.float32)
    w2_scale_1d = torch.empty((0,), device=device, dtype=torch.float32)
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
        group_size=-1,
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
        ref2 = torch_moe_gemm2(
            a2_q,
            w2_q,
            None,
            None,
            topk_ids.to(torch.int64),
            topk_weights,
            model_dim=model_dim,
            doweight_stage2=doweight_stage2,
        )
        assert verify_output(out.to(torch.float32), ref2, rtol=0.5, atol=0.5)

    # Launches full expert-block range; effective work is gated by num_valid_ids.
    flops = 2 * tokens * topk * model_dim * inter_dim
    tflops = flops / (us / 1e6) / 1e12

    a2_elem_bytes = 2
    bytes_a2 = tokens * topk * inter_dim * a2_elem_bytes
    bytes_w2 = experts * model_dim * inter_dim
    bytes_out = tokens * model_dim * (2 if out_torch_dtype == torch.float16 else 4)
    bytes_scale_a2 = tokens * topk * 4
    bytes_scale_w2 = experts * model_dim * 4
    bytes_route = (
        int(sorted_weights.numel()) * 4
        + int(sorted_token_ids.numel()) * 4
        + int(sorted_expert_ids.numel()) * 4
    )
    bytes_moved = bytes_a2 + bytes_w2 + bytes_out + bytes_scale_a2 + bytes_scale_w2 + bytes_route
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
    init_scale: float = 0.2,
    skip_ref: bool = False,
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
    out_s = str(out_dtype).strip().lower()
    if bool(use_reduce) and out_s in ("f32", "fp32", "float"):
        pytest.skip("reduce mode does not support out_dtype='f32' (compile_moe_gemm2(accumulate=False) forbids it).")
    device = torch.device("cuda")

    s = float(init_scale)
    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32) * s
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * s
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

    _shared = dict(
        seed=seed, num_iters=num_iters, num_warmup=num_warmup,
        moe_sort_mode=moe_sort_mode,
        x_fp32_in=x_fp32, w1_fp32_in=w1_fp32,
        topk_ids_in=topk_ids, topk_weights_in=topk_weights,
        routing_in=routing, return_outputs=True,
        skip_ref=bool(skip_ref), test_graph=test_graph,
        benchmark_mode=bool(benchmark_mode), flush_l2=bool(flush_l2),
        num_buffers=int(num_buffers), use_tdm_store=bool(use_tdm_store),
        inst_prefetch=bool(inst_prefetch), wave_specialized_tdm=bool(wave_specialized_tdm),
        cluster_m=int(cluster_m), cluster_n=int(cluster_n),
    )

    out1_fp16, _us1 = run_moe_stage1(
        tokens=tokens, model_dim=model_dim, inter_dim=inter_dim,
        experts=experts, topk=topk, in_dtype=in_dtype,
        tile_m=tile_m, tile_n=tile_n1, tile_k=tile_k1,
        doweight_stage1=bool(doweight_stage1),
        **_shared,
    )

    a2_q, a2_scale = _prepare_a2_from_stage1(out1_fp16, in_dtype)

    _out2_fp32, _us2 = run_moe_stage2(
        tokens=tokens, model_dim=model_dim, inter_dim=inter_dim,
        experts=experts, topk=topk, in_dtype=in_dtype, out_dtype=out_dtype,
        tile_m=tile_m, tile_n=tile_n2, tile_k=tile_k2,
        doweight_stage1=bool(doweight_stage1),
        w2_fp32_in=w2_fp32,
        a2_fp8_in=a2_q, a2_scale_in=a2_scale,
        use_reduce=bool(use_reduce), use_valid_mask=use_valid_mask,
        **_shared,
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
        in_dtype: str = "fp16",
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


def _bench_setup_data(tokens, model_dim, inter_dim, experts, topk, tile_m, seed=42):
    """Build random MoE data + routing buffers for bench sweeps."""
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


def _prepare_a2_from_stage1(out1_fp16: torch.Tensor, in_dtype: str):
    """Convert stage1 fp16 output to stage2 activation input (fp16 or bf16)."""
    if in_dtype == "fp16":
        return out1_fp16, None
    if in_dtype == "bf16":
        return out1_fp16.to(torch.bfloat16), None
    raise ValueError(f"in_dtype must be 'fp16' or 'bf16', got {in_dtype!r}")


def _bench_prepare_a2(out1_fp16, _tokens, _topk, _inter_dim, in_dtype):
    return _prepare_a2_from_stage1(out1_fp16, in_dtype)


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
        description="MoE 2-stage (FlyDSL WMMA fp16/bf16) test/benchmark on gfx1250.",
    )
    parser.add_argument(
        "--in_dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "all"],
        help="Kernel input dtype: fp16, bf16, or all (default: fp16).",
    )
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

    parser.add_argument("--moe_sort_mode", type=str, default=None, choices=["aiter", "torch"], help="Routing buffer build mode (aiter moe_sorting vs torch fallback).")
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

    # ── Benchmark sweep mode (--bench) ──
    add_moe_bench_args(parser)

    args = parser.parse_args()

    # ── Bench sweep mode: run and exit ──
    if args.bench:
        moe_bench_main(
            args,
            stage1_fn=run_moe_stage1,
            stage2_fn=run_moe_stage2,
            setup_data_fn=_bench_setup_data,
            prepare_a2_fn=_bench_prepare_a2,
        )
        sys.exit(0)

    model_dim, inter_dim = args.dim

    tile_n2 = int(args.tile_n2) if args.tile_n2 is not None else int(args.tile_n) * 2
    tile_k2 = int(args.tile_k2) if args.tile_k2 is not None else args.tile_k

    # Determine which gemm2 modes to run.
    if args.gemm2_mode == "both":
        reduce_flags = [False, True]
    elif args.gemm2_mode == "reduce":
        reduce_flags = [True]
    else:  # "atomic"
        reduce_flags = [False]

    # Common CLI arguments shared across stage1/stage2/2stage calls.
    _common = dict(
        tokens=int(args.tokenNum), model_dim=int(model_dim), inter_dim=int(inter_dim),
        experts=int(args.expert), topk=int(args.topk),
        doweight_stage1=bool(args.doweight_stage1),
        tile_m=int(args.tile_m),
        seed=int(args.seed), num_iters=int(args.num_iters), num_warmup=int(args.num_warmup),
        moe_sort_mode=args.moe_sort_mode,
        skip_ref=bool(args.skip_ref),
        test_graph=bool(args.test_graph),
        benchmark_mode=bool(args.benchmark),
        flush_l2=not bool(args.no_flush_l2),
        num_buffers=int(args.num_buffers),
        use_tdm_store=bool(args.use_tdm_store),
        inst_prefetch=bool(args.inst_prefetch),
        wave_specialized_tdm=bool(args.wave_specialized_tdm),
        cluster_m=int(args.cluster_m), cluster_n=int(args.cluster_n),
    )

    def run_one(dt: str, use_reduce: bool):
        out_s = str(args.out_dtype).strip().lower()
        if bool(use_reduce) and out_s in ("f32", "fp32", "float"):
            print("[skip] reduce mode does not support out_dtype='f32'")
            return
        if (not bool(use_reduce)) and bool(args.use_valid_mask):
            print("[skip] valid_mask is only used in reduce mode (atomic ignores it)")
            return
        run_moe_gemm_2stage(
            **_common, in_dtype=dt,
            out_dtype=str(args.out_dtype),
            tile_n1=int(args.tile_n), tile_k1=int(args.tile_k),
            tile_n2=tile_n2, tile_k2=tile_k2,
            use_reduce=use_reduce,
            use_valid_mask=bool(args.use_valid_mask),
        )
        print(f"PASSED: dtype={dt} reduce={use_reduce}")

    # Run 2-stage (gemm1 -> quantize -> gemm2) aiter-style test/benchmark.
    # Expand "all" to all supported dtypes.
    in_dtypes = args.in_dtype.split(",")
    if "all" in in_dtypes:
        in_dtypes = ["fp16", "bf16"]
    for dt in in_dtypes:
        for use_reduce in reduce_flags:
            run_one(dt, use_reduce)



# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("waves_per_eu", [1, 2], ids=["wpe1", "wpe2"])
def test_moe_2stage_waves_per_eu_smoke(waves_per_eu: int):
    """Smoke test for stage1/stage2 waves_per_eu plumbing on gfx1250."""
    _shape = dict(tokens=32, model_dim=256, inter_dim=128, experts=4, topk=2, tile_m=16)
    _fast = dict(num_iters=1, num_warmup=1, return_outputs=True, skip_ref=True,
                 in_dtype="fp16", doweight_stage1=False, waves_per_eu=waves_per_eu)

    stage1_out, _ = run_moe_stage1(**_shape, tile_n=64, tile_k=128, **_fast)
    stage2_out, _ = run_moe_stage2(
        **_shape, tile_n=64, tile_k=128, out_dtype="f16",
        a2_fp8_in=stage1_out.to(torch.float16), a2_scale_in=None, **_fast,
    )
    assert torch.isfinite(stage1_out).all()
    assert torch.isfinite(stage2_out).all()


# ---------------------------------------------------------------------------
# Main parametrized 2-stage test — WMMA dtypes (fp16 / bf16)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n1, tile_k1, tile_n2, tile_k2, doweight_stage1",
    [
        pytest.param(64, 256, 128, 4, 2, 16, 64, 128, 64, 128, False, id="S"),
        pytest.param(129, 1024, 256, 8, 2, 32, 128, 128, 128, 128, False, id="M"),
        pytest.param(333, 4096, 2048, 17, 9, 64, 128, 128, 256, 128, False, id="L", marks=pytest.mark.large_shape),
    ],
)
@pytest.mark.parametrize("in_dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("out_dtype", ["f16", "f32"], ids=["out_f16", "out_f32"])
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
