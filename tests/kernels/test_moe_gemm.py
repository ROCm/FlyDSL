#!/usr/bin/env python3
import argparse
import logging
import math
import os
import sys
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
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.kernels.test_ref import torch_moe_gemm1, torch_moe_gemm2
from tests.utils import pertoken_quant, shuffle_weight
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
from kernels.moe_gemm_2stage import (
    compile_moe_gemm1,
    compile_moe_gemm2,
    compile_moe_gemm2_ex,
    compile_moe_reduction,
    MoeGemm2Mode,
)

logging.basicConfig(level=logging.INFO)

# Reduce noisy aiter log spam (e.g. "type hints mismatch, override to --> ...") so test output
# stays readable. You can override via env: FLIR_AITER_LOG_LEVEL=INFO/WARNING/ERROR.
_aiter_level = os.environ.get("FLIR_AITER_LOG_LEVEL", "ERROR").upper().strip()
try:
    logging.getLogger("aiter").setLevel(getattr(logging, _aiter_level, logging.ERROR))
except Exception:
    # Best-effort only; never break tests due to logging configuration.
    pass

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


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
        (10240, 1024, 256, 128, 8, False),
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
    """Build routing buffers once (CK format), reusable across stage1 + stage2.

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
        # num_valid_ids[0] == total padded rows (CK-style); kernels use this for early-exit.
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
    # w2 is required by aiter CK API even for stage1; keep it allocated to avoid null ptr.
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

    if in_dtype not in ("fp8", "fp16", "int8", "int8smooth", "int4"):
        raise ValueError(
            f"in_dtype must be one of ('fp8','fp16','int8','int8smooth','int4'), got {in_dtype!r}"
        )
    is_int4 = in_dtype == "int4"
    is_int8 = in_dtype in ("int8", "int8smooth", "int4")
    is_int8smooth = in_dtype == "int8smooth"

    # Quantize inputs / weights.
    if in_dtype == "fp8":
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=DTYPE_FP8)  # [tokens,K], [tokens,1]
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=DTYPE_FP8)  # [E,2*inter,K], [E,2*inter,1]
    # w2 is not used by our kernel, but required by CK stage1 API
        w2_q, _scale_w2_unused = pertoken_quant(w2_fp32, quant_dtype=DTYPE_FP8)
    elif in_dtype == "fp16":
        x_q = x_fp32.to(torch.float16)
        w1_q = w1_fp32.to(torch.float16)
        w2_q = w2_fp32.to(torch.float16)
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
        # Match CK moe_smoothquant layout: slot-major [topk*tokens, K].
        x_q = x_q.permute(1, 0, 2).contiguous()
        scale_x = scale_x.permute(1, 0, 2).contiguous()
        # W quantization is unchanged for this harness (same as aiter perf tests: smooth scales
        # exercise the API rather than implementing the exact SQ calibration workflow).
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.int8)
        w2_q, _scale_w2_unused = pertoken_quant(w2_fp32, quant_dtype=torch.int8)
    else:
        # W4A8: X is int8, W is int4 packed (host packs from int8 values in [-8,7]).
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.int8)
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.int8, dtypeMax=7)
        w2_q, _scale_w2_unused = pertoken_quant(w2_fp32, quant_dtype=torch.int8, dtypeMax=7)

    # Preshuffle weights (aiter/CK layout) on the *unpacked* tensor.
    w1_shuffled = shuffle_weight(w1_q)
    w2_shuffled = shuffle_weight(w2_q) if in_dtype == "fp8" else None

    # Flatten W1 for our flir kernel (treat expert dim as part of N).
    w1_shuffled_flat = w1_shuffled.view(experts * (2 * inter_dim), model_dim)
    w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim)
    scale_w1_flat = None if scale_w1 is None else scale_w1.view(experts * (2 * inter_dim), 1)

    # No host-side padding: keep tensors contiguous and rely on kernel-side resource sizes / early-exit.
    x_q = (
        x_q.contiguous().view(tokens * topk, model_dim)
        if is_int8smooth
        else x_q.contiguous().view(tokens, model_dim)
    )
    w_kernel = (
        _pack_shuffled_int8_to_packed_int4_no_perm(w1_shuffled_flat) if is_int4 else w1_shuffled_flat
    ).contiguous()
    if not is_int4:
        w_kernel = w_kernel.view(experts * (2 * inter_dim), model_dim)

    # Flatten scales to 1D memrefs (fp16 path uses 0-sized scale tensors; kernel ignores them).
    if scale_x is None:
        scale_x_1d = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        scale_x_1d = scale_x.view(-1).contiguous()  # [tokens] or [tokens*topk] for int8smooth
    if scale_w1_flat is None:
        scale_w1_1d = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        scale_w1_1d = scale_w1_flat.view(-1).contiguous()  # [rows]
    sorted_weights_1d = sorted_weights.contiguous().view(-1)  # [sorted_size]

    # Output: [tokens, topk, inter_dim] fp16
    out = torch.empty((tokens, topk, inter_dim), device=device, dtype=torch.float16)

    from kernels.moe_gemm_2stage import compile_moe_gemm1
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
    )

    def launch(o, x, w, sx, sw, st, eids, sw_sorted):
        stream_ptr = torch.cuda.current_stream().cuda_stream
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
            stream_ptr,
        )

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
        )

        rtol = 0.5 if is_int4 else 0.25
        atol = 0.5 if is_int4 else 0.25
        assert verify_output(out.to(torch.float32), ref, rtol=rtol, atol=atol)

    # Note: kernel launches full expert-block range; effective work is gated by num_valid_ids.
    flops = 2 * tokens * topk * (2 * inter_dim) * model_dim
    tflops = flops / (us / 1e6) / 1e12

    # Rough bytes-moved accounting (same spirit as GEMM tests: count each tensor once).
    bytes_moved = 0
    bytes_moved += (tokens * topk if is_int8smooth else tokens) * model_dim * 1  # x int8/fp8
    bytes_moved += (experts * (2 * inter_dim) * model_dim) // (2 if is_int4 else 1)  # w (packed for int4)
    bytes_moved += tokens * topk * inter_dim * 2  # out fp16 (logical)
    bytes_moved += (tokens * topk if is_int8smooth else tokens) * 4  # scale_x f32 (1D)
    bytes_moved += experts * (2 * inter_dim) * 4  # scale_w f32 (1D)
    bytes_moved += int(sorted_weights.numel()) * 4  # sorted_weights f32
    bytes_moved += int(sorted_token_ids.numel()) * 4  # sorted_token_ids i32
    bytes_moved += int(sorted_expert_ids.numel()) * 4  # sorted_expert_ids i32
    tbps = bytes_moved / 1e12 / (us / 1e6)

    print(
        f"FLIR MoE stage1[{in_dtype}]: "
        f"{us:.1f} us, "
        f"{tflops:.2f} TFLOPS(logical, M={tokens*topk}), "
        f"{tbps:.3f} TB/s (doweight_stage1={doweight_stage1})"
    )
    # Compare + benchmark vs aiter CK stage1 (optional; enabled by default when aiter is runnable).
    if compare_aiter_ck is None:
        compare_ck = os.environ.get("COMPARE_AITER_CK", "1" if HAS_AITER else "0") == "1"
    else:
        compare_ck = bool(compare_aiter_ck)
    # aiter CK paths are fp8-only in our setup.
    compare_ck = compare_ck and (in_dtype == "fp8")
    if compare_ck:
        if not HAS_AITER:
            pytest.skip("aiter not available; cannot compare to CK moe stage1.", allow_module_level=False)
        try:
            from aiter.ops.moe_op import ck_moe_stage1_fwd
            from aiter.ops.enum import QuantType, ActivationType

            out_ck = torch.empty((tokens, topk, inter_dim), device=device, dtype=torch.float16)

            # aiter CK expects w1/w2 with expert dimension preserved.
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

            # Benchmark CK stage1
            # Align with aiter swap rules:
            # - CK takes quantized activations (fp8) + per-token scale
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

            # Correctness: flir vs CK
            assert verify_output(out.to(torch.float32), out_ck.to(torch.float32), rtol=0.25, atol=0.25, msg="flir vs aiter:")

            # Perf print: use the same flop model for both
            flops = 2 * tokens * topk * (2 * inter_dim) * model_dim
            tflops_ck = flops / (us_ck / 1e6) / 1e12
            print(f"[aiter CK] stage1: {us_ck:.1f} us, {tflops_ck:.2f} TFLOPS, flir vs aiter speedups: {tflops / tflops_ck:.2f}x")
        except Exception as e:
            # Treat CK compare as best-effort: many environments can import `aiter` but can't load
            # the full JIT .so dependency chain. Don't fail the FLIR test suite for that.
            logging.warning(f"Skipping aiter CK moe stage1 compare (not runnable here): {e}")
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

    if in_dtype not in ("fp8", "fp16", "int8", "int8smooth", "int4"):
        raise ValueError(
            f"in_dtype must be one of ('fp8','fp16','int8','int8smooth','int4'), got {in_dtype!r}"
        )
    is_int4 = in_dtype == "int4"
    is_int8 = in_dtype in ("int8", "int8smooth", "int4")
    is_int8smooth = in_dtype == "int8smooth"

    # Quantize inputs / weights.
    if in_dtype == "fp8":
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
    else:
        # W4A8: A2 is int8, W2 is int4 packed (host packs from int8 values in [-8,7]).
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.int8)
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.int8, dtypeMax=7)
        w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=torch.int8, dtypeMax=7)

    # Preshuffle weights (aiter/CK layout) on the *unpacked* tensor.
    w1_shuffled = shuffle_weight(w1_q)
    w2_shuffled = shuffle_weight(w2_q)

    # Stage2 input (A2): either provided (gemm1->quantize chaining) or built from stage1 reference.
    if a2_fp8_in is not None and (a2_scale_in is not None or in_dtype == "fp16"):
        a2_q = a2_fp8_in
        a2_scale = a2_scale_in
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
        elif in_dtype == "fp16":
            a2_q = out1_ref.to(torch.float16)
            a2_scale = None
        else:
            if is_int8smooth:
                # Apply a per-expert smooth scale to A2 before W8A8 quantization.
                smooth_scale2 = (0.75 + 0.5 * torch.rand((experts, inter_dim), device=device, dtype=torch.float32))
                out1_ref = out1_ref * smooth_scale2[topk_ids.to(torch.int64)]
            a2_q, a2_scale = pertoken_quant(out1_ref, quant_dtype=torch.int8)

    # Flatten weights/scales for the kernel.
    w2_shuffled_flat = w2_shuffled.view(experts * model_dim, inter_dim)
    scale_w2_flat = None if scale_w2 is None else scale_w2.view(experts * model_dim, 1)

    # For W4A8, pack preshuffled int8 weights into packed int4 bytes.
    w2_kernel = w2_shuffled_flat
    if is_int4:
        w2_kernel = _pack_shuffled_int8_to_packed_int4_no_perm(w2_shuffled_flat)

    w2_flat = w2_kernel.contiguous().view(-1)
    w2_kernel = w2_flat
    if not is_int4:
        w2_kernel = w2_kernel.view(experts * model_dim, inter_dim)

    # Flatten scales to 1D memrefs (fp16 path uses 0-sized scale tensors; kernel ignores them).
    if a2_scale is None:
        a2_scale_1d = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        a2_scale_1d = a2_scale.view(-1).contiguous()  # [tokens*topk]
    if scale_w2_flat is None:
        w2_scale_1d = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        w2_scale_1d = scale_w2_flat.view(-1).contiguous()  # [experts*model_dim]
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
    exe = compile_fn(
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
    )
    is_reduce_exe = (getattr(exe, "mode", None) == MoeGemm2Mode.REDUCE) or bool(use_reduce)

    def launch(o, x, w, sx, sw, st, eids, sw_sorted):
        stream_ptr = torch.cuda.current_stream().cuda_stream
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
                stream_ptr,
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
                stream_ptr,
            )
 
    # NOTE: stage2 uses atomic-add into `out`, so we cannot reuse the same output buffer
    # across perf iterations for correctness. Time into a dedicated buffer, then run
    # a single clean launch for correctness verification below.
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
        )
        assert verify_output(out.to(torch.float32), ref2, rtol=0.5, atol=0.5)

    # Launches full expert-block range; effective work is gated by num_valid_ids.
    flops = 2 * tokens * topk * model_dim * inter_dim
    tflops = flops / (us / 1e6) / 1e12

    bytes_moved = 0
    bytes_moved += tokens * topk * inter_dim * 1  # a2 fp8 (logical)
    bytes_moved += (experts * model_dim * inter_dim) // (2 if is_int4 else 1)  # w2 (packed for int4)
    bytes_moved += tokens * model_dim * (2 if out_torch_dtype == torch.float16 else 4)  # out
    bytes_moved += tokens * topk * 4  # a2_scale f32 (logical)
    bytes_moved += experts * model_dim * 4  # w2_scale f32 (1D)
    bytes_moved += int(sorted_weights.numel()) * 4
    bytes_moved += int(sorted_token_ids.numel()) * 4
    bytes_moved += int(sorted_expert_ids.numel()) * 4
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(
        f"FLIR MoE stage2 [{kernel_name}] {in_dtype} {'reduce' if use_reduce else 'atomic'} | "
        f"{model_dim}x{inter_dim}, E={experts}, K={topk}, M_eff={tokens*topk} | "
        f"{us:.1f} us, {tflops:.2f} TFLOPS, {tbps:.3f} TB/s"
    )
    # Optional compare vs aiter CK stage2.
    if compare_aiter_ck is None:
        compare_ck = os.environ.get("COMPARE_AITER_CK", "1" if HAS_AITER else "0") == "1"
    else:
        compare_ck = bool(compare_aiter_ck)
    # aiter CK paths are fp8-only in our setup.
    compare_ck = compare_ck and (in_dtype == "fp8")
    if compare_ck:
        if not HAS_AITER:
            pytest.skip("aiter not available; cannot compare to CK moe stage2.", allow_module_level=False)
        try:
            from aiter.ops.moe_op import ck_moe_stage2_fwd
            from aiter.ops.enum import QuantType, ActivationType

            # CK stage2 output type is fp16 in many builds; keep fp16 for compatibility.
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

            # Perf print (report both executed vs logical FLOPs, same convention as FLIR).
            flops = 2 * tokens * topk * model_dim * inter_dim
            tflops_ck = flops / (us_ck / 1e6) / 1e12
            print(
                f"[aiter CK] stage2: {us_ck:.1f} us, "
                f"{tflops_ck:.2f} TFLOPS(logical, M={tokens*topk}), flir vs aiter speedups: {tflops / tflops_ck:.2f}x"
            )

            # Correctness run (best-effort; do not fail perf comparison if CK diverges).
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
            if not verify_output(out.to(torch.float32), out_ck.to(torch.float32), rtol=0.5, atol=0.5, msg="[aiter CK] stage2:"):
                    logging.warning("[aiter CK] stage2 correctness mismatch vs FLIR (continuing; perf numbers still printed).")
        except Exception as e:
            logging.warning(f"Skipping aiter CK moe stage2 compare (not runnable here): {e}")

    # Print profile breakdown if the executor supports it
    if hasattr(exe, 'print_profile_stats'):
        exe.print_profile_stats()

    if return_outputs:
        return out, us
    return None


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
@pytest.mark.parametrize("in_dtype", ["fp8", "fp16", "int8", "int8smooth", "int4"])
@pytest.mark.parametrize("out_dtype", ["f16", "f32"], ids=["out_f16", "out_f32"])
@pytest.mark.parametrize("use_reduce", [False, True], ids=["atomic", "reduce"])
@pytest.mark.parametrize("use_valid_mask", [False, True], ids=["nomask", "mask"])
@pytest.mark.parametrize("test_graph", [
    pytest.param(False, id="graph"),
    pytest.param(True, id="eager", marks=pytest.mark.large_shape),
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
    """Single 2-stage test: gemm1 -> quantize -> gemm2, with routing built once."""
    if (not bool(use_reduce)) and bool(use_valid_mask):
        pytest.skip("valid_mask is only used in reduce mode (atomic mode ignores it).")
    out_s = str(out_dtype).strip().lower()
    if bool(use_reduce) and out_s in ("f32", "fp32", "float"):
        pytest.skip("reduce mode does not support out_dtype='f32' (compile_moe_gemm2(accumulate=False) forbids it).")
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
    # - Use torch routing (no CK dependency for sorting).
    # - Only compare CK for fp8, and only when explicitly requested.
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
    )

    if w_fp4_kernel:
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
    ):
        if use_flydsl_reduce:
            # Use unified implementation with FlyDSL reduce kernel
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
                # `compile_moe_gemm2_ex` uses `valid_mask is not None` as a compile-time sentinel
                # to enable masked reduction (different reduce kernel signature).
                valid_mask=(True if bool(use_valid_mask) else None),
                mode=MoeGemm2Mode.REDUCE,
            )
        else:
            # Use torch.sum for reduction (baseline comparison)
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
        stream_ptr,
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
            stream_ptr,
        )
        X = intermediate.view(tokens_in, self._topk, self._model_dim)
        if valid_mask is not None:
            X = X * valid_mask.view(tokens_in, self._topk, 1).to(dtype=X.dtype)
        torch.sum(X, dim=1, out=arg_out)

    @property
    def mode(self) -> str:
        return self._mode


# Reduce Kernel Performance Profiling
def profile_reduce_kernel(
    tokens: int,
    topk: int,
    model_dim: int,
    dtype: torch.dtype = torch.float16,
    num_iters: int = 20,
    num_warmup: int = 5,
    compare_torch: bool = True,
):
    """Profile reduce kernel bandwidth and latency.

    Args:
        tokens: Number of tokens
        topk: Top-k value
        model_dim: Model dimension
        dtype: Data type (torch.float16 or torch.bfloat16)
        num_iters: Number of benchmark iterations
        num_warmup: Number of warmup iterations
        compare_torch: If True, also benchmark torch.sum for comparison

    Returns:
        Dict with profiling results
    """
    import torch.profiler as tpf

    dtype_str = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32"}[dtype]
    reduce_exe = compile_moe_reduction(topk=topk, model_dim=model_dim, dtype_str=dtype_str)
    # Create test tensors
    X = torch.randn(tokens, topk, model_dim, device="cuda", dtype=dtype)
    Y = torch.empty(tokens, model_dim, device="cuda", dtype=dtype)
    # Calculate theoretical bandwidth
    elem_bytes = X.element_size()
    read_bytes = tokens * topk * model_dim * elem_bytes
    write_bytes = tokens * model_dim * elem_bytes
    total_bytes = read_bytes + write_bytes

    def _get_kernel_time_us(prof):
        """Extract CUDA kernel time from profiler (microseconds)."""
        total = 0.0
        for evt in prof.events():
            if str(getattr(evt, 'device_type', '')).endswith('CUDA'):
                total += getattr(evt, 'self_device_time_total', 0)
        return total

    results = {"shape": (tokens, topk, model_dim), "dtype": dtype_str}
    stream_ptr = torch.cuda.current_stream().cuda_stream
    valid_mask = torch.empty((0, topk), device="cuda", dtype=torch.uint8)

    # Benchmark FlyDSL reduce
    for _ in range(num_warmup):
        reduce_exe(X, Y, valid_mask, tokens, stream_ptr)
    torch.cuda.synchronize()

    with tpf.profile(activities=[tpf.ProfilerActivity.CUDA]) as prof:
        for _ in range(num_iters):
            reduce_exe(X, Y, valid_mask, tokens, stream_ptr)
        torch.cuda.synchronize()

    flydsl_us = _get_kernel_time_us(prof) / num_iters
    flydsl_bw = (total_bytes / 2**40) / (flydsl_us / 1e6)  # TB/s
    results["flydsl"] = {"latency_us": flydsl_us, "bandwidth_tb_s": flydsl_bw}

    # Benchmark torch.sum if requested
    if compare_torch:
        for _ in range(num_warmup):
            torch.sum(X, dim=1, out=Y)
        torch.cuda.synchronize()

        with tpf.profile(activities=[tpf.ProfilerActivity.CUDA]) as prof:
            for _ in range(num_iters):
                torch.sum(X, dim=1, out=Y)
            torch.cuda.synchronize()

        torch_us = _get_kernel_time_us(prof) / num_iters
        torch_bw = (total_bytes / 2**40) / (torch_us / 1e6)
        results["torch"] = {"latency_us": torch_us, "bandwidth_tb_s": torch_bw}
        results["speedup"] = torch_us / flydsl_us if flydsl_us > 0 else 0

    return results


def print_reduce_profile(results: dict):
    """Pretty print reduce profiling results."""
    tokens, topk, model_dim = results["shape"]
    print(f"\n[Reduce Kernel Profile] shape=({tokens}, {topk}, {model_dim}), dtype={results['dtype']}")
    print(f"  FlyDSL:  {results['flydsl']['latency_us']:.1f} us, {results['flydsl']['bandwidth_tb_s']:.2f} TB/s")
    if "torch" in results:
        print(f"  torch:   {results['torch']['latency_us']:.1f} us, {results['torch']['bandwidth_tb_s']:.2f} TB/s")
        print(f"  speedup: {results['speedup']:.2f}x")


@pytest.mark.parametrize(
    "tokens, topk, model_dim",
    [
        pytest.param(32769, 8, 7168, id="DS-TP8-prefill-L", marks=pytest.mark.large_shape),
        pytest.param(64, 8, 7168, id="DS-TP8-decode-S"),
        pytest.param(256, 8, 7168, id="DS-TP8-decode-L"),
        pytest.param(16384, 6, 5120, id="EP-K6-prefill", marks=pytest.mark.large_shape),
        pytest.param(64, 6, 5120, id="EP-K6-decode-S"),
        pytest.param(256, 6, 5120, id="EP-K6-decode-L"),
    ],
)
def test_moe_reduce_kernel(tokens: int, topk: int, model_dim: int):
    """Test reduce kernel correctness and performance vs torch.sum."""
    dtype = torch.float16
    dtype_str = "f16"

    reduce_exe = compile_moe_reduction(topk=topk, model_dim=model_dim, dtype_str=dtype_str)

    # Create test data
    X = torch.randn(tokens, topk, model_dim, device="cuda", dtype=dtype)
    Y_flydsl = torch.empty(tokens, model_dim, device="cuda", dtype=dtype)
    Y_ref = torch.empty(tokens, model_dim, device="cuda", dtype=dtype)

    # Run kernels
    stream_ptr = torch.cuda.current_stream().cuda_stream
    valid_mask = torch.empty((0, topk), device="cuda", dtype=torch.uint8)
    reduce_exe(X, Y_flydsl, valid_mask, tokens, stream_ptr)
    torch.sum(X, dim=1, out=Y_ref)
    torch.cuda.synchronize()

    # Correctness check using verify_output
    assert verify_output(Y_flydsl.float(), Y_ref.float(), rtol=1e-2, atol=1e-2, msg="[reduce kernel]")

    # Performance profiling
    results = profile_reduce_kernel(
        tokens=tokens, topk=topk, model_dim=model_dim,
        num_iters=20, num_warmup=5, compare_torch=True,
    )
    print_reduce_profile(results)



@pytest.mark.parametrize(
    "tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n, tile_k",
    [
        pytest.param(8192, 7168, 256, 128, 8, 64, 256, 128, id="DS-TP8-prefill-S", marks=pytest.mark.large_shape),
        pytest.param(1, 7168, 256, 256, 8, 16, 256, 128, id="DS-TP8-decode-bs1"),
        pytest.param(8, 7168, 256, 256, 8, 32, 256, 128, id="DS-TP8-decode-bs8"),
        pytest.param(1666, 5120, 1536, 64, 6, 64, 256, 128, id="EP-K6-prefill", marks=pytest.mark.large_shape),
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
        compile_fn=_make_reduce_mode_compile_fn(use_flydsl_reduce=True),
        kernel_name="moe_gemm2_reduce_flydsl",
    )

    # Run reduce mode and use valid mask with FlyDSL kernel
    run_moe_stage2(
        **common_args,
        compile_fn=_make_reduce_mode_compile_fn(use_flydsl_reduce=True, use_valid_mask=True),
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
        description="MoE 2-stage (FLIR MFMA FP8) test/benchmark (argparse subset aligned with aiter test_moe_2stage.py)",
    )
    parser.add_argument(
        "--in_dtype",
        type=str,
        default="fp8",
        choices=["fp8", "fp16", "int8", "int8smooth", "int4", "all"],
        help="Kernel input dtype: fp8 / fp16 / int8 / int8smooth / int4 / all (default: all). "
        "int8smooth expands X to [tokens*topk, K] with per-(token,slot) scales. "
        "int4 means W4A8: A int8, W packed int4.",
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

    args = parser.parse_args()

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

    def run_one(dt: str, use_reduce: bool):
        out_s = str(args.out_dtype).strip().lower()
        if bool(use_reduce) and out_s in ("f32", "fp32", "float"):
            print("[skip] reduce mode does not support out_dtype='f32'")
            return
        if (not bool(use_reduce)) and bool(args.use_valid_mask):
            print("[skip] valid_mask is only used in reduce mode (atomic ignores it)")
            return
        test_moe_gemm_2stage(
            tokens=int(args.tokenNum),
            model_dim=int(model_dim),
            inter_dim=int(inter_dim),
            experts=int(args.expert),
            topk=int(args.topk),
            tile_m=int(args.tile_m),
            tile_n1=int(args.tile_n),
            tile_k1=int(args.tile_k),
            tile_n2=tile_n2,
            tile_k2=tile_k2,
            doweight_stage1=bool(args.doweight_stage1),
            in_dtype=dt,
            out_dtype=str(args.out_dtype),
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

    # Run 2-stage (gemm1 -> quantize -> gemm2) aiter-style test/benchmark.
    for dt in args.in_dtype.split(","):
        for use_reduce in reduce_flags:
            run_one(dt, use_reduce)
