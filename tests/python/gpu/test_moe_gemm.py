#!/usr/bin/env python3
import logging
import os
import sys
from dataclasses import dataclass
from typing import Tuple, Optional

import pytest
import torch
import argparse

# -----------------------------------------------------------------------------
# Ensure we use the repo-local `pyflir` when running this file directly.
#
# Some environments have another `pyflir` (e.g. from a sibling checkout) earlier
# on `sys.path`, which can miss newer ROCDL wrappers (notably atomic fadd / MFMA).
# -----------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
_PYFLIR_SRC = os.path.join(_REPO_ROOT, "pyflir", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLIR_SRC not in sys.path:
    sys.path.insert(0, _PYFLIR_SRC)

from test_ref import torch_moe_gemm1, torch_moe_gemm2
from tests.utils import pertoken_quant, shuffle_weight
from tests.test_common import verify_output, run_perftest

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

# Kernel implementations live under `samples/`; this test file is the harness.
from samples.moe_gemm_2stage import compile_moe_gemm1, compile_moe_gemm2

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


def build_sorted_routing(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    model_dim: int,
    tile_m: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build CK-style routing buffers:
    - `sorted_token_ids[int32]` (fused token_idx[23:0] + topk_slot[31:24])
    - `sorted_weights[fp32]` aligned with sorted_token_ids
    - `sorted_expert_ids[int32]` per M-block

    Torch fallback routing builder (deterministic):
    - Sort by (expert, token, slot)
    - Pad each expert segment to `tile_m`

    NOTE: aiter routing is handled by `build_routing_buffers(..., moe_sort_mode="aiter")`.
    """
    assert topk_ids.is_cuda and topk_weights.is_cuda
    tokens, topk = topk_ids.shape
    # Build the same CK-style buffers using torch.
    topk_ids_i64 = topk_ids.to(torch.int64)
    topk_w_f32 = topk_weights.to(torch.float32)

    token_idx = torch.arange(tokens, device="cuda", dtype=torch.int64).unsqueeze(1).expand(tokens, topk)
    slot_idx = torch.arange(topk, device="cuda", dtype=torch.int64).unsqueeze(0).expand(tokens, topk)

    expert_flat = topk_ids_i64.reshape(-1)
    token_flat = token_idx.reshape(-1)
    slot_flat = slot_idx.reshape(-1)
    fused_flat = (token_flat & 0xFFFFFF) | ((slot_flat & 0xFF) << 24)
    weight_flat = topk_w_f32.reshape(-1)

    # Sort by (expert, token, slot) to make ordering stable/deterministic.
    linear = token_flat * topk + slot_flat
    key = expert_flat * (tokens * topk) + linear
    order = torch.argsort(key)

    expert_sorted = expert_flat[order]
    fused_sorted = fused_flat[order].to(torch.int32)
    w_sorted = weight_flat[order]

    counts = torch.bincount(expert_flat.clamp(min=0, max=num_experts - 1), minlength=num_experts).to("cpu").tolist()
    out_ids = []
    out_w = []
    out_expert_ids: list[int] = []
    off = 0
    sentinel = torch.tensor([tokens], device="cuda", dtype=torch.int32)
    sentinel_w = torch.tensor([0.0], device="cuda", dtype=torch.float32)
    for e in range(num_experts):
        cnt = int(counts[e])
        seg_ids = fused_sorted[off : off + cnt]
        seg_w = w_sorted[off : off + cnt]
        off += cnt

        # pad to tile_m
        pad = (-cnt) % tile_m
        if pad:
            seg_ids = torch.cat([seg_ids, sentinel.expand(pad)])
            seg_w = torch.cat([seg_w, sentinel_w.expand(pad)])

        out_ids.append(seg_ids)
        out_w.append(seg_w)
        for _ in range(int(seg_ids.numel() // tile_m)):
            out_expert_ids.append(e)

    sorted_token_ids = torch.cat(out_ids, dim=0)
    sorted_weights = torch.cat(out_w, dim=0)
    sorted_expert_ids = torch.tensor(out_expert_ids, device="cuda", dtype=torch.int32)
    return sorted_token_ids, sorted_weights, sorted_expert_ids


@pytest.mark.parametrize(
    "tokens,model_dim,inter_dim,experts,topk,doweight_stage1",
    [
        (256, 1024, 256, 4, 2, False),
        (256, 1024, 256, 4, 2, True),
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


def _pad_sorted_buffers_to_full_blocks(
    *,
    sorted_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    tokens: int,
    block_m: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad (sorted_ids, sorted_weights) to len(sorted_expert_ids)*block_m for safe OOB-free indexing."""
    assert sorted_ids.dtype == torch.int32
    assert sorted_weights.dtype == torch.float32
    blocks = int(sorted_expert_ids.numel())
    padded_len = blocks * block_m
    if int(sorted_ids.numel()) >= padded_len:
        return sorted_ids[:padded_len], sorted_weights[:padded_len]
    pad_ids = torch.empty((padded_len,), device="cuda", dtype=torch.int32)
    pad_w = torch.empty((padded_len,), device="cuda", dtype=torch.float32)
    pad_ids.fill_(tokens)  # sentinel fused token (token=tokens, slot=0)
    pad_w.zero_()
    pad_ids[: sorted_ids.numel()] = sorted_ids
    pad_w[: sorted_weights.numel()] = sorted_weights
    return pad_ids, pad_w


@dataclass(frozen=True)
class RoutingBuffers:
    sorted_token_ids: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    sorted_size: int


def build_routing_buffers(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    experts: int,
    model_dim: int,
    tile_m: int,
    moe_sort_mode: Optional[str] = None,
) -> RoutingBuffers:
    """Build/pad routing buffers once (CK format), reusable across stage1 + stage2."""
    device = topk_ids.device
    sort_mode = (
        (moe_sort_mode or os.environ.get("pyflir_MOE_SORT_MODE", "aiter" if HAS_AITER else "torch"))
        .lower()
        .strip()
    )

    sorted_token_ids = sorted_weights = sorted_expert_ids = num_valid_ids = None
    if sort_mode == "aiter":
        res = _maybe_aiter_moe_sorting(
            topk_ids,
            topk_weights,
            num_experts=experts,
            model_dim=model_dim,
            block_m=tile_m,
        )
        if res is not None:
            sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids = res
            # Some aiter builds return `sorted_expert_ids` with extra garbage entries.
            # Trim based on `num_valid_ids` to keep expert ids in-range and consistent with sorted ids.
            valid = int(num_valid_ids.view(-1)[0].item())
            blocks = (valid + int(tile_m) - 1) // int(tile_m)
            sorted_token_ids = sorted_token_ids[:valid].contiguous()
            sorted_weights = sorted_weights[:valid].contiguous()
            sorted_expert_ids = sorted_expert_ids[:blocks].contiguous()
        else:
            logging.warning(
                "aiter moe_sorting unavailable; falling back to torch routing buffers. "
                "Set pyflir_MOE_SORT_MODE=torch to silence this, or ensure aiter JIT deps are available."
            )
            sort_mode = "torch"

    if sort_mode != "aiter":
        sorted_token_ids, sorted_weights, sorted_expert_ids = build_sorted_routing(
            topk_ids,
            topk_weights,
            num_experts=experts,
            model_dim=model_dim,
            tile_m=tile_m,
        )
        num_valid_ids = torch.tensor([int(sorted_token_ids.numel())], device=device, dtype=torch.int32)

    sorted_token_ids, sorted_weights = _pad_sorted_buffers_to_full_blocks(
        sorted_ids=sorted_token_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        tokens=int(topk_ids.shape[0]),
        block_m=tile_m,
    )
    sorted_size = int(sorted_token_ids.numel())
    return RoutingBuffers(
        sorted_token_ids=sorted_token_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        sorted_size=sorted_size,
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
    sorted_token_ids = routing.sorted_token_ids
    sorted_weights = routing.sorted_weights
    sorted_expert_ids = routing.sorted_expert_ids
    num_valid_ids = routing.num_valid_ids
    sorted_size = routing.sorted_size

    if in_dtype not in ("fp8", "int8", "int4"):
        raise ValueError(f"in_dtype must be 'fp8', 'int8', or 'int4', got {in_dtype!r}")
    is_int4 = in_dtype == "int4"
    is_int8 = in_dtype in ("int8", "int4")

    # Quantize inputs / weights.
    if in_dtype == "fp8":
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.float8_e4m3fnuz)  # [tokens,K], [tokens,1]
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.float8_e4m3fnuz)  # [E,2*inter,K], [E,2*inter,1]
    # w2 is not used by our kernel, but required by CK stage1 API
        w2_q, _scale_w2_unused = pertoken_quant(w2_fp32, quant_dtype=torch.float8_e4m3fnuz)
    elif in_dtype == "int8":
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.int8)
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
    scale_w1_flat = scale_w1.view(experts * (2 * inter_dim), 1)

    # Pad storage for forced global dwordx4 loads (same trick as existing GEMM tests)
    PAD_ELEMS = 256
    x_flat = x_q.contiguous().view(-1)
    x_storage = torch.empty(x_flat.numel() + PAD_ELEMS, device=device, dtype=x_q.dtype)
    x_storage[: x_flat.numel()] = x_flat
    x_q = x_storage[: x_flat.numel()].view(tokens, model_dim)

    # Weight storage:
    # - fp8/int8: preshuffled bytes (1B/elem)
    # - int4: packed int4 bytes (2 values per byte)
    w_flat = (
        _pack_shuffled_int8_to_packed_int4_no_perm(w1_shuffled_flat) if is_int4 else w1_shuffled_flat
    ).contiguous().view(-1)
    w_storage = torch.empty(w_flat.numel() + PAD_ELEMS, device=device, dtype=w_flat.dtype)
    w_storage[: w_flat.numel()] = w_flat
    w_kernel = w_storage[: w_flat.numel()]
    w_kernel = (
        w_kernel.view(experts * (2 * inter_dim), model_dim) if (not is_int4) else w_kernel
    )

    # Flatten scales to 1D memrefs
    scale_x_1d = scale_x.view(-1).contiguous()  # [tokens]
    scale_w1_1d = scale_w1_flat.view(-1).contiguous()  # [rows]
    sorted_weights_1d = sorted_weights.contiguous().view(-1)  # [sorted_size]

    # Output: [tokens, topk, inter_dim] fp16
    out = torch.empty((tokens, topk, inter_dim), device=device, dtype=torch.float16)

    exe = compile_moe_gemm1(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        in_dtype=in_dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        sorted_size=sorted_size,
        size_expert_ids=int(sorted_expert_ids.numel()),
        doweight_stage1=bool(doweight_stage1),
    )

    def launch(o, x, w, sx, sw, st, eids, sw_sorted):
        exe(o, x, w, sx, sw, st, eids, sw_sorted, tokens, inter_dim, model_dim)

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
    )
    torch.cuda.synchronize()

    ref = torch_moe_gemm1(
        x_q,
        w1_q_flat,
        scale_x,
        scale_w1_flat,
        topk_ids.to(torch.int64),
        topk_weights,
        inter_dim=inter_dim,
        doweight_stage1=doweight_stage1,
    )

    rtol = 0.5 if is_int4 else 0.25
    atol = 0.5 if is_int4 else 0.25
    assert verify_output(out.to(torch.float32), ref, rtol=rtol, atol=atol)

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
                    splitk=1,
                    dst_type=torch.float16,
                )

            # Benchmark CK stage1
            _, us_ck = run_perftest(
                launch_ck,
                out_ck,
                x_q,
                w1_ck,
                w2_ck,
                sorted_token_ids,
                sorted_expert_ids,
                num_valid_ids,
                w1_scale_ck,
                scale_x,
                sorted_weights,
                num_iters=int(num_iters),
                num_warmup=int(num_warmup),
            )

            # Correctness: flir vs CK
            assert verify_output(out.to(torch.float32), out_ck.to(torch.float32), rtol=0.25, atol=0.25, msg="flir vs aiter:")

            # Perf print: use the same flop model for both
            flops = 2 * tokens * topk * (2 * inter_dim) * model_dim
            tflops_ck = flops / (us_ck / 1e6) / 1e12
            print(f"[aiter CK] stage1: {us_ck:.1f} us, {tflops_ck:.2f} TFLOPS")
        except Exception as e:
            # Treat CK compare as best-effort: many environments can import `aiter` but can't load
            # the full JIT .so dependency chain. Don't fail the FLIR test suite for that.
            logging.warning(f"Skipping aiter CK moe stage1 compare (not runnable here): {e}")
    # Note: kernel executes `sorted_size` rows (padded to full tile_m blocks per-expert),
    # which can be > tokens*topk. Report both:
    # - logical TFLOPS: based on tokens*topk (algorithmic work)
    # - executed TFLOPS: based on sorted_size (actual kernel work including padding)
    flops_logical = 2 * tokens * topk * (2 * inter_dim) * model_dim
    tflops_logical = flops_logical / (us / 1e6) / 1e12

    # Rough bytes-moved accounting (same spirit as GEMM tests: count each tensor once).
    bytes_moved = 0
    bytes_moved += tokens * model_dim * 1  # x fp8
    bytes_moved += (experts * (2 * inter_dim) * model_dim) // (2 if is_int4 else 1)  # w (packed for int4)
    # Output rows are logically tokens*topk, but kernel may touch padded rows due to routing blocks.
    bytes_moved += int(sorted_size) * inter_dim * 2  # out fp16 (upper bound for writes)
    bytes_moved += tokens * 4  # scale_x f32 (1D)
    bytes_moved += experts * (2 * inter_dim) * 4  # scale_w f32 (1D)
    bytes_moved += int(sorted_weights.numel()) * 4  # sorted_weights f32
    bytes_moved += int(sorted_token_ids.numel()) * 4  # sorted_token_ids i32
    bytes_moved += int(sorted_expert_ids.numel()) * 4  # sorted_expert_ids i32
    tbps = bytes_moved / 1e12 / (us / 1e6)

    print(
        f"FLIR MoE stage1[{in_dtype}]: "
        f"{us:.1f} us, "
        f"{tflops_logical:.2f} TFLOPS(logical, M={tokens*topk}), "
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
):
    """MoE stage2 (gemm2): out2[t] = sum_{slot} ( out1[t,slot] @ W2[expert]^T ) with optional routed weight."""
    assert model_dim % tile_n == 0
    assert inter_dim % tile_k == 0

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
    w2_fp32 = (
        w2_fp32_in
        if w2_fp32_in is not None
        else torch.randn((experts, model_dim, inter_dim), device=device, dtype=torch.float32)
    )

    # Routing: deterministic torch topk + softmax.
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
    sorted_token_ids = routing.sorted_token_ids
    sorted_weights = routing.sorted_weights
    sorted_expert_ids = routing.sorted_expert_ids
    num_valid_ids = routing.num_valid_ids
    sorted_size = routing.sorted_size

    if in_dtype not in ("fp8", "int8", "int4"):
        raise ValueError(f"in_dtype must be 'fp8', 'int8', or 'int4', got {in_dtype!r}")
    is_int4 = in_dtype == "int4"
    is_int8 = in_dtype in ("int8", "int4")

    # Quantize inputs / weights.
    if in_dtype == "fp8":
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.float8_e4m3fnuz)
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.float8_e4m3fnuz)
        w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=torch.float8_e4m3fnuz)
    elif in_dtype == "int8":
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.int8)
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.int8)
        w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=torch.int8)
    else:
        # W4A8: A2 is int8, W2 is int4 packed (host packs from int8 values in [-8,7]).
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.int8)
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.int8, dtypeMax=7)
        w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=torch.int8, dtypeMax=7)

    # Preshuffle weights (aiter/CK layout) on the *unpacked* tensor.
    w2_shuffled = shuffle_weight(w2_q)

    # Stage2 input (A2): either provided (gemm1->quantize chaining) or built from stage1 reference.
    if a2_fp8_in is not None and a2_scale_in is not None:
        a2_q = a2_fp8_in
        a2_scale = a2_scale_in
    else:
        w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim)
        scale_w1_flat = scale_w1.view(experts * (2 * inter_dim), 1)
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
            a2_q, a2_scale = pertoken_quant(out1_ref, quant_dtype=torch.float8_e4m3fnuz)
        else:
            a2_q, a2_scale = pertoken_quant(out1_ref, quant_dtype=torch.int8)

    # Flatten weights/scales for the kernel.
    w2_shuffled_flat = w2_shuffled.view(experts * model_dim, inter_dim)
    scale_w2_flat = scale_w2.view(experts * model_dim, 1)

    # For W4A8, pack preshuffled int8 weights into packed int4 bytes.
    w2_kernel = w2_shuffled_flat
    if is_int4:
        w2_kernel = _pack_shuffled_int8_to_packed_int4_no_perm(w2_shuffled_flat)

    # Pad storage for forced global dwordx4 loads (same trick as existing GEMM tests).
    PAD_ELEMS = 256
    a2_flat = a2_q.contiguous().view(-1)
    a2_storage = torch.empty(a2_flat.numel() + PAD_ELEMS, device=device, dtype=a2_q.dtype)
    a2_storage[: a2_flat.numel()] = a2_flat
    a2_q = a2_storage[: a2_flat.numel()].view(tokens, topk, inter_dim)

    w2_flat = w2_kernel.contiguous().view(-1)
    w2_storage = torch.empty(w2_flat.numel() + PAD_ELEMS, device=device, dtype=w2_flat.dtype)
    w2_storage[: w2_flat.numel()] = w2_flat
    w2_kernel = w2_storage[: w2_flat.numel()]
    if not is_int4:
        w2_kernel = w2_kernel.view(experts * model_dim, inter_dim)

    # Flatten scales to 1D memrefs.
    a2_scale_1d = a2_scale.view(-1).contiguous()  # [tokens*topk]
    w2_scale_1d = scale_w2_flat.view(-1).contiguous()  # [experts*model_dim]
    sorted_weights_1d = sorted_weights.contiguous().view(-1)  # [sorted_size]

    # Output: [tokens, model_dim] fp32 (atomic add).
    out = torch.zeros((tokens, model_dim), device=device, dtype=torch.float32)
    out_perf = torch.zeros_like(out)

    doweight_stage2 = not bool(doweight_stage1)
    exe = compile_moe_gemm2(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        in_dtype=in_dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        sorted_size=sorted_size,
        size_expert_ids=int(sorted_expert_ids.numel()),
        doweight_stage2=bool(doweight_stage2),
    )

    def launch(o, x, w, sx, sw, st, eids, sw_sorted):
        exe(o, x, w, sx, sw, st, eids, sw_sorted, tokens, model_dim, inter_dim)

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
    assert verify_output(out, ref2, rtol=0.5, atol=0.5)

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
                shuffle_weight(w1_q),  # stage2 signature includes w1; provide preshuffled tensor
                w2_shuffled,
                sorted_token_ids,
                sorted_expert_ids,
                num_valid_ids,
                scale_w2.contiguous(),
                a2_scale.contiguous(),
                sorted_weights,
                num_iters=int(num_iters),
                num_warmup=int(num_warmup),
            )

            # Perf print (report both executed vs logical FLOPs, same convention as FLIR).
            flops_logical = 2 * tokens * topk * model_dim * inter_dim
            tflops_ck_logical = flops_logical / (us_ck / 1e6) / 1e12
            print(
                f"[aiter CK] stage2: {us_ck:.1f} us, "
                f"{tflops_ck_logical:.2f} TFLOPS(logical, M={tokens*topk})"
            )

            # Correctness run (best-effort; do not fail perf comparison if CK diverges).
            out_ck.zero_()
            launch_ck(
                out_ck,
                a2_q,
                shuffle_weight(w1_q),
                w2_shuffled,
                sorted_token_ids,
                sorted_expert_ids,
                num_valid_ids,
                scale_w2.contiguous(),
                a2_scale.contiguous(),
                sorted_weights,
            )
            torch.cuda.synchronize()
            if not verify_output(out, out_ck.to(torch.float32), rtol=0.5, atol=0.5, msg="[aiter CK] stage2:"):
                    logging.warning("[aiter CK] stage2 correctness mismatch vs FLIR (continuing; perf numbers still printed).")
        except Exception as e:
            logging.warning(f"Skipping aiter CK moe stage2 compare (not runnable here): {e}")

    # Same note as stage1: executed rows = sorted_size (padded), logical rows = tokens*topk.
    flops_logical = 2 * tokens * topk * model_dim * inter_dim
    tflops_logical = flops_logical / (us / 1e6) / 1e12

    bytes_moved = 0
    bytes_moved += int(sorted_size) * inter_dim * 1  # a2 fp8 (upper bound: padded rows)
    bytes_moved += (experts * model_dim * inter_dim) // (2 if is_int4 else 1)  # w2 (packed for int4)
    bytes_moved += tokens * model_dim * 4  # out fp32
    bytes_moved += int(sorted_size) * 4  # a2_scale f32 (1D, padded upper bound)
    bytes_moved += experts * model_dim * 4  # w2_scale f32 (1D)
    bytes_moved += int(sorted_weights.numel()) * 4
    bytes_moved += int(sorted_token_ids.numel()) * 4
    bytes_moved += int(sorted_expert_ids.numel()) * 4
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(
        f"FLIR MoE stage2[{in_dtype}]: "
        f"{us:.1f} us, "
        f"{tflops_logical:.2f} TFLOPS(logical, M={tokens*topk}), "
        f"{tbps:.3f} TB/s (doweight_stage2={doweight_stage2})"
    )
    if return_outputs:
        return out, us
    return None


@pytest.mark.parametrize(
    "tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n1, tile_k1, tile_n2, tile_k2, doweight_stage1",
    [
        (256, 4096, 2048, 17, 9, 64, 128, 128, 256, 128, False),
    ],
)
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
    *,
    in_dtype: str = "fp8",
    seed: int = 0,
    num_iters: int = 5,
    num_warmup: int = 2,
    moe_sort_mode: Optional[str] = None,
    compare_aiter_ck: Optional[bool] = None,
    init_scale: float = 1.0,
):
    """Single 2-stage test: gemm1 -> quantize -> gemm2, with routing built once."""
    device = torch.device("cuda")
    torch.manual_seed(int(seed))

    # NOTE: With naive N(0,1) weights, stage1 output variance grows with K and can easily overflow fp16
    # outputs in CK paths for large (model_dim, inter_dim). Use a fan-in style init by default to keep
    # activations O(1) and avoid inf/nan (similar motivation to typical Transformer init).
    #
    # `init_scale` is an extra global multiplier (keep at 1.0 unless you intentionally want larger/smaller
    # activations).
    import math

    s = float(init_scale)
    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32) * s
    # fan_in = model_dim for W1: [E, 2*inter, model]
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * (s / math.sqrt(model_dim))
    # fan_in = inter_dim for W2: [E, model, inter]
    w2_fp32 = torch.randn((experts, model_dim, inter_dim), device=device, dtype=torch.float32) * (s / math.sqrt(inter_dim))

    score = torch.randn((tokens, experts), device=device, dtype=torch.float32)
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
        compare_aiter_ck=compare_aiter_ck,
        moe_sort_mode=moe_sort_mode,
        x_fp32_in=x_fp32,
        w1_fp32_in=w1_fp32,
        w2_fp32_in=w2_fp32,
        topk_ids_in=topk_ids,
        topk_weights_in=topk_weights,
        routing_in=routing,
        return_outputs=True,
    )

    out1_fp32 = out1_fp16.to(torch.float32)
    if in_dtype == "fp8":
        a2_q, a2_scale = pertoken_quant(out1_fp32, quant_dtype=torch.float8_e4m3fnuz)
    else:
        a2_q, a2_scale = pertoken_quant(out1_fp32, quant_dtype=torch.int8)

    _out2_fp32, _us2 = run_moe_stage2(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        in_dtype=in_dtype,
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
        default="all",
        choices=["fp8", "int8", "int4", "all"],
        help="Kernel input dtype: fp8 / int8 / int4 / all (default: all). "
        "int4 means W4A8: A int8, W packed int4.",
    )
    parser.add_argument("-d", "--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Input init dtype (currently data is quantized to FP8 per-token; init dtype mainly affects RNG range).")
    parser.add_argument("-dim", type=_str2tuple_dim, default=(8192, 5120), help="Model dimension: model_dim,inter_dim (e.g. -dim 6144,4096)")
    parser.add_argument("-t", "--tokenNum", type=int, default=2048, help="Number of tokens (e.g. -t 1024)")
    parser.add_argument("-e", "--expert", type=int, default=17, help="Number of experts (e.g. -e 8)")
    parser.add_argument("-k", "--topk", type=int, default=9, help="Top-k (e.g. -k 2)")
    parser.add_argument("-s", "--doweight_stage1", type=_str2bool, nargs="?", const=True, default=False, help="Whether to multiply routed weight in stage1 (t/f).")

    # Stage1-specific kernel tiling knobs
    parser.add_argument("--tile_m", type=int, default=64, help="Tile M / block_m (routing block size).")
    parser.add_argument("--tile_n", type=int, default=128, help="Tile N (inter dim tile).")
    parser.add_argument("--tile_k", type=int, default=128, help="Tile K (model dim tile).")
    parser.add_argument("--tile_n2", type=int, default=None, help="Stage2 tile N (model dim tile). Default: 2*tile_n.")
    parser.add_argument("--tile_k2", type=int, default=None, help="Stage2 tile K (inter dim tile). Default: tile_k.")

    # Sorting / comparison knobs
    parser.add_argument("--moe_sort_mode", type=str, default=None, choices=["aiter", "torch"], help="Routing buffer build mode (aiter moe_sorting vs torch fallback).")
    parser.add_argument("--compare_aiter_ck", type=_str2bool, nargs="?", const=True, default=None, help="Override COMPARE_AITER_CK (t/f). Default: env or HAS_AITER.")

    # Benchmark knobs
    parser.add_argument("--seed", type=int, default=0, help="torch.manual_seed(seed)")
    parser.add_argument("--num_iters", type=int, default=5, help="Benchmark iters")
    parser.add_argument("--num_warmup", type=int, default=2, help="Benchmark warmup iters")
    parser.add_argument("--init_scale", type=float, default=1.0, help="Extra scale applied on top of fan-in init for random x/w (use >1.0 only if you want to stress overflow).")

    args = parser.parse_args()

    model_dim, inter_dim = args.dim

    tile_n2 = int(args.tile_n2) if args.tile_n2 is not None else int(args.tile_n) * 2
    tile_k2 = int(args.tile_k2) if args.tile_k2 is not None else int(args.tile_k)

    # Run 2-stage (gemm1 -> quantize -> gemm2) aiter-style test/benchmark.
    dtypes_to_run = ["fp8", "int8", "int4"] if args.in_dtype == "all" else [str(args.in_dtype)]
    for dt in dtypes_to_run:
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
            seed=int(args.seed),
            num_iters=int(args.num_iters),
            num_warmup=int(args.num_warmup),
            moe_sort_mode=args.moe_sort_mode,
            compare_aiter_ck=args.compare_aiter_ck,
            init_scale=float(args.init_scale),
        )



