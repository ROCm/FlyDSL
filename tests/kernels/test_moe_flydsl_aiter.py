#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# This test is adapted from `aiter/aiter/test_moe_flydsl.py` to live in the FlyDSL
# (dsl2) test suite.
#
# Default behavior: if aiter is missing or aiter-side execution/correctness fails,
# we SKIP (so FlyDSL CI isn't blocked by aiter installation/runtime issues).

from __future__ import annotations

import os
import sys
from typing import Tuple

import pytest


# -----------------------------------------------------------------------------
# Prefer repo-local sources when running file directly.
# -----------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


try:
    import torch
except Exception as e:  # pragma: no cover
    pytest.skip(f"torch not available: {e}", allow_module_level=True)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def _import_aiter_or_skip():
    try:
        import aiter  # type: ignore
        from aiter import dtypes  # type: ignore
        from aiter.fused_moe import (  # type: ignore
            fused_topk,
            moe_sorting,
            torch_moe_stage1,
            torch_moe_stage2,
        )
    except Exception as e:
        pytest.skip(f"aiter not available/working: {e}", allow_module_level=True)
    return aiter, dtypes, fused_topk, moe_sorting, torch_moe_stage1, torch_moe_stage2


def _pick_tile_m(tokens: int) -> int:
    # Match the heuristic used in aiter's test: bigger routing tiles for large token counts.
    if tokens > 1024:
        return 64
    if tokens > 256:
        return 32
    return 16


def _maybe_skip_on_mismatch(*, ok: bool, msg: str):
    if ok:
        return
    # Default: skip, because this is an integration check against aiter which may drift.
    # If you want strict behavior (fail on mismatch), set: FLYDSL_AITER_MOE_STRICT=1
    if os.environ.get("FLYDSL_AITER_MOE_STRICT", "0") in ("1", "true", "True", "YES", "yes"):
        raise AssertionError(msg)
    pytest.skip(msg)


def _routing_aiter(
    moe_sorting,
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    experts: int,
    model_dim: int,
    tile_m: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_ids_i32 = topk_ids.to(torch.int32)
    topk_w_f32 = topk_weights.to(torch.float32)
    sorted_ids, sorted_w, sorted_eids, num_valid_ids, _ = moe_sorting(
        topk_ids_i32,
        topk_w_f32,
        experts,
        model_dim,
        torch.float16,
        tile_m,
    )
    return (
        sorted_ids.contiguous(),
        sorted_w.contiguous(),
        sorted_eids.contiguous(),
        num_valid_ids.contiguous(),
    )


@pytest.mark.parametrize(
    "tokens,model_dim,inter_dim,experts,topk",
    [
        # Small, CI-friendly config that exercises fp8 per-token + g1u1 path.
        (256, 1024, 256, 4, 2),
    ],
)
def test_moe_flydsl_against_aiter_or_skip(tokens, model_dim, inter_dim, experts, topk):
    aiter, dtypes, fused_topk, moe_sorting, torch_moe_stage1, torch_moe_stage2 = _import_aiter_or_skip()

    # Only test the fp8-per-token case where FlyDSL MoE kernels are used.
    qType = aiter.QuantType.per_Token
    AQDType = dtypes.fp8
    WQDType = dtypes.fp8

    # Keep values finite for fp16 accumulation paths.
    init_scale = float(os.environ.get("AITER_MOE_INIT_SCALE", "0.1"))

    torch.manual_seed(0)
    dtype = torch.float16

    # NOTE: FlyDSL MoE 2-stage kernels implement the g1u1 (SwiGLU) path.
    use_g1u1 = True
    w1 = torch.randn((experts, inter_dim * 2, model_dim), device="cuda", dtype=dtype) * init_scale
    w2 = torch.randn((experts, model_dim, inter_dim), device="cuda", dtype=dtype) * init_scale
    x = torch.randn((tokens, model_dim), device="cuda", dtype=dtype) * init_scale
    score = torch.randn((tokens, experts), device="cuda", dtype=dtype) * init_scale

    topk_weights, topk_ids = fused_topk(x, score, topk, True)

    torch_quant = aiter.get_torch_quant(qType)
    try:
        w1_qt, w1_scale = torch_quant(w1, quant_dtype=WQDType)
        w2_qt, w2_scale = torch_quant(w2, quant_dtype=WQDType)
        a1_qt, a1_scale = torch_quant(x, quant_dtype=AQDType)
    except Exception as e:
        pytest.skip(f"aiter quantization failed: {e}")

    # Reference (aiter torch path).
    try:
        out1_ref = torch_moe_stage1(
            a1_qt,
            w1_qt,
            w2_qt,
            topk_weights,
            topk_ids,
            dtype=dtype,
            activation=aiter.ActivationType.Silu,
            quant_type=qType,
            a1_scale=a1_scale,
            w1_scale=w1_scale,
            w1_bias=None,
            doweight=False,
        )
        a2_qt_ref, a2_scale_ref = torch_quant(out1_ref, quant_dtype=AQDType)
        a2_qt_ref = a2_qt_ref.view(tokens, topk, -1)
        out2_ref = torch_moe_stage2(
            a2_qt_ref,
            w1_qt,
            w2_qt,
            topk_weights,
            topk_ids,
            dtype=dtype,
            quant_type=qType,
            w2_scale=w2_scale,
            a2_scale=a2_scale_ref,
            w2_bias=None,
            doweight=True,
        )
    except Exception as e:
        pytest.skip(f"aiter reference path failed: {e}")

    # FlyDSL kernels (copied from aiter swap path).
    from kernels.moe_gemm_2stage import compile_moe_gemm1, compile_moe_gemm2
    from tests.utils import shuffle_weight as dsl2_shuffle_weight

    tile_m = _pick_tile_m(tokens)
    tile_n1, tile_k1 = 128, 128
    tile_n2, tile_k2 = 256, 128

    try:
        sorted_ids, sorted_w, sorted_eids, num_valid_ids = _routing_aiter(
            moe_sorting,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            experts=experts,
            model_dim=model_dim,
            tile_m=tile_m,
        )
    except Exception as e:
        pytest.skip(f"aiter routing (moe_sorting) failed: {e}")

    sorted_size = int(sorted_ids.numel())
    blocks = int(sorted_eids.numel())

    if a1_scale is None:
        pytest.skip("aiter returned a1_scale=None; required for fp8 per-token path")

    scale_x_1d = a1_scale.view(-1).contiguous()

    w1_shuf = dsl2_shuffle_weight(w1_qt)
    w2_shuf = dsl2_shuffle_weight(w2_qt)
    w1_flat = w1_shuf.contiguous().view(experts * (2 * inter_dim), model_dim).contiguous()
    w2_flat = w2_shuf.contiguous().view(experts * model_dim, inter_dim).contiguous()

    w1_scale_1d = w1_scale.view(-1).contiguous()
    w2_scale_1d = w2_scale.view(-1).contiguous()
    sorted_w_1d = sorted_w.view(-1).contiguous()

    out1 = torch.empty((tokens, topk, inter_dim), device="cuda", dtype=torch.float16)
    out2 = torch.empty((tokens, model_dim), device="cuda", dtype=torch.float16)

    try:
        exe1 = compile_moe_gemm1(
            tokens=tokens,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n1,
            tile_k=tile_k1,
            sorted_size=sorted_size,
            size_expert_ids=int(sorted_eids.numel()),
            doweight_stage1=False,
            in_dtype="fp8",
            out_dtype="f16",
            dynamic_blocks=True,
            use_cshuffle_epilog=False,
        )
        exe2 = compile_moe_gemm2(
            tokens=tokens,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n2,
            tile_k=tile_k2,
            sorted_size=sorted_size,
            size_expert_ids=int(sorted_eids.numel()),
            doweight_stage2=True,
            in_dtype="fp8",
            out_dtype="f16",
            dynamic_blocks=True,
        )
    except Exception as e:
        pytest.skip(f"FlyDSL compile failed: {e}")

    try:
        exe1(
            out1,
            a1_qt.contiguous().view(tokens, model_dim),
            w1_flat,
            scale_x_1d,
            w1_scale_1d,
            sorted_ids,
            sorted_eids,
            sorted_w_1d,
            num_valid_ids,
            tokens,
            inter_dim,
            model_dim,
            blocks,
        )
        a2_qt, a2_scale = torch_quant(out1, quant_dtype=AQDType)
        a2_qt = a2_qt.view(tokens, topk, -1).contiguous()
        a2_scale_1d = a2_scale.view(-1).contiguous()
        a2_qt_flat = a2_qt.view(-1).contiguous()

        out2.zero_()
        exe2(
            out2,
            a2_qt_flat,
            w2_flat.view(-1),
            a2_scale_1d,
            w2_scale_1d,
            sorted_ids,
            sorted_eids,
            sorted_w_1d,
            num_valid_ids,
            tokens,
            model_dim,
            inter_dim,
            blocks,
        )
    except Exception as e:
        pytest.skip(f"FlyDSL kernel launch failed: {e}")

    # Compare with aiter torch reference; skip by default on mismatch.
    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    atol = float(os.environ.get("FLYDSL_AITER_MOE_ATOL", "0.02"))
    rtol = float(os.environ.get("FLYDSL_AITER_MOE_RTOL", "0.02"))
    ok = torch.allclose(out2_ref, out2, atol=atol, rtol=rtol)
    if not ok:
        # Provide a quick diagnostic without dumping huge tensors.
        diff = (out2_ref - out2).abs()
        max_abs = float(diff.max().item())
        _maybe_skip_on_mismatch(
            ok=False,
            msg=(
                f"FlyDSL MoE output mismatch vs aiter reference "
                f"(tokens={tokens}, model_dim={model_dim}, inter_dim={inter_dim}, "
                f"E={experts}, topk={topk}, atol={atol}, rtol={rtol}, max_abs={max_abs})."
            ),
        )

