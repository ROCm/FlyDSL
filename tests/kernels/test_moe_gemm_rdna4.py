#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE GEMM tests for RDNA4 (gfx120x) WMMA fp16/bf16 kernels."""

from __future__ import annotations

import math
import os
import sys
from typing import Optional

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYTHON_CANDIDATES = [
    os.path.join(_REPO_ROOT, "build", "python_packages"),
    _REPO_ROOT,
]
for _p in reversed(_PYTHON_CANDIDATES):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from tests.kernels.test_ref import torch_moe_gemm1, torch_moe_gemm2
from tests.kernels.moe_test_utils import (
    RoutingBuffers,
    build_routing_buffers,
    get_topk_valid_mask,
)
from tests.test_common import verify_output, run_perftest
from flydsl.runtime.device import get_rocm_arch
from kernels.moe_gemm_2stage import MoeGemm2Mode
from kernels.rdna_moe_gemm_2stage import (
    compile_moe_gemm1,
    compile_moe_gemm2,
    compile_moe_gemm2_ex,
)

ARCH = str(get_rocm_arch())

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

if not ARCH.startswith("gfx120"):
    pytest.skip(f"RDNA4 MoE tests require gfx120x, got {ARCH}", allow_module_level=True)


def _make_reduce_mode_compile_fn(use_valid_mask: bool = False):
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
    ):
        _ = group_size
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
            zero_intermediate=True,
            expert_sched_mode=bool(expert_sched_mode),
        )

    return _compile


def _make_inputs(
    *,
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    even_dispatch: bool = False,
    seed: int = 0,
    init_scale: float = 0.2,
):
    device = torch.device("cuda")
    torch.manual_seed(int(seed))

    s = float(init_scale)
    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32) * s
    w1_fp32 = torch.randn(
        (experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32
    ) * s
    w2_fp32 = torch.randn(
        (experts, model_dim, inter_dim), device=device, dtype=torch.float32
    ) * (s / math.sqrt(inter_dim))

    if even_dispatch:
        topk_ids = torch.stack(
            [
                torch.arange(topk, device=device, dtype=torch.int32) + ((t * topk) % experts)
                for t in range(tokens)
            ]
        ) % experts
        topk_weights = torch.full(
            (tokens, topk), 1.0 / topk, device=device, dtype=torch.float32
        )
    else:
        score = torch.rand((tokens, experts), device=device, dtype=torch.float32)
        topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
        topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)

    return x_fp32, w1_fp32, w2_fp32, topk_ids, topk_weights


def _make_partial_valid_mask(topk_ids: torch.Tensor, experts: int) -> torch.Tensor:
    expert_mask = torch.ones((experts,), device=topk_ids.device, dtype=torch.uint8)
    expert_mask[1::2] = 0
    valid_mask = get_topk_valid_mask(
        topk_ids,
        expert_mask=expert_mask,
        dtype=torch.uint8,
    ).contiguous()
    num_valid = int(valid_mask.sum().item())
    if num_valid <= 0 or num_valid >= valid_mask.numel():
        raise ValueError("expected partial valid_mask with both zero and non-zero entries")
    return valid_mask


def run_moe_stage1(
    *,
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    in_dtype: str = "fp16",
    seed: int = 0,
    num_iters: int = 3,
    num_warmup: int = 1,
    test_graph: bool = False,
    waves_per_eu: Optional[int] = None,
    expert_sched_mode: bool = True,
    even_dispatch: bool = False,
    x_fp32_in: Optional[torch.Tensor] = None,
    w1_fp32_in: Optional[torch.Tensor] = None,
    w2_fp32_in: Optional[torch.Tensor] = None,
    topk_ids_in: Optional[torch.Tensor] = None,
    topk_weights_in: Optional[torch.Tensor] = None,
    routing_in: Optional[RoutingBuffers] = None,
    return_outputs: bool = False,
    skip_ref: bool = False,
    # Accepted for moe_bench_main compatibility; RDNA4 has no TDM hardware.
    use_tdm_store: bool = False,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
):
    _ = (w2_fp32_in, use_tdm_store, inst_prefetch, wave_specialized_tdm)
    assert model_dim % tile_k == 0
    assert inter_dim % tile_n == 0

    x_fp32, w1_fp32, _, topk_ids, topk_weights = _make_inputs(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        even_dispatch=even_dispatch,
        seed=seed,
    )
    if x_fp32_in is not None:
        x_fp32 = x_fp32_in
    if w1_fp32_in is not None:
        w1_fp32 = w1_fp32_in
    if topk_ids_in is not None:
        topk_ids = topk_ids_in
    if topk_weights_in is not None:
        topk_weights = topk_weights_in

    routing = routing_in or build_routing_buffers(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        experts=experts,
        tile_m=tile_m,
        model_dim=model_dim,
        moe_sort_mode="torch",
    )
    (
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        _sorted_size,
        blocks,
    ) = routing

    cast = torch.float16 if in_dtype == "fp16" else torch.bfloat16
    stage1_out_dtype = "f16" if in_dtype == "fp16" else "bf16"
    out_torch_dtype = cast

    x_q = x_fp32.to(cast).contiguous()
    w1_q = w1_fp32.to(cast)
    w1_q_flat = w1_q.view(experts * (2 * inter_dim), model_dim).contiguous()

    out = torch.zeros((tokens, topk, inter_dim), device=x_q.device, dtype=out_torch_dtype)
    scale_x_1d = torch.empty((0,), device=x_q.device, dtype=torch.float32)
    scale_w1_1d = torch.empty((0,), device=x_q.device, dtype=torch.float32)
    sorted_weights_1d = sorted_weights.contiguous().view(-1)

    exe = compile_moe_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        in_dtype=in_dtype,
        out_dtype=stage1_out_dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=bool(doweight_stage1),
        waves_per_eu=waves_per_eu,
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

    _, us = run_perftest(
        launch,
        out,
        x_q,
        w1_q_flat,
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
            doweight_stage1=bool(doweight_stage1),
        )
        assert verify_output(out.to(torch.float32), ref, rtol=0.25, atol=0.25)

    if return_outputs:
        return out, us
    return None


def run_moe_stage2(
    *,
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    in_dtype: str = "fp16",
    out_dtype: str = "f16",
    seed: int = 0,
    num_iters: int = 3,
    num_warmup: int = 1,
    test_graph: bool = False,
    waves_per_eu: Optional[int] = None,
    expert_sched_mode: bool = True,
    even_dispatch: bool = False,
    use_reduce: bool = False,
    use_valid_mask: bool = False,
    valid_mask_in: Optional[torch.Tensor] = None,
    x_fp32_in: Optional[torch.Tensor] = None,
    w1_fp32_in: Optional[torch.Tensor] = None,
    w2_fp32_in: Optional[torch.Tensor] = None,
    topk_ids_in: Optional[torch.Tensor] = None,
    topk_weights_in: Optional[torch.Tensor] = None,
    routing_in: Optional[RoutingBuffers] = None,
    a2_in: Optional[torch.Tensor] = None,
    # ``moe_bench_main`` uses the gfx1250 API naming; alias to ``a2_in`` and
    # drop ``a2_scale_in`` because RDNA4 fp16/bf16 MoE has no A2 scale.
    a2_fp8_in: Optional[torch.Tensor] = None,
    a2_scale_in: Optional[torch.Tensor] = None,
    return_outputs: bool = False,
    skip_ref: bool = False,
    # Accepted for moe_bench_main compatibility; RDNA4 has no TDM hardware.
    use_tdm_store: bool = False,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
):
    _ = (a2_scale_in, use_tdm_store, inst_prefetch, wave_specialized_tdm)
    if valid_mask_in is not None and (not bool(use_reduce) or not bool(use_valid_mask)):
        raise ValueError("valid_mask_in requires use_reduce=True and use_valid_mask=True")
    if a2_in is None and a2_fp8_in is not None:
        a2_in = a2_fp8_in
    if model_dim % tile_n != 0:
        raise ValueError(
            f"Invalid stage2 tiling: model_dim ({model_dim}) must be divisible by tile_n ({tile_n})."
        )
    if inter_dim % tile_k != 0:
        raise ValueError(
            f"Invalid stage2 tiling: inter_dim ({inter_dim}) must be divisible by tile_k ({tile_k})."
        )

    x_fp32, w1_fp32, w2_fp32, topk_ids, topk_weights = _make_inputs(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        even_dispatch=even_dispatch,
        seed=seed,
    )
    if x_fp32_in is not None:
        x_fp32 = x_fp32_in
    if w1_fp32_in is not None:
        w1_fp32 = w1_fp32_in
    if w2_fp32_in is not None:
        w2_fp32 = w2_fp32_in
    if topk_ids_in is not None:
        topk_ids = topk_ids_in
    if topk_weights_in is not None:
        topk_weights = topk_weights_in

    routing = routing_in or build_routing_buffers(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        experts=experts,
        tile_m=tile_m,
        model_dim=model_dim,
        moe_sort_mode="torch",
    )
    (
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        _sorted_size,
        blocks,
    ) = routing

    cast = torch.float16 if in_dtype == "fp16" else torch.bfloat16
    w2_q = w2_fp32.to(cast).contiguous()
    w2_kernel = w2_q.view(experts * model_dim, inter_dim).contiguous().view(-1)

    if a2_in is not None:
        a2_q = a2_in.contiguous()
    else:
        out1_ref = torch_moe_gemm1(
            x_fp32.to(cast),
            w1_fp32.to(cast).view(experts * (2 * inter_dim), model_dim).contiguous(),
            None,
            None,
            topk_ids.to(torch.int64),
            topk_weights,
            inter_dim=inter_dim,
            doweight_stage1=bool(doweight_stage1),
        )
        a2_q = out1_ref.to(cast).contiguous()

    a2_scale_1d = torch.empty((0,), device=a2_q.device, dtype=torch.float32)
    w2_scale_1d = torch.empty((0,), device=a2_q.device, dtype=torch.float32)
    sorted_weights_1d = sorted_weights.contiguous().view(-1)

    out_s = str(out_dtype).strip().lower()
    if out_s in ("f16", "fp16", "half"):
        out_torch_dtype = torch.float16
    elif out_s in ("f32", "fp32", "float"):
        out_torch_dtype = torch.float32
    else:
        raise ValueError(f"out_dtype must be 'f16' or 'f32', got {out_dtype!r}")

    if bool(use_reduce) and out_torch_dtype == torch.float32:
        pytest.skip("reduce mode does not support out_dtype='f32'")

    out = torch.zeros((tokens, model_dim), device=a2_q.device, dtype=out_torch_dtype)
    out_perf = torch.zeros_like(out)

    compile_fn = (
        _make_reduce_mode_compile_fn(use_valid_mask=bool(use_valid_mask))
        if bool(use_reduce)
        else compile_moe_gemm2
    )
    exe = compile_fn(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=not bool(doweight_stage1),
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        waves_per_eu=waves_per_eu,
        expert_sched_mode=bool(expert_sched_mode),
    )
    is_reduce_exe = getattr(exe, "mode", None) == MoeGemm2Mode.REDUCE

    def launch(o, x, w, sx, sw, st, eids, sw_sorted):
        stream = torch.cuda.current_stream()
        valid_mask = None
        if is_reduce_exe and bool(use_valid_mask):
            if valid_mask_in is not None:
                valid_mask = valid_mask_in.to(
                    device=topk_ids.device,
                    dtype=torch.uint8,
                ).contiguous()
            else:
                valid_mask = get_topk_valid_mask(
                    topk_ids,
                    expert_mask=None,
                    dtype=torch.uint8,
                ).contiguous()
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
        a2_ref = a2_q
        if bool(use_reduce) and bool(use_valid_mask):
            if valid_mask_in is not None:
                valid_mask_ref = valid_mask_in.to(device=a2_q.device, dtype=a2_q.dtype)
            else:
                valid_mask_ref = get_topk_valid_mask(
                    topk_ids,
                    expert_mask=None,
                    dtype=a2_q.dtype,
                )
            a2_ref = a2_q * valid_mask_ref.view(tokens, topk, 1)
        ref2 = torch_moe_gemm2(
            a2_ref,
            w2_q,
            None,
            None,
            topk_ids.to(torch.int64),
            topk_weights,
            model_dim=model_dim,
            doweight_stage2=not bool(doweight_stage1),
        )
        assert verify_output(out.to(torch.float32), ref2, rtol=0.5, atol=0.5)

    if return_outputs:
        return out, us
    return None


def run_moe_gemm_2stage(
    *,
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
    waves_per_eu: Optional[int] = None,
    even_dispatch: bool = False,
    expert_sched_mode: bool = True,
    seed: int = 0,
    skip_ref: bool = False,
):
    x_fp32, w1_fp32, w2_fp32, topk_ids, topk_weights = _make_inputs(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        even_dispatch=even_dispatch,
        seed=seed,
    )
    routing = build_routing_buffers(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        experts=experts,
        tile_m=tile_m,
        model_dim=model_dim,
        moe_sort_mode="torch",
    )

    stage1_out, _ = run_moe_stage1(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n1,
        tile_k=tile_k1,
        doweight_stage1=doweight_stage1,
        in_dtype=in_dtype,
        waves_per_eu=waves_per_eu,
        test_graph=test_graph,
        expert_sched_mode=expert_sched_mode,
        x_fp32_in=x_fp32,
        w1_fp32_in=w1_fp32,
        topk_ids_in=topk_ids,
        topk_weights_in=topk_weights,
        routing_in=routing,
        return_outputs=True,
        skip_ref=bool(skip_ref),
    )

    run_moe_stage2(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n2,
        tile_k=tile_k2,
        doweight_stage1=doweight_stage1,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        waves_per_eu=waves_per_eu,
        test_graph=test_graph,
        expert_sched_mode=expert_sched_mode,
        x_fp32_in=x_fp32,
        w1_fp32_in=w1_fp32,
        w2_fp32_in=w2_fp32,
        topk_ids_in=topk_ids,
        topk_weights_in=topk_weights,
        routing_in=routing,
        a2_in=stage1_out,
        use_reduce=use_reduce,
        use_valid_mask=use_valid_mask,
        skip_ref=bool(skip_ref),
    )


@pytest.mark.parametrize("waves_per_eu", [1, 2], ids=["wpe1", "wpe2"])
def test_moe_2stage_waves_per_eu_smoke(waves_per_eu: int):
    shape = dict(tokens=32, model_dim=256, inter_dim=128, experts=4, topk=2, tile_m=16)
    stage1_out, _ = run_moe_stage1(
        **shape,
        tile_n=64,
        tile_k=128,
        doweight_stage1=False,
        in_dtype="fp16",
        waves_per_eu=waves_per_eu,
        return_outputs=True,
        skip_ref=True,
    )
    stage2_out, _ = run_moe_stage2(
        **shape,
        tile_n=128,
        tile_k=128,
        doweight_stage1=False,
        in_dtype="fp16",
        out_dtype="f16",
        waves_per_eu=waves_per_eu,
        a2_in=stage1_out.to(torch.float16),
        return_outputs=True,
        skip_ref=True,
    )
    assert torch.isfinite(stage1_out).all()
    assert torch.isfinite(stage2_out).all()


def test_moe_reduce_valid_mask_masks_invalid_routes():
    shape = dict(tokens=48, model_dim=256, inter_dim=128, experts=6, topk=3, tile_m=16)
    x_fp32, w1_fp32, w2_fp32, topk_ids, topk_weights = _make_inputs(
        tokens=shape["tokens"],
        model_dim=shape["model_dim"],
        inter_dim=shape["inter_dim"],
        experts=shape["experts"],
        topk=shape["topk"],
        even_dispatch=True,
        seed=7,
    )
    routing = build_routing_buffers(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        experts=shape["experts"],
        tile_m=shape["tile_m"],
        model_dim=shape["model_dim"],
        moe_sort_mode="torch",
    )
    valid_mask = _make_partial_valid_mask(topk_ids, experts=shape["experts"])

    stage1_out, _ = run_moe_stage1(
        **shape,
        tile_n=64,
        tile_k=128,
        doweight_stage1=False,
        in_dtype="fp16",
        x_fp32_in=x_fp32,
        w1_fp32_in=w1_fp32,
        topk_ids_in=topk_ids,
        topk_weights_in=topk_weights,
        routing_in=routing,
        return_outputs=True,
        skip_ref=False,
    )
    stage2_out, _ = run_moe_stage2(
        **shape,
        tile_n=64,
        tile_k=128,
        doweight_stage1=False,
        in_dtype="fp16",
        out_dtype="f16",
        use_reduce=True,
        use_valid_mask=True,
        valid_mask_in=valid_mask,
        x_fp32_in=x_fp32,
        w1_fp32_in=w1_fp32,
        w2_fp32_in=w2_fp32,
        topk_ids_in=topk_ids,
        topk_weights_in=topk_weights,
        routing_in=routing,
        a2_in=stage1_out,
        return_outputs=True,
        skip_ref=False,
    )

    w2_q = w2_fp32.to(torch.float16).contiguous()
    unmasked_ref = torch_moe_gemm2(
        stage1_out,
        w2_q,
        None,
        None,
        topk_ids.to(torch.int64),
        topk_weights,
        model_dim=shape["model_dim"],
        doweight_stage2=True,
    )
    assert not torch.allclose(
        stage2_out.to(torch.float32),
        unmasked_ref,
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n1, tile_k1, tile_n2, tile_k2, doweight_stage1",
    [
        pytest.param(64, 256, 128, 4, 2, 16, 64, 128, 64, 128, False, id="S"),
    ],
)
@pytest.mark.parametrize("in_dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("out_dtype", ["f16", "f32"], ids=["out_f16", "out_f32"])
@pytest.mark.parametrize("use_reduce", [False, True], ids=["atomic", "reduce"])
@pytest.mark.parametrize("use_valid_mask", [False, True], ids=["nomask", "mask"])
@pytest.mark.parametrize("test_graph", [False, True], ids=["eager", "graph"])
def test_moe_gemm_2stage_smoke(
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
    if (not bool(use_reduce)) and bool(use_valid_mask):
        pytest.skip("valid_mask is only used in reduce mode.")
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


@pytest.mark.parametrize(
    "tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n1, tile_k1, tile_n2, tile_k2",
    [
        pytest.param(129, 1024, 256, 8, 2, 32, 64, 128, 64, 128, id="M"),
        pytest.param(
            333,
            4096,
            2048,
            17,
            9,
            64,
            128,
            128,
            64,
            128,
            id="L",
            marks=pytest.mark.large_shape,
        ),
    ],
)
def test_moe_gemm_2stage_perf_smoke(
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
        doweight_stage1=False,
        in_dtype="fp16",
        out_dtype="f16",
        use_reduce=False,
        use_valid_mask=False,
        test_graph=False,
        skip_ref=(tokens >= 129),
    )


# ---------------------------------------------------------------------------
# Benchmark entry point (mirrors tests/kernels/test_moe_gemm_wmma_gfx1250.py)
# ---------------------------------------------------------------------------


def _bench_setup_data(tokens, model_dim, inter_dim, experts, topk, tile_m, seed=42):
    """Build random MoE data + routing buffers for RDNA4 bench sweeps."""
    device = torch.device("cuda")
    torch.manual_seed(seed)
    s = 0.2
    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32) * s
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32) * s
    w2_fp32 = torch.randn((experts, model_dim, inter_dim), device=device, dtype=torch.float32) * (
        s / math.sqrt(inter_dim)
    )
    score = torch.rand((tokens, experts), device=device, dtype=torch.float32)
    topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)
    routing = build_routing_buffers(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        experts=experts,
        tile_m=tile_m,
        model_dim=model_dim,
        moe_sort_mode="torch",
    )
    return x_fp32, w1_fp32, w2_fp32, topk_ids, topk_weights, routing


def _bench_prepare_a2(out1_fp16, _tokens, _topk, _inter_dim, in_dtype):
    """Convert stage1 fp16 output to stage2 activation input (fp16 or bf16)."""
    if in_dtype == "fp16":
        return out1_fp16, None
    if in_dtype == "bf16":
        return out1_fp16.to(torch.bfloat16), None
    raise ValueError(f"in_dtype must be 'fp16' or 'bf16', got {in_dtype!r}")


if __name__ == "__main__":
    import argparse
    import sys
    from tests.kernels.benchmark_common import add_moe_bench_args, moe_bench_main

    torch.set_default_device("cuda")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MoE 2-stage (FlyDSL RDNA4 / gfx120x WMMA fp16/bf16) benchmark",
    )
    parser.add_argument(
        "--in_dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Kernel input dtype (default: fp16).",
    )
    # RDNA4 has no TDM hardware; accept the same flags as the gfx1250 harness so
    # scripts/run_benchmark.sh can invoke both paths uniformly. They are ignored.
    parser.add_argument("--use_tdm_store", action="store_true", default=False)
    parser.add_argument("--inst_prefetch", action="store_true", default=False)
    parser.add_argument("--wave_specialized_tdm", action="store_true", default=False)
    add_moe_bench_args(parser)
    args = parser.parse_args()

    if not args.bench:
        print("Use --bench to run the RDNA4 MoE benchmark sweep.", file=sys.stderr)
        sys.exit(2)

    moe_bench_main(
        args,
        stage1_fn=run_moe_stage1,
        stage2_fn=run_moe_stage2,
        setup_data_fn=_bench_setup_data,
        prepare_a2_fn=_bench_prepare_a2,
    )
