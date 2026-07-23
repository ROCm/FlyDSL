#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx1250 grouped MoE (MXFP8 A x MXFP4 B, per-1x32 e8m0, silu) test."""

from __future__ import annotations

import os
import statistics
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.moe.moe_a8w4_mxscale_gfx1250 import launch_moe_gemm_a8w4  # noqa: E402
from tests.kernels.utils import gemm_common_utils as gcu  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

SCALE_BLOCK = 32


def _ptr(t):
    return flyc.from_c_void_p(fx.Int8, t.data_ptr(), assumed_align=16)


def _require_gpu():
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"requires gfx1250, got {arch}")


def _quant_mxfp4_weight(w_bf16: torch.Tensor):
    """(E,N,K) bf16 -> (packed (E,N,K//2) u8, e8m0 (E,N,K//32) u8, dequant f32)."""
    E, N, K = w_bf16.shape
    packed, scale, _ = gcu.per_1x32_f4_quant(w_bf16.reshape(E * N, K).float())
    w_deq = (
        gcu.mxfp4_to_f32(packed.view(torch.uint8))[..., :K]
        * gcu.e8m0_to_f32(scale.view(torch.uint8)).repeat_interleave(SCALE_BLOCK, dim=-1)[..., :K]
    )
    packed = packed.view(torch.uint8).view(E, N, K // 2).contiguous()
    scale = scale.view(torch.uint8).view(E, N, K // SCALE_BLOCK).contiguous()
    return packed, scale, w_deq.view(E, N, K)


def _interleave_gate_up_rows(w: torch.Tensor) -> torch.Tensor:
    """(E, 2*I, ...) GGUU [g..,u..] -> GUGU [g0,u0,g1,u1,...] at the row level."""
    inter = w.shape[1] // 2
    return torch.stack([w[:, :inter], w[:, inter:]], dim=2).flatten(1, 2).contiguous()


def _shuffle_weight_16x16(x: torch.Tensor) -> torch.Tensor:
    """aiter shuffle_weight(layout=(16,16)) on packed-uint8 weights, per expert."""
    E, N, Kp = x.shape
    BN, BK, K = 16, 32, 16  # IN=IK=16; BK=IK*2; K=16//elem_size(uint8)=16
    x_ = x.view(E, N // BN, BN, Kp // BK, BK // K, K).permute(0, 1, 3, 4, 2, 5).contiguous()
    return x_.view(E, N, Kp)


def _shuffle_scale_n32k4(s: torch.Tensor, *, gate_up: bool) -> torch.Tensor:
    """aiter shuffle_scale_n32k4: (E,N,K//32) e8m0 -> (E, N//32, (K//32)*32)."""
    s = s.view(torch.uint8).contiguous()
    E, N, k_scale = s.shape
    if gate_up:  # GUGU row interleave to match the stage1 weight layout
        s = s.view(E, 2, N // 2, k_scale).permute(0, 2, 1, 3).reshape(E, N, k_scale)
    g = s.view(E, N // 32, 32, k_scale // 4, 4).permute(0, 1, 3, 2, 4).contiguous()
    return g.reshape(E, N // 32, k_scale * 32)


def _prep_stage_weight(packed, scale, *, gate_up):
    if gate_up:
        packed = _interleave_gate_up_rows(packed)
    return _shuffle_weight_16x16(packed.contiguous()), _shuffle_scale_n32k4(scale, gate_up=gate_up)


def _route_contiguous_m(topk_ids: torch.Tensor, E: int, tile_m: int):
    """Assign each route to a per-expert, tile_m-aligned contiguous row block."""
    device = topk_ids.device
    tokens, topk = topk_ids.shape
    flat = topk_ids.reshape(-1)
    counts = torch.bincount(flat, minlength=E)
    tile_counts = ((counts + tile_m - 1) // tile_m) * tile_m
    ends = torch.cumsum(tile_counts, 0).to(torch.int32)  # exclusive end per expert
    starts = ends - tile_counts.to(torch.int32)
    contiguous_m = int(ends[-1].item()) if E else 0
    # per-route slot within its expert (route order = token-major)
    order = torch.argsort(flat, stable=True)
    slot = torch.empty_like(flat, dtype=torch.int32)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(flat.numel(), device=device)
    exp_start_in_sorted = torch.cumsum(
        torch.cat([torch.zeros(1, device=device, dtype=torch.long), counts.to(torch.long)]), 0
    )
    slot_sorted = torch.arange(flat.numel(), device=device) - exp_start_in_sorted[flat[order].to(torch.long)]
    slot[order] = slot_sorted.to(torch.int32)
    rows = starts[flat.to(torch.long)] + slot
    return ends.contiguous(), rows.view(tokens, topk).contiguous(), contiguous_m


def _quant_a(grouped_bf16: torch.Tensor, wmma_rep: int):
    """(1, cm, K) bf16 -> (payload (cm, K) fp8 u8, preshuffled A-scale u8)."""
    _, cm, K = grouped_bf16.shape
    x_q, scale = gcu.per_1x32_f8_quant(grouped_bf16[0].float())  # codes (cm,K), e8m0 (cm, K//32)
    payload = x_q.view(torch.uint8).contiguous()
    # Preshuffle row-major e8m0 (cm, Ws) -> (cm//wmma_rep, Ws*wmma_rep):
    # tile-local move of the wmma_rep row axis next to the trailing dword.
    Ws = K // 32
    sd = Ws // 4
    s = scale.view(torch.uint8).view(cm // (wmma_rep * 16), wmma_rep, 16, sd, 4)
    s = s.permute(0, 2, 3, 1, 4).contiguous()  # (tiles, 16, sd, wmma_rep, 4)
    a_scale = s.reshape(cm // wmma_rep, Ws * wmma_rep)
    return payload, a_scale


def _gemm(
    out,
    a_payload,
    w,
    a_scale,
    w_scale,
    m_tile_map,
    *,
    E,
    cm,
    N,
    K,
    tile_m,
    tile_n,
    tile_k,
    stage1_act,
    cluster_n=1,
):
    nb = min(3, max(1, K // tile_k))
    launch_moe_gemm_a8w4(
        out,
        _ptr(a_payload),
        _ptr(w),
        a_scale.view(torch.int32),
        w_scale.view(torch.int32),
        cm,
        torch.cuda.current_stream(),
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        1,  # m_warp
        4,  # n_warp
        0,  # out_is_f16
        nb,
        0,  # a_is_fp4
        _ptr(m_tile_map),
        E,
        stage1_act,
        0,  # has_bias
        _ptr(a_payload),  # bias ptr (unused)
        7.0,  # swiglu_limit (unused for silu)
        1,  # cluster_m
        cluster_n,
    )


def _grouped_moe(
    hidden,
    w1_sh,
    w1_s,
    w2_sh,
    w2_s,
    topk_ids,
    topk_weight,
    *,
    E,
    inter_dim,
    tile_m=64,
    tile_n=256,
    tile_k=256,
    cluster_n=1,
):
    device = hidden.device
    tokens, model_dim = hidden.shape
    wmma_rep = tile_m // 16
    m_tile_map, topids_to_rows, cm = _route_contiguous_m(topk_ids, E, tile_m)

    # torch route+gather: scatter each token's hidden into its grouped row (pad=0).
    a1_bf16 = torch.zeros((1, cm, model_dim), dtype=torch.bfloat16, device=device)
    src = hidden[torch.arange(tokens, device=device).repeat_interleave(topk_ids.shape[1])]
    a1_bf16[0, topids_to_rows.reshape(-1).long()] = src

    a1_p, a1_s = _quant_a(a1_bf16, wmma_rep)

    # gemm1: fused silu, bf16 grouped intermediate; quant a2 in torch (like older archs).
    a2_bf16 = torch.empty((1, cm, inter_dim), dtype=torch.bfloat16, device=device)
    _gemm(
        a2_bf16,
        a1_p,
        w1_sh,
        a1_s,
        w1_s,
        m_tile_map,
        E=E,
        cm=cm,
        N=2 * inter_dim,
        K=model_dim,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        stage1_act=1,
        cluster_n=cluster_n,
    )
    a2_p, a2_s = _quant_a(a2_bf16, wmma_rep)

    grouped_out = torch.empty((1, cm, model_dim), dtype=torch.bfloat16, device=device)
    _gemm(
        grouped_out,
        a2_p,
        w2_sh,
        a2_s,
        w2_s,
        m_tile_map,
        E=E,
        cm=cm,
        N=model_dim,
        K=inter_dim,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        stage1_act=0,
        cluster_n=cluster_n,
    )

    # torch gather-reduce: out[t] = sum_k w[t,k] * grouped_out[topids_to_rows[t,k]].
    g = grouped_out[0][topids_to_rows.long()].float()  # (tokens, topk, model_dim)
    out = (g * topk_weight.unsqueeze(-1).float()).sum(1)
    return out.to(hidden.dtype)


def _reference_moe(hidden, w1_deq, w2_deq, topk_ids, topk_weight, inter_dim):
    tokens, model_dim = hidden.shape
    E = w1_deq.shape[0]
    x = hidden.float()
    out = torch.zeros(tokens, model_dim, dtype=torch.float32, device=hidden.device)
    for e in range(E):
        idx = (topk_ids == e).nonzero(as_tuple=False)
        if idx.numel() == 0:
            continue
        t, s = idx[:, 0], idx[:, 1]
        y1 = x[t] @ w1_deq[e].T
        gate, up = y1[:, :inter_dim], y1[:, inter_dim:]
        a2 = torch.nn.functional.silu(gate) * up
        out[t] += (a2 @ w2_deq[e].T) * topk_weight[t, s].unsqueeze(-1).float()
    return out


def _build_case(E, model_dim, inter_dim, token_num, topk, seed=0):
    _require_gpu()
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    hidden = (torch.randn(token_num, model_dim, dtype=torch.float32, device=dev) * 0.5).bfloat16()
    w1 = (torch.randn(E, 2 * inter_dim, model_dim, device=dev) * 0.2).bfloat16()
    w2 = (torch.randn(E, model_dim, inter_dim, device=dev) * 0.2).bfloat16()
    w1_p, w1_s, w1_deq = _quant_mxfp4_weight(w1)
    w2_p, w2_s, w2_deq = _quant_mxfp4_weight(w2)
    w1_sh, w1_s = _prep_stage_weight(w1_p, w1_s, gate_up=True)
    w2_sh, w2_s = _prep_stage_weight(w2_p, w2_s, gate_up=False)
    topk_ids = torch.randint(0, E, (token_num, topk), dtype=torch.int32, device=dev)
    topk_weight = torch.rand(token_num, topk, dtype=torch.float32, device=dev) + 0.5
    ref = _reference_moe(hidden, w1_deq, w2_deq, topk_ids, topk_weight, inter_dim)
    args = dict(
        hidden=hidden,
        w1_sh=w1_sh,
        w1_s=w1_s,
        w2_sh=w2_sh,
        w2_s=w2_s,
        topk_ids=topk_ids,
        topk_weight=topk_weight,
        E=E,
        inter_dim=inter_dim,
    )
    return args, ref


# (E, model_dim, inter_dim, token_num, topk); dims are multiples of tile (256).
_CASES = [
    (1, 512, 256, 64, 1),  # single expert, topk=1
    (8, 256, 256, 128, 2),  # min dims
    (8, 512, 256, 64, 2),
    (8, 512, 256, 128, 2),
    (16, 768, 512, 256, 4),
    (8, 1024, 512, 32, 2),
    (8, 2048, 512, 64, 2),  # larger K
    (32, 512, 256, 256, 4),  # many experts
]


def _check_accuracy(out, ref):
    out = out.float()
    ref = ref.to(out.device)
    assert torch.isfinite(out).all(), "non-finite MoE output"
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    rel_l2 = ((out - ref).norm() / ref.norm().clamp_min(1e-6)).item()
    assert cos >= 0.92, f"cosine similarity {cos:.4f} < 0.92"
    assert rel_l2 <= 0.5, f"relative L2 error {rel_l2:.4f} > 0.5"


@pytest.mark.parametrize("E, model_dim, inter_dim, token_num, topk", _CASES)
def test_grouped_moe_accuracy(E, model_dim, inter_dim, token_num, topk):
    args, ref = _build_case(E, model_dim, inter_dim, token_num, topk)
    _check_accuracy(_grouped_moe(**args), ref)


# Decode/prefill token-count sweep on a fixed config (routing edge cases).
@pytest.mark.parametrize("token_num", [1, 2, 7, 16, 63, 128, 512, 1024])
def test_grouped_moe_token_sweep(token_num):
    args, ref = _build_case(8, 512, 256, token_num, 2, seed=token_num)
    _check_accuracy(_grouped_moe(**args), ref)


def test_grouped_moe_stability():
    args, _ = _build_case(8, 512, 256, 128, 2, seed=1)
    first = _grouped_moe(**args).clone()
    torch.cuda.synchronize()
    for _ in range(5):
        again = _grouped_moe(**args)
        torch.cuda.synchronize()
        assert torch.equal(first, again), "MoE output is non-deterministic across launches"


@pytest.mark.parametrize("cluster_n", [2, 4])
def test_grouped_moe_cluster(cluster_n):
    args, _ = _build_case(8, 1024, 512, 128, 2, seed=3)
    base = _grouped_moe(**args).clone()
    torch.cuda.synchronize()
    clustered = _grouped_moe(**args, cluster_n=cluster_n)
    torch.cuda.synchronize()
    assert torch.equal(base, clustered), f"cluster_n={cluster_n} output differs from non-cluster"


@pytest.mark.benchmark
def test_grouped_moe_perf():
    """Time only the FlyDSL kernels (quant + gemm1 + gemm2); routing is torch glue."""
    args, _ = _build_case(16, 768, 512, 256, 4, seed=2)
    hidden, E, inter_dim = args["hidden"], args["E"], args["inter_dim"]
    tile_m, tile_n, tile_k = 64, 256, 256
    wmma_rep = tile_m // 16
    tokens, model_dim = hidden.shape
    device = hidden.device
    # Routing (torch glue) done once, outside the timed region.
    m_tile_map, topids_to_rows, cm = _route_contiguous_m(args["topk_ids"], E, tile_m)
    a1_bf16 = torch.zeros((1, cm, model_dim), dtype=torch.bfloat16, device=device)
    src = hidden[torch.arange(tokens, device=device).repeat_interleave(args["topk_ids"].shape[1])]
    a1_bf16[0, topids_to_rows.reshape(-1).long()] = src

    def kernels_only():
        a1_p, a1_s = _quant_a(a1_bf16, wmma_rep)
        a2_bf16 = torch.empty((1, cm, inter_dim), dtype=torch.bfloat16, device=device)
        _gemm(
            a2_bf16,
            a1_p,
            args["w1_sh"],
            a1_s,
            args["w1_s"],
            m_tile_map,
            E=E,
            cm=cm,
            N=2 * inter_dim,
            K=model_dim,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            stage1_act=1,
        )
        a2_p, a2_s = _quant_a(a2_bf16, wmma_rep)
        g_out = torch.empty((1, cm, model_dim), dtype=torch.bfloat16, device=device)
        _gemm(
            g_out,
            a2_p,
            args["w2_sh"],
            a2_s,
            args["w2_s"],
            m_tile_map,
            E=E,
            cm=cm,
            N=model_dim,
            K=inter_dim,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            stage1_act=0,
        )

    for _ in range(10):
        kernels_only()
    torch.cuda.synchronize()
    iters = 50
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starts, ends):
        s.record()
        kernels_only()
        e.record()
    torch.cuda.synchronize()
    us = statistics.median(sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)))
    print(f"\ngrouped MoE kernels (quant+gemm1+gemm2) E16 m768 i512 t256 topk4: {us:.2f} us")
    assert us > 0
