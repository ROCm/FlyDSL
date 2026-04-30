#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Self-contained a8w4smooth MoE GEMM 2-stage tests.

Exercises `kernels.a8w4_moe_gemm_2stage` (the a8w4smooth-only kernel module)
via standalone runner functions defined here.  All a8w4smooth-specific helpers
(weight generation, packing, dequant) live in this file so that the legacy
`test_moe_gemm.py` can stay clean of a8w4smooth additions.
"""

import os
import sys
from typing import Tuple, Optional

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "build", "python_packages")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import flydsl.compiler as flyc
from flydsl.runtime.device import get_rocm_arch
from tests.kernels.test_ref import torch_moe_gemm1, torch_moe_gemm2
from tests.utils import pertoken_quant, shuffle_weight
from tests.test_common import verify_output, run_perftest
from tests.kernels.test_moe_gemm import build_routing_buffers

from kernels.a8w4_moe_gemm_2stage import (
    compile_moe_gemm1,
    compile_moe_gemm2,
    compile_moe_gemm2_ex,
    compile_moe_reduction,
    MoeGemm2Mode,
)

ARCH = get_rocm_arch()

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


# ---------------------------------------------------------------------------
# a8w4smooth packing / weight helpers
# ---------------------------------------------------------------------------

def _pack_shuffled_int8_to_packed_int4_no_perm(x_shuf_i8: torch.Tensor) -> torch.Tensor:
    """Pack a K64-interleaved int8 tensor into packed int4 bytes.

    Input is K64-interleaved: [lo[0], hi[0], lo[1], hi[1], ...].
    Each contiguous 8-value block [v0..v7] -> 4 bytes with adjacent pairing:
      b0=(v1<<4)|v0, b1=(v3<<4)|v2, b2=(v5<<4)|v4, b3=(v7<<4)|v6.

    This ensures each byte contains (lo_nibble, hi_nibble), so the kernel's
    even/odd split cleanly separates lo64 from hi64.
    """
    flat = x_shuf_i8.contiguous().view(-1).to(torch.int16)
    assert flat.numel() % 8 == 0
    u = (flat & 0xF).to(torch.uint8).view(-1, 8)
    out = torch.empty((u.shape[0], 4), device=u.device, dtype=torch.uint8)
    out[:, 0] = u[:, 0] | (u[:, 1] << 4)
    out[:, 1] = u[:, 2] | (u[:, 3] << 4)
    out[:, 2] = u[:, 4] | (u[:, 5] << 4)
    out[:, 3] = u[:, 6] | (u[:, 7] << 4)
    return out.view(-1).to(torch.int8)


def _inverse_interleave_k64_in_128(x: torch.Tensor) -> torch.Tensor:
    """Inverse of 128-wise K-interleave (0,64,1,65,...)."""
    K = x.shape[-1]
    if K % 128 != 0:
        raise ValueError(f"requires K%128==0, got K={K}")
    x128 = x.view(*x.shape[:-1], K // 128, 128)
    low = x128[..., 0::2]
    high = x128[..., 1::2]
    y128 = torch.cat([low, high], dim=-1)
    return y128.view(*x.shape)


def _unshuffle_weight_base(x_shuf: torch.Tensor, layout=(16, 16), use_int4: bool = False) -> torch.Tensor:
    """Inverse of tests.utils.shuffle_weight (without interleave)."""
    x_type = x_shuf.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x_shuf = x_shuf.view(torch.uint8)

    IN, IK = layout
    BK = IK * 2
    Kpack = 16 // x_shuf.element_size() if not use_int4 else 32
    BN = IN

    assert x_shuf.shape[-2] % BN == 0, f"{x_shuf.shape[-2]} % {BN} != 0"
    assert x_shuf.shape[-1] % BK == 0, f"{x_shuf.shape[-1]} % {BK} != 0"

    x_ = x_shuf
    x_ = x_.view(-1, x_shuf.shape[-2] // BN, x_shuf.shape[-1] // BK, BK // Kpack, BN, Kpack)
    x_ = x_.permute(0, 1, 4, 2, 3, 5).contiguous()
    return x_.view(*x_shuf.shape).view(x_type)


def build_a8w4smooth_moe_weight(
    *,
    experts: int,
    rows_per_expert: int,
    K: int,
    device: torch.device,
    seed: int = 0,
    interleave_k64: bool = True,
) -> Tuple[
    torch.Tensor,  # w_packed_i8 (packed 4-bit bytes)
    torch.Tensor,  # qscale_u8 (physical u8 5D)
    torch.Tensor,  # qzero_u8  (physical u8 5D)
    torch.Tensor,  # qscale_packed_i32 (physical packed4 i32 4D)
    torch.Tensor,  # qzero_packed_i32  (physical packed4 i32 4D)
    torch.Tensor,  # w_int8_unshuffled_flat [experts*rows_per_expert, K]
]:
    """Generate a8w4smooth weights + qparams in the agreed PDF physical layouts."""
    if int(experts) <= 0 or int(rows_per_expert) <= 0:
        raise ValueError(f"invalid experts/rows_per_expert: {experts=}, {rows_per_expert=}")
    if K % 256 != 0:
        raise ValueError(f"requires K%256==0, got K={K}")
    if interleave_k64 and (K % 128 != 0):
        raise ValueError(f"interleave_k64 requires K%128==0, got K={K}")
    if rows_per_expert % 16 != 0:
        raise ValueError(f"requires rows_per_expert%16==0, got rows_per_expert={rows_per_expert}")

    torch.manual_seed(int(seed))

    nb = rows_per_expert // 16
    g256 = K // 256

    u4_unshuf = torch.randint(0, 16, (experts, rows_per_expert, K), device=device, dtype=torch.uint8)
    u4_unshuf_i8 = u4_unshuf.view(torch.int8)

    u4_shuf_base_i8 = shuffle_weight(u4_unshuf_i8, use_int4=True, interleave_k64=False)
    u4_shuf_base = (u4_shuf_base_i8.view(torch.uint8) & 0xF).contiguous()

    u4_shuf_i8 = shuffle_weight(u4_unshuf_i8, use_int4=True, interleave_k64=bool(interleave_k64))
    u4_shuf = (u4_shuf_i8.view(torch.uint8) & 0xF).contiguous()

    qparam_shape = (experts, nb, g256, 16, 4)
    qs_i32 = torch.randint(1, 3, qparam_shape, device=device, dtype=torch.int32)
    qz_i32 = torch.randint(0, 16, qparam_shape, device=device, dtype=torch.int32)
    qscale_u8 = qs_i32.to(torch.uint8)
    qzero_u8 = qz_i32.to(torch.uint8)

    u4_logical_view = u4_unshuf.view(experts, nb, 16, g256, 4, 64).to(torch.int32)
    u4_logical_view = u4_logical_view.permute(0, 1, 3, 2, 4, 5)

    u8_logical_view = (u4_logical_view * qs_i32.unsqueeze(-1)) + qz_i32.unsqueeze(-1)
    u8_logical_view = torch.clamp(u8_logical_view, 0, 255).to(torch.uint8)

    u8_unshuf = u8_logical_view.permute(0, 1, 3, 2, 4, 5).reshape(experts, rows_per_expert, K)

    w_u8_shuf_i8 = shuffle_weight(u8_unshuf.view(torch.int8), use_int4=True, interleave_k64=bool(interleave_k64))
    w_u8_shuf = w_u8_shuf_i8.view(torch.uint8)

    w_i8_shuf = (w_u8_shuf ^ 0x80).view(torch.int8)

    w_packed = _pack_shuffled_int8_to_packed_int4_no_perm(u4_shuf.reshape(-1, K).to(torch.int8))

    qscale_i32 = (
        qscale_u8[..., 0].to(torch.int32)
        | (qscale_u8[..., 1].to(torch.int32) << 8)
        | (qscale_u8[..., 2].to(torch.int32) << 16)
        | (qscale_u8[..., 3].to(torch.int32) << 24)
    )
    qzero_i32 = (
        qzero_u8[..., 0].to(torch.int32)
        | (qzero_u8[..., 1].to(torch.int32) << 8)
        | (qzero_u8[..., 2].to(torch.int32) << 16)
        | (qzero_u8[..., 3].to(torch.int32) << 24)
    )

    w_i8_unshuffled_flat = (u8_unshuf.to(torch.int32) ^ 0x80).to(torch.int8).reshape(-1, K).contiguous()

    w_packed_u8 = w_packed.view(torch.uint8).contiguous()
    assert w_packed_u8.numel() == (experts * rows_per_expert * K) // 2
    w_bytes_view = w_packed_u8.view(experts, rows_per_expert // 16, K // 128, 4, 16, 16)
    lo = w_bytes_view & 0xF
    hi = (w_bytes_view >> 4) & 0xF
    w_u4_view = torch.stack([lo, hi], dim=-1).reshape(experts, rows_per_expert // 16, K // 128, 4, 16, 32)
    assert tuple(w_u4_view.shape) == (experts, rows_per_expert // 16, K // 128, 4, 16, 32)

    return w_packed, qscale_u8, qzero_u8, qscale_i32, qzero_i32, w_i8_unshuffled_flat


def _a8w4smooth_packed_w_to_unshuffled_int8(
    w_packed_i8: torch.Tensor,
    qscale_kn: torch.Tensor,  # [K//block, N] int32
    qzero_kn: torch.Tensor,   # [K//block, N] int32
    *,
    N: int,
    K: int,
    experts: int,
    rows_per_expert: int,
    scale_block_k: int = 64,
    layout=(16, 16),
) -> torch.Tensor:
    """Rebuild unshuffled int8 weights from packed 4-bit + qscale/qzero.

    Dequant contract (byte-wise): int8_bits = (u4 * qscale + qzero) ^ 0x80
    """

    def _unshuffle_weight(x_shuf: torch.Tensor, layout=(16, 16), use_int4: bool = False) -> torch.Tensor:
        x_type = x_shuf.dtype
        if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
            x_shuf = x_shuf.view(torch.uint8)

        IN, IK = layout
        BK = IK * 2
        Kpack = 16 // x_shuf.element_size() if not use_int4 else 32
        BN = IN

        assert x_shuf.shape[-2] % BN == 0, f"{x_shuf.shape[-2]} % {BN} != 0"
        assert x_shuf.shape[-1] % BK == 0, f"{x_shuf.shape[-1]} % {BK} != 0"

        x_ = x_shuf
        x_ = x_.view(-1, x_shuf.shape[-2] // BN, x_shuf.shape[-1] // BK, BK // Kpack, BN, Kpack)
        x_ = x_.permute(0, 1, 4, 2, 3, 5).contiguous()
        x_ = x_.view(*x_shuf.shape).view(x_type)
        return x_

    def _unpack_uint4_from_packed_int4_no_perm(packed_i8: torch.Tensor) -> torch.Tensor:
        p = packed_i8.view(torch.uint8).contiguous().view(-1, 4)
        out = torch.empty((p.shape[0], 8), device=p.device, dtype=torch.uint8)

        b0, b1, b2, b3 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        out[:, 0] = b0 & 0xF
        out[:, 1] = b0 >> 4
        out[:, 2] = b1 & 0xF
        out[:, 3] = b1 >> 4
        out[:, 4] = b2 & 0xF
        out[:, 5] = b2 >> 4
        out[:, 6] = b3 & 0xF
        out[:, 7] = b3 >> 4

        return out.view(-1)

    assert int(experts) > 0 and int(rows_per_expert) > 0
    assert int(N) == int(experts) * int(rows_per_expert), f"N={N}, experts={experts}, rows_per_expert={rows_per_expert}"
    assert K % scale_block_k == 0
    num_blocks = K // scale_block_k
    assert tuple(qscale_kn.shape) == (num_blocks, N), f"qscale_kn.shape={tuple(qscale_kn.shape)}"
    assert tuple(qzero_kn.shape) == (num_blocks, N), f"qzero_kn.shape={tuple(qzero_kn.shape)}"

    w_u4 = _unpack_uint4_from_packed_int4_no_perm(w_packed_i8).view(N, K).to(torch.int32)
    w_u4 = w_u4.view(N, num_blocks, scale_block_k)

    qs = qscale_kn.t().contiguous().to(torch.int32)
    qz = qzero_kn.t().contiguous().to(torch.int32)

    u8 = w_u4 * qs.unsqueeze(-1) + qz.unsqueeze(-1)
    u8 = u8.clamp(0, 255).to(torch.uint8)
    w_i8_shuf = (u8 ^ 0x80).view(torch.int8).view(experts, rows_per_expert, K)

    w_i8 = _unshuffle_weight(w_i8_shuf, layout=layout, use_int4=False)
    return w_i8.view(N, K)


# ---------------------------------------------------------------------------
# Standalone runners
# ---------------------------------------------------------------------------

def run_moe_stage1_a8w4smooth(
    *,
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    seed: int = 0,
    num_iters: int = 5,
    num_warmup: int = 2,
    moe_sort_mode: Optional[str] = None,
    skip_ref: bool = False,
):
    """Stage1 a8w4smooth runner: W4A8 + smoothquant + zero-point dequant."""
    if model_dim % 256 != 0:
        raise ValueError(f"a8w4smooth requires model_dim%256==0, got {model_dim}")
    if int(tile_k) not in (128, 256):
        raise ValueError(f"a8w4smooth requires tile_k in (128,256), got {tile_k}")

    os.environ.setdefault("FLIR_A8W4SMOOTH_QPARAM_FORMAT", "packed4")
    os.environ.setdefault("FLIR_A8W4SMOOTH_INTERLEAVE_K64", "1")

    device = torch.device("cuda")
    torch.manual_seed(int(seed))

    # --- Activations (int8, smooth-expanded to [tokens*topk, K]) ---
    x_fp32 = torch.randn(tokens, model_dim, device=device, dtype=torch.float32)

    # Routing
    gating = torch.randn(tokens, experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), k=int(topk), dim=-1)
    topk_weights = topk_weights.to(torch.float32).contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()

    routing = build_routing_buffers(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        experts=int(experts),
        model_dim=int(model_dim),
        tile_m=int(tile_m),
        moe_sort_mode=moe_sort_mode,
    )
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, sorted_size, blocks = routing

    # smoothquant: per-expert smooth scale, then dynamic quant to int8 per (t,slot).
    smooth_scale = (0.75 + 0.5 * torch.rand((experts, model_dim), device=device, dtype=torch.float32))
    x_route = x_fp32[:, None, :].expand(tokens, topk, model_dim)
    x_route = x_route * smooth_scale[topk_ids.to(torch.int64)]
    amax = torch.amax(torch.abs(x_route), dim=-1, keepdim=True)
    scale_x = amax / 127.0
    scale_x[scale_x == 0] = 1.0
    x_q = (x_route / scale_x).to(torch.int8)
    x_q = x_q.permute(1, 0, 2).contiguous()
    scale_x = scale_x.permute(1, 0, 2).contiguous()
    x_q_flat = x_q.view(tokens * topk, model_dim)
    scale_x_1d = scale_x.view(-1).contiguous()

    # --- Weights (a8w4smooth packed4) ---
    rows_per_expert = 2 * inter_dim
    (
        w_packed_i8,
        qscale_u8,
        qzero_u8,
        qscale_i32,
        qzero_i32,
        w_i8_unshuffled_flat,
    ) = build_a8w4smooth_moe_weight(
        experts=int(experts),
        rows_per_expert=int(rows_per_expert),
        K=int(model_dim),
        device=device,
        seed=int(seed),
        interleave_k64=True,
    )

    # Kernel buffers (flat 1D as required).
    w_kernel = w_packed_i8.view(-1).contiguous()
    qs_1d = qscale_i32.view(-1).contiguous()
    qz_1d = qzero_i32.view(-1).contiguous()
    sw_1d = sorted_weights.contiguous().view(-1)
    scale_w_1d = torch.full((experts * rows_per_expert,), 1e-3, device=device, dtype=torch.float32)
    scale_w1_flat_ref = scale_w_1d.view(experts * rows_per_expert, 1)

    out = torch.empty((tokens, topk, inter_dim), device=device, dtype=torch.float16)

    exe = compile_moe_gemm1(
        model_dim=int(model_dim),
        inter_dim=int(inter_dim),
        experts=int(experts),
        topk=int(topk),
        in_dtype="a8w4smooth",
        tile_m=int(tile_m),
        tile_n=int(tile_n),
        tile_k=int(tile_k),
        doweight_stage1=False,
        use_cshuffle_epilog=False,
        out_dtype="f16",
    )

    def _args(o, x, w, sx, sw, qs, qz, st, eids, sw_sorted):
        return (o, x, w, sx, sw, qs, qz, st, eids, sw_sorted,
                num_valid_ids, tokens, inter_dim, model_dim, int(blocks),
                torch.cuda.current_stream())

    compiled_exe = flyc.compile(exe, *_args(out, x_q_flat, w_kernel, scale_x_1d, scale_w_1d,
                                             qs_1d, qz_1d,
                                             sorted_token_ids, sorted_expert_ids, sw_1d))

    def launch(o, x, w, sx, sw, qs, qz, st, eids, sw_sorted):
        compiled_exe(*_args(o, x, w, sx, sw, qs, qz, st, eids, sw_sorted))

    _, us = run_perftest(
        launch, out, x_q_flat, w_kernel, scale_x_1d, scale_w_1d, qs_1d, qz_1d,
        sorted_token_ids, sorted_expert_ids, sw_1d,
        num_iters=int(num_iters), num_warmup=int(num_warmup),
    )
    torch.cuda.synchronize()

    if not bool(skip_ref):
        K = model_dim
        x_ref = x_q.view(topk, tokens, model_dim).permute(1, 0, 2).contiguous()
        sx_ref = scale_x.view(topk, tokens, 1).permute(1, 0, 2).contiguous()
        ref = torch_moe_gemm1(
            x_ref,
            w_i8_unshuffled_flat.view(experts * rows_per_expert, K).to(torch.float32),
            sx_ref,
            scale_w1_flat_ref,
            topk_ids.to(torch.int64),
            topk_weights,
            inter_dim=inter_dim, doweight_stage1=False,
        )
        diff = (out.to(torch.float32) - ref)
        print(f"[DEBUG a8w4smooth] out shape={out.shape}, ref shape={ref.shape}")
        print(f"[DEBUG a8w4smooth] out[0,0,:16]={[f'{v:.4f}' for v in out[0,0,:16].tolist()]}")
        print(f"[DEBUG a8w4smooth] ref[0,0,:16]={[f'{v:.4f}' for v in ref[0,0,:16].tolist()]}")
        print(f"[DEBUG a8w4smooth] diff[0,0,:16]={[f'{v:.4f}' for v in diff[0,0,:16].tolist()]}")
        assert verify_output(out.to(torch.float32), ref, rtol=0.5, atol=0.5)

    flops = 2 * tokens * topk * (2 * inter_dim) * model_dim
    tflops = flops / (us / 1e6) / 1e12
    active_experts = min(experts, tokens * topk)
    rows_s1 = 2 * inter_dim
    bytes_moved = 0
    bytes_moved += tokens * topk * model_dim * 1  # x int8
    bytes_moved += (active_experts * rows_s1 * model_dim) // 2  # w packed int4
    bytes_moved += tokens * topk * inter_dim * 2  # out fp16
    bytes_moved += tokens * topk * 4  # scale_x f32
    bytes_moved += active_experts * rows_s1 * 4  # scale_w f32
    bytes_moved += active_experts * (rows_s1 // 16) * (model_dim // 256) * 16 * 4 * 2  # qscale+qzero i32
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(
        f"FLIR MoE stage1[a8w4smooth]: tokens={tokens} M={model_dim} N={inter_dim} "
        f"E={experts} topk={topk} tile=({tile_m},{tile_n},{tile_k}) "
        f"=> {us:.1f}us ({tflops:.1f} TFLOPS) {tbps:.3f} TB/s"
    )
    return out


def run_moe_stage2_a8w4smooth(
    *,
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    seed: int = 0,
    num_iters: int = 5,
    num_warmup: int = 2,
    moe_sort_mode: Optional[str] = None,
    skip_ref: bool = False,
):
    """Stage2 a8w4smooth runner: W4A8 + smoothquant + zero-point dequant on the down-projection."""
    if inter_dim % 256 != 0:
        raise ValueError(f"a8w4smooth stage2 requires inter_dim%256==0, got {inter_dim}")
    if int(tile_k) not in (128, 256):
        raise ValueError(f"a8w4smooth requires tile_k in (128,256), got {tile_k}")

    os.environ.setdefault("FLIR_A8W4SMOOTH_QPARAM_FORMAT", "packed4")
    os.environ.setdefault("FLIR_A8W4SMOOTH_INTERLEAVE_K64", "1")

    device = torch.device("cuda")
    torch.manual_seed(int(seed))

    # Routing
    gating = torch.randn(tokens, experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), k=int(topk), dim=-1)
    topk_weights = topk_weights.to(torch.float32).contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()

    routing = build_routing_buffers(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        experts=int(experts),
        model_dim=int(model_dim),
        tile_m=int(tile_m),
        moe_sort_mode=moe_sort_mode,
    )
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, sorted_size, blocks = routing

    # A2 (stage2 input): per-(t,slot) int8 with scale_x. Build directly from random fp32.
    a2_fp32 = torch.randn(tokens, topk, inter_dim, device=device, dtype=torch.float32)
    a2_q, a2_scale = pertoken_quant(a2_fp32, quant_dtype=torch.int8)
    a2_q_flat = a2_q.view(tokens * topk, inter_dim).contiguous()
    a2_scale_1d = a2_scale.view(-1).contiguous()

    # Stage2 weights (a8w4smooth packed4): rows_per_expert = model_dim, K = inter_dim.
    rows_per_expert = model_dim
    (
        w_packed_i8,
        qscale_u8,
        qzero_u8,
        qscale_i32,
        qzero_i32,
        w_i8_unshuffled_flat,
    ) = build_a8w4smooth_moe_weight(
        experts=int(experts),
        rows_per_expert=int(rows_per_expert),
        K=int(inter_dim),
        device=device,
        seed=int(seed) + 22,
        interleave_k64=True,
    )

    w_kernel = w_packed_i8.view(-1).contiguous()
    qs_1d = qscale_i32.view(-1).contiguous()
    qz_1d = qzero_i32.view(-1).contiguous()
    sw_1d = sorted_weights.contiguous().view(-1)
    scale_w_1d = torch.full((experts * rows_per_expert,), 1e-3, device=device, dtype=torch.float32)
    scale_w_ref = scale_w_1d.view(experts, model_dim, 1)

    out = torch.zeros((tokens, model_dim), device=device, dtype=torch.float16)
    out_perf = torch.zeros_like(out)

    exe = compile_moe_gemm2(
        model_dim=int(model_dim),
        inter_dim=int(inter_dim),
        experts=int(experts),
        topk=int(topk),
        in_dtype="a8w4smooth",
        tile_m=int(tile_m),
        tile_n=int(tile_n),
        tile_k=int(tile_k),
        doweight_stage2=True,
        out_dtype="f16",
    )

    def _args(o, x, w, sx, sw, qs, qz, st, eids, sw_sorted):
        return (o, x, w, sx, sw, qs, qz, st, eids, sw_sorted,
                num_valid_ids, tokens, model_dim, inter_dim, int(blocks),
                torch.cuda.current_stream())

    compiled_exe = flyc.compile(exe, *_args(out_perf, a2_q_flat, w_kernel, a2_scale_1d, scale_w_1d,
                                              qs_1d, qz_1d,
                                              sorted_token_ids, sorted_expert_ids, sw_1d))

    def launch(o, x, w, sx, sw, qs, qz, st, eids, sw_sorted):
        compiled_exe(*_args(o, x, w, sx, sw, qs, qz, st, eids, sw_sorted))

    _, us = run_perftest(
        launch, out_perf, a2_q_flat, w_kernel, a2_scale_1d, scale_w_1d, qs_1d, qz_1d,
        sorted_token_ids, sorted_expert_ids, sw_1d,
        num_iters=int(num_iters), num_warmup=int(num_warmup),
    )
    torch.cuda.synchronize()

    # Correctness run into a clean buffer (stage2 atomic-adds).
    out.zero_()
    launch(out, a2_q_flat, w_kernel, a2_scale_1d, scale_w_1d, qs_1d, qz_1d,
           sorted_token_ids, sorted_expert_ids, sw_1d)
    torch.cuda.synchronize()

    if not bool(skip_ref):
        w_ref = w_i8_unshuffled_flat.view(experts, model_dim, inter_dim).to(torch.float32)
        ref2 = torch_moe_gemm2(
            a2_q,                       # [tokens, topk, inter_dim] int8
            w_ref,
            a2_scale,                   # [tokens, topk, 1]
            scale_w_ref,                # [experts, model_dim, 1]
            topk_ids.to(torch.int64),
            topk_weights,
            model_dim=model_dim,
            doweight_stage2=True,
        )
        diff = (out.to(torch.float32) - ref2)
        print(f"[DEBUG a8w4smooth stage2] out shape={out.shape}, ref shape={ref2.shape}")
        print(f"[DEBUG a8w4smooth stage2] out[0,:16]={[f'{v:.4f}' for v in out[0,:16].tolist()]}")
        print(f"[DEBUG a8w4smooth stage2] ref[0,:16]={[f'{v:.4f}' for v in ref2[0,:16].tolist()]}")
        print(f"[DEBUG a8w4smooth stage2] diff[0,:16]={[f'{v:.4f}' for v in diff[0,:16].tolist()]}")
        assert verify_output(out.to(torch.float32), ref2, rtol=0.5, atol=0.5)

    flops = 2 * tokens * topk * model_dim * inter_dim
    tflops = flops / (us / 1e6) / 1e12
    active_experts = min(experts, tokens * topk)
    rows_s2 = model_dim
    bytes_moved = 0
    bytes_moved += tokens * topk * inter_dim * 1  # a2 int8
    bytes_moved += (active_experts * rows_s2 * inter_dim) // 2  # w2 packed int4
    bytes_moved += tokens * model_dim * 2  # out fp16
    bytes_moved += tokens * topk * 4  # scale_x f32
    bytes_moved += active_experts * rows_s2 * 4  # scale_w f32
    bytes_moved += active_experts * (rows_s2 // 16) * (inter_dim // 256) * 16 * 4 * 2  # qscale+qzero i32
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(
        f"FLIR MoE stage2[a8w4smooth]: tokens={tokens} M={model_dim} N={inter_dim} "
        f"E={experts} topk={topk} tile=({tile_m},{tile_n},{tile_k}) "
        f"=> {us:.1f}us ({tflops:.1f} TFLOPS) {tbps:.3f} TB/s"
    )
    return out


# ---------------------------------------------------------------------------
# Self-contained 2-stage test (stage1 -> quant -> stage2)
# ---------------------------------------------------------------------------

def run_moe_2stage_a8w4smooth(
    *,
    tokens: int = 256,
    model_dim: int = 1024,
    inter_dim: int = 256,
    experts: int = 4,
    topk: int = 2,
    tile_m: int = 32,
    tile_n: int = 64,
    tile_k: int = 256,
    seed: int = 0,
    num_iters: int = 3,
    num_warmup: int = 1,
    skip_ref: bool = False,
):
    """End-to-end a8w4smooth: stage1 -> per-token int8 quant -> stage2."""
    # Stage 1
    out1 = run_moe_stage1_a8w4smooth(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        seed=seed,
        num_iters=num_iters,
        num_warmup=num_warmup,
        skip_ref=skip_ref,
    )

    # Intermediate quantization: stage1 fp16 output -> int8 for stage2 input
    out1_fp32 = out1.to(torch.float32)
    a2_q, a2_scale = pertoken_quant(out1_fp32, quant_dtype=torch.int8)

    # Stage 2 — uses its own routing + weights; we just verify it runs and
    # produces correct results with the standalone runner.
    run_moe_stage2_a8w4smooth(
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        seed=seed,
        num_iters=num_iters,
        num_warmup=num_warmup,
        skip_ref=skip_ref,
    )


# ---------------------------------------------------------------------------
# Pytest entry points
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tile_k", [256, 128])
def test_a8w4_stage1(tile_k):
    """Stage1 a8w4smooth standalone."""
    run_moe_stage1_a8w4smooth(
        tokens=256,
        model_dim=1024,
        inter_dim=256,
        experts=4,
        topk=2,
        tile_m=32,
        tile_n=64,
        tile_k=int(tile_k),
        num_iters=3,
        num_warmup=1,
        skip_ref=False,
    )


@pytest.mark.parametrize("tile_k", [256, 128])
def test_a8w4_stage2(tile_k):
    """Stage2 a8w4smooth standalone."""
    run_moe_stage2_a8w4smooth(
        tokens=256,
        model_dim=1024,
        inter_dim=256,
        experts=4,
        topk=2,
        tile_m=32,
        tile_n=64,
        tile_k=int(tile_k),
        num_iters=3,
        num_warmup=1,
        skip_ref=False,
    )


@pytest.mark.parametrize("tile_k", [256])
def test_a8w4_gemm_2stage(tile_k):
    """Combined stage1 -> quant -> stage2 a8w4smooth flow."""
    run_moe_2stage_a8w4smooth(
        tokens=256,
        model_dim=1024,
        inter_dim=256,
        experts=4,
        topk=2,
        tile_m=32,
        tile_n=64,
        tile_k=int(tile_k),
        num_iters=3,
        num_warmup=1,
        skip_ref=False,
    )
