#!/usr/bin/env python3
"""Correctness and smoke perf tests for FlyDSL split-GDR ksplit2 kernel."""

import math
import os
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load

_repo = Path(__file__).resolve().parents[2]
_embedded = _repo / "build" / "python_packages" / "rocdsl"
if _embedded.exists():
    os.environ.setdefault("ROCDSL_USE_EMBEDDED_MLIR", "1")
    sys.path.insert(0, str(_embedded))
_src_py = _repo / "python"
if _src_py.exists():
    sys.path.insert(0, str(_src_py))
sys.path.insert(0, str(_repo))

import flydsl

from kernels.fused_split_gdr_update_ksplit2_flyc import (
    build_fused_split_gdr_update_ksplit2_flyc_module,
)


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)


_HIP_SRC = Path("/sgl-workspace/sglang/python/sglang/srt/layers/attention/fla/split_gdr_decode_hip.hip")
_split_gdr_hip = None


def _get_hip_module():
    """JIT-compile and cache HIP split-GDR extension."""
    global _split_gdr_hip
    if _split_gdr_hip is None:
        if not _HIP_SRC.exists():
            pytest.skip(f"HIP source not found: {_HIP_SRC}", allow_module_level=True)
        _split_gdr_hip = load(
            name="split_gdr_hip_flydsl",
            sources=[str(_HIP_SRC)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _split_gdr_hip


def _pack_mixed_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Pack q/k/v to (B, 2*key_dim + value_dim, T)."""
    bsz, t_seq, num_h, k_dim = q.shape
    _, _, hv, v_dim = v.shape
    key_dim = num_h * k_dim
    value_dim = hv * v_dim
    q_pack = q.permute(0, 2, 3, 1).contiguous().reshape(bsz, key_dim, t_seq)
    k_pack = k.permute(0, 2, 3, 1).contiguous().reshape(bsz, key_dim, t_seq)
    v_pack = v.permute(0, 2, 3, 1).contiguous().reshape(bsz, value_dim, t_seq)
    return torch.cat([q_pack, k_pack, v_pack], dim=1).contiguous()


def _swizzle_state(state_pool: torch.Tensor) -> torch.Tensor:
    """(N, HV, K, V) -> (N*HV, K//4, V, 4) float32."""
    n_state, hv, k_dim, v_dim = state_pool.shape
    assert k_dim % 4 == 0
    swz = (
        state_pool.float()
        .reshape(n_state, hv, k_dim // 4, 4, v_dim)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
    )
    return swz.reshape(n_state * hv, k_dim // 4, v_dim, 4).contiguous()


def _unswizzle_state(
    state_swizzled: torch.Tensor, n_state: int, hv: int, k_dim: int, v_dim: int
) -> torch.Tensor:
    """(N*HV, K//4, V, 4) -> (N, HV, K, V) float32."""
    return (
        state_swizzled.reshape(n_state, hv, k_dim // 4, v_dim, 4)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
        .reshape(n_state, hv, k_dim, v_dim)
    )


def split_gdr_reference(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
    use_qk_l2norm_in_kernel: bool,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
):
    """PyTorch reference (float32) for split-GDR recurrent update."""
    bsz, t_seq, h, k_dim = q.shape
    _, _, hv, v_dim = v.shape
    out = torch.empty(bsz, t_seq, hv, v_dim, device=q.device, dtype=torch.float32)
    state_after = state_pool.clone().float()
    hv_per_h = hv // h

    for n in range(bsz):
        state_idx = int(state_indices[n].item())
        for i_hv in range(hv):
            i_h = i_hv // hv_per_h
            h_reg = state_after[state_idx, i_hv].clone()
            for t in range(t_seq):
                q_t = q[n, t, i_h, :].float()
                k_t = k[n, t, i_h, :].float()
                if use_qk_l2norm_in_kernel:
                    q_t = q_t / torch.sqrt((q_t * q_t).sum() + 1e-6)
                    k_t = k_t / torch.sqrt((k_t * k_t).sum() + 1e-6)
                q_scaled = q_t * scale
                v_t = v[n, t, i_hv, :].float().clone()

                x = a[n, t, i_hv].float() + dt_bias[i_hv].float()
                beta_x = softplus_beta * x
                if beta_x <= softplus_threshold:
                    softplus_x = (1.0 / softplus_beta) * torch.log1p(torch.exp(beta_x))
                else:
                    softplus_x = x
                g_t = -torch.exp(A_log[i_hv].float()) * softplus_x
                beta_t = torch.sigmoid(b[n, t, i_hv].float())

                exp_g = torch.exp(g_t)
                v_t = v_t - (h_reg * k_t[:, None]).sum(dim=0) * exp_g
                v_t = v_t * beta_t
                h_reg = h_reg * exp_g + k_t[:, None] * v_t[None, :]
                out[n, t, i_hv, :] = (h_reg * q_scaled[:, None]).sum(dim=0)
            state_after[state_idx, i_hv] = h_reg
    return out, state_after


def _build_inputs(bsz, t_seq, h, hv, k_dim, v_dim, n_state, dtype, seed=0):
    torch.manual_seed(seed)
    A_log = torch.randn(hv, device="cuda", dtype=dtype) * 0.1
    dt_bias = torch.randn(hv, device="cuda", dtype=dtype) * 0.1
    a = torch.randn(bsz, t_seq, hv, device="cuda", dtype=dtype) * 0.1
    b = torch.randn(bsz, t_seq, hv, device="cuda", dtype=dtype) * 0.1
    q = torch.randn(bsz, t_seq, h, k_dim, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(bsz, t_seq, h, k_dim, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(bsz, t_seq, hv, v_dim, device="cuda", dtype=dtype) * 0.1
    state_pool = torch.randn(n_state, hv, k_dim, v_dim, device="cuda", dtype=dtype) * 0.1
    state_indices = torch.arange(bsz, device="cuda", dtype=torch.int32) % n_state
    return A_log, a, dt_bias, q, k, v, b, state_pool, state_indices


def _run_kernel(
    bsz,
    t_seq,
    h,
    hv,
    k_dim,
    v_dim,
    n_state,
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    state_pool,
    state_indices,
    use_qk_l2norm_in_kernel,
    dtype_str,
):
    module = build_fused_split_gdr_update_ksplit2_flyc_module(
        B=bsz,
        T_seq=t_seq,
        H=h,
        HV=hv,
        K=k_dim,
        V=v_dim,
        N_STATE=n_state,
        dtype_str=dtype_str,
        BV=64,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    exe = flydsl.compile(module)
    a_flat = a.reshape(bsz * t_seq, hv).contiguous()
    b_flat = b.reshape(bsz * t_seq, hv).contiguous()
    mixed_qkv = _pack_mixed_qkv(q, k, v)
    state_swz = _swizzle_state(state_pool)
    o = torch.empty(bsz, t_seq, hv, v_dim, device="cuda", dtype=state_pool.dtype)
    exe(A_log, a_flat, dt_bias, mixed_qkv, b_flat, state_swz, state_indices, o)
    torch.cuda.synchronize()
    state_after = _unswizzle_state(state_swz, n_state, hv, k_dim, v_dim)
    return o, state_after


def _run_hip_kernel(
    bsz,
    t_seq,
    h,
    hv,
    k_dim,
    v_dim,
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    state_pool,
    state_indices,
    use_qk_l2norm_in_kernel,
):
    hip_mod = _get_hip_module()
    a_flat = a.reshape(bsz * t_seq, hv).contiguous()
    b_flat = b.reshape(bsz * t_seq, hv).contiguous()
    mixed_qkv = _pack_mixed_qkv(q, k, v)
    state_swz = _swizzle_state(state_pool)

    out = hip_mod.fused_split_gdr_update_ksplit2(
        mixed_qkv=mixed_qkv,
        A_log=A_log.float(),
        a=a_flat,
        dt_bias=dt_bias,
        b_gate=b_flat,
        initial_state_source=state_swz,
        initial_state_indices=state_indices,
        key_dim=h * k_dim,
        value_dim=hv * v_dim,
        num_heads_qk=h,
        num_heads_v=hv,
        head_dim=k_dim,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        scale=1.0 / math.sqrt(k_dim),
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    torch.cuda.synchronize()
    state_after = _unswizzle_state(state_swz, state_pool.shape[0], hv, k_dim, v_dim)
    return out, state_after


@pytest.mark.parametrize(
    "shape_cfg,dtype,use_qk_l2norm_in_kernel",
    [
        ((64, 1, 4, 8, 128, 128, 64), torch.bfloat16, True),
    ],
)
def test_fused_split_gdr_update_ksplit2_flyc_correctness(
    ctx, shape_cfg, dtype, use_qk_l2norm_in_kernel
):
    bsz, t_seq, h, hv, k_dim, v_dim, n_state = shape_cfg
    A_log, a, dt_bias, q, k, v, b, state_pool, state_indices = _build_inputs(
        bsz, t_seq, h, hv, k_dim, v_dim, n_state, dtype
    )

    out_ref, state_ref = split_gdr_reference(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        state_pool=state_pool,
        state_indices=state_indices,
        scale=1.0 / math.sqrt(k_dim),
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    out_ker, state_ker = _run_kernel(
        bsz,
        t_seq,
        h,
        hv,
        k_dim,
        v_dim,
        n_state,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        state_pool,
        state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        dtype_str=("f32" if dtype == torch.float32 else "bf16"),
    )

    atol = 5e-3
    out_diff = (out_ker.float() - out_ref.float()).abs().max().item()
    state_diff = (state_ker.float() - state_ref.float()).abs().max().item()
    assert out_diff < atol
    assert state_diff < atol

    out_hip, state_hip = _run_hip_kernel(
        bsz,
        t_seq,
        h,
        hv,
        k_dim,
        v_dim,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        state_pool,
        state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    out_hip_diff = (out_ker.float() - out_hip.float()).abs().max().item()
    state_hip_diff = (state_ker.float() - state_hip.float()).abs().max().item()
    assert torch.isfinite(out_ker.float()).all()
    assert torch.isfinite(state_ker.float()).all()
    assert torch.isfinite(out_hip.float()).all()
    assert torch.isfinite(state_hip.float()).all()
    assert out_hip_diff < 5e-3
    assert state_hip_diff < 5e-3


@pytest.mark.large_shape
def test_fused_split_gdr_update_ksplit2_flyc_perf_smoke(ctx):
    bsz, t_seq, h, hv, k_dim, v_dim, n_state = (64, 1, 4, 8, 128, 128, 64)
    A_log, a, dt_bias, q, k, v, b, state_pool, state_indices = _build_inputs(
        bsz, t_seq, h, hv, k_dim, v_dim, n_state, torch.bfloat16, seed=11
    )
    a_flat = a.reshape(bsz * t_seq, hv).contiguous()
    b_flat = b.reshape(bsz * t_seq, hv).contiguous()
    mixed_qkv = _pack_mixed_qkv(q, k, v)
    module = build_fused_split_gdr_update_ksplit2_flyc_module(
        B=bsz,
        T_seq=t_seq,
        H=h,
        HV=hv,
        K=k_dim,
        V=v_dim,
        N_STATE=n_state,
        dtype_str="bf16",
        BV=64,
        use_qk_l2norm_in_kernel=True,
    )
    exe = flydsl.compile(module)
    state_swz = _swizzle_state(state_pool)
    o = torch.empty(bsz, t_seq, hv, v_dim, device="cuda", dtype=torch.bfloat16)

    hip_mod = _get_hip_module()
    atol = 5e-3

    # One-shot correctness check against HIP before timing loops.
    state_swz_fly = state_swz.clone()
    out_fly = torch.empty_like(o)
    exe(A_log, a_flat, dt_bias, mixed_qkv, b_flat, state_swz_fly, state_indices, out_fly)
    torch.cuda.synchronize()
    state_after_fly = _unswizzle_state(state_swz_fly, n_state, hv, k_dim, v_dim)

    state_swz_hip = state_swz.clone()
    out_hip = hip_mod.fused_split_gdr_update_ksplit2(
        mixed_qkv=mixed_qkv,
        A_log=A_log.float(),
        a=a_flat,
        dt_bias=dt_bias,
        b_gate=b_flat,
        initial_state_source=state_swz_hip,
        initial_state_indices=state_indices,
        key_dim=h * k_dim,
        value_dim=hv * v_dim,
        num_heads_qk=h,
        num_heads_v=hv,
        head_dim=k_dim,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        scale=1.0 / math.sqrt(k_dim),
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    state_after_hip = _unswizzle_state(state_swz_hip, n_state, hv, k_dim, v_dim)

    out_diff = (out_fly.float() - out_hip.float()).abs().max().item()
    state_diff = (state_after_fly.float() - state_after_hip.float()).abs().max().item()
    assert torch.isfinite(out_fly.float()).all()
    assert torch.isfinite(state_after_fly.float()).all()
    assert torch.isfinite(out_hip.float()).all()
    assert torch.isfinite(state_after_hip.float()).all()
    assert out_diff < atol
    assert state_diff < atol

    num_warmup = 10
    num_iters = 1000
    state_swz_template = state_swz.clone()

    def _benchmark_us(run_fn, state_template):
        state_work = state_template.clone()
        for _ in range(num_warmup):
            run_fn(state_work)
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(num_iters):
            run_fn(state_work)
        end_evt.record()
        torch.cuda.synchronize()
        return start_evt.elapsed_time(end_evt) / num_iters * 1000.0

    flydsl_us = _benchmark_us(
        lambda st: exe(A_log, a_flat, dt_bias, mixed_qkv, b_flat, st, state_indices, o),
        state_swz_template,
    )
    hip_us = _benchmark_us(
        lambda st: hip_mod.fused_split_gdr_update_ksplit2(
            mixed_qkv=mixed_qkv,
            A_log=A_log.float(),
            a=a_flat,
            dt_bias=dt_bias,
            b_gate=b_flat,
            initial_state_source=st,
            initial_state_indices=state_indices,
            key_dim=h * k_dim,
            value_dim=hv * v_dim,
            num_heads_qk=h,
            num_heads_v=hv,
            head_dim=k_dim,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            scale=1.0 / math.sqrt(k_dim),
            use_qk_l2norm_in_kernel=True,
        ),
        state_swz_template,
    )
    speed_ratio = hip_us / flydsl_us
    print(f"perf_us: flydsl={flydsl_us:.3f}, hip={hip_us:.3f}, ratio={speed_ratio:.3f}")

    assert torch.isfinite(o.float()).all()
    assert flydsl_us > 0.0
    assert hip_us > 0.0
    # assert speed_ratio >= 0.8
