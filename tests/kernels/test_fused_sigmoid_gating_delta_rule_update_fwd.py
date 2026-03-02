#!/usr/bin/env python3
"""Test FlyDSL fused sigmoid-gating delta-rule update-forward kernel (MVP)."""

import os
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
_embedded = _repo / "build" / "python_packages" / "rocdsl"
if _embedded.exists():
    os.environ.setdefault("ROCDSL_USE_EMBEDDED_MLIR", "1")
    sys.path.insert(0, str(_embedded))
_src_py = _repo / "python"
if _src_py.exists():
    sys.path.insert(0, str(_src_py))
sys.path.insert(0, str(_repo))

import pytest
import torch

import flydsl
from kernels.fused_sigmoid_gating_delta_rule_update_fwd_kernel import (
    build_fused_sigmoid_gating_delta_rule_update_fwd_module,
)


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)


def torch_ref_sigmoid_gating_update_fwd(
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    state_pool,
    state_indices,
    scale,
    softplus_beta=1.0,
    softplus_threshold=20.0,
):
    """Reference for fused sigmoid-gating + recurrent update forward.

    Shapes:
    - A_log, dt_bias: [HV]
    - a, b: [B,T,HV]
    - q, k: [B,T,H,K]
    - v: [B,T,HV,V]
    - state_pool: [N_STATE,HV,K,V]
    - state_indices: [B]
    """
    B, T_seq, H, K_dim = q.shape
    _, _, HV, V_dim = v.shape

    out = torch.empty(B, T_seq, HV, V_dim, device=q.device, dtype=torch.float32)
    state_after = state_pool.clone().float()

    for n in range(B):
        state_idx = int(state_indices[n].item())
        for hv in range(HV):
            h = state_after[state_idx, hv].clone()  # [K, V]
            for t in range(T_seq):
                q_t = q[n, t, hv % H, :].float() * scale
                k_t = k[n, t, hv % H, :].float()
                v_t = v[n, t, hv, :].float().clone()

                x = a[n, t, hv].float() + dt_bias[hv].float()
                beta_x = softplus_beta * x
                if beta_x <= softplus_threshold:
                    softplus_x = (1.0 / softplus_beta) * torch.log1p(torch.exp(beta_x))
                else:
                    softplus_x = x
                g_t = -torch.exp(A_log[hv].float()) * softplus_x

                beta_t = torch.sigmoid(b[n, t, hv].float())

                h = h * torch.exp(g_t)
                v_t = v_t - (h * k_t[:, None]).sum(dim=0)
                v_t = v_t * beta_t
                h = h + k_t[:, None] * v_t[None, :]
                out[n, t, hv, :] = (h * q_t[:, None]).sum(dim=0)

            state_after[state_idx, hv] = h

    return out, state_after


def test_fused_sigmoid_gating_delta_rule_update_fwd_mvp(ctx):
    B, T_seq, H, HV, K, V = 1, 4, 1, 1, 8, 8
    N_STATE = 2
    torch.manual_seed(42)

    A_log = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1
    a = torch.randn(B, T_seq, HV, device="cuda", dtype=torch.float32) * 0.1
    b = torch.randn(B, T_seq, HV, device="cuda", dtype=torch.float32) * 0.1
    q = torch.randn(B, T_seq, H, K, device="cuda", dtype=torch.float32) * 0.1
    k = torch.randn(B, T_seq, H, K, device="cuda", dtype=torch.float32) * 0.1
    v = torch.randn(B, T_seq, HV, V, device="cuda", dtype=torch.float32) * 0.1

    state_pool = torch.randn(N_STATE, HV, K, V, device="cuda", dtype=torch.float32) * 0.1
    state_indices = torch.tensor([1], device="cuda", dtype=torch.int32)

    scale = K ** (-0.5)
    ref_o, ref_state = torch_ref_sigmoid_gating_update_fwd(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        state_pool=state_pool,
        state_indices=state_indices,
        scale=scale,
    )

    m = build_fused_sigmoid_gating_delta_rule_update_fwd_module(
        B=B,
        T_seq=T_seq,
        H=H,
        HV=HV,
        K=K,
        V=V,
        N_STATE=N_STATE,
        dtype_str="f32",
    )
    exe = flydsl.compile(m)

    a_flat = a.reshape(B * T_seq, HV).contiguous()
    b_flat = b.reshape(B * T_seq, HV).contiguous()
    q_flat = q.reshape(B * T_seq * H, K).contiguous()
    k_flat = k.reshape(B * T_seq * H, K).contiguous()
    v_flat = v.reshape(B * T_seq * HV, V).contiguous()
    state_flat = state_pool.reshape(N_STATE * HV, K, V).contiguous()
    o_flat = torch.empty(B * T_seq * HV, V, device="cuda", dtype=torch.float32)

    exe(A_log, a_flat, dt_bias, q_flat, k_flat, v_flat, b_flat, state_flat, state_indices, o_flat)
    torch.cuda.synchronize()

    got_o = o_flat.reshape(B, T_seq, HV, V)
    got_state = state_flat.reshape(N_STATE, HV, K, V)

    print("got_o:", got_o)
    print("ref_o:", ref_o)
    print("got_state:", got_state)
    print("ref_state:", ref_state)
    print(f"got_o range: [{got_o.min().item():.6e}, {got_o.max().item():.6e}]")
    print(f"ref_o range: [{ref_o.min().item():.6e}, {ref_o.max().item():.6e}]")
    print(f"got_state range: [{got_state.min().item():.6e}, {got_state.max().item():.6e}]")
    print(f"ref_state range: [{ref_state.min().item():.6e}, {ref_state.max().item():.6e}]")

    out_err = (got_o - ref_o).abs().max().item()
    state_err = (got_state - ref_state).abs().max().item()
    print(f"max output error: {out_err:.3e}")
    print(f"max state  error: {state_err:.3e}")
    assert out_err < 1e-4
    assert state_err < 1e-4
