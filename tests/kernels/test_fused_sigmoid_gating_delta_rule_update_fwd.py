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
    use_qk_l2norm_in_kernel=False,
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
            h_idx = hv // (HV // H)
            h = state_after[state_idx, hv].clone()  # [K, V]
            for t in range(T_seq):
                q_t = q[n, t, h_idx, :].float()
                k_t = k[n, t, h_idx, :].float()
                if use_qk_l2norm_in_kernel:
                    q_t = q_t / torch.sqrt((q_t * q_t).sum() + 1e-6)
                    k_t = k_t / torch.sqrt((k_t * k_t).sum() + 1e-6)
                q_t = q_t * scale
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


def _build_inputs(B, T_seq, H, HV, K, V, N_STATE, state_indices, seed=42):
    torch.manual_seed(seed)
    A_log = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1
    a = torch.randn(B, T_seq, HV, device="cuda", dtype=torch.float32) * 0.1
    b = torch.randn(B, T_seq, HV, device="cuda", dtype=torch.float32) * 0.1
    q = torch.randn(B, T_seq, H, K, device="cuda", dtype=torch.float32) * 0.1
    k = torch.randn(B, T_seq, H, K, device="cuda", dtype=torch.float32) * 0.1
    v = torch.randn(B, T_seq, HV, V, device="cuda", dtype=torch.float32) * 0.1
    state_pool = torch.randn(N_STATE, HV, K, V, device="cuda", dtype=torch.float32) * 0.1
    state_indices = torch.tensor(state_indices, device="cuda", dtype=torch.int32)
    return A_log, a, dt_bias, q, k, v, b, state_pool, state_indices


def _run_kernel(
    B,
    T_seq,
    H,
    HV,
    K,
    V,
    N_STATE,
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    state_pool,
    state_indices,
    use_qk_l2norm_in_kernel=False,
    disable_state_update=False,
    disable_output_calculation=False,
):
    m = build_fused_sigmoid_gating_delta_rule_update_fwd_module(
        B=B,
        T_seq=T_seq,
        H=H,
        HV=HV,
        K=K,
        V=V,
        N_STATE=N_STATE,
        dtype_str="f32",
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        disable_state_update=disable_state_update,
        disable_output_calculation=disable_output_calculation,
    )
    exe = flydsl.compile(m)
    a_flat = a.reshape(B * T_seq, HV).contiguous()
    b_flat = b.reshape(B * T_seq, HV).contiguous()
    q_flat = q.reshape(B * T_seq * H, K).contiguous()
    k_flat = k.reshape(B * T_seq * H, K).contiguous()
    v_flat = v.reshape(B * T_seq * HV, V).contiguous()
    state_flat = state_pool.reshape(N_STATE * HV, K, V).contiguous()
    if disable_output_calculation:
        o_flat = torch.full(
            (B * T_seq * HV, V), float("nan"), device="cuda", dtype=torch.float32
        )
    else:
        o_flat = torch.empty(B * T_seq * HV, V, device="cuda", dtype=torch.float32)

    exe(A_log, a_flat, dt_bias, q_flat, k_flat, v_flat, b_flat, state_flat, state_indices, o_flat)
    torch.cuda.synchronize()
    return o_flat.reshape(B, T_seq, HV, V), state_flat.reshape(N_STATE, HV, K, V)


def _print_minmax(name, x):
    x_f = x.float()
    finite_mask = torch.isfinite(x_f)
    if finite_mask.any():
        v = x_f[finite_mask]
        print(f"{name} min={v.min().item():.3e}, max={v.max().item():.3e}")
    else:
        print(f"{name} min=nan, max=nan (all non-finite)")


@pytest.mark.parametrize(
    "B,T_seq,H,HV,K,V,N_STATE,state_indices,use_qk_l2norm_in_kernel",
    [
        (1, 4, 1, 1, 8, 8, 2, [1], False),  # MVP baseline
        (1, 4, 2, 4, 8, 8, 3, [2], False),  # HV > H mapping
        (2, 3, 2, 2, 8, 8, 4, [3, 1], False),  # multi-batch + state-pool indirection
        (1, 4, 1, 1, 8, 8, 2, [1], True),  # qk l2norm
    ],
)
def test_fused_sigmoid_gating_delta_rule_update_fwd_core(
    ctx, B, T_seq, H, HV, K, V, N_STATE, state_indices, use_qk_l2norm_in_kernel
):
    A_log, a, dt_bias, q, k, v, b, state_pool, state_indices = _build_inputs(
        B, T_seq, H, HV, K, V, N_STATE, state_indices
    )
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
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    got_o, got_state = _run_kernel(
        B=B,
        T_seq=T_seq,
        H=H,
        HV=HV,
        K=K,
        V=V,
        N_STATE=N_STATE,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        state_pool=state_pool,
        state_indices=state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    _print_minmax("got_o", got_o)
    _print_minmax("ref_o", ref_o)
    _print_minmax("got_state", got_state)
    _print_minmax("ref_state", ref_state)
    out_err = (got_o - ref_o).abs().max().item()
    state_err = (got_state - ref_state).abs().max().item()
    print(f"shape(B,T,H,HV,K,V)=({B},{T_seq},{H},{HV},{K},{V}), qk_l2={use_qk_l2norm_in_kernel}")
    print(f"max output error: {out_err:.3e}")
    print(f"max state  error: {state_err:.3e}")
    assert out_err < 1e-4
    assert state_err < 1e-4


def test_fused_sigmoid_gating_delta_rule_update_fwd_disable_state_update(ctx):
    B, T_seq, H, HV, K, V, N_STATE = 1, 4, 1, 1, 8, 8, 2
    A_log, a, dt_bias, q, k, v, b, state_pool, state_indices = _build_inputs(
        B, T_seq, H, HV, K, V, N_STATE, [1], seed=123
    )
    scale = K ** (-0.5)
    ref_o, _ = torch_ref_sigmoid_gating_update_fwd(
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
    ref_state = state_pool
    got_o, got_state = _run_kernel(
        B=B,
        T_seq=T_seq,
        H=H,
        HV=HV,
        K=K,
        V=V,
        N_STATE=N_STATE,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        state_pool=state_pool.clone(),
        state_indices=state_indices,
        disable_state_update=True,
    )
    _print_minmax("got_o", got_o)
    _print_minmax("ref_o", ref_o)
    _print_minmax("got_state", got_state)
    _print_minmax("ref_state", ref_state)
    assert got_o.isfinite().all()
    assert torch.allclose(got_state, ref_state, atol=0.0, rtol=0.0)


def test_fused_sigmoid_gating_delta_rule_update_fwd_disable_output_calculation(ctx):
    B, T_seq, H, HV, K, V, N_STATE = 1, 4, 1, 1, 8, 8, 2
    A_log, a, dt_bias, q, k, v, b, state_pool, state_indices = _build_inputs(
        B, T_seq, H, HV, K, V, N_STATE, [1], seed=456
    )
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
    got_o, got_state = _run_kernel(
        B=B,
        T_seq=T_seq,
        H=H,
        HV=HV,
        K=K,
        V=V,
        N_STATE=N_STATE,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        state_pool=state_pool,
        state_indices=state_indices,
        disable_output_calculation=True,
    )
    _print_minmax("got_o", got_o)
    _print_minmax("ref_o", ref_o)
    _print_minmax("got_state", got_state)
    _print_minmax("ref_state", ref_state)
    assert torch.isnan(got_o).all()
    state_err = (got_state - ref_state).abs().max().item()
    print(f"max state error (disable output): {state_err:.3e}")
    assert state_err < 1e-4


def test_fused_sigmoid_gating_delta_rule_update_fwd_softplus_threshold_branch(ctx):
    """Cover softplus threshold branch with extreme positive inputs."""
    B, T_seq, H, HV, K, V, N_STATE = 1, 4, 1, 1, 8, 8, 2
    A_log, a, dt_bias, q, k, v, b, state_pool, state_indices = _build_inputs(
        B, T_seq, H, HV, K, V, N_STATE, [1], seed=789
    )
    # Force beta*x >> threshold to trigger the linear branch in softplus.
    a.fill_(35.0)
    dt_bias.fill_(0.0)

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
    got_o, got_state = _run_kernel(
        B=B,
        T_seq=T_seq,
        H=H,
        HV=HV,
        K=K,
        V=V,
        N_STATE=N_STATE,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        state_pool=state_pool,
        state_indices=state_indices,
    )
    _print_minmax("got_o", got_o)
    _print_minmax("ref_o", ref_o)
    _print_minmax("got_state", got_state)
    _print_minmax("ref_state", ref_state)
    out_err = (got_o - ref_o).abs().max().item()
    state_err = (got_state - ref_state).abs().max().item()
    print(f"max output error (softplus threshold): {out_err:.3e}")
    print(f"max state  error (softplus threshold): {state_err:.3e}")
    assert got_o.isfinite().all()
    assert got_state.isfinite().all()
    assert out_err < 1e-4
    assert state_err < 1e-4
