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


# ---------------------------------------------------------------------------
# State layout conversion utilities
# ---------------------------------------------------------------------------

def to_swizzled_layout(state: torch.Tensor) -> torch.Tensor:
    """
    Convert state from standard (N, HV, K, V) to swizzled (N, HV, K/4, V, 4) layout.
    
    Swizzled layout enables float4 vectorized loads with cross-thread coalescing:
    - Each thread loads 4 consecutive K values as float4
    - All 64 threads access consecutive addresses (1024 bytes per load)
    
    Memory layout transformation:
        Standard:  h[n, hv, k, v] at address n*HV*K*V + hv*K*V + k*V + v
        Swizzled:  h[n, hv, kg, v, k4] at address n*HV*KG*V*4 + hv*KG*V*4 + kg*V*4 + v*4 + k4
                   where kg = k // 4, k4 = k % 4
    """
    N, HV, K, V = state.shape
    assert K % 4 == 0, f"K ({K}) must be divisible by 4 for swizzled layout"
    
    # Reshape: (N, HV, K, V) -> (N, HV, K/4, 4, V)
    state = state.reshape(N, HV, K // 4, 4, V)
    # Permute: (N, HV, K/4, 4, V) -> (N, HV, K/4, V, 4)
    state = state.permute(0, 1, 2, 4, 3)
    # Make contiguous
    return state.contiguous()


def from_swizzled_layout(state: torch.Tensor) -> torch.Tensor:
    """
    Convert state from swizzled (N, HV, K/4, V, 4) to standard (N, HV, K, V) layout.
    """
    N, HV, K4, V, four = state.shape
    assert four == 4, f"Last dimension must be 4, got {four}"
    K = K4 * 4
    
    # Permute: (N, HV, K/4, V, 4) -> (N, HV, K/4, 4, V)
    state = state.permute(0, 1, 2, 4, 3)
    # Reshape: (N, HV, K/4, 4, V) -> (N, HV, K, V)
    state = state.reshape(N, HV, K, V)
    # Make contiguous
    return state.contiguous()


def to_vsplit_layout(state: torch.Tensor) -> torch.Tensor:
    """
    Convert state from standard (N, HV, K, V) to vsplit (N, HV, V/4, K, 4) layout.
    
    Vsplit layout groups 4 contiguous V values together:
    - Each float4 = 4 V values for one K position
    - Optimized for the vsplit kernel's [8,8] thread layout
    """
    N, HV, K, V = state.shape
    assert V % 4 == 0, f"V ({V}) must be divisible by 4 for vsplit layout"
    return state.reshape(N, HV, K, V // 4, 4).permute(0, 1, 3, 2, 4).contiguous()


def from_vsplit_layout(state: torch.Tensor) -> torch.Tensor:
    """
    Convert state from vsplit (N, HV, V/4, K, 4) to standard (N, HV, K, V) layout.
    """
    N, HV, V4, K, four = state.shape
    assert four == 4, f"Last dimension must be 4, got {four}"
    V = V4 * 4
    return state.permute(0, 1, 3, 2, 4).reshape(N, HV, K, V).contiguous()


# ---------------------------------------------------------------------------
# Pure PyTorch CPU reference implementation
# ---------------------------------------------------------------------------

def split_gdr_reference(
    mixed_qkv: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    key_dim: int,
    value_dim: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim: int,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    scale: float = None,
    use_qk_l2norm_in_kernel: bool = True,
):
    """
    Pure PyTorch reference for fused_split_gdr_update_kernel_v3.

    Mirrors the Triton kernel logic step-by-step on CPU in float32.

    Args:
        mixed_qkv: (B, dim, T), bfloat16 — concatenated Q, K, V along dim axis.
                   Q region: [0, key_dim), K region: [key_dim, 2*key_dim),
                   V region: [2*key_dim, 2*key_dim + value_dim).
        A_log:     (HV,), float32
        a:         (B*T, HV), bfloat16 — time-variant gating parameter
        dt_bias:   (HV,), bfloat16
        b:         (B*T, HV), bfloat16 — beta gating parameter
        initial_state_source: (N_states, HV, K, V), float32
        initial_state_indices: (B,), int32
        key_dim:   total key dimension = num_heads_qk * head_dim
        value_dim: total value dimension = num_heads_v * head_dim
        num_heads_qk: number of QK heads (H)
        num_heads_v:  number of V heads  (HV)
        head_dim:     per-head dimension (K = V = head_dim)

    Returns:
        output: (B, T, HV, V), same dtype as mixed_qkv
    """
    B, dim, T = mixed_qkv.shape
    H = num_heads_qk
    HV = num_heads_v
    K = head_dim
    V = head_dim
    GROUP_SIZE = HV // H

    if scale is None:
        scale = K ** -0.5

    # Work in float32 on CPU
    mixed_qkv_f = mixed_qkv.float().cpu()
    A_log_f = A_log.float().cpu()
    dt_bias_f = dt_bias.float().cpu()
    a_f = a.float().cpu()          # (B*T, HV)
    b_f = b.float().cpu()          # (B*T, HV)

    # Reshape a, b to (B, T, HV)
    a_f = a_f.view(B, T, HV)
    b_f = b_f.view(B, T, HV)

    # Clone initial states — indexed by initial_state_indices
    # h: (B, HV, K, V) in float32
    h = torch.zeros(B, HV, K, V, dtype=torch.float32)
    indices = initial_state_indices.cpu()
    for n in range(B):
        idx = indices[n].item()
        if idx >= 0:
            h[n] = initial_state_source[idx].float().cpu()

    # Split mixed_qkv along dim axis
    # Q: (B, key_dim, T), K_tensor: (B, key_dim, T), V_tensor: (B, value_dim, T)
    Q_all = mixed_qkv_f[:, :key_dim, :]             # (B, key_dim, T)
    K_all = mixed_qkv_f[:, key_dim:2*key_dim, :]    # (B, key_dim, T)
    V_all = mixed_qkv_f[:, 2*key_dim:, :]           # (B, value_dim, T)

    output = torch.zeros(B, T, HV, V, dtype=torch.float32)

    for t in range(T):
        for hv in range(HV):
            i_h = hv // GROUP_SIZE  # corresponding QK head

            # Extract per-head Q, K, V for this timestep
            q_vec = Q_all[:, i_h * K:(i_h + 1) * K, t]   # (B, K)
            k_vec = K_all[:, i_h * K:(i_h + 1) * K, t]   # (B, K)
            v_vec = V_all[:, hv * V:(hv + 1) * V, t]      # (B, V)

            # Gating parameters for this timestep and head
            a_t = a_f[:, t, hv]    # (B,)
            b_t = b_f[:, t, hv]    # (B,)

            # g = -exp(A_log[hv]) * softplus(a_t + dt_bias[hv])
            x = a_t + dt_bias_f[hv]                       # (B,)
            beta_x = softplus_beta * x
            softplus_x = torch.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * torch.log(1.0 + torch.exp(beta_x)),
                x,
            )
            g = -torch.exp(A_log_f[hv]) * softplus_x      # (B,)

            # beta = sigmoid(b_t)
            beta = torch.sigmoid(b_t)                      # (B,)

            # L2 normalization
            if use_qk_l2norm_in_kernel:
                q_vec = q_vec / (torch.sqrt(torch.sum(q_vec * q_vec, dim=-1, keepdim=True) + 1e-6))
                k_vec = k_vec / (torch.sqrt(torch.sum(k_vec * k_vec, dim=-1, keepdim=True) + 1e-6))

            # Scale query
            q_vec = q_vec * scale                          # (B, K)

            # h *= exp(g)  — decay
            h[:, hv, :, :] *= torch.exp(g).unsqueeze(-1).unsqueeze(-1)  # (B, K, V)

            # v -= sum(h * k[:, :, None], dim=K)  — delta rule
            v_vec = v_vec - torch.einsum('bkv,bk->bv', h[:, hv, :, :], k_vec)  # (B, V)

            # v *= beta  — beta gating
            v_vec = v_vec * beta.unsqueeze(-1)             # (B, V)

            # h += k[:, :, None] * v[:, None, :]  — state update
            h[:, hv, :, :] += torch.einsum('bk,bv->bkv', k_vec, v_vec)  # (B, K, V)

            # o = sum(h * q[:, :, None], dim=K)  — output
            o_vec = torch.einsum('bkv,bk->bv', h[:, hv, :, :], q_vec)  # (B, V)
            output[:, t, hv, :] = o_vec

    # Write final state back to initial_state_source (in-place, on CPU copy)
    for n in range(B):
        idx = indices[n].item()
        if idx >= 0:
            initial_state_source[idx] = h[n].to(initial_state_source.dtype).to(initial_state_source.device)

    return output.to(mixed_qkv.dtype).to(mixed_qkv.device)


def _build_inputs(bsz, t_seq, h, hv, k_dim, v_dim, n_state, dtype, seed=0):
    torch.manual_seed(seed)
    A_log = torch.randn(hv, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(hv, device="cuda", dtype=dtype) * 0.1
    a = torch.randn(bsz, t_seq, hv, device="cuda", dtype=dtype) * 0.1
    b = torch.randn(bsz, t_seq, hv, device="cuda", dtype=dtype) * 0.1
    q = torch.randn(bsz, t_seq, h, k_dim, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(bsz, t_seq, h, k_dim, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(bsz, t_seq, hv, v_dim, device="cuda", dtype=dtype) * 0.1
    state_pool = torch.randn(n_state, hv, k_dim, v_dim, device="cuda", dtype=dtype) * 0.1
    state_indices = torch.arange(bsz, device="cuda", dtype=torch.int32) % n_state
    return A_log, a, dt_bias, q, k, v, b, state_pool, state_indices

def create_inputs(
    batch_size: int,
    seqlen: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim: int,
    device: str,
    dtype: torch.dtype,
):
    """Create test inputs (identical to TestFusedSplitGDRUpdateOpt.create_inputs)."""
    key_dim = num_heads_qk * head_dim
    value_dim = num_heads_v * head_dim
    dim = 2 * key_dim + value_dim

    # mixed_qkv: (batch, dim, seqlen)
    mixed_qkv = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)

    # Gating parameters — A_log must be float32
    A_log = torch.randn(num_heads_v, device=device, dtype=torch.float32)
    dt_bias = torch.randn(num_heads_v, device=device, dtype=dtype)

    # Time-variant gating: (batch * seqlen, num_heads_v)
    a = torch.randn(batch_size * seqlen, num_heads_v, device=device, dtype=dtype)
    b = torch.randn(batch_size * seqlen, num_heads_v, device=device, dtype=dtype)

    # SSM state must be float32
    # Shape: (batch + padding, num_heads_v, head_dim, head_dim)
    ssm_state = torch.randn(
        batch_size + 10, num_heads_v, head_dim, head_dim,
        device=device, dtype=torch.float32,
    )
    ssm_state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)

    return {
        "mixed_qkv": mixed_qkv,
        "A_log": A_log,
        "a": a,
        "dt_bias": dt_bias,
        "b": b,
        "ssm_state": ssm_state,
        "ssm_state_indices": ssm_state_indices,
        "key_dim": key_dim,
        "value_dim": value_dim,
    }

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


@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("seqlen", [1])
@pytest.mark.parametrize("num_heads_qk", [4])
@pytest.mark.parametrize("num_heads_v", [8])
@pytest.mark.parametrize("head_dim", [128])
def test_split_gdr_ksplit2_correctness_and_perf(
    ctx,
    batch_size,
    seqlen,
    num_heads_qk,
    num_heads_v,
    head_dim,
):
    """Test correctness of ksplit2 HIP kernel against torch CPU reference."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    inputs = create_inputs(
        batch_size, seqlen, num_heads_qk, num_heads_v, head_dim, device, dtype,
    )

    key_dim = inputs["key_dim"]
    value_dim = inputs["value_dim"]

    softplus_beta = 1.0
    softplus_threshold = 20.0
    scale = head_dim ** -0.5

    # ---- Reference: pure PyTorch CPU ----
    ssm_state_ref = inputs["ssm_state"].clone()
    output_ref = split_gdr_reference(
        mixed_qkv=inputs["mixed_qkv"],
        A_log=inputs["A_log"],
        a=inputs["a"],
        dt_bias=inputs["dt_bias"],
        b=inputs["b"],
        initial_state_source=ssm_state_ref,
        initial_state_indices=inputs["ssm_state_indices"],
        key_dim=key_dim,
        value_dim=value_dim,
        num_heads_qk=num_heads_qk,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )

    # ---- ksplit2 kernel under test ----
    hip_mod = _get_hip_module()
    ssm_state_hip = inputs["ssm_state"].clone()
    ssm_state_swizzled = to_swizzled_layout(ssm_state_hip)

    output_hip = hip_mod.fused_split_gdr_update_ksplit2(
        mixed_qkv=inputs["mixed_qkv"],
        A_log=inputs["A_log"],
        a=inputs["a"],
        dt_bias=inputs["dt_bias"],
        b_gate=inputs["b"],
        initial_state_source=ssm_state_swizzled,
        initial_state_indices=inputs["ssm_state_indices"],
        key_dim=key_dim,
        value_dim=value_dim,
        num_heads_qk=num_heads_qk,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )

    ssm_state_hip_final = from_swizzled_layout(ssm_state_swizzled)

    # ---- FlyDSL build module precision check ----
    fly_module = build_fused_split_gdr_update_ksplit2_flyc_module(
        B=batch_size,
        T_seq=seqlen,
        H=num_heads_qk,
        HV=num_heads_v,
        K=head_dim,
        V=head_dim,
        N_STATE=inputs["ssm_state"].shape[0],
        dtype_str=("f32" if dtype == torch.float32 else "bf16"),
        BV=64,
        use_qk_l2norm_in_kernel=True,
    )
    fly_exe = flydsl.compile(fly_module)
    ssm_state_fly = inputs["ssm_state"].clone()
    ssm_state_swizzled_fly = to_swizzled_layout(ssm_state_fly).reshape(
        ssm_state_fly.shape[0] * num_heads_v, head_dim // 4, head_dim, 4
    )
    output_fly = torch.empty_like(output_hip)
    fly_exe(
        inputs["A_log"].float(),
        inputs["a"],
        inputs["dt_bias"],
        inputs["mixed_qkv"],
        inputs["b"],
        ssm_state_swizzled_fly,
        inputs["ssm_state_indices"],
        output_fly,
    )
    torch.cuda.synchronize()
    ssm_state_fly_final = from_swizzled_layout(
        ssm_state_swizzled_fly.reshape(
            ssm_state_fly.shape[0], num_heads_v, head_dim // 4, head_dim, 4
        )
    )

    # print("output_ref[:128]:\n", output_ref.reshape(-1)[:128])
    # print("output_hip[:128]:\n", output_hip.reshape(-1)[:128])
    # print("output_fly[:128]:\n", output_fly.reshape(-1)[:128])
    # print("ssm_state_ref[:128]:\n", ssm_state_ref.reshape(-1)[:128])
    # print("ssm_state_hip_final[:128]:\n", ssm_state_hip_final.reshape(-1)[:128])
    # print("ssm_state_fly_final[:128]:\n", ssm_state_fly_final.reshape(-1)[:128])
    print(
        f"output_ref min/max: {output_ref.float().min().item():.6f} / {output_ref.float().max().item():.6f}"
    )
    print(
        f"output_hip min/max: {output_hip.float().min().item():.6f} / {output_hip.float().max().item():.6f}"
    )
    print(
        f"output_fly min/max: {output_fly.float().min().item():.6f} / {output_fly.float().max().item():.6f}"
    )
    print(
        f"ssm_state_ref min/max: {ssm_state_ref.float().min().item():.6f} / {ssm_state_ref.float().max().item():.6f}"
    )
    print(
        f"ssm_state_hip_final min/max: {ssm_state_hip_final.float().min().item():.6f} / {ssm_state_hip_final.float().max().item():.6f}"
    )
    print(
        f"ssm_state_fly_final min/max: {ssm_state_fly_final.float().min().item():.6f} / {ssm_state_fly_final.float().max().item():.6f}"
    )

    output_diff_hip_ref = (output_ref - output_hip).abs().max().item()
    state_diff_hip_ref = (ssm_state_ref - ssm_state_hip_final).abs().max().item()
    output_diff_fly_ref = (output_ref - output_fly).abs().max().item()
    state_diff_fly_ref = (ssm_state_ref - ssm_state_fly_final).abs().max().item()
    output_diff_fly_hip = (output_hip - output_fly).abs().max().item()
    state_diff_fly_hip = (ssm_state_hip_final - ssm_state_fly_final).abs().max().item()

    print(f"\n{'='*70}")
    print(f"Split GDR ksplit2 Correctness: batch={batch_size}, seqlen={seqlen}")
    print(f"  heads_qk={num_heads_qk}, heads_v={num_heads_v}, head_dim={head_dim}")
    print(f"{'='*70}")
    print(f"  Output max diff (hip vs ref): {output_diff_hip_ref:.6f}")
    print(f"  State  max diff (hip vs ref): {state_diff_hip_ref:.6f}")
    print(f"  Output max diff (fly vs ref): {output_diff_fly_ref:.6f}")
    print(f"  State  max diff (fly vs ref): {state_diff_fly_ref:.6f}")
    print(f"  Output max diff (fly vs hip): {output_diff_fly_hip:.6f}")
    print(f"  State  max diff (fly vs hip): {state_diff_fly_hip:.6f}")
    print(f"{'='*70}")

    assert output_diff_hip_ref < 1e-3, f"Hip output vs ref diff too large: {output_diff_hip_ref}"
    assert state_diff_hip_ref < 1e-3, f"Hip state vs ref diff too large: {state_diff_hip_ref}"
    assert output_diff_fly_ref < 1e-3, f"Fly output vs ref diff too large: {output_diff_fly_ref}"
    assert state_diff_fly_ref < 1e-3, f"Fly state vs ref diff too large: {state_diff_fly_ref}"
    assert output_diff_fly_hip < 1e-3, f"Fly output vs hip diff too large: {output_diff_fly_hip}"
    assert state_diff_fly_hip < 1e-3, f"Fly state vs hip diff too large: {state_diff_fly_hip}"

    # ---- Performance check: HIP vs FlyDSL ----
    warmup = 10
    num_iters = 1000
    state_swz_template = to_swizzled_layout(inputs["ssm_state"]).reshape(
        inputs["ssm_state"].shape[0] * num_heads_v, head_dim // 4, head_dim, 4
    )
    out_template = torch.empty_like(output_hip)

    def _benchmark_us(run_fn):
        for _ in range(warmup):
            run_fn()
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(num_iters):
            run_fn()
        end_evt.record()
        torch.cuda.synchronize()
        return (start_evt.elapsed_time(end_evt) * 1000.0) / num_iters

    def _run_fly_once():
        state_swz = state_swz_template.clone()
        out_fly_perf = out_template.clone()
        fly_exe(
            inputs["A_log"].float(),
            inputs["a"],
            inputs["dt_bias"],
            inputs["mixed_qkv"],
            inputs["b"],
            state_swz,
            inputs["ssm_state_indices"],
            out_fly_perf,
        )

    def _run_hip_once():
        state_swz = state_swz_template.clone()
        _ = hip_mod.fused_split_gdr_update_ksplit2(
            mixed_qkv=inputs["mixed_qkv"],
            A_log=inputs["A_log"],
            a=inputs["a"],
            dt_bias=inputs["dt_bias"],
            b_gate=inputs["b"],
            initial_state_source=state_swz,
            initial_state_indices=inputs["ssm_state_indices"],
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_dim=head_dim,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )

    fly_us = _benchmark_us(_run_fly_once)
    hip_us = _benchmark_us(_run_hip_once)
    speed_ratio = hip_us / fly_us if fly_us > 0 else float("inf")
    print(f"  Perf warmup/loop: {warmup}/{num_iters}")
    print(f"  FlyDSL time: {fly_us:.2f} us")
    print(f"  HIP time:    {hip_us:.2f} us")
    print(f"  HIP/FlyDSL:  {speed_ratio:.3f}x")
    print(f"  PASS — ksplit2 correctness test passed!")


@pytest.mark.large_shape
def test_fused_split_gdr_update_ksplit2_flyc_perf_smoke(ctx):
    bsz, t_seq, h, hv, k_dim, v_dim, n_state = (64, 1, 2, 4, 128, 128, 64)
    # bsz, t_seq, h, hv, k_dim, v_dim, n_state = (64, 1, 4, 8, 128, 128, 64)
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
