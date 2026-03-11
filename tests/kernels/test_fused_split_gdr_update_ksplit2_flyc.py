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
from kernels.split_gdr_triton import fused_sigmoid_gating_delta_rule_update


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
    """Test correctness of ksplit2 kernel against torch CPU reference."""
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
    use_qk_l2norm_in_kernel = True

    common_scalar_args = {
        "key_dim": key_dim,
        "value_dim": value_dim,
        "num_heads_qk": num_heads_qk,
        "num_heads_v": num_heads_v,
        "head_dim": head_dim,
        "softplus_beta": softplus_beta,
        "softplus_threshold": softplus_threshold,
        "scale": scale,
        "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
    }
    common_tensor_args = {
        "mixed_qkv": inputs["mixed_qkv"],
        "A_log": inputs["A_log"],
        "a": inputs["a"],
        "dt_bias": inputs["dt_bias"],
        "b_gate": inputs["b"],
        "initial_state_indices": inputs["ssm_state_indices"],
    }

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
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    # ---- ksplit2 kernel under test ----
    hip_mod = _get_hip_module()
    ssm_state_hip = inputs["ssm_state"].clone()
    ssm_state_swizzled = to_swizzled_layout(ssm_state_hip)

    output_hip = hip_mod.fused_split_gdr_update_ksplit2(
        **common_tensor_args,
        initial_state_source=ssm_state_swizzled,
        **common_scalar_args,
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
        use_qk_l2norm_in_kernel=common_scalar_args["use_qk_l2norm_in_kernel"],
    )
    fly_exe = flydsl.compile(fly_module)
    ssm_state_fly = inputs["ssm_state"].clone()
    ssm_state_swizzled_fly = to_swizzled_layout(ssm_state_fly)
    output_fly = torch.empty_like(output_hip)
    fly_exe(
        common_tensor_args["mixed_qkv"],
        common_tensor_args["A_log"],
        common_tensor_args["a"],
        common_tensor_args["dt_bias"],
        common_tensor_args["b_gate"],
        ssm_state_swizzled_fly,
        common_tensor_args["initial_state_indices"],
        output_fly,
    )
    torch.cuda.synchronize()
    ssm_state_fly_final = from_swizzled_layout(ssm_state_swizzled_fly)

    # ---- Triton fused_sigmoid_gating_delta_rule_update precision check ----
    q_triton = (
        inputs["mixed_qkv"][:, :key_dim, :]
        .permute(0, 2, 1)
        .reshape(batch_size, seqlen, num_heads_qk, head_dim)
        .contiguous()
    )
    k_triton = (
        inputs["mixed_qkv"][:, key_dim:2 * key_dim, :]
        .permute(0, 2, 1)
        .reshape(batch_size, seqlen, num_heads_qk, head_dim)
        .contiguous()
    )
    v_triton = (
        inputs["mixed_qkv"][:, 2 * key_dim:, :]
        .permute(0, 2, 1)
        .reshape(batch_size, seqlen, num_heads_v, head_dim)
        .contiguous()
    )
    a_triton = inputs["a"].reshape(batch_size, seqlen, num_heads_v).contiguous()
    b_triton = inputs["b"].reshape(batch_size, seqlen, num_heads_v).contiguous()
    ssm_state_triton = inputs["ssm_state"].clone()
    output_triton = torch.empty_like(output_hip)
    fused_sigmoid_gating_delta_rule_update(
        output_triton,
        A_log=inputs["A_log"],
        a=a_triton,
        dt_bias=inputs["dt_bias"],
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q_triton,
        k=k_triton,
        v=v_triton,
        b=b_triton,
        initial_state_source=ssm_state_triton,
        initial_state_indices=common_tensor_args["initial_state_indices"],
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=None,
    )
    torch.cuda.synchronize()

    stats_tensors = {
        "output_ref": output_ref,
        "output_hip": output_hip,
        "output_fly": output_fly,
        "output_triton": output_triton,
        "ssm_state_ref": ssm_state_ref,
        "ssm_state_hip_final": ssm_state_hip_final,
        "ssm_state_fly_final": ssm_state_fly_final,
        "ssm_state_triton": ssm_state_triton,
    }
    for name, tensor in stats_tensors.items():
        tensor_f32 = tensor.float()
        print(f"{name} min/max: {tensor_f32.min().item():.6f} / {tensor_f32.max().item():.6f}")

    diff_checks = [
        ("Output max diff (hip vs ref)", output_ref, output_hip, "Hip output vs ref"),
        ("State  max diff (hip vs ref)", ssm_state_ref, ssm_state_hip_final, "Hip state vs ref"),
        ("Output max diff (fly vs ref)", output_ref, output_fly, "Fly output vs ref"),
        ("State  max diff (fly vs ref)", ssm_state_ref, ssm_state_fly_final, "Fly state vs ref"),
        ("Output max diff (fly vs hip)", output_hip, output_fly, "Fly output vs hip"),
        ("State  max diff (fly vs hip)", ssm_state_hip_final, ssm_state_fly_final, "Fly state vs hip"),
        ("Output max diff (triton vs ref)", output_ref, output_triton, "Triton output vs ref"),
        ("State  max diff (triton vs ref)", ssm_state_ref, ssm_state_triton, "Triton state vs ref"),
    ]
    diff_results = {}
    for display_name, lhs, rhs, err_name in diff_checks:
        diff = (lhs - rhs).abs().max().item()
        diff_results[display_name] = (diff, err_name)

    print(f"\n{'='*70}")
    print(f"Split GDR ksplit2 Correctness: batch={batch_size}, seqlen={seqlen}")
    print(f"  heads_qk={num_heads_qk}, heads_v={num_heads_v}, head_dim={head_dim}")
    print(f"{'='*70}")
    for display_name, _lhs, _rhs, _err_name in diff_checks:
        print(f"  {display_name}: {diff_results[display_name][0]:.6f}")
    print(f"{'='*70}")

    for display_name, _lhs, _rhs, _err_name in diff_checks:
        diff, err_name = diff_results[display_name]
        assert diff < 1e-3, f"{err_name} diff too large: {diff}"

    # ---- Performance check: HIP vs FlyDSL vs Triton ----
    warmup = 10
    num_iters = 1000
    state_swz_template = to_swizzled_layout(inputs["ssm_state"])
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

    def _run_once(backend: str):
        if backend == "fly":
            state_swz = state_swz_template.clone()
            out = out_template.clone()
            fly_exe(
                common_tensor_args["mixed_qkv"],
                common_tensor_args["A_log"],
                common_tensor_args["a"],
                common_tensor_args["dt_bias"],
                common_tensor_args["b_gate"],
                state_swz,
                common_tensor_args["initial_state_indices"],
                out,
            )
            return

        if backend == "hip":
            state_swz = state_swz_template.clone()
            _ = hip_mod.fused_split_gdr_update_ksplit2(
                **common_tensor_args,
                initial_state_source=state_swz,
                **common_scalar_args,
            )
            return

        if backend == "triton":
            state_triton = inputs["ssm_state"].clone()
            out = out_template.clone()
            fused_sigmoid_gating_delta_rule_update(
                out,
                A_log=inputs["A_log"],
                a=a_triton,
                dt_bias=inputs["dt_bias"],
                softplus_beta=softplus_beta,
                softplus_threshold=softplus_threshold,
                q=q_triton,
                k=k_triton,
                v=v_triton,
                b=b_triton,
                initial_state_source=state_triton,
                initial_state_indices=common_tensor_args["initial_state_indices"],
                scale=scale,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=None,
            )
            return

        raise ValueError(f"Unknown backend: {backend}")

    fly_us = _benchmark_us(lambda: _run_once("fly"))
    hip_us = _benchmark_us(lambda: _run_once("hip"))
    triton_us = _benchmark_us(lambda: _run_once("triton"))
    speed_ratio = hip_us / fly_us if fly_us > 0 else float("inf")
    hip_vs_triton = hip_us / triton_us if triton_us > 0 else float("inf")
    print(f"  Perf warmup/loop: {warmup}/{num_iters}")
    print(f"  FlyDSL time: {fly_us:.2f} us")
    print(f"  HIP time:    {hip_us:.2f} us")
    print(f"  Triton time: {triton_us:.2f} us")
    print(f"  HIP/FlyDSL:  {speed_ratio:.3f}x")
    print(f"  HIP/Triton:  {hip_vs_triton:.3f}x")
    print(f"  PASS — ksplit2 correctness test passed!")
