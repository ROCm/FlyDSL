#!/usr/bin/env python3
"""Correctness and smoke perf tests for FlyDSL split-GDR ksplit2 kernel."""

import math
import os
import sys
from pathlib import Path
from typing import Optional

import pytest
import torch
import triton
import triton.language as tl
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

@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    if IS_KDA:
        p_a = a + (bos * HV + i_hv) * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    else:
        p_a = a + bos * HV + i_hv
        p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        # Apply softplus with numerical stability
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        if IS_KDA:
            b_h *= tl.exp(b_g[:, None])
        else:
            b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Update pointers for next timestep
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV

    # Store final state back to h0_source with bounds checking
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_sigmoid_gating_delta_rule_update(
    o: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    grid = (NK, NV, N * HV)

    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state_source is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        IS_KDA=is_kda,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_triton_kernel(out, A_log, dt_bias, q, k, v, a, b, initial_state, indices, scale, use_qk_l2norm_in_kernel):
    fused_sigmoid_gating_delta_rule_update(
        out,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=None,
    )

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
        initial_state_indices=inputs["ssm_state_indices"],
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
    )
    torch.cuda.synchronize()

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
        f"output_triton min/max: {output_triton.float().min().item():.6f} / {output_triton.float().max().item():.6f}"
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
    print(
        f"ssm_state_triton min/max: {ssm_state_triton.float().min().item():.6f} / {ssm_state_triton.float().max().item():.6f}"
    )

    output_diff_hip_ref = (output_ref - output_hip).abs().max().item()
    state_diff_hip_ref = (ssm_state_ref - ssm_state_hip_final).abs().max().item()
    output_diff_fly_ref = (output_ref - output_fly).abs().max().item()
    state_diff_fly_ref = (ssm_state_ref - ssm_state_fly_final).abs().max().item()
    output_diff_fly_hip = (output_hip - output_fly).abs().max().item()
    state_diff_fly_hip = (ssm_state_hip_final - ssm_state_fly_final).abs().max().item()
    output_diff_triton_ref = (output_ref - output_triton).abs().max().item()
    state_diff_triton_ref = (ssm_state_ref - ssm_state_triton).abs().max().item()

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
    print(f"  Output max diff (triton vs ref): {output_diff_triton_ref:.6f}")
    print(f"  State  max diff (triton vs ref): {state_diff_triton_ref:.6f}")
    print(f"{'='*70}")

    assert output_diff_hip_ref < 1e-3, f"Hip output vs ref diff too large: {output_diff_hip_ref}"
    assert state_diff_hip_ref < 1e-3, f"Hip state vs ref diff too large: {state_diff_hip_ref}"
    assert output_diff_fly_ref < 1e-3, f"Fly output vs ref diff too large: {output_diff_fly_ref}"
    assert state_diff_fly_ref < 1e-3, f"Fly state vs ref diff too large: {state_diff_fly_ref}"
    assert output_diff_fly_hip < 1e-3, f"Fly output vs hip diff too large: {output_diff_fly_hip}"
    assert state_diff_fly_hip < 1e-3, f"Fly state vs hip diff too large: {state_diff_fly_hip}"
    assert output_diff_triton_ref < 1e-3, f"Triton output vs ref diff too large: {output_diff_triton_ref}"
    assert state_diff_triton_ref < 1e-3, f"Triton state vs ref diff too large: {state_diff_triton_ref}"

    # ---- Performance check: HIP vs FlyDSL vs Triton ----
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

    def _run_triton_once():
        state_triton = inputs["ssm_state"].clone()
        out_triton_perf = out_template.clone()
        fused_sigmoid_gating_delta_rule_update(
            out_triton_perf,
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
            initial_state_indices=inputs["ssm_state_indices"],
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
        )

    fly_us = _benchmark_us(_run_fly_once)
    hip_us = _benchmark_us(_run_hip_once)
    triton_us = _benchmark_us(_run_triton_once)
    speed_ratio = hip_us / fly_us if fly_us > 0 else float("inf")
    triton_vs_fly = triton_us / fly_us if fly_us > 0 else float("inf")
    hip_vs_triton = hip_us / triton_us if triton_us > 0 else float("inf")
    print(f"  Perf warmup/loop: {warmup}/{num_iters}")
    print(f"  FlyDSL time: {fly_us:.2f} us")
    print(f"  HIP time:    {hip_us:.2f} us")
    print(f"  Triton time: {triton_us:.2f} us")
    print(f"  HIP/FlyDSL:  {speed_ratio:.3f}x")
    print(f"  HIP/Triton:  {hip_vs_triton:.3f}x")
    print(f"  PASS — ksplit2 correctness test passed!")
