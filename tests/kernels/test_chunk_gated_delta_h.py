"""
Tests for FlyDSL K5: chunk_gated_delta_rule_fwd_h (GDN hidden-state recurrence)

Correctness: compare FlyDSL kernel against a pure-PyTorch reference.
Performance: compare FlyDSL kernel against Triton opt3 kernel.

Runtime parameters derived from Qwen3.5-397B-A17B TP=8 serving config:
  K=128, V=128, Hk=16->Hg=2, Hv=64->H=8, BT=64
  max_num_batched_tokens=8192, full_prompt_len=8000

Usage:
  cd /workspace/FlyDSL
  python3 -m pytest tests/kernels/test_chunk_gated_delta_h.py -v -s
  python3 -m pytest tests/kernels/test_chunk_gated_delta_h.py -v -s -k "Correct"
  python3 -m pytest tests/kernels/test_chunk_gated_delta_h.py -v -s -k "Perf"
"""

import sys
import os
import pytest
import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kernels.chunk_gated_delta_h import chunk_gated_delta_rule_fwd_h_flydsl

# Also import Triton reference for performance comparison
TRITON_AVAILABLE = False
try:
    sys.path.insert(0, "/workspace/linear_attn_example")
    from kernel.triton.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h_opt3 as fwd_h_triton_opt3,
    )
    TRITON_AVAILABLE = True
except ImportError:
    pass


# ── Global test configuration ──────────────────────────────────────────

K = 128
V = 128
Hg = 2
H = 8
BT = 64

MAX_NUM_BATCHED_TOKENS = 8192
FULL_PROMPT_LENS = [1000, 8000]

NUM_WARMUP = 10
NUM_ITERS = 200


def _build_context_lens(full_prompt_len, max_tokens=MAX_NUM_BATCHED_TOKENS):
    context_lens = []
    remaining = max_tokens
    while remaining > 0:
        cur = min(full_prompt_len, remaining)
        context_lens.append(cur)
        remaining -= cur
    return context_lens


def _build_cu_seqlens(context_lens, device="cuda"):
    scheduled_q_lens = context_lens
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(scheduled_q_lens), 0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    return scheduled_q_lens, cu_seqlens


def _make_inputs(context_lens, dtype=torch.bfloat16, device="cuda",
                 with_initial_state=True):
    scheduled_q_lens, cu_seqlens = _build_cu_seqlens(context_lens, device=device)
    T_total = int(cu_seqlens[-1].item())
    N = len(scheduled_q_lens)
    B = 1

    k = torch.randn(B, T_total, Hg, K, dtype=dtype, device=device) * 0.1
    w_orig = torch.randn(B, T_total, H, K, dtype=dtype, device=device) * 0.1
    u_orig = torch.randn(B, T_total, H, V, dtype=dtype, device=device) * 0.1
    g = torch.randn(T_total, H, dtype=torch.float32, device=device).abs() * -0.5
    g = g.cumsum(dim=0)

    w_c = w_orig.permute(0, 2, 1, 3).contiguous()
    u_c = u_orig.permute(0, 2, 1, 3).contiguous()

    initial_state = None
    if with_initial_state:
        initial_state = torch.randn(N, H, K, V, dtype=torch.float32, device=device) * 0.01

    return k, w_orig, u_orig, w_c, u_c, g, initial_state, cu_seqlens, scheduled_q_lens


# ── Pure-PyTorch reference ──────────────────────────────────────────────

def ref_chunk_gated_delta_rule_fwd_h(
    k, w, u, g,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    cu_seqlens=None,
):
    """Reference in FP32 for correctness checking."""
    B, T, Hg_dim, K_dim = k.shape
    H_dim, V_dim = u.shape[-2], u.shape[-1]
    BT_dim = chunk_size
    if cu_seqlens is None:
        NT = triton.cdiv(T, BT_dim)
    else:
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        NT = sum(triton.cdiv(int(seq_len), BT_dim) for seq_len in seq_lens)
    gqa_ratio = H_dim // Hg_dim

    h_out = k.new_zeros(B, NT, H_dim, K_dim, V_dim, dtype=torch.float32)
    v_new_out = torch.zeros_like(u, dtype=torch.float32)

    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    final_state = torch.zeros(N, H_dim, K_dim, V_dim, dtype=torch.float32,
                              device=k.device) if output_final_state else None

    for b_idx in range(B):
        if cu_seqlens is not None:
            seqs = [(s, cu_seqlens[s].item(), cu_seqlens[s + 1].item())
                    for s in range(N)]
        else:
            seqs = [(b_idx, 0, T)]

        chunk_offset = 0
        for seq_idx, bos, eos in seqs:
            seq_len = eos - bos
            seq_nt = triton.cdiv(seq_len, BT_dim)

            for i_h in range(H_dim):
                i_hg = i_h // gqa_ratio
                h_state = torch.zeros(K_dim, V_dim, dtype=torch.float32,
                                      device=k.device)
                if initial_state is not None:
                    h_state = initial_state[seq_idx, i_h].float().clone()

                for i_t in range(seq_nt):
                    t_start = i_t * BT_dim
                    t_end = min(t_start + BT_dim, seq_len)
                    actual_bt = t_end - t_start

                    h_out[b_idx, chunk_offset + i_t, i_h] = h_state.clone()

                    w_chunk = w[b_idx, bos + t_start:bos + t_end, i_h].float()
                    u_chunk = u[b_idx, bos + t_start:bos + t_end, i_h].float()
                    b_v = u_chunk - w_chunk @ h_state
                    v_new_out[b_idx, bos + t_start:bos + t_end, i_h] = b_v

                    last_idx = bos + t_end - 1
                    g_last = g[last_idx, i_h].float()
                    g_chunk = g[bos + t_start:bos + t_end, i_h].float()

                    mask = torch.zeros(BT_dim, device=k.device)
                    mask[:actual_bt] = 1.0
                    gate = torch.where(
                        mask[:actual_bt].bool(),
                        torch.exp(g_last - g_chunk),
                        torch.zeros_like(g_chunk),
                    )
                    b_v_gated = b_v * gate.unsqueeze(-1)

                    h_state = h_state * torch.exp(g_last)
                    k_chunk = k[b_idx, bos + t_start:bos + t_end, i_hg].float()
                    b_v_gated_cast = b_v_gated.to(k.dtype).float()
                    h_state = h_state + k_chunk.T @ b_v_gated_cast

                if output_final_state:
                    final_state[seq_idx, i_h] = h_state

            chunk_offset += seq_nt

    return h_out, v_new_out.to(u.dtype), final_state


def _normalize_opt_v_new(vn_opt):
    """Convert opt v_new layout [B, H, T, V] back to [B, T, H, V]."""
    return vn_opt.permute(0, 2, 1, 3).contiguous()


# ── Correctness tests ───────────────────────────────────────────────────

class TestCorrectness:
    """Correctness against PyTorch reference."""

    @pytest.mark.parametrize("full_prompt_len", [1000])
    def test_correctness_flydsl(self, full_prompt_len):
        context_lens = _build_context_lens(full_prompt_len)
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(context_lens)

        h_fly, vn_fly, fs_fly = chunk_gated_delta_rule_fwd_h_flydsl(
            k, w_c, u_c, g=g, initial_state=h0,
            output_final_state=True, cu_seqlens=cu, wu_contiguous=True,
        )
        h_ref, vn_ref, fs_ref = ref_chunk_gated_delta_rule_fwd_h(
            k, w_orig, u_orig, g=g, initial_state=h0,
            output_final_state=True, cu_seqlens=cu,
        )

        torch.testing.assert_close(
            h_fly.float(), h_ref.float(), atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(
            _normalize_opt_v_new(vn_fly).float(), vn_ref.float(),
            atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(
            fs_fly.float(), fs_ref.float(), atol=1e-1, rtol=1e-1)


# ── Performance tests ───────────────────────────────────────────────────

def _bench_fn(fn, *args, **kwargs):
    """Warmup + measure, return average us."""
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    for _ in range(NUM_WARMUP):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(NUM_ITERS):
        fn(*args, **kwargs)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / NUM_ITERS * 1000


PERF_SHAPES = [
    pytest.param(fpl, id=f"full{fpl}")
    for fpl in FULL_PROMPT_LENS
]


class TestPerformance:
    """Performance comparison: FlyDSL vs Triton opt3."""

    @pytest.mark.parametrize("full_prompt_len", PERF_SHAPES)
    def test_perf_comparison(self, full_prompt_len):
        context_lens = _build_context_lens(full_prompt_len)
        k, w_orig, u_orig, w_c, u_c, g, h0, cu, scheduled_q_lens = _make_inputs(
            context_lens)
        total_tokens = int(cu[-1].item())

        # FlyDSL kernel
        us_fly = _bench_fn(
            chunk_gated_delta_rule_fwd_h_flydsl,
            k, w_c, u_c, g=g, initial_state=h0,
            output_final_state=True, cu_seqlens=cu, wu_contiguous=True,
        )

        print(f"\n[K5 FlyDSL T={total_tokens}]  {us_fly:.2f} us")

        # Triton opt3 kernel for comparison
        if TRITON_AVAILABLE:
            us_triton = _bench_fn(
                fwd_h_triton_opt3,
                k, w_c, u_c, g=g, initial_state=h0,
                output_final_state=True, cu_seqlens=cu, wu_contiguous=True,
            )
            speedup = us_triton / us_fly if us_fly > 0 else float('inf')
            print(f"[K5 Triton opt3 T={total_tokens}]  {us_triton:.2f} us")
            print(f"[Speedup FlyDSL/Triton]  {speedup:.3f}x")


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
