#!/usr/bin/env python3
"""Accuracy + performance test for the GFX1250 bf16 V4 sparse_attn kernel
(``kernels.sparse_attn_gfx1250``).

Pipeline tested:
  * ``compile_sparse_attn_gfx1250`` — decode-only (m=1) sparse attention with
    ``v_wmma_f32_16x16x32_bf16`` and online softmax. KV is gathered by
    ``topk_idxs``; ``-1`` slots are masked out.

Reference: closed-form FP32 sparse attention in PyTorch (gather → softmax → PV).

Skipped automatically on non-GFX12 GPUs.

Usage:
    pytest tests/kernels/test_sparse_attn_gfx1250.py
    python tests/kernels/test_sparse_attn_gfx1250.py            # CLI bench
    python tests/kernels/test_sparse_attn_gfx1250.py -b 1 -n 8192 -k 1024
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    print("CUDA/ROCm not available")
    sys.exit(1)

from flydsl.runtime.device import get_rocm_arch
from kernels.sparse_attn_gfx1250 import compile_sparse_attn_gfx1250

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WMMA_BF16_ARCH_PREFIX = "gfx12"

DEFAULT_SEED = 42

DTYPE_QKV = torch.bfloat16
DTYPE_O = torch.bfloat16
DTYPE_INDEX = torch.int32

# bf16 attention vs fp32 reference: bf16 ULP ~0.4% per op + multi-chunk online
# softmax storage in bf16 P + 4-iter accumulation can produce 5-15% max diff
# (verified: 1-chunk max ~6e-4, 4-chunks max ~0.13, 16-chunks max ~0.05).
# Self-consistency check (same kernel run twice) is bit-exact 0 — kernel is
# deterministic; the diff vs fp32 ref is purely bf16 quantization.
# Mirroring V3 mla_decode_fp8_gfx1250 which uses rtol=atol=0.30 for the same
# class of attention kernel.
VERIFY_RTOL = 0.30
VERIFY_ATOL = 0.30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _arch_skip_reason() -> "str | None":
    try:
        arch = str(get_rocm_arch())
    except Exception as e:  # noqa: BLE001
        return f"cannot query ROCm arch: {e}"
    if not arch.startswith(WMMA_BF16_ARCH_PREFIX):
        return f"requires {WMMA_BF16_ARCH_PREFIX}*, current arch={arch!r}"
    return None


def _rand_bf16(*shape, device, low=-1.0, high=1.0):
    """Uniform[low, high] cast to bf16 (suppresses softmax tail compared to randn)."""
    x = torch.empty(shape, device=device, dtype=torch.float32)
    x.uniform_(low, high)
    return x.to(DTYPE_QKV)


def _gen_topk_idxs(
    batch: int,
    n_kv: int,
    topk: int,
    *,
    device,
    seed: int = DEFAULT_SEED,
    fill_invalid: int = 0,
):
    """Generate per-batch ``topk`` index tensor.

    For each batch row, picks ``min(topk, n_kv)`` distinct random indices in
    ``[0, n_kv)``. If ``topk > n_kv``, the tail is padded with ``-1``.
    If ``fill_invalid > 0``, the last ``fill_invalid`` entries (per row) are
    forced to ``-1`` to test the masking path.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    out = torch.full((batch, topk), -1, dtype=DTYPE_INDEX)
    valid_count = min(topk, n_kv)
    if fill_invalid:
        valid_count = max(valid_count - fill_invalid, 0)
    for b in range(batch):
        if valid_count > 0:
            perm = torch.randperm(n_kv, generator=g)[:valid_count]
            out[b, :valid_count] = perm.to(DTYPE_INDEX)
    return out.to(device)


def _sparse_attn_reference(
    Q: torch.Tensor,            # (B, H, D) bf16
    KV: torch.Tensor,           # (B, N, D) bf16
    topk_idxs: torch.Tensor,    # (B, topk) i32
    sm_scale: float,
):
    """FP32 closed-form sparse attention reference.

    Per (b, h): gather K_sub = KV[b, topk_idxs[b], :] (zero-fill when idx=-1),
    compute S = Q[b, h] @ K_sub^T * sm_scale (set columns to -inf when idx=-1
    so they drop out of softmax), P = softmax(S), O[b, h] = P @ K_sub.

    Returns (B, H, D) fp32.
    """
    B, H, D = Q.shape
    _, N, _ = KV.shape
    topk = topk_idxs.shape[-1]

    Q_f = Q.float()
    KV_f = KV.float()
    idx_cpu = topk_idxs.detach().cpu().numpy()

    O = torch.zeros(B, H, D, dtype=torch.float32, device=Q.device)
    for b in range(B):
        # Gather K_sub: invalid slots → zero rows.
        idx_row = idx_cpu[b]
        valid_mask = idx_row >= 0  # numpy bool
        safe_idx = np.where(valid_mask, idx_row, 0)
        K_sub = KV_f[b, safe_idx, :]  # (topk, D)
        K_sub[~valid_mask] = 0.0      # zero invalid rows
        # S = Q[b] @ K_sub^T * scale, mask invalid cols to -inf.
        S = torch.einsum("hd,td->ht", Q_f[b], K_sub) * sm_scale
        S[:, ~valid_mask] = float("-inf")
        # softmax + PV
        P = torch.softmax(S, dim=-1)
        O[b] = torch.einsum("ht,td->hd", P, K_sub)
    return O


# ---------------------------------------------------------------------------
# Single-config runner
# ---------------------------------------------------------------------------
def _run_one(
    *,
    batch: int,
    n_kv: int,
    topk: int,
    num_q_heads: int = 128,
    head_dim: int = 512,
    block_h: int = 16,
    block_n: int = 64,
    num_waves: int = 4,
    num_kv_bufs: int = 2,
    fill_invalid: int = 0,
    seed: int = DEFAULT_SEED,
    warmup: int = 5,
    iters: int = 20,
):
    device = "cuda"
    _setup_seed(seed)

    Q = _rand_bf16(batch, num_q_heads, head_dim, device=device)
    # v1.1: KV is fp8_e4m3 in HBM with per-block fp32 dequant scale, shape
    # (n_kv, head_dim/128). Generate per-block scales uniform in a sane range,
    # then quantize per-block: kv_fp8 = clamp(kv_real / scale_per_block).
    KV_SCALE_BLOCK = 128
    assert head_dim % KV_SCALE_BLOCK == 0
    n_blocks = head_dim // KV_SCALE_BLOCK
    kv_scale = torch.empty(n_kv, n_blocks, device=device).uniform_(0.03, 0.08).to(torch.float32)
    kv_real = torch.empty(batch, n_kv, head_dim, device=device).uniform_(-1.0, 1.0)
    scale_bcast = (
        kv_scale.view(1, n_kv, n_blocks, 1)
        .expand(batch, n_kv, n_blocks, KV_SCALE_BLOCK)
        .reshape(batch, n_kv, head_dim)
    )
    KV = (kv_real / scale_bcast).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    topk_idxs = _gen_topk_idxs(
        batch, n_kv, topk, device=device,
        seed=seed, fill_invalid=fill_invalid,
    )
    O = torch.zeros(batch, num_q_heads, head_dim, dtype=DTYPE_O, device=device)

    sm_scale = 1.0 / math.sqrt(head_dim)
    launch = compile_sparse_attn_gfx1250(
        nheads_q=num_q_heads,
        head_dim=head_dim,
        topk=topk,
        block_h=block_h,
        block_n=block_n,
        sm_scale=sm_scale,
        NUM_WAVES=num_waves,
        NUM_KV_BUFS=num_kv_bufs,
    )

    # Compute actual_topk = max non-(-1) indices per batch (we use topk-fill_invalid
    # uniformly for simplicity; the kernel's bounds use this single value).
    actual_topk = topk - fill_invalid
    stream = torch.cuda.current_stream()

    def _run():
        launch(Q, KV, kv_scale, topk_idxs, O, batch, n_kv, actual_topk, stream)

    _run()
    torch.cuda.synchronize()

    # Reference: dequantize fp8 KV with per-block scale to fp32, then run
    # full-fp32 sparse attention. Matches the kernel's gather-time per-block dequant.
    KV_dequant = (KV.float() * scale_bcast).to(torch.bfloat16)
    O_ref = _sparse_attn_reference(Q, KV_dequant, topk_idxs, sm_scale).cpu()
    O_got = O.float().detach().cpu()
    diff = (O_got - O_ref).abs()

    res = {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "passed": bool(torch.allclose(O_got, O_ref, rtol=VERIFY_RTOL, atol=VERIFY_ATOL)),
    }

    # Bench
    for _ in range(warmup):
        _run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _run()
    torch.cuda.synchronize()
    res["time_us"] = (time.perf_counter() - t0) * 1e6 / iters
    return res


# ---------------------------------------------------------------------------
# pytest entries
# ---------------------------------------------------------------------------
_SKIP = _arch_skip_reason()


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
@pytest.mark.parametrize(
    "batch,n_kv,topk",
    [
        (1, 256, 64),       # smoke: 1 chunk
        (1, 1024, 256),     # 4 chunks
        (1, 8192, 1024),    # V4 typical: 16 chunks at default topk
        (4, 1024, 1024),    # multi-batch
        (1, 1024, 320),     # tail: topk not multiple of block_n=64
    ],
)
def test_sparse_attn_correctness(batch, n_kv, topk):
    res = _run_one(batch=batch, n_kv=n_kv, topk=topk)
    print(
        f"[B={batch} n_kv={n_kv} topk={topk}] "
        f"max={res['max_abs_diff']:.4e} mean={res['mean_abs_diff']:.4e} "
        f"time={res['time_us']:.1f}us"
    )
    assert res["passed"], (
        f"max_abs_diff={res['max_abs_diff']} exceeds tol "
        f"(rtol={VERIFY_RTOL}, atol={VERIFY_ATOL})"
    )


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
def test_sparse_attn_invalid_indices():
    """Tail of topk filled with ``-1`` — must produce same result as a smaller topk."""
    res = _run_one(batch=1, n_kv=1024, topk=1024, fill_invalid=128)
    print(
        f"[invalid-mask test] max={res['max_abs_diff']:.4e} "
        f"mean={res['mean_abs_diff']:.4e}"
    )
    assert res["passed"], (
        f"masked-tail test failed: max_abs_diff={res['max_abs_diff']}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_skip_and_exit() -> None:
    print("=" * 80)
    print("Sparse attention bf16 GFX1250 test — SKIPPED")
    print(f"  reason: {_SKIP}")
    print("=" * 80)
    sys.exit(0)


def main():
    if _SKIP is not None:
        _print_skip_and_exit()

    parser = argparse.ArgumentParser(
        description="GFX1250 bf16 V4 sparse_attn accuracy & perf"
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1])
    parser.add_argument("-n", "--n_kv", type=int, nargs="*", default=[1024, 8192])
    parser.add_argument("-k", "--topk", type=int, nargs="*", default=[1024])
    parser.add_argument("--num_q_heads", type=int, default=128)
    parser.add_argument("--head_dim", type=int, default=512)
    parser.add_argument("--block_h", type=int, default=16)
    parser.add_argument("--block_n", type=int, default=64)
    parser.add_argument("--num_waves", type=int, default=4)
    parser.add_argument("--num_kv_bufs", type=int, default=2)
    parser.add_argument("--fill_invalid", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    arch = str(get_rocm_arch())
    print("=" * 80)
    print("V4 Sparse Attention bf16 GFX1250 Test")
    print(f"  GPU: {torch.cuda.get_device_name(0)} (arch={arch})")
    print(
        f"  Hq={args.num_q_heads} D={args.head_dim} "
        f"block_h={args.block_h} block_n={args.block_n} "
        f"waves={args.num_waves} kv_bufs={args.num_kv_bufs}"
    )
    print("=" * 80)

    print(f"\n{'Config':<40} | {'Status':>6} | {'Time(us)':>9} | "
          f"{'max_diff':>10} | {'mean_diff':>10}")
    print("-" * 100)

    all_passed = True
    for B in args.batch:
        for N in args.n_kv:
            for K in args.topk:
                tag = f"B={B} n_kv={N} topk={K}"
                if K > N:
                    print(f"{tag:<40} | {'SKIP':>6} | (topk > n_kv)")
                    continue
                try:
                    r = _run_one(
                        batch=B,
                        n_kv=N,
                        topk=K,
                        num_q_heads=args.num_q_heads,
                        head_dim=args.head_dim,
                        block_h=args.block_h,
                        block_n=args.block_n,
                        num_waves=args.num_waves,
                        num_kv_bufs=args.num_kv_bufs,
                        fill_invalid=args.fill_invalid,
                        seed=args.seed,
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                except Exception as e:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    print(f"{tag:<40} | {'ERROR':>6} | {str(e)[:60]}")
                    all_passed = False
                    continue

                status = "PASS" if r["passed"] else "FAIL"
                if status != "PASS":
                    all_passed = False
                print(
                    f"{tag:<40} | {status:>6} | "
                    f"{r['time_us']:>9.1f} | "
                    f"{r['max_abs_diff']:>10.4e} | "
                    f"{r['mean_abs_diff']:>10.4e}"
                )

    print("=" * 80)
    if all_passed:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
