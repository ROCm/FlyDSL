#!/usr/bin/env python3
"""Accuracy + performance test for the GFX1250 fp8 MLA decode kernel
(``kernels.mla_decode_fp8_gfx1250``: Stage1 fp8 WMMA + Stage2 split-LSE merge).

Pipeline tested:
  * Stage1 (``compile_mla_decode_fp8_gfx1250``): paged fp8 KV → ``Mid_O``/``Mid_lse``
    via ``v_wmma_f32_16x16x64_fp8_fp8`` with online softmax.
  * Stage2 (``compile_mla_decode_fp8_gfx1250_stage2``): merges the
    ``num_kv_splits`` partials into final ``O``.

Reference: standard PyTorch MLA attention in fp32.

Skipped automatically on non-GFX12 GPUs (kernel requires ``v_wmma_*_fp8_fp8``).

Usage:
    pytest tests/kernels/test_mla_decode_fp8_gfx1250.py
    python tests/kernels/test_mla_decode_fp8_gfx1250.py            # CLI bench
    python tests/kernels/test_mla_decode_fp8_gfx1250.py -b 1 -c 512
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
from kernels.mla_decode_fp8_gfx1250 import (
    compile_mla_decode_fp8_gfx1250,
    compile_mla_decode_fp8_gfx1250_2stage,
    compile_mla_decode_fp8_gfx1250_stage2,
)

WMMA_FP8_ARCH_PREFIX = "gfx12"

DEFAULT_SEED = 42

DTYPE_QKV = getattr(torch, "float8_e4m3fn", None)
DTYPE_O = torch.float32
DTYPE_LSE = torch.float32
DTYPE_INDEX = torch.int32

# Tolerance vs the fp32 closed-form reference (`_mla_attention_reference`).
# Inputs are randn (N(0,1)) cast to fp8 e4m3 — wider than uniform[-1,1] so the
# logits std≈1, attention is peakier, and a small fraction (~1-2%) of P values
# fall into fp8 subnormal range. With block_n=64 and the per-block fp8(P)
# requantization the kernel does, the worst-row absolute error vs the closed-
# form fp32 reference lands around 0.29 for short sequences (it averages down
# to ~0.12 for kv_len=5120 because more blocks cancel noise). The kernel is
# still internally consistent — `passed_self` (vs CPU stage2 on the same
# Mid_*) stays at 1e-4.
VERIFY_RTOL = 0.30
VERIFY_ATOL = 0.30

def _setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rand_qkv(*shape, device):
    """Gaussian N(0, 1) inputs, cast to fp8 e4m3.

    Compared to uniform[-1,1] (std≈0.58), randn (std=1) gives a wider
    logits distribution and therefore peakier softmax — most P values
    stay within fp8 normal range, but the tails push down toward the
    subnormal threshold (2^-6 ≈ 0.0156), which is where P-scaling
    optimizations like ``P*=448`` actually matter.
    """
    if DTYPE_QKV is None:
        raise RuntimeError("torch.float8_e4m3fn not available")
    x = torch.randn(shape, device=device, dtype=torch.float32)
    return x.to(DTYPE_QKV)


def _create_paged_kv(kv_4d: torch.Tensor, page_block_size: int, *, shuffle=False):
    """[B, Skv, Hkv, D_QK] → paged ``(kv_paged, block_table)``."""
    B, Skv, Hkv, D_QK = kv_4d.shape
    num_blocks = (Skv + page_block_size - 1) // page_block_size
    total_blocks = B * num_blocks
    kv_paged = torch.zeros(
        total_blocks, page_block_size, Hkv, D_QK,
        dtype=kv_4d.dtype, device=kv_4d.device,
    )
    block_table = torch.zeros(
        B, num_blocks, dtype=DTYPE_INDEX, device=kv_4d.device,
    )
    phys_ids = list(range(total_blocks))
    if shuffle:
        random.shuffle(phys_ids)
    idx = 0
    for b in range(B):
        for bl in range(num_blocks):
            phys = phys_ids[idx]
            block_table[b, bl] = phys
            start = bl * page_block_size
            end = min(start + page_block_size, Skv)
            length = end - start
            kv_paged[phys, :length] = kv_4d[b, start:end]
            idx += 1
    return kv_paged, block_table


def _stage2_combine_cpu(mid_o: np.ndarray, mid_lse: np.ndarray, num_kv_splits: int):
    """Reference Stage2: online LSE merge over splits → ``O`` and ``attn_lse``."""
    B, _S, Hq, D_V = mid_o.shape
    O = np.zeros((B, Hq, D_V), dtype=np.float32)
    attn_lse = np.full((B, Hq), -np.inf, dtype=np.float32)
    for b in range(B):
        for h in range(Hq):
            e_max = -math.inf
            e_sum = 0.0
            acc = np.zeros(D_V, dtype=np.float32)
            for s in range(num_kv_splits):
                m = float(mid_lse[b, s, h])
                v = mid_o[b, s, h, :]
                n_max = max(e_max, m)
                scale_old = math.exp(e_max - n_max)
                acc *= scale_old
                scale_new = math.exp(m - n_max)
                acc += scale_new * v
                e_sum = e_sum * scale_old + scale_new
                e_max = n_max
            if e_sum > 0:
                O[b, h, :] = acc / e_sum
                attn_lse[b, h] = e_max + math.log(e_sum)
    return O, attn_lse


def _mla_attention_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    *,
    num_kv_heads: int,
    kv_lora_rank: int,
    sm_scale: float,
):
    """Closed-form MLA (no causal mask, decode Sq=1).

    q : (B, Sq, Hq, D_QK), kv : (B, Skv, Hkv, D_QK).
    Returns out (B, Sq, Hq, D_V) and lse (B, Hq, Sq) — both fp32.
    """
    B, Sq, Hq, D_QK = q.shape
    _, Skv, Hkv, _ = kv.shape
    D_V = kv_lora_rank
    gqa_ratio = Hq // Hkv

    k = kv
    v = kv[..., :D_V]
    k_exp = k.unsqueeze(3).expand(B, Skv, Hkv, gqa_ratio, D_QK).reshape(B, Skv, Hq, D_QK)
    v_exp = v.unsqueeze(3).expand(B, Skv, Hkv, gqa_ratio, D_V).reshape(B, Skv, Hq, D_V)

    q_h = q.transpose(1, 2).to(torch.float32)
    k_h = k_exp.transpose(1, 2).to(torch.float32)
    v_h = v_exp.transpose(1, 2).to(torch.float32)

    logits = torch.einsum("bhsd,bhtd->bhst", q_h, k_h) * sm_scale
    P = torch.softmax(logits, dim=-1)
    lse = torch.logsumexp(logits, dim=-1)
    out = torch.einsum("bhst,bhtv->bhsv", P, v_h).transpose(1, 2)
    return out, lse


def _arch_skip_reason() -> str | None:
    if DTYPE_QKV is None:
        return "torch.float8_e4m3fn not available"
    try:
        arch = str(get_rocm_arch())
    except Exception as e:  # noqa: BLE001
        return f"cannot query ROCm arch: {e}"
    if not arch.startswith(WMMA_FP8_ARCH_PREFIX):
        return f"requires {WMMA_FP8_ARCH_PREFIX}*, current arch={arch!r}"
    return None


def _run_one(
    *,
    batch: int,
    kv_len: int,
    num_q_heads: int = 16,
    num_kv_heads: int = 1,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    page_block_size: int = 64,
    num_kv_splits: int = 1,
    seed: int = DEFAULT_SEED,
    shuffle: bool = False,
    warmup: int = 5,
    iters: int = 20,
):
    """Runs Stage1+Stage2 once, returns ``dict`` with diff metrics + timings."""
    device = "cuda"
    D_QK = kv_lora_rank + qk_rope_head_dim
    D_V = kv_lora_rank
    Sq = 1

    _setup_seed(seed)
    q_4d = _rand_qkv(batch, Sq, num_q_heads, D_QK, device=device)
    kv_4d = _rand_qkv(batch, kv_len, num_kv_heads, D_QK, device=device)
    kv_paged, block_table = _create_paged_kv(
        kv_4d, page_block_size, shuffle=shuffle,
    )
    num_blocks_per_seq = block_table.shape[1]

    Q = q_4d.squeeze(1).contiguous()
    K_Buffer = kv_paged.contiguous()
    V_buffer = kv_paged[..., :D_V].contiguous()

    Mid_O = torch.zeros(
        batch, num_kv_splits, num_q_heads, D_V, dtype=DTYPE_O, device=device,
    )
    Mid_lse = torch.full(
        (batch, num_kv_splits, num_q_heads), float("-inf"),
        dtype=DTYPE_LSE, device=device,
    )
    O = torch.zeros(batch, num_q_heads, D_V, dtype=DTYPE_O, device=device)

    kv_indptr = torch.arange(
        0, (batch + 1) * kv_len, kv_len, dtype=DTYPE_INDEX, device=device,
    )
    stride_b_block_table = num_blocks_per_seq
    stream = torch.cuda.current_stream()

    sm_scale = 1.0 / math.sqrt(D_QK)
    launch_s1, launch_s2 = compile_mla_decode_fp8_gfx1250_2stage(
        batch=batch,
        num_q_heads=num_q_heads,
        num_kv_splits=num_kv_splits,
        seqlen_kv=kv_len,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_h=16,
        block_n=64,
        block_c=kv_lora_rank,
        block_r=qk_rope_head_dim,
        sm_scale=sm_scale,
    )

    def _run_stage1():
        launch_s1(
            Q, K_Buffer, V_buffer, Mid_O, Mid_lse,
            kv_indptr, block_table,
            batch, num_kv_splits, stride_b_block_table, page_block_size,
            stream,
        )

    def _run_stage2():
        launch_s2(Mid_O, Mid_lse, O, stream)

    _run_stage1()
    torch.cuda.synchronize()
    _run_stage2()
    torch.cuda.synchronize()

    ref_out, ref_lse = _mla_attention_reference(
        q_4d, kv_4d,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        sm_scale=sm_scale,
    )
    ref = ref_out.squeeze(1).float().detach().cpu()
    ref_lse_h = ref_lse.squeeze(-1).float().detach().cpu()  # (B, Hq)

    o_got = O.float().detach().cpu()
    o_diff = (o_got - ref).abs()

    # Self-consistency: GPU Stage2 vs CPU Stage2 on the same Mid_*
    o_cpu_np, lse_cpu_np = _stage2_combine_cpu(
        Mid_O.detach().cpu().numpy(),
        Mid_lse.detach().cpu().numpy(),
        num_kv_splits,
    )
    o_cpu_t = torch.from_numpy(o_cpu_np)
    lse_cpu_t = torch.from_numpy(lse_cpu_np)
    o_self_diff = (o_got - o_cpu_t).abs()
    lse_self_diff = (lse_cpu_t - ref_lse_h).abs()

    res = {
        "o_max_diff": o_diff.max().item(),
        "o_mean_diff": o_diff.mean().item(),
        "o_self_max_diff": o_self_diff.max().item(),
        "lse_self_max_diff": lse_self_diff.max().item(),
        "passed_o": bool(torch.allclose(o_got, ref, rtol=VERIFY_RTOL, atol=VERIFY_ATOL)),
        "passed_self": bool(torch.allclose(o_got, o_cpu_t, rtol=1e-4, atol=1e-4)),
    }

    # Timings: separate for Stage1 / Stage2 / total
    def _bench(fn, label: str):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        res[f"time_us_{label}"] = (time.perf_counter() - t0) * 1e6 / iters

    _bench(_run_stage1, "stage1")
    _bench(_run_stage2, "stage2")

    def _run_total():
        _run_stage1()
        _run_stage2()

    _bench(_run_total, "total")

    # Throughput rough estimate (Stage1 dominates).
    total_kv = batch * kv_len
    flops = 2 * num_q_heads * total_kv * (D_QK + D_V) * 2  # Q@K + P@V both 2x
    bytes_in = (
        total_kv * num_kv_heads * D_QK * 1  # fp8 KV
        + batch * num_q_heads * D_QK * 1  # fp8 Q
        + batch * num_q_heads * D_V * 4  # fp32 O
    )
    res["tflops_total"] = flops / res["time_us_total"] / 1e6
    res["tbps_total"] = bytes_in / res["time_us_total"] / 1e6
    return res


# ---------------------------------------------------------------------------
# pytest entries
# ---------------------------------------------------------------------------
_SKIP = _arch_skip_reason()


@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
@pytest.mark.parametrize(
    "batch,kv_len,num_kv_splits",
    [
        (1, 512, 1),
        (1, 1024, 2),
        (2, 512, 1),
        (1, 5120, 5),
    ],
)
def test_mla_decode_fp8_gfx1250(batch, kv_len, num_kv_splits):
    res = _run_one(
        batch=batch,
        kv_len=kv_len,
        num_kv_splits=num_kv_splits,
    )
    print(
        f"[B={batch} Skv={kv_len} splits={num_kv_splits}] "
        f"o_max_diff={res['o_max_diff']:.4f} "
        f"o_mean_diff={res['o_mean_diff']:.4f} "
        f"self_diff={res['o_self_max_diff']:.2e} "
        f"S1={res['time_us_stage1']:.1f}us "
        f"S2={res['time_us_stage2']:.1f}us "
        f"total={res['time_us_total']:.1f}us "
        f"({res['tflops_total']:.1f} TFLOPS, {res['tbps_total']:.2f} TB/s)"
    )
    assert res["passed_self"], (
        f"Stage2 vs CPU-combine self-check failed: max_diff={res['o_self_max_diff']}"
    )
    assert res["passed_o"], (
        f"O vs reference allclose failed: max_diff={res['o_max_diff']}, "
        f"mean_diff={res['o_mean_diff']}"
    )


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------
def _print_skip_and_exit() -> None:
    print("=" * 80)
    print("MLA decode fp8 GFX1250 test — SKIPPED")
    print(f"  reason: {_SKIP}")
    print("=" * 80)
    sys.exit(0)


def main():
    if _SKIP is not None:
        _print_skip_and_exit()

    parser = argparse.ArgumentParser(
        description="GFX1250 fp8 MLA decode (Stage1 + Stage2) accuracy & perf"
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1])
    parser.add_argument("-c", "--kv_len", type=int, nargs="*", default=[512, 1024])
    parser.add_argument("--num_q_heads", type=int, default=16)
    parser.add_argument("--num_kv_heads", type=int, default=1)
    parser.add_argument("--kv_lora_rank", type=int, default=512)
    parser.add_argument("--qk_rope_head_dim", type=int, default=64)
    parser.add_argument("--page_block_size", type=int, default=64)
    parser.add_argument("--num_kv_splits", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    arch = str(get_rocm_arch())
    print("=" * 80)
    print("MLA Decode fp8 GFX1250 Test (Stage1 fp8 WMMA + Stage2 split-LSE merge)")
    print(f"  GPU: {torch.cuda.get_device_name(0)} (arch={arch})")
    print(
        f"  Hq={args.num_q_heads} Hkv={args.num_kv_heads} "
        f"D_lora={args.kv_lora_rank} D_rope={args.qk_rope_head_dim} "
        f"page={args.page_block_size} splits={args.num_kv_splits}"
    )
    print("=" * 80)

    print(f"\n{'Config':<46} | {'Status':>6} | {'S1(us)':>8} | "
          f"{'S2(us)':>7} | {'Total(us)':>9} | TFLOPS")
    print("-" * 100)

    all_passed = True
    for B in args.batch:
        for Skv in args.kv_len:
            tag = (
                f"B={B} Skv={Skv} Hq={args.num_q_heads} "
                f"D={args.kv_lora_rank}+{args.qk_rope_head_dim} "
                f"splits={args.num_kv_splits}"
            )
            try:
                r = _run_one(
                    batch=B,
                    kv_len=Skv,
                    num_q_heads=args.num_q_heads,
                    num_kv_heads=args.num_kv_heads,
                    kv_lora_rank=args.kv_lora_rank,
                    qk_rope_head_dim=args.qk_rope_head_dim,
                    page_block_size=args.page_block_size,
                    num_kv_splits=args.num_kv_splits,
                    seed=args.seed,
                    shuffle=args.shuffle,
                )
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"{tag:<46} | {'ERROR':>6} | {str(e)[:60]}")
                all_passed = False
                continue

            status = "PASS" if (r["passed_o"] and r["passed_self"]) else "FAIL"
            if status != "PASS":
                all_passed = False
                print(
                    f"  diag: o_max_diff={r['o_max_diff']:.4f} "
                    f"self_max_diff={r['o_self_max_diff']:.2e} "
                    f"lse_self={r['lse_self_max_diff']:.4f}"
                )
            print(
                f"{tag:<46} | {status:>6} | "
                f"{r['time_us_stage1']:>8.1f} | "
                f"{r['time_us_stage2']:>7.2f} | "
                f"{r['time_us_total']:>9.1f} | "
                f"{r['tflops_total']:.1f}"
            )

    print("=" * 80)
    if all_passed:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
