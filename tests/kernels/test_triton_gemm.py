#!/usr/bin/env python3
"""Triton GEMM benchmark for RDNA4 comparison."""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(a, b, block_m=128, block_n=128, block_k=32, group_m=8):
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
    )
    return c


def benchmark(fn, *args, warmup=10, iters=100, label=""):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        fn(*args)
    end_event.record()
    torch.cuda.synchronize()

    ms = start_event.elapsed_time(end_event) / iters
    return ms


if __name__ == "__main__":
    torch.manual_seed(42)

    configs = [
        # (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
        (64, 64, 16, 8),
        (64, 64, 32, 8),
        (128, 128, 16, 8),
        (128, 128, 32, 8),
        (128, 128, 64, 8),
        (128, 256, 32, 8),
        (256, 128, 32, 8),
        (256, 256, 32, 8),
    ]

    for sz in [1024, 2048, 4096]:
        print(f"\n{'=' * 70}")
        print(f"  GEMM {sz}x{sz}x{sz}, bf16 -> f32")
        print(f"{'=' * 70}")

        A = torch.randn(sz, sz, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(sz, sz, device="cuda", dtype=torch.bfloat16)
        flops = 2 * sz**3

        # PyTorch reference
        ms = benchmark(lambda a, b: torch.mm(a, b), A, B, label="pytorch")
        print(
            f"  PyTorch torch.mm:  {ms:.3f} ms  {flops / (ms / 1000) / 1e12:.2f} TFLOPS"
        )

        # Triton configs
        for bm, bn, bk, gm in configs:
            try:
                # Verify correctness first
                C_tri = triton_matmul(
                    A, B, block_m=bm, block_n=bn, block_k=bk, group_m=gm
                )
                C_ref = A.float() @ B.float()
                err = (C_tri - C_ref).abs().max().item() / C_ref.abs().max().item()
                if err > 0.1:
                    print(f"  Triton {bm}x{bn}x{bk} g{gm}: INCORRECT err={err:.2e}")
                    continue

                ms = benchmark(
                    lambda a, b: triton_matmul(
                        a, b, block_m=bm, block_n=bn, block_k=bk, group_m=gm
                    ),
                    A,
                    B,
                )
                tflops = flops / (ms / 1000) / 1e12
                print(
                    f"  Triton {bm:3d}x{bn:3d}x{bk:2d} g{gm}: {ms:.3f} ms  {tflops:.2f} TFLOPS  err={err:.1e}"
                )
            except Exception as e:
                print(f"  Triton {bm:3d}x{bn:3d}x{bk:2d} g{gm}: FAILED - {e}")
