#!/usr/bin/env python3
"""Profile script for rocprofv3 — runs GEMM kernels for hardware counter collection."""

import torch
import sys
import os

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")

MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
SZ = int(sys.argv[2]) if len(sys.argv) > 2 else 4096


def run_pytorch(sz):
    """Run PyTorch (rocBLAS) GEMM."""
    A = torch.randn(sz, sz, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(sz, sz, device="cuda", dtype=torch.bfloat16)
    # warmup
    for _ in range(5):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    # profiled run
    for _ in range(10):
        C = torch.mm(A, B)
    torch.cuda.synchronize()


def run_triton(sz):
    """Run Triton GEMM (128x128x32)."""
    import triton
    import triton.language as tl

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

    A = torch.randn(sz, sz, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(sz, sz, device="cuda", dtype=torch.bfloat16)
    M, K = A.shape
    K, N = B.shape
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    # warmup
    for _ in range(5):
        matmul_kernel[grid](
            A,
            B,
            c,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=128,
            BLOCK_N=128,
            BLOCK_K=32,
            GROUP_M=8,
        )
    torch.cuda.synchronize()
    # profiled run
    for _ in range(10):
        matmul_kernel[grid](
            A,
            B,
            c,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=128,
            BLOCK_N=128,
            BLOCK_K=32,
            GROUP_M=8,
        )
    torch.cuda.synchronize()


def run_flydsl(sz):
    """Run our FlyDSL WMMA GEMM."""
    # Must add repo root to path for kernels module
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from kernels.wmma_gemm import create_wmma_gemm_module

    A = torch.randn(sz, sz, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(sz, sz, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(sz, sz, device="cuda", dtype=torch.float32)

    mod = create_wmma_gemm_module(sz, sz, sz, in_dtype="bf16", out_dtype="f32")
    # warmup
    for _ in range(5):
        mod(A, B, C)
    torch.cuda.synchronize()
    # profiled run
    for _ in range(10):
        mod(A, B, C)
    torch.cuda.synchronize()


if __name__ == "__main__":
    if MODE in ("pytorch", "all"):
        print(f"Running PyTorch GEMM {SZ}x{SZ}...")
        run_pytorch(SZ)
    if MODE in ("triton", "all"):
        print(f"Running Triton GEMM {SZ}x{SZ}...")
        run_triton(SZ)
    if MODE in ("flydsl", "all"):
        print(f"Running FlyDSL GEMM {SZ}x{SZ}...")
        run_flydsl(SZ)
    print("Done.")
