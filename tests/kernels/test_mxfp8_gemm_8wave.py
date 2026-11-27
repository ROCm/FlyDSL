#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP8 8-wave GEMM correctness + perf harness.

Kernel implementation: ``kernels/gemm/mxfp8_gemm_8wave.py`` (gfx950 only).

``C[M,N] = A[M,K] @ B[N,K]^T`` with per-1x32 E8M0 block scales on both operands,
bf16 output. A and B are row-major fp8 e4m3; both scales are ``shuffle_scale_w4``
permuted (the kernel reads that layout directly, see its module docstring), or raw
row-major ``[rows, K // 32]`` with ``scale_preshuffled=False``.
"""

import os
import sys

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.fp8_gemm_utils import preshuffle_b  # noqa: E402
from kernels.gemm.mxfp4_preshuffle import launch_splitk_reduce  # noqa: E402
from kernels.gemm.mxfp8_gemm_8wave import compile_mxfp8_gemm_8w  # noqa: E402
from tests.kernels.utils import gemm_common_utils  # noqa: E402
from tests.test_common import run_perftest, verify_output  # noqa: E402

OUT_DTYPE = torch.bfloat16
ARCH = str(get_rocm_arch())

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def _ref_mxfp8(a_q, b_q, a_scale, b_scale):
    """Dequantize fp8 + per-1x32 E8M0 scale, then mm in fp32."""
    a_f32 = a_q.to(torch.float32) * gemm_common_utils.e8m0_to_f32(a_scale).repeat_interleave(32, dim=1)
    b_f32 = b_q.to(torch.float32) * gemm_common_utils.e8m0_to_f32(b_scale).repeat_interleave(32, dim=1)
    return torch.mm(a_f32, b_f32.T)


def _run_mxfp8_gemm(
    M,
    N,
    K,
    *,
    b_preshuffled=False,
    scale_preshuffled=True,
    xcd_swizzle=0,
    out_cols=None,
    num_warmups=10,
    num_iters=100,
    report=False,
):
    if ARCH != "gfx950":
        pytest.skip(f"MXFP8 8-wave GEMM requires gfx950, got {ARCH}")
    assert M % 256 == 0 and N % 256 == 0, "kernel requires M/N aligned to 256"

    device = torch.device("cuda")
    torch.manual_seed(0)
    a_f32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_f32 = torch.randn(N, K, device=device, dtype=torch.float32)

    a_q, a_scale = gemm_common_utils.per_1x32_f8_quant(a_f32)
    b_q, b_scale = gemm_common_utils.per_1x32_f8_quant(b_f32)
    c_ref = _ref_mxfp8(a_q, b_q, a_scale, b_scale)
    # out_cols > N: C is the leading N columns of a wider (M, out_cols) output.
    ldc = N if out_cols is None else out_cols
    c_full = torch.zeros((M, ldc), dtype=OUT_DTYPE, device=device)
    c_out = c_full[:, :N]

    b_in = preshuffle_b(b_q.view(torch.int8)).view(torch.float8_e4m3fn) if b_preshuffled else b_q
    if scale_preshuffled:
        a_scale_in = gemm_common_utils.shuffle_scale_w4(a_scale, 1, False)
        b_scale_in = gemm_common_utils.shuffle_scale_w4(b_scale, 1, False)
    else:
        a_scale_in, b_scale_in = a_scale, b_scale
    launch_fn = compile_mxfp8_gemm_8w(
        K=K, b_preshuffled=b_preshuffled, scale_preshuffled=scale_preshuffled, xcd_swizzle=xcd_swizzle
    )

    def _args(c, a, b, sa, sb):
        return (
            a.contiguous().view(torch.int8).view(-1),
            b.contiguous().view(torch.int8).view(-1),
            c.as_strided(((M - 1) * ldc + N,), (1,)),
            sa.contiguous().view(-1),
            sb.contiguous().view(-1),
            M,
            N,
            ldc,
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(launch_fn, *_args(c_out, a_q, b_in, a_scale_in, b_scale_in))

    def _launch(c, a, b, sa, sb):
        compiled(*_args(c, a, b, sa, sb))

    _, us = run_perftest(_launch, c_out, a_q, b_in, a_scale_in, b_scale_in, num_iters=num_iters, num_warmup=num_warmups)
    torch.cuda.synchronize()

    if report:
        tflops = 2.0 * M * N * K / (us * 1e-6) / 1e12
        tag = f"B={'pre' if b_preshuffled else 'raw'} scale={'pre' if scale_preshuffled else 'raw'}"
        print(f"\n[mxfp8_gemm_8wave] M={M} N={N} K={K} {tag}: {us:.1f} us, {tflops:.1f} TFLOPS")

    assert verify_output(c_out.to(torch.float32), c_ref, rtol=0.05, atol=0.05)
    if ldc > N:
        assert not c_full[:, N:].any(), "store wrote past the column slice"


@pytest.mark.parametrize("scale_preshuffled", [True, False])
@pytest.mark.parametrize("b_preshuffled", [False, True])
@pytest.mark.parametrize(
    "M, N, K, xcd_swizzle, out_cols",
    [
        (256, 256, 512, 0, None),
        (256, 256, 768, 0, None),
        (1024, 1024, 2048, 0, None),
        (1024, 1024, 2048, 4, None),
        (8192, 8192, 512, 4, None),
        (2048, 2048, 4096, 0, None),
        # C as the leading column slice of a wider row-major output (ldc > N); nothing past
        # the slice may be written (the split-K test cannot see that: the reduce overwrites it).
        (512, 512, 512, 0, 768),
        pytest.param(8192, 8192, 8192, 0, None, marks=pytest.mark.large_shape),
        pytest.param(8192, 8192, 8192, 4, None, marks=pytest.mark.large_shape),
    ],
)
def test_mxfp8_gemm_8wave(M, N, K, xcd_swizzle, out_cols, b_preshuffled, scale_preshuffled):
    _run_mxfp8_gemm(
        M,
        N,
        K,
        b_preshuffled=b_preshuffled,
        scale_preshuffled=scale_preshuffled,
        xcd_swizzle=xcd_swizzle,
        out_cols=out_cols,
        report=True,
    )


def _tail_split_n1(M, N, n_cus):
    """Columns of the whole-wave head: the 256x256 tiles past the last whole
    wave over ``n_cus`` CUs are the tail. ``None`` when there is no partial wave."""
    m_tiles, n_tiles = -(-M // 256), N // 256
    head_n = (m_tiles * n_tiles // n_cus) * n_cus // m_tiles
    return None if head_n in (0, n_tiles) else head_n * 256


def _run_mxfp8_tail_split(
    M, N, K, n_split, *, b_preshuffled, n1=None, b_scale_block="1x32", num_warmups=10, num_iters=100, report=False
):
    """Columns ``[0, n1)`` unsplit in place (ldc = N); the rest K-split
    ``n_split`` ways into an fp32 workspace that the strided split-K reduce sums
    into ``Out[:, n1:]``. ``n1=None`` picks the head from the CU count;
    ``n_split=1`` runs every column unsplit (the baseline). Row-major 1x32 A and
    ``b_scale_block`` ("1x32" / "32x32") B scales. Returns the time (us)."""
    if ARCH != "gfx950":
        pytest.skip(f"MXFP8 8-wave GEMM requires gfx950, got {ARCH}")
    device = torch.device("cuda")
    if n_split == 1:
        n1 = N
    elif n1 is None:
        n1 = _tail_split_n1(M, N, torch.cuda.get_device_properties(device).multi_processor_count)
        assert n1 is not None, f"{M}x{N} fills whole waves; nothing to split"
    n2 = N - n1
    torch.manual_seed(0)
    b_rows = 32 if b_scale_block == "32x32" else 1  # N-rows per B scale row
    a_q, a_scale, a_deq = gemm_common_utils.per_block_f8_quant_dequant(torch.randn(M, K, device=device), 1, 32)
    b_q, b_scale, b_deq = gemm_common_utils.per_block_f8_quant_dequant(torch.randn(N, K, device=device), b_rows, 32)
    c_ref = torch.mm(a_deq, b_deq.T)
    b_in = preshuffle_b(b_q.view(torch.int8)) if b_preshuffled else b_q.view(torch.int8)
    b_in = b_in.reshape(N, K)  # preshuffle_b permutes within 16-row groups: rows slice as-is
    a_i8 = a_q.view(torch.int8).view(-1)
    a_sc, b_sc = a_scale.contiguous(), b_scale.contiguous()

    head = compile_mxfp8_gemm_8w(K=K, b_preshuffled=b_preshuffled, scale_preshuffled=False, b_scale_block=b_scale_block)
    tail = compile_mxfp8_gemm_8w(
        K=K, b_preshuffled=b_preshuffled, scale_preshuffled=False, k_split=n_split, b_scale_block=b_scale_block
    )
    c_out = torch.zeros((M, N), dtype=OUT_DTYPE, device=device)
    ws = torch.empty((n_split, M, n2), dtype=torch.float32, device=device)

    stream = torch.cuda.current_stream()
    b_head, b_tail = b_in[:n1].reshape(-1), b_in[n1:].reshape(-1)
    bs_head, bs_tail = b_sc[: n1 // b_rows].reshape(-1), b_sc[n1 // b_rows :].reshape(-1)
    c_head = c_out.as_strided(((M - 1) * N + n1,), (1,))
    calls = []
    if n1:
        calls.append((head, (a_i8, b_head, c_head, a_sc.view(-1), bs_head, M, n1, N, stream)))
    if n2:
        calls.append((tail, (a_i8, b_tail, ws.view(-1), a_sc.view(-1), bs_tail, M, n2, n2, stream)))
    reduce_args = (
        flyc.from_c_void_p(fx.Float32, ws.data_ptr()),
        flyc.from_c_void_p(fx.BFloat16, c_out[:, n1:].data_ptr()),
        M * n2 // 2,  # n_out_dw
        M * n2,  # slab_stride_dw
        n2 // 2,  # row_dw
        N // 2,  # ldc_dw
        stream,
        n_split,
        "bf16",
        n1 > 0,  # strided
    )
    if n2:
        calls.append((launch_splitk_reduce, reduce_args))
    # Compile once; the timed launches reuse the compiled kernels.
    compiled = [(flyc.compile(fn, *args), args) for fn, args in calls]

    def _launch():
        for exe, args in compiled:
            exe(*args)

    _, us = run_perftest(_launch, num_iters=num_iters, num_warmup=num_warmups)
    torch.cuda.synchronize()
    if report:
        tflops = 2.0 * M * N * K / (us * 1e-6) / 1e12
        print(f"\n[mxfp8_gemm_8wave] M={M} N={N} K={K} n1={n1} ns={n_split}: {us:.1f} us, {tflops:.1f} TFLOPS")
    assert verify_output(c_out.to(torch.float32), c_ref, rtol=0.05, atol=0.05)
    return us


@pytest.mark.parametrize(
    "M, N, K, n_split, n1, b_preshuffled, b_scale_block",
    [
        (256, 512, 1024, 2, 0, True, "1x32"),  # every tile split, contiguous reduce
        (300, 768, 2048, 4, 256, True, "1x32"),  # partial M tile, explicit head
        (300, 768, 2048, 4, 256, False, "1x32"),  # row-major B: its own K-segment offset
        (4096, 5120, 1024, 2, None, True, "1x32"),  # head from the CU count
        (300, 768, 2048, 4, 256, True, "32x32"),  # 32x32 B: head / tail slice its scale rows
    ],
)
def test_mxfp8_gemm_8wave_split_k(M, N, K, n_split, n1, b_preshuffled, b_scale_block):
    _run_mxfp8_tail_split(
        M, N, K, n_split, b_preshuffled=b_preshuffled, n1=n1, b_scale_block=b_scale_block, report=True
    )


@pytest.mark.large_shape
@pytest.mark.parametrize(
    "M, N, K, n_split",
    [
        # An aiter DSv4.1 row tuned to the split tail (..._ns4, gfx950): 3 M tiles x 100
        # N tiles overrun 256 CUs by 44 tiles, which the tail spreads over 4 K-segments.
        (672, 25600, 6144, 4),
    ],
)
def test_mxfp8_gemm_8wave_split_k_speedup(M, N, K, n_split):
    """Unsplit vs the split tail on shapes aiter tunes to it: prints both times."""
    base = _run_mxfp8_tail_split(M, N, K, 1, b_preshuffled=True)
    split = _run_mxfp8_tail_split(M, N, K, n_split, b_preshuffled=True)
    print(
        f"\n[mxfp8_gemm_8wave] M={M} N={N} K={K}: unsplit {base:.1f} us, "
        f"split tail ns{n_split} {split:.1f} us ({base / split:.2f}x)"
    )


@pytest.mark.parametrize("b_preshuffled", [False, True])
@pytest.mark.parametrize(
    "M, N, K, k_split, a_block, b_block",
    [
        (300, 512, 1024, 1, "1x32", "32x32"),
        (300, 512, 1024, 1, "1x128", "1x32"),
        (300, 512, 1024, 1, "1x32", "128x128"),
        (300, 512, 1024, 1, "1x128", "128x128"),
        # A split-K segment takes row-major 1x32 A and 1x32 / 32x32 B scales.
        (256, 512, 2048, 2, "1x32", "32x32"),
    ],
)
def test_mxfp8_gemm_8wave_coarse_scales(M, N, K, k_split, a_block, b_block, b_preshuffled):
    """Unshuffled coarse scales: a column-major 1x128 A, a 32x32 or 128x128 B."""
    if ARCH != "gfx950":
        pytest.skip(f"MXFP8 8-wave GEMM requires gfx950, got {ARCH}")
    device = torch.device("cuda")
    torch.manual_seed(0)
    blocks = {"1x32": (1, 32), "1x128": (1, 128), "32x32": (32, 32), "128x128": (128, 128)}
    a_q, sa, a_deq = gemm_common_utils.per_block_f8_quant_dequant(torch.randn(M, K, device=device), *blocks[a_block])
    if a_block == "1x128":
        sa = sa.T.contiguous()  # the column-major [K // 128, M] bytes
    b_q, sb, b_deq = gemm_common_utils.per_block_f8_quant_dequant(torch.randn(N, K, device=device), *blocks[b_block])
    c_ref = torch.mm(a_deq, b_deq.T)
    b_in = preshuffle_b(b_q.view(torch.int8)) if b_preshuffled else b_q.view(torch.int8)
    launch = compile_mxfp8_gemm_8w(
        K=K,
        b_preshuffled=b_preshuffled,
        scale_preshuffled=False,
        k_split=k_split,
        a_scale_block=a_block,
        b_scale_block=b_block,
    )
    c_out = torch.zeros((M, N), dtype=OUT_DTYPE, device=device)
    out = c_out if k_split == 1 else torch.empty((k_split, M, N), dtype=torch.float32, device=device)
    args = (a_q.view(torch.int8).view(-1), b_in.reshape(-1), out.view(-1), sa.view(-1), sb.view(-1), M, N, N)
    flyc.compile(launch, *args, torch.cuda.current_stream())
    torch.cuda.synchronize()
    if k_split > 1:
        c_out = out.sum(0).to(OUT_DTYPE)
    assert verify_output(c_out.to(torch.float32), c_ref, rtol=0.05, atol=0.05)
