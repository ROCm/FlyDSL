#!/usr/bin/env python3
"""Split-K HGEMM using WMMA + TDM tests for gfx1250.

Kernel implementation lives in `kernels/hgemm_splitk_gfx1250.py`.
This file is the correctness harness.
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLIR_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLIR_SRC not in sys.path:
    sys.path.insert(0, _PYFLIR_SRC)

# workaround for simulator
import flydsl  # noqa: E402,F401

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

from flydsl.runtime.device import get_rocm_arch
from kernels.hgemm_splitk_gfx1250 import compile_hgemm_splitk_gfx1250, hgemm_splitk_gfx1250_
from tests.test_common import run_perftest, verify_output


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


SPLIT_K_COUNTER_MAX_LEN = 512


def _validate_pipeline_depth(*, ks, tile_k, num_buffers):
    load_tile_k = 2 * tile_k  # kernel loads 2*tile_k per stage
    num_k_tiles = ks // load_tile_k
    if num_k_tiles < num_buffers:
        pytest.skip(
            f"{num_buffers}-buffer requires num_k_tiles >= {num_buffers}, got {num_k_tiles}"
        )


@pytest.mark.parametrize("in_dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, SPLIT_K",
    [
        # Non split-K (SPLIT_K=1)
        (256, 256, 256, 64, 256, 128, 1),
        (512, 512, 512, 128, 256, 128, 1),
        (1024, 1024, 1024, 256, 256, 128, 1),
        (201, 179, 128, 64, 128, 64, 1),
        # Split-K (SPLIT_K > 1)
        (32, 256, 1024, 64, 128, 128, 4),
        (64, 256, 2048, 64, 128, 128, 8),
        (256, 256, 1024, 64, 256, 128, 4),
        (128, 128, 512, 64, 128, 128, 2),
    ],
)
@pytest.mark.parametrize("num_buffers", [2, 3])
def test_hgemm_splitk_gfx1250(
    in_dtype, M, N, K, tile_m, tile_n, tile_k, SPLIT_K, num_buffers,
    m_warp=2, n_warp=4, l2_prefetch_distance=2,
    out_dtype=None, use_tdm_store=True, inst_prefetch=False,
):
    """Split-K HGEMM correctness test for gfx1250."""
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"WMMA requires gfx1250, got {arch}")

    assert K % SPLIT_K == 0, f"K={K} must be divisible by SPLIT_K={SPLIT_K}"
    ks = K // SPLIT_K

    _validate_pipeline_depth(ks=ks, tile_k=tile_k, num_buffers=num_buffers)

    # Check LDS budget (kernel loads 2*tile_k per stage)
    load_tile_k = 2 * tile_k
    lds_pad = 8
    elem_bytes = 2
    a_buf = tile_m * (load_tile_k + lds_pad) * elem_bytes
    b_buf = load_tile_k * (tile_n + lds_pad) * elem_bytes
    total_lds = (a_buf + b_buf) * num_buffers
    if total_lds > 327680:
        pytest.skip(f"LDS budget exceeded: {total_lds} > 327680")

    # Split-K forces buffer_store epilogue
    if SPLIT_K > 1:
        use_tdm_store = False

    _eff_out = out_dtype or ("f16" if in_dtype == "fp16" else "bf16")
    _out_torch = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}[_eff_out]
    torch_dtype = torch.float16 if in_dtype == "fp16" else torch.bfloat16
    torch.manual_seed(0)

    mpad = (M + tile_m - 1) // tile_m * tile_m
    npad = (N + tile_n - 1) // tile_n * tile_n

    print(
        f"Running Split-K HGEMM: M={M}, N={N}, K={K}, "
        f"dtype={in_dtype}, out={_eff_out}, SPLIT_K={SPLIT_K}, bufs={num_buffers}, "
        f"tdm_store={use_tdm_store}"
    )

    # A: M x K, B: K x N (gfx1250 convention)
    a = torch.randn((M, K), dtype=torch_dtype, device='cpu').cuda()
    b = torch.randn((K, N), dtype=torch_dtype, device='cpu').cuda()

    a_pad = torch.zeros((mpad, K), dtype=torch_dtype, device='cpu').cuda()
    b_pad = torch.zeros((K, npad), dtype=torch_dtype, device='cpu').cuda()
    a_pad[:M, :] = a
    b_pad[:, :N] = b

    c_pad = torch.zeros((mpad, npad), dtype=_out_torch, device='cpu').cuda()

    # Semaphore for split-K
    semaphore = torch.zeros(
        (3 * SPLIT_K_COUNTER_MAX_LEN,), dtype=torch.int32, device='cuda')
    signal_state = 0

    # Check grid constraint for split-K
    if SPLIT_K > 1:
        bm = (mpad + tile_m - 1) // tile_m
        bn = npad // tile_n
        if bm * bn > SPLIT_K_COUNTER_MAX_LEN:
            pytest.skip(f"Grid {bm}x{bn} exceeds SPLIT_K_COUNTER_MAX_LEN={SPLIT_K_COUNTER_MAX_LEN}")

    launch_fn = compile_hgemm_splitk_gfx1250(
        N=npad, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        SPLIT_K=SPLIT_K,
        m_warp=m_warp, n_warp=n_warp, in_dtype=in_dtype,
        out_dtype=out_dtype,
        num_buffers=num_buffers,
        l2_prefetch_distance=l2_prefetch_distance,
        use_tdm_store=use_tdm_store,
        inst_prefetch=inst_prefetch,
    )
    launch_fn(
        c_pad.contiguous().view(-1),
        a_pad.contiguous().view(-1),
        b_pad.contiguous().view(-1),
        mpad, npad, semaphore, signal_state,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    ref = torch.mm(a.cpu().to(torch.float32), b.cpu().to(torch.float32))
    rtol = 0.1
    atol = 0.1
    assert verify_output(c_pad[:M, :N].cpu().to(torch.float32), ref, rtol=rtol, atol=atol)
    print("PASSED")


@pytest.mark.parametrize("in_dtype", ["fp16", "bf16"])
def test_hgemm_splitk_gfx1250_wrapper(in_dtype):
    """Test the convenience wrapper function."""
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"WMMA requires gfx1250, got {arch}")

    torch_dtype = torch.float16 if in_dtype == "fp16" else torch.bfloat16
    torch.manual_seed(42)

    M, N, K = 256, 256, 256
    a = torch.randn((M, K), dtype=torch_dtype, device='cuda')
    b = torch.randn((K, N), dtype=torch_dtype, device='cuda')
    c = torch.zeros((M, N), dtype=torch_dtype, device='cuda')

    kwargs = {'tile_m': 64, 'tile_n': 256, 'tile_k': 128, 'SPLIT_K': 1, 'num_buffers': 2}
    hgemm_splitk_gfx1250_(c, a, b, kwargs, torch.cuda.current_stream())
    torch.cuda.synchronize()

    ref = torch.mm(a.cpu().to(torch.float32), b.cpu().to(torch.float32))
    assert verify_output(c.cpu().to(torch.float32), ref, rtol=0.1, atol=0.1)
    print("PASSED wrapper test")


@pytest.mark.parametrize("in_dtype", ["fp16"])
def test_hgemm_splitk_gfx1250_buffer_store(in_dtype):
    """Test with buffer_store epilogue (use_tdm_store=False, SPLIT_K=1)."""
    test_hgemm_splitk_gfx1250(
        in_dtype,
        256, 256, 512,
        64, 256, 128,
        SPLIT_K=1,
        num_buffers=2,
        use_tdm_store=False,
    )


DEFAULT_BENCH_ITERS = 50
DEFAULT_BENCH_WARMUP = 3


def bench_hgemm_splitk_gfx1250(
    in_dtype, M, N, K, tile_m, tile_n, tile_k, SPLIT_K, num_buffers,
    m_warp=2, n_warp=4, l2_prefetch_distance=2,
    out_dtype=None, use_tdm_store=True, inst_prefetch=False,
    bench_iters=DEFAULT_BENCH_ITERS, bench_warmup=DEFAULT_BENCH_WARMUP,
):
    """Split-K HGEMM benchmark for gfx1250 — reports TFLOPS and BW."""
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        print(f"Skipped: WMMA requires gfx1250, got {arch}")
        return

    assert K % SPLIT_K == 0, f"K={K} must be divisible by SPLIT_K={SPLIT_K}"
    ks = K // SPLIT_K

    load_tile_k = 2 * tile_k  # kernel loads 2*tile_k per stage
    num_k_tiles = ks // load_tile_k
    if num_k_tiles < num_buffers:
        print(f"Skipped: {num_buffers}-buffer requires num_k_tiles >= {num_buffers}, got {num_k_tiles}")
        return

    lds_pad = 8
    elem_bytes = 2
    a_buf = tile_m * (load_tile_k + lds_pad) * elem_bytes
    b_buf = load_tile_k * (tile_n + lds_pad) * elem_bytes
    total_lds = (a_buf + b_buf) * num_buffers
    if total_lds > 327680:
        print(f"Skipped: LDS budget exceeded: {total_lds} > 327680")
        return

    if SPLIT_K > 1:
        use_tdm_store = False

    _eff_out = out_dtype or ("f16" if in_dtype == "fp16" else "bf16")
    _out_torch = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}[_eff_out]
    torch_dtype = torch.float16 if in_dtype == "fp16" else torch.bfloat16
    torch.manual_seed(0)

    mpad = (M + tile_m - 1) // tile_m * tile_m
    npad = (N + tile_n - 1) // tile_n * tile_n

    print(
        f"Benchmarking Split-K HGEMM: M={M}, N={N}, K={K}, "
        f"dtype={in_dtype}, out={_eff_out}, SPLIT_K={SPLIT_K}, bufs={num_buffers}, "
        f"tdm_store={use_tdm_store}, iters={bench_iters}, warmup={bench_warmup}"
    )

    a = torch.randn((M, K), dtype=torch_dtype, device='cpu').cuda()
    b = torch.randn((K, N), dtype=torch_dtype, device='cpu').cuda()

    a_pad = torch.zeros((mpad, K), dtype=torch_dtype, device='cpu').cuda()
    b_pad = torch.zeros((K, npad), dtype=torch_dtype, device='cpu').cuda()
    a_pad[:M, :] = a
    b_pad[:, :N] = b

    c_pad = torch.zeros((mpad, npad), dtype=_out_torch, device='cpu').cuda()

    semaphore = torch.zeros(
        (3 * SPLIT_K_COUNTER_MAX_LEN,), dtype=torch.int32, device='cuda')
    signal_state = 0

    if SPLIT_K > 1:
        bm = (mpad + tile_m - 1) // tile_m
        bn = npad // tile_n
        if bm * bn > SPLIT_K_COUNTER_MAX_LEN:
            print(f"Skipped: Grid {bm}x{bn} exceeds SPLIT_K_COUNTER_MAX_LEN={SPLIT_K_COUNTER_MAX_LEN}")
            return

    launch_fn = compile_hgemm_splitk_gfx1250(
        N=npad, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        SPLIT_K=SPLIT_K,
        m_warp=m_warp, n_warp=n_warp, in_dtype=in_dtype,
        out_dtype=out_dtype,
        num_buffers=num_buffers,
        l2_prefetch_distance=l2_prefetch_distance,
        use_tdm_store=use_tdm_store,
        inst_prefetch=inst_prefetch,
    )

    c_flat = c_pad.contiguous().view(-1)
    a_flat = a_pad.contiguous().view(-1)
    b_flat = b_pad.contiguous().view(-1)
    stream = torch.cuda.current_stream()

    def run_kernel():
        launch_fn(c_flat, a_flat, b_flat, mpad, npad, semaphore, signal_state, stream)

    emu_mode = os.environ.get("EMU_MODE", "0") == "1"

    if emu_mode:
        # EMU_MODE: single iteration for correctness only
        run_kernel()
        torch.cuda.synchronize()
    else:
        # Warmup
        for _ in range(bench_warmup):
            run_kernel()
        torch.cuda.synchronize()
        # Timed iterations
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(bench_iters):
            run_kernel()
        end_event.record()
        end_event.synchronize()
        us = start_event.elapsed_time(end_event) * 1000 / bench_iters  # ms -> us
        torch.cuda.synchronize()

    # Correctness check
    ref = torch.mm(a.cpu().to(torch.float32), b.cpu().to(torch.float32))
    rtol = 0.1
    atol = 0.1
    correct = verify_output(c_pad[:M, :N].cpu().to(torch.float32), ref, rtol=rtol, atol=atol)

    status = "PASSED" if correct else "FAILED"
    if emu_mode:
        print(f"[flyc] {status} (EMU_MODE — timing disabled)")
    else:
        flops = 2 * M * N * K
        bytes_moved = (M * K + K * N) * elem_bytes + M * N * 2  # A + B reads + C write
        tflops = flops / (us / 1e6) / 1e12
        tbps = bytes_moved / 1e12 / (us / 1e6)
        print(
            f"[flyc] {status} | {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gfx1250 Split-K HGEMM test/benchmark")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("-M", type=int, default=256)
    parser.add_argument("-N", type=int, default=256)
    parser.add_argument("-K", type=int, default=256)
    parser.add_argument("--tile-m", type=int, default=64)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--split-k", type=int, default=1)
    parser.add_argument("--num-buffers", type=int, default=2)
    parser.add_argument("--m-warp", type=int, default=2)
    parser.add_argument("--n-warp", type=int, default=4)
    parser.add_argument("--no-tdm-store", action="store_true", default=False)
    parser.add_argument("--inst-prefetch", action="store_true", default=False)
    parser.add_argument("--num-iters", type=int, default=DEFAULT_BENCH_ITERS)
    parser.add_argument("--num-warmup", type=int, default=DEFAULT_BENCH_WARMUP)
    args = parser.parse_args()
    torch.set_default_device("cuda")
    try:
        bench_hgemm_splitk_gfx1250(
            args.dtype,
            M=args.M, N=args.N, K=args.K,
            tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k,
            SPLIT_K=args.split_k,
            num_buffers=args.num_buffers,
            m_warp=args.m_warp, n_warp=args.n_warp,
            use_tdm_store=not args.no_tdm_store,
            inst_prefetch=args.inst_prefetch,
            bench_iters=args.num_iters,
            bench_warmup=args.num_warmup,
        )
    except pytest.skip.Exception as e:
        print(f"Skipped: {e}")
