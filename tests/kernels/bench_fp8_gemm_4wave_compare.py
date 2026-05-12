#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FP8 4-wave GEMM perf comparison vs torch._scaled_mm.

Runs the FlyDSL kernel (``kernels.fp8_gemm_4wave``) and
``torch._scaled_mm`` on the same input tensors with the same warmup/iter
config, then prints a side-by-side TFLOPS / us table.
"""

import argparse
import os
import sys

import torch

import flydsl.compiler as flyc

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch
from kernels.fp8_gemm_4wave import compile_fp8_gemm
from tests.test_common import run_perftest, verify_output
from tests.utils import pertoken_quant

FP8_DTYPE = torch.float8_e4m3fn
OUT_DTYPE = torch.bfloat16

DEFAULT_SHAPES = [
    (5120, 5120, 8320, 256, 256),
    (8192, 8192, 8192, 256, 256),
    (9728, 8192, 8320, 256, 256),
]


def _as_i8(t):
    return t.view(torch.int8) if "float8" in str(t.dtype) else t


def _flydsl_args(c, a, b, sa, sb):
    return (
        _as_i8(a).contiguous().view(-1),
        _as_i8(b).contiguous().view(-1),
        c.contiguous().view(-1),
        sa.contiguous().view(-1),
        sb.contiguous().view(-1),
        torch.cuda.current_stream(),
    )


def _ref_dequant_mm(a, b, sa, sb):
    a_f32 = a.to(torch.float32) * sa.view(-1, 1)
    b_f32 = b.to(torch.float32) * sb.view(-1, 1)
    return torch.mm(a_f32, b_f32.T).to(torch.float32)


def _bench_flydsl(M, N, K, tile_m, tile_n, a_q, b_q, sa, sb, c_ref, num_warmups, num_iters):
    c_out = torch.zeros((M, N), dtype=OUT_DTYPE, device="cuda")
    launch_fn = compile_fp8_gemm(M=M, N=N, K=K, BLOCK_M=tile_m, BLOCK_N=tile_n, use_xcd_remap=True)
    compiled = flyc.compile(launch_fn, *_flydsl_args(c_out, a_q, b_q, sa, sb))

    def _launch(c, a, b, s_a, s_b):
        compiled(*_flydsl_args(c, a, b, s_a, s_b))

    _, us = run_perftest(_launch, c_out, a_q, b_q, sa, sb, num_iters=num_iters, num_warmup=num_warmups)
    torch.cuda.synchronize()
    ok = verify_output(c_out.to(torch.float32), c_ref, rtol=0.1, atol=0.1)
    return us, ok


def _bench_torch_scaled_mm(M, N, K, a_q, b_q, sa, sb, c_ref, num_warmups, num_iters):
    # b_q stored as (N, K) row-major. _scaled_mm wants the second operand in
    # (K, N) shape with column-major storage, i.e. ``b_q.t()``.
    b_t = b_q.t()
    sa_v = sa.view(M, 1).to(torch.float32).contiguous()
    sb_v = sb.view(1, N).to(torch.float32).contiguous()

    out = torch.empty((M, N), dtype=OUT_DTYPE, device="cuda")

    def _launch(_out):
        torch._scaled_mm(a_q, b_t, scale_a=sa_v, scale_b=sb_v, out_dtype=OUT_DTYPE, out=_out)

    _, us = run_perftest(_launch, out, num_iters=num_iters, num_warmup=num_warmups)
    torch.cuda.synchronize()
    ok = verify_output(out.to(torch.float32), c_ref, rtol=0.1, atol=0.1)
    return us, ok


def _bench_one(M, N, K, tile_m, tile_n, *, num_warmups, num_iters, skip_torch):
    print(f"\n===== M={M} N={N} K={K} BLOCK={tile_m}x{tile_n} =====")

    device = torch.device("cuda")
    torch.manual_seed(0)
    a_fp32 = torch.rand(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.rand(N, K, device=device, dtype=torch.float32)
    a_q, sa = pertoken_quant(a_fp32, quant_dtype=FP8_DTYPE)
    b_q, sb = pertoken_quant(b_fp32_t, quant_dtype=FP8_DTYPE)
    a_q = a_q.contiguous()
    b_q = b_q.contiguous()
    sa = sa.squeeze().contiguous()
    sb = sb.squeeze().contiguous()

    c_ref = _ref_dequant_mm(a_q, b_q, sa, sb)

    flops = 2 * M * N * K

    results = []
    us, ok = _bench_flydsl(M, N, K, tile_m, tile_n, a_q, b_q, sa, sb, c_ref, num_warmups, num_iters)
    results.append(("flydsl-4wave", us, ok))

    if not skip_torch:
        try:
            us, ok = _bench_torch_scaled_mm(M, N, K, a_q, b_q, sa, sb, c_ref, num_warmups, num_iters)
            results.append(("torch._scaled_mm", us, ok))
        except Exception as e:
            print(f"  torch._scaled_mm failed: {e}")
            results.append(("torch._scaled_mm", float("nan"), False))

    base_us = next((us for label, us, _ in results if label == "flydsl-4wave"), None)
    print(f"  {'kernel':<22} {'us':>10} {'TFLOPS':>10} {'vs flydsl':>11}  ok")
    for label, us, ok in results:
        tflops = flops / (us / 1e6) / 1e12 if us == us and us > 0 else float("nan")
        rel = (base_us / us * 100.0) if base_us and us == us and us > 0 else float("nan")
        print(f"  {label:<22} {us:>10.2f} {tflops:>10.2f} {rel:>10.1f}%  {ok}")

    return results


def main():
    parser = argparse.ArgumentParser(description="FP8 4-wave GEMM comparison vs torch._scaled_mm")
    parser.add_argument(
        "--shapes",
        type=str,
        default="all",
        help="'all' (default), or 'M,N,K,tile_m,tile_n' for a single shape",
    )
    parser.add_argument("--num_iters", type=int, default=10)
    parser.add_argument("--num_warmups", type=int, default=5)
    parser.add_argument("--skip_torch", action="store_true", default=False)
    args = parser.parse_args()

    arch = str(get_rocm_arch())
    if "gfx95" not in arch:
        print(f"FP8 4-wave GEMM requires CDNA4 (gfx95*), got {arch}")
        return 1

    torch.set_default_device("cuda")

    if args.shapes == "all":
        shapes = DEFAULT_SHAPES
    else:
        parts = [int(x) for x in args.shapes.split(",")]
        assert len(parts) == 5, "Expected M,N,K,tile_m,tile_n"
        shapes = [tuple(parts)]

    for M, N, K, tm, tn in shapes:
        _bench_one(
            M, N, K, tm, tn,
            num_warmups=args.num_warmups,
            num_iters=args.num_iters,
            skip_torch=args.skip_torch,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
