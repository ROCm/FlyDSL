#!/usr/bin/env python3
"""MFMA FP8/INT8 GEMM Test using flir with B preshuffle.

NOTE:
- Kernel implementation (IR builder) lives in `kernels/preshuffle_gemm.py`
  (same structure as `kernels/moe_gemm_2stage.py`).
- This file is only the correctness + perf harness.
"""

import os
import sys
import logging

import torch
import torch.nn.functional as F
import pytest

# -----------------------------------------------------------------------------
# Ensure we use the repo-local `flydsl` when running this file directly.
#
# Some environments have another `flydsl` (e.g. from a sibling checkout) earlier
# on `sys.path`, which can miss newer ROCDL wrappers (notably INT8 MFMA helpers).
# -----------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLIR_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLIR_SRC not in sys.path:
    sys.path.insert(0, _PYFLIR_SRC)

from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
from tests.test_common import run_perftest, verify_output
from tests.utils import pertoken_quant, shuffle_weight
from flydsl.runtime.device import get_rocm_arch


logging.basicConfig(level=logging.INFO)


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


# Aiter imports (optional)
try:
    import aiter

    HAS_AITER = True
except ImportError:
    print("Warning: Aiter not found, skipping comparison")
    HAS_AITER = False


DEFAULT_LDS_STAGE = 2
DEFAULT_BENCH_ITERS = 20
DEFAULT_BENCH_WARMUP = 3
DEFAULT_RUN_AITER_BENCH = True

ARCH = get_rocm_arch()
# GFX950 (MI350) and newer typically use OCP standard float8_e4m3fn
# GFX940/941/942 (MI300) use float8_e4m3fnuz
DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    """Torch reference implementation (from aiter project).

    Dequantize 8-bit inputs (FP8/INT8) and compute FP32 matmul.
    """
    x = x.to(torch.float32) if x_scale is None else (x.to(torch.float32) * x_scale)
    weight = weight.to(torch.float32) if w_scale is None else (weight.to(torch.float32) * w_scale)
    out = F.linear(x, weight)
    if bias is not None:
        out = out.to(bias.dtype) + bias
    return out.to(dtype)


@pytest.mark.parametrize("in_dtype", ["fp8", "int8", "bf16"])
@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k", 
    [
        (16, 5120, 8192, 16, 64, 512), 
        (5120, 5120, 8320, 64, 256, 128), 
        (9728, 8192, 8320, 128, 128, 128),
        (5133, 5120, 8320, 64, 256, 128),
    ]
)
def test_mfma_a8_flir_preshuffle(
    in_dtype,
    M,
    N,
    K,
    tile_m,
    tile_n,
    tile_k,
    *,
    lds_stage: int = DEFAULT_LDS_STAGE,
    bench_iters: int = DEFAULT_BENCH_ITERS,
    bench_warmup: int = DEFAULT_BENCH_WARMUP,
    run_aiter_bench: bool = DEFAULT_RUN_AITER_BENCH,
    use_cshuffle_epilog: bool = False,
):
    print("=" * 80)
    print(
        f"MFMA {in_dtype.upper()} GEMM Test (Tile: {tile_m}x{tile_n}x{tile_k}) [Torch Optimized]"
    )
    print("=" * 80)

    lds_stage = int(lds_stage)
    if lds_stage not in (1, 2):
        raise ValueError(
            f"lds_stage must be 1 or 2, got {lds_stage!r}"
        )
    exe = compile_preshuffle_gemm_a8(
        M=M,
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        lds_stage=lds_stage,
        use_cshuffle_epilog=bool(use_cshuffle_epilog),
    )
    print(f"✓ Compiled (lds_stage={lds_stage})")

    size_c = M * N
    size_a = M * K
    # B is packed int4 for W4A8: 2 values per byte.
    if in_dtype == "int4":
        size_b = (N * K) // 2
        elem_bytes = 1
    elif in_dtype in ("fp16", "bf16"):
        size_b = (N * K) * 2
        elem_bytes = 2
    else:
        size_b = (N * K)
        elem_bytes = 1

    device = torch.device("cuda")

    torch.manual_seed(42)
    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.randn(N, K, device=device, dtype=torch.float32)  # (N, K)

    is_int4 = in_dtype == "int4"
    # INT4 here means W4A8: A is INT8, B is packed INT4 and unpacked to INT8 in-kernel.
    is_int8 = (in_dtype == "int8") or is_int4

    if in_dtype in ("fp16", "bf16"):
        torch_dtype = torch.float16 if in_dtype == "fp16" else torch.bfloat16
        a_q = a_fp32.to(torch_dtype)
        b_q = b_fp32_t.to(torch_dtype)
        # Scale is semantically optional for fp16/bf16 (no dequant). Let callers pass None;
        # we will materialize an internal "all-ones" scale only when launching the kernel.
        scale_a = None
        scale_b = None
    else:
        quant_dtype = torch.int8 if is_int8 else DTYPE_FP8
        a_q, scale_a = pertoken_quant(a_fp32, quant_dtype=quant_dtype)  # (M, K)
        if is_int4:
            # Signed int4 range is [-8, 7]. Use dtypeMax=7 for symmetric per-row scaling.
            b_q, scale_b = pertoken_quant(b_fp32_t, quant_dtype=torch.int8, dtypeMax=7)  # (N, K)
        else:
            b_q, scale_b = pertoken_quant(b_fp32_t, quant_dtype=quant_dtype)  # (N, K)

    # Keep tensors contiguous for predictable buffer descriptor shapes.
    a_q = a_q.contiguous()
    b_q = b_q.contiguous()

    # Preshuffle B to CK/aiter layout.
    b_shuffled = shuffle_weight(b_q, layout=(16, 16))

    def _pack_shuffled_int8_to_packed_int4_no_perm(x_shuf_i8: torch.Tensor) -> torch.Tensor:
        """
        Pack a preshuffled int8 tensor (values in [-8,7]) into packed int4 bytes.

        Each contiguous 8-byte block [v0..v7] -> 4 bytes:
          b0=(v4<<4)|v0, b1=(v5<<4)|v1, b2=(v6<<4)|v2, b3=(v7<<4)|v3.

        This matches the 7-op in-kernel unpack sequence and avoids any v_perm.
        """
        flat = x_shuf_i8.contiguous().view(-1).to(torch.int16)
        assert flat.numel() % 8 == 0
        u = (flat & 0xF).to(torch.uint8).view(-1, 8)
        out = torch.empty((u.shape[0], 4), device=u.device, dtype=torch.uint8)
        out[:, 0] = u[:, 0] | (u[:, 4] << 4)
        out[:, 1] = u[:, 1] | (u[:, 5] << 4)
        out[:, 2] = u[:, 2] | (u[:, 6] << 4)
        out[:, 3] = u[:, 3] | (u[:, 7] << 4)
        return out.view(-1).to(torch.int8)

    b_packed = None
    if is_int4:
        b_packed = _pack_shuffled_int8_to_packed_int4_no_perm(b_shuffled)

    # Reference (dequant + matmul).
    c_ref = run_torch(a_q, b_q, scale_a, scale_b, bias=None, dtype=torch.float32)

    # Run kernel (f16 output, in-kernel scaling).
    c_out_raw = torch.zeros((M, N), dtype=torch.float16, device=device)

    def launch_kernel(c, a, b, sa, sb):
        # Keep kernel ABI consistent: for fp16/bf16, pass empty scale tensors (kernel ignores them).
        if sa is None:
            sa = torch.empty((0,), device=c.device, dtype=torch.float32)
        if sb is None:
            sb = torch.empty((0,), device=c.device, dtype=torch.float32)
        exe(c, a, b, sa, sb, M, N, K)

    # `run_perftest` requires num_iters > 1.
    bench_iters = max(2, int(bench_iters))
    bench_warmup = int(bench_warmup)
    _, us = run_perftest(
        launch_kernel,
        c_out_raw,
        a_q,
        b_packed if is_int4 else b_shuffled,
        scale_a,
        scale_b,
        num_iters=bench_iters,
        num_warmup=bench_warmup,
    )
    torch.cuda.synchronize()
    c_out_scaled = c_out_raw.to(torch.float32)

    assert verify_output(c_out_scaled, c_ref, rtol=0.1, atol=0.1)

    bytes_moved = (size_a * elem_bytes) + size_b + size_c * 2 + (M + N) * 4
    flops = 2 * M * N * K
    tflops = flops / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(f"Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")

    if HAS_AITER and bool(run_aiter_bench) and (not is_int4) and (in_dtype in ("fp8", "int8")):
        print("-" * 40)
        print("Running Aiter Benchmark...")
        try:
            def launch_aiter(a, b, sa, sb):
                return aiter.gemm_a8w8_bpreshuffle(a, b, sa, sb, None, torch.float16)

            c_aiter, us1 = run_perftest(launch_aiter, a_q, b_shuffled, scale_a, scale_b)
            verify_output(c_aiter.to(torch.float32), c_ref, rtol=0.1, atol=0.1)

            tflops_aiter = flops / (us1 / 1e6) / 1e12
            bw_aiter = bytes_moved / 1e9 / (us1 / 1e6)
            print(
                f"Aiter Throughput: {us1:.1f} us, {tflops_aiter:.2f} TFLOPS, BW: {bw_aiter:.2f} GB/s"
            )
            print(
                f"Speedup vs Aiter: {tflops / tflops_aiter:.2f}x, Tflops {tflops:.1f} vs {tflops_aiter:.1f}"
            )
            print("-" * 40)
        except Exception as e:
            # Best-effort only: aiter can be importable but fail to load its JIT .so deps.
            msg = str(e).splitlines()[0] if str(e) else repr(e)
            print(f"Skipping Aiter benchmark (not runnable here): {msg}")
            print("-" * 40)
    elif HAS_AITER and (not bool(run_aiter_bench)):
        print("-" * 40)
        print("Skipping Aiter benchmark (pass --run_aiter_bench to enable)")
        print("-" * 40)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preshuffle GEMM benchmark"
    )
    parser.add_argument(
        "--in_dtype",
        type=str,
        default="fp8",
        choices=["fp8", "int8", "int4", "fp16", "bf16"],
                        help="Input dtype")
    parser.add_argument("-M", type=int, default=16, help="M dimension")
    parser.add_argument("-N", type=int, default=10240, help="N dimension")
    parser.add_argument("-K", type=int, default=8192, help="K dimension")
    parser.add_argument("--tile_m", type=int, default=16, help="Tile M")
    parser.add_argument("--tile_n", type=int, default=64, help="Tile N")
    parser.add_argument("--tile_k", type=int, default=256, help="Tile K")
    # Explicit CLI knobs (no env vars).
    parser.add_argument(
        "--lds_stage",
        type=int,
        default=DEFAULT_LDS_STAGE,
        choices=[1, 2],
        help="LDS staging: 2=ping-pong (default), 1=single.",
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=DEFAULT_BENCH_ITERS,
        help="Benchmark iters (>=2).",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=DEFAULT_BENCH_WARMUP,
        help="Benchmark warmup iters.",
    )
    parser.add_argument(
        "--run_aiter_bench",
        action="store_true",
        default=DEFAULT_RUN_AITER_BENCH,
        help="Run aiter comparison benchmark (fp8/int8 only). Default: enabled.",
    )
    parser.add_argument(
        "--no_aiter_bench",
        action="store_false",
        dest="run_aiter_bench",
        help="Disable aiter benchmark.",
    )
    parser.add_argument(
        "--use_cshuffle_epilog",
        action="store_true",
        default=False,
        help="Enable LDS cshuffle epilogue (A/B perf experiment). Default: off.",
    )
    
    args = parser.parse_args()
    
    torch.set_default_device("cuda")
    test_mfma_a8_flir_preshuffle(
        args.in_dtype, 
        M=args.M, 
        N=args.N, 
        K=args.K, 
        tile_m=args.tile_m, 
        tile_n=args.tile_n, 
        tile_k=args.tile_k,
        lds_stage=args.lds_stage,
        bench_iters=args.num_iters,
        bench_warmup=args.num_warmup,
        run_aiter_bench=bool(args.run_aiter_bench),
        use_cshuffle_epilog=bool(args.use_cshuffle_epilog),
    )

