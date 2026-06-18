#!/usr/bin/env python3
"""BF16 GEMM experiment for ISA and ATT inspection.

Logical problem:
    A: (M, K) bf16, row-major
    B: (K, N) bf16, row-major
    C: (M, N) bf16

The FlyDSL HGEMM kernel consumes B in transposed row-major form
``B_T: (N, K)``.  This harness keeps the user-facing shape above, then passes
``B_T`` to the kernel.  The default shape is large enough to fill the GPU:

    M = 8192, N = 8192, K = 16384

The per-CTA tile is:

    BM = 256, BN = 256, BK = 64, stages = 2, waves = 2 x 2 x 1 = 4
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch
from kernels.hgemm_splitk import hgemm_splitk_


DEFAULT_M = 8192
DEFAULT_N = 8192
DEFAULT_K = 16384

HGEMM_KWARGS = {
    "TILE_M": 256,
    "TILE_N": 256,
    "TILE_K": 64,
    "STAGES": 2,
    "SPLIT_K": 1,
    "BLOCK_M_WARPS": 2,
    "BLOCK_N_WARPS": 2,
    "BLOCK_K_WARPS": 1,
    "B_TO_LDS": True,
}


def _kernel_name() -> str:
    # Keep in sync with kernels/hgemm_splitk.py.
    async_tag = "AS1" if str(get_rocm_arch()) != "gfx942" else "AS0"
    name = (
        "hgemm_bf16_256x256x64x2"
        "_SPK1_W2x2x1_BLDS1_TN"
        f"_{async_tag}"
    )
    if HGEMM_KWARGS.get("A_LDS_K32_BLOCKING", False) or HGEMM_KWARGS.get("K32_REGISTER_PIPELINE", False):
        name += "_AK32"
    if HGEMM_KWARGS.get("B_LDS_K32_BLOCKING", False) or HGEMM_KWARGS.get("K32_REGISTER_PIPELINE", False):
        name += "_BK32"
    if HGEMM_KWARGS.get("K32_REGISTER_PIPELINE", False):
        name += "_RP1"
    return name


def assert_no_spills() -> None:
    dump_dir = os.environ.get("FLYDSL_DUMP_DIR")
    if not dump_dir:
        return

    root = Path(dump_dir)
    if not root.exists():
        raise RuntimeError(f"FLYDSL_DUMP_DIR does not exist: {root}")

    candidates = [p for p in root.rglob("22_final_isa.s") if p.parent.name.startswith(_kernel_name())]
    if not candidates:
        raise RuntimeError(f"could not find final ISA for {_kernel_name()} under {root}")

    isa_path = max(candidates, key=lambda p: p.stat().st_mtime)
    text = isa_path.read_text()

    checks = {
        "sgpr_spill_count": re.compile(r"\.sgpr_spill_count:\s*(\d+)"),
        "vgpr_spill_count": re.compile(r"\.vgpr_spill_count:\s*(\d+)"),
        "private_segment_fixed_size": re.compile(r"\.private_segment_fixed_size:\s*(\d+)"),
        "uses_flat_scratch": re.compile(r"uses_flat_scratch,\s*(\d+)"),
    }
    failures = []
    for name, pattern in checks.items():
        matches = [int(m.group(1)) for m in pattern.finditer(text)]
        if matches and any(v != 0 for v in matches):
            failures.append(f"{name}={matches}")

    scratch_ops = re.findall(r"\b(?:scratch_(?:load|store)|buffer_(?:load|store)[^\n]*scratch)", text)
    if scratch_ops:
        failures.append("scratch memory ops present")

    if failures:
        detail = "; ".join(failures)
        raise RuntimeError(f"kernel spill check failed for {isa_path}: {detail}")

    print(f"spill_check=ok isa={isa_path}")


def preshuffle_rows_k(t: torch.Tensor, tile_row: int, tile_k: int) -> torch.Tensor:
    """Permute a row-major [R, K] tensor so each [tile_row, tile_k] block is
    physically contiguous: [R, K] -> [R//tr, K//tk, tr, tk] -> flat. Matches the
    kernel's preshuffle_off mapping (PRESHUFFLE=True)."""
    r, k = t.shape[-2:]
    assert r % tile_row == 0 and k % tile_k == 0, f"need R%{tile_row}==0 K%{tile_k}==0, got R={r} K={k}"
    return t.reshape(r // tile_row, tile_row, k // tile_k, tile_k).permute(0, 2, 1, 3).contiguous().view(r, k)


def make_inputs(m: int, n: int, k: int, *, skip_init: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    if skip_init:
        a = torch.empty((m, k), device=device, dtype=torch.bfloat16)
        b = torch.empty((k, n), device=device, dtype=torch.bfloat16)
    else:
        gen = torch.Generator(device=device)
        gen.manual_seed(123)
        # Small magnitude keeps bf16-output error easy to inspect.
        a = (torch.rand((m, k), device=device, dtype=torch.float32, generator=gen) - 0.5).to(torch.bfloat16) * 0.125
        b = (torch.rand((k, n), device=device, dtype=torch.float32, generator=gen) - 0.5).to(torch.bfloat16) * 0.125
    c = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    return a, b, c


def run_kernel(c: torch.Tensor, a: torch.Tensor, b_t: torch.Tensor) -> None:
    hgemm_splitk_(
        c,
        a,
        b_t,
        None,
        HGEMM_KWARGS,
        torch.cuda.current_stream(),
    )


def check_result(c: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    ref = torch.mm(a.float(), b.float())
    got = c.float()
    diff = got - ref
    max_abs = diff.abs().max().item()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    if not math.isfinite(max_abs) or not math.isfinite(cos):
        raise RuntimeError(f"non-finite check result: max_abs={max_abs}, cos={cos}")
    return max_abs, cos


def time_kernel(c: torch.Tensor, a: torch.Tensor, b_t: torch.Tensor, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        run_kernel(c, a, b_t)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        run_kernel(c, a, b_t)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start
    return elapsed_s * 1.0e6 / max(1, iters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Use torch.empty inputs. Useful for ATT so PyTorch init kernels do not clutter the trace.",
    )
    parser.add_argument(
        "--a-lds-k32-blocking",
        action="store_true",
        help="Enable the experimental K32-blocked LDS layout for A.",
    )
    parser.add_argument(
        "--k32-register-pipeline",
        action="store_true",
        help="Enable A/B K32 LDS layouts and register-prefetch the second K32 slice.",
    )
    parser.add_argument(
        "--b-lds-k32-blocking",
        action="store_true",
        help="Enable the experimental K32-blocked LDS layout for B.",
    )
    parser.add_argument(
        "--preshuffle",
        action="store_true",
        help="Preshuffle A/B so each [256, BLOCK_K] tile is contiguous in global memory.",
    )
    args = parser.parse_args()
    HGEMM_KWARGS["A_LDS_K32_BLOCKING"] = bool(args.a_lds_k32_blocking)
    HGEMM_KWARGS["B_LDS_K32_BLOCKING"] = bool(args.b_lds_k32_blocking)
    HGEMM_KWARGS["K32_REGISTER_PIPELINE"] = bool(args.k32_register_pipeline)
    HGEMM_KWARGS["PRESHUFFLE"] = bool(args.preshuffle)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is not available")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    tile_m = int(HGEMM_KWARGS["TILE_M"])
    tile_n = int(HGEMM_KWARGS["TILE_N"])
    tile_k = int(HGEMM_KWARGS["TILE_K"])
    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        raise ValueError("--m, --n, and --k must be positive")
    if args.n % tile_n != 0:
        raise ValueError(f"--n must be divisible by TILE_N={tile_n}, got {args.n}")
    if args.k % tile_k != 0:
        raise ValueError(f"--k must be divisible by TILE_K={tile_k}, got {args.k}")

    a, b, c = make_inputs(args.m, args.n, args.k, skip_init=args.skip_init)
    b_t = b.t().contiguous() if not args.skip_init else torch.empty((args.n, args.k), device=b.device, dtype=b.dtype)

    # The kernel consumes preshuffled A / B_T; the reference check below still uses
    # the original a / b, so correctness is validated against the true GEMM.
    tile_k = int(HGEMM_KWARGS["TILE_K"])
    if args.preshuffle:
        a_k = preshuffle_rows_k(a, 256, tile_k)
        b_t_k = preshuffle_rows_k(b_t, 256, tile_k)
    else:
        a_k, b_t_k = a, b_t

    # First call compiles and launches the fixed-shape FlyDSL kernel.
    run_kernel(c, a_k, b_t_k)
    torch.cuda.synchronize()
    assert_no_spills()

    max_abs = float("nan")
    cos = float("nan")
    if not args.no_check and not args.skip_init:
        max_abs, cos = check_result(c, a, b)

    us = time_kernel(c, a_k, b_t_k, warmup=args.warmup, iters=args.iters)
    tflops = (2.0 * args.m * args.n * args.k) / (us * 1.0e-6) / 1.0e12

    print(f"arch={get_rocm_arch()}")
    print(f"kernel={_kernel_name()}")
    print(f"shape=A({args.m},{args.k}) B({args.k},{args.n}) C({args.m},{args.n})")
    print(f"tile=BM256 BN256 BK64 stages=2 waves=4 b_to_lds=1")
    print(f"time_us={us:.3f} tflops={tflops:.3f}")
    print(f"check_max_abs={max_abs:.6f} check_cos={cos:.9f}")


if __name__ == "__main__":
    main()
