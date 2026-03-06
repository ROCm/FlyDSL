#!/usr/bin/env python3
"""AOT pre-compilation example for preshuffle GEMM kernels.

Pre-compiles preshuffle GEMM kernels for specified configurations and stores
the compiled binaries in a cache directory.

Usage:
    # Pre-compile with default configurations (auto-detect GPU arch)
    python tests/python/examples/aot_example.py

    # Custom cache directory
    FLYDSL_RUNTIME_CACHE_DIR=/my/cache python tests/python/examples/aot_example.py

    # Pre-compile and verify by running kernels on GPU
    python tests/python/examples/aot_example.py --run_kernel

Environment variables:
    FLYDSL_RUNTIME_CACHE_DIR  Cache directory (default: ~/.flydsl/cache)
    ARCH                      Target GPU architecture (e.g. gfx942, gfx950).
"""

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernels.preshuffle_gemm_flyc import compile_preshuffle_gemm_a8


def _run_kernel(
    launch_fn,
    M: int,
    N: int,
    K: int,
    in_dtype: str,
):
    """Launch the compiled kernel with random data to verify it runs."""
    import torch
    from tests.utils import pertoken_quant, shuffle_weight
    from flydsl.runtime.device import get_rocm_arch

    ARCH = get_rocm_arch()
    DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz

    device = torch.device("cuda")
    torch.manual_seed(0)

    a_fp32 = torch.rand(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.rand(N, K, device=device, dtype=torch.float32)

    is_int4 = in_dtype == "int4"
    is_int8 = (in_dtype == "int8") or is_int4
    is_f16_or_bf16 = in_dtype in ("fp16", "bf16")

    if is_f16_or_bf16:
        torch_dtype = torch.float16 if in_dtype == "fp16" else torch.bfloat16
        a_q = a_fp32.to(torch_dtype)
        b_q = b_fp32_t.to(torch_dtype)
        scale_a = None
        scale_b = None
    else:
        quant_dtype = torch.int8 if is_int8 else DTYPE_FP8
        a_q, scale_a = pertoken_quant(a_fp32, quant_dtype=quant_dtype)
        if is_int4:
            b_q, scale_b = pertoken_quant(b_fp32_t, quant_dtype=torch.int8, dtypeMax=7)
        else:
            b_q, scale_b = pertoken_quant(b_fp32_t, quant_dtype=quant_dtype)

    a_q = a_q.contiguous()
    b_q = b_q.contiguous()
    b_shuffled = shuffle_weight(b_q, layout=(16, 16))

    if is_int4:
        flat = b_shuffled.contiguous().view(-1).to(torch.int16)
        assert flat.numel() % 8 == 0
        u = (flat & 0xF).to(torch.uint8).view(-1, 8)
        out = torch.empty((u.shape[0], 4), device=u.device, dtype=torch.uint8)
        out[:, 0] = u[:, 0] | (u[:, 4] << 4)
        out[:, 1] = u[:, 1] | (u[:, 5] << 4)
        out[:, 2] = u[:, 2] | (u[:, 6] << 4)
        out[:, 3] = u[:, 3] | (u[:, 7] << 4)
        b_input = out.view(-1).to(torch.int8)
    else:
        b_input = b_shuffled

    c_out = torch.zeros((M, N), dtype=torch.float16, device=device)

    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    sa_flat = (
        torch.empty((0,), device=device, dtype=torch.float32)
        if scale_a is None
        else scale_a.contiguous().view(-1)
    )
    sb_flat = (
        torch.empty((0,), device=device, dtype=torch.float32)
        if scale_b is None
        else scale_b.contiguous().view(-1)
    )

    stream = torch.cuda.current_stream()
    launch_fn(
        c_out.contiguous().view(-1),
        _as_i8(a_q.contiguous().view(-1)),
        _as_i8(b_input.contiguous().view(-1)),
        sa_flat,
        sb_flat,
        M,
        N,
        stream,
    )
    torch.cuda.synchronize()
    print(
        f"    output shape={tuple(c_out.shape)}, "
        f"max={c_out.float().abs().max().item():.4f}, "
        f"mean={c_out.float().abs().mean().item():.4f}"
    )

    # Quick sanity check against torch reference
    if is_f16_or_bf16:
        a_f32 = a_q.to(torch.float32)
        b_f32 = b_q.to(torch.float32)
    else:
        a_f32 = a_q.to(torch.float32) * scale_a.view(-1, 1)
        b_f32 = b_q.to(torch.float32) * scale_b.view(-1, 1)
    c_ref = torch.mm(a_f32, b_f32.T).to(torch.float32)
    c_check = c_out.to(torch.float32)

    abs_diff = (c_check - c_ref).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    print(f"    ref check: max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f}")


# ---------------------------------------------------------------------------
# Configurations
#
# Each entry: (M, N, K, tile_m, tile_n, tile_k)
# ---------------------------------------------------------------------------

BENCHMARK_CONFIGS = [
    (5120, 5120, 8320, 64, 256, 128),
    (5120, 2048, 8320, 128, 128, 128),
]

SMALL_CONFIGS = [
    (256, 1024, 2048, 32, 64, 512),
    (16, 5120, 8192, 16, 64, 512),
]

BAD_TILE_CONFIGS = [
    (256, 1024, 256, 32, 128, 17),
]

CONFIG_PRESETS = {
    "benchmark": BENCHMARK_CONFIGS,
    "small": SMALL_CONFIGS,
    "bad_tile": BAD_TILE_CONFIGS,
}

DTYPE_PRESETS = {
    "fp8": ["fp8"],
    "int8": ["int8"],
    "both": ["fp8", "int8"],
}


def compile_one_config(
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    in_dtype: str = "fp8",
    lds_stage: int = 2,
    run_kernel: bool = False,
) -> dict:
    """Compile one preshuffle GEMM configuration.

    When run_kernel=True, also launches the kernel with random data to verify
    that the compiled binary actually runs on the GPU.

    Returns a dict with timing info.
    """
    shape_str = (
        f"M={M} N={N} K={K} "
        f"tile=({tile_m},{tile_n},{tile_k}) "
        f"dtype={in_dtype}"
    )
    result = {"shape": shape_str, "compile_time": None}

    t0 = time.time()
    try:
        launch_fn = compile_preshuffle_gemm_a8(
            M=M,
            N=N,
            K=K,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            in_dtype=in_dtype,
            lds_stage=lds_stage,
        )
        elapsed = time.time() - t0
        result["compile_time"] = elapsed
        print(f"  [OK] compile  {elapsed:6.1f}s  {shape_str}")

        if run_kernel and launch_fn is not None:
            _run_kernel(launch_fn, M, N, K, in_dtype)
    except Exception as e:
        print(f"  [FAIL] compile  {shape_str}: {e}")

    return result


def test_bad_tile_error():
    """Verify that an unsupported tile size produces a clear compile error."""
    for cfg in BAD_TILE_CONFIGS:
        M, N, K, tile_m, tile_n, tile_k = cfg
        shape_str = f"M={M} N={N} K={K} tile=({tile_m},{tile_n},{tile_k})"
        try:
            compile_preshuffle_gemm_a8(
                M=M, N=N, K=K,
                tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
                in_dtype="fp8",
            )
            raise AssertionError(f"No error raised for bad tile: {shape_str}")
        except (ValueError, RuntimeError) as e:
            print(f"  [OK] Correctly rejected bad tile: {shape_str}")
            print(f"       Error: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="AOT pre-compile preshuffle GEMM kernels into FlyDSL cache",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="small",
        choices=list(CONFIG_PRESETS.keys()),
        help="Configuration preset (default: small)",
    )
    parser.add_argument(
        "--in_dtype",
        type=str,
        default="both",
        choices=list(DTYPE_PRESETS.keys()) + ["fp16", "bf16", "int4"],
        help="Input data type(s): 'both' compiles fp8+int8 (default: both)",
    )
    parser.add_argument(
        "--lds_stage",
        type=int,
        default=2,
        choices=[1, 2],
        help="LDS pipeline stages (default: 2)",
    )
    parser.add_argument(
        "--run_kernel",
        action="store_true",
        help="After compilation, launch each kernel with random data to verify it runs",
    )
    parser.add_argument(
        "--test_bad_tile",
        action="store_true",
        help="Also test that an invalid tile size is properly rejected",
    )
    args = parser.parse_args()

    cache_dir = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "~/.flydsl/cache")
    arch = os.environ.get("ARCH", "(auto-detect)")

    configs = CONFIG_PRESETS[args.preset]
    dtypes = DTYPE_PRESETS.get(args.in_dtype, [args.in_dtype])

    total_jobs = len(configs) * len(dtypes)

    print("=" * 72)
    print("FlyDSL Preshuffle GEMM AOT Pre-compilation")
    print("=" * 72)
    print(
        f"  Preset:       {args.preset} ({len(configs)} shapes x {len(dtypes)} dtypes = {total_jobs} jobs)"
    )
    print(f"  in_dtype:     {dtypes}")
    print(f"  lds_stage:    {args.lds_stage}")
    print(f"  Cache dir:    {cache_dir}")
    print(f"  Target arch:  {arch}")
    print(f"  run_kernel:   {args.run_kernel}")
    print("=" * 72)

    total_t0 = time.time()
    results = []
    job_idx = 0

    for dt in dtypes:
        print(f"\n--- dtype: {dt} ---")
        for cfg in configs:
            job_idx += 1
            M, N, K, tile_m, tile_n, tile_k = cfg
            print(f"\n[{job_idx}/{total_jobs}] Compiling ({dt})...")
            r = compile_one_config(
                M=M,
                N=N,
                K=K,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                in_dtype=dt,
                lds_stage=args.lds_stage,
                run_kernel=args.run_kernel,
            )
            results.append(r)

    total_elapsed = time.time() - total_t0

    ok = sum(1 for r in results if r["compile_time"] is not None)
    fail = sum(1 for r in results if r["compile_time"] is None)

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Total time:   {total_elapsed:.1f}s")
    print(f"  Compiled:     {ok} ok, {fail} failed")
    print(f"  Cache dir:    {cache_dir}")

    if args.test_bad_tile:
        test_bad_tile_error()

    print()

    exit_code = 0
    if fail > 0:
        print("Some compilations failed. Check output above for details.")
        exit_code = 1
    else:
        print("All compilations succeeded. Cache is ready.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# pytest interface
# ---------------------------------------------------------------------------
import pytest  # noqa: E402


@pytest.mark.parametrize("in_dtype", ["fp8", "int8"])
@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k",
    SMALL_CONFIGS,
    ids=[f"{c[0]}x{c[1]}x{c[2]}" for c in SMALL_CONFIGS],
)
def test_aot_compile(M, N, K, tile_m, tile_n, tile_k, in_dtype):
    r = compile_one_config(
        M=M, N=N, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        in_dtype=in_dtype,
        run_kernel=True,
    )
    assert r["compile_time"] is not None, f"Compilation failed for {r['shape']}"


def test_bad_tile_rejected():
    test_bad_tile_error()
