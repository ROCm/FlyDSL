#!/usr/bin/env python3
"""AOT pre-compilation example for MOE kernels.

Pre-compiles MOE GEMM stage1 + stage2 + reduction kernels for specified
configurations and stores the compiled binaries in a cache directory.

Usage:
    # Pre-compile with default configurations (auto-detect GPU arch)
    python tests/python/examples/aot_example.py

    # Pre-compile for a specific arch (cross-compilation, no GPU needed)
    COMPILE_ONLY=1 ARCH=gfx942 python tests/python/examples/aot_example.py

    # Custom cache directory
    FLIR_CACHE_DIR=/my/cache python tests/python/examples/aot_example.py

    # Pre-compile and verify by running kernels on GPU
    python tests/python/examples/aot_example.py --run_kernel

    # Later, at runtime, load from cache:
    FLIR_CACHE_DIR=/my/cache python my_app.py

Environment variables:
    FLIR_CACHE_DIR      Cache directory (default: ~/.cache/flydsl)
    COMPILE_ONLY        Set to "1" to skip executor creation (no GPU needed)
    ARCH                Target GPU architecture (e.g. gfx942, gfx950)
    FLIR_CACHE_REBUILD  Set to "1" to force recompilation even if cached
"""

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernels.moe_gemm_2stage import (
    compile_moe_gemm1,
    compile_moe_gemm2,
    compile_moe_gemm2_ex,
    compile_moe_reduction,
    MoeGemm2Mode,
)


def _run_kernel_stage1(
    exe,
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    in_dtype: str,
    doweight_stage1: bool,
):
    """Launch the compiled stage1 kernel with random data to verify it runs."""
    import torch
    from tests.utils import pertoken_quant, shuffle_weight
    from flydsl.runtime.device import get_rocm_arch

    ARCH = get_rocm_arch()
    DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz

    device = torch.device("cuda")
    torch.manual_seed(0)

    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32)
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32)

    score = torch.randn((tokens, experts), device=device, dtype=torch.float32)
    topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)

    from tests.kernels.test_moe_gemm import build_routing_buffers
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, sorted_size, blocks = \
        build_routing_buffers(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            experts=experts,
            model_dim=model_dim,
            tile_m=tile_m,
            moe_sort_mode="torch",
        )

    is_int8 = in_dtype in ("int8", "int8smooth", "int4")
    if in_dtype == "fp8":
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=DTYPE_FP8)
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=DTYPE_FP8)
    elif in_dtype == "fp16":
        x_q = x_fp32.to(torch.float16)
        w1_q = w1_fp32.to(torch.float16)
        scale_x, scale_w1 = None, None
    elif in_dtype in ("int8", "int4"):
        dtypeMax = 7 if in_dtype == "int4" else None
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.int8, **({"dtypeMax": dtypeMax} if dtypeMax else {}))
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.int8, **({"dtypeMax": dtypeMax} if dtypeMax else {}))
    else:
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.int8)
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.int8)

    w1_shuffled_flat = shuffle_weight(w1_q).view(experts * (2 * inter_dim), model_dim)
    if in_dtype == "int4":
        from tests.kernels.test_moe_gemm import _pack_shuffled_int8_to_packed_int4_no_perm
        w_kernel = _pack_shuffled_int8_to_packed_int4_no_perm(w1_shuffled_flat).contiguous()
    else:
        w_kernel = w1_shuffled_flat.contiguous().view(experts * (2 * inter_dim), model_dim)

    x_q = x_q.contiguous().view(tokens, model_dim)
    scale_x_1d = torch.empty((0,), device=device, dtype=torch.float32) if scale_x is None else scale_x.view(-1).contiguous()
    scale_w1_flat = None if scale_w1 is None else scale_w1.view(experts * (2 * inter_dim), 1)
    scale_w1_1d = torch.empty((0,), device=device, dtype=torch.float32) if scale_w1_flat is None else scale_w1_flat.view(-1).contiguous()
    sorted_weights_1d = sorted_weights.contiguous().view(-1)

    out = torch.empty((tokens, topk, inter_dim), device=device, dtype=torch.float16)

    stream_ptr = torch.cuda.current_stream().cuda_stream
    exe(
        out,
        x_q,
        w_kernel,
        scale_x_1d,
        scale_w1_1d,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights_1d,
        num_valid_ids,
        tokens,
        inter_dim,
        model_dim,
        int(blocks),
        stream_ptr,
    )
    torch.cuda.synchronize()
    print(f"    output shape={tuple(out.shape)}, "
          f"max={out.float().abs().max().item():.4f}, "
          f"mean={out.float().abs().mean().item():.4f}")

# ---------------------------------------------------------------------------
# Model configurations
#
# Each entry: (in_dtype, tokens, model_dim, inter_dim, experts, topk,
#              tile_m, tile_n1, tile_k1, tile_n2, tile_k2)
#
# Shapes from scripts/run_benchmark.sh, compiled for both fp8 and int8.
# ---------------------------------------------------------------------------

BENCHMARK_CONFIGS = [
    # tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n1, tile_k1, tile_n2, tile_k2
    (32768, 8192, 8192, 16,  4, 64, 128, 128, 256, 128),
    (64,    6144, 1024, 128, 8, 16, 64,  256, 64,  256),
]

# Small shape for quick smoke-test
SMALL_CONFIGS = [
    (256, 1024, 256, 4, 2, 32, 128, 256, 256, 128),
]

# Invalid tile size to verify the compiler produces a clear error.
# tile_k=17 is not a power-of-2 / not aligned to MFMA requirements.
BAD_TILE_CONFIGS = [
    (256, 1024, 256, 4, 2, 32, 128, 17, 256, 128),
]

CONFIG_PRESETS = {
    "benchmark": BENCHMARK_CONFIGS,
    "small": SMALL_CONFIGS,
    "bad_tile": BAD_TILE_CONFIGS,
}

# dtype combinations to compile for each shape
DTYPE_PRESETS = {
    "fp8": ["fp8"],
    "int8": ["int8"],
    "both": ["fp8", "int8"],
}


def compile_one_config(
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n1: int,
    tile_k1: int,
    tile_n2: int,
    tile_k2: int,
    in_dtype: str = "fp8",
    out_dtype: str = "f16",
    doweight_stage1: bool = False,
    gemm2_mode: str = "atomic",
    run_kernel: bool = False,
) -> dict:
    """Compile one MOE configuration (stage1 + stage2 + optional reduction).

    When run_kernel=True, also launches the kernel with random data to verify
    that the compiled binary actually runs on the GPU.

    Returns a dict with timing info.
    """
    shape_str = (
        f"t={tokens} dim={model_dim}x{inter_dim} "
        f"e={experts} k={topk} "
        f"tile=({tile_m},{tile_n1},{tile_k1})/({tile_m},{tile_n2},{tile_k2})"
    )
    result = {"shape": shape_str, "stage1": None, "stage2": None, "reduce": None}

    # Stage 1
    t0 = time.time()
    try:
        exe_s1 = compile_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n1,
            tile_k=tile_k1,
            doweight_stage1=doweight_stage1,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
        )
        elapsed = time.time() - t0
        result["stage1"] = elapsed
        print(f"  [OK] stage1  {elapsed:6.1f}s  {shape_str}")
        if run_kernel and exe_s1 is not None:
            _run_kernel_stage1(
                exe_s1, tokens, model_dim, inter_dim, experts, topk,
                tile_m, in_dtype, doweight_stage1,
            )
    except Exception as e:
        print(f"  [FAIL] stage1  {shape_str}: {e}")

    # Stage 2
    t0 = time.time()
    try:
        if gemm2_mode == "reduce":
            compile_moe_gemm2_ex(
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=experts,
                topk=topk,
                tile_m=tile_m,
                tile_n=tile_n2,
                tile_k=tile_k2,
                doweight_stage2=not doweight_stage1,
                in_dtype=in_dtype,
                out_dtype=out_dtype,
                mode=MoeGemm2Mode.REDUCE,
            )
        else:
            compile_moe_gemm2(
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=experts,
                topk=topk,
                tile_m=tile_m,
                tile_n=tile_n2,
                tile_k=tile_k2,
                doweight_stage2=not doweight_stage1,
                in_dtype=in_dtype,
                out_dtype=out_dtype,
                accumulate=True,
            )
        elapsed = time.time() - t0
        result["stage2"] = elapsed
        print(f"  [OK] stage2  {elapsed:6.1f}s  {shape_str}")
    except Exception as e:
        print(f"  [FAIL] stage2  {shape_str}: {e}")

    # Reduction kernel (only for reduce mode)
    if gemm2_mode == "reduce":
        t0 = time.time()
        try:
            compile_moe_reduction(
                topk=topk,
                model_dim=model_dim,
                dtype_str=out_dtype,
            )
            elapsed = time.time() - t0
            result["reduce"] = elapsed
            print(f"  [OK] reduce  {elapsed:6.1f}s  {shape_str}")
        except Exception as e:
            print(f"  [FAIL] reduce  {shape_str}: {e}")

    return result


def test_bad_tile_error():
    """Verify that an unsupported tile size produces a clear compile error."""
    print("\n" + "=" * 72)
    print("Testing invalid tile size (expecting compile error)")
    print("=" * 72)

    for cfg in BAD_TILE_CONFIGS:
        tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n1, tile_k1, tile_n2, tile_k2 = cfg
        shape_str = (
            f"t={tokens} dim={model_dim}x{inter_dim} "
            f"e={experts} k={topk} "
            f"tile=({tile_m},{tile_n1},{tile_k1})"
        )
        try:
            compile_moe_gemm1(
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=experts,
                topk=topk,
                tile_m=tile_m,
                tile_n=tile_n1,
                tile_k=tile_k1,
                doweight_stage1=False,
                in_dtype="fp8",
                out_dtype="f16",
            )
            print(f"  [UNEXPECTED] No error raised for bad tile: {shape_str}")
            return False
        except Exception as e:
            print(f"  [OK] Correctly rejected bad tile: {shape_str}")
            print(f"       Error: {type(e).__name__}: {e}")
            return True


def main():
    parser = argparse.ArgumentParser(
        description="AOT pre-compile MOE kernels into FLIR cache",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="benchmark",
        choices=list(CONFIG_PRESETS.keys()),
        help="Configuration preset (default: benchmark)",
    )
    parser.add_argument(
        "--in_dtype",
        type=str,
        default="both",
        choices=list(DTYPE_PRESETS.keys()) + ["fp16", "int8smooth", "int4"],
        help="Input data type(s): 'both' compiles fp8+int8 (default: both)",
    )
    parser.add_argument(
        "--out_dtype",
        type=str,
        default="f16",
        choices=["f16", "bf16", "f32"],
        help="Output data type (default: f16)",
    )
    parser.add_argument(
        "--gemm2_mode",
        type=str,
        default="atomic",
        choices=["atomic", "reduce"],
        help="Stage2 accumulation mode (default: atomic)",
    )
    parser.add_argument(
        "--doweight_stage1",
        action="store_true",
        help="Apply routing weights in stage1 instead of stage2",
    )
    parser.add_argument(
        "--run_kernel",
        action="store_true",
        help="After compilation, launch each stage1 kernel with random data to verify it runs",
    )
    parser.add_argument(
        "--test_bad_tile",
        action="store_true",
        help="Also test that an invalid tile size is properly rejected",
    )
    args = parser.parse_args()

    cache_dir = os.environ.get("FLIR_CACHE_DIR", "~/.cache/flydsl")
    arch = os.environ.get("ARCH", "(auto-detect)")
    compile_only = os.environ.get("COMPILE_ONLY", "0") == "1"

    configs = CONFIG_PRESETS[args.preset]
    dtypes = DTYPE_PRESETS.get(args.in_dtype, [args.in_dtype])

    total_jobs = len(configs) * len(dtypes)

    print("=" * 72)
    print("FLIR MOE AOT Pre-compilation")
    print("=" * 72)
    print(f"  Preset:       {args.preset} ({len(configs)} shapes x {len(dtypes)} dtypes = {total_jobs} jobs)")
    print(f"  in_dtype:     {dtypes}")
    print(f"  out_dtype:    {args.out_dtype}")
    print(f"  gemm2_mode:   {args.gemm2_mode}")
    print(f"  Cache dir:    {cache_dir}")
    print(f"  Target arch:  {arch}")
    print(f"  COMPILE_ONLY: {compile_only}")
    print(f"  run_kernel:   {args.run_kernel}")
    print("=" * 72)

    total_t0 = time.time()
    results = []
    job_idx = 0

    for dt in dtypes:
        print(f"\n--- dtype: {dt} ---")
        for cfg in configs:
            job_idx += 1
            tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n1, tile_k1, tile_n2, tile_k2 = cfg
            print(f"\n[{job_idx}/{total_jobs}] Compiling ({dt})...")
            r = compile_one_config(
                tokens=tokens,
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=experts,
                topk=topk,
                tile_m=tile_m,
                tile_n1=tile_n1,
                tile_k1=tile_k1,
                tile_n2=tile_n2,
                tile_k2=tile_k2,
                in_dtype=dt,
                out_dtype=args.out_dtype,
                doweight_stage1=args.doweight_stage1,
                gemm2_mode=args.gemm2_mode,
                run_kernel=args.run_kernel,
            )
            results.append(r)

    total_elapsed = time.time() - total_t0

    # Summary
    ok_s1 = sum(1 for r in results if r["stage1"] is not None)
    ok_s2 = sum(1 for r in results if r["stage2"] is not None)
    fail_s1 = sum(1 for r in results if r["stage1"] is None)
    fail_s2 = sum(1 for r in results if r["stage2"] is None)

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Total time:   {total_elapsed:.1f}s")
    print(f"  Stage1:       {ok_s1} ok, {fail_s1} failed")
    print(f"  Stage2:       {ok_s2} ok, {fail_s2} failed")
    print(f"  Cache dir:    {cache_dir}")

    # Bad tile test
    if args.test_bad_tile:
        bad_tile_ok = test_bad_tile_error()
        if not bad_tile_ok:
            print("\n  Bad tile test: FAILED (expected error was not raised)")
        else:
            print("\n  Bad tile test: PASSED")

    print()

    exit_code = 0
    if fail_s1 + fail_s2 > 0:
        print("Some compilations failed. Check output above for details.")
        exit_code = 1
    else:
        print("All compilations succeeded. Cache is ready.")

    if args.test_bad_tile and not bad_tile_ok:
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
