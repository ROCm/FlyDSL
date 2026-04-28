#!/usr/bin/env python3
"""Launcher for FlyDSL A8W8 Blockscale GEMM kernel.

Edit the CONFIG block below — no CLI args. Run with:
    python run_gemm_a8w8_blockscale.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — edit these, then run the script
# ═══════════════════════════════════════════════════════════════════════════

# Problem shape
M = 32
N = 5120
K = 2880

# Scale granularity (must match how the scales were quantized)
SCALE_BLOCK_K = 128
SCALE_BLOCK_N = 128

# Tile dims (tile_k must equal SCALE_BLOCK_K for Phase 1)
TILE_M = 32 #224
TILE_N = 64
TILE_K = 512

# Warp grid
M_WARP = 2
N_WARP = 2

# Pipeline / perf knobs
NUM_BUFFERS = 3
WAVES_PER_EU = None
L2_PREFETCH_DISTANCE = 0
USE_TDM_STORE = True


# Kernel variant: "reg_preload" (operand frags loop-carried) or
# "no_op_preload" (operand frags loaded fresh per sub-stage, ~256 VGPR
# cheaper). Both variants share the 2-sub-stage-ahead scale prefetch.
VARIANT = "reg_preload"

# Output dtype ("bf16" / "fp16" / "f32")
OUT_DTYPE = "bf16"

# Epilogue: True → LDS-staged async TDM store (uses tensorcnt, single
# dispatch per warp). False → per-acc buffer_store_b128 (legacy path).

# ═══════════════════════════════════════════════════════════════════════════

import os
import sys
import importlib.util

# File now lives at FlyDSL/kernels/run_blockscale/ — go up two levels to hit FlyDSL root.
_FLYDSL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _FLYDSL_ROOT not in sys.path:
    sys.path.insert(0, _FLYDSL_ROOT)

import flydsl  # noqa: E402,F401 — preload comgr before torch/HIP
import torch

# Load the kernel module by file path (avoids package import quirks).
_spec = importlib.util.spec_from_file_location(
    "gemm_a8w8_blockscale",
    os.path.join(_FLYDSL_ROOT, "kernels/gemm_a8w8_blockscale.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compile_gemm_a8w8_blockscale = _mod.compile_gemm_a8w8_blockscale


def main():
    # Derived shapes
    scale_k = (K + SCALE_BLOCK_K - 1) // SCALE_BLOCK_K
    scale_n = (N + SCALE_BLOCK_N - 1) // SCALE_BLOCK_N

    # Padding for non-aligned problem sizes — our kernel requires
    # K % TILE_K == 0 and treats M/N via bounds-checked stores.
    K_padded = ((K + TILE_K - 1) // TILE_K) * TILE_K
    N_padded = ((N + TILE_N - 1) // TILE_N) * TILE_N

    # Torch dtypes
    _out_torch = {"bf16": torch.bfloat16,
                  "fp16": torch.float16,
                  "f32": torch.float32}[OUT_DTYPE]

    # Build inputs
    # FP8 E4M3FN on gfx1250 / MI350. If this fails on your torch build,
    # you may need torch.float8_e4m3fnuz (MI300) or a newer torch.
    fp8_dtype = torch.float8_e4m3fn

    # Use small values so the FP8 quantization doesn't saturate.
    x = (torch.rand((M, K_padded), device="cuda", dtype=torch.float32) / 10).to(fp8_dtype)
    w = (torch.rand((N, K_padded), device="cuda", dtype=torch.float32) / 10).to(fp8_dtype)
    x_scale = torch.rand((M, scale_k), device="cuda", dtype=torch.float32)
    w_scale = torch.rand((scale_n, scale_k), device="cuda", dtype=torch.float32)

    y = torch.empty((M, N_padded), device="cuda", dtype=_out_torch)

    print(
        f"Compiling A8W8 blockscale GEMM:\n"
        f"  shape:      M={M}, N={N}({N_padded}), K={K}({K_padded})\n"
        f"  scales:     scale_block_k={SCALE_BLOCK_K}, scale_block_n={SCALE_BLOCK_N}\n"
        f"              x_scale=({M},{scale_k}), w_scale=({scale_n},{scale_k})\n"
        f"  tiles:      tile_m={TILE_M}, tile_n={TILE_N}, tile_k={TILE_K}\n"
        f"  warps:      m_warp={M_WARP}, n_warp={N_WARP}\n"
        f"  pipeline:   num_buffers={NUM_BUFFERS}, waves_per_eu={WAVES_PER_EU}, "
        f"l2_prefetch={L2_PREFETCH_DISTANCE}\n"
        f"  variant:    {VARIANT}\n"
        f"  out_dtype:  {OUT_DTYPE}\n"
        f"  tdm_store:  {USE_TDM_STORE}"
    )

    launch_fn = compile_gemm_a8w8_blockscale(
        K=K_padded,
        N=N_padded,
        tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K,
        m_warp=M_WARP, n_warp=N_WARP,
        scale_block_k=SCALE_BLOCK_K,
        scale_block_n=SCALE_BLOCK_N,
        num_buffers=NUM_BUFFERS,
        waves_per_eu=WAVES_PER_EU,
        l2_prefetch_distance=L2_PREFETCH_DISTANCE,
        out_dtype=OUT_DTYPE,
        variant=VARIANT,
        use_tdm_store=USE_TDM_STORE,
    )

    print("Launching kernel...")
    stream = torch.cuda.current_stream().cuda_stream
    launch_fn(y, x, w, x_scale, w_scale, M, N_padded, stream=stream)
    torch.cuda.synchronize()

    if N_padded != N:
        y = y[:, :N]

    print(f"Output shape: {y.shape}, dtype: {y.dtype}")
    print(f"Output sample: y[0, :8] = {y[0, :8].tolist()}")

    # Machine-parseable stats line for perf_a8w8_blockscale.sh to grep.
    # Keep this format stable — the shell script uses regex to parse it.
    _elem_bytes_out = {"bf16": 2, "fp16": 2, "f32": 4}[OUT_DTYPE]
    print(
        f"PERF_STATS M={M} N={N} K={K_padded} "
        f"scale_k={scale_k} scale_n={scale_n} elem_bytes_out={_elem_bytes_out}"
    )
    print("Done.")


if __name__ == "__main__":
    main()
