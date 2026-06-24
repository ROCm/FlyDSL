#!/usr/bin/env python3
"""Launcher for FlyDSL A8W8 Blockscale GEMM kernel.

Edit the CONFIG block below — no CLI args. Run with:
    python run_gemm_a8w8_blockscale.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — edit these, then run the script
# ═══════════════════════════════════════════════════════════════════════════

# Problem shape
M = 1024 #896
N = 2048
K = 2880

# Scale granularity (must match how the scales were quantized)
SCALE_BLOCK_K = 128
SCALE_BLOCK_N = 128

# Tile dims (tile_k must equal SCALE_BLOCK_K for Phase 1)
TILE_M = 256 #224
TILE_N = 256
TILE_K = 128

# Warp grid
M_WARP = 2
N_WARP = 2

# Pipeline / perf knobs
NUM_BUFFERS = 3
WAVES_PER_EU = None
L2_PREFETCH_DISTANCE = 0
USE_TDM_STORE = False
# Experimental: forwards to AMDGPU LLVM `amdgpu-loop-carried-load-percent`
# function attribute via passthrough. Set to 0 to try less-conservative
# scheduling of loop-carried VGPRs. None disables (no attribute set).
LOOP_CARRIED_LOAD_PERCENT = 0
# Kernarg preload: marks each kernel arg as `inreg` so the AMDGPU backend
# preloads them into user SGPRs at dispatch (no s_load + s_wait_kmcnt at
# wave entry). Saves ~1786 cycles of prologue stall on the gfx1250 sim.
KERNARG_PRELOAD = True
# Kernel variant: "reg_preload" (primary) or "manual".
VARIANT = "manual"
# Manual-only: reuseB on consecutive WMMAs sharing the A-operand (a_cur, n>0) →
# matrix_b_reuse, skips re-reading those VGPRs on gfx1250.
WMMA_OPERAND_REUSE = True

# Manual-only EXPERIMENT: emit the steady main-loop workgroup barrier as a bare
# s_barrier_signal/wait (no release/acquire fences) so the scheduler may hoist the
# prefetch ds_loads up into the WMMA shadows. Verify via trace — a passing unit
# test is NOT proof (cross-wave race is timing-dependent). False = stock gpu.barrier().
USE_MANUAL_BARRIER = True

# Manual-only: fully preload the A panel (A0..A_{M-1}) in the prologue and refill
# each A row's registers IN PLACE for tile T+1 — timed one compute-row after the
# row's last read, so the reload never WARs a live WMMA read. Single A set (no
# second buffer); B stays double-buffered. Requires scales_per_tile==1 (tile_k ==
# scale_block_k). A ds_load win, not a VGPR win — verify the A-refill distance in
# the trace (a passing unit test is NOT proof).
A_RESIDENT_REFILL = True

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

from kernels.util.shuffle import preshuffle_fp8_weights_gfx1250

_warmup_spec = importlib.util.spec_from_file_location(
    "warmup_read_kernel",
    os.path.join(_FLYDSL_ROOT, "kernels/warmup_read_kernel.py"),
)
_warmup_mod = importlib.util.module_from_spec(_warmup_spec)
_warmup_spec.loader.exec_module(_warmup_mod)
warmup_read = _warmup_mod.warmup_read


def _as_f32_1d(t: torch.Tensor) -> torch.Tensor:
    """Return a 1-D float32 view over `t`'s storage (no copy). Total
    bytes must be a multiple of 4."""
    n_bytes = t.numel() * t.element_size()
    assert n_bytes % 4 == 0, f"tensor not 4-byte aligned: {n_bytes} bytes"
    n_f32 = n_bytes // 4
    return torch.empty(0, dtype=torch.float32, device=t.device).set_(
        t.untyped_storage(), 0, (n_f32,), (1,)
    )


def _run_warmup(inputs, stream):
    """Touch every 4-byte word of each input tensor so the L2 / TLB are
    warm before the real GEMM launches. The scratch store keeps the
    loads alive through DCE."""
    max_n = max(_as_f32_1d(t).numel() for t in inputs)
    scratch = torch.empty(max_n, dtype=torch.float32, device=inputs[0].device)
    for t in inputs:
        flat = _as_f32_1d(t)
        n = flat.numel()
        warmup_read(flat, scratch, n, n, stream=stream)


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
        f"  tdm_store:  {USE_TDM_STORE}\n"
        f"  manual_barrier: {USE_MANUAL_BARRIER}"
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
        loop_carried_load_percent=LOOP_CARRIED_LOAD_PERCENT,
        kernarg_preload=KERNARG_PRELOAD,
        wmma_operand_reuse=WMMA_OPERAND_REUSE,
        use_manual_barrier=USE_MANUAL_BARRIER,
        a_resident_refill=A_RESIDENT_REFILL,
    )

    print("Launching kernel...")
    stream = torch.cuda.current_stream().cuda_stream

    # Preshuffle W before warmup so warmup and timed launch read the same bytes.
    w = preshuffle_fp8_weights_gfx1250(w)

    print("Running warmup launch (same kernel + configs as timed launch)...")
    launch_fn(y, x, w, x_scale, w_scale, M, N_padded, stream=stream)

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
