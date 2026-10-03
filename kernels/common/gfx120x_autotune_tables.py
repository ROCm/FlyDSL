# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Measured gfx120x host-dispatch / tile / BT tables (R9700, HIP 7.17.26374).

Measurement method: FlyDSL ``do_bench`` warm+median; HIP baseline in a separate process
when comparing speed. Tables are the **wired defaults** for
``*_auto`` / ``pick_*`` / ``_block_threads`` host gates — not docs-only.

Override without editing constants: ``FLYDSL_DISPATCH_MODE=force_flydsl|force_hip|auto``.

Measured date: 2026-09-30.
"""

# --- int8 W8A16 vs iu8 ---
# Pareto: prefer iu8 when K >= this (quant+iu8 vs W8A16 on product grid).
DEFAULT_K_IU8_MIN = 256

# --- fused int8 act-quant+mm vs device-quant+iu8 ---
# Fused is faster for K<=128 on measured mid shapes; K>=256 → dq+iu8 (incl. (64,256,256)).
DEFAULT_K_FUSED_KERNEL_MAX = 128

# --- AWQ GEMV tile pick (documented in pick_awq_gemv_tiles) ---
# M=1: n_tile 1/4 by N; M in (1,4]: n_tile=1, block_threads=64

# --- SVDQuant n_tile ---
# M=1: 1/4/8 by N; else 8

# --- ConvRot ---
# default linear_dtype=int4 (native); unpack int8 fallback if native gate fails

# --- FA BLOCK_M soft-gap ---
# self: q<=128 → 64 else 128; cross: q<=96 → 16; q<=128 → 32; else 128


# --- AdaLN block_threads (measured vs HIP adaln, B=77) ---
# N→BT winners: 64→32, 128→32, 256→256, 512→512, 1024→128, 3072→512
def pick_adaln_block_threads(n: int) -> int:
    """Return the measured AdaLN block size for this row width."""
    n = int(n)
    if n <= 128:
        return 32
    if n <= 256:
        return 256
    if n <= 512:
        return 512
    if n <= 1024:
        return 128
    if n >= 12288:
        return 512 if n < 24576 else 1024
    # mid-large (e.g. 3072 flux): 512 beat default 256
    if n >= 2048:
        return 512
    return 256


# --- RoPE pair-launch block_threads (pair-count heuristic, ported 2026-09-30) ---
# n_pairs_total = B*dim1*dim2*(HD//2).
def pick_rope_block_threads(n_pairs_total: int) -> int:
    """Return the measured RoPE block size for this pair count."""
    n = int(n_pairs_total)
    if n < 4096:
        return 256
    if n < 32768:
        return 512
    return 1024


# --- RMS-RoPE block_threads (qk-fused do_bench vs HIP rms_rope w/ q_scale; 2026-09-30) ---
# Measured BSHD=(1,256,16,HD) bf16: HD<=1024 best BT=64; HD=3072 best BT=512 (ties or beats HIP).
# HD→BT: <=1024→64; >=2048→512; huge→512/1024
def pick_rms_rope_block_threads(hd: int) -> int:
    """Return the measured RMSNorm+RoPE block size for this head dim."""
    hd = int(hd)
    if hd <= 1024:
        return 64
    if hd >= 24576:
        return 1024
    if hd >= 12288:
        return 512
    if hd >= 2048:
        return 512
    return 256


# --- int8 rowwise quant BT (heuristic confirmed ties or beats HIP on the grid) ---
# --- asym W4A8 dequant block_threads (measured vs local heuristic; 2026-09-30) ---
# K in {128..4096}, N=256: BT=128 wins (BT=256 ~3× slower). Tiny K=128: 32≈128.
def pick_asym_w4a8_block_threads(k: int) -> int:
    """Return the measured asymmetric W4A8 block size."""
    del k  # size-insensitive on measured grid; keep signature for call-site symmetry
    return 128


def pick_quantize_int8_rowwise_block_threads(k: int) -> int:
    """Return the measured rowwise int8 quant block size."""
    k = int(k)
    if k <= 256:
        return 32
    if k <= 512:
        return 64
    if k >= 24576:
        return 1024
    if k >= 12288:
        return 512
    return 256


def table_doc() -> dict:
    """Return the measured gfx120x size-dispatch tables as a dict."""
    return {
        "hw": "gfx1201 R9700",
        "hip_stamp": "7.17.26374",
        "measured_date": "2026-09-30",
        "DEFAULT_K_IU8_MIN": DEFAULT_K_IU8_MIN,
        "DEFAULT_K_FUSED_KERNEL_MAX": DEFAULT_K_FUSED_KERNEL_MAX,
        "pick_adaln_block_threads": "N<=128→32; <=256→256; <=512→512; <=1024→128; >=2048→512",
        "pick_rope_block_threads": "n_pairs_total<4096→256; <32768→512; else 1024",
        "pick_rms_rope_block_threads": "HD<=1024→64; >=2048→512",
        "pick_asym_w4a8_block_threads": "K→128 (measured; 256 slower)",
        "pick_quantize_int8_rowwise_block_threads": "K<=256→32; <=512→64; else 256 (+huge)",
        "dispatch_mode_env": "FLYDSL_DISPATCH_MODE=auto|force_flydsl|force_hip",
    }


__all__ = [
    "DEFAULT_K_IU8_MIN",
    "DEFAULT_K_FUSED_KERNEL_MAX",
    "pick_adaln_block_threads",
    "pick_rope_block_threads",
    "pick_rms_rope_block_threads",
    "pick_asym_w4a8_block_threads",
    "pick_quantize_int8_rowwise_block_threads",
    "table_doc",
]
