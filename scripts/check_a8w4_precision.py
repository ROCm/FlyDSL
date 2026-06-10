#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A8W4 GEMM precision check: current kernel vs baseline kernel vs f32 reference.

For each config, runs BOTH the current pipeline and the baseline pipeline on
identical inputs/scales/preshuffle data, then compares each against the f32
reference. Prints a per-config side-by-side table and a final summary.

Run via scripts/run_check_a8w4_precision.sh (sets env + triton path).
"""
from __future__ import annotations

import sys
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

import flydsl.compiler as flyc
from flydsl.runtime.device import get_rocm_arch
from kernels.gemm_fp8fp4_gfx1250 import compile_mxscale_gemm as compile_current
from kernels.gemm_fp8fp4_gfx1250_baseline import compile_mxscale_gemm as compile_baseline
from tests.kernels.test_gemm_fp8fp4_gfx1250 import (
    SCALE_BLOCK,
    _get_padded_problem_shape,
    _pad_mxscale_inputs,
    fp4_utils,
    preshuffle_e8m0_scale,
    random_fp8_data,
    reference_a8w4_gemm,
)


# (M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers)
CONFIGS = [
    (1, 12288, 3072, 16, 64, 512, 1, 4, 4),
    (64, 12288, 3072, 16, 64, 512, 1, 4, 4),
]


def _run_kernel(compile_fn, data_format, a, b, a_scale, b_scale, padded_shape, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, M, N):
    """Compile & run a single kernel variant, return c_out[:M,:N] on CPU as float."""
    padded_m = padded_shape["M"]
    padded_n = padded_shape["N"]
    padded_k = padded_shape["K"]

    skt = tile_k // SCALE_BLOCK
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp

    a_s = preshuffle_e8m0_scale(a_scale.clone(), warp_tile_m, scale_k_per_tile=skt, coalesced=False)
    b_s = preshuffle_e8m0_scale(b_scale.clone(), warp_tile_n, scale_k_per_tile=skt, coalesced=False)

    K_packed = padded_k // padded_shape["pack_b"]
    b_shuf = fp4_utils.preshuffle_b_16x16(b.clone(), padded_n, K_packed)

    a_gpu = a.cuda().contiguous()
    b_gpu = b_shuf.cuda().contiguous()
    as_gpu = a_s.cuda().contiguous()
    bs_gpu = b_s.cuda().contiguous()
    c_gpu = torch.zeros(padded_m, padded_n, dtype=torch.bfloat16, device="cuda")

    launch_fn = compile_fn(
        data_format=data_format,
        N=padded_n,
        K=padded_k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=num_buffers,
        use_tdm_store=True,
        out_dtype="bf16",
        wave_specialized_tdm=True,
        l2_prefetch_distance=2,
    )

    flyc.compile(
        launch_fn,
        c_gpu.contiguous(),
        a_gpu,
        b_gpu,
        as_gpu,
        bs_gpu,
        padded_m,
        padded_n,
        padded_k,
        padded_n,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    return c_gpu[:M, :N].to(torch.bfloat16).cpu().float()


def _metrics(out_f, ref_f):
    """Compute precision metrics between kernel output and f32 reference."""
    diff = (out_f - ref_f).abs()
    cos = torch.nn.functional.cosine_similarity(
        out_f.flatten().unsqueeze(0).double(),
        ref_f.flatten().unsqueeze(0).double(),
    ).item()
    return {
        "cos": cos,
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
    }


def main():
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        print(f"Requires gfx1250, got {arch}")
        sys.exit(1)

    rows = []

    for cfg in CONFIGS:
        M, N, K, tm, tn, tk, mw, nw, nb = cfg
        tag = f"M{M} N{N} K{K} t({tm},{tn},{tk}) w{mw}x{nw} b{nb}"
        print(f"\n{'='*72}\n{tag}\n{'='*72}", flush=True)

        torch.manual_seed(0)
        a = random_fp8_data(M, K)
        b = fp4_utils.random_fp4_packed(N, K)
        a_scale = fp4_utils.random_e8m0(M, K // SCALE_BLOCK)
        b_scale = fp4_utils.random_e8m0(N, K // SCALE_BLOCK)

        ref = reference_a8w4_gemm(a, b, a_scale, b_scale, M, N, K)
        ref_f = ref.to(torch.bfloat16).float()

        padded_shape = _get_padded_problem_shape("a8w4", M, N, K, tm, tn, tk, 1)
        a_pad, b_pad, as_pad, bs_pad = _pad_mxscale_inputs(a, b, a_scale, b_scale, padded_shape)

        row = {"tag": tag}

        # --- Current kernel ---
        try:
            cur_f = _run_kernel(compile_current, "a8w4", a_pad, b_pad, as_pad, bs_pad, padded_shape, tm, tn, tk, mw, nw, nb, M, N)
            row["cur"] = _metrics(cur_f, ref_f)
            print(f"  Current:  cos={row['cur']['cos']:.6f}  max_diff={row['cur']['max_diff']:.4f}  mean_diff={row['cur']['mean_diff']:.4f}")
        except Exception as e:
            row["cur"] = None
            print(f"  Current:  ERROR ({type(e).__name__}: {e})")

        # --- Baseline kernel ---
        try:
            base_f = _run_kernel(compile_baseline, "a8w4", a_pad, b_pad, as_pad, bs_pad, padded_shape, tm, tn, tk, mw, nw, nb, M, N)
            row["base"] = _metrics(base_f, ref_f)
            print(f"  Baseline: cos={row['base']['cos']:.6f}  max_diff={row['base']['max_diff']:.4f}  mean_diff={row['base']['mean_diff']:.4f}")
        except Exception as e:
            row["base"] = None
            print(f"  Baseline: ERROR ({type(e).__name__}: {e})")

        # --- Current vs baseline direct comparison ---
        if row["cur"] is not None and row["base"] is not None:
            cvb = _metrics(cur_f, base_f)
            row["cvb"] = cvb
            print(f"  Cur↔Base: cos={cvb['cos']:.6f}  max_diff={cvb['max_diff']:.4f}  mean_diff={cvb['mean_diff']:.4f}")

        rows.append(row)

    # ---- Summary table ----
    print(f"\n\n{'='*100}")
    print("  A8W4 Precision Summary: Current vs Baseline (both compared to f32 reference)")
    print(f"{'='*100}")
    hdr = f"  {'Config':<42} {'':^5} {'Cosine':>8} {'MaxDiff':>9} {'MeanDiff':>9}"
    print(hdr)
    print(f"  {'-'*42} {'-'*5} {'-'*8} {'-'*9} {'-'*9}")

    for row in rows:
        tag = row["tag"]
        for label, key in [("cur", "cur"), ("base", "base"), ("c↔b", "cvb")]:
            m = row.get(key)
            if key == "cvb":
                tag_col = ""
            else:
                tag_col = tag if label == "cur" else ""
            if m is None:
                print(f"  {tag_col:<42} {label:>5}   {'ERROR':>8}")
            else:
                print(f"  {tag_col:<42} {label:>5} {m['cos']:>8.6f} {m['max_diff']:>9.4f} {m['mean_diff']:>9.4f}")
        print()

    print(f"{'='*100}")


if __name__ == "__main__":
    main()
