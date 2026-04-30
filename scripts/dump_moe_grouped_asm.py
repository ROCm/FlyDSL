#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Trigger compilation of the m-grouped masked MoE kernels and dump asm.

Usage::

    # No GPU required: AOT compile + dump ISA only.
    PYTHONPATH=./ python scripts/dump_moe_grouped_asm.py --skip-launch \\
        [--out-dir /tmp/moe_grouped_asm]

    # On a real gfx1250 box: full compile + launch.
    PYTHONPATH=./ python scripts/dump_moe_grouped_asm.py \\
        [--out-dir /tmp/moe_grouped_asm]

The script auto-sets:
    FLYDSL_DEBUG_DUMP_IR=1     # enables the IR/ISA dump pipeline
    FLYDSL_DEBUG_DUMP_DIR=...  # destination
    FLYDSL_GPU_ARCH=gfx1250    # forces target arch even on non-gfx1250 hosts
    FLYDSL_RUNTIME_ENABLE_CACHE=0  # always re-compile

Verification: after a successful run, look at
    <out-dir>/<kernel-name>/15_final_isa.s
and grep for ``tensor_load_to_lds_d2`` (TDM 2D loads),
``v_wmma_f32_16x16x32_f16`` (WMMA), ``v_exp_f32`` (silu, stage1 only) and
``buffer_store`` (epilogue).

Pre-requisite: FlyDSL must be built and ``python/flydsl/_mlir/...`` must be
populated (run ``bash scripts/build.sh`` if missing).
"""

from __future__ import annotations

import argparse
import os
import sys


def _ensure_env(out_dir: str, compile_only: bool) -> None:
    # FlyDSL produces the final amdgcn ISA dump (``15_final_isa.s``) as part
    # of the IR-dump pipeline driven by ``FLYDSL_DEBUG_DUMP_IR``. Set both that
    # and the legacy ``FLYDSL_DUMP_ASM`` flag for older builds, plus disable
    # runtime cache so the compile actually re-runs every time.
    os.environ["FLYDSL_DEBUG_DUMP_IR"] = "1"
    os.environ.setdefault("FLYDSL_DUMP_IR", "1")
    os.environ.setdefault("FLYDSL_DEBUG_DUMP_ASM", "1")
    os.environ.setdefault("FLYDSL_DUMP_ASM", "1")
    os.environ["FLYDSL_DEBUG_DUMP_DIR"] = out_dir
    os.environ["FLYDSL_DUMP_DIR"] = out_dir
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    if compile_only:
        # Forces FlyDSL to compile to ISA without invoking the GPU. This lets
        # the dump succeed on machines that don't have a gfx1250 device. The
        # arch is selected via FLYDSL_GPU_ARCH below.
        os.environ["FLYDSL_COMPILE_ONLY"] = "1"
    os.environ.setdefault("FLYDSL_GPU_ARCH", "gfx1250")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile masked m-grouped MoE WMMA kernels and dump asm.",
    )
    parser.add_argument(
        "--out-dir",
        default="/tmp/moe_grouped_asm",
        help="Directory for FLYDSL_DUMP_DIR (default: /tmp/moe_grouped_asm)",
    )
    parser.add_argument("--model-dim", type=int, default=2048)
    parser.add_argument("--inter-dim", type=int, default=1408)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--max-m", type=int, default=128)
    parser.add_argument("--tile-m", type=int, default=64)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--tile-k", type=int, default=64)
    parser.add_argument("--m-warp", type=int, default=2)
    parser.add_argument("--n-warp", type=int, default=4)
    parser.add_argument("--in-dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--out-dtype", choices=("f16", "bf16"), default="f16")
    parser.add_argument("--num-buffers", type=int, default=2)
    parser.add_argument("--skip-launch", action="store_true",
                        help="Only compile, do not launch (for non-GPU machines). "
                             "Equivalent to setting FLYDSL_COMPILE_ONLY=1.")
    parser.add_argument("--stage", choices=("1", "2", "both"), default="both")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    _ensure_env(out_dir, compile_only=args.skip_launch)
    print(f"[dump_moe_grouped_asm] FLYDSL_DEBUG_DUMP_DIR={out_dir}")
    print(f"[dump_moe_grouped_asm] FLYDSL_GPU_ARCH={os.environ['FLYDSL_GPU_ARCH']}")
    if args.skip_launch:
        print("[dump_moe_grouped_asm] FLYDSL_COMPILE_ONLY=1 (no GPU launch)")

    from kernels.moe_grouped_gemm_wmma_gfx1250 import (
        compile_moe_grouped_gemm1_masked,
        compile_moe_grouped_gemm2_masked,
    )

    common = dict(
        model_dim=args.model_dim,
        inter_dim=args.inter_dim,
        experts=args.experts,
        max_m=args.max_m,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        m_warp=args.m_warp,
        n_warp=args.n_warp,
        in_dtype=args.in_dtype,
        out_dtype=args.out_dtype,
        num_buffers=args.num_buffers,
    )

    # When compiling both stages on a host whose GPU does NOT match
    # FLYDSL_GPU_ARCH (e.g. compiling a gfx1250 kernel on a gfx950 host),
    # the runtime's failed hipModuleLoadData call leaves the HIP context in
    # a broken state, so any tensor allocation for stage2 right after stage1
    # raises ``hipErrorInvalidHandle``. Re-running this script as a fresh
    # subprocess for stage2 sidesteps that.
    if args.stage == "both" and args.skip_launch:
        import subprocess
        rc1 = subprocess.call(
            [sys.executable, __file__, "--stage", "1", "--skip-launch",
             "--out-dir", out_dir,
             "--model-dim", str(args.model_dim),
             "--inter-dim", str(args.inter_dim),
             "--experts", str(args.experts),
             "--max-m", str(args.max_m),
             "--tile-m", str(args.tile_m),
             "--tile-n", str(args.tile_n),
             "--tile-k", str(args.tile_k),
             "--m-warp", str(args.m_warp),
             "--n-warp", str(args.n_warp),
             "--in-dtype", args.in_dtype,
             "--out-dtype", args.out_dtype,
             "--num-buffers", str(args.num_buffers)],
        )
        rc2 = subprocess.call(
            [sys.executable, __file__, "--stage", "2", "--skip-launch",
             "--out-dir", out_dir,
             "--model-dim", str(args.model_dim),
             "--inter-dim", str(args.inter_dim),
             "--experts", str(args.experts),
             "--max-m", str(args.max_m),
             "--tile-m", str(args.tile_m),
             "--tile-n", str(args.tile_n),
             "--tile-k", str(args.tile_k),
             "--m-warp", str(args.m_warp),
             "--n-warp", str(args.n_warp),
             "--in-dtype", args.in_dtype,
             "--out-dtype", args.out_dtype,
             "--num-buffers", str(args.num_buffers)],
        )
        return rc1 or rc2

    s1 = None
    s2 = None
    if args.stage in ("1", "both"):
        print("[dump_moe_grouped_asm] compiling stage1 (gateup + silu*up) ...")
        s1 = compile_moe_grouped_gemm1_masked(**common)
        print("[dump_moe_grouped_asm] stage1 compiled.")
    if args.stage in ("2", "both"):
        print("[dump_moe_grouped_asm] compiling stage2 (down) ...")
        s2 = compile_moe_grouped_gemm2_masked(**common)
        print("[dump_moe_grouped_asm] stage2 compiled.")

    try:
        import torch
    except ImportError:
        print("[dump_moe_grouped_asm] torch not available, cannot construct dummy "
              "tensors for the AOT compile path. Install torch and rerun.")
        return 1

    have_gpu = torch.cuda.is_available()

    # NOTE: even in --skip-launch mode FlyDSL still needs to call the JIT to
    # produce the IR/ISA dump, and the JIT's DLPack-based argument conversion
    # requires GPU tensors (CPU DLPack does not accept ``stream=-1``). On a
    # GPU host we therefore always allocate GPU tensors; FLYDSL_COMPILE_ONLY=1
    # short-circuits the actual kernel launch so the device GFX arch (e.g.
    # gfx950) doesn't need to match FLYDSL_GPU_ARCH=gfx1250.
    if not have_gpu:
        print("[dump_moe_grouped_asm] No CUDA/ROCm device available -- the "
              "FlyDSL JIT requires GPU tensors for DLPack conversion. "
              "Re-run on a host with at least one ROCm GPU visible.")
        return 1

    device = "cuda"
    in_torch_dtype = torch.float16 if args.in_dtype == "fp16" else torch.bfloat16
    out_torch_dtype = torch.float16 if args.out_dtype == "f16" else torch.bfloat16

    E = args.experts
    M = args.max_m
    K = args.model_dim
    N_inter = args.inter_dim

    masked_m = torch.full((E,), M // 2, dtype=torch.int32, device=device)
    stream_arg = torch.cuda.current_stream()

    if s1 is not None:
        x = torch.randn(E, M, K, dtype=in_torch_dtype, device=device)
        w1 = torch.randn(E, 2 * N_inter, K, dtype=in_torch_dtype, device=device)
        y1 = torch.empty(E, M, N_inter, dtype=out_torch_dtype, device=device)
        s1(y1, x, w1, masked_m, M, N_inter, K, E, stream=stream_arg)
        if not args.skip_launch:
            torch.cuda.synchronize()
            print("[dump_moe_grouped_asm] stage1 launched.")
        else:
            print("[dump_moe_grouped_asm] stage1 compiled (COMPILE_ONLY).")

    if s2 is not None:
        x2 = torch.randn(E, M, N_inter, dtype=in_torch_dtype, device=device)
        w2 = torch.randn(E, K, N_inter, dtype=in_torch_dtype, device=device)
        y2 = torch.empty(E, M, K, dtype=out_torch_dtype, device=device)
        s2(y2, x2, w2, masked_m, M, K, N_inter, E, stream=stream_arg)
        if not args.skip_launch:
            torch.cuda.synchronize()
            print("[dump_moe_grouped_asm] stage2 launched.")
        else:
            print("[dump_moe_grouped_asm] stage2 compiled (COMPILE_ONLY).")

    print(f"[dump_moe_grouped_asm] dumps in {out_dir}:")
    found_any = False
    for root, _dirs, files in os.walk(out_dir):
        for name in sorted(files):
            if name.endswith((".s", ".asm", ".ll")):
                path = os.path.join(root, name)
                print(f"  {path} ({os.path.getsize(path)} bytes)")
                found_any = True
    if not found_any:
        print("  (no .s/.asm/.ll files yet -- compilation may have failed; "
              "check stderr above)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
