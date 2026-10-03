#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Device correctness for the gfx120x iu8 int8-linear GEMM."""

import os
import sys

import pytest  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import flydsl  # noqa: E402,F401 -- preload comgr before torch/HIP loads LLVM
import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.common.tensor_shim import _run_compiled  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

_ARCH = str(get_rocm_arch() or "")
if not _ARCH.startswith("gfx120"):
    pytest.skip(f"GFX120X integer WMMA requires gfx120*, got {_ARCH}", allow_module_level=True)

from flydsl.compiler.jit_argument import PointerJitArg  # noqa: E402
from kernels.gemm.rdna4_int8_linear import (  # noqa: E402
    TileConfig,
    create_wmma_int8_linear_module,
    pick_tile_config,
)


def _ptr(tensor: object) -> PointerJitArg:
    """Convert a CUDA tensor to the raw-pointer ABI used by the kernel."""
    return flyc.from_c_void_p(fx.Uint8, tensor.data_ptr())


def _reference(a: object, b: object, scale_a: torch.Tensor | None, scale_b: torch.Tensor | None) -> torch.Tensor:
    accum = a.float() @ b.float().T
    return accum * scale_a.reshape(-1, 1) * scale_b.reshape(1, -1)


def _run_case(M: object, N: object, K: object, out_name: str = "bfloat16", *, scalar_weight: bool = False) -> None:
    torch.manual_seed(2026 + M + N + K)
    a = torch.randint(-8, 8, (M, K), device="cuda", dtype=torch.int8).contiguous()
    b = torch.randint(-8, 8, (N, K), device="cuda", dtype=torch.int8).contiguous()
    scale_a = (torch.rand(M, device="cuda", dtype=torch.float32) * 0.01 + 0.001).contiguous()
    scale_b = (torch.rand(1 if scalar_weight else N, device="cuda", dtype=torch.float32) * 0.01 + 0.001).contiguous()
    out_torch = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[out_name]
    out = torch.empty((M, N), device="cuda", dtype=out_torch)

    cfg = pick_tile_config(M, N, K, wgps=48)
    launch = create_wmma_int8_linear_module(
        out_name,
        cfg,
        skip_bounds=(M % cfg.bm == 0 and N % cfg.bn == 0),
        w_scale_per_n=not scalar_weight,
    )
    _run_compiled(
        launch,
        _ptr(a),
        _ptr(b),
        _ptr(out),
        _ptr(scale_a),
        _ptr(scale_b),
        M,
        N,
        K,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    ref = _reference(a, b, scale_a, scale_b).to(out_torch)
    rtol, atol = {
        "bfloat16": (4e-3, 2e-3),
        "float16": (2e-3, 2e-3),
        "float32": (1e-5, 1e-6),
    }[out_name]
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("out_name", ["bfloat16", "float16", "float32"])
def test_int8_linear_dequant(out_name: str) -> None:
    _run_case(128, 128, 128, out_name)


def test_int8_linear_partial_tile_and_scalar_weight_scale() -> None:
    # Exercises bounds masking, a non-128 tile, and the scalar-B-scale path.
    _run_case(37, 70, 80, "float32", scalar_weight=True)


def test_int8_linear_tile_selection() -> None:
    assert pick_tile_config(32, 128, 64, wgps=48) == TileConfig(64, 64, 64, 2, 2, 2, 2)
    # K>=512 non-skinny → 128x128x128
    assert pick_tile_config(256, 256, 512, wgps=48) == TileConfig(128, 128, 128, 4, 2, 2, 4)
    # Deep-K first (K>=2048): prefer 128x128x128 over tall-M 256 (measured vs HIP)
    assert pick_tile_config(1024, 1024, 4096, wgps=48) == TileConfig(128, 128, 128, 4, 2, 2, 4)
    # Tall-M fat grid + mid-deep K (512<=K<2048) → 256x128x128
    assert pick_tile_config(1024, 1024, 1024, wgps=48) == TileConfig(256, 128, 128, 4, 2, 4, 4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
