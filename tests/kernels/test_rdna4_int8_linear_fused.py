#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Correctness tests for fused act-quant + int8_linear (+ LoRA host residual) on gfx120x."""

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.rdna4_int8_linear_fused import (  # noqa: E402
    int8_linear_fused,
    int8_linear_fused_multi,
)
from tests.kernels.oracles import (  # noqa: E402
    reference_int8_linear_fused,
    reference_int8_linear_fused_multi,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch() or "")
if not ARCH.startswith("gfx120"):
    pytest.skip(
        f"RDNA4 fused int8_linear requires gfx120x, got {ARCH}",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "m,n,k",
    [
        pytest.param(64, 64, 64, id="64x64x64"),
        pytest.param(128, 128, 64, id="128x128x64"),
    ],
)
def test_fused_act_quant_int8_linear(m: int, n: int, k: int) -> None:
    """Fused path matches separate int8 quant + int8_linear reference."""
    torch.manual_seed(17)
    a_f = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    b_nk = torch.randint(-64, 64, (n, k), device="cuda", dtype=torch.int8)
    # per-row act scales from amax/127
    scale_a = (a_f.float().abs().amax(dim=1) / 127.0).clamp_min(1e-6).contiguous()
    scale_b = torch.full((n,), 0.02, device="cuda", dtype=torch.float32)

    out = int8_linear_fused(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    ref = reference_int8_linear_fused(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.25)


@pytest.mark.parametrize("rank", [8, 16])
def test_fused_int8_linear_lora_host_residual(rank: int) -> None:
    """Single-adapter LoRA uses host residual after fused base (FP8 host-residual policy)."""
    torch.manual_seed(23)
    m = n = k = 64
    a_f = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    b_nk = torch.randint(-64, 64, (n, k), device="cuda", dtype=torch.int8)
    scale_a = (a_f.float().abs().amax(dim=1) / 127.0).clamp_min(1e-6).contiguous()
    scale_b = torch.full((n,), 0.02, device="cuda", dtype=torch.float32)
    # Keep LoRA residual >> base-GEMM bf16 noise so a separate delta assert
    # can catch no-op / wrong-sign (tiny *0.02 adapters drown in atol=0.3).
    lora_down = torch.randn((rank, k), device="cuda", dtype=torch.bfloat16) * 0.2
    lora_up = torch.randn((n, rank), device="cuda", dtype=torch.bfloat16) * 0.2
    lora_scale = 0.5

    out = int8_linear_fused(
        a_f,
        b_nk,
        scale_a,
        scale_b,
        out_dtype=torch.bfloat16,
        lora_down=lora_down,
        lora_up=lora_up,
        lora_scale=lora_scale,
    )
    torch.cuda.synchronize()
    ref = reference_int8_linear_fused(
        a_f,
        b_nk,
        scale_a,
        scale_b,
        out_dtype=torch.bfloat16,
        lora_down=lora_down,
        lora_up=lora_up,
        lora_scale=lora_scale,
    )
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.3)

    # Full-output atol is dominated by base GEMM noise and can hide a no-op
    # or wrong-sign LoRA. Assert the residual contribution separately.
    base_out = int8_linear_fused(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    base_ref = reference_int8_linear_fused(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16)
    delta_out = out.float() - base_out.float()
    delta_ref = ref.float() - base_ref.float()
    assert float(delta_ref.abs().amax()) > 0.5, "oracle LoRA residual too small to validate"
    torch.testing.assert_close(delta_out, delta_ref, rtol=0.1, atol=0.15)


@pytest.mark.parametrize("n_adapters", [0, 1, 2, 3, 8, 16])
def test_fused_int8_linear_lora_multi(n_adapters: int) -> None:
    """Unbounded N adapters: base once + host residuals in load order."""
    torch.manual_seed(29)
    m = n = k = 64
    rank = 8
    a_f = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    b_nk = torch.randint(-64, 64, (n, k), device="cuda", dtype=torch.int8)
    scale_a = (a_f.float().abs().amax(dim=1) / 127.0).clamp_min(1e-6).contiguous()
    scale_b = torch.full((n,), 0.02, device="cuda", dtype=torch.float32)
    downs = [torch.randn((rank, k), device="cuda", dtype=torch.bfloat16) * 0.2 for _ in range(n_adapters)]
    ups = [torch.randn((n, rank), device="cuda", dtype=torch.bfloat16) * 0.2 for _ in range(n_adapters)]
    scales = [0.5 + 0.25 * i for i in range(n_adapters)]

    out = int8_linear_fused_multi(
        a_f,
        b_nk,
        scale_a,
        scale_b,
        out_dtype=torch.bfloat16,
        lora_downs=downs if downs else None,
        lora_ups=ups if ups else None,
        lora_scales=scales if scales else None,
    )
    torch.cuda.synchronize()
    ref = reference_int8_linear_fused_multi(
        a_f,
        b_nk,
        scale_a,
        scale_b,
        out_dtype=torch.bfloat16,
        lora_downs=downs if downs else None,
        lora_ups=ups if ups else None,
        lora_scales=scales if scales else None,
    )
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.3)

    if n_adapters >= 1:
        # List args on single-entry also dispatch to multi/host residual.
        out2 = int8_linear_fused(
            a_f,
            b_nk,
            scale_a,
            scale_b,
            out_dtype=torch.bfloat16,
            lora_down=downs,
            lora_up=ups,
            lora_scale=scales,
        )
        torch.testing.assert_close(out2.float(), ref.float(), rtol=0.05, atol=0.3)

        # Residual delta vs base: catch no-op / wrong-sign even when full-output
        # atol is dominated by base GEMM noise.
        base_out = int8_linear_fused(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        base_ref = reference_int8_linear_fused(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16)
        delta_out = out.float() - base_out.float()
        delta_ref = ref.float() - base_ref.float()
        assert float(delta_ref.abs().amax()) > 0.5, "oracle multi-LoRA residual too small to validate"
        torch.testing.assert_close(delta_out, delta_ref, rtol=0.1, atol=0.2)


def test_reference_helper_cpu_math() -> None:
    torch.manual_seed(3)
    a = torch.randn(32, 32, dtype=torch.bfloat16)
    b = torch.randint(-40, 40, (32, 32), dtype=torch.int8)
    sa = (a.float().abs().amax(dim=1) / 127.0).clamp_min(1e-6)
    sb = torch.full((32,), 0.01)
    out = reference_int8_linear_fused(a, b, sa, sb)
    assert out.shape == (32, 32)


def test_int8_fused_lora_fp32_acts() -> None:
    """fp32 activations + LoRA: device residual casts to bf16 (supported path)."""
    torch.manual_seed(57)
    m, n, k, rank = 32, 32, 64, 8
    a_f = torch.randn((m, k), device="cuda", dtype=torch.float32).clamp(-1, 1)
    b = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8)
    scale_a = (torch.rand(m, device="cuda") * 0.01 + 0.001).float()
    scale_b = (torch.rand(n, device="cuda") * 0.01 + 0.001).float()
    down = torch.randn((rank, k), device="cuda", dtype=torch.float32) * 0.02
    up = torch.randn((n, rank), device="cuda", dtype=torch.float32) * 0.02
    out = int8_linear_fused_multi(
        a_f,
        b,
        scale_a,
        scale_b,
        out_dtype=torch.bfloat16,
        lora_downs=down,
        lora_ups=up,
        lora_scales=1.0,
    )
    torch.cuda.synchronize()
    assert out.shape == (m, n) and torch.isfinite(out).all()


def test_int8_fused_multi_n0_default_scales() -> None:
    """N=0 multi with default lora_scales=None must not raise."""
    M, N, K = 32, 64, 64
    torch.manual_seed(2)
    a_f = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b_nk = torch.randint(-8, 8, (N, K), device="cuda", dtype=torch.int8)
    scale_a = torch.ones(M, device="cuda", dtype=torch.float32)
    scale_b = torch.ones(N, device="cuda", dtype=torch.float32)
    out = int8_linear_fused_multi(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    assert tuple(out.shape) == (M, N)


def test_int8_fused_quant_is_round_half_even() -> None:
    """Ties round to even and -128 is representable. The ±0.5 path did neither."""
    torch.manual_seed(0)
    k = 16
    vals = [-127.5, 2.5, -2.5, -128.0] + [0.0] * (k - 4)
    a = torch.tensor([vals], device="cuda", dtype=torch.float32)
    b = torch.eye(k, device="cuda", dtype=torch.int8)
    sa = torch.ones(1, device="cuda")
    sb = torch.ones(1, device="cuda")
    out = int8_linear_fused(a, b, sa, sb, out_dtype=torch.float32)
    torch.cuda.synchronize()
    ref = reference_int8_linear_fused(a, b, sa, sb, out_dtype=torch.float32)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert int(out[0, 0].item()) == -128
    assert int(out[0, 1].item()) == 2
    assert int(out[0, 2].item()) == -2
    assert int(out[0, 3].item()) == -128


def test_int8_fused_k0_is_zeros() -> None:
    a = torch.empty((2, 0), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((3, 0), device="cuda", dtype=torch.int8)
    sa = torch.ones(2, device="cuda")
    sb = torch.ones(3, device="cuda")
    out = int8_linear_fused(a, b, sa, sb, out_dtype=torch.float32)
    torch.cuda.synchronize()
    assert out.shape == (2, 3) and torch.count_nonzero(out) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
