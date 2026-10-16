#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Correctness tests for fused act-quant + FP8 e4m3/e5m2 scaled_mm (+ device LoRA residual)."""

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.rdna4_scaled_mm_fp8_fused import (  # noqa: E402
    scaled_mm_fp8_fused,
    scaled_mm_fp8_fused_multi,
)
from tests.kernels.oracles import (  # noqa: E402
    reference_scaled_mm_fp8_fused,
    reference_scaled_mm_fp8_fused_multi,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch() or "")
if not ARCH.startswith("gfx120"):
    pytest.skip(
        f"RDNA4 fused FP8 scaled-MM requires gfx120x, got {ARCH}",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "m,n,k",
    [
        pytest.param(64, 64, 64, id="64x64x64"),
        pytest.param(128, 128, 128, id="128x128x128"),
        pytest.param(130, 144, 128, id="ragged-130x144x128"),
    ],
)
def test_fused_act_quant_scaled_mm(m: int, n: int, k: int) -> None:
    """Fused path matches separate quant + scaled_mm reference (e4m3)."""
    torch.manual_seed(17)
    a_f = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    b_f32 = torch.randn((n, k), device="cuda", dtype=torch.float32).clamp(-1, 1)
    b_nk = b_f32.to(torch.float8_e4m3fn).contiguous()
    scale_a = torch.tensor([0.75], device="cuda", dtype=torch.float32)
    scale_b = torch.tensor([1.25], device="cuda", dtype=torch.float32)

    out = scaled_mm_fp8_fused(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    ref = reference_scaled_mm_fp8_fused(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.15)


def test_fused_scaled_mm_odd_k_stays_in_kernel() -> None:
    """Odd K is read in the fused kernel. k=127 used to be padded on the host."""
    torch.manual_seed(41)
    m, n, k = 64, 64, 127
    assert k % 16 != 0
    a_f = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    b_f32 = torch.randn((n, k), device="cuda", dtype=torch.float32).clamp(-1, 1)
    b_nk = b_f32.to(torch.float8_e4m3fn).contiguous()
    scale_a = torch.tensor([0.75], device="cuda", dtype=torch.float32)
    scale_b = torch.tensor([1.25], device="cuda", dtype=torch.float32)

    out = scaled_mm_fp8_fused(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    ref = reference_scaled_mm_fp8_fused(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.15)


def test_fused_act_quant_scaled_mm_e5m2() -> None:
    """Fused e5m2 path (bf8 act quant + Float8E5M2 WMMA) matches reference."""
    torch.manual_seed(21)
    m = n = k = 64
    a_f = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    b_nk = torch.randn((n, k), device="cuda", dtype=torch.float32).clamp(-1, 1).to(torch.float8_e5m2).contiguous()
    scale_a = torch.tensor([0.75], device="cuda", dtype=torch.float32)
    scale_b = torch.tensor([1.25], device="cuda", dtype=torch.float32)

    out = scaled_mm_fp8_fused(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    ref = reference_scaled_mm_fp8_fused(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.08, atol=0.25)


@pytest.mark.parametrize("rank", [8, 16])
def test_fused_scaled_mm_lora_device_residual(rank: int) -> None:
    """Single-adapter LoRA via host device-residual path (not in-kernel epilogue).

    ``scaled_mm_fp8_fused(..., lora_*)`` diverts to ``scaled_mm_fp8_fused_multi``:
    base fused quant+mm (lora_rank=0) + ``gemm_bf16_nmajor_lds`` residuals.
    Matches separate quant+mm + (x@A.T)@B.T.
    """
    torch.manual_seed(23)
    m = n = k = 64
    a_f = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    b_nk = torch.randn((n, k), device="cuda", dtype=torch.float32).clamp(-1, 1).to(torch.float8_e4m3fn).contiguous()
    scale_a = torch.tensor([0.8], device="cuda", dtype=torch.float32)
    scale_b = torch.tensor([1.1], device="cuda", dtype=torch.float32)
    # Keep LoRA residual >> base-GEMM bf16 noise so a separate delta assert
    # can catch no-op / wrong-sign (tiny *0.02 adapters drown in atol=0.2).
    lora_down = torch.randn((rank, k), device="cuda", dtype=torch.bfloat16) * 0.2
    lora_up = torch.randn((n, rank), device="cuda", dtype=torch.bfloat16) * 0.2
    lora_scale = 0.5

    out = scaled_mm_fp8_fused(
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
    ref = reference_scaled_mm_fp8_fused(
        a_f,
        b_nk,
        scale_a,
        scale_b,
        out_dtype=torch.bfloat16,
        lora_down=lora_down,
        lora_up=lora_up,
        lora_scale=lora_scale,
    )
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.2)

    # Full-output atol is dominated by base GEMM noise and can hide a no-op
    # or wrong-sign LoRA. Assert the residual contribution separately.
    base_out = scaled_mm_fp8_fused(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    base_ref = reference_scaled_mm_fp8_fused(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16)
    delta_out = out.float() - base_out.float()
    delta_ref = ref.float() - base_ref.float()
    # Residual must be large enough that no-op (delta=0) or wrong-sign fail.
    assert float(delta_ref.abs().amax()) > 0.5, "oracle LoRA residual too small to validate"
    torch.testing.assert_close(delta_out, delta_ref, rtol=0.1, atol=0.15)


def test_fused_scaled_mm_lora_device_residual_e5m2() -> None:
    """Device-residual LoRA path also works with e5m2 weights (not in-kernel epilogue)."""
    torch.manual_seed(29)
    m = n = k = 64
    rank = 8
    a_f = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    b_nk = torch.randn((n, k), device="cuda", dtype=torch.float32).clamp(-1, 1).to(torch.float8_e5m2).contiguous()
    scale_a = torch.tensor([0.8], device="cuda", dtype=torch.float32)
    scale_b = torch.tensor([1.1], device="cuda", dtype=torch.float32)
    lora_down = torch.randn((rank, k), device="cuda", dtype=torch.bfloat16) * 0.02
    lora_up = torch.randn((n, rank), device="cuda", dtype=torch.bfloat16) * 0.02
    lora_scale = 0.5

    out = scaled_mm_fp8_fused(
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
    ref = reference_scaled_mm_fp8_fused(
        a_f,
        b_nk,
        scale_a,
        scale_b,
        out_dtype=torch.bfloat16,
        lora_down=lora_down,
        lora_up=lora_up,
        lora_scale=lora_scale,
    )
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.08, atol=0.3)


def test_fused_scaled_mm_lora_diverts_to_multi(monkeypatch) -> None:
    """Host LoRA args must call scaled_mm_fp8_fused_multi (device residual), not in-kernel."""
    import kernels.gemm.rdna4_scaled_mm_fp8_fused as mod

    calls: list[tuple] = []
    real_multi = mod.scaled_mm_fp8_fused_multi

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_multi(*args, **kwargs)

    monkeypatch.setattr(mod, "scaled_mm_fp8_fused_multi", _spy)
    torch.manual_seed(7)
    m = n = k = 64
    a_f = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    b_nk = torch.randn((n, k), device="cuda", dtype=torch.float32).clamp(-1, 1).to(torch.float8_e4m3fn).contiguous()
    sa = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    sb = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    down = torch.randn((8, k), device="cuda", dtype=torch.bfloat16) * 0.02
    up = torch.randn((n, 8), device="cuda", dtype=torch.bfloat16) * 0.02
    out = mod.scaled_mm_fp8_fused(
        a_f, b_nk, sa, sb, out_dtype=torch.bfloat16, lora_down=down, lora_up=up, lora_scale=0.5
    )
    torch.cuda.synchronize()
    assert len(calls) == 1, f"expected exactly one multi divert, got {len(calls)}"
    assert calls[0][1].get("lora_downs") is down
    assert calls[0][1].get("lora_ups") is up
    assert out.shape == (m, n)


def test_reference_helper_cpu_math() -> None:
    """Pure torch reference stays consistent without launching the kernel."""
    torch.manual_seed(3)
    a = torch.randn(32, 32, dtype=torch.bfloat16)
    b = torch.randn(32, 32).clamp(-1, 1).to(torch.float8_e4m3fn)
    sa = torch.tensor([1.0])
    sb = torch.tensor([1.0])
    down = torch.randn(8, 32, dtype=torch.bfloat16) * 0.01
    up = torch.randn(32, 8, dtype=torch.bfloat16) * 0.01
    out = reference_scaled_mm_fp8_fused(a, b, sa, sb, lora_down=down, lora_up=up, lora_scale=1.0)
    assert out.shape == (32, 32)
    assert out.dtype == torch.bfloat16

    b5 = torch.randn(32, 32).clamp(-1, 1).to(torch.float8_e5m2)
    out5 = reference_scaled_mm_fp8_fused(a, b5, sa, sb)
    assert out5.shape == (32, 32)


def test_fused_scaled_mm_lora_fp32_acts_and_adapters() -> None:
    """fp32 acts + fp32 LoRA adapters cast to bf16 for device residual (no reject)."""
    torch.manual_seed(55)
    m, n, k, rank = 32, 64, 64, 8
    a_f = torch.randn((m, k), device="cuda", dtype=torch.float32).clamp(-1, 1)
    b_nk = torch.randn((n, k), device="cuda", dtype=torch.float32).clamp(-1, 1).to(torch.float8_e4m3fn)
    scale_a = torch.tensor([0.75], device="cuda", dtype=torch.float32)
    scale_b = torch.tensor([1.25], device="cuda", dtype=torch.float32)
    down = torch.randn((rank, k), device="cuda", dtype=torch.float32) * 0.02
    up = torch.randn((n, rank), device="cuda", dtype=torch.float32) * 0.02
    out = scaled_mm_fp8_fused_multi(
        a_f,
        b_nk,
        scale_a,
        scale_b,
        out_dtype=torch.bfloat16,
        lora_downs=down,
        lora_ups=up,
        lora_scales=0.5,
    )
    torch.cuda.synchronize()
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


@pytest.mark.parametrize("n_adapters", [0, 1, 2, 3, 8, 16])
def test_fused_scaled_mm_lora_multi(n_adapters: int) -> None:
    """Unbounded N adapters in load order match sequential reference sum."""
    torch.manual_seed(31 + n_adapters)
    m = n = k = 64
    a_f = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    b_nk = torch.randn((n, k), device="cuda", dtype=torch.float32).clamp(-1, 1).to(torch.float8_e4m3fn).contiguous()
    scale_a = torch.tensor([0.8], device="cuda", dtype=torch.float32)
    scale_b = torch.tensor([1.1], device="cuda", dtype=torch.float32)

    # Cycle ranks so N can be arbitrarily large (unbounded-adapter contract).
    _rank_cycle = (8, 16, 8, 32, 16, 8)
    ranks = [_rank_cycle[i % len(_rank_cycle)] for i in range(n_adapters)]
    downs, ups, scales = [], [], []
    for i, rank in enumerate(ranks):
        downs.append(torch.randn((rank, k), device="cuda", dtype=torch.bfloat16) * (0.02 + 0.01 * i))
        ups.append(torch.randn((n, rank), device="cuda", dtype=torch.bfloat16) * (0.02 + 0.01 * i))
        scales.append(0.4 + 0.1 * i)

    if n_adapters == 0:
        # Documented N=0: no LoRA kwargs (lora_scales defaults to None).
        out = scaled_mm_fp8_fused_multi(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16)
    else:
        out = scaled_mm_fp8_fused_multi(
            a_f,
            b_nk,
            scale_a,
            scale_b,
            out_dtype=torch.bfloat16,
            lora_downs=downs,
            lora_ups=ups,
            lora_scales=scales,
        )
    torch.cuda.synchronize()
    ref = reference_scaled_mm_fp8_fused_multi(
        a_f,
        b_nk,
        scale_a,
        scale_b,
        out_dtype=torch.bfloat16,
        lora_downs=downs if downs else None,
        lora_ups=ups if ups else None,
        lora_scales=scales if scales else None,
    )
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.25)

    # List dispatch through scaled_mm_fp8_fused must match multi.
    if n_adapters >= 1:
        out2 = scaled_mm_fp8_fused(
            a_f,
            b_nk,
            scale_a,
            scale_b,
            out_dtype=torch.bfloat16,
            lora_down=downs,
            lora_up=ups,
            lora_scale=scales,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(out2.float(), ref.float(), rtol=0.05, atol=0.25)

    # Explicit sequential sum vs multi (order sensitivity check for N>=2).
    if n_adapters >= 2:
        base = reference_scaled_mm_fp8_fused(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16)
        seq = base.float()
        for down, up, sc in zip(downs, ups, scales):
            hid = a_f.float() @ down.float().T
            seq = seq + float(sc) * (hid @ up.float().T)
        torch.testing.assert_close(out.float(), seq, rtol=0.05, atol=0.25)
