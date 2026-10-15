# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""e5m2 attention stays inside one e5m2 step of the fp32 formula.

QK and PV both use the e5m2 matrix product. One step near 1.0 is 0.25, and
the step doubles when the output doubles.
"""

import math
import sys
from pathlib import Path

import pytest

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)
if not hasattr(torch, "float8_e5m2"):
    pytest.skip("torch.float8_e5m2 missing", allow_module_level=True)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.attention.flash_attn_gfx120x_host import flydsl_flash_attn_fp8_func  # noqa: E402


def _arch() -> str:
    return (torch.cuda.get_device_properties(0).gcnArchName or "").split(":")[0]


pytestmark = [
    pytest.mark.l2_device,
    pytest.mark.rocm_lower,
    pytest.mark.skipif(
        not str(get_rocm_arch() or "").startswith("gfx120"),
        reason=f"requires gfx120x, got {_arch()!r}",
    ),
]


def _fp32_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """BSHD fp32 attention. ``q``/``k``/``v`` are already dequantized."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    qq = q.float().permute(0, 2, 1, 3)
    kk = k.float().permute(0, 2, 1, 3)
    vv = v.float().permute(0, 2, 1, 3)
    logits = torch.matmul(qq, kk.transpose(-1, -2)) * scale
    weights = torch.softmax(logits, dim=-1)
    return torch.matmul(weights, vv).permute(0, 2, 1, 3)


def test_e5m2_short_seq_pv_matches_fp32_formula() -> None:
    """2 keys, head 64. The gap must stay inside one e5m2 step at the output."""
    torch.manual_seed(0)
    q = torch.randn(1, 2, 1, 64, device="cuda", dtype=torch.float32)
    k = torch.randn(1, 2, 1, 64, device="cuda", dtype=torch.float32)
    v = torch.randn(1, 2, 1, 64, device="cuda", dtype=torch.float32)
    q8 = q.to(torch.float8_e5m2)
    k8 = k.to(torch.float8_e5m2)
    v8 = v.to(torch.float8_e5m2)
    out = flydsl_flash_attn_fp8_func(q8, k8, v8, causal=False, q_descale=1.0, k_descale=1.0, v_descale=1.0)
    torch.cuda.synchronize()
    ref = _fp32_attention(q8.float(), k8.float(), v8.float())
    err = (out.float() - ref).abs().max().item()
    magnitude = max(out.float().abs().max().item(), 2.0**-24)
    allowed = 2.0 ** (math.floor(math.log2(magnitude)) - 2)
    assert err <= allowed, (err, allowed)
