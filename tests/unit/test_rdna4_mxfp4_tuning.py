# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The calibration oracle decodes storage independently and rejects bad output."""

import pytest
import torch

from kernels.gemm.rdna4_mxfp4.api import PreparedMXFP4
from kernels.gemm.rdna4_mxfp4.tuning import _Context

pytestmark = pytest.mark.l0_backend_agnostic


@pytest.mark.parametrize("layout", ["linear", "split"])
@pytest.mark.parametrize("tiny_scales", [False, True])
def test_sampled_oracle_handles_packing_padding_and_subnormal_scales(layout, tiny_scales):
    m, n, npad, k = 11, 19, 32, 96
    generator = torch.Generator().manual_seed(29)
    codes = torch.randint(0, 16, (npad, k), generator=generator, dtype=torch.uint8)
    exponents = torch.randint(
        0 if tiny_scales else 124, 4 if tiny_scales else 132, (npad, k // 32), generator=generator, dtype=torch.uint8
    )
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)[(codes & 7).long()]
    values *= torch.where(codes & 8 != 0, -1.0, 1.0)
    values *= torch.exp2(exponents.double() - 127).repeat_interleave(32, dim=1)
    if layout == "linear":
        packed = codes[:, ::2] | codes[:, 1::2] << 4
    else:
        groups = codes.view(npad, k // 8, 8)
        packed = (groups[..., :4] | groups[..., 4:] << 4).reshape(npad, k // 2)
    packed = packed.view(npad // 16, 16, k // 16, 2, 4).permute(0, 2, 3, 1, 4).contiguous()
    exponents = exponents.view(npad // 16, 16, k // 32).transpose(1, 2).contiguous()
    weights = PreparedMXFP4(packed, exponents, n, k, "e4m3", nibble_layout=layout)
    x = torch.randn(m, k, generator=generator).to(torch.float8_e4m3fn)
    xs = torch.linspace(0.5, 1.5, m) * (2.0**20 if tiny_scales else 1)
    expected = ((x.double() @ values[:n].T) * xs[:, None].double()).to(torch.bfloat16)
    out = torch.empty(m, n, dtype=torch.bfloat16)
    context = _Context(None, x, weights, xs, out, "auto")
    context.initialize()
    torch.testing.assert_close(
        context.reference, expected[context.rows[:, None], context.columns[None, :]].double(), atol=0, rtol=0
    )
    with context.validate({}):
        out.copy_(expected)
    with pytest.raises(ValueError, match="unexpected NaNs"):
        with context.validate({}):
            pass
    with pytest.raises(ValueError, match="relative RMS"):
        with context.validate({}):
            out.copy_(expected * 2)
