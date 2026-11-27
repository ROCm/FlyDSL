#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""``Vector.to`` float conversions on the device, in every rounding mode, against exact rounding.

The instructions are picked per chip by convert-rocdl-math-ops (tests/mlir/Conversion/float_conversions.mlir).
"""

import math
from fractions import Fraction

import pytest

try:
    import torch
except ImportError:
    torch = None

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

N = 8
MODES = {
    "to_nearest_even": None,
    "toward_zero": fx.RoundingMode.toward_zero,
    "to_nearest_away": fx.RoundingMode.to_nearest_away,
    "upward": fx.RoundingMode.upward,
    "downward": fx.RoundingMode.downward,
}
# (significand bits, minimum normal exponent, largest finite value)
FORMATS = {torch.bfloat16: (8, -126, float.fromhex("0x1.FEp127")), torch.float16: (11, -14, 65504.0)}


def exact_round(x, dtype, mode):
    """`x` rounded to `dtype` under `mode`, computed exactly."""
    sig, emin, largest = FORMATS[dtype]
    if math.isnan(x) or math.isinf(x) or x == 0:
        return x
    a = abs(Fraction(x))
    e = math.floor(math.log2(a))
    e -= Fraction(2) ** e > a
    e += Fraction(2) ** (e + 1) <= a
    ulp = Fraction(2) ** (max(e, emin) - sig + 1)
    lo = a // ulp * ulp
    hi = lo + ulp
    grow = (mode == "upward" and x > 0) or (mode == "downward" and x < 0)
    if lo == a:
        r = lo
    elif mode in ("to_nearest_even", "to_nearest_away"):
        tie_up = mode == "to_nearest_away" or (lo / ulp) % 2 == 1
        r = lo if a - lo < hi - a else hi if hi - a < a - lo else (hi if tie_up else lo)
    else:
        r = hi if grow else lo
    r = float(r)
    if r > largest:
        r = math.inf if mode in ("to_nearest_even", "to_nearest_away") or grow else largest
    return math.copysign(r, x)


def ties(base, sig):
    """The midpoint past `base` (with `sig` significand bits) in magnitude, and f32 values next to it."""
    half = math.copysign(2.0 ** (math.floor(math.log2(abs(base))) - sig), base)
    mid = base + half
    return [mid, mid + half / 4096, mid - half / 4096]


VALUES = [
    *ties(1.0, 8),
    *ties(1.0 + 2.0**-7, 8),
    *ties(-1.0, 8),
    *ties(1.0, 11),
    *ties(1.0 + 2.0**-10, 11),
    *ties(-3.0, 11),
    65519.0,
    65520.0,
    -65520.0,
    1.0e6,
    3.4e38,
    -3.4e38,
    2.0**-25,
    -(2.0**-25),
    2.0**-134,
    2.0**-140,
    0.1,
    -0.1,
    0.0,
    -0.0,
    math.inf,
    -math.inf,
    math.nan,
]
VALUES += [0.5] * (-len(VALUES) % N)


def convert(x, dst, out_dtype, mode=None):
    @flyc.kernel
    def kernel(A: fx.Tensor, C: fx.Tensor):
        offset = fx.thread_idx.x * N
        v = fx.generic_load(A.iter + offset, dtype=fx.Float32, count=N)
        fx.generic_store(C.iter + offset, v.to(dst, rounding_mode=mode))

    @flyc.jit
    def launch(A: fx.Tensor, C: fx.Tensor):
        kernel(A, C).launch(grid=(1, 1, 1), block=(x.numel() // N, 1, 1))

    c = torch.empty(x.numel(), dtype=out_dtype, device="cuda")
    launch(x.cuda(), c)
    torch.cuda.synchronize()
    return c.cpu()


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("dst,out_dtype", [(fx.BFloat16, torch.bfloat16), (fx.Float16, torch.float16)])
def test_f32_narrowing_rounds_exactly(dst, out_dtype, mode):
    x = torch.tensor(VALUES, dtype=torch.float32)
    got = convert(x, dst, out_dtype, MODES[mode]).double().tolist()
    want = [exact_round(v, out_dtype, mode) for v in x.double().tolist()]
    bad = [(v, g, w) for v, g, w in zip(x.tolist(), got, want) if not (g == w or (math.isnan(g) and math.isnan(w)))]
    assert not bad, f"{len(bad)} wrong (x, got, want): {bad[:4]}"
    signs = [math.copysign(1, g) == math.copysign(1, w) for g, w in zip(got, want) if g == w == 0]
    assert all(signs), "signed zero lost"


def _native_fp8():
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch.startswith("gfx94"):
        return fx.Float8E4M3FNUZ, torch.float8_e4m3fnuz
    if arch.startswith("gfx95") or arch.startswith("gfx12"):
        return fx.Float8E4M3FN, torch.float8_e4m3fn
    pytest.skip(f"no native fp8 conversion on {arch}")


def test_fp8_round_trip():
    fx_fp8, torch_fp8 = _native_fp8()
    x = torch.linspace(-240.0, 240.0, 64 * N, dtype=torch.float32)

    @flyc.kernel
    def kernel(A: fx.Tensor, Q: fx.Tensor, D: fx.Tensor):
        offset = fx.thread_idx.x * N
        q = fx.generic_load(A.iter + offset, dtype=fx.Float32, count=N).to(fx_fp8)
        fx.generic_store(fx.recast_iter(fx.Uint8, Q.iter) + offset, q.bitcast(fx.Uint8))
        fx.generic_store(D.iter + offset, q.to(fx.Float32))

    @flyc.jit
    def launch(A: fx.Tensor, Q: fx.Tensor, D: fx.Tensor):
        kernel(A, Q, D).launch(grid=(1, 1, 1), block=(64, 1, 1))

    q = torch.empty(x.numel(), dtype=torch.uint8, device="cuda")
    d = torch.empty(x.numel(), dtype=torch.float32, device="cuda")
    launch(x.cuda(), q, d)
    torch.cuda.synchronize()
    want = x.to(torch_fp8)
    assert torch.equal(q.cpu(), want.view(torch.uint8))
    assert torch.equal(d.cpu(), want.float())
