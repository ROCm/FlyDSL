#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device roundtrip tests for ROCDL FP8/BF8/F16 conversion DSL wrappers."""

import sys

import pytest

import flydsl  # noqa: F401 -- preload comgr before torch/HIP loads LLVM
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch, get_warp_size

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

_ARCH = str(get_rocm_arch() or "")
_HAS_FP8_CVT = _ARCH.startswith(("gfx942", "gfx950", "gfx117", "gfx12"))
_WARP_SIZE = get_warp_size(_ARCH)


def test_rocdl_cvt_pkrtz_roundtrip():
    @flyc.kernel
    def cvt_pkrtz_kernel(Out: fx.Tensor):
        tid = fx.thread_idx.x
        if tid == 0:
            a = fx.Float32(1.5)
            b = fx.Float32(-2.0)
            pk_f16 = fx.Vector(fx.rocdl.cvt_pkrtz(T.f16x2, a, b), dtype=fx.Float16)
            out1d = fx.make_view(fx.get_iter(Out), fx.make_layout((2,), (1,)))
            out1d[0] = pk_f16[0].to(fx.Float32)
            out1d[1] = pk_f16[1].to(fx.Float32)

    @flyc.jit
    def launch(Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        cvt_pkrtz_kernel(Out).launch(grid=(1, 1, 1), block=(_WARP_SIZE, 1, 1), stream=stream)

    out = torch.full((2,), -999.0, dtype=torch.float32, device="cuda")
    launch(out, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    expected = torch.tensor([1.5, -2.0], dtype=torch.float32)
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@pytest.mark.skipif(
    not _HAS_FP8_CVT,
    reason=f"ROCDL FP8/BF8 conversion ops require gfx942/gfx950/gfx117x/gfx12xx, got {_ARCH}",
)
def test_rocdl_fp8_bf8_f16_cvt_roundtrip():
    @flyc.kernel
    def cvt_roundtrip_kernel(Out: fx.Tensor):
        tid = fx.thread_idx.x
        if tid == 0:
            a = fx.Float32(1.5)
            b = fx.Float32(-2.0)
            c = fx.Float32(0.25)
            d = fx.Float32(4.0)

            pk_fp8_lo = fx.rocdl.cvt_pk_fp8_f32(T.i32, a, b, 0, False)
            pk_fp8 = fx.rocdl.cvt_pk_fp8_f32(T.i32, c, d, fx.Int32(pk_fp8_lo), True)

            pk_bf8_lo = fx.rocdl.cvt_pk_bf8_f32(T.i32, a, b, 0, False)
            pk_bf8 = fx.rocdl.cvt_pk_bf8_f32(T.i32, c, d, fx.Int32(pk_bf8_lo), True)

            sr_fp8 = fx.rocdl.cvt_sr_fp8_f32(T.i32, fx.Float32(0.5), 0, 0, 2)
            sr_bf8 = fx.rocdl.cvt_sr_bf8_f32(T.i32, fx.Float32(-0.5), 0, 0, 3)

            pk_f16 = fx.Vector(fx.rocdl.cvt_pkrtz(T.f16x2, a, b), dtype=fx.Float16)

            out1d = fx.make_view(fx.get_iter(Out), fx.make_layout((12,), (1,)))
            for idx in fx.range_constexpr(4):
                out1d[idx] = fx.Float32(fx.rocdl.cvt_f32_fp8(T.f32, fx.Int32(pk_fp8), idx))
                out1d[4 + idx] = fx.Float32(fx.rocdl.cvt_f32_bf8(T.f32, fx.Int32(pk_bf8), idx))
            out1d[8] = fx.Float32(fx.rocdl.cvt_f32_fp8(T.f32, fx.Int32(sr_fp8), 2))
            out1d[9] = fx.Float32(fx.rocdl.cvt_f32_bf8(T.f32, fx.Int32(sr_bf8), 3))
            out1d[10] = pk_f16[0].to(fx.Float32)
            out1d[11] = pk_f16[1].to(fx.Float32)

    @flyc.jit
    def launch(Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        cvt_roundtrip_kernel(Out).launch(grid=(1, 1, 1), block=(_WARP_SIZE, 1, 1), stream=stream)

    out = torch.full((12,), -999.0, dtype=torch.float32, device="cuda")
    launch(out, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    expected = torch.tensor([1.5, -2.0, 0.25, 4.0, 1.5, -2.0, 0.25, 4.0, 0.5, -0.5, 1.5, -2.0], dtype=torch.float32)
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@pytest.mark.skip(
    reason=(
        "cvt_pk_f32_bf8 DSL wrapper is commented out until upstream "
        "llvm/llvm-project#231721 lands: unmodified LLVM VOP1Instructions.td "
        "maps HasCvtFP8ByteSel v_cvt_pk_f32_{fp8,bf8} only to _fake16 pseudos "
        "(omitting UseRealTrue16Insts _t16 ISel patterns)."
    )
)
def test_rocdl_pk_f32_fp8_bf8_unpack():
    # Uncomment alongside ``cvt_pk_f32_bf8`` in ``flydsl.expr.rocdl`` once
    # https://github.com/llvm/llvm-project/pull/231721 is in FlyDSL's LLVM commit:
    #
    # @flyc.kernel
    # def cvt_pk_unpack_kernel(Out: fx.Tensor):
    #     tid = fx.thread_idx.x
    #     if tid == 0:
    #         a = fx.Float32(1.5)
    #         b = fx.Float32(-2.0)
    #         c = fx.Float32(0.25)
    #         d = fx.Float32(4.0)
    #
    #         pk_fp8_lo = fx.rocdl.cvt_pk_fp8_f32(T.i32, a, b, 0, False)
    #         pk_fp8 = fx.rocdl.cvt_pk_fp8_f32(T.i32, c, d, fx.Int32(pk_fp8_lo), True)
    #         pk_bf8_lo = fx.rocdl.cvt_pk_bf8_f32(T.i32, a, b, 0, False)
    #         pk_bf8 = fx.rocdl.cvt_pk_bf8_f32(T.i32, c, d, fx.Int32(pk_bf8_lo), True)
    #
    #         fp8_lo = fx.Vector(fx.rocdl.cvt_pk_f32_fp8(T.f32x2, fx.Int32(pk_fp8), False), dtype=fx.Float32)
    #         fp8_hi = fx.Vector(fx.rocdl.cvt_pk_f32_fp8(T.f32x2, fx.Int32(pk_fp8), True), dtype=fx.Float32)
    #         bf8_lo = fx.Vector(fx.rocdl.cvt_pk_f32_bf8(T.f32x2, fx.Int32(pk_bf8), False), dtype=fx.Float32)
    #         bf8_hi = fx.Vector(fx.rocdl.cvt_pk_f32_bf8(T.f32x2, fx.Int32(pk_bf8), True), dtype=fx.Float32)
    #
    #         out1d = fx.make_view(fx.get_iter(Out), fx.make_layout((8,), (1,)))
    #         out1d[0] = fp8_lo[0]
    #         out1d[1] = fp8_lo[1]
    #         out1d[2] = fp8_hi[0]
    #         out1d[3] = fp8_hi[1]
    #         out1d[4] = bf8_lo[0]
    #         out1d[5] = bf8_lo[1]
    #         out1d[6] = bf8_hi[0]
    #         out1d[7] = bf8_hi[1]
    #
    # @flyc.jit
    # def launch(Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    #     cvt_pk_unpack_kernel(Out).launch(grid=(1, 1, 1), block=(_WARP_SIZE, 1, 1), stream=stream)
    #
    # out = torch.full((8,), -999.0, dtype=torch.float32, device="cuda")
    # launch(out, stream=torch.cuda.current_stream())
    # torch.cuda.synchronize()
    #
    # expected = torch.tensor([1.5, -2.0, 0.25, 4.0, 1.5, -2.0, 0.25, 4.0], dtype=torch.float32)
    # torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
    pass


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
