# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device roundtrip tests for ROCDL conversion, intra-lane dot, DPP, and math DSL wrappers."""

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
_HAS_FDOT2_F16 = _ARCH.startswith(("gfx90a", "gfx94", "gfx95", "gfx11", "gfx120"))
_HAS_FDOT2_BF16 = _ARCH.startswith(("gfx95", "gfx11", "gfx120"))
_HAS_FDOT2_F16_BF16_ACC = _ARCH.startswith(("gfx11", "gfx120"))
_HAS_INT_DOT = _ARCH.startswith(("gfx90a", "gfx94", "gfx95", "gfx11", "gfx12"))
_HAS_MIXED_INT_DOT = _ARCH.startswith(("gfx11", "gfx12"))
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


def test_rocdl_exp2_rsq_and_update_dpp():
    @flyc.kernel
    def math_dpp_kernel(Out: fx.Tensor):
        tid = fx.Int32(fx.thread_idx.x)
        lane_val = (tid + fx.Int32(1)).to(fx.Float32)
        # 177 (0xB1) = quad_perm:[1,0,3,2] (XOR-1 neighbor swap within each 4-lane quad)
        swapped = fx.Float32(fx.rocdl.update_dpp(T.f32, lane_val, lane_val, 177, 0xF, 0xF, False))
        out1d = fx.make_view(fx.get_iter(Out), fx.make_layout((6,), (1,)))
        if tid == 0:
            out1d[0] = fx.Float32(fx.rocdl.exp2(T.f32, fx.Float32(3.0)))
            out1d[1] = fx.Float32(fx.rocdl.rsq(T.f32, fx.Float32(4.0)))
        if tid < 4:
            out1d[2 + tid] = swapped

    @flyc.jit
    def launch(Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        math_dpp_kernel(Out).launch(grid=(1, 1, 1), block=(_WARP_SIZE, 1, 1), stream=stream)

    out = torch.full((6,), -999.0, dtype=torch.float32, device="cuda")
    launch(out, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    expected = torch.tensor([8.0, 0.5, 2.0, 1.0, 4.0, 3.0], dtype=torch.float32)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not _HAS_FDOT2_F16, reason=f"rocdl.fdot2 not supported on {_ARCH}")
def test_rocdl_fdot2_f32_f16():
    @flyc.kernel
    def fdot2_kernel(Out: fx.Tensor):
        tid = fx.thread_idx.x
        if tid == 0:
            va = fx.rocdl.cvt_pkrtz(T.f16x2, fx.Float32(1.5), fx.Float32(-2.0))
            vb = fx.rocdl.cvt_pkrtz(T.f16x2, fx.Float32(4.0), fx.Float32(0.5))
            # 1.5 * 4.0 + (-2.0) * 0.5 + 1.25 = 6.25
            acc_vec = fx.Float32(fx.rocdl.fdot2(T.f32, va, vb, fx.Float32(1.25), clamp=False))
            # Packed i32 overload: 0x3C003C00 = (1.0, 1.0) in fp16 -> 1.0 + 1.0 + 0.5 = 2.5
            acc_i32 = fx.Float32(fx.rocdl.fdot2(T.f32, fx.Int32(0x3C003C00), fx.Int32(0x3C003C00), 0.5))
            out1d = fx.make_view(fx.get_iter(Out), fx.make_layout((2,), (1,)))
            out1d[0] = acc_vec
            out1d[1] = acc_i32

    @flyc.jit
    def launch(Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        fdot2_kernel(Out).launch(grid=(1, 1, 1), block=(_WARP_SIZE, 1, 1), stream=stream)

    out = torch.full((2,), -999.0, dtype=torch.float32, device="cuda")
    launch(out, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    torch.testing.assert_close(out.cpu(), torch.tensor([6.25, 2.5], dtype=torch.float32), atol=0, rtol=0)


@pytest.mark.skipif(not _HAS_FDOT2_BF16, reason=f"rocdl.fdot2_f32_bf16 not supported on {_ARCH}")
def test_rocdl_fdot2_f32_bf16():
    @flyc.kernel
    def fdot2_bf16_kernel(Out: fx.Tensor):
        tid = fx.thread_idx.x
        if tid == 0:
            # Pack (1.5, -2.0) and (4.0, 0.5) into bf16x2 i32 words via perm_b32
            a0 = fx.Float32(1.5).bitcast(fx.Int32)
            a1 = fx.Float32(-2.0).bitcast(fx.Int32)
            b0 = fx.Float32(4.0).bitcast(fx.Int32)
            b1 = fx.Float32(0.5).bitcast(fx.Int32)
            pa = fx.Int32(fx.rocdl.perm_b32(a1, a0, 0x07060302))
            pb = fx.Int32(fx.rocdl.perm_b32(b1, b0, 0x07060302))
            # 1.5 * 4.0 + (-2.0) * 0.5 + 0.75 = 5.75
            res = fx.Float32(fx.rocdl.fdot2_f32_bf16(T.f32, pa, pb, fx.Float32(0.75), clamp=False))
            out1d = fx.make_view(fx.get_iter(Out), fx.make_layout((1,), (1,)))
            out1d[0] = res

    @flyc.jit
    def launch(Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        fdot2_bf16_kernel(Out).launch(grid=(1, 1, 1), block=(_WARP_SIZE, 1, 1), stream=stream)

    out = torch.full((1,), -999.0, dtype=torch.float32, device="cuda")
    launch(out, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    torch.testing.assert_close(out.cpu(), torch.tensor([5.75], dtype=torch.float32), atol=0, rtol=0)


@pytest.mark.skipif(
    not _HAS_FDOT2_F16_BF16_ACC,
    reason=f"rocdl.fdot2_{{bf16_bf16,f16_f16}} not supported on {_ARCH}",
)
def test_rocdl_fdot2_f16_and_bf16_accum():
    @flyc.kernel
    def fdot2_acc_kernel(Out: fx.Tensor):
        tid = fx.thread_idx.x
        if tid == 0:
            # bf16: (2.0, -1.0) . (3.0, 0.5) + 1.0 = 6.5
            pa = fx.Int32(
                fx.rocdl.perm_b32(fx.Float32(-1.0).bitcast(fx.Int32), fx.Float32(2.0).bitcast(fx.Int32), 0x07060302)
            )
            pb = fx.Int32(
                fx.rocdl.perm_b32(fx.Float32(0.5).bitcast(fx.Int32), fx.Float32(3.0).bitcast(fx.Int32), 0x07060302)
            )
            r_bf16 = fx.BFloat16(fx.rocdl.fdot2_bf16_bf16(T.bf16, pa, pb, fx.BFloat16(1.0)))

            # f16: (2.0, -1.0) . (3.0, 0.5) + 0.5 = 6.0
            va = fx.rocdl.cvt_pkrtz(T.f16x2, fx.Float32(2.0), fx.Float32(-1.0))
            vb = fx.rocdl.cvt_pkrtz(T.f16x2, fx.Float32(3.0), fx.Float32(0.5))
            r_f16 = fx.Float16(fx.rocdl.fdot2_f16_f16(T.f16, va, vb, fx.Float16(0.5)))

            out1d = fx.make_view(fx.get_iter(Out), fx.make_layout((2,), (1,)))
            out1d[0] = r_bf16.to(fx.Float32)
            out1d[1] = r_f16.to(fx.Float32)

    @flyc.jit
    def launch(Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        fdot2_acc_kernel(Out).launch(grid=(1, 1, 1), block=(_WARP_SIZE, 1, 1), stream=stream)

    out = torch.full((2,), -999.0, dtype=torch.float32, device="cuda")
    launch(out, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    # 2.0 * 3.0 + (-1.0) * 0.5 = 5.5 -> + 1.0 = 6.5, + 0.5 = 6.0
    torch.testing.assert_close(out.cpu(), torch.tensor([6.5, 6.0], dtype=torch.float32), atol=0, rtol=0)


@pytest.mark.skipif(not _HAS_INT_DOT, reason=f"rocdl.sdot4/udot4/sdot8/udot8 not supported on {_ARCH}")
def test_rocdl_sdot_and_udot():
    @flyc.kernel
    def int_dot_kernel(Out: fx.Tensor):
        tid = fx.thread_idx.x
        if tid == 0:
            # 0xFF020103 as signed i8 = [3, 1, 2, -1]; 0x04030201 = [1, 2, 3, 4]
            # sdot4: 3*1 + 1*2 + 2*3 + (-1)*4 + 10 = 17
            a4 = fx.Int32(-16645885)  # 0xFF020103
            b4 = fx.Int32(0x04030201)
            sd4 = fx.Int32(fx.rocdl.sdot4(T.i32, a4, b4, 10, clamp=False))
            # udot4: 3*1 + 1*2 + 2*3 + 255*4 + 5 = 1036
            ud4 = fx.Int32(fx.rocdl.udot4(T.i32, a4, b4, 5, clamp=False))
            # 0xF2121212 nibbles: signed=[2,1,2,1,2,1,2,-1], unsigned=[2,1,2,1,2,1,2,15]
            # against 0x11111111 (all 1s): sdot8 = 10, udot8 = 26
            a8 = fx.Int32(-233696750)  # 0xF2121212
            b8 = fx.Int32(0x11111111)
            sd8 = fx.Int32(fx.rocdl.sdot8(T.i32, a8, b8, 0, clamp=False))
            ud8 = fx.Int32(fx.rocdl.udot8(T.i32, a8, b8, 0, clamp=False))

            out1d = fx.make_view(fx.get_iter(Out), fx.make_layout((4,), (1,)))
            out1d[0] = sd4
            out1d[1] = ud4
            out1d[2] = sd8
            out1d[3] = ud8

    @flyc.jit
    def launch(Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        int_dot_kernel(Out).launch(grid=(1, 1, 1), block=(_WARP_SIZE, 1, 1), stream=stream)

    out = torch.full((4,), -999, dtype=torch.int32, device="cuda")
    launch(out, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    torch.testing.assert_close(out.cpu(), torch.tensor([17, 1036, 10, 26], dtype=torch.int32), atol=0, rtol=0)


@pytest.mark.skipif(not _HAS_MIXED_INT_DOT, reason=f"rocdl.sudot4/sudot8 not supported on {_ARCH}")
def test_rocdl_sudot():
    @flyc.kernel
    def sudot_kernel(Out: fx.Tensor):
        tid = fx.thread_idx.x
        if tid == 0:
            # a signed [3, 1, 2, -1], b unsigned [1, 2, 3, 200] -> 3 + 2 + 6 - 200 = -189
            a4 = fx.Int32(-16645885)  # 0xFF020103
            b4 = fx.Int32(-939326975)  # 0xC8030201
            sud4 = fx.Int32(fx.rocdl.sudot4(T.i32, a4, b4, 0, sign_a=True, sign_b=False, clamp=False))
            # a signed [-1, 2, 1, 2, 1, 2, 1, 2], b unsigned [10, 1, 1, 1, 1, 1, 1, 1]
            # nibbles low-to-high of 0xF2121212: [2,1,2,1,2,1,2,-1]; of 0xA1111111: [1,1,1,1,1,1,1,10]
            # sum = 2+1+2+1+2+1+2 + (-1)*10 = 1
            a8 = fx.Int32(-233696750)  # 0xF2121212
            b8 = fx.Int32(-1592717039)  # 0xA1111111
            sud8 = fx.Int32(fx.rocdl.sudot8(T.i32, a8, b8, 0, sign_a=True, sign_b=False, clamp=False))

            out1d = fx.make_view(fx.get_iter(Out), fx.make_layout((2,), (1,)))
            out1d[0] = sud4
            out1d[1] = sud8

    @flyc.jit
    def launch(Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        sudot_kernel(Out).launch(grid=(1, 1, 1), block=(_WARP_SIZE, 1, 1), stream=stream)

    out = torch.full((2,), -999, dtype=torch.int32, device="cuda")
    launch(out, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    torch.testing.assert_close(out.cpu(), torch.tensor([-189, 1], dtype=torch.int32), atol=0, rtol=0)


@pytest.mark.skip(
    reason=(
        "cvt_pk_f32_bf8, fp8x4_to_bf16x2_pair, and bf8x4_to_bf16x2_pair are commented "
        "out until upstream llvm/llvm-project#231721 lands (ROCm/FlyDSL#1376): unmodified "
        "LLVM VOP1Instructions.td maps HasCvtFP8ByteSel v_cvt_pk_f32_{fp8,bf8} only to "
        "_fake16 pseudos (omitting UseRealTrue16Insts _t16 ISel patterns)."
    )
)
def test_rocdl_pk_f32_fp8_bf8_and_bf16x2_unpack():
    """Uncomment alongside True16 unpack wrappers once llvm/llvm-project#231721 lands (ROCm/FlyDSL#1376)."""
    # @flyc.kernel
    # def cvt_pk_unpack_kernel(In: fx.Tensor, Out: fx.Tensor):
    #     tid = fx.thread_idx.x
    #     in2d = fx.make_view(fx.get_iter(In), fx.make_layout((_WARP_SIZE, 4), (4, 1)))
    #     a = in2d[tid, 0]
    #     b = in2d[tid, 1]
    #     c = in2d[tid, 2]
    #     d = in2d[tid, 3]
    #
    #     pk_fp8_lo = fx.rocdl.cvt_pk_fp8_f32(T.i32, a, b, 0, False)
    #     pk_fp8 = fx.rocdl.cvt_pk_fp8_f32(T.i32, c, d, fx.Int32(pk_fp8_lo), True)
    #     pk_bf8_lo = fx.rocdl.cvt_pk_bf8_f32(T.i32, a, b, 0, False)
    #     pk_bf8 = fx.rocdl.cvt_pk_bf8_f32(T.i32, c, d, fx.Int32(pk_bf8_lo), True)
    #
    #     fp8_lo = fx.Vector(fx.rocdl.cvt_pk_f32_fp8(T.f32x2, fx.Int32(pk_fp8), False), dtype=fx.Float32)
    #     fp8_hi = fx.Vector(fx.rocdl.cvt_pk_f32_fp8(T.f32x2, fx.Int32(pk_fp8), True), dtype=fx.Float32)
    #     bf8_lo = fx.Vector(fx.rocdl.cvt_pk_f32_bf8(T.f32x2, fx.Int32(pk_bf8), False), dtype=fx.Float32)
    #     bf8_hi = fx.Vector(fx.rocdl.cvt_pk_f32_bf8(T.f32x2, fx.Int32(pk_bf8), True), dtype=fx.Float32)
    #
    #     fp8_b0, fp8_b1 = fx.rocdl.fp8x4_to_bf16x2_pair(fx.Int32(pk_fp8))
    #     bf8_b0, bf8_b1 = fx.rocdl.bf8x4_to_bf16x2_pair(fx.Int32(pk_bf8))
    #     ones_bf16x2 = fx.Int32(0x3F803F80)
    #     sum_fp8 = fx.Float32(
    #         fx.rocdl.fdot2_f32_bf16(
    #             T.f32,
    #             fp8_b1,
    #             ones_bf16x2,
    #             fx.rocdl.fdot2_f32_bf16(T.f32, fp8_b0, ones_bf16x2, 0.0),
    #         )
    #     )
    #     sum_bf8 = fx.Float32(
    #         fx.rocdl.fdot2_f32_bf16(
    #             T.f32,
    #             bf8_b1,
    #             ones_bf16x2,
    #             fx.rocdl.fdot2_f32_bf16(T.f32, bf8_b0, ones_bf16x2, 0.0),
    #         )
    #     )
    #
    #     if tid == 0:
    #         out1d = fx.make_view(fx.get_iter(Out), fx.make_layout((10,), (1,)))
    #         out1d[0] = fp8_lo[0]
    #         out1d[1] = fp8_lo[1]
    #         out1d[2] = fp8_hi[0]
    #         out1d[3] = fp8_hi[1]
    #         out1d[4] = bf8_lo[0]
    #         out1d[5] = bf8_lo[1]
    #         out1d[6] = bf8_hi[0]
    #         out1d[7] = bf8_hi[1]
    #         out1d[8] = sum_fp8
    #         out1d[9] = sum_bf8
    #
    # @flyc.jit
    # def launch(In: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    #     cvt_pk_unpack_kernel(In, Out).launch(grid=(1, 1, 1), block=(_WARP_SIZE, 1, 1), stream=stream)
    #
    # inp = torch.tensor([1.5, -2.0, 0.25, 4.0], dtype=torch.float32, device="cuda").repeat(_WARP_SIZE, 1).contiguous()
    # out = torch.full((10,), -999.0, dtype=torch.float32, device="cuda")
    # launch(inp, out, stream=torch.cuda.current_stream())
    # torch.cuda.synchronize()
    #
    # expected = torch.tensor([1.5, -2.0, 0.25, 4.0, 1.5, -2.0, 0.25, 4.0, 3.75, 3.75], dtype=torch.float32)
    # torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
