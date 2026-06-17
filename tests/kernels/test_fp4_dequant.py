# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Unit tests for the gfx942 MX-FP4 (E2M1) software dequant helpers.

Covers:
- ``_fp4_code_to_f32_bits`` / ``dequant_fp4_to_bf16`` (E2M1 nibble -> bf16)
- ``e8m0_to_f32_inkernel`` (E8M0 block-scale byte -> f32)

The pure-Python mirror validates the integer bit-construction algorithm against
the trusted ``fp4_utils`` reference for all 16 E2M1 codes (no GPU needed). The
on-device test exercises the actual emitted kernel on gfx942.
"""

import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None

from tests.kernels.utils import fp4_utils

# E2M1 LUT (matches fp4_utils.mxfp4_to_f32 ordering): code -> value.
_E2M1_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def _py_fp4_code_to_f32_bits(code: int) -> int:
    """Pure-Python mirror of kernels._fp4_code_to_f32_bits (same integer math)."""
    code &= 0xF
    sign = (code & 0x8) << 28
    mag = code & 0x7
    exp_field = mag >> 1
    mant = mag & 0x1
    norm_bits = ((exp_field + 126) << 23) | (mant << 22)
    if mag == 0:
        mag_bits = 0x00000000
    elif mag == 1:
        mag_bits = 0x3F000000
    else:
        mag_bits = norm_bits
    return (mag_bits | sign) & 0xFFFFFFFF


def test_fp4_code_to_f32_bits_matches_lut():
    """Bit-construction reproduces the E2M1 LUT exactly for all 16 codes."""
    for code in range(16):
        bits = _py_fp4_code_to_f32_bits(code)
        got = np.frombuffer(np.uint32(bits).tobytes(), dtype=np.float32)[0]
        want = _E2M1_LUT[code]
        # -0.0 vs 0.0: compare numerically (both zero).
        assert got == want or (got == 0.0 and want == 0.0), f"code={code}: got {got}, want {want}"


def test_fp4_bf16_truncation_is_exact():
    """f32->bf16 lshr-16 truncation is exact for every E2M1 magnitude."""
    for code in range(16):
        bits = _py_fp4_code_to_f32_bits(code)
        # bf16 = top 16 bits; truncation is exact iff low 16 bits are zero.
        assert (bits & 0xFFFF) == 0, f"code={code}: f32 low bits nonzero ({bits:#010x})"


def test_e8m0_decode_matches_reference():
    """(byte<<23) bitcast == fp4_utils.e8m0_to_f32 for the normal range 1..254."""
    if torch is None:
        pytest.skip("torch not available")
    bytes_ = torch.arange(1, 255, dtype=torch.uint8)
    ref = fp4_utils.e8m0_to_f32(bytes_)
    got_bits = (bytes_.to(torch.int32) << 23)
    got = got_bits.view(torch.float32)
    assert torch.equal(got, ref), "in-kernel E8M0 decode diverges from reference in normal range"


# ─────────────────────────── On-device kernel test ───────────────────────────

_BLOCK = 256


def _build_dequant_launcher():
    """Build the FP4 dequant test kernel lazily (requires a built flydsl)."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir.dialects.arith import CmpIPredicate
    from flydsl.expr import arith, buffer_ops, vector
    from flydsl.expr.typing import T

    from kernels.mfma_preshuffle_pipeline import dequant_fp4_to_bf16

    @flyc.kernel
    def _fp4_dequant_kernel(packed: fx.Tensor, out: fx.Tensor, n_dwords: fx.Constexpr[int]):
        tid = fx.thread_idx.x + fx.block_idx.x * fx.Int32(_BLOCK)
        p_rsrc = buffer_ops.create_buffer_resource(packed, max_size=False)
        o_rsrc = buffer_ops.create_buffer_resource(out, max_size=False)
        if arith.cmpi(CmpIPredicate.slt, tid, fx.Int32(n_dwords)):
            packed32 = buffer_ops.buffer_load(p_rsrc, tid, vec_width=1, dtype=T.i32)
            b0, b1 = dequant_fp4_to_bf16(packed32, arith, vector)
            buffer_ops.buffer_store(b0, o_rsrc, tid * fx.Int32(2))
            buffer_ops.buffer_store(b1, o_rsrc, tid * fx.Int32(2) + fx.Int32(1))

    @flyc.jit
    def _fp4_dequant(
        packed: fx.Tensor,
        out: fx.Tensor,
        n_dwords: fx.Constexpr[int],
        grid_x: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        _fp4_dequant_kernel(packed, out, n_dwords).launch(grid=(grid_x, 1, 1), block=(_BLOCK, 1, 1), stream=stream)

    return _fp4_dequant


@pytest.mark.l2_device
@pytest.mark.rocm_lower
def test_dequant_fp4_to_bf16_on_device():
    """End-to-end: emitted kernel dequantizes all 16 codes correctly on device."""
    if torch is None or not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")
    try:
        _fp4_dequant = _build_dequant_launcher()
    except ImportError as e:
        pytest.skip(f"flydsl not built: {e}")

    rng = np.random.default_rng(0)
    n_dwords = 512
    # 8 nibbles per dword; force the first two dwords to cover all 16 codes.
    nibbles = rng.integers(0, 16, size=(n_dwords, 8), dtype=np.uint8)
    nibbles[0] = np.arange(8, dtype=np.uint8)
    nibbles[1] = np.arange(8, 16, dtype=np.uint8)

    # Pack: byte j = nibble[2j] | (nibble[2j+1] << 4); dword little-endian.
    bytes4 = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).astype(np.uint8)  # (n,4)
    packed_u32 = bytes4.view(np.uint32).reshape(n_dwords)

    packed_i32 = torch.from_numpy(packed_u32.view(np.int32).copy()).cuda()
    out_i64 = torch.zeros(n_dwords * 2, dtype=torch.int64, device="cuda")
    grid_x = (n_dwords + _BLOCK - 1) // _BLOCK

    _fp4_dequant(packed_i32, out_i64, n_dwords, grid_x)
    torch.cuda.synchronize()

    got = out_i64.view(torch.bfloat16).reshape(n_dwords, 8).float().cpu().numpy()

    # Kernel emits b0=[n0,n2,n4,n6], b1=[n1,n3,n5,n7] -> reorder reference to match.
    order = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    want = _E2M1_LUT[nibbles[:, order]]

    np.testing.assert_array_equal(got, want)
