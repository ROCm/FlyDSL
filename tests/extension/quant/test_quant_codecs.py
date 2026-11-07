#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Unit and GPU numerical tests for ``flydsl.extension.quant`` (`fx.quant`)."""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    import torch
except ImportError:
    torch = None


_E2M1_TABLE = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _f32_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(val)))[0]


def _bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _ref_e4m3_decode(byte_val: int) -> float:
    b = byte_val & 0xFF
    sign = -1.0 if (b & 0x80) else 1.0
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    if exp == 0xF and mant == 0x7:
        return float("nan")
    if exp == 0:
        return sign * (mant * (2.0**-9))
    return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))


def _ref_e5m2_decode(byte_val: int) -> float:
    b = byte_val & 0xFF
    sign = -1.0 if (b & 0x80) else 1.0
    exp = (b >> 2) & 0x1F
    mant = b & 0x3
    if exp == 0x1F:
        return sign * float("inf") if mant == 0 else float("nan")
    if exp == 0:
        return sign * (mant * (2.0**-16))
    return sign * (1.0 + mant / 4.0) * (2.0 ** (exp - 15))


def _is_nan_f32_bits(bits: int) -> bool:
    return (bits & 0x7FFFFFFF) > 0x7F800000


@pytest.mark.l0_backend_agnostic
def test_quant_public_api():
    expected = [
        "f32_to_e8m0",
        "e8m0_to_f32",
        "f32_to_ue4m3",
        "ue4m3_to_f32",
        "pack_bf16x2",
        "unpack_bf16x2",
        "pack_e4m3x2",
        "pack_e4m3x4",
        "unpack_e4m3x4",
        "unpack_e4m3x4_bf16x2",
        "stoch_round_e4m3x4",
        "pack_e5m2x2",
        "pack_e5m2x4",
        "unpack_e5m2x4",
        "unpack_e5m2x4_bf16x2",
        "stoch_round_e5m2x4",
        "pack_e2m1x8",
        "unpack_e2m1x4",
        "unpack_e2m1x8",
        "unpack_e2m1x4_bf16x2",
        "unpack_e2m1x4_e4m3x4",
        "unpack_e2m1x8_int4_pair",
        "pack_int4x8",
        "unpack_int4x8",
        "unpack_uint4x8",
    ]
    assert fx.quant.__all__ == expected
    assert fx.quant.universal.__all__ == expected


@pytest.mark.l1a_compile_no_target_dialect
def test_universal_codecs_build_portable_ir():
    @flyc.jit
    def trace_all(x: fx.Float32, code: fx.Uint8, word: fx.Uint32):
        e8_up = fx.quant.universal.f32_to_e8m0(x, round_up=True)
        e8_rne = fx.quant.universal.f32_to_e8m0(x, round_up=False)
        f_e8 = fx.quant.universal.e8m0_to_f32(code)
        ue = fx.quant.universal.f32_to_ue4m3(x)
        f_ue = fx.quant.universal.ue4m3_to_f32(ue)
        bf = fx.quant.universal.pack_bf16x2(f_e8, f_ue)
        b0, b1 = fx.quant.universal.unpack_bf16x2(bf)
        fp8 = fx.quant.universal.pack_e4m3x4(b0, b1, x, -x)
        u0, u1, u2, u3 = fx.quant.universal.unpack_e4m3x4(fp8)
        sr4 = fx.quant.universal.stoch_round_e4m3x4(u0, u1, u2, u3, word)
        fp4 = fx.quant.universal.pack_e2m1x8((u0, u1, u2, u3, b0, b1, x, -x))
        d8 = fx.quant.universal.unpack_e2m1x8(fp4)
        e4_from_fp4 = fx.quant.universal.unpack_e2m1x4_e4m3x4(fp4, half=0)
        p_w, q_w = fx.quant.universal.unpack_e2m1x8_int4_pair(fp4)
        i4_vals = fx.quant.universal.unpack_int4x8(p_w)
        i4_pk = fx.quant.universal.pack_int4x8(i4_vals)
        u4_vals = fx.quant.universal.unpack_uint4x8(q_w)
        (
            fx.Uint32(e8_up) ^ fx.Uint32(e8_rne) ^ sr4 ^ e4_from_fp4 ^ i4_pk ^ u4_vals[0] ^ d8[0].bitcast(fx.Uint32)
        ).ir_value()

    trace_all(1.5, 127, 0x12345678)
    source = trace_all._last_compiled[1].source_ir
    assert "rocdl." not in source
    assert "llvm.amdgcn" not in source
    assert "arith.bitcast" in source


@pytest.mark.l0_backend_agnostic
def test_compile_time_argument_validation():
    # Pure unsigned integer codecs fold on host literals without an MLIR context.
    ints = [-8, -7, -3, -1, 0, 1, 5, 7]
    packed_i4 = fx.quant.universal.pack_int4x8([fx.Uint32(v & 0xF) for v in ints])
    assert [int(v.value) for v in fx.quant.universal.unpack_uint4x8(packed_i4)] == [v & 0xF for v in ints]

    with pytest.raises(ValueError, match="half"):
        fx.quant.universal.unpack_e2m1x4(fx.Uint32(0), half=2)
    with pytest.raises(ValueError, match="expected 8"):
        fx.quant.universal.pack_e2m1x8([fx.Float32(0.0)] * 7)
    with pytest.raises(ValueError, match="expected 4"):
        fx.quant.universal.stoch_round_e4m3x4(1.0, 1.0, 1.0, 1.0, (1, 2, 3))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_scale_and_bf16x2_codecs_device_parity():
    BLOCK = 256

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(
        Vals: fx.Tensor,
        E8Dec: fx.Tensor,
        UeDecHw: fx.Tensor,
        UeDecUni: fx.Tensor,
        CodesHw: fx.Tensor,
        CodesUni: fx.Tensor,
        BfOut: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        b = fx.Uint8(tid)
        E8Dec[tid] = fx.quant.e8m0_to_f32(b)
        UeDecHw[tid] = fx.quant.ue4m3_to_f32(b)
        UeDecUni[tid] = fx.quant.universal.ue4m3_to_f32(b)

        v = Vals[tid]
        e8_up = fx.Uint32(fx.quant.f32_to_e8m0(v, round_up=True))
        e8_rne = fx.Uint32(fx.quant.f32_to_e8m0(v, round_up=False))
        ue_hw = fx.Uint32(fx.quant.f32_to_ue4m3(v))
        ue_uni = fx.Uint32(fx.quant.universal.f32_to_ue4m3(v))
        CodesHw[tid] = (e8_up | (e8_rne << fx.Uint32(8)) | (ue_hw << fx.Uint32(16))).bitcast(fx.Int32)
        CodesUni[tid] = (e8_up | (e8_rne << fx.Uint32(8)) | (ue_uni << fx.Uint32(16))).bitcast(fx.Int32)

        bf_pk = fx.quant.pack_bf16x2(v, -v)
        u0, u1 = fx.quant.unpack_bf16x2(bf_pk)
        BfOut[tid * 2] = u0
        BfOut[tid * 2 + 1] = u1

    @flyc.jit
    def launch(
        Vals: fx.Tensor,
        E8Dec: fx.Tensor,
        UeDecHw: fx.Tensor,
        UeDecUni: fx.Tensor,
        CodesHw: fx.Tensor,
        CodesUni: fx.Tensor,
        BfOut: fx.Tensor,
    ):
        kernel(Vals, E8Dec, UeDecHw, UeDecUni, CodesHw, CodesUni, BfOut).launch(grid=(1, 1, 1), block=(BLOCK, 1, 1))

    special_vals = [
        0.0,
        -0.0,
        -1.0,
        2.0**-127,
        1.5 * 2.0**-127,
        2.0**-126,
        2.0**-10,
        2.0**-9,
        1.5 * 2.0**-9,
        0.125,
        0.375,
        1.0,
        1.0625,
        1.5,
        3.5,
        448.0,
        464.0,
        500.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ]
    rng = np.random.default_rng(123)
    random_pos = np.exp2(rng.uniform(-14.0, 10.0, size=BLOCK - len(special_vals))).astype(np.float32).tolist()
    vals = torch.tensor(special_vals + random_pos, dtype=torch.float32, device="cuda")

    e8_dec = torch.empty(BLOCK, dtype=torch.float32, device="cuda")
    ue_hw = torch.empty(BLOCK, dtype=torch.float32, device="cuda")
    ue_uni = torch.empty(BLOCK, dtype=torch.float32, device="cuda")
    codes_hw = torch.empty(BLOCK, dtype=torch.int32, device="cuda")
    codes_uni = torch.empty(BLOCK, dtype=torch.int32, device="cuda")
    bf_out = torch.empty(BLOCK * 2, dtype=torch.float32, device="cuda")

    launch(vals, e8_dec, ue_hw, ue_uni, codes_hw, codes_uni, bf_out)
    torch.cuda.synchronize()

    # Exact parity between ROCDL hardware UE4M3 and universal UE4M3 across all 256 codes and inputs.
    assert torch.equal(codes_hw.cpu(), codes_uni.cpu())
    for code in range(256):
        hw_bits = _f32_bits(ue_hw[code].item())
        uni_bits = _f32_bits(ue_uni[code].item())
        if (code & 0x7F) == 0x7F:
            assert _is_nan_f32_bits(hw_bits) and _is_nan_f32_bits(uni_bits)
        else:
            assert hw_bits == uni_bits
            assert ue_hw[code].item() == _ref_e4m3_decode(code & 0x7F)

        e8_val = e8_dec[code].item()
        if code == 255:
            assert math.isnan(e8_val)
        else:
            assert e8_val == float(np.float32(2.0 ** (code - 127)))

    # BF16x2 RNE parity with PyTorch bfloat16 conversion on finite inputs.
    finite_mask = torch.isfinite(vals.cpu())
    ref_bf0 = vals.cpu()[finite_mask].to(torch.bfloat16).to(torch.float32)
    ref_bf1 = (-vals.cpu()[finite_mask]).to(torch.bfloat16).to(torch.float32)
    bf_cpu = bf_out.cpu().reshape(BLOCK, 2)
    assert torch.equal(bf_cpu[finite_mask, 0], ref_bf0)
    assert torch.equal(bf_cpu[finite_mask, 1], ref_bf1)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_fp8_e4m3_e5m2_all_256_codes_and_pack_parity():
    BLOCK = 64  # 64 threads * 4 bytes = all 256 FP8 byte patterns!

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def unpack_all_kernel(
        Packed: fx.Tensor,
        E4Hw: fx.Tensor,
        E4Uni: fx.Tensor,
        E4BfHw: fx.Tensor,
        E4BfUni: fx.Tensor,
        E5Hw: fx.Tensor,
        E5Uni: fx.Tensor,
        E5BfHw: fx.Tensor,
        E5BfUni: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        w = Packed[tid].bitcast(fx.Uint32)
        a0, a1, a2, a3 = fx.quant.unpack_e4m3x4(w)
        u0, u1, u2, u3 = fx.quant.universal.unpack_e4m3x4(w)
        b0, b1, b2, b3 = fx.quant.unpack_e5m2x4(w)
        v0, v1, v2, v3 = fx.quant.universal.unpack_e5m2x4(w)

        base = tid * 4
        E4Hw[base + 0], E4Hw[base + 1], E4Hw[base + 2], E4Hw[base + 3] = a0, a1, a2, a3
        E4Uni[base + 0], E4Uni[base + 1], E4Uni[base + 2], E4Uni[base + 3] = u0, u1, u2, u3
        E5Hw[base + 0], E5Hw[base + 1], E5Hw[base + 2], E5Hw[base + 3] = b0, b1, b2, b3
        E5Uni[base + 0], E5Uni[base + 1], E5Uni[base + 2], E5Uni[base + 3] = v0, v1, v2, v3

        e4_bf0, e4_bf1 = fx.quant.unpack_e4m3x4_bf16x2(w)
        e4_bfu0, e4_bfu1 = fx.quant.universal.unpack_e4m3x4_bf16x2(w)
        e5_bf0, e5_bf1 = fx.quant.unpack_e5m2x4_bf16x2(w)
        e5_bfu0, e5_bfu1 = fx.quant.universal.unpack_e5m2x4_bf16x2(w)
        E4BfHw[tid * 2], E4BfHw[tid * 2 + 1] = e4_bf0.bitcast(fx.Int32), e4_bf1.bitcast(fx.Int32)
        E4BfUni[tid * 2], E4BfUni[tid * 2 + 1] = e4_bfu0.bitcast(fx.Int32), e4_bfu1.bitcast(fx.Int32)
        E5BfHw[tid * 2], E5BfHw[tid * 2 + 1] = e5_bf0.bitcast(fx.Int32), e5_bf1.bitcast(fx.Int32)
        E5BfUni[tid * 2], E5BfUni[tid * 2 + 1] = e5_bfu0.bitcast(fx.Int32), e5_bfu1.bitcast(fx.Int32)

    @flyc.jit
    def launch_unpack_all(
        Packed: fx.Tensor,
        E4Hw: fx.Tensor,
        E4Uni: fx.Tensor,
        E4BfHw: fx.Tensor,
        E4BfUni: fx.Tensor,
        E5Hw: fx.Tensor,
        E5Uni: fx.Tensor,
        E5BfHw: fx.Tensor,
        E5BfUni: fx.Tensor,
    ):
        unpack_all_kernel(Packed, E4Hw, E4Uni, E4BfHw, E4BfUni, E5Hw, E5Uni, E5BfHw, E5BfUni).launch(
            grid=(1, 1, 1),
            block=(BLOCK, 1, 1),
        )

    all_bytes = np.arange(256, dtype=np.uint8).view(np.int32)
    packed_dev = torch.from_numpy(all_bytes.copy()).to("cuda")
    e4_hw = torch.empty(256, dtype=torch.float32, device="cuda")
    e4_uni = torch.empty(256, dtype=torch.float32, device="cuda")
    e5_hw = torch.empty(256, dtype=torch.float32, device="cuda")
    e5_uni = torch.empty(256, dtype=torch.float32, device="cuda")
    e4_bf_hw = torch.empty(128, dtype=torch.int32, device="cuda")
    e4_bf_uni = torch.empty(128, dtype=torch.int32, device="cuda")
    e5_bf_hw = torch.empty(128, dtype=torch.int32, device="cuda")
    e5_bf_uni = torch.empty(128, dtype=torch.int32, device="cuda")

    launch_unpack_all(packed_dev, e4_hw, e4_uni, e4_bf_hw, e4_bf_uni, e5_hw, e5_uni, e5_bf_hw, e5_bf_uni)
    torch.cuda.synchronize()

    for code in range(256):
        ref4 = _ref_e4m3_decode(code)
        hw4_b = _f32_bits(e4_hw[code].item())
        uni4_b = _f32_bits(e4_uni[code].item())
        if math.isnan(ref4):
            assert _is_nan_f32_bits(hw4_b) and _is_nan_f32_bits(uni4_b)
        else:
            assert hw4_b == uni4_b == _f32_bits(ref4), f"E4M3 code 0x{code:02X}"

        ref5 = _ref_e5m2_decode(code)
        hw5_b = _f32_bits(e5_hw[code].item())
        uni5_b = _f32_bits(e5_uni[code].item())
        if math.isnan(ref5):
            assert _is_nan_f32_bits(hw5_b) and _is_nan_f32_bits(uni5_b)
        else:
            assert hw5_b == uni5_b == _f32_bits(ref5), f"E5M2 code 0x{code:02X}"

    # Check BF16x2 words match across all non-NaN halves (NaN payload bits may differ by quiet bit).
    e4_bf_hw_u16 = e4_bf_hw.cpu().numpy().view(np.uint16)
    e4_bf_uni_u16 = e4_bf_uni.cpu().numpy().view(np.uint16)
    e5_bf_hw_u16 = e5_bf_hw.cpu().numpy().view(np.uint16)
    e5_bf_uni_u16 = e5_bf_uni.cpu().numpy().view(np.uint16)
    for code in range(256):
        if (code & 0x7F) != 0x7F:
            assert e4_bf_hw_u16[code] == e4_bf_uni_u16[code], f"E4M3 bf16 code 0x{code:02X}"
        if (code & 0x7F) <= 0x7C:
            assert e5_bf_hw_u16[code] == e5_bf_uni_u16[code], f"E5M2 bf16 code 0x{code:02X}"

    # Now test pack_e4m3x4 and pack_e5m2x4 (saturate=True & saturate=False + scale_inv) hardware vs universal.
    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def pack_kernel(
        In: fx.Tensor,
        E4SatHw: fx.Tensor,
        E4SatUni: fx.Tensor,
        E4NosatHw: fx.Tensor,
        E4NosatUni: fx.Tensor,
        E5SatHw: fx.Tensor,
        E5SatUni: fx.Tensor,
        E5NosatHw: fx.Tensor,
        E5NosatUni: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        b = tid * 4
        v0, v1, v2, v3 = In[b + 0], In[b + 1], In[b + 2], In[b + 3]
        E4SatHw[tid] = fx.quant.pack_e4m3x4(v0, v1, v2, v3, scale_inv=0.5, saturate=True).bitcast(fx.Int32)
        E4SatUni[tid] = fx.quant.universal.pack_e4m3x4(v0, v1, v2, v3, scale_inv=0.5, saturate=True).bitcast(fx.Int32)
        E4NosatHw[tid] = fx.quant.pack_e4m3x4(v0, v1, v2, v3, saturate=False).bitcast(fx.Int32)
        E4NosatUni[tid] = fx.quant.universal.pack_e4m3x4(v0, v1, v2, v3, saturate=False).bitcast(fx.Int32)

        E5SatHw[tid] = fx.quant.pack_e5m2x4(v0, v1, v2, v3, scale_inv=0.5, saturate=True).bitcast(fx.Int32)
        E5SatUni[tid] = fx.quant.universal.pack_e5m2x4(v0, v1, v2, v3, scale_inv=0.5, saturate=True).bitcast(fx.Int32)
        E5NosatHw[tid] = fx.quant.pack_e5m2x4(v0, v1, v2, v3, saturate=False).bitcast(fx.Int32)
        E5NosatUni[tid] = fx.quant.universal.pack_e5m2x4(v0, v1, v2, v3, saturate=False).bitcast(fx.Int32)

    @flyc.jit
    def launch_pack(
        In: fx.Tensor,
        E4SatHw: fx.Tensor,
        E4SatUni: fx.Tensor,
        E4NosatHw: fx.Tensor,
        E4NosatUni: fx.Tensor,
        E5SatHw: fx.Tensor,
        E5SatUni: fx.Tensor,
        E5NosatHw: fx.Tensor,
        E5NosatUni: fx.Tensor,
    ):
        pack_kernel(In, E4SatHw, E4SatUni, E4NosatHw, E4NosatUni, E5SatHw, E5SatUni, E5NosatHw, E5NosatUni).launch(
            grid=(1, 1, 1),
            block=(BLOCK, 1, 1),
        )

    edge_inputs = [
        0.0,
        -0.0,
        0.5 * 2**-9,
        1.5 * 2**-9,
        2.5 * 2**-9,
        7.5 * 2**-9,
        0.5 * 2**-16,
        1.5 * 2**-16,
        3.5 * 2**-16,
        1.0,
        1.0625,
        1.125,
        1.1875,
        -1.0625,
        -1.1875,
        448.0,
        -448.0,
        464.0,
        -464.0,
        480.0,
        -480.0,
        896.0,
        1000.0,
        57344.0,
        -57344.0,
        61440.0,
        -61440.0,
        65536.0,
        -65536.0,
        120000.0,
        float("inf"),
        float("-inf"),
    ]
    rng = np.random.default_rng(77)
    extra = rng.standard_normal(256 - len(edge_inputs)).astype(np.float32) * 50.0
    in_dev = torch.tensor(edge_inputs + extra.tolist(), dtype=torch.float32, device="cuda")
    outs = [torch.empty(BLOCK, dtype=torch.int32, device="cuda") for _ in range(8)]
    launch_pack(in_dev, *outs)
    torch.cuda.synchronize()

    assert torch.equal(outs[0].cpu(), outs[1].cpu())
    assert torch.equal(outs[2].cpu(), outs[3].cpu())
    assert torch.equal(outs[4].cpu(), outs[5].cpu())
    assert torch.equal(outs[6].cpu(), outs[7].cpu())


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_stochastic_rounding_parity_and_unbiasedness():
    BLOCK = 256
    NBLOCKS = 4
    TOTAL = BLOCK * NBLOCKS  # 1024 threads * 4 lanes = 4096 stochastic samples!

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def sr_kernel(
        Vals: fx.Tensor,
        SrE4Hw: fx.Tensor,
        SrE4Uni: fx.Tensor,
        SrE5Hw: fx.Tensor,
        SrE5Uni: fx.Tensor,
        DeqE4: fx.Tensor,
        DeqE5: fx.Tensor,
    ):
        g = fx.block_idx.x * BLOCK + fx.thread_idx.x
        entropy = fx.random.randint4x(fx.Uint32(1337), fx.Uint32(g))
        b = g * 4
        v0, v1, v2, v3 = Vals[b + 0], Vals[b + 1], Vals[b + 2], Vals[b + 3]

        w4_hw = fx.quant.stoch_round_e4m3x4(v0, v1, v2, v3, entropy, saturate=True)
        w4_uni = fx.quant.universal.stoch_round_e4m3x4(v0, v1, v2, v3, entropy, saturate=True)
        w5_hw = fx.quant.stoch_round_e5m2x4(v0, v1, v2, v3, entropy, saturate=True)
        w5_uni = fx.quant.universal.stoch_round_e5m2x4(v0, v1, v2, v3, entropy, saturate=True)
        SrE4Hw[g] = w4_hw.bitcast(fx.Int32)
        SrE4Uni[g] = w4_uni.bitcast(fx.Int32)
        SrE5Hw[g] = w5_hw.bitcast(fx.Int32)
        SrE5Uni[g] = w5_uni.bitcast(fx.Int32)

        # Unbiasedness probe at 1.03125 (25% between 1.0 and 1.125 in E4M3FN)
        # and 1.0625 (25% between 1.0 and 1.25 in E5M2).
        c4 = fx.Float32(1.03125)
        c5 = fx.Float32(1.0625)
        p4 = fx.quant.stoch_round_e4m3x4(c4, c4, c4, c4, entropy)
        p5 = fx.quant.stoch_round_e5m2x4(c5, c5, c5, c5, entropy)
        d4 = fx.quant.unpack_e4m3x4(p4)
        d5 = fx.quant.unpack_e5m2x4(p5)
        for i in fx.range_constexpr(4):
            DeqE4[b + i] = d4[i]
            DeqE5[b + i] = d5[i]

    @flyc.jit
    def launch_sr(
        Vals: fx.Tensor,
        SrE4Hw: fx.Tensor,
        SrE4Uni: fx.Tensor,
        SrE5Hw: fx.Tensor,
        SrE5Uni: fx.Tensor,
        DeqE4: fx.Tensor,
        DeqE5: fx.Tensor,
    ):
        sr_kernel(Vals, SrE4Hw, SrE4Uni, SrE5Hw, SrE5Uni, DeqE4, DeqE5).launch(
            grid=(NBLOCKS, 1, 1),
            block=(BLOCK, 1, 1),
        )

    rng = np.random.default_rng(2026)
    vals_np = rng.uniform(-600.0, 600.0, size=TOTAL * 4).astype(np.float32)
    # Inject subnormals, exact boundaries, and infinities into the first 16 slots.
    vals_np[:16] = np.array(
        [
            0.0,
            -0.0,
            0.5 * 2**-9,
            7.5 * 2**-9,
            0.5 * 2**-16,
            3.5 * 2**-16,
            1.0625,
            -1.0625,
            448.0,
            -448.0,
            500.0,
            -500.0,
            57344.0,
            -57344.0,
            float("inf"),
            float("-inf"),
        ],
        dtype=np.float32,
    )
    vals_dev = torch.from_numpy(vals_np).to("cuda")
    sr_e4_hw = torch.empty(TOTAL, dtype=torch.int32, device="cuda")
    sr_e4_uni = torch.empty(TOTAL, dtype=torch.int32, device="cuda")
    sr_e5_hw = torch.empty(TOTAL, dtype=torch.int32, device="cuda")
    sr_e5_uni = torch.empty(TOTAL, dtype=torch.int32, device="cuda")
    deq_e4 = torch.empty(TOTAL * 4, dtype=torch.float32, device="cuda")
    deq_e5 = torch.empty(TOTAL * 4, dtype=torch.float32, device="cuda")

    launch_sr(vals_dev, sr_e4_hw, sr_e4_uni, sr_e5_hw, sr_e5_uni, deq_e4, deq_e5)
    torch.cuda.synchronize()

    # Exact bit-for-bit parity between v_cvt_sr_{fp8,bf8}_f32 and universal across all 4096 lanes!
    assert torch.equal(sr_e4_hw.cpu(), sr_e4_uni.cpu())
    assert torch.equal(sr_e5_hw.cpu(), sr_e5_uni.cpu())

    # Empirical mean of 4096 stochastic draws must match 1.03125 and 1.0625 unbiasedly
    # (deterministic RNE would collapse 100% of samples to 1.0!).
    mean_e4 = deq_e4.cpu().mean().item()
    mean_e5 = deq_e5.cpu().mean().item()
    assert abs(mean_e4 - 1.03125) < 0.006, f"stochastic E4M3 mean {mean_e4} != 1.03125"
    assert abs(mean_e5 - 1.0625) < 0.012, f"stochastic E5M2 mean {mean_e5} != 1.0625"


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_e2m1_subbyte_bridges_device_parity():
    BLOCK = 64

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def e2m1_kernel(
        PackedWords: fx.Tensor,
        F32Hw: fx.Tensor,
        F32Uni: fx.Tensor,
        BfHw: fx.Tensor,
        BfUni: fx.Tensor,
        Fp8Hw: fx.Tensor,
        Fp8Uni: fx.Tensor,
        PqHw: fx.Tensor,
        PqUni: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        w = PackedWords[tid].bitcast(fx.Uint32)

        f_hw = fx.quant.unpack_e2m1x8(w, scale=1.5)
        f_uni = fx.quant.universal.unpack_e2m1x8(w, scale=1.5)
        for i in fx.range_constexpr(8):
            F32Hw[tid * 8 + i] = f_hw[i]
            F32Uni[tid * 8 + i] = f_uni[i]

        bf0_hw, bf1_hw = fx.quant.unpack_e2m1x4_bf16x2(w, half=0)
        bf2_hw, bf3_hw = fx.quant.unpack_e2m1x4_bf16x2(w, half=1)
        bf0_u, bf1_u = fx.quant.universal.unpack_e2m1x4_bf16x2(w, half=0)
        bf2_u, bf3_u = fx.quant.universal.unpack_e2m1x4_bf16x2(w, half=1)
        BfHw[tid * 4 + 0] = bf0_hw.bitcast(fx.Int32)
        BfHw[tid * 4 + 1] = bf1_hw.bitcast(fx.Int32)
        BfHw[tid * 4 + 2] = bf2_hw.bitcast(fx.Int32)
        BfHw[tid * 4 + 3] = bf3_hw.bitcast(fx.Int32)
        BfUni[tid * 4 + 0] = bf0_u.bitcast(fx.Int32)
        BfUni[tid * 4 + 1] = bf1_u.bitcast(fx.Int32)
        BfUni[tid * 4 + 2] = bf2_u.bitcast(fx.Int32)
        BfUni[tid * 4 + 3] = bf3_u.bitcast(fx.Int32)

        Fp8Hw[tid * 2 + 0] = fx.quant.unpack_e2m1x4_e4m3x4(w, half=0).bitcast(fx.Int32)
        Fp8Hw[tid * 2 + 1] = fx.quant.unpack_e2m1x4_e4m3x4(w, half=1).bitcast(fx.Int32)
        Fp8Uni[tid * 2 + 0] = fx.quant.universal.unpack_e2m1x4_e4m3x4(w, half=0).bitcast(fx.Int32)
        Fp8Uni[tid * 2 + 1] = fx.quant.universal.unpack_e2m1x4_e4m3x4(w, half=1).bitcast(fx.Int32)

        p_hw, q_hw = fx.quant.unpack_e2m1x8_int4_pair(w)
        p_u, q_u = fx.quant.universal.unpack_e2m1x8_int4_pair(w)
        PqHw[tid * 2 + 0], PqHw[tid * 2 + 1] = p_hw.bitcast(fx.Int32), q_hw.bitcast(fx.Int32)
        PqUni[tid * 2 + 0], PqUni[tid * 2 + 1] = p_u.bitcast(fx.Int32), q_u.bitcast(fx.Int32)

    @flyc.jit
    def launch_e2m1(
        PackedWords: fx.Tensor,
        F32Hw: fx.Tensor,
        F32Uni: fx.Tensor,
        BfHw: fx.Tensor,
        BfUni: fx.Tensor,
        Fp8Hw: fx.Tensor,
        Fp8Uni: fx.Tensor,
        PqHw: fx.Tensor,
        PqUni: fx.Tensor,
    ):
        e2m1_kernel(PackedWords, F32Hw, F32Uni, BfHw, BfUni, Fp8Hw, Fp8Uni, PqHw, PqUni).launch(
            grid=(1, 1, 1),
            block=(BLOCK, 1, 1),
        )

    rng = np.random.default_rng(99)
    words = rng.integers(0, 2**32, size=BLOCK, dtype=np.uint32)
    words[0] = np.uint32(0x76543210)
    words[1] = np.uint32(0xFEDCBA98)
    words_dev = torch.from_numpy(words.view(np.int32).copy()).to("cuda")

    f32_hw = torch.empty(BLOCK * 8, dtype=torch.float32, device="cuda")
    f32_uni = torch.empty(BLOCK * 8, dtype=torch.float32, device="cuda")
    bf_hw = torch.empty(BLOCK * 4, dtype=torch.int32, device="cuda")
    bf_uni = torch.empty(BLOCK * 4, dtype=torch.int32, device="cuda")
    fp8_hw = torch.empty(BLOCK * 2, dtype=torch.int32, device="cuda")
    fp8_uni = torch.empty(BLOCK * 2, dtype=torch.int32, device="cuda")
    pq_hw = torch.empty(BLOCK * 2, dtype=torch.int32, device="cuda")
    pq_uni = torch.empty(BLOCK * 2, dtype=torch.int32, device="cuda")

    launch_e2m1(words_dev, f32_hw, f32_uni, bf_hw, bf_uni, fp8_hw, fp8_uni, pq_hw, pq_uni)
    torch.cuda.synchronize()

    # Exact bit-level parity across all 4 hardware E2M1 bridges (including signed -0.0)!
    assert torch.equal(f32_hw.cpu().view(torch.int32), f32_uni.cpu().view(torch.int32))
    assert torch.equal(bf_hw.cpu(), bf_uni.cpu())
    assert torch.equal(fp8_hw.cpu(), fp8_uni.cpu())
    assert torch.equal(pq_hw.cpu(), pq_uni.cpu())


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_e2m1_pack_rne_and_int4_pair_sum_identity_device():
    @flyc.kernel(known_block_size=[4, 1, 1])
    def rne_kernel(In: fx.Tensor, PackedOut: fx.Tensor, Sum2WOut: fx.Tensor):
        tid = fx.thread_idx.x
        vals = [In[tid * 8 + i] for i in fx.range_constexpr(8)]
        pk = fx.quant.pack_e2m1x8(vals)
        PackedOut[tid] = pk.bitcast(fx.Int32)
        p_w, q_w = fx.quant.unpack_e2m1x8_int4_pair(pk)
        p_i = fx.quant.unpack_int4x8(p_w)
        q_i = fx.quant.unpack_int4x8(q_w)
        for i in fx.range_constexpr(8):
            Sum2WOut[tid * 8 + i] = p_i[i] + q_i[i]

    @flyc.jit
    def launch_rne(In: fx.Tensor, PackedOut: fx.Tensor, Sum2WOut: fx.Tensor):
        rne_kernel(In, PackedOut, Sum2WOut).launch(grid=(1, 1, 1), block=(4, 1, 1))

    # Row 0..1: exact E2M1 table; Row 2: positive RNE midpoints + overflow; Row 3: negative RNE midpoints + overflow.
    # At exact midpoints {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}, IEEE RNE rounds to even codes {0, 2, 2, 4, 4, 6, 6}.
    row2 = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 100.0]
    row3 = [-0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, -float("inf")]
    inp = torch.tensor(list(_E2M1_TABLE) + row2 + row3, dtype=torch.float32, device="cuda")
    packed_out = torch.empty(4, dtype=torch.int32, device="cuda")
    sum2w_out = torch.empty(32, dtype=torch.int32, device="cuda")

    launch_rne(inp, packed_out, sum2w_out)
    torch.cuda.synchronize()

    words_u32 = [w & 0xFFFFFFFF for w in packed_out.cpu().tolist()]
    assert words_u32[0] == 0x76543210
    assert words_u32[1] == 0xFEDCBA98
    assert words_u32[2] == 0x76644220
    assert words_u32[3] == 0xFEECCAA8

    expected_2w = [int(round(2.0 * v)) for v in _E2M1_TABLE] + [0, 2, 2, 4, 4, 8, 8, 12, 0, -2, -2, -4, -4, -8, -8, -12]
    assert sum2w_out.cpu().tolist() == expected_2w
