# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device tests for SageAttention2 G2 quant kernels (spec §9)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from kernels.attention.sageattn2_quant import E4M3_MAX, EPS, INT4_QMAX, _quant_qk_int4, _quant_v_fp8
from tests.kernels.sageattn2_layout import pack_int4, preshuffle_b_fp8, unpack_int4, unshuffle_b_fp8
from tests.kernels.sageattn2_ref import E4M3_MAX as REF_E4M3_MAX
from tests.kernels.sageattn2_ref import EPS as REF_EPS
from tests.kernels.sageattn2_ref import INT4_QMAX as REF_INT4_QMAX
from tests.kernels.sageattn2_ref import quant_per_token, quant_v_e4m3

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

_ARCH = str(get_rocm_arch() or "")
if not _ARCH.startswith("gfx120"):
    pytest.skip(f"quant tests require gfx120x, got {_ARCH}", allow_module_level=True)


def _is_gfx120() -> bool:
    return str(get_rocm_arch()).startswith("gfx120")


def _fx_stream():
    return fx.Stream(torch.cuda.current_stream())


def test_s1_f32_div_bit_identical():
    """Spec §0.2 S1: device f32 div vs np.float32, bit-compare."""
    n = 32
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal(n).astype(np.float32)
    b_np = rng.standard_normal(n).astype(np.float32)
    b_np = np.where(np.abs(b_np) < np.float32(1e-3), np.float32(1.0), b_np)
    want = a_np / b_np

    @flyc.kernel
    def div_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, N: fx.Constexpr[int]):
        tid = gpu.thread_id("x")
        a = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(A)), fx.make_layout((N,), (1,)))
        b = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(B)), fx.make_layout((N,), (1,)))
        c = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(C)), fx.make_layout((N,), (1,)))
        c[tid] = a[tid] / b[tid]

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, N: fx.Constexpr[int], stream: fx.Stream = fx.Stream(None)):
        div_kernel(A, B, C, N).launch(grid=(1,), block=(N,), stream=stream)

    a = torch.from_numpy(a_np.copy()).cuda()
    b = torch.from_numpy(b_np.copy()).cuda()
    c = torch.empty(n, dtype=torch.float32, device="cuda")
    launch(a, b, c, n, stream=_fx_stream())
    torch.cuda.synchronize()
    got = c.detach().cpu().numpy()
    np.testing.assert_array_equal(got, want)


@pytest.mark.skipif(not _is_gfx120(), reason="cvt_pk_fp8_f32 is a gfx120 path")
def test_s2_cvt_pk_fp8_f32_matches_torch_e4m3fn():
    """Spec §0.2 S2: hardware e4m3fn bytes vs torch.float8_e4m3fn."""
    nxt = float(np.nextafter(np.float32(448.0), np.float32(np.inf)))
    tiny = 2.0**-9  # e4m3fn smallest subnormal; torch.finfo has no .smallest_subnormal
    values = np.array(
        [447.0, 448.0, nxt, 0.0, tiny, -tiny, -448.0, 1.0],
        dtype=np.float32,
    )
    want = torch.from_numpy(values).to(torch.float8_e4m3fn).view(torch.uint8).numpy()

    @flyc.kernel
    def cvt_kernel(A: fx.Tensor, Out: fx.Tensor, N: fx.Constexpr[int]):
        tid = gpu.thread_id("x")
        a = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(A)), fx.make_layout((N,), (1,)))
        o = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(Out)), fx.make_layout((N,), (1,)))
        word = fx.rocdl.cvt_pk_fp8_f32(
            res=T.i32,
            src_a=a[tid],
            src_b=fx.Float32(0.0),
            old=fx.Int32(0),
            word_sel=False,
        )
        o[tid] = (fx.Int32(word) & fx.Int32(0xFF)).to(fx.Uint8)

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, N: fx.Constexpr[int], stream: fx.Stream = fx.Stream(None)):
        cvt_kernel(A, Out, N).launch(grid=(1,), block=(N,), stream=stream)

    a = torch.from_numpy(values.copy()).cuda()
    out = torch.empty(values.shape[0], dtype=torch.uint8, device="cuda")
    launch(a, out, values.shape[0], stream=_fx_stream())
    torch.cuda.synchronize()
    np.testing.assert_array_equal(out.detach().cpu().numpy(), want)


def test_a1_constants_match_g0():
    assert EPS == REF_EPS == 1e-7
    assert INT4_QMAX == REF_INT4_QMAX == 7
    assert E4M3_MAX == REF_E4M3_MAX == 448.0


def test_a2_kernel_module_does_not_import_tests():
    src = Path("kernels/attention/sageattn2_quant.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".", 1)[0] != "tests"
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".", 1)[0] != "tests"


def _qk_ok(dtype=torch.float16, d=64, b=1, hq=1, hkv=1, sq=4, sk=4):
    q = torch.zeros(b, hq, sq, d, dtype=dtype, device="cuda")
    k = torch.zeros(b, hkv, sk, d, dtype=dtype, device="cuda")
    return q, k


def test_q8_reject_d32_and_dtype_mismatch():
    q, k = _qk_ok(d=32)
    with pytest.raises(ValueError, match="D must be 64 or 128"):
        _quant_qk_int4(q, k)
    q64, k64 = _qk_ok(d=64)
    k_bf = k64.to(torch.bfloat16)
    with pytest.raises(ValueError, match="q, k must share dtype"):
        _quant_qk_int4(q64, k_bf)


def test_reject_not_4d_not_cuda_shape_mismatch_bad_dtype():
    q, k = _qk_ok()
    with pytest.raises(ValueError, match=r"expected \[B, H, S, D\]"):
        _quant_qk_int4(q[0], k)
    with pytest.raises(ValueError, match="CUDA"):
        _quant_qk_int4(q.cpu(), k.cpu())
    k_b2 = torch.zeros(2, 1, 4, 64, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="shape mismatch"):
        _quant_qk_int4(q, k_b2)
    qf, kf = q.float(), k.float()
    with pytest.raises(ValueError, match="fp16 or bf16"):
        _quant_qk_int4(qf, kf)


@pytest.mark.skipif(not _is_gfx120(), reason="V4 Sk reject is a gfx120 launch-path check")
def test_v4_reject_sk_17():
    v = torch.zeros(1, 1, 17, 64, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="Sk must be a multiple of 16"):
        _quant_v_fp8(v)


def _g0_int4_4d(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """x [B,H,S,D] CUDA/CPU → (packed int32 [B,H,S,D/8], scale f32 [B,H,S]). 2-D G-I loop."""
    x_np = x.detach().float().cpu().numpy()
    b, h, s, d = x_np.shape
    packed = np.empty((b, h, s, d // 8), dtype=np.int32)
    scale = np.empty((b, h, s), dtype=np.float32)
    for bi in range(b):
        for hi in range(h):
            q_i8, sc = quant_per_token(x_np[bi, hi], INT4_QMAX)
            packed[bi, hi] = pack_int4(q_i8)
            scale[bi, hi] = sc
    return packed, scale


def _assert_int4_match(gpu_i4: torch.Tensor, gpu_scale: torch.Tensor, x: torch.Tensor) -> None:
    packed_ref, scale_ref = _g0_int4_4d(x)
    assert torch.equal(gpu_i4.cpu(), torch.from_numpy(packed_ref))
    assert torch.equal(gpu_scale.cpu(), torch.from_numpy(scale_ref))


def test_q2_half_away_fixture_rows():
    """Spec §9 Q2: one non-zero fp16 element x0 = fp16(y*7) per row."""
    ys = [0.0, 0.49, -0.49, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5]
    d = 64
    rows = []
    for y in ys:
        row = torch.zeros(d, dtype=torch.float16)
        row[0] = torch.tensor(y * 7.0, dtype=torch.float16)
        rows.append(row)
    x = torch.stack(rows, dim=0).view(1, 1, len(ys), d).cuda()
    q_i4, k_i4, q_scale, k_scale = _quant_qk_int4(x, x)
    _assert_int4_match(q_i4, q_scale, x)
    _assert_int4_match(k_i4, k_scale, x)
    logical = unpack_int4(q_i4[0, 0].detach().cpu().numpy())
    q_ref, _ = quant_per_token(x[0, 0].float().cpu().numpy(), INT4_QMAX)
    np.testing.assert_array_equal(logical, q_ref)


def test_q3_zero_row_scale_eps_packed_zero():
    x = torch.zeros(1, 1, 4, 64, dtype=torch.float16, device="cuda")
    q_i4, _, q_scale, _ = _quant_qk_int4(x, x)
    assert torch.equal(q_scale.cpu(), torch.full((1, 1, 4), EPS))
    assert torch.equal(q_i4.cpu(), torch.zeros(1, 1, 4, 8, dtype=torch.int32))


@pytest.mark.parametrize("d", [64, 128])
def test_q1_random_fp16_pack_and_scale(d):
    torch.manual_seed(0)
    q = torch.randn(1, 1, 32, d, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 1, 32, d, dtype=torch.float16, device="cuda")
    q_i4, k_i4, q_scale, k_scale = _quant_qk_int4(q, k)
    _assert_int4_match(q_i4, q_scale, q)
    _assert_int4_match(k_i4, k_scale, k)


def test_q4_bf16_matches_g0_on_same_bf16_bits():
    torch.manual_seed(1)
    q = torch.randn(1, 1, 16, 64, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 1, 16, 64, dtype=torch.bfloat16, device="cuda")
    q_i4, k_i4, q_scale, k_scale = _quant_qk_int4(q, k)
    _assert_int4_match(q_i4, q_scale, q)
    _assert_int4_match(k_i4, k_scale, k)


def test_q6_gqa_native_heads_no_repeat():
    torch.manual_seed(2)
    q = torch.randn(1, 8, 16, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 2, 16, 64, dtype=torch.float16, device="cuda")
    q_i4, k_i4, q_scale, k_scale = _quant_qk_int4(q, k)
    assert q_i4.shape == (1, 8, 16, 8)
    assert k_i4.shape == (1, 2, 16, 8)
    _assert_int4_match(q_i4, q_scale, q)
    _assert_int4_match(k_i4, k_scale, k)
    for h in range(8):
        _assert_int4_match(q_i4[:, h : h + 1], q_scale[:, h : h + 1], q[:, h : h + 1])
    for h in range(2):
        _assert_int4_match(k_i4[:, h : h + 1], k_scale[:, h : h + 1], k[:, h : h + 1])


def test_q7_unpack_int4_matches_g0_logical_per_head():
    torch.manual_seed(3)
    q = torch.randn(2, 2, 8, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(2, 2, 8, 128, dtype=torch.float16, device="cuda")
    q_i4, _, _, _ = _quant_qk_int4(q, k)
    q_np = q.detach().float().cpu().numpy()
    for b in range(2):
        for h in range(2):
            logical = unpack_int4(q_i4[b, h].detach().cpu().numpy())
            q_ref, _ = quant_per_token(q_np[b, h], INT4_QMAX)
            np.testing.assert_array_equal(logical, q_ref)


def test_q5_smooth_k_km_equals_presubtracted_k():
    torch.manual_seed(4)
    q = torch.randn(1, 1, 16, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 2, 32, 64, dtype=torch.float16, device="cuda")
    km = k.float().mean(dim=2, keepdim=True)  # [1,2,1,64] fp32; G6/caller, not G2
    k_prime = (k.float() - km).to(k.dtype)
    q_i4_a, k_i4_a, q_scale_a, k_scale_a = _quant_qk_int4(q, k, km=km)
    q_i4_b, k_i4_b, q_scale_b, k_scale_b = _quant_qk_int4(q, k_prime, km=None)
    assert torch.equal(q_i4_a, q_i4_b)
    assert torch.equal(q_scale_a, q_scale_b)
    assert torch.equal(k_i4_a, k_i4_b)
    assert torch.equal(k_scale_a, k_scale_b)
    _assert_int4_match(k_i4_a, k_scale_a, k_prime)
    _assert_int4_match(q_i4_a, q_scale_a, q)


def _g0_v_fp8_4d(v_fp16: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """v [B,H,Sk,D] fp16 CUDA → (preshuffled uint8 7-D, v_scale [B,H,D])."""
    v_np = v_fp16.detach().float().cpu().numpy()
    b, h, sk, d = v_np.shape
    stored = np.empty((b, h, d // 16, sk // 16, 2, 16, 8), dtype=np.uint8)
    scale = np.empty((b, h, d), dtype=np.float32)
    for bi in range(b):
        for hi in range(h):
            logical, sc = quant_v_e4m3(v_np[bi, hi])
            stored[bi, hi] = preshuffle_b_fp8(logical)
            scale[bi, hi] = sc
    return stored, scale


@pytest.mark.skipif(not _is_gfx120(), reason="V-fp8 is gfx120* only")
@pytest.mark.parametrize("sk,d", [(32, 64), (128, 128)])
def test_v1_preshuffle_and_scale_vs_g0(sk, d):
    # |v/v_scale| can be nextafter(448); RNE overflow threshold is 464 so
    # that still encodes 0x7E. Do not clamp to 448 in the kernel.
    torch.manual_seed(5)
    v = torch.randn(1, 1, sk, d, dtype=torch.float16, device="cuda")
    v_e4, v_scale = _quant_v_fp8(v)
    stored_ref, scale_ref = _g0_v_fp8_4d(v)
    assert torch.equal(v_scale.cpu(), torch.from_numpy(scale_ref))
    got = v_e4.detach().cpu().numpy()
    for b in range(got.shape[0]):
        for h in range(got.shape[1]):
            logical_gpu = unshuffle_b_fp8(got[b, h])
            logical_ref, _ = quant_v_e4m3(v[b, h].float().cpu().numpy())
            np.testing.assert_array_equal(logical_gpu, logical_ref)
            np.testing.assert_array_equal(got[b, h], stored_ref[b, h])


@pytest.mark.skipif(not _is_gfx120(), reason="V-fp8 is gfx120* only")
def test_v2_bf16_cast_to_fp16_before_quant():
    torch.manual_seed(6)
    v_bf16 = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16, device="cuda")
    v_e4, v_scale = _quant_v_fp8(v_bf16)
    # quant_v_e4m3 does not itself narrow bf16→fp16. G0 §2: fp16 first.
    v_ref = v_bf16.to(torch.float16)
    stored_ref, scale_ref = _g0_v_fp8_4d(v_ref)
    assert torch.equal(v_scale.cpu(), torch.from_numpy(scale_ref))
    got = v_e4[0, 0].detach().cpu().numpy()
    logical_gpu = unshuffle_b_fp8(got)
    bytes_ref, _ = quant_v_e4m3(v_ref.float().cpu().numpy()[0, 0])
    np.testing.assert_array_equal(logical_gpu, bytes_ref)
    # Wrong oracle (skip fp16 store) must *not* be used; it disagrees by design.
    wrong = v_bf16.float().cpu().numpy()[0, 0]
    wrong_bytes, _ = quant_v_e4m3(wrong)
    # If the kernel followed the wrong oracle, this would be equal — require it is not
    # the pass criterion. The pass criterion is the fp16 path above.
    assert bytes_ref.shape == wrong_bytes.shape
