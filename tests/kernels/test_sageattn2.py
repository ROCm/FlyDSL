# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device tests for the SageAttention2 host API (G6 spec §7)."""

from __future__ import annotations

import zlib

import numpy as np
import pytest
import torch

import kernels.attention.sageattn2 as sageattn2_module
from flydsl.runtime.device import get_rocm_arch
from kernels.attention.sageattn2 import (
    _head_group_size,
    _kv_bytes_per_head,
    _normalize_layout,
    _pad_kv,
    _require_arch,
    _validate_inputs,
    _validate_shapes,
    sageattn2,
)
from kernels.attention.sageattn2_rdna4 import CTA_Q
from tests.kernels.sageattn2_ref import sageattn2_ref

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

_ARCH = str(get_rocm_arch() or "")
if not _ARCH.startswith("gfx120"):
    pytest.skip(f"G6 requires gfx120x, got {_ARCH}", allow_module_level=True)


def test_require_arch_accepts_gfx120x():
    _require_arch("gfx1200")
    _require_arch("gfx1201")


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1151", "gfx1250", "gfx942", "gfx950", ""])
def test_require_arch_rejects_other(arch):
    with pytest.raises(RuntimeError, match="gfx120x"):
        _require_arch(arch)


def test_require_arch_accepts_the_live_arch():
    _require_arch(str(get_rocm_arch() or ""))


def _qkv(B, Hq, Hkv, Sq, Sk, D, dtype, seed):
    """Random HND q, k, v on the current device."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(B, Hq, Sq, D, generator=g, device="cuda") * 0.5).to(dtype)
    k = (torch.randn(B, Hkv, Sk, D, generator=g, device="cuda") * 0.5).to(dtype)
    v = (torch.randn(B, Hkv, Sk, D, generator=g, device="cuda") * 0.5).to(dtype)
    return q, k, v


def test_normalize_layout_hnd_is_a_passthrough():
    q, _, _ = _qkv(2, 4, 4, 16, 16, 64, torch.float16, seed=1)
    out = _normalize_layout(q, "HND")
    assert out.shape == (2, 4, 16, 64)
    assert out.data_ptr() == q.data_ptr()


def test_normalize_layout_nhd_transposes_and_compacts():
    q, _, _ = _qkv(2, 4, 4, 16, 16, 64, torch.float16, seed=2)
    q_nhd = q.transpose(1, 2).contiguous()
    out = _normalize_layout(q_nhd, "NHD")
    assert out.shape == (2, 4, 16, 64)
    assert out.is_contiguous()
    assert torch.equal(out, q)


def test_validate_inputs_rejects_bad_layout():
    q, k, v = _qkv(1, 1, 1, 16, 16, 64, torch.float16, seed=3)
    with pytest.raises(ValueError, match="tensor_layout"):
        _validate_inputs(q, k, v, "BHSD", None, None)


def test_validate_inputs_rejects_f32():
    q, k, v = _qkv(1, 1, 1, 16, 16, 64, torch.float16, seed=4)
    with pytest.raises(ValueError, match="fp16 or bf16"):
        _validate_inputs(q.float(), k.float(), v.float(), "HND", None, None)


def test_validate_inputs_rejects_mixed_dtype():
    q, k, v = _qkv(1, 1, 1, 16, 16, 64, torch.float16, seed=5)
    with pytest.raises(ValueError, match="share dtype"):
        _validate_inputs(q, k.bfloat16(), v, "HND", None, None)


def test_validate_inputs_rejects_cpu_and_rank():
    q, k, v = _qkv(1, 1, 1, 16, 16, 64, torch.float16, seed=6)
    with pytest.raises(ValueError, match="CUDA"):
        _validate_inputs(q.cpu(), k, v, "HND", None, None)
    with pytest.raises(ValueError, match="rank 4"):
        _validate_inputs(q[0], k, v, "HND", None, None)


def test_validate_inputs_rejects_bad_out_dtype_and_sm_scale():
    q, k, v = _qkv(1, 1, 1, 16, 16, 64, torch.float16, seed=7)
    with pytest.raises(ValueError, match="out_dtype"):
        _validate_inputs(q, k, v, "HND", torch.float32, None)
    with pytest.raises(ValueError, match="finite"):
        _validate_inputs(q, k, v, "HND", None, float("nan"))


def test_validate_shapes_rejects_bad_head_dim_and_gqa():
    q, k, v = _qkv(1, 1, 1, 16, 16, 32, torch.float16, seed=8)
    with pytest.raises(ValueError, match="head_dim"):
        _validate_shapes(q, k, v)
    q, k, v = _qkv(1, 3, 2, 16, 16, 64, torch.float16, seed=9)
    with pytest.raises(ValueError, match="multiple of Hkv"):
        _validate_shapes(q, k, v)


def test_validate_shapes_rejects_kv_mismatch():
    q, k, v = _qkv(1, 2, 1, 16, 16, 64, torch.float16, seed=10)
    with pytest.raises(ValueError, match="must equal k shape"):
        _validate_shapes(q, k, v[:, :, :8])


def test_pad_kv_is_a_noop_when_already_aligned():
    _, k, v = _qkv(1, 1, 1, 16, 64, 64, torch.float16, seed=20)
    k_pad, v_pad, skp = _pad_kv(k, v)
    assert skp == 64
    assert k_pad.data_ptr() == k.data_ptr()
    assert v_pad.data_ptr() == v.data_ptr()


def test_pad_kv_rounds_up_to_16_with_zeros():
    _, k, v = _qkv(1, 2, 2, 16, 100, 64, torch.float16, seed=21)
    k_pad, v_pad, skp = _pad_kv(k, v)
    assert skp == 112
    assert k_pad.shape == (1, 2, 112, 64)
    assert v_pad.shape == (1, 2, 112, 64)
    assert torch.equal(k_pad[:, :, :100], k)
    assert torch.equal(v_pad[:, :, :100], v)
    assert torch.count_nonzero(k_pad[:, :, 100:]) == 0
    assert torch.count_nonzero(v_pad[:, :, 100:]) == 0


def test_zero_padding_k_does_not_change_quantization():
    """Spec §5.4: per-token scales, and an EPS-guarded scale on the zero rows."""
    from kernels.attention.sageattn2_quant import _quant_qk_int4

    q, k, _ = _qkv(1, 1, 1, 16, 64, 64, torch.float16, seed=22)
    k_pad = torch.nn.functional.pad(k, (0, 0, 0, 16))
    _, k_i4_base, _, k_scale_base = _quant_qk_int4(q, k)
    _, k_i4_pad, _, k_scale_pad = _quant_qk_int4(q, k_pad)
    torch.cuda.synchronize()
    assert torch.equal(k_i4_base, k_i4_pad[:, :, :64])
    assert torch.equal(k_scale_base, k_scale_pad[:, :, :64])
    assert torch.count_nonzero(k_i4_pad[:, :, 64:]) == 0
    # A zero scale here would be a divide-by-zero in the kernel; the EPS at
    # sageattn2_quant.py:126 is what prevents it.
    assert torch.all(k_scale_pad[:, :, 64:] > 0)


def test_zero_padding_v_does_not_change_quantization():
    """Spec §5.4: appending zeros cannot raise a per-channel amax."""
    from kernels.attention.sageattn2_quant import _quant_v_fp8

    _, _, v = _qkv(1, 2, 2, 16, 64, 64, torch.float16, seed=23)
    v_pad = torch.nn.functional.pad(v, (0, 0, 0, 16))
    e4_base, scale_base = _quant_v_fp8(v)
    e4_pad, scale_pad = _quant_v_fp8(v_pad)
    torch.cuda.synchronize()
    assert torch.equal(scale_base, scale_pad)
    # V_e4m3 is [B, Hkv, D/16, Sk/16, 2, 16, 8]; axis 3 grows by one block and
    # the existing blocks must be byte-identical.
    k0 = e4_base.shape[3]
    assert e4_pad.shape[3] == k0 + 1
    assert torch.equal(e4_base, e4_pad[:, :, :, :k0])
    assert torch.count_nonzero(e4_pad[:, :, :, k0:]) == 0


def _assert_close(device_out, ref_out, tag):
    """Cosine floor is 0.999. rel_l2 allows the block-rescale gap on these
    short sequences: one 64-key stage quantises P to FP8 against a single max,
    about 1-3% versus the 16-key reference. Long sequences fall under 0.5%."""
    d, r = device_out.ravel(), ref_out.ravel()
    cos = float(np.dot(d, r) / (np.linalg.norm(d) * np.linalg.norm(r) + 1e-12))
    rel_l2 = float(np.linalg.norm(d - r) / (np.linalg.norm(r) + 1e-12))
    print(f"{tag}: cosine={cos:.6f} rel_l2={rel_l2:.3e}")
    assert cos >= 0.999, f"{tag}: cosine {cos}"
    assert rel_l2 <= 3e-2, f"{tag}: rel_l2 {rel_l2}"


@pytest.mark.parametrize(
    "tag,B,Hq,Hkv,Sq,Sk,D,is_causal,smooth_k,dtype,sm_scale",
    [
        ("T1a_baseline", 1, 1, 1, 64, 64, 64, False, True, torch.float16, None),
        ("T1b_causal", 1, 1, 1, 64, 64, 64, True, True, torch.float16, None),
        ("T1c_causal_sq_ne_sk", 1, 1, 1, 48, 64, 64, True, True, torch.float16, None),
        ("T1d_gqa", 1, 8, 2, 64, 64, 64, False, True, torch.float16, None),
        ("T1e_d128", 1, 1, 1, 64, 64, 128, False, True, torch.float16, None),
        ("T1f_bf16", 1, 1, 1, 64, 64, 64, False, True, torch.bfloat16, None),
        ("T1g_sk_pad", 1, 1, 1, 64, 100, 64, False, True, torch.float16, None),
        ("T1h_no_smooth_k", 1, 1, 1, 64, 64, 64, False, False, torch.float16, None),
        ("T1i_multi_batch", 2, 2, 1, 64, 64, 64, False, True, torch.float16, None),
        ("T1j_sm_scale", 1, 1, 1, 64, 64, 64, False, True, torch.float16, 0.05),
    ],
)
def test_matrix_matches_reference(tag, B, Hq, Hkv, Sq, Sk, D, is_causal, smooth_k, dtype, sm_scale):
    q, k, v = _qkv(B, Hq, Hkv, Sq, Sk, D, dtype, seed=zlib.crc32(tag.encode()) % 10000)
    out = sageattn2(q, k, v, is_causal=is_causal, smooth_k=smooth_k, sm_scale=sm_scale)
    torch.cuda.synchronize()
    ref = sageattn2_ref(q, k, v, is_causal=is_causal, sm_scale=sm_scale, smooth_k=smooth_k, qk="int4", pv="e4m3")
    _assert_close(out.float().cpu().numpy(), ref.float().cpu().numpy(), tag)


def test_t2_nhd_matches_hnd_bitwise():
    q, k, v = _qkv(1, 4, 2, 64, 64, 64, torch.float16, seed=200)
    out_hnd = sageattn2(q, k, v)
    out_nhd = sageattn2(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        tensor_layout="NHD",
    )
    torch.cuda.synchronize()
    assert out_nhd.shape == (1, 64, 4, 64)
    assert not out_nhd.is_contiguous()
    assert torch.equal(out_hnd, out_nhd.transpose(1, 2))


def test_t3_output_contract():
    q, k, v = _qkv(1, 2, 1, 48, 100, 64, torch.float16, seed=201)
    out = sageattn2(q, k, v)
    torch.cuda.synchronize()
    assert out.shape == (1, 2, 48, 64)  # Skp=112 must not leak into the result
    assert out.dtype == torch.float16
    assert out.device == q.device
    assert out.is_contiguous()

    out_bf16 = sageattn2(q, k, v, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    assert out_bf16.dtype == torch.bfloat16
    assert out_bf16.shape == out.shape

    qb, kb, vb = _qkv(1, 2, 1, 48, 64, 64, torch.bfloat16, seed=202)
    out_default = sageattn2(qb, kb, vb)
    torch.cuda.synchronize()
    assert out_default.dtype == torch.bfloat16


def test_t5_non_default_stream_matches():
    q, k, v = _qkv(1, 2, 1, 64, 64, 64, torch.float16, seed=203)
    out_default = sageattn2(q, k, v)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    out_side = sageattn2(q, k, v, stream=side)
    side.synchronize()
    torch.cuda.synchronize()
    assert torch.equal(out_default, out_side)


def test_t4_public_entry_point_rejects():
    """Every validator must actually be wired into sageattn2(), not just exist."""
    q, k, v = _qkv(1, 1, 1, 64, 64, 64, torch.float16, seed=204)

    with pytest.raises(ValueError, match="tensor_layout"):
        sageattn2(q, k, v, tensor_layout="BHSD")
    with pytest.raises(ValueError, match="fp16 or bf16"):
        sageattn2(q.float(), k.float(), v.float())
    with pytest.raises(ValueError, match="share dtype"):
        sageattn2(q, k.bfloat16(), v)
    with pytest.raises(ValueError, match="CUDA"):
        sageattn2(q.cpu(), k, v)
    with pytest.raises(ValueError, match="out_dtype"):
        sageattn2(q, k, v, out_dtype=torch.float32)
    with pytest.raises(ValueError, match="finite"):
        sageattn2(q, k, v, sm_scale=float("inf"))
    with pytest.raises(ValueError, match="must equal k shape"):
        sageattn2(q, k, v[:, :, :32].contiguous())

    q32, k32, v32 = _qkv(1, 1, 1, 64, 64, 32, torch.float16, seed=205)
    with pytest.raises(ValueError, match="head_dim"):
        sageattn2(q32, k32, v32)

    qg, kg, vg = _qkv(1, 3, 2, 64, 64, 64, torch.float16, seed=206)
    with pytest.raises(ValueError, match="multiple of Hkv"):
        sageattn2(qg, kg, vg)


@pytest.mark.benchmark
@pytest.mark.parametrize("is_causal", [False, True])
def test_t6_benchmark_4096_d128(is_causal):
    """Spec §7.6: recorded, not gated. Reports quant-inclusive and kernel-only."""
    from kernels.attention.sageattn2_quant import _quant_qk_int4, _quant_v_fp8
    from kernels.attention.sageattn2_rdna4 import _sageattn2_rdna4_prefill

    B, Hq, Hkv, S, D = 1, 8, 8, 4096, 128
    q, k, v = _qkv(B, Hq, Hkv, S, S, D, torch.float16, seed=300)

    km = k.float().mean(dim=2, keepdim=True)
    q_i4, k_i4, q_scale, k_scale = _quant_qk_int4(q, k, km=km)
    v_e4m3, v_scale = _quant_v_fp8(v)
    torch.cuda.synchronize()

    def _e2e():
        sageattn2(q, k, v, is_causal=is_causal)

    def _kernel_only():
        _sageattn2_rdna4_prefill(q_i4, q_scale, k_i4, k_scale, v_e4m3, v_scale, is_causal=is_causal)

    def _time_ms(fn, warmup=5, iters=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    ms_e2e = _time_ms(_e2e)
    ms_kernel = _time_ms(_kernel_only)
    # Same formula as G5's benchmark: QK + PV, 2 FMA-flops/MAC. Not halved for
    # causal, so the causal row counts masked work and is a lower bound.
    flops = 2 * S * S * D * 2 * Hq * B
    print(
        f"\nG6 causal={is_causal} Sq=Sk=4096 D=128 Hq=8: "
        f"e2e(quant+kernel)={ms_e2e:.4f} ms ({flops / (ms_e2e * 1e-3) / 1e12:.2f} TFLOP/s), "
        f"kernel_only={ms_kernel:.4f} ms ({flops / (ms_kernel * 1e-3) / 1e12:.2f} TFLOP/s) "
        f"[G5 recorded 0.7731 ms at KW=32]"
    )
    assert ms_e2e > 0.0 and ms_kernel > 0.0


_LLC_64_MIB = 64 * 1024 * 1024


def test_t8_kv_bytes_per_head_matches_hand_computed_bytes():
    """G8 spec §4.2 / Appendix C: exact byte counts for the three probed shapes."""
    assert _kv_bytes_per_head(skp=114672, d=128) == 22_476_224
    assert _kv_bytes_per_head(skp=32768, d=128) == 6_423_040
    assert _kv_bytes_per_head(skp=4096, d=128) == 803_328


def test_t8_head_group_size_picks_measured_optimum_large_shape():
    """G8 spec §5: pins the measured optimum (group=4, -8.21% isolated probe,
    -6.47% confirmed end-to-end) into the suite.

    ``grid_x`` is computed from the real ``cta_q(DEFAULT_WAVES)`` (G10 spec
    §6) rather than a hard-coded ``CTA_Q=64`` literal, so this stays honest
    if a future spec changes the default tile. ``multi_processor_count=32``
    is still a literal, not derived from a real ``sq`` here: this pinned test
    only tracks ``skp`` (the padded ``Sk``), so per final-review fix
    instructions we plug in the R9700's real CU count directly, generous
    enough that the occupancy guard (final review fix wave) does not fire
    and this keeps testing what it always tested. At today's default
    (``DEFAULT_WAVES=4`` -> ``CTA_Q=64``) this is
    ``grid_x = ceil(114660 / 64) = 1792``.
    """
    grid_x = -(-114660 // CTA_Q)
    g = _head_group_size(
        hq=28,
        hkv=28,
        skp=114672,
        d=128,
        b=1,
        is_causal=False,
        llc_bytes=_LLC_64_MIB,
        grid_x=grid_x,
        multi_processor_count=32,
    )
    assert g == 4


def test_t8_head_group_size_declines_medium_shape_below_enable_threshold():
    """G8 final-review fix wave: this shape's isolated-probe numbers (group=14,
    -9.0%, G8 spec §5) do not survive the real ``sageattn2()`` call path once
    quantization overhead is included -- the controller measured end-to-end
    +0.77%, not a win. ``total_kv`` here is 179.8 MB = 2.81x LLC, which clears
    the old ``_LLC_WORKING_SET_RATIO=1.35`` gate but not the new, higher
    ``_ENABLE_LLC_RATIO=5.0`` gate calibrated against the real call path, so
    the function now declines to group this shape at all. This replaces (not
    supplements) the old assertion that pinned ``group=14``, which is no
    longer the correct behavior.
    """
    g = _head_group_size(
        hq=28,
        hkv=28,
        skp=32768,
        d=128,
        b=1,
        is_causal=False,
        llc_bytes=_LLC_64_MIB,
        grid_x=512,
        multi_processor_count=32,
    )
    assert g == 28


def test_t8_head_group_size_declines_small_shape():
    """G8 spec §2.5, §5: total_kv (6.4 MB) is far below target; no grouping."""
    g = _head_group_size(
        hq=8,
        hkv=8,
        skp=4096,
        d=128,
        b=1,
        is_causal=False,
        llc_bytes=_LLC_64_MIB,
        grid_x=64,
        multi_processor_count=32,
    )
    assert g == 8


def test_t8_head_group_size_declines_low_occupancy_shape():
    """Final-review fix wave, occupancy guard: a long-Sk shape whose byte
    count alone clears both LLC gates (``total_kv`` = 411.0 MB = 6.12x LLC,
    above ``_ENABLE_LLC_RATIO=5.0``'s 335.5 MB gate) and whose sizing formula
    alone would pick ``group=2`` (``round(1.35 x 64MiB / 51,380,736)``) must
    still decline when the resulting launch would under-fill the GPU: at
    ``grid_x=16`` (a short-Sq shape, e.g. chunked prefill/cross-attention)
    ``grid_x * group_size = 32`` blocks, far under
    ``_MIN_BLOCKS_PER_LAUNCH_CUS * multi_processor_count = 128``. This is the
    exact shape class that regressed +18.35% end-to-end before this guard
    (controller's reproduction: B=1,Hq=Hkv=8,Sq=1024,Sk=131072,D=128; this
    unit test uses a larger ``skp`` so ``total_kv`` clears the Fix-2 enable
    gate and the occupancy guard -- not the enable gate -- is what fires).
    """
    g = _head_group_size(
        hq=8,
        hkv=8,
        skp=262144,
        d=128,
        b=1,
        is_causal=False,
        llc_bytes=_LLC_64_MIB,
        grid_x=16,
        multi_processor_count=32,
    )
    assert g == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(hq=28, hkv=28, skp=114672, d=128, b=1, is_causal=False, llc_bytes=0),
        dict(hq=28, hkv=28, skp=114672, d=128, b=2, is_causal=False, llc_bytes=_LLC_64_MIB),
        dict(hq=28, hkv=4, skp=114672, d=128, b=1, is_causal=False, llc_bytes=_LLC_64_MIB),
        dict(hq=28, hkv=28, skp=114672, d=128, b=1, is_causal=True, llc_bytes=_LLC_64_MIB),
        dict(hq=1, hkv=1, skp=1_048_576, d=128, b=1, is_causal=False, llc_bytes=_LLC_64_MIB),
    ],
    ids=["llc_unknown", "batch_gt_1", "gqa", "causal", "single_head"],
)
def test_t8_head_group_size_declines_the_excluded_envelope(kwargs):
    """G8 spec §2.2-§2.5, §4.1: every excluded condition returns hq (no
    grouping). ``grid_x=10000, multi_processor_count=32`` are generous on
    purpose (final review fix wave): these cases must keep declining for
    their ORIGINAL stated reason (GQA, B>1, causal, unknown LLC, single
    head), not accidentally start declining for occupancy instead.
    """
    assert _head_group_size(**kwargs, grid_x=10000, multi_processor_count=32) == kwargs["hq"]


@pytest.mark.parametrize(
    "hq,sq,d,group_size",
    [
        (4, 2048, 128, 1),
        (4, 2048, 128, 2),
        (8, 4096, 64, 1),
        (8, 4096, 64, 3),
        (28, 8192, 128, 1),
        (28, 8192, 128, 4),
    ],
)
def test_t7_grouping_is_bitwise_identical(monkeypatch, hq, sq, d, group_size):
    """G8 spec §7.1, §2.1: grouping changes launch count only, never the
    per-(b, h) arithmetic. Not a tolerance — torch.equal."""
    q, k, v = _qkv(1, hq, hq, sq, sq, d, torch.float16, seed=hq * 100_000 + sq + d + group_size)

    monkeypatch.setattr(sageattn2_module, "_FORCE_GROUP_SIZE", None)
    out_ungrouped = sageattn2(q, k, v)
    torch.cuda.synchronize()

    monkeypatch.setattr(sageattn2_module, "_FORCE_GROUP_SIZE", group_size)
    out_grouped = sageattn2(q, k, v)
    torch.cuda.synchronize()

    assert torch.equal(out_ungrouped, out_grouped)


def test_t7_ungrouped_path_is_a_direct_passthrough(monkeypatch):
    """G8 spec §4.3, acceptance item 8: disabling grouping must be exactly
    today's call — one launch, no copy_, output returned by identity."""
    real_prefill = sageattn2_module._sageattn2_rdna4_prefill
    calls = []

    def _spy(*args, **kwargs):
        out = real_prefill(*args, **kwargs)
        calls.append(out)
        return out

    monkeypatch.setattr(sageattn2_module, "_sageattn2_rdna4_prefill", _spy)
    monkeypatch.setattr(sageattn2_module, "_FORCE_GROUP_SIZE", None)

    q, k, v = _qkv(1, 4, 4, 256, 256, 64, torch.float16, seed=999)
    out = sageattn2(q, k, v)
    torch.cuda.synchronize()

    assert len(calls) == 1
    assert out.data_ptr() == calls[0].data_ptr()


def test_t7_grouped_path_makes_multiple_launches(monkeypatch):
    """Companion to the passthrough test: forcing group_size=2 over Hq=4
    must produce exactly 2 launches, each covering 2 heads."""
    real_prefill = sageattn2_module._sageattn2_rdna4_prefill
    shapes = []

    def _spy(q_i4, *args, **kwargs):
        shapes.append(q_i4.shape[1])
        return real_prefill(q_i4, *args, **kwargs)

    monkeypatch.setattr(sageattn2_module, "_sageattn2_rdna4_prefill", _spy)
    monkeypatch.setattr(sageattn2_module, "_FORCE_GROUP_SIZE", 2)

    q, k, v = _qkv(1, 4, 4, 256, 256, 64, torch.float16, seed=998)
    sageattn2(q, k, v)
    torch.cuda.synchronize()

    assert shapes == [2, 2]


def test_t7_real_policy_groups_and_stays_bitwise_identical(monkeypatch):
    """Final-review fix wave: nothing gated previously proved the REAL policy
    (no ``_FORCE_GROUP_SIZE`` override) actually chooses to group on some
    shape, as opposed to always testing a forced group size.

    Shape ``B=1, Hq=Hkv=56, Sq=768, Sk=32768, D=128``: ``kv_bytes_per_head``
    at ``skp=32768, d=128`` is 6,423,040 bytes (pinned by
    ``test_t8_kv_bytes_per_head_matches_hand_computed_bytes``), so
    ``total_kv = 56 * 6,423,040 = 359,690,240`` bytes = 5.36x the R9700's
    64 MiB LLC, clearing ``_ENABLE_LLC_RATIO=5.0``'s 335,544,320-byte gate.
    Sizing then picks ``group=14`` (``round(1.35 x 64MiB / 6,423,040)``), so
    ``grid_x = ceil(768 / cta_q(DEFAULT_WAVES)) = 12`` at today's default
    (``DEFAULT_WAVES=4`` -> ``CTA_Q=64``) gives
    ``grid_x * group_size = 168``, comfortably above the occupancy guard's
    ``_MIN_BLOCKS_PER_LAUNCH_CUS * multi_processor_count = 128``. A short
    ``Sq`` relative to ``Sk`` keeps this cheap for the gated suite (measured
    ~40 ms warm) while still a real, un-forced grouping decision.
    """
    real_prefill = sageattn2_module._sageattn2_rdna4_prefill
    shapes = []

    def _spy(q_i4, *args, **kwargs):
        shapes.append(q_i4.shape[1])
        return real_prefill(q_i4, *args, **kwargs)

    q, k, v = _qkv(1, 56, 56, 768, 32768, 128, torch.float16, seed=997)

    monkeypatch.setattr(sageattn2_module, "_sageattn2_rdna4_prefill", _spy)
    monkeypatch.setattr(sageattn2_module, "_FORCE_GROUP_SIZE", None)
    out_real = sageattn2(q, k, v)
    torch.cuda.synchronize()

    assert len(shapes) > 1

    monkeypatch.setattr(sageattn2_module, "_sageattn2_rdna4_prefill", real_prefill)
    monkeypatch.setattr(sageattn2_module, "_FORCE_GROUP_SIZE", 56)
    out_forced_ungrouped = sageattn2(q, k, v)
    torch.cuda.synchronize()

    assert torch.equal(out_real, out_forced_ungrouped)


@pytest.mark.benchmark
def test_t10_grouping_benchmark_medium_shape(monkeypatch):
    """G8 spec §7.4, §5: recorded, never gated -- this shape's own baseline
    moves ~1.8% run to run (Appendix C), the same order as the effect."""
    B, Hq, Hkv, S, D = 1, 28, 28, 32768, 128
    q, k, v = _qkv(B, Hq, Hkv, S, S, D, torch.float16, seed=400)

    def _time_ms(fn, warmup=1, iters=5):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    monkeypatch.setattr(sageattn2_module, "_FORCE_GROUP_SIZE", None)
    for _ in range(3):
        sageattn2(q, k, v)  # clock warmup, discarded
    torch.cuda.synchronize()

    ms_grouped = _time_ms(lambda: sageattn2(q, k, v))
    monkeypatch.setattr(sageattn2_module, "_FORCE_GROUP_SIZE", Hq)
    ms_ungrouped = _time_ms(lambda: sageattn2(q, k, v))
    pct = (ms_grouped - ms_ungrouped) / ms_ungrouped * 100.0
    print(
        f"\nG8 T10 [{B},{Hq},{S},{D}] grouped={ms_grouped:.3f} ms  " f"ungrouped={ms_ungrouped:.3f} ms  ({pct:+.2f}%)"
    )
    assert ms_grouped > 0.0 and ms_ungrouped > 0.0
