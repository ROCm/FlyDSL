# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device tests for the single SageAttention2 RDNA4 prefill path."""

from __future__ import annotations

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.attention.sageattn2_rdna4 import _use_key_peel, _sageattn2_rdna4_prefill


def test_use_key_peel_predicate():
    # Skp - sk < STAGE_K is unsound here: tail_base=64, so sk<64 stays masked.
    assert _use_key_peel(False, 49, 112) is False
    assert _use_key_peel(False, 17, 80) is False
    # The same shortcut would reject this shape; the tail predicate allows it.
    assert _use_key_peel(False, 64, 128) is True
    assert _use_key_peel(False, 128, 128) is True
    assert _use_key_peel(True, 128, 128) is False


pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

_ARCH = str(get_rocm_arch() or "")
if not _ARCH.startswith("gfx120"):
    pytest.skip(f"sageattn2 requires gfx120x, got {_ARCH}", allow_module_level=True)


def _valid(B=1, Hq=1, Hkv=1, Sq=16, Skp=16, D=64):
    q_i4 = torch.zeros(B, Hq, Sq, D // 8, dtype=torch.int32, device="cuda")
    q_scale = torch.full((B, Hq, Sq), 0.01, dtype=torch.float32, device="cuda")
    k_i4 = torch.zeros(B, Hkv, Skp, D // 8, dtype=torch.int32, device="cuda")
    k_scale = torch.full((B, Hkv, Skp), 0.01, dtype=torch.float32, device="cuda")
    v_e4m3 = torch.zeros(B, Hkv, D // 16, Skp // 16, 2, 16, 8, dtype=torch.uint8, device="cuda")
    v_scale = torch.full((B, Hkv, D), 0.01, dtype=torch.float32, device="cuda")
    return q_i4, q_scale, k_i4, k_scale, v_e4m3, v_scale


def test_rejects_bad_d():
    with pytest.raises(ValueError, match="D must be 64 or 128"):
        _sageattn2_rdna4_prefill(*_valid(D=96))


def test_rejects_bad_skp():
    q_i4, q_scale, k_i4, k_scale, v_e4m3, v_scale = _valid(Skp=20)
    with pytest.raises(ValueError, match="Skp must be a multiple of 16"):
        _sageattn2_rdna4_prefill(q_i4, q_scale, k_i4, k_scale, v_e4m3, v_scale)


def test_rejects_bad_gqa_ratio():
    q_i4, q_scale, k_i4, k_scale, v_e4m3, v_scale = _valid(Hq=3, Hkv=2)
    with pytest.raises(ValueError, match="multiple of Hkv"):
        _sageattn2_rdna4_prefill(q_i4, q_scale, k_i4, k_scale, v_e4m3, v_scale)


def test_rejects_bad_out_dtype():
    with pytest.raises(ValueError, match="out_dtype"):
        _sageattn2_rdna4_prefill(*_valid(), out_dtype=torch.float32)


def test_rejects_seq_len_k_too_large():
    with pytest.raises(ValueError, match="seq_len_k"):
        _sageattn2_rdna4_prefill(*_valid(), seq_len_k=32)


def test_rejects_nonfinite_scale():
    q_i4, q_scale, k_i4, k_scale, v_e4m3, v_scale = _valid()
    q_scale[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _sageattn2_rdna4_prefill(q_i4, q_scale, k_i4, k_scale, v_e4m3, v_scale)


def test_rejects_overflow_scale():
    q_i4, q_scale, k_i4, k_scale, v_e4m3, v_scale = _valid(D=128)
    q_scale.fill_(1e14)
    k_scale.fill_(1e14)
    with pytest.raises(ValueError, match="worst-case"):
        _sageattn2_rdna4_prefill(q_i4, q_scale, k_i4, k_scale, v_e4m3, v_scale)


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("D", [64, 128])
def test_prefill_runs(D, is_causal):
    out = _sageattn2_rdna4_prefill(*_valid(Sq=64, Skp=64, D=D), is_causal=is_causal)
    torch.cuda.synchronize()
    assert out.shape == (1, 1, 64, D)
    assert torch.isfinite(out.float()).all()
