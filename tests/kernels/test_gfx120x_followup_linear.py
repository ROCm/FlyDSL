# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Running-state linear attention matches the prefix-dot recurrence."""

import inspect
import sys
from pathlib import Path

import pytest

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

torch = pytest.importorskip("torch")

from kernels.attention.linear_attn_gfx120x import linear_attn  # noqa: E402


def _max_abs(a, b) -> float:
    return (a.float() - b.float()).abs().max().item()


def _matrix_state(q, k, v, q_scale=1.0, k_scale=1.0, v_scale=1.0):
    """``h_t = h_{t-1} + outer(k_t, v_t)``, ``o_t = q_t @ h_t``, ``h_{-1} = 0``."""
    qf = q.to(torch.float64) * q_scale
    kf = k.to(torch.float64) * k_scale
    vf = v.to(torch.float64) * v_scale
    b, s, h, d = qf.shape
    dv = vf.shape[-1]
    out = torch.empty(b, s, h, dv, dtype=torch.float64)
    for bi in range(b):
        for hi in range(h):
            state = torch.zeros(d, dv, dtype=torch.float64)
            for t in range(s):
                state = state + torch.outer(kf[bi, t, hi], vf[bi, t, hi])
                out[bi, t, hi] = qf[bi, t, hi] @ state
    return out


def _prefix_dots(q, k, v, q_scale=1.0, k_scale=1.0, v_scale=1.0):
    """Prefix dots: ``o_t[j] = sum_{s<=t} (q_t · k_s) v_s[j]``."""
    qf = q.to(torch.float64) * q_scale
    kf = k.to(torch.float64) * k_scale
    vf = v.to(torch.float64) * v_scale
    b, s, h, _d = qf.shape
    dv = vf.shape[-1]
    out = torch.empty(b, s, h, dv, dtype=torch.float64)
    for bi in range(b):
        for hi in range(h):
            for t in range(s):
                acc = torch.zeros(dv, dtype=torch.float64)
                qt = qf[bi, t, hi]
                for step in range(t + 1):
                    acc = acc + torch.dot(qt, kf[bi, step, hi]) * vf[bi, step, hi]
                out[bi, t, hi] = acc
    return out


def _columns(q, k, v, q_scale=1.0, k_scale=1.0, v_scale=1.0):
    """One register column per value channel, updated in token order."""
    qf = q.to(torch.float64) * q_scale
    kf = k.to(torch.float64) * k_scale
    vf = v.to(torch.float64) * v_scale
    b, s, h, d = qf.shape
    dv = vf.shape[-1]
    out = torch.empty(b, s, h, dv, dtype=torch.float64)
    for bi in range(b):
        for hi in range(h):
            for j in range(dv):
                state = [0.0] * d
                for t in range(s):
                    vv = float(vf[bi, t, hi, j])
                    acc = 0.0
                    for i in range(d):
                        state[i] = state[i] + float(kf[bi, t, hi, i]) * vv
                        acc = acc + float(qf[bi, t, hi, i]) * state[i]
                    out[bi, t, hi, j] = acc
    return out


def _exact_case():
    # t=0 writes the first outer product. t=1 adds nothing. t=2 extends h.
    q = torch.tensor([[[[1.0, 0.0]], [[1.0, 1.0]], [[0.0, 1.0]]]], dtype=torch.float64)
    k = torch.tensor([[[[1.0, 2.0]], [[0.0, 0.0]], [[3.0, 4.0]]]], dtype=torch.float64)
    v = torch.tensor([[[[5.0, 7.0]], [[0.0, 0.0]], [[6.0, 8.0]]]], dtype=torch.float64)
    expect = torch.tensor([[[[5.0, 7.0]], [[15.0, 21.0]], [[34.0, 46.0]]]], dtype=torch.float64)
    return q, k, v, expect


def test_signature_has_no_decay() -> None:
    names = list(inspect.signature(linear_attn).parameters)
    assert names == ["q", "k", "v", "q_scale", "k_scale", "v_scale", "stream"]


def test_running_state_matches_prefix_dots() -> None:
    q, k, v, expect = _exact_case()
    for got in (_matrix_state(q, k, v), _prefix_dots(q, k, v), _columns(q, k, v)):
        assert torch.equal(got, expect)

    torch.manual_seed(7)
    q = torch.randn(2, 6, 3, 5)
    k = torch.randn_like(q)
    v = torch.randn(2, 6, 3, 4)
    q[:, 2] = q[:, 1]
    k[:, 2] = 0
    v[:, 2] = 0
    scales = (0.5, -1.25, 2.0)
    matrix = _matrix_state(q, k, v, *scales)
    assert torch.allclose(matrix, _prefix_dots(q, k, v, *scales), rtol=1e-9, atol=1e-9)
    assert torch.allclose(matrix, _columns(q, k, v, *scales), rtol=1e-9, atol=1e-9)
    # A zero key/value token leaves h in place, so the same query repeats.
    assert torch.allclose(matrix[:, 2], matrix[:, 1], rtol=1e-9, atol=1e-9)


def _require_gfx120() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")
    from flydsl.runtime.device import get_rocm_arch

    if not str(get_rocm_arch()).startswith("gfx120"):
        pytest.skip(f"linear attention requires gfx120x, got {get_rocm_arch()}")


def test_linear_attn_matches_recurrence() -> None:
    _require_gfx120()
    q, k, v, expect = _exact_case()
    out = linear_attn(
        q.to(device="cuda", dtype=torch.bfloat16),
        k.to(device="cuda", dtype=torch.bfloat16),
        v.to(device="cuda", dtype=torch.bfloat16),
    )
    assert torch.equal(out.float().cpu(), expect.float())

    qh = q.to(device="cuda", dtype=torch.float16)
    kh = k.to(device="cuda", dtype=torch.float16)
    vh = v.to(device="cuda", dtype=torch.float16)
    out = linear_attn(qh, kh, vh)
    assert torch.equal(out.float().cpu(), expect.float())

    torch.manual_seed(8)
    q = torch.randn(2, 5, 3, 7, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(2, 5, 3, 4, device="cuda", dtype=torch.bfloat16)
    out = linear_attn(q, k, v)
    ref = _matrix_state(q.cpu(), k.cpu(), v.cpu()).float()
    ref_dt = ref.to(out.dtype)
    assert _max_abs(out.cpu(), ref) <= 2 * _max_abs(ref_dt, ref) + 1e-3

    codes = torch.ones(1, 2, 1, 1, device="cuda", dtype=torch.int8)
    values = torch.ones(1, 2, 1, 1, device="cuda", dtype=torch.int8)
    out = linear_attn(codes, codes, values, q_scale=2.0, k_scale=3.0, v_scale=4.0)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out.float().cpu(), torch.tensor([[[[24.0]], [[48.0]]]]))

    one = torch.ones(1, 2, 1, 1, device="cuda", dtype=torch.float32)
    for quant in (torch.float8_e4m3fn, torch.float8_e5m2):
        q8 = one.to(quant)
        out = linear_attn(q8, q8, q8, q_scale=2.0, k_scale=3.0, v_scale=4.0)
        assert out.dtype == torch.bfloat16
        assert torch.equal(out.float().cpu(), torch.tensor([[[[24.0]], [[48.0]]]])), quant
