# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x decode launch, MLA, linear attention, and quant QK-norm."""

import sys
from pathlib import Path

import pytest

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402

from kernels.attention.flash_attn_gfx120x_host import flydsl_flash_attn_func  # noqa: E402
from kernels.attention.linear_attn_gfx120x import linear_attn  # noqa: E402
from kernels.attention.mla_fwd_decode_gfx120x import flydsl_mla_fwd_decode  # noqa: E402
from kernels.attention.pa_decode_fp8_gfx120x import pa_decode_ps_launch  # noqa: E402
from kernels.attention.pa_decode_swa_gfx120x import compile_pa_decode_sw  # noqa: E402
from kernels.norm.qk_norm_rope_quant_gfx120x import qk_norm_rope_quant  # noqa: E402
from tests.kernels.test_pa_decode_tile_gfx120x import _pack_vectorized  # noqa: E402


def _max_abs(a, b) -> float:
    return (a.float() - b.float()).abs().max().item()


def test_ps_launch_window_matches_flash() -> None:
    torch.manual_seed(1)
    page, h, d, sk, window = 16, 2, 64, 32, 4
    q = torch.randn(1, 1, h, d, device="cuda", dtype=torch.bfloat16)
    k_lin = torch.randn(2, page, h, d, device="cuda", dtype=torch.bfloat16)
    v_lin = torch.randn(2, page, h, d, device="cuda", dtype=torch.bfloat16)
    k5, v5 = _pack_vectorized(k_lin, v_lin)
    bt = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    sk_t = torch.tensor([sk], device="cuda", dtype=torch.int32)
    out = torch.empty(1, h, d, device="cuda", dtype=torch.bfloat16)
    kind = pa_decode_ps_launch(
        out,
        q.reshape(1, h, d),
        k5,
        v5,
        sk_t,
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        torch.zeros(2, dtype=torch.int32, device="cuda"),
        d**-0.5,
        sliding_window=window,
        block_tables=bt,
        max_context_partition_num=1,
    )
    assert kind == "ps_sw_fused_partitioned"
    assert compile_pa_decode_sw(window, 1, head_dim=d)["sliding_window"] == window
    ref = flydsl_flash_attn_func(
        q,
        k_lin.reshape(1, sk, h, d),
        v_lin.reshape(1, sk, h, d),
        causal=True,
        sliding_window=(window, 0),
        sm_scale=d**-0.5,
    )
    assert _max_abs(out, ref.reshape(1, h, d)) < 1e-2


def test_mla_value_width() -> None:
    torch.manual_seed(2)
    batch, hq, qk, dv, page = 2, 4, 64, 32, 1
    counts = [2, 2]
    indptr = torch.tensor([0, 2, 4], dtype=torch.int32)
    indices = torch.tensor([3, 1, 0, 2], dtype=torch.int32)
    kv = torch.randn(4, page, 1, qk, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(batch, hq, qk, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(batch, hq, dv, device="cuda", dtype=torch.bfloat16)
    flydsl_mla_fwd_decode(
        q,
        kv,
        indices,
        indptr,
        torch.empty(1, dtype=torch.int32, device="cuda"),
        out,
        torch.empty(1, device="cuda"),
        torch.empty(1, device="cuda"),
        qk**-0.5,
    )
    for seq, n in enumerate(counts):
        pages = indices[int(indptr[seq]) : int(indptr[seq]) + n]
        k = kv[pages, 0, 0].reshape(1, n, 1, qk).expand(1, n, hq, qk)
        v = kv[pages, 0, 0, :dv].reshape(1, n, 1, dv).expand(1, n, hq, dv)
        ref = F.scaled_dot_product_attention(
            q[seq : seq + 1].float().permute(1, 0, 2).unsqueeze(0),
            k.float().permute(0, 2, 1, 3),
            v.float().permute(0, 2, 1, 3),
            scale=qk**-0.5,
        )
        got = out[seq].float()
        err = (got - ref[0, :, 0, :]).abs().max().item()
        assert err < 2e-2, err


def _recurrence(q, k, v, q_scale=1.0, k_scale=1.0, v_scale=1.0):
    b, s, h, d = q.shape
    dv = v.shape[-1]
    out = torch.empty(b, s, h, dv, device=q.device, dtype=torch.float32)
    qf, kf, vf = q.float() * q_scale, k.float() * k_scale, v.float() * v_scale
    for bi in range(b):
        for hi in range(h):
            state = torch.zeros(d, dv, device=q.device)
            for t in range(s):
                state = state + torch.outer(kf[bi, t, hi], vf[bi, t, hi])
                out[bi, t, hi] = qf[bi, t, hi] @ state
    return out


def test_linear_attn_bf16_and_fp8() -> None:
    torch.manual_seed(3)
    q = torch.randn(1, 4, 2, 16, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, 4, 2, 8, device="cuda", dtype=torch.bfloat16)
    out = linear_attn(q, k, v)
    ref = _recurrence(q, k, v)
    ref_dt = ref.to(out.dtype)
    assert _max_abs(out, ref) <= 2 * _max_abs(ref_dt, ref) + 1e-3

    qh = q.to(torch.float16)
    kh = k.to(torch.float16)
    vh = v.to(torch.float16)
    out = linear_attn(qh, kh, vh)
    ref = _recurrence(qh, kh, vh)
    ref_dt = ref.to(out.dtype)
    assert _max_abs(out, ref) <= 2 * _max_abs(ref_dt, ref) + 1e-3

    for quant, qmax, mult in (
        (torch.float8_e4m3fn, 448.0, 4),
        (torch.float8_e5m2, 57344.0, 4),
        (torch.int8, 127.0, 4),
    ):
        qf = torch.randn(1, 4, 2, 16, device="cuda")
        vf = torch.randn(1, 4, 2, 8, device="cuda")
        scale = float(max(qf.abs().amax().item(), vf.abs().amax().item(), 1e-6) / qmax)
        q8 = (qf / scale).to(quant)
        v8 = (vf / scale).to(quant)
        out = linear_attn(q8, q8, v8, q_scale=scale, k_scale=scale, v_scale=scale)
        ref = _recurrence(q8, q8, v8, scale, scale, scale)
        ref_dt = ref.to(out.dtype)
        assert _max_abs(out, ref) <= mult * _max_abs(ref_dt, ref) + 1e-2


def test_qk_norm_rope_quant_then_attention() -> None:
    torch.manual_seed(4)
    rows, head, rope = 4, 64, 32
    x = torch.randn(rows, head, device="cuda", dtype=torch.bfloat16)
    weight = torch.rand(head, device="cuda", dtype=torch.bfloat16) + 0.25
    pos = torch.arange(rows, device="cuda")
    pairs = rope // 2
    cos = torch.randn(rows, pairs, device="cuda")
    sin = torch.randn(rows, pairs, device="cuda")
    deq = None
    for quant, qmax, dtype in (
        ("fp8_e4m3fn", 448.0, torch.float8_e4m3fn),
        ("fp8_e5m2", 57344.0, torch.float8_e5m2),
        ("int8", 127.0, torch.int8),
    ):
        packed, scale = qk_norm_rope_quant(x, weight, cos, sin, pos, quant=quant)
        y = x.float()
        y = y * torch.rsqrt((y * y).mean(-1, keepdim=True) + 1e-6) * weight.float()
        tail = y[:, head - rope :].clone()
        t0, t1 = tail[:, 0::2].clone(), tail[:, 1::2].clone()
        tail[:, 0::2] = t0 * cos - t1 * sin
        tail[:, 1::2] = t0 * sin + t1 * cos
        y2 = y.clone()
        y2[:, head - rope :] = tail
        expect = y2.abs().amax(-1).clamp_min(1e-12) / qmax
        assert _max_abs(scale, expect) < 1e-3, quant
        got = packed.float() * scale[:, None]
        codes = (y2 / expect[:, None]).to(dtype)
        ref_q = codes.float() * expect[:, None]
        assert _max_abs(got, y2) <= 4 * _max_abs(ref_q, y2) + 1e-3, quant
        if quant == "fp8_e4m3fn":
            deq = got
    q = deq.view(1, rows, 1, head).to(torch.bfloat16)
    attn = flydsl_flash_attn_func(q, q, q, causal=False)
    assert attn.shape == q.shape
    assert torch.isfinite(attn.float()).all()
