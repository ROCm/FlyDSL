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


def test_ragged_paged_value_head() -> None:
    torch.manual_seed(5)
    page, hq, dq, dv = 16, 4, 64, 32
    k = torch.randn(3, page, 1, dq, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(3, page, 1, dv, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(2, 1, hq, dq, device="cuda", dtype=torch.bfloat16)
    bt = torch.tensor([[0, 0], [1, 2]], device="cuda", dtype=torch.int32)
    sk = torch.tensor([page, page * 2], device="cuda", dtype=torch.int32)
    out = flydsl_flash_attn_func(q, k, v, block_table=bt, seqlen_k=sk, causal=False)
    assert out.shape == (2, 1, hq, dv)
    lengths = [page, page * 2]
    pages = [bt[0, :1], bt[1]]
    for seq, n_pages in enumerate((1, 2)):
        kk = k[pages[seq], :, 0].reshape(1, lengths[seq], 1, dq).expand(1, lengths[seq], hq, dq)
        vv = v[pages[seq], :, 0].reshape(1, lengths[seq], 1, dv).expand(1, lengths[seq], hq, dv)
        ref = F.scaled_dot_product_attention(
            q[seq : seq + 1].float().permute(0, 2, 1, 3),
            kk.float().permute(0, 2, 1, 3),
            vv.float().permute(0, 2, 1, 3),
            scale=dq**-0.5,
        )
        err = (out[seq].float() - ref.permute(0, 2, 1, 3)).abs().max().item()
        assert err < 2e-2, err

    for quant, qmax in (
        (torch.float8_e4m3fn, 448.0),
        (torch.float8_e5m2, 57344.0),
        (torch.int8, 127.0),
    ):
        qf = q.float()
        kf = k.float()
        vf = v.float()
        scale = float(max(qf.abs().amax().item(), kf.abs().amax().item(), vf.abs().amax().item(), 1e-6) / qmax)
        if quant is torch.int8:
            q8 = (qf / scale).round().clamp(-128, 127).to(quant)
            k8 = (kf / scale).round().clamp(-128, 127).to(quant)
            v8 = (vf / scale).round().clamp(-128, 127).to(quant)
        else:
            q8 = (qf / scale).to(quant)
            k8 = (kf / scale).to(quant)
            v8 = (vf / scale).to(quant)
        out_q = flydsl_flash_attn_func(
            q8,
            k8,
            v8,
            block_table=bt,
            seqlen_k=sk,
            causal=False,
            q_descale=scale,
            k_descale=scale,
            v_descale=scale,
        )
        assert out_q.shape == (2, 1, hq, dv), quant
        for seq in range(2):
            qq = q8[seq : seq + 1]
            kk = k8[pages[seq]].reshape(1, lengths[seq], 1, dq).expand(1, lengths[seq], hq, dq)
            vv = v8[pages[seq]].reshape(1, lengths[seq], 1, dv).expand(1, lengths[seq], hq, dv)
            dense = flydsl_flash_attn_func(
                qq,
                kk,
                vv,
                causal=False,
                q_descale=scale,
                k_descale=scale,
                v_descale=scale,
            )
            assert _max_abs(out_q[seq], dense) < 1e-5, quant


def test_decode_and_mla_dtypes() -> None:
    """fp16, both fp8 types, and int8 on the decode launch and on MLA."""
    torch.manual_seed(6)
    page, h, d = 16, 2, 64
    bt = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    sk = torch.tensor([32], device="cuda", dtype=torch.int32)
    for dtype, qmax in (
        (torch.float16, None),
        (torch.float8_e4m3fn, 448.0),
        (torch.float8_e5m2, 57344.0),
        (torch.int8, 127.0),
    ):
        qf = torch.randn(1, 1, h, d, device="cuda")
        kf = torch.randn(2, page, h, d, device="cuda")
        vf = torch.randn(2, page, h, d, device="cuda")
        scale = 1.0
        if qmax is None:
            q, k, v = qf.to(dtype), kf.to(dtype), vf.to(dtype)
            key_scale = value_scale = None
        else:
            scale = float(max(qf.abs().amax().item(), kf.abs().amax().item(), vf.abs().amax().item(), 1e-6) / qmax)
            cast = (lambda t: (t / scale).round().clamp(-128, 127).to(dtype)) if dtype is torch.int8 else (lambda t: (t / scale).to(dtype))
            q, k, v = cast(qf), cast(kf), cast(vf)
            key_scale = value_scale = scale
        out = torch.empty(1, h, d, device="cuda", dtype=torch.bfloat16 if qmax else dtype)
        pa_decode_ps_launch(
            out,
            q.reshape(1, h, d),
            k,
            v,
            sk,
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.zeros(2, dtype=torch.int32, device="cuda"),
            d**-0.5,
            key_scale,
            value_scale,
            block_tables=bt,
            max_context_partition_num=1,
        )
        ref = flydsl_flash_attn_func(
            q,
            k.reshape(1, 32, h, d),
            v.reshape(1, 32, h, d),
            causal=False,
            sm_scale=d**-0.5,
            q_descale=key_scale,
            k_descale=key_scale,
            v_descale=value_scale,
        )
        assert _max_abs(out, ref.reshape(1, h, d)) < 1e-2, dtype

    for dtype in (torch.float16, torch.float8_e4m3fn, torch.int8):
        kv_f = torch.randn(4, 1, 1, 64, device="cuda")
        q_f = torch.randn(2, 4, 64, device="cuda")
        if dtype is torch.float16:
            kv, qq = kv_f.to(dtype), q_f.to(dtype)
            out_dtype = dtype
        else:
            qmax = 127.0 if dtype is torch.int8 else 448.0
            scale = float(max(kv_f.abs().amax().item(), q_f.abs().amax().item(), 1e-6) / qmax)
            cast = (lambda t: (t / scale).round().clamp(-128, 127).to(dtype)) if dtype is torch.int8 else (lambda t: (t / scale).to(dtype))
            kv, qq = cast(kv_f), cast(q_f)
            out_dtype = torch.bfloat16
        out = torch.empty(2, 4, 32, device="cuda", dtype=out_dtype)
        flydsl_mla_fwd_decode(
            qq,
            kv,
            torch.tensor([3, 1, 0, 2], dtype=torch.int32),
            torch.tensor([0, 2, 4], dtype=torch.int32),
            torch.empty(1, dtype=torch.int32, device="cuda"),
            out,
            torch.empty(1, device="cuda", dtype=out_dtype),
            torch.empty(1, device="cuda", dtype=out_dtype),
            64**-0.5,
        )
        assert out.shape[-1] == 32
        assert torch.isfinite(out.float()).all(), dtype


def test_window_plus_sink() -> None:
    torch.manual_seed(7)
    q = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    sink = torch.tensor([0.3, -0.2], device="cuda")
    out = flydsl_flash_attn_func(q, q, q, causal=True, sliding_window=(4, 0), sink=sink)
    qf = q.float().transpose(1, 2)
    scores = torch.matmul(qf, qf.transpose(-1, -2)) * (64**-0.5)
    i = torch.arange(32, device="cuda")
    shifted = i[None, :] - i[:, None] + 4
    allow = (shifted >= 0) & (shifted <= 4) & (i[None, :] <= i[:, None])
    scores = scores.masked_fill(~allow.view(1, 1, 32, 32), float("-inf"))
    scores = torch.cat([scores, sink.view(1, 2, 1, 1).expand(1, 2, 32, 1)], dim=-1)
    prob = torch.softmax(scores, dim=-1)[..., :-1]
    ref = torch.matmul(prob, qf).transpose(1, 2)
    ref_dt = torch.matmul(prob.to(torch.bfloat16).float(), qf).transpose(1, 2)
    assert _max_abs(out, ref) <= 2 * _max_abs(ref_dt, ref) + 1e-3
