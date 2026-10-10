# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""One pass over the gfx120x attention paths that are actually implemented.

Each section is named ``dtype/op``. The first failure raises with that name.
"""

import math
import sys
from pathlib import Path

import pytest

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.attention.flash_attn_gfx120x_host import fold_alibi_to_bias  # noqa: E402
from kernels.attention.flash_attn_interface import flydsl_flash_attn_func  # noqa: E402

_KINDS = ("bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "int8")
_FLOAT = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _arch() -> str:
    return (torch.cuda.get_device_properties(0).gcnArchName or "").split(":")[0]


pytestmark = [
    pytest.mark.l2_device,
    pytest.mark.rocm_lower,
    pytest.mark.skipif(
        not str(get_rocm_arch() or "").startswith("gfx120"),
        reason=f"requires gfx120x, got {_arch()!r}",
    ),
]


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(-1, a.shape[-1])
    bf = b.float().reshape(-1, b.shape[-1])
    return float(F.cosine_similarity(af, bf, dim=1).min())


def _sdpa(q, k, v, *, causal=False, attn_mask=None):
    qq, kk, vv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if attn_mask is not None and attn_mask.is_floating_point() and attn_mask.dtype != qq.dtype:
        attn_mask = attn_mask.to(dtype=qq.dtype)
    return F.scaled_dot_product_attention(qq, kk, vv, is_causal=causal, attn_mask=attn_mask).transpose(1, 2)


def _quantize(kind: str, x: torch.Tensor):
    amax = x.abs().amax().clamp(min=1e-6)
    if kind == "fp8_e4m3":
        scale = amax / 448
        return (x / scale).to(torch.float8_e4m3fn), scale
    if kind == "fp8_e5m2":
        scale = amax / 57344
        return (x / scale).to(torch.float8_e5m2), scale
    scale = amax / 127
    return (x / scale).round().clamp(-128, 127).to(torch.int8), scale


def _pack(kind: str, shape: tuple[int, ...], seed: int):
    """Return q, k, v, descale kwargs, and the bf16/fp16 tensors SDPA should see."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    raw = torch.randn(*shape, device="cuda", generator=g)
    if kind in _FLOAT:
        t = raw.to(_FLOAT[kind])
        return t, {}
    q, scale = _quantize(kind, raw)
    return q, {"q_descale": scale, "k_descale": scale, "v_descale": scale}


def _self(kind: str, n: int, h: int, d: int, seed: int):
    """One tensor used as q, k, and v. Quant descales are the same scale."""
    q, scales = _pack(kind, (1, n, h, d), seed)
    if scales:
        scale = scales["q_descale"]
        scales = {"q_descale": scale, "k_descale": scale, "v_descale": scale}
    ref = q if kind in _FLOAT else (q.float() * scales["q_descale"]).to(torch.bfloat16)
    return q, scales, ref


def _qkv(kind: str, b: int, sq: int, sk: int, hq: int, hkv: int, d: int, seed: int):
    q, qs = _pack(kind, (b, sq, hq, d), seed)
    k, ks = _pack(kind, (b, sk, hkv, d), seed + 1)
    v, vs = _pack(kind, (b, sk, hkv, d), seed + 2)
    scales = {}
    if kind not in _FLOAT:
        scales = {"q_descale": qs["q_descale"], "k_descale": ks["q_descale"], "v_descale": vs["q_descale"]}
    ref_q = q if kind in _FLOAT else (q.float() * scales["q_descale"]).to(torch.bfloat16)
    ref_k = k if kind in _FLOAT else (k.float() * scales["k_descale"]).to(torch.bfloat16)
    ref_v = v if kind in _FLOAT else (v.float() * scales["v_descale"]).to(torch.bfloat16)
    return q, k, v, scales, ref_q, ref_k, ref_v


def _call(q, k, v, scales, **kwargs):
    return flydsl_flash_attn_func(q, k, v, **scales, **kwargs)


def _check(name: str, out: torch.Tensor, ref: torch.Tensor, floor: float) -> None:
    if out.dtype != ref.dtype:
        raise AssertionError(f"{name} dtype {out.dtype} != {ref.dtype}")
    score = _cos(out, ref)
    if score <= floor:
        raise AssertionError(f"{name} cosine {score:.4f} <= {floor}")


def _window(sq: int, sk: int, left: int, right: int, causal: bool, device) -> torch.Tensor:
    rows = torch.arange(sq, device=device)[:, None]
    cols = torch.arange(sk, device=device)[None, :]
    # Causal windows sit on the bottom-right key, which is q + (Sk - Sq).
    origin = rows + (sk - sq) if causal else rows
    keep = (cols >= origin - left) & (cols <= origin + right)
    if causal:
        keep = keep & (cols <= rows + (sk - sq))
    return torch.where(keep, torch.zeros((), device=device), torch.full((), float("-inf"), device=device))


def _sink_ref(ref_q, ref_k, ref_v, sink: torch.Tensor, bias: torch.Tensor | None):
    b, sq, h, d = ref_q.shape
    qf = ref_q.float().transpose(1, 2)
    kf = ref_k.float().transpose(1, 2)
    vf = ref_v.float().transpose(1, 2)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) / math.sqrt(d)
    if bias is not None:
        scores = scores + bias.float().view(1, h, sq, -1)
    sink_col = sink.view(1, h, 1, 1).expand(b, h, sq, 1)
    scores_s = torch.cat([scores, sink_col], dim=-1)
    prob = torch.softmax(scores_s, dim=-1)[..., :-1]
    out = torch.matmul(prob, vf).transpose(1, 2)
    lse = torch.logsumexp(scores_s, dim=-1)
    return out, lse


def _run(name: str, fn) -> None:
    try:
        fn()
    except Exception as exc:
        raise AssertionError(f"{name} failed: {exc}") from exc


def _must_raise(name: str, fn, exc_type: type[BaseException]) -> None:
    try:
        fn()
    except exc_type:
        return
    except Exception as exc:
        raise AssertionError(f"{name} raised {type(exc).__name__}, expected {exc_type.__name__}: {exc}") from exc
    raise AssertionError(f"{name} did not raise {exc_type.__name__}")


def test_gfx120x_attention_covers_implemented_ops() -> None:
    """Walk every implemented dtype through the ops the router actually runs."""
    # bf16/fp16 versus PyTorch SDPA. 0.975 clears bf16 rounding on a small
    # additive bias. Split-K versus the same kernel stays at 0.99.
    floor_f, floor_q = 0.975, 0.97
    stream = torch.cuda.Stream()

    def kind_floor(kind: str) -> float:
        return floor_f if kind in _FLOAT else floor_q

    for kind in _KINDS:
        if kind == "fp8_e5m2" and not hasattr(torch, "float8_e5m2"):
            continue

        def dense(kind=kind):
            q, k, v, scales, rq, rk, rv = _qkv(kind, 1, 32, 32, 2, 2, 64, 1)
            with torch.cuda.stream(stream):
                out = _call(q, k, v, scales, causal=False, stream=stream)
            stream.synchronize()
            _check(f"{kind}/dense", out, _sdpa(rq, rk, rv), kind_floor(kind))

        def causal(kind=kind):
            q, k, v, scales, rq, rk, rv = _qkv(kind, 1, 32, 32, 2, 2, 64, 2)
            with torch.cuda.stream(stream):
                out = _call(q, k, v, scales, causal=True, stream=stream)
            stream.synchronize()
            _check(f"{kind}/causal", out, _sdpa(rq, rk, rv, causal=True), kind_floor(kind))

        def bias(kind=kind):
            q, k, v, scales, rq, rk, rv = _qkv(kind, 1, 32, 32, 2, 2, 64, 3)
            mask = torch.randn(32, 32, device="cuda") * 0.1
            with torch.cuda.stream(stream):
                out = _call(q, k, v, scales, causal=False, bias=mask, stream=stream)
            stream.synchronize()
            _check(f"{kind}/bias", out, _sdpa(rq, rk, rv, attn_mask=mask), kind_floor(kind))

        def alibi(kind=kind):
            # Self-attention. A large additive bias on q≠k does not match SDPA.
            q, scales, rq = _self(kind, 32, 2, 64, 4)
            slopes = 0.2
            folded = fold_alibi_to_bias(slopes, 32, 32, q.device, stream=stream)
            stream.synchronize()
            with torch.cuda.stream(stream):
                out = _call(q, q, q, scales, causal=False, alibi_slopes=slopes, stream=stream)
            stream.synchronize()
            _check(f"{kind}/alibi", out, _sdpa(rq, rq, rq, attn_mask=folded), kind_floor(kind))

        def sink_lse(kind=kind):
            q, k, v, scales, rq, rk, rv = _qkv(kind, 1, 32, 32, 2, 2, 64, 5)
            sink = torch.tensor([0.2, -0.3], device="cuda")
            with torch.cuda.stream(stream):
                out, lse = _call(q, k, v, scales, causal=False, sink=sink, return_lse=True, stream=stream)
            stream.synchronize()
            ref, lse_ref = _sink_ref(rq, rk, rv, sink, None)
            _check(f"{kind}/sink", out, ref.to(out.dtype), kind_floor(kind))
            if not torch.allclose(lse, lse_ref, rtol=2e-2, atol=2e-2):
                err = float((lse - lse_ref).abs().max())
                raise AssertionError(f"{kind}/lse max abs {err:.4f}")

        def splitk(kind=kind):
            q, k, v, scales, *_ = _qkv(kind, 1, 64, 64, 2, 2, 64, 6)
            with torch.cuda.stream(stream):
                one = _call(q, k, v, scales, causal=False, num_kv_splits=1, stream=stream)
                two = _call(q, k, v, scales, causal=False, num_kv_splits=2, stream=stream)
            stream.synchronize()
            _check(f"{kind}/splitk", two, one, 0.99)

        def gqa(kind=kind):
            q, k, v, scales, rq, rk, rv = _qkv(kind, 1, 32, 32, 4, 2, 64, 7)
            with torch.cuda.stream(stream):
                out = _call(q, k, v, scales, causal=False, num_kv_heads=2, stream=stream)
            stream.synchronize()
            ref = _sdpa(rq, rk.repeat_interleave(2, dim=2), rv.repeat_interleave(2, dim=2))
            _check(f"{kind}/gqa", out, ref, kind_floor(kind))

        def varlen(kind=kind):
            q, k, v, scales, *_ = _qkv(kind, 1, 32, 32, 2, 2, 64, 8)
            cu = torch.tensor([0, 32], device="cuda", dtype=torch.int32)
            with torch.cuda.stream(stream):
                dense = _call(q, k, v, scales, causal=False, stream=stream)
                packed = _call(
                    q[0],
                    k[0],
                    v[0],
                    scales,
                    causal=False,
                    cu_seqlens_q=cu,
                    cu_seqlens_kv=cu,
                    max_seqlen_q=32,
                    max_seqlen_kv=32,
                    stream=stream,
                )
            stream.synchronize()
            _check(f"{kind}/varlen", packed.unsqueeze(0), dense, 0.98)

        def varlen_bias(kind=kind):
            q, k, v, scales, *_ = _qkv(kind, 1, 32, 32, 2, 2, 64, 9)
            mask = torch.randn(32, 32, device="cuda") * 0.1
            cu = torch.tensor([0, 32], device="cuda", dtype=torch.int32)
            with torch.cuda.stream(stream):
                dense = _call(q, k, v, scales, causal=False, bias=mask, stream=stream)
                packed = _call(
                    q[0],
                    k[0],
                    v[0],
                    scales,
                    causal=False,
                    bias=mask,
                    cu_seqlens_q=cu,
                    cu_seqlens_kv=cu,
                    max_seqlen_q=32,
                    max_seqlen_kv=32,
                    stream=stream,
                )
            stream.synchronize()
            _check(f"{kind}/varlen-bias", packed.unsqueeze(0), dense, 0.98)

        def paged(kind=kind):
            q, k, v, scales, *_ = _qkv(kind, 1, 16, 32, 2, 2, 64, 10)
            page = 16
            n_pages = 2
            cache_k = k[0].view(n_pages, page, 2, 64)
            cache_v = v[0].view(n_pages, page, 2, 64)
            bt = torch.arange(n_pages, device="cuda", dtype=torch.int32).view(1, n_pages)
            seqlen = torch.tensor([32], device="cuda", dtype=torch.int32)
            with torch.cuda.stream(stream):
                dense = _call(q, k, v, scales, causal=False, stream=stream)
                got = _call(
                    q,
                    cache_k,
                    cache_v,
                    scales,
                    causal=False,
                    block_table=bt,
                    seqlen_k=seqlen,
                    stream=stream,
                )
            stream.synchronize()
            _check(f"{kind}/paged", got, dense, 0.98)

        def linear3d(kind=kind):
            q, k, v, scales, *_ = _qkv(kind, 1, 16, 16, 2, 2, 64, 11)
            bt = torch.arange(16, device="cuda", dtype=torch.int32).view(1, 16)
            seqlen = torch.tensor([16], device="cuda", dtype=torch.int32)
            with torch.cuda.stream(stream):
                dense = _call(q, k, v, scales, causal=False, stream=stream)
                got = _call(
                    q,
                    k[0],
                    v[0],
                    scales,
                    causal=False,
                    block_table=bt,
                    seqlen_k=seqlen,
                    kv_cache_layout="linear3d",
                    stream=stream,
                )
            stream.synchronize()
            _check(f"{kind}/linear3d", got, dense, 0.98)

        def varlen_paged(kind=kind):
            q, k, v, scales, *_ = _qkv(kind, 1, 16, 16, 2, 2, 64, 12)
            cu = torch.tensor([0, 16], device="cuda", dtype=torch.int32)
            seqlen = torch.tensor([16], device="cuda", dtype=torch.int32)
            bt = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
            cache_k = k.view(1, 16, 2, 64)
            cache_v = v.view(1, 16, 2, 64)
            with torch.cuda.stream(stream):
                dense = _call(q, k, v, scales, causal=False, stream=stream)
                got = _call(
                    q[0],
                    cache_k,
                    cache_v,
                    scales,
                    causal=False,
                    cu_seqlens_q=cu,
                    cu_seqlens_kv=cu,
                    max_seqlen_q=16,
                    max_seqlen_kv=16,
                    block_table=bt,
                    seqlen_k=seqlen,
                    stream=stream,
                )
            stream.synchronize()
            _check(f"{kind}/varlen-paged", got.unsqueeze(0), dense, 0.98)

        _run(f"{kind}/dense", dense)
        _run(f"{kind}/causal", causal)
        _run(f"{kind}/bias", bias)
        _run(f"{kind}/alibi", alibi)
        _run(f"{kind}/sink-lse", sink_lse)
        _run(f"{kind}/splitk", splitk)
        _run(f"{kind}/gqa", gqa)
        _run(f"{kind}/varlen", varlen)
        _run(f"{kind}/varlen-bias", varlen_bias)
        _run(f"{kind}/paged", paged)
        _run(f"{kind}/linear3d", linear3d)
        _run(f"{kind}/varlen-paged", varlen_paged)

        def window(kind=kind):
            q, k, v, scales, rq, rk, rv = _qkv(kind, 1, 32, 32, 2, 2, 64, 14)
            mask = _window(32, 32, 8, 4, False, q.device)
            with torch.cuda.stream(stream):
                out = _call(q, k, v, scales, causal=False, sliding_window=(8, 4), stream=stream)
            stream.synchronize()
            _check(f"{kind}/sliding-window", out, _sdpa(rq, rk, rv, attn_mask=mask), kind_floor(kind))

        _run(f"{kind}/sliding-window", window)

        def causal_window_decode(kind=kind):
            # Sq=1 attends the tail of the cache, not keys [0, left].
            q, k, v, scales, rq, rk, rv = _qkv(kind, 1, 1, 32, 2, 2, 64, 21)
            mask = _window(1, 32, 4, 0, True, q.device)
            with torch.cuda.stream(stream):
                out = _call(q, k, v, scales, causal=True, sliding_window=(4, 0), stream=stream)
            stream.synchronize()
            _check(f"{kind}/causal-window-decode", out, _sdpa(rq, rk, rv, attn_mask=mask), kind_floor(kind))

        def varlen_alibi_ragged(kind=kind):
            sqs, sks = [8, 20], [24, 12]
            h, d = 2, 64
            q_parts, k_parts, v_parts = [], [], []
            for i, (sq, sk) in enumerate(zip(sqs, sks)):
                qi, ki, vi, *_ = _qkv(kind if kind in _FLOAT else "bf16", 1, sq, sk, h, h, d, 40 + i)
                q_parts.append(qi[0].float())
                k_parts.append(ki[0].float())
                v_parts.append(vi[0].float())
            qf = torch.cat(q_parts, 0)
            kf = torch.cat(k_parts, 0)
            vf = torch.cat(v_parts, 0)
            scales = {}
            if kind in _FLOAT:
                q, k, v = qf.to(_FLOAT[kind]), kf.to(_FLOAT[kind]), vf.to(_FLOAT[kind])
                rq, rk, rv = q, k, v
            else:
                q, qs = _quantize(kind, qf)
                k, ks = _quantize(kind, kf)
                v, vs = _quantize(kind, vf)
                scales = {"q_descale": qs, "k_descale": ks, "v_descale": vs}
                rq = (q.float() * qs).to(torch.bfloat16)
                rk = (k.float() * ks).to(torch.bfloat16)
                rv = (v.float() * vs).to(torch.bfloat16)
            cu_q = torch.tensor([0, sqs[0], sqs[0] + sqs[1]], device="cuda", dtype=torch.int32)
            cu_k = torch.tensor([0, sks[0], sks[0] + sks[1]], device="cuda", dtype=torch.int32)
            with torch.cuda.stream(stream):
                packed = _call(
                    q,
                    k,
                    v,
                    scales,
                    causal=False,
                    alibi_slopes=0.25,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_kv=cu_k,
                    max_seqlen_q=max(sqs),
                    max_seqlen_kv=max(sks),
                    stream=stream,
                )
            stream.synchronize()
            q_off = 0
            k_off = 0
            for i, (sq, sk) in enumerate(zip(sqs, sks)):
                folded = fold_alibi_to_bias(0.25, sq, sk, q.device, stream=stream)
                stream.synchronize()
                ref = _sdpa(
                    rq[q_off : q_off + sq].unsqueeze(0),
                    rk[k_off : k_off + sk].unsqueeze(0),
                    rv[k_off : k_off + sk].unsqueeze(0),
                    attn_mask=folded,
                )
                _check(
                    f"{kind}/varlen-alibi-ragged-{i}",
                    packed[q_off : q_off + sq].unsqueeze(0),
                    ref.to(packed.dtype),
                    kind_floor(kind),
                )
                q_off += sq
                k_off += sk

        _run(f"{kind}/causal-window-decode", causal_window_decode)
        _run(f"{kind}/varlen-alibi-ragged", varlen_alibi_ragged)

        if kind not in _FLOAT:

            def vectorized_quant(kind=kind):
                q, k, v, scales, *_ = _qkv(kind, 1, 16, 16, 2, 2, 64, 15)
                kvs, h, d, page, sk = 16, 2, 64, 16, 16
                k5 = torch.zeros(1, h, d // kvs, page, kvs, device="cuda", dtype=k.dtype)
                v5 = torch.zeros(1, h, page // kvs, d, kvs, device="cuda", dtype=v.dtype)
                logical_k, logical_v = k[0], v[0]
                for t in range(sk):
                    for head in range(h):
                        for dim in range(d):
                            k5[0, head, dim // kvs, t, dim % kvs] = logical_k[t, head, dim]
                            grp, rem = divmod(t, kvs)
                            v5[0, head, grp, dim, rem] = logical_v[t, head, dim]
                if int(k5.numel()) != int(k.numel()) or int(v5.numel()) != int(v.numel()):
                    raise AssertionError(f"{kind}/vectorized pack numel")
                bt = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
                seqlen = torch.tensor([sk], device="cuda", dtype=torch.int32)
                with torch.cuda.stream(stream):
                    dense = _call(q, k, v, scales, causal=False, stream=stream)
                    got = _call(
                        q,
                        k5,
                        v5,
                        scales,
                        causal=False,
                        block_table=bt,
                        seqlen_k=seqlen,
                        kv_cache_layout="vectorized",
                        stream=stream,
                    )
                stream.synchronize()
                if got.dtype != dense.dtype:
                    raise AssertionError(f"{kind}/vectorized dtype {got.dtype} != {dense.dtype}")
                _check(f"{kind}/vectorized", got, dense, 0.98)

            _run(f"{kind}/vectorized", vectorized_quant)

        def split_varlen(kind=kind):
            q, k, v, scales, *_ = _qkv(kind, 1, 32, 32, 2, 2, 64, 16)
            cu = torch.tensor([0, 32], device="cuda", dtype=torch.int32)
            kw = dict(
                causal=False,
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                max_seqlen_q=32,
                max_seqlen_kv=32,
            )
            with torch.cuda.stream(stream):
                one = _call(q[0], k[0], v[0], scales, num_kv_splits=1, stream=stream, **kw)
                two = _call(q[0], k[0], v[0], scales, num_kv_splits=2, stream=stream, **kw)
            stream.synchronize()
            _check(f"{kind}/splitk-varlen", two, one, 0.99)

        _run(f"{kind}/splitk-varlen", split_varlen)

    # Float vectorized paged is implemented. One dtype is enough: fp16 uses the same kernel.
    def vectorized_bf16():
        b, sq, sk, h, d, page, kvs = 1, 16, 16, 2, 64, 16, 8
        q, k, v, scales, *_ = _qkv("bf16", b, sq, sk, h, h, d, 17)
        nb = 1
        k5 = torch.zeros(nb, h, d // kvs, page, kvs, device="cuda", dtype=torch.bfloat16)
        v5 = torch.zeros(nb, h, page // kvs, d, kvs, device="cuda", dtype=torch.bfloat16)
        logical_k, logical_v = k[0], v[0]
        pid = 0
        for t in range(sk):
            for head in range(h):
                for dim in range(d):
                    k5[pid, head, dim // kvs, t, dim % kvs] = logical_k[t, head, dim]
                    grp, rem = divmod(t, kvs)
                    v5[pid, head, grp, dim, rem] = logical_v[t, head, dim]
        bt = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
        seqlen = torch.tensor([sk], device="cuda", dtype=torch.int32)
        with torch.cuda.stream(stream):
            dense = _call(q, k, v, scales, causal=False, stream=stream)
            got = _call(
                q,
                k5,
                v5,
                scales,
                causal=False,
                block_table=bt,
                seqlen_k=seqlen,
                kv_cache_layout="vectorized",
                stream=stream,
            )
        stream.synchronize()
        _check("bf16/vectorized", got, dense, 0.99)

    _run("bf16/vectorized", vectorized_bf16)

    def paged_split_bf16():
        q, k, v, scales, *_ = _qkv("bf16", 1, 16, 32, 2, 2, 64, 18)
        page, n_pages = 16, 2
        cache_k = k[0].view(n_pages, page, 2, 64)
        cache_v = v[0].view(n_pages, page, 2, 64)
        bt = torch.arange(n_pages, device="cuda", dtype=torch.int32).view(1, n_pages)
        seqlen = torch.tensor([32], device="cuda", dtype=torch.int32)
        with torch.cuda.stream(stream):
            one = _call(
                q,
                cache_k,
                cache_v,
                scales,
                causal=False,
                block_table=bt,
                seqlen_k=seqlen,
                num_kv_splits=1,
                stream=stream,
            )
            two = _call(
                q,
                cache_k,
                cache_v,
                scales,
                causal=False,
                block_table=bt,
                seqlen_k=seqlen,
                num_kv_splits=2,
                stream=stream,
            )
        stream.synchronize()
        if two.dtype != q.dtype:
            raise AssertionError(f"bf16/paged-splitk dtype {two.dtype}")
        _check("bf16/paged-splitk", two, one, 0.99)

    def varlen_per_head_bias():
        q, scales, rq = _self("bf16", 32, 2, 64, 19)
        bias = torch.randn(2, 32, 32, device="cuda") * 0.1
        cu = torch.tensor([0, 32], device="cuda", dtype=torch.int32)
        with torch.cuda.stream(stream):
            out = _call(
                q[0],
                q[0],
                q[0],
                scales,
                causal=False,
                bias=bias,
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                max_seqlen_q=32,
                max_seqlen_kv=32,
                stream=stream,
            )
        stream.synchronize()
        if out.dtype != q.dtype:
            raise AssertionError(f"bf16/varlen-per-head dtype {out.dtype}")
        ref = _sdpa(rq, rq, rq, attn_mask=bias)
        _check("bf16/varlen-per-head", out.unsqueeze(0), ref, 0.98)

    def varlen_paged_split_and_bias():
        q, k, v, scales, *_ = _qkv("bf16", 1, 16, 16, 2, 2, 64, 20)
        cu = torch.tensor([0, 16], device="cuda", dtype=torch.int32)
        seqlen = torch.tensor([16], device="cuda", dtype=torch.int32)
        bt = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
        cache_k = k.view(1, 16, 2, 64)
        cache_v = v.view(1, 16, 2, 64)
        bias = torch.randn(2, 16, 16, device="cuda") * 0.05
        common = dict(
            causal=False,
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=16,
            max_seqlen_kv=16,
            block_table=bt,
            seqlen_k=seqlen,
        )
        with torch.cuda.stream(stream):
            one = _call(q[0], cache_k, cache_v, scales, num_kv_splits=1, stream=stream, **common)
            two = _call(q[0], cache_k, cache_v, scales, num_kv_splits=2, stream=stream, **common)
            headed = _call(q[0], cache_k, cache_v, scales, bias=bias, stream=stream, **common)
        stream.synchronize()
        if two.dtype != q.dtype or headed.dtype != q.dtype:
            raise AssertionError(f"bf16/varlen-paged dtype {two.dtype} {headed.dtype}")
        _check("bf16/varlen-paged-splitk", two, one, 0.99)
        ref = _sdpa(q, k, v, attn_mask=bias)
        _check("bf16/varlen-paged-per-head", headed.unsqueeze(0), ref, 0.98)

    def quant_paged_split(kind="fp8_e4m3"):
        q, k, v, scales, *_ = _qkv(kind, 1, 16, 32, 2, 2, 64, 21)
        page, n_pages = 16, 2
        cache_k = k[0].view(n_pages, page, 2, 64)
        cache_v = v[0].view(n_pages, page, 2, 64)
        bt = torch.arange(n_pages, device="cuda", dtype=torch.int32).view(1, n_pages)
        seqlen = torch.tensor([32], device="cuda", dtype=torch.int32)
        with torch.cuda.stream(stream):
            one = _call(
                q,
                cache_k,
                cache_v,
                scales,
                causal=False,
                block_table=bt,
                seqlen_k=seqlen,
                num_kv_splits=1,
                stream=stream,
            )
            two = _call(
                q,
                cache_k,
                cache_v,
                scales,
                causal=False,
                block_table=bt,
                seqlen_k=seqlen,
                num_kv_splits=2,
                stream=stream,
            )
        stream.synchronize()
        if two.dtype != torch.bfloat16:
            raise AssertionError(f"{kind}/paged-splitk dtype {two.dtype}")
        _check(f"{kind}/paged-splitk", two, one, 0.99)

    def dense_per_head_and_cross():
        q, k, v, scales, rq, rk, rv = _qkv("bf16", 1, 32, 48, 2, 2, 64, 30)
        bias = torch.randn(2, 32, 48, device="cuda") * 0.5
        with torch.cuda.stream(stream):
            out = _call(q, k, v, scales, causal=False, bias=bias, stream=stream)
        stream.synchronize()
        _check("bf16/dense-per-head", out, _sdpa(rq, rk, rv, attn_mask=bias.view(1, 2, 32, 48)), 0.99)
        slopes = torch.tensor([0.2, 0.5], device="cuda")
        folded = fold_alibi_to_bias(slopes, 32, 48, q.device, stream=stream)
        stream.synchronize()
        with torch.cuda.stream(stream):
            ali = _call(q, k, v, scales, causal=False, alibi_slopes=slopes, stream=stream)
        stream.synchronize()
        _check("bf16/cross-alibi", ali, _sdpa(rq, rk, rv, attn_mask=folded.view(1, 2, 32, 48)), 0.99)

    def odd_d_split():
        q, k, v, scales, *_ = _qkv("bf16", 1, 32, 32, 2, 2, 80, 31)
        with torch.cuda.stream(stream):
            one = _call(q, k, v, scales, causal=False, num_kv_splits=1, stream=stream)
            two = _call(q, k, v, scales, causal=False, num_kv_splits=2, stream=stream)
        stream.synchronize()
        if two.shape != q.shape:
            raise AssertionError(f"bf16/odd-d-split shape {tuple(two.shape)}")
        _check("bf16/odd-d-split", two, one, 0.99)

    def int64_table_and_fp8_varlen_bias():
        q, k, v, scales, rq, rk, rv = _qkv("fp8_e4m3", 1, 16, 32, 2, 2, 64, 32)
        page, n_pages = 16, 2
        cache_k = k[0].view(n_pages, page, 2, 64)
        cache_v = v[0].view(n_pages, page, 2, 64)
        bt32 = torch.arange(n_pages, device="cuda", dtype=torch.int32).view(1, n_pages)
        bt64 = bt32.to(torch.int64)
        seqlen = torch.tensor([32], device="cuda", dtype=torch.int64)
        with torch.cuda.stream(stream):
            a = _call(
                q,
                cache_k,
                cache_v,
                scales,
                causal=False,
                block_table=bt32,
                seqlen_k=seqlen.to(torch.int32),
                stream=stream,
            )
            b = _call(q, cache_k, cache_v, scales, causal=False, block_table=bt64, seqlen_k=seqlen, stream=stream)
        stream.synchronize()
        _check("fp8_e4m3/int64-table", b, a, 0.999)
        # Two packed sequences, one descale. A shared mask must hit the second sequence.
        n0, n1 = 16, 24
        g = torch.Generator(device="cuda")
        g.manual_seed(33)
        raw = torch.randn(n0 + n1, 2, 64, device="cuda", generator=g)
        amax = raw.abs().amax().clamp(min=1e-6)
        scale = amax / 448
        packed = (raw / scale).to(torch.float8_e4m3fn)
        ref = (packed.float() * scale).to(torch.bfloat16)
        cu = torch.tensor([0, n0, n0 + n1], device="cuda", dtype=torch.int32)
        bias = torch.randn(n1, n1, device="cuda") * 0.3
        with torch.cuda.stream(stream):
            got = _call(
                packed,
                packed,
                packed,
                {"q_descale": scale, "k_descale": scale, "v_descale": scale},
                causal=False,
                attn_mask=bias,
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                max_seqlen_q=n1,
                max_seqlen_kv=n1,
                stream=stream,
            )
        stream.synchronize()
        r0 = ref[:n0].unsqueeze(0)
        r1 = ref[n0:].unsqueeze(0)
        ref0 = _sdpa(r0, r0, r0, attn_mask=bias[:n0, :n0].view(1, 1, n0, n0))
        ref1 = _sdpa(r1, r1, r1, attn_mask=bias.view(1, 1, n1, n1))
        _check("fp8_e4m3/varlen-bias-0", got[:n0].unsqueeze(0), ref0.to(got.dtype), 0.97)
        _check("fp8_e4m3/varlen-bias-1", got[n0:].unsqueeze(0), ref1.to(got.dtype), 0.97)

    def default_stream_bf16():
        q, k, v, scales, rq, rk, rv = _qkv("bf16", 1, 32, 32, 2, 2, 64, 1)
        out = _call(q, k, v, scales, causal=False)
        torch.cuda.synchronize()
        _check("bf16/default-stream", out, _sdpa(rq, rk, rv), 0.99)

    def varlen_per_head_alibi_ragged():
        sqs, sks = [8, 20], [24, 12]
        h, d = 2, 64
        slopes = torch.tensor([0.15, 0.4], device="cuda")
        q_parts, k_parts, v_parts = [], [], []
        for i, (sq, sk) in enumerate(zip(sqs, sks)):
            qi, ki, vi, *_ = _qkv("bf16", 1, sq, sk, h, h, d, 50 + i)
            q_parts.append(qi[0])
            k_parts.append(ki[0])
            v_parts.append(vi[0])
        q, k, v = torch.cat(q_parts, 0), torch.cat(k_parts, 0), torch.cat(v_parts, 0)
        cu_q = torch.tensor([0, sqs[0], sum(sqs)], device="cuda", dtype=torch.int32)
        cu_k = torch.tensor([0, sks[0], sum(sks)], device="cuda", dtype=torch.int32)
        with torch.cuda.stream(stream):
            packed = _call(
                q,
                k,
                v,
                {},
                causal=False,
                alibi_slopes=slopes,
                cu_seqlens_q=cu_q,
                cu_seqlens_kv=cu_k,
                max_seqlen_q=max(sqs),
                max_seqlen_kv=max(sks),
                stream=stream,
            )
        stream.synchronize()
        q_off = k_off = 0
        for i, (sq, sk) in enumerate(zip(sqs, sks)):
            folded = fold_alibi_to_bias(slopes, sq, sk, q.device, stream=stream)
            stream.synchronize()
            ref = _sdpa(
                q[q_off : q_off + sq].unsqueeze(0),
                k[k_off : k_off + sk].unsqueeze(0),
                v[k_off : k_off + sk].unsqueeze(0),
                attn_mask=folded.view(1, h, sq, sk),
            )
            _check(f"bf16/varlen-per-head-alibi-{i}", packed[q_off : q_off + sq].unsqueeze(0), ref, 0.98)
            q_off += sq
            k_off += sk

    def _causal_br(sq: int, sk: int, device) -> torch.Tensor:
        rows = torch.arange(sq, device=device)[:, None]
        cols = torch.arange(sk, device=device)[None, :]
        keep = cols <= rows + (sk - sq)
        return torch.where(keep, torch.zeros((), device=device), torch.full((), float("-inf"), device=device))

    def _pack_ragged(kind: str, sqs: list[int], sks: list[int], h: int, d: int, seed: int):
        q_parts, k_parts, v_parts = [], [], []
        for i, (sq, sk) in enumerate(zip(sqs, sks)):
            qi, ki, vi, *_ = _qkv("bf16", 1, sq, sk, h, h, d, seed + i)
            q_parts.append(qi[0].float())
            k_parts.append(ki[0].float())
            v_parts.append(vi[0].float())
        qf, kf, vf = torch.cat(q_parts), torch.cat(k_parts), torch.cat(v_parts)
        if kind in _FLOAT:
            return (
                qf.to(_FLOAT[kind]),
                kf.to(_FLOAT[kind]),
                vf.to(_FLOAT[kind]),
                {},
                qf.to(_FLOAT[kind]),
                kf.to(_FLOAT[kind]),
                vf.to(_FLOAT[kind]),
            )
        q, qs = _quantize(kind, qf)
        k, ks = _quantize(kind, kf)
        v, vs = _quantize(kind, vf)
        scales = {"q_descale": qs, "k_descale": ks, "v_descale": vs}
        return (
            q,
            k,
            v,
            scales,
            (q.float() * qs).to(torch.bfloat16),
            (k.float() * ks).to(torch.bfloat16),
            (v.float() * vs).to(torch.bfloat16),
        )

    def varlen_mixed_per_head_alibi():
        """Ragged per-head mask plus per-head ALiBi. Each sequence uses its own lengths."""
        sqs, sks = [8, 20], [24, 12]
        h, d = 2, 64
        slopes = torch.tensor([0.15, 0.4], device="cuda")
        q, k, v, scales, rq, rk, rv = _pack_ragged("bf16", sqs, sks, h, d, 70)
        cu_q = torch.tensor([0, sqs[0], sum(sqs)], device="cuda", dtype=torch.int32)
        cu_k = torch.tensor([0, sks[0], sum(sks)], device="cuda", dtype=torch.int32)
        user3 = torch.randn(h, max(sqs), max(sks), device="cuda") * 0.1
        user2 = torch.randn(max(sqs), max(sks), device="cuda") * 0.1

        def _check_mix(name, bias, slope_arg, floor=0.98):
            with torch.cuda.stream(stream):
                packed = _call(
                    q,
                    k,
                    v,
                    scales,
                    causal=False,
                    bias=bias,
                    alibi_slopes=slope_arg,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_kv=cu_k,
                    max_seqlen_q=max(sqs),
                    max_seqlen_kv=max(sks),
                    stream=stream,
                )
            stream.synchronize()
            q_off = k_off = 0
            for i, (sq, sk) in enumerate(zip(sqs, sks)):
                folded = fold_alibi_to_bias(slope_arg, sq, sk, q.device, stream=stream)
                stream.synchronize()
                piece = bias[:, :sq, :sk] if bias.dim() == 3 else bias[:sq, :sk]
                mask = piece + folded
                if mask.dim() == 2:
                    mask = mask.view(1, 1, sq, sk)
                else:
                    mask = mask.view(1, h, sq, sk)
                ref = _sdpa(
                    rq[q_off : q_off + sq].unsqueeze(0),
                    rk[k_off : k_off + sk].unsqueeze(0),
                    rv[k_off : k_off + sk].unsqueeze(0),
                    attn_mask=mask,
                )
                _check(f"{name}-{i}", packed[q_off : q_off + sq].unsqueeze(0), ref, floor)
                q_off += sq
                k_off += sk

        _check_mix("bf16/varlen-mixed-3d", user3, slopes)
        _check_mix("bf16/varlen-mixed-3d-scalar", user3, 0.25)
        _check_mix("bf16/varlen-mixed-2d", user2, slopes)

        # Equal lengths stay on the folded bias. The sum must still match.
        qe, ke, ve, _, rqe, rke, rve = _pack_ragged("bf16", [16, 16], [16, 16], h, d, 90)
        cu = torch.tensor([0, 16, 32], device="cuda", dtype=torch.int32)
        ue = torch.randn(h, 16, 16, device="cuda") * 0.1
        with torch.cuda.stream(stream):
            packed = _call(
                qe,
                ke,
                ve,
                {},
                causal=False,
                bias=ue,
                alibi_slopes=slopes,
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                max_seqlen_q=16,
                max_seqlen_kv=16,
                stream=stream,
            )
        stream.synchronize()
        folded = fold_alibi_to_bias(slopes, 16, 16, qe.device, stream=stream)
        stream.synchronize()
        ref = _sdpa(
            rqe[:16].unsqueeze(0),
            rke[:16].unsqueeze(0),
            rve[:16].unsqueeze(0),
            attn_mask=(ue + folded).view(1, h, 16, 16),
        )
        _check("bf16/varlen-mixed-equal", packed[:16].unsqueeze(0), ref, 0.98)

        for kind in ("fp8_e4m3", "int8"):
            q, k, v, scales, rq, rk, rv = _pack_ragged(kind, sqs, sks, h, d, 110)
            with torch.cuda.stream(stream):
                packed = _call(
                    q,
                    k,
                    v,
                    scales,
                    causal=False,
                    bias=user3,
                    alibi_slopes=slopes,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_kv=cu_k,
                    max_seqlen_q=max(sqs),
                    max_seqlen_kv=max(sks),
                    stream=stream,
                )
            stream.synchronize()
            q_off = k_off = 0
            for i, (sq, sk) in enumerate(zip(sqs, sks)):
                folded = fold_alibi_to_bias(slopes, sq, sk, q.device, stream=stream)
                stream.synchronize()
                mask = (user3[:, :sq, :sk] + folded).view(1, h, sq, sk)
                ref = _sdpa(
                    rq[q_off : q_off + sq].unsqueeze(0),
                    rk[k_off : k_off + sk].unsqueeze(0),
                    rv[k_off : k_off + sk].unsqueeze(0),
                    attn_mask=mask,
                )
                _check(
                    f"{kind}/varlen-mixed-3d-{i}", packed[q_off : q_off + sq].unsqueeze(0), ref.to(packed.dtype), 0.97
                )
                q_off += sq
                k_off += sk

    def quant_paged_ragged(kind: str = "fp8_e4m3"):
        """Unequal seqlen_k on quant paged attention. No cu_seqlens on the call."""
        b, sq, h, d = 2, 16, 2, 64
        sks = [32, 16]
        page = 16
        q_parts, k_parts, v_parts = [], [], []
        for i, sk in enumerate(sks):
            qi, ki, vi, *_ = _qkv("bf16", 1, sq, sk, h, h, d, 130 + i)
            q_parts.append(qi)
            k_parts.append(ki[0])
            v_parts.append(vi[0])
        qf = torch.cat(q_parts, 0)
        if kind in _FLOAT:
            q, scales = qf.to(_FLOAT[kind]), {}
            ks = [p.to(_FLOAT[kind]) for p in k_parts]
            vs = [p.to(_FLOAT[kind]) for p in v_parts]
            rq = q
            rks, rvs = ks, vs
        else:
            # Descales must be float32. Quantizing the bf16 tensor leaves a bf16 scale.
            q, qs = _quantize(kind, qf.float())
            qs = qs.detach().to(dtype=torch.float32).reshape(1).contiguous()
            scales = {"q_descale": qs, "k_descale": qs, "v_descale": qs}
            ks, vs, rks, rvs = [], [], [], []
            for p in k_parts:
                # One descale for the whole launch. Re-quantize K with Q's scale.
                stored = (p.float() / qs).to(q.dtype)
                ks.append(stored)
                rks.append((stored.float() * qs).to(torch.bfloat16))
            for p in v_parts:
                stored = (p.float() / qs).to(q.dtype)
                vs.append(stored)
                rvs.append((stored.float() * qs).to(torch.bfloat16))
            rq = (q.float() * qs).to(torch.bfloat16)
        n_pages = 4
        cache_k = torch.zeros(n_pages, page, h, d, device="cuda", dtype=q.dtype)
        cache_v = torch.zeros(n_pages, page, h, d, device="cuda", dtype=q.dtype)
        cache_k[0] = ks[0][:16]
        cache_k[1] = ks[0][16:]
        cache_k[2] = ks[1][:16]
        cache_v[0] = vs[0][:16]
        cache_v[1] = vs[0][16:]
        cache_v[2] = vs[1][:16]
        bt = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)
        seqlen = torch.tensor(sks, device="cuda", dtype=torch.int32)
        with torch.cuda.stream(stream):
            ali = _call(
                q,
                cache_k,
                cache_v,
                scales,
                causal=False,
                alibi_slopes=0.25,
                block_table=bt,
                seqlen_k=seqlen,
                stream=stream,
            )
            cau = _call(q, cache_k, cache_v, scales, causal=True, block_table=bt, seqlen_k=seqlen, stream=stream)
        stream.synchronize()
        if ali.shape != (b, sq, h, d) or ali.dtype != (q.dtype if kind in _FLOAT else torch.bfloat16):
            raise AssertionError(f"{kind}/paged-ragged shape {tuple(ali.shape)} {ali.dtype}")
        for i, sk in enumerate(sks):
            folded = fold_alibi_to_bias(0.25, sq, sk, q.device, stream=stream)
            stream.synchronize()
            ref_a = _sdpa(rq[i : i + 1], rks[i].unsqueeze(0), rvs[i].unsqueeze(0), attn_mask=folded)
            ref_c = _sdpa(
                rq[i : i + 1], rks[i].unsqueeze(0), rvs[i].unsqueeze(0), attn_mask=_causal_br(sq, sk, q.device)
            )
            _check(f"{kind}/paged-ragged-alibi-{i}", ali[i : i + 1], ref_a.to(ali.dtype), 0.97)
            _check(f"{kind}/paged-ragged-causal-{i}", cau[i : i + 1], ref_c.to(cau.dtype), 0.97)

    _run("bf16/varlen-mixed-alibi", varlen_mixed_per_head_alibi)
    _run("fp8_e4m3/paged-ragged", lambda: quant_paged_ragged("fp8_e4m3"))
    _run("int8/paged-ragged", lambda: quant_paged_ragged("int8"))
    _run("bf16/default-stream", default_stream_bf16)
    _run("bf16/varlen-per-head-alibi", varlen_per_head_alibi_ragged)
    _run("bf16/paged-splitk", paged_split_bf16)
    _run("fp8_e4m3/paged-splitk", quant_paged_split)
    _run("int8/paged-splitk", lambda: quant_paged_split("int8"))
    _run("bf16/varlen-per-head", varlen_per_head_bias)
    _run("bf16/varlen-paged", varlen_paged_split_and_bias)
    _run("bf16/dense-per-head", dense_per_head_and_cross)
    _run("bf16/odd-d-split", odd_d_split)
    _run("fp8/table-and-varlen-bias", int64_table_and_fp8_varlen_bias)
