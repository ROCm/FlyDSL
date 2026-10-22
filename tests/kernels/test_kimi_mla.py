# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests and benchmark for the Kimi-K3 MLA decode/prefill kernels (gfx950).

Usage:
    PYTHONPATH=./ pytest tests/kernels/test_kimi_mla.py
    RUN_TESTS_FULL=1 PYTHONPATH=./ pytest tests/kernels/test_kimi_mla.py -m large_shape

    # Benchmark (CUDA-graph timings vs AITER ASM and the 128-head FlyDSL MLA decode)
    PYTHONPATH=./ python tests/kernels/test_kimi_mla.py --mode decode --lengths 8200 65544 262144
    PYTHONPATH=./ python tests/kernels/test_kimi_mla.py --mode prefill --lengths 2048 8192 65536
"""

import argparse
import functools
import json
import statistics
import sys

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_ARCH = str(get_rocm_arch() or "")
if _ARCH != "gfx950":
    pytest.skip(f"Kimi MLA kernels require gfx950, got {_ARCH}", allow_module_level=True)

from kernels.attention.kimi_mla import (  # noqa: E402
    DECODE_CP_WORLD,
    DECODE_HEADS,
    DECODE_QK_DIM,
    DECODE_QLEN,
    DECODE_V_DIM,
    PREFILL_HEADS,
    PREFILL_QK_DIM,
    PREFILL_V_DIM,
    SOFTMAX_SCALE,
    KimiMLADecodePlan,
    KimiMLAPrefillPlan,
)

FP8 = torch.float8_e4m3fn

# Rank-local KV slots shared by all decode tests: one compile per rank, and
# enough capacity for a 262144-token global sequence under DCP8.
DECODE_CAPACITY = 32768

# Decode O is compared with the FP32 reference of the FP8 inputs. The FP8
# probabilities dominate the error (about 2% normalized L2), so short contexts,
# where each output averages few V rows, use L2 metrics; the elementwise
# tolerance applies from 8192 tokens.
DECODE_NORMALIZED_L2 = 0.04
DECODE_ROW_L2 = 0.05
DECODE_ELEMENTWISE_MIN_LEN = 8192
DECODE_O_TOL = dict(atol=3e-3, rtol=3e-2)
DECODE_LSE_TOL = dict(atol=2e-4, rtol=2e-5)
# Prefill: FP8 P makes elementwise relative error meaningless near zero.
PREFILL_NORMALIZED_L2 = 0.04
PREFILL_ROW_L2 = 0.08
PREFILL_LSE_TOL = dict(atol=1e-3, rtol=1e-4)


# -- Decode reference and inputs ----------------------------------


def _local_len(length, rank):
    return max(0, (length + DECODE_CP_WORLD - 1 - rank) // DECODE_CP_WORLD)


def _visible(length, rank, t):
    """Rank-local tokens visible to verify token ``t`` under DCP round robin."""
    return max(0, (length - DECODE_QLEN + t + DECODE_CP_WORLD - rank) // DECODE_CP_WORLD)


def decode_reference(q, kv, indices, seq_len, qs, ks, length, rank):
    """FP32 reference for ``make_decode_case`` inputs; ``length`` mirrors the device ``seq_len``."""
    local = _local_len(length, rank)
    logical = kv.float()[indices[:local].long()]
    score = (q.float().reshape(-1, DECODE_QK_DIM) * qs) @ (logical * ks).T * SOFTMAX_SCALE
    limit = torch.tensor([_visible(length, rank, t) for t in range(DECODE_QLEN)], device=q.device)
    limit = limit.repeat_interleave(DECODE_HEADS)
    score.masked_fill_(torch.arange(local, device=q.device)[None, :] >= limit[:, None], -float("inf"))
    lse = score.logsumexp(-1).reshape(DECODE_QLEN, DECODE_HEADS)
    prob = score.softmax(-1).nan_to_num()
    out = (prob @ (logical[:, :DECODE_V_DIM] * ks)).reshape(DECODE_QLEN, DECODE_HEADS, DECODE_V_DIM)
    return out, lse


def make_decode_case(length, rank, *, seed=1234, layout="paged", capacity=DECODE_CAPACITY, magnitude=0.5):
    """Random FP8 decode inputs; ``capacity`` logical tokens scattered over as many slots."""
    gen = torch.Generator(device="cuda").manual_seed(seed + length)
    q = (torch.randn((DECODE_QLEN, DECODE_HEADS, DECODE_QK_DIM), device="cuda", generator=gen) * magnitude).to(FP8)
    logical = (torch.randn((capacity, DECODE_QK_DIM), device="cuda", generator=gen) * magnitude).to(FP8)
    if layout == "paged":
        pages = torch.randperm(capacity // 64, device="cuda", generator=gen)
        indices = (pages[:, None] * 64 + torch.arange(64, device="cuda")).flatten().int()
    elif layout == "random":
        indices = torch.randperm(capacity, device="cuda", generator=gen).int()
    else:
        indices = torch.arange(capacity, device="cuda", dtype=torch.int32)
    kv = torch.empty_like(logical)
    kv.view(torch.uint8)[indices.long()] = logical.view(torch.uint8)
    seq_len = torch.tensor([length], dtype=torch.int32, device="cuda")
    q_scale = torch.tensor([0.75], device="cuda")
    kv_scale = torch.tensor([1.2], device="cuda")
    return q, kv, indices, seq_len, q_scale, kv_scale


def _decode_plan(rank, capacity=DECODE_CAPACITY, **kwargs):
    return KimiMLADecodePlan(max_local_tokens=capacity, cache_slots=capacity, cp_rank=rank, **kwargs)


def _assert_decode_close(out, lse, ref, ref_lse, length):
    out = out.float()
    assert torch.isfinite(out).all()
    live = ref.norm(dim=-1) > 0
    # Rows with no visible token must be exactly zero (and LSE -inf, checked below).
    assert (out[~live] == 0).all()
    error = out - ref
    normalized_l2 = (error.norm() / ref.norm()).item() if live.any() else 0.0
    row_l2 = (error.norm(dim=-1)[live] / ref.norm(dim=-1)[live]).max().item() if live.any() else 0.0
    assert normalized_l2 < DECODE_NORMALIZED_L2, normalized_l2
    assert row_l2 < DECODE_ROW_L2, row_l2
    if length >= DECODE_ELEMENTWISE_MIN_LEN:
        torch.testing.assert_close(out, ref, **DECODE_O_TOL)
    if lse is not None:
        torch.testing.assert_close(lse, ref_lse, **DECODE_LSE_TOL)


# -- Decode tests --------------------------------------------------

# Lengths include the 8 verify tokens. They cover rows with no visible token
# (8 on high ranks), partial and single tiles, the 16/32/42-split merge
# thresholds (8192/16384 + rank) and long contexts.
DECODE_LENGTHS = [8, 9, 72, 520, 4104, 8192, 8193, 8200, 16391, 16392, 65544]


@pytest.mark.parametrize("rank", [0, 3, 7])
@pytest.mark.parametrize("length", DECODE_LENGTHS)
def test_kimi_mla_decode(length, rank):
    case = make_decode_case(length, rank)
    # NaN in slots past the rank-local length must not leak into the output.
    kv, indices = case[1], case[2]
    kv.view(torch.uint8)[indices[_local_len(length, rank) :].long()] = 0xFF
    ref, ref_lse = decode_reference(*case, length, rank)
    out, lse = _decode_plan(rank)(*case)
    torch.cuda.synchronize()
    _assert_decode_close(out, lse, ref, ref_lse, length)


@pytest.mark.large_shape
@pytest.mark.parametrize("layout", ["paged", "random"])
@pytest.mark.parametrize("rank", [0, 7])
def test_kimi_mla_decode_long(layout, rank):
    length = DECODE_CAPACITY * DECODE_CP_WORLD - rank
    case = make_decode_case(length, rank, layout=layout)
    ref, ref_lse = decode_reference(*case, length, rank)
    out, lse = _decode_plan(rank)(*case)
    torch.cuda.synchronize()
    _assert_decode_close(out, lse, ref, ref_lse, length)


def test_kimi_mla_decode_graph_dynamic_length():
    """One captured graph replays correctly as the device-side length changes."""
    rank = 3
    q, kv, indices, seq_len, q_scale, kv_scale = make_decode_case(8200, rank, layout="random")
    plan = _decode_plan(rank)
    plan(q, kv, indices, seq_len, q_scale, kv_scale)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = plan(q, kv, indices, seq_len, q_scale, kv_scale)
    for length in [8200, 65544, 9, 16392, 4104]:
        seq_len.fill_(length)
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        ref, ref_lse = decode_reference(q, kv, indices, seq_len, q_scale, kv_scale, length, rank)
        _assert_decode_close(out, lse, ref, ref_lse, length)


def test_kimi_mla_decode_dcp8_merge():
    """Merging the eight rank outputs by LSE reproduces unsharded attention."""
    length = 20000
    gen = torch.Generator(device="cuda").manual_seed(7)
    q = (torch.randn((DECODE_QLEN, DECODE_HEADS, DECODE_QK_DIM), device="cuda", generator=gen) * 0.5).to(FP8)
    tokens = (torch.randn((length, DECODE_QK_DIM), device="cuda", generator=gen) * 0.5).to(FP8)
    q_scale = torch.tensor(0.9, device="cuda")  # zero-dimensional scales, as passed by vLLM
    kv_scale = torch.tensor(1.1, device="cuda")
    seq_len = torch.tensor([length], dtype=torch.int32, device="cuda")
    capacity = (length + DECODE_CP_WORLD - 1) // DECODE_CP_WORLD
    outs, lses = [], []
    for rank in range(DECODE_CP_WORLD):
        local = tokens[rank::DECODE_CP_WORLD]
        indices = torch.randperm(capacity, device="cuda", generator=gen).int()
        kv = torch.zeros((capacity, DECODE_QK_DIM), dtype=FP8, device="cuda")
        kv.view(torch.uint8)[indices[: local.shape[0]].long()] = local.view(torch.uint8)
        plan = KimiMLADecodePlan(max_local_tokens=capacity, cache_slots=capacity, cp_rank=rank)
        out, lse = plan(q, kv, indices, seq_len, q_scale, kv_scale)
        outs.append(out.float().clone())
        lses.append(lse.clone())
    lse_all = torch.stack(lses)
    total = lse_all.logsumexp(0)
    merged = (torch.stack(outs) * (lse_all - total).exp()[..., None]).sum(0)

    score = (q.float().reshape(-1, DECODE_QK_DIM) * q_scale) @ (tokens.float() * kv_scale).T * SOFTMAX_SCALE
    limit = torch.arange(length - DECODE_QLEN, length, device="cuda").repeat_interleave(DECODE_HEADS)
    score.masked_fill_(torch.arange(length, device="cuda")[None, :] > limit[:, None], -float("inf"))
    ref_lse = score.logsumexp(-1).reshape(DECODE_QLEN, DECODE_HEADS)
    ref = (score.softmax(-1) @ (tokens[:, :DECODE_V_DIM].float() * kv_scale)).reshape_as(merged)
    _assert_decode_close(merged, total, ref, ref_lse, length)


def test_kimi_mla_decode_without_lse():
    length, rank = 8200, 5
    case = make_decode_case(length, rank)
    ref, _ = decode_reference(*case, length, rank)
    output = torch.empty((DECODE_QLEN, DECODE_HEADS, DECODE_V_DIM), dtype=torch.bfloat16, device="cuda")
    out, lse = _decode_plan(rank, return_lse=False)(*case, output=output)
    torch.cuda.synchronize()
    assert lse is None and out is output
    _assert_decode_close(out, None, ref, None, length)


def test_kimi_mla_decode_rejects_invalid_inputs():
    plan = _decode_plan(0)
    q, kv, indices, seq_len, q_scale, kv_scale = make_decode_case(72, 0, capacity=DECODE_CAPACITY)
    good = dict(q=q, kv=kv, indices=indices, global_seq_len=seq_len, q_scale=q_scale, kv_scale=kv_scale)
    bad_cases = [
        ("q", q[:, :64].contiguous(), ValueError),
        ("q", q.to(torch.bfloat16), ValueError),
        ("kv", kv[:-1], ValueError),
        ("kv", kv.view(torch.uint8), ValueError),
        ("indices", indices.long(), ValueError),
        ("indices", torch.stack([indices, indices], -1)[:, 0], ValueError),
        ("global_seq_len", seq_len.long(), ValueError),
        ("q_scale", torch.ones(2, device="cuda"), ValueError),
        ("kv_scale", 1.0, TypeError),
    ]
    for name, value, error in bad_cases:
        with pytest.raises(error):
            plan(**{**good, name: value})
    with pytest.raises(ValueError):
        plan(**good, lse=torch.empty((DECODE_QLEN, DECODE_HEADS + 1), device="cuda"))
    with pytest.raises(ValueError):
        _decode_plan(0, return_lse=False)(**good, lse=torch.empty((DECODE_QLEN, DECODE_HEADS), device="cuda"))
    with pytest.raises(ValueError):
        _decode_plan(DECODE_CP_WORLD)
    with pytest.raises(ValueError):
        KimiMLADecodePlan(max_local_tokens=0, cache_slots=64, cp_rank=0)
    with pytest.raises(ValueError):
        _decode_plan(0, workspace=(torch.empty(1, device="cuda"), torch.empty(1, device="cuda")))


# -- Prefill reference and tests ------------------------------------


def make_prefill_case(s, *, seed=1234, magnitude=0.5):
    gen = torch.Generator(device="cuda").manual_seed(seed + s)
    shape_qk = (s, PREFILL_HEADS, PREFILL_QK_DIM)
    q = torch.randn(shape_qk, device="cuda", generator=gen, dtype=torch.bfloat16) * magnitude
    k = torch.randn(shape_qk, device="cuda", generator=gen, dtype=torch.bfloat16) * magnitude
    v = torch.randn((s, PREFILL_HEADS, PREFILL_V_DIM), device="cuda", generator=gen, dtype=torch.bfloat16)
    return q, k, v


def _prefill_rows(s, full):
    if full:
        return torch.arange(s, device="cuda")
    rows = {0, 1, 31, 32, 63, 64, 127, 128, 255, 256, s // 4, s // 2, s - 129, s - 128, s - 2, s - 1}
    return torch.tensor(sorted(r for r in rows if 0 <= r < s), device="cuda")


def prefill_reference(q, k, v, rows):
    """FP32 causal attention of FP8-rounded inputs (the kernel's precision path) for ``rows``."""
    q, k, v = [t.to(FP8).float() for t in (q, k, v)]
    outs, lses = [], []
    for part in rows.split(128):
        scores = q[part].transpose(0, 1) @ k.permute(1, 2, 0) * SOFTMAX_SCALE
        scores.masked_fill_(torch.arange(k.shape[0], device="cuda")[None, None, :] > part[None, :, None], -torch.inf)
        outs.append((scores.softmax(-1) @ v.transpose(0, 1)).transpose(0, 1))
        lses.append(scores.logsumexp(-1).T)
    return torch.cat(outs), torch.cat(lses)


def _assert_prefill_close(out, lse, q, k, v, rows):
    ref, ref_lse = prefill_reference(q, k, v, rows)
    got = out[rows].float()
    assert torch.isfinite(got).all() and torch.isfinite(lse[rows]).all()
    error = got - ref
    normalized_l2 = (error.norm() / ref.norm()).item()
    row_l2 = (error.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-6)).max().item()
    assert normalized_l2 < PREFILL_NORMALIZED_L2, normalized_l2
    assert row_l2 < PREFILL_ROW_L2, row_l2
    torch.testing.assert_close(lse[rows], ref_lse, **PREFILL_LSE_TOL)


# S <= 512 with BF16 inputs converts inside attention; longer BF16 calls use the fused conversion launch.
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, FP8], ids=["bf16", "fp8"])
@pytest.mark.parametrize("s", [128, 512, 1024, 2048])
def test_kimi_mla_prefill(s, input_dtype):
    q, k, v = make_prefill_case(s, magnitude=1.5)
    plan = KimiMLAPrefillPlan(seq_len=s, input_dtype=input_dtype)
    out, lse = plan(*[t.to(input_dtype) for t in (q, k, v)])
    torch.cuda.synchronize()
    _assert_prefill_close(out, lse, q, k, v, _prefill_rows(s, full=True))


# 8320 / 32896 end in a partial 256-row query tile on the 4-wave / 8-wave launch paths;
# S >= 32768 also enables the wave-ballot conditional rescale.
@pytest.mark.large_shape
@pytest.mark.parametrize("s", [8320, 32896])
def test_kimi_mla_prefill_long(s):
    q, k, v = make_prefill_case(s, magnitude=2.0)
    out, lse = KimiMLAPrefillPlan(seq_len=s)(q, k, v)
    torch.cuda.synchronize()
    _assert_prefill_close(out, lse, q, k, v, _prefill_rows(s, full=False))


def test_kimi_mla_prefill_graph_and_shared_workspace():
    s = 1024
    q, k, v = make_prefill_case(s)
    first = KimiMLAPrefillPlan(seq_len=s)
    second = KimiMLAPrefillPlan(seq_len=s, workspace=first.workspace)
    assert all(a is b for a, b in zip(first.workspace, second.workspace))
    second(q, k, v)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = second(q, k, v)
    q2, k2, v2 = make_prefill_case(s, seed=99)
    for t, new in zip((q, k, v), (q2, k2, v2)):
        t.copy_(new)
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_prefill_close(out, lse, q2, k2, v2, _prefill_rows(s, full=True))


def test_kimi_mla_prefill_rejects_invalid_inputs():
    for seq_len in (0, 100, 200, 262144 + 128, 1024.0):
        with pytest.raises(ValueError):
            KimiMLAPrefillPlan(seq_len=seq_len)
    with pytest.raises(ValueError):
        KimiMLAPrefillPlan(seq_len=128, input_dtype=torch.float16)
    with pytest.raises(ValueError):
        KimiMLAPrefillPlan(seq_len=1024, input_dtype=FP8, workspace=(None, None, None))
    plan = KimiMLAPrefillPlan(seq_len=128)
    q, k, v = make_prefill_case(128)
    with pytest.raises(ValueError):
        plan(q.to(FP8), k, v)
    with pytest.raises(ValueError):
        plan(q[:, :8].contiguous(), k, v)
    with pytest.raises(ValueError):
        plan(q, k, v.transpose(0, 1).contiguous().transpose(0, 1))
    with pytest.raises(ValueError):
        plan(q, k, v, lse=torch.empty((128, PREFILL_HEADS), device="cuda", dtype=torch.bfloat16))


# -- Benchmark -------------------------------------------------------


def _capture(fn, outputs, calls):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    for t in outputs:
        t.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    for t in outputs:
        assert torch.isfinite(t).all(), "graph must write every output element"
    return graph


def _time_graph(graph, calls, replays):
    graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (calls * replays)


def _bench(graphs, samples, calls, replays):
    """Rotate and reverse implementation order per sample; report medians in us."""
    names = list(graphs)
    times = {name: [] for name in names}
    for i in range(samples):
        order = names[i % len(names) :] + names[: i % len(names)]
        if i % 2:
            order.reverse()
        for name in order:
            times[name].append(_time_graph(graphs[name], calls, replays))
    return {name: statistics.median(us) for name, us in times.items()}


def _aiter_decode_asm(case, length, rank, splits):
    """AITER ASM MLA decode with CP round robin; Q padded from 96 to 128 heads."""
    import aiter
    from aiter.mla import mla_decode_fwd

    q, kv, indices, _, q_scale, kv_scale = case
    padded = torch.empty((DECODE_QLEN, 128, DECODE_QK_DIM), dtype=q.dtype, device="cuda")
    padded.view(torch.uint8)[:, :DECODE_HEADS] = q.view(torch.uint8)
    padded.view(torch.uint8)[:, DECODE_HEADS:] = q.view(torch.uint8)[:, : 128 - DECODE_HEADS]
    local = _local_len(length, rank)
    qi = torch.tensor([0, DECODE_QLEN], dtype=torch.int32, device="cuda")
    ki = torch.tensor([0, local], dtype=torch.int32, device="cuda")
    gi = torch.tensor([0, length], dtype=torch.int32, device="cuda")
    last = torch.ones(1, dtype=torch.int32, device="cuda")
    info = aiter.get_mla_metadata_info_v1(
        1, DECODE_QLEN, 128, q.dtype, kv.dtype, is_sparse=False, fast_mode=True, num_kv_splits=splits,
        intra_batch_mode=False,
    )  # fmt: skip
    wm, wi, ws, ri, rf, rp = [torch.empty(size, dtype=dtype, device="cuda") for size, dtype in info]
    aiter.get_mla_metadata_v1(
        qi, ki, last, 128, 1, False, wm, ws, wi, ri, rf, rp, page_size=1, kv_granularity=16,
        max_seqlen_qo=DECODE_QLEN, uni_seqlen_qo=DECODE_QLEN, fast_mode=True, max_split_per_batch=splits,
        intra_batch_mode=False, dtype_q_nope=q.dtype, dtype_kv_nope=kv.dtype, is_cp_round_robin=True,
    )  # fmt: skip
    out = torch.zeros((DECODE_QLEN, 128, DECODE_V_DIM), device="cuda", dtype=torch.bfloat16)

    def run():
        return mla_decode_fwd(
            padded, kv.view(-1, 1, 1, DECODE_QK_DIM), out, qi, ki, indices[:local], last,
            max_seqlen_q=DECODE_QLEN, sm_scale=SOFTMAX_SCALE, num_kv_splits=splits, q_scale=q_scale,
            kv_scale=kv_scale, work_meta_data=wm, work_indptr=wi, work_info_set=ws, reduce_indptr=ri,
            reduce_final_map=rf, reduce_partial_map=rp, g_kv_indptr=gi, cp_world_size=DECODE_CP_WORLD,
            cp_rank=rank, return_lse=True,
        )  # fmt: skip

    _, lse = run()
    torch.cuda.synchronize()
    return run, out, out[:, :DECODE_HEADS], lse.view(DECODE_QLEN, 128)[:, :DECODE_HEADS]


def _flydsl_decode_128(case, length, rank):
    """Existing FlyDSL 128-head MLA decode, expressing the verify block as 8 one-token sequences.

    Each verify token becomes its own sequence over its causally visible local
    tokens, Q is padded to 128 heads and the scales are folded into sm_scale.
    The kernel has no V scale, so the returned view needs a kv_scale multiply.
    """
    from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1, mla_reduce_v1

    from kernels.attention.mla_fwd_decode import flydsl_mla_fwd_decode

    q, kv, indices, _, q_scale, kv_scale = case
    padded = torch.empty((DECODE_QLEN, 128, DECODE_QK_DIM), dtype=q.dtype, device="cuda")
    padded.view(torch.uint8)[:, :DECODE_HEADS] = q.view(torch.uint8)
    padded.view(torch.uint8)[:, DECODE_HEADS:] = q.view(torch.uint8)[:, : 128 - DECODE_HEADS]
    lens = [_visible(length, rank, t) for t in range(DECODE_QLEN)]
    if min(lens) < 1:
        raise ValueError("every verify token needs a visible token")
    kv_indptr = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    kv_indices = torch.cat([indices[:n] for n in lens])
    qo_indptr = torch.arange(DECODE_QLEN + 1, dtype=torch.int32, device="cuda")
    last = torch.ones(DECODE_QLEN, dtype=torch.int32, device="cuda")
    cu_num = torch.cuda.get_device_properties(0).multi_processor_count
    splits = min((cu_num + DECODE_QLEN - 1) // DECODE_QLEN, 32)
    info = get_mla_metadata_info_v1(
        DECODE_QLEN, 1, 128, FP8, FP8, is_sparse=False, fast_mode=True, num_kv_splits=splits, intra_batch_mode=False
    )
    wm, wi, ws, ri, rf, rp = [torch.empty(size, dtype=dtype, device="cuda") for size, dtype in info]
    get_mla_metadata_v1(
        qo_indptr, kv_indptr, last, 128, 1, False, wm, ws, wi, ri, rf, rp, kv_granularity=16, max_seqlen_qo=1,
        uni_seqlen_qo=1, fast_mode=True, max_split_per_batch=splits, intra_batch_mode=False, dtype_q=FP8,
        dtype_kv=FP8,
    )  # fmt: skip
    out = torch.empty((DECODE_QLEN, 128, DECODE_V_DIM), dtype=torch.bfloat16, device="cuda")
    logits = torch.empty((rp.size(0), 1, 128, DECODE_V_DIM), dtype=torch.float32, device="cuda")
    attn_lse = torch.empty((rp.size(0), 1, 128, 1), dtype=torch.float32, device="cuda")
    sm_scale = SOFTMAX_SCALE * q_scale.item() * kv_scale.item()
    kv_paged = kv.view(-1, 1, 1, DECODE_QK_DIM)

    def run():
        flydsl_mla_fwd_decode(padded, kv_paged, kv_indices, wi, ws, out, logits, attn_lse, sm_scale)
        mla_reduce_v1(logits, attn_lse, ri, rf, rp, 1, splits, out, None)

    run()
    torch.cuda.synchronize()
    return run, out, out[:, :DECODE_HEADS], None


def _normalized_l2(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


def bench_decode(args):
    """Each baseline is captured on its own and timed alternately with the Kimi kernel.

    Holding several AITER decode graphs at once faults on replay with the tested
    AITER build, so baseline graphs are released before the next one is captured.
    """
    results = []
    for length in args.lengths:
        case = make_decode_case(length, args.rank, seed=args.seed, layout=args.layout)
        ref, ref_lse = decode_reference(*case, length, args.rank)
        plan = _decode_plan(args.rank)
        out, lse = plan.output, plan.lse
        candidate = functools.partial(plan, *case)
        candidate()
        torch.cuda.synchronize()
        _assert_decode_close(out, lse, ref, ref_lse, length)
        kimi_graph = _capture(candidate, (out, lse), args.calls)
        errors = {"flydsl_kimi": _normalized_l2(out, ref)}

        baselines = {}
        if "flydsl128" in args.baselines:
            baselines["flydsl_128h"] = lambda: _flydsl_decode_128(case, length, args.rank)
        if "aiter" in args.baselines:
            for splits in args.asm_splits:
                baselines[f"asm_{splits}"] = functools.partial(_aiter_decode_asm, case, length, args.rank, splits)

        kimi_us, medians = [], {}
        for name, build in baselines.items():
            run, full, view, base_lse = build()
            scale = case[5].item() if name == "flydsl_128h" else 1.0
            errors[name] = _normalized_l2(view.float() * scale, ref)
            if base_lse is not None:
                _assert_decode_close(view, base_lse, ref, ref_lse, length)
            graph = _capture(run, (full,), args.calls)
            pair = _bench({"flydsl_kimi": kimi_graph, name: graph}, args.samples, args.calls, args.replays)
            kimi_us.append(pair["flydsl_kimi"])
            medians[name] = pair[name]
            del graph, run, full, view, base_lse
            torch.cuda.synchronize()
        if not kimi_us:
            kimi_us.append(_bench({"flydsl_kimi": kimi_graph}, args.samples, args.calls, args.replays)["flydsl_kimi"])
        medians = {"flydsl_kimi": statistics.median(kimi_us), **medians}
        record = dict(mode="decode", length=length, rank=args.rank, median_us=medians, normalized_l2=errors)
        asm = [us for name, us in medians.items() if name.startswith("asm")]
        if asm:
            record["speedup_vs_best_asm"] = min(asm) / medians["flydsl_kimi"]
        if "flydsl_128h" in medians:
            record["speedup_vs_flydsl_128h"] = medians["flydsl_128h"] / medians["flydsl_kimi"]
        print("RESULT", json.dumps(record), flush=True)
        results.append(record)
        del kimi_graph
    return results


def _aiter_prefill_asm(q8, k8, v8):
    """AITER FP8 ASM prefill on inputs already padded from 12 to 16 heads, plus its reduction."""
    import aiter

    s, h, d = q8.shape
    cpu_qi = torch.tensor([0, s], dtype=torch.int32)
    cpu_seq = torch.tensor([s], dtype=torch.int32)
    qi = cpu_qi.cuda()
    idx = torch.arange(s, dtype=torch.int32, device="cuda")
    info = aiter.get_ps_metadata_info_v1(batch_size=1, num_head_k=h, max_qlen=s, qlen_granularity=256)
    wm, wi, ws, ri, rf, rp = [torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in info]
    aiter.get_ps_metadata_v1(
        cpu_qi, cpu_qi, cpu_seq, 1, h, wm, wi, ws, ri, rf, rp, qhead_granularity=1, qlen_granularity=256,
        kvlen_granularity=128, block_size=1, is_causal=True, need_lse=True,
    )  # fmt: skip
    partial = torch.empty((rp.numel() * 256, h, PREFILL_V_DIM), device="cuda", dtype=torch.float32)
    stats = torch.empty((rp.numel() * 256, h), device="cuda", dtype=torch.float32)
    out = torch.empty((s, h, PREFILL_V_DIM), device="cuda", dtype=torch.bfloat16)
    lse = torch.empty((s, h), device="cuda", dtype=torch.float32)
    one = torch.ones(1, device="cuda", dtype=torch.float32)

    def run():
        aiter.mla_prefill_ps_asm_fwd(
            q8, k8, v8, qi, qi, idx, wi, ws, s, d ** (-0.5), True, partial, stats, out, one, one, one
        )
        aiter.mla_reduce_v1(partial, stats, ri, rf, rp, 256, 0, out, lse)

    return run, out, lse


def bench_prefill(args):
    results = []
    for s in args.lengths:
        q, k, v = make_prefill_case(s, seed=args.seed)
        rows = _prefill_rows(s, full=False)
        calls = max(1, min(20, 131072 // s))
        q8, k8, v8 = [t.to(FP8) for t in (q, k, v)]
        bf16_plan = KimiMLAPrefillPlan(seq_len=s)
        fp8_plan = KimiMLAPrefillPlan(seq_len=s, input_dtype=FP8)
        funcs = {
            "flydsl_kimi_bf16": (functools.partial(bf16_plan, q, k, v), bf16_plan.output, bf16_plan.lse),
            "flydsl_kimi_fp8": (functools.partial(fp8_plan, q8, k8, v8), fp8_plan.output, fp8_plan.lse),
        }
        if "aiter" in args.baselines:
            pad = 16 - PREFILL_HEADS
            q16, k16, v16 = [torch.cat((t, t[:, :pad]), dim=1).contiguous() for t in (q8, k8, v8)]
            run, out, lse = _aiter_prefill_asm(q16, k16, v16)
            funcs["asm_fp8"] = (run, out, lse)
        graphs = {}
        for name, (fn, out, lse) in funcs.items():
            fn()
            torch.cuda.synchronize()
            _assert_prefill_close(out[:, :PREFILL_HEADS], lse[:, :PREFILL_HEADS], q, k, v, rows)
            graphs[name] = _capture(fn, (out, lse), calls)
        medians = _bench(graphs, args.samples, calls, args.replays)
        record = dict(mode="prefill", seq_len=s, median_us=medians)
        if "asm_fp8" in medians:
            record["fp8_speedup_vs_asm"] = medians["asm_fp8"] / medians["flydsl_kimi_fp8"]
        print("RESULT", json.dumps(record), flush=True)
        results.append(record)
        del graphs, funcs
    return results


def main():
    parser = argparse.ArgumentParser(description="Kimi-K3 MLA decode/prefill benchmark")
    parser.add_argument("--mode", choices=["decode", "prefill"], default="decode")
    parser.add_argument("--lengths", type=int, nargs="+", default=None)
    parser.add_argument("--rank", type=int, default=0, choices=range(DECODE_CP_WORLD))
    parser.add_argument("--layout", choices=["paged", "random"], default="paged")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--baselines", nargs="*", choices=["aiter", "flydsl128"], default=["aiter", "flydsl128"])
    parser.add_argument("--asm-splits", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--calls", type=int, default=40, help="decode calls captured per graph")
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()
    if args.mode == "decode":
        args.lengths = args.lengths or [8192, 8200, 9216, 65536, 65544, 262144]
        if any(not DECODE_QLEN <= n <= DECODE_CAPACITY * DECODE_CP_WORLD for n in args.lengths):
            parser.error(f"decode lengths must be in [8, {DECODE_CAPACITY * DECODE_CP_WORLD}]")
        results = bench_decode(args)
    else:
        args.lengths = args.lengths or [128, 512, 2048, 8192, 65536]
        args.replays = min(args.replays, 5)
        if any(s < 128 or s > 262144 or s % 128 for s in args.lengths):
            parser.error("prefill lengths must be multiples of 128 in [128, 262144]")
        results = bench_prefill(args)
    if args.output:
        import flydsl

        env = dict(torch=torch.__version__, hip=torch.version.hip, flydsl=flydsl.__version__, arch=_ARCH)
        with open(args.output, "w") as f:
            json.dump(dict(environment=env, results=results), f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
