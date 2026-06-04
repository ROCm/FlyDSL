import math
import random as _random
import sys
import pytest
import torch
import os
from flydsl.runtime.device import get_rocm_arch

ARCH = get_rocm_arch()
DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz
FP8_MAX = float(torch.finfo(DTYPE_FP8).max)


def quant_paged_cache_pertoken(cache, block_size):
    num_blocks = cache.shape[0]
    head_dim = cache.shape[-1]
    num_head_kv = cache.shape[-2]
    # Round scale_rows UP so block_size * 4 bytes always fit in scale_rows *
    # head_dim bytes; pad the trailing slack with zeros.  Required when
    # block_size * 4 < head_dim (e.g. block_size=16, head_dim=128).
    scale_bytes = block_size * 4
    scale_rows = (scale_bytes + head_dim - 1) // head_dim
    padded_bytes = scale_rows * head_dim

    scale = cache[:, :block_size, :, :].float().abs().max(-1)[0] / FP8_MAX

    cache_fp8 = torch.empty_like(cache, dtype=DTYPE_FP8)
    cache_fp8[:, :block_size, :, :] = (cache[:, :block_size, :, :] / scale[:, :, :, None]).to(DTYPE_FP8)

    # scale: (B, T, H) fp32 → (B, H, T*4) fp8 bytes via permute + bit-reinterpret.
    scale_bytes_view = scale.permute(0, 2, 1).contiguous().view(DTYPE_FP8)
    if padded_bytes > scale_bytes:
        pad = torch.zeros(
            num_blocks, num_head_kv, padded_bytes - scale_bytes,
            dtype=DTYPE_FP8, device=cache.device,
        )
        scale_bytes_view = torch.cat([scale_bytes_view, pad], dim=-1).contiguous()
    scale = (
        scale_bytes_view
        .reshape(num_blocks, num_head_kv, scale_rows, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    cache_fp8[:, block_size:, :, :] = scale

    return cache_fp8, cache_fp8[:, block_size:, :, :]


def quant_paged_cache_perhead(cache, block_size):
    num_head_kv = cache.shape[-2]
    scale = cache[:, :block_size, :, :].float().abs().permute(2, 0, 1, 3).reshape(num_head_kv, -1).max(-1)[0] / FP8_MAX
    cache_fp8 = (cache.float() / scale[None, None, :, None]).to(DTYPE_FP8)

    return cache_fp8, scale * 0.1


def naive_attn_with_paged_kvcache_pscale_func(
    Q,
    K,
    V,
    kvcache,
    block_ids,
    nblocks,
    seqlenq,
    cu_seqlenq,
    num_seq_kvcache,
    QS,
    KS,
    VS,
    p_scale=None,
    p_scale_inv=None,
):
    """P_scale-aware variant of naive_attn_with_paged_kvcache_func (pertoken K
    + per-head V scale).
    """
    num_batch = seqlenq.shape[0]
    num_head_q = Q.shape[1]
    num_head_kv = K.shape[1]
    head_dim = K.shape[2]
    block_size = kvcache.shape[2]
    head_per_group = num_head_q // num_head_kv

    has_ps = p_scale is not None
    if has_ps:
        assert p_scale_inv is not None and p_scale.shape == (num_head_q,)

    Q = Q.reshape(num_batch, -1, num_head_q, head_dim)
    output = torch.empty_like(Q, dtype=torch.bfloat16)
    for bi in range(num_batch):
        BQ = Q[bi].transpose(0, 1).float()  # [num_heads, sq, head_dim]
        blk_ids = block_ids[bi, : nblocks[bi]]
        seqlen = seqlenq[bi] + num_seq_kvcache[bi]
        BK = (
            kvcache[blk_ids, 0, :, :, :]
            .reshape(-1, num_head_kv, head_dim)
            .transpose(0, 1)[:, :seqlen, :]
            .repeat_interleave(head_per_group, dim=0)
        ).float()
        BV = (
            kvcache[blk_ids, 1, :, :, :]
            .reshape(-1, num_head_kv, head_dim)
            .transpose(0, 1)[:, :seqlen, :]
            .repeat_interleave(head_per_group, dim=0)
        ).float()

        # KS shape: (nblocks_global, scale_rows, num_head_kv, head_dim) fp8.
        # fp32 view: (nblocks_global, scale_rows, H, head_dim/4) — scale[b, t, h]
        # at view[b, t//H4, h, t%H4].  Bytes [block_size*4 .. scale_rows*head_dim)
        # are zero-padding when block_size*4 < scale_rows*head_dim; strip them.
        _ks_view = KS[blk_ids, :, :, :].view(torch.float32)
        _sr = _ks_view.shape[1]
        _H4 = _ks_view.shape[3]
        _combined = (
            _ks_view.permute(0, 1, 3, 2)  # (nblocks_local, sr, H4, H)
            .contiguous()
            .reshape(_ks_view.shape[0], _sr * _H4, num_head_kv)  # (nblocks_local, bs_padded, H)
        )
        BKS = (
            _combined[:, :block_size, :]
            .reshape(-1, num_head_kv)        # (nblocks_local * block_size, H)
            .transpose(0, 1)[:, :seqlen]
            .repeat_interleave(head_per_group, dim=0)
        ).float()

        P = BQ @ BK.transpose(-1, -2)
        P = P / math.sqrt(head_dim) * BKS.unsqueeze(1) * QS[bi][:, None, None]

        causal_mask = torch.ones(seqlenq[bi], seqlen - seqlenq[bi], device=Q.device, dtype=torch.bool)
        tail_causal_mask = torch.tril(torch.ones(seqlenq[bi], seqlenq[bi], device=Q.device, dtype=torch.bool))
        causal_mask = torch.cat([causal_mask, tail_causal_mask], dim=-1).unsqueeze(0)
        P = P.masked_fill(~causal_mask, float("-inf"))

        attn_weights = torch.exp(P - P.max(dim=-1)[0][:, :, None])
        gSum = attn_weights.sum(dim=-1)[:, :, None]

        # Mirror the kernel: P_scale is applied per-q-head before fp8 quant.
        if has_ps:
            attn_weights = attn_weights * p_scale[:, None, None]

        attn_weights = attn_weights.to(DTYPE_FP8).float()

        Y = torch.matmul(attn_weights, BV)
        Y = Y / gSum

        # Per-head V scale, optionally compensated by p_scale_inv.
        v_scale_eff = VS[:, None, None].repeat_interleave(head_per_group, dim=0)
        if has_ps:
            v_scale_eff = v_scale_eff * p_scale_inv[:, None, None]
        Y = Y * v_scale_eff

        output[bi] = Y.transpose(0, 1)

    return output.reshape(-1, num_head_q, head_dim)


def _make_pscale(mode, num_head_q, device):
    """Helper for the four canonical p_scale modes."""
    if mode == "none":
        return None, None
    if mode == "all_ones":
        p = torch.ones(num_head_q, dtype=torch.float32, device=device)
        return p, p.clone()
    if mode == "all_2":
        p = torch.full((num_head_q,), 2.0, dtype=torch.float32, device=device)
        pi = torch.full((num_head_q,), 0.5, dtype=torch.float32, device=device)
        return p, pi
    if mode == "per_head_random":
        g = torch.Generator(device=device).manual_seed(20240514)
        p = 0.7 + 0.8 * torch.rand(num_head_q, generator=g, device=device, dtype=torch.float32)
        return p, 1.0 / p
    raise ValueError(mode)


def _flydsl_build_inputs_for_pscale(
    num_batch,
    num_seq_q,
    context_length,
    block_size,
    num_head_kv,
    num_head_q,
    head_dim,
    device,
):
    """Build inputs in FlyDSL's expected layout + an equivalent naive view.

    Returns:
        flydsl_inputs (dict for pa_decode_ps_launch),
        naive_inputs (dict for naive_attn_with_paged_kvcache_pscale_func).
    """
    _random.seed(123)
    torch.manual_seed(123)

    total_queries = num_batch * num_seq_q
    # Random BF16 Q (FlyDSL quantizes internally).
    query_bf16 = torch.randn(total_queries, num_head_q, head_dim, dtype=torch.bfloat16, device=device) * 0.3
    QS_naive = torch.ones((total_queries, num_head_q), dtype=torch.float32, device=device)

    context_lengths = torch.tensor([context_length] * num_batch, dtype=torch.int32, device=device)
    blocks_per_seq = (context_length + block_size - 1) // block_size
    max_blocks_per_seq = max(blocks_per_seq, (16384 + block_size - 1) // block_size)
    total_blocks = max_blocks_per_seq * num_batch
    block_tables_list = [
        [_random.randint(0, total_blocks - 1) for _ in range(blocks_per_seq)] for _ in range(num_batch)
    ]
    block_tables = torch.tensor(block_tables_list, dtype=torch.int32, device=device)

    # Match test_pa_decode_gluon2.py: build paged BF16 KV, then quantize K
    # per-token and V per-head with the packed K-scale tail rows.
    kvcache_scale_rows = (block_size * 4 + head_dim - 1) // head_dim
    kvcache = (
        torch.randn(
            total_blocks,
            2,
            block_size + kvcache_scale_rows,
            num_head_kv,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.2
    )

    kcache, KS = quant_paged_cache_pertoken(kvcache[:, 0, :, :, :], block_size)
    vcache, VS_naive = quant_paged_cache_perhead(kvcache[:, 1, :, :, :], block_size)
    kvcache_fp8 = torch.empty_like(kvcache, dtype=DTYPE_FP8)
    kvcache_fp8[:, 0, :, :, :] = kcache
    kvcache_fp8[:, 1, :, :, :] = vcache
    KS = kvcache_fp8[:, 0, block_size:, :, :]

    # FlyDSL key layout: [num_blocks, num_kv_heads, head_size//16, block_size, 16].
    key_cache_fp8 = (
        kvcache_fp8[:, 0, :block_size, :, :]
        .reshape(total_blocks, block_size, num_head_kv, head_dim // 16, 16)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )

    # FlyDSL trans-V layout: [num_blocks, num_kv_heads, block_size//x, head_size, x].
    value_cache_fp8 = kvcache_fp8[:, 1, :block_size, :, :].permute(0, 2, 3, 1).contiguous()
    x = 16 // value_cache_fp8.element_size()
    value_cache_trans = (
        value_cache_fp8.view(total_blocks, num_head_kv, head_dim, block_size // x, x)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )

    # naive_kvcache matches the hpc-backed test: FP8 K/V data only, with
    # K/V scales supplied separately (KS = per-token K; VS_naive = per-head V).
    naive_kvcache = kvcache_fp8[:, :, :block_size, :, :]
    BKS = KS

    # ── KV scales for FlyDSL ──
    # K scale: raw KS layout `[num_blocks, scale_rows, num_head_kv, head_dim]`
    # fp8 — 4 fp8 bytes bit-pack 1 fp32 scale.  Pass as fp32 view
    # `[num_blocks, scale_rows, num_head_kv, head_dim/4]` so the launcher's
    # `_prepare_scale_tensor` doesn't element-wise cast fp8→fp32 (which would
    # destroy the bit-packing).  Kernel's k_scale_per_token=True path
    # computes the per-(block, kv_head, token) fp32 offset directly.
    KS_flydsl = KS.contiguous().view(torch.float32)
    # V scale: per-kv-head fp32, layout (num_kv_heads,) — passed directly via
    # `v_scale_per_head=True` to the kernel.
    VS_flydsl = VS_naive.contiguous().to(torch.float32)

    return {
        "query_bf16": query_bf16,
        "key_cache_fp8": key_cache_fp8,
        "value_cache_trans": value_cache_trans,
        "context_lengths": context_lengths,
        "block_tables": block_tables,
        "block_tables_list": block_tables_list,
        "key_scale": KS_flydsl,
        "value_scale": VS_flydsl,
        "blocks_per_seq": blocks_per_seq,
    }, {
        "Q_bf16": query_bf16,
        "kvcache": naive_kvcache,
        "block_ids": block_tables,
        "nblocks": torch.tensor([blocks_per_seq] * num_batch, dtype=torch.int32, device=device),
        "seqlenq": torch.tensor([num_seq_q] * num_batch, dtype=torch.int32, device=device),
        "num_seq_kvcache": torch.tensor(
            [context_length - num_seq_q] * num_batch,
            dtype=torch.int32,
            device=device,
        ),
        "QS": QS_naive,
        "BKS": BKS,
        "VS": VS_naive,
    }


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] != 9, reason="skip on non sm90!")
@pytest.mark.parametrize("num_batch", [1, 16])
@pytest.mark.parametrize("num_seq_q", [1, 2])
@pytest.mark.parametrize("max_seq_kv", [1024])
@pytest.mark.parametrize("kv_head_q_head", [(2, 8), (4, 32)])
@pytest.mark.parametrize("p_scale_mode", ["none", "all_ones", "all_2", "per_head_random"])
def test_attn_fp8_pscale(
    num_batch,
    num_seq_q,
    max_seq_kv,
    kv_head_q_head,
    p_scale_mode,
):
    ok = _flydsl_pscale_test_case(
        num_batch=num_batch,
        num_seq_q=num_seq_q,
        context_length=max_seq_kv,
        block_size=16,
        num_head_kv=kv_head_q_head[0],
        num_head_q=kv_head_q_head[1],
        head_dim=128,
        p_scale_mode=p_scale_mode,
    )
    assert ok, "FlyDSL error: test_attn_fp8_pscale failed"


def _flydsl_pscale_test_case(
    *, num_batch, num_seq_q, context_length, block_size, num_head_kv, num_head_q, head_dim, p_scale_mode
):
    """Run one (config × p_scale_mode) combo: FlyDSL vs naive_pscale_func."""
    import sys as _s

    _ROOT = "/FlyDSL"
    if _ROOT not in _s.path:
        _s.path.insert(0, _ROOT)
    from kernels.pa_decode_fp8 import get_recommended_splits, pa_decode_ps_launch

    device = torch.device("cuda:0")
    torch.set_default_device(device)
    flydsl_inp, naive_inp = _flydsl_build_inputs_for_pscale(
        num_batch,
        num_seq_q,
        context_length,
        block_size,
        num_head_kv,
        num_head_q,
        head_dim,
        device,
    )

    # p_scale: per-q-head
    p_scale, p_scale_inv = _make_pscale(p_scale_mode, num_head_q, device="cuda")

    softmax_scale = 1.0 / (head_dim**0.5)

    # ── naive golden ──────────────────────────────────────────────────────
    cu_seqlenq = torch.cumsum(naive_inp["seqlenq"].to(torch.int32), dim=0)
    # naive uses K/V only for .shape — pass dummy tensors with the right dims.
    _dummy_K = torch.empty(1, num_head_kv, head_dim, dtype=torch.bfloat16, device=device)
    _dummy_V = torch.empty(1, num_head_kv, head_dim, dtype=torch.bfloat16, device=device)
    naive_out = naive_attn_with_paged_kvcache_pscale_func(
        naive_inp["Q_bf16"],
        _dummy_K,
        _dummy_V,
        naive_inp["kvcache"],
        naive_inp["block_ids"],
        naive_inp["nblocks"],
        naive_inp["seqlenq"],
        cu_seqlenq,
        naive_inp["num_seq_kvcache"],
        naive_inp["QS"],
        naive_inp["BKS"],
        naive_inp["VS"],
        p_scale=p_scale,
        p_scale_inv=p_scale_inv,
    )

    def _build_ps_page(blocks_list, ctx_lens, bs, dev):
        actual = (ctx_lens + bs - 1) // bs
        kv_indptr = torch.zeros(ctx_lens.shape[0] + 1, dtype=torch.int32, device=dev)
        kv_indptr[1:] = torch.cumsum(actual, dim=0)
        flat = []
        for bi, n in enumerate(actual.tolist()):
            flat.extend(blocks_list[bi][:n])
        return torch.tensor(flat, dtype=torch.int32, device=dev), kv_indptr

    kv_page_indices, kv_indptr = _build_ps_page(
        flydsl_inp["block_tables_list"],
        flydsl_inp["context_lengths"],
        block_size,
        device,
    )
    eqgs = num_seq_q * (num_head_q // num_head_kv)
    # max_context_partition_num for small-block PS path
    context_partition_size = 256
    blocks_per_partition = context_partition_size // block_size

    max_part = get_recommended_splits(num_batch, num_head_kv, blocks_per_partition)
    inter_shape = (num_batch, num_head_kv, max_part, eqgs)
    exp_sums = torch.empty(inter_shape, dtype=torch.float32, device=device)
    max_logits = torch.full(inter_shape, float("-inf"), dtype=torch.float32, device=device)
    tmp_out = torch.empty(*inter_shape, head_dim, dtype=torch.bfloat16, device=device)

    output_flydsl = torch.empty(
        num_batch * num_seq_q,
        num_head_q,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    pa_decode_ps_launch(
        output_flydsl,
        flydsl_inp["query_bf16"],
        flydsl_inp["key_cache_fp8"],
        flydsl_inp["value_cache_trans"],
        flydsl_inp["context_lengths"],
        kv_page_indices,
        kv_indptr,
        softmax_scale,
        key_scale=flydsl_inp["key_scale"],
        value_scale=flydsl_inp["value_scale"],
        sliding_window=0,
        metadata=None,
        block_tables=flydsl_inp["block_tables"],
        max_context_partition_num=max_part,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=tmp_out,
        p_scale=p_scale,
        p_scale_inv=p_scale_inv,
        v_scale_per_head=True,    # VS is per-kv-head (num_kv_heads,) fp32
        k_scale_per_token=True,   # KS is per-token (num_blocks, num_kv_heads, block_size) fp32
    )

    # Compare
    diff = (output_flydsl.float() - naive_out.float()).abs()
    max_diff = float(diff.max())
    rel = max_diff / (naive_out.float().abs().max().item() + 1e-8)
    try:
        torch.testing.assert_close(output_flydsl, naive_out, atol=0.05, rtol=0.05)
        ok = True
    except AssertionError as exc:
        ok = False
        raise exc
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] mode={p_scale_mode:<16}  "
        f"max_abs_diff={max_diff:.4f}  rel={rel:.4%}  "
        f"out_range=({float(output_flydsl.min()):.3f}, {float(output_flydsl.max()):.3f})"
    )
    return ok


def _measure_us(fn, *, warmup: int = 5, iters: int = 50, use_cuda_graph: bool = True) -> float:
    """Time `fn` on the current CUDA stream; returns mean microseconds per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = None
    if use_cuda_graph:
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(capture_stream):
                fn()
            torch.cuda.current_stream().wait_stream(capture_stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(capture_stream):
                with torch.cuda.graph(graph, stream=capture_stream):
                    fn()
            torch.cuda.current_stream().wait_stream(capture_stream)
            for _ in range(warmup):
                graph.replay()
            torch.cuda.synchronize()
        except RuntimeError as exc:
            graph = None
            print(f"  [warn] cuda-graph capture failed, falling back to eager: {exc}")
            torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        if graph is not None:
            graph.replay()
        else:
            fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def _flydsl_pscale_bench(
    *,
    num_batch,
    num_seq_q,
    context_length,
    block_size,
    num_head_kv,
    num_head_q,
    head_dim,
    p_scale_mode="all_2",
    warmup=5,
    iters=50,
    compare_gluon=os.environ.get("COMPARE_GLUON", "0") == "1",
):
    """Benchmark pa_decode_ps_launch (and optionally aiter's pa_decode_gluon)
    for one config under the requested p_scale mode.  Returns a dict with
    timings in microseconds.
    """
    from kernels.pa_decode_fp8 import get_recommended_splits, pa_decode_ps_launch

    device = torch.device("cuda:0")
    torch.set_default_device(device)
    flydsl_inp, _ = _flydsl_build_inputs_for_pscale(
        num_batch, num_seq_q, context_length, block_size,
        num_head_kv, num_head_q, head_dim, device,
    )
    p_scale, p_scale_inv = _make_pscale(p_scale_mode, num_head_q, device="cuda")
    softmax_scale = 1.0 / (head_dim ** 0.5)

    def _build_ps_page(blocks_list, ctx_lens, bs, dev):
        actual = (ctx_lens + bs - 1) // bs
        kv_indptr = torch.zeros(ctx_lens.shape[0] + 1, dtype=torch.int32, device=dev)
        kv_indptr[1:] = torch.cumsum(actual, dim=0)
        flat = []
        for bi, n in enumerate(actual.tolist()):
            flat.extend(blocks_list[bi][:n])
        return torch.tensor(flat, dtype=torch.int32, device=dev), kv_indptr

    kv_page_indices, kv_indptr = _build_ps_page(
        flydsl_inp["block_tables_list"], flydsl_inp["context_lengths"], block_size, device,
    )


    eqgs = num_seq_q * (num_head_q // num_head_kv)
    context_partition_size = 256
    blocks_per_partition = context_partition_size // block_size

    max_part = get_recommended_splits(num_batch, num_head_kv, blocks_per_partition)
    inter_shape = (num_batch, num_head_kv, max_part, eqgs)
    exp_sums = torch.empty(inter_shape, dtype=torch.float32, device=device)
    max_logits = torch.full(inter_shape, float("-inf"), dtype=torch.float32, device=device)
    tmp_out = torch.empty(*inter_shape, head_dim, dtype=torch.bfloat16, device=device)
    output_flydsl = torch.empty(
        num_batch * num_seq_q, num_head_q, head_dim,
        dtype=torch.bfloat16, device=device,
    )

    def _run_flydsl():
        pa_decode_ps_launch(
            output_flydsl,
            flydsl_inp["query_bf16"], flydsl_inp["key_cache_fp8"], flydsl_inp["value_cache_trans"],
            flydsl_inp["context_lengths"], kv_page_indices, kv_indptr, softmax_scale,
            key_scale=flydsl_inp["key_scale"], value_scale=flydsl_inp["value_scale"],
            sliding_window=0, metadata=None,
            block_tables=flydsl_inp["block_tables"], max_context_partition_num=max_part,
            exp_sums=exp_sums, max_logits=max_logits, temporary_output=tmp_out,
            p_scale=p_scale, p_scale_inv=p_scale_inv,
            v_scale_per_head=True, k_scale_per_token=True,
        )

    # Warm JIT + first launch outside the timed region.
    _run_flydsl()
    torch.cuda.synchronize()
    us_flydsl = _measure_us(_run_flydsl, warmup=warmup, iters=iters)

    us_gluon = None
    if compare_gluon:
        try:
            from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon
            out_gluon = torch.empty_like(output_flydsl)
            exp_sums_g = torch.empty(inter_shape, dtype=torch.float32, device=device)
            max_logits_g = torch.full(inter_shape, float("-inf"), dtype=torch.float32, device=device)
            tmp_out_g = torch.empty(*inter_shape, head_dim, dtype=torch.bfloat16, device=device)
            # gluon wants flat per-token K scale `[num_blocks, num_kv_heads, block_size, 1]` fp32;
            # unpack from FlyDSL's packed `[B, scale_rows, H, head_dim/4]` fp32 view.
            _ks_view = flydsl_inp["key_scale"]
            _B, _sr, _H, _H4 = _ks_view.shape
            ks_gluon = (
                _ks_view.permute(0, 2, 1, 3).contiguous()
                .reshape(_B, _H, _sr * _H4)[:, :, :block_size]
                .unsqueeze(-1).contiguous()
            )
            # gluon wants per-token V scale of the same `[B, H, bs, 1]` shape.
            # Our v_scale is per-kv-head `(H,)`; broadcast to per-token.
            _vs_per_head = flydsl_inp["value_scale"]
            vs_gluon = (
                _vs_per_head.view(1, _H, 1, 1)
                .expand(_B, _H, block_size, 1).contiguous()
            )

            def _run_gluon():
                pa_decode_gluon(
                    out_gluon,
                    flydsl_inp["query_bf16"], flydsl_inp["key_cache_fp8"], flydsl_inp["value_cache_trans"],
                    flydsl_inp["context_lengths"], flydsl_inp["block_tables"], softmax_scale,
                    query_length=num_seq_q, max_context_partition_num=max_part,
                    context_partition_size=context_partition_size,
                    key_scale=ks_gluon, value_scale=vs_gluon,
                    exp_sums=exp_sums_g, max_logits=max_logits_g, temporary_output=tmp_out_g,
                    sliding_window=0, ps=True,
                )

            _run_gluon()
            torch.cuda.synchronize()
            us_gluon = _measure_us(_run_gluon, warmup=warmup, iters=iters)
        except Exception as exc:
            print(f"  [warn] gluon comparison skipped: {exc}")

    speedup = (us_gluon / us_flydsl) if us_gluon else None
    cfg_str = (
        f"b={num_batch} sq={num_seq_q} ctx={context_length} bs={block_size} "
        f"H_kv={num_head_kv} H_q={num_head_q} hd={head_dim}"
    )
    gluon_str = (
        f"  gluon={us_gluon:7.2f}us  speedup={speedup:.2f}x" if us_gluon is not None else ""
    )
    print(f"  [perf] {cfg_str:<60} flydsl={us_flydsl:7.2f}us{gluon_str}")

    return {"flydsl_us": us_flydsl, "gluon_us": us_gluon, "speedup": speedup}


def run_flydsl_pscale_perf():
    """Run a sweep of perf configs covering small/large batch and context."""
    _kv_lens = [131072]
    _seq_counts = [256]
    _base = dict(num_seq_q=2, block_size=16, num_head_kv=1, num_head_q=8, head_dim=128)
    configs = [
        dict(num_batch=ns, context_length=kv, **_base)
        for kv in _kv_lens
        for ns in _seq_counts
    ]
    p_scale_mode="none"
    print(f"\n=== FlyDSL pa_decode_ps_kernel performance (p_scale_mode={p_scale_mode}) ===")
    results = []
    for cfg in configs:
        r = _flydsl_pscale_bench(**cfg, p_scale_mode=p_scale_mode)
        results.append((cfg, r))
    return results


def run_flydsl_pscale_smoke():
    """Smoke test: backward compat (mode=none) + the 3 p_scale modes."""
    # The small-batch / small-context cases used to fault in the
    # pa_decode_ps_kernel block_size=16 prologue (empty-slot partition_idx
    # read OOB phys_blocks → OOB K addr → segfault).  The
    # `_safe_partition_start` clamp in pa_decode_fp8.py now keeps the
    # prologue's reads in-bounds for empty slots.
    # configs = [
    #     dict(num_batch=4, num_seq_q=1, context_length=1024, block_size=16, num_head_kv=2, num_head_q=8, head_dim=128),
    #     dict(num_batch=8, num_seq_q=1, context_length=8192, block_size=16, num_head_kv=2, num_head_q=8, head_dim=128),
    # ]
    _kv_lens = [128, 256, 512, 1024, 16384, 32768, 65536, 131072]
    _seq_counts = [2, 4, 8, 16, 32, 64, 128, 256]
    _base = dict(num_seq_q=2, block_size=16, num_head_kv=1, num_head_q=8, head_dim=128)
    configs = [
        dict(num_batch=ns, context_length=kv, **_base)
        for kv in _kv_lens
        for ns in _seq_counts
    ]
    modes = ["none", "all_ones", "all_2", "per_head_random"]
    all_ok = True
    for cfg in configs:
        print(f"\nConfig: {cfg}")
        for m in modes:
            ok = _flydsl_pscale_test_case(**cfg, p_scale_mode=m)
            all_ok &= ok
    print("\n" + ("ALL FlyDSL p_scale tests PASSED" if all_ok else "SOME FlyDSL p_scale tests FAILED"))
    return all_ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", action="store_true", help="run kernel perf benchmark sweep")
    parser.add_argument("--smoke", action="store_true", help="run correctness smoke (default if neither flag set)")
    args = parser.parse_args()

    if args.perf:
        run_flydsl_pscale_perf()
        sys.exit(0)

    ok = run_flydsl_pscale_smoke()
    sys.exit(0 if ok else 1)
