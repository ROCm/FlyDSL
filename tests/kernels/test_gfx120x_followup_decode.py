# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Host follow-ups for gfx120x decode: CSR block tables, page dequant, one partition."""

import sys
from pathlib import Path

import pytest

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

torch = pytest.importorskip("torch")

import kernels.attention.pa_decode_fp8_gfx120x as ps  # noqa: E402
import kernels.attention.pa_decode_tile_gfx120x as tile  # noqa: E402


class _Metadata:
    def __getattr__(self, name):
        raise AssertionError(f"metadata.{name} was read")


@pytest.fixture
def gfx120(monkeypatch):
    monkeypatch.setattr(ps, "get_rocm_arch", lambda: "gfx1201")
    monkeypatch.setattr(tile, "get_rocm_arch", lambda: "gfx1201")
    return monkeypatch


def _install_flash(monkeypatch, fn):
    import kernels.attention.flash_attn_gfx120x_host as host

    monkeypatch.setattr(host, "flydsl_flash_attn_func", fn)


def _pack(linear: torch.Tensor, vec: int) -> tuple[torch.Tensor, torch.Tensor]:
    nb, page, heads, dim = linear.shape
    k5 = linear.permute(0, 2, 1, 3).reshape(nb, heads, page, dim // vec, vec).permute(0, 1, 3, 2, 4).contiguous()
    v5 = linear.permute(0, 2, 1, 3).reshape(nb, heads, page // vec, vec, dim).permute(0, 1, 2, 4, 3).contiguous()
    return k5, v5


def test_block_table_from_csr_is_rectangular_int32() -> None:
    indices = torch.tensor([5, 6, 8, 99], dtype=torch.int64)
    indptr = torch.tensor([0, 2, 2, 3], dtype=torch.int32)
    table = ps._block_table_from_csr(indices, indptr, 3)
    assert table.dtype == torch.int32
    assert table.shape == (3, 2)
    assert torch.equal(
        table,
        torch.tensor([[5, 6], [0, 0], [8, 0]], dtype=torch.int32),
    )
    # Tail past kv_indptr[-1] is not a page id and is not copied.
    assert int(table.max().item()) != 99


def test_csr_launch_builds_table_and_leaves_metadata(gfx120, monkeypatch) -> None:
    seen = {}

    def fake(q, k, v, **kwargs):
        seen["k"] = k
        seen["bt"] = kwargs["block_table"]
        seen["stream"] = kwargs["stream"]
        return torch.zeros_like(q)

    _install_flash(monkeypatch, fake)
    q = torch.zeros(2, 1, 4, dtype=torch.bfloat16)
    cache = torch.zeros(8, 4, 1, 4, dtype=torch.bfloat16)
    out = torch.empty(2, 1, 4, dtype=torch.bfloat16)
    kind = ps.pa_decode_ps_launch(
        out,
        q,
        cache,
        cache,
        torch.tensor([4, 7], dtype=torch.int32),
        torch.tensor([4, 1, 7], dtype=torch.int32),
        torch.tensor([0, 2, 3], dtype=torch.int32),
        0.5,
        block_tables=None,
        max_context_partition_num=2,
        metadata=_Metadata(),
        stream="host-stream",
    )
    assert kind == "ps_small_block"
    assert seen["k"] is cache
    assert seen["stream"] == "host-stream"
    assert torch.equal(seen["bt"], torch.tensor([[4, 1], [7, 0]], dtype=torch.int32))


def test_same_dtype_keeps_cache_scales_and_table(gfx120, monkeypatch) -> None:
    seen = {}

    def fake(q, k, v, **kwargs):
        seen["k"] = k
        seen["v"] = v
        seen["kw"] = kwargs
        return torch.ones_like(q)

    _install_flash(monkeypatch, fake)
    q = torch.zeros(1, 2, 4, dtype=torch.float16)
    cache = torch.zeros(2, 4, 2, 4, dtype=torch.float16)
    table = torch.tensor([[1, 0]], dtype=torch.int32)
    scale_k = torch.tensor([0.25], dtype=torch.float32)
    scale_v = torch.tensor([0.5], dtype=torch.float32)
    out = torch.empty(1, 2, 4, dtype=torch.float16)
    kind = ps.pa_decode_ps_launch(
        out,
        q,
        cache,
        cache,
        torch.tensor([4], dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        0.125,
        scale_k,
        scale_v,
        sliding_window=4,
        block_tables=table,
        max_context_partition_num=4,
    )
    assert kind == "ps_sw_fused_partitioned"
    assert seen["k"] is cache and seen["v"] is cache
    assert seen["kw"]["block_table"] is table
    assert seen["kw"]["q_descale"] is scale_k
    assert seen["kw"]["k_descale"] is scale_k
    assert seen["kw"]["v_descale"] is scale_v
    assert seen["kw"]["causal"] is True
    assert seen["kw"]["sliding_window"] == (4, 0)
    assert seen["kw"]["kv_cache_layout"] == "linear"
    assert torch.equal(out, torch.ones_like(out))


def test_mixed_dequant_reads_only_block_table_pages(gfx120, monkeypatch) -> None:
    seen = {}

    def fake(q, k, v, **kwargs):
        seen["k"] = k
        seen["v"] = v
        seen["kw"] = kwargs
        return torch.zeros_like(q)

    _install_flash(monkeypatch, fake)
    raw = torch.arange(4 * 2 * 1 * 4, dtype=torch.float32).reshape(4, 2, 1, 4)
    raw[1].fill_(99)
    cache = raw.to(torch.int8)
    q = torch.zeros(1, 1, 4, dtype=torch.bfloat16)
    table = torch.tensor([[0, 2]], dtype=torch.int32)
    out = torch.empty(1, 1, 4, dtype=torch.bfloat16)
    ps.pa_decode_ps_launch(
        out,
        q,
        cache,
        cache,
        torch.tensor([4], dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        1.0,
        2.0,
        3.0,
        block_tables=table,
        max_context_partition_num=1,
    )
    assert seen["k"].dtype == torch.bfloat16
    assert seen["k"].shape[0] == 2
    assert seen["k"] is not cache
    assert seen["kw"]["k_descale"] is None and seen["kw"]["v_descale"] is None
    assert torch.equal(seen["kw"]["block_table"], torch.tensor([[0, 1]], dtype=torch.int32))
    expect_k = (raw[[0, 2]].float() * 2).to(torch.bfloat16)
    expect_v = (raw[[0, 2]].float() * 3).to(torch.bfloat16)
    assert torch.equal(seen["k"], expect_k)
    assert torch.equal(seen["v"], expect_v)
    assert torch.equal(cache[1], torch.full_like(cache[1], 99))
    poisoned = (raw[1].float() * 2).to(torch.bfloat16)
    assert not any(torch.equal(seen["k"][i], poisoned) for i in range(seen["k"].shape[0]))


def test_fp8_dequant_scale_and_default() -> None:
    raw = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32).reshape(4, 1, 1, 1)
    table = torch.tensor([[3, 0]], dtype=torch.int32)
    for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        cache = raw.to(dtype)
        key, value, remap = ps._dequant_referenced_caches(cache, cache, table, 0.5, None, torch.bfloat16)
        assert key.shape[0] == 2 and value.shape[0] == 2
        assert torch.equal(remap, torch.tensor([[1, 0]], dtype=torch.int32))
        # Unique page ids are sorted, so the compact cache is page 0 then page 3.
        assert torch.equal(key.float().reshape(-1), torch.tensor([0.5, 4.0]))
        assert torch.equal(value.float().reshape(-1), torch.tensor([1.0, 8.0]))


def test_vectorized_per_token_scale_skips_other_pages() -> None:
    linear_k = torch.zeros(3, 16, 2, 16, dtype=torch.int32)
    linear_v = torch.zeros_like(linear_k)
    linear_k[0], linear_k[1], linear_k[2] = 1, 9, 4
    linear_v[0], linear_v[1], linear_v[2] = 3, 8, 5
    k5, _ = _pack(linear_k.to(torch.int8), 16)
    _, v5 = _pack(linear_v.to(torch.int8), 16)
    scale_k = torch.ones(3, 2, 16, dtype=torch.float32)
    scale_v = torch.ones(3, 2, 16, dtype=torch.float32)
    scale_k[0] = 2
    scale_k[2, 1] = 3
    scale_v[2] = 4
    table = torch.tensor([[2, 0]], dtype=torch.int32)
    key, value, remap = ps._dequant_referenced_caches(k5, v5, table, scale_k, scale_v, torch.float16)
    assert tuple(key.shape) == (2, 2, 1, 16, 16)
    assert tuple(value.shape) == (2, 2, 1, 16, 16)
    assert torch.equal(remap, torch.tensor([[1, 0]], dtype=torch.int32))
    pages = torch.tensor([0, 2])
    expect_k = linear_k[pages].float() * scale_k[pages].permute(0, 2, 1).unsqueeze(-1)
    expect_v = linear_v[pages].float() * scale_v[pages].permute(0, 2, 1).unsqueeze(-1)
    got_k, _ = _pack(expect_k, 16)
    _, got_v = _pack(expect_v, 16)
    assert torch.equal(key.float(), got_k.to(torch.float16).float())
    assert torch.equal(value.float(), got_v.to(torch.float16).float())


def test_default_scale_is_one() -> None:
    cache = torch.tensor([4, 5, 6], dtype=torch.int8).reshape(3, 1, 1, 1)
    table = torch.tensor([[2]], dtype=torch.int32)
    key, value, remap = ps._dequant_referenced_caches(cache, cache, table, None, None, torch.float16)
    assert torch.equal(remap, torch.tensor([[0]], dtype=torch.int32))
    assert torch.equal(key.float().reshape(-1), torch.tensor([6.0]))
    assert torch.equal(value, key)


def _workspace(batch, kv_heads, parts, eqgs, dim, fill):
    exp = torch.full((batch, kv_heads, parts, eqgs), fill, dtype=torch.float32)
    logits = torch.full((batch, kv_heads, parts, eqgs), fill - 1, dtype=torch.float32)
    temporary = torch.full((batch, kv_heads, parts, eqgs, dim), fill + 2, dtype=torch.bfloat16)
    return exp, logits, temporary


def _expect_partition(output, batch, kv_heads):
    rows, num_q_heads, dim = output.shape
    query_length = rows // batch
    group = num_q_heads // kv_heads
    expected = torch.empty(batch, kv_heads, query_length * group, dim, dtype=output.dtype)
    for bi in range(batch):
        for token in range(query_length):
            for kv in range(kv_heads):
                for local in range(group):
                    expected[bi, kv, token * group + local] = output[bi * query_length + token, kv * group + local]
    return expected


def test_single_partition_copies_output_and_leaves_stats(gfx120, monkeypatch) -> None:
    def fake(q, k, v, **kwargs):
        del k, v, kwargs
        batch, seq, heads, dim = q.shape
        vals = torch.arange(batch * seq * heads * dim, dtype=torch.float32).reshape(batch, seq, heads, dim)
        return vals.to(dtype=q.dtype)

    _install_flash(monkeypatch, fake)
    q = torch.zeros(2, 4, 4, dtype=torch.bfloat16)
    cache = torch.zeros(2, 4, 2, 4, dtype=torch.bfloat16)
    out = torch.empty(2, 4, 4, dtype=torch.bfloat16)
    exp, logits, temporary = _workspace(1, 2, 1, 4, 4, 3.25)
    exp_before, logits_before = exp.clone(), logits.clone()
    ps.pa_decode_ps_launch(
        out,
        q,
        cache,
        cache,
        torch.tensor([4], dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        1.0,
        block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
        max_context_partition_num=1,
        exp_sums=exp,
        max_logits=logits,
        temporary_output=temporary,
    )
    assert torch.equal(exp, exp_before)
    assert torch.equal(logits, logits_before)
    assert torch.equal(temporary[:, :, 0], _expect_partition(out, 1, 2))


def test_vectorized_single_partition_uses_head_axis(gfx120, monkeypatch) -> None:
    def fake(q, k, v, **kwargs):
        del k, v, kwargs
        batch, seq, heads, dim = q.shape
        vals = torch.arange(1, batch * seq * heads * dim + 1, dtype=torch.float32).reshape(q.shape)
        return vals.to(dtype=q.dtype)

    _install_flash(monkeypatch, fake)
    q = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
    cache = torch.zeros(1, 2, 1, 8, 8, dtype=torch.bfloat16)
    out = torch.empty(1, 2, 8, dtype=torch.bfloat16)
    exp, logits, temporary = _workspace(1, 2, 1, 1, 8, 1.5)
    ps.pa_decode_ps_launch(
        out,
        q,
        cache,
        cache,
        torch.tensor([8], dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        1.0,
        block_tables=torch.tensor([[0]], dtype=torch.int32),
        max_context_partition_num=1,
        exp_sums=exp,
        max_logits=logits,
        temporary_output=temporary,
    )
    assert torch.equal(temporary[:, :, 0], _expect_partition(out, 1, 2))
    assert torch.equal(exp, torch.full_like(exp, 1.5))


def test_other_partition_counts_leave_partials(gfx120, monkeypatch) -> None:
    def fake(q, k, v, **kwargs):
        del k, v, kwargs
        return torch.ones_like(q)

    _install_flash(monkeypatch, fake)
    q = torch.zeros(1, 1, 4, dtype=torch.bfloat16)
    cache = torch.zeros(1, 4, 1, 4, dtype=torch.bfloat16)
    out = torch.empty(1, 1, 4, dtype=torch.bfloat16)
    exp, logits, temporary = _workspace(1, 1, 2, 1, 4, 8.0)
    before = (exp.clone(), logits.clone(), temporary.clone())
    ps.pa_decode_ps_launch(
        out,
        q,
        cache,
        cache,
        torch.tensor([4], dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        1.0,
        block_tables=torch.tensor([[0]], dtype=torch.int32),
        max_context_partition_num=2,
        exp_sums=exp,
        max_logits=logits,
        temporary_output=temporary,
    )
    assert torch.equal(exp, before[0])
    assert torch.equal(logits, before[1])
    assert torch.equal(temporary, before[2])
    lone = torch.full_like(out, 4)
    ps.pa_decode_ps_launch(
        out,
        q,
        cache,
        cache,
        torch.tensor([4], dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        1.0,
        block_tables=torch.tensor([[0]], dtype=torch.int32),
        max_context_partition_num=1,
        temporary_output=lone,
    )
    assert torch.equal(lone, torch.full_like(lone, 4))


def test_arch_assert_runs_before_flash(monkeypatch) -> None:
    monkeypatch.setattr(ps, "get_rocm_arch", lambda: "gfx942")
    monkeypatch.setattr(tile, "get_rocm_arch", lambda: "gfx942")
    called = {"n": 0}

    def boom(*args, **kwargs):
        del args, kwargs
        called["n"] += 1
        raise RuntimeError("flash")

    _install_flash(monkeypatch, boom)
    q = torch.zeros(1, 1, 4)
    cache = torch.zeros(1, 1, 1, 4)
    with pytest.raises(AssertionError, match="gfx120"):
        ps.pa_decode_ps_launch(
            torch.empty(1, 1, 4),
            q,
            cache,
            cache,
            torch.tensor([1], dtype=torch.int32),
            None,
            None,
            1.0,
            block_tables=None,
            max_context_partition_num=1,
        )
    assert called["n"] == 0


def test_tile_rejects_mixed_dtype(gfx120) -> None:
    q = torch.zeros(1, 1, 4, dtype=torch.bfloat16)
    cache = torch.zeros(1, 4, 1, 4, dtype=torch.int8)
    with pytest.raises(ValueError, match="one dtype"):
        tile.pa_decode_tile(
            torch.empty(1, 1, 4, dtype=torch.bfloat16),
            q,
            cache,
            cache,
            torch.zeros(1, 1, dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            None,
            None,
        )


def test_missing_table_and_bad_dtype(gfx120, monkeypatch) -> None:
    _install_flash(monkeypatch, lambda *args, **kwargs: torch.zeros_like(args[0]))
    q = torch.zeros(1, 1, 4, dtype=torch.float32)
    cache = torch.zeros(1, 4, 1, 4, dtype=torch.bfloat16)
    out = torch.empty(1, 1, 4, dtype=torch.float32)
    ctx = torch.tensor([1], dtype=torch.int32)
    with pytest.raises(ValueError, match="block_tables"):
        ps.pa_decode_ps_launch(
            out,
            cache.new_zeros(1, 1, 4),
            cache,
            cache,
            ctx,
            None,
            None,
            1.0,
            block_tables=None,
            max_context_partition_num=1,
        )
    with pytest.raises(ValueError, match="one dtype"):
        ps.pa_decode_ps_launch(
            out,
            q,
            cache,
            cache,
            ctx,
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            1.0,
            block_tables=torch.zeros(1, 1, dtype=torch.int32),
            max_context_partition_num=1,
        )
    with pytest.raises(ValueError, match="positive"):
        ps.pa_decode_ps_launch(
            out,
            cache,
            cache,
            cache,
            ctx,
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            1.0,
            block_tables=torch.zeros(1, 1, dtype=torch.int32),
            max_context_partition_num=0,
        )
