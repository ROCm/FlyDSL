# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""CPU checks for gfx120x MLA one-partition split buffers. No kernel launch."""

import inspect

import pytest
import torch

from kernels.attention.flash_attn_gfx120x_host import flydsl_flash_attn_func
from kernels.attention.mla_fwd_decode_gfx120x import flydsl_mla_fwd_decode
from kernels.attention.mla_fwd_decode_m16x8_fp8_fp8_gfx120x import (
    launch_mla_fwd_decode_m16x8_fp8_fp8,
)

pytestmark = [pytest.mark.l0_backend_agnostic]


class _Flash:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.out: torch.Tensor | None = None
        self.lse: torch.Tensor | None = None

    def __call__(self, q, k, v, **kwargs):
        b, s, h, _qk = (int(x) for x in q.shape)
        dv = int(v.shape[-1])
        out = torch.arange(b * s * h * dv, dtype=torch.float32).reshape(b, s, h, dv)
        lse = torch.arange(b * h * s, dtype=torch.float32).reshape(b, h, s) + 100
        self.out = out
        self.lse = lse
        self.calls.append({"kwargs": dict(kwargs), "q": tuple(q.shape), "v": tuple(v.shape)})
        if kwargs.get("return_lse"):
            return out, lse
        return out


def _patch(monkeypatch, flash: _Flash, arch: str) -> None:
    monkeypatch.setattr("kernels.attention.mla_fwd_decode_gfx120x.get_rocm_arch", lambda: arch)
    monkeypatch.setattr(
        "kernels.attention.mla_fwd_decode_m16x8_fp8_fp8_gfx120x.get_rocm_arch",
        lambda: arch,
    )
    monkeypatch.setattr("kernels.attention.flash_attn_gfx120x_host.flydsl_flash_attn_func", flash)


def _pages(batch: int, heads: int, qk: int, per: int):
    indptr = torch.arange(0, batch * per + 1, per, dtype=torch.int32)
    indices = torch.arange(batch * per, dtype=torch.int32)
    kv = torch.randn(batch * per, 1, 1, qk)
    return indptr, indices, kv, torch.empty(1, dtype=torch.int32)


def test_flash_entry_exposes_return_lse_without_grid() -> None:
    params = inspect.signature(flydsl_flash_attn_func).parameters
    assert "return_lse" in params
    assert "grid" not in params


def test_arch_requires_gfx120_prefix(monkeypatch) -> None:
    flash = _Flash()
    _patch(monkeypatch, flash, "gfx950")
    q = torch.randn(1, 2, 32)
    kv = torch.randn(1, 1, 1, 32)
    out = torch.empty(1, 2, 16)
    work = torch.empty(1, dtype=torch.int32)
    indptr = torch.tensor([0, 1], dtype=torch.int32)
    indices = torch.tensor([0], dtype=torch.int32)
    with pytest.raises(AssertionError, match="gfx120"):
        flydsl_mla_fwd_decode(q, kv, indices, indptr, work, out, out, torch.empty(1), 1.0)
    _patch(monkeypatch, flash, "gfx12")
    with pytest.raises(AssertionError, match="gfx120"):
        launch_mla_fwd_decode_m16x8_fp8_fp8(
            q.reshape(2, 32),
            kv.reshape(1, 32),
            indices,
            indptr,
            work,
            out.reshape(2, 16),
            torch.empty(1),
            torch.empty(1),
            1.0,
            num_cus=4,
            lds_size=0,
        )
    assert flash.calls == []


def test_one_partition_writes_matching_splits_only(monkeypatch) -> None:
    """Same-numel split_output is copied. LSE is written only when it fits."""
    flash = _Flash()
    _patch(monkeypatch, flash, "gfx1201")
    batch, heads, qk, dv, per = 2, 4, 32, 16, 2
    indptr, indices, kv, work = _pages(batch, heads, qk, per)
    q = torch.randn(batch, heads, qk)
    final = torch.empty(batch, heads, dv)
    split_o = torch.empty(heads, batch * dv)
    split_lse = torch.empty(batch, heads, 1)
    small_o = torch.full((1,), -3.0)
    small_lse = torch.full((1,), -4.0)
    flydsl_mla_fwd_decode(
        q,
        kv,
        indices,
        indptr,
        work,
        final,
        split_o,
        split_lse,
        qk**-0.5,
        num_cus=10**6,
        lds_size=0,
    )
    assert len(flash.calls) == 1
    kwargs = flash.calls[0]["kwargs"]
    assert kwargs["return_lse"] is True
    assert kwargs["causal"] is False
    assert "grid" not in kwargs
    assert "num_cus" not in kwargs
    assert "lds_size" not in kwargs
    assert torch.equal(final, flash.out.reshape(final.shape))
    assert torch.equal(split_o.reshape(-1), final.reshape(-1))
    assert torch.equal(split_lse.reshape(-1), flash.lse.reshape(-1))

    flash.calls.clear()
    wide_lse = torch.full((batch + 1, 1, heads, 1), 7.0)
    held = wide_lse.clone()
    flydsl_mla_fwd_decode(q, kv, indices, indptr, work, final, small_o, wide_lse, 1.0)
    assert torch.equal(small_o, torch.full((1,), -3.0))
    assert torch.equal(wide_lse[:batch], flash.lse.reshape(batch, 1, heads, 1))
    assert torch.equal(wide_lse[batch:], held[batch:])

    flydsl_mla_fwd_decode(q, kv, indices, indptr, work, final, small_o, small_lse, 1.0)
    assert torch.equal(small_lse, torch.full((1,), -4.0))
    # calls were cleared after the first launch, so these two remain.
    assert len(flash.calls) == 2
    assert all(call["kwargs"]["return_lse"] is True for call in flash.calls)


def test_longer_query_and_strided_lse(monkeypatch) -> None:
    flash = _Flash()
    _patch(monkeypatch, flash, "gfx120")
    batch, heads, sq, qk, dv = 2, 3, 2, 32, 16
    indptr, indices, kv, work = _pages(batch, heads, qk, 1)
    q = torch.randn(batch * sq, heads, qk)
    final = torch.empty(batch * sq, heads, dv)
    split_o = torch.full((batch * sq, heads, dv * 2), 5.0)[..., :dv]
    assert split_o.shape == final.shape and not split_o.is_contiguous()
    raw = torch.full((batch * sq + 2, 1, heads, 2), 9.0)
    split_lse = raw[..., :1]
    assert not split_lse.is_contiguous()
    flydsl_mla_fwd_decode(q, kv, indices, indptr, work, final, split_o, split_lse, 0.5)
    assert flash.calls[0]["kwargs"]["causal"] is True
    assert torch.equal(split_o, final)
    expect = flash.lse.reshape(batch * sq, 1, heads, 1)
    assert torch.equal(split_lse[: batch * sq], expect)
    assert torch.equal(split_lse[batch * sq :], torch.full_like(split_lse[batch * sq :], 9.0))


def test_m16x8_flat_forwards_counts_without_a_grid(monkeypatch) -> None:
    flash = _Flash()
    _patch(monkeypatch, flash, "gfx1200")
    heads, qk, dv, pages = 2, 32, 16, 2
    q = torch.randn(heads, qk)
    kv = torch.randn(pages, qk)
    final = torch.empty(heads, dv)
    split_o = torch.empty(1, heads, dv)
    split_lse = torch.full((3,), -1.0)
    launch_mla_fwd_decode_m16x8_fp8_fp8(
        q,
        kv,
        torch.tensor([1, 0], dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        final,
        split_o,
        split_lse,
        qk**-0.5,
        num_cus=80,
        lds_size=1,
    )
    assert len(flash.calls) == 1
    assert flash.calls[0]["kwargs"]["return_lse"] is True
    assert "grid" not in flash.calls[0]["kwargs"]
    assert torch.equal(final.reshape(-1), flash.out.reshape(-1))
    assert torch.equal(split_o.reshape(-1), final.reshape(-1))
    assert split_lse[0].item() == pytest.approx(flash.lse.reshape(-1)[0].item())
    assert split_lse[1].item() == pytest.approx(flash.lse.reshape(-1)[1].item())
    assert split_lse[2].item() == pytest.approx(-1.0)
