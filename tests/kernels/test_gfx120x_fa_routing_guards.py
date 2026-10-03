# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""CPU checks for gfx120x FA routing guards. No GPU and no kernel launch."""

import pytest
import torch

from kernels.attention.flash_attn_gfx120x_ext import (
    lengths_uniform,
    reject_gqa,
    reject_paged_layout,
    reject_quant_extras,
    reject_sink_with_alibi,
    reject_splitk_extras,
    reject_varlen_with_paged,
)

pytestmark = [pytest.mark.l0_backend_agnostic]


def test_gqa_rejected() -> None:
    # Invalid groups still raise. Valid GQA (8 q / 2 kv) is in-kernel.
    with pytest.raises(ValueError, match="divisible"):
        reject_gqa(8, 3, None)
    with pytest.raises(ValueError, match="does not match"):
        reject_gqa(8, 2, 4)
    with pytest.raises(ValueError, match="does not match"):
        reject_gqa(8, 8, 2)
    reject_gqa(8, 2, None)
    reject_gqa(8, 2, 2)
    reject_gqa(8, 8, 8)


def test_fp8_extras_not_dropped_by_guard() -> None:
    # bias / ALiBi / sink / LSE are applied in the quant kernel; the guard must not raise.
    reject_quant_extras("fp8", bias=object(), alibi_slopes=object(), sink=object(), return_lse=True)
    reject_quant_extras("int8", attn_mask=object(), return_lse=True)
    reject_quant_extras("fp8")


def test_ragged_lengths_are_not_uniform() -> None:
    assert not lengths_uniform(torch.tensor([16, 32]), 32)
    assert lengths_uniform(torch.tensor([32, 32]), 32)
    assert lengths_uniform(torch.tensor([], dtype=torch.int32), 0)


def test_vectorized_paged_layout_rejected() -> None:
    reject_paged_layout("vectorized")
    reject_paged_layout("linear3d")
    reject_paged_layout("linear")
    reject_paged_layout(None)
    with pytest.raises(NotImplementedError, match="blocked"):
        reject_paged_layout("blocked")


def test_varlen_with_paged_allowed() -> None:
    reject_varlen_with_paged(True, True)
    reject_varlen_with_paged(True, False)
    reject_varlen_with_paged(False, True)


def test_varlen_batch_bias_slices_packed_rows() -> None:
    from kernels.attention.flash_attn_gfx120x_ext import varlen_batch_bias

    # Two batches, lengths 2 and 3. Packed bias rows follow cu_seqlens.
    cu = torch.tensor([0, 2, 5], dtype=torch.int64)
    bias = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    b0 = varlen_batch_bias(bias, cu, 0, 2, 3, max_seqlen_q=3, max_seqlen_kv=4)
    b1 = varlen_batch_bias(bias, cu, 1, 3, 4, max_seqlen_q=3, max_seqlen_kv=4)
    assert torch.equal(b0, bias[0:2, :3])
    assert torch.equal(b1, bias[2:5, :4])
    shared = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    prefix = varlen_batch_bias(shared, cu, 0, 2, 3, max_seqlen_q=3, max_seqlen_kv=4)
    assert torch.equal(prefix, shared[:2, :3])
    assert varlen_batch_bias(None, cu, 0, 2, 3, 3, 4) is None
    with pytest.raises(NotImplementedError):
        varlen_batch_bias(torch.zeros(2, 2, 3, 4), cu, 0, 2, 3, 3, 4)


def test_fold_sink_lse_matches_logaddexp() -> None:
    from kernels.attention.flash_attn_gfx120x_ext import fold_sink_lse

    lse = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])  # [1, 2, 2]
    sink = torch.tensor([0.5, -1.0])
    got = fold_sink_lse(lse, sink)
    expect = torch.logaddexp(lse, sink.view(1, 2, 1))
    assert torch.allclose(got, expect)
    assert fold_sink_lse(lse, None) is lse


def test_bool_mask_slice_is_additive() -> None:
    from kernels.attention.flash_attn_gfx120x_ext import mask_as_additive, slice_attn_mask_kv

    mask = torch.tensor([[True, False, True], [True, True, False]])
    sliced = slice_attn_mask_kv(mask, 1, 3)
    add = mask_as_additive(sliced)
    assert add.shape == (2, 2)
    assert add[0, 0].item() == float("-inf")
    assert add[0, 1].item() == 0.0


def test_commit_caller_out_writes_and_rejects() -> None:
    from kernels.attention.flash_attn_gfx120x_ext import commit_caller_out

    produced = torch.ones(2, 3)
    assert commit_caller_out(None, produced) is produced
    buf = torch.empty(2, 3)
    got = commit_caller_out(buf, produced)
    assert got is buf
    assert torch.equal(buf, produced)
    with pytest.raises(ValueError, match="mismatch"):
        commit_caller_out(torch.empty(1, 3), produced)


def test_sink_alibi_and_splitk_extras_allowed() -> None:
    reject_sink_with_alibi(object(), object())
    reject_sink_with_alibi(object(), None)
    reject_splitk_extras(alibi_slopes=object(), return_lse=True)
    reject_splitk_extras()
