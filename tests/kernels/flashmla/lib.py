# Adapted from https://github.com/deepseek-ai/FlashMLA/tree/main/tests/lib.py
import dataclasses
import random
from typing import List, Optional

import torch


@dataclasses.dataclass
class ExtraTestParamForDecode:
    b: int
    is_varlen: bool
    have_zero_seqlen_k: bool
    extra_s_k: Optional[int] = None
    extra_topk: Optional[int] = None
    block_size: int = 64
    extra_block_size: Optional[int] = None
    have_extra_topk_length: bool = False


@dataclasses.dataclass
class TestParam:
    s_q: int
    s_kv: int
    topk: int
    h_q: int = 128
    h_kv: int = 1
    d_qk: int = 512
    d_v: int = 512
    seed: int = -1
    check_correctness: bool = True
    is_all_indices_invalid: bool = False
    num_runs: int = 10
    have_attn_sink: bool = False
    have_topk_length: bool = False
    decode: Optional[ExtraTestParamForDecode] = None


@dataclasses.dataclass
class RawTestParamForDecode:
    b: int
    h_q: int
    s_q: int
    h_kv: int
    s_kv: int
    is_varlen: bool
    topk: int
    is_all_indices_invalid: bool = False
    have_zero_seqlen_k: bool = False
    have_topk_length: bool = False
    enable_attn_sink: bool = True
    extra_s_k: Optional[int] = None
    extra_topk: Optional[int] = None
    block_size: int = 64
    extra_block_size: Optional[int] = None
    have_extra_topk_length: bool = False
    d_qk: int = 576
    d_v: int = 512
    check_correctness: bool = True
    num_runs: int = 10
    seed: int = -1

    def to_test_param(self) -> TestParam:
        return TestParam(
            self.s_q, self.s_kv, self.topk, self.h_q, self.h_kv, self.d_qk, self.d_v,
            self.seed, self.check_correctness,
            self.is_all_indices_invalid,
            self.num_runs,
            self.enable_attn_sink,
            self.have_topk_length,
            decode=ExtraTestParamForDecode(
                self.b, self.is_varlen, self.have_zero_seqlen_k,
                self.extra_s_k, self.extra_topk,
                self.block_size, self.extra_block_size, self.have_extra_topk_length,
            ),
        )


@dataclasses.dataclass
class Testcase:
    p: TestParam
    dOut: torch.Tensor
    q: torch.Tensor
    kv: torch.Tensor
    indices: torch.Tensor
    sm_scale: float
    attn_sink: Optional[torch.Tensor]
    topk_length: Optional[torch.Tensor]


@dataclasses.dataclass
class KVScope:
    t: TestParam
    cache_seqlens: torch.Tensor
    block_table: torch.Tensor
    blocked_k: torch.Tensor
    abs_indices: torch.Tensor
    indices_in_kvcache: torch.Tensor
    topk_length: Optional[torch.Tensor]

    def apply_perm(self, perm: torch.Tensor) -> "KVScope":
        return KVScope(
            self.t,
            self.cache_seqlens[perm],
            self.block_table[perm],
            self.blocked_k,
            self.abs_indices[perm],
            self.indices_in_kvcache[perm],
            self.topk_length[perm] if self.topk_length is not None else None,
        )


@dataclasses.dataclass
class TestcaseForDecode:
    p: TestParam
    q: torch.Tensor
    attn_sink: Optional[torch.Tensor]
    sm_scale: float
    kv_scope: KVScope
    extra_kv_scope: Optional[KVScope]


def _set_random_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def _non_contiguousify(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(0, 0)


def _randperm_batch(batch_size: int, perm_range: torch.Tensor, perm_size: int, paddings: List[int]) -> torch.Tensor:
    assert not torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    device = perm_range.device
    perm_range_max = max(int(torch.max(perm_range).item()), perm_size)
    rand = torch.rand(batch_size, perm_range_max, dtype=torch.float32, device=device)
    rand[torch.arange(0, perm_range_max, device=device).broadcast_to(batch_size, perm_range_max) >= perm_range.view(batch_size, 1)] = float("-inf")
    res = rand.topk(perm_size, dim=-1, sorted=True).indices.to(torch.int32)
    res[res >= perm_range.view(batch_size, 1)] = paddings[0]
    torch.use_deterministic_algorithms(False)
    return res


def _abs_indices2indices_in_kvcache(abs_indices: torch.Tensor, block_table: torch.Tensor, block_size: int) -> torch.Tensor:
    flat_indices = abs_indices.reshape(-1).to(torch.int64)
    valid = flat_indices >= 0
    safe_indices = flat_indices.clamp_min(0)
    logical_block = safe_indices // block_size
    block_offset = safe_indices % block_size
    physical_block = block_table.reshape(-1).index_select(0, logical_block)
    result = physical_block * block_size + block_offset
    result[~valid] = -1
    return result.reshape_as(abs_indices).to(torch.int32)


def generate_testcase_for_decode(t: TestParam) -> TestcaseForDecode:
    _set_random_seed(t.seed)
    assert t.h_q % t.h_kv == 0
    assert t.decode is not None

    q = torch.randn((t.decode.b, t.s_q, t.h_q, t.d_qk))
    q.clamp_(min=-1.0, max=1.0)

    attn_sink = None
    if t.have_attn_sink:
        attn_sink = torch.randn((t.h_q,), dtype=torch.float32)
        inf_mask = torch.randn((t.h_q,), dtype=torch.float32)
        attn_sink[inf_mask > 0.5] = float("inf")
        attn_sink[inf_mask < -0.5] = float("-inf")

    def generate_one_k_scope(
        s_k: int,
        block_size: int,
        topk: int,
        is_varlen: bool,
        have_zero_seqlen: bool,
        is_all_indices_invalid: bool,
        have_topk_length: bool,
    ) -> KVScope:
        b = t.decode.b
        device = q.device
        cache_seqlens_cpu = torch.full((b,), s_k, dtype=torch.int32, device="cpu")
        if is_varlen:
            for i in range(b):
                cache_seqlens_cpu[i] = max(random.normalvariate(s_k, s_k / 2), t.s_q)

        if have_zero_seqlen:
            zeros_mask = torch.randn(b, dtype=torch.float32, device="cpu") > 0
            cache_seqlens_cpu[zeros_mask] = 0

        max_seqlen_alignment = 4 * block_size
        max_seqlen_pad = max(_cdiv(int(cache_seqlens_cpu.max().item()), max_seqlen_alignment), 1) * max_seqlen_alignment
        cache_seqlens = cache_seqlens_cpu.to(device)

        assert max_seqlen_pad % block_size == 0
        block_table = torch.arange(
            b * max_seqlen_pad // block_size, dtype=torch.int32, device=device
        ).view(b, max_seqlen_pad // block_size)
        block_table = block_table.view(-1)[torch.randperm(block_table.numel(), device=device)].view(b, -1)

        blocked_k = torch.randn(
            (block_table.numel(), block_size, t.h_kv, t.d_qk),
            dtype=torch.bfloat16,
            device=device,
        ) / 10
        blocked_k.clamp_(min=-1.0, max=1.0)

        abs_indices = torch.empty((b, t.s_q, topk), dtype=torch.int32, device=device)
        if is_all_indices_invalid:
            abs_indices.fill_(-1)
        else:
            abs_indices[:] = _randperm_batch(
                b * t.s_q, cache_seqlens.repeat_interleave(t.s_q), topk, [-1]
            ).view(b, t.s_q, topk)
        indices_in_kvcache = _abs_indices2indices_in_kvcache(abs_indices, block_table, block_size)

        topk_length = torch.randint(0, topk + 1, (b,), dtype=torch.int32, device=device) if have_topk_length else None

        if have_topk_length:
            indices_in_kvcache_masked = indices_in_kvcache.clone()
            indices_in_kvcache_masked[
                torch.arange(0, topk, device=device).view(1, 1, topk).broadcast_to(b, t.s_q, topk) >= topk_length.view(b, 1, 1)
            ] = -1
        else:
            indices_in_kvcache_masked = indices_in_kvcache

        blocked_k = blocked_k.view(-1, t.h_kv, t.d_qk)
        nonused_indices_mask = torch.ones(blocked_k.size(0) * blocked_k.size(1), dtype=torch.bool, device=device)
        valid_indices = indices_in_kvcache_masked[indices_in_kvcache_masked >= 0].to(torch.int64)
        nonused_indices_mask[valid_indices] = False
        blocked_k[nonused_indices_mask, :, :] = float("nan")
        blocked_k = blocked_k.view(-1, block_size, t.h_kv, t.d_qk)

        return KVScope(
            t,
            cache_seqlens,
            _non_contiguousify(block_table),
            blocked_k,
            _non_contiguousify(abs_indices),
            _non_contiguousify(indices_in_kvcache),
            topk_length,
        )

    kv_scope0 = generate_one_k_scope(
        t.s_kv,
        t.decode.block_size,
        t.topk,
        t.decode.is_varlen,
        t.decode.have_zero_seqlen_k,
        t.is_all_indices_invalid,
        t.have_topk_length,
    )
    if t.decode.extra_topk is not None:
        if t.decode.extra_s_k is None:
            t.decode.extra_s_k = t.decode.extra_topk * 2
        if t.decode.extra_block_size is None:
            t.decode.extra_block_size = t.decode.block_size
        kv_scope1 = generate_one_k_scope(
            t.decode.extra_s_k,
            t.decode.extra_block_size,
            t.decode.extra_topk,
            t.decode.is_varlen,
            t.decode.have_zero_seqlen_k,
            t.is_all_indices_invalid,
            t.decode.have_extra_topk_length,
        )
    else:
        assert t.decode.extra_block_size is None and t.decode.extra_s_k is None and not t.decode.have_extra_topk_length
        kv_scope1 = None

    sm_scale = t.d_qk ** -0.55

    q = _non_contiguousify(q)
    return TestcaseForDecode(t, q, attn_sink, sm_scale, kv_scope0, kv_scope1)
