# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""V4.0 Gen.1 MLA decode regression harness.

Tests the FlyDSL ``kernels.mla_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1``
kernel (NoPE fp8 + RoPE bf16, nhead=128, page_size=1, gfx950) against
the aiter V4 silver reference (which carries the same fp8 quantization
noise the kernel pays at v_cvt_scalef32_pk_bf16_fp8).

Inputs to the kernel are the v4-packed buffers:
  * query     : [total_q, nhead, 576] fp8  (NoPE 448 + dup-E8M0 16 + pad 112)
  * query_rope: [total_q, nhead, 64]  bf16
  * kv_buffer : [num_page, page_size, 1, 576] fp8
  * kv_buffer_rope: [num_page, page_size, 1, 64] bf16

Uses the helpers in ``aiter/op_tests/test_mla_v4_persistent.py`` for
quantization + packing + reference attention so the silver path here
matches aiter's reference exactly.
"""

from __future__ import annotations

import logging
import os
import sys

import pytest
import torch

sys.path.insert(0, "build-fly/python_packages")
sys.path.insert(1, ".")
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

logging.basicConfig(level=logging.INFO, format="%(message)s")
pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

aiter = pytest.importorskip("aiter", reason="aiter not installed; skipping v40 tests")
from aiter import dtypes  # noqa: E402

# V4 layout constants + quantize/pack/reference helpers, copied from
# aiter's op_tests/test_mla_v4_persistent.py (importing it directly is
# not possible -- its module body runs argparse and SystemExits).
V4_DIM_NOPE = 448
V4_DIM_ROPE = 64
V4_DIM_QK = V4_DIM_NOPE + V4_DIM_ROPE       # 512
V4_TILE = 64
V4_NUM_TILES = V4_DIM_NOPE // V4_TILE       # 7
V4_DIM_SCALE = V4_NUM_TILES + 1             # 8 (bpad8: 7 active + 1 pad)
V4_DIM_QK_PACKED = 576
V4_DIM_SCALE_DUP = V4_DIM_SCALE * 2         # 16
V4_PACK_OFF_NOPE = 0
V4_PACK_OFF_SCALE = V4_DIM_NOPE             # 448


def _fp32_pow2_to_e8m0(pow2_fp32: torch.Tensor) -> torch.Tensor:
    safe = torch.where(pow2_fp32 > 0, pow2_fp32, torch.ones_like(pow2_fp32))
    biased = torch.log2(safe).round().to(torch.int32) + 127
    biased = torch.clamp(biased, 0, 254)
    biased = torch.where(pow2_fp32 > 0, biased, torch.zeros_like(biased))
    return biased.to(torch.uint8)


def _e8m0_to_fp32(byte: torch.Tensor) -> torch.Tensor:
    b = byte.to(torch.int32)
    return torch.where(
        b == 0, torch.zeros_like(b, dtype=torch.float32),
        torch.where(
            b == 255, torch.full_like(b, float("inf"), dtype=torch.float32),
            torch.exp2((b - 127).to(torch.float32)),
        ),
    )


def _cast_scale_inv_to_ue8m0_pow2(scales_inv: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, torch.clamp_min(scales_inv, 1e-4).log2().ceil()).to(
        torch.float32,
    )


def quantize_v4_nope_bpad8(nope_fp32: torch.Tensor):
    fp8_amax = float(torch.finfo(dtypes.fp8).max)
    leading = nope_fp32.shape[:-1]
    tiled = nope_fp32.reshape(*leading, V4_NUM_TILES, V4_TILE)
    active_scale_pow2 = _cast_scale_inv_to_ue8m0_pow2(
        tiled.abs().amax(dim=-1) / fp8_amax,
    )
    nope_fp8 = (
        (tiled / active_scale_pow2.unsqueeze(-1)).to(dtypes.fp8)
        .reshape(*leading, V4_DIM_NOPE)
    )
    active_scale_e8m0 = _fp32_pow2_to_e8m0(active_scale_pow2)
    scale_e8m0 = torch.zeros(
        (*leading, V4_DIM_SCALE), dtype=torch.uint8, device=nope_fp32.device,
    )
    scale_e8m0[..., :V4_NUM_TILES] = active_scale_e8m0
    return nope_fp8, scale_e8m0


def quantize_v4_q(q: torch.Tensor):
    q_nope_fp32 = q[..., :V4_DIM_NOPE].float()
    q_rope_bf16 = q[..., V4_DIM_NOPE:].to(torch.bfloat16)
    q_nope_fp8, q_nope_scale_e8m0 = quantize_v4_nope_bpad8(q_nope_fp32)
    return q_nope_fp8, q_nope_scale_e8m0, q_rope_bf16


def _duplicate_each_lastdim(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(-1).expand(*x.shape, 2).reshape(*x.shape[:-1], x.shape[-1] * 2)


def pack_v4_nope_scale(nope_fp8: torch.Tensor, scale_e8m0_bpad8: torch.Tensor):
    leading = nope_fp8.shape[:-1]
    assert nope_fp8.shape[-1] == V4_DIM_NOPE
    assert scale_e8m0_bpad8.shape[-1] == V4_DIM_SCALE
    packed = torch.zeros(
        (*leading, V4_DIM_QK_PACKED), dtype=torch.uint8, device=nope_fp8.device,
    )
    packed[..., V4_PACK_OFF_NOPE:V4_PACK_OFF_NOPE + V4_DIM_NOPE] = nope_fp8.view(
        torch.uint8,
    )
    packed[..., V4_PACK_OFF_SCALE:V4_PACK_OFF_SCALE + V4_DIM_SCALE_DUP] = (
        _duplicate_each_lastdim(scale_e8m0_bpad8)
    )
    return packed.view(dtypes.fp8)


def init_v4_kv_cache(num_page: int, page_size: int):
    nope_fp32 = torch.randn(
        (num_page, page_size, 1, V4_DIM_NOPE), dtype=torch.float32,
    )
    rope_bf16 = torch.randn(
        (num_page, page_size, 1, V4_DIM_ROPE), dtype=torch.bfloat16,
    )
    nope_fp8, scale_e8m0 = quantize_v4_nope_bpad8(nope_fp32)
    return nope_fp8, scale_e8m0, rope_bf16


def _v4_dequant_nope_bpad8(nope_fp8, scale_e8m0):
    leading = nope_fp8.shape[:-1]
    active_scale = _e8m0_to_fp32(scale_e8m0[..., :V4_NUM_TILES])
    return (
        (nope_fp8.to(torch.float32).reshape(*leading, V4_NUM_TILES, V4_TILE)
         * active_scale.unsqueeze(-1))
        .reshape(*leading, V4_DIM_NOPE).to(torch.bfloat16)
    )


def torch_mla_extend_v4_silver(
    q_nope_fp8, q_nope_scale, q_rope_bf16,
    kv_nope_fp8, kv_nope_scale, kv_rope_bf16,
    qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
    sm_scale, out_dtype,
):
    """Silver reference: matches the kernel's fp8 round-trip exactly.

    Dequantizes the NoPE FP8 + E8M0 scale into BF16 (this is what the
    kernel sees after v_cvt_scalef32_pk_bf16_fp8), concatenates with the
    BF16 RoPE, then runs standard bf16 attention.
    """
    q_nope_bf16 = _v4_dequant_nope_bpad8(q_nope_fp8, q_nope_scale)
    q_silver = torch.cat([q_nope_bf16, q_rope_bf16], dim=-1)  # [total_q, h, 512]
    kv_nope_bf16 = _v4_dequant_nope_bpad8(kv_nope_fp8, kv_nope_scale)
    kv_silver = torch.cat([kv_nope_bf16, kv_rope_bf16], dim=-1)  # [page, ps, 1, 512]

    qs = torch.tensor_split(q_silver, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kv_silver, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    page_size = kv_silver.shape[1]
    bs = qo_indptr.shape[0] - 1

    outs = []
    for i in range(bs):
        cur_pages = kvs[i].shape[0]
        real_kv_len = (cur_pages - 1) * page_size + int(kv_last_page_lens[i].item())
        kvi = kvs[i].flatten(0, 1)[:real_kv_len]  # [s_k, 1, 512]
        q_i = qs[i]  # [s_q, h, 512]
        # K and V both use the full d_qk slice in v4.
        attn = torch.einsum("qhd,khd->hqk", q_i.float(), kvi.float()) * sm_scale
        m = attn.max(dim=-1, keepdim=True).values
        ae = torch.exp(attn - m)
        s = ae.sum(-1, keepdim=True)
        out = torch.einsum("hqk,khd->qhd", ae / s, kvi.float())
        outs.append(out.to(out_dtype))
    return torch.concat(outs)

from aiter.ops.attention import (  # noqa: E402
    get_mla_metadata_info_v1,
    get_mla_metadata_v1,
)

from kernels.mla_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1 import (  # noqa: E402
    NUM_QO_HEADS,
    OCCUPANCY,
    QK_HEAD_DIM,
    QK_PACKED_NOPE_BYTES,
    QK_ROPE_HEAD_DIM,
    V_HEAD_DIM,
    launch_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1,
)

torch.set_default_device("cuda")

NHEAD = NUM_QO_HEADS         # 128
NHEAD_KV = 1
PAGE_SIZE = 1

assert QK_PACKED_NOPE_BYTES == V4_DIM_QK_PACKED == 576
assert QK_ROPE_HEAD_DIM == V4_DIM_ROPE == 64
assert V_HEAD_DIM == V4_DIM_QK == 512


# ---------------------------------------------------------------------------
# Per-CU resource helpers (LDS budget queries)
# ---------------------------------------------------------------------------
def _lds_size_per_cu(arch: str) -> int:
    # gfx950 = 160 KiB / CU, gfx94x = 64 KiB / CU.  V40 needs gfx950.
    return 160 * 1024 if arch.startswith("gfx95") else 64 * 1024


def _gcn_arch_base(arch_str: str) -> str:
    # e.g. "gfx950" or "gfx950:sramecc+:xnack-" -> "gfx950"
    return arch_str.split(":")[0]


# ---------------------------------------------------------------------------
# Test harness: one config -> kernel output vs silver reference
# ---------------------------------------------------------------------------
def _run_v40_decode(batch_size: int, ctx_len: int, *, seed: int = 0):
    torch.manual_seed(seed + batch_size * 1000003 + ctx_len * 7919)
    decode_qlen = 1
    max_seqlen_qo = decode_qlen
    out_dtype = torch.bfloat16

    # -- Sequence metadata --
    seq_lens_kv = torch.full((batch_size,), ctx_len, dtype=torch.int)
    kv_block_nums = torch.full(
        (batch_size,), (ctx_len + PAGE_SIZE - 1) // PAGE_SIZE, dtype=torch.int,
    )
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
    if ctx_len % PAGE_SIZE != 0:
        kv_last_page_lens.fill_(ctx_len % PAGE_SIZE)

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr[1:] = torch.cumsum(kv_block_nums, dim=0)
    num_page = int(kv_indptr[-1].item())
    kv_indices = torch.randperm(num_page, dtype=torch.int)

    seq_lens_qo = torch.full((batch_size,), decode_qlen, dtype=torch.int)
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    qo_indptr[1:] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = int(qo_indptr[-1].item())

    # -- Random Q (fp32 source -> v4-packed FP8 + BF16 rope) --
    q_fp32 = torch.randn((total_q, NHEAD, V4_DIM_QK), dtype=torch.float32)
    q_nope_fp8, q_nope_scale_e8m0, q_rope_bf16 = quantize_v4_q(q_fp32)
    q_packed = pack_v4_nope_scale(q_nope_fp8, q_nope_scale_e8m0)
    q_rope_bf16 = q_rope_bf16.contiguous()

    # -- Random KV (paged) --
    kv_nope_fp8, kv_nope_scale_e8m0, kv_rope_bf16 = init_v4_kv_cache(
        num_page, PAGE_SIZE,
    )
    kv_packed = pack_v4_nope_scale(kv_nope_fp8, kv_nope_scale_e8m0)

    sm_scale = 1.0 / (V4_DIM_QK ** 0.5)

    # -- Silver reference (fp8 round-tripped) --
    out_ref = torch_mla_extend_v4_silver(
        q_nope_fp8, q_nope_scale_e8m0, q_rope_bf16,
        kv_nope_fp8, kv_nope_scale_e8m0, kv_rope_bf16,
        qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
        sm_scale, out_dtype,
    )

    # -- Metadata via aiter --
    # Force 1 split per batch so the kernel writes the FULL softmax result
    # directly to final_output (partial_qo_loc = -1 sentinel).  Multi-split
    # output would require running the reduce kernel afterward to merge
    # partial fp32 outputs -- not wired in stage 13 yet.
    gpu = torch.cuda.current_device()
    cu_num = torch.cuda.get_device_properties(gpu).multi_processor_count
    max_split_per_batch = 1

    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = get_mla_metadata_info_v1(
        batch_size, max_seqlen_qo, NHEAD,
        dtypes.fp8, dtypes.fp8,
        is_sparse=False, fast_mode=True,
        num_kv_splits=max_split_per_batch,
        intra_batch_mode=False,
    )
    work_meta_data = torch.empty(work_meta_data_size, dtype=work_meta_data_type)
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type)
    work_info_set = torch.empty(work_info_set_size, dtype=work_info_set_type)
    reduce_indptr = torch.empty(reduce_indptr_size, dtype=reduce_indptr_type)
    reduce_final_map = torch.empty(reduce_final_map_size, dtype=reduce_final_map_type)
    reduce_partial_map = torch.empty(reduce_partial_map_size, dtype=reduce_partial_map_type)
    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_lens,
        NHEAD // NHEAD_KV, NHEAD_KV, False,
        work_meta_data, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=int(max_seqlen_qo),
        uni_seqlen_qo=decode_qlen,
        fast_mode=True,
        max_split_per_batch=max_split_per_batch,
        intra_batch_mode=False,
        dtype_q=dtypes.fp8, dtype_kv=dtypes.fp8,
    )

    # -- Output / partial buffers --
    final_output = torch.empty((1, total_q, NHEAD, V_HEAD_DIM),
                               dtype=out_dtype).fill_(-1)
    split_output = torch.empty(
        (1, reduce_partial_map.size(0) * max_seqlen_qo, NHEAD, V_HEAD_DIM),
        dtype=torch.float32,
    )
    split_lse = torch.empty(
        (1, reduce_partial_map.size(0) * max_seqlen_qo, NHEAD, 1),
        dtype=torch.float32,
    )

    # -- LDS / CU budget --
    arch = _gcn_arch_base(torch.cuda.get_device_properties(gpu).gcnArchName)
    lds_size = _lds_size_per_cu(arch) // OCCUPANCY

    # -- Launch FlyDSL kernel --
    launch_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1(
        q_packed.view(total_q, NHEAD, V4_DIM_QK_PACKED),
        q_rope_bf16.view(total_q, NHEAD, V4_DIM_ROPE),
        kv_packed.view(num_page, PAGE_SIZE, NHEAD_KV, V4_DIM_QK_PACKED),
        kv_rope_bf16.view(num_page, PAGE_SIZE, NHEAD_KV, V4_DIM_ROPE),
        kv_indices,
        kv_last_page_lens,
        work_indptr,
        work_info_set,
        final_output,
        split_output,
        split_lse,
        sm_scale,
        7,  # log2(NHEAD) = log2(128)
        num_cus=cu_num,
        lds_size=lds_size,
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    return final_output[0], out_ref


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
V40_CONFIGS = [
    # Cover single-tile and multi-tile paths.
    (1, 32),       # single full tile
    (1, 31),       # single partial tile (boundary mask exercise)
    (1, 64),       # two full tiles
    (1, 128),      # 4 full tiles
    (1, 200),      # 7 tiles, last partial
    (4, 2048),     # 4 batches, long-context
]


@pytest.mark.parametrize("batch_size,ctx_len", V40_CONFIGS)
def test_mla_v40_decode_accuracy(batch_size: int, ctx_len: int):
    out_kernel, out_ref = _run_v40_decode(batch_size, ctx_len)
    # Numerical tolerance: fp8-quantized MLA inherits the v_cvt_scalef32
    # rounding error.  aiter uses rtol=0.04, atol=0.005 for v4 silver checks.
    torch.testing.assert_close(
        out_kernel.float(), out_ref.float(),
        rtol=0.04, atol=0.005,
    )


if __name__ == "__main__":
    # Smoke run for manual debugging: smallest config.
    out_kernel, out_ref = _run_v40_decode(1, 32)
    mae = (out_kernel.float() - out_ref.float()).abs().mean().item()
    max_abs = (out_kernel.float() - out_ref.float()).abs().max().item()
    print(f"V40 decode b=1 ctx=32: mae={mae:.6f}, max_abs={max_abs:.6f}")
    print(f"  out_kernel[0,0,:8] = {out_kernel[0, 0, :8].tolist()}")
    print(f"  out_ref[0,0,:8]    = {out_ref[0, 0, :8].tolist()}")
