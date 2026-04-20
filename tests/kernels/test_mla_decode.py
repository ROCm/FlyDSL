# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Simplified MLA decode test for FlyDSL kernel.

Tests the FlyDSL MLA decode kernel (fp8 Q, fp8 KV, nhead=128, page_size=1)
using aiter for metadata generation and reduce.

Usage:
    cd /jruan/ws/FlyDSL
    python tests/kernels/test_mla_decode.py -b 1 -c 128
    python tests/kernels/test_mla_decode.py -b 32 -c 8192
"""

import sys
import os
import argparse
import logging

import torch
import pytest

sys.path.insert(0, "build-fly/python_packages")
sys.path.insert(1, ".")
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "1"
logging.basicConfig(level=logging.INFO, format="%(message)s")

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

aiter = pytest.importorskip("aiter", reason="aiter is not installed, skipping MLA tests")
from aiter import dtypes
from aiter.ops.attention import (
    get_mla_metadata_info_v1,
    get_mla_metadata_v1,
    mla_reduce_v1,
)

from flydsl.runtime.device import get_rocm_arch

from tests.test_common import run_perftest, checkAllclose
from kernels.mla_fwd_decode import flydsl_mla_fwd_decode

torch.set_default_device("cuda")

logger = logging.getLogger("mla_decode_test")

_GPU_ARCH = str(get_rocm_arch())

# ── Model constants (DeepSeek-V3 / R1) ──────────────────────────
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = 512
NHEAD = 128
NHEAD_KV = 1
PAGE_SIZE = 1


# ── Pure-PyTorch reference ──────────────────────────────────────

def ref_masked_attention(query, key, value, scale, dtype, q_scale=None, kv_scale=None):
    """Single-sequence MLA attention (no causal mask needed for decode_qlen=1)."""
    s = scale
    if q_scale is not None:
        s *= q_scale
    if kv_scale is not None:
        s *= kv_scale

    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * s
    lse = attn_weights.logsumexp(dim=-1)
    m = attn_weights.max(-1).values
    attn_weights_exp = torch.exp(attn_weights - m.unsqueeze(-1))
    l = attn_weights_exp.sum(-1)
    out = torch.einsum("hqk,khd->qhd", attn_weights_exp.float(), value.float())
    out = out / l.transpose(0, 1).unsqueeze(-1)
    if kv_scale is not None:
        out *= kv_scale
    return out.to(dtype), lse


def torch_mla_extend(q, kvc_cache, qo_indptr, kv_indptr, kv_indices,
                     kv_last_page_lens, sm_scale, dtype):
    """Pure-PyTorch paged MLA attention reference."""
    is_fp8_q = q.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    is_fp8_kvc = kvc_cache.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)

    if is_fp8_q:
        q = q.to(torch.float)
    if is_fp8_kvc:
        kvc_cache = kvc_cache.to(torch.float)

    num_page, page_size, nhead_kv, _ = kvc_cache.shape
    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1

    os_list = []
    lses = []
    for i in range(bs):
        cur_num_page = kvs[i].shape[0]
        real_kv_seq_len = (cur_num_page - 1) * page_size + kv_last_page_lens.tolist()[i]
        kvc_i = kvs[i].flatten(0, 1)[:real_kv_seq_len]
        q_i = qs[i]
        k_i = kvc_i
        v_i = kvc_i[:, :, :KV_LORA_RANK]
        o_i, lse_i = ref_masked_attention(q_i, k_i, v_i, sm_scale, dtype)
        os_list.append(o_i)
        lses.append(lse_i)

    o = torch.concat(os_list)
    lse = torch.concat(lses, dim=1).transpose(0, 1)
    return o, lse


# ── Test driver ─────────────────────────────────────────────────

def run_single(batch_size, ctx_len, decode_qlen=1, max_split_per_batch=32):
    nhead = NHEAD
    nhead_kv = NHEAD_KV
    page_size = PAGE_SIZE
    fp8 = dtypes.fp8
    out_dtype = torch.bfloat16

    kv_max_sz = 65536 * 32
    num_page = (kv_max_sz + page_size - 1) // page_size

    # ── Sequence metadata ──
    seq_lens_kv = torch.full((batch_size,), ctx_len, dtype=torch.int)
    kv_block_nums = torch.full((batch_size,), (ctx_len + page_size - 1) // page_size,
                               dtype=torch.int)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
    if ctx_len % page_size != 0:
        kv_last_page_lens.fill_(ctx_len % page_size)

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr[1:] = torch.cumsum(kv_block_nums, dim=0)
    num_page = kv_indptr[-1].item()
    kv_indices = torch.randperm(num_page, dtype=torch.int)

    seq_lens_qo = torch.full((batch_size,), decode_qlen, dtype=torch.int)
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    qo_indptr[1:] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = qo_indptr[-1].item()
    max_seqlen_qo = decode_qlen

    # ── KV buffer and Q ──
    kv_buffer = torch.randn((num_page, page_size, nhead_kv, QK_HEAD_DIM),
                            dtype=torch.bfloat16)
    kv_buffer_fp8 = kv_buffer.to(fp8)

    q = torch.randn((total_q, nhead, QK_HEAD_DIM), dtype=torch.bfloat16)
    q_fp8 = q.to(fp8)

    sm_scale = 1.0 / (QK_HEAD_DIM ** 0.5)

    # ── PyTorch reference (using fp8 data, cast to float internally) ──
    out_ref, lse_ref = torch_mla_extend(
        q_fp8, kv_buffer_fp8.view(num_page, page_size, nhead_kv, QK_HEAD_DIM),
        qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
        sm_scale, out_dtype,
    )

    # ── Limit splits for large nhead ──
    gpu = torch.cuda.current_device()
    cu_num = torch.cuda.get_device_properties(gpu).multi_processor_count
    max_split_per_batch = min(
        (cu_num + batch_size - 1) // batch_size, max_split_per_batch
    )

    # ── Metadata via aiter ──
    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = get_mla_metadata_info_v1(
        batch_size, max_seqlen_qo, nhead, fp8, fp8,
        is_sparse=False, fast_mode=True,
        num_kv_splits=max_split_per_batch,
        intra_batch_mode=False,
    )

    work_meta_data = torch.empty(work_meta_data_size, dtype=work_meta_data_type, device="cuda")
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device="cuda")
    work_info_set = torch.empty(work_info_set_size, dtype=work_info_set_type, device="cuda")
    reduce_indptr = torch.empty(reduce_indptr_size, dtype=reduce_indptr_type, device="cuda")
    reduce_final_map = torch.empty(reduce_final_map_size, dtype=reduce_final_map_type, device="cuda")
    reduce_partial_map = torch.empty(reduce_partial_map_size, dtype=reduce_partial_map_type, device="cuda")

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_lens,
        nhead // nhead_kv, nhead_kv, False,
        work_meta_data, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        kv_granularity=max(page_size, 16),
        max_seqlen_qo=int(max_seqlen_qo),
        uni_seqlen_qo=decode_qlen,
        fast_mode=True,
        max_split_per_batch=max_split_per_batch,
        intra_batch_mode=False,
        dtype_q=fp8,
        dtype_kv=fp8,
    )

    # ── Allocate output / partial buffers ──
    out_asm = torch.empty((total_q, nhead, V_HEAD_DIM), dtype=out_dtype).fill_(-1)

    logits = torch.empty(
        (reduce_partial_map.size(0) * max_seqlen_qo, 1, nhead, V_HEAD_DIM),
        dtype=torch.float32, device="cuda",
    )
    attn_lse = torch.empty(
        (reduce_partial_map.size(0) * max_seqlen_qo, 1, nhead, 1),
        dtype=torch.float32, device="cuda",
    )

    # ── Launch FlyDSL kernel ──
    def launch_kernel():
        flydsl_mla_fwd_decode(
            q_fp8,
            kv_buffer_fp8.view(num_page, page_size, nhead_kv, QK_HEAD_DIM),
            kv_indices,
            work_indptr,
            work_info_set,
            out_asm,
            logits,
            attn_lse,
            sm_scale,
        )
        mla_reduce_v1(
            logits, attn_lse,
            reduce_indptr, reduce_final_map, reduce_partial_map,
            max_seqlen_qo, out_asm, None,
        )

    _, us = run_perftest(launch_kernel, num_iters=10, num_warmup=3)
    torch.cuda.synchronize()

    # ── Verify ──
    total_kv = seq_lens_kv.sum().item()
    err = checkAllclose(
        out_ref, out_asm,
        msg=f"[b={batch_size} c={ctx_len}] golden vs flydsl: {us:>8.2f} us ... ",
    )

    # Cosine similarity check
    x, y = out_ref.double(), out_asm.double()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)

    flops = decode_qlen * total_kv * nhead * (QK_HEAD_DIM + V_HEAD_DIM) * 2
    bw = (
        total_kv * nhead_kv * QK_HEAD_DIM * 1  # fp8 = 1 byte
        + total_q * nhead * QK_HEAD_DIM * 1
        + total_q * nhead * V_HEAD_DIM * 2  # bf16 = 2 bytes
    )

    logger.info(
        f"  cos_diff={cos_diff:.2e}  TFLOPS={flops / us / 1e6:.2f}  "
        f"TB/s={bw / us / 1e6:.2f}  err_ratio={err:.2%}"
    )
    assert cos_diff < 3e-2, f"cos_diff={cos_diff} exceeds threshold"
    return err, us


# ── pytest ──────────────────────────────────────────────────────

# On gfx950, AITER folds nh=128 + fp8/fp8 to a nh=16 work-info layout
# instead of generating the native nh=128 layout. This FlyDSL kernel only
# decodes the native nh=128 layout, so it cannot run against AITER's
# gfx950 metadata.
@pytest.mark.skipif(
    _GPU_ARCH == "gfx950",
    reason=(
        "AITER metadata for nh=128 + fp8/fp8 on gfx950 uses the folded "
        "nh=16 layout, which this FlyDSL MLA kernel does not support."
    ),
)
@pytest.mark.parametrize("batch_size,ctx_len", [
    (1, 128),
    (4, 2048),
    (32, 8192),
])
def test_mla_decode(batch_size, ctx_len):
    run_single(batch_size, ctx_len)


# ── CLI (local benchmarking) ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FlyDSL MLA decode test")
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1, 32])
    parser.add_argument("-c", "--ctx_len", type=int, nargs="*", default=[128, 8192])
    parser.add_argument("-ms", "--max_splits", type=int, default=32)
    args = parser.parse_args()

    for b in args.batch:
        for c in args.ctx_len:
            logger.info(f"\n{'='*60}")
            logger.info(f"batch={b}  ctx_len={c}")
            logger.info(f"{'='*60}")
            run_single(b, c, max_split_per_batch=args.max_splits)

    logger.info("\nAll tests passed.")


if __name__ == "__main__":
    main()
