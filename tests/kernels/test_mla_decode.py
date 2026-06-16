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

import argparse
import logging
import os
import statistics
import sys

import pytest
import torch

sys.path.insert(0, "build-fly/python_packages")
sys.path.insert(1, ".")
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "1"
logging.basicConfig(level=logging.INFO, format="%(message)s")

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

aiter = pytest.importorskip("aiter", reason="aiter is not installed, skipping MLA tests")
from aiter import dtypes  # noqa: E402  # pyright: ignore[reportMissingImports]
from aiter.ops.attention import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    get_mla_metadata_info_v1,
    get_mla_metadata_v1,
    hk_mla_decode_fwd,
    mla_decode_stage1_asm_fwd,
    mla_reduce_v1,
)  # noqa: E402

from kernels.mla_fwd_decode import flydsl_mla_fwd_decode  # noqa: E402
from tests.test_common import checkAllclose, run_perftest  # noqa: E402

torch.set_default_device("cuda")

logger = logging.getLogger("mla_decode_test")

# -- Model constants (DeepSeek-V3 / R1) --------------------------
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = 512
NHEAD = 128
NHEAD_KV = 1
PAGE_SIZE = 1

MLA_DECODE_BENCH_CONFIGS = [
    # Cover gfx950 BLOCK_N=64 single-tile paths before the larger multi-tile cases.
    (1, 63),
    (1, 64),
    (1, 128),
    (4, 2048),
    (33, 2333),
    (32, 8192),
]

DEFAULT_BENCH_WARMUP = 20
DEFAULT_BENCH_ITERS = 100
DEFAULT_BENCH_REPEATS = 3
DEFAULT_SEED = 0


# -- Pure-PyTorch reference --------------------------------------


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
    denom = attn_weights_exp.sum(-1)
    out = torch.einsum("hqk,khd->qhd", attn_weights_exp.float(), value.float())
    out = out / denom.transpose(0, 1).unsqueeze(-1)
    if kv_scale is not None:
        out *= kv_scale
    return out.to(dtype), lse


def torch_mla_extend(q, kvc_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens, sm_scale, dtype):
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


# -- Test driver -------------------------------------------------


def _filter_iqr(values):
    """Drop obvious timing outliers, keeping original data if filtering is too aggressive."""
    if len(values) < 8:
        return values
    ordered = sorted(values)
    q1 = ordered[len(ordered) // 4]
    q3 = ordered[(len(ordered) * 3) // 4]
    iqr = q3 - q1
    if iqr <= 0:
        return values
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    filtered = [v for v in values if lo <= v <= hi]
    return filtered if len(filtered) >= max(3, len(values) // 2) else values


def _profile_benchmark(func, *, num_iters, num_warmup, repeats):
    """Run profiler timing multiple times and report a stable median latency.

    Keep the same timing backend as the original test (`run_perftest`) because
    it measures the actual GPU kernel launched by FlyDSL. Repeating the profiler
    run and taking the median reduces run-to-run clock/cache noise without
    changing the reported TFLOPS definition.
    """
    repeat_us = []
    last_data = None

    for _ in range(repeats):
        last_data, us = run_perftest(
            func,
            num_iters=num_iters,
            num_warmup=num_warmup,
        )
        repeat_us.append(us)

    filtered = _filter_iqr(repeat_us)

    return last_data, {
        "us": statistics.median(filtered),
        "repeat_us": repeat_us,
        "min_us": min(filtered),
        "max_us": max(filtered),
    }


def run_single(
    batch_size,
    ctx_len,
    decode_qlen=1,
    max_split_per_batch=32,
    bench_iters=DEFAULT_BENCH_ITERS,
    bench_warmup=DEFAULT_BENCH_WARMUP,
    bench_repeats=DEFAULT_BENCH_REPEATS,
    seed=DEFAULT_SEED,
    bench_aiter=False,
):
    torch.manual_seed(seed + batch_size * 1000003 + ctx_len * 9176 + decode_qlen)

    nhead = NHEAD
    nhead_kv = NHEAD_KV
    page_size = PAGE_SIZE
    fp8 = dtypes.fp8
    out_dtype = torch.bfloat16

    kv_max_sz = 65536 * 32
    num_page = (kv_max_sz + page_size - 1) // page_size

    # -- Sequence metadata --
    seq_lens_kv = torch.full((batch_size,), ctx_len, dtype=torch.int)
    kv_block_nums = torch.full((batch_size,), (ctx_len + page_size - 1) // page_size, dtype=torch.int)
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

    # -- KV buffer and Q --
    kv_buffer = torch.randn((num_page, page_size, nhead_kv, QK_HEAD_DIM), dtype=torch.bfloat16)
    kv_buffer_fp8 = kv_buffer.to(fp8)

    q = torch.randn((total_q, nhead, QK_HEAD_DIM), dtype=torch.bfloat16)
    q_fp8 = q.to(fp8)

    sm_scale = 1.0 / (QK_HEAD_DIM**0.5)

    # -- PyTorch reference (using fp8 data, cast to float internally) --
    out_ref, lse_ref = torch_mla_extend(
        q_fp8,
        kv_buffer_fp8.view(num_page, page_size, nhead_kv, QK_HEAD_DIM),
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        sm_scale,
        out_dtype,
    )

    # -- Limit splits for large nhead --
    gpu = torch.cuda.current_device()
    cu_num = torch.cuda.get_device_properties(gpu).multi_processor_count
    max_split_per_batch = min((cu_num + batch_size - 1) // batch_size, max_split_per_batch)

    # -- Metadata via aiter --
    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = get_mla_metadata_info_v1(
        batch_size,
        max_seqlen_qo,
        nhead,
        fp8,
        fp8,
        is_sparse=False,
        fast_mode=True,
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
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        nhead // nhead_kv,
        nhead_kv,
        False,
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        kv_granularity=max(page_size, 16),
        max_seqlen_qo=int(max_seqlen_qo),
        uni_seqlen_qo=decode_qlen,
        fast_mode=True,
        max_split_per_batch=max_split_per_batch,
        intra_batch_mode=False,
        dtype_q=fp8,
        dtype_kv=fp8,
    )

    # -- Allocate output / partial buffers --
    out_asm = torch.empty((total_q, nhead, V_HEAD_DIM), dtype=out_dtype).fill_(-1)

    logits = torch.empty(
        (reduce_partial_map.size(0) * max_seqlen_qo, 1, nhead, V_HEAD_DIM),
        dtype=torch.float32,
        device="cuda",
    )
    attn_lse = torch.empty(
        (reduce_partial_map.size(0) * max_seqlen_qo, 1, nhead, 1),
        dtype=torch.float32,
        device="cuda",
    )

    # -- Launch FlyDSL kernel --
    def launch_decode():
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

    def launch_reduce():
        mla_reduce_v1(
            logits,
            attn_lse,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            max_seqlen_qo,
            max_split_per_batch,
            out_asm,
            None,
        )

    def _bench_decode_stage1(name, launch_stage1, out_tensor, logits_tensor, lse_tensor):
        _, stats = _profile_benchmark(
            launch_stage1,
            num_iters=bench_iters,
            num_warmup=bench_warmup,
            repeats=bench_repeats,
        )
        mla_reduce_v1(
            logits_tensor,
            lse_tensor,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            max_seqlen_qo,
            max_split_per_batch,
            out_tensor,
            None,
        )
        torch.cuda.synchronize()
        err_stage1 = checkAllclose(
            out_ref,
            out_tensor,
            msg=f"[b={batch_size} c={ctx_len}] golden vs {name}: {stats['us']:>8.2f} us ... ",
        )
        x_stage1, y_stage1 = out_ref.double(), out_tensor.double()
        cos_stage1 = 1 - 2 * (x_stage1 * y_stage1).sum().item() / max(
            (x_stage1 * x_stage1 + y_stage1 * y_stage1).sum().item(), 1e-12
        )
        logger.info(
            f"  {name}: cos_diff={cos_stage1:.2e}  TFLOPS={flops / stats['us'] / 1e6:.2f}  "
            f"TB/s={bw / stats['us'] / 1e6:.2f}  err_ratio={err_stage1:.2%}  "
            f"us_p50={stats['us']:.2f}  us_range=[{stats['min_us']:.2f}, {stats['max_us']:.2f}]"
        )
        assert cos_stage1 < 3e-2, f"{name} cos_diff={cos_stage1} exceeds threshold"
        return err_stage1, stats["us"]

    _, bench_stats = _profile_benchmark(
        launch_decode,
        num_iters=bench_iters,
        num_warmup=bench_warmup,
        repeats=bench_repeats,
    )
    us = bench_stats["us"]
    launch_reduce()
    torch.cuda.synchronize()

    # -- Verify --
    total_kv = seq_lens_kv.sum().item()
    err = checkAllclose(
        out_ref,
        out_asm,
        msg=f"[b={batch_size} c={ctx_len}] golden vs flydsl decode-only: {us:>8.2f} us ... ",
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
        f"TB/s={bw / us / 1e6:.2f}  err_ratio={err:.2%}  "
        f"us_p50={us:.2f}  us_range=[{bench_stats['min_us']:.2f}, {bench_stats['max_us']:.2f}]"
    )
    assert cos_diff < 3e-2, f"cos_diff={cos_diff} exceeds threshold"

    if bench_aiter:
        aiter_hk_out = torch.empty_like(out_asm).fill_(-1)
        aiter_hk_logits = torch.empty_like(logits)
        aiter_hk_lse = torch.empty_like(attn_lse)

        def launch_aiter_hk():
            hk_mla_decode_fwd(
                q_fp8,
                kv_buffer_fp8.view(num_page, page_size, nhead_kv, QK_HEAD_DIM),
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_lens,
                work_indptr,
                work_info_set,
                max_seqlen_qo,
                sm_scale,
                aiter_hk_logits,
                aiter_hk_lse,
                aiter_hk_out,
            )

        _bench_decode_stage1("aiter.hk_mla_decode_fwd", launch_aiter_hk, aiter_hk_out, aiter_hk_logits, aiter_hk_lse)

        aiter_asm_out = torch.empty_like(out_asm).fill_(-1)
        aiter_asm_logits = torch.empty_like(logits)
        aiter_asm_lse = torch.empty_like(attn_lse)
        aiter_asm_q_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
        aiter_asm_kv_scale = torch.ones((1,), dtype=torch.float32, device="cuda")

        def launch_aiter_asm():
            mla_decode_stage1_asm_fwd(
                q_fp8,
                kv_buffer_fp8.view(num_page, page_size, nhead_kv, QK_HEAD_DIM),
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_lens,
                None,
                work_meta_data,
                work_indptr,
                work_info_set,
                max_seqlen_qo,
                page_size,
                nhead_kv,
                sm_scale,
                aiter_asm_logits,
                aiter_asm_lse,
                aiter_asm_out,
                None,
                aiter_asm_q_scale,
                aiter_asm_kv_scale,
            )

        _bench_decode_stage1(
            "aiter.mla_decode_stage1_asm_fwd",
            launch_aiter_asm,
            aiter_asm_out,
            aiter_asm_logits,
            aiter_asm_lse,
        )
    return err, us


# -- pytest ------------------------------------------------------


@pytest.mark.parametrize("batch_size,ctx_len", MLA_DECODE_BENCH_CONFIGS)
def test_mla_decode(batch_size, ctx_len):
    run_single(batch_size, ctx_len)


# -- CLI (local benchmarking) -----------------------------------


def main():
    parser = argparse.ArgumentParser(description="FlyDSL MLA decode test")
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=None)
    parser.add_argument("-c", "--ctx_len", type=int, nargs="*", default=None)
    parser.add_argument("-ms", "--max_splits", type=int, default=32)
    parser.add_argument("--num_iters", type=int, default=DEFAULT_BENCH_ITERS)
    parser.add_argument("--num_warmup", type=int, default=DEFAULT_BENCH_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_BENCH_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--bench_aiter",
        action="store_true",
        help="Also benchmark aiter.hk_mla_decode_fwd and aiter.mla_decode_stage1_asm_fwd.",
    )
    args = parser.parse_args()

    if args.batch is None and args.ctx_len is None:
        configs = MLA_DECODE_BENCH_CONFIGS
    else:
        batches = args.batch if args.batch is not None else [b for b, _ in MLA_DECODE_BENCH_CONFIGS]
        ctx_lens = args.ctx_len if args.ctx_len is not None else [c for _, c in MLA_DECODE_BENCH_CONFIGS]
        configs = [(b, c) for b in batches for c in ctx_lens]

    for b, c in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"batch={b}  ctx_len={c}")
        logger.info(f"{'='*60}")
        run_single(
            b,
            c,
            max_split_per_batch=args.max_splits,
            bench_iters=args.num_iters,
            bench_warmup=args.num_warmup,
            bench_repeats=args.repeat,
            seed=args.seed,
            bench_aiter=args.bench_aiter,
        )

    logger.info("\nAll tests passed.")


if __name__ == "__main__":
    main()
