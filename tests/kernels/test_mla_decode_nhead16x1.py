#!/usr/bin/env python3
"""Test for MLA decode attention kernel — nhead16x1 variant (FP8).

seqlen_q == 1; 4 waves split KV positions within each tile.
Each wave writes its own sub-split, so actual_nkv_splits = nkv_splits * 4.
"""

import sys
import math
import argparse
import random
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

import torch
import numpy as np
import triton
import triton.language as tl

if not torch.cuda.is_available():
    print("CUDA/ROCm not available")
    sys.exit(1)

from kernels.mla_decode_nhead16x1 import KERNEL_NAME, compile_mla_decode
from tests.test_common import verify_output
from aiter.test_common import run_perftest

DEFAULT_SEED = 42

SHUFFLE = True
SEQLEN_Q = 1


def auto_num_kv_splits(bs, skv, num_q_heads):
    """Adaptive KV split selection (ported from mla.py:get_meta_param)."""
    cu_num = torch.cuda.get_device_properties(0).multi_processor_count
    overhead = 84.1
    best_score, best_sp = -1.0, 1
    for sp in range(1, 17):
        wgs = bs * sp
        occupancy = wgs / (((wgs + cu_num - 1) // cu_num) * cu_num)
        kv_eff = skv / (skv + overhead * sp)
        score = occupancy * kv_eff
        if score > best_score:
            best_score, best_sp = score, sp
    return best_sp


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_paged_kv(kv_4d, page_block_size, shuffle=False):
    """Convert contiguous KV to paged KV cache.

    Args:
        kv_4d: [B, Skv, Hkv, D_QK] contiguous KV tensor.
        page_block_size: tokens per physical page.
        shuffle: if True, randomize physical block order to stress-test.

    Returns:
        kv_paged:    [total_blocks, page_block_size, Hkv, D_QK] fp8 tensor.
        block_table: [B, num_blocks_per_seq] i32 tensor.
    """
    B, Skv, Hkv, D_QK = kv_4d.shape
    num_blocks = (Skv + page_block_size - 1) // page_block_size
    total_blocks = B * num_blocks

    kv_paged = torch.zeros(
        total_blocks, page_block_size, Hkv, D_QK,
        dtype=kv_4d.dtype, device=kv_4d.device,
    )
    block_table = torch.zeros(
        B, num_blocks, dtype=torch.int32, device=kv_4d.device,
    )

    phys_ids = list(range(total_blocks))
    if shuffle:
        random.shuffle(phys_ids)

    idx = 0
    for b in range(B):
        for bl in range(num_blocks):
            phys = phys_ids[idx]
            block_table[b, bl] = phys
            start = bl * page_block_size
            end = min(start + page_block_size, Skv)
            length = end - start
            kv_paged[phys, :length, :, :] = kv_4d[b, start:end, :, :]
            idx += 1

    return kv_paged, block_table


@triton.jit
def _fwd_kernel_stage2_asm(
    Mid_O,
    Mid_lse,
    O,
    qo_indptr,
    kv_indptr,
    num_kv_splits_indptr,
    stride_mid_ob: tl.int64,
    stride_mid_oh: tl.int64,
    stride_mid_os: tl.int64,
    stride_obs: tl.int64,
    stride_oh: tl.int64,
    MAYBE_FINAL_OUT: tl.constexpr,
    BATCH_NUM: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    mgc: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_qo_start = tl.load(qo_indptr + cur_batch)
    cur_qo_end = tl.load(qo_indptr + cur_batch + 1)
    cur_split_start = tl.load(num_kv_splits_indptr + cur_batch)
    cur_split_end = tl.load(num_kv_splits_indptr + cur_batch + 1)
    num_max_kv_splits = tl.load(num_kv_splits_indptr + BATCH_NUM)
    cur_kv_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    offs_logic = cur_qo_start * stride_mid_ob + cur_head * stride_mid_oh
    offs_v = offs_logic * Lv + offs_d
    num_valid_kv_splits = tl.minimum(
        cur_split_end - cur_split_start, tl.cdiv(cur_kv_seq_len, mgc)
    )
    FINAL_OUT = MAYBE_FINAL_OUT and num_max_kv_splits == BATCH_NUM

    for cur_qo in range(cur_qo_start, cur_qo_end):
        if FINAL_OUT:
            input_ptr = Mid_O.to(tl.pointer_type(O.type.element_ty))
            out = tl.load(
                input_ptr
                + Lv * (cur_qo * stride_mid_os + cur_head * stride_mid_oh)
                + offs_d,
                mask=mask_d,
                other=0.0,
            )
            tl.store(
                O + cur_qo * stride_obs + cur_head * stride_oh + offs_d,
                out,
                mask=mask_d,
            )
        else:
            e_sum = 0.0
            e_max = -float("inf")
            acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
            for split_kv_id in range(0, num_valid_kv_splits):
                tv = tl.load(
                    Mid_O + offs_v + split_kv_id * stride_mid_os * Lv,
                    mask=mask_d,
                    other=0.0,
                )
                tlogic = tl.load(Mid_lse + offs_logic + split_kv_id * stride_mid_os)
                n_e_max = tl.maximum(tlogic, e_max)

                old_scale = tl.exp(e_max - n_e_max)
                acc *= old_scale
                exp_logic = tl.exp(tlogic - n_e_max)
                acc += exp_logic * tv

                e_sum = e_sum * old_scale + exp_logic
                e_max = n_e_max
            offs_logic += stride_mid_ob
            offs_v += stride_mid_ob * Lv
            tl.store(
                O + cur_qo * stride_obs + cur_head * stride_oh + offs_d,
                acc / e_sum,
                mask=mask_d,
            )


def pytorch_ref_mla_attention(q, kv, num_kv_heads, kv_lora_rank, causal=True, sm_scale=None):
    """Reference MLA attention (contiguous KV)."""
    B, Sq, Hq, D_QK = q.shape
    _, Skv, Hkv, _ = kv.shape
    D_V = kv_lora_rank
    assert Hkv == num_kv_heads

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D_QK)

    k = kv
    v = kv[:, :, :, :D_V]

    gqa_group = Hq // Hkv
    k_exp = k.unsqueeze(3).expand(B, Skv, Hkv, gqa_group, D_QK).reshape(B, Skv, Hq, D_QK)
    v_exp = v.unsqueeze(3).expand(B, Skv, Hkv, gqa_group, D_V).reshape(B, Skv, Hq, D_V)

    q_t = q.transpose(1, 2).float()
    k_t = k_exp.transpose(1, 2).float()
    v_t = v_exp.transpose(1, 2).float()

    scores = torch.matmul(q_t, k_t.transpose(-1, -2)) * sm_scale

    if causal:
        q_positions = torch.arange(Sq, device=q.device).unsqueeze(1)
        kv_positions = torch.arange(Skv, device=q.device).unsqueeze(0) - (Skv - Sq)
        mask = kv_positions > q_positions
        scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))

    p = torch.softmax(scores, dim=-1)
    out = torch.matmul(p, v_t)
    return out.transpose(1, 2)


def run_config(
    batch, q_len, kv_len, num_q_heads, num_kv_heads,
    kv_lora_rank, qk_rope_head_dim, page_block_size,
    dtype, causal, exe, seed=DEFAULT_SEED, shuffle=False,
    num_kv_splits=1, warmup=50, iters=100,
):
    device = "cuda"
    results = {}

    B, Sq, Skv = batch, q_len, kv_len
    Hq, Hkv = num_q_heads, num_kv_heads
    D_QK = kv_lora_rank + qk_rope_head_dim
    D_V = kv_lora_rank
    total_q = B * Sq
    HEADS_PER_WAVE = 16
    num_head_groups = Hq // HEADS_PER_WAVE
    elem_bytes = 1  # fp8
    actual_nkv_splits = num_kv_splits

    setup_seed(seed)
    fp8_dtype = torch.float8_e4m3fnuz
    q_f32 = torch.randn(B, Sq, Hq, D_QK, dtype=torch.float32, device=device)
    kv_f32 = torch.randn(B, Skv, Hkv, D_QK, dtype=torch.float32, device=device)
    q_4d = q_f32.to(fp8_dtype)
    kv_4d = kv_f32.to(fp8_dtype)

    kv_paged, block_table = create_paged_kv(kv_4d, page_block_size, shuffle=shuffle)

    q_flat = q_4d.contiguous().view(torch.uint8).view(-1)
    kv_paged_flat = kv_paged.contiguous().view(torch.uint8).view(-1)
    bt_flat = block_table.contiguous().view(-1)
    num_blocks_per_seq = block_table.shape[1]

    mid_o_flat = torch.zeros(
        total_q * actual_nkv_splits * Hq * D_V,
        dtype=torch.float32, device=device,
    )
    mid_lse_flat = torch.full(
        (total_q * actual_nkv_splits * Hq,),
        float("-inf"), dtype=torch.float32, device=device,
    )

    mid_o_4d = mid_o_flat.view(total_q, actual_nkv_splits, Hq, D_V)
    mid_lse_3d = mid_lse_flat.view(total_q, actual_nkv_splits, Hq)
    output = torch.empty(total_q, Hq, D_V, dtype=torch.float16, device=device)

    qo_indptr = torch.arange(
        0, (B + 1) * Sq, Sq, dtype=torch.int32, device=device,
    )
    kv_indptr = torch.arange(
        0, (B + 1) * Skv, Skv, dtype=torch.int32, device=device,
    )
    nkv_splits_indptr = torch.arange(
        0, (B + 1) * actual_nkv_splits, actual_nkv_splits,
        dtype=torch.int32, device=device,
    )
    Lv = D_V
    BLOCK_DV = triton.next_power_of_2(Lv)
    grid_s2 = (B, Hq)
    stream = torch.cuda.current_stream()

    def _run():
        exe(
            q_flat, kv_paged_flat, mid_o_flat, mid_lse_flat,
            bt_flat, B, Sq, kv_indptr, num_blocks_per_seq, num_kv_splits,
            stream,
        )
        _fwd_kernel_stage2_asm[grid_s2](
            mid_o_4d, mid_lse_3d, output,
            qo_indptr, kv_indptr, nkv_splits_indptr,
            mid_lse_3d.stride(0), mid_lse_3d.stride(2), mid_lse_3d.stride(1),
            output.stride(0), output.stride(1),
            MAYBE_FINAL_OUT=False,
            BATCH_NUM=B,
            BLOCK_DV=BLOCK_DV,
            Lv=Lv,
            mgc=16,
            num_warps=4,
            num_stages=2,
            waves_per_eu=4,
        )

    try:
        _run()
        torch.cuda.synchronize()
    except Exception as e:
        results["err"] = f"exec: {e}"
        import traceback
        traceback.print_exc()
        return results

    # ---- Correctness ----
    sm = 1.0 / math.sqrt(D_QK)
    ref_4d = pytorch_ref_mla_attention(
        q_4d.float(), kv_4d.float(),
        num_kv_heads, kv_lora_rank, causal=causal, sm_scale=sm,
    ).to(torch.float16)
    ref_flat = ref_4d.contiguous().view(total_q, Hq, D_V).float()

    o_flat_cmp = output.float().contiguous().view(total_q, Hq, D_V)
    results["passed"] = verify_output(o_flat_cmp, ref_flat, rtol=1e-3, atol=1e-3)

    # ---- Benchmark ----
    _, us = run_perftest(_run, num_iters=iters, num_warmup=warmup)
    results["time_us"] = us

    # FLOPS: GEMM1 S=K@Q^T + GEMM2 O^T=V^T@P  (multiply-add = 2 ops)
    gemm1_flops = 2 * B * Hq * Sq * Skv * D_QK
    gemm2_flops = 2 * B * Hq * Sq * Skv * D_V
    total_flops = gemm1_flops + gemm2_flops
    results["tflops"] = total_flops / (us / 1e6) / 1e12

    q_bytes = B * Sq * Hq * D_QK * elem_bytes
    kv_bytes = num_head_groups * B * Skv * D_QK * elem_bytes
    mid_o_bytes = total_q * actual_nkv_splits * Hq * D_V * 4
    mid_lse_bytes = total_q * actual_nkv_splits * Hq * 4
    out_bytes = total_q * Hq * D_V * 2  # output is fp16
    total_bytes = q_bytes + kv_bytes + mid_o_bytes + mid_lse_bytes + out_bytes
    results["bw_gbs"] = total_bytes / (us / 1e6) / 1e9

    return results


def main():
    parser = argparse.ArgumentParser(description="MLA Decode Attention Test")
    parser.add_argument("--num_q_heads", type=int, default=16)
    parser.add_argument("--num_kv_heads", type=int, default=1)
    parser.add_argument("--kv_lora_rank", type=int, default=512)
    parser.add_argument("--qk_rope_head_dim", type=int, default=64)
    parser.add_argument("--page_block_size", type=int, default=64)
    parser.add_argument("--no-causal", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    causal = not args.no_causal
    dtype = torch.float8_e4m3fnuz

    num_q_heads = args.num_q_heads
    num_kv_heads = args.num_kv_heads
    kv_lora_rank = args.kv_lora_rank
    qk_rope_head_dim = args.qk_rope_head_dim
    page_block_size = args.page_block_size
    head_dim_qk = kv_lora_rank + qk_rope_head_dim
    head_dim_v = kv_lora_rank

    print("=" * 110)
    print(f"MLA Decode Attention ({'causal' if causal else 'non-causal'}, fp8)")
    print(f"  kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim}")
    print(f"  HEAD_DIM_QK={head_dim_qk}, HEAD_DIM_V={head_dim_v}")
    print(f"  num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}")
    print(f"  page_block_size={page_block_size}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 110)

    print("\nCompiling MLA decode kernel...", flush=True)
    exe = compile_mla_decode(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_block_size=page_block_size,
        causal=causal,
    )
    print("Compiled OK.\n", flush=True)

    # batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    # ctx_lens = [1024, 8192]
    batch_sizes = [64]
    ctx_lens = [8192]
    configs = []
    for B in batch_sizes:
        for Skv in ctx_lens:
            sp = auto_num_kv_splits(B, Skv, num_q_heads)
            configs.append((B, SEQLEN_Q, Skv, SHUFFLE, sp))

    hdr = (
        f"{'Config':>80s} | {'Status':>6s} | "
        f"{'Time(us)':>9s} {'TFLOPS':>7s} {'BW(GB/s)':>9s}"
    )
    print(hdr)
    print("-" * len(hdr))

    all_passed = True
    for B, Sq, Skv, shuf, nkv_splits in configs:
        shuf_tag = " shuf" if shuf else ""
        split_tag = f" sp={nkv_splits}" if nkv_splits > 1 else ""
        tag = (
            f"B={B} Sq={Sq} Skv={Skv} Hq={num_q_heads} Hkv={num_kv_heads} "
            f"D_QK={head_dim_qk} D_V={head_dim_v} pbs={page_block_size}"
            f"{shuf_tag}{split_tag}"
        )
        try:
            r = run_config(
                B, Sq, Skv, num_q_heads, num_kv_heads,
                kv_lora_rank, qk_rope_head_dim, page_block_size,
                dtype, causal, exe, seed=args.seed, shuffle=shuf,
                num_kv_splits=nkv_splits,
                warmup=args.warmup, iters=args.iters,
            )
            if "err" in r:
                print(f"{tag:>80s} | {'ERROR':>6s} | {r['err'][:60]}")
                all_passed = False
                continue
            status = "PASS" if r["passed"] else "FAIL"
            if not r["passed"]:
                all_passed = False
            print(
                f"{tag:>80s} | {status:>6s} | "
                f"{r['time_us']:>9.1f} {r['tflops']:>7.3f} {r['bw_gbs']:>9.1f}"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{tag:>80s} | {'ERROR':>6s} | {str(e)[:60]}")
            all_passed = False

    print("=" * 110)
    if all_passed:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
        # sys.exit(1)


if __name__ == "__main__":
    main()
