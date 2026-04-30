#!/usr/bin/env python3
"""Test for MLA decode attention kernel (FP8 variant, MFMA 16x16x32, GQA, seqlen_q <= 4).

Tests absorbed MLA with kv_lora_rank + qk_rope_head_dim split.
Paged KV cache: [num_physical_blocks, page_block_size, num_kv_heads, HEAD_DIM_QK].
Block table maps logical block indices to physical block indices.

The kernel outputs fp32 partial results (Mid_O, Mid_lse) per KV-split.
A Python combine step merges splits via online log-sum-exp, matching
the stage-2 Triton combine kernel interface.
"""

import sys
import math
import argparse
import random
from pathlib import Path
import logging
from typing import Optional, Tuple

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

from kernels.mla_decode_fp8_v4 import compile_mla_decode_fp8
from tests.test_common import verify_output

DEFAULT_SEED = 42
UNIFORM_RANGE = (-0.3, 0.3)

SHUFFLE = False
NUM_WAVES = 4  # seqlen_q


def auto_num_kv_splits(bs, skv, num_q_heads, block_n=32, min_tiles_per_split=3):
    cu_num = torch.cuda.get_device_properties(0).multi_processor_count
    overhead = 84.1
    total_tiles = (skv + block_n - 1) // block_n
    max_sp = max(1, total_tiles // min_tiles_per_split)
    best_score, best_sp = -1.0, 1
    for sp in range(1, min(17, max_sp + 1)):
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


def build_sparse_ref_indices(batch, s_q, s_kv, causal, device):
    indices = torch.full((batch, s_q, s_kv), -1, dtype=torch.int64, device=device)
    kv_positions = torch.arange(s_kv, device=device)
    for b in range(batch):
        batch_offset = b * s_kv
        for q_pos in range(s_q):
            valid_kv_count = s_kv - s_q + q_pos + 1 if causal else s_kv
            valid_kv_count = max(0, min(valid_kv_count, s_kv))
            indices[b, q_pos, :valid_kv_count] = batch_offset + kv_positions[:valid_kv_count]
    return indices


def create_arbitrary_indexed_kv(kv_4d, kv_lens, seed):
    """Build a page_size=1 KV cache plus packed arbitrary physical indices."""
    B, max_skv, Hkv, D_QK = kv_4d.shape
    assert len(kv_lens) == B
    assert all(0 < length <= max_skv for length in kv_lens)

    total_tokens = sum(kv_lens)
    kv_paged = torch.zeros(
        total_tokens, 1, Hkv, D_QK,
        dtype=kv_4d.dtype, device=kv_4d.device,
    )
    indices_flat = torch.empty(total_tokens, dtype=torch.int32, device=kv_4d.device)

    phys_ids = list(range(total_tokens))
    random.Random(seed).shuffle(phys_ids)

    cursor = 0
    indptr = [0]
    for b in range(B):
        for t in range(kv_lens[b]):
            phys = phys_ids[cursor]
            kv_paged[phys, 0, :, :] = kv_4d[b, t, :, :]
            indices_flat[cursor] = phys
            cursor += 1
        indptr.append(cursor)

    kv_indptr = torch.tensor(indptr, dtype=torch.int32, device=kv_4d.device)
    return kv_paged, indices_flat, kv_indptr


def build_ref_indices_from_packed(flat_indices, kv_lens, s_q, causal, device):
    max_len = max(kv_lens)
    ref_indices = torch.full(
        (len(kv_lens), s_q, max_len), -1, dtype=torch.int64, device=device,
    )

    cursor = 0
    for b, length in enumerate(kv_lens):
        batch_indices = flat_indices[cursor:cursor + length].to(torch.int64)
        cursor += length
        for q_pos in range(s_q):
            valid_kv_count = length - s_q + q_pos + 1 if causal else length
            valid_kv_count = max(0, min(valid_kv_count, length))
            ref_indices[b, q_pos, :valid_kv_count] = batch_indices[:valid_kv_count]
    return ref_indices


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
    B, Sq, Hq, D_QK = q.shape #1,4,16,576
    _, Skv, Hkv, _ = kv.shape #1,32,1,576
    D_V = kv_lora_rank #512
    assert Hkv == num_kv_heads

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D_QK)

    k = kv #1,32,1,576
    v = kv[:, :, :, :D_V] #1,32,1,512

    gqa_group = Hq // Hkv #16 // 1 = 16
    k_exp = k.unsqueeze(3).expand(B, Skv, Hkv, gqa_group, D_QK).reshape(B, Skv, Hq, D_QK) #1,32,16,576
    v_exp = v.unsqueeze(3).expand(B, Skv, Hkv, gqa_group, D_V).reshape(B, Skv, Hq, D_V) #1,32,16,512

    q_t = q.transpose(1, 2).float() #1,16,4,576
    k_t = k_exp.transpose(1, 2).float() # 1,16,32,576
    v_t = v_exp.transpose(1, 2).float() # 1,16,32,512

    scores = torch.matmul(q_t, k_t.transpose(-1, -2)) * sm_scale # 1,16,4,576 * 1,16,576,32 = 1,16,4,32

    if causal:
        q_positions = torch.arange(Sq, device=q.device).unsqueeze(1) #4
        kv_positions = torch.arange(Skv, device=q.device).unsqueeze(0) - (Skv - Sq) #32-32 = 0
        mask = kv_positions > q_positions
        scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))

    p = torch.softmax(scores, dim=-1)
    out = torch.matmul(p, v_t) # 1,16,4,32 * 1,16,32,576 = 1,16,4,576
    return out.transpose(1, 2)


# Reference adapted from https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py.
def ref_sparse_attn_decode_from_tensors(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices_in_kvcache: torch.Tensor,
    sm_scale: float,
    d_v: int,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standalone PyTorch version of tests/ref.py::ref_sparse_attn_decode.

    This does not depend on flash_mla, TestParam, TestcaseForDecode, or KVScope.
    All state is passed as individual tensors.

    Args:
        q: [batch, s_q, h_q, d_qk]
        k_cache: [num_blocks, block_size, h_kv, d_qk], h_kv is expected to be 1
        indices_in_kvcache: [batch, s_q, topk], flattened block-cache indices
        sm_scale: scale applied to q @ k
        d_v: value dimension, taken from the first d_v dims of k_cache
        attn_sink: optional [h_q]
        topk_length: optional [batch]
        extra_k_cache: optional [extra_num_blocks, extra_block_size, h_kv, d_qk]
        extra_indices_in_kvcache: optional [batch, s_q, extra_topk]
        extra_topk_length: optional [batch]

    Returns:
        output: [batch, s_q, h_q, d_v]
        lse: [batch, h_q, s_q]
    """
    batch, s_q, h_q, d_qk = q.shape

    def process_kv_scope(
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        length: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        topk = indices.size(-1)
        flat_kv = kv_cache.reshape(-1, d_qk)

        fixed_indices = torch.clamp_min(indices, 0)
        gathered_kv = flat_kv.index_select(0, fixed_indices.reshape(-1)).reshape(batch, s_q, topk, d_qk)
        invalid_mask = indices == -1

        if length is not None:
            topk_positions = torch.arange(topk, device=indices.device).view(1, 1, topk)
            invalid_mask |= topk_positions >= length.view(batch, 1, 1)

        return gathered_kv, invalid_mask

    gathered_kv, invalid_mask = process_kv_scope(k_cache, indices_in_kvcache, topk_length)

    if extra_k_cache is not None:
        assert extra_indices_in_kvcache is not None
        gathered_extra_kv, invalid_extra_mask = process_kv_scope(
            extra_k_cache,
            extra_indices_in_kvcache,
            extra_topk_length,
        )
        gathered_kv = torch.cat([gathered_kv, gathered_extra_kv], dim=2)
        invalid_mask = torch.cat([invalid_mask, invalid_extra_mask], dim=2)
    else:
        assert extra_indices_in_kvcache is None
        assert extra_topk_length is None

    gathered_kv = gathered_kv.reshape(batch * s_q, -1, d_qk).float()
    gathered_kv[gathered_kv != gathered_kv] = 0.0
    q_2d = q.float().reshape(batch * s_q, h_q, d_qk)
    attn_weight = q_2d @ gathered_kv.transpose(-1, -2)
    attn_weight *= sm_scale
    attn_weight[invalid_mask.reshape(batch * s_q, 1, -1).broadcast_to(attn_weight.shape)] = float("-inf")
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.exp(attn_weight - lse.unsqueeze(-1))
    output = attn_weight @ gathered_kv[..., :d_v]
    output = output.reshape(batch, s_q, h_q, d_v)
    lse = lse.reshape(batch, s_q, h_q)

    if attn_sink is not None:
        output *= (1.0 / (1.0 + torch.exp(attn_sink.view(1, 1, h_q) - lse))).unsqueeze(-1)

    lonely_q_mask = lse == float("-inf")
    output[lonely_q_mask.unsqueeze(-1).broadcast_to(batch, s_q, h_q, d_v)] = 0.0
    lse[lonely_q_mask] = float("+inf")

    return output.to(torch.bfloat16), lse.transpose(1, 2)


def run_config(
    batch, q_len, kv_len, num_q_heads, num_kv_heads,
    kv_lora_rank, qk_rope_head_dim, page_block_size,
    causal, exe, seed=DEFAULT_SEED, shuffle=False,
    num_kv_splits=1,
    arbitrary_indices=False,
    kv_lens=None,
):
    device = "cuda"
    results = {}

    B, Sq = batch, q_len
    if kv_lens is None:
        kv_lens = [kv_len] * B
    else:
        kv_lens = list(kv_lens)
        assert len(kv_lens) == B
    Skv = max(kv_lens)
    Hq, Hkv = num_q_heads, num_kv_heads
    D_QK = kv_lora_rank + qk_rope_head_dim
    D_V = kv_lora_rank
    total_q = B * Sq

    setup_seed(seed)
    fp8_dtype = torch.float8_e4m3fnuz
    q_f32 = torch.randn(B, Sq, Hq, D_QK, dtype=torch.float32, device=device)
    kv_f32 = torch.randn(B, Skv, Hkv, D_QK, dtype=torch.float32, device=device)

    q_4d = q_f32.to(fp8_dtype)
    kv_4d = kv_f32.to(fp8_dtype)

    if arbitrary_indices:
        assert page_block_size == 1, "arbitrary indices require page_block_size=1"
        kv_paged, block_table, kv_indptr = create_arbitrary_indexed_kv(
            kv_4d, kv_lens, seed=seed + 17,
        )
        num_blocks_per_seq = Skv
    else:
        kv_paged, block_table = create_paged_kv(kv_4d, page_block_size, shuffle=shuffle)
        num_blocks_per_seq = block_table.shape[1]
        kv_indptr = torch.arange(
            0, (B + 1) * Skv, Skv, dtype=torch.int32, device=device,
        )

    q_flat = q_4d.contiguous().view(torch.uint8).view(-1)
    kv_paged_flat = kv_paged.contiguous().view(torch.uint8).view(-1)
    bt_flat = block_table.contiguous().view(-1)

    mid_o_flat = torch.zeros(
        total_q * num_kv_splits * Hq * D_V,
        dtype=torch.float32, device=device,
    )
    mid_lse_flat = torch.full(
        (total_q * num_kv_splits * Hq,),
        float("-inf"), dtype=torch.float32, device=device,
    )

    mid_o_4d = mid_o_flat.view(total_q, num_kv_splits, Hq, D_V)
    mid_lse_3d = mid_lse_flat.view(total_q, num_kv_splits, Hq)
    output = torch.empty(total_q, Hq, D_V, dtype=torch.float16, device=device)

    qo_indptr = torch.arange(
        0, (B + 1) * Sq, Sq, dtype=torch.int32, device=device,
    )
    nkv_splits_indptr = torch.arange(
        0, (B + 1) * num_kv_splits, num_kv_splits,
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

    sm = 1.0 / math.sqrt(D_QK)
    assert Hkv == 1, "ref_sparse_attn_decode_from_tensors expects num_kv_heads == 1"
    if arbitrary_indices:
        sparse_ref_indices = build_ref_indices_from_packed(
            block_table, kv_lens, Sq, causal, device,
        )
        ref_k_cache = kv_paged
    else:
        sparse_ref_indices = build_sparse_ref_indices(B, Sq, Skv, causal, device)
        ref_k_cache = kv_4d
    ref_4d, _ = ref_sparse_attn_decode_from_tensors(
        q=q_4d.float(),
        k_cache=ref_k_cache.float(),
        indices_in_kvcache=sparse_ref_indices,
        sm_scale=sm,
        d_v=kv_lora_rank,
    )
    ref_4d = ref_4d.to(torch.float16)
    ref_flat = ref_4d.contiguous().view(total_q, Hq, D_V).float()

    o_flat_cmp = output.float().contiguous().view(total_q, Hq, D_V)
    results["passed"] = verify_output(o_flat_cmp, ref_flat, rtol=5e-2, atol=5e-2)

    if not results["passed"]:
        abs_diff = (o_flat_cmp - ref_flat).abs()
        per_query = abs_diff.mean(dim=(1, 2))
        worst_q = per_query.argmax().item()
        worst_h = abs_diff[worst_q].mean(dim=1).argmax().item()
        print(f"  [diag] worst query={worst_q} head={worst_h}")
        print(f"  [diag] out[:8]  = {o_flat_cmp[worst_q, worst_h, :8].tolist()}")
        print(f"  [diag] ref[:8]  = {ref_flat[worst_q, worst_h, :8].tolist()}")
        print(f"  [diag] out_abs_max={o_flat_cmp.abs().max():.4f} ref_abs_max={ref_flat.abs().max():.4f}")
        print(f"  [diag] out_mean={o_flat_cmp.mean():.6f} ref_mean={ref_flat.mean():.6f}")
        mid_o_nz = (mid_o_flat.abs() > 1e-10).sum().item()
        mid_lse_finite = (mid_lse_flat > -1e30).sum().item()
        print(f"  [diag] mid_o nonzero={mid_o_nz}/{mid_o_flat.numel()}"
              f" mid_lse finite={mid_lse_finite}/{mid_lse_flat.numel()}")
        for b_idx in range(B):
            b_start = b_idx * Sq
            b_end = b_start + Sq
            b_out = o_flat_cmp[b_start:b_end]
            b_ref = ref_flat[b_start:b_end]
            b_diff = (b_out - b_ref).abs().mean().item()
            b_out_norm = b_out.norm().item()
            b_ref_norm = b_ref.norm().item()
            if B <= 16 or b_diff > 0.01:
                print(f"  [diag] batch={b_idx} mean_diff={b_diff:.4f}"
                      f" out_norm={b_out_norm:.2f} ref_norm={b_ref_norm:.2f}")
        for sp_idx in range(num_kv_splits):
            sp_lse = mid_lse_3d[:, sp_idx, :].float()
            sp_lse_finite = (sp_lse > -1e30).sum().item()
            sp_o_nz = (mid_o_4d[:, sp_idx, :, :].abs() > 1e-10).sum().item()
            sp_lse_max = sp_lse[sp_lse > -1e30].max().item() if sp_lse_finite > 0 else float('-inf')
            sp_lse_min = sp_lse[sp_lse > -1e30].min().item() if sp_lse_finite > 0 else float('-inf')
            if num_kv_splits <= 8 or sp_lse_finite == 0:
                print(f"  [diag] split={sp_idx} mid_o_nz={sp_o_nz}"
                      f" lse_finite={sp_lse_finite} lse_range=[{sp_lse_min:.2f},{sp_lse_max:.2f}]")

    return results


def main():
    parser = argparse.ArgumentParser(description="MLA Decode Attention Test (FP8)")
    parser.add_argument("--num_q_heads", type=int, default=16)
    parser.add_argument("--num_kv_heads", type=int, default=1)
    parser.add_argument("--kv_lora_rank", type=int, default=512)
    parser.add_argument("--qk_rope_head_dim", type=int, default=64)
    parser.add_argument("--page_block_size", type=int, default=1)
    parser.add_argument("--no-causal", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    causal = not args.no_causal

    num_q_heads = args.num_q_heads
    num_kv_heads = args.num_kv_heads
    kv_lora_rank = args.kv_lora_rank
    qk_rope_head_dim = args.qk_rope_head_dim
    page_block_size = args.page_block_size
    head_dim_qk = kv_lora_rank + qk_rope_head_dim
    head_dim_v = kv_lora_rank

    print("=" * 110)
    print(f"MLA Decode Attention FP8 ({'causal' if causal else 'non-causal'})")
    print(f"  kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim}")
    print(f"  HEAD_DIM_QK={head_dim_qk}, HEAD_DIM_V={head_dim_v}")
    print(f"  num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}")
    print(f"  page_block_size={page_block_size}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 110)

    print("\nCompiling MLA decode FP8 kernel...", flush=True)
    exe = compile_mla_decode_fp8(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        page_block_size=page_block_size,
        causal=causal,
    )
    print("Compiled OK.\n", flush=True)

    rng = random.Random(args.seed + 7)
    random_configs = []
    for _ in range(20):
        b = rng.choice([1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 23, 31, 33, 47, 50, 63])
        skv = rng.randint(33, 10000)
        if skv % 16 == 0:
            skv += rng.choice([1, 3, 5, 7, 9, 11, 13, 15])
        shuf = rng.choice([True, False])
        random_configs.append((b, skv, shuf))

    arbitrary_base_configs = [
        (1,   32,  False),
        (1,   64,  False),
        (1,  256,  False),
        (1, 2048,  False),
        (64, 8192, True),
    ] + random_configs

    all_passed = True
    if page_block_size != 1:
        raise ValueError("Arbitrary packed indices sweep requires --page_block_size 1")

    arbitrary_configs = [(2, [32, 64])] + [
        (B_c, [Skv_c] * B_c) for B_c, Skv_c, _ in arbitrary_base_configs
    ]

    def _format_kv_lens(kv_lens):
        if len(kv_lens) > 1 and all(length == kv_lens[0] for length in kv_lens):
            return f"[{kv_lens[0]}]*{len(kv_lens)}"
        return str(kv_lens)

    print("--- Arbitrary packed indices sweep ---")
    for B_c, kv_lens_c in arbitrary_configs:
        Skv_c = max(kv_lens_c)
        nkv_c = auto_num_kv_splits(B_c, Skv_c, num_q_heads)
        tag_c = (
            f"B={B_c} Sq={NUM_WAVES} kv_lens={_format_kv_lens(kv_lens_c)}"
            f" arbitrary sp={nkv_c}"
        )
        r_c = run_config(
            B_c, NUM_WAVES, Skv_c, num_q_heads, num_kv_heads,
            kv_lora_rank, qk_rope_head_dim, page_block_size,
            causal, exe, seed=args.seed + Skv_c + 101, shuffle=False,
            num_kv_splits=nkv_c, arbitrary_indices=True,
            kv_lens=kv_lens_c,
        )
        status_c = "PASS" if r_c.get("passed") else "FAIL"
        if not r_c.get("passed"):
            all_passed = False
        print(f"  {tag_c:>60s} | {status_c}")

    print("=" * 110)
    if all_passed:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")


if __name__ == "__main__":
    main()
