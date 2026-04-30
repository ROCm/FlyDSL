# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Test for FP4 MQA logits kernel (gfx950, paged preshuffle KV cache)."""

import sys
import os
import math
import random

import torch
import pytest

sys.path.insert(0, "build-fly/python_packages")
sys.path.insert(1, ".")

os.environ.setdefault("FLYDSL_DUMP_IR", "1")

_VARIANT = os.environ.get("PA_MQA_VARIANT",
    "pipelined" if os.environ.get("PA_MQA_PIPELINED", "0") == "1" else "baseline")
_USE_PIPELINED = _VARIANT in ("pipelined", "single")
if _VARIANT == "single":
    from kernels.pa_mqa_logits_fp4_single import (
        build_pa_mqa_logits_fp4_module,
        compute_varctx_schedule,
        HEADS, HEAD_DIM, HEAD_DIM_PACKED, HEAD_DIM_SCALES,
        BLOCK_THREADS,
        allocator as _alloc_ref,
    )
    print("[test] using pa_mqa_logits_fp4_SINGLE kernel (single chunk per CTA)")
elif _VARIANT == "pipelined":
    from kernels.pa_mqa_logits_fp4_pipelined import (
        build_pa_mqa_logits_fp4_module,
        compute_varctx_schedule,
        HEADS, HEAD_DIM, HEAD_DIM_PACKED, HEAD_DIM_SCALES,
        BLOCK_THREADS,
        allocator as _alloc_ref,
    )
    print("[test] using pa_mqa_logits_fp4_PIPELINED kernel")
else:
    from kernels.pa_mqa_logits_fp4 import (
        build_pa_mqa_logits_fp4_module,
        compute_varctx_schedule,
        HEADS, HEAD_DIM, HEAD_DIM_PACKED, HEAD_DIM_SCALES,
        BLOCK_THREADS,
        allocator as _alloc_ref,
    )
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir import ir as _ir
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr.typing import T

from tests.test_common import run_perftest, checkAllclose

dev = "cuda"
SEED = 42

SCALE_BLOCK = 32  # fp4 elements per scale block


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── FP4 quant / dequant utilities ─────────────────────────────────────

# FP4 e2m1 representable values (ordered by magnitude)
_FP4_GRID_VALUES = [
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
    0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
]

# LUT: grid index → fp4 e2m1 4-bit encoding
_E2M1_LUT = [0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7]

# Inverse LUT: fp4 e2m1 4-bit encoding → grid index
_E2M1_INV_LUT = [7, 8, 9, 10, 11, 12, 13, 14, 7, 6, 5, 4, 3, 2, 1, 0]


def fp4_quant_e2m1_with_e8m0(x: torch.Tensor, block_size: int = 32):
    """Quantize bf16/fp32 tensor to FP4 e2m1 with UE8M0 block scales.

    Matches AMD CDNA4 hardware FP4 format.
    """
    *prefix, d = x.shape
    assert d % block_size == 0
    x_f = x.float()

    x_blk = x_f.reshape(*prefix, d // block_size, block_size)
    amax = x_blk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)

    fp4_max = 6.0
    exp_unbiased = torch.ceil(torch.log2(amax / fp4_max))
    exp_biased = (exp_unbiased + 127.0).clamp(0.0, 255.0).to(torch.uint8)
    e8m0_scales = exp_biased.squeeze(-1).contiguous()

    scale = torch.pow(2.0, exp_biased.float() - 127.0)
    x_scaled = x_blk / scale

    grid = torch.tensor(_FP4_GRID_VALUES, dtype=torch.float32, device=x.device)
    idx = (x_scaled.unsqueeze(-1) - grid).abs().argmin(dim=-1)

    e2m1_lut = torch.tensor(_E2M1_LUT, dtype=torch.uint8, device=x.device)
    x_fp4 = e2m1_lut[idx]

    x_4bit_flat = x_fp4.reshape(*prefix, d)
    packed = (x_4bit_flat[..., 0::2] | (x_4bit_flat[..., 1::2] << 4)).to(torch.uint8)

    return packed.contiguous(), e8m0_scales


def fp4_dequant_e2m1_with_e8m0(packed, e8m0_scales, block_size=32):
    """Dequantize FP4 e2m1 + UE8M0 back to float32."""
    *prefix, d_half = packed.shape
    d = d_half * 2

    low = packed & 0xF
    high = (packed >> 4) & 0xF
    x_4bit = torch.empty(*prefix, d, dtype=torch.uint8, device=packed.device)
    x_4bit[..., 0::2] = low
    x_4bit[..., 1::2] = high

    inv_lut = torch.tensor(_E2M1_INV_LUT, dtype=torch.long, device=packed.device)
    grid = torch.tensor(_FP4_GRID_VALUES, dtype=torch.float32, device=packed.device)
    idx = inv_lut[x_4bit.long()]
    x_vals = grid[idx]

    e8m0_u8 = e8m0_scales.float()
    scale = torch.pow(2.0, e8m0_u8 - 127.0)
    x_blk = x_vals.reshape(*prefix, d // block_size, block_size)
    x_dequant = x_blk * scale.unsqueeze(-1)

    return x_dequant.reshape(*prefix, d)


# ── Preshuffle layout helpers ─────────────────────────────────────


def create_paged_preshuffle_kv_fp4(kv_bf16, kv_block_size, num_blocks, block_tables):
    """Create paged preshuffle FP4 KV cache from dense bf16 KV.

    Args:
        kv_bf16: [B, T, D] bf16 dense KV
        kv_block_size: tokens per page block
        num_blocks: total number of page blocks
        block_tables: [B, max_blocks_per_seq] i32

    Returns:
        kv_cache: [num_blocks, 4, kv_block_size, 16] uint8 (preshuffle fp4)
        kv_scale: [num_blocks, kv_block_size, 4] uint8 (e8m0)
    """
    batch, t_max, d = kv_bf16.shape
    assert d == HEAD_DIM

    # Quantize per-token to FP4
    kv_flat = kv_bf16.reshape(-1, d)
    kv_packed, kv_e8m0 = fp4_quant_e2m1_with_e8m0(kv_flat, block_size=SCALE_BLOCK)
    kv_packed = kv_packed.reshape(batch, t_max, HEAD_DIM_PACKED)
    kv_e8m0 = kv_e8m0.reshape(batch, t_max, HEAD_DIM_SCALES)

    # Create paged preshuffle layout
    kv_cache = torch.zeros(num_blocks, 4, kv_block_size, 16, dtype=torch.uint8, device=dev)
    kv_scale = torch.zeros(num_blocks, kv_block_size, HEAD_DIM_SCALES, dtype=torch.uint8, device=dev)

    for b_idx in range(batch):
        for t_idx in range(t_max):
            block_idx = t_idx // kv_block_size
            token_in_block = t_idx % kv_block_size
            phys_block = block_tables[b_idx, block_idx].item()

            # Preshuffle: split 64 packed bytes into 4 tiles of 16 bytes
            for tile_idx in range(4):
                start = tile_idx * 16
                end = start + 16
                kv_cache[phys_block, tile_idx, token_in_block, :] = kv_packed[b_idx, t_idx, start:end]

            # Scale
            kv_scale[phys_block, token_in_block, :] = kv_e8m0[b_idx, t_idx, :]

    return kv_cache, kv_scale, kv_packed, kv_e8m0


# ── Reference implementation ─────────────────────────────────────


def ref_mqa_logits_fp4(q_packed, q_scale, kv_packed, kv_scale, weights, context_lens):
    """Reference: dequantize → einsum → relu → weight → sum."""
    batch = q_packed.shape[0]
    t_max = kv_packed.shape[1]

    q_dq = fp4_dequant_e2m1_with_e8m0(q_packed, q_scale)  # [B, H, D] float32
    kv_dq = fp4_dequant_e2m1_with_e8m0(kv_packed, kv_scale)  # [B, T, D] float32

    ref_logits = torch.full((batch, t_max), float("-inf"), device=dev, dtype=torch.float32)

    for b in range(batch):
        ctx = context_lens[b].item()
        if ctx == 0:
            continue
        qi = q_dq[b]            # [H, D]
        kvi = kv_dq[b, :ctx]    # [ctx, D]
        wi = weights[b]         # [H]

        qk = qi @ kvi.T         # [H, ctx]
        qk = torch.relu(qk) * wi[:, None]
        logits_i = qk.sum(dim=0)  # [ctx]
        ref_logits[b, :ctx] = logits_i

    return ref_logits


# ── Test + Benchmark ─────────────────────────────────────────────


def _torch_ref_step(q_dq, kv_dq, w):
    """logits[b,t] = sum_h(relu(Q[b,h,:] · K[b,t,:]) * w[b,h]).

    Note: torch baseline does the full t_max matmul (no varctx skip).
    The TFLOPS we report for it is based on USEFUL flops, so it'll look
    lower than its raw matmul throughput — that's the point: it wastes
    work on padded tokens.
    """
    qk = torch.bmm(q_dq, kv_dq.transpose(1, 2))   # [B, H, T_max]
    qk = torch.relu(qk) * w[:, :, None]
    return qk.sum(dim=1)


def _make_varctx(batch, max_ctx, kv_block_size):
    """Generate per-batch ctx lengths as a graduated sweep [max/B, ..., max],
    rounded up to a multiple of kv_block_size."""
    base = [max_ctx * (i + 1) // batch for i in range(batch)]
    # Round each up to the page boundary so the test exercises whole pages.
    return [
        min(((c + kv_block_size - 1) // kv_block_size) * kv_block_size, max_ctx)
        for c in base
    ]


@pytest.mark.parametrize(
    "batch, max_ctx, kv_block_size, block_k",
    [
        pytest.param(4, 16384, 16, 64, id="4x16k"),
        pytest.param(4, 32768, 16, 64, id="4x32k"),
        pytest.param(8, 65536, 16, 64, id="8x65k"),
    ],
)
def test_pa_mqa_logits_fp4(
    batch, max_ctx, kv_block_size, block_k,
    num_iters=20, num_warmup=3,
    num_warps=4, parallel_unit_num=512,
):
    """End-to-end varctx FP4 MQA logits: correctness vs ref + perf vs torch baselines."""
    setup_seed(SEED)
    batch_size = batch

    # Per-batch context lengths (varctx).
    ctx_list = _make_varctx(batch_size, max_ctx, kv_block_size)
    context_lens = torch.tensor(ctx_list, dtype=torch.int32, device=dev)
    total_tokens = int(context_lens.sum().item())

    print("=" * 96)
    print(f"FP4 MQA Logits varctx: batch={batch_size}, max_ctx={max_ctx}, "
          f"kv_block={kv_block_size}, block_k={block_k}")
    print(f"  ctx_lens = {ctx_list}  (sum={total_tokens}, "
          f"avg={total_tokens // batch_size}, util={total_tokens/(batch_size*max_ctx):.1%})")
    naive_ctas = batch_size * ((max_ctx + block_k - 1) // block_k)
    print("=" * 96)

    max_blocks_per_seq = (max_ctx + kv_block_size - 1) // kv_block_size
    num_blocks = max_blocks_per_seq * batch_size
    t_max = max_blocks_per_seq * kv_block_size

    # ---- Generate data ----
    q_bf16 = torch.randn(batch_size, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    kv_bf16 = torch.randn(batch_size, t_max, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    weights = torch.randn(batch_size, HEADS, dtype=torch.float32, device=dev) * 0.1

    q_packed, q_e8m0 = fp4_quant_e2m1_with_e8m0(
        q_bf16.reshape(batch_size * HEADS, HEAD_DIM), block_size=SCALE_BLOCK
    )
    q_packed = q_packed.reshape(batch_size, HEADS, HEAD_DIM_PACKED)
    q_e8m0 = q_e8m0.reshape(batch_size, HEADS, HEAD_DIM_SCALES)

    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=dev).reshape(
        batch_size, max_blocks_per_seq)
    kv_cache, kv_scale, kv_packed_dense, kv_e8m0_dense = create_paged_preshuffle_kv_fp4(
        kv_bf16, kv_block_size, num_blocks, block_tables)

    # ---- Reference (FP4 dequant + matmul) — uses per-batch ctx_lens ----
    ref_logits = ref_mqa_logits_fp4(
        q_packed, q_e8m0, kv_packed_dense, kv_e8m0_dense, weights, context_lens)

    # ── Persistent-grid schedule (gluon-style safe_chunks_per_cta) ──
    # parallel_unit_num: target CTA count (MI355X has 256 CUs; default 256×2).
    safe, cta_b, cta_cs, cta_cc, total_ctas = compute_varctx_schedule(
        context_lens, block_k, parallel_unit_num)
    print(f"  schedule: parallel_unit={parallel_unit_num} num_warps={num_warps} "
          f"safe_chunks_per_cta={safe}  total_ctas={total_ctas}  "
          f"(naive grid would be {naive_ctas})")

    # ---- Build flydsl kernel (pipelined kernel uses safe + num_warps as constexpr) ----
    _build_kwargs = dict(block_k=block_k, kv_block_size=kv_block_size,
                         max_blocks_per_seq=max_blocks_per_seq)
    if _USE_PIPELINED:
        _build_kwargs["max_chunks_per_cta"] = safe
        _build_kwargs["num_warps"] = num_warps
    kfn, alloc = build_pa_mqa_logits_fp4_module(**_build_kwargs)
    block_threads = getattr(alloc, "block_threads", BLOCK_THREADS)

    out_logits = torch.full((batch_size, t_max), float("-inf"),
                            dtype=torch.float32, device=dev)

    qe = q_e8m0.view(torch.uint8)
    stream = torch.cuda.current_stream()

    @flyc.jit
    def launch_kernel(out, q, qs, kv, kvs, bt, w, ctx_lens,
                      cta_b_, cta_cs_, cta_cc_,
                      stride_out: fx.Int32,
                      gx: fx.Int32, stream: fx.Stream):
        _ = (batch_size, kv_block_size, max_blocks_per_seq, block_k)
        alloc.finalized = False
        cctx = CompilationContext.get_current()
        with _ir.InsertionPoint(cctx.gpu_module_body):
            alloc.finalize()
        gxi = arith.index_cast(T.index, gx.ir_value())
        kfn(out, q, qs, kv, kvs, bt, w, ctx_lens,
            cta_b_, cta_cs_, cta_cc_, stride_out).launch(
            grid=(gxi,), block=(block_threads, 1, 1), stream=stream)

    def launch_flydsl():
        launch_kernel(out_logits, q_packed, qe, kv_cache, kv_scale,
                      block_tables, weights, context_lens,
                      cta_b, cta_cs, cta_cc,
                      t_max, total_ctas, stream)

    # ---- Correctness: one launch + cosine_sim ----
    out_logits.fill_(float("-inf"))
    launch_flydsl()
    torch.cuda.synchronize()

    mask = torch.arange(t_max, device=dev).unsqueeze(0) < context_lens.unsqueeze(1)
    valid_out = out_logits[mask].double()
    valid_ref = ref_logits[mask].double()
    cos = (valid_out * valid_ref).sum() / (
        valid_out.norm() * valid_ref.norm() + 1e-12)
    max_abs_err = (valid_out - valid_ref).abs().max().item()
    mean_abs_err = (valid_out - valid_ref).abs().mean().item()
    err_ratio = checkAllclose(valid_ref.float(), valid_out.float(),
                              rtol=0.05, atol=0.05,
                              msg="flydsl-fp4 vs ref", printLog=False)
    # Verify NEG_INF is preserved past each batch's ctx_len.
    out_past_ctx = out_logits.masked_select(~mask)
    neg_inf_ok = bool(torch.isneginf(out_past_ctx).all().item()) if out_past_ctx.numel() else True
    print(f"  correctness: cosine_sim={cos.item():.6f}  "
          f"max_abs_err={max_abs_err:.6f}  mean_abs_err={mean_abs_err:.6f}  "
          f"err_ratio={err_ratio:.4f}  past_ctx_neginf={neg_inf_ok}")
    assert cos.item() > 0.99, f"FlyDSL FP4 vs ref cosine_sim={cos.item():.4f} < 0.99"
    assert neg_inf_ok, "OOB tokens were not NEG_INF — early-exit / pre-init broken"

    # ---- Perf: flydsl ----
    _, us_fly = run_perftest(launch_flydsl, num_iters=num_iters, num_warmup=num_warmup)
    torch.cuda.synchronize()

    # ---- Perf: torch baselines (dequant excluded — pure matmul + relu/wsum) ----
    # Torch does full t_max matmul per batch (no varctx skip).
    q_dq_bf16 = fp4_dequant_e2m1_with_e8m0(
        q_packed.reshape(-1, HEAD_DIM_PACKED),
        q_e8m0.reshape(-1, HEAD_DIM_SCALES),
    ).reshape(batch_size, HEADS, HEAD_DIM).to(torch.bfloat16)
    kv_dq_bf16 = kv_bf16
    w_bf16 = weights.to(torch.bfloat16)

    _, us_bf16 = run_perftest(_torch_ref_step, q_dq_bf16, kv_dq_bf16, w_bf16,
                              num_iters=num_iters, num_warmup=num_warmup)
    _, us_fp32 = run_perftest(_torch_ref_step,
                              q_dq_bf16.float(), kv_dq_bf16.float(), weights,
                              num_iters=num_iters, num_warmup=num_warmup)

    # ---- USEFUL FLOPs / bytes (varctx — based on real ctx_lens, not max) ----
    flops = total_tokens * HEADS * (2 * HEAD_DIM + 3)
    bytes_q = batch_size * HEADS * (HEAD_DIM_PACKED + HEAD_DIM_SCALES)
    bytes_kv = total_tokens * (HEAD_DIM_PACKED + HEAD_DIM_SCALES)
    bytes_w = batch_size * HEADS * 4
    bytes_bt = batch_size * max_blocks_per_seq * 4
    bytes_out = total_tokens * 4
    bytes_total = bytes_q + bytes_kv + bytes_w + bytes_bt + bytes_out

    def metrics(us):
        if us <= 0:
            return 0.0, 0.0
        sec = us * 1e-6
        return flops / sec / 1e12, bytes_total / sec / 1e9

    tflops_fly, gbps_fly = metrics(us_fly)
    tflops_bf16, _ = metrics(us_bf16)
    tflops_fp32, _ = metrics(us_fp32)

    print(f"\n  {'':>16} | {'us':>10} | {'TFLOPS':>8} | {'GB/s':>8} | {'vs flydsl':>10}")
    print(f"  {'flydsl-fp4':>16} | {us_fly:>10.2f} | {tflops_fly:>8.2f} | {gbps_fly:>8.1f} |")
    print(f"  {'torch-bf16':>16} | {us_bf16:>10.2f} | {tflops_bf16:>8.2f} | {'-':>8} | "
          f"{us_bf16/us_fly:>9.2f}x")
    print(f"  {'torch-fp32':>16} | {us_fp32:>10.2f} | {tflops_fp32:>8.2f} | {'-':>8} | "
          f"{us_fp32/us_fly:>9.2f}x")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="FP4 MQA Logits Kernel Test + Benchmark (gfx950)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch", type=int, default=0,
                        help="Batch size (0 = run default sweep)")
    parser.add_argument("--ctx", type=int, default=0,
                        help="Context length (0 = run default sweep)")
    parser.add_argument("--kv_block_size", type=int, default=16)
    parser.add_argument("--block_k", type=int,
                        default=(128 if _USE_PIPELINED else 64),
                        help="Default 128 when PA_MQA_PIPELINED=1, else 64")
    parser.add_argument("--num_iters", type=int, default=20)
    parser.add_argument("--num_warmup", type=int, default=5)
    parser.add_argument("--num_warps", type=int, default=4,
                        help="warps per CTA (pipelined kernel only); BLOCK=num_warps*64")
    parser.add_argument("--parallel_unit_num", type=int, default=512,
                        help="target CTA count for host schedule (default 512)")
    args = parser.parse_args()

    if args.batch > 0 and args.ctx > 0:
        configs = [(args.batch, args.ctx)]
    else:
        configs = [
            # (4, 16384),
            # (4, 32768),
            (8, 65536),
        ]

    for b, c in configs:
        try:
            test_pa_mqa_logits_fp4(
                batch=b, max_ctx=c,
                kv_block_size=args.kv_block_size, block_k=args.block_k,
                num_iters=args.num_iters, num_warmup=args.num_warmup,
                num_warps=args.num_warps, parallel_unit_num=args.parallel_unit_num)
        except AssertionError as e:
            print(f"  FAIL: {e}\n")
        except Exception:
            import traceback
            traceback.print_exc()
