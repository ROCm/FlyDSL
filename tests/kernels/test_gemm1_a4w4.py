#!/usr/bin/env python3
"""Isolated unit test for the FlyDSL ``gemm1_a4w4`` port (BM=32, mxfp4 MoE GEMM1).

Reuses the KIMI mxfp4 inputs / weights / preshuffle layouts and per-1x32 quant
from ``/FlyDSL/test.py`` (the ``mx`` weight dict + ``build_inputs``), runs the
*aiter / HIP* ``mxfp4_moe_gemm1_a4w4`` kernel as the golden, and compares against
the FlyDSL port on the **same** sorted/quantised buffers.  This isolates GEMM1
(activation @ w1 -> SwiGLU -> requant to mxfp4); it does NOT run the whole fused
MoE.

Run on the remote gfx950 container (see scripts/REMOTE_DZM.md)::

  export PYTHONPATH=$REMOTE_FLYDSL/build-fly/python_packages:/tmp/aiter:$PWD
  export FLYDSL_RUNTIME_ENABLE_CACHE=0 ARCH=gfx950
  python3 -m pytest test_gemm1_a4w4.py -v -s
  # or as a plain script:
  python3 test_gemm1_a4w4.py
"""

import sys
from pathlib import Path as _Path

import aiter
import pytest
import torch
from aiter import QuantType, dtypes
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from kernels.gemm1_a4w4 import compile_gemm1_a4w4  # noqa: E402

# ----------------------------------------------------------------------------
# KIMI-K2.5 TP=4 shape (mirrors /FlyDSL/test.py).  This is the NE=385 family.
# ----------------------------------------------------------------------------
NE = 385  # routed experts + 1 shared
H = 7168  # model hidden / K
INTER = 512  # per-shard inter_dim  -> N_OUT = 2*INTER = 1024
TOPK = 9
BM = 32  # primary FlyDSL port target

N_OUT = 2 * INTER  # 1024
K = H  # 7168
MAX_M = 655360
NUM_N_BLOCKS = N_OUT // 256  # 4


# ----------------------------------------------------------------------------
# input / weight builders (copied from /FlyDSL/test.py, mx path only)
# ----------------------------------------------------------------------------
def build_mx_weights(device, seed=0):
    torch.manual_seed(seed)
    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1 = torch.randn((NE, 2 * INTER, H), dtype=dtypes.bf16, device=device) / 10
    w2 = torch.randn((NE, H, INTER), dtype=dtypes.bf16, device=device) / 10
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)

    mx_w1 = shuffle_weight_a16w4(w1_qt, 16, True)
    mx_w1.shuffle_kind = "mxfp4_moe"
    return dict(
        w1=mx_w1,
        w2=shuffle_weight_a16w4(w2_qt, 16, False),
        w1_scale=shuffle_scale_a16w4(w1_scale, NE, True),
        w2_scale=shuffle_scale_a16w4(w2_scale, NE, False),
    )


def build_inputs(M, device, seed=1):
    torch.manual_seed(seed)
    hidden = torch.randn((M, H), dtype=dtypes.bf16, device=device) / 10
    n_routed = NE - 1
    shared_id = NE - 1
    n_topk_routed = TOPK - 1
    g = torch.Generator(device=device).manual_seed(seed)
    bias = torch.randn(n_routed, generator=g, device=device) * 0.5
    scores = torch.randn(M, n_routed, generator=g, device=device) + bias
    routed_w, routed_ids = torch.topk(scores.softmax(-1), n_topk_routed, dim=-1)
    shared_ids = torch.full((M, 1), shared_id, device=device, dtype=routed_ids.dtype)
    shared_w = torch.ones((M, 1), device=device, dtype=routed_w.dtype)
    topk_ids = torch.cat([shared_ids, routed_ids], dim=1).to(torch.int32)
    topk_weight = torch.cat([shared_w, routed_w], dim=1).to(torch.float32)
    return hidden, topk_ids, topk_weight


def _empty_u8(device):
    return torch.empty((0,), dtype=torch.uint8, device=device)


def _empty_bf16(device):
    return torch.empty((0,), dtype=dtypes.bf16, device=device)


# ----------------------------------------------------------------------------
# Shared sort + quant prologue (mirrors aiter._mxfp4_moe_run threestage path)
# ----------------------------------------------------------------------------
def _prepare_gemm1_inputs(hidden, topk_ids, topk_weight, M, device):
    D_HIDDEN = H
    D_INTER = INTER

    active = min(NE, M * TOPK)
    cumsum_max = M * TOPK + active * (BM - 1)
    max_sorted = ((cumsum_max + BM - 1) // BM) * BM

    sorted_token_ids = torch.empty((max_sorted,), device=device, dtype=dtypes.i32)
    sorted_expert_ids = torch.empty((max_sorted // BM,), device=device, dtype=dtypes.i32)
    cumsum_tensor = torch.empty((1,), device=device, dtype=dtypes.i32)
    reverse_sorted = torch.empty((M * TOPK,), device=device, dtype=dtypes.i32)
    sorted_weights = torch.empty((max_sorted,), device=device, dtype=dtypes.fp32)
    masked_m = torch.empty((NE,), device=device, dtype=dtypes.i32)
    m_indices = torch.empty((max_sorted,), device=device, dtype=dtypes.i32)

    a_quant = torch.empty((M, D_HIDDEN // 2), device=device, dtype=torch.uint8)
    a_scale = torch.empty((M, D_HIDDEN // 32), device=device, dtype=torch.uint8)

    aiter.mxfp4_moe_sort(
        topk_ids=topk_ids,
        topk_weight=topk_weight,
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=cumsum_tensor,
        reverse_sorted=reverse_sorted,
        sorted_weights=sorted_weights,
        masked_m=masked_m,
        m_indices=m_indices,
        bf16_zero_out=_empty_bf16(device),
        bf16_zero_workspace=_empty_bf16(device),
        M_logical=M,
        NE=NE,
        TOPK=TOPK,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        MB=BM,
        prologue=1,
    )
    aiter.mxfp4_moe_quant(
        a_input=hidden,
        a_quant=a_quant,
        a_scale=a_scale,
        bf16_zero_out=_empty_bf16(device),
        NE=NE,
        TOPK=TOPK,
        D_HIDDEN=D_HIDDEN,
        MB=BM,
    )

    padded_rows = ((max_sorted + 31) // 32) * 32
    cols = D_HIDDEN // 32
    a_scale_sorted_shuffled = torch.empty((padded_rows * cols * 2,), device=device, dtype=torch.uint8)
    aiter.mxfp4_moe_sort_scales(
        a_scale=a_scale,
        sorted_token_ids=sorted_token_ids,
        cumsum_tensor=cumsum_tensor,
        a_scale_sorted_shuffled=a_scale_sorted_shuffled,
        NE=NE,
        TOPK=TOPK,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        MB=BM,
        max_sorted=max_sorted,
    )

    # gemm1 output buffers (sized as in aiter._mxfp4_moe_run)
    inter_sorted_quant = torch.empty((max_sorted, D_INTER // 2), device=device, dtype=torch.uint8)
    BM_MIN = 64
    inter_scale_cols = D_INTER // 32
    inter_scale_bytes = max_sorted * (1024 // BM_MIN) * 4
    inter_scale_rows = (inter_scale_bytes + inter_scale_cols - 1) // inter_scale_cols
    inter_scale_rows = (inter_scale_rows + 31) // 32 * 32
    inter_sorted_shuffled_scale = torch.empty((inter_scale_rows, inter_scale_cols), device=device, dtype=torch.uint8)

    return dict(
        cumsum_tensor=cumsum_tensor,
        a_quant=a_quant,
        a_scale_sorted_shuffled=a_scale_sorted_shuffled,
        sorted_expert_ids=sorted_expert_ids,
        m_indices=m_indices,
        inter_sorted_quant=inter_sorted_quant,
        inter_sorted_shuffled_scale=inter_sorted_shuffled_scale,
        max_sorted=max_sorted,
    )


def _grid(M):
    """Host-side grid for the BM!=128 launch() path (mirrors HIP launch())."""
    active = min(M * TOPK, NE)
    max_m_blocks = (M * TOPK + active * (BM - 1) + BM - 1) // BM
    return max_m_blocks * NUM_N_BLOCKS


def _run_one(M, device, w):
    hidden, topk_ids, topk_weight = build_inputs(M, device)

    # ---- golden via aiter / HIP gemm1 ------------------------------------
    g = _prepare_gemm1_inputs(hidden, topk_ids, topk_weight, M, device)
    kernelName1 = f"mxfp4_moe_g1_a4w4_NE{NE}_H{H}_E{INTER}_BM{BM}_CACHED"
    inter_q_gold = g["inter_sorted_quant"]
    inter_s_gold = g["inter_sorted_shuffled_scale"]
    aiter.mxfp4_moe_gemm1_a4w4(
        cumsum_tensor=g["cumsum_tensor"],
        a_quant=g["a_quant"],
        a_scale_sorted_shuffled=g["a_scale_sorted_shuffled"],
        w12_shuffled_quant=w["w1"],
        w12_shuffled_scale=w["w1_scale"],
        sorted_expert_ids=g["sorted_expert_ids"],
        m_indices=g["m_indices"],
        inter_sorted_quant=inter_q_gold,
        inter_sorted_shuffled_scale=inter_s_gold,
        hidden_states=hidden,
        kernelName=kernelName1,
    )
    torch.cuda.synchronize()
    inter_q_gold = inter_q_gold.clone()
    inter_s_gold = inter_s_gold.clone()

    # ---- FlyDSL gemm1 on the SAME sorted/quant buffers -------------------
    g2 = _prepare_gemm1_inputs(hidden, topk_ids, topk_weight, M, device)
    # use exactly the same shared inputs as the golden run
    g2["a_quant"] = g["a_quant"]
    g2["a_scale_sorted_shuffled"] = g["a_scale_sorted_shuffled"]
    g2["sorted_expert_ids"] = g["sorted_expert_ids"]
    g2["cumsum_tensor"] = g["cumsum_tensor"]
    g2["m_indices"] = g["m_indices"]
    inter_q_fly = torch.zeros_like(inter_q_gold)
    inter_s_fly = torch.zeros_like(inter_s_gold)

    launch = compile_gemm1_a4w4(
        MAX_M=MAX_M,
        NUM_EXPERTS=NE,
        K=K,
        N_OUT=N_OUT,
        BM=BM,
        kUseNT=False,
        kInlineQuant=False,
        kXcdSwizzle=0,
    )
    n_tokens = M
    grid = _grid(M)
    # FlyDSL DLPack rejects float4_e2m1fn_x2 (dtype code 17); view the packed
    # mxfp4 weight as raw uint8 bytes (same memory the HIP kernel reads).
    w1_q = w["w1"]
    if w1_q.element_size() == 1 and w1_q.dtype != torch.uint8:
        w1_q = w1_q.view(torch.uint8)
    w1_s = w["w1_scale"]
    if w1_s.element_size() == 1 and w1_s.dtype != torch.uint8:
        w1_s = w1_s.view(torch.uint8)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        launch(
            g2["a_quant"],
            g2["a_scale_sorted_shuffled"],
            w1_q,
            w1_s,
            g2["sorted_expert_ids"],
            g2["cumsum_tensor"],
            g2["m_indices"],
            int(n_tokens),
            inter_q_fly,
            inter_s_fly,
            int(grid),
            stream=stream,
        )
    torch.cuda.synchronize()

    # ---- compare only the rows that the sort actually populated ----------
    cumsum0 = int(g["cumsum_tensor"][0].item())
    valid_rows = cumsum0  # number of sorted (padded) rows actually written
    vq_g = inter_q_gold[:valid_rows]
    vq_f = inter_q_fly[:valid_rows]

    exact_q = torch.equal(vq_g, vq_f)
    mism_q = int((vq_g != vq_f).sum().item())
    total_q = vq_g.numel()

    return dict(
        M=M,
        cumsum=cumsum0,
        exact_q=exact_q,
        mism_q=mism_q,
        total_q=total_q,
        inter_s_gold=inter_s_gold,
        inter_s_fly=inter_s_fly,
    )


# ----------------------------------------------------------------------------
# pytest entry
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("M", [16, 32, 64])
def test_gemm1_a4w4_bm32(M):
    assert torch.cuda.is_available(), "needs a ROCm GPU"
    device = torch.device("cuda")
    w = build_mx_weights(device)
    r = _run_one(M, device, w)
    frac = r["mism_q"] / max(1, r["total_q"])
    print(
        f"\n[gemm1_a4w4 BM=32] M={M} cumsum={r['cumsum']} "
        f"exact_q={r['exact_q']} mism_q={r['mism_q']}/{r['total_q']} ({frac:.4%})"
    )
    # Packed fp4 + e8m0 requant: bit-exact 1:1 alignment is the goal.  Allow a
    # tiny fraction of off-by-one nibble rounding at the requant boundary.
    assert frac < 1e-3, f"too many mismatched fp4 bytes: {frac:.4%}"


if __name__ == "__main__":
    device = torch.device("cuda")
    w = build_mx_weights(device)
    for M in [16, 32, 64]:
        r = _run_one(M, device, w)
        frac = r["mism_q"] / max(1, r["total_q"])
        print(
            f"[gemm1_a4w4 BM=32] M={M} cumsum={r['cumsum']} "
            f"exact_q={r['exact_q']} mism_q={r['mism_q']}/{r['total_q']} ({frac:.4%})"
        )
