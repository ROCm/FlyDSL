#!/usr/bin/env python3
"""flex_attention kernel test and benchmark for FlyDSL.

Tests flydsl_flex_attention (score_mod / mask_mod on the generic flash-attention
kernel) against a PyTorch reference that applies the equivalent mods to the score
matrix. Mirrors test_flash_attn_fwd.py's structure.
"""

import argparse
import logging
import math
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

try:
    import torch
except ImportError:
    print("PyTorch not available")
    sys.exit(1)

if not torch.cuda.is_available():
    print("CUDA/ROCm not available")
    sys.exit(1)

import pytest  # noqa: E402

from kernels.attention.flex_attention import (  # noqa: E402
    alibi_score_mod,
    causal_mask_mod,
    flydsl_flex_attention,
    sliding_window_mask_mod,
)
from tests.kernels.test_flash_attn_fwd import _acc_metric, _flops  # noqa: E402
from tests.test_common import run_perftest  # noqa: E402

UNIFORM_RANGE = (-1, 1)
DEFAULT_SEED = 123

_DTYPE_MAP = {"bf16": torch.bfloat16, "f16": torch.float16}


# ── torch references for each mod (applied to the fp32 score matrix) ──────────


def _torch_ref(q, k, v, *, scale, score_bias=None, keep_mask=None):
    """Reference attention. q/k/v: [B, S, H, D]. GQA via repeat_interleave.

    score_bias: [Sq, Skv] additive bias applied AFTER scaling (matches score_mod
        seeing qk*scale, since an additive alibi bias is scale-independent).
    keep_mask: [Sq, Skv] bool; False positions -> -inf.
    """
    B, Sq, H, D = q.shape
    Skv = k.shape[1]
    Hkv = k.shape[2]
    qf = q.permute(0, 2, 1, 3).float()  # [B,H,Sq,D]
    kf = k.permute(0, 2, 1, 3).float()  # [B,Hkv,Skv,D]
    vf = v.permute(0, 2, 1, 3).float()
    if Hkv != H:
        rep = H // Hkv
        kf = kf.repeat_interleave(rep, dim=1)
        vf = vf.repeat_interleave(rep, dim=1)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale  # [B,H,Sq,Skv]
    if score_bias is not None:
        scores = scores + score_bias.view(1, 1, Sq, Skv)
    if keep_mask is not None:
        scores = scores.masked_fill(~keep_mask.view(1, 1, Sq, Skv), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # [B,H,Sq], natural log
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    out = torch.matmul(probs, vf)  # [B,H,Sq,D]
    return out.permute(0, 2, 1, 3).contiguous(), lse  # [B,Sq,H,D], [B,H,Sq]


def _alibi_bias(Sq, Skv, slope, device):
    q_idx = torch.arange(Sq, device=device).view(Sq, 1)
    kv_idx = torch.arange(Skv, device=device).view(1, Skv)
    return (kv_idx - q_idx).float() * slope


def _sliding_window_mask(Sq, Skv, window, device):
    q_idx = torch.arange(Sq, device=device).view(Sq, 1)
    kv_idx = torch.arange(Skv, device=device).view(1, Skv)
    return (kv_idx <= q_idx) & ((q_idx - kv_idx) <= window)


def _causal_mask(Sq, Skv, device):
    q_idx = torch.arange(Sq, device=device).view(Sq, 1)
    kv_idx = torch.arange(Skv, device=device).view(1, Skv)
    return kv_idx <= q_idx


# ── mod cases: (name, build flydsl mods, build torch ref args) ───────────────

_ALIBI_SLOPE = 0.125
_SW_WINDOW = 64


def _make_case(name, Sq, Skv, device):
    """Return (score_mod, mask_mod, torch_ref_kwargs) for a named case."""
    if name == "no_mod":
        return None, None, {}
    if name == "alibi":
        return (
            alibi_score_mod(_ALIBI_SLOPE),
            None,
            {"score_bias": _alibi_bias(Sq, Skv, _ALIBI_SLOPE, device)},
        )
    if name == "sliding_window":
        return (
            None,
            sliding_window_mask_mod(_SW_WINDOW),
            {"keep_mask": _sliding_window_mask(Sq, Skv, _SW_WINDOW, device)},
        )
    if name == "causal_via_mask":
        return (
            None,
            causal_mask_mod,
            {"keep_mask": _causal_mask(Sq, Skv, device)},
        )
    raise ValueError(f"unknown case {name}")


def run_flex_config(name, B, S, H, Hkv, D, dtype_str, *, S_kv=None, warmup=0, iters=0, return_lse=False, bench=False):
    dtype = _DTYPE_MAP[dtype_str]
    device = torch.device("cuda")
    torch.manual_seed(DEFAULT_SEED)

    S_kv = S if S_kv is None else S_kv
    q = torch.empty(B, S, H, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
    k = torch.empty(B, S_kv, Hkv, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
    v = torch.empty(B, S_kv, Hkv, D, dtype=dtype, device=device).uniform_(*UNIFORM_RANGE)
    scale = 1.0 / math.sqrt(D)

    score_mod, mask_mod, ref_kwargs = _make_case(name, S, S_kv, device)

    def _kernel():
        return flydsl_flex_attention(
            q,
            k,
            v,
            score_mod=score_mod,
            mask_mod=mask_mod,
            scale=scale,
            num_kv_heads=Hkv,
            return_lse=return_lse,
        )

    result = _kernel()
    out = result[0] if return_lse else result
    lse = result[1] if return_lse else None
    torch.cuda.synchronize()

    ref_out, ref_lse = _torch_ref(q, k, v, scale=scale, **ref_kwargs)
    max_err, min_cos, passed = _acc_metric(out.float(), ref_out.float(), D)
    logging.info(
        "[%s] B=%d S=%d Skv=%d H=%d Hkv=%d D=%d %s: max_err=%.4g min_cos=%.4g -> %s",
        name,
        B,
        S,
        S_kv,
        H,
        Hkv,
        D,
        dtype_str,
        max_err,
        min_cos,
        "PASS" if passed else "FAIL",
    )

    lse_err = None
    if return_lse:
        # kernel LSE is scale-folded natural log; ref_lse is natural log of scaled scores.
        finite = torch.isfinite(ref_lse)
        lse_err = (lse[finite] - ref_lse[finite]).abs().max().item() if bool(finite.any()) else 0.0
        logging.info("[%s] lse max_err=%.4g", name, lse_err)

    perf = None
    if bench:
        _, us = run_perftest(_kernel, num_iters=iters, num_warmup=warmup)
        perf = us

    return dict(passed=passed, max_err=max_err, min_cos=min_cos, lse_err=lse_err, us=perf)


# ── pytest ───────────────────────────────────────────────────────────────────

_CASES = ["no_mod", "alibi", "sliding_window", "causal_via_mask"]
_SHAPES = [
    (2, 256, 8, 8, 128),  # MHA
    (2, 256, 8, 2, 128),  # GQA
    (1, 256, 4, 4, 64),  # D=64
]


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("B,S,H,Hkv,D", _SHAPES)
def test_flex_attention(B, S, H, Hkv, D, case, dtype_str):
    r = run_flex_config(case, B, S, H, Hkv, D, dtype_str)
    assert r["passed"], f"{case}: max_err={r['max_err']} min_cos={r['min_cos']}"


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("case", ["no_mod", "alibi", "causal_via_mask"])
def test_flex_attention_lse(case):
    r = run_flex_config(case, 2, 256, 8, 8, 128, "bf16", return_lse=True)
    assert r["passed"], f"{case}: max_err={r['max_err']}"
    assert r["lse_err"] is not None and r["lse_err"] < 8e-3, f"{case}: lse_err={r['lse_err']}"


# Odd (non-multiple-of-64) and larger multi-tile seqlens, self-attention.
# S=250 exercises the pad-mask dispatch; S=384/1024 exercise multi-tile KV loops.
_SEQLEN_CASES = [
    (250, "sliding_window"),
    (250, "causal_via_mask"),
    (384, "causal_via_mask"),
    (1024, "causal_via_mask"),
]


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("S,case", _SEQLEN_CASES)
def test_flex_attention_seqlen(S, case):
    r = run_flex_config(case, 2, S, 8, 8, 128, "bf16")
    assert r["passed"], f"{case} S={S}: max_err={r['max_err']} min_cos={r['min_cos']}"


# Cross-attention (Sq != Skv). Masks use raw q_idx/kv_idx (top-left aligned),
# matching the kernel's non-causal mask_mod coordinates.
#
# sliding_window with Sq>Skv fully masks late query rows (q>=Skv+window have no
# valid kv); the kernel stores 0 for such rows (matching torch nan_to_num), which
# exercises the fully-masked-row guard in normalize_and_store_o.
_CROSS_CASES = [
    (128, 256, 8, 8, "no_mod"),
    (128, 256, 8, 8, "causal_via_mask"),
    (128, 256, 8, 8, "sliding_window"),  # Sq<=Skv: diagonal kv=q always kept
    (256, 128, 8, 8, "causal_via_mask"),  # Sq>Skv direction
    (256, 128, 8, 8, "sliding_window"),  # Sq>Skv: fully-masked late rows -> 0
    (128, 256, 8, 2, "causal_via_mask"),  # GQA
]


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("Sq,Skv,H,Hkv,case", _CROSS_CASES)
def test_flex_attention_cross(Sq, Skv, H, Hkv, case):
    r = run_flex_config(case, 2, Sq, H, Hkv, 128, "bf16", S_kv=Skv)
    assert r["passed"], f"{case} Sq={Sq} Skv={Skv}: max_err={r['max_err']} min_cos={r['min_cos']}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--num_heads", type=int, default=32)
    p.add_argument("--num_kv_heads", type=int, default=None)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "f16"])
    p.add_argument("--case", type=str, default="alibi", choices=_CASES)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    Hkv = args.num_kv_heads or args.num_heads
    r = run_flex_config(
        args.case,
        args.batch,
        args.seq_len,
        args.num_heads,
        Hkv,
        args.head_dim,
        args.dtype,
        warmup=args.warmup,
        iters=args.iters,
        bench=True,
    )
    if r["us"] is not None:
        flops = _flops(args.seq_len, args.seq_len, args.num_heads, args.head_dim, args.batch, causal=False)
        tflops = flops / (r["us"] * 1e-6) / 1e12
        # Column style mirrors test_flash_attn_fwd.py (_fmt_extra_normal_row / _fmt_result,
        # :1809/:2296). The "| St | MaxErr MinCos | Time TFLOPS" row form is what
        # scripts/run_benchmark.sh (_py_parse_and_emit) scrapes via its flash-attn table regex.
        prefix = (
            f"  {args.case:<16} B{args.batch:<3} S{args.seq_len:<6} "
            f"H{args.num_heads:>3} Hkv{Hkv:>3} D{args.head_dim:>3} {args.dtype:>5}"
        )
        status = "PASS" if r["passed"] else "FAIL"
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            gpu = "unknown"
        print(f"GPU: {gpu}")
        print(
            f"  {'config':<16} {'shape':<27} | {'St':>6} | {'MaxErr':>8} {'MinCos':>8} | {'Time(us)':>10} {'TFLOPS':>9}"
        )
        print(
            f"{prefix} | {status:>6} | {r['max_err']:>8.2e} {r['min_cos']:>8.5f} | " f"{r['us']:>10.1f} {tflops:>9.1f}"
        )


if __name__ == "__main__":
    main()
