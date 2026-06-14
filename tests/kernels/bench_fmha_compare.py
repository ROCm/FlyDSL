# SPDX-License-Identifier: Apache-2.0
"""Unified FMHA prefill benchmark: FlyDSL port variants vs the hand-written PyISA kernel.

Runs the customer (AITERKER-112) bs=1 causal shapes through:
  - one or more FlyDSL kernel modules (kernels/fmha_prefill_fp8*.py), via do_bench
  - the PyISA reference executable /workspaces/amir/asm/fwd_fp8 (parses its "time:/gflops:" line)

Usage:
  HIP_VISIBLE_DEVICES=2 python3 tests/kernels/bench_fmha_compare.py
  HIP_VISIBLE_DEVICES=2 python3 tests/kernels/bench_fmha_compare.py --kernels fmha_prefill_fp8_8wave fmha_prefill_fp8_v5
  HIP_VISIBLE_DEVICES=2 python3 tests/kernels/bench_fmha_compare.py --seqs 1024 16384 --no-pyisa

FLOP convention matches asm/fwd_fp8: causal FMHA = batch*nq*(2*sq*sk*hd_qk + 2*sq*sk*hd_v)/2.
"""

from __future__ import annotations

import argparse
import importlib
import re
import subprocess
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tests" / "kernels"))
sys.path.insert(0, str(_REPO / "kernels"))

import fmha_prefill_fp8_ref as R  # noqa: E402
from flydsl.autotune import do_bench  # noqa: E402

_PYISA_BIN = Path("/workspaces/amir/asm/fwd_fp8")
_PYISA_SYM_RE = re.compile(r"time:\s*([\d.]+)\s*ms.*gflops:\s*([\d.]+)", re.IGNORECASE)

# Customer cares about bs=1. (batch, seq, nq, nk, head_dim, causal, page_size)
DEFAULT_CASES = [
    (1, 1024, 8, 1, 128, 1, 16),
    (1, 16384, 8, 1, 128, 1, 16),
    (1, 32768, 8, 1, 128, 1, 16),
]
HD = 128


def causal_tflops(b, sq, sk, nq, ms):
    flop = b * nq * (2.0 * sq * sk * HD + 2.0 * sq * sk * HD) / 2.0
    return flop / 1e9 / ms


def bench_flydsl(mod_name, b, sq, sk, nq, nk, ps):
    K = importlib.import_module(mod_name)
    torch.manual_seed(0)
    sm = 1.0 / HD**0.5
    q = torch.randn(b, sq, nq, HD)
    k = torch.randn(b, sk, nk, HD)
    v = torch.randn(b, sk, nk, HD)
    qf, qd = R.quantize_per_token_head(q)
    kf, kd = R.quantize_per_token_head(k)
    vf, vd = R.quantize_per_head(v)
    c = R.pack_paged_cache(kf, vf, ps, scatter=True, v_col=getattr(K, "V_COL", False))
    args = [
        qf.to("cuda"),
        c.k_pool.view(torch.float8_e4m3fnuz).to("cuda"),
        c.v_pool.view(torch.float8_e4m3fnuz).to("cuda"),
        qd.to("cuda"),
        kd.to("cuda"),
        vd.to("cuda"),
        c.page_ids.to("cuda"),
        c.kv_indptr.to("cuda"),
        torch.full((b * nq,), 1.0, device="cuda"),
    ]
    Og = torch.zeros(b, sq, nq, HD, device="cuda", dtype=torch.bfloat16)
    grid = b * nq * ((sq + K.BM - 1) // K.BM)

    def fn():
        K.run_attn(*args, Og, sq, sk, nq, nk, ps, c.k_page_stride, c.v_page_stride, sm, 1, grid)

    fn()
    torch.cuda.synchronize()
    ms = do_bench(fn, warmup=10, rep=50)
    return ms, causal_tflops(b, sq, sk, nq, ms), grid


def bench_ck(b, sq, sk, nq, nk):
    """CK Tile fp8 paged FMHA via aiter's op_test helper (PER_TOKEN_HEAD vec_k_col_v).
    Requires the aiter repo on PYTHONPATH and its kernels JIT-compiled (first call ~5min)."""
    if b != 1:
        return None  # helper profile path validated for bs=1 here
    try:
        import torch as _t

        sys.path.insert(0, "/workspaces/amir/aiter")
        from op_tests.test_batch_prefill import run_batch_prefill_per_token_head as _run
    except Exception:
        return None
    try:
        r = _run(
            kvcache_layout="vec_k_col_v", table_layout="sglang", batch_size=b,
            qo_len=sq, kv_len=sk, page_size=64, num_qo_heads=nq, num_kv_heads=nk,
            head_dim=HD, causal=True, logits_soft_cap=0.0, dtype=_t.bfloat16,
            contiguous_kv=True, seed=42, profile=True, skip_reference=True,
        )
        return r["time_us"] / 1e3, r["tflops"], 0
    except Exception:
        return None


def bench_pyisa(b, sq, sk, nq, nk):
    if not _PYISA_BIN.exists():
        return None
    import os

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = "2"
    env.setdefault("LD_LIBRARY_PATH", "/opt/rocm/lib")
    proc = subprocess.run(
        [str(_PYISA_BIN), "causal=1", f"nheads={nq}", f"nheads_k={nk}", f"seq_len={sq}"],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=str(_PYISA_BIN.parent),  # fwd_fp8 hipModuleLoad's "fwd_causal.co" by relative path
    )
    m = _PYISA_SYM_RE.search(proc.stdout + proc.stderr)
    if not m:
        return None
    ms = float(m.group(1))
    return ms, float(m.group(2)) / 1e3, 1  # gflops→tflops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", nargs="+", default=["fmha_prefill_fp8_v8"])
    ap.add_argument("--seqs", nargs="+", type=int, default=None, help="override seq_lens (bs=1)")
    ap.add_argument("--no-pyisa", action="store_true")
    ap.add_argument("--ck", action="store_true", help="also bench CK Tile (aiter fp8); needs aiter built")
    args = ap.parse_args()

    cases = DEFAULT_CASES
    if args.seqs:
        cases = [(1, s, 8, 1, 128, 1, 16) for s in args.seqs]

    hdr = f"{'shape':<26}" + "".join(f"{k:>22}" for k in args.kernels)
    if args.ck:
        hdr += f"{'CK-Tile(fp8)':>22}"
    if not args.no_pyisa:
        hdr += f"{'PyISA(ref)':>22}"
    print(hdr)
    print("-" * len(hdr))

    for (b, sq, nq, nk, hd, causal, psz) in [(c[0], c[1], c[2], c[3], c[4], c[5], c[6]) for c in cases]:
        sk = sq
        row = f"b{b} sq{sq} nq{nq} nk{nk}".ljust(26)
        for mod in args.kernels:
            try:
                ms, tf, grid = bench_flydsl(mod, b, sq, sk, nq, nk, psz)
                row += f"{ms:.3f}ms/{tf:.0f}TF".rjust(22)
            except Exception as e:  # noqa: BLE001
                row += f"ERR:{str(e)[:14]}".rjust(22)
        if args.ck:
            r = bench_ck(b, sq, sk, nq, nk)
            row += (f"{r[0]:.3f}ms/{r[1]:.0f}TF".rjust(22)) if r else "n/a".rjust(22)
        if not args.no_pyisa:
            r = bench_pyisa(b, sq, sk, nq, nk)
            row += (f"{r[0]:.3f}ms/{r[1]:.0f}TF".rjust(22)) if r else "n/a".rjust(22)
        print(row)


if __name__ == "__main__":
    main()
