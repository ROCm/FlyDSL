# SPDX-License-Identifier: Apache-2.0
"""Device-fair FMHA bench: removes Python/JIT host-dispatch overhead via prebind +
CUDA-graph capture (the FlyKAT CompiledFunctionCache/prebind_launcher pattern), so the
timing is comparable to CK-Tile's device-side numbers.

`do_bench` in bench_fmha_compare.py times the @flyc.jit wrapper, whose per-call Python
dispatch is ~0.3ms — at small seq that dwarfs the kernel and makes us look far slower
than we are. Here we compile once, then time graph replay (host cost paid at capture).

Usage: HIP_VISIBLE_DEVICES=<g> python3 tests/kernels/bench_fmha_fair.py <module> [seqs...]
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tests" / "kernels"))
sys.path.insert(0, str(_REPO / "kernels"))

import importlib
import flydsl.compiler as flyc
import flydsl.expr as fx
import fmha_prefill_fp8_ref as R

HD = 128


def causal_tflops(b, sq, sk, nq, ms):
    return (b * nq * (2.0 * sq * sk * HD + 2.0 * sq * sk * HD) / 2.0) / 1e9 / ms


def bench(mod_name, sq, b=1, nq=8, nk=1, ps=16):
    K = importlib.import_module(mod_name)
    sk = sq
    sm = 1.0 / HD**0.5
    torch.manual_seed(0)
    q = torch.randn(b, sq, nq, HD); k = torch.randn(b, sk, nk, HD); v = torch.randn(b, sk, nk, HD)
    qf, qd = R.quantize_per_token_head(q); kf, kd = R.quantize_per_token_head(k); vf, vd = R.quantize_per_head(v)
    c = R.pack_paged_cache(kf, vf, ps, scatter=True, v_col=getattr(K, "V_COL", False))
    args = [
        qf.to("cuda"), c.k_pool.view(torch.float8_e4m3fnuz).to("cuda"), c.v_pool.view(torch.float8_e4m3fnuz).to("cuda"),
        qd.to("cuda"), kd.to("cuda"), vd.to("cuda"), c.page_ids.to("cuda"), c.kv_indptr.to("cuda"),
        torch.full((b * nq,), 1.0, device="cuda"),
    ]
    Og = torch.zeros(b, sq, nq, HD, device="cuda", dtype=torch.bfloat16)
    grid = b * nq * ((sq + K.BM - 1) // K.BM)
    tail = (sq, sk, nq, nk, ps, c.k_page_stride, c.v_page_stride, sm, 1, grid)

    # Prebind: compile once with the trailing stream param (FlyKAT prebind_launcher pattern).
    # Stream is re-resolved per call so torch.cuda.graph capture sees the capture stream.
    compiled = flyc.compile(K.run_attn, *args, Og, *tail, fx.Stream(torch.cuda.current_stream()))

    def call():
        compiled(*args, Og, *tail, fx.Stream(torch.cuda.current_stream()))

    for _ in range(10):
        call()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()

    ts = []
    for _ in range(50):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    ms = ts[len(ts) // 2]
    return ms, causal_tflops(b, sq, sk, nq, ms)


def main():
    mod = sys.argv[1] if len(sys.argv) > 1 else "fmha_prefill_fp8_ck_log2dom"
    seqs = [int(x) for x in sys.argv[2:]] or [1024, 2048, 16384, 32768]
    for sq in seqs:
        ms, tf = bench(mod, sq)
        print(f"  {mod} sq{sq}: {ms:.4f}ms / {tf:.0f}TF (device, graph)")


if __name__ == "__main__":
    main()
