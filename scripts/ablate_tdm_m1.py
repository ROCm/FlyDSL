#!/usr/bin/env python3
"""M=1 TDM load-only ablation, single process, compile fully excluded from timing.

Compiles all 5 variants {a,b,as,bs,all} up front (tdm_load_only is a COMPILE arg,
so each is a distinct cached kernel and they coexist in one process). Then, after
warmup, times each variant's kernel launches only — no compilation, no per-variant
subprocess, in the timed window.

wave map: a=wave0(A), b=wave1(B), as=wave2(A_scale), bs=wave3(B_scale).
Only "all" is numerically correct; the singles deliberately read stale LDS.

Two timers reported per variant:
  graph_us : hipGraph replay (warm L2; launch overhead amortized)
  cold_us  : per-iter L2 flush + direct launch (cold weight read)
"""
import argparse
import statistics
import sys

import torch

import flydsl.compiler as flyc
import tests.kernels.test_gemm_fp8fp4_gfx1250 as t
from kernels.gemm_fp8fp4_gfx1250 import compile_mxscale_gemm

# "b_split" = B streamed by two waves (32-N each), A_scale wave repurposed.
VARIANTS = ["a", "b", "as", "bs", "all", "b_split"]

# Configs from the ablation set, keyed by M: (M,N,K,tile_m,tile_n,tile_k,mw,nw,nb)
CONFIGS = {
    1: (1, 12288, 3072, 16, 64, 512, 1, 4, 4),
    4096: (4096, 12288, 3072, 32, 256, 256, 1, 4, 4),
}


def _build_inputs(M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, split_k=1):
    data_format = "a8w4"
    ps = t._get_padded_problem_shape(data_format, M, N, K, tile_m, tile_n, tile_k, split_k)
    pm, pn, pk = ps["M"], ps["N"], ps["K"]
    PACK_B = ps["pack_b"]
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp

    a, b, a_scale, b_scale, _ = t._fill_mode_inputs(M, N, K, data_format, "random")
    a, b, a_scale, b_scale = t._pad_mxscale_inputs(a, b, a_scale, b_scale, ps)
    skt = tile_k // t.SCALE_BLOCK
    a_scale = t.preshuffle_e8m0_scale(a_scale, warp_tile_m, scale_k_per_tile=skt, coalesced=False)
    b_scale = t.preshuffle_e8m0_scale(b_scale, warp_tile_n, scale_k_per_tile=skt, coalesced=False)
    K_packed = pk // PACK_B
    b = t.fp4_utils.preshuffle_b_16x16(b, pn, K_packed)

    a_gpu = a.cuda(); b_gpu = b.cuda()
    as_gpu = a_scale.cuda(); bs_gpu = b_scale.cuda()
    c_gpu = torch.zeros(pm, pn, dtype=torch.bfloat16, device="cuda")
    # launch sig: (c, a, b, a_scale, b_scale, M, N, lda, ldc, stream); lda=pk, ldc=pn
    return (c_gpu, a_gpu, b_gpu, as_gpu, bs_gpu, pm, pn, pk, pn)


def _time_graph(run_fn, warmup, iters, n_per_graph=20):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            run_fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(n_per_graph):
            run_fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        g.replay()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) * 1e3 / (iters * n_per_graph)


def _time_cold(run_fn, warmup, iters):
    l2 = max(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 4 << 20) * 2, 8 << 20)
    flush = torch.empty(l2, dtype=torch.uint8, device="cuda")
    for _ in range(warmup):
        flush.zero_(); run_fn()
    torch.cuda.synchronize()
    se = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush.zero_()
        se[i].record(); run_fn(); ee[i].record()
    torch.cuda.synchronize()
    lat = sorted(se[i].elapsed_time(ee[i]) * 1e3 for i in range(iters))
    n = len(lat)
    if n >= 8:
        q1, q3 = lat[n // 4], lat[3 * n // 4]; iqr = q3 - q1
        lat = [x for x in lat if q1 - 1.5 * iqr <= x <= q3 + 1.5 * iqr] or lat
    return statistics.mean(lat), min(lat), max(lat)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--M", type=int, default=1, choices=sorted(CONFIGS))
    args = p.parse_args()

    M, N, K, tm, tn, tk, mw, nw, nb = CONFIGS[args.M]
    print(f"M={M} N={N} K={K} tile=({tm},{tn},{tk}) warps={mw}x{nw} buffers={nb}", flush=True)

    inp = _build_inputs(M, N, K, tm, tn, tk, mw, nw)
    c_gpu = inp[0]
    stream = torch.cuda.current_stream()

    # ---- Phase 1: compile + first-launch ALL variants (NOT timed) ----
    print("[compile] building all variants up front (excluded from timing)...", flush=True)
    execs = {}
    for v in VARIANTS:
        kw = dict(
            data_format="a8w4", N=inp[6], K=inp[7],
            tile_m=tm, tile_n=tn, tile_k=tk, m_warp=mw, n_warp=nw, num_buffers=nb,
            out_dtype="bf16", wave_specialized_tdm=True, scale_load_path="tdm",
        )
        if v == "b_split":
            kw["tdm_b_split"] = True
        else:
            kw["tdm_load_only"] = v
        fn = compile_mxscale_gemm(**kw)
        ex = flyc.compile(fn, *inp, stream)
        ex(*inp, stream)  # first launch (JIT settle) — outside timed region
        execs[v] = ex
    torch.cuda.synchronize()
    print("[compile] done.\n", flush=True)

    # ---- Phase 2: time each variant (launches only) ----
    rows = []
    for v in VARIANTS:
        ex = execs[v]

        def run():
            ex(*inp, stream)

        c_avg, c_min, c_max = _time_cold(run, args.warmup, args.iters)
        rows.append((v, c_avg, c_min, c_max))
        print(f"  [{v:>6}] cold avg/min/max = "
              f"{c_avg:6.2f}/{c_min:6.2f}/{c_max:6.2f} us", flush=True)

    print(f"\n## M=1 TDM load-only ablation (single process, compile excluded, cold L2)\n")
    print("| variant | cold avg | cold min | cold max |")
    print("|---|---|---|---|")
    for v, ca, cmin, cmax in rows:
        print(f"| {v} | {ca:.2f} | {cmin:.2f} | {cmax:.2f} |")
    print("\nwave: a=A, b=B, as=A_scale, bs=B_scale; 'all'=production; "
          "'b_split'=B over 2 waves (32-N each). Only 'all' is numerically valid.")


if __name__ == "__main__":
    main()
