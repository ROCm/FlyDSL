import argparse
import time
from dataclasses import dataclass
import os

import torch

import aiter
from aiter import dtypes, ActivationType, QuantType
from aiter.fused_moe import fused_moe, get_padded_M, torch_moe
from aiter.ops.shuffle import shuffle_weight, shuffle_scale
from aiter.utility.fp4_utils import e8m0_shuffle


@dataclass(frozen=True)
class Shape:
    NE: int
    H: int          # model_dim / D_HIDDEN
    INTER: int      # per-shard inter_dim / D_INTER
    TOPK: int


KIMI = Shape(NE=385, H=7168, INTER=512, TOPK=9)

# mxfp4 tuned dispatch per token bucket (from kimik2_5_mxfp4_tuned_fmoe.csv).
# (M_max, block_m, g1_suffix, g2_suffix)
MXFP4_TUNED = [
    (128,   16, "BM16_INLINEQUANT", "TOPK{topk}_BM16_ATOMIC_NT"),  # M<=128: BM16 inline-quant, g2 non-temporal
    (256,   32, "BM32_CACHED",      "TOPK{topk}_BM32_ATOMIC"),     # M<=256: BM32 cached
    (10**9, 128, "BM128",           "BM128_NONATOMIC"),            # M>256: BM128 non-atomic
]


def _mxfp4_knames(shape: Shape, M: int):
    for m_max, bm, g1, g2 in MXFP4_TUNED:
        if M <= m_max:
            base1 = f"mxfp4_moe_g1_a4w4_NE{shape.NE}_H{shape.H}_E{shape.INTER}"
            base2 = f"mxfp4_moe_g2_a4w4_NE{shape.NE}_H{shape.H}_E{shape.INTER}"
            return bm, f"{base1}_{g1}", f"{base2}_{g2.format(topk=shape.TOPK)}"
    raise RuntimeError("unreachable")


# ── one-time weight prep (M-independent) ─────────────────────────────────────
def build_weights(shape: Shape, device, seed=0):
    torch.manual_seed(seed)
    NE, H, I = shape.NE, shape.H, shape.INTER
    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)

    # Plain bf16 weights, scaled small (matches flydsl test conventions).
    w1 = torch.randn((NE, 2 * I, H), dtype=dtypes.bf16, device=device) / 10
    w2 = torch.randn((NE, H, I), dtype=dtypes.bf16, device=device) / 10

    # Quantize ONCE -> identical fp4 values fed to both backends.
    # torch_quant returns w1_qt as (E, 2I, H//2) fp4x2 and w1_scale as a 2D
    # (E*2I, H//32) e8m0 tensor.
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)

    # mxfp4_moe layout: equivalent to mxfp4_moe_preshuffle_w1/w2 in upstream aiter.
    m_w1q = shuffle_weight(
        w1_qt, layout=(16, 16), is_guinterleave=True, gate_up=True
    ).view(w1_qt.dtype)
    m_w2q = shuffle_weight(
        w2_qt, layout=(16, 16), is_guinterleave=True, gate_up=False
    ).view(w2_qt.dtype)
    m_w1s = shuffle_scale(
        w1_scale.view(NE * 2 * I, H // 32),
        experts_cnt=NE, is_guinterleave=True, gate_up=True,
    ).view(NE, 2 * I, H // 32)
    m_w2s = shuffle_scale(
        w2_scale.view(NE * H, I // 32),
        experts_cnt=NE, is_guinterleave=True, gate_up=False,
    ).view(NE, H, I // 32)
    # .view() above drops the is_shuffled attribute shuffle_weight sets internally;
    # fused_moe reads it off w1/w2 (fused_moe.py:339,1617) to pick tuned preshuffled kernels.
    # shuffle_kind="mxfp4_moe" is what gates dispatch onto the new mxfp4 pipeline
    # (fused_moe.py:418-429, 1504-1508); without it we silently fall back to the
    # untagged CSV path and never call the mxfp4_moe_gemm1/2_a4w4 kernels.
    for t in (m_w1q, m_w2q, m_w1s, m_w2s):
        t.is_shuffled = True
        t.shuffle_kind = "mxfp4_moe"

    # Flydsl uses a DIFFERENT preshuffle layout than mxfp4_moe -- see flydsl's own
    # test at FlyDSL/tests/kernels/test_moe_gemm.py:523-528,1137-1139, which for
    # the fp4 path does plain `shuffle_weight(w.view(fp4x2))` (no layout/guinter-
    # leave/gate_up kwargs) and `e8m0_shuffle(scale)` (NOT `shuffle_scale`).
    # Feeding flydsl the mxfp4_moe layout yields cos~0 (verified) -- correctness
    # requires matching its native preshuffle.
    f_w1q = shuffle_weight(w1_qt.view(torch.float4_e2m1fn_x2)).view(w1_qt.dtype)
    f_w2q = shuffle_weight(w2_qt.view(torch.float4_e2m1fn_x2)).view(w2_qt.dtype)
    f_w1s = e8m0_shuffle(w1_scale).view(NE, 2 * I, H // 32)
    f_w2s = e8m0_shuffle(w2_scale).view(NE, H, I // 32)
    # Same dispatch gating as the mxfp4 side, minus the shuffle_kind tag --
    # absence of the tag is what makes the CSV lookup hit the untagged flydsl_*
    # rows instead of the "mxfp4_moe"-tagged ones (fused_moe.py:1504-1508).
    for t in (f_w1q, f_w2q, f_w1s, f_w2s):
        t.is_shuffled = True

    return dict(
        mxfp4=dict(w1=m_w1q, w1_scale=m_w1s, w2=m_w2q, w2_scale=m_w2s),
        flydsl=dict(w1=f_w1q, w1_scale=f_w1s, w2=f_w2q, w2_scale=f_w2s),
        # Raw bf16 weights kept for torch_moe ground-truth comparison. Note these
        # are PRE-quantization, so cos(kernel, ref) won't hit 1.0 -- expect ~0.97+
        # for a correct mxfp4 kernel; <0.9 means the kernel itself is wrong.
        ref=dict(w1=w1, w2=w2),
    )


def build_inputs(shape: Shape, M: int, device, seed=1):
    torch.manual_seed(seed)
    NE, H, TOPK = shape.NE, shape.H, shape.TOPK
    hidden = torch.randn((M, H), dtype=dtypes.bf16, device=device) / 10
    # Production-faithful Kimi-K2.5 routing (matches tune_mxfp4_moe_gemm.py):
    # the shared expert (id NE-1, weight 1.0) is always at slot 0, plus TOPK-1
    # softmax-routed experts sampled with a mild per-expert popularity skew.
    # This concentrates tokens onto fewer experts (vs uniform-random routing),
    # so moe_sorting emits fewer padded expert-blocks -- the real serving load.
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


# ── per-backend e2e closures ─────────────────────────────────────────────────
def make_mxfp4_fn(shape, M, hidden, topk_ids, topk_weight, w, device):
    # mxfp4 weights carry shuffle_kind="mxfp4_moe" (set by mxfp4_moe_preshuffle_*),
    # so the high-level fused_moe dispatches per-M straight from the tuned CSV —
    # including the MXFP4OUT mxfp4-intermediate path on the largest buckets.
    def fn():
        return fused_moe(
            hidden, w["w1"], w["w2"], topk_weight, topk_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=w["w1_scale"], w2_scale=w["w2_scale"],
        )

    pm = get_padded_M(M)
    bm = 16 if pm <= 128 else (32 if pm <= 2048 else 128)
    return fn, bm, ("fused_moe(csv)",)


def make_flydsl_fn(shape, M, hidden, topk_ids, topk_weight, w, device):
    # Identical entry point to mxfp4 -- both call fused_moe(), which routes to
    # _flydsl_stage1/2_wrapper because the CSV row for this shape (NE=385/
    # INTER=512/H=7168) under the UNTAGGED dict has kernelName1/2 starting with
    # "flydsl_" (fused_moe.py:1666). The mxfp4 side hits a separate "mxfp4_moe"-
    # tagged row via shuffle_kind. Same sort/quant/scatter aux kernels run on both
    # sides -- only the gemm1/gemm2 differ -- so the perf delta is apples-to-apples.
    def fn():
        return fused_moe(
            hidden, w["w1"], w["w2"], topk_weight, topk_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=w["w1_scale"], w2_scale=w["w2_scale"],
        )

    return fn


# ── timing ───────────────────────────────────────────────────────────────────
def bench(fn, warmup=5, iters=30):
    """Eager wall-clock timing. Includes per-kernel CPU launch overhead, so on
    launch/latency-bound (small-M) shapes it OVER-estimates vs the real e2e path,
    which replays a single CUDA graph (one launch, no inter-kernel CPU bubbles)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - s)
    ts.sort()
    return ts[len(ts) // 2] * 1e6  # median us

tag = 0
def bench_cudagraph(fn, warmup=5, iters=60, measure=5):
    """CUDA-graph-faithful timing: capture `iters` back-to-back fn() calls into a
    SINGLE graph, replay the whole graph, and divide total GPU time by `iters`.
    The whole MoE pipeline runs as one graph -- CPU launch overhead and inter-
    kernel bubbles are excluded, matching the production e2e cudagraph path. The
    /iters amortization also dilutes the per-replay CUDA-event overhead. May raise
    during capture if any kernel in fn() does a host sync / host-dependent op."""
    # Side-stream warmup is REQUIRED before torch.cuda.graph(): it triggers JIT
    # compile + autotune and primes the caching allocator on a private stream.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    # Capture `iters` calls into one graph. Internal allocations come from the
    # graph's private pool (reused across the captured calls and every replay).
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()

    g.replay()  # warm the full graph once
    # torch.cuda.synchronize()

    start = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    totals = []
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    from contextlib import nullcontext
    with nullcontext() as prof:
    # with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
        g.replay()
        g.replay()
        for _ in range(measure):
            start[_].record()
            g.replay()
            end[_].record()
        torch.cuda.synchronize()
        for _ in range(measure):
            totals.append(start[_].elapsed_time(end[_]) * 1e3)  # ms -> us, total for `iters`
        totals.sort()
    global tag
    tag += 1
    out = f"{tag}.json.gz"
    print(f"dump to: {out}")
    if prof is not None:
        prof.export_chrome_trace(out)
    
    return totals[len(totals) // 2] / iters  # median per-call us


def cosine(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("-M", "--M-list", default="4,8,16,32,64,128,256,8192,16384")
    ap.add_argument("-M", "--M-list", default="16384")
    ap.add_argument("--check", default=True, action="store_true",
                    help="cross-backend output cosine-sim sanity check per M")
    ap.add_argument("--cudagraph", default=True, action="store_true",
                    help="time via CUDA-graph capture+replay (excludes CPU launch "
                         "overhead; faithful to the e2e cudagraph path). Default is "
                         "eager perf_counter, which includes launch overhead.")
    ap.add_argument("--iters", type=int, default=100,
                    help="eager perf_counter iterations")
    ap.add_argument("--graph-iters", type=int, default=60,
                    help="fn() calls captured per graph; total GPU time / this")
    ap.add_argument("--ref-max-M", type=int, default=16385,
                    help="run torch_moe ground-truth ref for M<=this (slow on large "
                         "M since torch_moe loops over experts in python).")
    args = ap.parse_args()

    device = torch.device("cuda")
    shape = KIMI
    Ms = [int(x) for x in args.M_list.split(",")]
    do_bench = bench_cudagraph if args.cudagraph else bench
    bench_iters = args.graph_iters if args.cudagraph else args.iters
    timing_mode = (f"cudagraph replay ({args.graph_iters}/graph)"
                   if args.cudagraph else "eager perf_counter")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Timing: {timing_mode}")
    print(f"Shape Kimi-K2.5 TP=4: NE={shape.NE} H={shape.H} "
          f"INTER={shape.INTER} TOPK={shape.TOPK}")
    print("Preparing weights (once)...", flush=True)
    w = build_weights(shape, device)

    # flydsl tile_m intentionally omitted: fused_moe picks it from the untagged
    # CSV row at dispatch time and bench has no clean hook to read it back; see
    # dispatcher log '[aiter] [fused_moe] using 2stage (kernelName1=...)' for
    # the real flydsl tile per M. mxfp4 bm is the bench-side MXFP4_TUNED table
    # value, hand-aligned with kimik2_5_mxfp4_tuned_fmoe.csv so it IS accurate.
    hdr = (f"\n{'M':>6} | {'mxfp4 us':>10} {'bm':>3} | "
           f"{'flydsl us':>10} | {'flydsl/mxfp4':>12}")
    if args.check:
        hdr += f" | {'mx⋅fly':>7} | {'mx⋅ref':>7} | {'fly⋅ref':>7}"
    print(hdr)
    print("-" * len(hdr))

    for M in Ms:
        hidden, topk_ids, topk_weight = build_inputs(shape, M, device)

        # torch_moe ground truth (bf16 weights, fp32 accum). Skip on big M --
        # python-level expert loop is slow on NE=385. cos vs ref tells you which
        # backend is broken when the two kernels disagree.
        ref_out = None
        if M <= args.ref_max_M:
            try:
                ref_out = torch_moe(
                    hidden, w["ref"]["w1"], w["ref"]["w2"],
                    topk_weight, topk_ids, activation=ActivationType.Silu,
                ).float()
            except Exception as e:
                print(f"  [ref M={M}] torch_moe ERROR: {type(e).__name__}: {e}")

        mx_fn, mx_bm, _ = make_mxfp4_fn(
            shape, M, hidden, topk_ids, topk_weight, w["mxfp4"], device)

        try:
            mx_out = mx_fn().float()
        except Exception as e:
            print(f"{M:>6} | mxfp4 ERROR: {type(e).__name__}: {e}")
            continue
        if not torch.isfinite(mx_out).all() or mx_out.abs().sum() == 0:
            print(f"{M:>6} | mxfp4 produced non-finite/zero output — skipping")
            continue
        mx_ref_cos = cosine(mx_out, ref_out) if ref_out is not None else float("nan")

        # flydsl: same fused_moe entry; CSV (untagged kimik2 rows) routes to
        # _flydsl_stage1/2_wrapper. tile_m is picked by the tuner -- no sweep.
        fly_ok, fly_us, fly_mx_cos, fly_ref_cos = False, float("nan"), float("nan"), float("nan")
        try:
            f_fn = make_flydsl_fn(
                shape, M, hidden, topk_ids, topk_weight, w["flydsl"], device)
            out = f_fn()
            finite = torch.isfinite(out).all().item()
            fly_mx_cos = cosine(mx_out, out) if finite else float("nan")
            fly_ref_cos = (cosine(out.float(), ref_out)
                           if finite and ref_out is not None else float("nan"))
            print(f"bench flydsl, M={M} mx⋅fly={fly_mx_cos:.4f} fly⋅ref={fly_ref_cos:.4f}")
            fly_us = do_bench(f_fn, iters=bench_iters)
            fly_ok = True
        except Exception as e:
            print(f"  [flydsl M={M}] ERROR: {type(e).__name__}: {e}", flush=True)

        try:
            print(f"bench mxfp4, M={M} bm={mx_bm} mx⋅ref={mx_ref_cos:.4f}")
            mx_us = do_bench(mx_fn, iters=bench_iters)
        except Exception as e:
            print(f"{M:>6} | mxfp4 timing ERROR: {type(e).__name__}: {e} — skipping")
            continue

        if not fly_ok:
            line = (f"{M:>6} | {mx_us:>10.1f} {mx_bm:>3} | "
                    f"{'flydsl FAIL':>10} | {'-':>12}")
            if args.check:
                line += f" | {'-':>7} | {mx_ref_cos:>7.4f} | {'-':>7}"
            print(line, flush=True)
            continue
        ratio = fly_us / mx_us
        line = (f"{M:>6} | {mx_us:>10.1f} {mx_bm:>3} | "
                f"{fly_us:>10.1f} | {ratio:>11.2f}x")
        if args.check:
            line += (f" | {fly_mx_cos:>7.4f} | {mx_ref_cos:>7.4f} | "
                     f"{fly_ref_cos:>7.4f}")
        print(line, flush=True)


if __name__ == "__main__":
    main()