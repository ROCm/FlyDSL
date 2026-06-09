import argparse
import statistics

import bench as b
from kimi_fp4_moe_small_bm16 import (
    run_kimi_fp4_mxfp4_moe_small_bm16_all_flydsl,
    run_kimi_fp4_mxfp4_moe_small_bm16_flydsl_gemm1_aiter_gemm2,
    run_kimi_fp4_mxfp4_moe_small_bm16_flydsl_sort_aiter_gemm,
)


RUNNER_ORDER = (
    "aiter",
    "sort_aiter",
    "gemm1fly_aiter",
    "allfly",
)


def _time_fn(fn, args):
    samples = []
    for _ in range(args.repeat):
        if args.eager:
            samples.append(b.bench(fn, warmup=args.warmup, iters=args.iters))
        else:
            samples.append(
                b.bench_cudagraph(
                    fn,
                    warmup=args.warmup,
                    iters=args.graph_iters,
                    measure=args.measure,
                    graph_warmup_replays=args.graph_warmup_replays,
                )
            )
    return float(statistics.median(samples)), samples


def _fmt_samples(samples):
    if len(samples) <= 1:
        return ""
    body = ",".join(f"{sample:.3f}" for sample in samples)
    return f" samples={body}"


def _make_runners(shape, m, hidden, topk_ids, topk_weight, weights, device):
    w = weights["mxfp4"]
    aiter_fn, _, _ = b.make_mxfp4_fn(
        shape, m, hidden, topk_ids, topk_weight, w, device
    )

    def sort_aiter():
        return run_kimi_fp4_mxfp4_moe_small_bm16_flydsl_sort_aiter_gemm(
            hidden,
            w["w1"],
            w["w2"],
            topk_weight,
            topk_ids,
            w1_scale=w["w1_scale"],
            w2_scale=w["w2_scale"],
        )

    def gemm1fly_aiter():
        return run_kimi_fp4_mxfp4_moe_small_bm16_flydsl_gemm1_aiter_gemm2(
            hidden,
            w["w1"],
            w["w2"],
            topk_weight,
            topk_ids,
            w1_scale=w["w1_scale"],
            w2_scale=w["w2_scale"],
        )

    def allfly():
        return run_kimi_fp4_mxfp4_moe_small_bm16_all_flydsl(
            hidden,
            w["w1"],
            w["w2"],
            topk_weight,
            topk_ids,
            w1_scale=w["w1_scale"],
            w2_scale=w["w2_scale"],
        )

    return {
        "aiter": aiter_fn,
        "sort_aiter": sort_aiter,
        "gemm1fly_aiter": gemm1fly_aiter,
        "allfly": allfly,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", "--M-list", default="4,8,16,32,64,128")
    parser.add_argument("--warmup", type=int, default=b.DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=b.DEFAULT_EAGER_ITERS)
    parser.add_argument("--graph-iters", type=int, default=b.DEFAULT_GRAPH_ITERS)
    parser.add_argument("--measure", type=int, default=b.DEFAULT_GRAPH_MEASURE)
    parser.add_argument(
        "--graph-warmup-replays",
        type=int,
        default=b.DEFAULT_GRAPH_WARMUP_REPLAYS,
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")

    shape = b.KIMI
    device = b.torch.device("cuda")
    ms = [int(item) for item in args.M_list.split(",") if item.strip()]

    mode = (
        f"eager warmup={args.warmup} iters={args.iters}"
        if args.eager
        else (
            f"graph warmup={args.warmup} graph_iters={args.graph_iters} "
            f"measure={args.measure} graph_warmup_replays={args.graph_warmup_replays}"
        )
    )
    print(f"GPU: {b.torch.cuda.get_device_name(0)}")
    print(f"Timing: {mode} repeat={args.repeat}")
    print(
        f"Shape: NE={shape.NE} H={shape.H} INTER={shape.INTER} TOPK={shape.TOPK}"
    )
    print("Preparing weights...", flush=True)
    weights = b.build_weights(shape, device)

    header = (
        "M,aiter_us,sort_aiter_us,gemm1fly_aiter_us,allfly_us,"
        "sort_delta_us,gemm1_delta_us,gemm2_delta_us,"
        "cos_sort,cos_gemm1,cos_all,max_abs_all"
    )
    print(header)

    for m in ms:
        hidden, topk_ids, topk_weight = b.build_inputs(shape, m, device)
        runners = _make_runners(
            shape, m, hidden, topk_ids, topk_weight, weights, device
        )

        cos = {"sort_aiter": float("nan"), "gemm1fly_aiter": float("nan"), "allfly": float("nan")}
        max_abs_all = float("nan")
        if not args.no_check:
            ref = runners["aiter"]().float()
            b.torch.cuda.synchronize()
            for name in ("sort_aiter", "gemm1fly_aiter", "allfly"):
                out = runners[name]().float()
                b.torch.cuda.synchronize()
                if not b.torch.isfinite(out).all().item():
                    raise RuntimeError(f"{name} M={m} produced non-finite output")
                cos[name] = b.cosine(ref, out)
                if name == "allfly":
                    max_abs_all = (ref - out).abs().max().item()

        timings = {}
        for name in RUNNER_ORDER:
            timings[name], samples = _time_fn(runners[name], args)
            print(
                f"sample M={m} {name}={timings[name]:.3f} us{_fmt_samples(samples)}",
                flush=True,
            )

        sort_delta = timings["sort_aiter"] - timings["aiter"]
        gemm1_delta = timings["gemm1fly_aiter"] - timings["sort_aiter"]
        gemm2_delta = timings["allfly"] - timings["gemm1fly_aiter"]
        print(
            f"{m},{timings['aiter']:.3f},{timings['sort_aiter']:.3f},"
            f"{timings['gemm1fly_aiter']:.3f},{timings['allfly']:.3f},"
            f"{sort_delta:.3f},{gemm1_delta:.3f},{gemm2_delta:.3f},"
            f"{cos['sort_aiter']:.9f},{cos['gemm1fly_aiter']:.9f},"
            f"{cos['allfly']:.9f},{max_abs_all:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
