import argparse
import statistics

import bench as b
from kimi_fp4_moe_small_bm16 import (
    run_kimi_fp4_mxfp4_moe_small_bm16_all_flydsl,
)


RUNNER_ORDER = (
    "aiter",
    "allfly",
)

# Accuracy-first defaults for the tiny BM16 path. Capture enough calls per graph
# to amortize event/replay noise, but keep the default run finite for iterative
# kernel work. Override downward only for smoke tests.
DEFAULT_SMALL_WARMUP = 1000
DEFAULT_SMALL_EAGER_ITERS = 500000
DEFAULT_SMALL_GRAPH_ITERS = 512
DEFAULT_SMALL_GRAPH_MEASURE = 101
DEFAULT_SMALL_GRAPH_WARMUP_REPLAYS = 10
DEFAULT_REPEAT = 1


def _time_fn(fn, args):
    samples = []
    for _ in range(args.repeat):
        if args.eager:
            samples.append(b.bench(fn, warmup=args.warmup, iters=args.iters))
        else:
            print(f"{args.warmup} {args.graph_iters} {args.measure} {args.graph_warmup_replays}")
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
    return (
        f" samples={body}"
        f" range_us={min(samples):.3f}..{max(samples):.3f}"
        f" stdev_us={statistics.stdev(samples):.3f}"
    )


def _make_runners(shape, m, hidden, topk_ids, topk_weight, weights, device):
    w = weights["mxfp4"]
    aiter_fn, _, _ = b.make_mxfp4_fn(
        shape, m, hidden, topk_ids, topk_weight, w, device
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
        "allfly": allfly,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", "--M-list", default="4,8,16,32,64,128")
    parser.add_argument("--warmup", type=int, default=DEFAULT_SMALL_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_SMALL_EAGER_ITERS)
    parser.add_argument("--graph-iters", type=int, default=DEFAULT_SMALL_GRAPH_ITERS)
    parser.add_argument("--measure", type=int, default=DEFAULT_SMALL_GRAPH_MEASURE)
    parser.add_argument(
        "--graph-warmup-replays",
        type=int,
        default=DEFAULT_SMALL_GRAPH_WARMUP_REPLAYS,
    )
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
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
            f"measure={args.measure} graph_warmup_replays={args.graph_warmup_replays} "
            f"measured_calls_per_sample={args.graph_iters * args.measure} "
            f"measured_calls_per_runner={args.graph_iters * args.measure * args.repeat}"
        )
    )
    print(f"GPU: {b.torch.cuda.get_device_name(0)}")
    print(f"Timing: {mode} repeat={args.repeat}")
    print(
        f"Shape: NE={shape.NE} H={shape.H} INTER={shape.INTER} TOPK={shape.TOPK}"
    )
    print("Preparing weights...", flush=True)
    weights = b.build_weights(shape, device)

    header = "M,aiter_us,allfly_us,allfly_delta_us,cos_all,max_abs_all"
    print(header)

    for m in ms:
        hidden, topk_ids, topk_weight = b.build_inputs(shape, m, device)
        runners = _make_runners(
            shape, m, hidden, topk_ids, topk_weight, weights, device
        )

        cos_all = float("nan")
        max_abs_all = float("nan")
        if not args.no_check:
            ref = runners["aiter"]().float()
            b.torch.cuda.synchronize()
            out = runners["allfly"]().float()
            b.torch.cuda.synchronize()
            if not b.torch.isfinite(out).all().item():
                raise RuntimeError(f"allfly M={m} produced non-finite output")
            cos_all = b.cosine(ref, out)
            max_abs_all = (ref - out).abs().max().item()

        timings = {}
        for name in RUNNER_ORDER:
            timings[name], samples = _time_fn(runners[name], args)
            print(
                f"sample M={m} {name}={timings[name]:.3f} us{_fmt_samples(samples)}",
                flush=True,
            )

        allfly_delta = timings["allfly"] - timings["aiter"]
        print(
            f"{m},{timings['aiter']:.3f},{timings['allfly']:.3f},"
            f"{allfly_delta:.3f},{cos_all:.9f},{max_abs_all:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
