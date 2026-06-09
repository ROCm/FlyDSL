import argparse
import statistics
from pathlib import Path

import bench_small_bm16 as small
import bench as b
from torch.profiler import ProfilerActivity, profile


STAGE_NAMES = ("sort_zero_init", "GEMM1", "GEMM2")
DEFAULT_PROFILE_WARMUP = 300
DEFAULT_PROFILE_GRAPH_ITERS = 128
DEFAULT_PROFILE_REPLAYS = 4
DEFAULT_PROFILE_REPEAT = 41
DEFAULT_PROFILE_MAX_RETRIES = 500


def _is_cuda_event(evt) -> bool:
    device_type = str(getattr(evt, "device_type", ""))
    device_time = float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
    return device_time > 0.0 and "CUDA" in device_type.upper()


def _event_name(evt) -> str:
    return getattr(evt, "name", None) or getattr(evt, "key", "<unknown>")


def _capture_graph(fn, *, warmup: int, graph_iters: int):
    side = b.torch.cuda.Stream()
    side.wait_stream(b.torch.cuda.current_stream())
    with b.torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    b.torch.cuda.current_stream().wait_stream(side)
    b.torch.cuda.synchronize()

    graph = b.torch.cuda.CUDAGraph()
    with b.torch.cuda.graph(graph):
        for _ in range(graph_iters):
            fn()

    graph.replay()
    b.torch.cuda.synchronize()
    return graph


def _profile_runner_graph(
    fn,
    *,
    warmup: int,
    graph_iters: int,
    replays: int,
    trace_path: Path | None,
):
    graph = _capture_graph(fn, warmup=warmup, graph_iters=graph_iters)

    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=False,
        acc_events=True,
    ) as prof:
        for _ in range(replays):
            graph.replay()
        b.torch.cuda.synchronize()

    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))

    return [
        (_event_name(evt), float(evt.self_device_time_total))
        for evt in prof.events()
        if _is_cuda_event(evt)
    ]


def _ordered_rows(events, logical_iters: int):
    if not events or logical_iters <= 0 or len(events) % logical_iters != 0:
        return None

    per_iter = len(events) // logical_iters
    chunks = [
        events[i * per_iter : (i + 1) * per_iter]
        for i in range(logical_iters)
    ]
    total_us = sum(us for _, us in events) / logical_iters
    rows = []
    for idx in range(per_iter):
        names = [chunk[idx][0] for chunk in chunks]
        times = [chunk[idx][1] for chunk in chunks]
        kernel = (
            names[0]
            if all(name == names[0] for name in names)
            else "<mixed sequence>"
        )
        rows.append(
            {
                "idx": idx,
                "stage": STAGE_NAMES[idx] if idx < len(STAGE_NAMES) else f"kernel_{idx}",
                "avg_us": sum(times) / len(times),
                "kernel": kernel,
            }
        )
    return {"total_us": total_us, "kernel_calls_per_iter": per_iter, "rows": rows}


def _collect_reports(
    m: int,
    name: str,
    fn,
    *,
    warmup: int,
    graph_iters: int,
    replays: int,
    repeat: int,
    max_retries: int,
    expected_kernels: int | None,
    trace_dir: Path | None,
):
    logical_iters = graph_iters * replays
    reports = []
    attempts = 0
    dropped = 0
    while len(reports) < repeat and attempts < repeat + max_retries:
        attempts += 1
        trace_path = None
        if trace_dir is not None:
            trace_path = (
                trace_dir
                / f"profile_small_bm16_M{m}_{name}_sample{len(reports) + 1}.json.gz"
            )
        events = _profile_runner_graph(
            fn,
            warmup=warmup,
            graph_iters=graph_iters,
            replays=replays,
            trace_path=trace_path,
        )
        report = _ordered_rows(events, logical_iters)
        if report is None:
            dropped += 1
            if dropped <= 3 or dropped % 25 == 0:
                print(
                    f"M={m} {name}: dropped profiler sample {dropped} "
                    f"(attempt={attempts}, device_kernel_events={len(events)})",
                    flush=True,
                )
            continue
        if (
            expected_kernels is not None
            and report["kernel_calls_per_iter"] != expected_kernels
        ):
            dropped += 1
            if dropped <= 3 or dropped % 25 == 0:
                print(
                    f"M={m} {name}: dropped profiler sample {dropped} "
                    f"(attempt={attempts}, kernels/iter={report['kernel_calls_per_iter']}, "
                    f"expected={expected_kernels})",
                    flush=True,
                )
            continue
        reports.append(report)
        print(
            f"M={m} {name} sample {len(reports)}/{repeat}: "
            f"device_total_us_per_iter={report['total_us']:.3f}, "
            f"kernels/iter={report['kernel_calls_per_iter']}",
            flush=True,
        )
    if len(reports) < repeat:
        raise RuntimeError(
            f"M={m} {name}: only collected {len(reports)} valid profiler "
            f"samples after {attempts} attempts"
        )
    if dropped:
        print(
            f"M={m} {name}: accepted {len(reports)} samples after {attempts} "
            f"attempts; dropped {dropped} incomplete profiler samples",
            flush=True,
        )
    return reports


def _median(values) -> float:
    return float(statistics.median(values))


def _summarize_reports(m: int, name: str, reports):
    per_iter = reports[0]["kernel_calls_per_iter"]
    total_us = _median([report["total_us"] for report in reports])
    rows = []
    for idx in range(per_iter):
        values = [report["rows"][idx]["avg_us"] for report in reports]
        kernels = [report["rows"][idx]["kernel"] for report in reports]
        kernel = (
            kernels[0]
            if all(item == kernels[0] for item in kernels)
            else "<mixed sequence>"
        )
        rows.append(
            {
                "idx": idx,
                "stage": STAGE_NAMES[idx] if idx < len(STAGE_NAMES) else f"kernel_{idx}",
                "avg_us": _median(values),
                "min_us": min(values),
                "max_us": max(values),
                "kernel": kernel,
            }
        )

    print()
    print(f"== M={m} {name} median over {len(reports)} profiler samples ==")
    print(f"device_total_us_per_iter={total_us:.3f}")
    print(f"device_kernel_calls_per_iter={per_iter}")
    print(f"{'idx':>3} {'stage':<14} {'median_us':>10} {'range_us':>17}  kernel")
    for row in rows:
        range_text = f"{row['min_us']:.3f}..{row['max_us']:.3f}"
        print(
            f"{row['idx']:3d} {row['stage']:<14} {row['avg_us']:10.3f} "
            f"{range_text:>17}  {row['kernel']}"
        )
    return {"m": m, "name": name, "total_us": total_us, "rows": rows}


def _print_diff(base, candidate):
    if len(base["rows"]) != len(candidate["rows"]):
        print(
            f"M={candidate['m']} {candidate['name']}: "
            "kernel count differs; diff skipped"
        )
        return
    print()
    print(f"== M={candidate['m']} kernel diff: {candidate['name']} - {base['name']} ==")
    print(
        f"{'stage':<14} {'base_us':>10} {'cand_us':>10} "
        f"{'diff_us':>10} {'diff_pct':>9}"
    )
    for base_row, cand_row in zip(base["rows"], candidate["rows"]):
        diff = cand_row["avg_us"] - base_row["avg_us"]
        pct = 100.0 * diff / base_row["avg_us"] if base_row["avg_us"] else 0.0
        print(
            f"{base_row['stage']:<14} {base_row['avg_us']:10.3f} "
            f"{cand_row['avg_us']:10.3f} {diff:10.3f} {pct:8.1f}%"
        )
    total_diff = candidate["total_us"] - base["total_us"]
    total_pct = 100.0 * total_diff / base["total_us"] if base["total_us"] else 0.0
    print(
        f"{'total':<14} {base['total_us']:10.3f} "
        f"{candidate['total_us']:10.3f} {total_diff:10.3f} {total_pct:8.1f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", "--M-list", default="64,128")
    parser.add_argument(
        "--runners",
        default="aiter,sort_aiter,gemm1fly_aiter,allfly",
        help=f"comma-separated subset of: {','.join(small.RUNNER_ORDER)}",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_PROFILE_WARMUP)
    parser.add_argument("--graph-iters", type=int, default=DEFAULT_PROFILE_GRAPH_ITERS)
    parser.add_argument("--replays", type=int, default=DEFAULT_PROFILE_REPLAYS)
    parser.add_argument("--repeat", type=int, default=DEFAULT_PROFILE_REPEAT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_PROFILE_MAX_RETRIES)
    parser.add_argument("--expected-kernels", type=int, default=3)
    parser.add_argument("--trace-dir", default=None)
    args = parser.parse_args()

    if args.graph_iters <= 0 or args.replays <= 0 or args.repeat <= 0:
        raise ValueError("--graph-iters, --replays, and --repeat must be positive")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative")

    shape = b.KIMI
    device = b.torch.device("cuda")
    ms = [int(item) for item in args.M_list.split(",") if item.strip()]
    requested = [item.strip() for item in args.runners.split(",") if item.strip()]
    unknown = [item for item in requested if item not in small.RUNNER_ORDER]
    if unknown:
        raise ValueError(f"unknown runners: {', '.join(unknown)}")
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    expected_kernels = args.expected_kernels or None

    print(f"GPU: {b.torch.cuda.get_device_name(0)}")
    print(f"Shape: NE={shape.NE} H={shape.H} INTER={shape.INTER} TOPK={shape.TOPK}")
    print(
        f"mode=graph warmup={args.warmup} graph_iters={args.graph_iters} "
        f"replays={args.replays} logical_iters={args.graph_iters * args.replays} "
        f"repeat={args.repeat} "
        f"logical_iters_per_runner={args.graph_iters * args.replays * args.repeat}"
    )
    print("Preparing weights...", flush=True)
    weights = b.build_weights(shape, device)

    for m in ms:
        hidden, topk_ids, topk_weight = b.build_inputs(shape, m, device)
        runners = small._make_runners(
            shape,
            m,
            hidden,
            topk_ids,
            topk_weight,
            weights,
            device,
        )
        summaries = []
        for name in requested:
            reports = _collect_reports(
                m,
                name,
                runners[name],
                warmup=args.warmup,
                graph_iters=args.graph_iters,
                replays=args.replays,
                repeat=args.repeat,
                max_retries=args.max_retries,
                expected_kernels=expected_kernels,
                trace_dir=trace_dir,
            )
            summaries.append(_summarize_reports(m, name, reports))
        if summaries:
            base = summaries[0]
            for candidate in summaries[1:]:
                _print_diff(base, candidate)


if __name__ == "__main__":
    main()
