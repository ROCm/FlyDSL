import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from gpu_select import bind_empty_gpu_for_torch

bind_empty_gpu_for_torch("profile_flydsl_16384")

import torch
from torch.profiler import ProfilerActivity, profile


THIS_DIR = Path(__file__).resolve().parent
UP_AITER = Path("/root/up-aiter")
sys.path.insert(0, str(UP_AITER))
sys.path.insert(0, str(THIS_DIR))

from bench_flydsl_16384 import (  # noqa: E402
    M,
    SHAPE,
    build_inputs,
    build_weights,
    make_runners,
)


# Profiler windows need to be longer than smoke tests for stable per-kernel
# averages, especially in graph mode where the interesting unit is one logical
# pipeline iteration inside the captured graph.
DEFAULT_PROFILE_WARMUP = 100
DEFAULT_PROFILE_ITERS = 8
DEFAULT_PROFILE_GRAPH_ITERS = 64
DEFAULT_PROFILE_REPEAT = 21
DEFAULT_PROFILE_MAX_RETRIES = 400


def _is_cuda_event(evt) -> bool:
    device_type = str(getattr(evt, "device_type", ""))
    device_time = float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
    return device_time > 0.0 and "CUDA" in device_type.upper()


def _event_name(evt) -> str:
    return getattr(evt, "name", None) or getattr(evt, "key", "<unknown>")


def _profile_runner(fn, *, warmup: int, iters: int, trace_path: Path | None):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False, acc_events=True) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()

    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))

    events = [
        (_event_name(evt), float(evt.self_device_time_total))
        for evt in prof.events()
        if _is_cuda_event(evt)
    ]
    return events


def _capture_graph(fn, *, warmup: int, graph_iters: int):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_iters):
            fn()

    graph.replay()
    torch.cuda.synchronize()
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

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False, acc_events=True) as prof:
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()

    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))

    events = [
        (_event_name(evt), float(evt.self_device_time_total))
        for evt in prof.events()
        if _is_cuda_event(evt)
    ]
    return events


def _split_by_iteration(events, iters: int):
    if not events:
        return []
    if len(events) % iters != 0:
        return []
    per_iter = len(events) // iters
    chunks = []
    for i in range(iters):
        chunks.append(events[i * per_iter : (i + 1) * per_iter])
    return chunks


STAGE_NAMES = (
    "sort_count",
    "sort_cumsum",
    "sort_place_pad",
    "quant",
    "sort_scales",
    "GEMM1",
    "GEMM2",
    "scatter_reduce",
)


def _median(values) -> float:
    return float(statistics.median(values)) if values else 0.0


def _ordered_rows(events, iters: int):
    chunks = _split_by_iteration(events, iters)
    total_us = sum(us for _, us in events) / iters if iters else 0.0
    if not chunks:
        return None

    rows = []
    for idx in range(len(chunks[0])):
        names = [chunk[idx][0] for chunk in chunks]
        times = [chunk[idx][1] for chunk in chunks]
        same_name = all(item == names[0] for item in names)
        kernel = names[0] if same_name else "<mixed sequence>"
        avg_us = sum(times) / len(times)
        pct = 100.0 * avg_us / total_us if total_us else 0.0
        rows.append(
            {
                "idx": idx,
                "avg_us": avg_us,
                "pct": pct,
                "kernel": kernel,
            }
        )
    return {
        "events": events,
        "total_us": total_us,
        "kernel_calls_per_iter": len(chunks[0]),
        "rows": rows,
    }


def _print_ordered_report(name: str, events, iters: int, top: int | None):
    report = _ordered_rows(events, iters)
    total_us = sum(us for _, us in events) / iters if iters else 0.0
    print()
    print(f"== {name} ==")
    print(f"device_total_us_per_iter={total_us:.1f}")

    if report is None:
        print(f"device_kernel_events={len(events)}")
        print("warning=kernel sequence could not be evenly split by iteration")
        return

    per_iter = report["kernel_calls_per_iter"]
    print(f"device_kernel_calls_per_iter={per_iter}")
    print("ordered kernels:")
    print(f"{'idx':>3} {'avg_us':>9} {'pct':>6}  kernel")
    rows = report["rows"]

    rows_to_print = rows if top is None else rows[:top]
    for row in rows_to_print:
        print(
            f"{row['idx']:3d} {row['avg_us']:9.1f} "
            f"{row['pct']:5.1f}%  {row['kernel']}"
        )

    if top is not None and len(rows) > top:
        hidden_us = sum(row["avg_us"] for row in rows[top:])
        print(f"... hidden_tail_us={hidden_us:.1f}")

    grouped = defaultdict(lambda: [0, 0.0])
    for kernel, us in events:
        item = grouped[kernel]
        item[0] += 1
        item[1] += us

    print("grouped kernels:")
    print(f"{'calls/it':>8} {'avg_us/it':>10} {'avg_us/call':>12}  kernel")
    grouped_rows = []
    for kernel, (count, total) in grouped.items():
        calls_per_iter = count / iters
        avg_per_iter = total / iters
        avg_per_call = total / count
        grouped_rows.append((avg_per_iter, calls_per_iter, avg_per_call, kernel))
    grouped_rows.sort(reverse=True)
    grouped_to_print = grouped_rows if top is None else grouped_rows[:top]
    for avg_per_iter, calls_per_iter, avg_per_call, kernel in grouped_to_print:
        print(
            f"{calls_per_iter:8.1f} {avg_per_iter:10.1f} "
            f"{avg_per_call:12.1f}  {kernel}"
        )


def _collect_reports(
    name: str,
    fn,
    *,
    mode: str,
    warmup: int,
    iters: int,
    graph_iters: int,
    logical_iters: int,
    repeat: int,
    max_retries: int,
    trace_dir: Path | None,
    expected_kernel_calls: int | None,
):
    reports = []
    attempts = 0
    max_attempts = repeat + max_retries
    while len(reports) < repeat and attempts < max_attempts:
        attempts += 1
        trace_path = None
        if trace_dir is not None:
            trace_path = trace_dir / f"profile_16384_{name}_sample{len(reports) + 1}.json.gz"
        if mode == "graph":
            events = _profile_runner_graph(
                fn,
                warmup=warmup,
                graph_iters=graph_iters,
                replays=iters,
                trace_path=trace_path,
            )
        else:
            events = _profile_runner(
                fn,
                warmup=warmup,
                iters=iters,
                trace_path=trace_path,
            )
        report = _ordered_rows(events, logical_iters)
        if report is None:
            print(
                f"{name} sample attempt {attempts}: dropped "
                f"(device_kernel_events={len(events)})"
            )
            continue
        if (
            expected_kernel_calls is not None
            and report["kernel_calls_per_iter"] != expected_kernel_calls
        ):
            print(
                f"{name} sample attempt {attempts}: dropped "
                f"(kernels/iter={report['kernel_calls_per_iter']}, "
                f"expected={expected_kernel_calls})"
            )
            continue
        reports.append(report)
        print(
            f"{name} sample {len(reports)}/{repeat}: "
            f"device_total_us_per_iter={report['total_us']:.1f}, "
            f"kernels/iter={report['kernel_calls_per_iter']}"
        )
        if trace_path is not None:
            print(f"trace={trace_path}")

    if len(reports) < repeat:
        raise RuntimeError(
            f"{name}: only collected {len(reports)} valid profiler samples "
            f"after {attempts} attempts"
        )
    return reports


def _summarize_reports(name: str, reports, top: int | None):
    total_us = _median([report["total_us"] for report in reports])
    per_iter = reports[0]["kernel_calls_per_iter"]
    rows = []
    for idx in range(per_iter):
        values = [report["rows"][idx]["avg_us"] for report in reports]
        kernels = [report["rows"][idx]["kernel"] for report in reports]
        kernel = kernels[0] if all(item == kernels[0] for item in kernels) else "<mixed sequence>"
        avg_us = _median(values)
        rows.append(
            {
                "idx": idx,
                "stage": STAGE_NAMES[idx] if idx < len(STAGE_NAMES) else f"kernel_{idx}",
                "avg_us": avg_us,
                "pct": 100.0 * avg_us / total_us if total_us else 0.0,
                "kernel": kernel,
                "min_us": min(values),
                "max_us": max(values),
            }
        )

    print()
    print(f"== {name} median over {len(reports)} samples ==")
    print(f"device_total_us_per_iter={total_us:.1f}")
    print(f"device_kernel_calls_per_iter={per_iter}")
    print(f"{'idx':>3} {'stage':<15} {'median_us':>10} {'range_us':>17} {'pct':>6}  kernel")
    rows_to_print = rows if top is None else rows[:top]
    for row in rows_to_print:
        range_text = f"{row['min_us']:.1f}..{row['max_us']:.1f}"
        print(
            f"{row['idx']:3d} {row['stage']:<15} {row['avg_us']:10.1f} "
            f"{range_text:>17} {row['pct']:5.1f}%  {row['kernel']}"
        )
    if top is not None and len(rows) > top:
        hidden_us = sum(row["avg_us"] for row in rows[top:])
        print(f"... hidden_tail_us={hidden_us:.1f}")
    return {"name": name, "total_us": total_us, "rows": rows}


def _print_diff_report(base, candidate):
    if len(base["rows"]) != len(candidate["rows"]):
        print()
        print("== kernel diff ==")
        print("warning=runner kernel counts differ; ordered diff skipped")
        return

    print()
    print(f"== kernel diff: {candidate['name']} - {base['name']} ==")
    print(f"{'stage':<15} {'base_us':>9} {'cand_us':>9} {'diff_us':>9} {'diff_pct':>9}")
    for base_row, cand_row in zip(base["rows"], candidate["rows"]):
        diff_us = cand_row["avg_us"] - base_row["avg_us"]
        diff_pct = 100.0 * diff_us / base_row["avg_us"] if base_row["avg_us"] else 0.0
        print(
            f"{base_row['stage']:<15} {base_row['avg_us']:9.1f} "
            f"{cand_row['avg_us']:9.1f} {diff_us:9.1f} {diff_pct:8.1f}%"
        )
    total_diff = candidate["total_us"] - base["total_us"]
    total_pct = 100.0 * total_diff / base["total_us"] if base["total_us"] else 0.0
    print(
        f"{'total':<15} {base['total_us']:9.1f} "
        f"{candidate['total_us']:9.1f} {total_diff:9.1f} {total_pct:8.1f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runners",
        default="aiter_mxfp4_moe,local_mxfp4_all_flydsl",
        help="comma-separated runner names from bench_flydsl_16384.py",
    )
    parser.add_argument(
        "--mode",
        choices=("graph", "eager"),
        default="graph",
        help="profile CUDA/HIP graph replay by default; use eager for normal launch profiling",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_PROFILE_WARMUP)
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_PROFILE_ITERS,
        help="eager iterations, or graph replay count in graph mode",
    )
    parser.add_argument(
        "--graph-iters",
        type=int,
        default=DEFAULT_PROFILE_GRAPH_ITERS,
        help="pipeline iterations captured inside each graph replay",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULT_PROFILE_REPEAT,
        help="valid profiler samples to collect per runner",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_PROFILE_MAX_RETRIES,
        help="extra profiler attempts allowed when event sequences are incomplete",
    )
    parser.add_argument(
        "--expected-kernels",
        type=int,
        default=8,
        help="expected device kernels per logical iteration; set 0 to disable",
    )
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="optional directory for Chrome trace .json.gz files",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"M={M} NE={SHAPE.experts} H={SHAPE.model_dim} "
        f"INTER={SHAPE.inter_dim} TOPK={SHAPE.topk}"
    )
    if args.mode == "graph" and args.graph_iters <= 0:
        raise ValueError("--graph-iters must be positive in graph mode")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative")

    logical_iters = args.iters * args.graph_iters if args.mode == "graph" else args.iters
    print(
        f"mode={args.mode} warmup={args.warmup} "
        f"iters={args.iters} graph_iters={args.graph_iters if args.mode == 'graph' else 0} "
        f"logical_iters={logical_iters} repeat={args.repeat}"
    )

    weights = build_weights(SHAPE, device)
    hidden, topk_ids, topk_weight = build_inputs(SHAPE, device)
    runners = make_runners(hidden, topk_ids, topk_weight, weights)

    requested = [item.strip() for item in args.runners.split(",") if item.strip()]
    unknown = [item for item in requested if item not in runners]
    if unknown:
        raise ValueError(f"unknown runners: {', '.join(unknown)}")

    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    summaries = []
    for name in requested:
        reports = _collect_reports(
            name,
            runners[name],
            mode=args.mode,
            warmup=args.warmup,
            iters=args.iters,
            graph_iters=args.graph_iters,
            logical_iters=logical_iters,
            repeat=args.repeat,
            max_retries=args.max_retries,
            trace_dir=trace_dir,
            expected_kernel_calls=args.expected_kernels or None,
        )
        summaries.append(_summarize_reports(name, reports, args.top))

    if len(summaries) >= 2:
        _print_diff_report(summaries[0], summaries[1])


if __name__ == "__main__":
    main()
