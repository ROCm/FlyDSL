import argparse
import sys
from collections import defaultdict
from pathlib import Path

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

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
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


def _print_ordered_report(name: str, events, iters: int, top: int | None):
    chunks = _split_by_iteration(events, iters)
    total_us = sum(us for _, us in events) / iters if iters else 0.0

    print()
    print(f"== {name} ==")
    print(f"device_total_us_per_iter={total_us:.1f}")

    if not chunks:
        print(f"device_kernel_events={len(events)}")
        print("warning=kernel sequence could not be evenly split by iteration")
        return

    per_iter = len(chunks[0])
    print(f"device_kernel_calls_per_iter={per_iter}")
    print("ordered kernels:")
    print(f"{'idx':>3} {'avg_us':>9} {'pct':>6}  kernel")

    rows = []
    for idx in range(per_iter):
        names = [chunk[idx][0] for chunk in chunks]
        times = [chunk[idx][1] for chunk in chunks]
        same_name = all(item == names[0] for item in names)
        kernel = names[0] if same_name else "<mixed sequence>"
        avg_us = sum(times) / len(times)
        pct = 100.0 * avg_us / total_us if total_us else 0.0
        rows.append((idx, avg_us, pct, kernel))

    rows_to_print = rows if top is None else rows[:top]
    for idx, avg_us, pct, kernel in rows_to_print:
        print(f"{idx:3d} {avg_us:9.1f} {pct:5.1f}%  {kernel}")

    if top is not None and len(rows) > top:
        hidden_us = sum(row[1] for row in rows[top:])
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runners",
        default="aiter_mxfp4_moe,local_kimi_fp4,local_mxfp4_opt",
        help="comma-separated runner names from bench_flydsl_16384.py",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=5)
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
    print(f"warmup={args.warmup} iters={args.iters}")

    weights = build_weights(SHAPE, device)
    hidden, topk_ids, topk_weight = build_inputs(SHAPE, device)
    runners = make_runners(hidden, topk_ids, topk_weight, weights)

    requested = [item.strip() for item in args.runners.split(",") if item.strip()]
    unknown = [item for item in requested if item not in runners]
    if unknown:
        raise ValueError(f"unknown runners: {', '.join(unknown)}")

    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    for name in requested:
        trace_path = trace_dir / f"profile_16384_{name}.json.gz" if trace_dir else None
        events = _profile_runner(
            runners[name],
            warmup=args.warmup,
            iters=args.iters,
            trace_path=trace_path,
        )
        _print_ordered_report(name, events, args.iters, args.top)
        if trace_path is not None:
            print(f"trace={trace_path}")


if __name__ == "__main__":
    main()
