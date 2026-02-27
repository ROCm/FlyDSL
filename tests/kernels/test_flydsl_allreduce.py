#!/usr/bin/env python3
"""FlyDSL all-reduce kernel tests with accuracy and performance profiling (multi-GPU only).

This test file provides:
- **Accuracy tests** for FlyDSL allreduce operator (multi-GPU)
- **Performance tests** using torch.profiler (multi-GPU)
- **CSV export** of performance data
"""

import os
import sys
from pathlib import Path
import socket

# Prefer embedded MLIR/flydsl to avoid mixing multiple runtimes.
_repo = Path(__file__).resolve().parents[2]
_embedded = _repo / "build" / "python_packages" / "flydsl"
_embedded2 = _repo / ".flir" / "build" / "python_packages" / "flydsl"
_embedded_pick = _embedded if _embedded.exists() else _embedded2
if _embedded_pick.exists():
    os.environ.setdefault("FLYDSL_USE_EMBEDDED_MLIR", "1")
    sys.path.insert(0, str(_embedded_pick))
_src_py = _repo / "python"
if _src_py.exists():
    sys.path.insert(0, str(_src_py))
sys.path.insert(0, str(_repo))

import pytest

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

import torch.profiler as tpf
import pandas as pd
import numpy as np
from multiprocessing import freeze_support, set_start_method

DTYPE_FP32 = torch.float32
DTYPE_FP16 = torch.float16
DTYPE_BF16 = torch.bfloat16

from kernels.custom_all_reduce import init_custom_ar, meta_size


# ============================================================================
# Performance profiling utilities (copied from aiter/aiter/test_common.py)
# ============================================================================

def post_process_data(df, num_iters, warm_iter=1):
    """Remove abnormal data from profiling results."""
    device_df = df[df["device_type"].astype(str).str.contains("DeviceType.CUDA")]
    if device_df.empty:
        return [], 0
    kernels_num = int(len(device_df) / num_iters)

    act_iters = num_iters
    valid_n = len(device_df)
    dropped_indexs = []
    if len(device_df) % num_iters == 0:
        kernels_num = int(len(device_df) / num_iters)
    else:
        # Get correct kernel num
        name_list = device_df["name"].tolist()
        max_kernel_num = 20
        n = len(name_list)
        for step in range(1, min(max_kernel_num, n // 2 + 1)):
            sub_list = [name_list[i] for i in range(step)]
            m = len(sub_list)

            valid_n = int(n / m) * m
            pattern_match = all(
                name_list[i] == sub_list[i % m] for i in range(int(n / m) * m)
            )
            if pattern_match:
                kernels_num = m
                act_iters = valid_n / m
                break
        dropped_indexs = device_df.iloc[valid_n:].index.tolist()
        if kernels_num == 0:
            print("data missed, the time may be inaccurate!")

    test_df = device_df.iloc[:valid_n].reset_index()
    grouped_kernel_df = test_df.groupby(test_df.index // kernels_num, sort=False).agg(
        {"self_device_time_total": "sum", "index": list}
    )

    # Remove warm iters
    sum_df = grouped_kernel_df.iloc[warm_iter:].reset_index(drop=True)
    out_range_idx = []
    if num_iters > 30:
        # IQR to remove abnormal data
        k = 1.5
        Q1 = sum_df["self_device_time_total"].quantile(0.25)
        Q3 = sum_df["self_device_time_total"].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        out_range_idx = sum_df.index[
            (sum_df["self_device_time_total"] < lower)
            | (sum_df["self_device_time_total"] > upper)
        ].tolist()
    out_range_num = len(out_range_idx)

    indices = {idx for i in out_range_idx for idx in sum_df.iloc[i]["index"]}

    index_sublists = grouped_kernel_df["index"].head(warm_iter).tolist()
    indices_to_add = [idx for sublist in index_sublists for idx in sublist]
    indices.update(indices_to_add)
    indices.update(dropped_indexs)
    if int(os.environ.get("AITER_LOG_MORE", 0)):
        print(f"abnormal data indices: {indices}")
        for i in indices:
            print(f"abnormal data: {df.iloc[i]['self_device_time_total']}")
    return list(indices), out_range_num + warm_iter + num_iters - act_iters


def get_trace_perf(prof, num_iters):
    """Extract performance data from torch.profiler results."""
    assert num_iters > 1
    warm_iter = 1
    num_iters -= warm_iter
    df = []
    cols = [
        "name",
        "self_cpu_time_total",
        "self_device_time_total",
        "device_type",
        "device_index",
    ]
    for el in prof.events():
        df.append([getattr(el, x, None) for x in cols])
    df = pd.DataFrame(df, columns=cols)
    # Remove abnormal data
    dropped_num = warm_iter
    dropped_indexs, dropped_num = post_process_data(
        df, num_iters + warm_iter, warm_iter
    )
    df = df.drop(dropped_indexs)
    iter_init = 0  # warm_iter dropped
    df["cnt"] = 1
    rets = []

    for name, d in df.groupby("name", sort=False):
        kernel_num_per_iter = iter_init
        if str(d["device_type"].iat[0]).split(".")[-1] != "CUDA":
            kernel_num_per_iter = 1
        r = d.iloc[kernel_num_per_iter:][
            ["cnt", "self_cpu_time_total", "self_device_time_total"]
        ].sum()
        if not r.empty:
            device_type = str(d["device_type"].iat[0]).split(".")[-1]
            r["name"] = name
            r["device_type"] = device_type
            r["device_index"] = str(d["device_index"].iat[0])
            if device_type == "CUDA":
                r["device_time_sum"] = r["self_device_time_total"]
                r["host_time_sum"] = 0
            else:
                r["host_time_sum"] = r["self_device_time_total"]
                r["device_time_sum"] = 0
            r["device_time_avg"] = (
                r["device_time_sum"] / r["cnt"] if r["cnt"] > 0 else 0
            )
        rets.append(r)
    df = pd.DataFrame(rets)
    cols = [
        "name",
        "cnt",
        "host_time_sum",
        "device_time_sum",
        "device_time_avg",
        "device_type",
        "device_index",
    ]
    cols = [el for el in cols if el in df.columns]
    df = df[(df.host_time_sum > 0) | (df.device_time_sum > 0)]

    timerList = [
        "host_time_sum",
        "device_time_sum",
    ]
    df = df[cols].sort_values(timerList, ignore_index=True)
    actual_iters = num_iters + warm_iter - dropped_num
    if df.empty:
        print("no valid data after post process!")

    avg_name = "[avg us/iter]"
    for el in timerList:
        if el == "host_time_sum":
            df.at[avg_name, el] = df[el].sum() / num_iters
        else:
            df.at[avg_name, el] = df[el].sum() / actual_iters
    if int(os.environ.get("AITER_LOG_MORE", 0)):
        pd.set_option("display.expand_frame_repr", False)
        pd.set_option("display.max_colwidth", 90)
        pd.set_option("display.float_format", "{:,.1f}".format)
        print(f"{df}")
    return df


def _free_port() -> int:
    """Get a free port for distributed communication."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _normalize_dtype_arg(dtype_arg: str) -> str:
    """Accept both AIter-style names (fp16/bf16/fp32) and internal (f16/bf16/f32)."""
    d = (dtype_arg or "").strip().lower()
    if d in {"fp16", "f16"}:
        return "f16"
    if d in {"bf16"}:
        return "bf16"
    if d in {"fp32", "f32"}:
        return "f32"
    raise ValueError(f"unsupported dtype: {dtype_arg}")


# ============================================================================
# Distributed worker function for multi-GPU testing
# ============================================================================

def _dist_worker(
    rank: int,
    world_size: int,
    shape,
    dtype_str: str,
    port: int,
    num_iters: int,
    num_warmup: int,
    skip_check: bool,
    allreduce_impl: str,
    mode: str,
    result_dict: dict,
):
    """Worker function for distributed allreduce testing.
    
    Args:
        rank: Process rank
        world_size: Number of processes/GPUs
        shape: Shape of input tensor (tuple)
        dtype_str: Data type string ("f16", "bf16", "f32")
        port: Port for distributed communication
        num_iters: Number of profiling iterations
        num_warmup: Number of warmup iterations
        skip_check: Whether to skip accuracy check
        allreduce_impl: Allreduce implementation ("flydsl" or "aiter")
        mode: "eager" or "cudagraph" - which path to run (separate flows)
        result_dict: Shared dictionary to collect results from all ranks
    """
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    # Normalize dtype string (accept fp16/fp32 as well as f16/f32)
    dtype_str = _normalize_dtype_arg(dtype_str)

    # Set dtype and tolerance
    if dtype_str == "f32":
        dtype = DTYPE_FP32
        atol = 1e-4
    elif dtype_str == "f16":
        dtype = DTYPE_FP16
        # NOTE: reference is fp32 sum; output is fp16, so error is dominated by fp16 quantization.
        # Scale by sqrt(world_size) since sum magnitude grows ~sqrt(ws) for random inputs.
        atol = 2e-3 * (float(world_size) ** 0.5) + 2e-3
    elif dtype_str == "bf16":
        dtype = DTYPE_BF16
        # bf16 has lower mantissa; scale tolerance similarly.
        atol = 2e-3 * float(world_size) + 2e-3
    else:
        raise ValueError(f"unsupported dtype_str: {dtype_str}")

    # Set backend - always use aiter for multi-GPU
    os.environ["FLYDSL_CUSTOM_ALL_REDUCE_BACKEND"] = "aiter"
    # Set aiter impl from parameter (can be "flydsl" or "aiter")
    os.environ["FLYDSL_AITER_IMPL"] = allreduce_impl

    torch.manual_seed(0)  # Same seed for all ranks
    x = torch.randn(shape, device=device, dtype=dtype).contiguous()
    x_flat = x.reshape(-1).contiguous()

    # Initialize allreduce
    handles = [torch.empty((1,), device="cpu", dtype=torch.uint8) for _ in range(world_size)]
    offsets = [0 for _ in range(world_size)]
    meta = torch.empty((meta_size(),), device=device, dtype=torch.int8)
    rank_data = x_flat
    out = torch.empty_like(x_flat)
    fa = init_custom_ar(meta, rank_data, handles, offsets, rank=rank, full_nvlink=False, out=out)
    # fa = init_custom_ar(meta, rank_data, handles, offsets, rank=rank, full_nvlink=False)

    if rank == 0:
        fa_mod = getattr(getattr(fa, "__class__", None), "__module__", None)
        fa_name = getattr(getattr(fa, "__class__", None), "__name__", None)
        print(
            f"[custom_all_reduce] backend=aiter "
            f"allreduce_impl={allreduce_impl!r} fa={fa_mod}.{fa_name}",
            flush=True,
        )

    # Warmup: align all ranks
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    def _run_eager():
        nonlocal out
        if allreduce_impl == "aiter":
            # AIter: custom_all_reduce returns result, no out param
            result = fa.custom_all_reduce(x_flat, open_fp8_quant=False)
            if result is not None:
                out.copy_(result)
        else:
            # FlyDSL: uses out parameter
            fa.custom_all_reduce(x_flat, out=out, open_fp8_quant=False)

    try:
        if mode == "eager":
            for _ in range(num_warmup):
                _run_eager()
            torch.cuda.synchronize()
            dist.barrier(device_ids=[rank])
            if not skip_check:
                out.fill_(0)
                torch.cuda.synchronize()
                dist.barrier(device_ids=[rank])
                _run_eager()
                torch.cuda.synchronize()
                dist.barrier(device_ids=[rank])
                gathered = [torch.empty_like(x_flat) for _ in range(world_size)]
                dist.all_gather(gathered, x_flat, group=group)
                ref_f32 = torch.zeros_like(x_flat, dtype=torch.float32)
                for t in gathered:
                    ref_f32 += t.to(torch.float32)
                max_err = (out.to(torch.float32) - ref_f32).abs().max().item()
                if max_err >= atol:
                    print(out[:10])
                    print(ref_f32[:10])
                assert max_err < atol, f"[rank={rank}] max_err={max_err:.3e} >= atol={atol}"
            else:
                max_err = 0.0

            torch.cuda.synchronize()
            dist.barrier(device_ids=[rank])
            with tpf.profile(
                activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
                profile_memory=False,
                with_stack=True,
                with_modules=True,
            ) as prof:
                for _ in range(num_iters):
                    _run_eager()
                torch.cuda.synchronize()
            prof.export_chrome_trace(f"profiler_trace_{rank}_{allreduce_impl}_eager.json")
            perf_df = get_trace_perf(prof, num_iters)
            avg_name = "[avg us/iter]"
            if avg_name in perf_df.index and "device_time_sum" in perf_df.columns:
                avg_time_us = perf_df.at[avg_name, "device_time_sum"]
            else:
                device_time_col = perf_df.get("device_time_sum", pd.Series())
                avg_time_us = device_time_col.sum() / num_iters if not device_time_col.empty else 0.0
            device_time_sum = perf_df["device_time_sum"].sum() if "device_time_sum" in perf_df.columns else 0.0
            kernel_name = "unknown"
            if not perf_df.empty and "name" in perf_df.columns and "device_time_sum" in perf_df.columns:
                top_kernel = perf_df.nlargest(1, "device_time_sum")
                if not top_kernel.empty:
                    kernel_name = top_kernel.iloc[0]["name"]
            result = {
                "rank": rank,
                "shape": shape,
                "dtype": dtype_str,
                "world_size": world_size,
                "mode": "eager",
                "max_error": max_err,
                "avg_time_us": avg_time_us,
                "device_time_sum_us": device_time_sum,
                "kernel_name": kernel_name,
                "num_iters": num_iters,
                "num_warmup": num_warmup,
            }
            result_dict[rank] = result

        elif mode == "cudagraph":
            if not hasattr(fa, "capture"):
                if rank == 0:
                    print("[test_flydsl_allreduce] WARN: fa has no capture(); skipping cudagraph.", flush=True)
                result_dict[rank] = {
                    "rank": rank, "shape": shape, "dtype": dtype_str, "world_size": world_size,
                    "mode": "cudagraph", "max_error": float("nan"), "avg_time_us": 0.0,
                    "device_time_sum_us": 0.0, "kernel_name": "skip", "num_iters": num_iters,
                    "num_warmup": num_warmup, "error": "no capture()",
                }
            else:
                for _ in range(num_warmup):
                    _run_eager()
                torch.cuda.synchronize()
                dist.barrier(device_ids=[rank])
                graph = torch.cuda.CUDAGraph()
                try:
                    # Capture using simplified interface - tensors recorded during capture
                    with fa.capture():
                        # Warmup inside capture context to populate exe cache
                        for _ in range(3):
                            fa.custom_all_reduce(x_flat, out=out, open_fp8_quant=False)
                        torch.cuda.synchronize()
                        with torch.cuda.graph(graph):
                            fa.custom_all_reduce(x_flat, out=out, open_fp8_quant=False)
                    # IPC handles exchanged at capture exit
                except Exception as cap_e:
                    if rank == 0:
                        print(f"[rank={rank}] WARN: cudagraph capture failed: {cap_e}", flush=True)
                    result_dict[rank] = {
                        "rank": rank, "shape": shape, "dtype": dtype_str, "world_size": world_size,
                        "mode": "cudagraph", "max_error": float("nan"), "avg_time_us": 0.0,
                        "device_time_sum_us": 0.0, "kernel_name": "skip", "num_iters": num_iters,
                        "num_warmup": num_warmup, "error": str(cap_e),
                    }
                else:
                    # Replay: write new data directly to x_flat, read result from out
                    # No copy needed - kernel reads from user tensor addresses directly
                    torch.manual_seed(42 + rank)
                    x_flat.copy_(torch.randn_like(x_flat, device=device, dtype=dtype))
                    out.fill_(0)
                    torch.cuda.synchronize()

                    dist.barrier(device_ids=[rank])
                    graph.replay()
                    torch.cuda.synchronize()
                    dist.barrier(device_ids=[rank])
                    
                    if not skip_check:
                        gathered = [torch.empty_like(x_flat) for _ in range(world_size)]
                        dist.all_gather(gathered, x_flat, group=group)
                        ref_f32 = torch.zeros_like(x_flat, dtype=torch.float32)
                        for t in gathered:
                            ref_f32 += t.to(torch.float32)
                        max_err = (out.to(torch.float32) - ref_f32).abs().max().item()
                        assert max_err < atol, f"[rank={rank}] cudagraph max_err={max_err:.3e} >= atol={atol}"
                    else:
                        max_err = 0.0
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()

                    start_evt.record()
                    for _ in range(num_iters):
                        graph.replay()
                    end_evt.record()
                    torch.cuda.synchronize()
                    total_ms = start_evt.elapsed_time(end_evt)
                    avg_time_us = (total_ms * 1000.0) / num_iters if num_iters else 0.0
                    device_time_sum = total_ms * 1000.0
                    result_dict[rank] = {
                        "rank": rank, "shape": shape, "dtype": dtype_str, "world_size": world_size,
                        "mode": "cudagraph", "max_error": max_err, "avg_time_us": avg_time_us,
                        "device_time_sum_us": device_time_sum, "kernel_name": "cudagraph_replay",
                        "num_iters": num_iters, "num_warmup": num_warmup,
                    }
        else:
            raise ValueError(f"unsupported mode={mode!r}")

    except Exception as e:
        print(f"[rank={rank}] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        result = {
            "rank": rank,
            "shape": shape,
            "dtype": dtype_str,
            "world_size": world_size,
            "mode": mode,
            "max_error": float('inf'),
            "avg_time_us": 0.0,
            "device_time_sum_us": 0.0,
            "kernel_name": "error",
            "num_iters": num_iters,
            "num_warmup": num_warmup,
            "error": str(e),
        }
        result_dict[rank] = result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ============================================================================
# Main test function
# ============================================================================

def run_all_tests(
    world_size: int = 8,
    num_iters: int = 101,
    num_warmup: int = 5,
    skip_check: bool = False,
    allreduce_impl: str = "flydsl",
    mode: str = "eager",
    configs: list = None,
):
    """Run all accuracy and performance tests, collect results and save to CSV.
    
    Args:
        world_size: Number of GPUs/processes
        num_iters: Number of profiling iterations
        num_warmup: Number of warmup iterations
        skip_check: Whether to skip accuracy check
        allreduce_impl: Allreduce implementation ("flydsl" or "aiter")
        mode: "eager" or "cudagraph" - which path to run (default: eager)
        configs: List of (shape, dtype_str) tuples. If None, uses default configs.
    """
    ng = torch.cuda.device_count()
    if ng < world_size:
        raise RuntimeError(f"need >= {world_size} GPUs, got {ng}")
    
    # Test configurations
    # Format: (shape_tuple, dtype_str)
    # Shape numel must be multiple of pack size: f16/bf16 -> 8 elems, f32 -> 4 elems
    if configs is None:
        default_configs = [
            ((128, 8192), "fp16"),    # 1048576 elements, bf16
        ]
    else:
        default_configs = configs
    
    all_results = []
    
    print("=" * 80)
    print("Starting FlyDSL AllReduce Multi-GPU Tests")
    print("=" * 80)
    print(f"World size: {world_size}")
    print(f"Backend: aiter")
    print(f"Allreduce impl: {allreduce_impl}")
    print(f"Mode: {mode}")
    print(f"Profile iterations: {num_iters}, Warmup: {num_warmup}")
    print(f"Skip check: {skip_check}")
    print("=" * 80)
    
    for shape, dtype_str in default_configs:
        # Normalize dtype string
        dtype_str = _normalize_dtype_arg(dtype_str)
        print(f"\nTesting: shape={shape}, dtype={dtype_str}")
        
        # Verify pack alignment
        numel = 1
        for d in shape:
            numel *= d
        if dtype_str in {"f16", "bf16"}:
            pack = 8
        else:
            pack = 4
        
        if numel % pack != 0:
            print(f"  WARNING: shape numel={numel} is not multiple of {pack} for dtype {dtype_str}, skipping")
            continue
        
        try:
            # Create shared dictionary for results
            manager = mp.Manager()
            result_dict = manager.dict()
            
            # Get free port
            port = _free_port()
            
            # Spawn processes
            mp.spawn(
                _dist_worker,
                args=(world_size, shape, dtype_str, port, num_iters, num_warmup, skip_check, allreduce_impl, mode, result_dict),
                nprocs=world_size,
                join=True,
            )
            
            # Collect results from all ranks
            rank_results = [result_dict[i] for i in range(world_size) if i in result_dict]
            
            # Sort by rank
            rank_results.sort(key=lambda x: x["rank"])
            
            if rank_results:
                print(f"  ✓ Test completed for all {len(rank_results)} ranks")
                
                # Calculate aggregate statistics
                max_errors = [r["max_error"] for r in rank_results]
                avg_times = [r["avg_time_us"] for r in rank_results]
                
                max_error = max(max_errors)
                mean_avg_time = np.mean(avg_times)
                max_avg_time = max(avg_times)
                min_avg_time = min(avg_times)
                
                print(f"    Max error: {max_error:.3e}")
                print(f"    Avg time: mean={mean_avg_time:.3f} us/iter, min={min_avg_time:.3f}, max={max_avg_time:.3f}")
                
                # Add aggregate row
                aggregate_result = {
                    "rank": "aggregate",
                    "shape": str(shape),
                    "dtype": dtype_str,
                    "world_size": world_size,
                    "mode": mode,
                    "max_error": max_error,
                    "avg_time_us": mean_avg_time,
                    "min_avg_time_us": min_avg_time,
                    "max_avg_time_us": max_avg_time,
                    "device_time_sum_us": sum(r["device_time_sum_us"] for r in rank_results),
                    "kernel_name": rank_results[0]["kernel_name"] if rank_results else "unknown",
                    "num_iters": num_iters,
                    "num_warmup": num_warmup,
                }
                all_results.append(aggregate_result)
                
                # Add individual rank results
                for r in rank_results:
                    r["shape"] = str(r["shape"])  # Convert tuple to string for CSV
                    all_results.append(r)
            
        except Exception as e:
            print(f"    ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Convert to DataFrame and save to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        csv_filename = "flydsl_allreduce_perf.csv"
        df.to_csv(csv_filename, index=False)
        print("\n" + "=" * 80)
        print(f"Results saved to: {csv_filename}")
        print("=" * 80)
        print("\nSummary (aggregate rows only):")
        aggregate_df = df[df["rank"] == "aggregate"]
        if not aggregate_df.empty:
            print(aggregate_df.to_string(index=False))
        print("=" * 80)
        return df
    else:
        print("\nNo results to save.")
        return pd.DataFrame()


if __name__ == "__main__":
    freeze_support()
    # Align with AIter harness: use spawn to avoid fork+CUDA issues.
    set_start_method("spawn", force=True)
    
    import argparse
    
    parser = argparse.ArgumentParser(description="FlyDSL allreduce multi-GPU test runner")
    parser.add_argument(
        "--world_size",
        type=int,
        default=8,
        help="Number of GPUs/processes (default: 8)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=101,
        help="Number of profiling iterations (default: 101)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default="",
        help="Test shapes in format 'shape1,dtype1;shape2,dtype2' (e.g., '128,8192,f16;64,16384,f32')",
    )
    parser.add_argument(
        "--skip_check",
        action="store_true",
        help="Skip accuracy check",
    )
    parser.add_argument(
        "--allreduce_impl",
        type=str,
        default="flydsl",
        choices=["flydsl", "aiter"],
        help="Allreduce implementation (default: flydsl)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="eager",
        choices=["eager", "cudagraph"],
        help="Test mode: eager or cudagraph (default: eager)",
    )
    args = parser.parse_args()
    
    # Parse shapes if provided
    configs = None
    if args.shapes:
        configs = []
        for part in args.shapes.split(";"):
            p = part.strip()
            if not p:
                continue
            # Format: "128,8192,f16" or "128,8192" (defaults to f16)
            parts = [x.strip() for x in p.split(",")]
            if len(parts) >= 2:
                shape_tuple = tuple(int(x) for x in parts[:-1])
                dtype = parts[-1] if len(parts) > 2 else "f16"
                # Normalize dtype string (accept fp16/fp32 as well as f16/f32)
                dtype = _normalize_dtype_arg(dtype)
                configs.append((shape_tuple, dtype))
        if not configs:
            configs = None
    
    run_all_tests(
        world_size=args.world_size,
        num_iters=args.iters,
        num_warmup=args.warmup,
        skip_check=args.skip_check,
        allreduce_impl=args.allreduce_impl,
        mode=args.mode,
        configs=configs,
    )
