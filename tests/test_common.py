# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# from https://github.com/ROCm/aiter/blob/main/aiter/test_common.py
import copy
import logging
import os

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger("flydsl")

pd.set_option("display.max_rows", 200)
## debug ##
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)
# pd.set_option("display.width", None)
# pd.set_option("display.max_colwidth", None)
# pd.set_option("display.expand_frame_repr", False)


# Distribution (median + p95, microseconds) of the most recent perftest call,
# populated only when FLYDSL_PERF_DIST is set.  Lets callers report a true
# timed-loop median+p95 over num_iters without changing the (data, avg) return
# signature shared by every other caller.
LAST_PERF_DIST = {"median": None, "p95": None, "n_rotate": None}


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    idx = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def _timed_distribution(func, rotate_args, num_iters, time_call):
    """Run ``func`` for ``num_iters``, CYCLING through ``rotate_args`` (the
    cache-sized argument copies = L2-flush behavior), timing each call with
    ``time_call(func, args, kwargs) -> microseconds``.

    Returns ``(data, median_us, p95_us, n_rotate)``.  Pure/host-testable: the GPU
    event timing is injected via ``time_call`` so the rotation contract (iteration
    i uses ``rotate_args[i % n]``) can be unit-tested without a device.
    """
    n_rot = len(rotate_args)
    latencies = []
    data = None
    for i in range(num_iters):
        a_i, kw_i = rotate_args[i % n_rot]
        us, data = time_call(func, a_i, kw_i)
        latencies.append(us)
    ordered = sorted(latencies)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    return data, median, _percentile(ordered, 0.95), n_rot


def perftest(num_iters=20, num_warmup=3, testGraph=False, num_rotate_args=0, needTrace=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # ROCm torch.profiler (ROCTracer) is not always stable when invoked repeatedly
            # under pytest (multiple tests, repeated init/teardown). For unit tests, the
            # profiler is not required; fall back to simple timing.
            #
            num = num_rotate_args
            if num < 1:
                gpu_id = torch.cuda.current_device()
                iter_used_memory, inputSize, _, _ = device_memory_profiling(func, *args, **kwargs)

                properties = torch.cuda.get_device_properties(gpu_id)
                free_memory = torch.cuda.mem_get_info(gpu_id)[0]
                cache_size = min(
                    getattr(properties, "L2_cache_size", 4096 * 1024) * 64 * 128,
                    (free_memory - iter_used_memory + inputSize) * 0.9,
                )
                cache_size = max(cache_size, 0)
                num = int((cache_size + inputSize - 1) // inputSize)
            num = min(num, num_iters)

            rotate_args = [(copy.deepcopy(args), copy.deepcopy(kwargs)) for _ in range(num - 1)] + [(args, kwargs)]
            run_iters(num_warmup, func, *args, **kwargs)
            torch.cuda.synchronize()

            # True per-iteration timed-loop distribution (median + p95) over
            # num_iters, recorded in LAST_PERF_DIST.  Opt-in via FLYDSL_PERF_DIST so
            # the default profiler/event path is unchanged.  Returns the MEDIAN as
            # the central-tendency `avg` so the reported us is the median.
            #
            # Cycles through the SAME ``rotate_args`` set the default path uses
            # (``num`` cache-sized argument copies), so each iteration touches a
            # different working set -- this is the L2-flush behavior the recorded
            # protocol claims (l2_flush_per_iter=True), not a hot-cache reuse of one
            # tensor set.  LAST_PERF_DIST["n_rotate"] records how many copies cycled.
            if int(os.environ.get("FLYDSL_PERF_DIST", 0)):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                def _time_call(fn, a_i, kw_i):
                    start_event.record()
                    out = fn(*a_i, **kw_i)
                    end_event.record()
                    end_event.synchronize()
                    return start_event.elapsed_time(end_event) * 1000.0, out  # ms -> us

                data, median, p95, n_rot = _timed_distribution(func, rotate_args, num_iters, _time_call)
                torch.cuda.synchronize()
                LAST_PERF_DIST["median"] = median
                LAST_PERF_DIST["p95"] = p95
                LAST_PERF_DIST["n_rotate"] = n_rot
                logger.info(f"perf_dist: median={median:.3f} us p95={p95:.3f} us over {num_iters} iters")
                return data, median

            if int(os.environ.get("FLYDSL_LOG_MORE", 0)):
                latencies = []
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                for _ in range(num_iters):
                    start_event.record()
                    data = func(*args, **kwargs)
                    end_event.record()
                    end_event.synchronize()
                    latencies.append(start_event.elapsed_time(end_event))
                avg = np.mean(latencies) * 1000
                logger.info(f"avg: {avg} us/iter from cuda.Event")
            if int(os.environ.get("FLYDSL_PERFTEST_USE_EVENTS", 0)):
                # HIP-event timing, avoids nesting torch.profiler under an external rocprofv3 session.
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                data = run_iters_rotate(num_iters, func, rotate_args)
                end_event.record()
                end_event.synchronize()
                torch.cuda.empty_cache()
                avg = start_event.elapsed_time(end_event) / num_iters * 1000
            else:
                import torch.profiler as tpf

                with tpf.profile(
                    activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
                    profile_memory=False,
                    with_stack=False,
                    with_modules=True,
                ) as prof:
                    data = run_iters_rotate(num_iters, func, rotate_args)
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                avg = get_trace_perf(prof, num_iters)

            if testGraph:
                import torch.profiler as tpf

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    data = run_iters_rotate(num_iters, func, rotate_args)
                with tpf.profile(
                    activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
                    profile_memory=True,
                    with_stack=True,
                    with_modules=True,
                ) as prof:
                    run_iters(1, graph.replay)
                avg = get_trace_perf(prof, num_iters)
                logger.info(f"avg: {avg} us/iter with hipgraph")

            return data, avg

        return wrapper

    return decorator


def benchmark():
    def decorator(func):
        def wrapper(*args, **kwargs):
            callargs = log_args(func, *args, **kwargs)
            ret = func(*args, **kwargs)
            if ret is not None:
                callargs.update(ret)
            return callargs

        return wrapper

    return decorator


def device_memory_profiling(func, *args, **kwargs):
    gpu_id = torch.cuda.current_device()
    inputSize = sum([el.nbytes for el in args if isinstance(el, torch.Tensor) and el.device.index == gpu_id]) + 1
    torch.cuda.reset_peak_memory_stats(gpu_id)
    cuda_memory_before = torch.cuda.mem_get_info(gpu_id)[1] - torch.cuda.mem_get_info(gpu_id)[0]
    torch_memory_before = torch.cuda.memory_reserved(gpu_id)
    torch_peak_before = torch.cuda.memory_stats(gpu_id).get("allocated_bytes.all.peak", 0)
    non_torch_memory_before = cuda_memory_before - torch_memory_before

    _data = func(*args, **kwargs)

    torch.cuda.reset_peak_memory_stats(gpu_id)
    cuda_memory_after = torch.cuda.mem_get_info(gpu_id)[1] - torch.cuda.mem_get_info(gpu_id)[0]
    torch_memory_after = torch.cuda.memory_reserved(gpu_id)
    torch_peak_after = torch.cuda.memory_stats(gpu_id).get("allocated_bytes.all.peak", 0)
    non_torch_memory_after = cuda_memory_after - torch_memory_after

    torch_peak_increase = torch_peak_after - torch_peak_before
    non_torch_increase = non_torch_memory_after - non_torch_memory_before
    iter_used_memory = torch_peak_increase + non_torch_increase + inputSize

    return iter_used_memory, inputSize, torch_peak_increase, non_torch_increase


def run_iters(num_iters, func, *args, **kwargs):
    data = None
    for _ in range(num_iters):
        data = func(*args, **kwargs)
    return data


def run_iters_rotate(num_iters, func, rotate_args):
    data = None
    num_rotate_args = len(rotate_args)
    for _ in range(num_iters):
        args, kwargs = rotate_args[_ % num_rotate_args]
        data = func(*args, **kwargs)

    return data


def run_perftest(
    func,
    *args,
    num_iters=20,
    num_warmup=3,
    testGraph=False,
    num_rotate_args=0,
    needTrace=False,
    **kwargs,
):

    @perftest(
        num_iters=num_iters,
        num_warmup=num_warmup,
        testGraph=testGraph,
        num_rotate_args=num_rotate_args,
        needTrace=needTrace,
    )
    def worker(*args, **kwargs):
        return func(*args, **kwargs)

    return worker(*args, **kwargs)


def log_args(func, *args, **kwargs):
    import inspect

    callargs = inspect.getcallargs(func, *args, **kwargs)

    prefix = f"calling {func.__name__}("
    blanks = " " * (len(prefix))

    def getTensorInfo(el):
        if isinstance(el, torch.Tensor):
            return f"{el.shape} {el.dtype} {el.device} {hex(el.data_ptr())}"
        elif isinstance(el, tuple):
            viewNum = 5
            if len(el) > viewNum:
                el = list(el[:viewNum]) + ["..."]
            return f'\n{" "*(len(prefix)+31)}'.join(["("] + [f" {getTensorInfo(e)}" for e in el] + [")"])
        return el

    info = [f"{el:<28} = {getTensorInfo(callargs[el])}" for el in callargs]
    info = f",\n{blanks}".join(info)
    logger.info(f"\n{prefix}{info})")
    return callargs


def post_process_data(df, num_iters, warm_iter=1):
    """remove abnormal data"""

    device_df = df[df["device_type"].astype(str).str.contains("DeviceType.CUDA")]
    # print("devicedf is ", device_df)
    if device_df.empty:
        return [], 0
    kernels_num = int(len(device_df) / num_iters)

    act_iters = num_iters
    valid_n = len(device_df)
    dropped_indexs = []
    if len(device_df) % num_iters == 0:
        kernels_num = int(len(device_df) / num_iters)
    else:
        ##get correct kernel num
        name_list = device_df["name"].tolist()
        max_kernel_num = 20
        n = len(name_list)
        for step in range(1, min(max_kernel_num, n // 2 + 1)):
            sub_list = [name_list[i] for i in range(step)]
            m = len(sub_list)

            valid_n = int(n / m) * m
            pattern_match = all(name_list[i] == sub_list[i % m] for i in range(int(n / m) * m))
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

    # rm warm iters
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
            (sum_df["self_device_time_total"] < lower) | (sum_df["self_device_time_total"] > upper)
        ].tolist()
    out_range_num = len(out_range_idx)

    indices = {idx for i in out_range_idx for idx in sum_df.iloc[i]["index"]}

    index_sublists = grouped_kernel_df["index"].head(warm_iter).tolist()
    indices_to_add = [idx for sublist in index_sublists for idx in sublist]
    indices.update(indices_to_add)
    indices.update(dropped_indexs)
    if int(os.environ.get("FLYDSL_LOG_MORE", 0)):
        logger.info(f"abnormal data indices: {indices}")
        for i in indices:
            logger.info(f"abnormal data: {df.iloc[i]['self_device_time_total']}")
    return list(indices), out_range_num + warm_iter + num_iters - act_iters


def get_trace_perf(prof, num_iters):
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
    ###remove abnormal data
    dropped_num = warm_iter
    dropped_indexs, dropped_num = post_process_data(df, num_iters + warm_iter, warm_iter)
    df = df.drop(dropped_indexs)
    iter_init = 0  # warm_iter dropped
    df["cnt"] = 1
    rets = []

    for name, d in df.groupby("name", sort=False):
        kernel_num_per_iter = iter_init
        if str(d["device_type"].iat[0]).split(".")[-1] != "CUDA":
            kernel_num_per_iter = 1
        r = d.iloc[kernel_num_per_iter:][["cnt", "self_cpu_time_total", "self_device_time_total"]].sum()
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
        rets.append(r)
    df = pd.DataFrame(rets)
    cols = [
        "name",
        "cnt",
        "host_time_sum",
        "device_time_sum",
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
        logger.info("no valida data after post process!")

    avg_name = "[avg us/iter]"
    for el in timerList:
        if el == "host_time_sum":
            df.at[avg_name, el] = df[el].sum() / num_iters
        else:
            df.at[avg_name, el] = df[el].sum() / actual_iters
    if int(os.environ.get("FLYDSL_LOG_MORE", 0)):
        pd.set_option("display.expand_frame_repr", False)
        pd.set_option("display.max_colwidth", 90)
        pd.set_option("display.float_format", "{:,.1f}".format)
        logger.info(f"{df}")
    return df.at[avg_name, "device_time_sum"]


def checkAllclose(a, b, rtol=1e-2, atol=1e-2, tol_err_ratio=0.05, msg="", printNum=8, printLog=True):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)

    if isClose.all():
        if printLog:
            logger.info(f"{msg}[checkAllclose {atol=} {rtol=} \033[32mpassed~\033[0m]")
        return 0
    else:
        try:
            mask = ~isClose
            num = mask.sum()
            printNum = min(printNum, num)
            percent = (num / a.numel()).item()
            if not printLog:
                return percent
            a_msked = a[mask]
            b_msked = b[mask]
            delta = (a_msked - b_msked).abs()
        except RuntimeError:
            mask = ~isClose.to("cpu")
            num = mask.sum()
            printNum = min(printNum, num)
            percent = (num / a.numel()).item()
            if not printLog:
                return percent
            a_msked = a[mask]
            b_msked = b[mask]
            delta = (a_msked - b_msked).abs()
        if percent > tol_err_ratio:
            logger.info(f"""{msg}[checkAllclose {atol=} {rtol=} \033[31mfailed!\033[0m]
    a    : {a.shape}
           {a_msked[:printNum]}
    b    : {b.shape}
           {b_msked[:printNum]}
    delta:
           {delta[:printNum]}""")
        else:
            logger.info(
                f"""{msg}[checkAllclose {atol=} {rtol=} \033[33mwarning!\033[0m] a and b results are not all close"""
            )
        logger.info(f"-->max abs delta:{delta.max()}, delta details: {percent:.1%} ({num} of {a.numel()}) elements")
        return percent


def verify_output(c_out, c_ref, atol=1e-2, rtol=1e-2, msg="", logits_diff_threshold=2e-3):
    if checkAllclose(c_out, c_ref, rtol=rtol, atol=atol) < 0.05:
        return True

    # Calculate various error metrics
    abs_diff = (c_out - c_ref).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    def calc_diff(x: torch.Tensor, y: torch.Tensor):
        x, y = x.double(), y.double()
        denominator = (x * x + y * y).sum()
        if denominator == 0:
            return 0.0
        numerator = 2 * (x * y).sum()
        sim = numerator / denominator
        diff = (1 - sim).item()
        return diff if not torch.isnan(torch.tensor(diff)) else 1.0  # NaN means mismatch

    logits_diff = calc_diff(c_out, c_ref)
    print(f"Logits Diff: {logits_diff:.6f}, Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}")
    if logits_diff > logits_diff_threshold:
        print(f"✗ Check failed: logits_diff {logits_diff} > {logits_diff_threshold}")
        logging.error(f"logits_diff: {logits_diff} is too large (threshold: {logits_diff_threshold})")
        return False
    print(f"{msg} ✓ Check passed")
    return True


def tensor_dump(x: torch.tensor, name: str, dir="./"):
    x_cpu = x.cpu().view(torch.uint8)
    filename = f"{dir}/{name}.bin"
    x_cpu.numpy().tofile(filename)
    logger.info(f"saving {filename} {x.shape}, {x.dtype}")

    with open(f"{dir}/{name}.meta", "w") as f:
        f.writelines([f"{el}\n" for el in [x.shape, x.dtype]])


def tensor_load(filename: str):
    DWs = np.fromfile(filename, dtype=np.uint32)
    metafile = ".".join(filename.split(".")[:-1]) + ".meta"
    shape, dtype = [eval(line.strip()) for line in open(metafile)]
    return torch.tensor(DWs).view(dtype).view(shape)
