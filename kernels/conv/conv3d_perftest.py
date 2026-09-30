# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter ``run_perftest`` timing for the conv3d tuner.

Kept in step with ``aiter/test_common.py`` (``perftest``, ``get_trace_perf``,
``post_process_data``, ``device_memory_profiling``) so tuned ``us`` means the
same thing in both repos: device time of every kernel one call launches,
averaged over rotated argument copies, with warm-up and IQR outliers dropped.
Graph capture, trace export and SMI monitoring are left out.
"""

import copy

import pandas as pd
import torch
import torch.profiler as tpf


def device_memory_profiling(func, *args, **kwargs):
    gpu_id = torch.cuda.current_device()
    inputSize = sum([el.nbytes for el in args if isinstance(el, torch.Tensor) and el.device.index == gpu_id]) + 1
    torch.cuda.reset_peak_memory_stats(gpu_id)
    cuda_memory_before = torch.cuda.mem_get_info(gpu_id)[1] - torch.cuda.mem_get_info(gpu_id)[0]
    torch_memory_before = torch.cuda.memory_reserved(gpu_id)
    torch_peak_before = torch.cuda.memory_stats(gpu_id).get("allocated_bytes.all.peak", 0)
    non_torch_memory_before = cuda_memory_before - torch_memory_before

    _ = func(*args, **kwargs)

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
    for i in range(num_iters):
        args, kwargs = rotate_args[i % num_rotate_args]
        data = func(*args, **kwargs)
    return data


def post_process_data(df, num_iters, warm_iter=1):
    """Drop warm-up iterations and IQR outliers; return (dropped indices, dropped count)."""
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

    sum_df = grouped_kernel_df.iloc[warm_iter:].reset_index(drop=True)
    out_range_idx = []
    if num_iters > 30:
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
    indices.update(idx for sublist in index_sublists for idx in sublist)
    indices.update(dropped_indexs)
    return list(indices), out_range_num + warm_iter + num_iters - act_iters


def get_trace_perf(prof, num_iters):
    assert num_iters > 1
    warm_iter = 1
    num_iters -= warm_iter
    cols = ["name", "self_cpu_time_total", "self_device_time_total", "device_type", "device_index"]
    df = pd.DataFrame([[getattr(el, x, None) for x in cols] for el in prof.events()], columns=cols)
    dropped_indexs, dropped_num = post_process_data(df, num_iters + warm_iter, warm_iter)
    df = df.drop(dropped_indexs)
    df["cnt"] = 1
    rets = []
    for name, d in df.groupby("name", sort=False):
        kernel_num_per_iter = 0
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
    if df.empty:
        return 0.0
    df = df[(df.host_time_sum > 0) | (df.device_time_sum > 0)]
    actual_iters = num_iters + warm_iter - dropped_num
    return float(df["device_time_sum"].sum() / actual_iters)


def run_perftest(func, *args, num_iters=101, num_warmup=2, num_rotate_args=0, **kwargs):
    """Return ``(last output, us per call)`` exactly as aiter's ``run_perftest`` does."""
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
    with tpf.profile(
        activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
        profile_memory=False,
        with_stack=False,
        with_modules=True,
    ) as prof:
        data = run_iters_rotate(num_iters, func, rotate_args)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return data, get_trace_perf(prof, num_iters)
