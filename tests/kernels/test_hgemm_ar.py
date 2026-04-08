#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import os
import sys
import logging
import flydsl.compiler as flyc

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
import pytest
import pandas as pd

from dataclasses import dataclass
from torch.profiler import profile, ProfilerActivity
from kernels.custom_all_reduce import init_custom_ar


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLYDSL_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLYDSL_SRC not in sys.path:
    sys.path.insert(0, _PYFLYDSL_SRC)

from kernels.hgemm_ar import hgemm_ar_, hgemm_shuffle_b, GEMMARBackend
from tests.test_common import run_perftest, verify_output
from flydsl.runtime.device import get_rocm_arch

logging.basicConfig(level=logging.INFO)
ARCH = str(get_rocm_arch())

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

try:
    import aiter
    HAS_AITER = True
except Exception:
    HAS_AITER = False


DEFAULT_BENCH_ITERS = 50
DEFAULT_BENCH_WARMUP = 3


@dataclass
class Args:
    dtype: torch.dtype
    m: int
    n: int
    k: int
    tile_m: int
    tile_n: int
    tile_k: int
    split_k: int
    num_devices: int
    parts: int
    nsamples: int


def init_world(device_id, num_devices, parts, port=24317):
    torch.cuda.set_device(device_id)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=device_id,
        world_size=num_devices,
        device_id=device_id,
    )
    group_size = num_devices // parts
    group_id = device_id // group_size
    group_ranks = list(range(group_id * group_size, (group_id + 1) * group_size))
    group = dist.new_group(ranks=group_ranks)
    print(f"[init_world] device_id:{device_id}, group_ranks:{group_ranks}", flush=True)
    return group


def create_inputs(args):
    group_size = args.num_devices // args.parts
    inputs = []
    for part in range(args.parts):
        for rank in range(group_size):
            device_id = part * group_size + rank
            for i in range(args.nsamples):
                a = torch.empty((args.m, args.k), dtype=args.dtype, device=f'cuda:{device_id}')
                a.uniform_(-1, 1)
                b = torch.empty((args.n, args.k), dtype=args.dtype, device=f'cuda:{device_id}')
                b.uniform_(-1, 1)
                inputs.append([a, b])
    return inputs


def create_outputs(args):
    group_size = args.num_devices // args.parts
    outputs = []
    for part in range(args.parts):
        for rank in range(group_size):
            device_id = part * group_size + rank
            for i in range(args.nsamples):
                c = torch.randn((args.m, args.n), dtype=args.dtype, device=f"cuda:{device_id}")
                outputs.append(c)
    return outputs


def ref_worker(device_id, num_devices, parts, nsamples, inputs, outputs):
    group = init_world(device_id, num_devices, parts)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    warmup_iter = 4
    for i in range(nsamples):
        if i == warmup_iter:
            start_event.record()
        input = inputs[device_id * nsamples + i]
        output = outputs[device_id * nsamples + i]
        F.linear(input[0], input[1], out=output)
        dist.all_reduce(output, group=group)
    torch.cuda.synchronize()
    end_event.record()
    dist.barrier(group=group)
    dist.destroy_process_group()
    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / (nsamples - warmup_iter)
    print(f"device_id:{device_id}, avg_ms:{avg_ms}")


def ref_func(args, inputs, outputs):
    mp.spawn(
        ref_worker,
        args=(args.num_devices, args.parts, args.nsamples, inputs, outputs),
        nprocs=args.num_devices,
        join=True
    )


def worker(device_id, num_devices, parts, nsamples, inputs, outputs, kwargs):
    group = init_world(device_id, num_devices, parts)
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    meta = torch.empty((0,), device=device_id, dtype=torch.int8)
    rank_data = inputs[device_id * nsamples]
    handles = [torch.empty((1,), device="cpu", dtype=torch.uint8) for _ in range(world_size)]
    offsets = [0 for _ in range(world_size)]
    fa = init_custom_ar(meta, rank_data, handles, offsets, rank=rank, backend=GEMMARBackend)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    warmup_iter = 4
    for i in range(nsamples):
        if i == warmup_iter:
            start_event.record()
        input = inputs[device_id * nsamples + i]
        output = outputs[device_id * nsamples + i]
        fa.hgemm_ar_fusion(input[0], input[1], output, kwargs)
    torch.cuda.synchronize()
    end_event.record()
    dist.barrier(group=group)
    dist.destroy_process_group()
    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / (nsamples - warmup_iter)
    print(f"device_id:{device_id}, avg_ms:{avg_ms}")


def func(args, inputs, outputs, kwargs):
    mp.spawn(
        worker,
        args=(args.num_devices, args.parts, args.nsamples, inputs, outputs, kwargs),
        nprocs=args.num_devices,
        join=True
    )


# @pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("dtype", ["bf16"])
@pytest.mark.parametrize(
    "m, n, k, TILE_M, TILE_N, TILE_K, SPLIT_K, world_size",
    [
        (32, 7168, 2048, 32, 128, 128, 1, 4),
        # (32, 384, 7168, 32, 64, 64, 8, 4),
        # (4, 384, 7168, 32, 64, 64, 8, 8),
        # (65, 1024, 8192, 64, 64, 128, 2, 2),
    ]
)
# @pytest.mark.parametrize("test_graph", [
#     pytest.param(False, id="eager"),
#     pytest.param(True, id="graph"),
# ])
@pytest.mark.parametrize("test_graph", [
    pytest.param(False, id="eager"),
])
def test_mfma_flyc_hgemm_ar(
    dtype,
    m, n, k,
    TILE_M, TILE_N, TILE_K, SPLIT_K, world_size,
    *,
    test_graph,
    bench_iters: int = DEFAULT_BENCH_ITERS,
    bench_warmup: int = DEFAULT_BENCH_WARMUP,
):
    global ARCH
    if not (ARCH in ["gfx950", "gfx942"]):
        pytest.skip(f"Skip hgemm test: ARCH={ARCH}")
    
    print("=" * 80)
    print(
        f"[flyc] MFMA {dtype.upper()} HGEMM+Allreduce Test"
    )
    print("=" * 80)

    bench_iters = max(2, int(bench_iters))
    bench_warmup = int(bench_warmup)

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    args = Args(
        dtype=torch_dtype,
        m=m,
        n=n,
        k=k,
        tile_m=TILE_M,
        tile_n=TILE_N,
        tile_k=TILE_K,
        split_k=SPLIT_K,
        num_devices=world_size,
        parts=1,
        nsamples=50,
    )
    kwargs = {
        'TILE_M': args.tile_m,
        'TILE_N': args.tile_n,
        'TILE_K': args.tile_k,
        'SPLIT_K': args.split_k,
    }

    inputs = create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    func(args, inputs, outputs, kwargs)
    ref_func(args, inputs, ref_outputs)
    max_diff_global = float(-1)
    for output, ref_output in zip(outputs, ref_outputs):
        is_allclose = torch.allclose(output, ref_output)
        # assert is_allclose == True
        maxdiff_out = (output - ref_output).abs().max().item()
        max_diff_global = max(max_diff_global, maxdiff_out)
    print(f"max_diff_global:{max_diff_global}")
    assert max_diff_global < 1e-2 * args.k * args.num_devices

    # get ref_func perf
    print("===================== [REF] =====================")
    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        profile_memory=False,
        with_stack=True,
        with_modules=True
    ) as prof:
        ref_func(args, inputs, ref_outputs)
        torch.cuda.synchronize()
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
    print(table)

    # get func perf
    print("===================== [FLYDSL] =====================")
    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        profile_memory=False,
        with_stack=True,
        with_modules=True
    ) as prof:
        func(args, inputs, outputs, kwargs)
        torch.cuda.synchronize()
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
    print(table)
