#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Correctness and performance coverage for the gfx950 A16W16 universe GEMM."""

import itertools
import os
from dataclasses import dataclass

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from flydsl.runtime.device import get_rocm_arch
from kernels.gemm.gemm_a16w16_universe_gfx950 import (
    hgemm,
    make_hgemm_gfx950_param,
)
from tests.test_common import run_perftest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)
if not str(get_rocm_arch()).startswith("gfx950"):
    pytest.skip("A16W16 universe GEMM requires gfx950.", allow_module_level=True)

ROTARY_INPUTS_TARGET_BYTES = 8 * 1024**3


@dataclass
class _TestArgs:
    dtype: torch.dtype
    m: int
    n: int
    k: int
    block_m: int
    block_n: int
    block_k: int
    stages: int
    m_waves: int
    n_waves: int
    k_waves: int
    group_m: int
    has_bias: bool
    use_half_tile_interleaved: bool = False
    layout: str = "nt"
    split_k: int = 1
    out_dtype: torch.dtype | None = None


def empty_layout_matrix(rows: int, cols: int, dtype: torch.dtype, is_t: bool):
    if is_t:
        return torch.empty((cols, rows), dtype=dtype, device="cuda").t()
    return torch.empty((rows, cols), dtype=dtype, device="cuda")


def create_inputs(args: _TestArgs):
    a = empty_layout_matrix(
        args.m,
        args.k,
        args.dtype,
        args.layout[0] == "t",
    )
    a.uniform_(-1, 1)
    b = empty_layout_matrix(
        args.k,
        args.n,
        args.dtype,
        args.layout[1] == "t",
    )
    b.uniform_(-1, 1)
    if args.has_bias:
        bias = torch.empty((args.n,), dtype=args.dtype, device="cuda")
        bias.uniform_(10, 20)
    else:
        bias = None
    return (a, b, bias)


def create_outputs(args: _TestArgs):
    dtype = args.dtype if args.out_dtype is None else args.out_dtype
    c = torch.randn((args.m, args.n), dtype=dtype, device="cuda")
    return (c,)


def ref_func(*args):
    a, b, bias, c, _layout = args
    if c.dtype == a.dtype:
        if bias is None:
            torch.mm(a, b, out=c)
        else:
            torch.addmm(bias, a, b, out=c)
    else:
        if bias is None:
            ref = torch.mm(a.float(), b.float())
        else:
            ref = torch.addmm(bias.float(), a.float(), b.float())
        c.copy_(ref)


def make_triton_maxautotune_func():
    import torch._inductor.config as inductor_config

    inductor_config.max_autotune_gemm_backends = "TRITON"
    inductor_config.max_autotune_gemm_search_space = "DEFAULT"

    torch._dynamo.reset()

    def triton_maxautotune_func(a, b, bias, c):
        out = torch.mm(a, b)
        if bias is not None:
            out = out + bias
        c.copy_(out)

    return torch.compile(triton_maxautotune_func, mode="max-autotune", fullgraph=True)


def func(*args):
    a, b, bias, c, kwargs, layout = args
    hgemm(a, b, c, bias=bias, user_kwargs=kwargs, layout=layout)


def tensor_nbytes(tensors: torch.Tensor):
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def get_rotary_inputs(sample_inputs: torch.Tensor, sample_outputs: torch.Tensor):
    global ROTARY_INPUTS_TARGET_BYTES
    slot_bytes = 2 * (tensor_nbytes(sample_inputs) + tensor_nbytes(sample_outputs))
    rotary_inputs = ROTARY_INPUTS_TARGET_BYTES // slot_bytes
    return max(1, int(rotary_inputs))


def check_acc(args: _TestArgs):
    kwargs = {
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "stages": args.stages,
        "m_waves": args.m_waves,
        "n_waves": args.n_waves,
        "k_waves": args.k_waves,
        "group_m": args.group_m,
        "use_half_tile_interleaved": args.use_half_tile_interleaved,
        "split_k": args.split_k,
    }
    inputs = create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    inouts = inputs + outputs
    ref_inouts = inputs + ref_outputs
    maxdiff_out_ = []

    def get_tol(args):
        k_scale = (args.k / 8192) ** 0.5
        k_scale *= args.split_k * args.k_waves
        if args.dtype is torch.bfloat16:
            return 2e-1 * k_scale, 2e-1
        return 5e-2 * k_scale, 5e-2

    atol, rtol = get_tol(args)
    for _ in range(5):
        func(*(inouts + (kwargs, args.layout)))
        ref_func(*(ref_inouts + (args.layout,)))
        for output, ref_output in zip(outputs, ref_outputs):
            maxdiff_out = (output - ref_output).abs().max().item()
            maxdiff_out_.append(maxdiff_out)
            print(maxdiff_out, flush=True)
            torch.testing.assert_close(
                output,
                ref_output,
                atol=atol,
                rtol=rtol,
                check_dtype=True,
            )
    print(f"\n{args}\nmaxdiff_out:{maxdiff_out_}")


def benchmark(args: _TestArgs, warmup: int = 500, niters: int = 600):
    kwargs = {
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "stages": args.stages,
        "m_waves": args.m_waves,
        "n_waves": args.n_waves,
        "k_waves": args.k_waves,
        "group_m": args.group_m,
        "use_half_tile_interleaved": args.use_half_tile_interleaved,
        "split_k": args.split_k,
    }
    sample_inputs = create_inputs(args)
    sample_outputs = create_outputs(args)
    rotary_inputs = get_rotary_inputs(sample_inputs, sample_outputs)
    inputs = [sample_inputs] + [create_inputs(args) for _ in range(rotary_inputs - 1)]
    ref_inputs = [create_inputs(args) for _ in range(rotary_inputs)]
    outputs = [sample_outputs] + [
        create_outputs(args) for _ in range(rotary_inputs - 1)
    ]
    ref_outputs = [create_outputs(args) for _ in range(rotary_inputs)]
    triton_maxautotune_func = make_triton_maxautotune_func()
    global ROTARY_INPUTS_TARGET_BYTES
    print(
        f"rotary_inputs:{rotary_inputs}, target_bytes:{ROTARY_INPUTS_TARGET_BYTES}, "
        f"warmup:{warmup}, niters:{niters}"
    )

    def run_ref(idx):
        ref_func(*(ref_inputs[idx] + ref_outputs[idx] + (args.layout,)))

    def run_triton_maxautotune(idx):
        triton_maxautotune_func(*(ref_inputs[idx] + ref_outputs[idx]))

    def run_flydsl(idx):
        func(*(inputs[idx] + outputs[idx] + (kwargs, args.layout)))

    print("===================== [INTERLEAVED] =====================")
    for i in range(warmup):
        idx = i % rotary_inputs
        if i % 3 == 0:
            run_ref(idx)
            run_triton_maxautotune(idx)
            run_flydsl(idx)
        if i % 3 == 1:
            run_flydsl(idx)
            run_ref(idx)
            run_triton_maxautotune(idx)
        elif i % 3 == 2:
            run_triton_maxautotune(idx)
            run_flydsl(idx)
            run_ref(idx)
        torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CUDA],
    ) as prof:
        for i in range(warmup, niters):
            idx = i % rotary_inputs
            if i % 3 == 0:
                run_ref(idx)
                run_triton_maxautotune(idx)
                run_flydsl(idx)
            if i % 3 == 1:
                run_flydsl(idx)
                run_ref(idx)
                run_triton_maxautotune(idx)
            elif i % 3 == 2:
                run_triton_maxautotune(idx)
                run_flydsl(idx)
                run_ref(idx)
            torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))


def benchmark_flydsl(args: _TestArgs, warmup: int, niters: int):
    check_acc(args)
    inputs = create_inputs(args)
    outputs = create_outputs(args)
    kwargs = {
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "stages": args.stages,
        "m_waves": args.m_waves,
        "n_waves": args.n_waves,
        "k_waves": args.k_waves,
        "group_m": args.group_m,
        "use_half_tile_interleaved": args.use_half_tile_interleaved,
        "split_k": args.split_k,
    }
    _, us = run_perftest(
        func,
        *(inputs + outputs + (kwargs, args.layout)),
        num_iters=niters,
        num_warmup=warmup,
    )
    element_size = torch.empty((), dtype=args.dtype).element_size()
    bytes_moved = (
        args.m * args.k + args.k * args.n + args.m * args.n
    ) * element_size
    tflops = 2.0 * args.m * args.n * args.k / us * 1e-6
    tbps = bytes_moved / us * 1e-6
    print(
        f"Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, "
        f"BW: {tbps:.3f} TB/s"
    )
    return us


@pytest.mark.large_shape
@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize(
    "dtype",
    [
        "bf16",
        "fp16",
    ],
)
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias, is_hti",
    [
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 0, False, False),
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 4, True, False),
        (8160, 8160, 8192, 256, 256, 64, 2, 2, 4, 0, False, False),
        (8160, 8160, 8192, 256, 256, 64, 2, 2, 4, 0, True, False),
        (8192, 8192, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, False, False),
        (8192, 8192, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, True, False),
        (8192, 8192, 8192 + 32, 256, 256, 64, 2, 2, 4, 0, False, False),
        (8192, 8192, 8192 + 32, 256, 256, 64, 2, 2, 4, 0, True, False),
        (8160, 8160, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, False, False),
        (8160, 8160, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, True, False),
        (8160, 8160, 8192 + 32, 256, 256, 64, 2, 2, 4, 0, False, False),
        (8160, 8160, 8192 + 32, 256, 256, 64, 2, 2, 4, 0, True, False),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 4, 0, False, False),
        (2048, 2048, 2048, 128, 128, 64, 4, 4, 4, 0, True, False),
        (2048, 2048, 2048 - 64, 128, 128, 64, 2, 4, 4, 0, False, False),
        (2048, 2048, 2048 - 64, 128, 128, 64, 4, 4, 4, 0, True, False),
        # hti
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 0, False, True),
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 4, True, True),
        (8160, 8160, 8192, 256, 256, 64, 2, 2, 4, 0, False, True),
        (8160, 8160, 8192, 256, 256, 64, 2, 2, 4, 0, True, True),
        (8192, 8192, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, False, True),
        (8192, 8192, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, True, True),
        (8192, 8192, 8192 + 32, 256, 256, 64, 2, 2, 4, 0, False, True),
        (8192, 8192, 8192 + 32, 256, 256, 64, 2, 2, 4, 0, True, True),
        (8160, 8160, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, False, True),
        (8160, 8160, 8192 + 64, 256, 256, 64, 2, 2, 4, 0, True, True),
        (8160, 8160, 8192 + 32, 256, 256, 64, 2, 2, 4, 0, False, True),
        (8160, 8160, 8192 + 32, 256, 256, 64, 2, 2, 4, 0, True, True),
        (2048, 2048, 2048, 128, 128, 64, 2, 2, 2, 0, False, True),
        (2048, 2048, 2048 - 64, 128, 128, 64, 2, 2, 2, 0, False, True),
    ],
)
def test_hgemm_acc_main_loop(
    layout: str,
    dtype: str,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    group_m: int,
    has_bias: bool,
    is_hti: bool,
):
    dtype = torch.bfloat16 if "bf16" in dtype else torch.half
    args = _TestArgs(
        dtype,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        stages,
        m_waves,
        n_waves,
        1,
        group_m,
        has_bias,
        is_hti,
        layout,
    )
    check_acc(args)


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, split_k, "
    "m_waves, n_waves, k_waves, has_bias, group_m, "
    "use_half_tile_interleaved",
    [
        # Layout-kernel split-K smoke cases.
        (64, 64, 480, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, False),
        (64, 64, 480, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, True),
        # test_hgemm_acc_ft_stage_split_k
        (32, 384, 7168, 32, 64, 64, 5, 8, 2, 2, 1, True, 0, False),
        (32, 384, 7168, 32, 64, 64, 5, 8, 2, 2, 1, False, 0, False),
        (32, 384, 7168, 32, 64, 64, 5, 8, 2, 2, 1, True, 4, False),
        (32, 384, 7168, 32, 64, 64, 5, 8, 2, 2, 1, False, 4, False),
        # test_hgemm_acc_ht_split_k
        (64, 384, 7168, 64, 64, 64, 2, 8, 2, 2, 1, True, 0, True),
        (64, 384, 7168, 64, 64, 64, 2, 8, 2, 2, 1, False, 0, True),
        (64, 384, 7168, 64, 64, 64, 2, 8, 2, 2, 1, True, 4, True),
        (64, 384, 7168, 64, 64, 64, 2, 8, 2, 2, 1, False, 4, True),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 2, 2, 1, True, 0, True),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 2, 2, 1, False, 0, True),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 2, 2, 1, True, 4, True),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 2, 2, 1, False, 4, True),
        # test_hgemm_acc_small_m
        (3, 5120, 2880, 64, 64, 64, 5, 3, 2, 2, 1, True, 0, False),
        (3, 5120, 2880, 64, 64, 64, 5, 3, 2, 2, 1, False, 0, False),
        (3, 5120, 2880, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, True),
        (3, 5120, 2880, 64, 64, 64, 2, 3, 2, 2, 1, False, 0, True),
        (3, 5120, 2880, 64, 64, 64, 2, 3, 2, 2, 1, True, 4, True),
        # test_hgemm_acc_small_mnk
        (3, 16, 16, 64, 64, 64, 2, 3, 2, 2, 1, False, 0, False),
        (3, 16, 16, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, False),
        (3, 16, 48, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, False),
        (3, 16, 80, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, False),
        (3, 16, 144, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, False),
        (3, 16, 144, 64, 64, 64, 3, 3, 2, 2, 1, True, 0, False),
        (3, 16, 16, 64, 64, 64, 4, 3, 2, 2, 1, True, 0, False),
        (3, 16, 16, 64, 64, 64, 2, 3, 2, 2, 1, False, 0, True),
        (3, 16, 16, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, True),
        (3, 16, 48, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, True),
        (3, 16, 80, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, True),
        (3, 16, 144, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, True),
        (3, 16, 144, 64, 64, 64, 2, 3, 2, 2, 1, False, 4, True),
        # test_hgemm_acc_ft_slice_k
        (800, 384, 7168, 32, 64, 128, 6, 1, 1, 2, 2, True, 0, False),
        (800, 384, 7168, 32, 64, 128, 6, 1, 1, 2, 2, False, 0, False),
        (800, 384, 7168, 32, 64, 128, 6, 2, 1, 2, 2, True, 0, False),
        (800, 384, 7168, 32, 64, 128, 6, 2, 1, 2, 2, False, 0, False),
        (800, 384, 7168, 32, 64, 128, 6, 2, 1, 2, 2, False, 4, False),
        # test_hgemm_acc_ft_special
        (16, 2880, 512, 16, 128, 128, 4, 2, 1, 1, 4, True, 0, False),
        (48, 128, 2880, 80, 64, 64, 4, 12, 1, 1, 2, True, 0, False),
        (48, 640, 2880, 96, 64, 64, 5, 2, 1, 1, 1, True, 0, False),
    ],
)
def test_hgemm_acc_split_k(
    layout: str,
    dtype: torch.dtype,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    split_k: int,
    m_waves: int,
    n_waves: int,
    k_waves: int,
    has_bias: bool,
    group_m: int,
    use_half_tile_interleaved: bool,
):
    assert k_waves > 0
    assert split_k > 1 or k_waves > 1
    args = _TestArgs(
        dtype=dtype,
        m=m,
        n=n,
        k=k,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        k_waves=k_waves,
        group_m=group_m,
        has_bias=has_bias,
        use_half_tile_interleaved=use_half_tile_interleaved,
        layout=layout,
        split_k=split_k,
    )
    check_acc(args)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("split_k", [1, 3])
@pytest.mark.parametrize("use_half_tile_interleaved", [False, True])
def test_hgemm_acc_fp32_output(
    dtype: torch.dtype,
    split_k: int,
    use_half_tile_interleaved: bool,
):
    args = _TestArgs(
        dtype=dtype,
        m=64,
        n=64,
        k=480,
        block_m=64,
        block_n=64,
        block_k=64,
        stages=2,
        m_waves=2,
        n_waves=2,
        k_waves=1,
        group_m=0,
        has_bias=True,
        use_half_tile_interleaved=use_half_tile_interleaved,
        split_k=split_k,
        out_dtype=torch.float32,
    )
    check_acc(args)


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize(
    "dtype",
    [
        "bf16",
        "fp16",
    ],
)
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias, is_hti",
    [
        (3, 5120, 2880, 64, 64, 64, 5, 2, 2, 4, True, False),
        (3, 5120, 2880, 64, 64, 64, 5, 2, 2, 0, False, False),
        (3, 5120, 2880, 64, 64, 64, 2, 2, 2, 0, True, False),
        (3, 5120, 2880, 64, 64, 64, 2, 2, 2, 0, False, False),
        (3, 32, 128 + 64, 128, 128, 64, 3, 2, 2, 4, True, False),
        (3, 32, 128 - 64, 128, 128, 64, 3, 2, 2, 0, False, False),
        # hti
        (3, 5120, 2880, 64, 64, 64, 2, 2, 2, 4, True, True),
        (3, 5120, 2880, 64, 64, 64, 2, 2, 2, 0, False, True),
        (3, 5120, 2880, 64, 64, 64, 2, 2, 2, 0, True, True),
        (3, 5120, 2880, 64, 64, 64, 2, 2, 2, 0, False, True),
        (3, 32, 128 + 64, 128, 128, 64, 2, 2, 2, 4, True, True),
        (3, 32, 128 + 64 * 3, 128, 128, 64, 2, 2, 2, 4, True, True),
    ],
)
def test_hgemm_acc_small_m(
    layout: str,
    dtype: str,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    group_m: int,
    has_bias: bool,
    is_hti: bool,
):
    dtype = torch.bfloat16 if "bf16" in dtype else torch.half
    args = _TestArgs(
        dtype,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        stages,
        m_waves,
        n_waves,
        1,
        group_m,
        has_bias,
        is_hti,
        layout,
    )
    check_acc(args)


@pytest.mark.large_shape
@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize(
    "dtype",
    [
        "bf16",
        "fp16",
    ],
)
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias, is_hti",
    [
        (8, 4096, 4096, 16, 16, 128, 8, 1, 1, 4, True, False),
        (16, 4096, 4096, 16, 16, 128, 8, 1, 1, 4, True, False),
        (32, 4096, 4096, 16, 16, 64, 8, 1, 1, 0, True, False),
        (64, 4096, 4096, 32, 32, 64, 8, 2, 2, 4, True, False),
        (128, 4096, 4096, 64, 32, 128, 4, 4, 2, 4, True, False),
        (256, 4096, 4096, 64, 64, 64, 7, 4, 2, 4, True, False),
        (512, 4096, 4096, 64, 128, 64, 6, 2, 4, 4, True, False),
        (1024, 4096, 4096, 128, 128, 64, 4, 2, 4, 4, True, False),
        (2048, 4096, 4096, 128, 256, 64, 3, 4, 4, 4, True, False),
        (1024, 1024, 1024, 64, 64, 64, 4, 2, 4, 0, True, False),
        (2048, 2048, 2048, 128, 128, 64, 3, 4, 2, 0, True, False),
        (4096, 4096, 4096, 256, 256, 64, 2, 2, 4, 4, True, True),
        (4096, 4096, 8192, 256, 256, 64, 2, 2, 4, 4, True, True),
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 0, True, True),
        (16384, 16384, 16384, 256, 256, 64, 2, 2, 4, 4, True, True),
        (8, 7168, 2048, 16, 16, 128, 8, 1, 1, 4, True, False),
        (32, 384, 7168, 16, 16, 128, 8, 1, 1, 0, True, False),
        (32, 14336, 4096, 32, 64, 64, 8, 2, 2, 0, True, False),
        (16, 28672, 4096, 16, 64, 128, 3, 1, 4, 4, True, False),
        (4096, 256, 4096, 64, 64, 64, 6, 4, 2, 4, True, False),
    ],
)
def test_hgemm_acc_bench(
    layout: str,
    dtype: str,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    group_m: int,
    has_bias: bool,
    is_hti: bool,
):
    dtype = torch.bfloat16 if "bf16" in dtype else torch.half
    args = _TestArgs(
        dtype,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        stages,
        m_waves,
        n_waves,
        1,
        group_m,
        has_bias,
        is_hti,
        layout,
    )
    check_acc(args)


@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("use_half_tile_interleaved", [False, True])
def test_hgemm_padded_stride_and_storage_offset(
    layout: str,
    use_half_tile_interleaved: bool,
):
    m = n = 64
    k = 128
    dtype = torch.bfloat16
    column_offset = 8

    def make_padded_matrix(rows: int, cols: int, padding: int, is_t: bool):
        if is_t:
            pitch = rows + padding
            storage = torch.empty((cols + 1, pitch), dtype=dtype, device="cuda")
            storage.uniform_(-1, 1)
            return storage[1:, column_offset : column_offset + rows].t()
        pitch = cols + padding
        storage = torch.empty((rows + 1, pitch), dtype=dtype, device="cuda")
        storage.uniform_(-1, 1)
        return storage[1:, column_offset : column_offset + cols]

    a = make_padded_matrix(m, k, 32, layout[0] == "t")
    b = make_padded_matrix(k, n, 48, layout[1] == "t")

    assert a.shape == (m, k)
    assert b.shape == (k, n)
    for tensor, is_t in zip(
        (a, b),
        (layout[0] == "t", layout[1] == "t"),
    ):
        assert not tensor.is_contiguous()
        if is_t:
            assert tensor.stride(0) == 1
            assert tensor.stride(1) > tensor.shape[0]
            leading_stride = tensor.stride(1)
        else:
            assert tensor.stride(1) == 1
            assert tensor.stride(0) > tensor.shape[1]
            leading_stride = tensor.stride(0)
        assert tensor.storage_offset() > 0
        assert tensor.data_ptr() % 16 == 0
        assert leading_stride * tensor.element_size() % 16 == 0

    bias = torch.empty((n,), dtype=dtype, device="cuda").uniform_(-1, 1)
    out = torch.empty((m, n), dtype=dtype, device="cuda")
    ref = torch.empty_like(out)
    kwargs = {
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "stages": 2,
        "m_waves": 2,
        "n_waves": 2,
        "group_m": 0,
        "use_half_tile_interleaved": use_half_tile_interleaved,
    }

    result = hgemm(
        a,
        b,
        out,
        bias=bias,
        user_kwargs=kwargs,
        layout=layout,
    )
    ref_func(a, b, bias, ref, layout)

    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        out,
        ref,
        atol=3e-2,
        rtol=2e-1,
        check_dtype=True,
    )


# =========================================== benchmark ===========================================


@pytest.mark.benchmark
@pytest.mark.skipif(
    os.environ.get("FLYDSL_GEMM_A16W16_BENCH") != "1",
    reason="set FLYDSL_GEMM_A16W16_BENCH=1 to run performance benchmarks",
)
@pytest.mark.parametrize("layout", ["nn", "nt", "tn", "tt"])
@pytest.mark.parametrize("dtype", ["bf16"])
@pytest.mark.parametrize(
    "m, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, has_bias, is_hti",
    [
        (8, 4096, 4096, 16, 16, 128, 8, 1, 1, 4, True, False),
        (16, 4096, 4096, 16, 16, 128, 8, 1, 1, 4, True, False),
        (32, 4096, 4096, 16, 16, 64, 8, 1, 1, 0, True, False),
        (64, 4096, 4096, 32, 32, 64, 8, 2, 2, 4, True, False),
        (128, 4096, 4096, 64, 32, 128, 4, 4, 2, 4, True, False),
        (256, 4096, 4096, 64, 64, 64, 7, 4, 2, 4, True, False),
        (512, 4096, 4096, 64, 128, 64, 6, 2, 4, 4, True, False),
        (1024, 4096, 4096, 128, 128, 64, 4, 2, 4, 4, True, False),
        (2048, 4096, 4096, 128, 256, 64, 3, 4, 4, 4, True, False),
        (1024, 1024, 1024, 64, 64, 64, 4, 2, 4, 0, True, False),
        (2048, 2048, 2048, 128, 128, 64, 3, 4, 2, 0, True, False),
        (4096, 4096, 4096, 256, 256, 64, 2, 2, 4, 4, True, True),
        (4096, 4096, 8192, 256, 256, 64, 2, 2, 4, 4, True, True),
        (8192, 8192, 8192, 256, 256, 64, 2, 2, 4, 0, True, True),
        (16384, 16384, 16384, 256, 256, 64, 2, 2, 4, 4, True, True),
        (8, 7168, 2048, 16, 16, 128, 8, 1, 1, 4, True, False),
        (32, 384, 7168, 16, 16, 128, 8, 1, 1, 0, True, False),
        (32, 14336, 4096, 32, 64, 64, 8, 2, 2, 0, True, False),
        (16, 28672, 4096, 16, 64, 128, 3, 1, 4, 4, True, False),
        (4096, 256, 4096, 64, 64, 64, 6, 4, 2, 4, True, False),
    ],
)
def test_hgemm_benchmark_smoke(
    layout: str,
    dtype: str,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    group_m: int,
    has_bias: bool,
    is_hti: bool,
):
    dtype = torch.bfloat16 if "bf16" in dtype else torch.half
    args = _TestArgs(
        dtype,
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        stages,
        m_waves,
        n_waves,
        1,
        group_m,
        has_bias,
        is_hti,
        layout,
    )
    benchmark(args)


def hgemm_get_configs():
    selections = {
        "block_m": [16, 32, 48, 64, 80, 96, 128, 256],
        "block_n": [16, 32, 64, 80, 96, 128, 256],
        "block_k": [64, 128, 256],
        "stages": [i for i in range(2, 10)],
        "m_waves": [1, 2, 4],
        "n_waves": [1, 2, 4],
        "group_m": [0, 4],
    }
    keys = selections.keys()
    values = selections.values()
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    valid_configs = []
    for config in configs:
        mma_m_iters = config["block_m"] // config["m_waves"] // 16
        mma_n_iters = config["block_n"] // config["n_waves"] // 16
        if mma_m_iters > 4 or mma_n_iters > 4:
            continue
        try:
            make_hgemm_gfx950_param(**config)
            valid_configs.append(config)
        except Exception:
            pass
    return valid_configs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A16W16 universe GEMM benchmark")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--layout", choices=("nn", "nt", "tn", "tt"), default="nt")
    parser.add_argument("-m", type=int, default=2048)
    parser.add_argument("-n", type=int, default=2048)
    parser.add_argument("-k", type=int, default=2048)
    parser.add_argument("--block_m", type=int, default=128)
    parser.add_argument("--block_n", type=int, default=128)
    parser.add_argument("--block_k", type=int, default=64)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--split_k", type=int, default=1)
    parser.add_argument("--m_waves", type=int, default=4)
    parser.add_argument("--n_waves", type=int, default=4)
    parser.add_argument("--k_waves", type=int, default=1)
    parser.add_argument("--group_m", type=int, default=0)
    parser.add_argument("--has_bias", action="store_true")
    parser.add_argument("--use_half_tile_interleaved", action="store_true")
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--num_iters", type=int, default=50)
    cli = parser.parse_args()

    benchmark_flydsl(
        _TestArgs(
            dtype=torch.float16 if cli.dtype == "fp16" else torch.bfloat16,
            m=cli.m,
            n=cli.n,
            k=cli.k,
            block_m=cli.block_m,
            block_n=cli.block_n,
            block_k=cli.block_k,
            stages=cli.stages,
            m_waves=cli.m_waves,
            n_waves=cli.n_waves,
            k_waves=cli.k_waves,
            group_m=cli.group_m,
            has_bias=cli.has_bias,
            use_half_tile_interleaved=cli.use_half_tile_interleaved,
            layout=cli.layout,
            split_k=cli.split_k,
        ),
        warmup=cli.num_warmup,
        niters=cli.num_iters,
    )
