#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for warp_argmax, warp_argmin, and WarpReduce.argmax / argmin."""

from __future__ import annotations

import math

import pytest
from coop_common import WARP_SIZE, WARP_WIDTHS
from coop_test_utils import matrix_cases, run_kernel
from coop_test_utils import warp_default_device as warp_default_device

import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    import torch
except ImportError:
    torch = None

portable = fx.coop.universal
ALL_WIDTHS = (1, *WARP_WIDTHS)

ARGEXT_DTYPES = (
    (fx.Uint8, "uint8"),
    (fx.Int16, "int16"),
    (fx.Int32, "int32"),
    (fx.Int64, "int64"),
    (fx.Float16, "float16"),
    (fx.BFloat16, "bfloat16"),
    (fx.Float32, "float32"),
    (fx.Float64, "float64"),
)


def _ref_argext(group_vals: list[float | int], is_max: bool):
    """Exact host reference matching IEEE NaN-propagation and lowest-index tie-breaking."""
    nan_indices = [i for i, v in enumerate(group_vals) if isinstance(v, float) and math.isnan(v)]
    if nan_indices:
        return float("nan"), nan_indices[0]
    best_val = group_vals[0]
    best_idx = 0
    for idx in range(1, len(group_vals)):
        v = group_vals[idx]
        if is_max:
            if v > best_val:
                best_val = v
                best_idx = idx
            elif v == best_val and isinstance(v, float) and math.copysign(1.0, v) > math.copysign(1.0, best_val):
                best_val = v  # max(-0.0, +0.0) == +0.0 while keeping lowest index
        else:
            if v < best_val:
                best_val = v
                best_idx = idx
            elif v == best_val and isinstance(v, float) and math.copysign(1.0, v) < math.copysign(1.0, best_val):
                best_val = v  # min(+0.0, -0.0) == -0.0 while keeping lowest index
    return best_val, best_idx


@pytest.mark.l1a_compile_no_target_dialect
def test_universal_argext_emits_no_target_dialect(ctx, insert_point):
    """Portable warp_argmax/warp_argmin must lower through generic GPU/arith ops only."""
    x = fx.Float32(fx.arith.constant(1.5, type=fx.T.f32))
    max_v, max_i = portable.warp_argmax(x, width=8)
    min_v, min_i = portable.warp_argmin(x, width=8)
    R = portable.WarpReduce[fx.Float32, 8]
    rmax_v, rmax_i = R.argmax(x)
    _ = (max_v, max_i, min_v, min_i, rmax_v, rmax_i)

    ir_str = str(ctx.module)
    assert "gpu.shuffle" in ir_str
    assert "rocdl." not in ir_str


@pytest.mark.l1a_compile_no_target_dialect
def test_argext_validation_errors(insert_point):
    record = fx.Struct["a" : fx.Int32, "b" : fx.Int32]
    with pytest.raises(TypeError, match="scalar Numeric"):
        fx.coop.WarpReduce[record, 4].argmax(record(fx.Int32(1), fx.Int32(2)))
    with pytest.raises(TypeError, match="scalar Numeric"):
        fx.coop.warp_argmin(record(fx.Int32(1), fx.Int32(2)), width=4)
    with pytest.raises(TypeError, match="single item"):
        fx.coop.warp_argmax(fx.Vector.filled(2, 1, fx.Int32), width=4, valid_items=2)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "dtype_pair,width",
    matrix_cases(ARGEXT_DTYPES, ALL_WIDTHS),
)
@pytest.mark.usefixtures("warp_default_device")
def test_warp_argmax_argmin_dtypes_and_ties(dtype_pair, width):
    fly_dtype, torch_name = dtype_pair
    tdt = getattr(torch, torch_name)
    block = 2 * WARP_SIZE
    group = WARP_SIZE if width is None else width

    torch.manual_seed(11)
    if tdt.is_floating_point:
        # Quantize values to small integers / halves so repeated ties occur naturally.
        values = torch.randint(-6, 7, (block,), device="cuda").to(tdt) * 0.5
    elif tdt == torch.uint8:
        values = torch.randint(0, 9, (block,), dtype=tdt, device="cuda")
    else:
        values = torch.randint(-8, 9, (block,), dtype=tdt, device="cuda")

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, FastVal: fx.Tensor, FastIdx: fx.Tensor, RefVal: fx.Tensor, RefIdx: fx.Tensor):
        tid = fx.thread_idx.x
        v = A[tid]
        R_fast = fx.coop.WarpReduce[fly_dtype, width]
        R_ref = portable.WarpReduce[fly_dtype, width]

        f_max_v, f_max_i = fx.coop.warp_argmax(v, width=width)
        f_min_v, f_min_i = R_fast.argmin(v)
        r_max_v, r_max_i = portable.warp_argmax(v, width=width)
        r_min_v, r_min_i = R_ref.argmin(v)

        FastVal[tid * 2 + 0] = f_max_v
        FastVal[tid * 2 + 1] = f_min_v
        FastIdx[tid * 2 + 0] = f_max_i
        FastIdx[tid * 2 + 1] = f_min_i

        RefVal[tid * 2 + 0] = r_max_v
        RefVal[tid * 2 + 1] = r_min_v
        RefIdx[tid * 2 + 0] = r_max_i
        RefIdx[tid * 2 + 1] = r_min_i

    @flyc.jit
    def launch(
        A: fx.Tensor,
        FastVal: fx.Tensor,
        FastIdx: fx.Tensor,
        RefVal: fx.Tensor,
        RefIdx: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(A, FastVal, FastIdx, RefVal, RefIdx).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    f_val = torch.empty(block * 2, dtype=tdt, device="cuda")
    r_val = torch.empty(block * 2, dtype=tdt, device="cuda")
    f_idx = torch.empty(block * 2, dtype=torch.int32, device="cuda")
    r_idx = torch.empty(block * 2, dtype=torch.int32, device="cuda")
    launch(values, f_val, f_idx, r_val, r_idx, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    assert torch.equal(f_val.cpu(), r_val.cpu())
    assert torch.equal(f_idx.cpu(), r_idx.cpu())

    src_groups = values.cpu().reshape(-1, group).tolist()
    got_vals = f_val.cpu().reshape(block, 2)
    got_idxs = f_idx.cpu().reshape(block, 2)
    for g, row_vals in enumerate(src_groups):
        exp_max_v, exp_max_i = _ref_argext(row_vals, True)
        exp_min_v, exp_min_i = _ref_argext(row_vals, False)
        for k in range(group):
            lane_row = g * group + k
            assert float(got_vals[lane_row, 0].item()) == float(exp_max_v)
            assert float(got_vals[lane_row, 1].item()) == float(exp_min_v)
            assert int(got_idxs[lane_row, 0].item()) == exp_max_i
            assert int(got_idxs[lane_row, 1].item()) == exp_min_i


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("dtype_pair", ((fx.Float32, "float32"), (fx.Float16, "float16"), (fx.BFloat16, "bfloat16")))
@pytest.mark.usefixtures("warp_default_device")
def test_warp_argext_signed_zero_inf_and_nan(dtype_pair):
    """Verify exact IEEE signed-zero, infinity, and NaN tie-breaking between ROCDL and universal."""
    fly_dtype, torch_name = dtype_pair
    tdt = getattr(torch, torch_name)
    width = 8
    block = WARP_SIZE

    # Build 4 groups of 8 covering:
    # group 0: [-0.0, +0.0, -1.0, -2.0, ...] -> max = +0.0 at idx 0; min checks
    # group 1: [+0.0, -0.0, +1.0, +2.0, ...] -> min = -0.0 at idx 0
    # group 2: [1.0, inf, -inf, inf, -inf, 2.0, 0.0, -5.0] -> max = inf at idx 1, min = -inf at idx 2
    # group 3: [3.0, 10.0, nan, 99.0, nan, -inf, inf, 0.0] -> max = nan at idx 2, min = nan at idx 2
    groups = [
        [-0.0, +0.0, -1.0, -2.0, -3.0, -0.5, -4.0, -0.0],
        [+0.0, -0.0, +1.0, +2.0, +3.0, +0.5, +4.0, +0.0],
        [1.0, float("inf"), float("-inf"), float("inf"), float("-inf"), 2.0, 0.0, -5.0],
        [3.0, 10.0, float("nan"), 99.0, float("nan"), float("-inf"), float("inf"), 0.0],
    ]
    flat = [x for row in (groups * (block // 32 + 1)) for x in row][:block]
    values = torch.tensor(flat, dtype=torch.float32, device="cuda").to(tdt)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, FastVal: fx.Tensor, FastIdx: fx.Tensor, RefVal: fx.Tensor, RefIdx: fx.Tensor):
        tid = fx.thread_idx.x
        v = A[tid]
        f_max_v, f_max_i = fx.coop.warp_argmax(v, width=width)
        f_min_v, f_min_i = fx.coop.warp_argmin(v, width=width)
        r_max_v, r_max_i = portable.warp_argmax(v, width=width)
        r_min_v, r_min_i = portable.warp_argmin(v, width=width)
        FastVal[tid * 2 + 0] = f_max_v
        FastVal[tid * 2 + 1] = f_min_v
        FastIdx[tid * 2 + 0] = f_max_i
        FastIdx[tid * 2 + 1] = f_min_i
        RefVal[tid * 2 + 0] = r_max_v
        RefVal[tid * 2 + 1] = r_min_v
        RefIdx[tid * 2 + 0] = r_max_i
        RefIdx[tid * 2 + 1] = r_min_i
        _ = fly_dtype

    @flyc.jit
    def launch(
        A: fx.Tensor,
        FastVal: fx.Tensor,
        FastIdx: fx.Tensor,
        RefVal: fx.Tensor,
        RefIdx: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(A, FastVal, FastIdx, RefVal, RefIdx).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    f_val = torch.empty(block * 2, dtype=tdt, device="cuda")
    r_val = torch.empty(block * 2, dtype=tdt, device="cuda")
    f_idx = torch.empty(block * 2, dtype=torch.int32, device="cuda")
    r_idx = torch.empty(block * 2, dtype=torch.int32, device="cuda")
    launch(values, f_val, f_idx, r_val, r_idx, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    assert torch.equal(f_idx.cpu(), r_idx.cpu())
    f_f32 = f_val.cpu().to(torch.float32).reshape(block, 2)
    r_f32 = r_val.cpu().to(torch.float32).reshape(block, 2)
    f_idx_cpu = f_idx.cpu().reshape(block, 2)

    src_groups = values.cpu().to(torch.float32).reshape(-1, width).tolist()
    for g, row_vals in enumerate(src_groups):
        exp_max_v, exp_max_i = _ref_argext(row_vals, True)
        exp_min_v, exp_min_i = _ref_argext(row_vals, False)
        for k in range(width):
            r = g * width + k
            assert int(f_idx_cpu[r, 0].item()) == exp_max_i
            assert int(f_idx_cpu[r, 1].item()) == exp_min_i
            if math.isnan(exp_max_v):
                assert math.isnan(f_f32[r, 0].item()) and math.isnan(r_f32[r, 0].item())
            else:
                assert f_f32[r, 0].item() == exp_max_v == r_f32[r, 0].item()
                assert math.copysign(1.0, f_f32[r, 0].item()) == math.copysign(1.0, exp_max_v)
            if math.isnan(exp_min_v):
                assert math.isnan(f_f32[r, 1].item()) and math.isnan(r_f32[r, 1].item())
            else:
                assert f_f32[r, 1].item() == exp_min_v == r_f32[r, 1].item()
                assert math.copysign(1.0, f_f32[r, 1].item()) == math.copysign(1.0, exp_min_v)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("width,form", matrix_cases((4, None), ("vector", "tuple", "valid_items")))
@pytest.mark.usefixtures("warp_default_device")
def test_warp_argext_blocked_ranges_and_valid_items(width, form):
    """Verify multi-item blocked indices (lane * count + item) and guarded valid_items."""
    block = 2 * WARP_SIZE
    group = WARP_SIZE if width is None else width
    count = 1 if form == "valid_items" else 3
    valid = max(1, group - 2) if form == "valid_items" else None

    torch.manual_seed(19)
    values = torch.randint(-15, 16, (block * count,), dtype=torch.int32, device="cuda")

    def apply(a, out):
        tid = fx.thread_idx.x
        if form == "valid_items":
            item = a[tid]
            f_max_v, f_max_i = fx.coop.warp_argmax(item, width=width, valid_items=fx.Int32(valid))
            r_max_v, r_max_i = portable.warp_argmax(item, width=width, valid_items=fx.Int32(valid))
            f_min_v, f_min_i = fx.coop.warp_argmin(item, width=width, valid_items=valid)
            r_min_v, r_min_i = portable.warp_argmin(item, width=width, valid_items=valid)
        else:
            items = [a[tid * count + i] for i in fx.range_constexpr(count)]
            tile = fx.Vector.from_elements(items) if form == "vector" else tuple(items)
            f_max_v, f_max_i = fx.coop.warp_argmax(tile, width=width)
            r_max_v, r_max_i = portable.warp_argmax(tile, width=width)
            f_min_v, f_min_i = fx.coop.warp_argmin(tile, width=width)
            r_min_v, r_min_i = portable.warp_argmin(tile, width=width)

        base = tid * 8
        out[base + 0] = f_max_v
        out[base + 1] = f_max_i
        out[base + 2] = r_max_v
        out[base + 3] = r_max_i
        out[base + 4] = f_min_v
        out[base + 5] = f_min_i
        out[base + 6] = r_min_v
        out[base + 7] = r_min_i

    actual = run_kernel(apply, values, block * 8, block).reshape(block, 8)
    assert torch.equal(actual[:, 0:2], actual[:, 2:4])
    assert torch.equal(actual[:, 4:6], actual[:, 6:8])

    grouped = values.cpu().reshape(-1, group * count)
    active_len = valid if valid is not None else group * count
    for g in range(grouped.shape[0]):
        prefix = grouped[g, :active_len].tolist()
        exp_max_v, exp_max_i = _ref_argext(prefix, True)
        exp_min_v, exp_min_i = _ref_argext(prefix, False)
        for k in range(group):
            r = g * group + k
            assert actual[r, 0].item() == exp_max_v and actual[r, 1].item() == exp_max_i
            assert actual[r, 4].item() == exp_min_v and actual[r, 5].item() == exp_min_i
