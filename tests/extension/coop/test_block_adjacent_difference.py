# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Block adjacent differences across tile boundaries, dtypes, and record fields."""

import math

import pytest
from coop_common import BLOCK_THREADS, DTYPES
from coop_test_utils import ARCHES, matrix_cases, run_tile
from coop_test_utils import coop_default_device as coop_default_device

import flydsl.expr as fx

try:
    import torch
except ImportError:
    torch = None
import coop_block_utils as checks


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "block,count,boundary,method",
    matrix_cases([64, (8, 4, 4)], [1, 3], [None, -9], ["subtract_left", "subtract_right"]),
)
@pytest.mark.usefixtures("coop_default_device")
def test_neighbors(block, count, boundary, method):
    checks.check_neighbors(block, count, boundary, method)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "block,count,side,valid,boundary",
    [
        (block, count, side, valid, boundary)
        for (block, count), side, valid, boundary in matrix_cases(
            [(64, 1), (64, 3), (64, 9), (128, 3), ((4, 4, 12), 2), (1024, 1)],
            ["left", "right"],
            [0, 1, 7, None],
            [None, -11],
        )
    ],
)
def test_partial_adjacent(block, count, side, valid, boundary):
    threads = block if isinstance(block, int) else math.prod(block)
    size = threads * count
    limit = size if valid is None else min(size, valid)
    values = ((torch.arange(size, device="cuda", dtype=torch.int32) * 17) % 31) - 15

    def op(current, neighbor):
        return current * 3 - neighbor * 7

    def transform(value):
        cls = fx.coop.universal.BlockAdjacentDifference[fx.Int32, block]
        storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
        kwargs = {"predecessor" if side == "left" else "successor": boundary}
        return getattr(cls, "subtract_" + side + "_partial_tile")(value, fx.Int32(limit), op, storage=storage, **kwargs)

    actual = run_tile(transform, values, block, count)[0]
    expected = values.cpu().clone()
    for i in range(limit):
        j = i - 1 if side == "left" else i + 1
        if 0 <= j < limit:
            expected[i] = values.cpu()[i] * 3 - values.cpu()[j] * 7
        elif boundary is not None:
            expected[i] = values.cpu()[i] * 3 - boundary * 7
    torch.testing.assert_close(actual, expected)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.usefixtures("coop_default_device")
def test_right_difference_unsigned_zero_valid_items(partial):
    # An unsigned empty prefix must not wrap ``valid_items - 1`` and activate every item.
    block, count = 64, 2
    values = torch.arange(block * count, device="cuda", dtype=torch.int32) * 3

    def transform(value):
        cls = fx.coop.BlockAdjacentDifference[fx.Int32, block]
        storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
        if partial:
            return cls.subtract_right_partial_tile(value, fx.Uint32(0), storage=storage)
        return cls.subtract_right(value, storage=storage, valid_items=fx.Uint32(0))

    torch.testing.assert_close(run_tile(transform, values, block, count)[0], values.cpu())


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "dtype,name,method",
    [
        (dtype, name, method)
        for (dtype, name), method in matrix_cases(
            [
                (fx.Uint8, "uint8"),
                (fx.Int16, "int16"),
                (fx.Uint16, "int16"),
                (fx.Int32, "int32"),
                (fx.Int64, "int64"),
                (fx.Float32, "float32"),
                (fx.Float64, "float64"),
            ],
            ["partial_left", "partial_right"],
        )
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_neighborhood_contract_dtypes(dtype, name, method):
    checks.check_neighborhood_contract_dtypes(dtype, name, method)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "block,count,method,universal",
    [
        (block, count, method, universal)
        for (block, count), method, universal in matrix_cases(
            [(64, 1), (64, 3), (128, 3)], ["left", "right"], [False, True]
        )
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_sequence_records(block, count, method, universal, compile_only=False):
    checks.check_sequence_records(block, count, method, universal, compile_only)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.parametrize("arch,method", matrix_cases(ARCHES, ["left"]))
def test_record_sequence_compile(monkeypatch, arch, method):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    checks.check_sequence_records(192, 3, method, True, compile_only=True)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "entry,count,case",
    matrix_cases(
        DTYPES, [1, 9], [("BlockAdjacentDifference", "subtract_left"), ("BlockAdjacentDifference", "subtract_right")]
    ),
)
@pytest.mark.usefixtures("coop_default_device")
def test_matrix_block_dtypes(entry, count, case):
    checks.check_block_case(entry, 128, count, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "block,count,case",
    matrix_cases(
        BLOCK_THREADS,
        [1, 9],
        [("BlockAdjacentDifference", "subtract_left"), ("BlockAdjacentDifference", "subtract_right")],
    ),
)
@pytest.mark.usefixtures("coop_default_device")
def test_matrix_block_shapes(block, count, case):
    checks.check_block_case((fx.Int32, "torch.int32"), block, count, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "case",
    [("BlockAdjacentDifference", "subtract_left"), ("BlockAdjacentDifference", "subtract_right")],
    ids=lambda c: c[1],
)
@pytest.mark.usefixtures("coop_default_device")
def test_matrix_three_dimensional_block(case):
    checks.check_block_case((fx.Float64, "torch.float64"), (8, 4, 4), 3, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize(
    "method,valid,boundary", matrix_cases(["subtract_left", "subtract_right"], [None, 0, 1, 3], [False, True])
)
@pytest.mark.usefixtures("coop_default_device")
def test_neighbor_operator_calls(method, valid, boundary):
    checks.check_neighbor_operator_calls(method, valid, boundary)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize(
    "method,count,valid,boundary",
    matrix_cases(["subtract_left_partial_tile", "subtract_right_partial_tile"], [1, 3], [0, 1, 7], [False, True]),
)
@pytest.mark.usefixtures("coop_default_device")
def test_output_dtype_and_operator_calls(method, count, valid, boundary):
    """Output conversion must not change operands or invoke skipped differences."""
    from coop_test_utils import run_kernel

    block = 64
    size = block * count
    host = torch.arange(size, dtype=torch.float32, device="cpu") * 3
    left = "left" in method

    def apply(a, out):
        tid = fx.thread_idx.x
        items = [a[tid * count + i].to(fx.Int32) for i in range(count)]
        out[size + tid] = fx.Float32(0)

        def difference(current, neighbor):
            assert isinstance(current, fx.Int32) and isinstance(neighbor, fx.Int32)
            out[size + tid] = out[size + tid] + 1
            return (current.to(fx.Float64) - neighbor.to(fx.Float64)) * 0.5

        ns = fx.coop.universal if boundary else fx.coop
        p = ns.BlockAdjacentDifference[fx.Int32, block]
        storage = fx.SharedAllocator().allocate(p.SharedStorage).peek()
        kwargs = {"predecessor" if left else "successor": -7} if boundary else {}
        result = getattr(p, method)(
            items[0] if count == 1 else items,
            fx.Int32(valid),
            difference,
            storage=storage,
            output_dtype=fx.Float32,
            **kwargs,
        )
        result = (result,) if count == 1 else result
        for i in range(count):
            assert isinstance(result[i], fx.Float32)
            out[tid * count + i] = result[i]

    actual = run_kernel(apply, host.to(device="cuda"), size + block, block)
    expected = host.clone()
    calls = torch.zeros(block, dtype=torch.float32, device="cpu")
    for i in range(valid):
        j = i - 1 if left else i + 1
        if 0 <= j < valid or boundary:
            expected[i] = (host[i] - (host[j] if 0 <= j < valid else -7)) * 0.5
            calls[i // count] += 1
    torch.testing.assert_close(actual[:size], expected)
    torch.testing.assert_close(actual[size:], calls)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("record,array", matrix_cases([False, True], [False, True]))
@pytest.mark.usefixtures("coop_default_device")
def test_composite_output_dtype(record, array):
    """Complete Vector and Struct elements retain their boundaries after conversion."""
    from coop_test_utils import run_kernel

    block, count = 64, 2 if array else 1
    vector_t, vector_u = fx.Vector[fx.Int32, 2], fx.Vector[fx.Float32, 2]
    input_t = fx.Struct["data":vector_t] if record else vector_t
    output_t = fx.Struct["data":vector_u] if record else vector_u
    host = torch.arange(block * count * 2, dtype=torch.float32, device="cpu")

    def apply(a, out):
        tid = fx.thread_idx.x
        vectors = [
            fx.Vector.from_elements([a[(tid * count + i) * 2 + j].to(fx.Int32) for j in range(2)]) for i in range(count)
        ]
        items = [input_t(v) for v in vectors] if record else vectors

        def difference(current, neighbor):
            lhs, rhs = (current.data, neighbor.data) if record else (current, neighbor)
            result = (lhs.to(fx.Float32) - rhs.to(fx.Float32)) * 0.5
            return output_t(result) if record else result

        p = fx.coop.BlockAdjacentDifference[input_t, block]
        storage = fx.SharedAllocator().allocate(p.SharedStorage).peek()
        result = p.subtract_left(items if array else items[0], difference, storage=storage, output_dtype=output_t)
        assert isinstance(result, tuple) if array else not isinstance(result, tuple)
        for i, item in enumerate(result if array else (result,)):
            v = item.data if record else item
            assert v.dtype is fx.Float32 and v.shape == (2,)
            for j in range(2):
                out[(tid * count + i) * 2 + j] = v[j]

    actual = run_kernel(apply, host.to(device="cuda"), host.numel(), block)
    expected = host.reshape(-1, 2).clone()
    expected[1:] = (expected[1:] - expected[:-1]) * 0.5
    torch.testing.assert_close(actual, expected.flatten())


@pytest.mark.l1a_compile_no_target_dialect
@pytest.mark.parametrize("mismatch", ["shape", "element"])
def test_output_dtype_rejects_element_reshape(insert_point, monkeypatch, mismatch):
    monkeypatch.setenv("ARCH", "gfx942")
    from flydsl.extension.coop.block.adjacent_difference import _apply_if

    value = fx.Vector[fx.Int32, 2](1)
    output_t = fx.Vector[fx.Float32, 4] if mismatch == "shape" else fx.Float32
    with pytest.raises(TypeError, match="shape|element boundaries"):
        _apply_if(fx.Boolean(True), lambda a, b: a - b, value, value, output_t)


@fx.struct
class _DifferenceItems:
    data: fx.Vector[fx.Int32, 2]

    @property
    def dtype(self):
        return fx.Int32

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return self.data[index]

    @classmethod
    def from_elements(cls, items):
        return cls(fx.Vector.from_elements(items))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.usefixtures("coop_default_device")
def test_output_dtype_normalizes_changed_sov():
    from coop_test_utils import run_kernel

    host = torch.arange(128, dtype=torch.float32, device="cpu")

    def apply(a, out):
        tid = fx.thread_idx.x
        items = _DifferenceItems(fx.Vector.from_elements([a[2 * tid].to(fx.Int32), a[2 * tid + 1].to(fx.Int32)]))
        p = fx.coop.BlockAdjacentDifference[fx.Int32, 64]
        storage = fx.SharedAllocator().allocate(p.SharedStorage).peek()
        same = p.subtract_left(items, storage=storage)
        assert isinstance(same, _DifferenceItems)
        fx.barrier()
        changed = p.subtract_left(items, storage=storage, output_dtype=fx.Float32)
        assert isinstance(changed, fx.Vector) and changed.dtype is fx.Float32
        for i in range(2):
            out[2 * tid + i] = same[i].to(fx.Float32)
            out[128 + 2 * tid + i] = changed[i]

    actual = run_kernel(apply, host.to(device="cuda"), 256, 64).reshape(2, 128)
    expected = torch.ones(128, dtype=torch.float32, device="cpu")
    expected[0] = 0
    torch.testing.assert_close(actual, expected.expand(2, -1))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("universal", [False, True])
@pytest.mark.usefixtures("coop_default_device")
def test_output_dtype_docstring_example(universal):
    from coop_test_utils import Example, run_example

    rows = [[2, 5], [9, 10], [15, 21], [22, 30]] + [[30, 30]] * 60
    host = torch.tensor(rows, dtype=torch.int32, device="cpu").flatten()
    doubled = host - host.roll(1)
    doubled[0] = 2 * host[0]
    case = Example(
        module="block.adjacent_difference",
        symbol="BlockAdjacentDifference",
        inputs={"x": rows},
        checks=[("half = P.subtract_left(", "half * 2", doubled.reshape(64, 2).tolist())],
        block=64,
    )
    run_example(case, universal=universal)
