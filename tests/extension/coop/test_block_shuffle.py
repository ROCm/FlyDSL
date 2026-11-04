# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Block shifts and rotations with tile boundaries, offsets, and record fields."""

import pytest
from coop_common import BLOCK_THREADS, DTYPES
from coop_test_utils import coop_default_device as coop_default_device
from coop_test_utils import matrix_cases, run_kernel, run_tile

import flydsl.expr as fx

try:
    import torch
except ImportError:
    torch = None
import coop_block_utils as checks


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("method,distance", matrix_cases(["offset", "rotate"], [-129, -1, 0, 3, 129]))
@pytest.mark.usefixtures("coop_default_device")
def test_thread_shuffle(method, distance):
    values = torch.arange(128, dtype=torch.int32, device="cuda")

    def transform(value):
        primitive = fx.coop.BlockShuffle[fx.Int32, (8, 4, 4)]
        storage = fx.SharedAllocator().allocate(primitive.SharedStorage).peek()
        result = getattr(primitive, method)(value[0], distance, storage=storage)
        return fx.Vector.from_elements([result])

    result = run_tile(transform, values, (8, 4, 4), 1)[0]
    host = values.cpu()
    indices = torch.arange(128, device="cpu") + distance
    expected = host[indices % 128]
    if method == "offset":
        expected = torch.where((indices >= 0) & (indices < 128), expected, host)
    torch.testing.assert_close(result, expected)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "block,count,boundary,method", matrix_cases([64, (8, 4, 4)], [1, 3], [None, -9], ["up", "down"])
)
@pytest.mark.usefixtures("coop_default_device")
def test_neighbors(block, count, boundary, method):
    checks.check_neighbors(block, count, boundary, method)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "block,count,up",
    [(block, count, up) for (block, count), up in matrix_cases([(64, 3), (64, 1), (128, 9)], [False, True])],
)
def test_shuffle_boundary_outputs(block, count, up):
    values = torch.arange(block * count, device="cuda", dtype=torch.int32)

    def transform(value):
        cls = fx.coop.universal.BlockShuffle[fx.Int32, block]
        storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
        if up:
            moved, boundary = cls.up(value, storage=storage, predecessor=-9, return_suffix=True)
        else:
            moved, boundary = cls.down(value, storage=storage, successor=-9, return_prefix=True)
        return moved, fx.Vector.from_elements([boundary] * count)

    actual = run_tile(transform, values, block, count, outputs=2)
    expected = values.cpu().roll(1 if up else -1)
    expected[0 if up else -1] = -9
    torch.testing.assert_close(actual[0], expected)
    torch.testing.assert_close(actual[1], torch.full_like(expected, block * count - 1 if up else 0))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("distance,block,rotate", matrix_cases([-1001, -1, 0, 1, 1001], [64, 128, 192], [False, True]))
@pytest.mark.usefixtures("coop_default_device")
def test_full_warp_thread_offset(block, distance, rotate):
    values = torch.arange(block, device="cuda", dtype=torch.int32)

    def transform(value):
        cls = fx.coop.universal.BlockShuffle[fx.Int32, block]
        storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
        result = (
            cls.rotate(value[0], distance, storage=storage)
            if rotate
            else cls.offset(value[0], distance, storage=storage, fill=-9)
        )
        return fx.Vector.from_elements([result])

    actual = run_tile(transform, values, block, 1)[0]
    source = torch.arange(block, device="cpu") + distance
    expected = (
        source.remainder(block).int() if rotate else torch.where((source >= 0) & (source < block), source, -9).int()
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("distance", [2**31 - 1, -(2**31)])
@pytest.mark.usefixtures("coop_default_device")
def test_rotate_int32_limit_distance(distance):
    # A runtime distance at the Int32 limits must not overflow before the
    # modulo; a non-power-of-two block exposes the wrapped sum.
    block = 192
    host = torch.arange(block + 1, device="cpu", dtype=torch.int32)
    host[block] = distance

    def apply(a, out):
        cls = fx.coop.BlockShuffle[fx.Int32, block]
        storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
        out[fx.thread_idx.x] = cls.rotate(a[fx.thread_idx.x], a[block], storage=storage)

    actual = run_kernel(apply, host.cuda(), block, block)
    expected = ((torch.arange(block, device="cpu") + distance) % block).int()
    torch.testing.assert_close(actual, expected)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "block,count,method,universal",
    [
        (block, count, method, universal)
        for (block, count), method, universal in matrix_cases(
            [(64, 1), (64, 3), (128, 3)], ["up", "down"], [False, True]
        )
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_sequence_records(block, count, method, universal, compile_only=False):
    checks.check_sequence_records(block, count, method, universal, compile_only)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "entry,count,case", matrix_cases(DTYPES, [1, 9], [("BlockShuffle", "up"), ("BlockShuffle", "down")])
)
@pytest.mark.usefixtures("coop_default_device")
def test_matrix_block_dtypes(entry, count, case):
    checks.check_block_case(entry, 128, count, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "block,count,case", matrix_cases(BLOCK_THREADS, [1, 9], [("BlockShuffle", "up"), ("BlockShuffle", "down")])
)
@pytest.mark.usefixtures("coop_default_device")
def test_matrix_block_shapes(block, count, case):
    checks.check_block_case((fx.Int32, "torch.int32"), block, count, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("case", [("BlockShuffle", "up"), ("BlockShuffle", "down")], ids=lambda c: c[1])
@pytest.mark.usefixtures("coop_default_device")
def test_matrix_three_dimensional_block(case):
    checks.check_block_case((fx.Float64, "torch.float64"), (8, 4, 4), 3, case)
