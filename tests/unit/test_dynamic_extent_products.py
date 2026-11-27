# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shape extents known only at run time whose products reach 2^31, executed on the device.

The kernels take the extents as Int32 arguments and write sizes and offsets, not elements, so no
tensor of that size is allocated.
"""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

try:
    import torch
except ImportError:
    torch = None

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

LANES = 64


@flyc.kernel(known_block_size=[1, 1, 1])
def _sizes_kernel(out: fx.Tensor, m: fx.Int32, n: fx.Int32):
    for i, shape in enumerate(((m, n), (m, 8192, n), (8192, m, n), (m, n, 8192), (m, 8192))):
        out[i] = fx.Int64(fx.get_scalar(fx.size(fx.make_int_tuple(shape))))


@flyc.jit
def _sizes(out: fx.Tensor, m: fx.Int32, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _sizes_kernel(out, m, n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


def _make_offsets(tile, tail):
    """Lane l of tile b of a row-major (m, n, k, *tail) layout divided by tile:1, as an offset."""

    @flyc.kernel(known_block_size=[LANES, 1, 1])
    def kernel(out: fx.Tensor, base: fx.Int32, m: fx.Int32, n: fx.Int32, k: fx.Int32):
        shape = (m, n, k) + tail
        strides = [1]
        for extent in reversed(shape[1:]):
            strides.insert(0, strides[0] * (extent if isinstance(extent, int) else fx.Int64(extent)))
        divided = fx.logical_divide(fx.make_layout(shape, tuple(strides)), fx.make_layout(tile, 1))
        lane = fx.Int32(fx.thread_idx.x)
        offset = fx.crd2idx((lane % tile, base + lane // tile), divided)
        out[fx.thread_idx.x] = fx.Int64(fx.get_scalar(offset))

    @flyc.jit
    def launch(
        out: fx.Tensor, base: fx.Int32, m: fx.Int32, n: fx.Int32, k: fx.Int32, stream: fx.Stream = fx.Stream(None)
    ):
        kernel(out, base, m, n, k).launch(grid=(1, 1, 1), block=(LANES, 1, 1), stream=stream)

    return launch


@flyc.kernel(known_block_size=[LANES, 1, 1])
def _split_kernel(out: fx.Tensor, base: fx.Int32, a: fx.Int64, b: fx.Int32):
    layout = fx.make_layout(((a, b), 4), ((1, 4), 64))
    index = base + fx.Int32(fx.thread_idx.x)
    out[fx.thread_idx.x] = fx.Int64(fx.get_scalar(fx.crd2idx(index, layout)))


@flyc.jit
def _split(out: fx.Tensor, base: fx.Int32, a: fx.Int64, b: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _split_kernel(out, base, a, b).launch(grid=(1, 1, 1), block=(LANES, 1, 1), stream=stream)


@flyc.kernel(known_block_size=[LANES, 1, 1])
def _wide_divide_kernel(out: fx.Tensor, base: fx.Int32, a: fx.Int64):
    divided = fx.logical_divide(fx.make_layout((a, 32), (32, 1)), fx.make_layout(4, 1))
    lane = fx.Int32(fx.thread_idx.x)
    out[fx.thread_idx.x] = fx.Int64(fx.get_scalar(fx.crd2idx((lane % 4, base + lane // 4), divided)))


@flyc.jit
def _wide_divide(out: fx.Tensor, base: fx.Int32, a: fx.Int64, stream: fx.Stream = fx.Stream(None)):
    _wide_divide_kernel(out, base, a).launch(grid=(1, 1, 1), block=(LANES, 1, 1), stream=stream)


@flyc.kernel(known_block_size=[1, 1, 1])
def _static_index_kernel(out: fx.Tensor, a: fx.Int64, s: fx.Int32):
    layout = fx.make_layout(((16, a), 4), ((s, 1), 1))
    out[0] = fx.Int64(fx.get_scalar(fx.crd2idx(5, layout)))
    out[1] = fx.Int64(fx.get_scalar(fx.crd2idx(37, layout)))


@flyc.jit
def _static_index(out: fx.Tensor, a: fx.Int64, s: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _static_index_kernel(out, a, s).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


@flyc.kernel(known_block_size=[LANES, 1, 1])
def _unit_extent_kernel(out: fx.Tensor, m: fx.Int32, n: fx.Int32, s: fx.Int32, w: fx.Int64):
    layout = fx.make_layout((4, (m, n, 1), 2), (s, (s, 1, w), 1))
    index = fx.Int32(fx.thread_idx.x)
    out[fx.thread_idx.x] = fx.Int64(fx.get_scalar(fx.crd2idx(index, layout)))


@flyc.jit
def _unit_extent(
    out: fx.Tensor, m: fx.Int32, n: fx.Int32, s: fx.Int32, w: fx.Int64, stream: fx.Stream = fx.Stream(None)
):
    _unit_extent_kernel(out, m, n, s, w).launch(grid=(1, 1, 1), block=(LANES, 1, 1), stream=stream)


def _row_major_offset(index, shape):
    """Where the element of colexicographic index `index` is in a row-major layout of `shape`."""
    offset = 0
    stride = 1
    digits = []
    for extent in shape:
        digits.append(index % extent)
        index //= extent
    for digit, extent in zip(reversed(digits), reversed(shape)):
        offset += digit * stride
        stride *= extent
    return offset


def test_sizes_of_dynamic_extents_do_not_wrap():
    """2^18 * 8192 * 2 is 2^32 wherever the static extent stands; a single dynamic extent stays i32."""
    out = torch.zeros(5, dtype=torch.int64, device="cuda")
    _sizes(out, 1 << 18, 2)
    torch.cuda.synchronize()
    assert out.tolist()[:4] == [1 << 19, 1 << 32, 1 << 32, 1 << 32]
    _sizes(out, 3, 5)
    torch.cuda.synchronize()
    assert out.tolist() == [15, 15 * 8192, 15 * 8192, 15 * 8192, 3 * 8192]


@pytest.mark.parametrize(
    "tile, shape, tail",
    [
        (1, (4, 1 << 29, 1), ()),
        (1, (65536, 32768, 1), ()),
        (2, (65536, 65536, 1), ()),
        (64, (65536, 32768, 1), ()),
        (1, (256, 256, 32768), ()),
        (64, (256, 256, 32768), ()),
        (1, (65536, 65536, 4), ()),
        (1, (65536, 65536, 2), (32, 128)),
        (64, (65536, 65536, 2), (32, 128)),
        (4, (12, 7, 5), ()),
        (1, (4, 3, 2), (8, 16)),
        (64, (128, 4, 2), (8, 16)),
    ],
)
def test_divide_of_dynamic_extents_addresses_every_element(tile, shape, tail):
    """Each lane of the last tiles reachable by an Int32 index, and of the first, has its own offset."""
    launch = _make_offsets(tile, tail)
    full = shape + tail
    elements = 1
    for extent in full:
        elements *= extent
    tiles = min(elements // tile, 1 << 31)
    out = torch.zeros(LANES, dtype=torch.int64, device="cuda")
    for base in (0, tiles - LANES // tile):
        launch(out, base, *shape)
        torch.cuda.synchronize()
        want = [_row_major_offset(tile * (base + lane // tile) + lane % tile, full) for lane in range(LANES)]
        assert out.tolist() == want, (tile, shape, tail, base)


@pytest.mark.parametrize("a", [3, (1 << 31) - 1, 1 << 31, 1 << 33])
def test_index_through_an_i64_extent(a):
    """An Int32 index through the modes ((a, b), 4) with an Int64 a has its offset up to the largest index.

    The offset is Int64: with a = 3 and the static stride 64 it passes 2^31.
    """
    b = 5
    out = torch.zeros(LANES, dtype=torch.int64, device="cuda")
    for base in (0, (1 << 31) - LANES):
        _split(out, base, a, b)
        torch.cuda.synchronize()
        want = [i % a + 4 * (i // a % b) + 64 * (i // a // b) for i in range(base, base + LANES)]
        assert out.tolist() == want, (a, base)


def test_divide_of_an_i64_extent_with_static_strides():
    """A row-major (2^29, 32) layout with an Int64 extent by 4:1: the offsets of the last tiles pass 2^34."""
    a = 1 << 29
    out = torch.zeros(LANES, dtype=torch.int64, device="cuda")
    for base in (0, (1 << 31) - LANES // 4):
        _wide_divide(out, base, a)
        torch.cuda.synchronize()
        want = [_row_major_offset(4 * (base + lane // 4) + lane % 4, (a, 32)) for lane in range(LANES)]
        assert out.tolist() == want, base


def test_static_index_through_an_i64_extent():
    """A static index through ((16, a), 4) with an Int64 a: its product with an Int32 stride of 2^30 passes 2^31."""
    a = 3
    out = torch.zeros(2, dtype=torch.int64, device="cuda")
    for s in (7, 1 << 30):
        _static_index(out, a, s)
        torch.cuda.synchronize()
        assert out.tolist() == [5 * s, 5 * s + 2], s


def test_index_through_a_mode_that_ends_in_one():
    """The Int64 stride of the extent of 1 that ends (m, n, 1) makes the offset Int64: its terms pass 2^31."""
    m, n, s = 3, 5, 1 << 29
    out = torch.zeros(LANES, dtype=torch.int64, device="cuda")
    _unit_extent(out, m, n, s, 7)
    torch.cuda.synchronize()
    want = [s * (i % 4) + s * (i // 4 % m) + (i // 4 // m % n) + (i // 4 // m // n) for i in range(LANES)]
    assert max(want) >= 1 << 31
    assert out.tolist() == want
