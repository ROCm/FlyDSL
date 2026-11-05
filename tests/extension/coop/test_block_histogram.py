# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Block histogram policies, accumulation, counters, dtypes, and block widths."""

import pytest
from coop_common import BLOCK_THREADS, DTYPES
from coop_test_utils import coop_default_device as coop_default_device
from coop_test_utils import dtype_entry, matrix_cases

import flydsl.compiler as flyc
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
    "block,bins,count,composite",
    [
        (block, bins, count, composite)
        for (block, bins, count), composite in matrix_cases(
            [(64, 1, 3), (64, 3, 1), (64, 137, 3), (128, 17, 4)], [False, True]
        )
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_histogram(block, bins, count, composite, entry=(fx.Int32, "int32"), universal=False):
    checks.check_histogram(block, bins, count, composite, entry, universal)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "block,count,bins,algorithm,counter",
    [
        (block, count, bins, algorithm, counter)
        for (block, count, bins), algorithm, counter in matrix_cases(
            [(64, 3, 7), (64, 9, 17), (128, 3, 137), (128, 9, 1)],
            ["ATOMIC", "SORT"],
            [fx.Int32, fx.Int64],
            independent=True,
        )
    ],
)
def test_histogram_policies(block, count, bins, algorithm, counter):
    values = (((torch.arange(block * count, device="cuda") * 37) % (bins + 2)) - 1).int()

    def apply(a, out):
        cls = fx.coop.universal.BlockHistogram[
            (fx.Int32, counter), block, bins, getattr(fx.coop.BlockHistogramAlgorithm, algorithm)
        ]
        storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
        items = fx.Vector.from_elements([a[fx.thread_idx.x * count + i] for i in range(count)])
        cls(items, storage=storage)
        counts = cls.composite(items, storage=storage)
        return counts

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor):
        counts = apply(a, out)
        for i in fx.range_constexpr((bins + block - 1) // block):
            index = fx.thread_idx.x + i * block
            if index < bins:
                out[index] = counts[index]

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor, stream: fx.Stream):
        kernel(a, out).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    out = torch.empty(bins, device="cuda", dtype=torch.int64 if counter is fx.Int64 else torch.int32)
    launch(values, out, torch.cuda.current_stream())
    torch.cuda.synchronize()
    host = values.cpu().long()
    expected = host[(host >= 0) & (host < bins)].bincount(minlength=bins).to(out.dtype) * 2
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "entry,count,universal", matrix_cases([e for e in DTYPES if not e[0].is_float], [1, 9], [False, True])
)
@pytest.mark.usefixtures("coop_default_device")
def test_histogram_dtypes(entry, count, universal):
    checks.check_histogram(128, 137, count, True, entry=dtype_entry(entry), universal=universal)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("block,count", matrix_cases(BLOCK_THREADS, [1, 9]))
@pytest.mark.usefixtures("coop_default_device")
def test_histogram_block_widths(block, count):
    checks.check_histogram(block, 137, count, False)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize(
    "form,algorithm", matrix_cases(["list", "tuple", "vector"], list(fx.coop.BlockHistogramAlgorithm))
)
@pytest.mark.usefixtures("coop_default_device")
def test_item_containers(form, algorithm):
    checks.check_secondary_item_containers(form, "histogram", algorithm)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "entry,algorithm",
    matrix_cases(
        [(fx.Int64, "int64"), (fx.Float32, "float32"), (fx.Float16, "float16")], list(fx.coop.BlockHistogramAlgorithm)
    ),
)
@pytest.mark.usefixtures("coop_default_device")
def test_out_of_range_keys_skip_bins(entry, algorithm):
    checks.check_histogram_key_conversion(entry, algorithm)
