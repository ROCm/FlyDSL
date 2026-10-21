# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Block load/store policies, guarded tiles, conversions, and record fields."""

import pytest
from coop_common import BLOCK_THREADS, DTYPES
from coop_test_utils import ARCHES, dtype_entry, matrix_cases, run_kernel
from coop_test_utils import coop_default_device as coop_default_device

import flydsl.expr as fx
from flydsl.extension.coop.block.load import BlockLoad, BlockLoadAlgorithm
from flydsl.extension.coop.block.store import BlockStore, BlockStoreAlgorithm

try:
    import torch
except ImportError:
    torch = None
import coop_block_utils as checks


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("policy,valid", matrix_cases(["DIRECT", "STRIPED", "TRANSPOSE"], [None, 0, 1, 95, 96]))
@pytest.mark.usefixtures("coop_default_device")
def test_guarded_io(
    policy, valid, block=64, count=3, entry=(fx.Int32, "int32"), universal=False, width=8, dynamic=False
):
    checks.check_guarded_io(policy, valid, block, count, entry, universal, width, dynamic)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "dtype,name,valid",
    [
        (dtype, name, valid)
        for (dtype, name), valid in matrix_cases(
            [
                (fx.Uint8, "uint8"),
                (fx.Int16, "int16"),
                (fx.Int64, "int64"),
                (fx.Float16, "float16"),
                (fx.BFloat16, "bfloat16"),
                (fx.Float32, "float32"),
                (fx.Float64, "float64"),
            ],
            [None, 127],
        )
    ],
)
def test_packed_io_dtypes(dtype, name, valid):
    block, count = 64, 8
    size, offset = block * count, 5
    extent = size if valid is None else valid
    source = (torch.arange(offset + extent, dtype=torch.int32, device="cuda") % 83).to(getattr(torch, name))

    def apply(a, out):
        loader = BlockLoad[dtype, block, count, BlockLoadAlgorithm.VECTORIZE]
        storer = BlockStore[dtype, block, count, BlockStoreAlgorithm.VECTORIZE]
        items = loader(a, offset=offset, valid_items=valid)
        storer(out, items, offset=offset, valid_items=valid)

    actual = run_kernel(apply, source, size + offset + 3, block)
    expected = torch.full_like(actual, -99)
    expected[offset : offset + extent] = source.cpu()[offset:]
    torch.testing.assert_close(actual, expected)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_vectorize_strided_input_fallback():
    block, count = 64, 4
    source = torch.arange(block * count * 2, dtype=torch.int32, device="cuda")[::2]

    def apply(a, out):
        loader = BlockLoad[fx.Int32, block, count, BlockLoadAlgorithm.VECTORIZE]
        storer = BlockStore[fx.Int32, block, count, BlockStoreAlgorithm.VECTORIZE]
        storer(out, loader(a))

    actual = run_kernel(apply, source, block * count, block)
    torch.testing.assert_close(actual, source.cpu())


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "warp,policy,count,valid",
    [
        (warp, policy, count, valid)
        for (warp, policy), count, valid in matrix_cases(
            [(False, p.name) for p in BlockLoadAlgorithm], [3, 4], [0, 1, 95]
        )
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_movement_io_policies(warp, policy, count, valid):
    checks.check_movement_io_policies(warp, policy, count, valid)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "entry,count,policy,universal", matrix_cases(DTYPES, [1, 9], ["DIRECT", "STRIPED", "TRANSPOSE"], [False, True])
)
@pytest.mark.usefixtures("coop_default_device")
def test_io_dtypes(entry, count, policy, universal):
    checks.check_guarded_io(
        policy, 128 * count - 1, block=128, count=count, entry=dtype_entry(entry), universal=universal
    )


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("block,count,policy", matrix_cases(BLOCK_THREADS, [1, 9], ["DIRECT", "STRIPED", "TRANSPOSE"]))
@pytest.mark.usefixtures("coop_default_device")
def test_io_block_widths(block, count, policy):
    checks.check_guarded_io(policy, block * count - 1, block=block, count=count)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "case,policy,count",
    [
        (case, policy, count)
        for (case, policy), count in matrix_cases([("block_io", policy.name) for policy in BlockLoadAlgorithm], [3, 4])
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_record_movement(case, policy, count):
    checks.check_record_movement(case, policy, count)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize(
    "arch,case,policy",
    [
        (arch, case, policy)
        for arch, (case, policy) in matrix_cases(ARCHES, [("block_io", policy.name) for policy in BlockLoadAlgorithm])
    ],
)
def test_compile_record_movement(monkeypatch, arch, case, policy):
    checks.check_compile_record_movement(monkeypatch, arch, case, policy)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_vectorize_non_tensor_iterator_fallback():
    block, count = 64, 4
    values = torch.arange(block * count, device="cuda", dtype=torch.int32)

    def apply(a, out):
        @fx.struct
        class Column:
            data: fx.Tensor

            @property
            def dtype(self):
                return fx.Int32

            def __getitem__(self, index):
                return self.data[index]

            def __setitem__(self, index, value):
                self.data[index] = value

        source = Column(a)
        destination = Column(out)
        loader = BlockLoad[fx.Int32, block, count, BlockLoadAlgorithm.VECTORIZE]
        storer = BlockStore[fx.Int32, block, count, BlockStoreAlgorithm.VECTORIZE]
        storer(destination, loader(source))

    torch.testing.assert_close(run_kernel(apply, values, block * count, block), values.cpu())


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize(
    "arch,case",
    matrix_cases(ARCHES, ["block_io_VECTORIZE", "block_io_WARP_TRANSPOSE", "block_io_WARP_TRANSPOSE_TIMESLICED"]),
)
def test_compile_movement_policies(monkeypatch, arch, case):
    checks.check_compile_movement_policies(monkeypatch, arch, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "warp,policy,count,source_dtype,source_name,target_dtype,target_name,convert_load,partial",
    [
        (warp, policy, count, source_dtype, source_name, target_dtype, target_name, convert_load, partial)
        for (warp, policy, count), (
            source_dtype,
            source_name,
            target_dtype,
            target_name,
        ), convert_load, partial in matrix_cases(
            [(False, policy.name, 4) for policy in BlockLoadAlgorithm] + [(False, "VECTORIZE", 3)],
            [
                (fx.Int32, "int32", fx.Int64, "int64"),
                (fx.Float32, "float32", fx.Float64, "float64"),
                (fx.Int64, "int64", fx.Int16, "int16"),
            ],
            [False, True],
            ["one", "tail"],
        )
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_guarded_cross_dtype_io(
    warp, policy, count, source_dtype, source_name, target_dtype, target_name, convert_load, partial
):
    checks.check_guarded_cross_dtype_io(
        warp, policy, count, source_dtype, source_name, target_dtype, target_name, convert_load, partial
    )
