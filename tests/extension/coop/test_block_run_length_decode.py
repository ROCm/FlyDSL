# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Block run decoding from lengths and offsets, reusable state, and record fields."""

import pytest
from coop_common import BLOCK_THREADS, DTYPES
from coop_test_utils import ARCHES, dtype_entry, matrix_cases
from coop_test_utils import coop_default_device as coop_default_device

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
    "block,runs,count,offset,empty,universal",
    [
        (block, runs, count, offset, empty, universal)
        for (block, runs, count), offset, empty, universal in matrix_cases(
            [(64, 1, 3), (64, 3, 7), (64, 1, 4)], [0, 7, 10000], [False, True], [False, True]
        )
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_run_length_decode(block, runs, count, offset, empty, universal, entry=(fx.Int32, "int32")):
    checks.check_run_length_decode(block, runs, count, offset, empty, universal, entry)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "block,runs,count,from_offsets,offset_dtype",
    [
        (block, runs, count, from_offsets, offset_dtype)
        for (block, runs, count), from_offsets, offset_dtype in matrix_cases(
            [(64, 3, 7), (64, 9, 5), (128, 3, 9), (1024, 9, 1)], [False, True], [fx.Int32, fx.Int64, fx.Uint64]
        )
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_reusable_run_decode(block, runs, count, from_offsets, offset_dtype):
    values = (torch.arange(block * runs, device="cuda") * 11).int()
    lengths = ((torch.arange(block * runs, device="cuda") * 7) % 5).long()
    starts = lengths.cumsum(0) - lengths
    total = int(lengths.sum().item())
    inputs = starts if from_offsets else lengths
    window_offsets = (0, 7, total + 1)

    def apply(a, lens, out, relative):
        cls = fx.coop.universal.BlockRunLengthDecode[(fx.Int32, offset_dtype), block, runs]
        storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
        keys = fx.Vector.from_elements([a[fx.thread_idx.x * runs + i] for i in range(runs)])
        offsets = fx.Vector.from_elements([lens[fx.thread_idx.x * runs + i] for i in range(runs)])
        if from_offsets:
            state = cls.init_with_run_offsets(keys, offsets, storage=storage, total=total)
        else:
            state = cls.init_with_run_lengths(keys, offsets, storage=storage)
        for window, offset in enumerate(window_offsets):
            decoded, positions, _ = state.decode(decoded_items_per_thread=count, offset=offset, default=-9)
            for i in range(count):
                index = window * block * count + fx.thread_idx.x * count + i
                out[index] = decoded[i]
                relative[index] = positions[i].to(fx.Int64)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, lens: fx.Tensor, out: fx.Tensor, relative: fx.Tensor):
        apply(a, lens, out, relative)

    @flyc.jit
    def launch(a: fx.Tensor, lens: fx.Tensor, out: fx.Tensor, relative: fx.Tensor, stream: fx.Stream):
        kernel(a, lens, out, relative).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    out = torch.empty(3 * block * count, device="cuda", dtype=torch.int32)
    relative = torch.empty_like(out, dtype=torch.int64)
    launch(values, inputs, out, relative, torch.cuda.current_stream())
    torch.cuda.synchronize()
    expanded = values.cpu().repeat_interleave(lengths.cpu())
    rel = torch.cat([torch.arange(n, device="cpu") for n in lengths.cpu().tolist()])
    for window, offset in enumerate(window_offsets):
        size = block * count
        take = max(0, min(size, total - offset))
        expected = torch.full((size,), -9, dtype=torch.int32, device="cpu")
        expected_relative = torch.full((size,), -1, dtype=torch.int64, device="cpu")
        expected[:take], expected_relative[:take] = expanded[offset : offset + take], rel[offset : offset + take]
        torch.testing.assert_close(out.cpu().reshape(3, size)[window], expected)
        torch.testing.assert_close(relative.cpu().reshape(3, size)[window], expected_relative)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("from_offsets,bounded", matrix_cases([False, True], [False, True]))
@pytest.mark.usefixtures("coop_default_device")
def test_decode_offsets_beyond_32_bits(from_offsets, bounded):
    block, runs = 64, 2
    large = 1 << 34
    lengths = torch.tensor([large, 0, 5, 2, 0, 11] + [0] * (block * runs - 6), device="cuda", dtype=torch.int64)
    starts = lengths.cumsum(0) - lengths
    source = starts if from_offsets else lengths
    total = large + 18

    def apply(a, out):
        cls = fx.coop.universal.BlockRunLengthDecode[(fx.Int32, fx.Uint64), block, runs]
        storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
        keys = fx.Vector.from_elements([fx.thread_idx.x * runs + i for i in range(runs)])
        offsets = fx.Vector.from_elements([a[fx.thread_idx.x * runs + i] for i in range(runs)])
        if from_offsets:
            state = cls.init_with_run_offsets(keys, offsets, storage=storage, total=total if bounded else None)
        else:
            state = cls.init_with_run_lengths(keys, offsets, storage=storage)
        for j, start in enumerate((large - 1, large + 4)):
            decoded, relative, _ = state.decode(offset=start)
            out[j * block * 2 + fx.thread_idx.x] = decoded[0].to(fx.Int64)
            out[j * block * 2 + block + fx.thread_idx.x] = relative[0].to(fx.Int64)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor):
        apply(a, out)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor, stream: fx.Stream):
        kernel(a, out).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    out = torch.empty(block * 4, device="cuda", dtype=torch.int64)
    launch(source, out, torch.cuda.current_stream())
    torch.cuda.synchronize()
    expected = []
    host_starts = starts.cpu().tolist()
    for start in (large - 1, large + 4):
        decoded, relative = [], []
        for position in range(start, start + block):
            owner = max(i for i, begin in enumerate(host_starts) if begin <= position)
            valid = position < total or (from_offsets and not bounded)
            decoded.append(owner if valid else 0)
            relative.append(position - host_starts[owner] if valid else -1)
        expected.extend(decoded + relative)
    torch.testing.assert_close(out.cpu(), torch.tensor(expected, dtype=torch.int64, device="cpu"))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("entry,count,universal", matrix_cases(DTYPES, [1, 9], [False, True]))
@pytest.mark.usefixtures("coop_default_device")
def test_decode_dtypes(entry, count, universal):
    checks.check_run_length_decode(128, count, 9, 7, False, universal, entry=dtype_entry(entry))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("block,count", matrix_cases(BLOCK_THREADS, [1, 9]))
@pytest.mark.usefixtures("coop_default_device")
def test_decode_block_widths(block, count):
    checks.check_run_length_decode(block, count, 9, 7, False, False)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "block,count,method,universal",
    [
        (block, count, method, universal)
        for (block, count), method, universal in matrix_cases([(64, 1), (64, 3), (128, 3)], ["decode"], [False, True])
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_sequence_records(block, count, method, universal, compile_only=False):
    checks.check_sequence_records(block, count, method, universal, compile_only)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.parametrize("arch,method", matrix_cases(ARCHES, ["decode"]))
def test_record_sequence_compile(monkeypatch, arch, method):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    checks.check_sequence_records(192, 3, method, True, compile_only=True)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("form,case", matrix_cases(["list", "tuple", "vector"], ["decode_lengths", "decode_offsets"]))
@pytest.mark.usefixtures("coop_default_device")
def test_secondary_item_containers(form, case):
    checks.check_secondary_item_containers(form, case)
