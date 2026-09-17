# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Nested numeric records retain fields through shuffle, selection and LDS."""

import pytest
from coop_test_utils import ARCHES, run_kernel
from coop_test_utils import warp_default_device as warp_default_device
from coop_warp_record_movement_utils import PAYLOAD, RECORD

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.extension.coop._values import (
    _as_items,
    _flatten_record,
    _from_items,
    _item_dtype,
    _leaf_fields,
    _rebuild_record,
    _record_default,
    _record_select,
    _record_shuffle,
    _shared_array,
    _shared_load,
    _shared_store,
)

try:
    import torch
except ImportError:
    torch = None


@flyc.jit
def _roundtrip(value, slots, count, block):
    for i in fx.range_constexpr(count):
        _shared_store(slots, fx.thread_idx.x * count + i, value[i])
    fx.barrier()
    return _read_records(slots, count, block)


def _read_records(slots, count, block):
    return _from_items([_shared_load(slots, block * count - 1 - (fx.thread_idx.x * count + i)) for i in range(count)])


def _apply_records(a, out, count, block, mode):
    items = _from_items(
        [
            RECORD(
                a[fx.thread_idx.x * count + i].to(fx.Int32),
                PAYLOAD(
                    a[fx.thread_idx.x * count + i].to(fx.Float32) + 0.375,
                    a[fx.thread_idx.x * count + i].to(fx.Int64) * (1 << 35) + 17,
                ),
            )
            for i in range(count)
        ]
    )
    assert isinstance(items, tuple)
    assert _item_dtype(_as_items(items)[0]) is RECORD
    assert len(_leaf_fields(RECORD)) == 3
    items = _from_items([_rebuild_record(RECORD, _flatten_record(item)) for item in items])
    if mode == "shared":
        schema = fx.Struct["slots" : _shared_array(RECORD, block * count)]
        storage = fx.SharedAllocator().allocate(schema).peek()
        result = _roundtrip(items, storage.slots, count, block)
    else:
        result = _record_shuffle(items, (fx.lane_id() // 8) * 8 + (7 - fx.lane_id() % 8), 8)
    defaults = _from_items([_record_default(RECORD, -7) for _ in range(count)])
    result = _record_select(fx.thread_idx.x % 2 == 0, result, defaults)
    for i, item in enumerate(result):
        index = fx.thread_idx.x * count + i
        out[index] = item.key.to(fx.Float64)
        out[block * count + index] = item.payload.fraction.to(fx.Float64)
        out[2 * block * count + index] = item.payload.tag.to(fx.Float64)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("mode,block", [("shared", 64), ("shuffle", 64)])
@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.usefixtures("warp_default_device")
def test_record_helper_gpu(mode, block, count):
    size = block * count
    values = torch.arange(size, dtype=torch.float64, device="cuda")
    result = run_kernel(lambda a, out: _apply_records(a, out, count, block, mode), values, size * 3, block)
    indices = torch.arange(size, device="cpu")
    source = (
        size - 1 - indices
        if mode == "shared"
        else (indices // (8 * count)) * (8 * count) + (7 - (indices // count) % 8) * count + indices % count
    )
    mask = (indices // count) % 2 == 0
    expected = torch.cat(
        [
            torch.where(mask, source.double(), -7),
            torch.where(mask, source.double() + 0.375, -7),
            torch.where(mask, source.double() * (1 << 35) + 17, -7),
        ]
    )
    torch.testing.assert_close(result, expected)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch")
@pytest.mark.parametrize("arch", ARCHES)
def test_records_compile(monkeypatch, arch):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    block, count = 64, 3

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor):
        _apply_records(a, out, count, block, "shared")
        _apply_records(a, out, count, block, "shuffle")

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor):
        kernel(a, out).launch(grid=(1, 1, 1), block=(block, 1, 1))

    launch(
        torch.empty(block * count, dtype=torch.float64, device="cpu"),
        torch.empty(block * count * 3, dtype=torch.float64, device="cpu"),
    )


@flyc.jit
def _identity_record(value):
    return value


def _apply_pointer_records(a, out, block, mode):
    allocator = fx.SharedAllocator()
    shared = allocator.allocate(fx.Array[fx.Int32, block]).peek()
    shared[fx.thread_idx.x] = 100 + fx.thread_idx.x * 7
    fx.barrier()
    global_ptr = a.iter + fx.thread_idx.x
    shared_ptr = shared.ptr + fx.thread_idx.x
    schema = fx.Struct[
        "key" : fx.Int32,
        "global_ptr" : fx.Pointer[fx.Int32, fx.AddressSpace.Global],
        "shared_ptr" : fx.Pointer[fx.Int32, fx.AddressSpace.Shared],
    ]
    value = schema(fx.thread_idx.x, global_ptr, shared_ptr)
    value = _identity_record(value)
    if mode == "shuffle":
        result = _record_shuffle(value, (fx.lane_id() // 8) * 8 + 7 - fx.lane_id() % 8, 8)
    else:
        slots = allocator.allocate(_shared_array(schema, block)).peek()
        result = _roundtrip((value,), slots, 1, block)[0]
    out[fx.thread_idx.x] = result.key
    out[block + fx.thread_idx.x] = result.global_ptr.load()
    out[2 * block + fx.thread_idx.x] = result.shared_ptr.load()


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("mode,block", [("shared", 64), ("shuffle", 64)])
@pytest.mark.usefixtures("warp_default_device")
def test_pointer_fields_gpu(mode, block):
    values = torch.arange(block, device="cuda", dtype=torch.int32) * 3 + 10
    actual = run_kernel(lambda a, out: _apply_pointer_records(a, out, block, mode), values, 3 * block, block)
    indices = torch.arange(block, device="cpu")
    source = block - 1 - indices if mode == "shared" else (indices // 8) * 8 + 7 - indices % 8
    expected = torch.cat([source.int(), values.cpu()[source], (100 + source * 7).int()])
    torch.testing.assert_close(actual, expected)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch")
@pytest.mark.parametrize("arch", ARCHES)
def test_pointer_fields_compile(monkeypatch, arch):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    block = 64

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor):
        _apply_pointer_records(a, out, block, "shared")

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor):
        kernel(a, out).launch(grid=(1, 1, 1), block=(block, 1, 1))

    launch(torch.empty(block, dtype=torch.int32, device="cpu"), torch.empty(block * 3, dtype=torch.int32, device="cpu"))


def _apply_bool_wide(a, out, block, mode):
    schema = fx.Struct["flag" : fx.Boolean, "wide" : fx.Int128]
    key = fx.thread_idx.x + 1
    value = schema(key % 3 == 0, (key.to(fx.Int128) << 100) | (key * 17).to(fx.Int128))
    allocator = fx.SharedAllocator()
    flags = allocator.allocate(_shared_array(fx.Boolean, block)).peek()
    _shared_store(flags, fx.thread_idx.x, value.flag)
    fx.barrier()
    out[3 * block + fx.thread_idx.x] = _shared_load(flags, block - 1 - fx.thread_idx.x).to(fx.Int64)
    if mode == "shared":
        slots = allocator.allocate(_shared_array(schema, block)).peek()
        result = _roundtrip(_from_items([value]), slots, 1, block)[0]
    else:
        result = _record_shuffle(value, (fx.lane_id() // 8) * 8 + 7 - fx.lane_id() % 8, 8)
    out[fx.thread_idx.x] = result.flag.to(fx.Int64)
    out[block + fx.thread_idx.x] = (result.wide >> 100).to(fx.Int64)
    out[2 * block + fx.thread_idx.x] = (result.wide & ((1 << 64) - 1)).to(fx.Int64)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("mode,block", [("shared", 64), ("shuffle", 64)])
@pytest.mark.usefixtures("warp_default_device")
def test_boolean_storage_and_wide_record_shuffle(mode, block):
    values = torch.empty(block, dtype=torch.int64, device="cuda")
    actual = run_kernel(lambda a, out: _apply_bool_wide(a, out, block, mode), values, 4 * block, block)
    indices = torch.arange(block, device="cpu")
    source = block - 1 - indices if mode == "shared" else (indices // 8) * 8 + 7 - indices % 8
    key = source + 1
    expected = torch.cat([(key % 3 == 0).long(), key, key * 17, ((block - indices) % 3 == 0).long()])
    torch.testing.assert_close(actual, expected)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch")
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("mode", ["shared", "shuffle"])
def test_boolean_wide_records_compile(monkeypatch, arch, mode):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    block = 64

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor):
        _apply_bool_wide(a, out, block, mode)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor):
        kernel(a, out).launch(grid=(1, 1, 1), block=(block, 1, 1))

    launch(torch.empty(block, dtype=torch.int64, device="cpu"), torch.empty(block * 4, dtype=torch.int64, device="cpu"))


@flyc.jit
def _carry_array(value):
    return value


def _tuple_item_at(items, index):
    """Select a tuple field with a runtime index when checking moved records."""
    result = items[0]
    for i, value in enumerate(items[1:], 1):
        result = _record_select(index == i, value, result)
    return result


def _apply_native_fields(a, out, block, mode):
    allocator = fx.SharedAllocator()
    shared = allocator.allocate(fx.Array[fx.Int64, block * 2]).peek()
    for i in range(2):
        shared[fx.thread_idx.x * 2 + i] = (1000 + fx.thread_idx.x * 10 + i).to(fx.Int64)
    fx.barrier()
    indices = [fx.thread_idx.x * 2 + i for i in range(2)]
    global_ptrs = [a.iter + index for index in indices]
    shared_ptrs = [shared.ptr + index for index in indices]
    numbers_type = fx.Vector[fx.Int64, 3]
    nested_type = fx.Struct["key" : fx.Int32, "flag" : fx.Boolean]
    nested_pair = fx.Struct["first":nested_type, "second":nested_type]
    global_pair = fx.Struct[
        "first" : fx.Pointer[fx.Int64, fx.AddressSpace.Global], "second" : fx.Pointer[fx.Int64, fx.AddressSpace.Global]
    ]
    shared_pair = fx.Struct[
        "first" : fx.Pointer[fx.Int64, fx.AddressSpace.Shared], "second" : fx.Pointer[fx.Int64, fx.AddressSpace.Shared]
    ]
    schema = fx.Struct[
        "numbers":numbers_type, "nested":nested_pair, "global_ptrs":global_pair, "shared_ptrs":shared_pair
    ]
    numbers = numbers_type([fx.thread_idx.x.to(fx.Int64) * 100 + i for i in range(3)])
    nested = nested_pair(*[nested_type(fx.thread_idx.x * 7 + i, (fx.thread_idx.x + i) % 2 == 0) for i in range(2)])
    value = schema(numbers, nested, global_pair(*global_ptrs), shared_pair(*shared_ptrs))
    value = _carry_array(value)
    assert len(value.numbers) == 3
    if mode == "shared":
        slots = allocator.allocate(_shared_array(schema, block)).peek()
        result = _roundtrip((value,), slots, 1, block)[0]
    else:
        result = _record_shuffle(value, (fx.lane_id() // 8) * 8 + 7 - fx.lane_id() % 8, 8)
    index = fx.thread_idx.x
    out[index] = result.numbers[index % 3]
    out[block + index] = result.numbers[2]
    out[2 * block + index] = _tuple_item_at((result.nested.first, result.nested.second), index % 2).key.to(fx.Int64)
    out[3 * block + index] = _tuple_item_at((result.nested.first, result.nested.second), index % 2).flag.to(fx.Int64)
    out[4 * block + index] = _tuple_item_at((result.global_ptrs.first, result.global_ptrs.second), index % 2).load()
    out[5 * block + index] = _tuple_item_at((result.shared_ptrs.first, result.shared_ptrs.second), index % 2).load()
    total = fx.Int64(0)
    for element in result.numbers:
        total = total + element
    out[6 * block + index] = total


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("mode,block", [("shared", 64), ("shuffle", 64)])
@pytest.mark.parametrize("default_device", ("cpu", "cuda"))
def test_native_fields_gpu(mode, block, default_device):
    with torch.device(default_device):
        values = torch.arange(block * 2, device="cuda", dtype=torch.int64) * 11 + 5
        actual = run_kernel(lambda a, out: _apply_native_fields(a, out, block, mode), values, block * 7, block)
        indices = torch.arange(block, device="cpu")
        source = block - 1 - indices if mode == "shared" else (indices // 8) * 8 + 7 - indices % 8
        selected = indices % 2
        expected = torch.cat(
            [
                source * 100 + indices % 3,
                source * 100 + 2,
                source * 7 + selected,
                ((source + selected) % 2 == 0).long(),
                values.cpu()[source * 2 + selected],
                1000 + source * 10 + selected,
                source * 300 + 3,
            ]
        )
        torch.testing.assert_close(actual, expected)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch")
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("mode", ["shared", "shuffle"])
def test_native_fields_compile(monkeypatch, arch, mode):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    block = 64

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor):
        _apply_native_fields(a, out, block, mode)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor):
        kernel(a, out).launch(grid=(1, 1, 1), block=(block, 1, 1))

    launch(
        torch.empty(block * 2, dtype=torch.int64, device="cpu"), torch.empty(block * 7, dtype=torch.int64, device="cpu")
    )


@pytest.mark.l1a_compile_no_target_dialect
def test_only_native_value_types_are_exported():
    import flydsl.extension.coop as coop

    removed = ("RecordVector", "RecordArray", "RecordView", "PointerField", "FixedArray")
    for name in removed:
        assert not hasattr(coop, name)
        assert not hasattr(coop.universal, name)
    assert isinstance(_from_items([fx.Struct["x" : fx.Int32](1)]), tuple)


def _apply_native_payloads(a, out, mode):
    tid = fx.thread_idx.x
    vectors = fx.Vector[fx.Uint32, (2, 2)]
    payload_type = fx.Struct[
        "tag" : fx.Constexpr[int], "data":vectors, "ptr" : fx.Pointer[fx.Int64, fx.AddressSpace.Global]
    ]
    indices = [(tid // 8) * 16 + tid % 8 + i * 8 if mode == "bitonic" else tid * 2 + i for i in range(2)]
    keys = fx.Vector.from_elements([127 - index for index in indices])
    payloads = tuple(
        payload_type(
            7, vectors([fx.Uint32(0x80000000) + index.to(fx.Uint32) * 4 + j for j in range(4)]), a.iter + index
        )
        for index in indices
    )
    payloads = _carry_array(payloads)
    if mode == "warp_merge":
        _, results = fx.coop.warp_merge_sort(keys, payloads, width=8, compare_op=lambda a, b: a < b)
    else:
        _, results = fx.coop.warp_bitonic_sort(keys, payloads, width=8, compare_op=lambda a, b: a < b)
    assert isinstance(results, tuple)
    for i, result in enumerate(results):
        assert result.tag == 7
        assert result.data.dtype is fx.Uint32 and result.data.shape == (2, 2)
        out[tid * 2 + i] = result.data[tid % 4].to(fx.Int64)
        out[128 + tid * 2 + i] = result.ptr.load()


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("mode", ("warp_merge", "bitonic"))
@pytest.mark.usefixtures("warp_default_device")
def test_native_constexpr_vector_payloads_gpu(mode):
    values = torch.arange(128, device="cuda", dtype=torch.int64) * 11 + 3
    actual = run_kernel(lambda a, out: _apply_native_payloads(a, out, mode), values, 256, 64)
    indices = torch.arange(128, device="cpu")
    source = 127 - indices
    if mode == "warp_merge":
        source = (indices // 16) * 16 + 15 - indices % 16
    elif mode == "bitonic":
        source = (indices // 16) * 16 + 15 - ((indices // 2) % 8 + (indices % 2) * 8)
    expected = torch.cat([0x80000000 + source * 4 + (indices // 2) % 4, values.cpu()[source]])
    torch.testing.assert_close(actual, expected)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch")
@pytest.mark.parametrize("arch", ("gfx942", "gfx1100"))
@pytest.mark.parametrize("mode", ("warp_merge", "bitonic"))
def test_native_constexpr_vector_payloads_compile(monkeypatch, arch, mode):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kernel(a: fx.Tensor, out: fx.Tensor):
        _apply_native_payloads(a, out, mode)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor):
        kernel(a, out).launch(grid=(1, 1, 1), block=(64, 1, 1))

    launch(torch.empty(128, dtype=torch.int64, device="cpu"), torch.empty(256, dtype=torch.int64, device="cpu"))


@pytest.mark.l1a_compile_no_target_dialect
def test_collective_conversion_preserves_constexpr_specialization():
    from flydsl.extension.coop._values import _as_items, _record_cast, _record_select

    schema = fx.Struct["tag" : fx.Constexpr[int], "key" : fx.Int32]
    first, other = schema(7, 1), schema(8, 2)
    with pytest.raises(TypeError, match="same dtype"):
        _as_items((first, other))
    with pytest.raises(TypeError, match="Constexpr"):
        _record_cast(first, type(other))
    with pytest.raises(TypeError, match="Constexpr"):
        _record_select(fx.Boolean(True), first, other)
