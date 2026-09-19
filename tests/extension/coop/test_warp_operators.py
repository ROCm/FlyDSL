# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Operator specialization and equivalence to the direct warp entry points."""

import inspect

import pytest
import torch
from coop_test_utils import run_kernel as run
from coop_test_utils import warp_default_device as warp_default_device
from coop_test_utils import warp_storage

import flydsl.expr as fx
from flydsl.compiler.protocol import dsl_align_of, dsl_size_of, extract_to_ir_values

OPERATORS = (
    ("WarpReduce", False),
    ("WarpScan", False),
    ("WarpReduceBatched", True),
    ("WarpBitonicSort", True),
    ("WarpMergeSort", True),
    ("WarpLoad", True),
    ("WarpStore", True),
    ("WarpExchange", True),
)


@pytest.fixture(params=[False, True], ids=["dispatched", "universal"])
def api(request):
    return fx.coop.universal if request.param else fx.coop


@pytest.fixture(params=["omitted", "static", "dynamic"])
def storage_mode(request):
    return request.param


def empty_storage(mode, *operators):
    if mode == "omitted":
        return (None,) * len(operators)
    allocator = fx.SharedAllocator(static=mode == "static")
    storage = tuple(allocator.allocate(operator.SharedStorage).peek() for operator in operators)
    assert allocator.allocated_bytes == 0
    return storage


@pytest.mark.parametrize("name,tile", OPERATORS)
def test_specialization_metadata_and_target_cache(monkeypatch, name, tile, api):
    monkeypatch.setenv("ARCH", "gfx942")
    root = getattr(api, name)
    params = (fx.Int32, 8, 3) if tile else (fx.Int32, 8)
    operator = root[params]
    assert root[params] is operator
    assert operator.dtype is fx.Int32
    assert operator.warp_threads == 8
    assert operator.items_per_thread == (3 if tile else None)
    with pytest.raises(TypeError, match="already specialized"):
        operator[params]
    with pytest.raises(TypeError, match="member function"):
        operator()
    monkeypatch.setenv("ARCH", "gfx1100")
    assert root[params] is not operator
    monkeypatch.setenv("ARCH", "gfx942")
    assert root[params] is operator


@pytest.mark.parametrize("name,tile", OPERATORS)
@pytest.mark.parametrize("width", [0, 3, -1, 128, True])
def test_invalid_logical_widths(name, tile, width):
    params = (fx.Int32, width, 3) if tile else (fx.Int32, width)
    with pytest.raises((TypeError, ValueError), match="width"):
        getattr(fx.coop, name)[params]


@pytest.mark.parametrize("name,tile", OPERATORS)
def test_specialization_arity_and_dtype(name, tile):
    root = getattr(fx.coop, name)
    with pytest.raises(TypeError):
        root[fx.Int32, 8, 3, None, None]
    with pytest.raises(TypeError, match="collective"):
        root[(str, 8, 3) if tile else (str, 8)]
    if tile:
        with pytest.raises(TypeError):
            root[fx.Int32, 8]
        for count in (-1, True, 1.5):
            with pytest.raises(ValueError, match="items_per_thread"):
                root[fx.Int32, 8, count]


def test_algorithms_are_type_parameters_and_scratch_is_per_group(insert_point):
    for name, method in (("Load", "load"), ("Store", "store"), ("Exchange", "blocked_to_striped")):
        root = getattr(fx.coop, "Warp" + name)
        with pytest.raises(TypeError, match="Algorithm"):
            root[fx.Int32, 8, 3, "direct"]
        assert "algorithm" not in inspect.signature(getattr(root, method)).parameters
        assert "width" not in inspect.signature(getattr(root, method)).parameters
    assert fx.coop.WarpLoad[fx.Int32, 8, 3].algorithm is fx.coop.WarpLoadAlgorithm.DIRECT
    assert fx.coop.WarpStore[fx.Int32, 8, 3].algorithm is fx.coop.WarpStoreAlgorithm.DIRECT
    exchange = fx.coop.WarpExchange[fx.Int32, 8, 3]
    assert dsl_size_of(exchange.SharedStorage) == 8 * 3 * 4
    shuffle = fx.coop.WarpExchange[fx.Int32, 8, 8, fx.coop.WarpExchangeAlgorithm.SHUFFLE]
    assert shuffle.SharedStorage is fx.Empty
    assert dsl_size_of(shuffle.SharedStorage) == 0
    with pytest.raises(ValueError, match="items_per_thread == width"):
        fx.coop.WarpExchange[fx.Int32, 8, 3, fx.coop.WarpExchangeAlgorithm.SHUFFLE]
    with pytest.raises(TypeError, match="requires storage allocated from WarpExchange.SharedStorage"):
        exchange.blocked_to_striped(fx.Vector.filled(3, 1, fx.Int32))


def test_member_preconditions_and_pair_specialization(insert_point):
    with pytest.raises(TypeError, match="specialize"):
        fx.coop.WarpReduce.reduce(fx.Int32(1), fx.ReductionOp.ADD)
    with pytest.raises(ValueError, match="expected 3 items"):
        fx.coop.WarpStore[fx.Int32, 8, 3].store(None, fx.Vector.filled(2, 1, fx.Int32))
    sort = fx.coop.WarpMergeSort[(fx.Int32, fx.Float32), 8, 3]
    assert sort.key_dtype is fx.Int32 and sort.value_dtype is fx.Float32
    with pytest.raises(TypeError, match="payload presence"):
        sort.sort(fx.Vector.filled(3, 1, fx.Int32), compare_op=lambda a, b: a < b)
    with pytest.raises(TypeError, match="key/value dtype pair"):
        fx.coop.WarpLoad[(fx.Int32, fx.Int32), 8, 3]
    empty = fx.coop.WarpReduceBatched[fx.Int32, 8, 0]
    assert empty.reduce_to_blocked((), fx.ReductionOp.ADD) == ()
    assert empty.reduce_to_striped((), fx.ReductionOp.ADD) == ()
    with pytest.raises(ValueError, match="scalar batched"):
        empty.reduce((), fx.ReductionOp.ADD)


def test_member_dispatch_and_portable_namespace(monkeypatch, insert_point):
    from flydsl.extension.coop import warp

    seen = []

    def resolve(backend, name, portable):
        seen.append((backend, name))
        return lambda *args, **kwargs: "target-result"

    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setattr(warp._dispatch, "_resolve", resolve)
    value = fx.Int32(1)
    assert fx.coop.WarpReduce[fx.Int32, 8].reduce(value, fx.ReductionOp.ADD) == "target-result"
    assert fx.coop.WarpScan[fx.Int32, 8].inclusive_scan(value, fx.ReductionOp.ADD) == "target-result"
    assert seen == [("rocm", "warp_reduce"), ("rocm", "warp_inclusive_scan")]
    assert fx.coop.universal.WarpReduce[fx.Int32, 8]._dispatcher is None
    assert fx.coop.universal.WarpScan[fx.Int32, 8]._dispatcher is None


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.usefixtures("warp_default_device")
@pytest.mark.parametrize("width", [1, 8, 32])
def test_reduce_members_and_direct_functions(api, width, storage_mode):
    host = torch.arange(64, dtype=torch.int32, device="cpu") + 1

    def apply(a, out):
        tid = fx.thread_idx.x
        value = a[tid]
        P = api.WarpReduce[fx.Int32, width]
        S = api.WarpScan[fx.Int32, width]
        rstorage, sstorage = empty_storage(storage_mode, P, S)
        results = (
            P.reduce(value, fx.ReductionOp.ADD, storage=rstorage),
            api.warp_reduce(value, fx.ReductionOp.ADD, width=width),
            P.head_segmented_reduce(value, tid % width == 0, fx.ReductionOp.ADD, storage=rstorage),
            P.tail_segmented_reduce(value, tid % width == width - 1, fx.ReductionOp.ADD, storage=rstorage),
        )
        for i, result in enumerate(results):
            out[i * 64 + tid] = result if i < 2 else S.broadcast(result, 0, storage=sstorage)

    actual = run(apply, host.cuda(), 256, block=(64, 1, 1)).reshape(4, 64)
    expected = host.reshape(-1, width).sum(1).repeat_interleave(width).to(torch.int32)
    for result in actual:
        torch.testing.assert_close(result, expected)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.usefixtures("warp_default_device")
@pytest.mark.parametrize("width", [1, 8, 32])
def test_scan_members_and_direct_functions(api, width, storage_mode):
    host = torch.arange(64, dtype=torch.int32, device="cpu") % 7

    def apply(a, out):
        tid = fx.thread_idx.x
        value = a[tid]
        P = api.WarpScan[fx.Int32, width]
        (storage,) = empty_storage(storage_mode, P)
        inclusive, exclusive, aggregate = P.scan_with_aggregate(value, fx.ReductionOp.ADD, init=3, storage=storage)
        both = P.scan(value, fx.ReductionOp.ADD, init=3, storage=storage)
        results = (
            P.inclusive_scan(value, fx.ReductionOp.ADD, init=3, storage=storage),
            P.exclusive_scan(value, fx.ReductionOp.ADD, init=3, storage=storage),
            inclusive,
            exclusive,
            aggregate,
            both[0],
            both[1],
            api.warp_inclusive_scan(value, fx.ReductionOp.ADD, width=width, init=3),
            api.warp_exclusive_scan(value, fx.ReductionOp.ADD, width=width, init=3),
        )
        for i, result in enumerate(results):
            out[i * 64 + tid] = result

    actual = run(apply, host.cuda(), 64 * 9, block=(64, 1, 1)).reshape(9, 64)
    inclusive = host.reshape(-1, width).cumsum(1).reshape(-1).to(torch.int32) + 3
    exclusive = inclusive - host
    aggregate = host.reshape(-1, width).sum(1).repeat_interleave(width).to(torch.int32)
    for result, expected in zip(
        actual, (inclusive, exclusive, inclusive, exclusive, aggregate, inclusive, exclusive, inclusive, exclusive)
    ):
        torch.testing.assert_close(result, expected)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.usefixtures("warp_default_device")
@pytest.mark.parametrize("count", [3, 9])
@pytest.mark.parametrize("layout", ["blocked", "striped"])
def test_batched_members_and_direct_functions(api, count, layout, storage_mode):
    width = 8
    output_count = (count + width - 1) // width
    host = torch.arange(64 * count, dtype=torch.int32, device="cpu") % 13

    def apply(a, out):
        tid = fx.thread_idx.x
        value = fx.Vector.from_elements([a[tid * count + i] for i in range(count)])
        P = api.WarpReduceBatched[fx.Int32, width, count]
        (storage,) = empty_storage(storage_mode, P)
        method = P.reduce_to_blocked if layout == "blocked" else P.reduce_to_striped
        direct = api.warp_reduce_batched_to_blocked if layout == "blocked" else api.warp_reduce_batched_to_striped
        results = (method(value, fx.ReductionOp.ADD, storage=storage), direct(value, fx.ReductionOp.ADD, width=width))
        for j, result in enumerate(results):
            for i in range(output_count):
                out[j * 64 * output_count + tid * output_count + i] = result[i]
        if count <= width:
            out[128 * output_count + tid] = P.reduce(value, fx.ReductionOp.ADD, storage=storage)

    actual = run(apply, host.cuda(), 128 * output_count + 64, block=(64, 1, 1))
    totals = host.reshape(8, width, count).sum(1).to(torch.int32)
    for result in actual[: 128 * output_count].reshape(2, 8, width, output_count):
        arranged = result.reshape(8, -1) if layout == "blocked" else result.transpose(1, 2).reshape(8, -1)
        torch.testing.assert_close(arranged[:, :count], totals)
    if count <= width:
        torch.testing.assert_close(actual[128 * output_count :].reshape(8, width)[:, :count], totals)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.usefixtures("warp_default_device")
@pytest.mark.parametrize("family", ["Bitonic", "Merge"])
def test_sort_members_and_direct_functions(api, family, storage_mode):
    count, width = 3, 8
    host = (torch.arange(64 * count, device="cpu", dtype=torch.int32) * 17) % 191

    def apply(a, out):
        tid = fx.thread_idx.x
        keys = fx.Vector.from_elements([a[tid * count + i] for i in range(count)])
        payload = keys + 1000
        P = getattr(api, "Warp" + family + "Sort")[(fx.Int32, fx.Int32), width, count]
        (storage,) = empty_storage(storage_mode, P)
        direct = api.warp_bitonic_sort if family == "Bitonic" else api.warp_merge_sort
        results = (
            P.sort(keys, payload, compare_op=lambda a, b: a < b, storage=storage),
            direct(keys, payload, width=width, compare_op=lambda a, b: a < b),
        )
        for j, (sorted_keys, sorted_values) in enumerate(results):
            for i in range(count):
                out[j * 384 + tid * count + i] = sorted_keys[i]
                out[j * 384 + 192 + tid * count + i] = sorted_values[i]

    actual = run(apply, host.cuda(), 768, block=(64, 1, 1)).reshape(2, 2, 192)
    expected = host.reshape(-1, width * count).sort(1).values
    if family == "Bitonic":
        expected = expected.reshape(-1, count, width).transpose(1, 2)
    expected = expected.reshape(-1)
    for keys, values in actual:
        torch.testing.assert_close(keys, expected)
        torch.testing.assert_close(values, expected + 1000)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.parametrize("arch", ["gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1201"])
@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
def test_compile_operator_pipeline(monkeypatch, arch, api, default_device):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")

    def apply(a, out):
        tid = fx.thread_idx.x
        allocator = fx.SharedAllocator()
        L = api.WarpLoad[fx.Int32, 8, 2, api.WarpLoadAlgorithm.TRANSPOSE]
        E = api.WarpExchange[fx.Int32, 8, 2]
        B = api.WarpBitonicSort[fx.Int32, 8, 2]
        M = api.WarpMergeSort[fx.Int32, 8, 2]
        R = api.WarpReduce[fx.Int32, 8]
        S = api.WarpScan[fx.Int32, 8]
        RB = api.WarpReduceBatched[fx.Int32, 8, 2]
        ST = api.WarpStore[fx.Int32, 8, 2, api.WarpStoreAlgorithm.TRANSPOSE]
        value = L.load(a, offset=tid // 8 * 16, storage=warp_storage(L, allocator))
        value = E.blocked_to_striped(value, storage=warp_storage(E, allocator))
        value = B.sort(value, compare_op=lambda a, b: a < b, storage=warp_storage(B, allocator))
        value = M.sort(value, compare_op=lambda a, b: a < b, storage=warp_storage(M, allocator))
        reduced = R.reduce(value, fx.ReductionOp.ADD, storage=warp_storage(R, allocator))
        scanned = S.inclusive_scan(reduced, fx.ReductionOp.ADD, storage=warp_storage(S, allocator))
        batches = RB.reduce_to_striped(value, fx.ReductionOp.ADD, storage=warp_storage(RB, allocator))
        value = fx.Vector.from_elements([scanned, batches[0]])
        ST.store(out, value, offset=tid // 8 * 16, storage=warp_storage(ST, allocator))
        assert allocator.allocated_bytes == 64 * 2 * 4

    with torch.device(default_device):
        host = torch.empty(128, device="cpu", dtype=torch.int32)
        run(apply, host, 128, block=(64, 1, 1), compile_only=True)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.usefixtures("warp_default_device")
def test_native_record_operator_members(api):
    record = fx.Struct["value" : fx.Int32, "tag" : fx.Int32]
    host = torch.arange(64, device="cpu", dtype=torch.int32) + 1

    def apply(a, out):
        tid = fx.thread_idx.x
        value = record(a[tid], fx.Int32(1))

        def combine(a, b):
            return record(a.value + b.value, a.tag + b.tag)

        total = api.WarpReduce[record, 8].reduce(value, combine)
        scanned = api.WarpScan[record, 8].inclusive_scan(value, combine)
        keys, payload = api.WarpMergeSort[(record, fx.Int32), 8, 1].sort(
            (value,), fx.Vector.from_elements([tid]), compare_op=lambda a, b: a.value > b.value
        )
        out[tid] = total.value
        out[64 + tid] = total.tag
        out[128 + tid] = scanned.value
        out[192 + tid] = keys[0].value
        out[256 + tid] = payload[0]

    actual = run(apply, host.cuda(), 320, block=(64, 1, 1)).reshape(5, 64)
    expected = (
        host.reshape(-1, 8).sum(1).repeat_interleave(8).to(torch.int32),
        torch.full((64,), 8, dtype=torch.int32, device="cpu"),
        host.reshape(-1, 8).cumsum(1).reshape(-1).to(torch.int32),
        host.reshape(-1, 8).flip(1).reshape(-1),
        (host - 1).reshape(-1, 8).flip(1).reshape(-1),
    )
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference)


@pytest.mark.parametrize("name,tile", [entry for entry in OPERATORS if entry[0] != "WarpExchange"])
def test_register_operator_storage_has_empty_layout(name, tile):
    root = getattr(fx.coop, name)
    operator = root[(fx.Int32, 8, 2) if tile else (fx.Int32, 8)]
    assert operator.SharedStorage is fx.Empty
    assert dsl_size_of(operator.SharedStorage) == 0
    assert dsl_align_of(operator.SharedStorage) == 1
    assert extract_to_ir_values(operator.SharedStorage()) == []
    assert dsl_size_of(fx.Array[operator.SharedStorage, 8]) == 0


def test_explicit_storage_type_is_checked(insert_point):
    value = fx.Int32(1)
    with pytest.raises(TypeError, match="SharedStorage"):
        fx.coop.WarpReduce[fx.Int32, 8].reduce(value, fx.ReductionOp.ADD, storage=value)
    with pytest.raises(TypeError, match="SharedStorage"):
        fx.coop.WarpLoad[fx.Int32, 8, 2].load(None, storage=value)
    with pytest.raises(TypeError, match="SharedStorage"):
        fx.coop.WarpStore[fx.Int32, 8, 1].store(None, value, storage=value)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.usefixtures("warp_default_device")
@pytest.mark.parametrize("static", [True, False])
def test_empty_storage_io_and_shuffle(api, static):
    width, count, block = 8, 8, 64
    host = torch.arange(block * count, dtype=torch.int32, device="cpu")

    def apply(a, out):
        allocator = fx.SharedAllocator(static=static)
        L = api.WarpLoad[fx.Int32, width, count]
        E = api.WarpExchange[fx.Int32, width, count, api.WarpExchangeAlgorithm.SHUFFLE]
        S = api.WarpStore[fx.Int32, width, count]
        offset = fx.thread_idx.x // width * width * count
        items = L.load(a, offset=offset, storage=warp_storage(L, allocator))
        scratch = warp_storage(E, allocator)
        striped = E.blocked_to_striped(items, storage=scratch)
        restored = E.striped_to_blocked(striped, storage=scratch)
        S.store(out, restored, offset=offset, storage=warp_storage(S, allocator))
        assert allocator.allocated_bytes == 0

    actual = run(apply, host.cuda(), block * count, block=(block, 1, 1))
    torch.testing.assert_close(actual, host)
