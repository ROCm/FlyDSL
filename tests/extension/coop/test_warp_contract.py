# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Warp primitive contracts, tested on mainline language features.

CPU references are explicit even when another module sets the default to CUDA.
Only specified result slots are checked for reduce and partial sorts.
"""

import inspect
import math

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.extension.coop import warp
from flydsl.extension.coop.warp import rocdl


@pytest.fixture(params=["cpu", "cuda"], ids=["default-cpu", "default-cuda"])
def default_device(request):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    with torch.device(request.param):
        yield


@pytest.fixture(params=[False, True], ids=["dispatched", "universal"])
def api(request):
    return fx.coop.universal if request.param else fx.coop


def run(apply, values, size, block=(128, 1, 1), *, compile_only=False):
    @flyc.kernel(known_block_size=list(block))
    def kernel(a: fx.Tensor, out: fx.Tensor):
        apply(a, out)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(a, out).launch(grid=(1, 1, 1), block=block, stream=stream)

    out = torch.full((size,), -99, dtype=values.dtype, device="cpu" if compile_only else "cuda")
    if compile_only:
        return launch(values, out)
    launch(values, out, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    return out.cpu()


def test_public_surface_and_signatures():
    signatures = {
        "warp_reduce": ("value", "op", "width", "valid_items"),
        "warp_head_segmented_reduce": ("value", "head_flag", "op", "width"),
        "warp_tail_segmented_reduce": ("value", "tail_flag", "op", "width"),
        "warp_broadcast": ("value", "source_lane", "width"),
        "warp_inclusive_scan": ("value", "op", "width", "init", "valid_items"),
        "warp_exclusive_scan": ("value", "op", "width", "init", "valid_items"),
        "warp_scan": ("value", "op", "width", "init", "valid_items"),
        "warp_scan_with_aggregate": ("value", "op", "width", "init", "valid_items"),
        "warp_reduce_batched": ("value", "op", "width", "sync_physical_warp"),
        "warp_reduce_batched_to_blocked": ("value", "op", "width", "sync_physical_warp"),
        "warp_reduce_batched_to_striped": ("value", "op", "width", "sync_physical_warp"),
        "warp_exchange_storage": ("dtype", "items_per_thread", "block_size"),
        "warp_blocked_to_striped": ("value", "width", "algorithm", "storage"),
        "warp_striped_to_blocked": ("value", "width", "algorithm", "storage"),
        "warp_scatter_to_striped": ("value", "ranks", "width", "algorithm", "storage"),
        "warp_load": ("source", "items_per_thread", "width", "offset", "valid_items", "default", "algorithm"),
        "warp_store": ("destination", "value", "width", "offset", "valid_items", "algorithm"),
        "warp_bitonic_sort": ("keys", "values", "width", "compare_op", "valid_items"),
        "warp_merge_sort": ("keys", "values", "width", "compare_op", "valid_items"),
    }
    enums = {"WarpExchangeAlgorithm", "WarpLoadAlgorithm", "WarpStoreAlgorithm"}
    assert set(warp.__all__) == set(signatures) | enums
    for namespace in (fx.coop, fx.coop.universal, warp):
        for name, parameters in signatures.items():
            assert tuple(inspect.signature(getattr(namespace, name)).parameters) == parameters
        for name in ("warp_gather", "warp_bitonic_merge"):
            assert not hasattr(namespace, name)
    for name in signatures:
        params = inspect.signature(getattr(warp, name)).parameters
        if "width" in params:
            assert params["width"].default is None
            assert params["width"].kind is inspect.Parameter.KEYWORD_ONLY
        if "valid_items" in params:
            assert params["valid_items"].default is None
        if "compare_op" in params:
            assert params["compare_op"].default is inspect.Parameter.empty
        if "sync_physical_warp" in params:
            assert params["sync_physical_warp"].default is False
    assert inspect.signature(warp.warp_load).parameters["algorithm"].default is warp.WarpLoadAlgorithm.DIRECT
    assert inspect.signature(warp.warp_store).parameters["algorithm"].default is warp.WarpStoreAlgorithm.DIRECT
    assert inspect.signature(warp.warp_load).parameters["default"].default is None
    assert (
        inspect.signature(warp.warp_blocked_to_striped).parameters["algorithm"].default
        is warp.WarpExchangeAlgorithm.SMEM
    )
    for name in rocdl.__all__:
        assert inspect.signature(getattr(rocdl, name)) == inspect.signature(getattr(warp, name))


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("warp_inclusive_scan", {"prefix_callback": lambda x: x}),
        ("warp_reduce_batched", {"output_layout": "broadcast"}),
        ("warp_load", {"valid_items_scope": "block"}),
        ("warp_load", {"dtype": fx.Int32}),
        ("warp_store", {"striped": True}),
        ("warp_bitonic_sort", {"descending": True, "compare_op": lambda a, b: a < b}),
        ("warp_merge_sort", {"oob_default": 99, "compare_op": lambda a, b: a < b}),
    ],
)
def test_deferred_keywords_are_rejected(name, kwargs):
    with pytest.raises(TypeError):
        inspect.signature(getattr(fx.coop, name)).bind(None, None, **kwargs)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("width", [1, 8, 32, 64])
def test_reduce_array_guard_and_broadcast(default_device, api, width):
    if width > fx.num_warp_threads():
        pytest.skip("logical width exceeds target")
    host = torch.arange(256, dtype=torch.int32, device="cpu") % 13

    def apply(a, out):
        tid = fx.thread_idx.x
        items = fx.Vector.from_elements([a[tid * 2], a[tid * 2 + 1]])
        total = api.warp_reduce(items, fx.ReductionOp.ADD, width=width)
        out[tid] = api.warp_broadcast(total, 0, width=width)
        partial = api.warp_reduce(a[tid], fx.ReductionOp.ADD, width=width, valid_items=fx.Int32(max(1, width - 1)))
        out[128 + tid] = partial

    result = run(apply, host.cuda(), 256)
    expected = host.reshape(-1, width * 2).sum(1).repeat_interleave(width).to(torch.int32)
    torch.testing.assert_close(result[:128], expected)
    expected = host[:128].reshape(-1, width)[:, : max(1, width - 1)].sum(1).to(torch.int32)
    torch.testing.assert_close(result[128::width], expected)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("valid", [0, 1, 5, 8])
def test_partial_scan_seed_and_invalid_lanes(default_device, api, valid):
    host = torch.arange(1, 129, dtype=torch.int32, device="cpu")

    def apply(a, out):
        tid = fx.thread_idx.x
        inc, exc, total = api.warp_scan_with_aggregate(
            a[tid], fx.ReductionOp.ADD, width=8, init=10, valid_items=fx.Int32(valid)
        )
        out[tid] = inc
        out[128 + tid] = exc
        out[256 + tid] = total
        out[384 + tid] = api.warp_exclusive_scan(a[tid], fx.ReductionOp.ADD, width=8, valid_items=fx.Int32(valid))

    result = run(apply, host.cuda(), 512).reshape(4, 16, 8)
    tile = host.reshape(16, 8)
    inc, exc, unseeded = tile.clone(), tile.clone(), tile.clone()
    inc[:, :valid] = tile[:, :valid].cumsum(1) + 10
    exc[:, :valid] = tile[:, :valid].cumsum(1) - tile[:, :valid] + 10
    unseeded[:, :valid] = exc[:, :valid] - 10
    torch.testing.assert_close(result[0], inc)
    torch.testing.assert_close(result[1], exc)
    torch.testing.assert_close(result[3], unseeded)
    if valid:
        total = tile[:, :valid].sum(1).to(torch.int32).unsqueeze(1).expand(-1, 8)
        torch.testing.assert_close(result[2], total)


# Plain data fields only: no user-defined Struct methods or sequence protocol.
Pair = fx.Struct["scale" : fx.Int32, "bias" : fx.Int32]


def compose(left, right):
    # Affine composition is associative and noncommutative.
    return Pair(left.scale * right.scale, left.bias * right.scale + right.bias)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
def test_struct_noncommutative_scan_and_segment_heads(default_device, api):
    host = torch.arange(128, dtype=torch.int32, device="cpu") % 8 + 1

    def apply(a, out):
        tid = fx.thread_idx.x
        lane = tid % 8
        item = Pair(fx.Int32(2), a[tid])
        inc, exc, total = api.warp_scan_with_aggregate(
            item, compose, width=8, init=Pair(fx.Int32(1), fx.Int32(10)), valid_items=fx.Int32(5)
        )
        out[tid] = inc.bias
        out[128 + tid] = exc.bias
        out[256 + tid] = total.bias
        head = api.warp_head_segmented_reduce(item, lane % 3 == 0, compose, width=8)
        tail = api.warp_tail_segmented_reduce(item, (lane % 3 == 2) | (lane == 7), compose, width=8)
        out[384 + tid] = head.bias
        out[512 + tid] = tail.bias
        reduced = api.warp_reduce((item, Pair(fx.Int32(1), fx.Int32(1))), compose, width=8)
        out[640 + tid] = reduced.bias

    result = run(apply, host.cuda(), 768).reshape(6, 16, 8)
    prefix, excl, total, running = [], [], 0, 10
    for x in range(1, 6):
        excl.append(running)
        running = running * 2 + x
        prefix.append(running)
        total = total * 2 + x
    for row, expected in [(0, prefix + [6, 7, 8]), (1, excl + [6, 7, 8]), (2, [total] * 8)]:
        torch.testing.assert_close(result[row], torch.tensor(expected, dtype=torch.int32, device="cpu").expand(16, -1))
    for lane, expected in [(0, 11), (3, 32), (6, 22)]:
        assert torch.all(result[3:5, :, lane] == expected)
    total = 0
    for x in range(1, 9):
        total = total * 2 + x + 1
    assert torch.all(result[5, :, 0] == total)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("layout,batches", [("scalar", 3), ("blocked", 11), ("striped", 11), ("blocked", 0)])
@pytest.mark.parametrize("sync", [False, True])
def test_batched_ownership(default_device, api, layout, batches, sync):
    host = torch.arange(128, dtype=torch.int32, device="cpu") + 1
    slots = max(1, math.ceil(batches / 8)) if layout != "scalar" else 1
    name = "warp_reduce_batched" + ("" if layout == "scalar" else "_to_" + layout)

    def apply(a, out):
        tid = fx.thread_idx.x
        items = tuple(a[tid] + j for j in range(batches))
        result = getattr(api, name)(items, fx.ReductionOp.ADD, width=8, sync_physical_warp=sync)
        if batches == 0:
            out[tid] = len(result)
        elif layout == "scalar":
            out[tid] = result
        else:
            for j in range(slots):
                out[tid * slots + j] = result[j]

    result = run(apply, host.cuda(), 128 * slots).reshape(16, 8, slots)
    for group in range(16):
        for lane in range(8):
            for slot in range(slots):
                batch = lane * slots + slot if layout == "blocked" else slot * 8 + lane
                if batches == 0:
                    assert result[group, lane, slot] == 0
                elif batch < batches:
                    assert result[group, lane, slot] == host[group * 8 : group * 8 + 8].sum() + 8 * batch


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("policy", ["DIRECT", "STRIPED", "VECTORIZE", "TRANSPOSE"])
@pytest.mark.parametrize("valid", [0, 1, 19, 32])
def test_io_tile_base_guard_and_layout(default_device, api, policy, valid):
    # Two physical warps, xyz linear thread ids and permuted tile bases.
    host = torch.arange(16 * 37, dtype=torch.int32, device="cpu") + 100
    lp = getattr(api.WarpLoadAlgorithm, policy)
    sp = getattr(api.WarpStoreAlgorithm, policy)

    def apply(a, out):
        tid = fx.thread_idx.x + 16 * fx.thread_idx.y + 64 * fx.thread_idx.z
        base = (15 - tid // 8) * 37 + 2
        items = api.warp_load(a, 4, width=8, offset=base, valid_items=fx.Int32(valid), default=-7, algorithm=lp)
        for j in range(4):
            out[tid * 4 + j] = items[j]
        api.warp_store(out, items, width=8, offset=512 + base, valid_items=fx.Int32(valid), algorithm=sp)

    result = run(apply, host.cuda(), 512 + host.numel(), block=(16, 4, 2))
    expected = torch.full((128, 4), -7, dtype=torch.int32, device="cpu")
    stored = torch.full_like(host, -99)
    for tid in range(128):
        base = (15 - tid // 8) * 37 + 2
        stored[base : base + valid] = host[base : base + valid]
        for j in range(4):
            index = tid % 8 + j * 8 if policy == "STRIPED" else tid % 8 * 4 + j
            if index < valid:
                expected[tid, j] = host[base + index]
    torch.testing.assert_close(result[:512].reshape(128, 4), expected)
    torch.testing.assert_close(result[512:], stored)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("policy,count", [("SMEM", 3), ("SHUFFLE", 8)])
def test_exchange_layout_and_smem_scatter(default_device, api, policy, count):
    host = torch.arange(128 * count, dtype=torch.int32, device="cpu")
    algorithm = getattr(api.WarpExchangeAlgorithm, policy)

    def apply(a, out):
        tid = fx.thread_idx.x
        value = fx.Vector.from_elements([a[tid * count + j] for j in range(count)])
        storage = fx.SharedAllocator().allocate(api.warp_exchange_storage(fx.Int32, count)).peek()
        striped = api.warp_blocked_to_striped(value, width=8, algorithm=algorithm, storage=storage)
        for j in range(count):
            out[tid * count + j] = striped[j]
        fx.barrier()
        original = api.warp_striped_to_blocked(striped, width=8, algorithm=algorithm, storage=storage)
        for j in range(count):
            out[128 * count + tid * count + j] = original[j]
        fx.barrier()
        ranks = fx.Vector.from_elements([8 * count - 1 - (tid % 8 * count + j) for j in range(count)])
        reversed_items = api.warp_scatter_to_striped(value, ranks, width=8, storage=storage)
        for j in range(count):
            out[256 * count + tid * count + j] = reversed_items[j]

    result = run(apply, host.cuda(), 384 * count).reshape(3, 16, 8, count)
    logical = host.reshape(16, 8, count)
    torch.testing.assert_close(result[0], logical.reshape(16, count, 8).transpose(1, 2))
    torch.testing.assert_close(result[1], logical)
    torch.testing.assert_close(result[2], logical.reshape(16, -1).flip(1).reshape(16, count, 8).transpose(1, 2))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("family", ["bitonic", "merge"])
@pytest.mark.parametrize("count,valid", [(1, None), (3, 17), (3, 0)])
@pytest.mark.parametrize("descending", [False, True])
def test_sort_layout_pairs_and_stability(default_device, api, family, count, valid, descending):
    host = (torch.arange(128 * count, dtype=torch.int32, device="cpu") * 17 + 5) % 11
    name = "warp_" + family + "_sort"

    def compare(a, b):
        return a > b if descending else a < b

    def apply(a, out):
        tid = fx.thread_idx.x
        keys = fx.Vector.from_elements([a[tid * count + j] for j in range(count)])
        positions = fx.Vector.from_elements([tid * count + j for j in range(count)])
        keys, positions = getattr(api, name)(keys, positions, width=8, compare_op=compare, valid_items=valid)
        for j in range(count):
            out[tid * count + j] = keys[j]
            out[128 * count + tid * count + j] = positions[j]

    result = run(apply, host.cuda(), 256 * count).reshape(2, 16, 8, count)
    source = host.reshape(16, 8, count)
    positions = torch.arange(host.numel(), dtype=torch.int32, device="cpu").reshape(16, 8, count)
    if family == "bitonic":
        source = source.transpose(1, 2)
        positions = positions.transpose(1, 2)
        result = result.transpose(2, 3)
    size = 8 * count if valid is None else valid
    source = source.reshape(16, -1)[:, :size]
    positions = positions.reshape(16, -1)[:, :size]
    result = result.reshape(2, 16, -1)[:, :, :size]
    expected, order = torch.sort(source, dim=1, descending=descending, stable=True)
    torch.testing.assert_close(result[0], expected)
    torch.testing.assert_close(host[result[1].to(torch.int64)], expected)
    for group in range(16):
        assert sorted(result[1, group].tolist()) == sorted(positions[group].tolist())
    if family == "merge":
        torch.testing.assert_close(result[1], positions.gather(1, order))


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.parametrize("arch", ["gfx942", "gfx1100"])
def test_compile_on_wave64_and_wave32(monkeypatch, arch):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    host = torch.arange(256, dtype=torch.int32, device="cpu")

    def apply(a, out):
        tid = fx.thread_idx.x
        x = fx.coop.warp_load(a, 2, width=8, offset=(tid // 8) * 16, valid_items=13, default=-1)
        x = fx.coop.warp_merge_sort(x, width=8, compare_op=lambda a, b: a < b, valid_items=13)
        x = fx.coop.warp_bitonic_sort(x, width=8, compare_op=lambda a, b: a > b)
        total = fx.coop.warp_reduce(x, fx.ReductionOp.ADD, width=8)
        _, prefix, aggregate = fx.coop.warp_scan_with_aggregate(
            total, fx.ReductionOp.ADD, width=8, init=0, valid_items=5
        )
        out[tid] = prefix + aggregate

    run(apply, host, 128, compile_only=True)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.parametrize(
    "case,message",
    [
        ("reduce_range_guard", "single item per lane"),
        ("shuffle_shape", "items_per_thread == width"),
        ("shuffle_scatter", "only SMEM"),
        ("load_default", "default requires valid_items"),
        ("sort_comparator", "strict ordering callable"),
        ("batched_scalar", "1 <= batches <= width"),
    ],
)
def test_unsupported_overload_combinations(monkeypatch, case, message):
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    host = torch.ones(128, dtype=torch.int32, device="cpu")

    def apply(a, out):
        tid = fx.thread_idx.x
        x = fx.Vector.from_elements([a[tid], a[tid]])
        if case == "reduce_range_guard":
            out[tid] = fx.coop.warp_reduce(x, fx.ReductionOp.ADD, width=8, valid_items=3)
        elif case == "shuffle_shape":
            fx.coop.warp_blocked_to_striped(x, width=8, algorithm=fx.coop.WarpExchangeAlgorithm.SHUFFLE)
        elif case == "shuffle_scatter":
            fx.coop.warp_scatter_to_striped(x, x, width=8, algorithm=fx.coop.WarpExchangeAlgorithm.SHUFFLE)
        elif case == "sort_comparator":
            fx.coop.warp_bitonic_sort(x, width=8, compare_op=None)
        elif case == "load_default":
            fx.coop.warp_load(a, width=8, default=-1)
        else:
            fx.coop.warp_reduce_batched(x, fx.ReductionOp.ADD, width=1)

    with pytest.raises((TypeError, ValueError), match=message):
        run(apply, host, 128, compile_only=True)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("dtype", [torch.int16, torch.float16, torch.float32, torch.float64])
def test_numeric_io_sort_and_scan(default_device, api, dtype):
    host = (torch.arange(256, dtype=torch.int32, device="cpu") % 11 - 5).to(dtype)
    if dtype.is_floating_point:
        host *= 0.25

    def apply(a, out):
        tid = fx.thread_idx.x
        x = api.warp_load(a, 2, width=8, offset=tid // 8 * 16, algorithm=api.WarpLoadAlgorithm.VECTORIZE)
        x = api.warp_merge_sort(x, width=8, compare_op=lambda a, b: a < b)
        api.warp_store(out, x, width=8, offset=tid // 8 * 16, algorithm=api.WarpStoreAlgorithm.VECTORIZE)
        inc, exc, total = api.warp_scan_with_aggregate(a[tid], fx.ReductionOp.ADD, width=8, init=0)
        assert inc.dtype is a.dtype and exc.dtype is a.dtype and total.dtype is a.dtype
        out[256 + tid] = inc
        out[384 + tid] = exc
        out[512 + tid] = total

    result = run(apply, host.cuda(), 640)
    torch.testing.assert_close(result[:256], host.reshape(16, 16).sort(1).values.flatten())
    values = host[:128].reshape(16, 8)
    inclusive = values.cumsum(1).to(dtype)
    torch.testing.assert_close(result[256:384], inclusive.flatten())
    torch.testing.assert_close(result[384:512], (inclusive - values).flatten())
    torch.testing.assert_close(result[512:], values.sum(1).to(dtype).repeat_interleave(8))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("family", ["merge", "bitonic"])
def test_plain_struct_keys_and_pointer_payload(default_device, api, family):
    host = torch.arange(256, dtype=torch.int32, device="cpu") % 11

    def apply(a, out):
        tid = fx.thread_idx.x
        keys = tuple(Pair(a[tid * 2 + j], fx.Int32(tid * 2 + j)) for j in range(2))
        payload = tuple(a.iter + tid * 2 + j for j in range(2))
        keys, payload = getattr(api, "warp_" + family + "_sort")(
            keys, payload, width=8, compare_op=lambda a, b: a.scale < b.scale
        )
        for j in range(2):
            out[tid * 2 + j] = keys[j].scale
            out[256 + tid * 2 + j] = fx.make_view(payload[j], fx.make_layout(1, 1))[0]

    result = run(apply, host.cuda(), 512).reshape(2, 16, 8, 2)
    source = host.reshape(16, 8, 2)
    if family == "bitonic":
        result = result.transpose(2, 3)
        source = source.transpose(1, 2)
    expected = source.reshape(16, 16).sort(1).values
    torch.testing.assert_close(result.reshape(2, 16, 16)[0], expected)
    torch.testing.assert_close(result.reshape(2, 16, 16)[1], expected)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("partial", [False, True])
def test_vector_scan_keeps_seed_and_result_type(default_device, api, partial):
    host = (torch.arange(256, dtype=torch.int32, device="cpu") % 7).to(torch.float16) * 0.25

    def apply(a, out):
        tid = fx.thread_idx.x
        x = fx.Vector.from_elements([a[tid * 2], a[tid * 2 + 1]])
        inc, exc, total = api.warp_scan_with_aggregate(
            x, fx.ReductionOp.ADD, width=8, init=1, valid_items=5 if partial else None
        )
        assert inc.dtype is a.dtype and exc.dtype is a.dtype and total.dtype is a.dtype
        for j in range(2):
            out[tid * 2 + j] = inc[j]
            out[256 + tid * 2 + j] = exc[j]
            out[512 + tid * 2 + j] = total[j]

    result = run(apply, host.cuda(), 768).reshape(3, 16, 8, 2)
    values = host.reshape(16, 8, 2)
    count = 5 if partial else 8
    inc, exc = values.clone(), values.clone()
    inc[:, :count] = values[:, :count].cumsum(1) + 1
    exc[:, :count] = values[:, :count].cumsum(1) - values[:, :count] + 1
    total = values[:, :count].sum(1).unsqueeze(1).expand(-1, 8, -1)
    torch.testing.assert_close(result[0], inc)
    torch.testing.assert_close(result[1], exc)
    torch.testing.assert_close(result[2], total)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("policy", ["DIRECT", "STRIPED", "VECTORIZE", "TRANSPOSE"])
def test_plain_struct_array_io(default_device, api, policy):
    host = torch.arange(256, dtype=torch.int32, device="cpu")

    def apply(a, out):
        tid = fx.thread_idx.x
        allocator = fx.SharedAllocator()
        source = allocator.allocate(fx.Array[Pair, 256]).peek()
        destination = allocator.allocate(fx.Array[Pair, 256]).peek()
        for j in range(2):
            source[tid * 2 + j] = Pair(a[tid * 2 + j], a[tid * 2 + j] + 100)
            destination[tid * 2 + j] = Pair(fx.Int32(-99), fx.Int32(-99))
        fx.barrier()
        x = api.warp_load(
            source,
            2,
            width=8,
            offset=tid // 8 * 16,
            valid_items=13,
            default=Pair(fx.Int32(-7), fx.Int32(-9)),
            algorithm=getattr(api.WarpLoadAlgorithm, policy),
        )
        for j in range(2):
            out[tid * 2 + j] = x[j].scale
        api.warp_store(
            destination,
            x,
            width=8,
            offset=tid // 8 * 16,
            valid_items=13,
            algorithm=getattr(api.WarpStoreAlgorithm, policy),
        )
        fx.barrier()
        for j in range(2):
            out[256 + tid * 2 + j] = destination[tid * 2 + j].bias

    result = run(apply, host.cuda(), 512).reshape(2, 16, 8, 2)
    loaded = host.reshape(16, 16).clone()
    loaded[:, 13:] = -7
    expected = loaded.reshape(16, 2, 8).transpose(1, 2) if policy == "STRIPED" else loaded.reshape(16, 8, 2)
    torch.testing.assert_close(result[0], expected)
    stored = host.reshape(16, 16) + 100
    stored[:, 13:] = -99
    torch.testing.assert_close(result[1], stored.reshape(16, 8, 2))
