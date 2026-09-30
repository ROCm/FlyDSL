# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Keep complete elements distinct from outer item sequences at public APIs."""

import pytest
import torch
from coop_test_utils import coop_default_device as coop_default_device
from coop_test_utils import run_kernel

import flydsl.expr as fx

V = fx.Vector[fx.Int32, 2]


@fx.struct
class Record:
    x: fx.Int32
    y: fx.Int32


def element(kind, x, y):
    return fx.Vector.from_elements([x, y]) if kind == "vector" else Record(x, y)


def components(kind, value):
    return (value[0], value[1]) if kind == "vector" else (value.x, value.y)


BLOCK_POLICIES = [("BlockReduce", p) for p in fx.coop.BlockReduceAlgorithm] + [
    ("BlockScan", p) for p in fx.coop.BlockScanAlgorithm
]
WARP_METHODS = [
    ("WarpReduceBatched", "reduce"),
    ("WarpReduceBatched", "reduce_to_blocked"),
    ("WarpReduceBatched", "reduce_to_striped"),
    ("WarpBitonicSort", "sort"),
    ("WarpMergeSort", "sort"),
]


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.usefixtures("coop_default_device")
@pytest.mark.parametrize("kind", ["vector", "struct"])
@pytest.mark.parametrize("valid", [0, 3, 4])
@pytest.mark.parametrize("universal", [False, True], ids=["dispatched", "universal"])
@pytest.mark.parametrize("name,policy", BLOCK_POLICIES, ids=[f"{name}-{p.name}" for name, p in BLOCK_POLICIES])
def test_guarded_complete_element(kind, valid, universal, name, policy):
    namespace = fx.coop.universal if universal else fx.coop
    primitive = getattr(namespace, name)[V if kind == "vector" else Record, 4, policy]
    scan = name == "BlockScan"

    def add(a, b):
        ax, ay = components(kind, a)
        bx, by = components(kind, b)
        return element(kind, ax + bx, ay + by)

    add.commutative = True

    def apply(a, out):
        tid = fx.thread_idx.x
        value = element(kind, a[2 * tid], a[2 * tid + 1])
        zero = element(kind, fx.Int32(0), fx.Int32(0))
        storage = fx.SharedAllocator().allocate(primitive.SharedStorage).peek()
        if scan:
            init = element(kind, fx.Int32(7), fx.Int32(7))
            result, aggregate = primitive.inclusive_with_aggregate(
                value, add, storage=storage, valid_items=valid, identity=zero, init=init
            )
            fx.barrier()
            exclusive = primitive.exclusive(value, add, storage=storage, valid_items=valid, identity=zero, init=init)
            ex, ey = components(kind, exclusive)
            out[16 + 2 * tid] = ex
            out[17 + 2 * tid] = ey
        else:
            result = aggregate = primitive(value, add, storage=storage, valid_items=valid, identity=zero)
        x, y = components(kind, result)
        ax, ay = components(kind, aggregate)
        out[2 * tid] = x
        out[2 * tid + 1] = y
        out[8 + 2 * tid] = ax
        out[9 + 2 * tid] = ay

    host = torch.arange(1, 9, device="cpu", dtype=torch.int32).reshape(4, 2)
    actual = run_kernel(apply, host.flatten().to(device="cuda"), 24 if scan else 16, 4).reshape(-1, 4, 2)
    masked = host.clone()
    masked[valid:] = 0
    total = masked.sum(dim=0, dtype=torch.int32).expand(4, 2)
    prefix = masked.cumsum(dim=0, dtype=torch.int32)
    expected = torch.stack((prefix + 7, total, prefix - masked + 7) if scan else (total, total))
    torch.testing.assert_close(actual, expected)


@pytest.mark.l1a_compile_no_target_dialect
@pytest.mark.parametrize("universal", [False, True], ids=["dispatched", "universal"])
@pytest.mark.parametrize("name", ["BlockReduce", "BlockScan"])
@pytest.mark.parametrize(
    "kind,container,count",
    [
        ("scalar", list, 1),
        ("scalar", tuple, 3),
        ("scalar", fx.Vector.from_elements, 1),
        ("scalar", fx.Vector.from_elements, 3),
        ("vector", list, 1),
        ("vector", tuple, 3),
        ("struct", list, 1),
        ("struct", tuple, 3),
    ],
)
def test_guarded_rejects_outer_sequences(insert_point, monkeypatch, universal, name, kind, container, count):
    monkeypatch.setenv("ARCH", "gfx942")
    namespace = fx.coop.universal if universal else fx.coop
    dtype = {"scalar": fx.Int32, "vector": V, "struct": Record}[kind]
    value = fx.Int32(1) if kind == "scalar" else element(kind, fx.Int32(1), fx.Int32(2))
    values = container([value] * count)
    primitive = getattr(namespace, name)[dtype, 4]
    with pytest.raises(TypeError, match="valid_items is supported only for a single item"):
        if name == "BlockReduce":
            primitive(values, fx.ReductionOp.ADD, storage=None, valid_items=3)
        else:
            primitive.inclusive(values, fx.ReductionOp.ADD, storage=None, valid_items=3)


@pytest.mark.l1a_compile_no_target_dialect
@pytest.mark.parametrize("universal", [False, True], ids=["dispatched", "universal"])
@pytest.mark.parametrize("kind", ["scalar", "vector", "struct"])
@pytest.mark.parametrize("name,method", WARP_METHODS)
def test_warp_array_methods_reject_bare_elements(insert_point, monkeypatch, universal, kind, name, method):
    monkeypatch.setenv("ARCH", "gfx942")
    namespace = fx.coop.universal if universal else fx.coop
    dtype = {"scalar": fx.Int32, "vector": V, "struct": Record}[kind]
    value = fx.Int32(1) if kind == "scalar" else element(kind, fx.Int32(1), fx.Int32(2))
    call = getattr(getattr(namespace, name)[dtype, 4, 1], method)
    with pytest.raises(TypeError, match=r"outer item sequence.*wrap.*\["):
        if method == "sort":
            call(value, compare_op=lambda a, b: a < b)
        else:
            call(value, fx.ReductionOp.ADD)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.usefixtures("coop_default_device")
@pytest.mark.parametrize("universal", [False, True], ids=["dispatched", "universal"])
@pytest.mark.parametrize("kind", ["vector", "struct"])
@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("name,method", WARP_METHODS)
def test_warp_single_item_sequences(universal, kind, container, name, method):
    namespace = fx.coop.universal if universal else fx.coop
    dtype = V if kind == "vector" else Record
    sort = method == "sort"
    primitive = getattr(namespace, name)[(dtype, fx.Int32) if sort else dtype, 4, 1]

    def add(a, b):
        ax, ay = components(kind, a)
        bx, by = components(kind, b)
        return element(kind, ax + bx, ay + by)

    def compare(a, b):
        return components(kind, a)[0] < components(kind, b)[0]

    def apply(a, out):
        tid = fx.thread_idx.x
        value = element(kind, a[2 * tid], a[2 * tid + 1])
        values = container([value])
        call = getattr(primitive, method)
        if sort:
            result, payload = call(values, container([tid.to(fx.Int32)]), compare_op=compare)
            x, y = components(kind, result[0])
            out[2 * tid] = x
            out[2 * tid + 1] = y
            out[8 + tid] = payload[0]
        else:
            result = call(values, add)
            value = result if method == "reduce" else result[0]
            x, y = components(kind, value)
            out[2 * tid] = x
            out[2 * tid + 1] = y

    host = torch.tensor([[4, 40], [1, 10], [3, 30], [2, 20]], device="cpu", dtype=torch.int32)
    actual = run_kernel(apply, host.flatten().to(device="cuda"), 12 if sort else 8, 4)
    if sort:
        order = torch.tensor([1, 3, 2, 0], device="cpu", dtype=torch.int64)
        expected = torch.cat((host[order].flatten(), order.to(dtype=torch.int32)))
    else:
        # A single batch belongs to lane zero in all three output layouts.
        actual = actual[:2]
        expected = host.sum(dim=0, dtype=torch.int32)
    torch.testing.assert_close(actual, expected)
