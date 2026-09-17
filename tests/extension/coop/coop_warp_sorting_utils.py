# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Kernel runners and host references shared by sorting tests."""

import torch
from coop_test_utils import run_tile

import flydsl.expr as fx
from flydsl.extension.coop._values import _as_items, _from_items
from flydsl.extension.coop.warp.bitonic_sort import warp_bitonic_sort
from flydsl.extension.coop.warp.merge_sort import warp_merge_sort


def _striped_to_linear(x, width, count):
    return x.reshape(-1, width, count).transpose(1, 2).reshape(-1, width * count)


_FLOATS = [
    (fx.Float16, torch.float16),
    (fx.BFloat16, torch.bfloat16),
    (fx.Float32, torch.float32),
    (fx.Float64, torch.float64),
]
_PATTERN = [-7.75, -0.0, 2.125, -3.5, 2.125, 0.0, 0.1875, -7.75, 4.25, -0.5, float("inf"), -float("inf")]


def _namespace(universal):
    return fx.coop.universal if universal else fx.coop


def _logical(physical, width, count, striped=False):
    if striped:
        return physical.reshape(-1, width, count).transpose(1, 2).reshape(-1, width * count)
    return physical.reshape(-1, width * count)


def _assert_key_payload(keys, ids, source):
    expected = source[ids]
    torch.testing.assert_close(keys, expected.double(), rtol=0, atol=0)
    assert torch.equal(torch.signbit(keys), torch.signbit(expected))


def check_sort(primitive, block, count, descending):
    torch.manual_seed(14)
    values = torch.randint(-11, 12, (block * count,), dtype=torch.int32, device="cuda")
    width = min(8, block)

    def transform(value):
        return getattr(fx.coop, primitive)(value, width=width, compare_op=lambda a, b: a > b if descending else a < b)

    result = run_tile(transform, values, block, count)[0]
    group = width * count
    expected = values.cpu().reshape(-1, group).sort(dim=1, descending=descending).values.reshape(-1)
    if primitive == "warp_bitonic_sort":
        expected = expected.reshape(-1, count, width).transpose(1, 2).reshape(-1)
    torch.testing.assert_close(result, expected)


def check_sort_pairs(primitive, count, width, descending):
    block = 128
    group = (width or fx.num_warp_threads()) * count
    indices = torch.arange(block * count, dtype=torch.int32, device="cuda")

    def transform(value):
        keys = value * 17 % 11 - 5
        return getattr(fx.coop, primitive)(
            keys, value, width=width, compare_op=lambda a, b: a > b if descending else a < b
        )

    result = run_tile(transform, indices, block, count, outputs=2)
    keys = (indices.cpu() * 17 % 11 - 5).reshape(-1, group)
    expected, permutation = keys.sort(dim=1, descending=descending, stable=True)
    if primitive == "warp_bitonic_sort":
        expected = expected.reshape(-1, count, width or fx.num_warp_threads()).transpose(1, 2)
    torch.testing.assert_close(result[0], expected.reshape(-1))
    payload = result[1].long()
    torch.testing.assert_close((payload * 17 % 11 - 5).int(), result[0])
    torch.testing.assert_close(payload.sort().values, indices.cpu().long())
    if primitive == "warp_merge_sort":
        expected_payload = (
            permutation + torch.arange(block // (width or fx.num_warp_threads()), device="cpu")[:, None] * group
        )
        torch.testing.assert_close(payload, expected_payload.reshape(-1))


def check_bitonic_striped_pairs(count, partial):
    width, block = (8, 64)
    data = torch.arange(block * count, device="cuda", dtype=torch.int32) * 17 % 29
    valid = width * count - 3 if partial and count > 1 else width * count

    def transform(value):
        ids = fx.Vector.from_elements([fx.thread_idx.x * count + i for i in range(count)])
        return warp_bitonic_sort(
            value, ids.to(fx.Int64), width=width, compare_op=lambda a, b: a > b, valid_items=fx.Int32(valid)
        )

    result = run_tile(transform, data, block, count, outputs=2)
    logical = _striped_to_linear(data.cpu(), width, count)
    expected = logical[:, :valid].sort(descending=True).values
    actual = _striped_to_linear(result[0], width, count)
    torch.testing.assert_close(actual[:, :valid], expected)
    ids = _striped_to_linear(result[1], width, count)[:, :valid].long()
    torch.testing.assert_close(data.cpu()[ids], expected)


def check_merge_comparator_pairs_partial(scope, block, descending):
    count, width = (3, 8)
    data = torch.arange(block * count, device="cuda", dtype=torch.int32)
    total = width * count
    valid = total - 5

    def transform(value):
        keys = value * 19 % 43
        op = lambda a, b: a % 5 < b % 5
        return warp_merge_sort(
            keys,
            value,
            width=width,
            compare_op=lambda a, b: op(b, a) if descending else op(a, b),
            valid_items=fx.Int32(valid),
        )

    result = run_tile(transform, data, block, count, outputs=2)
    ids = data.cpu().reshape(-1, total)
    keys = ids * 19 % 43
    permutation = (keys[:, :valid] % 5).argsort(dim=1, descending=descending, stable=True)
    expected_ids = torch.gather(ids[:, :valid], 1, permutation)
    torch.testing.assert_close(result[1].reshape(-1, total)[:, :valid], expected_ids)
    torch.testing.assert_close(result[0].reshape(-1, total)[:, :valid], expected_ids * 19 % 43)


def check_record_keys_and_record_values(case, count):
    from flydsl.extension.coop._values import _from_items

    key_type = fx.Struct["major" : fx.Int32, "minor" : fx.Int64]
    value_type = fx.Struct["index" : fx.Int64, "tag" : fx.Float64]
    block, width = (8, 8)
    data = torch.arange(block * count, device="cuda", dtype=torch.int32)

    def transform(value):
        keys = _from_items([key_type(major=item * 17 % 11 - 5, minor=(item * 7 % 13).to(fx.Int64)) for item in value])
        payload = _from_items([value_type(index=item.to(fx.Int64), tag=item.to(fx.Float64) * 2.0) for item in value])
        compare = lambda a, b: (a.major < b.major) | (a.major == b.major) & (a.minor < b.minor)
        if case == "bitonic":
            result, carried = warp_bitonic_sort(keys, payload, width=width, compare_op=compare)
        elif case == "warp_merge":
            result, carried = warp_merge_sort(keys, payload, width=width, compare_op=compare)
        result, carried = (_as_items(result), _as_items(carried))
        return (
            fx.Vector.from_elements([key.major for key in result]),
            fx.Vector.from_elements([key.minor.to(fx.Int32) for key in result]),
            fx.Vector.from_elements([item.index.to(fx.Int32) for item in carried]),
            fx.Vector.from_elements([item.tag.to(fx.Int32) for item in carried]),
        )

    result = run_tile(transform, data, block, count, outputs=4)
    ids = data.cpu()
    major, minor = (ids * 17 % 11 - 5, ids * 7 % 13)
    logical = _striped_to_linear(ids, width, count).flatten() if case == "bitonic" else ids
    composite = (major[logical] + 5) * 13 + minor[logical]
    order = logical[composite.argsort(stable=True)]
    if case == "bitonic":
        for i in range(4):
            result[i] = _striped_to_linear(result[i], width, count).flatten()
    k = len(ids)
    torch.testing.assert_close(result[0, :k], major[order[:k]])
    torch.testing.assert_close(result[1, :k], minor[order[:k]])
    torch.testing.assert_close(result[2, :k], order[:k])
    torch.testing.assert_close(result[3, :k], order[:k] * 2)


def check_fractional_pairs_and_stability(case, dtype, torch_dtype, descending, universal):
    """Stable entry points preserve duplicate and signed-zero payload order."""
    ns = _namespace(universal)
    count, width = (3, 8)
    block = 16
    group = width * count
    valid = group - 5
    k = 13
    data = torch.tensor(
        (_PATTERN * ((block * count + len(_PATTERN) - 1) // len(_PATTERN)))[: block * count],
        dtype=torch_dtype,
        device="cuda",
    )
    striped_output = case == "bitonic" or (case == "radix_digit" and descending)

    def transform(keys):
        ids = fx.Vector.from_elements([fx.thread_idx.x * count + i for i in range(count)]).to(fx.Int64)
        if case == "bitonic":
            return ns.warp_bitonic_sort(
                keys,
                ids,
                width=width,
                valid_items=fx.Int32(valid),
                compare_op=lambda a, b: a > b if descending else a < b,
            )
        if case == "warp_merge":
            return ns.warp_merge_sort(
                keys,
                ids,
                width=width,
                valid_items=fx.Int32(valid),
                compare_op=lambda a, b: a > b if descending else a < b,
            )
        algorithm = ns.BlockTopKAlgorithm.MERGE
        op = ns.BlockTopK[(dtype, fx.Int64), block, count, algorithm]
        storage = fx.SharedAllocator().allocate(op.SharedStorage).peek()
        result, payload, selected = op(
            keys, fx.Int32(k), ids, storage=storage, largest=descending, num_valid=fx.Int32(valid)
        )
        return (result, payload, selected.to(fx.Int32))

    result = run_tile(transform, data, block, count, outputs=2, dtype=dtype, output_dtype=(fx.Float64, "float64"))
    group_width = width
    source_ids = _logical(torch.arange(len(data), device="cpu"), group_width, count, case == "bitonic")
    logical_keys = data.cpu()[source_ids]
    order = logical_keys[:, :valid].argsort(dim=1, descending=descending, stable=True)
    expected_ids = source_ids[:, :valid].gather(1, order)
    got_keys = _logical(result[0], group_width, count, striped_output)
    got_ids = _logical(result[1].long(), group_width, count, striped_output)
    selected_count = valid
    for group_id in range(len(source_ids)):
        ids = got_ids[group_id, :selected_count]
        keys = got_keys[group_id, :selected_count]
        expected = expected_ids[group_id, :selected_count]
        _assert_key_payload(keys, ids, data.cpu())
        if case == "bitonic":
            torch.testing.assert_close(keys, data.cpu()[expected].double(), rtol=0, atol=0)
            torch.testing.assert_close(ids.sort().values, source_ids[group_id, :valid].sort().values)
        else:
            torch.testing.assert_close(ids, expected)


def check_nested_float_record_keys(case, projected, universal):
    """Nested float fields and untouched fields follow comparator/decomposer order."""
    ns = _namespace(universal)
    inner = fx.Struct["major" : fx.Float16, "minor" : fx.Float32]
    key_type = fx.Struct["parts":inner, "tag" : fx.Float64]
    info = fx.Struct["id" : fx.Int64, "weight" : fx.Float32]
    value_type = fx.Struct["info":info]
    width, count = (8, 3)
    block = 16
    group_width = width
    n = block * count
    data = torch.tensor(([-3.5, 1.125, -0.0, 0.0, -3.5, 2.75, 1.125, -0.625] * ((n + 7) // 8))[:n], device="cuda")
    valid = group_width * count - 5
    k = 11

    def transform(value):
        ids = [fx.thread_idx.x * count + i for i in range(count)]
        keys = _from_items(
            [
                key_type(
                    parts=inner(major=item.to(fx.Float16), minor=(idx % 5 - 2).to(fx.Float32) * 0.375),
                    tag=item.to(fx.Float64) + idx.to(fx.Float64) * 0.125,
                )
                for item, idx in zip(value, ids)
            ]
        )
        payload = _from_items(
            [value_type(info=info(id=idx.to(fx.Int64), weight=idx.to(fx.Float32) * -0.25)) for idx in ids]
        )
        if projected:
            compare = lambda a, b: a.parts.major < b.parts.major
        else:
            compare = lambda a, b: (a.parts.major < b.parts.major) | (a.parts.major == b.parts.major) & (
                a.parts.minor < b.parts.minor
            )
        if case == "bitonic":
            ordered, carried = ns.warp_bitonic_sort(
                keys,
                payload,
                width=width,
                compare_op=lambda a, b: compare(b, a) if True else compare(a, b),
                valid_items=valid,
            )
        elif case == "warp_merge":
            ordered, carried = ns.warp_merge_sort(
                keys,
                payload,
                width=width,
                compare_op=lambda a, b: compare(b, a) if True else compare(a, b),
                valid_items=valid,
            )
        else:
            op = ns.BlockTopK[(key_type, value_type), block, count, ns.BlockTopKAlgorithm.MERGE]
            storage = fx.SharedAllocator().allocate(op.SharedStorage).peek()
            ordered, carried, _ = op(
                keys, k, payload, storage=storage, largest=True, compare_op=compare, num_valid=valid
            )
        return (
            fx.Vector.from_elements([key.parts.major for key in ordered]),
            fx.Vector.from_elements([key.parts.minor for key in ordered]),
            fx.Vector.from_elements([key.tag for key in ordered]),
            fx.Vector.from_elements([item.info.id for item in carried]),
            fx.Vector.from_elements([item.info.weight for item in carried]),
        )

    result = run_tile(transform, data, block, count, outputs=5, output_dtype=(fx.Float64, "float64"))
    source_ids = _logical(torch.arange(n, device="cpu"), group_width, count, case == "bitonic")
    host = data.cpu().half().double()
    minor = (torch.arange(n, device="cpu") % 5 - 2).double() * 0.375
    got = [_logical(row, group_width, count, case == "bitonic") for row in result]
    selected = valid
    for group_id, group_ids in enumerate(source_ids):
        expected_ids = torch.tensor(
            sorted(
                group_ids[:valid].tolist(),
                key=lambda i: (host[i].item(),) if projected else (host[i].item(), minor[i].item()),
                reverse=True,
            ),
            device="cpu",
        )[:selected]
        ids = got[3][group_id, :selected].long()
        _assert_key_payload(got[0][group_id, :selected], ids, host)
        torch.testing.assert_close(got[1][group_id, :selected], minor[ids], rtol=0, atol=0)
        torch.testing.assert_close(got[2][group_id, :selected], host[ids] + ids.double() * 0.125, rtol=0, atol=0)
        torch.testing.assert_close(got[4][group_id, :selected], ids.double() * -0.25, rtol=0, atol=0)
        if case == "bitonic":
            torch.testing.assert_close(got[0][group_id, :selected], host[expected_ids], rtol=0, atol=0)
            if not projected:
                torch.testing.assert_close(got[1][group_id, :selected], minor[expected_ids], rtol=0, atol=0)
            torch.testing.assert_close(ids.sort().values, group_ids[:valid].sort().values)
        else:
            torch.testing.assert_close(ids, expected_ids)
