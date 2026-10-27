# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Block discontinuity flags, indexed operators, boundaries, and record fields."""

import math
from functools import partial

import pytest
from coop_common import BLOCK_THREADS, DTYPES
from coop_test_utils import ARCHES, matrix_cases, run_tile
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
@pytest.mark.parametrize("heads,boundary", matrix_cases([False, True], [None, -3]))
def test_directional_flag_operator(heads, boundary):
    values = ((torch.arange(128 * 3, device="cuda", dtype=torch.int32) * 17) % 11) - 5

    def flag(left, right):
        return left < right

    def transform(value):
        primitive = fx.coop.BlockDiscontinuity[fx.Int32, 128]
        storage = fx.SharedAllocator().allocate(primitive.SharedStorage).peek()
        if heads:
            return primitive.flag_heads(value, flag, storage=storage, predecessor=boundary)
        return primitive.flag_tails(value, flag, storage=storage, successor=boundary)

    result = run_tile(transform, values, 128, 3)[0]
    host = values.cpu()
    neighbor = host.roll(1 if heads else -1)
    endpoint = 0 if heads else -1
    neighbor[endpoint] = 0 if boundary is None else boundary
    expected = (neighbor < host if heads else host < neighbor).int()
    if boundary is None:
        expected[endpoint] = 1
    torch.testing.assert_close(result, expected)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "block,count,boundary,method",
    matrix_cases([64, (8, 4, 4)], [1, 3], [None, -9], ["flag_heads", "flag_tails", "flag_heads_and_tails"]),
)
@pytest.mark.usefixtures("coop_default_device")
def test_neighbors(block, count, boundary, method):
    checks.check_neighbors(block, count, boundary, method)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "block,count,boundary",
    [
        (block, count, boundary)
        for (block, count), boundary in matrix_cases([(64, 1), (64, 3), (128, 9), ((4, 4, 12), 2)], [None, -11])
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_indexed_discontinuity_and_predecessors(block, count, boundary):
    threads = block if isinstance(block, int) else math.prod(block)
    values = (torch.arange(threads * count, device="cuda", dtype=torch.int32) // 3) % 7

    def flag(left, right, right_index):
        return (left < right) | ((right_index % 5) == 2)

    def transform(value):
        cls = fx.coop.universal.BlockDiscontinuity[fx.Int32, block]
        storage = fx.SharedAllocator().allocate(cls.SharedStorage).peek()
        return cls.flag_heads_and_tails(
            value, flag, storage=storage, predecessor=boundary, successor=boundary, return_predecessors=True
        )

    actual = run_tile(transform, values, block, count, outputs=3)
    host = values.cpu()
    left, right = host.roll(1), host.roll(-1)
    if boundary is not None:
        left[0], right[-1] = boundary, boundary
    index = torch.arange(host.numel(), device="cpu")
    heads = ((left < host) | ((index % 5) == 2)).int()
    tails = ((host < right) | (((index + 1) % 5) == 2)).int()
    if boundary is None:
        heads[0], tails[-1] = 1, 1
    torch.testing.assert_close(actual[0], heads)
    torch.testing.assert_close(actual[1], tails)
    torch.testing.assert_close(actual[2, 1:], left[1:])
    if boundary is not None:
        assert actual[2, 0] == boundary


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "dtype,name,method",
    [
        (dtype, name, method)
        for (dtype, name), method in matrix_cases(
            [
                (fx.Uint8, "uint8"),
                (fx.Int16, "int16"),
                (fx.Uint16, "int16"),
                (fx.Int32, "int32"),
                (fx.Int64, "int64"),
                (fx.Float32, "float32"),
                (fx.Float64, "float64"),
            ],
            ["indexed_heads", "indexed_tails"],
        )
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_neighborhood_contract_dtypes(dtype, name, method):
    checks.check_neighborhood_contract_dtypes(dtype, name, method)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "block,count,method,universal",
    [
        (block, count, method, universal)
        for (block, count), method, universal in matrix_cases(
            [(64, 1), (64, 3), (128, 3)], ["heads", "tails"], [False, True]
        )
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_sequence_records(block, count, method, universal, compile_only=False):
    checks.check_sequence_records(block, count, method, universal, compile_only)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.parametrize("arch,method", matrix_cases(ARCHES, ["heads"]))
def test_record_sequence_compile(monkeypatch, arch, method):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    checks.check_sequence_records(192, 3, method, True, compile_only=True)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "entry,count,case",
    matrix_cases(DTYPES, [1, 9], [("BlockDiscontinuity", "flag_heads"), ("BlockDiscontinuity", "flag_tails")]),
)
@pytest.mark.usefixtures("coop_default_device")
def test_matrix_block_dtypes(entry, count, case):
    checks.check_block_case(entry, 128, count, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "block,count,case",
    matrix_cases(BLOCK_THREADS, [1, 9], [("BlockDiscontinuity", "flag_heads"), ("BlockDiscontinuity", "flag_tails")]),
)
@pytest.mark.usefixtures("coop_default_device")
def test_matrix_block_shapes(block, count, case):
    checks.check_block_case((fx.Int32, "torch.int32"), block, count, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "case", [("BlockDiscontinuity", "flag_heads"), ("BlockDiscontinuity", "flag_tails")], ids=lambda c: c[1]
)
@pytest.mark.usefixtures("coop_default_device")
def test_matrix_three_dimensional_block(case):
    checks.check_block_case((fx.Float64, "torch.float64"), (8, 4, 4), 3, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("method,boundary", matrix_cases(["flag_heads", "flag_tails"], [False, True]))
@pytest.mark.usefixtures("coop_default_device")
def test_neighbor_operator_calls(method, boundary):
    checks.check_neighbor_operator_calls(method, None, boundary)


@flyc.jit
def _binary_flag(left, right):
    return left < right


@flyc.jit
def _indexed_flag(left, right, index):
    return (left < right) | ((index % 5) == 2)


@flyc.jit
def _configured_flag(period, left, right, index):
    return (left < right) | ((index % period) == 2)


@fx.struct
class _FlagPredicate:
    period: fx.Int32

    @flyc.jit
    def binary(self, left, right):
        return left < right

    @flyc.jit
    def indexed(self, left, right, index):
        return (left < right) | ((index % self.period) == 2)

    @flyc.jit
    def __call__(self, left, right, index):
        return (left < right) | ((index % self.period) == 2)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize(
    "kind,boundary",
    matrix_cases(
        ["binary", "indexed", "partial", "partial_binary", "bound_binary", "bound_indexed", "callable"], [False, True]
    ),
)
@pytest.mark.usefixtures("coop_default_device")
def test_jit_flag_signatures(kind, boundary):
    block, count = 64, 3
    host = (torch.arange(block * count, device="cpu", dtype=torch.int32) * 17) % 11 - 5

    def transform(items):
        obj = _FlagPredicate(fx.Int32(5))
        predicates = {
            "binary": _binary_flag,
            "indexed": _indexed_flag,
            "partial": partial(_configured_flag, 5),
            "partial_binary": partial(_indexed_flag, index=3),
            "bound_binary": obj.binary,
            "bound_indexed": obj.indexed,
            "callable": obj,
        }
        p = fx.coop.BlockDiscontinuity[fx.Int32, block]
        storage = fx.SharedAllocator().allocate(p.SharedStorage).peek()
        return p.flag_heads_and_tails(
            items,
            predicates[kind],
            storage=storage,
            predecessor=-7 if boundary else None,
            successor=-7 if boundary else None,
        )

    actual = run_tile(transform, host.to(device="cuda"), block, count, outputs=2)
    left, right = host.roll(1), host.roll(-1)
    left[0], right[-1] = -7, -7
    heads, tails = left < host, host < right
    if kind not in ("binary", "bound_binary", "partial_binary"):
        indices = torch.arange(host.numel(), device="cpu")
        heads |= indices % 5 == 2
        tails |= (indices + 1) % 5 == 2
    if not boundary:
        heads[0], tails[-1] = True, True
    torch.testing.assert_close(actual[0], heads.int())
    torch.testing.assert_close(actual[1], tails.int())


@pytest.mark.l1a_compile_no_target_dialect
def test_flag_signature_selection_does_not_call_predicate():
    from flydsl.extension.coop.block.discontinuity import _indexed

    def optional(left, right, index=0):
        raise AssertionError("signature inspection must not invoke the predicate")

    assert _indexed(optional)
    assert not _indexed(partial(optional, index=3))
    assert _indexed(_indexed_flag)
    assert not _indexed(_binary_flag)
    with pytest.raises(TypeError, match="flag_op must accept"):
        _indexed(lambda a: a)
    with pytest.raises(TypeError, match="flag_op must accept"):
        _indexed(lambda a, b, c, d: a)


@pytest.mark.l1a_compile_no_target_dialect
def test_flag_predicate_typeerror_is_not_retried(monkeypatch):
    from coop_test_utils import run_kernel

    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("ARCH", "gfx942")
    calls = []
    error = TypeError("error inside the user's predicate")

    def predicate(*args):
        calls.append(len(args))
        raise error

    def apply(a, out):
        p = fx.coop.BlockDiscontinuity[fx.Int32, 64]
        storage = fx.SharedAllocator().allocate(p.SharedStorage).peek()
        out[fx.thread_idx.x] = p.flag_heads(a[fx.thread_idx.x], predicate, storage=storage)

    with pytest.raises(TypeError) as caught:
        run_kernel(apply, torch.empty(64, dtype=torch.int32, device="cpu"), 64, 64, compile_only=True)
    assert caught.value is error
    assert calls == [3]


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("kind", ["vector", "record"])
@pytest.mark.usefixtures("coop_default_device")
def test_default_flags_compare_complete_elements(kind):
    checks.check_composite_default_flags(kind)
