#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The gfx9 ``warp_bitonic_sort`` override against the portable network.

``coop/warp/rocdl/bitonic_sort.py`` runs the portable network compare-exchange
for compare-exchange; only how each step is computed changes (DPP moves,
no validity word when every slot is valid, one comparator call per lane,
min/max for integer keys). So the dispatched ``fx.coop`` form has to agree
with ``fx.coop.universal`` bit for bit, not merely sort:

- **Keys** match the portable output bit for bit over the specified region --
  the first ``valid_items`` logical positions, everything when it is None.
  That pins ``-0.0``/``+0.0`` placement and what the default ``<`` makes of
  NaN, where only the network defines the result. They also equal a host sort.
- **Payloads** match the portable output bit for bit as well: the same
  compare-exchanges with the same strict predicate pair even equal keys with
  the same payloads. Against the host, the valid prefix keeps the input's
  (key, payload) multiset, and with unique keys each payload is the host's.
- **The override is the code that runs**, which only the ISA shows: the
  shuffles leave the LDS pipe for DPP, through the function and through
  ``WarpBitonicSort[...].sort``. Calls it does not cover reach the portable
  implementation.

The sort contract itself (records, descending order, partial tiles through
both namespaces) lives in ``test_warp_bitonic_sort.py``.

Slots past ``valid_items`` are unspecified and never compared: a key-only
integer sort fills them with the dtype's maximum where portable keeps the
invalid inputs.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest
from coop_test_utils import coop_default_device as coop_default_device
from coop_test_utils import matrix_cases

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends import current_target

try:
    import torch
except ImportError:
    torch = None


portable = fx.coop.universal
BLOCK = 128  # two waves; narrower widths also put several logical warps in each
CROSS_LANE_LDS = ("ds_swizzle_b32", "ds_bpermute_b32", "ds_permute_b32")
# A cross-row move of the portable shuffle: on gfx950 the full-wave xor-16/32
# gpu.shuffle lowers to v_permlane16/32_swap (VALU) instead of an LDS-pipe move.
CROSS_ROW_MOVES = (*CROSS_LANE_LDS, "v_permlane16_swap_b32", "v_permlane32_swap_b32")
_INT32_MAX, _UINT32_MAX = 2**31 - 1, 2**32 - 1


def DESC(a, b):
    return a > b


requires_gpu = pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
requires_dpp = pytest.mark.skipif(not current_target().arch.startswith("gfx9"), reason="the DPP network is gfx9-only")


def _override_module():
    return importlib.import_module("flydsl.extension.coop.warp.rocdl.bitonic_sort")


# ── running both forms ─────────────────────────────────────────────────────


def _run(keys, payload, *, width, count, dtype, compare_op=None, valid_items=None, runtime_valid=False):
    """Sort one striped tile with ``fx.coop`` and ``fx.coop.universal`` in one kernel.

    Keys travel as int32 bits (``u32`` is bitcast on the device) or float32.
    """
    has_payload = payload is not None
    if not has_payload:
        payload = torch.zeros(1, dtype=torch.int32, device="cuda")

    def sort(ns, ks, ps, vi):
        return ns.warp_bitonic_sort(ks, ps, width=width, compare_op=compare_op, valid_items=vi)

    def apply(K, P, FK, FP, RK, RP):
        base = fx.thread_idx.x * count
        ks = [K[base + i] for i in range(count)]
        if dtype == "u32":
            ks = [k.bitcast(fx.Uint32) for k in ks]
        ps = [P[base + i] for i in range(count)] if has_payload else None
        vi = fx.Int32(valid_items) if runtime_valid else valid_items
        for ns, OK, OP in ((fx.coop, FK, FP), (portable, RK, RP)):
            result = sort(ns, ks, ps, vi)
            out_k, out_p = result if has_payload else (result, None)
            for i in range(count):
                OK[base + i] = out_k[i].bitcast(fx.Int32) if dtype == "u32" else out_k[i]
                if has_payload:
                    OP[base + i] = out_p[i]

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(K: fx.Tensor, P: fx.Tensor, FK: fx.Tensor, FP: fx.Tensor, RK: fx.Tensor, RP: fx.Tensor):
        apply(K, P, FK, FP, RK, RP)

    @flyc.jit
    def launch(
        K: fx.Tensor,
        P: fx.Tensor,
        FK: fx.Tensor,
        FP: fx.Tensor,
        RK: fx.Tensor,
        RP: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(K, P, FK, FP, RK, RP).launch(grid=(1, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    fk, rk = torch.full_like(keys, -7), torch.full_like(keys, -7)
    fp, rp = torch.full_like(payload, -7), torch.full_like(payload, -7)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch(keys, payload, fk, fp, rk, rp, stream=stream)
    stream.synchronize()
    return fk.cpu(), rk.cpu(), (fp.cpu() if has_payload else None), (rp.cpu() if has_payload else None)


def _logical(physical, width, count):
    """[groups, width * count] in striped logical order from per-lane storage."""
    return physical.reshape(-1, width, count).transpose(1, 2).reshape(-1, width * count)


def _bits(t):
    return t.view(torch.int32) if t.dtype == torch.float32 else t


def _host_order(t, dtype):
    """Values the host sorts by: u32 bit patterns as unsigned, others as stored."""
    return t.to(torch.int64) & _UINT32_MAX if dtype == "u32" else t


def _check(keys, payload, *, width, count, dtype, compare_op=None, valid_items=None, host=True, **kw):
    fk, rk, fp, rp = _run(
        keys, payload, width=width, count=count, dtype=dtype, compare_op=compare_op, valid_items=valid_items, **kw
    )
    n = width * count if valid_items is None else valid_items
    fast, ref = _logical(fk, width, count)[:, :n], _logical(rk, width, count)[:, :n]
    assert torch.equal(_bits(fast), _bits(ref)), "keys differ from the portable network"
    source = _logical(keys.cpu(), width, count)[:, :n]
    descending = compare_op is DESC
    if host:
        expected = _host_order(source, dtype).sort(dim=1, descending=descending).values
        torch.testing.assert_close(_host_order(fast, dtype), expected, rtol=0, atol=0)
    if payload is None:
        return
    fast_p, ref_p = _logical(fp, width, count)[:, :n], _logical(rp, width, count)[:, :n]
    assert torch.equal(fast_p, ref_p), "payloads differ from the portable network"
    src_p = _logical(payload.cpu(), width, count)[:, :n]
    for g in range(source.shape[0]):
        got = sorted(zip(_bits(fast[g]).tolist(), fast_p[g].tolist()))
        assert got == sorted(zip(_bits(source[g]).tolist(), src_p[g].tolist())), "(key, payload) multiset changed"
        if host and len(set(source[g].tolist())) == n:  # unique keys (-0.0 == +0.0 counts as a tie)
            order = _host_order(source[g], dtype).argsort(descending=descending)
            assert torch.equal(fast_p[g], src_p[g][order]), "payload does not follow its unique key"


# ── inputs ─────────────────────────────────────────────────────────────────


def _keys(dtype, kind, count, seed):
    """Host-generated keys on the device: ``unique``, ``dups`` or ``extremes``."""
    n = BLOCK * count
    g = torch.Generator(device="cpu").manual_seed(seed)
    if dtype == "f32":
        values = (torch.randint(-6, 7, (n,), generator=g, device="cpu") * 0.5).to(torch.float32)
        if kind == "unique":
            values = (torch.randperm(n, generator=g, device="cpu") - n // 2).to(torch.float32) * 0.25
            values[:2] = torch.tensor([float("inf"), -float("inf")], device="cpu")
        else:
            special = [-0.0, 0.0, -0.0, 0.0, float("inf"), -float("inf"), 2.5, 2.5]
            if kind == "extremes":
                special += [float("inf"), -float("inf"), -0.0, 0.0] * 2
            where = torch.randperm(n, generator=g, device="cpu")[: len(special)]
            values[where] = torch.tensor(special, device="cpu")
        return values.to("cuda")
    if kind == "unique":
        values = (torch.randperm(n, generator=g, device="cpu") - n // 2).to(torch.int64) * 33_554_393
        values[:2] = torch.tensor([_INT32_MAX, -(2**31)], device="cpu")
    else:
        values = torch.randint(-3, 4, (n,), generator=g, device="cpu", dtype=torch.int64)
        if kind == "extremes":
            # Valid keys equal to the type's maximum tie with the key-only
            # sentinel; they must still land in the valid prefix.
            top = _UINT32_MAX if dtype == "u32" else _INT32_MAX
            values[torch.randperm(n, generator=g, device="cpu")[: n // 4]] = top
            values[torch.randperm(n, generator=g, device="cpu")[:3]] = 0 if dtype == "u32" else -(2**31)
    if dtype == "u32":
        values = values & _UINT32_MAX
    return values.to(torch.int32).to("cuda")  # wraps: the u32 bit pattern


def _ids(count):
    return torch.arange(BLOCK * count, dtype=torch.int32, device="cuda") * 7 + 3


def _valid(kind, width, count):
    total = width * count
    return {
        None: None,
        "odd": min(total, max(1, (total * 5) // 8) | 1),
        "runtime": max(1, total // 3),
    }[kind]


# ── the dispatched form agrees with the portable one ────────────────────────

# Width 64 is every lane distance (DPP in a row, ds_swizzle, ds_bpermute) and
# 2 the smallest group; one item has no register steps, three are padded to
# four. Inexpensive values first: the diagonal pairs 64 lanes with one item.
_WIDTHS = (64, 2, 16)
_COUNTS = (1, 3)
_PAIRS = (False, True)
# All valid, a static prefix and a runtime one; the min/max pairing with
# payloads (it needs every slot valid) is pinned in ``test_path_regressions``.
_VALIDS = (None, "odd", "runtime")
_DATA = ("unique", "dups", "extremes")
# Every path of the override: min/max (signed and unsigned integer keys,
# default order), the one-call select (a callable, or float keys under the
# default ``<``). Rotating the cheap axes per group varies the combinations.
_GROUPS = (("i32", None), ("i32", DESC), ("u32", None), ("f32", None))


def _rotate(axis, by):
    return axis[by % len(axis) :] + axis[: by % len(axis)]


_CASES = [
    (dtype, cmp, *case)
    for g, (dtype, cmp) in enumerate(_GROUPS)
    for case in matrix_cases(_WIDTHS, _COUNTS, _rotate(_PAIRS, g), _rotate(_VALIDS, g), _rotate(_DATA, g))
]


def _case_id(case):
    dtype, cmp, width, count, pairs, valid, data = case
    order = "desc" if cmp is DESC else "asc"
    return f"{dtype}-{order}-w{width}-n{count}-{'pairs' if pairs else 'keys'}-valid{valid}-{data}"


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@requires_dpp
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
@pytest.mark.usefixtures("coop_default_device")
def test_dispatched_matches_portable(case):
    dtype, cmp, width, count, pairs, valid, data = case
    keys = _keys(dtype, data, count, seed=width * 31 + count)
    _check(
        keys,
        _ids(count) if pairs else None,
        width=width,
        count=count,
        dtype=dtype,
        compare_op=cmp,
        valid_items=_valid(valid, width, count),
        runtime_valid=valid == "runtime",
    )


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@requires_dpp
@pytest.mark.parametrize("width,count,pairs,valid", [(8, 2, True, None), (64, 1, False, "odd")])
@pytest.mark.usefixtures("coop_default_device")
def test_nan_under_the_default_comparator(width, count, pairs, valid):
    """``<`` with NaN is no strict weak order: the result is whatever the network
    makes of it, and the override has to make exactly the same of it."""
    keys = _keys("f32", "extremes", count, seed=5)
    keys[torch.randperm(BLOCK * count, device="cpu")[:6].to("cuda")] = float("nan")
    _check(
        keys,
        _ids(count) if pairs else None,
        width=width,
        count=count,
        dtype="f32",
        valid_items=_valid(valid, width, count),
        host=False,
    )


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@requires_dpp
@pytest.mark.parametrize(
    "dtype,width,count,pairs,valid,data",
    [
        # The required shapes the matrix pairs differently: a full wave with
        # two items, payloads and a runtime count (validity on every step),
        # and min/max carrying payloads by ``new != old``.
        ("i32", 64, 2, True, "runtime", "dups"),
        ("i32", 16, 2, True, None, "unique"),
        # Key-only partial tiles whose valid keys tie with the sentinel.
        ("i32", 16, 3, False, "odd", "extremes"),
        ("u32", 32, 2, False, "odd", "extremes"),
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_path_regressions(dtype, width, count, pairs, valid, data):
    keys = _keys(dtype, data, count, seed=width + count)
    _check(
        keys,
        _ids(count) if pairs else None,
        width=width,
        count=count,
        dtype=dtype,
        valid_items=_valid(valid, width, count),
        runtime_valid=valid == "runtime",
    )


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@requires_dpp
@pytest.mark.parametrize(
    "dtype,cmp,count,pairs,valid",
    [
        ("i32", None, 3, False, "odd"),
        ("f32", DESC, 3, True, "odd"),
    ],
)
@pytest.mark.usefixtures("coop_default_device")
def test_single_lane_warps(dtype, cmp, count, pairs, valid):
    """width=1: every step stays inside the lane, in both networks."""
    keys = _keys(dtype, "dups", count, seed=count)
    _check(
        keys,
        _ids(count) if pairs else None,
        width=1,
        count=count,
        dtype=dtype,
        compare_op=cmp,
        valid_items=_valid(valid, 1, count),
    )


# ── the override is the code that runs ─────────────────────────────────────


def _isa(call, count, *, pairs=False, torch_dtype=None, tmp_path):
    """Compile one wave sorting ``count`` keys (and payloads) per lane; return the ISA."""
    torch_dtype = torch.int32 if torch_dtype is None else torch_dtype
    warp_threads = fx.num_warp_threads()
    n = count * (2 if pairs else 1)

    def apply(A, Out):
        base = fx.thread_idx.x * n
        values = [A[base + i] for i in range(n)]
        result = call(values[:count], values[count:] if pairs else None)
        out = [result[0][i] for i in range(count)] + [result[1][i] for i in range(count)] if pairs else result
        for i in range(n):
            Out[base + i] = out[i]

    @flyc.kernel(known_block_size=[warp_threads, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor):
        apply(A, Out)

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out).launch(grid=(1, 1, 1), block=(warp_threads, 1, 1), stream=stream)

    values = torch.ones(warp_threads * n, dtype=torch_dtype, device="cuda")
    out = torch.zeros_like(values)
    keys = ("FLYDSL_DUMP_IR", "FLYDSL_DUMP_DIR", "FLYDSL_RUNTIME_ENABLE_CACHE")
    previous = {k: os.environ.get(k) for k in keys}
    os.environ.update(FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=str(tmp_path), FLYDSL_RUNTIME_ENABLE_CACHE="0")
    try:
        launch(values, out, stream=torch.cuda.Stream())
        torch.cuda.synchronize()
    finally:
        for key, value in previous.items():
            os.environ.pop(key, None) if value is None else os.environ.update({key: value})
    dumped = list(Path(tmp_path).glob("*/*_final_isa.s"))
    assert dumped, f"no ISA dumped under {tmp_path}"
    return dumped[0].read_text()


def _lds(isa):
    return sum(line.count(needle) for line in isa.splitlines() for needle in CROSS_ROW_MOVES)


def _swaps(isa):
    return sum(line.count(needle) for line in isa.splitlines() for needle in PERMLANE_SWAPS)


# gfx950 crosses 16-lane rows with these instead of the LDS pipe.
PERMLANE_SWAPS = ("v_permlane16_swap_b32", "v_permlane32_swap_b32")


def _sorter(ns, width):
    return lambda ks, ps: ns.warp_bitonic_sort(ks, ps, width=width)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@requires_dpp
@pytest.mark.parametrize(
    "width,count,pairs,expected",
    [
        # Per moved value: xor 16 -> ds_swizzle, xor 32 -> ds_bpermute (both a
        # v_permlane*_swap on gfx950); the rest is DPP. Per item of a full
        # wave: d16 at spans 32/64/128, d32 at 64/128.
        (64, 2, False, 10),
        # Inside a row, payloads move by DPP too.
        (16, 2, True, 0),
    ],
)
def test_moves_leave_the_lds_pipe(width, count, pairs, expected, tmp_path):
    fast = _isa(_sorter(fx.coop, width), count, pairs=pairs, tmp_path=tmp_path)
    assert _lds(fast) == expected, "the DPP network did not take"
    if current_target().arch.startswith("gfx950"):
        assert _swaps(fast) == expected, "a cross-row move reached LDS"
    assert "v_min_i32_dpp" in fast and "v_max_i32_dpp" in fast, "the move did not fold into min/max"


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@requires_dpp
def test_float_keys_take_the_select_network(tmp_path):
    """Float keys keep the strict ``<`` select (min/max would lose NaN) but still move by DPP."""
    isa = _isa(_sorter(fx.coop, 16), 1, torch_dtype=torch.float32, tmp_path=tmp_path)
    assert _lds(isa) == 0 and "row_half_mirror" in isa and "row_ror:8" in isa
    assert "v_min_f32" not in isa and "v_max_f32" not in isa


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@requires_dpp
def test_class_reaches_the_override(tmp_path):
    """``WarpBitonicSort`` resolves through the dispatcher like the function."""
    sorter = lambda ks, ps: fx.coop.WarpBitonicSort[(fx.Int32, fx.Int32), 16, 2].sort(ks, ps)  # noqa: E731
    fast = _isa(sorter, 2, pairs=True, tmp_path=tmp_path)
    assert _lds(fast) == 0 and "v_min_i32_dpp" in fast


# ── calls the override does not cover ──────────────────────────────────────


@pytest.fixture
def portable_calls(monkeypatch):
    """Count the override's hand-offs to the portable implementation."""
    module = _override_module()
    calls = []
    original = module._portable_warp_bitonic_sort

    def spy(*args, **kwargs):
        calls.append(kwargs.get("width"))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_portable_warp_bitonic_sort", spy)
    return calls


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@requires_dpp
@pytest.mark.usefixtures("coop_default_device")
def test_wide_payload_falls_back(portable_calls):
    """An Int64 payload is not one 32-bit word: the call goes to the portable network."""
    count, width = 2, 8
    keys = _keys("i32", "dups", count, seed=1)
    valid = width * count - 1

    def apply(K, FK, FP, RK, RP):
        tid = fx.thread_idx.x
        base = tid * count
        ks = [K[base + i] for i in range(count)]
        ids = [(base + i).to(fx.Int64) for i in range(count)]
        for ns, OK, OP in ((fx.coop, FK, FP), (portable, RK, RP)):
            out_k, out_p = ns.warp_bitonic_sort(ks, ids, width=width, valid_items=valid)
            for i in range(count):
                OK[base + i] = out_k[i]
                OP[base + i] = out_p[i].to(fx.Int32)

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(K: fx.Tensor, FK: fx.Tensor, FP: fx.Tensor, RK: fx.Tensor, RP: fx.Tensor):
        apply(K, FK, FP, RK, RP)

    @flyc.jit
    def launch(
        K: fx.Tensor, FK: fx.Tensor, FP: fx.Tensor, RK: fx.Tensor, RP: fx.Tensor, stream: fx.Stream = fx.Stream(None)
    ):
        kernel(K, FK, FP, RK, RP).launch(grid=(1, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    outs = [torch.full_like(keys, -7) for _ in range(4)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch(keys, *outs, stream=stream)
    stream.synchronize()
    fk, fp, rk, rp = (_logical(o.cpu(), width, count)[:, :valid] for o in outs)
    assert portable_calls, "the override kept a 64-bit payload"
    assert torch.equal(fk, rk) and torch.equal(fp, rp)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
# The other gfx9 CI target, and a target without the DPP controls.
@pytest.mark.parametrize("arch", ["gfx950", "gfx1201"])
def test_targets_compile(monkeypatch, portable_calls, arch):
    """gfx9 targets compile the override; every other target hands the call back."""
    monkeypatch.setenv("FLYDSL_GPU_ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    assert current_target().arch == arch
    count, width = 2, 8

    def apply(K, Out):
        base = fx.thread_idx.x * count
        ks = [K[base + i] for i in range(count)]
        keys = fx.coop.warp_bitonic_sort(ks, width=width, valid_items=width * count - 3)
        keys, ids = fx.coop.warp_bitonic_sort(keys, list(ks), width=width, compare_op=DESC)
        for i in range(count):
            Out[base + i] = keys[i] + ids[i]

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kernel(K: fx.Tensor, Out: fx.Tensor):
        apply(K, Out)

    @flyc.jit
    def launch(K: fx.Tensor, Out: fx.Tensor):
        kernel(K, Out).launch(grid=(1, 1, 1), block=(64, 1, 1))

    tensor = torch.empty(64 * count, dtype=torch.int32, device="cpu")
    launch(tensor, tensor)
    assert bool(portable_calls) != arch.startswith("gfx9")
