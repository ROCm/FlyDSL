# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Warp load/store policies, logical widths, valid-item scopes, and conversions."""

import pytest
from coop_common import DTYPES, dtype_id
from coop_test_utils import ARCHES, dtype_entry, run_kernel, warp_indices
from coop_test_utils import coop_default_device as coop_default_device
from coop_test_utils import warp_default_device as warp_default_device

import flydsl.expr as fx
from flydsl.extension.coop._values import _as_items, _items_dtype
from flydsl.extension.coop.warp.load import WarpLoadAlgorithm
from flydsl.extension.coop.warp.store import WarpStoreAlgorithm

try:
    import torch
except ImportError:
    torch = None
import coop_warp_utils as checks


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("policy", ["warp", "warp_striped"])
@pytest.mark.parametrize("valid", [None, 0, 1, 95, 96])
@pytest.mark.usefixtures("warp_default_device")
def test_guarded_io(
    policy, valid, block=32, count=3, entry=(fx.Int32, "int32"), universal=False, width=8, dynamic=False
):
    checks.check_guarded_io(policy, valid, block, count, entry, universal, width, dynamic)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("policy", ["warp", "warp_striped"])
@pytest.mark.parametrize("valid", [0, 1, 95, 96])
@pytest.mark.usefixtures("coop_default_device")
def test_runtime_valid_items(policy, valid):
    checks.check_guarded_io(policy, valid, dynamic=True)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("policy", list(WarpLoadAlgorithm))
@pytest.mark.parametrize("valid", [0, 1, 9, 31, 32])
def test_warp_valid_items_are_per_warp(policy, valid):
    block, width, count = 32, 8, 4
    size = block * count
    source = torch.arange(size, dtype=torch.int32, device="cuda")

    def apply(a, out):
        warp_offset = (fx.thread_idx.x // width) * width * count
        items = fx.coop.WarpLoad[a.dtype, width, count, policy].load(
            a, offset=warp_offset, valid_items=fx.Int32(valid), default=-7
        )
        fx.coop.WarpStore[_items_dtype(items), width, len(_as_items(items)), WarpStoreAlgorithm[policy.name]].store(
            out, items, offset=warp_offset, valid_items=fx.Int32(valid)
        )
        for i in range(count):
            out[size + fx.thread_idx.x * count + i] = items[i]

    actual = run_kernel(apply, source, size * 2, block)
    host = source.cpu()
    mask = torch.arange(size, device="cpu") % (width * count) < valid
    stored = torch.where(mask, host, -99)
    loaded = torch.where(mask, host, -7)
    if policy is WarpLoadAlgorithm.STRIPED:
        loaded = loaded[warp_indices(block, count, width)]
    torch.testing.assert_close(actual, torch.cat([stored, loaded]))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("warp,policy", [(True, p.name) for p in WarpLoadAlgorithm])
@pytest.mark.parametrize("count", [3, 4])
@pytest.mark.parametrize("valid", [0, 1, 95])
@pytest.mark.usefixtures("coop_default_device")
def test_movement_io_policies(warp, policy, count, valid):
    checks.check_movement_io_policies(warp, policy, count, valid)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("entry", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("count", [1, 9])
@pytest.mark.parametrize("policy", ["warp", "warp_striped"])
@pytest.mark.parametrize("universal", [False, True])
@pytest.mark.usefixtures("coop_default_device")
def test_io_dtypes(entry, count, policy, universal):
    checks.check_guarded_io(
        policy, 128 * count - 1, block=128, count=count, entry=dtype_entry(entry), universal=universal
    )


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("entry", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("width", [1, 2, 4, 8, 16, 32, None])
@pytest.mark.parametrize("striped", [False, True])
@pytest.mark.usefixtures("coop_default_device")
def test_warp_io_widths(entry, width, striped):
    checks.check_guarded_io(
        "warp_striped" if striped else "warp", 128 * 3 - 1, block=128, count=3, entry=dtype_entry(entry), width=width
    )


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("case,policy", [("warp_io", policy.name) for policy in WarpLoadAlgorithm])
@pytest.mark.parametrize("count", [3, 4])
@pytest.mark.usefixtures("warp_default_device")
def test_record_movement(case, policy, count):
    checks.check_record_movement(case, policy, count)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("case,policy", [("warp_io", policy.name) for policy in WarpLoadAlgorithm])
def test_compile_record_movement(monkeypatch, arch, case, policy):
    checks.check_compile_record_movement(monkeypatch, arch, case, policy)


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("case", ["warp_io_VECTORIZE", "warp_io_TRANSPOSE"])
def test_compile_movement_policies(monkeypatch, arch, case):
    checks.check_compile_movement_policies(monkeypatch, arch, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    "warp,policy,count", [(True, policy.name, 4) for policy in WarpLoadAlgorithm] + [(True, "VECTORIZE", 3)]
)
@pytest.mark.parametrize(
    "source_dtype,source_name,target_dtype,target_name",
    [
        (fx.Int32, "int32", fx.Int64, "int64"),
        (fx.Float32, "float32", fx.Float64, "float64"),
        (fx.Int64, "int64", fx.Int16, "int16"),
    ],
)
@pytest.mark.parametrize("convert_load", [False, True])
@pytest.mark.parametrize("partial", ["one", "tail"])
@pytest.mark.usefixtures("warp_default_device")
def test_guarded_cross_dtype_io(
    warp, policy, count, source_dtype, source_name, target_dtype, target_name, convert_load, partial
):
    checks.check_guarded_cross_dtype_io(
        warp, policy, count, source_dtype, source_name, target_dtype, target_name, convert_load, partial
    )


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None, reason="requires torch for tensor signatures")
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("case", ["warp_load", "warp_store"])
def test_compile_family(monkeypatch, arch, case):
    checks.check_compile_family(monkeypatch, arch, case)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("policy", list(WarpLoadAlgorithm))
@pytest.mark.parametrize("addressing", ["offset", "view", "shared_source"])
@pytest.mark.parametrize("scope", ["warp", "block", "block_unsigned"])
@pytest.mark.parametrize("universal", [False, True], ids=["dispatched", "universal"])
@pytest.mark.usefixtures("warp_default_device")
def test_tile_addresses(policy, addressing, scope, universal):
    """Caller-selected tiles work across physical warps and a 3D block."""
    width, count, threads = 8, 4, 128
    tile, groups, padding = width * count, threads // width, 5
    # Leave gaps between tiles and permute source tiles. This detects implicit
    # warp offsets even if a load/store round trip would otherwise cancel them.
    pitch = tile + 7
    extent = padding + groups * pitch
    source = torch.arange(extent, dtype=torch.int32, device="cuda") * 7 + 1
    ns = fx.coop.universal if universal else fx.coop

    def apply(a, out):
        tid = fx.thread_idx.x + 4 * fx.thread_idx.y + 16 * fx.thread_idx.z
        group = tid // width
        source_group = 0 if addressing == "shared_source" else groups - 1 - group
        source_offset = padding + source_group * pitch
        destination_offset = padding + group * pitch
        counter = fx.Uint32 if scope == "block_unsigned" else fx.Int32
        # Keep arithmetic signed when deriving a tile count from an unsigned prefix.
        valid = counter(tile - 3 if scope == "warp" else (groups - 2) * tile + 5)
        if scope != "warp":
            valid = fx.min(fx.max(fx.Int32(valid) - group * tile, 0), tile)
        if addressing == "view":
            src = fx.make_view(a.iter + source_offset, fx.make_layout(tile, 1))
            dst = fx.make_view(out.iter + destination_offset, fx.make_layout(tile, 1))
            source_offset = destination_offset = 0
        else:
            src, dst = a, out
        items = ns.WarpLoad[src.dtype, width, count, policy].load(
            src, offset=source_offset, valid_items=valid, default=-7
        )
        ns.WarpStore[_items_dtype(items), width, len(_as_items(items)), WarpStoreAlgorithm[policy.name]].store(
            dst, items, offset=destination_offset, valid_items=valid
        )
        for i in range(count):
            out[extent + tid * count + i] = items[i]

    actual = run_kernel(apply, source, extent + threads * count, (4, 4, 8))
    expected = torch.full_like(actual, -99)
    host = source.cpu()
    for group in range(groups):
        source_group = 0 if addressing == "shared_source" else groups - 1 - group
        src, dst = padding + source_group * pitch, padding + group * pitch
        valid = tile - 3 if scope == "warp" else max(0, min(tile, (groups - 2 - group) * tile + 5))
        expected[dst : dst + valid] = host[src : src + valid]
        loaded = torch.full((tile,), -7, dtype=torch.int32, device="cpu")
        loaded[:valid] = host[src : src + valid]
        if policy is WarpLoadAlgorithm.STRIPED:
            loaded = loaded.reshape(count, width).T.reshape(-1)
        expected[extent + group * tile : extent + (group + 1) * tile] = loaded
    torch.testing.assert_close(actual, expected)
