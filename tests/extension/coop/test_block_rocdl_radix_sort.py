# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx9 BlockRadixSort override: dispatched fx.coop vs fx.coop.universal vs a host stable sort."""

from pathlib import Path

import pytest
import torch
from coop_test_utils import run_kernel

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends import current_target

A = fx.coop.BlockRadixSortAlgorithm
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
_KEYS = {
    "int32": (fx.Int32, torch.int32),
    "int64": (fx.Int64, torch.int64),
    "float32": (fx.Float32, torch.float32),
    "int16": (fx.Int16, torch.int16),
}


def _sort(ns, key_name, block, count, keys, values, algorithm=A.RADIX_SCAN, **kw):
    """Sort each block's tile; return CPU (keys, payload, input key - sorted key) in sorted order."""
    key_type = _KEYS[key_name][0]
    striped = kw.get("blocked_to_striped", False)
    tile, grid = block * count, keys.numel() // (block * count)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(K: fx.Tensor, V: fx.Tensor, OK: fx.Tensor, OV: fx.Tensor, OD: fx.Tensor):
        tid = fx.thread_idx.x
        base = fx.block_idx.x * tile
        k = fx.Vector.from_elements([K[base + tid * count + i] for i in range(count)]).to(key_type)
        v = fx.Vector.from_elements([V[base + tid * count + i] for i in range(count)])
        sort = ns.BlockRadixSort[(key_type, fx.Int32), block, count, algorithm]
        storage = fx.SharedAllocator().allocate(sort.SharedStorage).peek()
        sk, sv = sort(k, v, storage=storage, **kw)
        for i in range(count):
            position = tid + i * block if striped else tid * count + i
            OK[base + position] = sk[i]
            OV[base + position] = sv[i]
            # A sorted value subtracted from a register: catches a DPP move folded into
            # the user's subtraction (v_subrev_u32_dpp is miscompiled on gfx942).
            OD[base + position] = (k[i] - sk[i]) if key_name == "int32" else sv[i]

    @flyc.jit
    def launch(
        K: fx.Tensor, V: fx.Tensor, OK: fx.Tensor, OV: fx.Tensor, OD: fx.Tensor, stream: fx.Stream = fx.Stream(None)
    ):
        kernel(K, V, OK, OV, OD).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    ok, ov, od = torch.zeros_like(keys), torch.zeros_like(values), torch.zeros_like(values)
    launch(keys, values, ok, ov, od, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    return ok.cpu(), ov.cpu(), od.cpu()


def _ordered(keys, key_name):
    """Host image of the radix key transform, comparable as int64."""
    if key_name == "float32":
        bits = keys.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        bits = torch.where(keys == 0, torch.zeros_like(bits), bits)  # signed zeros sort equal
        return torch.where(bits >= (1 << 31), bits ^ 0xFFFFFFFF, bits ^ (1 << 31))
    if key_name == "int64":
        return keys.to(torch.int64)
    width = 32 if key_name == "int32" else 16
    return (keys.to(torch.int64) & ((1 << width) - 1)) ^ (1 << (width - 1))


def _host(keys, values, key_name, tile, descending=False, begin_bit=0, end_bit=None, **_):
    ordered = _ordered(keys, key_name)
    if begin_bit or end_bit is not None:
        end = {"int32": 32, "float32": 32, "int16": 16}[key_name] if end_bit is None else end_bit
        ordered = (ordered >> begin_bit) & ((1 << (end - begin_bit)) - 1)
    order = torch.cat(
        [
            start + ordered[start : start + tile].argsort(stable=True, descending=descending)
            for start in range(0, keys.numel(), tile)
        ]
    )
    return keys[order], values[order]


def _keys(key_name, n, seed):
    g = torch.Generator().manual_seed(seed)
    if key_name == "float32":
        pool = torch.tensor([-0.0, 0.0, 1.5, -2.25, float("inf"), -float("inf")])
        rand = (torch.rand(n, generator=g) - 0.5) * 100
        pick = torch.randint(0, 3, (n,), generator=g) == 0
        return torch.where(pick, pool[torch.randint(0, len(pool), (n,), generator=g)], rand)
    info = torch.iinfo(_KEYS[key_name][1])
    wide = torch.randint(info.min, info.max, (n,), generator=g, dtype=torch.int64)
    dup = torch.randint(-3, 4, (n,), generator=g, dtype=torch.int64)
    return torch.where(torch.rand(n, generator=g) < 0.3, dup, wide).to(_KEYS[key_name][1])


CASES = {
    "default_3bit": ("int32", 256, 4, {}),
    "one_bit_ballot": ("int32", 256, 4, {"radix_bits": 1}),
    "16_waves_2bit_descending": ("int32", 1024, 4, {"descending": True}),
    "one_wave_9_items_striped": ("int32", 64, 9, {"blocked_to_striped": True}),
    "192_bit_range_descending": ("int32", 192, 3, {"begin_bit": 3, "end_bit": 17, "descending": True}),
    "subwave_32": ("int32", 32, 1, {}),
    "empty_range_striped": ("int32", 128, 2, {"begin_bit": 5, "end_bit": 5, "blocked_to_striped": True}),
    "int64": ("int64", 256, 4, {}),
    "float32_descending": ("float32", 128, 3, {"descending": True}),
    "int16_wide_request": ("int16", 256, 2, {"radix_bits": 4}),
}


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@GPU
@pytest.mark.parametrize("case", list(CASES))
def test_dispatched_matches_universal_and_host(case):
    key_name, block, count, kw = CASES[case]
    grid = 1 if block * count >= 2048 else 2
    keys = _keys(key_name, block * count * grid, seed=len(case)).cuda()
    values = torch.arange(keys.numel(), dtype=torch.int32).cuda()
    got = _sort(fx.coop, key_name, block, count, keys, values, **kw)
    ref = _sort(fx.coop.universal, key_name, block, count, keys, values, **kw)
    host_keys, host_values = _host(keys.cpu(), values.cpu(), key_name, block * count, **kw)
    assert torch.equal(got[0].view(torch.uint8), ref[0].view(torch.uint8))
    assert torch.equal(got[1], ref[1]) and torch.equal(got[2], ref[2])
    assert torch.equal(got[0].view(torch.uint8), host_keys.view(torch.uint8))
    assert torch.equal(got[1], host_values)


# -- ISA: which path compiles ---------------------------------------------------------------


def _isa(monkeypatch, tmp_path, arch, radix_bits=None, end_bit=32, pairs=True, block=256, count=4):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

    def apply(a, out):
        tid = fx.thread_idx.x
        keys = fx.Vector.from_elements([a[tid * count + i] for i in range(count)])
        sort = fx.coop.BlockRadixSort[(fx.Int32, fx.Int32) if pairs else fx.Int32, block, count]
        storage = fx.SharedAllocator().allocate(sort.SharedStorage).peek()
        kw = dict(storage=storage, end_bit=end_bit, radix_bits=radix_bits)
        if pairs:
            sorted_keys, payload = sort(keys, keys, **kw)
            result = sorted_keys + payload
        else:
            result = sort(keys, **kw)
        for i in range(count):
            out[tid * count + i] = result[i]

    run_kernel(apply, torch.zeros(block * count, dtype=torch.int32), block * count, block, compile_only=True)
    text = sorted(Path(tmp_path).glob("**/*_final_isa.s"))[-1].read_text()
    body = []
    for line in text.splitlines():
        ins = line.strip().split(";")[0].strip()
        if ins and not ins.startswith((".", "//")) and not ins.endswith(":"):
            body.append(ins)
            if ins.startswith("s_endpgm"):
                break
    return body


def _count(body, prefix):
    return sum(ins.startswith(prefix) for ins in body)


@pytest.mark.l1b_target_dialect
def test_isa_gfx942_packed_counter(monkeypatch, tmp_path):
    body = _isa(monkeypatch, tmp_path, "gfx942")
    passes = 11  # 32 bits: ten 3-bit passes and one 2-bit pass
    assert _count(body, "s_barrier") == 2 * passes + 1  # count + scatter barrier per pass, one on return
    assert _count(body, "ds_bpermute_b32") == 4 * passes  # one digit-base gather per item
    assert _count(body, "ds_write2st64_b32") == 4 * passes  # key and payload planes stored together
    assert _count(body, "ds_read_b128") == 2 * passes
    assert _count(body, "v_mbcnt") == 0


@pytest.mark.l1b_target_dialect
def test_isa_gfx942_one_bit_ballots(monkeypatch, tmp_path):
    body = _isa(monkeypatch, tmp_path, "gfx942", radix_bits=1, end_bit=8, pairs=False)
    assert _count(body, "s_barrier") == 2 * 8 + 1
    assert _count(body, "s_bcnt1_i32_b64") == 4 * 8  # wave count from the item ballots
    assert _count(body, "v_mbcnt_lo_u32_b32") == 4 * 8
    assert not any("_dpp" in ins for ins in body)  # no block-scan DPP ladder


@pytest.mark.l1b_target_dialect
def test_isa_gfx1201_keeps_portable(monkeypatch, tmp_path):
    body = _isa(monkeypatch, tmp_path, "gfx1201", end_bit=8, pairs=False)
    # Portable: one bit per pass, a scan barrier and an exchange barrier each, one on
    # return; gfx12 splits every barrier into signal + wait.
    assert _count(body, "s_barrier_signal") == 2 * 8 + 1
    assert _count(body, "v_mbcnt") == 0 and _count(body, "s_bcnt1") == 0
    assert current_target().arch == "gfx1201"
    size = fx.coop.BlockRadixSort[fx.Int32, 256, 4].SharedStorage.__dsl_size_of__()
    assert size == fx.coop.universal.BlockRadixSort[fx.Int32, 256, 4].SharedStorage.__dsl_size_of__()


@pytest.mark.l1b_target_dialect
def test_override_is_selected(monkeypatch):
    from flydsl.extension.coop.block.rocdl.radix_sort import BlockRadixSort as Override

    monkeypatch.setenv("ARCH", "gfx942")
    assert issubclass(fx.coop.BlockRadixSort[fx.Int32, 256, 4], Override)
    assert not issubclass(fx.coop.universal.BlockRadixSort[fx.Int32, 256, 4], Override)
