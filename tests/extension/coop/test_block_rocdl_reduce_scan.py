#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx9 BlockReduce / BlockScan: the cross-warp phase combined across lanes with DPP.

``fx.coop.BlockReduce`` / ``fx.coop.BlockScan`` dispatch to ``block/rocdl``; each
kernel below runs the dispatched and the ``fx.coop.universal`` collective on the
same input. Integers must agree bit for bit and with the host; floating-point
sums are regrouped (both policies allow it), so they are checked against a
float64 host sum and for block-wide agreement instead.

Covered: the warp counts that pick each sequence (4-8 warps: butterflies,
9-16: row scan + ``ds_bpermute``, non-power-of-two counts padded with the
identity), the counts that keep the portable phase, guarded / callback scans,
the arrival-order policy, 64-bit and float dtypes, and results used as a
subtrahend (a DPP move folded into the caller's subtraction becomes
``v_subrev_u32_dpp``, which gfx942 computes wrongly). The ISA tests pin that
the DPP phase is the code that runs.
"""

from __future__ import annotations

import pytest
from coop_test_utils import coop_default_device as coop_default_device

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends import current_target

try:
    import torch
except ImportError:
    torch = None

RA, SA = fx.coop.BlockReduceAlgorithm, fx.coop.BlockScanAlgorithm
ADD, MAX, MIN = fx.ReductionOp.ADD, fx.ReductionOp.MAX, fx.ReductionOp.MIN
TORCH = {"Int32": "int32", "Int64": "int64", "Float32": "float32"}


def _has_dpp_phase():
    return current_target().arch.startswith(("gfx90a", "gfx94", "gfx95"))


requires_gpu = pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
requires_dpp = pytest.mark.skipif(not _has_dpp_phase(), reason="the DPP cross-warp phase needs row_newbcast (gfx9)")


def _torch(dtype):
    return getattr(torch, TORCH[dtype.__name__])


def _sample(dtype, numel, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if _torch(dtype).is_floating_point:
        return (torch.rand(numel, generator=g, dtype=torch.float64, device="cpu") * 4 - 2).to(_torch(dtype))
    return torch.randint(-(1 << 30), 1 << 30, (numel,), generator=g, device="cpu").to(_torch(dtype))


def _wrap(value, dtype):
    bits = torch.iinfo(_torch(dtype)).bits
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def _launch(kernel, block, values, columns, dtype, count):
    @flyc.jit
    def launch(A: fx.Tensor, Fast: fx.Tensor, Ref: fx.Tensor, Cnt: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Fast, Ref, Cnt).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    fast = torch.zeros(block * columns, dtype=_torch(dtype), device="cuda")
    ref = torch.zeros_like(fast)
    cnt = torch.tensor([count], dtype=torch.int32, device="cuda")
    values = values.cuda()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    launch(values, fast, ref, cnt, stream=stream)
    torch.cuda.synchronize()
    return fast.cpu().reshape(block, columns), ref.cpu().reshape(block, columns)


def _items(A, tid, items):
    values = [A[tid * items + i] for i in range(items)]
    return fx.Vector.from_elements(values) if items > 1 else values[0]


# ── BlockReduce ───────────────────────────────────────────────────────────


REDUCE_CASES = (
    # block, items, dtype, algorithm
    (64, 4, fx.Int32, RA.WARP_REDUCTIONS),  # one warp: no cross-warp phase
    (128, 1, fx.Int32, RA.WARP_REDUCTIONS_NONDETERMINISTIC),  # 2 warps: portable phase, slot order
    (256, 4, fx.Int32, RA.WARP_REDUCTIONS),  # 4 warps: the smallest butterfly
    (320, 1, fx.Int32, RA.WARP_REDUCTIONS),  # 5 warps: padded to 8 with the identity
    (512, 3, fx.Float32, RA.WARP_REDUCTIONS),
    (768, 1, fx.Int32, RA.WARP_REDUCTIONS_NONDETERMINISTIC),  # 12 warps
    (1024, 4, fx.Int32, RA.WARP_REDUCTIONS),
    (1024, 3, fx.Int64, RA.WARP_REDUCTIONS),  # 64-bit: the portable phase
)
REDUCE_OPS = (ADD, MAX, MIN)


def _reduce_kernel(dtype, block, items, algorithm):
    guarded = items == 1
    subtract = dtype is fx.Int32
    columns = len(REDUCE_OPS) + (2 if guarded else 0) + (1 if subtract else 0)

    def run(namespace, out, x, tid, count, allocator):
        R = namespace.BlockReduce[dtype, block, algorithm]
        storage = allocator.allocate(R.SharedStorage).peek()
        for j, op in enumerate(REDUCE_OPS):
            out[tid * columns + j] = R(x, op, storage=storage)
            fx.barrier()
        column = len(REDUCE_OPS)
        if guarded:
            out[tid * columns + column] = R(x, ADD, storage=storage, valid_items=37)
            fx.barrier()
            out[tid * columns + column + 1] = R(x, MAX, storage=storage, valid_items=count)
            fx.barrier()
            column += 2
        if subtract:
            first = x if items == 1 else x[0]
            out[tid * columns + column] = (first ^ 0x5A5A) - R(x, ADD, storage=storage)
            fx.barrier()

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Fast: fx.Tensor, Ref: fx.Tensor, Cnt: fx.Tensor):
        allocator = fx.SharedAllocator()
        tid = fx.thread_idx.x
        x = _items(A, tid, items)
        run(fx.coop, Fast, x, tid, Cnt[0], allocator)
        run(fx.coop.universal, Ref, x, tid, Cnt[0], allocator)

    return kernel, columns, guarded, subtract


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@pytest.mark.parametrize("block,items,dtype,algorithm", REDUCE_CASES, ids=lambda v: getattr(v, "name", str(v)))
@pytest.mark.usefixtures("coop_default_device")
def test_reduce_matches_universal_and_host(block, items, dtype, algorithm):
    values = _sample(dtype, block * items, seed=block + items)
    kernel, columns, guarded, subtract = _reduce_kernel(dtype, block, items, algorithm)
    count = block - 5
    fast, ref = _launch(kernel, block, values, columns, dtype, count)
    host = values.to(torch.float64 if _torch(dtype).is_floating_point else torch.int64)

    assert torch.equal(fast[:, : len(REDUCE_OPS)], fast[0, : len(REDUCE_OPS)].expand(block, -1))
    expected = [host.sum().item(), host.max().item(), host.min().item()]
    if guarded:
        expected += [host[:37].sum().item(), host[:count].max().item()]
    if _torch(dtype).is_floating_point:
        for j, want in enumerate(expected):
            assert fast[0, j].item() == pytest.approx(want, rel=1e-5, abs=1e-3)
            assert fast[0, j].item() == pytest.approx(ref[0, j].item(), rel=1e-5, abs=1e-3)
        return
    for j, want in enumerate(expected):
        assert fast[0, j].item() == _wrap(int(want), dtype), j
    if subtract:
        firsts = values.view(block, items)[:, 0].to(torch.int64)
        want = [_wrap((int(f) ^ 0x5A5A) - int(host.sum()), dtype) for f in firsts]
        assert fast[:, -1].tolist() == want
    assert torch.equal(fast, ref)


# ── BlockScan ─────────────────────────────────────────────────────────────


SCAN_CASES = (
    # block, items, dtype
    (256, 4, fx.Int32),  # 4 warps: the portable phase
    (512, 4, fx.Int32),  # 8 warps: two butterflies
    (512, 1, fx.Float32),
    (768, 3, fx.Int32),  # 12 warps: row scan padded to 16
    (960, 1, fx.Int32),  # 15 warps
    (1024, 1, fx.Int32),
    (1024, 4, fx.Float32),
    (1024, 3, fx.Int64),  # 64-bit: the portable phase
)


def _scan_kernel(dtype, block, items):
    guarded = items == 1
    subtract = dtype is fx.Int32
    # items x (exclusive ADD init=7, inclusive MAX, exclusive MIN) + 2 aggregates
    # [+ guarded exclusive, its aggregate, runtime-guarded MAX, callback exclusive, its aggregate]
    # [+ (x ^ 0x5A5A) - first prefix, (x ^ 0x5A5A) - aggregate]
    columns = 3 * items + 2 + (5 if guarded else 0) + (2 if subtract else 0)

    def callback(aggregate):
        return aggregate + 11

    def run(namespace, out, x, tid, count, allocator):
        S = namespace.BlockScan[dtype, block, SA.WARP_SCANS]
        storage = allocator.allocate(S.SharedStorage).peek()
        base = tid * columns
        y, total = S.exclusive_with_aggregate(x, ADD, storage=storage, init=7)
        fx.barrier()
        z, top = S.inclusive_with_aggregate(x, MAX, storage=storage)
        fx.barrier()
        w = S.exclusive(x, MIN, storage=storage)
        fx.barrier()
        for i in range(items):
            out[base + i] = y[i] if items > 1 else y
            out[base + items + i] = z[i] if items > 1 else z
            out[base + 2 * items + i] = w[i] if items > 1 else w
        out[base + 3 * items] = total
        out[base + 3 * items + 1] = top
        column = 3 * items + 2
        if guarded:
            g, gt = S.exclusive_with_aggregate(x, ADD, storage=storage, valid_items=37)
            fx.barrier()
            h = S.inclusive(x, MAX, storage=storage, valid_items=count)
            fx.barrier()
            c, ct = S.exclusive_with_aggregate(x, ADD, storage=storage, prefix_callback=callback)
            fx.barrier()
            for k, v in enumerate((g, gt, h, c, ct)):
                out[base + column + k] = v
            column += 5
        if subtract:
            first = x if items == 1 else x[0]
            p, a = S.exclusive_with_aggregate(x, ADD, storage=storage)
            fx.barrier()
            out[base + column] = (first ^ 0x5A5A) - (p if items == 1 else p[0])
            out[base + column + 1] = (first ^ 0x5A5A) - a

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Fast: fx.Tensor, Ref: fx.Tensor, Cnt: fx.Tensor):
        allocator = fx.SharedAllocator()
        tid = fx.thread_idx.x
        x = _items(A, tid, items)
        run(fx.coop, Fast, x, tid, Cnt[0], allocator)
        run(fx.coop.universal, Ref, x, tid, Cnt[0], allocator)

    return kernel, columns


def _host_scan(values, op, dtype, init=None, inclusive=True):
    out, acc = [], init
    for v in values.tolist():
        if not inclusive:
            out.append(acc)
        acc = v if acc is None else (acc + v if op is ADD else max(acc, v) if op is MAX else min(acc, v))
        if op is ADD and not _torch(dtype).is_floating_point:
            acc = _wrap(acc, dtype)
        if inclusive:
            out.append(acc)
    return out


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@pytest.mark.parametrize("block,items,dtype", SCAN_CASES, ids=lambda v: getattr(v, "__name__", str(v)))
@pytest.mark.usefixtures("coop_default_device")
def test_scan_matches_universal_and_host(block, items, dtype):
    values = _sample(dtype, block * items, seed=3 * block + items)
    kernel, columns = _scan_kernel(dtype, block, items)
    fast, ref = _launch(kernel, block, values, columns, dtype, block - 5)
    floating = _torch(dtype).is_floating_point

    exclusive = fast[:, :items].reshape(-1)
    want = _host_scan(values, ADD, dtype, init=7, inclusive=False)
    if floating:
        assert torch.allclose(
            exclusive.double(), torch.tensor(want, dtype=torch.float64, device="cpu"), rtol=1e-4, atol=1e-3
        )
    else:
        assert exclusive.tolist() == want
    assert fast[:, items : 2 * items].reshape(-1).tolist() == _host_scan(values, MAX, dtype)
    assert torch.equal(fast[:, 3 * items], fast[0, 3 * items].expand(block)), "aggregate differs between threads"
    assert fast[0, 3 * items + 1].item() == values.max().item()
    if floating:
        assert torch.allclose(fast.double(), ref.double(), rtol=1e-4, atol=1e-3)
        return
    if items == 1:
        base = 3 * items + 2
        guarded = values.clone()
        guarded[37:] = 0
        assert fast[:, base].tolist() == _host_scan(guarded, ADD, dtype, init=0, inclusive=False)
        assert fast[0, base + 1].item() == _wrap(int(values[:37].to(torch.int64).sum()), dtype)
        seed = _wrap(int(values.to(torch.int64).sum()) + 11, dtype)
        assert fast[:, base + 3].tolist() == _host_scan(values, ADD, dtype, init=seed, inclusive=False)
    assert torch.equal(fast, ref)


# ── the DPP phase is the code that runs ───────────────────────────────────


def _isa(body, block, tmp_path):
    """Instructions of a kernel running *body(A, tid)* once, without labels or comments."""
    import os
    from pathlib import Path

    @flyc.kernel(known_block_size=[block, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor):
        tid = fx.thread_idx.x
        Out[tid] = body(A, tid)

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    values = torch.ones(block * 4, dtype=torch.int32, device="cuda")
    out = torch.zeros(block, dtype=torch.int32, device="cuda")
    keys = ("FLYDSL_DUMP_IR", "FLYDSL_DUMP_DIR", "FLYDSL_RUNTIME_ENABLE_CACHE")
    previous = {k: os.environ.get(k) for k in keys}
    os.environ.update(FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=str(tmp_path), FLYDSL_RUNTIME_ENABLE_CACHE="0")
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        launch(values, out, stream=stream)
        torch.cuda.synchronize()
    finally:
        for key, value in previous.items():
            os.environ.pop(key, None) if value is None else os.environ.update({key: value})
    dumped = list(Path(tmp_path).glob("*/*_final_isa.s"))
    assert dumped, f"no ISA dumped under {tmp_path}"
    lines = []
    for line in dumped[0].read_text().splitlines():
        ins = line.split(";")[0].strip()
        if ins and not ins.startswith((".", "//")) and not ins.endswith(":"):
            lines.append(ins)
            if ins.startswith("s_endpgm"):
                break
    return lines


def _count(lines, *prefixes):
    return sum(line.split()[0].startswith(prefixes) for line in lines)


def _reduce_body(cls, block, algorithm=RA.WARP_REDUCTIONS):
    def body(A, tid):
        R = cls[fx.Int32, block, algorithm]
        storage = fx.SharedAllocator().allocate(R.SharedStorage).peek()
        return R(_items(A, tid, 4), ADD, storage=storage)

    return body


def _scan_body(cls, block, guarded=False):
    def body(A, tid):
        S = cls[fx.Int32, block, SA.WARP_SCANS]
        storage = fx.SharedAllocator().allocate(S.SharedStorage).peek()
        if guarded:
            y, total = S.exclusive_with_aggregate(A[tid], ADD, storage=storage, valid_items=block - 3)
            return y + total
        y, total = S.exclusive_with_aggregate(_items(A, tid, 4), ADD, storage=storage)
        return y[3] + total

    return body


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@requires_dpp
@pytest.mark.parametrize("algorithm", (RA.WARP_REDUCTIONS, RA.WARP_REDUCTIONS_NONDETERMINISTIC), ids=lambda a: a.name)
def test_reduce_reads_one_total_per_lane(algorithm, tmp_path):
    """1024 threads: one ds_read_b32 and a 16-lane butterfly, one barrier, no ticket atomic."""
    isa = _isa(_reduce_body(fx.coop.BlockReduce, 1024, algorithm), 1024, tmp_path)
    assert _count(isa, "ds_read") == _count(isa, "ds_read_b32") == 1
    assert _count(isa, "s_barrier") == 1
    assert _count(isa, "ds_add", "ds_inc") == 0
    assert any("row_ror:8" in line for line in isa)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@requires_dpp
@pytest.mark.parametrize("block,permutes", ((512, 0), (1024, 1)), ids=("butterflies", "row_scan"))
@pytest.mark.parametrize("guarded", (False, True), ids=("plain", "guarded"))
def test_scan_reads_one_total_per_lane(block, permutes, guarded, tmp_path):
    """8 warps: two butterflies (pure VALU); 16 warps: row scan + one ds_bpermute."""
    isa = _isa(_scan_body(fx.coop.BlockScan, block, guarded), block, tmp_path)
    assert _count(isa, "ds_read") == _count(isa, "ds_read_b32") == 1
    assert _count(isa, "ds_bpermute") == permutes
    assert _count(isa, "s_barrier") == 1


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@requires_dpp
@pytest.mark.parametrize("kind,block", (("reduce", 128), ("scan", 256)))
def test_below_the_threshold_is_the_portable_code(kind, block, tmp_path):
    """Too few warps for the DPP phase: the dispatched class compiles to the portable class's code."""
    from flydsl.extension.coop.block import reduce as portable_reduce
    from flydsl.extension.coop.block import scan as portable_scan

    if kind == "reduce":
        dispatched, portable = _reduce_body(fx.coop.BlockReduce, block), _reduce_body(
            portable_reduce.BlockReduce, block
        )
    else:
        dispatched, portable = _scan_body(fx.coop.BlockScan, block), _scan_body(portable_scan.BlockScan, block)
    assert _isa(dispatched, block, tmp_path / "dispatched") == _isa(portable, block, tmp_path / "portable")


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
def test_rocm_dispatches_reduce_and_scan():
    from flydsl.extension.coop.block import rocdl

    assert {"BlockReduce", "BlockScan"} <= set(rocdl.__all__)
    assert fx.coop.BlockReduce[fx.Int32, 256].__mro__[1] is rocdl.BlockReduce
    assert fx.coop.BlockScan[fx.Int32, 256].__mro__[1] is rocdl.BlockScan
    # The universal namespace keeps the portable classes.
    assert rocdl.BlockScan not in fx.coop.universal.BlockScan[fx.Int32, 256].__mro__
