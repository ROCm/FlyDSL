#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The ROCm override for the batched warp reductions, and the portable tree it displaces.

``coop/warp/rocdl/reduce_batched.py`` replaces the shuffle-xor tree behind
``warp_reduce_batched`` / ``_to_blocked`` / ``_to_striped`` (and ``WarpReduceBatched``)
with DPP moves on gfx9. Three things have to hold for that swap:

- **Same results.** Every case runs ``fx.coop`` and ``fx.coop.universal`` in one
  kernel and requires them to agree *bit for bit* on every owned position, floats
  included: both node shapes fold the same tree, the commutative one only swaps
  the two operands of an exactly commutative IEEE ``add``/``mul``/``maximum`` in
  half the lanes. Both must also agree with a host reference. MAX pins that no
  lane folds a value it must not see.
- **Same ownership.** Scalar / blocked / striped, batch counts below, at and
  above the width, several logical warps per wave and two waves per block.
- **It is the code that runs.** A lone call's ISA has the DPP moves and only the
  cross-row LDS moves the design keeps -- through the functions and through the
  ``WarpReduceBatched`` class -- and everything it does not cover reaches the
  portable tree.

The batched contract itself -- callables, records, tuples, the class and sum
shorthands, width 1 -- lives in ``test_warp_reduce_batched.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from coop_test_utils import matrix_cases
from coop_test_utils import warp_default_device as warp_default_device

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends import current_target

try:
    import torch
except ImportError:
    torch = None

portable = fx.coop.universal
WARP = 64
BLOCK = 2 * WARP  # two waves, so a fold that overran one would show
WIDTHS = (2, 4, 8, 16, 32, 64)
LAYOUTS = ("scalar", "blocked", "striped")
CROSS_LANE_LDS = ("ds_swizzle_b32", "ds_bpermute_b32", "ds_permute_b32")
FN = {
    "scalar": "warp_reduce_batched",
    "blocked": "warp_reduce_batched_to_blocked",
    "striped": "warp_reduce_batched_to_striped",
}


def _is_gfx9():
    return current_target().arch.startswith("gfx9")


requires_gpu = pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
requires_dpp = pytest.mark.skipif(not _is_gfx9(), reason="the DPP tree is gfx9-only")


def affine(lhs, rhs):
    """Composition over F_251, packed as (a << 16) | b; associative, noncommutative."""
    a, b = (lhs >> 16 & 65535, lhs & 65535)
    c, d = (rhs >> 16 & 65535, rhs & 65535)
    return a * c % 251 << 16 | (a * d + b) % 251


# Every node shape the override has: directional (integer ADD/MAX) and
# commutative (integer MUL, every float op). A callable keeps the portable tree.
OPS = {
    "add_i32": (fx.ReductionOp.ADD, "int32"),
    "max_i32": (fx.ReductionOp.MAX, "int32"),
    "mul_i32": (fx.ReductionOp.MUL, "int32"),
    "add_f32": (fx.ReductionOp.ADD, "float32"),
    "mul_f32": (fx.ReductionOp.MUL, "float32"),
    "max_f32": (fx.ReductionOp.MAX, "float32"),
}
# Kernels this big (64 lanes x 64+ columns, every op through both trees) take
# long enough to compile that they only carry one op of each node shape.
BIG_OPS = ("add_i32", "mul_i32", "add_f32")


def slots(width, batches, layout):
    return 1 if layout == "scalar" else -(-batches // width)


def owner(width, batches, layout, lane, slot):
    """The batch (lane, slot) owns, or None."""
    lane %= width
    count = slots(width, batches, layout)
    if layout == "scalar":
        batch = lane
    elif layout == "blocked":
        batch = lane * count + slot
    else:
        batch = slot * width + lane
    return batch if batch < batches else None


def sample(name, n, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if name.endswith("_i32"):
        return torch.randint(-(2**31), 2**31 - 1, (n,), generator=gen, dtype=torch.int32, device="cpu")
    if name == "mul_f32":
        # Keep a 64-way product near 1, so the check is about order, not overflow.
        return torch.rand(n, generator=gen, device="cpu") * 0.4 + 0.8
    return torch.randn(n, generator=gen, device="cpu")


def wrap32(value):
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= 1 << 31 else value


def host_reduce(name, column):
    """Fold one batch column on the host, in ascending lane order."""
    if name == "add_i32":
        return wrap32(int(column.to(torch.int64).sum()))
    if name == "mul_i32":
        acc = 1
        for item in column.tolist():
            acc = wrap32(acc * item)
        return acc
    if name.startswith("max"):
        return column.max().item()
    if name == "add_f32":
        return column.to(torch.float64).sum().item()
    return column.to(torch.float64).prod().item()


def _items(tensor, row, batches):
    return fx.Vector.from_elements([tensor[row * batches + j] for j in range(batches)])


def run_case(width, batches, layout, names, *, sync=False):
    """One kernel: every op in *names* through the dispatched and the portable tree."""
    count = slots(width, batches, layout)
    ints = [n for n in names if OPS[n][1] == "int32"]
    flts = [n for n in names if OPS[n][1] == "float32"]

    def reduce(namespace, value, op):
        out = getattr(namespace, FN[layout])(value, op, width=width, sync_physical_warp=sync)
        return [out] if layout == "scalar" else [out[j] for j in range(count)]

    def body(ai, af, oi, of):
        tid = fx.thread_idx.x
        for group, tensor, out in ((ints, ai, oi), (flts, af, of)):
            for k, name in enumerate(group):
                value = _items(tensor, tid + k * BLOCK, batches)
                for which, namespace in enumerate((fx.coop, portable)):
                    for j, item in enumerate(reduce(namespace, value, OPS[name][0])):
                        out[((k * 2 + which) * BLOCK + tid) * count + j] = item

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(AI: fx.Tensor, AF: fx.Tensor, OI: fx.Tensor, OF: fx.Tensor):
        body(AI, AF, OI, OF)

    @flyc.jit
    def launch(AI: fx.Tensor, AF: fx.Tensor, OI: fx.Tensor, OF: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(AI, AF, OI, OF).launch(grid=(1, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    n = BLOCK * batches
    host = {name: sample(name, n, seed=17 + k) for k, name in enumerate(names)}
    ai = torch.cat([host[name] for name in ints] or [torch.zeros(n, dtype=torch.int32, device="cpu")])
    af = torch.cat([host[name] for name in flts] or [torch.zeros(n, device="cpu")]).float()
    oi = torch.zeros(max(1, len(ints)) * 2 * BLOCK * count, dtype=torch.int32, device="cuda")
    of = torch.zeros(max(1, len(flts)) * 2 * BLOCK * count, dtype=torch.float32, device="cuda")
    ai, af = ai.cuda(), af.cuda()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    launch(ai, af, oi, of, stream=stream)
    torch.cuda.synchronize()
    oi = oi.cpu().reshape(-1, 2, BLOCK, count)
    of = of.cpu().reshape(-1, 2, BLOCK, count)

    checked = 0
    for out, group in ((oi, ints), (of, flts)):
        for k, name in enumerate(group):
            columns = host[name].reshape(BLOCK // width, width, batches)
            fast, ref = out[k, 0], out[k, 1]
            for lane in range(BLOCK):
                for slot in range(count):
                    batch = owner(width, batches, layout, lane, slot)
                    if batch is None:
                        continue
                    got, want = fast[lane, slot], ref[lane, slot]
                    expected = host_reduce(name, columns[lane // width, :, batch])
                    if got.dtype.is_floating_point:
                        # Same tree, exactly commutative ops: identical bits.
                        assert got.view(torch.int32) == want.view(torch.int32), (name, lane, slot, got, want)
                        assert abs(got.item() - expected) <= 1e-4 * max(1.0, abs(expected)), (name, lane, got)
                    else:
                        assert int(got) == int(want) == expected, (name, lane, slot, int(got), int(want), expected)
                    checked += 1
    assert checked > 0


def batches_for(width, layout, kind):
    if kind == "one":
        return 1
    if kind == "half":
        return max(1, width // 2)
    if kind == "over" and layout != "scalar":
        return 2 * width + 1
    return width


# The diagonal visits every width, layout and batch-count kind; the explicit
# rows are the interactions that matter most -- full waves at and above the
# width, and several slots in each distributed layout.
CASES = tuple(
    dict.fromkeys(
        (
            *matrix_cases(WIDTHS, LAYOUTS, ("one", "half", "full", "over")),
            (64, "scalar", "full"),
            (64, "blocked", "over"),
            (16, "striped", "over"),
            (8, "blocked", "over"),
        )
    )
)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@pytest.mark.parametrize("width,layout,kind", CASES)
@pytest.mark.usefixtures("warp_default_device")
def test_override_matches_portable_and_host(width, layout, kind):
    batches = batches_for(width, layout, kind)
    names = list(OPS) if width * batches <= 2048 else list(BIG_OPS)
    run_case(width, batches, layout, names)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@pytest.mark.usefixtures("warp_default_device")
def test_sync_physical_warp_changes_nothing():
    """DPP and ds_swizzle act on the whole wave; only the xor-32 shuffle of a full wave takes a width.

    Only the directional (integer ADD) tree keeps that shuffle at 64 lanes; a
    commutative one is portable there.
    """
    run_case(64, 65, "striped", ["add_i32"], sync=True)


# ── the override is actually the code that runs ────────────────────────────


def _isa(call, torch_dtype, batches, *, tmp_path):
    """Compile ``Out[tid] = call(items)`` once and return its final ISA as text."""

    @flyc.kernel(known_block_size=[WARP, 1, 1])
    def kernel(A: fx.Tensor, Out: fx.Tensor):
        tid = fx.thread_idx.x
        Out[tid] = call(_items(A, tid, batches))

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, Out).launch(grid=(1, 1, 1), block=(WARP, 1, 1), stream=stream)

    values = torch.ones(WARP * batches, dtype=torch_dtype, device="cuda")
    out = torch.zeros(WARP, dtype=torch_dtype, device="cuda")

    keys = ("FLYDSL_DUMP_IR", "FLYDSL_DUMP_DIR", "FLYDSL_RUNTIME_ENABLE_CACHE")
    previous = {key: os.environ.get(key) for key in keys}
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


def _count(isa, *needles):
    return sum(line.count(needle) for line in isa.splitlines() for needle in needles)


# Cross-row LDS moves the override keeps for batches == width: one ds_swizzle per
# stride-16 node and one ds_bpermute for the stride-32 node. Portable: width - 1.
KEPT_LDS = {2: 0, 4: 0, 8: 0, 16: 0, 32: 1, 64: 3}


def _reduce_add(namespace, width):
    return lambda v: namespace.warp_reduce_batched(v, fx.ReductionOp.ADD, width=width)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@requires_dpp
# The diagonal plus the full-wave directional tree (the one with bare DPP moves).
@pytest.mark.parametrize(
    "width,dtype", tuple(dict.fromkeys((*matrix_cases(WIDTHS, ("int32", "float32")), (64, "int32"))))
)
def test_the_fold_keeps_lds_to_the_cross_row_strides(width, dtype, tmp_path):
    torch_dtype = getattr(torch, dtype)
    fast = _isa(_reduce_add(fx.coop, width), torch_dtype, width, tmp_path=tmp_path)

    if dtype != "int32" and width == 64:
        # Commutative nodes stop at 32 lanes: across a full wave their VALU
        # moves cost more throughput than the shorter chain saves.
        assert _count(fast, *CROSS_LANE_LDS) == width - 1, "a full-wave commutative tree should stay portable"
        return
    assert _count(fast, *CROSS_LANE_LDS) == KEPT_LDS[width], "the DPP tree did not take"
    if dtype == "int32":
        # Directional nodes: row_shl/row_shr moves folded into v_add_u32_dpp.
        # (At 64 lanes, with 64 live columns, GCNDPPCombine leaves about a sixth
        # of them as bare v_mov_b32_dpp, hence the margin.)
        assert _count(fast, "row_shl:1") and _count(fast, "row_shr:1")
        bare = _count(fast, "v_mov_b32_dpp")
        folded = sum("_dpp " in line and "v_mov_b32_dpp" not in line for line in fast.splitlines())
        assert bare == 0 if width <= 32 else folded >= 4 * bare, (folded, bare)
    else:
        # Commutative nodes: exact xor moves, quad_perm[1,0,3,2] first.
        assert _count(fast, "quad_perm:[1,0,3,2]")


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@requires_dpp
def test_the_class_reaches_the_override(tmp_path):
    """``WarpReduceBatched`` dispatches like the functions."""
    fast = _isa(
        lambda v: fx.coop.WarpReduceBatched[fx.Float32, 16, 16].reduce(v, fx.ReductionOp.ADD),
        torch.float32,
        16,
        tmp_path=tmp_path,
    )
    assert _count(fast, *CROSS_LANE_LDS) == 0 and _count(fast, "quad_perm:[1,0,3,2]")


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@requires_dpp
def test_an_empty_right_half_folds_without_selects(tmp_path):
    """One batch over 64 lanes: one move and one add per stride, two LDS moves."""
    isa = _isa(_reduce_add(fx.coop, 64), torch.int32, 1, tmp_path=tmp_path)
    assert _count(isa, *CROSS_LANE_LDS) == 2
    assert _count(isa, "v_cndmask_b32") == 0


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@requires_gpu
@requires_dpp
@pytest.mark.parametrize("case", ["int16", "callable"])
def test_what_the_tree_cannot_take_stays_portable(case, tmp_path):
    """DPP moves 32 bits, and a callable needs the ordered portable tree."""
    if case == "int16":
        isa = _isa(_reduce_add(fx.coop, 16), torch.int16, 16, tmp_path=tmp_path)
    else:
        isa = _isa(lambda v: fx.coop.warp_reduce_batched(v, affine, width=16), torch.int32, 16, tmp_path=tmp_path)
    assert "row_shl" not in isa and "quad_perm" not in isa, "the DPP tree took a call it cannot serve"
    assert _count(isa, *CROSS_LANE_LDS) == 15
