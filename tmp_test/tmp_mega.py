#!/usr/bin/env python3
"""Standalone MegaMoE overlap demo.

This file is intentionally outside the production kernel path.  It models the
DeepGEMM-style idea we want to test on MI355X:

1. Dispatch workers publish per-tile arrival counts.
2. GEMM1 workers wait on the tile scoreboard instead of a global dispatch gate.
3. GEMM1 completion publishes an L2/GEMM2 ready mask.
4. GEMM2/combine workers consume ready tiles.

The demo uses CPU threads and sleeps to make the scheduling visible.  It does
not import or modify any FlyDSL production kernels.
"""

from __future__ import annotations

import argparse
import inspect
import os
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Iterable

try:
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    import mori.ir.flydsl as mori_shmem
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm
    from flydsl._mlir.dialects.arith import CmpIPredicate
    from flydsl.expr import T as _fxT
    from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
    from flydsl.expr.typing import T

    _HAS_FLYDSL = True
except Exception:  # noqa: BLE001 - demo must still run on machines without FlyDSL/GPU.
    _HAS_FLYDSL = False


@dataclass(frozen=True)
class Route:
    """One routed token copy after top-k routing."""

    token: int
    expert: int
    slot: int


@dataclass(frozen=True, order=True)
class Tile:
    """A GEMM M-tile in an expert-major fixed-slot layout."""

    expert: int
    tile_idx: int


class CountScoreboard:
    """Release/acquire-style per-tile arrival counter.

    In the GPU kernel, `arrive()` maps to payload stores followed by a release
    atomic, and `wait()` maps to an acquire spin before GEMM reads payload.
    """

    def __init__(self, expected: dict[Tile, int]):
        self.expected = dict(expected)
        self.counts: DefaultDict[Tile, int] = defaultdict(int)
        self._cv = threading.Condition()

    def arrive(self, tile: Tile) -> None:
        with self._cv:
            self.counts[tile] += 1
            self._cv.notify_all()

    def wait(self, tile: Tile) -> None:
        target = self.expected[tile]
        with self._cv:
            while self.counts[tile] < target:
                self._cv.wait()


class MaskScoreboard:
    """Bitmask ready state for GEMM1 -> GEMM2 dependencies."""

    def __init__(self, full_mask: int):
        self.full_mask = int(full_mask)
        self.masks: DefaultDict[Tile, int] = defaultdict(int)
        self._cv = threading.Condition()

    def release_bits(self, tile: Tile, bits: int) -> None:
        with self._cv:
            self.masks[tile] |= bits
            self._cv.notify_all()

    def wait_full(self, tile: Tile) -> None:
        with self._cv:
            while (self.masks[tile] & self.full_mask) != self.full_mask:
                self._cv.wait()


def make_routes(tokens: int, experts: int, topk: int, seed: int) -> list[Route]:
    """Generate deterministic token routes and fixed slots per expert."""

    rng = random.Random(seed)
    next_slot = [0 for _ in range(experts)]
    routes: list[Route] = []
    for token in range(tokens):
        picked = rng.sample(range(experts), k=min(topk, experts))
        for expert in picked:
            slot = next_slot[expert]
            next_slot[expert] += 1
            routes.append(Route(token=token, expert=expert, slot=slot))
    return routes


def expected_l1_tiles(routes: Iterable[Route], tile_m: int) -> dict[Tile, int]:
    """Return the number of payload rows required before each tile can run."""

    counts: DefaultDict[Tile, int] = defaultdict(int)
    for route in routes:
        counts[Tile(route.expert, route.slot // tile_m)] += 1
    return dict(counts)


def expert_wave_schedule(tiles: Iterable[Tile], experts_per_group: int) -> list[Tile]:
    """Simple AMD-friendly CTA/CU scheduler, not wave-in-block specialization."""

    grouped: DefaultDict[int, list[Tile]] = defaultdict(list)
    for tile in sorted(tiles):
        group = tile.expert // experts_per_group
        grouped[group].append(tile)
    schedule: list[Tile] = []
    for group in sorted(grouped):
        schedule.extend(grouped[group])
    return schedule


def dispatch_worker(
    *,
    name: str,
    routes: list[Route],
    tile_m: int,
    l1_ready: CountScoreboard,
    dispatch_delay: float,
    trace: list[str],
    t0: float,
) -> None:
    for route in routes:
        time.sleep(dispatch_delay)
        tile = Tile(route.expert, route.slot // tile_m)
        l1_ready.arrive(tile)
        trace.append(f"{time.perf_counter() - t0:8.4f}s {name}: payload -> {tile}")


def gemm1_worker(
    *,
    name: str,
    tiles: list[Tile],
    l1_ready: CountScoreboard,
    l2_ready: MaskScoreboard,
    n_tiles: int,
    gemm1_delay: float,
    trace: list[str],
    t0: float,
) -> None:
    full_bits = (1 << n_tiles) - 1
    for tile in tiles:
        l1_ready.wait(tile)
        trace.append(f"{time.perf_counter() - t0:8.4f}s {name}: GEMM1 start {tile}")
        time.sleep(gemm1_delay)
        # A real kernel can publish one bit per completed N/K dependency.  The
        # demo publishes the full mask at tile completion.
        l2_ready.release_bits(tile, full_bits)
        trace.append(f"{time.perf_counter() - t0:8.4f}s {name}: GEMM1 done  {tile}")


def gemm2_combine_worker(
    *,
    name: str,
    tiles: list[Tile],
    l2_ready: MaskScoreboard,
    gemm2_delay: float,
    combine_delay: float,
    trace: list[str],
    t0: float,
) -> None:
    for tile in tiles:
        l2_ready.wait_full(tile)
        trace.append(f"{time.perf_counter() - t0:8.4f}s {name}: GEMM2 start {tile}")
        time.sleep(gemm2_delay)
        time.sleep(combine_delay)
        trace.append(f"{time.perf_counter() - t0:8.4f}s {name}: combine done {tile}")


def chunked(items: list[Tile], chunks: int) -> list[list[Tile]]:
    out = [[] for _ in range(chunks)]
    for idx, item in enumerate(items):
        out[idx % chunks].append(item)
    return out


def make_flydsl_overlap_demo_launchers(
    *,
    block_threads: int = 256,
    tile_m: int = 16,
    tiles_per_expert: int = 256,
    n_tile_bits: int = 4,
    simulate_mfma_iters: int = 64,
    compute_mode: str = "fma",
    dispatch_burst: int = 4,
    topk: int = 1,
    token_i32_elems: int = 16,
):
    """Return FlyDSL launchers for the same scoreboard flow.

    This is a demo-only kernel sketch and is deliberately not wired into the
    production MegaMoE path.  It uses flat i32 buffers:

    - `routes`: `[num_routes, 3]` with `(token, expert, slot)`.
    - `l1_ready`: `[experts * tiles_per_expert]` arrival counts.
    - `l1_expected`: same shape, expected payload rows per tile.
    - `l2_ready_mask`: same shape, bitmask published by GEMM1.
    - `done`: `[1]`, incremented once per GEMM2-consumed tile.

    Mapping to a real kernel:

    - `dispatch_publish` is the communication/payload producer.
    - `gemm1_wait_publish` is the GEMM1 tile scheduler.
    - `gemm2_wait_consume` is the GEMM2/combine consumer.
    """

    if not _HAS_FLYDSL:
        raise RuntimeError("FlyDSL is not importable in this environment.")

    full_mask = (1 << int(n_tile_bits)) - 1

    def _to_i64(v):
        return arith.extui(_fxT.i64(), arith.unwrap(v))

    def _byte_addr_i32(base_i64, index_i32):
        i64 = ir.IntegerType.get_signless(64)
        i32 = ir.IntegerType.get_signless(32)
        nuw = ir.Attribute.parse("#llvm.overflow<none>")
        idx = arith.unwrap(index_i32)
        idx64 = llvm.ZExtOp(i64, idx).res if idx.type == i32 else idx
        four = llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, 4)).result
        byte_off = llvm.MulOp(idx64, four, nuw).result
        return llvm.AddOp(arith.unwrap(base_i64), byte_off, nuw).result

    def _atomic_add_agent_i32(base_i64, index_i32, val_i32):
        ptr = llvm.IntToPtrOp(
            llvm.PointerType.get(address_space=1),
            _byte_addr_i32(base_i64, index_i32),
        ).result
        return llvm.AtomicRMWOp(
            llvm.AtomicBinOp.add,
            ptr,
            arith.unwrap(val_i32),
            llvm.AtomicOrdering.monotonic,
            syncscope="agent",
        ).res

    def _readlane0_i32(val_i32):
        return rocdl.readlane(_fxT.i32(), arith.unwrap(val_i32), arith.unwrap(arith.constant(0)))

    def _fence_agent_acquire():
        llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope="agent-one-as")

    def _fence_agent_release():
        llvm.FenceOp(llvm.AtomicOrdering.release, syncscope="agent-one-as")

    def _simulate_compute_work(tile, tid, stage_id, sink_r, sink_offset):
        """Small compute loop that stands in for GEMM work.

        `compute_mode="mfma"` emits real ROCDL MFMA instructions with constant
        f16 fragments.  `compute_mode="fma"` is a scalar fallback that is easier
        to inspect and works on more targets.
        """

        if const_expr(compute_mode == "mfma"):
            # Four packed f16 values in one i64. 0x3c00 is half(1.0),
            # 0x3800 is half(0.5).  This is enough to issue MFMA without
            # staging real A/B tiles yet.
            a_bits = arith.constant(0x3C003C003C003C00, type=T.i64)
            b_bits = arith.constant(0x3800380038003800, type=T.i64)
            a_frag = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [a_bits]))
            b_frag = vector.bitcast(T.f16x4, vector.from_elements(T.vec(1, T.i64), [b_bits]))
            acc_vec = arith.constant_vector(0.0, T.f32x4)
            for _ in range_constexpr(int(simulate_mfma_iters)):
                acc_vec = rocdl.mfma_f32_16x16x16f16(T.f32x4, [a_frag, b_frag, acc_vec, 0, 0, 0])
            acc = vector.extract(acc_vec, static_position=[0], dynamic_position=[])
        else:
            seed_i32 = arith.index_cast(T.i32, tile) + arith.index_cast(T.i32, tid) + arith.constant(stage_id)
            seed = arith.sitofp(T.f32, seed_i32)
            acc = arith.constant(1.0, type=T.f32)
            c0 = arith.constant(1.0009765625, type=T.f32)
            c1 = arith.constant(0.000244140625, type=T.f32)
            for _ in range_constexpr(int(simulate_mfma_iters)):
                acc = arith.addf(arith.mulf(acc, c0), arith.mulf(seed, c1))
        sink_idx_i32 = sink_offset + arith.index_cast(T.i32, tile)
        buffer_ops.buffer_store(acc, sink_r, arith.index_cast(ir.IndexType.get(), sink_idx_i32))

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def dispatch_publish_kernel(
        addr_routes: fx.Int64,
        addr_l1_ready: fx.Int64,
        i32_num_routes: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        gtid = bid * arith.constant(block_threads) + tid
        stride = fx.grid_dim.x * arith.constant(block_threads)
        routes_r = buffer_ops.create_buffer_resource_from_addr(addr_routes)

        for route_idx in range(gtid, i32_num_routes, stride):
            base = route_idx * arith.constant(3)
            expert = buffer_ops.buffer_load(routes_r, base + arith.constant(1), vec_width=1, dtype=T.i32)
            slot = buffer_ops.buffer_load(routes_r, base + arith.constant(2), vec_width=1, dtype=T.i32)
            tile = expert * arith.constant(tiles_per_expert) + slot // arith.constant(tile_m)

            # In production, payload stores must happen before this release.
            _fence_agent_release()
            _atomic_add_agent_i32(addr_l1_ready, tile, arith.constant(1))

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def gemm1_wait_publish_kernel(
        addr_l1_ready: fx.Int64,
        addr_l1_expected: fx.Int64,
        addr_l2_ready_mask: fx.Int64,
        addr_compute_sink: fx.Int64,
        i32_num_tiles: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        gtid = bid * arith.constant(block_threads) + tid
        stride = fx.grid_dim.x * arith.constant(block_threads)
        expected_r = buffer_ops.create_buffer_resource_from_addr(addr_l1_expected)
        l2_r = buffer_ops.create_buffer_resource_from_addr(addr_l2_ready_mask)
        sink_r = buffer_ops.create_buffer_resource_from_addr(addr_compute_sink)

        for tile in range(gtid, i32_num_tiles, stride):
            expected = buffer_ops.buffer_load(expected_r, tile, vec_width=1, dtype=T.i32)
            has_work = arith.cmpi(CmpIPredicate.sgt, expected, arith.constant(0))
            if has_work:
                # Acquire wait for dispatch payload arrival.  This is intentionally
                # tile-granular; token-granular flags would create too much HBM traffic.
                ready_addr = addr_l1_ready + _to_i64(tile) * arith.constant(4, type=T.i64)
                threshold = expected - arith.constant(1)
                mori_shmem.int32_wait_until_greater_than(ready_addr, threshold)
                _fence_agent_acquire()

                _simulate_compute_work(tile, tid, 1, sink_r, arith.constant(0))

                # The demo publishes the full GEMM2 dependency mask once this tile is complete.
                _fence_agent_release()
                buffer_ops.buffer_store(arith.constant(full_mask), l2_r, tile)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def gemm2_wait_consume_kernel(
        addr_l1_expected: fx.Int64,
        addr_l2_ready_mask: fx.Int64,
        addr_done: fx.Int64,
        addr_compute_sink: fx.Int64,
        i32_num_tiles: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        gtid = bid * arith.constant(block_threads) + tid
        stride = fx.grid_dim.x * arith.constant(block_threads)
        expected_r = buffer_ops.create_buffer_resource_from_addr(addr_l1_expected)
        sink_r = buffer_ops.create_buffer_resource_from_addr(addr_compute_sink)

        for tile in range(gtid, i32_num_tiles, stride):
            expected = buffer_ops.buffer_load(expected_r, tile, vec_width=1, dtype=T.i32)
            has_work = arith.cmpi(CmpIPredicate.sgt, expected, arith.constant(0))
            if has_work:
                mask_addr = addr_l2_ready_mask + _to_i64(tile) * arith.constant(4, type=T.i64)
                mori_shmem.int32_wait_until_equals(mask_addr, arith.constant(full_mask))
                _fence_agent_acquire()

                _simulate_compute_work(tile, tid, 2, sink_r, i32_num_tiles)

                # Placeholder for combine completion.
                _atomic_add_agent_i32(addr_done, arith.constant(0), arith.constant(1))

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def fused_overlap_kernel(
        addr_routes: fx.Int64,
        addr_l1_ready: fx.Int64,
        addr_l1_expected: fx.Int64,
        addr_l2_ready_mask: fx.Int64,
        addr_done: fx.Int64,
        addr_compute_sink: fx.Int64,
        i32_num_routes: fx.Int32,
        i32_num_tiles: fx.Int32,
        i32_role_blocks: fx.Int32,
    ):
        """Single-launch scoreboard pipeline.

        Block role assignment is deliberately interleaved for the demo.  With
        `role_blocks = N`, grid has `3N` blocks:

        - block 3g + 0: dispatch publisher for group g
        - block 3g + 1: GEMM1 wait/publish for group g
        - block 3g + 2: GEMM2 wait/consume for group g

        Real MegaMoE would use many persistent CTAs and a scheduler/queue, but
        this validates the important primitive: cross-block producer/consumer
        progress inside one kernel launch without a global barrier.
        """

        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        routes_r = buffer_ops.create_buffer_resource_from_addr(addr_routes)
        expected_r = buffer_ops.create_buffer_resource_from_addr(addr_l1_expected)
        l2_r = buffer_ops.create_buffer_resource_from_addr(addr_l2_ready_mask)
        sink_r = buffer_ops.create_buffer_resource_from_addr(addr_compute_sink)
        role_stride = i32_role_blocks * arith.constant(block_threads)
        role = bid % arith.constant(3)
        role_group = bid // arith.constant(3)

        is_dispatch = arith.cmpi(CmpIPredicate.eq, role, arith.constant(0))
        if is_dispatch:
            start = role_group * arith.constant(block_threads) + tid
            for route_idx in range(start, i32_num_routes, role_stride):
                base = route_idx * arith.constant(3)
                expert = buffer_ops.buffer_load(
                    routes_r, base + arith.constant(1), vec_width=1, dtype=T.i32)
                slot = buffer_ops.buffer_load(
                    routes_r, base + arith.constant(2), vec_width=1, dtype=T.i32)
                tile = expert * arith.constant(tiles_per_expert) + slot // arith.constant(tile_m)
                _fence_agent_release()
                _atomic_add_agent_i32(addr_l1_ready, tile, arith.constant(1))

        is_gemm1 = arith.cmpi(CmpIPredicate.eq, role, arith.constant(1))
        if is_gemm1:
            for tile in range(role_group, i32_num_tiles, i32_role_blocks):
                expected = buffer_ops.buffer_load(expected_r, tile, vec_width=1, dtype=T.i32)
                has_work = arith.cmpi(CmpIPredicate.sgt, expected, arith.constant(0))
                if has_work:
                    ready_addr = addr_l1_ready + _to_i64(tile) * arith.constant(4, type=T.i64)
                    threshold = expected - arith.constant(1)
                    mori_shmem.int32_wait_until_greater_than(ready_addr, threshold)
                    _fence_agent_acquire()
                    _simulate_compute_work(tile, tid, 1, sink_r, arith.constant(0))
                    if tid == 0:
                        _fence_agent_release()
                        buffer_ops.buffer_store(arith.constant(full_mask), l2_r, tile)

        is_gemm2 = arith.cmpi(CmpIPredicate.eq, role, arith.constant(2))
        if is_gemm2:
            for tile in range(role_group, i32_num_tiles, i32_role_blocks):
                expected = buffer_ops.buffer_load(expected_r, tile, vec_width=1, dtype=T.i32)
                has_work = arith.cmpi(CmpIPredicate.sgt, expected, arith.constant(0))
                if has_work:
                    mask_addr = addr_l2_ready_mask + _to_i64(tile) * arith.constant(4, type=T.i64)
                    mori_shmem.int32_wait_until_equals(mask_addr, arith.constant(full_mask))
                    _fence_agent_acquire()
                    _simulate_compute_work(tile, tid, 2, sink_r, i32_num_tiles)
                    if tid == 0:
                        _atomic_add_agent_i32(addr_done, arith.constant(0), arith.constant(1))

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def dynamic_overlap_kernel(
        addr_routes: fx.Int64,
        addr_l1_ready: fx.Int64,
        addr_l1_expected: fx.Int64,
        addr_l2_ready_mask: fx.Int64,
        addr_done: fx.Int64,
        addr_compute_sink: fx.Int64,
        addr_sched: fx.Int64,
        addr_g1_claim: fx.Int64,
        addr_g2_claim: fx.Int64,
        addr_block_claim: fx.Int64,
        i32_num_routes: fx.Int32,
        i32_num_tiles: fx.Int32,
        i32_scheduler_iters: fx.Int32,
    ):
        """Single-launch dynamic worker scheduler.

        Every block runs the same loop:

        1. Threads claim dispatch routes dynamically and publish `l1_ready`.
        2. The block scans a tile candidate.  If it is ready and unclaimed,
           the whole block executes GEMM1 compute and publishes `l2_ready`.
        3. The block scans the same candidate for GEMM2 readiness and consumes
           it if unclaimed.

        No block is permanently assigned to a role; waiting is avoided by
        claiming only tiles whose scoreboard is already ready.
        """

        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        grid_x = fx.grid_dim.x
        gtid = bid * arith.constant(block_threads) + tid
        gstride = grid_x * arith.constant(block_threads)

        routes_r = buffer_ops.create_buffer_resource_from_addr(addr_routes)
        expected_r = buffer_ops.create_buffer_resource_from_addr(addr_l1_expected)
        l1_r = buffer_ops.create_buffer_resource_from_addr(addr_l1_ready)
        l2_r = buffer_ops.create_buffer_resource_from_addr(addr_l2_ready_mask)
        g1_claim_r = buffer_ops.create_buffer_resource_from_addr(addr_g1_claim)
        g2_claim_r = buffer_ops.create_buffer_resource_from_addr(addr_g2_claim)
        block_claim_r = buffer_ops.create_buffer_resource_from_addr(addr_block_claim)
        sink_r = buffer_ops.create_buffer_resource_from_addr(addr_compute_sink)

        for it in range(arith.constant(0), i32_scheduler_iters, arith.constant(1)):
            # Dynamic dispatch: all lanes claim a small burst before doing
            # compute work.  This prevents late dispatch from starving a tile
            # that was scanned before its last payload arrived.
            for _ in range_constexpr(int(dispatch_burst)):
                route_idx = _atomic_add_agent_i32(addr_sched, arith.constant(0), arith.constant(1))
                route_valid = arith.cmpi(CmpIPredicate.slt, route_idx, i32_num_routes)
                if route_valid:
                    base = route_idx * arith.constant(3)
                    expert = buffer_ops.buffer_load(
                        routes_r, base + arith.constant(1), vec_width=1, dtype=T.i32)
                    slot = buffer_ops.buffer_load(
                        routes_r, base + arith.constant(2), vec_width=1, dtype=T.i32)
                    tile = expert * arith.constant(tiles_per_expert) + slot // arith.constant(tile_m)
                    _fence_agent_release()
                    _atomic_add_agent_i32(addr_l1_ready, tile, arith.constant(1))

            # Non-blocking GEMM1 claim: only ready tiles are claimed.
            candidate = (bid + it * grid_x) % i32_num_tiles
            if tid == 0:
                buffer_ops.buffer_store(arith.constant(-1), block_claim_r, bid)
                expected = buffer_ops.buffer_load(expected_r, candidate, vec_width=1, dtype=T.i32)
                has_work = arith.cmpi(CmpIPredicate.sgt, expected, arith.constant(0))
                if has_work:
                    ready = buffer_ops.buffer_load(l1_r, candidate, vec_width=1, dtype=T.i32)
                    ready_ok = arith.cmpi(CmpIPredicate.sge, ready, expected)
                    if ready_ok:
                        old = _atomic_add_agent_i32(addr_g1_claim, candidate, arith.constant(1))
                        won = arith.cmpi(CmpIPredicate.eq, old, arith.constant(0))
                        claim_val = arith.select(won, candidate, arith.constant(-1))
                        buffer_ops.buffer_store(claim_val, block_claim_r, bid)
            gpu.barrier()

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def dynamic_megamoe_dispatch_kernel(
        addr_in_tok: fx.Int64,
        addr_topk_ids: fx.Int64,
        addr_in_wts: fx.Int64,
        addr_running: fx.Int64,
        addr_rx: fx.Int64,
        addr_idx_out: fx.Int64,
        addr_wts_out: fx.Int64,
        addr_srcmap_out: fx.Int64,
        addr_l1_ready: fx.Int64,
        addr_l1_expected: fx.Int64,
        addr_l2_ready_mask: fx.Int64,
        addr_done: fx.Int64,
        addr_compute_sink: fx.Int64,
        addr_sched: fx.Int64,
        addr_g1_claim: fx.Int64,
        addr_g2_claim: fx.Int64,
        addr_block_claim: fx.Int64,
        i32_num_routes: fx.Int32,
        i32_num_tiles: fx.Int32,
        i32_scheduler_iters: fx.Int32,
    ):
        """Dynamic worker scheduler with MegaMoE-style fixed-slot dispatch.

        This replaces the CPU-precomputed `(expert, slot)` route table with the
        same core mechanism used by MegaMoE fixedslot dispatch:

            expert = topk_ids[token * topk + k]
            slot   = atomic_add(running[expert], 1)
            tile   = expert * tiles_per_expert + slot // tile_m

        The demo writes local rx/idx/wts/srcmap buffers instead of P2P peer
        buffers, but the lane/slot structure mirrors the production payload
        loop.
        """

        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        grid_x = fx.grid_dim.x
        block_threads_c = arith.constant(block_threads)
        lane = tid & arith.constant(63)
        warp = tid >> arith.constant(6)
        num_waves = arith.constant(max(1, block_threads // 64))
        global_wave = bid * num_waves + warp
        wave_count = grid_x * num_waves

        tok_r = buffer_ops.create_buffer_resource_from_addr(addr_in_tok)
        topk_r = buffer_ops.create_buffer_resource_from_addr(addr_topk_ids)
        in_wts_r = buffer_ops.create_buffer_resource_from_addr(addr_in_wts)
        rx_r = buffer_ops.create_buffer_resource_from_addr(addr_rx)
        idx_out_r = buffer_ops.create_buffer_resource_from_addr(addr_idx_out)
        wts_out_r = buffer_ops.create_buffer_resource_from_addr(addr_wts_out)
        srcmap_out_r = buffer_ops.create_buffer_resource_from_addr(addr_srcmap_out)
        expected_r = buffer_ops.create_buffer_resource_from_addr(addr_l1_expected)
        l1_r = buffer_ops.create_buffer_resource_from_addr(addr_l1_ready)
        l2_r = buffer_ops.create_buffer_resource_from_addr(addr_l2_ready_mask)
        g1_claim_r = buffer_ops.create_buffer_resource_from_addr(addr_g1_claim)
        g2_claim_r = buffer_ops.create_buffer_resource_from_addr(addr_g2_claim)
        block_claim_r = buffer_ops.create_buffer_resource_from_addr(addr_block_claim)
        sink_r = buffer_ops.create_buffer_resource_from_addr(addr_compute_sink)

        for it in range(arith.constant(0), i32_scheduler_iters, arith.constant(1)):
            for _ in range_constexpr(int(dispatch_burst)):
                route_l0 = arith.constant(0)
                if lane == 0:
                    route_l0 = _atomic_add_agent_i32(addr_sched, arith.constant(0), arith.constant(1))
                route_idx = _readlane0_i32(route_l0)
                route_valid = arith.cmpi(CmpIPredicate.slt, route_idx, i32_num_routes)
                if route_valid:
                    src_tok = route_idx // arith.constant(int(topk))
                    k_slot = route_idx % arith.constant(int(topk))
                    expert = buffer_ops.buffer_load(topk_r, route_idx, vec_width=1, dtype=T.i32)
                    slot_l0 = arith.constant(0)
                    if lane == 0:
                        slot_l0 = _atomic_add_agent_i32(addr_running, expert, arith.constant(1))
                    slot = _readlane0_i32(slot_l0)
                    row = expert * arith.constant(int(tiles_per_expert * tile_m)) + slot
                    tile = row // arith.constant(tile_m)

                    if lane == 0:
                        wt_val = buffer_ops.buffer_load(in_wts_r, route_idx, vec_width=1, dtype=T.f32)
                        src_enc = src_tok | (k_slot << arith.constant(24))
                        buffer_ops.buffer_store(expert, idx_out_r, row)
                        buffer_ops.buffer_store(arith.bitcast(T.i32, wt_val), wts_out_r, row)
                        buffer_ops.buffer_store(src_enc, srcmap_out_r, row)

                    lane_off = lane * arith.constant(4)
                    for co in range(lane_off, arith.constant(int(token_i32_elems)), arith.constant(256)):
                        vals = buffer_ops.buffer_load(
                            tok_r,
                            src_tok * arith.constant(int(token_i32_elems)) + co,
                            vec_width=4,
                            dtype=T.i32,
                        )
                        buffer_ops.buffer_store(
                            vals,
                            rx_r,
                            row * arith.constant(int(token_i32_elems)) + co,
                        )

                    if lane == 0:
                        _fence_agent_release()
                        _atomic_add_agent_i32(addr_l1_ready, tile, arith.constant(1))

            candidate = (bid + it * grid_x) % i32_num_tiles
            if tid == 0:
                buffer_ops.buffer_store(arith.constant(-1), block_claim_r, bid)
                expected = buffer_ops.buffer_load(expected_r, candidate, vec_width=1, dtype=T.i32)
                has_work = arith.cmpi(CmpIPredicate.sgt, expected, arith.constant(0))
                if has_work:
                    ready = buffer_ops.buffer_load(l1_r, candidate, vec_width=1, dtype=T.i32)
                    ready_ok = arith.cmpi(CmpIPredicate.sge, ready, expected)
                    if ready_ok:
                        old = _atomic_add_agent_i32(addr_g1_claim, candidate, arith.constant(1))
                        won = arith.cmpi(CmpIPredicate.eq, old, arith.constant(0))
                        claim_val = arith.select(won, candidate, arith.constant(-1))
                        buffer_ops.buffer_store(claim_val, block_claim_r, bid)
            gpu.barrier()

            claim_tile = buffer_ops.buffer_load(block_claim_r, bid, vec_width=1, dtype=T.i32)
            claim_valid = arith.cmpi(CmpIPredicate.sge, claim_tile, arith.constant(0))
            if claim_valid:
                claim_tile_idx = arith.index_cast(ir.IndexType.get(), claim_tile)
                _fence_agent_acquire()
                _simulate_compute_work(claim_tile_idx, tid, 1, sink_r, arith.constant(0))
                if tid == 0:
                    _fence_agent_release()
                    buffer_ops.buffer_store(arith.constant(full_mask), l2_r, claim_tile_idx)
            gpu.barrier()

            if tid == 0:
                buffer_ops.buffer_store(arith.constant(-1), block_claim_r, bid)
                expected = buffer_ops.buffer_load(expected_r, candidate, vec_width=1, dtype=T.i32)
                has_work = arith.cmpi(CmpIPredicate.sgt, expected, arith.constant(0))
                if has_work:
                    mask = buffer_ops.buffer_load(l2_r, candidate, vec_width=1, dtype=T.i32)
                    ready_ok = arith.cmpi(CmpIPredicate.eq, mask, arith.constant(full_mask))
                    if ready_ok:
                        old = _atomic_add_agent_i32(addr_g2_claim, candidate, arith.constant(1))
                        won = arith.cmpi(CmpIPredicate.eq, old, arith.constant(0))
                        claim_val = arith.select(won, candidate, arith.constant(-1))
                        buffer_ops.buffer_store(claim_val, block_claim_r, bid)
            gpu.barrier()

            claim_tile = buffer_ops.buffer_load(block_claim_r, bid, vec_width=1, dtype=T.i32)
            claim_valid = arith.cmpi(CmpIPredicate.sge, claim_tile, arith.constant(0))
            if claim_valid:
                claim_tile_idx = arith.index_cast(ir.IndexType.get(), claim_tile)
                _fence_agent_acquire()
                _simulate_compute_work(claim_tile_idx, tid, 2, sink_r, i32_num_tiles)
                if tid == 0:
                    _atomic_add_agent_i32(addr_done, arith.constant(0), arith.constant(1))
            gpu.barrier()

            claim_tile = buffer_ops.buffer_load(block_claim_r, bid, vec_width=1, dtype=T.i32)
            claim_valid = arith.cmpi(CmpIPredicate.sge, claim_tile, arith.constant(0))
            if claim_valid:
                claim_tile_idx = arith.index_cast(ir.IndexType.get(), claim_tile)
                _fence_agent_acquire()
                _simulate_compute_work(claim_tile_idx, tid, 1, sink_r, arith.constant(0))
                if tid == 0:
                    _fence_agent_release()
                    buffer_ops.buffer_store(arith.constant(full_mask), l2_r, claim_tile_idx)
            gpu.barrier()

            # Non-blocking GEMM2/Combine claim.
            if tid == 0:
                buffer_ops.buffer_store(arith.constant(-1), block_claim_r, bid)
                expected = buffer_ops.buffer_load(expected_r, candidate, vec_width=1, dtype=T.i32)
                has_work = arith.cmpi(CmpIPredicate.sgt, expected, arith.constant(0))
                if has_work:
                    mask = buffer_ops.buffer_load(l2_r, candidate, vec_width=1, dtype=T.i32)
                    ready_ok = arith.cmpi(CmpIPredicate.eq, mask, arith.constant(full_mask))
                    if ready_ok:
                        old = _atomic_add_agent_i32(addr_g2_claim, candidate, arith.constant(1))
                        won = arith.cmpi(CmpIPredicate.eq, old, arith.constant(0))
                        claim_val = arith.select(won, candidate, arith.constant(-1))
                        buffer_ops.buffer_store(claim_val, block_claim_r, bid)
            gpu.barrier()

            claim_tile = buffer_ops.buffer_load(block_claim_r, bid, vec_width=1, dtype=T.i32)
            claim_valid = arith.cmpi(CmpIPredicate.sge, claim_tile, arith.constant(0))
            if claim_valid:
                claim_tile_idx = arith.index_cast(ir.IndexType.get(), claim_tile)
                _fence_agent_acquire()
                _simulate_compute_work(claim_tile_idx, tid, 2, sink_r, i32_num_tiles)
                if tid == 0:
                    _atomic_add_agent_i32(addr_done, arith.constant(0), arith.constant(1))
            gpu.barrier()

    @flyc.jit
    def dispatch_publish_launch(
        addr_routes: fx.Int64,
        addr_l1_ready: fx.Int64,
        i32_num_routes: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        # Separate demo launchers are useful for codegen experiments.  To overlap
        # on real hardware, launch them on independent streams with correct
        # lifetime ordering, or move the three kernels into one persistent
        # megakernel scheduler.
        dispatch_publish_kernel(addr_routes, addr_l1_ready, i32_num_routes).launch(
            grid=(i32_grid, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def gemm1_wait_publish_launch(
        addr_l1_ready: fx.Int64,
        addr_l1_expected: fx.Int64,
        addr_l2_ready_mask: fx.Int64,
        addr_compute_sink: fx.Int64,
        i32_num_tiles: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        gemm1_wait_publish_kernel(
            addr_l1_ready,
            addr_l1_expected,
            addr_l2_ready_mask,
            addr_compute_sink,
            i32_num_tiles,
        ).launch(
            grid=(i32_grid, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def gemm2_wait_consume_launch(
        addr_l1_expected: fx.Int64,
        addr_l2_ready_mask: fx.Int64,
        addr_done: fx.Int64,
        addr_compute_sink: fx.Int64,
        i32_num_tiles: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        gemm2_wait_consume_kernel(
            addr_l1_expected,
            addr_l2_ready_mask,
            addr_done,
            addr_compute_sink,
            i32_num_tiles,
        ).launch(
            grid=(i32_grid, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def fused_overlap_launch(
        addr_routes: fx.Int64,
        addr_l1_ready: fx.Int64,
        addr_l1_expected: fx.Int64,
        addr_l2_ready_mask: fx.Int64,
        addr_done: fx.Int64,
        addr_compute_sink: fx.Int64,
        i32_num_routes: fx.Int32,
        i32_num_tiles: fx.Int32,
        i32_role_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        fused_overlap_kernel(
            addr_routes,
            addr_l1_ready,
            addr_l1_expected,
            addr_l2_ready_mask,
            addr_done,
            addr_compute_sink,
            i32_num_routes,
            i32_num_tiles,
            i32_role_blocks,
        ).launch(
            grid=(i32_role_blocks * arith.constant(3), 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def dynamic_overlap_launch(
        addr_routes: fx.Int64,
        addr_l1_ready: fx.Int64,
        addr_l1_expected: fx.Int64,
        addr_l2_ready_mask: fx.Int64,
        addr_done: fx.Int64,
        addr_compute_sink: fx.Int64,
        addr_sched: fx.Int64,
        addr_g1_claim: fx.Int64,
        addr_g2_claim: fx.Int64,
        addr_block_claim: fx.Int64,
        i32_num_routes: fx.Int32,
        i32_num_tiles: fx.Int32,
        i32_scheduler_iters: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        dynamic_overlap_kernel(
            addr_routes,
            addr_l1_ready,
            addr_l1_expected,
            addr_l2_ready_mask,
            addr_done,
            addr_compute_sink,
            addr_sched,
            addr_g1_claim,
            addr_g2_claim,
            addr_block_claim,
            i32_num_routes,
            i32_num_tiles,
            i32_scheduler_iters,
        ).launch(
            grid=(i32_grid, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def dynamic_megamoe_dispatch_launch(
        addr_in_tok: fx.Int64,
        addr_topk_ids: fx.Int64,
        addr_in_wts: fx.Int64,
        addr_running: fx.Int64,
        addr_rx: fx.Int64,
        addr_idx_out: fx.Int64,
        addr_wts_out: fx.Int64,
        addr_srcmap_out: fx.Int64,
        addr_l1_ready: fx.Int64,
        addr_l1_expected: fx.Int64,
        addr_l2_ready_mask: fx.Int64,
        addr_done: fx.Int64,
        addr_compute_sink: fx.Int64,
        addr_sched: fx.Int64,
        addr_g1_claim: fx.Int64,
        addr_g2_claim: fx.Int64,
        addr_block_claim: fx.Int64,
        i32_num_routes: fx.Int32,
        i32_num_tiles: fx.Int32,
        i32_scheduler_iters: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        dynamic_megamoe_dispatch_kernel(
            addr_in_tok,
            addr_topk_ids,
            addr_in_wts,
            addr_running,
            addr_rx,
            addr_idx_out,
            addr_wts_out,
            addr_srcmap_out,
            addr_l1_ready,
            addr_l1_expected,
            addr_l2_ready_mask,
            addr_done,
            addr_compute_sink,
            addr_sched,
            addr_g1_claim,
            addr_g2_claim,
            addr_block_claim,
            i32_num_routes,
            i32_num_tiles,
            i32_scheduler_iters,
        ).launch(
            grid=(i32_grid, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return {
        "dispatch_publish": dispatch_publish_launch,
        "gemm1_wait_publish": gemm1_wait_publish_launch,
        "gemm2_wait_consume": gemm2_wait_consume_launch,
        "fused_overlap": fused_overlap_launch,
        "dynamic_overlap": dynamic_overlap_launch,
        "dynamic_megamoe_dispatch": dynamic_megamoe_dispatch_launch,
    }


def print_flydsl_kernel_sketch() -> None:
    """Print the FlyDSL demo code without compiling it."""

    print("FlyDSL overlap demo kernels are defined in:")
    print("  make_flydsl_overlap_demo_launchers(...)")
    print()
    print(inspect.getsource(make_flydsl_overlap_demo_launchers))


def compile_flydsl_kernel_sketch(args: argparse.Namespace) -> None:
    """Compile the FlyDSL demo kernels without loading/running GPU modules."""

    if not _HAS_FLYDSL:
        raise RuntimeError("FlyDSL is not importable in this environment.")
    os.environ["COMPILE_ONLY"] = "1"
    os.environ.setdefault("ARCH", args.arch)

    launchers = make_flydsl_overlap_demo_launchers(
        block_threads=args.block_threads,
        tile_m=args.tile_m,
        tiles_per_expert=args.tiles_per_expert,
        n_tile_bits=args.n_tiles,
        simulate_mfma_iters=args.simulate_mfma_iters,
        compute_mode=args.compute_mode,
        dispatch_burst=args.dispatch_burst,
        topk=args.topk,
        token_i32_elems=args.token_i32_elems,
    )
    stream = fx.Stream(None)

    def i64s(n: int):
        return tuple(fx.Int64(0) for _ in range(n))

    def i32s(n: int):
        return tuple(fx.Int32(1) for _ in range(n))

    compile_args = [
        ("dispatch_publish", i64s(2) + i32s(2) + (stream,)),
        ("gemm1_wait_publish", i64s(4) + i32s(2) + (stream,)),
        ("gemm2_wait_consume", i64s(4) + i32s(2) + (stream,)),
        ("fused_overlap", i64s(6) + i32s(3) + (stream,)),
        ("dynamic_overlap", i64s(10) + i32s(4) + (stream,)),
        ("dynamic_megamoe_dispatch", i64s(17) + i32s(4) + (stream,)),
    ]

    print(f"FlyDSL compile-only demo (ARCH={os.environ.get('ARCH')})")
    for name, launcher_args in compile_args:
        print(f"compile-only {name}", flush=True)
        launchers[name](*launcher_args)
    print("compile-only complete")


def run_flydsl_kernel_demo(args: argparse.Namespace) -> None:
    """Run the FlyDSL scoreboard kernels on real GPU buffers.

    This is still a demo, not production MegaMoE.  The concurrent pass launches
    only a tiny grid by default so consumer spin-waits do not occupy all CUs and
    starve the producer.
    """

    if not _HAS_FLYDSL:
        raise RuntimeError("FlyDSL is not importable in this environment.")
    import torch
    import flydsl.compiler as flyc
    import mori.shmem as ms

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is not available.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    shmem_uid = ms.shmem_get_unique_id()
    shmem_ret = ms.shmem_init_attr(ms.MORI_SHMEM_INIT_WITH_UNIQUEID, 0, 1, shmem_uid)
    if shmem_ret != 0:
        raise RuntimeError(f"mori shmem init failed: {shmem_ret}")
    routes = make_routes(args.tokens, args.experts, args.topk, args.seed)
    expected = expected_l1_tiles(routes, args.tile_m)
    num_tiles = int(args.experts) * int(args.tiles_per_expert)
    if any(tile.tile_idx >= args.tiles_per_expert for tile in expected):
        raise ValueError(
            "tiles_per_expert is too small for generated routes; "
            "raise --tiles-per-expert or lower --tokens/--topk"
        )

    routes_flat = torch.tensor(
        [(r.token, r.expert, r.slot) for r in routes],
        dtype=torch.int32,
        device=device,
    ).reshape(-1)
    topk_flat = torch.tensor([r.expert for r in routes], dtype=torch.int32, device=device)
    token_payload = torch.arange(
        args.tokens * args.token_i32_elems,
        dtype=torch.int32,
        device=device,
    ).reshape(args.tokens, args.token_i32_elems) + 1
    route_wts = torch.full((len(routes),), 1.0 / max(1, args.topk), dtype=torch.float32, device=device)
    l1_expected = torch.zeros(num_tiles, dtype=torch.int32, device=device)
    for tile, count in expected.items():
        l1_expected[tile.expert * args.tiles_per_expert + tile.tile_idx] = count

    l1_ready = torch.zeros_like(l1_expected)
    l2_ready_mask = torch.zeros_like(l1_expected)
    done = torch.zeros(1, dtype=torch.int32, device=device)
    compute_sink = torch.zeros(num_tiles * 2, dtype=torch.float32, device=device)
    running = torch.zeros(args.experts, dtype=torch.int32, device=device)
    max_slots = args.experts * args.tiles_per_expert * args.tile_m
    rx_em = torch.zeros(max_slots * args.token_i32_elems, dtype=torch.int32, device=device)
    idx_em = torch.full((max_slots,), -1, dtype=torch.int32, device=device)
    wts_em = torch.zeros(max_slots, dtype=torch.float32, device=device)
    srcmap_em = torch.zeros(max_slots, dtype=torch.int32, device=device)
    sched = torch.zeros(1, dtype=torch.int32, device=device)
    g1_claim = torch.zeros(num_tiles, dtype=torch.int32, device=device)
    g2_claim = torch.zeros(num_tiles, dtype=torch.int32, device=device)
    block_claim = torch.zeros(args.dynamic_blocks, dtype=torch.int32, device=device)
    active_tiles = int((l1_expected > 0).sum().item())

    launchers = make_flydsl_overlap_demo_launchers(
        block_threads=args.block_threads,
        tile_m=args.tile_m,
        tiles_per_expert=args.tiles_per_expert,
        n_tile_bits=args.n_tiles,
        simulate_mfma_iters=args.simulate_mfma_iters,
        compute_mode=args.compute_mode,
        dispatch_burst=args.dispatch_burst,
        topk=args.topk,
        token_i32_elems=args.token_i32_elems,
    )

    def _args_dispatch(stream):
        return (
            fx.Int64(routes_flat.data_ptr()),
            fx.Int64(l1_ready.data_ptr()),
            fx.Int32(len(routes)),
            fx.Int32(args.flydsl_grid),
            fx.Stream(stream.cuda_stream),
        )

    def _args_gemm1(stream):
        return (
            fx.Int64(l1_ready.data_ptr()),
            fx.Int64(l1_expected.data_ptr()),
            fx.Int64(l2_ready_mask.data_ptr()),
            fx.Int64(compute_sink.data_ptr()),
            fx.Int32(num_tiles),
            fx.Int32(args.flydsl_grid),
            fx.Stream(stream.cuda_stream),
        )

    def _args_gemm2(stream):
        return (
            fx.Int64(l1_expected.data_ptr()),
            fx.Int64(l2_ready_mask.data_ptr()),
            fx.Int64(done.data_ptr()),
            fx.Int64(compute_sink.data_ptr()),
            fx.Int32(num_tiles),
            fx.Int32(args.flydsl_grid),
            fx.Stream(stream.cuda_stream),
        )

    def _args_fused(stream):
        return (
            fx.Int64(routes_flat.data_ptr()),
            fx.Int64(l1_ready.data_ptr()),
            fx.Int64(l1_expected.data_ptr()),
            fx.Int64(l2_ready_mask.data_ptr()),
            fx.Int64(done.data_ptr()),
            fx.Int64(compute_sink.data_ptr()),
            fx.Int32(len(routes)),
            fx.Int32(num_tiles),
            fx.Int32(args.fused_role_blocks),
            fx.Stream(stream.cuda_stream),
        )

    def _args_dynamic(stream):
        return (
            fx.Int64(routes_flat.data_ptr()),
            fx.Int64(l1_ready.data_ptr()),
            fx.Int64(l1_expected.data_ptr()),
            fx.Int64(l2_ready_mask.data_ptr()),
            fx.Int64(done.data_ptr()),
            fx.Int64(compute_sink.data_ptr()),
            fx.Int64(sched.data_ptr()),
            fx.Int64(g1_claim.data_ptr()),
            fx.Int64(g2_claim.data_ptr()),
            fx.Int64(block_claim.data_ptr()),
            fx.Int32(len(routes)),
            fx.Int32(num_tiles),
            fx.Int32(args.dynamic_iters),
            fx.Int32(args.dynamic_blocks),
            fx.Stream(stream.cuda_stream),
        )

    def _args_megamoe_dispatch(stream):
        return (
            fx.Int64(token_payload.data_ptr()),
            fx.Int64(topk_flat.data_ptr()),
            fx.Int64(route_wts.data_ptr()),
            fx.Int64(running.data_ptr()),
            fx.Int64(rx_em.data_ptr()),
            fx.Int64(idx_em.data_ptr()),
            fx.Int64(wts_em.data_ptr()),
            fx.Int64(srcmap_em.data_ptr()),
            fx.Int64(l1_ready.data_ptr()),
            fx.Int64(l1_expected.data_ptr()),
            fx.Int64(l2_ready_mask.data_ptr()),
            fx.Int64(done.data_ptr()),
            fx.Int64(compute_sink.data_ptr()),
            fx.Int64(sched.data_ptr()),
            fx.Int64(g1_claim.data_ptr()),
            fx.Int64(g2_claim.data_ptr()),
            fx.Int64(block_claim.data_ptr()),
            fx.Int32(len(routes)),
            fx.Int32(num_tiles),
            fx.Int32(args.dynamic_iters),
            fx.Int32(args.dynamic_blocks),
            fx.Stream(stream.cuda_stream),
        )

    default_stream = torch.cuda.current_stream()
    print(
        f"FlyDSL runtime demo: routes={len(routes)}, active_tiles={active_tiles}, "
        f"num_tiles={num_tiles}, grid={args.flydsl_grid}, block={args.block_threads}, "
        f"fused_role_blocks={args.fused_role_blocks}, "
        f"compute_mode={args.compute_mode}, compute_iters={args.simulate_mfma_iters}",
        flush=True,
    )

    # First pass: compile and run sequentially.  This validates the kernels and
    # avoids consumer spin while code objects are being JIT-built.
    dispatch_compiled = flyc.compile(launchers["dispatch_publish"], *_args_dispatch(default_stream))
    torch.cuda.synchronize()
    gemm1_compiled = flyc.compile(launchers["gemm1_wait_publish"], *_args_gemm1(default_stream))
    torch.cuda.synchronize()
    gemm2_compiled = flyc.compile(launchers["gemm2_wait_consume"], *_args_gemm2(default_stream))
    torch.cuda.synchronize()
    sequential_done = int(done.cpu().item())
    print(f"sequential done={sequential_done} expected={active_tiles}")
    if sequential_done != active_tiles:
        raise RuntimeError("sequential FlyDSL demo produced wrong done count")

    l1_ready.zero_()
    l2_ready_mask.zero_()
    done.zero_()
    compute_sink.zero_()
    torch.cuda.synchronize()

    replay_t0 = time.perf_counter()
    dispatch_compiled(*_args_dispatch(default_stream))
    gemm1_compiled(*_args_gemm1(default_stream))
    gemm2_compiled(*_args_gemm2(default_stream))
    torch.cuda.synchronize()
    replay_ms = (time.perf_counter() - replay_t0) * 1e3
    replay_done = int(done.cpu().item())
    print(f"sequential replay done={replay_done} expected={active_tiles} wall={replay_ms:.3f} ms")
    if replay_done != active_tiles:
        raise RuntimeError("sequential replay FlyDSL demo produced wrong done count")

    if not args.skip_fixed_fused:
        l1_ready.zero_()
        l2_ready_mask.zero_()
        done.zero_()
        compute_sink.zero_()
        torch.cuda.synchronize()
        fused_compiled = flyc.compile(launchers["fused_overlap"], *_args_fused(default_stream))
        torch.cuda.synchronize()
        fused_done_compile = int(done.cpu().item())
        print(f"fused compile-run done={fused_done_compile} expected={active_tiles}")
        if fused_done_compile != active_tiles:
            raise RuntimeError("fused compile-run FlyDSL demo produced wrong done count")

        l1_ready.zero_()
        l2_ready_mask.zero_()
        done.zero_()
        compute_sink.zero_()
        torch.cuda.synchronize()
        fused_t0 = time.perf_counter()
        fused_compiled(*_args_fused(default_stream))
        torch.cuda.synchronize()
        fused_ms = (time.perf_counter() - fused_t0) * 1e3
        fused_done = int(done.cpu().item())
        sink_sum = float(compute_sink.sum().cpu().item())
        print(
            f"fused replay done={fused_done} expected={active_tiles} "
            f"wall={fused_ms:.3f} ms sink_sum={sink_sum:.3f}"
        )
        if fused_done != active_tiles:
            raise RuntimeError("fused replay FlyDSL demo produced wrong done count")

    if args.run_dynamic_flydsl:
        l1_ready.zero_()
        l2_ready_mask.zero_()
        done.zero_()
        compute_sink.zero_()
        sched.zero_()
        g1_claim.zero_()
        g2_claim.zero_()
        block_claim.zero_()
        torch.cuda.synchronize()

        dynamic_compiled = flyc.compile(launchers["dynamic_overlap"], *_args_dynamic(default_stream))
        torch.cuda.synchronize()
        dynamic_done_compile = int(done.cpu().item())
        print(
            f"dynamic compile-run done={dynamic_done_compile} expected={active_tiles} "
            f"blocks={args.dynamic_blocks} iters={args.dynamic_iters}"
        )
        if dynamic_done_compile != active_tiles:
            missing_g1 = torch.nonzero((l1_expected > 0) & (g1_claim == 0), as_tuple=False).flatten()
            missing_g2 = torch.nonzero((l1_expected > 0) & (g2_claim == 0), as_tuple=False).flatten()
            print(
                "dynamic debug compile-run: "
                f"l1_ready_sum={int(l1_ready.sum().cpu().item())} "
                f"expected_sum={int(l1_expected.sum().cpu().item())} "
                f"g1_claim_nnz={int((g1_claim > 0).sum().cpu().item())} "
                f"g2_claim_nnz={int((g2_claim > 0).sum().cpu().item())} "
                f"missing_g1={missing_g1[:8].cpu().tolist()} "
                f"missing_g2={missing_g2[:8].cpu().tolist()}"
            )
            raise RuntimeError("dynamic compile-run FlyDSL demo produced wrong done count")

        l1_ready.zero_()
        l2_ready_mask.zero_()
        done.zero_()
        compute_sink.zero_()
        sched.zero_()
        g1_claim.zero_()
        g2_claim.zero_()
        block_claim.zero_()
        torch.cuda.synchronize()
        dynamic_t0 = time.perf_counter()
        dynamic_compiled(*_args_dynamic(default_stream))
        torch.cuda.synchronize()
        dynamic_ms = (time.perf_counter() - dynamic_t0) * 1e3
        dynamic_done = int(done.cpu().item())
        dynamic_sink = float(compute_sink.sum().cpu().item())
        print(
            f"dynamic replay done={dynamic_done} expected={active_tiles} "
            f"wall={dynamic_ms:.3f} ms sink_sum={dynamic_sink:.3f}"
        )
        if dynamic_done != active_tiles:
            missing_g1 = torch.nonzero((l1_expected > 0) & (g1_claim == 0), as_tuple=False).flatten()
            missing_g2 = torch.nonzero((l1_expected > 0) & (g2_claim == 0), as_tuple=False).flatten()
            print(
                "dynamic debug replay: "
                f"l1_ready_sum={int(l1_ready.sum().cpu().item())} "
                f"expected_sum={int(l1_expected.sum().cpu().item())} "
                f"g1_claim_nnz={int((g1_claim > 0).sum().cpu().item())} "
                f"g2_claim_nnz={int((g2_claim > 0).sum().cpu().item())} "
                f"missing_g1={missing_g1[:8].cpu().tolist()} "
                f"missing_g2={missing_g2[:8].cpu().tolist()}"
            )
            raise RuntimeError("dynamic replay FlyDSL demo produced wrong done count")

    if args.run_megamoe_dispatch:
        l1_ready.zero_()
        l2_ready_mask.zero_()
        done.zero_()
        compute_sink.zero_()
        running.zero_()
        rx_em.zero_()
        idx_em.fill_(-1)
        wts_em.zero_()
        srcmap_em.zero_()
        sched.zero_()
        g1_claim.zero_()
        g2_claim.zero_()
        block_claim.zero_()
        torch.cuda.synchronize()

        mega_compiled = flyc.compile(
            launchers["dynamic_megamoe_dispatch"], *_args_megamoe_dispatch(default_stream))
        torch.cuda.synchronize()
        mega_done_compile = int(done.cpu().item())
        running_sum = int(running.sum().cpu().item())
        rx_sum = int(rx_em.sum().cpu().item())
        expected_rx_sum = int(token_payload[topk_flat.new_tensor([r.token for r in routes]).long()].sum().cpu().item())
        print(
            f"megamoe-dispatch compile-run done={mega_done_compile} expected={active_tiles} "
            f"running_sum={running_sum} expected_routes={len(routes)} "
            f"rx_sum={rx_sum} expected_rx_sum={expected_rx_sum} "
            f"blocks={args.dynamic_blocks} iters={args.dynamic_iters}"
        )
        if mega_done_compile != active_tiles or running_sum != len(routes) or rx_sum != expected_rx_sum:
            raise RuntimeError("megamoe-dispatch compile-run produced wrong counts")

        l1_ready.zero_()
        l2_ready_mask.zero_()
        done.zero_()
        compute_sink.zero_()
        running.zero_()
        rx_em.zero_()
        idx_em.fill_(-1)
        wts_em.zero_()
        srcmap_em.zero_()
        sched.zero_()
        g1_claim.zero_()
        g2_claim.zero_()
        block_claim.zero_()
        torch.cuda.synchronize()
        mega_t0 = time.perf_counter()
        mega_compiled(*_args_megamoe_dispatch(default_stream))
        torch.cuda.synchronize()
        mega_ms = (time.perf_counter() - mega_t0) * 1e3
        mega_done = int(done.cpu().item())
        running_sum = int(running.sum().cpu().item())
        rx_sum = int(rx_em.sum().cpu().item())
        sink_sum = float(compute_sink.sum().cpu().item())
        print(
            f"megamoe-dispatch replay done={mega_done} expected={active_tiles} "
            f"running_sum={running_sum} expected_routes={len(routes)} "
            f"rx_sum={rx_sum} expected_rx_sum={expected_rx_sum} "
            f"wall={mega_ms:.3f} ms sink_sum={sink_sum:.3f}"
        )
        if mega_done != active_tiles or running_sum != len(routes) or rx_sum != expected_rx_sum:
            raise RuntimeError("megamoe-dispatch replay produced wrong counts")

    if not args.concurrent_flydsl:
        return

    l1_ready.zero_()
    l2_ready_mask.zero_()
    done.zero_()
    compute_sink.zero_()
    torch.cuda.synchronize()

    s_dispatch = torch.cuda.Stream()
    s_gemm1 = torch.cuda.Stream()
    s_gemm2 = torch.cuda.Stream()

    concurrent_t0 = time.perf_counter()
    # Launch consumers first; they spin on scoreboards until producers publish.
    gemm2_compiled(*_args_gemm2(s_gemm2))
    gemm1_compiled(*_args_gemm1(s_gemm1))
    dispatch_compiled(*_args_dispatch(s_dispatch))
    torch.cuda.synchronize()
    concurrent_ms = (time.perf_counter() - concurrent_t0) * 1e3
    concurrent_done = int(done.cpu().item())
    print(f"concurrent done={concurrent_done} expected={active_tiles} wall={concurrent_ms:.3f} ms")
    if concurrent_done != active_tiles:
        raise RuntimeError("concurrent FlyDSL demo produced wrong done count")


def run_overlap(args: argparse.Namespace) -> tuple[float, list[str]]:
    routes = make_routes(args.tokens, args.experts, args.topk, args.seed)
    expected = expected_l1_tiles(routes, args.tile_m)
    tiles = expert_wave_schedule(expected, args.experts_per_group)
    l1_ready = CountScoreboard(expected)
    l2_ready = MaskScoreboard(full_mask=(1 << args.n_tiles) - 1)
    trace: list[str] = []
    threads: list[threading.Thread] = []
    t0 = time.perf_counter()

    route_shards = [[] for _ in range(args.dispatch_workers)]
    for idx, route in enumerate(routes):
        route_shards[idx % args.dispatch_workers].append(route)
    for idx, shard in enumerate(route_shards):
        threads.append(
            threading.Thread(
                target=dispatch_worker,
                kwargs=dict(
                    name=f"dispatch{idx}",
                    routes=shard,
                    tile_m=args.tile_m,
                    l1_ready=l1_ready,
                    dispatch_delay=args.dispatch_delay,
                    trace=trace,
                    t0=t0,
                ),
            )
        )

    for idx, shard in enumerate(chunked(tiles, args.gemm1_workers)):
        threads.append(
            threading.Thread(
                target=gemm1_worker,
                kwargs=dict(
                    name=f"gemm1_{idx}",
                    tiles=shard,
                    l1_ready=l1_ready,
                    l2_ready=l2_ready,
                    n_tiles=args.n_tiles,
                    gemm1_delay=args.gemm1_delay,
                    trace=trace,
                    t0=t0,
                ),
            )
        )

    for idx, shard in enumerate(chunked(tiles, args.gemm2_workers)):
        threads.append(
            threading.Thread(
                target=gemm2_combine_worker,
                kwargs=dict(
                    name=f"gemm2_{idx}",
                    tiles=shard,
                    l2_ready=l2_ready,
                    gemm2_delay=args.gemm2_delay,
                    combine_delay=args.combine_delay,
                    trace=trace,
                    t0=t0,
                ),
            )
        )

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return time.perf_counter() - t0, sorted(trace)


def _next_power_of_two(x: int) -> int:
    return 1 << (max(1, int(x)) - 1).bit_length()


def run_production_megamoe_demo(args: argparse.Namespace) -> None:
    """Run production MegaMoE: real dispatch+GEMM1 and real GEMM2+combine."""

    import sys
    import torch
    import mori.shmem as ms

    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    from tests.kernels.bench_moe_intranode_stage1_groupgemm import _prepare, _chunked_fp4_quant
    from tests.kernels.utils import fp4_utils
    from tests.utils import shuffle_weight
    from kernels.fused_moe_stage1_stage2 import MegaMoE

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is not available.")

    rank = 0
    world = 1
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    shmem_uid = ms.shmem_get_unique_id()
    shmem_ret = ms.shmem_init_attr(ms.MORI_SHMEM_INIT_WITH_UNIQUEID, rank, world, shmem_uid)
    if shmem_ret != 0:
        raise RuntimeError(f"mori shmem init failed: {shmem_ret}")

    tokens = int(args.tokens)
    mtpr = _next_power_of_two(max(tokens, int(args.production_mtpr)))
    model_dim = int(args.production_model_dim)
    inter_dim = int(args.production_inter_dim)
    experts = int(args.experts)
    topk = int(args.topk)
    quant = str(args.quant)

    prep = _prepare(
        dev,
        quant=quant,
        tokens=tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        seed=int(args.seed),
        rank=rank,
        world=world,
        keep_ref=False,
    )

    torch.manual_seed(int(args.seed) + 4242)
    w2_f32 = (
        torch.randn((experts * model_dim, inter_dim), device=dev, dtype=torch.float32)
        * (float(inter_dim) ** -0.25)
    )
    w2_fp4, w2_scale_raw = _chunked_fp4_quant(w2_f32)
    w2_kernel = shuffle_weight(w2_fp4).view(torch.uint8).contiguous().view(-1)
    w2_scale = fp4_utils.e8m0_shuffle(w2_scale_raw).view(torch.uint8).contiguous().view(-1)

    moe = MegaMoE(
        rank=rank,
        world_size=world,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        quant=quant,
        w1=prep["w_kernel"].reshape(-1).contiguous(),
        w1_scale=prep["scale_w1_1d"].reshape(-1).contiguous(),
        w2=w2_kernel,
        w2_scale=w2_scale,
        max_tok_per_rank=mtpr,
        tile_m=int(args.production_tile_m),
        tile_n=int(args.production_tile_n),
        tile_k=int(args.production_tile_k),
        gemm2_tile_m=int(args.production_gemm2_tile_m),
        gemm2_tile_n=int(args.production_gemm2_tile_n),
        gemm2_tile_k=int(args.production_gemm2_tile_k),
        stage2_mode="fused",
    )
    torch.cuda.synchronize()

    x_q = prep["x_payload"][:tokens].contiguous()
    scales = prep["scale_mx_u8"][:tokens].contiguous().view(torch.uint8)
    topk_ids = prep["topk_ids"][:tokens].contiguous()
    wts = prep["wts"][:tokens].contiguous()

    out = moe.forward(x_q, scales, wts, topk_ids)
    torch.cuda.synchronize()
    out_sum = float(out[:tokens].float().sum().cpu().item())
    nv = moe.stage1._nv.detach().cpu().tolist()
    total_recv = int(moe.comb_op.total_recv.cpu().item())
    print(
        "production MegaMoE warmup: "
        f"tokens={tokens}, mtpr={mtpr}, shape=({model_dim},{inter_dim}), "
        f"experts={experts}, topk={topk}, quant={quant}, "
        f"num_valid={nv}, total_recv={total_recv}, out_sum={out_sum:.6f}",
        flush=True,
    )

    t0 = time.perf_counter()
    for _ in range(int(args.production_iters)):
        out = moe.forward(x_q, scales, wts, topk_ids)
    torch.cuda.synchronize()
    ms_per = (time.perf_counter() - t0) * 1e3 / max(1, int(args.production_iters))
    out_sum = float(out[:tokens].float().sum().cpu().item())
    print(f"production MegaMoE replay wall={ms_per:.3f} ms out_sum={out_sum:.6f}", flush=True)


def run_mfma_tile_check(args: argparse.Namespace) -> None:
    """Validate a real 16x16x16 f16 MFMA tile vs torch (single GPU).

    This pins down the MFMA fragment/lane ABI before folding a real GEMM tile
    into the dynamic scheduler.  Layout for v_mfma_f32_16x16x16_f16 (CDNA):
      A[M=16,K=16] row-major, Bt[N=16,K=16] (B transposed so the K-fragment is
      contiguous), D[M=16,N=16] row-major.
      lane l: m=l%16, kgrp=l//16 -> a_frag[i]=A[m, kgrp*4+i]
              n=l%16, kgrp=l//16 -> b_frag[i]=Bt[n, kgrp*4+i]
      D out:  n=l%16, m=(l//16)*4+i -> D[m,n]=acc[i]
    """

    if not _HAS_FLYDSL:
        raise RuntimeError("FlyDSL is not importable in this environment.")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is not available.")
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)

    @flyc.kernel(known_block_size=[64, 1, 1])
    def mfma_tile_kernel(addr_a: fx.Int64, addr_bt: fx.Int64, addr_d: fx.Int64):
        lane = fx.thread_idx.x
        a_r = buffer_ops.create_buffer_resource_from_addr(addr_a)
        bt_r = buffer_ops.create_buffer_resource_from_addr(addr_bt)
        d_r = buffer_ops.create_buffer_resource_from_addr(addr_d)
        m = lane % arith.constant(16)
        n = lane % arith.constant(16)
        kgrp = lane // arith.constant(16)
        a_off = m * arith.constant(16) + kgrp * arith.constant(4)
        b_off = n * arith.constant(16) + kgrp * arith.constant(4)
        a_frag = buffer_ops.buffer_load(a_r, a_off, vec_width=4, dtype=T.f16)
        b_frag = buffer_ops.buffer_load(bt_r, b_off, vec_width=4, dtype=T.f16)
        acc = arith.constant_vector(0.0, T.f32x4)
        acc = rocdl.mfma_f32_16x16x16f16(T.f32x4, [a_frag, b_frag, acc, 0, 0, 0])
        d_m_base = kgrp * arith.constant(4)
        for i in range_constexpr(4):
            val = vector.extract(acc, static_position=[i], dynamic_position=[])
            d_off = (d_m_base + arith.constant(i)) * arith.constant(16) + n
            buffer_ops.buffer_store(val, d_r, d_off)

    @flyc.jit
    def mfma_tile_launch(addr_a: fx.Int64, addr_bt: fx.Int64, addr_d: fx.Int64, stream: fx.Stream):
        mfma_tile_kernel(addr_a, addr_bt, addr_d).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    torch.manual_seed(int(args.seed))
    a = (torch.randn(16, 16, device=dev, dtype=torch.float16))
    b = (torch.randn(16, 16, device=dev, dtype=torch.float16))
    bt = b.t().contiguous()
    d = torch.zeros(16, 16, device=dev, dtype=torch.float32)
    stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
    flyc.compile(mfma_tile_launch, fx.Int64(a.data_ptr()), fx.Int64(bt.data_ptr()),
                 fx.Int64(d.data_ptr()), stream)
    torch.cuda.synchronize()
    ref = (a.float() @ b.float())
    rel = float(((d - ref).norm() / ref.norm().clamp_min(1e-9)).cpu().item())
    print(f"[MFMA tile check] relL2(d, A@B)={rel:.3e}  max_abs_err={(d-ref).abs().max().item():.4e}",
          flush=True)
    print("MFMA tile ABI:", "OK" if rel < 1e-2 else "MISMATCH", flush=True)


def run_copied_megamoe_demo(args: argparse.Namespace) -> None:
    """Run the COPIED (editable) MegaMoE stack: real dispatch+GEMM1 + real GEMM2+combine.

    Stage1 comes from the editable copies under kernels/tmp_mega_*; stage2 reuses the
    production GEMM2+combine op (called, not edited).  Launch with torchrun, e.g.::

        MORI_SHMEM_HEAP_SIZE=40G torchrun --standalone --nproc_per_node=8 \
          tmp_mega.py --run-copied-megamoe --network v4_flash --quant a8w4 --tokens 64
    """

    import sys
    import torch
    import torch.distributed as dist
    import mori.shmem as ms

    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    from tests.kernels.bench_moe_intranode_stage1_groupgemm import (
        _prepare,
        _chunked_fp4_quant,
        _setup_dist,
        NETWORKS,
    )
    from tests.kernels.utils import fp4_utils
    from tests.utils import shuffle_weight
    if getattr(args, "use_megamoe_exp", False):
        from kernels.megamoe_exp import MegaMoEExp as MegaMoE  # experimental single-op (being fused)
    else:
        from kernels.tmp_mega_stage1_stage2 import MegaMoE  # COPIED stack (editable scoreboard)

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = _setup_dist(rank, world, args.master_port)
    dev = torch.device("cuda", local_rank)

    if args.network == "custom":
        model_dim = int(args.production_model_dim)
        inter_dim = int(args.production_inter_dim)
        experts = int(args.experts)
        topk = int(args.topk) if int(args.topk) > 0 else 1
    else:
        net = NETWORKS[args.network]
        model_dim, inter_dim, experts = net["model_dim"], net["inter_dim"], net["experts"]
        topk = int(args.topk) if int(args.topk) > 0 else int(net["topk"])
    run_tokens = max(int(args.tokens), 1)
    if experts % world != 0:
        raise SystemExit(f"experts={experts} must divide world={world}")
    epr = experts // world
    mtpr = max(16, run_tokens)

    T = _prepare(
        dev, quant=args.quant, tokens=run_tokens, model_dim=model_dim,
        inter_dim=inter_dim, experts=experts, topk=topk, seed=int(args.seed),
        rank=rank, world=world, keep_ref=False,
    )
    w_kernel, scale_w1_1d = T["w_kernel"], T["scale_w1_1d"]
    topk_ids, wts, x_bf16 = T["topk_ids"], T["wts"], T["x_bf16"]

    _wpe = w_kernel.numel() // experts
    _spe = scale_w1_1d.numel() // experts
    w1 = w_kernel.reshape(-1)[rank * epr * _wpe:(rank + 1) * epr * _wpe].contiguous()
    w1s = scale_w1_1d.reshape(-1)[rank * epr * _spe:(rank + 1) * epr * _spe].contiguous()

    torch.manual_seed(int(args.seed) + 4242)
    w2_f32 = (torch.randn((experts * model_dim, inter_dim), device=dev, dtype=torch.float32)
              * (float(inter_dim) ** -0.25))
    w2_fp4, w2_sr = _chunked_fp4_quant(w2_f32)
    _sl = slice(rank * epr * model_dim, (rank + 1) * epr * model_dim)
    w2 = shuffle_weight(w2_fp4[_sl]).view(torch.uint8).contiguous().view(-1)
    w2s = fp4_utils.e8m0_shuffle(w2_sr[_sl]).view(torch.uint8).contiguous().view(-1)

    moe = MegaMoE(
        rank=rank, world_size=world, model_dim=model_dim, inter_dim=inter_dim,
        experts=experts, topk=topk, quant=args.quant, w1=w1, w1_scale=w1s,
        w2=w2, w2_scale=w2s, max_tok_per_rank=mtpr, network=args.network,
        stage2_mode="fused",
    )
    torch.cuda.synchronize(); ms.shmem_barrier_all()

    out = moe.forward_bf16(x_bf16[:run_tokens].contiguous(), wts[:run_tokens].contiguous(),
                           topk_ids[:run_tokens].contiguous())
    torch.cuda.synchronize(); ms.shmem_barrier_all()
    out_sum = float(out[:run_tokens].float().sum().cpu().item())
    nv = moe.stage1._nv.detach().cpu().tolist()
    total_recv = int(moe.comb_op.total_recv.cpu().item())
    if rank == 0:
        print(
            "[COPIED MegaMoE] warmup: "
            f"net={args.network} quant={args.quant} bs={run_tokens} topk={topk} "
            f"num_valid={nv} total_recv={total_recv} out_sum={out_sum:.6f}",
            flush=True,
        )
        # Path-2 isolation tool: a2 content checksum is PERMUTATION-INVARIANT (logical vs
        # expert-major layouts write the same valid-row bytes to different positions, padding=0),
        # so any correct a2-layout change (sub-step A) must preserve a2_bytes_sum / a2_scale_sum.
        if os.environ.get("FLYDSL_TMP_DUMP_A2", "0") == "1":
            a2 = moe.stage1._out
            a2s = moe.stage1._osd
            a2_bytes_sum = int(a2.view(torch.uint8).to(torch.int64).sum().cpu().item())
            a2_scale_sum = int(a2s.view(torch.uint8).to(torch.int64).sum().cpu().item())
            a2_nonzero = int((a2.view(torch.uint8) != 0).sum().cpu().item())
            print(
                f"[a2-check] a2_bytes_sum={a2_bytes_sum} a2_scale_sum={a2_scale_sum} "
                f"a2_nonzero_bytes={a2_nonzero} a2_shape={tuple(a2.shape)}",
                flush=True,
            )
        # TMP-COPY scheduler overlap gate diagnostic: compare payload_done vs ll_count cardinality.
        if os.environ.get("FLYDSL_TMP_GATE_DEBUG", "0") == "1":
            pd = moe.stage1.op.payload_done.cpu().tolist()
            llc = moe.stage1.op.ll_count.cpu().tolist()
            mism = [(i, pd[i], llc[i]) for i in range(len(pd)) if pd[i] != llc[i]]
            print(f"[gate-debug] payload_done={pd}", flush=True)
            print(f"[gate-debug] ll_count   ={llc}", flush=True)
            print(f"[gate-debug] mismatches(le,pd,ll)={mism[:16]}  n_mismatch={len(mism)}", flush=True)

    _iters = max(1, int(args.production_iters))
    torch.cuda.synchronize(); ms.shmem_barrier_all()
    t0 = time.perf_counter()
    for _ in range(_iters):
        out = moe.forward_bf16(x_bf16[:run_tokens].contiguous(), wts[:run_tokens].contiguous(),
                               topk_ids[:run_tokens].contiguous())
    torch.cuda.synchronize()
    ms_per = (time.perf_counter() - t0) * 1e3 / _iters
    if rank == 0:
        print(f"[COPIED MegaMoE] replay wall={ms_per:.3f} ms/iter (incl. quant), out_sum={out_sum:.6f}",
              flush=True)

    # TMP-COPY experiment: dump per-phase s_memrealtime timeline (fixedslot non-compact only).
    pts = getattr(moe.stage1, "_pts_buf", None)
    if pts is not None and rank == 0:
        v = pts.cpu().tolist()
        # phase tags emitted by the copied GEMM1 prologue (block0 lane0):
        #  0=dispatch entry  1=after payload write  4=after local arrival
        #  5=after publish done2  2=after cross-PE done  6=after Plan-A recv
        #  7=after postpass meta  3=GEMM start
        t = {k: v[k] for k in (0, 1, 4, 5, 2, 6, 7, 3)}
        # s_memrealtime on gfx950 ticks at 100 MHz -> 1 tick = 10 ns -> ticks/100 = us.
        def us(a, b):
            return (t[b] - t[a]) / 100.0
        print(
            "[COPIED MegaMoE] phase-ts (us, assume 100MHz): "
            f"payload_write={us(0,1):.2f} local_arrival={us(1,4):.2f} "
            f"crossPE_done(xGMI floor)={us(5,2):.2f} planA_recv={us(2,6):.2f} "
            f"postpass_meta={us(6,7):.2f} -> total dispatch->GEMM bubble={us(1,3):.2f}",
            flush=True,
        )

    torch.cuda.synchronize(); dist.barrier()
    try:
        ms.shmem_finalize()
    except Exception:  # noqa: BLE001
        pass
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


def run_strict(args: argparse.Namespace) -> float:
    routes = make_routes(args.tokens, args.experts, args.topk, args.seed)
    tiles = list(expected_l1_tiles(routes, args.tile_m))
    dispatch_rounds = (len(routes) + args.dispatch_workers - 1) // args.dispatch_workers
    gemm1_rounds = (len(tiles) + args.gemm1_workers - 1) // args.gemm1_workers
    gemm2_rounds = (len(tiles) + args.gemm2_workers - 1) // args.gemm2_workers
    return (
        dispatch_rounds * args.dispatch_delay
        + args.global_barrier_delay
        + gemm1_rounds * args.gemm1_delay
        + args.global_barrier_delay
        + gemm2_rounds * (args.gemm2_delay + args.combine_delay)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--tile-m", type=int, default=16)
    parser.add_argument("--tiles-per-expert", type=int, default=256)
    parser.add_argument("--n-tiles", type=int, default=4)
    parser.add_argument("--block-threads", type=int, default=256)
    parser.add_argument("--experts-per-group", type=int, default=2)
    parser.add_argument("--dispatch-workers", type=int, default=4)
    parser.add_argument("--gemm1-workers", type=int, default=4)
    parser.add_argument("--gemm2-workers", type=int, default=4)
    parser.add_argument("--dispatch-delay", type=float, default=0.0004)
    parser.add_argument("--gemm1-delay", type=float, default=0.003)
    parser.add_argument("--gemm2-delay", type=float, default=0.002)
    parser.add_argument("--combine-delay", type=float, default=0.0005)
    parser.add_argument("--global-barrier-delay", type=float, default=0.006)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--show-flydsl",
        action="store_true",
        help="print the FlyDSL kernel sketch and exit without compiling",
    )
    parser.add_argument(
        "--compile-flydsl",
        action="store_true",
        help="compile the FlyDSL kernel sketch with COMPILE_ONLY=1 and exit",
    )
    parser.add_argument(
        "--run-flydsl",
        action="store_true",
        help="run the FlyDSL kernel demo on real GPU buffers",
    )
    parser.add_argument(
        "--run-dynamic-flydsl",
        action="store_true",
        help="also run the single-launch dynamic worker scheduler demo",
    )
    parser.add_argument(
        "--run-megamoe-dispatch",
        action="store_true",
        help="also run the dynamic worker demo with device-side MegaMoE fixedslot dispatch",
    )
    parser.add_argument(
        "--run-production-megamoe",
        action="store_true",
        help="run real production MegaMoE stage1 dispatch+GEMM1 and stage2 GEMM2+combine",
    )
    parser.add_argument(
        "--run-copied-megamoe",
        action="store_true",
        help="run the COPIED editable MegaMoE stack (real dispatch/GEMM1/GEMM2/combine) under torchrun",
    )
    parser.add_argument(
        "--use-megamoe-exp",
        action="store_true",
        help="use kernels.megamoe_exp.MegaMoEExp (the experimental single-op being fused) for --run-copied-megamoe",
    )
    parser.add_argument(
        "--network", default="v4_flash",
        choices=("r1_v3", "v4_flash", "v4_pro", "custom"),
        help="network preset for the copied/production MegaMoE real-case run "
             "('custom' uses --production-model-dim/--production-inter-dim/--experts/--topk)",
    )
    parser.add_argument("--master-port", type=int, default=29951)
    parser.add_argument(
        "--run-mfma-tile-check",
        action="store_true",
        help="validate the real 16x16x16 f16 MFMA fragment ABI vs torch (single GPU)",
    )
    parser.add_argument(
        "--concurrent-flydsl",
        action="store_true",
        help="after sequential validation, launch consumer/producer demo on separate streams",
    )
    parser.add_argument(
        "--flydsl-grid",
        type=int,
        default=1,
        help="grid size for each FlyDSL demo stage; keep small for spin-wait experiments",
    )
    parser.add_argument(
        "--fused-role-blocks",
        type=int,
        default=1,
        help="number of blocks per role in the single-launch fused demo; grid is 3x this value",
    )
    parser.add_argument(
        "--skip-fixed-fused",
        action="store_true",
        help="skip the fixed-role fused demo and only run sequential/dynamic paths",
    )
    parser.add_argument(
        "--dynamic-blocks",
        type=int,
        default=128,
        help="number of persistent worker blocks in the dynamic scheduler demo",
    )
    parser.add_argument(
        "--dynamic-iters",
        type=int,
        default=128,
        help="bounded scheduler loop iterations for the dynamic worker demo",
    )
    parser.add_argument(
        "--dispatch-burst",
        type=int,
        default=4,
        help="number of dispatch route claims each dynamic worker lane attempts per scheduler iteration",
    )
    parser.add_argument(
        "--token-i32-elems",
        type=int,
        default=16,
        help="int32 elements copied per token payload in the MegaMoE dispatch demo",
    )
    parser.add_argument(
        "--simulate-mfma-iters",
        type=int,
        default=64,
        help="number of f32 FMA iterations per active tile in GEMM1/GEMM2 demo regions",
    )
    parser.add_argument(
        "--compute-mode",
        choices=("fma", "mfma"),
        default="fma",
        help="compute placeholder used inside GEMM1/GEMM2 demo regions",
    )
    parser.add_argument("--quant", choices=("a8w4", "a4w4"), default="a8w4")
    parser.add_argument("--production-model-dim", type=int, default=128)
    parser.add_argument("--production-inter-dim", type=int, default=128)
    parser.add_argument("--production-mtpr", type=int, default=0)
    parser.add_argument("--production-iters", type=int, default=5)
    parser.add_argument("--production-tile-m", type=int, default=32)
    parser.add_argument("--production-tile-n", type=int, default=128)
    parser.add_argument("--production-tile-k", type=int, default=256)
    parser.add_argument("--production-gemm2-tile-m", type=int, default=32)
    parser.add_argument("--production-gemm2-tile-n", type=int, default=128)
    parser.add_argument("--production-gemm2-tile-k", type=int, default=256)
    parser.add_argument("--arch", default="gfx950", help="compile-only target arch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.show_flydsl:
        print_flydsl_kernel_sketch()
        return
    if args.compile_flydsl:
        compile_flydsl_kernel_sketch(args)
        return
    if args.run_flydsl:
        run_flydsl_kernel_demo(args)
        return
    if args.run_production_megamoe:
        run_production_megamoe_demo(args)
        return
    if args.run_copied_megamoe:
        run_copied_megamoe_demo(args)
        return
    if args.run_mfma_tile_check:
        run_mfma_tile_check(args)
        return

    strict_estimate = run_strict(args)
    overlap_time, trace = run_overlap(args)
    speedup = strict_estimate / overlap_time if overlap_time > 0 else float("inf")

    print("MegaMoE overlap demo (CPU simulation)")
    print(f"strict-phase estimate : {strict_estimate * 1e3:8.3f} ms")
    print(f"overlap wall time     : {overlap_time * 1e3:8.3f} ms")
    print(f"estimated speedup     : {speedup:8.3f}x")
    print()
    print("Mapping to the production kernel:")
    print("- CountScoreboard ~= l1_tile_ready[e, tile] with release/acquire atomics")
    print("- expert_wave_schedule ~= CTA/CU scheduler, not wave0/wave1 role split")
    print("- MaskScoreboard ~= l2_ready_mask for GEMM1 -> GEMM2 readiness")
    print("- global_barrier_delay ~= strict meta/done barrier bubble to hide")

    if args.trace:
        print()
        print("Trace:")
        for line in trace:
            print(line)


if __name__ == "__main__":
    main()
