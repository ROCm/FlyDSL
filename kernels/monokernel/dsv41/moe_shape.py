# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""K2b's build key and route shape on gfx1250: what a ``moe`` build is for
(``MoeBuild``), the task counts and placements that follow from its routing and
the LDS route table (``RouteShape``), and the ug / down task plan
(``moe_plan``'s functions, on traced Int32 in the kernel and on ints on the
host). Ported from the MI350 mono kernel; wave32: a route lane holds E / 32
experts, an ug task's 16 waves split K in quarters.
"""

from dataclasses import dataclass

import flydsl.expr as fx

from kernels.monokernel.dsv41.attn_pre import TILE, token_tiles
from kernels.monokernel.dsv41.dims import HIDDEN, ROWS, UG_PART, Dims, cdiv
from kernels.monokernel.execution_gfx1250 import BLOCKS, WAVE, WAVES

EXPERTS, TOPK = 384, 6  # the target's routing (a build's: ``MoeBuild``)
ROUTER_PARTS = 4  # contiguous K parts of a router tile, summed in order
ROUTER0 = 8  # the router's first task: past the xq tasks (a token each)
MASK_WORDS = 2  # an expert's token bitmap in the route table: tokens <= 64


@dataclass(frozen=True)
class MoeBuild:
    tokens: int
    experts: int = EXPERTS
    topk: int = TOPK
    tp: int = 4
    timeline: bool = False
    diag_off: int = -1
    # FP4 groups an ug task (``RouteShape.ug_groups``)
    ug_groups: int = 2
    # the router's token groups at most (``RouteShape.router_groups``; 0: as many as
    # one round of CTAs holds)
    router_groups: int = 0
    # HERD routing (aiter ``herd_topk``): route top-(k+1) candidates, each token drops
    # its least batch-popular one (``moe._herd_route``): fewer distinct experts, a
    # different routing than the model's (``reference.route_herd``)
    herd: bool = False


# ---------------------------------------------------------------- the task plan
def imin(a, b):
    return min(a, b) if isinstance(a, int) and isinstance(b, int) else fx.min(a, b)


def pick_tiles(count):
    """A slot's pick tiles: its ``count`` picks, ``TILE`` a tile."""
    return (count + TILE - 1) // TILE


def ug_flag(unit, part, ug_parts):
    """The UGF pair of ug task (``unit``, ``part``)."""
    return unit * ug_parts + part


def tile_pick0(slot_first, tile, tile_first):
    """Pick tile ``tile``'s first pick, its slot's picks from ``slot_first`` and
    tiles from ``tile_first``."""
    return slot_first + (tile - tile_first) * TILE


def polled_tile(tile_first, tiles, j):
    """The pick tile a down round polls as a slot's ``j``-th: the last past them."""
    return tile_first + imin(j, tiles - 1)


# down: |U| x DOWN_TILES units (slot u, 16-row tile r) in expert-major order,
# unit i = u DOWN_TILES + r; CTA b streams the contiguous range [lo, hi) of
# them (every CTA within one unit of the others, the w2 tiles of a range
# contiguous per expert), then combines COMBINE_ROWS rows of every token.
DOWN_TILES = HIDDEN // ROWS  # 320
COMBINE_ROWS = HIDDEN // BLOCKS  # 20
DOWN_SPAN = 3  # the experts a CTA's range touches at most (``check_down_plan``)


def down_units(n_u):
    return n_u * DOWN_TILES


def down_range(bid, n_u):
    """CTA ``bid``'s units [lo, hi) of the |U| ``n_u`` slots' (non-empty: n_u
    DOWN_TILES > BLOCKS)."""
    n = down_units(n_u)
    return bid * n // BLOCKS, (bid + 1) * n // BLOCKS


def down_owner(i, n_u):
    """The CTA whose range holds unit ``i``: the last b with b n / BLOCKS <= i."""
    return (BLOCKS * (i + 1) - 1) // down_units(n_u)


def ug_range(r, n, ctas=BLOCKS):
    """Rotated CTA r's ug units [lo, hi) of ``n`` (unit i = (slot or pick tile
    i / parts, part i % parts), tile-major) over the first ``ctas`` CTAs past the
    stage's base (the rest none): one a CTA while there are fewer than the CTAs,
    else contiguous balanced ranges (a CTA's units share their x tile)."""
    if isinstance(r, int) and isinstance(n, int):
        if r >= ctas:
            return n, n
        if n < ctas:
            return min(r, n), min(r + 1, n)
        return r * n // ctas, (r + 1) * n // ctas
    few = n < ctas
    lo = few.select(fx.min(r, n), r * n // ctas)
    hi = few.select(fx.min(r + 1, n), (r + 1) * n // ctas)
    if ctas < BLOCKS:
        out = r >= ctas
        lo, hi = out.select(n, lo), out.select(n, hi)
    return lo, hi


def combine_tiles(bid):
    """The two 16-row tiles CTA ``bid``'s combine rows [20 bid, + 20) touch."""
    t0 = COMBINE_ROWS * bid // 16
    return t0, t0 + 1


@dataclass(frozen=True)
class RouteShape:
    """A build's routing and the route table ``load_route`` lays out in LDS:
    every token's (ids, weights), |U|, the experts of U, each (t, k)'s slot in U,
    each expert's slot, the picks in slot order and each slot's first position
    (past one token tile also each slot's first pick tile and each pick tile's
    slot), each expert's token bitmap and the queue word."""

    experts: int
    topk: int
    tokens: int
    tp: int
    ug_groups: int = 2
    router_groups_max: int = 0

    @property
    def dims(self) -> Dims:
        return Dims(self.tp)

    @property
    def per_lane(self) -> int:
        """A route lane's experts (lane, lane + 32, ..)."""
        assert self.experts % WAVE == 0
        return self.experts // WAVE

    @property
    def ug_parts(self) -> int:
        """An expert's ug tasks: its real width's FP4 groups, ``ug_groups`` a task."""
        return cdiv(self.dims.inter_real // UG_PART, self.ug_groups)

    @property
    def ug_tiles(self) -> int:
        """An ug task's 16-row w13 tiles: GUGU interleaves gate / up a row, so an
        FP4 group (32 intermediate columns) is 64 rows, 4 tiles."""
        return 4 * self.ug_groups

    @property
    def ug_k_slices(self) -> int:
        """The K slices of an ug task's waves: a wave a (tile, slice)."""
        return WAVES // self.ug_tiles

    @property
    def picks(self) -> int:
        return self.tokens * self.topk

    @property
    def max_u(self) -> int:
        return min(self.picks, self.experts)

    @property
    def router_groups(self) -> tuple:
        """The router's token groups and their tokens: a group a token tile, as
        many as one round of CTAs holds."""
        per_group = self.experts // ROWS * ROUTER_PARTS
        groups = max(1, min(len(token_tiles(self.tokens)), BLOCKS // per_group))
        if self.router_groups_max:
            groups = min(groups, self.router_groups_max)
        tokens = -(-self.tokens // groups)
        assert (groups - 1) * tokens < self.tokens
        return groups, tokens

    @property
    def router_tasks(self) -> int:
        return self.router_groups[0] * self.experts // ROWS * ROUTER_PARTS

    @property
    def router0(self) -> int:
        return max(ROUTER0, self.tokens)

    @property
    def route0(self) -> int:
        return self.router0 + self.router_tasks

    @property
    def shared0(self) -> int:
        return self.route0 + max(8, self.tokens)

    @property
    def shared_units(self) -> int:
        return self.dims.shared_tasks * len(token_tiles(self.tokens))

    @property
    def ug0(self) -> int:
        return self.shared0 + self.shared_units

    @property
    def multi(self) -> bool:
        return self.tokens > TILE

    @property
    def slot_tiles(self) -> int:
        return len(token_tiles(self.tokens))

    @property
    def max_tiles(self) -> int:
        return min(self.picks, self.max_u + self.picks // TILE)

    @property
    def expert_slot0(self) -> int:
        return 3 * self.picks + 1 + self.max_u

    @property
    def order0(self) -> int:
        return self.expert_slot0 + self.experts

    @property
    def first0(self) -> int:
        return self.order0 + self.picks

    @property
    def mask0(self) -> int:
        return self.first0 + self.max_u + 1

    @property
    def queue0(self) -> int:
        return self.mask0 + MASK_WORDS * self.experts

    @property
    def tile_first0(self) -> int:
        return self.queue0 + 1

    @property
    def tile_slot0(self) -> int:
        return self.tile_first0 + self.max_u + 1

    @property
    def tab_words(self) -> int:
        return self.tile_slot0 + self.max_tiles if self.multi else self.queue0 + 1

    @property
    def tile_picks(self) -> int:
        return min(self.tokens, 16)

    @property
    def ug_flags(self) -> int:
        return (self.max_tiles if self.multi else self.max_u) * self.ug_parts

    @property
    def mid_rows(self) -> int:
        """The MID rows a CTA's down range stages in LDS: its <= DOWN_SPAN
        experts' picks (a token picks an expert once)."""
        return min(self.picks, DOWN_SPAN * self.tokens)

    # the route table's readers (LDS ``tab``)
    def route_id(self, tab, t, k):
        return fx.ptr_load(tab + (t * 2 * self.topk + k))

    def route_w(self, tab, t, k):
        return fx.ptr_load(tab + (t * 2 * self.topk + self.topk + k)).bitcast(fx.Float32)

    def n_union(self, tab):
        return fx.ptr_load(tab + 2 * self.picks)

    def union_expert(self, tab, u):
        return fx.ptr_load(tab + (2 * self.picks + 1 + u))

    def slot_first(self, tab, u):
        return fx.ptr_load(tab + (self.first0 + u))

    def pick_at(self, tab, i):
        return fx.ptr_load(tab + (self.order0 + i))

    def tile_first(self, tab, u):
        return fx.ptr_load(tab + (self.tile_first0 + u))

    def tile_slot(self, tab, i):
        return fx.ptr_load(tab + (self.tile_slot0 + i))


# The route table's readers as plain functions: inside a traced ``if`` the AST
# rewriter takes an object whose method is called there for branch state (and
# fails on it), a called module-level function it leaves alone.
def route_w(rs, tab, t, k):
    return rs.route_w(tab, t, k)


def n_union(rs, tab):
    return rs.n_union(tab)


def union_expert(rs, tab, u):
    return rs.union_expert(tab, u)


def slot_first(rs, tab, u):
    return rs.slot_first(tab, u)


def pick_at(rs, tab, i):
    return rs.pick_at(tab, i)


def tile_first(rs, tab, u):
    return rs.tile_first(tab, u)


def tile_slot(rs, tab, i):
    return rs.tile_slot(tab, i)


def route_shape(key: MoeBuild) -> RouteShape:
    return RouteShape(key.experts, key.topk, key.tokens, key.tp, key.ug_groups, key.router_groups)
