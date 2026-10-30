# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Compile-time storage layout and CTA schedule for the GLM-5 MonoKernel."""

from kernels.monokernel.config import (
    HIDDEN,
    INTER,
    KV_LORA,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    QKV_A_ROWS,
    TOP_K,
    V_DIM,
)
from kernels.monokernel.layout import (
    BLOCKS,
    Q_B_TILE,
    QKV_A_TILE,
    ROUTER_TILE,
    ROW_TILE,
    UG_TILE,
    UK_TILE,
    UV_TILE,
    WAVES,
)

INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_Q_ROWS = INDEX_HEADS * INDEX_DIM
INDEX_TILE = 16
INDEX_KEYS_PER_TASK = 64
DCP_SUMMARY_PAIRS = KV_LORA // 2 + 2


def sparse_heads_per_task(
    samples: int, heads: int, npes: int, native_fp4_mfma: bool = False, cohort_rows: bool = False
) -> int:
    """Share the staged KV tile across both MFMA head halves in TP4 cohorts.

    Cohort eligibility already enforces bounded storage and cache ownership.
    Retain the existing standalone S12 path.
    """
    return (
        2 * WAVES if native_fp4_mfma and npes == 4 and (samples == 12 or cohort_rows) and heads == 2 * WAVES else WAVES
    )


def split_acc_head(head_group, lane_group: int, element: int):
    """Map one split-PV MFMA row to its local attention head."""

    return head_group * WAVES + lane_group * 4 + element


def split_score_column(wave: int, lane: int):
    """Locate this wave's head column in a 16x16 MFMA score tile."""

    return wave + 16 * ((lane % 16) // 4)


def fp8_kv_upper_pair_lane(lane):
    return (lane & -4) + 2


def fp8_pe_upper_pair_lane(lane):
    return (lane & -2) + 1


INDEX_CP_TILE = 4096
INDEX_RADIX_DIGITS = 4
INDEX_RADIX_BINS = 256
INDEX_LOCAL_LDS_CAPACITY = 16384
ONE_WORKGROUP_MAX_WIDTH = 20000
SHORT_LDS_CAPACITY = (ONE_WORKGROUP_MAX_WIDTH + 511) // 512 * 512


def index_local_capacity(index_max_seq: int, indexer_quant: bool = False) -> int:
    # The allocated-width branch never uses short-score storage in the long path.
    # Keep the validated v102 long-path LDS and symmetric layout unchanged.
    limit = (
        SHORT_LDS_CAPACITY if indexer_quant and index_max_seq <= ONE_WORKGROUP_MAX_WIDTH else INDEX_LOCAL_LDS_CAPACITY
    )
    return min(index_max_seq, limit)


def index_radix_geometry(indexer_quant: bool) -> tuple[int, int]:
    """Retain BF16 geometry; FP4 uses three wider radix digits."""
    return (3, 2048) if indexer_quant else (INDEX_RADIX_DIGITS, INDEX_RADIX_BINS)


def indexer_uses_tiles(index_max_seq: int, indexer_cp: bool = False, indexer_quant: bool = False) -> bool:
    """Keep the existing short selector; use bounded tiles for larger contexts."""
    return index_max_seq > (ONE_WORKGROUP_MAX_WIDTH if indexer_quant else INDEX_LOCAL_LDS_CAPACITY)


def index_cp_world(npes: int, samples: int, index_max_seq: int) -> int:
    """Use two independent CP4 groups for TP8's single-query medium contexts."""
    if npes == 8 and samples == 1 and INDEX_LOCAL_LDS_CAPACITY < index_max_seq <= 131072:
        return 4
    return npes


def index_cp_capacity(index_max_seq: int, npes: int) -> int:
    """Maximum context shard, rounded to a complete scoring tile."""
    unit = npes * INDEX_KEYS_PER_TASK
    return (index_max_seq + unit - 1) // unit * INDEX_KEYS_PER_TASK


def index_cp_tile(indexer_quant: bool) -> int:
    """Use eighteen keys per lane for FP4; retain BF16's eight-key geometry."""
    return 9216 if indexer_quant else INDEX_CP_TILE


def index_cp_partitions(index_max_seq: int, npes: int, indexer_quant: bool = False) -> int:
    tile = index_cp_tile(indexer_quant)
    return (index_cp_capacity(index_max_seq, npes) + tile - 1) // tile


N_QKV_A = QKV_A_ROWS // QKV_A_TILE
N_ROW_TILES = HIDDEN // ROW_TILE
N_ROUTER = N_EXPERTS // ROUTER_TILE
N_UG_PER_SLOT = INTER // UG_TILE
XQ_BLOCKS = HIDDEN // 128
XQ_GROUPS = HIDDEN // 32
XQ_WAVES = (XQ_BLOCKS + N_ROUTER - 1) // N_ROUTER
assert XQ_WAVES * 4 <= WAVES


def row_tile_size(samples: int, native_fp4_mfma: bool = False) -> int:
    """Bound each resident row tile; preserve released shapes up to twelve."""
    if not native_fp4_mfma or samples <= 12:
        return samples
    return next(tile for tile in range(12, 0, -1) if samples % tile == 0)


def cohort_geometry(samples: int, native_fp4_mfma: bool) -> bool:
    """The CTA mask needs a power-of-two cohort; slot masks hold one wave."""
    tile = row_tile_size(samples, native_fp4_mfma)
    count = samples // tile
    if count < 2 or BLOCKS % count or samples > 64:
        return False
    width = BLOCKS // count
    return width > 0 and width & (width - 1) == 0


def cohort_eligible(samples: int, npes: int, with_indexer: bool, native_fp4_mfma: bool) -> bool:
    """Bound scratch publishers and one deferred commit owner per row."""
    tile = row_tile_size(samples, native_fp4_mfma)
    if not cohort_geometry(samples, native_fp4_mfma):
        return False
    publisher_rows = cache_samples_per_task(tile, npes, False, native_fp4_mfma)
    if with_indexer or publisher_rows == tile:
        return True
    width = BLOCKS // (samples // tile)
    # Each producer publishes disjoint row mailboxes before the front barrier.
    # The final barrier drains readers; the last matching row owns each slot.
    return (
        npes == 4
        and native_fp4_mfma
        and tile % publisher_rows == 0
        and tile // publisher_rows <= width
        and tile <= width
    )


def global_ug_chunks(samples: int, tile: int) -> int:
    """Bound all routed chunks plus the shared expert without negative routes."""
    routes = samples * TOP_K
    experts = min(N_EXPERTS, routes)
    return experts + (routes - experts) // tile + samples // tile


def dn_tile(samples: int, expert_mxfp4: bool = False) -> int:
    """Return rows per expert-down/FFN-reduce task for the tuned schedule."""

    return 32 if samples == 1 or (samples > 4 and expert_mxfp4) else HIDDEN // BLOCKS


def sparse_keys_per_task(samples: int, heads: int = WAVES) -> int:
    """Use narrower sparse tiles when samples or attention heads fill LDS."""

    return 32 if heads > 2 * WAVES or (heads == WAVES and samples > 4) else 64


def sample_wave_batches(samples: int) -> int:
    return (samples + WAVES - 1) // WAVES


NATIVE_UG_TILE = 8


def grouped_ug_enabled(samples: int, npes: int, native_fp4_mfma: bool, cohort_rows: bool = False) -> bool:
    return native_fp4_mfma and ((npes == 4 and samples in (8, 10, 12)) or (cohort_rows and npes == 8 and samples == 10))


def down_prefetch_batch(
    samples: int, native_fp4_mfma: bool = False, atom_experts: bool = False, inter: int = INTER
) -> int:
    if samples <= 4:
        return 9
    if native_fp4_mfma:
        if samples <= 6:
            return 4
        # Wider expert shards have more K units in flight; a shallower pipeline
        # reduces register pressure for large query groups without changing TP8.
        return 2 if inter > INTER and samples in (8, 10, 12) else 3
    return 8 if atom_experts else 4


def down_x_words(samples: int, inter: int, expert_mxfp4: bool, native_fp4_mfma: bool = False) -> int:
    return samples * MOE_SLOTS * inter // (4 if native_fp4_mfma or not expert_mxfp4 else 2)


def dcp_summary_index(source, sample, tile, item, samples, n_uv):
    return ((source * samples + sample) * n_uv + tile) * DCP_SUMMARY_PAIRS + item


def dcp_uv_owner(tile, output_heads, rank):
    return tile // (V_DIM // UV_TILE) // output_heads == rank


def dcp_local_uv_tile(tile, output_heads):
    return tile % (output_heads * V_DIM // UV_TILE)


def ug_split(samples: int, inter: int = INTER):
    """Return the balanced up/gate leftover split for batches two and four."""

    full_tiles, remainder = divmod((samples * TOP_K + 1) * (inter // UG_TILE), BLOCKS)
    if samples not in (2, 4) or remainder == 0 or BLOCKS % remainder:
        return None
    segments = BLOCKS // remainder
    if (HIDDEN // 128) % segments or (HIDDEN // 128) // segments > WAVES // 2:
        return None
    return full_tiles, segments


def _align(size: int, alignment: int = 256) -> int:
    return (size + alignment - 1) // alignment * alignment


def layout(
    samples: int,
    heads: int,
    npes: int,
    sparse_attention_topk: int,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    indexer_cp: bool = False,
    inter: int = INTER,
    output_heads: int | None = None,
    dcp_size: int = 1,
    native_fp4_mfma: bool = False,
    indexer_quant: bool = False,
):
    """Return byte offsets for per-rank scratch and symmetric peer buffers."""

    radix_digits, radix_bins = index_radix_geometry(indexer_quant)
    tile = row_tile_size(samples, native_fp4_mfma)
    if tile != samples:
        scratch, symmetric = layout(
            tile,
            heads,
            npes,
            sparse_attention_topk,
            with_indexer,
            index_max_seq,
            indexer_cp=indexer_cp,
            inter=inter,
            output_heads=output_heads,
            dcp_size=dcp_size,
            native_fp4_mfma=native_fp4_mfma,
            indexer_quant=indexer_quant,
        )
        count = samples // tile
        scratch = dict(scratch, _tile_rows=tile, _tile_count=count, _tile_stride=scratch["_bytes"])
        scratch["_tile_barriers"] = count * scratch["_tile_stride"]
        scratch["_bytes"] = scratch["_tile_barriers"] + _align(
            (max(2, count - 1) if npes == 4 else count - 1) * (BLOCKS + 1) * 8
        )
        symmetric = dict(symmetric, _tile_stride=symmetric["_bytes"])
        symmetric["_bytes"] *= count
        return scratch, symmetric

    split_count = sparse_attention_topk // sparse_keys_per_task(samples, heads)
    output_heads = heads if output_heads is None else output_heads
    tiled_indexer = with_indexer and indexer_uses_tiles(index_max_seq, indexer_cp, indexer_quant)
    index_ranks = index_cp_world(npes, samples, index_max_seq) if indexer_cp else 1
    pair_bytes = 8
    items = [
        ("q_a", samples * Q_LORA * pair_bytes),
        ("q_an", samples * Q_LORA // 2 * pair_bytes),
        ("kv_a", samples * (KV_LORA + PE_DIM) * pair_bytes),
        ("kvnew", samples * KV_LORA * pair_bytes),
        ("penew", samples * PE_DIM * pair_bytes),
        ("q_nope", samples * heads * NOPE_DIM * pair_bytes),
        ("q_pe", samples * heads * PE_DIM * pair_bytes),
        ("q_lat", samples * heads * KV_LORA * pair_bytes),
        ("sp_acc", samples * split_count * heads * KV_LORA * pair_bytes),
        ("sp_m", samples * split_count * heads * pair_bytes),
        ("sp_l", samples * split_count * heads * pair_bytes),
        ("o", samples * output_heads * V_DIM * pair_bytes),
        ("a", samples * HIDDEN * pair_bytes),
        ("scores", samples * N_EXPERTS * pair_bytes),
        ("xq", samples * HIDDEN // 4 * pair_bytes),
        ("xqs", samples * (XQ_GROUPS if native_fp4_mfma else XQ_BLOCKS) * pair_bytes),
        ("sel", samples * MOE_SLOTS * pair_bytes),
        ("prob", samples * MOE_SLOTS * pair_bytes),
        ("mid", samples * MOE_SLOTS * inter * pair_bytes),
        ("ugp", BLOCKS * samples * 2 * UG_TILE * pair_bytes),
        ("xqd", samples * HIDDEN * 4),
    ]
    if with_indexer:
        items += [
            ("index_k", samples * INDEX_DIM * pair_bytes),
            ("index_k_new", samples * INDEX_DIM // 2 * pair_bytes),
            ("index_ready", samples * pair_bytes),
            ("index_w", samples * INDEX_HEADS * pair_bytes),
            ("index_q", samples * INDEX_Q_ROWS // 2 * pair_bytes),
            ("index_q_quant", samples * INDEX_HEADS * 20 * pair_bytes if indexer_quant else 0),
            ("index_k_quant", samples * 20 * pair_bytes if indexer_quant else 0),
            (
                "index_scores",
                samples * (index_cp_capacity(index_max_seq, index_ranks) if indexer_cp else index_max_seq) * pair_bytes,
            ),
            ("indices", samples * sparse_attention_topk * 4),
            ("indices_ready", samples * pair_bytes),
        ]
        if tiled_indexer:
            parts = index_cp_partitions(index_max_seq, index_ranks, indexer_quant)
            items += [
                ("index_cp_hist", radix_digits * samples * parts * radix_bins * pair_bytes),
                ("index_cp_counts", samples * parts * 2 * pair_bytes),
                ("index_cp_offsets", samples * parts * 2 * pair_bytes),
            ]

    offset, scratch = 0, {}
    for name, size in items:
        scratch[name] = offset
        offset += _align(size)
    scratch["_bytes"] = offset

    part = npes * samples * HIDDEN * pair_bytes
    region = 2 * part
    dcp_part = dcp_size * samples * (heads * V_DIM // UV_TILE) * DCP_SUMMARY_PAIRS * pair_bytes if dcp_size > 1 else 0
    symmetric = {
        "attn": 0,
        "ffn": region,
        "dcp": 2 * region,
        "_part_stride": part,
        "_dcp_part_stride": dcp_part,
        "_bytes": 2 * region + 2 * dcp_part,
    }
    if tiled_indexer or (with_indexer and indexer_cp):
        # Two launch slots, like the attention/FFN peer buffers. Distinct radix
        # rounds must not overwrite a histogram or threshold still being read.
        offset = symmetric["_bytes"]
        cp_items = (
            [
                ("index_cp_hist", index_ranks * radix_digits * samples * radix_bins * pair_bytes),
                ("index_cp_threshold", radix_digits * samples * 2 * pair_bytes),
                ("index_cp_counts", index_ranks * samples * 2 * pair_bytes),
                ("index_cp_offsets", samples * 2 * pair_bytes),
                ("index_cp_indices", samples * sparse_attention_topk * pair_bytes),
            ]
            if tiled_indexer
            else []
        )
        if indexer_cp:
            cp_items += [("index_cp_scores", samples * index_local_capacity(index_max_seq, indexer_quant) * pair_bytes)]
        if indexer_cp and not tiled_indexer and npes > 1:
            cp_items += [("index_cp_q", samples * INDEX_Q_ROWS // 2 * pair_bytes)]
        for name, size in cp_items:
            symmetric[name] = offset
            symmetric[name + "_stride"] = _align(size)
            offset += 2 * _align(size)
        symmetric["_bytes"] = offset
    return scratch, symmetric


def cache_samples_per_task(samples: int, npes: int, with_indexer: bool, native_fp4_mfma: bool = False) -> int:
    """Bound the cache task's serial query work on wider TP4 decode groups."""
    per_query = (with_indexer and samples in (8, 10, 12)) or (native_fp4_mfma and samples == 12)
    return 1 if npes == 4 and per_query else samples


def stage_tasks(
    samples: int,
    heads: int,
    sparse_attention_topk: int,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    expert_mxfp4: bool = False,
    inter: int = INTER,
    indexer_cp: bool = False,
    npes: int = 1,
    native_fp4_mfma: bool = False,
    cohort_rows: bool = False,
    uv_reuse: bool = False,
    split_task_pair: int = 1,
    cohort_samples: int | None = None,
    indexer_quant: bool = False,
):
    """Return ``(stage name, task count)`` pairs in execution order."""

    tile = row_tile_size(samples, native_fp4_mfma)
    if tile != samples:
        stages = stage_tasks(
            tile,
            heads,
            sparse_attention_topk,
            with_indexer,
            index_max_seq,
            expert_mxfp4,
            inter,
            indexer_cp,
            npes,
            native_fp4_mfma,
            cohort_rows,
            uv_reuse,
            split_task_pair,
            samples if cohort_samples is None else cohort_samples,
            indexer_quant,
        )
        return [(f"tile{part}/{name}", count) for part in range(samples // tile) for name, count in stages]
    index_ranks = index_cp_world(npes, samples, index_max_seq) if indexer_cp else 1
    tasks = [
        ("qkv_a", N_QKV_A),
        ("q_norm", samples),
        ("cache", samples // cache_samples_per_task(samples, npes, with_indexer, native_fp4_mfma)),
        ("q_b", heads * (NOPE_DIM + PE_DIM) // Q_B_TILE),
    ]
    if with_indexer:
        shard_q = indexer_cp and not indexer_uses_tiles(index_max_seq, indexer_cp, indexer_quant)
        tasks += [("index_q", INDEX_Q_ROWS // INDEX_TILE // (index_ranks if shard_q else 1))]
    tasks += [("uk", heads * KV_LORA // UK_TILE)]
    if with_indexer:
        capacity = index_cp_capacity(index_max_seq, index_ranks) if indexer_cp else index_max_seq
        tasks += [("index_score", samples * ((capacity + INDEX_KEYS_PER_TASK - 1) // INDEX_KEYS_PER_TASK))]
        if indexer_uses_tiles(index_max_seq, indexer_cp, indexer_quant):
            parts = index_cp_partitions(index_max_seq, index_ranks, indexer_quant)
            for digit in range(index_radix_geometry(indexer_quant)[0]):
                tasks += [(f"index_hist_{digit}", samples * parts), (f"index_prefix_{digit}", samples)]
            tasks += [("index_count", samples * parts), ("index_offsets", samples), ("index_emit", samples * parts)]
        tasks += [("index_select", samples)]
    tasks += [
        (
            "split",
            samples
            * (heads // sparse_heads_per_task(samples, heads, npes, native_fp4_mfma, cohort_rows))
            * (sparse_attention_topk // sparse_keys_per_task(samples, heads))
            // split_task_pair,
        ),
        ("uv", samples * (heads * V_DIM // UV_TILE) // (4 if uv_reuse else 1)),
        ("o", N_ROW_TILES),
        ("router", samples * N_ROUTER),
        (
            "ug",
            (
                inter
                if samples == 1
                else samples * max(BLOCKS, ((MOE_SLOTS * inter // UG_TILE + BLOCKS - 1) // BLOCKS) * BLOCKS)
            ),
        ),
        ("down", HIDDEN // dn_tile(samples, expert_mxfp4)),
    ]
    if grouped_ug_enabled(samples, npes, native_fp4_mfma, cohort_rows):
        # Include the shared expert even when every routed slot is distinct.
        # Full-CTA rounding keeps the subsequent stage's tuned CTA placement.
        maximum = (samples * TOP_K + 1) * (inter // NATIVE_UG_TILE)
        capacity = ((maximum + BLOCKS - 1) // BLOCKS) * BLOCKS
        if cohort_rows and npes == 4 and cohort_samples is not None:
            width = 64 if with_indexer else 128
            chunks = global_ug_chunks(cohort_samples, samples)
            rounds = (chunks * (inter // width) + BLOCKS - 1) // BLOCKS
            cohort_width = BLOCKS // (cohort_samples // samples)
            capacity = max(capacity, rounds * cohort_width * (width // NATIVE_UG_TILE))
        tasks = [(name, max(count, capacity) if name == "ug" else count) for name, count in tasks]
    return tasks
