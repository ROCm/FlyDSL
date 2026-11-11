<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 FlyDSL Project Contributors -->

# Cooperative algorithms

`flydsl.extension.coop` (`fx.coop`) provides reusable warp- and block-scope collective algorithms
for reductions, prefix scans, predicate voting, extremum indexing, register-tile data movement,
and sorting.

Every lane or thread in the participating group must execute a collective with uniform control-flow
parameters.

## Logical warp width

Warp collectives accept an optional compile-time `width`:

- `width=None` (default) spans the target's physical warp (`64` lanes on `gfx9` CDNA targets; `32`
  lanes on `gfx10`, `gfx11`, and `gfx12` targets, including `gfx1250`).
- Passing a power-of-two `width` (`1, 2, 4, ..., warp_size`) partitions each physical warp into
  independent `width`-lane groups so a single kernel computes identical results across wave32 and
  wave64 GPUs.

## Public API

### Warp reductions and extremum indexing

| API | Return type | Description |
|---|---|---|
| `fx.coop.warp_reduce(value, op, *, width=None, valid_items=None)` | Input element type | Reduce scalar or per-lane blocked items and broadcast the aggregate to every lane |
| `fx.coop.warp_sum` / `warp_max` / `warp_min` | Input element type | Convenience shorthands for `ADD`, `MAX`, and `MIN` reductions |
| `fx.coop.warp_argmax(value, *, width=None, valid_items=None)` | `(elem, fx.Int32)` | Return `(max_value, winning_index)` with deterministic lowest-index tie-breaking |
| `fx.coop.warp_argmin(value, *, width=None, valid_items=None)` | `(elem, fx.Int32)` | Return `(min_value, winning_index)` with deterministic lowest-index tie-breaking |
| `fx.coop.warp_head_segmented_reduce` / `warp_tail_segmented_reduce` | Input element type | Reduce contiguous lane segments delimited by head or tail flags |
| `fx.coop.WarpReduce[dtype, width]` | Specialized warp class | Class form exposing `reduce`, `sum`, `max`, `min`, `argmax`, `argmin`, and segmented reductions |

For multi-item per-lane inputs, `warp_argmax` and `warp_argmin` number elements in blocked order
(`(lane_id() % width) * items_per_lane + item`), propagate `NaN` consistently with `warp_max` /
`warp_min`, and break ties (including `-0.0` vs `+0.0` and multiple `NaN` lanes) in favor of the
lowest index.

### Warp scans and broadcast

| API | Return type | Description |
|---|---|---|
| `fx.coop.warp_inclusive_scan(value, op, *, width=None, init=None, valid_items=None)` | Input shape and type | Ordered inclusive prefix scan across ascending lanes |
| `fx.coop.warp_exclusive_scan(value, op, *, width=None, init=None, valid_items=None)` | Input shape and type | Ordered exclusive prefix scan across ascending lanes |
| `fx.coop.warp_inclusive_sum` / `warp_exclusive_sum` | Input shape and type | Convenience `ADD` scan shorthands |
| `fx.coop.warp_scan` | `(inclusive, exclusive)` | Compute both prefixes in one pass |
| `fx.coop.warp_scan_with_aggregate` | `(inclusive, exclusive, aggregate)` | Compute both prefixes plus the unseeded group total |
| `fx.coop.warp_broadcast(value, src_lane, *, width=None)` | Input shape and type | Broadcast `src_lane`'s value to every lane in the logical warp |
| `fx.coop.WarpScan[dtype, width]` | Specialized warp class | Class form for reusable warp scan and broadcast specializations |

### Warp predicate votes, ballots, and compaction ranks

| API | Return type | Description |
|---|---|---|
| `fx.coop.warp_any(predicate, *, width=None)` | `fx.Boolean` | `True` in every lane if any lane in the logical warp has a nonzero predicate |
| `fx.coop.warp_all(predicate, *, width=None)` | `fx.Boolean` | `True` in every lane if every lane in the logical warp has a nonzero predicate |
| `fx.coop.warp_ballot(predicate, *, width=None)` | `fx.Uint64` | Bitmask in `[0, 2**width)` where bit `k` reflects logical lane `k` on both wave32 and wave64 |
| `fx.coop.warp_count(predicate, *, width=None)` | `fx.Int32` | Number of active predicates in `[0, width]` returned to every lane |
| `fx.coop.warp_ballot_rank(predicate, *, width=None)` | `(fx.Int32, fx.Int32)` | `(exclusive_rank, total_count)` for register-level stream compaction |
| `fx.coop.WarpVote[dtype, width]` | Specialized warp class | Class form exposing `any`, `all`, `ballot`, `count`, and `ballot_rank` |

### Warp data movement, batched reduction, and sorting

| API | Purpose |
|---|---|
| `WarpLoad` / `WarpStore` | Move blocked, striped, or transposed per-lane item tiles between memory and registers |
| `WarpExchange` | Rearrange per-lane item tiles between blocked and striped register layouts |
| `WarpReduceBatched` / `warp_reduce_batched*` | Reduce independent register columns across lanes |
| `WarpBitonicSort` / `WarpMergeSort` | Sort keys or key/value pairs across a logical warp |

### Block-scope primitives

Block collectives are specialized with bracket syntax (for example,
`fx.coop.BlockReduce[fx.Float32, fx.known_block_size()]` or
`fx.coop.BlockScan[fx.Int32, 256, fx.coop.BlockScanAlgorithm.WARP_SCANS]`). Allocate each
primitive's `SharedStorage` via `fx.SharedAllocator()` and synchronize with `fx.barrier()` before
reusing shared memory across back-to-back calls:

- **Reductions and scans:** `BlockReduce`, `BlockScan`, `BlockAdjacentDifference`, `BlockDiscontinuity`
- **Data movement:** `BlockLoad`, `BlockStore`, `BlockExchange`, `BlockShuffle`
- **Sorting, ranking, selection, and RLE:** `BlockRadixSort`, `BlockRadixRank`, `BlockMergeSort`,
  `BlockTopK`, `BlockRunLengthDecode`, `BlockHistogram`

## Usage

The following kernel compacts positive elements within a 32-lane logical warp using
`warp_ballot_rank` and records the peak element and its lane index using `warp_argmax` without
allocating shared memory:

```python
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

WIDTH = 32


@flyc.kernel
def compact_and_peak_kernel(
    Src: fx.Tensor,
    Compacted: fx.Tensor,
    Meta: fx.Tensor,
):
    tid = fx.thread_idx.x
    val = Src[tid]
    keep = val > 0.0

    rank, kept_count = fx.coop.warp_ballot_rank(keep, width=WIDTH)
    max_val, max_lane = fx.coop.warp_argmax(val, width=WIDTH)

    if keep:
        Compacted[rank] = val
    if tid == 0:
        Meta[0] = kept_count
        Meta[1] = max_lane
        Meta[2] = max_val.bitcast(fx.Int32)


@flyc.jit
def launch_compact_and_peak(
    Src: fx.Tensor,
    Compacted: fx.Tensor,
    Meta: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    compact_and_peak_kernel(Src, Compacted, Meta).launch(
        grid=(1, 1, 1),
        block=(WIDTH, 1, 1),
        stream=stream,
    )
```

Runnable examples:

- `examples/extension/coop/01-warp_collectives.py` — warp reduction and exclusive scan without LDS.
- `examples/extension/coop/02-block_scan.py` — vector-valued block scan for stream compaction.
- `examples/extension/coop/03-warp_vote_and_argmax.py` — warp ballot compaction and extremum
  indexing (`warp_any`, `warp_count`, `warp_ballot_rank`, `warp_argmax`, and `warp_argmin`).

## Target dispatch and True16 notes

`fx.coop` dispatches target-specific overrides while `fx.coop.universal` always runs the portable
shuffle implementation:

- **Wave64 CDNA (`gfx9*`)**: Uses VALU DPP (`row_shr`, `row_bcast:15`, `row_bcast:31`, `wave_shr:1`,
  `quad_perm`, `row_half_mirror`, `row_ror:8`), `ds_swizzle_b32` for sub-wave cross-row steps,
  `v_readlane_b32`, and 64-bit `ballot` + `v_mbcnt_lo_u32_b32` / `v_mbcnt_hi_u32_b32`.
- **Wave32 GFX10+/RDNA/CDNA5 (`gfx10*`, `gfx11*`, `gfx12*`)**: Uses pure-VALU DPP (`row_shr`,
  `row_xmask`, `row_share`, `quad_perm`) + `v_permlanex16_b32` + `v_readlane_b32` with zero
  cross-lane LDS instructions across all widths `1..32`, plus 32-bit `ballot` + `v_mbcnt_lo_u32_b32`.
- **Narrow dtypes and True16 (`ROCm/FlyDSL#1376`)**: Scalar narrow dtypes (`Boolean`, `Int8`/`Uint8`,
  `Int16`/`Uint16`, `Float16`, `BFloat16`) promote losslessly to 32-bit in `warp_argmax` /
  `warp_argmin` so VALU DPP folds execute cleanly across all targets. Packed OCP FP8 warp
  collectives (`warp_reduce_e4m3x4`, `warp_reduce_e5m2x4`, `warp_argmax_e4m3x4`,
  `warp_argmin_e4m3x4`) are built into `coop/warp/{reduce,rocdl}.py` in commented-out form awaiting
  upstream LLVM PR [llvm/llvm-project#231721](https://github.com/llvm/llvm-project/pull/231721)
  ([ROCm/FlyDSL#1376](https://github.com/ROCm/FlyDSL/issues/1376)), which adds missing
  `UseRealTrue16Insts` (`_t16`) ISel patterns for `v_cvt_pk_f32_{fp8,bf8}` on `+real-true16`
  targets (`gfx117x`, `gfx120x`, `gfx125x`).
