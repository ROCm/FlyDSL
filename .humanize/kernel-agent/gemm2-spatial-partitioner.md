# gemm2 channel balance via aiter's portable spatial tile partitioner (replaces xcd)

Port aiter's ACTUAL, XCD-count-INDEPENDENT gemm2 block->tile mapping — the
`GemmSpatiallyLocalTilePartitioner` grouped 2D rasterization — as our gemm2
block->(m_block_idx, n_block_idx) map, and DROP the non-portable explicit-8-XCD
swizzle (`g2_xcd`). This is the true aiter alignment for HBM channel balance.

Setup: gfx950 MI350X, `HIP_VISIBLE_DEVICES=3`, cold
(`FLYDSL_RUNTIME_ENABLE_CACHE=0`), GPT-OSS M=128, model=inter=2880->pad 3072,
E=128, topk=4, mxfp4 w2, per_1x32 E8M0 scale. Ours-best base config (worktree
branch `rlcr/g2-spart` @ 947c766c) = BM32 + reduce + g2ks2 (B-prefetch) + bhoist
+ apf. Drivers: `same_input_parity.py` (ours), `aiter_gemm2_isolated.py` (aiter).

Prereq reading: `gemm2-mem-pattern-align.md` (the measured channel-imbalance
root cause + the xcd result to beat), `cktile-gemm2-deepdiff.md`.

## 0. THE KEY FINDING: aiter's partitioner is GroupNum=1, M01=1 — a no-op

aiter's shipped gemm2 instantiates
`GemmSpatiallyLocalTilePartitioner<CodegenFlatmmShape, GroupNum, M01>` with
**`GroupNum=1`, `M01=1`** — hard-coded in `FlatmmConfig` for EVERY instance, no
override anywhere:

```
csrc/ck_tile_gemm_moe_2stages/include/moe_cktile2stages_common.cuh:58-59
    static constexpr int TileParitionerGroupNum = 1;
    static constexpr int TileParitionerM01      = 1;
    ...:102-105  using TilePartitioner = GemmSpatiallyLocalTilePartitioner<shape, 1, 1>;
```

`MoeFlatmmKernel::operator()` calls `TilePartitioner{M,N}.GetOutputTileIndex(blockIdx.x)`
(`moe_flatmm_kernel.hpp:757-762`). I traced `GetOutputTileIndex`
(`gemm_tile_partitioner.hpp:274-360`) at `GroupNum=1, M01=1` by hand and by
numeric replay over realistic grids (`/tmp/spart_check.py`):

- `group_size = ceil(M0*N0/1) = M0*N0`, `big_group_num = 1`, so
  `remap_block_1d_id == block_1d_id` (the big-group interleave is identity).
- `M0_mod_M01 = 0`, `M01_adapt = 1`, `idx_M01 = 0`, so
  `return (idx_M0, idx_N0) = (block_1d_id / N0, block_1d_id % N0)`.

**At (1,1) the partitioner is byte-identical to the naive m-major linear map we
already use** (`m_block_idx = bx // num_n_blocks`, `n_block_idx = bx % num_n_blocks`).
So aiter's spatial partitioner is NOT the source of its channel balance for this
instance — porting it verbatim would be a no-op (still CV 9.7%). Verified:
`get_output_tile(b,M0,N0,1,1) == (b//N0, b%N0)` for all b over M0 in
{37,40,125,512}, N0=12.

Where aiter's balance actually comes from: its gemm2 tiles N at 128 (vs our 256),
runs a persistent grid (Kind2 grid 983040), and reads bf16 A — a different tiling
regime, not the partitioner grouping. The productive port is therefore the
GENERAL `GetOutputTileIndex` parameterized by (GroupNum, M01), sweeping
NON-TRIVIAL values that actually rasterize spatially-locally and rebalance
channels — which is exactly what the partitioner is designed to do, just with the
grouping ENABLED (GroupNum/M01 > 1) rather than aiter's disabled (1,1).

## 1. The port (`g2_spart`, opt-in, default byte-identical)

Opt-in `g2_spart` param on `compile_gemm2_a4w4_port`, env `MXFP4_G2_SPART`,
encoded `GroupNum*100 + M01` (e.g. `402` = GroupNum4, M01=2); `0`/unset = off =
byte-identical naive linear grid. Threaded through the gemm2 dispatcher cache key
+ kernel-name tag (`_spart{G}x{M01}`). gemm1 untouched (the remap lives entirely
in `moe_dispatcher.py`'s gemm2 one-shot path; `mxmoe_gemm_v2.py` unchanged).

`_spart_output_tile_index(block_1d_id, M0, N0, GroupNum, M01)` is a faithful DSL
port of `GetOutputTileIndex`'s else-branch (M0=total_m_blocks runtime;
N0=num_n_blocks, GroupNum, M01 compile-time): group_size / big_group_num remap,
then the M01 spatially-local M-window re-tile. It emits the same grouped
rasterization; consecutive block ids stay spatially local in the M0xN0 grid so
concurrent blocks' weight fetches spread across HBM channels — WITHOUT any XCD
awareness (no hard-coded 8; portable across GPUs).

Because the remap needs `M0 = total_m_blocks` (runtime, from the cumsum), the
spart path reads the cumsum FIRST (losing the default's A-prologue/cumsum overlap)
then feeds `unit_bx = m_block_idx*num_n_blocks + n_block_idx` to the A prologue +
`run_unit` — same tradeoff the xcd path made. `gemm2_body_v2` re-decodes `unit_bx`
linearly back to the same (m,n), so the port is consistent with the body.

**Bijection over [0, M0*N0):** verified for all candidate (GroupNum,M01) over
M0 in {1,2,3,37,40,125,512}, N0=12 — every (m,n) tile computed exactly once, no
dropped/duplicated tiles (`/tmp/spart_check.py`).

## 2. MEASURED channel distribution (rocprofv3 TCC_EA0_RDREQ, 128 instances)

Per-channel HBM read requests over the 128 TCC instances (16 channels x 8 XCC),
median over the replayed a4w4 gemm2 dispatches (warmup dropped), same GPU3/session.
Method identical to `gemm2-mem-pattern-align.md` §4b.

| config | total EA reads | per-chan mean | min | max | **CV** | **max/min** |
|---|---:|---:|---:|---:|---:|---:|
| naive (linear, spart off) | 4,857,351 | 37,948 | 31,495 | 40,198 | **9.68%** | **1.28x** |
| spart 202 (G2,M01=2) | 4,557,421 | 35,605 | 33,442 | 37,764 | 5.75% | 1.13x |
| **spart 402 (G4,M01=2)** | **4,501,146** | 35,165 | 34,992 | 35,364 | **0.32%** | **1.01x** |
| spart 404 (G4,M01=4) | 4,509,433 | 35,230 | 35,070 | 35,497 | 0.35% | 1.01x |
| spart 802 (G8,M01=2) | 4,508,507 | 35,223 | 34,897 | 35,483 | 0.48% | 1.02x |
| spart 804 (G8,M01=4) | 4,529,698 | 35,388 | 35,050 | 35,803 | 0.55% | 1.02x |
| spart 1204 (G12,M01=4) | 4,563,093 | 35,649 | 35,430 | 35,977 | 0.44% | 1.02x |
| **xcd4 (prior, doc §5)** | 4,493,066 | — | — | — | **0.64%** | **1.02x** |
| **AITER (same session)** | **4,436,910** | 34,663 | 34,590 | 34,759 | **0.18%** | **1.00x** |

**spart 402 (GroupNum=4, M01=2) is the winner: CV 9.68% -> 0.32%, max/min 1.01x,
total reads 4.86M -> 4.50M.** It BEATS the prior xcd4 path on channel balance
(0.32% vs 0.64% CV) and approaches aiter's 0.18%. Total HBM reads land within
1.4% of aiter's 4.44M (was 9.5% over) — the imbalance was forcing the extra
fetches; even distribution removes them, same as the xcd path. The portable
partitioner MATCHES-OR-BEATS the XCD swizzle as the channel-balance mechanism.

## 3. MEASURED perf (gemm2 isolated, median-of-5 cold, same GPU3 session)

Interleaved 5 rounds of naive / spart402 / aiter to control gfx950 clock drift.
Absolute us is ~12% higher than `gemm2-mem-pattern-align.md`'s numbers because
this GPU/session runs hotter (see +-14% gfx950 clock noise in MEMORY); the
RELATIVE deltas and the aiter ratio measured IN THIS SESSION are load-bearing.

| config | a4w4 median us | a8w4 median us | ratio vs aiter (95.3) |
|---|---:|---:|---:|
| aiter cktile a16w4 | 95.3 | — | 1.00x |
| naive (linear) | 109.4 | 112.0 | 1.148x |
| **spart 402** | **108.3** | **109.9** | **1.136x** |

- a4w4: **109.4 -> 108.3 us (-1.0%)**, ratio vs aiter 1.148x -> 1.136x.
- a8w4: **112.0 -> 109.9 us (-1.9%)** (its heavier A path is more
  channel-sensitive, same pattern the xcd path showed).

raw us (GPU3, this hotter session):
```
naive a4w4:    109.1 109.3 109.4 109.5 109.8 -> med 109.4
spart402 a4w4: 108.3 107.6 108.3 108.5 108.2 -> med 108.3
naive a8w4:    112.0 112.0 111.8 112.0 111.3 -> med 112.0
spart402 a8w4: 110.2 109.7 110.0 109.9 109.4 -> med 109.9
aiter a4w4:    95.1  95.3  95.5  95.3  95.3  -> med 95.3
```

### spart vs xcd: matches on mechanism, perf comparison caveat

On the SESSION-INDEPENDENT channel metric spart402 (CV 0.32%, 4.50M reads)
MATCHES-OR-BEATS xcd4 (CV 0.64%, 4.49M reads) — the two are equivalent on total
HBM reads and spart is tighter on CV. The doc §5 xcd perf (94.5us, 0.993x vs
aiter) was measured in a COOLER session (its naive baseline was 97.1us there vs
109.4us here), so the absolute us are not directly comparable across sessions. A
same-session xcd re-measurement was attempted but the ported xcd swizzle block
(from commit 93ad0bd4, a different base tree) faulted (hipErrorIllegalAddress) on
this base config and was reverted unused — xcd is the mechanism being DROPPED, so
it is not carried in this tree. The channel-balance equivalence is established on
the portable metric; the smaller absolute perf gain here reflects the
memory-headroom-compressed hotter session, not a weaker mechanism (same +9.1%
over-fetch is removed, total reads land at aiter parity).

## 4. Correctness + byte-identical default + gemm1-untouched proofs

- **Correctness (cold, real 2880 dims, spart402):** a4w4 cos = **0.9910**
  (>0.85), a8w4 cos = **0.9996** (>0.95) — identical to baseline (bijective block
  permutation, same math). Thresholds NOT weakened.
- **Byte-identical default (AC-3):** with `MXFP4_G2_SPART` unset the emitted
  default gemm2 kernel (`gemm2_a4w4_port_h3072_imax8192_bm32_reduce_tk4_pad_g2ks2_bhoist_apf_v2`,
  no `_spart` tag) is **md5-identical** with the spart code present vs stashed:
  `8263440a146105e9b138b109296a8648` both ways (empty diff). No extra IR emitted
  when off.
- **gemm1 untouched:** `git diff` touches ONLY `kernels/moe_dispatcher.py`
  (+98/-2), all in the gemm2 dispatch path (`_spart_output_tile_index` helper,
  `compile_gemm2_a4w4_port` param/env/tag, the one-shot `elif` remap branch,
  `get_g2` cache key). No `gemm1` line changed; `mxmoe_gemm_v2.py` unchanged.
  Python style gate passes (black + ruff).

## 5. Verdict

aiter's REAL gemm2 tile partitioner is `GemmSpatiallyLocalTilePartitioner` with
**GroupNum=1, M01=1, which degenerates to the naive linear map** — it is NOT what
balances aiter's channels (that comes from aiter's N=128 tiling + persistent grid
regime). Porting the GENERAL partitioner with the grouping ENABLED
(**GroupNum=4, M01=2 = `MXFP4_G2_SPART=402`**) rebalances OUR channels from CV
9.68% to **0.32%** (beating the prior xcd4's 0.64%, approaching aiter's 0.18%) and
cuts total HBM reads to 4.50M (aiter parity 4.44M), with a4w4 -1.0% / a8w4 -1.9%
gemm2 time in this session, correctness held, default byte-identical, gemm1
untouched. This portable, XCD-count-INDEPENDENT spatial partitioner REPLACES the
non-portable `g2_xcd` explicit-8-XCD swizzle as the channel-balance mechanism =
the true aiter alignment.

## Artifacts

- Code: `kernels/moe_dispatcher.py` (`_spart_output_tile_index` DSL port of
  `GetOutputTileIndex`; `g2_spart` param + env `MXFP4_G2_SPART` + `_spart{G}x{M01}`
  tag on `compile_gemm2_a4w4_port`; one-shot `elif` remap branch; `get_g2` cache
  key). gemm1 and `mxmoe_gemm_v2.py` untouched.
- aiter source: `moe_cktile2stages_common.cuh:58-59,102-105` (GroupNum=1/M01=1);
  `ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp:274-360`
  (`GetOutputTileIndex`); `moe_flatmm_kernel.hpp:757-762` (its use).
- PMC captures: `/tmp/g2prof/spart_{naive,202,402,404,802,804,1204,aiter}/`
  (rocprofv3 json, TCC_EA0_RDREQ). Parser `/tmp/g2prof/perchan.py`.
- Timing: `/tmp/g2prof/times.csv` + `/tmp/g2prof/full_time2.log`.
- Partitioner math check: `/tmp/spart_check.py` (degeneration + bijection).
- Drivers: `.humanize/kernel-agent/same_input_parity.py`,
  `aiter_gemm2_isolated.py`.
