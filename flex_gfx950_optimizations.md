# Flex attention gfx950: optimization path

This note records the FlyDSL flex kernel work on MI350 (gfx950) against
**flash dualwave** (`flash_attn_gfx950.py`). It is the written companion to
the Cursor canvas `flex-opt-journey` (charts of the same numbers). Packed
causal-mask numbers are from 15 Sep 2026 on the same host; the dense/causal
sweeps below are the 15 Sep occupancy-held run after the full-tile 8c
causal split (absolute µs is slower than 14 Sep because clocks were
lower; same-session ratios are the comparison).

Kernel: `kernels/attention/flex_attention_gfx950.py`.

## Protocol (do not mix with other benches)

| Knob | Value |
| --- | --- |
| Shape | `B=2`, `Sq=3072`, `H=32`, `D=128`, bf16 |
| Tile | `block_n=64`, `num_groups=8`, `pipe_depth=1`, 512-thread CTA |
| Occupancy-held | same tensor for the full Skv sweep; only `k`/`v` seq dim changes |
| Physical GPU | HIP `0` → `GPU[3]`; clocks from `rocm-smi -d 3` |
| `N` | `Skv / 64` |
| Auto 8-cluster | dense `Skv ≥ 768`; causal `Skv ≥ 2048` |
| TFLOPS dense | `4 · Sq · Skv · D · H · B / time` |
| TFLOPS causal valid | unmasked bottom-right cells only |
| TFLOPS causal Sq×Skv | same rectangle formula as dense (not real FLOPS) |

Host of record: `smci350-rck-g03-d09-31.rck.dcgpu`, container
`rchamber_prompt_opt`. HIP device 0 is **not** the idle die that
`rocm-smi -d 0` reports.

## Outcome in one page

Dense long-KV flex is slightly ahead of flash (peak **1032 TFLOPS** at
`N=128`; **799 µs** vs **810 µs**, **−1.4%**). Flash is 1–2 µs ahead only
at dense `N=4/6/8`. Causal flex is faster at **every** `N` in this session:
grid shrink while `Skv < Sq`, then the full-tile 8c / masked 4c tail after
`N=32` (`N=128` **700 µs** vs **725 µs**, **−25 µs**). Score kernels on
**time**, not `µs × MHz`.

```
dense latency (µs) and vs-flash speedup, occupancy-held, 15 Sep 2026
N           1     4     8    16    24    32    48    64    96   128
flex       42.8  59.0  81.3 139.0 206.8 260.6 370.4 474.5 687.2 799.4
flash      58.1  57.0  80.9 149.9 210.4 264.5 375.9 483.4 698.8 810.4
vs flash  +26.4% −3.7% −0.6% +7.3% +1.7% +1.5% +1.5% +1.9% +1.7% +1.4%

causal latency (µs), vs-flash speedup, and TFLOPS
N              1     8    16    24    32    48    64    96   128
flex µs      19.3  40.1  56.1  80.6 117.1 251.6 373.7 600.6 700.3
flash µs     33.1  44.6  58.9  93.4 150.6 265.7 389.4 615.3 725.1
vs flash   +41.8% +10.2% +4.7% +13.7% +22.3% +5.3% +4.0% +2.4% +3.4%
flex valid T  3.5 107.4 306.6 479.8 587.2 614.7 689.8 772.4 956.9
flex Sq×Skv T 335 1286 1838 1918 1761 1229 1104 1030 1178
```

`vs flash` = `(t_flash − t_flex) / t_flash`. Positive means flex is faster.
`Sq×Skv T` uses the dense rectangle formula on causal times. It is **not**
real causal FLOPS; it is there to line up with the dense TFLOPS column.

Interactive plots live in the canvases **flex-flash-seq-sweep** and
**flex-opt-journey**.

## Steps, in the order we took them

### 1. Dualwave body and 8-cluster long KV — landed

Match flash’s dualwave CTA: 512 threads, `n=64`, same LDS map, online
softmax + C→B bridge + GEMM2. Compile-time 8-cluster (`long_seq_8c`)
widens the KV MMA when the sequence is long enough.

Cutoffs in the kernel:

```python
_LONG_SEQ_8C_SKV_DENSE = 768
_LONG_SEQ_8C_SKV_MASKED = 2048
```

**Why two cutoffs.** Masked paths skip KV tiles, so the 8c body only pays
once the *tensor* `Skv` is long enough for the last Q tiles. Ablation
(step 5) showed dense 8c is required; causal 8c does not move occupancy-held
time. Do not lower the causal cutoff unless a later run shows a **≥20 MHz**
sclk win (or a clear µs win at held occupancy).

### 2. Head-fast grid — landed

Launch order is `(hq, num_q_tiles, grid_z)` so `linear_id % 8` is the
**head**, not the Q tile. Q tiles of one head stay on one XCD and replay KV
from that XCD’s ~4 MiB L2.

Before this change, long dense KV thrashed L2 across XCDs, sclk collapsed
(~1420 MHz at 1000 W), and flex trailed flash by tens of microseconds.
After: dense `N≥24` is essentially tied; sclk at `N=128` dense recovered
to ~1548 MHz at the same wattage.

### 3. LLVM cachepolicy on Q / O / KV — SC1 rejected

`BufferCopy128b` cachepolicy bits on gfx950:

| Bit | Meaning |
| --- | --- |
| 0 | cached (default) |
| 2 | **SC1**, not nontemporal |
| 4 | NT |

An early “2 = NT” experiment was **SC1**. Occupancy-held dense regressed
**8–11 µs**. True NT (bit 4) was within noise of cached. Production stays
**cached**.

### 4. Causal bottom-right grid shrink — landed

Bottom-right causal has `max(Sq − Skv, 0)` fully-masked leading Q rows.
Flex drops every workgroup wholly contained in that prefix
(`_active_q_tiles`). Flash dualwave still launches the full Q grid.

rocprofv3 (Sq=3072, 256 rows/WG → 12 Q tiles):

| N | Skv | Flex WGs | Flash WGs | Flex µs | Flash µs |
| --- | --- | --- | --- | --- | --- |
| 24 | 1536 | **384** | 768 | 61 | 73 |
| 32 | 2048 | **512** | 768 | 95 | 106 |
| 24 dense | 1536 | 768 | 768 | tied | tied |

This is why causal flex **looks** much faster at `N=24/32`. The skip ends
when `Skv ≥ Sq` (`N≥48` here). It does **not** explain the long-seq
cycle gap.

### 5. Force 4-cluster at all Skv — 8c kept for dense only

Occupancy-held dense:

| N | Flex 4c µs | Flex 8c µs | Flash µs |
| --- | --- | --- | --- |
| 24 | 163.0 | 161.4 | 161.0 |
| 32 | 207.0 | 204.5 | 203.5 |
| 128 | 724.9 | 713.0 | 722.0 |

Causal 4c matched 8c at every Skv in that sweep. Production: auto 8c for
dense `Skv≥768`; causal stays 4c until `Skv≥2048` (and even then 8c is
clock/occupancy insurance, not a measured µs win).

### 6. Clock vs HBM (18 s sclk / wattage) — no extra NT/H/B change

Unique KV working set per XCD is about `Skv · D · 2 · 2 / 8` bytes
(K+V, bf16, 8 XCDs). That crosses ~4 MiB near `Skv=1024` (`N=16`).

Median GFXCLK, 18 s dispatch loop, `rocm-smi -d 3`:

**Dense**

| N | Flex MHz | Flash MHz | Notes |
| --- | --- | --- | --- |
| 8 | 2131 | 2111 | under the watt cap |
| 16 | 2050 | 1972 | approaching L2 cliff |
| 32 | 1696 | 1711 | 1000 W |
| 64 | 1626 | 1642 | gap **&lt;20 MHz** — no land |

**Causal** flex holds ~2190 MHz through `N=32` (partial KV, fewer live
WGs). Flash is already watt-limited there because it launches dead Q WGs.

Dense **4c** had a real clock collapse at `N=24` (**1475 MHz** vs 8c/flash
**~1584 MHz** at 1000 W). Later 4c can hold *more* clock than 8c and still
lose occupancy-held time. Scoring rule: land only a **≥20 MHz** sclk win
(or a clear µs win). NT / head-count / batch probes were **not** landed.

Power-up time series, 15 Sep 2026, `Skv=2048` (`N=32`), `rocm-smi -d 3`
from idle through compile into an 18 s loop (canvas
`warmup-to-steady-power-clock`):

| Path | Plateau W | Plateau sclk | mclk |
| --- | --- | --- | --- |
| Dense flex | 999 | 1558 MHz | 2000 MHz (idle and load) |
| Dense flash | 1000 | 1523 MHz | 2000 MHz |
| Causal flex | 824 | 2031 MHz | 2000 MHz |
| Causal flash | 897 | 1943 MHz | 2000 MHz |

Dense hits the 1000 W cap on the first hot sample. Causal stays under the
cap (flex uses fewer watts and keeps more sclk because of the smaller Q
grid). **mclk does not ramp**: `rocm-smi -d 3 --showclocks` reports
**2000 MHz** at idle (~40 MHz sclk) and throughout the 18 s loop for all
four kernels. The 11 Sep dense-flex 120 MHz deficit vs flash is gone after
head-fast.

ATT wave cycles at `Sq=Skv=3072` (warm dispatch, CU1, 15 Sep, latest 8c
kernel): dense even-wave median **129,962** (flex) vs **133,468** (flash),
**−2.6%**. Causal per-wave medians are not matched Q tiles; CU occupancy
span is **273k** vs **312k** (−12.5%). Canvas `att-cycles-flex-flash-3k`.

### 7. Packed causal mask (flash `attn_mask_vec2_imm`) — landed

Flash applies the causal diagonal with a **WG-uniform** `needs_mask`
predicate plus `_attn_mask_vec2_imm`. Flex used a per-wave / divergent
`scf.if` on the same scores. Flag stays on:

```python
_CAUSAL_WG_UNIFORM_PACKED_MASK = True
```

The taken path uses flash’s packed helper with **flex KV offsets**. Measured
15 Sep 2026, HIP 0, `B=2 Sq=3072 H=32 D=128` bf16, n64 g8 pd1. Correctness
vs bottom-right SDPA: max err 0.059 on 256², cosine ≥ 0.998.

| N | Skv | Flex µs | Flash µs | flex − flash |
| --- | --- | --- | --- | --- |
| 32 | 2048 | 90.19 | 114.57 | **−24.4 µs** (grid shrink still) |
| 48 | 3072 | 212.63 | 212.83 | **−0.2 µs** (tied) |
| 128 | 8192 | 729.60 | 722.88 | **+6.7 µs** (~0.9%) |

Keep the flag: no correctness fail, N=48 matches flash, N=128 gap is smaller
than the pre-change same-shape lead (~13 µs on 14 Sep). Do not mix the 14 Sep
absolute µs with this session; clocks moved.

## Time, not `µs × MHz`

A cycle count from a profiler is independent of frequency only if both
kernels ran at the same pinned clock, and even then HBM stalls cost more
core cycles at a higher sclk. We never had that measurement. Multiplying
wall time by GFXCLK (`651 µs × 1608 MHz`) *builds frequency into the
product*, so a hotter flex run looks like extra work. Compare **µs** (this
table) or pin sclk with `rocm-smi --setperfdeterminism` before reading
`SQ_INSTS_*`.

### 8. Pinned 1400 MHz + `SQ_INSTS_*` — 15 Sep 2026

`rocm-smi -d 3 --setperfdeterminism 1400` (HIP 0 = GPU[3], PCI `75:00.0`).
Under load GFXCLK read **1369 MHz**. Mean of 6 occupancy-held samples
(80 iters). PMC: 10 dispatches (2 warmup + 8),
`flash_attn_dualwave_swp_gfx950_kernel_0` vs
`flex_attn_bf16_m32n64d128_w1x1g8_dense_rsm_pd1_8c_stg_0`. Both 0 scratch,
grid 393216, 512 threads, **6144 waves**. Pin restored with sudo
`--resetperfdeterminism`.

Pinned wall time:

| N | Flex µs | Flash µs | Δ |
| --- | --- | --- | --- |
| 48 | 272.89 | 273.18 | −0.3 µs |
| 128 | 971.06 | 904.62 | **+66.4 µs** (+7.3%) |

The unpinned +6.7 µs at N=128 was flex running hotter. At a common cap the
real gap is **~66 µs**.

Issued-instruction means (chip-wide, per dispatch):

| Counter | N=48 flex | N=48 flash | N=128 flex | N=128 flash |
| --- | --- | --- | --- | --- |
| `SQ_INSTS_MFMA` | 5.112M | 5.112M | 20.840M | 20.840M |
| `SQ_INSTS_LDS` | 7.668M | 7.668M | 31.261M | 31.261M |
| `SQ_INSTS_VMEM` | 0.737M | 0.737M | 2.703M | 2.703M |
| `SQ_INSTS_VALU` | 29.259M | 29.648M | 104.958M | 107.849M |
| `SQ_INSTS_VALU_INT32` | 1.647M | 0.747M | 3.858M | 0.985M |
| `SQ_INSTS_SALU` | 2.390M | 2.169M | 7.305M | 7.514M |
| `SQ_BUSY_CYCLES` | 9.964M | 10.017M | 34.846M | 31.792M |

MFMA, LDS, and VMEM **match**. Flex issues **more INT32** (mask) but
**fewer total VALU**. N=128 time follows `SQ_BUSY_CYCLES` (+9.6%), not
instruction count. Remaining causal long-KV gap is **stall / schedule**,
not extra math or extra global loads.

Mask VALU is **diagonal-only**. Interior tiles are fully valid or fully
`-inf` skipped.

### 9. Full-tile 8c + masked 4c causal tail — landed

`_CAUSAL_8C_FULL_TILES_ONLY = True` splits each WG's live causal KV range:

- `[kv_lo, kv_full_hi)`: fully visible to the whole WG; use branch-free 8c.
- `[kv_full_hi, kv_hi)`: diagonal band; use 4c and packed mask before softmax.
- Existing epilogue only drains the final deferred `P@V`; it does not mask.

The exact full-tile boundary is `floor((q_min + 1) / block_n)`, clamped to
the live range. The 8c→4c handoff keeps `p_mixed` live, so the first 4c
tail step consumes the previous full tile's deferred PV without a drain /
re-prologue bubble. C2/C6 may prefetch the first diagonal K tile because
the 4c tail needs it; no diagonal QK runs in C1/C5.

Correctness:

- Boundary suite: max error 0.0561, cosine ≥ 0.9985 vs bottom-right SDPA.
- Long 8c path vs flash: max diff 0.000977 at N=48 and 0.000122 at N=128.

Unpinned, same-session:

| N | Flex full-tail µs | Flash µs | Δ |
| --- | --- | --- | --- |
| 32 | 84.63 | 115.44 | **−30.81 µs** |
| 48 | 201.59 | 212.23 | **−10.64 µs** |
| 128 | 698.48 | 718.00 | **−19.52 µs** |

Pinned 1400 MHz (observed ~1369 MHz under load):

| N | Before split flex | After split flex | Flash | After flex − flash |
| --- | --- | --- | --- | --- |
| 48 | 272.89 | **245.40** | 272.69 | **−27.29 µs** |
| 128 | 971.06 | **875.05** | 903.52 | **−28.47 µs** |

N=128 PMC after split (chip-wide, per dispatch):

| Counter | Before flex | After flex | Flash |
| --- | --- | --- | --- |
| `SQ_INSTS_MFMA` | 20.840M | 20.840M | 20.840M |
| `SQ_INSTS_LDS` | 31.261M | 31.261M | 31.261M |
| `SQ_INSTS_VMEM` | 2.703M | 2.722M | 2.703M |
| `SQ_INSTS_VALU_INT32` | 3.858M | **2.636M** | 0.985M |
| `SQ_INSTS_VALU` | 104.958M | 109.910M | 107.849M |
| `SQ_BUSY_CYCLES` | 34.846M | **30.578M** | 31.786M |
| Scratch bytes | 0 | **8** | 0 |

The main result is schedule, not instruction count: busy cycles fell
12.3% even though total VALU rose and the extra 4c tail loop introduced
an 8-byte scratch allocation. MFMA/LDS counts are unchanged. INT32 fell
1.22M by removing per-tile mask predicates from the full 8c prefix, but
the remaining dynamic tail control keeps it above flash.

## What we rejected

- **SC1** on global Q/O/KV copies.
- **Changing host `sysctl` / NUMA** as a kernel “win”.
- **Causal 8c cutoff below 2048** without a ≥20 MHz or µs win.
- Treating **N=24/32 causal** as an MFMA/softmax victory — it is launch size.
- Treating `µs × MHz` as a work metric when clocks differ.

## What is still open

1. Remove the 8-byte scratch allocation from the dynamic 4c tail without
   disturbing its 12.3% busy-cycle win.
2. Do not retune `long_seq_8c` cutoffs unless sclk or time moves enough
   to beat the 20 MHz / noise bar.

## Related artifacts

| Artifact | What |
| --- | --- |
| Canvas `flex-flash-seq-sweep` | 15 Sep occupancy-held Skv sweep |
| Canvas `flex-opt-journey` | Latency, TFLOPS, ratio, sclk, 4c vs 8c |
| Canvas `flex-vs-flash-full` | Same 15 Sep overlay |
| Canvas `sclk-vs-skv-hbm` | Clock vs working-set cliff |
| Canvas `warmup-to-steady-power-clock` | 15 Sep idle→plateau power/sclk |
| Canvas `flex-n24-n32-causal` | Grid-shrink explanation |
| Canvas `flex-cache-opt-cycle` | Cachepolicy / cycle notes |
| `flex_softmax_mask_opts.md` | Softmax/mask micro-opts (older) |
