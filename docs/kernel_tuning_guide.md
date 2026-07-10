# Kernel Tuning Guide

Practical techniques for optimizing FlyDSL GPU kernels on AMD CDNA GPUs
(MI300X `gfx942`, MI350/MI355X `gfx950`). The running example is the production
preshuffle GEMM (`kernels/gemm/preshuffle_gemm.py`), but the levers — tiling,
LDS double-buffering, bank-conflict swizzle, prefetch, MFMA scheduling, epilogue
choice, and occupancy management — apply to any compute-bound kernel.

This guide distills the project-local tuning skills. Each section points to the
skill that carries the full detail and reproducible commands:

| Skill | Focus |
|---|---|
| `/gemm-optimization` | End-to-end GEMM tiling, pipeline, scheduling, epilogue |
| `/lds-optimization` | LDS bank conflicts, swizzle/padding, write→read latency |
| `/prefetch-data-load` | Software prefetch (double-buffering) with loop-carried state |
| `/kernel-trace-analysis` | rocprofv3 ATT traces + PMC counters → hotspot plan |
| `/bisect-perf-regression` | `git bisect` a kernel perf regression to one commit |

Tune **only after** a correctness baseline passes and you have a profile. Guessing
at optimizations without a trace usually moves the bottleneck instead of removing it.

---

## 0. Background: The GPU Performance Model

Before the specific levers, it helps to have a mental model of *where a kernel's
time goes*. A GPU does not make individual instructions fast — it hides their
latency with parallelism. Each SIMD holds several wavefronts; when one wave
stalls waiting on memory, the scheduler issues from another ready wave. So tuning
is two problems: **keep enough independent work in flight to cover latency**, and
**don't exceed the machine's throughput ceilings** (compute FLOPs or memory
bandwidth). Almost every technique later in this guide is one of those two.

### Thread traces and instruction interleaving

An **ATT (Advanced Thread Trace)** records, per instruction on one CU, when it
issued and how many cycles it *stalled*. Within a single wave, instructions issue
in program order; a "stall" means the wave could not issue because it was waiting
on a dependency — almost always an `s_waitcnt` counter or a barrier.

The compiler (guided by the `sched_*` hints in §5) **interleaves independent
instruction classes** — MFMA, global loads (VMEM), LDS reads/writes — so that a
long-latency operation overlaps useful work instead of blocking it. Two hardware
counters gate this:

- **`vmcnt`** — outstanding VMEM (global/`buffer_load`) operations.
- **`lgkmcnt`** — outstanding LDS and scalar-memory operations.

An instruction that *consumes* a load result waits on the relevant counter
(`s_waitcnt vmcnt(0)` / `lgkmcnt(0)`). If the consumer is issued too close to the
load, the load's latency is exposed as stall cycles in the trace. Interleaving and
scheduling exist to convert those stall cycles into issue cycles. Collect and read
a trace with `/kernel-trace-analysis` (§9).

### The latencies you are hiding

Order-of-magnitude on CDNA (cycles; exact values vary by arch and contention):

| Operation | Latency | Notes |
|---|---|---|
| SALU / VALU | ~1–8 | address math, integer/float ALU |
| MFMA | ~16–64 | the compute you *want* to be waiting on |
| LDS `ds_read`/`ds_write` | ~20–40 (gfx942) | async; more with bank conflicts |
| Global load (HBM) | ~300+ | an order of magnitude above LDS |
| `s_barrier` | variable | whole workgroup must arrive |

The whole game is arranging enough independent instructions (and enough resident
waves) between issuing a high-latency op and needing its result.

### Hiding LDS latency

LDS ops are asynchronous, and a `ds_read` that depends on a prior `ds_write` must
be separated by `s_waitcnt lgkmcnt(0)` or an `s_barrier`. If the wait sits right
after the write, the ~20–40-cycle latency is fully exposed. Hide it by
**increasing the write→read distance** — schedule independent MFMAs, address math,
or the next tile's global loads between the write and the wait — and by
**A0-prefetching** the first LDS pack right after a barrier so the first
`ds_read` overlaps the following VMEM loads. See §3 (swizzle) and §4 (prefetch);
bank conflicts (§3) both raise per-op latency *and* cut effective LDS bandwidth.

### Hiding global-memory latency

Global loads are the longest-latency common op (~300+ cycles), so they dominate if
exposed. The primary tool is **software prefetch / double-buffering** (§4): issue
the *next* iteration's `buffer_load`s before consuming the current iteration's
data, so the fetch overlaps compute. Larger `tile_k` increases reuse per byte
fetched; higher **occupancy** gives the scheduler more waves to switch to while
one is waiting. A useful trick in multi-phase kernels: **hoist the next phase's
loads into a barrier-wait region** — the barrier is dead time anyway, so the load
arrives for free.

### Bandwidth

Once latency is hidden, you hit a **throughput ceiling**. There are two:

- **Compute** — MFMA FLOP/s. Peak (gfx942 MI300X, single GCD): ~653 TFLOPS FP8,
  ~326 TFLOPS BF16, ~653 TOPS INT8.
- **Memory bandwidth** — HBM GB/s, plus the L2 and LDS byte/cycle limits (LDS peak
  128 B/cyc on gfx942, 256 B/cyc on gfx950).

*Effective* memory bandwidth depends on access quality: **coalesced, vectorized**
accesses (`buffer_load_dwordx4`, full 64 B cache lines) approach peak, while
uncoalesced or partial-cache-line access wastes HBM. When a kernel is
memory-bound, cache/HBM PMC counters (§9) — L2 hit rate, 32 B-partial fraction,
over-fetch, per-channel balance — tell you *why*, which ISA inspection alone
usually gets wrong.

### Is the kernel memory-, compute-, or latency-bound?

Compute the arithmetic intensity `AI = FLOPs / bytes_moved` and compare it to the
roofline crossover (§8). Three regimes, each with a different fix and a different
trace/PMC signature:

| Regime | What's saturated | Trace / PMC signature | Where to look |
|---|---|---|---|
| **Compute-bound** | MFMA units | MFMA utilization high (≥ ~70%), memory pipes idle | §5 scheduling, §7 occupancy, cut non-MFMA overhead |
| **Bandwidth-bound** | HBM / L2 / LDS BW | memory BW near peak, low L2 hit or high over-fetch; MFMA util moderate | §3 coalescing/swizzle, smaller dtype, bigger `tile_k`, XCD balance (§9) |
| **Latency-bound** | *nothing* — work is stalled | **both** MFMA util and memory BW low, **high** `s_waitcnt`/`s_barrier` stall | §4 prefetch, §7 occupancy, §5 interleaving |

The distinction that trips people up is **latency-bound vs bandwidth-bound**: a
latency-bound kernel is slow while *neither* ceiling is saturated — the pipes are
simply idle waiting on exposed latency (common at small `M` or low occupancy). The
cure is more overlap and more waves in flight, not more bandwidth. A
bandwidth-bound kernel, by contrast, is already moving bytes as fast as the memory
system allows, so the cure is moving *fewer* or *better-shaped* bytes. Rule of
thumb: **M ≤ 512 → likely memory/latency-bound; M > 512 → likely compute-bound.**

---

## 1. Tiling Strategy

GEMM tiles the output `C[M, N]` and the reduction `K` into blocks. With a 256-thread
block (4 waves × 64 lanes) the typical mapping is:

```
block_x → M tiles (tile_m rows)      wave_id  = tid // 64  → N partitioning
block_y → N tiles (tile_n cols)      lane_id  = tid % 64   → M + N within wave
```

Derived per-tile parameters:

```python
m_repeat   = tile_m // 16            # M-direction 16x16 MFMA repeats
n_per_wave = tile_n // 4             # N range per wave (4 waves split tile_n)
num_acc_n  = n_per_wave // 16        # N-direction accumulators per wave
```

### Recommended configurations

| Scenario | tile_m | tile_n | tile_k | Notes |
|---|---|---|---|---|
| Small batch (M ≤ 32) | 16 | 64–128 | 256–512 | Memory-bound; large `tile_k` for reuse |
| Medium batch | 64 | 256 | 128 | Balanced compute/memory |
| Large batch (M ≥ 4096) | 128 | 256 | 128 | Compute-dense; benefits from async copy |
| FP4 (gfx950) | 32–64 | 128–256 | 256 | MFMA-scale instructions |

### Constraints

- `tile_m` must be a multiple of 16 (MFMA M dimension).
- `tile_n` must be a multiple of 64 (4 waves × 16 N per MFMA).
- `tile_k × elem_bytes` must be a multiple of 64 (the K64-byte micro-step).
- `tile_k` must divide `K` evenly (the pre-shuffled B layout requires it).
- LDS budget: `2 × tile_m × tile_k × elem_bytes` (double-buffered A) must fit in
  **64 KB** on gfx942 / **160 KB** on gfx950.

A quick MFMA count sanity check (FP8, K64 micro-step = 2× K32 MFMA):

```
MFMA_per_tile = k_unroll × m_repeat × num_acc_n × 2      # k_unroll = tile_k_bytes // pack // 64
# tile 64×256×128, FP8: (128/64) × (64/16) × (256/4/16) × 2 = 2×4×4×2 = 64
```

See `/gemm-optimization` §1 for the full derivation and the worked 5120×5120×8320 example.

---

## 2. LDS Double-Buffering (Ping-Pong)

With `lds_stage=2`, allocate **two** LDS buffers for the A tile. While one buffer
feeds the MFMAs, the next K-tile's A is loaded into the other, hiding the
global→LDS latency:

```
Buffer PONG: [compute k=0] [  load k=2  ] [compute k=2] ...
Buffer PING: [  load k=1  ] [compute k=1] [  load k=3  ] ...
```

Allocate the two buffers with `fx.SharedAllocator` over an `@fx.struct` storage
layout (the current LDS API — see the Kernel Authoring Guide):

```python
import flydsl.expr as fx

@fx.struct
class SharedStorage:
    a_ping: fx.Array[fx.Int8, tile_m * tile_k]
    a_pong: fx.Array[fx.Int8, tile_m * tile_k]

lds = fx.SharedAllocator().allocate(SharedStorage).peek()
a_ping = lds.a_ping.view(fx.make_layout((tile_m, tile_k), (tile_k, 1)))
a_pong = lds.a_pong.view(fx.make_layout((tile_m, tile_k), (tile_k, 1)))
```

The main loop then processes two K-tiles per iteration (one on each buffer),
carrying the accumulators, the current B tile, and the A0 prefetch as
loop-carried state. On spill-bound tiles (`num_acc_n = 8`), `lds_stage=1` (a
single A buffer with B carried in the loop) can reduce VGPR pressure — but it
regresses non-spilling tiles, so measure before switching. See `/gemm-optimization` §2.

---

## 3. LDS Bank-Conflict Swizzle

LDS is banked: **32 banks** (4 B each) on gfx942, **64 banks** on gfx950;
`bank = (byte_addr / 4) % num_banks`. When lanes in a wave access different
addresses that map to the **same** bank, the accesses serialize (a *bank
conflict*); accessing the **same** address broadcasts for free.

A row-major tile with stride `tile_k` makes every row land on the same banks.
The fix is an XOR swizzle that folds the row index into the column address at
16-byte granularity — zero extra LDS, ~1 SALU op per address:

```python
def swizzle_xor16(row, col_bytes, k_blocks16):
    return col_bytes ^ ((row % k_blocks16) * 16)   # k_blocks16 = tile_k_bytes // pack // 16
```

**The swizzle must be applied identically on the write (global→LDS) and read
(LDS→VGPR) paths** — if only one side swizzles, the data is read from the wrong
place. The physical write can stay linear as long as the read reverses the same
mapping.

### Arch note (gfx950 has 64 banks)

Masks tuned for 32-bank gfx942 may be suboptimal on gfx950: a 128-byte stride is
a *full* conflict on gfx942 but only a *2-way* conflict on gfx950 (full conflict
needs a 256-byte stride there). Re-check swizzle width when porting.

### Padding as an alternative

Adding 1–4 elements of padding per row breaks the stride alignment without XOR
math, at the cost of extra LDS. Prefer swizzle (zero overhead); use padding when
the swizzle is awkward to integrate and LDS has headroom (gfx950's 160 KB gives
plenty). See `/lds-optimization` for the diagnosis-by-trace workflow and the
gfx950 `DS_READ_*_TR_*` transpose-load instructions.

---

## 4. Data Prefetch Pipeline

Global loads (`buffer_load`) are **asynchronous**: the instruction returns
immediately and data arrives later. Issue the *next* iteration's loads before
consuming the *current* iteration's data, so load latency overlaps compute:

```
without: |load|stall|compute|load|stall|compute|
with:    |load0|compute0+load1|compute1+load2|compute2|
```

### Loop-carried prefetch with `range(..., init=...)`

A Python `for i in range(N)` is unrolled during tracing, so a `data = next_data`
swap becomes invisible to MLIR (both alias one SSA value, and LLVM hoists the
load as loop-invariant). Use FlyDSL's **runtime** loop with loop-carried values
to create genuine SSA phi nodes:

```python
# Prologue: load iteration 0 before the loop
next_a = buffer_ops.buffer_load(rsrc_a, offsets_0, vec_width=4)
init_state = [_unwrap(v) for v in [next_a, acc]]

# Runtime loop — bounds MUST be a typed DSL integer (fx.Int64), not Python ints (see pitfalls)
for iv, state in range(fx.Int64(0), fx.Int64(N - 1), fx.Int64(1), init=init_state):
    a, acc = state[0], state[1]
    next_a = buffer_ops.buffer_load(rsrc_a, compute_offsets(iv + 1), vec_width=4)  # async
    acc = rocdl.mfma_f32_16x16x16_f16(transform(a), b, acc)   # overlaps next load
    results = yield [_unwrap(v) for v in [next_a, acc]]

# Epilogue: process the last iteration from `results`
a, acc = results[0], results[1]
acc = rocdl.mfma_f32_16x16x16_f16(transform(a), b, acc)
```

Three pitfalls (all covered in `/prefetch-data-load`):

1. **Bounds must be a typed DSL integer (`fx.Int64(...)`).** Constant Python-int
   bounds make the rewriter unroll the loop and silently ignore `init=`.
2. **Unwrap init values at hard boundaries only.** Most carried values stay
   `fx.Int32`/`fx.Float32`/`Vector`; unwrap to raw `ir.Value` only where a
   low-level helper demands it.
3. **Clear `SmemPtr._view_cache = None` before the epilogue** when a shared view
   was created inside the loop, or the epilogue use hits an SSA dominance error.

### Async copy (global → LDS DMA)

On gfx942/gfx950, `use_async_copy=True` streams A directly from global memory
into LDS (`raw_ptr_buffer_load_lds`), bypassing VGPR. This saves arch_vgpr and
suits large `tile_m` (≥ 128) where there is enough compute to hide the DMA.
gfx950 moves 16 B/DMA vs gfx942's 4 B/DMA.

**Don't prefetch** a loop that is already memory-bound, or when occupancy is
already 1 wave and adding buffers would spill — profile both ways.

---

## 5. MFMA Instruction Scheduling

`hot_loop_scheduler()` emits `rocdl.sched_*` hints that tell the compiler how to
interleave instruction classes inside the hot loop:

| Hint | Allows |
|---|---|
| `rocdl.sched_mfma(N)` | N MFMA (`v_mfma_*`) |
| `rocdl.sched_dsrd(N)` | N LDS reads (`ds_read_*`) |
| `rocdl.sched_dswr(N)` | N LDS writes (`ds_write_*`) |
| `rocdl.sched_vmem(N)` | N global loads (`buffer_load_*`) |
| `rocdl.sched_barrier(0)` | scheduling fence — no reordering across |

The standard pattern interleaves one VMEM load and one LDS read per group of
MFMAs, pushing LDS writes to the tail so they overlap the last MFMAs and land
before the iteration-boundary `gpu.barrier()`:

```python
for sche_i in range_constexpr(sche_iters):
    rocdl.sched_vmem(1)                # global load (A or B)
    rocdl.sched_mfma(mfma_group)       # num_acc_n MFMAs
    rocdl.sched_dsrd(1)                # LDS read (A)
    rocdl.sched_mfma(mfma_group)       # more MFMAs
    if sche_i >= dswr_start - 1:
        rocdl.sched_dswr(1)            # LDS write for next tile, at the tail
rocdl.sched_barrier(0)
```

Async fat tiles benefit from evenly distributing `dsrd`/`vmem` across *all*
MFMAs (`enable_scheduler=True`). Toggling the scheduler is a real lever: for some
f16/bf16 tiles with `num_acc_n ≤ 2`, disabling it is faster; for async fat tiles
enabling it wins +8–10%. Measure per tile. See `/gemm-optimization` §5.

---

## 6. Epilogue Strategies

**Direct store (default).** Each thread writes its MFMA accumulators straight to
global memory. No extra LDS, simplest — but stores can be non-coalesced for some
tile shapes.

**CShuffle epilogue.** Route accumulators through LDS to re-map thread→element so
global writes are coalesced (`buffer_store_dwordx2`), at the cost of an LDS
allocation and one barrier:

```python
e_vec = 4 if (tile_n % 128 == 0) else 2
m_reps_shuffle = tile_m // 8
n_reps_shuffle = tile_n // (32 * e_vec)
```

Use CShuffle for large `tile_n` (≥ 128) where output coalescing matters. The
preshuffle GEMM also exposes fused epilogues via the `epilogue=` argument of
`compile_preshuffle_gemm` (`"none"`, `"bias"`, `"bias_relu"`, `"bias_silu"`,
`"bias_gelu"`).

---

## 7. Register Budget & Occupancy

**Occupancy** is how many wavefronts are resident per SIMD at once — the pool the
scheduler draws from to hide latency (§0). It is the **minimum across three
resource limiters**, capped by a hardware maximum:

```
occupancy (waves/SIMD) = min(vgpr_limit, lds_limit, sgpr_limit, HW_MAX)
```

Whichever resource you exhaust *first* caps occupancy — so all three must be
watched, and the arch you target changes each limit.

### The three resources

**1. VGPR.** On **CDNA3 (gfx942)** and **CDNA4 (gfx950)** the two vector-register
files — **arch_vgpr** (VALU, VMEM, LDS ops, prefetch buffers) and **accum_vgpr /
AGPR** (MFMA writeback) — share **one combined 512-entry budget** per SIMD, so
`vgpr_limit = 512 // (arch_vgpr + accum_vgpr)`. Growing prefetch/LDS-address
arch_vgpr therefore competes with MFMA accumulators for the same budget. (This is
*not* the old separate-pool `256 / max(arch, accum)` model — that was CDNA1
gfx908.)

| Combined arch + accum (wave64) | vgpr_limit (waves/SIMD) |
|---|---|
| ≤ 64 | 8 (hardware max) |
| ≤ 128 | 4 |
| ≤ 170 | 3 |
| ≤ 256 | 2 |
| ≤ 512 | 1 |
| > 512 | **SPILL** — severe regression |

**2. LDS.** LDS is a per-CU pool shared by all resident workgroups, so
`resident_workgroups = LDS_per_CU // LDS_per_workgroup`; more LDS per block →
fewer concurrent blocks → fewer waves. The pool size is arch-specific:
**64 KB (gfx942)**, **160 KB (gfx950)**, **320 KB (gfx1250)**. A 160 KB / 320 KB
budget lets a kernel keep bigger tiles (or more buffers) resident before LDS,
rather than VGPR, becomes the limiter.

**3. SGPR.** Scalar registers (kernel args, buffer descriptors, loop/scalar
state) also gate occupancy: on CDNA, ~**800 SGPRs per SIMD**, allocated in
granules of 16, so `sgpr_limit = 800 // sgpr_per_wave`. SGPR is rarely the binding
limit but can bite kernels with many buffer resources or scalar-heavy prologues.

### Architecture differences at a glance

| Arch | Wave | VGPR model | LDS/CU | Notes |
|---|---|---|---|---|
| gfx942 (CDNA3) | 64 | combined 512 (arch 256 + accum 256) / SIMD | 64 KB | MFMA; combined-pool occupancy |
| gfx950 (CDNA4) | 64 | combined 512 (arch 256 + accum 256) / SIMD | 160 KB | + MFMA-scale / transpose loads; 2.5× LDS headroom |
| gfx1250 | **32** | wave32 register file (per-wave counts differ) | 320 KB | wave32 accounting — a 32-lane wave; query per-kernel counts, don't assume the CDNA 512 rule |

Because gfx1250 runs **wave32**, its per-wave VGPR/SGPR accounting differs from the
wave64 CDNA parts — don't reuse the `512 // (arch+accum)` rule there. The
occupancy *formula* (min over VGPR/LDS/SGPR, capped at the HW max) still holds;
only the per-resource limits change. When in doubt, read the actual allocation
back from the profiler rather than estimating.

### Estimating and measuring

Rough VGPR estimate for a wave64 FP8 tile:

```
accumulators = m_repeat × num_acc_n × 4     # → accum_vgpr
B tile       = k_unroll × 2 × num_acc_n × 2 # → arch_vgpr
A prefetch   ≈ 4 ; A tile regs = num_a_loads × 4 ; addressing ≈ 10–20
```

Query the real allocation (all three resources) from the rocprofv3 database:

```sql
SELECT ks.KernelName, ki.arch_vgpr_count, ki.accum_vgpr_count,
       ki.sgpr_count, ki.lds_size
FROM rocpd_kernel_dispatch kd
JOIN rocpd_info_kernel_symbol ks ON kd.kernel_symbol_id = ks.id
JOIN rocpd_info_kernel ki ON kd.kernel_id = ki.id
WHERE ks.KernelName LIKE '%target_kernel%' LIMIT 5;
```

`/kernel-trace-analysis` (§9) reports the computed occupancy and which resource is
the binding limiter, so you know whether to cut VGPR, shrink LDS per block, or
reduce SGPR pressure.

**Do not** use `maxnreg` to force `accum_vgpr=0` — it spills MFMA results through
arch_vgpr via `v_accvgpr_read` (measured ~4.5× regression).

---

## 8. Performance Metrics & Roofline

```python
flops   = 2 * M * N * K
tflops  = flops / (us / 1e6) / 1e12

# bytes moved (FP8/INT8): A + B + C(bf16) + per-token scales
bytes_moved = M*K*eb + N*K*eb + M*N*2 + (M + N)*4
tbps = bytes_moved / 1e12 / (us / 1e6)
```

Peak references (gfx942 MI300X, single GCD): FP8 ~653 TFLOPS, INT8 ~653 TOPS,
BF16 ~326 TFLOPS. Compare `flops / bytes_moved` (arithmetic intensity) to the
roofline crossover. Practical rule of thumb: **M ≤ 512 → memory-bound** (chase
bandwidth), **M > 512 → compute-bound** (chase MFMA utilization).

> Benchmark noise on gfx950 can reach ±14% from clock variation. Run isolated
> (60–80 iters) and take a median-of-7; diff against a baseline with
> `python3 scripts/compare_benchmark.py base.csv cur.csv`.

---

## 9. Profiling: ATT Traces & PMC Counters

Instruction-level stall data comes from a rocprofv3 ATT trace. Collect and
analyze with the `/kernel-trace-analysis` skill, which bundles the
`hotspot_analyzer.py` (ATT hotspots) and `pmc_l2_analyzer.py` (cache counters)
helpers under `.claude/skills/kernel-trace-analysis/scripts/`:

```bash
# 1. discover the hot kernel
rocprofv3 --stats --kernel-trace -f csv -- <CMD>
# 2. collect ATT for that kernel (input.yaml: advanced_thread_trace, att_target_cu:1)
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 rocprofv3 -i /tmp/trace_input.yaml -- <CMD>
# 3. hotspots mapped to source
python .claude/skills/kernel-trace-analysis/scripts/hotspot_analyzer.py \
    <ui_output_agent_*_dispatch_*> --topk 15 --mode both
```

Map the dominant stall type to a fix:

| Trace symptom | Bottleneck | Action |
|---|---|---|
| `s_waitcnt vmcnt(0)` before MFMA | global-load latency exposed | improve prefetch overlap; bigger `tile_k` |
| `s_waitcnt lgkmcnt(0)` after `ds_write` | LDS write→read latency exposed | insert independent work between them |
| `ds_read`/`ds_write` high stall | LDS bank conflicts | apply XOR swizzle (§3) |
| high `s_barrier` stall | sync overhead | fewer barriers; hoist loads into the wait |
| MFMA utilization < 50% | memory-bound | larger tile; prefetch harder |
| `s_nop` between MFMAs | pipeline bubbles | tune `hot_loop_scheduler` (§5) |

**When the kernel is memory-bound, ATT is not enough** — it has no cache
counters. Collect L2/HBM PMCs and use `pmc_l2_analyzer.py` (same skill scripts
dir) for L2 hit rate, 32 B-partial fraction, and over-fetch. A frequent root cause is HBM channel
imbalance from a linear (m-major) grid; a bijective XCD swizzle (`xcd_swizzle` on
`compile_preshuffle_gemm`) rebalances channels. **Always confirm with PMCs — ISA
inspection alone routinely mis-diagnoses memory bottlenecks.**

---

## 10. Bisecting a Performance Regression

When a kernel got slower and you don't know which commit did it, binary-search
with `/bisect-perf-regression`:

```
/bisect-perf-regression <good_commit> [bad_commit] -- <bench_cmd>
```

It establishes good/bad baselines, verifies the regression is real, then bisects
(checking out, rebuilding if needed, extracting the metric) until one commit is
isolated, and reports its diff. The bench command must run at every commit in the
range and print a stable metric; the working tree is auto-stashed and restored.

---

## 11. Optimization Checklist

| Stage | Check | If failing |
|---|---|---|
| Tiling | enough blocks to fill the GPU | reduce tile size |
| Tiling | `tile_k × elem_bytes ≤ LDS/2` | reduce `tile_k` |
| LDS | bank-conflict stall on `ds_read` | apply XOR swizzle |
| Prefetch | VMEM stall before MFMA | loop-carried prefetch / async copy |
| Pipeline | using `lds_stage=2` | enable double-buffer (unless spill-bound) |
| Scheduler | `s_nop`/bubbles between MFMAs | tune `hot_loop_scheduler` |
| Epilogue | uncoalesced output stores | CShuffle for large `tile_n` |
| Registers | combined arch + accum ≤ 256 | fewer buffers / async copy |
| ISA | MFMA ratio ≥ 40% of hot loop | cut non-MFMA overhead |
| Memory | L2 hit / HBM balance (PMCs) | grid/XCD swizzle |

---

## See Also

- [Kernel Authoring Guide](kernel_authoring_guide.md) — `@flyc.kernel`/`@flyc.jit`, LDS, tiled copy/MMA
- [Pre-built Kernels](prebuilt_kernels_guide.md) — GEMM/MoE/attention configs and dtypes
- [Testing & Benchmarking](testing_benchmarking_guide.md) — benchmark harness and CSV comparison
- Project-local skills: `/gemm-optimization`, `/lds-optimization`, `/prefetch-data-load`,
  `/kernel-trace-analysis`, `/bisect-perf-regression`
