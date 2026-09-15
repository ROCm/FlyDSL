# KTRACE — In-Kernel Wave Tracing

**Status: unstable.** The API, record format and output schema may change in any minor
release. See [`api_stability.md`](api_stability.md).

**Hardware: gfx942 (CDNA3) and gfx950 (CDNA4) only.** Other targets raise at trace time.

KTRACE lets a kernel author annotate phases in the source and get a timeline back. It
answers questions ATT cannot: *how much of my mainloop is waiting on LDS versus issuing
MFMA?* ATT sees instructions, not the phases you meant.

## Quick start

```python
import flydsl.expr as fx

@flyc.kernel
def gemm(...):
    fx.ktrace.range_push("prologue")
    ...
    fx.ktrace.range_pop()

    fx.ktrace.range_push("mainloop")
    for k_tile in range(0, k_tiles, 1):
        fx.ktrace.range_push("k_tile", k_tile)      # payload = loop index
        ...
        fx.ktrace.range_pop()                        # k_tile
    fx.ktrace.range_pop()                            # mainloop
```

The nesting above stops at the loop body on purpose. Adding sub-phases *inside* a
software-pipelined mainloop costs 18-23x the nominal per-event price and reshapes the
schedule it measures -- see [Software-pipelined loops](#software-pipelined-loops-the-per-event-cost-is-not-40-ns).

Run it and collect:

```bash
FLYDSL_KTRACE_ENABLE=1 python my_kernel.py
```

```python
fx.ktrace.collect("gemm")     # writes gemm_ktrace.trace.json; open at ui.perfetto.dev
fx.ktrace.summary()           # or just per-phase totals, no file
```

Either call reads the records back and re-arms the device buffer, which keeps a repeated
trace -- a benchmark loop, an autotune sweep -- from accumulating claims until the buffer
overflows. The records are kept host-side, so calling **both** after one launch works and
both describe that same launch, in either order. After the next launch they describe the
new one; with nothing traced at all, both return `None`.

**Nothing is emitted unless tracing is enabled.** Without `FLYDSL_KTRACE_ENABLE=1` (or the
`ktrace` compile hint) an annotated kernel compiles to the same code as an unannotated one,
so annotations can be left in production kernels.

## API

| Function | Purpose |
|---|---|
| `mark(name, payload=None)` | A point event |
| `range_push(name, payload=None)` | Open a stack range |
| `range_pop()` | Close the innermost stack range |
| `range_start(name, payload=None) -> RangeToken` | Open a token range |
| `range_end(token, payload=None)` | Close a token range |
| `sentinel_token(name) -> RangeToken` | A token that emits nothing, for loop-carried ranges |
| `collect(kernel, out_dir=None) -> path` | Write the trace (host side) |
| `summary() -> dict` | Per-phase totals (host side) |

### Which range form to use

`range_push`/`range_pop` for phases nested in the same scope. They pair LIFO along an
executed path; pushing in one branch of an `if` and popping in the other is fine, because
each executed path pushes once and pops once.

`range_start`/`range_end` when the close site is not lexically nested — notably **across
loop iterations**, where the token is carried by the loop:

```python
token = fx.ktrace.sentinel_token("mma_k_tile")
for k_tile in fx.range(0, k_tiles, 1):
    ab_full = ab_consumer.wait_and_advance()
    fx.ktrace.range_end(token)            # no-op on the first iteration
    token = fx.ktrace.range_start("mma_k_tile")
    ...
fx.ktrace.range_end(token)
```

This measures the interval between *wait boundaries* rather than the body, which is what
you want for a software-pipelined loop.

### Payloads

Accept Python `bool` / `int` / `float` and FlyDSL numerics; use `fx.Int64(x)` /
`fx.Float64(x)` for 64-bit. Payloads should be **wave-uniform** — under divergence the
first active lane's value is recorded.

## Settings

| Variable | Default | Effect |
|---|---|---|
| `FLYDSL_KTRACE_ENABLE` | `0` | Emit instrumentation |
| `FLYDSL_KTRACE_BUFFER_BYTES` | `64 MiB` | Device trace buffer size |
| `FLYDSL_KTRACE_EVENTS_PER_WAVE` | `256` | Slots reserved per wave; one holds the wave's counter, so N-1 are usable |
| `FLYDSL_KTRACE_BLOCKS` | *(all)* | `x,y,z` records one workgroup; `xcc:N` records one XCD |
| `FLYDSL_KTRACE_DUMP_DIR` | *(cwd)* | Where traces are written |

Per compilation instead of process-wide:

```python
flyc.compile[{"ktrace": True}](launch, *args)
```

### `FLYDSL_KTRACE_BLOCKS` is the lever for large grids

A realistic GEMM launches thousands of identical workgroups. Tracing all of them
overflows the buffer, and an overflowed trace is **rejected, not trimmed** — a partial
trace has unbalanced push/pop and would render as a corrupt timeline.

Filter instead of shrinking the grid, which would change the workload you are measuring:

```bash
FLYDSL_KTRACE_ENABLE=1 FLYDSL_KTRACE_BLOCKS=0,0,0 python my_kernel.py
```

Filtering happens **on the device**, so a non-recording wave writes nothing at all.

## What the numbers mean

Timestamps come from `s_memrealtime`, a constant-rate 100 MHz counter (10 ns per tick)
that is invariant under clock throttling.

**Effective resolution is ~40 ns**, because the counter advances in steps of 4 ticks.
Ranges shorter than that collapse to zero duration and render as markers. Instrument at
k-tile granularity, not per instruction.

All XCDs share one coherent clock domain — measured spread across 8 XCDs is under 20
ticks — so timestamps are directly comparable device-wide with no per-XCD correction.

## Costs

Per wave, once at kernel entry: two `s_getreg` reads and one atomic to claim a slot range
(only waves that pass the block filter claim one). Per event: a timestamp, an uncontended
bump of the wave's own counter, and the record store.

**Measured at ~40 ns per event** on MI355X (38-43 ns across two work levels, R² ≥ 0.97) --
about one timestamp tick. That figure holds for loops whose bodies are not themselves
waiting on lgkm traffic; see the next section for a kernel where it is off by 20x. Budget
accordingly:

| Events per wave | Added time per wave |
|---|---|
| 16 | ~0.6 µs |
| 64 | ~2.6 µs |
| 256 | ~10 µs |

A phase shorter than ~1 µs instrumented at per-iteration granularity is measuring itself as
much as the kernel. Annotate k-tiles, not instructions.

Two caveats when reading a trace:

- The timestamp is an SMEM read, so its value is not usable until an
  `s_waitcnt lgkmcnt(0)` that drains *all* outstanding lgkm traffic: an event taken while
  `s_load`/LDS operations are in flight can be biased late. The backend places that wait
  at the first use of the timestamp, not immediately after the read, so the stall lands
  wherever the record is written rather than at the annotation.
- The reads are LLVM intrinsics, so the scheduler may move independent work across them.
  That keeps the instrumentation cheaper than an opaque asm block would, but it also means
  an event marks a point in the *scheduled* code, not in your source order. Compare
  against an ATT baseline before trusting a delta measured on a hot mainloop.

### Software-pipelined loops: the per-event cost is not ~40 ns

In a loop that keeps memory in flight across `s_barrier`s -- a hand-written software
pipeline -- the `s_waitcnt lgkmcnt(0)` above stops being a small bias and becomes the
dominant cost: each timestamp drains the very traffic the pipeline exists to overlap.

Measured on the gfx950 half-tile-interleaved GEMM, whose mainloop issues a `s_barrier` and
an async LDS load every beat, tracing one workgroup of a 4096³ problem:

| Instrumentation | Events per iteration | Iteration time |
|---|---|---|
| Phase only (`k_tile_pair`) | 0 | 3582 ns |
| One range per half-tile | 4 | 7218 ns |
| Three sub-phases per half-tile | 16 | 15238 ns |

That is **729-909 ns per event, 18-23x the nominal ~40 ns**, and the added cost scales
with event count rather than saturating. At the finest level the instrumented loop runs
4.3x slower than the real one.

Subtracting a per-event constant does not recover the true numbers: doing so leaves the
annotated region *shorter* than the uninstrumented loop, because the probes do not merely
add time, they change how the pipeline schedules. Within one such run the sub-phase totals
are still comparable to each other -- each carries the same number of probes -- but their
absolute values, and any split derived from them, are not usable.

Practical rule: on a software-pipelined mainloop, annotate at **phase granularity** and
stop there. If you need to know where a k-tile's time goes, use ATT, which observes
instructions without perturbing them.

This rule is not limited to the hand-interleaved kernel measured above. Both GEMMs in
`kernels/gemm/` are software-pipelined by the definition at the top of this section: the
non-HTI `gemm_a16w16_gfx950` runs a `stages >= 2` LDS pipeline that prefetches k-tile
`k + (stages - 1)` while computing k-tile `k`, and holds that traffic in flight across the
loop body with `wait_asyncmark(stages - 2)`. Both therefore carry **phase-granularity
annotations only** (`prologue` / `mainloop` / `k_tile` / `drain` / `epilogue`); neither is
an example of safe sub-phase tracing, and there is no in-tree kernel that is.

A loop qualifies as *not* pipelined -- and so is safe to sub-phase -- only when its body
waits for its own memory before using it, leaving nothing in flight across the timestamp:
an unstaged loop that issues a load, waits on it, and computes, with no prefetch of a
later iteration.

## Relationship to ATT

Complementary, not competing. ATT (`/capture-kernel-trace`) gives instruction-level
detail post hoc; KTRACE gives the phases you named, with the nesting you wrote. Use ATT to
ask *which instructions stalled*, KTRACE to ask *which phase was slow*.

They cannot run simultaneously in a way that makes both numbers trustworthy — instrument
for one at a time.

## Limitations

- gfx942/gfx950 only.
- Events are **wave-level**, not thread-level: one record per wave per event site.
- A wave emitting more than `FLYDSL_KTRACE_EVENTS_PER_WAVE - 1` events stops recording rather
  than writing into the next wave's range; the host reports the shortfall.
- Sub-phase granularity is unusable on software-pipelined loops: the probes cost 18-23x
  their nominal price there and reshape the schedule they measure. Both in-tree GEMMs are
  pipelined and are annotated at phase granularity only; see Costs.
- Prefer small, deterministic workloads. There is no capture-window selection.
