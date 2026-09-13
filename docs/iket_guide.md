# IKET — In-Kernel Event Tracing

**Status: unstable.** The API, record format and output schema may change in any minor
release. See [`api_stability.md`](api_stability.md).

**Hardware: gfx942 (CDNA3) and gfx950 (CDNA4) only.** Other targets raise at trace time.

IKET lets a kernel author annotate phases in the source and get a timeline back. It
answers questions ATT cannot: *how much of my mainloop is waiting on LDS versus issuing
MFMA?* ATT sees instructions, not the phases you meant.

## Quick start

```python
import flydsl.expr as fx

@flyc.kernel
def gemm(...):
    fx.iket.range_push("prologue")
    ...
    fx.iket.range_pop()

    fx.iket.range_push("mainloop")
    for k_tile in range(0, k_tiles, 1):
        fx.iket.range_push("k_tile", k_tile)      # payload = loop index

        fx.iket.range_push("lds_wait")
        ...
        fx.iket.range_pop()

        fx.iket.range_push("mfma_issue")
        ...
        fx.iket.range_pop()

        fx.iket.range_pop()                        # k_tile
    fx.iket.range_pop()                            # mainloop
```

Run it and collect:

```bash
FLYDSL_IKET_ENABLE=1 python my_kernel.py
```

```python
fx.iket.collect("gemm")     # writes gemm_iket.trace.json; open at ui.perfetto.dev
fx.iket.summary()           # or just per-phase totals, no file
```

**Nothing is emitted unless tracing is enabled.** Without `FLYDSL_IKET_ENABLE=1` (or the
`iket` compile hint) an annotated kernel compiles to the same code as an unannotated one,
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
token = fx.iket.sentinel_token("mma_k_tile")
for k_tile in fx.range(0, k_tiles, 1):
    ab_full = ab_consumer.wait_and_advance()
    fx.iket.range_end(token)            # no-op on the first iteration
    token = fx.iket.range_start("mma_k_tile")
    ...
fx.iket.range_end(token)
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
| `FLYDSL_IKET_ENABLE` | `0` | Emit instrumentation |
| `FLYDSL_IKET_BUFFER_BYTES` | `64 MiB` | Device trace buffer size |
| `FLYDSL_IKET_EVENTS_PER_WAVE` | `256` | Slots reserved per wave |
| `FLYDSL_IKET_BLOCKS` | *(all)* | `x,y,z` records one workgroup; `xcc:N` records one XCD |
| `FLYDSL_IKET_DUMP_DIR` | *(cwd)* | Where traces are written |

Per compilation instead of process-wide:

```python
flyc.compile[{"iket": True}](launch, *args)
```

### `FLYDSL_IKET_BLOCKS` is the lever for large grids

A realistic GEMM launches thousands of identical workgroups. Tracing all of them
overflows the buffer, and an overflowed trace is **rejected, not trimmed** — a partial
trace has unbalanced push/pop and would render as a corrupt timeline.

Filter instead of shrinking the grid, which would change the workload you are measuring:

```bash
FLYDSL_IKET_ENABLE=1 FLYDSL_IKET_BLOCKS=0,0,0 python my_kernel.py
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

Per wave, once at kernel entry: two `s_getreg` reads and one atomic to claim a slot range.
Per event: a timestamp and the record store, **no atomic**.

**Measured at ~40 ns per event** on MI355X (38-43 ns across two work levels, R² ≥ 0.97) --
about one timestamp tick. Budget accordingly:

| Events per wave | Added time per wave |
|---|---|
| 16 | ~0.6 µs |
| 64 | ~2.6 µs |
| 256 | ~10 µs |

A phase shorter than ~1 µs instrumented at per-iteration granularity is measuring itself as
much as the kernel. Annotate k-tiles, not instructions.

Two caveats when reading a trace:

- The `s_waitcnt lgkmcnt(0)` after each timestamp drains *all* outstanding lgkm traffic,
  so an event taken while `s_load`/LDS operations are in flight can be biased late.
- Instrumentation is inline asm, which is opaque to the scheduler and can act as a
  code-motion barrier. Compare against an ATT baseline before trusting a delta measured
  on a hot mainloop.

## Relationship to ATT

Complementary, not competing. ATT (`/capture-kernel-trace`) gives instruction-level
detail post hoc; IKET gives the phases you named, with the nesting you wrote. Use ATT to
ask *which instructions stalled*, IKET to ask *which phase was slow*.

They cannot run simultaneously in a way that makes both numbers trustworthy — instrument
for one at a time.

## Limitations

- gfx942/gfx950 only.
- Events are **wave-level**, not thread-level: one record per wave per event site.
- A wave emitting more events than `FLYDSL_IKET_EVENTS_PER_WAVE` stops recording rather
  than writing into the next wave's range; the host reports the shortfall.
- Prefer small, deterministic workloads. There is no capture-window selection.
