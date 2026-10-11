# KTRACE — In-Kernel Wave Tracing

**Status: unstable.** The API, record format and output schema may change in any minor
release. See [`api_stability.md`](api_stability.md).

**Hardware: gfx942 (CDNA3) and gfx950 (CDNA4) only.** Other targets raise at trace time.

KTRACE lets a kernel author annotate phases in the source and get a timeline back. It
answers questions ATT cannot: *how much of my mainloop is waiting on LDS versus issuing
MFMA?* ATT sees instructions, not the phases you meant.

## Quick start

```python
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.experimental import ktrace

@flyc.kernel
def gemm(...):
    with ktrace.range("prologue"):
        ...

    with ktrace.range("mainloop"):
        for k_tile in fx.range(0, k_tiles, 1):
            with ktrace.range("k_tile", k_tile):  # payload = loop index
                ...
```

The nesting above stops at the loop body on purpose. Adding sub-phases *inside* a
software-pipelined mainloop costs 18-23x the nominal per-event price and reshapes the
schedule it measures -- see [Software-pipelined loops](#software-pipelined-loops-the-per-event-cost-is-not-40-ns).

```python
with ktrace.capture("gemm", blocks=(0, 0, 0)) as trace:
    launch_gemm(...)

print(trace.summary())
path = trace.export()  # writes gemm_ktrace.trace.json; open at ui.perfetto.dev
```

The capture context enables tracing for JIT specializations compiled inside it and reads
the records on exit. Its summary and export use the same snapshot. `blocks=(x, y, z)`
selects one workgroup without changing the launched grid. For multiple workgroups, omit
`blocks` and use `FLYDSL_KTRACE_BLOCKS` if needed. `summary()` and `export()` return
`None` when no events were recorded. The snapshot stays with its capture after exit;
call `trace.read()` explicitly to replace it after a later graph replay.

### Switch tracing on and off

With tracing off by default, the same launcher switches modes through the capture
scope:

```python
launch_gemm(...)  # untraced
with ktrace.capture("gemm", blocks=(0, 0, 0)) as trace:
    launch_gemm(...)  # traced
launch_gemm(...)  # untraced again
```

The first call in each mode compiles its own specialization. Later calls reuse the
corresponding cached specialization; leaving the capture restores the prior compile
hints. The untraced dispatch does not acquire the trace collection lock or touch the
trace buffer. A CUDA Graph retains the specialization captured in its nodes, so capture
a separate untraced graph if you also need untraced replays.

An untraced kernel has no trace instructions. Enabled tracing adds timestamp reads and
record writes, so its kernel cost depends on the number and placement of annotations.
Use coarse phases for production-sized kernels and compare performance with an
untraced run. See [Software-pipelined loops](#software-pipelined-loops-the-per-event-cost-is-not-40-ns)
for a case where dense annotations substantially change the schedule.

### Repeated layers and separate trace files

A capture can contain several launches, including repeated calls to the **same**
kernel. KTRACE records the HSA queue and dispatch id for each wave, so the combined
timeline keeps those launches on separate tracks. `summary()` totals the whole window;
`launches()` gives per-launch totals and `export_launches()` writes one trace file per
recorded launch:

```python
with ktrace.capture("model", blocks=(0, 0, 0), out_dir="traces") as trace:
    for layer in layers:
        launch_layer(layer)

for launch in trace.launches():
    print(launch["launch_id"], launch["phases"])

paths = trace.export_launches(names=[f"layer_{i}" for i in range(len(layers))])
```

`names` labels files in first-recorded-timestamp order. Use it when layers run
sequentially on one stream. When launches overlap across streams, inspect the
backend's launch metadata before assigning layer names. On ROCm, `launches()`
also reports HSA queue and dispatch ids. A launch with no recorded events (for
example, every wave excluded by the block filter) has no per-launch output.

For a full CUDA Graph replay, capture the graph's **individual layer launches**
inside `torch.cuda.graph()`, then call `trace.clear()` and replay the graph while the
outer KTRACE capture is active. The same `launches()` and `export_launches()` calls
split the replay, even if every graph node calls the same compiled kernel. Run
`python examples/07-ktrace_wave_timeline.py --layers 3 --graph` for a complete
example that writes three layer files and one combined file.

The existing `FLYDSL_KTRACE_ENABLE=1` switch and `ktrace.collect("gemm")` /
`ktrace.summary()` functions also work without a capture context. Either free function
reads and re-arms the device buffer; the next call sees the same cached snapshot until
another traced host launch or a CUDA Graph replay writes new records. Untraced calls
leave the last trace snapshot available. A graph replay bypasses the JIT, so the free
functions cannot detect a replay that records no events. For graph replay, prefer a
capture and call `trace.read()` after replay to force a fresh snapshot.

**Nothing is emitted unless tracing is enabled.** Without `FLYDSL_KTRACE_ENABLE=1` (or the
`ktrace` compile hint) an annotated kernel compiles to the same code as an unannotated one,
so annotations can be left in production kernels.

## API

| Function | Purpose |
|---|---|
| `range(name, payload=None)` | Context manager for a named kernel phase |
| `mark(name, payload=None)` | A point event |
| `range_push(name, payload=None)` | Open a stack range |
| `range_pop()` | Close the innermost stack range |
| `range_start(name, payload=None) -> RangeToken` | Open a token range |
| `range_end(token, payload=None)` | Close a token range |
| `sentinel_token(name) -> RangeToken` | A token that emits nothing, for loop-carried ranges |
| `collect(kernel, out_dir=None) -> path` | Write the trace (host side) |
| `summary() -> dict` | Per-phase totals (host side) |
| `capture(kernel="kernel", blocks=None, out_dir=None)` | Enable tracing for launches in a host `with` block; returns a capture with `clear()`, `read()`, `summary()`, `launches()`, `export()` and `export_launches(names=None)` |

### Which range form to use

`range_push`/`range_pop` for phases nested in the same scope. They pair LIFO along an
executed path; pushing in one branch of an `if` and popping in the other is fine, because
each executed path pushes once and pops once.

`with ktrace.range("name"):` emits the same push/pop pair around its body and is the
usual form when both ends fit in one source block. A disabled compilation emits neither
annotation.

`range_start`/`range_end` when the close site is not lexically nested — notably **across
loop iterations**, where the token is carried by the loop:

```python
stage = ktrace.sentinel_token("k_tile")
for i in fx.range(0, k_tiles, 1):
    ab_full = ab_consumer.wait_and_advance()
    # Closes the previous iteration's range. The sentinel makes the first pass
    # through here record nothing, since no range is open yet.
    ktrace.range_end(stage)
    stage = ktrace.range_start("k_tile")
    ...
ktrace.range_end(stage)
```

This measures the interval between *wait boundaries* rather than the body, which is what
you want for a software-pipelined loop.

### Payloads

Accept Python `bool` / `int` / `float` and FlyDSL numerics; use `fx.Int64(x)` /
`fx.Float64(x)` for 64-bit. Payloads should be **wave-uniform** — under divergence the
first active lane's value is recorded. Omitting a payload writes `null` in the raw
JSON and no payload argument in the timeline; an explicit `0` remains `0`.

## Settings

| Variable | Default | Effect |
|---|---|---|
| `FLYDSL_KTRACE_ENABLE` | `0` | Emit instrumentation |
| `FLYDSL_KTRACE_BUFFER_BYTES` | `64 MiB` | Device trace buffer size |
| `FLYDSL_KTRACE_BLOCKS` | *(all)* | `x,y,z` records one workgroup; `xcc:N` records one XCD |
| `FLYDSL_KTRACE_DUMP_DIR` | *(cwd)* | Where traces are written |

To set a persistent tracing default on one launcher:

```python
flyc.compile[{"ktrace": True}](launch, *args)
```

`flyc.compile[...]` updates that launcher's compile hints for later calls too. Use
`ktrace.capture(...)` for a temporary tracing window; it restores the previous hints
on exit. An explicit `ktrace` hint takes precedence over `FLYDSL_KTRACE_ENABLE`.

### `FLYDSL_KTRACE_BLOCKS` is the lever for large grids

A realistic GEMM launches thousands of identical workgroups. Tracing all of them
overflows the buffer, and an overflowed trace is **rejected, not trimmed** — a partial
trace has unbalanced push/pop and would render as a corrupt timeline.

Filter instead of shrinking the grid, which would change the workload you are measuring:

```bash
FLYDSL_KTRACE_ENABLE=1 FLYDSL_KTRACE_BLOCKS=0,0,0 python my_kernel.py
```

Filtering happens **on the device**, so a non-recording wave writes nothing at all.

## CUDA Graph replay

Compile and launch the traced specialization once before graph capture, and enter the
ktrace capture context outside `torch.cuda.graph()`. The graph holds the compiled
kernel and its trace-buffer address; replay writes new records without calling the JIT
again.

```python
with ktrace.capture("gemm", blocks=(0, 0, 0)) as trace:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch_gemm(..., stream=fx.Stream(stream))  # compile and prepare

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            launch_gemm(..., stream=fx.Stream(stream))

    trace.clear()  # discard preparation and graph-construction records
    graph.replay()

print(trace.summary())
print(trace.export())
```

The runnable example is `python examples/07-ktrace_wave_timeline.py --graph`.
Add `--layers 3` to collect distinct traces from three graph nodes that reuse the
same kernel.

The traced launcher must pass its explicit `stream: fx.Stream` to each kernel launch;
an implicit default-stream launch will not enter a graph captured on PyTorch's side
stream. Call `trace.clear()` or `trace.read()` between groups of replays to keep the buffer
within its capacity. `trace.read()` stores a new snapshot and resets the device buffer;
`trace.summary()` and `trace.export()` can then inspect it in either order. After the
context exits, a graph replay still writes to its retained trace buffer; call
`trace.read()` again to inspect that replay. Entering a capture or reading or clearing
the buffer *inside* `torch.cuda.graph()` is invalid because those operations synchronize
the device. A graph created from an untraced specialization does not acquire trace
instructions when replayed later.

## What the numbers mean

Timestamps come from `s_memrealtime`, a constant-rate 100 MHz counter (10 ns per tick)
that is invariant under clock throttling.
This reference counter is separate from the GPU core clock. Compare durations after
converting ticks to time; raw counter differences are not core-cycle counts.

**Effective resolution is ~40 ns**, because the counter advances in steps of 4 ticks.
Ranges shorter than that collapse to zero duration and render as markers. Instrument at
k-tile granularity, not per instruction.

All XCDs share one coherent clock domain — measured spread across 8 XCDs is under 20
ticks — so timestamps are directly comparable device-wide with no per-XCD correction.

The timeline contains timestamps only where an instrumented wave records an event.
The space between two launch tracks is not a measurement of host launch latency or
complete kernel overlap; the first and last events need not coincide with kernel
entry and exit. Use a runtime profiler for whole-application scheduling questions.

## Costs

Per wave, once at kernel entry: two `s_getreg` reads and one atomic to claim a slot range
(only waves that pass the block filter claim one). Per event: a timestamp, an uncontended
bump of the wave's own counter, and the record store.

**Measured at ~40 ns per event** on MI355X (38-43 ns across two work levels, R² ≥ 0.97) --
about one observed timestamp step, or four counter ticks. That figure holds for loops
whose bodies are not themselves waiting on lgkm traffic; see the next section for a
kernel where it is off by 20x. Budget accordingly:

| Events per wave | Added time per wave |
|---|---|
| 16 | ~0.6 µs |
| 64 | ~2.6 µs |
| 256 | ~10 µs |

A phase shorter than ~1 µs instrumented at per-iteration granularity is measuring itself as
much as the kernel. Annotate k-tiles, not instructions.

Shrinking the record does **not** help. The wave identity (`hw_id`, `xcc_id`) is
wave-invariant and occupies 8 of the 32 bytes, so moving it into the wave's counter
word and writing 4 stores per event instead of 6 looks like a free 25% cut in traffic.
Measured on gfx950 it is not: per-event cost went 345 -> 370 ns at one wave and stayed
inside run-to-run noise at two and four. The dominant cost is the `s_waitcnt
lgkmcnt(0)` that the timestamp read below forces, which is a latency stall independent
of how many stores follow it. The layout is left alone, because the change would make
a record's meaning depend on a different slot parsed earlier -- exactly the kind of
coupling that produces a plausible-looking trace attributed to the wrong wave.

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

This rule is not limited to the hand-interleaved kernel measured above. Some GEMMs in
`kernels/gemm/` use software pipelines. For example, `gemm_a16w16_gfx950` with
`stages >= 2` prefetches k-tile `k + (stages - 1)` while computing k-tile `k`, and
holds that traffic in flight across the loop body with `wait_asyncmark(stages - 2)`.
If adding annotations to such a kernel, use phase granularity (`prologue`,
`mainloop`, `drain`, `epilogue`) rather than sub-phases inside the pipeline.

A loop qualifies as *not* pipelined -- and so is safe to sub-phase -- only when its body
waits for its own memory before using it, leaving nothing in flight across the timestamp:
an unstaged loop that issues a load, waits on it, and computes, with no prefetch of a
later iteration.

## How it is implemented

An annotation is an MLIR op, not inlined code. `ktrace.mark("x")` emits a single
`fly_ktrace.mark`; the code that actually writes a record -- the leader guard, the slot
claim, the bounds checks and the six stores -- is produced later by a conversion pass.

| Piece | Where |
|---|---|
| Public API | [`python/flydsl/expr/experimental/ktrace.py`](../python/flydsl/expr/experimental/ktrace.py) |
| Runnable example | [`examples/07-ktrace_wave_timeline.py`](../examples/07-ktrace_wave_timeline.py) |
| Dialect: 6 ops + `!fly_ktrace.token` | `include/flydsl/Dialect/FlyKtrace/` |
| The expansion | `lib/Conversion/FlyKtraceToROCDL/` |
| Where the pass runs | [`python/flydsl/compiler/backends/rocm.py`](../python/flydsl/compiler/backends/rocm.py) |
| ROCm buffer and trace provider | [`rocm_ktrace_buffer.py`](../python/flydsl/runtime/device_runtime/rocm_ktrace_buffer.py), [`_rocm_ktrace.py`](../python/flydsl/compiler/backends/_rocm_ktrace.py) |
| Shared event names and payloads | [`ktrace_common.py`](../python/flydsl/expr/experimental/ktrace_common.py) |

### Backend boundary

The frontend emits target-independent `fly_ktrace` annotations. A compile backend
selects its lowering pass and returns a trace provider through
`BaseBackend.ktrace_backend()`. The shared JIT passes an implicit buffer pointer to
traced kernels; the provider configures the module, owns buffer allocation and
readback, and turns its records into summaries and trace files. A new target adds
its own pass and provider without changing the capture API or JIT trace path.

The lowering pass publishes `fly_ktrace.event_names` for the shared name table.
`launches()` always returns an ordinal `launch_id`; target-specific launch fields
are additional metadata. The ROCm provider owns its record layout, HSA identity,
clock conversion, and stream-capture checks.

`convert-fly-ktrace-to-rocdl` runs just before `convert-fly-to-rocdl`, which is forced
from both sides: the pass emits `fly.to_llvm_ptr` for the trace buffer and only that pass
lowers it, and it emits `scf.if`, which must survive until `convert-scf-to-cf`.

Two settings reach the pass as module attributes rather than being re-read from the
environment, so one compilation cannot disagree with itself about what it built:
`FLYDSL_KTRACE_BUFFER_BYTES` becomes `fly_ktrace.capacity_slots`, and
`FLYDSL_KTRACE_BLOCKS` becomes `fly_ktrace.blocks`. The pass also assigns the event ids
and publishes them as `fly_ktrace.event_names`, which is what the host decodes against.

A kernel with no annotations is left untouched by the pass, and it is skipped entirely
unless tracing is on and the target is gfx942/gfx950.

## Relationship to ATT

Complementary, not competing. ATT (`/capture-kernel-trace`) gives instruction-level
detail post hoc; KTRACE gives the phases you named, with the nesting you wrote. Use ATT to
ask *which instructions stalled*, KTRACE to ask *which phase was slow*.

They cannot run simultaneously in a way that makes both numbers trustworthy — instrument
for one at a time.

## Limitations

- gfx942/gfx950 only.
- One capture window can be active per process. Its compile hints apply to the current
  Python thread, while its device buffer is process-wide. Other threads or retained
  traced graphs can add launches to the same window; isolate the profiling workload
  when you need an exact set of launches.
- A traced specialization cannot be exported through `CompiledFunction.export_to_c()`.
  Export an untraced specialization for that deployment path.
- Events are **wave-level**, not thread-level: one record per wave per event site.
- A wave emitting more than 255 events stops recording rather
  than writing into the next wave's range; the host reports the shortfall.
- **Collection is serialised against traced launches, not lock-free.** A trace read is a
  device copy followed by a separate reset, so a launch landing between them would write
  records the reset wipes. Both sides take one process-wide lock, which means a traced
  launch blocks while another thread is reading the trace. Untraced kernels never take it.
- **The global claim counter is 32-bit and does not saturate.** Every wave adds
  `EVENTS_PER_WAVE` to it, and a read resets it, so it only accumulates between reads.
  Reaching 2^32 takes 16.7M waves in one collection window -- 2048x the 8191 waves that
  already overflow the default buffer, which raises first. If a run somehow got there
  without reading, the counter would wrap to a small value and the overflow check would
  pass on a trace that is missing most of its records. Read the trace periodically rather
  than accumulating across very many launches.
- **One device per process.** The trace buffer is allocated once, on whichever device
  was current at the time, and reading it is refused from another device rather than
  decoding memory that device does not own. Trace multi-device work one device per
  process.
- **Both ends of a range must sit under the same guard.** Events are wave-level and
  the record is written by the first *active* lane, so a `range_start` taken with the
  whole wave active and a `range_end` taken inside `if tid != 0:` are written by
  different lanes: the second one holds no slot to pair on and the range is dropped
  without a diagnostic. Put both endpoints in the same branch, or in the same
  warp-specialised role (`if warp_id == 0:`), and pair `range_push`/`range_pop` the
  same way.
- Sub-phase granularity is unusable on software-pipelined loops: the probes cost 18-23x
  their nominal price there and reshape the schedule they measure; see Costs.
- Prefer small, deterministic workloads. Launches with no recorded events are absent
  from `launches()` and `export_launches()`.
