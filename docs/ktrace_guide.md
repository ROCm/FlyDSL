# KTRACE: in-kernel wave tracing

**Experimental.** The API, record format, and output schema may change in a minor
release. See [API stability](api_stability.md).

KTRACE records named phases inside kernels and exports a timeline for Perfetto.
It currently supports **gfx942 (CDNA3) and gfx950 (CDNA4)**, with one traced device
per process. Each event records one active lane's timestamp and optional payload
for its wave.

## Quick start

Add coarse ranges to the kernel:

```python
from flydsl.expr import ktrace

@flyc.kernel
def gemm(...):
    with ktrace.range("prologue"):
        ...
    with ktrace.range("mainloop"):
        ...
    with ktrace.range("epilogue"):
        ...
```

Enable tracing around the host launch, then open the exported
`gemm_ktrace.pftrace.gz` at [ui.perfetto.dev](https://ui.perfetto.dev):

```python
with ktrace.capture("gemm", blocks=(0, 0, 0)) as trace:
    launch_gemm(...)

print(trace.summary())
path = trace.export()
```

The trace nests tracks as kernel → launch → CU (`XCD/SE/SH/CU`) → `WG(x, y, z)` →
`WaveN`. Each wave track starts with a `Wave Span` slice from its first to its last
event; `range_push` ranges nest inside it and marks appear as instants. Token ranges
(`range_start`/`range_stop`) may overlap, so they go on `ranges N` child tracks. An event's
payload appears as its `payload` argument. A range still open at the wave's last event
ends there and is labelled `(unclosed)`. The block coordinates and wave index come
from each wave's buffer header; no annotation is needed.

`blocks=(x, y, z)` selects one workgroup without changing the launched grid.
Start with one workgroup and a few phases. Each wave can record **255 events per
launch**; a range consumes two. Read or clear the trace between groups of launches
to avoid filling the buffer.

The capture enables tracing for JIT specializations compiled inside it and reads
the buffer on exit. `summary()` and `export()` use the same snapshot and return
`None` when it contains no events. Calling them inside the capture keeps the
records read so far. The snapshot remains available after exit. A
`CompiledFunction` compiled before the capture runs untraced; the capture logs a
warning when it records nothing.

Tracing is disabled by default. Disabled annotations emit no trace instructions.
The first traced and untraced calls compile separate specializations; later calls
reuse them. Leaving a capture restores the previous compile hints. A graph retains
the specialization used during its construction.

See [the runnable example](../examples/07-ktrace_wave_timeline.py):

```bash
python examples/07-ktrace_wave_timeline.py --layers 3
```

## Kernel API

| Function | Purpose |
|---|---|
| `range(name, payload=None)` | Context manager for a named phase |
| `mark(name, payload=None)` | Point event |
| `range_push(name, payload=None)` | Open a stack range |
| `range_pop()` | Close the innermost stack range |
| `range_start(name, payload=None)` | Open a range and return its `RangeToken` |
| `range_stop(token, payload=None)` | Close a token range |
| `inactive_token(name)` | Inactive token for a loop-carried range |

Use `with ktrace.range(...)` when both endpoints fit in one source block.
`range_push` and `range_pop` pair in LIFO order along the executed path. A token
range pairs on its token. The lanes that reach a `range_stop` can hold different
tokens, as after a loop they leave at different iterations: the stop closes the
latest range any of them started, and records nothing if none did.

A wave numbers its events in a register. An event in a lane-divergent branch
takes the wave's next slot. After every branch or loop that records, the lanes of
the wave agree on the count, so later events continue past every slot it used,
whichever lanes record them. When both branches of an `if` record, as in
`if lane < 32: mark("a")` with `else: mark("b")`, the traced kernel runs the
branches as two `if`s on the same condition, the then branch first, so a wave
that runs both keeps both records. Under a wave-uniform condition, such as the
warp role, the agreement compiles to nothing and the second `if` adds one
uniform branch.

Use tokens when the range crosses loop iterations:

```python
stage = ktrace.inactive_token("k_tile")
for i in fx.range(0, k_tiles, 1):
    ab_full = ab_consumer.wait_and_advance()
    ktrace.range_stop(stage)
    stage = ktrace.range_start("k_tile")
    ...
ktrace.range_stop(stage)
```

This records intervals between wait boundaries. Sample iterations or use a
single outer range when the loop would exceed the per-wave event limit.

### Payloads

Payloads accept Python `bool`, `int`, and `float`; FlyDSL integers up to 64 bits;
and `Float16`, `BFloat16`, `Float32`, and `Float64`. Python integers must fit in
signed or unsigned 64 bits. Python floats use 64 bits; half-precision values are
promoted to `float32`.

Signedness and numeric values are preserved in the output. Non-finite floats are
shown as `nan`, `inf`, or `-inf`. An omitted payload adds no argument; an explicit
`0` remains `0`. Use wave-uniform payloads: under divergence, the first active lane supplies
the recorded value.

## Captures and repeated launches

A capture accumulates all launches until a read or clear. Repeated calls to the
same kernel appear on separate timeline tracks.

| Capture method | Effect |
|---|---|
| `clear()` | Synchronize and discard device records and the saved snapshot |
| `read()` | Synchronize, save a fresh snapshot, and reset the device buffer |
| `summary()` | Return per-phase totals for the snapshot |
| `launches()` | Return separate totals and metadata for each recorded launch |
| `export()` | Write a combined gzipped Perfetto trace |
| `export_launches(names=None)` | Write one gzipped Perfetto trace per recorded launch |

```python
with ktrace.capture("model", blocks=(0, 0, 0), out_dir="traces") as trace:
    for layer in layers:
        launch_layer(layer)

for launch in trace.launches():
    print(launch["launch_id"], launch["phases"])

paths = trace.export_launches(names=[f"layer_{i}" for i in range(len(layers))])
```

`names` must contain one string per recorded launch, in first-timestamp order.
For overlapping streams, inspect the launch metadata before assigning layer
names. ROCm includes HSA queue and dispatch IDs. Launches with no recorded events
have no per-launch output. Capture windows cannot be nested.

For tracing enabled outside a capture, `ktrace.summary()` and
`ktrace.collect("kernel", out_dir=None)` read and reset the device buffer. They
share a cached snapshot until another traced launch or graph replay occurs.
Untraced calls leave that snapshot available. Inside a capture they raise; use the
capture's methods.

## Graph replay

Enter the KTRACE context outside `torch.cuda.graph()`. Prepare the traced
specialization before graph construction and pass an explicit stream through the
launcher to every kernel launch:

```python
import torch
import flydsl.expr as fx

with ktrace.capture("gemm", blocks=(0, 0, 0)) as trace:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch_gemm(..., stream=fx.Stream(stream))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            launch_gemm(..., stream=fx.Stream(stream))

    trace.clear()
    graph.replay()

print(trace.summary())
print(trace.export())
```

An implicit default-stream launch does not enter a graph captured on PyTorch's
side stream. Entering, reading, or clearing a trace inside graph construction is
invalid because these operations synchronize the device.

The graph retains its traced binary and buffer address after the context exits.
Call `trace.read()` after a later replay to replace the saved snapshot. Read or
clear between replay groups to stay within capacity. For multiple layers, capture
each layer launch as a graph node; `export_launches()` also splits repeated calls
to the same kernel. Run the example with `--layers 3 --graph` to try this.

## Settings

| Variable | Default | Effect |
|---|---|---|
| `FLYDSL_KTRACE_ENABLE` | `0` | Enable instrumentation |
| `FLYDSL_KTRACE_BUFFER_BYTES` | `64 MiB` | Device buffer size; at most 2 GiB |
| `FLYDSL_KTRACE_BLOCKS` | all | `x,y,z` selects one workgroup; `xcc:N` selects one XCD |
| `FLYDSL_KTRACE_DUMP_DIR` | current directory | Trace output directory |

An explicit `ktrace` compile hint overrides the environment. To set a persistent
default for a launcher:

```python
flyc.compile[{"ktrace": True}](launch, *args)
```

Use `ktrace.capture(...)` for a temporary setting. The buffer is allocated once
per process. Keep its size consistent across compilation and execution; cached
and deferred calls validate the compiled bound against the live allocation.

If independently cached kernels have conflicting event IDs, the JIT rebuilds the
conflicting specialization and updates its cache. `FLYDSL_RUNTIME_RUN_ONLY=1`
prohibits that rebuild: prepare those traced kernels together once with run-only
mode disabled.

## Limits and interpreting timings

- **255 events per wave per launch.** Larger buffers and filtering other blocks
  do not increase this limit. Overflow raises during collection because dropped
  range endpoints would make the timeline incomplete.
- **Finite collection window.** Waves reserve 256 slots each, including a header
  slot. A wave takes its slots from one of up to 256 partitions, picked by hashing
  its workgroup, its hardware slot and the dispatch, so waves that start together
  do not contend for one counter. The default 64 MiB buffer has 64 partitions of
  about 128 waves, 8190 waves in all, and a trace overflows when its busiest partition
  fills: one large launch fits reliably up to about 6000 waves. Use block filters
  and periodic reads for large grids or repeated launches. The overflow error names
  a buffer size that holds the recorded launches.
- **One device per process.** Access from another device raises. Trace multiple
  devices in separate processes. Traced binaries cannot be exported to C.
- **Synchronized collection.** Reading waits for device work and serializes with
  traced host dispatches. Coordinate graph replays with reads and clears.
- **Reference-clock timestamps.** `s_memrealtime` runs at 100 MHz (10 ns per tick),
  independently of the GPU core clock. Observed resolution on gfx950 is about
  40 ns. Shorter ranges can appear as zero-duration markers.
- **Instrumentation changes timing.** A wave makes one atomic when it starts, on
  its partition's counter. An event reads the clock and issues two scalar stores,
  with no branch, no atomic and no VGPR. On MI355X an event in a loop cost about
  20 ns, and ranges around the body, the K loop and the epilogue of a 4096-wave FP16
  GEMM cost under 1%; a token range in every K-loop iteration cost about 2% and 16
  more VGPRs. Timestamp consumption can drain outstanding LDS/SMEM traffic, and
  scheduling can move independent work across a timestamp. Use coarse phases and
  compare with an untraced baseline; subtracting a constant per-event cost does not
  undo these scheduling changes. Reading a trace idles the GPU while the host
  decodes it, so time traced launches between reads rather than across them.
- **Partial launch coverage.** The first and last annotations need not coincide
  with kernel entry and exit. Gaps between tracks do not measure host launch
  latency or complete kernel overlap. Use a runtime profiler for those questions.
