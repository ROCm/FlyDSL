# Flytrace in-kernel profiler

`flydsl.extension.flytrace` records named events and ranges from inside GPU
kernels and exports a JSON trace that can be opened in Perfetto. Annotations
compile away when no capture is active, so the same API can remain in GEMM,
attention, collective, elementwise, and persistent-kernel sources.

## Annotate and capture

Use `mark()` for an instant, `range_start()`/`range_end()` for an explicitly
paired range, and `range_push()`/`range_pop()` for nested ranges. The
`boundary()`/`end()` pair is a compact FlyDSL extension for adjacent phases:

```python
from flydsl.extension import flytrace

@flyc.kernel(known_block_size=[256, 1, 1])
def kernel(iterations: fx.Int32):
    flytrace.range_push("kernel")
    flytrace.boundary("setup")
    i = fx.Int32(0)
    while i < iterations:
        flytrace.boundary("iteration", i)
        i = i + fx.Int32(1)
    flytrace.end()
    flytrace.range_pop()

with flytrace.capture(
    block=None,
    mode="auto",
    max_blocks=128,
    max_events=8192,
) as cap:
    launch(...)

stats = cap.export("flytrace.json")
```

Open `flytrace.json` in <https://ui.perfetto.dev/>. `stats` contains the number
of waves and records plus any records dropped because `max_events` was reached.
Overflow does not abort the workload and is also shown as a warning in the
trace.

Explicitly paired ranges can cross source-level scopes when the token remains
available. Start and end payload forms must match:

```python
token = flytrace.range_start("load", tile_id)
# kernel work
flytrace.range_end(token, tile_id)
```

The shorter `push()` and `pop()` spellings remain supported for existing
kernels.

## Recorder modes and block selection

- `mode="auto"` selects the compact static format when control flow and payloads
  can be reconstructed at compile time; otherwise it uses the dynamic recorder.
- `mode="static"` requires compile-time event structure. A runtime grid can only
  be used with explicitly selected block coordinates.
- `mode="dynamic"` records the executed site and payload, so it supports runtime
  `for`/`while`, conditions, payloads, and launch dimensions.
- `block=(x, y, z)` captures one block. A sequence such as
  `block=[(0, 0, 0), (1, 0, 0)]` captures several. `block=None` captures the
  grid, capped by `max_blocks`.

All waves in each selected block are recorded. The public API and trace schema
are target-neutral; platform lowering, buffer ABI, clocks, and hardware
identity decoding live behind a backend interface. The current ROCm backend
supports gfx942 and gfx950, including launches on non-default streams. The
optional `hardware=True` ATT identity/merge path currently requires gfx942.
Additional compiler targets can implement `flytrace.TraceBackend` and register
their implementation with `flytrace.register_backend(target_name, factory)`;
operator annotations and host capture code do not change.

Annotations are operator-independent: any `@flyc.kernel` reached by the
captured `@flyc.jit` launch can emit trace events. A capture may therefore
contain several different kernels and exports all of their recorded waves in
one timeline. For repeated calls to the same compiled launcher, the capture
retains the latest recording for that specialization. Kernels without
annotations remain valid and contribute no events.

## Production GEMM and MegaMoE

The production preshuffle GEMM emits `gemm`, `prologue`, `mainloop`, `k_tile`,
and `epilogue` ranges. MegaMoE stage 1 and stage 2 emit dispatch, synchronization,
work-loop, GEMM, and combine ranges. The MegaMoE test tool can capture one eager
forward per rank:

```bash
torchrun --nproc_per_node=8 tests/kernels/test_mega_moe_v2.py \
  --mega-only --tokens 64 --skip-acc --flytrace \
  --flytrace-max-blocks 128 --flytrace-max-events 8192 \
  --profile-dir /tmp/mega_flytrace
```

Each rank writes `*_rankN_flytrace.json` and a matching
`*_rankN_flytrace_summary.json`. Lower `max_blocks` or `max_events` when a full
capture would use too much device memory; one capture is limited to 512 MiB.
