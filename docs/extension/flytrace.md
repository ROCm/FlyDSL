# Flytrace wave profiler

`flydsl.extension.flytrace` records phase timestamps from GPU waves and exports
a JSON trace that can be opened in Perfetto. Annotations compile away when no
capture is active, so they can stay in production GEMM and megakernel sources.

## Annotate and capture

Use `boundary()` for adjacent phase ranges, `mark()` for an instant, and
`push()`/`pop()` for nested ranges:

```python
from flydsl.extension import flytrace

@flyc.kernel(known_block_size=[256, 1, 1])
def kernel(iterations: fx.Int32):
    flytrace.push("kernel")
    flytrace.boundary("setup")
    i = fx.Int32(0)
    while i < iterations:
        flytrace.boundary("iteration", i)
        i = i + fx.Int32(1)
    flytrace.end()
    flytrace.pop()

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

## Recorder modes and CTA selection

- `mode="auto"` selects the compact static format when control flow and payloads
  can be reconstructed at compile time; otherwise it uses the dynamic recorder.
- `mode="static"` requires compile-time event structure. A runtime grid can only
  be used with explicitly selected CTA coordinates.
- `mode="dynamic"` records the executed site and payload, so it supports runtime
  `for`/`while`, conditions, payloads, and launch dimensions.
- `block=(x, y, z)` captures one CTA. A sequence such as
  `block=[(0, 0, 0), (1, 0, 0)]` captures several. `block=None` captures the
  grid, capped by `max_blocks`.

All waves in each selected CTA are recorded. gfx942 and gfx950 are supported,
including launches on non-default streams. The optional `hardware=True` ATT
identity/merge path currently requires gfx942.

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
