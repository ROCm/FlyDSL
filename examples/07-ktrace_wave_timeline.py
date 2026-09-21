# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""In-kernel wave tracing with ``fx.experimental.ktrace`` (gfx942 / gfx950).

Profilers tell you *what* ran; ktrace tells you *which phase you meant* was running.
Each annotation records a timestamped event from inside the kernel, and the host turns
those events into per-phase totals or a Perfetto timeline.

**Experimental** -- the API, the record layout and the ``FLYDSL_KTRACE_*`` settings may
change in any minor release.

Two ways to delimit a phase, both shown below:

  1. ``range_push`` / ``range_pop`` -- a lexically nested pair, the common case.
  2. ``range_start`` / ``range_end`` -- an explicit token, for when the close site is
     *not* lexically nested. Measuring one loop iteration up to the next needs this:
     the ``range_end`` precedes, in source order, the ``range_start`` it pairs with.
     ``sentinel_token`` seeds the loop-carried token so the first pass records nothing.

Tracing is off unless asked for, so an annotated kernel compiles to the same code as an
unannotated one. This example turns it on with a ``ktrace`` compile hint; setting
``FLYDSL_KTRACE_ENABLE=1`` does the same thing globally.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.experimental import ktrace

M, N = 256, 256
TILE = 64
N_TILES = N // TILE


@flyc.kernel
def scaled_accumulate_kernel(src: fx.Tensor, dst: fx.Tensor, scale: fx.Float32):
    """Row-wise scale-and-accumulate, annotated with the phases we care about."""
    tid = fx.thread_idx.x
    row = fx.block_idx.x

    # (1) Lexically nested pair: the prologue is opened and closed in the same scope.
    ktrace.range_push("prologue")
    acc = fx.Float32(0.0)
    ktrace.range_pop()

    # (2) Token-carried range across loop iterations. Each range covers one tile up to
    # the start of the next, so the close site is the *next* iteration's top -- which
    # range_push/range_pop cannot express.
    tile = ktrace.sentinel_token("tile")
    for t in fx.range(0, N_TILES, 1):
        # Closes the previous iteration's range. The sentinel makes the first pass
        # through here record nothing, since no range is open yet.
        ktrace.range_end(tile)
        tile = ktrace.range_start("tile", t)  # payload = tile index

        col = t * TILE + tid
        acc = acc + src[row, col] * scale
    ktrace.range_end(tile)

    # A zero-duration marker rather than a range: records one instant, with a payload.
    ktrace.mark("store", row)
    dst[row, tid] = acc


@flyc.jit
def scaled_accumulate(src: fx.Tensor, dst: fx.Tensor, scale: fx.Float32):
    scaled_accumulate_kernel(src, dst, scale).launch(grid=(M, 1, 1), block=(TILE, 1, 1))


# Enable tracing for this kernel only. Without this (and without FLYDSL_KTRACE_ENABLE=1)
# every annotation above returns before emitting anything.
scaled_accumulate.compile_hints = {"ktrace": True}


src = torch.randn(M, N, dtype=torch.float32, device="cuda")
dst = torch.zeros(M, TILE, dtype=torch.float32, device="cuda")
scale = 2.0

scaled_accumulate(src, dst, fx.Float32(scale))
torch.cuda.synchronize()

expected = (src.view(M, N_TILES, TILE) * scale).sum(dim=1)
numerics_ok = torch.allclose(expected, dst, rtol=1e-4, atol=1e-4)

# The trace covers every launch since the last read. summary() gives per-phase totals;
# collect("...") instead writes a Perfetto JSON to open at ui.perfetto.dev.
stats = ktrace.summary()

# summary() returns None when tracing was never enabled -- the annotations above then
# emit nothing, which is exactly what a build without the hint is supposed to do.
if stats is None:
    print("FAIL: tracing produced no data; the ktrace compile hint did not take effect")
    print("Result correct: False")
    print("All passed: False")
    raise SystemExit(1)

print(f"waves traced: {stats['waves']}   records: {stats['records']}")
for phase, s in sorted(stats["phases"].items()):
    print(f"  {phase:<10} count={s['count']:<6} total_ns={s['total_ns']:.0f}")

# Every annotated phase must appear, or the trace silently lost events.
seen = set(stats["phases"])
expected_phases = {"prologue", "tile", "store"}
missing = expected_phases - seen
trace_ok = not missing and stats["records"] > 0

if not trace_ok:
    print(f"FAIL: missing phases {sorted(missing)}; got {sorted(seen)}")

print(f"Result correct: {numerics_ok}")
print(f"All passed: {numerics_ok and trace_ok}")
