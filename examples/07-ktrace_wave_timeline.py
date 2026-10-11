# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""In-kernel wave tracing with ``fx.experimental.ktrace`` (gfx942 / gfx950).

Profilers tell you *what* ran; ktrace tells you *which phase you meant* was running.
Each annotation records a timestamped event from inside the kernel, and the host turns
those events into per-phase totals or a Perfetto timeline.

**Experimental** -- the API, the record layout and the ``FLYDSL_KTRACE_*`` settings may
change in any minor release.

Two ways to delimit a phase, both shown below:

  1. ``with ktrace.range(...)`` -- a named phase in one source block.
  2. ``range_start`` / ``range_end`` -- an explicit token, for when the close site is
     *not* lexically nested. Measuring one loop iteration up to the next needs this:
     the ``range_end`` precedes, in source order, the ``range_start`` it pairs with.
     ``sentinel_token`` seeds the loop-carried token so the first pass records nothing.

Tracing is off unless asked for, so an annotated kernel compiles to the same code as an
unannotated one. This example turns it on with a host capture context. Pass
``--graph`` to trace a CUDA Graph replay. ``--layers 3`` runs the same kernel
three times and exports a separate trace for each launch.
"""

import argparse
from pathlib import Path

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

    # (1) The context records a named phase in the user's kernel body.
    with ktrace.range("prologue"):
        acc = fx.Float32(0.0)

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
def scaled_accumulate(src: fx.Tensor, dst: fx.Tensor, scale: fx.Float32, stream: fx.Stream = fx.Stream(None)):
    scaled_accumulate_kernel(src, dst, scale).launch(grid=(M, 1, 1), block=(TILE, 1, 1), stream=stream)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--graph", action="store_true", help="trace one CUDA Graph replay")
parser.add_argument("--layers", type=int, default=1, help="number of repeated kernel launches")
parser.add_argument("--out-dir", type=Path, default=None, help="directory for trace files")
args = parser.parse_args()
if args.layers < 1:
    parser.error("--layers must be at least 1")

src = torch.randn(M, N, dtype=torch.float32, device="cuda")
outputs = [torch.zeros(M, TILE, dtype=torch.float32, device="cuda") for _ in range(args.layers)]
scale = 2.0

with ktrace.capture("scaled_accumulate", blocks=(0, 0, 0), out_dir=args.out_dir) as trace:
    if args.graph:
        graph_stream = torch.cuda.Stream()
        graph_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(graph_stream):
            scaled_accumulate(src, outputs[0], fx.Float32(scale), fx.Stream(graph_stream))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=graph_stream):
                for output in outputs:
                    scaled_accumulate(src, output, fx.Float32(scale), fx.Stream(graph_stream))
        trace.clear()  # keep only replay records
        graph.replay()
    else:
        for output in outputs:
            scaled_accumulate(src, output, fx.Float32(scale))
torch.cuda.synchronize()

expected = (src.view(M, N_TILES, TILE) * scale).sum(dim=1)
numerics_ok = all(torch.allclose(expected, output, rtol=1e-4, atol=1e-4) for output in outputs)

# The context reads on exit; summary and export inspect the same snapshot.
stats = trace.summary()

if stats is None:
    print("FAIL: tracing produced no data")
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
launches = trace.launches()
trace_ok = not missing and stats["records"] > 0 and len(launches) == args.layers

if not trace_ok:
    print(f"FAIL: missing phases {sorted(missing)}; got {sorted(seen)}; launches {len(launches)}")

print(f"Combined trace: {trace.export()}")
if args.layers > 1 and len(launches) == args.layers:
    paths = trace.export_launches(names=[f"scaled_accumulate_layer_{i}" for i in range(args.layers)])
    for i, path in enumerate(paths):
        print(f"Layer {i} trace: {path}")

print(f"Result correct: {numerics_ok}")
print(f"All passed: {numerics_ok and trace_ok}")
if not numerics_ok or not trace_ok:
    raise SystemExit(1)
