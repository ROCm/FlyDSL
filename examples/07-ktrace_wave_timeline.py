# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors


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
    tid = fx.thread_idx.x
    row = fx.block_idx.x

    with ktrace.range("prologue"):
        acc = fx.Float32(0.0)

    tile = ktrace.sentinel_token("tile")
    for t in fx.range(0, N_TILES, 1):
        ktrace.range_end(tile)
        tile = ktrace.range_start("tile", t)

        col = t * TILE + tid
        acc = acc + src[row, col] * scale
    ktrace.range_end(tile)

    ktrace.mark("store", row)
    dst[row, tid] = acc


@flyc.jit
def scaled_accumulate(src: fx.Tensor, dst: fx.Tensor, scale: fx.Float32, stream: fx.Stream = fx.Stream(None)):
    scaled_accumulate_kernel(src, dst, scale).launch(grid=(M, 1, 1), block=(TILE, 1, 1), stream=stream)


parser = argparse.ArgumentParser(description="Trace repeated kernel launches on gfx942 or gfx950.")
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
        trace.clear()
        graph.replay()
    else:
        for output in outputs:
            scaled_accumulate(src, output, fx.Float32(scale))
torch.cuda.synchronize()

expected = (src.view(M, N_TILES, TILE) * scale).sum(dim=1)
numerics_ok = all(torch.allclose(expected, output, rtol=1e-4, atol=1e-4) for output in outputs)

stats = trace.summary()

if stats is None:
    print("FAIL: tracing produced no data")
    print("Result correct: False")
    print("All passed: False")
    raise SystemExit(1)

print(f"waves traced: {stats['waves']}   records: {stats['records']}")
for phase, s in sorted(stats["phases"].items()):
    print(f"  {phase:<10} count={s['count']:<6} total_ns={s['total_ns']:.0f}")

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
