#!/usr/bin/env python3
"""ONE case where adding warp/wave to FlyDSL does NOT line up with the CUTLASS CTA-multicast.

Case = gemm_mk_iter (nv=128) with a 4-way split:
  CUTLASS CTA-multicast (cta_layout=_4:_1): keeps the layout ((128,32),3) UNCHANGED and only
    shifts the iterator by offset = coord*32 (=nv/4, stepping WITHIN one call). For unit-stride
    s that is coord*0x20 bytes.
  FlyDSL wave (num_warps=4): SPLITS the staged box (s.mode0 = 4096) into 4 contiguous per-warp
    blocks -> per-warp layout becomes (32,32,3) and the WARP-mode stride is 4096/4 = 1024.

So they disagree on BOTH the layout (128->32) and the step granularity (32 vs 1024).

Run (no GPU needed, layout algebra only):
    cd /home/carhuang/xudong/FlyDSL
    PYTHONPATH=build-fly/python_packages COMPILE_ONLY=1 python tdm_warp_mismatch.py
"""
import os

# The layout dump happens during kernel TRACING; a JIT cache hit skips tracing (=> no output).
# Force a re-trace every run so the prints always fire.
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.rocdl import cdna5
from flydsl.expr.primitive import composition, make_tile
from flydsl._mlir.dialects import fly_rocdl
from flydsl.expr.typing import static

# gemm_mk_iter inputs (identical to tdm_partition_check.py), split 4 ways.
S_SH, S_ST = ((64, 64), 3), ((1, 64), 4096)
G_SH, G_ST = ((64, 64), 8), ((1, 64), 4096)
BOX_SH, BOX_ST, TILE = (64, 2), (1, 64), (64, 2)
NW = 4


@flyc.kernel(known_block_size=[256, 1, 1])
def probe(A: fx.Tensor):
    gT = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout(G_SH, G_ST)))
    sT = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout(S_SH, S_ST)))
    atom, _ = cdna5.make_tiled_tdm_atom(fx.rocdl.TensorLoad(), gT, fx.make_layout(BOX_SH, BOX_ST),
                                        TILE, num_warps=NW)

    print("CUTLASS CTA-multicast (cta_layout=_4:_1, nv=128):")
    print("  layout    : ((128,32),3):((1,128),4096)   # UNCHANGED vs single-CTA")
    print("  ptr offset: coord*32  -> cta0=0x00 cta1=0x20 cta2=0x40 cta3=0x60")
    print()

    lv = static(fly_rocdl.tdm_partition_layout(atom.type, sT.type, gT.type, NW))
    print(f"FlyDSL wave (num_warps={NW}):")
    print(f"  layout_V         : {lv}")
    print("                       ^ mode0 = ( ATOM=32 : ITER... ) split into 3 sub-modes:")
    print("                         ATOM=32 stride1 | WARP/COORD=4 stride1024 | ITER=32 stride32")
    print()

    # PRE-slice: apply layout_V to mode0 but DON'T slice the warp -> the COORD dim is visible.
    pre = composition(sT, make_tile(lv, None))       # None = pass through the PIPE rest mode
    print(f"  ps PRE-slice     : {pre.layout}")
    print("                       dims = ( (ATOM=32, COORD=4, ITER=32), PIPE=3 )   <-- 4 real dims")
    print("                       strides = ( (1, 1024, 32), 4096 )")
    print()

    # POST-slice: tdm_partition fixes warp_coord and slices COORD out -> only 3 dims remain.
    for wc in range_constexpr(NW):
        ps, _ = cdna5.tdm_partition(atom, wc, fx.make_layout(NW, 1), sT, gT)
        print(f"  ps POST-slice wc={wc}: {ps.layout}   # COORD=4 sliced out; ptr += wc*1024")
    print("                       dims = ( ATOM=32, ITER=32, PIPE=3 )   <-- COORD gone")
    print()
    print("VERDICT: MISMATCH")
    print("  layout:  CUTLASS keeps ((128,32))  vs  FlyDSL splits to (32,32)")
    print("  offset:  CUTLASS coord*32          vs  FlyDSL WARP stride 1024")
    print("  (the 32 CUTLASS steps by = FlyDSL's ITER stride, not its WARP stride)")


@flyc.jit
def launch(A: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    probe(A).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)


A = torch.zeros(1 << 20, dtype=torch.float16, device="cuda")
launch(A, stream=torch.cuda.current_stream())
