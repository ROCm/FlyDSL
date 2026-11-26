# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Register-scope micro-scaled quantization (MXFP4 and NVFP4) without shared memory.

Each 32-lane logical warp processes 256 ``Float32`` values (8 per lane):

  1. **MXFP4 (OCP 1x32 block, E2M1 + E8M0 scale)**: Four consecutive lanes hold
     32 values. ``fx.coop.warp_max(..., width=4)`` computes the 32-element absmax
     in registers, ``fx.quant.f32_to_e8m0`` encodes the power-of-two ``E8M0``
     shared exponent, and ``fx.quant.pack_e2m1x8`` packs 8 ``E2M1`` nibbles into
     one ``Uint32`` per lane before dequantizing back to ``Float32``.
  2. **NVFP4 (1x16 micro-block, E2M1 + UE4M3 scale)**: Two consecutive lanes hold
     16 values. ``fx.coop.warp_max(..., width=2)`` computes the 16-element absmax,
     ``fx.quant.f32_to_ue4m3`` encodes the non-negative ``UE4M3`` scale byte, and
     ``fx.quant.pack_e2m1x8`` / ``fx.quant.unpack_e2m1x8`` round-trip the block.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

WIDTH = 32
ELEMS_PER_LANE = 8
TOTAL_ELEMS = WIDTH * ELEMS_PER_LANE


@flyc.kernel
def microscaled_roundtrip_kernel(
    Src: fx.Tensor,
    OutMxfp4: fx.Tensor,
    OutNvfp4: fx.Tensor,
):
    tid = fx.thread_idx.x
    base = tid * ELEMS_PER_LANE

    vals = [Src[base + i] for i in fx.range_constexpr(ELEMS_PER_LANE)]
    local_amax = fx.Float32(0.0)
    for i in fx.range_constexpr(ELEMS_PER_LANE):
        ax = fx.math.absf(vals[i])
        local_amax = (ax > local_amax).select(ax, local_amax)

    # 1) MXFP4: 32-element block across 4 lanes, E8M0 power-of-two scale (E2M1 max = 6.0).
    amax_32 = fx.coop.warp_max(local_amax, width=4)
    e8m0_code = fx.quant.f32_to_e8m0(amax_32 * fx.Float32(1.0 / 6.0))
    mxfp4_scale = fx.quant.e8m0_to_f32(e8m0_code)
    mxfp4_packed = fx.quant.pack_e2m1x8(vals, scale_inv=fx.Float32(1.0) / mxfp4_scale)
    mxfp4_deq = fx.quant.unpack_e2m1x8(mxfp4_packed, scale=mxfp4_scale)
    for i in fx.range_constexpr(ELEMS_PER_LANE):
        OutMxfp4[base + i] = mxfp4_deq[i]

    # 2) NVFP4: 16-element micro-block across 2 lanes, UE4M3 scale (E2M1 max = 6.0).
    amax_16 = fx.coop.warp_max(local_amax, width=2)
    ue4m3_code = fx.quant.f32_to_ue4m3(amax_16 * fx.Float32(1.0 / 6.0))
    nvfp4_scale = fx.quant.ue4m3_to_f32(ue4m3_code)
    safe_inv = (nvfp4_scale > fx.Float32(0.0)).select(fx.Float32(1.0) / nvfp4_scale, fx.Float32(0.0))
    nvfp4_packed = fx.quant.pack_e2m1x8(vals, scale_inv=safe_inv)
    nvfp4_deq = fx.quant.unpack_e2m1x8(nvfp4_packed, scale=nvfp4_scale)
    for i in fx.range_constexpr(ELEMS_PER_LANE):
        OutNvfp4[base + i] = nvfp4_deq[i]


@flyc.jit
def run_microscaled_roundtrip(
    Src: fx.Tensor,
    OutMxfp4: fx.Tensor,
    OutNvfp4: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    microscaled_roundtrip_kernel(Src, OutMxfp4, OutNvfp4).launch(
        grid=(1, 1, 1),
        block=(WIDTH, 1, 1),
        stream=stream,
    )


torch.manual_seed(0)
src = torch.randn(TOTAL_ELEMS, dtype=torch.float32, device="cuda") * 3.5
out_mxfp4 = torch.zeros_like(src)
out_nvfp4 = torch.zeros_like(src)

run_microscaled_roundtrip(src, out_mxfp4, out_nvfp4)
torch.cuda.synchronize()

src_cpu = src.cpu()
mxfp4_cos = torch.nn.functional.cosine_similarity(src_cpu.flatten(), out_mxfp4.cpu().flatten(), dim=0).item()
nvfp4_cos = torch.nn.functional.cosine_similarity(src_cpu.flatten(), out_nvfp4.cpu().flatten(), dim=0).item()

if mxfp4_cos > 0.98 and nvfp4_cos > 0.985 and nvfp4_cos >= mxfp4_cos:
    print(f"PASS (MXFP4 cos={mxfp4_cos:.5f}, NVFP4 cos={nvfp4_cos:.5f})")
else:
    print(f"FAIL (MXFP4 cos={mxfp4_cos:.5f}, NVFP4 cos={nvfp4_cos:.5f})")
    raise SystemExit(1)
