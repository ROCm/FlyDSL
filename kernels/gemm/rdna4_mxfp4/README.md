# RDNA4 MXFP4 × FP8 GEMM

Dense OCP MXFP4 weights and static FP8 E4M3 activations on ROCm gfx1200/gfx1201.
The kernels use FlyDSL layouts, tiled copies and WMMA atoms. Weights stay
compressed; conversion to exact FP8 operands happens inside GEMM, followed by
native FP8 WMMA and FP32 accumulation. Hardware FP4 instructions are unnecessary.
Performance validation currently covers gfx1201 (Radeon AI PRO R9700).

```python
import torch
from flydsl.kernels.gemm.rdna4_mxfp4 import prepare_weights, gemm, choose_config

# packed: uint8 [N, K/2], even K in low nibble, odd K in high nibble.
# scales: uint8 [N, K/32], OCP E8M0 exponents.
weights = prepare_weights(packed, scales)
y = gemm(x_fp8_e4m3, weights, x_scale_fp32, cache_policy="rotating")
print(choose_config(x_fp8_e4m3.shape[0], weights.n, weights.k, weights.mode))
```

M and N can have tails. K must be a positive multiple of 32. Inputs and output
are contiguous on the same GPU; output is BF16 by default or FP16. Activation
scales are FP32 [M] or [M, 1]. Each addressed buffer must be smaller than 4 GiB.
Preparation is a load-time operation; GEMM supports current-stream execution
and graph replay with changed inputs. Prepared storage owns its device.

Preparation retains 4 bits per weight, one E8M0 byte per 32 weights, padding to
16 output columns, and one FP32 reference exponent per padded output column for
folded plans. It does not retain dense decoded weights. Auto preparation uses
lossless E4M3 folding, then E5M2 folding for wider exponent spans, then exact
grouped K32 rescaling. Forced folding raises when it would lose information.

## Two strategies

| Strategy | Dataflow | Configuration parameters |
|---|---|---|
| Register | Copy A and compressed B into register fragments; decode B before WMMA | M/N wave distribution and repeats, K32 unroll, masked buffer copies, byte-conversion method, epilogue |
| Hybrid | Keep A in registers; cooperatively decode and stage B in LDS with operand lookahead | M/N/K tile, wave distribution, one/two B stages, operand lookahead, M cache grouping, B padding/swizzle, barriers/scheduling, epilogue |

These are algorithm strategies. Tile sizes, unroll, stages and LDS layouts are
configurations of a strategy. Exact grouped scaling uses register GEMM.
Cooperative A+B LDS GEMM and historical HIP-style ports are not included.

The default shape policy is a frozen decision tree over dimensions, alignment,
K factors, compressed footprint and a cache hint. Leaves rank measured successful
configurations. Runtime adjusts K divisors and masked copies, checks legality,
and caches the choice on the CPU. It needs neither a machine-learning package
nor online tuning. `auto` balances reused and rotating weights; `hot` targets
repeated use of one weight; `rotating` targets cycling through model layers.
`rank_configurations` exposes the selected region and its priority order.
Its priorities are not latency predictions or a guarantee of optimality.

The exported policy was fitted on 198 shapes (126 original search shapes plus
72 development shapes), with separate hot/rotating measurements. A separate 36-shape cohort is reserved for final validation. The JSON
records its training shapes and input data digest. E4M3 is the performance
training domain; E5M2 and exact grouped paths have separate correctness coverage.

Explicit configuration is available for offline experiments:

```python
y = gemm(x_fp8_e4m3, weights, x_scale_fp32,
         config={"engine": "register", "waves_m": 1, "waves_n": 4,
                 "repeat_m": 1, "repeat_n": 1, "k_unroll": 4})
```

Run focused tests from a built FlyDSL checkout:

```bash
python -m pytest tests/unit/test_rdna4_mxfp4_dispatch.py
python -m pytest tests/kernels/test_rdna4_mxfp4.py
```
