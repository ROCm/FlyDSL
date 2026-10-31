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

The CPU fallback is a frozen decision tree over dimensions, alignment,
K factors, compressed footprint and a cache hint. Leaves rank measured successful
configurations. Runtime adjusts K divisors and masked copies, checks legality,
and caches the choice on the CPU. It needs neither a machine-learning package
nor GPU timing. `auto` balances reused and rotating weights; `hot` targets
repeated use of one weight; `rotating` targets cycling through model layers.
`rank_configurations` exposes the selected region and its priority order.
Its priorities are not latency predictions or a guarantee of optimality.

The exported policy was fitted on 198 shapes (126 original search shapes plus
72 development shapes), with separate hot/rotating measurements. On a subsequent
36-shape test across nine unseen N/K families, its geometric-mean latency ratio
to an independently tuned reference was 1.128 with `auto`, and 1.084 with the
matching cache hint; the worst ratios were 3.290 and 2.746. These misses motivate
optional calibration rather than a claim that the CPU policy is optimal.
The JSON records training shapes and the input-data digest. E4M3 is the
performance training domain; E5M2 and exact grouped paths have separate
correctness coverage.

For a deployment's new shapes, calibrate before graph capture:

```python
from flydsl.kernels.gemm.rdna4_mxfp4 import tune_gemm

config = tune_gemm(x_fp8_e4m3, weights, x_scale_fp32,
                   cache_policy="rotating")
y = gemm(x_fp8_e4m3, weights, x_scale_fp32, cache_policy="rotating")
```

Calibration uses FlyDSL's native `Autotuner` with at most 64 legal register/hybrid
configurations. It checks sampled outputs against independent FP64 MXFP4
decoding before timing each candidate. Hot and rotating-weight graph timings
match the requested cache objective; `auto` minimizes their geometric mean.
Compilation, temporary weight copies and timing occur during calibration.
They do not occur on a normal cache miss, which uses the CPU fallback.

Measured choices persist in `FLYDSL_AUTOTUNE_CACHE_DIR` (default
`~/.flydsl/autotune`) and are keyed by source, shape, dtype, layout, scale mode,
cache objective, device and FlyDSL's compiler/environment fingerprints.
Different processes serialize writes with a file lock. Ordinary GEMM reads this
cache without timing. `FLYDSL_AUTOTUNE=1` requests calibration on first use in
the process and must be used before graph capture; exact grouped plans retain
the conservative register fallback. Recalibrate after changing compiler or
kernel sources. Calibration finds the best measured member of its bounded
shortlist, not a mathematical optimum over all possible configurations.

Explicit configuration is available for offline experiments:

```python
y = gemm(x_fp8_e4m3, weights, x_scale_fp32,
         config={"engine": "register", "waves_m": 1, "waves_n": 4,
                 "repeat_m": 1, "repeat_n": 1, "k_unroll": 4})
```

Run focused tests from a built FlyDSL checkout:

```bash
python -m pytest tests/unit/test_rdna4_mxfp4_dispatch.py
python -m pytest tests/unit/test_rdna4_mxfp4_tuning.py
python -m pytest tests/kernels/test_rdna4_mxfp4.py
```

## Port performance validation

The source port was compared with the preceding native implementation on two Radeon AI PRO R9700 GPUs at default clocks, using identical explicit configurations. Eight shapes, both GPUs, hot/rotating weights and 15 randomized paired rounds give 32 latency comparisons. Port/reference latency is 0.9991× geometrically; the worst is 1.0064×. Every pair is bitwise equal, reads changed inputs on graph replay, and has the same decoded instruction fingerprint. All executed kernels use native FP8/BF8 WMMA with zero scratch or register spills. This isolates the source port from the quality of default configuration selection.

The table contains geometric means across GPUs. Ratios above one are slower. Raw timing rounds, configurations, ELF/ISA hashes and resource metadata are in [the validation data](../../../tests/perf/data/rdna4_mxfp4_port.json). Validation uses this branch’s exact kernel sources with the installed FlyDSL 0.3.4.1 compiler; the compiler SDK was not rebuilt.

| M | N | K | Hot port/reference | Rotating port/reference |
|---:|---:|---:|---:|---:|
| 1 | 5120 | 17408 | 0.9974× | 0.9996× |
| 8 | 34816 | 5120 | 1.0022× | 0.9995× |
| 64 | 8192 | 4096 | 1.0021× | 1.0010× |
| 73 | 4099 | 1568 | 1.0021× | 1.0025× |
| 128 | 17408 | 5120 | 0.9941× | 0.9865× |
| 512 | 5120 | 8704 | 0.9986× | 1.0003× |
| 8191 | 6144 | 8192 | 0.9991× | 0.9998× |
| 8192 | 5120 | 17408 | 1.0000× | 1.0004× |

## Benchmark and deployment warmup

From a checkout with the kernel package installed, benchmark reused and rotating weights:

```bash
python tests/perf/bench_rdna4_mxfp4.py --m 1 8 128 8192 --nk 5120x17408 --tune
```

For normal E4M3-foldable deployment weights, pre-calibrate specified shapes without a serving-time search:

```bash
FLYDSL_AUTOTUNE_CACHE_DIR=/path/to/cache \
python tests/perf/bench_rdna4_mxfp4.py --gpu 0 --m 1 2 4 8 16 \
  --nk 17408x5120 5120x8704 --cache-policy rotating --tune --calibrate-only \
  --output warmup.json
```

The helper uses synthetic operands with constant E8M0 exponents, so its cache entries cover normal E4M3-foldable plans. For other scale modes, call `tune_gemm` with actual prepared weights. Repeat for each device and share the same cache directory with the serving process. Compilation and calibration costs are reported separately. Ordinary GEMM timing excludes those costs. Reported effective GB/s uses minimum logical traffic rather than hardware DRAM counters.
