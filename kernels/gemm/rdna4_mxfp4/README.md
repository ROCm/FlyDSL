# RDNA4 MXFP4 × FP8 GEMM

Dense OCP MXFP4 weights and FP8 E4M3 activations on ROCm gfx1200/gfx1201.
The two FlyDSL strategies use layouts, tiled copies and WMMA atoms. Weights
stay compressed; conversion to exact FP8 operands happens inside GEMM,
followed by native FP8 WMMA with FP32 accumulation. Hardware FP4 instructions
are unnecessary. Performance measurements currently cover gfx1201 / R9700.

```python
from flydsl.kernels.gemm.rdna4_mxfp4 import prepare_weights, gemm

# packed: uint8 [N, K/2], even K in low nibble, odd K in high nibble.
# scales: uint8 [N, K/32], OCP E8M0 exponents.
weights = prepare_weights(packed, scales)
y = gemm(x_fp8_e4m3, weights, x_scale_fp32, cache_policy="rotating")
```

M and N can have tails. K must be a positive multiple of 32. Inputs and output
are contiguous on the same GPU; output is BF16 by default or FP16. Activation
scales are contiguous FP32 [M] or [M, 1]. Each addressed buffer must be smaller
than 4 GiB. GEMM supports the current stream and graph replay with changed inputs.

Preparation runs once during weight loading. Storage retains 4 bits per weight,
one E8M0 byte per 32 weights, padding to 16 output columns and one reference
exponent per padded column for folded plans. It does not retain decoded weights.
Auto preparation uses lossless E4M3 folding, then E5M2 folding for wider exponent
spans, then exact grouped K32 rescaling. Forced folding raises when it would
lose information. Grouped scaling uses the register strategy.

## Source structure

```text
rdna4_mxfp4/
├── __init__.py       # Lazy public exports
├── entrypoint.py     # Preparation, device capacities, launch, compile-only warmup
├── register.py       # Register-fed GPU algorithm
├── hybrid.py         # A registers / B LDS GPU algorithm
├── heuristic.py      # Pure host selector and named configuration presets
├── common.py         # Shared device layouts, unpacking and folding
└── README.md
```

`gemm()` calls `choose_config()` directly, then launches register or hybrid.
There is no runtime tuning cache, configuration search, profile lookup, JSON
load or learned model. `FLYDSL_AUTOTUNE` does not affect this kernel. The
previous `tune_gemm` API has been removed; compare explicit configurations
with the offline benchmark and use `gemm(config=...)` for an explicit override.

## Host selection

`choose_config(M, N, K, mode, cache_policy=..., num_processors=...,
cache_size_bytes=..., lds_size_bytes=...)` uses only the Python standard library.
It returns a fresh configuration dictionary and caches decisions on the CPU.
It can run without importing Torch, querying a GPU, compiling or timing kernels.

Selection uses static decode versus aligned/partial row classes, N/K aspect
ratio, K32 divisors, output tile coverage per processor, padded tile work and
the compressed weight footprint relative to cache capacity. Register rules
choose wave ownership, row reuse and K unrolling together, preserving enough
independent waves when extra reuse reduces lookahead. Narrow output rules
avoid unnecessary N padding and bound four-N-wave tail layouts to five-group
K lookahead. Cacheable K64 families can use legal register K96 iterations.
Named hybrid presets cover small
batches, narrow outputs, partial M tiles and larger prefill work. The selector
adapts K tiles to legal divisors and checks accumulator, thread and LDS limits.
K32 hybrid tiles widen N only when padding and CTA coverage permit it. Partial
row tiles with long power-of-two K can use a two-stage K128 pipeline; the hot
policy retains the smaller tile when weights fit within half the cache budget.
These are measured shape classes, not a guarantee about cache residency.
Tile sizes, stages, prefetch and wave distribution are configurations of the
two algorithms, rather than additional algorithms.

| Strategy | Tile geometry | Configuration controls |
|---|---|---|
| Register | `M=16*waves_m*repeat_m`, `N=16*waves_n*repeat_n`, `K=32*k_unroll` | Wave ownership, row/column reuse, legal K32 unroll 1..16, nibble conversion and epilogue |
| Hybrid | M 16/32/64/96/128/192/256, N 32/64/128, K 32/64/128/256 | Fifteen named presets; wave ownership, one/two LDS stages, lookahead, M grouping, LDS layout, scheduling and epilogue |

The hybrid K tile must be a power of two and divide K. Register unrolling can
use odd K32 divisors. Staging is reduced when the supplied LDS budget requires
it, with a legal register fallback if the hybrid tile still cannot fit.

For folded weights, the footprint estimate is
`ceil(N/16)*16 * (17*(K/32) + 4)` bytes. `hot` hints that a weight is reused;
`rotating` describes cycling between model layers; `auto` uses the balanced
rules. These hints do not inspect memory residency. Empirical thresholds describe
broad hardware regimes; model names and exact shape-to-configuration tables
are absent. A finite palette cannot guarantee the fastest possible launch for
every unseen shape.

`get_device_capabilities(device_id)` reads and caches processor count, last-level
data cache and per-block LDS during weight preparation. AMD SMI supplies the
last-level cache through Torch's HIP-to-device mapping; Torch's L2 size is the
fallback. Capacity defaults in the pure selector describe R9700. Use the queried
values when inspecting the configuration selected on another device:

```python
from flydsl.kernels.gemm.rdna4_mxfp4 import choose_config, get_device_capabilities

hardware = get_device_capabilities(weights.packed.device.index)
config = choose_config(M, weights.n, weights.k, weights.mode, cache_policy="rotating", **hardware)
```

## Graph startup

`get_warmup_keys` and runtime share the same selector and executable identity.
Pass capacities from initialization into warmup, including when compiling in
another process. Enumeration and compilation use no GPU allocations or dummy
launches. Existing static decode and aligned/partial prefill specializations
cover row counts through the configured maximum.

```python
from flydsl.kernels.gemm.rdna4_mxfp4 import get_warmup_keys, compile_gemm

keys = get_warmup_keys(
    weights.n,
    weights.k,
    max_m=8192,
    mode=weights.mode,
    normal_only=weights.normal_only,
    dtype=y.dtype,
    device_id=weights.packed.device.index,
    cache_policy="rotating",
    **hardware,
)
for key in keys:
    compile_gemm(key)
```

Call compile-only warmup during single-threaded startup. vLLM registers these
keys with its JIT warmup registry. Prepared weights retain their owning device;
execution uses that device's current stream.

## Offline measurements

`tests/perf/bench_rdna4_mxfp4.py` compares the analytical choice with an optional
JSON list supplied through `--configs`. Results are measurement artifacts and
never change normal dispatch. It validates sampled FP32 reference outputs and
changed-input replay, then measures paired captured GPU events for hot weights
and read-evicted weights with warm activations. Eviction runs outside timing.

```bash
python tests/perf/bench_rdna4_mxfp4.py --m 1 8 128 8192 --nk 5120x17408
python tests/perf/bench_rdna4_mxfp4.py --m 128 --configs candidates.json --output timings.json
python -m pytest tests/unit/test_rdna4_mxfp4_heuristic.py tests/kernels/test_rdna4_mxfp4.py
```

The earlier profile-selector validation is retained as historical data in
`tests/perf/data/rdna4_mxfp4_comprehensive.json` and
`rdna4_mxfp4_structure.json`; its timings describe the earlier selector.
The [analytical selector evidence](../../../tests/perf/data/rdna4_mxfp4_analytical.json)
records source hashes, paired latency summaries, worst cases, independent
cohort provenance, startup coverage, ISA/resources and model evaluation scope.
This benchmark evidence is outside the installed kernel package.

The [selector refinement evidence](../../../tests/perf/data/rdna4_mxfp4_refinement.json)
compares the refined host rules with the released analytical selector and the
earlier profile selector. It separates development measurements, reserved unseen
shapes, changed-input replay, compile-only coverage and model dispatch equivalence.
The register and hybrid GPU implementations are unchanged by this refinement.
