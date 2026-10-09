# GLM mono-kernel performance

**Model: GLM-5.2-MXFP4 | Hardware: 8 × AMD Instinct MI355X**

## Implementation highlights

1. **Full decoder-layer fusion.** A single persistent GPU kernel combines RMSNorm, QKV projections and multi-head latent attention (MLA), RoPE and KV-cache writes, MoE routing and both expert GEMMs, and tensor-parallel (TP) communication. With `with_indexer=True`, the same kernel also performs the indexer's Q/K/W projections, scoring, and top-k selection.
2. **MXFP4 memory-access optimization.** E8M0 scales are tiled over 16 rows × 128 elements along the reduction dimension. Groups of eight gate rows and eight up-projection rows are paired before weight shuffling to improve weight and scale access locality while preserving the arithmetic path. See [FlyDSL PR #1241](https://github.com/ROCm/FlyDSL/pull/1241).
3. **Checkpoint-backed comparison.** The benchmark reuses ATOM checkpoint weights, KV caches from an actual prefill, and the corresponding metadata, with synthetic hidden-state inputs. Both implementations use the same graph grouping, layer timing boundaries, and median aggregation across TP ranks. GPU traces verify the dispatch count before and after fusion.

## Compared implementations and configuration

The baseline is the ATOM code used for [ATOM PR #2435](https://github.com/ROCm/ATOM/pull/2435), at revision `16cc652b`, with the mono-kernel disabled and production operator and communication optimizations enabled. The candidate is the FlyDSL GLM mono-kernel benchmark checkout at `109f1cc5`, with native-interface compatibility fixes and the MXFP4 optimization described above.

Shared configuration: **TP8, batch size 1, S ∈ {1, 2, 4, 8}, context length 3,000 tokens, sparse-attention top-k 2,048**; PyTorch 2.10, ROCm 7.2.4, and FlyDSL runtime 0.3.4.1. Both timing boundaries include two TP collectives per layer.

| Component | ATOM production layer | FlyDSL mono-kernel |
| --- | --- | --- |
| MoE GEMM1 / GEMM2 | A4W4 / A4W4 | A8W4 / A16W4 |
| Attention weight quantization | FP8 PTPC | Block-scaled FP8 |
| KV / index cache | FP8 / FP4, paged, block size 64 | BF16 split KV / BF16 contiguous index cache, capacity 4,096 |
| Indexer execution | Separate kernels within the layer | Fused into the layer with `with_indexer=True` |

This compares the two implementations with their respective numerical formats and cache layouts; the measured speedups include those differences.

## B1 / S1 profiling: 34 kernels → 1 mono-kernel

![Recorded GPU stream comparison for the ATOM layer and FlyDSL mono-kernel](assets/glm-perf/glm_b1_s1_stream_comparison.webp)

The paired traces show **rank 0, stream 3, layer 6, including the indexer**. Each excerpt selects the eighth layer invocation in the second complete graph replay. The two windows are aligned at their respective starts, preserving the original kernel durations. ATOM executes **34 GPU kernels**; FlyDSL executes **one mono-kernel**, including the indexer, with no external conversion kernels.

The **194.24 / 54.68 μs** shown in the figure are durations from a single profiled window. **The performance tables below use separate measurements with profiling disabled.** The [original ATOM trace](assets/glm-perf/atom_native_stream.trace.json.gz), [original mono-kernel trace](assets/glm-perf/flydsl_mono_stream.trace.json.gz), and [aligned Perfetto/Chrome Trace excerpt](assets/glm-perf/stream_excerpt.json) are included.

## Performance results

**Without indexer computation: layer 3, reusing precomputed sparse indices**

| S | ATOM layer | Mono-kernel | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 118.214 | 37.464 | 3.16× |
| 2 | 110.626 | 43.636 | 2.54× |
| 4 | 109.197 | 56.067 | 1.95× |
| 8 | 116.685 | 82.966 | 1.41× |

**With indexer computation: layer 6, with the indexer fused into the mono-kernel**

| S | ATOM layer | Mono-kernel (`with_indexer=True`) | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 173.782 | 53.943 | 3.22× |
| 2 | 167.944 | 61.938 | 2.71× |
| 4 | 169.336 | 77.380 | 2.19× |
| 8 | 178.025 | 110.717 | 1.61× |

<sub>All results use Conc 1 (one request, batch size 1). S is the number of consecutive tokens processed for that request: 1, 2, 4, or 8. Latency is reported in μs/layer.</sub>

<sub>Each graph invokes the same layer 16 times, reusing its input and weights; the mono-kernel step counter advances once at the end. After 50 warmup graph replays, timing uses five trials of 256 replays each. Per-rank elapsed time is divided by 256 × 16; each trial takes the median across eight TP ranks, and the reported value is the median of the five trials. Latency includes amortized graph overhead and measures a single layer, not end-to-end time per output token (TPOT).</sub>
