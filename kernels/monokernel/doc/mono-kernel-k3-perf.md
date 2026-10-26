# Kimi-K3 mono-kernel performance

**Model: Kimi-K3 | Hardware: 8 × AMD Instinct MI355X**

## Implementation highlights

1. **Full decoder-layer fusion.** A single GPU kernel combines both AttnRes operations, Kimi Delta Attention (KDA), routing, latent and shared MoE computation, TP reductions, and the final residual addition.
2. **Multi-token support.** The kernel supports batch sizes 1–8 and 1–8 consecutive tokens per request, for up to 64 tokens per invocation. MTP state snapshots, MXFP8 scale handling, and synchronization mailboxes cover the expanded shapes. Input staging and accumulation order are selected by shape.
3. **Checkpoint-backed comparison.** The ATOM production-layer baseline and the mono-kernel use weights from the same checkpoint. The mono-kernel passed strict state-replay checks and full-grid residency checks on all eight GPUs.

## Compared implementations and configuration

The baseline is ATOM revision `a3e2b1a2551f` from ATOM PR `#2435`, with the mono-kernel disabled and production operator fusions enabled. The candidate is the FlyDSL Kimi-K3 mono-kernel based on Opt254, with the multi-token extension.

Shared configuration: **TP8 / DCP1, layer 1 (KDA + MoE)**; hidden dimension 7,168, latent dimension 3,584, and 896 experts with top-16 routing. Unprofiled measurements were collected on node 47 with PyTorch 2.9.1 and ROCm 7.1.1. ATOM used FlyDSL 0.3.2 and Triton 3.7.0; the mono-kernel used a custom FlyDSL runtime.

| Component | ATOM production layer | FlyDSL mono-kernel |
| --- | --- | --- |
| KDA / shared projections | PTPC FP8 | BF16 / MXFP8 |
| Latent projections | MXFP4 | MXFP8 |
| Routed experts | A4W4 | A16W4 |
| Recurrent state | FP16 | FP32 |
| TP reduction | Separate reductions for each branch | Combined final reduction |

The ATOM timing boundary is **materialized**: one additional elementwise kernel computes `BF16(FP32(prefix) + FP32(routed) + FP32(shared))` from the native outputs to match the mono-kernel's complete layer-output boundary. The comparison includes differences in numerical formats, state layouts, and communication strategies. The checks validate each implementation's operator and state handling; they do not establish numerical equivalence between ATOM and the mono-kernel.

## B1 / S1 profiling: 26 kernels → 7 kernels → 1 kernel

![Kimi-K3 kernel durations grouped by function](https://raw.githubusercontent.com/ROCm/FlyDSL/e650da587af17319520b51b4d1b73e24dbe488a8/kernels/monokernel/doc/assets/k3-perf/k3_b1s1_stream_trace.webp)

<sub>The Kimi-K3 single-kernel implementation shown here will be upstreamed soon.</sub>

Each excerpt selects **GPU 0, seed 1234, the ninth layer invocation in the final graph of 16 calls**. The ATOM path contains **25 native kernels plus one output-materialization kernel**; the FlyDSL staged path contains seven kernels; the full mono-kernel path contains one. Colors group kernel names by function, and the plot preserves each kernel's original duration.

The summed kernel durations are **137.92 / 90.96 / 72.40 μs** for ATOM, staged FlyDSL, and the mono-kernel, respectively. Overlapping work on different queues is counted separately, so these sums are not layer elapsed times. The performance table below uses separate measurements with profiling disabled.

<sub>ATOM and mono-kernel traces: node 47, ROCm 7.1.1. Seven-kernel trace: node 46, ROCm 7.2, using the same source weights and kernel source. The node and runtime differences apply to the illustrative profiling comparison.</sub>

## Performance results

**Full-layer latency: ATOM (materialized output) vs. FlyDSL mono-kernel — configurations with measured speedup**

| Batch size | Tokens per request | ATOM layer (μs/layer) | Mono-kernel (μs/layer) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 142.97 | 75.70 | **1.89×** |
| 1 | 2 | 148.34 | 97.69 | **1.52×** |
| 1 | 4 | 163.12 | 119.73 | **1.36×** |
| 2 | 1 | 147.22 | 93.25 | **1.58×** |
| 4 | 1 | 155.87 | 112.38 | **1.39×** |

<sub>Batch size is the number of requests. Tokens per request (S) includes the current token and S − 1 speculative tokens. Latency covers one complete layer for the entire batch.</sub>

<sub>Each graph invokes the same layer 16 times, reusing its weights and buffers. Each seed uses 50 timed graph replays. For each replay, the maximum elapsed time across eight TP ranks is divided by 16; results take the median across replays, followed by the median across seeds 1234, 2025, and 3141. State resets and synchronization are outside the timed region.</sub>

<sub>The table lists only configurations where the mono-kernel is faster; speedups do not apply to all supported shapes.</sub>
