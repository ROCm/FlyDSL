# Mono-kernel performance reports

- [GLM mono-kernel performance](mono-kernel-glm-perf.md): TP8, concurrency 1,
  with and without the fused indexer, compared against the ATOM layer.
- [Kimi-K3 mono-kernel performance](mono-kernel-k3-perf.md): TP8 KDA + MoE
  layer comparison against ATOM, including the materialized output boundary.

Each report includes implementation details, benchmark configuration, profiling
figures, and per-layer latency results. Referenced images, traces, and measurement
data are included under `assets/`.
