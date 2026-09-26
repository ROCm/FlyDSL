# Source-tree kernel library guide

FlyDSL's `kernels/` directory contains optimized GPU operators and reusable
implementation patterns used by repository tests, benchmarks, and downstream
integrations.

## Packaging and compatibility boundary

The published `flydsl` wheel contains the compiler and expression-language
package under `python/flydsl`; it does not install `kernels/`.

The source-tree kernel modules are not covered by FlyDSL's stable API policy.
Their signatures, layouts, workspace requirements, supported targets, and
tuning parameters can change between revisions. This guide therefore does not
define public entry points or signatures. Pin the repository revision for an
integration and treat its matching test as the executable contract.

## Library areas

| Area | Source | Tests |
|---|---|---|
| GEMM and low-precision matrix multiplication | `kernels/gemm/` | GEMM-focused files under `tests/kernels/` |
| Normalization and softmax | `kernels/norm/` | LayerNorm, RMSNorm, and softmax tests |
| Attention and sequence operations | `kernels/attention/` | FlashAttention, paged-attention, MLA, and RoPE tests |
| Mixture of Experts | `kernels/moe/` and `kernels/mega_moe/` | MoE routing, sorting, GEMM, and fused-operator tests |
| Convolution | `kernels/conv/` | Implicit-GEMM convolution tests |
| Communication | `kernels/comm/` | Multi-GPU and dispatch/combine tests |
| Shared implementation code | `kernels/common/` | Exercised through the operator tests that consume it |

The directory names describe implementation families, not stable import
names. Some modules are target-specific and some are active tuning work.

## Working with a kernel from the repository

Run from a source checkout so the repository packages and built FlyDSL package
are both importable. The standard scripts configure these paths automatically:

```bash
bash scripts/build.sh -j64
bash scripts/run_tests.sh
```

For focused work:

1. find the relevant source directory under `kernels/`;
2. locate its caller or test under `tests/kernels/`;
3. copy the test's dtype, physical layout, scale format, workspace, stream, and
   architecture constraints;
4. run a small correctness case and boundary shapes before benchmarking;
5. record the FlyDSL revision, ROCm version, target architecture, and exact
   shape with performance results.

Do not infer a callable's contract from its name or from documentation for a
different revision. Low-precision kernels in particular may require
preshuffled weights or scales whose physical ordering is not evident from the
logical tensor shape.

See [Testing and benchmarking](testing_benchmarking_guide.md) for test markers,
runners, profiling, and benchmark reporting.
