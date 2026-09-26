# Source-tree kernel library guide

FlyDSL's `kernels/` directory is a working library of optimized GPU operators
and reusable implementation patterns. It is used by repository tests,
benchmarks, and downstream integrations.

## 1. Packaging and compatibility boundary

The published `flydsl` wheel contains the compiler/DSL package under
`python/flydsl`; it does **not** install `kernels`. To import a kernel module,
run from a source checkout or add the repository root to `PYTHONPATH`:

```bash
export PYTHONPATH="${PWD}/build-fly/python_packages:${PWD}:${PYTHONPATH}"
```

Kernel modules are not covered by the stable API rules for `flydsl.*`. They
often expose architecture- and model-specific signatures and can change as
tuning evolves. Pin the repository revision in an integration, and use the
matching test as the executable interface contract.

## 2. How to use the library

The modules expose three common layers:

| Layer | Typical names | When to use it |
|---|---|---|
| PyTorch-friendly wrapper | `conv3d_implicit`, `flydsl_flash_attn_func`, `rmsnorm_fwd` | Integration code that wants allocation, validation, caching, and stream handling |
| Specialized builder/compiler | `build_*`, `compile_*`, `create_*_module` | Tests or integrations that own tensor allocation and want a cached launcher for fixed configuration |
| JIT launcher | `launch_*`, decorated functions | Low-overhead internal calls once types, layouts, and constexpr tuning parameters are known |

Do not infer a function's full contract from its name. Read its signature and
docstring, then check the corresponding test for input layout, dtype, shape,
workspace, stream, and architecture rules.

## 3. GEMM

| Module | Main entry points | Scope |
|---|---|---|
| `kernels/gemm/preshuffle_gemm.py` | `compile_preshuffle_gemm` | General CDNA preshuffle GEMM for f16, bf16, fp8, and int8 paths |
| `kernels/gemm/fp8_gemm_4wave.py` | `compile_fp8_gemm_4w` | Four-wave CDNA4 FP8 GEMM with row scaling |
| `kernels/gemm/fp8_gemm_8wave.py` | `compile_fp8_gemm_8w` | Eight-wave CDNA4 FP8 GEMM with optional B preshuffle |
| `kernels/gemm/mxfp8_gemm_8wave.py` | `compile_mxfp8_gemm_8w` | Eight-wave CDNA4 MXFP8 GEMM |
| `kernels/gemm/fp4_gemm_4wave.py` | `compile_fp4_gemm_4w` | gfx950 MXFP4 GEMM with E8M0 block scales |
| `kernels/gemm/mxfp4_preshuffle.py` | `launch_gemm`, `launch_splitk_reduce` | gfx950 MXFP4/MXFP6/MXFP8 operands with preshuffled weights and optional split-K |
| `kernels/gemm/gemm_a16w16_gfx950.py` | `gemm_a16w16`, `make_gemm_a16w16_gfx950_param` | FP16/BF16/FP32-oriented gfx950 GEMM with configurable staging and split-K |
| `kernels/gemm/gemm_bf16_gfx1250.py` | `launch_gemm_bf16` | gfx1250 BF16/FP16 WMMA GEMM using TDM staging |
| `kernels/gemm/gemm_a8w8_gfx1250.py` | `launch_gemm_a8w8` | gfx1250 FP8×FP8 preshuffle GEMM |
| `kernels/gemm/gemm_a8w4_mxscale_gfx1250.py` | `launch_gemm_a8w4_mxscale` | gfx1250 FP8×MXFP4 scaled WMMA GEMM |
| `kernels/gemm/rdna3_f16_gemm.py` | `create_wmma_gemm_module` | gfx11 wave32 FP16/BF16 WMMA GEMM |
| `kernels/gemm/rdna3_int8_gemm.py` | `create_wmma_int8_gemm_module` | gfx11 integer WMMA GEMM |
| `kernels/gemm/rdna_f16_gemm.py` | `create_wmma_gemm_module` | gfx120x wave32 FP16/BF16 WMMA GEMM |
| `kernels/gemm/rdna_fp8_preshuffle_gemm.py` | `compile_fp8_gemm` | RDNA4 FP8 preshuffle GEMM and quantization helpers |

Validation and usage examples live in:

- `tests/kernels/test_preshuffle_gemm.py`
- `tests/kernels/test_fp8_gemm_rowscale.py`
- `tests/kernels/test_mxfp8_gemm_8wave.py`
- `tests/kernels/test_fp4_gemm_4wave.py`
- `tests/kernels/test_gemm_a16w16_gfx950.py`
- `tests/kernels/test_gemm_bf16_gfx1250.py`
- `tests/kernels/test_gemm_fp8fp4_gfx1250.py`
- `tests/kernels/test_rdna_gemm.py` and `tests/kernels/test_rdna3_int8_gemm.py`

Weight and scale preshuffling is part of the interface for several low-precision
kernels. Always use the transformation exercised by that kernel's test; a
tensor with the right shape but the wrong physical ordering can produce valid
memory accesses and incorrect numbers.

## 4. Normalization and softmax

| Module | Main entry points | Scope |
|---|---|---|
| `kernels/norm/layernorm_kernel.py` | `build_layernorm_module`, `build_layernorm_bwd_module`, `build_fused_add_layernorm_module`, quantizing builder variants | LayerNorm forward/backward, fused residual, dynamic/smooth quantization |
| `kernels/norm/rmsnorm_kernel.py` | `build_rmsnorm_module`, `build_fused_add_rmsnorm_module`, `rmsnorm_direct`, quantizing builder variants | RMSNorm forward/fused residual and direct launcher path |
| `kernels/norm/rmsnorm_bwd_kernel.py` | `build_rmsnorm_bwd_module`, `build_rmsnorm_bwd_two_stage_module`, fused-add variants | One- and two-stage RMSNorm backward |
| `kernels/norm/rmsnorm_autotune.py` | `rmsnorm_autotuned` | Opt-in configuration selection for RMSNorm |
| `kernels/norm/softmax_kernel.py` | `build_softmax_module`, `softmax_direct` | Row-wise numerically stable softmax |
| `kernels/norm/softmax_bwd_kernel.py` | `build_softmax_bwd_module` | Softmax backward |
| `kernels/norm/softmax_autotune.py` | `softmax_autotuned` | Validated offline/scratch autotuning for softmax |

Use `tests/kernels/test_layernorm.py`, `test_rmsnorm.py`,
`test_softmax.py`, and `test_softmax_bwd.py` for reference formulas and
tolerances. The autotune-specific tests verify candidate legality, validation,
cache keys, and artifact behavior.

## 5. Attention and sequence kernels

| Module | Main entry points | Scope |
|---|---|---|
| `kernels/attention/flash_attn_interface.py` | `flydsl_flash_attn_func` | Main PyTorch-facing router for dense, variable-length, paged, split-K, BF16, and FP8 paths |
| `kernels/attention/flash_attn_generic.py` | `build_flash_attn_func_module_primary` | Generic FlashAttention builder |
| `kernels/attention/flash_attn_gfx950.py` | `build_flash_attn_dualwave_swp_module` | gfx950 dual-wave/split-K BF16 path |
| `kernels/attention/flash_attn_fp8_gfx950.py` | `build_flash_attn_dualwave_swp_fp8_module` | gfx950 FP8 dual-wave path |
| `kernels/attention/flash_attn_fp8_paged_gfx950.py` | `build_flash_attn_paged_fp8_module` | gfx950 paged FP8 path |
| `kernels/attention/pa_decode_fp8.py` | `pa_decode_ps_launch`, `get_pa_metadata`, `get_recommended_splits` | FP8 paged-attention decode and split selection |
| `kernels/attention/pa_decode_swa.py` | `compile_pa_decode_sw`, `compile_pa_decode_sw_reduce` | Sliding-window paged decode and reduction |
| `kernels/attention/pa_decode_tile.py` | `compile_pa_decode_tile`, `pa_decode_tile` | Tile-based paged decode |
| `kernels/attention/pa_metadata.py` | `get_pa_metadata_v1`, `compile_pa_decode_metadata`, `pa_metadata_reduce` | Device worklist generation and persistent decode metadata |
| `kernels/attention/mla_fwd_decode.py` | `flydsl_mla_fwd_decode` | PyTorch-facing MLA decode selection |
| `kernels/attention/mla_fwd_decode_m16x8_fp8_fp8.py` | `launch_mla_fwd_decode_m16x8_fp8_fp8` | Specialized FP8 MLA implementation |
| `kernels/attention/swa_gfx950.py` | `build_gqa_attn` | gfx950 grouped-query sliding-window attention |
| `kernels/attention/fused_rope_cache_kernel.py` | `build_fused_rope_cache_module` | Fused rotary embedding and cache update |
| `kernels/attention/qk_norm_rope_quant.py` | `flydsl_qk_norm_rope_quant`, `compile_flydsl_qk_norm_rope_quant` | Fused Q/K RMSNorm, GPT-J RoPE, and optional FP8 quantization |

Use `tests/kernels/test_flash_attn_fwd.py`, `test_pa.py`,
`test_mla_decode.py`, `test_swa_gfx950.py`, and
`test_fused_rope_cache.py`. Attention layouts have more axes and routing modes
than a function name can capture, so copying a launch from a different test
shape is unsafe without rechecking all strides and metadata.

## 6. Mixture of Experts

| Module/package | Main entry points | Scope |
|---|---|---|
| `kernels/moe/topk_gating_softmax_kernel.py` | `build_topk_gating_softmax_module` | Fused softmax, top-k selection, and optional renormalization |
| `kernels/moe/moe_sorting_kernel.py` | `moe_sorting_flydsl`, `moe_softmax_sort_flydsl`, `compile_moe_sorting` | Expert grouping, token packing, and fused gating/sorting paths |
| `kernels/moe/moe_gemm_2stage/` | `compile_moe_gemm1`, `compile_moe_gemm2`, `compile_moe_reduction` | FP8/int8/int4 two-stage expert GEMM on CDNA3/CDNA4 |
| `kernels/moe/mxfp_moe/` | `flydsl_mxfp4_gemm1`, `flydsl_mxfp4_gemm2`, compiler helpers | gfx950 fused A4W4/A8W4 stages with device-side FP4 requantization |
| `kernels/moe/moe_2stage_a16wmix/` | `compile_gemm1_a16w4_port`, `compile_gemm2_a16w4_port` | BF16 activation with MXFP4/int4 weight stages |
| `kernels/moe/moe_a8w4_mxscale_gfx1250.py` | `launch_moe_gemm_a8w4` | gfx1250 grouped contiguous-M FP8×MXFP4 expert GEMM |
| `kernels/mega_moe/` | `MegaMoEV2`, `MegaMoEConfig`, `select_mega_moe_config`, `resolve_mega_moe_config` | Fused multi-stage MoE operator with lazy public imports and tuning policy |

The stage-1/stage-2 packages have distinct intermediate tensor, sorting, scale,
and epilogue contracts; they are not drop-in replacements for one another.
Start with `tests/kernels/test_moe_gemm_2stage.py`,
`test_moe_sorting.py`, `test_topk_gating_softmax.py`, and
`test_mega_moe_v2.py`.

## 7. Convolution

`kernels/conv/conv3d_implicit.py` provides `conv3d_implicit` and
`compile_conv3d_implicit` for BF16 implicit-GEMM convolution. It supports NCDHW
and NDHWC layouts, stride, padding, dilation, bias, groups, and split-K.

`kernels/conv/conv3d_implicit_fp8.py` provides the FP8 variant. Its constraints
are intentionally narrower (including the documented channel and layout
requirements). `kernels/conv/conv3d_autotune.py` supplies a bounded manual tile
search.

Use `tests/kernels/test_conv3d_implicit.py` and
`tests/kernels/test_conv3d_implicit_fp8.py` as the interface examples.

## 8. Communication

`kernels/comm/custom_all_reduce.py` exposes `FlyDSLAllreduce`, `init_custom_ar`,
and compatibility helpers around the generated kernels in
`custom_all_reduce_kernel.py`. It requires a multi-GPU ROCm environment and a
correct peer/signal setup; use `tests/kernels/test_allreduce.py` rather than
constructing buffers from prose alone.

`flydsl_dispatch_combine_intranode_op.py` contains the Python integration for
intranode MoE dispatch/combine, with kernel factories in the sibling
`*_kernel.py` module. Its topology, tuning JSON, and process-group requirements
are integration-specific. The profiler test is
`tests/kernels/test_profiler_dispatch_combine.py`.

## 9. Shared implementation modules

The following are building blocks, not standalone operators:

| Path | Responsibility |
|---|---|
| `kernels/common/tensor_shim.py` | Common compile/run and dtype/layout shims |
| `kernels/common/buffer_ops.py` | Legacy/raw buffer helpers retained for existing kernels |
| `kernels/common/mem_ops.py` | Pointer and memory helpers used by newer kernels |
| `kernels/common/dpp_utils.py` | DPP reductions and lane operations |
| `kernels/common/gfx1250_cluster.py` | Cluster/multicast geometry helpers |
| `kernels/common/mma/mfma_preshuffle_pipeline.py` | Shared preshuffle layout, epilogue, and XCD scheduling components |
| `kernels/gemm/fp8_gemm_utils.py` | CDNA4 FP8 data-movement and store helpers |
| `kernels/gemm/gemm_common_gfx1250.py` | gfx1250 TDM/WMMA pipeline helpers |

Prefer stable `flydsl.expr` operations in new kernels. Reuse a common kernel
helper only when its architecture, tensor layout, and ownership model match the
new operator.

## 10. Choosing and validating an implementation

1. Select the domain and target architecture.
2. Open the matching test and identify the wrapper/builder actually exercised.
3. Preserve the test's dtype, physical layout, scale format, workspace, and
   stream conventions.
4. Run correctness on representative edge shapes before benchmarking.
5. Run the dedicated benchmark or `scripts/run_benchmark.sh` only after
   correctness passes.
6. Record the FlyDSL revision, ROCm version, GPU architecture, and exact shape
   with performance results.

See [Testing and benchmarking](testing_benchmarking_guide.md) for runners,
markers, output, and profiling workflow.
