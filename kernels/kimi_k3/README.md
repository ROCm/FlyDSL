# Kimi-K3 fused kernels

This package owns model-specific Kimi-K3 kernels. The KDA full-layer path is a
single GPU launch covering both AttnRes mixers, KDA projection/recurrence,
router and top-k selection, latent/shared projections, MXFP4 experts, TP8
reductions, and the residual update. Layer 0's dense FFN is intentionally out
of scope.

Run correctness checks and graph-replay benchmarks from the repository root:

```bash
export ROCM_PATH=/tmp/flydsl-rocm-sdk-kimi
export PYTHONPATH="$PWD:/root/FlyDSL/build-fly/python_packages"

python -m kernels.kimi_k3.tools.full_layer --samples 4 --layer-idx 1 --check
python -m kernels.kimi_k3.tools.full_layer \
  --samples 4 --layer-idx 1 --bench --layers 16 --repeats 30
```

Use `--profile` for the instrumented single-launch stage timeline and
`--attention-only` to isolate KDA. `--dump-ir-dir DIR` emits one compiler dump
directory per TP rank for resource inspection without cross-rank file races.
