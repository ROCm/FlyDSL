# Kimi-K3 fused kernels

This package owns model-specific Kimi-K3 kernels. The experimental KDA
full-layer path is one GPU launch covering both AttnRes mixers, KDA
projection/recurrence, router and top-k selection, latent/shared projections,
MXFP4 experts, TP8 reductions, and the residual update. The benchmark also
keeps the faster staged path so fusion work cannot hide a performance
regression. Layer 0's dense FFN is intentionally out of scope.

Run correctness checks and graph-replay benchmarks from the repository root:

```bash
export ROCM_PATH=/tmp/flydsl-rocm-sdk-kimi
export PYTHONPATH="$PWD:/root/FlyDSL/build-fly/python_packages"

python -m kernels.kimi_k3.tools.full_layer --samples 4 --layer-idx 1 --check
python -m kernels.kimi_k3.tools.full_layer \
  --samples 4 --layer-idx 1 --bench --layers 16 --repeats 30
python -m kernels.kimi_k3.tools.full_layer \
  --staged --samples 4 --layer-idx 1 --bench --layers 16 --repeats 30
```

On TP8, the retained performance path uses seven launches per layer at S=4.
At S=8 it uses nine launches because the three-stage KDA attention path is
faster than its one-launch attention specialization. The experimental complete
layer path always uses one application launch.

Use `--profile` for timing, `--attention-only` to isolate KDA, and `--staged`
for the fastest retained multi-launch path. `--dump-ir-dir DIR` emits one
compiler dump directory per TP rank for resource inspection without cross-rank
file races.
