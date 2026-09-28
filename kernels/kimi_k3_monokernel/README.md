# Kimi-K3 MonoKernel

This package owns the model-specific Kimi-K3 MonoKernel. The complete KDA
path is one GPU launch covering both AttnRes mixers, KDA
projection/recurrence, router and top-k selection, latent/shared projections,
MXFP4 experts, TP8 reductions, and the residual update. The benchmark also
keeps the faster staged path so fusion work cannot hide a performance
regression. Layer 0's dense FFN is intentionally out of scope.

Run correctness checks and graph-replay benchmarks from a configured FlyDSL
environment at the repository root:

```bash
python -m kernels.kimi_k3_monokernel.tools.monokernel --samples 4 --layer-idx 1 --check
python -m kernels.kimi_k3_monokernel.tools.monokernel \
  --samples 4 --layer-idx 1 --bench --layers 16 --repeats 30
python -m kernels.kimi_k3_monokernel.tools.monokernel \
  --staged --samples 4 --layer-idx 1 --bench --layers 16 --repeats 30
python -m kernels.kimi_k3_monokernel.tools.monokernel \
  --mtp --samples 8 --layer-idx 1 --check --bench --layers 16 --repeats 30
```

Pass `mtp=True` to `KimiK3MonoKernel` (or `--mtp` to the benchmark tool) to
interpret the `S` rows as one ordered speculative-token group. In this mode,
`state_indices` must be contiguous `int32` with shape `[S + 1]`: token `s`
reads the convolution and recurrent snapshots at `state_indices[s]` and writes
its complete post-token snapshots to `state_indices[s + 1]`. The extra entry
retains the incoming snapshot instead of overwriting it. Without `mtp=True`,
the existing `[S]` independent-request decode ABI is unchanged.

On TP8, the retained performance path uses seven launches per layer at S=4.
At S=8 it uses nine launches because the three-stage KDA attention path is
faster than its one-launch attention specialization. The complete MonoKernel
path always uses one application launch.

True MTP is also part of that single application launch. Its KDA schedule
uses a causal convolution publication chain followed by two disjoint 64-row
recurrent-state chains, with device-scope state traffic between dependent
CTAs.

Use `--profile` for timing, `--attention-only` to isolate KDA, and `--staged`
for the fastest retained multi-launch path. `--dump-ir-dir DIR` emits one
compiler dump directory per TP rank for resource inspection without cross-rank
file races.
