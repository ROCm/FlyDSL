# Kimi-K3 MonoKernel

The selected Opt254 implementation runs a complete KDA MoE layer in one GPU
launch on TP8/gfx950: both AttnRes mixers, KDA projection and recurrence,
routing, latent/shared projections, MXFP4 experts, TP reductions, and the
residual update. Layer 0's dense FFN is out of scope.

The supported shape range is batch 1–8 and sequence length 1–4, with
`samples = batch * seq_len`. Pass `seq_len` explicitly and set `mtp=True`
for sequences longer than one token. `KimiK3CompileConfig(path="auto")`
chooses the validated specialization at compile time. Non-default tuning
configurations require their own correctness and residency validation.

Decode uses contiguous int32 `state_indices` with shape `[batch]`. MTP uses
shape `[batch * (seq_len + 1)]`, grouped by independent state chain: token
`t` in chain `b` reads slot `b * (seq_len + 1) + t` and writes the next slot.
Each slot indexes the convolution and recurrent snapshot pools. The first
entry preserves the incoming snapshot. `--mtp --samples 8` alone would mean
an unsupported sequence of eight tokens; use `--batch 2 --seq 4` instead.

Run from the repository root in a configured eight-GPU FlyDSL environment,
after checking complete-grid residency for the compiled binary:

```bash
PYTHONPATH="$PWD/experiments/kimi_one_kernel/opt254${PYTHONPATH:+:$PYTHONPATH}" \
python -m kernels.kimi_k3_monokernel.tools.monokernel \
  --batch 1 --seq 4 --layer-idx 1 --check --full-replay-check \
  --bench --layers 16 --repeats 50
```

The PYTHONPATH entry selects the archived strict replay oracle. `--check`
alone also reports the generic reference diagnostics. `--staged` selects
the multi-launch comparison path; `--attention-only` isolates KDA.
`--dump-ir-dir DIR` writes separate compiler dumps for each TP rank.

The retained full-layer measurements are 71.83 μs for B1/S1 and 114.20 μs
for B1/S4. These are historical validated measurements, not a new benchmark
of this checkout. See [selection, validation, and reproduction details](../../docs/kimi_one_kernel/README.md).
