# Kimi-K3 MonoKernel

The Opt254-based implementation runs a complete KDA MoE layer in one GPU
launch on TP8/gfx950: both AttnRes mixers, KDA projection and recurrence,
routing, latent/shared projections, MXFP4 experts, TP reductions, and the
residual update. Layer 0's dense FFN is out of scope.

The supported shape range is batch 1–8 and sequence length 1–8, with
`samples = batch * seq_len`. Pass `seq_len` explicitly and set `mtp=True`
for sequences longer than one token. `KimiK3CompileConfig(path="auto")`
chooses the validated specialization at compile time. Non-default tuning
configurations require their own correctness and residency validation.

Decode uses contiguous int32 `state_indices` with shape `[batch]`. MTP uses
shape `[batch * (seq_len + 1)]`, grouped by independent state chain: token
`t` in chain `b` reads slot `b * (seq_len + 1) + t` and writes the next slot.
Each slot indexes the convolution and recurrent snapshot pools. The first
entry preserves the incoming snapshot. `--batch 1 --seq 8` (equivalently,
`--mtp --samples 8`) covers seven speculative tokens plus the current token.
The complete kernel supports up to 64 tokens (`--batch 8 --seq 8`).

Run from the repository root in a configured eight-GPU FlyDSL environment,
after checking complete-grid residency for the compiled binary:

```bash
python -m kernels.kimi_k3_monokernel.tools.monokernel \
  --batch 1 --seq 8 --layer-idx 1 --check --full-replay-check \
  --bench --layers 16 --repeats 50
```

The strict replay oracle is included in the kernel tools package. `--check`
alone also reports the generic reference diagnostics. `--staged` selects
the multi-launch comparison path, whose fused tail retains its 32-token limit;
`--attention-only` isolates KDA.
`--dump-ir-dir DIR` writes separate compiler dumps for each TP rank.

B1/S8 measures 259.58 μs for the complete TP8 layer 1 (median of three
seed medians, 16-layer graph, 50 repeats). All 32 new shapes passed strict
replay with three seeds. The retained measurements of 71.83 μs for B1/S1
and 114.20 μs for B1/S4 are historical; their compiled binaries are unchanged. See [selection, validation, and reproduction details](../../docs/kimi_one_kernel/README.md).
