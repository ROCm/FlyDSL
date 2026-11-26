# Serving integration, cross-layer workspace and MORI comparison

## Operator ownership versus layer ownership

The inspected historical ATOM adapter
`atom/model_ops/fused_moe/flydsl_mega_experts.py` caches a complete MegaMoEV2
engine by execution signature and rebinds every layer's weight pointers before
each launch. Per-layer wrappers and prepared weights still exist; the heavy
runtime owner is shared. Verify the current adapter before relying on this.

Cache identity must include every runtime ABI/compiled semantic: EP group/rank,
device, dimensions, expert placement, top-k, quant/scale format, capacity,
activation/clamp and relevant protocol. An omitted swiglu_limit can silently
reuse another layer's compiled activation. Concurrent lanes need separate
state or an explicit supported in-flight index; a process cache alone is not
multi-stream safety.

The customer fork is not necessarily called by ATOM. Its dated optional
constructor API is shared_instance=None (recorded in Byte commit
`d16f86b575d2ddac0641849e637a5a54e3a13ead`), not merely shared_comb_op.
Existing callers that omit it retain the normal construction behavior.
Confirm this API exists on the actual branch before giving a runnable example.

Safe reuse shares the communication buffers **and** their full mutable runtime:
epochs/parity, launch admission, dispatch workspace and compute scratch.
Sharing just comb_op while independently advancing two epoch owners can pass
synchronized calls and hang an owner->child->owner captured chain.

Layer weights remain separate. Bind all W1/W1-scale/W2/W2-scale pointers;
SmoothQuant/LQQ also needs its own scales/qparameters. Test with independently
different values for each pointer, not merely different addresses containing
identical weights. Keep referenced tensors alive for every captured graph.

Symmetric allocation is collective: all ranks must construct the same owner/
shared topology in the same order. Validate compatibility before allocation.
Do not introduce an implicit global cache inside the operator as a shortcut.

## Scratch, output and lifetime

Scratch is reusable transient storage, not a new per-layer model weight. It
normally remains allocated between calls; each kernel overwrites or resets the
parts required by its protocol. Blind memset after each layer neither frees
the allocation nor proves epoch correctness, and can add synchronization/work.

Output may alias a shared combine arena. A Python reference does not preserve
its value after the next shared call. Consume it or clone on the ordered
stream before reuse if the framework needs to retain it. Do not silently add
a per-forward allocation/copy to conceal a lifetime contract change.

Before claiming cross-layer reuse is safe, test changing weights, all saved
outputs, dynamic routing and token buckets, zero ranks, all layer slots,
queued Graph chains and epoch wrap. The October customer audit found
geometry-dependent admission and non-parity completion-epoch defects that
were absent from static-shape proxies. Those follow-up fixes were still
uncommitted/unpushed and final full acceptance incomplete in the saved report;
do not re-label the historical shared_instance API as proof they were fixed.

Local evidence when available:
`/data/ghu/m13_megamoe_shared_runtime_20260911.md`, including its newer
2026-10-27 status, and `/data/ghu/megamoe_reuse_20261027/`.

## Match MORI EP's full business pipeline

The existing AITER bench is
`op_tests/multigpu_tests/bench_mega_moe_v2.py`; its presence does not prove it
matches the latest framework. Re-check:

```text
BF16 input
 -> optional native MX input quant, included in measured E2E
 -> MORI dispatch with the actual payload dtype and scale layout
 -> receive trim according to the real per-group token contract
 -> fused_moe sorting/scale sort/GEMM1/activation/quant/GEMM2/local reduce
 -> MORI combine
 -> source-row output slice
```

Do not compare a prequantized MORI path to Mega's BF16 entry or assume all
MORI handles transport BF16. In an inspected ATOM FP8-activation configuration,
dispatch carried MXFP8/E8M0, scale_dim=H/32 and scale_type_size=1. Other modes
legitimately transport BF16 and quantize later: report them separately.

MORI destination-rank token dedup can make receive trimming use the sum of
source-rank token counts, not localTPR*EP and not that count times top-k.
Use the actual handle/framework contract and preserve zero/uneven ranks.

Log resolved dispatch/combine blocks and warps, fused_moe's full lookup tuple,
matched CSV and all GPU kernel names. EPR, runtime top-k, padded-M, activation,
quant and phase affect selection. Do not infer precision solely from a legacy
substring in a generated kernel name or manually choose a favorable config.

The dated 2026-11-04 call-chain check used prefill128x16, decode<=48 rows128x8,
and larger decode256x16 on256CU. This is historical configuration evidence,
not a permanent rule: locate the current ATOM selector before benchmarking.

MORI GEMM2 may use BF16 atomic accumulation, whose rounding order can vary.
Require declared reference accuracy and report replay drift; do not impose
bitwise determinism on that algorithm or use its exception to relax a
deterministic Mega refactor's own before/after requirement.

## Serving metadata and reproducible environments

P/D can mean prefill and decode phases within one server; it does not imply
two independently launched servers. Derive MTPR, EP group, Graph physical rows,
speculation/MTP and actual per-rank padding from framework source and logs.
Graph batch size equals local rows only when the adapter proves that relation.

For whole-model regressions freeze image digest, ATOM/SGLang, FlyDSL, MORI,
weights and command; change only AITER when that is the A/B under test. Check
Graph overlap, unexpected padding, module loads, memory/KV headroom and EPLB,
not just isolated kernel speed. Treat fixed-length and AgentX as separate
request distributions and preserve their input/output range and request rate.

Reproduction runbooks must contain direct server and benchmark commands with
all flags, versions and result provenance, not private run_case.py/run8.sh
dependencies. For a portable image, bake the correct libraries and source
installation inside it, validate the exact image locally on authorized GPUs,
then export in the requested Docker/OCI format. Merely committing an image
that depends on external code mounts is not a self-contained reproduction.
Do not claim docker-load compatibility or throughput from an untested archive.
