# TP MegaMoE entry design: transferable lessons and limits

Audited 2026-11-06 at AITER main
`4ecfecb2dcc014a471f7d5c376d1fdfeb58cbe00`. This is source analysis, not a
new TP GPU benchmark or a proof that EP can share its communication contract.

## Entry and configuration

Public source is `aiter/ops/flydsl/mega_moe_tp.py`:

- MegaMoeTPConfig: immutable public configuration, including capacity.
- MegaMoeTP.forward: one normal entry selects small engine or big engine.
- prepare: constructs resources and warms reachable launch variants before use.
- set_weights: updates shared/lazy engines without reconstructing communication.
- launch_config and bigm_plan: report resolved tuning rather than hiding it.

Kernel implementation lives under
`aiter/ops/flydsl/kernels/mega_moe_tp/`. Tuning sources are
`aiter/configs/mega_moe_tp_tuned.csv` and `mega_moe_tp_bigm_tuned.csv`.
Explicit overrides take precedence over CSV/defaults; print the actual choice.

There are three execution regimes, not simply two kernels:

1. Small-engine dynamic scheduling, normally A4 global tokens<256.
2. Small-engine load-balanced scheduling, normally global tokens>=256 while
   within that engine's capacity.
3. A separate SDMA chunked big-M engine beyond the small capacity/threshold.

These are defaults at this snapshot, not universal thresholds. Small capacity
depends on geometry, LDS, topology and table constraints. A8 small-engine
support is narrower and does not inherit A4 load-balanced/big-M support.

bigm_min_tokens semantics at the public config:

- None: automatic, entering big-M beyond the usable small-engine capacity
  when the requested profile is eligible.
- 0: disabled.
- Positive: an explicit threshold, converted to the engine's local-token
  convention and validated against its minimum granularity.

In ag_rs mode global token work is local rows times TP. In ar mode inputs are
replicated and the wrapper converts to a local shard count. Never compare an
EP local token count to a TP global token threshold without this conversion.

## Why its automatic selector cannot be copied verbatim to EP

The TP path relies on its collective contract: matching local row counts and
launch order, and replicated inputs/routes where ar requires them. Thus a
selector based on local shape can still agree across the TP group. EP supports
unequal local row counts and empty ranks; local shape can select incompatible
protocols there. EP needs the common runtime row bound or equivalent collective
metadata described in the EP fusion reference.

## Reuse only the useful parts

- One forward with a deterministic, inspectable selector.
- Immutable construction capacity/config, separate from per-call metadata.
- Explicit resource preparation and artifact/module loading before serving.
- Shared weight binding and a unified error/reset/close lifecycle.
- One canonical tuning representation consumed by diagnostics and execution.

Do not import these limitations into a production EP refactor:

- Inspected SDMA big-M is **not CUDA-Graph-capturable**. prepare does not make
  host parity updates/copy launch behavior graph-safe.
- Big-M eligibility is currently FP4 activation and specific activation/
  geometry/communication contracts, not generic EP A8W4 support.
- Small SymmetricArena and big SdmaComm allocate separately. One entry is not
  evidence of a unified workspace.
- Precompile/private caching is not proof of production EP offline AOT;
  require the separate fresh-cache/run-only gates.

AITER_BIGM_* strings inside big compute are also used as keys in opts passed
from tuning dictionaries. Inspect the call chain before calling all such keys
process environment variables. TP still has real diagnostic/config-path envs;
do not claim it is entirely env-free.

## EP/TP performance and overlap reasoning

At equal global token semantics and divisible expert placement, EP stores
E/P complete experts per rank while TP stores E experts with intermediate I/P.
Their useful weight and arithmetic volumes can be comparable. TP is not
automatically eight times less work merely because each expert is sharded.
Padding, redundant input work, routing skew, communication and kernel geometry
can still differ materially.

For logical W1 [H,2I], intermediate I is the GEMM1 output/N dimension. For W2
[I,H], it is the input/K dimension. EP can tune tile-M/N/K locally without
changing model sharding; introducing a new distributed shard is an algorithm
change. An EP smaller tile-M is not mathematically the same as TP sharding I.

TP routes may already be sorted before its compute/collective kernel. Count
sorting, quant/scale sorting, AG/RS, pack/unpack and launch gaps in the same
acceptance span. Prove overlap from ATT roles/timelines rather than "one
kernel". Wave specialization is not itself a speedup: producer residency,
HBM/XGMI supply, LDS rings, MFMA gaps and synchronization determine the result.
Rejected older EP WS experiments are not the merged #6881 implementation and
must not be revived or used as baselines without an explicit task.
