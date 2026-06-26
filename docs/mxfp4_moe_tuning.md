# MXFP4 MoE 2-Stage Tuning Harness (gfx950)

Measurement + verification infrastructure for tuning the MXFP4 (per-1×32
microscale fp4) MoE 2-stage GEMM pipeline on AMD gfx950 / MI350X, in support of
[ROCm/FlyDSL#708](https://github.com/ROCm/FlyDSL/issues/708) ("MXFP4 MoE low MFU
at large shapes and long latency at small tokens").

> **Status: tuning *infrastructure* + a validated baseline. This does NOT yet
> contain a performance change to any kernel** — it is the measurement, legality,
> and bookkeeping foundation that a tuning campaign runs on top of. No production
> kernel logic is modified by this change set.

## Components

- **`kernels/moe_tuning.py`** — pre-compile legality filter for stage1/stage2 tile
  configs (LDS footprint, thread/divisibility constraints, MX-FP4 floors). Lets a
  config search reject illegal tiles before spending GPU time; mirrors the
  builders' real LDS sizing (stage1 vs stage2 fp4 asymmetry included).
- **`kernels/moe_tuning_spec.py`** — locked spec constants + win/no-regression
  predicates (win margins, regime-aware no-regression band, token grid, MFU
  denominator, metric formula).
- **`scripts/moe_tuning_harness.py`** — the measurement harness: full provenance
  per point (GPU id+model, branch+commit, exact replayable command, warmup/iters,
  idle-GPU check, verified clock pinning), median+p95 from a faithful timed loop,
  and a fail-closed candidate sweep CLI (illegal/unmeasured configs are recorded
  as machine-readable rejections, never silently skipped).
- **`scripts/moe_tuning_ledger.py`** — attempt ledger + full-coverage Pareto
  comparator. A candidate is promotable only via a single `claimable_win` gate
  (full coverage + no kernel-path/e2e regression + a real win + a strict
  AOT/correctness hard gate). Includes ledger-integrity scans (duplicate /
  replayable-command / supersede-link).
- **`scripts/aiter_strict_point.py`** — strict, AOT-checked, model-correct single
  -case aiter fused-MoE e2e + correctness guardrail (`logits_diff <= 0.01`).
- **`scripts/sync_aiter_flydsl_kernels.sh`** — overlay the current FlyDSL MoE
  kernels onto aiter's vendored copies so the e2e guardrail runs against the same
  sources being tuned.
- **`docs/baseline_523ca1c7_validated.csv`** — a validated locked a4w4 baseline
  table (reference every candidate is compared against).

## Tests

`tests/unit/test_moe_tuning_harness.py` and
`tests/unit/test_moe_tuning_legality.py` cover the legality filter, provenance
contracts, the Pareto comparator + win gate, and the integrity scans (host-side,
no GPU required):

```bash
python3 -m pytest tests/unit/test_moe_tuning_harness.py \
                  tests/unit/test_moe_tuning_legality.py -q
```

## Scope notes

- This change set targets the a4w4 (fp4×fp4) path. a8w4 (fp8×fp4) correctness is
  currently environment-blocked by an aiter non-fp4-activation wrapper/layout
  contract mismatch (not a FlyDSL kernel bug); it is quarantined for win claims.
- The actual tile/lever tuning that produces MFU/latency wins runs on top of this
  harness and is tracked separately against #708.
