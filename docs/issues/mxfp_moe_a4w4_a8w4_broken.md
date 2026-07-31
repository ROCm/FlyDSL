<!--
GitHub issue DRAFT for ROCm/FlyDSL. The `gh` CLI is not available in this
environment, so this file is a ready-to-paste draft for a human to file. After
filing, replace `TODO(issue #NNN)` in tests/kernels/test_moe_gemm.py with the
real issue number.
-->

# Fused `mxfp_moe` a4w4 / a8w4 MoE kernels are numerically broken and memory-unsafe

## Summary

The fused MX-FP4 MoE pipeline in `kernels/moe/mxfp_moe/` produces numerically
wrong results for both the **a4w4** (MX-FP4 activation) and **a8w4** (MX-FP8
activation) paths. End-to-end cosine similarity against the torch reference is
~0.12 (a4w4) and ~0.07 (a8w4), i.e. essentially uncorrelated output, not a
quantization-precision effect. The `a16w4` (bf16 activation x MX-FP4 weight)
path, which uses *separate* kernels, is faithful (cos ~0.9999) and serves as the
control that proves the reference/quant/shuffle/verify machinery is correct.

The broken kernels are additionally **memory-unsafe**: launching them and then
unwinding under pytest teardown can raise `hipErrorIllegalAddress`, corrupting
the HIP module state and cascading a crash into unrelated tests in the same
session.

This is a pre-existing defect (not a regression introduced by the a16w4 work).
aiter's a4w4/a8w4 reach cos ~0.995 on the same math, so a correct implementation
exists — this is a real bug in the FlyDSL kernels, not a precision ceiling.

## Strict-cosine evidence (fresh capture)

Self-validating harness: the `a16w4` control clearing ~0.9999 proves the
reference / quantization / weight-shuffle / verify path is byte-identical to the
committed test and trustworthy; only then are the a4w4/a8w4 numbers meaningful.

Shape: `tokens=128, model_dim=1024, inter_dim=256, experts=8, topk=2, tile_m=32`
(seed=0, cold cache).

| Path            | Kernels                          | e2e cosine | max_abs_err | verdict |
|-----------------|----------------------------------|-----------:|------------:|---------|
| a16w4 (control) | `flydsl_a16w4_gemm1` / `_gemm2`  |   0.999997 |      0.0032 | FAITHFUL |
| a4w4  (fp4)     | `flydsl_mxfp4_gemm1` / `_gemm2`  |   0.123926 |      1.3164 | BROKEN  |
| a8w4  (fp8)     | `flydsl_mxfp4_gemm1` / `_gemm2`  |   0.070880 |      1.1406 | BROKEN  |

(a8w4 cosine varies at the ~1e-3 level run-to-run due to the atomic-scatter
epilog; it is consistently ~0.07.)

## Localization

The two broken paths share stage 2 and differ only in the stage-1 activation
dtype, which cleanly isolates the defects:

- **Shared broken down-proj (`flydsl_mxfp4_gemm2`).** a4w4 and a8w4 both route
  through `flydsl_mxfp4_gemm2` (down-proj) and both fail, while a16w4 uses its
  own `flydsl_a16w4_gemm2` and passes. The shared mxfp4 gemm2 down-proj is
  broken. See `kernels/moe/mxfp_moe/gemm2.py`. (Prior stage-isolation runs
  measured the a4w4 stage-1 fp4 intermediate at cos 1.000 while the isolated
  gemm2 down-proj was ~0.124, consistent with this.)

- **Additional broken fp8 gemm1 A-path.** a8w4 fails *worse* than a4w4 even
  though they share the same broken gemm2 and differ only in the stage-1 A
  dtype. The MX-FP8 activation load path in gemm1 is *additionally* broken.
  See `kernels/moe/mxfp_moe/gemm1.py:237-243,256-270` (the `a_dtype == "fp8"`
  128-K operand load / DS-read halves). Prior stage-isolation measured the fp8
  gemm1 stage-1 output at cos ~0.16.

## Memory-safety note

Beyond wrong numbers, the a4w4/a8w4 kernels are memory-unsafe. Launching them
and unwinding under pytest teardown corrupts the JIT/HIP module state and
cascades a `hipErrorIllegalAddress` into subsequent, unrelated tests in the same
process. For this reason the tests xfail with `run=False` (documented failure,
kernel not executed) rather than `run=True`.

## Root cause of the previously-masked pass (now fixed)

These defects were previously hidden by a too-loose correctness gate:

- `verify_output(...)`'s return value was discarded (not `assert`ed), and
- it was called with a loose `rtol/atol=0.5` plus `logits_diff_threshold=1`, and
  `verify_output` early-returns `True` when fewer than ~5% of elements exceed the
  loose allclose tolerance — so cos ~0.1 output passed silently.

This is now fixed on the verification branch: the e2e gate is a strict
`assert verify_output(out, ref2, rtol=2e-3, atol=2e-3, logits_diff_threshold=2e-3)`,
and the a4w4/a8w4 callers are `xfail(run=False)` so CI reports the expected
failure honestly (and stays stable, given the memory-safety cascade) while the
faithful a16w4 gate keeps running. See `tests/kernels/test_moe_gemm.py`.

## Reproducer

Committed at `tools/repro/repro_mxfp_moe_a4w4_a8w4_broken.py`. Deterministic
(seed=0), cold-cache, self-validating (fails if the a16w4 control does not clear
the fidelity floor). It reuses the exact committed e2e code path and prints the
cosine table + PASS/FAIL verdict.

```bash
cd <repo> && source .verify_runenv.sh && \
  HIP_VISIBLE_DEVICES=<gpu> FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  python3 tools/repro/repro_mxfp_moe_a4w4_a8w4_broken.py
```

Exit 0 = reproduced (control faithful, a4w4/a8w4 broken); exit 1 = did not
reproduce (control failed, or the kernel has been fixed — update the xfails and
this repro if so).

## Environment

- Arch: `gfx950` (MI350/MI355X, CDNA4)
- ROCm: 7.2.4
- Branch: `worktree-mxfp-moe-a16w4-verify`, commit `82a61ab2`
  (`[test] Fix masked mxfp_moe/a16w4 correctness gates (assert + strict cos);
  xfail broken a4w4/a8w4`)

## Suggested fix scope (out of scope for this issue)

Fixing the kernels is *not* part of this tracking issue. The two independent
defects to address are (1) the shared `flydsl_mxfp4_gemm2` down-proj and (2) the
fp8 gemm1 A-path load; and separately the illegal-address / memory-safety bug so
the kernel can be run under CI without crashing the session.
