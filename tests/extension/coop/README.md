# Coop test coverage and runtime

Follow [PR #1207](https://github.com/ROCm/FlyDSL/pull/1207): cover each parameter
value without compiling its full Cartesian product in every presubmit job.

- Use `matrix_cases` for independent axes. The default diagonal visits every
  value of every axis.
- Use `matrix_cases(..., independent=True)` when combining expensive values
  makes compilation much slower. This varies one axis at a time around the
  first values, so put inexpensive baseline values first. All dtypes and widths
  remain covered, including 1024-thread blocks and nine-item per-thread tiles.
- Keep cross-target checks in `test_target_compilation.py`. Each helper's matrix
  covers every primitive/policy and every architecture. Splitting the primitive
  axis across files silently restores the expensive primitive/target product.
- Shape checks sample public and universal namespaces rather than executing
  both in every kernel. These implementations are distinct; retain both axes.
- Radix shape checks use a short digit pass. Dedicated regressions cover multiple
  passes, unsigned digits, nine-bit capacity, callbacks and high-bit digits.

Default dtype checks use small blocks for nine-item merge tiles. Floating-point
and nested-record semantic checks also use small tiles, while the compact
multi-warp record regression remains in the default suite. Cross-target
sorting checks still exercise multiple items. The merge/topk LDS regression
always checks and allocates storage for the original 1024 x 9 tile; its default
GPU operation uses one item per thread. Exhaustive mode also executes the original larger stress inputs.

`FLYDSL_COOP_TESTS_FULL=1` restores Cartesian products and both CPU/CUDA defaults
in the shared device fixture. It is independent of `RUN_TESTS_FULL`, which only
controls the `large_shape` exclusion in `scripts/run_tests.sh`.

```bash
# Run all coop tests with representative parameter combinations.
python -m pytest tests/extension/coop

# Inspect the exhaustive matrix without compiling kernels.
FLYDSL_COOP_TESTS_FULL=1 python -m pytest tests/extension/coop --collect-only -q
```

Keep focused regressions for partial tiles, noncommutative operators,
record/vector elements, floating-point bit patterns, storage reuse and grid
publication. Keep any bug-specific parameter interaction explicitly instead of
relying on sampling to select it. Allocate device inputs and host references
with explicit devices; validate changed GPU checks under both PyTorch defaults.
