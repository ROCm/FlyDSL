<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2025 FlyDSL Project Contributors -->

# FlyDSL onboarding notebooks

An interactive, bottom-up introduction to the `flydsl.expr` foundation. Work through
them in order — each builds on the last, and the series stops short of layout algebra
(`make_layout`, `logical_divide`, tiled copy, MMA), which gets its own follow-up series.

| # | Notebook | Topic |
|---|----------|-------|
| 00 | [`00_hello_flydsl.ipynb`](00_hello_flydsl.ipynb) | the `@flyc.kernel` / `@flyc.jit` model; reading dumped IR |
| 01 | [`01_numeric_types.ipynb`](01_numeric_types.ipynb) | scalar types: ints, floats, `bf16`/`fp8`, casts, `Constexpr` |
| 02 | [`02_struct.ipynb`](02_struct.ipynb) | `@fx.struct` aggregate value types and their memory layout |
| 03 | [`03_universal_ops.ipynb`](03_universal_ops.ipynb) | target-agnostic `Universal*` atoms + a vector-add capstone |

## Running

These notebooks execute kernels, so they need a built/installed FlyDSL and a ROCm GPU,
plus a couple of notebook tools:

```bash
pip install jupyter wurlitzer
```

`wurlitzer` lets the notebooks show GPU `printf` output inline — Jupyter does not
capture device stdout on its own. Then open them with Jupyter, or run headless:

```bash
jupyter nbconvert --to notebook --execute --inplace examples/notebooks/*.ipynb
```

Cell outputs are committed **cleared**; run the cells to populate them.
