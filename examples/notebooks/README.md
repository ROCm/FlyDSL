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

## API cheat-sheet

The whole `flydsl.expr` foundation these notebooks cover, in one place — enough to
write a kernel without reading the source. Layout ops (`make_layout`,
`logical_divide`, tiled copy, MMA) are deliberately out of scope; they get their
own series.

```python
# Kernel + launch (00)
@flyc.kernel                       # device kernel; the body is traced to MLIR
@flyc.jit                          # host launch wrapper
kernel(args).launch(grid=(gx, 1, 1), block=[bx, 1, 1], stream=stream)
flyc.from_dlpack(t)                # torch tensor -> fx.Tensor view (jit also accepts a raw torch tensor)
    .mark_layout_dynamic(leading_dim=0, divisibility=4)   # dim sized at runtime, n-byte aligned for vectorization

# Scalars (01) — construct at top level; arithmetic and casts run only inside a trace
fx.Int32(7)   fx.Float32(2.0)   fx.Boolean(True)
v.to(fx.Float16)                   # cast (.width works at top level; ops need an active trace)
fx.Constexpr[int]                  # trace-time Python value; folds into the kernel and the JIT cache key

# Structs (02)
@fx.struct                         # frozen aggregate value type; v.replace(field=...) returns a copy
from flydsl.compiler.protocol import dsl_size_of, dsl_align_of   # host-side; NOT attributes of fx

# Copy atoms + register tensors (03)
atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)      # target-agnostic
fx.copy_atom_call(atom, src, dst)                                # copy src -> dst
rt = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)       # per-thread register tensor
fx.memref_load_vec(rt) / fx.memref_store_vec(val, rt)            # read / write a register tensor
fx.arith.addf(a, b)                # explicit op on loaded values (`+` is for fx scalar values, not tensors)
```

Three gotchas worth front-loading:

- `fx.printf` takes only bare `{}` (no `{:.2f}`); a literal `%` is consumed by the
  device printf (write `"mod"`); a true `Boolean` prints as `-1`.
- Device `printf` is not captured by Jupyter — wrap the launch in
  `with wurlitzer.pipes() as (out, _): launch(...); torch.cuda.synchronize()`, then `print(out.read())`.
- `Constexpr` `fp8`/`bf16` math is not rounded until the value is materialized as its
  MLIR type; only `f16`/`f32`/`f64` fold at trace time.

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
