<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2025 FlyDSL Project Contributors -->

# FlyDSL onboarding notebooks

An interactive, bottom-up introduction to FlyDSL. Notebooks **00–03** cover the
`flydsl.expr` foundation; **04–05** cover the **layout algebra** (`make_layout`,
`logical_divide`, tiled copy, swizzle). Work through them in order — each builds on the
last.

| # | Notebook | Topic |
|---|----------|-------|
| 00 | [`00_hello_flydsl.ipynb`](00_hello_flydsl.ipynb) | the `@flyc.kernel` / `@flyc.jit` model; reading dumped IR |
| 01 | [`01_numeric_types.ipynb`](01_numeric_types.ipynb) | scalar types: ints, floats, `bf16`/`fp8`, casts, `Constexpr` |
| 02 | [`02_struct.ipynb`](02_struct.ipynb) | `@fx.struct` aggregate value types and their memory layout |
| 03 | [`03_universal_ops.ipynb`](03_universal_ops.ipynb) | target-agnostic `Universal*` atoms + a vector-add capstone |
| 04 | [`04_layout.ipynb`](04_layout.ipynb) | layout algebra: shape/stride, `crd2idx`, `logical_divide`, memref vs coord tensors |
| 05 | [`05_tiled_copy_and_swizzle.ipynb`](05_tiled_copy_and_swizzle.ipynb) | thread-value layouts, partitioning, the LDS swizzle (AMD CDNA), drawing layouts with `print_typst` |

## API cheat-sheet

The whole API these notebooks cover, in one place — enough to write a kernel without
reading the source. The MMA atoms (`make_mma_atom`, `make_tiled_mma`, `gemm`) are the
one piece left for later; `examples/03-tiledMma.py` is the worked reference.

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

# Layout algebra (04) — runs inside a trace; nothing launches on the GPU
L = fx.make_layout((8, 8), (8, 1))              # (shape, stride): a coord -> index function
fx.crd2idx(fx.make_coord(2, 3), L)              # apply it; the stride decides where a coord lands
fx.size(L) / fx.cosize(L) / fx.rank(L) / fx.get_shape(L) / fx.get_stride(L)
fx.logical_divide(L, tiler)                     # cut a layout into (inside-a-tile, which-tile)
fx.make_identity_layout(shape)                  # coords map to themselves -> the basis of a coord tensor
fx.make_view(fx.make_coord(0, 0), id_layout)    # a coord tensor (has no element_type)

# Tiled copy + swizzle + visualization (05)
tile_mn, tv = fx.make_layout_tv(thr_layout, val_layout)         # threads x values -> a TV layout
tiled = fx.make_tiled_copy(atom, tv, tile_mn)
thr = tiled.get_slice(tid); thr.partition_S(src) / thr.partition_D(dst)   # per-thread memref views
fx.make_composed_layout(fx.static(fx.SwizzleType.get(mask, base, shift)), base)   # LDS bank swizzle
fx.utils.print_typst(layout_or_tiled_copy, file="x.typ")        # Typst diagram (render with the `typst` pkg)
```

A few gotchas worth front-loading:

- `fx.printf` takes only bare `{}` (no `{:.2f}`); a literal `%` is consumed by the
  device printf (write `"mod"`); a true `Boolean` prints as `-1`.
- Device `printf` is not captured by Jupyter — wrap the launch in
  `with wurlitzer.pipes() as (out, _): launch(...); torch.cuda.synchronize()`, then `print(out.read())`.
- `Constexpr` `fp8`/`bf16` math is not rounded until the value is materialized as its
  MLIR type; only `f16`/`f32`/`f64` fold at trace time.
- The layout cells in `04`/`05` (and `01` §6) print at *trace* time, so those notebooks
  set `FLYDSL_RUNTIME_ENABLE_CACHE=0`: a warm JIT disk cache would skip the re-trace, and
  the trace-time prints (and `print_typst`'s diagram files) would vanish on a re-run.

## Running

These notebooks execute kernels, so they need a built/installed FlyDSL (`pip install -e .`
from the project root — see the root README for the build) on a ROCm GPU, with `torch`,
plus a couple of notebook tools:

```bash
pip install jupyter wurlitzer typst
```

`wurlitzer` lets the notebooks show GPU `printf` output inline — Jupyter does not
capture device stdout on its own. `typst` renders the layout diagrams in `05`
(`print_typst`) to SVG inline; without it that notebook still runs and shows the raw
Typst source instead. Then open them with Jupyter, or run headless:

```bash
jupyter nbconvert --to notebook --execute --inplace examples/notebooks/*.ipynb
```

The notebooks ship without a pinned kernelspec, so `nbconvert` (and Jupyter) run them
against your **active** Python environment — make sure `flydsl` imports there. Work
through them in order; `00`–`03` are the foundation for `04`–`05`.

Cell outputs are committed **cleared**; run the cells to populate them.
