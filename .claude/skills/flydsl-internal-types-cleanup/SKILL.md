---
name: flydsl-internal-types-cleanup
description: >
  Clean up FlyDSL kernel code by replacing direct scf/arith/vector/memref dialect calls
  with FlyDSL internal types and helpers while preserving correctness, performance, and
  generated ASM. Use when refactoring FlyDSL kernels, removing redundant wrappers, or
  updating docs/skills to prefer fx.Int32/fx.Index/fx.Float32, ArithValue, Vector, and
  range(..., init=...).
---

# FlyDSL Internal Types Cleanup

## Default Rule

Prefer FlyDSL internal types and high-level helpers in kernel code:

- Constants and casts: `fx.Int32`, `fx.Int64`, `fx.Index`, `fx.Float32`
- Arithmetic and comparisons: Python operators on `ArithValue` / `Numeric`
- Runtime select: `cond.select(a, b)` or `arith.select(...)` only when a helper boundary requires it
- Vectors: `Vector` (`Vec`) indexing, `Vec.from_elements`, `Vec.filled`, `.bitcast(...)`, `.to(...)`, `.store(...)`
- Register memory: `fx.memref_alloca`, `fx.memref_load_vec`, `fx.memref_store_vec`
- Runtime loops with carried state: `range(start, stop, step, init=[...])` using `fx.Index(...)` bounds
- Compile-time loops: `range_constexpr(...)`

Avoid new direct `scf.*`, `vector.*`, `memref.*`, `arith.index`, `arith.index_cast`, and `arith.trunc_f` in kernel bodies unless a lower-level boundary requires the exact op.

## Replacement Map

| Low-level form | Preferred form |
|---|---|
| `arith.constant(0, type=T.i32)` | `fx.Int32(0)` |
| `arith.constant(0, index=True)` / `arith.index(0)` | `fx.Index(0)` |
| `arith.constant(1.0, type=T.f32)` | `fx.Float32(1.0)` |
| `arith.index_cast(T.i32, x)` | `fx.Int32(x)` |
| `arith.index_cast(T.index, x)` | `fx.Index(x)` |
| `vector.extract(v, static_position=[i], ...)` | `Vec(v)[i]` |
| `vector.bitcast(T.vec(...), v)` | `Vec(v).bitcast(fx.Int32)` etc. |
| `vector.from_elements(T.vec(n, T.i32), xs)` | `Vec.from_elements(xs, fx.Int32)` |
| `vector.store(v, memref, [idx])` | `Vec(v).store(memref, [idx])` |
| `arith.trunc_f(T.bf16x4, v)` | `Vec(v).to(fx.BFloat16)` |
| `arith.addf/mulf` | `a + b`, `a * b` |
| `arith.select(cond, a, b)` | `cond.select(a, b)` when `cond` is an `ArithValue` |

## Important Exceptions

Keep the exact lower-level op when it encodes semantics that internal types do not expose:

- `llvm.InlineAsmOp` for hand-scheduled ISA snippets
- `llvm.LoadOp` / `llvm.StoreOp` when `volatile`, `nontemporal`, address space, or alignment must be explicit
- `arith.*FOp(..., fastmath=...)` when performance depends on fastmath flags
- `arith.DivUIOp` / `arith.RemUIOp` for unsigned integer division/remainder
- `rocdl.*` intrinsics and MFMA/WMMA/TDM ops
- Backend dialect/C++ lowering docs and implementation code

Do not hide these exceptions behind new helper wrappers just to remove the visible op. If exact semantics are required, keep the direct op at the boundary and document why.

## Control Flow

- Compile-time / constant conditions must be written as `if const_expr(condition): ...`. Do not rely on a plain Python `if` unless the condition is already a Python `bool`.
- Use ordinary Python `if` on runtime values only when the AST rewriter keeps branch-local values and side effects correct.
- For runtime branches inside nested helper functions, wrap the dispatch in a local `@flyc.jit` helper. This keeps branch side effects and loop-carried state in the right rewritten region.
- For complex runtime branches with side effects, loop-carried state, or branch-local definitions, split branch bodies into local helper functions and dispatch through a local `@flyc.jit` helper. Verify correctness and ASM/perf.
- Do not hand-write `scf.IfOp` in new kernel code unless the `@flyc.jit` helper pattern cannot express the required branch.
- Use `range(..., init=[...])` for runtime loops with carried state; unwrap init values only if the API specifically requires raw `ir.Value`.

Pattern:

```python
def _then_path():
    ...

def _else_path():
    ...

@flyc.jit
def _dispatch():
    if runtime_cond:
        _then_path()
    else:
        _else_path()

_dispatch()
```

## Verification Loop

For performance-sensitive kernels:

1. Record baseline shape coverage, timing, ASM hash, VGPR/SGPR counts, and spill counts.
2. Apply one cleanup group at a time.
3. Run correctness on small and large representative shapes.
4. Compare performance; for strict cleanups, compare ASM hash.
5. If performance drops or results change, revert that cleanup group and keep the lower-level op.

Recommended checks:

```bash
PYTHONPATH=python:. FLYDSL_RUNTIME_ENABLE_CACHE=0 <kernel test command>
FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 PYTHONPATH=python:. <small compile command>
sha256sum ~/.flydsl/debug/<kernel>/21_final_isa.s
```
