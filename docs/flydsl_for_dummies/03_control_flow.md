# Control Flow: `if`, `for`, `while`

Chapter 1 established the one rule that governs everything: your Python body runs
*once, on the host, to build IR* (§1.1). Control flow is no exception. When you
write a `for` or an `if`, the question is never "what does this do?" — it is
**"does this run at trace time, shaping the IR, or does it become a device-side
op that runs per-thread at run time?"** This chapter answers that for every
construct, with the C++ equivalent alongside.

Two axes decide the answer:

1. **Is the condition / loop bound known at trace time?** A Python constant or a
   `Constexpr` (Ch. 4) is known; a value derived from `fx.thread_idx.x`, a tensor
   element, or a kernel `fx.Int32` argument is a *runtime* value.
2. **Does the branch or loop produce a value used afterwards** (a live-out)? If so,
   that value must cross the control-flow boundary as an explicit result.

Everything below is a consequence of these two axes plus the **AST rewriter**
(§2.3), which transforms your `if`/`for`/`while` into DSL dispatch calls *before*
tracing. You never call the dispatch functions yourself; you write ordinary
Python and the rewriter does the work. Set `FLYDSL_DEBUG_AST_DIFF=1` to see the
rewrite for any kernel.

## The two regimes

| Regime | You write | Rewriter emits | Runs |
|--------|-----------|----------------|------|
| Compile-time | `range_constexpr`, `if fx.const_expr(c):` | nothing — plain Python | at trace time (unrolled / dead-code-eliminated) |
| Run-time | plain `range(...)`, `if dyn:`, `while dyn:` | `scf.for` / `scf.if` / `scf.while` ops | on the device, per invocation |

The transformers responsible (all in
`python/flydsl/compiler/ast_rewriter.py`) are:

| Python construct | Transformer | Becomes |
|------------------|-------------|---------|
| `and`/`or`/`not`, `a < b < c` | `RewriteBoolOps` | `arith.andi`/`arith.ori` on DSL values |
| `if` / ternary | `ReplaceIfWithDispatch` | `scf.if` (dynamic) or inlined branch (static) |
| `for` | `InsertEmptyYieldForSCFFor` | unrolled Python, or `scf.for` |
| `yield` (in a loop body) | `ReplaceYieldWithSCFYield` | `scf.yield` |
| `while` | `CanonicalizeWhile` | `scf.while` |

> **HIP/CK-Tile → FlyDSL.** The compile-time regime is your preprocessor / template
> world: it decides *which code exists*. The run-time regime is the `if`/`for` that
> survives into the `__global__` body and executes on each wavefront. In C++ you
> keep these straight by where you write them (host template vs device body); in
> FlyDSL you keep them straight by *which construct you pick*.

## `for`: unrolled vs. runtime

### Unrolled: `range_constexpr`

`range_constexpr` is literally `range` (`python/flydsl/expr/primitive.py:331`); its
only job is to *signal intent* to the rewriter, which leaves the loop as plain
Python. Python then executes it during tracing, emitting the body once per
iteration. The bound must be known at trace time.

```python
# examples/01-vectorAdd.py:56 — one predicate slot emitted per element, inlined
for a in fx.range_constexpr(fx.size(thr_pC.shape).unpack()):
    thr_pC[a] = fx.elem_less(thr_cC[a], (M, N))
```

There is no loop in the resulting IR — just N copies of the body. This is a
`#pragma unroll` with a compile-time trip count, and it is the right tool for
tiling loops whose extents come from `Constexpr` block sizes.

### Runtime: plain `range`

A plain `range(...)` over a value that is not (only) trace-time known becomes an
`scf.for` that runs on the device. Variables you reassign inside the body are
auto-detected as **loop-carried values** (the `scf.for` `iter_args`):

```python
@flyc.kernel
def _accumulate(Out: fx.Tensor, n: fx.Int32):
    acc = fx.Int32(0)
    for i in range(n):            # scf.for %i = 0 to %n step 1
        acc = acc + fx.Int32(1)   # acc is auto-carried across iterations
    Out[0] = acc
```

If the body reassigns nothing, you get a side-effect-only `scf.for` with no
results.

### Runtime with explicit loop-carried state: `range(..., init=[...])`

For the common kernel pattern — carrying an accumulator register tile across a
K-loop — make the carried state explicit with `init=[...]` and `yield`:

```python
# adapted from kernels/gemm/preshuffle_gemm.py:516
for iv, state in range(0, num_tiles - 1, 1, init=[frag_C.load()]):
    frag_C.store(state[0])        # unpack the carried accumulator
    mma_kloop(iv)                 # ... do work, updating frag_C ...
    results = yield [frag_C.load()]   # hand the new value to the next iteration
frag_C.store(results)             # results holds the final carried value
```

This lowers to:

```mlir
%r = scf.for %iv = %c0 to %end step %c1
       iter_args(%acc = %init) -> (vector<...xf32>) {
  ...
  scf.yield %new_acc : vector<...xf32>
}
```

`init=[...]` is the initial value list; each element must be a DSL value backed by
an `ir.Value` (a `fx.Int32`, a fragment `.load()`, etc.). `state` inside the loop
is the tuple of `iter_args`; `yield [...]` supplies the next iteration's values,
and the loop expression's `results` are the final values after the last iteration.

> **Note on spelling.** Runtime loops use Python's **builtin `range`** and a plain
> **`yield`** — not `fx.range` / `fx.yield_`. The rewriter recognizes the builtins
> in a traced body; there is no `fx.range` to import.

> **HIP/CK-Tile → FlyDSL.** `for i in range_constexpr(N)` is your `#pragma unroll`
> fixed-count loop. `for iv, state in range(0, K, BLOCK_K, init=[acc])` is the
> runtime K-loop of a GEMM: the accumulator VGPRs you carry by hand across
> iterations become the `scf.for` `iter_args`, and `yield [acc]` is the assignment
> back into those registers for the next step.

## `if` / `else`

### Compile-time: `fx.const_expr`

Wrapping the condition in `fx.const_expr(...)` tells the rewriter the test is a
host Python value; it evaluates the branch during tracing and emits only the taken
side — no `scf.if`, no runtime cost (`ast_rewriter.py:724`). This is how you
specialize a kernel on a `Constexpr` flag.

```python
# pervasive in kernels/gemm/preshuffle_gemm.py — pick a code path at trace time
if fx.const_expr(use_async_copy):
    dma_a_to_lds(fx.Int32(0), 0)      # only these ops are emitted ...
else:
    fx.copy(buf_copy, gA, sA)         # ... or only these
```

It works as a compile-time ternary too:

```python
acc_zero = (Vec.filled(n, 0, fx.Int32) if fx.const_expr(is_int8)
            else Vec.filled(n, 0.0, fx.Float32))
```

> **HIP/CK-Tile → FlyDSL.** `if fx.const_expr(FLAG):` is `if constexpr (Flag)` — the
> untaken branch is never emitted, so it need not even type-check against the
> generated IR.

### Runtime: `scf.if`

An `if` whose condition is a DSL value (anything derived from `thread_idx`, a
load, a kernel argument, or a comparison of those) is rewritten to `scf.if`
(`ast_rewriter.py:597`). Two shapes come up.

**Side-effect only** (no value escapes the branch):

```python
@flyc.kernel
def conditional_store(Out: fx.Tensor, threshold: fx.Int32):
    tid = fx.thread_idx.x
    if tid < threshold:               # scf.if %cond { ... }
        Out[tid] = fx.Float32(1.0)    # store happens only on the taken lanes
```

**Live-out merge** — a value defined on both sides and used afterwards. Define it
*before* the `if` (so both branches have something to merge), reassign in each
branch, and use it after; the rewriter yields it from both regions as the
`scf.if` result:

```python
@flyc.kernel
def clamp_relu(A: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    v = A[tid]
    if v < fx.Float32(0.0):
        v = fx.Float32(0.0)   # then-branch value
    else:
        v = v                 # else-branch value (explicit)
    C[tid] = v                # v is the scf.if result, valid on all lanes
```

lowering to:

```mlir
%v = scf.if %cond -> (f32) {
  scf.yield %zero : f32
} else {
  scf.yield %loaded : f32
}
```

### Branchless: `arith.select`

When you just need to pick one of two scalar values and there are no side effects,
`arith.select(cond, a, b)` is a single hardware select — no control flow, no
divergence penalty:

```python
# kernels/attention/pa_metadata.py:983 style
out = fx.Int32(fx.arith.select(cond.ir_value(), a.ir_value(), b.ir_value()))
```

Prefer it over `scf.if` for simple value choices; reserve `scf.if` for branches
that must *skip work* (e.g. an out-of-bounds store).

> **HIP/CK-Tile → FlyDSL.** A runtime `if` is the device-side `if` in your
> `__global__` body → `scf.if`. `arith.select` is the ternary `cond ? a : b` that
> the compiler already turns into a `v_cndmask`; use it exactly where you would
> avoid a branch in HIP.

> **Gotcha — the live-out rule.** Do **not** define a value *only* inside one branch
> and use it after the `if`. MLIR's SSA form requires that a value used after
> `scf.if` be produced by *both* regions (dominance); the rewriter cannot invent a
> value for the missing side. Either hoist an initial definition before the `if`
> (as in `clamp_relu` above) or don't use the variable afterwards. This is the
> failure documented in `tests/system/test_if_liveout_minimal.py`.

## `while`

Python `while` over a dynamic condition becomes `scf.while`
(`ast_rewriter.py:1399`). Reassigned variables are carried the same way as a
runtime `for`. Use it for loops whose trip count is not an affine `range` — e.g.
walking a work queue until a sentinel:

```python
i = fx.Int32(0)
total = fx.Int32(0)
while i < n:                 # scf.while { condition } { body }
    total = total + A[i]
    i = i + fx.Int32(1)
Out[0] = total
```

The condition region emits `scf.condition`; the body region emits `scf.yield` of
the updated carried state. `while ... else:` is not supported and raises
`NotImplementedError`.

> **HIP/CK-Tile → FlyDSL.** This is your device-side `while` loop. Reach for it when
> the bound is data-dependent (variable-length KV pages, ragged batches); use
> `range`/`range_constexpr` whenever the trip count is affine.

## Rules of thumb

These follow directly from "the body is traced into SSA IR," and each one has bitten
real kernels:

- **Keep carried state explicit and compact.** Only what changes across iterations
  should be an `iter_arg`. Fewer carried values means simpler, faster IR.
- **No early `return` / branch-local `yield`.** A kernel must have one exit path so
  the emitted `func`/`gpu.func` has well-defined result types. An early `return`
  aborts tracing mid-IR and leaves the module malformed.
- **Invalidate cached smem views after `scf.for`.** If you recreate shared-memory
  views inside a loop, clear `SmemPtr._view_cache = None` after the loop so a stale
  SSA value (defined inside the loop's region) is not reused outside it — otherwise
  you hit an MLIR dominance error. See the data-movement chapter (Ch. 7) for the
  LDS view pattern.
- **Booleans on DSL values are ops.** `a < b`, `a & b`, `cond1 and cond2` on DSL
  values emit `arith` comparisons/`andi`/`ori`; they do not short-circuit like
  host Python. Keep both operands cheap.

With control flow settled, the remaining foundational question is *what the values
flowing through these loops and branches actually are* — the type system, next.
