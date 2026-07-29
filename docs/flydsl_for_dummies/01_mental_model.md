# The Mental Model

If you remember one thing from this book, make it this:

> **FlyDSL Python code does not run on the GPU. It runs on the host, once, to
> *build* an intermediate representation (IR). That IR is then compiled to a GPU
> binary.**

This is the concept that trips up every C++ GPU programmer, because in HIP the
code you write between `{` and `}` in a `__global__` function *is* the code that
executes on every thread. In FlyDSL, the code you write inside `@flyc.kernel` is a
**program that emits a program**. Get comfortable with this and the rest is
mechanical.

## Two worlds: tracing vs. execution

Consider this fragment:

```python
@flyc.kernel
def add_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    x = A[tid] + B[tid]
    C[tid] = x
```

In HIP the analogous code *is* the device program: each of the thousands of
threads evaluates `A[tid] + B[tid]`. In FlyDSL, when the framework **traces** this
function on the host:

- `fx.thread_idx.x` does not return an integer. It returns a **handle to an SSA
  value** — an MLIR `Value` representing "whatever `threadIdx.x` will be at
  runtime."
- `A[tid]` does not load a float. It **emits an op** into the IR that will, at
  runtime, perform the load.
- `+` does not add two numbers. It **emits an `arith.addf` op**.
- `C[tid] = x` **emits a store op**.

The Python function is executed *exactly once*, on the host, and its side effect
is a graph of MLIR operations. No arithmetic happens during tracing; the *record
of arithmetic to perform* is what is produced.

> **HIP/CK-Tile → FlyDSL.** In CK-Tile, C++ templates are instantiated at
> *compile time* to specialize a kernel for concrete tile sizes and types; the
> body still runs on the device. In FlyDSL, the Python body runs at *trace time*
> (host, once) and its job is to *generate* the device body. The Python
> interpreter plays the role the C++ template instantiator plays for you in
> CK-Tile — except you write ordinary imperative Python instead of template
> metaprogramming.

### Why types look different

Because tracing builds IR, you cannot use a bare Python `float` for a device
value — there is nothing to trace it into. Instead FlyDSL gives you typed value
wrappers (`fx.Float32`, `fx.Int32`, vector and struct types) that know how to emit
the right ops. This is why Chapter *arithmetic_types* in the main docs exists and
why "just use `float`" does not work: a `float` is a host constant, an
`fx.Float32` is a node in the IR.

There are two kinds of "known at trace time" values you *can* use directly:

- **Python constants** you close over (loop bounds, tile sizes). These are baked
  into the IR as literals — exactly like a `constexpr` in C++.
- **`fx.Constexpr` parameters** — kernel arguments whose value is fixed per
  compilation and specialize the generated code (see §1.4).

## `@flyc.kernel` vs `@flyc.jit`

FlyDSL splits a kernel launch into two decorated functions:

| Decorator | Role | C++ analogy |
|-----------|------|-------------|
| `@flyc.kernel` | The **device** function. Its traced body becomes a `gpu.func` (→ an AMDGPU kernel). | the `__global__` function |
| `@flyc.jit` | The **host launcher**. It sets up arguments, computes the grid, and calls `.launch(...)`. | your host code + `hipLaunchKernelGGL` |

```python
@flyc.kernel
def add_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    C[tid] = A[tid] + B[tid]

@flyc.jit
def add(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None)):
    n = A.shape.unpack()[0]
    add_kernel(A, B, C).launch(grid=(n // 256, 1, 1),
                               block=(256, 1, 1), stream=stream)
```

Calling `add(a, b, c)` from ordinary Python is what kicks off the whole machine:
argument inspection → cache lookup → (on a miss) trace → compile → load →
launch. Calling `add_kernel(...)` on its own only makes sense *inside* a
`@flyc.jit` trace; it returns a launcher object, it does not run anything.

- `@flyc.jit` is defined at `python/flydsl/compiler/jit_function.py:1618`
  (class `JitFunction`, `python/flydsl/compiler/jit_function.py:1139`).
- `@flyc.kernel` is defined at `python/flydsl/compiler/kernel_function.py:614`
  (class `KernelFunction`, `python/flydsl/compiler/kernel_function.py:431`).

> **HIP/CK-Tile → FlyDSL.** `add_kernel(...).launch(grid=..., block=..., stream=...)`
> is the FlyDSL `hipLaunchKernelGGL(add_kernel, grid, block, 0, stream, ...)`.
> `grid`/`block` are 3-tuples exactly like HIP's `dim3`. Dynamic LDS is requested
> with `launch(smem=...)`, the analogue of the shared-memory-bytes argument.

## Control flow is traced too

Because the body is traced, control flow needs care. There are two regimes:

- **Compile-time (unrolled) control flow.** A plain Python `for` over
  `fx.range_constexpr(...)` is unrolled during tracing — the loop body is emitted
  N times, like a `#pragma unroll` with a known bound. Use this for tiling loops
  whose bounds are known at trace time.

  ```python
  for i in fx.range_constexpr(num_tiles):   # unrolled at trace time
      ...
  ```

- **Runtime control flow.** A `for i in range(...)` (note: plain `range`) or an
  `if` on a *dynamic* value is rewritten into structured-control-flow ops
  (`scf.for`, `scf.if`) that exist in the IR and execute on the device. Loops that
  carry state across iterations use `range(start, stop, step, init=[...])`, which
  yields loop-carried values:

  ```python
  for k, (acc,) in fx.range(0, K, BLOCK_K, init=[acc0]):   # scf.for on device
      acc = acc + work(k)
      fx.yield_(acc)
  ```

This rewriting is not magic you must invoke — it is done by an **AST rewriter**
that transforms your function's syntax tree *before* tracing (Chapter 2, §2.3).
`and`/`or`/`not`, chained comparisons, `if`, `for`, `while`, and `yield` are all
rewritten to DSL dispatch calls (`python/flydsl/compiler/ast_rewriter.py`).

A few rules follow from this and recur in the puzzles:

- Don't define a value only inside one branch of an `if` and use it afterwards —
  the merged value is undefined. Hoist it, or return a single merged value.
- Avoid early `return`/`yield` inside traced control flow; keep one exit path so
  MLIR result types stay well-defined.

## `Constexpr`: specialization without templates

An `fx.Constexpr` kernel parameter is fixed *per compilation*: its value is baked
into the IR, and a different value produces a different compiled kernel. This is
how FlyDSL does what C++ non-type template parameters do for CK-Tile.

```python
@flyc.kernel
def scaled(A: fx.Tensor, C: fx.Tensor, ALPHA: fx.Constexpr):
    tid = fx.thread_idx.x
    C[tid] = A[tid] * ALPHA      # ALPHA is a literal in the generated IR
```

> **HIP/CK-Tile → FlyDSL.** `template <int Alpha>` → `ALPHA: fx.Constexpr`.
> `template <typename T>` → a `T: type[fx.Float16]`-style type parameter. Both
> participate in the compilation cache key, so each distinct specialization is
> compiled and cached once.

## What you gain, and what it costs

**Gain.** You write tiling and register-fragment logic in imperative Python with
real loops, real conditionals, and a rich layout algebra library — instead of
deeply nested C++ templates. The same source lowers to tuned AMDGPU code, and JIT
compilation is cached so repeated launches are cheap.

**Cost.** You must always keep "trace time vs run time" straight. When something
behaves oddly, the first question is *"is this value known at trace time or only
at run time?"* The debugging workflow in Chapter 2 (dumping the IR at each stage)
exists precisely so you can *see* what your Python actually built.

The next chapter follows a kernel all the way from this traced Python down to the
HSACO that the GPU executes.
