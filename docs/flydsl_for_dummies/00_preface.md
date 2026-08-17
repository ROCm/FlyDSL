# Preface {-}

## Who this book is for

You already write GPU kernels in **HIP C++** and **CK-Tile C++**. You know what a
wavefront is, why LDS bank conflicts hurt, how MFMA instructions consume register
fragments, and how a tile flows from global memory through shared memory into the
matrix cores. What you have *not* done much of is write kernels in a **Python
DSL**.

This book teaches **FlyDSL** to exactly that reader. It does not re-explain GPUs.
Instead it answers the questions a C++ GPU programmer actually asks when they open
a FlyDSL kernel for the first time:

- *"This is Python. Where does the GPU code come from? When does it run?"*
- *"In CK-Tile I instantiate a `tile_distribution`. What is the FlyDSL equivalent?"*
- *"How do `zipped_divide`, `partition_S`, and `make_fragment_like` map to the
  tiling and register-fragment math I already do by hand?"*
- *"How does a `.py` file become an HSACO code object, and how does that pipeline
  compare to `hipcc`?"*

## How FlyDSL relates to CuTe

If you have read any CUTLASS 3.x, you have met **CuTe** — the layout algebra at its
core, where a `Layout` is a `Shape:Stride` pair over hierarchical `IntTuple`s, and
`logical_divide`, `composition`, `complement`, thread-value layouts, copy/MMA
*atoms*, `TiledCopy`/`TiledMma`, and register *fragments* are the vocabulary for
mapping tiles onto threads. **FlyDSL's layout algebra is that same algebra.** The
names line up almost one to one — `make_layout`, `logical_divide`, `zipped_divide`,
`composition`, `partition_S/D`, `make_fragment_like`, `make_tiled_copy`,
`make_tiled_mma` — so if you know CuTe you already know most of Chapters 7–9, and
the CuTe identities you rely on (divide/product duality, the TV-layout contract)
hold here unchanged.

The relationship is *inspiration and semantics*, not a port. CuTe is a C++
**template metaprogramming** library: layouts are types, and the algebra is
evaluated by the compiler as it instantiates your kernel — much like the CK-Tile
templates you already write. FlyDSL takes the same algebra but expresses it as a
**Python DSL that traces to first-class MLIR ops** (the `fly` dialect, Chapter 2):
a layout is a runtime value you can `print`, and `logical_divide` is an op a pass
lowers to arithmetic, not a type computed away at compile time. The payoff is that
the tiling decisions are inspectable objects, and the target is AMD CDNA/RDNA via
ROCDL rather than NVIDIA PTX. Where a section maps a FlyDSL op to its CuTe (and
CK-Tile) equivalent, the `HIP/CK-Tile → FlyDSL` boxes call it out; the repository's
`docs/cute_layout_algebra_guide.md` gives the full mathematical correspondence.

> **CuTe → FlyDSL.** Same algebra, different host: CuTe is C++ templates resolved
> at compile time; FlyDSL is Python that emits MLIR layout ops resolved by compiler
> passes. `Layout<Shape,Stride>` becomes `fx.make_layout(shape, stride)`, and a
> `TiledCopy`/`TiledMMA` becomes an object you build and slice at trace time.

## What this book covers

1. **The mental model** — the single biggest shift: FlyDSL Python code *builds*
   an intermediate representation; it does not execute on the GPU (Chapter 1).
2. **The compilation pipeline** — from Python AST to Fly-dialect MLIR, through the
   named lowering passes, down to ROCDL/LLVM and the HSACO fat binary, with a
   side-by-side comparison to `hipcc` and CK-Tile template instantiation
   (Chapter 2).
3. **The language foundations** — how C++ control flow (`if`/`for`/`while`) maps to
   FlyDSL's trace-time vs. run-time regimes (Chapter 3), and the type system: how
   Python types become value types at each IR level (Chapter 4).
4. **Working with tensors** — how to access shape, stride, and individual elements of
   an `fx.Tensor`; element indexing, `None`-slice selection, and `make_view` for
   reinterpreting memory (Chapter 5).
5. **The Fly dialect, formally** — the layout algebra (Chapter 6), tiling and
   partitioning (Chapter 7), data movement / copy atoms (Chapter 8), and MMA
   atoms (Chapter 9), each with small worked examples and a mapping to the
   CK-Tile / CuTe concept you already know.
5. **Close to the metal** — the intrinsic level beneath the atoms: loads and stores
   down to `buffer_load`/`ds_read` (Chapter 10), MFMA down to the `rocdl.mfma.*`
   instruction and its operand/accumulator VGPRs (Chapter 11), and the escape
   hatches for when the high-level dialect is not enough — small MFMAs outside a
   GEMM, cross-lane ops, inline asm (Chapter 12).
6. **Worked examples** — three complete kernels annotated line by line
   (Chapter 13).
7. **Debugging** — how (and how far) you can actually debug a FlyDSL kernel:
   the three layers of failure, the IR-dump and `printf` tools, a repeatable
   isolation loop, and two real worked failures (Chapter 14).
8. **A reference appendix** — environment variables, an API cheat-sheet, and a
   HIP/CK-Tile ↔ FlyDSL glossary (Chapter 15).

## How to read it

Read Chapters 1–5 first — they install the mental model, the control-flow rules,
the type system, and the tensor access idioms that make everything else make sense.
Then either continue linearly, or jump to the algebra chapters (6–9) as a reference
while you work the companion **[FlyDSL Puzzles](../../puzzles/)**.
The puzzles are where the learning actually sticks: each one references the section
here that explains it.

## Conventions

Throughout, boxes like this bridge from what you know to what is new:

> **HIP/CK-Tile → FlyDSL.** Short statements of the form "you used to write *X*;
> in FlyDSL you write *Y*."

Code is shown in fenced blocks. FlyDSL Python uses the conventional aliases used
across the repository:

```python
import flydsl.compiler as flyc   # @flyc.kernel, @flyc.jit
import flydsl.expr as fx         # types, layout algebra, atoms, ops
```

File references point at the actual source, e.g.
`python/flydsl/compiler/jit_function.py:1618`, so you can read the implementation
when the prose is not enough.

Everything targets **CDNA (gfx942 / gfx950)** — MI300X / MI350, wave size 64,
MFMA matrix cores — because that is where the reference kernels and the puzzles
live. Where RDNA (WMMA, wave32) differs, it is called out explicitly.
