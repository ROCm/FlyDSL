# Lesson 00 — Hello FlyDSL

**Run:** `HIP_VISIBLE_DEVICES=2 python3 learn_fmha/lesson_00_hello_flydsl.py` → `PASS`.

## What we build
The minimum: a kernel that adds 1.0 to every element of a 256-vector, launched with one block of
256 threads. No matrix math — just the kernel/launch machinery so later lessons can focus on GPU ideas.

## CK Tile → FlyDSL dictionary
| CK Tile / HIP | FlyDSL |
|---|---|
| `__global__` device function | `@flyc.kernel` |
| host launcher + `<<<grid,block>>>` | `@flyc.jit` wrapper calling `.launch(grid=, block=)` |
| `threadIdx.x`, `blockIdx.x` | `fx.thread_idx.x`, `fx.block_idx.x` |
| `BufferView` / pointer | `fx.buffer_ops.create_buffer_resource(T)` |
| typed device scalars | `fx.Int32`, `fx.Index`, `fx.Float32`, `fx.BFloat16` |

**Key mental model:** code inside `@flyc.kernel` does **not** run in Python on the device. You are
*building an expression graph* (FlyDSL's IR) that gets lowered to MLIR → ROCDL → gfx942 ISA. So
`x = buffer_load(...); y = x + 1.0` records "emit a load, emit an add," it doesn't compute in Python.
That's why scalars are `fx.Int32(...)` etc. — they're IR values. (Same idea as building a CK
`tile_program` expression vs. running C++.)

## Two gotchas baked in on purpose
1. **Never name the `@flyc.jit` wrapper `launch`** — it shadows the `.launch()` method → infinite
   recursion. Convention: `run_*`.
2. **Two buffer APIs exist.** The high-level layout/copy-atom style (`make_buffer_tensor` +
   `copy_atom`, see `examples/01-vectorAdd.py`) is idiomatic for plain GEMMs and hides data
   movement. **This tutorial deliberately uses the low-level raw style** (`create_buffer_resource` +
   explicit `buffer_load`/`buffer_store` by element index), because every optimization we teach
   (wide loads, LDS staging, transposes, prefetch) is *about* explicit data movement — the
   high-level API would hide the very thing we're learning to control.

## `buffer_load` / `buffer_store` index by ELEMENT
`buffer_load(res, idx, vec_width, dtype)` reads `vec_width` elements of `dtype` starting at element
`idx` (not byte offset). `vec_width=4` is one 128-bit load — we'll use that in Lesson 02/09.

## Next
Lesson 01: the MFMA instruction and its all-important lane↔element layout.
