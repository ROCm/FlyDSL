---
orphan: true
---

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 FlyDSL Project Contributors -->

# Register allocation (experimental)

`flydsl.experimental.register_allocation` asks the register allocator to place per-thread register
memory at chosen numbers of a register class, such as the A/B operands of an MFMA in given AGPRs, an
accumulator in a given VGPR range, a few uniform values in fixed SGPRs.

The feature is experimental and may change or be removed without a deprecation period. It is
reached only as the subpackage `flydsl.experimental.register_allocation` (also
`fx.experimental.register_allocation`); this page imports it as `reg`:

```python
from flydsl.experimental import register_allocation as reg
```

| API                                                         | Description                                                                                        |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `reg.RegisterAllocator(register_class, start_idx)`          | Allocates new register memory in `register_class`, from `start_idx` upward                         |
| `reg.register_place(storage, register_class, *, start_idx)` | Places register memory created elsewhere, such as MMA fragments                                    |
| `reg.register_value(value, register_class, *, start_idx)`   | Returns an SSA value, such as a `Numeric` or a `Vector`, placed in `register_class`                |
| `register_class[n](value)`, `register_class[a:b](value)`    | `register_value` at `a`, for a value of at most `b - a` registers; `[n]` is `[n:n+1]`              |
| `reg.rocdl.SGPR32`, `.VGPR32`, `.AGPR32`                    | The AMDGPU register classes; numbers are the ISA `sN`, `vN`, `aN`                                  |
| `reg.RegisterClass(target, name, register_bits)`            | Any LLVM register class by name and member width, such as `RegisterClass("amdgcn", "VGPR_32", 32)` |

[`examples/experimental/register_allocation/register_allocation.py`](../../examples/experimental/register_allocation/register_allocation.py)
uses register class indexing and `register_place` in complete kernels.

## Register classes and numbers

The register class is required: it is what the feature places values in. Register memory without
placement, where the compiler chooses both the class and the numbers, is created with
`fx.make_rmem_tensor` or `fx.make_fragment_like`.

Numbers index the members of the class and count whole members: `start_idx=40` in `SGPR32` is
`s40`. Every entry point takes `start_idx`, the first register; the storage or value occupies it
and the registers after it, exactly.

Placement is a register allocation hint, not a guarantee. Nothing checks the result. When the
requested registers are taken, unusable, or the value has to be split or spilled, LLVM places the
value or part of it elsewhere, normally in the same register class, and compilation still succeeds
without a diagnostic, in a few cases with wrong code (see [Limitations](#limitations)). Check the
final ISA.

Even without such conflicts, the result can still fall short of the request in ordinary code, such
as an accumulator read element by element or carried around a loop: part of the value's live range
ends up in other registers. How closely a request is met also depends on the rest of the kernel and
on the LLVM version.

The registers are not reserved for the whole kernel: other values may use them before the first
write and after the last read. Because a fixed number sets a floor on the kernel's register count
(a value in `v200` makes the kernel use at least 201 VGPRs, which can lower occupancy), prefer low
numbers. The lowest registers hold kernel inputs such as the thread ID, the kernel argument
pointer, and the workgroup ID, so start fixed ranges above them.

## `reg.RegisterAllocator` — the register memory allocator

`RegisterAllocator` has the `allocate(T)` / `Storage[T]` interface of
[`fx.SharedAllocator`](../language/storage_and_allocator.md#fxsharedallocator--the-shared-memory-allocator)
and is used the same way inside a `@flyc.kernel` body. Every allocation is placed in the
allocator's class, so group values by class: one struct per class, allocated by an allocator for
that class.

```python
@fx.struct
class BlockRegisters:  # uniform across the block
    base: fx.Int32
    bias: fx.Int32

@fx.struct
class LaneRegisters:  # one value per lane
    x: fx.Int32
    y: fx.Int32
    result: fx.Int32

# Inside a @flyc.kernel body:
sgprs = reg.RegisterAllocator(reg.rocdl.SGPR32, start_idx=40)
vgprs = reg.RegisterAllocator(reg.rocdl.VGPR32, start_idx=32)
block = sgprs.allocate(BlockRegisters)  # s40-s41
lane = vgprs.allocate(LaneRegisters)    # v32-v34

block.base.poke(fx.block_idx.x * 64)
lane.x.poke(X[block.base.peek() + fx.thread_idx.x])
```

Sizes and alignments stay in bytes, as for shared memory: an allocation is aligned to the larger of
`dsl_align_of(T)` and `allocate(..., alignment=...)`. The same alignment constrains register
numbers: an allocation aligned to 16 bytes starts at a multiple of 4 registers, so the allocator
rounds its next number up.

```python
regs = reg.RegisterAllocator(reg.rocdl.VGPR32, start_idx=65)
count = regs.allocate(fx.Int32)                               # v65
tile = regs.allocate(fx.Vector[fx.Float32, 4], alignment=16)  # v68-v71
```

A type's alignment describes its byte layout, not hardware register rules. On gfx90a and later,
64-bit and wider VGPR/AGPR operands start at even registers, yet `fx.Vector[fx.Float16, 4]` is only
2-byte aligned. Raise the alignment with `fx.Align[T, A]` or `alignment=` where an instruction
requires it; a misaligned start leaves the hint without a usable register (see
[Limitations](#limitations)).

## `reg.register_place` — placing existing register memory

`register_place` places register memory that the allocator did not create, such as the fragments
made by the MMA API. Call it on the fragment itself, not on a view or slice of it, in the block that
creates it; the placement covers the whole fragment.

`start_idx` is used as given, without rounding: choose a start that meets the instruction rules
above, such as an even register for 64-bit and wider operands on gfx90a and later.

```python
frag_A = thr_mma.make_fragment_A(A)
frag_B = thr_mma.make_fragment_B(B)
frag_C = thr_mma.make_fragment_C(C)

# Request A/B in AGPRs and C in VGPRs.
reg.register_place(frag_A, reg.rocdl.AGPR32, start_idx=0)  # a0-a3
reg.register_place(frag_B, reg.rocdl.AGPR32, start_idx=4)  # a4-a7
reg.register_place(frag_C, reg.rocdl.VGPR32, start_idx=64)  # v64-v79
```

`RegisterAllocator` is a layout convenience over `register_place`.

## `reg.register_value` — placing an SSA value

`register_value` places a value instead of register memory: a `Numeric` or a `Vector` computed in
the kernel, or a kernel argument. It returns the value with the placement; only the returned value
is placed, so use it instead of the original.

```python
bias = reg.register_value(bias, reg.rocdl.SGPR32, start_idx=40)        # s40
x = reg.register_value(X[i], reg.rocdl.VGPR32, start_idx=32)           # v32
y = reg.register_value(x * 3 + bias, reg.rocdl.VGPR32, start_idx=33)  # v33
```

Indexing a register class gives the same call with a size check. `[a:b]` places a value of at most
`b - a` registers (32 bits each for the classes in `reg.rocdl`) at registers `a` to `b - 1`, and
`[n]` is `[n:n+1]`. A range takes both a start and a stop, and a value that does not fit raises
`ValueError`:

```python
bias = reg.rocdl.SGPR32[40](bias)   # reg.register_value(bias, reg.rocdl.SGPR32, start_idx=40)
acc = reg.rocdl.VGPR32[64:68](acc)  # v64-v67; acc has at most 4 x 32 bits
```

The instructions that compute the value are chosen as without placement; when one of them writes
other registers, its result is copied into the requested ones. In particular, placing the value read
from an MMA fragment (`frag.load()`) does not make the MMAs accumulate there: place the fragment
itself with `register_place` for that.

The limitations below apply to every entry point.

## Limitations

Placement does not change which instructions compute a value. The list below describes behavior
observed so far; it is not exhaustive, and other kernels may show other gaps between the request and
the result.

- **Conflicts move values silently.** If the requested registers are held by another live value,
  including another placed value, cannot hold the value (a misaligned start, or a range past the end
  of the register file), or register pressure forces a split or spill, the value or part of its
  live range ends up in other registers, with copies.
- **Placing part of a placed value.** Every read of placed storage is hinted, including a read of a
  single element, such as storing an MMA accumulator element by element. The hint of the element
  and the hint of the whole value compete, so the element can be copied out of the whole value,
  often through other registers, may not end up in its own register, and later reads of the whole
  value may come from a copy; the instructions that last write the whole value, such as the final
  MMAs into an accumulator, may write other registers. Where possible, read the whole storage once
  (for example `frag.load()`) and work on that value.
- **Loop-carried VGPR values may leave their registers.** A placed VGPR value carried around a
  loop, such as an MMA accumulator in VGPRs, can be in its registers within each iteration but not
  across the loop's back edge. Whether this happens depends on the rest of the kernel: with the
  accumulators of `examples/04-preshuffle_gemm.py` placed in VGPRs, the last MMA of each iteration
  writes other registers and the first MMA of the next iteration writes the placed ones again; a
  simpler K loop keeps them in place. MMA accumulators placed in AGPRs stayed in their registers
  across iterations in every kernel tried.
