<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 FlyDSL Project Contributors -->

# Register allocation (experimental)

`flydsl.experimental.register_allocation` asks the register allocator to place per-thread register
memory at chosen numbers of a register class: the A/B operands of an MFMA in given AGPRs, an
accumulator in a given VGPR range, a few uniform values in fixed SGPRs.

The feature is experimental and may change or be removed without a deprecation period. It is not
reachable from the stable `fx` namespace. Import its subpackage explicitly, so that every use is
visibly experimental:

```python
from flydsl.experimental import register_allocation as fx_regalloc
```

| API                                                                 | Description                                                                                        |
| ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `fx_regalloc.RegisterAllocator(register_class, start_idx)`          | Allocates new register memory in `register_class`, from `start_idx` upward                         |
| `fx_regalloc.register_place(storage, register_class, *, start_idx)` | Places register memory created elsewhere, such as MMA fragments                                    |
| `fx_regalloc.rocdl.SGPR32`, `.VGPR32`, `.AGPR32`                    | The AMDGPU register classes; numbers are the ISA `sN`, `vN`, `aN`                                  |
| `fx_regalloc.RegisterClass(target, name, register_bits)`            | Any LLVM register class by name and member width, such as `RegisterClass("amdgcn", "VGPR_32", 32)` |

[`examples/experimental/register_allocation.py`](../../examples/experimental/register_allocation.py)
uses both entry points in complete kernels.

## Register classes and numbers

The register class is required: it is what the feature places values in. Register memory without
placement, where the compiler chooses both the class and the numbers, is created with
`fx.make_rmem_tensor`.

Numbers index the members of the class and count whole members: `start_idx=40` in `SGPR32` is
`s40`. Both entry points take `start_idx`, the first register; the storage occupies it and the
registers after it, exactly.

Placement is a register allocation hint, not a guarantee. The values of placed storage are lowered
to LLVM's strict register hints (`rocdl.sched.vgpr`, `rocdl.sched.sgpr`
and `rocdl.sched.agpr`): the register allocator considers only the requested registers for that
value. Nothing checks the result. When the requested registers are taken, unusable, or the value
has to be split or spilled, LLVM places the value or part of it elsewhere and compilation still
succeeds, in a few cases with wrong code (see [Limitations](#limitations)). Check the final ISA.

The registers are not reserved for the whole kernel: other values may use them before the first
write and after the last read. Because a fixed number sets a floor on the kernel's register count
(a value in `v200` makes the kernel use at least 201 VGPRs, which can lower occupancy), prefer low
numbers. The lowest registers hold kernel inputs such as the thread ID, the kernel argument
pointer, and the workgroup ID, so start fixed ranges above them.

## `fx_regalloc.RegisterAllocator` — the register memory allocator

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
sgprs = fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.SGPR32, start_idx=40)
vgprs = fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.VGPR32, start_idx=32)
block = sgprs.allocate(BlockRegisters)  # s40-s41
lane = vgprs.allocate(LaneRegisters)    # v32-v34

block.base.poke(fx.Int32(fx.block_idx.x) * 64)
lane.x.poke(X[block.base.peek() + fx.Int32(fx.thread_idx.x)])
```

Fields are navigated and accessed exactly as for shared memory. What differs is the memory behind
them:

|                   | `SharedAllocator`             | `RegisterAllocator`                                                                        |
| ----------------- | ----------------------------- | ------------------------------------------------------------------------------------------ |
| Memory            | shared by the workgroup       | private to each thread, in `register_class`                                                |
| Each `allocate`   | a region of one shared space  | independent register memory; no `base_ptr`, no `free`                                      |
| Padding           | to the type's alignment       | also to whole registers: an `Int8` takes one 32-bit register, two `Int16` fields share one |
| `allocated_bytes` | the bytes the kernel reserves | the logical layout and padding, not the kernel's register usage                            |
| Per kernel        | one                           | any number; allocators do not coordinate their ranges                                      |

Sizes and alignments stay in bytes, as for shared memory: an allocation is aligned to the larger of
`dsl_align_of(T)` and `allocate(..., alignment=...)`. The same alignment constrains register
numbers: an allocation aligned to 16 bytes starts at a multiple of 4 registers, so the allocator
rounds its next number up.

```python
regs = fx_regalloc.RegisterAllocator(fx_regalloc.rocdl.VGPR32, start_idx=65)
count = regs.allocate(fx.Int32)                               # v65
tile = regs.allocate(fx.Vector[fx.Float32, 4], alignment=16)  # v68-v71
```

A type's alignment describes its byte layout, not hardware register rules. On gfx90a and later,
64-bit and wider VGPR/AGPR operands start at even registers, yet `fx.Vector[fx.Float16, 4]` is only
2-byte aligned. Raise the alignment with `fx.Align[T, A]` or `alignment=` where an instruction
requires it; a misaligned start leaves the hint without a usable register (see
[Limitations](#limitations)).

## `fx_regalloc.register_place` — placing existing register memory

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
fx_regalloc.register_place(frag_A, fx_regalloc.rocdl.AGPR32, start_idx=0)  # a0-a3
fx_regalloc.register_place(frag_B, fx_regalloc.rocdl.AGPR32, start_idx=4)  # a4-a7
fx_regalloc.register_place(frag_C, fx_regalloc.rocdl.VGPR32, start_idx=64)  # v64-v79
```

`RegisterAllocator` is a layout convenience over `register_place`; the limitations below apply to
both.

## Limitations

Placement constrains where the register allocator puts values; it does not change which
instructions compute them, and LLVM does not report a hint it could not meet. The cases below were
observed on gfx942.

- **Conflicts move values silently.** If the requested registers are held by another live value,
  including another placed value, or register pressure forces a split or spill, the value or part
  of its live range ends up in other registers, with copies. Keeping an old `peek()` result alive
  across a `poke()` to the same storage is an easy way to cause this.
- **AGPR and VGPR numbers are not kept apart.** On gfx90a and later, a value that may live in either
  bank, such as a load result or an MFMA operand, can be given the VGPR with the requested number
  instead of the AGPR (an `AGPR32` request for 5 yields `v5`), and the reverse. A hint can also
  follow a copy into another class: an `SGPR32` number can end up on a VGPR copy of the value.
- **Placing part of a placed value.** Every read of placed storage is hinted, including a read of a
  single element, such as storing an MMA accumulator element by element. The element's hint can
  override the hint of the whole value, which then usually ends up elsewhere, with copies.
- **Unusable numbers produce wrong code.** An odd start for a 64-bit or wider VGPR or AGPR value on
  gfx90a and later, or a number beyond the register file, makes register allocation fail with
  "ran out of registers during register allocation". FlyDSL does not stop on this LLVM error: it is
  printed and the kernel is built with conflicting registers.
- **SGPRs hold uniform values only.** Placing a value that differs between lanes in SGPRs fails with
  "illegal VGPR to SGPR copy", with the same consequence: the error is printed and the code is
  wrong.
- **Constants are materialized.** A placed constant, such as an accumulator filled with zero, is
  computed into registers instead of being folded into the instructions that use it.
- **Whole registers only.** Each read or write of placed storage covers whole 32-bit registers;
  for example, a single `Float16` element of an MMA fragment is rejected at compile time. Storage
  from `RegisterAllocator` is byte-backed and is placed through the registers covering each access,
  so its sub-word fields can be accessed individually.
- **Static access only.** Register memory is accessed at offsets known while tracing; dynamic
  indexing is not supported. Values that are optimized away occupy nothing. Do not rely on
  uninitialized contents or padding.
- **Supported configurations.** AMDGPU targets with optimized code generation; at `O0` the hints are
  ignored. External LLVM code generation (`FLYDSL_COMPILE_LLVM_DIR`) needs an LLVM built with
  FlyDSL's `schedule-bank-and-gpr.patch`.
