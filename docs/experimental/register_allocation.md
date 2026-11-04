<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 FlyDSL Project Contributors -->

# Register allocation (experimental)

`flydsl.experimental.register_allocation` places per-thread register memory at chosen numbers of a
register class: the A/B operands of an MFMA in given AGPRs, an accumulator in a given VGPR range, a
few uniform values in fixed SGPRs.

The feature is experimental and may change or be removed without a deprecation period. Its names
are not flattened into the stable `fx` namespace: like the extension libraries under `fx.coop` and
`fx.random`, it is a subpackage, reached as `fx.experimental.register_allocation`, so every use is
visibly experimental. Use `fx.experimental.register_allocation.<name>` directly, or bind the
subpackage to a short name, as the rest of this page does:

```python
import flydsl.expr as fx

fx_regalloc = fx.experimental.register_allocation
# or, equivalently:
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

A placement holds while the value is live. The registers are not reserved for the whole kernel:
other values may use them before the first write and after the last read. Because a fixed number
sets a floor on the kernel's register count (a value in `v200` makes the kernel use at least 201
VGPRs, which can lower occupancy), prefer low numbers. The lowest registers hold kernel inputs such
as the thread ID, the kernel argument pointer, and the workgroup ID, so start fixed ranges above
them.

Placement is a requirement, not a hint. A request that the generated code cannot honor fails
compilation with an error that points at the DSL source; it is never silently dropped.

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
requires it. A misaligned value is rejected where it is written and copied where it is read.

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

# MFMA reads A/B from AGPRs and accumulates C in VGPRs, so no copies are needed.
fx_regalloc.register_place(frag_A, fx_regalloc.rocdl.AGPR32, start_idx=0)  # a0-a3
fx_regalloc.register_place(frag_B, fx_regalloc.rocdl.AGPR32, start_idx=4)  # a4-a7
fx_regalloc.register_place(frag_C, fx_regalloc.rocdl.VGPR32, start_idx=64)  # v64-v79
```

`RegisterAllocator` is a layout convenience over `register_place`; the limitations below apply to
both.

## Limitations

Placement constrains where values live; it does not change which instructions compute them. A
request succeeds only if the instructions that compute and use the value can work in the requested
registers directly.

- **The writer must produce the value in place.** The instruction that writes a placed value must
  write exactly its registers, in its class; there is no write-back copy. An ordinary VGPR
  arithmetic result cannot be placed in AGPRs, and an instruction with a wider result is rejected:
  an integer multiply-add, for example, produces a 64-bit result (`v_mad_u64_u32`) and cannot fill
  one fixed 32-bit register.
- **Readers may copy.** Instructions whose results are not placed can read a placed value through a
  copy, for example to convert an AGPR `Float32` to `Float16` through a VGPR. The copy costs
  instructions and registers; check the final ISA when that matters.
- **SGPRs hold uniform values only.** A value that differs between lanes cannot be placed in SGPRs,
  and a uniform value placed in VGPRs or AGPRs cannot later feed an instruction that needs it in an
  SGPR. Conversely, the compiler may compute a uniform value with scalar instructions, such as a
  load from a uniform address, and such a value cannot then be fixed in VGPRs.
- **MFMA accumulators in AGPRs.** On gfx90a and later, AGPR placement of an MFMA accumulator is not
  supported yet.
- **Placed values must reach real instructions.** If the compiler forwards a placed value straight to
  its readers so that no instruction uses the placed registers, the request is rejected. Sub-word
  fields written and read one at a time are the common case: two `Float16` fields, four `Int8`
  fields, or a lone `Float16` in the upper half of a register. A single narrow value at the start of
  its register, such as one `Float16` used in arithmetic, works, as does writing or reading the
  covering word through a union or an `fx.Vector`. Fixed values inside a loop with a uniform trip
  count can be rejected for the same reason when the loop is unrolled.
- **Live fixed values must not overlap.** Two fixed values that are live at the same time in
  overlapping registers are rejected. Keeping an old `peek()` result alive across a `poke()` to the
  same storage is an easy way to cause this.
- **Placed values are not spilled.** If register pressure would require spilling a placed value,
  compilation fails.
- **Some registers are off limits.** Fixed numbers may not include registers that instructions use
  implicitly, such as `VCC` (`SGPR32` members 106 and 107).
- **Static access only.** Register memory is accessed at offsets known while tracing; dynamic
  indexing is not supported. Constants that the compiler folds into instructions are not forced
  into registers, and values that are optimized away occupy nothing. Do not rely on uninitialized
  contents or padding.
- **Supported configurations.** AMDGPU targets with the default optimized code generation. External
  LLVM code generation (`FLYDSL_COMPILE_LLVM_DIR`) rejects kernels that use placement.
