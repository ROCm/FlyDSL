<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 FlyDSL Project Contributors -->

# DSL Protocols

FlyDSL uses three structural protocols at compilation and storage boundaries:

- `JitArgument` describes how a Python value enters a compiled JIT function.
- `DslType` converts DSL values to and from MLIR SSA values.
- `Storable` defines a DSL type's byte layout and typed memory access.

They are capabilities, not base classes. A type implements a protocol by providing its hooks, and
may implement any combination of the three.

This page connects Python arguments, DSL values, and typed memory. The detailed rules for
[composites](composite_types.md), and [storage and allocation](storage_and_allocator.md) live in
their respective pages.

## `JitArgument`

| Hook | Meaning |
|---|---|
| `value.__get_ir_types__()` | MLIR argument types contributed to the function signature |
| `value.__cache_signature__()` | stable, hashable argument contribution to the compilation cache key |
| `value.__c_abi_spec__()` | ordered `(ctype, fill)` entries for the packed C interface |

The IR types and final ABI slots must describe the same arguments in corresponding order, and that
order must also agree with `DslType` reconstruction. Cache signatures should include
compilation-relevant metadata such as type, dtype, rank, address space, alignment, and static
layout, but exclude run-time values and addresses.

Each C-ABI `fill(argument, storage)` callback writes the current Python argument into preallocated
`ctypes` storage. It must read each call's value from `argument`, rather than capture that value
when the specification is built. `__c_abi_spec__()` must match the final ABI produced after the
types from `__get_ir_types__()` pass through MLIR lowering and `llvm.emit_c_interface`. This is not
necessarily a one-to-one mapping: one MLIR argument may lower to multiple packed ABI slots.
Implementing this hook therefore requires an understanding of the relevant MLIR-to-LLVM lowering and
the generated C interface; a mismatch in slot type or order passes incorrect data to the compiled
function.

`JitArgumentRegistry.register(py_type, dsl_type=...)` registers an adapter for a raw Python type.
The adapter defines the host contract; `dsl_type` defines the value visible while tracing. A value
implementing both `JitArgument` and `DslType` can be used directly without an adapter.


## `DslType`

| Hook | Meaning |
|---|---|
| `value.__extract_to_ir_values__()` | convert a DSL value into ordered `ir.Value`s |
| `Type.__construct_from_ir_values__(values, exemplar=None)` | rebuild the DSL value from those SSA values |

The two hooks form an ordered round trip. Reconstruction accepts replacement SSA values with the
same count, order, and types as extraction produces, such as a new function's block arguments.
`exemplar` carries metadata not encoded by the MLIR type; for example, `Vector` uses it to retain
its logical shape and element type.

A zero-length result is valid and is used by `Constexpr` values with no run-time representation.

## `Storable`

| Hook | Meaning |
|---|---|
| `Type.__dsl_size_of__()` | static byte size |
| `Type.__dsl_align_of__()` | required byte alignment |
| `Type.__peek_from_ptr__(ptr)` | materialize a typed value from an address |
| `Type.__poke_into_ptr__(ptr, value)` | write a typed value to an address |

The free functions `dsl_size_of`, `dsl_align_of`, `peek_from_ptr`, and `poke_into_ptr` dispatch to
these hooks. `Storage[T]` and the allocators use the same protocol; see [Storage and
Allocator](storage_and_allocator.md).


## Python-to-MLIR boundary

A Python argument and its corresponding DSL value may have different types. For example, a host
tensor is represented by a `JitArgument` adapter at the Python boundary and by an `fx.Tensor`
argument inside the JIT function.

The protocols connect these representations:

- `JitArgumentRegistry` associates a Python type with its host adapter and destination DSL type. A
  value implementing both `JitArgument` and `DslType` can be used directly.
- `__get_ir_types__()` supplies the ordered MLIR parameter types.
  `DslType.__construct_from_ir_values__()` rebuilds the corresponding DSL value from the function's
  block arguments, using the adapter as an exemplar when metadata is needed.
- The type sequence returned by `__get_ir_types__()` must match the SSA value types expected by the
  corresponding `DslType`, in count, order, and MLIR type.
- At execution, `__c_abi_spec__()` supplies the ABI slots and callbacks that fill them from the
  current Python argument. These slots must match the lowered C interface of those MLIR parameters.

```text
Python calls @jit
    │
    └─ adapt ─────> JitArgument
                      │
          compilation │ __cache_signature__ -> cache key
                      │ __get_ir_types__     -> JIT MLIR signature
                      │
                      v
              JIT function arguments
                      │ DslType.__construct_from_ir_values__
                      v
              DSL values in JIT body
                      │ call @kernel: trace its body
                      ├─ get_ir_types --------------------> gpu.func signature
                      ├─ __construct_from_ir_values__ ----> kernel DSL values
                      └─ .launch(): __extract_to_ir_values__ -> gpu.launch_func operands

             execution │ __c_abi_spec__ / fill
                       v
               packed C ABI -> lowered JIT function
```

MLIR lowering compiles the generated host `func.func` and device `gpu.func` into executable code. A
cache hit reuses the compiled artifact without tracing again; `fill` still reads the current
arguments on every call. Parameters annotated `Constexpr[T]` are supplied directly to tracing and
affect specialization, but contribute no run-time MLIR parameters or C-ABI slots.

`JitArgument` is used only at the Python-to-MLIR boundary. The jit-to-kernel boundary is already
inside MLIR construction and therefore uses `DslType`, not the host C ABI.


## Protocol support

| Representative type | `DslType` | `JitArgument` | `Storable` |
|---|:---:|:---:|:---:|
| supported scalar `Numeric`, such as `fx.Int32` | yes | yes | yes |
| `fx.Vector`, `fx.Pointer`, traced `fx.Tensor` | yes | no | no |
| `TorchTensorJitArg`, `DLTensorJitArg` | no | yes | no |
| `fx.Stream` | yes | yes | no |
| `fx.Array[E, N]` | yes | no | yes |
| `@fx.struct` | recursive | recursive | recursive |
| `@fx.union` | no | no | recursive |
| `fx.Constexpr[T]` | zero SSA values | special JIT handling | no |

For a struct, each protocol is supported when all non-`Constexpr` fields support it. The fields are
processed recursively in declaration order.

The table describes usable capabilities. All three protocols are `@runtime_checkable`, but
structural `isinstance` and `issubclass` checks confirm only that the hooks exist, including hooks
that raise for unsupported operations. Implementations remain responsible for correct ordering, MLIR
types, cache keys, and byte layout.
