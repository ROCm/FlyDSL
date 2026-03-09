Compiler & Pipeline
===================

FlyDSL includes a composable MLIR compiler pipeline for lowering kernel IR to
GPU binaries.

Pipeline
--------

The ``Pipeline`` class provides a fluent API for constructing MLIR pass pipelines:

.. code-block:: python

   from flydsl.compiler.pipeline import Pipeline

   pipeline = (
       Pipeline()
       .flir_to_standard()
       .canonicalize()
       .cse()
       .rocdl_attach_target(chip="gfx942")
       .Gpu(Pipeline().convert_gpu_to_rocdl(runtime="HIP"))
       .gpu_to_llvm()
       .lower_to_llvm()
       .gpu_module_to_binary(format="bin")
   )

Available pipeline passes include:

- **flir_to_standard** -- lower FLIR dialect ops to standard MLIR
- **canonicalize** -- standard canonicalization
- **cse** -- common subexpression elimination
- **rocdl_attach_target** -- attach ROCm device target (chip, features)
- **convert_gpu_to_rocdl** -- lower GPU dialect to ROCDL
- **gpu_to_llvm** -- lower GPU host-side ops to LLVM
- **lower_to_llvm** -- lower remaining ops to LLVM dialect
- **gpu_module_to_binary** -- compile GPU module to device binary

Compiler
--------

The ``compile`` function JIT-compiles an MLIR module into an executable:

.. code-block:: python

   import flydsl

   exe = flydsl.compile(module)
   exe(arg1, arg2, ...)

Executor
--------

The ``Executor`` class provides lower-level control over kernel execution,
including launch configuration and argument marshalling.

RAII Context
------------

``RAIIMLIRContextModule`` manages the MLIR context lifetime, ensuring proper
initialization and cleanup of dialects and passes.

Buffer Operations
-----------------

The ``flydsl.dialects.ext.buffer_ops`` module provides high-level Python wrappers
for AMD CDNA3/CDNA4 buffer load/store operations. Buffer operations use a scalar
base pointer (SGPRs) and per-thread offsets for efficient global memory access
with hardware bounds checking.

Key functions:

- **create_buffer_resource** -- create an AMD buffer resource descriptor from a memref
- **buffer_load** / **buffer_store** -- vectorized buffer load/store with optional masking
- **buffer_load_2d** / **buffer_store_2d** -- 2D buffer access with automatic row/col offset calculation
- **BufferResourceDescriptor** -- class wrapping the ROCDL buffer resource descriptor

.. code-block:: python

   from flydsl.dialects.ext import buffer_ops

   rsrc = buffer_ops.create_buffer_resource(A)
   data = buffer_ops.buffer_load(rsrc, offset, vec_width=4)
   buffer_ops.buffer_store(result, rsrc, offset)

Helper utilities: ``index_cast_to_i32``, ``i32_mul``, ``i32_add``, ``i32_select``.

flir-opt CLI
------------

The ``flir-opt`` tool is a command-line interface for running MLIR passes on
``.mlir`` files:

.. code-block:: bash

   flir-opt --flir-to-standard input.mlir
   flir-opt --help
