Basic Usage
===========

This tutorial covers the fundamentals of using FlyDSL to write and run GPU
kernels.

Setting Up the Environment
--------------------------

After :doc:`installing FlyDSL <../installation>`, ensure the Python path is
configured:

.. code-block:: bash

   export PYTHONPATH=$(pwd)/flydsl/src:$(pwd)/.flir/build/python_packages/flydsl:$PYTHONPATH

Understanding Layouts
---------------------

Layouts are the core abstraction in FlyDSL. A layout maps logical coordinates
to physical memory indices using a ``(Shape, Stride)`` pair:

.. code-block:: python

   from flydsl.dialects.ext import flir

   # Create a 2D layout: 8 rows x 16 columns, column-major
   shape = flir.make_shape(8, 16)
   stride = flir.make_stride(1, 8)
   layout = flir.make_layout(shape, stride)

   # Index = dot(Coord, Stride) = i*1 + j*8

Layout operations include:

- **size(layout)** -- total number of elements
- **rank(layout)** -- number of dimensions
- **crd2idx(coord, layout)** -- coordinate to linear index
- **idx2crd(index, layout)** -- linear index to coordinate

Defining a Kernel
-----------------

Kernels are defined as methods on an ``MlirModule`` subclass:

.. code-block:: python

   from flydsl.dialects.ext import flir
   import _mlir.extras.types as T

   class MyKernel(flir.MlirModule):
       GPU_MODULE_NAME = "my_module"
       GPU_MODULE_TARGETS = ['#rocdl.target<chip = "gfx942">']

       @flir.kernel
       def my_kernel(self: flir.T.i64,
                     data: lambda: T.memref(T.dynamic(), T.f32()),
                     n: lambda: T.index()):
           tid = flir.thread_idx("x")
           bid = flir.block_idx("x")
           # ... kernel body

Key points:

- ``GPU_MODULE_TARGETS`` specifies the target GPU architecture
- ``@flir.kernel`` compiles the method body into GPU IR
- Parameters use lambda type annotations for MLIR type construction

Compiling and Running
---------------------

.. code-block:: python

   import flydsl

   # Build the MLIR module
   module = MyKernel().module

   # JIT compile to GPU binary
   exe = flydsl.compile(module)

   # Launch on GPU
   exe(data_tensor, n)

Next Steps
----------

- :doc:`kernel_development` -- advanced kernel techniques
- :doc:`../layout_system_guide` -- deep dive into the layout system
- :doc:`../kernel_authoring_guide` -- comprehensive kernel authoring reference
