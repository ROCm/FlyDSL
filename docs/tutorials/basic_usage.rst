Basic Usage
===========

This tutorial covers the fundamentals of using FlyDSL to write and run GPU
kernels.

Setting Up the Environment
--------------------------

After :doc:`installing FlyDSL <../installation>`, ensure the Python path is
configured:

.. code-block:: bash

   export PYTHONPATH=$(pwd)/build-fly/python_packages:$(pwd):$PYTHONPATH

Understanding Layouts
---------------------

Layouts are the core abstraction in FlyDSL. A layout maps logical coordinates
to physical memory indices using a ``(Shape, Stride)`` pair:

.. code-block:: python

   import flydsl.expr as fx

   # Create a 2D layout: 8 rows x 16 columns, column-major
   layout = fx.make_layout((8, 16), (1, 8))

   # Index = dot(Coord, Stride) = i*1 + j*8

Layout operations include:

- **fx.size(layout)** -- total number of elements
- **fx.rank(layout)** -- number of dimensions
- **fx.crd2idx(coord, layout)** -- coordinate to linear index
- **fx.idx2crd(index, layout)** -- linear index to coordinate

Defining a Kernel
-----------------

Kernels are defined using the ``@flyc.kernel`` decorator:

.. code-block:: python

   import flydsl.compiler as flyc
   import flydsl.expr as fx

   @flyc.kernel
   def my_kernel(
       data: fx.Tensor,
       n: fx.Constexpr[int],
   ):
       tid = fx.thread_idx.x
       bid = fx.block_idx.x
       # ... kernel body using layout ops

Key points:

- ``@flyc.kernel`` compiles the function body into GPU IR via AST rewriting
- ``fx.Tensor`` denotes a GPU tensor argument
- ``fx.Constexpr[int]`` denotes a compile-time constant (affects cache key)
- GPU intrinsics are accessed via ``fx.thread_idx``, ``fx.block_idx``, etc.

Launching Kernels
------------------

Kernels are launched via ``@flyc.jit`` host-side functions:

.. code-block:: python

   @flyc.jit
   def launch(
       data: fx.Tensor,
       n: fx.Constexpr[int],
       stream: fx.Stream = fx.Stream(None),
   ):
       my_kernel(data, n).launch(
           grid=(grid_x, 1, 1),
           block=(256, 1, 1),
           stream=stream,
       )

   # Usage with PyTorch
   import torch
   data = torch.randn(1024, dtype=torch.float32).cuda()
   launch(data, 1024, stream=torch.cuda.Stream())
   torch.cuda.synchronize()

Next Steps
----------

- :doc:`kernel_development` -- advanced kernel techniques
- :doc:`../layout_system_guide` -- deep dive into the layout system
- :doc:`../kernel_authoring_guide` -- comprehensive kernel authoring reference
