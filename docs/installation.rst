Installation
============

Prerequisites
-------------

- **Python**: 3.10 or later
- **ROCm**: Required for GPU execution tests and benchmarks (IR-only tests do not need a GPU)
- **Build tools**: ``cmake`` (≥3.20), a C++17 compiler, and optionally ``ninja``
- **Python deps**: ``nanobind``, ``numpy``, ``pybind11`` (installed automatically)
- **Supported GPUs**: AMD MI300X/MI308X (gfx942), AMD MI350 (gfx950)
- **Supported OS**: Linux with ROCm 6.x or 7.x

Step 1: Build LLVM/MLIR
-------------------------

If you already have an MLIR build with Python bindings enabled, point to it:

.. code-block:: bash

   export MLIR_PATH=/path/to/llvm-project/build-flydsl/mlir_install

Otherwise, use the helper script which clones the ROCm llvm-project and builds MLIR:

.. code-block:: bash

   bash scripts/build_llvm.sh -j64
   export MLIR_PATH=/path/to/llvm-project/build-flydsl/mlir_install

Step 2: Build FlyDSL
---------------------

Build the Fly C++ dialect, compiler passes, and embedded Python bindings:

.. code-block:: bash

   bash scripts/build.sh -j64

``build.sh`` auto-detects ``MLIR_PATH`` from common locations. Override with:

.. code-block:: bash

   MLIR_PATH=/path/to/mlir_install bash scripts/build.sh -j64

After a successful build you will have:

- ``build-fly/bin/fly-opt`` -- the Fly optimization tool
- ``build-fly/python_packages/flydsl/`` -- Python package root containing:

  - ``flydsl/`` -- Python DSL API (sources from ``python/flydsl/``)
  - ``_mlir/`` -- embedded MLIR Python bindings (no external ``mlir`` wheel required)

Step 3: Install FlyDSL
-----------------------

For development (editable install):

.. code-block:: bash

   pip install -e .

Or using setup.py directly:

.. code-block:: bash

   python setup.py develop

This creates an editable install — changes to ``python/flydsl/`` are immediately
reflected.

**Without installing**, you can also set paths manually:

.. code-block:: bash

   export PYTHONPATH=$(pwd)/build-fly/python_packages:$(pwd):$PYTHONPATH
   export LD_LIBRARY_PATH=$(pwd)/build-fly/python_packages/flydsl/_mlir/_mlir_libs:$LD_LIBRARY_PATH

To build a distributable wheel:

.. code-block:: bash

   python setup.py bdist_wheel
   ls dist/

Step 4: Verify Installation
----------------------------

Run the test suite to verify everything works:

.. code-block:: bash

   bash scripts/run_tests.sh

This runs:

- **MLIR lit tests**: ``tests/mlir/{LayoutAlgebra,Conversion,Transforms}/*.mlir``
  through ``fly-opt``
- **Python IR tests**: ``tests/pyir/test_*.py`` (no GPU required)
- **Kernel/GPU execution tests** (only if ROCm is detected): ``tests/kernels/test_*.py``

Troubleshooting
---------------

**fly-opt not found**
   Run ``bash scripts/build.sh``, or build explicitly::

      cmake --build build-fly --target fly-opt -j$(nproc)

**Python import issues (No module named flydsl)**
   Ensure you are using the embedded package::

      export PYTHONPATH=$(pwd)/build-fly/python_packages:$(pwd):$PYTHONPATH

**MLIR .so load errors**
   Add the MLIR build lib dir to the loader path::

      export LD_LIBRARY_PATH=$MLIR_PATH/lib:$LD_LIBRARY_PATH
