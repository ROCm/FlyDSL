Installation
============

Prerequisites
-------------

- **Python**: 3.10 or later
- **ROCm**: Required for GPU execution tests and benchmarks (IR-only tests do not need a GPU)
- **Build tools**: ``cmake``, a C++ compiler, and optionally ``ninja``
- **Supported GPUs**: AMD MI300X/MI308X (gfx942), AMD MI350 (gfx950)
- **Supported OS**: Linux with ROCm 6.x or 7.x

Step 1: Build MLIR
-------------------

If you already have an MLIR build, point to it:

.. code-block:: bash

   export MLIR_PATH=/path/to/llvm-project/build

Otherwise, use the helper script which clones the ROCm llvm-project and builds MLIR:

.. code-block:: bash

   bash scripts/build_llvm.sh
   export MLIR_PATH=/path/to/llvm-project/mlir_install

Step 2: Build FLIR
-------------------

Build the FLIR C++ dialect, compiler passes, and embedded Python bindings:

.. code-block:: bash

   ./flir/build.sh

After a successful build you will have:

- ``.flir/build/bin/flir-opt`` -- the FLIR optimization tool (legacy ``build/bin/flir-opt`` also works)
- ``.flir/build/python_packages/flydsl/`` -- Python package root containing:

  - ``flydsl/`` -- Python API
  - ``_mlir/`` -- embedded MLIR Python bindings (no external ``mlir`` wheel required)

Step 3: Install FlyDSL
-----------------------

For development (editable install):

.. code-block:: bash

   pip install -e .

Or using setup.py directly:

.. code-block:: bash

   python setup.py develop

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

- **MLIR file tests**: ``tests/mlir/*.mlir`` through ``flir-opt --flir-to-standard``
- **Python IR tests**: ``tests/pyir/test_*.py`` (no GPU required)
- **Kernel/GPU execution tests** (only if ROCm is detected): ``tests/kernels/test_*.py``

Troubleshooting
---------------

**flir-opt not found**
   Run ``./flir/build.sh``, or build explicitly::

      cmake --build build --target flir-opt -j$(nproc)

**Python import issues (No module named flydsl / No module named mlir)**
   Ensure you are using the embedded package::

      export PYTHONPATH=$(pwd)/build/python_packages/flydsl:$PYTHONPATH

   Or prefer in-tree sources::

      export PYTHONPATH=$(pwd)/flydsl/src:$(pwd)/.flir/build/python_packages/flydsl:$PYTHONPATH

**MLIR .so load errors**
   Add the MLIR build lib dir to the loader path::

      export LD_LIBRARY_PATH=$MLIR_PATH/lib:$LD_LIBRARY_PATH
