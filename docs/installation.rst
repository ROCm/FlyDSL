Install FlyDSL
==============

FlyDSL is a Python DSL and MLIR compiler stack for writing high-performance AMD
GPU kernels. This page explains how to install FlyDSL using pip or from source,
verify the installation, and resolve common issues.

Prerequisites
-------------

- **Python**: 3.10 or later
- **ROCm**: Required for GPU execution tests and benchmarks (IR-only tests do not need a GPU)
- **GPU targets**: See the architecture and validation scope below.
- **OS**: Linux; use a ROCm version supported by your GPU and the selected wheel.

For the latest ROCm installation instructions, see the
`ROCm installation guide <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_.

Architecture and validation scope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compiler support for an instruction family does not imply that every prebuilt
kernel supports every target, dtype, or shape. The following describes the
current source tree; consult the tests and CI results for the revision you use.

.. list-table:: Target-specific implementation and validation
   :header-rows: 1
   :widths: 16 38 46

   * - Target
     - Implementation
     - Validation scope
   * - ``gfx942``
     - CDNA3 MFMA and buffer-copy paths
     - MI325 runners are included in source and wheel CI. Kernel tests cover
       selected shapes and dtypes, not all possible configurations.
   * - ``gfx950``
     - CDNA4 MFMA, including target-specific FP4 paths
     - MI35x source runners and MI355 wheel runners exercise selected kernels.
       Check per-kernel restrictions before reusing a configuration.
   * - ``gfx11*``
     - RDNA WMMA paths
     - Architecture-gated RDNA GEMM tests exist. Test presence alone is not
       evidence of a passing wheel validation run on every RDNA device.
   * - ``gfx120*``
     - RDNA WMMA paths, with separate dtype/shape restrictions
     - Architecture-gated RDNA GEMM tests exist; source CI includes a Navi
       runner. The wheel CI matrix is narrower than the source target set.
   * - ``gfx1250``
     - CDNA5 wave32 WMMA, TDM, cluster/multicast, and 320-KB LDS paths
     - Dedicated GEMM, MoE, and atom tests exist in the source tree, but the
       checked-in GitHub Actions matrices do not currently provide a gfx1250
       hardware validation job. Treat those paths as target-specific and
       verify them on the intended system.

For other target-specific APIs, see :doc:`api/compiler` and the corresponding
kernel tests. This table is not a blanket support guarantee. The authoritative
runner matrices and environments are in the
`source workflow <https://github.com/ROCm/FlyDSL/blob/main/.github/workflows/flydsl.yaml>`_
and `wheel workflow <https://github.com/ROCm/FlyDSL/blob/main/.github/workflows/test-whl.yaml>`_;
inspect their `run results <https://github.com/ROCm/FlyDSL/actions>`_ for validation
of a particular commit. See also
`RDNA GEMM tests <https://github.com/ROCm/FlyDSL/blob/main/tests/kernels/test_rdna_gemm.py>`_.

Install from PyPI
-----------------

For standalone use, install the published package directly:

.. code-block:: bash

   python -m pip install flydsl

Verify that Python can import FlyDSL:

.. code-block:: bash

   python -c "import flydsl; print('FlyDSL version', flydsl.__version__)"

.. _documentation-versions:

Documentation and integration versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GitHub Pages documentation tracks ``main``. Its version comes from the
source tree and can be ahead of the package available from PyPI or a GitHub
Release. Use the `release list <https://github.com/ROCm/FlyDSL/releases>`_ and
the ``docs/`` directory at the corresponding tag when working with a release;
APIs described on the main site may not be available in an older wheel.

When FlyDSL is a dependency of another project, follow that project's
installation instructions and dependency constraints instead of upgrading
FlyDSL independently:

.. list-table:: Integration compatibility sources
   :header-rows: 1
   :widths: 20 80

   * - Project
     - How to select a version
   * - AITER
     - Use the FlyDSL pin in the ``requirements.txt`` of the AITER revision
       being installed. See `AITER dependencies <https://github.com/ROCm/aiter/blob/main/requirements.txt>`_.
   * - MORI
     - Follow the selected MORI release's optional FlyDSL dependency and
       device-API instructions. See `MORI package metadata <https://github.com/ROCm/mori/blob/main/pyproject.toml>`_
       and the `MORI installation guide <https://rocm.github.io/mori/installation.html>`_.

Dependency constraints describe installation compatibility; they do not prove
that every kernel or communication topology has been tested. Record the
FlyDSL, ROCm and consuming-project versions when reporting an integration issue.

Build from source
-----------------

Build from source only if you are developing FlyDSL itself or need a custom
MLIR/LLVM build.

Start from a checkout and run the commands below from its root:

.. code-block:: bash

   git clone https://github.com/ROCm/FlyDSL.git
   cd FlyDSL

Additional prerequisites for source builds:

- **Build tools**: ``cmake`` (>=3.20), a C++17 compiler, and optionally ``ninja``
- **Python deps**: ``nanobind``, ``numpy``, ``pybind11`` (installed automatically)

Step 1: Build LLVM/MLIR
~~~~~~~~~~~~~~~~~~~~~~~

If you already have an MLIR build with Python bindings enabled, point to it:

.. code-block:: bash

   export MLIR_PATH=/path/to/llvm-project/build-flydsl/mlir_install

Otherwise, use the helper script that clones the ROCm llvm-project and builds MLIR:

.. code-block:: bash

   bash scripts/build_llvm.sh -j64
   export MLIR_PATH=/path/to/llvm-project/build-flydsl/mlir_install

Step 2: Build FlyDSL
~~~~~~~~~~~~~~~~~~~~

Build the Fly C++ dialect, compiler passes, and embedded Python bindings:

.. code-block:: bash

   bash scripts/build.sh -j64

``build.sh`` auto-detects ``MLIR_PATH`` from common locations. Override with:

.. code-block:: bash

   MLIR_PATH=/path/to/mlir_install bash scripts/build.sh -j64

After a successful build, you have:

- ``build-fly/bin/fly-opt`` -- the Fly optimization tool
- ``build-fly/bin/flydsl-lsp-server`` -- MLIR Language Server for FlyDSL ``.mlir``
- ``build-fly/python_packages/flydsl/`` -- Python package root containing:

  - ``flydsl/`` -- Python DSL API (sources from ``python/flydsl/``)
  - ``_mlir/`` -- embedded MLIR Python bindings (no external ``mlir`` wheel required)

Step 3: Install FlyDSL
~~~~~~~~~~~~~~~~~~~~~~

For development, use an editable install:

.. code-block:: bash

   python -m pip install -e .

This creates an editable install — changes to ``python/flydsl/`` are immediately reflected.

**Without installing**, you can also set paths manually:

.. code-block:: bash

   export PYTHONPATH=$(pwd)/build-fly/python_packages:$(pwd):$PYTHONPATH
   export LD_LIBRARY_PATH=$(pwd)/build-fly/python_packages/flydsl/_mlir/_mlir_libs:$LD_LIBRARY_PATH

To build a distributable wheel after the native build is available:

.. code-block:: bash

   python -m pip install build
   python -m build --wheel --no-isolation
   ls dist/

Verify installation
-------------------

Run the test suite to verify that everything works:

.. code-block:: bash

   bash scripts/run_tests.sh

This runs the following:

- **Pytest suites**: ``tests/kernels/``, ``tests/language/``, ``tests/unit/``,
  ``tests/system/``, ``tests/extension/``, and ``tests/python/examples/``
- **Standalone examples**: architecture-compatible scripts under ``examples/``
- **MLIR/FileCheck tests**: all ``.mlir`` files under ``tests/mlir/``

The broad runner requires a GPU for its device-tier tests. For test tiers,
focused commands, and compile-only coverage, see :doc:`testing_benchmarking_guide`.

Troubleshooting
---------------

**fly-opt not found**
   Run ``bash scripts/build.sh``, or build explicitly::

      cmake --build build-fly --target fly-opt -j$(nproc)

**Python import issues (No module named flydsl)**
   Install the published package, or use the source checkout after building::

      pip install flydsl
      pip install -e .

**MLIR .so load errors**
   For a source build, add the embedded package library directory to the loader
   path::

      export LD_LIBRARY_PATH=$(pwd)/build-fly/python_packages/flydsl/_mlir/_mlir_libs:$LD_LIBRARY_PATH

   If an external MLIR install is also required by a development tool, append
   ``$MLIR_PATH/lib`` rather than replacing the embedded directory.
