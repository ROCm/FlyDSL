# Adding a New Fly Target Stack

This document is the maintainer guide for extending FlyDSL with an additional value of **`FLYDSL_TARGET_STACK`**. Today the repository allows only **`rocdl`** (ROCm / HIP / FlyROCDL). Use the checklists below when introducing a second stack (for example, another GPU vendor or a host-only toolchain).

## Scope and assumptions

- **One build, one stack.** The produced binaries and Python package must contain only the dialects, conversions, CAPI libraries, Python bindings, JIT runtime, and registration logic for the selected stack.
- **Single CMake selector.** End users choose the stack with **`FLYDSL_TARGET_STACK`**. Do not add parallel per-stack **`ENABLE_*`** cache options; encode stack-specific behavior in the descriptor row for that stack id.
- **Consistency.** The active row in **`cmake/FlyDSLTargetStack.cmake`**, the strings substituted into **`python/mlir_flydsl/FlyRegisterEverything.cpp.in`**, and the allow-lists emitted into **`flydsl._build_config`** (**`FLYDSL_BUILD_CONFIG_*`**) must describe the **same** compile backends and device runtime kinds.

## Design constraints

### Registry safety (`_mlirRegisterEverything`)

Pass registration for the Python extension **`_mlirRegisterEverything`** must go through **CAPI entry points** linked via **`EMBED_CAPI_LINK_LIBS`** only.

Do **not** link C++ pass libraries (for example **`MLIRFlyToROCDL`**) into **`PRIVATE_LINK_LIBS`** for that extension. Doing so embeds a separate **`MLIRPass`** registry instance; passes then register in the wrong registry and **`PassManager.parse()`** fails at runtime.

### Generated sources

Do not hand-edit build-tree outputs:

| Generated file | Source |
|----------------|--------|
| **`FlyRegisterEverything.cpp`** | **`configure_file`** from **`python/mlir_flydsl/FlyRegisterEverything.cpp.in`**, driven by **`FLYDSL_REGISTER_*`** variables in **`cmake/FlyDSLTargetStack.cmake`**. |
| **`flydsl/_build_config.py`** | **`configure_file`** from **`cmake/FlyDSLBuildConfig.py.in`**, then copied into **`python_packages/flydsl/`** during the **`CopyFlyPythonSources`** custom target. |

---

## 1. Central descriptor: `cmake/FlyDSLTargetStack.cmake`

| Step | Action |
|------|--------|
| 1 | Append the new stack id to **`_FLYDSL_TARGET_STACK_ALLOWED`** and to **`set_property(CACHE FLYDSL_TARGET_STACK PROPERTY STRINGS ...)`** so the value appears in CMake GUI tools. |
| 2 | Replace the single **`if(FLYDSL_TARGET_STACK STREQUAL "rocdl")`** block with **`if` / `elseif` / `endif()`** (or factor shared variables) and add a branch for the new stack. |
| 3 | In that branch, set **descriptor variables** used elsewhere: **`FLYDSL_STACK_MLIR_TARGET_DIALECT_NAME`** (segment in **`mlirGetDialectHandle__<name>__()`**), **`FLYDSL_STACK_CPP_TARGET_TOKEN`** (naming pattern for **`MLIRFly${TOKEN}Dialect`**, CAPI targets, and so on), **`FLYDSL_STACK_UPSTREAM_MLIR_PY_DIALECT`** (suffix for **`MLIRPythonSources.Dialects.<name>`**). |
| 4 | Set **`FLYDSL_REGISTER_EXTRA_INCLUDES`**, **`FLYDSL_REGISTER_TARGET_DIALECT_HANDLES`**, and **`FLYDSL_REGISTER_TARGET_PASSES`** so the generated **`FlyRegisterEverything.cpp`** registers the correct dialect handles and CAPI pass registration functions. Preserve C++ indentation inside bracket strings. |
| 5 | Set **`FLYDSL_REGISTER_EMBED_CAPI_LINK_LIBS`** to the **CAPI** targets for the Fly dialect, the stack-specific CAPI library, and the upstream MLIR CAPI set. Omit non-CAPI pass libraries. |
| 6 | Set **`FLYDSL_BUILD_CONFIG_BACKEND_IDS`** and **`FLYDSL_BUILD_CONFIG_RUNTIME_KINDS`** to fragments that expand to valid Python tuple elements inside **`cmake/FlyDSLBuildConfig.py.in`** (for example **`"\"rocm\", "`** for a single id). |
| 7 | **Stack-wide CMake flag.** If the new stack shares the same HIP-based JIT layout as ROCm, you may keep using **`FLYDSL_BUILD_ROCDL_STACK`**. If not (no HIP, different runtime), introduce **`FLYDSL_BUILD_<STACK>_STACK`** (or a unified flag) and update **every** **`if(FLYDSL_BUILD_ROCDL_STACK)`** in CMake and every related preprocessor macro in C++ consistently. |

The header comment in **`cmake/FlyDSLTargetStack.cmake`** may point reviewers to this file (**`ADDING_TARGET_STACK.md`** at the repository root).

---

## 2. C++ and TableGen layout: `include/flydsl` and `lib`

### Parent `CMakeLists.txt` files to gate

Wrap stack-specific **`add_subdirectory(...)`** calls in the **same** condition as in **`FlyDSLTargetStack.cmake`** (today **`if(FLYDSL_BUILD_ROCDL_STACK)`**).

| File | Purpose |
|------|---------|
| **`include/flydsl/Dialect/CMakeLists.txt`** | Target dialect TableGen and headers (for example **`FlyROCDL`**). |
| **`include/flydsl/Conversion/CMakeLists.txt`** | Target lowering TableGen (for example **`FlyToROCDL`**). |
| **`lib/Dialect/CMakeLists.txt`** | Target dialect implementation libraries. |
| **`lib/Conversion/CMakeLists.txt`** | Target conversion pass libraries. |
| **`lib/CAPI/Dialect/CMakeLists.txt`** | Target stack CAPI wrapper subdirectory. |
| **`lib/CMakeLists.txt`** | **`lib/Runtime`** (JIT shared library). Omit **`add_subdirectory(Runtime)`** when the stack does not ship that target. |

### New directories (typical layout)

Mirror the existing ROCDL layout with stack-specific names, for example:

- **`include/flydsl/Dialect/<YourStack>/`**
- **`include/flydsl/Conversion/FlyTo<YourStack>/`**
- **`lib/Dialect/<YourStack>/`**, **`lib/Conversion/FlyTo<YourStack>/`**
- **`lib/CAPI/Dialect/<YourStack>/`** with **`add_mlir_public_c_api_library`**

Any translation unit that includes **`flydsl/Conversion/Passes.h`** and expects generated **`Passes.h.inc`** content must receive **`target_compile_definitions`** for the same preprocessor macro used in **`Passes.h`** (today **`FLYDSL_HAS_ROCDL_TARGET_STACK`** for the ROCDL conversion block).

---

## 3. `include/flydsl/Conversion/Passes.h` and `tools/fly-opt`

| Artifact | Requirement |
|----------|-------------|
| **`include/flydsl/Conversion/Passes.h`** | Guard stack-specific includes and **`GEN_PASS_REGISTRATION`** / **`Passes.h.inc`** inclusion with a preprocessor macro so builds without that conversion still configure. |
| **`tools/fly-opt/fly-opt.cpp`** | Match the same macro for extra includes, **`register*Pass`**, and **`registry.insert`** for stack dialects. |
| **`tools/fly-opt/CMakeLists.txt`** | Link only libraries that exist for the active stack; apply **`target_compile_definitions`** for the macro used above. |
| **`lib/CAPI/Dialect/<Stack>/CMakeLists.txt`** | If sources include **`Passes.h`**, define the macro on that target so pass registration declarations are visible. |

---

## 4. JIT runtime: `lib/Runtime/`

| Item | Notes |
|------|--------|
| **`lib/Runtime/CMakeLists.txt`** | Defines **`FlyJitRuntime`** (or a renamed / additional target). Vendor **`find_package`**, link libraries, and sources differ per stack. |
| **Output name** | The default **`OUTPUT_NAME`** is **`fly_jit_runtime`**. If you produce multiple runtime shared objects per product line, update **`jit_runtime_lib_basenames()`** in the relevant **`python/flydsl/compiler/backends/*.py`** and any install or copy rules. |
| **`python/mlir_flydsl/CMakeLists.txt`** | After **`_MLIR_LIBS_DIR`** is set, assign **`LIBRARY_OUTPUT_DIRECTORY`** for **`FlyJitRuntime`** and **`add_dependencies(FlyPythonCAPI FlyJitRuntime)`**. If a configuration omits **`FlyJitRuntime`**, wrap those lines in **`if(TARGET FlyJitRuntime)`**. |

---

## 5. Python MLIR bindings: `python/mlir_flydsl/`

| Area | Location / variable | Notes |
|------|---------------------|--------|
| Registration C++ template | **`FlyRegisterEverything.cpp.in`** | Usually one template for all stacks; fill with **`FLYDSL_REGISTER_*`** from CMake. Empty fragments must still yield valid C++. |
| Configure steps | **`CMakeLists.txt`** (top) | **`configure_file`** for **`FlyRegisterEverything.cpp.in`**; **`configure_file`** for **`cmake/FlyDSLBuildConfig.py.in`** into a generated path under **`CMAKE_CURRENT_BINARY_DIR`**. |
| Dialect bindings | **`declare_mlir_dialect_python_bindings`** | Declare stack-specific **`.td`** and Python stubs only when the stack is enabled. |
| Extension modules | **`declare_mlir_python_extension`** | Same gating for stack-specific nanobind modules (for example **`_mlirDialectsFlyROCDL`**). |
| Embed list | **`EMBED_CAPI_LINK_LIBS`** | Use **`${FLYDSL_REGISTER_EMBED_CAPI_LINK_LIBS}`**; avoid duplicating the list. |
| Upstream dialect bundle | **`MLIRFlyDSLSources`** | **`list(APPEND ... MLIRPythonSources.Dialects.<upstream>)`** when the stack is enabled; align **`<upstream>`** with **`FLYDSL_STACK_UPSTREAM_MLIR_PY_DIALECT`**. |
| Stub generation | **`_flydsl_stubgen_modules`** | Append **`-m`** entries only for extension modules built in that configuration. |
| TableGen Python copies | **`CopyFlyPythonSources`** | **`copy_if_different`** only for generated **`_fly_*`** files that exist for the stack; use a conditional command list (similar to **`_FLY_COPY_ROCDL_TABLEGEN`**) when a stack is disabled. |
| Install tree | **`CopyFlyPythonSources`** | After copying **`python/flydsl`**, copy **`_build_config.py`** into **`python_packages/flydsl/`** so imports work from the build/install layout. |

---

## 6. Python language surface: compilers, runtimes, install caps

| File | Change |
|------|--------|
| **`python/flydsl/compiler/backends/__init__.py`** | **`register_backend("<id>", ...)`**. The id must appear in **`flydsl._build_config.ENABLED_COMPILE_BACKEND_IDS`** (generated from CMake). |
| **`python/flydsl/compiler/backends/<stack>.py`** (or equivalent) | Implement **`BaseBackend`**: **`pipeline_fragments`**, **`jit_runtime_lib_basenames`**, target detection, and any stack-specific options. |
| **`python/flydsl/runtime/device_runtime/__init__.py`** | Extend **`COMPILE_BACKEND_TO_RUNTIME_KIND`** and **`_builtin_runtimes`**; document extension hooks (**`register_compile_runtime_mapping`**) if third parties map custom backend names to an existing runtime kind. Runtime kinds must appear in **`ENABLED_RUNTIME_KINDS`**. |
| **`python/flydsl/_install_limits.py`** | **`_FALLBACK`** is used when **`flydsl._build_config`** is missing (for example **`PYTHONPATH`** pointed only at a bare source tree). Update it if the default development story assumes a different stack or allow-list. |

**`get_backend`** and **`ensure_compile_runtime_pairing_from_env`** enforce install caps against **`_build_config`**. CMake-generated allow-lists must stay aligned with **`register_backend`** and device runtime registration.

---

## 7. Testing and CI

| Layer | Recommendation |
|-------|------------------|
| CMake | Add a configure job with **`FLYDSL_TARGET_STACK=<new_value>`**; optionally build **`fly-opt`** and selected Python extension targets. |
| Python | Use **`pytest.importorskip`** or markers for modules that exist only on some stacks. When tests monkeypatch **`install_caps`** or **`install_limits`**, keep allow-lists consistent with the scenario under test (see **`tests/unit/test_device_runtime.py`**). |
| Integration | Run a minimal kernel or a subset of **`tests/kernels`** on representative hardware or container images for that stack. |
