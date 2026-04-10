/* Version script for libFlyPythonCAPI.so
 *
 * Hides internal LLVM C/C++ symbols to prevent symbol conflicts with
 * ROCm's statically-linked LLVM (in libamd_comgr.so).
 *
 * Only exports:
 *   - mlir* C API functions (used by Python extension modules)
 *   - mlir:: C++ symbols (used by _mlirDialectsFly.so, _mlir.so, etc.)
 *   - PyInit_* Python module initializers
 */
FLYDSL_0.1 {
  global:
    /* MLIR C API (un-mangled, lowercase mlir prefix) */
    mlir*;
    flydsl*;

    /* Python module initializers */
    PyInit_*;

    /* mlir:: C++ namespace — needed by extension modules
     * (_mlirDialectsFly.so, _mlir.so, _mlirDialectsGPU.so, etc.)
     * Covers: mlir::fly::*, mlir::fly_rocdl::*, mlir::python::*,
     *         mlir::detail::*, mlir::*Type, mlir::*Attr, etc.
     */
    _ZN4mlir*;
    _ZNK4mlir*;
    _ZTVN4mlir*;
    _ZTIN4mlir*;
    _ZTSN4mlir*;

  local:
    /* Hide everything else — in particular:
     *   _ZN4llvm*  (llvm:: C++ symbols)
     *   LLVM*      (LLVM C API, not used by any extension module)
     */
    *;
};
