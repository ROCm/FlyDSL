// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_BINDINGS_PYTHON_BINDINGUTILS_H
#define FLYDSL_BINDINGS_PYTHON_BINDINGUTILS_H

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Wrap.h"

#define FLYDSL_REGISTER_TYPE_BINDING(CppType, PyClassName)                                         \
  static constexpr IsAFunctionTy isaFunction =                                                     \
      +[](MlirType type) { return ::mlir::isa<CppType>(unwrap(type)); };                           \
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =                                         \
      +[]() { return wrap(CppType::getTypeID()); };                                                \
  static constexpr const char *pyClassName = PyClassName;                                          \
  using Base::Base;                                                                                \
  CppType toCppType() { return ::mlir::cast<CppType>(unwrap(static_cast<MlirType>(*this))); }

#define FLYDSL_EXTRACT_TYPE_FROM_NB_HANDLE(CppType, nb_handle)                                     \
  [&]() {                                                                                          \
    auto capsule = nanobind::cast<nanobind::capsule>(nb_handle.attr(MLIR_PYTHON_CAPI_PTR_ATTR));   \
    MlirType mlirTy = mlirPythonCapsuleToType(capsule.ptr());                                      \
    auto cppType = dyn_cast<CppType>(unwrap(mlirTy));                                              \
    if (!cppType)                                                                                  \
      throw std::invalid_argument("Expected " #CppType ", got other MlirType");                    \
    return cppType;                                                                                \
  }()

#endif // FLYDSL_BINDINGS_PYTHON_BINDINGUTILS_H
