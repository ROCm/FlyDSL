// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_BINDINGS_PYTHON_BINDINGUTILS_H
#define FLYDSL_BINDINGS_PYTHON_BINDINGUTILS_H

#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"

#define FLYDSL_REGISTER_TYPE_BINDING(CppType, PyClassName)                                         \
  static constexpr IsAFunctionTy isaFunction =                                                     \
      +[](MlirType type) { return ::mlir::isa<CppType>(unwrap(type)); };                           \
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =                                         \
      +[]() { return wrap(CppType::getTypeID()); };                                                \
  static constexpr const char *pyClassName = PyClassName;                                          \
  using Base::Base;                                                                                \
  CppType toCppType() { return ::mlir::cast<CppType>(unwrap(static_cast<MlirType>(*this))); }

#endif // FLYDSL_BINDINGS_PYTHON_BINDINGUTILS_H
