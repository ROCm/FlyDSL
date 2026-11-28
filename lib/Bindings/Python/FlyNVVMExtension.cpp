// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyNVVM/IR/Dialect.h"

#include "BindingUtils.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace ::mlir::fly;
using namespace ::mlir::fly_nvvm;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace fly_nvvm {

struct PyCopyOpSM75_LdMatrixType : PyConcreteType<PyCopyOpSM75_LdMatrixType> {
  FLYDSL_REGISTER_TYPE_BINDING(CopyOpSM75_LdMatrixType, "CopyOpSM75_LdMatrixType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t num, bool trans, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          return PyCopyOpSM75_LdMatrixType(context->getRef(),
                                           wrap(CopyOpSM75_LdMatrixType::get(ctx, num, trans)));
        },
        "num"_a, "trans"_a = false, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyOpSM75_LdMatrixType (ldmatrix) with num tiles and transpose flag");
  }
};

struct PyMmaOpSM80_MmaSyncType : PyConcreteType<PyMmaOpSM80_MmaSyncType> {
  FLYDSL_REGISTER_TYPE_BINDING(MmaOpSM80_MmaSyncType, "MmaOpSM80_MmaSyncType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t m, int32_t n, int32_t k, PyType &elemTyA, PyType &elemTyB, PyType &elemTyAcc,
           DefaultingPyMlirContext context) {
          return PyMmaOpSM80_MmaSyncType(
              context->getRef(),
              wrap(MmaOpSM80_MmaSyncType::get(m, n, k, unwrap(elemTyA), unwrap(elemTyB),
                                              unwrap(elemTyAcc))));
        },
        "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a, "elem_ty_acc"_a, nb::kw_only(),
        "context"_a = nb::none(),
        "Create a MmaOpSM80_MmaSyncType with m, n, k dimensions and element types");
  }
};

struct PyCopyOpSM80_CpAsyncType : PyConcreteType<PyCopyOpSM80_CpAsyncType> {
  FLYDSL_REGISTER_TYPE_BINDING(CopyOpSM80_CpAsyncType, "CopyOpSM80_CpAsyncType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t bit_size, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          return PyCopyOpSM80_CpAsyncType(context->getRef(),
                                          wrap(CopyOpSM80_CpAsyncType::get(ctx, bit_size)));
        },
        "bit_size"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyOpSM80_CpAsyncType (cp.async.shared.global) with the given bit size");
  }
};

} // namespace fly_nvvm
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsFlyNVVM, m) {
  m.doc() = "MLIR Python FlyNVVM Extension";

  // clang-format off
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_nvvm::PyCopyOpSM75_LdMatrixType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_nvvm::PyMmaOpSM80_MmaSyncType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_nvvm::PyCopyOpSM80_CpAsyncType::bind(m);
  // clang-format on
}
