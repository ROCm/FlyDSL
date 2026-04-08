// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"

#include "BindingUtils.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace ::mlir::fly;
using namespace ::mlir::fly_rocdl;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace fly_rocdl {

struct PyMmaOpCDNA3_MFMAType : PyConcreteType<PyMmaOpCDNA3_MFMAType> {
  FLYDSL_REGISTER_TYPE_BINDING(MmaOpCDNA3_MFMAType, "MmaOpCDNA3_MFMAType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t m, int32_t n, int32_t k, PyType &elemTyA, PyType &elemTyB, PyType &elemTyAcc,
           DefaultingPyMlirContext context) {
          return PyMmaOpCDNA3_MFMAType(context->getRef(), wrap(MmaOpCDNA3_MFMAType::get(
                                                              m, n, k, unwrap(elemTyA),
                                                              unwrap(elemTyB), unwrap(elemTyAcc))));
        },
        "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a, "elem_ty_acc"_a, nb::kw_only(),
        "context"_a = nb::none(),
        "Create a MmaOpCDNA3_MFMAType with m, n, k dimensions and element types");
  }
};

struct PyMmaOpGFX1250_WMMAType : PyConcreteType<PyMmaOpGFX1250_WMMAType> {
  FLYDSL_REGISTER_TYPE_BINDING(MmaOpGFX1250_WMMAType, "MmaOpGFX1250_WMMAType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t m, int32_t n, int32_t k, PyType &elemTyA, PyType &elemTyB, PyType &elemTyAcc,
           DefaultingPyMlirContext context) {
          return PyMmaOpGFX1250_WMMAType(
              context->getRef(),
              wrap(MmaOpGFX1250_WMMAType::get(m, n, k, unwrap(elemTyA), unwrap(elemTyB),
                                              unwrap(elemTyAcc))));
        },
        "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a, "elem_ty_acc"_a, nb::kw_only(),
        "context"_a = nb::none(),
        "Create a MmaOpGFX1250_WMMAType with m, n, k dimensions and element types");
  }
};

struct PyCopyOpCDNA3BufferCopyType : PyConcreteType<PyCopyOpCDNA3BufferCopyType> {
  FLYDSL_REGISTER_TYPE_BINDING(CopyOpCDNA3BufferCopyType, "CopyOpCDNA3BufferCopyType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t bitSize, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          return PyCopyOpCDNA3BufferCopyType(context->getRef(),
                                             wrap(CopyOpCDNA3BufferCopyType::get(ctx, bitSize)));
        },
        "bit_size"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyOpCDNA3BufferCopyType with the given bit size");
  }
};

} // namespace fly_rocdl
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsFlyROCDL, m) {
  m.doc() = "MLIR Python FlyROCDL Extension";

  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_rocdl::PyMmaOpCDNA3_MFMAType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_rocdl::PyMmaOpGFX1250_WMMAType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_rocdl::PyCopyOpCDNA3BufferCopyType::bind(m);
}
