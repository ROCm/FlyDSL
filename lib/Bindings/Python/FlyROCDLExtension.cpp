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

struct PyMmaAtomCDNA3_MFMAType : PyConcreteType<PyMmaAtomCDNA3_MFMAType> {
  FLYDSL_REGISTER_TYPE_BINDING(MmaAtomCDNA3_MFMAType, "MmaAtomCDNA3_MFMAType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t m, int32_t n, int32_t k, PyType &elemTyA, PyType &elemTyB, PyType &elemTyAcc,
           DefaultingPyMlirContext context) {
          return PyMmaAtomCDNA3_MFMAType(
              context->getRef(),
              wrap(MmaAtomCDNA3_MFMAType::get(m, n, k, unwrap(elemTyA), unwrap(elemTyB),
                                              unwrap(elemTyAcc))));
        },
        "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a, "elem_ty_acc"_a, nb::kw_only(),
        "context"_a = nb::none(),
        "Create a MmaAtomCDNA3_MFMAType with m, n, k dimensions and element types");

    c.def_prop_ro("m", [](PyMmaAtomCDNA3_MFMAType &self) { return self.toCppType().getM(); });
    c.def_prop_ro("n", [](PyMmaAtomCDNA3_MFMAType &self) { return self.toCppType().getN(); });
    c.def_prop_ro("k", [](PyMmaAtomCDNA3_MFMAType &self) { return self.toCppType().getK(); });
    c.def_prop_ro("elem_ty_a", [](PyMmaAtomCDNA3_MFMAType &self) -> MlirType {
      return wrap(self.toCppType().getElemTyA());
    });
    c.def_prop_ro("elem_ty_b", [](PyMmaAtomCDNA3_MFMAType &self) -> MlirType {
      return wrap(self.toCppType().getElemTyB());
    });
    c.def_prop_ro("elem_ty_acc", [](PyMmaAtomCDNA3_MFMAType &self) -> MlirType {
      return wrap(self.toCppType().getElemTyAcc());
    });

    c.def_prop_ro("thr_layout", [](PyMmaAtomCDNA3_MFMAType &self) -> MlirType {
      auto ty = cast<MmaAtomTypeInterface>(self.toCppType());
      return wrap(LayoutType::get(cast<LayoutAttr>(ty.getThrLayout())));
    });
    c.def_prop_ro("shape_mnk", [](PyMmaAtomCDNA3_MFMAType &self) -> MlirType {
      auto ty = cast<MmaAtomTypeInterface>(self.toCppType());
      return wrap(IntTupleType::get(cast<IntTupleAttr>(ty.getShapeMNK())));
    });
    c.def_prop_ro("tv_layout_a", [](PyMmaAtomCDNA3_MFMAType &self) -> MlirType {
      auto ty = cast<MmaAtomTypeInterface>(self.toCppType());
      return wrap(LayoutType::get(cast<LayoutAttr>(ty.getThrValLayoutA())));
    });
    c.def_prop_ro("tv_layout_b", [](PyMmaAtomCDNA3_MFMAType &self) -> MlirType {
      auto ty = cast<MmaAtomTypeInterface>(self.toCppType());
      return wrap(LayoutType::get(cast<LayoutAttr>(ty.getThrValLayoutB())));
    });
    c.def_prop_ro("tv_layout_c", [](PyMmaAtomCDNA3_MFMAType &self) -> MlirType {
      auto ty = cast<MmaAtomTypeInterface>(self.toCppType());
      return wrap(LayoutType::get(cast<LayoutAttr>(ty.getThrValLayoutC())));
    });
  }
};

struct PyMmaAtomGFX1250_WMMAType : PyConcreteType<PyMmaAtomGFX1250_WMMAType> {
  FLYDSL_REGISTER_TYPE_BINDING(MmaAtomGFX1250_WMMAType, "MmaAtomGFX1250_WMMAType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t m, int32_t n, int32_t k, PyType &elemTyA, PyType &elemTyB, PyType &elemTyAcc,
           DefaultingPyMlirContext context) {
          return PyMmaAtomGFX1250_WMMAType(
              context->getRef(),
              wrap(MmaAtomGFX1250_WMMAType::get(m, n, k, unwrap(elemTyA), unwrap(elemTyB),
                                                unwrap(elemTyAcc))));
        },
        "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a, "elem_ty_acc"_a, nb::kw_only(),
        "context"_a = nb::none(),
        "Create a MmaAtomGFX1250_WMMAType with m, n, k dimensions and element types");

    c.def_prop_ro("m", [](PyMmaAtomGFX1250_WMMAType &self) { return self.toCppType().getM(); });
    c.def_prop_ro("n", [](PyMmaAtomGFX1250_WMMAType &self) { return self.toCppType().getN(); });
    c.def_prop_ro("k", [](PyMmaAtomGFX1250_WMMAType &self) { return self.toCppType().getK(); });
    c.def_prop_ro("elem_ty_a", [](PyMmaAtomGFX1250_WMMAType &self) -> MlirType {
      return wrap(self.toCppType().getElemTyA());
    });
    c.def_prop_ro("elem_ty_b", [](PyMmaAtomGFX1250_WMMAType &self) -> MlirType {
      return wrap(self.toCppType().getElemTyB());
    });
    c.def_prop_ro("elem_ty_acc", [](PyMmaAtomGFX1250_WMMAType &self) -> MlirType {
      return wrap(self.toCppType().getElemTyAcc());
    });

    c.def_prop_ro("thr_layout", [](PyMmaAtomGFX1250_WMMAType &self) -> MlirType {
      auto ty = cast<MmaAtomTypeInterface>(self.toCppType());
      return wrap(LayoutType::get(cast<LayoutAttr>(ty.getThrLayout())));
    });
    c.def_prop_ro("shape_mnk", [](PyMmaAtomGFX1250_WMMAType &self) -> MlirType {
      auto ty = cast<MmaAtomTypeInterface>(self.toCppType());
      return wrap(IntTupleType::get(cast<IntTupleAttr>(ty.getShapeMNK())));
    });
    c.def_prop_ro("tv_layout_a", [](PyMmaAtomGFX1250_WMMAType &self) -> MlirType {
      auto ty = cast<MmaAtomTypeInterface>(self.toCppType());
      return wrap(LayoutType::get(cast<LayoutAttr>(ty.getThrValLayoutA())));
    });
    c.def_prop_ro("tv_layout_b", [](PyMmaAtomGFX1250_WMMAType &self) -> MlirType {
      auto ty = cast<MmaAtomTypeInterface>(self.toCppType());
      return wrap(LayoutType::get(cast<LayoutAttr>(ty.getThrValLayoutB())));
    });
    c.def_prop_ro("tv_layout_c", [](PyMmaAtomGFX1250_WMMAType &self) -> MlirType {
      auto ty = cast<MmaAtomTypeInterface>(self.toCppType());
      return wrap(LayoutType::get(cast<LayoutAttr>(ty.getThrValLayoutC())));
    });
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

    c.def_prop_ro("bit_size",
                  [](PyCopyOpCDNA3BufferCopyType &self) { return self.toCppType().getBitSize(); });
  }
};

} // namespace fly_rocdl
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsFlyROCDL, m) {
  m.doc() = "MLIR Python FlyROCDL Extension";

  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_rocdl::PyMmaAtomCDNA3_MFMAType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_rocdl::PyMmaAtomGFX1250_WMMAType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_rocdl::PyCopyOpCDNA3BufferCopyType::bind(m);
}
