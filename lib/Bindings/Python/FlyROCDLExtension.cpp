#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>

#include "flydsl-c/FlyROCDLDialect.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace fly_rocdl {

struct PyMmaAtomCDNA3_MFMAType : PyConcreteType<PyMmaAtomCDNA3_MFMAType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFlyROCDLMmaAtomCDNA3_MFMAType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetTypeID;
  static constexpr const char *pyClassName = "MmaAtomCDNA3_MFMAType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t m, int32_t n, int32_t k, PyType &elemTyA, PyType &elemTyB, PyType &elemTyAcc,
           DefaultingPyMlirContext context) {
          return PyMmaAtomCDNA3_MFMAType(context->getRef(),
                                         wrap(::mlir::fly_rocdl::MmaAtomCDNA3_MFMAType::get(
                                             m, n, k, unwrap(static_cast<MlirType>(elemTyA)),
                                             unwrap(static_cast<MlirType>(elemTyB)),
                                             unwrap(static_cast<MlirType>(elemTyAcc)))));
        },
        "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a, "elem_ty_acc"_a, nb::kw_only(),
        "context"_a = nb::none(),
        "Create a MmaAtomCDNA3_MFMAType with m, n, k dimensions and element "
        "types");

    c.def_prop_ro("m", [](PyMmaAtomCDNA3_MFMAType &self) -> int32_t {
      return mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetM(self);
    });
    c.def_prop_ro("n", [](PyMmaAtomCDNA3_MFMAType &self) -> int32_t {
      return mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetN(self);
    });
    c.def_prop_ro("k", [](PyMmaAtomCDNA3_MFMAType &self) -> int32_t {
      return mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetK(self);
    });
    c.def_prop_ro("elem_ty_a", [](PyMmaAtomCDNA3_MFMAType &self) -> MlirType {
      return mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetElemTyA(self);
    });
    c.def_prop_ro("elem_ty_b", [](PyMmaAtomCDNA3_MFMAType &self) -> MlirType {
      return mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetElemTyB(self);
    });
    c.def_prop_ro("elem_ty_acc", [](PyMmaAtomCDNA3_MFMAType &self) -> MlirType {
      return mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetElemTyAcc(self);
    });

    c.def_prop_ro("thr_size", [](PyMmaAtomCDNA3_MFMAType &self) -> MlirType {
      auto ty = ::mlir::cast<::mlir::fly::MmaAtomTypeInterface>(unwrap(static_cast<MlirType>(self)));
      auto attr = ::mlir::cast<::mlir::fly::IntTupleAttr>(ty.getThrSize());
      return wrap(::mlir::fly::IntTupleType::get(attr));
    });
    c.def_prop_ro("shape_mnk", [](PyMmaAtomCDNA3_MFMAType &self) -> MlirType {
      auto ty = ::mlir::cast<::mlir::fly::MmaAtomTypeInterface>(unwrap(static_cast<MlirType>(self)));
      auto attr = ::mlir::cast<::mlir::fly::IntTupleAttr>(ty.getShapeMNK());
      return wrap(::mlir::fly::IntTupleType::get(attr));
    });
    c.def_prop_ro("tv_layout_a", [](PyMmaAtomCDNA3_MFMAType &self) -> MlirType {
      auto ty = ::mlir::cast<::mlir::fly::MmaAtomTypeInterface>(unwrap(static_cast<MlirType>(self)));
      auto attr = ::mlir::cast<::mlir::fly::LayoutAttr>(ty.getThrValLayoutA());
      return wrap(::mlir::fly::LayoutType::get(attr));
    });
    c.def_prop_ro("tv_layout_b", [](PyMmaAtomCDNA3_MFMAType &self) -> MlirType {
      auto ty = ::mlir::cast<::mlir::fly::MmaAtomTypeInterface>(unwrap(static_cast<MlirType>(self)));
      auto attr = ::mlir::cast<::mlir::fly::LayoutAttr>(ty.getThrValLayoutB());
      return wrap(::mlir::fly::LayoutType::get(attr));
    });
    c.def_prop_ro("tv_layout_c", [](PyMmaAtomCDNA3_MFMAType &self) -> MlirType {
      auto ty = ::mlir::cast<::mlir::fly::MmaAtomTypeInterface>(unwrap(static_cast<MlirType>(self)));
      auto attr = ::mlir::cast<::mlir::fly::LayoutAttr>(ty.getThrValLayoutC());
      return wrap(::mlir::fly::LayoutType::get(attr));
    });
  }
};

} // namespace fly_rocdl
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_fly_rocdl, m) {
  m.doc() = "MLIR Python FlyROCDL Extension";

  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_rocdl::PyMmaAtomCDNA3_MFMAType::bind(m);
}
