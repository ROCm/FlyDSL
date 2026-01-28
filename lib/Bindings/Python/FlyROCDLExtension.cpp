#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
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
using namespace mlir;
using namespace mlir::fly;
using namespace mlir::fly_rocdl;
using namespace mlir::python::nanobind_adaptors;

NB_MODULE(_fly_rocdl, m) {
  m.doc() = "MLIR Python FlyROCDL Extension";

  //===--------------------------------------------------------------------===//
  // MmaAtomCDNA3_MFMAType
  //===--------------------------------------------------------------------===//

  mlir_type_subclass(m, "MmaAtomCDNA3_MFMAType", mlirTypeIsAFlyROCDLMmaAtomCDNA3_MFMAType,
                     mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, int32_t m, int32_t n, int32_t k, MlirType elemTyA,
             MlirType elemTyB, MlirType elemTyAcc, MlirContext context) {
            return cls(wrap(MmaAtomCDNA3_MFMAType::get(m, n, k, unwrap(elemTyA), unwrap(elemTyB),
                                                       unwrap(elemTyAcc))));
          },
          "cls"_a, "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a, "elem_ty_acc"_a,
          "context"_a = nb::none(),
          // clang-format off
          nb::sig("def get(cls, m: int, n: int, k: int, elem_ty_a: " MAKE_MLIR_PYTHON_QUALNAME("ir.Type") ", elem_ty_b: " MAKE_MLIR_PYTHON_QUALNAME("ir.Type") ", elem_ty_acc: " MAKE_MLIR_PYTHON_QUALNAME("ir.Type") ", context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> MmaAtomCDNA3_MFMAType"),
          // clang-format on
          "Create a MmaAtomCDNA3_MFMAType with m, n, k dimensions and element types")
      .def_property_readonly(
          "m", [](MlirType self) { return mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetM(self); })
      .def_property_readonly(
          "n", [](MlirType self) { return mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetN(self); })
      .def_property_readonly(
          "k", [](MlirType self) { return mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetK(self); })
      .def_property_readonly(
          "elem_ty_a",
          [](MlirType self) { return mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetElemTyA(self); })
      .def_property_readonly(
          "elem_ty_b",
          [](MlirType self) { return mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetElemTyB(self); })
      .def_property_readonly("elem_ty_acc", [](MlirType self) {
        return mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetElemTyAcc(self);
      });

  //===--------------------------------------------------------------------===//
  // CopyAtom_CDNA3_BufferLSAType
  //===--------------------------------------------------------------------===//
}
