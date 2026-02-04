#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/IRCore.h"  // For populateIRCore
#include "mlir/Bindings/Python/IRTypes.h"  // For populateIRTypes
#include "mlir/Bindings/Python/IRAttributes.h"  // For populateIRAttributes
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

// Use the MLIR Python bindings domain namespace
namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN {

//===----------------------------------------------------------------------===//
// PyConcreteType definitions for FlyROCDL dialect types
//===----------------------------------------------------------------------===//

struct PyMmaAtomCDNA3_MFMAType : PyConcreteType<PyMmaAtomCDNA3_MFMAType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFlyROCDLMmaAtomCDNA3_MFMAType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetTypeID;
  static constexpr const char *pyClassName = "MmaAtomCDNA3_MFMAType";
  using PyConcreteType::PyConcreteType;
  
  static void bindDerived(ClassTy &c);
};

} // namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN

using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyMmaAtomCDNA3_MFMAType;

NB_MODULE(_fly_rocdl, m) {
  m.doc() = "MLIR Python FlyROCDL Extension";

  //===--------------------------------------------------------------------===//
  // MmaAtomCDNA3_MFMAType
  //===--------------------------------------------------------------------===//

  PyMmaAtomCDNA3_MFMAType::bind(m);

  //===--------------------------------------------------------------------===//
  // CopyAtom_CDNA3_BufferLSAType
  //===--------------------------------------------------------------------===//
}

//===----------------------------------------------------------------------===//
// bindDerived implementations
//===----------------------------------------------------------------------===//

void PyMmaAtomCDNA3_MFMAType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](int32_t m, int32_t n, int32_t k, MlirType elemTyA,
         MlirType elemTyB, MlirType elemTyAcc, DefaultingPyMlirContext context) {
        // Get the MLIRContext* from the passed context
        MlirContext ctx = context->get();
        mlir::MLIRContext *cppCtx = unwrap(ctx);
        
        // Ensure FlyROCDL dialect is loaded using C API
        // This avoids ODR issues with C++ loadDialect
        MlirDialectHandle flyROCDLHandle = mlirGetDialectHandle__fly_rocdl__();
        mlirDialectHandleRegisterDialect(flyROCDLHandle, ctx);
        mlirDialectHandleLoadDialect(flyROCDLHandle, ctx);
        
        MlirType t = wrap(MmaAtomCDNA3_MFMAType::get(cppCtx, m, n, k, unwrap(elemTyA), unwrap(elemTyB),
                                                      unwrap(elemTyAcc)));
        return PyMmaAtomCDNA3_MFMAType(context->getRef(), t);
      },
      "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a, "elem_ty_acc"_a,
      "context"_a = nb::none(),
      nb::sig("def get(m: int, n: int, k: int, elem_ty_a: " MAKE_MLIR_PYTHON_QUALNAME("ir.Type") ", elem_ty_b: " MAKE_MLIR_PYTHON_QUALNAME("ir.Type") ", elem_ty_acc: " MAKE_MLIR_PYTHON_QUALNAME("ir.Type") ", context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> MmaAtomCDNA3_MFMAType"),
      "Create a MmaAtomCDNA3_MFMAType with m, n, k dimensions and element types");
  c.def_prop_ro("m", [](MlirType self) { return mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetM(self); });
  c.def_prop_ro("n", [](MlirType self) { return mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetN(self); });
  c.def_prop_ro("k", [](MlirType self) { return mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetK(self); });
  c.def_prop_ro("elem_ty_a", [](MlirType self) { return mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetElemTyA(self); });
  c.def_prop_ro("elem_ty_b", [](MlirType self) { return mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetElemTyB(self); });
  c.def_prop_ro("elem_ty_acc", [](MlirType self) { return mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetElemTyAcc(self); });
}

