#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/BuiltinAttributes.h"  // For mlirStringAttrGet
#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"  // For mlirStringRefCreateFromCString
#include "mlir/Bindings/Python/IRCore.h"  // For populateIRCore
#include "mlir/Bindings/Python/IRTypes.h"  // For populateIRTypes, DefaultingPyMlirContext
#include "mlir/Bindings/Python/IRAttributes.h"  // For populateIRAttributes
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"  // For mlirGetDialectHandle__*__
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/BuiltinDialect.h"  // For loading builtin dialect

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>  // For BuiltinDialect
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>

#include "flydsl-c/FlyDialect.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
// NOTE: FlyROCDL is handled by _fly_rocdl.so to avoid ODR violations
#include "flydsl/Dialect/Fly/Utils/IntUtils.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"
#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"

#include "DLTensorAdaptor.h"

// Include pass registration
namespace mlir {
#define GEN_PASS_REGISTRATION
#include "flydsl/Conversion/Passes.h.inc"
} // namespace mlir

#include <cstdint>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

using namespace mlir;
using namespace mlir::fly;
using namespace mlir::python::nanobind_adaptors;

// Use the MLIR Python bindings domain namespace (must match upstream MLIR)
namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN {

//===----------------------------------------------------------------------===//
// PyConcreteType definitions for Fly dialect types
//===----------------------------------------------------------------------===//

struct PyIntTupleType : PyConcreteType<PyIntTupleType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFlyIntTupleType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = mlirFlyIntTupleTypeGetTypeID;
  static constexpr const char *pyClassName = "IntTupleType";
  using PyConcreteType::PyConcreteType;
  
  static void bindDerived(ClassTy &c);
};

struct PyLayoutType : PyConcreteType<PyLayoutType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFlyLayoutType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = mlirFlyLayoutTypeGetTypeID;
  static constexpr const char *pyClassName = "LayoutType";
  using PyConcreteType::PyConcreteType;
  
  static void bindDerived(ClassTy &c);
};

struct PySwizzleType : PyConcreteType<PySwizzleType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFlySwizzleType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = mlirFlySwizzleTypeGetTypeID;
  static constexpr const char *pyClassName = "SwizzleType";
  using PyConcreteType::PyConcreteType;
  
  static void bindDerived(ClassTy &c);
};

struct PyPointerType : PyConcreteType<PyPointerType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFlyPointerType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = mlirFlyPointerTypeGetTypeID;
  static constexpr const char *pyClassName = "PointerType";
  using PyConcreteType::PyConcreteType;
  
  static void bindDerived(ClassTy &c);
};

// ✅ RE-ADDED: Fly MemRefType (renamed to avoid conflict with upstream)
// This is Fly's custom MemRefType that uses LayoutType instead of shape/strides
struct PyFlyMemRefType : PyConcreteType<PyFlyMemRefType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFlyMemRefType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = mlirFlyMemRefTypeGetTypeID;
  static constexpr const char *pyClassName = "MemRefType";  // Keep same name for user code
  using PyConcreteType::PyConcreteType;
  
  static void bindDerived(ClassTy &c);
};

struct PyCopyAtomUniversalCopyType : PyConcreteType<PyCopyAtomUniversalCopyType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFlyCopyAtomUniversalCopyType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = mlirFlyCopyAtomUniversalCopyTypeGetTypeID;
  static constexpr const char *pyClassName = "CopyAtomUniversalCopyType";
  using PyConcreteType::PyConcreteType;
  
  static void bindDerived(ClassTy &c);
};

struct PyMmaAtomUniversalFMAType : PyConcreteType<PyMmaAtomUniversalFMAType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFlyMmaAtomUniversalFMAType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = mlirFlyMmaAtomUniversalFMATypeGetTypeID;
  static constexpr const char *pyClassName = "MmaAtomUniversalFMAType";
  using PyConcreteType::PyConcreteType;
  
  static void bindDerived(ClassTy &c);
};

} // namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN

// Bring the PyConcreteType classes into scope
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyIntTupleType;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyLayoutType;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PySwizzleType;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyPointerType;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyFlyMemRefType;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyCopyAtomUniversalCopyType;
using mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyMmaAtomUniversalFMAType;

namespace {

struct IntTupleAttrBuilder {
  MLIRContext *ctx;
  std::vector<nb::handle> dyncElems{};

  IntTupleAttrBuilder(MLIRContext *ctx) : ctx(ctx) {}

  void clear() { dyncElems.clear(); }

  IntTupleAttr operator()(nb::handle args) {
    if (PyTuple_Check(args.ptr())) {
      SmallVector<Attribute> elements;
      for (auto item : args) {
        elements.push_back((*this)(item));
      }
      return IntTupleAttr::get(ArrayAttr::get(ctx, elements));
    } else if (PyLong_Check(args.ptr())) {
      int32_t cInt = PyLong_AsLong(args.ptr());
      return IntTupleAttr::get(IntAttr::getStatic(ctx, cInt));
    } else if (args.is_none()) {
      return IntTupleAttr::getLeafNone(ctx);
    } else {
      if (!nb::hasattr(args, "_CAPIPtr")) {
        throw std::invalid_argument("Expected I32, got: " +
                                    std::string(nb::str(nb::type_name(args)).c_str()));
      }
      // dynamic value, default as i32
      dyncElems.push_back(args);
      return IntTupleAttr::get(IntAttr::getDynamic(ctx));
    }
  }
};

} // namespace

int32_t rank(MlirValue int_or_tuple) {
  mlir::Value val = unwrap(int_or_tuple);
  mlir::Type ty = val.getType();
  if (auto intTupleTy = dyn_cast<IntTupleType>(ty)) {
    return intTupleTy.getAttr().rank();
  } else if (auto layoutTy = dyn_cast<LayoutType>(ty)) {
    return layoutTy.getAttr().rank();
  } else if (auto composedLayoutTy = dyn_cast<ComposedLayoutType>(ty)) {
    return composedLayoutTy.getAttr().rank();
  } else if (auto coordTensorTy = dyn_cast<CoordTensorType>(ty)) {
    return coordTensorTy.getLayout().rank();
  } else if (auto memRefTy = dyn_cast<fly::MemRefType>(ty)) {
    return memRefTy.getLayout().rank();
  } else {
    throw std::invalid_argument("Unsupported type: ");
    ty.dump();
    return 0;
  }
}

int32_t depth(MlirValue int_or_tuple) {
  mlir::Value val = unwrap(int_or_tuple);
  mlir::Type ty = val.getType();
  if (auto intTupleTy = dyn_cast<IntTupleType>(ty)) {
    return intTupleTy.getAttr().depth();
  } else if (auto layoutTy = dyn_cast<LayoutType>(ty)) {
    return layoutTy.getAttr().depth();
  } else if (auto composedLayoutTy = dyn_cast<ComposedLayoutType>(ty)) {
    return composedLayoutTy.getAttr().depth();
  } else if (auto coordTensorTy = dyn_cast<CoordTensorType>(ty)) {
    return coordTensorTy.getLayout().depth();
  } else if (auto memRefTy = dyn_cast<fly::MemRefType>(ty)) {
    return memRefTy.getLayout().depth();
  } else {
    throw std::invalid_argument("Unsupported type: ");
    ty.dump();
    return 0;
  }
}

// nb::object getFlyTypingModule() {
//   static nb::object typing = nb::steal(nb::module_::import_("fly.lang.typing"));
//   return typing;
// }

// nb::object make_int32(int value) {
//   static nb::object int32_cls = getFlyTypingModule().attr("Int32");

//   return int32_cls(value);
// }

// nb::object make_int32_tuple(int value) {
//   static nb::object int32_cls = getFlyTypingModule().attr("Int32");

//   nb::list subList;
//   subList.append(int32_cls(value + 1));
//   nb::tuple subTuple = nb::tuple(subList);

//   nb::list retList;
//   retList.append(int32_cls(value));
//   retList.append(subTuple);
//   retList.append(nb::int_(0));

//   return nb::tuple(retList);
// }

NB_MODULE(_fly, m) {
  m.doc() = "MLIR Python FlyDSL Extension";

  // ✅ CRITICAL: Use the MLIR Python bindings domain namespace
  using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;

  // ==========================================================================
  // CRITICAL: Register Fly passes FIRST (global registration, happens once)
  // ==========================================================================
  // This must happen before any MLIR operations to ensure passes are available
  // for PassManager.parse()
  // ==========================================================================
  static bool passesRegistered = false;
  if (!passesRegistered) {
    mlir::fly::registerFlyPasses();
    mlir::registerFlyToROCDLConversionPass();
    passesRegistered = true;
  }

  // ==========================================================================
  // CRITICAL: Create PyGlobals instance and register MLIR core types FIRST
  // ==========================================================================
  // Since we're using a separate domain (flydsl), we need to:
  // 1. Create a PyGlobals instance for our domain
  // 2. Register all base types (PyType, PyAttribute, PyContext, etc.)
  //
  // This MUST happen BEFORE we register any Fly-specific types!
  // ==========================================================================
  
  populateRoot(m);          // Registers _Globals class and creates PyGlobals instance
  populateIRCore(m);        // Registers PyType, PyAttribute, PyContext, PyModule, PyValue, etc.
  populateIRTypes(m);       // Registers specific IR types (IntegerType, etc.)
  populateIRAttributes(m);  // Registers specific IR attributes

  // ==========================================================================
  // Now we can safely register Fly dialect
  // ==========================================================================

  // ✅ Register Fly dialect function (based on mlir/test/python/lib/PythonTestModuleNanobind.cpp)
  m.def("_register_dialect", 
      [](DefaultingPyMlirContext context, bool load) {
        MlirContext ctxC = context.get()->get();
        
        // Note: We DO NOT explicitly load BuiltinDialect here because:
        // 1. The Python-side context from upstream MLIR should already have it loaded
        // 2. Calling ctx->getOrLoadDialect<BuiltinDialect>() would use the static
        //    copy in _fly.so, causing symbol conflicts
        
        // Load Fly dialect using C API
        MlirDialectHandle flyHandle = mlirGetDialectHandle__fly__();
        mlirDialectHandleRegisterDialect(flyHandle, ctxC);
        if (load) {
          mlirDialectHandleLoadDialect(flyHandle, ctxC);
        }
        
        // NOTE: We do NOT load FlyROCDL here!
        // FlyROCDL should be loaded by _fly_rocdl.so to avoid ODR violations.
        // The dialect will be loaded on first type creation.
      }, 
      nb::arg("context").none() = nb::none(), 
      nb::arg("load") = true,
      "Register Fly dialect with the given context");

  using DLTensorAdaptor = utils::DLTensorAdaptor;

  nb::class_<DLTensorAdaptor>(m, "DLTensorAdaptor")
      .def(nb::init<nb::object, int32_t, bool, MlirContext>(), "dlpack_capsule"_a,
           "alignment"_a = 1, "use_32bit_stride"_a = false, "context"_a,
           "Create a DLTensorAdaptor from a DLPack capsule")
      .def_prop_ro("shape", &DLTensorAdaptor::getShape, "Get tensor shape as tuple")
      .def_prop_ro("stride", &DLTensorAdaptor::getStride, "Get tensor stride as tuple")
      .def_prop_ro("data_ptr", &DLTensorAdaptor::getDataPtr, "Get data pointer as int64")
      .def_prop_ro("address_space", &DLTensorAdaptor::getAddressSpace,
                   "Get address space (0=host, 1=device)")
      .def("size_in_bytes", &DLTensorAdaptor::getSizeInBytes, "Get total size in bytes")
      .def("build_memref_desc", &DLTensorAdaptor::buildMemRefDesc,
           "Build memref descriptor based on current dynamic marks")
      .def("get_memref_type", &DLTensorAdaptor::getMemRefType,
           "Get fly.memref MLIR type based on current dynamic marks")
      .def("get_c_pointers", &DLTensorAdaptor::getCPointers, "Get list of c pointers")
      .def("mark_layout_dynamic", &DLTensorAdaptor::markLayoutDynamic, "leading_dim"_a = -1,
           "divisibility"_a = 1, "Mark entire layout as dynamic except leading dim stride")
      .def("use_32bit_stride", &DLTensorAdaptor::use32BitStride, "use_32bit_stride"_a,
           "Decide whether to use 32-bit stride");

  m.def(
      "infer_int_tuple_type",
      [](nb::handle int_or_tuple, MlirContext context) {
        MLIRContext *ctx = unwrap(context);
        IntTupleAttrBuilder builder{ctx};
        IntTupleAttr attr = builder(int_or_tuple);
        return std::make_pair(wrap(IntTupleType::get(attr)), builder.dyncElems);
      },
      "int_or_tuple"_a, "context"_a = nb::none(),
      // clang-format off
      nb::sig("def infer_int_tuple_type(int_or_tuple, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None)"),
      // clang-format on
      "infer IntTupleType for given input");

  m.def("rank", &rank, "int_or_tuple"_a,
        nb::sig("def rank(int_or_tuple: " MAKE_MLIR_PYTHON_QUALNAME("ir.Value") ") -> int"));
  m.def("depth", &depth, "int_or_tuple"_a,
        nb::sig("def depth(int_or_tuple: " MAKE_MLIR_PYTHON_QUALNAME("ir.Value") ") -> int"));

  //===--------------------------------------------------------------------===//
  // Core Types
  //===--------------------------------------------------------------------===//

  PyIntTupleType::bind(m);
  PyLayoutType::bind(m);
  PySwizzleType::bind(m);
  PyPointerType::bind(m);
  
  // ✅ Fly MemRefType (uses LayoutType instead of shape/strides)
  PyFlyMemRefType::bind(m);
  
  PyCopyAtomUniversalCopyType::bind(m);
  PyMmaAtomUniversalFMAType::bind(m);
}

//===----------------------------------------------------------------------===//
// bindDerived implementations
//===----------------------------------------------------------------------===//

void PyIntTupleType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](nb::handle int_or_tuple, DefaultingPyMlirContext context) {
        MLIRContext *ctx = unwrap(context->get());
        IntTupleAttrBuilder builder{ctx};
        IntTupleAttr attr = builder(int_or_tuple);
        MlirType t = wrap(IntTupleType::get(attr));
        return PyIntTupleType(context->getRef(), t);
      },
      "int_or_tuple"_a, "context"_a = nb::none(),
      nb::sig("def get(int_or_tuple: int | tuple, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> IntTupleType"),
      "Create an IntTupleType from Python int or tuple");
  c.def_prop_ro("rank", [](MlirType self) { return mlirFlyIntTupleTypeGetRank(self); });
  c.def_prop_ro("depth", [](MlirType self) { return mlirFlyIntTupleTypeGetDepth(self); });
  c.def_prop_ro("is_leaf", [](MlirType self) { return mlirFlyIntTupleTypeIsLeaf(self); });
  c.def_prop_ro("is_static", [](MlirType self) { return mlirFlyIntTupleTypeIsStatic(self); });
}

void PyLayoutType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](nb::handle shape, nb::handle stride, DefaultingPyMlirContext context) {
        MLIRContext *ctx = unwrap(context->get());
        auto toIntTupleAttr = [ctx](nb::handle h) -> IntTupleAttr {
          if (nb::hasattr(h, "_CAPIPtr")) {
            auto capsule = nb::cast<nb::capsule>(h.attr(MLIR_PYTHON_CAPI_PTR_ATTR));
            MlirType mlirTy = mlirPythonCapsuleToType(capsule.ptr());
            auto intTupleType = dyn_cast<IntTupleType>(unwrap(mlirTy));
            if (!intTupleType) {
              throw std::invalid_argument("Expected IntTupleType, got other MlirType");
            }
            return intTupleType.getAttr();
          }
          IntTupleAttrBuilder builder{ctx};
          return builder(h);
        };

        IntTupleAttr shapeAttr = toIntTupleAttr(shape);
        IntTupleAttr strideAttr = toIntTupleAttr(stride);
        auto layoutAttr = LayoutAttr::get(ctx, shapeAttr, strideAttr);
        MlirType t = wrap(LayoutType::get(layoutAttr));
        return PyLayoutType(context->getRef(), t);
      },
      "shape"_a, "stride"_a, "context"_a = nb::none(),
      nb::sig("def get(shape: int | tuple | IntTupleType, stride: int | tuple | IntTupleType, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> LayoutType"),
      "Create a LayoutType with shape and stride");
  c.def_prop_ro("shape", [](MlirType self) { return mlirFlyLayoutTypeGetShape(self); });
  c.def_prop_ro("stride", [](MlirType self) { return mlirFlyLayoutTypeGetStride(self); });
  c.def_prop_ro("rank", [](MlirType self) { return mlirFlyLayoutTypeGetRank(self); });
  c.def_prop_ro("depth", [](MlirType self) { return mlirFlyLayoutTypeGetDepth(self); });
  c.def_prop_ro("is_leaf", [](MlirType self) { return mlirFlyLayoutTypeIsLeaf(self); });
  c.def_prop_ro("is_static", [](MlirType self) { return mlirFlyLayoutTypeIsStatic(self); });
  c.def_prop_ro("is_static_shape", [](MlirType self) { return mlirFlyLayoutTypeIsStaticShape(self); });
  c.def_prop_ro("is_static_stride", [](MlirType self) { return mlirFlyLayoutTypeIsStaticStride(self); });
}

void PySwizzleType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](int32_t mask, int32_t base, int32_t shift, DefaultingPyMlirContext context) {
        MLIRContext *ctx = unwrap(context->get());
        SwizzleAttr attr = SwizzleAttr::get(ctx, mask, base, shift);
        MlirType t = wrap(SwizzleType::get(attr));
        return PySwizzleType(context->getRef(), t);
      },
      "mask"_a, "base"_a, "shift"_a, "context"_a = nb::none(),
      nb::sig("def get(mask: int, base: int, shift: int, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> SwizzleType"),
      "Create a SwizzleType");
  c.def_prop_ro("mask", [](MlirType self) { return mlirFlySwizzleTypeGetMask(self); });
  c.def_prop_ro("base", [](MlirType self) { return mlirFlySwizzleTypeGetBase(self); });
  c.def_prop_ro("shift", [](MlirType self) { return mlirFlySwizzleTypeGetShift(self); });
}

void PyPointerType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](nb::object elemTyObj, std::optional<int32_t> addressSpace,
         std::optional<int32_t> alignment, DefaultingPyMlirContext context) {
        MLIRContext *ctx = unwrap(context->get());

        auto capsule = nb::cast<nb::capsule>(elemTyObj.attr(MLIR_PYTHON_CAPI_PTR_ATTR));
        MlirType elemTy = mlirPythonCapsuleToType(capsule.ptr());

        AddressSpace addr = AddressSpace::Register;
        if (addressSpace.has_value()) {
          addr = static_cast<AddressSpace>(addressSpace.value());
        }
        int32_t alignSize = 1;
        if (alignment.has_value()) {
          alignSize = alignment.value();
        }
        assert(alignSize > 0 && "alignment must be positive");

        MlirType t = wrap(fly::PointerType::get(unwrap(elemTy), AddressSpaceAttr::get(ctx, addr),
                                                 AlignAttr::get(ctx, alignSize)));
        return PyPointerType(context->getRef(), t);
      },
      "elem_ty"_a, "address_space"_a = nb::none(), "alignment"_a = nb::none(),
      "context"_a = nb::none(),
      nb::sig("def get(elem_ty: " MAKE_MLIR_PYTHON_QUALNAME("ir.Type") ", address_space: int = 0, alignment: int | None = None, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> PointerType"),
      "Create a PointerType with element type and address space");
  c.def_prop_ro("element_type", [](MlirType self) { return mlirFlyPointerTypeGetElementType(self); });
  c.def_prop_ro("address_space", [](MlirType self) { return mlirFlyPointerTypeGetAddressSpace(self); });
  c.def_prop_ro("alignment", [](MlirType self) { return mlirFlyPointerTypeGetAlignment(self); });
  c.def_prop_ro("swizzle", [](MlirType self) { return mlirFlyPointerTypeGetSwizzle(self); });
}

// ✅ Fly MemRefType binding (uses LayoutType instead of shape/strides)
void PyFlyMemRefType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](nb::object elemTyObj, nb::object layoutObj, int32_t addressSpace,
         std::optional<int32_t> alignment, DefaultingPyMlirContext context) {
        MLIRContext *ctx = unwrap(context->get());
        
        // Extract element type
        auto elemCapsule = nb::cast<nb::capsule>(elemTyObj.attr(MLIR_PYTHON_CAPI_PTR_ATTR));
        MlirType elemTy = mlirPythonCapsuleToType(elemCapsule.ptr());
        
        // Extract layout type
        auto layoutCapsule = nb::cast<nb::capsule>(layoutObj.attr(MLIR_PYTHON_CAPI_PTR_ATTR));
        MlirType layoutTy = mlirPythonCapsuleToType(layoutCapsule.ptr());
        
        int32_t alignSize = alignment.value_or(1);
        assert(alignSize > 0 && "alignment must be positive");
        
        MlirType t = mlirFlyMemRefTypeGet(elemTy, layoutTy, addressSpace, alignSize);
        return PyFlyMemRefType(context->getRef(), t);
      },
      "elem_ty"_a, "layout"_a, "address_space"_a = 0, "alignment"_a = nb::none(),
      "context"_a = nb::none(),
      nb::sig("def get(elem_ty: " MAKE_MLIR_PYTHON_QUALNAME("ir.Type") 
              ", layout: LayoutType, address_space: int = 0, alignment: int | None = None, "
              "context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> MemRefType"),
      "Create a Fly MemRefType with element type, layout, and address space");
  c.def_prop_ro("element_type", [](MlirType self) { return mlirFlyMemRefTypeGetElementType(self); });
  c.def_prop_ro("layout", [](MlirType self) { return mlirFlyMemRefTypeGetLayout(self); });
  c.def_prop_ro("address_space", [](MlirType self) { return mlirFlyMemRefTypeGetAddressSpace(self); });
  c.def_prop_ro("alignment", [](MlirType self) { return mlirFlyMemRefTypeGetAlignment(self); });
  c.def_prop_ro("swizzle", [](MlirType self) { return mlirFlyMemRefTypeGetSwizzle(self); });
}

void PyCopyAtomUniversalCopyType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](int32_t bitSize, DefaultingPyMlirContext context) {
        MLIRContext *ctx = unwrap(context->get());
        MlirType t = wrap(CopyAtomUniversalCopyType::get(ctx, bitSize));
        return PyCopyAtomUniversalCopyType(context->getRef(), t);
      },
      "bitSize"_a, "context"_a = nb::none(),
      nb::sig("def get(bitSize: int, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> CopyAtomUniversalCopyType"),
      "Create a CopyAtomUniversalCopyType with bit size");
  c.def_prop_ro("bit_size", [](MlirType self) {
    return mlirFlyCopyAtomUniversalCopyTypeGetBitSize(self);
  });
}

void PyMmaAtomUniversalFMAType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](MlirType elemTy, DefaultingPyMlirContext context) {
        MlirType t = wrap(MmaAtomUniversalFMAType::get(unwrap(elemTy)));
        return PyMmaAtomUniversalFMAType(context->getRef(), t);
      },
      "elem_ty"_a, "context"_a = nb::none(),
      nb::sig("def get(elem_ty: " MAKE_MLIR_PYTHON_QUALNAME("ir.Type") ", context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> MmaAtomUniversalFMAType"),
      "Create a MmaAtomUniversalFMAType with element type");
  c.def_prop_ro("elem_ty", [](MlirType self) { return mlirFlyMmaAtomUniversalFMATypeGetElemTy(self); });
}

