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

#include "flydsl-c/FlyDialect.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntUtils.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace mlir::fly;
using namespace mlir::python::nanobind_adaptors;

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

  mlir_type_subclass(m, "IntTupleType", mlirTypeIsAFlyIntTupleType, mlirFlyIntTupleTypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, nb::handle int_or_tuple, MlirContext context) {
            MLIRContext *ctx = unwrap(context);
            IntTupleAttrBuilder builder{ctx};
            IntTupleAttr attr = builder(int_or_tuple);
            return cls(wrap(IntTupleType::get(attr)));
          },
          "cls"_a, "int_or_tuple"_a, "context"_a = nb::none(),
          // clang-format off
          nb::sig("def get(cls, int_or_tuple: int | tuple, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> IntTupleType"),
          // clang-format on
          "Create an IntTupleType from Python int or tuple")
      .def_property_readonly("rank", [](MlirType self) { return mlirFlyIntTupleTypeGetRank(self); })
      .def_property_readonly("depth",
                             [](MlirType self) { return mlirFlyIntTupleTypeGetDepth(self); })
      .def_property_readonly("is_leaf",
                             [](MlirType self) { return mlirFlyIntTupleTypeIsLeaf(self); })
      .def_property_readonly("is_static",
                             [](MlirType self) { return mlirFlyIntTupleTypeIsStatic(self); });

  mlir_type_subclass(m, "LayoutType", mlirTypeIsAFlyLayoutType, mlirFlyLayoutTypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, nb::handle shape, nb::handle stride, MlirContext context) {
            MLIRContext *ctx = unwrap(context);
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
            return cls(wrap(LayoutType::get(layoutAttr)));
          },
          "cls"_a, "shape"_a, "stride"_a, "context"_a = nb::none(),
          // clang-format off
          nb::sig("def get(cls, shape: int | tuple | IntTupleType, stride: int | tuple | IntTupleType, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> LayoutType"),
          // clang-format on
          "Create a LayoutType with shape and stride")
      .def_property_readonly("shape", [](MlirType self) { return mlirFlyLayoutTypeGetShape(self); })
      .def_property_readonly("stride",
                             [](MlirType self) { return mlirFlyLayoutTypeGetStride(self); })
      .def_property_readonly("rank", [](MlirType self) { return mlirFlyLayoutTypeGetRank(self); })
      .def_property_readonly("depth", [](MlirType self) { return mlirFlyLayoutTypeGetDepth(self); })
      .def_property_readonly("is_leaf", [](MlirType self) { return mlirFlyLayoutTypeIsLeaf(self); })
      .def_property_readonly("is_static",
                             [](MlirType self) { return mlirFlyLayoutTypeIsStatic(self); })
      .def_property_readonly("is_static_shape",
                             [](MlirType self) { return mlirFlyLayoutTypeIsStaticShape(self); })
      .def_property_readonly("is_static_stride",
                             [](MlirType self) { return mlirFlyLayoutTypeIsStaticStride(self); });

  mlir_type_subclass(m, "SwizzleType", mlirTypeIsAFlySwizzleType, mlirFlySwizzleTypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, int32_t mask, int32_t base, int32_t shift,
             MlirContext context) {
            MLIRContext *ctx = unwrap(context);
            SwizzleAttr attr = SwizzleAttr::get(ctx, mask, base, shift);
            return cls(wrap(SwizzleType::get(attr)));
          },
          "cls"_a, "mask"_a, "base"_a, "shift"_a, "context"_a = nb::none(),
          // clang-format off
          nb::sig("def get(cls, mask: int, base: int, shift: int, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> SwizzleType"),
          // clang-format on
          "Create a SwizzleType")
      .def_property_readonly("mask", [](MlirType self) { return mlirFlySwizzleTypeGetMask(self); })
      .def_property_readonly("base", [](MlirType self) { return mlirFlySwizzleTypeGetBase(self); })
      .def_property_readonly("shift",
                             [](MlirType self) { return mlirFlySwizzleTypeGetShift(self); });

  mlir_type_subclass(m, "PointerType", mlirTypeIsAFlyPointerType, mlirFlyPointerTypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, nb::object elemTyObj, std::optional<int32_t> addressSpace,
             std::optional<int32_t> alignment, MlirContext context) {
            MLIRContext *ctx = unwrap(context);

            // Manual type conversion from nb::object to MlirType
            auto capsule = nb::cast<nb::capsule>(elemTyObj.attr(MLIR_PYTHON_CAPI_PTR_ATTR));
            MlirType elemTy = mlirPythonCapsuleToType(capsule.ptr());

            // default address space is Register
            AddressSpace addr = AddressSpace::Register;
            if (addressSpace.has_value()) {
              addr = static_cast<AddressSpace>(addressSpace.value());
            }
            int32_t alignSize = 1;
            if (alignment.has_value()) {
              alignSize = alignment.value();
            }
            assert(alignSize > 0 && "alignment must be positive");

            return cls(wrap(fly::PointerType::get(unwrap(elemTy), AddressSpaceAttr::get(ctx, addr),
                                                  AlignAttr::get(ctx, alignSize))));
          },
          "cls"_a, "elem_ty"_a, "address_space"_a = nb::none(), "alignment"_a = nb::none(),
          "context"_a = nb::none(),
          // clang-format off
          nb::sig("def get(cls, elem_ty: " MAKE_MLIR_PYTHON_QUALNAME("ir.Type") ", address_space: int = 0, alignment: int | None = None, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> PointerType"),
          // clang-format on
          "Create a PointerType with element type and address space")
      .def_property_readonly("element_type",
                             [](MlirType self) { return mlirFlyPointerTypeGetElementType(self); })
      .def_property_readonly("address_space",
                             [](MlirType self) { return mlirFlyPointerTypeGetAddressSpace(self); })
      .def_property_readonly("alignment",
                             [](MlirType self) { return mlirFlyPointerTypeGetAlignment(self); })
      .def_property_readonly("swizzle",
                             [](MlirType self) { return mlirFlyPointerTypeGetSwizzle(self); });

  mlir_type_subclass(m, "MemRefType", mlirTypeIsAFlyMemRefType, mlirFlyMemRefTypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirType elemTy, MlirType layoutMlirTy,
             std::optional<int32_t> addressSpace, std::optional<int32_t> alignment,
             MlirContext context) {
            MLIRContext *ctx = unwrap(context);

            auto layoutType = dyn_cast<LayoutType>(unwrap(layoutMlirTy));
            if (!layoutType) {
              throw std::invalid_argument("layout must be a LayoutType");
            }

            // default address space is Register
            AddressSpace addr = AddressSpace::Register;
            if (addressSpace.has_value()) {
              addr = static_cast<AddressSpace>(addressSpace.value());
            }

            int32_t alignSize = 1;
            if (alignment.has_value()) {
              alignSize = alignment.value();
            }
            assert(alignSize > 0 && "alignment must be positive");

            return cls(
                wrap(fly::MemRefType::get(unwrap(elemTy), AddressSpaceAttr::get(ctx, addr),
                                          layoutType.getAttr(), AlignAttr::get(ctx, alignSize))));
          },
          "cls"_a, "elem_ty"_a, "layout"_a, "address_space"_a = 0, "alignment"_a = nb::none(),
          "context"_a = nb::none(),
          // clang-format off
          nb::sig("def get(cls, elem_ty: " MAKE_MLIR_PYTHON_QUALNAME("ir.Type") ", layout: LayoutType, address_space: int = 0, alignment: int | None = None, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> MemRefType"),
          // clang-format on
          "Create a MemRefType with element type, layout, address space and alignment")
      .def_property_readonly("element_type",
                             [](MlirType self) { return mlirFlyMemRefTypeGetElementType(self); })
      .def_property_readonly("layout",
                             [](MlirType self) { return mlirFlyMemRefTypeGetLayout(self); })
      .def_property_readonly("address_space",
                             [](MlirType self) { return mlirFlyMemRefTypeGetAddressSpace(self); })
      .def_property_readonly("alignment",
                             [](MlirType self) { return mlirFlyMemRefTypeGetAlignment(self); })
      .def_property_readonly("swizzle",
                             [](MlirType self) { return mlirFlyMemRefTypeGetSwizzle(self); });

  mlir_type_subclass(m, "CopyAtomUniversalCopyType", mlirTypeIsAFlyCopyAtomUniversalCopyType,
                     mlirFlyCopyAtomUniversalCopyTypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, int32_t bitSize, MlirContext context) {
            MLIRContext *ctx = unwrap(context);
            return cls(wrap(CopyAtomUniversalCopyType::get(ctx, bitSize)));
          },
          "cls"_a, "bitSize"_a, "context"_a = nb::none(),
          // clang-format off
          nb::sig("def get(cls, bitSize: int, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> CopyAtomUniversalCopyType"),
          // clang-format on
          "Create a CopyAtomUniversalCopyType with bit size")
      .def_property_readonly("bit_size", [](MlirType self) {
        return mlirFlyCopyAtomUniversalCopyTypeGetBitSize(self);
      });

  mlir_type_subclass(m, "MmaAtomUniversalFMAType", mlirTypeIsAFlyMmaAtomUniversalFMAType,
                     mlirFlyMmaAtomUniversalFMATypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirType elemTy, MlirContext context) {
            return cls(wrap(MmaAtomUniversalFMAType::get(unwrap(elemTy))));
          },
          "cls"_a, "elem_ty"_a, "context"_a = nb::none(),
          // clang-format off
          nb::sig("def get(cls, elem_ty: " MAKE_MLIR_PYTHON_QUALNAME("ir.Type") ", context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None) -> MmaAtomUniversalFMAType"),
          // clang-format on
          "Create a MmaAtomUniversalFMAType with element type")
      .def_property_readonly(
          "elem_ty", [](MlirType self) { return mlirFlyMmaAtomUniversalFMATypeGetElemTy(self); });
}
