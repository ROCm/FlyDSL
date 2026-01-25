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

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntUtils.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace mlir::fly;

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

namespace {

/// Helper to convert Python value to IntTupleAttr
struct IntTupleAttrBuilder {
  MLIRContext *ctx;
  std::vector<nb::handle> dyncElems{};

  IntTupleAttrBuilder(MLIRContext *ctx) : ctx(ctx) {}

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
      // Dynamic value - for now treat as dynamic
      dyncElems.push_back(args);
      return IntTupleAttr::get(IntAttr::getDynamic(ctx));
    }
  }
};

} // namespace

int32_t rank(nb::handle int_or_tuple) {
  nb::object capsule = int_or_tuple.attr("_CAPIPtr");
  MlirValue mlirVal = mlirPythonCapsuleToValue(capsule.ptr());
  mlir::Value val = unwrap(mlirVal);
  mlir::Type ty = val.getType();
  if (auto intTupleTy = dyn_cast<IntTupleType>(ty)) {
    return intTupleTy.getAttr().rank();
  } else if (auto layoutTy = dyn_cast<LayoutType>(ty)) {
    return layoutTy.getAttr().rank();
  }
  return 1;
}

int32_t depth(nb::handle int_or_tuple) {
  nb::object capsule = int_or_tuple.attr("_CAPIPtr");
  MlirValue mlirVal = mlirPythonCapsuleToValue(capsule.ptr());
  mlir::Value val = unwrap(mlirVal);
  mlir::Type ty = val.getType();
  if (auto intTupleTy = dyn_cast<IntTupleType>(ty)) {
    return intTupleTy.getAttr().depth();
  } else if (auto layoutTy = dyn_cast<LayoutType>(ty)) {
    return layoutTy.getAttr().depth();
  }
  return 0;
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
      [](MlirContext context, nb::handle int_or_tuple) {
        MLIRContext *ctx = unwrap(context);
        IntTupleAttrBuilder builder{ctx};
        IntTupleAttr attr = builder(int_or_tuple);
        auto intTupleType = IntTupleType::get(attr);
        MlirType wrappedType = wrap(intTupleType);
        return std::make_pair(wrappedType, builder.dyncElems);
      },
      nb::arg("context"), nb::arg("int_or_tuple"));

  m.def(
      "infer_layout_type",
      [](MlirContext context, nb::handle shape, nb::handle stride) {
        MLIRContext *ctx = unwrap(context);
        IntTupleAttrBuilder builder{ctx};
        IntTupleAttr shapeAttr = builder(shape);
        IntTupleAttr strideAttr = builder(stride);
        auto layoutAttr = LayoutAttr::get(ctx, shapeAttr, strideAttr);
        auto layoutType = LayoutType::get(ctx, layoutAttr);
        MlirType wrappedType = wrap(layoutType);
        return wrappedType;
      },
      nb::arg("context"), nb::arg("shape"), nb::arg("stride"));

  m.def("rank", &rank, nb::arg("int_or_tuple"));
  m.def("depth", &depth, nb::arg("int_or_tuple"));

  //===--------------------------------------------------------------------===//
  // Fly Type Classes with static get() methods
  //===--------------------------------------------------------------------===//

  nb::class_<fly::PointerType>(m, "PointerType")
      .def_static(
          "get",
          [](MlirType elemTy, int32_t addressSpace, std::optional<int32_t> alignment) {
            mlir::Type unwrappedElemTy = unwrap(elemTy);
            MLIRContext *ctx = unwrappedElemTy.getContext();

            AddressSpaceAttr addrSpaceAttr =
                AddressSpaceAttr::get(ctx, static_cast<AddressSpace>(addressSpace));

            fly::PointerType ptrType;
            if (alignment.has_value()) {
              AlignAttr alignAttr = AlignAttr::get(ctx, alignment.value());
              ptrType = fly::PointerType::get(ctx, unwrappedElemTy, addrSpaceAttr, alignAttr,
                                              SwizzleAttr::getTrivialSwizzle(ctx));
            } else {
              ptrType = fly::PointerType::get(unwrappedElemTy, addrSpaceAttr);
            }
            return wrap(static_cast<mlir::Type>(ptrType));
          },
          nb::arg("elem_ty"), nb::arg("address_space"), nb::arg("alignment") = nb::none(),
          "Create a PointerType with element type and address space");

  nb::class_<fly::MemRefType>(m, "MemRefType")
      .def_static(
          "get",
          [](MlirType elemTy, int32_t addressSpace, MlirType layoutTy,
             std::optional<int32_t> alignment) {
            mlir::Type unwrappedElemTy = unwrap(elemTy);
            mlir::Type unwrappedLayoutTy = unwrap(layoutTy);
            MLIRContext *ctx = unwrappedElemTy.getContext();

            auto layoutType = dyn_cast<LayoutType>(unwrappedLayoutTy);
            if (!layoutType) {
              throw std::invalid_argument("layout must be a LayoutType");
            }

            AddressSpaceAttr addrSpaceAttr =
                AddressSpaceAttr::get(ctx, static_cast<AddressSpace>(addressSpace));
            LayoutAttr layoutAttr = layoutType.getAttr();

            fly::MemRefType memrefType;
            if (alignment.has_value()) {
              AlignAttr alignAttr = AlignAttr::get(ctx, alignment.value());
              memrefType = fly::MemRefType::get(ctx, unwrappedElemTy, addrSpaceAttr, layoutAttr,
                                                alignAttr, SwizzleAttr::getTrivialSwizzle(ctx));
            } else {
              memrefType = fly::MemRefType::get(unwrappedElemTy, addrSpaceAttr, layoutAttr);
            }
            return wrap(static_cast<mlir::Type>(memrefType));
          },
          nb::arg("elem_ty"), nb::arg("address_space"), nb::arg("layout"),
          nb::arg("alignment") = nb::none(),
          "Create a MemRefType with element type, address space and layout");

  nb::class_<fly::LayoutType>(m, "LayoutType")
      .def_static(
          "get",
          [](MlirContext context, nb::handle shape, nb::handle stride) {
            MLIRContext *ctx = unwrap(context);
            IntTupleAttrBuilder builder{ctx};
            IntTupleAttr shapeAttr = builder(shape);
            IntTupleAttr strideAttr = builder(stride);
            auto layoutAttr = LayoutAttr::get(ctx, shapeAttr, strideAttr);
            auto layoutType = LayoutType::get(ctx, layoutAttr);
            return wrap(static_cast<mlir::Type>(layoutType));
          },
          nb::arg("context"), nb::arg("shape"), nb::arg("stride"),
          "Create a LayoutType with shape and stride");

  // IntTupleType class
  nb::class_<fly::IntTupleType>(m, "IntTupleType")
      .def_static(
          "get",
          [](MlirContext context, nb::handle int_or_tuple) {
            MLIRContext *ctx = unwrap(context);
            IntTupleAttrBuilder builder{ctx};
            IntTupleAttr attr = builder(int_or_tuple);
            auto intTupleType = IntTupleType::get(attr);
            return std::make_pair(wrap(static_cast<mlir::Type>(intTupleType)), builder.dyncElems);
          },
          nb::arg("context"), nb::arg("int_or_tuple"),
          "Create an IntTupleType from Python int or tuple");
}
