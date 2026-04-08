// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

#include "BindingUtils.h"
#include "DLTensorAdaptor.h"
#include "TiledOpTraits.h"

#include "LlvmConfig/llvm.h"

#include <cstdint>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace ::mlir;
using namespace ::mlir::fly;

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
      dyncElems.push_back(args);
      return IntTupleAttr::get(IntAttr::getDynamic(ctx));
    }
  }
};

IntTupleAttr getIntTupleAttrFromHandle(nb::handle h, IntTupleAttrBuilder &builder) {
  if (nb::hasattr(h, MLIR_PYTHON_CAPI_PTR_ATTR)) {
    return FLYDSL_EXTRACT_TYPE_FROM_NB_HANDLE(::mlir::fly::IntTupleType, h).getAttr();
  }
  return builder(h);
}

int32_t rank(MlirValue int_or_tuple) {
  Value val = unwrap(int_or_tuple);
  Type ty = val.getType();
  if (auto t = dyn_cast<IntTupleType>(ty))
    return t.getAttr().rank();
  if (auto t = dyn_cast<LayoutType>(ty))
    return t.getAttr().rank();
  if (auto t = dyn_cast<ComposedLayoutType>(ty))
    return t.getAttr().rank();
  if (auto t = dyn_cast<CoordTensorType>(ty))
    return cast<NestedAttrInterface>(t.getLayout()).rank();
  if (auto t = dyn_cast<fly::MemRefType>(ty))
    return cast<NestedAttrInterface>(t.getLayout()).rank();
  throw std::invalid_argument("Unsupported type for rank()");
}

int32_t depth(MlirValue int_or_tuple) {
  Value val = unwrap(int_or_tuple);
  Type ty = val.getType();
  if (auto t = dyn_cast<IntTupleType>(ty))
    return t.getAttr().depth();
  if (auto t = dyn_cast<LayoutType>(ty))
    return t.getAttr().depth();
  if (auto t = dyn_cast<ComposedLayoutType>(ty))
    return t.getAttr().depth();
  if (auto t = dyn_cast<CoordTensorType>(ty))
    return cast<NestedAttrInterface>(t.getLayout()).depth();
  if (auto t = dyn_cast<fly::MemRefType>(ty))
    return cast<NestedAttrInterface>(t.getLayout()).depth();
  throw std::invalid_argument("Unsupported type for depth()");
}

bool has_none(MlirValue int_or_tuple) {
  ::mlir::Value val = unwrap(int_or_tuple);
  ::mlir::Type ty = val.getType();
  if (auto t = ::mlir::dyn_cast<::mlir::fly::IntTupleType>(ty))
    return ::mlir::fly::intTupleHasNone(t.getAttr());
  throw std::invalid_argument("has_none() expected IntTupleType");
}

} // namespace

// =============================================================================
// PyConcreteType definitions in the MLIR Python domain
// =============================================================================

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace fly {

// ---------------------------------------------------------------------------
// IntTupleType
// ---------------------------------------------------------------------------
struct PyIntTupleType : PyConcreteType<PyIntTupleType> {
  FLYDSL_REGISTER_TYPE_BINDING(::mlir::fly::IntTupleType, "IntTupleType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](nb::handle int_or_tuple, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          IntTupleAttrBuilder builder{ctx};
          auto attr = builder(int_or_tuple);
          return PyIntTupleType(context->getRef(), wrap(IntTupleType::get(attr)));
        },
        "int_or_tuple"_a, nb::kw_only(), "context"_a = nb::none(),
        // clang-format off
        nb::sig("def get(int_or_tuple, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None)"),
        // clang-format on
        "Create an IntTupleType from Python int or tuple");

    c.def_prop_ro("rank", [](PyIntTupleType &self) { return self.toCppType().rank(); });
    c.def_prop_ro("depth", [](PyIntTupleType &self) { return self.toCppType().depth(); });
    c.def_prop_ro("is_leaf", [](PyIntTupleType &self) { return self.toCppType().isLeaf(); });
    c.def_prop_ro("is_static", [](PyIntTupleType &self) { return self.toCppType().isStatic(); });
    c.def_prop_ro("get_static_leaf_int", [](PyIntTupleType &self) {
      auto ty = self.toCppType();
      assert(ty.isLeaf() && ty.isStatic());
      return ty.getAttr().getLeafAsInt().getValue();
    });
  }
};

// ---------------------------------------------------------------------------
// TileType
// ---------------------------------------------------------------------------
struct PyTileType : PyConcreteType<PyTileType> {
  FLYDSL_REGISTER_TYPE_BINDING(::mlir::fly::TileType, "TileType");

  static Attribute extractTileModeAttr(nb::handle mode, MLIRContext *ctx) {
    if (mode.is_none())
      return IntAttr::getNone(ctx);
    if (PyLong_Check(mode.ptr()))
      return IntAttr::getStatic(ctx, nb::cast<int32_t>(mode));
    if (PyTuple_Check(mode.ptr())) {
      SmallVector<Attribute> nested;
      for (auto item : mode)
        nested.push_back(extractTileModeAttr(nb::handle(item), ctx));
      return TileAttr::get(ArrayAttr::get(ctx, nested));
    }
    if (nb::hasattr(mode, MLIR_PYTHON_CAPI_PTR_ATTR)) {
      auto layoutTy = FLYDSL_EXTRACT_TYPE_FROM_NB_HANDLE(::mlir::fly::LayoutType, mode);
      return layoutTy.getAttr();
    }
    throw std::invalid_argument("TileType.get: expected int, None, tuple, or LayoutType");
  }

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](nb::object modeOrModes, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          if (nb::isinstance<nb::list>(modeOrModes)) {
            SmallVector<Attribute> attrs;
            for (auto mode : nb::cast<nb::list>(modeOrModes))
              attrs.push_back(extractTileModeAttr(nb::handle(mode), ctx));
            auto tileAttr = TileAttr::get(ArrayAttr::get(ctx, attrs));
            return PyTileType(context->getRef(), wrap(TileType::get(tileAttr)));
          } else {
            auto attr = extractTileModeAttr(modeOrModes, ctx);
            auto tileAttr = TileAttr::get(attr);
            return PyTileType(context->getRef(), wrap(TileType::get(tileAttr)));
          }
        },
        "modes"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a TileType from a list of modes or a single mode (leaf tile)");

    c.def_prop_ro("rank", [](PyTileType &self) { return self.toCppType().rank(); });
  }
};

// ---------------------------------------------------------------------------
// LayoutType
// ---------------------------------------------------------------------------
struct PyLayoutType : PyConcreteType<PyLayoutType> {
  FLYDSL_REGISTER_TYPE_BINDING(::mlir::fly::LayoutType, "LayoutType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](nb::handle shape, nb::handle stride, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());

          IntTupleAttrBuilder builder{ctx};
          auto shapeAttr = getIntTupleAttrFromHandle(shape, builder);
          auto strideAttr = getIntTupleAttrFromHandle(stride, builder);
          auto layoutAttr = LayoutAttr::get(ctx, shapeAttr, strideAttr);
          return PyLayoutType(context->getRef(), wrap(LayoutType::get(layoutAttr)));
        },
        "shape"_a, "stride"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a LayoutType with shape and stride");

    c.def_prop_ro("shape", [](PyLayoutType &self) -> MlirType {
      return wrap(IntTupleType::get(self.toCppType().getAttr().getShape()));
    });
    c.def_prop_ro("stride", [](PyLayoutType &self) -> MlirType {
      return wrap(IntTupleType::get(self.toCppType().getAttr().getStride()));
    });
    c.def_prop_ro("rank", [](PyLayoutType &self) { return self.toCppType().rank(); });
    c.def_prop_ro("depth", [](PyLayoutType &self) { return self.toCppType().depth(); });
    c.def_prop_ro("is_leaf", [](PyLayoutType &self) { return self.toCppType().isLeaf(); });
    c.def_prop_ro("is_static", [](PyLayoutType &self) { return self.toCppType().isStatic(); });
    c.def_prop_ro("is_static_shape",
                  [](PyLayoutType &self) { return self.toCppType().isStaticShape(); });
    c.def_prop_ro("is_static_stride",
                  [](PyLayoutType &self) { return self.toCppType().isStaticStride(); });
  }
};

// ---------------------------------------------------------------------------
// ComposedLayoutType
// ---------------------------------------------------------------------------
struct PyComposedLayoutType : PyConcreteType<PyComposedLayoutType> {
  FLYDSL_REGISTER_TYPE_BINDING(::mlir::fly::ComposedLayoutType, "ComposedLayoutType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &innerObj, nb::handle offset, PyType &outerObj, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          Type innerTy = unwrap(innerObj);
          Attribute innerAttr;

          if (auto layout = dyn_cast<LayoutType>(innerTy))
            innerAttr = layout.getAttr();
          else if (auto composed = dyn_cast<ComposedLayoutType>(innerTy))
            innerAttr = composed.getAttr();
          else if (auto swizzle = dyn_cast<SwizzleType>(innerTy))
            innerAttr = swizzle.getAttr();
          else
            throw std::invalid_argument(
                "inner must be a LayoutType, ComposedLayoutType or SwizzleType");

          IntTupleAttrBuilder builder{ctx};
          auto offsetAttr = getIntTupleAttrFromHandle(offset, builder);
          auto outerTy = dyn_cast<LayoutType>(unwrap(outerObj));
          if (!outerTy)
            throw std::invalid_argument("outer must be a LayoutType");

          auto attr = ComposedLayoutAttr::get(innerAttr, offsetAttr, outerTy.getAttr());
          return PyComposedLayoutType(context->getRef(), wrap(ComposedLayoutType::get(attr)));
        },
        "inner"_a, "offset"_a, "outer"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a ComposedLayoutType with inner, offset and outer");

    c.def_prop_ro("inner", [](PyComposedLayoutType &self) -> MlirType {
      auto innerAttr = self.toCppType().getAttr().getInner();
      if (auto layout = dyn_cast<LayoutAttr>(innerAttr))
        return wrap(LayoutType::get(layout));
      if (auto composed = dyn_cast<ComposedLayoutAttr>(innerAttr))
        return wrap(ComposedLayoutType::get(composed));
      if (auto swizzle = dyn_cast<SwizzleAttr>(innerAttr))
        return wrap(SwizzleType::get(swizzle));
      throw std::invalid_argument("Expected LayoutAttr, ComposedLayoutAttr or SwizzleAttr");
    });
    c.def_prop_ro("offset", [](PyComposedLayoutType &self) -> MlirType {
      return wrap(IntTupleType::get(self.toCppType().getAttr().getOffset()));
    });
    c.def_prop_ro("outer", [](PyComposedLayoutType &self) -> MlirType {
      return wrap(LayoutType::get(self.toCppType().getAttr().getOuter()));
    });
    c.def_prop_ro("rank", [](PyComposedLayoutType &self) { return self.toCppType().rank(); });
    c.def_prop_ro("depth", [](PyComposedLayoutType &self) { return self.toCppType().depth(); });
    c.def_prop_ro("is_leaf", [](PyComposedLayoutType &self) { return self.toCppType().isLeaf(); });
    c.def_prop_ro("is_static",
                  [](PyComposedLayoutType &self) { return self.toCppType().isStatic(); });
    c.def_prop_ro("is_static_outer",
                  [](PyComposedLayoutType &self) { return self.toCppType().isStaticOuter(); });
    c.def_prop_ro("is_static_inner",
                  [](PyComposedLayoutType &self) { return self.toCppType().isStaticInner(); });
    c.def_prop_ro("is_static_offset",
                  [](PyComposedLayoutType &self) { return self.toCppType().isStaticOffset(); });
  }
};

// ---------------------------------------------------------------------------
// SwizzleType
// ---------------------------------------------------------------------------
struct PySwizzleType : PyConcreteType<PySwizzleType> {
  FLYDSL_REGISTER_TYPE_BINDING(::mlir::fly::SwizzleType, "SwizzleType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t mask, int32_t base, int32_t shift, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          auto attr = SwizzleAttr::get(ctx, mask, base, shift);
          return PySwizzleType(context->getRef(), wrap(SwizzleType::get(attr)));
        },
        "mask"_a, "base"_a, "shift"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a SwizzleType");

    c.def_prop_ro("mask", [](PySwizzleType &self) { return self.toCppType().getAttr().getMask(); });
    c.def_prop_ro("base", [](PySwizzleType &self) { return self.toCppType().getAttr().getBase(); });
    c.def_prop_ro("shift",
                  [](PySwizzleType &self) { return self.toCppType().getAttr().getShift(); });
  }
};

// ---------------------------------------------------------------------------
// PointerType
// ---------------------------------------------------------------------------
struct PyPointerType : PyConcreteType<PyPointerType> {
  FLYDSL_REGISTER_TYPE_BINDING(::mlir::fly::PointerType, "PointerType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &elemTyObj, std::optional<int32_t> addressSpace, std::optional<int32_t> alignment,
           DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          auto elemType = unwrap(elemTyObj);

          auto addr = AddressSpace::Global;
          if (addressSpace.has_value())
            addr = static_cast<AddressSpace>(addressSpace.value());

          int32_t alignSize =
              alignment.value_or(AlignAttr::getTrivialAlignment(elemType).getAlignment());
          int32_t elemByte = (elemType.getIntOrFloatBitWidth() + 7) / 8;
          if (alignSize <= 0 || alignSize % elemByte != 0)
            throw std::invalid_argument(
                "alignment must be a positive multiple of element byte size (" +
                std::to_string(elemByte) + "), got " + std::to_string(alignSize));

          return PyPointerType(context->getRef(),
                               wrap(PointerType::get(elemType, AddressSpaceAttr::get(ctx, addr),
                                                     AlignAttr::get(ctx, alignSize))));
        },
        "elem_ty"_a, "address_space"_a = nb::none(), "alignment"_a = nb::none(), nb::kw_only(),
        "context"_a = nb::none(), "Create a PointerType with element type and address space");

    c.def_prop_ro("element_type", [](PyPointerType &self) -> MlirType {
      return wrap(self.toCppType().getElemTy());
    });
    c.def_prop_ro("address_space", [](PyPointerType &self) -> int32_t {
      return static_cast<int32_t>(self.toCppType().getAddressSpace().getValue());
    });
    c.def_prop_ro("alignment", [](PyPointerType &self) -> int32_t {
      return self.toCppType().getAlignment().getAlignment();
    });
    c.def_prop_ro("swizzle", [](PyPointerType &self) -> MlirType {
      return wrap(SwizzleType::get(self.toCppType().getSwizzle()));
    });
  }
};

// ---------------------------------------------------------------------------
// MemRefType
// ---------------------------------------------------------------------------
struct PyMemRefType : PyConcreteType<PyMemRefType> {
  FLYDSL_REGISTER_TYPE_BINDING(::mlir::fly::MemRefType, "MemRefType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &elemTyObj, PyType &layoutObj, std::optional<int32_t> addressSpace,
           std::optional<int32_t> alignment, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());

          Type layoutTy = unwrap(layoutObj);
          Attribute layoutAttr;
          if (auto layoutType = dyn_cast<LayoutType>(layoutTy))
            layoutAttr = layoutType.getAttr();
          else if (auto composedType = dyn_cast<ComposedLayoutType>(layoutTy))
            layoutAttr = composedType.getAttr();
          else
            throw std::invalid_argument("layout must be a LayoutType or ComposedLayoutType");

          auto addr = AddressSpace::Register;
          if (addressSpace.has_value())
            addr = static_cast<AddressSpace>(addressSpace.value());

          auto elemType = unwrap(elemTyObj);
          int32_t alignSize =
              alignment.value_or(AlignAttr::getTrivialAlignment(elemType).getAlignment());
          int32_t elemByte = (elemType.getIntOrFloatBitWidth() + 7) / 8;
          if (alignSize <= 0 || alignSize % elemByte != 0)
            throw std::invalid_argument(
                "alignment must be a positive multiple of element byte size (" +
                std::to_string(elemByte) + "), got " + std::to_string(alignSize));

          return PyMemRefType(context->getRef(), wrap(::mlir::fly::MemRefType::get(
                                                     elemType, AddressSpaceAttr::get(ctx, addr),
                                                     layoutAttr, AlignAttr::get(ctx, alignSize))));
        },
        "elem_ty"_a, "layout"_a, "address_space"_a = 0, "alignment"_a = nb::none(), nb::kw_only(),
        "context"_a = nb::none(),
        "Create a MemRefType with element type, layout, address space and "
        "alignment");

    c.def_prop_ro("element_type", [](PyMemRefType &self) -> MlirType {
      return wrap(self.toCppType().getElemTy());
    });
    c.def_prop_ro("layout", [](PyMemRefType &self) -> MlirType {
      Attribute layout = self.toCppType().getLayout();
      if (auto la = dyn_cast<LayoutAttr>(layout))
        return wrap(LayoutType::get(la));
      return wrap(ComposedLayoutType::get(cast<ComposedLayoutAttr>(layout)));
    });
    c.def_prop_ro("address_space", [](PyMemRefType &self) -> int32_t {
      return static_cast<int32_t>(self.toCppType().getAddressSpace().getValue());
    });
    c.def_prop_ro("alignment", [](PyMemRefType &self) -> int32_t {
      return self.toCppType().getAlignment().getAlignment();
    });
    c.def_prop_ro("swizzle", [](PyMemRefType &self) -> MlirType {
      return wrap(SwizzleType::get(self.toCppType().getSwizzle()));
    });
    c.def_prop_ro(
        "leading_dim",
        [](PyMemRefType &self) -> nb::object {
          Attribute layout = self.toCppType().getLayout();
          if (auto la = dyn_cast<LayoutAttr>(layout)) {
            IntTupleAttr stride = la.getStride();
            std::vector<int32_t> path{};

            auto findLeadingDimPath = [&](auto &&self, IntTupleAttr stride) -> bool {
              if (stride.isLeaf())
                return stride.isLeafStaticValue(1);

              for (int32_t i = 0; i < stride.rank(); ++i) {
                path.push_back(i);
                if (self(self, stride.at(i)))
                  return true;
                path.pop_back();
              }
              return false;
            };

            if (!findLeadingDimPath(findLeadingDimPath, stride))
              return nb::none();

            if (path.empty()) // for leaf layout
              return nb::int_(0);
            if (path.size() == 1)
              return nb::int_(path.front());

            nb::list result;
            for (int32_t idx : path)
              result.append(nb::int_(idx));
            return nb::tuple(result);
          }
          throw std::invalid_argument("leading_dim() does not support MemRefType with "
                                      "ComposedLayout");
        },
        "Return the first left-to-right mode whose stride is statically 1");
  }
};

// ---------------------------------------------------------------------------
// CoordTensorType
// ---------------------------------------------------------------------------
struct PyCoordTensorType : PyConcreteType<PyCoordTensorType> {
  FLYDSL_REGISTER_TYPE_BINDING(::mlir::fly::CoordTensorType, "CoordTensorType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](nb::handle base, PyType &layoutObj, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());

          IntTupleAttrBuilder builder{ctx};
          auto baseAttr = getIntTupleAttrFromHandle(base, builder);

          Type layoutTy = unwrap(layoutObj);
          Attribute layoutAttr;
          if (auto layoutType = dyn_cast<LayoutType>(layoutTy))
            layoutAttr = layoutType.getAttr();
          else if (auto composedType = dyn_cast<ComposedLayoutType>(layoutTy))
            layoutAttr = composedType.getAttr();
          else
            throw std::invalid_argument("layout must be a LayoutType or ComposedLayoutType");

          return PyCoordTensorType(context->getRef(),
                                   wrap(CoordTensorType::get(baseAttr, layoutAttr)));
        },
        "base"_a, "layout"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a CoordTensorType with base and layout");

    c.def_prop_ro("base", [](PyCoordTensorType &self) -> MlirType {
      return wrap(IntTupleType::get(self.toCppType().getBase()));
    });
    c.def_prop_ro("layout", [](PyCoordTensorType &self) -> MlirType {
      Attribute layout = self.toCppType().getLayout();
      if (auto la = dyn_cast<LayoutAttr>(layout))
        return wrap(LayoutType::get(la));
      return wrap(ComposedLayoutType::get(cast<ComposedLayoutAttr>(layout)));
    });
    c.def_prop_ro("rank", [](PyCoordTensorType &self) { return self.toCppType().rank(); });
    c.def_prop_ro("depth", [](PyCoordTensorType &self) { return self.toCppType().depth(); });
    c.def_prop_ro("is_leaf", [](PyCoordTensorType &self) { return self.toCppType().isLeaf(); });
    c.def_prop_ro("is_static", [](PyCoordTensorType &self) { return self.toCppType().isStatic(); });
  }
};

// ---------------------------------------------------------------------------
// CopyAtomType
// ---------------------------------------------------------------------------
struct PyCopyAtomType : PyConcreteType<PyCopyAtomType> {
  FLYDSL_REGISTER_TYPE_BINDING(::mlir::fly::CopyAtomType, "CopyAtomType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &copyOp, int32_t valBits, DefaultingPyMlirContext context) {
          return PyCopyAtomType(context->getRef(),
                                wrap(CopyAtomType::get(unwrap(copyOp), valBits)));
        },
        "copy_op"_a, "val_bits"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyAtomType with the given copy op type and value bits");

    c.def_prop_ro("copy_op", [](PyCopyAtomType &self) -> MlirType {
      return wrap(self.toCppType().getCopyOp());
    });
    c.def_prop_ro("val_bits", [](PyCopyAtomType &self) { return self.toCppType().getValBits(); });
    c.def_prop_ro("thr_layout", [](PyCopyAtomType &self) -> MlirType {
      return wrap(LayoutType::get(cast<LayoutAttr>(self.toCppType().getThrLayout())));
    });
    c.def_prop_ro("tv_layout_src", [](PyCopyAtomType &self) -> MlirType {
      return wrap(LayoutType::get(cast<LayoutAttr>(self.toCppType().getThrValLayoutSrc())));
    });
    c.def_prop_ro("tv_layout_dst", [](PyCopyAtomType &self) -> MlirType {
      return wrap(LayoutType::get(cast<LayoutAttr>(self.toCppType().getThrValLayoutDst())));
    });
    c.def_prop_ro("tv_layout_ref", [](PyCopyAtomType &self) -> MlirType {
      return wrap(LayoutType::get(cast<LayoutAttr>(self.toCppType().getThrValLayoutRef())));
    });
  }
};

// ---------------------------------------------------------------------------
// MmaAtomType
// ---------------------------------------------------------------------------
struct PyMmaAtomType : PyConcreteType<PyMmaAtomType> {
  FLYDSL_REGISTER_TYPE_BINDING(::mlir::fly::MmaAtomType, "MmaAtomType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &mmaOp, DefaultingPyMlirContext context) {
          return PyMmaAtomType(context->getRef(), wrap(MmaAtomType::get(unwrap(mmaOp))));
        },
        "mma_op"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a MmaAtomType wrapping an MmaOpTypeInterface type");

    c.def_prop_ro("mma_op", [](PyMmaAtomType &self) -> MlirType {
      return wrap(self.toCppType().getMmaOp());
    });
    c.def_prop_ro("thr_layout", [](PyMmaAtomType &self) -> MlirType {
      return wrap(LayoutType::get(cast<LayoutAttr>(self.toCppType().getThrLayout())));
    });
    c.def_prop_ro("shape_mnk", [](PyMmaAtomType &self) -> MlirType {
      return wrap(IntTupleType::get(cast<IntTupleAttr>(self.toCppType().getShapeMNK())));
    });
    c.def_prop_ro("tv_layout_a", [](PyMmaAtomType &self) -> MlirType {
      return wrap(LayoutType::get(cast<LayoutAttr>(self.toCppType().getThrValLayoutA())));
    });
    c.def_prop_ro("tv_layout_b", [](PyMmaAtomType &self) -> MlirType {
      return wrap(LayoutType::get(cast<LayoutAttr>(self.toCppType().getThrValLayoutB())));
    });
    c.def_prop_ro("tv_layout_c", [](PyMmaAtomType &self) -> MlirType {
      return wrap(LayoutType::get(cast<LayoutAttr>(self.toCppType().getThrValLayoutC())));
    });
  }
};

// ---------------------------------------------------------------------------
// TiledCopyType
// ---------------------------------------------------------------------------
struct PyTiledCopyType : PyConcreteType<PyTiledCopyType> {
  FLYDSL_REGISTER_TYPE_BINDING(::mlir::fly::TiledCopyType, "TiledCopyType");

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro("copy_atom", [](PyTiledCopyType &self) -> MlirType {
      return wrap(self.toCppType().getCopyAtom());
    });
    c.def_prop_ro("layout_thr_val", [](PyTiledCopyType &self) -> MlirType {
      return wrap(self.toCppType().getLayoutThrVal());
    });
    c.def_prop_ro("tile_mn", [](PyTiledCopyType &self) -> MlirType {
      return wrap(self.toCppType().getTileMN());
    });
    c.def_prop_ro("tiled_tv_layout_src", [](PyTiledCopyType &self) -> MlirType {
      auto ty = self.toCppType();
      auto copyAtom = cast<CopyAtomType>(ty.getCopyAtom());
      auto result = tiledCopyGetTiledThrValLayoutSrc(copyAtom, ty.getLayoutThrVal().getAttr(),
                                                     ty.getTileMN().getAttr());
      return wrap(LayoutType::get(result));
    });
    c.def_prop_ro("tiled_tv_layout_dst", [](PyTiledCopyType &self) -> MlirType {
      auto ty = self.toCppType();
      auto copyAtom = cast<CopyAtomType>(ty.getCopyAtom());
      auto result = tiledCopyGetTiledThrValLayoutDst(copyAtom, ty.getLayoutThrVal().getAttr(),
                                                     ty.getTileMN().getAttr());
      return wrap(LayoutType::get(result));
    });
  }
};

// ---------------------------------------------------------------------------
// TiledMmaType
// ---------------------------------------------------------------------------
struct PyTiledMmaType : PyConcreteType<PyTiledMmaType> {
  FLYDSL_REGISTER_TYPE_BINDING(::mlir::fly::TiledMmaType, "TiledMmaType");

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro("mma_atom", [](PyTiledMmaType &self) -> MlirType {
      return wrap(self.toCppType().getMmaAtom());
    });
    c.def_prop_ro("atom_layout", [](PyTiledMmaType &self) -> MlirType {
      return wrap(self.toCppType().getAtomLayout());
    });
    c.def_prop_ro("permutation", [](PyTiledMmaType &self) -> MlirType {
      return wrap(self.toCppType().getPermutation());
    });
    c.def_prop_ro("tile_size_mnk", [](PyTiledMmaType &self) -> MlirType {
      auto ty = self.toCppType();
      auto mmaAtom = cast<MmaAtomType>(ty.getMmaAtom());
      auto result = tiledMmaGetTileSizeMNK(mmaAtom, ty.getAtomLayout().getAttr(),
                                           ty.getPermutation().getAttr());
      return wrap(IntTupleType::get(result));
    });
    c.def_prop_ro("thr_layout_vmnk", [](PyTiledMmaType &self) -> MlirType {
      auto ty = self.toCppType();
      auto mmaAtom = cast<MmaAtomType>(ty.getMmaAtom());
      auto result = tiledMmaGetThrLayoutVMNK(mmaAtom, ty.getAtomLayout().getAttr());
      return wrap(LayoutType::get(result));
    });
    c.def_prop_ro("tiled_tv_layout_a", [](PyTiledMmaType &self) -> MlirType {
      auto ty = self.toCppType();
      auto mmaAtom = cast<MmaAtomType>(ty.getMmaAtom());
      auto result = tiledMmaGetTiledThrValLayout(mmaAtom, ty.getAtomLayout().getAttr(),
                                                 ty.getPermutation().getAttr(), MmaOperand::A);
      return wrap(LayoutType::get(result));
    });
    c.def_prop_ro("tiled_tv_layout_b", [](PyTiledMmaType &self) -> MlirType {
      auto ty = self.toCppType();
      auto mmaAtom = cast<MmaAtomType>(ty.getMmaAtom());
      auto result = tiledMmaGetTiledThrValLayout(mmaAtom, ty.getAtomLayout().getAttr(),
                                                 ty.getPermutation().getAttr(), MmaOperand::B);
      return wrap(LayoutType::get(result));
    });
    c.def_prop_ro("tiled_tv_layout_c", [](PyTiledMmaType &self) -> MlirType {
      auto ty = self.toCppType();
      auto mmaAtom = cast<MmaAtomType>(ty.getMmaAtom());
      auto result = tiledMmaGetTiledThrValLayout(mmaAtom, ty.getAtomLayout().getAttr(),
                                                 ty.getPermutation().getAttr(), MmaOperand::C);
      return wrap(LayoutType::get(result));
    });
  }
};

// ---------------------------------------------------------------------------
// CopyOpUniversalCopyType
// ---------------------------------------------------------------------------
struct PyCopyOpUniversalCopyType : PyConcreteType<PyCopyOpUniversalCopyType> {
  FLYDSL_REGISTER_TYPE_BINDING(::mlir::fly::CopyOpUniversalCopyType, "CopyOpUniversalCopyType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t bitSize, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          return PyCopyOpUniversalCopyType(context->getRef(),
                                           wrap(CopyOpUniversalCopyType::get(ctx, bitSize)));
        },
        "bitSize"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyOpUniversalCopyType with bit size");
  }
};

// ---------------------------------------------------------------------------
// CopyOpUniversalAtomicType
// ---------------------------------------------------------------------------
struct PyCopyOpUniversalAtomicType : PyConcreteType<PyCopyOpUniversalAtomicType> {
  FLYDSL_REGISTER_TYPE_BINDING(::mlir::fly::CopyOpUniversalAtomicType, "CopyOpUniversalAtomicType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t atomicOp, PyType &valTypeObj, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          auto atomicOpAttr = ::mlir::fly::AtomicOpAttr::get(
              ctx, static_cast<::mlir::fly::AtomicOp>(atomicOp));
          return PyCopyOpUniversalAtomicType(
              context->getRef(),
              wrap(CopyOpUniversalAtomicType::get(atomicOpAttr, unwrap(valTypeObj))));
        },
        "atomic_op"_a, "val_type"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyOpUniversalAtomicType with atomic op and value type");
  }
};

// ---------------------------------------------------------------------------
// MmaOpUniversalFMAType
// ---------------------------------------------------------------------------
struct PyMmaOpUniversalFMAType : PyConcreteType<PyMmaOpUniversalFMAType> {
  FLYDSL_REGISTER_TYPE_BINDING(::mlir::fly::MmaOpUniversalFMAType, "MmaOpUniversalFMAType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &elemTyObj, DefaultingPyMlirContext context) {
          return PyMmaOpUniversalFMAType(context->getRef(),
                                         wrap(MmaOpUniversalFMAType::get(unwrap(elemTyObj))));
        },
        "elem_ty"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a MmaOpUniversalFMAType with element type");

    c.def_prop_ro("elem_ty", [](PyMmaOpUniversalFMAType &self) -> MlirType {
      return wrap(self.toCppType().getElemTy());
    });
  }
};

} // namespace fly
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

// =============================================================================
// Module definition
// =============================================================================

NB_MODULE(_mlirDialectsFly, m) {
  m.doc() = "MLIR Python FlyDSL Extension";

  // -------------------------------------------------------------------------
  // DLTensorAdaptor (standalone, not an MLIR type)
  // -------------------------------------------------------------------------
  using DLTensorAdaptor = utils::DLTensorAdaptor;

  nb::class_<DLTensorAdaptor>(m, "DLTensorAdaptor")
      .def(nb::init<nb::object, std::optional<int32_t>, bool>(), "dlpack_capsule"_a,
           "alignment"_a = nb::none(), "use_32bit_stride"_a = false,
           "Create a DLTensorAdaptor from a DLPack capsule. "
           "If alignment is None, defaults to element size in bytes (minimum "
           "1). ")
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

  // -------------------------------------------------------------------------
  // Module-level helper functions
  // -------------------------------------------------------------------------
  m.def(
      "infer_int_tuple_type",
      [](nb::handle int_or_tuple, MlirContext context) {
        MLIRContext *ctx = unwrap(context);
        IntTupleAttrBuilder builder{ctx};
        auto attr = builder(int_or_tuple);
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
  m.def("has_none", &has_none, "int_or_tuple"_a,
        nb::sig("def has_none(int_or_tuple: " MAKE_MLIR_PYTHON_QUALNAME("ir.Value") ") -> bool"));

  // -------------------------------------------------------------------------
  // Bind Fly dialect types (PyConcreteType pattern)
  // -------------------------------------------------------------------------
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyIntTupleType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyTileType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyLayoutType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PySwizzleType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyComposedLayoutType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyPointerType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyMemRefType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyCoordTensorType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyCopyAtomType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyMmaAtomType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyTiledCopyType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyTiledMmaType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyCopyOpUniversalCopyType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyCopyOpUniversalAtomicType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyMmaOpUniversalFMAType::bind(m);

  m.def(
      "set_llvm_option_bool",
      [](const std::string &name, bool value) -> bool {
        bool oldValue = false;
        int rc = flydslSetLLVMOptionBool(name.c_str(), value, &oldValue);
        if (rc == 1)
          throw std::runtime_error("Unknown LLVM option: " + name);
        if (rc == 2)
          throw std::runtime_error("LLVM option '" + name + "' is not a bool option");
        return oldValue;
      },
      "name"_a, "value"_a, "Set an LLVM bool cl::opt at runtime; returns the previous value.");

  m.def(
      "set_llvm_option_int",
      [](const std::string &name, int value) -> int {
        int oldValue = 0;
        int rc = flydslSetLLVMOptionInt(name.c_str(), value, &oldValue);
        if (rc == 1)
          throw std::runtime_error("Unknown LLVM option: " + name);
        if (rc == 2)
          throw std::runtime_error("LLVM option '" + name + "' is not an int option");
        return oldValue;
      },
      "name"_a, "value"_a, "Set an LLVM int cl::opt at runtime; returns the previous value.");

  m.def(
      "set_llvm_option_str",
      [](const std::string &name, const std::string &value) -> std::string {
        char *oldValue = nullptr;
        int rc = flydslSetLLVMOptionStr(name.c_str(), value.c_str(), &oldValue);
        if (rc == 1)
          throw std::runtime_error("Unknown LLVM option: " + name);
        if (rc == 2)
          throw std::runtime_error("LLVM option '" + name + "' is not a string option");
        std::string result(oldValue ? oldValue : "");
        flydslFreeLLVMOptionStr(oldValue);
        return result;
      },
      "name"_a, "value"_a, "Set an LLVM string cl::opt at runtime; returns the previous value.");
}
