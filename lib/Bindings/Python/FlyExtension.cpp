// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

#include "BindingUtils.h"
#include "DLTensorAdaptor.h"

#include "HostObject/host_object.h"
#include "LlvmConfig/llvm.h"

#include <cstdint>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace ::mlir;
using namespace ::mlir::fly;

namespace {
namespace detail {

IntTupleAttr getProfileAttrFromLayoutAttr(Attribute layout) {
  if (auto layoutAttr = dyn_cast<LayoutAttr>(layout))
    return layoutAttr.getShape();
  if (auto composedAttr = dyn_cast<ComposedLayoutAttr>(layout))
    return getProfileAttrFromLayoutAttr(composedAttr.getOuter());
  throw std::invalid_argument("expected LayoutAttr or ComposedLayoutAttr");
}

IntTupleAttr getProfileAttrFromType(Type ty) {
  if (auto intTupleTy = dyn_cast<IntTupleType>(ty))
    return intTupleTy.getAttr();
  if (auto layoutTy = dyn_cast<LayoutType>(ty))
    return layoutTy.getAttr().getShape();
  if (auto composedTy = dyn_cast<ComposedLayoutType>(ty))
    return getProfileAttrFromLayoutAttr(composedTy.getAttr());
  if (auto memrefTy = dyn_cast<fly::MemRefType>(ty))
    return getProfileAttrFromLayoutAttr(memrefTy.getLayout());
  if (auto coordTensorTy = dyn_cast<CoordTensorType>(ty))
    return getProfileAttrFromLayoutAttr(coordTensorTy.getLayout());
  throw std::invalid_argument(
      "expected IntTupleType, LayoutType, ComposedLayoutType, MemRefType or CoordTensorType");
}

} // namespace detail

struct PyBasis {
  nb::object value;
  std::vector<int32_t> modes;
};

bool isPythonRatio(nb::handle value) {
  return nb::isinstance(value, nb::module_::import_("fractions").attr("Fraction"));
}

LeafAttr staticIntFromPython(MLIRContext *ctx, nb::handle value) {
  if (!isPythonRatio(value))
    return LeafAttr::getStatic(ctx, nb::cast<int64_t>(value));
  // A Fraction is already reduced with a positive denominator, so the int32
  // range of its parts is the only representability condition.
  int32_t numerator, denominator;
  if (!nb::try_cast(value.attr("numerator"), numerator) ||
      !nb::try_cast(value.attr("denominator"), denominator))
    throw std::invalid_argument("ratio is outside the representable range");
  return LeafAttr::get(ctx, core::Leaf<core::Trait>::getRatio(numerator, denominator));
}

nb::object staticIntToPython(LeafAttr value) {
  if (value.isNone())
    return nb::none();
  if (value.isRatio())
    return nb::module_::import_("fractions")
        .attr("Fraction")(value.getNumerator(), value.getDenominator());
  return nb::cast(value.getValue());
}

struct IntTupleAttrBuilder {
  MLIRContext *ctx;
  std::vector<nb::handle> dyncElems{};

  IntTupleAttrBuilder(MLIRContext *ctx) : ctx(ctx) {}

  void clear() { dyncElems.clear(); }

  /// The leaf a dynamic value binds: an i32 or i64, or, where `fastDivMod`
  /// allows, the handle of a fast-divmod divisor.
  LeafAttr dynamicIntFromValue(nb::handle handle, const char *what, bool fastDivMod = false) {
    auto typeName = [&] { return std::string(nb::str(nb::type_name(handle.type())).c_str()); };
    if (!nb::hasattr(handle, MLIR_PYTHON_CAPI_PTR_ATTR))
      throw std::invalid_argument(std::string(what) +
                                  " must be an i32/i64 MLIR Value, got: " + typeName());
    auto capsule = nb::cast<nb::capsule>(handle.attr(MLIR_PYTHON_CAPI_PTR_ATTR));
    MlirValue mlirVal = mlirPythonCapsuleToValue(capsule.ptr());
    if (mlirValueIsNull(mlirVal))
      throw std::invalid_argument(std::string(what) + " must be an MLIR Value, got: " + typeName());
    if (auto divisor = dyn_cast<FastDivModType>(unwrap(mlirVal).getType()); divisor && fastDivMod) {
      dyncElems.push_back(handle);
      auto leaf = LeafAttr::getDynamic(ctx, divisor.getDivisorType().getWidth());
      return LeafAttr::get(ctx, leaf.getRef().withFastDivMod());
    }
    auto intTy = dyn_cast<IntegerType>(unwrap(mlirVal).getType());
    if (!intTy || !intTy.isSignless() || (intTy.getWidth() != 32 && intTy.getWidth() != 64))
      throw std::invalid_argument(std::string(what) +
                                  " must be an i32 or i64 value, got: " + typeName());
    dyncElems.push_back(handle);
    return LeafAttr::getDynamic(ctx, intTy.getWidth());
  }

  using Tuple = core::IntTuple<core::Trait>;

  Tuple buildBasisLeaf(const PyBasis &basis) {
    if (basis.modes.size() > core::kMaxBasisModes)
      throw std::invalid_argument("Basis exceeds the maximum mode depth");
    if (basis.modes.empty())
      throw std::invalid_argument("Basis must have at least one mode");
    SmallVector<int32_t> modes(basis.modes.begin(), basis.modes.end());
    for (int32_t m : modes) {
      if (m < 0 || m > UINT8_MAX)
        throw std::invalid_argument("Basis mode must be non-negative, got " + std::to_string(m));
    }
    // A static coefficient is a python int; a dynamic one is an i32/i64 MLIR Value.
    nb::handle value = basis.value;
    LeafAttr scale = PyLong_Check(value.ptr()) || isPythonRatio(value)
                         ? staticIntFromPython(ctx, value)
                         : dynamicIntFromValue(value, "Basis coefficient");
    auto leaf = LeafAttr::get(scale, modes);
    if (!leaf)
      throw std::invalid_argument("invalid Basis coefficient or modes");
    return Tuple::fromLeaf(leaf.getRef());
  }

  Tuple build(nb::handle args) {
    if (PyTuple_Check(args.ptr())) {
      SmallVector<Tuple> elements;
      for (auto item : args)
        elements.push_back(build(item));
      return core::make_tuple(elements);
    }
    if (PyLong_Check(args.ptr()) || isPythonRatio(args))
      return Tuple::fromLeaf(staticIntFromPython(ctx, args).getRef());
    if (args.is_none())
      return Tuple::fromLeaf(core::Leaf<core::Trait>::getNone());
    if (nb::isinstance<PyBasis>(args))
      return buildBasisLeaf(nb::cast<const PyBasis &>(args));
    return Tuple::fromLeaf(
        dynamicIntFromValue(args, "Dynamic int_tuple leaf", /*fastDivMod=*/true).getRef());
  }

  IntTupleAttr operator()(nb::handle args) {
    auto attr = IntTupleAttr::get(ctx, build(args));
    if (!attr)
      throw std::invalid_argument("int_tuple exceeds the tree capacity");
    return attr;
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
    return llvm::any_of(t.getRef().leaves(), [](auto leaf) { return leaf.isNone(); });
  throw std::invalid_argument("has_none() expected IntTupleType");
}

bool isProfileCongruent(MlirValue lhs, MlirValue rhs) {
  Type lhsTy = unwrap(lhs).getType();
  Type rhsTy = unwrap(rhs).getType();
  auto lhsProfile = detail::getProfileAttrFromType(lhsTy);
  auto rhsProfile = detail::getProfileAttrFromType(rhsTy);
  return core::congruent(lhsProfile.getRef(), rhsProfile.getRef());
}

bool isProfileWeaklyCongruent(MlirValue lhs, MlirValue rhs) {
  Type lhsTy = unwrap(lhs).getType();
  Type rhsTy = unwrap(rhs).getType();
  auto lhsProfile = detail::getProfileAttrFromType(lhsTy);
  auto rhsProfile = detail::getProfileAttrFromType(rhsTy);
  return core::weakly_congruent(lhsProfile.getRef(), rhsProfile.getRef());
}

// Setter accepts either an int (interpreted as a Fly_AddressSpace enum case)
// or any AddressSpaceAttr-compatible MLIR Attribute (e.g. `#fly_rocdl.buffer_desc`).
Attribute getAddressSpaceFromObj(MLIRContext *ctx, nb::object obj, AddressSpace defaultAS) {
  if (obj.is_none())
    return AddressSpaceAttr::get(ctx, defaultAS);
  if (nb::hasattr(obj, MLIR_PYTHON_CAPI_PTR_ATTR)) {
    auto capsule = nb::cast<nb::capsule>(obj.attr(MLIR_PYTHON_CAPI_PTR_ATTR));
    MlirAttribute mlirAttr = mlirPythonCapsuleToAttribute(capsule.ptr());
    Attribute attr = unwrap(mlirAttr);
    if (!attr)
      throw std::invalid_argument("address_space: invalid Attribute");
    return attr;
  }
  if (PyLong_Check(obj.ptr())) {
    int32_t addrInt = nb::cast<int32_t>(obj);
    if (addrInt < static_cast<int32_t>(AddressSpace::Generic) ||
        addrInt > static_cast<int32_t>(AddressSpace::Register))
      throw std::invalid_argument("address_space int must be a valid Fly_AddressSpace case");
    return AddressSpaceAttr::get(ctx, static_cast<AddressSpace>(addrInt));
  }
  throw std::invalid_argument(
      "address_space must be int (Fly_AddressSpace case) or an MLIR Attribute");
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
      if (!ty.isLeaf() || !ty.isStatic() || !ty.getAttr().isLeafInt() ||
          ty.getAttr().getLeaf().isRatio())
        throw std::invalid_argument("expected a static integer leaf");
      return ty.getAttr().getLeaf().getValue();
    });
    c.def_prop_ro("is_ratio", [](PyIntTupleType &self) {
      auto attr = self.toCppType().getAttr();
      return attr.isLeafInt() && attr.getLeaf().isRatio();
    });
    c.def_prop_ro("static_value", [](PyIntTupleType &self) {
      auto attr = self.toCppType().getAttr();
      if (!attr.isLeafInt() || !attr.isStatic())
        throw std::invalid_argument("expected a static scalar leaf");
      return staticIntToPython(attr.getLeaf());
    });
    c.def_prop_ro("is_leaf_int",
                  [](PyIntTupleType &self) { return self.toCppType().getAttr().isLeafInt(); });
    c.def_prop_ro("is_leaf_basis",
                  [](PyIntTupleType &self) { return self.toCppType().getAttr().isLeafBasis(); });
    c.def_prop_ro("get_leaf_as_basis", [](PyIntTupleType &self) -> MlirType {
      if (!self.toCppType().getAttr().isLeafBasis())
        throw std::invalid_argument("expected a basis leaf");
      return wrap(BasisType::get(self.toCppType().getAttr().getLeaf()));
    });
    c.def("at", [](PyIntTupleType &self, int32_t idx) -> MlirType {
      IntTupleType ty = self.toCppType();
      int32_t rank = ty.getAttr().isLeaf() ? 1 : ty.rank();
      if (idx < 0 || idx >= rank)
        throw std::invalid_argument("IntTupleType.at: index out of range");
      return wrap(ty.at(idx));
    });
  }
};

// ---------------------------------------------------------------------------
// BasisType
// ---------------------------------------------------------------------------
struct PyBasisType : PyConcreteType<PyBasisType> {
  FLYDSL_REGISTER_TYPE_BINDING(::mlir::fly::BasisType, "BasisType");

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro("is_static",
                  [](PyBasisType &self) { return self.toCppType().getAttr().isStatic(); });
    c.def_prop_ro("value", [](PyBasisType &self) {
      auto basis = self.toCppType().getAttr();
      if (!basis.isStatic())
        throw std::invalid_argument("basis coefficient is dynamic; no static value");
      return staticIntToPython(basis.getScale());
    });
    c.def_prop_ro("modes", [](PyBasisType &self) {
      auto basis = self.toCppType().getAttr();
      auto modes = basis.getModes();
      return std::vector<int32_t>(modes.begin(), modes.end());
    });
  }
};

// ---------------------------------------------------------------------------
// TileType
// ---------------------------------------------------------------------------
struct PyTileType : PyConcreteType<PyTileType> {
  FLYDSL_REGISTER_TYPE_BINDING(::mlir::fly::TileType, "TileType");

  using Tile = core::Tile<core::Trait>;

  static Tile extractTileMode(nb::handle mode, MLIRContext *ctx) {
    if (mode.is_none())
      return Tile::getNone();
    if (PyLong_Check(mode.ptr()))
      return Tile(staticIntFromPython(ctx, mode).getRef());
    if (PyTuple_Check(mode.ptr())) {
      SmallVector<Tile> nested;
      for (auto item : mode)
        nested.push_back(extractTileMode(nb::handle(item), ctx));
      return Tile::getTuple(nested);
    }
    if (nb::hasattr(mode, MLIR_PYTHON_CAPI_PTR_ATTR)) {
      auto layoutTy = FLYDSL_EXTRACT_TYPE_FROM_NB_HANDLE(::mlir::fly::LayoutType, mode);
      return Tile(layoutTy.getRef());
    }
    throw std::invalid_argument("TileType.get: expected int, None, tuple, or LayoutType");
  }

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](nb::object modeOrModes, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          Tile tile;
          if (nb::isinstance<nb::list>(modeOrModes)) {
            SmallVector<Tile> modes;
            for (auto mode : nb::cast<nb::list>(modeOrModes))
              modes.push_back(extractTileMode(nb::handle(mode), ctx));
            tile = Tile::getTuple(modes);
          } else {
            tile = extractTileMode(modeOrModes, ctx);
          }
          auto tileAttr = TileAttr::get(ctx, std::move(tile));
          if (!tileAttr)
            throw std::invalid_argument("TileType.get: invalid tile");
          return PyTileType(context->getRef(), wrap(TileType::get(tileAttr)));
        },
        "modes"_a.none(), nb::kw_only(), "context"_a = nb::none(),
        "Create a TileType from a list of modes or a single mode (leaf tile)");

    c.def_prop_ro("rank", [](PyTileType &self) { return self.toCppType().rank(); });
    c.def_prop_ro("is_leaf", [](PyTileType &self) { return self.toCppType().getAttr().isLeaf(); });
    c.def("at", [](PyTileType &self, int32_t idx) -> MlirType {
      TileType ty = self.toCppType();
      if (idx < 0 || idx >= ty.rank())
        throw std::invalid_argument("TileType.at: index out of range");
      MLIRContext *ctx = ty.getContext();
      auto &tile = ty.getRef();
      Tile mode = tile.isLeaf() ? tile : tile.at(idx);
      if (mode.isLayout())
        return wrap(LayoutType::get(ctx, mode.getLayout()));
      if (mode.isNone())
        return wrap(IntTupleType::get(IntTupleAttr::getLeafNone(ctx)));
      if (mode.isScalar())
        return wrap(
            IntTupleType::get(ctx, core::IntTuple<core::Trait>::fromLeaf(mode.getScalar())));
      return wrap(TileType::get(ctx, mode));
    });
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
          if (!layoutAttr)
            throw std::invalid_argument("LayoutType.get: shape and stride must be congruent");
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
          else if (auto coordSwizzle = dyn_cast<CoordSwizzleType>(innerTy))
            innerAttr = coordSwizzle.getAttr();
          else
            throw std::invalid_argument("inner must be a LayoutType, ComposedLayoutType, "
                                        "SwizzleType, or CoordSwizzleType");

          IntTupleAttrBuilder builder{ctx};
          auto offsetAttr = getIntTupleAttrFromHandle(offset, builder);
          Attribute outerAttr;
          if (auto outerLayout = dyn_cast<LayoutType>(unwrap(outerObj)))
            outerAttr = outerLayout.getAttr();
          else if (auto outerComposed = dyn_cast<ComposedLayoutType>(unwrap(outerObj)))
            outerAttr = outerComposed.getAttr();
          else
            throw std::invalid_argument("outer must be a LayoutType or ComposedLayoutType");

          auto attr = ComposedLayoutAttr::get(innerAttr, offsetAttr, outerAttr);
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
      if (auto coordSwizzle = dyn_cast<CoordSwizzleAttr>(innerAttr))
        return wrap(CoordSwizzleType::get(coordSwizzle));
      throw std::invalid_argument(
          "Expected LayoutAttr, ComposedLayoutAttr, SwizzleAttr, or CoordSwizzleAttr");
    });
    c.def_prop_ro("offset", [](PyComposedLayoutType &self) -> MlirType {
      return wrap(IntTupleType::get(self.toCppType().getAttr().getOffset()));
    });
    c.def_prop_ro("outer", [](PyComposedLayoutType &self) -> MlirType {
      Attribute outerAttr = self.toCppType().getAttr().getOuter();
      if (auto layout = dyn_cast<LayoutAttr>(outerAttr))
        return wrap(LayoutType::get(layout));
      if (auto composed = dyn_cast<ComposedLayoutAttr>(outerAttr))
        return wrap(ComposedLayoutType::get(composed));
      throw std::invalid_argument("Expected LayoutAttr or ComposedLayoutAttr");
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
          if (!attr)
            throw std::invalid_argument("invalid Swizzle parameters");
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
// CoordSwizzleType
// ---------------------------------------------------------------------------
struct PyCoordSwizzleType : PyConcreteType<PyCoordSwizzleType> {
  FLYDSL_REGISTER_TYPE_BINDING(::mlir::fly::CoordSwizzleType, "CoordSwizzleType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t mask, int32_t baseRow, std::vector<int32_t> modeRow, int32_t baseCol,
           std::vector<int32_t> modeCol, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          auto attr = CoordSwizzleAttr::get(ctx, mask, baseRow, modeRow, baseCol, modeCol);
          if (!attr)
            throw std::invalid_argument("invalid CoordSwizzle parameters");
          return PyCoordSwizzleType(context->getRef(), wrap(CoordSwizzleType::get(attr)));
        },
        "mask"_a, "base_row"_a, "mode_row"_a, "base_col"_a, "mode_col"_a, nb::kw_only(),
        "context"_a = nb::none(), "Create a CoordSwizzleType");

    c.def_prop_ro("mask",
                  [](PyCoordSwizzleType &self) { return self.toCppType().getAttr().getMask(); });
    c.def_prop_ro("base_row",
                  [](PyCoordSwizzleType &self) { return self.toCppType().getAttr().getBaseRow(); });
    c.def_prop_ro("mode_row", [](PyCoordSwizzleType &self) {
      auto modes = self.toCppType().getAttr().getModeRow();
      return std::vector<int32_t>(modes.begin(), modes.end());
    });
    c.def_prop_ro("base_col",
                  [](PyCoordSwizzleType &self) { return self.toCppType().getAttr().getBaseCol(); });
    c.def_prop_ro("mode_col", [](PyCoordSwizzleType &self) {
      auto modes = self.toCppType().getAttr().getModeCol();
      return std::vector<int32_t>(modes.begin(), modes.end());
    });
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
        [](PyType &elemTyObj, nb::object addressSpace, std::optional<int32_t> alignment,
           DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          auto elemType = unwrap(elemTyObj);

          Attribute addrAttr = getAddressSpaceFromObj(ctx, addressSpace, AddressSpace::Global);

          int32_t alignSize =
              alignment.value_or(AlignAttr::getTrivialAlignment(elemType).getAlignment());
          int32_t elemByte = (elemType.getIntOrFloatBitWidth() + 7) / 8;
          if (alignSize <= 0 || alignSize % elemByte != 0)
            throw std::invalid_argument(
                "alignment must be a positive multiple of element byte size (" +
                std::to_string(elemByte) + "), got " + std::to_string(alignSize));

          return PyPointerType(
              context->getRef(),
              wrap(PointerType::get(elemType, addrAttr, AlignAttr::get(ctx, alignSize))));
        },
        "elem_ty"_a, "address_space"_a = nb::none(), "alignment"_a = nb::none(), nb::kw_only(),
        "context"_a = nb::none(),
        "Create a PointerType. address_space accepts an int (Fly_AddressSpace "
        "case) or a target-specific MLIR Attribute (e.g. "
        "`#fly_rocdl.buffer_desc`).");

    c.def_prop_ro("element_type", [](PyPointerType &self) -> MlirType {
      return wrap(self.toCppType().getElemTy());
    });
    c.def_prop_ro("address_space", [](PyPointerType &self) -> nb::typed<nb::object, PyAttribute> {
      return PyAttribute(self.getContext(), wrap(self.toCppType().getAddressSpace()))
          .maybeDownCast();
    });
    c.def_prop_ro("bit_alignment", [](PyPointerType &self) -> int64_t {
      return self.toCppType().getAlignment().getBitAlignment();
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
        [](PyType &elemTyObj, PyType &layoutObj, nb::object addressSpace,
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

          Attribute addrAttr = getAddressSpaceFromObj(ctx, addressSpace, AddressSpace::Register);

          auto elemType = unwrap(elemTyObj);
          int32_t alignSize =
              alignment.value_or(AlignAttr::getTrivialAlignment(elemType).getAlignment());
          int32_t elemByte = (elemType.getIntOrFloatBitWidth() + 7) / 8;
          if (alignSize <= 0 || alignSize % elemByte != 0)
            throw std::invalid_argument(
                "alignment must be a positive multiple of element byte size (" +
                std::to_string(elemByte) + "), got " + std::to_string(alignSize));

          return PyMemRefType(context->getRef(),
                              wrap(::mlir::fly::MemRefType::get(elemType, addrAttr, layoutAttr,
                                                                AlignAttr::get(ctx, alignSize))));
        },
        "elem_ty"_a, "layout"_a, "address_space"_a = nb::none(), "alignment"_a = nb::none(),
        nb::kw_only(), "context"_a = nb::none(),
        "Create a MemRefType. address_space accepts an int (Fly_AddressSpace "
        "case) or a target-specific MLIR Attribute (e.g. "
        "`#fly_rocdl.buffer_desc`).");

    // Build a layout-dynamic MemRefType from per-dim *encoded* values each
    // entry is ``v >= 0`` -> static size/stride ``v``; ``v < 0`` -> dynamic dim
    // with divisibility ``-v``. Shape dynamic leaves are always 32-bit; stride
    // dynamic leaves follow ``use_32bit_stride``.  Address space is global.
    c.def_static(
        "get",
        [](PyType &elemTyObj, const std::vector<int64_t> &shapeEnc,
           const std::vector<int64_t> &strideEnc, bool use32BitStride, int32_t alignment,
           DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          Type elemType = unwrap(elemTyObj);
          int n = static_cast<int>(shapeEnc.size());
          using Tuple = core::IntTuple<core::Trait>;
          auto leaf = [&](int64_t v, bool i32) -> Tuple {
            if (v >= 0)
              return Tuple::getSInt(v);
            return Tuple::fromLeaf(core::Leaf<core::Trait>::getDInt(
                i32 ? core::LogWidth::I32 : core::LogWidth::I64, static_cast<int32_t>(-v)));
          };
          SmallVector<Tuple> sh, st;
          for (int i = 0; i < n; ++i) {
            sh.push_back(leaf(shapeEnc[i], /*i32=*/true));
            st.push_back(leaf(strideEnc[i], use32BitStride));
          }
          auto shape = n == 1 ? sh.front() : core::make_tuple(sh);
          auto stride = n == 1 ? st.front() : core::make_tuple(st);
          LayoutAttr layoutAttr =
              LayoutAttr::get(ctx, core::Layout<core::Trait>(std::move(shape), std::move(stride)));
          AddressSpaceAttr addrSpaceAttr = AddressSpaceAttr::get(ctx, AddressSpace::Global);
          AlignAttr alignAttr = AlignAttr::get(ctx, alignment);
          return PyMemRefType(
              context->getRef(),
              wrap(::mlir::fly::MemRefType::get(elemType, addrSpaceAttr, layoutAttr, alignAttr)));
        },
        "elem_ty"_a, "shape_enc"_a, "stride_enc"_a, "use_32bit_stride"_a, "alignment"_a,
        nb::kw_only(), "context"_a = nb::none(),
        "Build a layout-dynamic MemRefType from encoded per-dim values: v>=0 static "
        "size/stride, v<0 dynamic with divisibility -v (same encoding as the cache "
        "signature). Python owns the layout state.");

    c.def_prop_ro("element_type", [](PyMemRefType &self) -> MlirType {
      return wrap(self.toCppType().getElemTy());
    });
    c.def_prop_ro("layout", [](PyMemRefType &self) -> MlirType {
      Attribute layout = self.toCppType().getLayout();
      if (auto la = dyn_cast<LayoutAttr>(layout))
        return wrap(LayoutType::get(la));
      return wrap(ComposedLayoutType::get(cast<ComposedLayoutAttr>(layout)));
    });
    c.def_prop_ro("address_space", [](PyMemRefType &self) -> nb::typed<nb::object, PyAttribute> {
      return PyAttribute(self.getContext(), wrap(self.toCppType().getAddressSpace()))
          .maybeDownCast();
    });
    c.def_prop_ro("bit_alignment", [](PyMemRefType &self) -> int64_t {
      return self.toCppType().getAlignment().getBitAlignment();
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
      if (!ty.isValidRef())
        throw std::invalid_argument("invalid tiled copy");
      return wrap(LayoutType::get(LayoutAttr::get(ty.getContext(), ty.getRef().get_layoutS_TV())));
    });
    c.def_prop_ro("tiled_tv_layout_dst", [](PyTiledCopyType &self) -> MlirType {
      auto ty = self.toCppType();
      if (!ty.isValidRef())
        throw std::invalid_argument("invalid tiled copy");
      return wrap(LayoutType::get(LayoutAttr::get(ty.getContext(), ty.getRef().get_layoutD_TV())));
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
      auto tiled = self.toCppType().getRef();
      auto size =
          core::make_tuple(tiled.tile_size_mnk(0), tiled.tile_size_mnk(1), tiled.tile_size_mnk(2));
      return wrap(IntTupleType::get(IntTupleAttr::get(self.toCppType().getContext(), size)));
    });
    c.def_prop_ro("thr_layout_vmnk", [](PyTiledMmaType &self) -> MlirType {
      auto ty = self.toCppType();
      return wrap(
          LayoutType::get(LayoutAttr::get(ty.getContext(), ty.getRef().get_thr_layout_vmnk())));
    });
    c.def_prop_ro("tiled_tv_layout_a", [](PyTiledMmaType &self) -> MlirType {
      auto ty = self.toCppType();
      return wrap(LayoutType::get(LayoutAttr::get(ty.getContext(), ty.getRef().get_layoutA_TV())));
    });
    c.def_prop_ro("tiled_tv_layout_b", [](PyTiledMmaType &self) -> MlirType {
      auto ty = self.toCppType();
      return wrap(LayoutType::get(LayoutAttr::get(ty.getContext(), ty.getRef().get_layoutB_TV())));
    });
    c.def_prop_ro("tiled_tv_layout_c", [](PyTiledMmaType &self) -> MlirType {
      auto ty = self.toCppType();
      return wrap(LayoutType::get(LayoutAttr::get(ty.getContext(), ty.getRef().get_layoutC_TV())));
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
        [](int32_t atomicOp, PyType &valTypeObj, const std::string &syncscope,
           DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          auto atomicOpAttr =
              ::mlir::fly::AtomicOpAttr::get(ctx, static_cast<::mlir::fly::AtomicOp>(atomicOp));
          return PyCopyOpUniversalAtomicType(
              context->getRef(),
              wrap(CopyOpUniversalAtomicType::get(atomicOpAttr, unwrap(valTypeObj), syncscope)));
        },
        "atomic_op"_a, "val_type"_a, "syncscope"_a = std::string(), nb::kw_only(),
        "context"_a = nb::none(),
        "Create a CopyOpUniversalAtomicType with atomic op, value type and sync scope");
  }
};

// ---------------------------------------------------------------------------
// FastDivModType
// ---------------------------------------------------------------------------
struct PyFastDivModType : PyConcreteType<PyFastDivModType> {
  FLYDSL_REGISTER_TYPE_BINDING(::mlir::fly::FastDivModType, "FastDivModType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &divisorTypeObj, DefaultingPyMlirContext context) {
          auto divisorType = dyn_cast<IntegerType>(unwrap(divisorTypeObj));
          if (!divisorType || !divisorType.isSignless() ||
              (divisorType.getWidth() != 32 && divisorType.getWidth() != 64))
            throw std::invalid_argument("FastDivModType needs an i32 or i64 divisor type");
          return PyFastDivModType(context->getRef(),
                                  wrap(FastDivModType::get(divisorType.getContext(), divisorType)));
        },
        "divisor_type"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create the FastDivModType of an i32 or i64 divisor");

    c.def_prop_ro("divisor_type", [](PyFastDivModType &self) -> MlirType {
      return wrap(self.toCppType().getDivisorType());
    });
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
      .def(nb::init<nb::object>(), "dlpack_capsule"_a,
           "Create a DLTensorAdaptor from a DLPack capsule.")
      .def_prop_ro("shape", &DLTensorAdaptor::getShape, "Get tensor shape as tuple")
      .def_prop_ro("stride", &DLTensorAdaptor::getStride, "Get tensor stride as tuple")
      .def_prop_ro("data_ptr", &DLTensorAdaptor::getDataPtr, "Get data pointer as int64")
      .def_prop_ro("address_space", &DLTensorAdaptor::getAddressSpace,
                   "Get address space (0=host, 1=device)")
      .def_prop_ro(
          "dtype", [](DLTensorAdaptor &self) { return wrap(self.getDtype()); },
          "The dtype as an MLIR element type (ir Type); requires an active MLIR context")
      .def_prop_ro("dtype_id", &DLTensorAdaptor::getDtypeId,
                   "Context-free dtype id (code, bits, lanes) for use as a cache discriminator")
      .def_prop_ro("element_bits", &DLTensorAdaptor::getElementBits,
                   "Element width in bits (bits * lanes), at sub-byte granularity")
      .def("size_in_bytes", &DLTensorAdaptor::getSizeInBytes, "Get total size in bytes");

  // -------------------------------------------------------------------------
  // Internal Basis leaf marker: the Python Basis is converted into this at
  // infer_int_tuple_type time so the int-tuple builder recognizes a basis leaf
  // via nb::isinstance.
  // -------------------------------------------------------------------------
  nb::class_<PyBasis>(m, "_Basis")
      .def(
          "__init__",
          [](PyBasis *self, nb::object value, std::vector<int32_t> modes) {
            new (self) PyBasis{std::move(value), std::move(modes)};
          },
          "value"_a, "modes"_a)
      .def_ro("value", &PyBasis::value)
      .def_ro("modes", &PyBasis::modes);

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
  m.def("is_profile_congruent", &isProfileCongruent, "lhs"_a, "rhs"_a,
        nb::sig("def is_profile_congruent(lhs: " MAKE_MLIR_PYTHON_QUALNAME(
            "ir.Value") ", rhs: " MAKE_MLIR_PYTHON_QUALNAME("ir.Value") ") -> bool"));
  m.def("is_profile_weakly_congruent", &isProfileWeaklyCongruent, "lhs"_a, "rhs"_a,
        nb::sig("def is_profile_weakly_congruent(lhs: " MAKE_MLIR_PYTHON_QUALNAME(
            "ir.Value") ", rhs: " MAKE_MLIR_PYTHON_QUALNAME("ir.Value") ") -> bool"));

  // -------------------------------------------------------------------------
  // Bind Fly dialect types (PyConcreteType pattern)
  // -------------------------------------------------------------------------
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyIntTupleType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyBasisType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyTileType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyLayoutType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PySwizzleType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyCoordSwizzleType::bind(m);
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
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyFastDivModType::bind(m);

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

  static constexpr auto appendTo = [](MlirStringRef str, void *userData) {
    static_cast<std::string *>(userData)->append(str.data, str.length);
  };

  m.def(
      "_host_target_triple",
      []() {
        std::string triple;
        flydslHostTargetTriple(appendTo, &triple);
        return triple;
      },
      "Normalized target triple of the host process.");

  m.def(
      "_emit_host_object",
      [](MlirOperation module, int optLevel) {
        std::pair<std::string, std::string> out;
        auto appendObject = [](MlirStringRef str, void *userData) {
          appendTo(str, &static_cast<std::pair<std::string, std::string> *>(userData)->first);
        };
        auto appendError = [](MlirStringRef str, void *userData) {
          appendTo(str, &static_cast<std::pair<std::string, std::string> *>(userData)->second);
        };
        if (mlirLogicalResultIsFailure(
                flydslEmitHostObject(module, optLevel, appendObject, appendError, &out)))
          throw std::runtime_error("host object emission failed: " + out.second);
        return nb::bytes(out.first.data(), out.first.size());
      },
      "module"_a, "opt_level"_a = 2,
      "Emit a position-independent host object for an LLVM-dialect module; returns its bytes.");
}
