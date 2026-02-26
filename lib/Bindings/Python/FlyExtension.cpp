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
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>

#include "flydsl-c/FlyDialect.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntUtils.h"

#include "DLTensorAdaptor.h"

#include <cstdint>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

struct IntTupleAttrBuilder {
  ::mlir::MLIRContext *ctx;
  std::vector<nb::handle> dyncElems{};

  IntTupleAttrBuilder(::mlir::MLIRContext *ctx) : ctx(ctx) {}

  void clear() { dyncElems.clear(); }

  ::mlir::fly::IntTupleAttr operator()(nb::handle args) {
    if (PyTuple_Check(args.ptr())) {
      ::mlir::SmallVector<::mlir::Attribute> elements;
      for (auto item : args) {
        elements.push_back((*this)(item));
      }
      return ::mlir::fly::IntTupleAttr::get(::mlir::ArrayAttr::get(ctx, elements));
    } else if (PyLong_Check(args.ptr())) {
      int32_t cInt = PyLong_AsLong(args.ptr());
      return ::mlir::fly::IntTupleAttr::get(::mlir::fly::IntAttr::getStatic(ctx, cInt));
    } else if (args.is_none()) {
      return ::mlir::fly::IntTupleAttr::getLeafNone(ctx);
    } else {
      if (!nb::hasattr(args, "_CAPIPtr")) {
        throw std::invalid_argument("Expected I32, got: " +
                                    std::string(nb::str(nb::type_name(args)).c_str()));
      }
      dyncElems.push_back(args);
      return ::mlir::fly::IntTupleAttr::get(::mlir::fly::IntAttr::getDynamic(ctx));
    }
  }
};

int32_t rank(MlirValue int_or_tuple) {
  ::mlir::Value val = unwrap(int_or_tuple);
  ::mlir::Type ty = val.getType();
  if (auto t = ::mlir::dyn_cast<::mlir::fly::IntTupleType>(ty))
    return t.getAttr().rank();
  if (auto t = ::mlir::dyn_cast<::mlir::fly::LayoutType>(ty))
    return t.getAttr().rank();
  if (auto t = ::mlir::dyn_cast<::mlir::fly::ComposedLayoutType>(ty))
    return t.getAttr().rank();
  if (auto t = ::mlir::dyn_cast<::mlir::fly::CoordTensorType>(ty))
    return t.getLayout().rank();
  if (auto t = ::mlir::dyn_cast<::mlir::fly::MemRefType>(ty))
    return t.getLayout().rank();
  throw std::invalid_argument("Unsupported type for rank()");
}

int32_t depth(MlirValue int_or_tuple) {
  ::mlir::Value val = unwrap(int_or_tuple);
  ::mlir::Type ty = val.getType();
  if (auto t = ::mlir::dyn_cast<::mlir::fly::IntTupleType>(ty))
    return t.getAttr().depth();
  if (auto t = ::mlir::dyn_cast<::mlir::fly::LayoutType>(ty))
    return t.getAttr().depth();
  if (auto t = ::mlir::dyn_cast<::mlir::fly::ComposedLayoutType>(ty))
    return t.getAttr().depth();
  if (auto t = ::mlir::dyn_cast<::mlir::fly::CoordTensorType>(ty))
    return t.getLayout().depth();
  if (auto t = ::mlir::dyn_cast<::mlir::fly::MemRefType>(ty))
    return t.getLayout().depth();
  throw std::invalid_argument("Unsupported type for depth()");
}

/// Convert nb::handle (Python int|tuple|IntTupleType) to IntTupleAttr.
::mlir::fly::IntTupleAttr toIntTupleAttr(nb::handle h, ::mlir::MLIRContext *ctx) {
  if (nb::hasattr(h, MLIR_PYTHON_CAPI_PTR_ATTR)) {
    auto capsule = nb::cast<nb::capsule>(h.attr(MLIR_PYTHON_CAPI_PTR_ATTR));
    MlirType mlirTy = mlirPythonCapsuleToType(capsule.ptr());
    auto intTupleType = ::mlir::dyn_cast<::mlir::fly::IntTupleType>(unwrap(mlirTy));
    if (!intTupleType)
      throw std::invalid_argument("Expected IntTupleType, got other MlirType");
    return intTupleType.getAttr();
  }
  IntTupleAttrBuilder builder{ctx};
  return builder(h);
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
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFlyIntTupleType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = mlirFlyIntTupleTypeGetTypeID;
  static constexpr const char *pyClassName = "IntTupleType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](nb::handle int_or_tuple, DefaultingPyMlirContext context) {
          ::mlir::MLIRContext *ctx = unwrap(context.get()->get());
          IntTupleAttrBuilder builder{ctx};
          auto attr = builder(int_or_tuple);
          return PyIntTupleType(context->getRef(), wrap(::mlir::fly::IntTupleType::get(attr)));
        },
        "int_or_tuple"_a, nb::kw_only(), "context"_a = nb::none(),
        // clang-format off
        nb::sig("def get(int_or_tuple, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None)"),
        // clang-format on
        "Create an IntTupleType from Python int or tuple");

    c.def_prop_ro("rank",
                  [](PyIntTupleType &self) -> int32_t { return mlirFlyIntTupleTypeGetRank(self); });
    c.def_prop_ro(
        "depth", [](PyIntTupleType &self) -> int32_t { return mlirFlyIntTupleTypeGetDepth(self); });
    c.def_prop_ro("is_leaf",
                  [](PyIntTupleType &self) -> bool { return mlirFlyIntTupleTypeIsLeaf(self); });
    c.def_prop_ro("is_static",
                  [](PyIntTupleType &self) -> bool { return mlirFlyIntTupleTypeIsStatic(self); });
    c.def_prop_ro("static_value", [](PyIntTupleType &self) -> int32_t {
      return mlirFlyIntTupleTypeGetStaticValue(self);
    });
  }
};

// ---------------------------------------------------------------------------
// LayoutType
// ---------------------------------------------------------------------------
struct PyLayoutType : PyConcreteType<PyLayoutType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFlyLayoutType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = mlirFlyLayoutTypeGetTypeID;
  static constexpr const char *pyClassName = "LayoutType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](nb::handle shape, nb::handle stride, DefaultingPyMlirContext context) {
          ::mlir::MLIRContext *ctx = unwrap(context.get()->get());
          auto shapeAttr = toIntTupleAttr(shape, ctx);
          auto strideAttr = toIntTupleAttr(stride, ctx);
          auto layoutAttr = ::mlir::fly::LayoutAttr::get(ctx, shapeAttr, strideAttr);
          return PyLayoutType(context->getRef(), wrap(::mlir::fly::LayoutType::get(layoutAttr)));
        },
        "shape"_a, "stride"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a LayoutType with shape and stride");

    c.def_prop_ro("shape",
                  [](PyLayoutType &self) -> MlirType { return mlirFlyLayoutTypeGetShape(self); });
    c.def_prop_ro("stride",
                  [](PyLayoutType &self) -> MlirType { return mlirFlyLayoutTypeGetStride(self); });
    c.def_prop_ro("rank",
                  [](PyLayoutType &self) -> int32_t { return mlirFlyLayoutTypeGetRank(self); });
    c.def_prop_ro("depth",
                  [](PyLayoutType &self) -> int32_t { return mlirFlyLayoutTypeGetDepth(self); });
    c.def_prop_ro("is_leaf",
                  [](PyLayoutType &self) -> bool { return mlirFlyLayoutTypeIsLeaf(self); });
    c.def_prop_ro("is_static",
                  [](PyLayoutType &self) -> bool { return mlirFlyLayoutTypeIsStatic(self); });
    c.def_prop_ro("is_static_shape",
                  [](PyLayoutType &self) -> bool { return mlirFlyLayoutTypeIsStaticShape(self); });
    c.def_prop_ro("is_static_stride",
                  [](PyLayoutType &self) -> bool { return mlirFlyLayoutTypeIsStaticStride(self); });
  }
};

// ---------------------------------------------------------------------------
// SwizzleType
// ---------------------------------------------------------------------------
struct PySwizzleType : PyConcreteType<PySwizzleType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFlySwizzleType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = mlirFlySwizzleTypeGetTypeID;
  static constexpr const char *pyClassName = "SwizzleType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t mask, int32_t base, int32_t shift, DefaultingPyMlirContext context) {
          ::mlir::MLIRContext *ctx = unwrap(context.get()->get());
          auto attr = ::mlir::fly::SwizzleAttr::get(ctx, mask, base, shift);
          return PySwizzleType(context->getRef(), wrap(::mlir::fly::SwizzleType::get(attr)));
        },
        "mask"_a, "base"_a, "shift"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a SwizzleType");

    c.def_prop_ro("mask",
                  [](PySwizzleType &self) -> int32_t { return mlirFlySwizzleTypeGetMask(self); });
    c.def_prop_ro("base",
                  [](PySwizzleType &self) -> int32_t { return mlirFlySwizzleTypeGetBase(self); });
    c.def_prop_ro("shift",
                  [](PySwizzleType &self) -> int32_t { return mlirFlySwizzleTypeGetShift(self); });
  }
};

// ---------------------------------------------------------------------------
// PointerType
// ---------------------------------------------------------------------------
struct PyPointerType : PyConcreteType<PyPointerType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFlyPointerType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = mlirFlyPointerTypeGetTypeID;
  static constexpr const char *pyClassName = "PointerType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &elemTyObj, std::optional<int32_t> addressSpace, std::optional<int32_t> alignment,
           DefaultingPyMlirContext context) {
          ::mlir::MLIRContext *ctx = unwrap(context.get()->get());
          MlirType elemTy = elemTyObj;

          auto addr = ::mlir::fly::AddressSpace::Register;
          if (addressSpace.has_value())
            addr = static_cast<::mlir::fly::AddressSpace>(addressSpace.value());

          int32_t alignSize = alignment.value_or(1);
          assert(alignSize > 0 && "alignment must be positive");

          return PyPointerType(context->getRef(),
                               wrap(::mlir::fly::PointerType::get(
                                   unwrap(elemTy), ::mlir::fly::AddressSpaceAttr::get(ctx, addr),
                                   ::mlir::fly::AlignAttr::get(ctx, alignSize))));
        },
        "elem_ty"_a, "address_space"_a = nb::none(), "alignment"_a = nb::none(), nb::kw_only(),
        "context"_a = nb::none(), "Create a PointerType with element type and address space");

    c.def_prop_ro("element_type", [](PyPointerType &self) -> MlirType {
      return mlirFlyPointerTypeGetElementType(self);
    });
    c.def_prop_ro("address_space", [](PyPointerType &self) -> int32_t {
      return mlirFlyPointerTypeGetAddressSpace(self);
    });
    c.def_prop_ro("alignment", [](PyPointerType &self) -> int32_t {
      return mlirFlyPointerTypeGetAlignment(self);
    });
    c.def_prop_ro("swizzle", [](PyPointerType &self) -> MlirType {
      return mlirFlyPointerTypeGetSwizzle(self);
    });
  }
};

// ---------------------------------------------------------------------------
// MemRefType
// ---------------------------------------------------------------------------
struct PyMemRefType : PyConcreteType<PyMemRefType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFlyMemRefType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = mlirFlyMemRefTypeGetTypeID;
  static constexpr const char *pyClassName = "MemRefType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &elemTyObj, PyType &layoutObj, std::optional<int32_t> addressSpace,
           std::optional<int32_t> alignment, DefaultingPyMlirContext context) {
          ::mlir::MLIRContext *ctx = unwrap(context.get()->get());
          MlirType layoutMlirTy = layoutObj;
          auto layoutType = ::mlir::dyn_cast<::mlir::fly::LayoutType>(unwrap(layoutMlirTy));
          if (!layoutType)
            throw std::invalid_argument("layout must be a LayoutType");

          auto addr = ::mlir::fly::AddressSpace::Register;
          if (addressSpace.has_value())
            addr = static_cast<::mlir::fly::AddressSpace>(addressSpace.value());

          int32_t alignSize = alignment.value_or(1);
          assert(alignSize > 0 && "alignment must be positive");

          MlirType elemTy = elemTyObj;
          return PyMemRefType(
              context->getRef(),
              wrap(::mlir::fly::MemRefType::get(
                  unwrap(elemTy), ::mlir::fly::AddressSpaceAttr::get(ctx, addr),
                  layoutType.getAttr(), ::mlir::fly::AlignAttr::get(ctx, alignSize))));
        },
        "elem_ty"_a, "layout"_a, "address_space"_a = 0, "alignment"_a = nb::none(), nb::kw_only(),
        "context"_a = nb::none(),
        "Create a MemRefType with element type, layout, address space and "
        "alignment");

    c.def_prop_ro("element_type", [](PyMemRefType &self) -> MlirType {
      return mlirFlyMemRefTypeGetElementType(self);
    });
    c.def_prop_ro("layout",
                  [](PyMemRefType &self) -> MlirType { return mlirFlyMemRefTypeGetLayout(self); });
    c.def_prop_ro("address_space", [](PyMemRefType &self) -> int32_t {
      return mlirFlyMemRefTypeGetAddressSpace(self);
    });
    c.def_prop_ro("alignment", [](PyMemRefType &self) -> int32_t {
      return mlirFlyMemRefTypeGetAlignment(self);
    });
    c.def_prop_ro("swizzle",
                  [](PyMemRefType &self) -> MlirType { return mlirFlyMemRefTypeGetSwizzle(self); });
  }
};

// ---------------------------------------------------------------------------
// CopyOpUniversalCopyType
// ---------------------------------------------------------------------------
struct PyCopyOpUniversalCopyType : PyConcreteType<PyCopyOpUniversalCopyType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFlyCopyOpUniversalCopyType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFlyCopyOpUniversalCopyTypeGetTypeID;
  static constexpr const char *pyClassName = "CopyOpUniversalCopyType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t bitSize, DefaultingPyMlirContext context) {
          ::mlir::MLIRContext *ctx = unwrap(context.get()->get());
          return PyCopyOpUniversalCopyType(
              context->getRef(), wrap(::mlir::fly::CopyOpUniversalCopyType::get(ctx, bitSize)));
        },
        "bitSize"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyOpUniversalCopyType with bit size");

    c.def_prop_ro("bit_size", [](PyCopyOpUniversalCopyType &self) -> int32_t {
      return mlirFlyCopyOpUniversalCopyTypeGetBitSize(self);
    });
  }
};

// ---------------------------------------------------------------------------
// CopyAtomType
// ---------------------------------------------------------------------------
struct PyCopyAtomType : PyConcreteType<PyCopyAtomType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFlyCopyAtomType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = mlirFlyCopyAtomTypeGetTypeID;
  static constexpr const char *pyClassName = "CopyAtomType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro("copy_op",
                  [](PyCopyAtomType &self) -> MlirType { return mlirFlyCopyAtomTypeGetCopyOp(self); });
    c.def_prop_ro("val_bits",
                  [](PyCopyAtomType &self) -> int32_t { return mlirFlyCopyAtomTypeGetValBits(self); });
    c.def_prop_ro("thr_layout",
                  [](PyCopyAtomType &self) -> MlirType { return mlirFlyCopyAtomTypeGetThrLayout(self); });
    c.def_prop_ro("tv_layout_src",
                  [](PyCopyAtomType &self) -> MlirType { return mlirFlyCopyAtomTypeGetThrValLayoutSrc(self); });
    c.def_prop_ro("tv_layout_dst",
                  [](PyCopyAtomType &self) -> MlirType { return mlirFlyCopyAtomTypeGetThrValLayoutDst(self); });
    c.def_prop_ro("tv_layout_ref",
                  [](PyCopyAtomType &self) -> MlirType { return mlirFlyCopyAtomTypeGetThrValLayoutRef(self); });
  }
};

// ---------------------------------------------------------------------------
// MmaAtomUniversalFMAType
// ---------------------------------------------------------------------------
struct PyMmaAtomUniversalFMAType : PyConcreteType<PyMmaAtomUniversalFMAType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFlyMmaAtomUniversalFMAType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = mlirFlyMmaAtomUniversalFMATypeGetTypeID;
  static constexpr const char *pyClassName = "MmaAtomUniversalFMAType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &elemTyObj, DefaultingPyMlirContext context) {
          MlirType elemTy = elemTyObj;
          return PyMmaAtomUniversalFMAType(
              context->getRef(), wrap(::mlir::fly::MmaAtomUniversalFMAType::get(unwrap(elemTy))));
        },
        "elem_ty"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a MmaAtomUniversalFMAType with element type");

    c.def_prop_ro("elem_ty", [](PyMmaAtomUniversalFMAType &self) -> MlirType {
      return mlirFlyMmaAtomUniversalFMATypeGetElemTy(self);
    });

    c.def_prop_ro("thr_layout", [](PyMmaAtomUniversalFMAType &self) -> MlirType {
      auto ty =
          ::mlir::cast<::mlir::fly::MmaAtomTypeInterface>(unwrap(static_cast<MlirType>(self)));
      auto attr = ::mlir::cast<::mlir::fly::LayoutAttr>(ty.getThrLayout());
      return wrap(::mlir::fly::LayoutType::get(attr));
    });
    c.def_prop_ro("shape_mnk", [](PyMmaAtomUniversalFMAType &self) -> MlirType {
      auto ty =
          ::mlir::cast<::mlir::fly::MmaAtomTypeInterface>(unwrap(static_cast<MlirType>(self)));
      auto attr = ::mlir::cast<::mlir::fly::IntTupleAttr>(ty.getShapeMNK());
      return wrap(::mlir::fly::IntTupleType::get(attr));
    });
    c.def_prop_ro("tv_layout_a", [](PyMmaAtomUniversalFMAType &self) -> MlirType {
      auto ty =
          ::mlir::cast<::mlir::fly::MmaAtomTypeInterface>(unwrap(static_cast<MlirType>(self)));
      auto attr = ::mlir::cast<::mlir::fly::LayoutAttr>(ty.getThrValLayoutA());
      return wrap(::mlir::fly::LayoutType::get(attr));
    });
    c.def_prop_ro("tv_layout_b", [](PyMmaAtomUniversalFMAType &self) -> MlirType {
      auto ty =
          ::mlir::cast<::mlir::fly::MmaAtomTypeInterface>(unwrap(static_cast<MlirType>(self)));
      auto attr = ::mlir::cast<::mlir::fly::LayoutAttr>(ty.getThrValLayoutB());
      return wrap(::mlir::fly::LayoutType::get(attr));
    });
    c.def_prop_ro("tv_layout_c", [](PyMmaAtomUniversalFMAType &self) -> MlirType {
      auto ty =
          ::mlir::cast<::mlir::fly::MmaAtomTypeInterface>(unwrap(static_cast<MlirType>(self)));
      auto attr = ::mlir::cast<::mlir::fly::LayoutAttr>(ty.getThrValLayoutC());
      return wrap(::mlir::fly::LayoutType::get(attr));
    });
  }
};

struct PyTiledCopyType : PyConcreteType<PyTiledCopyType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFlyTiledCopyType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = mlirFlyTiledCopyTypeGetTypeID;
  static constexpr const char *pyClassName = "TiledCopyType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro("copy_atom", [](PyTiledCopyType &self) -> MlirType {
      return mlirFlyTiledCopyTypeGetCopyAtom(self);
    });
    c.def_prop_ro("layout_thr_val", [](PyTiledCopyType &self) -> MlirType {
      return mlirFlyTiledCopyTypeGetLayoutThrVal(self);
    });
    c.def_prop_ro("tile_mn", [](PyTiledCopyType &self) -> MlirType {
      return mlirFlyTiledCopyTypeGetTileMN(self);
    });
    c.def_prop_ro("tiled_tv_layout_src", [](PyTiledCopyType &self) -> MlirType {
      return mlirFlyTiledCopyTypeGetTiledTVLayoutSrc(self);
    });
    c.def_prop_ro("tiled_tv_layout_dst", [](PyTiledCopyType &self) -> MlirType {
      return mlirFlyTiledCopyTypeGetTiledTVLayoutDst(self);
    });
  }
};

struct PyTiledMmaType : PyConcreteType<PyTiledMmaType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFlyTiledMmaType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = mlirFlyTiledMmaTypeGetTypeID;
  static constexpr const char *pyClassName = "TiledMmaType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro("mma_atom", [](PyTiledMmaType &self) -> MlirType {
      return mlirFlyTiledMmaTypeGetMmaAtom(self);
    });
    c.def_prop_ro("atom_layout", [](PyTiledMmaType &self) -> MlirType {
      return mlirFlyTiledMmaTypeGetAtomLayout(self);
    });
    c.def_prop_ro("permutation", [](PyTiledMmaType &self) -> MlirType {
      return mlirFlyTiledMmaTypeGetPermutation(self);
    });
    c.def_prop_ro("tile_size_mnk", [](PyTiledMmaType &self) -> MlirType {
      return mlirFlyTiledMmaTypeGetTileSizeMNK(self);
    });
    c.def_prop_ro("thr_layout_vmnk", [](PyTiledMmaType &self) -> MlirType {
      return mlirFlyTiledMmaTypeGetThrLayoutVMNK(self);
    });
    c.def_prop_ro("tiled_tv_layout_a", [](PyTiledMmaType &self) -> MlirType {
      return mlirFlyTiledMmaTypeGetTiledTVLayoutA(self);
    });
    c.def_prop_ro("tiled_tv_layout_b", [](PyTiledMmaType &self) -> MlirType {
      return mlirFlyTiledMmaTypeGetTiledTVLayoutB(self);
    });
    c.def_prop_ro("tiled_tv_layout_c", [](PyTiledMmaType &self) -> MlirType {
      return mlirFlyTiledMmaTypeGetTiledTVLayoutC(self);
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

NB_MODULE(_fly, m) {
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
        ::mlir::MLIRContext *ctx = unwrap(context);
        IntTupleAttrBuilder builder{ctx};
        auto attr = builder(int_or_tuple);
        return std::make_pair(wrap(::mlir::fly::IntTupleType::get(attr)), builder.dyncElems);
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

  // -------------------------------------------------------------------------
  // Bind Fly dialect types (PyConcreteType pattern)
  // -------------------------------------------------------------------------
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyIntTupleType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyLayoutType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PySwizzleType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyPointerType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyMemRefType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyCopyOpUniversalCopyType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyCopyAtomType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyMmaAtomUniversalFMAType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyTiledCopyType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyTiledMmaType::bind(m);
}
