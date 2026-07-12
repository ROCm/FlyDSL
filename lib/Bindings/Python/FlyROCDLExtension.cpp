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

struct PyMmaOpCDNA3_MFMAType : PyConcreteType<PyMmaOpCDNA3_MFMAType> {
  FLYDSL_REGISTER_TYPE_BINDING(MmaOpCDNA3_MFMAType, "MmaOpCDNA3_MFMAType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t m, int32_t n, int32_t k, PyType &elemTyA, PyType &elemTyB, PyType &elemTyAcc,
           DefaultingPyMlirContext context) {
          return PyMmaOpCDNA3_MFMAType(context->getRef(), wrap(MmaOpCDNA3_MFMAType::get(
                                                              m, n, k, unwrap(elemTyA),
                                                              unwrap(elemTyB), unwrap(elemTyAcc))));
        },
        "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a, "elem_ty_acc"_a, nb::kw_only(),
        "context"_a = nb::none(),
        "Create a MmaOpCDNA3_MFMAType with m, n, k dimensions and element types");
  }
};

struct PyMmaOpCDNA4_MFMAScaleType : PyConcreteType<PyMmaOpCDNA4_MFMAScaleType> {
  FLYDSL_REGISTER_TYPE_BINDING(MmaOpCDNA4_MFMAScaleType, "MmaOpCDNA4_MFMAScaleType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t m, int32_t n, int32_t k, PyType &elemTyA, PyType &elemTyB, PyType &elemTyAcc,
           int32_t opselA, int32_t opselB, DefaultingPyMlirContext context) {
          return PyMmaOpCDNA4_MFMAScaleType(
              context->getRef(),
              wrap(MmaOpCDNA4_MFMAScaleType::get(m, n, k, unwrap(elemTyA), unwrap(elemTyB),
                                                 unwrap(elemTyAcc), opselA, opselB)));
        },
        "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a, "elem_ty_acc"_a, "opsel_a"_a = 0,
        "opsel_b"_a = 0, nb::kw_only(), "context"_a = nb::none(),
        "Create a MmaOpCDNA4_MFMAScaleType with m, n, k dimensions, element types, "
        "and optional opsel_a / opsel_b (compile-time lane index into the scale "
        "vector, default 0)");
  }
};

struct PyMmaOpGFX1250_WMMAType : PyConcreteType<PyMmaOpGFX1250_WMMAType> {
  FLYDSL_REGISTER_TYPE_BINDING(MmaOpGFX1250_WMMAType, "MmaOpGFX1250_WMMAType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t m, int32_t n, int32_t k, PyType &elemTyA, PyType &elemTyB, PyType &elemTyAcc,
           DefaultingPyMlirContext context) {
          return PyMmaOpGFX1250_WMMAType(
              context->getRef(),
              wrap(MmaOpGFX1250_WMMAType::get(m, n, k, unwrap(elemTyA), unwrap(elemTyB),
                                              unwrap(elemTyAcc))));
        },
        "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a, "elem_ty_acc"_a, nb::kw_only(),
        "context"_a = nb::none(),
        "Create a MmaOpGFX1250_WMMAType with m, n, k dimensions and element types");
  }
};

struct PyMmaOpGFX1250_WMMAScaleType : PyConcreteType<PyMmaOpGFX1250_WMMAScaleType> {
  FLYDSL_REGISTER_TYPE_BINDING(MmaOpGFX1250_WMMAScaleType, "MmaOpGFX1250_WMMAScaleType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t m, int32_t n, int32_t k, PyType &elemTyA, PyType &elemTyB, PyType &elemTyAcc,
           int32_t opselA, int32_t opselB, int32_t modC, bool reuseA, bool reuseB,
           int32_t blockSize, DefaultingPyMlirContext context) {
          return PyMmaOpGFX1250_WMMAScaleType(
              context->getRef(), wrap(MmaOpGFX1250_WMMAScaleType::get(
                                     m, n, k, unwrap(elemTyA), unwrap(elemTyB), unwrap(elemTyAcc),
                                     opselA, opselB, modC, reuseA, reuseB, blockSize)));
        },
        "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a, "elem_ty_acc"_a, "opsel_a"_a = 0,
        "opsel_b"_a = 0, "mod_c"_a = 0, "reuse_a"_a = false, "reuse_b"_a = false,
        "block_size"_a = 32, nb::kw_only(), "context"_a = nb::none(),
        "Create a MmaOpGFX1250_WMMAScaleType (MX-scaled WMMA, E8M0 block scale) with "
        "m, n, k dimensions, element types, optional opsel_a / opsel_b (compile-time "
        "lane index into the scale vector, default 0), the intrinsic mod_c / reuse_a / "
        "reuse_b attributes (default 0 / false), and block_size (16 or 32, default 32) "
        "selecting the V_WMMA_SCALE16 / V_WMMA_SCALE form");
  }
};

struct PyMmaOpGFX11_WMMAType : PyConcreteType<PyMmaOpGFX11_WMMAType> {
  FLYDSL_REGISTER_TYPE_BINDING(MmaOpGFX11_WMMAType, "MmaOpGFX11_WMMAType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t m, int32_t n, int32_t k, PyType &elemTyA, PyType &elemTyB, PyType &elemTyAcc,
           bool signA, bool signB, bool clamp, DefaultingPyMlirContext context) {
          return PyMmaOpGFX11_WMMAType(
              context->getRef(),
              wrap(MmaOpGFX11_WMMAType::get(m, n, k, unwrap(elemTyA), unwrap(elemTyB),
                                            unwrap(elemTyAcc), signA, signB, clamp)));
        },
        "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a, "elem_ty_acc"_a, nb::kw_only(),
        "sign_a"_a = false, "sign_b"_a = false, "clamp"_a = false, "context"_a = nb::none(),
        "Create a MmaOpGFX11_WMMAType with m, n, k dimensions and element types "
        "(RDNA3 / RDNA3.5 wave32 WMMA, v16 operand ABI). "
        "sign_a/sign_b/clamp are forwarded to the iu8/iu4 intrinsic for integer "
        "paths; must be false for fp16/bf16.");
  }
};

struct PyCopyOpCDNA3BufferCopyType : PyConcreteType<PyCopyOpCDNA3BufferCopyType> {
  FLYDSL_REGISTER_TYPE_BINDING(CopyOpCDNA3BufferCopyType, "CopyOpCDNA3BufferCopyType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t bitSize, int32_t cacheModifier, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          return PyCopyOpCDNA3BufferCopyType(
              context->getRef(), wrap(CopyOpCDNA3BufferCopyType::get(ctx, bitSize, cacheModifier)));
        },
        "bit_size"_a, "cache_modifier"_a = 0, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyOpCDNA3BufferCopyType with the given bit size and "
        "cache_modifier (0=cached, 2=non-temporal)");
  }
};

struct PyCopyOpCDNA3BufferCopyLDSType : PyConcreteType<PyCopyOpCDNA3BufferCopyLDSType> {
  FLYDSL_REGISTER_TYPE_BINDING(CopyOpCDNA3BufferCopyLDSType, "CopyOpCDNA3BufferCopyLDSType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t bitSize, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          return PyCopyOpCDNA3BufferCopyLDSType(
              context->getRef(), wrap(CopyOpCDNA3BufferCopyLDSType::get(ctx, bitSize)));
        },
        "bit_size"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyOpCDNA3BufferCopyLDSType with the given bit size");
  }
};

struct PyCopyOpCDNA3BufferAtomicType : PyConcreteType<PyCopyOpCDNA3BufferAtomicType> {
  FLYDSL_REGISTER_TYPE_BINDING(CopyOpCDNA3BufferAtomicType, "CopyOpCDNA3BufferAtomicType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t atomicOp, PyType &valTypeObj, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          auto atomicOpAttr =
              ::mlir::fly::AtomicOpAttr::get(ctx, static_cast<::mlir::fly::AtomicOp>(atomicOp));
          return PyCopyOpCDNA3BufferAtomicType(
              context->getRef(),
              wrap(CopyOpCDNA3BufferAtomicType::get(atomicOpAttr, unwrap(valTypeObj))));
        },
        "atomic_op"_a, "val_type"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyOpCDNA3BufferAtomicType with atomic op and value type");
  }
};

struct PyCopyOpGFX1250TDMType : PyConcreteType<PyCopyOpGFX1250TDMType> {
  FLYDSL_REGISTER_TYPE_BINDING(CopyOpGFX1250TDMType, "CopyOpGFX1250TDMType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t rank, int32_t numWarps, int32_t padInterval, int32_t padAmount,
           int32_t cacheModifier, bool atomicBarrier, bool earlyTimeout,
           DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          return PyCopyOpGFX1250TDMType(
              context->getRef(),
              wrap(CopyOpGFX1250TDMType::get(ctx, rank, numWarps, padInterval, padAmount,
                                             cacheModifier, atomicBarrier, earlyTimeout)));
        },
        "rank"_a, "num_warps"_a, "pad_interval"_a = 0, "pad_amount"_a = 0, "cache_modifier"_a = 0,
        "atomic_barrier"_a = false, "early_timeout"_a = false, nb::kw_only(),
        "context"_a = nb::none(),
        "Create a CopyOpGFX1250TDMType (N-D TDM Global<->LDS copy) with tensor rank (1-5), "
        "warp count, optional LDS padding (interval/amount in elements), cache modifier, and "
        "the descriptor atomic_barrier / early_timeout config bits (default false)");
  }
};

struct PyCopyOpCDNA4LdsReadTransposeType : PyConcreteType<PyCopyOpCDNA4LdsReadTransposeType> {
  FLYDSL_REGISTER_TYPE_BINDING(CopyOpCDNA4LdsReadTransposeType, "CopyOpCDNA4LdsReadTransposeType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t transGranularity, int32_t bitSize, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          return PyCopyOpCDNA4LdsReadTransposeType(
              context->getRef(),
              wrap(CopyOpCDNA4LdsReadTransposeType::get(ctx, transGranularity, bitSize)));
        },
        "trans_granularity"_a, "bit_size"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyOpCDNA4LdsReadTransposeType with transpose granularity and bit size");
  }
};

} // namespace fly_rocdl
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsFlyROCDL, m) {
  m.doc() = "MLIR Python FlyROCDL Extension";

  // clang-format off
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_rocdl::PyMmaOpCDNA3_MFMAType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_rocdl::PyMmaOpCDNA4_MFMAScaleType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_rocdl::PyMmaOpGFX1250_WMMAType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_rocdl::PyMmaOpGFX1250_WMMAScaleType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_rocdl::PyMmaOpGFX11_WMMAType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_rocdl::PyCopyOpCDNA3BufferCopyType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_rocdl::PyCopyOpCDNA3BufferCopyLDSType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_rocdl::PyCopyOpCDNA3BufferAtomicType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_rocdl::PyCopyOpGFX1250TDMType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_rocdl::PyCopyOpCDNA4LdsReadTransposeType::bind(m);
  // clang-format on
}
