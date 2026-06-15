// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#pragma once

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntUtils.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

using namespace mlir;
using namespace mlir::fly;
using namespace mlir::python::nanobind_adaptors;

namespace mlir::fly::utils {

inline MLIRContext *getCurrentContext() {
  nb::object currentCtx = mlir::python::irModule().attr("Context").attr("current");
  if (currentCtx.is_none()) {
    throw std::runtime_error("No MLIR context available. Either pass a context explicitly or "
                             "call within an active ir.Context (using 'with context:')");
  }
  auto capsule = mlirApiObjectToCapsule(currentCtx);
  if (!capsule) {
    throw std::runtime_error("Invalid MLIR context capsule");
  }
  return unwrap(mlirPythonCapsuleToContext(capsule->ptr()));
}

class MemRefSpec {
private:
  struct DimInfo {
    int64_t dimSize = 0;
    int32_t divisibility = 1;
    bool isDynamic = false;

    DimInfo() = default;
    DimInfo(int64_t dimSize) : dimSize(dimSize), divisibility(dimSize) {}

    DimInfo &setDynamic(int32_t divisibility = 1) {
      isDynamic = true;
      this->divisibility = divisibility;
      return *this;
    }

    IntTupleAttr getIntAttr(MLIRContext *ctx_, bool use32bitDynamic = false) const {
      if (isDynamic) {
        return IntTupleAttr::getLeafDynamic(ctx_, use32bitDynamic ? 32 : 64, divisibility);
      }
      return IntTupleAttr::getLeafStatic(ctx_, dimSize);
    }
  };

public:
  MemRefSpec(int32_t elementBits, const std::vector<int64_t> &shape,
             const std::vector<int64_t> &strides, std::optional<int32_t> alignment,
             bool use32BitStride)
      : use32BitStride_(use32BitStride) {
    if (shape.size() != strides.size()) {
      throw std::runtime_error("MemRefSpec: shape and strides must have equal rank");
    }
    ndim_ = static_cast<int32_t>(shape.size());
    if (ndim_ == 0) {
      throw std::runtime_error("MemRefSpec: must have at least one dimension");
    }

    // Default byte alignment is the element width rounded up to whole bytes
    // (sub-byte types fall back to 1 byte).
    alignment_ = alignment.value_or((elementBits + 7) / 8);
    if (alignment_ < 1) {
      throw std::runtime_error("Alignment must be at least 1");
    }

    shape_.resize(ndim_);
    stride_.resize(ndim_);
    for (int i = 0; i < ndim_; ++i) {
      shape_[i] = DimInfo(shape[i]);
    }
    for (int i = 0; i < ndim_; ++i) {
      stride_[i] = DimInfo(strides[i]);
    }
  }

  MemRefSpec &markLayoutDynamic(int leadingDim = -1, int divisibility = 1) {
    if (leadingDim == -1) {
      // First (lowest-index) unit-stride axis.  Degenerate tensors (size-1 /
      // size-0 axes whose stride is coerced to 1) can have several; the earliest
      // is chosen rather than rejecting.
      for (int i = 0; i < ndim_; ++i) {
        if (stride_[i].dimSize == 1) {
          leadingDim = i;
          break;
        }
      }
    }
    if (leadingDim < 0 || leadingDim >= ndim_) {
      throw std::runtime_error(
          "tensor has no axis with stride == 1; layout-dynamic memref requires one");
    }
    if (stride_[leadingDim].dimSize != 1) {
      throw std::runtime_error("Leading dimension must have stride 1");
    }
    for (int i = 0; i < ndim_; ++i) {
      shape_[i].setDynamic();
    }
    for (int i = 0; i < ndim_; ++i) {
      if (i != leadingDim) {
        stride_[i].setDynamic(divisibility);
      }
    }
    return *this;
  }

  MemRefSpec &markShapeDynamic(nb::list dims, nb::list divisibilities) {
    markDynamic(shape_, dims, divisibilities);
    return *this;
  }

  MemRefSpec &markStrideDynamic(nb::list dims, nb::list divisibilities) {
    markDynamic(stride_, dims, divisibilities);
    return *this;
  }

  MemRefSpec &use32BitStride(bool use32BitStride) {
    use32BitStride_ = use32BitStride;
    return *this;
  }

  // ``elementType`` is built by the caller in the active (compile) context.
  MlirType getMemRefType(MlirType elementType) const {
    MLIRContext *ctx = getCurrentContext();
    SmallVector<Attribute> shapeLeaves(ndim_), strideLeaves(ndim_);
    for (int i = 0; i < ndim_; ++i) {
      shapeLeaves[i] = shape_[i].getIntAttr(ctx, true);
      strideLeaves[i] = stride_[i].getIntAttr(ctx, use32BitStride_);
    }

    IntTupleAttr shapeAttr = shapeLeaves.size() == 1
                                 ? cast<IntTupleAttr>(shapeLeaves[0])
                                 : IntTupleAttr::get(ArrayAttr::get(ctx, shapeLeaves));
    IntTupleAttr strideAttr = strideLeaves.size() == 1
                                  ? cast<IntTupleAttr>(strideLeaves[0])
                                  : IntTupleAttr::get(ArrayAttr::get(ctx, strideLeaves));

    LayoutAttr layoutAttr = LayoutAttr::get(ctx, shapeAttr, strideAttr);
    AddressSpaceAttr addrSpaceAttr = AddressSpaceAttr::get(ctx, AddressSpace::Global);
    AlignAttr alignAttr = AlignAttr::get(ctx, alignment_);
    return wrap(fly::MemRefType::get(unwrap(elementType), addrSpaceAttr, layoutAttr, alignAttr));
  }

  // Encoding: static→dimSize, dynamic→-divisibility, so the cache key reflects
  // the resolved layout state.
  nb::tuple getCacheSignature() const {
    auto encode = [](const DimInfo &dim) {
      return dim.isDynamic ? -dim.divisibility : dim.dimSize;
    };
    nb::object shapeTuple = nb::steal(PyTuple_New(static_cast<Py_ssize_t>(shape_.size())));
    for (size_t i = 0; i < shape_.size(); ++i) {
      PyTuple_SET_ITEM(shapeTuple.ptr(), static_cast<Py_ssize_t>(i),
                       PyLong_FromLongLong(encode(shape_[i])));
    }
    nb::object strideTuple = nb::steal(PyTuple_New(static_cast<Py_ssize_t>(stride_.size())));
    for (size_t i = 0; i < stride_.size(); ++i) {
      PyTuple_SET_ITEM(strideTuple.ptr(), static_cast<Py_ssize_t>(i),
                       PyLong_FromLongLong(encode(stride_[i])));
    }
    return nb::make_tuple(alignment_, use32BitStride_, shapeTuple, strideTuple);
  }

  // Dynamic-dim index tuples, so the Python side reads the masks from this single
  // source instead of duplicating them.
  nb::tuple getShapeDynIndices() const { return dynIndices(shape_); }
  nb::tuple getStrideDynIndices() const { return dynIndices(stride_); }

private:
  static nb::tuple dynIndices(const std::vector<DimInfo> &dims) {
    nb::list result;
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i].isDynamic) {
        result.append(nb::int_(static_cast<int64_t>(i)));
      }
    }
    return nb::tuple(result);
  }

  void markDynamic(std::vector<DimInfo> &dims_, nb::list dims, nb::list divisibilities) {
    int ndim = static_cast<int>(dims_.size());
    size_t count = nb::len(dims);
    if (nb::len(divisibilities) != count) {
      throw std::runtime_error("markDynamic: dims and divisibilities must have equal length");
    }
    for (size_t k = 0; k < count; ++k) {
      int idx = nb::cast<int>(dims[k]);
      if (idx < 0 || idx >= ndim) {
        throw std::runtime_error("markDynamic: dimension index out of range");
      }
      dims_[idx].setDynamic(nb::cast<int>(divisibilities[k]));
    }
  }

  int32_t alignment_;
  bool use32BitStride_;
  int32_t ndim_;
  std::vector<DimInfo> shape_;
  std::vector<DimInfo> stride_;
};

} // namespace mlir::fly::utils
