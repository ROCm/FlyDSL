// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

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

#include "dlpack/dlpack.h"

#include <cstdint>
#include <string>
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

class DLTensorAdaptor {
public:
  DLTensorAdaptor(nb::object dlpackCapsule) : dlpackCapsule_(dlpackCapsule) {
    DLManagedTensor *managed =
        static_cast<DLManagedTensor *>(PyCapsule_GetPointer(dlpackCapsule.ptr(), "dltensor"));
    if (!managed) {
      throw std::runtime_error("Invalid DLPack capsule: expected 'dltensor'");
    }
    tensor_ = &managed->dl_tensor;

    ndim_ = tensor_->ndim;
    if (ndim_ == 0) {
      throw std::runtime_error("DLTensor must have at least one dimension");
    }
    shape_.assign(tensor_->shape, tensor_->shape + ndim_);
    if (tensor_->strides) {
      stride_.assign(tensor_->strides, tensor_->strides + ndim_);
    } else {
      // DLPack: NULL strides denotes a row-major compact tensor. Strides are in
      // *elements* (not bytes): last dim is 1, each earlier dim the product of
      // the trailing shapes.
      stride_.resize(ndim_);
      int64_t s = 1;
      for (int i = ndim_ - 1; i >= 0; --i) {
        stride_[i] = s;
        s *= shape_[i];
      }
    }
  }

  nb::tuple getShape() const {
    nb::list result;
    for (int64_t s : shape_) {
      result.append(nb::int_(s));
    }
    return nb::tuple(result);
  }

  nb::tuple getStride() const {
    nb::list result;
    for (int64_t s : stride_) {
      result.append(nb::int_(s));
    }
    return nb::tuple(result);
  }

  int64_t getDataPtr() const {
    return reinterpret_cast<int64_t>(static_cast<char *>(tensor_->data) + tensor_->byte_offset);
  }

  int64_t getSizeInBytes() const {
    int64_t numElements = 1;
    for (int64_t s : shape_) {
      numElements *= s;
    }
    return (numElements * getElementBits() + 7) / 8;
  }

  // Element width in bits (bits * lanes), kept at bit granularity so sub-byte
  // types (e.g. fp4 / i4) describe their true width when fed to MemRefSpec.
  // Context-free.
  int32_t getElementBits() const { return tensor_->dtype.bits * tensor_->dtype.lanes; }

  // dlpack dtype as (code, bits, lanes): a context-free hashable id a frontend
  // can use as a cache discriminator without ingesting the capsule again.
  nb::tuple getDtypeId() const {
    return nb::make_tuple(static_cast<int>(tensor_->dtype.code),
                          static_cast<int>(tensor_->dtype.bits),
                          static_cast<int>(tensor_->dtype.lanes));
  }

  int getAddressSpace() const {
    switch (tensor_->device.device_type) {
    case kDLCPU:
      return 0; // Host
    case kDLCUDA:
      [[fallthrough]];
    case kDLCUDAHost:
      [[fallthrough]];
    case kDLCUDAManaged:
      [[fallthrough]];
    case kDLROCM:
      [[fallthrough]];
    case kDLROCMHost:
      return 1; // Global (device memory)
    default:
      return 0;
    }
  }

  Type getDtype() {
    DLDataType dtype = tensor_->dtype;
    MLIRContext *ctx = getCurrentContext();

    switch (dtype.code) {
    case kDLFloat:
      switch (dtype.bits) {
      case 16:
        return Float16Type::get(ctx);
      case 32:
        return Float32Type::get(ctx);
      case 64:
        return Float64Type::get(ctx);
      default:
        throw std::runtime_error("Unsupported float bit width: " + std::to_string(dtype.bits));
      }
    case kDLInt:
      return IntegerType::get(ctx, dtype.bits);
    case kDLUInt:
      return IntegerType::get(ctx, dtype.bits);
    case kDLBfloat:
      return BFloat16Type::get(ctx);
    case kDLBool:
      return IntegerType::get(ctx, 1);
    case kDLFloat8_e5m2:
      return Float8E5M2Type::get(ctx);
    case kDLFloat8_e4m3fn:
      return Float8E4M3FNType::get(ctx);
    case kDLFloat8_e5m2fnuz:
      return Float8E5M2FNUZType::get(ctx);
    case kDLFloat8_e4m3fnuz:
      return Float8E4M3FNUZType::get(ctx);
    case kDLFloat8_e4m3b11fnuz:
      return Float8E4M3B11FNUZType::get(ctx);
    case kDLComplex:
      switch (dtype.bits) {
      case 64:
        return ComplexType::get(Float32Type::get(ctx));
      case 128:
        return ComplexType::get(Float64Type::get(ctx));
      default:
        throw std::runtime_error("Unsupported complex bit width: " + std::to_string(dtype.bits));
      }
    default:
      throw std::runtime_error("Unsupported DLPack dtype code: " + std::to_string(dtype.code));
    }
  }

private:
  nb::object dlpackCapsule_;
  DLTensor *tensor_ = nullptr;
  int32_t ndim_ = 0;
  std::vector<int64_t> shape_;
  std::vector<int64_t> stride_;
};

} // namespace mlir::fly::utils
