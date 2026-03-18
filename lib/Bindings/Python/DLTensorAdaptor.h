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
#include <cstring>
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

class DLTensorAdaptor {
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
      } else {
        return IntTupleAttr::getLeafStatic(ctx_, dimSize);
      }
    }
  };

  struct MemRefDescriptor {
    Type memrefType = nullptr;
    void *dataPtr = nullptr;
    std::vector<char> layoutBuffer;
  };

public:
  DLTensorAdaptor(nb::object dlpackCapsule, std::optional<int32_t> alignment, bool use32BitStride)
      : dlpackCapsule_(dlpackCapsule), use32BitStride_(use32BitStride) {
    DLManagedTensor *managed =
        static_cast<DLManagedTensor *>(PyCapsule_GetPointer(dlpackCapsule.ptr(), "dltensor"));
    if (!managed) {
      throw std::runtime_error("Invalid DLPack capsule: expected 'dltensor'");
    }
    tensor_ = &managed->dl_tensor;

    // Calculate element size in bytes (minimum 1 byte)
    int32_t bitsPerElem = tensor_->dtype.bits * tensor_->dtype.lanes;
    int32_t bytesPerElem = (bitsPerElem + 7) / 8;

    // Set alignment: use provided value or default to element size
    alignment_ = alignment.value_or(bytesPerElem);
    if (alignment_ < 1) {
      throw std::runtime_error("Alignment must be at least 1");
    }

    ndim_ = tensor_->ndim;
    if (ndim_ == 0) {
      throw std::runtime_error("DLTensor must have at least one dimension");
    }

    shape_.resize(ndim_);
    stride_.resize(ndim_);
    for (int i = 0; i < ndim_; ++i) {
      shape_[i] = DimInfo(tensor_->shape[i]);
    }
    for (int i = 0; i < ndim_; ++i) {
      stride_[i] = DimInfo(tensor_->strides[i]);
    }
  }

  nb::tuple getShape() const {
    nb::list result;
    for (const auto &s : shape_) {
      result.append(nb::int_(s.dimSize));
    }
    return nb::tuple(result);
  }

  nb::tuple getStride() const {
    nb::list result;
    for (const auto &s : stride_) {
      result.append(nb::int_(s.dimSize));
    }
    return nb::tuple(result);
  }

  int64_t getDataPtr() const {
    return reinterpret_cast<int64_t>(static_cast<char *>(tensor_->data) + tensor_->byte_offset);
  }

  int64_t getSizeInBytes() const {
    int64_t numElements = 1;
    for (const auto &s : shape_) {
      numElements *= s.dimSize;
    }
    int64_t bitsPerElem = tensor_->dtype.bits * tensor_->dtype.lanes;
    return (numElements * bitsPerElem + 7) / 8;
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

  Type getElementType() {
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
      return IntegerType::get(ctx, dtype.bits, IntegerType::Signed);
    case kDLUInt:
      return IntegerType::get(ctx, dtype.bits, IntegerType::Unsigned);
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

  void buildMemRefDesc() {
    if (!isMemrefStale_) {
      return;
    }

    MLIRContext *ctx = getCurrentContext();
    SmallVector<Attribute> shapeLeaves, strideLeaves;
    shapeLeaves.resize(ndim_);
    strideLeaves.resize(ndim_);

    size_t shapeDyncCount = 0;
    size_t strideDyncCount = 0;
    for (int i = 0; i < ndim_; ++i) {
      shapeLeaves[i] = shape_[i].getIntAttr(ctx, true);
      strideLeaves[i] = stride_[i].getIntAttr(ctx, use32BitStride_);

      if (shape_[i].isDynamic)
        shapeDyncCount++;
      if (stride_[i].isDynamic)
        strideDyncCount++;
    }

    IntTupleAttr shapeAttr, strideAttr;
    if (shapeLeaves.size() == 1) {
      shapeAttr = cast<IntTupleAttr>(shapeLeaves[0]);
    } else {
      shapeAttr = IntTupleAttr::get(ArrayAttr::get(ctx, shapeLeaves));
    }
    if (strideLeaves.size() == 1) {
      strideAttr = cast<IntTupleAttr>(strideLeaves[0]);
    } else {
      strideAttr = IntTupleAttr::get(ArrayAttr::get(ctx, strideLeaves));
    }

    LayoutAttr layoutAttr = LayoutAttr::get(ctx, shapeAttr, strideAttr);

    if (getAddressSpace() != 1) {
      throw std::runtime_error("Only device address space is supported");
    }
    AddressSpaceAttr addrSpaceAttr = AddressSpaceAttr::get(ctx, AddressSpace::Global);

    assert(alignment_ > 0 && "alignment must be positive");
    AlignAttr alignAttr = AlignAttr::get(ctx, alignment_);

    memrefDesc_.memrefType =
        fly::MemRefType::get(getElementType(), addrSpaceAttr, layoutAttr, alignAttr);

    // Get data pointer (with byte offset applied)
    memrefDesc_.dataPtr =
        static_cast<void *>(static_cast<char *>(tensor_->data) + tensor_->byte_offset);

    // Build packed layout struct for dynamic elements
    // Layout: [shape_dync_elems (i32)...][stride_dync_elems (i32 or i64)...]
    size_t strideElemSize = use32BitStride_ ? sizeof(int32_t) : sizeof(int64_t);
    size_t layoutSize = shapeDyncCount * sizeof(int32_t) + strideDyncCount * strideElemSize;

    if (layoutSize > 0) {
      memrefDesc_.layoutBuffer.resize(layoutSize);
      char *ptr = memrefDesc_.layoutBuffer.data();

      // Pack dynamic shape elements (i32)
      for (int i = 0; i < ndim_; ++i) {
        if (shape_[i].isDynamic) {
          int32_t val = static_cast<int32_t>(shape_[i].dimSize);
          std::memcpy(ptr, &val, sizeof(int32_t));
          ptr += sizeof(int32_t);
        }
      }
      // Pack dynamic stride elements (i32 or i64)
      for (int i = 0; i < ndim_; ++i) {
        if (stride_[i].isDynamic) {
          if (use32BitStride_) {
            int32_t val = static_cast<int32_t>(stride_[i].dimSize);
            std::memcpy(ptr, &val, sizeof(int32_t));
            ptr += sizeof(int32_t);
          } else {
            int64_t val = stride_[i].dimSize;
            std::memcpy(ptr, &val, sizeof(int64_t));
            ptr += sizeof(int64_t);
          }
        }
      }
    }

    isMemrefStale_ = false;
  }

  MlirType getMemRefType() {
    if (isMemrefStale_) {
      throw std::runtime_error("Memref descriptor is stale");
    }
    return wrap(memrefDesc_.memrefType);
  }

  nb::list getCPointers() const {
    if (isMemrefStale_) {
      throw std::runtime_error("Memref descriptor is stale");
    }
    nb::list result;
    // Add data pointer as integer
    result.append(nb::int_(reinterpret_cast<intptr_t>(&memrefDesc_.dataPtr)));
    // If layout has dynamic elements, add layout struct pointer
    if (!memrefDesc_.layoutBuffer.empty()) {
      result.append(nb::int_(reinterpret_cast<intptr_t>(memrefDesc_.layoutBuffer.data())));
    }
    return result;
  }

  DLTensorAdaptor &markLayoutDynamic(int leadingDim = -1, int divisibility = 1) {
    int ndim_ = static_cast<int>(shape_.size());
    if (leadingDim == -1) {
      for (int i = 0; i < ndim_; ++i) {
        if (stride_[i].dimSize == 1) {
          if (leadingDim != -1) {
            throw std::runtime_error("Multiple dimensions have stride 1");
          }
          leadingDim = i;
        }
      }
    }
    if (leadingDim < 0 || leadingDim >= ndim_) {
      throw std::runtime_error("Cannot determine leading dimension");
    }
    if (stride_[leadingDim].dimSize != 1) {
      throw std::runtime_error("Leading dimension must have stride 1");
    }

    isMemrefStale_ = true;
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

  DLTensorAdaptor &use32BitStride(bool use32BitStride) {
    if (use32BitStride_ == use32BitStride) {
      return *this;
    }
    isMemrefStale_ = true;
    use32BitStride_ = use32BitStride;
    return *this;
  }

private:
  nb::object dlpackCapsule_;
  int32_t alignment_;
  bool use32BitStride_;

  DLTensor *tensor_;
  int32_t ndim_;
  std::vector<DimInfo> shape_;
  std::vector<DimInfo> stride_;
  MemRefDescriptor memrefDesc_;
  bool isMemrefStale_{true};
};

} // namespace mlir::fly::utils
