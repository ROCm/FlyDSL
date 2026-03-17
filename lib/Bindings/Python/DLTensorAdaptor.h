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
    MLIRContext *bindingCtx = nullptr;
    Type memrefType = nullptr;
    void *dataPtr = nullptr;
    std::vector<char> layoutBuffer;
  };

public:
  /// Constructor from raw tensor attributes.
  /// data_ptr should already include byte_offset (e.g. from torch.Tensor.data_ptr()).
  DLTensorAdaptor(int64_t dataPtr, nb::tuple shape, nb::tuple strides,
                  uint8_t dtypeCode, uint8_t dtypeBits, int deviceType,
                  std::optional<int32_t> alignment, bool use32BitStride) {
    use32BitStride_ = use32BitStride;
    dataPtr_ = reinterpret_cast<void *>(dataPtr);
    dtype_.code = dtypeCode;
    dtype_.bits = dtypeBits;
    dtype_.lanes = 1;
    deviceType_ = static_cast<DLDeviceType>(deviceType);

    int32_t bitsPerElem = dtype_.bits * dtype_.lanes;
    int32_t bytesPerElem = (bitsPerElem + 7) / 8;
    alignment_ = alignment.value_or(bytesPerElem);
    if (alignment_ < 1) {
      throw std::runtime_error("Alignment must be at least 1");
    }

    ndim_ = static_cast<int32_t>(nb::len(shape));
    if (ndim_ == 0) {
      throw std::runtime_error("Tensor must have at least one dimension");
    }

    shape_.resize(ndim_);
    stride_.resize(ndim_);
    for (int i = 0; i < ndim_; ++i) {
      shape_[i] = DimInfo(nb::cast<int64_t>(shape[i]));
    }
    for (int i = 0; i < ndim_; ++i) {
      stride_[i] = DimInfo(nb::cast<int64_t>(strides[i]));
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

  int64_t getDataPtr() const { return reinterpret_cast<int64_t>(dataPtr_); }

  int64_t getSizeInBytes() const {
    int64_t numElements = 1;
    for (const auto &s : shape_) {
      numElements *= s.dimSize;
    }
    int64_t bitsPerElem = dtype_.bits * dtype_.lanes;
    return (numElements * bitsPerElem + 7) / 8;
  }

  int getAddressSpace() const {
    switch (deviceType_) {
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
    MLIRContext *ctx = getCurrentContext();

    switch (dtype_.code) {
    case kDLFloat:
      switch (dtype_.bits) {
      case 16:
        return Float16Type::get(ctx);
      case 32:
        return Float32Type::get(ctx);
      case 64:
        return Float64Type::get(ctx);
      default:
        throw std::runtime_error("Unsupported float bit width: " + std::to_string(dtype_.bits));
      }
    case kDLInt:
      return IntegerType::get(ctx, dtype_.bits, IntegerType::Signed);
    case kDLUInt:
      return IntegerType::get(ctx, dtype_.bits, IntegerType::Unsigned);
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
      switch (dtype_.bits) {
      case 64:
        return ComplexType::get(Float32Type::get(ctx));
      case 128:
        return ComplexType::get(Float64Type::get(ctx));
      default:
        throw std::runtime_error("Unsupported complex bit width: " + std::to_string(dtype_.bits));
      }
    default:
      throw std::runtime_error("Unsupported DLPack dtype code: " + std::to_string(dtype_.code));
    }
  }

  void buildMemRefDesc() {
    MLIRContext *ctx = getCurrentContext();
    if (!isMemrefStale_ && memrefDesc_.bindingCtx == ctx) {
      return;
    }
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

    // Data pointer already stored
    memrefDesc_.dataPtr = dataPtr_;

    // Build packed layout struct for dynamic elements
    // Shape always uses i32; stride uses i32 or i64 based on use32BitStride_
    size_t strideElemSize = use32BitStride_ ? sizeof(int32_t) : sizeof(int64_t);
    size_t layoutSize = shapeDyncCount * sizeof(int32_t) + strideDyncCount * strideElemSize;

    if (layoutSize > 0) {
      memrefDesc_.layoutBuffer.resize(layoutSize);
      char *ptr = memrefDesc_.layoutBuffer.data();

      for (int i = 0; i < ndim_; ++i) {
        if (shape_[i].isDynamic) {
          int32_t val = static_cast<int32_t>(shape_[i].dimSize);
          std::memcpy(ptr, &val, sizeof(int32_t));
          ptr += sizeof(int32_t);
        }
      }
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

    memrefDesc_.bindingCtx = ctx;
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

  /// Update only the data pointer (for hot-path dispatch with same tensor layout).
  void updateDataPtr(int64_t newDataPtr) {
    dataPtr_ = reinterpret_cast<void *>(newDataPtr);
    memrefDesc_.dataPtr = dataPtr_;
  }

  /// Return stable pointers for hot-path dispatch.
  /// Must be called after build_memref_desc() + getCPointers() at least once.
  /// Returns: [ptr_to_dataPtr, ptr_to_layoutBuffer (if any)]
  /// These pointer addresses are stable across calls (no reallocation).
  nb::list getStablePointers() {
    nb::list result;
    result.append(nb::int_(reinterpret_cast<intptr_t>(&memrefDesc_.dataPtr)));
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

  friend class FastDispatcher;

private:
  int32_t alignment_;
  bool use32BitStride_;

  // Tensor data stored directly (no pointer to external DLTensor)
  void *dataPtr_ = nullptr;
  DLDataType dtype_{};
  DLDeviceType deviceType_{};

  int32_t ndim_;
  std::vector<DimInfo> shape_;
  std::vector<DimInfo> stride_;
  MemRefDescriptor memrefDesc_;
  bool isMemrefStale_{true};
};

// ─────────────────────────────────────────────────────────────────────────
// FastDispatcher: stateless C++ hot-path kernel dispatch
//
// No cached DLTensorAdaptor — owns its own storage for data pointers and
// layout buffers.  On each call, reads data_ptr/shape/stride directly from
// Python tensor objects, fills its own buffers, and calls the JIT function.
// ─────────────────────────────────────────────────────────────────────────
class FastDispatcher {
public:
  FastDispatcher(int64_t funcPtr)
      : funcPtr_(reinterpret_cast<void (*)(void *)>(funcPtr)), streamValue_(0),
        nPositional_(-1), hasUserStream_(false) {}

  // ── Builder API ────────────────────────────────────────────────────────

  /// Add a tensor slot.  Reads layout structure from adaptor (which dims
  /// are dynamic, use32BitStride) but does NOT keep a reference to it.
  void addTensorSlot(int argIdx, DLTensorAdaptor &adaptor) {
    TensorSlot slot;
    slot.argIdx = argIdx;
    slot.ndim = adaptor.ndim_;
    slot.use32BitStride = adaptor.use32BitStride_;
    for (int i = 0; i < adaptor.ndim_; ++i) {
      if (adaptor.shape_[i].isDynamic) {
        slot.dynamicShapeDims.push_back(i);
      } else {
        slot.staticGuards.push_back({i, true, adaptor.shape_[i].dimSize});
      }
      if (adaptor.stride_[i].isDynamic) {
        slot.dynamicStrideDims.push_back(i);
      } else {
        slot.staticGuards.push_back({i, false, adaptor.stride_[i].dimSize});
      }
    }
    // Pre-allocate layout buffer (shape=i32, stride=i32 or i64)
    size_t strideElemSz = slot.use32BitStride ? sizeof(int32_t) : sizeof(int64_t);
    size_t layoutSz = slot.dynamicShapeDims.size() * sizeof(int32_t) +
                      slot.dynamicStrideDims.size() * strideElemSz;
    slot.layoutBuffer.resize(layoutSz);
    slotOrder_.push_back({TENSOR, static_cast<int>(tensorSlots_.size())});
    tensorSlots_.push_back(std::move(slot));
  }

  // Scalar type kinds for correct value extraction
  static constexpr int SK_INT = 0;    // nb::cast<int64_t>
  static constexpr int SK_FLOAT = 1;  // nb::cast<double>, memcpy bit pattern
  static constexpr int SK_STREAM = 2; // stream handle extraction

  /// Add a scalar argument slot.
  /// kind: SK_INT (0), SK_FLOAT (1), SK_STREAM (2)
  void addScalarSlot(int argIdx, int byteWidth, int kind) {
    slotOrder_.push_back({SCALAR, static_cast<int>(scalarSlots_.size())});
    scalarSlots_.push_back({argIdx, byteWidth, kind, 0});
  }

  void addConstexprGuard(int argIdx, int64_t expectedValue) {
    constexprGuards_.push_back({argIdx, expectedValue});
  }

  void setKwargsInfo(int nPositional, nb::list names, nb::list defaults) {
    nPositional_ = nPositional;
    for (size_t i = 0; i < nb::len(names); ++i) {
      trailingNames_.push_back(nb::borrow<nb::str>(names[i]));
      trailingDefaults_.push_back(nb::borrow(defaults[i]));
    }
  }

  /// Must be called after all addXxxSlot calls.  Builds the packed pointer
  /// array with stable addresses into this object's own storage.
  void finalize(bool hasUserStream) {
    hasUserStream_ = hasUserStream;
    packedArgs_.clear();
    for (auto &ord : slotOrder_) {
      switch (ord.kind) {
      case TENSOR: {
        auto &s = tensorSlots_[ord.idx];
        packedArgs_.push_back(&s.dataPtrStorage);
        if (!s.layoutBuffer.empty())
          packedArgs_.push_back(s.layoutBuffer.data());
        break;
      }
      case SCALAR:
        packedArgs_.push_back(&scalarSlots_[ord.idx].storage);
        break;
      }
    }
    if (!hasUserStream_)
      packedArgs_.push_back(&streamValue_);
  }

  // ── Hot-path call ──────────────────────────────────────────────────────
  // Returns  0 on success,
  //         -1 on constexpr guard failure,
  //         -2 on shape/stride guard failure.
  int call(nb::args args, nb::kwargs kwargs) {
    // 1. Merge positional + trailing kwargs
    size_t nArgs = args.size();
    nb::object trailingBuf[8];
    int nTrailing = 0;
    if (nPositional_ >= 0 && nArgs == static_cast<size_t>(nPositional_)) {
      for (size_t i = 0; i < trailingNames_.size(); ++i) {
        if (kwargs && kwargs.contains(trailingNames_[i]))
          trailingBuf[nTrailing++] = nb::borrow(kwargs[trailingNames_[i]]);
        else
          trailingBuf[nTrailing++] = trailingDefaults_[i];
      }
    }
    auto getArg = [&](int idx) -> nb::handle {
      return idx < static_cast<int>(nArgs) ? args[idx] : trailingBuf[idx - nArgs];
    };

    // 2. Constexpr guard
    for (auto &g : constexprGuards_)
      if (nb::cast<int64_t>(getArg(g.argIdx)) != g.expectedValue)
        return -1;

    // 3. Tensor slots: static-dim guard + fill data_ptr & layout
    for (auto &slot : tensorSlots_) {
      nb::handle t = getArg(slot.argIdx);
      nb::object pyShape = t.attr("shape");
      if (nb::len(pyShape) != slot.ndim)
        return -2;
      nb::object pyStride = t.attr("stride")();

      // Only guard static dims (dynamic dims are allowed to change)
      for (auto &g : slot.staticGuards) {
        int64_t actual = nb::cast<int64_t>(g.isShape ? pyShape[g.dimIdx] : pyStride[g.dimIdx]);
        if (actual != g.expectedValue)
          return -2;
      }

      // Data pointer
      slot.dataPtrStorage =
          reinterpret_cast<void *>(nb::cast<int64_t>(t.attr("data_ptr")()));

      // Layout buffer: shape always i32, stride i32 or i64 (matches MLIR ABI)
      if (!slot.layoutBuffer.empty()) {
        char *ptr = slot.layoutBuffer.data();
        for (int d : slot.dynamicShapeDims) {
          int32_t v = nb::cast<int32_t>(pyShape[d]);
          std::memcpy(ptr, &v, sizeof(int32_t));
          ptr += sizeof(int32_t);
        }
        for (int d : slot.dynamicStrideDims) {
          if (slot.use32BitStride) {
            int32_t v = nb::cast<int32_t>(pyStride[d]);
            std::memcpy(ptr, &v, sizeof(int32_t));
            ptr += sizeof(int32_t);
          } else {
            int64_t v = nb::cast<int64_t>(pyStride[d]);
            std::memcpy(ptr, &v, sizeof(int64_t));
            ptr += sizeof(int64_t);
          }
        }
      }
    }

    // 4. Scalar slots — type-correct extraction
    for (auto &slot : scalarSlots_) {
      nb::handle arg = getArg(slot.argIdx);
      slot.storage = 0; // zero-fill (important for < 8 byte types)
      switch (slot.kind) {
      case SK_INT:
        slot.storage = nb::cast<int64_t>(arg);
        break;
      case SK_FLOAT:
        if (slot.byteWidth <= 4) {
          float v = nb::cast<float>(arg);
          std::memcpy(&slot.storage, &v, sizeof(float));
        } else {
          double v = nb::cast<double>(arg);
          std::memcpy(&slot.storage, &v, sizeof(double));
        }
        break;
      case SK_STREAM:
        if (nb::hasattr(arg, "cuda_stream")) {
          slot.storage = static_cast<int64_t>(nb::cast<uintptr_t>(arg.attr("cuda_stream")));
        } else if (nb::hasattr(arg, "value")) {
          nb::object v = arg.attr("value");
          if (v.is_none())
            slot.storage = 0;
          else if (nb::isinstance<nb::int_>(v))
            slot.storage = nb::cast<int64_t>(v);
          else
            slot.storage = static_cast<int64_t>(nb::cast<uintptr_t>(v.attr("cuda_stream")));
        }
        break;
      }
    }

    // 6. Call JIT function
    funcPtr_(packedArgs_.data());
    return 0;
  }

private:
  void (*funcPtr_)(void *);

  // Tensor slots — own their storage, no DLTensorAdaptor reference
  struct StaticDimGuard {
    int dimIdx;
    bool isShape;       // true=shape, false=stride
    int64_t expectedValue;
  };
  struct TensorSlot {
    int argIdx;
    int ndim = 0;
    void *dataPtrStorage = nullptr;
    std::vector<char> layoutBuffer;
    std::vector<int> dynamicShapeDims;
    std::vector<int> dynamicStrideDims;
    bool use32BitStride = false;
    std::vector<StaticDimGuard> staticGuards;  // only static dims checked
  };
  std::vector<TensorSlot> tensorSlots_;

  // Scalar slots — storage is always 8 bytes, JIT reads byteWidth bytes
  // from the address.  kind determines how to extract value from Python.
  struct ScalarSlot {
    int argIdx;
    int byteWidth;
    int kind;          // SK_INT, SK_FLOAT, or SK_STREAM
    int64_t storage = 0;
  };
  std::vector<ScalarSlot> scalarSlots_;

  bool hasUserStream_ = false;
  uintptr_t streamValue_ = 0;  // auto-stream

  std::vector<void *> packedArgs_;

  struct ConstexprGuard {
    int argIdx;
    int64_t expectedValue;
  };
  std::vector<ConstexprGuard> constexprGuards_;

  int nPositional_;
  std::vector<nb::str> trailingNames_;
  std::vector<nb::object> trailingDefaults_;

  enum SlotKind { TENSOR, SCALAR };
  struct OrderedSlot {
    SlotKind kind;
    int idx;
  };
  std::vector<OrderedSlot> slotOrder_;
};

} // namespace mlir::fly::utils
