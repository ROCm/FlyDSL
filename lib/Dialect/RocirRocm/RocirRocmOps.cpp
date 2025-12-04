//===- RocirRocmOps.cpp - Rocir ROCm Operations Implementation -----------===//
//
// Implementation of Rocir ROCm dialect operations
//
//===----------------------------------------------------------------------===//

#include "rocir/RocirRocmOps.h"
#include "rocir/RocirRocmDialect.h"
#include "rocir/RocirOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::rocir::rocm;

//===----------------------------------------------------------------------===//
// MakeCopyAtomOp
//===----------------------------------------------------------------------===//

LogicalResult MakeCopyAtomOp::verify() {
  // Verify element type is valid (f16, f32, f64, bf16, i8, i32, etc.)
  auto elemType = getElementTypeAttr();
  if (!elemType) {
    return emitOpError("element_type attribute is required");
  }
  
  // Verify vector size is reasonable (typically powers of 2: 1, 2, 4, 8, 16)
  int64_t vecSize = getVectorSize();
  if (vecSize <= 0 || vecSize > 16) {
    return emitOpError("vector_size must be between 1 and 16, got ") << vecSize;
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// MakeTiledCopyTVOp
//===----------------------------------------------------------------------===//

LogicalResult MakeTiledCopyTVOp::verify() {
  // Verify that atom is a CopyAtomType
  auto atomType = getAtom().getType();
  if (!atomType.isa<CopyAtomType>()) {
    return emitOpError("atom operand must be of CopyAtomType");
  }
  
  // Verify that thr_layout and val_layout are LayoutType
  // Note: We accept AnyType in TD definition for flexibility, but verify here
  auto thrLayout = getThrLayout();
  auto valLayout = getValLayout();
  
  // TODO: Add type checking for layout types
  // if (!thrLayout.getType().isa<rocir::LayoutType>()) {
  //   return emitOpError("thr_layout must be a LayoutType");
  // }
  
  return success();
}

//===----------------------------------------------------------------------===//
// PartitionSrcOp / PartitionDstOp
//===----------------------------------------------------------------------===//

LogicalResult PartitionSrcOp::verify() {
  // Verify tiled_copy is TiledCopyType
  auto tiledCopyType = getTiledCopy().getType();
  if (!tiledCopyType.isa<TiledCopyType>()) {
    return emitOpError("tiled_copy operand must be of TiledCopyType");
  }
  
  return success();
}

LogicalResult PartitionDstOp::verify() {
  // Similar to PartitionSrcOp
  auto tiledCopyType = getTiledCopy().getType();
  if (!tiledCopyType.isa<TiledCopyType>()) {
    return emitOpError("tiled_copy operand must be of TiledCopyType");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// MakeTensorOp
//===----------------------------------------------------------------------===//

LogicalResult MakeTensorOp::verify() {
  // Verify result is TensorType
  auto resultType = getResult().getType();
  if (!resultType.isa<TensorType>()) {
    return emitOpError("result must be of TensorType");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// MakeFragmentOp
//===----------------------------------------------------------------------===//

LogicalResult MakeFragmentOp::verify() {
  // Verify result is FragmentType
  auto resultType = getResult().getType();
  if (!resultType.isa<FragmentType>()) {
    return emitOpError("result must be of FragmentType");
  }
  
  // Verify element_type attribute is present
  auto elemType = getElementTypeAttr();
  if (!elemType) {
    return emitOpError("element_type attribute is required");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// FragmentLoadOp
//===----------------------------------------------------------------------===//

LogicalResult FragmentLoadOp::verify() {
  // Verify operand is FragmentType
  auto fragType = getFragment().getType();
  if (!fragType.isa<FragmentType>()) {
    return emitOpError("fragment operand must be of FragmentType");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// FragmentStoreOp
//===----------------------------------------------------------------------===//

LogicalResult FragmentStoreOp::verify() {
  // Verify fragment operand is FragmentType
  auto fragType = getFragment().getType();
  if (!fragType.isa<FragmentType>()) {
    return emitOpError("fragment operand must be of FragmentType");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

LogicalResult CopyOp::verify() {
  // Verify source and destination types are compatible
  auto srcType = getSrc().getType();
  auto dstType = getDst().getType();
  
  // TODO: Add detailed type compatibility checking
  // For now, accept any tensor types
  
  // Verify vector size is reasonable
  int64_t vecSize = getVectorSize();
  if (vecSize <= 0 || vecSize > 16) {
    return emitOpError("vector_size must be between 1 and 16, got ") << vecSize;
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// MfmaOp
//===----------------------------------------------------------------------===//

LogicalResult MfmaOp::verify() {
  // Verify shape attribute exists and has correct size
  auto shapeAttr = getShape();
  if (!shapeAttr || shapeAttr.size() != 3) {
    return emitOpError("shape attribute must be array of 3 elements [M, N, K]");
  }
  
  // Verify shape values are valid for GFX942 MFMA
  // Common shapes: [32,32,8], [16,16,16], [32,32,16], etc.
  auto M = shapeAttr[0].cast<IntegerAttr>().getInt();
  auto N = shapeAttr[1].cast<IntegerAttr>().getInt();
  auto K = shapeAttr[2].cast<IntegerAttr>().getInt();
  
  // Basic sanity checks
  if (M <= 0 || N <= 0 || K <= 0) {
    return emitOpError("shape dimensions must be positive");
  }
  
  // TODO: Add more specific validation for supported MFMA shapes
  
  return success();
}

//===----------------------------------------------------------------------===//
// LDS Operations
//===----------------------------------------------------------------------===//

LogicalResult LdsAllocOp::verify() {
  // Verify size is reasonable (64KB max on GFX942)
  int64_t size = getSize();
  if (size <= 0 || size > 65536) {
    return emitOpError("LDS size must be between 1 and 65536 bytes, got ") << size;
  }
  
  // Verify result type is LdsBufferType
  auto resultType = getResult().getType();
  if (!resultType.isa<LdsBufferType>()) {
    return emitOpError("result must be of LdsBufferType");
  }
  
  return success();
}

LogicalResult LdsLoadOp::verify() {
  // Verify buffer is LdsBufferType
  auto bufferType = getBuffer().getType();
  if (!bufferType.isa<LdsBufferType>()) {
    return emitOpError("buffer operand must be of LdsBufferType");
  }
  
  // Verify vector size
  int64_t vecSize = getVectorSize();
  if (vecSize <= 0 || vecSize > 16) {
    return emitOpError("vector_size must be between 1 and 16, got ") << vecSize;
  }
  
  return success();
}

LogicalResult LdsStoreOp::verify() {
  // Verify buffer is LdsBufferType
  auto bufferType = getBuffer().getType();
  if (!bufferType.isa<LdsBufferType>()) {
    return emitOpError("buffer operand must be of LdsBufferType");
  }
  
  // Verify vector size
  int64_t vecSize = getVectorSize();
  if (vecSize <= 0 || vecSize > 16) {
    return emitOpError("vector_size must be between 1 and 16, got ") << vecSize;
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// ThreadIdOp / WavefrontIdOp / LaneIdOp
//===----------------------------------------------------------------------===//

LogicalResult ThreadIdOp::verify() {
  // Verify dimension is 0, 1, or 2 (x, y, z)
  int32_t dim = getDim();
  if (dim < 0 || dim > 2) {
    return emitOpError("dimension must be 0, 1, or 2, got ") << dim;
  }
  
  return success();
}

LogicalResult WavefrontIdOp::verify() {
  // No specific verification needed
  return success();
}

LogicalResult LaneIdOp::verify() {
  // No specific verification needed
  // Lane ID is 0-63 on GFX942, but this is checked at runtime
  return success();
}

//===----------------------------------------------------------------------===//
// BarrierOp
//===----------------------------------------------------------------------===//

LogicalResult BarrierOp::verify() {
  // No specific verification needed
  return success();
}

LogicalResult WavefrontBarrierOp::verify() {
  // No specific verification needed
  return success();
}

#define GET_OP_CLASSES
#include "rocir/RocirRocmOps.cpp.inc"

