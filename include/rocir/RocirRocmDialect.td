//===- RocirRocmDialect.td - Rocir ROCm Dialect Definition -*- tablegen -*-===//
//
// Rocir AMD GPU Hardware-Aware Dialect for GFX942
// Extends rocdsl with AMD ROCm-specific types and operations
//
//===----------------------------------------------------------------------===//

#ifndef ROCIR_ROCM_DIALECT
#define ROCIR_ROCM_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Dialect Definition
//===----------------------------------------------------------------------===//

def RocirRocm_Dialect : Dialect {
  let name = "rocir_rocm";
  let cppNamespace = "::mlir::rocir::rocm";
  let summary = "AMD GPU hardware-aware Rocir IR for GFX942";
  let description = [{
    Rocir ROCm IR extends rocdsl with AMD GPU-specific types and operations.
    It targets AMD GFX942 (MI300 series) matrix cores with layout-aware interfaces.
    
    **Design Philosophy**: 
    - Accepts rocdsl::LayoutType/TileType as partition descriptors
    - Encodes AMD MFMA instruction metadata (shape, element types, layout)
    - Generates architecture-specific code for GFX942
    
    **Lowering Chain**:
    rocir_rocm_ir → rocdl dialect → LLVM IR → AMDGCN
    
    **Target Architecture**:
    - GFX942 (MI300A/MI300X)
    - Matrix Core Operations (MFMA)
    - LDS (Local Data Share) optimization
    - Wavefront size: 64
  }];
  
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

class RocirRocm_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<RocirRocm_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class RocirRocm_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<RocirRocm_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class RocirRocm_Op<string mnemonic, list<Trait> traits = []>
    : Op<RocirRocm_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Enumerations
//===----------------------------------------------------------------------===//

def MfmaOperandEnum : I32EnumAttr<"MfmaOperand",
    "MFMA matrix operand type",
    [
      I32EnumAttrCase<"A", 0, "a">,
      I32EnumAttrCase<"B", 1, "b">,
      I32EnumAttrCase<"C", 2, "c">
    ]> {
  let cppNamespace = "::mlir::rocir::rocm";
}

def ReductionKindEnum : I32EnumAttr<"ReductionKind",
    "Reduction operation type",
    [
      I32EnumAttrCase<"None", 0, "none">,
      I32EnumAttrCase<"Sum", 1, "sum">,
      I32EnumAttrCase<"Max", 2, "max">,
      I32EnumAttrCase<"Min", 3, "min">
    ]> {
  let cppNamespace = "::mlir::rocir::rocm";
}

def GfxArchEnum : I32EnumAttr<"GfxArch",
    "AMD GPU architecture version",
    [
      I32EnumAttrCase<"GFX908", 0, "gfx908">,
      I32EnumAttrCase<"GFX90a", 1, "gfx90a">,
      I32EnumAttrCase<"GFX940", 2, "gfx940">,
      I32EnumAttrCase<"GFX941", 3, "gfx941">,
      I32EnumAttrCase<"GFX942", 4, "gfx942">,
      I32EnumAttrCase<"GFX950", 5, "gfx950">
    ]> {
  let cppNamespace = "::mlir::rocir::rocm";
}

//===----------------------------------------------------------------------===//
// MFMA Atom Types
//===----------------------------------------------------------------------===//

def RocirRocm_MfmaAtomType : RocirRocm_Type<"MfmaAtom", "mfma_atom"> {
  let summary = "MFMA instruction atom descriptor for GFX942";
  let description = [{
    Encodes metadata for a single MFMA (Matrix Fused Multiply-Add) instruction:
    - Shape: [M, N, K] dimensions
    - Element types: A, B, C/D operand types
    - Layout: Thread-to-data mapping for each operand
    - Architecture: Target GFX architecture (e.g., GFX942)
    
    GFX942 MFMA instruction support:
    - mfma_f32_32x32x8_f16
    - mfma_f32_16x16x16_f16
    - mfma_f64_16x16x4_f64
    - mfma_i32_32x32x16_i8
    - mfma_f32_32x32x16_bf16
  }];
  let parameters = (ins 
    ArrayRefParameter<"int64_t">:$shape,        // [M, N, K]
    "Type":$aType,                               // A operand element type
    "Type":$bType,                               // B operand element type
    "Type":$cType,                               // C/D operand element type
    "GfxArch":$arch                              // Target architecture
  );
  let assemblyFormat = "`<` $shape `,` $aType `,` $bType `,` $cType `,` $arch `>`";
}

//===----------------------------------------------------------------------===//
// Tiled MFMA Types
//===----------------------------------------------------------------------===//

def RocirRocm_TiledMfmaType : RocirRocm_Type<"TiledMfma", "tiled_mfma"> {
  let summary = "Tiled MFMA operation descriptor";
  let description = [{
    Describes a tiled MFMA operation composition:
    - Base MFMA atom
    - Tiling pattern across wavefronts
    - Layout per operand
    
    Supports hierarchical compositions for larger tile sizes.
  }];
  let parameters = (ins
    "MfmaAtomType":$atom,
    "Type":$tileLayout                           // Layout describing tiling pattern
  );
  let assemblyFormat = "`<` $atom `,` $tileLayout `>`";
}

//===----------------------------------------------------------------------===//
// LDS (Local Data Share) Types
//===----------------------------------------------------------------------===//

def RocirRocm_LdsBufferType : RocirRocm_Type<"LdsBuffer", "lds_buffer"> {
  let summary = "LDS buffer descriptor with layout";
  let description = [{
    Represents a buffer in LDS (Local Data Share) memory with:
    - Element type
    - Layout specification
    - Bank conflict avoidance metadata
    
    LDS is the AMD equivalent of NVIDIA shared memory.
    GFX942 has 64KB LDS per CU.
  }];
  let parameters = (ins
    "Type":$elementType,
    "Type":$layout,                              // Layout for LDS access pattern
    "int64_t":$sizeInBytes
  );
  let assemblyFormat = "`<` $elementType `,` $layout `,` $sizeInBytes `>`";
}

//===----------------------------------------------------------------------===//
// Copy Atom Types
//===----------------------------------------------------------------------===//

def RocirRocm_CopyAtomType : RocirRocm_Type<"CopyAtom", "copy_atom"> {
  let summary = "Copy instruction atom for GFX942";
  let description = [{
    Encodes a data movement operation:
    - Global to LDS
    - LDS to Register
    - Register to Global
    - Register to LDS
    
    Supports buffer_load/buffer_store and flat_load/flat_store instructions.
    
    The copy atom contains:
    - Element type and vector size
    - Thread ID layout (ThrID)
    - Source/Destination TV layouts (layout_src_tv, layout_dst_tv)
  }];
  let parameters = (ins
    "Type":$elementType,
    "int64_t":$vectorSize,                       // Number of elements per instruction
    "bool":$isCoalesced,                         // Whether accesses are coalesced
    "Type":$thrId,                               // Thread ID layout
    "Type":$layoutSrcTV,                         // Source thread-value layout
    "Type":$layoutDstTV                          // Destination thread-value layout
  );
  let assemblyFormat = "`<` $elementType `,` $vectorSize `,` $isCoalesced `,` $thrId `,` $layoutSrcTV `,` $layoutDstTV `>`";
}

def RocirRocm_TiledCopyType : RocirRocm_Type<"TiledCopy", "tiled_copy"> {
  let summary = "Tiled copy operation descriptor";
  let description = [{
    Describes a tiled copy operation that distributes data movement across threads.
    
    Composition of:
    - Base copy atom (defines single-thread copy behavior)
    - Tiler MN (tile size per thread block)
    - TV layout tiled (thread-value layout for the tile)
    - Layouts for source and destination
    
    This type encodes how to partition and copy data across a thread block.
  }];
  let parameters = (ins
    "CopyAtomType":$copyAtom,                    // Base copy atom
    "Type":$tilerMN,                             // Tiler shape (tile size)
    "Type":$layoutTVTiled,                       // Tiled TV layout
    "Type":$layoutSrcTVTiled,                    // Source TV layout tiled
    "Type":$layoutDstTVTiled                     // Destination TV layout tiled
  );
  let assemblyFormat = "`<` $copyAtom `,` $tilerMN `,` $layoutTVTiled `,` $layoutSrcTVTiled `,` $layoutDstTVTiled `>`";
}

//===----------------------------------------------------------------------===//
// Tensor Types
//===----------------------------------------------------------------------===//

def RocirRocm_TensorType : RocirRocm_Type<"Tensor", "tensor"> {
  let summary = "Tensor type with iterator and layout";
  let description = [{
    Represents a tensor as (iterator, layout) pair.
    
    A tensor evaluates to: T(coord) = *(iterator + layout(coord))
    
    Supports different memory spaces:
    - Global memory (gmem)
    - LDS (smem equivalent)
    - Register file (rmem)
    
    The layout encodes the coordinate-to-offset mapping.
  }];
  let parameters = (ins
    "Type":$elementType,                         // Element type (f16, f32, etc.)
    "Type":$layout,                              // Layout type
    "Type":$memorySpace                          // Memory space (gmem, smem, rmem)
  );
  let assemblyFormat = "`<` $elementType `,` $layout `,` $memorySpace `>`";
}

def RocirRocm_FragmentType : RocirRocm_Type<"Fragment", "fragment"> {
  let summary = "Register memory fragment type";
  let description = [{
    Represents a fragment of data in register memory.
    
    Fragments are allocated per-thread and used for computation.
    Size is determined by the layout.
  }];
  let parameters = (ins
    "Type":$elementType,                         // Element type
    "Type":$layout                               // Layout defining fragment shape
  );
  let assemblyFormat = "`<` $elementType `,` $layout `>`";
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

def RocirRocm_MfmaShapeAttr : RocirRocm_Attr<"MfmaShape", "mfma_shape"> {
  let summary = "MFMA instruction shape attribute";
  let description = [{
    Encodes the [M, N, K] shape of an MFMA instruction.
  }];
  let parameters = (ins
    "int64_t":$m,
    "int64_t":$n,
    "int64_t":$k
  );
  let assemblyFormat = "`<` $m `,` $n `,` $k `>`";
}

def RocirRocm_WavefrontSizeAttr : RocirRocm_Attr<"WavefrontSize", "wavefront_size"> {
  let summary = "Wavefront size attribute";
  let description = [{
    Specifies wavefront size (32 or 64).
    GFX942 uses wavefront size 64.
  }];
  let parameters = (ins "int64_t":$size);
  let assemblyFormat = "`<` $size `>`";
}

def RocirRocm_LdsConfigAttr : RocirRocm_Attr<"LdsConfig", "lds_config"> {
  let summary = "LDS configuration attribute";
  let description = [{
    Configuration for LDS usage:
    - Total size allocated
    - Bank conflict strategy
    - Padding configuration
  }];
  let parameters = (ins
    "int64_t":$totalSize,
    "int64_t":$padding,
    "bool":$avoidBankConflicts
  );
  let assemblyFormat = "`<` $totalSize `,` $padding `,` $avoidBankConflicts `>`";
}

//===----------------------------------------------------------------------===//
// Operation Base
//===----------------------------------------------------------------------===//

// Base class for all Rocir ROCm operations
class RocirRocm_OpBase<string mnemonic, list<Trait> traits = []>
    : RocirRocm_Op<mnemonic, traits>;

#endif // ROCIR_ROCM_DIALECT
