//===- RocirRocmOps.td - Rocir ROCm Operations Definition --*- tablegen -*-===//
//
// Rocir AMD GPU operations for GFX942
//
//===----------------------------------------------------------------------===//

#ifndef ROCIR_ROCM_OPS
#define ROCIR_ROCM_OPS

include "rocir/RocirRocmDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// MFMA Operations
//===----------------------------------------------------------------------===//

def RocirRocm_MfmaOp : RocirRocm_OpBase<"mfma", [Pure]> {
  let summary = "MFMA (Matrix Fused Multiply-Add) operation for GFX942";
  let description = [{
    Performs matrix multiplication using AMD MFMA instructions on GFX942.
    
    Operation: D = A * B + C
    
    Supports various shapes and data types:
    - mfma_f32_32x32x8_f16: 32x32x8 with FP16 inputs, FP32 output
    - mfma_f32_16x16x16_f16: 16x16x16 with FP16 inputs, FP32 output
    - mfma_f64_16x16x4_f64: 16x16x4 with FP64 inputs/outputs
    - mfma_f32_32x32x16_bf16: 32x32x16 with BF16 inputs, FP32 output
    
    Example:
    ```mlir
    %d = rocir.rocm.mfma %a, %b, %c {
      shape = [32, 32, 8],
      arch = "gfx942"
    } : (!rocir.tensor<f16, layout_a>, !rocir.tensor<f16, layout_b>, 
         !rocir.tensor<f32, layout_c>) -> !rocir.tensor<f32, layout_d>
    ```
  }];
  
  let arguments = (ins
    AnyType:$a,                          // A operand (LHS)
    AnyType:$b,                          // B operand (RHS)
    AnyType:$c,                          // C operand (accumulator)
    ArrayAttr:$shape,                    // [M, N, K] shape
    StrAttr:$arch                        // Target architecture (gfx942, etc.)
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $a `,` $b `,` $c attr-dict `:` 
    `(` type($a) `,` type($b) `,` type($c) `)` `->` type($result)
  }];
}

def RocirRocm_TiledMfmaOp : RocirRocm_OpBase<"tiled_mfma", [Pure]> {
  let summary = "Tiled MFMA operation across multiple wavefronts";
  let description = [{
    Performs a tiled matrix multiplication using MFMA instructions.
    Distributes computation across multiple wavefronts based on tiling pattern.
    
    Example:
    ```mlir
    %d = rocir.rocm.tiled_mfma %a, %b, %c {
      atom = #rocir_rocm.mfma_atom<[32, 32, 8], f16, f16, f32, gfx942>,
      tile_shape = [128, 128, 32]
    } : (!rocir.tensor<...>, !rocir.tensor<...>, !rocir.tensor<...>) 
        -> !rocir.tensor<...>
    ```
  }];
  
  let arguments = (ins
    AnyType:$a,
    AnyType:$b,
    AnyType:$c,
    TypeAttr:$atom,                      // MFMA atom descriptor
    ArrayAttr:$tile_shape                // Tiling pattern
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $a `,` $b `,` $c attr-dict `:` 
    `(` type($a) `,` type($b) `,` type($c) `)` `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// LDS Operations
//===----------------------------------------------------------------------===//

def RocirRocm_LdsAllocOp : RocirRocm_OpBase<"lds_alloc"> {
  let summary = "Allocate LDS (Local Data Share) buffer";
  let description = [{
    Allocates a buffer in LDS memory with specified layout.
    LDS is AMD's equivalent to NVIDIA's shared memory.
    
    GFX942 provides 64KB LDS per compute unit.
    
    Example:
    ```mlir
    %lds = rocir.rocm.lds_alloc {
      element_type = f16,
      layout = !rocir.layout<...>,
      size = 32768
    } : !rocir.rocm.lds_buffer<f16, ..., 32768>
    ```
  }];
  
  let arguments = (ins
    TypeAttr:$element_type,
    TypeAttr:$layout,
    I64Attr:$size
  );
  
  let results = (outs RocirRocm_LdsBufferType:$result);
  
  let assemblyFormat = "attr-dict `:` type($result)";
}

def RocirRocm_LdsLoadOp : RocirRocm_OpBase<"lds_load", [Pure]> {
  let summary = "Load data from LDS to registers";
  let description = [{
    Loads data from LDS buffer to register file.
    
    Example:
    ```mlir
    %data = rocir.rocm.lds_load %lds_buffer, %coord {
      vector_size = 4
    } : (!rocir.rocm.lds_buffer<f16, ...>, !rocir.coord<2>) -> !rocir.tensor<f16, ...>
    ```
  }];
  
  let arguments = (ins
    RocirRocm_LdsBufferType:$buffer,
    AnyType:$coord,
    I64Attr:$vector_size
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $buffer `,` $coord attr-dict `:` 
    `(` type($buffer) `,` type($coord) `)` `->` type($result)
  }];
}

def RocirRocm_LdsStoreOp : RocirRocm_OpBase<"lds_store"> {
  let summary = "Store data from registers to LDS";
  let description = [{
    Stores data from register file to LDS buffer.
    
    Example:
    ```mlir
    rocir.rocm.lds_store %data, %lds_buffer, %coord {
      vector_size = 4
    } : (!rocir.tensor<f16, ...>, !rocir.rocm.lds_buffer<f16, ...>, !rocir.coord<2>)
    ```
  }];
  
  let arguments = (ins
    AnyType:$data,
    RocirRocm_LdsBufferType:$buffer,
    AnyType:$coord,
    I64Attr:$vector_size
  );
  
  let assemblyFormat = [{
    $data `,` $buffer `,` $coord attr-dict `:` 
    `(` type($data) `,` type($buffer) `,` type($coord) `)`
  }];
}

//===----------------------------------------------------------------------===//
// Atom Construction Operations
//===----------------------------------------------------------------------===//

def RocirRocm_MakeCopyAtomOp : RocirRocm_OpBase<"make_copy_atom", [Pure]> {
  let summary = "Create a copy atom for GFX942";
  let description = [{
    Creates a copy atom descriptor for data movement operations on AMD GPUs.
    
    A copy atom encodes:
    - Element type being copied
    - Vector size (elements per instruction)
    - Coalescing strategy for memory bandwidth
    
    Example:
    ```mlir
    %atom = rocir.rocm.make_copy_atom !f16, 8, true
      : !rocir_rocm.copy_atom<f16, 8, true>
    ```
  }];
  
  let arguments = (ins
    TypeAttr:$element_type,
    I64Attr:$vector_size,
    BoolAttr:$is_coalesced
  );
  
  let results = (outs RocirRocm_CopyAtomType:$result);
  
  let assemblyFormat = [{
    $element_type `,` $vector_size `,` $is_coalesced attr-dict
    `:` type($result)
  }];
}

def RocirRocm_MakeTiledCopyOp : RocirRocm_OpBase<"make_tiled_copy", [Pure]> {
  let summary = "Create a tiled copy from copy atom and TV layout";
  let description = [{
    Creates a tiled copy operation by composing a copy atom with a
    thread-value (TV) layout and tiler.
    
    This defines how data movement is distributed across threads in a
    thread block.
    
    Example:
    ```mlir
    %tiled_copy = rocir.rocm.make_tiled_copy %atom, %tv_layout, %tiler
      : (!rocir_rocm.copy_atom<...>, !rocir.layout<2>, !rocir.shape<2>)
        -> !rocir_rocm.tiled_copy<...>
    ```
  }];
  
  let arguments = (ins
    RocirRocm_CopyAtomType:$atom,
    AnyType:$tv_layout,
    AnyType:$tiler
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $atom `,` $tv_layout `,` $tiler attr-dict `:`
    `(` type($atom) `,` type($tv_layout) `,` type($tiler) `)` `->` type($result)
  }];
}

def RocirRocm_MakeTiledCopyTVOp : RocirRocm_OpBase<"make_tiled_copy_tv", [Pure]> {
  let summary = "Create a tiled copy from copy atom and separate thread/value layouts";
  let description = [{
    Creates a tiled copy from a copy atom and separate thread and value layouts.
    
    The TV layout is inferred from the thread layout (mapping threads to tile coords)
    and value layout (mapping values per thread to tile coords).
    
    Implementation:
    1. Compute tiler_mn = product_each(raked_product(thr_layout, val_layout).shape)
    2. Compute layout_tv from thr_layout and val_layout composition
    3. Create TiledCopyType with (atom, tiler_mn, layout_tv)
    
    Example:
    ```mlir
    %tiled_copy = rocir.rocm.make_tiled_copy_tv %atom, %thr_layout, %val_layout
      : (!rocir_rocm.copy_atom<...>, !rocir.layout<2>, !rocir.layout<2>)
        -> !rocir_rocm.tiled_copy<...>
    ```
  }];
  
  let arguments = (ins
    RocirRocm_CopyAtomType:$atom,
    AnyType:$thr_layout,
    AnyType:$val_layout
  );
  
  let results = (outs RocirRocm_TiledCopyType:$result);
  
  let assemblyFormat = [{
    $atom `,` $thr_layout `,` $val_layout attr-dict `:`
    `(` type($atom) `,` type($thr_layout) `,` type($val_layout) `)` `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Partition Operations
//===----------------------------------------------------------------------===//

def RocirRocm_PartitionSrcOp : RocirRocm_OpBase<"partition_src", [Pure]> {
  let summary = "Partition source tensor according to tiled copy layout";
  let description = [{
    Partitions a source tensor for a specific thread according to the
    tiled copy's TV layout.
    
    Given:
    - tiled_copy: TiledCopy descriptor
    - src_tensor: Source tensor to partition
    - thr_idx: Thread index
    
    Returns:
    - Partitioned tensor for this thread
    
    Example:
    ```mlir
    %thrA = rocir.rocm.partition_src %tiled_copy, %blkA, %tidx
      : (!rocir_rocm.tiled_copy<...>, !rocir_rocm.tensor<...>, index)
        -> !rocir_rocm.tensor<...>
    ```
  }];
  
  let arguments = (ins
    RocirRocm_TiledCopyType:$tiled_copy,
    AnyType:$src_tensor,
    Index:$thr_idx
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $tiled_copy `,` $src_tensor `,` $thr_idx attr-dict `:`
    `(` type($tiled_copy) `,` type($src_tensor) `,` type($thr_idx) `)` `->` type($result)
  }];
}

def RocirRocm_PartitionDstOp : RocirRocm_OpBase<"partition_dst", [Pure]> {
  let summary = "Partition destination tensor according to tiled copy layout";
  let description = [{
    Partitions a destination tensor for a specific thread according to the
    tiled copy's TV layout.
    
    Similar to partition_src but for destination tensors.
    
    Example:
    ```mlir
    %thrC = rocir.rocm.partition_dst %tiled_copy, %blkC, %tidx
      : (!rocir_rocm.tiled_copy<...>, !rocir_rocm.tensor<...>, index)
        -> !rocir_rocm.tensor<...>
    ```
  }];
  
  let arguments = (ins
    RocirRocm_TiledCopyType:$tiled_copy,
    AnyType:$dst_tensor,
    Index:$thr_idx
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $tiled_copy `,` $dst_tensor `,` $thr_idx attr-dict `:`
    `(` type($tiled_copy) `,` type($dst_tensor) `,` type($thr_idx) `)` `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Tensor Operations
//===----------------------------------------------------------------------===//

def RocirRocm_MakeTensorOp : RocirRocm_OpBase<"make_tensor", [Pure]> {
  let summary = "Create a tensor from pointer and layout";
  let description = [{
    Creates a tensor by combining an iterator (pointer) with a layout.
    
    Tensor semantics: T(coord) = *(ptr + layout(coord))
    
    Example:
    ```mlir
    %tensor = rocir.rocm.make_tensor %ptr, %layout
      : (!llvm.ptr, !rocir.layout<2>) -> !rocir_rocm.tensor<f16, ...>
    ```
  }];
  
  let arguments = (ins
    AnyType:$ptr,
    AnyType:$layout
  );
  
  let results = (outs RocirRocm_TensorType:$result);
  
  let assemblyFormat = [{
    $ptr `,` $layout attr-dict `:`
    `(` type($ptr) `,` type($layout) `)` `->` type($result)
  }];
}

def RocirRocm_MakeFragmentOp : RocirRocm_OpBase<"make_fragment", [Pure]> {
  let summary = "Create a fragment (register memory tensor)";
  let description = [{
    Allocates a fragment in register memory with given layout.
    
    Fragments are per-thread local storage used for computation.
    
    Example:
    ```mlir
    %frg = rocir.rocm.make_fragment %layout, %element_type
      : (!rocir.layout<2>, !f16) -> !rocir_rocm.fragment<f16, ...>
    ```
  }];
  
  let arguments = (ins
    AnyType:$layout,
    TypeAttr:$element_type
  );
  
  let results = (outs RocirRocm_FragmentType:$result);
  
  let assemblyFormat = [{
    $layout `,` $element_type attr-dict `:`
    `(` type($layout) `)` `->` type($result)
  }];
}

def RocirRocm_FragmentLoadOp : RocirRocm_OpBase<"fragment_load", [Pure]> {
  let summary = "Load data from fragment for computation";
  let description = [{
    Loads fragment data into computational value.
    The compiler optimizes away redundant loads.
    
    Example:
    ```mlir
    %val = rocir.rocm.fragment_load %frg
      : (!rocir_rocm.fragment<f16, ...>) -> vector<16xf16>
    ```
  }];
  
  let arguments = (ins RocirRocm_FragmentType:$fragment);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $fragment attr-dict `:` `(` type($fragment) `)` `->` type($result)
  }];
}

def RocirRocm_FragmentStoreOp : RocirRocm_OpBase<"fragment_store"> {
  let summary = "Store computed value into fragment";
  let description = [{
    Stores a computed value back into the fragment.
    
    Example:
    ```mlir
    rocir.rocm.fragment_store %value, %frg
      : (vector<16xf16>, !rocir_rocm.fragment<f16, ...>)
    ```
  }];
  
  let arguments = (ins
    AnyType:$value,
    RocirRocm_FragmentType:$fragment
  );
  
  let assemblyFormat = [{
    $value `,` $fragment attr-dict `:`
    `(` type($value) `,` type($fragment) `)`
  }];
}

//===----------------------------------------------------------------------===//
// Copy Operations
//===----------------------------------------------------------------------===//

def RocirRocm_CopyOp : RocirRocm_OpBase<"copy"> {
  let summary = "Copy data with layout transformation";
  let description = [{
    Copies data from source to destination with optional layout transformation.
    Optimized for GFX942 memory hierarchy.
    
    Supports:
    - Global to LDS (buffer_load)
    - LDS to Register (ds_read)
    - Register to Global (buffer_store)
    - Register to LDS (ds_write)
    
    Example:
    ```mlir
    rocir.rocm.copy %src, %dst {
      vector_size = 8,
      coalesced = true
    } : (!rocir.tensor<f16, layout_src>, !rocir.tensor<f16, layout_dst>)
    ```
  }];
  
  let arguments = (ins
    AnyType:$src,
    AnyType:$dst,
    I64Attr:$vector_size,
    BoolAttr:$coalesced
  );
  
  let assemblyFormat = [{
    $src `,` $dst attr-dict `:` `(` type($src) `,` type($dst) `)`
  }];
}

def RocirRocm_AsyncCopyOp : RocirRocm_OpBase<"async_copy"> {
  let summary = "Asynchronous copy operation";
  let description = [{
    Initiates an asynchronous copy from global memory to LDS.
    Uses buffer_load instructions with asynchronous semantics.
    
    Example:
    ```mlir
    %token = rocir.rocm.async_copy %src, %dst {
      vector_size = 16
    } : (!rocir.tensor<f16, ...>, !rocir.rocm.lds_buffer<f16, ...>) 
        -> !rocir.rocm.async_token
    ```
  }];
  
  let arguments = (ins
    AnyType:$src,
    AnyType:$dst,
    I64Attr:$vector_size
  );
  
  let results = (outs AnyType:$token);
  
  let assemblyFormat = [{
    $src `,` $dst attr-dict `:` 
    `(` type($src) `,` type($dst) `)` `->` type($token)
  }];
}

def RocirRocm_CopyWaitOp : RocirRocm_OpBase<"copy_wait"> {
  let summary = "Wait for async copy completion";
  let description = [{
    Waits for async copy operations to complete.
    Inserts s_waitcnt instruction.
    
    Example:
    ```mlir
    rocir.rocm.copy_wait %token
    ```
  }];
  
  let arguments = (ins AnyType:$token);
  
  let assemblyFormat = "$token attr-dict `:` type($token)";
}

//===----------------------------------------------------------------------===//
// Synchronization Operations
//===----------------------------------------------------------------------===//

def RocirRocm_BarrierOp : RocirRocm_OpBase<"barrier"> {
  let summary = "Workgroup barrier synchronization";
  let description = [{
    Synchronizes all work items in a workgroup.
    Implements s_barrier instruction.
    
    Example:
    ```mlir
    rocir.rocm.barrier
    ```
  }];
  
  let assemblyFormat = "attr-dict";
}

def RocirRocm_WavefrontBarrierOp : RocirRocm_OpBase<"wavefront_barrier"> {
  let summary = "Wavefront-level barrier";
  let description = [{
    Synchronizes all lanes within a wavefront.
    Uses s_waitcnt with appropriate flags.
    
    Example:
    ```mlir
    rocir.rocm.wavefront_barrier
    ```
  }];
  
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Utility Operations
//===----------------------------------------------------------------------===//

def RocirRocm_ThreadIdOp : RocirRocm_OpBase<"thread_id", [Pure]> {
  let summary = "Get thread ID within workgroup";
  let description = [{
    Returns the thread ID in the specified dimension (x, y, z).
    
    Example:
    ```mlir
    %tid_x = rocir.rocm.thread_id { dim = 0 } : i32
    %tid_y = rocir.rocm.thread_id { dim = 1 } : i32
    ```
  }];
  
  let arguments = (ins I32Attr:$dim);
  let results = (outs I32:$result);
  
  let assemblyFormat = "attr-dict `:` type($result)";
}

def RocirRocm_WavefrontIdOp : RocirRocm_OpBase<"wavefront_id", [Pure]> {
  let summary = "Get wavefront ID within workgroup";
  let description = [{
    Returns the wavefront ID.
    
    Example:
    ```mlir
    %wid = rocir.rocm.wavefront_id : i32
    ```
  }];
  
  let results = (outs I32:$result);
  
  let assemblyFormat = "attr-dict `:` type($result)";
}

def RocirRocm_LaneIdOp : RocirRocm_OpBase<"lane_id", [Pure]> {
  let summary = "Get lane ID within wavefront";
  let description = [{
    Returns the lane ID (0-63 for GFX942).
    
    Example:
    ```mlir
    %lane = rocir.rocm.lane_id : i32
    ```
  }];
  
  let results = (outs I32:$result);
  
  let assemblyFormat = "attr-dict `:` type($result)";
}

#endif // ROCIR_ROCM_OPS
