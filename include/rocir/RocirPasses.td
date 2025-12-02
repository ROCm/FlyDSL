//===- RocirPasses.td - CuTe IR Lowering Passes ------------*- tablegen -*-===//
//
// Lowering passes for CuTe IR dialects
// Transformation chain: rocdsl → gpu/scf → nvgpu → nvvm → LLVM IR
//
//===----------------------------------------------------------------------===//

#ifndef ROCIR_PASSES
#define ROCIR_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// rocdsl → Standard Dialects (scf, arith, memref)
//===----------------------------------------------------------------------===//

def RocirToStandardPass : Pass<"rocir-to-standard", "::mlir::func::FuncOp"> {
  let summary = "Lower CuTe IR to standard MLIR dialects";
  let description = [{
    Converts rocdsl operations to combinations of:
    - scf.for for iteration over layouts
    - arith for coordinate/index arithmetic
    - memref for tensor data access
    
    Lowering Strategy:
    - cute.make_shape/stride/layout → Compile-time constants
    - cute.crd2idx → arith.muli + arith.addi chain
    - cute.local_partition → scf.for with stride calculations
    - cute.copy → memref.load + memref.store loops
    
    Example:
    ```mlir
    // Before:
    %layout = cute.make_layout %shape, %stride
    %idx = cute.crd2idx %coord, %layout
    
    // After:
    %idx = arith.muli %coord_0, %stride_0
    %idx1 = arith.addi %idx, %coord_1_times_stride_1
    ...
    ```
  }];
  
  let constructor = "mlir::rocir::createRocirToStandardPass()";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::func::FuncDialect"
  ];
}


def RocirCoordLoweringPass : Pass<"rocir-coord-lowering", "::mlir::ModuleOp"> {
  let summary = "Lower Rocir coordinate operations to arithmetic";
  let description = [{
    Converts coordinate operations to arithmetic operations:
    
    - rocir.crd2idx → arith.muli + arith.addi chain
    - rocir.idx2crd → arith.divui + arith.remui
    - rocir.get → Direct value forwarding
    - rocir.rank → Constant folding
    
    Example:
    ```mlir
    // Before:
    %coord = rocir.make_coord %i, %j
    %idx = rocir.crd2idx %coord, %layout
    
    // After:
    %t0 = arith.muli %i, %stride_0
    %t1 = arith.muli %j, %stride_1
    %idx = arith.addi %t0, %t1
    ```
  }];
  
  let constructor = "mlir::rocir::createRocirCoordLoweringPass()";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect"
  ];
}

def CuteLayoutCanonicalizePass : Pass<"cute-layout-canonicalize", "::mlir::func::FuncOp"> {
  let summary = "Canonicalize and simplify Layout operations";
  let description = [{
    Applies algebraic simplifications to Layout operations:
    - Fuse adjacent composition operations
    - Eliminate identity layouts
    - Coalesce compatible dimensions
    - Constant-fold static shapes/strides
    
    Example:
    ```mlir
    // Before:
    %layout2 = cute.composition %layout1, %identity_layout
    
    // After:
    // (replaced with %layout1)
    ```
  }];
  
  let constructor = "mlir::rocir::createCuteLayoutCanonicalizePass()";
}

def CuteTensorPartitionPass : Pass<"cute-tensor-partition", "::mlir::func::FuncOp"> {
  let summary = "Materialize tensor partitioning into explicit indexing";
  let description = [{
    Converts high-level partition operations into explicit loops:
    - cute.local_partition → Thread-local index calculations
    - cute.local_tile → Tile extraction with boundary checks
    - cute.slice → Dimension reduction
    
    Generates optimized code for common patterns like:
    - Thread-block partitioning (blockIdx, threadIdx)
    - Warp-level tiling
    - Register blocking
  }];
  
  let constructor = "mlir::rocir::createCuteTensorPartitionPass()";
  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::gpu::GPUDialect"
  ];
}

//===----------------------------------------------------------------------===//
// cute_nvgpu_ir → nvgpu dialect
//===----------------------------------------------------------------------===//

def RocirNvgpuToNvgpuPass : Pass<"cute-nvgpu-to-nvgpu", "::mlir::gpu::GPUModuleOp"> {
  let summary = "Lower cute_nvgpu_ir to MLIR nvgpu dialect";
  let description = [{
    Converts CuTe GPU-aware operations to standard MLIR nvgpu operations.
    
    Key Transformations:
    
    **MMA Operations**:
    - cute_nvgpu.warpgroup_mma → nvgpu.warpgroup.mma.sync
    - cute_nvgpu.warp_mma_f16bf16 → nvgpu.mma.sync
    - Extracts shape/type metadata from MmaAtom
    - Generates proper fragment layouts
    
    **Copy Operations**:
    - cute_nvgpu.ldmatrix → nvgpu.ldmatrix
    - cute_nvgpu.cpasync_g2s → nvgpu.device_async_copy
    - Handles synchronization (cpasync.commit/wait)
    
    **TMA Operations (SM90+)**:
    - cute_nvgpu.tma_load_execute → nvgpu.tma.async.load
    - cute_nvgpu.tma_store_execute → nvgpu.tma.async.store
    - Materializes TMA descriptor creation
    - Integrates mbarrier synchronization
    
    Example:
    ```mlir
    // Before (cute_nvgpu_ir):
    %atom = cute_nvgpu.make_mma_atom !f16, !f16, !f32, [64,128,16], "SM90"
    %result = cute_nvgpu.warpgroup_mma %fragA, %fragB, %fragC, %atom
    
    // After (nvgpu):
    %result = nvgpu.warpgroup.mma.sync 
      %fragA, %fragB, %fragC 
      {mmaShape = [64, 128, 16]} : 
      (vector<..xf16>, vector<..xf16>, vector<..xf32>) -> vector<..xf32>
    ```
  }];
  
  let constructor = "mlir::rocir::createRocirNvgpuToNvgpuPass()";
  let dependentDialects = [
    "::mlir::nvgpu::NVGPUDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::vector::VectorDialect"
  ];
  
  let options = [
    Option<"targetArch", "target-arch", "std::string",
           /*default=*/"\"sm_90\"",
           "Target GPU architecture (sm_80, sm_90, sm_100)">,
    Option<"enableTMA", "enable-tma", "bool",
           /*default=*/"true",
           "Enable TMA operations (SM90+)">
  ];
}

def RocirNvgpuMmaLoweringPass : Pass<"cute-nvgpu-mma-lowering", "::mlir::gpu::GPUModuleOp"> {
  let summary = "Lower TiledMma to architecture-specific MMA instructions";
  let description = [{
    Decomposes TiledMma into individual MMA atoms with proper thread mapping.
    
    For Hopper (SM90) Warpgroup MMA:
    1. Extract warpgroup layout (4 warps)
    2. Partition fragments per warp
    3. Generate warpgroup.mma.sync calls
    4. Handle accumulator distribution
    
    For Ampere (SM80) Warp MMA:
    1. Extract warp layout (32 threads)
    2. Generate mma.sync calls per warp
    3. Handle different shapes: m16n8k16, m16n8k8
  }];
  
  let constructor = "mlir::rocir::createRocirNvgpuMmaLoweringPass()";
}

def RocirNvgpuCopyLoweringPass : Pass<"cute-nvgpu-copy-lowering", "::mlir::gpu::GPUModuleOp"> {
  let summary = "Lower TiledCopy to architecture-specific copy instructions";
  let description = [{
    Selects optimal copy instruction based on:
    - Source/destination address spaces
    - Data alignment and size
    - Target GPU architecture
    
    Copy Instruction Selection:
    - Global→Shared (aligned, SM80+): cp.async
    - Global→Shared (SM90+, large tile): TMA
    - Shared→Register (matrix layout): ldmatrix
    - Default: vectorized load/store
    
    Generates proper synchronization:
    - cp.async: commit_group + wait_group
    - TMA: mbarrier arrive + wait
    - ldmatrix: __syncthreads() if needed
  }];
  
  let constructor = "mlir::rocir::createRocirNvgpuCopyLoweringPass()";
}

def RocirNvgpuTMAMaterializationPass : Pass<"cute-nvgpu-tma-materialize", "::mlir::ModuleOp"> {
  let summary = "Materialize TMA descriptor creation and initialization";
  let description = [{
    Handles TMA descriptor lifecycle:
    
    **Host-Side (Pre-kernel)**:
    1. Allocate descriptor memory (cuTensorMapEncodeTiled)
    2. Initialize with tensor metadata
    3. Pass descriptor pointer to kernel
    
    **Device-Side (Kernel)**:
    1. Load descriptor from kernel parameters
    2. Set descriptor in atom via atom_set_value
    3. Create exec TMA with mbarrier
    4. Execute TMA operations
    
    Generated Code Pattern:
    ```cpp
    // Host:
    CUtensorMap desc;
    cuTensorMapEncodeTiled(&desc, ...);
    kernel<<<...>>>(A_ptr, B_ptr, C_ptr, &desc);
    
    // Device:
    __shared__ CUtensorMap* tma_desc = kernel_param_tma_desc;
    cute_nvgpu.atom_set_value %tma_load, "tma_descriptor", %tma_desc
    ```
  }];
  
  let constructor = "mlir::rocir::createRocirNvgpuTMAMaterializationPass()";
  let dependentDialects = [
    "::mlir::LLVM::LLVMDialect",
    "::mlir::nvgpu::NVGPUDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Optimization Passes
//===----------------------------------------------------------------------===//

def CuteLayoutFusionPass : Pass<"cute-layout-fusion", "::mlir::func::FuncOp"> {
  let summary = "Fuse adjacent Layout transformations";
  let description = [{
    Combines multiple Layout operations to reduce overhead:
    - Fuse composition chains: compose(compose(A, B), C) → compose(A, compose(B, C))
    - Merge partition + slice operations
    - Eliminate redundant coalesce operations
    
    Benefits:
    - Reduced runtime layout calculations
    - Better constant propagation
    - Simplified code generation
  }];
  
  let constructor = "mlir::rocir::createCuteLayoutFusionPass()";
}

def CuteVectorizationPass : Pass<"cute-vectorization", "::mlir::func::FuncOp"> {
  let summary = "Vectorize tensor copy operations based on Layout";
  let description = [{
    Analyzes Layout to determine optimal vectorization width:
    - Detects contiguous access patterns from stride analysis
    - Generates vector loads/stores (vector<4xf16>, etc.)
    - Respects alignment constraints
    
    Example:
    ```mlir
    // Before:
    scf.for %i (loads element by element)
    
    // After:
    scf.for %i (loads vector<8xf16> per iteration)
    ```
  }];
  
  let constructor = "mlir::rocir::createCuteVectorizationPass()";
  let dependentDialects = ["::mlir::vector::VectorDialect"];
}

def CuteMemoryCoalescingPass : Pass<"cute-memory-coalescing", "::mlir::gpu::GPUModuleOp"> {
  let summary = "Optimize memory access patterns for coalescing";
  let description = [{
    Analyzes thread access patterns via Layout and reorganizes:
    - Detects non-coalesced global memory accesses
    - Suggests layout transformations (blocked_product, raked_product)
    - Applies automatic padding for bank conflict avoidance
    
    Works with cute.blocked_product to ensure:
    - Adjacent threads access adjacent memory
    - Full cache line utilization (128 bytes)
  }];
  
  let constructor = "mlir::rocir::createCuteMemoryCoalescingPass()";
}

def CuteSMEMSwizzlingPass : Pass<"cute-smem-swizzling", "::mlir::gpu::GPUModuleOp"> {
  let summary = "Apply shared memory swizzling to avoid bank conflicts";
  let description = [{
    Implements swizzling patterns for shared memory layouts:
    - 32B, 64B, 128B swizzle modes (Hopper)
    - XOR-based permutation patterns
    - Compatible with TMA requirements
    
    Detects conflict patterns:
    - Column-major access in row-major layout
    - Power-of-2 strides causing same-bank access
    
    Generates swizzled layouts using cute.composition with swizzle strides.
  }];
  
  let constructor = "mlir::rocir::createCuteSMEMSwizzlingPass()";
}

//===----------------------------------------------------------------------===//
// Pipeline Passes
//===----------------------------------------------------------------------===//

def CuteAsyncPipelinePass : Pass<"cute-async-pipeline", "::mlir::gpu::GPUModuleOp"> {
  let summary = "Insert async copy pipeline for overlapping compute and data transfer";
  let description = [{
    Implements multi-stage async copy pipeline:
    
    **Ampere (cp.async, up to 8 stages)**:
    ```mlir
    // Prologue: Fill pipeline
    scf.for %stage = 0 to %depth
      cpasync_g2s (stage %stage)
      cpasync_commit_group
    
    // Steady state: Compute + Copy
    scf.for %k
      cpasync_wait_group<%depth-1>
      mma (using buffer %k % %depth)
      cpasync_g2s (stage %k+%depth)
      cpasync_commit_group
    
    // Epilogue: Drain pipeline
    scf.for %stage = 0 to %depth
      cpasync_wait_group<%depth-1-%stage>
      mma (remaining stages)
    ```
    
    **Hopper (TMA + mbarrier, producer-consumer)**:
    - TMA producer thread issues loads
    - mbarrier phases coordinate pipeline stages
    - Warpgroup MMA consumes data asynchronously
  }];
  
  let constructor = "mlir::rocir::createCuteAsyncPipelinePass()";
  let dependentDialects = [
    "::mlir::nvgpu::NVGPUDialect",
    "::mlir::scf::SCFDialect"
  ];
  
  let options = [
    Option<"pipelineDepth", "pipeline-depth", "unsigned",
           /*default=*/"2",
           "Number of pipeline stages (2-8)">,
    Option<"useWarpSpecialization", "warp-specialization", "bool",
           /*default=*/"false",
           "Use warp specialization (SM90+ producer-consumer)">
  ];
}

def CuteWarpSpecializationPass : Pass<"cute-warp-specialization", "::mlir::gpu::GPUModuleOp"> {
  let summary = "Apply warp specialization for Hopper producer-consumer pattern";
  let description = [{
    Splits kernel into specialized warp roles (SM90+):
    
    **Producer Warps**:
    - Issue TMA operations
    - Manage mbarrier phases
    - Minimal register usage
    
    **Consumer Warps**:
    - Execute warpgroup MMA
    - Wait on mbarrier
    - Full register pressure
    
    Code Pattern:
    ```mlir
    %warp_id = gpu.thread_id y
    scf.if (%warp_id < %num_producer_warps) {
      // Producer logic
      cute_nvgpu.tma_load_execute ...
      cute_nvgpu.mbarrier_arrive ...
    } else {
      // Consumer logic
      cute_nvgpu.mbarrier_wait ...
      cute_nvgpu.warpgroup_mma ...
    }
    ```
  }];
  
  let constructor = "mlir::rocir::createCuteWarpSpecializationPass()";
  
  let options = [
    Option<"numProducerWarps", "num-producer-warps", "unsigned",
           /*default=*/"1",
           "Number of producer warps (typically 1 for TMA)">
  ];
}

//===----------------------------------------------------------------------===//
// Analysis Passes
//===----------------------------------------------------------------------===//

def CuteLayoutAnalysisPass : Pass<"cute-layout-analysis", "::mlir::func::FuncOp"> {
  let summary = "Analyze Layout properties for optimization decisions";
  let description = [{
    Computes Layout metadata:
    - Coalescing score (adjacent threads → adjacent memory)
    - Bank conflict probability
    - Vectorization width
    - Alignment guarantees
    
    Used by downstream optimization passes.
  }];
  
  let constructor = "mlir::rocir::createCuteLayoutAnalysisPass()";
  let options = [
    Option<"printAnalysis", "print-analysis", "bool",
           /*default=*/"false",
           "Print layout analysis results">
  ];
}

def CuteAtomValidationPass : Pass<"cute-atom-validation", "::mlir::ModuleOp"> {
  let summary = "Validate MmaAtom/CopyAtom configurations";
  let description = [{
    Checks:
    - Shape compatibility (M, N, K dimensions)
    - Element type support (f16/bf16/tf32/fp8/int8)
    - Architecture availability (SM version)
    - TMA descriptor validity (alignment, swizzle modes)
    
    Emits errors for invalid configurations:
    - Unsupported shapes for target architecture
    - Mismatched fragment types
    - Invalid TMA swizzle patterns
  }];
  
  let constructor = "mlir::rocir::createCuteAtomValidationPass()";
}

//===----------------------------------------------------------------------===//
// Pass Pipelines
//===----------------------------------------------------------------------===//

def CuteToRocmPass : Pass<"cute-to-rocm", "::mlir::ModuleOp"> {
  let summary = "Lower CuTe IR to ROCm-specific dialects";
  let description = [{
    Converts rocdsl operations to cute_rocm_ir operations for AMD GFX942.
  }];
  let constructor = "mlir::rocir::createCuteToRocmPass()";
  let dependentDialects = [
    "::mlir::rocir::rocm::RocirRocmDialect"
  ];
}

#endif // ROCIR_PASSES
